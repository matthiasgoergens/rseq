//! The restartable-sequence IR.
//!
//! A program is a straight-line sequence of operations with the shape a valid
//! rseq critical section must have:
//!
//! - a side-effect-free prefix of loads and register arithmetic,
//! - at most one branch-to-exit,
//! - writes to *scratch* (unpublished) memory anywhere before the commit,
//! - exactly one committing store, which is the final operation.
//!
//! The same data structure is meant to feed three backends: the simulator and
//! checker (this milestone), and later an asm-template + `rseq_cs` descriptor
//! emitter.

use std::fmt;

/// A virtual register. Programs are built in SSA-ish style: the builder hands
/// out a fresh register for every defining operation.
pub type Reg = usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Operand {
    Reg(Reg),
    Imm(u64),
}

/// Shorthand for `Operand::Reg`.
#[must_use] 
pub fn reg(r: Reg) -> Operand {
    Operand::Reg(r)
}

/// Shorthand for `Operand::Imm`.
#[must_use] 
pub fn imm(v: u64) -> Operand {
    Operand::Imm(v)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

/// Unsigned comparison conditions for the exit branch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cond {
    Eq,
    Ne,
    Lt,
    Ge,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegionId(pub usize);

/// A word address: a region plus a word index into it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Addr {
    pub region: RegionId,
    pub index: Operand,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// Read the *current* CPU id. Re-executes on every restart, so it is
    /// always fresh — this is the correct way to index per-CPU data.
    CpuId { dst: Reg },
    /// Deliberately wrong: models a CPU id that was read once, before the
    /// sequence, and cached (the classic hoisting bug). The simulator returns
    /// the CPU the run *started* on, ignoring migrations. Exists so tests can
    /// demonstrate that the checker catches this bug class.
    CpuIdHoisted { dst: Reg },
    Const { dst: Reg, value: u64 },
    Load { dst: Reg, addr: Addr },
    Bin { dst: Reg, op: BinOp, lhs: Operand, rhs: Operand },
    /// Early exit without committing (e.g. "freelist is empty").
    ExitIf { cond: Cond, lhs: Operand, rhs: Operand },
    /// Write to unpublished scratch memory. Replay-safe by construction: no
    /// other actor may observe scratch, so re-executing after an abort is
    /// harmless (the tcmalloc scribble-then-bump trick).
    StoreScratch { addr: Addr, src: Operand },
    /// The single committing store. Must be the final operation.
    Commit { addr: Addr, src: Operand },
}

/// Declaration of a memory region.
///
/// `published` regions are visible to other actors: only `Commit` may write
/// them, and for per-CPU published regions a commit must target the slice
/// owned by the CPU the sequence is currently running on.
///
/// `per_cpu` regions are sized `ncpus * words` at simulation time; index
/// arithmetic (`cpu * words + offset`) is the program's job, exactly as it is
/// in the real asm.
#[derive(Clone, Debug)]
pub struct RegionDecl {
    pub name: String,
    pub published: bool,
    pub per_cpu: bool,
    /// Size in words (per CPU if `per_cpu`).
    pub words: usize,
    /// Initial fill value.
    pub init: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub regions: Vec<RegionDecl>,
}

impl Layout {
    #[must_use] 
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, decl: RegionDecl) -> RegionId {
        self.regions.push(decl);
        RegionId(self.regions.len() - 1)
    }

    pub fn published_per_cpu(&mut self, name: &str, words: usize, init: u64) -> RegionId {
        self.add(RegionDecl { name: name.into(), published: true, per_cpu: true, words, init })
    }

    pub fn published_shared(&mut self, name: &str, words: usize, init: u64) -> RegionId {
        self.add(RegionDecl { name: name.into(), published: true, per_cpu: false, words, init })
    }

    pub fn scratch_per_cpu(&mut self, name: &str, words: usize, init: u64) -> RegionId {
        self.add(RegionDecl { name: name.into(), published: false, per_cpu: true, words, init })
    }

    pub fn scratch_shared(&mut self, name: &str, words: usize, init: u64) -> RegionId {
        self.add(RegionDecl { name: name.into(), published: false, per_cpu: false, words, init })
    }

    #[must_use] 
    pub fn decl(&self, r: RegionId) -> &RegionDecl {
        &self.regions[r.0]
    }
}

#[derive(Clone, Debug)]
pub struct Program {
    pub name: String,
    pub ops: Vec<Op>,
    /// Optional result, evaluated after a successful commit (e.g. the popped
    /// element). Early exit yields no result.
    pub ret: Option<Operand>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidateError {
    Empty,
    CommitNotLast { at: usize },
    MissingCommit,
    MultipleExits { at: usize },
    UnknownRegion { at: usize, region: RegionId },
    ScratchStoreToPublished { at: usize, region: RegionId },
    CommitToScratch { region: RegionId },
    UseBeforeDef { at: usize, reg: Reg },
    RetUndefined { reg: Reg },
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Program {
    /// Check the structural invariants of a valid restartable sequence.
    ///
    /// # Errors
    ///
    /// Returns the first structural violation found: no/misplaced commit,
    /// multiple exits, writes to the wrong region kind, or a register used
    /// before it is defined.
    pub fn validate(&self, layout: &Layout) -> Result<(), ValidateError> {
        if self.ops.is_empty() {
            return Err(ValidateError::Empty);
        }
        let mut defined = vec![false; self.max_reg().map_or(0, |r| r + 1)];
        let check_use = |at: usize, operand: Operand, defined: &[bool]| match operand {
            Operand::Reg(r) if !defined.get(r).copied().unwrap_or(false) => {
                Err(ValidateError::UseBeforeDef { at, reg: r })
            }
            _ => Ok(()),
        };
        let check_region = |at: usize, region: RegionId| {
            if region.0 >= layout.regions.len() {
                Err(ValidateError::UnknownRegion { at, region })
            } else {
                Ok(())
            }
        };
        let mut seen_exit = false;
        let last = self.ops.len() - 1;
        for (at, op) in self.ops.iter().enumerate() {
            match *op {
                Op::Commit { .. } if at != last => {
                    return Err(ValidateError::CommitNotLast { at });
                }
                _ => {}
            }
            match *op {
                Op::CpuId { dst } | Op::CpuIdHoisted { dst } | Op::Const { dst, .. } => {
                    defined[dst] = true;
                }
                Op::Load { dst, addr } => {
                    check_region(at, addr.region)?;
                    check_use(at, addr.index, &defined)?;
                    defined[dst] = true;
                }
                Op::Bin { dst, lhs, rhs, .. } => {
                    check_use(at, lhs, &defined)?;
                    check_use(at, rhs, &defined)?;
                    defined[dst] = true;
                }
                Op::ExitIf { lhs, rhs, .. } => {
                    if seen_exit {
                        return Err(ValidateError::MultipleExits { at });
                    }
                    seen_exit = true;
                    check_use(at, lhs, &defined)?;
                    check_use(at, rhs, &defined)?;
                }
                Op::StoreScratch { addr, src } => {
                    check_region(at, addr.region)?;
                    if layout.decl(addr.region).published {
                        return Err(ValidateError::ScratchStoreToPublished { at, region: addr.region });
                    }
                    check_use(at, addr.index, &defined)?;
                    check_use(at, src, &defined)?;
                }
                Op::Commit { addr, src } => {
                    check_region(at, addr.region)?;
                    if !layout.decl(addr.region).published {
                        return Err(ValidateError::CommitToScratch { region: addr.region });
                    }
                    check_use(at, addr.index, &defined)?;
                    check_use(at, src, &defined)?;
                }
            }
        }
        match self.ops[last] {
            Op::Commit { .. } => {}
            _ => return Err(ValidateError::MissingCommit),
        }
        if let Some(Operand::Reg(r)) = self.ret
            && !defined.get(r).copied().unwrap_or(false) {
                return Err(ValidateError::RetUndefined { reg: r });
            }
        Ok(())
    }

    fn max_reg(&self) -> Option<Reg> {
        let mut regs: Vec<Reg> = Vec::new();
        let push_operand = |regs: &mut Vec<Reg>, o: Operand| {
            if let Operand::Reg(r) = o {
                regs.push(r);
            }
        };
        for op in &self.ops {
            match *op {
                Op::CpuId { dst } | Op::CpuIdHoisted { dst } | Op::Const { dst, .. } => {
                    regs.push(dst);
                }
                Op::Load { dst, addr } => {
                    regs.push(dst);
                    push_operand(&mut regs, addr.index);
                }
                Op::Bin { dst, lhs, rhs, .. } => {
                    regs.push(dst);
                    push_operand(&mut regs, lhs);
                    push_operand(&mut regs, rhs);
                }
                Op::ExitIf { lhs, rhs, .. } => {
                    push_operand(&mut regs, lhs);
                    push_operand(&mut regs, rhs);
                }
                Op::StoreScratch { addr, src } | Op::Commit { addr, src } => {
                    push_operand(&mut regs, addr.index);
                    push_operand(&mut regs, src);
                }
            }
        }
        if let Some(Operand::Reg(r)) = self.ret {
            regs.push(r);
        }
        regs.into_iter().max()
    }
}

/// Builder for programs. Consuming `commit`/`commit_ret` finishers make
/// "exactly one committing store, and it is terminal" true by construction.
pub struct SeqBuilder {
    name: String,
    ops: Vec<Op>,
    next: Reg,
}

impl SeqBuilder {
    #[must_use] 
    pub fn new(name: &str) -> Self {
        Self { name: name.into(), ops: Vec::new(), next: 0 }
    }

    fn fresh(&mut self) -> Reg {
        let r = self.next;
        self.next += 1;
        r
    }

    pub fn cpu_id(&mut self) -> Reg {
        let dst = self.fresh();
        self.ops.push(Op::CpuId { dst });
        dst
    }

    /// See [`Op::CpuIdHoisted`] — deliberately buggy, for checker tests.
    pub fn cpu_id_hoisted(&mut self) -> Reg {
        let dst = self.fresh();
        self.ops.push(Op::CpuIdHoisted { dst });
        dst
    }

    pub fn constant(&mut self, value: u64) -> Reg {
        let dst = self.fresh();
        self.ops.push(Op::Const { dst, value });
        dst
    }

    pub fn load(&mut self, region: RegionId, index: Operand) -> Reg {
        let dst = self.fresh();
        self.ops.push(Op::Load { dst, addr: Addr { region, index } });
        dst
    }

    pub fn bin(&mut self, op: BinOp, lhs: Operand, rhs: Operand) -> Reg {
        let dst = self.fresh();
        self.ops.push(Op::Bin { dst, op, lhs, rhs });
        dst
    }

    pub fn exit_if(&mut self, cond: Cond, lhs: Operand, rhs: Operand) {
        self.ops.push(Op::ExitIf { cond, lhs, rhs });
    }

    pub fn store_scratch(&mut self, region: RegionId, index: Operand, src: Operand) {
        self.ops.push(Op::StoreScratch { addr: Addr { region, index }, src });
    }

    #[must_use] 
    pub fn commit(self, region: RegionId, index: Operand, src: Operand) -> Program {
        self.finish(region, index, src, None)
    }

    #[must_use] 
    pub fn commit_ret(self, region: RegionId, index: Operand, src: Operand, ret: Operand) -> Program {
        self.finish(region, index, src, Some(ret))
    }

    fn finish(
        mut self,
        region: RegionId,
        index: Operand,
        src: Operand,
        ret: Option<Operand>,
    ) -> Program {
        self.ops.push(Op::Commit { addr: Addr { region, index }, src });
        Program { name: self.name, ops: self.ops, ret }
    }
}
