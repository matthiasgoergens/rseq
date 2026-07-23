//! Interpreter for the IR, with deterministic abort injection.
//!
//! An abort models the kernel restarting the sequence: execution jumps back to
//! the first instruction, possibly on a different CPU. The simulator takes an
//! explicit plan of abort points, so every interleaving the kernel could
//! produce can be replayed on demand.

use crate::ir::{Addr, BinOp, Cond, Layout, Op, Operand, Program, Reg, RegionId};

#[derive(Clone, Debug)]
pub struct Memory {
    layout: Layout,
    ncpus: usize,
    regions: Vec<Vec<u64>>,
}

impl Memory {
    #[must_use]
    pub fn new(layout: &Layout, ncpus: usize) -> Self {
        let regions = layout
            .regions
            .iter()
            .map(|d| {
                let len = if d.per_cpu { d.words * ncpus } else { d.words };
                vec![d.init; len]
            })
            .collect();
        Self {
            layout: layout.clone(),
            ncpus,
            regions,
        }
    }

    #[must_use]
    pub fn ncpus(&self) -> usize {
        self.ncpus
    }

    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[must_use]
    pub fn region(&self, r: RegionId) -> &[u64] {
        &self.regions[r.0]
    }

    /// Direct mutable access, for test setup only — the simulator itself goes
    /// through `Op` semantics.
    pub fn region_mut(&mut self, r: RegionId) -> &mut [u64] {
        &mut self.regions[r.0]
    }

    fn read(&self, r: RegionId, index: usize) -> Result<u64, SimError> {
        self.regions[r.0]
            .get(index)
            .copied()
            .ok_or(SimError::OutOfBounds {
                region: r,
                index,
                len: self.regions[r.0].len(),
            })
    }

    fn write(&mut self, r: RegionId, index: usize, value: u64) -> Result<(), SimError> {
        let len = self.regions[r.0].len();
        match self.regions[r.0].get_mut(index) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(SimError::OutOfBounds {
                region: r,
                index,
                len,
            }),
        }
    }

    /// Snapshot of all published regions, for prefix-purity checks.
    #[must_use]
    pub fn published_snapshot(&self) -> Vec<(RegionId, Vec<u64>)> {
        self.layout
            .regions
            .iter()
            .enumerate()
            .filter(|(_, d)| d.published)
            .map(|(i, _)| (RegionId(i), self.regions[i].clone()))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimError {
    OutOfBounds {
        region: RegionId,
        index: usize,
        len: usize,
    },
    /// A commit targeted a per-CPU published slice owned by a CPU other than
    /// the one the sequence is currently running on. This is the stale-CPU-id
    /// bug class: the index was computed from a CPU id that migration made
    /// obsolete.
    ForeignCommit {
        region: RegionId,
        index: usize,
        running_on: usize,
        owner: usize,
    },
    UndefinedReg(Reg),
    BadCpu {
        cpu: usize,
        ncpus: usize,
    },
    MissingParam {
        index: usize,
        nparams: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Committed { ret: Option<u64> },
    Exited,
}

/// Abort when about to execute op `at` of the current attempt (so `at = 0`
/// aborts before anything ran, `at = len - 1` aborts just before the commit).
/// `at = len` means "after the commit" and is only honoured when the simulator
/// is told to model a buggy descriptor whose window includes the committing
/// store (`post_commit_in_window`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbortPoint {
    pub at: usize,
    pub new_cpu: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimConfig {
    /// Model an off-by-one `post_commit_offset`: the abort window incorrectly
    /// includes the committing store, so an abort delivered right after the
    /// commit restarts the sequence and commits twice.
    pub post_commit_in_window: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunResult {
    pub outcome: Outcome,
    /// The CPU the run actually finished on — the destination of the last
    /// abort that *fired* (plan entries positioned past an early exit never
    /// fire and do not migrate anything).
    pub final_cpu: usize,
}

/// Run `prog` to completion, applying `plan` aborts in order. Each abort
/// restarts the sequence from the top on `new_cpu`. Plan entries that never
/// match (e.g. beyond an early exit) are simply unused.
///
/// # Errors
///
/// Returns a [`SimError`] on out-of-bounds access, a commit to another CPU's
/// slice, an undefined register, or a CPU id outside the simulated range.
pub fn run(
    prog: &Program,
    mem: &mut Memory,
    params: &[u64],
    start_cpu: usize,
    plan: &[AbortPoint],
    cfg: SimConfig,
) -> Result<RunResult, SimError> {
    if start_cpu >= mem.ncpus {
        return Err(SimError::BadCpu {
            cpu: start_cpu,
            ncpus: mem.ncpus,
        });
    }
    let mut cpu = start_cpu;
    let mut plan_iter = plan.iter();
    let mut pending = plan_iter.next();
    'attempt: loop {
        let mut regs = RegFile::default();
        for (i, op) in prog.ops.iter().enumerate() {
            if let Some(p) = pending
                && p.at == i
            {
                if p.new_cpu >= mem.ncpus {
                    return Err(SimError::BadCpu {
                        cpu: p.new_cpu,
                        ncpus: mem.ncpus,
                    });
                }
                cpu = p.new_cpu;
                pending = plan_iter.next();
                continue 'attempt;
            }
            match step(op, &mut regs, mem, params, cpu, start_cpu)? {
                Step::Continue => {}
                Step::Exit => {
                    return Ok(RunResult {
                        outcome: Outcome::Exited,
                        final_cpu: cpu,
                    });
                }
            }
        }
        // The commit (last op) has executed. A correctly-sized abort window
        // ends before it; only the modelled descriptor bug lets an abort
        // land here and restart a completed sequence.
        if cfg.post_commit_in_window
            && let Some(p) = pending
            && p.at == prog.ops.len()
        {
            if p.new_cpu >= mem.ncpus {
                return Err(SimError::BadCpu {
                    cpu: p.new_cpu,
                    ncpus: mem.ncpus,
                });
            }
            cpu = p.new_cpu;
            pending = plan_iter.next();
            continue 'attempt;
        }
        let ret = match prog.ret {
            Some(o) => Some(regs.eval(o)?),
            None => None,
        };
        return Ok(RunResult {
            outcome: Outcome::Committed { ret },
            final_cpu: cpu,
        });
    }
}

/// Execute only `ops[0..k]` on `cpu`, then stop (as if the thread were
/// preempted there and never resumed). Used for prefix-purity checking:
/// no prefix that excludes the commit may change published memory.
///
/// # Errors
///
/// Same failure modes as [`run`].
pub fn run_prefix(
    prog: &Program,
    mem: &mut Memory,
    params: &[u64],
    cpu: usize,
    k: usize,
) -> Result<(), SimError> {
    if cpu >= mem.ncpus {
        return Err(SimError::BadCpu {
            cpu,
            ncpus: mem.ncpus,
        });
    }
    let mut regs = RegFile::default();
    for op in &prog.ops[..k] {
        match step(op, &mut regs, mem, params, cpu, cpu)? {
            Step::Continue => {}
            Step::Exit => return Ok(()),
        }
    }
    Ok(())
}

enum Step {
    Continue,
    Exit,
}

#[derive(Default)]
struct RegFile {
    regs: Vec<Option<u64>>,
}

impl RegFile {
    fn set(&mut self, r: Reg, v: u64) {
        if r >= self.regs.len() {
            self.regs.resize(r + 1, None);
        }
        self.regs[r] = Some(v);
    }

    fn eval(&self, o: Operand) -> Result<u64, SimError> {
        match o {
            Operand::Imm(v) => Ok(v),
            Operand::Reg(r) => self
                .regs
                .get(r)
                .copied()
                .flatten()
                .ok_or(SimError::UndefinedReg(r)),
        }
    }
}

fn step(
    op: &Op,
    regs: &mut RegFile,
    mem: &mut Memory,
    params: &[u64],
    cpu: usize,
    start_cpu: usize,
) -> Result<Step, SimError> {
    match *op {
        Op::CpuId { dst } => regs.set(dst, cpu as u64),
        Op::CpuIdHoisted { dst } => regs.set(dst, start_cpu as u64),
        Op::Const { dst, value } => regs.set(dst, value),
        Op::Param { dst, index } => {
            let v = params.get(index).copied().ok_or(SimError::MissingParam {
                index,
                nparams: params.len(),
            })?;
            regs.set(dst, v);
        }
        Op::Load { dst, addr } => {
            let index = index_of(regs, addr)?;
            let v = mem.read(addr.region, index)?;
            regs.set(dst, v);
        }
        Op::Bin { dst, op, lhs, rhs } => {
            let a = regs.eval(lhs)?;
            let b = regs.eval(rhs)?;
            let v = match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::And => a & b,
                BinOp::Or => a | b,
                BinOp::Xor => a ^ b,
                BinOp::Shl => a.wrapping_shl(b as u32),
                BinOp::Shr => a.wrapping_shr(b as u32),
            };
            regs.set(dst, v);
        }
        Op::ExitIf { cond, lhs, rhs } => {
            let a = regs.eval(lhs)?;
            let b = regs.eval(rhs)?;
            let taken = match cond {
                Cond::Eq => a == b,
                Cond::Ne => a != b,
                Cond::Lt => a < b,
                Cond::Ge => a >= b,
            };
            if taken {
                return Ok(Step::Exit);
            }
        }
        Op::StoreScratch { addr, src } => {
            let index = index_of(regs, addr)?;
            let v = regs.eval(src)?;
            mem.write(addr.region, index, v)?;
        }
        Op::Commit { addr, src } => {
            let index = index_of(regs, addr)?;
            let decl = mem.layout.decl(addr.region);
            if decl.per_cpu {
                let owner = index / decl.words;
                if owner != cpu {
                    return Err(SimError::ForeignCommit {
                        region: addr.region,
                        index,
                        running_on: cpu,
                        owner,
                    });
                }
            }
            let v = regs.eval(src)?;
            mem.write(addr.region, index, v)?;
        }
    }
    Ok(Step::Continue)
}

fn index_of(regs: &RegFile, addr: Addr) -> Result<usize, SimError> {
    Ok(regs.eval(addr.index)? as usize)
}
