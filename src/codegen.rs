//! Codegen backend: compile the IR to executable x86-64 machine code with an
//! `rseq_cs` descriptor, so the program the model checker verified and the
//! program the kernel runs are the same artifact.
//!
//! Emitted layout, all in one anonymous mapping (RW while writing, then RX):
//!
//! ```text
//! offset  0: struct rseq_cs descriptor (32-byte aligned at page start)
//! offset 32: entry:  lea rax, [rip + descriptor]
//!            retry:  mov [rdi + 8], rax          ; arm rseq_cs
//!            start:  ...body...
//!                    mov [mem], src              ; the committing store
//!            post:   ...move ret value to rax...
//!                    ret
//!            exit:   mov rax, -1                 ; EXITED sentinel
//!                    ret
//!                    .long 0x53053053            ; glibc's RSEQ_SIG
//!            abort:  jmp retry
//! ```
//!
//! Calling convention of the generated function (System V):
//! `fn(area: *mut RseqArea /*rdi*/, bases: *const u64 /*rsi*/,
//!     params: *const u64 /*rdx*/) -> u64 /*rax*/`.
//! `bases[r]` is the base address of region `r`; per-CPU regions are sized
//! `MAX_CPUS * words`, and index arithmetic is in the program, as in the IR.
//!
//! Register discipline: rdi/rsi/rdx are pinned to the three pointers for the
//! whole sequence, rax and r11 are transient scratch (descriptor address,
//! region bases, immediates), and virtual registers live in {rcx, r8, r9,
//! r10}, allocated linearly with reuse after last use.

use core::cell::UnsafeCell;
use std::fmt;

use crate::ir::{BinOp, Cond, Layout, Op, Operand, Program, Reg, RegionId, ValidateError};
use crate::rt::{MAX_CPUS, RseqArea, current_area};

/// Return value of a compiled sequence that exited early instead of
/// committing. Programs whose committed `ret` could legitimately be
/// `u64::MAX` need a different sentinel; none of ours do.
pub const EXITED: u64 = u64::MAX;

const RSEQ_SIG: u32 = 0x5305_3053;

// Physical register numbers.
const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RSI: u8 = 6;
const RDI: u8 = 7;
const R8: u8 = 8;
const R9: u8 = 9;
const R10: u8 = 10;
const R11: u8 = 11;

/// Virtual registers live here; rax/r11 stay free as transient scratch.
const POOL: [u8; 4] = [RCX, R8, R9, R10];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    Invalid(ValidateError),
    /// `CpuIdHoisted` is a model-only bug demonstrator; generating real code
    /// for it would be manufacturing the bug it exists to catch.
    HoistedCpuId,
    /// Shift amounts must be immediates (variable shifts need cl juggling
    /// the allocator does not do yet).
    ShiftByRegister,
    /// More simultaneously-live virtual registers than the pool holds.
    OutOfRegisters,
    MmapFailed(isize),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A program compiled to executable memory, descriptor included.
#[derive(Debug)]
pub struct CompiledSeq {
    base: *mut u8,
    map_len: usize,
    entry: unsafe extern "C" fn(*mut RseqArea, *const u64, *const u64) -> u64,
    nregions: usize,
    nparams: usize,
    pub name: String,
}

// Safety: the mapping is immutable (RX) after construction.
unsafe impl Send for CompiledSeq {}
unsafe impl Sync for CompiledSeq {}

impl Drop for CompiledSeq {
    fn drop(&mut self) {
        // A thread that ran this sequence may still have its rseq_cs armed
        // with our descriptor; the kernel reads the descriptor on the next
        // resume and only then clears it. `RegionSet::call` disarms after
        // every call, so by the time a correctly-used CompiledSeq drops, no
        // thread points here any more.
        unsafe {
            sys::munmap(self.base, self.map_len);
        }
    }
}

impl CompiledSeq {
    /// Compile `prog` against `layout`.
    ///
    /// # Errors
    ///
    /// Fails on an invalid program, on model-only ops (`CpuIdHoisted`), on
    /// variable shift amounts, on register-pool exhaustion, or if the
    /// executable mapping cannot be created.
    pub fn compile(prog: &Program, layout: &Layout) -> Result<Self, CompileError> {
        prog.validate(layout).map_err(CompileError::Invalid)?;
        let blob = emit(prog)?;
        let map_len = blob.buf.len().next_multiple_of(4096);
        unsafe {
            let base = sys::mmap_rw(map_len).map_err(CompileError::MmapFailed)?;
            core::ptr::copy_nonoverlapping(blob.buf.as_ptr(), base, blob.buf.len());
            // Fill in the descriptor now that absolute addresses exist.
            // (The mapping is page-aligned, so offset 0 is 8-aligned.)
            #[allow(clippy::cast_ptr_alignment)]
            let d = base.cast::<u64>();
            d.add(0).write(0); // version, flags
            d.add(1).write(base as u64 + blob.start_off as u64);
            d.add(2).write((blob.post_off - blob.start_off) as u64);
            d.add(3).write(base as u64 + blob.abort_off as u64);
            sys::mprotect_rx(base, map_len).map_err(CompileError::MmapFailed)?;
            Ok(Self {
                base,
                map_len,
                entry: core::mem::transmute::<
                    *mut u8,
                    unsafe extern "C" fn(*mut RseqArea, *const u64, *const u64) -> u64,
                >(base.add(ENTRY_OFF)),
                nregions: layout.regions.len(),
                nparams: prog
                    .ops
                    .iter()
                    .filter_map(|op| match op {
                        Op::Param { index, .. } => Some(index + 1),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0),
                name: prog.name.clone(),
            })
        }
    }

    /// Raw entry to the generated code.
    ///
    /// # Safety
    ///
    /// `area` must be the *calling thread's* registered rseq area; `bases`
    /// must hold one valid base address per region of the layout the program
    /// was compiled against (per-CPU regions sized `MAX_CPUS * words`);
    /// `params` must hold at least `nparams` values. The caller should clear
    /// `area->rseq_cs` before this `CompiledSeq` is dropped.
    pub unsafe fn call_raw(
        &self,
        area: *mut RseqArea,
        bases: *const u64,
        params: *const u64,
    ) -> u64 {
        unsafe { (self.entry)(area, bases, params) }
    }

    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.base, self.map_len) }
    }
}

const ENTRY_OFF: usize = 32;

struct Blob {
    buf: Vec<u8>,
    start_off: usize,
    post_off: usize,
    abort_off: usize,
}

fn emit(prog: &Program) -> Result<Blob, CompileError> {
    // Last use (op index) of every virtual register; the ret operand keeps
    // its register live to the end.
    let mut last_use: Vec<usize> = Vec::new();
    let mark = |o: Operand, at: usize, last_use: &mut Vec<usize>| {
        if let Operand::Reg(r) = o {
            if r >= last_use.len() {
                last_use.resize(r + 1, 0);
            }
            last_use[r] = last_use[r].max(at);
        }
    };
    for (at, op) in prog.ops.iter().enumerate() {
        match *op {
            Op::Load { addr, .. } => mark(addr.index, at, &mut last_use),
            Op::Bin { lhs, rhs, .. } | Op::ExitIf { lhs, rhs, .. } => {
                mark(lhs, at, &mut last_use);
                mark(rhs, at, &mut last_use);
            }
            Op::StoreScratch { addr, src } | Op::Commit { addr, src } => {
                mark(addr.index, at, &mut last_use);
                mark(src, at, &mut last_use);
            }
            _ => {}
        }
    }
    if let Some(o) = prog.ret {
        mark(o, usize::MAX, &mut last_use);
    }

    let mut a = Asm { buf: vec![0u8; ENTRY_OFF], exit_patches: Vec::new() };
    let mut ra = RegAlloc { map: vec![None; last_use.len().max(prog.max_reg_bound())], free: POOL.to_vec() };

    // entry: lea rax, [rip + descriptor]   (descriptor is at offset 0)
    a.buf.extend_from_slice(&[0x48, 0x8D, 0x05]);
    let disp = -((a.buf.len() + 4) as i64);
    a.imm32(disp as i32);
    // retry: mov [rdi + 8], rax
    let retry_off = a.buf.len();
    a.buf.extend_from_slice(&[0x48, 0x89, 0x47, 0x08]);
    let start_off = a.buf.len();

    for (at, op) in prog.ops.iter().enumerate() {
        emit_op(&mut a, &mut ra, *op)?;
        ra.release_dead(&last_use, at);
    }
    let post_off = a.buf.len();

    // Committed-path epilogue: ret value into rax, then ret.
    match prog.ret {
        Some(Operand::Reg(r)) => {
            let p = ra.phys(r)?;
            a.mov_rr(RAX, p);
        }
        Some(Operand::Imm(v)) => a.mov_ri(RAX, v),
        None => a.buf.extend_from_slice(&[0x31, 0xC0]), // xor eax, eax
    }
    a.buf.push(0xC3);

    // Early-exit stub: mov rax, -1 ; ret
    let exit_off = a.buf.len();
    a.buf.extend_from_slice(&[0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF]);
    a.buf.push(0xC3);
    for patch in a.exit_patches.clone() {
        let disp = (exit_off as i64 - (patch as i64 + 4)) as i32;
        a.buf[patch..patch + 4].copy_from_slice(&disp.to_le_bytes());
    }

    // Abort signature and handler.
    a.buf.extend_from_slice(&RSEQ_SIG.to_le_bytes());
    let abort_off = a.buf.len();
    a.buf.push(0xE9);
    let disp = (retry_off as i64 - (a.buf.len() as i64 + 4)) as i32;
    a.imm32(disp);

    Ok(Blob { buf: a.buf, start_off, post_off, abort_off })
}

// One match arm per op; splitting it would scatter the encoding logic.
#[allow(clippy::too_many_lines)]
fn emit_op(a: &mut Asm, ra: &mut RegAlloc, op: Op) -> Result<(), CompileError> {
    match op {
        Op::CpuId { dst } => {
            let d = ra.alloc(dst, &[])?;
            // mov d32, [rdi + 4] — 32-bit load zero-extends.
            a.rex_opt(false, d, 0, RDI);
            a.buf.push(0x8B);
            a.modrm(0b01, d, RDI);
            a.buf.push(0x04);
        }
        Op::CpuIdHoisted { .. } => return Err(CompileError::HoistedCpuId),
        Op::Const { dst, value } => {
            let d = ra.alloc(dst, &[])?;
            a.mov_ri(d, value);
        }
        Op::Param { dst, index } => {
            let d = ra.alloc(dst, &[])?;
            // mov d, [rdx + index*8]
            a.rex(true, d, 0, RDX);
            a.buf.push(0x8B);
            a.modrm(0b10, d, RDX);
            a.imm32((index * 8) as i32);
        }
        Op::Load { dst, addr } => {
            let index = ra.operand_phys(addr.index)?;
            a.load_base(addr.region);
            let d = ra.alloc(dst, &index.into_iter().collect::<Vec<_>>())?;
            match addr.index {
                Operand::Reg(r) => {
                    let i = ra.phys(r)?;
                    // mov d, [rax + i*8]
                    a.rex(true, d, i, RAX);
                    a.buf.push(0x8B);
                    a.modrm(0b00, d, 0b100);
                    a.sib(3, i, RAX);
                }
                Operand::Imm(k) => {
                    // mov d, [rax + k*8]
                    a.rex(true, d, 0, RAX);
                    a.buf.push(0x8B);
                    a.modrm(0b10, d, RAX);
                    a.imm32((k * 8) as i32);
                }
            }
        }
        Op::Bin { dst, op, lhs, rhs } => {
            let mut avoid = Vec::new();
            avoid.extend(ra.operand_phys(lhs)?);
            avoid.extend(ra.operand_phys(rhs)?);
            let d = ra.alloc(dst, &avoid)?;
            match lhs {
                Operand::Reg(r) => {
                    let l = ra.phys(r)?;
                    if l != d {
                        a.mov_rr(d, l);
                    }
                }
                Operand::Imm(v) => a.mov_ri(d, v),
            }
            match op {
                BinOp::Shl | BinOp::Shr => {
                    let Operand::Imm(v) = rhs else {
                        return Err(CompileError::ShiftByRegister);
                    };
                    // shl/shr d, imm8
                    a.rex(true, 0, 0, d);
                    a.buf.push(0xC1);
                    a.modrm(0b11, if matches!(op, BinOp::Shl) { 4 } else { 5 }, d);
                    a.buf.push((v & 63) as u8);
                }
                _ => {
                    let s = match rhs {
                        Operand::Reg(r) => ra.phys(r)?,
                        Operand::Imm(v) => {
                            a.mov_ri(R11, v);
                            R11
                        }
                    };
                    match op {
                        // op r/m64, r64 forms: d = d OP s.
                        BinOp::Add => a.alu_rr(0x01, d, s),
                        BinOp::Sub => a.alu_rr(0x29, d, s),
                        BinOp::And => a.alu_rr(0x21, d, s),
                        BinOp::Or => a.alu_rr(0x09, d, s),
                        BinOp::Xor => a.alu_rr(0x31, d, s),
                        BinOp::Mul => {
                            // imul d, s (r64, r/m64 form).
                            a.rex(true, d, 0, s);
                            a.buf.extend_from_slice(&[0x0F, 0xAF]);
                            a.modrm(0b11, d, s);
                        }
                        BinOp::Shl | BinOp::Shr => unreachable!(),
                    }
                }
            }
        }
        Op::ExitIf { cond, lhs, rhs } => {
            let (va, vb) = match (lhs, rhs) {
                (Operand::Reg(l), Operand::Reg(r)) => (ra.phys(l)?, ra.phys(r)?),
                (Operand::Reg(l), Operand::Imm(v)) => {
                    a.mov_ri(R11, v);
                    (ra.phys(l)?, R11)
                }
                (Operand::Imm(v), Operand::Reg(r)) => {
                    a.mov_ri(R11, v);
                    (R11, ra.phys(r)?)
                }
                (Operand::Imm(l), Operand::Imm(r)) => {
                    a.mov_ri(RAX, l);
                    a.mov_ri(R11, r);
                    (RAX, R11)
                }
            };
            // cmp va, vb
            a.alu_rr(0x39, va, vb);
            // jcc rel32 to the exit stub (patched later).
            let cc: u8 = match cond {
                Cond::Eq => 0x84,
                Cond::Ne => 0x85,
                Cond::Lt => 0x82, // jb, unsigned
                Cond::Ge => 0x83, // jae, unsigned
            };
            a.buf.extend_from_slice(&[0x0F, cc]);
            a.exit_patches.push(a.buf.len());
            a.imm32(0);
        }
        Op::StoreScratch { addr, src } | Op::Commit { addr, src } => {
            a.load_base(addr.region);
            let s = match src {
                Operand::Reg(r) => ra.phys(r)?,
                Operand::Imm(v) => {
                    a.mov_ri(R11, v);
                    R11
                }
            };
            match addr.index {
                Operand::Reg(r) => {
                    let i = ra.phys(r)?;
                    // mov [rax + i*8], s
                    a.rex(true, s, i, RAX);
                    a.buf.push(0x89);
                    a.modrm(0b00, s, 0b100);
                    a.sib(3, i, RAX);
                }
                Operand::Imm(k) => {
                    // mov [rax + k*8], s
                    a.rex(true, s, 0, RAX);
                    a.buf.push(0x89);
                    a.modrm(0b10, s, RAX);
                    a.imm32((k * 8) as i32);
                }
            }
        }
    }
    Ok(())
}

struct Asm {
    buf: Vec<u8>,
    exit_patches: Vec<usize>,
}

impl Asm {
    fn rex(&mut self, w: bool, r: u8, x: u8, b: u8) {
        self.buf.push(
            0x40 | (u8::from(w) << 3) | ((r >> 3) << 2) | ((x >> 3) << 1) | (b >> 3),
        );
    }

    /// REX only when needed (for 32-bit ops touching r8..r15).
    fn rex_opt(&mut self, w: bool, r: u8, x: u8, b: u8) {
        if w || r >= 8 || x >= 8 || b >= 8 {
            self.rex(w, r, x, b);
        }
    }

    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        self.buf.push((mode << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    fn sib(&mut self, scale: u8, index: u8, base: u8) {
        self.buf.push((scale << 6) | ((index & 7) << 3) | (base & 7));
    }

    fn imm32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// movabs r, imm64
    fn mov_ri(&mut self, r: u8, v: u64) {
        self.rex(true, 0, 0, r);
        self.buf.push(0xB8 | (r & 7));
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// mov dst, src (r/m64, r64)
    fn mov_rr(&mut self, dst: u8, src: u8) {
        self.alu_rr(0x89, dst, src);
    }

    /// `opcode` in "r/m64, r64" form: dst = dst OP src.
    fn alu_rr(&mut self, opcode: u8, dst: u8, src: u8) {
        self.rex(true, src, 0, dst);
        self.buf.push(opcode);
        self.modrm(0b11, src, dst);
    }

    /// mov rax, [rsi + region*8] — region base into the rax scratch.
    fn load_base(&mut self, region: RegionId) {
        self.rex(true, RAX, 0, RSI);
        self.buf.push(0x8B);
        self.modrm(0b10, RAX, RSI);
        self.imm32((region.0 * 8) as i32);
    }
}

struct RegAlloc {
    map: Vec<Option<u8>>,
    free: Vec<u8>,
}

impl RegAlloc {
    fn phys(&self, r: Reg) -> Result<u8, CompileError> {
        self.map
            .get(r)
            .copied()
            .flatten()
            .ok_or(CompileError::Invalid(ValidateError::UseBeforeDef { at: 0, reg: r }))
    }

    fn operand_phys(&self, o: Operand) -> Result<Option<u8>, CompileError> {
        match o {
            Operand::Reg(r) => Ok(Some(self.phys(r)?)),
            Operand::Imm(_) => Ok(None),
        }
    }

    /// Allocate a physical register for `dst`, avoiding the physical
    /// registers in `avoid` (operands of the same op, which must stay
    /// readable while the op computes).
    fn alloc(&mut self, dst: Reg, avoid: &[u8]) -> Result<u8, CompileError> {
        let pos = self
            .free
            .iter()
            .position(|p| !avoid.contains(p))
            .ok_or(CompileError::OutOfRegisters)?;
        let p = self.free.remove(pos);
        if dst >= self.map.len() {
            self.map.resize(dst + 1, None);
        }
        self.map[dst] = Some(p);
        Ok(p)
    }

    /// Return pool registers whose virtual register died at or before `at`.
    fn release_dead(&mut self, last_use: &[usize], at: usize) {
        for (v, slot) in self.map.iter_mut().enumerate() {
            if let Some(p) = *slot {
                let dead = last_use.get(v).copied().unwrap_or(0) <= at;
                if dead {
                    *slot = None;
                    self.free.push(p);
                }
            }
        }
    }
}

impl Program {
    /// Upper bound on virtual register numbers, for allocator sizing.
    fn max_reg_bound(&self) -> usize {
        self.ops.len() * 2
    }
}

/// Owns the memory regions a compiled program operates on, and drives calls.
pub struct RegionSet {
    layout: Layout,
    allocs: Vec<Box<[UnsafeCell<u64>]>>,
    bases: Box<[u64]>,
}

// Safety: cross-thread mutation happens only inside the rseq sequences.
unsafe impl Sync for RegionSet {}

impl RegionSet {
    #[must_use]
    pub fn new(layout: &Layout) -> Self {
        let allocs: Vec<Box<[UnsafeCell<u64>]>> = layout
            .regions
            .iter()
            .map(|d| {
                let len = if d.per_cpu { d.words * MAX_CPUS } else { d.words };
                (0..len).map(|_| UnsafeCell::new(d.init)).collect()
            })
            .collect();
        let bases = allocs.iter().map(|a| a.as_ptr() as u64).collect();
        Self { layout: layout.clone(), allocs, bases }
    }

    /// Run `seq` on the current thread. Returns the sequence's result
    /// ([`EXITED`] for an early exit), or None if rseq is unavailable.
    ///
    /// # Panics
    ///
    /// Panics if `seq` was compiled for a different region count or needs
    /// more parameters than given.
    #[must_use]
    pub fn call(&self, seq: &CompiledSeq, params: &[u64]) -> Option<u64> {
        assert_eq!(seq.nregions, self.layout.regions.len(), "layout mismatch");
        assert!(params.len() >= seq.nparams, "missing parameters");
        let area = current_area()?;
        let out = unsafe { seq.call_raw(area, self.bases.as_ptr(), params.as_ptr()) };
        // Disarm: leaving rseq_cs pointing at the descriptor is legal while
        // it stays mapped, but clearing here makes dropping the CompiledSeq
        // safe without cross-thread cleanup.
        unsafe {
            core::ptr::write_volatile(&raw mut (*area).rseq_cs, 0);
        }
        Some(out)
    }

    /// Quiescent access to a region (`&mut self`: no calls in flight).
    pub fn region_mut(&mut self, r: RegionId) -> &mut [u64] {
        let a = &mut self.allocs[r.0];
        unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr().cast::<u64>(), a.len()) }
    }
}

/// Raw syscalls for the executable mapping — keeps the crate dependency-free.
mod sys {
    use core::arch::asm;

    const SYS_MMAP: usize = 9;
    const SYS_MPROTECT: usize = 10;
    const SYS_MUNMAP: usize = 11;

    const PROT_READ: usize = 1;
    const PROT_WRITE: usize = 2;
    const PROT_EXEC: usize = 4;
    const MAP_PRIVATE_ANON: usize = 0x22;

    unsafe fn syscall6(nr: usize, args: [usize; 6]) -> isize {
        let ret: isize;
        unsafe {
            asm!(
                "syscall",
                inlateout("rax") nr => ret,
                in("rdi") args[0],
                in("rsi") args[1],
                in("rdx") args[2],
                in("r10") args[3],
                in("r8") args[4],
                in("r9") args[5],
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack)
            );
        }
        ret
    }

    pub unsafe fn mmap_rw(len: usize) -> Result<*mut u8, isize> {
        let ret = unsafe {
            syscall6(
                SYS_MMAP,
                [0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE_ANON, usize::MAX, 0],
            )
        };
        if ret < 0 { Err(ret) } else { Ok(ret as *mut u8) }
    }

    pub unsafe fn mprotect_rx(p: *mut u8, len: usize) -> Result<(), isize> {
        let ret = unsafe { syscall6(SYS_MPROTECT, [p as usize, len, PROT_READ | PROT_EXEC, 0, 0, 0]) };
        if ret < 0 { Err(ret) } else { Ok(()) }
    }

    pub unsafe fn munmap(p: *mut u8, len: usize) {
        let _ = unsafe { syscall6(SYS_MUNMAP, [p as usize, len, 0, 0, 0, 0]) };
    }
}
