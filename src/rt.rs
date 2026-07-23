//! Live-kernel runtime: access to the glibc-registered rseq area, plus
//! hand-written x86-64 restartable sequences with their `rseq_cs`
//! descriptors.
//!
//! These sequences are the asm the IR backend will eventually emit; writing
//! them by hand first validates the ABI understanding (descriptor layout,
//! abort signature, retry protocol) against the real kernel, and their
//! stress tests become the harness generated code must pass too.
//!
//! Registration: since glibc 2.35 every thread is auto-registered at start
//! and the area is reachable at `thread pointer + __rseq_offset`, so no
//! syscalls are needed here.

use core::arch::global_asm;
use core::cell::UnsafeCell;

/// Upper bound on CPU ids we size per-CPU arrays for. 32 KiB per array —
/// cheap, and safely above any machine this will meet.
pub const MAX_CPUS: usize = 4096;

/// "No node" sentinel for the freelist (must match the `-1` in the asm).
pub const NIL: u64 = u64::MAX;

/// The kernel's `struct rseq` (uapi/linux/rseq.h). Only `cpu_id` and
/// `rseq_cs` are touched here; both exist in every registered version.
#[repr(C)]
pub struct RseqArea {
    pub cpu_id_start: u32,
    pub cpu_id: u32,
    pub rseq_cs: u64,
    pub flags: u32,
    pub node_id: u32,
    pub mm_cid: u32,
}

unsafe extern "C" {
    static __rseq_offset: isize;
    static __rseq_size: u32;
}

/// The current thread's rseq area, or None if glibc did not register one
/// (pre-2.35 glibc, or a kernel without rseq).
#[must_use]
pub fn current_area() -> Option<*mut RseqArea> {
    unsafe {
        if __rseq_size == 0 {
            return None;
        }
        // %fs:0 holds the TCB self-pointer on x86-64 glibc.
        let tp: usize;
        core::arch::asm!(
            "mov {tp}, qword ptr fs:[0]",
            tp = out(reg) tp,
            options(nostack, preserves_flags, readonly)
        );
        let area = (tp as isize + __rseq_offset) as *mut RseqArea;
        let cpu_id = core::ptr::read_volatile(&raw const (*area).cpu_id);
        if (cpu_id as i32) < 0 {
            None
        } else {
            Some(area)
        }
    }
}

unsafe extern "C" {
    /// counters: base of one u64 per CPU; aborts: per-thread abort tally.
    fn rseq_counter_inc(counters: *mut u64, area: *mut RseqArea, aborts: *mut u64);
    /// Push `node` onto the current CPU's freelist.
    fn rseq_freelist_push(heads: *mut u64, nodes: *mut u64, area: *mut RseqArea, node: u64);
    /// Pop from the current CPU's freelist; returns NIL if empty.
    fn rseq_freelist_pop(heads: *mut u64, nodes: *mut u64, area: *mut RseqArea) -> u64;
}

// The signature glibc registers with (RSEQ_SIG); the 4 bytes preceding the
// abort ip must equal it or the kernel delivers SIGSEGV on abort.
//
// Descriptor layout (struct rseq_cs, 32-byte aligned):
//   u32 version, u32 flags, u64 start_ip, u64 post_commit_offset, u64 abort_ip
//
// Protocol per attempt: arm rseq_cs with the descriptor address, then enter
// the window [start_ip, start_ip + post_commit_offset). Preemption, signal
// delivery, or migration inside the window makes the kernel clear rseq_cs
// and jump to abort_ip, which retries. The committing store is the last
// instruction of the window: post_commit_offset points one past it, so an
// abort can land before the commit but never after it.
global_asm!(
    r#"
.pushsection .data.rseq_cs, "aw"
.balign 32
.Lcounter_inc_cs:
    .long 0, 0
    .quad .Lcounter_inc_start
    .quad .Lcounter_inc_post - .Lcounter_inc_start
    .quad .Lcounter_inc_abort
.popsection

.text
.globl rseq_counter_inc
.p2align 4
rseq_counter_inc:
    // rdi = counters, rsi = area, rdx = abort tally
    lea rax, [rip + .Lcounter_inc_cs]
.Lcounter_inc_retry:
    mov qword ptr [rsi + 8], rax        // arm rseq_cs
.Lcounter_inc_start:
    mov ecx, dword ptr [rsi + 4]        // cpu = area->cpu_id (fresh each attempt)
    mov r8, qword ptr [rdi + rcx*8]
    add r8, 1
    mov qword ptr [rdi + rcx*8], r8     // commit
.Lcounter_inc_post:
    ret
    .long 0x53053053
.Lcounter_inc_abort:
    add qword ptr [rdx], 1
    jmp .Lcounter_inc_retry
"#
);

global_asm!(
    r#"
.pushsection .data.rseq_cs, "aw"
.balign 32
.Lfl_push_cs:
    .long 0, 0
    .quad .Lfl_push_start
    .quad .Lfl_push_post - .Lfl_push_start
    .quad .Lfl_push_abort
.popsection

.text
.globl rseq_freelist_push
.p2align 4
rseq_freelist_push:
    // rdi = heads, rsi = nodes, rdx = area, rcx = node
    lea rax, [rip + .Lfl_push_cs]
.Lfl_push_retry:
    mov qword ptr [rdx + 8], rax        // arm rseq_cs
.Lfl_push_start:
    mov r8d, dword ptr [rdx + 4]        // cpu
    mov r9, qword ptr [rdi + r8*8]      // head = heads[cpu]
    mov qword ptr [rsi + rcx*8], r9     // nodes[node] = head (scratch)
    mov qword ptr [rdi + r8*8], rcx     // commit: heads[cpu] = node
.Lfl_push_post:
    ret
    .long 0x53053053
.Lfl_push_abort:
    jmp .Lfl_push_retry
"#
);

global_asm!(
    r#"
.pushsection .data.rseq_cs, "aw"
.balign 32
.Lfl_pop_cs:
    .long 0, 0
    .quad .Lfl_pop_start
    .quad .Lfl_pop_post - .Lfl_pop_start
    .quad .Lfl_pop_abort
.popsection

.text
.globl rseq_freelist_pop
.p2align 4
rseq_freelist_pop:
    // rdi = heads, rsi = nodes, rdx = area; returns node or NIL in rax
    lea r10, [rip + .Lfl_pop_cs]
.Lfl_pop_retry:
    mov qword ptr [rdx + 8], r10        // arm rseq_cs
.Lfl_pop_start:
    mov r8d, dword ptr [rdx + 4]        // cpu
    mov rax, qword ptr [rdi + r8*8]     // head = heads[cpu]
    cmp rax, -1
    je .Lfl_pop_out                     // empty: exit without committing
    mov r9, qword ptr [rsi + rax*8]     // next = nodes[head]
    mov qword ptr [rdi + r8*8], r9      // commit: heads[cpu] = next
.Lfl_pop_post:
.Lfl_pop_out:
    ret
    .long 0x53053053
.Lfl_pop_abort:
    jmp .Lfl_pop_retry
"#
);

/// A per-CPU array of u64 counters, incremented restartably.
pub struct PerCpuCounter {
    slots: Box<[UnsafeCell<u64>]>,
}

// Safety: cross-thread mutation happens only inside the rseq sequences,
// whose per-CPU commit discipline is the whole point.
unsafe impl Sync for PerCpuCounter {}

impl PerCpuCounter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: (0..MAX_CPUS).map(|_| UnsafeCell::new(0)).collect(),
        }
    }

    /// Increment the current CPU's counter. Aborted-and-retried attempts add
    /// to `aborts`. Returns false if rseq is unavailable on this thread.
    pub fn inc(&self, aborts: &mut u64) -> bool {
        let Some(area) = current_area() else {
            return false;
        };
        unsafe {
            rseq_counter_inc(self.slots.as_ptr() as *mut u64, area, aborts);
        }
        true
    }

    /// Total across CPUs. `&mut self` guarantees no concurrent increments.
    pub fn sum(&mut self) -> u64 {
        self.slots.iter_mut().map(|c| *c.get_mut()).sum()
    }
}

impl Default for PerCpuCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-CPU freelists over a shared node pool, as in `progs::freelist`:
/// `heads[cpu]` is the published head, `nodes[i]` is node i's next pointer.
pub struct PerCpuFreelist {
    heads: Box<[UnsafeCell<u64>]>,
    nodes: Box<[UnsafeCell<u64>]>,
}

unsafe impl Sync for PerCpuFreelist {}

impl PerCpuFreelist {
    #[must_use]
    pub fn new(nnodes: usize) -> Self {
        Self {
            heads: (0..MAX_CPUS).map(|_| UnsafeCell::new(NIL)).collect(),
            nodes: (0..nnodes).map(|_| UnsafeCell::new(NIL)).collect(),
        }
    }

    /// Push `node` onto the current CPU's list.
    ///
    /// # Safety
    ///
    /// The caller must own `node` exclusively (popped earlier, or never yet
    /// pushed, and not concurrently pushed by anyone else) — that ownership
    /// is what makes the scratch write to `nodes[node]` unobservable. Two
    /// threads pushing the same node race on its next pointer.
    ///
    /// # Panics
    ///
    /// Panics if `node` is out of range for the pool.
    #[must_use]
    pub unsafe fn push(&self, node: u64) -> bool {
        assert!((node as usize) < self.nodes.len());
        let Some(area) = current_area() else {
            return false;
        };
        unsafe {
            rseq_freelist_push(
                self.heads.as_ptr() as *mut u64,
                self.nodes.as_ptr() as *mut u64,
                area,
                node,
            );
        }
        true
    }

    /// Pop from the current CPU's list; None if it is empty (other CPUs'
    /// lists are not touched — cross-CPU draining needs the membarrier
    /// fence protocol, a later milestone).
    #[must_use]
    pub fn pop(&self) -> Option<u64> {
        let area = current_area()?;
        let node = unsafe {
            rseq_freelist_pop(
                self.heads.as_ptr() as *mut u64,
                self.nodes.as_ptr() as *mut u64,
                area,
            )
        };
        (node != NIL).then_some(node)
    }

    /// Walk every CPU's list. `&mut self` guarantees quiescence.
    ///
    /// # Panics
    ///
    /// Panics if a list contains a cycle (which would mean a broken commit).
    pub fn drain_all(&mut self) -> Vec<Vec<u64>> {
        let nnodes = self.nodes.len();
        (0..MAX_CPUS)
            .map(|c| {
                let mut out = Vec::new();
                let mut cur = *self.heads[c].get_mut();
                let mut fuel = nnodes + 1;
                while cur != NIL {
                    assert!(fuel > 0, "cycle in freelist of cpu {c}");
                    fuel -= 1;
                    out.push(cur);
                    cur = *self.nodes[cur as usize].get_mut();
                }
                out
            })
            .collect()
    }
}
