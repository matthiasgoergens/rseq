//! Deterministic abort-point conformance harness.
//!
//! A forked child loops on a compiled sequence under ptrace. For every
//! instruction boundary inside the critical window, the parent runs the
//! child into an int3 planted at that boundary, rewinds it, and then forces
//! a migration with `sched_setaffinity` while the child is stopped — a
//! guaranteed rseq event. Breakpoints on the two possible continuations
//! (abort handler vs post-commit) then observe what the kernel actually
//! did on resume:
//!
//! - stopped at any in-window boundary, the child must resume in the abort
//!   handler (the fixup redirected it), and no commit may be visible;
//! - stopped one boundary past the committing store, the child must NOT be
//!   redirected — it proceeds through the epilogue and the commit counts
//!   exactly once. This is precisely the `post_commit_offset` off-by-one a
//!   buggy descriptor would get wrong.
//!
//! This exercises the real contract end-to-end at every abort point: our
//! descriptor bytes and their placement, the kernel's window arithmetic,
//! the abort signature, index recomputation after migration, and the retry
//! path.

#![cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]

use rseq::codegen::{CompiledSeq, RegionSet};
use rseq::progs;
use rseq::rt;
use rseq::sys::{self, ptrace};

const RSEQ_SIG: u32 = 0x5305_3053;

/// Read the child's per-CPU counter total via ptrace.
unsafe fn counter_sum(pid: i32, base: u64, words: usize) -> u64 {
    (0..words)
        .map(|i| unsafe { ptrace::peek(pid, base as usize + i * 8) })
        .sum()
}

/// Plant int3 at `addr`, run the child into it, rewind to `addr`.
/// Returns with the child stopped at `addr`, breakpoint removed.
unsafe fn run_to(pid: i32, addr: usize) {
    unsafe {
        let orig = ptrace::peek(pid, addr);
        ptrace::poke(pid, addr, (orig & !0xff) | 0xCC);
        ptrace::cont(pid, 0);
        let status = sys::wait4(pid);
        assert_eq!(
            sys::stop_signal(status),
            Some(sys::SIGTRAP),
            "expected int3 trap at {addr:#x}, status {status:#x}"
        );
        let mut regs = ptrace::getregs(pid);
        assert_eq!(regs[ptrace::RIP] as usize, addr + 1, "trap rip");
        ptrace::poke(pid, addr, orig);
        regs[ptrace::RIP] = addr as u64;
        ptrace::setregs(pid, &regs);
    }
}

/// Resume the child with int3 planted at each of `arms`; report which one
/// it reached first. Returns with the child stopped at the hit address,
/// all breakpoints removed.
unsafe fn race_to(pid: i32, arms: &[usize]) -> usize {
    unsafe {
        let origs: Vec<u64> = arms.iter().map(|&a| ptrace::peek(pid, a)).collect();
        for (&a, &orig) in arms.iter().zip(&origs) {
            ptrace::poke(pid, a, (orig & !0xff) | 0xCC);
        }
        ptrace::cont(pid, 0);
        let status = sys::wait4(pid);
        assert_eq!(
            sys::stop_signal(status),
            Some(sys::SIGTRAP),
            "expected a trap at one of {arms:#x?}, status {status:#x}"
        );
        let mut regs = ptrace::getregs(pid);
        let hit = regs[ptrace::RIP] as usize - 1;
        assert!(
            arms.contains(&hit),
            "trapped at unexpected rip {:#x}",
            regs[ptrace::RIP]
        );
        for (&a, &orig) in arms.iter().zip(&origs) {
            ptrace::poke(pid, a, orig);
        }
        regs[ptrace::RIP] = hit as u64;
        ptrace::setregs(pid, &regs);
        hit
    }
}

/// Migrate the stopped child to a CPU other than the one it last ran on —
/// a guaranteed rseq migration event.
unsafe fn force_migration(pid: i32, area: usize, allowed: u64) {
    unsafe {
        let cpu_words = ptrace::peek(pid, area);
        let cpu_id = (cpu_words >> 32) as u32; // cpu_id sits at offset 4
        let other = allowed & !(1u64 << cpu_id);
        assert!(other != 0, "need a second allowed CPU");
        let ret = sys::sched_setaffinity(pid, other);
        assert!(ret >= 0, "sched_setaffinity failed: {ret}");
    }
}

#[test]
fn every_abort_point_redirects_and_post_commit_does_not() {
    if rt::current_area().is_none() {
        eprintln!("rseq unavailable on this system; skipping");
        return;
    }
    let allowed = sys::sched_getaffinity_self();
    if allowed.count_ones() < 2 {
        eprintln!("need at least two CPUs; skipping");
        return;
    }
    let (layout, prog, counters) = progs::counter_inc();
    let seq = CompiledSeq::compile(&prog, &layout).expect("compiles");
    let rs = RegionSet::new(&layout);
    let counters_base = rs.region_base(counters);
    let counters_len = rs.region_len(counters);
    let window = seq.window_insn_addrs();
    assert!(window.len() >= 5, "window suspiciously small: {window:#x?}");

    // The child's completed-call count lives at a known address (all
    // mappings are inherited at the same addresses across fork).
    let calls: Box<u64> = Box::new(0);
    let calls_addr = &raw const *calls as usize;

    let child = unsafe { sys::fork() };
    assert!(child >= 0, "fork failed: {child}");
    if child == 0 {
        // Child: become a tracee, then hammer the sequence forever. No
        // allocation and no locks past this point (fork from a threaded
        // test harness).
        unsafe {
            ptrace::traceme();
            let _ = sys::kill(sys::getpid(), sys::SIGSTOP);
            let calls_ptr = calls_addr as *mut u64;
            loop {
                let _ = rs.call(&seq, &[]);
                calls_ptr.write_volatile(calls_ptr.read_volatile() + 1);
            }
        }
    }
    let pid = child as i32;

    unsafe {
        let status = sys::wait4(pid);
        assert_eq!(sys::stop_signal(status), Some(sys::SIGSTOP), "initial stop");

        // Registration sanity from the outside (PTRACE_GET_RSEQ_CONFIGURATION):
        // the kernel agrees on the area address and the abort signature that
        // our emitted bytes carry.
        let cfg = ptrace::rseq_configuration(pid);
        assert_eq!(cfg.signature, RSEQ_SIG, "kernel-registered signature");
        let area = rt::current_area().unwrap() as usize;
        assert_eq!(
            cfg.rseq_abi_pointer as usize, area,
            "area inherited across fork"
        );

        // Positive case: a migration injected at EVERY window boundary
        // must redirect the child to the abort handler before any further
        // window instruction — in particular before the commit.
        for &addr in &window {
            run_to(pid, addr);
            // The kernel must see our armed descriptor at this stop —
            // this is what caught the poisoned-retry bug: an abort after
            // the first region-base load used to re-arm rseq_cs with a
            // region base instead of the descriptor.
            let armed = ptrace::peek(pid, area + 8);
            let descriptor = seq.code_bytes().as_ptr() as usize;
            assert_eq!(
                armed as usize, descriptor,
                "rseq_cs must point at our descriptor at {addr:#x}, got {armed:#x}"
            );
            let sum = counter_sum(pid, counters_base, counters_len);
            let calls_done = ptrace::peek(pid, calls_addr);
            assert_eq!(
                sum, calls_done,
                "no commit may be visible while stopped pre-commit at {addr:#x}"
            );
            force_migration(pid, area, allowed);
            let hit = race_to(pid, &[seq.abort_addr(), seq.post_addr()]);
            assert_eq!(
                hit,
                seq.abort_addr(),
                "abort point {addr:#x} was not redirected (reached post_commit)"
            );
            // Still nothing committed on the aborted attempt.
            let sum = counter_sum(pid, counters_base, counters_len);
            assert_eq!(sum, calls_done, "aborted attempt must not have committed");
            let ret = sys::sched_setaffinity(pid, allowed);
            assert!(ret >= 0, "restore affinity: {ret}");
        }

        // Negative case: the same injected migration one boundary past the
        // committing store must NOT redirect — the child sails on through
        // the epilogue into its next call (reaching the re-arm point via
        // the entry path, not the abort handler).
        run_to(pid, seq.post_addr());
        let sum = counter_sum(pid, counters_base, counters_len);
        let calls_done = ptrace::peek(pid, calls_addr);
        assert_eq!(sum, calls_done + 1, "commit must be visible at post_commit");
        force_migration(pid, area, allowed);
        let hit = race_to(pid, &[seq.abort_addr(), seq.retry_addr()]);
        assert_eq!(
            hit,
            seq.retry_addr(),
            "post-commit interruption must not redirect to the abort handler"
        );
        // By now the child has returned from the interrupted call and
        // tallied it, so the ledger must balance again: the commit counted
        // exactly once, not zero times and not twice.
        let sum = counter_sum(pid, counters_base, counters_len);
        let calls_now = ptrace::peek(pid, calls_addr);
        assert!(
            calls_now > calls_done,
            "child should have finished the call"
        );
        assert_eq!(sum, calls_now, "the commit counted exactly once");
        let ret = sys::sched_setaffinity(pid, allowed);
        assert!(ret >= 0, "restore affinity: {ret}");

        let _ = sys::kill(pid, sys::SIGKILL);
        let _ = sys::wait4(pid);
    }
    eprintln!(
        "verified {} abort points redirect, post_commit does not",
        window.len()
    );
}
