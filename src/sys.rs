//! Minimal raw-syscall layer (x86-64 Linux only), keeping the crate free of
//! dependencies. Used by the JIT for its executable mapping and by the
//! ptrace conformance harness in `tests/ptrace.rs`.

// Harness support: every wrapper asserts syscall success by design, and the
// two mmap wrappers document their errno at the type.
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

use core::arch::asm;

pub const SIGTRAP: i32 = 5;
pub const SIGKILL: i32 = 9;
pub const SIGSTOP: i32 = 19;

/// # Safety
///
/// Raw syscall: the caller is responsible for the ABI contract of `nr`.
#[inline]
#[must_use] 
pub unsafe fn syscall6(nr: usize, args: [usize; 6]) -> isize {
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

const SYS_MMAP: usize = 9;
const SYS_MPROTECT: usize = 10;
const SYS_MUNMAP: usize = 11;
const SYS_GETPID: usize = 39;
const SYS_FORK: usize = 57;
const SYS_WAIT4: usize = 61;
const SYS_KILL: usize = 62;
const SYS_PTRACE: usize = 101;
const SYS_SCHED_SETAFFINITY: usize = 203;
const SYS_SCHED_GETAFFINITY: usize = 204;
const SYS_EXIT_GROUP: usize = 231;

const PROT_READ: usize = 1;
const PROT_WRITE: usize = 2;
const PROT_EXEC: usize = 4;
const MAP_PRIVATE_ANON: usize = 0x22;

/// # Safety
///
/// Returns fresh anonymous RW memory or a negative errno.
pub unsafe fn mmap_rw(len: usize) -> Result<*mut u8, isize> {
    let ret = unsafe {
        syscall6(SYS_MMAP, [0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE_ANON, usize::MAX, 0])
    };
    if ret < 0 { Err(ret) } else { Ok(ret as *mut u8) }
}

/// # Safety
///
/// `p..p+len` must be a mapping owned by the caller.
pub unsafe fn mprotect_rx(p: *mut u8, len: usize) -> Result<(), isize> {
    let ret =
        unsafe { syscall6(SYS_MPROTECT, [p as usize, len, PROT_READ | PROT_EXEC, 0, 0, 0]) };
    if ret < 0 { Err(ret) } else { Ok(()) }
}

/// # Safety
///
/// `p..p+len` must be a mapping owned by the caller, with no references into
/// it outliving the call.
pub unsafe fn munmap(p: *mut u8, len: usize) {
    let _ = unsafe { syscall6(SYS_MUNMAP, [p as usize, len, 0, 0, 0, 0]) };
}

/// # Safety
///
/// Real fork(2): the child returns 0 with a single thread; the caller must
/// keep the child away from any state other threads may have had locked.
#[must_use] 
pub unsafe fn fork() -> isize {
    unsafe { syscall6(SYS_FORK, [0; 6]) }
}

#[must_use]
pub fn getpid() -> i32 {
    unsafe { syscall6(SYS_GETPID, [0; 6]) as i32 }
}

/// # Safety
///
/// Sends `sig` to `pid`.
#[must_use] 
pub unsafe fn kill(pid: i32, sig: i32) -> isize {
    unsafe { syscall6(SYS_KILL, [pid as usize, sig as usize, 0, 0, 0, 0]) }
}

/// # Safety
///
/// Terminates the whole process immediately.
pub unsafe fn exit_group(code: i32) -> ! {
    unsafe {
        let _ = syscall6(SYS_EXIT_GROUP, [code as usize, 0, 0, 0, 0, 0]);
    }
    unreachable!("exit_group returned")
}

/// Block until `pid` changes state; returns the raw wait status.
///
/// # Safety
///
/// `pid` must be a child of the calling thread's process.
#[must_use] 
pub unsafe fn wait4(pid: i32) -> i32 {
    let mut status: i32 = 0;
    let ret = unsafe {
        syscall6(SYS_WAIT4, [pid as usize, (&raw mut status) as usize, 0, 0, 0, 0])
    };
    assert!(ret > 0, "wait4 failed: {ret}");
    status
}

/// Restrict `pid` to the CPUs in `mask` (bit per CPU, single word). A
/// stopped task is migrated immediately, which records an rseq migration
/// event for it.
///
/// # Safety
///
/// Changes scheduling of another process.
#[must_use] 
pub unsafe fn sched_setaffinity(pid: i32, mask: u64) -> isize {
    let mask = [mask];
    unsafe {
        syscall6(SYS_SCHED_SETAFFINITY, [pid as usize, 8, mask.as_ptr() as usize, 0, 0, 0])
    }
}

/// The calling thread's allowed-CPU mask (first 64 CPUs).
#[must_use]
pub fn sched_getaffinity_self() -> u64 {
    let mut mask = [0u64];
    let ret = unsafe {
        syscall6(SYS_SCHED_GETAFFINITY, [0, 8, mask.as_mut_ptr() as usize, 0, 0, 0])
    };
    assert!(ret > 0, "sched_getaffinity failed: {ret}");
    mask[0]
}

/// The signal that stopped the tracee, or None if it did not stop.
#[must_use]
pub fn stop_signal(status: i32) -> Option<i32> {
    if status & 0xff == 0x7f { Some((status >> 8) & 0xff) } else { None }
}

/// ptrace(2) operations, raw-syscall flavoured: PEEK requests write through
/// the `data` pointer instead of using libc's return-value convention.
pub mod ptrace {
    use super::{SYS_PTRACE, syscall6};

    const TRACEME: usize = 0;
    const PEEKTEXT: usize = 1;
    const POKETEXT: usize = 4;
    const CONT: usize = 7;
    const SINGLESTEP: usize = 9;
    const GETREGS: usize = 12;
    const SETREGS: usize = 13;
    const GET_RSEQ_CONFIGURATION: usize = 0x420f;

    /// Index of rip in `user_regs_struct` (x86-64).
    pub const RIP: usize = 16;
    /// Number of u64 slots in `user_regs_struct` (x86-64).
    pub const NREGS: usize = 27;

    /// `struct ptrace_rseq_configuration` from the kernel uapi.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct RseqConfiguration {
        pub rseq_abi_pointer: u64,
        pub rseq_abi_size: u32,
        pub signature: u32,
        pub flags: u32,
        pub pad: u32,
    }

    unsafe fn req(op: usize, pid: i32, addr: usize, data: usize) -> isize {
        unsafe { syscall6(SYS_PTRACE, [op, pid as usize, addr, data, 0, 0]) }
    }

    /// # Safety
    ///
    /// Makes the calling process traced by its parent.
    pub unsafe fn traceme() {
        let ret = unsafe { req(TRACEME, 0, 0, 0) };
        assert!(ret >= 0, "PTRACE_TRACEME failed: {ret}");
    }

    /// # Safety
    ///
    /// `pid` must be a stopped tracee; `addr` readable in it.
    #[must_use] 
    pub unsafe fn peek(pid: i32, addr: usize) -> u64 {
        let mut out: u64 = 0;
        let ret = unsafe { req(PEEKTEXT, pid, addr, (&raw mut out) as usize) };
        assert!(ret >= 0, "PTRACE_PEEKTEXT {addr:#x} failed: {ret}");
        out
    }

    /// # Safety
    ///
    /// `pid` must be a stopped tracee; writes the full word at `addr`.
    pub unsafe fn poke(pid: i32, addr: usize, word: u64) {
        let ret = unsafe { req(POKETEXT, pid, addr, word as usize) };
        assert!(ret >= 0, "PTRACE_POKETEXT {addr:#x} failed: {ret}");
    }

    /// # Safety
    ///
    /// Resumes a stopped tracee, delivering `sig` (0 = none).
    pub unsafe fn cont(pid: i32, sig: i32) {
        let ret = unsafe { req(CONT, pid, 0, sig as usize) };
        assert!(ret >= 0, "PTRACE_CONT failed: {ret}");
    }

    /// # Safety
    ///
    /// Resumes a stopped tracee for one instruction.
    pub unsafe fn singlestep(pid: i32) {
        let ret = unsafe { req(SINGLESTEP, pid, 0, 0) };
        assert!(ret >= 0, "PTRACE_SINGLESTEP failed: {ret}");
    }

    /// # Safety
    ///
    /// `pid` must be a stopped tracee.
    #[must_use] 
    pub unsafe fn getregs(pid: i32) -> [u64; NREGS] {
        let mut regs = [0u64; NREGS];
        let ret = unsafe { req(GETREGS, pid, 0, regs.as_mut_ptr() as usize) };
        assert!(ret >= 0, "PTRACE_GETREGS failed: {ret}");
        regs
    }

    /// # Safety
    ///
    /// `pid` must be a stopped tracee; `regs` must be a full register set.
    pub unsafe fn setregs(pid: i32, regs: &[u64; NREGS]) {
        let ret = unsafe { req(SETREGS, pid, 0, regs.as_ptr() as usize) };
        assert!(ret >= 0, "PTRACE_SETREGS failed: {ret}");
    }

    /// # Safety
    ///
    /// `pid` must be a stopped tracee. Kernel 5.13+.
    #[must_use] 
    pub unsafe fn rseq_configuration(pid: i32) -> RseqConfiguration {
        let mut cfg = RseqConfiguration::default();
        let ret = unsafe {
            req(
                GET_RSEQ_CONFIGURATION,
                pid,
                core::mem::size_of::<RseqConfiguration>(),
                (&raw mut cfg) as usize,
            )
        };
        assert!(ret >= 0, "PTRACE_GET_RSEQ_CONFIGURATION failed: {ret}");
        cfg
    }
}
