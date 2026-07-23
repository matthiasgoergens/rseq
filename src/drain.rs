//! Cross-CPU draining: the tcmalloc `FenceCpu` protocol on top of the
//! lockable freelist.
//!
//! The per-CPU ownership discipline means only the thread currently on CPU
//! `c` may commit to `heads[c]` — so a drainer cannot simply CAS a remote
//! head against plain rseq stores. Instead every sequence loads `locks[cpu]`
//! inside its window and exits early if set. The drainer:
//!
//! 1. takes `locks[c]` with an atomic compare-exchange (drainer vs drainer),
//! 2. fires `MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ` — on return, any
//!    sequence that read `locks[c] == 0` before the store has been
//!    restarted, and its retry re-reads the lock and exits,
//! 3. now owns `heads[c]` and the nodes on its list outright: reads and
//!    rewrites them with plain accesses, then
//! 4. releases the lock.
//!
//! Without step 2 an in-flight sequence could commit a head based on
//! pre-drain state and lose or duplicate nodes — the conservation stress
//! test in `tests/drain.rs` is built to catch exactly that.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::codegen::{CompileError, CompiledSeq, EXITED, RegionSet};
use crate::progs::{self, LockableFreelist, NIL};
use crate::rt::{self, MAX_CPUS};
use crate::sys;

/// A per-CPU freelist supporting cross-CPU draining.
pub struct DrainableFreelist {
    rs: RegionSet,
    push: CompiledSeq,
    pop: CompiledSeq,
    fl: LockableFreelist,
}

#[derive(Debug)]
pub enum DrainError {
    Compile(CompileError),
    /// The kernel refused `MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ`.
    MembarrierUnsupported(isize),
}

impl DrainableFreelist {
    /// # Errors
    ///
    /// Fails if the sequences do not compile or the rseq membarrier is
    /// unavailable on this kernel.
    pub fn new(nnodes: usize) -> Result<Self, DrainError> {
        let ret = sys::membarrier(sys::MEMBARRIER_REGISTER_RSEQ);
        if ret < 0 {
            return Err(DrainError::MembarrierUnsupported(ret));
        }
        let fl = progs::lockable_freelist(nnodes);
        let push = CompiledSeq::compile(&fl.push(), &fl.layout).map_err(DrainError::Compile)?;
        let pop = CompiledSeq::compile(&fl.pop(), &fl.layout).map_err(DrainError::Compile)?;
        let rs = RegionSet::new(&fl.layout);
        Ok(Self { rs, push, pop, fl })
    }

    /// The lock word of `cpu`, viewed atomically. The rseq sequences read
    /// it with plain loads inside their windows; the drainer side uses
    /// atomics so drainers exclude each other.
    fn lock(&self, cpu: usize) -> &AtomicU64 {
        assert!(cpu < MAX_CPUS);
        let base = self.rs.region_base(self.fl.locks) as *const AtomicU64;
        unsafe { &*base.add(cpu) }
    }

    /// Push `node` onto the current CPU's list, retrying while the CPU is
    /// locked by a drainer.
    ///
    /// # Panics
    ///
    /// Panics if rseq is unavailable or `node` is out of range.
    pub fn push(&self, node: u64) {
        assert!((node as usize) < self.fl.nnodes, "node out of range");
        loop {
            // Safety: indices are the kernel CPU id and a validated node.
            let r = unsafe { self.rs.call(&self.push, &[node]) }.expect("rseq required");
            if r != EXITED {
                return;
            }
            // EXITED here means "locked" (push has no other exit): a drain
            // of our CPU is in flight and they are short.
            std::thread::yield_now();
        }
    }

    /// Pop from the current CPU's list; None means genuinely empty.
    /// Retries while the CPU is locked.
    ///
    /// # Panics
    ///
    /// Panics if rseq is unavailable.
    #[must_use]
    pub fn pop(&self) -> Option<u64> {
        loop {
            // Safety: indices are the kernel CPU id and list-internal node
            // ids that only ever entered via a bounds-checked push.
            let r = unsafe { self.rs.call(&self.pop, &[]) }.expect("rseq required");
            if r != EXITED {
                return Some(r);
            }
            // Locked or empty? Check the lock of the CPU we are now on; if
            // unlocked, the exit meant empty. (We may have migrated between
            // the call and this read — then the retry just goes round.)
            let area = rt::current_area().expect("rseq required");
            let cpu = unsafe { core::ptr::read_volatile(&raw const (*area).cpu_id) } as usize;
            if self.lock(cpu).load(Ordering::Acquire) == 0 {
                return None;
            }
            std::thread::yield_now();
        }
    }

    /// Take every node currently on `cpu`'s list. Returns the empty vec if
    /// another drainer holds that CPU's lock.
    ///
    /// # Panics
    ///
    /// Panics if the fence syscall fails or the list is corrupt (a cycle).
    #[must_use]
    pub fn drain(&self, cpu: usize) -> Vec<u64> {
        if self
            .lock(cpu)
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Vec::new();
        }
        // The fence: after this returns, no in-flight sequence that read
        // locks[cpu] == 0 can still commit — the kernel restarted it.
        let ret = sys::membarrier(sys::MEMBARRIER_FENCE_RSEQ);
        assert!(ret >= 0, "membarrier fence failed: {ret}");

        // We own heads[cpu] and its chain now.
        let heads = self.rs.region_base(self.fl.heads) as *mut u64;
        let nodes = self.rs.region_base(self.fl.nodes) as *mut u64;
        let mut out = Vec::new();
        unsafe {
            let mut cur = heads.add(cpu).read_volatile();
            let mut fuel = self.fl.nnodes + 1;
            while cur != NIL {
                assert!(fuel > 0, "cycle in freelist of cpu {cpu}");
                fuel -= 1;
                out.push(cur);
                cur = nodes.add(cur as usize).read_volatile();
            }
            heads.add(cpu).write_volatile(NIL);
        }
        self.lock(cpu).store(0, Ordering::Release);
        out
    }

    /// Walk every CPU's list. `&mut self` guarantees quiescence.
    ///
    /// # Panics
    ///
    /// Panics on a corrupt list.
    pub fn snapshot_all(&mut self) -> Vec<Vec<u64>> {
        let heads_region = self.fl.heads;
        let nodes_region = self.fl.nodes;
        let nnodes = self.fl.nnodes;
        let heads = self.rs.region_mut(heads_region).to_vec();
        let nexts = self.rs.region_mut(nodes_region).to_vec();
        heads
            .iter()
            .map(|&head| {
                let mut out = Vec::new();
                let mut cur = head;
                let mut fuel = nnodes + 1;
                while cur != NIL {
                    assert!(fuel > 0, "cycle in freelist");
                    fuel -= 1;
                    out.push(cur);
                    cur = nexts[cur as usize];
                }
                out
            })
            .collect()
    }
}
