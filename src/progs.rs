//! Example programs: the classic per-CPU data structures rseq exists for,
//! plus deliberately buggy variants that the checker must reject.

use crate::ir::{BinOp, Cond, Layout, Program, RegionId, SeqBuilder, imm, reg};

/// Sentinel for "no node" in the freelist.
pub const NIL: u64 = u64::MAX;

/// Per-CPU counter increment: the "hello world" of restartable sequences.
#[must_use]
pub fn counter_inc() -> (Layout, Program, RegionId) {
    let mut layout = Layout::new();
    let counters = layout.published_per_cpu("counters", 1, 0);
    let mut b = SeqBuilder::new("counter_inc");
    let cpu = b.cpu_id();
    let v = b.load(counters, reg(cpu));
    let inc = b.bin(BinOp::Add, reg(v), imm(1));
    let prog = b.commit(counters, reg(cpu), reg(inc));
    (layout, prog, counters)
}

/// Per-CPU counter increment with `words_per_cpu` stride, so each CPU's
/// counter can occupy its own cache line (stride 8 words = 64 bytes). The
/// benchmark uses this; the unit-stride version above false-shares.
#[must_use]
pub fn counter_inc_strided(words_per_cpu: usize) -> (Layout, Program, RegionId) {
    let mut layout = Layout::new();
    let counters = layout.published_per_cpu("counters", words_per_cpu, 0);
    let mut b = SeqBuilder::new("counter_inc_strided");
    let cpu = b.cpu_id();
    let slot = b.bin(BinOp::Mul, reg(cpu), imm(words_per_cpu as u64));
    let v = b.load(counters, reg(slot));
    let inc = b.bin(BinOp::Add, reg(v), imm(1));
    let prog = b.commit(counters, reg(slot), reg(inc));
    (layout, prog, counters)
}

/// The hoisting bug: the CPU id is read once before the sequence and cached,
/// so after a migration the commit targets the *old* CPU's counter.
#[must_use]
pub fn counter_inc_hoisted() -> (Layout, Program, RegionId) {
    let mut layout = Layout::new();
    let counters = layout.published_per_cpu("counters", 1, 0);
    let mut b = SeqBuilder::new("counter_inc_hoisted");
    let cpu = b.cpu_id_hoisted();
    let v = b.load(counters, reg(cpu));
    let inc = b.bin(BinOp::Add, reg(v), imm(1));
    let prog = b.commit(counters, reg(cpu), reg(inc));
    (layout, prog, counters)
}

pub struct Freelist {
    pub layout: Layout,
    pub heads: RegionId,
    pub nodes: RegionId,
    pub nnodes: usize,
}

/// Per-CPU freelist over a shared node pool. `heads[cpu]` is published;
/// `nodes[i]` holds node i's next pointer and is scratch (a node being
/// pushed is owned by the pusher until the head swings to it).
#[must_use]
#[allow(clippy::similar_names)]
pub fn freelist(nnodes: usize) -> Freelist {
    let mut layout = Layout::new();
    let heads = layout.published_per_cpu("heads", 1, NIL);
    let nodes = layout.scratch_shared("nodes", nnodes, NIL);
    Freelist {
        layout,
        heads,
        nodes,
        nnodes,
    }
}

impl Freelist {
    /// Push the node given as runtime parameter 0 onto the current CPU's
    /// freelist.
    #[must_use]
    pub fn push(&self) -> Program {
        let mut b = SeqBuilder::new("freelist_push");
        let cpu = b.cpu_id();
        let node = b.param(0);
        let head = b.load(self.heads, reg(cpu));
        b.store_scratch(self.nodes, reg(node), reg(head));
        b.commit(self.heads, reg(cpu), reg(node))
    }

    /// Pop from the current CPU's freelist; exits early if empty, otherwise
    /// returns the popped node.
    #[must_use]
    pub fn pop(&self) -> Program {
        let mut b = SeqBuilder::new("freelist_pop");
        let cpu = b.cpu_id();
        let head = b.load(self.heads, reg(cpu));
        b.exit_if(Cond::Eq, reg(head), imm(NIL));
        let next = b.load(self.nodes, reg(head));
        b.commit_ret(self.heads, reg(cpu), reg(next), reg(head))
    }
}

pub struct LockableFreelist {
    pub layout: Layout,
    pub locks: RegionId,
    pub heads: RegionId,
    pub nodes: RegionId,
    pub nnodes: usize,
}

/// A freelist whose sequences honour a per-CPU drain lock: every sequence
/// loads `locks[cpu]` inside the window and exits early if it is set. A
/// drainer that sets the lock and then fires the rseq membarrier fence
/// (restarting any sequence that could have read the old value) owns that
/// CPU's list outright until it clears the lock — the tcmalloc `FenceCpu`
/// protocol. The lock word is published: the sequences only read it; the
/// drainer writes it from outside.
#[must_use]
#[allow(clippy::similar_names)]
pub fn lockable_freelist(nnodes: usize) -> LockableFreelist {
    let mut layout = Layout::new();
    let locks = layout.published_per_cpu("locks", 1, 0);
    let heads = layout.published_per_cpu("heads", 1, NIL);
    let nodes = layout.scratch_shared("nodes", nnodes, NIL);
    LockableFreelist {
        layout,
        locks,
        heads,
        nodes,
        nnodes,
    }
}

impl LockableFreelist {
    /// Push runtime parameter 0; exits early (without committing) if the
    /// current CPU is locked by a drainer.
    #[must_use]
    pub fn push(&self) -> Program {
        let mut b = SeqBuilder::new("lockable_push");
        let cpu = b.cpu_id();
        let lk = b.load(self.locks, reg(cpu));
        b.exit_if(Cond::Ne, reg(lk), imm(0));
        let node = b.param(0);
        let head = b.load(self.heads, reg(cpu));
        b.store_scratch(self.nodes, reg(node), reg(head));
        b.commit(self.heads, reg(cpu), reg(node))
    }

    /// Pop from the current CPU's list; exits early if locked or empty.
    #[must_use]
    pub fn pop(&self) -> Program {
        let mut b = SeqBuilder::new("lockable_pop");
        let cpu = b.cpu_id();
        let lk = b.load(self.locks, reg(cpu));
        b.exit_if(Cond::Ne, reg(lk), imm(0));
        let head = b.load(self.heads, reg(cpu));
        b.exit_if(Cond::Eq, reg(head), imm(NIL));
        let next = b.load(self.nodes, reg(head));
        b.commit_ret(self.heads, reg(cpu), reg(next), reg(head))
    }
}

pub struct PushArray {
    pub layout: Layout,
    pub committed: RegionId,
    pub slots: RegionId,
    pub cap: usize,
}

/// The tcmalloc trick: scribble the new element into the uncommitted region
/// of a per-CPU array (scratch), then publish it by bumping the committed
/// index with the single committing store.
#[must_use]
pub fn push_array(cap: usize) -> PushArray {
    let mut layout = Layout::new();
    let committed = layout.published_per_cpu("committed", 1, 0);
    let slots = layout.scratch_per_cpu("slots", cap, 0);
    PushArray {
        layout,
        committed,
        slots,
        cap,
    }
}

impl PushArray {
    /// Push `value` onto the current CPU's array; exits early if full.
    #[must_use]
    pub fn push(&self, value: u64) -> Program {
        let mut b = SeqBuilder::new("array_push");
        let cpu = b.cpu_id();
        let idx = b.load(self.committed, reg(cpu));
        b.exit_if(Cond::Ge, reg(idx), imm(self.cap as u64));
        let base = b.bin(BinOp::Mul, reg(cpu), imm(self.cap as u64));
        let pos = b.bin(BinOp::Add, reg(base), reg(idx));
        b.store_scratch(self.slots, reg(pos), imm(value));
        let next = b.bin(BinOp::Add, reg(idx), imm(1));
        b.commit(self.committed, reg(cpu), reg(next))
    }
}
