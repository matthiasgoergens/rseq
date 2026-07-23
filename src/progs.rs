//! Example programs: the classic per-CPU data structures rseq exists for,
//! plus deliberately buggy variants that the checker must reject.

use crate::ir::{BinOp, Cond, Layout, Program, RegionId, SeqBuilder, imm, reg};

/// Sentinel for "no node" in the freelist.
pub const NIL: u64 = u64::MAX;

/// Per-CPU counter increment: the "hello world" of restartable sequences.
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

/// The hoisting bug: the CPU id is read once before the sequence and cached,
/// so after a migration the commit targets the *old* CPU's counter.
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
pub fn freelist(nnodes: usize) -> Freelist {
    let mut layout = Layout::new();
    let heads = layout.published_per_cpu("heads", 1, NIL);
    let nodes = layout.scratch_shared("nodes", nnodes, NIL);
    Freelist { layout, heads, nodes, nnodes }
}

impl Freelist {
    /// Push `node` onto the current CPU's freelist.
    pub fn push(&self, node: u64) -> Program {
        let mut b = SeqBuilder::new("freelist_push");
        let cpu = b.cpu_id();
        let head = b.load(self.heads, reg(cpu));
        b.store_scratch(self.nodes, imm(node), reg(head));
        b.commit(self.heads, reg(cpu), imm(node))
    }

    /// Pop from the current CPU's freelist; exits early if empty, otherwise
    /// returns the popped node.
    pub fn pop(&self) -> Program {
        let mut b = SeqBuilder::new("freelist_pop");
        let cpu = b.cpu_id();
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
pub fn push_array(cap: usize) -> PushArray {
    let mut layout = Layout::new();
    let committed = layout.published_per_cpu("committed", 1, 0);
    let slots = layout.scratch_per_cpu("slots", cap, 0);
    PushArray { layout, committed, slots, cap }
}

impl PushArray {
    /// Push `value` onto the current CPU's array; exits early if full.
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
