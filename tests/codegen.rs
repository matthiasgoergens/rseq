//! Tests for the codegen backend: the compiled bytes must pass the same
//! live-kernel stress harness as the hand-written sequences, and must agree
//! with the simulator on single-threaded runs.

#![cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]

use std::collections::BTreeSet;
use std::thread;

use rseq::codegen::{CompiledSeq, EXITED, RegionSet};
use rseq::ir::{BinOp, Layout, SeqBuilder, imm, reg};
use rseq::progs::{self, NIL};
use rseq::rt;
use rseq::sim::{self, Memory, Outcome, SimConfig};

fn threads() -> usize {
    2 * thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(4)
}

fn rseq_available() -> bool {
    if rt::current_area().is_none() {
        eprintln!("rseq unavailable on this system; skipping");
        return false;
    }
    true
}

/// Single-threaded equivalence against the simulator on a shared region:
/// same commits, same return value, no per-CPU indexing involved.
#[test]
fn compiled_matches_simulator_on_shared_region() {
    if !rseq_available() {
        return;
    }
    let mut layout = Layout::new();
    let cell = layout.published_shared("cell", 1, 10);
    let mut b = SeqBuilder::new("add7");
    let v = b.load(cell, imm(0));
    let w = b.bin(BinOp::Add, reg(v), imm(7));
    let prog = b.commit_ret(cell, imm(0), reg(w), reg(w));

    // Simulator reference.
    let mut mem = Memory::new(&layout, 1);
    let want = sim::run(&prog, &mut mem, &[], 0, &[], SimConfig::default()).unwrap();
    assert_eq!(want.outcome, Outcome::Committed { ret: Some(17) });

    // Real execution.
    let seq = CompiledSeq::compile(&prog, &layout).expect("compiles");
    let mut rs = RegionSet::new(&layout);
    let got = rs.call(&seq, &[]).expect("rseq available");
    assert_eq!(got, 17);
    assert_eq!(rs.region_mut(cell)[0], mem.region(cell)[0]);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn compiled_param_and_shifts_match_simulator() {
    if !rseq_available() {
        return;
    }
    let mut layout = Layout::new();
    let cell = layout.published_shared("cell", 4, 0);
    let mut b = SeqBuilder::new("mix");
    let p = b.param(0);
    let q = b.param(1);
    let x = b.bin(BinOp::Shl, reg(p), imm(4));
    let y = b.bin(BinOp::Xor, reg(x), reg(q));
    let z = b.bin(BinOp::Sub, reg(y), imm(1));
    let prog = b.commit_ret(cell, imm(2), reg(z), reg(z));

    let params = [0xABCD, 0xFF];
    let mut mem = Memory::new(&layout, 1);
    let want = sim::run(&prog, &mut mem, &params, 0, &[], SimConfig::default()).unwrap();

    let seq = CompiledSeq::compile(&prog, &layout).expect("compiles");
    let mut rs = RegionSet::new(&layout);
    let got = rs.call(&seq, &params).expect("rseq available");
    assert_eq!(Outcome::Committed { ret: Some(got) }, want.outcome);
    assert_eq!(rs.region_mut(cell)[2], mem.region(cell)[2]);
}

/// The model-checked counter program, compiled and stress-run on the real
/// kernel with the same conservation law as the hand-written version.
#[test]
fn compiled_counter_stress_conserves_increments() {
    const ITERS: u64 = 500_000;
    if !rseq_available() {
        return;
    }
    let (layout, prog, counters) = progs::counter_inc();
    let seq = CompiledSeq::compile(&prog, &layout).expect("compiles");
    let mut rs = RegionSet::new(&layout);
    let nthreads = threads();
    thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| {
                for _ in 0..ITERS {
                    rs.call(&seq, &[]).expect("rseq available");
                }
            });
        }
    });
    let total: u64 = rs.region_mut(counters).iter().sum();
    assert_eq!(total, nthreads as u64 * ITERS);
}

/// The model-checked freelist programs, compiled and stress-run: pop a node
/// wherever we run, push it back wherever we run then; conservation must
/// hold across all per-CPU lists.
#[test]
fn compiled_freelist_stress_conserves_nodes() {
    const NNODES: u64 = 512;
    const ITERS: usize = 100_000;
    if !rseq_available() {
        return;
    }
    let fl = progs::freelist(NNODES as usize);
    let push = CompiledSeq::compile(&fl.push(), &fl.layout).expect("push compiles");
    let pop = CompiledSeq::compile(&fl.pop(), &fl.layout).expect("pop compiles");
    let mut rs = RegionSet::new(&fl.layout);
    for node in 0..NNODES {
        rs.call(&push, &[node]).expect("rseq available");
    }
    let nthreads = threads();
    thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| {
                for i in 0..ITERS {
                    let node = rs.call(&pop, &[]).expect("rseq available");
                    if node != EXITED {
                        rs.call(&push, &[node]).expect("rseq available");
                    }
                    if i % 1024 == 0 {
                        thread::yield_now();
                    }
                }
            });
        }
    });
    // Conservation: walk every CPU's list; every node exactly once.
    let heads = rs.region_mut(fl.heads).to_vec();
    let nodes = rs.region_mut(fl.nodes).to_vec();
    let mut seen = BTreeSet::new();
    for (cpu, &head) in heads.iter().enumerate() {
        let mut cur = head;
        let mut fuel = NNODES + 1;
        while cur != NIL {
            assert!(fuel > 0, "cycle in freelist of cpu {cpu}");
            fuel -= 1;
            assert!(seen.insert(cur), "node {cur} on two lists");
            cur = nodes[cur as usize];
        }
    }
    assert_eq!(seen, (0..NNODES).collect::<BTreeSet<_>>(), "nodes lost");
}

/// The tcmalloc-style array push, compiled: number of successful commits
/// must equal the total number of elements published across CPUs, and every
/// published slot must hold the pushed value.
#[test]
fn compiled_array_push_publishes_consistently() {
    if !rseq_available() {
        return;
    }
    let pa = progs::push_array(4);
    let seq = CompiledSeq::compile(&pa.push(42), &pa.layout).expect("compiles");
    let mut rs = RegionSet::new(&pa.layout);
    let mut committed_calls = 0u64;
    for _ in 0..64 {
        let r = rs.call(&seq, &[]).expect("rseq available");
        if r != EXITED {
            committed_calls += 1;
        }
    }
    let committed = rs.region_mut(pa.committed).to_vec();
    let slots = rs.region_mut(pa.slots).to_vec();
    let published: u64 = committed.iter().sum();
    assert_eq!(published, committed_calls);
    for (cpu, &n) in committed.iter().enumerate() {
        for i in 0..n as usize {
            assert_eq!(slots[cpu * pa.cap + i], 42, "cpu {cpu} slot {i}");
        }
    }
}

/// The model-only hoisted-CPU-id op must be rejected, not compiled.
#[test]
fn hoisted_cpu_id_refuses_to_compile() {
    let (layout, prog, _) = progs::counter_inc_hoisted();
    let err = CompiledSeq::compile(&prog, &layout).expect_err("must not compile");
    assert_eq!(err, rseq::codegen::CompileError::HoistedCpuId);
}
