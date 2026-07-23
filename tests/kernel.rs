//! Live-kernel stress tests for the hand-written sequences.
//!
//! Oversubscription (2 threads per logical CPU) forces preemption, so the
//! abort/retry path gets exercised for real; correctness is asserted by
//! conservation laws that any lost or doubled commit would break.

#![cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]

use std::collections::BTreeSet;
use std::thread;

use rseq::rt::{self, PerCpuCounter, PerCpuFreelist};

fn threads() -> usize {
    2 * thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
}

#[test]
fn counter_stress_conserves_increments() {
    const ITERS: u64 = 1_000_000;
    if rt::current_area().is_none() {
        eprintln!("rseq unavailable on this system; skipping");
        return;
    }
    let nthreads = threads();
    let mut counter = PerCpuCounter::new();
    let total_aborts: u64 = thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|_| {
                s.spawn(|| {
                    let mut aborts = 0;
                    for _ in 0..ITERS {
                        assert!(counter.inc(&mut aborts));
                    }
                    aborts
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    assert_eq!(counter.sum(), nthreads as u64 * ITERS);
    // Not asserted (a lightly loaded box could get lucky), but the count is
    // the evidence that the abort path actually ran.
    eprintln!("{nthreads} threads x {ITERS} increments, {total_aborts} aborts");
}

#[test]
fn freelist_stress_conserves_nodes() {
    const NNODES: u64 = 512;
    const ITERS: usize = 200_000;
    if rt::current_area().is_none() {
        eprintln!("rseq unavailable on this system; skipping");
        return;
    }
    let nthreads = threads();
    let mut fl = PerCpuFreelist::new(NNODES as usize);
    // Seed: all nodes pushed from the main thread (whatever CPUs it lands on).
    for node in 0..NNODES {
        assert!(fl.push(node));
    }
    thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| {
                // Pop a node from wherever we run, push it back from wherever
                // we run then — migrations between the two shuffle nodes
                // across per-CPU lists, which is exactly the traffic pattern
                // that catches a broken commit.
                for i in 0..ITERS {
                    if let Some(node) = fl.pop() {
                        assert!(fl.push(node));
                    }
                    if i % 1024 == 0 {
                        thread::yield_now();
                    }
                }
            });
        }
    });
    // Conservation: every node on exactly one list, none lost, none doubled.
    let lists = fl.drain_all();
    let mut seen = BTreeSet::new();
    for node in lists.iter().flatten() {
        assert!(seen.insert(*node), "node {node} appears on two lists");
    }
    assert_eq!(
        seen,
        (0..NNODES).collect::<BTreeSet<_>>(),
        "nodes lost: {:?}",
        (0..NNODES)
            .filter(|n| !seen.contains(n))
            .collect::<Vec<_>>()
    );
}
