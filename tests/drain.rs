//! Live-kernel tests for the cross-CPU drain protocol.
//!
//! The stress test makes the membarrier fence load-bearing: workers pop and
//! push while a drainer continuously locks CPUs, takes their lists, and
//! pushes the stolen nodes back from wherever it runs. If the fence did not
//! restart in-flight sequences that read the pre-lock state, a worker's
//! commit could resurrect a drained head and lose or duplicate nodes — the
//! conservation law at the end would break.

#![cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]

use std::collections::BTreeSet;
use std::thread;

use rseq::drain::{DrainError, DrainableFreelist};
use rseq::rt;

fn skip(e: &DrainError) -> bool {
    matches!(e, DrainError::MembarrierUnsupported(_))
}

#[test]
fn drain_moves_all_nodes() {
    const NNODES: u64 = 64;
    if rt::current_area().is_none() {
        eprintln!("rseq unavailable; skipping");
        return;
    }
    let mut fl = match DrainableFreelist::new(NNODES as usize) {
        Ok(fl) => fl,
        Err(e) if skip(&e) => {
            eprintln!("rseq membarrier unavailable; skipping");
            return;
        }
        Err(e) => panic!("{e:?}"),
    };
    for node in 0..NNODES {
        fl.push(node);
    }
    // Drain every CPU; the union must be exactly the pushed nodes, and all
    // lists must be empty afterwards.
    let mut got = BTreeSet::new();
    for cpu in 0..64 {
        for node in fl.drain(cpu) {
            assert!(got.insert(node), "node {node} drained twice");
        }
    }
    assert_eq!(got, (0..NNODES).collect::<BTreeSet<_>>());
    assert!(fl.snapshot_all().iter().all(Vec::is_empty));
}

#[test]
fn drain_stress_conserves_nodes() {
    const NNODES: u64 = 512;
    const ITERS: usize = 60_000;
    const DRAIN_ROUNDS: usize = 400;
    if rt::current_area().is_none() {
        eprintln!("rseq unavailable; skipping");
        return;
    }
    let mut fl = match DrainableFreelist::new(NNODES as usize) {
        Ok(fl) => fl,
        Err(e) if skip(&e) => {
            eprintln!("rseq membarrier unavailable; skipping");
            return;
        }
        Err(e) => panic!("{e:?}"),
    };
    for node in 0..NNODES {
        fl.push(node);
    }
    let nworkers = 2 * thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4);
    thread::scope(|s| {
        for _ in 0..nworkers {
            s.spawn(|| {
                for i in 0..ITERS {
                    if let Some(node) = fl.pop() {
                        fl.push(node);
                    }
                    if i % 512 == 0 {
                        thread::yield_now();
                    }
                }
            });
        }
        // The drainer: sweep the CPUs, stealing every list and pushing the
        // stolen nodes back from wherever we are running now. Each stolen
        // node is exclusively ours between the drain and the push.
        s.spawn(|| {
            for round in 0..DRAIN_ROUNDS {
                let cpu = round % 64;
                for node in fl.drain(cpu) {
                    fl.push(node);
                }
            }
        });
    });
    // Conservation: every node on exactly one list, none lost or doubled.
    let lists = fl.snapshot_all();
    let mut seen = BTreeSet::new();
    for node in lists.iter().flatten() {
        assert!(seen.insert(*node), "node {node} appears on two lists");
    }
    assert_eq!(
        seen,
        (0..NNODES).collect::<BTreeSet<_>>(),
        "nodes lost under drain traffic"
    );
}
