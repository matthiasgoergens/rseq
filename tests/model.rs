//! Model-checker integration tests: correct programs survive exhaustive abort
//! injection, and the classic bug classes are caught.

use rseq::check::{self, CheckConfig, FailureKind};
use rseq::ir::{Addr, Op, Operand, Program, ValidateError, imm, reg};
use rseq::progs::{self, Freelist, NIL};
use rseq::sim::{Memory, SimError};

fn no_setup(_: &mut Memory) {}

#[test]
fn counter_inc_survives_all_abort_schedules() {
    let (layout, prog, counters) = progs::counter_inc();
    let stats = check::check(
        &prog,
        &layout,
        no_setup,
        |mem| mem.region(counters).to_vec(),
        &[],
        CheckConfig::default(),
    )
    .expect("counter_inc must check");
    assert!(
        stats.schedules > 100,
        "expected exhaustive enumeration, got {stats:?}"
    );
}

#[test]
fn strided_counter_survives_all_abort_schedules() {
    let (layout, prog, counters) = progs::counter_inc_strided(8);
    check::check(
        &prog,
        &layout,
        no_setup,
        |mem| mem.region(counters).to_vec(),
        &[],
        CheckConfig::default(),
    )
    .expect("counter_inc_strided must check");
}

#[test]
fn hoisted_cpu_id_is_caught_as_foreign_commit() {
    let (layout, prog, counters) = progs::counter_inc_hoisted();
    let failure = check::check(
        &prog,
        &layout,
        no_setup,
        |mem| mem.region(counters).to_vec(),
        &[],
        CheckConfig::default(),
    )
    .expect_err("the hoisted CPU id must be caught");
    assert!(
        matches!(
            failure.kind,
            FailureKind::Sim(SimError::ForeignCommit { .. })
        ),
        "expected ForeignCommit, got {failure:?}"
    );
    assert!(
        !failure.plan.is_empty(),
        "only a migration can expose the bug: {failure:?}"
    );
}

#[test]
fn post_commit_abort_window_is_caught() {
    // Model a descriptor whose window wrongly includes the committing store:
    // an abort delivered after the commit restarts and double-increments.
    let (layout, prog, counters) = progs::counter_inc();
    let cfg = CheckConfig {
        sim: rseq::sim::SimConfig {
            post_commit_in_window: true,
        },
        ..CheckConfig::default()
    };
    let failure = check::check(
        &prog,
        &layout,
        no_setup,
        |mem| mem.region(counters).to_vec(),
        &[],
        cfg,
    )
    .expect_err("a post-commit restart must be observable");
    assert!(
        matches!(failure.kind, FailureKind::ObservableMismatch { .. }),
        "expected ObservableMismatch, got {failure:?}"
    );
}

fn lists(mem: &Memory, fl: &Freelist) -> Vec<Vec<u64>> {
    (0..mem.ncpus())
        .map(|c| {
            let mut out = Vec::new();
            let mut cur = mem.region(fl.heads)[c];
            let mut fuel = fl.nnodes + 1;
            while cur != NIL && fuel > 0 {
                out.push(cur);
                cur = mem.region(fl.nodes)[cur as usize];
                fuel -= 1;
            }
            out
        })
        .collect()
}

fn populate(fl: &Freelist) -> impl Fn(&mut Memory) + '_ {
    // CPU 0: [0], CPU 1: empty, CPU 2: [1, 2].
    |mem: &mut Memory| {
        mem.region_mut(fl.heads)[0] = 0;
        mem.region_mut(fl.nodes)[0] = NIL;
        mem.region_mut(fl.heads)[2] = 1;
        mem.region_mut(fl.nodes)[1] = 2;
        mem.region_mut(fl.nodes)[2] = NIL;
    }
}

#[test]
fn freelist_push_survives_all_abort_schedules() {
    let fl = progs::freelist(8);
    let prog = fl.push();
    check::check(
        &prog,
        &fl.layout,
        populate(&fl),
        |mem| lists(mem, &fl),
        &[5],
        CheckConfig::default(),
    )
    .expect("freelist push must check");
}

#[test]
fn freelist_pop_survives_all_abort_schedules() {
    let fl = progs::freelist(8);
    let prog = fl.pop();
    check::check(
        &prog,
        &fl.layout,
        populate(&fl),
        |mem| lists(mem, &fl),
        &[],
        CheckConfig::default(),
    )
    .expect("freelist pop must check");
}

#[test]
fn freelist_pop_from_empty_exits_consistently() {
    let fl = progs::freelist(8);
    let prog = fl.pop();
    check::check(
        &prog,
        &fl.layout,
        no_setup,
        |mem| lists(mem, &fl),
        &[],
        CheckConfig::default(),
    )
    .expect("pop from empty lists must check");
}

#[test]
fn array_push_survives_all_abort_schedules() {
    let pa = progs::push_array(4);
    let prog = pa.push(42);
    let observe = |mem: &Memory| -> Vec<Vec<u64>> {
        // Only slots below the committed index are observable; scribbles from
        // aborted attempts beyond it are private scratch.
        (0..mem.ncpus())
            .map(|c| {
                let committed = mem.region(pa.committed)[c] as usize;
                mem.region(pa.slots)[c * pa.cap..c * pa.cap + committed].to_vec()
            })
            .collect()
    };
    check::check(
        &prog,
        &pa.layout,
        no_setup,
        observe,
        &[],
        CheckConfig::default(),
    )
    .expect("array push must check");
}

#[test]
fn array_push_full_exits_early() {
    let pa = progs::push_array(2);
    let prog = pa.push(42);
    let setup = |mem: &mut Memory| {
        for c in 0..mem.ncpus() {
            mem.region_mut(pa.committed)[c] = 2;
        }
    };
    let observe = |mem: &Memory| mem.region(pa.committed).to_vec();
    check::check(
        &prog,
        &pa.layout,
        setup,
        observe,
        &[],
        CheckConfig::default(),
    )
    .expect("push to full arrays must exit early consistently");
}

#[test]
fn observing_raw_scratch_is_too_strict() {
    // Deliberate demonstration: aborted attempts legitimately scribble on
    // scratch, so an observable function that exposes raw scratch reports a
    // mismatch. The abstraction function is part of the correctness claim.
    let pa = progs::push_array(4);
    let prog = pa.push(42);
    let failure = check::check(
        &prog,
        &pa.layout,
        no_setup,
        |mem| {
            (
                mem.region(pa.committed).to_vec(),
                mem.region(pa.slots).to_vec(),
            )
        },
        &[],
        CheckConfig::default(),
    )
    .expect_err("raw scratch differs between aborted and clean runs");
    assert!(
        matches!(failure.kind, FailureKind::ObservableMismatch { .. }),
        "{failure:?}"
    );
}

#[test]
fn validation_rejects_malformed_programs() {
    let (layout, good, counters) = progs::counter_inc();
    assert!(good.validate(&layout).is_ok());

    let commit = |src: u64| Op::Commit {
        addr: Addr {
            region: counters,
            index: Operand::Imm(0),
        },
        src: Operand::Imm(src),
    };

    let empty = Program {
        name: "empty".into(),
        ops: vec![],
        ret: None,
    };
    assert_eq!(empty.validate(&layout), Err(ValidateError::Empty));

    let two_commits = Program {
        name: "two_commits".into(),
        ops: vec![commit(1), commit(2)],
        ret: None,
    };
    assert_eq!(
        two_commits.validate(&layout),
        Err(ValidateError::CommitNotLast { at: 0 })
    );

    let no_commit = Program {
        name: "no_commit".into(),
        ops: vec![Op::CpuId { dst: 0 }],
        ret: None,
    };
    assert_eq!(
        no_commit.validate(&layout),
        Err(ValidateError::MissingCommit)
    );

    let use_before_def = Program {
        name: "use_before_def".into(),
        ops: vec![
            Op::Load {
                dst: 0,
                addr: Addr {
                    region: counters,
                    index: Operand::Reg(7),
                },
            },
            commit(1),
        ],
        ret: None,
    };
    assert_eq!(
        use_before_def.validate(&layout),
        Err(ValidateError::UseBeforeDef { at: 0, reg: 7 })
    );

    let scratch_to_published = Program {
        name: "scratch_to_published".into(),
        ops: vec![
            Op::StoreScratch {
                addr: Addr {
                    region: counters,
                    index: Operand::Imm(0),
                },
                src: Operand::Imm(9),
            },
            commit(1),
        ],
        ret: None,
    };
    assert_eq!(
        scratch_to_published.validate(&layout),
        Err(ValidateError::ScratchStoreToPublished {
            at: 0,
            region: counters
        })
    );

    let fl = progs::freelist(2);
    let commit_to_scratch = Program {
        name: "commit_to_scratch".into(),
        ops: vec![Op::Commit {
            addr: Addr {
                region: fl.nodes,
                index: Operand::Imm(0),
            },
            src: Operand::Imm(1),
        }],
        ret: None,
    };
    assert_eq!(
        commit_to_scratch.validate(&fl.layout),
        Err(ValidateError::CommitToScratch { region: fl.nodes })
    );

    // Multiple exit branches are legal: each leaves the window uncommitted
    // (the lockable freelist pop needs two — locked and empty).
    let mut b = rseq::ir::SeqBuilder::new("two_exits");
    let cpu = b.cpu_id();
    b.exit_if(rseq::ir::Cond::Eq, reg(cpu), imm(9));
    b.exit_if(rseq::ir::Cond::Eq, reg(cpu), imm(8));
    let two_exits = b.commit(counters, reg(cpu), imm(1));
    assert_eq!(two_exits.validate(&layout), Ok(()));
}

fn llists(mem: &Memory, fl: &progs::LockableFreelist) -> Vec<Vec<u64>> {
    (0..mem.ncpus())
        .map(|c| {
            let mut out = Vec::new();
            let mut cur = mem.region(fl.heads)[c];
            let mut fuel = fl.nnodes + 1;
            while cur != NIL && fuel > 0 {
                out.push(cur);
                cur = mem.region(fl.nodes)[cur as usize];
                fuel -= 1;
            }
            out
        })
        .collect()
}

#[test]
fn lockable_freelist_unlocked_behaves_normally() {
    let fl = progs::lockable_freelist(8);
    for prog in [fl.push(), fl.pop()] {
        check::check(
            &prog,
            &fl.layout,
            |mem: &mut Memory| {
                mem.region_mut(fl.heads)[0] = 0;
                mem.region_mut(fl.nodes)[0] = NIL;
            },
            |mem| llists(mem, &fl),
            &[5],
            CheckConfig::default(),
        )
        .unwrap_or_else(|e| panic!("{} must check unlocked: {e:?}", prog.name));
    }
}

#[test]
fn lockable_freelist_locked_cpu_never_commits() {
    let fl = progs::lockable_freelist(8);
    // CPU 1 is locked by a drainer; sequences ending there must exit
    // without committing, everywhere else business as usual. The checker's
    // per-final-CPU reference handles the mixed case automatically.
    let setup = |mem: &mut Memory| {
        mem.region_mut(fl.locks)[1] = 1;
        mem.region_mut(fl.heads)[0] = 0;
        mem.region_mut(fl.nodes)[0] = NIL;
        mem.region_mut(fl.heads)[1] = 1;
        mem.region_mut(fl.nodes)[1] = NIL;
    };
    for prog in [fl.push(), fl.pop()] {
        check::check(
            &prog,
            &fl.layout,
            setup,
            |mem| (llists(mem, &fl), mem.region(fl.locks).to_vec()),
            &[5],
            CheckConfig::default(),
        )
        .unwrap_or_else(|e| panic!("{} must check with cpu 1 locked: {e:?}", prog.name));
    }
}

#[test]
fn deeper_schedules_still_pass_for_counter() {
    let (layout, prog, counters) = progs::counter_inc();
    let cfg = CheckConfig {
        max_aborts: 3,
        ..CheckConfig::default()
    };
    let stats = check::check(
        &prog,
        &layout,
        no_setup,
        |mem| mem.region(counters).to_vec(),
        &[],
        cfg,
    )
    .expect("counter_inc must check at depth 3");
    assert!(stats.schedules > 1000, "{stats:?}");
}
