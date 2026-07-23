//! Exhaustive abort-schedule model checker.
//!
//! For a given program the checker enumerates every abort schedule up to a
//! bounded depth — every abort position within an attempt, crossed with every
//! possible destination CPU — and verifies that the run is indistinguishable
//! from the abort-free run:
//!
//! 1. **Prefix purity**: executing any prefix that excludes the commit leaves
//!    published memory untouched.
//! 2. **Restart equivalence**: for every abort schedule, the outcome and the
//!    user-supplied observable state equal those of a clean (abort-free) run
//!    on the CPU the schedule ends on. Scratch scribbles from aborted
//!    attempts must be masked by the observable function — that is the
//!    abstraction the tcmalloc trick relies on.
//! 3. **Ownership**: the simulator rejects commits to another CPU's slice,
//!    so schedules that migrate a stale CPU id surface as `ForeignCommit`.

use std::fmt::Debug;

use crate::ir::{Layout, Program, ValidateError};
use crate::sim::{self, AbortPoint, Memory, Outcome, SimConfig, SimError};

#[derive(Clone, Copy, Debug)]
pub struct CheckConfig {
    /// Number of simulated CPUs. 3 distinguishes "back where I started" from
    /// "somewhere new" after two migrations.
    pub ncpus: usize,
    /// Maximum number of aborts per schedule.
    pub max_aborts: usize,
    pub sim: SimConfig,
}

impl Default for CheckConfig {
    fn default() -> Self {
        Self { ncpus: 3, max_aborts: 2, sim: SimConfig::default() }
    }
}

#[derive(Clone, Debug)]
pub struct CheckFailure {
    pub start_cpu: usize,
    pub plan: Vec<AbortPoint>,
    pub kind: FailureKind,
}

#[derive(Clone, Debug)]
pub enum FailureKind {
    Invalid(ValidateError),
    Sim(SimError),
    OutcomeMismatch { got: Outcome, want: Outcome },
    ObservableMismatch { got: String, want: String },
    PrefixImpure { prefix_len: usize, region: String, index: usize, before: u64, after: u64 },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub schedules: usize,
    pub prefixes: usize,
}

/// Exhaustively check `prog`. `setup` initialises memory before every run;
/// `observe` extracts the state that other actors could legitimately see.
pub fn check<V, S, O>(
    prog: &Program,
    layout: &Layout,
    setup: S,
    observe: O,
    cfg: CheckConfig,
) -> Result<Stats, CheckFailure>
where
    V: PartialEq + Debug,
    S: Fn(&mut Memory),
    O: Fn(&Memory) -> V,
{
    if let Err(e) = prog.validate(layout) {
        return Err(CheckFailure { start_cpu: 0, plan: vec![], kind: FailureKind::Invalid(e) });
    }
    let mut stats = Stats::default();

    // Phase 1: prefix purity. The commit is the last op, so every proper
    // prefix must leave published memory bit-identical.
    for cpu in 0..cfg.ncpus {
        for k in 0..prog.ops.len() {
            let mut mem = Memory::new(layout, cfg.ncpus);
            setup(&mut mem);
            let before = mem.published_snapshot();
            if let Err(e) = sim::run_prefix(prog, &mut mem, cpu, k) {
                return Err(CheckFailure {
                    start_cpu: cpu,
                    plan: vec![],
                    kind: FailureKind::Sim(e),
                });
            }
            stats.prefixes += 1;
            for (region, old) in &before {
                let new = mem.region(*region);
                if let Some(index) = old.iter().zip(new).position(|(a, b)| a != b) {
                    return Err(CheckFailure {
                        start_cpu: cpu,
                        plan: vec![],
                        kind: FailureKind::PrefixImpure {
                            prefix_len: k,
                            region: layout.decl(*region).name.clone(),
                            index,
                            before: old[index],
                            after: new[index],
                        },
                    });
                }
            }
        }
    }

    // Phase 2: restart equivalence over all abort schedules.
    let max_at = prog.ops.len() + usize::from(cfg.sim.post_commit_in_window);
    let mut plan: Vec<AbortPoint> = Vec::new();
    for start_cpu in 0..cfg.ncpus {
        check_schedules(prog, layout, &setup, &observe, &cfg, start_cpu, &mut plan, &mut stats, max_at)?;
    }
    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn check_schedules<V, S, O>(
    prog: &Program,
    layout: &Layout,
    setup: &S,
    observe: &O,
    cfg: &CheckConfig,
    start_cpu: usize,
    plan: &mut Vec<AbortPoint>,
    stats: &mut Stats,
    max_at: usize,
) -> Result<(), CheckFailure>
where
    V: PartialEq + Debug,
    S: Fn(&mut Memory),
    O: Fn(&Memory) -> V,
{
    run_one(prog, layout, setup, observe, cfg, start_cpu, plan, stats)?;
    if plan.len() == cfg.max_aborts {
        return Ok(());
    }
    for at in 0..max_at {
        for new_cpu in 0..cfg.ncpus {
            plan.push(AbortPoint { at, new_cpu });
            check_schedules(prog, layout, setup, observe, cfg, start_cpu, plan, stats, max_at)?;
            plan.pop();
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_one<V, S, O>(
    prog: &Program,
    layout: &Layout,
    setup: &S,
    observe: &O,
    cfg: &CheckConfig,
    start_cpu: usize,
    plan: &[AbortPoint],
    stats: &mut Stats,
) -> Result<(), CheckFailure>
where
    V: PartialEq + Debug,
    S: Fn(&mut Memory),
    O: Fn(&Memory) -> V,
{
    stats.schedules += 1;
    let fail = |kind| CheckFailure { start_cpu, plan: plan.to_vec(), kind };

    let mut mem = Memory::new(layout, cfg.ncpus);
    setup(&mut mem);
    let got = sim::run(prog, &mut mem, start_cpu, plan, cfg.sim)
        .map_err(|e| fail(FailureKind::Sim(e)))?;

    // The final attempt runs entirely on the CPU the run actually finished
    // on, so the reference is a clean run on that CPU from the same initial
    // state. (Plan entries positioned past an early exit never fire, so this
    // is the simulator-reported final CPU, not the plan's last entry.)
    let mut want_mem = Memory::new(layout, cfg.ncpus);
    setup(&mut want_mem);
    let want = sim::run(prog, &mut want_mem, got.final_cpu, &[], SimConfig::default())
        .map_err(|e| fail(FailureKind::Sim(e)))?;
    let want_obs = observe(&want_mem);

    if got.outcome != want.outcome {
        return Err(fail(FailureKind::OutcomeMismatch { got: got.outcome, want: want.outcome }));
    }
    let obs = observe(&mem);
    if obs != want_obs {
        return Err(fail(FailureKind::ObservableMismatch {
            got: format!("{obs:?}"),
            want: format!("{want_obs:?}"),
        }));
    }
    Ok(())
}
