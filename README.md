# rseq

A combinator algebra for Linux restartable sequences, built model-first.

## Why

Restartable sequences (`rseq(2)`) give a thread atomicity over per-CPU data
for a short instruction window: if the thread is preempted or migrated inside
the window, the kernel restarts it at the top instead of letting it commit.
Unlike transactional memory, the abort condition is not "someone touched my
read set" but "my thread lost its claim to this CPU" — the closer analogy is
load-linked/store-conditional stretched to a handful of instructions with a
single committing store. That is what makes per-CPU counters, freelists, and
tcmalloc-style caches run at plain-store speed with no lock prefix at all.

The catch is twofold. A valid sequence has a rigid shape — a replay-safe
prefix of loads and register arithmetic, writes only to unpublished scratch
memory, exactly one committing store, terminal — and every prefix must be
idempotent, because the kernel may restart the sequence at any instruction
boundary with a possibly different CPU id. And the classic bugs (a hoisted
CPU id that goes stale across a migration, an off-by-one
`post_commit_offset` that re-executes a completed commit) are nearly
untestable by stress alone, because natural aborts are rare and land on
arbitrary instructions.

The answer here is to make the sequence shape a *data structure* rather than
a discipline: programs are built from a small algebra whose consuming
finisher makes "one terminal commit" true by construction, and the same IR
feeds every backend, so the artifact that is model-checked is the artifact
that runs. Aborts stop being rare events you hope to hit: the checker
enumerates all of them, and a ptrace harness replays each one against the
live kernel.

## The backends

1. **Simulator + model checker** (this milestone, done): the IR is
   interpreted with deterministic abort injection. The checker enumerates
   every abort schedule up to a bounded depth — every prefix boundary crossed
   with every destination CPU — and verifies that each run is
   indistinguishable from an abort-free run: prefixes never touch published
   memory, restarts converge to the clean result, and commits only ever
   target the current CPU's slice. The classic bug classes fall out as
   checker failures: a hoisted (stale) CPU id surfaces as a foreign commit
   under migration, and an abort window that wrongly includes the committing
   store surfaces as a double commit.
2. **Live-kernel runtime spike** (done): hand-written x86-64 sequences
   (per-CPU counter, freelist push/pop) with their `rseq_cs` descriptors,
   riding glibc's auto-registration via `__rseq_offset`. Oversubscribed
   stress tests exercise the real abort/retry path and assert conservation
   laws that any lost or doubled commit would break. These are the bytes the
   IR backend must learn to emit, and their tests are the harness generated
   code must pass.
3. **Codegen** (done): a small JIT compiles the same IR the checker verified
   into executable x86-64 machine code, descriptor included — one anonymous
   mapping holding the 32-byte-aligned `rseq_cs` at offset 0 and the
   arm/start/commit/exit/abort code after it, W^X protected. Runtime
   parameters (`Op::Param`) let one compiled push serve every node. The
   compiled programs pass the same live-kernel stress harness as the
   hand-written ones and agree with the simulator on single-threaded runs,
   so model-checked and executed programs are now the same artifact.
4. **ptrace conformance harness** (done): a forked child loops on a compiled
   sequence under ptrace; for every instruction boundary inside the window
   the parent runs it into an int3, rewinds, forces a migration with
   `sched_setaffinity` (a guaranteed rseq event), and observes with
   breakpoints on both possible continuations what the kernel did on resume.
   Every in-window boundary must redirect to the abort handler with no
   commit visible; one boundary past the committing store must NOT redirect
   and the commit must count exactly once. Registration is verified from
   outside via `PTRACE_GET_RSEQ_CONFIGURATION`. The harness caught a real
   bug on its first run: the JIT held the descriptor address in rax, which
   also serves as the region-base scratch, so a retry after an abort
   re-armed `rseq_cs` with a region base and ran unprotected — a
   double-abort-in-one-call bug (~1e-8 per call) that stress testing had no
   realistic chance of finding.

## Layout

- [src/ir.rs](src/ir.rs) — the IR, structural validation, and a builder whose
  consuming `commit` finisher makes "exactly one terminal committing store"
  true by construction.
- [src/sim.rs](src/sim.rs) — interpreter with plan-driven abort injection,
  including an opt-in model of the buggy post-commit descriptor window.
- [src/check.rs](src/check.rs) — exhaustive bounded checker. Observability is
  a user-supplied abstraction function, which is what lets the tcmalloc
  scribble-then-bump trick check cleanly: scratch scribbles from aborted
  attempts are real but unobservable.
- [src/progs.rs](src/progs.rs) — example programs: per-CPU counter, per-CPU
  freelist push/pop, tcmalloc-style array push, plus deliberately buggy
  variants the checker must reject.
- [src/rt.rs](src/rt.rs) — live-kernel runtime: rseq-area access via glibc's
  `__rseq_offset`, plus hand-written asm sequences with descriptors.
- [src/codegen.rs](src/codegen.rs) — the JIT: x86-64 encoder, linear-scan
  register allocation, descriptor emission, raw-syscall mmap, and
  `RegionSet` for driving compiled programs.
- [src/sys.rs](src/sys.rs) — minimal raw-syscall layer (mmap, fork, wait,
  ptrace, affinity), keeping the crate dependency-free.
- [tests/model.rs](tests/model.rs) — the checker run against the example
  programs; [tests/kernel.rs](tests/kernel.rs) and
  [tests/codegen.rs](tests/codegen.rs) — live-kernel stress and
  sim-equivalence tests for hand-written and compiled sequences;
  [tests/ptrace.rs](tests/ptrace.rs) — the deterministic abort-point
  conformance harness.

5. **Benchmarks** (started): `src/bin/bench.rs`, a randomised complete block
   design — each block samples θ = (affinity domain, thread count), runs
   every arm once in randomised order, and appends to a CSV with continuing
   block ids, so runs are start/stop-able anytime and results concatenate.
   Analysis is paired within-block (median ratios vs the mutex baseline,
   sign tests), never pooled means, so background-load drift cannot bias
   the comparison. Domains include P-core-only and E-core-only masks on
   hybrid machines (from `/sys/devices/cpu_core`/`cpu_atom`). Six arms
   isolate one effect per adjacent pair: JIT vs hand-written asm (call
   overhead), 64-byte vs 8-byte stride (false sharing), rseq vs per-CPU
   `lock xadd` (the lock prefix), sharded vs shared atomic (contention),
   and a `Mutex<u64>` baseline. First 100-block dataset on a 32-CPU Raptor
   Lake box: rseq arms run 5-60x the mutex and 2-8x the sharded atomic in
   every regime (unanimous within-block sign test), padding is worth ~2x at
   high thread counts and nothing single-threaded, and abort rates are
   ~2e-7 per op under oversubscription, zero otherwise.

6. **Cross-CPU draining** (done): the tcmalloc `FenceCpu` protocol. Every
   sequence of the lockable freelist checks a per-CPU drain lock inside its
   window and exits early if set; a drainer takes the lock, fires
   `MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ` (restarting any in-flight
   sequence that could have read the old lock value), then owns that CPU's
   list outright until it unlocks. The lock-guarded programs are
   model-checked like everything else (locked CPUs never commit, unlocked
   behave identically), and the live-kernel stress test makes the fence
   load-bearing: a continuous drainer races pop/push workers, and the
   end-state conservation law would break if the fence failed to restart a
   stale-read sequence. Multiple exit branches per program (locked, empty)
   are now allowed by the validator — each just leaves the window
   uncommitted.

## Running

```
cargo test
cargo run --release --bin bench -- counter --blocks 50   # appends bench-counter.csv
cargo run --release --bin bench -- analyze bench-counter.csv
```
