# rseq

A combinator algebra for Linux restartable sequences, built model-first.

Restartable sequences (`rseq(2)`) give a thread atomicity over per-CPU data
for a short instruction window: if the thread is preempted or migrated inside
the window, the kernel restarts it at the top instead of letting it commit.
The catch is that a valid sequence has a very rigid shape — a side-effect-free
prefix, at most one branch-to-exit, exactly one committing store — and getting
that shape (or its `rseq_cs` descriptor) subtly wrong is nearly untestable by
stress alone, because natural aborts are rare.

The design here is one IR with three backends:

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
4. **ptrace conformance harness** (next): single-step the real binary to
   every abort point and inject signals/migrations against the live kernel,
   closing the gap between the checked IR and the emitted bytes (descriptor
   placement, window bounds, signature).

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
- [tests/model.rs](tests/model.rs) — the checker run against the example
  programs; [tests/kernel.rs](tests/kernel.rs) and
  [tests/codegen.rs](tests/codegen.rs) — live-kernel stress and
  sim-equivalence tests for hand-written and compiled sequences.

## Running

```
cargo test
```
