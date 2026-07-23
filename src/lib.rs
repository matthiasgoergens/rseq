//! A combinator algebra for Linux restartable sequences (rseq), starting from
//! the model side: an IR whose shape is exactly the shape of a valid rseq
//! critical section, an interpreter with deterministic abort injection, and
//! an exhaustive bounded model checker.
//!
//! The design: one IR, three backends. This crate currently implements the
//! simulator and checker
//! backends; the asm-template + `rseq_cs` descriptor emitter and the
//! ptrace-based conformance harness against the live kernel come next.

pub mod check;
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
pub mod codegen;
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
pub mod drain;
pub mod ir;
pub mod progs;
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
pub mod rt;
pub mod sim;
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
pub mod sys;
