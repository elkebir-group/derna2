//! What `--threads` may change, and what `--deterministic` promises.
//!
//! Threads change the wall clock and nothing a caller reads as an answer: the
//! objectives — MFE and CAI — are the optimum of the same problem at any
//! thread count. What they may change is the **spelling** chosen among designs
//! that tie on the objective, since the order ties are merged in follows the
//! order work finishes.
//!
//! `--deterministic` promises the whole output is byte-identical across thread
//! counts. It is implemented as the single-threaded path, which is the
//! mechanism's cheapest correct form; its cost is therefore the whole of the
//! threading speed-up, and a canonical tie-break is the alternative if a
//! command needs both.
//!
//! The tests below are the two halves that can be asserted on one host: the
//! objectives agree across thread counts, and `--deterministic` reproduces the
//! single-threaded bytes exactly. That ties may differ is a claim about other
//! inputs — it was measured on a 500-residue walk — so no test here asserts a
//! difference, which would be asserting that a tie exists in this instance.

use derna_cli::design::{self, Solved};
use derna_cli::molecule::Molecule;
use derna_cli::result;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// Design the 20-residue fixture at three weights, under the given thread
/// count.
fn solve(threads: usize) -> Vec<Solved> {
    let fasta = root().join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta");
    let argv = words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.1,0.5,0.9",
    ]);
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    design::solve(&m, &args.lambdas, threads).unwrap()
}

fn lines(solved: &[Solved]) -> Vec<String> {
    solved
        .iter()
        .map(|s| {
            let point = result::checked_point(&s.design, Some(s.lambda), true, s.checked.as_ref());
            serde_json::to_string(&point).unwrap()
        })
        .collect()
}

#[test]
fn the_objectives_are_the_same_at_any_thread_count() {
    let one = solve(1);
    let many = solve(4);
    assert_eq!(one.len(), many.len());
    for (a, b) in one.iter().zip(&many) {
        assert_eq!(a.lambda, b.lambda);
        assert_eq!(
            a.design.mfe_kcal_mol, b.design.mfe_kcal_mol,
            "λ={}: the MFE moved with the thread count",
            a.lambda
        );
        assert_eq!(
            a.design.cai_log, b.design.cai_log,
            "λ={}: the adaptation moved with the thread count",
            a.lambda
        );
    }
}

#[test]
fn deterministic_reproduces_the_single_threaded_bytes() {
    // `--deterministic` is the single-threaded path, so the promise is that a
    // run asking for both it and many threads writes what one thread writes.
    let fasta = root().join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta");
    let argv = words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.4",
        "--threads",
        "4",
        "--deterministic",
    ]);
    let args = design::parse(argv).unwrap();
    assert!(args.shared.deterministic);
    assert_eq!(
        args.shared.threads(),
        4,
        "the flag does not rewrite --threads"
    );

    let m = Molecule::resolve(&args.shared).unwrap();
    // The rule itself, read from the code that applies it rather than from a
    // copy of it here: determinism wins over the thread count.
    let threads = args.shared.resolved_threads();
    assert_eq!(threads, 1);
    let deterministic = design::solve(&m, &args.lambdas, threads).unwrap();
    let single = design::solve(&m, &args.lambdas, 1).unwrap();
    assert_eq!(lines(&deterministic), lines(&single));
}

#[test]
fn the_header_records_the_thread_count_the_run_used() {
    // A result taken under `--deterministic` records 1, not the 4 the caller
    // wrote: the header says what ran, so a reader comparing two files is
    // comparing the runs rather than the command lines.
    let fasta = root().join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta");
    let argv = words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.4",
        "--threads",
        "8",
        "--deterministic",
    ]);
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let mut extra = serde_json::Map::new();
    extra.insert("threads".into(), args.shared.resolved_threads().into());
    extra.insert("deterministic".into(), args.shared.deterministic.into());
    let header = result::header("design", &m, extra);
    assert_eq!(header["threads"], Value::from(1));
    assert_eq!(header["deterministic"], Value::Bool(true));
}
