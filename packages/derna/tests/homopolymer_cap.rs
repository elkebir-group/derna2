//! `--max-homopolymer K` on an emitted design, from both sides.
//!
//! The flag's expansion — exactly the four `(K + 1)`-runs — is tested where it
//! is written (`sequence_constraint`), and the equivalence between those four
//! patterns and a run-length cap is proven against an independent reference in
//! `derna-lib`'s `homopolymer_cap`. What is left, and what this file does, is
//! the end-to-end claim: a run under the flag emits a design whose longest run
//! fits the cap, and it does so **because of the flag**.
//!
//! The second half is the one worth writing down. A cap chosen without looking
//! at the molecule may forbid nothing on it — the unconstrained design's runs
//! may already fit — and then "the design fits the cap" is true of the
//! unconstrained run too and says nothing about the constraint. So the cap here
//! is derived from a *measurement*: design once with no cap, read the longest
//! run off that design, and cap one below it. The constraint then provably
//! binds, and the emitted sequence is asserted to differ.

use derna_cli::design::{self, Solved};
use derna_cli::molecule::Molecule;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fasta() -> String {
    root()
        .join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta")
        .to_str()
        .unwrap()
        .to_string()
}

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// Design at one weight under the given words.
fn design_once(words: &[String]) -> Vec<Solved> {
    let args = design::parse(words.to_vec()).unwrap();
    args.sequence.validate().unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None).unwrap()
}

/// The longest run of one nucleotide in `seq`, counted here rather than read
/// from the metrics the design carries: the assertion and the reported field
/// should be able to disagree.
fn longest_run(seq: &str) -> usize {
    let mut best = 0;
    let mut run = 0;
    let mut last = None;
    for b in seq.bytes() {
        run = if last == Some(b) { run + 1 } else { 1 };
        last = Some(b);
        best = best.max(run);
    }
    best
}

#[test]
fn a_cap_measured_to_bind_is_met_and_changes_the_design() {
    let free = design_once(&words(&["--protein", &fasta(), "--lambda", "0.5"]));
    let free_seq = free[0].design.sequence.clone();
    let free_run = longest_run(&free_seq);
    // The cap has to be able to bite, or the rest of the test is a statement
    // about an unconstrained run. One below the measured run is the cap used,
    // and it has to stay at 2 or above: this protein holds `K` (`AAA`/`AAG`)
    // and `F` (`UUU`/`UUC`), so a cap of 1 has no encoding at all and would
    // make the run below a refusal rather than a constrained design.
    assert!(
        free_run >= 3,
        "the unconstrained design's longest run is {free_run}, so the cap below it \
         is not one this protein can meet",
    );
    let k = free_run - 1;
    let capped = design_once(&words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--max-homopolymer",
        &k.to_string(),
    ]));
    let capped_seq = capped[0].design.sequence.clone();
    assert!(
        longest_run(&capped_seq) <= k,
        "the capped design spells a run of {}: {capped_seq}",
        longest_run(&capped_seq),
    );
    // The binds check: the constraint changed the emitted design. Byte-equal
    // sequences would mean the cap forbade nothing on this molecule, and the
    // assertion above would have been met by the unconstrained run.
    assert_ne!(
        free_seq, capped_seq,
        "the cap at {k} left the design unchanged, so it forbade nothing here",
    );
    // A cap is a constraint, so it cannot improve the objective. MFE alone can
    // move either way at a weight below 1; the constrained optimum's BRD
    // cannot beat the unconstrained one.
    let brd = |s: &Solved| 0.5 * s.design.mfe_kcal_mol - 0.5 * s.design.cai_log;
    assert!(
        brd(&capped[0]) >= brd(&free[0]) - 1e-9,
        "the capped design beat the unconstrained optimum",
    );
}

#[test]
fn a_cap_the_molecule_already_meets_forbids_nothing() {
    // The mirror, and the reason the test above derives its cap from a
    // measurement: a cap at or above the unconstrained design's longest run
    // leaves that design byte-identical. The same "fits the cap" assertion
    // passes here while the constraint does nothing, which is what makes a cap
    // chosen by hand worthless as evidence.
    let free = design_once(&words(&["--protein", &fasta(), "--lambda", "0.5"]));
    let free_seq = free[0].design.sequence.clone();
    let k = longest_run(&free_seq);
    let capped = design_once(&words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--max-homopolymer",
        &k.to_string(),
    ]));
    assert_eq!(free_seq, capped[0].design.sequence);
}

#[test]
fn the_cap_is_recorded_as_the_four_patterns_it_forbids() {
    // A file read later has to be able to say what the run was under, and the
    // constraint it was under is the four patterns — so they are what the
    // header records, rather than the flag's value alone.
    let args = design::parse(words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--max-homopolymer",
        "4",
    ]))
    .unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let fields = args.sequence.header_fields(&m).unwrap();
    assert_eq!(
        fields["motifs"],
        serde_json::json!(["AAAAA", "CCCCC", "GGGGG", "UUUUU"])
    );
    assert_eq!(
        fields["motif_window"],
        serde_json::json!("the coding sequence")
    );
    assert!(!fields.contains_key("motif_scopes"), "{fields:?}");
}

#[test]
fn a_cap_that_no_encoding_can_meet_is_refused_by_name() {
    // `--max-homopolymer 1` forbids `AA`, `CC`, `GG` and `UU`, which no
    // encoding of a protein holding a two-of-a-kind codon can avoid. The
    // refusal has to name the patterns, because the flag's value alone does
    // not say which letters ran out.
    let args = design::parse(words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--max-homopolymer",
        "1",
    ]))
    .unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let e = design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("no encoding of the protein avoids every motif"),
        "{e}"
    );
    assert!(e.contains("AA"), "{e}");
}
