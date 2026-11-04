//! `--beam-first` refutes a design without the exact fold, and says so.
//!
//! The flag is the one speed lever on the command surface, and it is a flag
//! rather than a default because it alone changes what a result reports: a
//! point the beam refutes carries `free_mfe_at_most` and `gap_at_least` in
//! place of the exact `free_mfe` and `gap`, so a bound cannot be read as an
//! exact value.
//!
//! Two things have to hold together. The refutation must agree with the exact
//! check on the **verdict** — it is the same claim, reached cheaply — and it
//! must not report the exact fields it never computed. A test of either alone
//! would pass on an implementation that silently ran the exact fold anyway, or
//! on one that reported a bound as though it were exact.

use derna_cli::design::{self};
use derna_cli::free_check::Checked;
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

/// The constrained case the fixtures record as failing its own check: the
/// 20-residue protein with the hbb25 flanks under all three crossing classes.
fn failing_case(extra: &[&str]) -> Vec<String> {
    let text =
        std::fs::read_to_string(root().join("packages/derna/tests/data/flank-pairs.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    let flanks = &v["hbb25"];
    let fasta = root().join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta");
    let mut argv = words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.1",
        "--utr5",
        flanks["utr5"].as_str().unwrap(),
        "--utr3",
        flanks["utr3"].as_str().unwrap(),
    ]);
    argv.extend(words(extra));
    argv
}

fn solve(argv: Vec<String>) -> (Molecule, Vec<design::Solved>) {
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let solved =
        design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, args.shared.beam_first)
            .unwrap();
    (m, solved)
}

#[test]
fn a_refuted_design_reports_a_bound_and_not_the_exact_pair() {
    let (_, exact) = solve(failing_case(&[]));
    let (_, bounded) = solve(failing_case(&["--beam-first", "50"]));

    // The design itself is the same: the flag changes the check, not the solve.
    assert_eq!(exact[0].design, bounded[0].design);

    // The exact check finds this design violates the constraint...
    let exact_gap = match exact[0]
        .checked
        .as_ref()
        .expect("a constrained run is checked")
    {
        Checked::Exact(gap) => gap.clone(),
        Checked::Refuted { .. } => panic!("no beam was asked for"),
    };
    assert!(
        !exact_gap.is_modular(),
        "the case is modular, so nothing to refute"
    );

    // ...and so does the beam, without the exact fold.
    let line = result::checked_point(
        &bounded[0].design,
        Some(bounded[0].lambda),
        true,
        bounded[0].checked.as_ref(),
    );
    match bounded[0].checked.as_ref().unwrap() {
        Checked::Refuted {
            free_mfe_at_most,
            gap_at_least,
        } => {
            // The bound is sound: a real structure's energy, so it cannot sit
            // below the exact free MFE.
            assert!(
                *free_mfe_at_most >= exact_gap.free_mfe_kcal_mol - 1e-9,
                "the bound {free_mfe_at_most} claims less than the exact free MFE {}",
                exact_gap.free_mfe_kcal_mol
            );
            assert!(*gap_at_least > 0.0);
            assert!(
                *gap_at_least <= exact_gap.gap_kcal_mol + 1e-9,
                "the bound claims a larger gap than the exact one"
            );
            // And the line names it as a bound.
            assert_eq!(
                line["free_mfe_at_most"].as_f64().unwrap(),
                *free_mfe_at_most
            );
            assert_eq!(line["gap_at_least"].as_f64().unwrap(), *gap_at_least);
            assert!(
                line.get("free_mfe").is_none(),
                "an exact field on a bounded check"
            );
            assert!(
                line.get("gap").is_none(),
                "an exact field on a bounded check"
            );
            assert_eq!(line["modular"], Value::Bool(false));
        }
        Checked::Exact(_) => panic!("the beam refuted nothing on a design that violates"),
    }
}

#[test]
fn the_two_checks_agree_on_the_verdict() {
    let (_, exact) = solve(failing_case(&[]));
    let (_, bounded) = solve(failing_case(&["--beam-first", "50"]));
    assert_eq!(
        exact[0].checked.as_ref().unwrap().modular(),
        bounded[0].checked.as_ref().unwrap().modular()
    );
}

#[test]
fn without_the_flag_the_result_is_unchanged() {
    // The default has to stay exact: every recorded result was taken without
    // this flag, and a bound in place of a value would be a silent change.
    let (_, solved) = solve(failing_case(&[]));
    let line = result::checked_point(
        &solved[0].design,
        Some(0.1),
        true,
        solved[0].checked.as_ref(),
    );
    assert!(line.get("free_mfe").is_some());
    assert!(line.get("gap").is_some());
    assert!(line.get("free_mfe_at_most").is_none());
}

#[test]
fn a_zero_width_beam_is_refused() {
    assert!(design::parse(failing_case(&["--beam-first", "0"])).is_err());
    assert!(design::parse(failing_case(&["--beam-first", "x"])).is_err());
}
