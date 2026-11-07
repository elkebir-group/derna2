//! `solve_brd` against reference designs at fixed weights.
//!
//! Each fixture records the reference design at a weight. The reference
//! scores raw log-CAI where this solver scales it by 100, so the equivalent
//! weight on the reference's scale is `lam / (100 - 99*lam)`; the fixtures
//! carry both ends of it.
//!
//! The sequence is the load-bearing assertion. CAI is a function of the
//! sequence and MFE a function of sequence and structure, so an exact
//! sequence match subsumes them; they are checked anyway because a
//! disagreement among them would mean something stranger than a different
//! design. CAI is compared at tolerance because the reference records six
//! digits.
//!
//! The `glpe78_*` cases are the regression: at each of those three weights
//! the quantized objective returned a design the reference does not.

use std::fs;
use std::path::PathBuf;

use derna_codon_dp::codon_usage::CodonUsage;
use derna_codon_dp::{solve_brd, BrdOptions};
use serde::Deserialize;

/// CAI is recorded by the reference to six significant digits.
const CAI_TOL: f64 = 1e-6;
/// MFE is read from an integer dekacal field, so this only absorbs the
/// division by 100.
const MFE_TOL: f64 = 1e-9;

#[derive(Debug, Deserialize)]
struct ReferenceAnchor {
    name: String,
    protein: String,
    lambda_: f64,
    rna_seq: String,
    structure: String,
    mfe_kcal_mol: f64,
    cai: f64,
}

fn load(name: &str) -> ReferenceAnchor {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(format!("reference_anchor_{name}.json"));
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn assert_matches_reference(name: &str) {
    let anchor = load(name);
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    let got =
        solve_brd(&anchor.protein, &codon_usage, anchor.lambda_, &options).expect("solve_brd");

    assert_eq!(
        got.rna_seq,
        anchor.rna_seq,
        "[{}] sequence differs from the reference at λ={} (raw-scale λ={})",
        anchor.name,
        anchor.lambda_,
        anchor.lambda_ / (100.0 - 99.0 * anchor.lambda_),
    );
    assert_eq!(
        got.structure, anchor.structure,
        "[{}] structure differs from the reference at λ={}",
        anchor.name, anchor.lambda_,
    );
    assert!(
        (got.mfe_kcal_mol - anchor.mfe_kcal_mol).abs() <= MFE_TOL,
        "[{}] MFE {} != reference {}",
        anchor.name,
        got.mfe_kcal_mol,
        anchor.mfe_kcal_mol,
    );
    assert!(
        (got.cai - anchor.cai).abs() <= CAI_TOL,
        "[{}] CAI {} != reference {} (tolerance {CAI_TOL})",
        anchor.name,
        got.cai,
        anchor.cai,
    );
}

#[test]
fn toy_lambda_25() {
    assert_matches_reference("toy_lambda_25");
}

#[test]
fn toy_lambda_50() {
    assert_matches_reference("toy_lambda_50");
}

#[test]
fn toy_lambda_75() {
    assert_matches_reference("toy_lambda_75");
}

#[test]
fn toy_lambda_100() {
    assert_matches_reference("toy_lambda_100");
}

#[test]
#[ignore = "corpus-length acceptance check; run with --ignored --release"]
fn glpe78_lambda_0234() {
    assert_matches_reference("glpe78_lambda_0234");
}

#[test]
#[ignore = "corpus-length acceptance check; run with --ignored --release"]
fn glpe78_lambda_0391() {
    assert_matches_reference("glpe78_lambda_0391");
}

#[test]
#[ignore = "corpus-length acceptance check; run with --ignored --release"]
fn glpe78_lambda_1016() {
    assert_matches_reference("glpe78_lambda_1016");
}
