//! Multi-loop fixture parity for [`rna_mfe::fold`].
//!
//! Loads the curated fixture corpus
//! (`tests/data/mfe_fixtures.json`), filters to the
//! `motif: "multi_loop"`, `dangling_5_prime: false` records, and asserts:
//!
//! 1. **Energy parity** — exact integer-dekacal equality between the
//!    folder's energy and the fixture's `energy_kcal_mol * 100` rounded
//!    to int.
//! 2. **Structure parity** — exact-string equality on the dot-bracket.
//! 3. **Round-trip** — feeding the returned structure back through
//!    [`rna_mfe::evaluate`] must agree exactly with the folder's
//!    reported energy. Catches "fold returns a structure that doesn't
//!    match its reported energy" bugs — important for multi-loops where
//!    the per-loop evaluator pre-existed the multi-loop DP cells and
//!    serves as an independent oracle.

use rna_mfe::{evaluate, fold, Energy, Params};
use serde::Deserialize;

/// Exactly 4 multi-loop dangles=0 fixtures in the corpus (3-branch
/// and 4-branch cases). All must pass.
const MULTI_LOOP_FIXTURE_COUNT: usize = 4;

#[test]
fn multi_loop_fixtures_match_fold() {
    let params = Params::load_default().expect("load Turner 2004");
    let fixtures = load_multi_loop_fixtures();
    assert_eq!(
        fixtures.len(),
        MULTI_LOOP_FIXTURE_COUNT,
        "expected {MULTI_LOOP_FIXTURE_COUNT} multi-loop dangles=0 fixtures, got {}",
        fixtures.len(),
    );
    for fixture in &fixtures {
        check_fixture(&params, fixture);
    }
}

fn check_fixture(params: &Params, fixture: &Fixture) {
    let (structure, energy) = fold(params, fixture.seq.as_bytes()).expect("fold succeeds");

    let expected_dekacal = (fixture.energy_kcal_mol * 100.0).round() as i32;
    assert_eq!(
        energy.dekacal(),
        expected_dekacal,
        "energy mismatch: seq={} expected={} kcal/mol ({} dekacal) computed={} dekacal structure={}",
        fixture.seq,
        fixture.energy_kcal_mol,
        expected_dekacal,
        energy.dekacal(),
        structure,
    );

    assert_eq!(
        structure, fixture.structure,
        "structure mismatch: seq={} expected={} computed={}",
        fixture.seq, fixture.structure, structure,
    );

    let round_trip = evaluate(params, fixture.seq.as_bytes(), &structure).expect("evaluate");
    assert_eq!(
        round_trip,
        energy,
        "round-trip mismatch: seq={} fold-energy={} evaluate-energy={} structure={}",
        fixture.seq,
        energy.dekacal(),
        round_trip.dekacal(),
        structure,
    );
    let _: Energy = round_trip; // anchor the type for readers
}

fn load_multi_loop_fixtures() -> Vec<Fixture> {
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mfe_fixtures.json",);
    let raw = std::fs::read_to_string(PATH).expect("read mfe_fixtures.json");
    let parsed: FixtureFile = serde_json::from_str(&raw).expect("parse mfe_fixtures.json");
    parsed
        .fixtures
        .into_iter()
        .filter(|f| !f.dangling_5_prime && f.motif == "multi_loop")
        .collect()
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    dangling_5_prime: bool,
    energy_kcal_mol: f64,
    motif: String,
    seq: String,
    structure: String,
}
