//! Whole-corpus fixture parity for [`rna_mfe::fold`].
//!
//! Loads the full curated fixture corpus
//! (`tests/data/mfe_fixtures.json`), filters to all
//! `dangling_5_prime: false` records (47 fixtures across hairpin, bulge,
//! internal_loop, multi_loop, and long_form motifs), then for each runs
//! [`rna_mfe::fold`] and asserts:
//!
//! 1. **Energy parity** — exact integer-dekacal equality.
//! 2. **Structure parity** — exact-string equality on the dot-bracket.
//! 3. **Round-trip** — `evaluate(seq, returned_structure) == fold_energy`.
//!
//! The long_form record (234 nt) is the integration gate for
//! multi-branch exterior loops.

use rna_mfe::{evaluate, fold, Energy, Params};
use serde::Deserialize;

#[test]
fn all_dangles_zero_fixtures_match_fold() {
    let params = Params::load_default().expect("load Turner 2004");
    let fixtures = load_dangles_zero_fixtures();
    assert!(
        fixtures.len() >= 47,
        "expected at least 47 dangles=0 fixtures, got {}",
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
        "energy mismatch: motif={} closure={} seq={} expected={} kcal/mol ({} dekacal) computed={} dekacal structure={}",
        fixture.motif,
        fixture.closure,
        fixture.seq,
        fixture.energy_kcal_mol,
        expected_dekacal,
        energy.dekacal(),
        structure,
    );

    assert_eq!(
        structure, fixture.structure,
        "structure mismatch: motif={} closure={} seq={} expected={} computed={}",
        fixture.motif, fixture.closure, fixture.seq, fixture.structure, structure,
    );

    let round_trip = evaluate(params, fixture.seq.as_bytes(), &structure).expect("evaluate");
    assert_eq!(
        round_trip,
        energy,
        "round-trip mismatch: motif={} closure={} seq={} fold-energy={} evaluate-energy={} structure={}",
        fixture.motif,
        fixture.closure,
        fixture.seq,
        energy.dekacal(),
        round_trip.dekacal(),
        structure,
    );
    let _: Energy = round_trip;
}

fn load_dangles_zero_fixtures() -> Vec<Fixture> {
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mfe_fixtures.json",);
    let raw = std::fs::read_to_string(PATH).expect("read mfe_fixtures.json");
    let parsed: FixtureFile = serde_json::from_str(&raw).expect("parse mfe_fixtures.json");
    parsed
        .fixtures
        .into_iter()
        .filter(|f| !f.dangling_5_prime)
        .collect()
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    closure: String,
    dangling_5_prime: bool,
    energy_kcal_mol: f64,
    motif: String,
    seq: String,
    structure: String,
}
