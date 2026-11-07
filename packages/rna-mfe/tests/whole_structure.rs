//! End-to-end whole-structure parity test for [`rna_mfe::evaluate`].
//!
//! Loads the full curated fixture corpus (`tests/data/mfe_fixtures.json`),
//! filters to `dangling_5_prime: false`
//! records, runs each through [`rna_mfe::evaluate`], and asserts
//! agreement at the `0.001 kcal/mol` tolerance. The long-form fixture
//! (234 nt) is the integration acceptance gate — every other fixture is
//! a localized regression check.
//!
//! Additionally asserts exact integer-dekacal equality on every
//! fixture; if any fixture has float slop within the tolerance window
//! but not at exact-dekacal, the test surfaces the gap.

use rna_mfe::{evaluate, Energy, Params};
use serde::Deserialize;

const TOLERANCE_KCAL_PER_MOL: f64 = 0.001;

#[test]
fn every_dangles_zero_fixture_matches_evaluator() {
    let params = Params::load_default().expect("load Turner 2004");
    let fixtures = load_fixtures();
    assert!(
        !fixtures.is_empty(),
        "expected at least one dangles=0 fixture in the corpus",
    );

    let mut all_exact = true;
    for fixture in &fixtures {
        let computed = evaluate(&params, fixture.seq.as_bytes(), &fixture.structure)
            .expect("evaluate succeeds on curated corpus");
        let computed_kcal = computed.to_kcal_mol();
        let delta = (computed_kcal - fixture.energy_kcal_mol).abs();
        assert!(
            delta < TOLERANCE_KCAL_PER_MOL,
            "fixture mismatch: motif={} closure={} seq={} expected={:.4} computed={:.4} delta={:.4}",
            fixture.motif,
            fixture.closure,
            fixture.seq,
            fixture.energy_kcal_mol,
            computed_kcal,
            delta,
        );
        if !exact_dekacal(computed, fixture.energy_kcal_mol) {
            all_exact = false;
        }
    }

    assert!(
        all_exact,
        "expected every fixture to agree at exact integer-dekacal equality; \
         some fixtures sit within tolerance but not exact",
    );
}

/// Compare a computed [`Energy`] (integer dekacal) against a fixture's
/// `f64 kcal/mol` value at exact integer-dekacal equality. The reference
/// energies are stored to 0.01 kcal/mol precision, so a
/// successful round-trip means `(computed.dekacal() / 100.0).round()`
/// equals the fixture's recorded value when expressed as dekacal.
fn exact_dekacal(computed: Energy, fixture_kcal: f64) -> bool {
    // Reference energies are written as `%6.2f` → dekacal is exactly
    // 100 × the kcal value (within the smallest representable f64
    // increment). Round-trip via i32 to compare integers, not floats.
    let fixture_dekacal = (fixture_kcal * 100.0).round() as i32;
    computed.dekacal() == fixture_dekacal
}

fn load_fixtures() -> Vec<Fixture> {
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
