//! Random-corpus property test for [`rna_mfe::evaluate`].
//!
//! The curated `mfe_fixtures.json` corpus pins specific Turner-2004 motif
//! combinations at controlled boundaries — strong evidence of correctness
//! *for those structures*. This test consumes a complementary random-sequence
//! corpus committed at `tests/data/property_fixtures.json`: random
//! sequences folded by Vienna at `dangles=0`, with the resulting
//! `(seq, structure, energy)` triples stored deterministically (fixed RNG
//! seeds). Corpus is static — the original generator lived alongside the
//! retired Python reference; if regeneration is needed, reach for git
//! history.
//!
//! ## Corpus layout
//!
//! - **`fixtures`** (500 records, 12–100 nt): random sequences folded by
//!   `RNAfold -d 0`. The base corpus.
//! - **`extended_fixtures.long_sequences`** (75 records, 150–300 nt): same
//!   MFE-folded triples at longer lengths. Multi-branch exterior loops and
//!   nested multi-loops compound; tie-break density goes up. Valid for both
//!   evaluate and fold.
//! - **`extended_fixtures.lxc_cases`** (20 records): template-constructed
//!   structures with interior loops of size `n1+n2 > 30`, scored by
//!   `RNAeval`. These exercise the `lxc` extrapolation path
//!   (`loops/internal.rs:interior_size_penalty`). Vienna's `RNAfold` caps
//!   loop search at `MAXLOOP=30` so these structures are never MFE — but
//!   `evaluate` can score them and must agree with `RNAeval`. The fold
//!   test ([`property_fold.rs`](property_fold.rs)) excludes this section.
//!
//! ## Coverage delta vs the curated corpus
//!
//! The random fixtures land on whatever-shape structures Vienna's MFE
//! search lands on for arbitrary inputs — multi-loops with branch counts the
//! curated set doesn't reach, mixed motifs, base-pair combinations the curated
//! stack-cell coverage doesn't pin. A failure here means a code path the
//! curated corpus doesn't exercise has a bug.
//!
//! Asserts exact integer-dekacal equality against Vienna's recorded energy.
//! The looser `0.001 kcal/mol` safety net is also enforced — same posture
//! as the whole-structure integration test.

use rna_mfe::{evaluate, Energy, Params};
use serde::Deserialize;

const TOLERANCE_KCAL_PER_MOL: f64 = 0.001;

#[test]
fn property_fixtures_match_evaluator() {
    let params = Params::load_default().expect("load Turner 2004");
    let fixtures = load_property_fixtures();
    assert!(
        fixtures.len() >= 500,
        "expected at least 500 property fixtures, got {} — re-run \
         scripts/generate_property_fixtures.py",
        fixtures.len(),
    );

    for fixture in &fixtures {
        let computed = evaluate(&params, fixture.seq.as_bytes(), &fixture.structure)
            .expect("evaluate succeeds on Vienna-folded corpus");
        let computed_kcal = computed.to_kcal_mol();
        let delta = (computed_kcal - fixture.energy_kcal_mol).abs();
        assert!(
            delta < TOLERANCE_KCAL_PER_MOL,
            "property fixture mismatch: seq={} structure={} expected={:.4} computed={:.4} delta={:.4}",
            fixture.seq,
            fixture.structure,
            fixture.energy_kcal_mol,
            computed_kcal,
            delta,
        );
        let fixture_dekacal = (fixture.energy_kcal_mol * 100.0).round() as i32;
        assert_eq!(
            computed.dekacal(),
            fixture_dekacal,
            "property fixture failed exact-dekacal equality \
             (within {:.4} kcal/mol but not bit-identical): seq={} structure={} expected_dekacal={} computed_dekacal={}",
            TOLERANCE_KCAL_PER_MOL,
            fixture.seq,
            fixture.structure,
            fixture_dekacal,
            computed.dekacal(),
        );
        let _ = Energy::ZERO;
    }
}

fn load_property_fixtures() -> Vec<Fixture> {
    const PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/property_fixtures.json",
    );
    let raw = std::fs::read_to_string(PATH).expect("read property_fixtures.json");
    let parsed: FixtureFile = serde_json::from_str(&raw).expect("parse property_fixtures.json");
    let mut all = parsed.fixtures;
    all.extend(parsed.extended_fixtures.long_sequences.fixtures);
    // lxc_cases: template structures with interior loops > MAXLOOP=30,
    // scored by RNAeval. Exercises the lxc extrapolation in evaluate().
    all.extend(parsed.extended_fixtures.lxc_cases.fixtures);
    all
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
    extended_fixtures: ExtendedFixtures,
}

#[derive(Debug, Deserialize)]
struct ExtendedFixtures {
    long_sequences: ExtendedSection,
    lxc_cases: ExtendedSection,
}

#[derive(Debug, Deserialize)]
struct ExtendedSection {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    seq: String,
    structure: String,
    energy_kcal_mol: f64,
}
