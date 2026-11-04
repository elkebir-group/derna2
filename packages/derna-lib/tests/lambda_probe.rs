//! The λ probe against the grid it claims to have walked.
//!
//! The probe makes one positive claim and one negative one, and both are
//! checkable by brute force at this size: the λ it returns is modular, and it is
//! the *nearest* modular λ on the grid — no earlier point on the walk would have
//! done. The negative form ("no λ on this grid verifies modular") is checked the
//! same way, by solving every point.
//!
//! What is deliberately not asserted is that a rescue exists at all. Whether the
//! λ axis reaches modularity is a property of the configuration, not of the
//! instrument, and a test that demanded one would be pinning a measurement.

use derna_lib::{
    design_with_utr_spec, probe_grid, probe_lambda, sequence_gap, CdsWindow, CodonUsage,
    Enforcement, Params, ProbeInputs, ProbeStep, RegionConstraints, ScanConfig, UtrSpec,
};

const PROTEIN: &str = "LSKF";
const UTR5: &str = "GGGGGG";
const UTR3: &str = "CCCCCC";

/// Coarse enough to solve the whole grid in a test, fine enough to have
/// somewhere to go.
const RESOLUTION: usize = 11;

struct Fixture {
    params: Params,
    usage: CodonUsage,
    utr5: UtrSpec,
    utr3: UtrSpec,
    residues: usize,
}

impl Fixture {
    fn new() -> Self {
        Self {
            params: Params::load_default().expect("Turner 2004 parameters"),
            usage: CodonUsage::load_default().expect("the canonical human table"),
            utr5: UtrSpec::Fixed(UTR5.to_string()),
            utr3: UtrSpec::Fixed(UTR3.to_string()),
            residues: PROTEIN.chars().count(),
        }
    }

    fn inputs(&self, constraints: RegionConstraints) -> ProbeInputs<'_> {
        ProbeInputs {
            params: &self.params,
            protein: PROTEIN,
            usage: &self.usage,
            utr5: &self.utr5,
            utr3: &self.utr3,
            constraints,
            config: ScanConfig::default(),
        }
    }

    /// Design at `lambda` and verify it — one point of the axis, computed the
    /// long way so the probe has something independent to be checked against.
    fn step(&self, constraints: RegionConstraints, lambda: f64) -> ProbeStep {
        let design = design_with_utr_spec(
            &self.params,
            PROTEIN,
            &self.usage,
            lambda,
            &self.utr5,
            &self.utr3,
            constraints,
            ScanConfig::default(),
        )
        .expect("the design");
        let cds = CdsWindow::new(UTR5.len(), self.residues);
        let gap =
            sequence_gap(&self.params, &design.sequence, cds, constraints).expect("the verdict");
        ProbeStep {
            lambda,
            mfe_kcal_mol: design.mfe_kcal_mol,
            cai: design.cai,
            gap_kcal_mol: gap.gap_kcal_mol,
        }
    }
}

fn symmetric() -> RegionConstraints {
    RegionConstraints::symmetric_isolation(Enforcement::Forbid)
}

/// A λ the fixture is dirty at, so the probe has a reason to run.
const DIRTY_LAMBDA: f64 = 0.5;

#[test]
fn the_fixture_is_dirty_at_the_requested_lambda() {
    let f = Fixture::new();
    assert!(
        !f.step(symmetric(), DIRTY_LAMBDA).is_modular(),
        "the probe fixture must need a probe"
    );
}

#[test]
fn the_probe_returns_the_nearest_modular_lambda_on_the_grid() {
    let f = Fixture::new();
    let baseline = f.step(symmetric(), DIRTY_LAMBDA);
    let report = probe_lambda(&f.inputs(symmetric()), baseline, RESOLUTION).expect("the probe");

    // The same grid, walked the long way. The first modular point is what the
    // probe must have returned, and the points before it must all be dirty.
    let grid = probe_grid(DIRTY_LAMBDA, RESOLUTION);
    let expected = grid.iter().find(|&&l| f.step(symmetric(), l).is_modular());

    match (expected, report.rescued()) {
        (Some(&lambda), Some(rescue)) => {
            assert_eq!(
                rescue.lambda, lambda,
                "not the nearest modular λ on the grid"
            );
            assert!(rescue.is_modular(), "the rescue must verify modular");
            assert!(
                report.steps[..report.steps.len() - 1]
                    .iter()
                    .all(|s| !s.is_modular()),
                "the probe walked past a modular λ"
            );
        }
        (None, None) => assert_eq!(
            report.solves(),
            grid.len(),
            "a negative result must have covered the whole grid"
        ),
        (a, b) => panic!("the probe and the exhaustive walk disagree: {a:?} vs {b:?}"),
    }
}

#[test]
fn the_deltas_price_the_move_against_the_requested_design() {
    let f = Fixture::new();
    let baseline = f.step(symmetric(), DIRTY_LAMBDA);
    let report = probe_lambda(&f.inputs(symmetric()), baseline, RESOLUTION).expect("the probe");
    let Some(rescue) = report.rescued() else {
        assert_eq!(report.delta_mfe_kcal_mol(), None);
        assert_eq!(report.delta_cai(), None);
        return;
    };
    assert_eq!(
        report.delta_mfe_kcal_mol(),
        Some(rescue.mfe_kcal_mol - baseline.mfe_kcal_mol)
    );
    assert_eq!(report.delta_cai(), Some(rescue.cai - baseline.cai));
    // A design at another λ is a different design, so at least one of the two
    // trade-off terms has to move — a free rescue would mean the requested λ
    // was not optimal for its own objective.
    assert!(
        report.delta_mfe_kcal_mol() != Some(0.0) || report.delta_cai() != Some(0.0),
        "a rescue that costs nothing in either term is not a trade-off"
    );
}

/// The probe stops at the first modular point, so it must never solve the whole
/// grid when a rescue exists.
#[test]
fn the_probe_stops_at_the_first_modular_lambda() {
    let f = Fixture::new();
    let baseline = f.step(symmetric(), DIRTY_LAMBDA);
    let report = probe_lambda(&f.inputs(symmetric()), baseline, RESOLUTION).expect("the probe");
    assert_eq!(
        report.solves(),
        report.steps.len(),
        "one solve per step, and no step unaccounted for"
    );
    assert!(report.solves() <= probe_grid(DIRTY_LAMBDA, RESOLUTION).len());
    if report.rescue.is_some() {
        assert_eq!(
            report.rescue,
            Some(report.steps.len() - 1),
            "the rescue is the last step walked"
        );
    }
}

/// A modular design needs no rescue, so the probe must cost nothing at all —
/// which is what makes leaving it on for a whole sweep affordable.
#[test]
fn a_modular_baseline_costs_no_solves() {
    let f = Fixture::new();
    let inert = RegionConstraints::default();
    let baseline = f.step(inert, DIRTY_LAMBDA);
    assert!(
        baseline.is_modular(),
        "an inert table cannot produce a dirty verdict"
    );
    let report = probe_lambda(&f.inputs(inert), baseline, RESOLUTION).expect("the probe");
    assert_eq!(report.solves(), 0);
    assert_eq!(report.rescued(), None);
    assert!(report.steps.is_empty());
}
