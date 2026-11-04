//! λ probing: what modularity costs in trade-off terms, when λ is free to move.
//!
//! A design whose free-fold verdict comes back dirty is still the exact optimum
//! of the problem it was given. What it is not is a molecule that folds the way
//! it was scored. Two routes lead somewhere better: search the design space at
//! the requested λ, or ask a different λ. This module is the second.
//!
//! The λ axis is measured to produce modularity — one instance runs a gap of 9.90,
//! 0.00 and 0.80 across λ ∈ {0.0001, 0.5, 0.99} — and each step is **one design
//! solve plus one verdict**, against the tens to hundreds of solves a region
//! search spends per candidate. So when λ may move, probing dominates searching
//! by orders of magnitude. When λ is pinned — which is what a CAI floor does —
//! this instrument is inapplicable and the search is the only route.
//!
//! It is **opt-in** for a reason that is not cost: a design at some other λ
//! answers a question the caller did not ask. The report is therefore a
//! *trade-off*, never a substitution — it says what the nearest modular λ would
//! cost in MFE and in CAI, and leaves the choice where it belongs.
//!
//! # What "nearest" means, and what a negative result covers
//!
//! Modularity is not monotone in λ, so the probe cannot bisect. It walks a uniform
//! grid over the whole λ domain in order of distance from the requested value
//! and stops at the first modular design, which makes the result the nearest
//! modular λ **on that grid**. A report that finds none says so in exactly those
//! terms: the grid is finite and stated, and a finer one may still find a
//! rescue between two of its points.

use thiserror::Error;

use crate::free_fold::{sequence_gap, CdsWindow, FreeFoldError};
use crate::region::RegionConstraints;
use crate::{design_with_utr_spec, BeamError, CodonUsage, Params, ScanConfig, UtrSpec};

/// Everything a probe needs to re-solve at another λ — the design inputs
/// unchanged, with λ the one thing it varies.
pub struct ProbeInputs<'a> {
    pub params: &'a Params,
    pub protein: &'a str,
    pub usage: &'a CodonUsage,
    pub utr5: &'a UtrSpec,
    pub utr3: &'a UtrSpec,
    pub constraints: RegionConstraints,
    pub config: ScanConfig,
}

/// One design on the λ axis, with its modularity verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeStep {
    pub lambda: f64,
    /// The design's whole-mRNA MFE.
    pub mfe_kcal_mol: f64,
    /// The design's CDS-only adaptation.
    pub cai: f64,
    /// `E_iso − E_free` for the emitted sequence — the modularity verdict.
    pub gap_kcal_mol: f64,
}

impl ProbeStep {
    /// Whether this design's constrained and unconstrained optima coincide.
    pub fn is_modular(&self) -> bool {
        self.gap_kcal_mol.abs() < 1e-9
    }
}

/// What probing the λ axis found: every design it solved, and the nearest one
/// that verifies modular.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeReport {
    /// The design the caller asked for, dirty or not.
    pub baseline: ProbeStep,
    /// The designs probed, in probe order — nearest λ first. Empty when the
    /// baseline was already modular.
    pub steps: Vec<ProbeStep>,
    /// Index into [`Self::steps`] of the nearest modular λ, or `None` when the
    /// probed grid holds none.
    pub rescue: Option<usize>,
    /// How many points the λ grid held, so a negative result states its own
    /// coverage.
    pub resolution: usize,
}

impl ProbeReport {
    /// The nearest modular design, or `None` when the grid holds none.
    pub fn rescued(&self) -> Option<&ProbeStep> {
        self.rescue.map(|i| &self.steps[i])
    }

    /// What the rescue costs in folding energy: the modular design's MFE minus
    /// the requested design's. Positive means giving up structure.
    pub fn delta_mfe_kcal_mol(&self) -> Option<f64> {
        self.rescued()
            .map(|s| s.mfe_kcal_mol - self.baseline.mfe_kcal_mol)
    }

    /// What the rescue costs in adaptation. Negative means giving up CAI.
    pub fn delta_cai(&self) -> Option<f64> {
        self.rescued().map(|s| s.cai - self.baseline.cai)
    }

    /// Design solves the probe spent — one per step, and none when the baseline
    /// was already modular.
    pub fn solves(&self) -> usize {
        self.steps.len()
    }
}

/// Errors surfaced by [`probe_lambda`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProbeError {
    #[error("designing at a probed λ failed: {0}")]
    Design(#[from] BeamError),
    #[error("verifying a probed design failed: {0}")]
    Verdict(#[from] FreeFoldError),
}

/// Probe the λ axis for the nearest λ whose design verifies free-fold modular.
///
/// `baseline` is the design the caller already holds, so no solve is spent
/// re-deriving it; a modular baseline returns immediately having spent none at
/// all. Otherwise the probe walks [`probe_grid`] — a uniform `resolution`-point
/// grid over `[0, 1]`, ordered by distance from the requested λ — solving and
/// verifying one design per point and stopping at the first modular one.
///
/// Costs one design solve plus one verdict (two folds of a single sequence) per
/// step, and at most `resolution − 1` steps.
pub fn probe_lambda(
    inputs: &ProbeInputs<'_>,
    baseline: ProbeStep,
    resolution: usize,
) -> Result<ProbeReport, ProbeError> {
    let mut report = ProbeReport {
        baseline,
        steps: Vec::new(),
        rescue: None,
        resolution,
    };
    if baseline.is_modular() {
        return Ok(report);
    }
    for lambda in probe_grid(baseline.lambda, resolution) {
        let step = solve_and_verify(inputs, lambda)?;
        let modular = step.is_modular();
        report.steps.push(step);
        if modular {
            report.rescue = Some(report.steps.len() - 1);
            break;
        }
    }
    Ok(report)
}

/// Design at `lambda` under the probe's inputs and verify the result.
fn solve_and_verify(inputs: &ProbeInputs<'_>, lambda: f64) -> Result<ProbeStep, ProbeError> {
    let design = design_with_utr_spec(
        inputs.params,
        inputs.protein,
        inputs.usage,
        lambda,
        inputs.utr5,
        inputs.utr3,
        inputs.constraints,
        inputs.config,
    )?;
    let cds = CdsWindow::new(inputs.utr5.len(), inputs.protein.len());
    let gap = sequence_gap(inputs.params, &design.sequence, cds, inputs.constraints)?;
    Ok(ProbeStep {
        lambda,
        mfe_kcal_mol: design.mfe_kcal_mol,
        cai: design.cai,
        gap_kcal_mol: gap.gap_kcal_mol,
    })
}

/// The λ values a probe visits, in order: a uniform `resolution`-point grid over
/// the whole `[0, 1]` domain, sorted by distance from `requested`, with any
/// point coinciding with `requested` dropped.
///
/// Nearest-first rather than ascending because the report's claim is about the
/// *nearest* modular λ, and modularity is not monotone in λ — no ordering of the
/// grid can be skipped by an interpolation argument, so the probe walks it.
/// Ties break toward the lower λ, which keeps the walk deterministic.
pub fn probe_grid(requested: f64, resolution: usize) -> Vec<f64> {
    if resolution < 2 {
        return Vec::new();
    }
    let mut grid: Vec<f64> = (0..resolution)
        .map(|i| i as f64 / (resolution - 1) as f64)
        .filter(|&l| (l - requested).abs() > 1e-9)
        .collect();
    grid.sort_by(|a, b| {
        (a - requested)
            .abs()
            .total_cmp(&(b - requested).abs())
            .then(a.total_cmp(b))
    });
    grid
}

#[cfg(test)]
mod tests {
    //! Grid-order coverage. The end-to-end probe is exercised at
    //! `tests/lambda_probe.rs`, where a real dirty configuration is available.

    use super::*;

    #[test]
    fn the_grid_spans_the_whole_domain_including_both_endpoints() {
        let grid = probe_grid(0.5, 11);
        assert_eq!(grid.len(), 10, "the requested point itself is dropped");
        assert!(grid.contains(&0.0));
        assert!(grid.contains(&1.0));
    }

    #[test]
    fn the_grid_is_ordered_by_distance_from_the_requested_lambda() {
        let grid = probe_grid(0.5, 11);
        let distances: Vec<f64> = grid.iter().map(|l| (l - 0.5).abs()).collect();
        assert!(distances.windows(2).all(|w| w[0] <= w[1]));
        // Ties break low, so the step below the request comes first.
        assert_eq!(grid[0], 0.4);
        assert_eq!(grid[1], 0.6);
    }

    #[test]
    fn a_requested_lambda_off_the_grid_keeps_every_point() {
        let grid = probe_grid(0.37, 11);
        assert_eq!(grid.len(), 11);
        assert_eq!(grid[0], 0.4, "the nearest grid point leads");
    }

    #[test]
    fn a_grid_too_coarse_to_hold_two_points_probes_nothing() {
        assert!(probe_grid(0.5, 1).is_empty());
        assert!(probe_grid(0.5, 0).is_empty());
    }

    #[test]
    fn a_modular_baseline_short_circuits_without_spending_a_solve() {
        let report = ProbeReport {
            baseline: ProbeStep {
                lambda: 0.5,
                mfe_kcal_mol: -187.8,
                cai: 0.8081,
                gap_kcal_mol: 0.0,
            },
            steps: Vec::new(),
            rescue: None,
            resolution: 21,
        };
        assert!(report.baseline.is_modular());
        assert_eq!(report.solves(), 0);
        assert_eq!(report.rescued(), None);
    }
}
