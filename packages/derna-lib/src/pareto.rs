//! Convex-hull classification of a recovered Pareto front.
//!
//! The scalar λ-sweep, minimizing `λ·MFE − (1−λ)·CAI`, reaches exactly the
//! upper-left convex-hull vertices of the `(MFE↓, CAI↑)` plane; every other
//! non-dominated point is **off-hull** — Pareto-optimal but unreachable by any
//! linear scalarization, the points the front-valued DP recovers by
//! construction. This module partitions a front into on-/off-hull and measures
//! each off-hull point's vertical gap below the hull chord (the depth of the
//! non-convex region it sits in). Shared by the front-emitting binaries so the
//! hull semantics have one definition.

use crate::FrontPoint;

/// Slop for the convex-hull cross-product turn test (log-CAI · kcal/mol units).
const HULL_CROSS_EPS: f64 = 1e-12;

/// One front point's hull classification, aligned to the input order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontClassification {
    /// Whether the point is an upper-left convex-hull vertex — a point the
    /// scalar λ-sweep can reach.
    pub on_hull: bool,
    /// For an off-hull point, its vertical gap in log-CAI below the hull chord
    /// bracketing its MFE (`0.0` for on-hull points).
    pub gap_to_hull_cai_log: f64,
}

/// Classify every front point as on- or off-hull, returning classifications
/// aligned to `points`. Off-hull points carry their gap below the hull chord.
pub fn classify(points: &[FrontPoint]) -> Vec<FrontClassification> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &j| points[i].mfe_kcal_mol.total_cmp(&points[j].mfe_kcal_mol));
    let hull = upper_hull(points, &order);
    points
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let on_hull = hull.contains(&i);
            FrontClassification {
                on_hull,
                gap_to_hull_cai_log: if on_hull {
                    0.0
                } else {
                    hull_gap(points, &hull, i)
                },
            }
        })
        .collect()
}

/// The count of off-hull points — the headline recovery figure.
pub fn off_hull_count(classes: &[FrontClassification]) -> usize {
    classes.iter().filter(|c| !c.on_hull).count()
}

/// The upper-left convex-hull vertex indices of the `mfe`-ascending staircase
/// (Andrew's monotone chain in the `(MFE, CAI)` plane): the points a linear
/// scalarization `λ·MFE − (1−λ)·CAI` can be optimal at.
fn upper_hull(points: &[FrontPoint], order: &[usize]) -> Vec<usize> {
    let mut hull: Vec<usize> = Vec::new();
    for &i in order {
        while hull.len() >= 2 {
            let a = &points[hull[hull.len() - 2]];
            let b = &points[hull[hull.len() - 1]];
            let c = &points[i];
            let cross = (b.mfe_kcal_mol - a.mfe_kcal_mol) * (c.cai_log - a.cai_log)
                - (b.cai_log - a.cai_log) * (c.mfe_kcal_mol - a.mfe_kcal_mol);
            if cross >= -HULL_CROSS_EPS {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(i);
    }
    hull
}

/// The vertical gap in log-CAI of off-hull point `i` below the hull chord that
/// brackets its MFE — the depth of the non-convex region the point sits in.
fn hull_gap(points: &[FrontPoint], hull: &[usize], i: usize) -> f64 {
    let mfe = points[i].mfe_kcal_mol;
    let (mut lo, mut hi) = (hull[0], hull[hull.len() - 1]);
    for w in hull.windows(2) {
        if points[w[0]].mfe_kcal_mol <= mfe && mfe <= points[w[1]].mfe_kcal_mol {
            lo = w[0];
            hi = w[1];
        }
    }
    let (a, b) = (&points[lo], &points[hi]);
    if (b.mfe_kcal_mol - a.mfe_kcal_mol).abs() < 1e-9 {
        return 0.0;
    }
    let t = (mfe - a.mfe_kcal_mol) / (b.mfe_kcal_mol - a.mfe_kcal_mol);
    let chord_cai = a.cai_log + t * (b.cai_log - a.cai_log);
    (chord_cai - points[i].cai_log).max(0.0)
}
