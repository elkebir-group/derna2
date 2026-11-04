//! Brute-force oracle for designing under an energy ceiling.
//!
//! Asking for the best-adapted coding sequence that still folds at least as
//! stably as some threshold has an obvious answer available once the whole
//! Pareto front is in hand: truncate the staircase at the threshold and read
//! off the end. This harness proves that answer is the *right* one, by
//! enumerating every encoding of a small protein, folding each to its true
//! minimum free energy, filtering the bag to the ones that clear the threshold,
//! and asserting the surviving Pareto set is exactly what truncating
//! [`design_front`]'s output produces — at every threshold the instance admits,
//! not at a chosen few.
//!
//! That equality is what any faster route has to beat rather than reproduce, so
//! it is asserted here before anything is built against it. Two facts make it
//! non-obvious enough to be worth a test. Filtering and dominance-eliminating
//! commute for a **ceiling** but not for a floor — a point kept out by the
//! ceiling can never dominate one that is let in, because domination requires
//! the *lower* energy. And the answer under a ceiling routinely sits off the
//! convex hull of the front, where no scalarized sweep can reach it, so the
//! cheap substitute for this filter is not merely slower but wrong.
//!
//! The construction-known thresholds are asserted alongside: a threshold above
//! everything reproduces the unconstrained front, one at the achievable minimum
//! returns exactly the energy-corner design, and one below it returns nothing.
//! Those hold at any length, so they stay checkable where enumeration does not
//! reach.
//!
//! ## Tractability
//!
//! The reference enumerates *every* encoding, so it stays cheap only for short
//! proteins with a bounded encoding count (the product of per-residue codon
//! counts). The cases here keep that product small; the ceiling question is
//! about which points survive a filter, and a handful of points exercises it as
//! well as a thousand.

use derna_lib::{design_front, CodonUsage, Params};

mod common;
use common::{
    assert_same, best_adaptation, eliminate, enumerate_all, front_objectives, Objective, CAI_EPS,
    CASES, WIDE_CASES,
};

#[test]
fn truncating_the_front_returns_the_enumerated_answer_at_every_reachable_ceiling() {
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES {
        let bag = enumerate_all(&params, &usage, protein);
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        assert_same(
            &format!("{protein} unconstrained"),
            &eliminate(bag.clone()),
            &front,
        );

        // Every distinct energy on the front is a ceiling a caller could set,
        // and each has to be checked: the answer moves at exactly these values
        // and nowhere else, so a sampled ladder could step over a disagreement.
        for point in &front {
            let ceiling = point.mfe;
            let feasible: Vec<Objective> =
                bag.iter().copied().filter(|o| o.mfe <= ceiling).collect();
            let truncated: Vec<Objective> =
                front.iter().copied().filter(|o| o.mfe <= ceiling).collect();
            assert_same(
                &format!("{protein} at ceiling {ceiling}"),
                &eliminate(feasible.clone()),
                &truncated,
            );

            // The default deliverable is one design rather than the whole
            // feasible sub-front, so its own statement is asserted too: the
            // best-adapted feasible design is the end of the truncated
            // staircase.
            let enumerated = best_adaptation(&feasible);
            let read_off = best_adaptation(&truncated);
            assert!(
                (enumerated - read_off).abs() < CAI_EPS,
                "{protein} at ceiling {ceiling}: enumeration reaches {enumerated}, \
                 the truncated staircase reaches {read_off}",
            );
        }
    }
}

#[test]
fn the_construction_known_ceilings_hold() {
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let corner = front[0];
        let feasible = |ceiling: i32| -> Vec<Objective> {
            front.iter().copied().filter(|o| o.mfe <= ceiling).collect()
        };

        assert_eq!(
            feasible(front.last().unwrap().mfe + 1),
            front,
            "{protein}: a ceiling above everything must reproduce the whole front",
        );
        assert_eq!(
            feasible(corner.mfe),
            vec![corner],
            "{protein}: a ceiling at the achievable minimum must return exactly the \
             energy-corner design",
        );
        assert!(
            feasible(corner.mfe - 1).is_empty(),
            "{protein}: a ceiling below the achievable minimum must return nothing",
        );
    }
}

#[test]
fn a_ceiling_reaches_designs_no_scalarized_sweep_can() {
    // The whole reason the filter has to run over the exact front rather than
    // over a sweep's output: at some ceilings the best feasible design is not a
    // vertex of the front's lower convex hull, and no weighting of the two
    // objectives selects it. Hull membership is taken in the space the sweep
    // actually scalarizes, which is log-adaptation, not adaptation.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    let mut found = 0;
    for protein in WIDE_CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let hull = lower_hull(&front);
        for point in &front {
            let best_feasible = front
                .iter()
                .filter(|o| o.mfe <= point.mfe)
                .max_by(|a, b| a.cai_log.total_cmp(&b.cai_log))
                .unwrap();
            let sweep_best = hull
                .iter()
                .filter(|o| o.mfe <= point.mfe)
                .map(|o| o.cai_log)
                .fold(f64::NEG_INFINITY, f64::max);
            if best_feasible.cai_log > sweep_best + CAI_EPS {
                found += 1;
            }
        }
    }
    assert!(
        found > 0,
        "no ceiling on any case isolated an off-hull answer — either the cases lost \
         their non-convexity or the hull is being taken in the wrong space, and \
         either way the filter is no longer being tested against the thing it beats",
    );
}

/// The vertices of the front's lower convex hull in `(energy, log-adaptation)`
/// — exactly the points a scalarized sweep can return.
fn lower_hull(front: &[Objective]) -> Vec<Objective> {
    let mut hull: Vec<Objective> = Vec::new();
    for &p in front {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            // Keep `b` only if the turn from `a` to `b` to `p` is convex.
            let cross = (f64::from(b.mfe - a.mfe)) * (p.cai_log - a.cai_log)
                - (f64::from(p.mfe - a.mfe)) * (b.cai_log - a.cai_log);
            if cross >= 0.0 {
                break;
            }
            hull.pop();
        }
        hull.push(p);
    }
    hull
}
