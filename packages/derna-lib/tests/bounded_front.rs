//! The exactness proof for designing under an energy ceiling with the ceiling
//! carried *into* the fold.
//!
//! Its sibling [`front_ceiling_oracle`] proves that truncating a complete front
//! at a threshold returns what enumeration returns. That settles what the answer
//! is; it says nothing about a fold that discards derivations before the front
//! is complete. This file is about that fold: the same enumeration, compared
//! against a solver that never materializes the points above the ceiling.
//!
//! # Why the interesting test here is a mutation
//!
//! The drop is exact because a cell's completion cost is credited
//! `optimum - cell_min`, which no completion can beat (`derna_lib::bounds`
//! carries the argument). That argument is a consequence of the decomposition
//! the fold already relies on, so there is no instance on which it fails and no
//! case that can be constructed to discriminate it — a test asserting the credit
//! never exceeds the true cost-to-go would restate the theorem and pass
//! whatever the code did.
//!
//! What *can* break is a precondition, and the load-bearing one is that the
//! achievable minimum the credit is stated against is no higher than the true
//! one. So the discriminating test inflates the credit deliberately, by the
//! smallest amount the energy representation admits, and asserts the answer
//! changes. That is not an artificial mutation: adding to the credit is
//! arithmetically identical to the corner coming back too high, which is the one
//! way this can go wrong in practice.
//!
//! Coverage is asserted alongside, because a mutation test can be killed at a
//! site that was never in doubt. An estimate built by decomposing the outside
//! into a left piece and a right piece is inadmissible precisely at the cells
//! inside a multi-loop, and sound at the exterior; a drop exercised only at the
//! exterior would demonstrate nothing about the tables where the difficulty
//! lives. So the tests check the drop reaches all four multiloop-interior
//! tables on the cases they run.

use derna_lib::{
    design_bounded, design_front, design_front_bounded, Bounded, BoundedFront, Bounds, Ceiling,
    CellTable, CodonUsage, DropTally, Energy, Params, Prune,
};

mod common;
use common::{
    assert_same, best_adaptation, eliminate, enumerate_all, front_objectives, refold, Objective,
    CAI_EPS, CASES, WIDE_CASES,
};

/// Solve under `ceiling` and unwrap the feasible answer, failing loudly if the
/// ceiling turned out to be unreachable.
fn solve(params: &Params, protein: &str, usage: &CodonUsage, ceiling: Ceiling) -> BoundedFront {
    solve_with(params, protein, usage, Bounds::with_ceiling(ceiling), 1)
}

fn solve_with(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    bounds: Bounds,
    threads: usize,
) -> BoundedFront {
    match design_front_bounded(params, protein, usage, Prune::None, bounds, threads).unwrap() {
        Bounded::Feasible(front) => front,
        Bounded::Unreachable {
            ceiling_kcal_mol,
            achievable_kcal_mol,
        } => panic!(
            "{protein}: ceiling {ceiling_kcal_mol} reported unreachable against an \
             achievable {achievable_kcal_mol}"
        ),
    }
}

/// Every energy on the unconstrained front — each one a ceiling a caller could
/// set, and the only values at which the answer moves.
fn reachable_ceilings(params: &Params, protein: &str, usage: &CodonUsage) -> Vec<i32> {
    front_objectives(&design_front(params, protein, usage).unwrap())
        .iter()
        .map(|o| o.mfe)
        .collect()
}

#[test]
fn the_bounded_fold_reproduces_enumeration_at_every_reachable_ceiling() {
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES {
        let bag = enumerate_all(&params, &usage, protein);
        for sigma in reachable_ceilings(&params, protein, &usage) {
            let expected = eliminate(bag.iter().copied().filter(|o| o.mfe <= sigma).collect());
            let got = solve(&params, protein, &usage, Ceiling::Absolute(Energy(sigma)));
            assert_same(
                &format!("{protein} sub-front at ceiling {sigma}"),
                &expected,
                &front_objectives(&got.front),
            );

            // The default deliverable is one design, so it carries its own
            // assertion rather than being inferred from the sub-front.
            let single =
                match design_bounded(&params, protein, &usage, Ceiling::Absolute(Energy(sigma)))
                    .unwrap()
                {
                    Bounded::Feasible(point) => point,
                    Bounded::Unreachable { .. } => panic!("{protein} at {sigma}: unreachable"),
                };
            let enumerated = best_adaptation(&expected);
            assert!(
                (single.cai_log - enumerated).abs() < CAI_EPS,
                "{protein} at ceiling {sigma}: solver reaches {}, enumeration reaches \
                 {enumerated}",
                single.cai_log,
            );
            assert!(
                (single.mfe_kcal_mol * 100.0).round() as i32 <= sigma,
                "{protein} at ceiling {sigma}: the returned design folds to {}",
                single.mfe_kcal_mol,
            );
        }
    }
}

#[test]
fn an_unbounded_box_reproduces_the_unconstrained_fold_byte_for_byte() {
    // A box that constrains nothing must leave the fold exactly as it found it,
    // and "exactly" means every recovered field rather than the objectives
    // alone: the whole feasibility apparatus has to be provably inert before any
    // answer it changes can be trusted.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let plain = design_front(&params, protein, &usage).unwrap();
        let bounded = solve_with(&params, protein, &usage, Bounds::UNBOUNDED, 1);
        assert_eq!(
            bounded.ceiling_kcal_mol, None,
            "{protein}: an unbounded fold must report no ceiling",
        );
        assert_eq!(
            bounded.dropped,
            DropTally::default(),
            "{protein}: an unbounded fold must discard nothing",
        );
        assert_eq!(
            plain, bounded.front,
            "{protein}: the unbounded box changed the recovered front",
        );
    }
}

#[test]
fn the_construction_known_ceilings_hold_for_the_solver() {
    // These follow from the construction rather than from enumeration, so they
    // stay checkable at lengths no oracle reaches.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let corner = front[0].mfe;
        let highest = front.last().unwrap().mfe;

        let everything = solve(&params, protein, &usage, Ceiling::Absolute(Energy(highest)));
        assert_same(
            &format!("{protein} at a ceiling above everything"),
            &front,
            &front_objectives(&everything.front),
        );

        let at_corner = solve(&params, protein, &usage, Ceiling::Absolute(Energy(corner)));
        assert_same(
            &format!("{protein} at the achievable minimum"),
            &front[..1],
            &front_objectives(&at_corner.front),
        );

        let below = design_front_bounded(
            &params,
            protein,
            &usage,
            Prune::None,
            Bounds::with_ceiling(Ceiling::Absolute(Energy(corner - 1))),
            1,
        )
        .unwrap();
        match below {
            Bounded::Unreachable {
                ceiling_kcal_mol,
                achievable_kcal_mol,
            } => {
                assert_eq!(ceiling_kcal_mol, Energy(corner - 1).to_kcal_mol());
                assert_eq!(
                    achievable_kcal_mol,
                    Energy(corner).to_kcal_mol(),
                    "{protein}: an unreachable ceiling must report what is reachable",
                );
            }
            Bounded::Feasible(_) => {
                panic!("{protein}: a ceiling below the achievable minimum returned designs")
            }
        }
    }
}

#[test]
fn the_corner_the_credit_is_built_on_is_the_front_the_fold_recovers() {
    // The credit is admissible only while the achievable minimum it is stated
    // against is no higher than the true one, and the two come from different
    // implementations — a scalar solve for the corner, the front fold for
    // everything else. Every bounded solve checks the two agree; this asserts
    // the check is satisfied rather than merely present, at a ceiling wide
    // enough that nothing else could be keeping the corner in the answer.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let solved = solve(
            &params,
            protein,
            &usage,
            Ceiling::Absolute(Energy(front.last().unwrap().mfe)),
        );
        assert_eq!(
            (solved.optimum_kcal_mol * 100.0).round() as i32,
            front[0].mfe,
            "{protein}: the corner disagrees with the front's first point",
        );
    }
}

#[test]
fn a_relative_ceiling_names_the_same_answer_as_the_absolute_one_it_resolves_to() {
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let corner = front[0].mfe;
        for point in &front {
            let offset = point.mfe - corner;
            let relative = solve(
                &params,
                protein,
                &usage,
                Ceiling::AboveOptimum(Energy(offset)),
            );
            let absolute = solve(
                &params,
                protein,
                &usage,
                Ceiling::Absolute(Energy(point.mfe)),
            );
            assert_eq!(
                relative.ceiling_kcal_mol, absolute.ceiling_kcal_mol,
                "{protein}: the two forms resolved to different ceilings",
            );
            assert_same(
                &format!("{protein} at an offset of {offset}"),
                &front_objectives(&absolute.front),
                &front_objectives(&relative.front),
            );
        }
    }
}

#[test]
fn the_drop_reaches_the_tables_inside_a_multi_loop() {
    // The shipped rule never decomposes the outside, so it has no
    // exterior-versus-interior asymmetry; this is what that claim looks like as
    // a measurement on the fold rather than an argument about it.
    //
    // It is a coverage statement and only that. Firing at a table means the rule
    // was exercised there, not that the answer depends on it — on some instances
    // the multi-loop tables can be narrowed to a single point each without the
    // recovered front moving. Whether an inadmissible credit at those tables is
    // *detectable* is a separate question, asked by the mutation test below over
    // its own case list.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    let interior = [
        CellTable::Multi,
        CellTable::MultiOpen,
        CellTable::Branch,
        CellTable::Branch2,
    ];
    for protein in CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        // The tightest reachable ceiling: the corner itself, where the drop has
        // the most to remove and the exterior table the least to contribute.
        let solved = solve(
            &params,
            protein,
            &usage,
            Ceiling::Absolute(Energy(front[0].mfe)),
        );
        for table in interior {
            assert!(
                solved.dropped.at(table) > 0,
                "{protein}: the drop removed nothing at `{}`, so it was never \
                 exercised where a decomposed completion estimate would fail",
                table.name(),
            );
        }
        assert!(solved.dropped.multiloop_interior() > 0);
    }
}

/// A ceiling at the front's second point, with the window it implies.
///
/// The point at the ceiling is feasible by construction and sits exactly on the
/// boundary, and there is one below it, so the fold is not reduced to the corner
/// alone. `None` for a front too narrow to offer such a ceiling.
fn boundary_ceiling(front: &[Objective]) -> Option<(Ceiling, i32, i32)> {
    let boundary = front.get(1)?;
    let window = boundary.mfe - front[0].mfe;
    Some((
        Ceiling::Absolute(Energy(boundary.mfe)),
        boundary.mfe,
        window,
    ))
}

#[test]
fn a_credit_inflated_by_one_unit_discards_a_design_the_oracle_proves_feasible() {
    // The admissibility test at its sharpest. The shipped credit cannot be
    // refuted on an instance — it is a theorem — so what is tested is that the
    // comparison would notice if it stopped holding. One dekacalorie is the
    // smallest inadmissibility the energy representation can express, and a
    // ceiling pinned exactly on a front point's own energy puts that point on
    // the boundary, so a credit inflated by one unit must discard it.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    let mut exercised = 0;

    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let Some((ceiling, boundary, _)) = boundary_ceiling(&front) else {
            continue;
        };

        let modular = solve(&params, protein, &usage, ceiling);
        assert!(
            front_objectives(&modular.front)
                .iter()
                .any(|o| o.mfe == boundary),
            "{protein}: the design on the boundary is missing from the modular answer",
        );

        let inflated = solve_with(
            &params,
            protein,
            &usage,
            Bounds::with_ceiling(ceiling).with_credit_offset(Energy(1)),
            1,
        );
        assert!(
            !front_objectives(&inflated.front)
                .iter()
                .any(|o| o.mfe == boundary),
            "{protein}: a credit inflated past the true cost-to-go kept the design on \
             the boundary, so this comparison would not notice an inadmissible bound",
        );
        assert!(
            inflated.dropped.total() > modular.dropped.total(),
            "{protein}: the inflated credit discarded no more than the modular one",
        );

        // That the discarded design is feasible is established by enumeration,
        // not by the front the solver under test produced.
        if CASES.contains(protein) {
            let bag = enumerate_all(&params, &usage, protein);
            assert!(
                bag.iter().any(|o| o.mfe == boundary),
                "{protein}: no encoding folds to {boundary}, so the discarded point was \
                 not a feasible design",
            );
        }
        exercised += 1;
    }

    assert!(
        exercised >= 4,
        "the mutation was only exercised on {exercised} cases — too few for the absence \
         of a failure to mean anything",
    );
}

/// Cases whose answer is *carried* by cells inside the fold, not merely
/// computed alongside them.
///
/// Every case populates the multi-loop tables and has points discarded from
/// them, but that is a weaker property than it sounds: on some instances no
/// front point's derivation passes through a multi-loop cell whose window
/// binds, so those tables can be pruned to a single point each without the
/// answer moving at all. `MWWCCGGAAGGCCWWM` is such an instance — narrowing
/// every non-exterior window to zero leaves its answer untouched — and a
/// sensitivity test written over it would pass whatever the credit did there.
/// The cases below are the ones where the interior is load-bearing, which is
/// the property the test needs and the reason the list is named rather than
/// inherited from [`CASES`].
const INTERIOR_LOAD_BEARING: [&str; 3] = ["MCWFGGWCKM", "MYGKIIFVLL", "MYGKIIFVLLL"];

#[test]
fn the_comparison_notices_a_credit_that_is_modular_only_at_the_exterior() {
    // The hazard this test targets is a completion estimate that
    // is exactly valid where a cell is definitionally an exterior branch and
    // inadmissible at the five tables where it is not — so a mutation applied
    // everywhere does not reproduce its shape, and one killed at the exterior
    // says nothing about the tables where the front's cost is generated. This
    // mutation is confined to those five: the exterior window stays modular, so
    // any change in the answer is attributable to a cell inside the fold.
    //
    // One dekacalorie is the smallest inadmissibility the energy representation
    // can express, and it is enough on every case here.
    //
    // Sensitivity is asserted as a change in the answer, never as a change in
    // the drop counts — not even at the exterior, whose window this mutation
    // leaves alone. A tally counts removals, and removals upstream leave fewer
    // points to remove downstream, so every table's count moves under a
    // mutation confined to five of them. That the confinement holds is a
    // property of the resolved windows, asserted where they are resolved.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();

    for protein in INTERIOR_LOAD_BEARING {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let (ceiling, boundary, _) = boundary_ceiling(&front)
            .unwrap_or_else(|| panic!("{protein}: front too narrow to pin a ceiling on"));
        let reference: Vec<Objective> = front
            .iter()
            .copied()
            .filter(|o| o.mfe <= boundary)
            .collect();

        let modular = solve(&params, protein, &usage, ceiling);
        assert_same(
            &format!("{protein} modular arm"),
            &reference,
            &front_objectives(&modular.front),
        );

        let inflated = solve_with(
            &params,
            protein,
            &usage,
            Bounds::with_ceiling(ceiling).with_credit_offset_beyond_exterior(Energy(1)),
            1,
        );
        assert_ne!(
            front_objectives(&inflated.front),
            reference,
            "{protein}: a credit inadmissible by one dekacalorie at every table but the \
             exterior left the answer unchanged, so this comparison cannot tell a bound \
             sound only at the exterior from one sound everywhere",
        );
    }
}

#[test]
fn worker_count_does_not_change_the_bounded_answer() {
    // The drop is applied to cells that are still accumulating, so the order
    // merges arrive in could in principle decide what survives. It cannot: a
    // partially accumulated cell has a higher best point and so a wider window,
    // and anything a wider window excludes a narrower one excludes too.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in WIDE_CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        for point in &front {
            let bounds = Bounds::with_ceiling(Ceiling::Absolute(Energy(point.mfe)));
            let serial = solve_with(&params, protein, &usage, bounds, 1);
            let parallel = solve_with(&params, protein, &usage, bounds, 8);
            assert_eq!(
                serial.front, parallel.front,
                "{protein} at ceiling {}: eight workers reached a different answer",
                point.mfe,
            );
            assert_eq!(serial.dropped, parallel.dropped);
        }
    }
}

#[test]
fn every_feasible_point_carries_a_traceback_that_folds_to_its_reported_energy() {
    // A front point is only an answer if the sequence attached to it is one.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    for protein in CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let solved = solve(
            &params,
            protein,
            &usage,
            Ceiling::Absolute(Energy(front.last().unwrap().mfe)),
        );
        for point in &solved.front {
            assert_eq!(
                point.sequence.len(),
                protein.len() * 3,
                "{protein}: a recovered sequence is the wrong length",
            );
            assert_eq!(
                refold(&params, protein, &point.sequence),
                (point.mfe_kcal_mol * 100.0).round() as i32,
                "{protein}: the recovered sequence {} does not fold to its reported energy",
                point.sequence,
            );
        }
    }
}

#[test]
fn a_ceiling_recovers_an_answer_no_scalarized_sweep_reaches() {
    // The reason any of this exists: under some ceilings the best feasible
    // design is not a vertex of the front's lower convex hull, so no weighting
    // of the two objectives selects it, and the constrained solver has to return
    // it anyway.
    let usage = CodonUsage::load_default().unwrap();
    let params = Params::load_default().unwrap();
    let mut recovered = 0;
    for protein in WIDE_CASES {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let hull = lower_hull(&front);
        for point in &front {
            let sweep_best = hull
                .iter()
                .filter(|o| o.mfe <= point.mfe)
                .map(|o| o.cai_log)
                .fold(f64::NEG_INFINITY, f64::max);
            let solved = match design_bounded(
                &params,
                protein,
                &usage,
                Ceiling::Absolute(Energy(point.mfe)),
            )
            .unwrap()
            {
                Bounded::Feasible(point) => point,
                Bounded::Unreachable { .. } => panic!("{protein}: unreachable"),
            };
            if solved.cai_log > sweep_best + CAI_EPS {
                recovered += 1;
            }
        }
    }
    assert!(
        recovered > 0,
        "no ceiling on any case produced an answer the hull cannot reach, so the \
         solver was never tested against the thing it exists to beat",
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
