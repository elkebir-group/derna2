//! The exact cost to go, credited inside the bounded fold.
//!
//! [`bounded_front`] proves the window rule exact. This file proves the same
//! things of the fold with each cell's true cost to go credited beside the
//! window ([`Bounds::with_exact_credit`]), and one thing the window rule cannot
//! have: cells emptied outright.
//!
//! Both credits are admissible, so the answer may not move. What the exact
//! credit changes is *what is discarded*, and the tests read that directly — a
//! credit that changed nothing would pass every answer test here and be worth
//! nothing, so the emptied-cell tally is asserted to fire, and to fire inside
//! the multi-loop tables where the fold's cost is generated.
//!
//! # What "the same answer" means here
//!
//! The same `(MFE, CAI)` points, each carrying a traceback that is a real
//! answer — **not** the same bytes. An emptied cell is removed from its column
//! so nothing downstream composes it, and removing a key changes the column's
//! iteration order, which decides which of two *tied* derivations a cell keeps.
//! Measured: on one flanked case two points came back with the same sequence,
//! energy and CAI and a different one of two equal-energy structures; keeping
//! the emptied entries instead made the two arms byte-identical, so the flip is
//! the order and not a dropped derivation. Keeping them would also keep every
//! composition the lever exists to remove.

use derna_lib::{
    design_front, design_front_bounded, design_front_bounded_with_utr, Bounded, BoundedFront,
    Bounds, Ceiling, CellTable, CodonUsage, DropTally, Energy, FrontPoint, Params, Prune,
    UtrProfile, UtrSpec,
};

mod common;
use common::{
    assert_same, eliminate, enumerate_all, front_objectives, Objective, CASES, WIDE_CASES,
};

/// The cases whose answer is carried by the multi-loop interior, measured in
/// `bounded_front.rs`: narrowing every non-exterior window there moves the
/// answer. A mutation test confined to those tables is only meaningful here.
const INTERIOR_LOAD_BEARING: [&str; 3] = ["MCWFGGWCKM", "MYGKIIFVLL", "MYGKIIFVLLL"];

fn fixtures() -> (Params, CodonUsage) {
    (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    )
}

fn solve(params: &Params, protein: &str, usage: &CodonUsage, bounds: Bounds) -> BoundedFront {
    solve_threaded(params, protein, usage, bounds, 1)
}

fn solve_threaded(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    bounds: Bounds,
    threads: usize,
) -> BoundedFront {
    match design_front_bounded(params, protein, usage, Prune::None, bounds, threads).unwrap() {
        Bounded::Feasible(front) => front,
        Bounded::Unreachable { .. } => panic!("{protein}: expected a feasible answer"),
    }
}

/// Every energy on the unconstrained front — the only ceilings at which the
/// answer moves.
fn reachable_ceilings(params: &Params, protein: &str, usage: &CodonUsage) -> Vec<i32> {
    front_objectives(&design_front(params, protein, usage).unwrap())
        .iter()
        .map(|o| o.mfe)
        .collect()
}

fn exact(sigma: i32) -> Bounds {
    Bounds::with_ceiling(Ceiling::Absolute(Energy(sigma))).with_exact_credit()
}

fn window(sigma: i32) -> Bounds {
    Bounds::with_ceiling(Ceiling::Absolute(Energy(sigma)))
}

#[test]
fn the_exact_credit_reproduces_enumeration_at_every_reachable_ceiling() {
    let (params, usage) = fixtures();
    for protein in CASES {
        let bag = enumerate_all(&params, &usage, protein);
        for sigma in reachable_ceilings(&params, protein, &usage) {
            let expected = eliminate(bag.iter().copied().filter(|o| o.mfe <= sigma).collect());
            let got = solve(&params, protein, &usage, exact(sigma));
            assert_same(
                &format!("{protein} sub-front at ceiling {sigma} under the exact credit"),
                &expected,
                &front_objectives(&got.front),
            );
        }
    }
}

/// Assert `got` is the same answer as `want`: the same objectives, point for
/// point, and every point's own sequence and structure evaluating to the energy
/// it reports — so a tie resolved differently is still a real answer.
fn assert_same_answer(name: &str, params: &Params, want: &[FrontPoint], got: &[FrontPoint]) {
    assert_eq!(
        front_objectives(want),
        front_objectives(got),
        "{name}: the exact credit changed the objectives",
    );
    for point in got {
        let reported = (point.mfe_kcal_mol * 100.0).round() as i32;
        let evaluated = rna_mfe::evaluate(params, point.sequence.as_bytes(), &point.structure)
            .unwrap_or_else(|e| panic!("{name}: {} does not evaluate: {e:?}", point.structure));
        assert_eq!(
            evaluated.0, reported,
            "{name}: {} on {} evaluates to {}, not the reported {reported}",
            point.structure, point.sequence, evaluated.0,
        );
    }
}

/// The two credits reach the same answer, in the sense of
/// [`assert_same_answer`], and the window alone never empties a cell.
#[test]
fn the_exact_credit_changes_what_is_discarded_and_not_the_answer() {
    let (params, usage) = fixtures();
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        for sigma in reachable_ceilings(&params, protein, &usage) {
            let by_window = solve(&params, protein, &usage, window(sigma));
            let by_exact = solve(&params, protein, &usage, exact(sigma));
            assert_same_answer(
                &format!("{protein} at ceiling {sigma}"),
                &params,
                &by_window.front,
                &by_exact.front,
            );
            assert_eq!(by_window.optimum_kcal_mol, by_exact.optimum_kcal_mol);
            assert_eq!(
                by_window.dropped.emptied_total(),
                0,
                "{protein} at {sigma}: the window alone emptied a cell, which the \
                 window rule's own argument says it cannot",
            );
        }
    }
}

/// The lever the window cannot pull, asserted to fire where it was measured to.
///
/// At the corner ceiling the credit empties **663–2,511 closed-pair cells** per
/// case and 4–9 `M` cells. `M2`, `Multi` and the shared openings read **zero
/// emptied and zero dropped** there, and that is the lever working rather than
/// missing: the pairs and branches that would have fed them were emptied first,
/// so nothing infeasible ever reaches those tables. An assertion over `M2` or
/// `Multi` would therefore fail exactly when the credit bites hardest — a tally
/// counts what arrives, and a cut upstream leaves less to arrive.
#[test]
fn the_exact_credit_empties_cells_the_window_keeps() {
    let (params, usage) = fixtures();
    let mut branches_emptied = 0;
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let corner = reachable_ceilings(&params, protein, &usage)[0];
        let solved = solve(&params, protein, &usage, exact(corner));
        assert!(
            solved.dropped.emptied_at(CellTable::Pair) > 0,
            "{protein}: the exact credit emptied no closed pair at the corner, so it \
             pulled no lever the window does not",
        );
        branches_emptied += solved.dropped.emptied_at(CellTable::Branch);
    }
    assert!(
        branches_emptied > 0,
        "the exact credit never emptied a multi-branch component on any case",
    );
}

/// A ceiling at the front's second point: the point there is feasible by
/// construction and sits on the boundary.
fn boundary_ceiling(front: &[Objective]) -> Option<(i32, Vec<Objective>)> {
    let boundary = front.get(1)?.mfe;
    let reference = front
        .iter()
        .copied()
        .filter(|o| o.mfe <= boundary)
        .collect();
    Some((boundary, reference))
}

/// The exactness comparison can fail under the exact credit too: inflated by
/// one dekacalorie, it discards the design on the boundary. Without this, a
/// credit offset that stopped reaching the exact cutoff would leave the tests
/// above passing against an unsound credit.
#[test]
fn an_exact_credit_inflated_by_one_unit_discards_the_boundary_design() {
    let (params, usage) = fixtures();
    let mut exercised = 0;
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let Some((boundary, reference)) = boundary_ceiling(&front) else {
            continue;
        };
        let modular = solve(&params, protein, &usage, exact(boundary));
        assert_same(
            &format!("{protein} modular exact arm"),
            &reference,
            &front_objectives(&modular.front),
        );
        let inflated = solve(
            &params,
            protein,
            &usage,
            exact(boundary).with_credit_offset(Energy(1)),
        );
        assert!(
            !front_objectives(&inflated.front)
                .iter()
                .any(|o| o.mfe == boundary),
            "{protein}: an exact credit inflated by one unit kept the boundary design",
        );
        exercised += 1;
    }
    assert!(
        exercised >= 4,
        "only {exercised} cases exercised the mutation"
    );
}

/// The same, confined to every table but the exterior, on the cases whose
/// answer the interior carries — so the exact cutoff at `P`, `M`, `M2` and
/// `Multi` is shown to be read, rather than the exterior doing all the work.
#[test]
fn an_exact_credit_inflated_beyond_the_exterior_changes_the_answer() {
    let (params, usage) = fixtures();
    for protein in INTERIOR_LOAD_BEARING {
        let front = front_objectives(&design_front(&params, protein, &usage).unwrap());
        let (boundary, reference) = boundary_ceiling(&front).unwrap();
        let inflated = solve(
            &params,
            protein,
            &usage,
            exact(boundary).with_credit_offset_beyond_exterior(Energy(1)),
        );
        assert_ne!(
            front_objectives(&inflated.front),
            reference,
            "{protein}: an exact credit inadmissible by one unit inside the fold left \
             the answer unchanged",
        );
    }
}

#[test]
fn an_unbounded_box_with_the_exact_credit_is_the_unconstrained_fold() {
    let (params, usage) = fixtures();
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let plain = design_front(&params, protein, &usage).unwrap();
        let bounded = solve(
            &params,
            protein,
            &usage,
            Bounds::UNBOUNDED.with_exact_credit(),
        );
        assert_eq!(bounded.dropped, DropTally::default());
        assert_eq!(plain, bounded.front, "{protein}: the unbounded box moved");
    }
}

#[test]
fn worker_count_does_not_change_the_exact_credit_answer() {
    let (params, usage) = fixtures();
    for protein in WIDE_CASES {
        for sigma in reachable_ceilings(&params, protein, &usage) {
            let serial = solve_threaded(&params, protein, &usage, exact(sigma), 1);
            let parallel = solve_threaded(&params, protein, &usage, exact(sigma), 8);
            assert_eq!(
                serial.front, parallel.front,
                "{protein} at {sigma}: eight workers reached a different answer",
            );
            assert_eq!(serial.dropped, parallel.dropped);
        }
    }
}

/// The flanked path builds its lattice with the flanks and a profile, and the
/// credit is taken on that same lattice object; a credit keyed on a rebuilt
/// lattice could address different cells. The answer must match the window's.
#[test]
fn the_exact_credit_on_a_flanked_molecule_changes_no_answer() {
    let (params, usage) = fixtures();
    let fixed = |s: &str| UtrSpec::Fixed(s.to_string());
    let flanked = |protein: &str, bounds: Bounds| match design_front_bounded_with_utr(
        &params,
        protein,
        &usage,
        &fixed("GCAUUG"),
        &fixed("AUCGA"),
        UtrProfile::inert(),
        Prune::None,
        bounds,
        1,
    )
    .unwrap()
    {
        Bounded::Feasible(f) => f,
        Bounded::Unreachable { .. } => panic!("{protein}: expected a feasible answer"),
    };
    let mut emptied = 0;
    for protein in CASES {
        let full = flanked(protein, Bounds::UNBOUNDED);
        let energies: Vec<i32> = front_objectives(&full.front)
            .iter()
            .map(|o| o.mfe)
            .collect();
        for sigma in energies {
            let by_window = flanked(protein, window(sigma));
            let by_exact = flanked(protein, exact(sigma));
            assert_same_answer(
                &format!("{protein} flanked at {sigma}"),
                &params,
                &by_window.front,
                &by_exact.front,
            );
            emptied += by_exact.dropped.emptied_total();
        }
    }
    assert!(emptied > 0, "the flanked exact credit never emptied a cell");
}
