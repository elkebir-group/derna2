//! The bifurcation's compose (`M ⊗ P → M2`), the internal-loop corner's
//! (`P ⊗ corner → P`) and the exterior attach (`C ⊗ P → C`) generate only
//! the products their merges keep.
//!
//! A product above the target cell's cutoff, or excluded by a box's cut, is
//! discarded by the merge whatever else arrives, so the compose skips it
//! instead of generating, sorting and eliminating it. The answer cannot move,
//! and the claim here is stronger than the same objectives: the same bytes,
//! sequences and structures included, because the survivors are generated in
//! the same order and sorted stably, so every tie resolves as before.
//!
//! An answer test passes whether or not anything was skipped, so the skip
//! tally is asserted to fire at every table, and a filter one unit too tight
//! at any one site alone is shown to lose a design. The shared fixtures are too short for a multi-loop to reach
//! their fronts, and only a multi-loop passes through the bifurcation, so the
//! tests also run on a protein whose near-optimal designs are multi-loops.

use derna_lib::{
    clock_front, design_front, design_front_bounded, Bounded, BoundedFront, Bounds, BoxSpec,
    Ceiling, CellTable, CodonUsage, Energy, FrontPoint, Params, Prune,
};

mod common;
use common::{front_objectives, CASES, WIDE_CASES};

/// A protein whose second-best design by energy, the boundary of the
/// tightest non-trivial ceiling, closes a multi-loop.
const MULTILOOP: &str = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS";

/// How far above its optimum the multi-loop protein's front is taken: its
/// optimum and the multi-loop boundary design. Kept short because each of its
/// solves costs as much as the shared fixtures' together.
const MULTILOOP_SLACK: Energy = Energy(50);

fn fixtures() -> (Params, CodonUsage) {
    (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    )
}

fn feasible(solved: Bounded<BoundedFront>, label: &str) -> BoundedFront {
    match solved {
        Bounded::Feasible(front) => front,
        Bounded::Unreachable { .. } => panic!("{label}: expected a feasible answer"),
    }
}

fn bounded(params: &Params, protein: &str, usage: &CodonUsage, bounds: Bounds) -> BoundedFront {
    let solved = design_front_bounded(params, protein, usage, Prune::None, bounds, 1).unwrap();
    feasible(solved, protein)
}

fn boxed(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    bounds: Bounds,
    band: &BoxSpec,
) -> BoundedFront {
    let (solved, _) = clock_front(params, protein, usage, bounds, Some(band), 1).unwrap();
    feasible(solved, protein)
}

fn energy(point: &FrontPoint) -> i32 {
    (point.mfe_kcal_mol * 100.0).round() as i32
}

fn ceiling(sigma: i32) -> Bounds {
    Bounds::with_ceiling(Ceiling::Absolute(Energy(sigma)))
}

/// Each case with the front its ceilings and boxes are taken from: the whole
/// front for the shared fixtures, the multi-loop protein's near-optimal part.
fn cases(params: &Params, usage: &CodonUsage) -> Vec<(&'static str, Vec<FrontPoint>)> {
    let mut out: Vec<_> = CASES
        .iter()
        .chain(WIDE_CASES.iter())
        .map(|&p| (p, design_front(params, p, usage).unwrap()))
        .collect();
    let near = Bounds::with_ceiling(Ceiling::AboveOptimum(MULTILOOP_SLACK)).with_exact_credit();
    out.push((MULTILOOP, bounded(params, MULTILOOP, usage, near).front));
    out
}

/// Filtered and unfiltered, at every ceiling a front point sits on, under the
/// window alone and under the exact credit: the same front byte for byte, and
/// the filter fired at `M2`.
#[test]
fn the_filter_changes_no_byte_of_the_bounded_answer() {
    let (params, usage) = fixtures();
    let mut skipped = [0; 3];
    for (protein, front) in cases(&params, &usage) {
        for sigma in front.iter().map(energy) {
            for bounds in [ceiling(sigma), ceiling(sigma).with_exact_credit()] {
                let on = bounded(&params, protein, &usage, bounds);
                let off = bounded(&params, protein, &usage, bounds.without_compose_filter());
                assert_eq!(on.front, off.front, "{protein} at {sigma}: {bounds:?}");
                assert_eq!(off.dropped.skipped_at(CellTable::Branch2), 0);
                assert_eq!(off.dropped.skipped_at(CellTable::Pair), 0);
                assert_eq!(off.dropped.skipped_at(CellTable::Exterior), 0);
                skipped[0] += on.dropped.skipped_at(CellTable::Branch2);
                skipped[1] += on.dropped.skipped_at(CellTable::Pair);
                skipped[2] += on.dropped.skipped_at(CellTable::Exterior);
            }
        }
    }
    assert!(skipped[0] > 0, "the filter never skipped a product at M2");
    assert!(skipped[1] > 0, "the filter never skipped a product at P");
    assert!(skipped[2] > 0, "the filter never skipped a product at C");
}

/// The same for the box, whose cuts the filter also applies, at several
/// weights, on the boxes whose corners are adjacent front points and on the
/// widest one at each ceiling.
#[test]
fn the_filter_changes_no_byte_of_the_box_answer() {
    let (params, usage) = fixtures();
    let mut skipped = [0; 3];
    for (protein, front) in cases(&params, &usage) {
        for hi in 1..front.len() {
            let widest = (hi > 1).then_some(0);
            for lo in std::iter::once(hi - 1).chain(widest) {
                let band = BoxSpec {
                    cai_floor_log: front[lo].cai_log,
                    lambdas: vec![0.0, 0.3, 0.5, 0.7],
                };
                let bounds = ceiling(energy(&front[hi])).with_exact_credit();
                let on = boxed(&params, protein, &usage, bounds, &band);
                let off = boxed(
                    &params,
                    protein,
                    &usage,
                    bounds.without_compose_filter(),
                    &band,
                );
                assert_eq!(on.front, off.front, "{protein} box ({lo}, {hi})");
                skipped[0] += on.dropped.skipped_at(CellTable::Branch2);
                skipped[1] += on.dropped.skipped_at(CellTable::Pair);
                skipped[2] += on.dropped.skipped_at(CellTable::Exterior);
            }
        }
    }
    assert!(skipped[0] > 0, "the filter never skipped a product at M2");
    assert!(skipped[1] > 0, "the filter never skipped a product at P");
    assert!(skipped[2] > 0, "the filter never skipped a product at C");
}

/// A bifurcation filter one unit tighter than the merge's cutoff skips a
/// product the merge would have kept: at the ceiling the multi-loop boundary
/// design sits on, it loses that design. Without this, a filter that stopped
/// reaching the cutoff would pass every test above.
#[test]
fn a_filter_one_unit_too_tight_loses_the_multiloop_boundary_design() {
    let (params, usage) = fixtures();
    let near = Bounds::with_ceiling(Ceiling::AboveOptimum(MULTILOOP_SLACK)).with_exact_credit();
    let front = bounded(&params, MULTILOOP, &usage, near).front;
    let boundary = energy(&front[1]);
    let bounds = ceiling(boundary).with_exact_credit();
    let sound = bounded(&params, MULTILOOP, &usage, bounds);
    assert!(sound.front.iter().any(|p| energy(p) == boundary));
    let tight = bounded(
        &params,
        MULTILOOP,
        &usage,
        bounds.with_filter_offset_at(CellTable::Branch2, Energy(1)),
    );
    assert!(
        !tight.front.iter().any(|p| energy(p) == boundary),
        "a bifurcation filter one unit too tight kept the boundary design",
    );
}

/// The same for the internal-loop corner alone, at `P`, and the exterior
/// attach alone, at `C`: one unit too tight at either changes some fixture's
/// answer at some ceiling.
#[test]
fn a_corner_or_exterior_filter_one_unit_too_tight_changes_an_answer() {
    let (params, usage) = fixtures();
    for table in [CellTable::Pair, CellTable::Exterior] {
        let mut changed = 0;
        for protein in CASES.iter().chain(WIDE_CASES.iter()) {
            for sigma in design_front(&params, protein, &usage)
                .unwrap()
                .iter()
                .map(energy)
            {
                let bounds = ceiling(sigma).with_exact_credit();
                let sound = bounded(&params, protein, &usage, bounds);
                let tight = bounds.with_filter_offset_at(table, Energy(1));
                // Too tight at the exterior can discard the optimum itself,
                // which the solve reports as an error: a changed answer too.
                let kept =
                    match design_front_bounded(&params, protein, &usage, Prune::None, tight, 1) {
                        Ok(Bounded::Feasible(t)) => {
                            front_objectives(&t.front) == front_objectives(&sound.front)
                        }
                        _ => false,
                    };
                changed += usize::from(!kept);
            }
        }
        assert!(
            changed > 0,
            "a filter one unit too tight at {table:?} changed no answer"
        );
    }
}

/// The bifurcations are composed ahead of their merges, fanned out where a
/// position has enough closed pairs: one worker and eight give the same front
/// byte for byte, bounded and in the widest box at that ceiling, on the
/// multi-loop protein. Both run under the window alone: the exact credit
/// empties so many cells that no position keeps enough closed pairs to fan
/// out (measured: none of its positions does at 3 kcal/mol with the credit,
/// 68 without), and so does a narrower box. Where a development build turns
/// the serial detail on, the test also asserts that the eight-worker runs
/// fanned out.
#[test]
fn worker_count_does_not_change_the_multiloop_answer() {
    let (params, usage) = fixtures();
    let wide = Bounds::with_ceiling(Ceiling::AboveOptimum(Energy(300)));
    let solve = |band: Option<&BoxSpec>, bounds, threads| {
        let (solved, clock) =
            clock_front(&params, MULTILOOP, &usage, bounds, band, threads).unwrap();
        // Under the serial split, show the eight-worker run fanned out.
        let split = clock.fold.serial;
        if threads > 1 && split.composes > 0 {
            assert!(
                split.fanned > 0,
                "no position fanned out (box: {})",
                band.is_some()
            );
        }
        feasible(solved, MULTILOOP)
    };
    let (one, eight) = (solve(None, wide, 1), solve(None, wide, 8));
    assert_eq!(one.front, eight.front);
    assert_eq!(one.dropped, eight.dropped);
    let front = &one.front;
    let band = BoxSpec {
        cai_floor_log: front[0].cai_log,
        lambdas: vec![0.5],
    };
    let bounds = ceiling(energy(&front[front.len() - 1]));
    let (one, eight) = (solve(Some(&band), bounds, 1), solve(Some(&band), bounds, 8));
    assert_eq!(one.front, eight.front);
    assert_eq!(one.dropped, eight.dropped);
}
