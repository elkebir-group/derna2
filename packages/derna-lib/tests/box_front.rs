//! The box fold: the front of one band, between an adaptation floor and an
//! energy ceiling, folded with the box's cuts applied at every merge.
//!
//! The contract is one sentence — the box fold's front is the ceiling's front
//! filtered to the floor — and every test here checks it against the complete
//! unbounded front, for every box whose corners are front points, on the
//! coding-only lattice and on flanked ones. The cuts are exact, so they may
//! remove work and never an answer; the corner point of each box sits exactly
//! on every cut, which is what makes an over-credited cut visible.

use derna_lib::{
    achievable_minimum, clock_front, design_front, design_front_band, design_front_band_with_utr,
    design_front_between, design_front_between_with_utr, design_front_bounded,
    design_front_bounded_with_utr, design_front_box, design_front_box_with_utr, design_front_walk,
    design_front_walk_with_utr, replay_gap, tie_weight, BeamError, Bounded, BoundedFront, Bounds,
    BoxSpec, Ceiling, CellTable, CodonUsage, Energy, FrontPoint, GapCut, GapSpec, HullPoint,
    Nucleotide, Params, PassGrid, Prune, RefineLoose, RefinePriced, RegionConstraints, Slicing,
    TaskKind, UtrProfile, UtrSpec, WalkBudget, WalkOptions,
};

use derna_lib::task_clock::ThreadCpu;

mod common;
use common::{CASES, WIDE_CASES};

/// The floor alone, the floor with one coupled cut, and several coupled cuts.
const WEIGHT_SETS: [&[f64]; 3] = [&[0.0], &[0.0, 0.5], &[0.0, 0.3, 0.5, 0.7]];

fn fixtures() -> (Params, CodonUsage) {
    (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    )
}

/// A point's `(energy, adaptation accumulator)`: the two axes the front is
/// non-dominated over, whether or not a profile prices the flanks.
fn axes(front: &[FrontPoint]) -> Vec<(i32, f64)> {
    front
        .iter()
        .map(|p| {
            (
                (p.mfe_kcal_mol * 100.0).round() as i32,
                p.cai_log + p.utr_profile_log,
            )
        })
        .collect()
}

fn assert_same(label: &str, want: &[(i32, f64)], got: &[(i32, f64)]) {
    assert_eq!(want.len(), got.len(), "{label}: {want:?} vs {got:?}");
    for (w, g) in want.iter().zip(got) {
        assert_eq!(w.0, g.0, "{label}: energy differs, {want:?} vs {got:?}");
        assert!((w.1 - g.1).abs() < 1e-9, "{label}: adaptation differs");
    }
}

/// Every point's own sequence and structure evaluate to the energy it reports.
fn assert_real(label: &str, params: &Params, front: &[FrontPoint]) {
    for p in front {
        let reported = (p.mfe_kcal_mol * 100.0).round() as i32;
        let evaluated = rna_mfe::evaluate(params, p.sequence.as_bytes(), &p.structure)
            .unwrap_or_else(|e| panic!("{label}: {} does not evaluate: {e:?}", p.structure));
        assert_eq!(evaluated.0, reported, "{label}: {}", p.sequence);
    }
}

/// A point on the two axes: `(energy, adaptation accumulator)`.
type Axes = (i32, f64);

/// Every box with both corners on the front: ceiling at point `i`'s energy,
/// floor at point `j`'s adaptation, `j <= i` so the box holds points `j..=i`.
fn boxes(full: &[Axes]) -> Vec<(i32, f64, Vec<Axes>)> {
    let mut out = Vec::new();
    for i in 0..full.len() {
        for j in 0..=i {
            let (ceiling, floor) = (full[i].0, full[j].1);
            let expected = full
                .iter()
                .copied()
                .filter(|&(e, a)| e <= ceiling && a >= floor)
                .collect();
            out.push((ceiling, floor, expected));
        }
    }
    out
}

fn bounded_front(solved: Bounded<BoundedFront>) -> BoundedFront {
    match solved {
        Bounded::Feasible(f) => f,
        Bounded::Unreachable { .. } => panic!("expected a feasible answer"),
    }
}

fn feasible(solved: Bounded<BoundedFront>) -> Vec<FrontPoint> {
    bounded_front(solved).front
}

#[test]
fn the_box_front_is_the_ceilings_front_filtered_to_the_floor() {
    let (params, usage) = fixtures();
    let mut boxes_checked = 0;
    let mut boxed = 0;
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        for (ceiling, floor, expected) in boxes(&full) {
            for weights in WEIGHT_SETS {
                let band = BoxSpec {
                    cai_floor_log: floor,
                    lambdas: weights.to_vec(),
                };
                let got = bounded_front(
                    design_front_box(
                        &params,
                        protein,
                        &usage,
                        Ceiling::Absolute(Energy(ceiling)),
                        &band,
                        1,
                    )
                    .unwrap(),
                );
                let label = format!("{protein} box ({ceiling}, {floor:.4}) at {weights:?}");
                assert_same(&label, &expected, &axes(&got.front));
                assert_real(&label, &params, &got.front);
                boxed += got.dropped.boxed_total();
                boxes_checked += 1;
            }
        }
    }
    assert!(boxes_checked >= 200, "{boxes_checked}");
    assert!(boxed > 0, "no box cut removed anything on any case");
}

/// The cuts are exact, so the answers above would come back the same with
/// them switched off. A protein long enough that each cut removes something
/// the other does not shows both are read inside the fold, at the closed-pair
/// table and not only at the root, and that the answer holds there too.
#[test]
fn each_cut_removes_points_inside_the_fold_and_the_answer_holds() {
    let (params, usage) = fixtures();
    let protein = "MKTAYIAKQRQISFVKSHFSRQ";
    let optimum = achievable_minimum(&params, protein, &usage).unwrap();
    let ceiling = Ceiling::AboveOptimum(Energy(optimum.0.abs() / 10));
    let bounded = feasible(
        design_front_bounded(
            &params,
            protein,
            &usage,
            Prune::None,
            Bounds::with_ceiling(ceiling).with_exact_credit(),
            1,
        )
        .unwrap(),
    );
    assert!(bounded.len() >= 3, "{}", bounded.len());
    let floor = bounded[bounded.len() / 3].cai_log;
    let expected: Vec<(i32, f64)> = axes(&bounded)
        .into_iter()
        .filter(|&(_, a)| a >= floor)
        .collect();
    for weights in [vec![0.0], vec![0.5], vec![0.0, 0.5]] {
        let band = BoxSpec {
            cai_floor_log: floor,
            lambdas: weights.clone(),
        };
        let Bounded::Feasible(boxed) =
            design_front_box(&params, protein, &usage, ceiling, &band, 1).unwrap()
        else {
            panic!("feasible");
        };
        let label = format!("{protein} at {weights:?}");
        assert_same(&label, &expected, &axes(&boxed.front));
        assert!(
            boxed.dropped.boxed_at(CellTable::Pair) > 0,
            "{label}: the cuts removed no closed-pair point",
        );
    }
}

#[test]
fn worker_count_does_not_change_the_box_answer() {
    let (params, usage) = fixtures();
    for protein in WIDE_CASES {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let (ceiling, floor) = (full[full.len() - 2].0, full[1].1);
        let band = BoxSpec {
            cai_floor_log: floor,
            lambdas: vec![0.0, 0.3, 0.5, 0.7],
        };
        // The whole solve, drop tallies included: the front is the same under
        // any cut weights, so only the tallies would show a cut folded at the
        // wrong weight or paired with the wrong chain.
        let solve = |threads| {
            bounded_front(
                design_front_box(
                    &params,
                    protein,
                    &usage,
                    Ceiling::Absolute(Energy(ceiling)),
                    &band,
                    threads,
                )
                .unwrap(),
            )
        };
        // One worker runs the five scalar chains in turn; two run them two at
        // a time on one worker each; eight on one or two each, five at once.
        let serial = solve(1);
        assert_eq!(serial, solve(2), "{protein}");
        assert_eq!(serial, solve(8), "{protein}");
    }
}

/// The flanked lattice, with a designed flank whose alphabet excludes the
/// profile's favourite nucleotide: its positions carry a normalization offset,
/// so the floor stated on the reported accumulator and the one the cuts
/// compare differ by a non-zero constant. A box fold that skipped the
/// conversion would compare against the wrong floor.
#[test]
fn the_box_front_on_a_flanked_molecule_with_a_narrowed_alphabet() {
    let (params, usage) = fixtures();
    // `U` is the profile's favourite and `S` is `{G, C}`, so each designed
    // 5′ position's best available weight is `w(G) < 0`.
    let profile = UtrProfile::new([0.2, 0.25, 0.3, 0.6], 1.0).unwrap();
    assert!(
        profile.weight(Nucleotide::G) < 0.0,
        "the offset would be zero"
    );
    assert!(profile.weight(Nucleotide::C) < profile.weight(Nucleotide::G));
    let flanks = [
        (
            UtrSpec::Mixed("SSA".into()),
            UtrSpec::Fixed("AU".into()),
            profile,
        ),
        (
            UtrSpec::Fixed("GCAUUG".into()),
            UtrSpec::Fixed("AUCGA".into()),
            UtrProfile::inert(),
        ),
    ];
    let mut boxes_checked = 0;
    let mut boxed = 0;
    for (utr5, utr3, profile) in &flanks {
        for protein in CASES {
            let full_front = feasible(
                design_front_bounded_with_utr(
                    &params,
                    protein,
                    &usage,
                    utr5,
                    utr3,
                    *profile,
                    Prune::None,
                    Bounds::UNBOUNDED,
                    1,
                )
                .unwrap(),
            );
            let full = axes(&full_front);
            for (ceiling, floor, expected) in boxes(&full) {
                let band = BoxSpec {
                    cai_floor_log: floor,
                    lambdas: vec![0.0, 0.5],
                };
                let got = bounded_front(
                    design_front_box_with_utr(
                        &params,
                        protein,
                        &usage,
                        utr5,
                        utr3,
                        *profile,
                        Ceiling::Absolute(Energy(ceiling)),
                        &band,
                        1,
                    )
                    .unwrap(),
                );
                let label = format!("{protein} flanked {utr5:?} box ({ceiling}, {floor:.4})");
                assert_same(&label, &expected, &axes(&got.front));
                assert_real(&label, &params, &got.front);
                boxed += got.dropped.boxed_total();
                boxes_checked += 1;
            }
        }
    }
    assert!(boxes_checked >= 40, "{boxes_checked}");
    assert!(boxed > 0, "no box cut removed anything on a flanked case");
}

/// The front `full` restricted to a band: the best point at or below `lower`
/// (the band's edge), then every point with `lower < MFE <= upper`.
fn band_of(full: &[Axes], lower: i32, upper: i32) -> Vec<Axes> {
    let edge = full.iter().copied().rfind(|&(e, _)| e <= lower);
    let mut band: Vec<Axes> = edge.filter(|&(e, _)| e <= upper).into_iter().collect();
    band.extend(
        full.iter()
            .copied()
            .filter(|&(e, _)| lower < e && e <= upper),
    );
    band
}

/// Ceilings on the front's energies and one unit either side of each, so a
/// band edge lands on, immediately above and immediately below a front point.
fn band_ceilings(full: &[Axes]) -> Vec<i32> {
    let mut out: Vec<i32> = full.iter().flat_map(|&(e, _)| [e - 1, e, e + 1]).collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[test]
fn the_band_is_the_front_between_its_two_energies() {
    let (params, usage) = fixtures();
    let mut bands = 0;
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let ceilings = band_ceilings(&full);
        for &lower in &ceilings {
            for upper in full.iter().map(|&(e, _)| e).filter(|&u| u >= lower) {
                let got = feasible(
                    design_front_band(
                        &params,
                        protein,
                        &usage,
                        Ceiling::Absolute(Energy(lower)),
                        Ceiling::Absolute(Energy(upper)),
                        &[0.0, 0.5],
                        1,
                    )
                    .unwrap(),
                );
                let label = format!("{protein} band ({lower}, {upper}]");
                assert_same(&label, &band_of(&full, lower, upper), &axes(&got));
                bands += 1;
            }
        }
    }
    assert!(bands >= 100, "{bands}");
}

#[test]
fn a_band_below_the_optimum_is_the_front_up_to_its_top_and_an_inverted_one_is_empty() {
    let (params, usage) = fixtures();
    for protein in WIDE_CASES {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let (optimum, upper) = (full[0].0, full[full.len() - 2].0);
        let solve = |lower: i32, upper: i32| {
            feasible(
                design_front_band(
                    &params,
                    protein,
                    &usage,
                    Ceiling::Absolute(Energy(lower)),
                    Ceiling::Absolute(Energy(upper)),
                    &[0.0, 0.5],
                    1,
                )
                .unwrap(),
            )
        };
        let below: Vec<Axes> = full.iter().copied().filter(|&(e, _)| e <= upper).collect();
        assert_same(protein, &below, &axes(&solve(optimum - 1, upper)));
        assert!(solve(upper, optimum).is_empty(), "{protein}");
    }
}

#[test]
fn the_band_on_a_flanked_molecule_with_a_narrowed_alphabet() {
    let (params, usage) = fixtures();
    let profile = UtrProfile::new([0.2, 0.25, 0.3, 0.6], 1.0).unwrap();
    let (utr5, utr3) = (UtrSpec::Mixed("SSA".into()), UtrSpec::Fixed("AU".into()));
    for protein in CASES {
        let full = axes(&feasible(
            design_front_bounded_with_utr(
                &params,
                protein,
                &usage,
                &utr5,
                &utr3,
                profile,
                Prune::None,
                Bounds::UNBOUNDED,
                1,
            )
            .unwrap(),
        ));
        for i in 0..full.len() {
            for j in i..full.len() {
                let (lower, upper) = (full[i].0 + 1, full[j].0);
                let got = feasible(
                    design_front_band_with_utr(
                        &params,
                        protein,
                        &usage,
                        &utr5,
                        &utr3,
                        profile,
                        Ceiling::Absolute(Energy(lower)),
                        Ceiling::Absolute(Energy(upper)),
                        &[0.0, 0.5],
                        1,
                    )
                    .unwrap(),
                );
                let label = format!("{protein} flanked band ({lower}, {upper}]");
                assert_same(&label, &band_of(&full, lower, upper), &axes(&got));
            }
        }
    }
}

/// The vertices of `full`'s convex hull in the space the λ-sweep is linear in,
/// `(energy, −100·adaptation)`: the points some weighting selects. `full` is a
/// staircase, energy ascending, so its lower hull runs from the most stable
/// point to the best adapted. A point exactly on an edge ties at that edge's
/// weight, so the sweep can return it; only a strict turn pops.
fn hull(full: &[Axes]) -> Vec<usize> {
    let xy = |i: usize| (f64::from(full[i].0), -100.0 * full[i].1);
    let mut out: Vec<usize> = Vec::new();
    for i in 0..full.len() {
        while out.len() >= 2 {
            let (o, p, q) = (xy(out[out.len() - 2]), xy(out[out.len() - 1]), xy(i));
            let cross = (p.0 - o.0) * (q.1 - o.1) - (p.1 - o.1) * (q.0 - o.0);
            if cross < 0.0 {
                out.pop();
            } else {
                break;
            }
        }
        out.push(i);
    }
    out
}

fn hull_point(p: Axes) -> HullPoint {
    HullPoint {
        mfe_kcal_mol: f64::from(p.0) / 100.0,
        adaptation_log: p.1,
    }
}

/// The front between two hull points `i < j` of `full`, both included.
fn between_of(full: &[Axes], i: usize, j: usize) -> Vec<Axes> {
    full[i..=j].to_vec()
}

/// The contract of the hull-gap solve, for every pair of hull vertices: the
/// front between them, both ends included; for an adjacent pair, the interior
/// is exactly the front's unsupported points there.
#[test]
fn between_two_hull_points_is_the_front_between_them() {
    let (params, usage) = fixtures();
    let (mut adjacent, mut gaps, mut boxed) = (0, 0, 0);
    let long = ["MKTAYIAKQRQISFVKSHFSRQ"];
    for protein in CASES.iter().chain(WIDE_CASES.iter()).chain(long.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let vertices = hull(&full);
        for (k, &i) in vertices.iter().enumerate() {
            for &j in &vertices[k + 1..] {
                let (a, b) = (hull_point(full[i]), hull_point(full[j]));
                let got =
                    bounded_front(design_front_between(&params, protein, &usage, a, b, 1).unwrap());
                let label = format!("{protein} between {i} and {j}");
                assert_same(&label, &between_of(&full, i, j), &axes(&got.front));
                assert_real(&label, &params, &got.front);
                boxed += got.dropped.boxed_total();
                if j == vertices[k + 1] {
                    adjacent += 1;
                    let interior = &axes(&got.front)[1..got.front.len() - 1];
                    let unsupported: Vec<Axes> = (i + 1..j)
                        .filter(|x| !vertices.contains(x))
                        .map(|x| full[x])
                        .collect();
                    assert_same(&label, &unsupported, interior);
                    gaps += usize::from(!interior.is_empty());
                }
            }
        }
    }
    assert!(adjacent >= 14, "{adjacent}");
    assert!(
        gaps >= 10,
        "only {gaps} adjacent hull pairs had a point between them"
    );
    assert!(boxed > 0, "the tie-weight cut removed nothing on any case");
}

/// The walk's contract: one hull gap at a time, from the most stable design
/// to the most adapted, it recovers the whole exact front, each point a real
/// design, on any worker count.
#[test]
fn the_walked_front_is_the_whole_front() {
    let (params, usage) = fixtures();
    let (mut walks, mut multi_gap, mut sliced) = (0, 0, 0);
    let mut sliced_by_width = 0;
    let (mut uncut, mut chorded, mut interpolated, mut probes_cut) = (0, 0, 0, 0);
    let mut seeds_saved = 0;
    let long = ["MKTAYIAKQRQISFVKSHFSRQ"];
    let plan = |stop_slack, slice_slack, uncut_within, chord_bracket, cut_probes| WalkOptions {
        stop_slack,
        slice_slack,
        uncut_within,
        chord_bracket,
        cut_probes,
        pass_grid: None,
        seed_sweep: false,
        overlap_credit: false,
        bracket_cuts: 0,
        grid_ends: false,
        heavy_fold: None,
        fold_by_width: false,
        max_running: None,
        refine_loose: None,
        refine_priced: None,
        lend_idle_workers: false,
        speculative_slices: false,
        own_when_alone: None,
        own_when_costly: None,
        own_watch_s: None,
        own_min_looseness: None,
        slice_width: None,
        budget: WalkBudget::default(),
        floor_cut: false,
        gap_folds: None,
        slice_points: None,
    };
    let grid = PassGrid::two_step(0.1, 0.25, 0.3);
    // Wide at the top: its last bracket runs from 0.3 to the credit's own
    // pass at 1, which the cases' highest tie weights (~0.4-0.63) read.
    let placed = PassGrid {
        segments: vec![(0.3, 0.1), (1.0, 0.7)],
    };
    let gridded = |grid: &PassGrid, options: WalkOptions| WalkOptions {
        pass_grid: Some(grid.clone()),
        ..options
    };
    let seeded = |options: WalkOptions| WalkOptions {
        seed_sweep: true,
        ..options
    };
    let overlapped = |options: WalkOptions| WalkOptions {
        overlap_credit: true,
        ..options
    };
    let bracketed = |options: WalkOptions| WalkOptions {
        bracket_cuts: 1,
        ..options
    };
    let grid_ended = |options: WalkOptions| WalkOptions {
        grid_ends: true,
        ..options
    };
    // Refined to adjacency, stopped early, solved as one gap from end to end,
    // with wide gaps cut into energy slices, with no cut near the optimum or
    // anywhere, with gaps bounded by the chord between other gaps' passes
    // however far apart they lie, and with probes cut by the same chords.
    let plans = [
        (1, plan(None, None, None, None, false)),
        (4, plan(None, None, None, None, false)),
        (4, plan(Some(150.0), None, None, None, false)),
        (1, plan(Some(1e12), None, None, None, false)),
        (4, plan(None, Some(5.0), None, None, false)),
        (1, plan(Some(1e12), Some(20.0), None, None, false)),
        (4, plan(None, None, Some(30), None, false)),
        (1, plan(None, Some(5.0), Some(i32::MAX / 2), None, false)),
        (1, plan(None, None, None, Some(1.0), false)),
        (4, plan(None, Some(5.0), Some(30), Some(1.0), false)),
        (1, plan(None, None, None, Some(1.0), true)),
        (4, plan(None, Some(5.0), Some(30), Some(1.0), true)),
        (1, gridded(&grid, plan(None, None, None, None, true))),
        (
            4,
            gridded(&grid, plan(None, Some(5.0), Some(30), None, true)),
        ),
        (1, gridded(&placed, plan(None, None, None, None, true))),
        (4, gridded(&placed, plan(None, Some(5.0), None, None, true))),
        // The sweep started from the grid's own optima.
        (
            1,
            seeded(gridded(&grid, plan(None, None, None, None, true))),
        ),
        (
            4,
            seeded(gridded(
                &placed,
                plan(None, Some(5.0), Some(30), None, true),
            )),
        ),
        (
            1,
            seeded(gridded(&placed, plan(None, None, None, None, false))),
        ),
        // The credit and the ends folded beside the grid. On one worker every
        // probe is cut, so the placed grid's top bracket must find the
        // credit's pass at 1.
        (
            1,
            overlapped(seeded(gridded(&placed, plan(None, None, None, None, true)))),
        ),
        (
            4,
            overlapped(seeded(gridded(
                &placed,
                plan(None, Some(5.0), Some(30), None, true),
            ))),
        ),
        // Gaps cut also at the two passes their chord is drawn between.
        (
            1,
            bracketed(seeded(gridded(&grid, plan(None, None, None, None, true)))),
        ),
        (
            4,
            bracketed(seeded(gridded(
                &placed,
                plan(None, Some(5.0), None, None, true),
            ))),
        ),
        // The adapted end read off the grid's lowest pass, moved to `lo`.
        (
            1,
            grid_ended(seeded(gridded(&grid, plan(None, None, None, None, true)))),
        ),
        // ... with the credit and the stable end as queue tasks.
        (
            4,
            overlapped(grid_ended(bracketed(seeded(gridded(
                &placed,
                plan(None, Some(5.0), None, None, true),
            ))))),
        ),
        (
            4,
            grid_ended(bracketed(seeded(gridded(
                &placed,
                plan(None, Some(5.0), Some(30), None, true),
            )))),
        ),
        // Narrow pairs under a pooled bracket folded without a probe.
        (
            1,
            seeded(gridded(&grid, plan(Some(150.0), None, None, None, true))),
        ),
        (
            4,
            grid_ended(bracketed(seeded(gridded(
                &placed,
                plan(Some(60.0), Some(5.0), Some(30), None, true),
            )))),
        ),
        // Wide gaps folded ahead of the probes.
        (
            4,
            WalkOptions {
                heavy_fold: Some(50),
                ..grid_ended(bracketed(seeded(gridded(
                    &placed,
                    plan(None, Some(5.0), Some(30), None, true),
                ))))
            },
        ),
        // The folds taken widest first.
        (
            4,
            WalkOptions {
                fold_by_width: true,
                ..grid_ended(bracketed(seeded(gridded(
                    &placed,
                    plan(None, Some(5.0), Some(30), None, true),
                ))))
            },
        ),
        // Gaps sliced by energy width, alone and over a slack.
        (
            1,
            WalkOptions {
                slice_width: Some(20.0),
                ..plan(None, None, Some(30), None, false)
            },
        ),
        (
            4,
            WalkOptions {
                slice_width: Some(20.0),
                fold_by_width: true,
                ..grid_ended(bracketed(seeded(gridded(
                    &placed,
                    plan(None, Some(5.0), Some(30), None, true),
                ))))
            },
        ),
        // Three pooled passes on each side.
        (
            4,
            WalkOptions {
                bracket_cuts: 3,
                ..seeded(gridded(&grid, plan(None, Some(5.0), None, None, true)))
            },
        ),
    ];
    let (mut overlaps, mut narrowed) = (0, 0);
    for protein in CASES.iter().chain(WIDE_CASES.iter()).chain(long.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        // A walk refined to adjacency finds the hull's vertices, whatever it
        // cuts: a probe that missed one would still give the whole front,
        // through a wider gap, so the front alone cannot catch it.
        let mut hull_vertices: Option<Vec<f64>> = None;
        for (threads, options) in &plans {
            let threads = *threads;
            let WalkOptions {
                stop_slack,
                slice_slack,
                ..
            } = *options;
            let walked = design_front_walk(
                &params,
                protein,
                &usage,
                1e-4,
                0.999,
                options.clone(),
                threads,
            )
            .unwrap();
            let label = format!("{protein} walked on {threads} workers, {options:?}");
            assert_same(&label, &full, &axes(&walked.front));
            assert_real(&label, &params, &walked.front);
            assert_eq!(walked.gaps.len() + 1, walked.vertices.len(), "{label}");
            for gap in &walked.gaps {
                assert_eq!(tie_weight(gap.a, gap.b), Some(gap.lambda), "{label}");
            }
            // The timeline holds every task: the probes past the two end
            // folds, and one fold per gap.
            let count =
                |f: &dyn Fn(TaskKind) -> bool| walked.timeline.iter().filter(|s| f(s.kind)).count();
            let probes = count(&|k| matches!(k, TaskKind::Probe { .. }));
            // Ends read off the grid are not folded on their own.
            let grid_end = usize::from(options.grid_ends && options.seed_sweep);
            assert_eq!(probes + 2 - grid_end, walked.sweep_scans, "{label}");
            let cut = count(&|k| k == TaskKind::Probe { cut: true });
            assert_eq!(cut, walked.probes_cut, "{label}");
            narrowed += count(&|k| k == TaskKind::Narrow);
            assert_eq!(
                count(&|k| k == TaskKind::Fold),
                walked.gaps.len(),
                "{label}"
            );
            // Overlapped, the credit and the ends not read off the grid are
            // tasks too, and no probe is queued before the credit is built.
            // (The task releases its pairs before the queue records its end.)
            let setup = usize::from(options.overlap_credit);
            assert_eq!(count(&|k| k == TaskKind::Credit), setup, "{label}");
            assert_eq!(
                count(&|k| k == TaskKind::End),
                setup * (2 - grid_end),
                "{label}"
            );
            if let Some(credit) = walked.timeline.iter().find(|s| s.kind == TaskKind::Credit) {
                overlaps += 1;
                assert!(
                    walked
                        .timeline
                        .iter()
                        .filter(|s| matches!(s.kind, TaskKind::Probe { .. }))
                        .all(|s| s.queued_s >= credit.start_s + walked.credit_s),
                    "{label}"
                );
            }
            assert!(
                walked
                    .timeline
                    .iter()
                    .all(|s| s.queued_s <= s.start_s && s.start_s <= s.end_s),
                "{label}"
            );
            walks += 1;
            if stop_slack.is_none() && slice_slack.is_none() {
                multi_gap += usize::from(walked.gaps.len() > 1);
            }
            if stop_slack == Some(1e12) {
                assert!(
                    walked.gaps.len() <= 1,
                    "{label}: the whole range is one gap"
                );
            }
            sliced += walked.gaps.iter().filter(|g| g.slices > 1).count();
            if options.slice_width.is_some() {
                sliced_by_width += walked.gaps.iter().filter(|g| g.slices > 1).count();
            }
            probes_cut += walked.probes_cut;
            if options.pass_grid.is_some() && options.cut_probes && threads == 1 {
                // On one worker the grid is pooled before any probe starts, so
                // every probe but the two ends finds a grid bracket and no gap
                // takes a pass of its own.
                assert_eq!(
                    walked.probes_cut + 2 - grid_end,
                    walked.sweep_scans,
                    "{label}"
                );
                assert!(
                    walked.gaps.iter().all(|g| g.cut != GapCut::Exact),
                    "{label}"
                );
            }
            if options.seed_sweep && stop_slack.is_none() && walked.vertices.len() > 1 {
                // From two ends the sweep folds the ends, splits to find every
                // other vertex and certifies every adjacent pair: 2V - 1
                // scans. Each grid optimum found in place of a split saves one.
                let unseeded = 2 * walked.vertices.len() - 1;
                assert!(walked.sweep_scans <= unseeded, "{label}");
                seeds_saved += unseeded - walked.sweep_scans;
            }
            if stop_slack.is_none() {
                let found: Vec<f64> = walked.vertices.iter().map(|v| v.mfe_kcal_mol).collect();
                let first = hull_vertices.get_or_insert_with(|| found.clone());
                assert_eq!(*first, found, "{label}: the hull's vertices");
            }
            for gap in &walked.gaps {
                match gap.cut {
                    GapCut::Uncut => uncut += 1,
                    GapCut::Chord { lo, hi } => {
                        assert!(lo <= gap.lambda && gap.lambda <= hi, "{label}");
                        chorded += 1;
                        interpolated += usize::from(lo < hi);
                    }
                    GapCut::Exact => {}
                }
            }
        }
    }
    assert!(walks >= 2);
    assert!(multi_gap > 0, "no case walked more than one gap");
    assert!(sliced > 0, "no gap was solved in slices");
    assert!(sliced_by_width > 0, "no gap was sliced by width");
    assert!(uncut > 0, "no gap was solved without a cut");
    assert!(interpolated > 0, "no gap read a chord between two passes");
    assert!(chorded >= interpolated);
    assert!(probes_cut > 0, "no probe was cut at its tie value");
    assert!(seeds_saved > 0, "no grid optimum stood in for a probe");
    assert!(overlaps > 0, "no walk folded its credit beside the grid");
    assert!(narrowed > 0, "no narrow pair skipped its probe");
}

/// The two ends score the same at the tie weight, so the chord between them
/// is level there; an unordered pair is refused.
/// Refining a loose bracket mid-walk adds exact passes only, so the front is
/// the whole front whether every bracket refines, none does, or the rule
/// decides; a refined bracket adds exactly its candidate weights, each run as
/// a grid task, and its gaps fold only after them.
#[test]
fn refining_loose_brackets_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let base = WalkOptions {
        slice_slack: Some(5.0),
        slice_width: None,
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        ..WalkOptions::default()
    };
    let refine = |threshold_kcal| WalkOptions {
        refine_loose: Some(RefineLoose {
            candidates: PassGrid::two_step(0.05, 0.05, 1.0),
            threshold_kcal,
        }),
        ..base.clone()
    };
    let grid_tasks = |timeline: &[derna_lib::TaskSpan]| {
        timeline.iter().filter(|s| s.kind == TaskKind::Grid).count()
    };
    let (mut refined, mut kept) = (0, 0);
    for protein in CASES.iter().chain(["MKTAYIAKQRQISFVKSHFSRQ"].iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walk = |options: WalkOptions| {
            design_front_walk(&params, protein, &usage, 1e-4, 0.999, options, 4).unwrap()
        };
        let plain = walk(base.clone());
        assert!(plain.refinements.is_empty(), "{protein}");
        assert_same(protein, &full, &axes(&plain.front));
        for threshold in [0.0, 1e9, 0.5] {
            let label = format!("{protein} refined above {threshold} kcal/mol");
            let walked = walk(refine(threshold));
            assert_same(&label, &full, &axes(&walked.front));
            assert_real(&label, &params, &walked.front);
            let added: usize = walked.refinements.iter().map(|r| r.added.len()).sum();
            assert_eq!(
                grid_tasks(&walked.timeline),
                grid_tasks(&plain.timeline) + added,
                "{label}"
            );
            for r in &walked.refinements {
                assert!(
                    r.candidates.iter().all(|&w| r.lo < w && w < r.hi),
                    "{label}"
                );
                let decided = r.decided_s.is_some();
                assert_eq!(decided, r.released_s.is_some(), "{label}");
                if threshold >= 1e9 || r.looseness_kcal.is_none() {
                    assert!(r.added.is_empty(), "{label}");
                }
                if threshold == 0.0 && r.looseness_kcal.is_some_and(|l| l > 0.0) {
                    assert_eq!(r.added, r.candidates, "{label}");
                }
                if r.added.is_empty() {
                    kept += usize::from(decided);
                    continue;
                }
                refined += 1;
                // No gap inside a refined bracket folded before its passes.
                let released = r.released_s.unwrap();
                let early = walked.timeline.iter().any(|s| {
                    s.kind == TaskKind::Fold
                        && r.lo <= s.lambda
                        && s.lambda < r.hi
                        && s.start_s < released
                });
                assert!(!early, "{label}: a fold in [{}, {}) ran early", r.lo, r.hi);
            }
        }
    }
    assert!(refined > 0, "no bracket refined on any case");
    assert!(kept > 0, "no bracket was decided and kept on any case");
}

/// Priced refinement adds exact passes at held gaps' tie weights only, so the
/// front is the whole front whether every bracket adds passes, none does, or
/// the predicted saving decides; each added pass is a grid task at a gap's tie
/// weight inside its bracket, and the bracket's gaps fold only after it. A pass
/// at a gap's own tie weight cuts that gap with the exact pass, which is the
/// case a fixed candidate grid never reaches.
#[test]
fn pricing_refinements_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let base = WalkOptions {
        slice_slack: Some(5.0),
        slice_width: None,
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        ..WalkOptions::default()
    };
    let priced = |margin_passes, max_passes| WalkOptions {
        refine_priced: Some(RefinePriced {
            margin_passes,
            max_passes,
            ..RefinePriced::default()
        }),
        ..base.clone()
    };
    let grid_tasks = |timeline: &[derna_lib::TaskSpan]| {
        timeline.iter().filter(|s| s.kind == TaskKind::Grid).count()
    };
    let (mut added_any, mut kept) = (0, 0);
    for protein in CASES.iter().chain(["MKTAYIAKQRQISFVKSHFSRQ"].iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walk = |options: WalkOptions| {
            design_front_walk(&params, protein, &usage, 1e-4, 0.999, options, 4).unwrap()
        };
        let plain = walk(base.clone());
        for (margin, most) in [(-1e9, 2), (1e9, 3), (1.0, 1)] {
            let label = format!("{protein} priced at margin {margin}, at most {most}");
            let walked = walk(priced(margin, most));
            assert_same(&label, &full, &axes(&walked.front));
            assert_real(&label, &params, &walked.front);
            let added: usize = walked.refinements.iter().map(|r| r.added.len()).sum();
            assert_eq!(
                grid_tasks(&walked.timeline),
                grid_tasks(&plain.timeline) + added,
                "{label}"
            );
            let ties: Vec<f64> = walked.gaps.iter().map(|g| g.lambda).collect();
            for r in &walked.refinements {
                assert!(r.candidates.is_empty(), "{label}");
                assert!(r.added.len() <= most, "{label}");
                let decided = r.decided_s.is_some();
                assert_eq!(decided, r.released_s.is_some(), "{label}");
                assert_eq!(decided, r.predicted_passes.is_some(), "{label}");
                for &w in &r.added {
                    assert!(
                        r.lo < w && w < r.hi,
                        "{label}: {w} outside [{}, {})",
                        r.lo,
                        r.hi
                    );
                    assert!(ties.contains(&w), "{label}: {w} is no gap's tie weight");
                }
                if margin >= 1e9 {
                    assert!(r.added.is_empty(), "{label}");
                }
                if let Some((before, after)) = r.predicted_passes {
                    assert!(after <= before, "{label}");
                    if margin <= -1e9 && r.gaps > 0 && before > 0.0 {
                        assert!(!r.added.is_empty(), "{label}: [{}, {})", r.lo, r.hi);
                    }
                }
                if r.added.is_empty() {
                    kept += usize::from(decided);
                    continue;
                }
                added_any += 1;
                let released = r.released_s.unwrap();
                let early = walked.timeline.iter().any(|s| {
                    s.kind == TaskKind::Fold
                        && r.lo <= s.lambda
                        && s.lambda < r.hi
                        && s.start_s < released
                });
                assert!(!early, "{label}: a fold in [{}, {}) ran early", r.lo, r.hi);
            }
        }
    }
    assert!(added_any > 0, "no bracket added a pass on any case");
    assert!(kept > 0, "no bracket was decided and kept on any case");
}

/// Lending idle workers to folds leaves the walked front whole, and only a
/// fold ever runs on more workers than it started with. Whether a fold finds
/// idle workers depends on how the tasks happen to finish, so that a loan
/// reaches each slice is asserted on a fold alone, in the library's
/// `a_fold_borrows_once_per_slice_and_keeps_its_front`.
#[test]
fn lending_idle_workers_to_folds_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let lending = WalkOptions {
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        lend_idle_workers: true,
        ..WalkOptions::default()
    };
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walked =
            design_front_walk(&params, protein, &usage, 1e-4, 0.999, lending.clone(), 4).unwrap();
        assert_same(protein, &full, &axes(&walked.front));
        assert_real(protein, &params, &walked.front);
        for span in &walked.timeline {
            assert!(span.peak_workers >= span.workers && span.peak_workers <= 4);
            if span.kind != TaskKind::Fold {
                assert_eq!(span.peak_workers, span.workers, "{protein}");
            }
        }
    }
}

/// Spending idle workers on a fold's next slices at once leaves the walked
/// front whole: each slice run early only has a lower floor. That a loan
/// reaches the slices is asserted on a fold alone, in the library's
/// `speculative_slices_keep_the_serial_front`.
#[test]
fn speculative_slices_keep_the_whole_front() {
    let (params, usage) = fixtures();
    let speculating = WalkOptions {
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        slice_width: Some(50.0),
        speculative_slices: true,
        ..WalkOptions::default()
    };
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walked = design_front_walk(
            &params,
            protein,
            &usage,
            1e-4,
            0.999,
            speculating.clone(),
            4,
        )
        .unwrap();
        assert_same(protein, &full, &axes(&walked.front));
        assert_real(protein, &params, &walked.front);
    }
}

/// A chord-cut fold that changes to an exact pass at its own tie weight part
/// way leaves the walked front whole, and so does a replayed gap changed at
/// its second slice. On one worker a fold always holds every worker, so with
/// no price on the change every fold cut by a chord between two passes that
/// has three or more slices changes at its second.
#[test]
fn a_fold_changing_to_its_own_pass_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let width = 20.0;
    let changing = WalkOptions {
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        slice_width: Some(width),
        own_when_alone: Some(0.0),
        ..WalkOptions::default()
    };
    let (mut changed, mut replayed) = (0, 0);
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walked =
            design_front_walk(&params, protein, &usage, 1e-4, 0.999, changing.clone(), 1).unwrap();
        assert_same(protein, &full, &axes(&walked.front));
        assert_real(protein, &params, &walked.front);
        for gap in &walked.gaps {
            let chord = match gap.cut {
                GapCut::Chord { lo, hi } if lo < hi => Some((lo, hi)),
                _ => None,
            };
            let expected = (chord.is_some() && gap.slices >= 3).then_some(2);
            assert_eq!(
                gap.clock.own_from.map(|(slice, _)| slice),
                expected,
                "{protein}: gap at {} ({:?}, {} slices)",
                gap.lambda,
                gap.cut,
                gap.slices
            );
            changed += usize::from(expected.is_some());
            if let (Some((lo, hi)), Some(_), 0) = (chord, expected, replayed) {
                let spec = GapSpec {
                    a: gap.a,
                    b: gap.b,
                    lambda: gap.lambda,
                    chord: Some((lo, hi)),
                    bracket: vec![lo, hi],
                    own_from: Some(2),
                };
                let slicing = Slicing {
                    slack: None,
                    width: Some(width),
                };
                let again = replay_gap(&params, protein, &usage, &spec, slicing, 2).unwrap();
                assert_eq!(again.clock.own_from.map(|(slice, _)| slice), Some(2));
                assert_same(protein, &axes(&gap.front), &axes(&again.front));
                replayed += 1;
            }
        }
    }
    assert!(changed > 0, "no fold changed cut");
    assert_eq!(replayed, 1);
}

/// Changing cut on cost alone, with other tasks running beside the fold,
/// also leaves the walked front whole. Only a fold cut by a chord between two
/// passes with three or more slices asks for its own pass, before its second
/// slice when the change is free. A fold holding every worker runs the pass
/// and changes there; one beside other tasks queues it and changes at a later
/// slice boundary, or never when the pass lands after its last slice. Which
/// of these happens follows the schedule, so only the asking is required
/// here; a queued pass taken by the fold is required by the watched test
/// below, where every ask is queued.
#[test]
fn a_fold_changing_cut_on_cost_beside_other_tasks_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let changing = WalkOptions {
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        slice_width: Some(20.0),
        own_when_costly: Some(0.0),
        ..WalkOptions::default()
    };
    let (mut asks, mut ran, mut queued) = (0, 0, 0);
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walked =
            design_front_walk(&params, protein, &usage, 1e-4, 0.999, changing.clone(), 4).unwrap();
        assert_same(protein, &full, &axes(&walked.front));
        assert_real(protein, &params, &walked.front);
        for gap in &walked.gaps {
            let label = format!(
                "{protein}: gap at {} ({:?}, {} slices)",
                gap.lambda, gap.cut, gap.slices
            );
            let Some(asked) = gap.clock.own_asked else {
                assert_eq!(gap.clock.own_from, None, "{label}");
                continue;
            };
            assert!(
                matches!(gap.cut, GapCut::Chord { lo, hi } if lo < hi) && gap.slices >= 3,
                "{label}"
            );
            assert_eq!(asked, 2, "{label}");
            asks += 1;
            match gap.clock.own_from {
                Some((slice, _)) if slice == asked => ran += 1,
                Some((slice, _)) => {
                    assert!(slice > asked && slice <= gap.slices, "{label}");
                    queued += 1;
                }
                None => {}
            }
        }
    }
    eprintln!("asked {asks}; changed cut: {ran} running the pass, {queued} after queueing it");
    assert!(asks > 0, "no fold asked for its own pass");
}

/// The root looseness the looseness test holds folds to, kcal/mol: between
/// the chords its walks read with no threshold (0.02–0.81), four on each side.
const MIN_LOOSENESS: f64 = 0.5;

/// How loose the chord of `lo` and `hi` cuts at `lambda` over the exact
/// `front`, in kcal/mol. A walk reads it over the vertices it has found, a
/// subset, so its looseness is never above this one.
fn root_looseness(front: &[Axes], lambda: f64, lo: f64, hi: f64) -> f64 {
    let g = |w: f64| {
        front
            .iter()
            .map(|p| w * f64::from(p.0) - 100.0 * (1.0 - w) * p.1)
            .fold(f64::INFINITY, f64::min)
            / 100.0
    };
    let t = (hi - lambda) / (hi - lo);
    (g(lambda) - (t * g(lo) + (1.0 - t) * g(hi))) / lambda
}

/// Changing cut on cost only where the chord is loose: a fold whose chord is
/// tighter than the threshold never asks for its own pass, and the walked
/// front stays whole. Both a fold that asks and a tight one held back must
/// occur for the comparison to say anything about the threshold.
#[test]
fn a_fold_changing_cut_only_under_a_loose_chord_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let changing = WalkOptions {
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        slice_width: Some(20.0),
        own_when_costly: Some(0.0),
        own_min_looseness: Some(MIN_LOOSENESS),
        ..WalkOptions::default()
    };
    let (mut asks, mut held) = (0, 0);
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walked =
            design_front_walk(&params, protein, &usage, 1e-4, 0.999, changing.clone(), 4).unwrap();
        assert_same(protein, &full, &axes(&walked.front));
        assert_real(protein, &params, &walked.front);
        for gap in &walked.gaps {
            let GapCut::Chord { lo, hi } = gap.cut else {
                continue;
            };
            if lo >= hi || gap.slices < 3 {
                continue;
            }
            let loose = root_looseness(&full, gap.lambda, lo, hi);
            if gap.clock.own_asked.is_some() {
                assert!(loose >= MIN_LOOSENESS, "{protein}: asked at {loose}");
                asks += 1;
            } else if loose < MIN_LOOSENESS {
                assert_eq!(gap.clock.own_from, None, "{protein}");
                held += 1;
            }
        }
    }
    eprintln!("asked {asks}; tight chords held back {held}");
    assert!(asks > 0, "no fold asked for its own pass");
    assert!(held > 0, "no tight chord was held back");
}

/// A protein whose gap folds' slices last long enough for a watcher to read
/// one while it runs.
const WATCH_CASES: [&str; 1] = ["MWWKFYCLMWMYGKIIFVLLMCWFGGWCKM"];

/// How many times the watched walk may run before one of its folds changes to
/// a pass it asked for during a slice. Whether a watcher reads a slice before
/// it ends, and whether the queued pass lands before the fold's last slice,
/// depend on the host's scheduling; a third of single walks miss on a loaded
/// host, so six all missing is under one in five hundred.
const WATCH_TRIES: usize = 6;

/// Watching a running slice's CPU, a fold beside other tasks asks for its own
/// pass during its first slice, and the walked front stays whole. Only a fold
/// cut by a chord between two passes with three or more slices asks, for the
/// boundary before its second slice, and changes there or later: every such
/// ask is queued, so a change here is a queued pass taken by the fold.
///
/// The ask itself needs a clock that names another thread, which
/// [`ThreadCpu::AVAILABLE`] reports and macOS does not have; the walk runs
/// the watched options everywhere regardless, so the front stays under test
/// on every platform and only the ask is asserted where it can happen.
#[test]
fn a_fold_asking_for_its_pass_during_a_slice_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let watching = WalkOptions {
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        slice_width: Some(20.0),
        own_when_costly: Some(0.0),
        own_watch_s: Some(1e-5),
        ..WalkOptions::default()
    };
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        watched_walk(&params, &usage, protein, &watching);
    }
    // Without a clock naming another thread the watch does not run at all,
    // so no fold can ask and there is nothing here to assert. The walks above
    // have already checked the front is whole under the same options.
    if !ThreadCpu::AVAILABLE {
        return;
    }
    // The watcher must read a slice while it runs, so the watched case is one
    // long enough that its slices can outlast the watcher's first read.
    let mut seen = Vec::new();
    for _ in 0..WATCH_TRIES {
        let (mid, taken) = watched_walk(&params, &usage, WATCH_CASES[0], &watching);
        seen.push((mid, taken));
        if taken > 0 {
            break;
        }
    }
    eprintln!("(asked during a slice, taken) per walk: {seen:?}");
    assert!(
        seen.iter().any(|&(_, taken)| taken > 0),
        "no fold changed to a pass it queued during a slice: {seen:?}"
    );
}

/// One walk of `protein` under `watching`: its front whole and real, every ask
/// where only a chord-cut fold of three or more slices can make it, and the
/// counts of folds that asked during a slice and that took that pass.
fn watched_walk(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    watching: &WalkOptions,
) -> (usize, usize) {
    let full = axes(&design_front(params, protein, usage).unwrap());
    let walked =
        design_front_walk(params, protein, usage, 1e-4, 0.999, watching.clone(), 4).unwrap();
    assert_same(protein, &full, &axes(&walked.front));
    assert_real(protein, params, &walked.front);
    let (mut mid, mut taken) = (0, 0);
    for gap in &walked.gaps {
        let label = format!(
            "{protein}: gap at {} ({:?}, {} slices)",
            gap.lambda, gap.cut, gap.slices
        );
        let Some(asked) = gap.clock.own_asked else {
            assert_eq!(gap.clock.own_from, None, "{label}");
            assert!(!gap.clock.own_asked_mid_slice, "{label}");
            continue;
        };
        assert!(
            matches!(gap.cut, GapCut::Chord { lo, hi } if lo < hi) && gap.slices >= 3,
            "{label}"
        );
        assert_eq!(asked, 2, "{label}");
        if let Some((slice, _)) = gap.clock.own_from {
            assert!(slice >= asked && slice <= gap.slices, "{label}");
        }
        mid += usize::from(gap.clock.own_asked_mid_slice);
        taken += usize::from(gap.clock.own_asked_mid_slice && gap.clock.own_from.is_some());
    }
    (mid, taken)
}

/// Proteins long enough that multi-loop openings survive a gap fold's box; at
/// the shorter cases the cuts remove every two-branch core first.
const SCREEN_CASES: [&str; 1] = ["MWWKFYCLMWMYGKIIFVLMMCWFGGWCKMMYGKIIFVLLMYGKIIFVLLL"];

/// A gap fold skips closing a multi-loop opening wherever a cut removes the
/// whole closed front, enumerating a hairpin wherever a cut removes every
/// interior, and closing a stack, bulge or internal loop wherever a cut
/// removes everything it would make, and building an opening only once a
/// close might keep one of its points, and the walked front stays whole. Every
/// skip must fire, and some opening must stay unbuilt, for the comparison to
/// say anything about it.
#[test]
fn a_fold_skipping_what_the_cuts_remove_keeps_the_whole_front() {
    let (params, usage) = fixtures();
    let options = WalkOptions {
        cut_probes: true,
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        slice_width: Some(20.0),
        ..WalkOptions::default()
    };
    let (mut screened, mut boxed, mut points) = (0, 0, 0);
    let (mut hairpins, mut singles, mut shorts) = (0, 0, 0);
    let (mut recipes, mut built) = (0, 0);
    for protein in SCREEN_CASES {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        let walked =
            design_front_walk(&params, protein, &usage, 1e-4, 0.999, options.clone(), 4).unwrap();
        assert_same(protein, &full, &axes(&walked.front));
        assert_real(protein, &params, &walked.front);
        for gap in &walked.gaps {
            screened += gap.clock.fold.open_close_screened;
            boxed += gap.clock.fold.open_close_boxed;
            points += gap.clock.fold.open_close_points;
            hairpins += gap.clock.fold.hairpin_screened;
            singles += gap.clock.fold.single_screened;
            shorts += gap.clock.fold.short_screened;
            recipes += gap.clock.fold.open_recipes;
            built += gap.clock.fold.open_built;
        }
    }
    eprintln!(
        "closed-opening points screened: {screened} of {boxed} boxed, {points} closed; \
         hairpins screened: {hairpins}; single-branch points screened: {singles}; \
         short-flank points screened: {shorts}; openings built: {built} of {recipes}"
    );
    assert!(screened > 0, "no close was skipped");
    assert!(screened <= boxed, "{screened} screened, {boxed} boxed");
    assert!(hairpins > 0, "no hairpin was skipped");
    assert!(singles > 0, "no single-branch close was skipped");
    assert!(shorts > 0, "no short-flank close was skipped");
    assert!(
        built < recipes,
        "every opening was built: {built} of {recipes}"
    );
}

#[test]
fn the_tie_weight_levels_the_chord_and_an_unordered_pair_is_refused() {
    let (params, usage) = fixtures();
    let protein = WIDE_CASES[0];
    let full = axes(&design_front(&params, protein, &usage).unwrap());
    let vertices = hull(&full);
    assert!(vertices.len() >= 3, "{vertices:?}");
    for pair in vertices.windows(2) {
        let (a, b) = (full[pair[0]], full[pair[1]]);
        let lambda = tie_weight(hull_point(a), hull_point(b)).unwrap();
        assert!(0.0 < lambda && lambda < 1.0, "{lambda}");
        let score = |p: Axes| lambda * f64::from(p.0) - 100.0 * (1.0 - lambda) * p.1;
        assert!(
            (score(a) - score(b)).abs() < 1e-6,
            "{a:?} {b:?} at {lambda}"
        );
        // Every front point scores at least the tie value: the edge is on the hull.
        for &p in &full {
            assert!(
                score(p) >= score(a) - 1e-6,
                "{p:?} below the edge {a:?}–{b:?}"
            );
        }
    }
    let (a, b) = (hull_point(full[vertices[0]]), hull_point(full[vertices[1]]));
    let same_energy = HullPoint {
        mfe_kcal_mol: a.mfe_kcal_mol,
        ..b
    };
    for (x, y) in [(b, a), (a, a), (a, same_energy)] {
        assert_eq!(
            design_front_between(&params, protein, &usage, x, y, 1),
            Err(BeamError::UnorderedHullPair),
        );
    }
}

/// The hull-gap solve on the flanked lattice with a narrowed designed flank,
/// whose normalization offset separates the floor as stated from the one the
/// cut compares.
#[test]
fn between_on_a_flanked_molecule_with_a_narrowed_alphabet() {
    let (params, usage) = fixtures();
    let profile = UtrProfile::new([0.2, 0.25, 0.3, 0.6], 1.0).unwrap();
    let (utr5, utr3) = (UtrSpec::Mixed("SSA".into()), UtrSpec::Fixed("AU".into()));
    let mut pairs = 0;
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&feasible(
            design_front_bounded_with_utr(
                &params,
                protein,
                &usage,
                &utr5,
                &utr3,
                profile,
                Prune::None,
                Bounds::UNBOUNDED,
                1,
            )
            .unwrap(),
        ));
        let vertices = hull(&full);
        for (k, &i) in vertices.iter().enumerate() {
            for &j in &vertices[k + 1..] {
                let got = feasible(
                    design_front_between_with_utr(
                        &params,
                        protein,
                        &usage,
                        &utr5,
                        &utr3,
                        profile,
                        hull_point(full[i]),
                        hull_point(full[j]),
                        1,
                    )
                    .unwrap(),
                );
                let label = format!("{protein} flanked between {i} and {j}");
                assert_same(&label, &between_of(&full, i, j), &axes(&got));
                pairs += 1;
            }
        }
    }
    assert!(pairs >= 5, "{pairs}");
}

/// The clocked entry point answers exactly what the box and the bounded solve
/// answer, and its counts are on: the openings hold points, their closes
/// produce some, and the box's cuts remove no more than the closes produced.
#[test]
fn the_clocked_solve_answers_the_same_and_counts_the_openings() {
    let (params, usage) = fixtures();
    let protein = "MKTAYIAKQRQISFVKSHFSRQ";
    let optimum = achievable_minimum(&params, protein, &usage).unwrap();
    // A third of the optimum: at a tenth the credit empties every multi-loop
    // core of this protein, and there would be no opening to count.
    let ceiling = Ceiling::AboveOptimum(Energy(optimum.0.abs() / 3));
    let bounds = Bounds::with_ceiling(ceiling).with_exact_credit();
    let bounded =
        feasible(design_front_bounded(&params, protein, &usage, Prune::None, bounds, 1).unwrap());
    let band = BoxSpec {
        cai_floor_log: bounded[bounded.len() / 3].cai_log,
        lambdas: vec![0.5],
    };
    let boxed = feasible(design_front_box(&params, protein, &usage, ceiling, &band, 1).unwrap());
    for (spec, want) in [(None, &bounded), (Some(&band), &boxed)] {
        let (solved, clock) = clock_front(&params, protein, &usage, bounds, spec, 1).unwrap();
        assert_eq!(&feasible(solved), want, "{spec:?}");
        let f = clock.fold;
        assert!(clock.credit_scan_s > 0.0 && clock.credit_pass_s > 0.0);
        assert_eq!(clock.cuts.len(), spec.map_or(0, |b| b.lambdas.len()));
        assert!(
            f.open_arrivals > 0 && f.open_points_held > 0 && f.open_close_points > 0,
            "{f:?}"
        );
        assert!(f.open_close_boxed <= f.open_close_points, "{f:?}");
        assert_eq!(f.open_close_boxed > 0, spec.is_some(), "{f:?}");
        assert!(
            f.total_s() <= clock.fold_s + 1e-3,
            "{f:?} vs {}",
            clock.fold_s
        );
    }
}

/// With empty flanks and no region constraint, the flanked walk is the
/// coding-only walk. `design_front_walk` is now that call, so it cannot serve
/// as its own reference; the reference is instead the coding-only lattice and
/// corner as `design_front` and `design_front_between` still build them. The
/// walk recovers the coding-only front, and every gap's front is, byte for
/// byte, the front `design_front_between` returns for the same pair.
#[test]
fn empty_flanks_reproduce_the_coding_only_walk() {
    let (params, usage) = fixtures();
    let none = UtrSpec::Fixed(String::new());
    for protein in CASES.iter().chain(WIDE_CASES.iter()) {
        let full = axes(&design_front(&params, protein, &usage).unwrap());
        for threads in [1, 4] {
            let walked = design_front_walk_with_utr(
                &params,
                protein,
                &usage,
                &none,
                &none,
                RegionConstraints::default(),
                1e-4,
                0.999,
                WalkOptions::default(),
                threads,
            )
            .unwrap();
            let label = format!("{protein} on {threads} workers");
            assert_same(&label, &full, &axes(&walked.front));
            for gap in &walked.gaps {
                let between = bounded_front(
                    design_front_between(&params, protein, &usage, gap.a, gap.b, 1).unwrap(),
                );
                assert_eq!(gap.front, between.front, "{label}: gap {:?}", gap.lambda);
            }
        }
    }
}
