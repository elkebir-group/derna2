//! The window walk's modular front against exhaustive
//! enumeration.
//!
//! Every encoding of a protein small enough to enumerate is folded twice, under
//! the constraint and free; those whose two optima agree are the modular
//! set `S`, and the Pareto front of `(constrained energy, adaptation)` over
//! `S` inside a window is what the walk must return, point for point. Windows
//! are cut at the quantiles of the designs' adaptations, so some hold front
//! points off the constrained front and some hold none.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::outside::lazy::window::{
    adaptation, window_front, window_front_sharing, PassCache, WindowReport,
};
use derna_lib::outside::lazy::{Build, ExactFold, Verify, Walk};
use derna_lib::{
    sequence_constrained_mfe, sequence_gap, CdsWindow, CodonUsage, Enforcement, PairClass, Params,
    RegionConstraints, ScanConfig,
};

const CAI_TOL: f64 = 1e-9;

fn full_sets(protein: &str) -> Vec<Vec<[u8; 3]>> {
    protein
        .chars()
        .map(|c| {
            let a = aa_to_index(c).expect("a standard residue");
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

fn encodings(sets: &[Vec<[u8; 3]>]) -> Vec<String> {
    let mut out = vec![String::new()];
    for set in sets {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                set.iter().map(move |c| {
                    let mut e = prefix.clone();
                    e.extend(c.iter().map(|&n| ['A', 'C', 'G', 'U'][n as usize]));
                    e
                })
            })
            .collect();
    }
    out
}

struct Case<'a> {
    protein: &'a str,
    utr5: &'a str,
    utr3: &'a str,
    constraints: RegionConstraints,
}

/// One encoding: constrained energy (dekacalories), adaptation, and whether it
/// is modular.
#[derive(Debug, Clone)]
struct Point {
    energy: i32,
    cai: f64,
    in_s: bool,
    seq: String,
}

fn enumerate(params: &Params, usage: &CodonUsage, case: &Case<'_>) -> Vec<Point> {
    let sets = full_sets(case.protein);
    let cds = CdsWindow {
        start: case.utr5.len(),
        end: case.utr5.len() + 3 * case.protein.len(),
    };
    encodings(&sets)
        .into_iter()
        .map(|coding| {
            let seq = format!("{}{coding}{}", case.utr5, case.utr3);
            let cai = adaptation(case.protein, usage, &sets, case.utr5, case.utr3, &seq)
                .unwrap()
                .expect("an encoding is on its lattice");
            let constrained =
                sequence_constrained_mfe(params, &seq, cds, case.constraints).unwrap();
            let in_s = sequence_gap(params, &seq, cds, case.constraints)
                .unwrap()
                .is_modular();
            Point {
                energy: (constrained * 100.0).round() as i32,
                cai,
                in_s,
                seq,
            }
        })
        .collect()
}

/// The Pareto front of `(energy, cai)` over `points`, energy ascending.
fn pareto<'p>(points: impl Iterator<Item = &'p Point>) -> Vec<(i32, f64)> {
    let mut sorted: Vec<(i32, f64)> = points.map(|p| (p.energy, p.cai)).collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut front: Vec<(i32, f64)> = Vec::new();
    for (e, c) in sorted {
        if front.last().is_some_and(|l| l.1 >= c - CAI_TOL) {
            continue;
        }
        front.push((e, c));
    }
    front
}

#[allow(clippy::too_many_arguments)]
fn run(
    params: &Params,
    usage: &CodonUsage,
    case: &Case<'_>,
    (lo, hi, known): (f64, f64, &[(i32, f64)]),
    (mu, energy): (f64, &[f64]),
    max_derivations: u64,
    verify: Verify,
    walk: Walk,
) -> WindowReport {
    window_front(
        params,
        case.protein,
        usage,
        &full_sets(case.protein),
        case.utr5,
        case.utr3,
        case.constraints,
        ScanConfig::default(),
        (lo, hi),
        mu,
        known,
        energy,
        max_derivations,
        f64::INFINITY,
        verify,
        walk,
        None,
    )
    .expect("the walk runs")
    .expect("a lattice")
}

/// The energy bound's slopes for a walk at slope 30: below, at and far
/// above it, and the pure-energy slope.
const ENERGY: [f64; 5] = [0.0, 10.0, 60.0, 300.0, 3000.0];

const DEDUP: Walk = Walk {
    build: Build::Production,
    node_dedup: true,
    incremental_spell: true,
    pipelined_visit: false,
    spellers: 0,
    dedup_index: false,
    tight_lists: false,
};

fn wide() -> Verify {
    Verify {
        threads: 4,
        witnesses: 16,
        witness_stop: 1,
        witness_promote: true,
        fold_slots: 1,
        window_slots: 0,
        handoff_batch: 8,
        fold: ExactFold::SparseCut,
        ..Verify::folds(4)
    }
}

/// What the cases exercised, summed: front points off the constrained front,
/// windows holding no point, and derivations the bound cut.
#[derive(Default, Debug)]
struct Coverage {
    off_constrained: usize,
    empty_windows: usize,
    cut: u64,
    energy_cut: u64,
    closed: u64,
    skipped: u64,
    budget_stops: usize,
    capped: usize,
    resumed: usize,
    tight: usize,
    cut_outside: u64,
    cut_covered: u64,
    window_deferred: u64,
    window_copies: u64,
}

/// Every cut the bound makes has one reason: the derivation's completions
/// all lie outside the window, or all in-window ones lie in a covered span,
/// or the energy bound dominates them, or it closed the cell. Each test sees
/// what the one before it let through, and no more.
fn cuts_by_reason(got: &WindowReport, cov: &mut Coverage) {
    let s = &got.stats;
    assert_eq!(
        s.bound_cut,
        s.bound_outside_cut + s.bound_covered_cut + s.bound_energy_cut + s.bound_nodes_closed,
        "{s:?}"
    );
    assert_eq!(
        s.bound_covered_tested,
        s.bound_outside_tested - s.bound_outside_cut,
        "{s:?}"
    );
    assert!(
        s.bound_energy_tested <= s.bound_covered_tested - s.bound_covered_cut,
        "{s:?}"
    );
    cov.cut_outside += s.bound_outside_cut;
    cov.cut_covered += s.bound_covered_cut;
}

fn assert_windows(case: &Case<'_>, cov: &mut Coverage) {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let all = enumerate(&params, &usage, case);
    let constrained_front = pareto(all.iter());
    let mut cais: Vec<f64> = all.iter().map(|p| p.cai).collect();
    cais.sort_by(f64::total_cmp);
    let q = |f: f64| cais[((cais.len() - 1) as f64 * f) as usize];
    let (min, max) = (cais[0], cais[cais.len() - 1]);
    let windows = [
        (min, max),
        (q(0.5), max),
        (q(0.25), q(0.75)),
        (min, q(0.3)),
        (q(0.9), max),
    ];
    for (wi, (lo, hi)) in windows.into_iter().enumerate() {
        let want = pareto(
            all.iter()
                .filter(|p| p.in_s && p.cai >= lo - CAI_TOL && p.cai <= hi + CAI_TOL),
        );
        let same = |label: &str, got: &WindowReport, want: &[(i32, f64)]| {
            let pts: Vec<(i32, f64)> = got.front.iter().map(|p| (p.energy, p.cai)).collect();
            assert!(
                got.complete,
                "{}: {label} [{lo}, {hi}] incomplete",
                case.protein
            );
            assert!(
                pts.len() == want.len()
                    && pts
                        .iter()
                        .zip(want)
                        .all(|(g, w)| g.0 == w.0 && (g.1 - w.1).abs() < 1e-6),
                "{}: {label} [{lo}, {hi}] got {pts:?} want {want:?}",
                case.protein
            );
            for p in &got.front {
                let e = all
                    .iter()
                    .find(|x| x.seq == p.sequence)
                    .expect("an encoding");
                assert!(
                    e.in_s && e.energy == p.energy,
                    "{}: the reported design",
                    p.sequence
                );
            }
        };
        // By energy under every walk and check; at two slopes, one with the
        // windowed fold in front of the exact one, and with the energy bound.
        let configs: [(f64, &[f64], Verify, Walk); 12] = [
            (0.0, &[], Verify::folds(1), Walk::from(Build::Production)),
            (0.0, &[], Verify::folds(1), DEDUP),
            (0.0, &[], wide(), DEDUP),
            (30.0, &[], wide(), DEDUP),
            (
                30.0,
                &[],
                Verify {
                    window: Some((3, 3)),
                    ..wide()
                },
                DEDUP,
            ),
            // One windowed fold at a time: the other workers' designs wait
            // and are offered its structure; then with windowed folds
            // copying from recent ones as well.
            (
                30.0,
                &[],
                Verify {
                    window: Some((3, 3)),
                    window_slots: 1,
                    ..wide()
                },
                DEDUP,
            ),
            (
                30.0,
                &[],
                Verify {
                    window: Some((3, 3)),
                    window_slots: 1,
                    window_reuse: true,
                    fold_reuse: 8,
                    ..wide()
                },
                DEDUP,
            ),
            (300.0, &[], Verify::folds(1), DEDUP),
            (30.0, &ENERGY, Verify::folds(1), DEDUP),
            (0.0, &ENERGY, wide(), DEDUP),
            // Each design's adaptation and the bound's count reach the
            // visitor through a piped visitor and through spellers too.
            (
                30.0,
                &[],
                wide(),
                Walk {
                    pipelined_visit: true,
                    ..DEDUP
                },
            ),
            (
                30.0,
                &[],
                wide(),
                Walk {
                    spellers: 2,
                    ..DEDUP
                },
            ),
        ];
        for (mu, energy, verify, walk) in configs {
            let got = run(
                &params,
                &usage,
                case,
                (lo, hi, &[]),
                (mu, energy),
                u64::MAX,
                verify,
                walk,
            );
            same(&format!("slope {mu}, energy {energy:?}"), &got, &want);
            cuts_by_reason(&got, cov);
            cov.cut += got.stats.bound_cut;
            cov.energy_cut += got.energy_cut;
            cov.closed += got.stats.bound_nodes_closed;
            cov.window_deferred += got.window_deferred;
        }
        // The windowed-fold cache is sized apart from the full-fold one: four
        // windowed folds kept in front of a full cache at its production size
        // return the window's whole front, and so does a windowed cache of
        // none beside the same full cache. With no witnesses every design
        // reaches a windowed fold, so the small cache is populated, consulted,
        // and written over, four entries against more designs than that.
        //
        // The pair is the equality's own control: the cells copied must be
        // positive on the first and zero on the second, since the one
        // difference between them is the windowed cache's size. Without it
        // the equality would read the same on a cache nothing ever consulted.
        //
        // The widest window only (`wi == 0`): it holds every design, so it
        // folds the most and copies the most, and a run with no witnesses and
        // 512 folds kept is the costliest configuration in this file.
        let sizes: &[(usize, bool)] = if wi == 0 {
            &[(4, true), (0, false)]
        } else {
            &[]
        };
        for &(size, copies) in sizes {
            let got = run(
                &params,
                &usage,
                case,
                (lo, hi, &[]),
                (30.0, &[]),
                u64::MAX,
                Verify {
                    witnesses: 0,
                    window: Some((3, 3)),
                    window_slots: 1,
                    window_reuse: true,
                    fold_reuse: 512,
                    window_reuse_size: Some(size),
                    ..wide()
                },
                DEDUP,
            );
            same(&format!("windowed cache of {size}"), &got, &want);
            cuts_by_reason(&got, cov);
            cov.cut += got.stats.bound_cut;
            cov.closed += got.stats.bound_nodes_closed;
            cov.window_deferred += got.window_deferred;
            if copies {
                cov.window_copies += got.window_cells_copied;
            } else {
                assert_eq!(
                    got.window_cells_copied, 0,
                    "{}: [{lo}, {hi}] a windowed cache of none copied cells, so the size \
                     the check reads is not the windowed one",
                    case.protein
                );
            }
        }
        // Stand-ins one dekacalorie above every front point dominate nothing
        // the front holds, so the front is unchanged, and every stair the
        // energy bound reads is as tight as whole energies allow. One more at
        // `hi`, above every window design's energy, gives the stairs a top.
        if !want.is_empty() {
            let mut tight: Vec<(i32, f64)> = want.iter().map(|&(e, a)| (e + 1, a)).collect();
            let top = all
                .iter()
                .filter(|p| p.cai >= lo - CAI_TOL && p.cai <= hi + CAI_TOL)
                .map(|p| p.energy)
                .max()
                .expect("a window design");
            tight.push((top + 1, hi));
            for (mu, energy) in [
                (30.0, &ENERGY[..]),
                (300.0, &ENERGY[..]),
                (0.0, &ENERGY[..]),
                (30.0, &[][..]),
            ] {
                let got = run(
                    &params,
                    &usage,
                    case,
                    (lo, hi, &tight),
                    (mu, energy),
                    u64::MAX,
                    Verify::folds(1),
                    DEDUP,
                );
                same(
                    &format!("tight stairs, slope {mu}, energy {energy:?}"),
                    &got,
                    &want,
                );
                cuts_by_reason(&got, cov);
                cov.energy_cut += got.energy_cut;
                cov.closed += got.stats.bound_nodes_closed;
                cov.skipped += got.stats.bound_edges_skipped;
                cov.tight += 1;
            }
        }
        cov.off_constrained += want
            .iter()
            .filter(|w| {
                !constrained_front
                    .iter()
                    .any(|g| g.0 == w.0 && (g.1 - w.1).abs() < 1e-6)
            })
            .count();
        cov.empty_windows += usize::from(want.is_empty());
        // A modular design at or above the window's top
        // dominates every window design at or above its energy: known to the
        // walk, it leaves the window's front below it, complete.
        let incumbent = all
            .iter()
            .filter(|p| p.in_s && p.cai >= hi - CAI_TOL)
            .min_by_key(|p| p.energy);
        if let Some(inc) = incumbent {
            // The incumbent dominates every window point at or above its
            // energy (it has at least their adaptation).
            let below: Vec<(i32, f64)> = want
                .iter()
                .copied()
                .filter(|w| w.0 < inc.energy || (w.0 == inc.energy && w.1 >= inc.cai - CAI_TOL))
                .collect();
            for (mu, energy) in [(0.0, &[][..]), (30.0, &[]), (30.0, &ENERGY), (0.0, &ENERGY)] {
                let capped = run(
                    &params,
                    &usage,
                    case,
                    (lo, hi, &[(inc.energy, inc.cai)]),
                    (mu, energy),
                    u64::MAX,
                    wide(),
                    DEDUP,
                );
                same(
                    &format!("capped, slope {mu}, energy {energy:?}"),
                    &capped,
                    &below,
                );
                cov.capped += usize::from(capped.capped);
                cov.energy_cut += capped.energy_cut;
                cov.closed += capped.stats.bound_nodes_closed;
            }
        }
        // Cut short, every point found is a front point, and every point
        // missing has an open adaptation and an energy of at least
        // `last_g + slope·A`.
        for mu in [0.0, 30.0] {
            let full = run(
                &params,
                &usage,
                case,
                (lo, hi, &[]),
                (mu, &[]),
                u64::MAX,
                Verify::folds(1),
                DEDUP,
            );
            for budget in [1u64, 3, 10, 30] {
                if budget >= full.stats.derivations {
                    break;
                }
                let short = run(
                    &params,
                    &usage,
                    case,
                    (lo, hi, &[]),
                    (mu, &[]),
                    budget,
                    Verify::folds(1),
                    DEDUP,
                );
                // Verdicts landing after the walk stopped can complete it.
                if short.complete {
                    same(&format!("budget {budget}, complete"), &short, &want);
                    continue;
                }
                cov.budget_stops += 1;
                // Resumed with what it found, a second walk adds the rest.
                let known: Vec<(i32, f64)> =
                    short.front.iter().map(|p| (p.energy, p.cai)).collect();
                let rest = run(
                    &params,
                    &usage,
                    case,
                    (lo, hi, &known),
                    (mu, &[]),
                    u64::MAX,
                    Verify::folds(1),
                    DEDUP,
                );
                assert!(rest.complete, "{}: resumed walk incomplete", case.protein);
                // The energy bound reads the known designs from the start; it
                // adds the same rest.
                let bounded = run(
                    &params,
                    &usage,
                    case,
                    (lo, hi, &known),
                    (mu, &ENERGY),
                    u64::MAX,
                    Verify::folds(1),
                    DEDUP,
                );
                let points = |r: &WindowReport| -> Vec<(i32, f64)> {
                    r.front.iter().map(|p| (p.energy, p.cai)).collect()
                };
                let (a, b) = (points(&rest), points(&bounded));
                assert!(
                    bounded.complete
                        && a.len() == b.len()
                        && a.iter()
                            .zip(&b)
                            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-6),
                    "{}: [{lo}, {hi}] resumed under the energy bound {b:?}, without {a:?}",
                    case.protein
                );
                cov.energy_cut += bounded.energy_cut;
                let mut union: Vec<(i32, f64)> = known.clone();
                union.extend(rest.front.iter().map(|p| (p.energy, p.cai)));
                union.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
                let mut merged: Vec<(i32, f64)> = Vec::new();
                for (e, c) in union {
                    if merged.last().is_some_and(|l| l.1 >= c - CAI_TOL) {
                        continue;
                    }
                    merged.push((e, c));
                }
                assert!(
                    merged.len() == want.len()
                        && merged
                            .iter()
                            .zip(&want)
                            .all(|(g, w)| g.0 == w.0 && (g.1 - w.1).abs() < 1e-6),
                    "{}: [{lo}, {hi}] resumed {merged:?} want {want:?}",
                    case.protein
                );
                cov.resumed += 1;
                for p in &short.front {
                    assert!(
                        want.iter()
                            .any(|w| w.0 == p.energy && (w.1 - p.cai).abs() < 1e-6),
                        "{}: [{lo}, {hi}] budget {budget}: {:?} is not a front point",
                        case.protein,
                        (p.energy, p.cai)
                    );
                }
                let g = short.last_g.expect("a design was reached");
                for w in &want {
                    let found = short
                        .front
                        .iter()
                        .any(|p| p.energy == w.0 && (p.cai - w.1).abs() < 1e-6);
                    let open = short
                        .open
                        .iter()
                        .any(|&(x, y)| x - 1e-6 <= w.1 && w.1 <= y + 1e-6);
                    assert!(
                        found || (open && f64::from(w.0) >= g + short.slope * w.1 - 1e-6),
                        "{}: [{lo}, {hi}] budget {budget}: {w:?} missing outside the open region \
                         {:?} at g {g}",
                        case.protein,
                        short.open
                    );
                }
            }
        }
    }
}

/// Windowed slots change which designs are folded, never the front. With no
/// witnesses every design reaches a windowed fold, so four workers sharing
/// one slot leave designs waiting; the front's objectives then match a run
/// with no slots. Ties may be spelled differently, as at another thread
/// count, so only the objectives are compared. Some cases' whole range holds
/// no modular point, so the non-vacuity checks are over all of them.
#[test]
fn windowed_slots_change_no_front() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let three_class = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
    let cases = [
        Case {
            protein: "MLRS",
            utr5: "AAAAAA",
            utr3: "AAAAAA",
            constraints: RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        },
        Case {
            protein: "MLRS",
            utr5: "GGGAUCCG",
            utr3: "AAAAAAAA",
            constraints: RegionConstraints::five_prime_isolation(Enforcement::Forbid)
                .with(PairClass::Utr5Utr3, Enforcement::Forbid),
        },
        Case {
            protein: "MLRS",
            utr5: "GGGAUC",
            utr3: "GAUCCC",
            constraints: three_class,
        },
        Case {
            protein: "GLRSA",
            utr5: "GGCGC",
            utr3: "GCGCC",
            constraints: three_class,
        },
    ];
    let check = |window_slots: usize, window_reuse: bool| Verify {
        witnesses: 0,
        window: Some((3, 3)),
        window_slots,
        window_reuse,
        fold_reuse: if window_reuse { 8 } else { 0 },
        ..wide()
    };
    let objectives = |r: &WindowReport| -> Vec<(i32, i64)> {
        r.front
            .iter()
            .map(|p| (p.energy, (p.cai * 1e9).round() as i64))
            .collect()
    };
    let (mut points, mut window_rejects, mut waited) = (0, 0, [0u64; 2]);
    for case in &cases {
        let cais: Vec<f64> = enumerate(&params, &usage, case)
            .iter()
            .map(|p| p.cai)
            .collect();
        let lo = cais.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = cais.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let walk = |verify: Verify| {
            run(
                &params,
                &usage,
                case,
                (lo, hi, &[]),
                (30.0, &[]),
                u64::MAX,
                verify,
                DEDUP,
            )
        };
        let plain = walk(check(0, false));
        assert!(
            plain.complete,
            "{}: the plain walk did not finish",
            case.protein
        );
        points += plain.front.len();
        window_rejects += plain.window_rejects;
        for (i, reuse) in [false, true].into_iter().enumerate() {
            let slotted = walk(check(1, reuse));
            assert!(
                slotted.complete,
                "{}: the slotted walk did not finish",
                case.protein
            );
            assert_eq!(
                objectives(&slotted),
                objectives(&plain),
                "{}: one windowed slot (reuse {reuse}) changed the front",
                case.protein
            );
            waited[i] += slotted.window_deferred;
        }
    }
    eprintln!(
        "front points {points}, windowed rejections {window_rejects}, designs that waited {waited:?}"
    );
    assert!(points > 0, "no case's front held a point");
    assert!(window_rejects > 0, "no windowed fold rejected a design");
    assert!(
        waited.iter().all(|&w| w > 0),
        "no design waited for the one windowed slot: {waited:?}"
    );
}

/// A cut held by the caller is built into once and then reused: every window
/// walked with it returns exactly the front a walk building its own cut
/// returns, and only the first walk pays for the build.
#[test]
fn a_shared_cut_changes_no_window_front() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "GLRSA",
        utr5: "GGCGC",
        utr3: "GCGCC",
        constraints: RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
    };
    let all = enumerate(&params, &usage, &case);
    let mut cais: Vec<f64> = all.iter().map(|p| p.cai).collect();
    cais.sort_by(f64::total_cmp);
    let q = |f: f64| cais[((cais.len() - 1) as f64 * f) as usize];
    let shared = std::sync::OnceLock::new();
    let mut builds = Vec::new();
    for (lo, hi) in [(q(0.0), q(1.0)), (q(0.5), q(1.0)), (q(0.25), q(0.75))] {
        let own = run(
            &params,
            &usage,
            &case,
            (lo, hi, &[]),
            (30.0, &[]),
            u64::MAX,
            wide(),
            DEDUP,
        );
        let with = window_front_sharing(
            &params,
            case.protein,
            &usage,
            &full_sets(case.protein),
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig::default(),
            (lo, hi),
            30.0,
            &[],
            &[],
            u64::MAX,
            f64::INFINITY,
            wide(),
            DEDUP,
            (Some(&shared), 0),
            None,
            None,
        )
        .expect("the walk runs")
        .expect("a lattice");
        assert!(
            shared.get().is_some(),
            "the walk builds into the shared cut"
        );
        let key = |r: &WindowReport| -> Vec<(i32, String)> {
            r.front
                .iter()
                .map(|p| (p.energy, p.sequence.clone()))
                .collect()
        };
        assert_eq!(key(&own), key(&with), "window [{lo}, {hi}]");
        assert_eq!(own.complete, with.complete);
        builds.push(with.cut_s);
    }
    assert!(
        builds[1..].iter().all(|&s| s <= builds[0]),
        "only the first walk builds the cut: {builds:?}"
    );
}

/// `outside` counts the designs the walk reaches beyond its window, and the
/// bound cuts a whole molecule outside it by more than its slack, so only a
/// design within that slack of an edge is ever counted. A window of zero
/// width set just below a design's adaptation plants such designs, and the
/// count is predicted from the enumeration: every encoding whose adaptation
/// lies within the slack of the edge but beyond the walk's tolerance, and
/// nothing else, since nothing lies inside the window to offer or cover.
#[test]
fn outside_counts_the_designs_reached_beyond_the_window() {
    const SLACK: f64 = 1e-7;
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "MLRS",
        utr5: "GGGAUCCG",
        utr3: "AAAAAAAA",
        constraints: RegionConstraints::five_prime_isolation(Enforcement::Forbid),
    };
    let all = enumerate(&params, &usage, &case);
    let mut cais: Vec<f64> = all.iter().map(|p| p.cai).collect();
    cais.sort_by(f64::total_cmp);
    let edge = cais[cais.len() / 2] - SLACK / 2.0;
    let beyond = |c: f64| {
        let d = (c - edge).abs();
        d > CAI_TOL && d <= SLACK
    };
    let want = all.iter().filter(|p| beyond(p.cai)).count() as u64;
    assert!(want > 0, "no design planted at the edge");
    assert!(
        all.iter().all(|p| (p.cai - edge).abs() > 2.0 * CAI_TOL
            && ((p.cai - edge).abs() - SLACK).abs() > 1e-12),
        "a design sits on a tolerance, so the count is not predictable"
    );
    let got = run(
        &params,
        &usage,
        &case,
        (edge, edge, &[]),
        (30.0, &[]),
        u64::MAX,
        Verify::folds(1),
        DEDUP,
    );
    assert_eq!(got.outside, want, "{got:?}");
    assert_eq!(got.reached, want, "{got:?}");
    assert_eq!(
        (got.covered, got.refused, got.offered),
        (0, 0, 0),
        "{got:?}"
    );
}

/// A window of zero width between two designs' adaptations, farther than
/// the bound's slack from both, holds no design and reaches none: the walk
/// derives its way through the lattice and every cut it makes is for lying
/// outside the window. Nothing is reached, so nothing covers a span, and
/// without the energy bound nothing is cut or closed for dominance.
#[test]
fn a_window_between_designs_cuts_only_as_outside() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "MLRS",
        utr5: "GGGAUCCG",
        utr3: "AAAAAAAA",
        constraints: RegionConstraints::five_prime_isolation(Enforcement::Forbid),
    };
    let all = enumerate(&params, &usage, &case);
    let mut cais: Vec<f64> = all.iter().map(|p| p.cai).collect();
    cais.sort_by(f64::total_cmp);
    cais.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let mid = cais.len() / 2;
    let edge = (cais[mid] + cais[mid + 1]) / 2.0;
    assert!(all.iter().all(|p| (p.cai - edge).abs() > 1e-6));
    let got = run(
        &params,
        &usage,
        &case,
        (edge, edge, &[]),
        (30.0, &[]),
        u64::MAX,
        Verify::folds(1),
        DEDUP,
    );
    let s = &got.stats;
    assert_eq!(got.reached, 0, "{got:?}");
    assert!(s.bound_cut > 0, "{s:?}");
    assert_eq!(s.bound_outside_cut, s.bound_cut, "{s:?}");
    // What passes the first test is a derivation with completions on both
    // sides of the window: it reaches the covered test, which has no span to
    // cut on, and the energy bound is off.
    assert_eq!(
        s.bound_covered_tested,
        s.bound_outside_tested - s.bound_outside_cut,
        "{s:?}"
    );
    assert_eq!(
        (
            s.bound_covered_cut,
            s.bound_energy_tested,
            s.bound_energy_cut,
            s.bound_nodes_closed
        ),
        (0, 0, 0, 0),
        "{s:?}"
    );
}

#[test]
fn the_window_front_is_the_enumerated_front() {
    let three_class = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
    let cases = [
        Case {
            protein: "MLRS",
            utr5: "AAAAAA",
            utr3: "AAAAAA",
            constraints: RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        },
        Case {
            protein: "MLRS",
            utr5: "GGGAUCCG",
            utr3: "AAAAAAAA",
            constraints: RegionConstraints::five_prime_isolation(Enforcement::Forbid)
                .with(PairClass::Utr5Utr3, Enforcement::Forbid),
        },
        Case {
            protein: "MLRS",
            utr5: "GGGAUC",
            utr3: "GAUCCC",
            constraints: three_class,
        },
        Case {
            protein: "GLRSA",
            utr5: "GGCGC",
            utr3: "GCGCC",
            constraints: three_class,
        },
    ];
    let mut cov = Coverage::default();
    for case in &cases {
        assert_windows(case, &mut cov);
    }
    eprintln!("{cov:?}");
    assert!(
        cov.off_constrained > 0,
        "no front point off the constrained front: {cov:?}"
    );
    assert!(cov.empty_windows > 0, "every window held a point: {cov:?}");
    assert!(cov.cut > 0, "the bound cut nothing: {cov:?}");
    assert!(cov.cut_outside > 0, "nothing was cut as outside: {cov:?}");
    assert!(cov.cut_covered > 0, "nothing was cut as covered: {cov:?}");
    assert!(cov.energy_cut > 0, "the energy bound cut nothing: {cov:?}");
    assert!(cov.closed > 0, "the energy bound closed no node: {cov:?}");
    assert!(
        cov.skipped > 0,
        "the energy bound kept no instance from being lifted: {cov:?}"
    );
    assert!(cov.budget_stops > 0, "no walk was cut short: {cov:?}");
    assert!(cov.capped > 0, "no walk ended at its cap: {cov:?}");
    assert!(cov.resumed > 0, "no walk was resumed: {cov:?}");
    assert!(
        cov.window_copies > 0,
        "no windowed fold copied a cell from the four-entry windowed cache, so the fronts \
         it returned say nothing about it: {cov:?}"
    );
    // Designs this small rarely wait for a windowed slot: each windowed fold
    // is too quick for two to overlap. `windowed_slots_change_no_front` is
    // the case built so that they do.
    eprintln!(
        "designs that waited for a windowed slot: {}",
        cov.window_deferred
    );
}

/// A window whose time is spent before its setup ends stops there: no
/// outside pass starts, the walk takes no derivation, and the report says
/// the setup stopped and why, with the whole window left open. Before, the
/// setup ran every pass whatever the budget said and only the walk read it.
#[test]
fn a_spent_budget_stops_the_window_in_its_setup() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "MLRS",
        utr5: "GGGAUCCG",
        utr3: "AAAAAAAA",
        constraints: RegionConstraints::five_prime_isolation(Enforcement::Forbid),
    };
    let got = window_front(
        &params,
        case.protein,
        &usage,
        &full_sets(case.protein),
        case.utr5,
        case.utr3,
        case.constraints,
        ScanConfig::default(),
        (f64::NEG_INFINITY, f64::INFINITY),
        30.0,
        &[],
        &ENERGY,
        u64::MAX,
        0.0,
        wide(),
        DEDUP,
        None,
    )
    .expect("the walk runs")
    .expect("a lattice");
    assert!(got.energy_slopes.is_empty(), "a pass ran: {got:?}");
    assert_eq!(
        (got.stats.derivations, got.reached, got.found),
        (0, 0, 0),
        "{got:?}"
    );
    assert!(!got.complete && got.last_g.is_none(), "{got:?}");
    assert_eq!(got.open, [(got.cai_min, got.cai_max)], "{got:?}");
    assert_eq!(got.stats.stopped, "seconds", "{got:?}");
    assert!(got.stopped_in_setup, "{got:?}");
}

/// Walks of one molecule at one slope repeat the energy bound's weights bit
/// for bit: a cache held by the caller computes each pass once, not once per
/// walk, and the front each walk returns is the front a walk taking every
/// pass afresh returns. The pass at the walk's own weight is the exception: it
/// keeps its tables for the walk, so it is taken afresh each time.
#[test]
fn a_pass_cache_computes_each_weight_once_and_changes_no_front() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "GLRSA",
        utr5: "GGCGC",
        utr3: "GCGCC",
        constraints: RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
    };
    let all = enumerate(&params, &usage, &case);
    let mut cais: Vec<f64> = all.iter().map(|p| p.cai).collect();
    cais.sort_by(f64::total_cmp);
    let q = |f: f64| cais[((cais.len() - 1) as f64 * f) as usize];
    let cache = PassCache::default();
    let walk = |lo: f64, hi: f64, passes: Option<&PassCache>| {
        window_front_sharing(
            &params,
            case.protein,
            &usage,
            &full_sets(case.protein),
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig::default(),
            (lo, hi),
            30.0,
            &[],
            &ENERGY,
            u64::MAX,
            f64::INFINITY,
            wide(),
            DEDUP,
            (None, 0),
            passes,
            None,
        )
        .expect("the walk runs")
        .expect("a lattice")
    };
    let key = |r: &WindowReport| -> Vec<(i32, String)> {
        r.front
            .iter()
            .map(|p| (p.energy, p.sequence.clone()))
            .collect()
    };
    let mut seen = 0;
    for (n, (lo, hi)) in [(q(0.0), q(1.0)), (q(0.5), q(1.0)), (q(0.25), q(0.75))]
        .into_iter()
        .enumerate()
    {
        let fresh = walk(lo, hi, None);
        let cached = walk(lo, hi, Some(&cache));
        assert_eq!(key(&fresh), key(&cached), "window [{lo}, {hi}]");
        assert_eq!(fresh.energy_cut, cached.energy_cut, "window [{lo}, {hi}]");
        assert_eq!(fresh.complete, cached.complete);
        let now = cache.computed();
        if n == 0 {
            // Every weight, and the walk's own slope when no weight is it.
            assert!(now >= ENERGY.len() as u64, "{now} passes computed");
        } else {
            assert_eq!(now - seen, 1, "walk {n} computed {} passes", now - seen);
        }
        seen = now;
    }
}

/// Walks of one molecule at one slope share one production scan: a cache held
/// by the caller runs the scan once however many windows are walked, a walk at
/// another slope scans again, and the fronts are those of walks sharing
/// nothing. The count is of scans run, so a cache that was ignored (every walk
/// scanning for itself) fails it where an output comparison would not; the
/// last leg shows the count moves when a scan is run.
#[test]
fn a_pass_cache_runs_the_production_scan_once_per_slope() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "GLRSA",
        utr5: "GGCGC",
        utr3: "GCGCC",
        constraints: RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
    };
    let all = enumerate(&params, &usage, &case);
    let mut cais: Vec<f64> = all.iter().map(|p| p.cai).collect();
    cais.sort_by(f64::total_cmp);
    let q = |f: f64| cais[((cais.len() - 1) as f64 * f) as usize];
    let windows = [(q(0.0), q(1.0)), (q(0.5), q(1.0)), (q(0.25), q(0.75))];
    let walk = |(lo, hi): (f64, f64), mu: f64, energy: &[f64], passes: Option<&PassCache>| {
        window_front_sharing(
            &params,
            case.protein,
            &usage,
            &full_sets(case.protein),
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig::default(),
            (lo, hi),
            mu,
            &[],
            energy,
            u64::MAX,
            f64::INFINITY,
            wide(),
            DEDUP,
            (None, 0),
            passes,
            None,
        )
        .expect("the walk runs")
        .expect("a lattice")
    };
    let key = |r: &WindowReport| -> Vec<(i32, String)> {
        r.front
            .iter()
            .map(|p| (p.energy, p.sequence.clone()))
            .collect()
    };
    // With the energy bound's passes (the scan is the pass at the walk's own
    // weight) and without (the walk's scan is taken ahead of the walk).
    for energy in [&ENERGY[..], &[]] {
        let cache = PassCache::default();
        for (n, &w) in windows.iter().enumerate() {
            let fresh = walk(w, 30.0, energy, None);
            let shared = walk(w, 30.0, energy, Some(&cache));
            assert_eq!(key(&fresh), key(&shared), "window {w:?}");
            assert_eq!(fresh.complete, shared.complete);
            assert_eq!(cache.scans(), 1, "walk {n} scanned again");
        }
        // Another slope is another weight: a scan, and the first one is gone.
        walk(windows[0], 31.0, energy, Some(&cache));
        assert_eq!(cache.scans(), 2, "a new slope scans");
        walk(windows[0], 31.0, energy, Some(&cache));
        assert_eq!(cache.scans(), 2);
    }
}
