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
    adaptation, window_front, window_front_sharing, WindowReport,
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
    for (lo, hi) in windows {
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
        let configs: [(f64, &[f64], Verify, Walk); 10] = [
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
