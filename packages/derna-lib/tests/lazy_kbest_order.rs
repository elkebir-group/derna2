//! The lazy k-best walk against exhaustive enumeration.
//!
//! The walk's value rests on one claim: its designs come out in non-decreasing
//! constrained objective, each exactly once, so the first that passes a
//! predicate is optimal among those that pass. That is checked here over the
//! first designs of each walk against every encoding of proteins small enough to
//! enumerate, each encoding folded on its own under the same region constraint. A
//! whole walk is out of reach even here: the last design appears only after
//! every suboptimal structure of every better one.
//! The cases put designed codons inside hairpins, internal loops and a
//! multi-loop's unpaired flank, so the lifted fills carry part of the answer.

use std::collections::HashSet;

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::outside::lazy::{
    first_modular, walk_designs, Build, CutSchedule, DedupDrops, ExactFold, Verify, Walk, WalkStats,
};
use derna_lib::{
    design_restricted_with_region, sequence_constrained_mfe, sequence_gap, CdsWindow, CodonUsage,
    Design, Enforcement, PairClass, Params, RegionConstraints, ScanConfig,
};

const TOL: f64 = 1e-6;

fn brd(lambda: f64, d: &Design) -> f64 {
    lambda * (d.mfe_kcal_mol * 100.0) - (1.0 - lambda) * 100.0 * d.cai_log
}

fn full_sets(protein: &str) -> Vec<Vec<[u8; 3]>> {
    protein
        .chars()
        .map(|c| {
            let a = aa_to_index(c).expect("a standard residue");
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

/// Every encoding as one codon per position.
fn encodings(sets: &[Vec<[u8; 3]>]) -> Vec<Vec<[u8; 3]>> {
    let mut out = vec![Vec::new()];
    for set in sets {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                set.iter().map(move |c| {
                    let mut e = prefix.clone();
                    e.push(*c);
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
    lambda: f64,
    constraints: RegionConstraints,
}

/// `(objective, sequence)` for every encoding, each folded alone, ascending.
fn brute_force(params: &Params, usage: &CodonUsage, case: &Case<'_>) -> Vec<(f64, String)> {
    let mut out: Vec<(f64, String)> = encodings(&full_sets(case.protein))
        .into_iter()
        .filter_map(|enc| {
            let pinned: Vec<Vec<[u8; 3]>> = enc.iter().map(|c| vec![*c]).collect();
            design_restricted_with_region(
                params,
                case.protein,
                usage,
                case.lambda,
                &pinned,
                case.utr5,
                case.utr3,
                case.constraints,
                ScanConfig::default(),
            )
            .expect("a valid instance")
            .map(|d| (brd(case.lambda, &d), d.sequence))
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// How many designs of each walk are checked.
const PREFIX: usize = 40;

/// A walk's first `k` designs as `(objective, sequence, constrained MFE)`.
type Visited = Vec<(f64, String, i32)>;

/// Production-valued, with every cell keeping one derivation per subsequence.
/// Under debug assertions every incremental spelling is also checked against
/// a full one.
const DEDUP: Walk = Walk {
    build: Build::Production,
    node_dedup: true,
    incremental_spell: true,
    pipelined_visit: false,
    spellers: 0,
    dedup_index: false,
    tight_lists: false,
};

/// [`DEDUP`], with each design read off and visited on a thread of its own.
const PIPED: Walk = Walk {
    pipelined_visit: true,
    ..DEDUP
};

/// [`PIPED`], with designs spelled on three threads of their own as well,
/// from the walk's shelf. Under debug assertions every shelf spelling is
/// checked against the forest's.
const SPELLED: Walk = Walk {
    spellers: 3,
    ..PIPED
};

/// [`SPELLED`] with one speller.
const SPELLED_ONE: Walk = Walk {
    spellers: 1,
    ..PIPED
};

fn walk_prefix(
    params: &Params,
    usage: &CodonUsage,
    case: &Case<'_>,
    k: usize,
    walk: impl Into<Walk>,
) -> (Visited, WalkStats) {
    let mut got = Vec::new();
    let stats = walk_designs(
        params,
        case.protein,
        usage,
        case.lambda,
        &full_sets(case.protein),
        case.utr5,
        case.utr3,
        case.constraints,
        ScanConfig::default(),
        u64::MAX,
        f64::INFINITY,
        walk,
        |obj, seq, energy, _| {
            // Each design's energy, read off its first derivation, is its
            // constrained MFE.
            let cds = CdsWindow {
                start: case.utr5.len(),
                end: case.utr5.len() + 3 * case.protein.len(),
            };
            let want = sequence_constrained_mfe(params, seq, cds, case.constraints).expect("folds");
            let got_kcal = f64::from(energy.expect("an energy at λ > 0")) / 100.0;
            assert!(
                (got_kcal - want).abs() < TOL,
                "{seq}: the walk's constrained MFE {got_kcal} against the fold's {want}"
            );
            got.push((obj, seq.to_string(), energy.expect("an energy at λ > 0")));
            Ok(got.len() >= k)
        },
    )
    .expect("the walk runs")
    .expect("the lattice admits a design");
    assert!(matches!(stats.stopped, "visitor" | "exhausted"));
    assert_eq!(
        stats.duplicate_molecules,
        cfg!(debug_assertions).then_some(0),
        "a (sequence, structure) pair came out twice"
    );
    (got, stats)
}

/// Two walks visit the same designs in the same order, each at the same
/// objective and constrained MFE.
fn assert_same_walk(label: &str, a: &Visited, b: &Visited) {
    assert_eq!(a.len(), b.len(), "[{label}] the same number of designs");
    for (rank, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.1, y.1, "[{label}] rank {rank}: a different design");
        assert_eq!(
            x.2, y.2,
            "[{label}] rank {rank}: a different constrained MFE"
        );
        assert!(
            (x.0 - y.0).abs() < TOL,
            "[{label}] rank {rank}: objective {} against {}",
            x.0,
            y.0
        );
    }
}

/// Drops below the root: repeats a node removed before they reached a design.
fn below_root(d: &DedupDrops) -> u64 {
    d.total() - d.root
}

/// Checks the walk against brute force, with and without node dedup, and
/// returns the dedup walk's drops.
fn assert_matches(case: &Case<'_>) -> DedupDrops {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let want = brute_force(&params, &usage, case);
    let k = PREFIX.min(want.len());
    let label = format!(
        "{} λ={} ({}, {})",
        case.protein, case.lambda, case.utr5, case.utr3
    );
    let (got, _) = walk_prefix(&params, &usage, case, k, Build::OnDemand);
    // Inside values read from the production tables walk the same designs in
    // the same order, spelled the same way.
    let (production, plain) = walk_prefix(&params, &usage, case, k, Build::Production);
    assert_same_walk(&format!("{label} production"), &got, &production);
    // Keeping one derivation per subsequence at every node changes nothing a
    // visitor sees.
    let (dedup, stats) = walk_prefix(&params, &usage, case, k, DEDUP);
    assert_same_walk(&format!("{label} node dedup"), &production, &dedup);
    // Visited on its own thread, the walk shows the visitor the same designs
    // and reports itself as it stood at the design the visitor stopped on,
    // though it may have spelled designs past it.
    // Spelled on its own thread too, the same again.
    for (name, walk) in [
        ("piped", PIPED),
        ("spelled by one", SPELLED_ONE),
        ("spelled by three", SPELLED),
    ] {
        let (piped, piped_stats) = walk_prefix(&params, &usage, case, k, walk);
        assert_same_walk(&format!("{label} {name}"), &dedup, &piped);
        assert_eq!(
            (
                piped_stats.designs,
                piped_stats.derivations,
                piped_stats.stopped
            ),
            (stats.designs, stats.derivations, stats.stopped),
            "[{label}] a {name} walk reports where its visitor stopped"
        );
        assert!(
            (piped_stats.last_value - stats.last_value).abs() < TOL,
            "[{label}] a {name} walk's last value"
        );
    }
    let drops = stats.node_dedup.expect("a dedup walk reports its drops");
    assert_eq!(plain.node_dedup, None);
    assert_eq!(
        stats.derivations, stats.designs,
        "[{label}] every root derivation of a dedup walk is a new design"
    );
    assert!(
        stats.derivations + drops.root <= plain.derivations,
        "[{label}] the dedup walk takes no more root derivations"
    );
    assert_eq!(got.len(), k, "[{label}] the walk reaches {k} designs");
    for (rank, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g.0 - w.0).abs() < TOL,
            "[{label}] rank {rank}: walk {} against brute force {}",
            g.0,
            w.0
        );
    }
    let a: HashSet<&String> = got.iter().map(|x| &x.1).collect();
    let drops_out = drops;
    assert_eq!(a.len(), k, "[{label}] no design twice");
    // Past ties at the cut, the prefix holds exactly the encodings below it.
    let cut = got[k - 1].0;
    for (obj, seq) in want.iter().filter(|w| w.0 < cut - TOL) {
        assert!(
            a.contains(seq),
            "[{label}] {seq} at {obj} is missing below the cut {cut}"
        );
    }
    drops_out
}

#[test]
fn designs_come_out_in_order_under_a_constraint() {
    for lambda in [0.3, 0.7] {
        let drops = assert_matches(&Case {
            protein: "MLRS",
            utr5: "GGGAUC",
            utr3: "GAUCCC",
            lambda,
            constraints: RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        });
        // Without drops below the root the dedup arm compares nothing.
        assert!(
            below_root(&drops) > 0,
            "λ={lambda}: no node dropped a repeat"
        );
    }
}

/// Six- and four-codon residues inside a molecule long enough to close loops
/// around them, so designed codons land in loop interiors the grammar fills.
#[test]
fn designs_come_out_in_order_with_codons_in_loop_interiors() {
    let drops = assert_matches(&Case {
        protein: "GLRSA",
        utr5: "GGCGC",
        utr3: "GCGCC",
        lambda: 0.5,
        constraints: RegionConstraints::default(),
    });
    assert!(below_root(&drops) > 0, "no node dropped a repeat");
}

/// The coding sequence is a multi-loop's unpaired flank.
#[test]
fn designs_come_out_in_order_inside_a_multiloop() {
    for (protein, utr5, utr3) in [
        ("KK", "GAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUC"),
        ("MA", "GGAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUCC"),
    ] {
        let drops = assert_matches(&Case {
            protein,
            utr5,
            utr3,
            lambda: 0.5,
            constraints: RegionConstraints::default(),
        });
        assert!(
            drops.multi + drops.m + drops.m2 > 0,
            "{protein}: no multiloop node dropped a repeat"
        );
    }
}

/// Walked to exhaustion with node dedup, every encoding comes out exactly
/// once, in brute-force order: nothing is lost with the dropped repeats.
#[test]
fn a_dedup_walk_exhausts_every_design_in_order() {
    dedup_walk_exhausts_every_design_in_order(DEDUP);
}

/// The same under both memory levers: dedup sets of indices into each
/// node's derivations, and lists grown by a quarter. The case drops repeats
/// below the root, so the index sets are exercised.
#[test]
fn the_memory_levers_keep_every_design_in_order() {
    dedup_walk_exhausts_every_design_in_order(Walk {
        dedup_index: true,
        tight_lists: true,
        ..DEDUP
    });
}

fn dedup_walk_exhausts_every_design_in_order(walk: Walk) {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "MLR",
        utr5: "GGGAUC",
        utr3: "GAUCCC",
        lambda: 0.5,
        constraints: RegionConstraints::symmetric_isolation(Enforcement::Forbid),
    };
    let want = brute_force(&params, &usage, &case);
    let (got, stats) = walk_prefix(&params, &usage, &case, usize::MAX, walk);
    assert_eq!(stats.stopped, "exhausted");
    assert_eq!(got.len(), want.len(), "every encoding, once");
    let seqs: HashSet<&String> = got.iter().map(|x| &x.1).collect();
    assert_eq!(seqs.len(), want.len(), "no design twice");
    for (rank, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g.0 - w.0).abs() < TOL,
            "rank {rank}: walk {} against brute force {}",
            g.0,
            w.0
        );
    }
    assert!(below_root(&stats.node_dedup.expect("drops")) > 0);
    // Each design was also spelled in full and compared under debug
    // assertions; this shows the copies happened.
    assert!(stats.positions_copied > 0, "no position was copied");
}

/// A walk out of time before its first request gives up at whichever node it
/// is in, and what it visited before stopping is a prefix of the full walk:
/// a node left short never answers a later request out of order.
#[test]
fn a_walk_out_of_time_stops_without_a_design_out_of_order() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "MLR",
        utr5: "GGGAUC",
        utr3: "GAUCCC",
        lambda: 0.5,
        constraints: RegionConstraints::symmetric_isolation(Enforcement::Forbid),
    };
    let (full, _) = walk_prefix(&params, &usage, &case, usize::MAX, DEDUP);
    let mut got: Vec<String> = Vec::new();
    let stats = walk_designs(
        &params,
        case.protein,
        &usage,
        case.lambda,
        &full_sets(case.protein),
        case.utr5,
        case.utr3,
        case.constraints,
        ScanConfig::default(),
        u64::MAX,
        0.0,
        DEDUP,
        |_, seq, _, _| {
            got.push(seq.to_string());
            Ok(false)
        },
    )
    .expect("the walk runs")
    .expect("the lattice admits a design");
    assert_eq!(stats.stopped, "seconds");
    assert!(got.len() < full.len(), "a walk with no time left exhausted");
    for (rank, seq) in got.iter().enumerate() {
        assert_eq!(seq, &full[rank].1, "rank {rank} out of order");
    }
}

/// The deadline can land anywhere in a dedup walk, including while a
/// candidate's tails are still being materialized. Across a sweep of short
/// deadlines the walk must stop cleanly ("seconds", or "exhausted" when it
/// finishes first) with its designs still in order. Timing decides where
/// each deadline lands, so this constraints the class of interruption points
/// rather than one of them.
#[test]
fn a_walk_stops_cleanly_wherever_the_deadline_lands() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let case = Case {
        protein: "MLRS",
        utr5: "GGGAUC",
        utr3: "GAUCCC",
        lambda: 0.5,
        constraints: RegionConstraints::symmetric_isolation(Enforcement::Forbid),
    };
    let (full, _) = walk_prefix(&params, &usage, &case, 2_000, DEDUP);
    for k in 0..40 {
        let seconds = 1e-5 * 1.3_f64.powi(k);
        let mut got: Vec<String> = Vec::new();
        let stats = walk_designs(
            &params,
            case.protein,
            &usage,
            case.lambda,
            &full_sets(case.protein),
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig::default(),
            u64::MAX,
            seconds,
            DEDUP,
            |_, seq, _, _| {
                got.push(seq.to_string());
                Ok(got.len() >= full.len())
            },
        )
        .expect("the walk runs")
        .expect("the lattice admits a design");
        assert!(
            ["seconds", "visitor", "exhausted"].contains(&stats.stopped),
            "{seconds} s: stopped {}",
            stats.stopped
        );
        for (rank, seq) in got.iter().enumerate() {
            assert_eq!(seq, &full[rank].1, "{seconds} s: rank {rank} out of order");
        }
    }
}

/// The first modular design of a walk is the least objective among modular
/// encodings, and a walk cut short before it reports a lower bound on it.
/// Returns how many designs `verify`'s witness pool rejected without a fold,
/// how many waited for a fold slot, and of the folds, how many were made
/// uncut before the cut's lattice landed, out of all folds, and how many a
/// windowed fold rejected.
fn assert_first_modular(case: &Case<'_>, verify: Verify, walk: Walk) -> (u64, u64, u64, u64, u64) {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let cds = CdsWindow {
        start: case.utr5.len(),
        end: case.utr5.len() + 3 * case.protein.len(),
    };
    let all = brute_force(&params, &usage, case);
    let modular: Vec<&(f64, String)> = all
        .iter()
        .filter(|(_, s)| {
            sequence_gap(&params, s, cds, case.constraints)
                .expect("folds")
                .is_modular()
        })
        .collect();
    let want = modular.first().expect("the case has a modular design").0;
    assert!(
        all.first().map(|x| &x.1) != modular.first().map(|x| &x.1),
        "the constraint's optimum is already modular here, so the test does not exercise the walk"
    );
    let walk_with = |max_derivations: u64, verify: Verify| {
        first_modular(
            &params,
            case.protein,
            &usage,
            case.lambda,
            &full_sets(case.protein),
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig::default(),
            max_derivations,
            f64::INFINITY,
            verify,
            walk,
        )
        .expect("the walk runs")
        .expect("a design exists")
    };
    let walk = |max_derivations: u64| walk_with(max_derivations, verify);
    let report = walk(u64::MAX);
    assert_eq!(
        report.witness_rejects + report.folds + report.skipped,
        report.stats.designs,
        "every design is rejected by a witness, folded, or queued past the answer"
    );
    let (got, _, rank) = report
        .modular
        .clone()
        .expect("the walk finds a modular design");
    assert!(
        (want - got).abs() < TOL,
        "walk {got} against brute-force modular optimum {want}"
    );
    // Cut short before the answer, the walk's last value bounds it from below.
    let short = walk(rank - 1);
    assert!(
        short.modular.is_none(),
        "the short walk stops before the answer"
    );
    assert!(
        short.stats.last_value <= want + TOL,
        "bound {} above the modular optimum {want}",
        short.stats.last_value
    );
    // An incumbent at or above the answer leaves it unchanged.
    for incumbent in modular.iter().map(|h| h.0).take(3) {
        let bounded = walk_with(
            u64::MAX,
            Verify {
                incumbent: Some(incumbent),
                ..verify
            },
        );
        let (got, _, _) = bounded.modular.expect("the incumbent's walk answers");
        assert!(
            (want - got).abs() < TOL,
            "incumbent {incumbent} moved the answer to {got}"
        );
    }
    // One below the answer is not a modular design's objective:
    // the walk stops past it, unanswered, with every design it met checked.
    let false_incumbent = walk_with(
        u64::MAX,
        Verify {
            incumbent: Some(want - 1.0),
            ..verify
        },
    );
    assert_eq!(false_incumbent.stats.stopped, "incumbent");
    assert!(false_incumbent.modular.is_none());
    assert!(false_incumbent.stats.last_value > want - 1.0);
    (
        report.witness_rejects,
        report.deferred,
        report.uncut_folds,
        report.folds,
        report.window_rejects,
    )
}

/// The constraint's own optimum is not modular in any case; the answer is the
/// sixth design in the first and the second in the second. A witness pool
/// changes how designs are rejected and not the answer: every witness
/// rejection is checked against the exact free fold under debug assertions,
/// and the pool must reject at least one design without a fold, or the
/// comparison says nothing.
#[test]
fn the_first_modular_design_is_the_modular_optimum() {
    let cases = [
        Case {
            protein: "MLRS",
            utr5: "AAAAAA",
            utr3: "AAAAAA",
            lambda: 0.5,
            constraints: RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        },
        Case {
            protein: "MLRS",
            utr5: "GGGAUCCG",
            utr3: "AAAAAAAA",
            lambda: 0.5,
            constraints: RegionConstraints::five_prime_isolation(Enforcement::Forbid)
                .with(PairClass::Utr5Utr3, Enforcement::Forbid),
        },
    ];
    let mut witness_rejects = 0;
    let mut deferred = 0;
    let mut window_rejects = 0;
    let plain = Walk::from(Build::Production);
    for case in &cases {
        assert_eq!(assert_first_modular(case, Verify::folds(1), plain).0, 0);
        assert_eq!(assert_first_modular(case, Verify::folds(1), DEDUP).0, 0);
        // One fold slot among four workers: designs no witness rejects wait
        // for a finished fold's structure, then the slot. Under debug
        // assertions every rejection by that structure is checked against the
        // exact free MFE. Designs reach the workers eight at a time.
        let (rejects, waited, _, _, _) = assert_first_modular(
            case,
            Verify {
                threads: 4,
                witnesses: 16,
                witness_stop: 1,
                witness_promote: true,
                fold_slots: 1,
                handoff_batch: 8,
                fold: ExactFold::SparseCut,
                ..Verify::folds(4)
            },
            DEDUP,
        );
        witness_rejects += rejects;
        deferred += waited;
        // Folds copying spans from recent folds: under debug assertions every
        // verdict is checked against the exact free MFE.
        witness_rejects += assert_first_modular(
            case,
            Verify {
                threads: 4,
                witnesses: 16,
                witness_stop: 0,
                witness_promote: true,
                fold_reuse: 4,
                fold_reuse_rows: true,
                fold_slots: 0,
                handoff_batch: 1,
                fold: ExactFold::SparseCut,
                cut: CutSchedule::Before,
                window: None,
                window_utr: false,
                window_reuse: false,
                window_auto: false,
                incumbent: None,
            },
            DEDUP,
        )
        .0;
        // The cut's lattice built beside the walk, and never handed over:
        // folds before it lands are the exact uncut Zuker fold, and under
        // debug assertions every verdict is checked against the exact free
        // MFE. Withheld, every fold must be uncut, and there must be folds.
        for cut in [CutSchedule::Background, CutSchedule::Withheld] {
            let (rejects, _, uncut, folds, _) = assert_first_modular(
                case,
                Verify {
                    threads: 4,
                    witnesses: 16,
                    witness_stop: 1,
                    witness_promote: true,
                    fold_reuse: 4,
                    fold_reuse_rows: true,
                    fold_slots: 1,
                    handoff_batch: 8,
                    fold: ExactFold::SparseCut,
                    cut,
                    ..Verify::folds(4)
                },
                DEDUP,
            );
            witness_rejects += rejects;
            assert!(uncut <= folds, "{cut:?}: more uncut folds than folds");
            if cut == CutSchedule::Withheld {
                assert!(folds > 0, "withheld: no design reached a fold");
                assert_eq!(uncut, folds, "withheld: a fold used the cut");
            }
        }
        // A windowed fold before every full fold: a rejection it makes is
        // checked against the exact free MFE under debug assertions. The
        // windows run from none of the CDS to all of it.
        // The last arm builds the cut beside the walk, as the probe's
        // defaults do.
        for (window, window_utr) in [
            (Some((1, 1)), false),
            (Some((3, 0)), false),
            (Some((12, 12)), false),
            (Some((0, 0)), true),
        ] {
            let (rejects, _, _, _, windowed) = assert_first_modular(
                case,
                Verify {
                    threads: 4,
                    witnesses: 4,
                    witness_stop: 1,
                    fold_slots: 1,
                    fold: ExactFold::SparseCut,
                    window,
                    window_utr,
                    cut: if window == Some((12, 12)) {
                        CutSchedule::Background
                    } else {
                        CutSchedule::Before
                    },
                    ..Verify::folds(4)
                },
                DEDUP,
            );
            witness_rejects += rejects;
            window_rejects += windowed;
        }
        // Windowed folds copying from earlier windowed folds, the cut in
        // hand from the start: under debug assertions every windowed
        // rejection is re-scored exactly and must lie below the design's
        // constrained MFE.
        let (rejects, _, _, _, windowed) = assert_first_modular(
            case,
            Verify {
                threads: 4,
                witnesses: 4,
                witness_stop: 1,
                fold_reuse: 4,
                fold_reuse_rows: true,
                fold_slots: 1,
                fold: ExactFold::SparseCut,
                cut: CutSchedule::Before,
                window: Some((3, 1)),
                window_reuse: true,
                ..Verify::folds(4)
            },
            DEDUP,
        );
        witness_rejects += rejects;
        window_rejects += windowed;
        // The window's levers switched on by the check itself, from a window
        // over the flanks alone: under debug assertions every windowed
        // rejection, before and after a switch, is checked against the exact
        // free MFE (and, reusing, re-scored).
        let (rejects, _, _, _, windowed) = assert_first_modular(
            case,
            Verify {
                threads: 4,
                witnesses: 4,
                witness_stop: 1,
                fold_reuse: 4,
                fold_reuse_rows: true,
                fold_slots: 1,
                fold: ExactFold::SparseCut,
                cut: CutSchedule::Before,
                window: Some((0, 0)),
                window_auto: true,
                ..Verify::folds(4)
            },
            DEDUP,
        );
        witness_rejects += rejects;
        window_rejects += windowed;
        // The probe's defaults, with the walk handing its designs to the
        // check from a thread of its own, then spelling them there too.
        for walk in [PIPED, SPELLED] {
            let (rejects, _, _, _, windowed) = assert_first_modular(
                case,
                Verify {
                    threads: 4,
                    witnesses: 16,
                    witness_stop: 1,
                    witness_promote: true,
                    fold_reuse: 4,
                    fold_reuse_rows: true,
                    fold_slots: 1,
                    handoff_batch: 8,
                    fold: ExactFold::SparseCut,
                    cut: CutSchedule::Background,
                    window: Some((3, 1)),
                    ..Verify::folds(4)
                },
                walk,
            );
            witness_rejects += rejects;
            window_rejects += windowed;
        }
        // The cut with its inside bound: under debug assertions every pair's
        // bound is checked against the `V` the fold computes for it.
        assert_eq!(
            assert_first_modular(
                case,
                Verify {
                    fold: ExactFold::SparseCutInside,
                    ..Verify::folds(1)
                },
                DEDUP,
            )
            .0,
            0
        );
        for (i, (threads, witness_stop)) in [(1, 0), (4, 0), (1, 1), (4, 3)].into_iter().enumerate()
        {
            witness_rejects += assert_first_modular(
                case,
                Verify {
                    threads,
                    witnesses: 16,
                    witness_stop,
                    witness_promote: i == 3,
                    fold_reuse: if i % 2 == 0 { 0 } else { 4 },
                    fold_reuse_rows: false,
                    fold_slots: 0,
                    handoff_batch: [1, 3, 8, 16][i],
                    fold: [
                        ExactFold::Zuker,
                        ExactFold::LinearFold,
                        ExactFold::Sparse,
                        ExactFold::SparseCut,
                    ][(threads + usize::from(witness_stop > 0) * 2) % 4],
                    cut: CutSchedule::Before,
                    window: None,
                    window_utr: false,
                    window_reuse: false,
                    window_auto: false,
                    incumbent: None,
                },
                if i % 2 == 0 { plain } else { DEDUP },
            )
            .0;
        }
    }
    assert!(witness_rejects > 0, "no design was rejected by a witness");
    assert!(deferred > 0, "no design waited for a fold slot");
    assert!(window_rejects > 0, "no windowed fold rejected a design");
}

/// Forbidding every class with a 5′ endpoint leaves the 5′ UTR unpaired in
/// every design the walk visits, in the order brute force ranks them. The
/// unconstrained walk over the same case pairs a 5′ position, so the constraint has
/// something to forbid.
#[test]
fn a_five_prime_unpaired_walk_never_pairs_the_five_prime_utr() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let constrained = Case {
        protein: "MLRS",
        utr5: "GGGAUC",
        utr3: "GAUCCC",
        lambda: 0.5,
        constraints: RegionConstraints::five_prime_unpaired(Enforcement::Forbid),
    };
    assert_matches(&constrained);
    let five_prime_pairs = |case: &Case<'_>| {
        let len5 = case.utr5.len() as u32;
        let mut touching = 0_usize;
        walk_designs(
            &params,
            case.protein,
            &usage,
            case.lambda,
            &full_sets(case.protein),
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig::default(),
            PREFIX as u64,
            f64::INFINITY,
            DEDUP,
            |_, _, _, pairs| {
                touching += pairs.iter().filter(|&&(i, j)| i.min(j) < len5).count();
                Ok(false)
            },
        )
        .expect("the walk runs")
        .expect("the lattice admits a design");
        touching
    };
    assert_eq!(
        five_prime_pairs(&constrained),
        0,
        "a design pairs the 5′ UTR"
    );
    let unconstrained = Case {
        constraints: RegionConstraints::default(),
        ..constrained
    };
    assert!(
        five_prime_pairs(&unconstrained) > 0,
        "the unconstrained walk never pairs the 5′ UTR, so the constraint forbids nothing"
    );
}
