//! The motif-constrained hull walk with a shared outside bound, against the
//! same walk with every probe a full exact fold.
//!
//! A hull is a claim about a lower envelope: for every λ in range, the best
//! any vertex scores is the constrained optimum at that λ. Two exact walks may
//! keep different collinear points on one edge without either being wrong, so
//! the comparison is of envelopes, not of vertex lists.
//!
//! **The differential half is blind to the walk.** Both arms run the same
//! walk; only the folds differ. So comparing them verifies the *bound* and
//! says nothing about whether the walk visits every gap — a walk that skipped
//! half its gaps would skip them in both arms and agree with itself. The walk
//! is checked against an independent oracle instead: at many λ, the hull's
//! envelope must equal the optimum `design_with_motifs` reaches on its own.
//!
//! Ways this could pass vacuously, and what rules each out:
//!
//! * **A hull of two points.** Then no gap probe ever finds a vertex and the
//!   split logic is never exercised. The sweep asserts some walk found more.
//! * **The bound prunes nothing.** Then the bounded walk is the reference.
//!   The sweep asserts pruned pairs.
//! * **No closed gap probed.** The claim that matters for speed is that a
//!   closed gap's ceiling sits exactly on its optimum. The sweep asserts that
//!   such probes occurred, and that every one of them certified in one fold
//!   under that ceiling (after any guesses below it).
//! * **The pair floor cut nothing.** Then the cut the bounded walk makes
//!   before scoring a closure went untested; the sweep asserts cuts.
//! * **No probe guessed.** A probe whose inherited ceiling is far from the
//!   optimum first searches up from the unconstrained optimum; if none did,
//!   the guesses' certification went unchecked by the envelope comparison.
//!   The sweep asserts that some probe guessed.

use derna_lib::bounded_motifs::{
    default_workers, walk_motif_hull, walk_motif_hull_aware, walk_motif_hull_within, MotifHullWalk,
};
use derna_lib::{design, design_with_motifs, CodonUsage, Params, ScanConfig};

const ENDS: (f64, f64) = (0.0001, 0.99);
const ANCHORS: &[f64] = &[0.0001, 0.25, 0.5, 0.75, 0.99];
const ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next(&mut self) -> usize {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x as usize
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.next() % (hi - lo + 1)
    }
}

fn random_protein(rng: &mut Rng, len: usize) -> String {
    (0..len)
        .map(|_| ALPHABET[rng.next() % ALPHABET.len()] as char)
        .collect()
}

fn cut(rng: &mut Rng, seq: &str, len: usize) -> String {
    let at = rng.range(0, seq.len() - len);
    seq[at..at + len].to_string()
}

fn envelope(walk: &MotifHullWalk, lambda: f64) -> f64 {
    walk.vertices
        .iter()
        .map(|v| lambda * f64::from(v.mfe) - (1.0 - lambda) * v.cai * 100.0)
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn bounded_walk_finds_the_reference_hull() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let config = ScanConfig::default();
    let mut rng = Rng::new(20_261_005);
    let (mut walks, mut most_vertices, mut pruned) = (0, 0, 0);
    let (mut closed, mut closed_on_the_answer, mut closed_one_fold) = (0, 0, 0);
    let mut guessed = 0;
    let mut cuts = 0;
    for _ in 0..8 {
        let len = rng.range(10, 14);
        let protein = random_protein(&mut rng, len);
        // Cut from the free optimum, so the constraint costs something.
        let free = design(&params, &protein, &usage, 0.5).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
        let bounded = walk_motif_hull(
            &params,
            &protein,
            &usage,
            &motifs,
            ENDS,
            Some(ANCHORS),
            config,
        )
        .unwrap();
        let reference =
            walk_motif_hull(&params, &protein, &usage, &motifs, ENDS, None, config).unwrap();
        let (Some(bounded), Some(reference)) = (bounded, reference) else {
            continue;
        };
        for k in 0..=400 {
            let lambda = ENDS.0 + (ENDS.1 - ENDS.0) * f64::from(k) / 400.0;
            let (b, r) = (envelope(&bounded, lambda), envelope(&reference, lambda));
            assert!(
                (b - r).abs() < 1e-6,
                "envelopes differ at λ={lambda}: bounded {b}, reference {r}, on {protein}"
            );
        }
        // Independent of the walk: the envelope is the optimum at every λ.
        // Dense at small λ, where the front is steep and gaps are many.
        for &lambda in &[
            0.001, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95,
        ] {
            let one = design_with_motifs(&params, &protein, &usage, lambda, &motifs, config)
                .expect("single solve");
            let alone =
                lambda * (one.mfe_kcal_mol * 100.0).round() - (1.0 - lambda) * one.cai_log * 100.0;
            let hull = envelope(&bounded, lambda);
            assert!(
                (hull - alone).abs() < 1e-6,
                "the hull misses the optimum at λ={lambda} on {protein}: \
                 envelope {hull}, single solve {alone}"
            );
        }
        walks += 1;
        most_vertices = most_vertices.max(bounded.vertices.len());
        for p in &bounded.probes {
            pruned += p.pruned;
            guessed += usize::from(p.guesses > 0);
            cuts += p.cut;
            if p.ceiling.is_finite() && !p.new_vertex {
                closed += 1;
                closed_on_the_answer += usize::from((p.ceiling - p.optimum).abs() < 1e-6);
                closed_one_fold += usize::from(p.folds - p.guesses <= 1);
            }
        }
    }
    eprintln!(
        "walks {walks} / most vertices {most_vertices} / pruned {pruned} / cuts {cuts} / guessed {guessed} / \
         closed gaps {closed} (ceiling on the answer {closed_on_the_answer}, one fold {closed_one_fold})"
    );
    assert!(walks > 0, "no case was feasible");
    assert!(
        most_vertices > 2,
        "every hull had two points, so no gap was ever split"
    );
    assert!(pruned > 0, "the bound never pruned a pair");
    assert!(closed > 0, "no closed gap was probed");
    assert!(
        guessed > 0,
        "no probe searched up from the unconstrained optimum"
    );
    assert!(cuts > 0, "the pair floor never cut a closure");
    // The property the walk's speed rests on: a closed gap's ceiling is the
    // score both ends share at its tie weight, which is then the optimum.
    assert_eq!(
        closed_on_the_answer, closed,
        "a closed gap's ceiling was not its optimum"
    );
    assert_eq!(
        closed_one_fold, closed,
        "a closed gap needed more than one fold under its ceiling"
    );
}

/// Threads change where the walk's work runs, never what it finds: the
/// anchors are computed side by side and each fold fans out, and both are
/// exact, so the hull, every probe's fold count and every pruned count must
/// match the serial walk's byte for byte — with threaded folds and workers,
/// with side-by-side anchors and gap probes alone, and at the default worker
/// count. A hull of more than two vertices had a round of two or more gaps,
/// probed side by side; some walk must have one.
#[test]
fn threaded_walk_is_the_serial_walk() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_017);
    let mut walks = 0;
    let mut split = false;
    for _ in 0..4 {
        let len = rng.range(10, 14);
        let protein = random_protein(&mut rng, len);
        let free = design(&params, &protein, &usage, 0.5).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
        let config = |threads: usize| ScanConfig {
            threads,
            ..ScanConfig::default()
        };
        let walk = |threads: usize, workers: usize| {
            walk_motif_hull_within(
                &params,
                &protein,
                &usage,
                &motifs,
                ENDS,
                Some(ANCHORS),
                config(threads),
                0.0,
                workers,
            )
            .unwrap()
        };
        let Some(serial) = walk(1, 1) else {
            continue;
        };
        let by_default = walk_motif_hull(
            &params,
            &protein,
            &usage,
            &motifs,
            ENDS,
            Some(ANCHORS),
            config(1),
        )
        .unwrap()
        .unwrap();
        let vertices = |w: &MotifHullWalk| -> Vec<(i32, u64, String, String)> {
            w.vertices
                .iter()
                .map(|v| {
                    (
                        v.mfe,
                        v.cai.to_bits(),
                        v.sequence.clone(),
                        v.structure.clone(),
                    )
                })
                .collect()
        };
        let probes = |w: &MotifHullWalk| -> Vec<(u64, usize, usize, usize, usize)> {
            w.probes
                .iter()
                .map(|p| (p.lambda.to_bits(), p.folds, p.guesses, p.pruned, p.cut))
                .collect()
        };
        for (threads, workers) in [(4, 4), (1, 4)] {
            let other = walk(threads, workers).unwrap();
            let at = format!("on {protein} with {threads} fold threads, {workers} anchor workers");
            assert_eq!(vertices(&serial), vertices(&other), "{at}");
            assert_eq!(probes(&serial), probes(&other), "{at}");
        }
        assert_eq!(
            vertices(&serial),
            vertices(&by_default),
            "on {protein} by default"
        );
        assert_eq!(
            probes(&serial),
            probes(&by_default),
            "on {protein} by default"
        );
        split |= serial.vertices.len() > 2;
        walks += 1;
    }
    assert!(walks > 0, "no case was feasible");
    assert!(split, "no walk had a round of two gaps");
}

/// The adapted end needs no anchor of its own: below the lowest anchor the
/// chords to the motif-free outside at λ = 0 bound the folds, so a walk whose
/// anchors stop short of the low end finds the same hull through the same
/// probes, and does not widen its anchors to reach it.
#[test]
fn the_adapted_end_needs_no_anchor() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_019);
    let mut walks = 0;
    for _ in 0..6 {
        let len = rng.range(10, 14);
        let protein = random_protein(&mut rng, len);
        // One motif from the adapted design, so the constraint binds at the
        // low end, and one from the middle of the front.
        let adapted = design(&params, &protein, &usage, 0.01).unwrap().sequence;
        let middle = design(&params, &protein, &usage, 0.5).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &adapted, 5), cut(&mut rng, &middle, 6)];
        let walk = |anchors: &[f64]| {
            walk_motif_hull_within(
                &params,
                &protein,
                &usage,
                &motifs,
                ENDS,
                Some(anchors),
                ScanConfig::default(),
                0.0,
                1,
            )
            .unwrap()
        };
        let (Some(full), Some(short)) =
            (walk(&[0.0001, 0.05, 0.3, 0.99]), walk(&[0.05, 0.3, 0.99]))
        else {
            continue;
        };
        let vertices = |w: &MotifHullWalk| -> Vec<(i32, u64, String, String)> {
            w.vertices
                .iter()
                .map(|v| {
                    (
                        v.mfe,
                        v.cai.to_bits(),
                        v.sequence.clone(),
                        v.structure.clone(),
                    )
                })
                .collect()
        };
        let probes = |w: &MotifHullWalk| -> Vec<(u64, u64)> {
            w.probes
                .iter()
                .map(|p| (p.lambda.to_bits(), p.optimum.to_bits()))
                .collect()
        };
        assert_eq!(short.anchor_lambdas, vec![0.05, 0.3, 0.99], "on {protein}");
        assert_eq!(vertices(&full), vertices(&short), "on {protein}");
        assert_eq!(probes(&full), probes(&short), "on {protein}");
        walks += 1;
    }
    assert!(walks > 0, "no case was feasible");
}

/// Motif-aware anchors only add floors, so the walk finds the same probes and
/// the same hull with them as without, at any worker count, whether its probes
/// wait for them or not. Waiting, what each probe prunes is the same at any
/// worker count too. Some probe must
/// have pruned on their chord, and that chord must have changed what some
/// walk pruned, or the motif-aware half went untested.
#[test]
fn motif_aware_anchors_keep_the_walk() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_020);
    let anchors = [0.01, 0.02, 0.04, 0.07, 0.3, 0.99];
    let (mut walks, mut aware_probes, mut moved) = (0, 0, 0);
    for _ in 0..6 {
        let len = rng.range(12, 16);
        let protein = random_protein(&mut rng, len);
        // Motifs from the adapted design, so the constraint costs adaptation
        // where the motif-aware anchors sit.
        let adapted = design(&params, &protein, &usage, 0.02).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &adapted, 5), cut(&mut rng, &adapted, 6)];
        let walk = |aware: &[f64], workers: usize, waits: bool| {
            walk_motif_hull_aware(
                &params,
                &protein,
                &usage,
                &motifs,
                ENDS,
                Some(&anchors),
                ScanConfig::default(),
                0.0,
                workers,
                aware,
                waits,
            )
            .unwrap()
        };
        let (Some(plain), Some(aware), Some(aware3), Some(free3)) = (
            walk(&[], 1, true),
            walk(&[0.01, 0.02, 0.04], 1, true),
            walk(&[0.01, 0.02, 0.04], 3, true),
            walk(&[0.01, 0.02, 0.04], 3, false),
        ) else {
            continue;
        };
        let probes = |w: &MotifHullWalk| -> Vec<(u64, u64)> {
            w.probes
                .iter()
                .map(|p| (p.lambda.to_bits(), p.optimum.to_bits()))
                .collect()
        };
        assert_eq!(probes(&plain), probes(&aware), "on {protein}");
        // Probes that do not wait prune on whatever chord is in, and find the same.
        assert_eq!(probes(&plain), probes(&free3), "on {protein}");
        for k in 0..=400 {
            let lambda = ENDS.0 + (ENDS.1 - ENDS.0) * f64::from(k) / 400.0;
            let (a, p) = (envelope(&aware, lambda), envelope(&plain, lambda));
            assert!(
                (a - p).abs() < 1e-6,
                "envelopes differ at λ={lambda}: motif-aware {a}, plain {p}, on {protein}"
            );
        }
        let pruned = |w: &MotifHullWalk| -> Vec<(usize, bool)> {
            w.probes.iter().map(|p| (p.pruned, p.motif_aware)).collect()
        };
        assert_eq!(pruned(&aware), pruned(&aware3), "on {protein}");
        aware_probes += aware.probes.iter().filter(|p| p.motif_aware).count();
        moved += usize::from(pruned(&aware) != pruned(&plain));
        walks += 1;
    }
    eprintln!(
        "walks {walks}, motif-aware probes {aware_probes}, walks whose pruning moved {moved}"
    );
    assert!(walks > 0, "no case was feasible");
    assert!(aware_probes > 0, "no probe pruned on a motif-aware chord");
    assert!(moved > 0, "the motif-aware chord changed no walk's pruning");
}

/// The default runs the anchors in two rounds at most, never more workers
/// than cores, and always at least one worker.
#[test]
fn default_workers_takes_one_per_anchor_up_to_the_cores() {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let most = default_workers(usize::MAX);
    assert!((1..=cores).contains(&most), "{most} of {cores} cores");
    assert_eq!(default_workers(0), 1);
    assert_eq!(default_workers(1), 1);
    for anchors in 2..=24 {
        assert_eq!(
            default_workers(anchors),
            anchors.min(most),
            "{anchors} anchors"
        );
    }
}
