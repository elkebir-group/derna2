//! The certified-approximate hull walk and sweep, against independent exact
//! solves.
//!
//! An approximate solve makes one claim per λ: its design scores at most its
//! reported gap above the optimum. The check is against `design_with_motifs`,
//! which shares no bound, ceiling or walk with the solves under test. The
//! walk makes one more: its envelope is within `ε` of the exact envelope at
//! every λ, which follows from the per-probe claims only through the walk's
//! own argument, so it is checked separately on a dense grid.
//!
//! Ways this could pass vacuously, and what rules each out:
//!
//! * **Every probe certified exactly.** Then each gap is `0`, any gap
//!   reported passes, and a solve that under-reported its gap (or claimed `0`
//!   while returning a worse design) would go unnoticed. The test asserts
//!   that some probe returned a design strictly above the optimum, so its true
//!   gap is positive and an under-report shows.
//! * **`ε` changes nothing.** Ruled out by the first point: an inexact probe
//!   is one the exact walk would not have returned. (A total of pruned pairs
//!   would not show it — an approximate walk closes gaps sooner, so it runs
//!   fewer probes and can prune fewer pairs in all while pruning more per
//!   fold.)

use derna_lib::bounded_motifs::{
    design_motifs_swept, design_motifs_swept_within, walk_motif_hull, walk_motif_hull_within,
    MotifHullWalk,
};
use derna_lib::{design, design_with_motifs, CodonUsage, Params, ScanConfig};

const ENDS: (f64, f64) = (0.0001, 0.99);
const ANCHORS: &[f64] = &[0.0001, 0.25, 0.5, 0.75, 0.99];
const ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
const TOL: f64 = 1e-6;

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

fn score(lambda: f64, mfe: i32, cai: f64) -> f64 {
    lambda * f64::from(mfe) - (1.0 - lambda) * cai * 100.0
}

fn envelope(walk: &MotifHullWalk, lambda: f64) -> f64 {
    walk.vertices
        .iter()
        .map(|v| score(lambda, v.mfe, v.cai))
        .fold(f64::INFINITY, f64::min)
}

/// The constrained optimum at `lambda` by a single exact solve.
fn optimum(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    motifs: &[String],
) -> f64 {
    let one = design_with_motifs(
        params,
        protein,
        usage,
        lambda,
        motifs,
        ScanConfig::default(),
    )
    .expect("single solve");
    lambda * (one.mfe_kcal_mol * 100.0).round() - (1.0 - lambda) * one.cai_log * 100.0
}

/// Random short proteins with two motifs cut from their free optimum, so the
/// constraint costs something.
fn cases() -> Vec<(String, Vec<String>)> {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_017);
    (0..6)
        .map(|_| {
            let len = rng.range(10, 14);
            let protein = random_protein(&mut rng, len);
            let free = design(&params, &protein, &usage, 0.5).unwrap().sequence;
            let motifs = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
            (protein, motifs)
        })
        .collect()
}

#[test]
fn zero_epsilon_is_the_exact_walk() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let config = ScanConfig::default();
    for (protein, motifs) in cases() {
        let exact = walk_motif_hull(
            &params,
            &protein,
            &usage,
            &motifs,
            ENDS,
            Some(ANCHORS),
            config,
        )
        .unwrap();
        let within = walk_motif_hull_within(
            &params,
            &protein,
            &usage,
            &motifs,
            ENDS,
            Some(ANCHORS),
            config,
            0.0,
            1,
        )
        .unwrap();
        let (Some(exact), Some(within)) = (exact, within) else {
            continue;
        };
        let spell = |w: &MotifHullWalk| -> Vec<(i32, u64, String, String)> {
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
        assert_eq!(
            spell(&exact),
            spell(&within),
            "vertices differ on {protein}"
        );
        let probes = |w: &MotifHullWalk| -> Vec<(u64, usize, usize, u64)> {
            w.probes
                .iter()
                .map(|p| (p.lambda.to_bits(), p.folds, p.pruned, p.gap.to_bits()))
                .collect()
        };
        assert_eq!(
            probes(&exact),
            probes(&within),
            "probes differ on {protein}"
        );
        assert!(within.probes.iter().all(|p| p.gap == 0.0));
    }
}

#[test]
fn approximate_walk_reports_a_gap_that_bounds_the_true_one() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let config = ScanConfig::default();
    let (mut probes, mut inexact, mut probes_exact, mut probes_within) = (0, 0, 0, 0);
    for (protein, motifs) in cases() {
        let Some(exact) = walk_motif_hull(
            &params,
            &protein,
            &usage,
            &motifs,
            ENDS,
            Some(ANCHORS),
            config,
        )
        .unwrap() else {
            continue;
        };
        probes_exact += exact.probes.len();
        for epsilon in [2.0, 20.0] {
            let within = walk_motif_hull_within(
                &params,
                &protein,
                &usage,
                &motifs,
                ENDS,
                Some(ANCHORS),
                config,
                epsilon,
                1,
            )
            .unwrap()
            .expect("feasible for the exact walk");
            if epsilon == 20.0 {
                probes_within += within.probes.len();
            }
            for p in &within.probes {
                let best = optimum(&params, &protein, &usage, p.lambda, &motifs);
                let true_gap = p.optimum - best;
                assert!(
                    true_gap >= -TOL,
                    "a probe scored below the optimum at λ={} on {protein}",
                    p.lambda
                );
                assert!(
                    p.gap + TOL >= true_gap,
                    "λ={} on {protein}: reported gap {} under the true gap {true_gap}",
                    p.lambda,
                    p.gap
                );
                // The floor carries the bound's float slack, so a gap may exceed ε by it.
                assert!(
                    p.gap <= epsilon + 2.0 * TOL,
                    "gap {} over ε {epsilon}",
                    p.gap
                );
                probes += 1;
                inexact += usize::from(true_gap > TOL);
            }
            for k in 0..=400 {
                let lambda = ENDS.0 + (ENDS.1 - ENDS.0) * f64::from(k) / 400.0;
                let excess = envelope(&within, lambda) - envelope(&exact, lambda);
                assert!(
                    (-TOL..=epsilon + TOL).contains(&excess),
                    "envelope {excess} off the exact one at λ={lambda} on {protein} (ε {epsilon})"
                );
            }
        }
    }
    eprintln!(
        "probes {probes} / inexact {inexact} / walk probes exact {probes_exact} vs ε=20 {probes_within}"
    );
    assert!(inexact > 0, "every probe was exact, so no gap was tested");
}

#[test]
fn approximate_sweep_reports_a_gap_that_bounds_the_true_one() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let config = ScanConfig::default();
    let lambdas = [0.9, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.02];
    let (mut inexact, mut from_incumbent) = (0, 0);
    for (protein, motifs) in cases() {
        let Some(exact) = design_motifs_swept(
            &params, &protein, &usage, &lambdas, ANCHORS, &motifs, config, None,
        )
        .unwrap() else {
            continue;
        };
        let within = design_motifs_swept_within(
            &params, &protein, &usage, &lambdas, ANCHORS, &motifs, config, None, 20.0, 1,
        )
        .unwrap()
        .expect("feasible for the exact sweep");
        for (e, s) in exact.solves.iter().zip(&within.solves) {
            let best = score(s.lambda, e.mfe, e.cai);
            let true_gap = score(s.lambda, s.mfe, s.cai) - best;
            assert!(true_gap >= -TOL);
            assert!(
                s.gap + TOL >= true_gap,
                "λ={} on {protein}: reported gap {} under the true gap {true_gap}",
                s.lambda,
                s.gap
            );
            assert!(s.gap <= 20.0 + 2.0 * TOL);
            assert!(!s.sequence.is_empty(), "an answer without its spelling");
            inexact += usize::from(true_gap > TOL);
            from_incumbent += usize::from(s.incumbent.is_some());
        }
    }
    eprintln!("sweep: inexact {inexact} / answered by an incumbent {from_incumbent}");
    assert!(inexact > 0, "every solve was exact, so no gap was tested");
}
