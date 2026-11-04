//! A λ sweep that shares its outside passes, against the same λ values solved
//! independently.
//!
//! [`design_motifs_swept`] claims every λ's full constrained optimum while
//! paying one unconstrained outside pass per anchor rather than per λ. Its
//! bound at a λ between two anchors is the **chord** of their outside values,
//! which is a floor only because each cell's outside value is concave in λ.
//! The reference is [`design_with_motifs`], whose exactness is proven against
//! brute-force enumeration in `motif_blacklist_fuzz`.
//!
//! Three ways this could pass vacuously, and what rules each out.
//!
//! * **The bound prunes nothing.** Then a bounded fold is the full fold and
//!   agreement is trivial. Every case reports its pruned pairs and the sweep
//!   asserts they are many.
//! * **The chord is never used.** A λ that coincides with an anchor reads one
//!   exact table, so the concavity argument is never exercised. The λ values
//!   here are offset from the anchors and the sweep asserts that most solves
//!   ran with two distinct anchors.
//! * **The chord is slack enough to be unfalsifiable.** A floor far below
//!   every cell's true value prunes nothing and cannot be caught by a
//!   differential test. `chord_floor_is_tight_enough_to_bite` asserts the
//!   chord-bounded folds prune nearly as hard as exact-anchor ones, so an
//!   inflated chord would change the answer rather than only the speed.

use derna_lib::bounded_motifs::{design_motifs_swept, CeilingSource};
use derna_lib::{design, design_with_motifs, CodonUsage, Params, ScanConfig};

/// Offset from every anchor, so each λ reads a chord of two.
const LAMBDAS: &[f64] = &[0.12, 0.37, 0.52, 0.68, 0.88];
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

/// A window of `seq` to forbid, so the constraint costs something on the
/// design it was cut from.
fn cut(rng: &mut Rng, seq: &str, len: usize) -> String {
    let at = rng.range(0, seq.len() - len);
    seq[at..at + len].to_string()
}

#[derive(Default)]
struct Coverage {
    solves: usize,
    chorded: usize,
    pruned: usize,
    one_fold: usize,
}

/// Every λ of one sweep against the same λ solved on its own.
fn check(protein: &str, motifs: &[String], cov: &mut Coverage) {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let config = ScanConfig::default();
    let swept = design_motifs_swept(
        &params, protein, &usage, LAMBDAS, ANCHORS, motifs, config, None,
    )
    .unwrap();
    let Some(swept) = swept else {
        // Infeasible for the sweep must be infeasible for every single solve.
        for &lambda in LAMBDAS {
            assert!(
                design_with_motifs(&params, protein, &usage, lambda, motifs, config).is_err(),
                "sweep found {protein} infeasible but a single solve did not"
            );
        }
        return;
    };
    assert_eq!(swept.solves.len(), LAMBDAS.len());
    for solve in &swept.solves {
        let one = design_with_motifs(&params, protein, &usage, solve.lambda, motifs, config)
            .expect("single solve");
        assert_eq!(
            (solve.mfe, (solve.cai * 1e9).round()),
            (
                (one.mfe_kcal_mol * 100.0).round() as i32,
                (one.cai_log * 1e9).round()
            ),
            "swept optimum differs at λ={} on {protein} under {motifs:?}",
            solve.lambda
        );
        cov.solves += 1;
        cov.chorded += usize::from(solve.anchors.0 != solve.anchors.1);
        cov.pruned += solve.pruned;
        cov.one_fold += usize::from(solve.folds == 1);
    }
}

#[test]
fn swept_sweep_matches_independent_solves() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_004);
    let mut cov = Coverage::default();
    for _ in 0..8 {
        let len = rng.range(8, 14);
        let protein = random_protein(&mut rng, len);
        // Cut from the λ=0.5 free optimum, so the constraint costs something
        // and the sweep's first ceiling is not already the answer.
        let free = design(&params, &protein, &usage, 0.5).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
        check(&protein, &motifs, &mut cov);
    }
    eprintln!(
        "solves {} / chorded {} / one fold {} / pruned pairs {}",
        cov.solves, cov.chorded, cov.one_fold, cov.pruned
    );
    assert!(cov.solves > 0, "no case was feasible");
    assert!(
        cov.chorded * 2 > cov.solves,
        "most solves read a single exact anchor, so the chord went untested"
    );
    assert!(cov.pruned > 0, "the bound never pruned a pair");
    // Every λ past the first inherits a feasible ceiling, so it certifies in
    // one fold. If this regresses the sweep still answers correctly, only
    // slower — which is exactly the failure a differential test cannot see.
    assert!(
        cov.one_fold * 2 > cov.solves,
        "most solves needed a raised ceiling, so the inherited ceiling is not working"
    );
}

/// The chord must be a floor, and a floor only earns its place by being tight
/// enough to prune. Compare the pruning a chorded λ achieves against the same
/// λ given its own exact anchor: a chord that had gone slack would prune
/// visibly less, and one that had gone *above* the true value would change
/// the optimum — which `swept_sweep_matches_independent_solves` would catch.
#[test]
fn chord_floor_is_tight_enough_to_bite() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_005);
    let config = ScanConfig::default();
    let mut compared = 0;
    for _ in 0..6 {
        let len = rng.range(10, 14);
        let protein = random_protein(&mut rng, len);
        let free = design(&params, &protein, &usage, 0.5).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
        let lambda = 0.52;
        // One sweep whose anchors bracket λ, one whose anchor is λ itself.
        let chorded = design_motifs_swept(
            &params,
            &protein,
            &usage,
            &[lambda],
            ANCHORS,
            &motifs,
            config,
            None,
        )
        .unwrap();
        let exact = design_motifs_swept(
            &params,
            &protein,
            &usage,
            &[lambda],
            &[lambda],
            &motifs,
            config,
            None,
        )
        .unwrap();
        let (Some(chorded), Some(exact)) = (chorded, exact) else {
            continue;
        };
        let (c, e) = (&chorded.solves[0], &exact.solves[0]);
        assert_ne!(c.anchors.0, c.anchors.1, "λ={lambda} did not read a chord");
        assert_eq!(e.anchors.0, e.anchors.1, "the control read a chord");
        assert_eq!((c.mfe, c.cai), (e.mfe, e.cai), "chord changed the optimum");
        // A chord to λ = 0 floors pairs before their cells are scored, so it
        // lowers the pruned-cell count while discarding more: the count then
        // no longer compares the two anchors' bounds alone.
        if c.zero_chords > 0 || e.zero_chords > 0 {
            continue;
        }
        assert!(
            e.pruned > 0,
            "the exact-anchor control pruned nothing, so there was nothing for the chord to lose"
        );
        // Both ran against the unconstrained optimum as their first ceiling,
        // so the comparison is of the bounds alone.
        let ratio = c.pruned as f64 / e.pruned as f64;
        assert!(
            ratio > 0.5,
            "the chord pruned {ratio:.2} of what its own exact anchor did on {protein}: \
             a floor this slack buys nothing"
        );
        compared += 1;
    }
    assert!(compared > 0, "no case produced a chord to compare");
}

/// A λ outside the anchor grid must still be solved exactly. The chord is a
/// floor only *between* two anchors, so the sweep widens its grid to span the
/// λ values rather than extrapolate — which would prune cells an optimal
/// design needs and return something suboptimal.
#[test]
fn lambda_outside_the_anchor_grid_is_still_exact() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let config = ScanConfig::default();
    let mut rng = Rng::new(20_261_006);
    // Anchors that cover only the middle, and λ values past both ends.
    let anchors = [0.4, 0.6];
    let lambdas = [0.1, 0.5, 0.9];
    let mut checked = 0;
    for _ in 0..4 {
        let len = rng.range(10, 14);
        let protein = random_protein(&mut rng, len);
        let free = design(&params, &protein, &usage, 0.5).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
        let Some(swept) = design_motifs_swept(
            &params, &protein, &usage, &lambdas, &anchors, &motifs, config, None,
        )
        .unwrap() else {
            continue;
        };
        for solve in &swept.solves {
            let one = design_with_motifs(&params, &protein, &usage, solve.lambda, &motifs, config)
                .expect("single solve");
            assert_eq!(
                (solve.mfe, (solve.cai * 1e9).round()),
                (
                    (one.mfe_kcal_mol * 100.0).round() as i32,
                    (one.cai_log * 1e9).round()
                ),
                "λ={} outside the grid lost the optimum on {protein}",
                solve.lambda
            );
        }
        // The grid was widened to the extreme λ values, not left as given.
        assert!(
            swept.anchor_lambdas.first().copied() == Some(0.1)
                && swept.anchor_lambdas.last().copied() == Some(0.9),
            "the anchor grid {:?} does not span the λ values",
            swept.anchor_lambdas
        );
        checked += 1;
    }
    assert!(checked > 0, "no case was feasible");
}

/// A beam-seeded ceiling must not change any answer. A beam design is
/// feasible, so its objective caps the optimum; seeding from it only prunes
/// harder. If it ever changed an optimum, the beam fold would be emitting
/// something the exact lattice does not contain.
#[test]
fn a_beam_seeded_ceiling_keeps_every_optimum() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let config = ScanConfig::default();
    let mut rng = Rng::new(20_261_007);
    let mut seeded_by_beam = 0;
    let mut compared = 0;
    for _ in 0..6 {
        let len = rng.range(10, 14);
        let protein = random_protein(&mut rng, len);
        let free = design(&params, &protein, &usage, 0.5).unwrap().sequence;
        let motifs = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
        let with = design_motifs_swept(
            &params,
            &protein,
            &usage,
            LAMBDAS,
            ANCHORS,
            &motifs,
            config,
            Some(derna_lib::Beam::flat(16)),
        )
        .unwrap();
        let without = design_motifs_swept(
            &params, &protein, &usage, LAMBDAS, ANCHORS, &motifs, config, None,
        )
        .unwrap();
        let (Some(with), Some(without)) = (with, without) else {
            continue;
        };
        for (a, b) in with.solves.iter().zip(&without.solves) {
            assert_eq!(
                (a.lambda, a.mfe, (a.cai * 1e9).round()),
                (b.lambda, b.mfe, (b.cai * 1e9).round()),
                "a beam-seeded ceiling changed the optimum at λ={} on {protein}",
                a.lambda
            );
            compared += 1;
            if a.ceiling_from == CeilingSource::Beam {
                seeded_by_beam += 1;
            }
        }
    }
    assert!(compared > 0, "no case was feasible");
    // Without this the test passes whenever the beam never wins the ceiling,
    // which is the same as not running it.
    assert!(
        seeded_by_beam > 0,
        "the beam never supplied a ceiling, so nothing about it was tested"
    );
}
