//! The outside-bounded threshold search against the full constrained fold.
//!
//! [`design_motifs_bounded`] claims the full constrained optimum's BRD
//! objective while folding only the closed pairs whose unconstrained outside
//! leaves them under a ceiling. The reference is [`design_with_motifs`],
//! whose exactness is proven against brute-force enumeration in
//! `motif_blacklist_fuzz`.
//!
//! The claim can pass vacuously in two ways, and the sweep rules both out. A
//! bound that prunes nothing reproduces the full fold trivially, so every
//! case must report pruned pairs and the sweep asserts a large share. A
//! search that certifies at its first ceiling never exercises the raise, so
//! motifs are also cut from the free optimum — the constraint then costs
//! something, the first ceiling is too low, and the sweep asserts that some
//! case needed several folds.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::{Ac, MotifWindow};
use derna_lib::bounded_motifs::design_motifs_bounded;
use derna_lib::positional::PositionalConstraints;
use derna_lib::region::RegionConstraints;
use derna_lib::{
    design, design_with_motifs, Beam, BeamError, CodonUsage, CompositeSpec, Nucleotide, Params,
    ScanConfig, UtrProfile, UtrSpec,
};

const LAMBDAS: &[f64] = &[0.0001, 0.5, 0.99];
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

fn random_encoding(rng: &mut Rng, protein: &str) -> String {
    protein
        .chars()
        .flat_map(|c| {
            let codons = &AMINO_ACIDS[aa_to_index(c).unwrap()].codons;
            let codon = &codons[rng.next() % codons.len()];
            codon.nuc.map(|b| b"ACGU"[b as usize] as char)
        })
        .collect()
}

fn cut(rng: &mut Rng, seq: &str, len: usize) -> String {
    let start = rng.range(0, seq.len() - len);
    seq[start..start + len].to_string()
}

fn brd(mfe_kcal_mol: f64, cai_log: f64, lambda: f64) -> f64 {
    lambda * (mfe_kcal_mol * 100.0) - (1.0 - lambda) * (cai_log * 100.0)
}

fn is_motif_free(motifs: &[String], seq: &str) -> bool {
    let ac = Ac::build(motifs).unwrap();
    let mut q = ac.start();
    !seq.bytes().any(|b| {
        q = ac.step(q, Nucleotide::from_byte(b).unwrap());
        ac.is_accept(q)
    })
}

fn spec<'a>(empty: &'a UtrSpec, motifs: &'a [String]) -> CompositeSpec<'a, String> {
    CompositeSpec {
        utr5: empty,
        utr3: empty,
        constraints: RegionConstraints::default(),
        profile: UtrProfile::inert(),
        motifs,
        motif_window: MotifWindow::all(),
        flank_complement: None,
        complement_classes: None,
        positional: PositionalConstraints::empty(),
    }
}

#[derive(Default)]
struct Coverage {
    bounded: usize,
    several_folds: usize,
    pruned: usize,
    seen: usize,
}

fn check(protein: &str, motifs: &[String], lambda: f64, cov: &mut Coverage) {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let empty = UtrSpec::Fixed(String::new());
    let ctx = format!("protein={protein} motifs={motifs:?} λ={lambda}");
    let bounded = design_motifs_bounded(
        &params,
        protein,
        &usage,
        lambda,
        &spec(&empty, motifs),
        ScanConfig::default(),
    );
    let full = design_with_motifs(
        &params,
        protein,
        &usage,
        lambda,
        motifs,
        ScanConfig::default(),
    );
    match (bounded, full) {
        (Err(BeamError::Infeasible), Err(BeamError::Infeasible)) => {}
        (Ok(b), Ok(f)) => {
            assert!(
                is_motif_free(motifs, &b.design.sequence),
                "{ctx}: bounded spelled a motif"
            );
            let (got, want) = (
                brd(b.design.mfe_kcal_mol, b.design.cai_log, lambda),
                brd(f.mfe_kcal_mol, f.cai_log, lambda),
            );
            assert!(
                (got - want).abs() < 1e-6,
                "{ctx}: bounded={got} full={want}"
            );
            let r = &b.report;
            if r.bounded {
                cov.bounded += 1;
                cov.several_folds += usize::from(r.ceilings.len() > 1);
                cov.pruned += r.pruned.iter().sum::<usize>();
                assert_eq!(r.ceilings.len(), r.pruned.len(), "{ctx}");
                assert!(r.free_optimum.is_finite(), "{ctx}");
            }
        }
        (b, f) => panic!(
            "{ctx}: verdicts differ: bounded={:?} full={:?}",
            b.map(|d| d.report.ceilings),
            f.map(|d| d.mfe_kcal_mol)
        ),
    }
    cov.seen += 1;
}

#[test]
fn bounded_matches_full_constrained_fold() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_004);
    let mut cov = Coverage::default();
    for _ in 0..16 {
        let len = rng.range(8, 14);
        let protein = random_protein(&mut rng, len);
        for &lambda in LAMBDAS {
            let free = design(&params, &protein, &usage, lambda).unwrap().sequence;
            let other = random_encoding(&mut rng, &protein);
            // Cut from the free optimum, the constraint costs something and the
            // first ceiling is too low.
            let from_free = vec![cut(&mut rng, &free, 5), cut(&mut rng, &free, 6)];
            let from_other = vec![cut(&mut rng, &other, 5), cut(&mut rng, &other, 6)];
            check(&protein, &from_free, lambda, &mut cov);
            check(&protein, &from_other, lambda, &mut cov);
        }
    }
    eprintln!(
        "bounded {} of {} / several folds {} / pruned pairs {}",
        cov.bounded, cov.seen, cov.several_folds, cov.pruned
    );
    assert!(cov.bounded * 2 > cov.seen, "most cases never ran bounded");
    assert!(cov.several_folds > 0, "no case needed a raised ceiling");
    assert!(cov.pruned > 0, "the bound never pruned a pair");
}

/// A pruned fold is a minimum only on the exact lattice, so a finite beam
/// takes the ordinary fold.
#[test]
fn finite_beam_is_not_bounded() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let empty = UtrSpec::Fixed(String::new());
    let motifs = vec!["GCC".to_string()];
    let config = ScanConfig {
        beam: Beam::flat(4),
        ..ScanConfig::default()
    };
    let got = design_motifs_bounded(&params, "MAGA", &usage, 0.5, &spec(&empty, &motifs), config)
        .unwrap();
    assert!(!got.report.bounded);
    assert!(got.report.ceilings.is_empty());
}
