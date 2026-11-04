//! Lazy motif enforcement against the full constrained fold.
//!
//! [`design_motifs_lazy`] claims the full constrained optimum's BRD objective
//! while folding only under the motifs the optimum actually spells. The
//! reference is [`design_with_motifs`], whose exactness is proven against
//! brute-force enumeration in `motif_blacklist_fuzz`; agreement with it on the
//! objective, a motif-free sequence and the same feasibility verdict is the
//! claim here.
//!
//! Two paths need covering and either could pass vacuously without the other:
//! a single round (the free optimum was already clean, under a set that does
//! constrain the lattice) and several rounds (the free optimum spelled a
//! forbidden motif). Motifs cut from the free optimum itself force the second;
//! motifs cut from a random encoding usually give the first. The sweep counts
//! both and asserts each occurred.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::{Ac, MotifWindow};
use derna_lib::lazy_motifs::design_motifs_lazy;
use derna_lib::positional::PositionalConstraints;
use derna_lib::region::RegionConstraints;
use derna_lib::{
    design, design_with_motifs, drops_lattice_edge, Beam, BeamError, CodonUsage, CompositeSpec,
    Nucleotide, Params, ScanConfig, UtrProfile, UtrSpec,
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
    one_round_binding: usize,
    multi_round: usize,
    infeasible: usize,
}

fn check(protein: &str, motifs: &[String], lambda: f64, cov: &mut Coverage) {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let empty = UtrSpec::Fixed(String::new());
    let ctx = format!("protein={protein} motifs={motifs:?} λ={lambda}");
    let lazy = design_motifs_lazy(
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
    match (lazy, full) {
        (Err(BeamError::Infeasible), Err(BeamError::Infeasible)) => cov.infeasible += 1,
        (Ok(l), Ok(f)) => {
            assert!(l.lazy, "{ctx}: an eligible run was not lazy");
            assert!(
                is_motif_free(motifs, &l.design.sequence),
                "{ctx}: lazy spelled a motif"
            );
            let (got, want) = (
                brd(l.design.mfe_kcal_mol, l.design.cai_log, lambda),
                brd(f.mfe_kcal_mol, f.cai_log, lambda),
            );
            assert!((got - want).abs() < 1e-6, "{ctx}: lazy={got} full={want}");
            assert!(l.rounds <= motifs.len() + 1, "{ctx}: {} rounds", l.rounds);
            if l.rounds > 1 {
                cov.multi_round += 1;
            } else if drops_lattice_edge(protein, &usage, &spec(&empty, motifs)).unwrap()
                == Some(true)
            {
                cov.one_round_binding += 1;
            }
        }
        (l, f) => panic!(
            "{ctx}: verdicts differ: lazy={:?} full={:?}",
            l.map(|d| d.rounds),
            f.map(|d| d.mfe_kcal_mol)
        ),
    }
}

#[test]
fn lazy_matches_full_constrained_fold() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rng = Rng::new(20_261_003);
    let mut cov = Coverage::default();
    for _ in 0..24 {
        let len = rng.range(5, 9);
        let protein = random_protein(&mut rng, len);
        for &lambda in LAMBDAS {
            let free = design(&params, &protein, &usage, lambda).unwrap().sequence;
            let other = random_encoding(&mut rng, &protein);
            let k = rng.range(4, 6);
            // From the free optimum: the first round must spell it.
            let from_free = vec![cut(&mut rng, &free, k), cut(&mut rng, &other, 5)];
            // From a random encoding only: binds the lattice, often misses the optimum.
            let from_other = vec![cut(&mut rng, &other, 5), cut(&mut rng, &other, 6)];
            check(&protein, &from_free, lambda, &mut cov);
            check(&protein, &from_other, lambda, &mut cov);
        }
    }
    assert!(cov.multi_round > 0, "no case needed a second round");
    assert!(
        cov.one_round_binding > 0,
        "no binding case finished in one round"
    );
    eprintln!(
        "multi-round {} / one-round binding {} / infeasible {}",
        cov.multi_round, cov.one_round_binding, cov.infeasible
    );
}

/// A finite beam bounds nothing, so the relaxation argument does not hold and
/// the run must take the full fold.
#[test]
fn finite_beam_is_not_lazy() {
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
    let got =
        design_motifs_lazy(&params, "MAGA", &usage, 0.5, &spec(&empty, &motifs), config).unwrap();
    assert!(!got.lazy);
    assert_eq!(got.rounds, 1);
}
