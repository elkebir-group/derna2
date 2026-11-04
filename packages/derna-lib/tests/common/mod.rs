//! The enumeration oracle for designing under an energy ceiling, shared by
//! everything that checks a ceiling's answer.
//!
//! Shared rather than duplicated for a reason that is not tidiness. The oracle
//! settles ties the same way the front does — among designs of equal adaptation
//! the lower-energy one is kept — and a comparison between two reductions that
//! tie-break differently proves nothing about either. One implementation is what
//! makes "byte-exact against the oracle" a statement about the solver rather
//! than about two conventions agreeing.

#![allow(dead_code)]

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::{dfa, fold_single_path, CodonUsage, FrontPoint, Nucleotide, Params};

/// Dominance tolerance, matching the front operator's own: two log-CAI values
/// within this slop are the same trade-off (ULP-twin codon multisets).
pub const CAI_DOM_EPS: f64 = 1e-9;

/// Match tolerance when comparing surviving points, after both sides have
/// collapsed their ULP-twins.
pub const CAI_EPS: f64 = 1e-6;

/// One design's objectives, energy in the integer dekacalories the fold works
/// in so a threshold comparison is exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Objective {
    pub mfe: i32,
    pub cai_log: f64,
}

impl Objective {
    pub fn weakly_dominates(self, other: Objective) -> bool {
        self.mfe <= other.mfe && self.cai_log >= other.cai_log - CAI_DOM_EPS
    }
}

/// Every encoding of `protein`, folded, with no elimination — the bag a ceiling
/// is applied to.
pub fn enumerate_all(params: &Params, usage: &CodonUsage, protein: &str) -> Vec<Objective> {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = dfa::build(&aa, usage);
    graph
        .paths_between(graph.start(), graph.end())
        .into_iter()
        .map(|(cai_log, seq)| {
            let codons: Vec<[u8; 3]> = seq
                .chunks_exact(3)
                .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
                .collect();
            Objective {
                mfe: (fold_single_path(params, &aa, &codons).mfe_kcal_mol * 100.0).round() as i32,
                cai_log,
            }
        })
        .collect()
}

/// Dominance-eliminate a bag into the staircase, energy ascending.
pub fn eliminate(mut bag: Vec<Objective>) -> Vec<Objective> {
    bag.sort_by(|a, b| {
        a.mfe
            .cmp(&b.mfe)
            .then_with(|| b.cai_log.total_cmp(&a.cai_log))
    });
    let mut front: Vec<Objective> = Vec::new();
    for cand in bag {
        if front.last().is_some_and(|k| k.weakly_dominates(cand)) {
            continue;
        }
        front.push(cand);
    }
    front
}

/// A recovered front in the oracle's units and order.
pub fn front_objectives(points: &[FrontPoint]) -> Vec<Objective> {
    let mut objs: Vec<Objective> = points
        .iter()
        .map(|p| Objective {
            mfe: (p.mfe_kcal_mol * 100.0).round() as i32,
            cai_log: p.cai_log,
        })
        .collect();
    objs.sort_by_key(|o| o.mfe);
    objs
}

pub fn assert_same(label: &str, expected: &[Objective], actual: &[Objective]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: {} points expected, {} produced\n  expected {expected:?}\n  produced {actual:?}",
        expected.len(),
        actual.len(),
    );
    for (e, a) in expected.iter().zip(actual) {
        assert_eq!(e.mfe, a.mfe, "{label}: energy differs");
        assert!(
            (e.cai_log - a.cai_log).abs() < CAI_EPS,
            "{label}: adaptation differs at energy {}: {} vs {}",
            e.mfe,
            e.cai_log,
            a.cai_log,
        );
    }
}

/// The best log-adaptation in a bag, or negative infinity if it is empty.
pub fn best_adaptation(bag: &[Objective]) -> f64 {
    bag.iter()
        .map(|o| o.cai_log)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Proteins small enough to enumerate, chosen to span the loop motifs the fold
/// has to get right under a ceiling: a hairpin-only case, one that admits a
/// bifurcation, and one whose residues are degenerate enough to give the front
/// real width.
pub const CASES: [&str; 3] = ["MWWKFYCLMW", "MYGKIIFVLM", "MCWFGGWCKM"];

/// Cases for the assertions that need a front wide enough to be non-convex but
/// do not enumerate. Keeping them separate is what lets the enumerating tests
/// stay cheap: the encoding count is the enumeration's whole cost and has
/// nothing to do with whether the front has a dent in it.
pub const WIDE_CASES: [&str; 2] = ["MYGKIIFVLL", "MYGKIIFVLLL"];

/// Fold a recovered coding sequence back to its own minimum free energy, in the
/// oracle's integer units.
///
/// The check that a front point's traceback is a real answer rather than a
/// bookkeeping artifact: the sequence is re-folded from scratch, by the
/// single-path evaluator rather than by the DP that emitted it.
pub fn refold(params: &Params, protein: &str, sequence: &str) -> i32 {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let index = |b: u8| Nucleotide::from_byte(b).unwrap() as u8;
    let codons: Vec<[u8; 3]> = sequence
        .as_bytes()
        .chunks_exact(3)
        .map(|c| [index(c[0]), index(c[1]), index(c[2])])
        .collect();
    (fold_single_path(params, &aa, &codons).mfe_kcal_mol * 100.0).round() as i32
}
