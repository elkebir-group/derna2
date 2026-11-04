//! A per-position nucleotide preference over designed UTR positions — the
//! structural analogue of CAI.
//!
//! A designed UTR position earns no CAI: the adaptation half of the objective is
//! the CDS codons alone. Nothing else in the objective mentions such a position
//! either, so its nucleotide is settled entirely by whichever folding energy the
//! surrounding structure can reach — and when a constraint set forbids the flank
//! every pair it could form, by nothing at all, leaving every spelling of the
//! flank exactly co-optimal. This term is what prices it.
//!
//! # The term
//!
//! For a nucleotide profile `f` over `{A, C, G, U}` and a strength `β ≥ 0`,
//! every designed UTR position contributes
//!
//! ```text
//! w(x) = β · log( f(x) / max_y f(y) )
//! ```
//!
//! and a design's total is the sum over its designed UTR positions. CAI is a
//! per-codon log-odds; this is the same quantity one nucleotide at a time.
//!
//! # Why the weights are normalized to a maximum of zero
//!
//! Not for interpretability — for correctness. The scan does not sum edge
//! weights across long unpaired spans; it calls [`crate::dfa::Dfa`]'s `O(1)`
//! span closed form, which returns the **best still-reachable** weight over the
//! span and relies on every whole unit in the span's interior contributing
//! zero. CAI's own weights are cost-to-go normalized so a codon's best synonym
//! telescopes to zero, which is what makes that closed form exact. Normalizing
//! these weights the same way — maximum exactly zero at the profile's most
//! favoured nucleotide — makes a designed UTR position contribute zero for the
//! same reason, so the closed form and every bound derived from it stay exact
//! with no change.
//!
//! # What this term is not
//!
//! It states a **preference**, never a **target**. `U` is a sum of independent
//! per-position weights, so on its own it is maximized by the profile's argmax
//! nucleotide at *every* position — a homopolymer. Any intermediate composition
//! an optimum displays comes from the folding energy it is trading against, not
//! from the profile, and is therefore a property of the instance rather than of
//! the knob.
//!
//! In particular **a GC-fraction target is not expressible here, or anywhere in
//! this DP**. A fraction is a ratio of global counts: the cost of setting one
//! position to G depends on every other position, so it is not a sum over parts
//! and cannot be carried in a min-plus cell without carrying the count as DP
//! state. That is the same obstruction the free-fold modularity predicate has. A
//! profile weighted towards G and C raises GC on average; it does not hit a
//! number, and no choice of weights makes it do so.

use rna_mfe::Nucleotide;

use crate::BeamError;

/// Per-nucleotide weights for designed UTR positions, log-odds of a caller
/// supplied profile scaled by a strength, normalized to a maximum of zero.
///
/// A default-constructed value is **inert** — every weight zero, so a designed
/// UTR position is priced by nothing and the design is byte-identical to one
/// built without the term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtrProfile {
    /// Weight per nucleotide, indexed by [`Nucleotide`]'s discriminant
    /// (`A`, `C`, `G`, `U`). Every entry is `≤ 0` and at least one is exactly
    /// `0`.
    weights: [f64; 4],
}

impl Default for UtrProfile {
    fn default() -> Self {
        Self::inert()
    }
}

impl UtrProfile {
    /// The inert profile: every nucleotide weight zero. A designed UTR position
    /// under it is priced by nothing, which is the substrate's behaviour without
    /// the term.
    pub fn inert() -> Self {
        Self { weights: [0.0; 4] }
    }

    /// Build a profile from nucleotide frequencies `[A, C, G, U]` and a strength
    /// `weight`.
    ///
    /// The frequencies need not sum to one — only their ratios matter, since the
    /// weights are log-odds against the largest. Every frequency must be finite
    /// and strictly positive: a zero would forbid a nucleotide outright, which is
    /// a hard constraint wearing an objective's clothing and belongs in the DFA
    /// rather than in a weight. `weight` must be finite and non-negative; zero
    /// yields the inert profile whatever the frequencies say.
    pub fn new(frequencies: [f64; 4], weight: f64) -> Result<Self, BeamError> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(BeamError::InvalidUtrProfile(
                "strength must be finite and >= 0",
            ));
        }
        if frequencies.iter().any(|f| !f.is_finite() || *f <= 0.0) {
            return Err(BeamError::InvalidUtrProfile(
                "every nucleotide frequency must be finite and > 0",
            ));
        }
        let max = frequencies.iter().copied().fold(f64::MIN, f64::max);
        let mut weights = [0.0; 4];
        for (w, f) in weights.iter_mut().zip(frequencies) {
            *w = weight * (f / max).ln();
        }
        Ok(Self { weights })
    }

    /// Whether every weight is zero, so the term prices nothing. The scan's fast
    /// path and the byte-identity floors turn on this.
    pub fn is_inert(&self) -> bool {
        self.weights.iter().all(|w| *w == 0.0)
    }

    /// The weight a designed UTR position spelling `nt` contributes. Always
    /// `≤ 0`, and exactly `0` at the profile's most favoured nucleotide.
    pub fn weight(&self, nt: Nucleotide) -> f64 {
        self.weights[nt as usize]
    }

    /// The term's total over an mRNA, counting only the positions in
    /// `designed`. Used to report the term beside the CDS-only CAI rather than
    /// folded into it, and by the oracle to score a candidate directly.
    pub fn score(&self, sequence: &[Nucleotide], designed: impl Fn(usize) -> bool) -> f64 {
        sequence
            .iter()
            .enumerate()
            .filter(|(i, _)| designed(*i))
            .map(|(_, nt)| self.weight(*nt))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    //! The two properties everything downstream rests on: the weights are
    //! normalized to a maximum of zero, and an inert profile is recognizable as
    //! such.

    use super::*;

    #[test]
    fn inert_profile_weighs_nothing() {
        let p = UtrProfile::inert();
        assert!(p.is_inert());
        for nt in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
            assert_eq!(p.weight(nt), 0.0);
        }
    }

    #[test]
    fn weights_are_nonpositive_with_a_zero_maximum() {
        let p = UtrProfile::new([0.4, 0.1, 0.1, 0.4], 1.0).unwrap();
        let ws: Vec<f64> = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U]
            .map(|nt| p.weight(nt))
            .to_vec();
        assert!(ws.iter().all(|w| *w <= 0.0), "weights must be <= 0: {ws:?}");
        assert_eq!(ws.iter().copied().fold(f64::MIN, f64::max), 0.0);
        // Equally-favoured nucleotides get equal weights, and the ratio is what
        // the log-odds reads — not the absolute frequency.
        assert_eq!(p.weight(Nucleotide::A), p.weight(Nucleotide::U));
        assert!((p.weight(Nucleotide::C) - 0.25_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn strength_scales_the_log_odds() {
        let one = UtrProfile::new([0.4, 0.1, 0.1, 0.4], 1.0).unwrap();
        let three = UtrProfile::new([0.4, 0.1, 0.1, 0.4], 3.0).unwrap();
        assert!((3.0 * one.weight(Nucleotide::C) - three.weight(Nucleotide::C)).abs() < 1e-12);
    }

    #[test]
    fn zero_strength_is_inert_whatever_the_frequencies() {
        assert!(UtrProfile::new([0.97, 0.01, 0.01, 0.01], 0.0)
            .unwrap()
            .is_inert());
    }

    #[test]
    fn a_uniform_profile_is_inert() {
        assert!(UtrProfile::new([0.25; 4], 1.0).unwrap().is_inert());
    }

    #[test]
    fn degenerate_inputs_are_rejected() {
        assert!(UtrProfile::new([0.0, 0.3, 0.3, 0.4], 1.0).is_err());
        assert!(UtrProfile::new([0.25; 4], -1.0).is_err());
        assert!(UtrProfile::new([f64::NAN, 0.3, 0.3, 0.4], 1.0).is_err());
    }
}
