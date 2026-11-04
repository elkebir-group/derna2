//! The realization verdict: whether the molecule delivers what the design
//! required of it.
//!
//! A positional structure constraint binds the **designed** structure. The
//! returned design is the exact optimum among designs admitting a structure that
//! honours the constraint — a statement about the design space, not a prediction
//! about the molecule, which folds as its energy dictates. Two things can go
//! wrong once the sequence exists, and they are duals:
//!
//! | shape | how the free fold betrays it |
//! |---|---|
//! | forbid (`x`, a forbidden pair) | it **reinstates** the forbidden pair |
//! | require (`\|`, `( )`, a helix) | it **omits** the required pair |
//!
//! The reinstating half is the region families' problem and is already measured
//! ([`crate::free_fold_gap`]). The omitting half arrives with the require-shaped
//! constraint family, and it is what this module adds.
//!
//! Like its neighbour this is a **report, never a gate**. A design whose free
//! fold omits a required pair is still returned, still the exact constrained
//! optimum, and still carries its verdict. Gating would be inverse folding —
//! "the target is the MFE structure" is a condition on the optimum rather than
//! an additive term, so it can no more ride this DP than modularity could.
//!
//! # Why the verdict is an energy comparison and not a structure reading
//!
//! The tempting test for *"did the free fold omit this pair?"* is to fold the
//! sequence unconstrained and look for the pair in the recovered dot-bracket. That
//! test is wrong, and wrong in the direction that manufactures failures:
//! free-optimal structures are routinely **co-optimal**, so one traceback may
//! omit a pair that an equally optimal structure forms, and the report would
//! name a gap that does not exist.
//!
//! So every verdict here compares **optima**:
//!
//! - the whole constraint set is realized iff the best fold honouring it costs
//!   the same as the free fold;
//! - one required pair is realized iff the best fold *containing that pair*
//!   costs the same as the free fold.
//!
//! Both are exact `b → ∞` folds of one fixed sequence over a lane-of-one graph,
//! so a verdict costs `2 + |required pairs|` folds and no design work.
//!
//! The recovered free structure is still reported, and the reinstated list still
//! read off it — but as *attribution* for a verdict the energies already
//! delivered, never as the verdict itself.

use rna_mfe::Params;

use crate::coords::Layout;
use crate::free_fold::{for_each_pair, FreeFoldError};
use crate::must_pair::MustPairConstraint;
use crate::region::{PairConstraint, RegionConstraints};
use crate::structure::StructureConstraints;
use crate::Beam;

/// A beam wide enough that the folder runs the exact lattice, matching the
/// `b → ∞` convention of the fold-parity oracles.
const EXACT_BEAM: usize = 1 << 20;

/// Energies are integer dekacalories under the model, so equality is exact up to
/// the `/100.0` float division.
const EXACT: f64 = 1e-9;

/// What one required pair's own free fold says about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequiredPairVerdict {
    /// The pair's 5′ endpoint, absolute and 0-based.
    pub open: u32,
    /// Its 3′ endpoint.
    pub close: u32,
    /// Whether **some** free-optimal fold of the emitted sequence forms this
    /// pair. Decided by energy, so a co-optimal structure that forms it counts
    /// even when the recovered traceback does not show it.
    pub realized: bool,
    /// What insisting on this pair alone costs the emitted sequence:
    /// `MFE(best fold containing it) − MFE(free fold)`, in kcal/mol.
    /// Non-negative, and exactly zero when the pair is realized.
    pub cost_kcal_mol: f64,
}

/// Whether the molecule delivers the positional constraints a design was built
/// under.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationVerdict {
    /// `E_free` — the emitted sequence folded with no constraint at all.
    pub free_mfe_kcal_mol: f64,
    /// `E_honoured` — the best fold of that sequence honouring every positional
    /// constraint.
    pub honoured_mfe_kcal_mol: f64,
    /// `E_honoured − E_free`: what the constraint set costs the molecule, and
    /// the verdict. Non-negative; zero exactly when the sequence's own
    /// unconstrained optimum already honours everything asked of it.
    pub gap_kcal_mol: f64,
    /// One entry per required pair, in ascending 5′-endpoint order.
    pub required: Vec<RequiredPairVerdict>,
    /// Pairs of the free fold the positional mask forbids — the reinstating
    /// direction, keyed by position rather than by class. Because free optima
    /// are routinely co-optimal, an empty list is weaker evidence than a zero
    /// gap.
    pub reinstated: Vec<(u32, u32)>,
    /// Dot-bracket of the free fold, over the whole mRNA.
    pub free_structure: String,
}

impl RealizationVerdict {
    /// Whether the molecule's own optimum already honours everything the design
    /// required. The verdict, and the only field that is one.
    pub fn is_realized(&self) -> bool {
        self.gap_kcal_mol.abs() < EXACT
    }

    /// Required pairs no free-optimal fold forms.
    pub fn omitted(&self) -> impl Iterator<Item = &RequiredPairVerdict> {
        self.required.iter().filter(|v| !v.realized)
    }

    /// How many required pairs the molecule declines to form.
    pub fn omitted_count(&self) -> usize {
        self.omitted().count()
    }

    /// The share of required pairs some free-optimal fold forms. `None` when
    /// nothing was required, where a rate would be an invented `1.0`.
    pub fn realization_rate(&self) -> Option<f64> {
        (!self.required.is_empty()).then(|| {
            let formed = self.required.len() - self.omitted_count();
            formed as f64 / self.required.len() as f64
        })
    }
}

/// Measure `sequence` against the positional `constraints` it was designed
/// under, over the molecule `layout` describes.
///
/// Costs `2 + |required pairs|` exact folds of one fixed sequence: the free
/// fold, the fold honouring everything, and one per required pair. No design
/// work — the graph is lane-of-one, so the codon lattice is empty and the
/// adaptation half of the objective is identically zero.
///
/// With an inert constraint set the report is trivially realized: the honouring
/// fold *is* the free fold, the gap is zero, and no pair is required or
/// reinstated.
pub fn realization_verdict(
    params: &Params,
    sequence: &str,
    layout: &Layout,
    constraints: &StructureConstraints,
) -> Result<RealizationVerdict, FreeFoldError> {
    let seq = sequence.as_bytes();
    let (free_structure, free_energy) = rna_linearfold::fold(params, seq, EXACT_BEAM)?;
    let free = free_energy.to_kcal_mol();
    let honoured = honouring_mfe(params, sequence, layout, constraints, Beam::exact())?;

    // One pair at a time, so a verdict names which element the molecule
    // declined rather than only that it declined something.
    let required = constraints
        .required_pairs()
        .into_iter()
        .map(|(open, close)| {
            let mut only = StructureConstraints::none(layout.len());
            only.require_pair(layout, open, close).expect(
                "a pair already recorded in a valid constraint set is well-formed on its own",
            );
            let cost = honouring_mfe(params, sequence, layout, &only, Beam::exact())? - free;
            Ok(RequiredPairVerdict {
                open,
                close,
                realized: cost.abs() < EXACT,
                cost_kcal_mol: cost,
            })
        })
        .collect::<Result<Vec<_>, FreeFoldError>>()?;

    Ok(RealizationVerdict {
        free_mfe_kcal_mol: free,
        honoured_mfe_kcal_mol: honoured,
        gap_kcal_mol: honoured - free,
        required,
        reinstated: reinstated_pairs(&free_structure, constraints),
        free_structure,
    })
}

/// The best energy over structures of `sequence` that `constraints` honours, via
/// a lane-of-one scan at `λ = 1` (pure MFE — the CAI half is identically zero
/// over a graph with no codon lattice), under the CKY column policy `beam`.
///
/// Returns `+∞` when no structure honours the set. On [`Beam::exact`] that is a
/// real answer for a **fixed** sequence — unlike the design case, where
/// infeasibility is a statement about a whole lattice — and it makes the
/// constraint unrealizable rather than merely unrealized.
///
/// **What a finite `beam` changes, and it changes both readings.** The returned
/// energy is then an *upper* bound on the honouring optimum, because a pruned
/// column may drop the cell the optimum runs through; and `+∞` becomes a
/// statement about one search rather than a proof that no honouring structure
/// exists. A caller differencing two of these numbers gets a difference of two
/// upper bounds, which is a bound in neither direction unless both were measured
/// under the same policy — so a difference is reportable only as a matched pair,
/// with the width named.
pub fn honouring_mfe(
    params: &Params,
    sequence: &str,
    layout: &Layout,
    constraints: &StructureConstraints,
    beam: Beam,
) -> Result<f64, FreeFoldError> {
    let bad_byte = || {
        FreeFoldError::Fold(rna_linearfold::FoldError::InvalidNucleotide(
            sequence.as_bytes().first().copied().unwrap_or(b'?'),
        ))
    };
    let nts = crate::parse_utr(sequence).map_err(|_| bad_byte())?;
    let graph = crate::dfa::build_fixed_mrna(&nts);
    let region = PairConstraint::for_cds(
        &RegionConstraints::default(),
        layout.cds_start(),
        layout.cds_end(),
    )
    .with_mask(constraints.pair_mask());
    let must = MustPairConstraint::new(&constraints.required_positions(), graph.seq_len);
    let tables = crate::scan::scan_with_constraints(
        params,
        &graph,
        1.0,
        crate::ScanConfig {
            beam,
            ..crate::ScanConfig::default()
        },
        region,
        must,
    );
    Ok(tables.c[graph.ix(graph.end())].map_or(f64::INFINITY, |best| best.mfe.to_kcal_mol()))
}

/// The free fold's pairs that the positional mask forbids, ascending.
fn reinstated_pairs(structure: &str, constraints: &StructureConstraints) -> Vec<(u32, u32)> {
    let mask = constraints.pair_mask();
    let mut hits = Vec::new();
    for_each_pair(structure, |i, j| {
        let (open, close) = (i as u32, j as u32);
        if !mask.allows(open, close) {
            hits.push((open, close));
        }
    });
    hits.sort_unstable();
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hairpin whose own optimum is the stem it looks like.
    const HAIRPIN: &str = "GGGGGAAAACCCCC";

    fn params() -> Params {
        Params::load_default().expect("params")
    }

    /// A layout over `n` nucleotides of pure flank. A verdict folds a fixed
    /// sequence over a lane-of-one graph, which carries no codons, so "all
    /// flank" is the faithful shape — and unlike a CDS layout it spans any length
    /// rather than only multiples of three.
    fn layout(n: u32) -> Layout {
        Layout::new(n, 0, 0)
    }

    #[test]
    fn an_inert_constraint_set_is_trivially_realized() {
        let l = layout(HAIRPIN.len() as u32);
        let sc = StructureConstraints::none(l.len());
        let v = realization_verdict(&params(), HAIRPIN, &l, &sc).expect("folds");
        assert!(v.is_realized());
        assert_eq!(v.gap_kcal_mol, 0.0);
        assert!(v.required.is_empty());
        assert!(v.reinstated.is_empty());
        assert_eq!(v.realization_rate(), None, "no rate without a requirement");
    }

    /// A pair the sequence's own optimum already forms costs nothing and is
    /// reported realized.
    #[test]
    fn a_pair_the_free_fold_already_forms_is_realized() {
        let l = layout(HAIRPIN.len() as u32);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 0, 13).expect("well-formed");
        let v = realization_verdict(&params(), HAIRPIN, &l, &sc).expect("folds");
        assert!(v.is_realized(), "gap {}", v.gap_kcal_mol);
        assert_eq!(v.required.len(), 1);
        assert!(v.required[0].realized);
        assert_eq!(v.required[0].cost_kcal_mol, 0.0);
        assert_eq!(v.realization_rate(), Some(1.0));
        assert_eq!(v.omitted_count(), 0);
    }

    /// A pair the molecule declines: it is *spellable* — both endpoints can
    /// pair — but forming it costs energy, so no free-optimal fold contains it.
    /// This is the case the whole verdict exists for.
    #[test]
    fn a_pair_the_free_fold_declines_is_reported_omitted_with_its_cost() {
        let l = layout(HAIRPIN.len() as u32);
        let mut sc = StructureConstraints::none(l.len());
        // A register shift: `(0, 12)` is spellable `G · C`, but taking it costs
        // the stem a pair and widens the loop, so no free-optimal fold forms it.
        sc.require_pair(&l, 0, 12).expect("well-formed");
        let v = realization_verdict(&params(), HAIRPIN, &l, &sc).expect("folds");
        assert!(!v.is_realized(), "gap {}", v.gap_kcal_mol);
        assert!(v.gap_kcal_mol > 0.0);
        assert_eq!(v.omitted_count(), 1);
        assert!(v.required[0].cost_kcal_mol > 0.0);
        assert_eq!(v.realization_rate(), Some(0.0));
    }

    /// The gap is a claim about optima, so it never goes negative: insisting on
    /// something cannot make a sequence fold better than it freely would.
    #[test]
    fn the_gap_is_never_negative() {
        let l = layout(HAIRPIN.len() as u32);
        for (open, close) in [(0u32, 13u32), (1, 12), (4, 9), (2, 11)] {
            let mut sc = StructureConstraints::none(l.len());
            sc.require_pair(&l, open, close).expect("well-formed");
            let v = realization_verdict(&params(), HAIRPIN, &l, &sc).expect("folds");
            assert!(
                v.gap_kcal_mol >= -EXACT,
                "({open}, {close}) gave a negative gap {}",
                v.gap_kcal_mol
            );
        }
    }

    /// The forbidding direction, keyed by position: an `x` over a stem position
    /// is reinstated by the free fold, and the gap prices it.
    #[test]
    fn a_forbidden_position_the_free_fold_pairs_is_reported_reinstated() {
        let l = layout(HAIRPIN.len() as u32);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_unpaired(&l, 0, 1).expect("well-formed");
        let v = realization_verdict(&params(), HAIRPIN, &l, &sc).expect("folds");
        assert!(!v.is_realized());
        assert!(
            v.reinstated.iter().any(|&(i, _)| i == 0 || i == 1),
            "the free fold pairs a position marked x: {} / {:?}",
            v.free_structure,
            v.reinstated
        );
        assert!(v.required.is_empty(), "x requires nothing");
    }

    /// A required pair the sequence cannot spell at all is unrealizable rather
    /// than merely unrealized, and the infinite cost says so instead of
    /// panicking on an unreachable end node.
    #[test]
    fn a_pair_the_sequence_cannot_spell_reports_an_infinite_cost() {
        let l = layout(HAIRPIN.len() as u32);
        let mut sc = StructureConstraints::none(l.len());
        // Both endpoints are `G`, and `G · G` is not a canonical pair.
        sc.require_pair(&l, 0, 4).expect("well-formed as a request");
        let v = realization_verdict(&params(), HAIRPIN, &l, &sc).expect("folds");
        assert!(v.gap_kcal_mol.is_infinite());
        assert!(!v.required[0].realized);
        assert!(v.required[0].cost_kcal_mol.is_infinite());
    }

    /// The per-pair verdicts are decided by energy, not by reading the recovered
    /// structure — so a pair some co-optimal fold forms counts as realized even
    /// where the traceback shows a different one.
    ///
    /// `GGGGAAAACCCCAAAAGGGGAAAACCCC` has two identical hairpins; the exterior
    /// can also close them into one larger arrangement at equal cost, so at
    /// least one required pair here is realized without appearing in every
    /// optimal traceback.
    #[test]
    fn a_verdict_is_read_off_energies_rather_than_one_traceback() {
        const TWO: &str = "GGGGAAAACCCCAAAAGGGGAAAACCCC";
        let l = layout(TWO.len() as u32);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 0, 11).expect("well-formed");
        sc.require_pair(&l, 16, 27).expect("well-formed");
        let v = realization_verdict(&params(), TWO, &l, &sc).expect("folds");
        for pv in &v.required {
            let in_traceback = v.free_structure.as_bytes()[pv.open as usize] == b'('
                && v.free_structure.as_bytes()[pv.close as usize] == b')';
            if pv.realized {
                continue;
            }
            assert!(
                !in_traceback,
                "({}, {}) is in the recovered fold but was called omitted",
                pv.open, pv.close
            );
        }
        // Whatever the traceback shows, the two verdicts agree with the gap.
        assert_eq!(v.is_realized(), v.omitted_count() == 0);
    }
}
