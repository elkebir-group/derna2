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
use crate::free_fold::FreeFoldError;
use crate::must_pair::MustPairConstraint;
use crate::region::{PairConstraint, RegionConstraints};
use crate::structure::StructureConstraints;
use crate::Beam;

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
    let bad_byte =
        || FreeFoldError::InvalidNucleotide(sequence.as_bytes().first().copied().unwrap_or(b'?'));
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
