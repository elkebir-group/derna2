//! Pricing a cheap region bound for the branch-and-bound search, before one is
//! built.
//!
//! A region's key in [`crate::bounded_search`] is its own constrained optimum,
//! and one whole-mRNA solve computes it. A lower bound readable from the region
//! itself could reject a region *before* that solve, and that is the only thing
//! left that can make a proof shorter: a best-first walk opens every region
//! whose bound falls below the optimum, a set fixed by the bound and by the
//! answer, so neither a better incumbent nor a different branch order can
//! shrink it.
//!
//! This module composes such bounds and records them beside the exact ones the
//! walk paid for. **It never prunes.** What it answers is whether a prune would
//! be worth building: how many of the regions a walk actually solved carry a
//! cheap bound that already reaches the optimum, and — where none do — by how
//! much the cheap bound falls short, which is what says whether the route is
//! close or hopeless.
//!
//! # The two halves of a bound
//!
//! Adaptation is separable. No structure can make a position spell better than
//! the best codon its region leaves it, so summing those per-position maxima
//! ceils every member's adaptation, and one pass over the positions computes it
//! ([`crate::dfa::cai_ceiling`]).
//!
//! Free energy is not separable, and a floor under it has to come from a solve
//! of *something*. Which solve is the whole question, and the three candidates
//! differ by an order of magnitude in price — see [`MfeFloors`].
//!
//! # Rounding
//!
//! The objective rounds the energy and the adaptation to dekacalories
//! separately before weighting them. A cheap bound that rounded the same way
//! could land *above* the value it claims to be under, and a bound above its
//! own region's optimum prunes that optimum away. So the composition floors the
//! energy term and ceils the adaptation one ([`cheap_bound`]), which costs at
//! most one dekacalorie of tightness and cannot cost correctness.

/// Dekacalories per kcal/mol — the scale a region bound and an incumbent are
/// compared on.
const DEKACAL: f64 = 100.0;

/// A lower bound on every objective a region admits, composed from a floor
/// under its free energy and a ceiling over its adaptation.
///
/// Valid whenever `mfe_floor` really is a floor: for any member `v`,
/// `λ·round(E(v)·100) ≥ λ·floor(mfe_floor·100)` and
/// `(1−λ)·round(CAI(v)·100) ≤ (1−λ)·ceil(cai_ceiling·100)`, so the composed
/// value is under the member's objective term by term.
pub fn cheap_bound(lambda: f64, mfe_floor: f64, cai_ceiling: f64) -> f64 {
    weigh(lambda, (mfe_floor * DEKACAL).floor())
        - weigh(1.0 - lambda, (cai_ceiling * DEKACAL).ceil())
}

/// `weight · value`, with a zero weight annihilating an infinite value rather
/// than producing a `NaN`.
///
/// A term the objective does not weigh contributes nothing, and that has to
/// hold when the quantity behind it is unbounded too — an absent floor at
/// `λ = 0` is a term the objective ignores, not an undefined bound. A `NaN`
/// here would compare false against every incumbent and quietly never prune.
fn weigh(weight: f64, value: f64) -> f64 {
    if weight == 0.0 {
        0.0
    } else {
        weight * value
    }
}

/// Three free energies available to a region before it is solved: two floors
/// under every encoding it admits, and one quantity that is not a floor at all.
///
/// They are ordered by price, and the point of recording all three is that the
/// cheapest is the loosest and the free one is unsound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfeFloors {
    /// The optimum of a pure-energy solve over the whole design space. One
    /// extra solve for an entire walk, valid for every region in it, and the
    /// loosest of the three — it knows nothing about where in the tree the
    /// region sits.
    pub space: f64,
    /// The optimum of a pure-energy solve over the region's parent. One extra
    /// solve per branching, amortised over the parent's children, and tighter
    /// by as much as the parent is smaller than the space.
    pub parent: f64,
    /// The free energy of the parent's own trade-off optimum — free, already in
    /// hand, and **not a floor**. The parent's optimum minimises
    /// `λE − (1−λ)CAI`, so an encoding trading energy away for adaptation can
    /// sit beneath it in energy alone. It is recorded to measure both halves of
    /// what it is worth: what a bound composed from it would have rejected, and
    /// how often it exceeds the exact bound it claims to be under.
    pub parent_optimum: f64,
}

/// What each of [`MfeFloors`] composes to against one region's adaptation
/// ceiling, on the objective's scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheapBounds {
    /// From [`MfeFloors::space`].
    pub space: f64,
    /// From [`MfeFloors::parent`].
    pub parent: f64,
    /// From [`MfeFloors::parent_optimum`]. Sound only if measurement says so.
    pub parent_optimum: f64,
}

impl MfeFloors {
    /// Each floor composed against `cai_ceiling`.
    pub fn compose(&self, lambda: f64, cai_ceiling: f64) -> CheapBounds {
        CheapBounds {
            space: cheap_bound(lambda, self.space, cai_ceiling),
            parent: cheap_bound(lambda, self.parent, cai_ceiling),
            parent_optimum: cheap_bound(lambda, self.parent_optimum, cai_ceiling),
        }
    }
}

/// What became of a region after it entered the heap.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Probed {
    /// Never examined — the walk ended, or its inherited bound reached the
    /// incumbent, before it was reached.
    #[default]
    Unsolved,
    /// Solved, and found to admit no encoding at all. The solve was spent to
    /// learn that; the adaptation ceiling reports the same emptiness in one
    /// pass over the positions.
    Empty,
    /// Solved: the region's exact bound, and the two quantities composing it.
    Solved {
        /// The exact bound — the objective at the region's own optimum.
        exact: f64,
        /// That optimum's constrained free energy.
        mfe: f64,
        /// That optimum's adaptation, which the ceiling bounds from above.
        cai: f64,
    },
}

/// One region's contribution to the probe: what was knowable about its bound
/// before its solve, and what the solve returned.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundProbeRow {
    /// How many of the region's positions are pinned to one codon.
    ///
    /// Depth in the region tree, and the axis both halves of a cheap bound are
    /// expected to improve along: a pinned position's adaptation ceiling is the
    /// codon it spells rather than the best of its family, and a deeper
    /// parent's energy floor is over a smaller set.
    pub pinned: usize,
    /// This region's index in its parent's partition — equivalently, how many
    /// positions it pins beyond its parent. Part `k` restricts `k + 1` of the
    /// parent's positions: `k` pinned to the parent's own solution and one more
    /// excluding it.
    ///
    /// Separate from [`Self::pinned`], which is depth in the tree. A warm start
    /// inherits from a region's *parent*, not from the root, so what it can
    /// reuse is set by this difference and not by that depth — a deep region
    /// entered as part 0 differs from its parent in one position.
    pub part: usize,
    /// The codon positions this region restricts relative to its parent, in
    /// branch order — so the **last** of them is the one carrying the added
    /// exclusion and the rest are pinned to the parent's own solution.
    ///
    /// The whole list rather than its extent, because a branch order need not
    /// hand a fold a contiguous run and the two shapes are not worth the same:
    /// a CKY cell is inheritable when its span misses every restricted
    /// position, so `k + 1` restrictions packed together leave far more spans
    /// clear than the same number spread across the molecule. Recording the
    /// list is what lets a set a walk actually produced be replayed into a
    /// fold, for any order, rather than approximated by one.
    pub restricted: Vec<usize>,
    /// The bound the region entered the heap under — its parent's exact one.
    /// A cheap bound is only worth something where it beats this.
    pub inherited: f64,
    /// `Σ_i max_{c ∈ A_i} log CAI(c)`, or `None` when some position admits no
    /// codon. The `None` is itself a result: emptiness costs the walk a solve
    /// and costs the ceiling one pass over the positions.
    pub cai_ceiling: Option<f64>,
    /// The three energies of [`MfeFloors`], for this region's parent.
    pub floors: MfeFloors,
    /// What the walk learned by solving it, if it ever did.
    pub probed: Probed,
    /// When the walk examined it, or `None` if it never did. Rows are opened in
    /// push order, so this is the only record of the order regions were
    /// *solved* in — which is what decides whether a bound could have fired on
    /// them against the incumbent held at that instant rather than the final
    /// one.
    pub closed_at: Option<ProbeInstant>,
    /// The same region solved with the region constraint dropped, beside what the
    /// exact solve cost. `None` unless the region was solved non-empty.
    pub relaxed: Option<RelaxedPrice>,
    /// The parent's max-marginal at the position this region excludes a codon
    /// at: the exact optimum of the parent with only that exclusion added,
    /// and what solving it cost. This region is a subset of that one, so the
    /// optimum is a floor under this region's own — the quantity an outside
    /// pass over the parent would read off without a solve per child. `None`
    /// unless probed, or where that set admits no encoding.
    pub marginal: Option<(f64, f64)>,
}

/// A region's optimum under the constraint-free energy model, and what each solve
/// cost.
///
/// Dropping the constraint only admits structures, so over the same codon sets the
/// constraint-free optimum can only sit at or below the constrained one. That is what
/// makes it a candidate floor; whether it is cheaper and tight enough is what
/// these fields measure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelaxedPrice {
    /// The constraint-free optimum's free energy, kcal/mol.
    pub mfe: f64,
    /// The constraint-free optimum's adaptation, `log CAI`.
    pub cai: f64,
    /// Seconds the constraint-free solve took.
    pub relaxed_s: f64,
    /// Seconds the exact, constrained solve took.
    pub exact_s: f64,
}

/// An instant of a probed walk, on both of its clocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeInstant {
    /// Heap pops so far — the same counter a trajectory records, so a row and
    /// an incumbent instant can be ordered against each other exactly.
    pub regions_popped: usize,
    /// Seconds of walk so far, *excluding* the probe's own floor solves, so the
    /// clock reads what an unprobed walk would have spent reaching this point.
    pub walk_s: f64,
}

impl BoundProbeRow {
    /// The cheap bounds this region's ceiling and floors compose to, or `None`
    /// where the region admits no encoding and no bound is needed.
    pub fn bounds(&self, lambda: f64) -> Option<CheapBounds> {
        self.cai_ceiling
            .map(|ceiling| self.floors.compose(lambda, ceiling))
    }
}

/// What a probed walk recorded, alongside the walk's own report.
#[derive(Debug, Clone, Default)]
pub struct BoundProbeReport {
    /// Seconds spent on [`Self::floor_solves`], kept so a probed walk's clock
    /// can be read net of what the probe itself cost.
    pub floor_s: f64,
    /// One row per region pushed below the root. The root has no parent and so
    /// no cheap bound; it is always solved.
    pub rows: Vec<BoundProbeRow>,
    /// Pure-energy solves the floors cost, over and above the walk's own. The
    /// price of the tightest floor, to be set against what it rejects.
    pub floor_solves: usize,
}

impl BoundProbeReport {
    /// Record a region entering the heap, returning the handle its solve is
    /// reported under.
    pub fn open(&mut self, row: BoundProbeRow) -> usize {
        self.rows.push(row);
        self.rows.len() - 1
    }

    /// Record what the region opened at `handle` turned out to be.
    pub fn close(&mut self, handle: usize, probed: Probed, at: ProbeInstant) {
        self.rows[handle].probed = probed;
        self.rows[handle].closed_at = Some(at);
    }

    /// Record the constraint-free price of the region opened at `handle`.
    pub fn price_relaxed(&mut self, handle: usize, price: RelaxedPrice) {
        self.rows[handle].relaxed = Some(price);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A floor and a ceiling that are both already whole dekacalories: the
    /// composition is then the objective's own arithmetic, so any drift in the
    /// rounding shows up as a plain inequality failure.
    #[test]
    fn a_cheap_bound_on_exact_dekacalories_is_the_objective_itself() {
        let lambda = 0.5;
        let exact = lambda * (-30.0f64 * DEKACAL) - (1.0 - lambda) * (-4.0f64 * DEKACAL);
        assert_eq!(cheap_bound(lambda, -30.0, -4.0), exact);
    }

    /// Both roundings go the way that keeps the bound under the value it
    /// bounds: the energy term down, the adaptation term up.
    ///
    /// Each quantity is chosen so that its two roundings actually differ. Where
    /// they coincide the composition equals the objective whatever it rounded
    /// with, and the case would pass without testing anything.
    #[test]
    fn a_cheap_bound_rounds_away_from_the_value_it_bounds() {
        let energy = -30.004;
        assert_ne!((energy * DEKACAL).floor(), (energy * DEKACAL).round());
        assert!(cheap_bound(1.0, energy, 0.0) < (energy * DEKACAL).round());

        let adaptation = -4.006;
        assert_ne!(
            (adaptation * DEKACAL).ceil(),
            (adaptation * DEKACAL).round()
        );
        assert!(cheap_bound(0.0, 0.0, adaptation) < -(adaptation * DEKACAL).round());
    }

    /// A term the objective gives no weight contributes nothing even when the
    /// quantity behind it is unbounded — the shape an absent floor takes at the
    /// objective's pure-adaptation end.
    #[test]
    fn an_unweighted_term_annihilates_an_absent_bound() {
        let bound = cheap_bound(0.0, f64::NEG_INFINITY, -4.0);
        assert!(bound.is_finite(), "got {bound}");
        assert_eq!(bound, -(-4.0f64 * DEKACAL).ceil());
    }

    /// A region admitting nothing needs no bound, and says so rather than
    /// composing one out of an absent ceiling.
    #[test]
    fn an_empty_region_composes_no_bound() {
        let row = BoundProbeRow {
            pinned: 0,
            part: 0,
            restricted: Vec::new(),
            inherited: -1.0,
            cai_ceiling: None,
            floors: MfeFloors {
                space: -30.0,
                parent: -29.0,
                parent_optimum: -28.0,
            },
            probed: Probed::Empty,
            closed_at: None,
            relaxed: None,
            marginal: None,
        };
        assert!(row.bounds(0.5).is_none());
    }

    /// The tighter floor composes to the tighter bound, which is the whole
    /// reason the expensive one is priced at all.
    #[test]
    fn a_tighter_floor_composes_to_a_tighter_bound() {
        let floors = MfeFloors {
            space: -30.0,
            parent: -29.0,
            parent_optimum: -28.0,
        };
        let bounds = floors.compose(0.5, -4.0);
        assert!(bounds.space < bounds.parent);
        assert!(bounds.parent < bounds.parent_optimum);
    }
}
