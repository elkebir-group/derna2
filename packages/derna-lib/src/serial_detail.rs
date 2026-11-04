//! Where the front fold's serial pair phase goes, measured below its phase
//! clock.
//!
//! [`crate::FoldClock::pair_serial_s`] charges a closed pair's exterior attach,
//! its branch open and the bifurcation `M ⊗ P → M2` together. This splits the
//! seconds between them, splits the bifurcation into its compose (product,
//! sort, eliminate) and its merge (the box's cuts, the merge and the window's
//! truncation), and counts the bifurcation's points at each of those stages, so
//! the product volume the merge discards is a count rather than an inference
//! from time.
//!
//! Beside it, the internal-loop corner's compose in the fanned-out
//! single-branch phase is counted the same way, so the two sites' waste can be
//! compared.
//!
//! The split costs a clock read per compose and a pass over every product, so
//! it is off unless the environment sets `DERNA_SERIAL_DETAIL`; the fold's
//! answer does not depend on it either way.

use rna_mfe::Energy;

use crate::box_census::BoxCuts;
use crate::front::Front;

/// The serial pair phase, split, and the bifurcation's points stage by stage,
/// summed over a fold. All zero unless the split was enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SerialDetail {
    /// Seconds in the exterior attach `C ⊗ P`, its merge included.
    pub exterior_s: f64,
    /// Seconds opening a branch, `M = P`, its merge included.
    pub open_s: f64,
    /// Seconds in the bifurcation, everything below included, the counting
    /// excepted.
    pub bifurcation_s: f64,
    /// Of `bifurcation_s`: the compose — every product, the sort and the
    /// elimination.
    pub compose_s: f64,
    /// Of `bifurcation_s`: the merge into `M2`, the cuts and the truncation
    /// included.
    pub merge_s: f64,
    /// Seconds spent counting, excluded from the rest.
    pub counting_s: f64,
    /// Bifurcation composes, one per `(closed pair, M cell)`.
    pub composes: u64,
    /// Positions whose bifurcations were composed fanned out across workers.
    pub fanned: u64,
    /// `Σ |M|·|P|`: every product of the composes' operands.
    pub products: u64,
    /// Of those, the products the composes generated; the rest were skipped
    /// as ones the merge would discard.
    pub generated: u64,
    /// Points the composes return after elimination.
    pub after_eliminate: u64,
    /// Of those, the points the box's cuts keep (all of them without a box).
    pub after_cuts: u64,
    /// Points of each arrival still in its cell after the merge and the
    /// truncation to the window and the exact cutoff.
    pub after_merge: u64,
    /// Arrivals the filter emptied whose unfiltered merge would have inserted
    /// an absent cell and emptied it, replayed so the column's hash table
    /// matches.
    pub replayed: u64,
    /// Products at or below the target cell's exact cutoff (all of them where
    /// the fold carries no exact credit).
    pub within_cutoff: u64,
    /// Of those, the products every cut of the box admits.
    pub within_box: u64,
    /// `Σ |C|·|P|` over the exterior attaches.
    pub exterior_products: u64,
    /// Of those, the products at or below the exterior cell's exact cutoff.
    pub exterior_within_cutoff: u64,
    /// The internal-loop corner's composes (`P ⊗ corner → P`), counted by the
    /// fanned-out workers.
    pub corner: ComposeCount,
}

/// One compose site's products, and those its merge could keep.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ComposeCount {
    /// `Σ |A|·|B|`.
    pub products: u64,
    /// Of those, the products at or below the target cell's exact cutoff.
    pub within_cutoff: u64,
    /// Of those, the products every cut of the box admits.
    pub within_box: u64,
}

impl ComposeCount {
    /// Add another count, a worker's into the fold's.
    pub(crate) fn absorb(&mut self, other: &ComposeCount) {
        self.products += other.products;
        self.within_cutoff += other.within_cutoff;
        self.within_box += other.within_box;
    }
}

/// Whether the environment asks for the split.
pub(crate) fn enabled() -> bool {
    std::env::var_os("DERNA_SERIAL_DETAIL").is_some()
}

/// How many products of `left ⊗ right`, shifted by `dmfe`, have energy at most
/// `cutoff`. Both fronts are `mfe`-ascending, so for each left point the right
/// points that qualify are a prefix.
pub(crate) fn within_cutoff<B, C>(
    left: &Front<B>,
    right: &Front<C>,
    dmfe: Energy,
    cutoff: Option<Energy>,
) -> u64 {
    let Some(cutoff) = cutoff else {
        return (left.len() * right.len()) as u64;
    };
    left.iter()
        .map(|a| prefix_within(right, a.mfe + dmfe, cutoff) as u64)
        .sum()
}

/// How many products of `left ⊗ right`, shifted by `dmfe`, have energy at most
/// `cutoff` and are admitted by every cut of `cuts` at a cell whose costs to go
/// are `outs`. The cuts are tested on the product's accumulators summed as the
/// compose sums them.
pub(crate) fn within_box<B, C>(
    left: &Front<B>,
    right: &Front<C>,
    dmfe: Energy,
    cutoff: Option<Energy>,
    cuts: &BoxCuts,
    outs: &[f64],
) -> u64 {
    let mut kept = 0;
    for a in left.iter() {
        let base = a.mfe + dmfe;
        let end = cutoff.map_or(right.len(), |c| prefix_within(right, base, c));
        kept += right.points()[..end]
            .iter()
            .filter(|b| cuts.admits(outs, base + b.mfe, a.cai + b.cai + 0.0))
            .count() as u64;
    }
    kept
}

/// The length of `right`'s prefix whose points, shifted by `base`, have energy
/// at most `cutoff`.
fn prefix_within<C>(right: &Front<C>, base: Energy, cutoff: Energy) -> usize {
    right
        .points()
        .partition_point(|b| (base + b.mfe).0 <= cutoff.0)
}
