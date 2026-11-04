//! Counting what a box around one band of the front would remove from the
//! bounded fold, without removing anything.
//!
//! A band of the CAI/MFE front — every Pareto design between two front points
//! `A` (more stable, less adapted) and `B` (less stable, better adapted) — lies
//! inside the box whose corner is `Q = (E_B, log CAI_A)`: at or below the
//! energy ceiling `E_B`, and at or above the adaptation floor `log CAI_A`. The
//! energy ceiling is what the bounded fold already enforces with the exact cost
//! to go. This module measures what the rest of the box would add.
//!
//! # One family of cuts
//!
//! Every edge the box has is an instance of one test. At a weight `λ`, a
//! partial point `(mfe, cai)` at cell `X` can complete into the box only if
//!
//! ```text
//! λ·mfe − κ·cai + Out_λ(X) <= λ·E_B − κ·log CAI_A,   κ = 100(1 − λ)
//! ```
//!
//! where `Out_λ(X)` is the exact scalar cost to go at `λ` — the best completion
//! of the cell's key, on the scalar fold's own BRD scale. Every design in the
//! box satisfies the right-hand inequality at every `λ ∈ [0, 1]`, since both of
//! its coordinates are on the right side of `Q`; and no completion of the point
//! scores below `Out_λ(X)`. A point failing the test at any `λ` therefore has
//! no completion in the box.
//!
//! `λ = 1` is the energy ceiling itself, which the fold has already applied —
//! so a cut there removes nothing more and is carried as a control. `λ = 0` is
//! the adaptation floor alone. The interior weights are the coupled cuts: they
//! ask **one** completion to be good at both, where the ceiling and the floor
//! each consult their own best completion.
//!
//! # Why counting the finalized cells counts the cascade
//!
//! A descendant of a point is that point plus more of the molecule, and its
//! completion is a completion of the ancestor, so the test value can only rise
//! along a derivation: a point that fails has only failing descendants. A point
//! dominated by a failing one fails too. So a finalized cell's passing points
//! are exactly what a fold applying the cut would hold there, and the counts
//! here include what an applied cut would remove downstream, not only at the
//! cell it first fires in.
//!
//! # Abstentions
//!
//! A shared multi-loop opening pools openings from different cores under one
//! key and has no single cost to go, so it is counted and never cut. A cell
//! whose cost to go is infinite is not cut either, matching the energy credit,
//! which abstains there too.

use std::sync::Arc;

use rna_mfe::Energy;
use smallvec::SmallVec;

use crate::cost_to_go::{CellRef, Outside};
use crate::front::{Front, Point, Screen};
use crate::front_probe::{CellTable, ComposeSite};
use crate::tables::Tables;

/// Slack on every comparison, in BRD units. The cost to go and the point are
/// sums of `f64` log-weights taken in different orders, so equality is not
/// exact; the census errs toward keeping a point.
const TOLERANCE: f64 = 1e-6;

/// Extra slack, in accumulated log-CAI, on a cut tested on a compose's products
/// before their elimination rather than on its result after. The elimination
/// takes `cai` values within [`crate::front`]'s dominance tolerance (`1e-9`) as
/// equal, so a product failing a cut by less than that could still have
/// eliminated a twin that passes it. One failing by more than this cannot: the
/// twins it could eliminate, directly or through a chain of twins, sit within a
/// few tolerances of it, and distinct trade-offs differ by `~1e-3`.
const PREFILTER_CAI_SLACK: f64 = 1e-6;

/// One linear cut through the box's corner at a single weight `λ`.
pub struct BoxCut {
    lambda: f64,
    kk: f64,
    /// Shared, not copied: the walk cuts many slices and gaps at one pass.
    /// `None` is a cost to go of zero at every cell ([`Self::floor`]).
    outside: Option<Arc<Outside>>,
    threshold: f64,
    /// The scalar tables the cost to go was taken over, kept only when the
    /// census is asked to check that a cell's points and the scalar fold price
    /// its inside on one scale.
    inside: Option<Tables>,
}

impl BoxCut {
    /// The cut at `lambda` through `(ceiling, cai_floor)`, where `outside` is
    /// the λ-general cost to go over the exact scalar fold at `lambda` and
    /// `cai_floor` is on the fold's accumulated scale (the reported log-CAI less
    /// the lattice's normalization offsets).
    pub fn new(
        lambda: f64,
        outside: impl Into<Arc<Outside>>,
        ceiling: Energy,
        cai_floor: f64,
    ) -> Self {
        let kk = (1.0 - lambda) * 100.0;
        Self {
            lambda,
            kk,
            outside: Some(outside.into()),
            threshold: lambda * f64::from(ceiling.0) - kk * cai_floor,
            inside: None,
        }
    }

    /// The box's floor alone (`λ = 0`) through `cai_floor`, on the fold's
    /// accumulated scale, with a cost to go of zero at every cell: exact where
    /// no edge of the lattice carries a positive adaptation weight, since then
    /// no completion adds adaptation and the best adds none.
    pub fn floor(cai_floor: f64) -> Self {
        Self {
            lambda: 0.0,
            kk: 100.0,
            outside: None,
            threshold: -100.0 * cai_floor,
            inside: None,
        }
    }

    /// This cut's cost to go at `cell`.
    fn out_at(&self, cell: CellRef) -> f64 {
        self.outside.as_ref().map_or(0.0, |o| o.raw_at(cell))
    }

    /// Keep `tables` — the scalar fold the cost to go was taken over — so every
    /// counted cell is also checked against it (see [`BoxTally::inside_equal`]).
    pub fn with_inside(mut self, tables: Tables) -> Self {
        self.inside = Some(tables);
        self
    }

    /// The best of `front`'s points at this cut's weight, and the scalar
    /// fold's value for the same cell, when both exist.
    fn inside_pair<B>(&self, cell: CellRef, front: &Front<B>) -> Option<(f64, f64)> {
        let tables = self.inside.as_ref()?;
        let scalar = scalar_inside(tables, cell)?;
        let best = front
            .points()
            .iter()
            .map(|p| self.lambda * f64::from(p.mfe.0) - self.kk * p.cai)
            .fold(f64::INFINITY, f64::min);
        best.is_finite().then_some((best, scalar))
    }

    /// The weight this cut is taken at.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Whether no completion of `(mfe, cai)` at `cell` reaches the box.
    fn excludes(&self, cell: CellRef, mfe: Energy, cai: f64) -> bool {
        self.excludes_given(self.out_at(cell), mfe, cai)
    }

    /// [`Self::excludes`] with the cell's cost to go already looked up.
    fn excludes_given(&self, out: f64, mfe: Energy, cai: f64) -> bool {
        self.excludes_beyond(out, mfe, cai, 0.0)
    }

    /// [`Self::excludes_given`] with `slack` more room on the threshold.
    fn excludes_beyond(&self, out: f64, mfe: Energy, cai: f64, slack: f64) -> bool {
        out.is_finite() && self.value(out, mfe, cai) > self.threshold + TOLERANCE + slack
    }

    /// The quantity the cut compares against `threshold + TOLERANCE`.
    fn value(&self, out: f64, mfe: Energy, cai: f64) -> f64 {
        self.lambda * f64::from(mfe.0) - self.kk * cai + out
    }

    /// A point's own share of [`Self::value`].
    fn share<B>(&self, p: &Point<B>) -> f64 {
        self.lambda * f64::from(p.mfe.0) - self.kk * p.cai
    }

    /// The point of `front` with the lowest share.
    fn lowest<'f, B>(&self, front: &'f Front<B>) -> Option<&'f Point<B>> {
        front
            .points()
            .iter()
            .min_by(|x, y| self.share(x).total_cmp(&self.share(y)))
    }

    /// The product of `left ⊗ right`, shifted by `dmfe`, with the lowest value
    /// under this cut, as its `(mfe, cai)` summed as the compose sums them. The
    /// value is linear in a product's accumulators, so it pairs each side's
    /// lowest.
    fn best_product<B, C>(
        &self,
        left: &Front<B>,
        right: &Front<C>,
        dmfe: Energy,
    ) -> Option<(Energy, f64)> {
        let (a, b) = (self.lowest(left)?, self.lowest(right)?);
        Some((a.mfe + b.mfe + dmfe, a.cai + b.cai + 0.0))
    }
}

/// The cuts of one box: counted by a census, or applied by a box fold.
///
/// Applying them is exact for the same reason counting them is sound: a point
/// failing any cut has no completion in the box, and neither has anything
/// derived from it or dominated by it. The test reads one point and its cell's
/// key and nothing else, so it can run at every merge, on the incoming front
/// alone, in any order and under any worker count.
pub struct BoxCuts {
    cuts: Vec<BoxCut>,
    /// Read at each position with the points the fold has stored: `true`
    /// ends the fold there ([`Self::with_stop`]).
    stop: Option<std::sync::Arc<dyn Fn(usize) -> bool + Send + Sync>>,
}

/// Per-cell counts: the cell's size, then how many points survive each cut
/// alone, then how many survive all of them together.
pub(crate) type Survivors = SmallVec<[u32; 12]>;

/// A lower bound on each cut's lowest share over one cell's points.
pub(crate) type Lows = SmallVec<[f64; 4]>;

impl BoxCuts {
    /// A census over `cuts`, reported in the order given.
    pub fn new(cuts: Vec<BoxCut>) -> Self {
        Self { cuts, stop: None }
    }

    /// These cuts, with `stop` read by the fold at each position with the
    /// points it has stored; once it returns `true` the fold ends there and
    /// returns no point.
    pub fn with_stop(mut self, stop: std::sync::Arc<dyn Fn(usize) -> bool + Send + Sync>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Whether the fold stops at a position, with `stored` points stored.
    pub(crate) fn stops_at(&self, stored: usize) -> bool {
        self.stop.as_ref().is_some_and(|stop| stop(stored))
    }

    /// These cuts and `cut` after them.
    pub fn with(mut self, cut: BoxCut) -> Self {
        self.cuts.push(cut);
        self
    }

    /// The cuts, in reporting order.
    pub fn cuts(&self) -> &[BoxCut] {
        &self.cuts
    }

    /// How many of `front`'s points survive each cut and all of them, at
    /// `cell`. `None` is a cell with no single cost to go, where every cut
    /// abstains.
    pub(crate) fn survivors<B>(&self, cell: Option<CellRef>, front: &Front<B>) -> Survivors {
        let n = self.cuts.len();
        let mut out: Survivors = SmallVec::from_elem(0, n + 2);
        out[0] = front.len() as u32;
        let Some(cell) = cell else {
            out.iter_mut().for_each(|s| *s = front.len() as u32);
            return out;
        };
        for p in front.points() {
            let mut kept = true;
            for (i, cut) in self.cuts.iter().enumerate() {
                if cut.excludes(cell, p.mfe, p.cai) {
                    kept = false;
                } else {
                    out[i + 1] += 1;
                }
            }
            out[n + 1] += u32::from(kept);
        }
        out
    }

    /// Check a counted cell's points against each cut's scalar tables, where
    /// they were kept.
    pub(crate) fn check_inside<B>(&self, cell: CellRef, front: &Front<B>, tally: &mut BoxTally) {
        for cut in &self.cuts {
            let Some((best, scalar)) = cut.inside_pair(cell, front) else {
                continue;
            };
            tally.inside_checked += 1;
            if best < scalar - TOLERANCE {
                tally.inside_below += 1;
            } else if best <= scalar + TOLERANCE {
                tally.inside_equal += 1;
            }
        }
    }

    /// Discard every point of `front`, bound for `cell`, that some cut
    /// excludes, returning how many were removed. Not monotone along the
    /// staircase, so a filter pass rather than a truncation.
    pub(crate) fn retain_in_box<B>(&self, cell: CellRef, front: &mut Front<B>) -> usize {
        self.retain_at(&self.outs_at(cell), front)
    }

    /// [`Self::retain_in_box`] at a cell whose costs to go are `outs`, as
    /// [`Self::outs_at`] returns them.
    pub(crate) fn retain_at<B>(&self, outs: &[f64], front: &mut Front<B>) -> usize {
        let before = front.len();
        front.retain(|p| self.admits(outs, p.mfe, p.cai));
        before - front.len()
    }

    /// Each cut's cost to go at `cell`, looked up once for a run of tests there.
    pub(crate) fn outs_at(&self, cell: CellRef) -> SmallVec<[f64; 8]> {
        self.cuts.iter().map(|cut| cut.out_at(cell)).collect()
    }

    /// [`Self::admits`] for a product tested before its compose's elimination:
    /// each cut gets [`PREFILTER_CAI_SLACK`] more room, so a product this
    /// rejects is one the merge's own test rejects and one whose absence
    /// changes nothing else the elimination keeps.
    pub(crate) fn admits_before_elimination(&self, outs: &[f64], mfe: Energy, cai: f64) -> bool {
        !self
            .cuts
            .iter()
            .zip(outs)
            .any(|(cut, &out)| cut.excludes_beyond(out, mfe, cai, cut.kk * PREFILTER_CAI_SLACK))
    }

    /// The first cut's test, before a compose's elimination, at a cell whose
    /// costs to go are `outs`, as a [`Screen`] that
    /// [`Self::admits_before_elimination`] implies; `None` where that cut
    /// excludes nothing there.
    pub(crate) fn screen(&self, outs: &[f64]) -> Option<Screen> {
        let (cut, &out) = self.cuts.iter().zip(outs).next()?;
        let slack = cut.kk * PREFILTER_CAI_SLACK;
        out.is_finite()
            .then(|| Screen::new(cut.lambda, cut.kk, cut.threshold + TOLERANCE + slack - out))
    }

    /// Whether the staircase of `left ⊗ right`, shifted by `dmfe`, keeps a
    /// point every cut admits at a cell whose costs to go are `outs`: what a
    /// merge of the unfiltered compose would be left with after the cuts.
    /// `None` where that is not decided without composing, because the best
    /// candidate sits within [`PREFILTER_CAI_SLACK`] of a threshold.
    ///
    /// A product failing a cut by more than the slack has nothing on the
    /// staircase that passes it in its place, and one passing every cut by
    /// more than the slack has a staircase point that passes them too (see
    /// [`Self::admits_before_elimination`]). Each cut's best product pairs each
    /// side's best, so a cut no product can pass is found without a product;
    /// with one cut that settles it, and with several the products are scanned
    /// for one that passes them all.
    pub(crate) fn product_reaches<B, C>(
        &self,
        outs: &[f64],
        left: &Front<B>,
        right: &Front<C>,
        dmfe: Energy,
    ) -> Option<bool> {
        let mut single = None;
        for (cut, &out) in self.cuts.iter().zip(outs) {
            let (mfe, cai) = cut.best_product(left, right, dmfe)?;
            let slack = cut.kk * PREFILTER_CAI_SLACK;
            if cut.excludes_beyond(out, mfe, cai, slack) {
                return Some(false);
            }
            single = Some(!cut.excludes_beyond(out, mfe, cai, -slack));
        }
        match single {
            Some(passes) if self.cuts.len() == 1 => passes.then_some(true),
            _ => self.scan_reaches(outs, left, right, dmfe),
        }
    }

    /// [`Self::product_reaches`] over every product: `Some(true)` at the first
    /// passing every cut by more than the slack, `Some(false)` when none passes
    /// them all at all, `None` in between.
    fn scan_reaches<B, C>(
        &self,
        outs: &[f64],
        left: &Front<B>,
        right: &Front<C>,
        dmfe: Energy,
    ) -> Option<bool> {
        let passes = |mfe, cai, sign: f64| {
            self.cuts.iter().zip(outs).all(|(cut, &out)| {
                !cut.excludes_beyond(out, mfe, cai, sign * cut.kk * PREFILTER_CAI_SLACK)
            })
        };
        let mut near = false;
        for a in left.iter() {
            for b in right.iter() {
                let (mfe, cai) = (a.mfe + b.mfe + dmfe, a.cai + b.cai + 0.0);
                if passes(mfe, cai, -1.0) {
                    return Some(true);
                }
                near |= passes(mfe, cai, 0.0);
            }
        }
        (!near).then_some(false)
    }

    /// Lower `lows` to each cut's lowest share of `front`'s points, in cut
    /// order; a share is a point's part of the cut's value, its cost to go
    /// aside. Starting from infinity, then lowered by every front merged into
    /// one cell, `lows` stays at or below the cell's own lowest shares
    /// whatever the merges remove.
    pub(crate) fn lower_shares<B>(&self, lows: &mut Lows, front: &Front<B>) {
        if lows.is_empty() {
            lows.resize(self.cuts.len(), f64::INFINITY);
        }
        for (cut, low) in self.cuts.iter().zip(lows.iter_mut()) {
            for p in front.points() {
                *low = low.min(cut.share(p));
            }
        }
    }

    /// Lower `lows` to `from` moved by `dcai`: the bound on a front whose
    /// shares are bounded by `from`, once every point gains `dcai` of
    /// adaptation. An empty `from` bounds nothing and leaves `lows` as it is.
    pub(crate) fn lower_moved(&self, lows: &mut Lows, from: &[f64], dcai: f64) {
        if from.is_empty() {
            return;
        }
        if lows.is_empty() {
            lows.resize(self.cuts.len(), f64::INFINITY);
        }
        for ((cut, low), &f) in self.cuts.iter().zip(lows.iter_mut()).zip(from) {
            *low = low.min(f - cut.kk * dcai);
        }
    }

    /// Whether a cut removes every point of a front whose shares are at least
    /// `lows` ([`Self::lower_shares`]) once each is shifted by `dcai`, at a
    /// cell whose costs to go are `outs`. A shift moves every point's value
    /// by the same amount, so a front whose lowest share fails fails whole;
    /// it must fail by more than [`PREFILTER_CAI_SLACK`], which covers the
    /// rounding of the shift point by point.
    pub(crate) fn excludes_shifted(&self, outs: &[f64], lows: &[f64], dcai: f64) -> bool {
        self.excludes_moved(outs, lows, Energy::ZERO, dcai)
    }

    /// [`Self::excludes_shifted`] for a front whose points all move by
    /// `(dmfe, dcai)`, or by at least `dmfe` and at most `dcai`: a point's
    /// value only rises with its energy and falls with its adaptation, so
    /// bounds on the move bound every point's value from below.
    pub(crate) fn excludes_moved(
        &self,
        outs: &[f64],
        lows: &[f64],
        dmfe: Energy,
        dcai: f64,
    ) -> bool {
        self.cuts
            .iter()
            .zip(outs)
            .zip(lows)
            .any(|((cut, &out), &low)| {
                out.is_finite()
                    && low + cut.lambda * f64::from(dmfe.0) - cut.kk * dcai + out
                        > cut.threshold + TOLERANCE + cut.kk * PREFILTER_CAI_SLACK
            })
    }

    /// Whether no cut excludes `(mfe, cai)` at a cell whose costs to go are
    /// `outs`, as [`Self::outs_at`] returns them.
    pub(crate) fn admits(&self, outs: &[f64], mfe: Energy, cai: f64) -> bool {
        !self
            .cuts
            .iter()
            .zip(outs)
            .any(|(cut, &out)| cut.excludes_given(out, mfe, cai))
    }

    /// An empty tally with a slot per cut.
    pub(crate) fn tally(&self) -> BoxTally {
        BoxTally::new(self.cuts.len())
    }
}

/// What the cuts would have removed over one fold.
///
/// Every vector is indexed like [`Survivors`]: slot `0` is the whole, slots
/// `1..=cuts` each cut alone, and the last slot all cuts together.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxTally {
    /// Points in finalized cells, by table; per slot, how many survive.
    pub points: [Vec<u64>; 6],
    /// Finalized non-empty cells, by table; per slot, how many keep a point.
    pub cells: [Vec<u64>; 6],
    /// Front-by-front product volume `Σ |A|·|B|`, by compose site; per slot,
    /// the volume left once both operands are cut.
    pub volume: [Vec<u128>; 3],
    /// Survivors at the final exterior cell, whose front is the molecule's
    /// feasible staircase: per slot, what the cut keeps of the answer itself.
    pub root: Vec<u64>,
    /// Cells checked against a cut's scalar tables.
    pub inside_checked: u64,
    /// Of those, cells whose best point equals the scalar fold's value. Every
    /// one of them in an unbounded fold, where a cell holds every derivation.
    pub inside_equal: u64,
    /// Of those, cells whose best point beats the scalar fold's minimum —
    /// impossible on one scale, so any at all means the cuts compare a point
    /// and a cost to go priced differently.
    pub inside_below: u64,
}

impl BoxTally {
    fn new(cuts: usize) -> Self {
        let slots = cuts + 2;
        Self {
            points: std::array::from_fn(|_| vec![0; slots]),
            cells: std::array::from_fn(|_| vec![0; slots]),
            volume: std::array::from_fn(|_| vec![0; slots]),
            root: vec![0; slots],
            inside_checked: 0,
            inside_equal: 0,
            inside_below: 0,
        }
    }

    /// Record a finalized cell.
    pub(crate) fn observe_cell(&mut self, table: CellTable, s: &Survivors) {
        if s[0] == 0 {
            return;
        }
        let t = table.index();
        for (slot, &k) in s.iter().enumerate() {
            self.points[t][slot] += u64::from(k);
            self.cells[t][slot] += u64::from(k > 0);
        }
    }

    /// Record the final exterior cell.
    pub(crate) fn observe_root(&mut self, s: &Survivors) {
        self.root = s.iter().map(|&k| u64::from(k)).collect();
    }

    /// Record a product of two finalized cells.
    pub(crate) fn observe_product(&mut self, site: ComposeSite, a: &Survivors, b: &Survivors) {
        let v = &mut self.volume[site.index()];
        for (slot, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            v[slot] += u128::from(x) * u128::from(y);
        }
    }

    /// Record a product of a finalized cell with a front no cut reaches.
    pub(crate) fn observe_uncut_product(&mut self, site: ComposeSite, a: &Survivors, len: usize) {
        let v = &mut self.volume[site.index()];
        for (slot, &x) in a.iter().enumerate() {
            v[slot] += u128::from(x) * len as u128;
        }
    }

    /// Fold a worker's tally into this one. Every counter is a sum, so the
    /// result does not depend on how the work was divided.
    pub(crate) fn absorb(&mut self, other: &BoxTally) {
        for (mine, theirs) in self.points.iter_mut().zip(&other.points) {
            add(mine, theirs);
        }
        for (mine, theirs) in self.cells.iter_mut().zip(&other.cells) {
            add(mine, theirs);
        }
        self.inside_checked += other.inside_checked;
        self.inside_equal += other.inside_equal;
        self.inside_below += other.inside_below;
        for (mine, theirs) in self.volume.iter_mut().zip(&other.volume) {
            for (a, b) in mine.iter_mut().zip(theirs) {
                *a += b;
            }
        }
    }
}

fn add(mine: &mut [u64], theirs: &[u64]) {
    for (a, b) in mine.iter_mut().zip(theirs) {
        *a += b;
    }
}

/// The scalar fold's value at `cell`, on its own BRD scale.
fn scalar_inside(tables: &Tables, cell: CellRef) -> Option<f64> {
    let lambda = tables.lambda();
    let state = match cell {
        CellRef::C(n) => tables.c[tables.layout().ix(n)],
        CellRef::P(outer, key) => tables.p[tables.layout().ix(outer)].get(&key).copied(),
        CellRef::Multi(outer, key) => tables.multi[tables.layout().ix(outer)].get(&key).copied(),
        CellRef::M(outer, inner) => tables.m[tables.layout().ix(outer)].get(inner),
        CellRef::M2(outer, inner) => tables.m2[tables.layout().ix(outer)].get(inner),
    }?;
    Some(state.brd(lambda))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::Windows;
    use crate::cost_to_go::Outside;
    use crate::scan::scan;
    use crate::{
        census_front_bounded, design_front_bounded, dfa, Bounded, Bounds, BoxSpec, Ceiling,
        CodonUsage, Params, Prune, ScanConfig,
    };
    use derna_codon_dp::codon_table::aa_to_index;

    const LAMBDAS: [f64; 5] = [0.0, 0.3, 0.5, 0.7, 1.0];

    fn residues(protein: &str) -> Vec<usize> {
        protein.chars().map(|c| aa_to_index(c).unwrap()).collect()
    }

    /// In an unbounded fold every cell holds every derivation, so its best
    /// point at `λ` is the scalar fold's value there. That is what puts a
    /// point and a cost to go on one scale; a cut comparing a point against a
    /// cost to go priced under another accounting would be unsound.
    #[test]
    fn every_cells_best_point_is_the_scalar_folds_value_at_every_weight() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        for protein in ["MYGKIIFVLM", "MKTAYIAKQRQISF", "MCWHPKRDEQNY"] {
            let graph = dfa::build(&residues(protein), &usage);
            let cuts = LAMBDAS
                .iter()
                .map(|&lambda| {
                    let tables = scan(&params, &graph, lambda, ScanConfig::default());
                    let outside = Outside::compute_at_any_lambda(
                        &params,
                        &graph,
                        &tables,
                        ScanConfig::default(),
                    );
                    BoxCut::new(lambda, outside, Energy(0), -1.0).with_inside(tables)
                })
                .collect();
            let census = BoxCuts::new(cuts);
            let (_, _, tally) = crate::front_scan::fold_census(
                &params,
                &graph,
                Windows::UNBOUNDED,
                None,
                1,
                &census,
            );
            assert!(tally.inside_checked > 100, "{protein}: {tally:?}");
            assert_eq!(tally.inside_below, 0, "{protein}");
            assert_eq!(tally.inside_equal, tally.inside_checked, "{protein}");
        }
    }

    /// The census removes nothing, the floor alone keeps exactly the answer's
    /// in-box points at the root, the ceiling's own cut removes nothing
    /// anywhere, and some cut removes something below the root.
    #[test]
    fn the_census_counts_the_box_and_changes_nothing() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let protein = "MKTAYIAKQRQISFVKSH";
        let slack = Energy(300);
        let Bounded::Feasible(plain) = design_front_bounded(
            &params,
            protein,
            &usage,
            Prune::None,
            Bounds::with_ceiling(Ceiling::AboveOptimum(slack)).with_exact_credit(),
            1,
        )
        .unwrap() else {
            panic!("feasible");
        };
        assert!(plain.front.len() >= 3, "{}", plain.front.len());
        let floor = plain.front[plain.front.len() / 2].cai_log;
        let spec = BoxSpec {
            cai_floor_log: floor,
            lambdas: LAMBDAS.to_vec(),
        };
        let census = census_front_bounded(&params, protein, &usage, slack, 1, &spec).unwrap();

        let seqs =
            |f: &[crate::FrontPoint]| f.iter().map(|p| p.sequence.clone()).collect::<Vec<_>>();
        assert_eq!(seqs(&census.front), seqs(&plain.front));

        let in_box = plain.front.iter().filter(|p| p.cai_log >= floor).count() as u64;
        let tally = &census.tally;
        assert_eq!(tally.root[0], plain.front.len() as u64);
        assert_eq!(tally.root[1], in_box, "the floor alone at the root");
        assert!(in_box < tally.root[0], "the floor binds at the root");
        let control = LAMBDAS.len();
        for t in 0..6 {
            assert_eq!(tally.points[t][control], tally.points[t][0], "table {t}");
        }
        let removed: u64 = (0..6)
            .map(|t| tally.points[t][0] - tally.points[t][control + 1])
            .sum();
        assert!(
            removed > tally.root[0] - in_box,
            "cuts reach below the root"
        );
    }
}
