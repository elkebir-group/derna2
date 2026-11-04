//! How much of a restricted fold's CKY table a less-restricted fold already
//! holds — measured, before any warm start is written.
//!
//! A branch-and-bound over regions of the design space pays a whole-mRNA fold
//! per region, and a child region restricts only some of its parent's
//! positions — one in the shallowest case, a handful typically. The tables are
//! keyed on DFA nodes rather than on the codon lattice, so the two folds
//! address the same cells and can be compared directly. This module folds a
//! parent and a child and reports, cell by cell, how many of the child's the
//! parent already computed to the same value.
//!
//! How many positions a child actually restricts is a property of the
//! partition, not of this module, and it is the axis every share here is read
//! against, measured from a walk's own probe rows.
//!
//! **It never reuses anything.** What it answers is whether a warm start is
//! worth building: the share of a child's table a parent already holds is the
//! ceiling on what any warm start could skip.
//!
//! # Why a cell can be reused at all
//!
//! Restricting position `q` removes edges inside that codon's three nucleotide
//! slots and changes nothing anywhere else. A CKY cell accumulates over its own
//! span alone, so a cell whose span does not reach into codon `q` is a value the
//! two folds derive from identical sub-lattices — identical, not merely close.
//! Cells whose span *covers* `q` may still agree, when the parent's optimal
//! derivation never spelled the removed codon, and that surplus is reported
//! separately because it is the part no static analysis could have predicted.
//!
//! The span is **rounded out to codon boundaries** before that test. A cell
//! whose boundary falls inside a codon carries that codon's broken-codon
//! adaptation — the partial-codon closed form the multi-loop split and the
//! interior reductions credit — so its value depends on a unit reaching up to
//! two nucleotides outside the span itself. Testing the raw span instead calls
//! a handful of such cells clean and they then disagree, which is how the
//! rounding was found: the equality `clean == clean_identical` is asserted, so
//! a predicate that is even slightly too permissive fails rather than quietly
//! inflating the ceiling.
//!
//! # The normalization that makes equality the wrong test
//!
//! A restricted codon unit is re-normalized so its best *available* codon again
//! scores zero ([`crate::dfa`]'s `renormalize`), which shifts every CAI weight
//! on that codon's edges by a constant. A child cell whose derivation crosses a
//! restricted codon therefore carries a different `cai` from the parent's *even
//! when the derivation is identical*. That shift is recoverable — the constant
//! is recorded — so the two are counted apart: [`TableCounts::dirty_identical`]
//! is agreement on both accumulators, [`TableCounts::dirty_mfe_only`] is
//! agreement on the energy with the adaptation moved, which is the
//! normalization's signature and not a changed fold.
//!
//! Whether that class is *also* reusable turns on whether the shift is one
//! constant. An identical derivation crossing one re-normalized codon is off by
//! exactly that codon's constant, so a warm start could copy the cell and add it
//! back; a genuinely different derivation that happens to tie on integer
//! dekacalories is off by whatever it is off by. The two are told apart by
//! counting the *distinct* shifts ([`ReuseReport::offset_census`]): a handful
//! means the constant, a spread means coincidence.
//!
//! # Cells are not work
//!
//! A long span costs more to close than a short one — the bifurcation over it
//! scans every split point — so a share of *cells* overstates a share of
//! *work* whenever the reusable cells are the short ones, which is exactly the
//! shape a single restricted position produces. Every count therefore carries a
//! span-length-weighted twin, and the two are reported side by side.

use rna_mfe::Energy;

use crate::dfa::{self, Dfa};
use crate::hash::FxHashMap;
use crate::region::{PairConstraint, RegionConstraints};
use crate::scan;
use crate::state::PairKey;
use crate::{BeamError, CodonUsage, Params, ScanConfig};

/// One CKY table's agreement between a parent fold and a child fold.
///
/// The three classes partition the child's cells: a cell is `clean` when its
/// span avoids every restricted codon, `dirty` when it covers one, and each is
/// then split by whether the parent holds the same value. `absent` counts the
/// child cells the parent's table has no entry for at all — the child's lattice
/// is a subset of the parent's, so this should be empty, and a non-zero count
/// means the two addressings have diverged rather than that a cell was gained.
#[derive(Debug, Clone, Default)]
pub struct TableCounts {
    /// Which table these counts are over.
    pub name: &'static str,
    /// Cells the parent's fold materialized.
    pub parent_cells: usize,
    /// Cells the child's fold materializes.
    pub child_cells: usize,
    /// Child cells whose span avoids every restricted codon.
    pub clean: usize,
    /// Of those, the ones the parent holds with a bit-identical `(mfe, cai)`.
    pub clean_identical: usize,
    /// Child cells whose span covers a restricted codon.
    pub dirty: usize,
    /// Of those, the ones the parent holds with a bit-identical `(mfe, cai)`.
    pub dirty_identical: usize,
    /// Of those, the ones agreeing on `mfe` with `cai` moved — the
    /// re-normalization's signature rather than a changed derivation.
    pub dirty_mfe_only: usize,
    /// Child cells with no parent entry at their coordinate.
    ///
    /// The child's lattice restricts the parent's, so a child cell ought always
    /// to have a parent counterpart — except where a restriction dropped a
    /// whole prefix group and renumbered the parallel lanes inside that codon,
    /// which renames the nodes such a cell is keyed on. Those cells are dirty
    /// by construction, which is why the count is split: `absent_clean` must be
    /// zero, and a non-zero one means the two addressings have diverged
    /// somewhere the clean class relies on.
    pub absent: usize,
    /// Of those, the ones whose span is clear of every restriction.
    pub absent_clean: usize,
    /// Total span length over the child's cells, in nucleotides.
    pub span_nt: u64,
    /// Span length over the `clean` cells.
    pub clean_span_nt: u64,
    /// Span length over the `clean_identical` cells.
    pub clean_identical_span_nt: u64,
    /// Span length over the `dirty_identical` cells.
    pub dirty_identical_span_nt: u64,
    /// Span length over the `dirty_mfe_only` cells. Carried because that class
    /// swings from 0.0 to 39.7% of a child's cells across the measured rows, so
    /// a count alone cannot say whether it is worth anything — and the share
    /// that feeds a speed-up is the span-weighted one.
    pub dirty_mfe_only_span_nt: u64,
}

impl TableCounts {
    /// Child cells the parent already holds to the same value, clean or not.
    pub fn identical(&self) -> usize {
        self.clean_identical + self.dirty_identical
    }

    /// Span length over every cell the parent already holds to the same value.
    pub fn identical_span_nt(&self) -> u64 {
        self.clean_identical_span_nt + self.dirty_identical_span_nt
    }

    /// The share of the child's cells a warm start could take unchanged.
    pub fn cell_share(&self) -> f64 {
        share(self.identical(), self.child_cells)
    }

    /// The same share weighted by span length — what the cells are worth, as
    /// against how many there are.
    pub fn work_share(&self) -> f64 {
        if self.span_nt == 0 {
            return 0.0;
        }
        self.identical_span_nt() as f64 / self.span_nt as f64
    }

    fn add(&mut self, other: &Self) {
        self.parent_cells += other.parent_cells;
        self.child_cells += other.child_cells;
        self.clean += other.clean;
        self.clean_identical += other.clean_identical;
        self.dirty += other.dirty;
        self.dirty_identical += other.dirty_identical;
        self.dirty_mfe_only += other.dirty_mfe_only;
        self.absent += other.absent;
        self.absent_clean += other.absent_clean;
        self.span_nt += other.span_nt;
        self.clean_span_nt += other.clean_span_nt;
        self.clean_identical_span_nt += other.clean_identical_span_nt;
        self.dirty_identical_span_nt += other.dirty_identical_span_nt;
        self.dirty_mfe_only_span_nt += other.dirty_mfe_only_span_nt;
    }
}

/// Whether two adaptation accumulators are the same value — equal, or within
/// [`CAI_ULPS`] of each other.
fn same_cai(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= CAI_ULPS * f64::EPSILON * a.abs().max(b.abs())
}

fn share(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// What one parent/child fold pair agrees on.
#[derive(Debug, Clone)]
pub struct ReuseReport {
    /// Codon positions the child restricts and the parent does not.
    pub restricted: Vec<usize>,
    /// Per-table agreement, then the five tables summed.
    pub tables: Vec<TableCounts>,
    /// The five tables summed.
    pub total: TableCounts,
    /// Node coordinates at which the child's out-edges are **not** a subset of
    /// the parent's — the addressings have diverged and the cells keyed on them
    /// are not comparable. Restricting a codon can renumber the parallel lanes
    /// inside it, so this is expected to be non-zero over restricted codons and
    /// must be zero everywhere else; the clean class never touches such a node.
    pub lane_divergent_nodes: usize,
    /// Of those, the ones lying outside every restricted codon. A non-zero
    /// count here invalidates the clean class and the report with it.
    pub lane_divergent_outside: usize,
    /// Up to [`WITNESSES`] cells that the span test called clear of every
    /// restriction and that disagree anyway: table, span, and the two
    /// accumulators. Empty is the expected state — the span test is supposed to
    /// be exact — so a non-empty list is a defect in the test rather than a
    /// property of the folds, and it names the cell to go and look at.
    pub clean_witnesses: Vec<CleanWitness>,
    /// The distinct adaptation shifts across the cells agreeing on energy and
    /// not on adaptation, commonest first — the test for whether that class is
    /// one re-normalization constant or a pile of coincidences.
    pub offset_census: Vec<(f64, usize)>,
    /// How many distinct shifts there were in all, of which
    /// [`Self::offset_census`] names the commonest.
    pub offset_distinct: usize,
}

impl ReuseReport {
    /// Whether the two folds address the same cells outside the restricted
    /// codons — the precondition the clean class rests on.
    pub fn comparable(&self) -> bool {
        self.lane_divergent_outside == 0
    }
}

/// One cell the span test called clear of every restriction and that disagreed.
#[derive(Debug, Clone, Copy)]
pub struct CleanWitness {
    /// Which table it lives in.
    pub table: &'static str,
    /// The cell's span, `[lo, hi)`.
    pub lo: u32,
    pub hi: u32,
    /// The parent's accumulators, then the child's.
    pub parent: (i32, f64),
    pub child: (i32, f64),
}

/// How many disagreeing clean cells a report names before it stops collecting.
const WITNESSES: usize = 8;

/// How many units in the last place two adaptation accumulators may differ by
/// and still count as the same value.
///
/// Not a fudge factor. Two folds that derive a cell identically can still sum
/// its weights in a different order, because an equal-BRD tie somewhere
/// upstream breaks the other way and the surviving derivation adds the same
/// terms in a different sequence. Measured, that is **one** ulp — four cells in
/// 222 527 at n = 55, relative error 1.6e-16, with the energy bit-identical —
/// and a warm start copying the parent's value is unaffected, since the DP's
/// answer is not defined below that. Cells outside this window are counted as
/// disagreeing and named in [`ReuseReport::clean_witnesses`].
const CAI_ULPS: f64 = 8.0;

/// How many of the commonest adaptation shifts a report carries. Enough to see
/// whether one dominates without carrying a histogram of every tie.
const OFFSETS_REPORTED: usize = 4;

/// Adaptation shifts are compared at this resolution. An identical derivation
/// sums the same weights in a different order, so its shift reproduces to
/// rounding rather than to the bit — and a shift that is really one constant
/// must not be split into thousands by the last few ulps.
const OFFSET_RESOLUTION: f64 = 1e9;

/// The inputs a pair of folds share: everything but the codon lattice.
#[derive(Clone, Copy)]
pub struct ReuseInputs<'a> {
    pub params: &'a Params,
    pub aa: &'a [usize],
    pub usage: &'a CodonUsage,
    pub lambda: f64,
    /// Given 5′ UTR, as the design entry points spell one.
    pub utr5: &'a str,
    /// Given 3′ UTR.
    pub utr3: &'a str,
    pub constraints: RegionConstraints,
    pub config: ScanConfig,
}

/// Fold `parent` and `child` over the same instance and report what the child's
/// table inherits.
///
/// `child` must restrict `parent` — every position's codon set a subset of the
/// parent's — since that is the only relation a branch-and-bound produces and
/// the only one under which a parent cell is a candidate value at all.
/// `restricted` names the positions where the two differ; it is derived here
/// rather than taken, so it cannot fall out of step with the lattices.
///
/// Returns `Ok(None)` when either lattice leaves a position empty.
pub fn table_reuse(
    inputs: &ReuseInputs<'_>,
    parent: &[Vec<[u8; 3]>],
    child: &[Vec<[u8; 3]>],
) -> Result<Option<ReuseReport>, BeamError> {
    inputs.config.validate()?;
    let restricted: Vec<usize> = (0..parent.len())
        .filter(|&i| parent[i].len() != child[i].len())
        .collect();
    let (utr5, utr3) = (
        crate::parse_utr(inputs.utr5)?,
        crate::parse_utr(inputs.utr3)?,
    );
    let (Some(parent_dfa), Some(child_dfa)) = (
        dfa::build_restricted_with_utr(inputs.aa, inputs.usage, parent, &utr5, &utr3),
        dfa::build_restricted_with_utr(inputs.aa, inputs.usage, child, &utr5, &utr3),
    ) else {
        return Ok(None);
    };
    let parent_tables = scan::scan_with_region(
        inputs.params,
        &parent_dfa,
        inputs.lambda,
        inputs.config,
        PairConstraint::new(&inputs.constraints, &parent_dfa),
    );
    let child_tables = scan::scan_with_region(
        inputs.params,
        &child_dfa,
        inputs.lambda,
        inputs.config,
        PairConstraint::new(&inputs.constraints, &child_dfa),
    );
    let dirt = Dirt::new(&parent_dfa, &restricted);
    let (lane_divergent_nodes, lane_divergent_outside) =
        lane_divergence(&parent_dfa, &child_dfa, &dirt);
    let mut offsets = Offsets::default();
    let tables = vec![
        pair_table(
            "P",
            parent_tables.layout(),
            &parent_tables.p,
            &child_tables.p,
            &dirt,
            &mut offsets,
        ),
        pair_table(
            "Multi",
            parent_tables.layout(),
            &parent_tables.multi,
            &child_tables.multi,
            &dirt,
            &mut offsets,
        ),
        dense_table(
            "M",
            &parent_tables,
            &child_tables,
            Which::M,
            &dirt,
            &mut offsets,
        ),
        dense_table(
            "M2",
            &parent_tables,
            &child_tables,
            Which::M2,
            &dirt,
            &mut offsets,
        ),
        exterior_table(
            parent_tables.layout(),
            &parent_tables.c,
            &child_tables.c,
            &dirt,
            &mut offsets,
        ),
    ];
    let mut total = TableCounts {
        name: "all",
        ..TableCounts::default()
    };
    for t in &tables {
        total.add(t);
    }
    Ok(Some(ReuseReport {
        restricted,
        tables,
        total,
        lane_divergent_nodes,
        lane_divergent_outside,
        clean_witnesses: offsets.witnesses.clone(),
        offset_census: offsets.census(),
        offset_distinct: offsets.distinct(),
    }))
}

/// What a pass over the tables accumulates besides its counters: the adaptation
/// shifts seen on cells that agreed on energy, and any clean cell that
/// disagreed.
#[derive(Default)]
struct Offsets {
    seen: FxHashMap<i64, usize>,
    witnesses: Vec<CleanWitness>,
    /// Set for the duration of one table's pass, so a witness knows its table.
    table: &'static str,
}

impl Offsets {
    /// Name a clean cell that disagreed, up to [`WITNESSES`] of them.
    fn witness(&mut self, lo: u32, hi: u32, parent: (Energy, f64), child: (Energy, f64)) {
        if self.witnesses.len() < WITNESSES {
            self.witnesses.push(CleanWitness {
                table: self.table,
                lo,
                hi,
                parent: (parent.0 .0, parent.1),
                child: (child.0 .0, child.1),
            });
        }
    }

    fn record(&mut self, delta: f64) {
        let bucket = (delta * OFFSET_RESOLUTION).round() as i64;
        *self.seen.entry(bucket).or_default() += 1;
    }

    /// The commonest shifts, with how many cells carried each.
    fn census(&self) -> Vec<(f64, usize)> {
        let mut rows: Vec<(f64, usize)> = self
            .seen
            .iter()
            .map(|(&b, &n)| (b as f64 / OFFSET_RESOLUTION, n))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.total_cmp(&b.0)));
        rows.truncate(OFFSETS_REPORTED);
        rows
    }

    fn distinct(&self) -> usize {
        self.seen.len()
    }
}

/// The nucleotide slots a restriction reaches into, and the test for whether a
/// cell's value can depend on them.
struct Dirt<'a> {
    dfa: &'a Dfa,
    /// One half-open nucleotide interval per restricted codon.
    windows: Vec<(u32, u32)>,
}

impl<'a> Dirt<'a> {
    fn new(dfa: &'a Dfa, restricted: &[usize]) -> Self {
        let cds_start = dfa.cds_start();
        Self {
            dfa,
            windows: restricted
                .iter()
                .map(|&q| {
                    let lo = cds_start + 3 * q as u32;
                    (lo, lo + 3)
                })
                .collect(),
        }
    }

    /// Whether a cell spanning `[lo, hi)` can depend on a restricted codon.
    ///
    /// The span is widened to the codon boundaries around it first: a boundary
    /// falling mid-codon makes the cell carry that codon's broken-codon
    /// adaptation, so its value reaches up to two nucleotides beyond the span.
    fn touches(&self, lo: u32, hi: u32) -> bool {
        let lo = self.dfa.codon_start(lo);
        let hi = self.dfa.next_codon_boundary(hi);
        self.windows.iter().any(|&(a, b)| lo < b && hi > a)
    }

    /// Whether a single nucleotide position lies in a restricted codon.
    fn covers(&self, pos: u32) -> bool {
        self.windows.iter().any(|&(a, b)| pos >= a && pos < b)
    }
}

/// Node coordinates whose child out-edges are not a subset of the parent's,
/// and how many of those lie outside every restricted codon.
///
/// Restricting a codon drops edges, and may drop a whole prefix group and so
/// renumber the lanes of the groups after it. That renames nodes *inside* the
/// restricted codon, which is harmless — every cell keyed on such a node is
/// dirty by construction — but it would be fatal anywhere else, so the two
/// cases are counted apart.
fn lane_divergence(parent: &Dfa, child: &Dfa, dirt: &Dirt) -> (usize, usize) {
    let mut divergent = 0;
    let mut outside = 0;
    for pos in 0..=parent.seq_len {
        for &node in child.nodes_at(pos) {
            let mut child_nucs: Vec<u8> = child.right(node).iter().map(|e| e.nuc as u8).collect();
            let mut parent_nucs: Vec<u8> = if parent.nodes_at(pos).contains(&node) {
                parent.right(node).iter().map(|e| e.nuc as u8).collect()
            } else {
                Vec::new()
            };
            child_nucs.sort_unstable();
            child_nucs.dedup();
            parent_nucs.sort_unstable();
            parent_nucs.dedup();
            if child_nucs.iter().all(|n| parent_nucs.contains(n)) {
                continue;
            }
            divergent += 1;
            // A node's own position is the slot its out-edge spells, so a node
            // renumbered by a restriction sits inside that restricted codon.
            if !dirt.covers(pos) {
                outside += 1;
            }
        }
    }
    (divergent, outside)
}

/// Agreement over a closed-pair table (`P` / `Multi`), whose cells span
/// `[key.open.pos, outer.pos)`.
fn pair_table(
    name: &'static str,
    layout: crate::dfa::NodeLayout,
    parent: &[FxHashMap<PairKey, crate::state::State>],
    child: &[FxHashMap<PairKey, crate::state::State>],
    dirt: &Dirt,
    offsets: &mut Offsets,
) -> TableCounts {
    offsets.table = name;
    let mut counts = TableCounts {
        name,
        ..TableCounts::default()
    };
    for (index, column) in child.iter().enumerate() {
        counts.parent_cells += parent[index].len();
        let outer = layout.node_of(index).pos;
        for (key, cell) in column {
            let found = parent[index].get(key).map(|p| (p.mfe, p.cai));
            counts.record(
                key.open.pos,
                outer,
                (cell.mfe, cell.cai),
                found,
                dirt,
                offsets,
            );
        }
    }
    counts
}

/// Which of the two multi-branch tables a read is over.
#[derive(Clone, Copy)]
enum Which {
    M,
    M2,
}

/// Agreement over a multi-branch table (`M` / `M2`), whose cells span
/// `[inner.pos, outer.pos)`.
fn dense_table(
    name: &'static str,
    parent: &crate::tables::Tables,
    child: &crate::tables::Tables,
    which: Which,
    dirt: &Dirt,
    offsets: &mut Offsets,
) -> TableCounts {
    let (parent_cols, child_cols) = match which {
        Which::M => (&parent.m, &child.m),
        Which::M2 => (&parent.m2, &child.m2),
    };
    offsets.table = name;
    let mut counts = TableCounts {
        name,
        ..TableCounts::default()
    };
    for (index, parent_col) in parent_cols.iter().enumerate().take(child_cols.len()) {
        counts.parent_cells += parent_col.len();
        let node = parent.layout().node_of(index);
        let outer = node.pos;
        let states = match which {
            Which::M => child.m_states(node),
            Which::M2 => child.m2_states(node),
        };
        for (inner, cell) in states {
            let found = parent_col.get(inner).map(|p| (p.mfe, p.cai));
            counts.record(inner.pos, outer, (cell.mfe, cell.cai), found, dirt, offsets);
        }
    }
    counts
}

/// Agreement over the exterior table, whose cells span the whole prefix
/// `[0, node.pos)` — so every one of them past the first restricted codon is
/// dirty by construction.
fn exterior_table(
    layout: crate::dfa::NodeLayout,
    parent: &[Option<crate::state::State>],
    child: &[Option<crate::state::State>],
    dirt: &Dirt,
    offsets: &mut Offsets,
) -> TableCounts {
    offsets.table = "C";
    let mut counts = TableCounts {
        name: "C",
        ..TableCounts::default()
    };
    for (index, cell) in child.iter().enumerate() {
        counts.parent_cells += usize::from(parent[index].is_some());
        let Some(cell) = cell else { continue };
        let outer = layout.node_of(index).pos;
        let found = parent[index].map(|p| (p.mfe, p.cai));
        counts.record(0, outer, (cell.mfe, cell.cai), found, dirt, offsets);
    }
    counts
}

impl TableCounts {
    /// Classify one child cell spanning `[lo, hi)` against what the parent holds
    /// at the same coordinate.
    fn record(
        &mut self,
        lo: u32,
        hi: u32,
        cell: (Energy, f64),
        parent: Option<(Energy, f64)>,
        dirt: &Dirt,
        offsets: &mut Offsets,
    ) {
        let span = u64::from(hi.saturating_sub(lo));
        self.child_cells += 1;
        self.span_nt += span;
        let clean = !dirt.touches(lo, hi);
        if clean {
            self.clean += 1;
            self.clean_span_nt += span;
        } else {
            self.dirty += 1;
        }
        let Some(parent) = parent else {
            self.absent += 1;
            self.absent_clean += usize::from(clean);
            return;
        };
        // Energy is an integer dekacalorie and compares exactly. Adaptation is
        // a sum of logs and compares to within [`CAI_ULPS`], for the reason
        // recorded there.
        let same_mfe = parent.0 == cell.0;
        let same = same_mfe && same_cai(parent.1, cell.1);
        if clean && !same {
            offsets.witness(lo, hi, parent, cell);
        }
        match (clean, same) {
            (true, true) => {
                self.clean_identical += 1;
                self.clean_identical_span_nt += span;
            }
            (false, true) => {
                self.dirty_identical += 1;
                self.dirty_identical_span_nt += span;
            }
            (false, false) if same_mfe => {
                self.dirty_mfe_only += 1;
                self.dirty_mfe_only_span_nt += span;
                offsets.record(cell.1 - parent.1);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{same_cai, CAI_ULPS};

    /// The adaptation tolerance accepts the accumulation noise it exists for
    /// and rejects anything a fold could act on.
    ///
    /// Both halves are asserted. A tolerance checked only from below would pass
    /// on one that accepted everything, and that one would report every cell
    /// reusable — the failure mode this whole measurement is most exposed to,
    /// since its output is a share.
    #[test]
    fn the_adaptation_tolerance_accepts_noise_and_nothing_larger() {
        let x = -10.903_407_775_f64;
        let one_ulp = f64::from_bits(x.to_bits() - 1);
        assert_ne!(x, one_ulp, "the perturbation must actually move the value");
        assert!(same_cai(x, one_ulp), "one ulp is the measured noise");

        // Two dekacalories of log-CAI: far below anything the objective would
        // notice, and still an order of magnitude outside the window.
        let real = x + 100.0 * CAI_ULPS * f64::EPSILON * x.abs();
        assert!(!same_cai(x, real), "a real difference was swallowed");
    }
}
