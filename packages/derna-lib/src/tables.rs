//! The six node-keyed CKY tables and their BRD-merged updates.
//!
//! Re-keyed from LinearFold's sequence positions to DFA boundary nodes:
//! the closed-pair tables (`P`, `Multi`) additionally key on the pair's
//! boundary nucleotides, while the multi-branch tables (`M`, `M2`) and
//! the exterior table (`C`) key on nodes alone. Each cell keeps the
//! lowest combined-BRD derivation; the `(table, outer, key)` coordinate
//! is the equivalence class, exactly LinearFold's state packing extended
//! with a CAI accumulator.

use rna_mfe::Energy;

use crate::dfa::{Dfa, Node, NodeLayout};
use crate::hash::FxHashMap;
use crate::profile::Profile;
use crate::region::{PairConstraint, Verdict};
use crate::state::{Back, MultiOpenKey, PairKey, State};
use crate::{Beam, GroupPolicy};

/// All six CKY tables for one scan, plus the fixed λ used to rank merges.
pub struct Tables {
    /// Closed pairs, indexed by outer (3' boundary) node, keyed by
    /// [`PairKey`].
    pub p: Vec<FxHashMap<PairKey, State>>,
    /// Multi-loops in progress, same shape as `p`.
    pub multi: Vec<FxHashMap<PairKey, State>>,
    /// Multi-branch components (≥1 branch), indexed by outer node; each
    /// column is a dense node-indexed array keyed by inner node (see
    /// [`DenseCol`]).
    pub m: Vec<DenseCol>,
    /// Multi-branch components (≥2 branches), same shape as `m`.
    pub m2: Vec<DenseCol>,
    /// Exterior parses, indexed by node.
    pub c: Vec<Option<State>>,
    /// In-progress multi-loops, shared across every closing frontier: the
    /// best (BRD) derivation per `(opening node, opening nucleotide)`, with
    /// the 5' flank and the opening codon's broken-codon CAI already folded
    /// in. Closed against any frontier `>=` the opening codon boundary by
    /// adding that frontier's broken-codon CAI. Sharing one entry across all
    /// closing frontiers is what turns the `O(n³)` multi-loop scan into
    /// `O(n²)`.
    multi_open: FxHashMap<MultiOpenKey, State>,
    /// Activation buffer: `multi_open_pending[b]` holds opening states that
    /// become eligible (their core's right boundary reaches the opening
    /// codon boundary) at position `b`, merged into `multi_open` when the
    /// scan reaches `b`. Deferral keeps the separable `C(core)+D(frontier)`
    /// CAI split valid — an opening is only closed cross-codon.
    multi_open_pending: Vec<Vec<(MultiOpenKey, State)>>,
    /// Drained activation buffers, kept with their capacity for the next
    /// boundary that starts filling.
    multi_open_spare: Vec<Vec<(MultiOpenKey, State)>>,
    /// Fixed λ for the BRD scalarization (drives merge ordering).
    lambda: f64,
    /// The lattice's dense node layout, which every column is indexed by.
    layout: NodeLayout,
    /// The per-column pruning policy applied by the `prune_*` entry points.
    beam: Beam,
    /// Best still-achievable adaptation of the prefix **ending** at each node —
    /// [`Dfa::span_tail_cai`], indexed by [`Dfa::ix`]. The 5′-side term of
    /// the exact CAI-half completion bound: for a cell whose 5′ boundary is
    /// this node, the units before it contribute their best weights and the
    /// partial codon ending at it its best partial path. Empty unless [`Beam::cai_bound`] is set, since only
    /// completion-bound ranking reads it.
    end_cai: Vec<f64>,
    /// Region-pair constraint constraint, consulted at the single `P`-write choke
    /// point [`Self::update_p`]. Inert ([`PairConstraint::inert`]) on the CDS-only /
    /// no-constraint paths, so every `P` write is byte-identical there.
    region: PairConstraint,
    /// Where the scan's wall-clock went, when the scan was asked to record it.
    ///
    /// Carried on the tables so a caller reads the attribution from the same
    /// value it reads [`Self::cell_counts`] from, rather than from a second
    /// return channel. Disabled (and empty) unless the scan enabled it.
    profile: Profile,
    /// A cut every pruned column passes through, when the scan was asked only
    /// for designs scoring at most a threshold.
    cut: Option<ScanCut>,
    /// What the scan's pull indexes held when it ended, carried here for the
    /// same reason the profile is: one return channel for a scan's
    /// attribution. Zero unless the scan recorded them.
    index_sizes: IndexSizes,
}

/// What a scan's Lyngsø–Zuker pull indexes held at its end: the heap each
/// one's capacities account for, and how many entries each holds. The
/// indexes live only inside the scan, so their size is recorded on the way
/// out — a scan's resident memory is its tables plus these.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexSizes {
    /// The size-diagonal `H` index, its sparse range-min table included.
    pub h_bytes: usize,
    /// Entries on its diagonals.
    pub h_entries: usize,
    /// The one-sided index keyed by the inner pair's 3′ base.
    pub one_side_bytes: usize,
    /// The one-sided index keyed by the inner pair's 5′ base.
    pub one_side_k_bytes: usize,
    /// Entries in the 3′-keyed index.
    pub one_side_entries: usize,
    /// Entries in the 5′-keyed index.
    pub one_side_k_entries: usize,
}

impl IndexSizes {
    /// The heap all three indexes account for.
    pub fn total_bytes(&self) -> usize {
        self.h_bytes + self.one_side_bytes + self.one_side_k_bytes
    }
}

/// A cut on a scan: a cell survives only while its best inside plus a lower
/// bound on its cost to go at the scan's weight is at most `threshold`. A
/// derivation scoring at most the threshold passes through cells that all
/// survive, since each one's inside plus its true cost to go is at most that
/// derivation's score; so the scan still finds the optimum whenever the
/// optimum meets the threshold. Every other cell may be missing, so the
/// tables are no longer exact (see [`Tables::is_exact`]).
pub(crate) struct ScanCut {
    pub(crate) bound: crate::cost_to_go::CutBound,
    pub(crate) threshold: f64,
}

/// Room above a cut's threshold: a cell's inside and a bound's cost to go are
/// sums taken in different orders, so equal scores can differ in the last
/// bits.
const CUT_TOLERANCE: f64 = 1e-3;

impl ScanCut {
    /// Whether a cell scoring `inside` at `cell` survives.
    fn keeps(&self, inside: f64, cell: crate::cost_to_go::CellRef) -> bool {
        inside + self.bound.raw_at(cell) <= self.threshold + CUT_TOLERANCE
    }
}

impl Tables {
    /// The fixed λ this scan ranks merges by.
    /// The lattice's dense node layout, which every column is indexed by.
    pub fn layout(&self) -> NodeLayout {
        self.layout
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Whether the scan kept every candidate, so each cell holds the true
    /// minimum over its derivations rather than the best of a beam or of a
    /// cut.
    pub fn is_exact(&self) -> bool {
        self.beam == Beam::exact() && self.cut.is_none()
    }

    /// Cut every column this scan prunes from here on (see [`ScanCut`]).
    pub(crate) fn set_cut(&mut self, cut: ScanCut) {
        self.cut = Some(cut);
    }

    /// The region-pair constraint every `P` write was filtered through.
    pub(crate) fn region(&self) -> &PairConstraint {
        &self.region
    }

    /// Where the scan's wall-clock went. Disabled unless the scan recorded it.
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Store the finished scan's phase attribution.
    pub(crate) fn set_profile(&mut self, profile: Profile) {
        self.profile = profile;
    }

    /// What the finished scan's pull indexes held.
    pub fn index_sizes(&self) -> IndexSizes {
        self.index_sizes
    }

    /// Store what the finished scan's pull indexes held.
    pub(crate) fn set_index_sizes(&mut self, sizes: IndexSizes) {
        self.index_sizes = sizes;
    }

    /// Allocate empty tables sized for `dfa` at the given λ, under the pruning
    /// policy `beam` and the region-pair `region` constraint (pass
    /// [`PairConstraint::inert`] for an unconstrained scan).
    pub(crate) fn new(dfa: &Dfa, lambda: f64, beam: Beam, region: PairConstraint) -> Self {
        Self::with_columns(dfa, lambda, beam, region, DenseCol::new)
    }

    /// [`Self::new`] with values-only `M` and `M2` columns: each cell keeps its
    /// accumulators and reads back [`Back::Unrecorded`] for how it was reached.
    /// For a scan whose tables only feed an outside pass, which reads values
    /// alone; the columns' back-pointers are most of their memory.
    pub(crate) fn values_only(dfa: &Dfa, lambda: f64, beam: Beam, region: PairConstraint) -> Self {
        Self::with_columns(dfa, lambda, beam, region, DenseCol::values_only)
    }

    fn with_columns(
        dfa: &Dfa,
        lambda: f64,
        beam: Beam,
        region: PairConstraint,
        column: fn(NodeLayout, usize) -> DenseCol,
    ) -> Self {
        let layout = dfa.layout();
        let n = dfa.slots();
        // An `M`/`M2` column at outer index `i` only ever holds inner nodes
        // with index `< i` (the inner boundary is strictly left of the
        // outer), so column `i` is sized to exactly `i` slots.
        Self {
            p: vec![FxHashMap::default(); n],
            multi: vec![FxHashMap::default(); n],
            m: (0..n).map(|i| column(layout, i)).collect(),
            m2: (0..n).map(|i| column(layout, i)).collect(),
            c: vec![None; n],
            multi_open: FxHashMap::default(),
            multi_open_pending: vec![Vec::new(); dfa.seq_len as usize + 1],
            multi_open_spare: Vec::new(),
            lambda,
            layout,
            beam,
            end_cai: end_cai_table(dfa, beam),
            region,
            profile: Profile::new(false, 1),
            cut: None,
            index_sizes: IndexSizes::default(),
        }
    }

    /// Merge `cand` into `P[outer][key]`, keeping the lower-BRD state, after
    /// consulting the region-pair constraint on the pair's two endpoint positions
    /// (`key.open.pos`, `key.close.pos`). This is the single choke point every
    /// `P`-write funnels through — hairpins, `multi_close`, the stack/bulge and
    /// internal-loop closures, and all four Lyngsø–Zuker pull paths — so no site
    /// can bypass the constraint. A hard-forbidden pair is dropped; a soft
    /// penalty is added to the candidate's folding energy before the merge.
    /// `update_p` is called once per candidate derivation, and the penalty is a
    /// constant offset for a given pair, so a soft penalty is credited exactly
    /// once into the cell's stored value and cannot reorder its merges.
    pub fn update_p(&mut self, outer: Node, key: PairKey, cand: State) {
        let cand = match self.region.verdict(key.open.pos, key.close.pos) {
            Verdict::Allow => cand,
            Verdict::Forbid => return,
            Verdict::Penalize(pen) => State {
                mfe: cand.mfe + pen,
                ..cand
            },
        };
        merge_map(&mut self.p[self.layout.ix(outer)], key, cand, self.lambda);
    }

    /// The BRD [`Self::update_p`] compares `cand` by when merging it into
    /// `P[·][key]` — after the region constraint's penalty — or `None` when the
    /// constraint forbids the pair.
    pub fn p_score(&self, key: PairKey, cand: State) -> Option<f64> {
        let cand = match self.region.verdict(key.open.pos, key.close.pos) {
            Verdict::Allow => cand,
            Verdict::Forbid => return None,
            Verdict::Penalize(pen) => State {
                mfe: cand.mfe + pen,
                ..cand
            },
        };
        Some(cand.brd(self.lambda))
    }

    /// Merge `cand` into `Multi[outer][key]`, unless the region-pair constraint
    /// forbids the pair that would close it.
    ///
    /// A `Multi` cell's only consumer is the `P` cell of the same key, written
    /// through [`Self::update_p`], which drops a forbidden pair; so a forbidden
    /// `Multi` cell can reach no structure, and its cost to go is infinite in
    /// any outside pass. Behind a long flank under the default region constraint over a
    /// third of a scan's `Multi` cells were of this kind. A penalized pair is
    /// kept: the penalty is paid where the pair closes.
    pub fn update_multi(&mut self, outer: Node, key: PairKey, cand: State) {
        if matches!(
            self.region.verdict(key.open.pos, key.close.pos),
            Verdict::Forbid
        ) {
            return;
        }
        merge_map(
            &mut self.multi[self.layout.ix(outer)],
            key,
            cand,
            self.lambda,
        );
    }

    /// Merge `cand` into `M[outer][inner]`.
    pub fn update_m(&mut self, outer: Node, inner: Node, cand: State) {
        self.m[self.layout.ix(outer)].merge(inner, cand, self.lambda);
    }

    /// Merge `cand` into `M2[outer][inner]`.
    pub fn update_m2(&mut self, outer: Node, inner: Node, cand: State) {
        self.m2[self.layout.ix(outer)].merge(inner, cand, self.lambda);
    }

    /// Bifurcate `M[open]` into `M2[o]`: for each in-progress multibranch
    /// component `M[open][inner]`, form the candidate accumulators
    /// `(m_mfe + add_mfe, m_cai + add_cai)` and merge them into `M2[o][inner]`,
    /// building the back-pointer (`make_back`) only when the candidate wins the
    /// cell. `m` and `m2` are separate `Vec` fields, so `M[open]` is iterated in
    /// place while `M2[o]` is written — the two column borrows are disjoint and
    /// no per-pair snapshot is allocated. The hot read touches only the dense
    /// `(mfe, cai)` arrays; the cold back-pointer array is untouched except on a
    /// winning write. The merge keeps the lowest-BRD state, so the result is
    /// independent of iteration order.
    pub fn bifurcate_m_into_m2(
        &mut self,
        open: Node,
        o: Node,
        add_mfe: Energy,
        add_cai: f64,
        mut make_back: impl FnMut(Node) -> Back,
    ) {
        let lambda = self.lambda;
        let m_col = &self.m[self.layout.ix(open)];
        let m2_col = &mut self.m2[self.layout.ix(o)];
        for (inner, m_mfe, m_cai) in m_col.iter_hot() {
            m2_col.merge_hot(
                inner,
                m_mfe + add_mfe,
                m_cai + add_cai,
                lambda,
                &mut make_back,
            );
        }
    }

    /// Merge `cand` into `C[node]`.
    pub fn update_c(&mut self, node: Node, cand: State) {
        match &mut self.c[self.layout.ix(node)] {
            slot @ None => *slot = Some(cand),
            Some(existing) if cand.brd(self.lambda) < existing.brd(self.lambda) => *existing = cand,
            Some(_) => {}
        }
    }

    /// Merge an opening state into `multi_open` directly — used when the
    /// opening codon boundary coincides with the core's right boundary
    /// (an aligned core, eligible to close immediately).
    pub fn multi_open_insert(&mut self, key: MultiOpenKey, cand: State) {
        merge_map(&mut self.multi_open, key, cand, self.lambda);
    }

    /// Defer an opening state until the scan reaches position `at` (the
    /// opening codon boundary), when [`Self::multi_open_activate`] merges it
    /// into `multi_open`.
    pub fn multi_open_defer(&mut self, at: u32, key: MultiOpenKey, cand: State) {
        let slot = &mut self.multi_open_pending[at as usize];
        if slot.capacity() == 0 {
            if let Some(spare) = self.multi_open_spare.pop() {
                *slot = spare;
            }
        }
        slot.push((key, cand));
    }

    /// Merge every opening state buffered for position `at` into
    /// `multi_open`. Called once as the scan enters `at`.
    ///
    /// The emptied buffer keeps its capacity for a boundary still ahead
    /// ([`Self::multi_open_defer`]). Only the few boundaries within a codon of
    /// the scan are filling at once, so a handful of buffers serve the whole
    /// scan instead of one fresh, growing buffer per boundary.
    pub fn multi_open_activate(&mut self, at: u32) {
        let mut pending = std::mem::take(&mut self.multi_open_pending[at as usize]);
        for (key, cand) in pending.drain(..) {
            merge_map(&mut self.multi_open, key, cand, self.lambda);
        }
        if pending.capacity() > 0 {
            self.multi_open_spare.push(pending);
        }
    }

    /// Evict every active opening. An opening's 3′ unpaired stretch runs from
    /// its core's right boundary to whichever frontier closes it, so a scan that
    /// has to keep some position paired evicts the pool once it passes that
    /// position: no opening still in it could reach a later frontier without
    /// spanning the position. Deferred openings whose boundary is still ahead
    /// are untouched — their stretch has not started yet.
    pub fn multi_open_clear(&mut self) {
        self.multi_open.clear();
    }

    /// Snapshot of `multi_open` as a `(key, state)` vector, so the scan can
    /// close every active opening against a frontier while writing `Multi`.
    pub fn multi_open_states(&self) -> Vec<(MultiOpenKey, State)> {
        self.multi_open.iter().map(|(&k, &s)| (k, s)).collect()
    }

    /// Snapshot of `P[outer]` as a `(key, state)` vector, so the scan can
    /// iterate a column while writing into others.
    pub fn p_states(&self, outer: Node) -> Vec<(PairKey, State)> {
        self.p[self.layout.ix(outer)]
            .iter()
            .map(|(&k, &s)| (k, s))
            .collect()
    }

    /// The closed pair `key` at its outer node, when the table holds it.
    pub fn p_at(&self, outer: Node, key: PairKey) -> Option<State> {
        self.p[self.layout.ix(outer)].get(&key).copied()
    }

    /// Snapshot of `Multi[outer]`.
    pub fn multi_states(&self, outer: Node) -> Vec<(PairKey, State)> {
        self.multi[self.layout.ix(outer)]
            .iter()
            .map(|(&k, &s)| (k, s))
            .collect()
    }

    /// Snapshot of `M[outer]`, so the scan can iterate a column while
    /// writing into the same table field (`m_extend`).
    pub fn m_states(&self, outer: Node) -> Vec<(Node, State)> {
        self.m[self.layout.ix(outer)].iter_full().collect()
    }

    /// Snapshot of `M2[outer]`.
    pub fn m2_states(&self, outer: Node) -> Vec<(Node, State)> {
        self.m2[self.layout.ix(outer)].iter_full().collect()
    }

    /// Prune `P[outer]` under the scan's [`Beam`] policy (no-op at the exact
    /// width, where the column holds `<= cap` and no open node exceeds
    /// `per_open`). Called once per column, immediately before the column is expanded,
    /// so every surviving derivation that references it was built from a cell
    /// that survives — the traceback never follows a pruned pointer (see
    /// [`prune_map`] / [`prune_grouped`]).
    pub fn prune_p(&mut self, outer: Node) {
        self.prune_pairs(outer, PairTable::P);
    }

    /// Prune `Multi[outer]` under the scan's [`Beam`] policy.
    pub fn prune_multi(&mut self, outer: Node) {
        self.prune_pairs(outer, PairTable::Multi);
    }

    /// Prune one closed-pair table's column, dispatching between the flat knob
    /// (byte-for-byte the original behaviour when `per_open` is disabled and
    /// the completion bound is off) and the per-open / completion-bound path.
    fn prune_pairs(&mut self, outer: Node, table: PairTable) {
        let map = match table {
            PairTable::P => &mut self.p[self.layout.ix(outer)],
            PairTable::Multi => &mut self.multi[self.layout.ix(outer)],
        };
        if self.beam.per_open == usize::MAX && !self.beam.cai_bound && !self.beam.mfe_bound {
            prune_map(map, self.beam.cap, self.lambda);
        } else {
            prune_grouped(
                map,
                self.beam,
                self.lambda,
                &self.end_cai,
                &self.c,
                self.layout,
            );
        }
        if let Some(cut) = &self.cut {
            let lambda = self.lambda;
            map.retain(|&key, state| {
                let cell = match table {
                    PairTable::P => crate::cost_to_go::CellRef::P(outer, key),
                    PairTable::Multi => crate::cost_to_go::CellRef::Multi(outer, key),
                };
                cut.keeps(state.brd(lambda), cell)
            });
        }
    }

    /// Prune `M[outer]` under the scan's [`Beam`] policy. The dense `M` table
    /// holds one cell per open (inner) node, so the per-open cap is a no-op
    /// here; only the flat cap and the completion-bound rank apply.
    pub fn prune_m(&mut self, outer: Node) {
        self.m[self.layout.ix(outer)].prune_beam(self.beam, self.lambda, &self.end_cai, &self.c);
        if let Some(cut) = &self.cut {
            self.m[self.layout.ix(outer)].retain(self.lambda, |inner, brd| {
                cut.keeps(brd, crate::cost_to_go::CellRef::M(outer, inner))
            });
        }
    }

    /// Prune `M2[outer]` under the scan's [`Beam`] policy.
    pub fn prune_m2(&mut self, outer: Node) {
        self.m2[self.layout.ix(outer)].prune_beam(self.beam, self.lambda, &self.end_cai, &self.c);
        if let Some(cut) = &self.cut {
            self.m2[self.layout.ix(outer)].retain(self.lambda, |inner, brd| {
                cut.keeps(brd, crate::cost_to_go::CellRef::M2(outer, inner))
            });
        }
    }

    /// The heap the tables hold, estimated from their allocated capacities:
    /// hash-map slots at key plus value plus one control byte, dense columns at
    /// their three parallel arrays. An estimate, for budgeting how many sets of
    /// tables a caller keeps alive — not an allocator measurement.
    pub fn approx_bytes(&self) -> usize {
        use std::mem::size_of;
        let map_slot = size_of::<PairKey>() + size_of::<State>() + 1;
        let maps: usize = self
            .p
            .iter()
            .chain(&self.multi)
            .map(|m| m.capacity() * map_slot)
            .sum();
        let col = |c: &DenseCol| {
            c.mfe.capacity() * size_of::<i32>()
                + c.cai.capacity() * size_of::<f64>()
                + c.cold.capacity() * size_of::<Option<Back>>()
        };
        let cols: usize = self.m.iter().chain(&self.m2).map(col).sum();
        maps + cols + self.c.capacity() * size_of::<Option<State>>()
    }

    /// How many cells the finished scan actually materialized, per table.
    ///
    /// Read after the scan returns: the closed-pair tables are sparse maps and
    /// the multi-branch columns track their own occupancy, so the total is the
    /// lattice's realized size rather than its dense bound.
    ///
    /// It counts what **survives**. Nothing is pruned at [`Beam::exact`], so
    /// there the surviving count is the count; under a finite width a column is
    /// pruned as the scan passes it and the transient a column held before its
    /// prune is not recorded.
    pub fn cell_counts(&self) -> CellCounts {
        CellCounts {
            p_cells: self.p.iter().map(FxHashMap::len).sum(),
            multi_cells: self.multi.iter().map(FxHashMap::len).sum(),
            m_cells: self.m.iter().map(DenseCol::len).sum(),
            m2_cells: self.m2.iter().map(DenseCol::len).sum(),
            c_cells: self.c.iter().filter(|s| s.is_some()).count(),
        }
    }
}

/// Materialized cells across the five CKY tables of one finished scan.
///
/// The deterministic memory proxy: the same inputs give the same counts on any
/// host and under any load, where wall-clock and resident set size do not. A
/// constraint that removes candidates shows up here directly, since a forbidden
/// pair never writes a `P` cell for anything downstream to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CellCounts {
    /// Closed-pair (`P`) cells.
    pub p_cells: usize,
    /// In-progress multi-loop (`Multi`) cells.
    pub multi_cells: usize,
    /// Single-branch (`M`) multi-branch cells.
    pub m_cells: usize,
    /// `M2` multi-branch cells.
    pub m2_cells: usize,
    /// Exterior (`C`) cells.
    pub c_cells: usize,
}

impl CellCounts {
    /// Cells across all five tables.
    pub fn total(&self) -> usize {
        self.p_cells + self.multi_cells + self.m_cells + self.m2_cells + self.c_cells
    }
}

/// Which closed-pair table a [`Tables::prune_pairs`] call targets.
#[derive(Clone, Copy)]
enum PairTable {
    P,
    Multi,
}

/// A dense node-indexed column of an `M` / `M2` table, split into a hot
/// `(mfe, cai)` pair of arrays and a cold [`Back`] array, all indexed by
/// [`Dfa::ix`]. The hot arrays carry the accumulators the BRD merge and
/// the `O(n³)` bifurcation scan read on every candidate; the cold array carries
/// the back-pointer only the traceback reads. Splitting them keeps the hot
/// working set ~6× denser (4 B `mfe` + 8 B `cai` per slot, vs a ~72 B full
/// `State`), so the cache-linear ascending scan the bifurcation runs touches far
/// less memory. `mfe[i] == EMPTY` marks an empty slot (no `Option` tag, so the
/// skip stride is a bare `i32`); `cai[i]` and `cold[i]` are meaningful exactly
/// when `mfe[i] != EMPTY`. `count` tracks occupied cells so density and pruning
/// reads are `O(1)`.
#[derive(Clone)]
pub struct DenseCol {
    mfe: Vec<i32>,
    cai: Vec<f64>,
    /// Empty in a values-only column, which records no back-pointer.
    cold: Vec<Option<Back>>,
    keeps_back: bool,
    count: usize,
    /// The lattice's dense node layout: a slot index is an inner node's
    /// [`NodeLayout::ix`].
    layout: NodeLayout,
}

/// Sentinel `mfe` marking an empty [`DenseCol`] slot. A real accumulated
/// energy (bounded dekacalories) never reaches `i32::MAX`, so this cannot
/// collide with a stored state.
const EMPTY: i32 = i32::MAX;

/// The combined BRD of a hot `(mfe, cai)` cell — the same scalarization as
/// [`State::brd`], evaluated without materializing a `State`.
fn hot_brd(mfe: i32, cai: f64, lambda: f64) -> f64 {
    lambda * f64::from(mfe) - (1.0 - lambda) * (cai * 100.0)
}

impl DenseCol {
    /// An empty column holding inner indices `0..width`.
    fn new(layout: NodeLayout, width: usize) -> Self {
        Self {
            mfe: vec![EMPTY; width],
            cai: vec![0.0; width],
            cold: vec![None; width],
            keeps_back: true,
            count: 0,
            layout,
        }
    }

    /// An empty column holding inner indices `0..width` that keeps each cell's
    /// accumulators and not its back-pointer.
    fn values_only(layout: NodeLayout, width: usize) -> Self {
        Self {
            mfe: vec![EMPTY; width],
            cai: vec![0.0; width],
            cold: Vec::new(),
            keeps_back: false,
            count: 0,
            layout,
        }
    }

    /// Slot `idx`'s back-pointer: [`Back::Unrecorded`] in a values-only column.
    fn back(&self, idx: usize) -> Back {
        if self.keeps_back {
            self.cold[idx].expect("cold set whenever the slot is occupied")
        } else {
            Back::Unrecorded
        }
    }

    /// Merge a full `cand` at inner node `inner`, keeping the lower-BRD state
    /// (ties keep the incumbent, matching [`merge_map`]).
    fn merge(&mut self, inner: Node, cand: State, lambda: f64) {
        let idx = self.layout.ix(inner);
        if self.mfe[idx] == EMPTY {
            self.write(idx, cand.mfe.0, cand.cai, cand.back);
            self.count += 1;
        } else if cand.brd(lambda) < hot_brd(self.mfe[idx], self.cai[idx], lambda) {
            self.write(idx, cand.mfe.0, cand.cai, cand.back);
        }
    }

    /// Merge a candidate given by its hot accumulators, deferring the
    /// back-pointer to `make_back` — invoked only when the candidate wins the
    /// cell, so the cold array is written on merges that keep the state, never
    /// on the majority that lose. Byte-identical to [`Self::merge`] on the state
    /// it keeps; only the moment the back-pointer is built differs.
    fn merge_hot(
        &mut self,
        inner: Node,
        cand_mfe: Energy,
        cand_cai: f64,
        lambda: f64,
        make_back: impl FnOnce(Node) -> Back,
    ) {
        let idx = self.layout.ix(inner);
        if self.mfe[idx] == EMPTY {
            self.write(idx, cand_mfe.0, cand_cai, make_back(inner));
            self.count += 1;
        } else if hot_brd(cand_mfe.0, cand_cai, lambda)
            < hot_brd(self.mfe[idx], self.cai[idx], lambda)
        {
            self.write(idx, cand_mfe.0, cand_cai, make_back(inner));
        }
    }

    /// Store `(mfe, cai, back)` at slot `idx`. A real accumulated energy never
    /// reaches the [`EMPTY`] sentinel, so storing one would corrupt occupancy.
    fn write(&mut self, idx: usize, mfe: i32, cai: f64, back: Back) {
        debug_assert_ne!(mfe, EMPTY, "stored mfe collides with the empty sentinel");
        self.mfe[idx] = mfe;
        self.cai[idx] = cai;
        if self.keeps_back {
            self.cold[idx] = Some(back);
        }
    }

    /// Occupied cells in this column.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the column holds no cell at all.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The state for inner node `inner`, if present — reconstructed from the
    /// hot and cold arrays. Read only by the traceback.
    pub fn get(&self, inner: Node) -> Option<State> {
        let idx = self.layout.ix(inner);
        match self.mfe.get(idx) {
            Some(&mfe) if mfe != EMPTY => {
                Some(State::new(Energy(mfe), self.cai[idx], self.back(idx)))
            }
            _ => None,
        }
    }

    /// Occupied `(inner node, BRD)` cells in ascending index order, for a pass
    /// that reads the column's values and never its back-pointers.
    pub(crate) fn brd_cells(&self, lambda: f64) -> impl Iterator<Item = (Node, f64)> + '_ {
        self.iter_hot()
            .map(move |(inner, mfe, cai)| (inner, hot_brd(mfe.0, cai, lambda)))
    }

    /// Occupied `(inner node, mfe, cai)` cells in ascending index order — the
    /// hot bifurcation scan, which never touches the cold back-pointer array.
    /// The hot `mfe` and `cai` arrays over slots `[from, to)`, clipped to the
    /// column; an empty slot's `mfe` is the `EMPTY` sentinel (`i32::MAX`).
    pub(crate) fn hot_slices(&self, from: usize, to: usize) -> (&[i32], &[f64]) {
        let to = to.min(self.mfe.len());
        let from = from.min(to);
        (&self.mfe[from..to], &self.cai[from..to])
    }

    /// The number of node slots this column holds, occupied or not.
    pub(crate) fn slots(&self) -> usize {
        self.mfe.len()
    }

    fn iter_hot(&self) -> impl Iterator<Item = (Node, Energy, f64)> + '_ {
        self.mfe
            .iter()
            .enumerate()
            .filter(|&(_, &mfe)| mfe != EMPTY)
            .map(move |(i, &mfe)| (self.layout.node_of(i), Energy(mfe), self.cai[i]))
    }

    /// Occupied `(inner node, State)` cells in ascending index order, each
    /// reconstructed from the hot and cold arrays — used for the per-position
    /// `M`/`M2` snapshots.
    fn iter_full(&self) -> impl Iterator<Item = (Node, State)> + '_ {
        self.mfe
            .iter()
            .enumerate()
            .filter(|&(_, &mfe)| mfe != EMPTY)
            .map(move |(i, &mfe)| {
                (
                    self.layout.node_of(i),
                    State::new(Energy(mfe), self.cai[i], self.back(i)),
                )
            })
    }

    /// Prune the column under the [`Beam`] policy. The dense table holds one
    /// cell per open (inner) node — its slot index *is* [`Dfa::ix`] — so
    /// the per-open cap can never bite; only the flat `cap` and the ranking
    /// choice matter. The completion-bound rank adjusts each cell's BRD by its
    /// CAI-half bound and/or its exterior-MFE estimate (both keyed by its own
    /// slot / inner node); without either flag this is byte-for-byte the
    /// original flat-BRD prune.
    fn prune_beam(&mut self, beam: Beam, lambda: f64, end_cai: &[f64], c: &[Option<State>]) {
        if !beam.cai_bound && !beam.mfe_bound {
            self.prune(beam.cap, lambda);
            return;
        }
        let kk = (1.0 - lambda) * 100.0;
        self.prune_ranked(beam.cap, |i, mfe, cai| {
            let mut r = hot_brd(mfe, cai, lambda);
            if beam.cai_bound {
                r -= kk * end_cai[i];
            }
            if beam.mfe_bound {
                r += ext_completion(c, i, lambda);
            }
            r
        });
    }

    /// Keep only the `beam` lowest-BRD cells, clearing the rest to `EMPTY`. A
    /// no-op when the column already holds `<= beam` (so `beam == usize::MAX`
    /// leaves the exact lattice untouched). Selection is by BRD with
    /// `total_cmp`, deterministic given the ascending-index scan order.
    fn prune(&mut self, beam: usize, lambda: f64) {
        self.prune_ranked(beam, |_, mfe, cai| hot_brd(mfe, cai, lambda));
    }

    /// Keep the `beam` cells with the lowest `rank(slot, mfe, cai)`, clearing
    /// the rest to `EMPTY`. Shared core of [`Self::prune`] (rank = BRD) and the
    /// completion-bound prune (rank = BRD − CAI-half bound). Deterministic
    /// given the ascending-index scan order and `total_cmp`.
    fn prune_ranked(&mut self, beam: usize, rank: impl Fn(usize, i32, f64) -> f64) {
        if self.count <= beam {
            return;
        }
        let mut scored: Vec<(f64, usize)> = self
            .mfe
            .iter()
            .enumerate()
            .filter(|&(_, &mfe)| mfe != EMPTY)
            .map(|(i, &mfe)| (rank(i, mfe, self.cai[i]), i))
            .collect();
        scored.select_nth_unstable_by(beam, |a, b| a.0.total_cmp(&b.0));
        for &(_, i) in &scored[beam..] {
            self.mfe[i] = EMPTY;
        }
        self.count = beam;
    }

    /// Keep only the cells `keep(inner, brd)` accepts, clearing the rest to
    /// `EMPTY`.
    fn retain(&mut self, lambda: f64, keep: impl Fn(Node, f64) -> bool) {
        for i in 0..self.mfe.len() {
            let mfe = self.mfe[i];
            if mfe != EMPTY && !keep(self.layout.node_of(i), hot_brd(mfe, self.cai[i], lambda)) {
                self.mfe[i] = EMPTY;
                self.count -= 1;
            }
        }
    }
}

/// Keep only the `beam` lowest-BRD entries of `map`, dropping the rest.
/// A no-op when `map.len() <= beam` — so `beam == usize::MAX` leaves the
/// exact (`b → ∞`) lattice untouched. Selection is by BRD with `total_cmp`,
/// deterministic given `map`'s (deterministic) iteration order.
fn prune_map<K: std::hash::Hash + Eq + Copy>(
    map: &mut FxHashMap<K, State>,
    beam: usize,
    lambda: f64,
) {
    if map.len() <= beam {
        return;
    }
    let mut scored: Vec<(f64, K)> = map.iter().map(|(&k, &s)| (s.brd(lambda), k)).collect();
    scored.select_nth_unstable_by(beam, |a, b| a.0.total_cmp(&b.0));
    for &(_, k) in &scored[beam..] {
        map.remove(&k);
    }
}

/// Prune a closed-pair (`P` / `Multi`) column under the per-open / completion
/// bound policy — the path taken whenever [`Beam::per_open`] is finite or
/// [`Beam::cai_bound`] is set (the flat-BRD case routes to [`prune_map`], which
/// this must never perturb).
///
/// Each cell is ranked by its partial BRD, optionally less the CAI-half
/// completion bound `(1−λ)·100·end_cai[open]` (the best still-achievable
/// adaptation of the codons before its 5′ open node) and/or plus the
/// exterior-MFE completion estimate `C[open]` (the exact best left-exterior BRD
/// of the prefix before the open node; the 3′-side of both completions is
/// constant across a column and so cannot change the within-column order, and is
/// omitted). Two limits then apply, lowest-rank first: at most `per_open` cells
/// survive per distinct 5′ open node, and at most `cap` survive overall.
/// Reserving budget per open node is the structural fix for the closed-pair
/// collapse — a few long-span open nodes can no longer spend a column's whole
/// budget and starve the span-start variety a downstream merge needs; the
/// exterior-MFE estimate is the complementary scoring fix — it lets the flat
/// `cap` compare short- and long-span open nodes on their full
/// prefix-plus-inside cost rather than inside alone.
fn prune_grouped(
    map: &mut FxHashMap<PairKey, State>,
    beam: Beam,
    lambda: f64,
    end_cai: &[f64],
    c: &[Option<State>],
    layout: NodeLayout,
) {
    if beam.per_open == usize::MAX && map.len() <= beam.cap {
        return;
    }
    let kk = (1.0 - lambda) * 100.0;
    let rank = |s: &State, open: Node| {
        let mut r = s.brd(lambda);
        if beam.cai_bound {
            r -= kk * end_cai[layout.ix(open)];
        }
        if beam.mfe_bound {
            r += ext_completion(c, layout.ix(open), lambda);
        }
        r
    };
    let scored: Vec<(f64, PairKey, usize)> = map
        .iter()
        .map(|(&k, s)| (rank(s, k.open), k, layout.ix(k.open)))
        .collect();
    let mut keep = vec![true; scored.len()];
    match beam.group {
        GroupPolicy::Limit => {
            if beam.per_open != usize::MAX {
                cap_per_open(&scored, beam.per_open, &mut keep);
            }
            if keep.iter().filter(|&&k| k).count() > beam.cap {
                cap_flat(&scored, beam.cap, &mut keep);
            }
        }
        GroupPolicy::Reserve => reserve_per_open(&scored, beam, &mut keep),
    }
    for (i, &kept) in keep.iter().enumerate() {
        if !kept {
            map.remove(&scored[i].1);
        }
    }
}

/// Clear `keep[i]` for every cell beyond the `per_open` lowest-rank in its 5′
/// open-node group — the per-position beam's core. `scored[i]` is
/// `(rank, key, open_index)`; groups with `<= per_open` cells are untouched.
fn cap_per_open(scored: &[(f64, PairKey, usize)], per_open: usize, keep: &mut [bool]) {
    let mut by_open: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (i, cell) in scored.iter().enumerate() {
        by_open.entry(cell.2).or_default().push(i);
    }
    for idxs in by_open.values_mut() {
        if idxs.len() > per_open {
            idxs.select_nth_unstable_by(per_open, |&a, &b| scored[a].0.total_cmp(&scored[b].0));
            for &i in &idxs[per_open..] {
                keep[i] = false;
            }
        }
    }
}

/// Clear `keep[i]` for every still-kept cell beyond the `cap` lowest-rank
/// overall — the flat column cap, applied to the per-open survivors.
fn cap_flat(scored: &[(f64, PairKey, usize)], cap: usize, keep: &mut [bool]) {
    let mut idxs: Vec<usize> = (0..scored.len()).filter(|&i| keep[i]).collect();
    idxs.select_nth_unstable_by(cap, |&a, &b| scored[a].0.total_cmp(&scored[b].0));
    for &i in &idxs[cap..] {
        keep[i] = false;
    }
}

/// Keep the best `per_open` cells of each admitted group **unconditionally**,
/// then fill the rest of the column by global rank — the reservation floor.
///
/// The difference from [`cap_per_open`] followed by [`cap_flat`] is which cells a
/// column can lose. Under the ceiling reading, a group's best cells survive the
/// per-group step and can still be evicted by the column-wide rank, so a few
/// long-span open nodes that dominate the ranking take the whole budget. Under
/// the floor they cannot: the reservation is spent before the flat fill sees any
/// budget at all.
///
/// **Reservations can over-subscribe the column**, since `groups · per_open` has
/// nothing to do with `cap`. Groups are therefore admitted in order of their best
/// cell until the budget runs out or [`Beam::group_budget`] is reached, whichever
/// comes first — the point at which a cap stops being only a cell budget and
/// starts being a group budget too.
fn reserve_per_open(scored: &[(f64, PairKey, usize)], beam: Beam, keep: &mut [bool]) {
    let groups = groups_by_best(scored);
    let mut reserved = vec![false; scored.len()];
    let mut spent = 0usize;
    for group in groups.iter().take(beam.group_budget()) {
        let take = beam.per_open.min(group.len());
        if spent + take > beam.cap {
            break;
        }
        for &i in &group[..take] {
            reserved[i] = true;
        }
        spent += take;
    }
    fill_by_rank(scored, &reserved, beam.cap - spent, keep);
}

/// Every open-node group's cell indices, each group rank-sorted, and the groups
/// themselves ordered by their best cell.
///
/// Both orderings are what make the admission deterministic: the best cell of a
/// group decides whether the group is admitted at all, and the order within a
/// group decides which of its cells the reservation covers.
fn groups_by_best(scored: &[(f64, PairKey, usize)]) -> Vec<Vec<usize>> {
    let mut by_open: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (i, cell) in scored.iter().enumerate() {
        by_open.entry(cell.2).or_default().push(i);
    }
    let mut groups: Vec<Vec<usize>> = by_open.into_values().collect();
    for group in &mut groups {
        group.sort_unstable_by(|&a, &b| scored[a].0.total_cmp(&scored[b].0));
    }
    groups.sort_unstable_by(|a, b| scored[a[0]].0.total_cmp(&scored[b[0]].0));
    groups
}

/// Clear `keep` for every unreserved cell beyond the `room` lowest-rank ones.
/// Reserved cells are already kept and do not compete for this budget.
fn fill_by_rank(
    scored: &[(f64, PairKey, usize)],
    reserved: &[bool],
    room: usize,
    keep: &mut [bool],
) {
    let mut rest: Vec<usize> = (0..scored.len()).filter(|&i| !reserved[i]).collect();
    if rest.len() <= room {
        return;
    }
    rest.select_nth_unstable_by(room, |&a, &b| scored[a].0.total_cmp(&scored[b].0));
    for &i in &rest[room..] {
        keep[i] = false;
    }
}

/// The 5′-side term of the CAI-half completion bound for every node index:
/// `end_cai[node] = span_tail_cai(node)`, the best achievable adaptation of
/// the prefix ending at that node — every whole unit before it at its best,
/// plus the partial codon ending at it. Empty unless [`Beam::cai_bound`] is set, so
/// the exact and flat-BRD paths pay nothing for it.
fn end_cai_table(dfa: &Dfa, beam: Beam) -> Vec<f64> {
    if !beam.cai_bound {
        return Vec::new();
    }
    (0..dfa.slots())
        .map(|i| dfa.span_tail_cai(dfa.node_of(i)).unwrap_or(0.0))
        .collect()
}

/// The exterior-MFE completion estimate for a cell whose 5′ open node has index
/// `open_index`: the exact best left-exterior BRD `C[open]` of the prefix
/// `[0, open)`, or `0.0` when no exterior parse reaches the node. Read live from
/// the exterior table `c`, whose entry for every node strictly left of a column
/// is finalized by the time that column is pruned (the scan finalizes `C[pos]`
/// as it consumes `P[pos]`, before any later column's prune). A real 5′ open
/// node always has an exterior parse (the all-unpaired prefix), so the `0.0`
/// fallback is a neutral constraint, not a live case. Empty-constrained so the exact /
/// flat / CAI-only paths that leave [`Beam::mfe_bound`] unset never read it.
fn ext_completion(c: &[Option<State>], open_index: usize, lambda: f64) -> f64 {
    c.get(open_index)
        .and_then(|slot| slot.as_ref())
        .map_or(0.0, |s| s.brd(lambda))
}

fn merge_map<K: std::hash::Hash + Eq>(
    map: &mut FxHashMap<K, State>,
    key: K,
    cand: State,
    lambda: f64,
) {
    match map.get_mut(&key) {
        Some(existing) if cand.brd(lambda) < existing.brd(lambda) => *existing = cand,
        Some(_) => {}
        None => {
            map.insert(key, cand);
        }
    }
}

#[cfg(test)]
mod prune_tests {
    use super::*;
    use crate::state::Back;
    use rna_mfe::Nucleotide;

    /// A column from an explicit `(group, rank)` listing, so both the group a
    /// cell belongs to and its position in the column-wide order are fixed by
    /// the test rather than by the energy model.
    fn column(cells: &[(u32, i32)]) -> FxHashMap<PairKey, State> {
        let mut map = FxHashMap::default();
        for &(group, rank) in cells {
            let key = PairKey {
                open: Node::new(group, 0),
                close: Node::new(1000 + rank as u32, 0),
                opening: crate::state::Opening::new(Nucleotide::A, 0),
                nuc_close: Nucleotide::U,
            };
            map.insert(key, State::new(Energy(rank), 0.0, Back::Hairpin));
        }
        map
    }

    fn policy(cap: usize, per_open: usize, group: GroupPolicy) -> Beam {
        Beam {
            per_open,
            cap,
            cai_bound: false,
            mfe_bound: false,
            group,
            max_groups: usize::MAX,
        }
    }

    /// λ = 1 makes the rank the raw MFE, so the listed order is the order the
    /// policy sees. Both completion tables are empty because neither bound is on.
    fn prune(map: &mut FxHashMap<PairKey, State>, beam: Beam) {
        prune_grouped(map, beam, 1.0, &[], &[], NodeLayout::PLAIN);
    }

    fn groups_of(map: &FxHashMap<PairKey, State>) -> std::collections::BTreeSet<u32> {
        map.keys().map(|k| k.open.pos).collect()
    }

    fn ranks(map: &FxHashMap<PairKey, State>) -> Vec<i32> {
        let mut out: Vec<i32> = map.values().map(|s| s.mfe.0).collect();
        out.sort_unstable();
        out
    }

    /// One group rich in good cells, beside two groups holding one apiece.
    /// This is the shape that separates the two policies; a column where every
    /// group is the same size does not, because both policies then keep the same
    /// cells and the comparison is vacuous.
    fn uneven() -> Vec<(u32, i32)> {
        vec![(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6)]
    }

    /// The floor spends its reservations out of the cap rather than on top of it.
    ///
    /// `groups * per_open` has nothing to do with `cap`, so nothing about the
    /// arithmetic stops a floor from returning columns wider than the caller
    /// asked for — and a width that stops bounding the column stops bounding
    /// memory. This is the invariant an integration test cannot reach, because
    /// the cell counts it can see are totals rather than per-column maxima.
    #[test]
    fn a_reservation_never_exceeds_the_cap() {
        let wide: Vec<(u32, i32)> = (0..40)
            .flat_map(|g| (0..5).map(move |i| (g, g as i32 * 100 + i)))
            .collect();
        for cells in [&uneven()[..], &wide[..]] {
            for cap in [1usize, 4, 17] {
                for per_open in [1usize, 2, 8] {
                    let mut map = column(cells);
                    prune(&mut map, policy(cap, per_open, GroupPolicy::Reserve));
                    assert!(
                        map.len() <= cap,
                        "cap={cap} per_open={per_open} left {} cells",
                        map.len()
                    );
                }
            }
        }
    }

    /// **The distinction between the two policies, and it runs the opposite way
    /// to the name.**
    ///
    /// The ceiling discards everything beyond `per_open` in a group *before* the
    /// column-wide cap applies, so a group rich in good cells has its surplus
    /// thrown away — and the freed budget then goes to other groups. The floor
    /// keeps that surplus in play, but pays for it: reservations are rationed by
    /// [`Beam::group_budget`], which is `cap / per_open` when not given, so at a
    /// tight cap only a couple of groups reserve at all.
    ///
    /// On this seven-cell column at `cap = 5`, `per_open = 2`, the derived budget
    /// is **2 groups**. The ceiling keeps four cells spread over all three
    /// groups; the floor keeps five cells over **two**, spending the surplus on
    /// the richest group rather than on breadth. **The floor concentrates the
    /// column and the ceiling spreads it**, which is the reverse of what
    /// "reservation" suggests and is why the two cannot be ranked without
    /// measuring.
    #[test]
    fn the_floor_concentrates_where_the_ceiling_spreads() {
        let mut ceiling = column(&uneven());
        prune(&mut ceiling, policy(5, 2, GroupPolicy::Limit));
        assert_eq!(
            ranks(&ceiling),
            vec![0, 1, 5, 6],
            "the ceiling should keep two from the rich group and one from each other"
        );

        let mut floor = column(&uneven());
        prune(&mut floor, policy(5, 2, GroupPolicy::Reserve));
        assert_eq!(
            ranks(&floor),
            vec![0, 1, 2, 3, 5],
            "the floor should reserve for two groups and spend the rest on the surplus"
        );

        assert_ne!(
            ranks(&ceiling),
            ranks(&floor),
            "the two policies returned the same column, so the floor is a renaming \
             of the ceiling and every other assertion here is vacuous"
        );
        assert!(
            groups_of(&floor).len() < groups_of(&ceiling).len(),
            "the floor did not concentrate the column relative to the ceiling"
        );
        assert!(
            ceiling.len() < floor.len(),
            "the ceiling left budget unspent and the floor did not"
        );
    }

    /// Every **admitted** group keeps its best cell. Admission is what the group
    /// budget rations, so the guarantee is over the groups that got a
    /// reservation, not over every group in the column.
    #[test]
    fn the_floor_keeps_each_admitted_groups_best_cell() {
        let mut map = column(&uneven());
        prune(&mut map, policy(5, 2, GroupPolicy::Reserve));
        assert_eq!(
            groups_of(&map),
            [0u32, 1]
                .into_iter()
                .collect::<std::collections::BTreeSet<u32>>(),
            "the derived budget admits two groups here"
        );
        // Both admitted groups kept their own best cell.
        assert!(ranks(&map).contains(&0), "group 0 lost its best cell");
        assert!(ranks(&map).contains(&5), "group 1 lost its best cell");
    }

    /// Raising the group budget above the derived value is what buys breadth,
    /// and it is the knob that makes the two axes separable: at `max_groups = 3`
    /// the same column keeps five cells across all three groups, which neither
    /// the ceiling nor the derived-budget floor does.
    #[test]
    fn a_larger_group_budget_buys_breadth_the_derived_one_cannot() {
        let mut map = column(&uneven());
        let beam = Beam {
            max_groups: 3,
            ..policy(5, 2, GroupPolicy::Reserve)
        };
        prune(&mut map, beam);
        assert_eq!(ranks(&map), vec![0, 1, 2, 5, 6]);
        assert_eq!(groups_of(&map).len(), 3, "a budget of three admitted fewer");
    }

    /// An explicit group budget bounds how many groups may reserve; the rest of
    /// the column is then filled by global rank, which concentrates it.
    #[test]
    fn an_explicit_group_budget_bounds_the_reservations() {
        let mut map = column(&uneven());
        let beam = Beam {
            max_groups: 1,
            ..policy(5, 2, GroupPolicy::Reserve)
        };
        prune(&mut map, beam);
        // Only group 0 may reserve, so it takes two slots; the remaining three go
        // to the best unreserved cells, which are its own ranks 2..4.
        assert_eq!(ranks(&map), vec![0, 1, 2, 3, 4]);
        assert_eq!(
            groups_of(&map).len(),
            1,
            "a budget of one admitted more than one group"
        );
    }

    /// The derived budget is what a cap and a per-open budget already imply.
    #[test]
    fn the_group_budget_is_derived_from_the_two_widths() {
        assert_eq!(policy(100, 2, GroupPolicy::Reserve).group_budget(), 50);
        assert_eq!(policy(3, 8, GroupPolicy::Reserve).group_budget(), 1);
    }

    /// The ceiling path is untouched by the policy field.
    #[test]
    fn the_ceiling_keeps_the_best_cells_overall() {
        let even: Vec<(u32, i32)> = (0..4)
            .flat_map(|g| (0..5).map(move |i| (g, g as i32 * 100 + i)))
            .collect();
        let mut map = column(&even);
        prune(&mut map, policy(6, 2, GroupPolicy::Limit));
        assert_eq!(ranks(&map), vec![0, 1, 100, 101, 200, 201]);
    }
}
