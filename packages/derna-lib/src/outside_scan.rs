//! The outside pass over a finished scan's own tables.
//!
//! [`crate::outside`] states what an edge marginal is and checks it against one
//! fold per exclusion, by recording every rule instance of a small lattice. This
//! module computes the same quantity without recording anything: it walks the
//! production [`Tables`] backwards, re-enumerating each rule of the scan's
//! unreduced grammar from its tail, and reads every inside value from the cells
//! the scan already filled.
//!
//! # Order
//!
//! Positions are visited from the 3′ end, and at each one the tables in the
//! reverse of the order the forward scan finalizes them: the unpaired extensions,
//! the multi-loop frontiers, `M2`, `P`, then `Multi`. Every rule instance is
//! visited from one designated tail, at a moment its head's outside value is
//! final, and hands a value to each of its tails.
//!
//! # The shared multi-loop opening
//!
//! The scan closes multi-loops through a pool of openings shared across every
//! closing frontier (`Tables::multi_open`). Enumerating each opening against each
//! frontier would make this pass cubic where the scan is not, so the pool is
//! mirrored on both sides instead:
//!
//! - **inside**, a forward replay of the pool, snapshotted per position — the best
//!   opening per key that a frontier at that position closes against;
//! - **outside**, a backward running minimum over closing frontiers, snapshotted
//!   per position — the best frontier an opening activated there can reach.
//!
//! An opening's outside is the frontier snapshot at its codon boundary, and a
//! frontier's inside is the pool snapshot at its position. The whole codons of a
//! multi-loop's 3′ flank lie between the two, so each such codon's marginal is a
//! pool snapshot before it plus a frontier snapshot after it.
//!
//! # Fills
//!
//! An edge inside a best-adaptation fill `a → b` takes the fill's best path
//! through it. The best value of any rule instance using a span is aggregated per
//! span first. A codon lying wholly inside a span changes nothing outside itself,
//! so its edges read the least span value over every span that contains it — a
//! two-dimensional suffix minimum — and only the partial codons at a span's two
//! ends are visited per span.
//!
//! Only coding-sequence edges are recorded: a codon exclusion reads nothing else.

use rna_mfe::loops::{
    bulge_energy, hairpin_energy, hairpin_energy_scalar, internal_energy_scalar,
    pair_from_nucleotides, stack_energy, BulgeLoop, HairpinLoop, InternalLoopScalar, StackPair,
};
use rna_mfe::params::{BasePair, Nuc5};
use rna_mfe::{Energy, Nucleotide, Params};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub(crate) mod crossing;

use crate::dfa::{Dfa, Edge, Node, NodeLayout};
use crate::hash::FxHashMap;
use crate::interior::{LoopPairs, OneByNRows};
use crate::region::{PairConstraint, Verdict};
use crate::state::PairKey;
use crate::tables::Tables;

/// Smallest number of unpaired bases inside a hairpin.
const MIN_HAIRPIN_LOOP: u32 = 3;

/// Closed pairs a worker takes from the shared counter at a time: small enough
/// to even out cells of very different cost, large enough that the counter is
/// not contended.
const SINGLES_BATCH: usize = 4;

/// Lattice-unit edge marginals of every coding-sequence edge, indexed by
/// `node.index() * 4 + nucleotide`.
pub(crate) struct Marginals {
    pub(crate) by_edge: Vec<f64>,
    /// Every closed pair's outside value, by outer node and key.
    pub(crate) pair_outside: Vec<FxHashMap<PairKey, f64>>,
}

/// The cost to go of every cell of the scan's tables, on the tables' own BRD
/// scale. Unreachable cells hold infinity.
#[derive(Clone)]
pub(crate) struct CellCosts {
    pub(crate) c: Vec<f64>,
    /// `m[outer][inner]`, as the scan's `M` columns are indexed.
    pub(crate) m: Vec<Vec<f64>>,
    pub(crate) m2: Vec<Vec<f64>>,
    pub(crate) p: Vec<FxHashMap<PairKey, f64>>,
    pub(crate) multi: Vec<FxHashMap<PairKey, f64>>,
    /// Enclosing pairs the single-branch consumers visited, per loop family.
    pub(crate) tally: SinglesTally,
}

/// One choice's path pieces, as the outside pass hands values to them.
#[derive(Clone, Copy)]
enum Piece {
    Edge(Node, Nucleotide),
    Fill(Node, Node),
}

pub(crate) struct OutsideScan<'a> {
    params: &'a Params,
    dfa: &'a Dfa,
    /// `dfa`'s node layout, held here so the hot index arithmetic reads one
    /// field rather than chasing the lattice reference.
    layout: NodeLayout,
    t: &'a Tables,
    constraint: &'a PairConstraint,
    lambda: f64,
    kk: f64,
    single_max_len: usize,
    /// Whether edge marginals are recorded. Off, the pass computes the per-cell
    /// cost to go alone and skips every structure that only the marginals read.
    record: bool,
    /// Workers for the single-branch consumers of a position's closed pairs.
    /// Only the record-free pass fans out: recording writes shared structures.
    threads: usize,
    /// Whether every split loop the record-free pass scores is also scored over
    /// its corners and the two asserted equal. A split that is wrong only at a
    /// consumer that never wins its cell leaves every cell's value unchanged,
    /// so a cell-by-cell oracle cannot see it; this check can, and it runs at
    /// lengths no oracle reaches. Set by `DERNA_SPLIT_VALIDATE`, or by the
    /// caller of [`Self::cost_to_go`].
    validate_split: bool,
    /// Which closing pairs' `1×n` loops score the same at every corner.
    one_by_n: OneByNRows,
    /// This pass's identity among every pass in the process, so the
    /// thread-local strand slots can tell their own pass's strands apart.
    id: u64,
    /// Enclosing pairs the record-free pass visited, per loop family.
    tally: SharedTally,
    /// Nodes per table column: `(seq_len + 1) · 2`.
    width: usize,
    out_c: Vec<f64>,
    out_m: Vec<Vec<f64>>,
    out_m2: Vec<Vec<f64>>,
    out_p: Vec<FxHashMap<PairKey, f64>>,
    out_multi: Vec<FxHashMap<PairKey, f64>>,
    by_edge: Vec<f64>,
    /// The least value of any rule instance through each fill span, indexed
    /// `a.index() * width + b.index()`.
    spans: Vec<f64>,
    /// Inside: the opening pool a frontier at each position closes against,
    /// indexed `[pos][node.index() * 4 + nucleotide]`.
    pool_at: Vec<Vec<f64>>,
    /// Outside: the best frontier at or past each position, per opening key.
    frontier_at: Vec<Vec<f64>>,
    /// Every closed pair, as the pair enclosing a single-branch loop sees it.
    outers: Vec<Outer>,
    /// Each closed pair's index in [`Self::outers`], by outer node and key.
    /// A range: one per opening edge spelling the key's nucleotide.
    outer_id: FxHashMap<(usize, PairKey), (u32, u32)>,
    /// [`Self::outers`] by closing position, each list ordered by opening
    /// position.
    by_close: Vec<Vec<u32>>,
    /// [`Self::by_close`] with each pair's walked fields copied out in list
    /// order, so both single-branch walks read their enclosing pairs in sequence
    /// rather than through their indices.
    walk_close: Vec<Vec<Walk>>,
    /// Each closed pair's place in [`Self::walk_close`]: closing position and
    /// index, so its outside value can be copied in once final.
    walk_at: Vec<(u32, u32)>,
    /// [`Self::walk_close`] in runs of one opening position and one opening
    /// edge target, each with its least split outer half; filled for a closing
    /// position once its pairs' outside values are final, and only when not
    /// recording.
    runs: Vec<Vec<Run>>,
    /// `λ·(size + asymmetry)` of a generic loop, by `(n1, n2)`.
    shape: Vec<f64>,
    /// The same terms unweighted, as the energy model's integer: a generic
    /// loop's energy is this plus its two terminal mismatches.
    shape_e: Vec<i32>,
}

fn relax(slot: &mut f64, v: f64) {
    if v < *slot {
        *slot = v;
    }
}

impl<'a> OutsideScan<'a> {
    /// One past the largest unpaired total of a single-branch loop
    /// (`single_max_len`): each flank, and their sum, lie below it — the
    /// singles' slot windows, their shape tables' stride, and their bounds.
    fn single_span(&self) -> usize {
        self.single_max_len + 1
    }

    pub(crate) fn new(
        params: &'a Params,
        dfa: &'a Dfa,
        t: &'a Tables,
        constraint: &'a PairConstraint,
        single_max_len: usize,
    ) -> Self {
        Self::with_recording(params, dfa, t, constraint, single_max_len, true, 1)
    }

    fn with_recording(
        params: &'a Params,
        dfa: &'a Dfa,
        t: &'a Tables,
        constraint: &'a PairConstraint,
        single_max_len: usize,
        record: bool,
        threads: usize,
    ) -> Self {
        let lambda = t.lambda();
        let width = dfa.slots();
        let positions = dfa.seq_len as usize + 1;
        Self {
            params,
            layout: dfa.layout(),
            dfa,
            t,
            constraint,
            lambda,
            kk: (1.0 - lambda) * 100.0,
            single_max_len,
            record,
            threads: threads.max(1),
            validate_split: std::env::var_os("DERNA_SPLIT_VALIDATE").is_some(),
            one_by_n: OneByNRows::of(params),
            id: PASS_IDS.fetch_add(1, Ordering::Relaxed),
            tally: SharedTally::default(),
            width,
            out_c: vec![f64::INFINITY; width],
            out_m: (0..width).map(|i| vec![f64::INFINITY; i]).collect(),
            out_m2: (0..width).map(|i| vec![f64::INFINITY; i]).collect(),
            out_p: vec![FxHashMap::default(); width],
            out_multi: vec![FxHashMap::default(); width],
            by_edge: vec![f64::INFINITY; if record { width * 4 } else { 0 }],
            spans: vec![f64::INFINITY; if record { width * width } else { 0 }],
            pool_at: vec![Vec::new(); positions],
            frontier_at: vec![Vec::new(); positions],
            outers: Vec::new(),
            outer_id: FxHashMap::default(),
            by_close: vec![Vec::new(); positions],
            walk_close: Vec::new(),
            walk_at: Vec::new(),
            runs: vec![Vec::new(); positions],
            shape: Vec::new(),
            shape_e: Vec::new(),
        }
    }

    /// Run every pass and return the coding-sequence edge marginals.
    pub(crate) fn run(mut self) -> Marginals {
        let mut clock = Phases::from_env();
        clock.time("replay_pool", || self.replay_pool());
        clock.time("index_outers", || self.index_outers());
        self.sweep(&mut clock);
        clock.time("outer_pieces", || self.outer_pieces());
        clock.time("hairpins", || self.hairpins());
        clock.time("resolve_spans", || self.resolve_spans());
        clock.time("pool_codons", || self.pool_codons());
        clock.report();
        Marginals {
            by_edge: self.by_edge,
            pair_outside: self.out_p,
        }
    }

    /// The cost to go of every cell alone, without the edge marginals: the
    /// backward sweep and nothing after it, with the single-branch consumers of
    /// each position's closed pairs spread over `threads` workers. `validate`
    /// checks every split loop against its corners, as `DERNA_SPLIT_VALIDATE`
    /// does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cost_to_go(
        params: &'a Params,
        dfa: &'a Dfa,
        t: &'a Tables,
        constraint: &'a PairConstraint,
        single_max_len: usize,
        threads: usize,
        validate: bool,
    ) -> CellCosts {
        let mut scan =
            Self::with_recording(params, dfa, t, constraint, single_max_len, false, threads);
        scan.validate_split |= validate;
        let mut clock = Phases::from_env();
        clock.time("index_outers", || scan.index_outers());
        scan.sweep(&mut clock);
        clock.report();
        if clock.on {
            eprintln!("  outside singles {:?}", scan.tally.get());
        }
        CellCosts {
            c: scan.out_c,
            m: scan.out_m,
            m2: scan.out_m2,
            p: scan.out_p,
            multi: scan.out_multi,
            tally: scan.tally.get(),
        }
    }

    /// Every cell's outside, from the 3′ end, each table visited in the
    /// reverse of the order the forward scan finalizes it.
    fn sweep(&mut self, clock: &mut Phases) {
        // Every terminal completes for free: on a stop-labelled lattice a
        // cell's cost to go is then its best completion to any stop, a floor
        // under its completion to each.
        for &end in self.dfa.completion_ends() {
            self.out_c[self.layout.ix(end)] = 0.0;
        }
        for pos in (0..=self.dfa.seq_len).rev() {
            if pos < self.dfa.seq_len {
                clock.time("extensions", || self.extensions(pos));
            }
            clock.time("frontiers", || self.frontiers(pos));
            clock.time("m2_tails", || self.m2_tails(pos));
            let cells = self.p_cells(pos);
            let single_best = if self.record {
                Vec::new()
            } else {
                clock.time("p_singles", || self.singles_all(&cells))
            };
            clock.time("p_tails", || self.p_tails(&cells, &single_best));
            if !self.record && pos > 0 {
                clock.time("p_runs", || self.fill_runs(pos - 1));
            }
            clock.time("multi_tails", || self.multi_tails(pos));
            if !self.record {
                self.release_frontier_behind(pos);
            }
        }
    }

    /// Free the frontiers no later position reads. A position reads its own
    /// frontier, the next one's and the one at its codon boundary, at most two
    /// past it; only the marginals' [`Self::pool_codons`] reads every codon's,
    /// so a pass that records nothing keeps a few rows, not one per position.
    fn release_frontier_behind(&mut self, pos: u32) {
        if let Some(row) = self.frontier_at.get_mut(pos as usize + 3) {
            *row = Vec::new();
        }
    }

    // ---- inside reads ------------------------------------------------------

    fn in_c(&self, n: Node) -> f64 {
        self.t.c[self.layout.ix(n)].map_or(f64::INFINITY, |s| s.brd(self.lambda))
    }

    fn fill(&self, a: Node, b: Node) -> Option<f64> {
        self.dfa.broken_codon_cai(a, b)
    }

    fn e(&self, energy: Energy) -> f64 {
        self.lambda * f64::from(energy.0)
    }

    fn terminal_au(&self, pair: BasePair) -> Energy {
        if pair.applies_terminal_au() {
            Energy(self.params.misc.terminal_au_37)
        } else {
            Energy::ZERO
        }
    }

    fn stem(&self, pair: BasePair) -> Energy {
        Energy(self.params.ml.intern_37) + self.terminal_au(pair)
    }

    fn constraint(&self, i: u32, j: u32) -> Option<f64> {
        match self.constraint.verdict(i, j) {
            Verdict::Allow => Some(0.0),
            Verdict::Forbid => None,
            Verdict::Penalize(pen) => Some(self.lambda * f64::from(pen.0)),
        }
    }

    fn in_cds(&self, pos: u32) -> bool {
        pos >= self.dfa.cds_start() && pos < self.dfa.cds_end()
    }

    /// The dense slot of an opening at `pn` on nucleotide `nuc`. A designed
    /// stop's first node has two `U` edges (the `UA` and `UG` lanes), which
    /// share a slot: every array indexed by it is relaxed to a minimum, so the
    /// shared slot holds the lower of the two lanes' values, a bound on each.
    fn open_key(&self, pn: Node, nuc: Nucleotide) -> usize {
        self.layout.ix(pn) * 4 + nuc as usize
    }

    /// [`Self::open_key`] with the opening edge's lane, for a cache whose
    /// value is looked up under one exact opening and so cannot be shared
    /// between lanes.
    fn open_edge_key(&self, pn: Node, ep: Edge) -> usize {
        crate::state::lane_key(self.open_key(pn, ep.nuc), ep.to.lane)
    }

    // ---- recording ---------------------------------------------------------

    fn edge(&mut self, from: Node, nuc: Nucleotide, v: f64) {
        if self.record && self.in_cds(from.pos) {
            relax(
                &mut self.by_edge[self.layout.ix(from) * 4 + nuc as usize],
                v,
            );
        }
    }

    fn span(&mut self, a: Node, b: Node, v: f64) {
        if self.record
            && a.pos < b.pos
            && a.pos < self.dfa.cds_end()
            && b.pos > self.dfa.cds_start()
        {
            relax(
                &mut self.spans[self.layout.ix(a) * self.width + self.layout.ix(b)],
                v,
            );
        }
    }

    fn corner(&mut self, c: &Corner, v: f64) {
        for piece in c.pieces.iter().flatten() {
            self.pieces(std::slice::from_ref(piece), v);
        }
    }

    fn pieces(&mut self, pieces: &[Piece], v: f64) {
        if !self.record || !v.is_finite() {
            return;
        }
        for piece in pieces {
            match *piece {
                Piece::Edge(from, nuc) => self.edge(from, nuc, v),
                Piece::Fill(a, b) => self.span(a, b, v),
            }
        }
    }

    // ---- the shared opening pool, inside -----------------------------------

    /// Replay the scan's opening pool forward and keep its state at every
    /// frontier position.
    fn replay_pool(&mut self) {
        let keys = self.width * 4;
        let mut pool = vec![f64::INFINITY; keys];
        let mut pending: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.pool_at.len()];
        for pos in 0..=self.dfa.seq_len {
            for (key, v) in std::mem::take(&mut pending[pos as usize]) {
                relax(&mut pool[key], v);
            }
            for &jn in self.dfa.nodes_at(pos) {
                let boundary = self.dfa.next_codon_boundary(jn.pos);
                let Some(c) = self.dfa.span_head_cai(jn) else {
                    continue;
                };
                let mut emit = |key: usize, v: f64| {
                    if boundary == pos {
                        relax(&mut pool[key], v);
                    } else {
                        pending[boundary as usize].push((key, v));
                    }
                };
                let mut far = vec![f64::INFINITY; keys];
                let mut top = None;
                for (inner, st) in self.t.m2[self.layout.ix(jn)].brd_cells(self.lambda) {
                    let near = inner.pos.saturating_sub(self.single_max_len as u32);
                    for (pn, ep, left) in self.left_flanks(inner, near) {
                        emit(
                            self.open_key(pn, ep.nuc),
                            st - self.kk * (ep.cai + left + c),
                        );
                    }
                    if near > 0 {
                        top = top.max(Some(near - 1));
                        for (pn, ep, left) in self.flanks_at(inner, near - 1) {
                            relax(
                                &mut far[self.open_key(pn, ep.nuc)],
                                st - self.kk * (ep.cai + left),
                            );
                        }
                    }
                }
                self.far_pool(&mut far, top, |key, v| emit(key, v - self.kk * c));
            }
            self.pool_at[pos as usize] = pool.clone();
        }
    }

    /// A multi-loop's 5′ flanks around a core starting at `inner`, opening at
    /// `from` or later: the opening node and edge, and the flank's best
    /// adaptation. The leading run is unbounded, as in the scan's
    /// `m2_to_multi`; callers that do not record take the flanks shorter than
    /// `SINGLE_MAX_LEN` here and the longer ones through a sweep shared by all
    /// cores of one `core_outer` ([`Self::far_pool`], [`Self::far_m2_tails`]).
    fn left_flanks(&self, inner: Node, from: u32) -> Vec<(Node, Edge, f64)> {
        let mut out = Vec::new();
        self.left_flanks_into(inner, from, &mut out);
        out
    }

    /// The 5′ flanks around a core starting at `inner` that open at `p`.
    fn flanks_at(&self, inner: Node, p: u32) -> Vec<(Node, Edge, f64)> {
        let mut out = Vec::new();
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                if let Some(left) = self.fill(ep.to, inner) {
                    out.push((pn, *ep, left));
                }
            }
        }
        out
    }

    /// [`Self::left_flanks`] into `out`, which is cleared first.
    fn left_flanks_into(&self, inner: Node, from: u32, out: &mut Vec<(Node, Edge, f64)>) {
        out.clear();
        for p in from..inner.pos {
            for &pn in self.dfa.nodes_at(p) {
                for ep in self.dfa.right(pn) {
                    if let Some(left) = self.fill(ep.to, inner) {
                        out.push((pn, *ep, left));
                    }
                }
            }
        }
    }

    /// Sweep the long-flank openings of one `core_outer` right to left: each
    /// opening (`far`, by [`Self::open_key`], seeded at its flank of exactly
    /// `SINGLE_MAX_LEN`) is handed to `emit`, then offered one position left.
    /// A flank grows at a fixed price — the unpaired base is free and the new
    /// opening edge adds its adaptation — so the best opening one position left
    /// is the best over those it extends, whatever their cores.
    fn far_pool(&self, far: &mut [f64], top: Option<u32>, mut emit: impl FnMut(usize, f64)) {
        let Some(top) = top else {
            return;
        };
        for p in (0..=top).rev() {
            for &pn in self.dfa.nodes_at(p) {
                for ep in self.dfa.right(pn) {
                    let f = far[self.open_key(pn, ep.nuc)];
                    if !f.is_finite() {
                        continue;
                    }
                    emit(self.open_key(pn, ep.nuc), f);
                    if p == 0 {
                        continue;
                    }
                    for l in self.dfa.left(pn) {
                        relax(&mut far[self.open_key(l.to, l.nuc)], f - self.kk * l.cai);
                    }
                }
            }
        }
    }

    // ---- the backward sweep ------------------------------------------------

    /// `C` and `M` extended by one unpaired nucleotide.
    fn extensions(&mut self, pos: u32) {
        for &jn in self.dfa.nodes_at(pos) {
            let c_in = self.in_c(jn);
            for e in self.dfa.right(jn).to_vec() {
                let v = self.out_c[self.layout.ix(e.to)] - self.kk * e.cai;
                relax(&mut self.out_c[self.layout.ix(jn)], v);
                self.edge(jn, e.nuc, v + c_in);
                let cells: Vec<(Node, f64)> = self.t.m[self.layout.ix(jn)]
                    .brd_cells(self.lambda)
                    .collect();
                for (inner, m_in) in cells {
                    let v =
                        self.out_m[self.layout.ix(e.to)][self.layout.ix(inner)] - self.kk * e.cai;
                    relax(
                        &mut self.out_m[self.layout.ix(jn)][self.layout.ix(inner)],
                        v,
                    );
                    self.edge(jn, e.nuc, v + m_in);
                }
            }
        }
    }

    /// Frontiers at `pos`: every `Multi` cell whose closing base sits here,
    /// folded into the running frontier minimum, and the pool-path instances
    /// that close through them handed to their closing edge and partial codon.
    fn frontiers(&mut self, pos: u32) {
        let mut best = if (pos as usize + 1) < self.frontier_at.len() {
            self.frontier_at[pos as usize + 1].clone()
        } else {
            vec![f64::INFINITY; self.width * 4]
        };
        if pos < self.dfa.seq_len {
            for &qn in self.dfa.nodes_at(pos) {
                let d_start = Node::new(self.dfa.codon_start(qn.pos), 0);
                let Some(d) = self.dfa.span_tail_cai(qn) else {
                    continue;
                };
                for eq in self.dfa.right(qn).to_vec() {
                    let cells: Vec<(PairKey, f64)> = self.out_multi[self.layout.ix(eq.to)]
                        .iter()
                        .filter(|(k, _)| k.close == qn && k.nuc_close == eq.nuc)
                        .map(|(k, v)| (*k, *v))
                        .collect();
                    for (key, out) in cells {
                        let val = out - self.kk * (d + eq.cai);
                        let slot = self.open_key(key.open, key.nuc_open());
                        relax(&mut best[slot], val);
                        if self.record {
                            let v = self.pool_at[pos as usize][slot] + val;
                            self.pieces(&[Piece::Edge(qn, eq.nuc), Piece::Fill(d_start, qn)], v);
                        }
                    }
                }
            }
        }
        self.frontier_at[pos as usize] = best;
    }

    /// `M2` tails: `M = M2`, and the multi-loop wrap — its short 3′ flanks
    /// directly, its long ones through the frontier snapshot at the core's codon
    /// boundary.
    fn m2_tails(&mut self, pos: u32) {
        for &jn in self.dfa.nodes_at(pos) {
            let cells: Vec<(Node, f64)> = self.t.m2[self.layout.ix(jn)]
                .brd_cells(self.lambda)
                .collect();
            let boundary = self.dfa.next_codon_boundary(jn.pos);
            let c_outer = self.dfa.span_head_cai(jn);
            // The short 3′ flanks out of `jn` — closing node, closing edge and
            // the flank's fill — in the order the wrap visits them; and, per
            // opening `(pn, edge)`, those whose multi-loop has an outside
            // value, with it. Neither depends on the core's inner node, so every
            // cell at `jn` shares them.
            let mut closes: Vec<(Node, Edge, f64)> = Vec::new();
            for q in jn.pos..boundary {
                for &qn in self.dfa.nodes_at(q) {
                    if let Some(right) = self.fill(jn, qn) {
                        closes.extend(self.dfa.right(qn).iter().map(|&eq| (qn, eq, right)));
                    }
                }
            }
            // Recording needs every flank's own pieces, so it enumerates them
            // all; otherwise the flanks of `SINGLE_MAX_LEN` and longer go through
            // one sweep for the whole column (`far_m2_tails`).
            let span = if self.record {
                u32::MAX
            } else {
                self.single_max_len as u32
            };
            if !self.record {
                self.far_m2_tails(jn, &cells, boundary, c_outer);
            }
            let mut wraps: FxHashMap<usize, Vec<(u32, f64)>> = FxHashMap::default();
            let mut flanks = Vec::new();
            for (inner, st) in cells {
                let head = self.out_m[self.layout.ix(jn)][self.layout.ix(inner)];
                relax(
                    &mut self.out_m2[self.layout.ix(jn)][self.layout.ix(inner)],
                    head,
                );
                self.left_flanks_into(inner, inner.pos.saturating_sub(span), &mut flanks);
                for &(pn, ep, left) in &flanks {
                    let left_w = -self.kk * (ep.cai + left);
                    let found = wraps
                        .entry(self.open_edge_key(pn, ep))
                        .or_insert_with(|| self.wraps_found(pn, ep, &closes));
                    for &(ci, h) in found.iter() {
                        let (qn, eq, right) = closes[ci as usize];
                        let w = left_w - self.kk * (right + eq.cai);
                        relax(
                            &mut self.out_m2[self.layout.ix(jn)][self.layout.ix(inner)],
                            h + w,
                        );
                        self.pieces(
                            &[
                                Piece::Edge(pn, ep.nuc),
                                Piece::Edge(qn, eq.nuc),
                                Piece::Fill(ep.to, inner),
                                Piece::Fill(jn, qn),
                            ],
                            h + w + st,
                        );
                    }
                    let Some(c) = c_outer else {
                        continue;
                    };
                    let g = self.frontier_at[boundary as usize][self.open_key(pn, ep.nuc)];
                    let w = left_w - self.kk * c;
                    relax(
                        &mut self.out_m2[self.layout.ix(jn)][self.layout.ix(inner)],
                        g + w,
                    );
                    self.pieces(
                        &[
                            Piece::Edge(pn, ep.nuc),
                            Piece::Fill(ep.to, inner),
                            Piece::Fill(jn, Node::new(boundary, 0)),
                        ],
                        g + w + st,
                    );
                }
            }
        }
    }

    /// Of the short 3′ flanks `closes`, those a multi-loop opened at `pn` by
    /// `ep` can close through, by index, with that multi-loop's outside value.
    fn wraps_found(&self, pn: Node, ep: Edge, closes: &[(Node, Edge, f64)]) -> Vec<(u32, f64)> {
        closes
            .iter()
            .enumerate()
            .filter(|(_, (_, eq, _))| pair_from_nucleotides(ep.nuc, eq.nuc) != BasePair::NN)
            .filter_map(|(ci, &(qn, eq, _))| {
                let key = PairKey {
                    open: pn,
                    close: qn,
                    opening: ep.opening,
                    nuc_close: eq.nuc,
                };
                let h = *self.out_multi[self.layout.ix(eq.to)].get(&key)?;
                Some((ci as u32, h))
            })
            .collect()
    }

    /// The outside of one opening `(pn, ep)` of a multi-loop around a core
    /// ending at `jn`, less the flank to the core: the best of its short 3′
    /// closures inside `jn`'s codon and of the frontier snapshot past it. It
    /// does not depend on which core the loop wraps.
    fn opening_outside(
        &self,
        jn: Node,
        boundary: u32,
        c_outer: Option<f64>,
        pn: Node,
        ep: Edge,
    ) -> f64 {
        let mut best = f64::INFINITY;
        for q in jn.pos..boundary {
            for &qn in self.dfa.nodes_at(q) {
                let Some(right) = self.fill(jn, qn) else {
                    continue;
                };
                for eq in self.dfa.right(qn) {
                    if pair_from_nucleotides(ep.nuc, eq.nuc) == BasePair::NN {
                        continue;
                    }
                    let key = PairKey {
                        open: pn,
                        close: qn,
                        opening: ep.opening,
                        nuc_close: eq.nuc,
                    };
                    if let Some(&h) = self.out_multi[self.layout.ix(eq.to)].get(&key) {
                        relax(&mut best, h - self.kk * (right + eq.cai));
                    }
                }
            }
        }
        if let Some(c) = c_outer {
            let g = self.frontier_at[boundary as usize][self.open_key(pn, ep.nuc)];
            relax(&mut best, g - self.kk * c);
        }
        best
    }

    /// The multi-loop wrap of every core ending at `jn` through left flanks of
    /// `SINGLE_MAX_LEN` or longer, by one left-to-right sweep instead of a scan
    /// per core. `reach[y]` is the best opening outside plus flank adaptation
    /// over openings strictly left of node `y` whose flank passes through `y`;
    /// a flank grows by one position at the price of the edge it crosses, so
    /// `reach` at one position follows from the position before. A core
    /// starting at `inner` reads `reach` at the nodes `SINGLE_MAX_LEN` before it,
    /// plus the rest of its flank.
    fn far_m2_tails(
        &mut self,
        jn: Node,
        cells: &[(Node, f64)],
        boundary: u32,
        c_outer: Option<f64>,
    ) {
        let l = self.single_max_len as u32;
        let Some(top) = cells
            .iter()
            .filter(|(i, _)| i.pos > l)
            .map(|(i, _)| i.pos - l)
            .max()
        else {
            return;
        };
        let mut reach = vec![f64::INFINITY; self.width];
        for s in 1..=top {
            for &y in self.dfa.nodes_at(s) {
                let mut best = f64::INFINITY;
                for e in self.dfa.left(y) {
                    let opened =
                        self.opening_outside(jn, boundary, c_outer, e.to, Edge { to: y, ..*e });
                    relax(
                        &mut best,
                        opened.min(reach[self.layout.ix(e.to)]) - self.kk * e.cai,
                    );
                }
                reach[self.layout.ix(y)] = best;
            }
        }
        for &(inner, _) in cells {
            if inner.pos <= l {
                continue;
            }
            let mut best = f64::INFINITY;
            for &y in self.dfa.nodes_at(inner.pos - l) {
                if let Some(rest) = self.fill(y, inner) {
                    relax(&mut best, reach[self.layout.ix(y)] - self.kk * rest);
                }
            }
            relax(
                &mut self.out_m2[self.layout.ix(jn)][self.layout.ix(inner)],
                best,
            );
        }
    }

    /// `P` tails: onto the exterior, into a multibranch, and grown into every
    /// enclosing stack, bulge and internal loop.
    ///
    /// The single-branch consumers are most of the pass, and each is a pure
    /// read of `out_p` at strictly later positions — final by now — so without
    /// recording they are computed for the whole position first, across
    /// workers, and the minimum makes the result independent of how they are
    /// split. Recording keeps them in the serial loop, where each cell's pieces
    /// are applied as soon as that cell is done.
    ///
    /// `single_best` holds each cell's single-branch value, from
    /// [`Self::singles_all`], when not recording, and is empty when recording.
    fn p_tails(&mut self, cells: &[(Node, PairKey, f64)], single_best: &[f64]) {
        let bifurcations = self.bifurcation_tails(cells);
        for (idx, &(jn, key, p_in)) in cells.iter().enumerate() {
            let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
            let mut out = f64::INFINITY;
            let c_in = self.in_c(key.open);
            if c_in.is_finite() {
                let h = self.out_c[self.layout.ix(jn)] + self.e(self.terminal_au(pair));
                relax(&mut out, h + c_in);
                relax(&mut self.out_c[self.layout.ix(key.open)], h + p_in);
            }
            let stem = self.e(self.stem(pair));
            relax(
                &mut out,
                self.out_m[self.layout.ix(jn)][self.layout.ix(key.open)] + stem,
            );
            relax(&mut out, bifurcations[idx]);
            let singles = if self.record {
                self.singles_recorded(jn, key, p_in)
            } else {
                single_best[idx]
            };
            relax(&mut out, singles);
            let slot = self.out_p[self.layout.ix(jn)]
                .entry(key)
                .or_insert(f64::INFINITY);
            relax(slot, out);
            let (first, last) = self.outer_id[&(self.layout.ix(jn), key)];
            for id in first as usize..last as usize {
                self.outers[id].out = *slot;
                let (q, i) = self.walk_at[id];
                self.walk_close[q as usize][i as usize].out = *slot;
            }
        }
    }

    /// The bifurcation tails of each cell, `M2 = M + P`: the cell's best
    /// outside through an `M` beside it, and each such `M` cell's outside
    /// through it. Cells at one outer node opening at one node read the same
    /// `M` column and the same `M2` row, so the column is walked once for all
    /// of them, each cell scored by the same sums as alone.
    fn bifurcation_tails(&mut self, cells: &[(Node, PairKey, f64)]) -> Vec<f64> {
        let t = self.t;
        let mut order: Vec<usize> = (0..cells.len()).collect();
        order.sort_by_key(|&i| (self.layout.ix(cells[i].0), self.layout.ix(cells[i].1.open)));
        let mut best = vec![f64::INFINITY; cells.len()];
        for group in
            order.chunk_by(|&i, &j| (cells[i].0, cells[i].1.open) == (cells[j].0, cells[j].1.open))
        {
            let (jn, open) = (cells[group[0]].0, cells[group[0]].1.open);
            let sides: Vec<(usize, f64, f64)> = group
                .iter()
                .map(|&c| {
                    let key = cells[c].1;
                    let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                    (c, self.e(self.stem(pair)), cells[c].2)
                })
                .collect();
            for (i, m_in) in t.m[self.layout.ix(open)].brd_cells(self.lambda) {
                let m2 = self.out_m2[self.layout.ix(jn)][self.layout.ix(i)];
                let slot = &mut self.out_m[self.layout.ix(open)][self.layout.ix(i)];
                for &(c, stem, p_in) in &sides {
                    let h = m2 + stem;
                    relax(&mut best[c], h + m_in);
                    relax(slot, h + p_in);
                }
            }
        }
        best
    }

    /// Every closed pair whose outer node sits at `pos`, with its inside value.
    fn p_cells(&self, pos: u32) -> Vec<(Node, PairKey, f64)> {
        let (t, lambda) = (self.t, self.lambda);
        self.dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&jn| {
                t.p_states(jn)
                    .into_iter()
                    .map(move |(key, st)| (jn, key, st.brd(lambda)))
            })
            .collect()
    }

    /// [`Self::singles_group`] for every cell, split over the workers.
    ///
    /// Cells at one outer node opening at one node are one group, walked once.
    /// Groups differ widely in how many flanks they enumerate, so workers take
    /// small batches from a shared counter rather than one fixed slice each; a
    /// fixed split leaves the position waiting on its slowest slice.
    fn singles_all(&self, cells: &[(Node, PairKey, f64)]) -> Vec<f64> {
        let mut order: Vec<usize> = (0..cells.len()).collect();
        order.sort_by_key(|&i| (self.layout.ix(cells[i].0), self.layout.ix(cells[i].1.open)));
        let groups: Vec<&[usize]> = order
            .chunk_by(|&i, &j| (cells[i].0, cells[i].1.open) == (cells[j].0, cells[j].1.open))
            .collect();
        let one = |group: &[usize], done: &mut Vec<(usize, f64)>| {
            let keys: Vec<PairKey> = group.iter().map(|&i| cells[i].1).collect();
            let (values, tally) = self.singles_group(cells[group[0]].0, &keys);
            self.tally.add(tally);
            done.extend(group.iter().copied().zip(values));
        };
        let mut best = vec![f64::INFINITY; cells.len()];
        if self.threads <= 1 || groups.len() < 2 * self.threads {
            let mut done = Vec::with_capacity(cells.len());
            for group in &groups {
                one(group, &mut done);
            }
            for (i, v) in done {
                best[i] = v;
            }
            return best;
        }
        let next = AtomicUsize::new(0);
        let workers = self.threads.min(groups.len().div_ceil(SINGLES_BATCH));
        let parts: Vec<Vec<(usize, f64)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    let (next, groups, one) = (&next, &groups, &one);
                    crate::task_clock::spawn(scope, move || {
                        let mut done = Vec::new();
                        loop {
                            let start = next.fetch_add(SINGLES_BATCH, Ordering::Relaxed);
                            if start >= groups.len() {
                                break done;
                            }
                            let end = (start + SINGLES_BATCH).min(groups.len());
                            for group in &groups[start..end] {
                                one(group, &mut done);
                            }
                        }
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a singles worker panicked"))
                .collect()
        });
        for (i, v) in parts.into_iter().flatten() {
            best[i] = v;
        }
        best
    }

    /// Index every closed pair as an enclosing pair: its two edges, the constraint's
    /// adjustment, and the outer half of every split loop it can close — the
    /// same for whichever pair it encloses. Also tabulates the generic loop's
    /// size and asymmetry terms.
    fn index_outers(&mut self) {
        let dfa = self.dfa;
        let table = &self.params.mismatch_interior;
        for oi in 0..self.width {
            let o = dfa.node_of(oi);
            if o.pos > dfa.seq_len {
                continue;
            }
            for (key, _) in self.t.p_states(o) {
                let (pn, qn) = (key.open, key.close);
                // The cell sits at its closing edge's target, which names that
                // edge even where two of `qn`'s out-edges spell the nucleotide.
                let eq = *dfa
                    .right(qn)
                    .iter()
                    .find(|e| e.nuc == key.nuc_close && e.to == o)
                    .expect("a pair's closing edge is in the lattice");
                let Some(constraint) = self.constraint(pn.pos, qn.pos) else {
                    continue;
                };
                let first = self.outers.len() as u32;
                // The key names its opening edge's lane, so each of a designed
                // stop's two `U` edges opens cells of its own.
                let ep = key
                    .open_edge(dfa)
                    .expect("a pair's opening edge is in the lattice");
                self.index_outer(key, (ep, eq), constraint);
                let last = self.outers.len() as u32;
                assert!(first < last, "a pair is indexed as an enclosing pair");
                self.outer_id
                    .insert((self.layout.ix(o), key), (first, last));
            }
        }
        let outers = &self.outers;
        for list in &mut self.by_close {
            list.sort_by_key(|&id| {
                let x = &outers[id as usize];
                (x.pn.pos, x.ep.to.lane)
            });
        }
        self.walk_at = vec![(0, 0); outers.len()];
        self.walk_close = self
            .by_close
            .iter()
            .enumerate()
            .map(|(q, list)| {
                list.iter()
                    .enumerate()
                    .map(|(i, &id)| {
                        self.walk_at[id as usize] = (q as u32, i as u32);
                        Walk::of(id, &outers[id as usize])
                    })
                    .collect()
            })
            .collect();
        // The generic loop's size and asymmetry terms, read off the energy
        // model at one corner choice less its two mismatches, so they cannot
        // drift from it.
        let l = self.single_span();
        self.shape = vec![f64::INFINITY; l * l];
        self.shape_e = vec![0; l * l];
        let (au, n) = (BasePair::AU, Nucleotide::A);
        for n1 in 0..l {
            for n2 in 0..l {
                if !is_generic(n1, n2) || n1 + n2 >= l {
                    continue;
                }
                let whole = internal_energy_scalar(
                    self.params,
                    &InternalLoopScalar {
                        outer: au,
                        inner: au,
                        boundary_left5p: n,
                        boundary_left3p: n,
                        boundary_right5p: n,
                        boundary_right3p: n,
                        n1,
                        n2,
                    },
                );
                let mm = table.get(au, n.to_nuc5(), n.to_nuc5())
                    + table.get(au.reversed(), n.to_nuc5(), n.to_nuc5());
                self.shape[n1 * l + n2] = self.lambda * f64::from(whole.0 - mm);
                self.shape_e[n1 * l + n2] = whole.0 - mm;
            }
        }
    }

    /// Index the closed pair `key` at `o`, opened by `ep` and closed by `eq`,
    /// as an enclosing pair.
    fn index_outer(&mut self, key: PairKey, (ep, eq): (Edge, Edge), constraint: f64) {
        let table = &self.params.mismatch_interior;
        let (pn, qn) = (key.open, key.close);
        let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let (corners, ao_best) = if self.record {
            let firsts = self.firsts(ep.to);
            let lasts = if qn.pos > 0 {
                self.lasts(qn)
            } else {
                Vec::new()
            };
            let mut ao = Vec::with_capacity(firsts.len() * lasts.len());
            for l5 in &firsts {
                for r3 in &lasts {
                    let mm = table.get(pair, l5.nuc.to_nuc5(), r3.nuc.to_nuc5());
                    ao.push(self.lambda * f64::from(mm) - self.kk * (l5.cai + r3.cai));
                }
            }
            let best = ao.iter().copied().fold(f64::INFINITY, f64::min);
            (Some(Box::new(OuterCorners { firsts, lasts, ao })), best)
        } else {
            (None, self.outer_ao_best(pair, ep.to, qn))
        };
        let id = self.outers.len() as u32;
        self.by_close[qn.pos as usize].push(id);
        self.outers.push(Outer {
            pn,
            qn,
            ep,
            eq,
            pair,
            stat: constraint - self.kk * (ep.cai + eq.cai),
            corners,
            ao_best,
            out: f64::INFINITY,
            edge: f64::INFINITY,
            ins: f64::INFINITY,
        });
    }

    /// The stack, bulge and internal-loop closures around each `(o, key)` of
    /// `keys`, without recording: each pair's best outside through them, and
    /// how many enclosing pairs each loop family visited (counted per pair). A
    /// pure read of the enclosing pairs' finalized values.
    ///
    /// Every key opens at the same node, so the pairs differ only in their two
    /// nucleotides: they enclose the same pairs through the same strands, and
    /// the walk runs once for all of them, scoring each at every visit.
    ///
    /// The enclosing pairs are visited from [`Self::runs`], runs of
    /// [`Self::walk_close`] that share an opening position and an opening edge
    /// target, and so share the loop's shape and opening strand. A generic loop
    /// whose two strands each cross a codon boundary is the enclosing pair's
    /// outer half ([`Outer::ao_best`]), the loop's size and asymmetry
    /// ([`Self::shape`]) and the enclosed pair's inner half; only the least
    /// value is kept, so such a run is scored once from its least outer half
    /// ([`Run::best`]) and its pairs are not visited. The pairs of every other
    /// run are visited one by one, and a strand's corners are enumerated only
    /// for loops that do not split.
    fn singles_group(&self, o: Node, keys: &[PairKey]) -> (Vec<f64>, SinglesTally) {
        let a = keys[0].open.pos;
        let b = o.pos - 1;
        let l = self.single_span();
        let mut walk = GroupWalk::new(self, o, keys);
        let mut best = vec![f64::INFINITY; keys.len()];
        let q_top = (b + 1 + l as u32).min(self.dfa.seq_len);
        let p_lo = a.saturating_sub(l as u32);
        STRANDS.with_borrow_mut(|(opens, closes)| {
            opens.reset(StrandSlots::opening_len(l, self.dfa.stride()));
            // A closing strand runs from `o` to its closing node, so the
            // enclosed pairs at one `o`, which a worker takes in a row, share
            // them.
            if closes.owner != Some((self.id, o)) {
                closes.reset(StrandSlots::closing_len(l, self.dfa.stride()));
                closes.owner = Some((self.id, o));
            }
            let held = closes.len();
            for q in (b + 1)..q_top {
                let n2 = (q - o.pos) as usize;
                // n1 + n2 ≤ l − 1, and n1 = a − p − 1, so p ≥ a + n2 − l.
                let p_min = (a + n2 as u32).saturating_sub(l as u32).max(p_lo);
                let runs = &self.runs[q as usize];
                let start = runs.partition_point(|r| r.p < p_min);
                let list = &self.walk_close[q as usize];
                walk.close_split = self.separable(o, Node::new(q, 0));
                for run in runs[start..].iter().take_while(|r| r.p < a) {
                    walk.run(q, run, list, &mut best, (opens, closes));
                }
            }
            walk.tally.strands = (opens.len() + closes.len() - held) as u64;
        });
        (best, walk.tally)
    }

    /// Cut [`Self::walk_close`] at `q` into its [`Run`]s, each with its least
    /// split outer half. Every pair closing at `q` has its outer node at
    /// `q + 1`, so its outside value is final once that position is done.
    fn fill_runs(&mut self, q: u32) {
        let list = &self.walk_close[q as usize];
        let mut runs = Vec::new();
        let mut start = 0;
        for chunk in list.chunk_by(|x, y| (x.pn.pos, x.ep.to) == (y.pn.pos, y.ep.to)) {
            let finite = chunk.iter().filter(|x| x.out.is_finite());
            runs.push(Run {
                p: chunk[0].pn.pos,
                to: chunk[0].ep.to,
                start: start as u32,
                end: (start + chunk.len()) as u32,
                finite: finite.clone().count() as u32,
                best: finite
                    .map(|x| x.out + x.stat + x.ao_best)
                    .fold(f64::INFINITY, f64::min),
            });
            start += chunk.len();
        }
        self.runs[q as usize] = runs;
    }

    /// The least internal loop from `outer` to `inner` over every pair of
    /// strand corners, with the corners' CAI; each strand comes with its length.
    /// A loop that scores the same at every corner
    /// ([`OneByNRows::corner_free`]) is scored once.
    fn corner_min(
        &self,
        outer: BasePair,
        inner: BasePair,
        (lefts, n1): (&[Corner], usize),
        (rights, n2): (&[Corner], usize),
    ) -> f64 {
        let energy_at = |lc: &Corner, rc: &Corner| {
            internal_energy_scalar(
                self.params,
                &InternalLoopScalar {
                    outer,
                    inner,
                    boundary_left5p: lc.first,
                    boundary_left3p: lc.last,
                    boundary_right5p: rc.first,
                    boundary_right3p: rc.last,
                    n1,
                    n2,
                },
            )
        };
        let (Some(l0), Some(r0)) = (lefts.first(), rights.first()) else {
            return f64::INFINITY;
        };
        if self
            .one_by_n
            .corner_free(&LoopPairs { outer, inner }, n1, n2)
        {
            let energy = self.e(energy_at(l0, r0));
            return self.least_corners((lefts, rights), |_, _| energy);
        }
        self.least_corners((lefts, rights), |lc, rc| self.e(energy_at(lc, rc)))
    }

    /// The least `energy_at − kk·cai` over every pair of strand corners.
    fn least_corners(
        &self,
        (lefts, rights): (&[Corner], &[Corner]),
        energy_at: impl Fn(&Corner, &Corner) -> f64,
    ) -> f64 {
        let mut best = f64::INFINITY;
        for lc in lefts {
            for rc in rights {
                relax(&mut best, energy_at(lc, rc) - self.kk * (lc.cai + rc.cai));
            }
        }
        best
    }

    /// The stack, bulge and internal-loop closures around `(o, key)`, recording:
    /// returns the pair's best outside through them and hands each choice's
    /// value to its pieces.
    ///
    /// The enclosing pairs are visited from [`Self::by_close`], so every visit
    /// is a pair that exists. A generic loop whose two strands each cross a
    /// codon boundary splits into an outer half, which belongs to the
    /// enclosing pair and was scored once in [`Self::index_outers`], and an
    /// inner half, which belongs to this pair and is scored once here — so such
    /// a visit costs a few additions. Its pieces are handed on per side: the
    /// enclosing pair's accumulates in [`Outer::ins`] and is handed out in
    /// [`Self::outer_pieces`], this pair's below, and each strand's middle
    /// through the span table by position.
    fn singles_recorded(&mut self, o: Node, key: PairKey, p_in: f64) -> f64 {
        let dfa = self.dfa;
        let a = key.open.pos;
        let b = o.pos - 1;
        let l = self.single_span();
        let inner_pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let inner = self.inner_half(o, key, inner_pair);
        let l_mid_to = if a > 0 {
            Some(self.boundary_before(key.open))
        } else {
            None
        };
        let r_mid_from = self.boundary_after(o);
        let q_top = (b + 1 + l as u32).min(dfa.seq_len);
        let p_lo = a.saturating_sub(l as u32);
        // Opening sides by their edge's two nodes and closing sides by (q, node),
        // built as first reached. An opening is named by its target rather than
        // its nucleotide: two out-edges of one node can spell the same one.
        let mut opens: FxHashMap<(Node, Node), OpenSide> = FxHashMap::default();
        let mut closes: FxHashMap<Node, CloseSide> = FxHashMap::default();
        let mut outer_best = f64::INFINITY;
        let mut best = f64::INFINITY;
        for q in (b + 1)..q_top {
            let n2 = (q - o.pos) as usize;
            // n1 + n2 ≤ l − 1, and n1 = a − p − 1, so p ≥ a + n2 − l.
            let p_min = (a + n2 as u32).saturating_sub(l as u32).max(p_lo);
            let list = std::mem::take(&mut self.walk_close[q as usize]);
            let start = list.partition_point(|x| x.pn.pos < p_min);
            for x in &list[start..] {
                let (p, id) = (x.pn.pos, x.id);
                if p >= a {
                    break;
                }
                if !x.out.is_finite() {
                    continue;
                }
                let (pn, ep, qn) = (x.pn, &x.ep, x.qn);
                let n1 = (a - ep.to.pos) as usize;
                let open = opens.entry((pn, ep.to)).or_insert_with(|| OpenSide {
                    left_cai: self.fill(ep.to, key.open),
                    lefts: if n1 > 0 {
                        self.corners(ep.to, key.open)
                    } else {
                        Vec::new()
                    },
                    split: self.separable(ep.to, key.open),
                    corner: Vec::new(),
                    stack: f64::INFINITY,
                });
                let Some(left_cai) = open.left_cai else {
                    continue;
                };
                let close = closes.entry(qn).or_insert_with(|| CloseSide {
                    right_cai: self.fill(o, qn),
                    rights: if n2 > 0 {
                        self.corners(o, qn)
                    } else {
                        Vec::new()
                    },
                    split: self.separable(o, qn),
                    corner: Vec::new(),
                    stack: f64::INFINITY,
                });
                let base = x.out + x.stat;
                let total = if n1 > 0 && n2 > 0 {
                    match inner.as_ref() {
                        Some(inner) if is_generic(n1, n2) && open.split && close.split => {
                            let shape = self.shape[n1 * l + n2];
                            let here = base + x.ao_best + shape;
                            relax(&mut outer_best, here);
                            let total = here + inner.best;
                            let ins = shape + inner.best + p_in;
                            relax(&mut self.outers[id as usize].ins, ins);
                            if let Some(to) = l_mid_to {
                                let from = self.boundary_after(ep.to);
                                self.span(from, to, total + p_in);
                            }
                            let to = self.boundary_before(qn);
                            self.span(r_mid_from, to, total + p_in);
                            total
                        }
                        _ => {
                            if open.corner.is_empty() {
                                open.corner = vec![f64::INFINITY; open.lefts.len()];
                            }
                            if close.corner.is_empty() {
                                close.corner = vec![f64::INFINITY; close.rights.len()];
                            }
                            let mut here = f64::INFINITY;
                            for (i, lc) in open.lefts.iter().enumerate() {
                                for (j, rc) in close.rights.iter().enumerate() {
                                    let energy = internal_energy_scalar(
                                        self.params,
                                        &InternalLoopScalar {
                                            outer: x.pair,
                                            inner: inner_pair,
                                            boundary_left5p: lc.first,
                                            boundary_left3p: lc.last,
                                            boundary_right5p: rc.first,
                                            boundary_right3p: rc.last,
                                            n1,
                                            n2,
                                        },
                                    );
                                    let v = base + self.lambda * f64::from(energy.0)
                                        - self.kk * (lc.cai + rc.cai);
                                    relax(&mut here, v);
                                    relax(&mut open.corner[i], v);
                                    relax(&mut close.corner[j], v);
                                }
                            }
                            here
                        }
                    }
                } else {
                    let Some(right_cai) = close.right_cai else {
                        continue;
                    };
                    let energy = if n1 == 0 && n2 == 0 {
                        stack_energy(
                            self.params,
                            &StackPair {
                                outer: x.pair,
                                inner: inner_pair,
                            },
                        )
                    } else {
                        bulge_energy(
                            self.params,
                            &BulgeLoop {
                                outer: x.pair,
                                inner: inner_pair,
                                size: n1 + n2,
                            },
                        )
                    };
                    let v =
                        base + self.lambda * f64::from(energy.0) - self.kk * (left_cai + right_cai);
                    relax(&mut open.stack, v);
                    relax(&mut close.stack, v);
                    v
                };
                relax(&mut best, total);
                relax(&mut self.outers[id as usize].edge, total + p_in);
            }
            self.walk_close[q as usize] = list;
        }
        for (&(_, ep_to), open) in &opens {
            for (lc, &x) in open.lefts.iter().zip(&open.corner) {
                self.corner(lc, x + p_in);
            }
            self.span(ep_to, key.open, open.stack + p_in);
        }
        for (qn, close) in &closes {
            for (rc, &x) in close.rights.iter().zip(&close.corner) {
                self.corner(rc, x + p_in);
            }
            self.span(o, *qn, close.stack + p_in);
        }
        if let Some(inner) = &inner {
            if outer_best.is_finite() {
                for (ri, r5) in inner.firsts.iter().enumerate() {
                    for (li, l3) in inner.lasts.iter().enumerate() {
                        let v = outer_best + inner.by_corner[ri * inner.lasts.len() + li] + p_in;
                        self.side(r5, v);
                        self.side(l3, v);
                    }
                }
            }
        }
        best
    }

    /// Hand every enclosing pair's own pieces their value: its two edges, and
    /// the outer corners of every split loop it closes, each with the best
    /// inner remainder any enclosed pair offered it.
    fn outer_pieces(&mut self) {
        for id in 0..self.outers.len() {
            let x = &self.outers[id];
            if !x.out.is_finite() {
                continue;
            }
            let split = x.out + x.stat + x.ins;
            let edge = x.edge.min(split + x.ao_best);
            let (pn, ep, qn, eq) = (x.pn, x.ep, x.qn, x.eq);
            self.edge(pn, ep.nuc, edge);
            self.edge(qn, eq.nuc, edge);
            if !split.is_finite() {
                continue;
            }
            let x = self.outers[id]
                .corners
                .as_deref()
                .expect("the recording pass keeps every pair's corners");
            let n = x.lasts.len();
            let pieces: Vec<(Side, f64)> = x
                .firsts
                .iter()
                .enumerate()
                .flat_map(|(i, l5)| {
                    x.lasts.iter().enumerate().flat_map(move |(j, r3)| {
                        let v = split + x.ao[i * n + j];
                        [(l5.clone(), v), (r3.clone(), v)]
                    })
                })
                .collect();
            for (side, v) in pieces {
                self.side(&side, v);
            }
        }
    }

    /// Whether a strand `from → to` of at least two nucleotides crosses a codon
    /// boundary between its first and last nucleotide, so that its best middle
    /// splits into a part beside each end.
    fn separable(&self, from: Node, to: Node) -> bool {
        to.pos >= from.pos + 2
            && self.dfa.next_codon_boundary(from.pos + 1) <= self.dfa.codon_start(to.pos - 1)
            && self.dfa.halves_combine(to)
    }

    /// The first codon boundary after a strand's first nucleotide, from `from`.
    fn boundary_after(&self, from: Node) -> Node {
        Node::new(self.dfa.next_codon_boundary(from.pos + 1), 0)
    }

    /// The last codon boundary before a strand's last nucleotide, into `to`.
    fn boundary_before(&self, to: Node) -> Node {
        Node::new(self.dfa.codon_start(to.pos - 1), 0)
    }

    /// The least outer half over the split-loop corners of a pair opening by
    /// an edge into `from` and closing into `to` — [`Outer::ao_best`] — scored
    /// without listing the corners.
    fn outer_ao_best(&self, pair: BasePair, from: Node, to: Node) -> f64 {
        if to.pos == 0 {
            return f64::INFINITY;
        }
        let table = &self.params.mismatch_interior;
        let mut best = f64::INFINITY;
        for e5 in self.dfa.right(from) {
            let Some(head) = self.dfa.span_head_cai(e5.to) else {
                continue;
            };
            let l5 = e5.cai + head;
            for e3 in self.dfa.left(to) {
                let Some(tail) = self.dfa.span_tail_cai(e3.to) else {
                    continue;
                };
                let mm = table.get(pair, e5.nuc.to_nuc5(), e3.nuc.to_nuc5());
                best = best.min(self.lambda * f64::from(mm) - self.kk * (l5 + (tail + e3.cai)));
            }
        }
        best
    }

    /// A separable strand's first-nucleotide choices out of `from`, each with
    /// its edge and the best path on to the boundary after it.
    fn firsts(&self, from: Node) -> Vec<Side> {
        let to = self.boundary_after(from);
        self.dfa
            .right(from)
            .iter()
            .filter_map(|e| {
                let fill = self.dfa.span_head_cai(e.to)?;
                Some(Side {
                    nuc: e.nuc,
                    cai: e.cai + fill,
                    from,
                    fill: (e.to, to),
                })
            })
            .collect()
    }

    /// A separable strand's last-nucleotide choices into `to`, each with its
    /// edge and the best path to it from the boundary before it.
    fn lasts(&self, to: Node) -> Vec<Side> {
        if to.pos == 0 {
            return Vec::new();
        }
        let from = self.boundary_before(to);
        self.dfa
            .left(to)
            .iter()
            .filter_map(|e| {
                let fill = self.dfa.span_tail_cai(e.to)?;
                Some(Side {
                    nuc: e.nuc,
                    cai: fill + e.cai,
                    from: e.to,
                    fill: (from, e.to),
                })
            })
            .collect()
    }

    /// The inner half of every split loop around `(o, key)`: its value for each
    /// pair of corners beside the pair, and the least of them.
    fn inner_half(&self, o: Node, key: PairKey, inner_pair: BasePair) -> Option<InnerHalf> {
        if key.open.pos == 0 {
            return None;
        }
        let firsts = self.firsts(o);
        let lasts = self.lasts(key.open);
        let table = &self.params.mismatch_interior;
        let reversed = inner_pair.reversed();
        let mut by_corner = Vec::with_capacity(firsts.len() * lasts.len());
        for r5 in &firsts {
            for l3 in &lasts {
                let mm = table.get(reversed, r5.nuc.to_nuc5(), l3.nuc.to_nuc5());
                by_corner.push(self.lambda * f64::from(mm) - self.kk * (r5.cai + l3.cai));
            }
        }
        let best = by_corner.iter().copied().fold(f64::INFINITY, f64::min);
        best.is_finite().then_some(InnerHalf {
            firsts,
            lasts,
            by_corner,
            best,
        })
    }

    fn side(&mut self, s: &Side, v: f64) {
        self.pieces(
            &[Piece::Edge(s.from, s.nuc), Piece::Fill(s.fill.0, s.fill.1)],
            v,
        );
    }

    /// `Multi` tails: `P = Multi + closing cost`.
    fn multi_tails(&mut self, pos: u32) {
        for &jn in self.dfa.nodes_at(pos) {
            for (key, _) in self.t.multi_states(jn) {
                let Some(constraint) = self.constraint(key.open.pos, key.close.pos) else {
                    continue;
                };
                let Some(&h) = self.out_p[self.layout.ix(jn)].get(&key) else {
                    continue;
                };
                let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                let cost = Energy(self.params.ml.closing_37) + self.stem(closing);
                let slot = self.out_multi[self.layout.ix(jn)]
                    .entry(key)
                    .or_insert(f64::INFINITY);
                relax(slot, h + self.lambda * f64::from(cost.0) + constraint);
            }
        }
    }

    /// Every hairpin choice, handed its full value: hairpins have no tail, so
    /// they are read only once every `P` cell's outside is final.
    fn hairpins(&mut self) {
        for a in 0..self.dfa.seq_len {
            for &an in self.dfa.nodes_at(a) {
                for eo in self.dfa.right(an).to_vec() {
                    for b in (a + MIN_HAIRPIN_LOOP + 1)..self.dfa.seq_len {
                        for &bn in self.dfa.nodes_at(b) {
                            for ec in self.dfa.right(bn).to_vec() {
                                self.hairpin(an, eo, bn, ec);
                            }
                        }
                    }
                }
            }
        }
    }

    fn hairpin(&mut self, an: Node, eo: Edge, bn: Node, ec: Edge) {
        let closing = pair_from_nucleotides(eo.nuc, ec.nuc);
        if closing == BasePair::NN {
            return;
        }
        let key = PairKey {
            open: an,
            close: bn,
            opening: eo.opening,
            nuc_close: ec.nuc,
        };
        let Some(&h) = self.out_p[self.layout.ix(ec.to)].get(&key) else {
            return;
        };
        let Some(constraint) = self.constraint(an.pos, bn.pos) else {
            return;
        };
        let base = h + constraint - self.kk * (eo.cai + ec.cai);
        let ends = [Piece::Edge(an, eo.nuc), Piece::Edge(bn, ec.nuc)];
        let size = (bn.pos - eo.to.pos) as usize;
        if matches!(size, 3 | 4 | 6) {
            for (cai, seq) in self.dfa.paths_between(eo.to, bn) {
                let energy = hairpin_energy(
                    self.params,
                    &HairpinLoop {
                        closing,
                        loop_seq: &seq,
                    },
                );
                let v = base + self.e(energy) - self.kk * cai;
                let mut pieces = Vec::with_capacity(seq.len());
                let nucs: Vec<u8> = seq.iter().map(|&n| n as u8).collect();
                let mut at = eo.to;
                for e in self
                    .dfa
                    .spell_to(at, &nucs, Some(bn))
                    .expect("a spelled path follows the lattice")
                {
                    pieces.push(Piece::Edge(at, e.nuc));
                    at = e.to;
                }
                self.pieces(&ends, v);
                self.pieces(&pieces, v);
            }
            return;
        }
        for c in self.corners(eo.to, bn) {
            let energy = hairpin_energy_scalar(self.params, closing, size, c.first, c.last);
            let v = base + self.e(energy) - self.kk * c.cai;
            self.pieces(&ends, v);
            self.corner(&c, v);
        }
    }

    /// A strand's terminal-mismatch choices from `from` to `to` (length `≥ 1`).
    fn corners(&self, from: Node, to: Node) -> Vec<Corner> {
        let mut out = Vec::new();
        self.corners_into(from, to, &mut out);
        out
    }

    /// [`Self::corners`] into `out`, cleared first and its capacity kept.
    fn corners_into(&self, from: Node, to: Node, out: &mut Vec<Corner>) {
        out.clear();
        if to.pos - from.pos == 1 {
            for e in self.dfa.right(from).iter().filter(|e| e.to == to) {
                out.push(Corner {
                    first: e.nuc,
                    last: e.nuc,
                    cai: e.cai,
                    pieces: [Some(Piece::Edge(from, e.nuc)), None, None],
                });
            }
            return;
        }
        for first in self.dfa.right(from) {
            for last in self.dfa.left(to) {
                let Some(mid) = self.fill(first.to, last.to) else {
                    continue;
                };
                out.push(Corner {
                    first: first.nuc,
                    last: last.nuc,
                    cai: first.cai + mid + last.cai,
                    pieces: [
                        Some(Piece::Edge(from, first.nuc)),
                        Some(Piece::Fill(first.to, last.to)),
                        Some(Piece::Edge(last.to, last.nuc)),
                    ],
                });
            }
        }
    }

    // ---- the fills ---------------------------------------------------------

    /// The best path through `e` within the whole codon starting at `cs`,
    /// against that codon's best.
    fn codon_penalty(&self, cs: u32, from: Node, e: &Edge) -> Option<f64> {
        let (entry, exit) = (Node::new(cs, 0), Node::new(cs + 3, 0));
        let best = self.fill(entry, exit)?;
        let x = self.fill(entry, from)?;
        let y = self.fill(e.to, exit)?;
        Some(self.kk * (best - (x + e.cai + y)))
    }

    /// Hand every span's value to the edges inside it: the partial codons at
    /// its ends directly, the whole codons through a suffix minimum over spans.
    fn resolve_spans(&mut self) {
        let (s, end) = (self.dfa.cds_start(), self.dfa.cds_end());
        let positions = self.dfa.seq_len as usize + 1;
        let mut by_pos = vec![f64::INFINITY; positions * positions];
        for ai in 0..self.width {
            for bi in 0..self.width {
                let v = self.spans[ai * self.width + bi];
                if !v.is_finite() {
                    continue;
                }
                let (a, b) = (self.dfa.node_of(ai), self.dfa.node_of(bi));
                relax(&mut by_pos[a.pos as usize * positions + b.pos as usize], v);
                let Some(best) = self.fill(a, b) else {
                    continue;
                };
                for pos in a.pos.max(s)..b.pos.min(end) {
                    let cs = self.dfa.codon_start(pos);
                    if cs >= a.pos && cs + 3 <= b.pos {
                        continue;
                    }
                    for &n in self.dfa.nodes_at(pos) {
                        for e in self.dfa.right(n) {
                            if e.to.pos > b.pos {
                                continue;
                            }
                            let (Some(x), Some(y)) = (self.fill(a, n), self.fill(e.to, b)) else {
                                continue;
                            };
                            let v = v + self.kk * (best - (x + e.cai + y));
                            relax(&mut self.by_edge[self.layout.ix(n) * 4 + e.nuc as usize], v);
                        }
                    }
                }
            }
        }
        // suffix[a][x] = min over b ≥ x of by_pos[a][b]
        let mut suffix = vec![f64::INFINITY; positions * (positions + 1)];
        for a in 0..positions {
            for x in (0..positions).rev() {
                let here = by_pos[a * positions + x];
                let next = suffix[a * (positions + 1) + x + 1];
                suffix[a * (positions + 1) + x] = here.min(next);
            }
        }
        let mut cs = s;
        while cs + 3 <= end {
            let mut w = f64::INFINITY;
            for a in 0..=cs as usize {
                w = w.min(suffix[a * (positions + 1) + cs as usize + 3]);
            }
            if w.is_finite() {
                self.whole_codon(cs, w);
            }
            cs += 3;
        }
    }

    /// Hand `w`, the best value of anything spanning the whole codon at `cs`,
    /// to each of its edges with that edge's penalty.
    fn whole_codon(&mut self, cs: u32, w: f64) {
        for pos in cs..cs + 3 {
            for &n in self.dfa.nodes_at(pos) {
                for e in self.dfa.right(n) {
                    if let Some(pen) = self.codon_penalty(cs, n, e) {
                        relax(
                            &mut self.by_edge[self.layout.ix(n) * 4 + e.nuc as usize],
                            w + pen,
                        );
                    }
                }
            }
        }
    }

    /// The whole codons of multi-loop 3′ flanks closed through the shared pool:
    /// an opening active before the codon plus a frontier after it.
    fn pool_codons(&mut self) {
        let (s, end) = (self.dfa.cds_start(), self.dfa.cds_end());
        let mut cs = s;
        while cs + 3 <= end {
            let before = &self.pool_at[cs as usize];
            let after = &self.frontier_at[(cs + 3) as usize];
            let w = before
                .iter()
                .zip(after)
                .map(|(x, y)| x + y)
                .fold(f64::INFINITY, f64::min);
            if w.is_finite() {
                self.whole_codon(cs, w);
            }
            cs += 3;
        }
    }
}

/// A strand's terminal-mismatch choice with the path pieces it spells.
#[derive(Clone, Copy)]
struct Corner {
    first: Nucleotide,
    last: Nucleotide,
    cai: f64,
    pieces: [Option<Piece>; 3],
}

/// Seconds per phase of the pass, reported to stderr when
/// `DERNA_OUTSIDE_PHASES` is set and not measured otherwise.
struct Phases {
    on: bool,
    spent: Vec<(&'static str, f64)>,
}

impl Phases {
    fn from_env() -> Self {
        Self {
            on: std::env::var_os("DERNA_OUTSIDE_PHASES").is_some(),
            spent: Vec::new(),
        }
    }

    fn time<T>(&mut self, name: &'static str, body: impl FnOnce() -> T) -> T {
        if !self.on {
            return body();
        }
        let started = std::time::Instant::now();
        let out = body();
        let s = started.elapsed().as_secs_f64();
        match self.spent.iter_mut().find(|(n, _)| *n == name) {
            Some((_, t)) => *t += s,
            None => self.spent.push((name, s)),
        }
        out
    }

    fn report(&self) {
        for (name, s) in &self.spent {
            eprintln!("  outside phase {name:<14} {s:>8.3}s");
        }
    }
}

impl Marginals {
    /// The best lattice-unit objective of any design that does not spell
    /// `codon` at codon position `q`: the least edge marginal over the edges of
    /// `q`'s sub-lattice off `codon`'s path. `None` when every design spells it.
    pub(crate) fn excluding(&self, dfa: &Dfa, q: usize, codon: [u8; 3]) -> Option<f64> {
        let base = dfa.cds_start() + 3 * q as u32;
        let mut on_path = [(Node::new(0, 0), 0u8); 3];
        let mut at = Node::new(base, 0);
        let path = dfa
            .spell_to(at, &codon, None)
            .expect("the codon is admitted at this position");
        for (k, (e, &nuc)) in path.iter().zip(&codon).enumerate() {
            on_path[k] = (at, nuc);
            at = e.to;
        }
        let mut best = f64::INFINITY;
        for pos in base..base + 3 {
            for &n in dfa.nodes_at(pos) {
                for e in dfa.right(n) {
                    if !on_path.contains(&(n, e.nuc as u8)) {
                        best = best.min(self.by_edge[dfa.ix(n) * 4 + e.nuc as usize]);
                    }
                }
            }
        }
        best.is_finite().then_some(best)
    }
}

/// One end of a separable strand: the nucleotide beside the loop's pair, the
/// edge that spells it, and the best path between it and the strand's nearest
/// codon boundary, with their joint adaptation.
#[derive(Clone)]
struct Side {
    nuc: Nucleotide,
    cai: f64,
    /// The node the edge leaves.
    from: Node,
    fill: (Node, Node),
}

/// The inner half of the split loops around one pair.
struct InnerHalf {
    /// First nucleotides of the 3′ strand, out of the pair's outer node.
    firsts: Vec<Side>,
    /// Last nucleotides of the 5′ strand, into the pair's opening node.
    lasts: Vec<Side>,
    /// `firsts × lasts`, row-major: the inner mismatch and the two ends'
    /// adaptation.
    by_corner: Vec<f64>,
    best: f64,
}

/// The internal-loop shapes the energy model scores as size, asymmetry and
/// the two terminal mismatches alone: both strands at least two long, and not
/// one of the tabulated small shapes.
fn is_generic(n1: usize, n2: usize) -> bool {
    n1 >= 2 && n2 >= 2 && !matches!((n1, n2), (2, 2) | (2, 3) | (3, 2))
}

/// A closed pair as the enclosing pair of single-branch loops.
struct Outer {
    pn: Node,
    qn: Node,
    ep: Edge,
    eq: Edge,
    pair: BasePair,
    /// The constraint's adjustment and the two edges' adaptation.
    stat: f64,
    /// The outer corners of its split loops, kept only by the recording pass:
    /// the record-free pass reads their least, [`Self::ao_best`], alone, and
    /// one copy per closed pair is most of what it would otherwise hold.
    corners: Option<Box<OuterCorners>>,
    ao_best: f64,
    /// This pair's outside, once final.
    out: f64,
    /// The best full value of any loop it closes, for its two edges.
    edge: f64,
    /// The best inner remainder of any split loop it closes: shape, inner
    /// half and enclosed pair's inside.
    ins: f64,
}

/// The outer corners of the split loops one pair closes.
struct OuterCorners {
    /// First nucleotides of the 5′ strand, out of `ep`, for split loops.
    firsts: Vec<Side>,
    /// Last nucleotides of the 3′ strand, into `qn`, for split loops.
    lasts: Vec<Side>,
    /// `firsts × lasts`, row-major: the outer mismatch and the two ends'
    /// adaptation.
    ao: Vec<f64>,
}

/// The fields of an [`Outer`] the single-branch walks read, kept in
/// walk order beside it.
struct Walk {
    /// The pair's index in [`OutsideScan::outers`], for the recording walk's
    /// writes.
    id: u32,
    pn: Node,
    qn: Node,
    ep: WalkEdge,
    pair: BasePair,
    stat: f64,
    ao_best: f64,
    out: f64,
}

/// The opening edge's target, all the walk reads of it.
struct WalkEdge {
    to: Node,
}

impl Walk {
    fn of(id: u32, x: &Outer) -> Self {
        Self {
            id,
            pn: x.pn,
            qn: x.qn,
            ep: WalkEdge { to: x.ep.to },
            pair: x.pair,
            stat: x.stat,
            ao_best: x.ao_best,
            out: x.out,
        }
    }
}

/// A run of [`OutsideScan::walk_close`] at one closing position: the pairs
/// that open at one position by an edge into one node. Around any enclosed
/// pair they close loops of one shape through one opening strand, so either
/// every one of them splits or none does.
#[derive(Clone)]
struct Run {
    p: u32,
    /// The opening edge's target, where the opening strand starts.
    to: Node,
    start: u32,
    end: u32,
    /// Pairs with a finite outside value.
    finite: u32,
    /// The least split outer half, `out + stat + ao_best`, over those pairs.
    best: f64,
}

/// The walk of [`OutsideScan::singles_group`]: the enclosed pairs' shared
/// opening and their inner halves, and what the walk visited.
struct GroupWalk<'s, 'a> {
    scan: &'s OutsideScan<'a>,
    o: Node,
    open_node: Node,
    keys: &'s [PairKey],
    /// Each key's pair and the best inner half of a split loop around it.
    inner: Vec<(BasePair, f64)>,
    /// Whether every key has an inner half, so a split run serves them all.
    all_inner: bool,
    /// The enclosing pair's part of a generic loop scored over its corners,
    /// per corner pair (see [`Self::outer_corners`]).
    corners: Vec<OuterCorner>,
    /// Whether the closing strand of the run being walked crosses a codon
    /// boundary; it depends only on the closing position.
    close_split: bool,
    tally: SinglesTally,
}

impl<'s, 'a> GroupWalk<'s, 'a> {
    fn new(scan: &'s OutsideScan<'a>, o: Node, keys: &'s [PairKey]) -> Self {
        let inner: Vec<(BasePair, f64)> = keys
            .iter()
            .map(|&key| {
                let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                let best = scan
                    .inner_half(o, key, pair)
                    .map_or(f64::INFINITY, |h| h.best);
                (pair, best)
            })
            .collect();
        Self {
            scan,
            o,
            open_node: keys[0].open,
            keys,
            all_inner: inner.iter().all(|&(_, h)| h.is_finite()),
            inner,
            close_split: false,
            corners: Vec::new(),
            tally: SinglesTally::default(),
        }
    }

    fn per_key(&self) -> u64 {
        self.keys.len() as u64
    }

    /// The loops through one run of enclosing pairs closing at `q`: scored once
    /// from the run's least outer half when they all split, else pair by pair.
    fn run(
        &mut self,
        q: u32,
        run: &Run,
        list: &[Walk],
        best: &mut [f64],
        (opens, closes): (&mut StrandSlots, &mut StrandSlots),
    ) {
        self.tally.walked += self.per_key() * u64::from(run.end - run.start);
        if run.finite == 0 {
            return;
        }
        let shape = (
            (self.open_node.pos - run.to.pos) as usize,
            (q - self.o.pos) as usize,
        );
        let pairs = &list[run.start as usize..run.end as usize];
        if !(self.all_inner && self.splits(run.to, shape)) {
            for x in pairs {
                self.pair(x, shape, best, (opens, closes));
            }
            return;
        }
        let (scan, open_node) = (self.scan, self.open_node);
        let open = opens.get_or_fill(
            StrandSlots::opening(open_node.pos, run.to, self.scan.dfa.stride()),
            |s| {
                s.refill(scan, run.to, open_node);
            },
        );
        if open.cai.is_none() {
            return;
        }
        let half = run.best + scan.shape[shape.0 * scan.single_span() + shape.1];
        if scan.validate_split {
            self.check_run(q, run, (half, pairs), shape, (opens, closes));
        }
        self.tally.split += self.per_key() * u64::from(run.finite);
        for (slot, &(_, inner)) in best.iter_mut().zip(&self.inner) {
            relax(slot, half + inner);
        }
    }

    /// Asserts a split run's value is what walking its pairs gives, bit for
    /// bit, each pair's split checked against its corners on the way.
    fn check_run(
        &mut self,
        q: u32,
        run: &Run,
        (half, pairs): (f64, &[Walk]),
        shape: (usize, usize),
        strands: (&mut StrandSlots, &mut StrandSlots),
    ) {
        let (opens, closes) = strands;
        let held = self.tally;
        let mut each = vec![f64::INFINITY; self.keys.len()];
        for x in pairs {
            self.pair(x, shape, &mut each, (opens, closes));
        }
        self.tally = held;
        for (&walked, &(_, inner)) in each.iter().zip(&self.inner) {
            assert!(
                walked.to_bits() == (half + inner).to_bits(),
                "a split run at {q} from {:?} scores {}, its pairs {walked}",
                run.to,
                half + inner,
            );
        }
    }

    /// Whether a loop of `shape` opening by an edge into `to` splits: generic,
    /// and both strands cross a codon boundary.
    fn splits(&self, to: Node, (n1, n2): (usize, usize)) -> bool {
        self.close_split && is_generic(n1, n2) && self.scan.separable(to, self.open_node)
    }

    /// The loop through one enclosing pair `x`, around every key.
    fn pair(
        &mut self,
        x: &Walk,
        shape: (usize, usize),
        best: &mut [f64],
        (opens, closes): (&mut StrandSlots, &mut StrandSlots),
    ) {
        if !x.out.is_finite() {
            return;
        }
        let (scan, o, open_node) = (self.scan, self.o, self.open_node);
        let open = opens.get_or_fill(
            StrandSlots::opening(open_node.pos, x.ep.to, self.scan.dfa.stride()),
            |s| {
                s.refill(scan, x.ep.to, open_node);
            },
        );
        let Some(left_cai) = open.cai else {
            return;
        };
        let close = closes.get_or_fill(
            StrandSlots::closing(o.pos - 1, x.qn, self.scan.dfa.stride()),
            |s| {
                s.refill(scan, o, x.qn);
            },
        );
        if shape.0 == 0 || shape.1 == 0 {
            if let Some(right_cai) = close.cai {
                self.stack_bulge(x, shape, left_cai + right_cai, best);
            }
            return;
        }
        self.internal(x, shape, (open, close), best);
    }

    /// A stack or bulge from `x` around every key; `cai` is both strands'.
    fn stack_bulge(&mut self, x: &Walk, (n1, n2): (usize, usize), cai: f64, best: &mut [f64]) {
        let scan = self.scan;
        let base = x.out + x.stat;
        self.tally.stack_bulge += self.per_key();
        for (slot, &(inner_pair, _)) in best.iter_mut().zip(&self.inner) {
            let energy = if n1 == 0 && n2 == 0 {
                stack_energy(
                    scan.params,
                    &StackPair {
                        outer: x.pair,
                        inner: inner_pair,
                    },
                )
            } else {
                bulge_energy(
                    scan.params,
                    &BulgeLoop {
                        outer: x.pair,
                        inner: inner_pair,
                        size: n1 + n2,
                    },
                )
            };
            relax(slot, base + scan.e(energy) - scan.kk * cai);
        }
    }

    /// An internal loop from `x` around every key: from its two halves when it
    /// splits, else over every pair of its strands' corners.
    fn internal(
        &mut self,
        x: &Walk,
        (n1, n2): (usize, usize),
        (open, close): (&mut Strand, &mut Strand),
        best: &mut [f64],
    ) {
        let (scan, o, open_node) = (self.scan, self.o, self.open_node);
        let base = x.out + x.stat;
        let generic = is_generic(n1, n2);
        let splits = generic && open.split && close.split;
        let outer_half = base + x.ao_best + scan.shape[n1 * scan.single_span() + n2];
        let mut outer_ready = false;
        for (k, (slot, &(inner_pair, inner_best))) in best.iter_mut().zip(&self.inner).enumerate() {
            let total = if inner_best.is_finite() && splits {
                self.tally.split += 1;
                let v = outer_half + inner_best;
                if scan.validate_split {
                    let lefts = open.corners(scan, x.ep.to, open_node);
                    let rights = close.corners(scan, o, x.qn);
                    let direct =
                        base + scan.corner_min(x.pair, inner_pair, (lefts, n1), (rights, n2));
                    let key = self.keys[k];
                    assert!(
                        (v - direct).abs() <= 1e-6 * direct.abs().max(1.0),
                        "a split {n1}x{n2} loop around {key:?} at {o:?} scores {v}, \
                         its corners {direct}",
                    );
                }
                v
            } else {
                self.tally.corners += 1;
                self.tally.corners_generic += u64::from(generic);
                let lefts = open.corners(scan, x.ep.to, open_node);
                let rights = close.corners(scan, o, x.qn);
                self.tally.corner_pairs += (lefts.len() * rights.len()) as u64;
                if generic {
                    if !outer_ready {
                        Self::outer_corners(
                            scan,
                            &mut self.corners,
                            x.pair,
                            (n1, n2),
                            (lefts, rights),
                        );
                        outer_ready = true;
                    }
                    let least = self.generic_corner_min(inner_pair);
                    if scan.validate_split {
                        let direct = scan.corner_min(x.pair, inner_pair, (lefts, n1), (rights, n2));
                        assert_eq!(
                            least.to_bits(),
                            direct.to_bits(),
                            "a {n1}x{n2} loop around {inner_pair:?} from its outer corners"
                        );
                    }
                    base + least
                } else {
                    base + scan.corner_min(x.pair, inner_pair, (lefts, n1), (rights, n2))
                }
            };
            relax(slot, total);
        }
    }
}

/// One corner pair of a generic loop with the enclosing pair's part of its
/// score: the size, asymmetry and outer mismatch (the energy model's integer),
/// the corners' weighted CAI, and the two nucleotides the inner mismatch reads.
#[derive(Clone, Copy)]
struct OuterCorner {
    energy: i32,
    cai: f64,
    r5: Nuc5,
    l3: Nuc5,
}

impl GroupWalk<'_, '_> {
    /// The enclosing pair's part of a generic `(n1, n2)` loop closed by
    /// `outer`, for each corner pair, in [`OutsideScan::corner_min`]'s order.
    fn outer_corners(
        scan: &OutsideScan<'_>,
        corners: &mut Vec<OuterCorner>,
        outer: BasePair,
        (n1, n2): (usize, usize),
        (lefts, rights): (&[Corner], &[Corner]),
    ) {
        let table = &scan.params.mismatch_interior;
        let shape = scan.shape_e[n1 * scan.single_span() + n2];
        corners.clear();
        for lc in lefts {
            for rc in rights {
                corners.push(OuterCorner {
                    energy: shape + table.get(outer, lc.first.to_nuc5(), rc.last.to_nuc5()),
                    cai: scan.kk * (lc.cai + rc.cai),
                    r5: rc.first.to_nuc5(),
                    l3: lc.last.to_nuc5(),
                });
            }
        }
    }

    /// [`OutsideScan::corner_min`] of a generic loop around `inner`, from
    /// [`Self::outer_corners`]: the same integer energy and the same sums.
    fn generic_corner_min(&self, inner: BasePair) -> f64 {
        let scan = self.scan;
        let table = &scan.params.mismatch_interior;
        let inner_rev = inner.reversed();
        let mut best = f64::INFINITY;
        for c in &self.corners {
            let energy = Energy(c.energy + table.get(inner_rev, c.r5, c.l3));
            relax(&mut best, scan.e(energy) - c.cai);
        }
        best
    }
}

/// Enclosing pairs visited by the record-free single-branch consumers, by how
/// each loop was scored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SinglesTally {
    /// Stacks and bulges, scored directly.
    pub stack_bulge: u64,
    /// Generic internal loops scored from their two halves.
    pub split: u64,
    /// Internal loops scored over every corner pair.
    pub corners: u64,
    /// Of those, generic loops that did not split because a strand stays inside
    /// one codon; the rest are the special shapes.
    pub corners_generic: u64,
    /// Enclosing pairs walked, including those with no outside value.
    pub walked: u64,
    /// Corner pairs the loops scored over their corners evaluated.
    pub corner_pairs: u64,
    /// Strands whose fill and separability were computed: one per distinct
    /// opening side around each enclosed pair, and one per closing node for
    /// each run of enclosed pairs at one outer node that a worker takes in a
    /// row (so this count, unlike the others, depends on how the pairs were
    /// shared out).
    pub strands: u64,
}

impl SinglesTally {
    fn fields(self) -> [u64; 7] {
        [
            self.stack_bulge,
            self.split,
            self.corners,
            self.corners_generic,
            self.walked,
            self.corner_pairs,
            self.strands,
        ]
    }
}

/// [`SinglesTally`] summed across workers.
#[derive(Default)]
struct SharedTally([AtomicU64; 7]);

impl SharedTally {
    fn add(&self, t: SinglesTally) {
        for (slot, v) in self.0.iter().zip(t.fields()) {
            if v > 0 {
                slot.fetch_add(v, Ordering::Relaxed);
            }
        }
    }

    fn get(&self) -> SinglesTally {
        let [a, b, c, g, d, e, f] = self.0.each_ref().map(|x| x.load(Ordering::Relaxed));
        SinglesTally {
            stack_bulge: a,
            split: b,
            corners: c,
            corners_generic: g,
            walked: d,
            corner_pairs: e,
            strands: f,
        }
    }
}

/// One strand of the loops around an enclosed pair, for the record-free pass:
/// its best fill, whether it splits at a codon boundary, and its corners once a
/// loop that does not split asks for them. A slot's strand is refilled in place
/// for the next enclosed pair, so its corner buffer is allocated once.
#[derive(Default)]
struct Strand {
    cai: Option<f64>,
    split: bool,
    corners: Vec<Corner>,
    cornered: bool,
}

impl Strand {
    fn refill(&mut self, scan: &OutsideScan<'_>, from: Node, to: Node) {
        self.cai = scan.fill(from, to);
        self.split = scan.separable(from, to);
        self.cornered = false;
    }

    fn corners(&mut self, scan: &OutsideScan<'_>, from: Node, to: Node) -> &[Corner] {
        if !self.cornered {
            scan.corners_into(from, to, &mut self.corners);
            self.cornered = true;
        }
        &self.corners
    }
}

/// Hands each pass its [`OutsideScan::id`].
static PASS_IDS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The record-free pass's opening and closing strands, reused from one
    /// enclosed pair to the next on each worker.
    static STRANDS: RefCell<(StrandSlots, StrandSlots)> = RefCell::default();
}

/// The strands met around one enclosed pair, one slot per strand end.
///
/// An enclosing pair's opening node lies fewer than `l` positions before the
/// enclosed pair and its closing node fewer than that many after, so
/// the ends index a small dense window: an opening strand by its node and first
/// nucleotide, a closing strand by its node. Only the slots filled are cleared
/// for the next pair.
#[derive(Default)]
struct StrandSlots {
    slots: Vec<Strand>,
    live: Vec<bool>,
    filled: Vec<usize>,
    /// The pass and the node the slots' strands were filled for, when the
    /// caller keeps them past one enclosed pair.
    owner: Option<(u64, Node)>,
}

impl StrandSlots {
    /// Slots for opening strands fewer than `l` positions before the enclosed pair.
    fn opening_len(l: usize, stride: usize) -> usize {
        (l + 1) * stride
    }

    /// Slots for closing strands fewer than `l` positions after the enclosed pair.
    fn closing_len(l: usize, stride: usize) -> usize {
        l * stride
    }

    /// The slot of the opening strand out of `from`, the opening edge's
    /// target, for an enclosed pair opening at `a`.
    fn opening(a: u32, from: Node, stride: usize) -> usize {
        (a - from.pos) as usize * stride + usize::from(from.lane)
    }

    /// The slot of the closing strand into `qn`, for an enclosed pair closing
    /// at `b`.
    fn closing(b: u32, qn: Node, stride: usize) -> usize {
        (qn.pos - b - 1) as usize * stride + usize::from(qn.lane)
    }

    /// Empty every slot, and hold at least `len` of them.
    fn reset(&mut self, len: usize) {
        for &i in &self.filled {
            self.live[i] = false;
        }
        self.filled.clear();
        if self.slots.len() < len {
            self.slots.resize_with(len, Strand::default);
            self.live.resize(len, false);
        }
    }

    /// The strand in slot `i`, refilled by `fill` when the slot is empty.
    fn get_or_fill(&mut self, i: usize, fill: impl FnOnce(&mut Strand)) -> &mut Strand {
        let slot = &mut self.slots[i];
        if !self.live[i] {
            self.live[i] = true;
            self.filled.push(i);
            fill(slot);
        }
        slot
    }

    /// Strands held.
    fn len(&self) -> usize {
        self.filled.len()
    }
}

/// The opening side of the loops around one enclosed pair: its strand's
/// choices and each piece's least value.
struct OpenSide {
    left_cai: Option<f64>,
    lefts: Vec<Corner>,
    split: bool,
    corner: Vec<f64>,
    stack: f64,
}

/// The closing side of the loops around one enclosed pair.
struct CloseSide {
    right_cai: Option<f64>,
    rights: Vec<Corner>,
    split: bool,
    corner: Vec<f64>,
    stack: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::must_pair::MustPairConstraint;
    use crate::{CodonUsage, ScanConfig};

    /// The record-free pass scores each enclosing pair's least outer half
    /// without listing its corners; it must be the least of the corners the
    /// recording pass lists, bit for bit.
    #[test]
    fn listless_outer_half_is_the_least_corner() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa = [10, 3, 15, 7, 0, 12, 5, 18, 2, 9, 14, 6];
        let graph = crate::dfa::build(&aa, &usage);
        let region = PairConstraint::inert();
        let config = ScanConfig::default();
        let mut checked = 0;
        for lambda in [0.01, 0.3, 0.9] {
            let tables = crate::scan::scan_values_with_constraints(
                &params,
                &graph,
                lambda,
                config,
                region.clone(),
                MustPairConstraint::inert(),
            );
            let mut scan =
                OutsideScan::new(&params, &graph, &tables, &region, config.single_max_len);
            scan.index_outers();
            for x in &scan.outers {
                let listless = scan.outer_ao_best(x.pair, x.ep.to, x.qn);
                assert_eq!(x.ao_best.to_bits(), listless.to_bits(), "λ = {lambda}");
                checked += usize::from(x.ao_best.is_finite());
            }
        }
        assert!(checked > 0, "no enclosing pair had a split-loop corner");
    }
}
