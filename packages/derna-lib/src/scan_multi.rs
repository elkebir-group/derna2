//! The exact inside scan at several weights in one traversal.
//!
//! An exact scan of one lattice holds the same cells at every weight `λ`: a
//! cell exists when some derivation reaches it, whatever it scores. The scan
//! reads `λ` only where it compares candidates, so the traversal, the cells it
//! creates, the candidates it enumerates in their order and every loop energy
//! it evaluates are the same at every weight. This module runs the scan's
//! default path ([`crate::InternalLoopMode::Lz`] with the column fill, one
//! worker) once for `K` weights. Each cell holds the `K` weights' `(mfe, cai)`
//! side by side, one lane per weight; every merge, and every choice among
//! candidates inside the enumeration, is made lane by lane under the single
//! scan's rule (strict `<` on the lane's BRD, the first arrival kept on a tie),
//! so each lane sees its own single scan's candidates in the same order and
//! keeps the same winner.
//!
//! The Lyngsø–Zuker reductions pick, per outer cell, the inner pair with the
//! least factored score and rescore it exactly. Their indexes keep the shared
//! key structure once and the `λ`-dependent minima per lane, and each lane
//! rescores its own winner.
//!
//! Only values are kept: every cell's back-pointer is
//! [`Back::Unrecorded`](crate::state::Back::Unrecorded). Required-pair and cut
//! constraints are not supported; the region-pair constraint is.

// A lane loop reads the same lane of several arrays at once.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;

use rna_mfe::loops::{
    bulge_energy, internal_energy_scalar, pair_from_nucleotides, stack_energy, BulgeLoop,
    InternalLoopScalar, StackPair,
};
use rna_mfe::params::{BasePair, Triplet37};
use rna_mfe::{hairpin_energy, hairpin_energy_scalar, Energy, HairpinLoop, Nucleotide, Params};

use crate::dfa::{Dfa, Edge, Node, NodeLayout};
use crate::hash::FxHashMap;
use crate::interior::{corner_scores, corner_scores_into, Corner, LoopPairs, OneByNRows};
use crate::region::{PairConstraint, Verdict};
use crate::state::{Back, MultiOpenKey, PairKey, State};
use crate::tables::Tables;
use crate::{Beam, InternalLoopMode, ScanConfig};

/// Smallest number of unpaired bases inside a hairpin.
const MIN_HAIRPIN_LOOP: u32 = 3;

/// The dense columns' empty-slot marker, as [`Tables`] keeps it.
const EMPTY: i32 = i32::MAX;

// ---- lane values -----------------------------------------------------------

/// One cell's `(mfe, cai)` in every lane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct St<const K: usize> {
    pub(crate) mfe: [i32; K],
    pub(crate) cai: [f64; K],
}

impl<const K: usize> St<K> {
    /// Lane `l`'s [`State::brd`] at `lambda`.
    fn brd(&self, l: usize, lambda: f64) -> f64 {
        lambda * f64::from(self.mfe[l]) - (1.0 - lambda) * (self.cai[l] * 100.0)
    }

    /// Every lane's `mfe` raised by `add`, its `cai` unchanged.
    fn plus_mfe(&self, add: Energy) -> Self {
        Self {
            mfe: self.mfe.map(|m| m + add.0),
            cai: self.cai,
        }
    }

    /// Every lane's `cai` raised by `cai`, its `mfe` unchanged.
    fn plus_cai(&self, cai: f64) -> Self {
        Self {
            mfe: self.mfe,
            cai: self.cai.map(|c| c + cai),
        }
    }
}

/// Keep `cand`'s lane wherever it scores strictly below `slot`'s.
fn merge_lanes<const K: usize>(slot: &mut St<K>, cand: &St<K>, lambda: &[f64; K]) {
    for l in 0..K {
        if cand.brd(l, lambda[l]) < slot.brd(l, lambda[l]) {
            slot.mfe[l] = cand.mfe[l];
            slot.cai[l] = cand.cai[l];
        }
    }
}

fn merge_map<Q: std::hash::Hash + Eq, const K: usize>(
    map: &mut FxHashMap<Q, St<K>>,
    key: Q,
    cand: St<K>,
    lambda: &[f64; K],
) {
    match map.get_mut(&key) {
        Some(existing) => merge_lanes(existing, &cand, lambda),
        None => {
            map.insert(key, cand);
        }
    }
}

/// The interior helpers' BRD, with their association.
fn brd_interior(lambda: f64, energy: Energy, cai: f64) -> f64 {
    lambda * f64::from(energy.0) - (1.0 - lambda) * cai * 100.0
}

/// A dense `M`/`M2` column in every lane: one occupancy for all lanes.
#[derive(Clone)]
struct Col<const K: usize> {
    mfe: Vec<[i32; K]>,
    cai: Vec<[f64; K]>,
    /// The lattice's dense node layout: a slot index is an inner node's
    /// [`NodeLayout::ix`].
    layout: NodeLayout,
}

impl<const K: usize> Col<K> {
    fn new(layout: NodeLayout, width: usize) -> Self {
        Self {
            mfe: vec![[EMPTY; K]; width],
            cai: vec![[0.0; K]; width],
            layout,
        }
    }

    fn occupied(&self, i: usize) -> bool {
        self.mfe[i][0] != EMPTY
    }

    fn get(&self, i: usize) -> St<K> {
        St {
            mfe: self.mfe[i],
            cai: self.cai[i],
        }
    }

    fn merge(&mut self, inner: Node, cand: &St<K>, lambda: &[f64; K]) {
        let i = self.layout.ix(inner);
        if !self.occupied(i) {
            self.mfe[i] = cand.mfe;
            self.cai[i] = cand.cai;
            return;
        }
        let mut cur = self.get(i);
        merge_lanes(&mut cur, cand, lambda);
        self.mfe[i] = cur.mfe;
        self.cai[i] = cur.cai;
    }

    /// Occupied cells in ascending index order.
    fn states(&self) -> Vec<(Node, St<K>)> {
        (0..self.mfe.len())
            .filter(|&i| self.occupied(i))
            .map(|i| (self.layout.node_of(i), self.get(i)))
            .collect()
    }
}

/// The six tables of `K` exact scans, lane by lane.
pub(crate) struct LaneTables<const K: usize> {
    p: Vec<FxHashMap<PairKey, St<K>>>,
    multi: Vec<FxHashMap<PairKey, St<K>>>,
    m: Vec<Col<K>>,
    m2: Vec<Col<K>>,
    c: Vec<Option<St<K>>>,
    multi_open: FxHashMap<MultiOpenKey, St<K>>,
    multi_open_pending: Vec<Vec<(MultiOpenKey, St<K>)>>,
    lambda: [f64; K],
    region: PairConstraint,
    /// The lattice's dense node layout, which every table is indexed by.
    layout: NodeLayout,
}

impl<const K: usize> LaneTables<K> {
    fn new(dfa: &Dfa, lambda: [f64; K], region: PairConstraint) -> Self {
        let layout = dfa.layout();
        let n = dfa.slots();
        Self {
            layout,
            p: vec![FxHashMap::default(); n],
            multi: vec![FxHashMap::default(); n],
            m: (0..n).map(|i| Col::new(layout, i)).collect(),
            m2: (0..n).map(|i| Col::new(layout, i)).collect(),
            c: vec![None; n],
            multi_open: FxHashMap::default(),
            multi_open_pending: vec![Vec::new(); dfa.seq_len as usize + 1],
            lambda,
            region,
        }
    }

    fn update_p(&mut self, outer: Node, key: PairKey, cand: St<K>) {
        let cand = match self.region.verdict(key.open.pos, key.close.pos) {
            Verdict::Allow => cand,
            Verdict::Forbid => return,
            Verdict::Penalize(pen) => St {
                mfe: cand.mfe.map(|m| m + pen.0),
                ..cand
            },
        };
        merge_map(&mut self.p[self.layout.ix(outer)], key, cand, &self.lambda);
    }

    fn update_multi(&mut self, outer: Node, key: PairKey, cand: St<K>) {
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
            &self.lambda,
        );
    }

    fn update_m(&mut self, outer: Node, inner: Node, cand: St<K>) {
        self.m[self.layout.ix(outer)].merge(inner, &cand, &self.lambda);
    }

    fn update_m2(&mut self, outer: Node, inner: Node, cand: St<K>) {
        self.m2[self.layout.ix(outer)].merge(inner, &cand, &self.lambda);
    }

    fn update_c(&mut self, node: Node, cand: St<K>) {
        match &mut self.c[self.layout.ix(node)] {
            slot @ None => *slot = Some(cand),
            Some(existing) => merge_lanes(existing, &cand, &self.lambda),
        }
    }

    fn multi_open_insert(&mut self, key: MultiOpenKey, cand: St<K>) {
        merge_map(&mut self.multi_open, key, cand, &self.lambda);
    }

    fn multi_open_defer(&mut self, at: u32, key: MultiOpenKey, cand: St<K>) {
        self.multi_open_pending[at as usize].push((key, cand));
    }

    fn multi_open_activate(&mut self, at: u32) {
        let pending = std::mem::take(&mut self.multi_open_pending[at as usize]);
        for (key, cand) in pending {
            merge_map(&mut self.multi_open, key, cand, &self.lambda);
        }
    }

    fn p_states(&self, outer: Node) -> Vec<(PairKey, St<K>)> {
        self.p[self.layout.ix(outer)]
            .iter()
            .map(|(&k, &s)| (k, s))
            .collect()
    }

    fn multi_states(&self, outer: Node) -> Vec<(PairKey, St<K>)> {
        self.multi[self.layout.ix(outer)]
            .iter()
            .map(|(&k, &s)| (k, s))
            .collect()
    }

    /// Lane `l` as a values-only [`Tables`] at its weight. Every cell's
    /// back-pointer is [`Back::Unrecorded`].
    fn lane(&self, dfa: &Dfa, l: usize) -> Tables {
        let lambda = self.lambda[l];
        let mut t = Tables::values_only(dfa, lambda, Beam::exact(), self.region.clone());
        let state = |s: &St<K>| State::new(Energy(s.mfe[l]), s.cai[l], Back::Unrecorded);
        for (o, map) in self.p.iter().enumerate() {
            t.p[o] = map.iter().map(|(&k, s)| (k, state(s))).collect();
        }
        for (o, map) in self.multi.iter().enumerate() {
            t.multi[o] = map.iter().map(|(&k, s)| (k, state(s))).collect();
        }
        for (o, col) in self.m.iter().enumerate() {
            for (inner, s) in col.states() {
                t.update_m(self.layout.node_of(o), inner, state(&s));
            }
        }
        for (o, col) in self.m2.iter().enumerate() {
            for (inner, s) in col.states() {
                t.update_m2(self.layout.node_of(o), inner, state(&s));
            }
        }
        for (o, c) in self.c.iter().enumerate() {
            if let Some(s) = c {
                t.c[o] = Some(state(s));
            }
        }
        t
    }
}

// ---- lane indexes ----------------------------------------------------------

/// Entry offsets of a list sorted by a small integer key (as the single
/// scan's index keeps them).
#[derive(Default, Clone)]
struct Starts {
    base: u32,
    first: Vec<u32>,
}

impl Starts {
    fn push(&mut self, key: u32, at: usize) {
        if self.first.is_empty() {
            self.base = key;
        }
        while self.base as usize + self.first.len() <= key as usize {
            self.first.push(at as u32);
        }
    }

    fn lower(&self, key: u32, len: usize) -> usize {
        if key <= self.base {
            return 0;
        }
        self.first
            .get((key - self.base) as usize)
            .map_or(len, |&i| i as usize)
    }

    fn window(&self, lo: u32, hi: u32, len: usize) -> std::ops::Range<usize> {
        let a = self.lower(lo, len);
        let b = self.lower(hi.saturating_add(1), len);
        a..b.max(a)
    }
}

/// One inner pair's skewed-reduction record, its `H` per lane.
#[derive(Clone, Copy)]
struct HEntry<const K: usize> {
    k: u32,
    h: [f64; K],
    io: Node,
    ikey: PairKey,
    st: St<K>,
}

/// One size diagonal: shared entries, and a sparse range-min table per lane.
struct Diag<const K: usize> {
    entries: Vec<HEntry<K>>,
    below: Vec<u32>,
    table: Vec<Vec<[u32; K]>>,
}

impl<const K: usize> Default for Diag<K> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            below: Vec::new(),
            table: Vec::new(),
        }
    }
}

impl<const K: usize> Diag<K> {
    fn push(&mut self, e: HEntry<K>) {
        let m = self.entries.len();
        self.entries.push(e);
        let k = e.k as usize;
        if self.below.len() <= k {
            self.below.resize(k + 1, m as u32);
        }
        let mut j = 1;
        while (1usize << j) <= m + 1 {
            if self.table.len() < j {
                self.table.push(Vec::new());
            }
            let i = m + 1 - (1 << j);
            let row: [u32; K] = std::array::from_fn(|l| {
                self.least(
                    l,
                    self.at(l, j - 1, i),
                    self.at(l, j - 1, i + (1 << (j - 1))),
                )
            });
            self.table[j - 1].push(row);
            j += 1;
        }
    }

    fn at(&self, l: usize, j: usize, i: usize) -> u32 {
        if j == 0 {
            i as u32
        } else {
            self.table[j - 1][i][l]
        }
    }

    fn least(&self, l: usize, a: u32, b: u32) -> u32 {
        if self.entries[a as usize].h[l] <= self.entries[b as usize].h[l] {
            a
        } else {
            b
        }
    }

    fn count_below(&self, k: u32) -> usize {
        self.below
            .get(k as usize)
            .map_or(self.entries.len(), |&c| c as usize)
    }

    fn window_min(&self, klo: u32, khi: u32) -> Option<([f64; K], [u32; K])> {
        let lo = self.count_below(klo);
        let hi = self.count_below(khi.saturating_add(1));
        if lo >= hi {
            return None;
        }
        let (a, b) = (lo, hi - 1);
        let j = (b - a + 1).ilog2() as usize;
        let idx: [u32; K] = std::array::from_fn(|l| {
            self.least(l, self.at(l, j, a), self.at(l, j, b + 1 - (1 << j)))
        });
        let h = std::array::from_fn(|l| self.entries[idx[l] as usize].h[l]);
        Some((h, idx))
    }
}

#[derive(Default)]
struct HIndex<const K: usize> {
    by_diag: Vec<Diag<K>>,
}

impl<const K: usize> HIndex<K> {
    fn insert(&mut self, delta: i64, entry: HEntry<K>) {
        let d = usize::try_from(delta).expect("an inner pair's span is not negative");
        if self.by_diag.len() <= d {
            self.by_diag.resize_with(d + 1, Diag::default);
        }
        self.by_diag[d].push(entry);
    }

    fn window_min(&self, delta: i64, klo: u32, khi: u32) -> Option<([f64; K], [u32; K])> {
        self.by_diag
            .get(usize::try_from(delta).ok()?)?
            .window_min(klo, khi)
    }

    fn entry(&self, delta: i64, i: u32) -> &HEntry<K> {
        &self.by_diag[delta as usize].entries[i as usize]
    }
}

/// One inner pair's one-sided record: shared coordinates, per-lane values.
#[derive(Clone, Copy)]
struct OneSideEntry<const K: usize> {
    k: u32,
    io: Node,
    ikey: PairKey,
    st: St<K>,
    brd: [f64; K],
    bmin: [[f64; 4]; K],
}

struct OneSideIndex<const K: usize> {
    by_col: Vec<Vec<OneSideEntry<K>>>,
    starts: Vec<Starts>,
}

impl<const K: usize> OneSideIndex<K> {
    fn new(seq_len: u32) -> Self {
        Self {
            by_col: vec![Vec::new(); seq_len as usize + 1],
            starts: vec![Starts::default(); seq_len as usize + 1],
        }
    }

    fn push(&mut self, l: u32, entry: OneSideEntry<K>) {
        self.by_col[l as usize].push(entry);
    }

    /// Sort column `l` by `k` in the single index's order (see [`SortProxy`]).
    fn sort_col(&mut self, l: u32) {
        let col = &mut self.by_col[l as usize];
        *col = SortProxy::order(col.iter().map(|e| e.k))
            .map(|i| col[i])
            .collect();
        let starts = &mut self.starts[l as usize];
        *starts = Starts::default();
        for (i, e) in col.iter().enumerate() {
            starts.push(e.k, i);
        }
    }

    fn window(&self, l: u32, klo: u32, khi: u32) -> &[OneSideEntry<K>] {
        let col = &self.by_col[l as usize];
        &col[self.starts[l as usize].window(klo, khi, col.len())]
    }
}

struct OneSideKIndex<const K: usize> {
    by_k: Vec<Vec<OneSideEntry<K>>>,
    starts: Vec<Starts>,
}

impl<const K: usize> OneSideKIndex<K> {
    fn new(seq_len: u32) -> Self {
        Self {
            by_k: vec![Vec::new(); seq_len as usize + 1],
            starts: vec![Starts::default(); seq_len as usize + 1],
        }
    }

    fn push(&mut self, k: u32, entry: OneSideEntry<K>) {
        let bucket = &mut self.by_k[k as usize];
        self.starts[k as usize].push(entry.io.pos, bucket.len());
        bucket.push(entry);
    }

    fn window(&self, k: u32, llo: u32, lhi: u32) -> &[OneSideEntry<K>] {
        let bucket = &self.by_k[k as usize];
        &bucket[self.starts[k as usize].window(llo + 1, lhi + 1, bucket.len())]
    }
}

/// The single index's column sort: an unstable sort by `k` of 128-byte
/// proxies, whose permutation of entries sharing a `k` the single scan's
/// recorded tie order depends on.
#[derive(Clone, Copy)]
struct SortProxy {
    k: u32,
    at: u32,
    _pad: [u64; 15],
}

const _: () = assert!(std::mem::size_of::<SortProxy>() == 128);

impl SortProxy {
    fn order(keys: impl Iterator<Item = u32>) -> impl Iterator<Item = usize> {
        let mut proxies: Vec<Self> = (0u32..)
            .zip(keys)
            .map(|(at, k)| Self {
                k,
                at,
                _pad: [0; 15],
            })
            .collect();
        proxies.sort_unstable_by_key(|p| p.k);
        proxies.into_iter().map(|p| p.at as usize)
    }
}

// ---- the scan --------------------------------------------------------------

/// A buffered `P` candidate.
type Cand<const K: usize> = (Node, PairKey, St<K>);

/// One single-branch closure's fixed coordinates, with the inner pair's
/// values in every lane.
struct SingleCtx<const K: usize> {
    st: St<K>,
    ep: Edge,
    pn: Node,
    qn: Node,
    eq: Edge,
    io: Node,
    outer_pair: BasePair,
    inner_pair: BasePair,
}

impl<const K: usize> SingleCtx<K> {
    fn out_key(&self) -> PairKey {
        PairKey {
            open: self.pn,
            close: self.qn,
            opening: self.ep.opening,
            nuc_close: self.eq.nuc,
        }
    }
}

struct OuterCell {
    pn: Node,
    ep: Edge,
    qn: Node,
    eq: Edge,
    outer_pair: BasePair,
}

struct CoreEnd {
    outer: Node,
    closes: Vec<(Node, f64)>,
    boundary: u32,
    c_outer: Option<f64>,
}

/// A long-flank multi-loop opening carried leftward, per lane: the lane's
/// best core and its values, its opening edge and its adaptation.
#[derive(Clone, Copy)]
struct FarOpening<const K: usize> {
    core_inner: [Node; K],
    st: St<K>,
    ep: [Edge; K],
    cai: [f64; K],
}

impl<const K: usize> FarOpening<K> {
    fn score(&self, l: usize, lambda: f64) -> f64 {
        lambda * f64::from(self.st.mfe[l]) - (1.0 - lambda) * (self.cai[l] * 100.0)
    }

    fn merge_into(
        self,
        far: &mut FxHashMap<MultiOpenKey, FarOpening<K>>,
        key: MultiOpenKey,
        lambda: &[f64; K],
    ) {
        let Some(old) = far.get_mut(&key) else {
            far.insert(key, self);
            return;
        };
        for l in 0..K {
            if old.score(l, lambda[l]) <= self.score(l, lambda[l]) {
                continue;
            }
            old.core_inner[l] = self.core_inner[l];
            old.st.mfe[l] = self.st.mfe[l];
            old.st.cai[l] = self.st.cai[l];
            old.ep[l] = self.ep[l];
            old.cai[l] = self.cai[l];
        }
    }
}

struct Gathered<const K: usize> {
    io: Node,
    ikey: PairKey,
    st: St<K>,
    n1: usize,
    n2: usize,
    pushed: bool,
}

impl<const K: usize> Clone for Gathered<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const K: usize> Copy for Gathered<K> {}

/// Buffers a position's target fills reuse from one `(side, p)` target to
/// the next.
#[derive(Default)]
struct FillScratch<const K: usize> {
    w: Vec<Cand<K>>,
    gathered: Vec<Gathered<K>>,
    grouped: Vec<Gathered<K>>,
    counts: Vec<usize>,
    skewed: Vec<SkewCand<K>>,
}

struct SkewCand<const K: usize> {
    h: [f64; K],
    add: [f64; K],
    delta: i64,
    idx: [u32; K],
}

const SHAPE_PUSHED: u8 = 1;
const SHAPE_DIRECT: u8 = 2;

fn fill_shapes(ls: usize, dskew: usize) -> Vec<u8> {
    let mut table = vec![0; (ls + 1) * (ls + 1)];
    for n1 in 0..=ls {
        for n2 in 0..=ls {
            let mut shape = 0;
            if pushed_shape(ls, n1, n2) {
                shape |= SHAPE_PUSHED;
            }
            if direct_shape(ls, dskew, n1, n2) {
                shape |= SHAPE_DIRECT;
            }
            table[n1 * (ls + 1) + n2] = shape;
        }
    }
    table
}

fn fill_n1_hi(ls: usize, shapes: &[u8]) -> Vec<Option<usize>> {
    (0..=ls)
        .map(|n2| (0..=ls).rev().find(|&n1| shapes[n1 * (ls + 1) + n2] != 0))
        .collect()
}

fn pushed_shape(ls: usize, n1: usize, n2: usize) -> bool {
    if n1 + n2 > ls || is_generic(n1, n2) {
        return false;
    }
    n1 < 2 || n2 <= 3
}

fn direct_shape(ls: usize, dskew: usize, n1: usize, n2: usize) -> bool {
    if !is_generic(n1, n2) || !(2..=ls - 2).contains(&n2) {
        return false;
    }
    let (lo, hi) = if n2 <= 4 {
        (2, (n2 + dskew - 1).min(ls - n2))
    } else {
        ((n2.saturating_sub(dskew) + 1).max(2), 4)
    };
    (lo..=hi).contains(&n1)
}

fn is_generic(n1: usize, n2: usize) -> bool {
    n1 >= 2 && n2 >= 2 && !matches!((n1, n2), (2, 2) | (2, 3) | (3, 2))
}

fn canonical(a: Nucleotide, b: Nucleotide) -> Option<BasePair> {
    match pair_from_nucleotides(a, b) {
        BasePair::NN => None,
        canonical => Some(canonical),
    }
}

fn stack_or_bulge_energy(
    params: &Params,
    outer: BasePair,
    inner: BasePair,
    n1: usize,
    n2: usize,
) -> Energy {
    if n1 == 0 && n2 == 0 {
        return stack_energy(params, &StackPair { outer, inner });
    }
    bulge_energy(
        params,
        &BulgeLoop {
            outer,
            inner,
            size: n1 + n2,
        },
    )
}

fn mismatch_table(params: &Params, n1: usize, n2: usize) -> Option<&Triplet37> {
    match (n1, n2) {
        (1, n) | (n, 1) if n >= 3 => Some(&params.mismatch_interior_1n),
        (2, 3) | (3, 2) => Some(&params.mismatch_interior_23),
        (2, 2) => None,
        _ if n1 >= 2 && n2 >= 2 => Some(&params.mismatch_interior),
        _ => None,
    }
}

/// The 3′ flank corners from every inner node a loop closing at `q` can
/// enclose to each closing node at `q`.
struct RightCorners {
    layout: NodeLayout,
    lo: usize,
    by_lane: [Vec<Vec<Corner>>; 3],
}

impl RightCorners {
    fn build(dfa: &Dfa, single_max_len: usize, q: u32) -> Self {
        let from = q.saturating_sub(single_max_len as u32);
        let layout = dfa.layout();
        let lo = layout.ix(Node::new(from, 0));
        let span = layout.ix(Node::new(q, 0)) - lo;
        // A third lane only on a stop-labelled lattice: an empty `Vec`
        // allocates nothing.
        let lane = || vec![Vec::new(); span];
        let mut by_lane = [
            lane(),
            lane(),
            if layout.stride() > 2 {
                lane()
            } else {
                Vec::new()
            },
        ];
        for &qn in dfa.nodes_at(q) {
            for pos in from..q {
                for &io in dfa.nodes_at(pos) {
                    let out = &mut by_lane[usize::from(qn.lane)][dfa.ix(io) - lo];
                    corner_scores_into(dfa, io, qn, out);
                }
            }
        }
        RightCorners {
            layout,
            lo,
            by_lane,
        }
    }

    fn get(&self, io: Node, qn: Node) -> &[Corner] {
        &self.by_lane[usize::from(qn.lane)][self.layout.ix(io) - self.lo]
    }
}

thread_local! {
    static LEFTS: RefCell<Vec<Corner>> = const { RefCell::new(Vec::new()) };
    static RIGHTS: RefCell<Vec<Corner>> = const { RefCell::new(Vec::new()) };
}

struct MScan<'a, const K: usize, const S: usize> {
    params: &'a Params,
    dfa: &'a Dfa,
    lambda: [f64; K],
    single_max_len: usize,
    ninio_skew: usize,
    fill_shapes: Vec<u8>,
    fill_n1_hi: Vec<Option<usize>>,
    one_by_n: OneByNRows,
}

/// The `K` exact scans of `dfa` at `lambdas` under `region`, in one
/// traversal; lane `l` equals
/// `scan_with_constraints(params, dfa, lambdas[l], config, region, inert)`
/// in every cell's `(mfe, cai)`.
///
/// # Panics
///
/// When `config` is not the exact single-worker `Lz` scan, or when a lane
/// would create a cell another lane does not.
pub(crate) fn scan_multi<const K: usize>(
    params: &Params,
    dfa: &Dfa,
    lambdas: [f64; K],
    config: ScanConfig,
    region: PairConstraint,
) -> LaneTables<K> {
    assert!(
        matches!(config.mode, InternalLoopMode::Lz) && config.beam == Beam::exact(),
        "the lane scan runs the exact Lz scan only"
    );
    assert!(
        std::env::var_os("DERNA_PUSH_SCAN").is_none(),
        "the lane scan runs the column fill only"
    );
    let ninio_skew = {
        let n = &params.ninio;
        ((n.max_ninio + n.ninio_37 - 1) / n.ninio_37) as usize
    };
    let shapes = fill_shapes(config.single_max_len, ninio_skew);
    let fill = Fill {
        lambdas,
        config,
        region,
        ninio_skew,
        shapes,
    };
    // The scan is monomorphized over the lattice's lane stride, so its index
    // arithmetic multiplies by a constant and a lattice without stop lanes
    // compiles out every test for one.
    match dfa.stride() {
        2 => scan_strided::<K, 2>(params, dfa, fill),
        3 => scan_strided::<K, 3>(params, dfa, fill),
        stride => unreachable!("a lattice's lane stride is 2 or 3, not {stride}"),
    }
}

/// What [`scan_multi`] has prepared before it knows the lattice's stride.
struct Fill<const K: usize> {
    lambdas: [f64; K],
    config: ScanConfig,
    region: PairConstraint,
    ninio_skew: usize,
    shapes: Vec<u8>,
}

/// [`scan_multi`] over a lattice whose lane stride is `S`.
fn scan_strided<const K: usize, const S: usize>(
    params: &Params,
    dfa: &Dfa,
    fill: Fill<K>,
) -> LaneTables<K> {
    let Fill {
        lambdas,
        config,
        region,
        ninio_skew,
        shapes,
    } = fill;
    let s = MScan::<K, S> {
        params,
        dfa,
        lambda: lambdas,
        single_max_len: config.single_max_len,
        ninio_skew,
        fill_n1_hi: fill_n1_hi(config.single_max_len, &shapes),
        fill_shapes: shapes,
        one_by_n: OneByNRows::of(params),
    };
    let mut t = LaneTables::new(dfa, lambdas, region);
    t.update_c(
        dfa.start(),
        St {
            mfe: [0; K],
            cai: [0.0; K],
        },
    );
    let mut hx = HIndex::default();
    let mut osx = OneSideIndex::new(dfa.seq_len);
    let mut okx = OneSideKIndex::new(dfa.seq_len);
    s.seed_hairpins(&mut t);
    for pos in 0..=dfa.seq_len {
        s.step(&mut t, (&mut hx, &mut osx, &mut okx), pos);
    }
    t
}

impl<const K: usize> LaneTables<K> {
    /// Every lane as a values-only [`Tables`].
    pub(crate) fn into_lanes(self, dfa: &Dfa) -> [Tables; K] {
        std::array::from_fn(|l| self.lane(dfa, l))
    }
}

impl<const K: usize, const S: usize> MScan<'_, K, S> {
    /// Whether a path runs from `a` to `b` ([`Dfa::connected`]); always true
    /// on a lattice whose positions hold no stop lane.
    #[inline]
    fn linked(&self, a: Node, b: Node) -> bool {
        S == 2 || Dfa::connected(self.dfa, a, b)
    }

    /// The stops a node still reaches ([`Dfa::reach_class`]); every one of
    /// them on a lattice whose positions hold no stop lane.
    #[inline]
    fn reach_of(&self, node: Node) -> u8 {
        if S == 2 {
            0
        } else {
            Dfa::reach_class(self.dfa, node)
        }
    }

    /// Whether an opening reaching the stops `reach` can close into `node`
    /// ([`Dfa::reaches`]).
    #[inline]
    fn admits(&self, reach: u8, node: Node) -> bool {
        S == 2 || Dfa::reaches(self.dfa, reach, node)
    }

    fn step(
        &self,
        t: &mut LaneTables<K>,
        (hx, osx, okx): (&mut HIndex<K>, &mut OneSideIndex<K>, &mut OneSideKIndex<K>),
        pos: u32,
    ) {
        let dfa = self.dfa;
        t.multi_open_activate(pos);
        for &jn in dfa.nodes_at(pos) {
            self.multi_close(t, jn);
        }
        self.pull_fill_column(t, hx, osx, okx, pos);
        self.p_beam_all(t, pos);
        self.index_finalized_p(t, hx, osx, okx, pos);
        self.m2_fill_column(t, pos);
        let openings: Vec<(MultiOpenKey, St<K>)> =
            t.multi_open.iter().map(|(&k, &s)| (k, s)).collect();
        for &jn in dfa.nodes_at(pos) {
            self.multi_open_close(t, jn, &openings);
        }
        if pos < dfa.seq_len {
            for &jn in dfa.nodes_at(pos) {
                self.m_extend(t, jn);
            }
            for &jn in dfa.nodes_at(pos) {
                self.c_extend(t, jn);
            }
        }
    }

    fn start_contrib(&self, node: Node) -> f64 {
        self.dfa.span_head_cai(node).unwrap_or(0.0)
    }

    fn end_contrib(&self, node: Node) -> f64 {
        self.dfa.span_tail_cai(node).unwrap_or(0.0)
    }

    fn m_stem_cost(&self, pair: BasePair) -> Energy {
        Energy(self.params.ml.intern_37) + self.terminal_au(pair)
    }

    fn multi_close_cost(&self, closing: BasePair) -> Energy {
        Energy(self.params.ml.closing_37) + self.m_stem_cost(closing)
    }

    fn terminal_au(&self, pair: BasePair) -> Energy {
        if pair.applies_terminal_au() {
            Energy(self.params.misc.terminal_au_37)
        } else {
            Energy::ZERO
        }
    }

    // ---- hairpins ---------------------------------------------------------

    fn seed_hairpins(&self, t: &mut LaneTables<K>) {
        for a in 0..self.dfa.seq_len {
            for &an in self.dfa.nodes_at(a) {
                for eo in self.dfa.right(an) {
                    let b_hi = self.dfa.seq_len;
                    let mut b = self.dfa.next_partner(eo.nuc, an.pos + MIN_HAIRPIN_LOOP + 1);
                    while b < b_hi {
                        for &bn in self.dfa.nodes_at(b) {
                            for ec in self.dfa.right(bn) {
                                self.seed_hairpin_close(t, an, *eo, bn, *ec);
                            }
                        }
                        b = self.dfa.next_partner(eo.nuc, b + 1);
                    }
                }
            }
        }
    }

    fn seed_hairpin_close(&self, t: &mut LaneTables<K>, an: Node, eo: Edge, bn: Node, ec: Edge) {
        let Some(closing) = canonical(eo.nuc, ec.nuc) else {
            return;
        };
        let Some(best) = self.score_hairpin(eo.to, bn, closing) else {
            return;
        };
        let key = PairKey {
            open: an,
            close: bn,
            opening: eo.opening,
            nuc_close: ec.nuc,
        };
        let cand = St {
            mfe: best.map(|(e, _)| e.0),
            cai: best.map(|(_, c)| eo.cai + c + ec.cai),
        };
        t.update_p(ec.to, key, cand);
    }

    /// [`crate::interior::score_hairpin`] in every lane: the candidates and
    /// their energies once, each lane's first least BRD.
    fn score_hairpin(
        &self,
        start: Node,
        end: Node,
        closing: BasePair,
    ) -> Option<[(Energy, f64); K]> {
        let size = (end.pos - start.pos) as usize;
        let cands: Vec<(Energy, f64)> = if matches!(size, 3 | 4 | 6) {
            self.dfa
                .paths_between(start, end)
                .into_iter()
                .map(|(cai, seq)| {
                    (
                        hairpin_energy(
                            self.params,
                            &HairpinLoop {
                                closing,
                                loop_seq: &seq,
                            },
                        ),
                        cai,
                    )
                })
                .collect()
        } else {
            corner_scores(self.dfa, start, end)
                .into_iter()
                .map(|c| {
                    (
                        hairpin_energy_scalar(self.params, closing, size, c.first, c.last),
                        c.cai,
                    )
                })
                .collect()
        };
        let first = *cands.first()?;
        let mut best = [first; K];
        for &c in &cands[1..] {
            for l in 0..K {
                let lam = self.lambda[l];
                let (b, x) = (
                    brd_interior(lam, best[l].0, best[l].1),
                    brd_interior(lam, c.0, c.1),
                );
                if b.partial_cmp(&x).expect("finite BRD scores") == std::cmp::Ordering::Greater {
                    best[l] = c;
                }
            }
        }
        Some(best)
    }

    // ---- the P column -----------------------------------------------------

    fn multi_close(&self, t: &mut LaneTables<K>, o: Node) {
        for (key, st) in t.multi_states(o) {
            let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
            t.update_p(o, key, st.plus_mfe(self.multi_close_cost(closing)));
        }
    }

    fn pull_fill_column(
        &self,
        t: &mut LaneTables<K>,
        hx: &HIndex<K>,
        osx: &OneSideIndex<K>,
        okx: &OneSideKIndex<K>,
        pos: u32,
    ) {
        if pos == 0 {
            return;
        }
        let q = pos - 1;
        let sides: Vec<(Node, Edge)> = self
            .dfa
            .nodes_at(q)
            .iter()
            .flat_map(|&qn| self.dfa.right(qn).iter().map(move |&eq| (qn, eq)))
            .collect();
        if sides.is_empty() || q == 0 {
            return;
        }
        let n_p = q;
        let total = sides.len() as u32 * n_p;
        let rights = RightCorners::build(self.dfa, self.single_max_len, q);
        let mut s = FillScratch::default();
        let mut out: Vec<Cand<K>> = Vec::new();
        for i in 0..total {
            let (qn, eq) = sides[(i / n_p) as usize];
            self.fill_target(&mut s, (hx, osx, okx), &rights, qn, eq, i % n_p);
            self.reduce_into(&mut s.w, &mut out);
        }
        for (o, key, st) in out {
            t.update_p(o, key, st);
        }
    }

    /// Fold `w` into one best per cell lane by lane, first kept on a tie.
    fn reduce_into(&self, w: &mut Vec<Cand<K>>, out: &mut Vec<Cand<K>>) {
        let from = out.len();
        for &(o, key, st) in w.iter() {
            match out[from..]
                .iter_mut()
                .find(|(oo, kk, _)| *oo == o && *kk == key)
            {
                Some((_, _, best)) => merge_lanes(best, &st, &self.lambda),
                None => out.push((o, key, st)),
            }
        }
        w.clear();
    }

    #[inline(always)]
    fn fill_target(
        &self,
        s: &mut FillScratch<K>,
        (hx, osx, okx): (&HIndex<K>, &OneSideIndex<K>, &OneSideKIndex<K>),
        rights: &RightCorners,
        qn: Node,
        eq: Edge,
        p: u32,
    ) {
        let q = qn.pos;
        let ls = self.single_max_len;
        let gathered = &mut s.gathered;
        gathered.clear();
        for n2 in 0..=ls {
            let Some(l) = q.checked_sub(1 + n2 as u32) else {
                break;
            };
            if l <= p + 1 {
                continue;
            }
            let Some(n1_hi) = self.fill_n1_hi[n2] else {
                continue;
            };
            let khi = (p + 1 + n1_hi as u32).min(l);
            for e in osx.window(l, p + 1, khi) {
                let n1 = (e.k - p - 1) as usize;
                let shape = self.fill_shapes[n1 * (ls + 1) + n2];
                if shape == 0 {
                    continue;
                }
                gathered.push(Gathered {
                    io: e.io,
                    ikey: e.ikey,
                    st: e.st,
                    n1,
                    n2,
                    pushed: shape & SHAPE_PUSHED != 0,
                });
            }
        }
        if !gathered.is_empty() {
            // The single scan's stable counting sort by (n1, lane).
            let stride = S;
            let bucket = |g: &Gathered<K>| g.n1 * stride + usize::from(g.ikey.open.lane);
            let buckets = stride * (ls + 1);
            let counts = &mut s.counts;
            counts.clear();
            counts.resize(buckets + 1, 0);
            for g in gathered.iter() {
                counts[bucket(g) + 1] += 1;
            }
            for b in 0..buckets {
                counts[b + 1] += counts[b];
            }
            let grouped = &mut s.grouped;
            grouped.clear();
            grouped.resize(gathered.len(), gathered[0]);
            for g in gathered.iter() {
                let slot = &mut counts[bucket(g)];
                grouped[*slot] = *g;
                *slot += 1;
            }
            self.close_gathered(&mut s.w, rights, qn, eq, p, grouped);
        }
        if p <= q.saturating_sub(13) {
            self.skewed_pull_at(&mut s.w, &mut s.skewed, hx, (qn, eq), p);
            self.oneside_pull_at(&mut s.w, osx, qn, eq, p);
            self.oneside_pull3_at(&mut s.w, okx, qn, eq, p);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn close_gathered(
        &self,
        w: &mut Vec<Cand<K>>,
        rights: &RightCorners,
        qn: Node,
        eq: Edge,
        p: u32,
        gathered: &[Gathered<K>],
    ) {
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                for group in gathered.chunk_by(|a, b| a.ikey.open == b.ikey.open) {
                    let open = group[0].ikey.open;
                    let left_cai = self.dfa.broken_codon_cai_s::<S>(ep.to, open);
                    let mut corners_ready = false;
                    LEFTS.with_borrow_mut(|lefts| {
                        for g in group {
                            if g.pushed && left_cai.is_none() {
                                continue;
                            }
                            let ctx = SingleCtx {
                                st: g.st,
                                ep: *ep,
                                pn,
                                qn,
                                eq,
                                io: g.io,
                                outer_pair,
                                inner_pair: pair_from_nucleotides(
                                    g.ikey.nuc_open(),
                                    g.ikey.nuc_close,
                                ),
                            };
                            if g.n1 == 0 || g.n2 == 0 {
                                let left_cai = left_cai.expect("checked for a pushed shape");
                                if let Some(c) =
                                    self.stack_or_bulge_cand(&ctx, left_cai, g.n1, g.n2)
                                {
                                    w.push(c);
                                }
                                continue;
                            }
                            if !corners_ready {
                                corner_scores_into(self.dfa, ep.to, open, lefts);
                                corners_ready = true;
                            }
                            if let Some(c) = self.internal_loop_cand(
                                &ctx,
                                lefts,
                                rights.get(g.io, qn),
                                g.n1,
                                g.n2,
                            ) {
                                w.push(c);
                            }
                        }
                    });
                }
            }
        }
    }

    fn stack_or_bulge_cand(
        &self,
        ctx: &SingleCtx<K>,
        left_cai: f64,
        n1: usize,
        n2: usize,
    ) -> Option<Cand<K>> {
        let right_cai = self.dfa.broken_codon_cai_s::<S>(ctx.io, ctx.qn)?;
        let energy = stack_or_bulge_energy(self.params, ctx.outer_pair, ctx.inner_pair, n1, n2);
        let st = St {
            mfe: ctx.st.mfe.map(|m| m + energy.0),
            cai: ctx
                .st
                .cai
                .map(|c| c + ctx.ep.cai + left_cai + right_cai + ctx.eq.cai),
        };
        Some((ctx.eq.to, ctx.out_key(), st))
    }

    /// The internal-loop candidate in every lane: each lane's best corner
    /// pair, energies evaluated once.
    fn internal_loop_cand(
        &self,
        ctx: &SingleCtx<K>,
        lefts: &[Corner],
        rights: &[Corner],
        n1: usize,
        n2: usize,
    ) -> Option<Cand<K>> {
        let best = self.score_internal(
            &LoopPairs {
                outer: ctx.outer_pair,
                inner: ctx.inner_pair,
            },
            (lefts, n1),
            (rights, n2),
            [true; K],
        )?;
        let st = St {
            mfe: std::array::from_fn(|l| ctx.st.mfe[l] + best[l].0 .0),
            cai: std::array::from_fn(|l| ctx.st.cai[l] + ctx.ep.cai + best[l].1 + ctx.eq.cai),
        };
        Some((ctx.eq.to, ctx.out_key(), st))
    }

    /// [`crate::interior::score_internal_from`] for the lanes `mask` names,
    /// each corner pair's energy evaluated once; lanes outside `mask` are
    /// left at the first corner pair.
    fn score_internal(
        &self,
        pairs: &LoopPairs,
        (lefts, n1): (&[Corner], usize),
        (rights, n2): (&[Corner], usize),
        mask: [bool; K],
    ) -> Option<[(Energy, f64); K]> {
        let params = self.params;
        let energy_at = |l: &Corner, r: &Corner| {
            internal_energy_scalar(
                params,
                &InternalLoopScalar {
                    outer: pairs.outer,
                    inner: pairs.inner,
                    boundary_left5p: l.first,
                    boundary_left3p: l.last,
                    boundary_right5p: r.first,
                    boundary_right3p: r.last,
                    n1,
                    n2,
                },
            )
        };
        if self.one_by_n.corner_free(pairs, n1, n2) {
            let energy = energy_at(lefts.first()?, rights.first()?);
            return self.best_corners((lefts, rights), mask, |_, _| energy);
        }
        if let Some(table) = mismatch_table(params, n1, n2) {
            let inner_rev = pairs.inner.reversed();
            let mismatches = |l: &Corner, r: &Corner| {
                table.get(pairs.outer, l.first.to_nuc5(), r.last.to_nuc5())
                    + table.get(inner_rev, r.first.to_nuc5(), l.last.to_nuc5())
            };
            let (l0, r0) = (lefts.first()?, rights.first()?);
            let shape = energy_at(l0, r0).0 - mismatches(l0, r0);
            return self.best_corners((lefts, rights), mask, |l, r| {
                Energy(shape + mismatches(l, r))
            });
        }
        self.best_corners((lefts, rights), mask, energy_at)
    }

    fn best_corners(
        &self,
        (lefts, rights): (&[Corner], &[Corner]),
        mask: [bool; K],
        energy_at: impl Fn(&Corner, &Corner) -> Energy,
    ) -> Option<[(Energy, f64); K]> {
        let mut best: Option<[(f64, Energy, f64); K]> = None;
        for l in lefts {
            for r in rights {
                let energy = energy_at(l, r);
                let cai = l.cai + r.cai;
                match &mut best {
                    None => {
                        best = Some(std::array::from_fn(|k| {
                            (brd_interior(self.lambda[k], energy, cai), energy, cai)
                        }));
                    }
                    Some(b) => {
                        for k in 0..K {
                            if !mask[k] {
                                continue;
                            }
                            let score = brd_interior(self.lambda[k], energy, cai);
                            if score < b[k].0 {
                                b[k] = (score, energy, cai);
                            }
                        }
                    }
                }
            }
        }
        best.map(|b| b.map(|(_, e, c)| (e, c)))
    }

    /// Rescore each lane's own winner of a reduction at the outer cell `oc`
    /// and buffer the candidate: lanes that chose the same inner pair and
    /// flanks share one corner enumeration.
    fn rescore_winners(
        &self,
        w: &mut Vec<Cand<K>>,
        oc: &OuterCell,
        winners: [(Node, PairKey, St<K>, usize, usize); K],
    ) {
        let mut mfe = [0i32; K];
        let mut cai = [0.0f64; K];
        let mut done = [false; K];
        let (mut some, mut none) = (false, false);
        for l in 0..K {
            if done[l] {
                continue;
            }
            let (io, ikey, _, n1, n2) = winners[l];
            let mask: [bool; K] = std::array::from_fn(|j| {
                !done[j] && winners[j].0 == io && winners[j].1 == ikey && winners[j].3 == n1
            });
            let pairs = LoopPairs {
                outer: oc.outer_pair,
                inner: pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close),
            };
            let got = LEFTS.with_borrow_mut(|lefts| {
                corner_scores_into(self.dfa, oc.ep.to, ikey.open, lefts);
                RIGHTS.with_borrow_mut(|rights| {
                    corner_scores_into(self.dfa, io, oc.qn, rights);
                    self.score_internal(&pairs, (lefts, n1), (rights, n2), mask)
                })
            });
            for j in 0..K {
                if !mask[j] {
                    continue;
                }
                done[j] = true;
                let Some(best) = got else {
                    none = true;
                    continue;
                };
                some = true;
                let st = winners[j].2;
                mfe[j] = st.mfe[j] + best[j].0 .0;
                cai[j] = st.cai[j] + oc.ep.cai + best[j].1 + oc.eq.cai;
            }
        }
        assert!(
            !(some && none),
            "a reduction's winner scores in some lanes only"
        );
        let any = some;
        if !any {
            return;
        }
        let key = PairKey {
            open: oc.pn,
            close: oc.qn,
            opening: oc.ep.opening,
            nuc_close: oc.eq.nuc,
        };
        w.push((oc.eq.to, key, St { mfe, cai }));
    }

    // ---- the reductions -----------------------------------------------------

    fn kk(&self, l: usize) -> f64 {
        (1.0 - self.lambda[l]) * 100.0
    }

    fn outer_min(&self, oc: &OuterCell) -> [f64; K] {
        let mm = &self.params.mismatch_interior;
        let mut omin = [f64::INFINITY; K];
        for l5 in self.dfa.right(oc.ep.to) {
            let cl = l5.cai + self.start_contrib(l5.to);
            for r3 in self.dfa.left(oc.qn) {
                let e = f64::from(mm.get(oc.outer_pair, l5.nuc.to_nuc5(), r3.nuc.to_nuc5()));
                let cr = r3.cai + self.end_contrib(r3.to);
                for l in 0..K {
                    omin[l] = omin[l].min(self.lambda[l] * e - self.kk(l) * (cl + cr));
                }
            }
        }
        std::array::from_fn(|l| omin[l] - self.kk(l) * (oc.ep.cai + oc.eq.cai))
    }

    /// The `H`-index slot of diagonal `delta` for inner pairs of reach class
    /// `class`: as in the single-weight scan, one slot range per class.
    fn h_slot(&self, delta: i64, class: u8) -> i64 {
        delta + i64::from(class) * (i64::from(self.dfa.seq_len) + 1)
    }

    /// The inner-pair classes an outer pair can enclose from the flank start
    /// `open` to `qn`, as in the single-weight scan.
    fn h_classes(&self, open: Node, qn: Node) -> smallvec::SmallVec<[u8; 5]> {
        if S == 2 {
            return smallvec::smallvec![0];
        }
        let outer = self.reach_of(open);
        (0..8u8)
            .filter(|&c| c == 0 || (outer == 0 || c & !outer == 0) && self.admits(c, qn))
            .collect()
    }

    fn skewed_pull_at(
        &self,
        w: &mut Vec<Cand<K>>,
        cands: &mut Vec<SkewCand<K>>,
        hx: &HIndex<K>,
        (qn, eq): (Node, Edge),
        p: u32,
    ) {
        let mut ready: Option<u8> = None;
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                let oc = OuterCell {
                    pn,
                    ep: *ep,
                    qn,
                    eq,
                    outer_pair,
                };
                let om = self.outer_min(&oc);
                if !om[0].is_finite() {
                    continue;
                }
                let reach = self.reach_of(ep.to);
                if ready != Some(reach) {
                    self.skewed_candidates(hx, (pn, ep.to, qn), cands);
                    ready = Some(reach);
                }
                self.skewed_outer_cell(w, hx, cands, &oc, om);
            }
        }
    }

    fn skewed_candidates(
        &self,
        hx: &HIndex<K>,
        (pn, open, qn): (Node, Node, Node),
        cands: &mut Vec<SkewCand<K>>,
    ) {
        if S == 2 {
            self.skewed_candidates_in(hx, (pn.pos, qn.pos), &[0], cands);
        } else {
            let classes = self.h_classes(open, qn);
            self.skewed_candidates_in(hx, (pn.pos, qn.pos), &classes, cands);
        }
    }

    /// [`Self::skewed_candidates`] over the inner-pair `classes` the outer
    /// cell can enclose.
    #[inline(always)]
    fn skewed_candidates_in(
        &self,
        hx: &HIndex<K>,
        (p, q): (u32, u32),
        classes: &[u8],
        cands: &mut Vec<SkewCand<K>>,
    ) {
        cands.clear();
        let max_ninio = f64::from(self.params.ninio.max_ninio);
        let ninio_37 = self.params.ninio.ninio_37;
        let dskew = self.ninio_skew;
        let span = q as i64 - p as i64 - 2;
        let klo = p + 6;
        for s in 10..=self.single_max_len {
            let delta = span - s as i64;
            let khi = q as i64 - 6 - delta;
            if delta < 1 || khi < klo as i64 {
                continue;
            }
            let khi = khi as u32;
            let size = f64::from(self.params.interior.values[s]);
            for &class in classes {
                let slot = self.h_slot(delta, class);
                if let Some((h, idx)) = hx.window_min(slot, klo, khi) {
                    let add = self.lambda.map(|lam| lam * (size + max_ninio));
                    cands.push(SkewCand {
                        h,
                        add,
                        delta: slot,
                        idx,
                    });
                }
            }
            for d in 0..dskew {
                if (s + d) % 2 != 0 || (s - d) / 2 < 5 {
                    continue;
                }
                let nin = f64::from(d as i32 * ninio_37);
                for n1 in [(s + d) / 2, (s - d) / 2] {
                    let k = p + 1 + n1 as u32;
                    if k >= klo && k <= khi {
                        for &class in classes {
                            let slot = self.h_slot(delta, class);
                            if let Some((h, idx)) = hx.window_min(slot, k, k) {
                                let add = self.lambda.map(|lam| lam * (size + nin));
                                cands.push(SkewCand {
                                    h,
                                    add,
                                    delta: slot,
                                    idx,
                                });
                            }
                        }
                    }
                    if d == 0 {
                        break;
                    }
                }
            }
        }
    }

    fn skewed_outer_cell(
        &self,
        w: &mut Vec<Cand<K>>,
        hx: &HIndex<K>,
        cands: &[SkewCand<K>],
        oc: &OuterCell,
        om: [f64; K],
    ) {
        if cands.is_empty() {
            return;
        }
        let (p, q) = (oc.pn.pos, oc.qn.pos);
        let winners = std::array::from_fn(|l| {
            let mut best: Option<(f64, &SkewCand<K>)> = None;
            for c in cands {
                let cand = om[l] + c.h[l] + c.add[l];
                if best.as_ref().is_none_or(|b| cand < b.0) {
                    best = Some((cand, c));
                }
            }
            let (_, c) = best.expect("a candidate");
            let entry = hx.entry(c.delta, c.idx[l]);
            let n1 = (entry.k - p - 1) as usize;
            let n2 = (q - entry.io.pos) as usize;
            (entry.io, entry.ikey, entry.st, n1, n2)
        });
        self.rescore_winners(w, oc, winners);
    }

    fn oneside_pull_at(
        &self,
        w: &mut Vec<Cand<K>>,
        osx: &OneSideIndex<K>,
        qn: Node,
        eq: Edge,
        p: u32,
    ) {
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                let oc = OuterCell {
                    pn,
                    ep: *ep,
                    qn,
                    eq,
                    outer_pair,
                };
                self.oneside_outer_cell(w, osx, &oc);
            }
        }
    }

    fn oneside_outer_cell(&self, w: &mut Vec<Cand<K>>, osx: &OneSideIndex<K>, oc: &OuterCell) {
        let (p, q) = (oc.pn.pos, oc.qn.pos);
        let max_ninio = f64::from(self.params.ninio.max_ninio);
        let dskew = self.ninio_skew as u32;
        let mm = &self.params.mismatch_interior;
        let mut a_by_r3 = [[f64::INFINITY; 4]; K];
        for l5 in self.dfa.right(oc.ep.to) {
            let cl = l5.cai + self.start_contrib(l5.to);
            let l5n = l5.nuc.to_nuc5();
            for r3 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = f64::from(mm.get(oc.outer_pair, l5n, r3.to_nuc5()));
                let idx = r3 as usize;
                for l in 0..K {
                    let v = self.lambda[l] * e - self.kk(l) * cl;
                    if v < a_by_r3[l][idx] {
                        a_by_r3[l][idx] = v;
                    }
                }
            }
        }
        if !a_by_r3[0].iter().any(|a| a.is_finite()) {
            return;
        }
        let edge: [f64; K] = std::array::from_fn(|l| -self.kk(l) * (oc.ep.cai + oc.eq.cai));
        let mut best: [Option<(f64, &OneSideEntry<K>, usize, usize)>; K] = [None; K];
        for n2 in 2..=4u32 {
            if q < n2 + 2 {
                continue;
            }
            let l = q - 1 - n2;
            let klo = p + 1 + n2 + dskew;
            let khi = (p + 1 + self.single_max_len as u32 - n2).min(q - 2 - n2);
            if klo > khi {
                continue;
            }
            let labelled = self.dfa.stop_exit().is_some();
            for entry in osx.window(l, klo, khi) {
                // Entries share a window across stop lanes; only those both
                // flanks reach can be enclosed.
                if labelled
                    && (!self.linked(oc.ep.to, entry.ikey.open) || !self.linked(entry.io, oc.qn))
                {
                    continue;
                }
                let n1 = (entry.k - p - 1) as usize;
                let size = f64::from(self.params.interior.values[n1 + n2 as usize]);
                let cms = self.oneside_corner_min(entry, oc.qn, &a_by_r3);
                for k in 0..K {
                    let cm = cms[k];
                    if !cm.is_finite() {
                        continue;
                    }
                    let cand = entry.brd[k] + self.lambda[k] * (size + max_ninio) + cm + edge[k];
                    if best[k].as_ref().is_none_or(|b| cand < b.0) {
                        best[k] = Some((cand, entry, n1, n2 as usize));
                    }
                }
            }
        }
        if best[0].is_none() {
            assert!(
                best.iter().all(Option::is_none),
                "a one-sided winner in some lanes only"
            );
            return;
        }
        let winners = std::array::from_fn(|l| {
            let (_, e, n1, n2) = best[l].expect("a one-sided winner in every lane");
            (e.io, e.ikey, e.st, n1, n2)
        });
        self.rescore_winners(w, oc, winners);
    }

    fn oneside_corner_min(
        &self,
        entry: &OneSideEntry<K>,
        qn: Node,
        a_by_r3: &[[f64; 4]; K],
    ) -> [f64; K] {
        let mut cm = [f64::INFINITY; K];
        for r5 in self.dfa.right(entry.io) {
            for r3 in self.dfa.left(qn) {
                let Some(mid) = self.dfa.broken_codon_cai_s::<S>(r5.to, r3.to) else {
                    continue;
                };
                let cair = r5.cai + mid + r3.cai;
                for l in 0..K {
                    let b = entry.bmin[l][r5.nuc as usize];
                    if !b.is_finite() {
                        continue;
                    }
                    let a = a_by_r3[l][r3.nuc as usize];
                    if a.is_finite() {
                        cm[l] = cm[l].min(a + b - self.kk(l) * cair);
                    }
                }
            }
        }
        cm
    }

    fn oneside_pull3_at(
        &self,
        w: &mut Vec<Cand<K>>,
        okx: &OneSideKIndex<K>,
        qn: Node,
        eq: Edge,
        p: u32,
    ) {
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                let oc = OuterCell {
                    pn,
                    ep: *ep,
                    qn,
                    eq,
                    outer_pair,
                };
                self.oneside_outer_cell3(w, okx, &oc);
            }
        }
    }

    fn oneside_outer_cell3(&self, w: &mut Vec<Cand<K>>, okx: &OneSideKIndex<K>, oc: &OuterCell) {
        let (p, q) = (oc.pn.pos, oc.qn.pos);
        let max_ninio = f64::from(self.params.ninio.max_ninio);
        let dskew = self.ninio_skew as u32;
        let mm = &self.params.mismatch_interior;
        let mut am_by_l5 = [[f64::INFINITY; 4]; K];
        for r3 in self.dfa.left(oc.qn) {
            let cr = r3.cai + self.end_contrib(r3.to);
            let r3n = r3.nuc.to_nuc5();
            for l5 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = f64::from(mm.get(oc.outer_pair, l5.to_nuc5(), r3n));
                let idx = l5 as usize;
                for l in 0..K {
                    let v = self.lambda[l] * e - self.kk(l) * cr;
                    if v < am_by_l5[l][idx] {
                        am_by_l5[l][idx] = v;
                    }
                }
            }
        }
        if !am_by_l5[0].iter().any(|a| a.is_finite()) {
            return;
        }
        let edge: [f64; K] = std::array::from_fn(|l| -self.kk(l) * (oc.ep.cai + oc.eq.cai));
        let mut best: [Option<(f64, &OneSideEntry<K>, usize, usize)>; K] = [None; K];
        for n1 in 2..=4u32 {
            let k = p + 1 + n1;
            let lhi = q.saturating_sub(1 + n1 + dskew);
            let llo = (q + n1)
                .saturating_sub(self.single_max_len as u32 + 1)
                .max(k + 1);
            if llo > lhi {
                continue;
            }
            let labelled = self.dfa.stop_exit().is_some();
            for entry in okx.window(k, llo, lhi) {
                // Entries share a window across stop lanes; only those both
                // flanks reach can be enclosed.
                if labelled
                    && (!self.linked(oc.ep.to, entry.ikey.open) || !self.linked(entry.io, oc.qn))
                {
                    continue;
                }
                let n2 = (q - entry.io.pos) as usize;
                let size = f64::from(self.params.interior.values[n1 as usize + n2]);
                let cms = self.oneside_corner_min3(entry, oc.ep.to, &am_by_l5);
                for j in 0..K {
                    let cm = cms[j];
                    if !cm.is_finite() {
                        continue;
                    }
                    let cand = entry.brd[j] + self.lambda[j] * (size + max_ninio) + cm + edge[j];
                    if best[j].as_ref().is_none_or(|b| cand < b.0) {
                        best[j] = Some((cand, entry, n1 as usize, n2));
                    }
                }
            }
        }
        if best[0].is_none() {
            assert!(
                best.iter().all(Option::is_none),
                "a one-sided winner in some lanes only"
            );
            return;
        }
        let winners = std::array::from_fn(|l| {
            let (_, e, n1, n2) = best[l].expect("a one-sided winner in every lane");
            (e.io, e.ikey, e.st, n1, n2)
        });
        self.rescore_winners(w, oc, winners);
    }

    fn oneside_corner_min3(
        &self,
        entry: &OneSideEntry<K>,
        ep_to: Node,
        am_by_l5: &[[f64; 4]; K],
    ) -> [f64; K] {
        let mut cm = [f64::INFINITY; K];
        for l5 in self.dfa.right(ep_to) {
            for l3 in self.dfa.left(entry.ikey.open) {
                let Some(mid) = self.dfa.broken_codon_cai_s::<S>(l5.to, l3.to) else {
                    continue;
                };
                let cail = l5.cai + mid + l3.cai;
                for l in 0..K {
                    let a = am_by_l5[l][l5.nuc as usize];
                    if !a.is_finite() {
                        continue;
                    }
                    let bm = entry.bmin[l][l3.nuc as usize];
                    if bm.is_finite() {
                        cm[l] = cm[l].min(a + bm - self.kk(l) * cail);
                    }
                }
            }
        }
        cm
    }

    // ---- the closed column's own closures ---------------------------------

    fn p_beam_all(&self, t: &mut LaneTables<K>, pos: u32) {
        for &jn in self.dfa.nodes_at(pos) {
            for (key, st) in t.p_states(jn) {
                let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                if let Some(prefix) = t.c[NodeLayout::ix_s::<S>(key.open)] {
                    let au = self.terminal_au(pair);
                    let cand = St {
                        mfe: std::array::from_fn(|l| prefix.mfe[l] + st.mfe[l] + au.0),
                        cai: std::array::from_fn(|l| prefix.cai[l] + st.cai[l]),
                    };
                    t.update_c(jn, cand);
                }
                let stem = self.m_stem_cost(pair);
                t.update_m(jn, key.open, st.plus_mfe(stem));
            }
        }
        self.bifurcate_fill_column(t, pos);
    }

    fn bifurcate_fill_column(&self, t: &mut LaneTables<K>, pos: u32) {
        let lambda = self.lambda;
        for &o in self.dfa.nodes_at(pos) {
            let cells: Vec<(Node, St<K>)> = t
                .p_states(o)
                .into_iter()
                .map(|(key, st)| {
                    let stem =
                        self.m_stem_cost(pair_from_nucleotides(key.nuc_open(), key.nuc_close));
                    (key.open, st.plus_mfe(stem))
                })
                .collect();
            if cells.is_empty() {
                continue;
            }
            let slots = cells
                .iter()
                .map(|(open, _)| t.m[NodeLayout::ix_s::<S>(*open)].mfe.len())
                .max()
                .unwrap_or(0);
            let mut found: Vec<(usize, St<K>)> = Vec::new();
            {
                let mut bm = vec![[EMPTY; K]; slots];
                let mut bc = vec![[0.0f64; K]; slots];
                let mut bb = vec![[f64::INFINITY; K]; slots];
                for (open, add) in &cells {
                    let col = &t.m[NodeLayout::ix_s::<S>(*open)];
                    for k in 0..col.mfe.len() {
                        if !col.occupied(k) {
                            continue;
                        }
                        for l in 0..K {
                            let (mfe, cai) =
                                (col.mfe[k][l] + add.mfe[l], col.cai[k][l] + add.cai[l]);
                            let brd =
                                lambda[l] * f64::from(mfe) - (1.0 - lambda[l]) * (cai * 100.0);
                            if brd < bb[k][l] {
                                (bm[k][l], bc[k][l], bb[k][l]) = (mfe, cai, brd);
                            }
                        }
                    }
                }
                for k in 0..slots {
                    if bm[k][0] != EMPTY {
                        found.push((
                            k,
                            St {
                                mfe: bm[k],
                                cai: bc[k],
                            },
                        ));
                    }
                }
            }
            for (i, st) in found {
                t.update_m2(o, self.dfa.node_of(i), st);
            }
        }
    }

    fn index_finalized_p(
        &self,
        t: &LaneTables<K>,
        hx: &mut HIndex<K>,
        osx: &mut OneSideIndex<K>,
        okx: &mut OneSideKIndex<K>,
        pos: u32,
    ) {
        if pos == 0 {
            return;
        }
        let cells: Vec<(Node, PairKey, St<K>)> = self
            .dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&io| t.p_states(io).into_iter().map(move |(k, s)| (io, k, s)))
            .collect();
        let l = pos - 1;
        for &(io, ikey, st) in &cells {
            let k = ikey.open.pos;
            hx.insert(
                self.h_slot(l as i64 - k as i64, self.reach_of(io)),
                HEntry {
                    k,
                    h: self.inner_h(io, ikey, &st),
                    io,
                    ikey,
                    st,
                },
            );
        }
        for &(io, ikey, st) in &cells {
            let entry = OneSideEntry {
                k: ikey.open.pos,
                io,
                ikey,
                st,
                brd: std::array::from_fn(|j| st.brd(j, self.lambda[j])),
                bmin: self.oneside_bmin(ikey),
            };
            osx.push(l, entry);
            okx.push(
                entry.k,
                OneSideEntry {
                    bmin: self.oneside_bmin3(io, ikey),
                    ..entry
                },
            );
        }
        osx.sort_col(l);
    }

    fn inner_h(&self, io: Node, ikey: PairKey, st: &St<K>) -> [f64; K] {
        let inner_rev = pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close).reversed();
        let mm = &self.params.mismatch_interior;
        let mut imin = [f64::INFINITY; K];
        for l3 in self.dfa.left(ikey.open) {
            let dl = self.end_contrib(l3.to) + l3.cai;
            for r5 in self.dfa.right(io) {
                let e = f64::from(mm.get(inner_rev, r5.nuc.to_nuc5(), l3.nuc.to_nuc5()));
                let dr = r5.cai + self.start_contrib(r5.to);
                for l in 0..K {
                    imin[l] = imin[l].min(self.lambda[l] * e - self.kk(l) * (dl + dr));
                }
            }
        }
        std::array::from_fn(|l| st.brd(l, self.lambda[l]) + imin[l])
    }

    fn oneside_bmin(&self, ikey: PairKey) -> [[f64; 4]; K] {
        let irev = pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close).reversed();
        let mm = &self.params.mismatch_interior;
        let mut bmin = [[f64::INFINITY; 4]; K];
        for l3 in self.dfa.left(ikey.open) {
            let dl = self.end_contrib(l3.to) + l3.cai;
            let l3n = l3.nuc.to_nuc5();
            for r5 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = f64::from(mm.get(irev, r5.to_nuc5(), l3n));
                let idx = r5 as usize;
                for l in 0..K {
                    let v = self.lambda[l] * e - self.kk(l) * dl;
                    if v < bmin[l][idx] {
                        bmin[l][idx] = v;
                    }
                }
            }
        }
        bmin
    }

    fn oneside_bmin3(&self, io: Node, ikey: PairKey) -> [[f64; 4]; K] {
        let irev = pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close).reversed();
        let mm = &self.params.mismatch_interior;
        let mut bmin = [[f64::INFINITY; 4]; K];
        for r5 in self.dfa.right(io) {
            let dr = r5.cai + self.start_contrib(r5.to);
            let r5n = r5.nuc.to_nuc5();
            for l3 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = f64::from(mm.get(irev, r5n, l3.to_nuc5()));
                let idx = l3 as usize;
                for l in 0..K {
                    let v = self.lambda[l] * e - self.kk(l) * dr;
                    if v < bmin[l][idx] {
                        bmin[l][idx] = v;
                    }
                }
            }
        }
        bmin
    }

    // ---- M2 and the multi-loop openings ------------------------------------

    fn m2_fill_column(&self, t: &mut LaneTables<K>, pos: u32) {
        let cols: Vec<(Node, Vec<(Node, St<K>)>)> = self
            .dfa
            .nodes_at(pos)
            .iter()
            .map(|&o| (o, t.m2[NodeLayout::ix_s::<S>(o)].states()))
            .collect();
        for (o, cells) in &cols {
            for &(inner, st) in cells {
                t.update_m(*o, inner, st);
            }
        }
        let ls = self.single_max_len as u32;
        let inners = cols.iter().flat_map(|(_, c)| c.iter().map(|(i, _)| i.pos));
        let (Some(lo), Some(hi)) = (inners.clone().min(), inners.max()) else {
            return;
        };
        let boundary = self.dfa.next_codon_boundary(pos);
        for p in lo.saturating_sub(ls)..hi {
            let (opens, multis) = self.m2_opening(&cols, pos, boundary, p);
            for (key, st) in opens {
                if boundary == pos {
                    t.multi_open_insert(key, st);
                } else {
                    t.multi_open_defer(boundary, key, st);
                }
            }
            for (outer, key, st) in multis {
                t.update_multi(outer, key, st);
            }
        }
        for (o, cells) in &cols {
            if cells.is_empty() {
                continue;
            }
            let end = self.core_end(*o);
            let mut far = FxHashMap::default();
            for &(inner, st) in cells {
                self.offer_far(inner, &st, &mut far);
            }
            self.far_multi_open(t, &end, far);
        }
    }

    #[allow(clippy::type_complexity)]
    fn m2_opening(
        &self,
        cols: &[(Node, Vec<(Node, St<K>)>)],
        pos: u32,
        boundary: u32,
        p: u32,
    ) -> (Vec<(MultiOpenKey, St<K>)>, Vec<Cand<K>>) {
        let mut opens: Vec<(MultiOpenKey, St<K>)> = Vec::new();
        let mut multis: Vec<Cand<K>> = Vec::new();
        let ls = self.single_max_len as u32;
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                for (o, cells) in cols {
                    let from = cells.partition_point(|(i, _)| i.pos <= p);
                    let to = cells.partition_point(|(i, _)| i.pos <= p + ls);
                    for &(inner, st) in &cells[from..to] {
                        let Some(flank_cai) = self.dfa.broken_codon_cai_s::<S>(ep.to, inner) else {
                            continue;
                        };
                        let left_cai = ep.cai + flank_cai;
                        for q in pos..boundary {
                            for &qn in self.dfa.nodes_at(q) {
                                let Some(right_cai) = self.dfa.broken_codon_cai_s::<S>(*o, qn)
                                else {
                                    continue;
                                };
                                for eq in self.dfa.right(qn) {
                                    if canonical(ep.nuc, eq.nuc).is_none() {
                                        continue;
                                    }
                                    let key = PairKey {
                                        open: pn,
                                        close: qn,
                                        opening: ep.opening,
                                        nuc_close: eq.nuc,
                                    };
                                    let cand = St {
                                        mfe: st.mfe,
                                        cai: st.cai.map(|c| c + left_cai + right_cai + eq.cai),
                                    };
                                    match multis
                                        .iter_mut()
                                        .find(|(oo, kk, _)| *oo == eq.to && *kk == key)
                                    {
                                        Some((_, _, best)) => {
                                            merge_lanes(best, &cand, &self.lambda)
                                        }
                                        None => multis.push((eq.to, key, cand)),
                                    }
                                }
                            }
                        }
                        let Some(c_outer) = self.dfa.span_head_cai(*o) else {
                            continue;
                        };
                        let key = MultiOpenKey {
                            open: pn,
                            nuc_open: ep.nuc,
                            lane_open: ep.to.lane,
                            reach: self.reach_of(*o),
                        };
                        let cand = St {
                            mfe: st.mfe,
                            cai: st.cai.map(|c| c + left_cai + c_outer),
                        };
                        match opens.iter_mut().find(|(kk, _)| *kk == key) {
                            Some((_, best)) => merge_lanes(best, &cand, &self.lambda),
                            None => opens.push((key, cand)),
                        }
                    }
                }
            }
        }
        (opens, multis)
    }

    fn core_end(&self, core_outer: Node) -> CoreEnd {
        let boundary = self.dfa.next_codon_boundary(core_outer.pos);
        let mut closes = Vec::new();
        for q in core_outer.pos..boundary {
            for &qn in self.dfa.nodes_at(q) {
                if let Some(right_cai) = self.dfa.broken_codon_cai_s::<S>(core_outer, qn) {
                    closes.push((qn, right_cai));
                }
            }
        }
        CoreEnd {
            outer: core_outer,
            closes,
            boundary,
            c_outer: self.dfa.span_head_cai(core_outer),
        }
    }

    fn offer_far(
        &self,
        core_inner: Node,
        st: &St<K>,
        far: &mut FxHashMap<MultiOpenKey, FarOpening<K>>,
    ) {
        let near = core_inner.pos.saturating_sub(self.single_max_len as u32);
        if near == 0 {
            return;
        }
        for &pn in self.dfa.nodes_at(near - 1) {
            for ep in self.dfa.right(pn) {
                let Some(flank) = self.dfa.broken_codon_cai_s::<S>(ep.to, core_inner) else {
                    continue;
                };
                let cand = FarOpening {
                    core_inner: [core_inner; K],
                    st: *st,
                    ep: [*ep; K],
                    cai: st.cai.map(|c| c + ep.cai + flank),
                };
                let key = MultiOpenKey {
                    open: pn,
                    nuc_open: ep.nuc,
                    lane_open: ep.to.lane,
                    reach: 0,
                };
                cand.merge_into(far, key, &self.lambda);
            }
        }
    }

    fn far_multi_open(
        &self,
        t: &mut LaneTables<K>,
        end: &CoreEnd,
        mut far: FxHashMap<MultiOpenKey, FarOpening<K>>,
    ) {
        let Some(top) = far.keys().map(|k| k.open.pos).max() else {
            return;
        };
        for p in (0..=top).rev() {
            for &pn in self.dfa.nodes_at(p) {
                for ep in self.dfa.right(pn) {
                    let key = MultiOpenKey {
                        open: pn,
                        nuc_open: ep.nuc,
                        lane_open: ep.to.lane,
                        reach: 0,
                    };
                    let Some(f) = far.remove(&key) else {
                        continue;
                    };
                    self.seed_multi_open(t, end, &f, pn);
                    if p == 0 {
                        continue;
                    }
                    for l in self.dfa.left(pn) {
                        let cand = FarOpening {
                            ep: [Edge { to: pn, ..*l }; K],
                            cai: f.cai.map(|c| c + l.cai),
                            ..f
                        };
                        let key = MultiOpenKey {
                            open: l.to,
                            nuc_open: l.nuc,
                            lane_open: pn.lane,
                            reach: 0,
                        };
                        cand.merge_into(&mut far, key, &self.lambda);
                    }
                }
            }
        }
    }

    /// [`crate::scan`]'s `seed_multi_open` for a far opening, whose core and
    /// opening edge are each lane's own.
    fn seed_multi_open(&self, t: &mut LaneTables<K>, end: &CoreEnd, f: &FarOpening<K>, pn: Node) {
        let flank: [Option<f64>; K] = std::array::from_fn(|l| {
            self.dfa
                .broken_codon_cai_s::<S>(f.ep[l].to, f.core_inner[l])
        });
        if flank[0].is_none() {
            assert!(
                flank.iter().all(Option::is_none),
                "a far flank in some lanes only"
            );
            return;
        }
        let left_cai: [f64; K] =
            std::array::from_fn(|l| f.ep[l].cai + flank[l].expect("a far flank in every lane"));
        let nuc_open = f.ep[0].nuc;
        for &(qn, right_cai) in &end.closes {
            for eq in self.dfa.right(qn) {
                if canonical(nuc_open, eq.nuc).is_none() {
                    continue;
                }
                let key = PairKey {
                    open: pn,
                    close: qn,
                    opening: f.ep[0].opening,
                    nuc_close: eq.nuc,
                };
                let cand = St {
                    mfe: f.st.mfe,
                    cai: std::array::from_fn(|l| f.st.cai[l] + left_cai[l] + right_cai + eq.cai),
                };
                t.update_multi(eq.to, key, cand);
            }
        }
        let Some(c_outer) = end.c_outer else {
            return;
        };
        let key = MultiOpenKey {
            open: pn,
            nuc_open,
            lane_open: f.ep[0].to.lane,
            reach: self.reach_of(end.outer),
        };
        let cand = St {
            mfe: f.st.mfe,
            cai: std::array::from_fn(|l| f.st.cai[l] + left_cai[l] + c_outer),
        };
        if end.boundary == end.outer.pos {
            t.multi_open_insert(key, cand);
        } else {
            t.multi_open_defer(end.boundary, key, cand);
        }
    }

    fn multi_open_close(
        &self,
        t: &mut LaneTables<K>,
        qn: Node,
        openings: &[(MultiOpenKey, St<K>)],
    ) {
        let Some(d_qn) = self.dfa.span_tail_cai(qn) else {
            return;
        };
        for &(key, state) in openings {
            if !self.admits(key.reach, qn) {
                continue;
            }
            for eq in self.dfa.right(qn) {
                if canonical(key.nuc_open, eq.nuc).is_none() {
                    continue;
                }
                let pk = PairKey {
                    open: key.open,
                    close: qn,
                    opening: crate::state::Opening::new(key.nuc_open, key.lane_open),
                    nuc_close: eq.nuc,
                };
                let cand = St {
                    mfe: state.mfe,
                    cai: state.cai.map(|c| c + d_qn + eq.cai),
                };
                t.update_multi(eq.to, pk, cand);
            }
        }
    }

    fn m_extend(&self, t: &mut LaneTables<K>, jn: Node) {
        for (inner, st) in t.m[NodeLayout::ix_s::<S>(jn)].states() {
            for e in self.dfa.right(jn) {
                t.update_m(e.to, inner, st.plus_cai(e.cai));
            }
        }
    }

    fn c_extend(&self, t: &mut LaneTables<K>, jn: Node) {
        let Some(st) = t.c[NodeLayout::ix_s::<S>(jn)] else {
            return;
        };
        for e in self.dfa.right(jn) {
            t.update_c(e.to, st.plus_cai(e.cai));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dfa::{self, UtrChain};
    use crate::region::PairConstraint;
    use crate::{CodonUsage, Params, ScanConfig, UtrProfile};
    use derna_codon_dp::codon_table::{aa_to_index, STOP};
    use rna_mfe::Nucleotide::{A, C, G, U};

    /// On a stop-labelled lattice the K-weight scan fills every lane's tables
    /// exactly as a single-weight scan does, including every stop terminal: the
    /// lanes share the reductions whose inner pairs are kept apart by stop.
    #[test]
    fn labelled_lanes_are_the_single_scans() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("codon usage");
        let cases: [(&str, &[rna_mfe::Nucleotide], &[rna_mfe::Nucleotide]); 2] = [
            (
                "MKWV",
                &[G, G, C, C, A],
                &[A, G, G, C, C, U, U, C, G, G, C, C, A, U, G, G, C, C],
            ),
            (
                "MCRGW",
                &[C, C, G, U],
                &[G, C, G, C, A, A, A, G, C, G, C, U, U, A, C, G, G, A],
            ),
        ];
        let lambdas = [0.1, 0.4, 0.7, 1.0];
        for (protein, utr5, utr3) in cases {
            let mut aa: Vec<usize> = protein
                .chars()
                .map(|c| aa_to_index(c).expect("residue"))
                .collect();
            aa.push(STOP);
            let graph = dfa::build_stop_labelled(
                &aa,
                &usage,
                UtrChain::Fixed(utr5),
                UtrChain::Fixed(utr3),
                UtrProfile::inert(),
            )
            .expect("a designed stop");
            let config = ScanConfig::default();
            let (lanes, _) = crate::outside_multi::scan_lanes(
                &params,
                &graph,
                &lambdas,
                config,
                &PairConstraint::inert(),
            );
            for (lambda, lane) in lambdas.into_iter().zip(&lanes) {
                let alone = crate::scan::scan(&params, &graph, lambda, config);
                for &end in graph.ends() {
                    let (a, b) = (alone.c[graph.ix(end)], lane.c[graph.ix(end)]);
                    assert_eq!(
                        a.map(|s| (s.mfe, s.cai.to_bits())),
                        b.map(|s| (s.mfe, s.cai.to_bits())),
                        "{protein} λ={lambda} terminal {end:?}"
                    );
                }
                for o in 0..graph.slots() {
                    let mut x: Vec<_> = alone.p[o]
                        .iter()
                        .map(|(k, s)| (format!("{k:?}"), s.mfe, s.cai.to_bits()))
                        .collect();
                    let mut y: Vec<_> = lane.p[o]
                        .iter()
                        .map(|(k, s)| (format!("{k:?}"), s.mfe, s.cai.to_bits()))
                        .collect();
                    x.sort();
                    y.sort();
                    assert_eq!(x, y, "{protein} λ={lambda} P column {o}");
                }
            }
        }
    }
}
