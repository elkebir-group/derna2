//! The cost to go of several weights' scans in one backward sweep.
//!
//! [`OutsideScan::cost_to_go`](crate::outside_scan::OutsideScan) computes every
//! cell's cost to go over one scan's tables. Exact scans of one lattice at
//! different weights `λ` hold the same cells, and the pass reads `λ` only
//! through the values it adds: the walk over the lattice, the cells it visits,
//! the hash keys it looks up and the loop energies it evaluates are the same at
//! every weight. This module takes the pass once for `K` such scans. Each cell
//! holds the `K` weights' values side by side, one lane per scan, so one visit,
//! one lookup and one energy evaluation serve every lane.
//!
//! Each lane's arithmetic is the single pass's, term for term and in the same
//! order, so each lane equals [`OutsideScan::cost_to_go`] at its scan's weight
//! bit for bit. The walk follows the first scan's tables, and the other lanes
//! read their inside values by the first scan's keys; every scan must hold the
//! same cells ([`same_cells`]).
//!
//! Only the values are computed: no edge marginals, no split-loop validation
//! and no visit tally.

// A lane loop reads the same lane of several arrays at once.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rna_mfe::loops::{
    bulge_energy, internal_energy_scalar, pair_from_nucleotides, stack_energy, BulgeLoop,
    InternalLoopScalar, StackPair,
};
use rna_mfe::params::{BasePair, Nuc5};
use rna_mfe::{Energy, Nucleotide, Params};

use crate::dfa::{Dfa, Edge, Node, NodeLayout};
use crate::hash::FxHashMap;
use crate::interior::{LoopPairs, OneByNRows};
use crate::outside_scan::{CellCosts, OutsideScan, SinglesTally};
use crate::region::{PairConstraint, Verdict};
use crate::state::{PairKey, State};
use crate::tables::Tables;
use crate::{BeamError, CodonUsage, RegionConstraints, ScanConfig};

const INF: f64 = f64::INFINITY;

/// Closed pairs a worker takes from the shared counter at a time, as in the
/// single pass.
const SINGLES_BATCH: usize = 4;

/// The most scans one sweep takes.
pub const MAX_LANES: usize = 8;

/// One value per lane.
type Vals<const K: usize> = [f64; K];

fn relax(slot: &mut f64, v: f64) {
    if v < *slot {
        *slot = v;
    }
}

/// [`relax`] lane by lane.
fn relax_all<const K: usize>(slot: &mut Vals<K>, v: &Vals<K>) {
    for (s, &x) in slot.iter_mut().zip(v) {
        relax(s, x);
    }
}

/// `λ·mfe − (1 − λ)·(cai·100)`: the tables' own scalarization of a cell.
fn brd(mfe: i32, cai: f64, lambda: f64) -> f64 {
    lambda * f64::from(mfe) - (1.0 - lambda) * (cai * 100.0)
}

/// The internal-loop shapes the energy model scores as size, asymmetry and
/// the two terminal mismatches alone.
fn is_generic(n1: usize, n2: usize) -> bool {
    n1 >= 2 && n2 >= 2 && !matches!((n1, n2), (2, 2) | (2, 3) | (3, 2))
}

/// Every lane's cost to go, cell by cell, as [`CellCosts`] holds one.
pub(crate) struct MultiCosts<const K: usize> {
    c: Vec<Vals<K>>,
    m: Vec<Vec<Vals<K>>>,
    m2: Vec<Vec<Vals<K>>>,
    p: Vec<FxHashMap<PairKey, Vals<K>>>,
    multi: Vec<FxHashMap<PairKey, Vals<K>>>,
}

impl<const K: usize> MultiCosts<K> {
    /// One [`CellCosts`] per lane. The tally is left empty.
    pub(crate) fn split(self) -> Vec<CellCosts> {
        let lane = |l: usize| CellCosts {
            c: self.c.iter().map(|v| v[l]).collect(),
            m: cols(&self.m, l),
            m2: cols(&self.m2, l),
            p: maps(&self.p, l),
            multi: maps(&self.multi, l),
            tally: SinglesTally::default(),
        };
        (0..K).map(lane).collect()
    }
}

fn cols<const K: usize>(t: &[Vec<Vals<K>>], l: usize) -> Vec<Vec<f64>> {
    t.iter()
        .map(|col| col.iter().map(|v| v[l]).collect())
        .collect()
}

fn maps<const K: usize>(
    t: &[FxHashMap<PairKey, Vals<K>>],
    l: usize,
) -> Vec<FxHashMap<PairKey, f64>> {
    t.iter()
        .map(|m| m.iter().map(|(&k, v)| (k, v[l])).collect())
        .collect()
}

/// The cost to go of every cell of each of `tables`, at each one's own weight,
/// in one sweep: lane `l` equals
/// [`OutsideScan::cost_to_go`]`(params, dfa, tables[l], region, single_max_len,
/// threads, false)`.
pub(crate) fn cost_to_go_multi<const K: usize>(
    params: &Params,
    dfa: &Dfa,
    tables: [&Tables; K],
    region: &PairConstraint,
    single_max_len: usize,
    threads: usize,
) -> MultiCosts<K> {
    debug_assert!(same_cells(dfa, &tables).is_ok(), "the lanes' tables differ");
    // The sweep is monomorphized over the lattice's lane stride, so its index
    // arithmetic multiplies by a constant and a lattice without stop lanes
    // compiles out every test for one.
    match dfa.stride() {
        2 => sweep_strided::<K, 2>(params, dfa, tables, region, single_max_len, threads),
        3 => sweep_strided::<K, 3>(params, dfa, tables, region, single_max_len, threads),
        stride => unreachable!("a lattice's lane stride is 2 or 3, not {stride}"),
    }
}

/// [`cost_to_go_multi`] over a lattice whose lane stride is `S`.
fn sweep_strided<const K: usize, const S: usize>(
    params: &Params,
    dfa: &Dfa,
    tables: [&Tables; K],
    region: &PairConstraint,
    single_max_len: usize,
    threads: usize,
) -> MultiCosts<K> {
    let mut scan = MultiScan::<K, S>::new(params, dfa, tables, region, single_max_len, threads);
    scan.index_outers();
    scan.sweep();
    MultiCosts {
        c: scan.out_c,
        m: scan.out_m,
        m2: scan.out_m2,
        p: scan.out_p,
        multi: scan.out_multi,
    }
}

/// [`cost_to_go_multi`] over any number of scans up to [`MAX_LANES`], with the
/// time spent splitting the lanes into one [`CellCosts`] each.
pub(crate) fn cost_to_go_lanes(
    params: &Params,
    dfa: &Dfa,
    tables: &[&Tables],
    region: &PairConstraint,
    single_max_len: usize,
    threads: usize,
) -> (Vec<CellCosts>, Duration) {
    fn at<const K: usize>(
        params: &Params,
        dfa: &Dfa,
        tables: &[&Tables],
        region: &PairConstraint,
        single_max_len: usize,
        threads: usize,
    ) -> (Vec<CellCosts>, Duration) {
        let tables: [&Tables; K] = tables.try_into().expect("K tables");
        let costs = cost_to_go_multi(params, dfa, tables, region, single_max_len, threads);
        let start = Instant::now();
        let lanes = costs.split();
        (lanes, start.elapsed())
    }
    let args = (params, dfa, tables, region, single_max_len, threads);
    match tables.len() {
        1 => at::<1>(args.0, args.1, args.2, args.3, args.4, args.5),
        2 => at::<2>(args.0, args.1, args.2, args.3, args.4, args.5),
        3 => at::<3>(args.0, args.1, args.2, args.3, args.4, args.5),
        4 => at::<4>(args.0, args.1, args.2, args.3, args.4, args.5),
        5 => at::<5>(args.0, args.1, args.2, args.3, args.4, args.5),
        6 => at::<6>(args.0, args.1, args.2, args.3, args.4, args.5),
        7 => at::<7>(args.0, args.1, args.2, args.3, args.4, args.5),
        8 => at::<8>(args.0, args.1, args.2, args.3, args.4, args.5),
        k => panic!("{k} lanes: one sweep takes 1 to {MAX_LANES}"),
    }
}

/// Whether every scan of `tables` holds the cells of the first: the same `P`
/// and `Multi` keys, and the same occupied `C`, `M` and `M2` cells. True of
/// exact scans of one lattice under one constraint, whatever their weights.
pub(crate) fn same_cells(dfa: &Dfa, tables: &[&Tables]) -> Result<(), String> {
    let Some((first, rest)) = tables.split_first() else {
        return Ok(());
    };
    let width = dfa.slots();
    for (l, t) in rest.iter().enumerate() {
        let lane = l + 1;
        for o in 0..width {
            if first.c[o].is_some() != t.c[o].is_some() {
                return Err(format!("lane {lane}: C at node {o}"));
            }
            for (name, a, b) in [
                ("P", &first.p[o], &t.p[o]),
                ("Multi", &first.multi[o], &t.multi[o]),
            ] {
                if a.len() != b.len() || a.keys().any(|k| !b.contains_key(k)) {
                    return Err(format!("lane {lane}: {name} keys at node {o}"));
                }
            }
            for (name, a, b) in [("M", &first.m[o], &t.m[o]), ("M2", &first.m2[o], &t.m2[o])] {
                let (fa, _) = a.hot_slices(0, a.slots());
                let (fb, _) = b.hot_slices(0, b.slots());
                let occupied = |s: &[i32]| s.iter().map(|&m| m != i32::MAX).collect::<Vec<_>>();
                if occupied(fa) != occupied(fb) {
                    return Err(format!("lane {lane}: {name} cells at node {o}"));
                }
            }
        }
    }
    Ok(())
}

struct MultiScan<'a, const K: usize, const S: usize> {
    params: &'a Params,
    dfa: &'a Dfa,
    t: [&'a Tables; K],
    constraint: &'a PairConstraint,
    lambda: Vals<K>,
    kk: Vals<K>,
    single_max_len: usize,
    threads: usize,
    one_by_n: OneByNRows,
    /// This sweep's identity, so the thread-local strand slots can tell their
    /// own sweep's strands apart.
    id: u64,
    width: usize,
    out_c: Vec<Vals<K>>,
    out_m: Vec<Vec<Vals<K>>>,
    out_m2: Vec<Vec<Vals<K>>>,
    out_p: Vec<FxHashMap<PairKey, Vals<K>>>,
    out_multi: Vec<FxHashMap<PairKey, Vals<K>>>,
    /// The best frontier at or past each position, per opening key.
    frontier_at: Vec<Vec<Vals<K>>>,
    outers: Vec<Outer<K>>,
    outer_id: FxHashMap<(usize, PairKey), (u32, u32)>,
    by_close: Vec<Vec<u32>>,
    walk_close: Vec<Vec<Walk<K>>>,
    walk_at: Vec<(u32, u32)>,
    runs: Vec<Vec<Run<K>>>,
    /// `λ·(size + asymmetry)` of a generic loop, by `(n1, n2)`, per lane.
    shape: Vec<Vals<K>>,
    shape_e: Vec<i32>,
}

/// A closed pair as the enclosing pair of single-branch loops.
struct Outer<const K: usize> {
    pn: Node,
    qn: Node,
    ep: Edge,
    pair: BasePair,
    stat: Vals<K>,
    ao_best: Vals<K>,
    out: Vals<K>,
}

/// The fields of an [`Outer`] the single-branch walk reads, in walk order.
struct Walk<const K: usize> {
    pn: Node,
    qn: Node,
    to: Node,
    pair: BasePair,
    stat: Vals<K>,
    ao_best: Vals<K>,
    out: Vals<K>,
}

/// A run of [`MultiScan::walk_close`] at one closing position: the pairs that
/// open at one position by an edge into one node.
#[derive(Clone)]
struct Run<const K: usize> {
    p: u32,
    to: Node,
    start: u32,
    end: u32,
    /// Whether any lane of any pair has a finite outside value.
    finite: bool,
    /// The least split outer half, `out + stat + ao_best`, per lane.
    best: Vals<K>,
}

/// A strand's terminal-mismatch choice.
#[derive(Clone, Copy)]
struct Corner {
    first: Nucleotide,
    last: Nucleotide,
    cai: f64,
}

impl<'a, const K: usize, const S: usize> MultiScan<'a, K, S> {
    /// Whether a strand into `to` may be scored as two halves minimized apart
    /// ([`Dfa::halves_combine`]); always so on a lattice whose positions hold
    /// no stop lane.
    #[inline]
    fn halves_combine(&self, to: Node) -> bool {
        S == 2 || Dfa::halves_combine(self.dfa, to)
    }

    fn new(
        params: &'a Params,
        dfa: &'a Dfa,
        t: [&'a Tables; K],
        constraint: &'a PairConstraint,
        single_max_len: usize,
        threads: usize,
    ) -> Self {
        let lambda = t.map(Tables::lambda);
        let width = dfa.slots();
        let positions = dfa.seq_len as usize + 1;
        Self {
            params,
            dfa,
            t,
            constraint,
            lambda,
            kk: lambda.map(|l| (1.0 - l) * 100.0),
            single_max_len,
            threads: threads.max(1),
            one_by_n: OneByNRows::of(params),
            id: PASS_IDS.fetch_add(1, Ordering::Relaxed),
            width,
            out_c: vec![[INF; K]; width],
            out_m: (0..width).map(|i| vec![[INF; K]; i]).collect(),
            out_m2: (0..width).map(|i| vec![[INF; K]; i]).collect(),
            out_p: vec![FxHashMap::default(); width],
            out_multi: vec![FxHashMap::default(); width],
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

    fn single_span(&self) -> usize {
        self.single_max_len + 1
    }

    fn sweep(&mut self) {
        // Every terminal completes for free, as in the single-weight pass.
        for &end in self.dfa.completion_ends() {
            self.out_c[NodeLayout::ix_s::<S>(end)] = [0.0; K];
        }
        for pos in (0..=self.dfa.seq_len).rev() {
            if pos < self.dfa.seq_len {
                self.extensions(pos);
            }
            self.frontiers(pos);
            self.m2_tails(pos);
            let cells = self.p_cells(pos);
            let single_best = self.singles_all(&cells);
            self.p_tails(&cells, &single_best);
            if pos > 0 {
                self.fill_runs(pos - 1);
            }
            self.multi_tails(pos);
            // A position reads its own frontier, the next one's and the one at
            // its codon boundary, at most two past it, so the rows behind are
            // freed: a few rows live, not one per position.
            if let Some(row) = self.frontier_at.get_mut(pos as usize + 3) {
                *row = Vec::new();
            }
        }
    }

    // ---- inside reads ------------------------------------------------------

    fn in_c(&self, n: Node) -> Vals<K> {
        std::array::from_fn(|l| {
            self.t[l].c[NodeLayout::ix_s::<S>(n)].map_or(INF, |s| s.brd(self.lambda[l]))
        })
    }

    /// The occupied inner nodes of the first lane's `M` (or `M2`) column.
    fn inners(&self, col: &crate::tables::DenseCol) -> Vec<Node> {
        col.brd_cells(self.lambda[0]).map(|(i, _)| i).collect()
    }

    fn fill(&self, a: Node, b: Node) -> Option<f64> {
        self.dfa.broken_codon_cai_s::<S>(a, b)
    }

    fn e(&self, energy: Energy) -> Vals<K> {
        self.lambda.map(|l| l * f64::from(energy.0))
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

    fn constraint(&self, i: u32, j: u32) -> Option<Vals<K>> {
        match self.constraint.verdict(i, j) {
            Verdict::Allow => Some([0.0; K]),
            Verdict::Forbid => None,
            Verdict::Penalize(pen) => Some(self.lambda.map(|l| l * f64::from(pen.0))),
        }
    }

    fn open_key(&self, pn: Node, nuc: Nucleotide) -> usize {
        NodeLayout::ix_s::<S>(pn) * 4 + nuc as usize
    }

    /// [`Self::open_key`] with the opening edge's lane, for a cache whose
    /// value is looked up under one exact opening and so cannot be shared
    /// between lanes.
    fn open_edge_key(&self, pn: Node, ep: Edge) -> usize {
        crate::state::lane_key(self.open_key(pn, ep.nuc), ep.to.lane)
    }

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

    // ---- the backward sweep ------------------------------------------------

    fn extensions(&mut self, pos: u32) {
        let (dfa, t0) = (self.dfa, self.t[0]);
        for &jn in dfa.nodes_at(pos) {
            for e in dfa.right(jn).to_vec() {
                let to = self.out_c[NodeLayout::ix_s::<S>(e.to)];
                let v: Vals<K> = std::array::from_fn(|l| to[l] - self.kk[l] * e.cai);
                relax_all(&mut self.out_c[NodeLayout::ix_s::<S>(jn)], &v);
                for inner in self.inners(&t0.m[NodeLayout::ix_s::<S>(jn)]) {
                    let to = self.out_m[NodeLayout::ix_s::<S>(e.to)][NodeLayout::ix_s::<S>(inner)];
                    let v: Vals<K> = std::array::from_fn(|l| to[l] - self.kk[l] * e.cai);
                    relax_all(
                        &mut self.out_m[NodeLayout::ix_s::<S>(jn)][NodeLayout::ix_s::<S>(inner)],
                        &v,
                    );
                }
            }
        }
    }

    fn frontiers(&mut self, pos: u32) {
        let dfa = self.dfa;
        let mut best = if (pos as usize + 1) < self.frontier_at.len() {
            self.frontier_at[pos as usize + 1].clone()
        } else {
            vec![[INF; K]; self.width * 4]
        };
        if pos < dfa.seq_len {
            for &qn in dfa.nodes_at(pos) {
                let Some(d) = dfa.span_tail_cai(qn) else {
                    continue;
                };
                for eq in dfa.right(qn).to_vec() {
                    let cells: Vec<(PairKey, Vals<K>)> = self.out_multi
                        [NodeLayout::ix_s::<S>(eq.to)]
                    .iter()
                    .filter(|(k, _)| k.close == qn && k.nuc_close == eq.nuc)
                    .map(|(k, v)| (*k, *v))
                    .collect();
                    for (key, out) in cells {
                        let val: Vals<K> =
                            std::array::from_fn(|l| out[l] - self.kk[l] * (d + eq.cai));
                        relax_all(&mut best[self.open_key(key.open, key.nuc_open())], &val);
                    }
                }
            }
        }
        self.frontier_at[pos as usize] = best;
    }

    fn m2_tails(&mut self, pos: u32) {
        let (dfa, t0) = (self.dfa, self.t[0]);
        for &jn in dfa.nodes_at(pos) {
            let cells = self.inners(&t0.m2[NodeLayout::ix_s::<S>(jn)]);
            let boundary = dfa.next_codon_boundary(jn.pos);
            let c_outer = dfa.span_head_cai(jn);
            let mut closes: Vec<(Node, Edge, f64)> = Vec::new();
            for q in jn.pos..boundary {
                for &qn in dfa.nodes_at(q) {
                    if let Some(right) = self.fill(jn, qn) {
                        closes.extend(dfa.right(qn).iter().map(|&eq| (qn, eq, right)));
                    }
                }
            }
            let span = self.single_max_len as u32;
            self.far_m2_tails(jn, &cells, boundary, c_outer);
            let mut wraps: FxHashMap<usize, Vec<(u32, Vals<K>)>> = FxHashMap::default();
            let mut flanks = Vec::new();
            for inner in cells {
                let head = self.out_m[NodeLayout::ix_s::<S>(jn)][NodeLayout::ix_s::<S>(inner)];
                relax_all(
                    &mut self.out_m2[NodeLayout::ix_s::<S>(jn)][NodeLayout::ix_s::<S>(inner)],
                    &head,
                );
                self.left_flanks_into(inner, inner.pos.saturating_sub(span), &mut flanks);
                for &(pn, ep, left) in &flanks {
                    let left_w: Vals<K> = self.kk.map(|kk| -kk * (ep.cai + left));
                    let found = wraps
                        .entry(self.open_edge_key(pn, ep))
                        .or_insert_with(|| self.wraps_found(pn, ep, &closes));
                    for &(ci, h) in found.iter() {
                        let (_, eq, right) = closes[ci as usize];
                        let v: Vals<K> = std::array::from_fn(|l| {
                            let w = left_w[l] - self.kk[l] * (right + eq.cai);
                            h[l] + w
                        });
                        relax_all(
                            &mut self.out_m2[NodeLayout::ix_s::<S>(jn)]
                                [NodeLayout::ix_s::<S>(inner)],
                            &v,
                        );
                    }
                    let Some(c) = c_outer else {
                        continue;
                    };
                    let g = self.frontier_at[boundary as usize][self.open_key(pn, ep.nuc)];
                    let v: Vals<K> = std::array::from_fn(|l| {
                        let w = left_w[l] - self.kk[l] * c;
                        g[l] + w
                    });
                    relax_all(
                        &mut self.out_m2[NodeLayout::ix_s::<S>(jn)][NodeLayout::ix_s::<S>(inner)],
                        &v,
                    );
                }
            }
        }
    }

    fn wraps_found(&self, pn: Node, ep: Edge, closes: &[(Node, Edge, f64)]) -> Vec<(u32, Vals<K>)> {
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
                let h = *self.out_multi[NodeLayout::ix_s::<S>(eq.to)].get(&key)?;
                Some((ci as u32, h))
            })
            .collect()
    }

    fn opening_outside(
        &self,
        jn: Node,
        boundary: u32,
        c_outer: Option<f64>,
        pn: Node,
        ep: Edge,
    ) -> Vals<K> {
        let mut best = [INF; K];
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
                    if let Some(h) = self.out_multi[NodeLayout::ix_s::<S>(eq.to)].get(&key) {
                        for l in 0..K {
                            relax(&mut best[l], h[l] - self.kk[l] * (right + eq.cai));
                        }
                    }
                }
            }
        }
        if let Some(c) = c_outer {
            let g = &self.frontier_at[boundary as usize][self.open_key(pn, ep.nuc)];
            for l in 0..K {
                relax(&mut best[l], g[l] - self.kk[l] * c);
            }
        }
        best
    }

    fn far_m2_tails(&mut self, jn: Node, cells: &[Node], boundary: u32, c_outer: Option<f64>) {
        let l = self.single_max_len as u32;
        let Some(top) = cells.iter().filter(|i| i.pos > l).map(|i| i.pos - l).max() else {
            return;
        };
        let dfa = self.dfa;
        let mut reach = vec![[INF; K]; self.width];
        for s in 1..=top {
            for &y in dfa.nodes_at(s) {
                let mut best = [INF; K];
                for e in dfa.left(y) {
                    let opened =
                        self.opening_outside(jn, boundary, c_outer, e.to, Edge { to: y, ..*e });
                    let r = &reach[NodeLayout::ix_s::<S>(e.to)];
                    for k in 0..K {
                        relax(&mut best[k], opened[k].min(r[k]) - self.kk[k] * e.cai);
                    }
                }
                reach[NodeLayout::ix_s::<S>(y)] = best;
            }
        }
        for &inner in cells {
            if inner.pos <= l {
                continue;
            }
            let mut best = [INF; K];
            for &y in dfa.nodes_at(inner.pos - l) {
                if let Some(rest) = self.fill(y, inner) {
                    let r = &reach[NodeLayout::ix_s::<S>(y)];
                    for k in 0..K {
                        relax(&mut best[k], r[k] - self.kk[k] * rest);
                    }
                }
            }
            relax_all(
                &mut self.out_m2[NodeLayout::ix_s::<S>(jn)][NodeLayout::ix_s::<S>(inner)],
                &best,
            );
        }
    }

    fn p_tails(&mut self, cells: &[(Node, PairKey, Vals<K>)], single_best: &[Vals<K>]) {
        let bifurcations = self.bifurcation_tails(cells);
        for (idx, &(jn, key, p_in)) in cells.iter().enumerate() {
            let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
            let mut out = [INF; K];
            let c_in = self.in_c(key.open);
            let au = self.e(self.terminal_au(pair));
            for l in 0..K {
                if c_in[l].is_finite() {
                    let h = self.out_c[NodeLayout::ix_s::<S>(jn)][l] + au[l];
                    relax(&mut out[l], h + c_in[l]);
                    relax(
                        &mut self.out_c[NodeLayout::ix_s::<S>(key.open)][l],
                        h + p_in[l],
                    );
                }
            }
            let stem = self.e(self.stem(pair));
            let m = self.out_m[NodeLayout::ix_s::<S>(jn)][NodeLayout::ix_s::<S>(key.open)];
            for l in 0..K {
                relax(&mut out[l], m[l] + stem[l]);
            }
            relax_all(&mut out, &bifurcations[idx]);
            relax_all(&mut out, &single_best[idx]);
            let slot = self.out_p[NodeLayout::ix_s::<S>(jn)]
                .entry(key)
                .or_insert([INF; K]);
            relax_all(slot, &out);
            let value = *slot;
            let (first, last) = self.outer_id[&(NodeLayout::ix_s::<S>(jn), key)];
            for id in first as usize..last as usize {
                self.outers[id].out = value;
                let (q, i) = self.walk_at[id];
                self.walk_close[q as usize][i as usize].out = value;
            }
        }
    }

    fn bifurcation_tails(&mut self, cells: &[(Node, PairKey, Vals<K>)]) -> Vec<Vals<K>> {
        let t = self.t;
        let lambda = self.lambda;
        let mut order: Vec<usize> = (0..cells.len()).collect();
        order.sort_by_key(|&i| {
            (
                NodeLayout::ix_s::<S>(cells[i].0),
                NodeLayout::ix_s::<S>(cells[i].1.open),
            )
        });
        let mut best = vec![[INF; K]; cells.len()];
        for group in
            order.chunk_by(|&i, &j| (cells[i].0, cells[i].1.open) == (cells[j].0, cells[j].1.open))
        {
            let (jn, open) = (cells[group[0]].0, cells[group[0]].1.open);
            let sides: Vec<(usize, Vals<K>, Vals<K>)> = group
                .iter()
                .map(|&c| {
                    let key = cells[c].1;
                    let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                    (c, self.e(self.stem(pair)), cells[c].2)
                })
                .collect();
            let hot: [(&[i32], &[f64]); K] = std::array::from_fn(|l| {
                let col = &t[l].m[NodeLayout::ix_s::<S>(open)];
                col.hot_slices(0, col.slots())
            });
            for (i, m_in0) in t[0].m[NodeLayout::ix_s::<S>(open)].brd_cells(lambda[0]) {
                let m_in: Vals<K> = std::array::from_fn(|l| {
                    if l == 0 {
                        m_in0
                    } else {
                        brd(
                            hot[l].0[NodeLayout::ix_s::<S>(i)],
                            hot[l].1[NodeLayout::ix_s::<S>(i)],
                            lambda[l],
                        )
                    }
                });
                let m2 = self.out_m2[NodeLayout::ix_s::<S>(jn)][NodeLayout::ix_s::<S>(i)];
                let slot = &mut self.out_m[NodeLayout::ix_s::<S>(open)][NodeLayout::ix_s::<S>(i)];
                for &(c, stem, p_in) in &sides {
                    for l in 0..K {
                        let h = m2[l] + stem[l];
                        relax(&mut best[c][l], h + m_in[l]);
                        relax(&mut slot[l], h + p_in[l]);
                    }
                }
            }
        }
        best
    }

    /// Every closed pair whose outer node sits at `pos`, with each lane's
    /// inside value read by the first lane's key.
    fn p_cells(&self, pos: u32) -> Vec<(Node, PairKey, Vals<K>)> {
        let (t, lambda) = (self.t, self.lambda);
        self.dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&jn| {
                t[0].p_states(jn).into_iter().map(move |(key, st)| {
                    let v: Vals<K> = std::array::from_fn(|l| {
                        if l == 0 {
                            st.brd(lambda[0])
                        } else {
                            t[l].p[NodeLayout::ix_s::<S>(jn)]
                                .get(&key)
                                .map_or(INF, |s| s.brd(lambda[l]))
                        }
                    });
                    (jn, key, v)
                })
            })
            .collect()
    }

    fn singles_all(&self, cells: &[(Node, PairKey, Vals<K>)]) -> Vec<Vals<K>> {
        let mut order: Vec<usize> = (0..cells.len()).collect();
        order.sort_by_key(|&i| {
            (
                NodeLayout::ix_s::<S>(cells[i].0),
                NodeLayout::ix_s::<S>(cells[i].1.open),
            )
        });
        let groups: Vec<&[usize]> = order
            .chunk_by(|&i, &j| (cells[i].0, cells[i].1.open) == (cells[j].0, cells[j].1.open))
            .collect();
        let one = |group: &[usize], done: &mut Vec<(usize, Vals<K>)>| {
            let keys: Vec<PairKey> = group.iter().map(|&i| cells[i].1).collect();
            let values = self.singles_group(cells[group[0]].0, &keys);
            done.extend(group.iter().copied().zip(values));
        };
        let mut best = vec![[INF; K]; cells.len()];
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
        let parts: Vec<Vec<(usize, Vals<K>)>> = std::thread::scope(|scope| {
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

    fn index_outers(&mut self) {
        let dfa = self.dfa;
        let table = &self.params.mismatch_interior;
        for oi in 0..self.width {
            let o = dfa.node_of(oi);
            if o.pos > dfa.seq_len {
                continue;
            }
            for (key, _) in self.t[0].p_states(o) {
                let (pn, qn) = (key.open, key.close);
                let eq = *dfa
                    .right(qn)
                    .iter()
                    .find(|e| e.nuc == key.nuc_close && e.to == o)
                    .expect("a pair's closing edge is in the lattice");
                let Some(constraint) = self.constraint(pn.pos, qn.pos) else {
                    continue;
                };
                let first = self.outers.len() as u32;
                let ep = key
                    .open_edge(dfa)
                    .expect("a pair's opening edge is in the lattice");
                self.index_outer(key, (ep, eq), constraint);
                let last = self.outers.len() as u32;
                assert!(first < last, "a pair is indexed as an enclosing pair");
                self.outer_id
                    .insert((NodeLayout::ix_s::<S>(o), key), (first, last));
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
                        let x = &outers[id as usize];
                        Walk {
                            pn: x.pn,
                            qn: x.qn,
                            to: x.ep.to,
                            pair: x.pair,
                            stat: x.stat,
                            ao_best: x.ao_best,
                            out: x.out,
                        }
                    })
                    .collect()
            })
            .collect();
        let l = self.single_span();
        self.shape = vec![[INF; K]; l * l];
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
                self.shape[n1 * l + n2] = self.lambda.map(|lam| lam * f64::from(whole.0 - mm));
                self.shape_e[n1 * l + n2] = whole.0 - mm;
            }
        }
    }

    fn index_outer(&mut self, key: PairKey, (ep, eq): (Edge, Edge), constraint: Vals<K>) {
        let (pn, qn) = (key.open, key.close);
        let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let ao_best = self.outer_ao_best(pair, ep.to, qn);
        let id = self.outers.len() as u32;
        self.by_close[qn.pos as usize].push(id);
        self.outers.push(Outer {
            pn,
            qn,
            ep,
            pair,
            stat: std::array::from_fn(|l| constraint[l] - self.kk[l] * (ep.cai + eq.cai)),
            ao_best,
            out: [INF; K],
        });
    }

    fn singles_group(&self, o: Node, keys: &[PairKey]) -> Vec<Vals<K>> {
        let a = keys[0].open.pos;
        let b = o.pos - 1;
        let l = self.single_span();
        let mut walk = GroupWalk::new(self, o, keys);
        let mut best = vec![[INF; K]; keys.len()];
        let q_top = (b + 1 + l as u32).min(self.dfa.seq_len);
        let p_lo = a.saturating_sub(l as u32);
        STRANDS.with_borrow_mut(|(opens, closes)| {
            opens.reset(StrandSlots::opening_len(l, self.dfa.stride()));
            if closes.owner != Some((self.id, o)) {
                closes.reset(StrandSlots::closing_len(l, self.dfa.stride()));
                closes.owner = Some((self.id, o));
            }
            for q in (b + 1)..q_top {
                let n2 = (q - o.pos) as usize;
                let p_min = (a + n2 as u32).saturating_sub(l as u32).max(p_lo);
                let runs = &self.runs[q as usize];
                let start = runs.partition_point(|r| r.p < p_min);
                let list = &self.walk_close[q as usize];
                walk.close_split = self.separable(o, Node::new(q, 0));
                for run in runs[start..].iter().take_while(|r| r.p < a) {
                    walk.run(q, run, list, &mut best, (opens, closes));
                }
            }
        });
        best
    }

    fn fill_runs(&mut self, q: u32) {
        let list = &self.walk_close[q as usize];
        let mut runs = Vec::new();
        let mut start = 0;
        for chunk in list.chunk_by(|x, y| (x.pn.pos, x.to) == (y.pn.pos, y.to)) {
            let mut best = [INF; K];
            let mut finite = false;
            for l in 0..K {
                let lane = chunk.iter().filter(|x| x.out[l].is_finite());
                finite |= lane.clone().next().is_some();
                best[l] = lane
                    .map(|x| x.out[l] + x.stat[l] + x.ao_best[l])
                    .fold(INF, f64::min);
            }
            runs.push(Run {
                p: chunk[0].pn.pos,
                to: chunk[0].to,
                start: start as u32,
                end: (start + chunk.len()) as u32,
                finite,
                best,
            });
            start += chunk.len();
        }
        self.runs[q as usize] = runs;
    }

    fn corner_min(
        &self,
        outer: BasePair,
        inner: BasePair,
        (lefts, n1): (&[Corner], usize),
        (rights, n2): (&[Corner], usize),
    ) -> Vals<K> {
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
        let mut best = [INF; K];
        let (Some(l0), Some(r0)) = (lefts.first(), rights.first()) else {
            return best;
        };
        if self
            .one_by_n
            .corner_free(&LoopPairs { outer, inner }, n1, n2)
        {
            let energy = self.e(energy_at(l0, r0));
            for lc in lefts {
                for rc in rights {
                    for l in 0..K {
                        relax(&mut best[l], energy[l] - self.kk[l] * (lc.cai + rc.cai));
                    }
                }
            }
            return best;
        }
        for lc in lefts {
            for rc in rights {
                let energy = f64::from(energy_at(lc, rc).0);
                for l in 0..K {
                    relax(
                        &mut best[l],
                        self.lambda[l] * energy - self.kk[l] * (lc.cai + rc.cai),
                    );
                }
            }
        }
        best
    }

    fn separable(&self, from: Node, to: Node) -> bool {
        to.pos >= from.pos + 2
            && self.dfa.next_codon_boundary(from.pos + 1) <= self.dfa.codon_start(to.pos - 1)
            && self.halves_combine(to)
    }

    fn outer_ao_best(&self, pair: BasePair, from: Node, to: Node) -> Vals<K> {
        let mut best = [INF; K];
        if to.pos == 0 {
            return best;
        }
        let table = &self.params.mismatch_interior;
        for e5 in self.dfa.right(from) {
            let Some(head) = self.dfa.span_head_cai(e5.to) else {
                continue;
            };
            let l5 = e5.cai + head;
            for e3 in self.dfa.left(to) {
                let Some(tail) = self.dfa.span_tail_cai(e3.to) else {
                    continue;
                };
                let mm = f64::from(table.get(pair, e5.nuc.to_nuc5(), e3.nuc.to_nuc5()));
                for l in 0..K {
                    best[l] =
                        best[l].min(self.lambda[l] * mm - self.kk[l] * (l5 + (tail + e3.cai)));
                }
            }
        }
        best
    }

    /// A separable strand's first-nucleotide choices out of `from`: the
    /// nucleotide and its adaptation to the boundary after it.
    fn firsts(&self, from: Node) -> Vec<(Nucleotide, f64)> {
        self.dfa
            .right(from)
            .iter()
            .filter_map(|e| Some((e.nuc, e.cai + self.dfa.span_head_cai(e.to)?)))
            .collect()
    }

    /// A separable strand's last-nucleotide choices into `to`.
    fn lasts(&self, to: Node) -> Vec<(Nucleotide, f64)> {
        if to.pos == 0 {
            return Vec::new();
        }
        self.dfa
            .left(to)
            .iter()
            .filter_map(|e| Some((e.nuc, self.dfa.span_tail_cai(e.to)? + e.cai)))
            .collect()
    }

    /// The least inner half of every split loop around `(o, key)`, per lane.
    fn inner_best(&self, o: Node, key: PairKey, inner_pair: BasePair) -> Vals<K> {
        let mut best = [INF; K];
        if key.open.pos == 0 {
            return best;
        }
        let firsts = self.firsts(o);
        let lasts = self.lasts(key.open);
        let table = &self.params.mismatch_interior;
        let reversed = inner_pair.reversed();
        for &(r5, r5_cai) in &firsts {
            for &(l3, l3_cai) in &lasts {
                let mm = f64::from(table.get(reversed, r5.to_nuc5(), l3.to_nuc5()));
                for l in 0..K {
                    best[l] = best[l].min(self.lambda[l] * mm - self.kk[l] * (r5_cai + l3_cai));
                }
            }
        }
        best
    }

    fn multi_tails(&mut self, pos: u32) {
        let (dfa, t0) = (self.dfa, self.t[0]);
        for &jn in dfa.nodes_at(pos) {
            for (key, _) in t0.multi_states(jn) {
                let Some(constraint) = self.constraint(key.open.pos, key.close.pos) else {
                    continue;
                };
                let Some(&h) = self.out_p[NodeLayout::ix_s::<S>(jn)].get(&key) else {
                    continue;
                };
                let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                let cost = f64::from((Energy(self.params.ml.closing_37) + self.stem(closing)).0);
                let v: Vals<K> =
                    std::array::from_fn(|l| h[l] + self.lambda[l] * cost + constraint[l]);
                let slot = self.out_multi[NodeLayout::ix_s::<S>(jn)]
                    .entry(key)
                    .or_insert([INF; K]);
                relax_all(slot, &v);
            }
        }
    }

    fn corners_into(&self, from: Node, to: Node, out: &mut Vec<Corner>) {
        out.clear();
        if to.pos - from.pos == 1 {
            for e in self.dfa.right(from).iter().filter(|e| e.to == to) {
                out.push(Corner {
                    first: e.nuc,
                    last: e.nuc,
                    cai: e.cai,
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
                });
            }
        }
    }
}

/// The walk of [`MultiScan::singles_group`].
struct GroupWalk<'s, 'a, const K: usize, const S: usize> {
    scan: &'s MultiScan<'a, K, S>,
    o: Node,
    open_node: Node,
    /// Each key's pair and the best inner half of a split loop around it.
    inner: Vec<(BasePair, Vals<K>)>,
    all_inner: bool,
    corners: Vec<OuterCorner<K>>,
    close_split: bool,
}

/// One corner pair of a generic loop with the enclosing pair's part of its
/// score: the energy model's integer, and the corners' weighted CAI per lane.
#[derive(Clone, Copy)]
struct OuterCorner<const K: usize> {
    energy: i32,
    cai: Vals<K>,
    r5: Nuc5,
    l3: Nuc5,
}

impl<'s, 'a, const K: usize, const S: usize> GroupWalk<'s, 'a, K, S> {
    fn new(scan: &'s MultiScan<'a, K, S>, o: Node, keys: &'s [PairKey]) -> Self {
        let inner: Vec<(BasePair, Vals<K>)> = keys
            .iter()
            .map(|&key| {
                let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                (pair, scan.inner_best(o, key, pair))
            })
            .collect();
        Self {
            scan,
            o,
            open_node: keys[0].open,
            all_inner: inner.iter().all(|(_, h)| h[0].is_finite()),
            inner,
            close_split: false,
            corners: Vec::new(),
        }
    }

    fn run(
        &mut self,
        q: u32,
        run: &Run<K>,
        list: &[Walk<K>],
        best: &mut [Vals<K>],
        (opens, closes): (&mut StrandSlots, &mut StrandSlots),
    ) {
        if !run.finite {
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
        let sh = &scan.shape[shape.0 * scan.single_span() + shape.1];
        let half: Vals<K> = std::array::from_fn(|l| run.best[l] + sh[l]);
        for (slot, (_, inner)) in best.iter_mut().zip(&self.inner) {
            for l in 0..K {
                relax(&mut slot[l], half[l] + inner[l]);
            }
        }
    }

    fn splits(&self, to: Node, (n1, n2): (usize, usize)) -> bool {
        self.close_split && is_generic(n1, n2) && self.scan.separable(to, self.open_node)
    }

    fn pair(
        &mut self,
        x: &Walk<K>,
        shape: (usize, usize),
        best: &mut [Vals<K>],
        (opens, closes): (&mut StrandSlots, &mut StrandSlots),
    ) {
        if !x.out.iter().any(|v| v.is_finite()) {
            return;
        }
        let (scan, o, open_node) = (self.scan, self.o, self.open_node);
        let open = opens.get_or_fill(
            StrandSlots::opening(open_node.pos, x.to, self.scan.dfa.stride()),
            |s| {
                s.refill(scan, x.to, open_node);
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

    fn stack_bulge(
        &mut self,
        x: &Walk<K>,
        (n1, n2): (usize, usize),
        cai: f64,
        best: &mut [Vals<K>],
    ) {
        let scan = self.scan;
        let base: Vals<K> = std::array::from_fn(|l| x.out[l] + x.stat[l]);
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
            let energy = f64::from(energy.0);
            for l in 0..K {
                relax(
                    &mut slot[l],
                    base[l] + scan.lambda[l] * energy - scan.kk[l] * cai,
                );
            }
        }
    }

    fn internal(
        &mut self,
        x: &Walk<K>,
        (n1, n2): (usize, usize),
        (open, close): (&mut Strand, &mut Strand),
        best: &mut [Vals<K>],
    ) {
        let (scan, o, open_node) = (self.scan, self.o, self.open_node);
        let base: Vals<K> = std::array::from_fn(|l| x.out[l] + x.stat[l]);
        let generic = is_generic(n1, n2);
        let splits = generic && open.split && close.split;
        let sh = &scan.shape[n1 * scan.single_span() + n2];
        let outer_half: Vals<K> = std::array::from_fn(|l| base[l] + x.ao_best[l] + sh[l]);
        let mut outer_ready = false;
        for (slot, &(inner_pair, inner_best)) in best.iter_mut().zip(&self.inner) {
            if inner_best[0].is_finite() && splits {
                for l in 0..K {
                    relax(&mut slot[l], outer_half[l] + inner_best[l]);
                }
                continue;
            }
            let lefts = open.corners(scan, x.to, open_node);
            let rights = close.corners(scan, o, x.qn);
            let least = if generic {
                if !outer_ready {
                    Self::outer_corners(scan, &mut self.corners, x.pair, (n1, n2), (lefts, rights));
                    outer_ready = true;
                }
                self.generic_corner_min(inner_pair)
            } else {
                scan.corner_min(x.pair, inner_pair, (lefts, n1), (rights, n2))
            };
            for l in 0..K {
                relax(&mut slot[l], base[l] + least[l]);
            }
        }
    }

    fn outer_corners<const S2: usize>(
        scan: &MultiScan<'_, K, S2>,
        corners: &mut Vec<OuterCorner<K>>,
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
                    cai: scan.kk.map(|kk| kk * (lc.cai + rc.cai)),
                    r5: rc.first.to_nuc5(),
                    l3: lc.last.to_nuc5(),
                });
            }
        }
    }

    fn generic_corner_min(&self, inner: BasePair) -> Vals<K> {
        let scan = self.scan;
        let table = &scan.params.mismatch_interior;
        let inner_rev = inner.reversed();
        let mut best = [INF; K];
        for c in &self.corners {
            let energy = f64::from(c.energy + table.get(inner_rev, c.r5, c.l3));
            for l in 0..K {
                relax(&mut best[l], scan.lambda[l] * energy - c.cai[l]);
            }
        }
        best
    }
}

/// One strand of the loops around an enclosed pair: its best fill, whether it
/// splits at a codon boundary, and its corners once asked for.
#[derive(Default)]
struct Strand {
    cai: Option<f64>,
    split: bool,
    corners: Vec<Corner>,
    cornered: bool,
}

impl Strand {
    fn refill<const K: usize, const S: usize>(
        &mut self,
        scan: &MultiScan<'_, K, S>,
        from: Node,
        to: Node,
    ) {
        self.cai = scan.fill(from, to);
        self.split = scan.separable(from, to);
        self.cornered = false;
    }

    fn corners<const K: usize, const S: usize>(
        &mut self,
        scan: &MultiScan<'_, K, S>,
        from: Node,
        to: Node,
    ) -> &[Corner] {
        if !self.cornered {
            scan.corners_into(from, to, &mut self.corners);
            self.cornered = true;
        }
        &self.corners
    }
}

/// Hands each sweep its [`MultiScan::id`].
static PASS_IDS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The opening and closing strands, reused from one enclosed pair to the
    /// next on each worker.
    static STRANDS: RefCell<(StrandSlots, StrandSlots)> = RefCell::default();
}

/// The strands met around one enclosed pair, one slot per strand end.
#[derive(Default)]
struct StrandSlots {
    slots: Vec<Strand>,
    live: Vec<bool>,
    filled: Vec<usize>,
    owner: Option<(u64, Node)>,
}

impl StrandSlots {
    fn opening_len(l: usize, stride: usize) -> usize {
        (l + 1) * stride
    }

    fn closing_len(l: usize, stride: usize) -> usize {
        l * stride
    }

    fn opening(a: u32, from: Node, stride: usize) -> usize {
        (a - from.pos) as usize * stride + usize::from(from.lane)
    }

    fn closing(b: u32, qn: Node, stride: usize) -> usize {
        (qn.pos - b - 1) as usize * stride + usize::from(qn.lane)
    }

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

    fn get_or_fill(&mut self, i: usize, fill: impl FnOnce(&mut Strand)) -> &mut Strand {
        let slot = &mut self.slots[i];
        if !self.live[i] {
            self.live[i] = true;
            self.filled.push(i);
            fill(slot);
        }
        slot
    }
}

// ---- an instrument over public inputs --------------------------------------

/// Exact scans of one molecule at several weights, for timing and checking the
/// one-sweep pass against one pass per weight.
#[doc(hidden)]
pub struct Instance {
    params: Params,
    graph: Dfa,
    region: PairConstraint,
    tables: Vec<Tables>,
    single_max_len: usize,
}

/// One weight's cost to go of every cell.
#[doc(hidden)]
pub struct LaneCosts(CellCosts);

impl Instance {
    /// Exact scans of `protein` between `utr5` and `utr3`, every codon allowed,
    /// at each of `lambdas`. `None` when the lattice admits no design.
    ///
    /// # Errors
    ///
    /// When the instance is not valid.
    pub fn scan(
        protein: &str,
        (utr5, utr3): (&str, &str),
        constraints: RegionConstraints,
        lambdas: &[f64],
        threads: usize,
    ) -> Result<Option<Self>, BeamError> {
        let params = Params::load_default().expect("the default energy model loads");
        let usage = CodonUsage::load_default().expect("the default codon usage loads");
        let config = ScanConfig {
            threads,
            ..ScanConfig::default()
        };
        let allowed = every_codon(protein);
        let Some(&first) = lambdas.first() else {
            return Ok(None);
        };
        let Some((graph, t0, region)) = crate::scan_restricted_with_region(
            &params,
            protein,
            &usage,
            first,
            &allowed,
            utr5,
            utr3,
            constraints,
            config,
        )?
        else {
            return Ok(None);
        };
        let mut tables = vec![t0];
        for &lambda in &lambdas[1..] {
            tables.push(crate::scan::scan_with_region(
                &params,
                &graph,
                lambda,
                config,
                region.clone(),
            ));
        }
        Ok(Some(Self {
            params,
            graph,
            region,
            tables,
            single_max_len: config.single_max_len,
        }))
    }

    /// Exact scans of a built lattice at each of `lambdas`, unconstrained.
    pub fn scan_lattice(graph: Dfa, lambdas: &[f64]) -> Self {
        let params = Params::load_default().expect("the default energy model loads");
        let config = ScanConfig::default();
        let region = PairConstraint::inert();
        let tables = lambdas
            .iter()
            .map(|&l| crate::scan::scan_with_region(&params, &graph, l, config, region.clone()))
            .collect();
        Self {
            params,
            graph,
            region,
            tables,
            single_max_len: config.single_max_len,
        }
    }

    /// [`Self::scan`] with the scans run as one traversal
    /// ([`crate::scan_multi`]) instead of one scan per weight, and the time
    /// spent splitting its lanes into one [`Tables`] each.
    ///
    /// # Errors
    ///
    /// When the instance is not valid.
    ///
    /// # Panics
    ///
    /// When `lambdas` holds more than [`MAX_LANES`] weights.
    pub fn scan_together(
        protein: &str,
        (utr5, utr3): (&str, &str),
        constraints: RegionConstraints,
        lambdas: &[f64],
    ) -> Result<Option<(Self, Duration)>, BeamError> {
        let params = Params::load_default().expect("the default energy model loads");
        let usage = CodonUsage::load_default().expect("the default codon usage loads");
        let config = ScanConfig::default();
        config.validate()?;
        let aa = crate::residues(protein)?;
        let (u5, u3) = (crate::parse_utr(utr5)?, crate::parse_utr(utr3)?);
        let Some(graph) =
            crate::dfa::build_restricted_with_utr(&aa, &usage, &every_codon(protein), &u5, &u3)
        else {
            return Ok(None);
        };
        let region = PairConstraint::new(&constraints, &graph);
        let (tables, split) = scan_lanes(&params, &graph, lambdas, config, &region);
        Ok(Some((
            Self {
                params,
                graph,
                region,
                tables,
                single_max_len: config.single_max_len,
            },
            split,
        )))
    }

    /// The first cell whose `(mfe, cai)` differs in any bit between this
    /// instance's tables and `other`'s, lane by lane, or whose cell one side
    /// lacks; `None` when every lane's tables agree. Back-pointers are not
    /// compared.
    pub fn first_table_difference(&self, other: &Self) -> Option<String> {
        if self.tables.len() != other.tables.len() {
            return Some("lane counts differ".to_string());
        }
        let same = |a: &State, b: &State| a.mfe == b.mfe && a.cai.to_bits() == b.cai.to_bits();
        let width = self.graph.slots();
        for (l, (a, b)) in self.tables.iter().zip(&other.tables).enumerate() {
            if a.lambda().to_bits() != b.lambda().to_bits() {
                return Some(format!("lane {l}: weights differ"));
            }
            for o in 0..width {
                let node = self.graph.node_of(o);
                match (a.c[o], b.c[o]) {
                    (None, None) => {}
                    (Some(x), Some(y)) if same(&x, &y) => {}
                    (x, y) => return Some(format!("lane {l} C[{o}]: {x:?} against {y:?}")),
                }
                for (name, x, y) in [
                    ("M", a.m_states(node), b.m_states(node)),
                    ("M2", a.m2_states(node), b.m2_states(node)),
                ] {
                    let agree = x.len() == y.len()
                        && x.iter()
                            .zip(&y)
                            .all(|(u, v)| u.0 == v.0 && same(&u.1, &v.1));
                    if !agree {
                        return Some(format!("lane {l} {name}[{o}] differs"));
                    }
                }
                for (name, x, y) in [("P", &a.p[o], &b.p[o]), ("Multi", &a.multi[o], &b.multi[o])] {
                    if x.len() != y.len() {
                        return Some(format!(
                            "lane {l} {name}[{o}]: {} cells against {}",
                            x.len(),
                            y.len()
                        ));
                    }
                    for (k, u) in x {
                        match y.get(k) {
                            Some(v) if same(u, v) => {}
                            v => {
                                return Some(format!(
                                    "lane {l} {name}[{o}] {k:?}: {u:?} against {v:?}"
                                ))
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Cells in every lane's tables.
    pub fn table_cells(&self) -> usize {
        self.tables
            .iter()
            .map(|t| {
                let c = t.cell_counts();
                c.total()
            })
            .sum()
    }

    /// Positions of the lattice.
    pub fn seq_len(&self) -> u32 {
        self.graph.seq_len
    }

    /// Every scan's weight.
    pub fn lambdas(&self) -> Vec<f64> {
        self.tables.iter().map(Tables::lambda).collect()
    }

    /// Whether every scan kept every candidate.
    pub fn all_exact(&self) -> bool {
        self.tables.iter().all(Tables::is_exact)
    }

    /// Whether every scan holds the first scan's cells ([`same_cells`]).
    ///
    /// # Errors
    ///
    /// The first cell that differs.
    pub fn same_cells(&self) -> Result<(), String> {
        let tables: Vec<&Tables> = self.tables.iter().collect();
        same_cells(&self.graph, &tables)
    }

    /// One [`OutsideScan::cost_to_go`] per scan.
    pub fn separate(&self, threads: usize) -> Vec<LaneCosts> {
        self.tables
            .iter()
            .map(|t| {
                LaneCosts(OutsideScan::cost_to_go(
                    &self.params,
                    &self.graph,
                    t,
                    &self.region,
                    self.single_max_len,
                    threads,
                    false,
                ))
            })
            .collect()
    }

    /// The cost to go of every scan in one sweep, with the time spent
    /// splitting its lanes apart afterwards.
    pub fn fused(&self, threads: usize) -> (Vec<LaneCosts>, Duration) {
        let tables: Vec<&Tables> = self.tables.iter().collect();
        let (lanes, split) = cost_to_go_lanes(
            &self.params,
            &self.graph,
            &tables,
            &self.region,
            self.single_max_len,
            threads,
        );
        (lanes.into_iter().map(LaneCosts).collect(), split)
    }
}

impl LaneCosts {
    /// Cells with a finite cost to go.
    pub fn finite_cells(&self) -> usize {
        let c = &self.0;
        let dense = |t: &[Vec<f64>]| t.iter().flatten().filter(|v| v.is_finite()).count();
        let sparse = |t: &[FxHashMap<PairKey, f64>]| {
            t.iter()
                .flat_map(|m| m.values())
                .filter(|v| v.is_finite())
                .count()
        };
        c.c.iter().filter(|v| v.is_finite()).count()
            + dense(&c.m)
            + dense(&c.m2)
            + sparse(&c.p)
            + sparse(&c.multi)
    }

    /// Cells compared by [`Self::first_difference`].
    pub fn cells(&self) -> usize {
        let c = &self.0;
        c.c.len()
            + c.m.iter().map(Vec::len).sum::<usize>()
            + c.m2.iter().map(Vec::len).sum::<usize>()
            + c.p.iter().map(|m| m.len()).sum::<usize>()
            + c.multi.iter().map(|m| m.len()).sum::<usize>()
    }

    /// The first cell whose value differs from `other`'s in any bit, or whose
    /// key one side lacks; `None` when every cell agrees bit for bit.
    pub fn first_difference(&self, other: &Self) -> Option<String> {
        let (a, b) = (&self.0, &other.0);
        let flat = |name: &str, x: &[f64], y: &[f64]| -> Option<String> {
            if x.len() != y.len() {
                return Some(format!("{name}: {} cells against {}", x.len(), y.len()));
            }
            x.iter()
                .zip(y)
                .position(|(u, v)| u.to_bits() != v.to_bits())
                .map(|i| format!("{name}[{i}]: {} against {}", x[i], y[i]))
        };
        let dense = |name: &str, x: &[Vec<f64>], y: &[Vec<f64>]| -> Option<String> {
            x.iter()
                .zip(y)
                .enumerate()
                .find_map(|(o, (u, v))| flat(&format!("{name}[{o}]"), u, v))
        };
        let sparse = |name: &str,
                      x: &[FxHashMap<PairKey, f64>],
                      y: &[FxHashMap<PairKey, f64>]|
         -> Option<String> {
            x.iter().zip(y).enumerate().find_map(|(o, (u, v))| {
                if u.len() != v.len() {
                    return Some(format!("{name}[{o}]: {} keys against {}", u.len(), v.len()));
                }
                u.iter().find_map(|(k, x)| match v.get(k) {
                    None => Some(format!("{name}[{o}] {k:?}: missing")),
                    Some(y) if x.to_bits() != y.to_bits() => {
                        Some(format!("{name}[{o}] {k:?}: {x} against {y}"))
                    }
                    Some(_) => None,
                })
            })
        };
        flat("C", &a.c, &b.c)
            .or_else(|| dense("M", &a.m, &b.m))
            .or_else(|| dense("M2", &a.m2, &b.m2))
            .or_else(|| (a.m.len() != b.m.len()).then(|| "M columns".to_string()))
            .or_else(|| sparse("P", &a.p, &b.p))
            .or_else(|| sparse("Multi", &a.multi, &b.multi))
    }
}

/// [`crate::scan_multi::scan_multi`] over any number of weights up to
/// [`MAX_LANES`], with the time spent splitting the lanes into [`Tables`].
pub(crate) fn scan_lanes(
    params: &Params,
    dfa: &Dfa,
    lambdas: &[f64],
    config: ScanConfig,
    region: &PairConstraint,
) -> (Vec<Tables>, Duration) {
    fn at<const K: usize>(
        params: &Params,
        dfa: &Dfa,
        lambdas: &[f64],
        config: ScanConfig,
        region: &PairConstraint,
    ) -> (Vec<Tables>, Duration) {
        let lambdas: [f64; K] = lambdas.try_into().expect("K weights");
        let lanes = crate::scan_multi::scan_multi(params, dfa, lambdas, config, region.clone());
        let start = Instant::now();
        let tables = lanes.into_lanes(dfa).into();
        (tables, start.elapsed())
    }
    let args = (params, dfa, lambdas, config, region);
    match lambdas.len() {
        1 => at::<1>(args.0, args.1, args.2, args.3, args.4),
        2 => at::<2>(args.0, args.1, args.2, args.3, args.4),
        3 => at::<3>(args.0, args.1, args.2, args.3, args.4),
        4 => at::<4>(args.0, args.1, args.2, args.3, args.4),
        5 => at::<5>(args.0, args.1, args.2, args.3, args.4),
        6 => at::<6>(args.0, args.1, args.2, args.3, args.4),
        7 => at::<7>(args.0, args.1, args.2, args.3, args.4),
        8 => at::<8>(args.0, args.1, args.2, args.3, args.4),
        k => panic!("{k} lanes: one traversal takes 1 to {MAX_LANES}"),
    }
}

/// Every codon of each residue, a designed stop as the stop family.
fn every_codon(protein: &str) -> Vec<Vec<[u8; 3]>> {
    use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS, STOP, STOP_CODE};
    protein
        .chars()
        .map(|c| {
            let a = match c {
                STOP_CODE => STOP,
                c => aa_to_index(c).unwrap_or(0),
            };
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}
