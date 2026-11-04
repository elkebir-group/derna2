//! Exact max-marginals of the constrained design objective, by inside–outside
//! over the codon lattice.
//!
//! A region of the bounded search is a codon lattice. Excluding one codon at one
//! position shrinks it to a subset whose optimum is a floor under every child
//! region inside that subset. This module computes that optimum for **every**
//! `(position, codon)` at once, from one inside and one outside pass, rather than
//! one constrained fold per exclusion.
//!
//! # What is computed
//!
//! The grammar is the scan's ([`crate::scan`]) in its unreduced internal-loop
//! closure: every rule the scan applies, enumerated directly, over the same DFA,
//! the same loop energies and the same region constraint. Each rule instance is a
//! hyperedge — a head cell, at most two tail cells, and a local weight — and the
//! instances are recorded in the order a forward sweep generates them, which is a
//! topological order. The three passes are then:
//!
//! 1. **inside** — the best value of each cell, as the scan computes it;
//! 2. **outside** — the best value of everything *around* each cell, pushed from
//!    heads to tails in reverse generation order;
//! 3. **edge marginals** — for every lattice edge, the best objective of any
//!    design whose path uses it.
//!
//! A design avoids codon `c` at position `q` exactly when its path takes an edge
//! of `q`'s sub-lattice off `c`'s path, so the exclusion's optimum is the least
//! edge marginal over those edges. That is an equality, not a bound.
//!
//! # The second channel
//!
//! Not every edge on a design's path is consumed by a table cell. A loop's
//! energy-irrelevant interior is filled by the best-adaptation path between its
//! boundaries, and hairpins and internal loops choose their terminal-mismatch
//! nucleotides inside a single rule. Each rule instance is therefore emitted once
//! per local choice, with the choice's edges and fills as [`Seg`]ments. An edge
//! inside a fill `a → b` takes the fill's best path *through* it,
//! `fill(a, e.from) + e + fill(e.to, b)`, which is exact because a path through
//! `e` decomposes there.
//!
//! # Scope
//!
//! This is the reference: it materializes the hypergraph and is sized for small
//! instances, where it is checked for equality against one constrained fold per
//! exclusion. The fixed-flank lattice without a must-pair set is the only shape
//! it serves.

use rna_mfe::loops::{
    bulge_energy, hairpin_energy, hairpin_energy_scalar, internal_energy_scalar,
    pair_from_nucleotides, stack_energy, BulgeLoop, HairpinLoop, InternalLoopScalar, StackPair,
};
use rna_mfe::params::BasePair;
use rna_mfe::{Energy, Nucleotide, Params};

use crate::cost_to_go::CellRef;
use crate::dfa::{self, Dfa, Edge, Node};
use crate::hash::FxHashMap;
use crate::must_pair::MustPairConstraint;
use crate::region::{PairConstraint, RegionConstraints, Verdict};
use crate::state::PairKey;
use crate::{parse_utr, BeamError, ScanConfig};
use derna_codon_dp::CodonUsage;

pub mod lazy;

/// Smallest number of unpaired bases inside a hairpin.
const MIN_HAIRPIN_LOOP: u32 = 3;

/// One cell of the scan's tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Cell {
    /// An exterior prefix ending at the node.
    C(Node),
    /// A multibranch component with at least one branch, `(outer, inner)`.
    M(Node, Node),
    /// A multibranch component with at least two branches, `(outer, inner)`.
    M2(Node, Node),
    /// A closed pair, at its outer node.
    P(Node, PairKey),
    /// A multi-loop awaiting its closing cost, at its outer node.
    Multi(Node, PairKey),
}

/// A piece of the lattice path one local choice of a rule spells.
#[derive(Debug, Clone, Copy)]
enum Seg {
    /// One edge out of a node, spelled explicitly.
    Edge(Node, Edge),
    /// The best-adaptation path between two nodes.
    Fill(Node, Node),
}

/// One rule instance with one local choice: `head ← tails + weight`, and the
/// path pieces the choice spells.
struct Inst {
    head: Cell,
    tails: [Option<Cell>; 2],
    weight: f64,
    segs: Vec<Seg>,
}

/// The rule generator: the scan's grammar over one lattice.
struct Grammar<'a> {
    params: &'a Params,
    dfa: &'a Dfa,
    lambda: f64,
    /// `(1 − λ)·100`, the adaptation term's weight on the dekacalorie scale.
    kk: f64,
    constraint: PairConstraint,
    /// Positions every derivation must pair: a rule never declares one
    /// unpaired ([`MustPairConstraint`]). Inert unless a caller asks for it.
    must: MustPairConstraint,
    single_max_len: usize,
}

/// Where the rules hand the instances they generate.
enum Sink<'s> {
    /// The forward sweep: relax each head's inside value, and keep the
    /// instances when they are wanted.
    Sweep { insts: Option<&'s mut Vec<Inst>> },
    /// The instances with one head, generated after every inside value is
    /// final, which the sweep's reads then see unchanged.
    Pull { head: Cell, out: &'s mut Vec<Inst> },
}

/// One enclosing pair a single-loop or multi-loop rule may close: its opening
/// node and edge, and its closing node and edge. Restricting a rule to it
/// generates exactly the instances whose head is that pair.
#[derive(Clone, Copy)]
struct Outer {
    pn: Node,
    ep: Edge,
    qn: Node,
    eq: Edge,
}

/// Which of a closed pair's continuations to generate.
#[derive(Clone, Copy)]
struct Parts {
    exterior: bool,
    stem: bool,
    m2: bool,
    singles: bool,
}

impl Parts {
    const ALL: Self = Self {
        exterior: true,
        stem: true,
        m2: true,
        singles: true,
    };
}

/// Every design's best objective through each lattice edge, with the inside
/// optimum it is measured against.
pub struct EdgeMarginals {
    optimum: f64,
    by_edge: FxHashMap<(Node, u8), f64>,
    /// The first CDS position.
    cds_start: u32,
    /// Added to a lattice value to report it on a design's own scale.
    report_offset: f64,
    dfa: Dfa,
    /// Seconds the inside scan took, and the outside pass after it.
    seconds: (f64, f64),
    /// Every closed pair's outside value, finite ones only.
    pair_outside: Vec<(Node, PairKey, f64)>,
}

impl EdgeMarginals {
    /// The optimum over the whole lattice, on a design's reported scale:
    /// `λ·MFE − (1−λ)·100·log CAI`, energies in dekacalories.
    pub fn optimum(&self) -> f64 {
        self.optimum + self.report_offset
    }

    /// Seconds spent on the inside scan and on the outside pass after it.
    pub fn seconds(&self) -> (f64, f64) {
        self.seconds
    }

    /// Every closed pair with a finite outside value — the best objective of
    /// any design whose structure contains that pair, less the pair's inside —
    /// sorted, on the lattice's own scale. Every closed pair, not only those on
    /// good designs, so two passes that agree here agree on the whole grammar
    /// around the pairs rather than on the few derivations a marginal reads.
    pub fn pair_outsides(&self) -> &[(Node, PairKey, f64)] {
        &self.pair_outside
    }

    /// Every coding-sequence edge's marginal — the best objective of any design
    /// whose path uses it — sorted, on a design's reported scale. Every edge,
    /// not only the least off a codon's path, so two passes that agree here
    /// agree on each channel that hands an edge its value.
    pub fn cds_edges(&self) -> Vec<(Node, u8, f64)> {
        let end = self.dfa.cds_end();
        let mut v: Vec<_> = self
            .by_edge
            .iter()
            .filter(|((n, _), x)| n.pos >= self.cds_start && n.pos < end && x.is_finite())
            .map(|(&(n, nuc), &x)| (n, nuc, x + self.report_offset))
            .collect();
        v.sort_by_key(|&(n, nuc, _)| (self.dfa.ix(n), nuc));
        v
    }

    /// The best objective of any design that does **not** spell `codon` at
    /// codon position `q`, on the same scale as [`Self::optimum`]; `None` when
    /// every design spells it.
    ///
    /// `codon` must be one this lattice admits at `q`.
    pub fn excluding(&self, q: usize, codon: [u8; 3]) -> Option<f64> {
        let base = self.cds_start + 3 * q as u32;
        let mut on_path = Vec::with_capacity(3);
        let mut at = Node::new(base, 0);
        let path = self
            .dfa
            .spell_to(at, &codon, None)
            .expect("the codon is admitted at this position");
        for (e, &nuc) in path.iter().zip(&codon) {
            on_path.push((at, nuc));
            at = e.to;
        }
        let mut best = f64::INFINITY;
        for pos in base..base + 3 {
            for &n in self.dfa.nodes_at(pos) {
                for e in self.dfa.right(n) {
                    let id = (n, e.nuc as u8);
                    if on_path.contains(&id) {
                        continue;
                    }
                    if let Some(&v) = self.by_edge.get(&id) {
                        best = best.min(v);
                    }
                }
            }
        }
        best.is_finite().then_some(best + self.report_offset)
    }
}

/// Every `(position, codon)` exclusion's optimum over the lattice `allowed`
/// spells, by one inside–outside pass. `None` when the lattice admits nothing.
///
/// Serves the fixed-flank shape of [`crate::design_restricted_with_region`] and
/// nothing wider. Only `config.single_max_len` is read: the pass always runs the
/// exact lattice, whatever beam `config` names, since a pruned inside pass is
/// not a minimum.
#[allow(clippy::too_many_arguments)]
pub fn edge_marginals(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<EdgeMarginals>, BeamError> {
    config.validate()?;
    let aa = crate::residues(protein)?;
    let (utr5, utr3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &utr5, &utr3) else {
        return Ok(None);
    };
    let started = std::time::Instant::now();
    let grammar = Grammar {
        params,
        dfa: &graph,
        lambda,
        kk: (1.0 - lambda) * 100.0,
        constraint: PairConstraint::new(&constraints, &graph),
        must: MustPairConstraint::inert(),
        single_max_len: config.single_max_len,
    };
    let (insts, inside) = grammar.inside();
    let Some(&optimum) = inside.get(&Cell::C(graph.end())) else {
        return Ok(None);
    };
    let inside_s = started.elapsed().as_secs_f64();
    let outside = outside(&insts, &inside, graph.end());
    let by_edge = grammar.edge_marginals(&insts, &inside, &outside);
    let pair_outside = sorted_pairs(outside.iter().filter_map(|(c, &v)| match *c {
        Cell::P(o, key) => Some((o, key, v)),
        _ => None,
    }));
    Ok(Some(EdgeMarginals {
        optimum,
        by_edge,
        cds_start: graph.cds_start(),
        report_offset: -grammar.kk * graph.codon_offset(),
        dfa: graph,
        seconds: (inside_s, started.elapsed().as_secs_f64() - inside_s),
        pair_outside,
    }))
}

/// [`edge_marginals`] computed from the production scan's own tables, without
/// recording the rule instances: one restricted scan, then one backward pass
/// over what it filled ([`crate::outside_scan`]). The same quantity, at the
/// scale the bounded search runs at.
#[allow(clippy::too_many_arguments)]
pub fn edge_marginals_fast(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<EdgeMarginals>, BeamError> {
    let started = std::time::Instant::now();
    let Some((graph, tables, constraint)) = crate::scan_restricted_with_region(
        params,
        protein,
        usage,
        lambda,
        allowed,
        utr5,
        utr3,
        constraints,
        config,
    )?
    else {
        return Ok(None);
    };
    let Some(best) = tables.c[graph.ix(graph.end())] else {
        return Ok(None);
    };
    let scan_s = started.elapsed().as_secs_f64();
    let marginals = crate::outside_scan::OutsideScan::new(
        params,
        &graph,
        &tables,
        &constraint,
        config.single_max_len,
    )
    .run();
    let pair_outside = sorted_pairs(marginals.pair_outside.iter().enumerate().flat_map(
        |(oi, cells)| {
            let o = graph.node_of(oi);
            cells.iter().map(move |(k, &v)| (o, *k, v))
        },
    ));
    let mut by_edge: FxHashMap<(Node, u8), f64> = FxHashMap::default();
    for (i, &v) in marginals.by_edge.iter().enumerate() {
        if v.is_finite() {
            let node = graph.node_of(i / 4);
            by_edge.insert((node, (i % 4) as u8), v);
        }
    }
    let kk = (1.0 - lambda) * 100.0;
    Ok(Some(EdgeMarginals {
        optimum: best.brd(lambda),
        by_edge,
        cds_start: graph.cds_start(),
        report_offset: -kk * graph.codon_offset(),
        dfa: graph,
        seconds: (scan_s, started.elapsed().as_secs_f64() - scan_s),
        pair_outside,
    }))
}

/// The finite entries of a pair-outside list, in a fixed order so two passes'
/// lists compare position by position.
fn sorted_pairs(it: impl Iterator<Item = (Node, PairKey, f64)>) -> Vec<(Node, PairKey, f64)> {
    let mut v: Vec<_> = it.filter(|x| x.2.is_finite()).collect();
    v.sort_by_key(|&(o, k, _)| (o, k.open, k.close, k.nuc_open() as u8, k.nuc_close as u8));
    v
}

/// Every cell's inside and outside value over `dfa`, by the recorded hypergraph:
/// the reference the production cost-to-go pass
/// ([`crate::cost_to_go::Outside`]) is checked against, cell by cell.
///
/// The multi-loop's 3′ flank is enumerated here per core and per closing
/// frontier, never through the shared opening pool, so the check is independent
/// of the pool mirror the production pass relies on. Sized for small lattices
/// only. An unreachable cell's outside is infinite.
pub fn reference_cell_values(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Vec<(CellRef, f64, f64)> {
    let grammar = Grammar {
        params,
        dfa,
        lambda,
        kk: (1.0 - lambda) * 100.0,
        constraint: PairConstraint::new(&constraints, dfa),
        must: MustPairConstraint::inert(),
        single_max_len: config.single_max_len,
    };
    let (insts, inside) = grammar.inside();
    let outside = outside(&insts, &inside, dfa.end());
    inside
        .iter()
        .map(|(cell, &v)| {
            let out = outside.get(cell).copied().unwrap_or(f64::INFINITY);
            (cell.public(), v, out)
        })
        .collect()
}

impl Cell {
    fn public(self) -> CellRef {
        match self {
            Cell::C(n) => CellRef::C(n),
            Cell::M(outer, inner) => CellRef::M(outer, inner),
            Cell::M2(outer, inner) => CellRef::M2(outer, inner),
            Cell::P(outer, key) => CellRef::P(outer, key),
            Cell::Multi(outer, key) => CellRef::Multi(outer, key),
        }
    }
}

/// The best value of everything around each cell: `out(end) = 0`, and each
/// instance hands its head's outside, its weight and its other tails' insides to
/// each of its tails. Reverse generation order is a valid order, because an
/// instance is only generated once all of its tails are final, and so after
/// every instance that writes one of them.
fn outside(insts: &[Inst], inside: &FxHashMap<Cell, f64>, end: Node) -> FxHashMap<Cell, f64> {
    let mut out: FxHashMap<Cell, f64> = FxHashMap::default();
    out.insert(Cell::C(end), 0.0);
    for inst in insts.iter().rev() {
        let Some(&head) = out.get(&inst.head) else {
            continue;
        };
        let tails: Vec<Cell> = inst.tails.iter().flatten().copied().collect();
        for (i, &tail) in tails.iter().enumerate() {
            let others: f64 = tails
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, t)| inside[t])
                .sum();
            let v = head + inst.weight + others;
            let slot = out.entry(tail).or_insert(f64::INFINITY);
            if v < *slot {
                *slot = v;
            }
        }
    }
    out
}

impl Grammar<'_> {
    /// The best-adaptation path weight from `a` to `b`, or `None` when no path
    /// connects them.
    fn fill(&self, a: Node, b: Node) -> Option<f64> {
        self.dfa.broken_codon_cai(a, b)
    }

    /// Every `(node, right edge)` at positions `from..to`, in lattice order.
    fn edges_from(&self, from: u32, to: u32) -> Vec<(Node, Edge)> {
        let mut out = Vec::new();
        for p in from..to {
            for &n in self.dfa.nodes_at(p) {
                for e in self.dfa.right(n) {
                    out.push((n, *e));
                }
            }
        }
        out
    }

    /// The right edge that the left edge `last` of `to` corresponds to: the
    /// edge out of `last.to` spelling `last.nuc`.
    fn right_of_left(&self, last: &Edge, to: Node) -> Edge {
        *self
            .dfa
            .right(last.to)
            .iter()
            .find(|e| e.nuc == last.nuc && e.to == to)
            .expect("a left edge is some node's right edge")
    }

    /// The region constraint's adjustment to a pair written at `(i, j)`: `None` when
    /// the pair is forbidden, otherwise the weight it adds.
    fn constraint(&self, i: u32, j: u32) -> Option<f64> {
        match self.constraint.verdict(i, j) {
            Verdict::Allow => Some(0.0),
            Verdict::Forbid => None,
            Verdict::Penalize(pen) => Some(self.lambda * f64::from(pen.0)),
        }
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

    /// A strand's terminal-mismatch choices from `from` to `to` (length `≥ 1`):
    /// each its two terminal nucleotides, its adaptation and its path pieces.
    fn corners(&self, from: Node, to: Node) -> Vec<(Nucleotide, Nucleotide, f64, Vec<Seg>)> {
        let mut out = Vec::new();
        if to.pos - from.pos == 1 {
            for e in self.dfa.right(from).iter().filter(|e| e.to == to) {
                out.push((e.nuc, e.nuc, e.cai, vec![Seg::Edge(from, *e)]));
            }
            return out;
        }
        for first in self.dfa.right(from) {
            for last in self.dfa.left(to) {
                let Some(mid) = self.fill(first.to, last.to) else {
                    continue;
                };
                let last_edge = self.right_of_left(last, to);
                out.push((
                    first.nuc,
                    last.nuc,
                    first.cai + mid + last.cai,
                    vec![
                        Seg::Edge(from, *first),
                        Seg::Fill(first.to, last.to),
                        Seg::Edge(last.to, last_edge),
                    ],
                ));
            }
        }
        out
    }

    /// Hand `inst` to the sink: relax its head's inside value and keep it on the
    /// sweep, or keep it when it has the pulled head.
    fn emit(&self, sink: &mut Sink<'_>, inside: &mut FxHashMap<Cell, f64>, inst: Inst) {
        match sink {
            Sink::Sweep { insts } => {
                let v = inst.weight + inst.tails.iter().flatten().map(|t| inside[t]).sum::<f64>();
                let slot = inside.entry(inst.head).or_insert(f64::INFINITY);
                if v < *slot {
                    *slot = v;
                }
                if let Some(insts) = insts {
                    insts.push(inst);
                }
            }
            Sink::Pull { head, out } => {
                if inst.head == *head {
                    out.push(inst);
                }
            }
        }
    }

    /// The forward sweep: every rule instance, in an order where each instance
    /// follows every instance writing one of its tails, and the inside values.
    fn inside(&self) -> (Vec<Inst>, FxHashMap<Cell, f64>) {
        self.sweep(true)
    }

    /// [`Self::inside`], keeping the instances only when `record` is set.
    fn sweep(&self, record: bool) -> (Vec<Inst>, FxHashMap<Cell, f64>) {
        let mut insts = Vec::new();
        let mut inside: FxHashMap<Cell, f64> = FxHashMap::default();
        inside.insert(Cell::C(self.dfa.start()), 0.0);
        let mut sink = Sink::Sweep {
            insts: record.then_some(&mut insts),
        };
        self.hairpins(&mut sink, &mut inside);
        for pos in 0..=self.dfa.seq_len {
            let at = |inside: &FxHashMap<Cell, f64>, pick: fn(&Cell) -> Option<Node>| {
                let mut cells: Vec<Cell> = inside
                    .keys()
                    .filter(|c| pick(c).is_some_and(|n| n.pos == pos))
                    .copied()
                    .collect();
                cells.sort_by_key(|c| format!("{c:?}"));
                cells
            };
            for cell in at(&inside, |c| match c {
                Cell::Multi(o, _) => Some(*o),
                _ => None,
            }) {
                self.multi_close(&mut sink, &mut inside, cell);
            }
            for cell in at(&inside, |c| match c {
                Cell::P(o, _) => Some(*o),
                _ => None,
            }) {
                self.close_pair(&mut sink, &mut inside, cell, Parts::ALL, None);
            }
            for cell in at(&inside, |c| match c {
                Cell::M2(o, _) => Some(*o),
                _ => None,
            }) {
                self.close_m2(&mut sink, &mut inside, cell, true, None);
            }
            if pos < self.dfa.seq_len {
                for cell in at(&inside, |c| match c {
                    Cell::M(o, _) => Some(*o),
                    _ => None,
                }) {
                    self.extend(&mut sink, &mut inside, cell);
                }
                for cell in at(&inside, |c| match c {
                    Cell::C(o) => Some(*o),
                    _ => None,
                }) {
                    self.extend(&mut sink, &mut inside, cell);
                }
            }
        }
        (insts, inside)
    }

    /// Every hairpin: an opening edge, an interior of at least three, a closing
    /// edge. The interior's terminal mismatches — or, at sizes 3/4/6, its whole
    /// sequence — are local choices.
    fn hairpins(&self, sink: &mut Sink<'_>, inside: &mut FxHashMap<Cell, f64>) {
        for a in 0..self.dfa.seq_len {
            for &an in self.dfa.nodes_at(a) {
                for eo in self.dfa.right(an) {
                    for b in (a + MIN_HAIRPIN_LOOP + 1)..self.dfa.seq_len {
                        for &bn in self.dfa.nodes_at(b) {
                            for ec in self.dfa.right(bn) {
                                self.hairpin(sink, inside, an, *eo, bn, *ec);
                            }
                        }
                    }
                }
            }
        }
    }

    fn hairpin(
        &self,
        sink: &mut Sink<'_>,
        inside: &mut FxHashMap<Cell, f64>,
        an: Node,
        eo: Edge,
        bn: Node,
        ec: Edge,
    ) {
        let closing = pair_from_nucleotides(eo.nuc, ec.nuc);
        if closing == BasePair::NN || !self.must.free_span(an.pos + 1, bn.pos) {
            return;
        }
        let Some(constraint) = self.constraint(an.pos, bn.pos) else {
            return;
        };
        let key = PairKey {
            open: an,
            close: bn,
            opening: eo.opening,
            nuc_close: ec.nuc,
        };
        let size = (bn.pos - eo.to.pos) as usize;
        let mut choices: Vec<(Energy, f64, Vec<Seg>)> = Vec::new();
        if matches!(size, 3 | 4 | 6) {
            for (cai, seq) in self.dfa.paths_between(eo.to, bn) {
                let energy = hairpin_energy(
                    self.params,
                    &HairpinLoop {
                        closing,
                        loop_seq: &seq,
                    },
                );
                let mut segs = Vec::with_capacity(seq.len());
                let nucs: Vec<u8> = seq.iter().map(|&n| n as u8).collect();
                let mut at = eo.to;
                for e in self
                    .dfa
                    .spell_to(at, &nucs, Some(bn))
                    .expect("a spelled path follows the lattice")
                {
                    segs.push(Seg::Edge(at, e));
                    at = e.to;
                }
                choices.push((energy, cai, segs));
            }
        } else {
            for (first, last, cai, segs) in self.corners(eo.to, bn) {
                let energy = hairpin_energy_scalar(self.params, closing, size, first, last);
                choices.push((energy, cai, segs));
            }
        }
        for (energy, cai, mut segs) in choices {
            segs.push(Seg::Edge(an, eo));
            segs.push(Seg::Edge(bn, ec));
            self.emit(
                sink,
                inside,
                Inst {
                    head: Cell::P(ec.to, key),
                    tails: [None, None],
                    weight: self.e(energy) - self.kk * (eo.cai + cai + ec.cai) + constraint,
                    segs,
                },
            );
        }
    }

    /// `P = Multi + closing cost`.
    fn multi_close(&self, sink: &mut Sink<'_>, inside: &mut FxHashMap<Cell, f64>, cell: Cell) {
        let Cell::Multi(o, key) = cell else {
            unreachable!()
        };
        let Some(constraint) = self.constraint(key.open.pos, key.close.pos) else {
            return;
        };
        let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let cost = Energy(self.params.ml.closing_37) + self.stem(closing);
        self.emit(
            sink,
            inside,
            Inst {
                head: Cell::P(o, key),
                tails: [Some(cell), None],
                weight: self.e(cost) + constraint,
                segs: Vec::new(),
            },
        );
    }

    /// A finished closed pair: onto the exterior, into a multibranch, and grown
    /// into every enclosing stack, bulge and internal loop — the continuations
    /// `parts` names. `lefts`, when given, stands in for the multibranch
    /// components the pair can extend, which are otherwise found by a scan.
    fn close_pair(
        &self,
        sink: &mut Sink<'_>,
        inside: &mut FxHashMap<Cell, f64>,
        cell: Cell,
        parts: Parts,
        lefts: Option<&[Cell]>,
    ) {
        let Cell::P(o, key) = cell else {
            unreachable!()
        };
        let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        if parts.exterior && inside.contains_key(&Cell::C(key.open)) {
            self.emit(
                sink,
                inside,
                Inst {
                    head: Cell::C(o),
                    tails: [Some(Cell::C(key.open)), Some(cell)],
                    weight: self.e(self.terminal_au(pair)),
                    segs: Vec::new(),
                },
            );
        }
        let stem = self.e(self.stem(pair));
        if parts.stem {
            self.emit(
                sink,
                inside,
                Inst {
                    head: Cell::M(o, key.open),
                    tails: [Some(cell), None],
                    weight: stem,
                    segs: Vec::new(),
                },
            );
        }
        let lefts: Vec<Cell> = match (parts.m2, lefts) {
            (false, _) => Vec::new(),
            (true, Some(given)) => given.to_vec(),
            // A component `M(key.open, inner)` has its inner node strictly
            // before `key.open`, so probing those nodes finds every one without
            // scanning the table.
            (true, None) => (0..key.open.pos)
                .flat_map(|p| self.dfa.nodes_at(p).iter().copied())
                .map(|inner| Cell::M(key.open, inner))
                .filter(|m| inside.contains_key(m))
                .collect(),
        };
        for m in lefts {
            let Cell::M(_, inner) = m else { unreachable!() };
            self.emit(
                sink,
                inside,
                Inst {
                    head: Cell::M2(o, inner),
                    tails: [Some(m), Some(cell)],
                    weight: stem,
                    segs: Vec::new(),
                },
            );
        }
        if parts.singles {
            self.singles(sink, inside, cell, o, key, None);
        }
    }

    /// Stack, bulge and internal-loop closures around the pair `(o, key)`, or
    /// only the one closing the enclosing pair `outer` names.
    fn singles(
        &self,
        sink: &mut Sink<'_>,
        inside: &mut FxHashMap<Cell, f64>,
        cell: Cell,
        o: Node,
        key: PairKey,
        outer: Option<Outer>,
    ) {
        let a = key.open.pos;
        let inner_pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let b = o.pos - 1;
        // n1 = a − p − 1 ≤ L and n1 + n2 ≤ L (each flank may take the whole cap).
        let p_lo = a.saturating_sub(self.single_max_len as u32 + 1);
        let opens: Vec<(Node, Edge)> = match outer {
            Some(t) if t.pn.pos >= p_lo && t.pn.pos < a => vec![(t.pn, t.ep)],
            Some(_) => Vec::new(),
            None => self.edges_from(p_lo, a),
        };
        for (pn, ep) in opens {
            {
                {
                    let ep = &ep;
                    let Some(left_cai) = self.fill(ep.to, key.open) else {
                        continue;
                    };
                    let n1 = (key.open.pos - ep.to.pos) as usize;
                    let q_hi = (b + 2 + (self.single_max_len - n1) as u32).min(self.dfa.seq_len);
                    let closes: Vec<(Node, Edge)> = match outer {
                        Some(t) if t.qn.pos > b && t.qn.pos < q_hi => vec![(t.qn, t.eq)],
                        Some(_) => Vec::new(),
                        None => self.edges_from(b + 1, q_hi),
                    };
                    for (qn, eq) in closes {
                        {
                            {
                                let eq = &eq;
                                let outer_pair = pair_from_nucleotides(ep.nuc, eq.nuc);
                                if outer_pair == BasePair::NN
                                    || !self.must.free_span(pn.pos + 1, key.open.pos)
                                    || !self.must.free_span(o.pos, qn.pos)
                                {
                                    continue;
                                }
                                let Some(constraint) = self.constraint(pn.pos, qn.pos) else {
                                    continue;
                                };
                                let n2 = (qn.pos - o.pos) as usize;
                                let head = Cell::P(
                                    eq.to,
                                    PairKey {
                                        open: pn,
                                        close: qn,
                                        opening: ep.opening,
                                        nuc_close: eq.nuc,
                                    },
                                );
                                let ends = [Seg::Edge(pn, *ep), Seg::Edge(qn, *eq)];
                                let edge_cai = ep.cai + eq.cai;
                                if n1 > 0 && n2 > 0 {
                                    for (l5, l3, lcai, lsegs) in self.corners(ep.to, key.open) {
                                        for (r5, r3, rcai, rsegs) in self.corners(o, qn) {
                                            let energy = internal_energy_scalar(
                                                self.params,
                                                &InternalLoopScalar {
                                                    outer: outer_pair,
                                                    inner: inner_pair,
                                                    boundary_left5p: l5,
                                                    boundary_left3p: l3,
                                                    boundary_right5p: r5,
                                                    boundary_right3p: r3,
                                                    n1,
                                                    n2,
                                                },
                                            );
                                            let mut segs = ends.to_vec();
                                            segs.extend(lsegs.iter().copied());
                                            segs.extend(rsegs);
                                            self.emit(
                                                sink,
                                                inside,
                                                Inst {
                                                    head,
                                                    tails: [Some(cell), None],
                                                    weight: self.e(energy)
                                                        - self.kk * (edge_cai + lcai + rcai)
                                                        + constraint,
                                                    segs,
                                                },
                                            );
                                        }
                                    }
                                    continue;
                                }
                                let Some(right_cai) = self.fill(o, qn) else {
                                    continue;
                                };
                                let energy = if n1 == 0 && n2 == 0 {
                                    stack_energy(
                                        self.params,
                                        &StackPair {
                                            outer: outer_pair,
                                            inner: inner_pair,
                                        },
                                    )
                                } else {
                                    bulge_energy(
                                        self.params,
                                        &BulgeLoop {
                                            outer: outer_pair,
                                            inner: inner_pair,
                                            size: n1 + n2,
                                        },
                                    )
                                };
                                let mut segs = ends.to_vec();
                                segs.push(Seg::Fill(ep.to, key.open));
                                segs.push(Seg::Fill(o, qn));
                                self.emit(
                                    sink,
                                    inside,
                                    Inst {
                                        head,
                                        tails: [Some(cell), None],
                                        weight: self.e(energy)
                                            - self.kk * (edge_cai + left_cai + right_cai)
                                            + constraint,
                                        segs,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// `M = M2`, and the multi-loop wrap: a closing pair around an `M2` core
    /// with an unpaired flank on each side — the 5′ one at most the single-loop
    /// cap, the 3′ one unbounded.
    ///
    /// `identity` generates the `M = M2` instance; `outer`, when given,
    /// restricts the wrap to the one closing pair it names.
    fn close_m2(
        &self,
        sink: &mut Sink<'_>,
        inside: &mut FxHashMap<Cell, f64>,
        cell: Cell,
        identity: bool,
        outer: Option<Outer>,
    ) {
        self.close_m2_parts(sink, inside, cell, identity, true, outer);
    }

    /// [`Self::close_m2`], with the wrap generated only when `wrap` is set.
    fn close_m2_parts(
        &self,
        sink: &mut Sink<'_>,
        inside: &mut FxHashMap<Cell, f64>,
        cell: Cell,
        identity: bool,
        wrap: bool,
        outer: Option<Outer>,
    ) {
        let Cell::M2(core_outer, core_inner) = cell else {
            unreachable!()
        };
        if identity {
            self.emit(
                sink,
                inside,
                Inst {
                    head: Cell::M(core_outer, core_inner),
                    tails: [Some(cell), None],
                    weight: 0.0,
                    segs: Vec::new(),
                },
            );
        }
        if !wrap {
            return;
        }
        // The leading run is unbounded, as in the scan's `m2_to_multi`.
        let p_lo = 0;
        let opens: Vec<(Node, Edge)> = match outer {
            Some(t) if t.pn.pos >= p_lo && t.pn.pos < core_inner.pos => vec![(t.pn, t.ep)],
            Some(_) => Vec::new(),
            None => self.edges_from(p_lo, core_inner.pos),
        };
        for (pn, ep) in opens {
            {
                {
                    let ep = &ep;
                    let Some(left) = self.fill(ep.to, core_inner) else {
                        continue;
                    };
                    let q_nodes: Vec<Node> = match outer {
                        Some(t) if t.qn.pos >= core_outer.pos && t.qn.pos < self.dfa.seq_len => {
                            vec![t.qn]
                        }
                        Some(_) => Vec::new(),
                        None => (core_outer.pos..self.dfa.seq_len)
                            .flat_map(|q| self.dfa.nodes_at(q).iter().copied())
                            .collect(),
                    };
                    for qn in q_nodes {
                        {
                            let Some(right) = self.fill(core_outer, qn) else {
                                continue;
                            };
                            let closing: Vec<Edge> = match outer {
                                Some(t) => vec![t.eq],
                                None => self.dfa.right(qn).to_vec(),
                            };
                            if !self.must.free_span(pn.pos + 1, core_inner.pos)
                                || !self.must.free_span(core_outer.pos, qn.pos)
                            {
                                continue;
                            }
                            // A multi-loop its closing pair cannot close is no
                            // cell: its only consumer is that pair's `P` cell,
                            // which the constraint forbids.
                            if self.constraint(pn.pos, qn.pos).is_none() {
                                continue;
                            }
                            for eq in &closing {
                                if pair_from_nucleotides(ep.nuc, eq.nuc) == BasePair::NN {
                                    continue;
                                }
                                self.emit(
                                    sink,
                                    inside,
                                    Inst {
                                        head: Cell::Multi(
                                            eq.to,
                                            PairKey {
                                                open: pn,
                                                close: qn,
                                                opening: ep.opening,
                                                nuc_close: eq.nuc,
                                            },
                                        ),
                                        tails: [Some(cell), None],
                                        weight: -self.kk * (ep.cai + left + right + eq.cai),
                                        segs: vec![
                                            Seg::Edge(pn, *ep),
                                            Seg::Fill(ep.to, core_inner),
                                            Seg::Fill(core_outer, qn),
                                            Seg::Edge(qn, *eq),
                                        ],
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// One unpaired nucleotide onto the right of an exterior prefix or a
    /// multibranch component.
    fn extend(&self, sink: &mut Sink<'_>, inside: &mut FxHashMap<Cell, f64>, cell: Cell) {
        let from = match cell {
            Cell::C(n) | Cell::M(n, _) => n,
            _ => unreachable!(),
        };
        if self.must.is_must(from.pos) {
            return;
        }
        for e in self.dfa.right(from) {
            let head = match cell {
                Cell::C(_) => Cell::C(e.to),
                Cell::M(_, inner) => Cell::M(e.to, inner),
                _ => unreachable!(),
            };
            self.emit(
                sink,
                inside,
                Inst {
                    head,
                    tails: [Some(cell), None],
                    weight: -self.kk * e.cai,
                    segs: vec![Seg::Edge(from, *e)],
                },
            );
        }
    }

    /// Each instance's full value — its head's outside, its weight and its
    /// tails' insides — handed to every edge its choice spells.
    fn edge_marginals(
        &self,
        insts: &[Inst],
        inside: &FxHashMap<Cell, f64>,
        outside: &FxHashMap<Cell, f64>,
    ) -> FxHashMap<(Node, u8), f64> {
        let mut by_edge: FxHashMap<(Node, u8), f64> = FxHashMap::default();
        let mut relax = |id: (Node, u8), v: f64| {
            let slot = by_edge.entry(id).or_insert(f64::INFINITY);
            if v < *slot {
                *slot = v;
            }
        };
        for inst in insts {
            let Some(&head) = outside.get(&inst.head) else {
                continue;
            };
            let v =
                head + inst.weight + inst.tails.iter().flatten().map(|t| inside[t]).sum::<f64>();
            for seg in &inst.segs {
                match *seg {
                    Seg::Edge(from, e) => relax((from, e.nuc as u8), v),
                    Seg::Fill(a, b) => {
                        let best = self.fill(a, b).expect("a scored fill is connectable");
                        for pos in a.pos..b.pos {
                            for &n in self.dfa.nodes_at(pos) {
                                for e in self.dfa.right(n) {
                                    if e.to.pos > b.pos {
                                        continue;
                                    }
                                    let (Some(x), Some(y)) = (self.fill(a, n), self.fill(e.to, b))
                                    else {
                                        continue;
                                    };
                                    relax((n, e.nuc as u8), v + self.kk * (best - (x + e.cai + y)));
                                }
                            }
                        }
                    }
                }
            }
        }
        by_edge
    }
}
