//! The Aho–Corasick × codon-DFA product.
//!
//! Lifts each base-DFA node `(pos, lane)` to a product node `(pos, lane, q)`
//! with `q` an [`Ac`] state. Each base edge spelling nucleotide `x` from
//! `(pos, lane)` to `(pos+1, lane′)` lifts to
//! `(pos, lane, q) → (pos+1, lane′, δ(q, x))` for every reachable `q`, **except**
//! any edge whose target AC state is accepting — that edge is a forbidden
//! transition (a motif has completed) and is removed. Consequently every path
//! from `(0, 0, q₀)` to a terminal spells a motif-free sequence, and every
//! motif-free encoding is exactly one such path.
//!
//! This is a thin lift over [`Dfa`]: it reuses the base DFA's edges and codon
//! bookkeeping unchanged, adding only the AC-state axis and the per-node set of
//! reachable AC states — the raw material for the sparse layout the CKY tables
//! will thread `q` through, and for measuring how sparse the reachable
//! `(node, q)` set is in practice.
//!
//! The AC state at a node is an *absolute* function of the linear prefix
//! `x₀…x_{pos-1}`: a single concrete path pins one AC state at each position it
//! visits. A boundary shared by two sub-derivations of the same sequence — a
//! CKY bifurcation split — is therefore a single product node both agree on, so
//! a motif straddling the split threads continuously through that one state and
//! is caught by the removed accept transition on the continuing side. No
//! boundary node needs an independent entry-and-exit AC pair.
//!
//! **The argument holds unchanged when enforcement is scoped to a window**, and
//! is worth re-stating rather than assumed to carry, because it is the reason a
//! boundary at either edge of the window needs no special treatment. A scoped
//! state is still an absolute function of the prefix: the root below the
//! window's start, the automaton state of the in-window prefix inside it, and a
//! single sink at or past its end ([`crate::ac::ScopedAc`]). Two sub-derivations
//! meeting at a boundary therefore still agree on one state there — including a
//! boundary that *is* the window's edge, where they agree on the held root or on
//! the sink — so the product node remains single-valued and the bifurcation
//! argument is untouched.

use crate::ac::{Ac, MotifWindow, ScopedAc};
use crate::dfa::{Dfa, Edge, Node};
use crate::hash::{FxHashMap, FxHashSet};
use derna_codon_dp::codon_table::AMINO_ACIDS;
use rna_mfe::Nucleotide;
use std::collections::VecDeque;

/// The AC × DFA product for one protein and one forbidden-motif set.
///
/// Borrows the base [`Dfa`] and [`Ac`]; holds the per-node set of reachable AC
/// states computed by a forward pass from the start node.
pub struct ProductDfa<'a> {
    dfa: &'a Dfa,
    ac: ScopedAc<'a>,
    /// Reachable AC states at each base node, indexed by [`Dfa::ix`];
    /// sorted ascending and deduplicated. A node with no motif-free path from
    /// the start carries the empty set.
    reachable: Vec<Vec<u32>>,
    /// Prefix sum of `reachable` lengths, indexed by [`Dfa::ix`]: the dense
    /// id of a base node's first product node. `offset[i]` is the number of
    /// product nodes at base nodes with a strictly smaller index, so
    /// `offset[i] .. offset[i] + reachable[i].len()` is base node `i`'s
    /// contiguous block of dense product ids. Has `reachable.len() + 1` entries;
    /// the last is [`Self::num_product_nodes`]. This is the sparse layout's
    /// addressing material — a `(node, q)` product node maps to one dense id the
    /// CKY tables index by, materializing only the reachable `(node, q)` cells.
    offset: Vec<usize>,
    /// Whether the forward pass removed any edge at a product node it reached —
    /// see [`Self::dropped_edge`].
    dropped_edge: bool,
    /// The future-equivalence quotient, when [`Self::merge_equivalent`] built
    /// one: `reachable` then lists class representatives only.
    merged: Option<Quotient>,
}

/// Each forward-reachable state's class representative, per base node.
///
/// `states[i]` is base node `i`'s forward-reachable states as the forward pass
/// found them (sorted), and `rep[i][k]` the representative of `states[i][k]` —
/// the smallest state equivalent to it at that node — or [`DEAD`] when no
/// motif-free completion leaves it.
struct Quotient {
    states: Vec<Vec<u32>>,
    rep: Vec<Vec<u32>>,
}

/// A state with no motif-free completion to the terminal.
const DEAD: u32 = u32::MAX;

impl Quotient {
    fn rep_of(&self, i: usize, q: u32) -> Option<u32> {
        let k = self.states[i].binary_search(&q).ok()?;
        let r = self.rep[i][k];
        (r != DEAD).then_some(r)
    }
}

/// The automaton a fold threads through the product: the scoped Aho–Corasick
/// transition, followed — when the product was merged — by the move to the
/// target node's class representative.
///
/// Stepping is keyed by the **source node**, not only its position, because a
/// class is a property of a state at a node: two lanes at one position can
/// disagree about which states are interchangeable.
#[derive(Clone, Copy)]
pub struct LatticeAc<'a> {
    ac: ScopedAc<'a>,
    dfa: &'a Dfa,
    merged: Option<&'a Quotient>,
}

impl<'a> LatticeAc<'a> {
    /// `ac` over `dfa` with no quotient — the plain scoped transition.
    pub fn unmerged(ac: ScopedAc<'a>, dfa: &'a Dfa) -> Self {
        Self {
            ac,
            dfa,
            merged: None,
        }
    }

    /// The automaton's start state, which is its own representative.
    pub fn start(&self) -> u32 {
        self.ac.start()
    }

    /// The underlying automaton's state count — the nominal width of the axis,
    /// not the number of classes.
    pub fn num_states(&self) -> usize {
        self.ac.num_states()
    }

    /// Read `x` from state `q` at node `from`: `None` when the transition is
    /// forbidden, or — on a merged product — when it lands on a state no
    /// motif-free completion leaves. Otherwise the target state, as its class
    /// representative when merged.
    pub fn step(&self, from: Node, q: u32, x: Nucleotide) -> Option<u32> {
        let next = self.ac.step_at(from.pos, q, x)?;
        let Some(merged) = self.merged else {
            return Some(next);
        };
        let to = self.dfa.right(from).iter().find(|e| e.nuc == x)?.to;
        merged.rep_of(self.dfa.ix(to), next)
    }

    /// [`Self::step`] across the out-edge `e` of `from`, for a caller that
    /// already holds the edge: the same transition without searching
    /// `from`'s out-edges for the one spelling `e.nuc`.
    pub fn step_edge(&self, from: Node, q: u32, e: Edge) -> Option<u32> {
        debug_assert!(
            self.dfa
                .right(from)
                .iter()
                .any(|o| o.nuc == e.nuc && o.to == e.to),
            "not an out-edge of `from`"
        );
        let next = self.ac.step_at(from.pos, q, e.nuc)?;
        match self.merged {
            None => Some(next),
            Some(merged) => merged.rep_of(self.dfa.ix(e.to), next),
        }
    }
}

impl<'a> ProductDfa<'a> {
    /// Build the product with the motif set enforced over the whole molecule.
    pub fn build(dfa: &'a Dfa, ac: &'a Ac) -> Self {
        Self::build_scoped(dfa, ac, MotifWindow::all())
    }

    /// Build the product by a forward reachability pass from `(start, q₀)`,
    /// following base edges and threading the AC state, skipping every edge the
    /// scoped automaton removes.
    ///
    /// `window` is the range of positions enforcement applies over; see
    /// [`MotifWindow`] for exactly which occurrences that forbids.
    /// [`MotifWindow::all`] reproduces global enforcement transition for
    /// transition.
    pub fn build_scoped(dfa: &'a Dfa, ac: &'a Ac, window: MotifWindow) -> Self {
        let ac = ScopedAc::new(ac, window);
        let n_nodes = dfa.slots();
        let mut reachable: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        let mut visited: FxHashSet<(u32, u32)> = FxHashSet::default();
        let mut queue = VecDeque::new();

        let (start, q0) = (dfa.start(), ac.start());
        visited.insert((dfa.ix(start) as u32, q0));
        reachable[dfa.ix(start)].push(q0);
        queue.push_back((start, q0));

        let mut dropped_edge = false;
        while let Some((node, q)) = queue.pop_front() {
            for edge in dfa.right(node) {
                let Some(next_q) = ac.step_at(node.pos, q, edge.nuc) else {
                    dropped_edge = true;
                    continue;
                };
                if visited.insert((dfa.ix(edge.to) as u32, next_q)) {
                    reachable[dfa.ix(edge.to)].push(next_q);
                    queue.push_back((edge.to, next_q));
                }
            }
        }
        for states in &mut reachable {
            states.sort_unstable();
        }
        let offset = prefix_sums(&reachable);
        Self {
            dfa,
            ac,
            reachable,
            offset,
            dropped_edge,
            merged: None,
        }
    }

    /// Merge states that admit the same motif-free completions, node by node,
    /// and drop states that admit none.
    ///
    /// Equivalent states step to equivalent states along every surviving edge,
    /// so a fold that keys its cells on the class representative instead of the
    /// state searches exactly the same feasible set, with fewer cells: every
    /// span's paths, costs and exit classes are unchanged. Dead states carry no
    /// encoding, so dropping them removes nothing either.
    ///
    /// Computed in one backward pass. A state's signature is, per base out-edge
    /// in edge order, the representative it lands on (or nothing, when the edge
    /// is removed or lands dead); equal signatures at one node are one class,
    /// represented by its smallest state. The live states at one terminal are
    /// one class, since nothing follows a terminal.
    ///
    /// Requires the base DFA to be deterministic per node — one out-edge per
    /// nucleotide — because a step finds its target node by the nucleotide it
    /// reads. A DFA that is not is returned unmerged.
    pub fn merge_equivalent(mut self) -> Self {
        let deterministic = (0..=self.dfa.seq_len).all(|pos| {
            self.dfa.nodes_at(pos).iter().all(|&node| {
                let edges = self.dfa.right(node);
                (0..edges.len()).all(|a| (a + 1..edges.len()).all(|b| edges[a].nuc != edges[b].nuc))
            })
        });
        if !deterministic {
            return self;
        }
        let mut rep: Vec<Vec<u32>> = self.reachable.iter().map(|s| vec![DEAD; s.len()]).collect();
        for &end in self.dfa.ends() {
            if let Some(&first) = self.reachable[self.dfa.ix(end)].first() {
                rep[self.dfa.ix(end)].fill(first);
            }
        }
        for pos in (0..self.dfa.seq_len).rev() {
            for &node in self.dfa.nodes_at(pos) {
                let mut seen: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
                for (k, &q) in self.reachable[self.dfa.ix(node)].iter().enumerate() {
                    let sig: Vec<u32> = self
                        .dfa
                        .right(node)
                        .iter()
                        .map(|edge| {
                            self.ac
                                .step_at(node.pos, q, edge.nuc)
                                .and_then(|next| {
                                    let i = self.dfa.ix(edge.to);
                                    let at = self.reachable[i].binary_search(&next).ok()?;
                                    Some(rep[i][at])
                                })
                                .unwrap_or(DEAD)
                        })
                        .collect();
                    if sig.iter().all(|&r| r == DEAD) {
                        continue;
                    }
                    // States are visited ascending, so the first holder of a
                    // signature is the smallest member of its class.
                    rep[self.dfa.ix(node)][k] = *seen.entry(sig).or_insert(q);
                }
            }
        }
        let classes: Vec<Vec<u32>> = rep
            .iter()
            .map(|r| {
                let mut live: Vec<u32> = r.iter().copied().filter(|&x| x != DEAD).collect();
                live.sort_unstable();
                live.dedup();
                live
            })
            .collect();
        let states = std::mem::replace(&mut self.reachable, classes);
        self.offset = prefix_sums(&self.reachable);
        self.merged = Some(Quotient { states, rep });
        self
    }

    /// Whether [`Self::merge_equivalent`] quotiented this product.
    pub fn is_merged(&self) -> bool {
        self.merged.is_some()
    }

    /// The automaton a fold over this product threads — see [`LatticeAc`].
    pub fn lattice_ac(&self) -> LatticeAc<'_> {
        LatticeAc {
            ac: self.ac,
            dfa: self.dfa,
            merged: self.merged.as_ref(),
        }
    }

    /// The AC states reachable at `node` under the constraint, sorted
    /// ascending. Empty when no motif-free path reaches `node`.
    pub fn reachable_states(&self, node: Node) -> &[u32] {
        &self.reachable[self.dfa.ix(node)]
    }

    /// The base codon DFA the product lifts.
    pub fn dfa(&self) -> &Dfa {
        self.dfa
    }

    /// The Aho–Corasick automaton the product threads, carrying the window
    /// enforcement is scoped to.
    pub fn ac(&self) -> ScopedAc<'a> {
        self.ac
    }

    /// The window enforcement is scoped to.
    pub fn window(&self) -> MotifWindow {
        self.ac.window()
    }

    /// The total number of materialized product nodes `(node, q)` — the
    /// reachable-cell count that sizes the sparse CKY layout and quantifies the
    /// AC-dimension sparsity.
    pub fn num_product_nodes(&self) -> usize {
        self.offset[self.offset.len() - 1]
    }

    /// The dense id of the product node `(node, q)`, or `None` when `q` is not
    /// reachable at `node`. Ids are contiguous per base node in [`Dfa::ix`]
    /// order and cover `0 .. num_product_nodes()`, so a CKY table can address
    /// the sparse AC dimension by a single index while materializing only the
    /// reachable `(node, q)` cells rather than the nominal `|nodes| · |A_M|`. The
    /// base node of a dense id is recovered with [`Self::base_node_of`].
    pub fn product_id(&self, node: Node, q: u32) -> Option<usize> {
        let states = &self.reachable[self.dfa.ix(node)];
        states
            .binary_search(&q)
            .ok()
            .map(|local| self.offset[self.dfa.ix(node)] + local)
    }

    /// The base [`Node`] a dense product id belongs to — the inverse of
    /// [`Self::product_id`]'s node component, found by locating the id's block
    /// in the per-node offset table. Panics if `id >= num_product_nodes()`.
    pub fn base_node_of(&self, id: usize) -> Node {
        debug_assert!(id < self.num_product_nodes(), "product id out of range");
        // `offset` is non-decreasing; the block containing `id` is the last
        // whose start is `<= id`. `partition_point` finds the first start
        // `> id`, so its predecessor is that block.
        let idx = self.offset.partition_point(|&start| start <= id) - 1;
        self.dfa.node_of(idx)
    }

    /// Whether some terminal is reachable under the constraint — i.e. the
    /// protein has at least one motif-free encoding. When false, the design is
    /// infeasible under the given motif set.
    pub fn is_feasible(&self) -> bool {
        self.dfa
            .ends()
            .iter()
            .any(|&end| !self.reachable[self.dfa.ix(end)].is_empty())
    }

    /// Whether the automaton imposes no constraint, so the product is
    /// structurally identical to the base DFA: every reachable node carries
    /// exactly the single root AC state, and no edge is removed.
    pub fn is_pass_through(&self) -> bool {
        self.ac.is_pass_through()
    }

    /// Whether the constraint removed any transition the base lattice offers —
    /// i.e. whether this pattern set forbids anything **this** protein can
    /// spell, as opposed to something some other protein could.
    ///
    /// False means the forward pass reached every product node the base DFA's
    /// paths lift to and never skipped an edge, so by induction from
    /// `(start, q₀)` every base-DFA path from the start is present in the
    /// product: the two lattices accept the same sequences and the constrained
    /// optimum is the unconstrained one. A caller may then route to the base
    /// scan and emit the same design far more cheaply.
    ///
    /// It is deliberately **not** a census of surviving codons. An all-true
    /// census is a per-codon statement, and a pattern straddling a codon
    /// boundary can remove a spelling while leaving every codon usable
    /// somewhere — the census reads inert and the constraint is not. It is also
    /// deliberately not a threshold on how much was removed: the predicate is
    /// conservative in one direction only, because routing a genuinely
    /// constraining set to the base scan would emit an infeasible design with
    /// no error.
    ///
    /// Conservative in the safe direction on unreachable tails, too: an edge
    /// skipped at a node from which the terminal is unreachable still sets the
    /// flag, so the answer errs toward the constrained path.
    pub fn dropped_edge(&self) -> bool {
        self.dropped_edge
    }

    /// Which synonymous codons the motif set leaves usable, per residue.
    ///
    /// `out[i][c]` says whether codon `c` of residue `i` — indexed into that
    /// amino acid's own synonym list — lies on some start-to-terminal path, so
    /// that **some** motif-free encoding of the protein spells it there.
    ///
    /// It is what separates two situations [`Self::is_feasible`] cannot tell
    /// apart on its own: a set that forbade something the protein could have
    /// spelled and left it room anyway, against a set that forbade nothing
    /// capable of occurring. An all-true census under a non-empty motif set is
    /// the second — the constraint is satisfiable because it is inert here, not
    /// because the lattice absorbed it. An all-false census is infeasibility,
    /// where no encoding survives to use any codon.
    ///
    /// Feasibility is a property of the terminal alone, so it needs only the
    /// forward pass. This needs the backward companion too: a state the start
    /// reaches but which reaches no terminal carries no encoding, and counting
    /// it would credit a codon to a path that dies later.
    pub fn surviving_codons(&self) -> Vec<Vec<bool>> {
        let co = self.co_reachable();
        let cds_start = self.dfa.cds_start();
        self.dfa
            .protein
            .iter()
            .enumerate()
            .map(|(i, &residue)| {
                let pos = cds_start + 3 * i as u32;
                let seeds = self.live_at(&co, pos);
                AMINO_ACIDS[residue]
                    .codons
                    .iter()
                    .map(|codon| self.spells_codon(&co, pos, &seeds, codon.nuc))
                    .collect()
            })
            .collect()
    }

    /// The best adaptation a motif-free encoding reaches, as the sum of its
    /// edges' CAI log-weights, and which codons of each residue lie on some
    /// encoding that reaches it — indexed as [`Self::surviving_codons`]
    /// indexes them. `None` when no encoding survives.
    ///
    /// Adaptation is a sum over edges, so the best encoding through each
    /// product node is the best prefix into it plus the best suffix out of
    /// it, and a codon is on a best encoding exactly when some live node at
    /// its residue's first base, threaded through its three bases, closes
    /// that sum at the best. Two encodings within `1e-9` of each other count
    /// as equally adapted.
    pub fn most_adapted_codons(&self) -> Option<(f64, Vec<Vec<bool>>)> {
        const SAME: f64 = 1e-9;
        let lattice_ac = self.lattice_ac();
        let ac = &lattice_ac;
        let at = |values: &[Vec<f64>], node: Node, q: u32| {
            self.reachable_states(node)
                .binary_search(&q)
                .map_or(f64::NEG_INFINITY, |i| values[self.dfa.ix(node)][i])
        };
        let blank = || -> Vec<Vec<f64>> {
            self.reachable
                .iter()
                .map(|states| vec![f64::NEG_INFINITY; states.len()])
                .collect()
        };
        // The best prefix into each product node, swept 5′ to 3′.
        let mut into = blank();
        let start = self.dfa.start();
        let first = self
            .reachable_states(start)
            .binary_search(&self.ac.start())
            .ok()?;
        into[self.dfa.ix(start)][first] = 0.0;
        for pos in 0..self.dfa.seq_len {
            for &node in self.dfa.nodes_at(pos) {
                for (i, &q) in self.reachable_states(node).iter().enumerate() {
                    let here = into[self.dfa.ix(node)][i];
                    if here == f64::NEG_INFINITY {
                        continue;
                    }
                    for edge in self.dfa.right(node) {
                        let Some(next) = ac.step(node, q, edge.nuc) else {
                            continue;
                        };
                        if let Ok(j) = self.reachable_states(edge.to).binary_search(&next) {
                            let to = &mut into[self.dfa.ix(edge.to)][j];
                            *to = to.max(here + edge.cai);
                        }
                    }
                }
            }
        }
        // The best suffix out of each product node, swept 3′ to 5′.
        let mut out = blank();
        for &end in self.dfa.ends() {
            out[self.dfa.ix(end)].fill(0.0);
        }
        for pos in (0..self.dfa.seq_len).rev() {
            for &node in self.dfa.nodes_at(pos) {
                for (i, &q) in self.reachable_states(node).iter().enumerate() {
                    let best = self
                        .dfa
                        .right(node)
                        .iter()
                        .filter_map(|edge| {
                            let next = ac.step(node, q, edge.nuc)?;
                            Some(edge.cai + at(&out, edge.to, next))
                        })
                        .fold(f64::NEG_INFINITY, f64::max);
                    out[self.dfa.ix(node)][i] = best;
                }
            }
        }
        let best = self
            .dfa
            .ends()
            .iter()
            .flat_map(|&end| self.reachable_states(end).iter().map(move |&q| (end, q)))
            .map(|(end, q)| at(&into, end, q))
            .fold(f64::NEG_INFINITY, f64::max);
        if best == f64::NEG_INFINITY {
            return None;
        }
        let cds_start = self.dfa.cds_start();
        let codons = self
            .dfa
            .protein
            .iter()
            .enumerate()
            .map(|(i, &residue)| {
                let pos = cds_start + 3 * i as u32;
                AMINO_ACIDS[residue]
                    .codons
                    .iter()
                    .map(|codon| {
                        // Every live node at `pos` with its best prefix,
                        // threaded through the codon's bases with the
                        // weight they add.
                        let mut frontier: Vec<(Node, u32, f64)> = self
                            .dfa
                            .nodes_at(pos)
                            .iter()
                            .flat_map(|&node| {
                                self.reachable_states(node).iter().map(move |&q| (node, q))
                            })
                            .map(|(node, q)| (node, q, at(&into, node, q)))
                            .filter(|&(_, _, w)| w > f64::NEG_INFINITY)
                            .collect();
                        for &base in &codon.nuc {
                            let x = nucleotide(base);
                            frontier = frontier
                                .iter()
                                .flat_map(|&(node, q, w)| {
                                    self.dfa.right(node).iter().filter_map(move |edge| {
                                        let next = ac.step(node, q, x).filter(|_| edge.nuc == x)?;
                                        Some((edge.to, next, w + edge.cai))
                                    })
                                })
                                .collect();
                        }
                        frontier
                            .iter()
                            .any(|&(node, q, w)| w + at(&out, node, q) >= best - SAME)
                    })
                    .collect()
            })
            .collect();
        Some((best, codons))
    }

    /// The product nodes from which the terminal is reachable, indexed by
    /// [`Dfa::ix`] and sorted ascending like [`Self::reachable_states`].
    ///
    /// Swept from the 3′ end back so every state is tested with a **forward**
    /// step: the scoped transition is a function, so asking which successors a
    /// state reaches is a lookup, while asking which predecessors reach it would
    /// need that function inverted.
    fn co_reachable(&self) -> Vec<Vec<u32>> {
        let mut co: Vec<Vec<u32>> = vec![Vec::new(); self.reachable.len()];
        for &end in self.dfa.ends() {
            co[self.dfa.ix(end)] = self.reachable_states(end).to_vec();
        }
        for pos in (0..self.dfa.seq_len).rev() {
            for &node in self.dfa.nodes_at(pos) {
                co[self.dfa.ix(node)] = self
                    .reachable_states(node)
                    .iter()
                    .copied()
                    .filter(|&q| self.continues(&co, node, q))
                    .collect();
            }
        }
        co
    }

    /// Whether some out-edge of `(node, q)` survives the constraint and lands on
    /// a product node the terminal is reachable from.
    fn continues(&self, co: &[Vec<u32>], node: Node, q: u32) -> bool {
        let ac = self.lattice_ac();
        self.dfa.right(node).iter().any(|edge| {
            ac.step(node, q, edge.nuc)
                .is_some_and(|next| co[self.dfa.ix(edge.to)].binary_search(&next).is_ok())
        })
    }

    /// Every product node at `pos` that lies on some complete encoding.
    fn live_at(&self, co: &[Vec<u32>], pos: u32) -> Vec<(Node, u32)> {
        self.dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&node| co[self.dfa.ix(node)].iter().map(move |&q| (node, q)))
            .collect()
    }

    /// Whether some complete motif-free encoding spells `codon` at `pos`.
    ///
    /// Threads the three nucleotides from every live product node at `pos`,
    /// keeping only successors that are themselves live — so a surviving
    /// frontier at the codon's end is a start-to-terminal path through it, not
    /// a prefix that has merely not died yet.
    fn spells_codon(
        &self,
        co: &[Vec<u32>],
        pos: u32,
        seeds: &[(Node, u32)],
        codon: [u8; 3],
    ) -> bool {
        let mut frontier = seeds.to_vec();
        for (offset, &base) in codon.iter().enumerate() {
            let at = pos + offset as u32;
            frontier = self.advance(&frontier, at, nucleotide(base));
            frontier.retain(|&(node, q)| co[self.dfa.ix(node)].binary_search(&q).is_ok());
            if frontier.is_empty() {
                return false;
            }
        }
        true
    }

    /// Whether the product spells `seq`: a start-to-terminal path exists that
    /// spells `seq` nucleotide-for-nucleotide without traversing a forbidden
    /// (accept-target) edge. Returns false if the base DFA cannot spell `seq`
    /// or if `seq` contains a forbidden occurrence inside the window.
    ///
    /// The AC state is pinned by the spelled prefix, so it threads identically
    /// down every base-DFA lane; only the base path branches.
    pub fn spells(&self, seq: &[Nucleotide]) -> bool {
        let mut frontier = vec![(self.dfa.start(), self.ac.start())];
        for (pos, &x) in seq.iter().enumerate() {
            frontier = self.advance(&frontier, pos as u32, x);
            if frontier.is_empty() {
                return false;
            }
        }
        frontier
            .iter()
            .any(|&(node, _)| self.dfa.ends().contains(&node))
    }

    /// One reachability step at sequence position `pos`: from every product node
    /// in `frontier`, take the base edges spelling `x`, thread the AC state, and
    /// drop forbidden transitions. Deduplicated so parallel lanes converging on
    /// one node do not multiply the frontier.
    fn advance(&self, frontier: &[(Node, u32)], pos: u32, x: Nucleotide) -> Vec<(Node, u32)> {
        debug_assert!(frontier.iter().all(|&(node, _)| node.pos == pos));
        let ac = self.lattice_ac();
        let mut next = Vec::new();
        for &(node, q) in frontier {
            for edge in self.dfa.right(node) {
                if edge.nuc == x {
                    if let Some(next_q) = ac.step(node, q, x) {
                        next.push((edge.to, next_q));
                    }
                }
            }
        }
        next.sort_unstable_by_key(|&(node, q)| (self.dfa.ix(node), q));
        next.dedup();
        next
    }
}

/// Prefix sums of the per-node reachable-state counts: `out[i]` is the total
/// number of product nodes at base nodes with index `< i`, so `out[i]` is base
/// node `i`'s first dense product id and `out[reachable.len()]` is the grand
/// total. Length is `reachable.len() + 1`.
fn prefix_sums(reachable: &[Vec<u32>]) -> Vec<usize> {
    let mut out = Vec::with_capacity(reachable.len() + 1);
    let mut acc = 0;
    out.push(0);
    for states in reachable {
        acc += states.len();
        out.push(acc);
    }
    out
}

/// The codon table's nucleotide encoding (`A=0, C=1, G=2, U=3`), which is
/// [`Nucleotide`]'s own discriminant.
fn nucleotide(index: u8) -> Nucleotide {
    match index {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        _ => Nucleotide::U,
    }
}
