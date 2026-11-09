//! Designs in exact objective order, by lazy k-best over the recorded grammar.
//!
//! The recorded grammar of [`super`] is a hypergraph whose derivations are
//! `(sequence, structure)` pairs, each valued at its constrained objective. Its
//! best derivation is the design optimum, and its derivations in ascending
//! order visit every design at its own optimum the first time the design's
//! sequence appears — a design's best structure is its first derivation. So the
//! first derivation whose sequence verifies modular is the optimum over modular
//! designs, the same answer the bounded search proves by partitioning.
//!
//! Two things make the recorded instances unfit for enumeration as they stand,
//! and both are undone here:
//!
//! - **Fills.** A loop's energy-irrelevant interior is spelled by the
//!   best-adaptation path inside a single instance, so the instance stands for
//!   one spelling out of many. Each [`Seg::Fill`] becomes a tail node whose own
//!   derivations are the paths between its ends in ascending cost, expanded on
//!   demand. An instance's weight carries its fills' best costs, and they are
//!   taken back out so that the tail supplies them.
//! - **Minimization.** The inside pass keeps one value per cell. The walk keeps,
//!   per node, the derivations requested so far and a candidate heap, and asks a
//!   tail for its next derivation only when a head needs it (Huang & Chiang,
//!   *Better k-best parsing*, 2005, Algorithm 3).
//!
//! Distinct sequences are what the search needs, and a sequence can recur under
//! many structures before the next one appears. That recurrence is what this
//! walk is here to measure.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::OnceLock;
use std::time::Instant;

use super::{Cell, Grammar, Inst, Outer, Parts, Seg, Sink, MIN_HAIRPIN_LOOP};
use crate::dfa::{self, Node};
use crate::free_fold::{CdsWindow, FreeFoldError};
use fingerprint::Fingerprint;

mod fingerprint;
use crate::hash::{FxHashMap, FxHashSet};
use crate::must_pair::MustPairConstraint;
use crate::region::{PairConstraint, RegionConstraints};
use crate::tables::Tables;
mod arena;
mod check;
mod seen;
pub mod window;
use crate::{parse_utr, BeamError, CodonUsage, Params, ScanConfig};

/// The finished cells, indexed by where a pull looks for them.
struct Index {
    /// Closed pairs by their outer node.
    p_at: FxHashMap<Node, Vec<Cell>>,
    /// Closed pairs by their opening position.
    p_by_open: FxHashMap<u32, Vec<Cell>>,
    /// Two-branch multibranch components by their inner position.
    m2_by_inner: FxHashMap<u32, Vec<Cell>>,
    /// The same components by inside value, ascending: each value with the
    /// component's place in `m2_by_inner`.
    m2_by_value: FxHashMap<u32, Vec<(f64, u32)>>,
}

impl Index {
    fn new(inside: &FxHashMap<Cell, f64>) -> Self {
        let mut ix = Index {
            p_at: FxHashMap::default(),
            p_by_open: FxHashMap::default(),
            m2_by_inner: FxHashMap::default(),
            m2_by_value: FxHashMap::default(),
        };
        for &c in inside.keys() {
            match c {
                Cell::P(o, key) => {
                    ix.p_at.entry(o).or_default().push(c);
                    ix.p_by_open.entry(key.open.pos).or_default().push(c);
                }
                Cell::M2(_, inner) => ix.m2_by_inner.entry(inner.pos).or_default().push(c),
                _ => {}
            }
        }
        // A fixed order, so a pull generates its instances the same way each run.
        for v in ix.p_at.values_mut() {
            v.sort_unstable_by_key(order_key);
        }
        for v in ix.p_by_open.values_mut() {
            v.sort_unstable_by_key(order_key);
        }
        for v in ix.m2_by_inner.values_mut() {
            v.sort_unstable_by_key(order_key);
        }
        for (&pos, cells) in &ix.m2_by_inner {
            let mut by_value: Vec<(f64, u32)> = cells
                .iter()
                .enumerate()
                .map(|(i, c)| (inside[c], i as u32))
                .collect();
            by_value.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            ix.m2_by_value.insert(pos, by_value);
        }
        ix
    }
}

/// `x`'s decimal digits as nibbles `digit + 1`, most significant first and
/// left-aligned in 32 bits. Two such keys compare as the decimal strings do,
/// a string before any longer one it begins.
fn decimal_key(x: u32) -> u32 {
    assert!(x < 100_000_000, "a position past eight decimal digits");
    let digits = x.checked_ilog10().unwrap_or(0) + 1;
    let (mut key, mut rest) = (0u32, x);
    for i in 0..digits {
        key |= (rest % 10 + 1) << (4 * (8 - digits + i));
        rest /= 10;
    }
    key
}

/// A node's [`decimal_key`] position followed by its lane.
fn node_key(n: Node) -> u128 {
    debug_assert!(n.lane < 10, "a lane is one decimal digit");
    (u128::from(decimal_key(n.pos)) << 4) | u128::from(n.lane)
}

/// An integer that orders the cells an [`Index`] holds as their `Debug`
/// strings order lexicographically: field by field in declaration order, each
/// position compared as a decimal string. A bucket holds one variant, so the
/// variant needs no bits.
fn order_key(c: &Cell) -> u128 {
    match *c {
        Cell::P(o, k) => {
            debug_assert!(k.lane_open() < 10, "a lane is one decimal digit");
            let nucs = ((k.nuc_open() as u128) << 2) | k.nuc_close as u128;
            (node_key(o) << 80)
                | (node_key(k.open) << 44)
                | (node_key(k.close) << 8)
                | (nucs << 4)
                | u128::from(k.lane_open())
        }
        Cell::M2(o, inner) => (node_key(o) << 36) | node_key(inner),
        _ => unreachable!("an index holds closed pairs and two-branch components"),
    }
}

/// Every cell a finished production scan holds, valued on the scale the
/// reference sweep values its cells: `λ·energy − (1−λ)·100·adaptation`, on the
/// lattice's own scale.
fn production_inside(tables: &Tables, lambda: f64) -> FxHashMap<Cell, f64> {
    let mut inside = FxHashMap::default();
    for (i, c) in tables.c.iter().enumerate() {
        let o = tables.layout().node_of(i);
        if let Some(s) = c {
            inside.insert(Cell::C(o), s.brd(lambda));
        }
        for (k, s) in tables.p_states(o) {
            inside.insert(Cell::P(o, k), s.brd(lambda));
        }
        for (k, s) in tables.multi_states(o) {
            inside.insert(Cell::Multi(o, k), s.brd(lambda));
        }
        for (inner, s) in tables.m_states(o) {
            inside.insert(Cell::M(o, inner), s.brd(lambda));
        }
        for (inner, s) in tables.m2_states(o) {
            inside.insert(Cell::M2(o, inner), s.brd(lambda));
        }
    }
    inside
}

/// How one kind of cell compares between the reference sweep and a production
/// scan of the same lattice.
#[derive(Debug, Clone, Default)]
pub struct KindComparison {
    pub kind: &'static str,
    /// Cells of this kind each side holds.
    pub reference: usize,
    pub production: usize,
    /// Cells only one side holds.
    pub only_reference: usize,
    pub only_production: usize,
    /// Cells both hold whose values differ by more than `1e-6`.
    pub differing: usize,
    /// The largest `|reference − production|` over cells both hold.
    pub max_abs_diff: f64,
    /// One cell of each disagreement, for a first look.
    pub example_only_reference: Option<String>,
    pub example_only_production: Option<String>,
    pub example_differing: Option<String>,
}

/// The reference sweep's cells against a production scan's, kind by kind, with
/// the seconds each took.
#[derive(Debug, Clone)]
pub struct InsideComparison {
    pub kinds: Vec<KindComparison>,
    pub sweep_s: f64,
    pub scan_s: f64,
}

impl InsideComparison {
    /// Whether the two hold the same cells with the same values.
    pub fn identical(&self) -> bool {
        self.kinds
            .iter()
            .all(|k| k.only_reference == 0 && k.only_production == 0 && k.differing == 0)
    }
}

fn kind_of(c: &Cell) -> usize {
    match c {
        Cell::C(_) => 0,
        Cell::P(..) => 1,
        Cell::Multi(..) => 2,
        Cell::M(..) => 3,
        Cell::M2(..) => 4,
    }
}

fn compare_cells(
    reference: &FxHashMap<Cell, f64>,
    production: &FxHashMap<Cell, f64>,
) -> Vec<KindComparison> {
    let mut kinds: Vec<KindComparison> = ["C", "P", "Multi", "M", "M2"]
        .iter()
        .map(|&kind| KindComparison {
            kind,
            ..KindComparison::default()
        })
        .collect();
    for (c, &r) in reference {
        let k = &mut kinds[kind_of(c)];
        k.reference += 1;
        match production.get(c) {
            None => {
                k.only_reference += 1;
                k.example_only_reference
                    .get_or_insert_with(|| format!("{c:?} = {r}"));
            }
            Some(&p) => {
                let d = (r - p).abs();
                k.max_abs_diff = k.max_abs_diff.max(d);
                if d > 1e-6 {
                    k.differing += 1;
                    k.example_differing
                        .get_or_insert_with(|| format!("{c:?}: reference {r}, production {p}"));
                }
            }
        }
    }
    for (c, &p) in production {
        let k = &mut kinds[kind_of(c)];
        k.production += 1;
        if !reference.contains_key(c) {
            k.only_production += 1;
            k.example_only_production
                .get_or_insert_with(|| format!("{c:?} = {p}"));
        }
    }
    kinds
}

/// The reference sweep's inside values against the production scan's tables
/// over the same lattice: the same cells, and the same value in each.
/// `Ok(None)` when the lattice admits no design.
#[allow(clippy::too_many_arguments)]
pub fn compare_inside(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<InsideComparison>, BeamError> {
    compare_inside_must(
        params,
        protein,
        usage,
        lambda,
        allowed,
        (utr5, utr3),
        constraints,
        &[],
        config,
    )
}

/// [`compare_inside`] with every position of `must_pair` (absolute, 0-based)
/// required paired on both sides: the reference sweep's rules and the
/// production scan.
#[allow(clippy::too_many_arguments)]
pub fn compare_inside_must(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    (utr5, utr3): (&str, &str),
    constraints: RegionConstraints,
    must_pair: &[u32],
    config: ScanConfig,
) -> Result<Option<InsideComparison>, BeamError> {
    config.validate()?;
    let aa = crate::residues(protein)?;
    let (u5, u3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &u5, &u3) else {
        return Ok(None);
    };
    let grammar = Grammar {
        params,
        dfa: &graph,
        lambda,
        kk: (1.0 - lambda) * 100.0,
        constraint: PairConstraint::new(&constraints, &graph),
        must: MustPairConstraint::new(must_pair, graph.seq_len),
        single_max_len: config.single_max_len,
    };
    let t0 = Instant::now();
    let (_, reference) = grammar.sweep(false);
    let sweep_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let Some((_, tables)) = crate::scan_restricted_with_region_must(
        params,
        protein,
        usage,
        lambda,
        allowed,
        (utr5, utr3),
        constraints,
        must_pair,
        config,
    )?
    else {
        return Ok(None);
    };
    let scan_s = t1.elapsed().as_secs_f64();
    let production = production_inside(&tables, lambda);
    Ok(Some(InsideComparison {
        kinds: compare_cells(&reference, &production),
        sweep_s,
        scan_s,
    }))
}

impl Grammar<'_> {
    /// Every rule instance whose head is `head`, generated from the finished
    /// inside values by the same rules the sweep ran, each restricted to the
    /// tails that can produce `head`.
    ///
    /// `closed` says whether `head`'s energy bound closes a derivation valued
    /// at least the given value. A multi-loop wrap is valued at least its
    /// core's inside value — its flanks and edges only add adaptation — so a
    /// core closed at that value is skipped before its wraps are built: with
    /// the leading run unbounded, most cores lie far from the opening and
    /// most of those are closed.
    fn pull(
        &self,
        inside: &mut FxHashMap<Cell, f64>,
        ix: &Index,
        head: Cell,
        closed: &dyn Fn(f64) -> bool,
    ) -> Vec<Inst> {
        let mut out = Vec::new();
        let mut sink = Sink::Pull {
            head,
            out: &mut out,
        };
        let cap = self.single_max_len as u32;
        let only = |exterior, stem, m2| Parts {
            exterior,
            stem,
            m2,
            singles: false,
        };
        match head {
            Cell::C(o) => {
                if o.pos > 0 {
                    for &pn in self.dfa.nodes_at(o.pos - 1) {
                        if inside.contains_key(&Cell::C(pn)) {
                            self.extend(&mut sink, inside, Cell::C(pn));
                        }
                    }
                }
                for &pc in ix.p_at.get(&o).map_or(&[][..], |v| &v[..]) {
                    self.close_pair(&mut sink, inside, pc, only(true, false, false), None);
                }
            }
            Cell::M(o, inner) => {
                if o.pos > 0 {
                    for &pn in self.dfa.nodes_at(o.pos - 1) {
                        if inside.contains_key(&Cell::M(pn, inner)) {
                            self.extend(&mut sink, inside, Cell::M(pn, inner));
                        }
                    }
                }
                for &pc in ix.p_at.get(&o).map_or(&[][..], |v| &v[..]) {
                    if matches!(pc, Cell::P(_, key) if key.open == inner) {
                        self.close_pair(&mut sink, inside, pc, only(false, true, false), None);
                    }
                }
                if inside.contains_key(&Cell::M2(o, inner)) {
                    self.close_m2_parts(&mut sink, inside, Cell::M2(o, inner), true, false, None);
                }
            }
            Cell::M2(o, inner) => {
                for &pc in ix.p_at.get(&o).map_or(&[][..], |v| &v[..]) {
                    let Cell::P(_, key) = pc else { unreachable!() };
                    let m = Cell::M(key.open, inner);
                    if inside.contains_key(&m) {
                        self.close_pair(
                            &mut sink,
                            inside,
                            pc,
                            only(false, false, true),
                            Some(&[m]),
                        );
                    }
                }
            }
            Cell::P(o, key) => {
                let (Some(ep), Some(eq)) = (key.open_edge(self.dfa), key.close_edge(self.dfa, o))
                else {
                    return out;
                };
                if key.close.pos > key.open.pos + MIN_HAIRPIN_LOOP {
                    self.hairpin(&mut sink, inside, key.open, ep, key.close, eq);
                }
                if inside.contains_key(&Cell::Multi(o, key)) {
                    self.multi_close(&mut sink, inside, Cell::Multi(o, key));
                }
                let outer = Outer {
                    pn: key.open,
                    ep,
                    qn: key.close,
                    eq,
                };
                for pos in (key.open.pos + 1)..=(key.open.pos + 1 + cap) {
                    for &inner in ix.p_by_open.get(&pos).map_or(&[][..], |v| &v[..]) {
                        let Cell::P(o_in, key_in) = inner else {
                            unreachable!()
                        };
                        if o_in.pos <= key.close.pos {
                            self.singles(&mut sink, inside, inner, o_in, key_in, Some(outer));
                        }
                    }
                }
            }
            Cell::Multi(o, key) => {
                let (Some(ep), Some(eq)) = (key.open_edge(self.dfa), key.close_edge(self.dfa, o))
                else {
                    return out;
                };
                let outer = Outer {
                    pn: key.open,
                    ep,
                    qn: key.close,
                    eq,
                };
                // A branching loop's leading run is unbounded.
                for pos in (key.open.pos + 1)..=key.close.pos {
                    let Some(cores) = ix.m2_by_inner.get(&pos) else {
                        continue;
                    };
                    // `closed` rises with the value, so the open cores are a
                    // prefix of the by-value order; they are taken in the
                    // fixed order all the same.
                    let by_value = &ix.m2_by_value[&pos];
                    let open = by_value.partition_point(|&(v, _)| !closed(v));
                    let mut keep: Vec<u32> = by_value[..open].iter().map(|&(_, i)| i).collect();
                    keep.sort_unstable();
                    for i in keep {
                        let m2 = cores[i as usize];
                        let Cell::M2(core_outer, _) = m2 else {
                            unreachable!()
                        };
                        if core_outer.pos <= key.close.pos {
                            self.close_m2_parts(&mut sink, inside, m2, false, true, Some(outer));
                        }
                    }
                }
            }
        }
        out
    }
}

/// The finished inside values and their index, for a walk that generates each
/// cell's instances when it first asks for them.
struct OnDemand {
    inside: FxHashMap<Cell, f64>,
    index: Index,
}

/// No hyperedge: the derivation of a leaf.
const LEAF: u32 = u32::MAX;

/// The edge of a candidate standing for a node's hyperedges whose first
/// candidates are not yet queued ([`Unqueued`]): the least of them, at its
/// value and order.
const QUEUED: u32 = u32::MAX - 1;

/// A node's hyperedges whose first candidates are not yet queued, least
/// first by value and then by place: each edge with its place among the
/// node's edges, which fixes the order its candidate takes. `next` is the
/// first not yet taken; one candidate in the node's heap stands for it.
///
/// Most first candidates are never popped, so a node keeps 8 bytes per edge
/// here rather than a queued [`Cand`] each, and the heap pops what it would
/// have popped had every one been queued.
struct Unqueued {
    edges: Vec<(u32, u32)>,
    first_order: u64,
    next: usize,
}

/// A `(sequence, structure)` pair's 128-bit fingerprint
/// ([`Fingerprint::molecule`]).
type Molecule = u128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Cell(Cell),
    /// The paths from the first node to the second.
    Fill(Node, Node),
}

/// One rule instance with its fills turned into tails.
struct Hyper {
    tails: Tails,
    base: f64,
    /// The `(position, nucleotide)` pairs this instance spells itself, as an
    /// id into the forest's [`SpellPool`].
    spelled: u32,
    /// The base pair this instance closes, when it closes one.
    pair: Option<(u32, u32)>,
    /// The fingerprint of what this instance spells and closes itself.
    key: Fingerprint,
    /// The adaptation of the edges this instance spells itself.
    cai: f64,
}

/// The distinct spellings of a walk's hyperedges, each held once.
///
/// Most instances spell one of a small set of contents, so an edge holds the
/// id of its spelling rather than a vector of its own. An id only names a
/// content; it carries no order.
#[derive(Default)]
struct SpellPool {
    /// Every distinct spelling, back to back.
    flat: Vec<(u32, u8)>,
    /// Spelling `i` is `flat[ends[i - 1]..ends[i]]` (`ends[-1]` is 0).
    ends: Vec<u32>,
    /// The first id with a given content hash.
    first: FxHashMap<u64, u32>,
    /// The next id with the same content hash, `LEAF` when none.
    chain: Vec<u32>,
}

impl SpellPool {
    fn hash(s: &[(u32, u8)]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = crate::hash::FxHasher::default();
        s.hash(&mut h);
        h.finish()
    }

    fn get(&self, id: u32) -> &[(u32, u8)] {
        let i = id as usize;
        let lo = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.flat[lo..self.ends[i] as usize]
    }

    /// The id of `s`, adding it when it is new.
    fn intern(&mut self, s: &[(u32, u8)]) -> u32 {
        let h = Self::hash(s);
        let mut tail = None;
        if let Some(&head) = self.first.get(&h) {
            let mut id = head;
            loop {
                if self.get(id) == s {
                    return id;
                }
                tail = Some(id);
                id = self.chain[id as usize];
                if id == LEAF {
                    break;
                }
            }
        }
        let id = self.ends.len() as u32;
        self.flat.extend_from_slice(s);
        self.ends.push(self.flat.len() as u32);
        self.chain.push(LEAF);
        match tail {
            Some(t) => self.chain[t as usize] = id,
            None => {
                self.first.insert(h, id);
            }
        }
        id
    }

    fn bytes(&self) -> usize {
        vec_bytes(&self.flat)
            + vec_bytes(&self.ends)
            + vec_bytes(&self.chain)
            + table_bytes(
                self.first.len(),
                std::mem::size_of::<u64>() + std::mem::size_of::<u32>(),
            )
    }
}

/// The fingerprint of an instance's own positions and pair.
fn instance_key(spelled: &[(u32, u8)], pair: Option<(u32, u32)>) -> Fingerprint {
    let own = spelled.iter().fold(Fingerprint::default(), |f, &(p, n)| {
        f.with(Fingerprint::spelled(p, n))
    });
    pair.map_or(own, |(i, j)| own.with(Fingerprint::pair(i, j)))
}

/// The most tails a hyperedge has: a rule's cell tail and a loop's two fills.
const MAX_TAILS: usize = 3;

/// Up to [`MAX_TAILS`] values held inline, one per tail of a hyperedge. The
/// walk keeps one per hyperedge ([`Tails`]), one per derivation and one per
/// candidate ([`Idx`]), so a heap vector each would cost more than the rest of
/// the entry. Slots past the hyperedge's tails hold [`UNUSED`], so the entry
/// needs no length.
#[derive(Clone, Copy)]
struct Slots {
    at: [u32; MAX_TAILS],
}

/// A hyperedge's tail nodes.
type Tails = Slots;

/// A derivation's index into each of its hyperedge's tails.
type Idx = Slots;

/// A [`Slots`] entry past the last tail. No node holds `u32::MAX` derivations,
/// and no forest holds `u32::MAX` nodes.
const UNUSED: u32 = u32::MAX;

impl Slots {
    /// No tails.
    fn empty() -> Self {
        Slots {
            at: [UNUSED; MAX_TAILS],
        }
    }

    /// The first derivation of every one of `len` tails.
    fn zeros(len: usize) -> Self {
        assert!(len <= MAX_TAILS, "a hyperedge with {len} tails");
        let mut at = [UNUSED; MAX_TAILS];
        at[..len].fill(0);
        Slots { at }
    }

    /// Append `x` after the last used slot.
    fn push(&mut self, x: u32) {
        let len = self.len();
        assert!(
            len < MAX_TAILS,
            "a hyperedge with more than {MAX_TAILS} tails"
        );
        debug_assert_ne!(x, UNUSED, "a tail at the sentinel");
        self.at[len] = x;
    }

    fn len(&self) -> usize {
        self.at
            .iter()
            .position(|&x| x == UNUSED)
            .unwrap_or(MAX_TAILS)
    }
}

impl std::ops::Deref for Slots {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        &self.at[..self.len()]
    }
}

impl std::ops::DerefMut for Slots {
    fn deref_mut(&mut self) -> &mut [u32] {
        let len = self.len();
        &mut self.at[..len]
    }
}

struct Deriv {
    val: f64,
    edge: u32,
    idx: Idx,
    /// What the derivation's whole tree spells and closes.
    fp: Fingerprint,
}

struct Cand {
    val: f64,
    order: u64,
    edge: u32,
    idx: Idx,
}

// The walk's heap is nearly all derivations and candidates; keep them small.
#[cfg(not(debug_assertions))]
const _: () = assert!(std::mem::size_of::<Deriv>() == 40 && std::mem::size_of::<Cand>() == 32);

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    /// Reversed, so the standard max-heap pops the least value first; ties
    /// break by insertion order, which keeps the walk deterministic.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .val
            .total_cmp(&self.val)
            .then(other.order.cmp(&self.order))
    }
}

#[derive(Default)]
struct Kbest {
    init: bool,
    expanded: usize,
    d: Vec<Deriv>,
    cand: BinaryHeap<Cand>,
    /// The rest, made when the node first records any of it: most nodes of
    /// a forest are tails no derivation ever asks for, and a walk that
    /// keeps none of it makes none.
    x: Option<Box<KbExtra>>,
}

/// What a node keeps beside its derivations.
struct KbExtra {
    /// The subsequences `d` spells ([`Fingerprint::design`]), kept only under
    /// [`Walk::node_dedup`].
    seen: seen::Seen,
    /// Each of `d`'s records on the [`Shelf`], when there is one.
    sids: Vec<u32>,
    /// Each derivation's adaptation, beside `d`, kept only under a
    /// [`window::Bound`].
    cai: Vec<f64>,
}

impl Kbest {
    /// The node's [`KbExtra`]; a node that recorded any of it has one.
    fn x(&self) -> &KbExtra {
        self.x.as_deref().expect("a node with side lists")
    }

    /// The node's [`KbExtra`], made now if the node has none, its dedup set
    /// an index set when `dedup_index`.
    fn x_mut(&mut self, dedup_index: bool) -> &mut KbExtra {
        self.x.get_or_insert_with(|| {
            Box::new(KbExtra {
                seen: seen::Seen::new(dedup_index),
                sids: Vec::new(),
                cai: Vec::new(),
            })
        })
    }

    /// The adaptation of derivation `k`, NaN when none was recorded.
    fn cai_at(&self, k: usize) -> f64 {
        self.x
            .as_deref()
            .and_then(|x| x.cai.get(k).copied())
            .unwrap_or(f64::NAN)
    }
}

/// A kept derivation as a speller on another thread reads it
/// ([`Walk::spellers`]): its node, its hyperedge (`LEAF` for the start)
/// and its tails' derivations, by their own records.
#[derive(Clone, Copy, Default)]
struct SpellNode {
    node: u32,
    edge: u32,
    kids: [u32; MAX_TAILS],
}

/// A hyperedge as a speller reads it: the pair it closes, `NO_PAIR` when it
/// closes none, and its own positions, `spelled[from..from + len]`.
#[derive(Clone, Copy, Default)]
struct EdgeSpell {
    pair: (u32, u32),
    from: u32,
    len: u32,
}

const NO_PAIR: (u32, u32) = (u32::MAX, u32::MAX);

/// What a speller needs of the forest, appended as the walk goes, in records
/// that never move: every hyperedge's spelling, and every kept derivation.
struct Shelf {
    nodes: arena::Writer<SpellNode>,
    edges: arena::Writer<EdgeSpell>,
    spelled: arena::Writer<(u32, u8)>,
}

/// A speller's view of a [`Shelf`].
struct ShelfReader {
    nodes: arena::Reader<SpellNode>,
    edges: arena::Reader<EdgeSpell>,
    spelled: arena::Reader<(u32, u8)>,
}

impl Shelf {
    fn new() -> Self {
        Self {
            nodes: arena::arena().0,
            edges: arena::arena().0,
            spelled: arena::arena().0,
        }
    }

    fn reader(&self) -> ShelfReader {
        ShelfReader {
            nodes: self.nodes.reader(),
            edges: self.edges.reader(),
            spelled: self.spelled.reader(),
        }
    }

    /// Mirror the hyperedge most recently pushed.
    fn edge(&mut self, h: &Hyper, spelled: &[(u32, u8)]) {
        let from = self.spelled.len() as u32;
        for &x in spelled {
            self.spelled.push(x);
        }
        self.edges.push(EdgeSpell {
            pair: h.pair.unwrap_or(NO_PAIR),
            from,
            len: spelled.len() as u32,
        });
    }
}

struct Forest<'g, 'a> {
    grammar: &'g Grammar<'a>,
    ids: FxHashMap<Key, u32>,
    keys: Vec<Key>,
    inside: Vec<f64>,
    incoming: Vec<Vec<u32>>,
    edges: Vec<Hyper>,
    spell_pool: SpellPool,
    kb: Vec<Kbest>,
    /// Per node with more than one hyperedge, those whose first candidates
    /// are not yet queued; dropped once the last is taken or the node closes.
    unqueued: FxHashMap<u32, Unqueued>,
    /// Kept derivations and hyperedges for a speller on another thread
    /// ([`Walk::spellers`]).
    shelf: Option<Shelf>,
    order: u64,
    start: u32,
    /// Set when cells' instances are generated on first request rather than
    /// recorded up front.
    on_demand: Option<OnDemand>,
    /// Seconds of the grammar's sweep, and of lifting recorded instances.
    sweep_s: f64,
    lift_s: f64,
    inside_s: f64,
    index_s: f64,
    /// Cells the sweep finished.
    cells: usize,
    /// Keep one derivation per subsequence at every cell ([`Walk::node_dedup`]).
    node_dedup: bool,
    /// [`Walk::dedup_index`] and [`Walk::tight_lists`].
    dedup_index: bool,
    tight_lists: bool,
    /// Derivations dropped as repeats, by the kind of node that dropped them.
    dropped: DedupDrops,
    /// The root, and when a request to it should give up: under node dedup
    /// one request can drop a long run of repeats before it finds a new
    /// design, so the walk's time cap and [`stop_walks`] are honored there
    /// too. Set when the root gave up rather than ran out.
    root: u32,
    deadline: Option<Instant>,
    /// The [`shed_walks`] count when the walk began: a later shed ends it.
    shed: u64,
    interrupted: bool,
    /// Drop every derivation no completion of which lands in an adaptation
    /// window ([`window`]).
    bound: Option<window::Bound>,
    /// Derivations the bound dropped, and candidates it queued again at
    /// their own value.
    bound_cut: u64,
    bound_requeued: u64,
    /// Fills a cut closed ([`Bounded::Cut`] at a fill).
    bound_fills_closed: u64,
    /// Cells the window's energy bound closed ([`Bounded::Close`]), and
    /// instances it kept from being lifted.
    bound_nodes_closed: u64,
    bound_edges_skipped: u64,
    /// Hyperedges of initialized nodes, counted as each node's incoming list
    /// is read and freed at its initialization.
    instances_seen: usize,
}

/// What a bounded walk does with a candidate it pops.
enum Bounded {
    /// A tail has no derivation at the index named, nor at any higher one.
    Dead,
    /// Its value is above its key: queue it again at the value.
    Later(f64),
    /// No completion lands in the window.
    Cut,
    /// Nor does a completion of any later derivation of the node.
    Close,
    /// Keep it, with its adaptation.
    Keep(f64),
}

impl<'g, 'a> Forest<'g, 'a> {
    fn node(&mut self, key: Key, inside: f64) -> u32 {
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.keys.len() as u32;
        self.ids.insert(key, id);
        self.keys.push(key);
        self.inside.push(inside);
        self.incoming.push(Vec::new());
        self.kb.push(Kbest::default());
        id
    }

    fn fill_node(&mut self, a: Node, b: Node) -> Option<u32> {
        let best = self.grammar.fill(a, b)?;
        Some(self.node(Key::Fill(a, b), -self.grammar.kk * best))
    }

    fn empty(grammar: &'g Grammar<'a>) -> Self {
        Forest {
            grammar,
            ids: FxHashMap::default(),
            keys: Vec::new(),
            inside: Vec::new(),
            incoming: Vec::new(),
            edges: Vec::new(),
            spell_pool: SpellPool::default(),
            kb: Vec::new(),
            unqueued: FxHashMap::default(),
            shelf: None,
            order: 0,
            start: 0,
            on_demand: None,
            sweep_s: 0.0,
            lift_s: 0.0,
            inside_s: 0.0,
            index_s: 0.0,
            cells: 0,
            node_dedup: false,
            dedup_index: false,
            tight_lists: false,
            dropped: DedupDrops::default(),
            root: u32::MAX,
            deadline: None,
            shed: 0,
            interrupted: false,
            bound: None,
            bound_cut: 0,
            bound_requeued: 0,
            bound_fills_closed: 0,
            bound_nodes_closed: 0,
            bound_edges_skipped: 0,
            instances_seen: 0,
        }
    }

    /// Lift one rule instance into a hyperedge, its fills turned into tails, and
    /// return the hyperedge and its head.
    fn lift(&mut self, inst: Inst, inside: &FxHashMap<Cell, f64>) -> (u32, u32) {
        let grammar = self.grammar;
        let mut tails = Tails::empty();
        for c in inst.tails.iter().flatten() {
            let t = self.node(Key::Cell(*c), inside[c]);
            tails.push(t);
        }
        let mut base = inst.weight;
        let mut spelled = Vec::new();
        let mut cai = 0.0;
        for seg in &inst.segs {
            match *seg {
                Seg::Edge(n, e) => {
                    spelled.push((n.pos, e.nuc as u8));
                    cai += e.cai;
                }
                Seg::Fill(a, b) => {
                    if a == b {
                        continue;
                    }
                    let best = grammar.fill(a, b).expect("a scored fill is connectable");
                    base += grammar.kk * best;
                    tails.push(self.fill_node(a, b).expect("connectable"));
                }
            }
        }
        let pair = match inst.head {
            Cell::P(_, key) => Some((key.open.pos, key.close.pos)),
            _ => None,
        };
        let head = self.node(Key::Cell(inst.head), inside[&inst.head]);
        let key = instance_key(&spelled, pair);
        let spelled = self.spell_pool.intern(&spelled);
        let id = self.push_edge(Hyper {
            tails,
            base,
            spelled,
            pair,
            key,
            cai,
        });
        (id, head)
    }

    /// Append a hyperedge, mirrored on the shelf when there is one; its id.
    fn push_edge(&mut self, h: Hyper) -> u32 {
        if let Some(shelf) = self.shelf.as_mut() {
            shelf.edge(&h, self.spell_pool.get(h.spelled));
        }
        self.edges.push(h);
        self.edges.len() as u32 - 1
    }

    /// Keep a shelf from here on ([`Walk::spellers`]): before any
    /// derivation is kept, with the hyperedges recorded so far mirrored.
    fn open_shelf(&mut self) {
        assert!(
            self.kb.iter().all(|k| k.d.is_empty()),
            "a shelf opened after the walk began"
        );
        let mut shelf = Shelf::new();
        for h in &self.edges {
            shelf.edge(h, self.spell_pool.get(h.spelled));
        }
        self.shelf = Some(shelf);
    }

    /// Record the derivation of `v` that was kept, taken through `edge` with tail
    /// indices `idx`, on the shelf when there is one.
    fn shelve(&mut self, v: u32, edge: u32, idx: Idx) {
        let Some(shelf) = self.shelf.as_mut() else {
            return;
        };
        let mut kids = [UNUSED; MAX_TAILS];
        if edge != LEAF {
            for (i, &t) in self.edges[edge as usize].tails.iter().enumerate() {
                kids[i] = self.kb[t as usize].x().sids[idx[i] as usize];
            }
        }
        let sid = shelf.nodes.push(SpellNode {
            node: v,
            edge,
            kids,
        });
        let di = self.dedup_index;
        push_tight(
            self.tight_lists,
            &mut self.kb[v as usize].x_mut(di).sids,
            sid,
        );
    }

    /// Record the grammar's instances up front, fills lifted into tails.
    fn build(grammar: &'g Grammar<'a>) -> Self {
        let t0 = Instant::now();
        let (insts, inside) = grammar.inside();
        let sweep_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut f = Forest::empty(grammar);
        f.sweep_s = sweep_s;
        f.cells = inside.len();
        f.edges.reserve(insts.len());
        f.start = f.node(Key::Cell(Cell::C(grammar.dfa.start())), 0.0);
        for inst in insts {
            let (id, head) = f.lift(inst, &inside);
            f.incoming[head as usize].push(id);
        }
        f.lift_s = t1.elapsed().as_secs_f64();
        f
    }

    /// Only the inside values up front; each cell's instances are generated the
    /// first time the walk asks the cell for a derivation.
    fn build_on_demand(grammar: &'g Grammar<'a>) -> Self {
        let t0 = Instant::now();
        let (_, inside) = grammar.sweep(false);
        Self::from_inside(grammar, inside, t0.elapsed().as_secs_f64())
    }

    /// [`Self::build_on_demand`], with the inside values read from a production
    /// scan's tables rather than the reference grammar's sweep. The two hold
    /// the same cells with the same values; the reference rules still generate
    /// each cell's instances when the walk asks for them.
    fn build_production(grammar: &'g Grammar<'a>, tables: &Tables, scan_s: f64) -> Self {
        let t = Instant::now();
        let inside = production_inside(tables, grammar.lambda);
        let mut f = Self::from_inside(grammar, inside, scan_s);
        f.inside_s = t.elapsed().as_secs_f64() - f.index_s;
        f
    }

    fn from_inside(grammar: &'g Grammar<'a>, inside: FxHashMap<Cell, f64>, sweep_s: f64) -> Self {
        let t = Instant::now();
        let index = Index::new(&inside);
        let index_s = t.elapsed().as_secs_f64();
        let mut f = Forest::empty(grammar);
        f.sweep_s = sweep_s;
        f.index_s = index_s;
        f.cells = inside.len();
        f.start = f.node(Key::Cell(Cell::C(grammar.dfa.start())), 0.0);
        let root_value = inside.get(&Cell::C(grammar.dfa.end())).copied();
        if let Some(v) = root_value {
            f.node(Key::Cell(Cell::C(grammar.dfa.end())), v);
        }
        f.on_demand = Some(OnDemand { inside, index });
        f
    }

    /// A cell's incoming hyperedges, generated from the finished inside values.
    fn expand_cell(&mut self, v: u32) {
        let Key::Cell(cell) = self.keys[v as usize] else {
            return;
        };
        let Some(mut od) = self.on_demand.take() else {
            return;
        };
        let insts = {
            let gate = self.bound.as_mut().and_then(|b| b.gate_at(v, cell));
            let closed = |val: f64| gate.as_ref().is_some_and(|g| g.closes(val));
            self.grammar.pull(&mut od.inside, &od.index, cell, &closed)
        };
        for inst in insts {
            // Every derivation through an instance is valued at least its
            // weight plus its tails' inside values; one the window's energy
            // bound closes at that key is never lifted ([`window::Bound`]).
            if let Some(bound) = self.bound.as_mut() {
                let key = inst.weight
                    + inst
                        .tails
                        .iter()
                        .flatten()
                        .map(|c| od.inside[c])
                        .sum::<f64>();
                if bound.closed_at(v, cell, key) {
                    self.bound_edges_skipped += 1;
                    continue;
                }
            }
            let (id, head) = self.lift(inst, &od.inside);
            debug_assert_eq!(head, v, "a pulled instance has another head");
            self.incoming[head as usize].push(id);
        }
        self.on_demand = Some(od);
    }

    /// A fill node's incoming edges: one first step, then the rest of the path.
    fn expand_fill(&mut self, v: u32) {
        let Key::Fill(a, b) = self.keys[v as usize] else {
            return;
        };
        let steps: Vec<_> = self.grammar.dfa.right(a).to_vec();
        for e in steps {
            if e.to.pos > b.pos {
                continue;
            }
            let mut tails = Tails::empty();
            if e.to != b {
                match self.fill_node(e.to, b) {
                    Some(t) => tails.push(t),
                    None => continue,
                }
            }
            let spelled = self.spell_pool.intern(&[(a.pos, e.nuc as u8)]);
            let id = self.push_edge(Hyper {
                tails,
                base: -self.grammar.kk * e.cai,
                spelled,
                pair: None,
                key: Fingerprint::spelled(a.pos, e.nuc as u8),
                cai: e.cai,
            });
            self.incoming[v as usize].push(id);
        }
    }

    /// Hyperedge `e`'s first candidate's value: its base plus its tails'
    /// inside values, in tail order.
    fn first_value(&self, e: u32) -> f64 {
        let h = &self.edges[e as usize];
        h.base
            + h.tails
                .iter()
                .map(|&t| self.inside[t as usize])
                .sum::<f64>()
    }

    /// Queue `v`'s first candidates: each hyperedge's at once when it has
    /// one, otherwise one candidate standing for the least ([`Unqueued`]),
    /// every edge's order taken now as if each had been pushed in turn.
    fn queue_first(&mut self, v: u32, incoming: &[u32]) {
        if incoming.len() <= 1 {
            for &e in incoming {
                let val = self.first_value(e);
                let idx = Idx::zeros(self.edges[e as usize].tails.len());
                self.push(v, e, idx, val);
            }
            return;
        }
        let first_order = self.order + 1;
        self.order += incoming.len() as u64;
        let mut edges: Vec<(f64, u32, u32)> = incoming
            .iter()
            .enumerate()
            .map(|(i, &e)| (self.first_value(e), i as u32, e))
            .collect();
        edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let (val, i, _) = edges[0];
        self.kb[v as usize].cand.push(Cand {
            val,
            order: first_order + u64::from(i),
            edge: QUEUED,
            idx: Idx::empty(),
        });
        self.unqueued.insert(
            v,
            Unqueued {
                edges: edges.into_iter().map(|(_, i, e)| (e, i)).collect(),
                first_order,
                next: 0,
            },
        );
    }

    /// `v`'s least hyperedge whose first candidate is not yet queued, as that
    /// candidate; the next one's stands in for it in the heap.
    fn next_queued(&mut self, v: u32) -> Cand {
        let u = self.unqueued.get_mut(&v).expect("an unqueued edge");
        let (e, i) = u.edges[u.next];
        let order = u.first_order + u64::from(i);
        u.next += 1;
        match u.edges.get(u.next) {
            Some(&(e2, i2)) => {
                let order2 = u.first_order + u64::from(i2);
                let val = self.first_value(e2);
                self.kb[v as usize].cand.push(Cand {
                    val,
                    order: order2,
                    edge: QUEUED,
                    idx: Idx::empty(),
                });
            }
            None => {
                self.unqueued.remove(&v);
            }
        }
        Cand {
            val: self.first_value(e),
            order,
            edge: e,
            idx: Idx::zeros(self.edges[e as usize].tails.len()),
        }
    }

    /// Empty `v`'s heap, with the hyperedges not yet queued.
    fn close(&mut self, v: u32) {
        self.kb[v as usize].cand.clear();
        self.unqueued.remove(&v);
    }

    fn push(&mut self, v: u32, edge: u32, idx: Idx, val: f64) {
        self.order += 1;
        self.kb[v as usize].cand.push(Cand {
            val,
            order: self.order,
            edge,
            idx,
        });
    }

    fn init(&mut self, v: u32) {
        self.kb[v as usize].init = true;
        if v == self.start {
            push_tight(
                self.tight_lists,
                &mut self.kb[v as usize].d,
                Deriv {
                    val: 0.0,
                    edge: LEAF,
                    idx: Idx::zeros(0),
                    fp: Fingerprint::default(),
                },
            );
            self.shelve(v, LEAF, Idx::zeros(0));
            if self.bound.is_some() {
                let di = self.dedup_index;
                push_tight(
                    self.tight_lists,
                    &mut self.kb[v as usize].x_mut(di).cai,
                    0.0,
                );
            }
            return;
        }
        // A cell the window's energy bound closes at its inside value, the
        // least any derivation of it has, is never expanded.
        if let (Key::Cell(cell), Some(bound)) = (self.keys[v as usize], self.bound.as_mut()) {
            if bound.closed_at(v, cell, self.inside[v as usize]) {
                self.bound_nodes_closed += 1;
                return;
            }
        }
        self.expand_fill(v);
        self.expand_cell(v);
        // Read once here and nowhere after, so the list is freed.
        let incoming = std::mem::take(&mut self.incoming[v as usize]);
        self.instances_seen += incoming.len();
        self.queue_first(v, &incoming);
        self.pop(v);
    }

    /// Take `v`'s next derivation off its heap, and materialize the tail
    /// derivations it names so that it can be spelled. Under
    /// [`Walk::node_dedup`], a cell's derivation that spells a subsequence an
    /// earlier one at the cell spelled is dropped rather than kept, and the
    /// next one taken.
    fn pop(&mut self, v: u32) -> bool {
        loop {
            let Some(c) = self.kb[v as usize].cand.pop() else {
                return false;
            };
            let c = if c.edge == QUEUED {
                self.next_queued(v)
            } else {
                c
            };
            let mut cai = f64::NAN;
            if self.bound.is_some() {
                match self.bounded(v, &c) {
                    // A tail that answers nothing: it ran out, or the walk
                    // gave up on time ([`Self::get`]), and then so does this
                    // node.
                    Bounded::Dead if self.interrupted => return false,
                    Bounded::Dead => continue,
                    Bounded::Later(val) => {
                        self.bound_requeued += 1;
                        self.push(v, c.edge, c.idx, val);
                        // Requeues alone can run without end under loose
                        // values, deriving nothing, so they give up on the
                        // walk's limits as a cut does: `out_of_time` is the
                        // time cap, [`stop_walks`] and [`shed_walks`]
                        // together, so one test covers a memory stop too.
                        if self.out_of_time() {
                            self.interrupted = true;
                            return false;
                        }
                        continue;
                    }
                    Bounded::Close => {
                        self.bound_cut += 1;
                        self.bound_nodes_closed += 1;
                        self.close(v);
                        return false;
                    }
                    Bounded::Cut => {
                        self.bound_cut += 1;
                        // A fill spells no energy, so its paths come out in
                        // descending adaptation: once one is cut, every later
                        // one is, and the fill is done.
                        if matches!(self.keys[v as usize], Key::Fill(..)) {
                            self.bound_fills_closed += 1;
                            self.close(v);
                            return false;
                        }
                        // As for a repeat: its successors are reachable
                        // through it alone, and may land in the window.
                        if c.edge != LEAF {
                            self.successors(v, c.edge, c.idx);
                        }
                        if self.out_of_time() {
                            self.interrupted = true;
                            return false;
                        }
                        continue;
                    }
                    Bounded::Keep(a) => cai = a,
                }
            }
            // A tail that answers nothing means the walk gave up on time
            // while this candidate waited ([`Self::get`]); so does this node.
            let Some(fp) = self.fingerprint(&c) else {
                self.interrupted = true;
                return false;
            };
            if self.is_repeat(v, fp) {
                // Its successors are reachable through it alone: push them
                // now, as `get` would have had it been kept.
                if c.edge != LEAF {
                    self.successors(v, c.edge, c.idx);
                }
                // Any node gives up once time is out: a node that has run out
                // of new subsequences would otherwise pop every remaining
                // derivation it has before answering. A parent indexes into
                // the list left short, so from here every request answers
                // nothing ([`Self::get`]) and the walk ends.
                if self.out_of_time() {
                    self.interrupted = true;
                    return false;
                }
                continue;
            }
            push_tight(
                self.tight_lists,
                &mut self.kb[v as usize].d,
                Deriv {
                    val: c.val,
                    edge: c.edge,
                    idx: c.idx,
                    fp,
                },
            );
            self.shelve(v, c.edge, c.idx);
            if self.bound.is_some() {
                let di = self.dedup_index;
                push_tight(
                    self.tight_lists,
                    &mut self.kb[v as usize].x_mut(di).cai,
                    cai,
                );
                #[cfg(debug_assertions)]
                self.assert_span(v);
            }
            return true;
        }
    }

    /// Whether the walk's time cap has passed, [`stop_walks`] was called, or
    /// [`shed_walks`] was called since the walk began.
    fn out_of_time(&self) -> bool {
        STOP.load(std::sync::atomic::Ordering::Relaxed)
            || shed_since(self.shed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// The lattice span a node's derivations spell.
    fn span(&self, v: u32) -> (Node, Node) {
        match self.keys[v as usize] {
            Key::Fill(a, b) => (a, b),
            Key::Cell(cell) => (span_start(&cell, self.grammar), span_end(&cell)),
        }
    }

    /// That `v`'s newest derivation spells every position of its span and no
    /// other, with the adaptation recorded beside it: the bound reads the
    /// rest of the molecule as everything outside that span.
    #[cfg(debug_assertions)]
    fn assert_span(&self, v: u32) {
        let k = self.kb[v as usize].d.len() - 1;
        let mut seq = vec![u8::MAX; self.grammar.dfa.seq_len as usize];
        let mut pairs = Vec::new();
        self.spell(v, k, &mut seq, &mut pairs);
        let (a, b) = self.span(v);
        for (pos, &nuc) in seq.iter().enumerate() {
            let inside = (a.pos as usize..b.pos as usize).contains(&pos);
            assert_eq!(
                nuc != u8::MAX,
                inside,
                "{:?}: position {pos} against the span {a:?}..{b:?}",
                self.keys[v as usize]
            );
        }
        // A lane is not chosen greedily: the stop family's two lanes share
        // their first nucleotide, and a span may end inside that codon.
        let cai: f64 = self
            .grammar
            .dfa
            .spell_to(a, &seq[a.pos as usize..b.pos as usize], Some(b))
            .unwrap_or_else(|| {
                panic!(
                    "{:?}: no lattice path spells the span and ends at its end {b:?}",
                    self.keys[v as usize]
                )
            })
            .iter()
            .map(|e| e.cai)
            .sum();
        let kept = self.kb[v as usize].x().cai[k];
        assert!(
            (kept - cai).abs() < 1e-9,
            "{:?}: adaptation {kept} recorded, {cai} spelled",
            self.keys[v as usize]
        );
    }

    /// Settle candidate `c` of `v` against the walk's [`window::Bound`],
    /// materializing the tail derivations it names.
    ///
    /// Under the bound a node's list skips the derivations it cuts, so a
    /// tail's first kept derivation can be worth more than its inside value,
    /// and a candidate queued at inside values (by [`Self::init`]) holds a
    /// key below its value. Keys are therefore lower bounds: a candidate
    /// whose value is above its key goes back at its value, and one popped at
    /// its own value is the least left. A successor's key is its value.
    fn bounded(&mut self, v: u32, c: &Cand) -> Bounded {
        if c.edge == LEAF {
            return Bounded::Keep(0.0);
        }
        if let Key::Cell(cell) = self.keys[v as usize] {
            let bound = self.bound.as_mut().expect("a bounded walk");
            if bound.closed_at(v, cell, c.val) {
                return Bounded::Close;
            }
        }
        let tails = self.tails(c.edge);
        let (mut val, mut cai) = {
            let h = &self.edges[c.edge as usize];
            (h.base, h.cai)
        };
        for (i, &t) in tails.iter().enumerate() {
            let k = c.idx[i] as usize;
            if self.get(t, k).is_none() {
                return Bounded::Dead;
            }
            let kb = &self.kb[t as usize];
            val += kb.d[k].val;
            cai += kb.x().cai[k];
        }
        if val > c.val + BOUND_KEY_TOL {
            return Bounded::Later(val);
        }
        let (a, b) = self.span(v);
        let cell = match self.keys[v as usize] {
            Key::Cell(cell) => Some(cell),
            Key::Fill(..) => None,
        };
        let bound = self.bound.as_mut().expect("a bounded walk");
        match bound.admits(v, cell, (a, b), val, cai) {
            window::Admit::Keep => Bounded::Keep(cai),
            window::Admit::Cut => Bounded::Cut,
            window::Admit::Close => Bounded::Close,
        }
    }

    /// Under a window bound, audit the root's `k`-th derivation against the
    /// design `seq` it spells ([`Self::audit_energy`]), whichever thread
    /// later reads the design off.
    #[cfg(debug_assertions)]
    fn audit_design(&mut self, grammar: &Grammar<'_>, root: u32, k: usize, val: f64, seq: &[u8]) {
        if self.bound.is_none() {
            return;
        }
        if let Some(e) = constrained_energy(grammar, val, seq) {
            let cai = self.kb[root as usize].cai_at(k);
            self.audit_energy(root, k, (e, cai));
        }
    }

    /// Run the window's energy audit ([`window::Bound::audit`]) at every
    /// cell derivation in `v`'s `k`-th derivation's tree, against the design
    /// it completes to.
    #[cfg(debug_assertions)]
    fn audit_energy(&mut self, v: u32, k: usize, design: (i32, f64)) {
        let (val, edge, idx) = {
            let d = &self.kb[v as usize].d[k];
            (d.val, d.edge, d.idx)
        };
        let cai = self.kb[v as usize].x().cai[k];
        if let Key::Cell(cell) = self.keys[v as usize] {
            let span = self.span(v);
            if let Some(bound) = self.bound.as_mut() {
                bound.audit(v, cell, span, (val, cai), design);
            }
        }
        if edge == LEAF {
            return;
        }
        let tails = self.tails(edge);
        for (i, &t) in tails.iter().enumerate() {
            self.audit_energy(t, idx[i] as usize, design);
        }
    }

    /// Take [`Walk::dedup_index`] and [`Walk::tight_lists`] before the walk
    /// keeps any derivation: nodes the build made get empty sets of the
    /// kind asked for, and nodes made later get them at creation.
    fn memory_layout(&mut self, dedup_index: bool, tight_lists: bool) {
        self.dedup_index = dedup_index;
        self.tight_lists = tight_lists;
        for kb in &mut self.kb {
            debug_assert!(kb.d.is_empty(), "derivations kept before the walk");
            kb.x = None;
        }
    }

    /// Whether `v` is a cell that already holds a derivation spelling what
    /// `fp` spells, under [`Walk::node_dedup`]; records `fp` otherwise.
    ///
    /// Two derivations of one cell that spell the same subsequence are
    /// interchangeable inside any parent: the parent's loop energies read
    /// only the nucleotides the cell's key fixes, so swapping the later one
    /// for the earlier spells the same molecule at a value no larger. The
    /// later one is therefore never the first derivation of a new design.
    fn is_repeat(&mut self, v: u32, fp: Fingerprint) -> bool {
        if !self.node_dedup {
            return false;
        }
        let Key::Cell(cell) = self.keys[v as usize] else {
            // A fill's derivations are distinct paths, so distinct spellings.
            return false;
        };
        let di = self.dedup_index;
        let kb = &mut self.kb[v as usize];
        kb.x_mut(di);
        let Kbest { x, d, .. } = kb;
        let seen = &mut x.as_deref_mut().expect("made above").seen;
        if seen.insert(fp.design(), d.len() as u32, |i| d[i as usize].fp.design()) {
            return false;
        }
        let root = Key::Cell(Cell::C(self.grammar.dfa.end()));
        if self.keys[v as usize] == root {
            self.dropped.root += 1;
        } else {
            self.dropped.add(&cell);
        }
        true
    }

    /// `edge`'s tails, copied out of the hyperedge, so that the walk can
    /// recurse into them while it mutates the forest.
    fn tails(&self, edge: u32) -> Tails {
        self.edges[edge as usize].tails
    }

    /// What candidate `c`'s whole tree spells and closes, its tail
    /// derivations materialized; `None` when a tail answers nothing, which
    /// happens only once the walk has given up on time.
    fn fingerprint(&mut self, c: &Cand) -> Option<Fingerprint> {
        let mut fp = Fingerprint::default();
        if c.edge != LEAF {
            let tails = self.tails(c.edge);
            fp = self.edges[c.edge as usize].key;
            for (i, &t) in tails.iter().enumerate() {
                let got = self.get(t, c.idx[i] as usize);
                debug_assert!(
                    got.is_some() || self.interrupted,
                    "a candidate names a derivation its tail lacks"
                );
                got?;
                if c.idx[i] == 0 && self.bound.is_none() {
                    debug_assert!(
                        (got.unwrap_or(f64::NAN) - self.inside[t as usize]).abs() < 1e-6,
                        "a tail's best derivation disagrees with its inside value"
                    );
                }
                fp = fp.with(self.kb[t as usize].d[c.idx[i] as usize].fp);
            }
        }
        Some(fp)
    }

    /// The value of `v`'s `k`-th best derivation, computing it if needed.
    fn get(&mut self, v: u32, k: usize) -> Option<f64> {
        if self.interrupted {
            return None;
        }
        if !self.kb[v as usize].init {
            self.init(v);
        }
        while self.kb[v as usize].d.len() <= k {
            let last = self.kb[v as usize].d.len().checked_sub(1)?;
            if self.kb[v as usize].expanded <= last {
                self.kb[v as usize].expanded = last + 1;
                let d = &self.kb[v as usize].d[last];
                let (edge, idx) = (d.edge, d.idx);
                if edge != LEAF {
                    self.successors(v, edge, idx);
                }
            }
            if !self.pop(v) {
                return None;
            }
        }
        // A request a node gave up on may have left a list short below
        // this one; nothing computed past that point is answered.
        if self.interrupted {
            return None;
        }
        Some(self.kb[v as usize].d[k].val)
    }

    /// Push the successors of `edge`'s index vector `idx`: `idx` with one
    /// tail's index raised by one, for each tail at or after the last one
    /// raised. Each vector then has exactly one parent — itself with its last
    /// raised index lowered — so no vector is pushed twice and no record of
    /// the pushed ones is kept, and each is reached through vectors whose
    /// values never decrease, so the order stays exact.
    fn successors(&mut self, v: u32, edge: u32, idx: Idx) {
        let tails = self.tails(edge);
        let base = self.edges[edge as usize].base;
        let from = idx.iter().rposition(|&x| x > 0).unwrap_or(0);
        for i in from..tails.len() {
            let mut j = idx;
            j[i] += 1;
            if self.get(tails[i], j[i] as usize).is_none() {
                continue;
            }
            let mut val = base;
            for (l, &t) in tails.iter().enumerate() {
                val += self.kb[t as usize].d[j[l] as usize].val;
            }
            self.push(v, edge, j, val);
        }
    }

    /// Spell `v`'s `k`-th derivation into `seq`, and collect its pairs.
    fn spell(&self, v: u32, k: usize, seq: &mut [u8], pairs: &mut Vec<(u32, u32)>) {
        let d = &self.kb[v as usize].d[k];
        if d.edge == LEAF {
            return;
        }
        let h = &self.edges[d.edge as usize];
        for &(pos, nuc) in self.spell_pool.get(h.spelled) {
            debug_assert_eq!(seq[pos as usize], u8::MAX, "a position spelled twice");
            seq[pos as usize] = nuc;
        }
        if let Some(p) = h.pair {
            pairs.push(p);
        }
        for (i, &t) in h.tails.iter().enumerate() {
            self.spell(t, d.idx[i] as usize, seq, pairs);
        }
    }
}

/// Push onto one of a node's per-derivation lists, growing it by a quarter
/// rather than doubling when `tight` ([`Walk::tight_lists`]).
fn push_tight<T>(tight: bool, list: &mut Vec<T>, x: T) {
    if tight && list.len() == list.capacity() {
        list.reserve_exact(list.len() / 4 + 4);
    }
    list.push(x);
}

/// How far a candidate's value may sit above its key before a bounded walk
/// queues it again: the rounding of the same sum taken in another order.
const BOUND_KEY_TOL: f64 = 1e-9;

/// The lattice node a cell's derivations start from.
fn span_start(cell: &Cell, grammar: &Grammar<'_>) -> Node {
    match *cell {
        Cell::C(_) => grammar.dfa.start(),
        Cell::P(_, key) | Cell::Multi(_, key) => key.open,
        Cell::M(_, inner) | Cell::M2(_, inner) => inner,
    }
}

/// The lattice node a cell's derivations end at.
fn span_end(cell: &Cell) -> Node {
    match *cell {
        Cell::C(o) | Cell::P(o, _) | Cell::Multi(o, _) | Cell::M(o, _) | Cell::M2(o, _) => o,
    }
}

/// What the last incrementally spelled design left for the next one: its
/// sequence and sorted pairs, and the sub-derivations `(node, rank)` it
/// spelled or copied, each with the positions `[lo, hi)` it fills.
///
/// A sub-derivation always spells the same nucleotides and pairs, so a
/// later design that reuses one can copy its range instead of walking it.
/// A range is recorded only when the sub-derivation fills every position in
/// it, so a copy never touches a position another part of the tree spells;
/// and a pair with both ends in the range was closed inside it, since the
/// hyperedge closing a pair spells both its ends.
#[derive(Default)]
struct Spelled {
    /// The last design spelled, which the next one copies from.
    seq: Vec<u8>,
    pairs: Vec<(u32, u32)>,
    /// The buffers the next design is spelled into, swapped with `seq` and
    /// `pairs` once it is, so spelling allocates nothing per design.
    next_seq: Vec<u8>,
    next_pairs: Vec<(u32, u32)>,
    /// Per node, the rank and range it filled and the design that filled it.
    /// A node covers a fixed span, so it occurs at most once in a tree: one
    /// entry per node suffices, and an entry stamped with the previous design
    /// is that design's.
    spans: Vec<SpanEntry>,
    /// The design being spelled, counting from 1; `0` stamps no design.
    design: u32,
    /// Positions copied rather than spelled, over every design.
    copied: u64,
}

/// A node's sub-derivation in a spelled design: its rank, `[lo, hi)`, and
/// the design ([`Spelled::design`]) it was recorded in.
#[derive(Clone, Copy, Default)]
struct SpanEntry {
    design: u32,
    rank: u32,
    lo: u32,
    hi: u32,
}

impl Spelled {
    /// The range `(v, rank)` filled in the previous design, when it did.
    fn previous(&self, v: u32, rank: u32) -> Option<(u32, u32)> {
        let e = self.spans[v as usize];
        (e.design != 0 && e.design + 1 == self.design && e.rank == rank).then_some((e.lo, e.hi))
    }

    /// Record that `(v, rank)` fills `[lo, hi)` in the design being spelled.
    fn record(&mut self, v: u32, rank: u32, lo: u32, hi: u32) {
        self.spans[v as usize] = SpanEntry {
            design: self.design,
            rank,
            lo,
            hi,
        };
    }
}

/// The positions a sub-derivation filled: `[lo, hi)` and how many.
#[derive(Clone, Copy)]
struct Filled {
    lo: u32,
    hi: u32,
    count: u32,
}

impl Filled {
    const NONE: Self = Self {
        lo: u32::MAX,
        hi: 0,
        count: 0,
    };

    fn add(&mut self, other: Filled) {
        self.lo = self.lo.min(other.lo);
        self.hi = self.hi.max(other.hi);
        self.count += other.count;
    }

    fn contiguous(&self) -> bool {
        self.count > 0 && self.hi - self.lo == self.count
    }
}

impl Forest<'_, '_> {
    /// [`spell_root`], copying from `previous` every range a shared
    /// sub-derivation fills, and leaving this design in `previous.seq` and
    /// `previous.pairs` (sorted), where the next one copies from.
    fn spell_root_reusing(&self, root: u32, k: usize, len: usize, previous: &mut Spelled) {
        let mut seq = std::mem::take(&mut previous.next_seq);
        let mut pairs = std::mem::take(&mut previous.next_pairs);
        seq.clear();
        seq.resize(len, u8::MAX);
        pairs.clear();
        previous.design += 1;
        if previous.spans.len() < self.keys.len() {
            previous.spans.resize(self.keys.len(), SpanEntry::default());
        }
        self.spell_reusing(root, k, &mut seq, &mut pairs, previous);
        assert!(
            seq.iter().all(|&b| b < 4),
            "a derivation left a position unspelled"
        );
        pairs.sort_unstable();
        previous.next_seq = std::mem::replace(&mut previous.seq, seq);
        previous.next_pairs = std::mem::replace(&mut previous.pairs, pairs);
    }

    fn spell_reusing(
        &self,
        v: u32,
        k: usize,
        seq: &mut [u8],
        pairs: &mut Vec<(u32, u32)>,
        previous: &mut Spelled,
    ) -> Filled {
        if let Some((lo, hi)) = previous.previous(v, k as u32) {
            let (a, b) = (lo as usize, hi as usize);
            seq[a..b].copy_from_slice(&previous.seq[a..b]);
            let start = previous.pairs.partition_point(|p| p.0 < lo);
            pairs.extend(
                previous.pairs[start..]
                    .iter()
                    .take_while(|p| p.0 < hi)
                    .filter(|p| p.1 < hi),
            );
            previous.record(v, k as u32, lo, hi);
            previous.copied += u64::from(hi - lo);
            return Filled {
                lo,
                hi,
                count: hi - lo,
            };
        }
        let d = &self.kb[v as usize].d[k];
        if d.edge == LEAF {
            return Filled::NONE;
        }
        let h = &self.edges[d.edge as usize];
        let mut filled = Filled::NONE;
        for &(pos, nuc) in self.spell_pool.get(h.spelled) {
            debug_assert_eq!(seq[pos as usize], u8::MAX, "a position spelled twice");
            seq[pos as usize] = nuc;
            filled.add(Filled {
                lo: pos,
                hi: pos + 1,
                count: 1,
            });
        }
        if let Some(p) = h.pair {
            pairs.push(p);
        }
        for (i, &t) in h.tails.iter().enumerate() {
            filled.add(self.spell_reusing(t, d.idx[i] as usize, seq, pairs, previous));
        }
        if filled.contiguous() {
            previous.record(v, k as u32, filled.lo, filled.hi);
        }
        filled
    }
}

impl ShelfReader {
    /// [`Forest::spell_root_reusing`] for the root derivation shelved as
    /// `sid`, read from the shelf on the speller's own thread.
    fn spell_root_reusing(&mut self, sid: u32, len: usize, previous: &mut Spelled) {
        let mut seq = std::mem::take(&mut previous.next_seq);
        let mut pairs = std::mem::take(&mut previous.next_pairs);
        seq.clear();
        seq.resize(len, u8::MAX);
        pairs.clear();
        previous.design += 1;
        self.spell_reusing(sid, &mut seq, &mut pairs, previous);
        assert!(
            seq.iter().all(|&b| b < 4),
            "a derivation left a position unspelled"
        );
        pairs.sort_unstable();
        previous.next_seq = std::mem::replace(&mut previous.seq, seq);
        previous.next_pairs = std::mem::replace(&mut previous.pairs, pairs);
    }

    /// [`Forest::spell_reusing`] over shelved records: a derivation is named
    /// by its record, which is as particular to it as its node and rank.
    fn spell_reusing(
        &mut self,
        sid: u32,
        seq: &mut [u8],
        pairs: &mut Vec<(u32, u32)>,
        previous: &mut Spelled,
    ) -> Filled {
        let d = self.nodes.get(sid);
        let v = d.node;
        if previous.spans.len() <= v as usize {
            let grown = v as usize + 1 + (v as usize >> 2);
            previous.spans.resize(grown, SpanEntry::default());
        }
        if let Some((lo, hi)) = previous.previous(v, sid) {
            let (a, b) = (lo as usize, hi as usize);
            seq[a..b].copy_from_slice(&previous.seq[a..b]);
            let start = previous.pairs.partition_point(|p| p.0 < lo);
            pairs.extend(
                previous.pairs[start..]
                    .iter()
                    .take_while(|p| p.0 < hi)
                    .filter(|p| p.1 < hi),
            );
            previous.record(v, sid, lo, hi);
            previous.copied += u64::from(hi - lo);
            return Filled {
                lo,
                hi,
                count: hi - lo,
            };
        }
        if d.edge == LEAF {
            return Filled::NONE;
        }
        let h = self.edges.get(d.edge);
        let mut filled = Filled::NONE;
        for i in h.from..h.from + h.len {
            let (pos, nuc) = self.spelled.get(i);
            debug_assert_eq!(seq[pos as usize], u8::MAX, "a position spelled twice");
            seq[pos as usize] = nuc;
            filled.add(Filled {
                lo: pos,
                hi: pos + 1,
                count: 1,
            });
        }
        if h.pair != NO_PAIR {
            pairs.push(h.pair);
        }
        for &kid in d.kids.iter().take_while(|&&k| k != UNUSED) {
            filled.add(self.spell_reusing(kid, seq, pairs, previous));
        }
        if filled.contiguous() {
            previous.record(v, sid, filled.lo, filled.hi);
        }
        filled
    }
}

/// The `k`-th derivation of `root`: its sequence (nucleotide codes) over `len`
/// positions and its sorted pairs.
fn spell_root(
    forest: &Forest<'_, '_>,
    root: u32,
    k: usize,
    len: usize,
) -> (Vec<u8>, Vec<(u32, u32)>) {
    let mut seq = vec![u8::MAX; len];
    let mut pairs = Vec::new();
    forest.spell(root, k, &mut seq, &mut pairs);
    assert!(
        seq.iter().all(|&b| b < 4),
        "a derivation left a position unspelled"
    );
    pairs.sort_unstable();
    (seq, pairs)
}

/// A design's constrained MFE in dekacalories, read off its first derivation.
///
/// A sequence's first derivation is its best structure under the constraint, and its
/// value is `λ·E − (1−λ)·100·Σ cai` over the sequence's own lattice path (a
/// soft constraint's penalties included in `E`, as the constrained fold includes
/// them). The path's adaptation is fixed by the sequence, so `E` follows. `None`
/// at `λ = 0`, where the value carries no energy, or when the result is not an
/// integer, which would mean the value is not what this assumes.
/// The text and constrained MFE of each new design under
/// [`Walk::incremental_spell`], recomputed only where it differs from the last
/// one. Consecutive designs differ in a short stretch: past its end, the
/// automaton path rejoins the last design's at the first node they share, and
/// everything after is unchanged. The CAI total is kept as a running sum,
/// re-added in order every [`Readoff::RESUM`] designs so its rounding cannot
/// drift; the energy it yields is an integer, read within `1e-4`.
#[derive(Default)]
struct Readoff {
    /// The last design, its text, the automaton node before each position
    /// (one more than positions), each position's CAI, and their sum.
    seq: Vec<u8>,
    text: Vec<u8>,
    node: Vec<Node>,
    cai: Vec<f64>,
    total: f64,
    /// Whether the arrays describe the last design in full.
    complete: bool,
    updates: u64,
}

impl Readoff {
    const RESUM: u64 = 4096;

    /// The text and [`constrained_energy`] of `seq`, the design of value `val`.
    fn update(&mut self, grammar: &Grammar<'_>, val: f64, seq: &[u8]) -> (&str, Option<i32>) {
        let n = seq.len();
        if !(self.complete && self.seq.len() == n) {
            self.seq.clear();
            self.seq.resize(n, u8::MAX);
            self.text.clear();
            self.text.resize(n, b'N');
            self.node.clear();
            self.node.resize(n + 1, grammar.dfa.start());
            self.cai.clear();
            self.cai.resize(n, 0.0);
            self.total = 0.0;
        }
        self.complete = false;
        // Walk each stretch from its first changed position until the path
        // has rejoined the last design's at a position that is unchanged:
        // from there to the next change both read the same nucleotides from
        // the same node, so nothing differs and the stretch is skipped.
        let mut i = first_difference(seq, &self.seq, 0);
        while i < n {
            // Back up to the unit's start before walking. A lane can diverge
            // one position *before* the first differing nucleotide — the stop
            // family's lanes are `UA` and `UG`, which share a `U` — so the last
            // design's node at `i` need not lie on this design's path at all.
            // At a unit boundary the node is the lane-0 one whatever the design.
            //
            // `differs` is where the walk must reach before it may rejoin: the
            // backed-up positions agree with the last design, so a rejoin test
            // allowed to pass before it would stop short, back up to the same
            // unit again and never advance.
            let differs = i;
            i = grammar.dfa.codon_start(i as u32) as usize;
            let mut at = Node::new(i as u32, 0);
            loop {
                let b = seq[i];
                let Some(e) = grammar.dfa.step_on_path(at, seq, i) else {
                    return (self.text(), None);
                };
                self.seq[i] = b;
                self.text[i] = b"ACGU"[b as usize];
                self.total += e.cai - self.cai[i];
                self.cai[i] = e.cai;
                let rejoined = self.node[i + 1] == e.to;
                self.node[i + 1] = e.to;
                at = e.to;
                i += 1;
                if i == n || (i > differs && rejoined && seq[i] == self.seq[i]) {
                    break;
                }
            }
            i = first_difference(seq, &self.seq, i);
        }
        self.complete = true;
        self.updates += 1;
        if self.updates.is_multiple_of(Self::RESUM) {
            self.total = self.cai.iter().sum();
        }
        if grammar.lambda <= 0.0 {
            return (self.text(), None);
        }
        let energy = (val + grammar.kk * self.total) / grammar.lambda;
        let rounded = energy.round();
        let energy = ((energy - rounded).abs() < 1e-4).then_some(rounded as i32);
        (self.text(), energy)
    }

    fn text(&self) -> &str {
        std::str::from_utf8(&self.text).expect("a design's text is ACGU")
    }
}

/// The first position at or after `from` where `a` and `b` differ, or their
/// length; compared a word at a time.
fn first_difference(a: &[u8], b: &[u8], from: usize) -> usize {
    debug_assert_eq!(a.len(), b.len());
    let mut i = from;
    while i + 8 <= a.len() {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().expect("eight bytes"))
            ^ u64::from_le_bytes(b[i..i + 8].try_into().expect("eight bytes"));
        if x != 0 {
            return i + (x.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < a.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

fn constrained_energy(grammar: &Grammar<'_>, val: f64, seq: &[u8]) -> Option<i32> {
    if grammar.lambda <= 0.0 {
        return None;
    }
    let mut at = grammar.dfa.start();
    let mut cai = 0.0;
    for i in 0..seq.len() {
        let e = grammar.dfa.step_on_path(at, seq, i)?;
        cai += e.cai;
        at = e.to;
    }
    let energy = (val + grammar.kk * cai) / grammar.lambda;
    let rounded = energy.round();
    ((energy - rounded).abs() < 1e-4).then_some(rounded as i32)
}

/// How the walk's forest is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Build {
    /// Every rule instance recorded by the reference sweep before the walk.
    Recorded,
    /// Inside values from the reference sweep; each cell's instances generated
    /// the first time the walk asks it for a derivation.
    OnDemand,
    /// Inside values from one exact production scan; instances on demand.
    Production,
}

/// How a walk is built and what it keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Walk {
    pub build: Build,
    /// Keep, at every cell, only the best derivation of each subsequence it
    /// spells. The walk visits the same designs in the same order, each at
    /// its best constrained structure, and keeps and expands fewer derivations;
    /// a dropped derivation's successors are still pushed.
    pub node_dedup: bool,
    /// Spell each new design by copying, from the previous design, the
    /// position ranges of the sub-derivations the two share
    /// ([`Spelled`]). The same designs, spelled the same way.
    pub incremental_spell: bool,
    /// Read each new design off and hand it to the visitor on a thread of its
    /// own, in walk order, while the walk spells the next ones. The visitor
    /// sees the same designs in the same order; once it returns `true` it is
    /// called no more, though the walk may have taken designs past that one:
    /// up to about [`PIPE_DEPTH`] here, and `(2 * PIPE_DEPTH + 1)` per
    /// speller under [`Self::spellers`].
    pub pipelined_visit: bool,
    /// Spell new designs on this many threads of their own, in turn, from
    /// records the walk keeps of every derivation it keeps (a shelf), and
    /// hand them on to the visitor's thread, which takes them back in walk
    /// order: the walk thread only takes derivations. `0` spells on the walk
    /// thread. Implies [`Self::pipelined_visit`]; each speller spells a
    /// design from its own last one whatever [`Self::incremental_spell`]
    /// says. The visitor sees the same designs in the same order.
    pub spellers: usize,
    /// Keep each node's dedup set as bare indices into its own derivations
    /// ([`seen::IndexSet`]) rather than 16-byte fingerprints: a slot is a
    /// 4-byte index and a 1-byte tag of the key's hash, and a probe reads a
    /// key back out of the derivation list only where a tag agrees. The same
    /// designs in the same order, in less memory.
    pub dedup_index: bool,
    /// Grow each node's per-derivation lists by a quarter rather than
    /// doubling: the walk keeps hundreds of millions of derivations across
    /// hundreds of thousands of lists, and doubling left 42% of their bytes
    /// unused (P0DTC2 n = 500 actb, 600 s).
    pub tight_lists: bool,
}

impl From<Build> for Walk {
    fn from(build: Build) -> Self {
        Self {
            build,
            node_dedup: false,
            incremental_spell: false,
            pipelined_visit: false,
            spellers: 0,
            dedup_index: false,
            tight_lists: false,
        }
    }
}

/// Stack for a speller's or a piped visitor's thread. Spelling recurses
/// once per derivation level, and derivations nest as deep as the molecule
/// is long, so the default 2 MiB is not enough at the target lengths; the
/// space is reserved, not committed.
const DEEP_STACK: usize = 1 << 30;

/// Spawn `f` on a scoped thread with [`DEEP_STACK`].
fn spawn_deep<'scope, 'env, T: Send + 'scope>(
    s: &'scope std::thread::Scope<'scope, 'env>,
    f: impl FnOnce() -> T + Send + 'scope,
) -> std::thread::ScopedJoinHandle<'scope, T> {
    let deep = std::thread::Builder::new().stack_size(DEEP_STACK);
    crate::task_clock::spawn_built(deep, s, f).expect("spawn a walk thread")
}

/// Designs a [`Walk::pipelined_visit`] walk may spell ahead of its visitor.
pub const PIPE_DEPTH: usize = 64;

/// A spelled design on its way to the visitor's thread.
struct Piece {
    objective: f64,
    val: f64,
    /// The walk's design and derivation counts with this design taken.
    designs: u64,
    derivations: u64,
    seq: Vec<u8>,
    pairs: Vec<(u32, u32)>,
    /// The design's adaptation and the derivations a window's bound had cut
    /// when the walk took it (`NaN` and `0` without a window).
    cai: f64,
    bound_cut: u64,
}

/// The visitor's half of a walk: a new design's text and constrained MFE,
/// read off, then handed to the visitor.
struct Deliver<'g, 'a, F> {
    grammar: &'g Grammar<'a>,
    incremental: bool,
    readoff: Readoff,
    text: String,
    visit: F,
}

impl<F> Deliver<'_, '_, F>
where
    F: FnMut(f64, &str, Option<i32>, &[(u32, u32)], f64, u64) -> Result<bool, LazyError>,
{
    fn deliver(
        &mut self,
        objective: f64,
        val: f64,
        seq: &[u8],
        pairs: &[(u32, u32)],
        (cai, bound_cut): (f64, u64),
    ) -> Result<bool, LazyError> {
        if self.incremental {
            let (text, energy) = self.readoff.update(self.grammar, val, seq);
            debug_assert_eq!(
                energy,
                constrained_energy(self.grammar, val, seq),
                "an incremental read-off differs from a full one"
            );
            debug_assert!(
                text.bytes()
                    .zip(seq)
                    .all(|(t, &b)| t == b"ACGU"[b as usize]),
                "an incremental text differs from the design"
            );
            (self.visit)(objective, text, energy, pairs, cai, bound_cut)
        } else {
            self.text.clear();
            self.text
                .extend(seq.iter().map(|&b| b"ACGU"[b as usize] as char));
            let energy = constrained_energy(self.grammar, val, seq);
            (self.visit)(objective, &self.text, energy, pairs, cai, bound_cut)
        }
    }
}

/// A new design, unspelled, on its way to the speller's thread
/// ([`Walk::spellers`]): its root derivation's shelf record.
struct Job {
    sid: u32,
    objective: f64,
    val: f64,
    designs: u64,
    derivations: u64,
    cai: f64,
    bound_cut: u64,
    /// The design spelled from the forest, which the shelf's must match.
    #[cfg(debug_assertions)]
    expect: (Vec<u8>, Vec<(u32, u32)>),
}

/// Where the walk sends each new design: straight to the visitor, down a
/// bounded queue to the visitor's thread ([`Walk::pipelined_visit`]), or
/// unspelled to the speller's thread ([`Walk::spellers`]).
enum Outlet<'s, 'g, 'a, F> {
    Direct(Deliver<'g, 'a, F>),
    Piped {
        to: std::sync::mpsc::SyncSender<Piece>,
        free: std::sync::mpsc::Receiver<Piece>,
        stop: &'s std::sync::atomic::AtomicBool,
    },
    /// One queue per speller; the walk's `d`-th design goes to queue
    /// `(d - 1) % spellers`.
    Spell {
        to: Vec<std::sync::mpsc::SyncSender<Job>>,
        stop: &'s std::sync::atomic::AtomicBool,
    },
}

/// The speller's thread of a [`Walk::spellers`] walk: spell each job
/// from the shelf, in order, each from the last, and pass it to the visitor's
/// thread; once the visitor has stopped, drain the queue. Returns the
/// positions copied rather than spelled.
fn spell_jobs<F>(
    from: std::sync::mpsc::Receiver<Job>,
    mut shelf: ShelfReader,
    len: usize,
    mut pieces: Outlet<'_, '_, '_, F>,
    stop: &std::sync::atomic::AtomicBool,
) -> u64
where
    F: FnMut(f64, &str, Option<i32>, &[(u32, u32)], f64, u64) -> Result<bool, LazyError>,
{
    let mut previous = Spelled::default();
    for job in from {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            continue;
        }
        shelf.spell_root_reusing(job.sid, len, &mut previous);
        #[cfg(debug_assertions)]
        assert_eq!(
            (&previous.seq, &previous.pairs),
            (&job.expect.0, &job.expect.1),
            "a design spelled from the shelf differs from the forest's"
        );
        // `true` when the visitor has stopped, or its thread is gone (a
        // panic, which reaches the caller at the join): stop the walk too.
        if pieces
            .take(
                job.objective,
                job.val,
                (job.designs, job.derivations),
                &previous.seq,
                &previous.pairs,
                (job.cai, job.bound_cut),
            )
            .unwrap_or(true)
        {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    previous.copied
}

impl<F> Outlet<'_, '_, '_, F>
where
    F: FnMut(f64, &str, Option<i32>, &[(u32, u32)], f64, u64) -> Result<bool, LazyError>,
{
    /// Hand on one design, the walk's `designs`-th after `derivations` root
    /// derivations; `true` once the visitor has asked to stop.
    fn take(
        &mut self,
        objective: f64,
        val: f64,
        (designs, derivations): (u64, u64),
        seq: &[u8],
        pairs: &[(u32, u32)],
        window: (f64, u64),
    ) -> Result<bool, LazyError> {
        match self {
            Outlet::Direct(d) => d.deliver(objective, val, seq, pairs, window),
            Outlet::Piped { to, free, stop } => {
                let mut piece = free.try_recv().unwrap_or_else(|_| Piece {
                    objective: 0.0,
                    val: 0.0,
                    designs: 0,
                    derivations: 0,
                    seq: Vec::with_capacity(seq.len()),
                    pairs: Vec::with_capacity(pairs.len()),
                    cai: f64::NAN,
                    bound_cut: 0,
                });
                (piece.cai, piece.bound_cut) = window;
                piece.objective = objective;
                piece.val = val;
                piece.designs = designs;
                piece.derivations = derivations;
                piece.seq.clear();
                piece.seq.extend_from_slice(seq);
                piece.pairs.clear();
                piece.pairs.extend_from_slice(pairs);
                // The visitor's thread ends only once it has been told to stop
                // or the queue closes, so a failed send means it stopped.
                if to.send(piece).is_err() {
                    return Ok(true);
                }
                Ok(stop.load(std::sync::atomic::Ordering::Relaxed))
            }
            Outlet::Spell { .. } => unreachable!("a speller's outlet takes jobs, not designs"),
        }
    }
}

/// The visitor's thread of a [`Walk::pipelined_visit`] walk: deliver each
/// piece in order until the visitor stops or fails, then drain the queues.
/// Pieces arrive on `lanes` in turn (one per speller, or the walk's one), so
/// reading the lanes in turn restores walk order, and the first lane found
/// closed and empty is the end of the walk.
/// Returns the walk's counts and objective at the last design the visitor
/// stopped on, when it stopped: designs the walk spelled past it were never
/// visited.
fn visit_pieces<F>(
    mut deliver: Deliver<'_, '_, F>,
    lanes: Vec<(
        std::sync::mpsc::Receiver<Piece>,
        std::sync::mpsc::Sender<Piece>,
    )>,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<Option<(u64, u64, f64)>, LazyError>
where
    F: FnMut(f64, &str, Option<i32>, &[(u32, u32)], f64, u64) -> Result<bool, LazyError>,
{
    let mut failed = None;
    let mut stopped_at = None;
    for (from, free) in lanes.iter().cycle() {
        let Ok(piece) = from.recv() else {
            break;
        };
        if failed.is_none() && stopped_at.is_none() {
            match deliver.deliver(
                piece.objective,
                piece.val,
                &piece.seq,
                &piece.pairs,
                (piece.cai, piece.bound_cut),
            ) {
                Ok(false) => {}
                Ok(true) => {
                    stopped_at = Some((piece.designs, piece.derivations, piece.objective));
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                Err(e) => {
                    failed = Some(e);
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        let _ = free.send(piece);
    }
    failed.map_or(Ok(stopped_at), Err)
}

/// Derivations a [`Walk::node_dedup`] walk dropped as repeats of a
/// subsequence its node already held, by node kind. `root` is the molecule's
/// cell (`C` at the last node), where a repeat is a whole design.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DedupDrops {
    pub c: u64,
    pub p: u64,
    pub multi: u64,
    pub m: u64,
    pub m2: u64,
    pub root: u64,
}

impl DedupDrops {
    fn add(&mut self, cell: &Cell) {
        *[
            &mut self.c,
            &mut self.p,
            &mut self.multi,
            &mut self.m,
            &mut self.m2,
        ][kind_of(cell)] += 1;
    }

    /// Every drop, root included.
    pub fn total(&self) -> u64 {
        self.c + self.p + self.multi + self.m + self.m2 + self.root
    }
}

/// What a walk over the root's derivations saw and cost.
#[derive(Debug, Clone)]
pub struct WalkStats {
    /// Seconds recording the grammar and lifting its fills.
    pub build_s: f64,
    /// Nodes and hyperedges when the walk ended (fills grow both on demand).
    pub nodes: usize,
    pub edges: usize,
    /// The root's best derivation, on a design's reported scale.
    pub optimum: f64,
    /// The last derivation taken off the root, on the same scale. Every design
    /// not yet visited has an objective at least this, so when no design
    /// visited qualifies it is a lower bound on the best one that does.
    pub last_value: f64,
    /// Derivations taken off the root, in order.
    pub derivations: u64,
    /// Distinct sequences among them — the designs, in objective order.
    pub designs: u64,
    /// Derivations whose `(sequence, structure)` had been seen already: the
    /// grammar spelling one molecule two ways. Counted only in builds with
    /// debug assertions (see [`fingerprint`]); `None` elsewhere.
    pub duplicate_molecules: Option<u64>,
    /// The most derivations any one sequence took before the walk ended.
    pub max_per_design: u64,
    /// Seconds walking, visits included.
    pub walk_s: f64,
    /// Why the walk ended: `visitor`, `exhausted`, `derivations`, `seconds`,
    /// `requested` ([`stop_walks`]) or `shed` ([`shed_walks`]) (and
    /// `incumbent` from [`first_modular`]).
    pub stopped: &'static str,
    /// Of the build: seconds computing the inside values (the reference sweep,
    /// or the production scan), and in lifting recorded instances (zero on
    /// demand).
    pub sweep_s: f64,
    pub lift_s: f64,
    /// Of the build past the sweep: seconds reading a production scan's tables
    /// into the cell map (`inside_s`, zero otherwise) and indexing the cells
    /// for the pulls (`index_s`).
    pub inside_s: f64,
    pub index_s: f64,
    /// Cells the sweep finished, and how many of them the walk asked for more
    /// than their best — the cells whose instances it read.
    pub cells: usize,
    pub touched_cells: usize,
    /// Fill nodes the walk expanded, and the instances enumerated at every
    /// node it touched.
    pub touched_fills: usize,
    pub touched_instances: usize,
    /// Heap the walk held when it ended, estimated per structure.
    pub memory: MemoryBreakdown,
    /// Under [`Walk::incremental_spell`], positions of new designs copied
    /// from the previous design rather than spelled.
    pub positions_copied: u64,
    /// Under [`Walk::node_dedup`], the derivations dropped as repeats; `None`
    /// otherwise. With it, every root derivation is a new design.
    pub node_dedup: Option<DedupDrops>,
    /// Of `walk_s`: seconds taking the next derivation off the root, spelling
    /// it, keeping the duplicate counts, and in `visit`.
    pub phases: WalkPhases,
    /// Under an adaptation window ([`window`]): derivations its bound cut,
    /// and candidates queued again at their own value.
    pub bound_cut: u64,
    pub bound_requeued: u64,
    pub bound_fills_closed: u64,
    /// Of `bound_cut`: derivations cut by the window's energy bound
    /// ([`window::Bound`]), every in-window completion dominated by a find.
    pub bound_energy_cut: u64,
    /// Of `bound_cut`: derivations whose every completion falls outside the
    /// window, and derivations whose every in-window completion falls in an
    /// adaptation span the finds already cover.
    pub bound_outside_cut: u64,
    pub bound_covered_cut: u64,
    /// Derivations that reached the bound's outside-the-window test, its
    /// covered-span test and its energy test: the populations the three cut
    /// counts are taken from.
    pub bound_outside_tested: u64,
    pub bound_covered_tested: u64,
    pub bound_energy_tested: u64,
    /// Cells the energy bound closed: no later derivation of theirs can
    /// complete to a point the window needs.
    pub bound_nodes_closed: u64,
    /// Rule instances the energy bound kept from being lifted: every
    /// derivation through them would have been closed.
    pub bound_edges_skipped: u64,
    /// Per slope of the energy bound, the cuts whose bound it gave.
    pub bound_energy_binding: Vec<u64>,
}

impl WalkStats {
    /// A walk that took no derivation and ended for `stopped`, after
    /// `build_s` seconds of building.
    pub(crate) fn unwalked(build_s: f64, stopped: &'static str) -> Self {
        Self {
            build_s,
            nodes: 0,
            edges: 0,
            optimum: f64::NAN,
            last_value: f64::NAN,
            derivations: 0,
            designs: 0,
            duplicate_molecules: cfg!(debug_assertions).then_some(0),
            max_per_design: 0,
            walk_s: 0.0,
            stopped,
            sweep_s: 0.0,
            lift_s: 0.0,
            inside_s: 0.0,
            index_s: 0.0,
            cells: 0,
            touched_cells: 0,
            touched_fills: 0,
            touched_instances: 0,
            memory: MemoryBreakdown::default(),
            node_dedup: None,
            positions_copied: 0,
            phases: WalkPhases::default(),
            bound_cut: 0,
            bound_requeued: 0,
            bound_fills_closed: 0,
            bound_energy_cut: 0,
            bound_outside_cut: 0,
            bound_covered_cut: 0,
            bound_outside_tested: 0,
            bound_covered_tested: 0,
            bound_energy_tested: 0,
            bound_nodes_closed: 0,
            bound_edges_skipped: 0,
            bound_energy_binding: Vec::new(),
        }
    }
}

/// Where a walk's seconds went, phase by phase.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WalkPhases {
    pub next_s: f64,
    pub spell_s: f64,
    pub dedup_s: f64,
    pub visit_s: f64,
}

/// The walk's heap by structure, estimated from lengths and element sizes
/// (allocator slack and hash-table control bytes approximated), in bytes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryBreakdown {
    /// Derivations kept per node, with a window's adaptations and each
    /// derivation's shelf index (the shelf itself is [`Self::shelf`]).
    pub derivations: usize,
    /// Candidate heaps per node, with the hyperedges whose first candidates
    /// are not yet queued.
    pub candidates: usize,
    /// Hyperedges (their tails held inline), the pool of distinct spellings,
    /// and the incoming lists.
    pub edges: usize,
    /// Node keys, their id map and inside values, each node's entry, and
    /// the side lists' boxes of the nodes that made one.
    pub nodes: usize,
    /// Every `(sequence, structure)` fingerprint seen, for the duplicate count
    /// (debug assertions only).
    pub molecules: usize,
    /// Every sequence seen, with its derivation count.
    pub designs: usize,
    /// The subsequences each node kept ([`Walk::node_dedup`]).
    pub node_sets: usize,
    /// Derivations held below the root (a count, not bytes), and of them the
    /// ones some kept derivation of a parent uses, and the ones only a queued
    /// candidate names. The rest were requested by candidates the bound cut or
    /// closed: held, and read by nothing the walk still has.
    pub held: usize,
    pub held_used: usize,
    pub held_queued: usize,
    /// `derivations` and `node_sets` by node kind: `C`, `P`, `Multi`, `M`,
    /// `M2`, then fills.
    pub derivations_by_kind: [usize; 6],
    pub node_sets_by_kind: [usize; 6],
    /// `derivations` counted by length rather than capacity: the rest is the
    /// vectors' growth slack.
    pub derivations_used: usize,
    /// The spellers' shelf ([`Walk::spellers`]): derivation records,
    /// hyperedge records and the copied spellings, by allocated chunk.
    pub shelf: usize,
}

/// Derivations held below the root, those a kept parent derivation uses, and
/// those only a queued candidate names (see [`MemoryBreakdown::held`]).
fn held_use(forest: &Forest<'_, '_>) -> (usize, usize, usize) {
    let bits = |kb: &Kbest| vec![0u64; kb.d.len().div_ceil(64)];
    let mut used: Vec<Vec<u64>> = forest.kb.iter().map(bits).collect();
    let mut queued: Vec<Vec<u64>> = forest.kb.iter().map(bits).collect();
    let mark = |set: &mut Vec<Vec<u64>>, edge: u32, idx: &Idx| {
        if edge == LEAF || edge == QUEUED {
            return;
        }
        for (&t, &k) in forest.edges[edge as usize].tails.iter().zip(idx.iter()) {
            if let Some(w) = set[t as usize].get_mut(k as usize / 64) {
                *w |= 1 << (k % 64);
            }
        }
    };
    for kb in &forest.kb {
        for d in &kb.d {
            mark(&mut used, d.edge, &d.idx);
        }
        for c in kb.cand.iter() {
            mark(&mut queued, c.edge, &c.idx);
        }
    }
    // A hyperedge not yet queued names its tails' first derivations, as its
    // queued candidate would.
    for u in forest.unqueued.values() {
        for &(e, _) in &u.edges[u.next..] {
            let idx = Idx::zeros(forest.edges[e as usize].tails.len());
            mark(&mut queued, e, &idx);
        }
    }
    let (mut held, mut held_used, mut held_queued) = (0, 0, 0);
    for (v, kb) in forest.kb.iter().enumerate() {
        if v as u32 == forest.start {
            continue;
        }
        held += kb.d.len();
        for (u, q) in used[v].iter().zip(&queued[v]) {
            held_used += u.count_ones() as usize;
            held_queued += (q & !u).count_ones() as usize;
        }
    }
    (held, held_used, held_queued)
}

/// Bytes of a vector's heap buffer.
fn vec_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity() * std::mem::size_of::<T>()
}

/// Bytes of a hash table holding `len` entries of `entry` bytes each, at the
/// swiss table's 7/8 load and one control byte per slot.
fn table_bytes(len: usize, entry: usize) -> usize {
    (len * 8 / 7 + 1) * (entry + 1)
}

impl MemoryBreakdown {
    fn measure(
        forest: &Forest<'_, '_>,
        molecules: &FxHashSet<Molecule>,
        per_design: &FxHashMap<u128, u64>,
    ) -> Self {
        let mut m = MemoryBreakdown::default();
        for (kb, key) in forest.kb.iter().zip(&forest.keys) {
            let kind = match key {
                Key::Cell(cell) => kind_of(cell),
                Key::Fill(..) => 5,
            };
            let (side, side_used) = kb.x.as_deref().map_or((0, 0), |x| {
                (
                    vec_bytes(&x.cai) + vec_bytes(&x.sids),
                    x.cai.len() * std::mem::size_of::<f64>()
                        + x.sids.len() * std::mem::size_of::<u32>(),
                )
            });
            let d = vec_bytes(&kb.d) + side;
            m.derivations += d;
            m.derivations_by_kind[kind] += d;
            m.derivations_used += kb.d.len() * std::mem::size_of::<Deriv>() + side_used;
            m.candidates += kb.cand.capacity() * std::mem::size_of::<Cand>();
            let t = kb.x.as_deref().map_or(0, |x| x.seen.bytes());
            m.node_sets += t;
            m.node_sets_by_kind[kind] += t;
        }
        m.candidates += table_bytes(
            forest.unqueued.len(),
            std::mem::size_of::<u32>() + std::mem::size_of::<Unqueued>(),
        ) + forest
            .unqueued
            .values()
            .map(|u| vec_bytes(&u.edges))
            .sum::<usize>();
        if let Some(shelf) = &forest.shelf {
            m.shelf = shelf.nodes.bytes() + shelf.edges.bytes() + shelf.spelled.bytes();
        }
        m.edges = vec_bytes(&forest.edges)
            + forest.spell_pool.bytes()
            + vec_bytes(&forest.incoming)
            + forest.incoming.iter().map(vec_bytes).sum::<usize>();
        m.nodes = vec_bytes(&forest.keys)
            + vec_bytes(&forest.inside)
            + vec_bytes(&forest.kb)
            + forest.kb.iter().filter(|kb| kb.x.is_some()).count() * std::mem::size_of::<KbExtra>()
            + table_bytes(
                forest.ids.len(),
                std::mem::size_of::<Key>() + std::mem::size_of::<u32>(),
            );
        m.molecules = table_bytes(molecules.len(), std::mem::size_of::<Molecule>());
        m.designs = table_bytes(
            per_design.len(),
            std::mem::size_of::<u128>() + std::mem::size_of::<u64>(),
        );
        (m.held, m.held_used, m.held_queued) = held_use(forest);
        m
    }
}

/// Set by [`stop_walks`]; every running walk ends at its next derivation.
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// End every running walk at its next derivation, with `stopped` reading
/// `"requested"` and its report intact — for a caller watching a resource
/// the walk does not track, such as its memory.
pub fn stop_walks() {
    STOP.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// How many times [`shed_walks`] has been called.
static SHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// End the walks running now at their next derivation, with `stopped` reading
/// `"shed"` and their reports intact, and leave later walks free to run.
///
/// [`stop_walks`] ends every walk for good, so one walk whose frontier
/// outgrows a memory budget ends the whole search. Shedding drops only that
/// walk's frontier: its region stays open with the floor its report leaves,
/// and the search goes on to the regions after it.
pub fn shed_walks() {
    SHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// How many times [`shed_walks`] has been called.
pub fn walks_shed() -> u64 {
    SHED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether [`shed_walks`] was called after the count read `since`.
fn shed_since(since: u64) -> bool {
    SHED.load(std::sync::atomic::Ordering::Relaxed) != since
}

/// Whether [`stop_walks`] has been called, for a caller whose own loop must
/// end too: a walk that is asked to stop reports `"requested"` and returns,
/// so a loop that only asks "is there budget left?" would call it again
/// forever.
pub fn walks_stopped() -> bool {
    STOP.load(std::sync::atomic::Ordering::Relaxed)
}

/// Visit every design over the lattice `allowed` spells in ascending
/// constrained objective, each once, at the first derivation that spells it.
/// `visit` receives the objective on a design's reported scale, the whole
/// sequence and the design's constrained MFE in dekacalories (see
/// [`constrained_energy`]) and the base pairs of that derivation's structure,
/// sorted, and returns `true` to stop. `Ok(None)` when the
/// lattice admits no design.
///
/// `build` says where the inside values come from and when each cell's rule
/// instances are generated; every choice visits the same designs in the same
/// order.
///
/// Serves the fixed-flank shape [`super::edge_marginals`] does.
#[allow(clippy::too_many_arguments)]
pub fn walk_designs<F>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
    max_derivations: u64,
    max_seconds: f64,
    walk: impl Into<Walk>,
    mut visit: F,
) -> Result<Option<WalkStats>, LazyError>
where
    F: FnMut(f64, &str, Option<i32>, &[(u32, u32)]) -> Result<bool, LazyError> + Send,
{
    walk_designs_bounded(
        params,
        protein,
        usage,
        lambda,
        allowed,
        utr5,
        utr3,
        constraints,
        config,
        max_derivations,
        max_seconds,
        walk.into(),
        None,
        &[],
        None,
        |objective, seq, energy, own, _, _| visit(objective, seq, energy, own),
    )
}

/// [`walk_designs`] over the derivations whose structure pairs every
/// position of `must_pair` (absolute, 0-based over the whole mRNA): the
/// grammar never declares one unpaired and the production tables come from
/// the scan with the same requirement. Each design is visited once, at its
/// best structure that meets the requirement, in order of that value; the
/// energy handed to `visit` is that structure's, which can sit above the
/// design's constrained MFE.
#[allow(clippy::too_many_arguments)]
pub fn walk_designs_must<F>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    (utr5, utr3): (&str, &str),
    constraints: RegionConstraints,
    must_pair: &[u32],
    config: ScanConfig,
    (max_derivations, max_seconds): (u64, f64),
    walk: impl Into<Walk>,
    mut visit: F,
) -> Result<Option<WalkStats>, LazyError>
where
    F: FnMut(f64, &str, Option<i32>, &[(u32, u32)]) -> Result<bool, LazyError> + Send,
{
    walk_designs_bounded(
        params,
        protein,
        usage,
        lambda,
        allowed,
        utr5,
        utr3,
        constraints,
        config,
        max_derivations,
        max_seconds,
        walk.into(),
        None,
        must_pair,
        None,
        |objective, seq, energy, own, _, _| visit(objective, seq, energy, own),
    )
}

/// [`walk_designs`], with every derivation outside `window` dropped as the
/// walk meets it ([`window::Bound`]) and each design's adaptation (the sum of
/// its lattice edges' `cai`, `NaN` without a window) and the derivations
/// the bound has cut so far handed to `visit`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walk_designs_bounded<F>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
    max_derivations: u64,
    max_seconds: f64,
    walk: Walk,
    window: Option<&window::Window>,
    must_pair: &[u32],
    scanned: Option<Tables>,
    visit: F,
) -> Result<Option<WalkStats>, LazyError>
where
    F: FnMut(f64, &str, Option<i32>, &[(u32, u32)], f64, u64) -> Result<bool, LazyError> + Send,
{
    config.validate()?;
    let aa = crate::residues(protein)?;
    let (u5, u3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &u5, &u3) else {
        return Ok(None);
    };
    let grammar = Grammar {
        params,
        dfa: &graph,
        lambda,
        kk: (1.0 - lambda) * 100.0,
        constraint: PairConstraint::new(&constraints, &graph),
        must: MustPairConstraint::new(must_pair, graph.seq_len),
        single_max_len: config.single_max_len,
    };
    let Walk {
        build,
        node_dedup,
        incremental_spell,
        pipelined_visit,
        spellers,
        dedup_index,
        tight_lists,
    } = walk;
    let started = Instant::now();
    let mut forest = match build {
        Build::Recorded => Forest::build(&grammar),
        Build::OnDemand => Forest::build_on_demand(&grammar),
        Build::Production => {
            // A pruned column keeps a value that is not the cell's minimum.
            let exact = ScanConfig {
                beam: crate::Beam::exact(),
                ..config
            };
            // Tables the caller already scanned at this weight over this
            // lattice are the ones the scan below would fill, so the walk
            // builds from them rather than scanning twice.
            let tables = match scanned {
                Some(tables) => {
                    assert!(must_pair.is_empty(), "scanned tables require no pair");
                    tables
                }
                None => {
                    let Some((_, tables)) = crate::scan_restricted_with_region_must(
                        params,
                        protein,
                        usage,
                        lambda,
                        allowed,
                        (utr5, utr3),
                        constraints,
                        must_pair,
                        exact,
                    )?
                    else {
                        return Ok(None);
                    };
                    tables
                }
            };
            Forest::build_production(&grammar, &tables, started.elapsed().as_secs_f64())
        }
    };
    let build_s = started.elapsed().as_secs_f64();
    // In a development build, a file created as the walk starts, so a profiler
    // started with collection off can begin at the walk rather than the build.
    if let Some(mark) = crate::dev_env::var_os(crate::dev_env::Setting::WalkMark) {
        let _ = std::fs::write(mark, b"");
    }
    forest.node_dedup = node_dedup;
    forest.memory_layout(dedup_index, tight_lists);
    forest.bound = window.map(|w| window::Bound::new(&graph, w, lambda));
    let Some(&root) = forest.ids.get(&Key::Cell(Cell::C(graph.end()))) else {
        return Ok(None);
    };
    let shelves: Vec<ShelfReader> = if spellers > 0 {
        forest.open_shelf();
        let shelf = forest.shelf.as_ref().expect("just opened");
        (0..spellers).map(|_| shelf.reader()).collect()
    } else {
        Vec::new()
    };
    let offset = -grammar.kk * graph.codon_offset();
    let len = graph.seq_len as usize;
    let mut per_design: FxHashMap<u128, u64> = FxHashMap::default();
    let mut previous = Spelled::default();
    let deliver = Deliver {
        grammar: &grammar,
        incremental: incremental_spell,
        readoff: Readoff::default(),
        text: String::new(),
        visit,
    };
    #[cfg(debug_assertions)]
    let mut spelled_by: FxHashMap<u128, Vec<u8>> = FxHashMap::default();
    #[cfg(debug_assertions)]
    let mut molecules: FxHashSet<Molecule> = FxHashSet::default();
    let mut stats = WalkStats::unwalked(build_s, "exhausted");
    let walk_start = Instant::now();
    forest.root = root;
    forest.shed = SHED.load(std::sync::atomic::Ordering::Relaxed);
    forest.deadline = std::time::Duration::try_from_secs_f64(max_seconds)
        .ok()
        .and_then(|d| walk_start.checked_add(d));
    let mut k = 0usize;
    let mut lap = Instant::now();
    let mut split = |into: &mut f64| {
        let now = Instant::now();
        *into += (now - lap).as_secs_f64();
        lap = now;
    };
    let mut run = |outlet: &mut Outlet<'_, '_, '_, F>| -> Result<(), LazyError> {
        while let Some(val) = forest.get(root, k) {
            split(&mut stats.phases.next_s);
            if k == 0 {
                stats.optimum = val + offset;
            }
            stats.derivations += 1;
            stats.last_value = val + offset;
            let fp = forest.kb[root as usize].d[k].fp;
            #[cfg(debug_assertions)]
            if !molecules.insert(fp.molecule()) {
                *stats.duplicate_molecules.get_or_insert(0) += 1;
            }
            // Under node dedup every root derivation is a new design, so release
            // builds keep no per-design record; debug builds keep it to check.
            let first = if node_dedup && !cfg!(debug_assertions) {
                stats.max_per_design = 1;
                true
            } else {
                let count = per_design.entry(fp.design()).or_insert(0);
                *count += 1;
                stats.max_per_design = stats.max_per_design.max(*count);
                *count == 1
            };
            debug_assert!(
                first || !node_dedup,
                "a node-deduplicated root repeats a design"
            );
            split(&mut stats.phases.dedup_s);
            #[cfg(debug_assertions)]
            {
                let (seq, _) = spell_root(&forest, root, k, len);
                let seen = spelled_by.entry(fp.design()).or_insert_with(|| seq.clone());
                assert_eq!(*seen, seq, "two sequences share a fingerprint");
            }
            if let (true, Outlet::Spell { to, stop }) = (first, &*outlet) {
                #[cfg(debug_assertions)]
                {
                    let (seq, _) = spell_root(&forest, root, k, len);
                    forest.audit_design(&grammar, root, k, val, &seq);
                }
                stats.designs += 1;
                let job = Job {
                    sid: forest.kb[root as usize].x().sids[k],
                    cai: forest.kb[root as usize].cai_at(k),
                    bound_cut: forest.bound_cut,
                    objective: val + offset,
                    val,
                    designs: stats.designs,
                    derivations: stats.derivations,
                    #[cfg(debug_assertions)]
                    expect: spell_root(&forest, root, k, len),
                };
                // A speller ends only once the walk's queue closes.
                to[(stats.designs - 1) as usize % to.len()]
                    .send(job)
                    .expect("the speller's thread");
                split(&mut stats.phases.spell_s);
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    stats.stopped = "visitor";
                    break;
                }
            } else if first {
                let full;
                let (seq, pairs): (&[u8], &[(u32, u32)]) = if incremental_spell {
                    forest.spell_root_reusing(root, k, len, &mut previous);
                    debug_assert_eq!(
                        (previous.seq.clone(), previous.pairs.clone()),
                        spell_root(&forest, root, k, len),
                        "an incremental spelling differs from a full one"
                    );
                    (&previous.seq, &previous.pairs)
                } else {
                    full = spell_root(&forest, root, k, len);
                    (&full.0, &full.1)
                };
                split(&mut stats.phases.spell_s);
                #[cfg(debug_assertions)]
                forest.audit_design(&grammar, root, k, val, seq);
                stats.designs += 1;
                let stop = outlet.take(
                    val + offset,
                    val,
                    (stats.designs, stats.derivations),
                    seq,
                    pairs,
                    (forest.kb[root as usize].cai_at(k), forest.bound_cut),
                )?;
                split(&mut stats.phases.visit_s);
                if stop {
                    stats.stopped = "visitor";
                    break;
                }
            }
            k += 1;
            if stats.derivations >= max_derivations {
                stats.stopped = "derivations";
                break;
            }
            if walk_start.elapsed().as_secs_f64() >= max_seconds {
                stats.stopped = "seconds";
                break;
            }
            if STOP.load(std::sync::atomic::Ordering::Relaxed) {
                stats.stopped = "requested";
                break;
            }
            if shed_since(forest.shed) {
                stats.stopped = "shed";
                break;
            }
        }
        Ok(())
    };
    let mut spell_copied = None;
    let stopped_at = if !shelves.is_empty() {
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|s| {
            let mut lanes = Vec::new();
            let mut jobs_to = Vec::new();
            let mut spelling = Vec::new();
            for shelf in shelves {
                let (to, from) = std::sync::mpsc::sync_channel(PIPE_DEPTH);
                let (free_to, free) = std::sync::mpsc::channel();
                lanes.push((from, free_to));
                let (job_to, jobs) = std::sync::mpsc::sync_channel(PIPE_DEPTH);
                jobs_to.push(job_to);
                let pieces: Outlet<'_, '_, '_, F> = Outlet::Piped {
                    to,
                    free,
                    stop: &stop,
                };
                let stop = &stop;
                spelling.push(spawn_deep(s, move || {
                    spell_jobs(jobs, shelf, len, pieces, stop)
                }));
            }
            let visitor = spawn_deep(s, || visit_pieces(deliver, lanes, &stop));
            let walked = run(&mut Outlet::Spell {
                to: jobs_to,
                stop: &stop,
            });
            spell_copied = Some(
                spelling
                    .into_iter()
                    .map(|h| h.join().expect("a speller's thread"))
                    .sum::<u64>(),
            );
            let visited = visitor.join().expect("the visitor's thread");
            walked.and(visited)
        })?
    } else if pipelined_visit {
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|s| {
            let (to, from) = std::sync::mpsc::sync_channel(PIPE_DEPTH);
            let (free_to, free) = std::sync::mpsc::channel();
            let visitor = spawn_deep(s, || visit_pieces(deliver, vec![(from, free_to)], &stop));
            let walked = run(&mut Outlet::Piped {
                to,
                free,
                stop: &stop,
            });
            let visited = visitor.join().expect("the visitor's thread");
            walked.and(visited)
        })?
    } else {
        run(&mut Outlet::Direct(deliver))?;
        None
    };
    if forest.interrupted {
        stats.stopped = if STOP.load(std::sync::atomic::Ordering::Relaxed) {
            "requested"
        } else if shed_since(forest.shed) {
            "shed"
        } else {
            "seconds"
        };
    }
    // A piped walk may have spelled designs past the one its visitor stopped
    // on; report the walk as it stood at that design.
    if let Some((designs, derivations, objective)) = stopped_at {
        stats.designs = designs;
        stats.derivations = derivations;
        stats.last_value = objective;
        stats.stopped = "visitor";
    }
    stats.walk_s = walk_start.elapsed().as_secs_f64();
    let touched: Vec<usize> = (0..forest.keys.len())
        .filter(|&v| forest.kb[v].init)
        .collect();
    stats.touched_cells = touched
        .iter()
        .filter(|&&v| matches!(forest.keys[v], Key::Cell(_)))
        .count();
    stats.touched_fills = touched.len() - stats.touched_cells;
    stats.touched_instances = forest.instances_seen
        + touched
            .iter()
            .map(|&v| forest.incoming[v].len())
            .sum::<usize>();
    stats.sweep_s = forest.sweep_s;
    stats.lift_s = forest.lift_s;
    stats.inside_s = forest.inside_s;
    stats.index_s = forest.index_s;
    stats.cells = forest.cells;
    stats.nodes = forest.keys.len();
    stats.edges = forest.edges.len();
    #[cfg(not(debug_assertions))]
    let molecules = FxHashSet::<Molecule>::default();
    stats.memory = MemoryBreakdown::measure(&forest, &molecules, &per_design);
    stats.node_dedup = node_dedup.then_some(forest.dropped);
    stats.positions_copied = spell_copied.unwrap_or(previous.copied);
    stats.bound_cut = forest.bound_cut;
    stats.bound_requeued = forest.bound_requeued;
    stats.bound_fills_closed = forest.bound_fills_closed;
    stats.bound_energy_cut = forest.bound.as_ref().map_or(0, |b| b.energy_cut);
    stats.bound_outside_cut = forest.bound.as_ref().map_or(0, |b| b.cut_outside);
    stats.bound_covered_cut = forest.bound.as_ref().map_or(0, |b| b.cut_covered);
    stats.bound_outside_tested = forest.bound.as_ref().map_or(0, |b| b.tested_outside);
    stats.bound_covered_tested = forest.bound.as_ref().map_or(0, |b| b.tested_covered);
    stats.bound_energy_tested = forest.bound.as_ref().map_or(0, |b| b.tested_energy);
    stats.bound_nodes_closed = forest.bound_nodes_closed;
    stats.bound_edges_skipped = forest.bound_edges_skipped;
    stats.bound_energy_binding = forest
        .bound
        .as_ref()
        .map_or_else(Vec::new, |b| b.binding.clone());
    Ok(Some(stats))
}

/// The first modular design of a walk: its objective, sequence and design rank,
/// with the seconds spent verifying designs along the way and how the designs
/// before it were rejected.
#[derive(Debug, Clone)]
pub struct LazyReport {
    pub stats: WalkStats,
    /// Seconds building the fold cut's lattice and cost to go
    /// ([`ExactFold::SparseCut`]): before the walk, or beside it under
    /// [`CutSchedule::Background`]; zero otherwise.
    pub cut_s: f64,
    /// Folds under the cut made before its lattice landed, by the uncut Zuker
    /// fold ([`CutSchedule`]).
    pub uncut_folds: u64,
    /// Under [`Verify::window`], designs a windowed fold rejected (counted in
    /// `witness_rejects` too), and worker-seconds in windowed folds.
    pub window_rejects: u64,
    pub window_s: f64,
    /// Under [`Verify::window_reuse`]: cells windowed folds copied, and
    /// windowed structures an exact re-score did not confirm (expected 0).
    pub window_cells_copied: u64,
    pub window_unconfirmed: u64,
    /// Under [`Verify::window_auto`]: the rank whose full fold switched the
    /// UTR pairs on, and the designs checked when reuse switched on.
    pub window_utr_from: Option<u64>,
    pub window_reuse_from: Option<u64>,
    pub modular: Option<(f64, String, u64)>,
    /// Worker-seconds checking designs: `witness_s + fold_s`. The checks run
    /// beside the walk, so this is not part of `stats.walk_s`.
    pub verify_s: f64,
    /// Designs rejected by a witness structure, without a fold.
    pub witness_rejects: u64,
    /// Designs settled by an exact fold.
    pub folds: u64,
    /// Designs the walk queued past an answer already found, never checked.
    pub skipped: u64,
    /// Worker-seconds offering designs to the witness pool.
    pub witness_s: f64,
    /// Worker-seconds in exact folds.
    pub fold_s: f64,
    /// Seconds the walk waited for room in the check's queue; part of
    /// `stats.walk_s`.
    pub wait_s: f64,
    /// Worker-seconds the counters above leave out, and the pool's lock waits
    /// inside `witness_s`: queue locking, waiting on an empty queue, a
    /// rejecting fold adding to the pool, and taking a snapshot of the pool
    /// and adding a witness's structure to it.
    pub queue_s: f64,
    pub starve_s: f64,
    pub pool_add_s: f64,
    pub pool_read_s: f64,
    pub pool_write_s: f64,
    /// Witness rejections by where in the pool the rejecting candidate sat:
    /// buckets of its index in scoring order `0, 1, 2–3, 4–7, …, 64–127`
    /// (witness `w` alone is `2·w`, spliced into the design's structure
    /// `2·w + 1`): the candidate kept, among those scored under
    /// [`Verify::witness_stop`].
    pub witness_rank: [u64; 8],
    /// Witness rejections by a witness spliced into the design's structure.
    pub witness_spliced: u64,
    /// Under [`Verify::fold_reuse`], folds that copied cells from a recent
    /// fold, and the cells they copied.
    pub reuse_hits: u64,
    pub cells_copied: u64,
    /// Under [`Verify::fold_slots`], designs that waited for a slot, and of
    /// them those a finished fold's structure rejected.
    pub deferred: u64,
    pub deferred_rejected: u64,
}

/// The exact free fold behind the check. All four decide the same verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExactFold {
    /// The Zuker DP (`rna_mfe::fold`).
    #[default]
    Zuker,
    /// The sparse Zuker fold (`rna_mfe::dp::fold_sparse`), no cut.
    Sparse,
    /// The sparse fold under a Carrillo–Lipman cut at the design's
    /// constrained MFE ([`crate::fold_cut`]); costs one extra exact scan and
    /// outside pass of the unconstrained λ = 1 lattice before the walk.
    SparseCut,
    /// [`Self::SparseCut`], with each pair also bounded from below by the
    /// best `V` any design of that lattice has there, so the fold drops it
    /// before computing its own; keeps the lattice's tables for the walk.
    SparseCutInside,
}

/// When the cut's lattice ([`ExactFold::SparseCut`]) reaches the check.
/// Every schedule decides the same verdicts: a fold made before the lattice
/// lands is the exact uncut Zuker fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CutSchedule {
    /// Built before the walk starts; every fold is cut.
    #[default]
    Before,
    /// Built on its own thread beside the walk's build and the walk; folds
    /// are uncut until it lands.
    Background,
    /// Never built: every fold is uncut. What the cut buys, measured.
    Withheld,
}

/// How [`first_modular`] checks each design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verify {
    /// Workers checking designs side by side.
    pub threads: usize,
    /// Witness structures kept to reject designs without a fold; `0` folds
    /// every design.
    pub witnesses: usize,
    /// Stop scoring a design's witness candidates after this many score
    /// below its constrained MFE, and keep the lowest of them; `0` scores
    /// every candidate for the lowest, `1` keeps the first. Every choice is
    /// sound; it trades scoring per design against which structure joins
    /// the pool, and so how many later designs a witness rejects
    /// ([`WitnessPool::reject_ranked`](crate::witness::WitnessPool::reject_ranked)).
    pub witness_stop: usize,
    /// When a candidate rejects a design, move the pool witness it came
    /// from to the front, so witnesses that reject often are tried first.
    /// Changes only the scoring order, never a verdict.
    pub witness_promote: bool,
    /// Under the cut ([`ExactFold::SparseCut`]), keep the last this many
    /// folds and copy into each new fold the spans it shares with the most
    /// similar one whose threshold is at least its own; `0` folds afresh.
    /// Exact ([`crate::fold_cut::CachedFold`]).
    pub fold_reuse: usize,
    /// Under [`Self::fold_reuse`], copy each row from whichever recent fold
    /// reaches furthest, rather than every row from the one sharing most.
    pub fold_reuse_rows: bool,
    /// At most this many folds at once; `0` lets every worker fold. A design
    /// no witness rejects while every slot is folding waits, ordered by rank;
    /// each finished fold's structure is offered to the waiting designs, and
    /// its slot then folds the lowest-ranked one still unrejected. Waiting
    /// designs are near neighbours of the ones folding, so one fold's
    /// structure often rejects several of them. Changes which designs are
    /// folded, never a verdict.
    pub fold_slots: usize,
    /// Designs the walk hands to the check per lock, and a quarter of it per
    /// worker's take; `0` or `1` hands them over one at a time. Changes only
    /// when a design reaches a worker, never a verdict; on a loaded host each
    /// lock and wake-up can stall behind a descheduled thread.
    pub handoff_batch: usize,
    /// The exact free fold that settles a design no witness rejects.
    pub fold: ExactFold,
    /// When the cut's lattice reaches the check, under [`ExactFold::SparseCut`].
    pub cut: CutSchedule,
    /// Before folding a design no witness rejects, fold it restricted to the
    /// pairs of a window — the UTRs, the first `w5` and the last `w3` CDS
    /// positions for `Some((w5, w3))` — and the design's own pairs; a
    /// structure below its constrained MFE rejects it. `None` skips this.
    /// Changes which designs reach a full fold, never a verdict.
    pub window: Option<(usize, usize)>,
    /// With [`Self::window`], let the windowed fold also use any pair with an
    /// end in a UTR, wherever its other end lies: a UTR pairing with the
    /// middle of the CDS lies outside every `(w5, w3)`. Changes which designs
    /// reach a full fold, never a verdict.
    pub window_utr: bool,
    /// With [`Self::window`] and [`Self::fold_reuse`] (its count and rows
    /// setting), windowed folds once the cut has landed copy from recent
    /// windowed folds the spans where both designs spell alike and their own
    /// structures agree; every windowed rejection is then confirmed by
    /// scoring its structure on the design exactly. Changes which designs
    /// reach a full fold, never a verdict.
    pub window_reuse: bool,
    /// With [`Self::window`], switch both of the above on when they pay:
    /// the UTR pairs once a design the window let through is rejected by a
    /// full fold's structure that uses one, and reuse (with
    /// [`Self::fold_reuse`]) once more than 0.5% of the first 100,000 or more
    /// checked designs have reached a windowed fold. Exact, as both are.
    pub window_auto: bool,
    /// The objective of a modular design found some other way.
    /// The walk stops at the first design ranked past it, since no design
    /// beyond can beat it. The incumbent's own sequence is among the designs
    /// before that point, so the walk normally answers with it or something
    /// better; stopping there unanswered means the walk's check or objective
    /// disagrees with the incumbent's. `None` walks to the caps.
    pub incumbent: Option<f64>,
}

impl Verify {
    /// Every design settled by an exact fold, `threads` at a time.
    pub fn folds(threads: usize) -> Self {
        Self {
            threads,
            witnesses: 0,
            witness_stop: 0,
            witness_promote: false,
            fold_reuse: 0,
            fold_reuse_rows: false,
            fold_slots: 0,
            handoff_batch: 1,
            fold: ExactFold::Zuker,
            cut: CutSchedule::Before,
            window: None,
            window_utr: false,
            window_reuse: false,
            window_auto: false,
            incumbent: None,
        }
    }
}

/// Walk designs in ascending constrained objective and stop at the first whose
/// sequence verifies modular — the optimum over modular designs — or at a
/// derivation or seconds cap, or past `verify.incumbent` (`stopped` reads
/// `incumbent`).
///
/// A design is modular when its free MFE equals its constrained MFE. The walk
/// already holds the constrained MFE ([`constrained_energy`]), so a design
/// costs at most one exact free fold; both folds run only where that value is
/// unavailable, or where the walk's loop-length cap is not the one the
/// constrained fold uses.
///
/// Designs are checked by `verify.threads` workers beside the walk
/// ([`check`]). A worker first offers its design to a pool of
/// `verify.witnesses` free structures that rejected earlier designs
/// ([`WitnessPool`](crate::witness::WitnessPool)), and folds it only when
/// none rejects it. A pool structure only ever *rejects*, and only by scoring
/// below the design's constrained MFE as a structure the exact fold could
/// return, so the answer is the same with or without it; and every design
/// ranked before a modular one is settled before that one is
/// reported, so the answer is the first in the walk whatever the width.
#[allow(clippy::too_many_arguments)]
pub fn first_modular(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
    max_derivations: u64,
    max_seconds: f64,
    verify: Verify,
    walk: impl Into<Walk>,
) -> Result<Option<LazyReport>, LazyError> {
    first_modular_must(
        params,
        protein,
        usage,
        lambda,
        allowed,
        (utr5, utr3),
        constraints,
        &[],
        config,
        (max_derivations, max_seconds),
        verify,
        walk,
    )
}

/// [`first_modular`] over the walk of [`walk_designs_must`]: the designs in
/// order of their best structure that pairs every position of `must_pair`,
/// each checked against that structure's energy. A design that passes folds
/// freely no lower than that structure, so it is modular and
/// its constrained optimum is that structure: the first one found is the best
/// modular design whose optimal structure meets the
/// requirement, and an upper bound on the unrestricted answer.
#[allow(clippy::too_many_arguments)]
pub fn first_modular_must(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    (utr5, utr3): (&str, &str),
    constraints: RegionConstraints,
    must_pair: &[u32],
    config: ScanConfig,
    (max_derivations, max_seconds): (u64, f64),
    verify: Verify,
    walk: impl Into<Walk>,
) -> Result<Option<LazyReport>, LazyError> {
    let walk = walk.into();
    let cut = OnceLock::new();
    // Builds the cut into `cut`: whether the lattice admits a design, and the
    // seconds it took.
    let build_cut = || -> Result<(bool, f64), LazyError> {
        let start = Instant::now();
        let built = crate::fold_cut::FoldCut::build(
            params,
            protein,
            usage,
            allowed,
            utr5,
            utr3,
            config,
            verify.fold == ExactFold::SparseCutInside,
        )?;
        let admits = built.is_some();
        if let Some(c) = built {
            let _ = cut.set(c);
        }
        Ok((admits, start.elapsed().as_secs_f64()))
    };
    let schedule = match verify.fold {
        ExactFold::SparseCut | ExactFold::SparseCutInside => verify.cut,
        _ => CutSchedule::Withheld,
    };
    let mut cut_s = 0.0;
    if schedule == CutSchedule::Before {
        let (admits, secs) = build_cut()?;
        if !admits {
            return Ok(None);
        }
        cut_s = secs;
    }
    let ctx = check::Context {
        params,
        cds: CdsWindow {
            start: utr5.len(),
            end: utr5.len() + 3 * protein.len(),
        },
        constraints,
        walk_energy_usable: config.single_max_len == crate::DEFAULT_SINGLE_MAX_LEN,
        fold: verify.fold,
        cut: &cut,
        slack: 0,
    };
    // Past the incumbent by more than rounding: the same sequence scored by
    // another route can differ in the last bits.
    let past = |objective: f64| {
        verify
            .incumbent
            .is_some_and(|u| objective > u + INCUMBENT_TOL)
    };
    let mut passed = false;
    let mut run_check = || {
        check::check_while(ctx, verify, |offer| {
            walk_designs_must(
                params,
                protein,
                usage,
                lambda,
                allowed,
                (utr5, utr3),
                constraints,
                must_pair,
                config,
                (max_derivations, max_seconds),
                walk,
                |objective, sequence, energy, own| {
                    if past(objective) {
                        passed = true;
                        return Ok(true);
                    }
                    Ok(offer(objective, sequence, energy, own))
                },
            )
        })
    };
    let (mut stats, out) = if schedule == CutSchedule::Background {
        let (checked, built) = std::thread::scope(|s| {
            let builder = crate::task_clock::spawn(s, build_cut);
            let checked = run_check();
            (checked, builder.join().expect("the cut's thread"))
        });
        cut_s = built?.1;
        checked?
    } else {
        run_check()?
    };
    if let Some(stats) = stats.as_mut().filter(|_| passed) {
        stats.stopped = "incumbent";
    }
    Ok(stats.map(|stats| LazyReport {
        stats,
        cut_s,
        uncut_folds: out.uncut_folds,
        window_rejects: out.window_rejects,
        window_s: out.window_s,
        window_cells_copied: out.window_cells_copied,
        window_unconfirmed: out.window_unconfirmed,
        window_utr_from: out.window_utr_from,
        window_reuse_from: out.window_reuse_from,
        modular: out.modular,
        verify_s: out.witness_s + out.fold_s,
        witness_rejects: out.witness_rejects,
        folds: out.folds,
        skipped: out.skipped,
        witness_s: out.witness_s,
        fold_s: out.fold_s,
        wait_s: out.wait_s,
        queue_s: out.queue_s,
        starve_s: out.starve_s,
        pool_add_s: out.pool_add_s,
        pool_read_s: out.pool_read_s,
        pool_write_s: out.pool_write_s,
        witness_rank: out.witness_rank,
        witness_spliced: out.witness_spliced,
        reuse_hits: out.reuse_hits,
        cells_copied: out.cells_copied,
        deferred: out.deferred,
        deferred_rejected: out.deferred_rejected,
    }))
}

/// How far past [`Verify::incumbent`] a design's objective must be before the
/// walk stops on it.
pub const INCUMBENT_TOL: f64 = 1e-6;

/// Errors surfaced by [`first_modular`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LazyError {
    /// The lattice or configuration was rejected.
    #[error(transparent)]
    Beam(#[from] BeamError),
    /// A design's modularity could not be decided.
    #[error(transparent)]
    Verdict(#[from] FreeFoldError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_difference_is_found_inside_and_across_words() {
        let a: Vec<u8> = (0..37).map(|i| (i % 4) as u8).collect();
        assert_eq!(first_difference(&a, &a, 0), 37);
        for at in 0..37 {
            let mut b = a.clone();
            b[at] ^= 1;
            assert_eq!(first_difference(&a, &b, 0), at);
            assert_eq!(first_difference(&a, &b, at), at);
            assert_eq!(first_difference(&a, &b, at + 1), 37);
        }
    }

    #[test]
    fn the_index_order_key_orders_cells_as_their_debug_strings() {
        use crate::state::PairKey;
        use rna_mfe::Nucleotide;
        let nucs = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
        let positions = [
            0, 1, 2, 9, 10, 11, 19, 20, 99, 100, 101, 109, 1000, 1273, 3820,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |m: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % m as u64) as usize
        };
        let mut node = || Node::new(positions[next(positions.len())], next(2) as u8);
        let (mut ps, mut m2s) = (Vec::new(), Vec::new());
        for _ in 0..2000 {
            let (o, open, close) = (node(), node(), node());
            let (a, b) = (node(), node());
            let key = PairKey {
                open,
                close,
                opening: crate::state::Opening::new(
                    nucs[(o.pos as usize + open.pos as usize) % 4],
                    b.lane,
                ),
                nuc_close: nucs[(close.pos as usize + a.pos as usize) % 4],
            };
            ps.push(Cell::P(o, key));
            m2s.push(Cell::M2(a, b));
        }
        let strings = |v: &[Cell]| v.iter().map(|c| format!("{c:?}")).collect::<Vec<_>>();
        for cells in [ps, m2s] {
            let mut by_key = cells.clone();
            by_key.sort_by_key(order_key);
            let mut by_debug = cells;
            by_debug.sort_by_key(|c| format!("{c:?}"));
            assert_eq!(strings(&by_key), strings(&by_debug));
        }
    }
    use crate::region::PairClass;
    use crate::Enforcement;
    use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};

    fn full_sets(protein: &str) -> Vec<Vec<[u8; 3]>> {
        protein
            .chars()
            .map(|c| {
                AMINO_ACIDS[aa_to_index(c).expect("a residue")]
                    .codons
                    .iter()
                    .map(|c| c.nuc)
                    .collect()
            })
            .collect()
    }

    fn describe(inst: &Inst) -> String {
        format!("{:?} {:.9} {:?}", inst.tails, inst.weight, inst.segs)
    }

    /// Every cell's pulled instances equal, as a multiset, the instances the
    /// recording sweep generated with that cell as head.
    fn assert_pull_matches_record(
        protein: &str,
        utr5: &str,
        utr3: &str,
        constraints: RegionConstraints,
    ) {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        let (u5, u3) = (parse_utr(utr5).unwrap(), parse_utr(utr3).unwrap());
        let graph = dfa::build_restricted_with_utr(&aa, &usage, &full_sets(protein), &u5, &u3)
            .expect("a lattice");
        let grammar = Grammar {
            params: &params,
            dfa: &graph,
            lambda: 0.5,
            kk: 50.0,
            constraint: PairConstraint::new(&constraints, &graph),
            must: MustPairConstraint::inert(),
            single_max_len: crate::DEFAULT_SINGLE_MAX_LEN,
        };
        let (insts, recorded_inside) = grammar.inside();
        let mut by_head: FxHashMap<Cell, Vec<String>> = FxHashMap::default();
        for inst in &insts {
            by_head.entry(inst.head).or_default().push(describe(inst));
        }
        let (_, mut inside) = grammar.sweep(false);
        assert_eq!(
            inside.len(),
            recorded_inside.len(),
            "the same cells either way"
        );
        let index = Index::new(&inside);
        let cells: Vec<Cell> = inside.keys().copied().collect();
        let mut checked = 0;
        for cell in cells {
            if cell == Cell::C(graph.start()) {
                continue;
            }
            let mut want = by_head.remove(&cell).unwrap_or_default();
            let mut got: Vec<String> = grammar
                .pull(&mut inside, &index, cell, &|_| false)
                .iter()
                .map(describe)
                .collect();
            want.sort();
            got.sort();
            assert_eq!(got, want, "{protein}: instances with head {cell:?}");
            checked += want.len();
        }
        assert!(by_head.is_empty(), "every recorded head is a cell");
        assert_eq!(
            checked,
            insts.len(),
            "{protein}: every recorded instance pulled"
        );
    }

    fn assert_production_matches_reference(
        protein: &str,
        utr5: &str,
        utr3: &str,
        constraints: RegionConstraints,
    ) {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let cmp = compare_inside(
            &params,
            protein,
            &usage,
            0.5,
            &full_sets(protein),
            utr5,
            utr3,
            constraints,
            ScanConfig::default(),
        )
        .expect("a valid instance")
        .expect("a lattice");
        for k in &cmp.kinds {
            eprintln!("{protein}: {k:?}");
        }
        assert!(cmp.identical(), "{protein}: {:#?}", cmp.kinds);
    }

    /// A walk handed the tables a scan at its weight already filled visits
    /// exactly the designs, in exactly the order, of one that scans them
    /// itself.
    #[test]
    fn a_walk_on_scanned_tables_is_the_walk_that_scans() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let constraint = RegionConstraints::five_prime_isolation(Enforcement::Forbid)
            .with(PairClass::Utr5Utr3, Enforcement::Forbid);
        let (protein, utr5, utr3) = ("MLRSGK", "GGGAUCCG", "CGGAUCCC");
        let allowed = full_sets(protein);
        for lambda in [0.05, 0.5, 0.95] {
            let walk = |scanned: Option<Tables>| -> Vec<(String, Option<i32>)> {
                let mut seen = Vec::new();
                walk_designs_bounded(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    &allowed,
                    utr5,
                    utr3,
                    constraint,
                    ScanConfig::default(),
                    20_000,
                    f64::INFINITY,
                    Walk::from(Build::Production),
                    None,
                    &[],
                    scanned,
                    |_, sequence, energy, _, _, _| {
                        seen.push((sequence.to_string(), energy));
                        Ok(false)
                    },
                )
                .expect("the walk runs")
                .expect("a lattice");
                seen
            };
            let exact = ScanConfig {
                beam: crate::Beam::exact(),
                ..ScanConfig::default()
            };
            let (_, tables, _) = crate::scan_restricted_with_region(
                &params, protein, &usage, lambda, &allowed, utr5, utr3, constraint, exact,
            )
            .expect("a valid instance")
            .expect("a lattice");
            let own = walk(None);
            assert!(
                own.len() > 50,
                "the walk reaches designs at {lambda}: {}",
                own.len()
            );
            assert_eq!(own, walk(Some(tables)), "at {lambda}");
        }
    }

    /// A forest with one bounded node whose first candidate is queued below
    /// its own value, so the first [`Forest::pop`] takes the
    /// [`Bounded::Later`] branch and nothing else.
    ///
    /// Both nodes are degenerate fills, which spell no position and have no
    /// cell, so no energy bound and no span audit reads them; the window is
    /// `[-∞, -∞]`, which admits nothing, so the pop that follows a requeue
    /// cuts rather than keeping. `deadline` is the walk's time cap, past or
    /// ahead.
    fn requeue_forest<'g, 'a>(
        grammar: &'g Grammar<'a>,
        graph: &dfa::Dfa,
        deadline: Instant,
    ) -> Forest<'g, 'a> {
        let window = window::Window {
            lo: f64::NEG_INFINITY,
            hi: f64::NEG_INFINITY,
            floor: std::sync::Arc::new(window::Floor::default()),
            slopes: std::sync::Arc::new(Vec::new()),
        };
        let mut forest = Forest::empty(grammar);
        forest.bound = Some(window::Bound::new(graph, &window, grammar.lambda));
        forest.deadline = Some(deadline);
        let (start, end) = (graph.start(), graph.end());
        // The tail: one derivation already settled, worth more than the key
        // its parent's candidate carries.
        let tail = forest.node(Key::Fill(end, end), 0.0);
        forest.kb[tail as usize].init = true;
        forest.kb[tail as usize].d.push(Deriv {
            val: 1.0,
            edge: LEAF,
            idx: Idx::empty(),
            fp: Fingerprint::default(),
        });
        forest.kb[tail as usize].x_mut(false).cai.push(0.0);
        let v = forest.node(Key::Fill(start, start), 0.0);
        let mut tails = Tails::empty();
        tails.push(tail);
        let spelled = forest.spell_pool.intern(&[]);
        let edge = forest.push_edge(Hyper {
            tails,
            base: 0.0,
            spelled,
            pair: None,
            key: Fingerprint::default(),
            cai: 0.0,
        });
        forest.push(v, edge, Idx::zeros(1), 0.0);
        forest.start = v;
        forest.root = v;
        forest
    }

    /// A bounded walk that requeues a candidate reads the walk's limits: with
    /// the time cap already past, the requeue ends the node instead of going
    /// round again. Mirrored on a cap still ahead, where the same node does
    /// go round again — so the test turns on the cap and not on the branch
    /// being reached at all.
    #[test]
    fn a_bounded_requeue_past_the_time_cap_ends_the_walk() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let protein = "MLRS";
        let (u5, u3) = (
            parse_utr("GGGAUCCG").expect("a 5' flank"),
            parse_utr("CGGAUCCC").expect("a 3' flank"),
        );
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        let graph = dfa::build_restricted_with_utr(&aa, &usage, &full_sets(protein), &u5, &u3)
            .expect("a lattice");
        let grammar = Grammar {
            params: &params,
            dfa: &graph,
            lambda: 0.5,
            kk: 50.0,
            constraint: PairConstraint::new(&RegionConstraints::default(), &graph),
            must: MustPairConstraint::inert(),
            single_max_len: crate::DEFAULT_SINGLE_MAX_LEN,
        };
        let hour = std::time::Duration::from_secs(3600);

        let mut past = requeue_forest(&grammar, &graph, Instant::now() - hour);
        let root = past.root;
        assert!(!past.pop(root), "the node answered past the cap");
        assert_eq!(
            past.bound_requeued, 1,
            "the requeue branch is what the pop took"
        );
        assert!(past.interrupted, "the requeue ran on past the time cap");
        assert_eq!(
            past.bound_fills_closed, 0,
            "the requeue went round again past the cap"
        );

        let mut ahead = requeue_forest(&grammar, &graph, Instant::now() + hour);
        let root = ahead.root;
        assert!(!ahead.pop(root), "the window admits nothing");
        assert_eq!(
            ahead.bound_requeued, 1,
            "the requeue branch is what the pop took"
        );
        assert!(
            !ahead.interrupted,
            "a cap still ahead ended the walk anyway"
        );
        assert_eq!(
            ahead.bound_fills_closed, 1,
            "the requeue did not go round again under a cap still ahead"
        );
    }

    #[test]
    fn production_cells_equal_reference_cells() {
        let constraint = RegionConstraints::five_prime_isolation(Enforcement::Forbid)
            .with(PairClass::Utr5Utr3, Enforcement::Forbid);
        assert_production_matches_reference("MLRS", "GGGAUCCG", "CGGAUCCC", constraint);
        assert_production_matches_reference(
            "MLRS",
            "GGGAUC",
            "GAUCCC",
            RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        );
        assert_production_matches_reference(
            "GLRSA",
            "GGCGC",
            "GCGCC",
            RegionConstraints::default(),
        );
        assert_production_matches_reference(
            "KK",
            "GAGCGGACUUCGGUCCA",
            "ACAGGGAAACCUGGCUC",
            RegionConstraints::default(),
        );
        assert_production_matches_reference(
            "MA",
            "GGAGCGGACUUCGGUCCA",
            "ACAGGGAAACCUGGCUCC",
            RegionConstraints::default(),
        );
    }

    #[test]
    fn pulled_instances_equal_recorded_ones() {
        let constraint = RegionConstraints::five_prime_isolation(Enforcement::Forbid)
            .with(PairClass::Utr5Utr3, Enforcement::Forbid);
        assert_pull_matches_record("MLRS", "GGGAUCCG", "CGGAUCCC", constraint);
        assert_pull_matches_record(
            "MLRS",
            "GGGAUC",
            "GAUCCC",
            RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        );
        assert_pull_matches_record("GLRSA", "GGCGC", "GCGCC", RegionConstraints::default());
        assert_pull_matches_record(
            "KK",
            "GAGCGGACUUCGGUCCA",
            "ACAGGGAAACCUGGCUC",
            RegionConstraints::default(),
        );
        assert_pull_matches_record(
            "MA",
            "GGAGCGGACUUCGGUCCA",
            "ACAGGGAAACCUGGCUCC",
            RegionConstraints::default(),
        );
    }
}
