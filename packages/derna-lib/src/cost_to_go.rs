//! The best energy of everything *outside* a cell — the true cost to go.
//!
//! The forward fold fills each cell with the best parse of what is *inside* it.
//! This is the mirror: for each cell, the best energy of the rest of the
//! molecule, given that cell's key. Together they say what the best whole
//! molecule routed through a cell costs, which is what a feasibility drop under
//! an energy ceiling actually needs.
//!
//! # Why a cheaper credit exists, and what it gives up
//!
//! [`crate::bounds`] credits a point's completion `optimum - cell_min`, which is
//! admissible because inside and outside are independent given the key. Its
//! looseness at a cell has an exact form:
//!
//! ```text
//! true_outside(X) - (optimum - cell_min(X)) = (cell_min(X) + true_outside(X)) - optimum
//! ```
//!
//! — the best whole molecule through `X`, less the global optimum. **The cheap
//! credit's looseness at a cell is exactly that cell's regret.** It is therefore
//! *zero* on every cell the optimal derivation passes through, and grows with
//! how poor a cell is.
//!
//! The consequence is the reason this module exists. Under the true outside a
//! cell is **empty** whenever the best molecule through it misses the ceiling,
//! and an empty cell contributes no compose calls at all. The cheap credit can
//! never empty a cell — a cell's own best point is trivially within any budget
//! of itself — so it removes stored volume while leaving every call in place.
//! The two are different levers, not the same lever at different strengths.
//!
//! # This is a scalar pass over the scalar lattice
//!
//! An energy ceiling constrains one accumulator, so the cost to go is a single
//! [`Energy`] per cell rather than a front. The pass is therefore priced against
//! the scalar fold and not the front-valued one, which is what makes it cheap
//! relative to the fold it prunes.
//!
//! The tables it reads are the forward tables at `λ = 1`, where the objective is
//! pure folding energy: the adaptation half of the BRD carries weight `1 - λ`
//! and drops out. A ceiling is stated over energy alone, so that is the right
//! lattice to take the cost to go on.
//!
//! # What makes it admissible
//!
//! Each value is a minimum over every derivation of the rest of the molecule,
//! so it is exact only when the tables it reads are. Two preconditions follow,
//! and both are asserted rather than assumed: the tables are the **exact**
//! fold, since a beam drops consumers and a missing consumer makes a cost to go
//! too *large*; and they are taken at `λ = 1`.
//!
//! The pass enumerates the unconstrained consumer set wherever a must-pair set
//! would have truncated a flank or evicted a multi-loop opening. Enumerating
//! *more* consumers than the fold admitted can only lower a minimum, so a
//! must-pair fold receives a looser credit, never an unsound one.

use std::sync::Arc;

use crate::dfa::{Dfa, Node, NodeLayout};
use crate::hash::FxHashMap;
pub use crate::outside_scan::SinglesTally;
use crate::outside_scan::{CellCosts, OutsideScan};
use crate::region::RegionConstraints;
use crate::state::PairKey;
use crate::tables::Tables;
use crate::{Beam, BeamError, ScanConfig};
use derna_codon_dp::CodonUsage;
use rna_mfe::{Energy, Params};

/// One cell of the forward fold's tables, as the cost to go is looked up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellRef {
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

/// The cost to go for each cell of the forward lattice.
///
/// Mixing it with a cheaper credit is sound in one direction only: **the
/// pointwise maximum of two admissible credits is admissible**, since each
/// underestimates the true cost to go and so does the larger of them.
///
/// # Every table or none
///
/// A cell's cost to go is a **minimum over its consumers**, so omitting one
/// makes the value too *large* — which over-credits the completion and discards
/// derivations that could have finished. **A half-built table is unsound rather
/// than loose**, and from a call site it looks exactly like a complete one. The
/// tables sit in one mutually-recursive system — `P → Multi → M2 → M → P`, with
/// the exterior hanging off `P` — so they are computed in one backward sweep and
/// never one at a time.
#[derive(Clone)]
pub struct Outside {
    /// The lattice's dense node layout, which every table is indexed by.
    layout: NodeLayout,
    costs: Costs,
}

/// A pass's values: the exterior as the scan left it, the spans trimmed to
/// their finite ranges ([`SpanCol`]), the closed-pair and multi-loop tables as
/// [`PairCosts`].
#[derive(Clone)]
struct Costs {
    c: Vec<f64>,
    m: Vec<SpanCol>,
    m2: Vec<SpanCol>,
    p: PairCosts,
    multi: PairCosts,
    tally: SinglesTally,
}

/// One outer node's span costs, by inner node, kept from its first finite
/// value to its last: everything outside that range is infinite.
///
/// A pass is pooled for the whole walk, and the spans are most of what it
/// holds. About half of each dense column lies outside its finite range, so a
/// pooled pass holds about half the span values it would dense, with every
/// value read back unchanged.
#[derive(Clone, Default)]
struct SpanCol {
    /// The dense column's length.
    width: u32,
    /// The inner node index `values[0]` is at.
    first: u32,
    values: Vec<f64>,
}

impl SpanCol {
    /// `dense` trimmed to its finite range.
    fn trim(dense: &[f64]) -> Self {
        let width = dense.len() as u32;
        match (
            dense.iter().position(|v| v.is_finite()),
            dense.iter().rposition(|v| v.is_finite()),
        ) {
            (Some(first), Some(last)) => Self {
                width,
                first: first as u32,
                values: dense[first..=last].to_vec(),
            },
            _ => Self {
                width,
                ..Self::default()
            },
        }
    }

    /// The value at inner node index `i`: infinite outside the kept range.
    fn get(&self, i: usize) -> f64 {
        i.checked_sub(self.first as usize)
            .and_then(|k| self.values.get(k))
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    /// Every kept value with its inner node index.
    fn indexed(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        let first = self.first as usize;
        self.values
            .iter()
            .enumerate()
            .map(move |(k, &v)| (first + k, v))
    }

    /// `mix` of two columns at every index either keeps, trimmed again: a
    /// value outside both ranges is infinite in both, and `mix` keeps it so.
    fn mixed(a: &Self, b: &Self, mix: impl Fn(f64, f64) -> f64) -> Self {
        let width = a.width.max(b.width);
        let ends =
            |c: &Self| (!c.values.is_empty()).then(|| (c.first, c.first + c.values.len() as u32));
        let range = match (ends(a), ends(b)) {
            (Some((a0, a1)), Some((b0, b1))) => Some((a0.min(b0), a1.max(b1))),
            (one, other) => one.or(other),
        };
        let Some((lo, hi)) = range else {
            return Self {
                width,
                ..Self::default()
            };
        };
        let dense: Vec<f64> = (lo..hi)
            .map(|i| mix(a.get(i as usize), b.get(i as usize)))
            .collect();
        let mut out = Self::trim(&dense);
        out.first += lo;
        out.width = width;
        out
    }
}

impl From<CellCosts> for Costs {
    fn from(costs: CellCosts) -> Self {
        Self {
            c: costs.c,
            m: costs.m.iter().map(|col| SpanCol::trim(col)).collect(),
            m2: costs.m2.iter().map(|col| SpanCol::trim(col)).collect(),
            p: PairCosts::from_maps(costs.p),
            multi: PairCosts::from_maps(costs.multi),
            tally: costs.tally,
        }
    }
}

/// One keyed table's costs: per outer node, each key's slot, and the values in
/// slot order.
///
/// Every exact pass over one lattice holds the same keys, whatever its weight,
/// and the keys are most of what a pass stores. So passes kept side by side
/// can share one set of slots ([`Outside::share_keys`]) and store only their
/// values, and a chord between two such passes is a zip over their values.
#[derive(Clone)]
struct PairCosts {
    slots: Arc<Vec<PairCol>>,
    values: Vec<f64>,
}

/// One outer node's keys and their slots, grouped by the pair's 5' position:
/// the keys opening at `pos` are `keys[starts[pos - lo]..starts[pos - lo + 1]]`.
/// A lookup reads one short contiguous run, and lookups at neighbouring 5'
/// positions read neighbouring memory, where a hash map scatters them.
#[derive(Default)]
struct PairCol {
    lo: u32,
    starts: Vec<u32>,
    keys: Vec<(PairKey, u32)>,
}

impl PairCol {
    fn from_keys(mut keys: Vec<(PairKey, u32)>) -> Self {
        let (Some(first), Some(last)) = (
            keys.iter().map(|(k, _)| k.open.pos).min(),
            keys.iter().map(|(k, _)| k.open.pos).max(),
        ) else {
            return Self::default();
        };
        keys.sort_by_key(|(k, _)| k.open.pos);
        let mut starts = vec![0u32; (last - first + 2) as usize];
        for (k, _) in &keys {
            starts[(k.open.pos - first + 1) as usize] += 1;
        }
        for i in 1..starts.len() {
            starts[i] += starts[i - 1];
        }
        Self {
            lo: first,
            starts,
            keys,
        }
    }

    /// `key`'s slot, `None` when this column does not hold it.
    fn slot(&self, key: &PairKey) -> Option<u32> {
        let i = key.open.pos.checked_sub(self.lo)? as usize;
        let (start, end) = (*self.starts.get(i)?, *self.starts.get(i + 1)?);
        self.keys[start as usize..end as usize]
            .iter()
            .find(|(k, _)| k == key)
            .map(|&(_, slot)| slot)
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

impl PairCosts {
    fn from_maps(maps: Vec<FxHashMap<PairKey, f64>>) -> Self {
        let mut values = Vec::with_capacity(maps.iter().map(FxHashMap::len).sum());
        let slots = maps
            .into_iter()
            .map(|map| {
                PairCol::from_keys(
                    map.into_iter()
                        .map(|(key, value)| {
                            let slot = u32::try_from(values.len()).expect("fewer than 2^32 cells");
                            values.push(value);
                            (key, slot)
                        })
                        .collect(),
                )
            })
            .collect();
        Self {
            slots: Arc::new(slots),
            values,
        }
    }

    /// The value at `key` of `outer`'s column, infinite where it holds none.
    fn get(&self, outer: usize, key: &PairKey) -> f64 {
        self.slots[outer]
            .slot(key)
            .map_or(f64::INFINITY, |slot| self.values[slot as usize])
    }

    /// Every entry as `(outer, key, value)`.
    fn entries(&self) -> impl Iterator<Item = (usize, &PairKey, f64)> + '_ {
        self.slots.iter().enumerate().flat_map(move |(outer, col)| {
            col.keys
                .iter()
                .map(move |(key, slot)| (outer, key, self.values[*slot as usize]))
        })
    }

    fn shares_slots(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.slots, &other.slots)
    }

    /// Whether every column holds exactly the keys `other`'s does.
    fn same_keys(&self, other: &Self) -> bool {
        self.shares_slots(other)
            || (self.slots.len() == other.slots.len()
                && self.slots.iter().zip(other.slots.iter()).all(|(a, b)| {
                    a.len() == b.len() && a.keys.iter().all(|(key, _)| b.slot(key).is_some())
                }))
    }

    /// Lay these values out in `other`'s slots and take them, when the two
    /// hold the same keys; `false`, unchanged, when they do not.
    fn share(&mut self, other: &Self) -> bool {
        if self.shares_slots(other) {
            return true;
        }
        if !self.same_keys(other) {
            return false;
        }
        let mut values = vec![f64::INFINITY; self.values.len()];
        for (mine, theirs) in self.slots.iter().zip(other.slots.iter()) {
            for (key, slot) in &mine.keys {
                let theirs = theirs.slot(key).expect("the same keys");
                values[theirs as usize] = self.values[*slot as usize];
            }
        }
        self.values = values;
        self.slots = Arc::clone(&other.slots);
        true
    }

    /// `mix(a, b)` at every key both hold.
    fn mixed(lo: &Self, hi: &Self, mix: impl Fn(f64, f64) -> f64) -> Self {
        if lo.shares_slots(hi) {
            return Self {
                slots: Arc::clone(&lo.slots),
                values: lo
                    .values
                    .iter()
                    .zip(&hi.values)
                    .map(|(&a, &b)| mix(a, b))
                    .collect(),
            };
        }
        let maps = (0..lo.slots.len())
            .map(|outer| {
                lo.slots[outer]
                    .keys
                    .iter()
                    .filter_map(|(key, slot)| {
                        let b = hi.slots[outer].slot(key)?;
                        Some((*key, mix(lo.values[*slot as usize], hi.values[b as usize])))
                    })
                    .collect()
            })
            .collect();
        Self::from_maps(maps)
    }

    /// Bytes of values, and of the slots (keys and run starts, charged at
    /// capacity).
    fn bytes(&self) -> (u64, u64) {
        let slot = std::mem::size_of::<(PairKey, u32)>() as u64;
        let keys = self
            .slots
            .iter()
            .map(|col| col.keys.capacity() as u64 * slot + col.starts.capacity() as u64 * 4)
            .sum();
        (self.values.capacity() as u64 * 8, keys)
    }
}

/// What one pass holds, by kind of table, as [`Outside::footprint`] counts it.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Footprint {
    /// Exterior values, one per node.
    pub c_values: u64,
    pub c_bytes: u64,
    /// `M` and `M2` span values, and how many of them are finite.
    pub span_values: u64,
    pub span_finite: u64,
    /// The span values kept: each column from its first finite value to its
    /// last (`span_values` counts the dense columns).
    pub span_trimmed: u64,
    pub span_bytes: u64,
    /// Closed-pair and multi-loop entries, their values' bytes and their
    /// keys' (the slots, which passes can share).
    pub pair_values: u64,
    pub pair_value_bytes: u64,
    pub pair_key_bytes: u64,
}

impl Footprint {
    /// Every table's bytes together, keys included.
    pub fn total_bytes(&self) -> u64 {
        self.c_bytes + self.span_bytes + self.pair_value_bytes + self.pair_key_bytes
    }
}

/// How two passes' cells agree, as [`Outside::cell_agreement`] counts it.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CellAgreement {
    pub pair_both: u64,
    pub pair_only_self: u64,
    pub pair_only_other: u64,
    pub span_columns: u64,
    pub span_length_differs: u64,
}

impl Outside {
    /// Compute the cost to go over `tables`, which must be the **exact,
    /// `λ = 1`** forward fold of `dfa` under `config`'s loop bound — see the
    /// module note on why that lattice and why exactness. `config.threads`
    /// workers share the pass; the values do not depend on how many.
    ///
    /// # Panics
    ///
    /// When `tables` were folded under a beam or at any other `λ`.
    pub fn compute(params: &Params, dfa: &Dfa, tables: &Tables, config: ScanConfig) -> Self {
        assert!(
            tables.lambda() == 1.0,
            "the cost to go is taken on the pure-energy lattice"
        );
        Self::compute_at_any_lambda(params, dfa, tables, config)
    }

    /// [`Self::compute`] over exact tables folded at any `λ`, whose values are
    /// read with [`Self::raw_at`]. Not a credit: the search's cost to go is the
    /// `λ = 1` one. It exists so the pass's CAI terms, which vanish at `λ = 1`,
    /// are checked against the recorded hypergraph too.
    ///
    /// # Panics
    ///
    /// When `tables` were folded under a beam.
    #[doc(hidden)]
    pub fn compute_at_any_lambda(
        params: &Params,
        dfa: &Dfa,
        tables: &Tables,
        config: ScanConfig,
    ) -> Self {
        Self::compute_checked(params, dfa, tables, config, false)
    }

    /// [`Self::compute_at_any_lambda`], scoring every internal loop the pass
    /// splits into halves over its corners too, and panicking where the two
    /// disagree.
    ///
    /// # Panics
    ///
    /// When `tables` were folded under a beam, or a split loop is mis-scored.
    #[doc(hidden)]
    pub fn compute_validated(
        params: &Params,
        dfa: &Dfa,
        tables: &Tables,
        config: ScanConfig,
    ) -> Self {
        Self::compute_checked(params, dfa, tables, config, true)
    }

    fn compute_checked(
        params: &Params,
        dfa: &Dfa,
        tables: &Tables,
        config: ScanConfig,
        validate: bool,
    ) -> Self {
        assert!(
            tables.is_exact(),
            "a beam drops consumers, and a cost to go missing one is unsound"
        );
        let costs = OutsideScan::cost_to_go(
            params,
            dfa,
            tables,
            tables.region(),
            config.single_max_len,
            config.threads,
            validate,
        );
        Self {
            layout: dfa.layout(),
            costs: costs.into(),
        }
    }

    /// The pass whose cost to go is `costs`: one lane of
    /// [`crate::outside_multi::cost_to_go_lanes`], which equals
    /// [`Self::compute_at_any_lambda`] over that lane's tables.
    pub(crate) fn from_costs(layout: NodeLayout, costs: CellCosts) -> Self {
        Self {
            layout,
            costs: costs.into(),
        }
    }

    /// Store this pass's closed-pair and multi-loop values in `with`'s slots,
    /// so the two share their keys, when both tables hold exactly the same
    /// keys — as every exact pass over one lattice does. `false`, unchanged,
    /// otherwise. Values and every lookup are unchanged either way.
    pub fn share_keys(&mut self, with: &Self) -> bool {
        let (mine, theirs) = (&mut self.costs, &with.costs);
        if !(mine.p.same_keys(&theirs.p) && mine.multi.same_keys(&theirs.multi)) {
            return false;
        }
        mine.p.share(&theirs.p) && mine.multi.share(&theirs.multi)
    }

    /// The cost to go for `cell`, or [`None`] where no completion reaches the
    /// end of the molecule from it.
    pub fn at(&self, cell: CellRef) -> Option<Energy> {
        energy(self.raw_at(cell))
    }

    /// The cost to go for `cell` on the tables' own scale, infinite where no
    /// completion reaches the end.
    #[doc(hidden)]
    pub fn raw_at(&self, cell: CellRef) -> f64 {
        match cell {
            CellRef::C(n) => self.costs.c[self.layout.ix(n)],
            CellRef::M(outer, inner) => span_col(&self.costs.m, self.layout, outer, inner),
            CellRef::M2(outer, inner) => span_col(&self.costs.m2, self.layout, outer, inner),
            CellRef::P(outer, key) => self.costs.p.get(self.layout.ix(outer), &key),
            CellRef::Multi(outer, key) => self.costs.multi.get(self.layout.ix(outer), &key),
        }
    }

    /// How many enclosing pairs the single-branch consumers visited, per way a
    /// loop was scored.
    #[doc(hidden)]
    pub fn singles_tally(&self) -> SinglesTally {
        self.costs.tally
    }

    /// A lower bound on the λ-general cost to go at `t·λ_lo + (1 − t)·λ_hi`,
    /// from the exact passes `lo` and `hi` at those two weights, `t` in
    /// `[0, 1]`.
    ///
    /// A cell's cost to go at `λ` is the least of `λ·E_c − 100·(1 − λ)·L_c`
    /// over its completions `c`: a minimum of functions linear in `λ`, so
    /// concave in it, and a concave function lies on or above its chords. The
    /// completions do not depend on `λ`, so a cell with none at one end has
    /// none at the other, and a cell missing from either pass stays infinite —
    /// a cut then abstains there, which removes less and never an answer.
    pub fn chord(lo: &Self, hi: &Self, t: f64) -> Self {
        let mix = |a: f64, b: f64| chord_mix(a, b, t);
        let spans = |a: &[SpanCol], b: &[SpanCol]| -> Vec<SpanCol> {
            a.iter()
                .zip(b)
                .map(|(x, y)| SpanCol::mixed(x, y, mix))
                .collect()
        };
        let (a, b) = (&lo.costs, &hi.costs);
        Self {
            layout: lo.layout,
            costs: Costs {
                c: a.c.iter().zip(&b.c).map(|(&x, &y)| mix(x, y)).collect(),
                m: spans(&a.m, &b.m),
                m2: spans(&a.m2, &b.m2),
                p: PairCosts::mixed(&a.p, &b.p, mix),
                multi: PairCosts::mixed(&a.multi, &b.multi, mix),
                tally: SinglesTally::default(),
            },
        }
    }

    /// Every finite value raised by `margin`: an inadmissible bound for any
    /// positive margin, which tests use as a mutant.
    #[doc(hidden)]
    #[must_use]
    pub fn shifted(mut self, margin: f64) -> Self {
        let costs = &mut self.costs;
        let spans = costs
            .m
            .iter_mut()
            .chain(costs.m2.iter_mut())
            .flat_map(|col| col.values.iter_mut());
        let pairs = costs
            .p
            .values
            .iter_mut()
            .chain(costs.multi.values.iter_mut());
        for v in costs.c.iter_mut().chain(spans).chain(pairs) {
            if v.is_finite() {
                *v += margin;
            }
        }
        self
    }

    /// Per table, `(C, M, M2, P, Multi)`: the cells `tables` hold, and how
    /// many of them could still reach a design scoring at most `threshold`,
    /// their best inside plus this cost to go. `tables` are the scalar fold
    /// this cost to go is taken against, at its weight.
    #[doc(hidden)]
    pub fn reach_census(&self, tables: &Tables, threshold: f64) -> [(u64, u64); 5] {
        let lambda = tables.lambda();
        let mut out = [(0u64, 0u64); 5];
        let mut count = |table: usize, inside: f64, outside: f64| {
            out[table].0 += 1;
            out[table].1 += u64::from(inside + outside <= threshold + 1e-6);
        };
        for (i, state) in tables.c.iter().enumerate() {
            if let Some(state) = state {
                count(0, state.brd(lambda), self.costs.c[i]);
            }
        }
        for (table, cols, costs) in [
            (1, &tables.m, &self.costs.m),
            (2, &tables.m2, &self.costs.m2),
        ] {
            for (outer, col) in cols.iter().enumerate() {
                for (inner, brd) in col.brd_cells(lambda) {
                    count(table, brd, costs[outer].get(self.layout.ix(inner)));
                }
            }
        }
        for (table, maps, costs) in [
            (3, &tables.p, &self.costs.p),
            (4, &tables.multi, &self.costs.multi),
        ] {
            for (outer, map) in maps.iter().enumerate() {
                for (key, state) in map {
                    count(table, state.brd(lambda), costs.get(outer, key));
                }
            }
        }
        out
    }

    /// Every cell's value here beside `other`'s for the same cell, over the
    /// cells this pass holds, infinite where `other` holds none.
    #[doc(hidden)]
    pub fn paired(&self, other: &Self, mut each: impl FnMut(f64, f64)) {
        let (a, b) = (&self.costs, &other.costs);
        let at = |col: &[f64], i: usize| col.get(i).copied().unwrap_or(f64::INFINITY);
        a.c.iter()
            .enumerate()
            .for_each(|(i, &v)| each(v, at(&b.c, i)));
        for (x, y) in a.m.iter().zip(&b.m).chain(a.m2.iter().zip(&b.m2)) {
            x.indexed().for_each(|(i, v)| each(v, y.get(i)));
        }
        for (x, y) in [(&a.p, &b.p), (&a.multi, &b.multi)] {
            for (outer, key, v) in x.entries() {
                each(v, y.get(outer, key));
            }
        }
    }

    /// What this pass holds and the heap its containers reserve, by kind of
    /// table; containers are charged at their capacity. An estimate, as `Tables::approx_bytes` is. Keys shared with other
    /// passes ([`Self::share_keys`]) are charged to each of them.
    #[doc(hidden)]
    pub fn footprint(&self) -> Footprint {
        let costs = &self.costs;
        let f64_bytes = std::mem::size_of::<f64>() as u64;
        let spans = costs.m.iter().chain(&costs.m2);
        let mut out = Footprint {
            c_values: costs.c.len() as u64,
            c_bytes: costs.c.capacity() as u64 * f64_bytes,
            ..Footprint::default()
        };
        for col in spans {
            out.span_values += u64::from(col.width);
            out.span_finite += col.values.iter().filter(|v| v.is_finite()).count() as u64;
            out.span_trimmed += col.values.len() as u64;
            out.span_bytes += col.values.capacity() as u64 * f64_bytes;
        }
        for table in [&costs.p, &costs.multi] {
            let (values, keys) = table.bytes();
            out.pair_values += table.values.len() as u64;
            out.pair_value_bytes += values;
            out.pair_key_bytes += keys;
        }
        out
    }

    /// Whether this pass's keyed tables share their slots with `other`'s.
    #[doc(hidden)]
    pub fn shares_keys_with(&self, other: &Self) -> bool {
        self.costs.p.shares_slots(&other.costs.p)
            && self.costs.multi.shares_slots(&other.costs.multi)
    }

    /// How far this pass's cells agree with `other`'s: closed-pair and
    /// multi-loop keys held by both or by one side only, and span columns
    /// whose lengths differ. Passes that agree everywhere could share their
    /// keys and store only values.
    #[doc(hidden)]
    pub fn cell_agreement(&self, other: &Self) -> CellAgreement {
        let (a, b) = (&self.costs, &other.costs);
        let mut out = CellAgreement::default();
        for (x, y) in [(&a.p, &b.p), (&a.multi, &b.multi)] {
            for (cx, cy) in x.slots.iter().zip(y.slots.iter()) {
                let both = cx.keys.iter().filter(|(k, _)| cy.slot(k).is_some()).count() as u64;
                out.pair_both += both;
                out.pair_only_self += cx.len() as u64 - both;
                out.pair_only_other += cy.len() as u64 - both;
            }
        }
        for (x, y) in a.m.iter().zip(&b.m).chain(a.m2.iter().zip(&b.m2)) {
            out.span_columns += 1;
            out.span_length_differs += u64::from(x.width != y.width);
        }
        out
    }

    /// The cost to go for the exterior cell at `node`.
    pub fn exterior_at(&self, node: Node) -> Option<Energy> {
        self.at(CellRef::C(node))
    }

    /// The cost to go for the closed pair `key` at its outer node.
    pub fn pair_at(&self, outer: Node, key: PairKey) -> Option<Energy> {
        self.at(CellRef::P(outer, key))
    }
}

/// The exact `λ = 1` fold of a flanked, region-constrained molecule over the
/// codons `allowed` admits, with its cost to go: the lattice, the tables and
/// the [`Outside`] over them. `None` when the lattice admits nothing.
///
/// Only `config.single_max_len` and `config.threads` are read. The fold always
/// runs the exact lattice at `λ = 1`, whatever beam and weight a caller's own
/// fold uses, because the cost to go is a minimum only over exact tables. The
/// worker count changes at most which tied traceback a cell keeps, never its
/// energy, so it never changes the cost to go.
#[allow(clippy::too_many_arguments)]
pub fn fold_with_cost_to_go(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<(Dfa, Tables, Outside)>, BeamError> {
    fold_with_cost_to_go_at_lambda(
        params,
        protein,
        usage,
        1.0,
        allowed,
        utr5,
        utr3,
        constraints,
        config,
    )
}

/// [`fold_with_cost_to_go`] at any `λ`, through
/// [`Outside::compute_at_any_lambda`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn fold_with_cost_to_go_at_lambda(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<(Dfa, Tables, Outside)>, BeamError> {
    let exact = ScanConfig {
        beam: Beam::exact(),
        ..config
    };
    let Some((graph, tables, _)) = crate::scan_restricted_with_region(
        params,
        protein,
        usage,
        lambda,
        allowed,
        utr5,
        utr3,
        constraints,
        exact,
    )?
    else {
        return Ok(None);
    };
    let outside = Outside::compute_at_any_lambda(params, &graph, &tables, exact);
    Ok(Some((graph, tables, outside)))
}

/// An `M`/`M2` entry; a column holds only the inner nodes before its outer one.
fn span_col(cols: &[SpanCol], layout: NodeLayout, outer: Node, inner: Node) -> f64 {
    cols[layout.ix(outer)].get(layout.ix(inner))
}

/// The chord's value from its two ends' values at one cell: `t·a + (1 − t)·b`,
/// infinite where either end is.
fn chord_mix(a: f64, b: f64, t: f64) -> f64 {
    if a.is_finite() && b.is_finite() {
        t * a + (1.0 - t) * b
    } else {
        f64::INFINITY
    }
}

/// A cost to go read one cell at a time: a pass, or the chord between two
/// passes, interpolated at each cell it is asked for rather than built whole
/// ([`Outside::chord`] gives the same values). A scan cut reads only the cells
/// it generates, far fewer than a pass holds.
pub(crate) enum CutBound {
    /// A pass's own values.
    Pass(std::sync::Arc<Outside>),
    /// [`Outside::chord`]`(lo, hi, t)`, by cell.
    Chord {
        lo: std::sync::Arc<Outside>,
        hi: std::sync::Arc<Outside>,
        t: f64,
    },
}

impl CutBound {
    /// The cost to go at `cell`, as [`Outside::raw_at`].
    pub(crate) fn raw_at(&self, cell: CellRef) -> f64 {
        match self {
            Self::Pass(outside) => outside.raw_at(cell),
            Self::Chord { lo, hi, t } => chord_mix(lo.raw_at(cell), hi.raw_at(cell), *t),
        }
    }
}

/// At `λ = 1` every value is a sum of integer energies, so the conversion back
/// is exact; infinity is the pass's marker for no completion.
fn energy(v: f64) -> Option<Energy> {
    if !v.is_finite() {
        return None;
    }
    debug_assert_eq!(v.fract(), 0.0, "a λ = 1 cost to go is an integer energy");
    Some(Energy(v as i32))
}

#[cfg(test)]
mod tests {
    //! The exterior is the one table whose consumers close over itself, so it
    //! has a second, hand-written derivation that shares none of the full
    //! pass's code. The two must agree on every node.

    use super::*;
    use crate::dfa;
    use crate::scan::scan;
    use derna_codon_dp::codon_table::aa_to_index;
    use derna_codon_dp::CodonUsage;
    use rna_mfe::loops::pair_from_nucleotides;

    /// Unreachable cells carry [`Energy::MAX`] internally so the minimisation needs
    /// no option at every step; the accessors convert it back at the boundary.
    fn finite(e: Energy) -> Option<Energy> {
        (e != Energy(i32::MAX)).then_some(e)
    }

    /// The exterior cost to go, mirroring `C = C + e` and `C = C + P`.
    ///
    /// `C[o]` is built from the exterior prefix `C[key.open]` plus a closed pair
    /// consumed at `o`, or from `C` at a predecessor plus one unpaired edge. Read
    /// backwards: the molecule 3′ of `o` is either an unpaired edge followed by the
    /// rest, or a closed pair opening at `o` whose own outer cell then carries the
    /// rest.
    ///
    /// Every closed pair is indexed by its **opening** node because that is the
    /// direction the backward sweep needs and the forward tables key on the outer
    /// node. The sweep runs 3′ to 5′ so `c[e.to]` and `c[outer]` are both settled
    /// before any cell that consumes them: an edge and a closed pair both advance
    /// the position strictly, so neither can point back at the cell being filled.
    fn exterior(params: &Params, dfa: &Dfa, tables: &Tables) -> Vec<Energy> {
        let n = tables.c.len();
        let mut pairs_by_open: Vec<Vec<(usize, Energy)>> = vec![Vec::new(); n];
        for pos in 0..=dfa.seq_len {
            for &outer in dfa.nodes_at(pos) {
                for (key, st) in tables.p_states(outer) {
                    let cost = st.mfe + terminal_au(params, key);
                    pairs_by_open[dfa.ix(key.open)].push((dfa.ix(outer), cost));
                }
            }
        }

        let mut c = vec![Energy(i32::MAX); n];
        for &end in dfa.completion_ends() {
            c[dfa.ix(end)] = Energy::ZERO;
        }
        for pos in (0..=dfa.seq_len).rev() {
            for &o in dfa.nodes_at(pos) {
                let oi = dfa.ix(o);
                let mut best = c[oi];
                // An unpaired exterior nucleotide credits no energy at `dangles=0`,
                // so the edge contributes its successor's cost to go alone.
                for e in dfa.right(o) {
                    best = min_finite(best, c[dfa.ix(e.to)], Energy::ZERO);
                }
                for &(outer, cost) in &pairs_by_open[oi] {
                    best = min_finite(best, c[outer], cost);
                }
                c[oi] = best;
            }
        }
        c
    }

    /// `min(best, rest + add)`, treating an unreachable `rest` as no candidate.
    fn min_finite(best: Energy, rest: Energy, add: Energy) -> Energy {
        match finite(rest) {
            Some(rest) => best.min(rest + add),
            None => best,
        }
    }

    /// The terminal-AU penalty a closing pair pays where the model applies it.
    fn terminal_au(params: &Params, key: PairKey) -> Energy {
        if pair_from_nucleotides(key.nuc_open(), key.nuc_close).applies_terminal_au() {
            Energy(params.misc.terminal_au_37)
        } else {
            Energy::ZERO
        }
    }

    #[test]
    fn the_full_pass_agrees_with_the_standalone_exterior_derivation() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        for protein in ["MYGKIIFVLM", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"] {
            let aa: Vec<usize> = protein
                .bytes()
                .map(|b| aa_to_index(b as char).unwrap())
                .collect();
            let graph = dfa::build(&aa, &usage);
            let tables = scan(&params, &graph, 1.0, ScanConfig::default());
            let full = Outside::compute(&params, &graph, &tables, ScanConfig::default());
            let alone = exterior(&params, &graph, &tables);
            for pos in 0..=graph.seq_len {
                for &node in graph.nodes_at(pos) {
                    assert_eq!(
                        full.exterior_at(node),
                        finite(alone[graph.ix(node)]),
                        "{protein} at {node:?}",
                    );
                }
            }
        }
    }

    /// The chord between exact passes at two weights is at most the exact
    /// pass between them at every cell, equals it somewhere (a cell whose
    /// best completion is the same at both ends), and is that end's pass at
    /// either end. A margin of 1 lifts it above the exact pass somewhere.
    #[test]
    fn the_chord_bounds_the_pass_between_its_weights_from_below() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = "MKTAYIAKQRQISFVKSHFSRQ"
            .bytes()
            .map(|b| aa_to_index(b as char).unwrap())
            .collect();
        let graph = dfa::build(&aa, &usage);
        let pass = |lambda| {
            let tables = scan(&params, &graph, lambda, ScanConfig::default());
            Outside::compute_at_any_lambda(&params, &graph, &tables, ScanConfig::default())
        };
        let (lo, mid, hi) = (pass(0.1), pass(0.25), pass(0.5));
        let chord = Outside::chord(&lo, &hi, (0.5 - 0.25) / (0.5 - 0.1));
        let (mut cells, mut tight, mut loose) = (0, 0, 0);
        mid.paired(&chord, |exact, bound| {
            if exact.is_finite() {
                cells += 1;
                assert!(bound <= exact + 1e-6, "{bound} above {exact}");
                tight += usize::from((exact - bound).abs() < 1e-6);
                loose += usize::from(bound + 1.0 > exact + 1e-6);
            }
        });
        assert!(
            cells > 1000 && tight > 0 && loose > 0,
            "{cells} {tight} {loose}"
        );
        for (end, t) in [(&lo, 1.0), (&hi, 0.0)] {
            end.paired(&Outside::chord(&lo, &hi, t), |x, y| {
                assert!(x == y || (x - y).abs() < 1e-9, "{x} {y}");
            });
        }
    }

    /// Passes at different weights hold the same keys, so one can take the
    /// other's slots; every value it reads is then bit-for-bit what it read
    /// before, and a chord between passes sharing slots — the zip — is
    /// bit-for-bit the chord between the same passes unshared — the keyed
    /// merge. Both sides of the comparison are read through `paired`, which
    /// looks values up by key, so a value stored in the wrong slot shows.
    #[test]
    fn sharing_keys_changes_no_value_and_no_chord() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = "MKTAYIAKQRQISFVKSHFSRQ"
            .bytes()
            .map(|b| aa_to_index(b as char).unwrap())
            .collect();
        let graph = dfa::build(&aa, &usage);
        let pass = |lambda| {
            let tables = scan(&params, &graph, lambda, ScanConfig::default());
            Outside::compute_at_any_lambda(&params, &graph, &tables, ScanConfig::default())
        };
        let (lo, hi) = (pass(0.1), pass(0.5));
        let mut shared = hi.clone();
        assert!(!shared.shares_keys_with(&lo));
        assert!(shared.share_keys(&lo));
        assert!(shared.shares_keys_with(&lo));
        let same = |a: &Outside, b: &Outside| {
            let mut cells = 0;
            a.paired(b, |x, y| {
                cells += 1;
                assert_eq!(x.to_bits(), y.to_bits(), "{x} {y}");
            });
            b.paired(a, |x, y| assert_eq!(x.to_bits(), y.to_bits(), "{x} {y}"));
            cells
        };
        assert!(same(&hi, &shared) > 1000);
        let (zip, merged) = (
            Outside::chord(&lo, &shared, 0.4),
            Outside::chord(&lo, &hi, 0.4),
        );
        assert!(zip.shares_keys_with(&lo) && !merged.shares_keys_with(&lo));
        same(&zip, &merged);
    }
}
