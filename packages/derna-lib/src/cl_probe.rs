//! How much a Carrillo–Lipman cut could prune, measured before building one.
//!
//! Two places a bound on everything *outside* a cell could cut work:
//!
//! - **the check's exact free fold.** A design is rejected when its free MFE
//!   falls below its constrained MFE `E_c`, which is known before the fold
//!   starts. A Zuker cell `V[i, j]` matters to that verdict only if
//!   `V[i, j] + outside(i, j) < E_c`. The outside here is a lower bound shared
//!   by every design: the exact outside of the matching cell in the
//!   *unconstrained, λ = 1* design lattice — the best rest-of-molecule over every
//!   design passing through that cell, so never above the design's own.
//!   [`fold_survivors`] counts the cells that survive the cut.
//! - **the walk's table build.** A walk that stops at objective `U` reads no
//!   cell whose best whole derivation, inside plus exact outside, exceeds `U`.
//!   [`scan_survivors`] counts the cells of the constrained scan within `δ` of the
//!   optimum, with the scan's own exact outside: the most any outside bound
//!   could prune, before asking how cheaply such a bound could be had.

use rna_mfe::dp::fill_dp_tables;
use rna_mfe::{Nucleotide, Params};

use crate::cost_to_go::{CellRef, Outside};
use crate::dfa::{Dfa, Node};
use crate::hash::FxHashMap;
use crate::state::PairKey;
use crate::tables::Tables;

/// One design's Zuker cells against the cut `V + outside < E_c`.
#[derive(Debug, Clone)]
pub struct FoldSurvivors {
    pub len: usize,
    /// The design's constrained MFE and exact free MFE, dekacalories.
    pub constrained: i32,
    pub free: i32,
    /// Finite `V[i, j]` cells.
    pub cells: u64,
    /// Of those, the cells with `V + bound < E_c`.
    pub survivors: u64,
    /// The cells whose lattice cell's best whole molecule, over every design
    /// through it (inside plus outside at λ = 1), lies below `E_c`: the cells
    /// a fold could skip *before* computing them, since the lattice inside
    /// is never above the design's own `V`.
    pub pre_survivors: u64,
    /// Finite cells whose lattice cell has no outside (should be none).
    pub unmatched: u64,
    /// Pairs of the design's own MFE structure where `V + bound` exceeds the
    /// free MFE — a broken bound or a broken mapping (should be none).
    pub violations: u64,
}

/// The lattice node before each position of `seq` (bytes of `ACGU`), and the
/// end node: `len + 1` nodes along the design's path. `None` when the lattice
/// does not spell `seq`.
fn path(dfa: &Dfa, seq: &[u8]) -> Option<Vec<Node>> {
    crate::fold_cut::path(dfa, seq)
}

/// The λ = 1 lattice's closed pairs, `(outer, key) → inside` in dekacalories,
/// for [`fold_survivors`]'s pre-computation cut.
pub fn lattice_pairs(dfa: &Dfa, tables: &Tables) -> FxHashMap<(Node, PairKey), i32> {
    let end = dfa.end();
    let mut out = FxHashMap::default();
    for pos in 0..=end.pos {
        for lane in 0..2u8 {
            let o = Node::new(pos, lane);
            for (k, s) in tables.p_states(o) {
                out.insert((o, k), s.mfe.0);
            }
        }
    }
    out
}

/// Count `seq`'s Zuker cells that survive `V + outside < constrained`, with
/// `outside` the unconstrained λ = 1 lattice's cost to go (`lattice` must be
/// [`Outside::compute`] over the exact λ = 1 fold of `dfa` with no constraint,
/// `pairs` [`lattice_pairs`] over the same fold).
pub fn fold_survivors(
    params: &Params,
    dfa: &Dfa,
    lattice: &Outside,
    pairs: &FxHashMap<(Node, PairKey), i32>,
    seq: &[u8],
    constrained: i32,
) -> Option<FoldSurvivors> {
    let nodes = path(dfa, seq)?;
    let nt: Vec<Nucleotide> = seq
        .iter()
        .map(|&b| Nucleotide::from_byte(b))
        .collect::<Option<_>>()?;
    let tables = fill_dp_tables(params, &nt);
    let free = tables.f[nt.len()].map_or(0, |e| e.0);
    let (db, _) = rna_mfe::fold(params, seq).ok()?;
    let mfe_pairs = crate::witness::pairs_of(&db);
    let key = |i: usize, j: usize| PairKey {
        open: nodes[i],
        close: nodes[j],
        opening: crate::state::Opening::new(nt[i], nodes[i + 1].lane),
        nuc_close: nt[j],
    };
    let bound = |i: usize, j: usize| lattice.pair_at(nodes[j + 1], key(i, j)).map(|e| e.0);
    let mut out = FoldSurvivors {
        len: nt.len(),
        constrained,
        free,
        cells: 0,
        survivors: 0,
        pre_survivors: 0,
        unmatched: 0,
        violations: 0,
    };
    for j in 0..nt.len() {
        for i in 0..j {
            let Some(v) = tables.v.get(i, j) else {
                continue;
            };
            out.cells += 1;
            let Some(b) = bound(i, j) else {
                out.unmatched += 1;
                continue;
            };
            if v.0 + b < constrained {
                out.survivors += 1;
            }
            let inside = pairs.get(&(nodes[j + 1], key(i, j))).copied();
            debug_assert!(
                inside.is_some_and(|x| x <= v.0),
                "the lattice inside exceeds V"
            );
            if inside.is_some_and(|x| x + b < constrained) {
                out.pre_survivors += 1;
            }
        }
    }
    for &(i, j) in &mfe_pairs {
        let (i, j) = (i as usize, j as usize);
        let v = tables.v.get(i, j).map_or(i32::MAX, |e| e.0);
        if bound(i, j).is_none_or(|b| v.saturating_add(b) > free) {
            out.violations += 1;
        }
    }
    Some(out)
}

/// Cells of a constrained scan by kind, and how many lie within each `δ` of the
/// optimum by inside plus exact outside.
#[derive(Debug, Clone)]
pub struct ScanSurvivors {
    /// The optimum: the end cell's inside value.
    pub optimum: f64,
    /// Cells of each kind: C, P, Multi, M, M2.
    pub cells: [u64; 5],
    /// Per `δ`, per kind, the cells with `inside + outside <= optimum + δ`.
    pub within: Vec<(f64, [u64; 5])>,
}

/// Count `tables`' cells (an exact scan of `dfa` at its own λ) whose best
/// whole derivation lies within each of `deltas` of the optimum, by the exact
/// outside `outside` ([`Outside::compute_at_any_lambda`] over the same
/// tables).
pub fn scan_survivors(
    dfa: &Dfa,
    tables: &Tables,
    outside: &Outside,
    deltas: &[f64],
) -> ScanSurvivors {
    let lambda = tables.lambda();
    let end = dfa.end();
    let mut values: Vec<(usize, f64)> = Vec::new();
    let mut optimum = f64::NAN;
    for pos in 0..=end.pos {
        for lane in 0..2u8 {
            let o = Node::new(pos, lane);
            if let Some(s) = tables.c.get(dfa.ix(o)).copied().flatten() {
                let inside = s.brd(lambda);
                if o == end {
                    optimum = inside;
                }
                values.push((0, inside + outside.raw_at(CellRef::C(o))));
            }
            for (k, s) in tables.p_states(o) {
                values.push((1, s.brd(lambda) + outside.raw_at(CellRef::P(o, k))));
            }
            for (k, s) in tables.multi_states(o) {
                values.push((2, s.brd(lambda) + outside.raw_at(CellRef::Multi(o, k))));
            }
            for (inner, s) in tables.m_states(o) {
                values.push((3, s.brd(lambda) + outside.raw_at(CellRef::M(o, inner))));
            }
            for (inner, s) in tables.m2_states(o) {
                values.push((4, s.brd(lambda) + outside.raw_at(CellRef::M2(o, inner))));
            }
        }
    }
    let mut cells = [0u64; 5];
    for &(kind, _) in &values {
        cells[kind] += 1;
    }
    let within = deltas
        .iter()
        .map(|&d| {
            let mut n = [0u64; 5];
            for &(kind, total) in &values {
                if total <= optimum + d + 1e-6 {
                    n[kind] += 1;
                }
            }
            (d, n)
        })
        .collect();
    ScanSurvivors {
        optimum,
        cells,
        within,
    }
}
