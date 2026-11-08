//! The modular front inside an adaptation window, by walking
//! designs at a fixed slope between energy and adaptation.
//!
//! A design is modular (in `S`) when its free MFE equals its
//! constrained energy `E`, the constrained MFE under the walk's region constraint. Its
//! front point is then `(E, A)`, with `A` its adaptation: the sum of its
//! lattice path's edge weights (`cai`), the same normalized log-adaptation the
//! lattice scores everywhere. Within a window `lo ≤ A ≤ hi` the front is the
//! set of `(E, A)` over `S` that no other such point dominates (lower or equal
//! energy and higher or equal adaptation, one strictly).
//!
//! # The walk
//!
//! The lazy walk ([`super::walk_designs`]) runs at a weight `λ`, so designs
//! come out in ascending `g = E − μ·A`, with slope `μ = 100·(1 − λ)/λ`
//! dekacalories per unit of adaptation. A slope of zero is taken as the
//! smallest `μ` that never lets adaptation reorder two energies — energies are
//! whole dekacalories, so `μ·(A_max − A_min) < 1` suffices — so designs come
//! out by energy, then by descending adaptation. A slope along the window's
//! chord starts the walk at the window: the lowest `g` over the lattice is the
//! hull vertex at that slope, which for a window between two adjacent hull
//! points is at its ends, where an energy walk would first pass every design
//! below the window.
//!
//! # The stopping rule
//!
//! When the walk reaches a design at `g`, every design `u` it has not reached
//! has `g(u) ≥ g`, so `E(u) ≥ g + μ·A(u)`. A found modular design
//! `i` at `(E_i, A_i)` therefore dominates (or ties) every unreached `u` with
//! `A(u) ≤ A_i` and `g + μ·A(u) ≥ E_i`: it **covers** the adaptations
//! `[(E_i − g)/μ, A_i]`. So does a design the caller already knows to be in
//! `S` (`known`): one with adaptation at least `hi`, such as a certified point
//! immediately above a stretch, caps the window; the finds of an earlier walk spare a
//! new one their checks, though not the walk up to them. A design that is
//! covered
//! need not be checked, and once the
//! covered set holds all of `[lo, min(hi, A_max)]` the walk is complete:
//! nothing left can add a point. Coverage only grows as `g` rises and finds
//! arrive, and finds are used as soon as the check reports them, whatever
//! their rank — any modular design dominates what it dominates.
//!
//! The same rule prunes inside the walk. A partial derivation over a lattice
//! span `[a, b)` with adaptation `x` completes to an adaptation between
//! `x + worst(start → a) + worst(b → end)` and the same with the best paths,
//! since adaptation is a sum over positions ([`Bound`]); and every completion
//! the walk has yet to reach has `g` at least that of the last design reached.
//! One whose completions all fall outside `[lo, hi]` or inside the covered set
//! can add no point, and the walk drops it as it meets it, as it drops a repeat
//! under node dedup.
//!
//! # The energy bound
//!
//! Adaptation alone says little about a small derivation: the rest of the
//! molecule can add almost any adaptation. Energy and adaptation together say
//! more. Take an outside pass at another slope `ν ≥ 0` (weight
//! `λ_ν = 100/(100 + ν)`): it gives each cell the least
//! `λ_ν·E(C) − 100·(1 − λ_ν)·A(C)` over every context `C` around it. A
//! derivation `D` of the cell, with energy `E(D)` and adaptation `x`, reaches an
//! adaptation of at least `t` only through a context with `A(C) ≥ t − x`, so
//! every such completion has energy at least
//!
//! ```text
//! E(D) + (out_ν + 100·(1 − λ_ν)·(t − x)) / λ_ν,
//! ```
//!
//! for every `ν` at once; the walk takes the largest over the slopes it was
//! given. The finds form a staircase: a design at adaptation `a` is dominated
//! once its energy reaches the least energy of a find with adaptation at least
//! `a`. A derivation each of whose stairs, from the lowest in-window
//! adaptation it can reach, has the bound at or above the stair's energy has
//! no completion the window needs, and the walk drops it. Energies are whole
//! dekacalories, so a bound above `e − 1` already rules out every energy below
//! `e`. A cell no context completes at a slope gives that slope nothing, and
//! fills are left to the adaptation bound alone.
//!
//! At the walk's own slope the bound does not depend on `x`: it is
//! `(val(D) + out_μ + 100·(1 − λ)·t) / λ`, rising with the derivation's value.
//! A node hands out its derivations in ascending value, so once the bound at
//! `t = lo` clears the stair that holds `hi`, the highest energy any window
//! adaptation needs, no later derivation of the node can complete to a point
//! the window needs, and the walk closes the node — from a candidate's key,
//! before its tails are taken, since a key is at most the value. The walk's
//! slope is therefore always among the bound's slopes.
//!
//! **Points are certified under a budget too.** A found point `p` that no
//! other found point dominates is on the window's front: a dominating `u` has
//! `E(u) ≤ E(p)` and `A(u) ≥ A(p)`, so `g(u) ≤ g(p)` and `u` was reached and
//! checked before the walk stopped, or skipped as covered by a find that would
//! then dominate `p` too. What a budget leaves open is stated exactly: the
//! uncovered adaptations at the last `g` reached, each at energies of at
//! least `g + μ·A`.
//!
//! **What a walk pays.** The walk passes every design whose `g` is below the
//! last one it needs, inside the window or not; the bound drops those it can
//! show are outside, but a design's adaptation is known only near the root,
//! so most are dropped late. Covering a window between two known points needs
//! `g` to rise by the energy between them above their chord, so a narrow
//! window with a wide energy span costs every design in that band at every
//! adaptation, while a window on a hull edge starts at its own ends.
//!
//! **The window's front is not yet the whole front's.** A point in the window
//! can be dominated by a design of `S` above `hi`; the window whose `hi` is
//! the lattice's best adaptation has no such outside, and a known design
//! above `hi` rules out what it dominates.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::super::Cell;
use super::check::{self, Found};
use super::{walk_designs_bounded, Build, ExactFold, LazyError, Verify, Walk, WalkStats};
use crate::cost_to_go::Outside;
use crate::dfa::{self, Dfa, Node};
use crate::free_fold::CdsWindow;
use crate::region::RegionConstraints;
use crate::tables::Tables;
use crate::{parse_utr, BeamError, CodonUsage, Nucleotide, Params, ScanConfig};

/// How far past a bound a partial derivation's completions must reach before
/// the walk drops it: generous against the rounding of sums taken in
/// different orders, so the prune never drops a derivation the exact
/// comparison would keep.
const PRUNE_SLACK: f64 = 1e-7;

/// How far a design's `g` may fall below the last one's: the walk's objective
/// and `g` are the same quantity summed differently.
const ORDER_TOL: f64 = 1e-6;

/// How close two adaptations are to count as equal.
const CAI_TOL: f64 = 1e-9;

/// The stack of each thread running an energy-bound outside pass.
const PASS_STACK: usize = 256 << 20;

/// How far the energy bound must clear a stair, relative to its size: the
/// bound is a sum of values taken in different orders.
const ENERGY_SLACK: f64 = 1e-6;

/// The modular designs found so far, shared between the check
/// that records them and the walk that reads them, with the coverage they
/// give at the walk's last `g`.
#[derive(Debug)]
pub struct Floor {
    /// The best adaptation found, as `f64` bits; `−∞` before the first.
    best: AtomicU64,
    found: Mutex<Vec<(i32, f64)>>,
    /// Merged covered intervals of adaptation, ascending, as of the last
    /// design the walk reached; read by the prune.
    covered: Mutex<Vec<(f64, f64)>>,
    /// The finds' staircase as of the last design reached: `(adaptation,
    /// energy)`, both ascending, no point dominating another.
    stairs: Mutex<Vec<(f64, i32)>>,
}

impl Default for Floor {
    fn default() -> Self {
        Self {
            best: AtomicU64::new(f64::NEG_INFINITY.to_bits()),
            found: Mutex::new(Vec::new()),
            covered: Mutex::new(Vec::new()),
            stairs: Mutex::new(Vec::new()),
        }
    }
}

impl Floor {
    /// The best adaptation found so far.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.best.load(Relaxed))
    }

    /// Record a modular design at constrained energy `energy`.
    pub fn record(&self, energy: i32, cai: f64) {
        self.found
            .lock()
            .expect("the found designs")
            .push((energy, cai));
        let _ = self.best.try_update(Relaxed, Relaxed, |bits| {
            (cai > f64::from_bits(bits)).then_some(cai.to_bits())
        });
    }

    /// Whether a design found so far dominates or ties `(energy, cai)`.
    pub fn dominates(&self, energy: i32, cai: f64) -> bool {
        self.get() >= cai - CAI_TOL
            && self
                .found
                .lock()
                .expect("the found designs")
                .iter()
                .any(|&(e, a)| e <= energy && a >= cai - CAI_TOL)
    }

    fn snapshot(&self) -> Vec<(i32, f64)> {
        self.found.lock().expect("the found designs").clone()
    }

    /// Whether every adaptation in `[a, b]` is covered.
    fn covers(&self, a: f64, b: f64) -> bool {
        self.covered
            .lock()
            .expect("the coverage")
            .iter()
            .any(|&(x, y)| x <= a + PRUNE_SLACK && b <= y + PRUNE_SLACK)
    }
}

/// The covered adaptations at `g` and slope `mu` (see the module docs),
/// merged, ascending.
fn coverage(found: &[(i32, f64)], g: f64, mu: f64) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = found
        .iter()
        .map(|&(e, a)| ((f64::from(e) - g) / mu, a))
        .filter(|(x, y)| x <= y)
        .collect();
    spans.sort_by(|p, q| p.0.total_cmp(&q.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (x, y) in spans {
        match merged.last_mut() {
            Some(last) if x <= last.1 + CAI_TOL => last.1 = last.1.max(y),
            _ => merged.push((x, y)),
        }
    }
    merged
}

/// The finds' staircase: the points no other find dominates, adaptation
/// ascending (so energy ascending).
fn stairs(found: &[(i32, f64)]) -> Vec<(f64, i32)> {
    let mut sorted = found.to_vec();
    sorted.sort_by(|p, q| q.1.total_cmp(&p.1).then(p.0.cmp(&q.0)));
    let mut out: Vec<(f64, i32)> = Vec::new();
    for (e, a) in sorted {
        if out.last().is_some_and(|&(_, le)| le <= e) {
            continue;
        }
        out.push((a, e));
    }
    out.reverse();
    out
}

/// Outside values at the slopes of the energy bound: per slope, its weight
/// `λ_ν`, the adaptation's weight `100·(1 − λ_ν)`, and the pass.
pub(super) type Slopes = Vec<(f64, f64, Outside)>;

/// One slope's pass, and the tables its scan filled when the walk keeps them.
type SlopePass = (f64, f64, Outside, Option<Tables>);

/// An adaptation window `[lo, hi]` and the designs a walk inside it finds.
#[derive(Clone)]
pub struct Window {
    pub lo: f64,
    pub hi: f64,
    pub floor: Arc<Floor>,
    /// The energy bound's outside passes; empty leaves it off.
    pub(super) slopes: Arc<Slopes>,
}

/// The best and worst adaptation any lattice path adds before each node and
/// after it, against a [`Window`].
pub(super) struct Bound {
    lanes: usize,
    best_in: Vec<f64>,
    best_out: Vec<f64>,
    worst_in: Vec<f64>,
    worst_out: Vec<f64>,
    lo: f64,
    hi: f64,
    floor: Arc<Floor>,
    /// The energy bound's arithmetic, and the outside passes it reads.
    eb: EnergyBound,
    slopes: Arc<Slopes>,
    /// Each walk node's outside value per slope, `NaN` until first read.
    outs: Vec<f64>,
    /// Derivations the energy bound cut.
    pub(super) energy_cut: u64,
    /// Per slope, the cuts whose bound at the lowest in-window adaptation
    /// that slope gave.
    pub(super) binding: Vec<u64>,
}

/// Per-node best and worst path adaptation from the start and to the end,
/// indexed `pos · lanes + lane`; `NaN` where no path runs.
struct Reach {
    lanes: usize,
    best_in: Vec<f64>,
    best_out: Vec<f64>,
    worst_in: Vec<f64>,
    worst_out: Vec<f64>,
}

impl Reach {
    fn new(graph: &Dfa) -> Self {
        let len = graph.seq_len as usize;
        let lanes = (0..=graph.seq_len)
            .flat_map(|p| graph.nodes_at(p).iter().map(|n| n.lane as usize + 1))
            .max()
            .unwrap_or(1);
        let at = |n: Node| n.pos as usize * lanes + n.lane as usize;
        let size = (len + 1) * lanes;
        let mut r = Reach {
            lanes,
            best_in: vec![f64::NAN; size],
            best_out: vec![f64::NAN; size],
            worst_in: vec![f64::NAN; size],
            worst_out: vec![f64::NAN; size],
        };
        r.best_in[at(graph.start())] = 0.0;
        r.worst_in[at(graph.start())] = 0.0;
        for p in 0..graph.seq_len {
            for &n in graph.nodes_at(p) {
                let (b, w) = (r.best_in[at(n)], r.worst_in[at(n)]);
                if b.is_nan() {
                    continue;
                }
                for e in graph.right(n) {
                    let t = at(e.to);
                    r.best_in[t] = r.best_in[t].max(b + e.cai);
                    r.worst_in[t] = r.worst_in[t].min(w + e.cai);
                }
            }
        }
        r.best_out[at(graph.end())] = 0.0;
        r.worst_out[at(graph.end())] = 0.0;
        for p in (0..graph.seq_len).rev() {
            for &n in graph.nodes_at(p) {
                for e in graph.right(n) {
                    let (b, w) = (r.best_out[at(e.to)], r.worst_out[at(e.to)]);
                    if b.is_nan() {
                        continue;
                    }
                    let s = at(n);
                    r.best_out[s] = r.best_out[s].max(b + e.cai);
                    r.worst_out[s] = r.worst_out[s].min(w + e.cai);
                }
            }
        }
        r
    }

    fn at(&self, n: Node) -> usize {
        n.pos as usize * self.lanes + n.lane as usize
    }

    /// The best and worst adaptation of a whole design.
    fn extremes(&self, graph: &Dfa) -> (f64, f64) {
        let s = self.at(graph.start());
        (self.best_out[s], self.worst_out[s])
    }
}

impl Bound {
    pub(super) fn new(graph: &Dfa, window: &Window, lambda: f64) -> Self {
        let r = Reach::new(graph);
        Bound {
            lanes: r.lanes,
            best_in: r.best_in,
            best_out: r.best_out,
            worst_in: r.worst_in,
            worst_out: r.worst_out,
            lo: window.lo,
            hi: window.hi,
            floor: Arc::clone(&window.floor),
            eb: EnergyBound {
                weights: window.slopes.iter().map(|&(l, kk, _)| (l, kk)).collect(),
                walk_at: window.slopes.iter().position(|&(l, _, _)| l == lambda),
                lambda,
                kk: (1.0 - lambda) * 100.0,
                lo: window.lo,
                hi: window.hi,
            },
            slopes: Arc::clone(&window.slopes),
            outs: Vec::new(),
            energy_cut: 0,
            binding: vec![0; window.slopes.len()],
        }
    }

    /// What the walk does with node `v`'s derivation spelling the lattice span
    /// `[a, b)`, valued `val` at the walk's weight with adaptation `cai`, the
    /// least value the node has left. `cell` is the node's cell, `None` for a
    /// fill.
    pub(super) fn admits(
        &mut self,
        v: u32,
        cell: Option<Cell>,
        (a, b): (Node, Node),
        val: f64,
        cai: f64,
    ) -> Admit {
        let energy = self.eb.energy(val, cai);
        let stairs = match cell {
            Some(_) if !self.slopes.is_empty() => {
                self.floor.stairs.lock().expect("the stairs").clone()
            }
            _ => Vec::new(),
        };
        let outs = match cell {
            Some(cell) if !stairs.is_empty() => Some(self.outsides(v, cell)),
            _ => None,
        };
        if let Some(outs) = &outs {
            if self.eb.closes(outs, &stairs, val) {
                return Admit::Close;
            }
        }
        let at = |n: Node| n.pos as usize * self.lanes + n.lane as usize;
        let best = cai + self.best_in[at(a)] + self.best_out[at(b)];
        let worst = cai + self.worst_in[at(a)] + self.worst_out[at(b)];
        // A span no path reaches gives `NaN` and is kept: the walk only
        // derives spans on the lattice, so this never prunes on it.
        if best < self.lo - PRUNE_SLACK || worst > self.hi + PRUNE_SLACK {
            return Admit::Cut;
        }
        let (x, y) = (worst.max(self.lo), best.min(self.hi));
        if x <= y && self.floor.covers(x, y) {
            return Admit::Cut;
        }
        if let Some(outs) = outs.filter(|_| x <= y) {
            if self.eb.dominated(&outs, &stairs, energy, cai, (x, y)) {
                self.energy_cut += 1;
                if let Some(i) = self.eb.binding(&outs, cai, x) {
                    self.binding[i] += 1;
                }
                return Admit::Cut;
            }
        }
        Admit::Keep
    }

    /// Check the energy bound against a design the walk reached, through
    /// node `v`'s derivation spelling `[a, b)` at `val` and adaptation `cai`:
    /// the bound at the design's own adaptation is at most its energy, and
    /// when the design lands in the window undominated, neither a cut nor a
    /// closure claims the derivation. Debug builds run it at every design.
    #[cfg(debug_assertions)]
    pub(super) fn audit(
        &mut self,
        v: u32,
        cell: Cell,
        (a, b): (Node, Node),
        (val, cai): (f64, f64),
        (energy_of, cai_of): (i32, f64),
    ) {
        if self.slopes.is_empty() {
            return;
        }
        let outs = self.outsides(v, cell);
        let energy = self.eb.energy(val, cai);
        let floor = self.eb.floor_at(&outs, energy, cai, cai_of);
        assert!(
            floor <= f64::from(energy_of) + ENERGY_SLACK * (1.0 + floor.abs()),
            "{cell:?}: energy floor {floor} above a completion at {energy_of}"
        );
        let inside = cai_of >= self.lo - CAI_TOL && cai_of <= self.hi + CAI_TOL;
        // One snapshot for both questions: a find recorded between them
        // would make a cut look wrong.
        let found = self.floor.snapshot();
        let dominated = found
            .iter()
            .any(|&(e, a)| e <= energy_of && a >= cai_of - CAI_TOL);
        if !inside || dominated {
            return;
        }
        let stairs = stairs(&found);
        let at = |n: Node| n.pos as usize * self.lanes + n.lane as usize;
        let best = cai + self.best_in[at(a)] + self.best_out[at(b)];
        let worst = cai + self.worst_in[at(a)] + self.worst_out[at(b)];
        let range = (worst.max(self.lo), best.min(self.hi));
        assert!(
            !self.eb.dominated(&outs, &stairs, energy, cai, range),
            "{cell:?}: cut, yet completes to ({energy_of}, {cai_of}) undominated"
        );
        assert!(
            !self.eb.closes(&outs, &stairs, val),
            "{cell:?}: closed, yet completes to ({energy_of}, {cai_of}) undominated"
        );
    }

    /// Node `v`'s outside value at each slope, read once.
    fn outsides(&mut self, v: u32, cell: Cell) -> Vec<f64> {
        let k = self.slopes.len();
        let at = v as usize * k;
        if self.outs.len() < at + k {
            self.outs.resize(at + k, f64::NAN);
        }
        if self.outs[at].is_nan() {
            let public = cell.public();
            for (i, (_, _, out)) in self.slopes.iter().enumerate() {
                self.outs[at + i] = out.raw_at(public);
            }
        }
        self.outs[at..at + k].to_vec()
    }

    /// Whether node `v`, holding no derivation below `val`, is closed:
    /// [`Self::closes`] before a candidate's tails are taken, from its key.
    pub(super) fn closed_at(&mut self, v: u32, cell: Cell, val: f64) -> bool {
        let Some(i) = self.eb.walk_at else {
            return false;
        };
        let Some(top) = self.eb.top(&self.floor.stairs.lock().expect("the stairs")) else {
            return false;
        };
        let out = self.outsides(v, cell)[i];
        self.eb.closes_under(out, top, val)
    }

    /// [`Self::closed_at`] for node `v` as a test on values alone, with the
    /// stairs read once: `None` when the bound cannot close `v` now.
    pub(super) fn gate_at(&mut self, v: u32, cell: Cell) -> Option<Gate<'_>> {
        let i = self.eb.walk_at?;
        let top = self
            .eb
            .top(&self.floor.stairs.lock().expect("the stairs"))?;
        let out = self.outsides(v, cell)[i];
        out.is_finite().then_some(Gate {
            eb: &self.eb,
            out,
            top,
        })
    }
}

/// [`Bound::closed_at`] frozen at one node and one reading of the stairs.
pub(super) struct Gate<'b> {
    eb: &'b EnergyBound,
    out: f64,
    top: i32,
}

impl Gate<'_> {
    /// Whether a derivation of the node valued `val` or more is closed.
    pub(super) fn closes(&self, val: f64) -> bool {
        self.eb.closes_under(self.out, self.top, val)
    }
}

/// The energy bound's arithmetic (see the module docs), apart from the walk
/// and the outside passes: each slope's weights, which slope is the walk's
/// own, and the window.
#[derive(Debug, Clone)]
struct EnergyBound {
    /// Per slope, `(λ_ν, 100·(1 − λ_ν))`.
    weights: Vec<(f64, f64)>,
    walk_at: Option<usize>,
    /// The walk's weight and the adaptation's weight in its values.
    lambda: f64,
    kk: f64,
    lo: f64,
    hi: f64,
}

impl EnergyBound {
    /// The energy of a derivation valued `val` at the walk's weight with
    /// adaptation `cai`.
    fn energy(&self, val: f64, cai: f64) -> f64 {
        (val + self.kk * cai) / self.lambda
    }

    /// The least energy of a completion reaching adaptation `t` or more, of a
    /// derivation at `energy` and adaptation `cai` whose cell has outside
    /// values `outs`, one per slope.
    fn floor_at(&self, outs: &[f64], energy: f64, cai: f64, t: f64) -> f64 {
        let rest = self
            .weights
            .iter()
            .zip(outs)
            .filter(|(_, o)| o.is_finite())
            .map(|(&(l, kk), &o)| (o + kk * (t - cai)) / l)
            .fold(f64::NEG_INFINITY, f64::max);
        energy + rest
    }

    /// Which slope gives the bound at adaptation `t`.
    fn binding(&self, outs: &[f64], cai: f64, t: f64) -> Option<usize> {
        self.weights
            .iter()
            .zip(outs)
            .enumerate()
            .filter(|(_, (_, o))| o.is_finite())
            .map(|(i, (&(l, kk), &o))| (i, (o + kk * (t - cai)) / l))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// Whether the bound clears energy `e`: every completion it bounds has an
    /// energy of `e` or more.
    fn clears(bound: f64, e: i32) -> bool {
        bound - ENERGY_SLACK * (1.0 + bound.abs()) > f64::from(e) - 1.0
    }

    /// Whether no derivation of the cell valued `val` or more can complete
    /// to a point the window needs. At the walk's own slope the bound reads
    /// only the value, not the adaptation.
    fn closes(&self, outs: &[f64], stairs: &[(f64, i32)], val: f64) -> bool {
        match (self.walk_at, self.top(stairs)) {
            (Some(i), Some(top)) => self.closes_under(outs[i], top, val),
            _ => false,
        }
    }

    /// The energy of the stair that holds `hi`: the highest any window
    /// adaptation needs.
    fn top(&self, stairs: &[(f64, i32)]) -> Option<i32> {
        stairs
            .iter()
            .find(|&&(a, _)| a >= self.hi - CAI_TOL)
            .map(|&(_, e)| e)
    }

    /// [`Self::closes`] from the cell's outside value at the walk's slope
    /// and the stair that holds `hi`.
    fn closes_under(&self, out: f64, top: i32, val: f64) -> bool {
        out.is_finite() && Self::clears((val + out + self.kk * self.lo) / self.lambda, top)
    }

    /// Whether every completion landing at an adaptation in `[p, q]` is
    /// dominated by a stair.
    fn dominated(
        &self,
        outs: &[f64],
        stairs: &[(f64, i32)],
        energy: f64,
        cai: f64,
        (p, q): (f64, f64),
    ) -> bool {
        let mut from = p;
        for &(a, e) in stairs {
            if a < from - CAI_TOL {
                continue;
            }
            if !Self::clears(self.floor_at(outs, energy, cai, from), e) {
                return false;
            }
            if a >= q - CAI_TOL {
                return true;
            }
            from = a;
        }
        // Above the best find nothing is dominated.
        false
    }
}

/// What a bounded walk does with a derivation ([`Bound::admits`]).
pub(super) enum Admit {
    Keep,
    /// No completion lands where the window needs it.
    Cut,
    /// Nor does any of the node's later derivations'.
    Close,
}

/// One point of the window's modular front.
#[derive(Debug, Clone)]
pub struct FrontPoint {
    /// Constrained energy, which is the free MFE, in dekacalories.
    pub energy: i32,
    /// Adaptation: the lattice path's summed edge `cai`.
    pub cai: f64,
    pub sequence: String,
    /// The design's rank among those the walk reached, and the derivations
    /// the bound had cut by then.
    pub rank: u64,
    pub cut_before: u64,
}

/// How a window walk went.
#[derive(Debug, Clone)]
pub struct WindowReport {
    pub stats: WalkStats,
    /// The weight the walk ran at, and its slope `μ` in dekacalories per unit
    /// of adaptation.
    pub lambda: f64,
    pub slope: f64,
    /// The lattice's best and worst whole-design adaptation.
    pub cai_max: f64,
    pub cai_min: f64,
    /// The window's front, energy ascending (so adaptation ascending).
    pub front: Vec<FrontPoint>,
    /// Every modular design found, front or not.
    pub found: usize,
    /// Whether the front is the window's whole front: the covered set took
    /// the whole window, or the walk ran out of designs.
    pub complete: bool,
    /// Whether the known designs took part in the coverage that completed
    /// the walk.
    pub capped: bool,
    /// The `g` of the last design reached, and the adaptations of the window
    /// still uncovered there: under a budget, every point still missing has
    /// an uncovered adaptation `A` and energy at least `last_g + slope·A`.
    pub last_g: Option<f64>,
    pub open: Vec<(f64, f64)>,
    /// Designs reached, those outside the window or covered when reached,
    /// and those offered to the check.
    pub reached: u64,
    pub outside: u64,
    pub covered: u64,
    pub offered: u64,
    /// Designs the caller's `admit` refused, before the check.
    pub refused: u64,
    pub cut_s: f64,
    /// The energy bound's slopes, the seconds its outside passes took, and
    /// the derivations it cut.
    pub energy_slopes: Vec<f64>,
    pub energy_s: f64,
    pub energy_cut: u64,
    /// Per slope of `energy_slopes`, the cuts whose bound it gave.
    pub energy_binding: Vec<u64>,
    pub verify_s: f64,
    pub witness_rejects: u64,
    pub folds: u64,
    pub skipped: u64,
    pub wall_s: f64,
}

/// A design's adaptation over the lattice a window walk uses: the sum of
/// its path's edge `cai`. `Ok(None)` when the lattice is empty or does not
/// spell `sequence`.
pub fn adaptation(
    protein: &str,
    usage: &CodonUsage,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    sequence: &str,
) -> Result<Option<f64>, BeamError> {
    let Some(graph) = lattice(protein, usage, allowed, utr5, utr3)? else {
        return Ok(None);
    };
    let Some(nts) = sequence
        .bytes()
        .map(Nucleotide::from_byte)
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    Ok(graph.path_cai(&nts))
}

fn lattice(
    protein: &str,
    usage: &CodonUsage,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
) -> Result<Option<Dfa>, BeamError> {
    let aa = crate::residues(protein)?;
    let (u5, u3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    Ok(dfa::build_restricted_with_utr(
        &aa, usage, allowed, &u5, &u3,
    ))
}

/// The weight at slope `mu` (dekacalories per unit of adaptation); a slope of
/// zero or less is the smallest one that never reorders two energies over
/// designs whose adaptation spans `range`.
pub fn slope_lambda(mu: f64, range: f64) -> f64 {
    let mu = if mu > 0.0 {
        mu
    } else {
        1.0 / (2.0 * range.max(0.0) + 2.0)
    };
    100.0 / (100.0 + mu)
}

/// The window's modular front over the lattice `allowed`
/// spells, under `constraints` (the constraint whose energy the walk orders by),
/// walked at slope `mu` ([`slope_lambda`]). `known` holds designs the caller
/// knows to be in `S`, as (constrained energy in dekacalories, adaptation): they
/// cover what they dominate and are not reported. `energy_slopes` are the
/// slopes `ν ≥ 0` of the energy bound, one outside pass each; empty leaves it
/// off. `max_seconds` bounds the whole call: the cut and the outside passes
/// come out of it, and the walk stops when it runs out. `admit`, when given,
/// keeps a design only when it accepts the whole molecule; `S` is then the
/// modular designs it accepts, and a design it refuses is neither
/// checked nor offered. `Ok(None)` when the
/// lattice admits no design. See the module docs for the rule and its proof.
#[allow(clippy::too_many_arguments)]
pub fn window_front(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
    window: (f64, f64),
    mu: f64,
    known: &[(i32, f64)],
    energy_slopes: &[f64],
    max_derivations: u64,
    max_seconds: f64,
    verify: Verify,
    walk: Walk,
    admit: Option<&(dyn Fn(&str) -> bool + Sync)>,
) -> Result<Option<WindowReport>, LazyError> {
    window_front_sharing(
        params,
        protein,
        usage,
        allowed,
        utr5,
        utr3,
        constraints,
        config,
        window,
        mu,
        known,
        energy_slopes,
        max_derivations,
        max_seconds,
        verify,
        walk,
        (None, 0),
        admit,
    )
}

/// [`window_front`] with the check's fold cut held by the caller: `cut`,
/// when given, is used as it stands if already built and is built into
/// otherwise. The cut is a fact about the lattice alone — not the window, the
/// slope or the known designs — so walks of one molecule can share it, and a
/// caller walking many windows builds it once rather than once per window.
/// It must have been built under the same `verify.fold`.
///
/// `slack`, in dekacalories, is how far a design's free MFE may sit below
/// its constrained MFE for the check to pass it;
/// `0` walks the modular designs. The walk's own rules — the
/// floor, the coverage that ends it — are domination over the designs it has
/// passed, so they hold for whichever set the check passes.
#[allow(clippy::too_many_arguments)]
pub fn window_front_sharing(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
    (lo, hi): (f64, f64),
    mu: f64,
    known: &[(i32, f64)],
    energy_slopes: &[f64],
    max_derivations: u64,
    max_seconds: f64,
    verify: Verify,
    walk: Walk,
    (cut, slack): (Option<&OnceLock<crate::fold_cut::FoldCut>>, i32),
    admit: Option<&(dyn Fn(&str) -> bool + Sync)>,
) -> Result<Option<WindowReport>, LazyError> {
    let started = Instant::now();
    let Some(graph) = lattice(protein, usage, allowed, utr5, utr3)? else {
        return Ok(None);
    };
    let (cai_max, cai_min) = Reach::new(&graph).extremes(&graph);
    let lambda = slope_lambda(mu, cai_max - cai_min);
    let slope = 100.0 * (1.0 - lambda) / lambda;
    let (lo, top) = (lo.max(cai_min), hi.min(cai_max));

    // The cut's lattice is built before the walk whatever `verify.cut` says:
    // a window walk is long, and the cut is a small part of it.
    let cut_start = Instant::now();
    let own = OnceLock::new();
    let cut = cut.unwrap_or(&own);
    if cut.get().is_none()
        && matches!(
            verify.fold,
            ExactFold::SparseCut | ExactFold::SparseCutInside
        )
    {
        match crate::fold_cut::FoldCut::build(
            params,
            protein,
            usage,
            allowed,
            utr5,
            utr3,
            config,
            verify.fold == ExactFold::SparseCutInside,
        )? {
            Some(built) => {
                let _ = cut.set(built);
            }
            None => return Ok(None),
        }
    }
    let cut_s = cut_start.elapsed().as_secs_f64();
    let energy_start = Instant::now();
    // The walk's own slope closes nodes; it is taken at the walk's weight
    // exactly, so the bound finds it.
    let mut weights: Vec<f64> = energy_slopes
        .iter()
        .map(|&nu| {
            assert!(nu >= 0.0, "an energy bound's slope is not negative");
            100.0 / (100.0 + nu)
        })
        .collect();
    if !weights.is_empty() && !weights.contains(&lambda) {
        weights.push(lambda);
    }
    // One exact scan and outside pass per slope, each on its own thread: they
    // share nothing, and together they can cost more than the walk. The
    // passes already run side by side, so they split the scan's threads
    // between them rather than each taking all of them.
    let exact = ScanConfig {
        beam: crate::Beam::exact(),
        threads: (config.threads / weights.len().max(1)).max(1),
        ..config
    };
    let seq_len = graph.seq_len;
    // The pass at the walk's own weight fills the tables the walk's
    // production build would scan again; it keeps them for the walk.
    let reuse = matches!(walk.build, Build::Production);
    let pass = |l: f64| -> Result<Option<SlopePass>, BeamError> {
        let Some((dfa, tables, _)) = crate::scan_restricted_with_region(
            params,
            protein,
            usage,
            l,
            allowed,
            utr5,
            utr3,
            constraints,
            exact,
        )?
        else {
            return Ok(None);
        };
        assert_eq!(
            dfa.seq_len, seq_len,
            "the outside pass runs over the walk's lattice"
        );
        let out = Outside::compute_at_any_lambda(params, &dfa, &tables, exact);
        let keep = (reuse && l == lambda).then_some(tables);
        Ok(Some((l, (1.0 - l) * 100.0, out, keep)))
    };
    let passes: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = weights
            .iter()
            .map(|&l| {
                let deep = std::thread::Builder::new().stack_size(PASS_STACK);
                crate::task_clock::spawn_built(deep, scope, move || pass(l))
                    .expect("spawn an outside pass")
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("an outside pass panicked"))
            .collect()
    });
    let mut slopes: Slopes = Vec::new();
    let mut scanned = None;
    for p in passes {
        match p? {
            Some((l, kk, mut out, keep)) => {
                // Every pass over the one lattice holds the same keys, and the
                // keys are most of what a pass stores, so the passes kept
                // through the walk hold one set of them between them.
                if let Some((_, _, first)) = slopes.first() {
                    out.share_keys(first);
                }
                scanned = scanned.or(keep);
                slopes.push((l, kk, out));
            }
            None => return Ok(None),
        }
    }
    // The keys the passes no longer hold go back to the system before the
    // walk, whose memory budget reads resident memory.
    if slopes.len() > 1 {
        crate::motif_front::release_freed();
    }
    let energy_s = energy_start.elapsed().as_secs_f64();
    let ctx = check::Context {
        params,
        cds: CdsWindow {
            start: utr5.len(),
            end: utr5.len() + 3 * protein.len(),
        },
        constraints,
        walk_energy_usable: config.single_max_len == crate::DEFAULT_SINGLE_MAX_LEN,
        fold: verify.fold,
        cut,
        slack,
    };
    let floor = Arc::new(Floor::default());
    for &(e, a) in known {
        floor.record(e, a);
    }
    // The bound reads the stairs from the first derivation on, before any
    // design is reached.
    *floor.stairs.lock().expect("the stairs") = stairs(known);
    let window = Window {
        lo,
        hi: top,
        floor: Arc::clone(&floor),
        slopes: Arc::new(slopes),
    };
    let (mut reached, mut outside, mut covered, mut offered) = (0u64, 0u64, 0u64, 0u64);
    let mut refused = 0u64;
    let mut last_g: Option<f64> = None;
    let mut disorder: Option<String> = None;
    let mut done = false;
    // Whether the window is covered at `g`, refreshing the prune's coverage.
    let settle = |g: f64| -> bool {
        let found = floor.snapshot();
        *floor.stairs.lock().expect("the stairs") = stairs(&found);
        let spans = coverage(&found, g, slope);
        let whole = spans
            .iter()
            .any(|&(x, y)| x <= lo + CAI_TOL && top <= y + CAI_TOL);
        *floor.covered.lock().expect("the coverage") = spans;
        whole
    };
    // The cut and the outside passes spend the window's own time: the walk
    // gets what they leave.
    let walk_seconds = (max_seconds - started.elapsed().as_secs_f64()).max(0.0);
    let (stats, out) = check::check_collecting(ctx, verify, &floor, |offer| {
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
            walk_seconds,
            walk,
            Some(&window),
            &[],
            scanned,
            |objective, sequence, energy, own, cai, cut_before| {
                reached += 1;
                let e = energy.expect("a window walk reads every design's energy");
                let g = f64::from(e) - slope * cai;
                // The rule's premise: designs come out in ascending `g`. A
                // panic here would leave the check's workers waiting on the
                // queue, so the walk stops and the driver fails after.
                if let Some(last) = last_g {
                    if g < last - ORDER_TOL * (1.0 + last.abs()) {
                        disorder = Some(format!("g {last} then {g} at ({e}, {cai})"));
                        return Ok(true);
                    }
                }
                last_g = Some(g);
                if settle(g) {
                    done = true;
                    return Ok(true);
                }
                if cai < lo - CAI_TOL || cai > top + CAI_TOL {
                    outside += 1;
                    return Ok(false);
                }
                if floor.dominates(e, cai) {
                    covered += 1;
                    return Ok(false);
                }
                if admit.is_some_and(|admit| !admit(sequence)) {
                    refused += 1;
                    return Ok(false);
                }
                offered += 1;
                Ok(offer(objective, sequence, energy, own, cai, cut_before))
            },
        )
    })?;
    if let Some(d) = disorder {
        panic!("the walk left ascending g = E - slope*A: {d}");
    }
    let Some(stats) = stats else {
        return Ok(None);
    };
    // Finds that landed after the walk stopped can complete the coverage.
    let found = floor.snapshot();
    let final_spans = last_g.map(|g| coverage(&found, g, slope));
    let whole = |spans: &[(f64, f64)]| {
        spans
            .iter()
            .any(|&(x, y)| x <= lo + CAI_TOL && top <= y + CAI_TOL)
    };
    let exhausted = matches!(stats.stopped, "exhausted");
    let complete = exhausted || done || final_spans.as_deref().is_some_and(whole);
    let own: Vec<(i32, f64)> = out
        .found
        .iter()
        .filter_map(|f| f.energy.map(|e| (e, f.cai)))
        .collect();
    let capped = complete
        && !exhausted
        && !known.is_empty()
        && !last_g.is_some_and(|g| whole(&coverage(&own, g, slope)));
    let open = if complete {
        Vec::new()
    } else {
        uncovered(lo, top, final_spans.as_deref().unwrap_or(&[]))
    };
    Ok(Some(WindowReport {
        lambda,
        slope,
        cai_max,
        cai_min,
        found: out.found.len(),
        front: pareto(&out.found),
        complete,
        capped,
        last_g,
        open,
        reached,
        outside,
        covered,
        offered,
        refused,
        cut_s,
        energy_slopes: weights.iter().map(|l| 100.0 * (1.0 - l) / l).collect(),
        energy_s,
        energy_cut: stats.bound_energy_cut,
        energy_binding: stats.bound_energy_binding.clone(),
        verify_s: out.witness_s + out.fold_s,
        witness_rejects: out.witness_rejects,
        folds: out.folds,
        skipped: out.skipped,
        wall_s: started.elapsed().as_secs_f64(),
        stats,
    }))
}

/// The parts of `[lo, hi]` no merged span covers.
fn uncovered(lo: f64, hi: f64, spans: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut open = Vec::new();
    let mut from = lo;
    for &(x, y) in spans {
        if y < from {
            continue;
        }
        if x > from {
            open.push((from, x.min(hi)));
        }
        from = from.max(y);
        if from >= hi {
            break;
        }
    }
    if from < hi {
        open.push((from, hi));
    }
    open.retain(|(x, y)| x < y);
    open
}

/// The points no other found design dominates, energy ascending; one design
/// per point, the lowest-ranked.
fn pareto(found: &[Found]) -> Vec<FrontPoint> {
    let mut sorted: Vec<&Found> = found.iter().collect();
    sorted.sort_by(|a, b| {
        a.energy
            .cmp(&b.energy)
            .then(b.cai.total_cmp(&a.cai))
            .then(a.rank.cmp(&b.rank))
    });
    let mut front: Vec<FrontPoint> = Vec::new();
    for f in sorted {
        let energy = f
            .energy
            .expect("a collecting walk reads every design's energy");
        if front.last().is_some_and(|p| p.cai >= f.cai - CAI_TOL) {
            continue;
        }
        front.push(FrontPoint {
            energy,
            cai: f.cai,
            sequence: f.seq.clone(),
            rank: f.rank,
            cut_before: f.cut_before,
        });
    }
    front
}

#[cfg(test)]
mod tests {
    use super::{stairs, EnergyBound, CAI_TOL};

    /// A small deterministic generator, so the cases are the same each run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn int(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }
    }

    /// The least energy of a stair at or above adaptation `a`.
    fn stair_energy(stairs: &[(f64, i32)], a: f64) -> Option<i32> {
        stairs
            .iter()
            .filter(|&&(s, _)| s >= a - CAI_TOL)
            .map(|&(_, e)| e)
            .min()
    }

    /// Whether every completion of a derivation at `(energy, cai)` through
    /// `contexts` that lands in `[lo, hi]` is dominated by a stair.
    fn truly_dominated(
        contexts: &[(i32, f64)],
        stairs: &[(f64, i32)],
        (energy, cai): (i32, f64),
        (lo, hi): (f64, f64),
    ) -> bool {
        contexts.iter().all(|&(e, a)| {
            let at = cai + a;
            !(lo..=hi).contains(&at) || stair_energy(stairs, at).is_some_and(|s| energy + e >= s)
        })
    }

    /// Against every completion of synthetic contexts: the floor is at most
    /// each completion's energy at its adaptation, a cut claims only
    /// derivations whose in-window completions are all dominated, and a
    /// closure only cells whose derivations from that value on are.
    #[test]
    fn the_energy_bound_claims_only_dominated_completions() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut cuts, mut closes) = (0, 0);
        for _ in 0..20_000 {
            let mu = [30.0, 200.0, 900.0][rng.int(0, 2) as usize];
            let lambda = 100.0 / (100.0 + mu);
            let weights: Vec<(f64, f64)> = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0]
                .iter()
                .map(|m| {
                    let l = 100.0 / (100.0 + m * mu);
                    (l, 100.0 * (1.0 - l))
                })
                .collect();
            let contexts: Vec<(i32, f64)> = (0..rng.int(1, 12))
                .map(|_| (rng.int(-300, 0), -3.0 * rng.unit()))
                .collect();
            let outs: Vec<f64> = weights
                .iter()
                .map(|&(l, kk)| {
                    contexts
                        .iter()
                        .map(|&(e, a)| l * f64::from(e) - kk * a)
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let (min_a, max_a) = contexts
                .iter()
                .fold((0.0f64, -3.0f64), |(lo, hi), &(_, a)| {
                    (lo.min(a), hi.max(a))
                });
            let d = (rng.int(-200, 0), -3.0 * rng.unit());
            let (u, v) = (rng.unit(), rng.unit());
            let span = (d.1 + min_a - 0.2, d.1 + max_a + 0.2);
            let at = |t: f64| span.0 + t * (span.1 - span.0);
            let (lo, hi) = (at(u.min(v)), at(u.max(v)));
            let found: Vec<(i32, f64)> = (0..rng.int(1, 5))
                .map(|_| (rng.int(-500, 0), lo - 0.3 + (hi - lo + 0.6) * rng.unit()))
                .collect();
            let st = stairs(&found);
            let eb = EnergyBound {
                weights: weights.clone(),
                walk_at: Some(3),
                lambda,
                kk: 100.0 * (1.0 - lambda),
                lo,
                hi,
            };
            assert_eq!(
                weights[3].0, lambda,
                "the walk's slope is among the bound's"
            );
            let energy = f64::from(d.0);
            for &(e, a) in &contexts {
                let floor = eb.floor_at(&outs, energy, d.1, d.1 + a);
                assert!(
                    floor <= f64::from(d.0 + e) + 1e-6,
                    "floor {floor} above a completion at {}",
                    d.0 + e
                );
            }
            let range = ((d.1 + min_a).max(lo), (d.1 + max_a).min(hi));
            if range.0 <= range.1 && eb.dominated(&outs, &st, energy, d.1, range) {
                cuts += 1;
                assert!(
                    truly_dominated(&contexts, &st, d, (lo, hi)),
                    "a cut claims an undominated completion: {d:?} {contexts:?} {st:?} [{lo}, {hi}]"
                );
            }
            let val = lambda * energy - eb.kk * d.1;
            if eb.closes(&outs, &st, val) {
                closes += 1;
                // Every derivation of the cell valued `val` or more.
                for _ in 0..8 {
                    let later = (rng.int(-200, 0), -3.0 * rng.unit());
                    if lambda * f64::from(later.0) - eb.kk * later.1 >= val {
                        assert!(
                            truly_dominated(&contexts, &st, later, (lo, hi)),
                            "a closure claims an undominated completion: {later:?} after {d:?}"
                        );
                    }
                }
                assert!(truly_dominated(&contexts, &st, d, (lo, hi)));
            }
        }
        assert!(cuts > 100 && closes > 100, "cuts {cuts}, closures {closes}");
    }
}
