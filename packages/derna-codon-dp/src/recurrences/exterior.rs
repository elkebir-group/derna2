//! Exterior-loop recurrence (`O` table).
//!
//! Codon-axis lift of the single-sequence MFE `W` table — computes the BRD value of an
//! exterior loop on RNA segment `v[σ(a, s), σ(b, t)]` with codons `x` at
//! AA `a` and `y` at AA `b` pinned (main paper lines 292–320).
//!
//! Nine cases (1-based `s, t` in the paper translated to 0-based):
//!
//! 1. Single-codon base case (`a == b, x == y`): `−(1−λ) log_cai[a, x]`.
//! 2. Impossible base case (`a == b, x != y`): `INF`.
//! 3. Within-cell left shift (`s ∈ {0, 1}, a < b`): `O[a, b, s+1, t, x, y]`.
//!    Prefix cells omit this case because the equivalent optima
//!    are reached through right shifts, boundary crosses, and bifurcations.
//! 4. Within-cell right shift (`t ∈ {1, 2}, a < b`): `O[a, b, s, t-1, x, y]`.
//! 5. Left codon-boundary cross (`s == 2, a < b`): pick `x' ∈ S(w_{a+1})`
//!    consistent with `y`; pays `log_cai[a, x]` for the outgoing codon.
//! 6. Right codon-boundary cross (`t == 0, a < b`): pick `y' ∈ S(w_{b−1})`
//!    consistent with `x`; pays `log_cai[b, y]` for the outgoing codon.
//! 7. Within-codon bifurcation (`a < b`): for `c ∈ [a, b − 1], t' ∈ {0, 1},
//!    x' ∈ S(w_c)` consistent with `x`: `O[a, c, s, t', x, x'] +
//!    E_with_ext_loop[c, b, t'+1, t, x', y] + (1−λ) log_cai[c, x']`. The
//!    correction undoes the double charge applied by O's case 1.
//! 8. Between-codon bifurcation (`a < b − 1`): for `c ∈ [a, b − 2], y' ∈
//!    S(w_c), x' ∈ S(w_{c+1})`: `O[a, c, s, 2, x, y'] +
//!    E_with_ext_loop[c+1, b, 0, t, x', y]`. No CAI correction.
//! 9. Closure (`a < b, (σ(a, s), σ(b, t)) canonical`):
//!    `E[idx] + brd_value(λ, ext_loop, 0)` — adds the per-stem AU/GU
//!    bonus.
//!
//! The within-cell case order requires the descending-`s` / ascending-`t`
//! traversal documented at `super::ST_BY_NT_SPAN_FOR_N`: cases 3 / 4
//! self-reference at `(s + 1, t)` / `(s, t − 1)` so those cells must be
//! filled before the outer cell is reduced.
//!
//! Each case is a candidate emitter and `fill_o` reduces via `min`. At
//! `dangles = 0`, `ext_loop` collapses to `0` for non-AU/GU closing pairs and
//! `terminal_au_37` otherwise; no dangle credits.

use rna_mfe::loops::{pair_from_nucleotides, Nucleotide};
use rna_mfe::params::BasePair;
use smallvec::{smallvec, SmallVec};

use crate::brd_value::brd_value;
use crate::plan::{
    aa_of_nt, consistent_choices, nt_position, nuc_at, nuc_from_index, CellIdx, Plan, INF,
};

use super::{Candidate, OTrace, Trace};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

#[inline]
fn read_finite(table: &[f64], flat: usize) -> Option<f64> {
    let v = table[flat];
    (v < INF).then_some(v)
}

/// One `O` candidate's specification. The traceback walker reads `case`
/// and the inner-cell indices to descend; `aux_codon` / `aux_codon_2`
/// pin codon assignments at boundary-cross AAs (cases 5 / 6) or shared
/// AAs of the bifurcation (cases 7 / 8).
#[derive(Debug, Clone, Copy)]
struct OSpec {
    case: u8,
    value: f64,
    inner_a: CellIdx,
    inner_b: Option<CellIdx>,
    aux: usize,
    aux2: usize,
}

fn o_candidate(spec: OSpec) -> Candidate {
    Candidate {
        value: spec.value,
        trace: Trace::O(OTrace {
            case: spec.case,
            inner_idx_a: spec.inner_a,
            inner_idx_b: spec.inner_b,
            aux_codon: spec.aux,
            aux_codon_2: spec.aux2,
        }),
    }
}

/// Per-stem AU/GU bonus at `dangles=0`: `terminal_au_37` if the closing
/// pair applies the terminal-AU penalty, otherwise 0. Equivalent to
/// `exterior_loop_energy(params, &[pair])` for a single stem but avoids
/// allocating a 1-element slice.
fn ext_loop_stem_credit(plan: &Plan<'_>, pair: BasePair) -> i32 {
    if pair.applies_terminal_au() {
        plan.params.misc.terminal_au_37
    } else {
        0
    }
}

/// Nucleotide at position `nt` resolved against `pinned`: if any pinned
/// `(p, ci)` matches `aa_of_nt(nt)`, use that codon; otherwise fall back
/// to the per-position default. Avoids building a full assembled RNA when
/// only one or two positions are read.
///
/// `consistent_choices` filters pinned codons before they reach this
/// function, so duplicate-AA pins are guaranteed consistent — the first
/// match wins.
fn pinned_nuc_at(plan: &Plan<'_>, pinned: &[(usize, usize)], nt: usize) -> u8 {
    let aa = aa_of_nt(nt);
    let pos = nt % 3;
    let ci = pinned
        .iter()
        .find_map(|&(p, ci)| (p == aa).then_some(ci))
        .unwrap_or(plan.default_codon_idx[aa]);
    nuc_at(plan, aa, ci, pos)
}

/// Read `E[e_idx] + brd_value(λ, ext_loop, 0)` for an exterior-loop
/// closing pair at positions `nt_pq`. Returns `None` if the inner `E`
/// cell is `INF` or the closing pair at `nt_pq` (resolved against the
/// `pinned` codons plus per-position defaults) is non-canonical.
fn e_with_ext_loop(
    plan: &Plan<'_>,
    e_idx: CellIdx,
    nt_pq: (usize, usize),
    pinned: &[(usize, usize)],
) -> Option<f64> {
    let e_value = read_finite(&plan.tables.e, plan.tables.flat_index(e_idx))?;
    let (nt_p, nt_q) = nt_pq;
    let pair = pair_from_nucleotides(
        nuc_from_index(pinned_nuc_at(plan, pinned, nt_p)),
        nuc_from_index(pinned_nuc_at(plan, pinned, nt_q)),
    );
    if matches!(pair, BasePair::NN) {
        return None;
    }
    Some(e_value + brd_value(plan.lambda_, ext_loop_stem_credit(plan, pair), 0.0))
}

/// `(1 − λ) · log_cai[c, x']` — case 7's double-charge correction.
fn case_7_correction(plan: &Plan<'_>, c: usize, x_prime: usize) -> f64 {
    (1.0 - plan.lambda_) * plan.log_cai_dekacal_per_pos[c][x_prime]
}

// ---------------------------------------------------------------------------
// Within-cell shift cases (3 / 4)
// ---------------------------------------------------------------------------

/// Emit one within-cell shift candidate (case 3 or 4) if the inner cell
/// is finite.
fn emit_shift<F: FnMut(Candidate)>(plan: &Plan<'_>, case: u8, inner_idx: CellIdx, emit: &mut F) {
    if let Some(inner) = read_finite(&plan.tables.o, plan.tables.flat_index(inner_idx)) {
        emit(o_candidate(OSpec {
            case,
            value: inner,
            inner_a: inner_idx,
            inner_b: None,
            aux: 0,
            aux2: 0,
        }));
    }
}

/// Whether case 3 (the within-cell left shift) is taken. A prefix cell
/// (`a == 0 && s == 0`) omits it: the same optima are reached through right
/// shifts, boundary crosses and bifurcations.
fn allow_prefix_left_shift(idx: CellIdx) -> bool {
    let (a, _b, s, _t, _x, _y) = idx;
    !(a == 0 && s == 0)
}

fn o_within_cell_shift<F: FnMut(Candidate)>(plan: &Plan<'_>, idx: CellIdx, emit: &mut F) {
    let (a, b, s, t, x, y) = idx;
    if a >= b {
        return;
    }
    if s < 2 && allow_prefix_left_shift(idx) {
        emit_shift(plan, 3, (a, b, s + 1, t, x, y), emit);
    }
    if t > 0 {
        emit_shift(plan, 4, (a, b, s, t - 1, x, y), emit);
    }
}

// ---------------------------------------------------------------------------
// Boundary-cross cases (5 / 6)
// ---------------------------------------------------------------------------

/// Which side of the outer cell the cross moves through.
#[derive(Debug, Clone, Copy)]
enum CrossSide {
    Left,
    Right,
}

/// Computed inputs for the cross-case enumeration: the pinned AA (other
/// side), the AA whose codon needs choosing, the CAI charge for the
/// outgoing codon, and the case tag.
struct CrossSetup {
    other_pin: (usize, usize),
    pin_aa: usize,
    cai_charge: f64,
    case: u8,
}

fn cross_setup(plan: &Plan<'_>, idx: CellIdx, side: CrossSide) -> Option<CrossSetup> {
    let (a, b, s, t, x, y) = idx;
    if a >= b {
        return None;
    }
    let log_cai = &plan.log_cai_dekacal_per_pos;
    match side {
        CrossSide::Left if s == 2 => Some(CrossSetup {
            other_pin: (b, y),
            pin_aa: a + 1,
            cai_charge: brd_value(plan.lambda_, 0, log_cai[a][x]),
            case: 5,
        }),
        CrossSide::Right if t == 0 => Some(CrossSetup {
            other_pin: (a, x),
            pin_aa: b - 1,
            cai_charge: brd_value(plan.lambda_, 0, log_cai[b][y]),
            case: 6,
        }),
        _ => None,
    }
}

fn cross_inner_idx(idx: CellIdx, side: CrossSide, pinned: usize) -> CellIdx {
    let (a, b, s, t, x, y) = idx;
    match side {
        CrossSide::Left => (a + 1, b, 0, t, pinned, y),
        CrossSide::Right => (a, b - 1, s, 2, x, pinned),
    }
}

fn o_boundary_cross<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    side: CrossSide,
    emit: &mut F,
) {
    let Some(setup) = cross_setup(plan, idx, side) else {
        return;
    };
    for pinned in consistent_choices(plan, setup.pin_aa, &[setup.other_pin]) {
        let inner_idx = cross_inner_idx(idx, side, pinned);
        let Some(inner) = read_finite(&plan.tables.o, plan.tables.flat_index(inner_idx)) else {
            continue;
        };
        emit(o_candidate(OSpec {
            case: setup.case,
            value: inner + setup.cai_charge,
            inner_a: inner_idx,
            inner_b: None,
            aux: pinned,
            aux2: 0,
        }));
    }
}

// ---------------------------------------------------------------------------
// Bifurcation cases (7 / 8)
// ---------------------------------------------------------------------------

/// Outer-cell context for the bifurcation emitters.
struct BifContext<'p, 'a, F: FnMut(Candidate)> {
    plan: &'p Plan<'a>,
    idx: CellIdx,
    emit: &'p mut F,
}

fn try_emit_bifurcation<F: FnMut(Candidate)>(ctx: &mut BifContext<'_, '_, F>, spec: BifSpec) {
    let Some(left) = read_finite(
        &ctx.plan.tables.o,
        ctx.plan.tables.flat_index(spec.left_idx),
    ) else {
        return;
    };
    try_emit_bifurcation_with_left(
        ctx,
        BifLeftRead {
            idx: spec.left_idx,
            value: left,
        },
        spec,
    );
}

fn try_emit_bifurcation_with_left<F: FnMut(Candidate)>(
    ctx: &mut BifContext<'_, '_, F>,
    left: BifLeftRead,
    spec: BifSpec,
) {
    let Some(right) = e_with_ext_loop(ctx.plan, spec.right_idx, spec.nt_pq, &spec.pinned) else {
        return;
    };
    (ctx.emit)(o_candidate(OSpec {
        case: spec.case,
        value: left.value + right + spec.correction,
        inner_a: left.idx,
        inner_b: Some(spec.right_idx),
        aux: spec.aux,
        aux2: spec.aux2,
    }));
}

/// Read-only inputs for a single bifurcation emission attempt.
struct BifSpec {
    case: u8,
    left_idx: CellIdx,
    right_idx: CellIdx,
    nt_pq: (usize, usize),
    pinned: SmallVec<[(usize, usize); 4]>,
    correction: f64,
    aux: usize,
    aux2: usize,
}

#[derive(Debug, Clone, Copy)]
struct BifLeftRead {
    idx: CellIdx,
    value: f64,
}

fn o_within_codon_for_split<F: FnMut(Candidate)>(
    ctx: &mut BifContext<'_, '_, F>,
    c: usize,
    t_prime: usize,
) {
    let (a, b, s, t, x, y) = ctx.idx;
    // When c == a the consistency filter forces x' == x (left half is the
    // span-0 cell O[a, a, ..., x, x'] which is INF for x' != x by case 2);
    // passing the full ((a, x), (b, y)) base lets the filter drop those
    // candidates without dereferencing the inner cell.
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for x_prime in consistent_choices(ctx.plan, c, &base) {
        let spec = BifSpec {
            case: 7,
            left_idx: (a, c, s, t_prime, x, x_prime),
            right_idx: (c, b, t_prime + 1, t, x_prime, y),
            nt_pq: (nt_position(c, t_prime + 1), nt_position(b, t)),
            pinned: smallvec![(a, x), (c, x_prime), (b, y)],
            correction: case_7_correction(ctx.plan, c, x_prime),
            aux: x_prime,
            aux2: 0,
        };
        try_emit_bifurcation(ctx, spec);
    }
}

fn o_within_codon_bifurcation<F: FnMut(Candidate)>(ctx: &mut BifContext<'_, '_, F>) {
    let (a, b, _s, _t, _x, _y) = ctx.idx;
    if a >= b {
        return;
    }
    for c in a..b {
        for t_prime in 0..2 {
            o_within_codon_for_split(ctx, c, t_prime);
        }
    }
}

fn o_between_codon_for_y_prime<F: FnMut(Candidate)>(
    ctx: &mut BifContext<'_, '_, F>,
    c: usize,
    y_prime: usize,
) {
    let (a, b, s, t, x, y) = ctx.idx;
    let base_pins: [(usize, usize); 3] = [(a, x), (b, y), (c, y_prime)];
    let left_idx = (a, c, s, 2, x, y_prime);
    let Some(left_value) = read_finite(&ctx.plan.tables.o, ctx.plan.tables.flat_index(left_idx))
    else {
        return;
    };
    let left = BifLeftRead {
        idx: left_idx,
        value: left_value,
    };
    for x_prime in consistent_choices(ctx.plan, c + 1, &base_pins) {
        let spec = BifSpec {
            case: 8,
            left_idx: left.idx,
            right_idx: (c + 1, b, 0, t, x_prime, y),
            nt_pq: (nt_position(c + 1, 0), nt_position(b, t)),
            pinned: smallvec![(a, x), (c, y_prime), (c + 1, x_prime), (b, y)],
            correction: 0.0,
            aux: y_prime,
            aux2: x_prime,
        };
        try_emit_bifurcation_with_left(ctx, left, spec);
    }
}

fn o_between_codon_bifurcation<F: FnMut(Candidate)>(ctx: &mut BifContext<'_, '_, F>) {
    let (a, b, _s, _t, x, y) = ctx.idx;
    if a + 1 >= b {
        return;
    }
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for c in a..(b - 1) {
        for y_prime in consistent_choices(ctx.plan, c, &base) {
            o_between_codon_for_y_prime(ctx, c, y_prime);
        }
    }
}

// ---------------------------------------------------------------------------
// Closure (case 9)
// ---------------------------------------------------------------------------

fn o_closure<F: FnMut(Candidate)>(plan: &Plan<'_>, idx: CellIdx, emit: &mut F) {
    let (a, b, s, t, x, y) = idx;
    if a >= b {
        return;
    }
    let nt_pq = (nt_position(a, s), nt_position(b, t));
    let pinned: [(usize, usize); 2] = [(a, x), (b, y)];
    let Some(value) = e_with_ext_loop(plan, idx, nt_pq, &pinned) else {
        return;
    };
    emit(o_candidate(OSpec {
        case: 9,
        value,
        inner_a: idx,
        inner_b: None,
        aux: 0,
        aux2: 0,
    }));
}

// ---------------------------------------------------------------------------
// Generator + filler
// ---------------------------------------------------------------------------

/// Yield every `O` candidate for cell `idx` (cases 3–9). Cases 1 and 2
/// are the `a == b` base cases handled directly by [`fill_o`].
///
/// The `_scratch` parameter is unused — every closing pair is read
/// directly from `Plan::codon_lists` — but the signature is shared with
/// the other recurrence generators so the traceback dispatcher in
/// `recurrences::traceback` can match on them uniformly.
pub(crate) fn o_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    _scratch: &mut Vec<Nucleotide>,
    emit: F,
) {
    let mut emit = emit;
    emit_o_candidates(plan, idx, &mut emit);
}

fn emit_o_candidates<F: FnMut(Candidate)>(plan: &Plan<'_>, idx: CellIdx, emit: &mut F) {
    let (a, b, _s, _t, _x, _y) = idx;
    if a >= b {
        return;
    }
    o_within_cell_shift(plan, idx, emit);
    o_boundary_cross(plan, idx, CrossSide::Left, emit);
    o_boundary_cross(plan, idx, CrossSide::Right, emit);
    {
        let mut bif_ctx = BifContext { plan, idx, emit };
        o_within_codon_bifurcation(&mut bif_ctx);
        o_between_codon_bifurcation(&mut bif_ctx);
    }
    o_closure(plan, idx, emit);
}

/// Fill `O[idx]`. Base cases at `a == b`; `min` over candidates otherwise.
///
/// The `_scratch` parameter is unused — every closing pair is read
/// directly from `Plan::codon_lists` — but the signature is shared with
/// the other recurrence fillers so the dense fill driver in
/// `recurrences::sweep_cells_in_order` can pass the outer scratch
/// through uniformly.
pub(crate) fn fill_o(plan: &mut Plan<'_>, idx: CellIdx, _scratch: &mut Vec<Nucleotide>) {
    let (a, b, _s, _t, x, y) = idx;
    let i = plan.tables.flat_index(idx);
    if a == b {
        if x == y {
            plan.tables.o[i] = brd_value(plan.lambda_, 0, plan.log_cai_dekacal_per_pos[a][x]);
        }
        return;
    }
    let mut best = INF;
    let mut keep_best = |c: Candidate| {
        if c.value < best {
            best = c.value;
        }
    };
    emit_o_candidates(plan, idx, &mut keep_best);
    plan.tables.o[i] = best;
}
