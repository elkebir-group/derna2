//! Multi-branch loop candidate generators and fillers (`E_m`, `M`, `N`).
//!
//! Follows supplement.tex lines 252–325. The codon-axis multi-loop
//! recurrence splits across three cooperating sub-tables:
//!
//! * `E_m[idx]` — multi-loop closure entry. Min over four cases keyed on
//!   whether `σ(a, s) + 1` and / or `σ(b, t) − 1` cross codon boundaries.
//!   Each case pays the multi-loop closing constant `A = MLclosing +
//!   ml_closing(σ(a,s), σ(b,t))` plus boundary-cross CAI and reads `M`
//!   at the resulting inner cell.
//! * `M[idx]` — multi-loop body, exactly one bifurcation into two `N`
//!   halves. The within-codon split (`c == d`) double-counts the shared
//!   AA's CAI in the two halves' descents, so the candidate adds
//!   `+(1 − λ) · log_cai[c, y']` to undo one charge.
//! * `N[idx]` — multi-loop fragment with at least one branch. Seven
//!   cases: N1 closes a branch via `E[idx]`; N2 / N3 self-reference within
//!   the same `(a, b)` cell at adjusted `(s, t)` (drives the within-cell
//!   `s` descending / `t` ascending order); N4 / N5 cross a codon
//!   boundary unpaired; N6 / N7 bifurcate.
//!
//! Paper typos handled here:
//!
//! * E_m case 4 — the formula prints
//!   `λA − (1 − λ) g(w_a, x) + g(w_b, y)` without parens; both `g(...)`
//!   terms share the `(1 − λ)` factor (prose: "cross codon boundaries
//!   at both sides").
//! * M's LHS index `[s][t']` should read `[s][t]` (`t'` is a sum
//!   variable that doesn't escape the bifurcation).
//! * N case 5's inner `[s][t]` should read `[s][3]` (1-based) = `[s][2]`
//!   (0-based) — by symmetry with case 4 the boundary-cross step lands
//!   at the previous codon's last nucleotide.
//! * N case 6's prose says correction `+(1 − λ) g(w_c, x')` but the
//!   formula uses `y'`; this code uses `y'` since both halves share the codon
//!   at AA `c`.
//!
//! Each recurrence is a candidate generator (`*_candidates`) plus a
//! `min`-over-generator filler (`fill_*`). Reduction is the only
//! non-generator code per recurrence; future v2 variants swap the
//! reduction over the same candidate stream without re-shaping the
//! generators. At `dangles = 0`, `ml_stem` and `ml_closing` collapse to the same flank-zeroed
//! computation, so a single per-stem helper covers both.

use rna_mfe::loops::{pair_from_nucleotides, Nucleotide};
use rna_mfe::params::BasePair;

use crate::brd_value::brd_value;
use crate::plan::{
    aa_of_nt, consistent_choices, nt_position, nuc_at, nuc_from_index, CellIdx, Plan, INF,
};

use super::{Candidate, MTrace, MultiLoopTrace, NTrace, Trace};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Closing pair at `(σ(a, s), σ(b, t))` for `idx`, read directly from the
/// codons pinned at `(a, x)` and `(b, y)`. Returns `None` for non-canonical
/// closures — those cells stay at `INF`.
fn closing_pair_at(plan: &Plan<'_>, idx: CellIdx) -> Option<BasePair> {
    let (a, b, s, t, x, y) = idx;
    let pair = pair_from_nucleotides(
        nuc_from_index(nuc_at(plan, a, x, s)),
        nuc_from_index(nuc_at(plan, b, y, t)),
    );
    if matches!(pair, BasePair::NN) {
        None
    } else {
        Some(pair)
    }
}

/// Per-stem flank credit at `dangles=0`: `MLintern + (terminal_au if
/// the pair is AU / UA / GU / UG)`. Under `dangles=0` both the closing-pair
/// credit `ml_closing(i, j)` and the stem credit `ml_stem(i, j)` reduce to
/// this (with `pair`
/// in sequence order for `ml_stem`, reversed for `ml_closing` — but the
/// terminal-AU check is orientation-agnostic at the [`BasePair`] level,
/// so the same expression covers both).
fn ml_stem_credit(plan: &Plan<'_>, pair: BasePair) -> i32 {
    let intern = plan.params.ml.intern_37;
    let terminal_au = if pair.applies_terminal_au() {
        plan.params.misc.terminal_au_37
    } else {
        0
    };
    intern + terminal_au
}

#[inline]
fn read_finite(table: &[f64], flat: usize) -> Option<f64> {
    let v = table[flat];
    (v < INF).then_some(v)
}

/// `(1 - λ) · log_cai[c, y']` — the shared-AA double-charge correction
/// used by `M`'s within-codon bifurcation and `N`'s case 6.
fn cai_double_charge_correction(plan: &Plan<'_>, c: usize, y_prime: usize) -> f64 {
    (1.0 - plan.lambda_) * plan.log_cai_dekacal_per_pos[c][y_prime]
}

/// Run a candidate generator and reduce the stream via `min`. The
/// generator emits through the supplied `&mut dyn FnMut(Candidate)` so
/// the reduction is uniform across the three multi-loop recurrences.
fn min_reduce<G>(gen: G) -> f64
where
    G: FnOnce(&mut dyn FnMut(Candidate)),
{
    let mut best = INF;
    gen(&mut |c: Candidate| {
        if c.value < best {
            best = c.value;
        }
    });
    best
}

// ---------------------------------------------------------------------------
// E_m — multi-loop closure entry
// ---------------------------------------------------------------------------

/// Bundled per-cell context: outer cell index plus precomputed
/// `A_dekacal = MLclosing + ml_closing(σ(a,s), σ(b,t))` — the closing-
/// pair flank-aware multi-loop constant. Sequence-dependent only
/// through the closing pair; the four cases all charge the same
/// `A_dekacal` so it's computed once per cell.
#[derive(Debug, Clone, Copy)]
struct EmSite {
    idx: CellIdx,
    a_dekacal: i32,
}

/// One E_m candidate's specification: which case, which inner `M` cell
/// to read, the boundary-cross codon assignments (`0` for sides that
/// don't cross), and the CAI penalty for the boundary-cross AAs.
#[derive(Debug, Clone, Copy)]
struct EmSpec {
    case: u8,
    inner_idx: CellIdx,
    x_prime: usize,
    y_prime: usize,
    cai: f64,
}

/// Read `M[spec.inner_idx]` and form one [`Candidate`] paying
/// `site.a_dekacal` plus the spec's boundary-cross CAI penalty.
fn em_build_candidate(plan: &Plan<'_>, site: EmSite, spec: EmSpec) -> Option<Candidate> {
    let inner = read_finite(&plan.tables.m, plan.tables.flat_index(spec.inner_idx))?;
    Some(Candidate {
        value: inner + brd_value(plan.lambda_, site.a_dekacal, spec.cai),
        trace: Trace::MultiLoop(MultiLoopTrace {
            case: spec.case,
            x_prime: spec.x_prime,
            y_prime: spec.y_prime,
            inner_idx: spec.inner_idx,
        }),
    })
}

/// Which codon boundaries the outer multi-loop closure crosses inward.
/// Cases 2 / 3 cross exactly one side; case 4 crosses both.
#[derive(Debug, Clone, Copy)]
enum EmSide {
    Left,
    Right,
}

fn em_case_1<F: FnMut(Candidate)>(plan: &Plan<'_>, site: EmSite, emit: &mut F) {
    let (a, b, s, t, x, y) = site.idx;
    let within_cell = s < 2 && t > 0;
    if a >= b || !within_cell {
        return;
    }
    let spec = EmSpec {
        case: 1,
        inner_idx: (a, b, s + 1, t - 1, x, y),
        x_prime: 0,
        y_prime: 0,
        cai: 0.0,
    };
    if let Some(cand) = em_build_candidate(plan, site, spec) {
        emit(cand);
    }
}

fn em_one_side_spec(idx: CellIdx, side: EmSide, pinned: usize, cai: f64) -> EmSpec {
    let (a, b, s, t, x, y) = idx;
    let (inner_idx, x_prime, y_prime, case) = match side {
        EmSide::Left => ((a + 1, b, 0, t - 1, pinned, y), pinned, 0, 2),
        EmSide::Right => ((a, b - 1, s + 1, 2, x, pinned), 0, pinned, 3),
    };
    EmSpec {
        case,
        inner_idx,
        x_prime,
        y_prime,
        cai,
    }
}

/// Side-specific `(pin_aa, cai)` for E_m's one-boundary-cross cases.
/// Returns `None` when the cell doesn't admit this side's cross.
fn em_one_side_pin(plan: &Plan<'_>, idx: CellIdx, side: EmSide) -> Option<(usize, f64)> {
    let (a, b, s, t, x, y) = idx;
    if a + 1 >= b {
        return None;
    }
    match side {
        EmSide::Left if s == 2 && t > 0 => Some((a + 1, plan.log_cai_dekacal_per_pos[a][x])),
        EmSide::Right if s < 2 && t == 0 => Some((b - 1, plan.log_cai_dekacal_per_pos[b][y])),
        _ => None,
    }
}

/// E_m cases 2 / 3 — exactly one codon boundary crosses inward.
fn em_one_side<F: FnMut(Candidate)>(plan: &Plan<'_>, site: EmSite, side: EmSide, emit: &mut F) {
    let Some((pin_aa, cai)) = em_one_side_pin(plan, site.idx, side) else {
        return;
    };
    let (a, b, _s, _t, x, y) = site.idx;
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for pinned in consistent_choices(plan, pin_aa, &base) {
        let spec = em_one_side_spec(site.idx, side, pinned, cai);
        if let Some(cand) = em_build_candidate(plan, site, spec) {
            emit(cand);
        }
    }
}

/// E_m case 4 — both boundaries cross. Charges CAI at both `a` and `b`
/// (paper prints the two `g` terms without shared parens; the corrected form
/// is used).
fn em_case_4<F: FnMut(Candidate)>(plan: &Plan<'_>, site: EmSite, emit: &mut F) {
    let (a, b, s, t, x, y) = site.idx;
    let both_cross = s == 2 && t == 0;
    if a + 2 >= b || !both_cross {
        return;
    }
    let log_cai = &plan.log_cai_dekacal_per_pos;
    let cai = log_cai[a][x] + log_cai[b][y];
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for x_prime in consistent_choices(plan, a + 1, &base) {
        let extended: [(usize, usize); 3] = [(a, x), (b, y), (a + 1, x_prime)];
        for y_prime in consistent_choices(plan, b - 1, &extended) {
            let spec = EmSpec {
                case: 4,
                inner_idx: (a + 1, b - 1, 0, 2, x_prime, y_prime),
                x_prime,
                y_prime,
                cai,
            };
            if let Some(cand) = em_build_candidate(plan, site, spec) {
                emit(cand);
            }
        }
    }
}

/// Yield every multi-loop closure candidate for cell `idx`.
///
/// Reads `M[...]` at the per-case inner cell; pays `MLclosing + ml_closing`
/// once per candidate plus the boundary-cross CAI penalty.
///
/// The `scratch` parameter is unused — the closing pair is read directly
/// from `Plan::codon_lists` via `closing_pair_at` — but the signature is
/// shared with the other E-motif generators so they all dispatch through
/// a single match in `recurrences::extract_motif_trace`.
pub(crate) fn e_m_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    _scratch: &mut Vec<Nucleotide>,
    mut emit: F,
) {
    let Some(closing) = closing_pair_at(plan, idx) else {
        return;
    };
    let site = EmSite {
        idx,
        a_dekacal: plan.ml_closing_const + ml_stem_credit(plan, closing),
    };
    em_case_1(plan, site, &mut emit);
    em_one_side(plan, site, EmSide::Left, &mut emit);
    em_one_side(plan, site, EmSide::Right, &mut emit);
    em_case_4(plan, site, &mut emit);
}

/// Fill `E_m[idx]` via `min` over multi-loop closure candidates.
pub(crate) fn fill_e_m(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let best = min_reduce(|emit| e_m_candidates(plan, idx, scratch, emit));
    let i = plan.tables.flat_index(idx);
    plan.tables.e_m[i] = best;
}

// ---------------------------------------------------------------------------
// M — multi-loop body, single bifurcation
// ---------------------------------------------------------------------------

/// One `(c, t', d, s')` split. Adjacency forces either a within-codon
/// split (`d == c`, `s' == t' + 1`) or a between-codon split
/// (`d == c + 1`, `t' == 2`, `s' == 0`).
#[derive(Debug, Clone, Copy)]
struct MSplit {
    c: usize,
    t_prime: usize,
    d: usize,
    s_prime: usize,
}

#[derive(Debug, Clone, Copy)]
struct MLeftRead {
    idx: CellIdx,
    value: f64,
}

#[derive(Debug, Clone, Copy)]
struct MSite {
    idx: CellIdx,
    split: MSplit,
}

fn m_split_iter(nt_a: usize, nt_b: usize) -> impl Iterator<Item = MSplit> {
    // Split points `k ∈ [nt_a + 1, nt_b - 2]`.
    let lo = nt_a.saturating_add(1);
    let hi = nt_b.saturating_sub(1);
    (lo..hi).map(|k| MSplit {
        c: aa_of_nt(k),
        t_prime: k % 3,
        d: aa_of_nt(k + 1),
        s_prime: (k + 1) % 3,
    })
}

fn m_build_candidate(
    plan: &Plan<'_>,
    site: MSite,
    left: MLeftRead,
    codons: (usize, usize),
) -> Option<Candidate> {
    let (_a, b, _s, t, _x, y) = site.idx;
    let (y_prime, x_prime) = codons;
    let right_idx = (site.split.d, b, site.split.s_prime, t, x_prime, y);
    let right = read_finite(&plan.tables.n, plan.tables.flat_index(right_idx))?;
    let mut value = left.value + right;
    if site.split.c == site.split.d {
        value += cai_double_charge_correction(plan, site.split.c, y_prime);
    }
    Some(Candidate {
        value,
        trace: Trace::M(MTrace {
            c: site.split.c,
            d: site.split.d,
            y_prime,
            x_prime,
            left_idx: left.idx,
            right_idx,
        }),
    })
}

fn m_candidates_at_split<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    split: MSplit,
    emit: &mut F,
) {
    let (a, b, s, _t, x, y) = idx;
    if !(a < split.c && split.d < b) {
        return;
    }
    let site = MSite { idx, split };
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for y_prime in consistent_choices(plan, split.c, &base) {
        let extended: [(usize, usize); 3] = [(a, x), (b, y), (split.c, y_prime)];
        let left_idx = (a, split.c, s, split.t_prime, x, y_prime);
        let Some(left_value) = read_finite(&plan.tables.n, plan.tables.flat_index(left_idx)) else {
            continue;
        };
        let left = MLeftRead {
            idx: left_idx,
            value: left_value,
        };
        for x_prime in consistent_choices(plan, split.d, &extended) {
            if let Some(cand) = m_build_candidate(plan, site, left, (y_prime, x_prime)) {
                emit(cand);
            }
        }
    }
}

/// Yield every bifurcation candidate for the multi-loop body `M[idx]`.
///
/// Per `M = N + N` with the paper's `[s][t']` → `[s][t]` typo fix. The
/// within-codon split (`c == d`) double-charges the shared AA's CAI in
/// the two halves' descents, so the candidate adds
/// `+(1 − λ) · log_cai[c, y']` to undo one charge.
pub(crate) fn m_candidates<F: FnMut(Candidate)>(plan: &Plan<'_>, idx: CellIdx, mut emit: F) {
    let (a, b, s, t, _x, _y) = idx;
    if a >= b {
        return;
    }
    let nt_a = nt_position(a, s);
    let nt_b = nt_position(b, t);
    for split in m_split_iter(nt_a, nt_b) {
        m_candidates_at_split(plan, idx, split, &mut emit);
    }
}

/// Fill `M[idx]` via `min` over single-bifurcation candidates.
pub(crate) fn fill_m(plan: &mut Plan<'_>, idx: CellIdx) {
    let best = min_reduce(|emit| m_candidates(plan, idx, emit));
    let i = plan.tables.flat_index(idx);
    plan.tables.m[i] = best;
}

// ---------------------------------------------------------------------------
// N — multi-loop fragment, at-least-one branch
// ---------------------------------------------------------------------------

/// Specification for one `N` candidate. The traceback walker reads
/// `case` and the inner-cell indices to descend; `aux_codon` /
/// `aux_codon_2` pin codon assignments at the boundary-cross AA
/// (cases 4 / 5) or the shared-AA AAs of the bifurcation (cases 6 / 7).
#[derive(Debug, Clone, Copy)]
struct NSpec {
    case: u8,
    value: f64,
    inner_a: CellIdx,
    inner_b: Option<CellIdx>,
    aux: usize,
    aux2: usize,
}

#[derive(Debug, Clone, Copy)]
struct NLeftRead {
    idx: CellIdx,
    value: f64,
}

fn n_candidate(spec: NSpec) -> Candidate {
    Candidate {
        value: spec.value,
        trace: Trace::N(NTrace {
            case: spec.case,
            inner_idx_a: spec.inner_a,
            inner_idx_b: spec.inner_b,
            aux_codon: spec.aux,
            aux_codon_2: spec.aux2,
        }),
    }
}

/// N case 1 — close a branch via the inner `E` cell, paying the per-branch
/// `ml_stem` credit. No CAI charge: the inner `E` cell already accounts
/// for `log_cai` at AAs `a` and `b` via its own motif bookkeeping.
fn n_branch_closure_candidate(plan: &Plan<'_>, idx: CellIdx) -> Option<Candidate> {
    let e_value = read_finite(&plan.tables.e, plan.tables.flat_index(idx))?;
    let closing = closing_pair_at(plan, idx)?;
    let stem = ml_stem_credit(plan, closing);
    Some(n_candidate(NSpec {
        case: 1,
        value: e_value + brd_value(plan.lambda_, stem, 0.0),
        inner_a: idx,
        inner_b: None,
        aux: 0,
        aux2: 0,
    }))
}

/// Which codon boundary an `N` unpaired-bump moves through.
#[derive(Debug, Clone, Copy)]
enum NSide {
    Left,
    Right,
}

/// Bundled inputs for one unpaired-bump candidate (cases 2 / 3 / 4 / 5).
/// Cases 2 / 3 have `cai = 0` and `aux = 0`; cases 4 / 5 charge the
/// boundary-cross AA's CAI and carry the new codon assignment in `aux`.
#[derive(Debug, Clone, Copy)]
struct NUnpaired {
    case: u8,
    inner_idx: CellIdx,
    cai: f64,
    aux: usize,
}

/// Build an unpaired-bump candidate (cases 2 / 3 / 4 / 5). Reads `N` at
/// the inner cell, adds `brd_value(λ, MLbase, cai)`, and (for the
/// boundary-crossing cases) carries the new codon assignment in `aux`.
fn n_unpaired_candidate(plan: &Plan<'_>, sweep: NUnpaired) -> Option<Candidate> {
    let inner = read_finite(&plan.tables.n, plan.tables.flat_index(sweep.inner_idx))?;
    Some(n_candidate(NSpec {
        case: sweep.case,
        value: inner + brd_value(plan.lambda_, plan.ml_base_const, sweep.cai),
        inner_a: sweep.inner_idx,
        inner_b: None,
        aux: sweep.aux,
        aux2: 0,
    }))
}

/// N cases 2 / 3 — within-cell unpaired bump on the chosen side.
fn n_within_cell_bump<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    side: NSide,
    emit: &mut F,
) {
    let (a, b, s, t, x, y) = idx;
    if a >= b {
        return;
    }
    let (case, inner_idx) = match side {
        NSide::Left if s < 2 => (2, (a, b, s + 1, t, x, y)),
        NSide::Right if t > 0 => (3, (a, b, s, t - 1, x, y)),
        _ => return,
    };
    let sweep = NUnpaired {
        case,
        inner_idx,
        cai: 0.0,
        aux: 0,
    };
    if let Some(cand) = n_unpaired_candidate(plan, sweep) {
        emit(cand);
    }
}

fn n_cross_inner_idx(idx: CellIdx, side: NSide, pinned: usize) -> CellIdx {
    let (a, b, s, t, x, y) = idx;
    match side {
        NSide::Left => (a + 1, b, 0, t, pinned, y),
        NSide::Right => (a, b - 1, s, 2, x, pinned),
    }
}

/// N cases 4 / 5 — codon-boundary-crossing unpaired bump.
///
/// Case 5 paper typo: inner `t` should be `3` (1-based) = `2` (0-based)
/// — the last nucleotide of codon `b − 1` — not the outer `t = 0`. This
/// code reads `(a, b - 1, s, 2, x, y')`.
fn n_boundary_cross<F: FnMut(Candidate)>(plan: &Plan<'_>, idx: CellIdx, side: NSide, emit: &mut F) {
    let (a, b, s, t, x, y) = idx;
    if a + 1 >= b {
        return;
    }
    let (case, pin_aa, cai, other_pin) = match side {
        NSide::Left if s == 2 => (4, a + 1, plan.log_cai_dekacal_per_pos[a][x], (b, y)),
        NSide::Right if t == 0 => (5, b - 1, plan.log_cai_dekacal_per_pos[b][y], (a, x)),
        _ => return,
    };
    let base: [(usize, usize); 1] = [other_pin];
    for pinned in consistent_choices(plan, pin_aa, &base) {
        let sweep = NUnpaired {
            case,
            inner_idx: n_cross_inner_idx(idx, side, pinned),
            cai,
            aux: pinned,
        };
        if let Some(cand) = n_unpaired_candidate(plan, sweep) {
            emit(cand);
        }
    }
}

fn n_pair_value(plan: &Plan<'_>, left_idx: CellIdx, right_idx: CellIdx) -> Option<f64> {
    let left = read_finite(&plan.tables.n, plan.tables.flat_index(left_idx))?;
    let right = read_finite(&plan.tables.n, plan.tables.flat_index(right_idx))?;
    Some(left + right)
}

/// One `(c, k)` split for case 6 — enumerates the shared-AA codon `y'`.
fn n_case_6_for_split<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    split: (usize, usize),
    emit: &mut F,
) {
    let (c, k) = split;
    let (a, b, s, t, x, y) = idx;
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for y_prime in consistent_choices(plan, c, &base) {
        let left_idx = (a, c, s, k, x, y_prime);
        let right_idx = (c, b, k + 1, t, y_prime, y);
        let Some(pair) = n_pair_value(plan, left_idx, right_idx) else {
            continue;
        };
        let correction = cai_double_charge_correction(plan, c, y_prime);
        emit(n_candidate(NSpec {
            case: 6,
            value: pair + correction,
            inner_a: left_idx,
            inner_b: Some(right_idx),
            aux: y_prime,
            aux2: 0,
        }));
    }
}

/// N case 6 — within-codon bifurcation at AA `c`, codon-internal split
/// position `k ∈ {0, 1}`. Both halves charge `log_cai[c, y']` along
/// their descent, so the candidate adds `+(1 − λ) · log_cai[c, y']` to
/// undo one charge. Paper prose says `g(w_c, x')`; this code uses `y'` to match
/// the bifurcation's shared-AA codon.
fn n_case_6<F: FnMut(Candidate)>(plan: &Plan<'_>, idx: CellIdx, emit: &mut F) {
    let (a, b, _s, _t, _x, _y) = idx;
    if a + 1 >= b {
        return;
    }
    for c in (a + 1)..b {
        for k in 0..2 {
            n_case_6_for_split(plan, idx, (c, k), emit);
        }
    }
}

/// One `(c, y_prime)` split for case 7 — enumerates `x_prime` at `c + 1`.
fn n_case_7_for_y_prime<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    cy: (usize, usize),
    emit: &mut F,
) {
    let (c, y_prime) = cy;
    let (a, b, s, t, x, y) = idx;
    let extended: [(usize, usize); 3] = [(a, x), (b, y), (c, y_prime)];
    let left_idx = (a, c, s, 2, x, y_prime);
    let Some(left_value) = read_finite(&plan.tables.n, plan.tables.flat_index(left_idx)) else {
        return;
    };
    let left = NLeftRead {
        idx: left_idx,
        value: left_value,
    };
    for x_prime in consistent_choices(plan, c + 1, &extended) {
        let right_idx = (c + 1, b, 0, t, x_prime, y);
        let Some(right) = read_finite(&plan.tables.n, plan.tables.flat_index(right_idx)) else {
            continue;
        };
        emit(n_candidate(NSpec {
            case: 7,
            value: left.value + right,
            inner_a: left.idx,
            inner_b: Some(right_idx),
            aux: y_prime,
            aux2: x_prime,
        }));
    }
}

/// N case 7 — between-codon bifurcation at AAs `c` and `c + 1`. No CAI
/// correction: the two halves pin distinct AAs.
fn n_case_7<F: FnMut(Candidate)>(plan: &Plan<'_>, idx: CellIdx, emit: &mut F) {
    let (a, b, _s, _t, x, y) = idx;
    if a + 2 >= b {
        return;
    }
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for c in (a + 1)..(b - 1) {
        for y_prime in consistent_choices(plan, c, &base) {
            n_case_7_for_y_prime(plan, idx, (c, y_prime), emit);
        }
    }
}

/// Yield every candidate for the seven cases of the `N` recurrence.
///
/// The `scratch` parameter is unused — case 1's closing pair is read
/// directly from `Plan::codon_lists` via `closing_pair_at` — but the
/// signature is shared with the `fill_n` driver so callers can pass the
/// outer scratch through uniformly with the other multi-loop fillers.
pub(crate) fn n_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    _scratch: &mut Vec<Nucleotide>,
    mut emit: F,
) {
    let (a, b, _s, _t, _x, _y) = idx;
    if a >= b {
        return;
    }
    if let Some(cand) = n_branch_closure_candidate(plan, idx) {
        emit(cand);
    }
    n_within_cell_bump(plan, idx, NSide::Left, &mut emit);
    n_within_cell_bump(plan, idx, NSide::Right, &mut emit);
    n_boundary_cross(plan, idx, NSide::Left, &mut emit);
    n_boundary_cross(plan, idx, NSide::Right, &mut emit);
    n_case_6(plan, idx, &mut emit);
    n_case_7(plan, idx, &mut emit);
}

/// Fill `N[idx]` via `min` over the seven N candidates.
pub(crate) fn fill_n(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let best = min_reduce(|emit| n_candidates(plan, idx, scratch, emit));
    let i = plan.tables.flat_index(idx);
    plan.tables.n[i] = best;
}
