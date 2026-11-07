//! Multi-loop DP cells (M, M1) and the V multi-loop closing case.
//!
//! Factors the affine multi-loop scoring across four interacting
//! cells so each branch only pays its share as it gets incorporated.
//! See [`crate::loops::multi`] for the closed-form (non-DP) total. The
//! cell semantics:
//!
//! | Cell           | Meaning                                                                                              |
//! |----------------|------------------------------------------------------------------------------------------------------|
//! | `M[i, j]`      | MFE on `seq[i..=j]` of a *partial* multi-loop interior — ≥1 branch, optional flanks on either side   |
//! | `M1[i, j]`     | MFE on `seq[i..=j]` of *exactly one* branch + trailing unpaired (no leading unpaired)                |
//! | `V_multi(i,j)` | V's multi-loop closing case at pair `(i, j)`: closing-stem cost + best `M + M1` decomposition inside |
//!
//! ## Recurrences
//!
//! ```text
//! M1[i, j] = min(
//!     V[i, j] + intern_37 + au_term((i, j)),    // exactly this branch, no trailing
//!     M1[i, j-1] + base_37,                     // extend with unpaired on right
//! )
//!
//! M[i, j] = min(
//!     V[i, j] + intern_37 + au_term((i, j)),    // single branch, no flanks
//!     M[i+1, j] + base_37,                      // unpaired on left
//!     M[i, j-1] + base_37,                      // unpaired on right
//!     min over k in [i+1, j-1]:
//!         M[i, k] + M1[k+1, j],                 // bifurcation: ≥1 branch + exactly 1 more
//! )
//!
//! V_multi(i, j) = closing_37 + intern_37 + au_term((i, j))
//!               + min over k in [i+1, j-2]:
//!                   M[i+1, k] + M1[k+1, j-1]
//! ```
//!
//! Where `au_term(p) = terminal_au_37` if `p.applies_terminal_au()`
//! (anything other than `CG`/`GC`), else `0`.
//!
//! ## Why this factoring matches the per-loop evaluator
//!
//! The closing pair contributes `closing_37 + intern_37 + au_term`
//! once (at the V_multi cell). Each branch contributes
//! `intern_37 + au_term` once (when it joins via M's "single branch"
//! base case or M1's "exactly this branch"). Each unpaired position
//! contributes `base_37` once (when it's nibbled by an unpaired-flank
//! case in M or M1). Summed over the DP path, this exactly equals
//! the affine total — see the per-loop evaluator at
//! [`crate::loops::multi`] for the closed-form sanity check.
//!
//! ## Recurrence notes
//!
//! - **V multi-loop closing at `dangles=0`**: the closing pair pays
//!   `closing_37` plus the multi-loop stem cost `intern_37 + au_term`,
//!   on top of the best two-part split of the interior. The split is
//!   written `M[i+1, k] + M1[k+1, j-1]`; an `M + M` split gives the
//!   same MFE (any structure with ≥2 branches admits a "rightmost
//!   branch in M1" decomposition).
//! - **M recurrence at `dangles=0`**: single-branch, right-extend,
//!   left-extend, and bifurcation `min over k: M[i, k] + M1[k+1, j]`.
//! - **M1 recurrence at `dangles=0`**: the two cases above (single
//!   branch, right-extend) without left-extend or bifurcation — i.e.,
//!   M1's "exactly one branch + trailing unpaired" semantics.
//! - **Canonical traceback order (V multi-loop closing)**: `k` from
//!   `i+1` upward, first equal split wins (None-handled empty cells set
//!   the effective lower bound).
//! - **Canonical traceback order (M)**: right-nibble unpaired,
//!   left-nibble unpaired, single-component (V[i,j] + stem cost), then
//!   bifurcation (smallest split first). M1's traceback uses the
//!   M1-specific subset (right-nibble unpaired, then single branch).

use super::fill::{traceback_v, FillCtx, OuterClose, Span};
use super::tables::{TriangleTable, MIN_HAIRPIN_LOOP};
use crate::loops::{pair_from_nucleotides, Energy};
use crate::params::{BasePair, Params};

/// Branchless-ish zip-add-min over two stride-1 `i32` slices with the
/// `TriangleTable::INF` sentinel for "no valid structure". Returns
/// `INF` if no `(a[i], b[i])` pair has both finite.
///
/// Equal-length stride-1 slices in, single `i32` out, no `Option`, no
/// `?`-branches, so the loop body compiles to a short scalar sequence
/// with conditional selects for the INF-aware min.
///
/// Caller invariants:
/// - `a.len() == b.len()`. Debug-asserted, not enforced at runtime.
/// - Slice elements are either a finite energy in dekacalories per mol
///   or [`TriangleTable::INF`].
#[inline]
pub fn zip_add_min_inf(a: &[i32], b: &[i32]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut best = TriangleTable::INF;
    let n = a.len().min(b.len());
    for i in 0..n {
        let x = a[i];
        let y = b[i];
        if x != TriangleTable::INF && y != TriangleTable::INF {
            let s = x + y;
            if s < best {
                best = s;
            }
        }
    }
    best
}

/// Smallest span `(j - i)` that can fit two branches plus the closing
/// pair: `i` + branch1 (≥`MIN_HAIRPIN_LOOP + 2` = 5 nt) + branch2
/// (≥5 nt) + `j`. Below this, V's multi-loop closing case is `None`.
pub(super) const MIN_V_MULTI_SPAN: usize = 2 * (MIN_HAIRPIN_LOOP + 2) + 1;

/// Rolling row scratch for the multi-loop bifurcation sites.
///
/// `fmi[k]` holds `M[i, k]` (raw `i32` with [`TriangleTable::INF`]
/// sentinel) for the *current* outer `i` being filled. `dmli[k]` holds
/// `M[i+1, k]` from the previous outer-`i` iteration.
///
/// Both buffers are length `n` and initialized to `INF`. After each
/// outer-`i` iteration finishes filling its inner-`j` row,
/// [`Self::rotate`] swaps the buffers (the just-filled `fmi` becomes
/// the new `dmli`) and resets the freshly-vacated `fmi` to `INF`.
///
/// Lets the cubic-loop bifurcation kernel ([`zip_add_min_inf`]) read
/// `M`'s row pattern (varying `k`, fixed outer `i` or `i+1`) as a
/// stride-1 slice without paying column-major-triangular's natural
/// stride-growing-with-`k` cost on row-pattern reads.
pub(super) struct BifurcationScratch {
    pub fmi: Vec<i32>,
    pub dmli: Vec<i32>,
}

impl BifurcationScratch {
    pub fn new(n: usize) -> Self {
        Self {
            fmi: vec![TriangleTable::INF; n],
            dmli: vec![TriangleTable::INF; n],
        }
    }

    /// Promote the just-finished outer-`i` row to `dmli` and reset the
    /// freshly-vacated `fmi` to `INF`. Call once per outer-`i` after
    /// the inner-`j` loop completes; the next, smaller outer-`i` will
    /// then read `M[i+1, k]` from the rotated `dmli`.
    pub fn rotate(&mut self) {
        std::mem::swap(&mut self.fmi, &mut self.dmli);
        self.fmi.fill(TriangleTable::INF);
    }

    /// Write `M[outer_i, j]` into the current-outer `fmi`. `None`
    /// writes the `INF` sentinel.
    pub fn write_fmi(&mut self, j: usize, m: Option<Energy>) {
        self.fmi[j] = m.map_or(TriangleTable::INF, |Energy(raw)| raw);
    }
}

/// M1[i, j] — MFE of exactly one multi-loop branch on `seq[i..=j]`
/// with trailing unpaired allowed on the right (none on the left).
pub(super) fn compute_m1(ctx: &FillCtx<'_>, span: Span) -> Option<Energy> {
    let mut best = ml_branch_total(ctx, span);
    super::fill::update_min(&mut best, m1_extend_right(ctx, span));
    best
}

/// M[i, j] — MFE of a multi-loop interior with ≥1 branch on
/// `seq[i..=j]`, optionally with unpaired flanks on either side.
pub(super) fn compute_m(
    ctx: &FillCtx<'_>,
    span: Span,
    scratch: &BifurcationScratch,
) -> Option<Energy> {
    let mut best = ml_branch_total(ctx, span);
    super::fill::update_min(&mut best, m_extend_left(ctx, span));
    super::fill::update_min(&mut best, m_extend_right(ctx, span));
    super::fill::update_min(&mut best, m_bifurcation(ctx, span, scratch));
    best
}

/// V's multi-loop closing case at pair `(i, j) = closing`. The closing
/// pair pays `closing_37 + intern_37 + au_term` once; the rest of the
/// loop is the best `M[i+1, k] + M1[k+1, j-1]` split.
pub(super) fn compute_v_multi_close(
    ctx: &FillCtx<'_>,
    span: Span,
    closing: BasePair,
    scratch: &BifurcationScratch,
) -> Option<Energy> {
    let inner = best_multi_split(ctx, span, scratch)?;
    Some(inner + multi_closing_stem_cost(ctx.params, closing))
}

/// Multi-loop closing traceback: dispatch from V[i, j]'s walk. Returns
/// `true` iff the multi-loop closing case matched and recursion was
/// performed. Smallest `k` first wins.
pub(super) fn traceback_v_multi_close(
    ctx: &FillCtx<'_>,
    outer: OuterClose,
    target: Energy,
    structure: &mut [u8],
) -> bool {
    let Some(split_k) = find_v_multi_split(ctx, outer, target) else {
        return false;
    };
    traceback_m(
        ctx,
        Span {
            i: outer.span.i + 1,
            j: split_k,
        },
        structure,
    );
    traceback_m1(
        ctx,
        Span {
            i: split_k + 1,
            j: outer.span.j - 1,
        },
        structure,
    );
    true
}

/// Walk M[i, j] back to its source. Order at `dangles=0`: right-nibble
/// unpaired, left-nibble unpaired, single-branch (V + stem cost), then
/// bifurcation.
pub(super) fn traceback_m(ctx: &FillCtx<'_>, span: Span, structure: &mut [u8]) {
    let target = ctx.tables.m.get(span.i, span.j).expect("M[i, j] is filled");
    let case = classify_m(ctx, span, target).unwrap_or_else(|| {
        unreachable!(
            "M[{}, {}] traceback failed: no case matched target={}",
            span.i,
            span.j,
            target.dekacal(),
        )
    });
    apply_m_case(ctx, span, case, structure);
}

/// Walk M1[i, j] back to its source. Order: right-nibble unpaired
/// first, then single branch.
pub(super) fn traceback_m1(ctx: &FillCtx<'_>, span: Span, structure: &mut [u8]) {
    let target = ctx
        .tables
        .m1
        .get(span.i, span.j)
        .expect("M1[i, j] is filled");
    let case = classify_m1(ctx, span, target).unwrap_or_else(|| {
        unreachable!(
            "M1[{}, {}] traceback failed: no case matched target={}",
            span.i,
            span.j,
            target.dekacal(),
        )
    });
    apply_m1_case(ctx, span, case, structure);
}

// ---- recurrence helpers ----------------------------------------------------

fn ml_branch_total(ctx: &FillCtx<'_>, span: Span) -> Option<Energy> {
    if span.j - span.i < MIN_HAIRPIN_LOOP + 1 {
        return None;
    }
    let v = ctx.tables.v.get(span.i, span.j)?;
    let pair = pair_from_nucleotides(ctx.seq[span.i], ctx.seq[span.j]);
    if matches!(pair, BasePair::NN) {
        return None;
    }
    Some(v + branch_stem_cost(ctx.params, pair))
}

fn m1_extend_right(ctx: &FillCtx<'_>, span: Span) -> Option<Energy> {
    if span.j == span.i {
        return None;
    }
    ctx.tables
        .m1
        .get(span.i, span.j - 1)
        .map(|e| e + ml_base_cost(ctx.params))
}

fn m_extend_left(ctx: &FillCtx<'_>, span: Span) -> Option<Energy> {
    if span.j == span.i {
        return None;
    }
    ctx.tables
        .m
        .get(span.i + 1, span.j)
        .map(|e| e + ml_base_cost(ctx.params))
}

fn m_extend_right(ctx: &FillCtx<'_>, span: Span) -> Option<Energy> {
    if span.j == span.i {
        return None;
    }
    ctx.tables
        .m
        .get(span.i, span.j - 1)
        .map(|e| e + ml_base_cost(ctx.params))
}

/// `min over k in k_range: row_scratch[k] + m1_col[k+1]`. Returns
/// `None` when no `k` produces a finite sum. Shared kernel-call shape
/// for both bifurcation sites: [`m_bifurcation`] supplies
/// `(scratch.fmi, m1.raw_col(j), (i+1)..j)`; [`best_multi_split`]
/// supplies `(scratch.dmli, m1.raw_col(j-1), (i+1)..(j-1))`.
#[inline]
fn flat_bifurcation_min(
    row_scratch: &[i32],
    m1_col: &[i32],
    k_range: std::ops::Range<usize>,
) -> Option<Energy> {
    if k_range.is_empty() {
        return None;
    }
    let row = &row_scratch[k_range.clone()];
    let column = &m1_col[(k_range.start + 1)..(k_range.end + 1)];
    let raw = zip_add_min_inf(row, column);
    if raw < TriangleTable::INF {
        Some(Energy(raw))
    } else {
        None
    }
}

/// `min over k in [i+1, j-1]: M[i, k] + M1[k+1, j]`.
///
/// `M[i, k]` from `scratch.fmi` (current outer-`i` row, populated as
/// inner-`j` advances); `M1[k+1, j]` column-wise from `M1`'s `j` slice.
/// Uses the `M + M1` form; an `M + M` split gives the same MFE for
/// `≥2`-branch decompositions.
fn m_bifurcation(ctx: &FillCtx<'_>, span: Span, scratch: &BifurcationScratch) -> Option<Energy> {
    let m1_col = ctx.tables.m1.raw_col(span.j);
    flat_bifurcation_min(&scratch.fmi, m1_col, (span.i + 1)..span.j)
}

/// Per-`k` cell read used by traceback only. Fill uses
/// [`m_bifurcation`]'s flat-kernel path; traceback runs once per fold
/// and consults the already-filled tables directly.
fn m_split_total(ctx: &FillCtx<'_>, span: Span, k: usize) -> Option<Energy> {
    let left = ctx.tables.m.get(span.i, k)?;
    let right = ctx.tables.m1.get(k + 1, span.j)?;
    Some(left + right)
}

/// `min over k in [i+1, j-2]: M[i+1, k] + M1[k+1, j-1]`.
///
/// `M[i+1, k]` from `scratch.dmli` (previous outer-`i` row); `M1[k+1, j-1]`
/// column-wise from `M1`'s `j-1` slice: the interior split of V's
/// multi-loop closing at `dangles=0`.
fn best_multi_split(ctx: &FillCtx<'_>, span: Span, scratch: &BifurcationScratch) -> Option<Energy> {
    if span.j < span.i + MIN_V_MULTI_SPAN {
        return None;
    }
    let m1_col = ctx.tables.m1.raw_col(span.j - 1);
    flat_bifurcation_min(&scratch.dmli, m1_col, (span.i + 1)..(span.j - 1))
}

fn multi_split_total(ctx: &FillCtx<'_>, span: Span, k: usize) -> Option<Energy> {
    let left = ctx.tables.m.get(span.i + 1, k)?;
    let right = ctx.tables.m1.get(k + 1, span.j - 1)?;
    Some(left + right)
}

// ---- traceback case classification + dispatch -----------------------------

/// Which recurrence case M[i, j] decomposed under. The DP considers
/// cases in this order; the first equality with `target` wins.
#[derive(Debug, Clone, Copy)]
enum MCase {
    ExtendRight,
    ExtendLeft,
    SingleBranch,
    Bifurcate(usize),
}

/// Which recurrence case M1[i, j] decomposed under. M1 only has the
/// trailing-unpaired and single-branch cases (no left-extend, no
/// bifurcation) — that's the "exactly one branch" semantics.
#[derive(Debug, Clone, Copy)]
enum M1Case {
    ExtendRight,
    SingleBranch,
}

fn classify_m(ctx: &FillCtx<'_>, span: Span, target: Energy) -> Option<MCase> {
    if m_extend_right(ctx, span) == Some(target) {
        return Some(MCase::ExtendRight);
    }
    if m_extend_left(ctx, span) == Some(target) {
        return Some(MCase::ExtendLeft);
    }
    if ml_branch_total(ctx, span) == Some(target) {
        return Some(MCase::SingleBranch);
    }
    ((span.i + 1)..span.j)
        .find(|&k| m_split_total(ctx, span, k) == Some(target))
        .map(MCase::Bifurcate)
}

fn apply_m_case(ctx: &FillCtx<'_>, span: Span, case: MCase, structure: &mut [u8]) {
    match case {
        MCase::ExtendRight => traceback_m(
            ctx,
            Span {
                i: span.i,
                j: span.j - 1,
            },
            structure,
        ),
        MCase::ExtendLeft => traceback_m(
            ctx,
            Span {
                i: span.i + 1,
                j: span.j,
            },
            structure,
        ),
        MCase::SingleBranch => emit_single_branch(ctx, span, structure),
        MCase::Bifurcate(k) => {
            traceback_m(ctx, Span { i: span.i, j: k }, structure);
            traceback_m1(
                ctx,
                Span {
                    i: k + 1,
                    j: span.j,
                },
                structure,
            );
        }
    }
}

fn classify_m1(ctx: &FillCtx<'_>, span: Span, target: Energy) -> Option<M1Case> {
    if m1_extend_right(ctx, span) == Some(target) {
        return Some(M1Case::ExtendRight);
    }
    if ml_branch_total(ctx, span) == Some(target) {
        return Some(M1Case::SingleBranch);
    }
    None
}

fn apply_m1_case(ctx: &FillCtx<'_>, span: Span, case: M1Case, structure: &mut [u8]) {
    match case {
        M1Case::ExtendRight => traceback_m1(
            ctx,
            Span {
                i: span.i,
                j: span.j - 1,
            },
            structure,
        ),
        M1Case::SingleBranch => emit_single_branch(ctx, span, structure),
    }
}

/// Find the smallest `k` such that V[i, j]'s multi-loop closing is
/// realized by `M[i+1, k] + M1[k+1, j-1] + closing-stem-cost == target`.
/// Returns `None` if `(i, j)` is too short for two branches or no split
/// matches (caller treats this as "not the multi-loop case").
fn find_v_multi_split(ctx: &FillCtx<'_>, outer: OuterClose, target: Energy) -> Option<usize> {
    if outer.span.j < outer.span.i + MIN_V_MULTI_SPAN {
        return None;
    }
    let stem_cost = multi_closing_stem_cost(ctx.params, outer.pair);
    ((outer.span.i + 1)..(outer.span.j - 1)).find(
        |&k| matches!(multi_split_total(ctx, outer.span, k), Some(s) if s + stem_cost == target),
    )
}

fn emit_single_branch(ctx: &FillCtx<'_>, span: Span, structure: &mut [u8]) {
    structure[span.i] = b'(';
    structure[span.j] = b')';
    traceback_v(ctx, span, structure);
}

// ---- per-stem / per-base cost helpers --------------------------------------

pub(super) fn branch_stem_cost(params: &Params, pair: BasePair) -> Energy {
    Energy(params.ml.intern_37 + per_stem_terminal_au(params, pair))
}

pub(super) fn multi_closing_stem_cost(params: &Params, closing: BasePair) -> Energy {
    Energy(params.ml.closing_37 + params.ml.intern_37 + per_stem_terminal_au(params, closing))
}

fn per_stem_terminal_au(params: &Params, pair: BasePair) -> i32 {
    if pair.applies_terminal_au() {
        params.misc.terminal_au_37
    } else {
        0
    }
}

pub(super) fn ml_base_cost(params: &Params) -> Energy {
    Energy(params.ml.base_37)
}
