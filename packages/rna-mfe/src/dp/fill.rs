//! Outer recurrence for the Zuker DP tables.
//!
//! V leaf cases (hairpin / stack / bulge / internal closings), the
//! non-bifurcation parts of W, and the simple exterior F recurrence
//! live here. The M / M1 tables and the V multi-loop closing case are
//! filled in [`super::multi`]; this module wires them in.
//!
//! # Vienna source citations (v2.7.0)
//!
//! - **Outer fill order** (decreasing `i`, increasing `j`):
//!   `mfe/mfe.c::fill_arrays` lines 478–510. The fill mirrors Vienna's order
//!   `for i in (length-1)..=1` outer, `for j in (i+1)..=length` inner.
//!   At outer `i`, all `(i', j')` with `i' > i` are already filled
//!   (from previous outer iterations); within the inner-`j` loop, all
//!   `(i, j')` with `j' < j` are filled (from previous inner steps).
//!   This shape lets the multi-loop bifurcation kernels read `M`'s
//!   row pattern from a rolling per-outer-`i` scratch buffer rather
//!   than as stride-growing-with-`k` table reads (see
//!   [`super::multi::BifurcationScratch`]).
//!
//! - **V cell decomposition**: `mfe/mfe.c::decompose_pair` lines
//!   4312–4371. Four leaf cases compose: `vrna_eval_hairpin`
//!   (line 4336), `vrna_mfe_multibranch_loop_fast` (line 4340), and
//!   `vrna_mfe_internal` (line 4350 — covers stack + bulge + internal
//!   in a single dispatch).
//!
//! - **Internal-loop iteration shape**: `mfe/mfe_internal.c::mfe_internal_loop`
//!   lines 554–611. Stacks and bulges (lines 554–566) dispatch via
//!   `mfe_stacks` and `mfe_bulges`; the proper-internal-loop branch
//!   (lines 567–611) iterates outer `l` downward from `j-2` and inner
//!   `k` upward, with the `u1 + u2 <= MAXLOOP` cap via
//!   `last_k + u2 > i + 1 + MAXLOOP` at line 573. With `l` fixed per
//!   outer iteration, `V[k, l]` reads for varying `k` are stride-1 in
//!   the column-major layout (see
//!   [`super::tables::TriangleTable::raw_col`]). The per-`l` kernel is
//!   [`inner_min_at_l`]; traceback walks the same span set via
//!   [`inner_pair_candidates`].
//!
//! - **Minimum hairpin size**: `params/constants.h::TURN = 3`
//!   (referenced via [`super::tables::MIN_HAIRPIN_LOOP`]).
//!
//! - **Maximum bulge/internal cutoff**: `params/constants.h::MAXLOOP = 30`
//!   (referenced via [`super::tables::MAX_LOOP_SIZE`]).
//!
//! - **F-table simple recurrence at `dangles=0`**:
//!   `mfe/mfe_exterior.c::decompose_f5_ext_stem_d0` lines 263–284 plus
//!   `reduce_f5_up` lines 381–430. The recurrence reduces to
//!   `F[j] = min(F[j-1], min over i: F[i-1] + V[i, j-1] + ext_stem(i,j-1))`
//!   where `ext_stem(i, k) = (pair(i, k) applies_terminal_au ? terminal_au_37 : 0)`
//!   per `vrna_E_exterior_stem` at `dangles=0` (`get_stem_contributions_d0`
//!   line 474).

use super::tables::{DpTables, TriangleTable, MAX_LOOP_SIZE, MIN_HAIRPIN_LOOP};
use crate::loops::{
    bulge_energy, hairpin_energy, internal_energy, pair_from_nucleotides, stack_energy, BulgeLoop,
    Energy, HairpinLoop, InternalLoop, Nucleotide, StackPair,
};
use crate::params::{BasePair, Params};

/// Fill all five DP tables for `seq`. The public `fold` entry point in
/// crate root then walks them.
///
/// Per `(i, j)` cell, the order is V → M1 → M → W. V's multi-loop
/// closing case reads M / M1 cells with strictly smaller spans (filled
/// in earlier `len` iterations); M1[i, j] reads V[i, j] (computed immediately before)
/// plus M1[i, j-1] (smaller span); M[i, j] reads V[i, j] plus M / M1
/// cells of smaller spans; W is filled last.
pub fn fill_dp_tables(params: &Params, seq: &[Nucleotide]) -> DpTables {
    let mut tables = DpTables::empty(seq.len());
    if seq.len() >= MIN_HAIRPIN_LOOP + 2 {
        fill_inner_tables(params, seq, &mut tables);
    }
    // Sequences too short for any pair short-circuit to F[j] = 0 for
    // all j, mirroring Vienna's `mfe.c:466` constraint.
    fill_f_table(params, seq, &mut tables);
    tables
}

/// Fill all five DP tables for `seq`, which differs from the sequence
/// `base` was filled for only at positions `lo..=hi`.
///
/// A cell `(i, j)` reads only `seq[i..=j]`, so only the cells with
/// `i <= hi` and `j >= lo` can change: the rest are copied from `base`. Rows
/// above `hi + 1` are skipped outright; row `hi + 1` and the copied cells of
/// the recomputed rows still write their `M` into the bifurcation scratch,
/// which the recomputed cells read. `F[j]` reads cells ending at `j - 1`, so
/// it is refilled from `j = lo + 1` on. A change near the 5' end recomputes
/// about `2 (hi + 1) / n` of the triangle, one near the 3' end about
/// `2 (n - lo) / n`; the result equals [`fill_dp_tables`] on `seq`.
pub fn fill_dp_tables_from(
    params: &Params,
    seq: &[Nucleotide],
    base: &DpTables,
    lo: usize,
    hi: usize,
) -> DpTables {
    let n = seq.len();
    assert_eq!(
        base.n, n,
        "base tables are for a sequence of another length"
    );
    assert!(lo <= hi && hi < n, "change span {lo}..={hi} outside 0..{n}");
    let mut tables = base.clone();
    if n >= MIN_HAIRPIN_LOOP + 2 {
        let mut scratch = super::multi::BifurcationScratch::new(n);
        let top = (hi + 1).min(n - 2);
        for i in (0..=top).rev() {
            for j in (i + 1)..n {
                if i > hi || j < lo {
                    scratch.write_fmi(j, tables.m.get(i, j));
                    continue;
                }
                let span = Span { i, j };
                let v = compute_v(&FillCtx::new(params, seq, &tables), span, &scratch);
                tables.v.set(span.i, span.j, v);
                let m1 = super::multi::compute_m1(&FillCtx::new(params, seq, &tables), span);
                tables.m1.set(span.i, span.j, m1);
                let m =
                    super::multi::compute_m(&FillCtx::new(params, seq, &tables), span, &scratch);
                tables.m.set(span.i, span.j, m);
                scratch.write_fmi(j, m);
                let w = compute_w(&tables, span);
                tables.w.set(span.i, span.j, w);
            }
            scratch.rotate();
        }
    }
    fill_f_from(params, seq, &mut tables, lo + 1);
    tables
}

/// Outer iteration on decreasing `i`, inner on increasing `j`. Per
/// `(i, j)` the order is V → M1 → M → W; after `M[i, j]` lands, its
/// raw value is written into the rolling `fmi` scratch so the next
/// inner-`j` step's bifurcation can read it stride-1. After the inner
/// loop finishes, [`super::multi::BifurcationScratch::rotate`]
/// promotes the just-filled row to `dmli` for the next, smaller
/// outer-`i` to consume in V's multi-loop closing.
fn fill_inner_tables(params: &Params, seq: &[Nucleotide], tables: &mut DpTables) {
    let n = seq.len();
    let mut scratch = super::multi::BifurcationScratch::new(n);
    for i in (0..n.saturating_sub(1)).rev() {
        for j in (i + 1)..n {
            let span = Span { i, j };
            let v = compute_v(&FillCtx::new(params, seq, tables), span, &scratch);
            tables.v.set(span.i, span.j, v);
            let m1 = super::multi::compute_m1(&FillCtx::new(params, seq, tables), span);
            tables.m1.set(span.i, span.j, m1);
            let m = super::multi::compute_m(&FillCtx::new(params, seq, tables), span, &scratch);
            tables.m.set(span.i, span.j, m);
            scratch.write_fmi(j, m);
            let w = compute_w(tables, span);
            tables.w.set(span.i, span.j, w);
        }
        scratch.rotate();
    }
}

/// Read-only view of the inputs every fill / traceback step needs.
pub(super) struct FillCtx<'a> {
    pub(super) params: &'a Params,
    pub(super) seq: &'a [Nucleotide],
    pub(super) tables: &'a DpTables,
}

impl<'a> FillCtx<'a> {
    pub(super) fn new(params: &'a Params, seq: &'a [Nucleotide], tables: &'a DpTables) -> Self {
        Self {
            params,
            seq,
            tables,
        }
    }
}

/// Closed-inclusive sub-region `[i, j]`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Span {
    pub(super) i: usize,
    pub(super) j: usize,
}

/// An outer V-cell candidate: the span and its (canonical) closing pair.
/// Bundling these lets the two-pair search and traceback pass one value
/// instead of two through every layer.
#[derive(Debug, Clone, Copy)]
pub(super) struct OuterClose {
    pub(super) span: Span,
    pub(super) pair: BasePair,
}

/// V[i, j] — minimum free energy on `seq[i..=j]` *given that* `(i, j)`
/// pair. Composes the four leaf closing cases plus the multi-loop
/// closing case (delegated to [`super::multi::compute_v_multi_close`]).
fn compute_v(
    ctx: &FillCtx<'_>,
    span: Span,
    scratch: &super::multi::BifurcationScratch,
) -> Option<Energy> {
    let outer = OuterClose {
        span,
        pair: closing_pair(ctx.seq, span)?,
    };
    let mut best: Option<Energy> = Some(closing_hairpin(ctx, outer));
    update_min(&mut best, closing_two_pair_min(ctx, outer));
    update_min(
        &mut best,
        super::multi::compute_v_multi_close(ctx, span, outer.pair, scratch),
    );
    best
}

/// Canonical closing pair for span `(i, j)`, or `None` if the span is
/// shorter than the minimum hairpin loop or the pair is non-canonical.
fn closing_pair(seq: &[Nucleotide], span: Span) -> Option<BasePair> {
    if span.j - span.i < MIN_HAIRPIN_LOOP + 1 {
        return None;
    }
    match pair_from_nucleotides(seq[span.i], seq[span.j]) {
        BasePair::NN => None,
        canonical => Some(canonical),
    }
}

fn closing_hairpin(ctx: &FillCtx<'_>, outer: OuterClose) -> Energy {
    hairpin_energy(
        ctx.params,
        &HairpinLoop {
            closing: outer.pair,
            loop_seq: &ctx.seq[outer.span.i + 1..outer.span.j],
        },
    )
}

/// Find the minimum `V[k, l] + leaf((i,j),(k,l))` over all inner pairs
/// `(k, l)` with `i < k < l < j`, using a per-column flat kernel.
/// Iterates outer `l` decreasing, inner `k` increasing; `V[k, l]` reads
/// for fixed `l` are stride-1 in the column-major layout via
/// [`inner_min_at_l`]. Mirrors Vienna `mfe/mfe_internal.c::mfe_internal_loop`
/// lines 567–611.
fn closing_two_pair_min(ctx: &FillCtx<'_>, outer: OuterClose) -> Option<Energy> {
    let i = outer.span.i;
    let j = outer.span.j;
    let l_lower = (i + MIN_HAIRPIN_LOOP + 2).max(j.saturating_sub(MAX_LOOP_SIZE + 1));
    let mut best = TriangleTable::INF;
    for l in (l_lower..j).rev() {
        let row_min = inner_min_at_l(ctx, outer, l);
        if row_min < best {
            best = row_min;
        }
    }
    (best != TriangleTable::INF).then_some(Energy(best))
}

/// Per-column inner kernel: fix outer `l`, iterate `k` upward from `i+1`
/// to `last_k`, reading `V[k, l]` via `raw_col(l)` for stride-1 access.
/// Inlines the `(u1, u2)` shape dispatch (stack / bulge / internal) per
/// iteration. `last_k` is bounded by `u1 + u2 <= MAX_LOOP_SIZE` and the
/// minimum-hairpin constraint `l - k >= MIN_HAIRPIN_LOOP + 1`.
#[inline]
#[allow(clippy::needless_range_loop)]
fn inner_min_at_l(ctx: &FillCtx<'_>, outer: OuterClose, l: usize) -> i32 {
    let i = outer.span.i;
    let j = outer.span.j;
    let u2 = j - l - 1;
    let last_k =
        ((i + 1) + MAX_LOOP_SIZE.saturating_sub(u2)).min(l.saturating_sub(MIN_HAIRPIN_LOOP + 1));
    if last_k < i + 1 {
        return TriangleTable::INF;
    }
    let v_col = ctx.tables.v.raw_col(l);
    let s_l = ctx.seq[l];
    let mut best = TriangleTable::INF;
    for k in (i + 1)..=last_k {
        let v = v_col[k];
        if v == TriangleTable::INF {
            continue;
        }
        let inner = pair_from_nucleotides(ctx.seq[k], s_l);
        if matches!(inner, BasePair::NN) {
            continue;
        }
        let leaf = score_two_pair(ctx, outer, inner, Span { i: k, j: l }).0;
        let total = v + leaf;
        if total < best {
            best = total;
        }
    }
    best
}

/// Iterate every `(k, l)` candidate inner span for an outer span,
/// respecting the minimum-hairpin lower bound and the `MAX_LOOP_SIZE`
/// cap. Used by [`find_v_two_pair_split`] (traceback) only — fill uses
/// [`inner_min_at_l`] directly.
///
/// Tie-break: outer `l` decreasing from `j-1`; inner `k` increasing
/// from `i+1`. Mirrors Vienna `mfe/mfe_internal.c::mfe_internal_loop`
/// lines 567–611 outer-`l`-decreasing-first, so the canonical
/// tie-break structure on energy-tied pairs matches Vienna's order.
fn inner_pair_candidates(outer: Span) -> impl Iterator<Item = Span> {
    let i = outer.i;
    let j = outer.j;
    let l_min_hairpin = i + MIN_HAIRPIN_LOOP + 2;
    let l_min_loop = j.saturating_sub(MAX_LOOP_SIZE + 1);
    let l_lower = l_min_hairpin.max(l_min_loop);
    (l_lower..j).rev().flat_map(move |l| {
        let u2 = j - l - 1;
        let cap_loop = (i + 1) + MAX_LOOP_SIZE.saturating_sub(u2);
        let cap_hairpin = l.saturating_sub(MIN_HAIRPIN_LOOP + 1);
        let last_k = cap_loop.min(cap_hairpin);
        ((i + 1)..=last_k).map(move |k| Span { i: k, j: l })
    })
}

/// Total energy `V[k, l] + leaf((i,j),(k,l))`, or `None` if the inner
/// V cell has no valid structure or the inner pair is non-canonical.
/// Used by [`find_v_two_pair_split`] during traceback only; the fill
/// path uses [`inner_min_at_l`] for stride-1 column reads.
#[inline]
fn two_pair_total(ctx: &FillCtx<'_>, outer: OuterClose, inner_span: Span) -> Option<Energy> {
    let inner_v = ctx.tables.v.get(inner_span.i, inner_span.j)?;
    let inner = pair_from_nucleotides(ctx.seq[inner_span.i], ctx.seq[inner_span.j]);
    if matches!(inner, BasePair::NN) {
        return None;
    }
    Some(inner_v + score_two_pair(ctx, outer, inner, inner_span))
}

/// Score the leaf two-pair motif. Dispatches on the unpaired counts
/// `u1 = k - i - 1` and `u2 = j - l - 1`.
pub(super) fn score_two_pair(
    ctx: &FillCtx<'_>,
    outer: OuterClose,
    inner: BasePair,
    inner_span: Span,
) -> Energy {
    let (i, j) = (outer.span.i, outer.span.j);
    let (k, l) = (inner_span.i, inner_span.j);
    let u1 = k - i - 1;
    let u2 = j - l - 1;
    if u1 == 0 && u2 == 0 {
        return stack_energy(
            ctx.params,
            &StackPair {
                outer: outer.pair,
                inner,
            },
        );
    }
    if u1 == 0 || u2 == 0 {
        return bulge_energy(
            ctx.params,
            &BulgeLoop {
                outer: outer.pair,
                inner,
                size: u1.max(u2),
            },
        );
    }
    internal_energy(
        ctx.params,
        &InternalLoop {
            outer: outer.pair,
            inner,
            left: &ctx.seq[i + 1..k],
            right: &ctx.seq[l + 1..j],
        },
    )
}

/// W[i, j] — minimum free energy on `seq[i..=j]` over *any* structure.
/// Two non-bifurcation cases:
///
/// 1. Position `i` unpaired: `W[i+1, j]`
/// 2. Position `j` unpaired: `W[i, j-1]`
///
/// Multi-loop *bifurcation* (two or more independent stems meeting at
/// a multi-loop closing node) is handled inside V via the M / M1 cells
/// in [`super::multi`], not in W.
fn compute_w(tables: &DpTables, span: Span) -> Option<Energy> {
    if span.i == span.j {
        return Some(Energy::ZERO);
    }
    let mut best: Option<Energy> = None;
    update_min(&mut best, tables.w.get(span.i + 1, span.j));
    update_min(&mut best, tables.w.get(span.i, span.j - 1));
    best
}

/// F[j] for `j in 1..=n` — minimum free energy of the prefix `seq[0..j]`
/// (open-end on the right). At `dangles=0`:
///
/// ```text
/// F[j] = min(
///     F[j - 1],                                       // extend by one unpaired nt
///     min over i in 0..j: F[i] + V[i, j-1] + ext_stem(i, j-1)
/// )
/// ```
///
/// Handles arbitrary numbers of exterior stems via the `F` recurrence's
/// `prefix + V + ext_stem` chain. Each F-table entry covers an
/// unbounded prefix; chains of two or more exterior stems are recovered
/// naturally by the recursive `traceback_f` walk.
fn fill_f_table(params: &Params, seq: &[Nucleotide], tables: &mut DpTables) {
    fill_f_from(params, seq, tables, 1);
}

/// Refill `F[j]` for `j >= start` (the cells below `start` are kept).
fn fill_f_from(params: &Params, seq: &[Nucleotide], tables: &mut DpTables, start: usize) {
    for j in start.max(1)..=seq.len() {
        let mut best = tables.f[j - 1];
        let ctx = FillCtx::new(params, seq, tables);
        for i in 0..j {
            update_min(&mut best, prefix_plus_stem(&ctx, i, j - 1));
        }
        tables.f[j] = best;
    }
}

/// `F[i] + V[i, end] + ext_stem(i, end)`, or `None` if either piece is
/// `None`. `end = j - 1` for an F-table cell `F[j]` (open-end-on-right
/// convention).
fn prefix_plus_stem(ctx: &FillCtx<'_>, i: usize, end: usize) -> Option<Energy> {
    let prefix = ctx.tables.f[i]?;
    let stem = ctx.tables.v.get(i, end)?;
    let outer = pair_from_nucleotides(ctx.seq[i], ctx.seq[end]);
    Some(prefix + stem + ext_stem_penalty(ctx.params, outer))
}

/// `vrna_E_exterior_stem(type, -1, -1, P)` reduces at `dangles=0` to
/// the terminal-AU penalty per `mfe/mfe_exterior.c::get_stem_contributions_d0`
/// line 474. `BasePair::applies_terminal_au` is the equivalent of
/// Vienna's `type > 2` predicate on the 0-based encoding.
pub(super) fn ext_stem_penalty(params: &Params, outer: BasePair) -> Energy {
    if outer.applies_terminal_au() {
        Energy(params.misc.terminal_au_37)
    } else {
        Energy::ZERO
    }
}

/// Reduce two `Option<Energy>` to the smaller, in place. `None` is
/// "no valid structure" — any `Some(x)` beats it.
pub(super) fn update_min(best: &mut Option<Energy>, candidate: Option<Energy>) {
    match (*best, candidate) {
        (None, c) => *best = c,
        (Some(b), Some(c)) if c < b => *best = Some(c),
        _ => (),
    }
}

/// Recover the dot-bracket structure that achieves `F[n]`.
///
/// Handles every `dangles=0` input. Tie-break order mirrors Vienna
/// v2.7.0 (cited per-function below). The walk consults the same
/// energies used during fill rather than caching back-pointers,
/// keeping memory usage linear in the table size.
pub fn traceback(params: &Params, seq: &[Nucleotide], tables: &DpTables) -> String {
    let n = seq.len();
    let mut structure = vec![b'.'; n];
    if n >= MIN_HAIRPIN_LOOP + 2 {
        let ctx = FillCtx::new(params, seq, tables);
        traceback_f(&ctx, n, &mut structure);
    }
    String::from_utf8(structure).expect("dots and brackets are ASCII")
}

/// Walk F[j] back to its source: either an unpaired-extension from
/// `F[j-1]` or a split into `F[i] + V[i, j-1] + ext_stem(i, j-1)`.
///
/// Tie-break: unpaired extension wins over any stem split when both
/// equal the target (`backtrack/bt_exterior_f5.c::bt_ext_loop_f5`
/// nibbles unpaired bases before entering the `case 0` stem loop).
/// Among stems, rightmost `i` wins (see [`find_f_split`]).
fn traceback_f(ctx: &FillCtx<'_>, j: usize, structure: &mut [u8]) {
    if j == 0 {
        return;
    }
    let Some(target) = ctx.tables.f[j] else {
        return;
    };
    if extends_unpaired(ctx, j, target) {
        traceback_f(ctx, j - 1, structure);
        return;
    }
    if let Some(i) = find_f_split(ctx, j, target) {
        traceback_f(ctx, i, structure);
        structure[i] = b'(';
        structure[j - 1] = b')';
        traceback_v(ctx, Span { i, j: j - 1 }, structure);
    }
}

/// True if `F[j] == F[j-1]` — i.e., position `j-1` is unpaired in the
/// MFE prefix.
fn extends_unpaired(ctx: &FillCtx<'_>, j: usize, target: Energy) -> bool {
    matches!(ctx.tables.f[j - 1], Some(prev) if prev == target)
}

/// Find the split point `i` such that
/// `F[i] + V[i, j-1] + ext_stem(i, j-1) == target`.
///
/// Tie-break: iterates `i` from `j-1` downward to `0`, matching
/// `backtrack/bt_exterior_f5.c::bt_ext_loop_f5` line 217 at v2.7.0
/// (`for (u = j - 1; u >= 1; u--)` where `u-1` in 1-indexed = the
/// 0-indexed `i` here). Rightmost stem start wins on ties.
fn find_f_split(ctx: &FillCtx<'_>, j: usize, target: Energy) -> Option<usize> {
    (0..j)
        .rev()
        .find(|&i| prefix_plus_stem(ctx, i, j - 1) == Some(target))
}

/// Walk V[i, j] back to its source. Dispatches across the four leaf
/// closing cases (hairpin / stack / bulge / internal) and the
/// multi-loop closing (delegated to
/// [`super::multi::traceback_v_multi_close`]).
///
/// Tie-break: hairpin first, then two-pair (stack/bulge/internal),
/// then multi-loop closing. Matches `mfe/mfe.c` lines 4254–4284 at
/// v2.7.0 (`vrna_bt_hairpin` → `vrna_bt_internal_loop` →
/// `vrna_bt_multibranch_loop`).
pub(super) fn traceback_v(ctx: &FillCtx<'_>, span: Span, structure: &mut [u8]) {
    let outer = OuterClose {
        span,
        pair: pair_from_nucleotides(ctx.seq[span.i], ctx.seq[span.j]),
    };
    let target = ctx.tables.v.get(span.i, span.j).expect("V[i,j] is filled");
    if closing_hairpin(ctx, outer) == target {
        return;
    }
    if let Some(inner_span) = find_v_two_pair_split(ctx, outer, target) {
        structure[inner_span.i] = b'(';
        structure[inner_span.j] = b')';
        traceback_v(ctx, inner_span, structure);
        return;
    }
    let matched = super::multi::traceback_v_multi_close(ctx, outer, target, structure);
    debug_assert!(
        matched,
        "V[{}, {}] traceback failed: no leaf or multi-loop case matched",
        span.i, span.j,
    );
}

/// Find the inner pair `(k, l)` whose two-pair leaf closes V[i, j] at
/// the recorded MFE. Iteration order mirrors `inner_pair_candidates`
/// (Vienna's `backtrack/bt_internal.c::bt_int_loop` order) — first
/// match wins.
fn find_v_two_pair_split(ctx: &FillCtx<'_>, outer: OuterClose, target: Energy) -> Option<Span> {
    inner_pair_candidates(outer.span)
        .find(|&inner_span| two_pair_total(ctx, outer, inner_span) == Some(target))
}
