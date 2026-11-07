//! Per-recurrence candidate generators and fillers.
//!
//! Every recurrence is two functions: a candidate generator (yields
//! [`Candidate`] records describing each transition the recurrence
//! considers) and a filler (runs `min(...)` over the generator and stores
//! the result in the corresponding sub-table). Reduction is the only
//! non-generator code per recurrence; everything else is in the generator,
//! so a richer reduction (a front rather than a scalar) can be swapped in
//! over the same candidate streams without rewriting the recurrences.
//! Future v2 / beam variants swap the reduction without re-shaping the
//! generators.
//!
//! `e_table::fill_e` is the per-cell reduction `E[idx] = min(E_h, E_s, E_b,
//! E_i, E_m)[idx]`. The single-loop fill driver leaves `E_m`, `M`, `N`,
//! `O` at [`crate::plan::INF`]; the reduction's `min` over `E_m` is a
//! no-op until the multi-loop recurrence populates it.

use rna_mfe::loops::{pair_from_nucleotides, Nucleotide};
use rna_mfe::params::BasePair;

use crate::plan::{nuc_from_index, CellIdx, EMotifKind, EStorageMode, Plan, INF};

pub(crate) mod bulge;
pub(crate) mod exterior;
pub(crate) mod hairpin;
pub(crate) mod internal;
pub(crate) mod multi;
pub(crate) mod stack;
pub(crate) mod traceback;
pub(crate) mod two_pair;

/// One transition considered by a recurrence — the unit element of the
/// candidate-generator + reduction split.
///
/// `value` is the BRD objective in dekacal `i32`. `trace` is a
/// recurrence-specific record the traceback walker reads to recover
/// which case produced the cell value. Future v2 / beam variants swap the
/// reduction over this same candidate stream without changing the
/// generators.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub value: f64,
    pub trace: Trace,
}

/// Trace records carried by [`Candidate`]. One variant per recurrence;
/// each records what its recurrence chose, so the traceback walker can dispatch case-by-case from the stored record.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Trace {
    Hairpin(HairpinTrace),
    Stacking(StackingTrace),
    Bulge(BulgeTrace),
    Internal(InternalLoopTrace),
    MultiLoop(MultiLoopTrace),
    M(MTrace),
    N(NTrace),
    O(OTrace),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HairpinTrace {
    pub h: usize,
    pub r: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StackingTrace {
    pub h: usize,
    pub r: usize,
    pub inner_idx: CellIdx,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BulgeTrace {
    pub h: usize,
    pub r: usize,
    pub inner_idx: CellIdx,
}

/// Trace record for the internal-loop recurrence at the nt-axis
/// candidate shape: the generator enumerates boundary nucleotides rather
/// than flank codons, folding the flank CAI through a per-nucleotide
/// best-synonym table, since a flank codon affects the energy only through
/// its boundary nucleotide.
///
/// `xi_` / `hi_` / `kj_` / `yj_` are the four loop boundary nucleotides
/// at positions `(nt_i + 1, nt_p − 1, nt_q + 1, nt_j − 1)`. `h` / `r` are the enclosed-pair
/// codon indices (same shape as the other two-pair traces). The
/// traceback walker recovers the optimal flank codon at each AA by
/// looking up [`crate::plan::Plan::best_synonym_idx`]; the joint
/// flank-AA collision case (`a1 == c1` or `b1 == d1`, both free) is
/// resolved by re-searching synonyms for the codon whose nucleotides
/// match both boundary nts.

#[derive(Debug, Clone, Copy)]
pub(crate) struct InternalLoopTrace {
    pub xi_: u8,
    pub hi_: u8,
    pub kj_: u8,
    pub yj_: u8,
    pub h: usize,
    pub r: usize,
    pub inner_idx: CellIdx,
}

/// Trace record for the multi-loop closure recurrence `E_m`.
///
/// `case` is 1–4 keyed on whether `σ(a, s) + 1` and / or `σ(b, t) − 1`
/// cross codon boundaries. `x_prime` / `y_prime` carry the boundary-
/// cross codon assignments at `a + 1` / `b − 1` (zero for cases that
/// don't cross the corresponding side). `inner_idx` points at the
/// `M[...]` cell read by this candidate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MultiLoopTrace {
    pub case: u8,
    pub x_prime: usize,
    pub y_prime: usize,
    pub inner_idx: CellIdx,
}

/// Trace record for the `M` recurrence's single bifurcation into two `N`
/// halves at AAs `c ≤ d` with codon offsets `t'`, `s'` on either side of
/// the cut. `y_prime` / `x_prime` are the codon assignments at `c` / `d`
/// (forced equal under the consistency filter when `c == d`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct MTrace {
    pub c: usize,
    pub d: usize,
    pub y_prime: usize,
    pub x_prime: usize,
    pub left_idx: CellIdx,
    pub right_idx: CellIdx,
}

/// Trace record for the `N` recurrence. `case` is 1–7:
///
/// * 1 — closes a branch via `E[idx]` (no inner index needed).
/// * 2 / 3 — within-cell unpaired bump at adjusted `(s, t)`.
/// * 4 / 5 — boundary-crossing unpaired bump; `aux_codon` carries the
///   new codon assignment at `a + 1` or `b − 1`.
/// * 6 / 7 — bifurcation into two `N` halves; `inner_idx_b` is `Some`,
///   `aux_codon` is the shared-AA codon at `c`, and case 7 also carries
///   the codon at `c + 1` in `aux_codon_2`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NTrace {
    pub case: u8,
    pub inner_idx_a: CellIdx,
    pub inner_idx_b: Option<CellIdx>,
    pub aux_codon: usize,
    pub aux_codon_2: usize,
}

/// Trace record for the `O` (exterior loop) recurrence. `case` is 3–9:
///
/// * 1 / 2 — base cases at `a == b` (handled directly by `fill_o`; no
///   trace record is emitted).
/// * 3 / 4 — within-cell unpaired shift at adjusted `(s, t)`.
/// * 5 / 6 — codon-boundary cross; `aux_codon` carries the new codon
///   assignment at `a + 1` or `b − 1`.
/// * 7 — within-codon bifurcation into an `O` left half and an `E + ext_loop`
///   right half. `aux_codon` is the shared-AA codon at `c`.
/// * 8 — between-codon bifurcation. `aux_codon` is the codon at `c`;
///   `aux_codon_2` is the codon at `c + 1`.
/// * 9 — closure via `E[idx] + ext_loop`. `inner_idx_a == idx`.
///
/// `inner_idx_b` is `Some` only for the bifurcation cases (7 / 8); it
/// points at the `E` cell read by the right half.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OTrace {
    pub case: u8,
    pub inner_idx_a: CellIdx,
    pub inner_idx_b: Option<CellIdx>,
    pub aux_codon: usize,
    pub aux_codon_2: usize,
}

/// `(s, t)` pairs sorted by ascending nucleotide-span offset `t - s`.
///
/// Stacking within a fixed `(a, b)` AA-span refers to inner cell
/// `(s+1, t-1)` whose nt-span offset is `t - s - 2` — two less than the
/// outer cell. Filling cells in ascending `t - s` order ensures every
/// stacking inner cell is populated before any outer cell that references
/// it.
pub(crate) const ST_BY_NT_SPAN: [(usize, usize); 9] = [
    // t - s = -2
    (2, 0),
    // t - s = -1
    (1, 0),
    (2, 1),
    // t - s = 0
    (0, 0),
    (1, 1),
    (2, 2),
    // t - s = 1
    (0, 1),
    (1, 2),
    // t - s = 2
    (0, 2),
];

/// Within-cell `(s, t)` order for the `N` and `O` recurrences.
///
/// Both `N` and `O` have self-referencing within-cell cases that read
/// `(s + 1, t)` or `(s, t − 1)` at the same `(a, b)` AA pair:
///
/// * N cases 2 (`s + 1`) and 3 (`t − 1`).
/// * O cases 3 (`s + 1`) and 4 (`t − 1`).
///
/// Iterating `s` descending (`2 → 1 → 0`) and `t` ascending
/// (`0 → 1 → 2`) ensures the inner cell each case reads is already
/// filled. The product order is
/// `(s, t) ∈ {(2,0), (2,1), (2,2), (1,0), (1,1), (1,2), (0,0), (0,1), (0,2)}`.
pub(crate) const ST_BY_NT_SPAN_FOR_N: [(usize, usize); 9] = [
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (0, 0),
    (0, 1),
    (0, 2),
];

/// Reduce per-motif tables to `E[idx]`.
pub(crate) fn fill_e(plan: &mut Plan<'_>, idx: CellIdx) {
    debug_assert_eq!(plan.tables.e_storage_mode(), EStorageMode::FullMotifTables);
    let i = plan.tables.flat_index(idx);
    let v = plan.tables.e_h[i]
        .min(plan.tables.e_s[i])
        .min(plan.tables.e_b[i])
        .min(plan.tables.e_i[i])
        .min(plan.tables.e_m[i]);
    plan.tables.e[i] = v;
}

fn keep_best(best: &mut f64, cand: Candidate) {
    if cand.value < *best {
        *best = cand.value;
    }
}

fn compact_hairpin_value(plan: &Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) -> f64 {
    let mut best = INF;
    hairpin::hairpin_candidates(plan, idx, scratch, |cand| keep_best(&mut best, cand));
    best
}

fn compact_stack_value(plan: &Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) -> f64 {
    let mut best = INF;
    stack::stacking_candidates(plan, idx, scratch, |cand| keep_best(&mut best, cand));
    best
}

fn compact_bulge_value(plan: &Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) -> f64 {
    let mut best = INF;
    bulge::bulge_candidates(plan, idx, scratch, |cand| keep_best(&mut best, cand));
    best
}

fn compact_internal_value(plan: &Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) -> f64 {
    let mut best = INF;
    internal::internal_candidates(plan, idx, scratch, |cand| keep_best(&mut best, cand));
    best
}

fn compact_multi_loop_value(plan: &Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) -> f64 {
    let mut best = INF;
    multi::e_m_candidates(plan, idx, scratch, |cand| keep_best(&mut best, cand));
    best
}

struct MotifValues {
    h: f64,
    s: f64,
    b: f64,
    i: f64,
    m: f64,
}

fn select_e_motif(values: MotifValues) -> (EMotifKind, f64) {
    let mut winner = EMotifKind::Hairpin;
    let mut best = values.h;
    for (kind, value) in [
        (EMotifKind::Stack, values.s),
        (EMotifKind::Bulge, values.b),
        (EMotifKind::Internal, values.i),
        (EMotifKind::MultiLoop, values.m),
    ] {
        if value < best {
            winner = kind;
            best = value;
        }
    }
    (winner, best)
}

/// Returns `true` iff the outer closing pair at `(a, s, x)` / `(b, t, y)` is
/// canonical. Non-canonical cells produce `INF` from every motif generator;
/// the gate short-circuits before running any of them.
#[inline]
fn outer_pair_is_canonical(plan: &Plan<'_>, idx: CellIdx) -> bool {
    let (a, b, s, t, x, y) = idx;
    let outer = pair_from_nucleotides(
        nuc_from_index(plan.codon_lists[a][x].nuc[s]),
        nuc_from_index(plan.codon_lists[b][y].nuc[t]),
    );
    !matches!(outer, BasePair::NN)
}

fn fill_compact_e_cell(
    plan: &mut Plan<'_>,
    idx: CellIdx,
    scratch: &mut Vec<Nucleotide>,
    include_multi_loop: bool,
) {
    if !outer_pair_is_canonical(plan, idx) {
        let flat = plan.tables.flat_index(idx);
        plan.tables.e[flat] = INF;
        plan.tables.e_winner[flat] = EMotifKind::Hairpin;
        return;
    }
    let (winner, value) = {
        let p: &Plan<'_> = plan;
        select_e_motif(MotifValues {
            h: compact_hairpin_value(p, idx, scratch),
            s: compact_stack_value(p, idx, scratch),
            b: compact_bulge_value(p, idx, scratch),
            i: compact_internal_value(p, idx, scratch),
            m: if include_multi_loop {
                compact_multi_loop_value(p, idx, scratch)
            } else {
                INF
            },
        })
    };
    let flat = plan.tables.flat_index(idx);
    plan.tables.e[flat] = value;
    plan.tables.e_winner[flat] = winner;
}

fn fill_full_e_cell(
    plan: &mut Plan<'_>,
    idx: CellIdx,
    scratch: &mut Vec<Nucleotide>,
    include_multi_loop: bool,
) {
    if !outer_pair_is_canonical(plan, idx) {
        return;
    }
    hairpin::fill_e_h(plan, idx, scratch);
    stack::fill_e_s(plan, idx, scratch);
    bulge::fill_e_b(plan, idx, scratch);
    internal::fill_e_i(plan, idx, scratch);
    if include_multi_loop {
        multi::fill_e_m(plan, idx, scratch);
    }
    fill_e(plan, idx);
}

/// Run every single-loop sub-recurrence at one cell, then reduce into `E`.
fn fill_one_cell_m2(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    match plan.tables.e_storage_mode() {
        EStorageMode::FullMotifTables => fill_full_e_cell(plan, idx, scratch, false),
        EStorageMode::CompactMotifWinners => fill_compact_e_cell(plan, idx, scratch, false),
    }
}

fn fill_codon_corners(
    plan: &mut Plan<'_>,
    abst: (usize, usize, usize, usize),
    scratch: &mut Vec<Nucleotide>,
) {
    let (a, b, s, t) = abst;
    let nx = plan.codon_lists[a].len();
    let ny = plan.codon_lists[b].len();
    for x in 0..nx {
        for y in 0..ny {
            fill_one_cell_m2(plan, (a, b, s, t, x, y), scratch);
        }
    }
}

fn fill_ab_cell(plan: &mut Plan<'_>, ab: (usize, usize), scratch: &mut Vec<Nucleotide>) {
    for &(s, t) in &ST_BY_NT_SPAN {
        fill_codon_corners(plan, (ab.0, ab.1, s, t), scratch);
    }
}

/// Fill every single-loop sub-table at every cell, in dependency order.
///
/// Span 0 (`a == b`) is included so future O / N base cases find their
/// cells initialised; the single-loop generators all constraint `a >= b`, so
/// span-0 cells stay at [`crate::plan::INF`] for these tables.
///
/// Multi-loop (`E_m`, `M`, `N`) and exterior (`O`) tables are not touched
/// here — they're owned by the multi-loop and exterior recurrences. The
/// fill driver therefore stops at the `E` reduction.
pub(crate) fn fill_single_loop_tables(plan: &mut Plan<'_>) {
    let n_aa = plan.n_aa();
    let mut scratch: Vec<Nucleotide> = Vec::with_capacity(plan.default_seq.len());
    for span in 0..n_aa {
        for a in 0..(n_aa - span) {
            fill_ab_cell(plan, (a, a + span), &mut scratch);
        }
    }
}

/// Fill every single-loop sub-table plus the multi-loop tables (`M`,
/// `E_m`, `N`) at every cell, in dependency order.
///
/// Three intra-cell steps per `(a, b)` AA pair (the full fill's four
/// minus the `fill_o` pass):
///
/// 1. `fill_m` for every `(s, t, x, y)`. `M` reads `N` at strictly
///    smaller AA spans (already populated by the outer span loop) and is
///    `(s, t)`-order-insensitive, so any order works here. `E_m` (step 2)
///    reads `M[idx]`, so step 1 must precede step 2.
/// 2. The five motif fillers (`E_h`, `E_s`, `E_b`, `E_i`, `E_m`) and the
///    `E` reduction. `E_s` requires the `ST_BY_NT_SPAN` ordering so its
///    stacking inner cell `(s + 1, t − 1)` is already filled when read.
/// 3. `fill_n` for every `(s, t, x, y)` in `ST_BY_NT_SPAN_FOR_N` order.
///    Cases N2 / N3 self-reference within the same `(a, b)` cell at
///    adjusted `(s, t)`; the descending-`s` / ascending-`t` traversal
///    ensures those reads land on already-filled cells. Case N1 reads
///    `E[idx]` (reduced in step 2). Cases N4 / N5 / N6 / N7 read
///    `N` at strictly smaller AA spans.
///
/// The exterior table `O` is untouched here.
pub(crate) fn fill_through_multi_loop_tables(plan: &mut Plan<'_>) {
    let n_aa = plan.n_aa();
    let mut scratch: Vec<Nucleotide> = Vec::with_capacity(plan.default_seq.len());
    for span in 0..n_aa {
        for a in 0..(n_aa - span) {
            fill_ab_cell_through_multi(plan, (a, a + span), &mut scratch);
        }
    }
}

/// Sweep `(s, t, x, y)` in the supplied `(s, t)` order at the fixed
/// `(a, b)` AA pair, invoking `body` on each cell. Used to lift the
/// three intra-cell fill steps out of [`fill_ab_cell_through_multi`] so
/// each step is a one-line composition.
fn sweep_cells_in_order<B>(
    plan: &mut Plan<'_>,
    ab: (usize, usize),
    st_order: &[(usize, usize)],
    mut body: B,
) where
    B: FnMut(&mut Plan<'_>, CellIdx),
{
    let (a, b) = ab;
    let nx = plan.codon_lists[a].len();
    let ny = plan.codon_lists[b].len();
    for &(s, t) in st_order {
        for x in 0..nx {
            for y in 0..ny {
                body(plan, (a, b, s, t, x, y));
            }
        }
    }
}

fn fill_ab_cell_through_multi(
    plan: &mut Plan<'_>,
    ab: (usize, usize),
    scratch: &mut Vec<Nucleotide>,
) {
    // Step 1: M (any `(s, t)` order works — reads `N` at smaller AA spans).
    sweep_cells_in_order(plan, ab, &ST_BY_NT_SPAN, multi::fill_m);
    // Step 2: single-loop motifs + E_m + E reduction (stacking order).
    sweep_cells_in_order(plan, ab, &ST_BY_NT_SPAN, |p, idx| {
        fill_one_cell_through_multi(p, idx, scratch);
    });
    // Step 3: N (within-cell descending-`s` / ascending-`t` order).
    sweep_cells_in_order(plan, ab, &ST_BY_NT_SPAN_FOR_N, |p, idx| {
        multi::fill_n(p, idx, scratch);
    });
}

fn fill_one_cell_through_multi(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    match plan.tables.e_storage_mode() {
        EStorageMode::FullMotifTables => fill_full_e_cell(plan, idx, scratch, true),
        EStorageMode::CompactMotifWinners => fill_compact_e_cell(plan, idx, scratch, true),
    }
}

/// Fill every DP table at every cell, in dependency order.
///
/// Four intra-cell steps per `(a, b)` AA pair:
///
/// 1. `fill_m` for every `(s, t, x, y)`. `M` reads `N` at strictly
///    smaller AA spans (already populated by the outer span loop) and is
///    `(s, t)`-order-insensitive, so any order works here. `E_m` (step 2)
///    reads `M[idx]`, so step 1 must precede step 2.
/// 2. The five motif fillers (`E_h`, `E_s`, `E_b`, `E_i`, `E_m`) and the
///    `E` reduction. `E_s` requires the `ST_BY_NT_SPAN` ordering so its
///    stacking inner cell `(s + 1, t − 1)` is already filled when read.
/// 3. `fill_n` for every `(s, t, x, y)` in `ST_BY_NT_SPAN_FOR_N` order.
///    Cases N2 / N3 self-reference within the same `(a, b)` cell at
///    adjusted `(s, t)`; the descending-`s` / ascending-`t` traversal
///    ensures those reads land on already-filled cells. Case N1 reads
///    `E[idx]` (reduced in step 2). Cases N4 / N5 / N6 / N7 read
///    `N` at strictly smaller AA spans.
/// 4. `fill_o` for prefix cells only: `(a=0, s=0, t, x, y)`. Cases O9
///    (closure) and O7-with-`c==a` read `E[idx]` (filled in step 2). Cases
///    O4 / O6 / O7 / O8 read `O` at the same prefix with smaller `t` or
///    strictly smaller right endpoints. The production solve and traceback
///    start from `O[0, n_aa - 1, 0, 2, x, y]`, so non-prefix exterior cells
///    are not filled on the hot path.
pub(crate) fn fill_dp_tables(plan: &mut Plan<'_>) {
    let n_aa = plan.n_aa();
    let mut scratch: Vec<Nucleotide> = Vec::with_capacity(plan.default_seq.len());
    for span in 0..n_aa {
        for a in 0..(n_aa - span) {
            fill_ab_cell_full(plan, (a, a + span), &mut scratch);
        }
    }
}

fn fill_ab_cell_full(plan: &mut Plan<'_>, ab: (usize, usize), scratch: &mut Vec<Nucleotide>) {
    sweep_cells_in_order(plan, ab, &ST_BY_NT_SPAN, multi::fill_m);
    sweep_cells_in_order(plan, ab, &ST_BY_NT_SPAN, |p, idx| {
        fill_one_cell_through_multi(p, idx, scratch);
    });
    sweep_cells_in_order(plan, ab, &ST_BY_NT_SPAN_FOR_N, |p, idx| {
        multi::fill_n(p, idx, scratch);
    });
    fill_o_prefix_cells(plan, ab, scratch);
}

fn fill_o_prefix_cells(plan: &mut Plan<'_>, ab: (usize, usize), scratch: &mut Vec<Nucleotide>) {
    let (a, b) = ab;
    if a != 0 {
        return;
    }
    let nx = plan.codon_lists[a].len();
    let ny = plan.codon_lists[b].len();
    for t in 0..3 {
        for x in 0..nx {
            for y in 0..ny {
                exterior::fill_o(plan, (a, b, 0, t, x, y), scratch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_e_winner_preserves_motif_tie_order() {
        let values = |h, s, b, i, m| MotifValues { h, s, b, i, m };

        let (kind, value) = select_e_motif(values(7.0, 7.0, 7.0, 7.0, 7.0));
        assert_eq!(kind, EMotifKind::Hairpin);
        assert_eq!(value, 7.0);

        let (kind, value) = select_e_motif(values(9.0, 3.0, 3.0, 3.0, 3.0));
        assert_eq!(kind, EMotifKind::Stack);
        assert_eq!(value, 3.0);

        let (kind, value) = select_e_motif(values(9.0, 8.0, 5.0, 5.0, 5.0));
        assert_eq!(kind, EMotifKind::Bulge);
        assert_eq!(value, 5.0);

        let (kind, value) = select_e_motif(values(9.0, 8.0, 7.0, 4.0, 4.0));
        assert_eq!(kind, EMotifKind::Internal);
        assert_eq!(value, 4.0);

        let (kind, value) = select_e_motif(values(9.0, 8.0, 7.0, 6.0, 2.0));
        assert_eq!(kind, EMotifKind::MultiLoop);
        assert_eq!(value, 2.0);
    }
}
