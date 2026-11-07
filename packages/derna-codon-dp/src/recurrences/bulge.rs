//! Bulge-loop recurrence (E_b).
//!
//! Follows supplement.tex lines 209–226. Enumerates every enclosed-pair position `(nt_p, nt_q)`
//! where exactly one strand has zero unpaired nucleotides, and yields one
//! candidate per consistent `(h, r)` codon pair at the enclosed AAs.

use rna_mfe::loops::Nucleotide;
use rna_mfe::{bulge_energy, BulgeLoop};

use crate::brd_value::brd_value;
use crate::plan::{nt_position, CellIdx, Plan, INF, MAXLOOP, TURN};

use super::two_pair::{
    closing_pair_cai, iter_two_pair_inner, make_site, pair_at, InnerRef, MotifSite,
};
use super::{BulgeTrace, Candidate, Trace};

/// Read-only context for one bulge-recurrence enclosed-pair position:
/// the motif geometry and the closing-pair CAI charge every candidate
/// at this site shares.
#[derive(Debug, Clone, Copy)]
struct BulgeCell {
    site: MotifSite,
    cai_closing: f64,
}

fn bulge_one_candidate(plan: &Plan<'_>, cell: BulgeCell, inner: InnerRef) -> Option<Candidate> {
    let (c, sp, d, tp) = cell.site.enclosed;
    let (outer, inner_pair) = pair_at(plan, cell.site, inner.h, inner.r)?;
    let (nt_i, nt_j, nt_p, nt_q) = cell.site.nt_pairs;
    let size = (nt_p - nt_i - 1).max(nt_j - nt_q - 1);
    let energy = bulge_energy(
        plan.params,
        &BulgeLoop {
            outer,
            inner: inner_pair,
            size,
        },
    );
    let value = inner.inner_val + brd_value(plan.lambda_, energy.dekacal(), cell.cai_closing);
    let inner_idx = (c, d, sp, tp, inner.h, inner.r);
    Some(Candidate {
        value,
        trace: Trace::Bulge(BulgeTrace {
            h: inner.h,
            r: inner.r,
            inner_idx,
        }),
    })
}

fn emit_for_site<F: FnMut(Candidate)>(plan: &Plan<'_>, cell: BulgeCell, emit: &mut F) {
    iter_two_pair_inner(plan, cell.site, |inner_ref| {
        if let Some(cand) = bulge_one_candidate(plan, cell, inner_ref) {
            emit(cand);
        }
    });
}

/// Bulge candidates: enumerate enclosed-pair positions where exactly one
/// strand has zero unpaired nucleotides.
///
/// The `scratch` parameter is unused — closing and inner nucleotides are
/// read directly from `Plan::codon_lists` via `two_pair::pair_at` — but
/// the signature is shared with the other E-motif generators so they all
/// dispatch through a single match in `recurrences::extract_motif_trace`.
pub(crate) fn bulge_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    _scratch: &mut Vec<Nucleotide>,
    mut emit: F,
) {
    let (a, b, s, t, _x, _y) = idx;
    let nt_i = nt_position(a, s);
    let nt_j = nt_position(b, t);
    for_each_enclosed_pair_position(nt_i, nt_j, MotifKind::Bulge, |nt_p, nt_q| {
        if let Some(site) = make_site(idx, (nt_i, nt_j, nt_p, nt_q)) {
            let cell = BulgeCell {
                site,
                cai_closing: closing_pair_cai(plan, site),
            };
            emit_for_site(plan, cell, &mut emit);
        }
    });
}

/// Fill `E_b[idx]` via `min` over bulge candidates.
pub(crate) fn fill_e_b(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let best = {
        let p: &Plan<'_> = plan;
        let mut best = INF;
        bulge_candidates(p, idx, scratch, |cand| {
            if cand.value < best {
                best = cand.value;
            }
        });
        best
    };
    let i = plan.tables.flat_index(idx);
    plan.tables.e_b[i] = best;
}

/// Motif filter for enclosed-pair enumeration: bulges keep `(n1, n2)`
/// pairs where exactly one strand is empty; internal loops keep pairs
/// where both are non-empty. Stacking is handled separately by the
/// stack recurrence (which sets `(nt_p, nt_q) = (nt_i + 1, nt_j - 1)`
/// directly).
#[derive(Debug, Clone, Copy)]
pub(crate) enum MotifKind {
    Bulge,
    Internal,
}

impl MotifKind {
    fn skip(self, n1: usize, n2: usize) -> bool {
        match self {
            // Keep iff exactly one strand is empty: skip when both
            // predicates match (both 0 or both non-0).
            MotifKind::Bulge => (n1 == 0) == (n2 == 0),
            // Internal loops require both strands non-empty.
            MotifKind::Internal => n1 == 0 || n2 == 0,
        }
    }
}

/// Emit every `(nt_p, nt_q)` enclosed-pair position bounded by
/// `MAXLOOP = 30` unpaired bases and the per-strand TURN constraint,
/// filtered by motif kind.
pub(crate) fn for_each_enclosed_pair_position<F: FnMut(usize, usize)>(
    nt_i: usize,
    nt_j: usize,
    motif: MotifKind,
    mut emit: F,
) {
    if nt_j == 0 {
        return;
    }
    let p_max = (nt_i + MAXLOOP + 1).min(nt_j.saturating_sub(1));
    for nt_p in (nt_i + 1)..=p_max {
        let n1 = nt_p - nt_i - 1;
        let q_min = (nt_p + TURN + 1).max((nt_j.saturating_sub(1)).saturating_sub(MAXLOOP - n1));
        for nt_q in q_min..nt_j {
            let n2 = nt_j - nt_q - 1;
            if !motif.skip(n1, n2) {
                emit(nt_p, nt_q);
            }
        }
    }
}
