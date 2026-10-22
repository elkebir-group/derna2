//! Stacking recurrence (E_s).
//!
//! Mirrors v1's `derna.codon_dp._recurrences.fill_e_s` — supplement.tex
//! lines 170–188. The enclosed pair `(c, s', d, t')` is fully determined
//! by the closing pair: `nt_p = nt_i + 1`, `nt_q = nt_j - 1`, so there is
//! no enumeration over enclosed-pair positions. The recurrence reads the
//! inner cell `E[c, d, s', t', h, r]` and adds the BRD value of the
//! single-stack motif at the closing pair.

use rna_mfe::loops::Nucleotide;
use rna_mfe::{stack_energy, StackPair};

use crate::brd_value::brd_value;
use crate::plan::{nt_position, CellIdx, Plan, INF};

use super::two_pair::{
    closing_pair_cai, iter_two_pair_inner, make_site, pair_at, InnerRef, MotifSite,
};
use super::{Candidate, StackingTrace, Trace};

/// Read-only context for one stacking-recurrence cell: the motif geometry
/// and the closing-pair CAI charge that every candidate at this cell
/// shares. Bundles four naturally co-located quantities so the per-
/// candidate evaluator stays at one ergonomic argument.
#[derive(Debug, Clone, Copy)]
struct StackingCell {
    site: MotifSite,
    cai_closing: f64,
}

fn stacking_one_candidate(
    plan: &Plan<'_>,
    cell: StackingCell,
    inner: InnerRef,
) -> Option<Candidate> {
    let (c, sp, d, tp) = cell.site.enclosed;
    let (outer, inner_pair) = pair_at(plan, cell.site, inner.h, inner.r)?;
    let energy = stack_energy(
        plan.params,
        &StackPair {
            outer,
            inner: inner_pair,
        },
    );
    let value = inner.inner_val + brd_value(plan.lambda_, energy.dekacal(), cell.cai_closing);
    let inner_idx = (c, d, sp, tp, inner.h, inner.r);
    Some(Candidate {
        value,
        trace: Trace::Stacking(StackingTrace {
            h: inner.h,
            r: inner.r,
            inner_idx,
        }),
    })
}

fn stacking_site(idx: CellIdx) -> Option<MotifSite> {
    let (a, b, s, t, _x, _y) = idx;
    let nt_i = nt_position(a, s);
    let nt_j = nt_position(b, t);
    if nt_j == 0 || nt_j - 1 <= nt_i {
        return None;
    }
    let nt_p = nt_i + 1;
    let nt_q = nt_j - 1;
    if nt_p >= nt_q {
        return None;
    }
    make_site(idx, (nt_i, nt_j, nt_p, nt_q))
}

/// Stacking candidates: one per `(h, r)` codon pair at the enclosed-pair
/// AAs. The enclosed pair sits at `(nt_i + 1, nt_j - 1)` and its AAs
/// `(c, d)` may collide with the closing-pair AAs.
///
/// The `scratch` parameter is unused — closing and inner nucleotides are
/// read directly from `Plan::codon_lists` via `two_pair::pair_at` — but
/// the signature is shared with the other E-motif generators so they all
/// dispatch through a single match in `recurrences::extract_motif_trace`.
pub(crate) fn stacking_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    _scratch: &mut Vec<Nucleotide>,
    mut emit: F,
) {
    let Some(site) = stacking_site(idx) else {
        return;
    };
    let cell = StackingCell {
        site,
        cai_closing: closing_pair_cai(plan, site),
    };
    iter_two_pair_inner(plan, site, |inner_ref| {
        if let Some(cand) = stacking_one_candidate(plan, cell, inner_ref) {
            emit(cand);
        }
    });
}

/// Fill `E_s[idx]` via `min` over stacking candidates.
pub(crate) fn fill_e_s(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let best = {
        let p: &Plan<'_> = plan;
        let mut best = INF;
        stacking_candidates(p, idx, scratch, |cand| {
            if cand.value < best {
                best = cand.value;
            }
        });
        best
    };
    let i = plan.tables.flat_index(idx);
    plan.tables.e_s[i] = best;
}
