//! Hairpin recurrence (E_h).
//!
//! Follows supplement.tex "Hairpin Loop". For every `(h, r)` codon pair at the flanking AAs `c`
//! and `d`, build the candidate RNA, skip non-canonical closing pairs,
//! score the per-loop hairpin energy via [`rna_mfe::hairpin_energy`], and
//! yield a [`Candidate`] carrying the BRD value.

use rna_mfe::loops::{pair_from_nucleotides, Nucleotide};
use rna_mfe::params::BasePair;
use rna_mfe::{hairpin_energy, HairpinLoop};

use crate::brd_value::brd_value;
use crate::plan::{
    aa_of_nt, assemble_rna_into, consistent_choices, nt_position, CellIdx, Plan, INF, TURN,
};

use super::{Candidate, HairpinTrace, Trace};

/// Hairpin geometry: closing-pair nucleotide positions plus the flanking
/// AA indices. `idx` is carried along so helpers downstream do not have
/// to re-thread it.
#[derive(Debug, Clone, Copy)]
struct HairpinSite {
    idx: CellIdx,
    c: usize,
    d: usize,
    nt_i: usize,
    nt_j: usize,
}

/// `(h, r)` codon assignments at the flanking AAs `(c, d)`.
type HrPair = (usize, usize);

fn hairpin_loop_too_short(idx: CellIdx) -> bool {
    let (a, b, s, t, _x, _y) = idx;
    let nt_i = nt_position(a, s);
    let nt_j = nt_position(b, t);
    nt_j.checked_sub(nt_i + 1).is_none_or(|gap| gap < TURN)
}

fn hairpin_site(idx: CellIdx) -> HairpinSite {
    let (a, b, s, t, _x, _y) = idx;
    let nt_i = nt_position(a, s);
    let nt_j = nt_position(b, t);
    HairpinSite {
        idx,
        c: aa_of_nt(nt_i + 1),
        d: aa_of_nt(nt_j - 1),
        nt_i,
        nt_j,
    }
}

fn flanking_cai_term(log_cai: &[Vec<f64>], p: usize, ci: usize, excludes: &[usize]) -> f64 {
    if excludes.contains(&p) {
        0.0
    } else {
        log_cai[p][ci]
    }
}

fn hairpin_cai_dekacal(plan: &Plan<'_>, site: HairpinSite, hr: HrPair) -> f64 {
    let (a, b, _s, _t, x, y) = site.idx;
    let (h, r) = hr;
    let log_cai = &plan.log_cai_dekacal_per_pos;
    let mut contrib = log_cai[a][x] + log_cai[b][y];
    contrib += flanking_cai_term(log_cai, site.c, h, &[a, b]);
    contrib += flanking_cai_term(log_cai, site.d, r, &[a, b, site.c]);
    contrib
}

fn pinned_for(site: HairpinSite, hr: HrPair) -> [(usize, usize); 4] {
    let (a, b, _s, _t, x, y) = site.idx;
    let (h, r) = hr;
    [(a, x), (b, y), (site.c, h), (site.d, r)]
}

fn closing_pair_of(scratch: &[Nucleotide], site: HairpinSite) -> Option<BasePair> {
    let pair = pair_from_nucleotides(scratch[site.nt_i], scratch[site.nt_j]);
    if matches!(pair, BasePair::NN) {
        None
    } else {
        Some(pair)
    }
}

fn hairpin_one_candidate(
    plan: &Plan<'_>,
    scratch: &mut Vec<Nucleotide>,
    site: HairpinSite,
    hr: HrPair,
) -> Option<Candidate> {
    let pinned = pinned_for(site, hr);
    assemble_rna_into(plan, &pinned, scratch);
    let closing = closing_pair_of(scratch, site)?;
    let energy = hairpin_energy(
        plan.params,
        &HairpinLoop {
            closing,
            loop_seq: &scratch[site.nt_i + 1..site.nt_j],
        },
    );
    let cai_dekacal = hairpin_cai_dekacal(plan, site, hr);
    let (h, r) = hr;
    Some(Candidate {
        value: brd_value(plan.lambda_, energy.dekacal(), cai_dekacal),
        trace: Trace::Hairpin(HairpinTrace { h, r }),
    })
}

/// Yield every hairpin candidate for cell `idx`.
///
/// The closing pair sits at `(nt_i, nt_j)` where `nt_i = 3·a + s`,
/// `nt_j = 3·b + t`. Flanking AAs `c = aa_of_nt(nt_i+1)` and
/// `d = aa_of_nt(nt_j-1)` may collide with `a` or `b`, in which case
/// the consistency filter forces the corresponding codon choice.
pub(crate) fn hairpin_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    scratch: &mut Vec<Nucleotide>,
    mut emit: F,
) {
    if hairpin_loop_too_short(idx) {
        return;
    }
    let site = hairpin_site(idx);
    let (a, b, _s, _t, x, y) = idx;
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for h in consistent_choices(plan, site.c, &base) {
        let extended: [(usize, usize); 3] = [(a, x), (b, y), (site.c, h)];
        for r in consistent_choices(plan, site.d, &extended) {
            if let Some(cand) = hairpin_one_candidate(plan, scratch, site, (h, r)) {
                emit(cand);
            }
        }
    }
}

/// Fill `E_h[idx]` via `min` over hairpin candidates.
pub(crate) fn fill_e_h(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let best = {
        let p: &Plan<'_> = plan;
        let mut best = INF;
        hairpin_candidates(p, idx, scratch, |cand| {
            if cand.value < best {
                best = cand.value;
            }
        });
        best
    };
    let i = plan.tables.flat_index(idx);
    plan.tables.e_h[i] = best;
}
