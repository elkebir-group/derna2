//! Traceback walkers for every recurrence in the codon DP.
//!
//! Mirrors v1's `traceback_e` / `traceback_e_h` / `traceback_e_s` /
//! `traceback_e_b` / `traceback_e_i` / `traceback_e_m` /
//! `traceback_m` / `traceback_n` / `traceback_o`. Each walker reads the
//! winning cell value at `(a, b, s, t, x, y)`, re-runs the matching
//! candidate generator, finds the candidate whose value equals the cell,
//! reads its trace record, pins the implied codon assignments on
//! [`Plan::trace`], appends any closing-pair record, then recurses into
//! the inner cell(s) the trace points at.
//!
//! Termination: every recursion either descends into a strictly smaller
//! AA span (cases that read smaller `(a, c)` / `(c, b)` halves) or
//! shrinks a within-cell monovariant — `s` strictly increases via N2 /
//! O3, `t` strictly decreases via N3 / O4 — until the recurrence
//! transitions into a different table. Each transition step pays for
//! itself: the next table's recurrence either closes at a leaf (`E_h`
//! re-derives `(h, r)`) or recurses into a strictly smaller span.
//!
//! The non-canonical-outer-pair filter at `recurrences/two_pair.rs::pair_at`
//! keeps filtered cells at `INF`. The traceback walker never picks a
//! winning candidate at `INF`, so it should never descend into a filtered
//! cell. If a traceback dispatcher is reached at an apparently impossible
//! state, the walker panics rather than silently picking a fallback.

use rna_mfe::loops::Nucleotide;

use crate::plan::{aa_of_nt, nt_position, CellIdx, EMotifKind, EStorageMode, Plan, INF};

use super::{
    bulge::bulge_candidates,
    exterior::o_candidates,
    hairpin::hairpin_candidates,
    internal::internal_candidates,
    multi::{e_m_candidates, m_candidates, n_candidates},
    stack::stacking_candidates,
    Candidate, InternalLoopTrace, NTrace, OTrace, Trace,
};

// ---------------------------------------------------------------------------
// Trace-extraction helper
// ---------------------------------------------------------------------------

/// Find the first candidate whose `value` equals `target` and return its
/// raw [`Trace`]. Panics with `name`-tagged context when nothing matches.
fn find_winning_trace<G>(target: f64, name: &str, idx: CellIdx, gen: G) -> Trace
where
    G: FnOnce(&mut dyn FnMut(Candidate)),
{
    let mut found: Option<Trace> = None;
    gen(&mut |c: Candidate| {
        if found.is_none() && c.value == target {
            found = Some(c.trace);
        }
    });
    found.unwrap_or_else(|| panic!("{name} traceback failed at {idx:?}; target={target}"))
}

fn decode_enclosed_from_inner(inner: CellIdx) -> (usize, usize, usize, usize) {
    let (c, d, s_prime, t_prime, _h, _r) = inner;
    (c, s_prime, d, t_prime)
}

// ---------------------------------------------------------------------------
// E dispatcher + unified motif walker
// ---------------------------------------------------------------------------

/// Find which sub-table produced `E[idx]` and recurse into its walker.
pub(crate) fn traceback_e(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let i = plan.tables.flat_index(idx);
    let target = plan.tables.e[i];
    let kind = match plan.tables.e_storage_mode() {
        EStorageMode::CompactMotifWinners => plan.tables.e_winner[i],
        EStorageMode::FullMotifTables => {
            if plan.tables.e_h[i] == target {
                EMotifKind::Hairpin
            } else if plan.tables.e_s[i] == target {
                EMotifKind::Stack
            } else if plan.tables.e_b[i] == target {
                EMotifKind::Bulge
            } else if plan.tables.e_i[i] == target {
                EMotifKind::Internal
            } else if plan.tables.e_m[i] == target {
                EMotifKind::MultiLoop
            } else {
                panic!("E traceback failed at {idx:?}; target={target}");
            }
        }
    };
    let trace = extract_motif_trace(plan, idx, scratch, kind);
    apply_motif_trace(plan, idx, scratch, trace);
}

/// Read the per-motif table value at `idx`, run the matching candidate
/// generator, and return the winning [`Trace`]. The generator selection
/// is `match`-driven so adding a sub-table is local to this function.
fn extract_motif_trace(
    plan: &Plan<'_>,
    idx: CellIdx,
    scratch: &mut Vec<Nucleotide>,
    kind: EMotifKind,
) -> Trace {
    let target = plan.tables.e[plan.tables.flat_index(idx)];
    let name = kind.name();
    find_winning_trace(target, name, idx, |emit| match kind {
        EMotifKind::Hairpin => hairpin_candidates(plan, idx, scratch, emit),
        EMotifKind::Stack => stacking_candidates(plan, idx, scratch, emit),
        EMotifKind::Bulge => bulge_candidates(plan, idx, scratch, emit),
        EMotifKind::Internal => internal_candidates(plan, idx, scratch, emit),
        EMotifKind::MultiLoop => e_m_candidates(plan, idx, scratch, emit),
    })
}

/// Apply the per-motif effect: pin codon assignments, push the closing
/// pair, and (for stack / bulge / internal / multi-loop) recurse into
/// the inner cell.
fn apply_motif_trace(
    plan: &mut Plan<'_>,
    idx: CellIdx,
    scratch: &mut Vec<Nucleotide>,
    trace: Trace,
) {
    let (a, b, s, t, _x, _y) = idx;
    let nt_pair = (nt_position(a, s), nt_position(b, t));
    match trace {
        Trace::Hairpin(h) => apply_hairpin(plan, nt_pair, h.h, h.r),
        Trace::Stacking(st) => apply_two_pair(
            plan,
            scratch,
            nt_pair,
            TwoPairInfo {
                inner_idx: st.inner_idx,
                h: st.h,
                r: st.r,
            },
        ),
        Trace::Bulge(bt) => apply_two_pair(
            plan,
            scratch,
            nt_pair,
            TwoPairInfo {
                inner_idx: bt.inner_idx,
                h: bt.h,
                r: bt.r,
            },
        ),
        Trace::Internal(it) => apply_internal(plan, scratch, nt_pair, it),
        Trace::MultiLoop(mt) => apply_multi_loop_closure(plan, scratch, idx, mt),
        _ => panic!("apply_motif_trace: non-motif variant {trace:?}"),
    }
}

/// Bundled `(inner_idx, h, r)` for stack / bulge handlers — both
/// recurrences pin the same enclosed-pair codon assignments and recurse
/// into the same shape of inner `E` cell.
struct TwoPairInfo {
    inner_idx: CellIdx,
    h: usize,
    r: usize,
}

fn apply_hairpin(plan: &mut Plan<'_>, nt_pair: (usize, usize), h: usize, r: usize) {
    let (nt_i, nt_j) = nt_pair;
    plan.trace.codon_assignments[aa_of_nt(nt_i + 1)] = h;
    plan.trace.codon_assignments[aa_of_nt(nt_j - 1)] = r;
    plan.trace.pairs.push(nt_pair);
}

fn apply_two_pair(
    plan: &mut Plan<'_>,
    scratch: &mut Vec<Nucleotide>,
    nt_pair: (usize, usize),
    info: TwoPairInfo,
) {
    let (c, _sp, d, _tp) = decode_enclosed_from_inner(info.inner_idx);
    plan.trace.codon_assignments[c] = info.h;
    plan.trace.codon_assignments[d] = info.r;
    plan.trace.pairs.push(nt_pair);
    traceback_e(plan, info.inner_idx, scratch);
}

fn apply_internal(
    plan: &mut Plan<'_>,
    scratch: &mut Vec<Nucleotide>,
    nt_pair: (usize, usize),
    it: InternalLoopTrace,
) {
    pin_internal_loop_codons(plan, it, nt_pair);
    plan.trace.pairs.push(nt_pair);
    traceback_e(plan, it.inner_idx, scratch);
}

fn pin_internal_loop_codons(plan: &mut Plan<'_>, it: InternalLoopTrace, nt_pair: (usize, usize)) {
    let (nt_i, nt_j) = nt_pair;
    let (c, s_prime, d, t_prime) = decode_enclosed_from_inner(it.inner_idx);
    let nt_p = nt_position(c, s_prime);
    let nt_q = nt_position(d, t_prime);
    let a = aa_of_nt(nt_i);
    let b = aa_of_nt(nt_j);
    let a1 = aa_of_nt(nt_i + 1);
    let b1 = aa_of_nt(nt_j - 1);
    let c1 = aa_of_nt(nt_p - 1);
    let d1 = aa_of_nt(nt_q + 1);
    let oa = (nt_i + 1) % 3;
    let oc = (nt_p - 1) % 3;
    let od = (nt_q + 1) % 3;
    let ob = (nt_j - 1) % 3;

    plan.trace.codon_assignments[c] = it.h;
    plan.trace.codon_assignments[d] = it.r;

    let pair_aas = [a, b, c, d];
    pin_one_flank_pair(
        plan,
        &pair_aas,
        FlankPin {
            first_aa: a1,
            second_aa: c1,
            first_nt: it.xi_,
            second_nt: it.hi_,
            first_offset: oa,
            second_offset: oc,
        },
    );
    pin_one_flank_pair(
        plan,
        &pair_aas,
        FlankPin {
            first_aa: b1,
            second_aa: d1,
            first_nt: it.yj_,
            second_nt: it.kj_,
            first_offset: ob,
            second_offset: od,
        },
    );
}

/// Bundles the per-strand flank-pin descriptor consumed by
/// [`pin_one_flank_pair`]. First and second slots correspond to the
/// `(a1, c1)` / `(b1, d1)` flank ordering — same slot ordering as the
/// fill-time `InnerScope::flank_kinds`.
struct FlankPin {
    first_aa: usize,
    second_aa: usize,
    first_nt: u8,
    second_nt: u8,
    first_offset: usize,
    second_offset: usize,
}

/// Pin codon assignments for one strand's two flank AAs. Mirrors the
/// fill-time classification: closing-pair-pinned AAs are skipped
/// (their assignment is already correct), free AAs read
/// [`crate::plan::Plan::best_synonym_idx`], and the joint flank-AA case
/// (`first_aa == second_aa`, both free) re-searches synonyms for the
/// codon whose nucleotides match both boundary nts at max CAI.
fn pin_one_flank_pair(plan: &mut Plan<'_>, pair_aas: &[usize; 4], pin: FlankPin) {
    let first_collides_pair = pair_aas.contains(&pin.first_aa);
    let second_collides_pair = pair_aas.contains(&pin.second_aa);
    let joint = pin.first_aa == pin.second_aa && !first_collides_pair && !second_collides_pair;

    if joint {
        let ci = find_joint_codon(plan, &pin).unwrap_or_else(|| {
            panic!(
                "internal-loop traceback: no synonym at AA {} matches \
                 nt[{}]={} and nt[{}]={}",
                pin.first_aa, pin.first_offset, pin.first_nt, pin.second_offset, pin.second_nt,
            )
        });
        plan.trace.codon_assignments[pin.first_aa] = ci;
        return;
    }

    if !first_collides_pair {
        plan.trace.codon_assignments[pin.first_aa] =
            plan.best_synonym_idx[pin.first_aa][pin.first_nt as usize][pin.first_offset];
    }
    if !second_collides_pair && pin.second_aa != pin.first_aa {
        plan.trace.codon_assignments[pin.second_aa] =
            plan.best_synonym_idx[pin.second_aa][pin.second_nt as usize][pin.second_offset];
    }
}

/// Find the max-CAI synonym at `pin.first_aa` whose nucleotides at the
/// two flank offsets match the trace's boundary nts. Mirrors the
/// fill-time min-BRD rule: at fixed boundary nts the lowest BRD value
/// collapses to max-CAI among matching synonyms.
fn find_joint_codon(plan: &Plan<'_>, pin: &FlankPin) -> Option<usize> {
    let aa = pin.first_aa;
    let mut best_ci: Option<usize> = None;
    let mut best_cai = f64::NEG_INFINITY;
    for (ci, codon) in plan.codon_lists[aa].iter().enumerate() {
        if codon.nuc[pin.first_offset] == pin.first_nt
            && codon.nuc[pin.second_offset] == pin.second_nt
        {
            let cai = plan.log_cai_dekacal_per_pos[aa][ci];
            if best_ci.is_none() || cai > best_cai {
                best_ci = Some(ci);
                best_cai = cai;
            }
        }
    }
    best_ci
}

fn apply_multi_loop_closure(
    plan: &mut Plan<'_>,
    scratch: &mut Vec<Nucleotide>,
    idx: CellIdx,
    mt: super::MultiLoopTrace,
) {
    let (a, b, s, t, _x, _y) = idx;
    if matches!(mt.case, 2 | 4) {
        plan.trace.codon_assignments[a + 1] = mt.x_prime;
    }
    if matches!(mt.case, 3 | 4) {
        plan.trace.codon_assignments[b - 1] = mt.y_prime;
    }
    plan.trace
        .pairs
        .push((nt_position(a, s), nt_position(b, t)));
    traceback_m(plan, mt.inner_idx, scratch);
}

// ---------------------------------------------------------------------------
// Multi-loop body / fragment walkers (M / N)
// ---------------------------------------------------------------------------

fn traceback_m(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let target = plan.tables.m[plan.tables.flat_index(idx)];
    let trace = find_winning_trace(target, "M", idx, |emit| m_candidates(plan, idx, emit));
    let mt = match trace {
        Trace::M(m) => m,
        _ => panic!("M traceback: non-M variant {trace:?} at {idx:?}"),
    };
    plan.trace.codon_assignments[mt.c] = mt.y_prime;
    plan.trace.codon_assignments[mt.d] = mt.x_prime;
    traceback_n(plan, mt.left_idx, scratch);
    traceback_n(plan, mt.right_idx, scratch);
}

fn traceback_n(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let target = plan.tables.n[plan.tables.flat_index(idx)];
    let trace = find_winning_trace(target, "N", idx, |emit| {
        n_candidates(plan, idx, scratch, emit)
    });
    let nt = match trace {
        Trace::N(n) => n,
        _ => panic!("N traceback: non-N variant {trace:?} at {idx:?}"),
    };
    dispatch_traceback_n_case(plan, idx, nt, scratch);
}

fn dispatch_traceback_n_case(
    plan: &mut Plan<'_>,
    idx: CellIdx,
    trace: NTrace,
    scratch: &mut Vec<Nucleotide>,
) {
    match trace.case {
        1 => traceback_e(plan, idx, scratch),
        2 | 3 => traceback_n(plan, trace.inner_idx_a, scratch),
        4 | 5 => n_unpaired_cross(plan, idx, trace, scratch),
        6 | 7 => n_bifurcation(plan, trace, scratch),
        other => panic!("N traceback: invalid case {other} at {idx:?}"),
    }
}

fn n_unpaired_cross(
    plan: &mut Plan<'_>,
    idx: CellIdx,
    trace: NTrace,
    scratch: &mut Vec<Nucleotide>,
) {
    let (a, b, _s, _t, _x, _y) = idx;
    if trace.case == 4 {
        plan.trace.codon_assignments[a + 1] = trace.aux_codon;
    } else {
        plan.trace.codon_assignments[b - 1] = trace.aux_codon;
    }
    traceback_n(plan, trace.inner_idx_a, scratch);
}

fn n_bifurcation(plan: &mut Plan<'_>, trace: NTrace, scratch: &mut Vec<Nucleotide>) {
    let inner_b = trace
        .inner_idx_b
        .expect("N bifurcation candidates set inner_idx_b");
    let c = trace.inner_idx_a.1;
    plan.trace.codon_assignments[c] = trace.aux_codon;
    if trace.case == 7 {
        let cp1 = inner_b.0;
        plan.trace.codon_assignments[cp1] = trace.aux_codon_2;
    }
    traceback_n(plan, trace.inner_idx_a, scratch);
    traceback_n(plan, inner_b, scratch);
}

// ---------------------------------------------------------------------------
// Exterior walker (O)
// ---------------------------------------------------------------------------

/// Walk `O[idx]`, dispatching the winning case and recursing.
pub(crate) fn traceback_o(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let (a, b, _s, _t, x, y) = idx;
    let i = plan.tables.flat_index(idx);
    if a == b {
        assert!(
            x == y,
            "O case 2 (INF) reached in traceback at {idx:?}; cell={}",
            plan.tables.o[i]
        );
        plan.trace.codon_assignments[a] = x;
        return;
    }
    let target = plan.tables.o[i];
    assert!(
        target < INF,
        "O traceback at INF cell {idx:?} — should be unreachable from any finite parent"
    );
    let trace = find_winning_trace(target, "O", idx, |emit| {
        o_candidates(plan, idx, scratch, emit)
    });
    let ot = match trace {
        Trace::O(o) => o,
        _ => panic!("O traceback: non-O variant {trace:?} at {idx:?}"),
    };
    dispatch_traceback_o_case(plan, ot, scratch);
}

fn dispatch_traceback_o_case(plan: &mut Plan<'_>, trace: OTrace, scratch: &mut Vec<Nucleotide>) {
    match trace.case {
        3 | 4 => traceback_o(plan, trace.inner_idx_a, scratch),
        5 => {
            plan.trace.codon_assignments[trace.inner_idx_a.0] = trace.aux_codon;
            traceback_o(plan, trace.inner_idx_a, scratch);
        }
        6 => {
            plan.trace.codon_assignments[trace.inner_idx_a.1] = trace.aux_codon;
            traceback_o(plan, trace.inner_idx_a, scratch);
        }
        7 => {
            plan.trace.codon_assignments[trace.inner_idx_a.1] = trace.aux_codon;
            o_pair_and_recurse(plan, trace, scratch);
        }
        8 => {
            let c = trace.inner_idx_a.1;
            let c_plus_one = trace.inner_idx_b.expect("O case 8 sets inner_idx_b").0;
            plan.trace.codon_assignments[c] = trace.aux_codon;
            plan.trace.codon_assignments[c_plus_one] = trace.aux_codon_2;
            o_pair_and_recurse(plan, trace, scratch);
        }
        9 => {
            let (a, b, s, t, _x, _y) = trace.inner_idx_a;
            plan.trace
                .pairs
                .push((nt_position(a, s), nt_position(b, t)));
            traceback_e(plan, trace.inner_idx_a, scratch);
        }
        other => panic!("O traceback: invalid case {other}"),
    }
}

fn o_pair_and_recurse(plan: &mut Plan<'_>, trace: OTrace, scratch: &mut Vec<Nucleotide>) {
    let inner_b = trace
        .inner_idx_b
        .expect("O bifurcation candidates set inner_idx_b");
    let (cp_aa, b_aa, t_inner, t_outer, _x, _y) = inner_b;
    plan.trace
        .pairs
        .push((nt_position(cp_aa, t_inner), nt_position(b_aa, t_outer)));
    traceback_o(plan, trace.inner_idx_a, scratch);
    traceback_e(plan, inner_b, scratch);
}
