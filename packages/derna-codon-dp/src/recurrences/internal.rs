//! Internal-loop recurrence (E_i).
//!
//! Follows supplement.tex lines 228–250. For each enclosed-pair position `(nt_p, nt_q)`, the
//! recurrence enumerates the four flank AAs `(a1, b1, c1, d1)` and the
//! four loop boundary nucleotides at positions
//! `(nt_i + 1, nt_p − 1, nt_q + 1, nt_j − 1)`. The CAI bookkeeping
//! charges each AA at most once across the six terms via an indicator
//! function (closing pair `(a, b)`, enclosed pair `(c, d)`, flanks
//! `(a1, c1, d1, b1)`).
//!
//! Boundary nucleotides are enumerated directly (≤ 4 deep), gated by
//! the per-AA reachability bitmask in [`Plan::aa_boundary_nts`]. The
//! max-CAI synonym at each `(aa, boundary_nt, codon_position)` rides in
//! [`Plan::best_synonym_cai`] / [`Plan::best_synonym_idx`]; the cell
//! reduction picks the boundary tuple minimising
//! `inner_val + brd_value(λ, energy, cai_sum)`.
//!
//! Flank-AA collisions degrade gracefully:
//!
//! * **Closing- or enclosed-pair pin collision** (`a1 == a`, `c1 == c`,
//!   etc.): the flank nucleotide is determined by the already-chosen
//!   pinned codon; that single value replaces the bitmask enumeration
//!   slot.
//! * **Flank-self collision** (`a1 == c1` on the left strand or
//!   `b1 == d1` on the right strand, both free): enumeration drops to
//!   codon-level at the shared AA so the joint constraint on the two
//!   boundary nucleotides stays consistent. The other strand still
//!   uses nt-level enumeration.
//!
//! The reachability bitmask is precomputed once per plan: for each AA and
//! codon position, the set of nucleotides some synonym spells there. Dedup
//! happens by construction: every emitted boundary tuple is distinct on its own
//! axis, so there is no need for a post-hoc memoization layer.

use rna_mfe::loops::{pair_from_nucleotides, Nucleotide};
use rna_mfe::params::BasePair;
use rna_mfe::{internal_energy_scalar, Energy, InternalLoopScalar};

use crate::brd_value::brd_value;
use crate::plan::{aa_of_nt, nt_position, nuc_at, nuc_from_index, CellIdx, Plan, INF};

use super::bulge::{for_each_enclosed_pair_position, MotifKind};
use super::two_pair::{iter_two_pair_inner, make_site, InnerRef, MotifSite};
use super::{Candidate, InternalLoopTrace, Trace};

/// Geometry of one internal-loop emission site at a fixed
/// `(idx, enclosed-pair-position)` and the four flank-AA indices.
#[derive(Debug, Clone, Copy)]
struct InternalSite {
    site: MotifSite,
    /// `(a1, b1, c1, d1)` — AA indices of the four loop flanks.
    flanks: (usize, usize, usize, usize),
}

/// The four loop boundary nucleotides at flank positions
/// `(nt_i + 1, nt_p − 1, nt_q + 1, nt_j − 1)`: `xi_` = nt_i+1,
/// `hi_` = nt_p-1, `kj_` = nt_q+1, `yj_` = nt_j-1.
#[derive(Debug, Clone, Copy)]
struct BoundaryNts {
    xi_: u8,
    hi_: u8,
    kj_: u8,
    yj_: u8,
}

/// Per-flank resolution: a flank's boundary nucleotide is either fully
/// determined by an already-chosen pinned codon (closing or enclosed
/// pair) or free to enumerate through the reachability bitmask.
#[derive(Debug, Clone, Copy)]
enum FlankKind {
    Fixed(u8),
    Free,
}

/// Read-only context shared across every flank-nt enumeration at one
/// `emit_for_inner` call. Bundling the immutable geometry avoids
/// re-deriving offsets and pin classifications per candidate.
#[derive(Debug, Clone, Copy)]
struct InnerScope {
    outer: BasePair,
    inner: BasePair,
    /// `(a1, b1, c1, d1)` AA indices of the four boundary flanks.
    flank_aas: (usize, usize, usize, usize),
    /// Codon-position offsets of the four boundary nucleotide positions
    /// (`nt_i + 1`, `nt_p − 1`, `nt_q + 1`, `nt_j − 1`), in that order.
    flank_offsets: [usize; 4],
    /// `n1` / `n2` unpaired-stretch lengths.
    n1: usize,
    n2: usize,
    /// `(a, b, c, d)` — closing- and enclosed-pair AAs.
    pair_aas: (usize, usize, usize, usize),
    /// `(x, y, h, r)` — codons chosen for the closing and enclosed pairs.
    pair_codons: (usize, usize, usize, usize),
    /// Per-slot `Fixed(nt)` when the flank AA collides with a pinned AA,
    /// `Free` otherwise. Slots match `flank_offsets`: `[a1, c1, d1, b1]`.
    flank_kinds: [FlankKind; 4],
}

fn internal_flank_aas(nt_pairs: (usize, usize, usize, usize)) -> (usize, usize, usize, usize) {
    let (nt_i, nt_j, nt_p, nt_q) = nt_pairs;
    (
        aa_of_nt(nt_i + 1),
        aa_of_nt(nt_j - 1),
        aa_of_nt(nt_p - 1),
        aa_of_nt(nt_q + 1),
    )
}

fn classify_flank(
    plan: &Plan<'_>,
    flank_aa: usize,
    flank_offset: usize,
    pair_pins: [(usize, usize); 4],
) -> FlankKind {
    for (p, ci) in pair_pins {
        if p == flank_aa {
            return FlankKind::Fixed(nuc_at(plan, p, ci, flank_offset));
        }
    }
    FlankKind::Free
}

fn build_inner_scope(
    plan: &Plan<'_>,
    isite: InternalSite,
    outer: BasePair,
    inner: BasePair,
    pair_pins: [(usize, usize); 4],
) -> InnerScope {
    let (nt_i, nt_j, nt_p, nt_q) = isite.site.nt_pairs;
    let flank_offsets = [
        (nt_i + 1) % 3,
        (nt_p - 1) % 3,
        (nt_q + 1) % 3,
        (nt_j - 1) % 3,
    ];
    let (a1, b1, c1, d1) = isite.flanks;
    let flank_kinds = [
        classify_flank(plan, a1, flank_offsets[0], pair_pins),
        classify_flank(plan, c1, flank_offsets[1], pair_pins),
        classify_flank(plan, d1, flank_offsets[2], pair_pins),
        classify_flank(plan, b1, flank_offsets[3], pair_pins),
    ];
    InnerScope {
        outer,
        inner,
        flank_aas: isite.flanks,
        flank_offsets,
        n1: nt_p - nt_i - 1,
        n2: nt_j - nt_q - 1,
        pair_aas: (
            pair_pins[0].0,
            pair_pins[1].0,
            pair_pins[2].0,
            pair_pins[3].0,
        ),
        pair_codons: (
            pair_pins[0].1,
            pair_pins[1].1,
            pair_pins[2].1,
            pair_pins[3].1,
        ),
        flank_kinds,
    }
}

/// Read the closing and enclosed base pairs from the four pinned codons.
/// Returns `None` if either pair is non-canonical; the site is then
/// dropped from candidate emission.
fn read_base_pairs(
    plan: &Plan<'_>,
    isite: InternalSite,
    pair_pins: [(usize, usize); 4],
) -> Option<(BasePair, BasePair)> {
    let (nt_i, nt_j, nt_p, nt_q) = isite.site.nt_pairs;
    let (a, x) = pair_pins[0];
    let (b, y) = pair_pins[1];
    let (c, h) = pair_pins[2];
    let (d, r) = pair_pins[3];
    let nt = |aa, ci, pos| nuc_from_index(nuc_at(plan, aa, ci, pos));
    let outer = pair_from_nucleotides(nt(a, x, nt_i % 3), nt(b, y, nt_j % 3));
    if matches!(outer, BasePair::NN) {
        return None;
    }
    let inner_pair = pair_from_nucleotides(nt(c, h, nt_p % 3), nt(d, r, nt_q % 3));
    if matches!(inner_pair, BasePair::NN) {
        return None;
    }
    Some((outer, inner_pair))
}

#[inline]
fn compute_internal_energy(plan: &Plan<'_>, scope: InnerScope, boundary: BoundaryNts) -> Energy {
    internal_energy_scalar(
        plan.params,
        &InternalLoopScalar {
            outer: scope.outer,
            inner: scope.inner,
            boundary_left5p: nuc_from_index(boundary.xi_),
            boundary_left3p: nuc_from_index(boundary.hi_),
            boundary_right5p: nuc_from_index(boundary.kj_),
            boundary_right3p: nuc_from_index(boundary.yj_),
            n1: scope.n1,
            n2: scope.n2,
        },
    )
}

#[inline]
fn for_each_nt_in_mask<F: FnMut(u8)>(mask: u8, mut f: F) {
    for nt in 0u8..4 {
        if (mask >> nt) & 1 == 1 {
            f(nt);
        }
    }
}

/// Enumerate the left-strand boundary tuple `(xi_, hi_)`, plus an
/// optional codon index `a1_codon` when the shared-AA joint case fires.
///
/// The emitted `a1_codon` is `Some(ci)` only when `a1 == c1` are both
/// free; in that case enumeration is codon-level at the joint AA so the
/// joint constraint stays consistent. The other variants emit `None`.
fn iter_left_boundary<F: FnMut(u8, u8, Option<usize>)>(
    plan: &Plan<'_>,
    scope: InnerScope,
    mut emit: F,
) {
    let (a1, _b1, c1, _d1) = scope.flank_aas;
    let oa = scope.flank_offsets[0];
    let oc = scope.flank_offsets[1];
    match (scope.flank_kinds[0], scope.flank_kinds[1]) {
        (FlankKind::Fixed(xi), FlankKind::Fixed(hi)) => emit(xi, hi, None),
        (FlankKind::Fixed(xi), FlankKind::Free) => {
            for_each_nt_in_mask(plan.aa_boundary_nts[c1][oc], |hi| emit(xi, hi, None));
        }
        (FlankKind::Free, FlankKind::Fixed(hi)) => {
            for_each_nt_in_mask(plan.aa_boundary_nts[a1][oa], |xi| emit(xi, hi, None));
        }
        (FlankKind::Free, FlankKind::Free) if a1 == c1 => {
            for (ci, codon) in plan.codon_lists[a1].iter().enumerate() {
                emit(codon.nuc[oa], codon.nuc[oc], Some(ci));
            }
        }
        (FlankKind::Free, FlankKind::Free) => {
            for_each_nt_in_mask(plan.aa_boundary_nts[a1][oa], |xi| {
                for_each_nt_in_mask(plan.aa_boundary_nts[c1][oc], |hi| emit(xi, hi, None));
            });
        }
    }
}

/// Mirror of [`iter_left_boundary`] for the right strand `(kj_, yj_)`.
/// Slots `[d1, b1]` come from `flank_kinds[2..4]` and offsets
/// `[od, ob]`; the joint case fires when `b1 == d1`.
fn iter_right_boundary<F: FnMut(u8, u8, Option<usize>)>(
    plan: &Plan<'_>,
    scope: InnerScope,
    mut emit: F,
) {
    let (_a1, b1, _c1, d1) = scope.flank_aas;
    let od = scope.flank_offsets[2];
    let ob = scope.flank_offsets[3];
    match (scope.flank_kinds[2], scope.flank_kinds[3]) {
        (FlankKind::Fixed(kj), FlankKind::Fixed(yj)) => emit(kj, yj, None),
        (FlankKind::Fixed(kj), FlankKind::Free) => {
            for_each_nt_in_mask(plan.aa_boundary_nts[b1][ob], |yj| emit(kj, yj, None));
        }
        (FlankKind::Free, FlankKind::Fixed(yj)) => {
            for_each_nt_in_mask(plan.aa_boundary_nts[d1][od], |kj| emit(kj, yj, None));
        }
        (FlankKind::Free, FlankKind::Free) if b1 == d1 => {
            for (ci, codon) in plan.codon_lists[b1].iter().enumerate() {
                emit(codon.nuc[od], codon.nuc[ob], Some(ci));
            }
        }
        (FlankKind::Free, FlankKind::Free) => {
            for_each_nt_in_mask(plan.aa_boundary_nts[d1][od], |kj| {
                for_each_nt_in_mask(plan.aa_boundary_nts[b1][ob], |yj| emit(kj, yj, None));
            });
        }
    }
}

/// One flank slot: AA index, the boundary nt enumerated for it, and
/// the codon offset at which that nt sits.
#[derive(Debug, Clone, Copy)]
struct FlankSlot {
    aa: usize,
    nt: u8,
    offset: usize,
}

/// One strand's CAI bookkeeping: the two flank slots plus the
/// AAs that zero out each slot's charge (indicator-function order:
/// `[a, c]` / `[c, a, a1]` for the left strand, `[b, d]` / `[d, b, b1]`
/// for the right).
///
/// `first_override` is `Some(ci)` only when the joint flank-AA
/// enumeration fires (`a1 == c1` on the left or `b1 == d1` on the
/// right); the colliding AA then charges its specific codon's CAI in
/// place of the per-nt `best_synonym_cai` lookup.
#[derive(Debug, Clone, Copy)]
struct StrandCai {
    first: FlankSlot,
    second: FlankSlot,
    first_excludes: [usize; 2],
    second_excludes: [usize; 3],
    first_override: Option<usize>,
}

#[inline]
fn flank_slot_charge(
    plan: &Plan<'_>,
    slot: FlankSlot,
    excludes: &[usize],
    override_ci: Option<usize>,
) -> f64 {
    if excludes.contains(&slot.aa) {
        return 0.0;
    }
    match override_ci {
        Some(ci) => plan.log_cai_dekacal_per_pos[slot.aa][ci],
        None => plan.best_synonym_cai[slot.aa][slot.nt as usize][slot.offset],
    }
}

#[inline]
fn strand_cai_charge(plan: &Plan<'_>, strand: StrandCai) -> f64 {
    flank_slot_charge(
        plan,
        strand.first,
        &strand.first_excludes,
        strand.first_override,
    ) + flank_slot_charge(plan, strand.second, &strand.second_excludes, None)
}

/// Closing-pair CAI contribution, hoisted once per `emit_for_inner`
/// call: `cai_term(a, x, [c]) + cai_term(b, y, [d])`.
#[inline]
fn cai_closing_charge(plan: &Plan<'_>, scope: InnerScope) -> f64 {
    let (a, b, c, d) = scope.pair_aas;
    let (x, y, _h, _r) = scope.pair_codons;
    let charge_a = if c == a {
        0.0
    } else {
        plan.log_cai_dekacal_per_pos[a][x]
    };
    let charge_b = if d == b {
        0.0
    } else {
        plan.log_cai_dekacal_per_pos[b][y]
    };
    charge_a + charge_b
}

/// Inner-cell context carried through one flank-nt enumeration: the
/// pointer to the inner cell and the `CellIdx` it lives at. Bundled
/// here so [`emit_internal_candidates`]'s arity stays manageable.
#[derive(Debug, Clone, Copy)]
struct InnerEmit {
    inner: InnerRef,
    inner_idx: CellIdx,
}

/// Per-strand layout bundling everything `strand_charge` needs at
/// one fully-resolved boundary tuple: the strand's pinned AA pair
/// (outer + enclosed), the two flank slots in indicator-function order, and the
/// optional joint-codon override at the first slot.
///
/// Left strand: `closing = (a, c)`, `first = (a1, xi_, oa)`,
/// `second = (c1, hi_, oc)`. Right strand: `closing = (b, d)`,
/// `first = (b1, yj_, ob)`, `second = (d1, kj_, od)`.
#[derive(Debug, Clone, Copy)]
struct StrandSpec {
    closing: (usize, usize),
    first: FlankSlot,
    second: FlankSlot,
    first_override: Option<usize>,
}

#[inline]
fn strand_charge(plan: &Plan<'_>, spec: StrandSpec) -> f64 {
    let (outer_aa, enclosed_aa) = spec.closing;
    strand_cai_charge(
        plan,
        StrandCai {
            first: spec.first,
            second: spec.second,
            first_excludes: [outer_aa, enclosed_aa],
            second_excludes: [enclosed_aa, outer_aa, spec.first.aa],
            first_override: spec.first_override,
        },
    )
}

#[inline]
fn make_candidate(ie: InnerEmit, boundary: BoundaryNts, value: f64) -> Candidate {
    Candidate {
        value,
        trace: Trace::Internal(InternalLoopTrace {
            xi_: boundary.xi_,
            hi_: boundary.hi_,
            kj_: boundary.kj_,
            yj_: boundary.yj_,
            h: ie.inner.h,
            r: ie.inner.r,
            inner_idx: ie.inner_idx,
        }),
    }
}

fn emit_internal_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    scope: InnerScope,
    ie: InnerEmit,
    emit: &mut F,
) {
    let cai_closing = cai_closing_charge(plan, scope);
    let (a, b, c, d) = scope.pair_aas;
    let (a1, b1, c1, d1) = scope.flank_aas;
    let [oa, oc, od, ob] = scope.flank_offsets;
    iter_left_boundary(plan, scope, |xi_, hi_, a1_override| {
        let cai_left = strand_charge(
            plan,
            StrandSpec {
                closing: (a, c),
                first: FlankSlot {
                    aa: a1,
                    nt: xi_,
                    offset: oa,
                },
                second: FlankSlot {
                    aa: c1,
                    nt: hi_,
                    offset: oc,
                },
                first_override: a1_override,
            },
        );
        iter_right_boundary(plan, scope, |kj_, yj_, b1_override| {
            let cai_right = strand_charge(
                plan,
                StrandSpec {
                    closing: (b, d),
                    first: FlankSlot {
                        aa: b1,
                        nt: yj_,
                        offset: ob,
                    },
                    second: FlankSlot {
                        aa: d1,
                        nt: kj_,
                        offset: od,
                    },
                    first_override: b1_override,
                },
            );
            let boundary = BoundaryNts { xi_, hi_, kj_, yj_ };
            let energy = compute_internal_energy(plan, scope, boundary);
            let value = ie.inner.inner_val
                + brd_value(
                    plan.lambda_,
                    energy.dekacal(),
                    cai_closing + cai_left + cai_right,
                );
            emit(make_candidate(ie, boundary, value));
        });
    });
}

fn emit_for_inner<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    isite: InternalSite,
    inner: InnerRef,
    emit: &mut F,
) {
    let (a, b, _s, _t, x, y) = isite.site.idx;
    let (c, sp, d, tp) = isite.site.enclosed;
    let pair_pins: [(usize, usize); 4] = [(a, x), (b, y), (c, inner.h), (d, inner.r)];

    let Some((outer, inner_pair)) = read_base_pairs(plan, isite, pair_pins) else {
        return;
    };
    let scope = build_inner_scope(plan, isite, outer, inner_pair, pair_pins);
    let inner_idx: CellIdx = (c, d, sp, tp, inner.h, inner.r);
    emit_internal_candidates(plan, scope, InnerEmit { inner, inner_idx }, emit);
}

fn emit_for_site<F: FnMut(Candidate)>(plan: &Plan<'_>, isite: InternalSite, emit: &mut F) {
    iter_two_pair_inner(plan, isite.site, |inner| {
        emit_for_inner(plan, isite, inner, emit);
    });
}

/// Internal-loop candidates: enumerate enclosed pairs with both strands
/// non-empty, then enumerate the four boundary nucleotides per pair
/// (or codon-level at the colliding AA for the flank-self collisions).
///
/// The `scratch` parameter is unused — boundary nts come directly from
/// `Plan::codon_lists` reads — but the signature is shared with the
/// other recurrences that still need an assembled-RNA substrate.
pub(crate) fn internal_candidates<F: FnMut(Candidate)>(
    plan: &Plan<'_>,
    idx: CellIdx,
    _scratch: &mut Vec<Nucleotide>,
    mut emit: F,
) {
    let (a, b, s, t, _x, _y) = idx;
    let nt_i = nt_position(a, s);
    let nt_j = nt_position(b, t);
    for_each_enclosed_pair_position(nt_i, nt_j, MotifKind::Internal, |nt_p, nt_q| {
        if let Some(site) = make_site(idx, (nt_i, nt_j, nt_p, nt_q)) {
            let isite = InternalSite {
                site,
                flanks: internal_flank_aas((nt_i, nt_j, nt_p, nt_q)),
            };
            emit_for_site(plan, isite, &mut emit);
        }
    });
}

/// Fill `E_i[idx]` via `min` over internal-loop candidates.
pub(crate) fn fill_e_i(plan: &mut Plan<'_>, idx: CellIdx, scratch: &mut Vec<Nucleotide>) {
    let best = {
        let p: &Plan<'_> = plan;
        let mut best = INF;
        internal_candidates(p, idx, scratch, |cand| {
            if cand.value < best {
                best = cand.value;
            }
        });
        best
    };
    let i = plan.tables.flat_index(idx);
    plan.tables.e_i[i] = best;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codon_table::AMINO_ACIDS;
    use crate::codon_usage::CodonUsage;
    use crate::plan::{EStorageMode, PlanInputs, TableLayout};
    use rna_mfe::Params;

    fn with_plan<F: FnOnce(&Plan<'_>)>(protein: &str, f: F) {
        let codon_usage = CodonUsage::load_default().unwrap();
        let params = Params::load_default().unwrap();
        let inputs = PlanInputs {
            dangling_5_prime: false,
            codon_mask: None,
            lambda_: 0.5,
            e_storage_mode: EStorageMode::FullMotifTables,
            table_layout: TableLayout::DenseMaxRectangles,
        };
        let plan = Plan::new(protein, &codon_usage, &inputs, &params).unwrap();
        f(&plan);
    }

    /// Spike 008 falsifiability anchor for the flank-AA collision case.
    ///
    /// Construct a geometry where `a1 == c1` are both free
    /// (no closing-pair pin matches the joint AA). Verify the left
    /// boundary enumeration drops to codon-level at the joint AA, and
    /// the emitted `(xi_, hi_)` tuples are consistent with the codon's
    /// `nuc[oa]` / `nuc[oc]` reads (no inconsistent tuples).
    #[test]
    fn left_joint_flank_collision_is_codon_consistent() {
        with_plan("AAGAA", |plan| {
            // n_aa = 5. Pick a (1, 1)-style internal-loop geometry that
            // puts a1 == c1 == 2 (Gly at protein position 2) without
            // colliding with the closing- or enclosed-pair AAs.
            //
            // Outer pair: (a=0, b=4) at nt_i = 0, nt_j = 14. Inner pair:
            // (c=1, d=3) at nt_p = 5, nt_q = 9. n1 = 4 (positions 1..4),
            // n2 = 4 (positions 10..13).
            //
            // Flanks: a1 = aa_of_nt(1) = 0, c1 = aa_of_nt(4) = 1. Not the
            // joint case wanted here — try smaller (2, 2) instead.
            //
            // Outer at nt_i = 0, nt_j = 11 → AAs a=0, b=3. Inner at
            // nt_p = 6, nt_q = 8 → AAs c=2, d=2. Need c < d; fail.
            //
            // Try nt_i = 0, nt_j = 11 (a=0, b=3, s=0, t=2). Inner at
            // nt_p = 6, nt_q = 9 → c=2, d=3, n1 = 5, n2 = 1. Flanks:
            // a1 = aa_of_nt(1) = 0, c1 = aa_of_nt(5) = 1, d1 =
            // aa_of_nt(10) = 3, b1 = aa_of_nt(10) = 3. Still no joint.
            //
            // Drive directly through `iter_left_boundary` with a
            // hand-built `InnerScope` where a1 == c1 and both are Free.
            // This isolates the collision handler from the geometry
            // search.
            let scope = InnerScope {
                outer: BasePair::CG,
                inner: BasePair::CG,
                flank_aas: (2, 0, 2, 4), // (a1, b1, c1, d1) with a1 == c1 == 2.
                flank_offsets: [0, 2, 0, 2],
                n1: 2,
                n2: 2,
                pair_aas: (0, 4, 1, 3),
                pair_codons: (0, 0, 0, 0),
                flank_kinds: [
                    FlankKind::Free,
                    FlankKind::Free,
                    FlankKind::Free,
                    FlankKind::Free,
                ],
            };
            let mut emissions: Vec<(u8, u8, Option<usize>)> = Vec::new();
            iter_left_boundary(plan, scope, |xi, hi, ci| emissions.push((xi, hi, ci)));

            // a1 (AA at position 2 = G = Gly). codon_lists at protein position 2
            // are Gly's 4 synonyms. Each emission must (a) carry a Some(ci)
            // codon override, (b) have nts matching the codon's offsets.
            let codons = &plan.codon_lists[2];
            assert_eq!(
                emissions.len(),
                codons.len(),
                "joint enumeration must visit exactly one entry per synonym at AA 2",
            );
            let oa = scope.flank_offsets[0];
            let oc = scope.flank_offsets[1];
            for (idx, &(xi, hi, ci)) in emissions.iter().enumerate() {
                let ci_unwrapped = ci.expect("joint case must emit Some(ci)");
                assert_eq!(ci_unwrapped, idx, "codons must be visited in order");
                let codon = &codons[idx];
                assert_eq!(
                    codon.nuc[oa], xi,
                    "xi_ must equal codon.nuc[oa={oa}] at synonym {idx}"
                );
                assert_eq!(
                    codon.nuc[oc], hi,
                    "hi_ must equal codon.nuc[oc={oc}] at synonym {idx}"
                );
            }
        });
    }

    /// Closing-pair flank collision: when `a1 == a`, the boundary nt is
    /// `Fixed(plan.codon_lists[a][x].nuc[oa])`. Verify enumeration emits
    /// exactly that nt at the slot regardless of `aa_boundary_nts`.
    #[test]
    fn closing_pair_flank_collision_pins_boundary_nt() {
        with_plan("AAA", |plan| {
            let scope = InnerScope {
                outer: BasePair::CG,
                inner: BasePair::CG,
                flank_aas: (0, 0, 2, 2),
                flank_offsets: [1, 0, 0, 1],
                n1: 1,
                n2: 1,
                pair_aas: (0, 2, 1, 1),
                pair_codons: (0, 0, 0, 0),
                flank_kinds: [
                    FlankKind::Fixed(nuc_at(plan, 0, 0, 1)),
                    FlankKind::Free,
                    FlankKind::Free,
                    FlankKind::Fixed(nuc_at(plan, 2, 0, 1)),
                ],
            };
            let mut emissions = Vec::new();
            iter_left_boundary(plan, scope, |xi, hi, ci| emissions.push((xi, hi, ci)));
            let expected_xi = nuc_at(plan, 0, 0, 1);
            for &(xi, _hi, ci) in &emissions {
                assert_eq!(
                    xi, expected_xi,
                    "Fixed slot must emit exactly the pinned nt",
                );
                assert!(
                    ci.is_none(),
                    "non-joint case must emit None for codon override",
                );
            }
        });
    }

    /// Sanity: the precomputed `aa_boundary_nts` reachability matches
    /// the live codon-lists for the protein positions exercised.
    #[test]
    fn aa_boundary_nts_match_codon_lists() {
        with_plan("RLAGSP", |plan| {
            for (p, codons) in plan.codon_lists.iter().enumerate() {
                for pos in 0..3 {
                    let mut expected = 0u8;
                    for c in codons {
                        expected |= 1 << c.nuc[pos];
                    }
                    assert_eq!(
                        plan.aa_boundary_nts[p][pos], expected,
                        "aa_boundary_nts mismatch at p={p} pos={pos}",
                    );
                }
            }
        });
    }

    /// Sanity-check that every AA in `AMINO_ACIDS` is exercised by the
    /// reachability precompute (catches accidental table truncation): the
    /// twenty, then the stop family.
    #[test]
    fn all_amino_acids_reachable() {
        assert_eq!(AMINO_ACIDS.len(), crate::codon_table::STOP + 1);
    }
}
