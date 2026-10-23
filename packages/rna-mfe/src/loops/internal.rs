//! Internal loop evaluator.
//!
//! An internal loop has unpaired nucleotides on *both* sides of the
//! loop. Vienna's reference is
//! `src/ViennaRNA/eval/eval_internal.c::vrna_E_internal` (v2.7.0
//! branch); the long axis count `nl = max(n1, n2)` and short count
//! `ns = min(n1, n2)` drive the dispatch.
//!
//! Sections consulted (per shape):
//!
//! | Shape `(n1, n2)`             | Tables                                                                            |
//! |------------------------------|-----------------------------------------------------------------------------------|
//! | `(1, 1)`                     | `int11`                                                                           |
//! | `(1, 2) \| (2, 1)`           | `int21`                                                                           |
//! | `(2, 2)`                     | `int22`                                                                           |
//! | `(1, n>=3) \| (n>=3, 1)`     | `interior` (size table), `mismatch_interior_1n`, `NINIO`                          |
//! | `(2, 3) \| (3, 2)`           | `interior[5]` (hardcoded), `mismatch_interior_23`, `NINIO` (hardcoded `ninio_37`) |
//! | other (`ns >= 2 && nl >= 4`) | `interior` (size table), `mismatch_interior`, `NINIO`                             |
//!
//! ## OQ1 resolution — `mismatch_interior_1n` cutoff
//!
//! Vienna applies `mismatch_interior_1n` for `(1, n)` only when
//! `n >= 3` (equivalently `(ns, nl) = (1, nl >= 3)`). The `(1, 2)` /
//! `(2, 1)` cases route to `int21` instead — see the `case 1:` branch
//! of `eval_internal.c::vrna_E_internal` (v2.7.0
//! `eval_internal.c:141-159`). The `1xn` else-arm fires only after
//! `nl == 1` (1×1) and `nl == 2` (1×2 / 2×1) are handled, so any
//! `(1, n)` shape with `n >= 3` falls through to the generic
//! `interior` size table plus the `mismatch1nI` (= this crate's
//! `mismatch_interior_1n`) lookup.

use super::{Energy, Nucleotide, MAX_TABULATED_SIZE};
use crate::params::{BasePair, Int11Key, Int21Key, Int22Key, Params, Triplet37, INF as PARAM_INF};

/// An internal loop's two closing pairs (sequence-order) and the
/// unpaired stretches between them. `left` is the 5'-side strand
/// (positions `i+1 .. k-1`); `right` is the 3'-side strand
/// (positions `l+1 .. j-1`) where `(i, j)` is the outer pair and
/// `(k, l)` is the inner pair.
#[derive(Debug, Clone, Copy)]
pub struct InternalLoop<'a> {
    pub outer: BasePair,
    pub inner: BasePair,
    pub left: &'a [Nucleotide],
    pub right: &'a [Nucleotide],
}

/// All inputs to [`internal_energy_scalar`] except the parameter
/// table. Each parameter name follows the C++ `Zuker.cpp:1207-1229`
/// convention for the boundary nts of an internal loop:
///
/// * `boundary_left5p` is the loop's 5' opening flank at `i+1` (`xi_`).
/// * `boundary_left3p` is the loop's 3' closure on the left strand at
///   `p-1` (`_hi`). Equals `boundary_left5p` when `n1 == 1`.
/// * `boundary_right5p` is the loop's 5' opening flank on the right
///   strand at `q+1` (`kj_`).
/// * `boundary_right3p` is the loop's 3' closure at `j-1` (`_yj`).
///   Equals `boundary_right5p` when `n2 == 1`.
///
/// `n1` / `n2` are the left / right unpaired-stretch lengths.
#[derive(Debug, Clone, Copy)]
pub struct InternalLoopScalar {
    pub outer: BasePair,
    pub inner: BasePair,
    pub boundary_left5p: Nucleotide,
    pub boundary_left3p: Nucleotide,
    pub boundary_right5p: Nucleotide,
    pub boundary_right3p: Nucleotide,
    pub n1: usize,
    pub n2: usize,
}

/// Internal-loop energy. Dispatches on the `(n1, n2) =
/// (left.len(), right.len())` shape per the table at the top of the
/// module. `(0, 0)` (a true stack) is rejected as a programmer
/// error — callers should consult [`super::stack`] for that case.
pub fn internal_energy(params: &Params, lp: &InternalLoop<'_>) -> Energy {
    let n1 = lp.left.len();
    let n2 = lp.right.len();
    debug_assert!(
        !(n1 == 0 && n2 == 0),
        "internal_energy called on a stack — use stack_energy instead",
    );
    internal_energy_scalar(
        params,
        &InternalLoopScalar {
            outer: lp.outer,
            inner: lp.inner,
            boundary_left5p: lp.left[0],
            boundary_left3p: lp.left[n1 - 1],
            boundary_right5p: lp.right[0],
            boundary_right3p: lp.right[n2 - 1],
            n1,
            n2,
        },
    )
}

/// Scalar-args overload of [`internal_energy`]. Reads the four loop
/// boundary nucleotides directly from [`InternalLoopScalar`] instead
/// of slicing through a `[Nucleotide]` substrate. Callers that
/// already know the boundary nts can skip slice construction
/// entirely.
///
/// The dispatch table matches [`internal_energy`] line-for-line.
#[inline(always)]
pub fn internal_energy_scalar(params: &Params, b: &InternalLoopScalar) -> Energy {
    debug_assert!(
        !(b.n1 == 0 && b.n2 == 0),
        "internal_energy_scalar called on a stack — use stack_energy instead",
    );
    match (b.n1, b.n2) {
        (1, 1) => energy_int11_scalar(params, b),
        (1, 2) => energy_int21_scalar(params, b, true),
        (2, 1) => energy_int21_scalar(params, b, false),
        (2, 2) => energy_int22_scalar(params, b),
        (1, n) | (n, 1) if n >= 3 => energy_one_by_n_scalar(params, b, n),
        (2, 3) | (3, 2) => energy_two_by_three_scalar(params, b),
        _ => energy_generic_scalar(params, b),
    }
}

/// Vienna `int11[type][type_2][si1][sj1]`. When `n2 == 1`, the
/// `right[0]` read in [`internal_energy`] is the loop's only
/// right-side mismatch nucleotide — equivalently
/// `boundary_right5p`, which the caller's convention forces to equal
/// `boundary_right3p`.
#[inline]
fn energy_int11_scalar(params: &Params, b: &InternalLoopScalar) -> Energy {
    Energy(params.int11.get(Int11Key {
        outer: b.outer,
        inner_reversed: b.inner.reversed(),
        m5: b.boundary_left5p.to_nuc5(),
        m3: b.boundary_right5p.to_nuc5(),
    }))
}

/// Vienna `int21` covers `(1, 2)` and `(2, 1)` symmetrically.
/// `single_on_left = true` for `(1, 2)`: the singleton-strand
/// nucleotide is `boundary_left5p` and the two-unpaired side feeds
/// `(boundary_right5p, boundary_right3p)`. `false` flips both.
#[inline]
fn energy_int21_scalar(params: &Params, b: &InternalLoopScalar, single_on_left: bool) -> Energy {
    let key = if single_on_left {
        Int21Key {
            pair_a: b.outer,
            pair_b: b.inner.reversed(),
            single: b.boundary_left5p.to_nuc5(),
            double_inner: b.boundary_right5p.to_nuc5(),
            double_outer: b.boundary_right3p.to_nuc5(),
        }
    } else {
        Int21Key {
            pair_a: b.inner.reversed(),
            pair_b: b.outer,
            single: b.boundary_right5p.to_nuc5(),
            double_inner: b.boundary_left5p.to_nuc5(),
            double_outer: b.boundary_left3p.to_nuc5(),
        }
    };
    Energy(params.int21.get(key))
}

/// Vienna `int22[type][type_2][si1][sp1][sq1][sj1]`. Non-canonical
/// closing pairs return the Vienna `INF` sentinel.
#[inline]
fn energy_int22_scalar(params: &Params, b: &InternalLoopScalar) -> Energy {
    let Some(outer_idx) = b.outer.as_int22_index() else {
        return Energy(PARAM_INF);
    };
    let Some(inner_reversed_idx) = b.inner.reversed().as_int22_index() else {
        return Energy(PARAM_INF);
    };
    let mismatches = [
        b.boundary_left5p.as_int22_index(),
        b.boundary_left3p.as_int22_index(),
        b.boundary_right5p.as_int22_index(),
        b.boundary_right3p.as_int22_index(),
    ];
    Energy(params.int22.get(Int22Key {
        outer: outer_idx,
        inner_reversed: inner_reversed_idx,
        mismatches,
    }))
}

/// `(1, n>=3)` and `(n>=3, 1)` share the `mismatch_interior_1n` table
/// and `size(n+1) + ninio(n-1)` decomposition. `n` is the long strand's
/// unpaired count.
#[inline]
fn energy_one_by_n_scalar(params: &Params, b: &InternalLoopScalar, n: usize) -> Energy {
    interior_size_penalty(params, n + 1)
        + ninio_asymmetry(params, n - 1)
        + mismatch_sum_scalar(&params.mismatch_interior_1n, b)
}

/// `(2, 3)` / `(3, 2)` share Vienna's hardcoded
/// `interior_loop[5] + ninio_37` decomposition.
#[inline]
fn energy_two_by_three_scalar(params: &Params, b: &InternalLoopScalar) -> Energy {
    Energy(params.interior.values[5])
        + Energy(params.ninio.ninio_37)
        + mismatch_sum_scalar(&params.mismatch_interior_23, b)
}

/// Generic internal loop: everything not handled by the special
/// cases. Reaches this branch only when both `n1, n2 >= 2` and the
/// shape is not `(2, 2)` / `(2, 3)` / `(3, 2)`.
#[inline]
fn energy_generic_scalar(params: &Params, b: &InternalLoopScalar) -> Energy {
    interior_size_penalty(params, b.n1 + b.n2)
        + ninio_asymmetry(params, b.n1.abs_diff(b.n2))
        + mismatch_sum_scalar(&params.mismatch_interior, b)
}

/// Scalar shape of [`mismatch_sum`]. Outer side feeds
/// `(closing, left[0], right[-1])`; inner-reversed side feeds
/// `(inner_reversed, right[0], left[-1])`. The `right[0]` /
/// `right[-1]` / `left[-1]` reads collapse to boundary nts in the
/// strand-length-1 cases via the caller's input convention.
#[inline]
fn mismatch_sum_scalar(table: &Triplet37, b: &InternalLoopScalar) -> Energy {
    Energy(
        table.get(
            b.outer,
            b.boundary_left5p.to_nuc5(),
            b.boundary_right3p.to_nuc5(),
        ) + table.get(
            b.inner.reversed(),
            b.boundary_right5p.to_nuc5(),
            b.boundary_left3p.to_nuc5(),
        ),
    )
}

/// The `size > MAX_TABULATED_SIZE` (`lxc`) branch is reachable only via
/// [`crate::evaluate`] on user-provided structures: the Zuker DP search
/// in [`crate::dp`] caps interior loops at `MAX_TABULATED_SIZE = 30`
/// (Vienna's `MAXLOOP`), matching `RNAfold` — universal across Vienna,
/// DERNA C++, LinearFold, and LinearDesign.
fn interior_size_penalty(params: &Params, size: usize) -> Energy {
    if size <= MAX_TABULATED_SIZE {
        return Energy(params.interior.values[size]);
    }
    let extrapolation = super::extrapolation(size);
    Energy(params.interior.values[MAX_TABULATED_SIZE] + extrapolation)
}

/// `MIN2(MAX_NINIO, |n1 - n2| * ninio_37)` — Vienna's NINIO term in
/// `eval_internal.c`. `ninio_37` is `P->ninio[2]` at 37°C.
fn ninio_asymmetry(params: &Params, asymmetry: usize) -> Energy {
    let raw = (asymmetry as i32) * params.ninio.ninio_37;
    Energy(raw.min(params.ninio.max_ninio))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nts() -> [Nucleotide; 4] {
        [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U]
    }

    fn canonical_pairs() -> [BasePair; 6] {
        [
            BasePair::AU,
            BasePair::UA,
            BasePair::CG,
            BasePair::GC,
            BasePair::GU,
            BasePair::UG,
        ]
    }

    fn make_strand(
        boundary_5p: Nucleotide,
        boundary_3p: Nucleotide,
        len: usize,
    ) -> Vec<Nucleotide> {
        // The slice form reads only boundary positions for every shape that
        // the codon-DP layer hits; positions 1 .. len-1 (for n >= 3) are
        // filler. Use Nucleotide::A as a deterministic filler.
        if len == 1 {
            return vec![boundary_5p];
        }
        let mut s = vec![Nucleotide::A; len];
        s[0] = boundary_5p;
        s[len - 1] = boundary_3p;
        s
    }

    /// One exhaustive-test combo: closing pairs + four boundary nts.
    #[derive(Debug, Clone, Copy)]
    struct AgreementCase {
        outer: BasePair,
        inner: BasePair,
        xi: Nucleotide,
        hi: Nucleotide,
        kj: Nucleotide,
        yj: Nucleotide,
    }

    fn agreement_cases() -> Vec<AgreementCase> {
        let pairs = canonical_pairs();
        let nuc = nts();
        pairs
            .iter()
            .copied()
            .flat_map(|outer| pairs.iter().copied().map(move |inner| (outer, inner)))
            .flat_map(|(o, i)| nuc.iter().copied().map(move |xi| (o, i, xi)))
            .flat_map(|(o, i, xi)| nuc.iter().copied().map(move |hi| (o, i, xi, hi)))
            .flat_map(|(o, i, xi, hi)| nuc.iter().copied().map(move |kj| (o, i, xi, hi, kj)))
            .flat_map(|(o, i, xi, hi, kj)| {
                nuc.iter().copied().map(move |yj| AgreementCase {
                    outer: o,
                    inner: i,
                    xi,
                    hi,
                    kj,
                    yj,
                })
            })
            .collect()
    }

    fn assert_agreement(n1: usize, n2: usize) {
        let params = Params::load_default().unwrap();
        for case in agreement_cases() {
            check_one_case(&params, case, n1, n2);
        }
    }

    fn check_one_case(params: &Params, case: AgreementCase, n1: usize, n2: usize) {
        // For (1, _) / (_, 1) the strand has length 1, so the boundary-5p slot
        // stands in for both ends of the strand.
        let hi_eff = if n1 == 1 { case.xi } else { case.hi };
        let yj_eff = if n2 == 1 { case.kj } else { case.yj };
        let left = make_strand(case.xi, hi_eff, n1);
        let right = make_strand(case.kj, yj_eff, n2);
        let lp = InternalLoop {
            outer: case.outer,
            inner: case.inner,
            left: &left,
            right: &right,
        };
        let slice_form = internal_energy(params, &lp);
        let scalar_form = internal_energy_scalar(
            params,
            &InternalLoopScalar {
                outer: case.outer,
                inner: case.inner,
                boundary_left5p: case.xi,
                boundary_left3p: hi_eff,
                boundary_right5p: case.kj,
                boundary_right3p: yj_eff,
                n1,
                n2,
            },
        );
        assert_eq!(
            slice_form, scalar_form,
            "disagreement at shape=({n1},{n2}) case={case:?}",
        );
    }

    #[test]
    fn scalar_agrees_on_small_shapes() {
        for (n1, n2) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
            assert_agreement(n1, n2);
        }
    }

    #[test]
    fn scalar_agrees_on_one_by_n_and_n_by_one() {
        for (n1, n2) in [(1, 3), (1, 5), (3, 1), (5, 1)] {
            assert_agreement(n1, n2);
        }
    }

    #[test]
    fn scalar_agrees_on_two_by_three_and_three_by_two() {
        assert_agreement(2, 3);
        assert_agreement(3, 2);
    }

    #[test]
    fn scalar_agrees_on_generic_shapes() {
        for (n1, n2) in [(3, 3), (5, 7), (4, 6), (10, 10)] {
            assert_agreement(n1, n2);
        }
    }
}
