//! Hairpin loop evaluator.
//!
//! At `dangles=0` and assuming Vienna's default `special_hp = 1`, the
//! decomposition (per `include/ViennaRNA/eval/hairpin.h::E_Hairpin`,
//! v2.7.0 lines 344–410):
//!
//! - **size < 3**: not a real loop — returns the size-table sentinel
//!   (`INF`) lifted to [`Energy`]. The DP layers above filter these out.
//! - **special-hairpin hit (size 3 / 4 / 6)**: a match in
//!   [`crate::params::Params::triloops`] / `tetraloops` / `hexaloops`
//!   short-circuits the rest of the formula. Vienna's source `return`s
//!   the bonus value directly so the size-table penalty, the AU/GU
//!   `terminal_au_37`, and the `mismatch_hairpin` lookup are *all*
//!   skipped on a hit. See [`super::special_hairpin`].
//! - **size == 3 (no triloop hit)**: `hairpin[3]` plus a
//!   `Misc.terminal_au_37` penalty when the closing pair is
//!   `AU/UA/GU/UG`. *No* `mismatch_hairpin` contribution — Vienna's
//!   source `return`s *before* reaching the mismatch line.
//! - **size >= 4 (no tetra/hexaloop hit)**: `hairpin[size]` plus
//!   `mismatch_hairpin[closing, loop[0], loop[-1]]`. The AU/GU end
//!   penalty for size ≥ 4 is folded into the mismatch table by Vienna's
//!   parameter file design — there is no second additive term.
//! - **size > 30**: extrapolation `hairpin[30] + lxc * ln(size / 30)`
//!   per [`super::LXC_37`]. No corpus fixture exercises this path.
//!
//! Sections consulted: `hairpin` (size table), `mismatch_hairpin`
//! (size ≥ 4, on a special-hairpin miss only), `Misc.terminal_au_37`
//! (size 3, on a triloop miss only), `Triloops` / `Tetraloops` /
//! `Hexaloops` (size 3 / 4 / 6 short-circuit).

use super::{special_hairpin, Energy, Nucleotide, MAX_TABULATED_SIZE};
use crate::params::{BasePair, Params};

/// A hairpin loop's free-energy inputs at dangles=0.
///
/// `loop_seq` carries the unpaired nucleotides between the closing pair
/// in 5' → 3' order, so `loop_seq[0]` sits at sequence position
/// `i + 1` and `loop_seq[len - 1]` sits at `j - 1` for closing pair
/// `(i, j)`. The size of the loop is `loop_seq.len()`.
#[derive(Debug, Clone, Copy)]
pub struct HairpinLoop<'a> {
    pub closing: BasePair,
    pub loop_seq: &'a [Nucleotide],
}

/// Hairpin loop energy. See module-level docs for the case split.
pub fn hairpin_energy(params: &Params, h: &HairpinLoop<'_>) -> Energy {
    let size = h.loop_seq.len();

    if size < 3 {
        return size_penalty(params, size);
    }

    if let Some(bonus) = special_hairpin::lookup_bonus(params, h.closing, h.loop_seq) {
        return Energy(bonus);
    }

    let base = size_penalty(params, size);
    if size == 3 {
        return base + closing_terminal_au(params, h.closing);
    }
    base + mismatch_term(params, h.closing, h.loop_seq)
}

/// Scalar-args overload of [`hairpin_energy`] for the *generic* (non
/// special-hairpin) formula. Reads only the loop size and its two
/// terminal-mismatch nucleotides — `first` at `i + 1` and `last` at
/// `j - 1` — instead of slicing a `[Nucleotide]` substrate.
///
/// The caller must have already ruled out a special-hairpin hit (the
/// size 3 / 4 / 6 `Triloop` / `Tetraloop` / `Hexaloop` tables), which
/// requires the full loop sequence and cannot be decided from the
/// corners alone. For sizes 5 and ≥ 7 no special table exists, so the
/// corners are sufficient; for sizes 3 / 4 / 6 the caller is responsible
/// for the table probe and only falls through to this helper on a miss.
pub fn hairpin_energy_scalar(
    params: &Params,
    closing: BasePair,
    size: usize,
    first: Nucleotide,
    last: Nucleotide,
) -> Energy {
    let base = size_penalty(params, size);
    if size == 3 {
        return base + closing_terminal_au(params, closing);
    }
    Energy(
        base.0
            + params
                .mismatch_hairpin
                .get(closing, first.to_nuc5(), last.to_nuc5()),
    )
}

/// Size-table contribution `hairpin[size]` with logarithmic
/// extrapolation for `size > 30` mirroring Vienna's
/// `(int)(P->lxc * log(size / 30.))` cast.
fn size_penalty(params: &Params, size: usize) -> Energy {
    if size <= MAX_TABULATED_SIZE {
        return Energy(params.hairpin.values[size]);
    }
    let extrapolation = super::extrapolation(size);
    Energy(params.hairpin.values[MAX_TABULATED_SIZE] + extrapolation)
}

/// `Misc.terminal_au_37` penalty applied to size-3 hairpins closed by
/// AU / UA / GU / UG. Vienna writes this as `(type > 2 ? TerminalAU : 0)`
/// in `eval_hairpin.c:119`; [`BasePair::applies_terminal_au`] is the
/// equivalent on the 0-based encoding.
fn closing_terminal_au(params: &Params, closing: BasePair) -> Energy {
    if closing.applies_terminal_au() {
        Energy(params.misc.terminal_au_37)
    } else {
        Energy::ZERO
    }
}

/// `mismatch_hairpin[closing, loop[0], loop[-1]]` for size ≥ 4 hairpins.
fn mismatch_term(params: &Params, closing: BasePair, loop_seq: &[Nucleotide]) -> Energy {
    let n5 = loop_seq[0].to_nuc5();
    let n3 = loop_seq[loop_seq.len() - 1].to_nuc5();
    Energy(params.mismatch_hairpin.get(closing, n5, n3))
}
