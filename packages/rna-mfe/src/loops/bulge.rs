//! Bulge loop evaluator.
//!
//! A bulge has unpaired nucleotides on exactly one side of the loop.
//! Vienna source: `eval_internal.c::vrna_E_internal`, the `case 0:`
//! branch (`ns == 0`):
//!
//! - **size == 1**: `bulge[1] + stack[outer, inner_reversed]`. The
//!   single-nucleotide bulge is "transparently stacked" — Vienna adds
//!   the stack term *instead of* the per-branch terminal-AU penalties.
//! - **size >= 2**: `bulge[size] + per_branch_terminal_au` where the
//!   per-branch correction adds `Misc.terminal_au_37` once per AU/UA/GU/UG
//!   end, applied independently to the outer and inner closing pairs.
//! - **size > 30**: logarithmic extrapolation as for the other size
//!   tables; no corpus fixture exercises this.
//!
//! Sections consulted: `bulge` (size table), `stack` (size 1 only),
//! `Misc.terminal_au_37` (size ≥ 2 only).

use super::{Energy, MAX_TABULATED_SIZE};
use crate::params::{BasePair, Params};

/// A bulge loop is two pairs separated by `size` unpaired nucleotides
/// on exactly one side; both pairs are sequence-order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BulgeLoop {
    pub outer: BasePair,
    pub inner: BasePair,
    pub size: usize,
}

/// Bulge energy. See module-level docs for the case split.
pub fn bulge_energy(params: &Params, b: &BulgeLoop) -> Energy {
    let base = size_penalty(params, b.size);
    if b.size == 1 {
        return base + Energy(params.stack.get(b.outer, b.inner.reversed()));
    }
    base + per_branch_terminal_au(params, b.outer) + per_branch_terminal_au(params, b.inner)
}

fn size_penalty(params: &Params, size: usize) -> Energy {
    if size <= MAX_TABULATED_SIZE {
        return Energy(params.bulge.values[size]);
    }
    let extrapolation = super::extrapolation(size);
    Energy(params.bulge.values[MAX_TABULATED_SIZE] + extrapolation)
}

fn per_branch_terminal_au(params: &Params, pair: BasePair) -> Energy {
    if pair.applies_terminal_au() {
        Energy(params.misc.terminal_au_37)
    } else {
        Energy::ZERO
    }
}
