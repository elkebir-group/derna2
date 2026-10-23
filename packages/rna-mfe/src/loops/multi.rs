//! Multi-loop (multibranch) evaluator.
//!
//! Vienna's affine multi-loop scoring at `dangles=0` reduces to a sum
//! of three table-driven terms plus a per-stem `terminal_au_37`
//! correction. The composition is built up by `vrna_eval_loop_pt_v` /
//! `eval_mb_loop` (which sums per-stem contributions from
//! `E_MLstem` — `include/ViennaRNA/eval/multibranch.h::E_MLstem`,
//! v2.7.0 lines 163–184) and adds the `MLclosing + n_unpaired *
//! MLbase` initiation. At `dangles=0`, each `E_MLstem` invocation
//! returns `MLintern[type] + (type > 2 ? TerminalAU : 0)` because both
//! mismatch encodings are negative — the `mismatchM` / `dangle*`
//! branches at `multibranch.h:171–176` are skipped.
//!
//! The closing pair counts as a stem too — Vienna invokes `E_MLstem`
//! for the closing pair exactly as for each branch, with `type` set to
//! the *reversed* (loop-side) closing-pair type. This module mirrors that:
//! [`MultiLoop::closing`] is stored in sequence order, and the AU/GU
//! check uses [`BasePair::applies_terminal_au`] which is orientation-
//! agnostic, so no reversal is needed for the terminal-AU accounting.
//!
//! Final formula:
//!
//! ```text
//! E = ML.closing_37
//!   + ML.intern_37 * (1 + branches.len())          // closing counts as a stem
//!   + ML.base_37   * unpaired
//!   + (closing applies_terminal_au ? Misc.terminal_au_37 : 0)
//!   + Σ_{b in branches} (b applies_terminal_au ? Misc.terminal_au_37 : 0)
//! ```
//!
//! Sections consulted: `ML_params` (`closing_37`, `intern_37`,
//! `base_37`); `Misc.terminal_au_37` once per AU/GU stem (closing or
//! branch).

use super::Energy;
use crate::params::{BasePair, Params};

/// A multi-loop's closing pair, immediately enclosed branch pairs (in
/// 5' → 3' order), and unpaired-nucleotide count. Branch pairs are
/// stored in sequence order — the orientation at `BasePair` level is
/// preserved so the [`BasePair::applies_terminal_au`] check is
/// well-defined for each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiLoop {
    pub closing: BasePair,
    pub branches: Vec<BasePair>,
    pub unpaired: usize,
}

/// Multi-loop energy at `dangles=0`. See module-level docs for the
/// formula and the Vienna source pin.
pub fn multi_loop_energy(params: &Params, m: &MultiLoop) -> Energy {
    let stems = 1 + m.branches.len();
    let initiation = params.ml.closing_37
        + params.ml.intern_37 * (stems as i32)
        + params.ml.base_37 * (m.unpaired as i32);
    let mut total = initiation + per_stem_terminal_au(params, m.closing);
    for branch in &m.branches {
        total += per_stem_terminal_au(params, *branch);
    }
    Energy(total)
}

fn per_stem_terminal_au(params: &Params, pair: BasePair) -> i32 {
    if pair.applies_terminal_au() {
        params.misc.terminal_au_37
    } else {
        0
    }
}
