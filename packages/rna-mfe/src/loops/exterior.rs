//! Exterior-loop evaluator.
//!
//! The exterior loop is the parent of the outermost base pairs (no
//! enclosing pair). Vienna's `vrna_E_exterior_stem`
//! (`include/ViennaRNA/eval/exterior.h`, with the `dangles=0` reduction
//! mirroring `E_MLstem` from `multibranch.h:163–184`) scores each
//! outermost stem at `dangles=0` as `(type > 2 ? TerminalAU : 0)` — no
//! `mismatchE`, no `dangle*`. There is no init term and no per-branch
//! `MLintern` term: the exterior loop is *not* a multi-loop.
//!
//! Final formula:
//!
//! ```text
//! E_exterior = Σ_{b in branches} (b applies_terminal_au ? Misc.terminal_au_37 : 0)
//! ```
//!
//! Sections consulted: `Misc.terminal_au_37` only.

use super::Energy;
use crate::params::{BasePair, Params};

/// Exterior-loop contribution at `dangles=0`. `branches` is the
/// (sequence-order) list of outermost base pairs.
pub fn exterior_loop_energy(params: &Params, branches: &[BasePair]) -> Energy {
    let mut total = 0;
    for branch in branches {
        if branch.applies_terminal_au() {
            total += params.misc.terminal_au_37;
        }
    }
    Energy(total)
}
