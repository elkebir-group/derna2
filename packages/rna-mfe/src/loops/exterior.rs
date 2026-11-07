//! Exterior-loop evaluator.
//!
//! The exterior loop is the parent of the outermost base pairs (no
//! enclosing pair). At `dangles=0` each outermost stem scores the
//! terminal-AU penalty if its pair is not `CG` / `GC`, and nothing
//! otherwise — no `mismatchE`, no `dangle*`. There is no init term and no per-branch
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
