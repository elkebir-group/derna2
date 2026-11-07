//! `rna-mfe` — Turner 2004 RNA minimum-free-energy evaluator and
//! MFE folder at `dangles=0`.
//!
//! No external runtime dependency: the Turner 2004 parameter file is
//! parsed and the loop energies evaluated natively. The test suite checks
//! both entry points against reference energies and structures.
//!
//! ## Public surface
//!
//! - [`fold`] — MFE folder, returns the MFE `(structure, energy)`
//!   for an RNA sequence at `dangles=0`. The interior-loop search caps
//!   loops at 30 unpaired bases (`MAXLOOP`).
//! - [`evaluate`] — whole-structure evaluator; ergonomic entry point that
//!   accepts ASCII bytes. Allocates per call; prefer [`evaluate_encoded`]
//!   for hot paths that evaluate repeatedly.
//! - [`evaluate_encoded`] — pre-encoded, allocation-free-on-hot-path
//!   evaluator. The caller owns an [`EvalScratch`] that is reused across
//!   calls, paying encoding cost once per sequence. Intended for
//!   use cases like codon-optimisation DP that evaluate many structures
//!   over the same underlying sequence.
//! - [`encode_seq`] — convert ASCII bytes to `Vec<Nucleotide>` for use
//!   with [`evaluate_encoded`].
//! - [`EvalScratch`] — caller-owned scratch for [`evaluate_encoded`].
//! - [`Params`] / [`ParseError`] — Turner 2004 parameter loading.
//! - [`Energy`] and loop types via [`loops`] re-exports.
//! - [`decompose()`] / [`Decomposition`] / [`Loop`] — structure decomposer.
//! - [`EvalError`] / [`FoldError`] — error types for both entry points.

pub mod decompose;
pub mod dp;
pub mod loops;
pub mod params;

pub use decompose::{decompose, DecomposeError, Decomposition, Loop};
pub use loops::{
    bulge_energy, exterior_loop_energy, hairpin_energy, hairpin_energy_scalar, internal_energy,
    internal_energy_scalar, multi_loop_energy, stack_energy, BulgeLoop, Energy, HairpinLoop,
    InternalLoop, InternalLoopScalar, MultiLoop, Nucleotide, StackPair,
};
pub use params::{Params, ParseError};

use loops::pair_from_nucleotides;
use thiserror::Error;

// Per-pair state pushed onto the frame stack during the structure walk.
// u32 fields keep the struct at 24 bytes (vs 48 for usize), fitting 2× as
// many frames per cache line and improving stack-walk throughput.
struct LoopFrame {
    i: u32,          // sequence position of the opening '('
    n_children: u32, // immediately enclosed pairs seen so far
    child_i: u32,    // last child's opening position (degree-2 case)
    child_j: u32,    // last child's closing position (degree-2 case)
    covered: u32,    // sum of (child_j - child_i + 1) for all children
    branch_au: i32,  // terminal-AU sum for child branches (multi-loop)
}

/// Caller-owned scratch reused across [`evaluate_encoded`] calls. Construct
/// once per "session" (e.g. per codon-DP search over a fixed sequence), then
/// pass to each [`evaluate_encoded`] call. The evaluator clears and refills
/// the stack on every call, so no manual reset is needed between calls — but
/// the underlying heap allocation is kept, eliminating per-call allocator
/// traffic after the first call.
pub struct EvalScratch {
    frames: Vec<LoopFrame>,
}

impl EvalScratch {
    /// Allocate scratch with a capacity hint for sequences of length `n`.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            frames: Vec::with_capacity(n / 4 + 1),
        }
    }

    /// Reset the scratch to empty, keeping the underlying allocation.
    /// [`evaluate_encoded`] clears internally, so calling this between
    /// evaluations is not required.
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

/// Convert ASCII bytes (`b'A' | b'C' | b'G' | b'U'`) to a
/// `Vec<Nucleotide>` for use with [`evaluate_encoded`]. Returns
/// `Err(EvalError::InvalidNucleotide)` on the first unrecognised byte.
pub fn encode_seq(seq: &[u8]) -> Result<Vec<Nucleotide>, EvalError> {
    seq.iter()
        .map(|&b| Nucleotide::from_byte(b).ok_or(EvalError::InvalidNucleotide(b)))
        .collect()
}

/// Pre-encoded, allocation-free-on-hot-path evaluator. The caller provides:
/// - `seq` — a pre-encoded nucleotide sequence (use [`encode_seq`] to
///   produce it from ASCII bytes; pay this cost once per sequence).
/// - `scratch` — caller-owned frame stack reused across calls (construct
///   once via [`EvalScratch::with_capacity`]).
///
/// Implements a single-pass left-to-right structure walk with a frame
/// stack, scoring each loop as its closing `)` is consumed. No auxiliary
/// Vecs are allocated in the hot path: no pair table, no exterior-branch
/// list, no per-loop children buffer. Callers that evaluate many structures
/// over the same encoded sequence (e.g. a codon-design DP) amortise
/// encoding and frame-stack allocation across the entire search.
///
/// For one-off use, prefer [`evaluate`] which is simpler to call.
pub fn evaluate_encoded(
    params: &Params,
    seq: &[Nucleotide],
    structure: &str,
    scratch: &mut EvalScratch,
) -> Result<Energy, EvalError> {
    if seq.len() != structure.len() {
        return Err(EvalError::Decompose(DecomposeError::LengthMismatch {
            seq_len: seq.len(),
            structure_len: structure.len(),
        }));
    }
    scratch.frames.clear();
    let mut total = Energy::ZERO;
    let mut ext_energy = 0i32;
    for (pos, c) in structure.bytes().enumerate() {
        match c {
            b'(' => scratch.frames.push(LoopFrame {
                i: pos as u32,
                n_children: 0,
                child_i: 0,
                child_j: 0,
                covered: 0,
                branch_au: 0,
            }),
            b')' => {
                let (e, ext_delta) = process_close(params, seq, &mut scratch.frames, pos)?;
                total += e;
                ext_energy += ext_delta;
            }
            b'.' => (),
            other => {
                return Err(EvalError::Decompose(DecomposeError::InvalidStructureChar(
                    other as char,
                )))
            }
        }
    }
    if let Some(frame) = scratch.frames.pop() {
        return Err(EvalError::Decompose(DecomposeError::UnbalancedBrackets {
            bracket: '(',
            position: frame.i as usize,
        }));
    }
    total += Energy(ext_energy);
    Ok(total)
}

// Process a closing ')' at `pos`: pop the frame, score the loop, update the
// parent frame or exterior-energy accumulator. Returns `(loop_energy,
// ext_energy_delta)` so the caller can accumulate both.
#[inline]
fn process_close(
    params: &Params,
    seq: &[Nucleotide],
    frames: &mut Vec<LoopFrame>,
    pos: usize,
) -> Result<(Energy, i32), EvalError> {
    let frame = frames
        .pop()
        .ok_or(EvalError::Decompose(DecomposeError::UnbalancedBrackets {
            bracket: ')',
            position: pos,
        }))?;
    let j = pos;
    let i = frame.i as usize;
    let outer = pair_from_nucleotides(seq[i], seq[j]);
    let outer_au = if outer.applies_terminal_au() {
        params.misc.terminal_au_37
    } else {
        0
    };
    let energy = match frame.n_children {
        0 => hairpin_energy(
            params,
            &HairpinLoop {
                closing: outer,
                loop_seq: &seq[i + 1..j],
            },
        ),
        1 => score_degree_two(
            params,
            seq,
            (i, j),
            (frame.child_i as usize, frame.child_j as usize),
        ),
        _ => {
            let stems = 1 + frame.n_children as i32;
            let unpaired = (j - i - 1 - frame.covered as usize) as i32;
            Energy(
                params.ml.closing_37
                    + params.ml.intern_37 * stems
                    + params.ml.base_37 * unpaired
                    + outer_au
                    + frame.branch_au,
            )
        }
    };
    let ext_delta = if let Some(parent) = frames.last_mut() {
        parent.n_children += 1;
        parent.child_i = i as u32;
        parent.child_j = j as u32;
        parent.covered += (j - i + 1) as u32;
        parent.branch_au += outer_au;
        0
    } else {
        outer_au
    };
    Ok((energy, ext_delta))
}

/// Whole-structure evaluator. Ergonomic entry point that accepts ASCII
/// bytes and handles encoding internally. Allocates per call (encoding +
/// frame-stack scratch); for hot paths that call evaluate repeatedly over
/// the same sequence, use [`evaluate_encoded`] instead.
pub fn evaluate(params: &Params, seq: &[u8], structure: &str) -> Result<Energy, EvalError> {
    let nt_seq = encode_seq(seq)?;
    let mut scratch = EvalScratch::with_capacity(nt_seq.len());
    evaluate_encoded(params, &nt_seq, structure, &mut scratch)
}

/// MFE folder. Returns the dot-bracket MFE structure of `seq`
/// and its energy at `dangles=0`. The interior-loop search is bounded
/// at `MAXLOOP = 30` unpaired bases; structures with larger
/// interior loops are never produced by `fold` (use [`evaluate`] to
/// score user-provided structures with such loops).
pub fn fold(params: &Params, seq: &[u8]) -> Result<(String, Energy), FoldError> {
    let nt_seq = parse_seq_for_fold(seq)?;
    let tables = dp::fill_dp_tables(params, &nt_seq);
    let energy = tables.f[nt_seq.len()].unwrap_or(Energy::ZERO);
    let structure = dp::traceback(params, &nt_seq, &tables);
    Ok((structure, energy))
}

/// [`fold`], also returning the filled tables, the base for [`fold_from`].
pub fn fold_tables(
    params: &Params,
    seq: &[u8],
) -> Result<(String, Energy, dp::DpTables), FoldError> {
    let nt_seq = parse_seq_for_fold(seq)?;
    let tables = dp::fill_dp_tables(params, &nt_seq);
    let energy = tables.f[nt_seq.len()].unwrap_or(Energy::ZERO);
    let structure = dp::traceback(params, &nt_seq, &tables);
    Ok((structure, energy, tables))
}

/// [`fold`] of `seq`, which differs from the sequence `base` was filled for
/// only at positions `lo..=hi`: only the cells that span the change are
/// recomputed ([`dp::fill_dp_tables_from`]). Same result as [`fold`].
pub fn fold_from(
    params: &Params,
    base: &dp::DpTables,
    seq: &[u8],
    lo: usize,
    hi: usize,
) -> Result<(String, Energy), FoldError> {
    let nt_seq = parse_seq_for_fold(seq)?;
    let tables = dp::fill_dp_tables_from(params, &nt_seq, base, lo, hi);
    let energy = tables.f[nt_seq.len()].unwrap_or(Energy::ZERO);
    let structure = dp::traceback(params, &nt_seq, &tables);
    Ok((structure, energy))
}

fn parse_seq_for_fold(seq: &[u8]) -> Result<Vec<Nucleotide>, FoldError> {
    seq.iter()
        .map(|&b| Nucleotide::from_byte(b).ok_or(FoldError::InvalidNucleotide(b)))
        .collect()
}

#[inline]
fn score_degree_two(
    params: &Params,
    seq: &[Nucleotide],
    (i, j): (usize, usize),
    (k, l): (usize, usize),
) -> Energy {
    let outer = pair_from_nucleotides(seq[i], seq[j]);
    let inner = pair_from_nucleotides(seq[k], seq[l]);
    let n1 = k - i - 1;
    let n2 = j - l - 1;
    if n1 == 0 && n2 == 0 {
        stack_energy(params, &StackPair { outer, inner })
    } else if n1 == 0 || n2 == 0 {
        bulge_energy(
            params,
            &BulgeLoop {
                outer,
                inner,
                size: n1.max(n2),
            },
        )
    } else {
        internal_energy(
            params,
            &InternalLoop {
                outer,
                inner,
                left: &seq[i + 1..k],
                right: &seq[l + 1..j],
            },
        )
    }
}

/// Errors surfaced by [`evaluate`] and [`evaluate_encoded`]. Wraps the
/// decomposition errors and adds an invalid-byte variant for the ASCII →
/// [`Nucleotide`] conversion.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EvalError {
    #[error(transparent)]
    Decompose(#[from] DecomposeError),
    #[error("invalid nucleotide byte: {0:?}")]
    InvalidNucleotide(u8),
}

/// Errors surfaced by [`fold`]. Currently only the invalid-byte case;
/// any internal DP invariant failure is a debug-assert (programmer
/// error rather than user-input error).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FoldError {
    #[error("invalid nucleotide byte: {0:?}")]
    InvalidNucleotide(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolchain_smoke() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn loads_default_params() {
        Params::load_default().expect("the built-in parameter table parses");
    }

    #[test]
    fn evaluate_and_evaluate_encoded_agree() {
        let params = Params::load_default().expect("load Turner 2004");
        let seq = b"GGGGAAAACCCC";
        let structure = "((((....))))";
        let e_shell = evaluate(&params, seq, structure).unwrap();
        let nt_seq = encode_seq(seq).unwrap();
        let mut scratch = EvalScratch::with_capacity(nt_seq.len());
        let e_encoded = evaluate_encoded(&params, &nt_seq, structure, &mut scratch).unwrap();
        assert_eq!(e_shell, e_encoded);
    }

    #[test]
    fn evaluate_encoded_reuse_gives_same_result() {
        let params = Params::load_default().expect("load Turner 2004");
        let seq = b"GGGGAAAACCCC";
        let structure = "((((....))))";
        let nt_seq = encode_seq(seq).unwrap();
        let mut scratch = EvalScratch::with_capacity(nt_seq.len());
        let e1 = evaluate_encoded(&params, &nt_seq, structure, &mut scratch).unwrap();
        let e2 = evaluate_encoded(&params, &nt_seq, structure, &mut scratch).unwrap();
        assert_eq!(e1, e2);
    }
}
