//! Whole-structure decomposition: parse a `(sequence, dot-bracket)`
//! pair into the per-loop structures the evaluators in [`crate::loops`]
//! consume.
//!
//! The decomposer is a pure function with no [`Params`](crate::Params) dependency —
//! it shapes input for the per-loop evaluators rather than scoring
//! directly. [`crate::evaluate`] is the thin shell that pairs
//! [`decompose`] with the per-loop scoring functions.
//!
//! Each base pair `(i, j)` with `i < j` closes exactly one loop, and
//! the loop's kind is determined by the immediately enclosed children
//! (Vienna-style decomposition):
//!
//! - 0 children → hairpin
//! - 1 child `(k, l)` with `k - i - 1 == 0 && j - l - 1 == 0` → stack
//! - 1 child with one of the unpaired sides equal to 0 → bulge
//! - 1 child with both unpaired sides ≥ 1 → internal loop
//! - ≥ 2 children → multi-loop
//!
//! The exterior loop is the parent of the outermost pairs (no
//! enclosing pair); its energy contribution is computed separately by
//! [`crate::loops::exterior_loop_energy`].

use crate::loops::{
    pair_from_nucleotides, BulgeLoop, HairpinLoop, InternalLoop, MultiLoop, Nucleotide, StackPair,
};
use crate::params::BasePair;
use thiserror::Error;

/// One scored region of a folded structure.
#[derive(Debug, Clone)]
pub enum Loop<'a> {
    Hairpin(HairpinLoop<'a>),
    Stack(StackPair),
    Bulge(BulgeLoop),
    Internal(InternalLoop<'a>),
    Multi(MultiLoop),
}

/// Result of decomposing a `(seq, structure)` pair.
#[derive(Debug, Clone)]
pub struct Decomposition<'a> {
    pub loops: Vec<Loop<'a>>,
    pub exterior_branches: Vec<BasePair>,
}

/// Decomposition errors. Surface invalid inputs cleanly so the caller
/// can build a typed wrapper at the API boundary (see
/// [`crate::EvalError`]).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DecomposeError {
    #[error("sequence length {seq_len} does not match structure length {structure_len}")]
    LengthMismatch {
        seq_len: usize,
        structure_len: usize,
    },
    #[error("unbalanced dot-bracket: extra '{bracket}' at position {position}")]
    UnbalancedBrackets { bracket: char, position: usize },
    #[error("invalid dot-bracket character {0:?}")]
    InvalidStructureChar(char),
}

/// Decompose `(seq, structure)` into a list of typed loops plus the
/// exterior branches. Returns `Err` on malformed input.
pub fn decompose<'a>(
    seq: &'a [Nucleotide],
    structure: &str,
) -> Result<Decomposition<'a>, DecomposeError> {
    if seq.len() != structure.len() {
        return Err(DecomposeError::LengthMismatch {
            seq_len: seq.len(),
            structure_len: structure.len(),
        });
    }
    let pairs = pair_table(structure)?;
    let exterior_branches = collect_exterior_branches(seq, &pairs);
    let mut loops = Vec::new();
    for i in 0..pairs.len() {
        let Some(j) = pairs[i] else { continue };
        if j <= i {
            continue;
        }
        loops.push(classify_loop(seq, &pairs, i, j));
    }
    Ok(Decomposition {
        loops,
        exterior_branches,
    })
}

/// Build a position → partner-position table from a dot-bracket
/// string, validating that brackets balance.
fn pair_table(structure: &str) -> Result<Vec<Option<usize>>, DecomposeError> {
    let mut stack = Vec::new();
    let mut pairs = vec![None; structure.len()];
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => {
                let opener = stack.pop().ok_or(DecomposeError::UnbalancedBrackets {
                    bracket: ')',
                    position: i,
                })?;
                pairs[opener] = Some(i);
                pairs[i] = Some(opener);
            }
            b'.' => (),
            other => return Err(DecomposeError::InvalidStructureChar(other as char)),
        }
    }
    if let Some(opener) = stack.pop() {
        return Err(DecomposeError::UnbalancedBrackets {
            bracket: '(',
            position: opener,
        });
    }
    Ok(pairs)
}

fn collect_exterior_branches(seq: &[Nucleotide], pairs: &[Option<usize>]) -> Vec<BasePair> {
    let mut branches = Vec::new();
    let mut cursor = 0;
    while cursor < pairs.len() {
        match pairs[cursor] {
            Some(j) if j > cursor => {
                branches.push(pair_from_nucleotides(seq[cursor], seq[j]));
                cursor = j + 1;
            }
            _ => cursor += 1,
        }
    }
    branches
}

/// Immediately enclosed pairs `(k, l)` with `i < k < l < j` (Vienna's
/// "children of pair (i, j)").
fn enclosed_children(pairs: &[Option<usize>], i: usize, j: usize) -> Vec<(usize, usize)> {
    let mut children = Vec::new();
    let mut cursor = i + 1;
    while cursor < j {
        match pairs[cursor] {
            Some(end) if end > cursor && end < j => {
                children.push((cursor, end));
                cursor = end + 1;
            }
            _ => cursor += 1,
        }
    }
    children
}

fn classify_loop<'a>(
    seq: &'a [Nucleotide],
    pairs: &[Option<usize>],
    i: usize,
    j: usize,
) -> Loop<'a> {
    let outer = pair_from_nucleotides(seq[i], seq[j]);
    let children = enclosed_children(pairs, i, j);
    match children.as_slice() {
        [] => Loop::Hairpin(HairpinLoop {
            closing: outer,
            loop_seq: &seq[i + 1..j],
        }),
        &[(k, l)] => degree_two_loop(seq, outer, (i, j), (k, l)),
        _ => Loop::Multi(build_multi_loop(seq, outer, i, j, &children)),
    }
}

fn degree_two_loop<'a>(
    seq: &'a [Nucleotide],
    outer: BasePair,
    outer_pair: (usize, usize),
    inner_pair: (usize, usize),
) -> Loop<'a> {
    let (i, j) = outer_pair;
    let (k, l) = inner_pair;
    let inner = pair_from_nucleotides(seq[k], seq[l]);
    let n1 = k - i - 1;
    let n2 = j - l - 1;
    if n1 == 0 && n2 == 0 {
        return Loop::Stack(StackPair { outer, inner });
    }
    if n1 == 0 || n2 == 0 {
        return Loop::Bulge(BulgeLoop {
            outer,
            inner,
            size: n1.max(n2),
        });
    }
    Loop::Internal(InternalLoop {
        outer,
        inner,
        left: &seq[i + 1..k],
        right: &seq[l + 1..j],
    })
}

fn build_multi_loop(
    seq: &[Nucleotide],
    closing: BasePair,
    i: usize,
    j: usize,
    children: &[(usize, usize)],
) -> MultiLoop {
    let mut branches = Vec::with_capacity(children.len());
    let mut covered = 0;
    for &(k, l) in children {
        branches.push(pair_from_nucleotides(seq[k], seq[l]));
        covered += l - k + 1;
    }
    let unpaired = (j - i - 1) - covered;
    MultiLoop {
        closing,
        branches,
        unpaired,
    }
}
