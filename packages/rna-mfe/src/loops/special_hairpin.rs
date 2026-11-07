//! Special-hairpin bonus lookup (`Triloops` / `Tetraloops` / `Hexaloops`).
//!
//! The hairpin evaluator probes the size-3 / size-4 / size-6
//! special-hairpin tables and **returns the bonus value directly** on a
//! hit.
//!
//! That short-circuit means the bonus value *replaces* the entire generic
//! hairpin formula — `hairpin[size]`, the AU/GU `terminal_au_37` penalty,
//! and the `mismatch_hairpin` lookup are all skipped when the table hits.
//! At default salt the entries on disk are the total free energies of the
//! special hairpin, not deltas to add on top.
//!
//! The lookup key is `closing_5'-side + loop_seq +
//! closing_3'-side` — i.e., the 5-/6-/8-character window of the sequence
//! starting at the 5' base of the closing pair and ending at the 3' base.
//! That matches the format of the entries in
//! [`crate::params::SpecialHairpinTable`].

use super::Nucleotide;
use crate::params::{BasePair, Params};

/// Look up the bonus for a hairpin loop of size 3 / 4 / 6 in the
/// matching special-hairpin table. Returns `None` for unsupported sizes
/// or for loop sequences absent from the table — callers fall back to
/// the generic-hairpin formula on a miss.
pub(super) fn lookup_bonus(
    params: &Params,
    closing: BasePair,
    loop_seq: &[Nucleotide],
) -> Option<i32> {
    let table = match loop_seq.len() {
        3 => &params.triloops,
        4 => &params.tetraloops,
        6 => &params.hexaloops,
        _ => return None,
    };
    // Build key on the stack — max length is 1 + 6 + 1 = 8 bytes (hexaloop).
    let (key, len) = build_key_bytes(closing, loop_seq);
    table.lookup_37(&key[..len])
}

fn build_key_bytes(closing: BasePair, loop_seq: &[Nucleotide]) -> ([u8; 8], usize) {
    let (c5, c3) = closing_ascii(closing);
    let mut buf = [0u8; 8];
    buf[0] = c5;
    for (i, &n) in loop_seq.iter().enumerate() {
        buf[i + 1] = nucleotide_ascii(n);
    }
    let n = loop_seq.len();
    buf[n + 1] = c3;
    (buf, n + 2)
}

const fn nucleotide_ascii(n: Nucleotide) -> u8 {
    match n {
        Nucleotide::A => b'A',
        Nucleotide::C => b'C',
        Nucleotide::G => b'G',
        Nucleotide::U => b'U',
    }
}

/// 5'-side and 3'-side ASCII letters of a closing pair. `NN` is not a
/// real corpus pair and never reaches a special-hairpin lookup; the
/// function returns `(b'N', b'N')` for that case so the lookup just
/// misses cleanly rather than panicking.
const fn closing_ascii(pair: BasePair) -> (u8, u8) {
    match pair {
        BasePair::CG => (b'C', b'G'),
        BasePair::GC => (b'G', b'C'),
        BasePair::GU => (b'G', b'U'),
        BasePair::UG => (b'U', b'G'),
        BasePair::AU => (b'A', b'U'),
        BasePair::UA => (b'U', b'A'),
        BasePair::NN => (b'N', b'N'),
    }
}
