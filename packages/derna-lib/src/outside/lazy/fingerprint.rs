//! Fingerprints of what a derivation spells, composed from its parts.
//!
//! A derivation spells every position of the molecule exactly once, across
//! the hyperedges of its tree, and closes a set of pairs. Each `(position,
//! nucleotide)` and each pair gets a fixed pseudo-random key, and a
//! derivation's fingerprint is the wrapping sum of the keys its tree spells:
//! its hyperedge's own keys plus its tails' fingerprints. That is O(tails) per
//! derivation, where spelling the sequence is O(length), and it depends only
//! on the sequence (and the pairs), not on how the tree splits it.
//!
//! The sequence lane is 64 bits. Equal sequences always fingerprint equal;
//! two different sequences can fingerprint equal too, so a fingerprint only
//! says where two derivations *may* spell the same thing. The walk keeps one
//! fingerprint per derivation, and wherever two agree it spells both
//! derivations and compares them before treating one as a repeat
//! ([`super::Forest::same_spelling`]): a collision costs that spelling and
//! never passes over a design.
//!
//! Builds with debug assertions also carry a lane over the pairs, which only
//! counts duplicate `(sequence, structure)` molecules — a check on the walk,
//! not something it needs. Release builds leave it out.

/// A derivation's fingerprint: a 64-bit lane over its sequence and, with
/// debug assertions, one over its pairs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Fingerprint {
    seq: u64,
    #[cfg(debug_assertions)]
    pairs: u64,
}

/// The splitmix64 finalizer: a fixed bijection that scatters nearby inputs.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

const LANE_SEQ: u64 = 0x9e37_79b9_7f4a_7c15;
#[cfg(debug_assertions)]
const LANE_PAIRS: u64 = 0x8cb9_2ba7_2f3d_8dd7;

impl Fingerprint {
    /// Position `pos` spelling nucleotide `nuc`.
    pub(super) fn spelled(pos: u32, nuc: u8) -> Self {
        let x = (u64::from(pos) << 2) | u64::from(nuc);
        Self {
            seq: mix(x ^ LANE_SEQ),
            #[cfg(debug_assertions)]
            pairs: 0,
        }
    }

    /// The pair `(i, j)`; empty without debug assertions.
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    pub(super) fn pair(i: u32, j: u32) -> Self {
        Self {
            seq: 0,
            #[cfg(debug_assertions)]
            pairs: mix(((u64::from(i) << 32) | u64::from(j)) ^ LANE_PAIRS),
        }
    }

    /// The fingerprint of two disjoint parts together.
    pub(super) fn with(self, other: Self) -> Self {
        Self {
            seq: self.seq.wrapping_add(other.seq),
            #[cfg(debug_assertions)]
            pairs: self.pairs.wrapping_add(other.pairs),
        }
    }

    /// The sequence alone. Equal for equal sequences; equal for different
    /// ones only by collision.
    pub(super) fn design(self) -> u64 {
        self.seq
    }

    /// Sequence and pairs together. It only counts duplicate molecules, so a
    /// collision would miscount one, never pass over a design.
    #[cfg(debug_assertions)]
    pub(super) fn molecule(self) -> u128 {
        (u128::from(self.seq) << 64) | u128::from(self.pairs)
    }

    /// The fingerprint with its sequence lane cut to its low `bits` bits, so
    /// that a test can make distinct sequences collide at will. The low bits
    /// of a wrapping sum are the wrapping sum of the low bits, so cutting a
    /// composed fingerprint equals composing cut parts.
    #[cfg(test)]
    pub(super) fn truncated(self, bits: u32) -> Self {
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let mut cut = self;
        cut.seq &= mask;
        cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fingerprint_does_not_depend_on_how_the_parts_are_split() {
        let parts: Vec<Fingerprint> = (0..6)
            .map(|p| Fingerprint::spelled(p, (p % 4) as u8))
            .collect();
        let left = parts[..3]
            .iter()
            .fold(Fingerprint::default(), |a, &b| a.with(b));
        let right = parts[3..]
            .iter()
            .fold(Fingerprint::default(), |a, &b| a.with(b));
        let all = parts
            .iter()
            .rev()
            .fold(Fingerprint::default(), |a, &b| a.with(b));
        assert_eq!(left.with(right), all);
        let other = Fingerprint::spelled(0, 1).with(all);
        assert_ne!(other.design(), all.design());
        let paired = all.with(Fingerprint::pair(0, 5));
        assert_eq!(paired.design(), all.design());
        #[cfg(debug_assertions)]
        assert_ne!(paired.molecule(), all.molecule());
    }

    /// Cutting the lane commutes with composing, so a walk that cuts every
    /// fingerprint it composes holds the cut of the true one everywhere.
    #[test]
    fn a_truncated_fingerprint_composes_like_the_full_one() {
        let parts: Vec<Fingerprint> = (0..9)
            .map(|p| Fingerprint::spelled(p * 7, (p % 4) as u8))
            .collect();
        for bits in [0, 1, 3, 17, 63, 64] {
            let cut_then_sum = parts
                .iter()
                .fold(Fingerprint::default(), |a, &b| a.with(b.truncated(bits)))
                .truncated(bits);
            let sum_then_cut = parts
                .iter()
                .fold(Fingerprint::default(), |a, &b| a.with(b))
                .truncated(bits);
            assert_eq!(cut_then_sum, sum_then_cut, "{bits} bits");
        }
    }
}
