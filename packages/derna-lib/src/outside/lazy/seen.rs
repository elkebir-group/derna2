//! The subsequences a node has kept, held as indices into its own derivation
//! list.
//!
//! Under node dedup ([`super::Walk::node_dedup`]) each cell remembers what its
//! kept derivations spell, so that a later derivation spelling the same
//! subsequence can be dropped. Every derivation that enters the set is then
//! kept, at the next index of the cell's list, and its fingerprint is stored
//! there already. A set of 16-byte fingerprints therefore holds every key
//! twice. [`IndexSet`] stores a 4-byte index and a 1-byte tag of the key's
//! hash instead, reading the key back through the list only where a tag
//! agrees: open addressing, linear probing, at most three quarters full. Which
//! one a node holds is [`super::Walk::dedup_index`].
//!
//! The index vector's bytes are resident rather than merely counted: the empty
//! marker is not all-zero, so it is written through at allocation and no
//! shared zero page can serve it. The tag vector's free marker *is* zero, so a
//! fresh tag page is reserved rather than committed and becomes resident as
//! the slots around it are written. Narrowing a slot therefore gives back
//! resident memory, four fifths of it immediately.

use crate::hash::FxHashSet;

/// A node's dedup set: the fingerprints themselves, or indices into the
/// node's derivations.
pub(super) enum Seen {
    Keys(FxHashSet<u128>),
    Index(IndexSet),
}

impl Default for Seen {
    fn default() -> Self {
        Seen::Keys(FxHashSet::default())
    }
}

impl Seen {
    /// An empty set of indices (`index`) or of fingerprints.
    pub(super) fn new(index: bool) -> Self {
        if index {
            Seen::Index(IndexSet::default())
        } else {
            Seen::default()
        }
    }

    /// Whether no derivation kept yet spells `key` (`spelled(i)` is what
    /// derivation `i` spells); if so, record it as derivation `next`.
    pub(super) fn insert(&mut self, key: u128, next: u32, spelled: impl Fn(u32) -> u128) -> bool {
        match self {
            Seen::Keys(set) => set.insert(key),
            Seen::Index(set) => set.insert(key, next, spelled),
        }
    }

    /// The set's heap bytes (a hash set's estimated at the swiss table's
    /// 7/8 load and one control byte per slot).
    pub(super) fn bytes(&self) -> usize {
        match self {
            Seen::Keys(set) if set.capacity() > 0 => (set.capacity() * 8 / 7 + 1) * 17,
            Seen::Keys(_) => 0,
            Seen::Index(set) => set.bytes(),
        }
    }
}

/// An index vector entry no index occupies.
const EMPTY: u32 = u32::MAX;

/// A tag no key owns, so the slot is free.
const FREE: u8 = 0;

/// An open-addressing set of derivation indices, keyed by what each index's
/// derivation spells. A slot is a 4-byte index and a 1-byte tag of the key's
/// hash, held in two vectors of one length: a probe walks the tags, which are
/// a byte apiece and so dense in cache, and reads a key back out of the
/// derivation list only where a tag agrees — one spurious read in 256.
/// Growing the table reads every index it moves, a tag being too short to
/// name a home slot.
#[derive(Default)]
pub(super) struct IndexSet {
    tags: Vec<u8>,
    at: Vec<u32>,
    len: usize,
}

/// A key's 32-bit hash; its low bits pick the home slot. Fingerprints are
/// hashes already; the multiply only mixes the two halves together.
fn hash(key: u128) -> u32 {
    let x = (key as u64) ^ ((key >> 64) as u64);
    (x.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32
}

/// The tag a hash gives a slot, never [`FREE`].
fn tag(h: u32) -> u8 {
    ((h >> 24) as u8) | 1
}

impl IndexSet {
    /// Whether no index held yet spells `key` (`spelled(i)` is what index `i`
    /// spells); if so, record `next`, the index the caller keeps it at.
    pub(super) fn insert(&mut self, key: u128, next: u32, spelled: impl Fn(u32) -> u128) -> bool {
        debug_assert_ne!(next, EMPTY, "no node holds u32::MAX derivations");
        if (self.len + 1) * 4 > self.tags.len() * 3 {
            self.grow(&spelled);
        }
        let mask = self.tags.len() - 1;
        let h = hash(key);
        let t = tag(h);
        let mut at = h as usize & mask;
        loop {
            match self.tags[at] {
                FREE => {
                    self.tags[at] = t;
                    self.at[at] = next;
                    self.len += 1;
                    return true;
                }
                g if g == t && spelled(self.at[at]) == key => return false,
                _ => at = (at + 1) & mask,
            }
        }
    }

    /// Double the table (eight slots at first) and place every index again,
    /// by the hash of what it spells.
    fn grow(&mut self, spelled: &impl Fn(u32) -> u128) {
        let size = (self.tags.len() * 2).max(8);
        let tags = std::mem::replace(&mut self.tags, vec![FREE; size]);
        let old = std::mem::replace(&mut self.at, vec![EMPTY; size]);
        let mask = size - 1;
        for (t, i) in tags.into_iter().zip(old).filter(|&(t, _)| t != FREE) {
            let mut at = hash(spelled(i)) as usize & mask;
            while self.tags[at] != FREE {
                at = (at + 1) & mask;
            }
            self.tags[at] = t;
            self.at[at] = i;
        }
    }

    /// The table's heap bytes.
    pub(super) fn bytes(&self) -> usize {
        self.at.capacity() * std::mem::size_of::<u32>() + self.tags.capacity()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Keys drawn with many repeats and some colliding home slots, offered in
    /// order: the table agrees with a standard set on every answer, through
    /// several growths, and each kept index reads back its own key.
    #[test]
    fn agrees_with_a_hash_set() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut kept: Vec<u128> = Vec::new();
        let mut seen = IndexSet::default();
        let mut reference = HashSet::new();
        for _ in 0..50_000 {
            let r = next();
            // About one key in four repeats an earlier one; some share their
            // low halves, so they meet in the same home slot.
            let key = if r % 4 == 0 && !kept.is_empty() {
                kept[(r >> 8) as usize % kept.len()]
            } else if r % 4 == 2 && !kept.is_empty() {
                // The halves swapped: a different key with the same hash.
                kept[(r >> 8) as usize % kept.len()].rotate_left(64)
            } else if r % 4 == 1 {
                (u128::from(r) << 64) | 7
            } else {
                (u128::from(next()) << 64) | u128::from(r)
            };
            let fresh = seen.insert(key, kept.len() as u32, |i| kept[i as usize]);
            assert_eq!(fresh, reference.insert(key), "key {key:x}");
            if fresh {
                kept.push(key);
            }
        }
        assert_eq!(seen.len(), reference.len());
        // Five bytes to a slot, and the table is at least three quarters of
        // the way to full after the growth that made room for the last entry.
        assert!(seen.bytes() >= seen.len() * 5 * 4 / 3);
        assert!(seen.bytes() <= (seen.len() + 1) * 5 * 8 / 3);
        for (i, &key) in kept.iter().enumerate() {
            assert!(
                !seen.insert(key, kept.len() as u32, |j| kept[j as usize]),
                "index {i}"
            );
        }
    }

    /// A slot is five bytes. Six distinct keys fit the first table the set
    /// allocates, so its bytes are the eight slots' and nothing else: a slot
    /// of the bare index reads 32 here, and one carrying a 32-bit hash 64.
    #[test]
    fn six_keys_hold_in_forty_bytes() {
        let kept: Vec<u128> = (0..6).map(|i| (0xABCD_0000_u128 << 64) | i).collect();
        let mut seen = IndexSet::default();
        for (i, &key) in kept.iter().enumerate() {
            assert!(seen.insert(key, i as u32, |j| kept[j as usize]), "key {i}");
        }
        assert_eq!(seen.len(), 6);
        assert_eq!(seen.bytes(), 40, "eight slots of four bytes and a tag");
    }

    /// Which of the two vectors is resident where it is allocated. The index
    /// vector's free marker is not all-zero, so no shared zero page can serve
    /// it and every one of its bytes is written at allocation. The tag
    /// vector's is zero, so its pages are reserved and become resident as the
    /// slots around them are written — four fifths of a slot is given back at
    /// once, and the fifth as the table fills.
    #[test]
    fn the_index_vector_is_written_and_the_tag_vector_is_not() {
        let kept = [0x0F1E_2D3C_u128 << 64];
        let mut seen = IndexSet::default();
        assert!(seen.insert(kept[0], 0, |i| kept[i as usize]));
        let index: Vec<u8> = seen.at.iter().flat_map(|s| s.to_ne_bytes()).collect();
        assert_eq!(index.len(), 32, "eight slots of four bytes");
        // Seven free entries of four non-zero bytes each; the eighth holds
        // index 0, which is four zero bytes.
        assert_eq!(index.iter().filter(|&&b| b != 0).count(), 28);
        // One tag of eight is written; the other seven are still zero.
        assert_eq!(seen.tags.iter().filter(|&&t| t != FREE).count(), 1);
    }

    /// Two distinct keys with the same 32-bit hash, and so the same tag and
    /// the same home slot: the set separates them on what each index spells,
    /// and keeps doing so after the table has been grown and both moved.
    #[test]
    fn keys_sharing_a_hash_stay_distinct_across_a_growth() {
        let a = 0x0123_4567_89AB_CDEF_u128 << 64 | 0xFEDC_BA98_7654_3210;
        let b = a.rotate_left(64);
        assert_ne!(a, b);
        assert_eq!(hash(a), hash(b), "the halves swapped keep the hash");
        let mut kept = vec![a, b];
        let mut seen = IndexSet::default();
        assert!(seen.insert(a, 0, |i| kept[i as usize]));
        assert!(seen.insert(b, 1, |i| kept[i as usize]));
        // Enough further keys to double the table twice, so both have been
        // placed again by the hash of what they spell.
        for i in 0..40u128 {
            let key = (i << 96) | 0x5555;
            let at = kept.len() as u32;
            if seen.insert(key, at, |j| kept[j as usize]) {
                kept.push(key);
            }
        }
        assert!(
            !seen.insert(a, kept.len() as u32, |i| kept[i as usize]),
            "a"
        );
        assert!(
            !seen.insert(b, kept.len() as u32, |i| kept[i as usize]),
            "b"
        );
    }

    /// Two distinct keys that share a tag but not a hash: the tag is one byte
    /// of the hash, so a probe reads a key back where it need not have, and
    /// still answers correctly. Without this the spurious read is untested.
    #[test]
    fn keys_sharing_only_a_tag_stay_distinct() {
        let mut found = None;
        let a = 0x1111_2222_3333_4444_u128;
        for i in 1..100_000u128 {
            let b = a ^ (i << 70);
            if b != a && tag(hash(a)) == tag(hash(b)) && hash(a) != hash(b) {
                found = Some(b);
                break;
            }
        }
        let b = found.expect("a key sharing the tag but not the hash");
        let kept = [a, b];
        let mut seen = IndexSet::default();
        assert!(seen.insert(a, 0, |i| kept[i as usize]));
        assert!(seen.insert(b, 1, |i| kept[i as usize]));
        assert_eq!(seen.len(), 2);
        assert!(!seen.insert(a, 2, |i| kept[i as usize]));
        assert!(!seen.insert(b, 2, |i| kept[i as usize]));
    }
}
