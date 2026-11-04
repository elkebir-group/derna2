//! The subsequences a node has kept, held as indices into its own derivation
//! list.
//!
//! Under node dedup ([`super::Walk::node_dedup`]) each cell remembers what its
//! kept derivations spell, so that a later derivation spelling the same
//! subsequence can be dropped. Every derivation that enters the set is then
//! kept, at the next index of the cell's list, and its fingerprint is stored
//! there already. A set of 16-byte fingerprints therefore holds every key
//! twice. [`IndexSet`] stores the 4-byte index and a 4-byte hash instead,
//! reading the key back through the list only when the hashes agree: open
//! addressing, linear probing, at most three quarters full. Which one a node holds is [`super::Walk::dedup_index`].

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

/// A slot no index occupies.
const EMPTY: u32 = u32::MAX;

/// An open-addressing set of derivation indices, keyed by what each index's
/// derivation spells. Each slot keeps the key's 32-bit hash beside the index,
/// so a probe reads a derivation only when the hashes agree, and growing the
/// table reads none.
#[derive(Default)]
pub(super) struct IndexSet {
    slots: Vec<(u32, u32)>,
    len: usize,
}

/// A key's 32-bit hash; its low bits pick the home slot. Fingerprints are
/// hashes already; the multiply only mixes the two halves together.
fn hash(key: u128) -> u32 {
    let x = (key as u64) ^ ((key >> 64) as u64);
    (x.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32
}

impl IndexSet {
    /// Whether no index held yet spells `key` (`spelled(i)` is what index `i`
    /// spells); if so, record `next`, the index the caller keeps it at.
    pub(super) fn insert(&mut self, key: u128, next: u32, spelled: impl Fn(u32) -> u128) -> bool {
        debug_assert_ne!(next, EMPTY, "no node holds u32::MAX derivations");
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let h = hash(key);
        let mut at = h as usize & mask;
        loop {
            match self.slots[at] {
                (EMPTY, _) => {
                    self.slots[at] = (next, h);
                    self.len += 1;
                    return true;
                }
                (i, g) if g == h && spelled(i) == key => return false,
                _ => at = (at + 1) & mask,
            }
        }
    }

    /// Double the table (eight slots at first) and place every index again,
    /// by its stored hash.
    fn grow(&mut self) {
        let size = (self.slots.len() * 2).max(8);
        let old = std::mem::replace(&mut self.slots, vec![(EMPTY, 0); size]);
        let mask = size - 1;
        for (i, h) in old.into_iter().filter(|&(i, _)| i != EMPTY) {
            let mut at = h as usize & mask;
            while self.slots[at].0 != EMPTY {
                at = (at + 1) & mask;
            }
            self.slots[at] = (i, h);
        }
    }

    /// The table's heap bytes.
    pub(super) fn bytes(&self) -> usize {
        self.slots.capacity() * std::mem::size_of::<(u32, u32)>()
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
        assert!(seen.bytes() >= seen.len() * 8 * 4 / 3);
        for (i, &key) in kept.iter().enumerate() {
            assert!(
                !seen.insert(key, kept.len() as u32, |j| kept[j as usize]),
                "index {i}"
            );
        }
    }
}
