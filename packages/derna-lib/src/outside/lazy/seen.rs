//! The subsequences a node has kept, held as indices into its own derivation
//! list.
//!
//! Under node dedup ([`super::Walk::node_dedup`]) each cell remembers what its
//! kept derivations spell, so that a later derivation spelling the same
//! subsequence can be dropped. Every derivation that enters the set is then
//! kept, at the next index of the cell's list, and its fingerprint is stored
//! there already. [`IndexSet`] stores a 4-byte index and a 1-byte tag of the
//! key's hash, reading the key back through the list only where a tag agrees:
//! open addressing, linear probing, at most three quarters full. Which one a
//! node holds is [`super::Walk::dedup_index`].
//!
//! A key is a 64-bit fingerprint, which two different subsequences can share.
//! Both forms therefore treat an equal key as a candidate match only, and ask
//! the caller whether the two derivations really spell the same subsequence;
//! a collision keeps the newcomer as a distinct entry beside the old one.
//!
//! The index vector's bytes are resident rather than merely counted: the empty
//! marker is not all-zero, so it is written through at allocation and no
//! shared zero page can serve it. The tag vector's free marker *is* zero, so a
//! fresh tag page is reserved rather than committed and becomes resident as
//! the slots around it are written. Narrowing a slot therefore gives back
//! resident memory, four fifths of it immediately.

use crate::hash::FxHashMap;

/// A node's dedup set: each key's first index in a hash map, or indices
/// alone in an [`IndexSet`].
pub(super) enum Seen {
    Keys(KeySet),
    Index(IndexSet),
}

impl Default for Seen {
    fn default() -> Self {
        Seen::Keys(KeySet::default())
    }
}

impl Seen {
    /// An empty set of indices (`index`) or of keys.
    pub(super) fn new(index: bool) -> Self {
        if index {
            Seen::Index(IndexSet::default())
        } else {
            Seen::default()
        }
    }

    /// Whether no derivation kept yet spells what the newcomer spells, its
    /// fingerprint being `key`; if so, record it as derivation `next`.
    /// `key_of(i)` is derivation `i`'s fingerprint, and `same(i)` whether
    /// derivation `i` spells exactly what the newcomer does — asked only where
    /// the fingerprints agree.
    pub(super) fn insert(
        &mut self,
        key: u64,
        next: u32,
        key_of: impl Fn(u32) -> u64,
        same: impl FnMut(u32) -> bool,
    ) -> bool {
        match self {
            Seen::Keys(set) => set.insert(key, next, key_of, same),
            Seen::Index(set) => set.insert(key, next, key_of, same),
        }
    }

    /// The set's heap bytes (a hash map's estimated at the swiss table's
    /// 7/8 load and one control byte per slot).
    pub(super) fn bytes(&self) -> usize {
        match self {
            Seen::Keys(set) => set.bytes(),
            Seen::Index(set) => set.bytes(),
        }
    }
}

/// Each key's first index, and the indices whose keys repeat an earlier one's
/// while spelling something else: a collision, so `more` is nearly always
/// empty.
#[derive(Default)]
pub(super) struct KeySet {
    first: FxHashMap<u64, u32>,
    more: Vec<u32>,
}

impl KeySet {
    fn insert(
        &mut self,
        key: u64,
        next: u32,
        key_of: impl Fn(u32) -> u64,
        mut same: impl FnMut(u32) -> bool,
    ) -> bool {
        let i = *self.first.entry(key).or_insert(next);
        if i == next {
            return true;
        }
        if same(i) {
            return false;
        }
        let repeat = self.more.iter().any(|&j| key_of(j) == key && same(j));
        if !repeat {
            self.more.push(next);
        }
        !repeat
    }

    fn bytes(&self) -> usize {
        let cap = self.first.capacity();
        let map = if cap > 0 {
            (cap * 8 / 7 + 1) * (std::mem::size_of::<(u64, u32)>() + 1)
        } else {
            0
        };
        map + self.more.capacity() * std::mem::size_of::<u32>()
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
/// hashes already; the multiply moves the high bits' mixing to the low ones.
fn hash(key: u64) -> u32 {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32
}

/// The tag a hash gives a slot, never [`FREE`].
fn tag(h: u32) -> u8 {
    ((h >> 24) as u8) | 1
}

impl IndexSet {
    /// Whether no index held yet spells what the newcomer spells, its key
    /// being `key` (`key_of(i)` is index `i`'s key, and `same(i)` whether
    /// index `i` spells exactly what the newcomer does); if so, record
    /// `next`, the index the caller keeps it at. An index whose key agrees
    /// but whose spelling does not is a collision: the probe goes on past it.
    pub(super) fn insert(
        &mut self,
        key: u64,
        next: u32,
        key_of: impl Fn(u32) -> u64,
        mut same: impl FnMut(u32) -> bool,
    ) -> bool {
        debug_assert_ne!(next, EMPTY, "no node holds u32::MAX derivations");
        if (self.len + 1) * 4 > self.tags.len() * 3 {
            self.grow(&key_of);
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
                g if g == t && key_of(self.at[at]) == key && same(self.at[at]) => {
                    return false;
                }
                _ => at = (at + 1) & mask,
            }
        }
    }

    /// Double the table (eight slots at first) and place every index again,
    /// by the hash of what it spells.
    fn grow(&mut self, key_of: &impl Fn(u32) -> u64) {
        let size = (self.tags.len() * 2).max(8);
        let tags = std::mem::replace(&mut self.tags, vec![FREE; size]);
        let old = std::mem::replace(&mut self.at, vec![EMPTY; size]);
        let mask = size - 1;
        for (t, i) in tags.into_iter().zip(old).filter(|&(t, _)| t != FREE) {
            let mut at = hash(key_of(i)) as usize & mask;
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

    /// What the tests store per index: a key, and the identity of what the
    /// index spells. Two entries spell the same thing exactly when their
    /// identities agree; their keys agree then too, and may agree otherwise.
    type Entry = (u64, u64);

    fn offer(set: &mut Seen, kept: &mut Vec<Entry>, e: Entry) -> bool {
        let fresh = set.insert(
            e.0,
            kept.len() as u32,
            |i| kept[i as usize].0,
            |i| kept[i as usize].1 == e.1,
        );
        if fresh {
            kept.push(e);
        }
        fresh
    }

    /// Identities drawn with many repeats, offered in order under a key that
    /// is the identity itself and under one cut to `bits` bits, so that
    /// distinct identities share keys: either form agrees with a standard
    /// set of identities on every answer, through several growths.
    fn agrees_with_a_hash_set(index: bool, bits: u32) {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let cut = |x: u64| if bits >= 64 { x } else { x & ((1 << bits) - 1) };
        let mut kept: Vec<Entry> = Vec::new();
        let mut seen = Seen::new(index);
        let mut reference = HashSet::new();
        let mut by_key: std::collections::HashMap<u64, u64> = Default::default();
        let mut collided = 0;
        for _ in 0..20_000 {
            let r = next();
            // About one identity in three repeats an earlier one.
            let id = if r % 3 == 0 && !kept.is_empty() {
                kept[(r >> 8) as usize % kept.len()].1
            } else {
                next()
            };
            let key = cut(id);
            collided += usize::from(*by_key.entry(key).or_insert(id) != id);
            let fresh = offer(&mut seen, &mut kept, (key, id));
            assert_eq!(fresh, reference.insert(id), "identity {id:x}");
        }
        assert_eq!(kept.len(), reference.len());
        if bits < 64 {
            assert!(collided > 1000, "only {collided} collisions offered");
        }
        for &e in kept.clone().iter() {
            assert!(!offer(&mut seen, &mut kept, e), "{e:x?}");
        }
    }

    #[test]
    fn the_index_set_agrees_with_a_hash_set() {
        agrees_with_a_hash_set(true, 64);
    }

    #[test]
    fn the_key_set_agrees_with_a_hash_set() {
        agrees_with_a_hash_set(false, 64);
    }

    #[test]
    fn the_index_set_agrees_with_a_hash_set_through_collisions() {
        agrees_with_a_hash_set(true, 8);
    }

    #[test]
    fn the_key_set_agrees_with_a_hash_set_through_collisions() {
        agrees_with_a_hash_set(false, 8);
    }

    /// The index set's size bounds: five bytes a slot, and at least three
    /// quarters of the way to full after the growth that made room for the
    /// last entry.
    #[test]
    fn an_index_slot_is_five_bytes_at_three_quarters_load() {
        let mut set = IndexSet::default();
        let kept: Vec<u64> = (0..5_000u64)
            .map(|i| i.wrapping_mul(0x517C_C1B7_2722_0A95))
            .collect();
        for (i, &k) in kept.iter().enumerate() {
            assert!(set.insert(k, i as u32, |j| kept[j as usize], |_| unreachable!()));
        }
        assert_eq!(set.len(), kept.len());
        assert!(set.bytes() >= set.len() * 5 * 4 / 3);
        assert!(set.bytes() <= (set.len() + 1) * 5 * 8 / 3);
    }

    /// Two different spellings under one key: each form keeps both, answers
    /// a repeat of either as a repeat, and asks for a spelling only where
    /// the keys agree.
    #[test]
    fn a_shared_key_with_different_spellings_keeps_both() {
        for index in [true, false] {
            let mut seen = Seen::new(index);
            let mut kept: Vec<Entry> = Vec::new();
            assert!(offer(&mut seen, &mut kept, (7, 1)));
            assert!(offer(&mut seen, &mut kept, (7, 2)), "a collision dropped");
            assert!(offer(&mut seen, &mut kept, (7, 3)), "a second collision");
            assert!(offer(&mut seen, &mut kept, (8, 4)));
            for e in [(7, 1), (7, 2), (7, 3), (8, 4)] {
                assert!(!offer(&mut seen, &mut kept, e), "{index} {e:?}");
            }
            let mut asked = Vec::new();
            let fresh = seen.insert(
                9,
                kept.len() as u32,
                |i| kept[i as usize].0,
                |i| {
                    asked.push(i);
                    false
                },
            );
            assert!(fresh);
            assert!(
                asked.is_empty(),
                "{index}: spelled {asked:?} under another key"
            );
        }
    }

    /// A slot is five bytes. Six distinct keys fit the first table the set
    /// allocates, so its bytes are the eight slots' and nothing else: a slot
    /// of the bare index reads 32 here, and one carrying a 32-bit hash 64.
    #[test]
    fn six_keys_hold_in_forty_bytes() {
        let kept: Vec<u64> = (0..6).map(|i| (0xABCD_u64 << 48) | i).collect();
        let mut seen = IndexSet::default();
        for (i, &key) in kept.iter().enumerate() {
            assert!(
                seen.insert(key, i as u32, |j| kept[j as usize], |_| false),
                "key {i}"
            );
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
        let kept = [0x0F1E_2D3C_u64 << 32];
        let mut seen = IndexSet::default();
        assert!(seen.insert(kept[0], 0, |i| kept[i as usize], |_| false));
        let index: Vec<u8> = seen.at.iter().flat_map(|s| s.to_ne_bytes()).collect();
        assert_eq!(index.len(), 32, "eight slots of four bytes");
        // Seven free entries of four non-zero bytes each; the eighth holds
        // index 0, which is four zero bytes.
        assert_eq!(index.iter().filter(|&&b| b != 0).count(), 28);
        // One tag of eight is written; the other seven are still zero.
        assert_eq!(seen.tags.iter().filter(|&&t| t != FREE).count(), 1);
    }

    /// The multiplier's inverse modulo 2^64, by Newton's iteration.
    fn inverse(c: u64) -> u64 {
        let mut x = c;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(c.wrapping_mul(x)));
        }
        assert_eq!(c.wrapping_mul(x), 1);
        x
    }

    /// Two distinct keys with the same 32-bit hash, and so the same tag and
    /// the same home slot: the set separates them on their keys, and keeps
    /// doing so after the table has been grown and both moved.
    #[test]
    fn keys_sharing_a_hash_stay_distinct_across_a_growth() {
        let inv = inverse(0x9E37_79B9_7F4A_7C15);
        let a = 0x0123_4567_89AB_CDEF_u64.wrapping_mul(inv);
        let b = 0x0123_4567_89AB_CDEE_u64.wrapping_mul(inv);
        assert_ne!(a, b);
        assert_eq!(hash(a), hash(b), "products sharing their high half");
        let mut kept = vec![a, b];
        let mut seen = IndexSet::default();
        let never = |_| -> bool { unreachable!("no two keys here agree") };
        assert!(seen.insert(a, 0, |i| kept[i as usize], never));
        assert!(seen.insert(b, 1, |i| kept[i as usize], never));
        // Enough further keys to double the table twice, so both have been
        // placed again by the hash of their keys.
        for i in 0..40u64 {
            let key = (i << 40) | 0x5555;
            let at = kept.len() as u32;
            if seen.insert(key, at, |j| kept[j as usize], never) {
                kept.push(key);
            }
        }
        let n = kept.len() as u32;
        assert!(!seen.insert(a, n, |i| kept[i as usize], |i| kept[i as usize] == a));
        assert!(!seen.insert(b, n, |i| kept[i as usize], |i| kept[i as usize] == b));
    }

    /// Two distinct keys that share a tag but not a hash: the tag is one byte
    /// of the hash, so a probe reads a key back where it need not have, and
    /// still answers correctly. Without this the spurious read is untested.
    #[test]
    fn keys_sharing_only_a_tag_stay_distinct() {
        let a = 0x1111_2222_3333_4444_u64;
        let b = (1..100_000u64)
            .map(|i| a ^ (i << 20))
            .find(|&b| tag(hash(a)) == tag(hash(b)) && hash(a) != hash(b))
            .expect("a key sharing the tag but not the hash");
        let kept = [a, b];
        let mut seen = IndexSet::default();
        let eq = |k: u64| move |i: u32| kept[i as usize] == k;
        assert!(seen.insert(a, 0, |i| kept[i as usize], eq(a)));
        assert!(seen.insert(b, 1, |i| kept[i as usize], eq(b)));
        assert_eq!(seen.len(), 2);
        assert!(!seen.insert(a, 2, |i| kept[i as usize], eq(a)));
        assert!(!seen.insert(b, 2, |i| kept[i as usize], eq(b)));
    }
}
