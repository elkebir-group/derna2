//! A small, fast, deterministic hasher for the CKY tables.
//!
//! The six node-keyed tables are hashed on every merge and snapshot; with
//! `std`'s default `RandomState` (SipHash) that dominates the scan's
//! non-enumeration time (~35% of self-time at GLPE-78). The keys are tiny
//! integer tuples (`Node` is `(u32, u8)`; `PairKey` adds two nucleotide
//! bytes), so a cryptographic hash is overkill — the FxHash multiply-rotate
//! mixing used by `rustc` / Firefox is both faster and deterministic, which
//! also makes a scan's table iteration order reproducible across runs.
//!
//! Vendored (≈30 lines) rather than pulled in as a dependency to avoid the
//! OSV / package-quarantine surface for a function this small.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// A `HashMap` keyed by [`FxHasher`] — a drop-in for `std`'s `HashMap` on
/// the small integer keys the CKY tables use.
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// A `HashSet` keyed by [`FxHasher`] — the set analogue of [`FxHashMap`], used
/// for the small integer `(node, state)` keys the product-DFA reachability
/// pass visits.
pub type FxHashSet<T> = HashSet<T, BuildHasherDefault<FxHasher>>;

/// FxHash 64-bit mixing constant (the `rustc_hash` / Firefox value).
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
/// Per-word rotate before the XOR-multiply mix.
const ROTATE: u32 = 5;

/// The FxHash hasher: rotate, XOR in the next word, multiply by [`SEED`].
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[..8]);
            self.add(u64::from_le_bytes(buf));
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut buf = [0u8; 8];
            buf[..bytes.len()].copy_from_slice(bytes);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
