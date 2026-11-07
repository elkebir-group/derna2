//! Special-hairpin bonus tables: `Triloops`, `Tetraloops`, `Hexaloops`.
//!
//! Each entry is `(loop_sequence, bonus_37, bonus_dh)`. Lookup is
//! O(N) in the entry count, which is fine — the file format caps
//! these tables at 40 entries each and the built-in table has ≤ 16. A
//! perfect-hash construction (`phf`) would be premature; `Vec` plus a
//! linear scan keeps the parser dependency surface minimal.

/// One row of a special-hairpin bonus table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialHairpinEntry {
    pub seq: String,
    pub bonus_37: i32,
    pub bonus_dh: i32,
}

/// Sequence-keyed bonus table for triloops / tetraloops / hexaloops.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialHairpinTable {
    pub entries: Vec<SpecialHairpinEntry>,
}

impl SpecialHairpinTable {
    /// Returns the 37°C bonus for `seq` if the table contains it.
    /// Accepts bytes so callers can use stack-allocated keys without
    /// a heap allocation.
    pub fn lookup_37(&self, seq: &[u8]) -> Option<i32> {
        self.entries
            .iter()
            .find(|e| e.seq.as_bytes() == seq)
            .map(|e| e.bonus_37)
    }
}
