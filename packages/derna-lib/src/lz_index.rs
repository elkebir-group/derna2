//! Per-size-diagonal index of inner-pair `H` values for the Lyngsø–Zuker
//! reduction of the skewed-large internal-loop closure.
//!
//! For an outer pair `(p, q)` enclosing an inner pair `(k, l)` with both
//! unpaired flanks `≥ 5` (so the broken-codon CAI separates across a codon
//! boundary) and a large asymmetry `d > T` (so `ninio` is saturated and
//! constant), the closure's contribution factors into an outer-corner part
//! and a per-inner-cell value `H(inner)` that is **independent of the outer
//! pair**. The skewed contribution of an outer cell is then
//!
//! ```text
//! OuterMin + min over s [ λ·(size(s) + MAX_NINIO) + min_{k} H(k, k+Δ) ]
//! ```
//!
//! where the inner pair sits on the size diagonal `Δ = l − k = (q−p−2) − s`.
//! This index holds, per diagonal `Δ`, the inner cells' precomputed `H` keyed
//! by their 5′ base position `k`, and answers the inner `min over k ∈ [klo,
//! khi]` the reduction needs in `O(1)`.
//!
//! Inner cells are appended as their `P`-cells finalize — ascending `l`, hence
//! ascending `k` within a diagonal — so each diagonal's entries are
//! `k`-sorted by construction. A sparse table over each diagonal is grown
//! incrementally on append (`O(log)` per insert), and a dense count of the
//! entries below each `k` turns the `[klo, khi]` window into entry indexes in
//! `O(1)`, followed by an `O(1)` range-min. The two one-sided indexes find
//! their windows through the same kind of dense table ([`Starts`]).

use crate::dfa::Node;
use crate::state::{PairKey, Value};

/// One inner pair's reduction record: its precomputed outer-independent
/// `H` value and the coordinates the closure write needs (the `P`-cell table
/// node `io`, its key, and its state). `k = ikey.open.pos`, `l = io.pos − 1`.
#[derive(Debug, Clone, Copy)]
pub struct HEntry {
    /// Inner pair 5′ base position (`ikey.open.pos`); the diagonal's sort key.
    pub k: u32,
    /// Outer-independent reduction value `H(inner)`.
    pub h: f64,
    /// The inner `P`-cell's table node (`l + 1`).
    pub io: Node,
    /// The inner `P`-cell's key.
    pub ikey: PairKey,
    /// The inner `P`-cell's merged accumulators.
    pub st: Value,
}

/// Entry offsets of a list sorted by a small integer key: for each key `v`
/// from the first entry's key up to the last's, the index of the first entry
/// whose key is at least `v`. Replaces the two binary searches of a window
/// lookup with two loads.
#[derive(Default, Clone)]
struct Starts {
    base: u32,
    first: Vec<u32>,
}

impl Starts {
    /// Record that entry `at` (the list's next index) has key `key`, keys
    /// arriving in ascending order.
    fn push(&mut self, key: u32, at: usize) {
        if self.first.is_empty() {
            self.base = key;
        }
        while self.base as usize + self.first.len() <= key as usize {
            self.first.push(at as u32);
        }
    }

    /// The index of the first entry whose key is at least `key`, in a list
    /// of `len` entries.
    fn lower(&self, key: u32, len: usize) -> usize {
        if key <= self.base {
            return 0;
        }
        self.first
            .get((key - self.base) as usize)
            .map_or(len, |&i| i as usize)
    }

    /// The entry-index range whose keys lie in `[lo, hi]`, in a list of
    /// `len` entries.
    fn window(&self, lo: u32, hi: u32, len: usize) -> std::ops::Range<usize> {
        let a = self.lower(lo, len);
        let b = self.lower(hi.saturating_add(1), len);
        a..b.max(a)
    }
}

/// One size diagonal's `k`-sorted entries plus a sparse table for `O(1)`
/// range-min over entry-index windows. `table[j - 1][i]` is the index of the
/// entry with the least `H` in `entries[i ..= i + 2^j − 1]`, the first such on
/// a tie; the windows of one entry need no row. Indexes alone, with `H` read
/// from the entry, hold the table at a quarter of `(H, index)` pairs, and the
/// table is most of the index.
#[derive(Default)]
struct Diag {
    entries: Vec<HEntry>,
    /// `below[k]`: how many entries have a 5′ position below `k`, for every
    /// `k` up to the last entry's; a window's ends are two reads of it rather
    /// than two searches.
    below: Vec<u32>,
    table: Vec<Vec<u32>>,
}

impl Diag {
    /// Append `e` (with `k` at or above every prior entry's) and extend the
    /// sparse table by the windows that now close at the new last index.
    fn push(&mut self, e: HEntry) {
        let m = self.entries.len();
        self.entries.push(e);
        let k = e.k as usize;
        if self.below.len() <= k {
            self.below.resize(k + 1, m as u32);
        }
        let mut j = 1;
        while (1usize << j) <= m + 1 {
            if self.table.len() < j {
                self.table.push(Vec::new());
            }
            let i = m + 1 - (1 << j);
            let least = self.least(self.at(j - 1, i), self.at(j - 1, i + (1 << (j - 1))));
            self.table[j - 1].push(least);
            j += 1;
        }
    }

    /// The index of the entry with the least `H` over the `2^j` entries from
    /// `i`.
    fn at(&self, j: usize, i: usize) -> u32 {
        if j == 0 {
            i as u32
        } else {
            self.table[j - 1][i]
        }
    }

    /// Of entries `a` and `b`, the one with the lesser `H`; `a` on a tie.
    fn least(&self, a: u32, b: u32) -> u32 {
        if self.entries[a as usize].h <= self.entries[b as usize].h {
            a
        } else {
            b
        }
    }

    /// How many entries have a 5′ position below `k`.
    fn count_below(&self, k: u32) -> usize {
        self.below
            .get(k as usize)
            .map_or(self.entries.len(), |&c| c as usize)
    }

    /// The smallest `H` whose `k` lies in `[klo, khi]` and its entry's
    /// index, or `None` when the window is empty.
    fn window_min(&self, klo: u32, khi: u32) -> Option<(f64, u32)> {
        let lo = self.count_below(klo);
        let hi = self.count_below(khi.saturating_add(1));
        if lo >= hi {
            return None;
        }
        let (a, b) = (lo, hi - 1);
        let j = (b - a + 1).ilog2() as usize;
        let least = self.least(self.at(j, a), self.at(j, b + 1 - (1 << j)));
        Some((self.entries[least as usize].h, least))
    }
}

/// Size-diagonal index of [`HEntry`] records, keyed by the inner pair span
/// `Δ = l − k` (a pair has `k < l`, so `Δ ≥ 1`); slot `Δ` holds its diagonal.
#[derive(Default)]
pub struct HIndex {
    by_diag: Vec<Diag>,
}

impl HIndex {
    /// An empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `entry` to its size diagonal `delta = l − k`.
    pub fn insert(&mut self, delta: i64, entry: HEntry) {
        let d = usize::try_from(delta).expect("an inner pair's span is not negative");
        if self.by_diag.len() <= d {
            self.by_diag.resize_with(d + 1, Diag::default);
        }
        self.by_diag[d].push(entry);
    }

    /// The entry with the smallest `H` on diagonal `delta` whose `k` lies in
    /// `[klo, khi]`, or `None` when the window is empty.
    #[cfg(test)]
    pub fn range_min(&self, delta: i64, klo: u32, khi: u32) -> Option<HEntry> {
        let (_, i) = self.window_min(delta, klo, khi)?;
        Some(self.entry(delta, i))
    }

    /// [`Self::range_min`]'s `H` and the entry's index on its diagonal, without
    /// reading the entry: a caller choosing among many windows reads only the
    /// winner's, with [`Self::entry`].
    pub fn window_min(&self, delta: i64, klo: u32, khi: u32) -> Option<(f64, u32)> {
        self.by_diag
            .get(usize::try_from(delta).ok()?)?
            .window_min(klo, khi)
    }

    /// Entry `i` of diagonal `delta`, as [`Self::window_min`] named it.
    pub fn entry(&self, delta: i64, i: u32) -> HEntry {
        self.by_diag[delta as usize].entries[i as usize]
    }

    /// The heap the index holds, estimated from its allocated capacities:
    /// the entries, the per-position counts, and the sparse range-min table
    /// (whose rows together hold about `entries * log2(entries)` pairs). An
    /// estimate, for attributing a scan's memory — not an allocator
    /// measurement.
    pub fn approx_bytes(&self) -> usize {
        use std::mem::size_of;
        self.by_diag
            .iter()
            .map(|d| {
                d.entries.capacity() * size_of::<HEntry>()
                    + d.below.capacity() * size_of::<u32>()
                    + d.table.capacity() * size_of::<Vec<u32>>()
                    + d.table
                        .iter()
                        .map(|row| row.capacity() * size_of::<u32>())
                        .sum::<usize>()
            })
            .sum::<usize>()
            + self.by_diag.capacity() * size_of::<Diag>()
    }

    /// How many entries the index holds, over every diagonal.
    pub fn entries(&self) -> usize {
        self.by_diag.iter().map(|d| d.entries.len()).sum()
    }
}

/// One inner pair's record for the **one-sided** reduction (one flank `≥ 5`,
/// the other `∈ {2,3,4}`): its `P`-cell coordinates and state, its own BRD, and
/// the precomputed inner-half minima of the *large* separated flank.
///
/// For the large-5′ family (large `n1`, small `n2`) the stored minimum is
/// `B(r5) = min over l3 [ λ·MM_inner(I_rev, r5, l3) − K·DL(l3) ]`, keyed by the
/// 3′-flank 5′ terminal `r5`; for the large-3′ family (small `n1`, large `n2`)
/// it is `Bm(l3) = min over r5 [ λ·MM_inner(I_rev, r5, l3) − K·DR(r5) ]`, keyed
/// by the 5′-flank 3′ terminal `l3`. Both are outer-independent (they read only
/// the inner pair and one terminal nucleotide), so they are computed once when
/// the inner `P`-cell finalizes. `INFINITY` marks a nucleotide with no edge.
/// Each index reads one family's, so an entry holds only that one:
/// [`OneSideIndex`]'s `B`, [`OneSideKIndex`]'s `Bm`.
#[derive(Debug, Clone, Copy)]
pub struct OneSideEntry {
    /// Inner pair 5′ base position (`ikey.open.pos`); the column's sort key.
    pub k: u32,
    /// The inner `P`-cell's table node (`l + 1`).
    pub io: Node,
    /// The inner `P`-cell's key.
    pub ikey: PairKey,
    /// The inner `P`-cell's merged accumulators.
    pub st: Value,
    /// The inner pair's own BRD `st.brd(lambda)`.
    pub brd: f64,
    /// The index's family's inner half: in [`OneSideIndex`], the large-5′
    /// `B(r5)` keyed by the 3′-flank 5′ terminal `r5`; in [`OneSideKIndex`],
    /// the large-3′ `Bm(l3)` keyed by the 5′-flank 3′ terminal `l3`
    /// (`A/C/G/U → 0..4`).
    pub bmin: [f64; 4],
    /// The cell's place among its position's cells in the order the scan
    /// visits them (each node in turn, each node's cells in table order).
    pub rank: u32,
}

/// Column index of [`OneSideEntry`] keyed by the inner pair's 3′ base position
/// `l = io.pos − 1`. Each column is `k`-sorted (ascending) so the outer cell's
/// eligible `k`-window is a contiguous slice answered by binary search.
#[derive(Default)]
pub struct OneSideIndex {
    by_col: Vec<Vec<OneSideEntry>>,
    starts: Vec<Starts>,
}

impl OneSideIndex {
    /// An index sized for a `seq_len`-nucleotide DFA.
    pub fn new(seq_len: u32) -> Self {
        Self {
            by_col: vec![Vec::new(); seq_len as usize + 1],
            starts: vec![Starts::default(); seq_len as usize + 1],
        }
    }

    /// Append `entry` to column `l`. Entries within a column are added when one
    /// `P`-position finalizes, so the column is sorted once afterwards.
    pub fn push(&mut self, l: u32, entry: OneSideEntry) {
        self.by_col[l as usize].push(entry);
    }

    /// Sort column `l` ascending by `k` (called once after a position's cells
    /// are appended), so [`Self::window`] can binary-search the `k`-range.
    pub fn sort_col(&mut self, l: u32) {
        let col = &mut self.by_col[l as usize];
        // Gathered into a new column of exactly its length: it takes no more
        // entries once sorted.
        *col = SortProxy::order(col).map(|i| col[i]).collect();
        let starts = &mut self.starts[l as usize];
        *starts = Starts::default();
        for (i, e) in col.iter().enumerate() {
            starts.push(e.k, i);
        }
    }

    /// The `k`-sorted entries of column `l` whose `k` lies in `[klo, khi]`.
    pub fn window(&self, l: u32, klo: u32, khi: u32) -> &[OneSideEntry] {
        let col = &self.by_col[l as usize];
        &col[self.starts[l as usize].window(klo, khi, col.len())]
    }

    /// The heap the index holds, estimated from its columns' capacities.
    pub fn approx_bytes(&self) -> usize {
        columns_bytes(&self.by_col)
    }

    /// How many entries the index holds, over every column.
    pub fn entries(&self) -> usize {
        self.by_col.iter().map(Vec::len).sum()
    }
}

/// A column entry's sort key and place, padded to the size a column entry
/// had when the scan's recorded designs were made. An unstable sort picks
/// its method by element size, so sorting the slimmer entries directly puts
/// entries that share a `k` in another order, and a fold then breaks a tie
/// between equal designs the other way. Sorting these instead keeps the
/// order, and with it every recorded design.
#[derive(Clone, Copy)]
struct SortProxy {
    k: u32,
    at: u32,
    _pad: [u64; 15],
}

const _: () = assert!(std::mem::size_of::<SortProxy>() == 128);

impl SortProxy {
    /// The places of `col`'s entries in the order a `k`-keyed unstable sort
    /// of 128-byte entries leaves them.
    fn order(col: &[OneSideEntry]) -> impl Iterator<Item = usize> {
        let mut proxies: Vec<Self> = (0u32..)
            .zip(col)
            .map(|(at, e)| Self {
                k: e.k,
                at,
                _pad: [0; 15],
            })
            .collect();
        proxies.sort_unstable_by_key(|p| p.k);
        proxies.into_iter().map(|p| p.at as usize)
    }
}

/// The heap a bucketed [`OneSideEntry`] index holds, from its buckets'
/// allocated capacities.
fn columns_bytes(buckets: &[Vec<OneSideEntry>]) -> usize {
    use std::mem::size_of;
    buckets
        .iter()
        .map(|c| c.capacity() * size_of::<OneSideEntry>())
        .sum::<usize>()
        + std::mem::size_of_val(buckets)
}

/// Index of [`OneSideEntry`] keyed by the inner pair's 5′ base position `k`
/// (`ikey.open.pos`), for the large-3′ family (small `n1`, so `k` is pinned
/// near the outer opening while the 3′ base `l` varies). Inner cells finalize
/// in ascending `l` order, so each bucket is `l`-sorted by construction — no
/// post-hoc sort — and the eligible `l`-window is a binary-searched slice.
/// The bucket's sort key is `io.pos = l + 1`.
#[derive(Default)]
pub struct OneSideKIndex {
    by_k: Vec<Vec<OneSideEntry>>,
    starts: Vec<Starts>,
}

impl OneSideKIndex {
    /// An index sized for a `seq_len`-nucleotide DFA.
    pub fn new(seq_len: u32) -> Self {
        Self {
            by_k: vec![Vec::new(); seq_len as usize + 1],
            starts: vec![Starts::default(); seq_len as usize + 1],
        }
    }

    /// Append `entry` to bucket `k`. Cells finalize in ascending `l` (hence
    /// ascending `io.pos`), so buckets stay `l`-sorted without an explicit sort.
    pub fn push(&mut self, k: u32, entry: OneSideEntry) {
        let bucket = &mut self.by_k[k as usize];
        self.starts[k as usize].push(entry.io.pos, bucket.len());
        bucket.push(entry);
    }

    /// The `l`-sorted entries of bucket `k` whose 3′ base `l = io.pos − 1` lies
    /// in `[llo, lhi]`.
    pub fn window(&self, k: u32, llo: u32, lhi: u32) -> &[OneSideEntry] {
        let bucket = &self.by_k[k as usize];
        &bucket[self.starts[k as usize].window(llo + 1, lhi + 1, bucket.len())]
    }

    /// The heap the index holds, estimated from its buckets' capacities.
    pub fn approx_bytes(&self) -> usize {
        columns_bytes(&self.by_k)
    }

    /// How many entries the index holds, over every bucket.
    pub fn entries(&self) -> usize {
        self.by_k.iter().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::PairKey;
    use rna_mfe::{Energy, Nucleotide};

    fn entry(k: u32, h: f64) -> HEntry {
        let node = Node::new(k, 0);
        HEntry {
            k,
            h,
            io: node,
            ikey: PairKey {
                open: node,
                close: node,
                opening: crate::state::Opening::new(Nucleotide::A, 0),
                nuc_close: Nucleotide::U,
            },
            st: Value {
                mfe: Energy::ZERO,
                cai: 0.0,
            },
        }
    }

    /// The sparse-table range-min agrees with a brute-force scan over every
    /// `[klo, khi]` window, for a diagonal built by incremental append.
    #[test]
    fn range_min_matches_brute_force() {
        let hs = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 0.5, 3.5];
        let mut idx = HIndex::new();
        for (i, &h) in hs.iter().enumerate() {
            idx.insert(7, entry(i as u32 * 2, h)); // k = 0, 2, 4, ... (sorted)
        }
        for klo in 0..20u32 {
            for khi in klo..20u32 {
                let brute = hs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        let k = *i as u32 * 2;
                        k >= klo && k <= khi
                    })
                    .map(|(_, &h)| h)
                    .fold(f64::INFINITY, f64::min);
                let got = idx.range_min(7, klo, khi).map_or(f64::INFINITY, |e| e.h);
                assert_eq!(got, brute, "klo={klo} khi={khi}");
            }
        }
    }

    /// The key-offset windows agree with binary search on a list with
    /// repeated and missing keys, for every query window.
    #[test]
    fn starts_window_matches_binary_search() {
        let keys = [3u32, 3, 4, 7, 7, 7, 9, 12, 12, 15];
        let mut starts = Starts::default();
        for (i, &k) in keys.iter().enumerate() {
            starts.push(k, i);
        }
        for lo in 0..20u32 {
            for hi in 0..20u32 {
                let a = keys.partition_point(|&k| k < lo);
                let b = keys.partition_point(|&k| k <= hi).max(a);
                assert_eq!(starts.window(lo, hi, keys.len()), a..b, "lo={lo} hi={hi}");
            }
        }
    }

    /// A query on an absent diagonal yields nothing.
    #[test]
    fn empty_diagonal_is_none() {
        let idx = HIndex::new();
        assert!(idx.range_min(3, 0, 100).is_none());
    }
}
