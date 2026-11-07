//! MFE DP table representations.
//!
//! Five tables drive the recurrence (fill order is documented in
//! [`super::fill`]). Their meanings at `dangles=0`:
//!
//! | Table  | Meaning                                                                                                |
//! |--------|--------------------------------------------------------------------------------------------------------|
//! | `V`    | MFE of structures on `seq[i..=j]` *given that* `i` and `j` pair                                        |
//! | `W`    | MFE of *any* structure on `seq[i..=j]`                                                                 |
//! | `F`    | MFE of an exterior loop covering the prefix `seq[0..=j]`                                               |
//! | `M`    | MFE of a *partial* multi-loop on `seq[i..=j]` (≥1 branch + unpaired flanks)                            |
//! | `M1`   | MFE of *exactly one* branch + trailing unpaired in a multi-loop on `seq[i..=j]`                        |
//!
//! ## Indexing convention
//!
//! All `(i, j)` indices are **closed-inclusive, zero-based** —
//! `V[i, j]` covers `seq[i..=j]`. The
//! diagonal `i == j` is reserved for empty / unpaired single bases
//! and the upper triangle `j < i` is unused.
//!
//! ## Storage
//!
//! Each pairwise table is a flat `Vec<i32>` of length `n * (n + 1) / 2`,
//! holding only the lower-triangular cells with `0 <= i <= j < n`.
//! Cells are addressed via [`TriangleTable::idx`] as `indx[j] + i`, a
//! column-wise packing with `indx[j] = j * (j + 1) / 2` (the 0-based
//! form of the 1-based `idx[j] = j * (j - 1) / 2`). The "no valid
//! structure" sentinel is `TriangleTable::INF = 10_000_000`; accessor
//! methods round-trip to `Option<Energy>` for caller convenience.
//!
//! Two cache wins versus the prior `n * n` row-major `Vec<Option<Energy>>`:
//!
//! - **Column-major:** reads with fixed `j` and varying `i` (the cubic
//!   inner loops in `multi::m_split_total` / `multi::multi_split_total`
//!   and the F-table's `prefix_plus_stem`) are stride-1 in this layout
//!   versus stride-`n` in the prior row-major shape.
//! - **4× smaller:** packing only the `i <= j` triangle removes the
//!   unused half, and replacing `Option<Energy>` (8 bytes per cell at
//!   the `i32` payload) with raw `i32 + INF` halves the per-cell width
//!   again.
//!
//! `F` is a 1-D vector of length `n + 1`. `F[0]` is the base case
//! `Some(Energy::ZERO)` (empty prefix has zero free energy at
//! `dangles=0`); `F[j]` for `j >= 1` is the MFE of `seq[0..j]`
//! (note: open-end on `F` only — `F[j]` covers the first `j` bases,
//! so `F[n]` is the MFE of the whole sequence).

use crate::loops::Energy;

/// Pairwise DP table — one cell per `(i, j)` with `i <= j < n`. Cells
/// store the raw `i32` energy in dekacalories per mol; the `INF`
/// sentinel encodes "no valid structure".
#[derive(Debug, Clone)]
pub struct TriangleTable {
    pub n: usize,
    cells: Vec<i32>,
    indx: Vec<usize>,
}

impl TriangleTable {
    /// Sentinel for "no valid structure". Comfortably below
    /// `i32::MAX / 2`, so a handful of `INF + small` additions cannot
    /// overflow even if the caller forgets to short-circuit on the
    /// sentinel — the `get`/`set` shims on this type round-trip
    /// through `Option<Energy>` to preclude that case in practice.
    pub const INF: i32 = 10_000_000;

    /// Fresh `n * (n + 1) / 2`-cell table with all cells initialized
    /// to `INF`. The companion `indx` array of length `n` is computed
    /// once at construction.
    pub fn empty(n: usize) -> Self {
        let cells = vec![Self::INF; n * (n + 1) / 2];
        let indx = (0..n).map(|j| j * (j + 1) / 2).collect();
        Self { n, cells, indx }
    }

    /// Read `T[i, j]`. Returns `None` for INF-sentinel cells. Caller
    /// must ensure `i <= j < n`.
    pub fn get(&self, i: usize, j: usize) -> Option<Energy> {
        let raw = self.cells[self.idx(i, j)];
        if raw == Self::INF {
            None
        } else {
            Some(Energy(raw))
        }
    }

    /// Write `T[i, j]`. `None` writes the INF sentinel. Caller must
    /// ensure `i <= j < n`.
    pub fn set(&mut self, i: usize, j: usize, e: Option<Energy>) {
        let idx = self.idx(i, j);
        self.cells[idx] = e.map_or(Self::INF, |Energy(raw)| raw);
    }

    /// Stride-1 view of column `j`: `cells[indx[j]..=indx[j] + j]`.
    /// Element `k` of the returned slice is the raw `i32` of `T[k, j]`
    /// (with `INF` for empty cells), for `k` in `0..=j`. Used by the
    /// flat zip-add-min kernel in [`super::multi`] to read varying-`i`
    /// fixed-`j` cells contiguously.
    pub fn raw_col(&self, j: usize) -> &[i32] {
        debug_assert!(j < self.n, "raw_col({j}) out of bounds for n={}", self.n);
        let start = self.indx[j];
        &self.cells[start..start + j + 1]
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(
            i <= j && j < self.n,
            "TriangleTable index ({i}, {j}) out of bounds for n={}",
            self.n,
        );
        self.indx[j] + i
    }
}

/// Bundle of all five MFE DP tables for a sequence of length `n`.
/// `f` carries `n + 1` cells (prefix indexing); the other four tables
/// pack the `i <= j` triangle into `n * (n + 1) / 2` cells each.
#[derive(Debug, Clone)]
pub struct DpTables {
    pub n: usize,
    pub v: TriangleTable,
    pub w: TriangleTable,
    pub m: TriangleTable,
    pub m1: TriangleTable,
    pub f: Vec<Option<Energy>>,
}

impl DpTables {
    /// Allocate empty tables sized for a sequence of length `n`.
    /// `F[0]` is initialized to `Some(Energy::ZERO)` (the empty prefix).
    pub fn empty(n: usize) -> Self {
        let mut f = vec![None; n + 1];
        f[0] = Some(Energy::ZERO);
        Self {
            n,
            v: TriangleTable::empty(n.max(1)),
            w: TriangleTable::empty(n.max(1)),
            m: TriangleTable::empty(n.max(1)),
            m1: TriangleTable::empty(n.max(1)),
            f,
        }
    }
}

/// Minimum unpaired count between a hairpin's closing pair. A pair
/// `(i, j)` can close a hairpin only when `j - i - 1 >= MIN_HAIRPIN_LOOP`.
pub const MIN_HAIRPIN_LOOP: usize = 3;

/// Largest unpaired count (per side or summed) the fold allows in
/// bulge/internal loops before the size table runs out and would have
/// to extrapolate (`MAXLOOP = 30`). Bulges cap on the single-side
/// count; internal loops cap on the total `u1 + u2 <= MAX_LOOP_SIZE`.
pub const MAX_LOOP_SIZE: usize = 30;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `(i, j)` with `0 <= i <= j < n` maps to a unique flat
    /// index in `[0, n * (n + 1) / 2)`. Catches any off-by-one in
    /// `indx[]` (e.g., using the 1-based formula unadjusted).
    #[test]
    fn idx_is_injective_and_dense() {
        let n = 10;
        let table = TriangleTable::empty(n);
        let total = n * (n + 1) / 2;
        let mut seen = vec![false; total];
        for j in 0..n {
            for i in 0..=j {
                let idx = table.idx(i, j);
                assert!(idx < total, "idx({i},{j}) = {idx} out of bounds");
                assert!(!seen[idx], "idx({i},{j}) = {idx} collides");
                seen[idx] = true;
            }
        }
        assert!(seen.into_iter().all(|hit| hit), "indexing leaves holes");
    }

    #[test]
    fn get_set_round_trips_some_and_none() {
        let mut table = TriangleTable::empty(4);
        assert_eq!(table.get(1, 3), None);
        table.set(1, 3, Some(Energy(-250)));
        assert_eq!(table.get(1, 3), Some(Energy(-250)));
        table.set(1, 3, None);
        assert_eq!(table.get(1, 3), None);
    }
}
