//! Positions required to carry a base pair, and the `O(1)` range queries the
//! scan's loop closures use to honour them.
//!
//! A must-pair constraint is a predicate on an **absolute position**: the
//! recovered structure may not leave that position unpaired. It needs no DP
//! state, because "unpaired" is not a property a cell carries — it is *declared*
//! by whichever site skips over the position. The scan leaves bases unpaired at
//! exactly four kinds of site:
//!
//! - the two single-base extensions (`C` on the exterior loop, `M` inside a
//!   multi-loop), each declaring one position;
//! - a hairpin's interior, `[a + 1, b)` for the closing pair `(a, b)`;
//! - a single-branch loop's two flanks, `[p + 1, k)` and `[l + 1, q)` for the
//!   outer pair `(p, q)` around the inner pair `(k, l)`;
//! - a multi-loop's two unpaired stretches, `[p + 1, core_start)` and
//!   `[core_end, q)`.
//!
//! Every one of those declares a **range**, the ranges a derivation declares are
//! disjoint, and their union is exactly its unpaired set. So enforcing "no
//! must-pair position inside the range about to be declared unpaired" at each
//! site enforces the constraint globally, with no coordination between sites.
//!
//! Each range is delimited by coordinates the enclosing closure already
//! enumerates, so the predicate never filters a set a reduction has already
//! minimized over — it *narrows the enumeration range* the reduction queries.
//! [`MustPairConstraint::max_right`] and [`MustPairConstraint::min_left`] express that
//! directly: they convert "this flank must be free" into a bound on the flank's
//! far coordinate, which the interior reductions apply to the `k`-window their
//! index is already binary-searched over.
//!
//! A constraint built from an empty position set is **inert**: every query answers as
//! though no position were constrained, so the fold is byte-identical to the
//! unconstrained substrate.

/// Answer to [`MustPairConstraint::max_right`] when no must-pair position bounds the
/// range — the flank may extend as far as the rest of the recurrence allows.
const UNBOUNDED: u32 = u32::MAX;

/// The must-pair predicate resolved against a concrete sequence length: two
/// precomputed next/previous tables answering every site's range query in
/// `O(1)`.
///
/// A default-constructed constraint is inert. Construct an active one with
/// [`MustPairConstraint::new`].
#[derive(Debug, Clone, Default)]
pub(crate) struct MustPairConstraint {
    /// `first_from[i]` is the smallest must-pair position `>= i`, or
    /// [`UNBOUNDED`] when there is none.
    first_from: Vec<u32>,
    /// `last_before[i]` is the largest must-pair position `< i`, or `0` when
    /// there is none — `0` is the vacuous bound for every query that reads it.
    last_before: Vec<u32>,
    /// Fast path: skip every query when no position is constrained.
    active: bool,
}

impl MustPairConstraint {
    /// The inert constraint — no position is required to pair.
    pub(crate) fn inert() -> Self {
        Self::default()
    }

    /// Whether some position is required to pair.
    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    /// Resolve `positions` (absolute, 0-based, out-of-range entries ignored)
    /// against a `seq_len`-nucleotide sequence. An empty set yields
    /// [`MustPairConstraint::inert`].
    pub(crate) fn new(positions: &[u32], seq_len: u32) -> Self {
        if positions.is_empty() {
            return Self::inert();
        }
        let n = seq_len as usize + 2;
        let mut required = vec![false; n];
        for &p in positions {
            if p < seq_len {
                required[p as usize] = true;
            }
        }
        let mut first_from = vec![UNBOUNDED; n];
        for i in (0..n).rev() {
            if required[i] {
                first_from[i] = i as u32;
            } else if i + 1 < n {
                first_from[i] = first_from[i + 1];
            }
        }
        let mut last_before = vec![0u32; n];
        for i in 1..n {
            last_before[i] = if required[i - 1] {
                i as u32 - 1
            } else {
                last_before[i - 1]
            };
        }
        Self {
            first_from,
            last_before,
            active: true,
        }
    }

    /// Table index for `p`, clamped to the tables' domain so a query one past
    /// the sequence end (a closure's exclusive upper bound) is well defined.
    fn idx(&self, p: u32) -> usize {
        (p as usize).min(self.first_from.len() - 1)
    }

    /// Whether position `p` must pair — the predicate the two single-base
    /// extensions consult.
    pub(crate) fn is_must(&self, p: u32) -> bool {
        self.active && self.first_from[self.idx(p)] == p
    }

    /// Whether the half-open range `[lo, end)` contains no must-pair position.
    /// Vacuously true for an empty range, and for an inert constraint.
    pub(crate) fn free_span(&self, lo: u32, end: u32) -> bool {
        !self.active || end <= lo || self.first_from[self.idx(lo)] >= end
    }

    /// The largest `y` for which `[start, y)` is free of must-pair positions —
    /// the upper bound a flank beginning at `start` puts on the coordinate that
    /// closes it. [`UNBOUNDED`] when nothing bounds it.
    pub(crate) fn max_right(&self, start: u32) -> u32 {
        if !self.active {
            return UNBOUNDED;
        }
        self.first_from[self.idx(start)]
    }

    /// The smallest `x` for which `[x + 1, end)` is free of must-pair positions
    /// — the lower bound a flank ending at `end` puts on the coordinate that
    /// opens it. `0` when nothing bounds it.
    pub(crate) fn min_left(&self, end: u32) -> u32 {
        if !self.active {
            return 0;
        }
        self.last_before[self.idx(end)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference answers by linear scan over the constrained set, so the
    /// precomputed tables are checked against the definition rather than
    /// against themselves.
    fn brute(positions: &[u32], lo: u32, end: u32) -> bool {
        !positions.iter().any(|&p| p >= lo && p < end)
    }

    #[test]
    fn inert_constraint_constrains_nothing() {
        let g = MustPairConstraint::inert();
        for p in 0..40u32 {
            assert!(!g.is_must(p));
            assert!(g.free_span(0, p));
            assert_eq!(g.min_left(p), 0);
            assert_eq!(g.max_right(p), UNBOUNDED);
        }
    }

    /// An empty position set answers every query exactly as the inert constraint
    /// does — the property the byte-identical empty-constraint floor rests on.
    #[test]
    fn an_empty_position_set_is_inert() {
        let g = MustPairConstraint::new(&[], 40);
        for p in 0..40u32 {
            assert!(!g.is_must(p));
            assert!(g.free_span(0, p));
            assert_eq!(g.min_left(p), 0);
            assert_eq!(g.max_right(p), UNBOUNDED);
        }
    }

    #[test]
    fn free_span_matches_a_linear_scan() {
        let positions = [3u32, 4, 11, 25];
        let g = MustPairConstraint::new(&positions, 30);
        for lo in 0..=31u32 {
            for end in 0..=31u32 {
                assert_eq!(
                    g.free_span(lo, end),
                    end <= lo || brute(&positions, lo, end),
                    "free_span({lo}, {end})",
                );
            }
        }
    }

    #[test]
    fn the_two_bounds_are_the_tightest_free_flank_limits() {
        let positions = [3u32, 4, 11, 25];
        let g = MustPairConstraint::new(&positions, 30);
        for start in 0..=30u32 {
            let y = g.max_right(start);
            // Everything at or below the bound is free; the first value past it
            // is not (unless nothing bounds the flank at all).
            assert!(g.free_span(start, y.min(31)));
            if y != UNBOUNDED {
                assert!(!g.free_span(start, y + 1));
            }
        }
        for end in 0..=30u32 {
            let x = g.min_left(end);
            assert!(g.free_span(x + 1, end));
            if x > 0 {
                assert!(!g.free_span(x, end));
            }
        }
    }

    #[test]
    fn is_must_is_exactly_the_constrained_set() {
        let positions = [0u32, 7, 8, 29];
        let g = MustPairConstraint::new(&positions, 30);
        for p in 0..30u32 {
            assert_eq!(g.is_must(p), positions.contains(&p), "is_must({p})");
        }
    }

    /// Positions at or past the sequence end are ignored rather than widening
    /// the tables, so a caller's out-of-range entry cannot bound a real flank.
    #[test]
    fn out_of_range_positions_are_ignored() {
        let g = MustPairConstraint::new(&[50, 99], 30);
        assert!(g.free_span(0, 31));
        assert_eq!(g.max_right(0), UNBOUNDED);
    }
}
