//! What the region width loses, and to which of its two losses.
//!
//! The bounded search examines regions a batch at a time: it takes up to
//! `width` entries off the heap, examines them side by side, and makes every
//! decision afterwards in pop order. A batch is filled to at most the width and
//! the fan-out is handed that same width, so **each worker takes exactly one
//! entry** — there is no scheduling choice inside a batch to get wrong, and no
//! work-stealing schedule to gain from.
//!
//! What the arrangement does cost is two other things, and they are not the
//! same problem:
//!
//! - **The barrier.** A batch takes as long as its *slowest* entry, and a
//!   region's solve cost varies with how restricted its lattice is. Every
//!   worker that finishes early waits.
//! - **Thin batches.** The heap does not always hold `width` entries that pass
//!   the stop rule, so some batches leave cores idle for want of work rather
//!   than for want of balance.
//!
//! One ratio separates them. A continuously fed pool would place the batch's
//! total work across all `width` workers and finish in `total / width`; the
//! barrier finishes in `slowest`. So `width · slowest / total` is what the
//! arrangement costs, it is `1` when the batch is full and its entries cost the
//! same, and breaking it down by how many entries were taken says which of the
//! two is responsible. Only if the barrier dominates is a continuously fed pool
//! the thing to build.
//!
//! Recording this changes nothing the walk decides.

/// One batch's examining.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchRow {
    /// Entries actually taken. Fewer than [`Self::width`] means the heap held
    /// nothing more that passed the stop rule.
    pub taken: usize,
    /// The width the walk was asked for — the cores available to this batch.
    pub width: usize,
    /// The slowest entry. With one entry per worker this is the batch's own
    /// critical path, so it is what the barrier costs.
    pub slowest_s: f64,
    /// Every entry's time added up: the work a continuously fed pool would have
    /// had to place across the same cores.
    pub total_s: f64,
    /// The batch's wall, fan-out and join included. Above [`Self::slowest_s`]
    /// by whatever the threading itself costs.
    pub wall_s: f64,
}

impl BatchRow {
    /// What this batch cost against a continuously fed pool of the same width.
    ///
    /// `1.0` is a full batch of equal-cost entries. Above that is the barrier,
    /// the thinness, or both — which is what [`Self::taken`] distinguishes.
    pub fn cost_factor(&self) -> f64 {
        if self.total_s <= 0.0 {
            return 1.0;
        }
        self.width as f64 * self.slowest_s / self.total_s
    }

    /// The cores this batch left idle for want of a queued entry.
    pub fn idle_for_want_of_work(&self) -> usize {
        self.width.saturating_sub(self.taken)
    }
}

/// Every batch a probed walk examined.
#[derive(Debug, Clone, Default)]
pub struct BatchProbeReport {
    pub rows: Vec<BatchRow>,
}

impl BatchProbeReport {
    pub fn push(&mut self, row: BatchRow) {
        self.rows.push(row);
    }

    /// Time the walk spent inside batches, against what a continuously fed pool
    /// of the same width would have spent on the same work.
    ///
    /// Returns `(examined, ideal)` in seconds. Their ratio is the whole of what
    /// the arrangement costs; the walk's own elapsed time above `examined` is
    /// the serial bookkeeping between batches, which is ordered by design.
    pub fn against_a_fed_pool(&self) -> (f64, f64) {
        let examined = self.rows.iter().map(|r| r.wall_s).sum();
        let ideal = self
            .rows
            .iter()
            .map(|r| r.total_s / r.width.max(1) as f64)
            .sum();
        (examined, ideal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(taken: usize, width: usize, slowest: f64, total: f64) -> BatchRow {
        BatchRow {
            taken,
            width,
            slowest_s: slowest,
            total_s: total,
            wall_s: slowest,
        }
    }

    /// A full batch of equal-cost entries is what the width promises, and it
    /// costs exactly nothing against a fed pool.
    #[test]
    fn a_full_batch_of_equal_entries_costs_nothing() {
        assert_eq!(row(4, 4, 1.0, 4.0).cost_factor(), 1.0);
        assert_eq!(row(4, 4, 1.0, 4.0).idle_for_want_of_work(), 0);
    }

    /// One slow entry among fast ones holds every other worker at the barrier,
    /// and the batch is full — so the cost is the barrier and not thinness.
    #[test]
    fn one_slow_entry_holds_a_full_batch() {
        let r = row(4, 4, 4.0, 7.0);
        assert!((r.cost_factor() - 16.0 / 7.0).abs() < 1e-12);
        assert_eq!(r.idle_for_want_of_work(), 0);
    }

    /// A batch the heap could not fill leaves cores idle for want of work, and
    /// says so in a different field from the barrier's.
    #[test]
    fn a_thin_batch_is_idle_for_want_of_work() {
        let r = row(1, 4, 1.0, 1.0);
        assert_eq!(r.cost_factor(), 4.0);
        assert_eq!(r.idle_for_want_of_work(), 3);
    }
}
