//! What happens inside the front fold's fanned-out phases.
//!
//! [`crate::FoldClock`] times each phase on the driver thread, around the
//! whole fan-out and the serial fold of its partials together. That is a
//! correct wall clock at any thread count, and it says nothing about the
//! inside: how much of a phase is the workers' compute and how much the serial
//! apply, how busy the workers were, and whether one chunk was the floor. This
//! tallies those per fanned site, summed over the fold's positions.
//!
//! Worker busy time is worker-seconds, not wall, so it sits outside the
//! phase attribution. `busy / capacity` is a site's efficiency, where
//! `capacity` sums each fanned position's workers times its compute wall;
//! `slowest / mean` is its imbalance.
//!
//! The tally costs a clock read per chunk, so it is off unless the environment
//! sets `DERNA_FAN_DETAIL`. `DERNA_FAN_WORKERS` caps the workers a fan-out
//! starts, so the threaded path can run on one worker; it changes the work's
//! division, never the fold's answer.

use std::time::Duration;

/// A fanned-out site of the front fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanSite {
    /// The internal-loop push, `P → P` at later positions.
    Push,
    /// The bifurcation composes, `M ⊗ P` bound for `M2`.
    Bifurcate,
    /// Seeding the shared multi-loop openings from each `M2` core.
    Seed,
    /// Closing the shared openings against the frontier into `Multi`.
    Close,
}

impl FanSite {
    /// Every site, in report order.
    pub const ALL: [FanSite; 4] = [
        FanSite::Push,
        FanSite::Bifurcate,
        FanSite::Seed,
        FanSite::Close,
    ];

    /// The site's name in a report.
    pub fn name(self) -> &'static str {
        match self {
            FanSite::Push => "push",
            FanSite::Bifurcate => "bifurcate",
            FanSite::Seed => "seed",
            FanSite::Close => "close",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// One site's tally, summed over the positions it ran at.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FanTally {
    /// Positions fanned out across workers.
    pub fanned: u64,
    /// Positions run serially, below the fan-out threshold or on one thread.
    pub serial: u64,
    /// Wall of the fan-outs' parallel sections, the sort of their partials
    /// included.
    pub compute_s: f64,
    /// Wall of the serial fold of the fanned positions' partials.
    pub apply_s: f64,
    /// Wall of the positions run serially, their apply included.
    pub serial_s: f64,
    /// Worker-seconds the workers spent in their loops.
    pub busy_s: f64,
    /// `Σ workers × compute wall` over the fanned positions.
    pub capacity_s: f64,
    /// `Σ` per position of the slowest worker's busy time.
    pub slowest_s: f64,
    /// `Σ` per position of the mean worker's busy time.
    pub mean_s: f64,
    /// `Σ` per position of the longest single item.
    pub item_max_s: f64,
    /// `Σ` per position of the longest single chunk.
    pub chunk_max_s: f64,
    /// Items produced, fanned or not.
    pub items: u64,
    /// Chunks the fanned positions were dealt in.
    pub chunks: u64,
    /// Candidates the site buffered for its apply.
    pub candidates: u64,
    /// Points those candidates carried.
    pub points: u64,
}

impl FanTally {
    /// Count `items` produced and the `(candidates, points)` they buffered.
    pub(crate) fn hold(&mut self, items: usize, (candidates, points): (u64, u64)) {
        self.items += items as u64;
        self.candidates += candidates;
        self.points += points;
    }
}

/// What the workers of one fan-out did, as the fan-out measures it.
#[derive(Debug, Clone, Default)]
pub(crate) struct WorkerTimes {
    /// Each worker's busy time.
    pub busy: Vec<Duration>,
    /// The longest single item any worker ran.
    pub item_max: Duration,
    /// The longest single chunk any worker ran.
    pub chunk_max: Duration,
    /// Chunks dealt.
    pub chunks: usize,
}

/// Every fanned site's tally, and the bifurcation buffer's peak.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FanDetail {
    sites: [FanTally; 4],
    /// The most bifurcation arrival points held at once, at one position.
    pub bifurcation_peak_points: u64,
    /// The most bifurcation arrivals held at once, at one position.
    pub bifurcation_peak_arrivals: u64,
    /// The most bytes of arrival points held at once (points × point size).
    pub bifurcation_peak_bytes: u64,
    /// The push's sharded apply: the serial bucketing of its candidates by
    /// owning worker, the parallel merge's wall, and its workers' busy time
    /// summed and slowest-per-position summed.
    pub sharded_bucket_s: f64,
    pub sharded_merge_s: f64,
    pub sharded_busy_s: f64,
    pub sharded_slowest_s: f64,
}

impl FanDetail {
    /// One site's tally.
    pub fn site(&self, site: FanSite) -> &FanTally {
        &self.sites[site.index()]
    }

    /// Whether anything was tallied.
    pub fn is_empty(&self) -> bool {
        self.sites.iter().all(|s| s.items == 0)
    }

    pub(crate) fn site_mut(&mut self, site: FanSite) -> &mut FanTally {
        &mut self.sites[site.index()]
    }

    /// Record one fanned position's parallel section.
    pub(crate) fn record_fanned(&mut self, site: FanSite, compute: Duration, times: &WorkerTimes) {
        let t = self.site_mut(site);
        let workers = times.busy.len().max(1);
        let busy: Duration = times.busy.iter().sum();
        let slowest = times.busy.iter().max().copied().unwrap_or_default();
        t.fanned += 1;
        t.chunks += times.chunks as u64;
        t.compute_s += compute.as_secs_f64();
        t.busy_s += busy.as_secs_f64();
        t.capacity_s += workers as f64 * compute.as_secs_f64();
        t.slowest_s += slowest.as_secs_f64();
        t.mean_s += busy.as_secs_f64() / workers as f64;
        t.chunk_max_s += times.chunk_max.as_secs_f64();
        t.item_max_s += times.item_max.as_secs_f64();
    }

    /// Record one sharded apply of the push's partials.
    pub(crate) fn record_sharded_apply(
        &mut self,
        bucket: Duration,
        merge: Duration,
        busy: &[Duration],
    ) {
        self.sharded_bucket_s += bucket.as_secs_f64();
        self.sharded_merge_s += merge.as_secs_f64();
        self.sharded_busy_s += busy.iter().sum::<Duration>().as_secs_f64();
        self.sharded_slowest_s += busy.iter().max().copied().unwrap_or_default().as_secs_f64();
    }

    /// Record the peak of the bifurcation buffer at one position.
    pub(crate) fn hold_bifurcations(&mut self, arrivals: u64, points: u64, bytes: u64) {
        self.bifurcation_peak_arrivals = self.bifurcation_peak_arrivals.max(arrivals);
        self.bifurcation_peak_points = self.bifurcation_peak_points.max(points);
        self.bifurcation_peak_bytes = self.bifurcation_peak_bytes.max(bytes);
    }
}

/// Whether the environment asks for the tally.
pub(crate) fn enabled() -> bool {
    std::env::var_os("DERNA_FAN_DETAIL").is_some()
}

/// The cap on a fan-out's workers the environment sets, if any.
pub(crate) fn workers_cap() -> Option<usize> {
    std::env::var("DERNA_FAN_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn a_fanned_position_charges_efficiency_and_imbalance() {
        let mut d = FanDetail::default();
        let times = WorkerTimes {
            busy: vec![ms(4), ms(2)],
            chunk_max: ms(3),
            item_max: ms(1),
            chunks: 5,
        };
        d.record_fanned(FanSite::Push, ms(5), &times);
        d.site_mut(FanSite::Push).hold(40, (7, 70));
        let t = d.site(FanSite::Push);
        assert_eq!((t.fanned, t.items, t.chunks), (1, 40, 5));
        assert_eq!((t.candidates, t.points), (7, 70));
        assert!((t.busy_s - 0.006).abs() < 1e-12);
        assert!((t.capacity_s - 0.010).abs() < 1e-12);
        assert!((t.slowest_s - 0.004).abs() < 1e-12);
        assert!((t.mean_s - 0.003).abs() < 1e-12);
        assert!((t.chunk_max_s - 0.003).abs() < 1e-12);
        assert!((t.item_max_s - 0.001).abs() < 1e-12);
        assert!(d.site(FanSite::Seed).items == 0 && !d.is_empty());
    }

    #[test]
    fn the_bifurcation_peak_keeps_the_largest_position() {
        let mut d = FanDetail::default();
        d.hold_bifurcations(3, 30, 300);
        d.hold_bifurcations(2, 50, 100);
        assert_eq!(
            (
                d.bifurcation_peak_arrivals,
                d.bifurcation_peak_points,
                d.bifurcation_peak_bytes
            ),
            (3, 50, 300)
        );
    }
}
