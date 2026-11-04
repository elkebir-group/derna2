//! Per-phase wall-clock attribution for the CKY scan.
//!
//! The scan's cost is not spread evenly across its phases, and *which* phases
//! shrink when the beam narrows is the question any runtime characterization of
//! the approximate family has to answer: a phase whose iteration domain is the
//! surviving table cells gets cheaper as the width drops, while one that sweeps
//! span endpoints costs the same at every width. Wall-clock alone cannot tell
//! the two apart, so the scan reports where its own time went.
//!
//! The attribution rides in the run's own record rather than in stderr
//! scrollback, so a sweep carries its breakdown with its results.
//!
//! # Reading a profile
//!
//! Two properties bound what the numbers mean.
//!
//! **It is off unless [`Profile::from_env`] finds `DERNA_PROFILE` set.** The
//! variable is read once, when the scan builds its profile, so the hot-path
//! branch is a predictable load of an already-resolved `bool` rather than an
//! environment lookup. A disabled profile takes no clock reading at all and
//! every phase reads zero.
//!
//! **It attributes correctly only on a single-threaded scan.** The two fat
//! phases fan out across workers when `threads > 1`, and each worker's elapsed
//! time would then be summed against a wall-clock total it does not divide —
//! phase shares would exceed the run. A multi-threaded profile is still
//! emitted, because silently withholding it invites the reader to assume the
//! run was single-threaded, but it carries `single_threaded: false` and its
//! shares are not fractions of the run.
//!
//! Timing costs one clock reading per phase boundary. The per-cell phases
//! ([`Phase::PToExterior`], [`Phase::PToMultibranch`], [`Phase::PToSingle`])
//! take four readings per surviving closed pair, which is the dominant
//! overhead; `elapsed_ns` on the whole scan is recorded alongside so the
//! instrument's own cost can be measured by differencing an enabled run against
//! a disabled one.

use std::time::{Duration, Instant};

/// One timed region of the scan.
///
/// The variants name the phases a cost attribution is quoted in, and their
/// [`Phase::name`] strings are the keys of the emitted record. Grouping is
/// deliberate: the four `Pull*` families are separated because they differ in
/// what they iterate over — the direct family's inner sweep is bounded by the
/// maximum single-branch span, while the skewed and one-sided families sweep
/// the outer 5′ position across the whole prefix — and an attribution that
/// merged them could not say which.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Hairpin seeding: the `(a, b)` span-endpoint sweep that opens the scan.
    SeedHairpins,
    /// Multi-loop closing, `Multi[o] → P[o]`.
    MultiClose,
    /// Every column-prune call site (`P`, `Multi`, `M`, `M2`).
    ///
    /// Routes to the grouped per-open/completion-bounded policy whenever a
    /// per-open limit or a completion bound is active, and to the flat cap
    /// otherwise; the timer sits at the call site and covers both.
    PruneGrouped,
    /// The direct generic internal-loop pull (small and balanced flank shapes).
    PullDirect,
    /// The skewed-large internal-loop pull, via the size-diagonal index.
    PullSkewed,
    /// The large-5′ one-sided pull, via the per-column index.
    PullOneside,
    /// The large-5′ one-sided pull over the `k`-keyed index.
    PullOneside3,
    /// Folding the pull families' buffered candidates into the `P` table.
    ///
    /// Separate from the four families because one buffer carries all of their
    /// writes: charging it to any single family would overstate that family,
    /// and leaving it out would understate the pull as a whole. Its domain is
    /// the buffered candidates, not the span endpoints that produced them.
    PullApply,
    /// Attaching a closed pair to the exterior prefix, `C = C + P`.
    PToExterior,
    /// Opening and bifurcating a multi-branch, `M = P` and `M2 = M + P`.
    PToMultibranch,
    /// Growing a closed pair into its stack / bulge / internal-loop closures.
    PToSingle,
    /// Indexing a finalized `P` column on its size diagonals: each cell's `H`,
    /// appended.
    IndexFinalizedP,
    /// The `M2` beam, including the multi-loop openings it seeds.
    M2Beam,
    /// Closing every active multi-loop opening against this column's frontier.
    MultiOpenClose,
    /// Extending an unpaired base in a multi-loop, `M`.
    MExtend,
    /// Extending an unpaired base in the exterior, `C`.
    CExtend,
    /// The threaded pull's fan-out: the wall of its scoped workers.
    FanPullCompute,
    /// The threaded pull's serial fold of the workers' buffers into `P`.
    FanPullApply,
    /// The threaded push's serial first pass: exterior attach, branch open and
    /// bifurcation, which write columns the fan-out does not.
    FanPushSerial,
    /// The threaded push at a position too small to fan out, run serially.
    FanPushSmall,
    /// The threaded push's fan-out: the wall of its scoped workers.
    FanPushCompute,
    /// The threaded push's serial fold of the workers' buffers.
    FanPushApply,
    /// Snapshotting a finalized `P` column's cells for indexing.
    IndexCompute,
    /// Indexing a finalized `P` column for the one-sided reductions: each
    /// cell's entry, appended to the column and `k` indexes, and the column
    /// sorted.
    IndexOneSide,
    /// The column fill's per-column table of 3′ flank corner scores.
    FillRights,
    /// The column fill's exterior attach and branch open, `C = C + P` and
    /// `M = P`, one write per closed pair.
    FillExterior,
    /// The column fill's `M = M2` copy and the snapshot of the `M2` column.
    M2Copy,
    /// The column fill's `M2` opening units (fanned out above one thread).
    M2Units,
    /// Inserting the `M2` units' openings into the pool and the `Multi` cells.
    M2Apply,
    /// The bifurcation's per-node set-up: the closed pairs it reads and the
    /// slot count.
    BifPrep,
    /// The bifurcation's slot blocks (fanned out above one thread).
    BifUnits,
    /// Writing the bifurcation's best slots into `M2`.
    BifApply,
}

/// Every phase, in the order the emitted record lists them.
pub const PHASES: [Phase; 32] = [
    Phase::SeedHairpins,
    Phase::MultiClose,
    Phase::PruneGrouped,
    Phase::PullDirect,
    Phase::PullSkewed,
    Phase::PullOneside,
    Phase::PullOneside3,
    Phase::PullApply,
    Phase::PToExterior,
    Phase::PToMultibranch,
    Phase::PToSingle,
    Phase::IndexFinalizedP,
    Phase::M2Beam,
    Phase::MultiOpenClose,
    Phase::MExtend,
    Phase::CExtend,
    Phase::FanPullCompute,
    Phase::FanPullApply,
    Phase::FanPushSerial,
    Phase::FanPushSmall,
    Phase::FanPushCompute,
    Phase::FanPushApply,
    Phase::IndexCompute,
    Phase::IndexOneSide,
    Phase::FillRights,
    Phase::FillExterior,
    Phase::M2Copy,
    Phase::M2Units,
    Phase::M2Apply,
    Phase::BifPrep,
    Phase::BifUnits,
    Phase::BifApply,
];

impl Phase {
    /// This phase's key in the emitted record.
    pub fn name(self) -> &'static str {
        match self {
            Phase::SeedHairpins => "seed_hairpins",
            Phase::MultiClose => "multi_close",
            Phase::PruneGrouped => "prune_grouped",
            Phase::PullDirect => "pull_direct",
            Phase::PullSkewed => "pull_skewed",
            Phase::PullOneside => "pull_oneside",
            Phase::PullOneside3 => "pull_oneside3",
            Phase::PullApply => "pull_apply",
            Phase::PToExterior => "p_to_exterior",
            Phase::PToMultibranch => "p_to_multibranch",
            Phase::PToSingle => "p_to_single",
            Phase::IndexFinalizedP => "index_finalized_p",
            Phase::M2Beam => "m2_beam",
            Phase::MultiOpenClose => "multi_open_close",
            Phase::MExtend => "m_extend",
            Phase::CExtend => "c_extend",
            Phase::FanPullCompute => "fan_pull_compute",
            Phase::FanPullApply => "fan_pull_apply",
            Phase::FanPushSerial => "fan_push_serial",
            Phase::FanPushSmall => "fan_push_small",
            Phase::FanPushCompute => "fan_push_compute",
            Phase::FanPushApply => "fan_push_apply",
            Phase::IndexCompute => "index_compute",
            Phase::IndexOneSide => "index_oneside",
            Phase::FillRights => "fill_rights",
            Phase::FillExterior => "fill_exterior",
            Phase::M2Copy => "m2_copy",
            Phase::M2Units => "m2_units",
            Phase::M2Apply => "m2_apply",
            Phase::BifPrep => "bif_prep",
            Phase::BifUnits => "bif_units",
            Phase::BifApply => "bif_apply",
        }
    }

    /// Whether this phase's iteration domain is over span endpoints or open-node
    /// identities rather than over already-pruned table cells.
    ///
    /// This is the structural half of the question a width study asks of a
    /// phase: a domain of surviving cells shrinks when the beam narrows, a
    /// domain of span endpoints does not. It is a property of the source, read
    /// off the enumeration each phase runs, and it is reported so a reader need
    /// not re-derive it from the call graph. It is *not* on its own a claim that
    /// the domain can be pruned — that additionally requires the enumeration to
    /// compose with the interior-loop reductions, which no timer can decide.
    pub fn sweeps_span_endpoints(self) -> bool {
        matches!(
            self,
            Phase::SeedHairpins
                | Phase::PullDirect
                | Phase::PullSkewed
                | Phase::PullOneside
                | Phase::PullOneside3
        )
    }
}

/// Accumulated per-phase elapsed time for one scan.
///
/// Construct with [`Profile::from_env`], hand `&mut` to the scan, and read the
/// result with [`Profile::to_json`]. A profile that [`Profile::is_on`] reports
/// `false` costs a branch per phase boundary and nothing else.
pub struct Profile {
    on: bool,
    threads: usize,
    started: Option<Instant>,
    elapsed_ns: u64,
    phase_ns: [u64; PHASES.len()],
    phase_calls: [u64; PHASES.len()],
    fan: [FanTally; FAN_SITES],
    /// Work run inside a fanned-out site by its caller, beside the workers'
    /// units (the finalized-column indexing), summed. Already inside the
    /// site's wall, so kept out of the phase attribution.
    overlapped_ns: u64,
    overlapped_calls: u64,
}

/// One of the scan's fanned-out sites.
#[derive(Clone, Copy)]
pub enum FanSite {
    /// The generic internal-loop pull, `single_close_pull`, or under the
    /// column fill its `P`-column units.
    Pull,
    /// The single-branch push, `p_beam_all`'s second pass.
    Push,
    /// The column fill's `M2` opening units.
    M2,
    /// The column fill's bifurcation slot blocks.
    Bif,
}

const FAN_SITES: usize = 4;

impl FanSite {
    fn index(self) -> usize {
        match self {
            FanSite::Pull => 0,
            FanSite::Push => 1,
            FanSite::M2 => 2,
            FanSite::Bif => 3,
        }
    }
}

/// What one fanned-out site did across the scan, beside its wall phases.
///
/// Busy times are **worker-seconds**, not wall, so they are kept out of the
/// phase attribution. `busy_ns / (workers · compute wall)` bounds the fan-out's
/// efficiency; `busy_max_ns` against `busy_ns / workers` is its imbalance.
#[derive(Clone, Copy, Default)]
struct FanTally {
    /// Positions whose work fanned out.
    fanned: u64,
    /// Workers started, summed over fanned positions.
    workers: u64,
    /// Every worker's own loop time, summed.
    busy_ns: u64,
    /// The slowest worker's loop time at each fanned position, summed.
    busy_max_ns: u64,
    /// The longest single chunk at each fanned position, summed: a floor under
    /// the fan-out's wall that no ordering of the chunks can go below.
    chunk_max_ns: u64,
    /// Worker time in the site's heavy family (the pull's direct closures).
    heavy_ns: u64,
    /// Candidates the workers buffered for the serial fold.
    candidates: u64,
}

impl FanTally {
    fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "fanned": self.fanned,
            "workers": self.workers,
            "busy_ns": self.busy_ns,
            "busy_max_ns": self.busy_max_ns,
            "chunk_max_ns": self.chunk_max_ns,
            "heavy_ns": self.heavy_ns,
            "candidates": self.candidates,
        })
    }
}

impl Profile {
    /// A profile enabled iff `DERNA_PROFILE` is present in the environment.
    ///
    /// The lookup happens here and nowhere else, so the scan's hot path never
    /// touches the environment.
    pub fn from_env(threads: usize) -> Self {
        Self::new(std::env::var_os("DERNA_PROFILE").is_some(), threads)
    }

    /// A profile with the gate set explicitly, for tests and callers that do not
    /// want to depend on the ambient environment.
    pub fn new(on: bool, threads: usize) -> Self {
        Self {
            on,
            threads,
            started: None,
            elapsed_ns: 0,
            phase_ns: [0; PHASES.len()],
            phase_calls: [0; PHASES.len()],
            fan: [FanTally::default(); FAN_SITES],
            overlapped_ns: 0,
            overlapped_calls: 0,
        }
    }

    /// Whether this profile is recording.
    pub fn is_on(&self) -> bool {
        self.on
    }

    /// Start the whole-scan clock. A no-op when off.
    pub fn begin(&mut self) {
        if self.on {
            self.started = Some(Instant::now());
        }
    }

    /// Stop the whole-scan clock. A no-op when off.
    pub fn end(&mut self) {
        if let Some(t0) = self.started.take() {
            self.elapsed_ns = nanos(t0.elapsed());
        }
    }

    /// A clock reading, or `None` when off.
    ///
    /// Pair with [`Profile::record`]; the `Option` is what keeps a disabled
    /// profile from reading the clock at all.
    pub fn mark(&self) -> Option<Instant> {
        self.on.then(Instant::now)
    }

    /// Charge the time since `mark` to `phase`, counting one call.
    pub fn record(&mut self, phase: Phase, mark: Option<Instant>) {
        let Some(t0) = mark else {
            return;
        };
        self.add(phase, t0.elapsed());
    }

    /// Charge an already-measured `span` to `phase`, counting one call.
    ///
    /// The direct form, for a run of adjacent phases timed from a single
    /// sequence of clock readings rather than one reading per phase.
    pub fn add(&mut self, phase: Phase, span: Duration) {
        let i = index_of(phase);
        self.phase_ns[i] += nanos(span);
        self.phase_calls[i] += 1;
    }

    /// Record one fanned-out position at `site`: each worker's busy time, the
    /// longest single chunk any worker ran, the time spent in the site's heavy
    /// family, and the candidates the workers buffered. A no-op when off.
    pub fn record_fan(
        &mut self,
        site: FanSite,
        busy: &[Duration],
        chunk_max: Duration,
        heavy: Duration,
        candidates: usize,
    ) {
        if !self.on {
            return;
        }
        let tally = &mut self.fan[site.index()];
        tally.fanned += 1;
        tally.workers += busy.len() as u64;
        tally.busy_ns += busy.iter().map(|&d| nanos(d)).sum::<u64>();
        tally.busy_max_ns += busy.iter().map(|&d| nanos(d)).max().unwrap_or(0);
        tally.chunk_max_ns += nanos(chunk_max);
        tally.heavy_ns += nanos(heavy);
        tally.candidates += candidates as u64;
    }

    /// Charge `span` of work the caller ran inside a fanned-out site. A no-op
    /// when off.
    pub fn add_overlapped(&mut self, span: Duration) {
        if self.on {
            self.overlapped_ns += nanos(span);
            self.overlapped_calls += 1;
        }
    }

    /// Total elapsed nanoseconds across every timed phase.
    fn attributed_ns(&self) -> u64 {
        self.phase_ns.iter().sum()
    }

    /// The profile as a JSON object, or [`serde_json::Value::Null`] when off.
    ///
    /// `elapsed_ns` is the whole scan; `attributed_ns` is what the phases
    /// account for, and the difference is time the phase list does not cover —
    /// reported rather than silently folded into a phase, since an attribution
    /// that sums to less than the run is a fact about the phase list.
    pub fn to_json(&self) -> serde_json::Value {
        if !self.on {
            return serde_json::Value::Null;
        }
        serde_json::json!({
            "single_threaded": self.threads <= 1,
            "threads": self.threads,
            "elapsed_ns": self.elapsed_ns,
            "attributed_ns": self.attributed_ns(),
            "unattributed_ns": self.elapsed_ns.saturating_sub(self.attributed_ns()),
            "phases": self.phase_records(),
            "fan": {
                "pull": self.fan[FanSite::Pull.index()].to_json(),
                "push": self.fan[FanSite::Push.index()].to_json(),
                "m2": self.fan[FanSite::M2.index()].to_json(),
                "bif": self.fan[FanSite::Bif.index()].to_json(),
            },
            "overlapped": {
                "ns": self.overlapped_ns,
                "calls": self.overlapped_calls,
            },
        })
    }

    /// One record per phase: its elapsed time, its call count, and whether its
    /// iteration domain sweeps span endpoints.
    fn phase_records(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for phase in PHASES {
            let i = index_of(phase);
            map.insert(
                phase.name().to_string(),
                serde_json::json!({
                    "ns": self.phase_ns[i],
                    "calls": self.phase_calls[i],
                    "sweeps_span_endpoints": phase.sweeps_span_endpoints(),
                }),
            );
        }
        serde_json::Value::Object(map)
    }
}

/// A phase's slot in the accumulator arrays.
fn index_of(phase: Phase) -> usize {
    PHASES
        .iter()
        .position(|&p| p == phase)
        .expect("every phase is in PHASES")
}

/// Whole nanoseconds of a duration, saturating rather than wrapping.
fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_profile_records_nothing_and_serializes_to_null() {
        let mut p = Profile::new(false, 1);
        p.begin();
        assert!(p.mark().is_none());
        p.record(Phase::PToSingle, p.mark());
        p.end();
        assert_eq!(p.attributed_ns(), 0);
        assert_eq!(p.to_json(), serde_json::Value::Null);
    }

    #[test]
    fn enabled_profile_attributes_a_span_to_its_phase() {
        let mut p = Profile::new(true, 1);
        p.begin();
        p.add(Phase::SeedHairpins, Duration::from_millis(3));
        p.end();
        let v = p.to_json();
        assert_eq!(v["phases"]["seed_hairpins"]["ns"], 3_000_000);
        assert_eq!(v["phases"]["seed_hairpins"]["calls"], 1);
        assert_eq!(v["single_threaded"], true);
    }

    #[test]
    fn every_phase_has_a_distinct_name_and_slot() {
        let mut names: Vec<&str> = PHASES.iter().map(|p| p.name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before);
        for (i, &phase) in PHASES.iter().enumerate() {
            assert_eq!(index_of(phase), i);
        }
    }

    #[test]
    fn a_fan_tally_counts_worker_time_outside_the_attribution() {
        let mut off = Profile::new(false, 8);
        let ms = Duration::from_millis;
        off.record_fan(FanSite::Pull, &[ms(5)], ms(5), ms(0), 3);
        assert_eq!(off.to_json(), serde_json::Value::Null);

        let mut p = Profile::new(true, 8);
        let busy = [ms(2), ms(6)];
        p.record_fan(FanSite::Push, &busy, ms(3), ms(1), 10);
        p.record_fan(FanSite::Push, &busy, ms(3), ms(1), 5);
        let v = p.to_json();
        let push = &v["fan"]["push"];
        assert_eq!(push["fanned"], 2);
        assert_eq!(push["workers"], 4);
        assert_eq!(push["busy_ns"], 16_000_000);
        assert_eq!(push["busy_max_ns"], 12_000_000);
        assert_eq!(push["chunk_max_ns"], 6_000_000);
        assert_eq!(push["heavy_ns"], 2_000_000);
        assert_eq!(push["candidates"], 15);
        assert_eq!(v["fan"]["pull"]["fanned"], 0);
        assert_eq!(v["attributed_ns"], 0, "worker-seconds are not wall");
    }

    #[test]
    fn a_multi_threaded_profile_says_so() {
        let p = Profile::new(true, 8);
        assert_eq!(p.to_json()["single_threaded"], false);
        assert_eq!(p.to_json()["threads"], 8);
    }

    #[test]
    fn the_span_endpoint_phases_are_the_seeding_and_pull_families() {
        let sweeping: Vec<&str> = PHASES
            .iter()
            .filter(|p| p.sweeps_span_endpoints())
            .map(|p| p.name())
            .collect();
        assert_eq!(
            sweeping,
            vec![
                "seed_hairpins",
                "pull_direct",
                "pull_skewed",
                "pull_oneside",
                "pull_oneside3"
            ]
        );
        // The fold of what those sweeps produced is not itself a sweep.
        assert!(!Phase::PullApply.sweeps_span_endpoints());
    }
}
