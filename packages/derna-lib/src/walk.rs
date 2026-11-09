//! The front walk over a codon lattice: one driver for the plain lattice and
//! for its motif-free product.
//!
//! The plain lattice is the motif-free lattice of an empty motif set, so the
//! walk has one control flow, and a lattice enters it as a [`Kernel`]: how it
//! folds an end, a grid pass, a probe and a gap, and how it reports a walk.
//! The driver owns what does not depend on the lattice:
//!
//! - one work queue ([`crate::task_queue`]) for the ends, the grid's passes,
//!   the probes and the gap folds, with no barrier between them;
//! - the hull sweep's bookkeeping: a probe either finds a vertex strictly
//!   between its pair, which splits the pair in two, or certifies the pair
//!   adjacent, which makes it a gap;
//! - the budget that stops a walk short, and the boxes it then leaves open,
//!   with the rest of any gap it solved part way;
//! - resuming from such a walk, and the checkpoints that hand a running walk
//!   out after every settled pair, in the order the walk reached them;
//! - the sweep's seeds, when the grid's own optima start it.
//!
//! What a kernel does differently is a matter of its folds and of which of
//! these the driver turns on for it ([`Policy`]); a walk's points are each
//! kernel's to find, and the driver only decides which pairs are folded,
//! when, and what is reported.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::hash::FxHashMap;
use crate::task_queue::{self, Shares};
use crate::{rss_kb, task_clock, BeamError, TaskKind, TaskSpan, WalkBudget};

/// A probe's priority in the walk's queue, unless its kernel orders probes:
/// above every fold, so the sweep that finds the gaps runs first, first in
/// first out.
pub(crate) const PROBE: (u8, u64) = (1, 0);

/// A grid pass's priority: above every probe, so the grid is taken first and
/// probes fill the workers its last passes leave idle.
pub(crate) const GRID: (u8, u64) = (2, 0);

/// The credit's and the ends' priority: above the grid, so they start in its
/// first round and every pair is released as early as the grid allows.
pub(crate) const SETUP: (u8, u64) = (3, 0);

/// The priority of a gap that moved on under the per-gap time cap
/// ([`Drive::gap_secs`]): below every other task, first in first out, so it
/// is taken up again once nothing else is left.
pub(crate) const MOVED_ON: (u8, u64) = (0, 0);

/// The class of a resumed box that carries a rest, under the per-gap time
/// cap: after every box without one, by its probe's weight within the class.
const RESTED: u8 = 0;

/// A walk's unit of work.
pub(crate) enum Task<K: Kernel> {
    /// Fold at the tie weight of the pair `(a, b)`, `depth` refinements from
    /// the first pair.
    Probe {
        a: K::Vertex,
        b: K::Vertex,
        depth: usize,
        /// When it was queued, in seconds from the walk's start.
        queued: f64,
        /// The watched bracket its seed pair refines inside, if the kernel
        /// watches any.
        bracket: Option<usize>,
    },
    /// Solve a certified gap.
    Fold(Box<K::Job>),
    /// Take the exact pass at a grid weight.
    Grid(f64),
    /// Build the credit the gap folds read.
    Credit,
    /// Fold at an end's weight: the stable end when `stable`.
    End { lambda: f64, stable: bool },
    /// The front of a hull with one vertex.
    Lone(K::Vertex),
}

/// The walk's queue, as a running task sees it.
pub(crate) type WalkQueue<K> = task_queue::Queue<Task<K>>;

/// What a probe found.
pub(crate) enum Probed<K: Kernel> {
    /// A hull vertex strictly between the pair, which splits it in two.
    Split(K::Vertex, K::Note),
    /// The pair is a gap, which the kernel queued for its fold; `None` when
    /// it was queued without a probe's scan.
    Queued(Option<K::Note>),
    /// The pair is a gap, settled in the probe's own task; `None` when no
    /// probe fold ran for it.
    Settled(Settled<K>, Option<K::Note>),
}

/// What became of a gap's fold.
pub(crate) enum Settled<K: Kernel> {
    /// Its whole front.
    Gap(K::Gap),
    /// Its front up to a rest, where a later walk continues it.
    Rest(K::Rest, Vec<K::Point>),
    /// Its front up to a rest, left there by the per-gap time cap: the walk
    /// queues the gap again behind every other task.
    MovedOn(K::Rest, Vec<K::Point>),
    /// Nothing: the budget ran out before any range of it was solved.
    Open,
}

/// Which of the driver's behaviours a kernel takes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Policy {
    /// How a task's workers are counted.
    pub shares: Shares,
    /// The most tasks run at once.
    pub max_running: usize,
    /// Whether a task the budget skips is handed out as a checkpoint.
    pub checkpoint_skips: bool,
    /// Whether the ends are folded one after the other, the stable first,
    /// rather than side by side.
    pub ends_in_turn: bool,
}

/// What a walk starts from, beyond its queue's first tasks.
pub(crate) struct Start<K: Kernel> {
    /// Hull vertices known before the queue starts.
    pub vertices: Vec<K::Vertex>,
    /// The ends known before the queue starts, the stable first.
    pub ends: [Option<K::Vertex>; 2],
    /// Pairs to settle from the start, each probed, or folded alone when its
    /// two points are one.
    pub pairs: Vec<(K::Vertex, K::Vertex)>,
    /// Tasks queued at the start besides the pairs': the credit, the ends,
    /// the grid's passes.
    pub tasks: Vec<((u8, u64), Task<K>)>,
    /// The sweep's seeds, when the grid's optima start it.
    pub seeds: Option<Seeds<K>>,
    /// Gaps an earlier walk solved part way, by their ends' energies.
    pub rests: FxHashMap<(i32, i32), K::Rest>,
    /// How many times each box's gap moved on under the per-gap time cap in
    /// earlier walks, by its ends' energies: a resumed gap's first visit is
    /// the one after these, so a doubling cap ([`visit_cap`]) carries over.
    pub moved: FxHashMap<(i32, i32), u32>,
}

/// What may stop a walk short, and what did.
pub(crate) struct Budget {
    deadline: Option<std::time::Instant>,
    max_rss_gb: Option<f64>,
    /// Gap folds that solved their range so far, and how many may.
    pub folds: Option<(AtomicUsize, usize)>,
    /// Stop once the boxes not settled span at most this share of the energy
    /// between the two ends.
    stop_open_share: Option<f64>,
    sticky: bool,
    /// 0: running; else the reason ([`Self::reason`]).
    stopped: AtomicU8,
}

const TIME: u8 = 1;
const MEMORY: u8 = 2;
const COVERAGE: u8 = 3;
const FOLDS: u8 = 4;

impl Budget {
    pub(crate) fn new(
        budget: WalkBudget,
        start: std::time::Instant,
        gap_folds: Option<usize>,
        stop_open_share: Option<f64>,
        sticky: bool,
    ) -> Self {
        Self {
            deadline: budget
                .max_secs
                .map(|secs| start + std::time::Duration::from_secs_f64(secs)),
            max_rss_gb: budget.max_rss_gb,
            folds: gap_folds.map(|n| (AtomicUsize::new(0), n)),
            stop_open_share,
            sticky,
            stopped: AtomicU8::new(0),
        }
    }

    /// The deadline, if the walk has one.
    pub(crate) fn deadline(&self) -> Option<std::time::Instant> {
        self.deadline
    }

    fn record(&self, why: u8) {
        if self.sticky {
            let _ = self
                .stopped
                .compare_exchange(0, why, Ordering::Relaxed, Ordering::Relaxed);
        } else if self.stopped.load(Ordering::Relaxed) == 0 {
            self.stopped.store(why, Ordering::Relaxed);
        }
    }

    /// Whether the budget is spent, recording why the first time; read
    /// before a pair or a gap is started.
    pub(crate) fn spent(&self) -> bool {
        if self.sticky && self.stopped.load(Ordering::Relaxed) != 0 {
            return true;
        }
        // Read before anything else, since this runs once per task and a
        // walk without a budget is the common one: the memory check reads
        // `/proc/self/status`, which a task should not pay for to learn
        // there is no cap.
        if self.deadline.is_none() && self.max_rss_gb.is_none() && self.folds.is_none() {
            return false;
        }
        let why = if self.timed_out() {
            TIME
        } else if self.folds_spent() {
            FOLDS
        } else if self.memory_over() {
            MEMORY
        } else {
            return false;
        };
        if self.sticky {
            self.stopped.store(why, Ordering::Relaxed);
        } else {
            self.record(why);
        }
        true
    }

    fn timed_out(&self) -> bool {
        self.deadline.is_some_and(|d| std::time::Instant::now() > d)
    }

    /// `max_rss_gb` is a **GiB**, matching every other reader of the figure;
    /// `rss_kb` is KiB, so the comparison is two binary steps up.
    fn memory_over(&self) -> bool {
        self.max_rss_gb
            .is_some_and(|gb| rss_kb().is_some_and(|kb| kb as f64 > gb * 1024.0 * 1024.0))
    }

    /// Counts one more gap fold that solved its range.
    pub(crate) fn fold_solved(&self) {
        if let Some((run, _)) = &self.folds {
            run.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether the count of gap folds that solved their range is spent.
    pub(crate) fn folds_spent(&self) -> bool {
        self.folds
            .as_ref()
            .is_some_and(|(run, n)| run.load(Ordering::Relaxed) >= *n)
    }

    /// The deadline as a fold reads it, recorded as the walk's stop.
    pub(crate) fn time_spent(&self) -> bool {
        let spent = self.timed_out();
        if spent {
            self.record(TIME);
        }
        spent
    }

    /// The memory budget as a fold reads it, recorded as the walk's stop.
    pub(crate) fn memory_spent(&self) -> bool {
        let spent = self.memory_over();
        if spent {
            self.record(MEMORY);
        }
        spent
    }

    /// Record that the deadline would pass inside the next fold.
    pub(crate) fn stop_for_time(&self) {
        self.stopped.store(TIME, Ordering::Relaxed);
    }

    /// Whether a walk is to stop by its time, memory or fold budget at all.
    pub(crate) fn has_deadline(&self) -> bool {
        self.deadline.is_some()
    }

    pub(crate) fn has_memory(&self) -> bool {
        self.max_rss_gb.is_some()
    }

    /// Why the walk stopped, given whether it left anything open: a walk
    /// that left nothing open was not stopped, though an idle worker can
    /// find the budget spent while the last task finishes. A gap the fold
    /// count stopped does not stop the walk itself.
    fn reason(&self, open: bool) -> Option<&'static str> {
        if !open {
            return None;
        }
        match self.stopped.load(Ordering::Relaxed) {
            0 if self.folds_spent() => Some("folds"),
            0 => None,
            TIME => Some("time"),
            MEMORY => Some("memory"),
            COVERAGE => Some("coverage"),
            _ => Some("folds"),
        }
    }
}

/// A lattice the walk runs over.
pub(crate) trait Kernel: Sync + Sized {
    /// A hull point in the kernel's own units.
    type Vertex: Copy + Send + Sync;
    /// A certified gap waiting for its fold.
    type Job: Send;
    /// A solved gap.
    type Gap: Send;
    /// Where a gap stopped part way continues.
    type Rest: Copy + Send + Sync;
    /// A front point a gap stopped part way found.
    type Point: Send + Clone;
    /// What a probe took, beside what it found.
    type Note: Copy + Send;
    /// The walk's counts of what it ran.
    type Stats: Send + Clone;
    /// The walk as reported, at its end and at each checkpoint.
    type Out;

    /// The vertex's energy, in the fold's integer units.
    fn energy(v: Self::Vertex) -> i32;
    /// The weight at which `a` and `b` score alike, if they are ordered.
    fn tie(a: Self::Vertex, b: Self::Vertex) -> Option<f64>;
    /// The driver's behaviours this kernel takes.
    fn policy(&self) -> Policy;

    /// A probe's priority in the queue.
    fn probe_priority(&self, _a: Self::Vertex, _b: Self::Vertex) -> (u8, u64) {
        PROBE
    }
    /// The rest a pair of adjacent seeds is settled from in place of its
    /// probe, solved up to nowhere: the kernel then folds the pair's whole
    /// box at once. `None` probes the pair.
    fn cell_rest(&self, _a: Self::Vertex, _b: Self::Vertex) -> Option<Self::Rest> {
        None
    }
    /// A lone vertex's priority in the queue.
    fn lone_priority(&self) -> (u8, u64) {
        PROBE
    }
    /// The watched bracket a pair refines inside, if any.
    fn tag(&self, _a: Self::Vertex, _b: Self::Vertex) -> Option<usize> {
        None
    }
    /// A probe was queued in `bracket`, at the root of its seed pair or not.
    fn probe_queued(&self, _bracket: Option<usize>, _root: bool) {}
    /// A probe in `bracket` finished; the kernel may queue what it decides
    /// from the hull's `vertices` so far.
    fn probe_done(
        &self,
        _bracket: Option<usize>,
        _queue: &WalkQueue<Self>,
        _vertices: &dyn Fn() -> Vec<Self::Vertex>,
    ) {
    }
    /// Whether a probe of `(a, b)` is not started at all, since the budget
    /// would run out inside it; recorded on `budget` when so.
    fn probe_would_overrun(
        &self,
        _pair: (Self::Vertex, Self::Vertex),
        _rest: bool,
        _budget: &Budget,
    ) -> bool {
        false
    }
    /// Work for a worker that finds nothing waiting while other tasks run;
    /// `false` when there is none, or when `budget` is spent.
    fn idle(&self, _budget: &Budget) -> bool {
        false
    }
    /// The weight the adapted end is folded at, when the stable end queues it.
    fn adapted_weight(&self) -> f64 {
        f64::NAN
    }
    /// What a probe task did, for its span: `note` is `None` for a pair
    /// handed to its fold without a probe's scan.
    fn kind(note: Option<Self::Note>) -> TaskKind;

    /// Build the credit.
    fn credit(&self, workers: usize, queue: &WalkQueue<Self>) -> Result<(), BeamError>;
    /// Fold an end.
    fn end(&self, lambda: f64, stable: bool, workers: usize) -> Result<Self::Vertex, BeamError>;
    /// Take the pass at a grid weight, with its optimum when the walk's seeds
    /// wait for it.
    fn grid(
        &self,
        lambda: f64,
        seed: bool,
        workers: usize,
        queue: &WalkQueue<Self>,
    ) -> Result<Option<Self::Vertex>, BeamError>;
    /// Probe the pair `(a, b)`, continuing from `rest` when an earlier walk
    /// solved its gap part way; `cap` is the seconds its gap's fold may take
    /// before it moves on ([`Settled::MovedOn`]), if any, with which visit to
    /// the gap this is, the first being 1.
    #[allow(clippy::too_many_arguments)]
    fn probe(
        &self,
        pair: (Self::Vertex, Self::Vertex),
        rest: (Option<Self::Rest>, Option<GapCap>),
        bracket: Option<usize>,
        workers: usize,
        queue: &WalkQueue<Self>,
        budget: &Budget,
        vertices: &dyn Fn() -> Vec<Self::Vertex>,
    ) -> Result<Probed<Self>, BeamError>;
    /// Solve a certified gap; also the most workers it ran on.
    fn fold(
        &self,
        job: Self::Job,
        workers: usize,
        queue: &WalkQueue<Self>,
        budget: &Budget,
        vertices: &dyn Fn() -> Vec<Self::Vertex>,
    ) -> Result<(Settled<Self>, usize), BeamError>;
    /// The pair a certified gap's job settles, its tie weight, and when it
    /// was queued.
    fn job_pair(job: &Self::Job) -> (Self::Vertex, Self::Vertex, f64, f64);
    /// Solve the gap of a lone vertex.
    fn lone(&self, v: Self::Vertex, budget: &Budget) -> Result<Settled<Self>, BeamError>;

    /// Record a probe's note in the walk's counts.
    fn note(stats: &mut Self::Stats, note: Self::Note);
    /// Record a gap settled `at_s` seconds into the walk.
    fn settled_gap(stats: &mut Self::Stats, gap: &mut Self::Gap, at_s: f64);

    /// The walk so far, for a checkpoint: `open` holds every pair not
    /// settled.
    fn snapshot(&self, walk: &Walk<Self>, open: Vec<(Self::Vertex, Self::Vertex)>) -> Self::Out;
    /// The walk at its end.
    fn finish(self, walk: Walk<Self>, end: End, threads: usize) -> Result<Self::Out, BeamError>;
}

/// What the driver hands a kernel's [`Kernel::finish`] besides the walk.
pub(crate) struct End {
    /// Every task the queue ran, in the order each finished.
    pub timeline: Vec<TaskSpan>,
    /// What stopped the walk short, if anything did.
    pub stopped_by: Option<&'static str>,
    /// Wall seconds from the queue's start to its end.
    pub wall_s: f64,
}

/// What a walk has found so far.
pub(crate) struct Walk<K: Kernel> {
    pub vertices: Vec<K::Vertex>,
    /// Each solved gap, with its more stable end's energy.
    pub gaps: Vec<(i32, K::Gap)>,
    /// Every pair handed out and not yet settled, the more stable point
    /// first: a probe queued or running, a gap waiting for its fold, and a
    /// pair a budget skipped. What is left here when the queue ends is what
    /// the walk did not settle.
    pub outstanding: Vec<(K::Vertex, K::Vertex)>,
    /// The gaps solved part way, by their ends' energies.
    pub rests: FxHashMap<(i32, i32), K::Rest>,
    /// The points of the gaps solved part way.
    pub rest_points: Vec<K::Point>,
    /// Gap folds that moved on under the per-gap time cap.
    pub moved_on: usize,
    /// How many times each pair's gap moved on, by its ends' energies.
    moved: FxHashMap<(i32, i32), u32>,
    pub stats: K::Stats,
    /// The sweep's depth: the most refinements any pair took from the first.
    pub depth: usize,
    /// Seconds from the walk's start to the end of the latest probe.
    pub last_probe: Option<f64>,
}

impl<K: Kernel> Walk<K> {
    /// Record `pair` as handed out.
    fn queued(&mut self, pair: (K::Vertex, K::Vertex)) {
        self.outstanding.push(pair);
    }

    /// Record `pair` as settled: its points are on the front, or it was
    /// refined into pairs of its own.
    fn settled(&mut self, pair: (K::Vertex, K::Vertex)) {
        let energies = (K::energy(pair.0), K::energy(pair.1));
        if let Some(i) = self
            .outstanding
            .iter()
            .position(|&(a, b)| (K::energy(a), K::energy(b)) == energies)
        {
            self.outstanding.swap_remove(i);
        }
    }

    /// A design is a vertex once, whichever fold found it first.
    fn add_vertex(&mut self, v: K::Vertex) {
        if !self.vertices.iter().any(|&u| K::energy(u) == K::energy(v)) {
            self.vertices.push(v);
        }
    }

    /// The rest of the gap between `a` and `b`, if one is held.
    pub(crate) fn rest_of(&self, a: K::Vertex, b: K::Vertex) -> Option<K::Rest> {
        self.rests.get(&(K::energy(a), K::energy(b))).copied()
    }

    /// How many times the gap between `a` and `b` moved on under the per-gap
    /// time cap, counting earlier walks this one resumed.
    pub(crate) fn moves_of(&self, a: K::Vertex, b: K::Vertex) -> u32 {
        self.moved
            .get(&(K::energy(a), K::energy(b)))
            .copied()
            .unwrap_or(0)
    }

    /// The energy the walk has not settled, as a share of `span`: a box
    /// solved part way is open only above where it is solved.
    fn open_span(&self, solved_to: impl Fn(&K::Rest) -> i32) -> f64 {
        self.outstanding
            .iter()
            .map(|&(a, b)| {
                let (lo, hi) = (K::energy(a), K::energy(b));
                let from = self.rest_of(a, b).map_or(lo, |r| solved_to(&r).max(lo));
                f64::from(hi - from)
            })
            .sum()
    }
}

/// The sweep's seeds from the pass grid: the exact optimum at each grid
/// weight inside the walk's range, between its two end vertices.
///
/// The optimum at a weight lies on the hull, and optima move from the stable
/// end to the adapted one as the weight falls, so consecutive slots are an
/// ordered pair of hull points with every hull vertex between them optimal at
/// a weight between theirs. Refining every such pair to adjacency, as the
/// sweep does from its two ends, finds every vertex.
pub(crate) struct Seeds<K: Kernel> {
    /// By descending weight: the stable end, the grid weights inside the
    /// range, the adapted end; the optimum once that slot's scan is done.
    pub slots: Vec<(f64, Option<K::Vertex>)>,
    /// Whether the pair from slot `i` to `i + 1` was handed out.
    handed: Vec<bool>,
    /// The grid weights immediately beyond each end, above the stable one and below
    /// the adapted one, until their passes are pooled: the pair at that end
    /// waits for them, since its tie weight may fall between the end and the
    /// range's last grid weight, and a probe without its bracket runs exact.
    gates: [Option<f64>; 2],
    /// Whether every pair waits for the credit: a probe that certifies its
    /// pair and every gap's fold read it.
    credit_pending: bool,
    /// The weight whose grid pass is the adapted end, if one is.
    pub end_pass: Option<f64>,
}

impl<K: Kernel> Seeds<K> {
    /// Slots between the two ends, each end's weight with its vertex when it
    /// is already folded; an end left `None` is filled by [`Self::end`].
    pub(crate) fn new(
        stable: (f64, Option<K::Vertex>),
        adapted: (f64, Option<K::Vertex>),
        weights: &[f64],
        credit_pending: bool,
        end_pass: Option<f64>,
    ) -> Self {
        let mut slots = vec![stable];
        let inside = weights.iter().filter(|&&w| adapted.0 < w && w < stable.0);
        slots.extend(inside.rev().map(|&w| (w, None)));
        slots.push(adapted);
        let handed = vec![false; slots.len() - 1];
        let above = weights.iter().copied().filter(|&w| w >= stable.0);
        let below = weights.iter().copied().filter(|&w| w <= adapted.0);
        let gates = [above.reduce(f64::min), below.reduce(f64::max)];
        Self {
            slots,
            handed,
            gates,
            credit_pending,
            end_pass,
        }
    }

    /// The grid's slots, between the two ends.
    fn inner(&self) -> &[(f64, Option<K::Vertex>)] {
        &self.slots[1..self.slots.len() - 1]
    }

    /// Whether `lambda` is a grid slot still waiting for its optimum.
    pub(crate) fn waits_for(&self, lambda: f64) -> bool {
        self.inner()
            .iter()
            .any(|&(w, p)| w == lambda && p.is_none())
    }

    /// Record that the pass at `lambda` is pooled, with its scan's optimum
    /// when `lambda` is a grid slot.
    fn pooled(&mut self, lambda: f64, optimum: Option<K::Vertex>) {
        for gate in &mut self.gates {
            if *gate == Some(lambda) {
                *gate = None;
            }
        }
        let last = self.slots.len() - 1;
        if let Some(slot) = self.slots[1..last].iter_mut().find(|(w, _)| *w == lambda) {
            slot.1 = optimum;
        }
    }

    /// Record an end's vertex: the stable end's when `stable`, else the
    /// adapted end's.
    fn end(&mut self, stable: bool, vertex: K::Vertex) {
        let last = self.slots.len() - 1;
        self.slots[if stable { 0 } else { last }].1 = Some(vertex);
    }

    /// Whether both ends' vertices are known.
    pub(crate) fn ends_known(&self) -> bool {
        self.slots[0].1.is_some() && self.slots[self.slots.len() - 1].1.is_some()
    }

    /// Record that the credit stands.
    fn credit_built(&mut self) {
        self.credit_pending = false;
    }

    /// Every pair of consecutive slots both known and not yet handed out
    /// whose optima differ: the pairs the sweep may now refine. None while
    /// the credit is pending.
    pub(crate) fn ready(&mut self) -> Vec<(K::Vertex, K::Vertex)> {
        let mut pairs = Vec::new();
        if self.credit_pending {
            return pairs;
        }
        let last = self.handed.len() - 1;
        for i in 0..self.handed.len() {
            let gated =
                (i == 0 && self.gates[0].is_some()) || (i == last && self.gates[1].is_some());
            if let (false, false, Some(a), Some(b)) =
                (gated, self.handed[i], self.slots[i].1, self.slots[i + 1].1)
            {
                self.handed[i] = true;
                // Two slots with one optimum leave nothing between them; any
                // other pair is strictly ordered, or the grid's optima would
                // not lie on one hull.
                if K::energy(a) != K::energy(b) {
                    assert!(
                        K::tie(a, b).is_some(),
                        "the grid's optima at consecutive weights are not ordered"
                    );
                    pairs.push((a, b));
                }
            }
        }
        pairs
    }

    /// The ranges of the hull no pair has been handed out for yet, each
    /// between the nearest two slots whose optima are known, or `None` while
    /// an end is not known.
    fn unhanded(&self) -> Option<Vec<(K::Vertex, K::Vertex)>> {
        let last = self.slots.len() - 1;
        self.slots[0].1?;
        self.slots[last].1?;
        let mut ranges = Vec::new();
        let mut from = 0;
        for (i, &(_, b)) in self.slots.iter().enumerate().skip(1) {
            let Some(b) = b else { continue };
            let a = self.slots[from].1.expect("a known slot");
            let handed = i == from + 1 && self.handed[from];
            if !handed && K::energy(a) < K::energy(b) {
                ranges.push((a, b));
            }
            from = i;
        }
        Some(ranges)
    }
}

/// A callback handed the walk so far.
pub(crate) type Checkpoint<'a, T> = &'a (dyn Fn(&T) + Sync);

/// How a walk runs beyond its kernel and start: its budget, the stops only
/// some walks take, and what to hand the walk so far.
pub(crate) struct Drive<'a, K: Kernel> {
    pub budget: Budget,
    /// Where a rest is solved up to, in the fold's integer units, for the
    /// coverage stop.
    pub solved_to: fn(&K::Rest) -> i32,
    /// Called once both ends of the hull are known, before any pair settles,
    /// and after each pair settles, with the walk so far.
    pub checkpoint: Option<Checkpoint<'a, K::Out>>,
    /// The walk's start, every span is timed from.
    pub t: std::time::Instant,
    /// The seconds a gap's fold may take in one task before it moves on to
    /// the back of the queue, keeping its rest; also queues a resumed box
    /// with a rest behind every box without one. `None`: no cap. A gap that
    /// moved on and is taken up again as the only work left runs without it.
    pub gap_secs: Option<f64>,
    /// Whether a gap's cap doubles each time it moves on ([`visit_cap`]).
    pub gap_double: bool,
}

/// The seconds a gap's fold may take in one task before it moves on, and
/// which visit to the gap that task is, the first being 1.
pub(crate) type GapCap = (f64, u32);

/// The cap on a gap's fold that moved on `moves` times before: `secs`, or
/// with `double` `secs` doubled once per move, so a gap the cap keeps ending
/// gets a longer turn each time it is taken up again.
fn visit_cap(secs: f64, moves: u32, double: bool) -> f64 {
    if double {
        secs * 2f64.powf(f64::from(moves))
    } else {
        secs
    }
}

/// Walk the front with `kernel` on `threads` workers from `start`.
pub(crate) fn run<K: Kernel>(
    kernel: K,
    start: Start<K>,
    drive: Drive<'_, K>,
    initial_stats: K::Stats,
    threads: usize,
) -> Result<K::Out, BeamError> {
    let policy = kernel.policy();
    let Drive {
        budget,
        solved_to,
        checkpoint: hand,
        t,
        gap_secs,
        gap_double,
    } = drive;
    let Start {
        vertices,
        ends,
        pairs,
        tasks,
        seeds,
        rests,
        moved,
    } = start;
    let ends_pending = ends.iter().filter(|e| e.is_none()).count();
    let walk: Mutex<Walk<K>> = Mutex::new(Walk {
        vertices,
        gaps: Vec::new(),
        outstanding: Vec::new(),
        rests,
        rest_points: Vec::new(),
        moved_on: 0,
        moved,
        stats: initial_stats,
        depth: 0,
        last_probe: None,
    });
    let ends = Mutex::new((ends, ends_pending));
    let seeds = seeds.map(Mutex::new);
    let k = &kernel;
    let vertices = || walk.lock().expect("the walk's lock").vertices.clone();

    // One checkpoint at a time, each taken inside its turn, so they are
    // handed over in the order the walk reached them. The seeds' lock is
    // taken before the walk's, as where pairs are handed out.
    let turn = Mutex::new(());
    let checkpoint = || {
        let Some(hand) = hand else {
            return;
        };
        let _turn = turn.lock().expect("the checkpoints' lock");
        let seeds = seeds.as_ref().map(|s| s.lock().expect("the seeds' lock"));
        let Some(unhanded) = seeds.as_ref().map_or(Some(Vec::new()), |s| s.unhanded()) else {
            return;
        };
        let walk = walk.lock().expect("the walk's lock");
        let mut open = unhanded;
        open.extend_from_slice(&walk.outstanding);
        let so_far = k.snapshot(&walk, open);
        drop(walk);
        drop(seeds);
        hand(&so_far);
    };
    let pair_task = |(a, b): (K::Vertex, K::Vertex), queued: f64| {
        if K::energy(a) < K::energy(b) {
            let bracket = k.tag(a, b);
            k.probe_queued(bracket, true);
            let task = Task::Probe {
                a,
                b,
                depth: 1,
                queued,
                bracket,
            };
            (k.probe_priority(a, b), task)
        } else {
            (k.lone_priority(), Task::Lone(a))
        }
    };
    // The pairs the seeds hold ready, recorded as outstanding while the
    // seeds' lock is still held: a checkpoint reads both under it, so a range
    // is either still with the seeds or owed by the walk, never neither.
    let hand_out = |seeds: &mut Seeds<K>| {
        let pairs = seeds.ready();
        let mut walk = walk.lock().expect("the walk's lock");
        for &(a, b) in &pairs {
            walk.queued((a, b));
            if let Some(rest) = k.cell_rest(a, b) {
                walk.rests
                    .entry((K::energy(a), K::energy(b)))
                    .or_insert(rest);
            }
        }
        pairs
    };
    let release = |pairs: Vec<(K::Vertex, K::Vertex)>, queue: &WalkQueue<K>| {
        let queued = t.elapsed().as_secs_f64();
        for pair in pairs {
            let (priority, task) = pair_task(pair, queued);
            queue.push(priority, task);
        }
    };
    let initial: Vec<((u8, u64), Task<K>)> = tasks
        .into_iter()
        .chain(pairs.into_iter().map(|pair| {
            let rested = {
                let mut walk = walk.lock().expect("the walk's lock");
                walk.queued(pair);
                walk.rest_of(pair.0, pair.1).is_some()
            };
            let (priority, task) = pair_task(pair, 0.0);
            match priority {
                (_, weight) if rested && gap_secs.is_some() => ((RESTED, weight), task),
                _ => (priority, task),
            }
        }))
        .collect();

    // An end the queue folded: with seeds, a slot; without, the first pair
    // once both are known.
    let end_found = |stable: bool, v: K::Vertex, queue: &WalkQueue<K>| {
        walk.lock().expect("the walk's lock").add_vertex(v);
        if let Some(seeds) = &seeds {
            let mut seeds = seeds.lock().expect("the seeds' lock");
            let had_ends = seeds.ends_known();
            seeds.end(stable, v);
            let has_ends = seeds.ends_known();
            let adapted_owed = stable && policy.ends_in_turn && !has_ends;
            let pairs = hand_out(&mut seeds);
            drop(seeds);
            release(pairs, queue);
            if adapted_owed {
                queue.push(
                    SETUP,
                    Task::End {
                        lambda: k.adapted_weight(),
                        stable: false,
                    },
                );
            }
            if has_ends && !had_ends {
                checkpoint();
            }
            return;
        }
        let mut ends = ends.lock().expect("the ends' lock");
        ends.0[usize::from(!stable)] = Some(v);
        ends.1 -= 1;
        if stable && policy.ends_in_turn && ends.0[1].is_none() {
            queue.push(
                SETUP,
                Task::End {
                    lambda: k.adapted_weight(),
                    stable: false,
                },
            );
            return;
        }
        if ends.1 == 0 {
            let (Some(a), Some(b)) = (ends.0[0], ends.0[1]) else {
                unreachable!("both ends are known")
            };
            drop(ends);
            let pair = (a, b);
            walk.lock().expect("the walk's lock").queued(pair);
            let (priority, task) = pair_task(pair, t.elapsed().as_secs_f64());
            checkpoint();
            queue.push(priority, task);
        }
    };
    // A probe's or a gap's outcome in the walk's state, after which the
    // coverage stop is read.
    let coverage = |walk: &Walk<K>| {
        let Some(share) = budget.stop_open_share else {
            return;
        };
        let ends = ends.lock().expect("the ends' lock");
        let span = match (ends.0[0], ends.0[1]) {
            (Some(a), Some(b)) => f64::from(K::energy(b) - K::energy(a)),
            _ => return,
        };
        drop(ends);
        if walk.open_span(solved_to) <= share * span.max(1.0) {
            let _ =
                budget
                    .stopped
                    .compare_exchange(0, COVERAGE, Ordering::Relaxed, Ordering::Relaxed);
        }
    };
    let record = |pair: (K::Vertex, K::Vertex), settled: Settled<K>| {
        let mut walk = walk.lock().expect("the walk's lock");
        let key = (K::energy(pair.0), K::energy(pair.1));
        match settled {
            Settled::Gap(mut gap) => {
                walk.rests.remove(&key);
                K::settled_gap(&mut walk.stats, &mut gap, t.elapsed().as_secs_f64());
                walk.gaps.push((key.0, gap));
                walk.settled(pair);
            }
            Settled::Rest(rest, points) => {
                walk.rests.insert(key, rest);
                walk.rest_points.extend(points);
            }
            Settled::MovedOn(rest, points) => {
                walk.rests.insert(key, rest);
                walk.rest_points.extend(points);
                walk.moved_on += 1;
                *walk.moved.entry(key).or_insert(0) += 1;
            }
            Settled::Open => {}
        }
        coverage(&walk);
    };
    let skip = |task: &Task<K>| match *task {
        Task::Probe { a, b, .. } => {
            let rest = walk
                .lock()
                .expect("the walk's lock")
                .rest_of(a, b)
                .is_some();
            budget.spent() || k.probe_would_overrun((a, b), rest, &budget)
        }
        Task::Lone(_) | Task::Fold(_) => budget.spent(),
        _ => false,
    };
    let probe = |(a, b): (K::Vertex, K::Vertex),
                 depth: usize,
                 bracket: Option<usize>,
                 workers: usize,
                 queue: &WalkQueue<K>|
     -> Result<TaskKind, BeamError> {
        let (rest, moves) = {
            let walk = walk.lock().expect("the walk's lock");
            (walk.rest_of(a, b), walk.moves_of(a, b))
        };
        // A gap that moved on and is now the only work left runs to its end.
        let cap = gap_secs
            .filter(|_| !(moves > 0 && queue.alone()))
            .map(|secs| (visit_cap(secs, moves, gap_double), moves + 1));
        let probed = k.probe(
            (a, b),
            (rest, cap),
            bracket,
            workers,
            queue,
            &budget,
            &vertices,
        )?;
        let at = t.elapsed().as_secs_f64();
        let seen = |walk: &mut Walk<K>, note: Option<K::Note>| {
            if let Some(note) = note {
                K::note(&mut walk.stats, note);
                walk.depth = walk.depth.max(depth);
                walk.last_probe = Some(at);
            }
        };
        match probed {
            Probed::Split(p, note) => {
                {
                    let mut walk = walk.lock().expect("the walk's lock");
                    walk.vertices.push(p);
                    seen(&mut walk, Some(note));
                    walk.settled((a, b));
                    // The cap's count belongs to the energy range the gap
                    // spans, so each part of a split box keeps it.
                    let moves = walk.moves_of(a, b);
                    for pair in [(a, p), (p, b)] {
                        walk.queued(pair);
                        if moves > 0 {
                            walk.moved
                                .insert((K::energy(pair.0), K::energy(pair.1)), moves);
                        }
                    }
                    coverage(&walk);
                }
                checkpoint();
                for (a, b) in [(a, p), (p, b)] {
                    k.probe_queued(bracket, false);
                    let task = Task::Probe {
                        a,
                        b,
                        depth: depth + 1,
                        queued: t.elapsed().as_secs_f64(),
                        bracket,
                    };
                    queue.push(k.probe_priority(a, b), task);
                }
                Ok(K::kind(Some(note)))
            }
            Probed::Queued(note) => {
                seen(&mut walk.lock().expect("the walk's lock"), note);
                Ok(K::kind(note))
            }
            Probed::Settled(settled, note) => {
                seen(&mut walk.lock().expect("the walk's lock"), note);
                let moved_on = matches!(settled, Settled::MovedOn(..));
                record((a, b), settled);
                checkpoint();
                // Recorded first, so the task queued again reads the rest.
                if moved_on {
                    let task = Task::Probe {
                        a,
                        b,
                        depth,
                        queued: t.elapsed().as_secs_f64(),
                        bracket,
                    };
                    queue.push(MOVED_ON, task);
                }
                Ok(K::kind(note))
            }
        }
    };
    // The walk is handed out as soon as both ends are known, before any pair
    // settles: a walk killed inside its first box then resumes without
    // folding them again. Ends known before the queue starts, or taken from
    // the walk resumed, are known here; ends the queue folds hand it out as
    // the second one lands.
    let known = match &seeds {
        Some(s) => s.lock().expect("the seeds' lock").ends_known(),
        None => ends.lock().expect("the ends' lock").1 == 0,
    };
    if known {
        checkpoint();
    }
    let timeline: Mutex<Vec<TaskSpan>> = Mutex::new(Vec::new());
    let max_running = policy.max_running.min(threads.max(1));
    let idle = || k.idle(&budget);
    task_queue::run_with(
        threads,
        max_running,
        policy.shares,
        initial,
        idle,
        |task, workers, queue| {
            let start_s = t.elapsed().as_secs_f64();
            let cpu = task_clock::TaskClock::start();
            let mut peak_workers = workers;
            // A pair or a gap the budget skips is an open box: its two points
            // are hull points the walk already has, and a later run settles
            // what lies between them. The ends, the credit and the grid are
            // never skipped, since the walk cannot report an open box without
            // them.
            if skip(&task) {
                if policy.checkpoint_skips {
                    checkpoint();
                }
                return Ok(());
            }
            let (kind, lambda, queued_s) = match task {
                Task::Probe {
                    a,
                    b,
                    depth,
                    queued,
                    bracket,
                } => {
                    let kind = probe((a, b), depth, bracket, workers, queue)?;
                    k.probe_done(bracket, queue, &vertices);
                    let lambda = K::tie(a, b).unwrap_or(f64::NAN);
                    (kind, lambda, queued)
                }
                Task::Fold(job) => {
                    let (a, b, lambda, queued) = K::job_pair(&job);
                    let (settled, peak) = k.fold(*job, workers, queue, &budget, &vertices)?;
                    peak_workers = peak;
                    record((a, b), settled);
                    checkpoint();
                    (TaskKind::Fold, lambda, queued)
                }
                Task::Grid(lambda) => {
                    let seed = seeds.as_ref().is_some_and(|s| {
                        let s = s.lock().expect("the seeds' lock");
                        s.end_pass == Some(lambda) || s.waits_for(lambda)
                    });
                    let optimum = k.grid(lambda, seed, workers, queue)?;
                    // The pass is pooled before its seed is recorded, so a
                    // pair handed out here finds both its bracket's passes.
                    if let Some(seeds) = &seeds {
                        let mut seeds = seeds.lock().expect("the seeds' lock");
                        let is_end = seeds.end_pass == Some(lambda);
                        let had_ends = seeds.ends_known();
                        if is_end {
                            seeds.end(false, optimum.expect("an end's optimum"));
                        }
                        let has_ends = seeds.ends_known();
                        seeds.pooled(lambda, if seed { optimum } else { None });
                        if let Some(seed) = optimum.filter(|_| seed) {
                            walk.lock().expect("the walk's lock").add_vertex(seed);
                        }
                        let pairs = hand_out(&mut seeds);
                        drop(seeds);
                        release(pairs, queue);
                        if has_ends && !had_ends {
                            checkpoint();
                        }
                    }
                    (TaskKind::Grid, lambda, 0.0)
                }
                Task::Credit => {
                    k.credit(workers, queue)?;
                    if let Some(seeds) = &seeds {
                        let mut seeds = seeds.lock().expect("the seeds' lock");
                        seeds.credit_built();
                        let pairs = hand_out(&mut seeds);
                        drop(seeds);
                        release(pairs, queue);
                    }
                    (TaskKind::Credit, 1.0, 0.0)
                }
                Task::End { lambda, stable } => {
                    let v = k.end(lambda, stable, workers)?;
                    end_found(stable, v, queue);
                    (TaskKind::End, lambda, 0.0)
                }
                Task::Lone(v) => {
                    let settled = k.lone(v, &budget)?;
                    record((v, v), settled);
                    checkpoint();
                    (TaskKind::Fold, f64::NAN, 0.0)
                }
            };
            let cpu_s = cpu.finish();
            timeline
                .lock()
                .expect("the timeline's lock")
                .push(TaskSpan {
                    kind,
                    lambda,
                    queued_s,
                    start_s,
                    end_s: t.elapsed().as_secs_f64(),
                    workers,
                    peak_workers,
                    cpu_s,
                    rss_kb: task_clock::rss_kb(),
                    peak_rss_kb: task_clock::peak_rss_kb(),
                });
            Ok::<(), BeamError>(())
        },
    )?;
    let walk = walk.into_inner().expect("the walk's lock");
    let stopped_by = budget.reason(!walk.outstanding.is_empty());
    let end = End {
        timeline: timeline.into_inner().expect("the timeline's lock"),
        stopped_by,
        wall_s: t.elapsed().as_secs_f64(),
    };
    kernel.finish(walk, end, threads)
}

#[cfg(test)]
mod tests {
    use super::visit_cap;

    #[test]
    fn a_doubling_cap_doubles_per_move_and_a_fixed_one_does_not() {
        let doubled: Vec<f64> = (0..4).map(|k| visit_cap(60.0, k, true)).collect();
        assert_eq!(doubled, [60.0, 120.0, 240.0, 480.0]);
        let fixed: Vec<f64> = (0..4).map(|k| visit_cap(60.0, k, false)).collect();
        assert_eq!(fixed, [60.0; 4]);
        assert_eq!(visit_cap(0.0, 3, true), 0.0);
    }
}
