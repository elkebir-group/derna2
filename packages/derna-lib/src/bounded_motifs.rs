//! The motif-constrained optimum by an outside-bounded threshold search.
//!
//! Every motif-free design is also a design of the unconstrained lattice, so
//! the unconstrained **outside** of a closed pair — the best way to complete a
//! structure around it — is never worse than any motif-free completion. A
//! constrained `P` cell's objective plus that outside is therefore a floor
//! under every motif-free design through the cell. Given a ceiling, the
//! constrained fold can skip every cell whose floor is above it: no design
//! scoring at or below the ceiling goes through one.
//!
//! The ceiling is not known in advance, so the search guesses one and checks.
//! It starts a little above the unconstrained optimum, which bounds the
//! constrained one from below. A fold that returns a design scoring at or
//! below its ceiling has found the optimum: every cell on the optimal
//! derivation passed the bound. Otherwise the margin doubles. A design found
//! above its ceiling is still feasible, so its score caps every later
//! ceiling, and a fold at that cap certifies. The unconstrained scan and its
//! outside pass are paid once.
//!
//! **Where it is offered.** The bound is admissible only against the lattice
//! the unconstrained scan searches with the same constraints, which the composite
//! path builds for both; and a pruned fold is only a minimum on the exact
//! lattice, so a finite beam takes the ordinary fold.
//!
//! **Equal objective, not equal bytes.** Skipping cells changes which of
//! several tied designs the fold reaches first.
//!
//! **Certified-approximate mode.** With a feasible incumbent scoring `T`, a
//! fold may prune against `T - ε` instead of `T`. A design it returns at or
//! below `T - ε` is still the optimum, by the argument above. Otherwise no
//! design scores below `T - ε` — every one that did would have survived the
//! bound and been returned — so the optimum is at least `T - ε`, and the best
//! design in hand (the fold's, or the incumbent) is within `ε` of it. The
//! solve reports that proven gap. `ε = 0` is the exact search, unchanged.

use crate::constrained::{self, ChainCai, ConstrainedFold, OutsideBound, PairTable, ZeroChord};
use crate::dfa::Dfa;
use crate::must_pair::MustPairConstraint;
use crate::outside_scan::OutsideScan;
use crate::product::ProductDfa;
use crate::region::PairConstraint;
use crate::{
    design_composite_routed, BeamError, CodonUsage, CompositeSpec, Design, Params, ScanConfig,
};

/// What a bounded search did.
#[derive(Debug, Clone, Default)]
pub struct BoundReport {
    /// Whether the constrained fold ran bounded at all. `false` when the run
    /// never reached the constrained folder (an inert set) or took the ordinary
    /// fold (a finite beam).
    pub bounded: bool,
    /// The unconstrained optimum the ceilings start from, on the folder's BRD
    /// scale.
    pub free_optimum: f64,
    /// The ceiling of each bounded fold, in order. The last one certified.
    pub ceilings: Vec<f64>,
    /// Closed pairs each bounded fold ruled out, in the same order.
    pub pruned: Vec<usize>,
    /// Seconds: the unconstrained scan, its outside pass, and every bounded
    /// fold together.
    pub seconds: (f64, f64, f64),
}

/// A design reached by the bounded search, with what it took.
#[derive(Debug, Clone)]
pub struct BoundedDesign {
    /// The design. Its BRD objective is the full constrained optimum's.
    pub design: Design,
    pub report: BoundReport,
}

/// Design under `spec` with the constrained fold run as the bounded search
/// (see the module docs). The answer has the ordinary fold's objective.
pub fn design_motifs_bounded<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    spec: &CompositeSpec<'_, S>,
    config: ScanConfig,
) -> Result<BoundedDesign, BeamError> {
    let mut report = BoundReport::default();
    let (design, _) = design_composite_routed(
        params,
        protein,
        usage,
        lambda,
        spec,
        config,
        Some(&mut report),
    )?;
    Ok(BoundedDesign { design, report })
}

/// A margin beyond any design's objective, after which the search folds
/// unbounded rather than keep doubling.
const MAX_MARGIN: f64 = 1e9;

/// How far an inherited ceiling may sit above where the constraint's observed
/// cost puts the optimum, in first margins, before a solve searches up from
/// the unconstrained optimum instead of folding under the ceiling.
const GUESS_SPAN: f64 = 16.0;

/// The walk's first rounds, whose gaps are wide: each probes a chord across a
/// long stretch of a curved front, so its inherited ceiling sits far above the
/// optimum wherever the front bends, whatever the constraint cost elsewhere.
/// Their probes guess once the ceiling is [`WIDE_GUESS_SPAN`] first margins
/// above the unconstrained bound, whatever the earlier folds found.
const WIDE_ROUNDS: usize = 4;

/// [`GUESS_SPAN`] for the probes of the [`WIDE_ROUNDS`].
const WIDE_GUESS_SPAN: f64 = 2.0;

/// How far above the best design in hand a motif-aware anchor folds, BRD
/// units. Its table has an entry only for the pairs its pruned fold built, and
/// a probe's chord reads nothing where either table lacks one, so a ceiling at
/// the optimum leaves the chord all but empty; much above this the anchor costs
/// more and prunes no further (P0DTC2 and P31645 at 200 residues).
const MOTIF_AWARE_MARGIN: f64 = 30.0;

/// Threads each motif-aware anchor's outside pass fans out over; its pruned
/// fold stays serial. Two halve the time its chord is missing from the probes.
const MOTIF_AWARE_THREADS: usize = 2;

/// Motif-aware anchors at or below this λ start in the walk's second round,
/// the rest once the [`WIDE_ROUNDS`] are done. Near λ = 0 the vertices in hand
/// score close to the optimum early (the open chain's end is nearly exact
/// there), so an early ceiling costs little; higher up the first rounds' hull
/// sits far above it and an anchor folded under that ceiling costs far more.
const MOTIF_AWARE_EARLY: f64 = 0.02;

/// The constrained optimum's distance above the unconstrained bound `free`,
/// in first margins: what the constraint cost at one solved λ, on the scale
/// [`GUESS_SPAN`] is measured in.
fn constraint_cost(optimum: f64, free: f64) -> f64 {
    (optimum - free) / first_margin(free)
}

/// The best whole design through each closed pair on the unconstrained
/// lattice at the scan's λ: the pair's inside (`scan.p`) plus its outside
/// (`outside`), indexed and keyed as both are. The table
/// [`constrained::PairBound::through`] holds.
pub(crate) fn through_table(
    scan: &crate::tables::Tables,
    outside: &[crate::hash::FxHashMap<crate::state::PairKey, f64>],
    lambda: f64,
) -> Vec<crate::hash::FxHashMap<crate::state::PairKey, f64>> {
    outside
        .iter()
        .zip(&scan.p)
        .map(|(out, inside)| {
            out.iter()
                .filter_map(|(key, &o)| inside.get(key).map(|s| (*key, s.brd(lambda) + o)))
                .collect()
        })
        .collect()
}

/// [`PairTable`] of an unconstrained scan at `lambda` and its outside pass:
/// each pair's outside and its inside plus outside, one map per column.
fn anchor_table(
    scan: &crate::tables::Tables,
    outside: Vec<crate::hash::FxHashMap<crate::state::PairKey, f64>>,
    lambda: f64,
) -> Vec<constrained::PairColumn> {
    constrained::pair_table(outside, |o, key| scan.p[o].get(key).map(|s| s.brd(lambda)))
}

/// What one anchor λ contributes to a sweep or walk: the unconstrained
/// outside of every pair and the best design through it, the unconstrained
/// optimum, and the seconds it took.
struct Anchor {
    table: Vec<constrained::PairColumn>,
    free: Option<f64>,
    seconds: f64,
}

/// One anchor: an unconstrained scan at `lambda` and its outside pass.
fn anchor_at(
    params: &Params,
    graph: &Dfa,
    lambda: f64,
    config: ScanConfig,
    region: &PairConstraint,
    must: &MustPairConstraint,
) -> Anchor {
    let started = std::time::Instant::now();
    let scan = crate::scan::scan_values_with_constraints(
        params,
        graph,
        lambda,
        config,
        region.clone(),
        must.clone(),
    );
    let free = scan.c[graph.ix(graph.end())].map(|s| s.brd(lambda));
    let costs = OutsideScan::cost_to_go(
        params,
        graph,
        &scan,
        region,
        config.single_max_len,
        config.threads,
        false,
    );
    Anchor {
        table: anchor_table(&scan, costs.p, lambda),
        free,
        seconds: started.elapsed().as_secs_f64(),
    }
}

/// How many jobs to run side by side when the caller does not say, for a walk
/// or sweep over `anchors` anchors: one per anchor, or one per physical core
/// this process may run on if that is fewer. A walk runs its anchors, and then
/// each round's gap probes, this many at a time. The cores follow the
/// process's CPU affinity, so a run pinned to a core set stays inside it.
///
/// Physical, not logical: two scans on the hyperthreads of one core run
/// slower than one scan there (11 workers on 8 two-thread cores lost to 6).
/// Each anchor in flight holds a whole unconstrained scan's tables, so a
/// single round costs memory: on 12 single-thread cores it finished a
/// 350-residue walk 1.22x faster than two rounds, for ~65% more peak memory.
/// [`memory_capped_workers`] lowers the count where memory is short. The walk
/// or sweep is the same at any count.
#[must_use]
pub fn default_workers(anchors: usize) -> usize {
    let cores = physical_cores().unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    anchors.min(cores).max(1)
}

/// The physical cores among the CPUs this process may run on: distinct
/// (package, core) pairs in Linux's CPU topology over `Cpus_allowed_list`.
/// `None` where either cannot be read.
fn physical_cores() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?
        .trim();
    let mut cores = std::collections::BTreeSet::new();
    for part in list.split(',') {
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?),
            None => {
                let c = part.parse::<usize>().ok()?;
                (c, c)
            }
        };
        for cpu in lo..=hi {
            let read = |field: &str| {
                std::fs::read_to_string(format!(
                    "/sys/devices/system/cpu/cpu{cpu}/topology/{field}"
                ))
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
            };
            cores.insert((read("physical_package_id")?, read("core_id")?));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

/// Peak bytes per anchor in flight, per squared lattice edge count: the
/// unconstrained scan's tables and pull indexes, alive until its outside pass
/// is done. Those tables are keyed by pairs of lattice edges, so a protein
/// whose codons branch more needs more at the same length. Measured with
/// values-only anchor scans, whose multi-branch columns keep no back links,
/// on an allocator that returns freed pages at once (jemalloc with
/// `dirty_decay_ms:0,muzzy_decay_ms:0`): one
/// that keeps them, such as the system allocator, peaks about a fifth higher.
const ANCHOR_BYTES_PER_EDGE2: f64 = 216.75;

/// Peak bytes of the rest of a walk, per squared lattice edge count: every
/// anchor's kept outside and through tables, and the folds running beside
/// the anchors. Measured as [`ANCHOR_BYTES_PER_EDGE2`] is; the system
/// allocator adds about two fifths.
const WALK_BYTES_PER_EDGE2: f64 = 210.0;

/// Peak bytes of one probe fold in flight, per squared lattice edge count: its
/// constrained fold's tables and its worker's span memo. A probe round reads
/// the memory left beside the kept anchor tables, so they are not counted here.
/// The bound prunes most cells, so a fold holds far less than an anchor: one
/// low-λ fold of a 400-residue protein added under 10 MB of live memory, well
/// within this allowance.
const FOLD_BYTES_PER_EDGE2: f64 = 32.0;

/// The share of the memory available a walk plans to fill: the rest absorbs
/// the per-edge constants' error and whatever else the process allocates.
const MEMORY_BUDGET: f64 = 0.9;

/// [`default_workers`] for a walk or sweep over `anchors` anchors on the
/// lattice `graph`, lowered until the anchors in flight fit in the memory this
/// process can still take: the system's available memory, or less under a
/// control-group limit. Where neither can be read, [`default_workers`]
/// unchanged. The walk or sweep is the same at any count; only its peak memory
/// and wall time move.
#[must_use]
pub fn memory_capped_workers(anchors: usize, graph: &Dfa) -> usize {
    let workers = default_workers(anchors);
    available_memory().map_or(workers, |bytes| {
        workers_within(workers, bytes, lattice_edges(graph))
    })
}

/// How many edges the lattice has.
fn lattice_edges(graph: &Dfa) -> usize {
    (0..=graph.seq_len)
        .flat_map(|pos| graph.nodes_at(pos))
        .map(|&node| graph.right(node).len())
        .sum()
}

/// The most of `workers` whose peak fits in [`MEMORY_BUDGET`] of `available`
/// bytes on a lattice of `edges` edges, by the per-edge constants above; at
/// least one, which is the serial walk's peak and the least any walk needs.
fn workers_within(workers: usize, available: u64, edges: usize) -> usize {
    let area = (edges as f64).powi(2);
    let room = MEMORY_BUDGET * available as f64 - WALK_BYTES_PER_EDGE2 * area;
    let fit = (room / (ANCHOR_BYTES_PER_EDGE2 * area)).floor();
    if fit >= 1.0 {
        workers.min(fit as usize)
    } else {
        1
    }
}

/// The workers for a probe round once every anchor is in: as many folds as fit
/// in [`MEMORY_BUDGET`] of the memory this process can still take, at most one
/// per physical core and at least `least`, which is what the anchors ran on.
/// `least` where the memory cannot be read.
fn probe_workers(least: usize, graph: &Dfa) -> usize {
    let cores = default_workers(usize::MAX);
    available_memory().map_or(least, |bytes| {
        folds_within(cores, bytes, lattice_edges(graph)).max(least)
    })
}

/// The most of `workers` probe folds that fit in [`MEMORY_BUDGET`] of
/// `available` bytes on a lattice of `edges` edges, by
/// [`FOLD_BYTES_PER_EDGE2`]; possibly none.
fn folds_within(workers: usize, available: u64, edges: usize) -> usize {
    let area = (edges as f64).powi(2);
    let fit = (MEMORY_BUDGET * available as f64 / (FOLD_BYTES_PER_EDGE2 * area)).floor();
    if fit >= 1.0 {
        workers.min(fit as usize)
    } else {
        0
    }
}

/// Bytes this process can still allocate: `MemAvailable` from
/// `/proc/meminfo`, lowered to the headroom under any control-group (v2)
/// memory limit on the process's group or an ancestor's. `None` off Linux or
/// where `/proc/meminfo` cannot be read.
fn available_memory() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let system = meminfo.lines().find_map(|line| {
        let kb = line
            .strip_prefix("MemAvailable:")?
            .trim()
            .strip_suffix("kB")?;
        kb.trim().parse::<u64>().ok().map(|kb| kb * 1024)
    })?;
    Some(system.min(cgroup_headroom().unwrap_or(u64::MAX)))
}

/// The least `memory.max - memory.current` over this process's control group
/// (v2) and its ancestors, or `None` where no limit is set or none can be read.
fn cgroup_headroom() -> Option<u64> {
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = groups.lines().find_map(|line| line.strip_prefix("0::"))?;
    let mut dir = std::path::PathBuf::from("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let read = |dir: &std::path::Path, file: &str| -> Option<u64> {
        std::fs::read_to_string(dir.join(file))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let mut least: Option<u64> = None;
    loop {
        if let (Some(max), Some(current)) = (read(&dir, "memory.max"), read(&dir, "memory.current"))
        {
            let headroom = max.saturating_sub(current);
            least = Some(least.map_or(headroom, |l| l.min(headroom)));
        }
        if dir.as_os_str() == "/sys/fs/cgroup" || !dir.pop() {
            break;
        }
    }
    least
}

/// Every anchor in `lambdas`, in order; `workers = 0` takes
/// [`memory_capped_workers`]. With `workers > 1` the anchors are
/// computed side by side, each single-threaded: they are independent, and one
/// scan per worker uses the cores far better than threads inside each scan in
/// turn. Each anchor's tables are the same either way (they hold objective
/// values, which a threaded scan reproduces exactly); its `seconds` are then
/// its own, overlapping the others'.
fn anchors_at(
    params: &Params,
    graph: &Dfa,
    lambdas: &[f64],
    config: ScanConfig,
    workers: usize,
    region: &PairConstraint,
    must: &MustPairConstraint,
) -> Vec<Anchor> {
    let workers = match workers {
        0 => memory_capped_workers(lambdas.len(), graph),
        given => given,
    };
    if workers <= 1 || lambdas.len() < 2 {
        return lambdas
            .iter()
            .map(|&l| anchor_at(params, graph, l, config, region, must))
            .collect();
    }
    let single = ScanConfig {
        threads: 1,
        ..config
    };
    let mut slots = vec![(); workers.min(lambdas.len())];
    side_by_side(&mut slots, lambdas.len(), |i, ()| {
        anchor_at(params, graph, lambdas[i], single, region, must)
    })
}

/// `job(i, state)` for every `i < n`, one worker per entry of `states`, each
/// holding its state for every job it runs; the results in index order. The
/// workers take jobs off a shared counter, so which state a job sees varies
/// from run to run: a state may only be a cache.
fn side_by_side<S: Send, T: Send>(
    states: &mut [S],
    n: usize,
    job: impl Fn(usize, &mut S) -> T + Sync,
) -> Vec<T> {
    if states.len() <= 1 || n < 2 {
        let Some(state) = states.first_mut() else {
            return Vec::new();
        };
        return (0..n).map(|i| job(i, state)).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::Mutex::new(Vec::with_capacity(n));
    std::thread::scope(|scope| {
        for state in states.iter_mut().take(n) {
            let (next, done, job) = (&next, &done, &job);
            scope.spawn(move || loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let out = job(i, state);
                done.lock().expect("a worker panicked").push((i, out));
            });
        }
    });
    let mut done = done.into_inner().expect("a worker panicked");
    done.sort_unstable_by_key(|&(i, _)| i);
    done.into_iter().map(|(_, out)| out).collect()
}

/// A walk's anchors, computed in the background while the walk runs: each
/// fold waits only for the two anchors bracketing its λ, so the walk's first
/// folds overlap the last anchors. The anchors a walk needs first — its two
/// ends, then the rest from the stable end down — are dealt first. Which
/// anchors are ready when changes no fold, so the walk is the same either way.
struct AnchorFeed {
    cells: Vec<std::sync::OnceLock<Anchor>>,
    /// Indices into `cells` in the order the workers take them.
    order: Vec<usize>,
    next: std::sync::atomic::AtomicUsize,
    /// Set when a worker panicked, so no fold waits forever.
    failed: std::sync::atomic::AtomicBool,
    lock: std::sync::Mutex<()>,
    ready: std::sync::Condvar,
}

/// How `n` anchors split over `workers` workers: how many are dealt in full
/// rounds of `workers`, and the threads each later one takes, an equal share of
/// the workers.
fn tail_split(n: usize, workers: usize) -> (usize, usize) {
    let tail = n % workers.max(1);
    if tail == 0 {
        (n, 1)
    } else {
        (n - tail, workers / tail)
    }
}

/// Leaves a motif-aware anchor's slot empty-handed if its job unwinds, so no
/// probe waits on it forever.
struct FillOnPanic<'a, T>(&'a std::sync::OnceLock<(Option<T>, f64)>);

impl<T> Drop for FillOnPanic<'_, T> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let _ = self.0.set((None, 0.0));
        }
    }
}

/// Wakes every waiting fold if an anchor worker unwinds.
struct WakeOnPanic<'a>(&'a AnchorFeed);

impl Drop for WakeOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0
                .failed
                .store(true, std::sync::atomic::Ordering::SeqCst);
            self.0.wake();
        }
    }
}

impl AnchorFeed {
    fn new(n: usize) -> Self {
        let order = match n {
            0 => Vec::new(),
            _ => std::iter::once(n - 1)
                .chain((n > 1).then_some(0))
                .chain((1..n.saturating_sub(1)).rev())
                .collect(),
        };
        Self {
            cells: (0..n).map(|_| std::sync::OnceLock::new()).collect(),
            order,
            next: std::sync::atomic::AtomicUsize::new(0),
            failed: std::sync::atomic::AtomicBool::new(false),
            lock: std::sync::Mutex::new(()),
            ready: std::sync::Condvar::new(),
        }
    }

    fn wake(&self) {
        let _held = self
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ready.notify_all();
    }

    /// Compute the anchors at `lambdas` until none is left, one of `workers`
    /// workers sharing the feed. A lone worker scans at `config`. Several scan
    /// single-threaded (see [`anchors_at`]), except the anchors dealt after the
    /// last full round of workers: they would leave the other workers idle, so
    /// each takes an equal share of the workers' threads. A threaded scan holds
    /// the same values, so the anchors are the same either way.
    #[allow(clippy::too_many_arguments)]
    fn work(
        &self,
        params: &Params,
        graph: &Dfa,
        lambdas: &[f64],
        config: ScanConfig,
        (workers, share): (usize, usize),
        region: &PairConstraint,
        must: &MustPairConstraint,
    ) {
        let _wake = WakeOnPanic(self);
        let (full, threads) = tail_split(self.order.len(), workers);
        loop {
            let k = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&i) = self.order.get(k) else {
                break;
            };
            let config = match workers {
                0 | 1 => config,
                _ => ScanConfig {
                    threads: share * if k < full { 1 } else { threads },
                    ..config
                },
            };
            let anchor = anchor_at(params, graph, lambdas[i], config, region, must);
            assert!(
                self.cells[i].set(anchor).is_ok(),
                "an anchor computed twice"
            );
            self.wake();
        }
    }

    /// The anchor at index `i`, once some worker has computed it.
    fn get(&self, i: usize) -> &Anchor {
        let mut held = self
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(anchor) = self.cells[i].get() {
                return anchor;
            }
            assert!(
                !self.failed.load(std::sync::atomic::Ordering::SeqCst),
                "an anchor worker panicked"
            );
            held = self
                .ready
                .wait(held)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Whether every anchor is computed.
    fn complete(&self) -> bool {
        self.cells.iter().all(|c| c.get().is_some())
    }

    /// Every anchor's own seconds, in λ order, once all are computed.
    fn seconds(&self) -> Vec<f64> {
        (0..self.cells.len()).map(|i| self.get(i).seconds).collect()
    }
}

/// The chord of two anchors' unconstrained optima at weight `weight` on the
/// first: a lower bound on the unconstrained optimum between them, which is
/// concave in λ. Either one alone where the other is missing.
fn free_chord(near: Option<f64>, far: Option<f64>, weight: f64) -> Option<f64> {
    match (near, far) {
        (Some(a), Some(b)) => Some(weight * a + (1.0 - weight) * b),
        (a, b) => a.or(b),
    }
}

/// The first ceiling's margin above the unconstrained optimum: a small
/// fraction of its magnitude, so the first fold prunes hard and a clean
/// optimum certifies at once.
fn first_margin(free: f64) -> f64 {
    (free.abs() * 0.005).max(1.0)
}

/// The bounded search over one product, after the composite path has built
/// the lattice and the constraints.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold(
    params: &Params,
    graph: &Dfa,
    product: &ProductDfa,
    lambda: f64,
    region: PairConstraint,
    must: MustPairConstraint,
    config: ScanConfig,
    report: &mut BoundReport,
) -> Option<ConstrainedFold> {
    let started = std::time::Instant::now();
    let tables = crate::scan::scan_with_constraints(
        params,
        graph,
        lambda,
        config,
        region.clone(),
        must.clone(),
    );
    let free = tables.c[graph.ix(graph.end())]?.brd(lambda);
    let scan_s = started.elapsed().as_secs_f64();
    let costs = OutsideScan::cost_to_go(
        params,
        graph,
        &tables,
        &region,
        config.single_max_len,
        config.threads,
        false,
    );
    let table = anchor_table(&tables, costs.p, lambda);
    let outside_s = started.elapsed().as_secs_f64() - scan_s;
    report.free_optimum = free;
    let score = |f: &ConstrainedFold| lambda * f64::from(f.mfe.0) - (1.0 - lambda) * f.cai * 100.0;
    let mut margin = first_margin(free);
    let mut ceiling = free + margin;
    // The best feasible score seen: a ceiling the optimum cannot exceed.
    let mut cap = f64::INFINITY;
    let mut spans = None;
    let found = loop {
        report.ceilings.push(ceiling);
        let bound = OutsideBound {
            outside: &table,
            chord: None,
            through: true,
            zero: &[],
            motif_free: None,
            aware: None,
            ceiling,
        };
        let (fold, counts) = constrained::fold_bounded(
            product,
            params,
            lambda,
            region.clone(),
            must.clone(),
            config.threads,
            bound,
            &mut spans,
        );
        report.pruned.push(counts.pruned);
        if let Some(f) = &fold {
            if score(f) <= ceiling + constrained::BOUND_SLACK {
                break fold;
            }
            cap = cap.min(score(f));
        } else if ceiling.is_infinite() {
            // Nothing feasible at any ceiling: the constraints leave no motif-free
            // design, and the unbounded fold has said so.
            break None;
        }
        // A design found above the ceiling caps the next one, but only caps
        // it: a fold this heavily pruned can return a design far worse than
        // the optimum, and a ceiling there would prune almost nothing.
        margin *= 2.0;
        ceiling = if margin > MAX_MARGIN {
            f64::INFINITY
        } else {
            free + margin
        }
        .min(cap);
    };
    report.seconds = (
        scan_s,
        outside_s,
        started.elapsed().as_secs_f64() - scan_s - outside_s,
    );
    found
}

/// One λ of a swept solve: the design and what the bounded fold took.
#[derive(Debug, Clone)]
pub struct SweptSolve {
    pub lambda: f64,
    /// Free energy in dekacalories and the summed log codon weights.
    pub mfe: i32,
    pub cai: f64,
    pub sequence: String,
    pub structure: String,
    /// The ceiling the fold ran under, and whether it certified at the first
    /// one. A sweep that certifies first time pays one fold per λ.
    pub ceiling: f64,
    pub folds: usize,
    /// How many of `folds` ran under a guessed ceiling, searching up from the
    /// unconstrained optimum (see [`solve_one`]), the last one included when
    /// it certified; the rest ran under the incumbent or unbounded.
    pub guesses: usize,
    pub pruned: usize,
    /// Closures the pair floor skipped before scoring them, over all folds.
    pub cut: usize,
    /// How many chords to the motif-free outside at λ = 0 the folds pruned on
    /// besides the anchors' own bound ([`constrained::ZeroChord`]).
    pub zero_chords: usize,
    pub fold_seconds: f64,
    /// The anchor λ values the bound's chord was taken between.
    pub anchors: (f64, f64),
    /// Which feasible design the ceiling came from, and how far above the
    /// optimum it sat, in BRD units. A sweep whose ceilings are poor shows it
    /// here before it shows it in the clock.
    pub ceiling_from: CeilingSource,
    pub ceiling_gap: f64,
    /// How far above the optimum this design is proven to be at most, in BRD
    /// units: `0` where the fold certified, otherwise at most the solve's `ε`
    /// plus the bound's float slack ([`crate::constrained::BOUND_SLACK`]).
    pub gap: f64,
    /// Where the answer is an earlier design rather than this fold's, its
    /// index among the incumbents the solve was given. The caller holds that
    /// design's spelling; `sequence` and `structure` are then empty.
    pub incumbent: Option<usize>,
}

/// Which feasible design a solve's ceiling came from.
///
/// Ordered by strength. A rescored neighbour is the optimum itself wherever
/// adjacent λ share a design, and a poor bound where the front is steep; a
/// beam design does not vary that way; the unconstrained optimum plus a
/// doubling margin is the weakest and only used when no feasible design is in
/// hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeilingSource {
    Beam,
    Neighbour,
    /// The best-adapted motif-free spelling left unpaired, which beats every
    /// neighbour near λ = 0.
    OpenChain,
    FreeOptimum,
}

/// A λ sweep that shares its outside passes.
#[derive(Debug, Clone)]
pub struct SweptSweep {
    pub solves: Vec<SweptSolve>,
    pub anchor_lambdas: Vec<f64>,
    /// Seconds per anchor, each one unconstrained scan plus its outside pass.
    pub anchor_seconds: Vec<f64>,
}

/// `anchor_lambdas`, sorted, widened to span every λ in `lambdas`.
///
/// An anchor exactly at a λ gives that λ its tightest available bound, so
/// widening by the extreme λ values costs one outside pass each and loses
/// nothing. Without it, a λ beyond the end of the grid would extrapolate the
/// chord, which is not a floor. With `chained` (a motif-free spelling exists)
/// the low end is not widened: below the lowest anchor the chords to the
/// motif-free outside at λ = 0 bound the fold ([`below_anchors`]).
fn span_lambdas(anchor_lambdas: &[f64], lambdas: &[f64], chained: bool) -> Vec<f64> {
    let mut anchors: Vec<f64> = anchor_lambdas.to_vec();
    anchors.sort_by(|a, b| a.partial_cmp(b).expect("finite λ"));
    let lo = lambdas.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = lambdas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo.is_finite() && (anchors.is_empty() || (lo < anchors[0] && !chained)) {
        anchors.insert(0, lo);
    }
    if hi.is_finite() && hi > anchors[anchors.len() - 1] {
        anchors.push(hi);
    }
    anchors
}

/// The chords to the motif-free outside at λ = 0 ([`ZeroChord`]) a fold at
/// `lambda` prunes on: against the upper anchor of its bracket `hi`, and
/// against the highest anchor, each where it lies above `lambda` and raises
/// the floor under the whole design.
///
/// A chord's weight on its anchor `ν` is `λ / ν`, so its energy term is the
/// anchor's at full weight λ whichever anchor it is; what differs is how much
/// unconstrained adaptation rides along, which the nearer anchor keeps small
/// and the highest keeps smallest, at the price of an energy term from a
/// completion chosen at a far λ. Under the whole design the chord reads
/// `weight · free_ν − (1 − weight) · 100 · open_chain`, against `free`, the
/// anchors' own chord of unconstrained optima. A chord that does not raise
/// that floor seldom raises any cell's and costs lookups at every one, so it
/// is left out: near λ = 0, where the motif set's cost in adaptation dwarfs
/// what energy weighs, the chords stay; elsewhere they drop.
///
/// `anchor(u)` is anchor `u`'s table and unconstrained optimum.
#[allow(clippy::too_many_arguments)]
fn zero_chords<'a>(
    chain: &'a ChainCai,
    open_chain: Option<f64>,
    anchors: &[f64],
    hi: usize,
    lambda: f64,
    free: Option<f64>,
    anchor: impl Fn(usize) -> (&'a PairTable, Option<f64>),
) -> Vec<ZeroChord<'a>> {
    let (Some(open_chain), Some(free)) = (open_chain, free) else {
        return Vec::new();
    };
    let top = anchors.len() - 1;
    [Some(hi), (top != hi).then_some(top)]
        .into_iter()
        .flatten()
        .filter(|&u| anchors[u] > lambda + 1e-12)
        .filter_map(|u| {
            let (upper, free_u) = anchor(u);
            let weight = lambda / anchors[u];
            let root = weight * free_u? - (1.0 - weight) * 100.0 * open_chain;
            (root > free + constrained::BOUND_SLACK).then_some(ZeroChord {
                chain,
                upper,
                weight,
            })
        })
        .collect()
}

/// The bound for a fold at `lambda` below the lowest anchor, where no chord of
/// two anchors reaches: the chords to the motif-free outside at λ = 0
/// ([`ZeroChord`]) against the lowest anchor and the highest, each a floor on
/// all of `[0, ν]`, and the higher of their roots, a floor under the
/// constrained optimum. The optimum is concave in λ, it is exactly
/// `−100 · open_chain` at λ = 0, and an anchor's unconstrained optimum is
/// below it at `ν`, so the chord of the two is below it in between. `None`
/// without a motif-free spelling, or where no anchor found a design.
/// `anchor(u)` is as for [`zero_chords`].
fn below_anchors<'a>(
    chain: &'a ChainCai,
    open_chain: Option<f64>,
    anchors: &[f64],
    lambda: f64,
    anchor: impl Fn(usize) -> (&'a PairTable, Option<f64>),
) -> Option<(Vec<ZeroChord<'a>>, f64)> {
    let open_chain = open_chain?;
    let top = anchors.len() - 1;
    let mut root = f64::NEG_INFINITY;
    let chords: Vec<ZeroChord<'a>> = [Some(0), (top != 0).then_some(top)]
        .into_iter()
        .flatten()
        .filter_map(|u| {
            let (upper, free_u) = anchor(u);
            let weight = lambda / anchors[u];
            root = root.max(weight * free_u? - (1.0 - weight) * 100.0 * open_chain);
            Some(ZeroChord {
                chain,
                upper,
                weight,
            })
        })
        .collect();
    (!chords.is_empty()).then_some((chords, root))
}

/// The two anchors bracketing `lambda`, as indices, with the weight on the
/// first.
///
/// `anchors` must be sorted and must **span** `lambda`: concavity puts the
/// chord between two anchors below the true value only *between* them, so
/// extrapolating past either end is not a floor and would prune cells a
/// design needs. [`span_lambdas`] guarantees the span; this asserts it rather
/// than return a bound it cannot justify.
fn bracket(anchors: &[f64], lambda: f64) -> (usize, usize, f64) {
    assert!(
        lambda >= anchors[0] - 1e-12 && lambda <= anchors[anchors.len() - 1] + 1e-12,
        "λ={lambda} lies outside the anchor range: the chord would extrapolate"
    );
    let hi = anchors
        .iter()
        .position(|&a| a >= lambda - 1e-12)
        .unwrap_or(anchors.len() - 1);
    if (anchors[hi] - lambda).abs() < 1e-12 || hi == 0 {
        return (hi, hi, 1.0);
    }
    let lo = hi - 1;
    // lambda = w*anchors[hi] + (1-w)*anchors[lo]
    let w = (lambda - anchors[lo]) / (anchors[hi] - anchors[lo]);
    (hi, lo, w)
}

/// Solve the motif-constrained problem at every λ in `lambdas`, sharing one
/// set of unconstrained outside passes taken at `anchor_lambdas`.
///
/// Each λ's bound is the chord of the two anchors bracketing it, which is a
/// floor because a cell's outside value is concave in λ (see
/// [`crate::constrained::OutsideBound`]). Each λ's ceiling is the best design
/// found at any earlier λ, rescored here: a feasible design's objective is an
/// upper bound on the optimum, so every cell on the optimal derivation
/// survives the bound and one fold certifies. The first λ has no incumbent and
/// starts from the unconstrained optimum, doubling its margin until a fold
/// certifies.
///
/// The anchors are computed [`memory_capped_workers`] at a time and
/// `config.threads` fans each fold out.
///
/// Returns `None` when the pattern set admits no design for the protein.
#[allow(clippy::too_many_arguments)]
pub fn design_motifs_swept<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambdas: &[f64],
    anchor_lambdas: &[f64],
    motifs: &[S],
    config: ScanConfig,
    beam: Option<crate::Beam>,
) -> Result<Option<SweptSweep>, BeamError> {
    design_motifs_swept_within(
        params,
        protein,
        usage,
        lambdas,
        anchor_lambdas,
        motifs,
        config,
        beam,
        0.0,
        0,
    )
}

/// [`design_motifs_swept`] in certified-approximate mode: each λ's design is
/// within its reported [`SweptSolve::gap`] `<= epsilon` of that λ's optimum
/// (see the module docs). `epsilon = 0` is [`design_motifs_swept`].
/// `anchor_workers > 1` computes the anchors side by side, independently of
/// `config.threads`, which fans each fold out; `anchor_workers = 0` takes
/// [`memory_capped_workers`], as [`design_motifs_swept`] does.
#[allow(clippy::too_many_arguments)]
pub fn design_motifs_swept_within<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambdas: &[f64],
    anchor_lambdas: &[f64],
    motifs: &[S],
    config: ScanConfig,
    beam: Option<crate::Beam>,
    epsilon: f64,
    anchor_workers: usize,
) -> Result<Option<SweptSweep>, BeamError> {
    let aa = crate::residues(protein)?;
    let graph = crate::dfa::build(&aa, usage);
    let ac = crate::ac::Ac::build(motifs)?;
    let product = ProductDfa::build(&graph, &ac);
    if !product.is_feasible() {
        return Ok(None);
    }
    let product = product.merge_equivalent();
    let region = PairConstraint::inert();
    let must = MustPairConstraint::inert();
    let open_chain = constrained::open_chain_cai(&product);
    let chain = ChainCai::build(&product);
    let anchors = span_lambdas(anchor_lambdas, lambdas, false);
    let mut tables = Vec::new();
    let mut free_optima = Vec::new();
    let mut anchor_seconds = Vec::new();
    for a in anchors_at(
        params,
        &graph,
        &anchors,
        config,
        anchor_workers,
        &region,
        &must,
    ) {
        tables.push(a.table);
        free_optima.push(a.free);
        anchor_seconds.push(a.seconds);
    }
    let mut solves = Vec::new();
    // Every design found so far, as (mfe, cai): each is feasible at every λ,
    // so each caps the optimum once rescored.
    let mut found: Vec<(i32, f64)> = Vec::new();
    let mut seen_cost: f64 = 0.0;
    let spans = std::cell::RefCell::new(None);
    for &lambda in lambdas {
        let (hi, lo, weight) = bracket(&anchors, lambda);
        let free = free_chord(free_optima[hi], free_optima[lo], weight);
        let zero = zero_chords(&chain, open_chain, &anchors, hi, lambda, free, |u| {
            (&tables[u][..], free_optima[u])
        });
        let solve = solve_one(
            SolveOne {
                params,
                product: &product,
                lambda,
                region: &region,
                must: &must,
                config,
                beam,
                open_chain,
                epsilon,
                seen_cost,
                guess_span: GUESS_SPAN,
                spans: &spans,
                zero: &zero,
                motif_free: None,
            },
            &tables[hi],
            (hi != lo).then_some((&tables[lo], weight)),
            (anchors[hi], anchors[lo]),
            &found,
            free,
        );
        let Some(mut solve) = solve else {
            return Ok(None);
        };
        if let Some(i) = solve.incumbent {
            let earlier: &SweptSolve = &solves[i];
            solve.sequence.clone_from(&earlier.sequence);
            solve.structure.clone_from(&earlier.structure);
        }
        if let Some(f) = free.filter(|_| solve.ceiling_from != CeilingSource::OpenChain) {
            let score = lambda * f64::from(solve.mfe) - (1.0 - lambda) * solve.cai * 100.0;
            seen_cost = seen_cost.max(constraint_cost(score, f));
        }
        found.push((solve.mfe, solve.cai));
        solves.push(solve);
    }
    Ok(Some(SweptSweep {
        solves,
        anchor_lambdas: anchors,
        anchor_seconds,
    }))
}

/// What one λ of a sweep folds over, past the bound itself.
struct SolveOne<'a> {
    params: &'a Params,
    product: &'a ProductDfa<'a>,
    lambda: f64,
    region: &'a PairConstraint,
    must: &'a MustPairConstraint,
    config: ScanConfig,
    /// A finite beam to seed this λ's ceiling from, if any.
    beam: Option<crate::Beam>,
    /// The log-CAI of the best motif-free spelling, left unpaired: an
    /// attainable objective at every λ (see
    /// [`crate::constrained::open_chain_cai`]).
    open_chain: Option<f64>,
    /// How far below a feasible incumbent the fold prunes: the largest gap
    /// to the optimum the solve may report. `0` is the exact search.
    epsilon: f64,
    /// The largest [`constraint_cost`] the earlier solves of this sweep or
    /// walk found (those under the open chain aside): how far above the
    /// unconstrained bound a ceiling can sit and still be near the optimum.
    seen_cost: f64,
    /// How far beyond `seen_cost`, in first margins, an incumbent may sit
    /// before the solve guesses: [`GUESS_SPAN`] but in a walk's wide rounds.
    guess_span: f64,
    /// The span memos the sweep's or walk's folds pass along
    /// ([`constrained::SpanMemo`]).
    spans: &'a std::cell::RefCell<Option<constrained::SpanMemo>>,
    /// Floors from the motif-free outside at λ = 0 ([`zero_chords`]).
    zero: &'a [ZeroChord<'a>],
    /// The chord of the motif-aware anchors bracketing λ, if the walk has them.
    motif_free: Option<constrained::MotifFreeChord<'a>>,
}

/// Fold one λ under the chord of two anchors, starting from the best ceiling
/// the earlier λ values left.
///
/// With an incumbent the first fold **must** certify: the incumbent is a
/// feasible design's objective, so the optimum is at or below it, so every
/// cell on the optimum's derivation clears the floor and survives. A fold that
/// comes back above its ceiling therefore means the bound was not a floor, and
/// the only safe response is to re-fold with nothing pruned — raising the
/// ceiling cannot help, because it is already an attainable score. That
/// fallback is recorded in `folds`, so a sweep that silently loses its bound
/// shows up as two folds per λ rather than as a wrong answer.
///
/// With `it.epsilon > 0` and an incumbent among `found`, the first fold prunes
/// against `incumbent - ε` instead (see the module docs); a fold that does not
/// certify there settles the λ with a proven gap rather than re-folding.
///
/// `free` is a lower bound on the unconstrained optimum at this λ, which bounds
/// the constrained one from below. Where the incumbent sits more than
/// [`GUESS_SPAN`] first margins beyond the largest constraint cost seen so far
/// (`it.seen_cost`) above it — the first probes of a hull walk, whose
/// inherited ceilings are the tie scores of far-apart ends — the solve
/// first searches up from that bound, doubling the margin, and only then
/// folds under the incumbent. A guess below the optimum prunes almost every
/// pair, so it comes back empty or above itself at little cost; the first
/// guess above the optimum certifies under a ceiling far tighter than the
/// inherited one. No guess is attainable, so a failed one only moves to the
/// next, and the incumbent fold that ends the search keeps its guarantee.
/// Measuring against the observed cost keeps a ceiling already near the
/// optimum — a closed gap's — from guessing, however much the motif set
/// costs.
fn solve_one<'t>(
    it: SolveOne<'_>,
    near: &'t PairTable,
    far: Option<(&'t PairTable, f64)>,
    anchors: (f64, f64),
    found: &[(i32, f64)],
    free: Option<f64>,
) -> Option<SweptSolve> {
    let lambda = it.lambda;
    let brd = |mfe: i32, cai: f64| lambda * f64::from(mfe) - (1.0 - lambda) * cai * 100.0;
    let started = std::time::Instant::now();
    let neighbour = found
        .iter()
        .map(|&(m, c)| brd(m, c))
        .fold(f64::INFINITY, f64::min);
    // A beam design at this λ is feasible, so it caps the optimum too — and
    // unlike a neighbour's design it does not degrade where the front is
    // steep, which is exactly where a rescored neighbour becomes a poor
    // ceiling.
    let beamed = it.beam.and_then(|b| {
        crate::constrained::fold_beamed(
            it.product,
            it.params,
            lambda,
            it.region.clone(),
            it.must.clone(),
            it.config.threads,
            b,
        )
        .map(|f| brd(f.mfe.0, f.cai))
    });
    // The unpaired best-adapted spelling is feasible too, and near λ = 0,
    // where every other design in hand scores far above the optimum, it is
    // nearly the answer. Elsewhere it is a poor ceiling — far worse than
    // searching up from the unconstrained optimum — so with no other design
    // in hand it only caps that search.
    let open = it
        .open_chain
        .map_or(f64::INFINITY, |cai| -(1.0 - lambda) * cai * 100.0);
    let held = beamed.map_or(neighbour, |b| b.min(neighbour));
    let incumbent = if held.is_finite() {
        held.min(open)
    } else {
        held
    };
    let mut source = match beamed {
        _ if held.is_finite() && open < held => CeilingSource::OpenChain,
        Some(b) if !neighbour.is_finite() || b <= neighbour => CeilingSource::Beam,
        _ if neighbour.is_finite() => CeilingSource::Neighbour,
        _ => CeilingSource::FreeOptimum,
    };
    // Without any feasible design, start a little above the unconstrained
    // optimum — which bounds the constrained one from below — and double until
    // a fold certifies, as the single-λ search does.
    let mut margin = free.map_or(f64::INFINITY, first_margin);
    let mut ceiling = if incumbent.is_finite() {
        incumbent
    } else {
        source = CeilingSource::FreeOptimum;
        free.map_or(f64::INFINITY, |f| f + margin).min(open)
    };
    let first_ceiling = ceiling;
    // The ε cut needs the incumbent's design in hand to fall back on, which a
    // beam ceiling's is not.
    let best_found = (0..found.len())
        .min_by(|&i, &j| brd(found[i].0, found[i].1).total_cmp(&brd(found[j].0, found[j].1)));
    let approximate = it.epsilon > 0.0
        && neighbour.is_finite()
        && neighbour <= open
        && beamed.is_none_or(|b| neighbour <= b);
    let mut prune_at = if approximate {
        ceiling - it.epsilon
    } else {
        ceiling
    };
    // A solve under the open chain does not guess: near λ = 0 it is all but
    // the answer already, and the first margin is too small a unit to read
    // the distance in. An approximate solve guesses like an exact one and
    // makes its ε cut under the incumbent once the guesses reach it.
    let guess_from = free.filter(|&f| {
        source != CeilingSource::OpenChain
            && incumbent.is_finite()
            && constraint_cost(incumbent, f) > it.seen_cost + it.guess_span
    });
    if let Some(f) = guess_from {
        prune_at = f + margin;
    }
    let mut guessing = guess_from.is_some();
    let mut guesses = 0;
    let mut cap = open;
    let mut folds = 0;
    let mut pruned_total = 0;
    let mut cut_total = 0;
    let settled =
        |mfe: i32, cai: f64, gap: f64, incumbent: Option<usize>, folds, pruned| SweptSolve {
            lambda,
            mfe,
            cai,
            sequence: String::new(),
            structure: String::new(),
            ceiling: first_ceiling,
            folds,
            guesses: 0,
            pruned,
            cut: 0,
            zero_chords: it.zero.len(),
            fold_seconds: started.elapsed().as_secs_f64(),
            anchors,
            ceiling_from: source,
            ceiling_gap: first_ceiling - brd(mfe, cai),
            gap,
            incumbent,
        };
    loop {
        let bound = crate::constrained::OutsideBound {
            outside: near,
            chord: far,
            through: true,
            zero: it.zero,
            motif_free: it.motif_free,
            aware: None,
            ceiling: prune_at,
        };
        let (fold, counts) = crate::constrained::fold_bounded(
            it.product,
            it.params,
            lambda,
            it.region.clone(),
            it.must.clone(),
            it.config.threads,
            bound,
            &mut it.spans.borrow_mut(),
        );
        folds += 1;
        guesses += usize::from(guessing);
        pruned_total += counts.pruned;
        cut_total += counts.cut;
        if let Some(f) = &fold {
            if brd(f.mfe.0, f.cai) <= prune_at + crate::constrained::BOUND_SLACK {
                return Some(SweptSolve {
                    sequence: f.sequence.clone(),
                    structure: f.structure.clone(),
                    guesses,
                    cut: cut_total,
                    ..settled(f.mfe.0, f.cai, 0.0, None, folds, pruned_total)
                });
            }
            cap = cap.min(brd(f.mfe.0, f.cai));
        }
        if guessing {
            // The optimum is above this guess; try the next one, or end the
            // search with the incumbent fold once a guess would reach it.
            margin *= 2.0;
            let next = guess_from.map_or(f64::INFINITY, |f| f + margin);
            // The ε cut falls back on an incumbent among `found`, so an
            // approximate solve keeps that incumbent as its ceiling rather than
            // a cheaper guess's design it would not have in hand.
            let top = if approximate {
                ceiling - it.epsilon
            } else {
                ceiling.min(cap)
            };
            guessing = next < top;
            prune_at = if guessing { next } else { top };
            if !approximate {
                ceiling = ceiling.min(cap);
            }
            continue;
        }
        if prune_at < ceiling {
            // Nothing scores below `prune_at`, or it would have survived the
            // bound and come back: the optimum is at least that, and the best
            // design in hand is within the gap of it.
            let floor = prune_at - crate::constrained::BOUND_SLACK;
            let i = best_found.expect("an approximate solve has an incumbent");
            let (mfe, cai) = found[i];
            let answer = match &fold {
                Some(f) if brd(f.mfe.0, f.cai) < brd(mfe, cai) => SweptSolve {
                    sequence: f.sequence.clone(),
                    structure: f.structure.clone(),
                    ..settled(
                        f.mfe.0,
                        f.cai,
                        brd(f.mfe.0, f.cai) - floor,
                        None,
                        folds,
                        pruned_total,
                    )
                },
                _ => settled(
                    mfe,
                    cai,
                    brd(mfe, cai) - floor,
                    Some(i),
                    folds,
                    pruned_total,
                ),
            };
            return Some(SweptSolve {
                cut: cut_total,
                ..answer
            });
        }
        if fold.is_none() && ceiling.is_infinite() {
            // Nothing feasible with nothing pruned: the set admits no design.
            return None;
        }
        if ceiling.is_infinite() {
            // An unbounded fold returned a design above an infinite ceiling,
            // which cannot happen.
            unreachable!("a fold with nothing pruned did not certify");
        }
        // An inherited ceiling is already attainable, so raising toward it is
        // not available: go straight to pruning nothing.
        ceiling = if incumbent.is_finite() {
            f64::INFINITY
        } else {
            margin *= 2.0;
            if margin > MAX_MARGIN {
                f64::INFINITY
            } else {
                free.map_or(f64::INFINITY, |f| f + margin).min(cap)
            }
        };
        prune_at = ceiling;
    }
}

/// One vertex of a motif-constrained front's convex hull: a motif-free design
/// that is the constrained optimum at some λ.
#[derive(Debug, Clone)]
pub struct MotifVertex {
    /// Free energy in dekacalories and the summed log codon weights.
    pub mfe: i32,
    pub cai: f64,
    pub sequence: String,
    pub structure: String,
}

impl MotifVertex {
    fn score(&self, lambda: f64) -> f64 {
        lambda * f64::from(self.mfe) - (1.0 - lambda) * self.cai * 100.0
    }
}

/// One fold a hull walk ran: the λ it ran at, the ceiling it ran under, what
/// it found, and what it cost.
#[derive(Debug, Clone)]
pub struct HullProbe {
    pub lambda: f64,
    /// BRD units. For a gap probe, the score both ends of the gap share at
    /// its tie weight; `INFINITY` for an end fold and for an unbounded walk.
    pub ceiling: f64,
    /// The constrained optimum at `lambda`, BRD units.
    pub optimum: f64,
    /// Whether the probe found a vertex strictly below the gap's chord.
    pub new_vertex: bool,
    pub folds: usize,
    /// How many of `folds` ran under a guessed ceiling ([`SweptSolve::guesses`]).
    pub guesses: usize,
    /// Closures the pair floor skipped ([`SweptSolve::cut`]).
    pub cut: usize,
    /// Chords to λ = 0 the folds pruned on ([`SweptSolve::zero_chords`]).
    pub zero_chords: usize,
    pub pruned: usize,
    /// The probe's own wall seconds, from when its bracketing anchors are
    /// ready; the probes of one round overlap.
    pub seconds: f64,
    /// The proven bound on how far the probe's design sits above the optimum
    /// at `lambda`: `0` for an exact probe.
    pub gap: f64,
    /// Whether the folds also pruned on the chord of motif-aware anchors.
    pub motif_aware: bool,
}

/// The convex hull of a motif-constrained front, found by folds at
/// adaptively chosen λ.
#[derive(Debug, Clone)]
pub struct MotifHullWalk {
    /// Hull vertices, most stable first.
    pub vertices: Vec<MotifVertex>,
    /// Every fold: the two ends, then each round's gaps in order.
    pub probes: Vec<HullProbe>,
    /// The anchor λ values the bound was shared over; empty for an unbounded
    /// walk.
    pub anchor_lambdas: Vec<f64>,
    pub anchor_seconds: Vec<f64>,
    /// How many jobs the walk ran side by side: the caller's count, or the
    /// one [`memory_capped_workers`] chose.
    pub workers: usize,
    /// The most probe folds a round ran side by side: `workers`, or more where
    /// the walk chose its own count and the folds fit the memory left once
    /// every anchor was in.
    pub probe_workers: usize,
    /// The λ of the motif-aware anchors the walk computed, and each one's
    /// seconds; empty when it computed none.
    pub motif_aware_lambdas: Vec<f64>,
    pub motif_aware_seconds: Vec<f64>,
}

/// Find the convex hull of the motif-constrained front between `ends.0` (the
/// adapted end, small λ) and `ends.1` (the stable end, large λ).
///
/// The walk folds both ends, then repeatedly takes a gap between two known
/// vertices `a` (more stable) and `b` (more adapted) and folds at their tie
/// weight — the λ at which they score the same, the slope of the chord
/// between them. A design scoring strictly below that common score is a new
/// vertex and splits the gap in two; otherwise no λ in the gap prefers any
/// design to `a` or `b`, and the gap is closed. This places λ where the front
/// bends rather than on a fixed grid, and it stops when the hull is complete.
///
/// **Why it suits the shared bound.** At the tie weight both ends score the
/// same value `T`, and both are feasible, so `T` is a ceiling on the optimum
/// there. When the gap is closed — which the last probe of every gap is — the
/// optimum *equals* `T`, so the ceiling sits exactly on the answer: the
/// regime in which the outside bound prunes hardest. A fixed λ grid instead
/// inherits a neighbour's design, whose rescored score can sit far above the
/// optimum wherever the front is steep.
///
/// With `anchor_lambdas` the probes are bounded folds under the chord of the
/// bracketing anchors (see [`design_motifs_swept`]); without, every probe is a
/// full exact fold, which is the reference the bounded walk must reproduce.
/// The anchors need not reach `ends.0`: below the lowest one the probes are
/// bounded by the chords to the motif-free outside at λ = 0 instead.
/// The anchors, and then each round's gap probes, run [`memory_capped_workers`] at a
/// time, and `config.threads` fans each fold out.
///
/// Returns `None` when the pattern set admits no design for the protein.
#[allow(clippy::too_many_arguments)]
pub fn walk_motif_hull<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    motifs: &[S],
    ends: (f64, f64),
    anchor_lambdas: Option<&[f64]>,
    config: ScanConfig,
) -> Result<Option<MotifHullWalk>, BeamError> {
    walk_motif_hull_within(
        params,
        protein,
        usage,
        motifs,
        ends,
        anchor_lambdas,
        config,
        0.0,
        0,
    )
}

/// [`walk_motif_hull`] in certified-approximate mode: each bounded probe that
/// has feasible designs in hand prunes `epsilon` below the best of them (see
/// the module docs), and a gap closes once no design is found more than
/// `epsilon` below its tie score.
///
/// The hull it returns lies within `epsilon` of the exact one: its lower
/// envelope is above the constrained optimum by at most `epsilon` at every λ
/// in `ends`. At a closed gap's tie weight every design scores at least the
/// tie score minus the probe's gap; a design's objective is linear in λ and,
/// at each gap end's own λ, no lower than that end's score less its gap; so
/// it stays within the larger of the two across the gap. `epsilon = 0` is
/// [`walk_motif_hull`]; an unbounded walk (`anchor_lambdas = None`) ignores
/// `epsilon`. The walk runs `workers` jobs side by side — the anchors, then
/// each round's gap probes — independently of `config.threads`, which fans
/// each fold out; `workers = 0` takes [`memory_capped_workers`] for the
/// anchors, each scanning on its share of the cores the cap leaves idle, and
/// once every anchor is in, each round takes as many workers as
/// its folds fit in the memory left ([`probe_workers`]). The walk is the same
/// at any count.
#[allow(clippy::too_many_arguments)]
pub fn walk_motif_hull_within<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    motifs: &[S],
    ends: (f64, f64),
    anchor_lambdas: Option<&[f64]>,
    config: ScanConfig,
    epsilon: f64,
    workers: usize,
) -> Result<Option<MotifHullWalk>, BeamError> {
    walk_motif_hull_aware(
        params,
        protein,
        usage,
        motifs,
        ends,
        anchor_lambdas,
        config,
        epsilon,
        workers,
        &[],
        true,
    )
}

/// [`walk_motif_hull_within`] with motif-aware anchors at `motif_aware`.
///
/// The unconstrained anchors' bound misses what the motif set costs, and at
/// low λ that cost, divided by λ, is a wide energy window every structure in
/// it survives. A motif-aware anchor is the outside of every closed pair of the
/// constrained lattice itself ([`constrained::outside`]), so a probe between two
/// of them prunes on their chord as well, which carries the motifs' cost.
/// Each is a pruned fold under the ordinary bound at its λ, below the best
/// design in hand plus [`MOTIF_AWARE_MARGIN`], so its values are floors, not
/// exact. Each is computed once, in the background, from a fixed round on:
/// the second for those at or below [`MOTIF_AWARE_EARLY`], else the first after
/// the [`WIDE_ROUNDS`]. With `aware_waits`, a probe between two started ones
/// waits for both, so which floors each probe prunes on depends only on the
/// walk, at any worker count. Without it, a probe folds without their chord
/// until both are in, and the probes still waiting on one are dealt last in
/// their round: the aware anchors stay off the critical path, and only what
/// a probe prunes, never what it finds, depends on timing. `motif_aware`
/// values outside the anchors' range are dropped; an unbounded walk ignores
/// them all.
#[allow(clippy::too_many_arguments)]
pub fn walk_motif_hull_aware<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    motifs: &[S],
    ends: (f64, f64),
    anchor_lambdas: Option<&[f64]>,
    config: ScanConfig,
    epsilon: f64,
    workers: usize,
    motif_aware: &[f64],
    aware_waits: bool,
) -> Result<Option<MotifHullWalk>, BeamError> {
    const TOL: f64 = 1e-6;
    let aa = crate::residues(protein)?;
    let graph = crate::dfa::build(&aa, usage);
    let ac = crate::ac::Ac::build(motifs)?;
    let product = ProductDfa::build(&graph, &ac);
    if !product.is_feasible() {
        return Ok(None);
    }
    let product = product.merge_equivalent();
    let region = PairConstraint::inert();
    let must = MustPairConstraint::inert();
    let open_chain = constrained::open_chain_cai(&product);
    let chain = ChainCai::build(&product);
    let (lo, hi) = ends;

    // Shared outside passes, when the walk is bounded.
    let anchors = anchor_lambdas.map_or_else(Vec::new, |given| {
        span_lambdas(given, &[lo, hi], open_chain.is_some())
    });
    let chosen = workers == 0;
    let aware: Vec<f64> = if anchors.is_empty() {
        Vec::new()
    } else {
        let (first, last) = (anchors[0], anchors[anchors.len() - 1]);
        let mut aware: Vec<f64> = motif_aware
            .iter()
            .copied()
            .filter(|&nu| nu >= first - 1e-12 && nu <= last + 1e-12)
            .collect();
        aware.sort_by(f64::total_cmp);
        aware.dedup();
        aware
    };
    let workers = match workers {
        // The aware anchors start after the unconstrained ones are in, and
        // the probe rounds then read the memory left.
        0 => memory_capped_workers(anchors.len(), &graph),
        given => given,
    };
    let feed = AnchorFeed::new(anchors.len());
    let mut widest = workers.max(1);
    // Where memory held the walk below the cores, each anchor scans on its
    // share of the cores the missing workers leave idle.
    let share = if chosen {
        (default_workers(usize::MAX) / workers.max(1)).max(1)
    } else {
        1
    };
    if workers <= 1 {
        let config = ScanConfig {
            threads: config.threads.max(share),
            ..config
        };
        feed.work(params, &graph, &anchors, config, (1, 1), &region, &must);
    }
    // The bound for a fold at `lambda`: the unconstrained anchors bracketing
    // it (once computed), their tables, their λ, the chords to λ = 0,
    // and a floor under the unconstrained optimum.
    let bounds_at = |lambda: f64| {
        let table = |u: usize| {
            let a = feed.get(u);
            (&a.table[..], a.free)
        };
        if lambda < anchors[0] - 1e-12 {
            let (zero, root) = below_anchors(&chain, open_chain, &anchors, lambda, table)
                .expect("the anchors stop short of λ only with a motif-free spelling");
            let none: &PairTable = &[];
            return (none, None, (anchors[0], 0.0), zero, Some(root));
        }
        let (h, l, w) = bracket(&anchors, lambda);
        let (near, far) = (feed.get(h), feed.get(l));
        let free = free_chord(near.free, far.free, w);
        let zero = zero_chords(&chain, open_chain, &anchors, h, lambda, free, table);
        (
            &near.table[..],
            (h != l).then(|| (&far.table[..], w)),
            (anchors[h], anchors[l]),
            zero,
            free,
        )
    };
    // The motif-aware anchors' tables, each `None` where its fold found no
    // design, with its seconds, set once by a background job.
    type AwareTable = Option<Vec<crate::hash::FxHashMap<constrained::PKey, f64>>>;
    let aware_tables: Vec<std::sync::OnceLock<(AwareTable, f64)>> =
        aware.iter().map(|_| std::sync::OnceLock::new()).collect();
    // Whether each aware anchor's job has started; set before the round that
    // starts it is dealt.
    let aware_live: Vec<std::sync::atomic::AtomicBool> = aware
        .iter()
        .map(|_| std::sync::atomic::AtomicBool::new(false))
        .collect();
    let aware_table = |nu: f64, ceiling: f64| -> AwareTable {
        let (near, far, _, zero, _) = bounds_at(nu);
        constrained::outside::motif_free_outside_pruned(
            &product,
            params,
            nu,
            OutsideBound {
                outside: near,
                chord: far,
                through: true,
                zero: &zero,
                motif_free: None,
                aware: None,
                ceiling,
            },
            MOTIF_AWARE_THREADS,
        )
    };
    // The chord of the two motif-aware anchors bracketing `lambda`; `None`
    // outside their range, before both have started, where either found
    // nothing, and (without `aware_waits`) until both are computed.
    let aware_chord = |lambda: f64| {
        let (Some(&first), Some(&last)) = (aware.first(), aware.last()) else {
            return None;
        };
        if lambda < first - 1e-12 || lambda > last + 1e-12 {
            return None;
        }
        let (h, l, w) = bracket(&aware, lambda);
        let get = |k: usize| {
            if !aware_live[k].load(std::sync::atomic::Ordering::Acquire) {
                None
            } else if aware_waits {
                Some(aware_tables[k].wait())
            } else {
                aware_tables[k].get()
            }
        };
        let hi = get(h)?.0.as_deref()?;
        let lo = get(l)?.0.as_deref()?;
        Some(constrained::MotifFreeChord {
            lo,
            hi,
            weight: w,
            complete: false,
        })
    };
    let walked = std::thread::scope(|scope| {
        if workers > 1 {
            for _ in 0..workers.min(anchors.len()) {
                scope.spawn(|| {
                    feed.work(
                        params,
                        &graph,
                        &anchors,
                        config,
                        (workers, share),
                        &region,
                        &must,
                    );
                });
            }
        }
        // One fold at `lambda`. Bounded: under the chord of the anchors bracketing
        // it, with `seed` as the feasible designs that cap the optimum (none for
        // an end, which then searches up from the unconstrained optimum), and
        // `seen_cost` as the constraint cost the earlier folds found. Unbounded:
        // the full exact fold. `spans` is the calling worker's span memo. Returns
        // the design, the probe's record and the constraint cost this fold saw.
        type Folded = (MotifVertex, HullProbe, Option<f64>);
        let fold = |lambda: f64,
                    seed: &[&MotifVertex],
                    ceiling: f64,
                    (seen_cost, guess_span): (f64, f64),
                    spans: &std::cell::RefCell<Option<constrained::SpanMemo>>,
                    aware_on: bool|
         -> Option<Folded> {
            // The bracketing anchors, once computed; the probe's clock starts after.
            let bracketed = (!anchors.is_empty()).then(|| bounds_at(lambda));
            let motif_free = if aware_on { aware_chord(lambda) } else { None };
            let started = std::time::Instant::now();
            let (vertex, folds, guesses, pruned, cut, zero_chords, gap, observed) =
                if let Some((near, far, pair, zero, free)) = bracketed {
                    let seeds: Vec<(i32, f64)> = seed.iter().map(|v| (v.mfe, v.cai)).collect();
                    let it = SolveOne {
                        params,
                        product: &product,
                        lambda,
                        region: &region,
                        must: &must,
                        config,
                        beam: None,
                        open_chain,
                        epsilon,
                        seen_cost,
                        guess_span,
                        spans,
                        zero: &zero,
                        motif_free,
                    };
                    let s = solve_one(it, near, far, pair, &seeds, free)?;
                    // Near λ = 0 the first margin is too small a unit (see
                    // `solve_one`), so the open chain's solves set no scale.
                    let observed = free
                        .filter(|_| s.ceiling_from != CeilingSource::OpenChain)
                        .map(|f| {
                            let score = lambda * f64::from(s.mfe) - (1.0 - lambda) * s.cai * 100.0;
                            constraint_cost(score, f)
                        });
                    let vertex = match s.incumbent {
                        Some(i) => seed[i].clone(),
                        None => MotifVertex {
                            mfe: s.mfe,
                            cai: s.cai,
                            sequence: s.sequence,
                            structure: s.structure,
                        },
                    };
                    (
                        vertex,
                        s.folds,
                        s.guesses,
                        s.pruned,
                        s.cut,
                        s.zero_chords,
                        s.gap,
                        observed,
                    )
                } else {
                    let f = crate::constrained::fold_beamed(
                        &product,
                        params,
                        lambda,
                        region.clone(),
                        must.clone(),
                        config.threads,
                        crate::Beam::exact(),
                    )?;
                    (
                        MotifVertex {
                            mfe: f.mfe.0,
                            cai: f.cai,
                            sequence: f.sequence,
                            structure: f.structure,
                        },
                        1,
                        0,
                        0,
                        0,
                        0,
                        0.0,
                        None,
                    )
                };
            let probe = HullProbe {
                lambda,
                ceiling,
                optimum: vertex.score(lambda),
                new_vertex: false,
                folds,
                guesses,
                pruned,
                cut,
                zero_chords,
                seconds: started.elapsed().as_secs_f64(),
                gap,
                motif_aware: motif_free.is_some(),
            };
            Some((vertex, probe, observed))
        };

        let mut memos: Vec<std::cell::RefCell<Option<constrained::SpanMemo>>> = (0..workers.max(1))
            .map(|_| std::cell::RefCell::new(None))
            .collect();
        let mut seen_cost = 0.0_f64;
        let mut probes: Vec<HullProbe> = Vec::new();
        let (stable, probe, observed) = fold(
            hi,
            &[],
            f64::INFINITY,
            (seen_cost, GUESS_SPAN),
            &memos[0],
            false,
        )?;
        if let Some(c) = observed {
            seen_cost = seen_cost.max(c);
        }
        probes.push(probe);
        // The stable end is feasible at every λ, so it caps the adapted end too.
        let (adapted, probe, observed) = fold(
            lo,
            &[&stable],
            f64::INFINITY,
            (seen_cost, GUESS_SPAN),
            &memos[0],
            false,
        )?;
        if let Some(c) = observed {
            seen_cost = seen_cost.max(c);
        }
        probes.push(probe);
        let mut vertices = vec![stable.clone()];
        let mut open = Vec::new();
        if stable.mfe < adapted.mfe && stable.cai < adapted.cai {
            vertices.push(adapted.clone());
            open.push((stable, adapted));
        }
        // In rounds: every gap a round opens is probed in the next, side by side,
        // under the constraint cost the earlier rounds found (the first
        // `WIDE_ROUNDS` guess on their own span instead), and the results are
        // taken in gap order. A gap's refinement reads only its own two ends,
        // so the hull does not depend on the order; rounds make every probe the
        // same at any worker count as well.
        for depth in 0.. {
            if open.is_empty() {
                break;
            }
            let round: Vec<(MotifVertex, MotifVertex, f64)> = open
                .drain(..)
                .filter_map(|(a, b): (MotifVertex, MotifVertex)| {
                    let lambda = crate::tie_weight(
                        crate::HullPoint {
                            mfe_kcal_mol: f64::from(a.mfe) / 100.0,
                            adaptation_log: a.cai,
                        },
                        crate::HullPoint {
                            mfe_kcal_mol: f64::from(b.mfe) / 100.0,
                            adaptation_log: b.cai,
                        },
                    )?;
                    Some((a, b, lambda))
                })
                .collect();
            // Dealt from the lowest λ up: the costliest probes sit near λ = 0, and
            // starting them first evens out the workers' share of the round.
            let mut deal: Vec<usize> = (0..round.len()).collect();
            deal.sort_by(|&i, &j| round[i].2.total_cmp(&round[j].2));
            let guess = if depth < WIDE_ROUNDS {
                (0.0, WIDE_GUESS_SPAN)
            } else {
                (seen_cost, GUESS_SPAN)
            };
            // Once every anchor is in, a round needs only its folds' memory.
            let width = if chosen && feed.complete() {
                probe_workers(workers.max(1), &graph)
            } else {
                workers.max(1)
            };
            if width > memos.len() {
                memos.resize_with(width, || std::cell::RefCell::new(None));
            }
            widest = widest.max(width);
            // Each motif-aware anchor starts at its round, below the best
            // design in hand there.
            for (u, (slot, &nu)) in aware_tables.iter().zip(&aware).enumerate() {
                let first = if nu <= MOTIF_AWARE_EARLY + 1e-12 {
                    1
                } else {
                    WIDE_ROUNDS
                };
                if depth < first || aware_live[u].load(std::sync::atomic::Ordering::Acquire) {
                    continue;
                }
                let ceiling = vertices
                    .iter()
                    .map(|v| v.score(nu))
                    .fold(f64::INFINITY, f64::min)
                    + MOTIF_AWARE_MARGIN;
                aware_live[u].store(true, std::sync::atomic::Ordering::Release);
                let aware_table = &aware_table;
                scope.spawn(move || {
                    let _fill = FillOnPanic(slot);
                    let started = std::time::Instant::now();
                    let table = aware_table(nu, ceiling);
                    let _ = slot.set((table, started.elapsed().as_secs_f64()));
                });
            }
            // Probes still short of their aware chord go last, so the tables
            // have the round's length to arrive.
            if !aware_waits && !aware.is_empty() {
                let pending = |lambda: f64| {
                    lambda >= aware[0] - 1e-12
                        && lambda <= aware[aware.len() - 1] + 1e-12
                        && aware_chord(lambda).is_none()
                };
                deal.sort_by_key(|&i| pending(round[i].2));
            }
            let dealt = side_by_side(&mut memos[..width], deal.len(), |k, spans| {
                let (a, b, lambda) = &round[deal[k]];
                // Both ends score the ceiling here, and both are feasible.
                fold(
                    *lambda,
                    &[a, b],
                    a.score(*lambda),
                    guess,
                    spans,
                    !aware.is_empty(),
                )
            });
            let mut solved: Vec<Option<Option<Folded>>> = (0..round.len()).map(|_| None).collect();
            for (k, out) in dealt.into_iter().enumerate() {
                solved[deal[k]] = Some(out);
            }
            for ((a, b, lambda), solved) in round.into_iter().zip(solved) {
                let (p, mut probe, observed) = solved.expect("every gap is dealt once")?;
                if let Some(c) = observed {
                    seen_cost = seen_cost.max(c);
                }
                let ceiling = a.score(lambda);
                let new = p.score(lambda) < ceiling - TOL && a.mfe < p.mfe && p.mfe < b.mfe;
                probe.new_vertex = new;
                probes.push(probe);
                if new {
                    vertices.push(p.clone());
                    open.push((a, p.clone()));
                    open.push((p, b));
                }
            }
        }
        Some((vertices, probes))
    });
    let Some((mut vertices, probes)) = walked else {
        return Ok(None);
    };
    vertices.sort_by_key(|v| v.mfe);
    Ok(Some(MotifHullWalk {
        vertices,
        probes,
        anchor_lambdas: anchors,
        anchor_seconds: feed.seconds(),
        workers,
        probe_workers: widest,
        motif_aware_lambdas: aware
            .iter()
            .zip(&aware_tables)
            .filter(|(_, t)| t.get().is_some())
            .map(|(&nu, _)| nu)
            .collect(),
        motif_aware_seconds: aware_tables
            .iter()
            .filter_map(|t| t.get().map(|x| x.1))
            .collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Energy;

    fn next(state: &mut u64) -> usize {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as usize
    }

    /// A random protein and two motifs cut from its free design at λ = 0.5,
    /// so the motif set costs adaptation the unconstrained bound counts.
    fn case(rng: &mut u64, params: &Params, usage: &CodonUsage) -> (String, Vec<String>) {
        const ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
        let len = 12 + next(rng) % 5;
        let protein: String = (0..len)
            .map(|_| ALPHABET[next(rng) % ALPHABET.len()] as char)
            .collect();
        let free = crate::design(params, &protein, usage, 0.5)
            .unwrap()
            .sequence;
        let motifs = [5, 6]
            .into_iter()
            .map(|k| {
                let at = next(rng) % (free.len() - k + 1);
                free[at..at + k].to_string()
            })
            .collect();
        (protein, motifs)
    }

    /// Anchors past the last full round of workers share the workers' threads;
    /// the rest scan single-threaded.
    #[test]
    fn tail_anchors_share_the_idle_workers() {
        assert_eq!(tail_split(10, 8), (8, 4));
        assert_eq!(tail_split(11, 10), (10, 10));
        assert_eq!(tail_split(10, 5), (10, 1));
        assert_eq!(tail_split(10, 3), (9, 3));
        assert_eq!(tail_split(10, 12), (0, 1));
        assert_eq!(tail_split(3, 0), (3, 1));
    }

    /// Probe rounds take as many folds as fit the memory left, up to the cores;
    /// none fit where the memory would not hold one.
    #[test]
    fn probe_folds_fit_the_memory_left() {
        let edges = 1746;
        let area = 1746.0_f64 * 1746.0;
        let room = |folds: f64| (folds * FOLD_BYTES_PER_EDGE2 * area / MEMORY_BUDGET).ceil() as u64;
        assert_eq!(folds_within(8, u64::MAX, edges), 8);
        assert_eq!(folds_within(8, room(5.0), edges), 5);
        assert_eq!(folds_within(8, room(5.0) - 1, edges), 4);
        assert_eq!(folds_within(8, room(0.5), edges), 0);
        let graph = crate::dfa::build(&[0, 1, 2, 3], &CodonUsage::load_default().unwrap());
        assert!(probe_workers(3, &graph) >= 3);
        assert!(probe_workers(1, &graph) <= default_workers(usize::MAX).max(1));
    }

    /// A closed pair as the constrained fold keys it, by position: the opening
    /// and closing positions and nucleotides, the automaton state before the
    /// opening nucleotide and the one after the closing nucleotide.
    type PairAt = (u32, u32, u8, u8, u32, u32);

    /// The exact motif-free **through** value of every closed pair at `lambda`,
    /// by enumeration: over every motif-free encoding, the best objective of a
    /// structure containing the pair. On one encoding the adaptation is fixed,
    /// so the best structure through a pair is the energy through table of the
    /// encoding's own single-path lattice (scan and outside at λ = 1, exact
    /// there). Also returns the constrained optimum, the least of them all.
    ///
    /// The reference a motif-aware outside must reproduce, and the floor every
    /// bound the constrained fold prunes on must stay under.
    fn motif_free_through(
        params: &Params,
        usage: &CodonUsage,
        protein: &str,
        motifs: &[String],
        lambda: f64,
    ) -> Option<(crate::hash::FxHashMap<PairAt, f64>, f64)> {
        use derna_codon_dp::codon_table::aa_to_index;
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        let graph = crate::dfa::build(&aa, usage);
        let ac = crate::ac::Ac::build(motifs).unwrap();
        let product = ProductDfa::build(&graph, &ac);
        if !product.is_feasible() {
            return None;
        }
        let product = product.merge_equivalent();
        let lattice = product.lattice_ac();
        let config = ScanConfig::default();
        let inert = PairConstraint::inert();
        let mut out: crate::hash::FxHashMap<PairAt, f64> = Default::default();
        let mut optimum = f64::INFINITY;
        'encodings: for (cai, seq) in graph.paths_between(graph.start(), graph.end()) {
            let (mut node, mut q) = (graph.start(), lattice.start());
            let mut states = vec![q];
            for &x in &seq {
                let Some(next) = lattice.step(node, q, x) else {
                    continue 'encodings;
                };
                q = next;
                node = graph.right(node).iter().find(|e| e.nuc == x).unwrap().to;
                states.push(q);
            }
            let codons: Vec<[u8; 3]> = seq
                .chunks_exact(3)
                .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
                .collect();
            let single = crate::dfa::build_single_path(&aa, &codons);
            let scan = crate::scan::scan_values_with_constraints(
                params,
                &single,
                1.0,
                config,
                inert.clone(),
                MustPairConstraint::inert(),
            );
            let energy = scan.c[single.ix(single.end())].map(|s| s.brd(1.0)).unwrap();
            optimum = optimum.min(lambda * energy - (1.0 - lambda) * cai * 100.0);
            let costs = OutsideScan::cost_to_go(
                params,
                &single,
                &scan,
                &inert,
                config.single_max_len,
                1,
                false,
            );
            for column in through_table(&scan, &costs.p, 1.0) {
                for (key, through) in column {
                    let (i, j) = (key.open.pos, key.close.pos);
                    let at = (
                        i,
                        j,
                        key.nuc_open() as u8,
                        key.nuc_close as u8,
                        states[i as usize],
                        states[j as usize + 1],
                    );
                    let value = lambda * through - (1.0 - lambda) * cai * 100.0;
                    let slot = out.entry(at).or_insert(f64::INFINITY);
                    *slot = slot.min(value);
                }
            }
        }
        Some((out, optimum))
    }

    /// The brute-force motif-free through values are sound and the bound under
    /// them is admissible: their least is the constrained fold's optimum, and
    /// the unconstrained through value of the same pair at the same λ (the
    /// tightest anchor a walk can have) never exceeds any of them. Reports how
    /// many pairs each would rule out at the optimum, the gap a motif-aware
    /// outside closes; the walk's floors at low λ prune on the unconstrained one.
    #[test]
    fn motif_free_through_values_bound_the_unconstrained_ones() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let mut rng = 20_261_019_u64;
        let (mut cases, mut checked) = (0, 0);
        let (mut exact_out, mut free_out, mut pairs, mut paired_optima) = (0, 0, 0, 0);
        for _ in 0..6 {
            const ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
            let len = 4 + next(&mut rng) % 2;
            let protein: String = (0..len)
                .map(|_| ALPHABET[next(&mut rng) % ALPHABET.len()] as char)
                .collect();
            let free = crate::design(&params, &protein, &usage, 0.02)
                .unwrap()
                .sequence;
            let at = next(&mut rng) % (free.len() - 4);
            let motifs = vec![free[at..at + 4].to_string()];
            for lambda in [0.02, 0.3] {
                let Some((through, optimum)) =
                    motif_free_through(&params, &usage, &protein, &motifs, lambda)
                else {
                    continue;
                };
                let aa: Vec<usize> = protein
                    .chars()
                    .map(|c| derna_codon_dp::codon_table::aa_to_index(c).unwrap())
                    .collect();
                let graph = crate::dfa::build(&aa, &usage);
                let ac = crate::ac::Ac::build(&motifs).unwrap();
                let product = ProductDfa::build(&graph, &ac).merge_equivalent();
                let fold = constrained::fold_beamed(
                    &product,
                    &params,
                    lambda,
                    PairConstraint::inert(),
                    MustPairConstraint::inert(),
                    1,
                    crate::Beam::exact(),
                )
                .unwrap();
                let folded = lambda * f64::from(fold.mfe.0) - (1.0 - lambda) * fold.cai * 100.0;
                assert!(
                    (folded - optimum).abs() < 1e-6,
                    "{protein} at λ={lambda}: fold {folded}, enumeration {optimum}"
                );
                // Equal where the optimum pairs anything; an unpaired optimum
                // lies through no pair.
                let least = through.values().copied().fold(f64::INFINITY, f64::min);
                assert!(least >= optimum - 1e-6, "{protein} at λ={lambda}");
                paired_optima += usize::from((least - optimum).abs() < 1e-6);
                let anchor = anchor_at(
                    &params,
                    &graph,
                    lambda,
                    ScanConfig::default(),
                    &PairConstraint::inert(),
                    &MustPairConstraint::inert(),
                );
                let mut unconstrained: crate::hash::FxHashMap<(u32, u32, u8, u8), f64> =
                    Default::default();
                for column in &anchor.table {
                    for (key, v) in column
                        .iter()
                        .map(|(k, b)| (k, b.through))
                        .filter(|&(_, t)| t > f64::NEG_INFINITY)
                    {
                        let at = (
                            key.open.pos,
                            key.close.pos,
                            key.nuc_open() as u8,
                            key.nuc_close as u8,
                        );
                        let slot = unconstrained.entry(at).or_insert(f64::INFINITY);
                        *slot = slot.min(v);
                    }
                }
                for (&(i, j, a, b, _, _), &exact) in &through {
                    let free = unconstrained[&(i, j, a, b)];
                    assert!(
                        free <= exact + constrained::BOUND_SLACK,
                        "{protein} at λ={lambda}: pair ({i}, {j}) unconstrained {free} above motif-free {exact}"
                    );
                    pairs += 1;
                    exact_out += usize::from(exact > optimum + constrained::BOUND_SLACK);
                    free_out += usize::from(free > optimum + constrained::BOUND_SLACK);
                    checked += 1;
                }
                cases += 1;
            }
        }
        eprintln!(
            "cases {cases} ({paired_optima} with a paired optimum): of {pairs} motif-free pairs \
             the exact through rules out {exact_out}, the unconstrained one {free_out}"
        );
        assert!(cases > 0 && checked > 0, "no feasible case");
        assert!(paired_optima > 0, "no optimum paired anything");
    }

    /// The constrained fold's outside pass against enumeration: every closed
    /// pair's inside plus outside is the least objective of a motif-free design
    /// through it, and no pair lies on a design the enumeration does not have.
    /// Some pairs must take their outside from a multi-loop rule, or the
    /// multi-loop half of the pass went untested.
    #[test]
    fn the_outside_pass_is_the_motif_free_through_value() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let mut rng = 20_261_020_u64;
        let (mut cases, mut pairs, mut multiloops) = (0, 0, 0);
        for k in 0..8 {
            // Mostly one-codon residues keep the encodings few, so the proteins
            // can be long enough for a multi-loop to close.
            const TWO: &[u8] = b"CDEFHKNQY";
            let protein: String = (0..24 + 3 * (k % 4))
                .map(|_| match next(&mut rng) % 20 {
                    0..=8 => 'M',
                    9..=16 => 'W',
                    _ => TWO[next(&mut rng) % TWO.len()] as char,
                })
                .collect();
            let free = crate::design(&params, &protein, &usage, 0.02)
                .unwrap()
                .sequence;
            let at = next(&mut rng) % (free.len() - 4);
            let motifs = vec![free[at..at + 4].to_string()];
            for lambda in [0.02, 0.3, 1.0] {
                let Some((oracle, optimum)) =
                    motif_free_through(&params, &usage, &protein, &motifs, lambda)
                else {
                    continue;
                };
                let aa: Vec<usize> = protein
                    .chars()
                    .map(|c| derna_codon_dp::codon_table::aa_to_index(c).unwrap())
                    .collect();
                let graph = crate::dfa::build(&aa, &usage);
                let ac = crate::ac::Ac::build(&motifs).unwrap();
                let product = ProductDfa::build(&graph, &ac).merge_equivalent();
                let found =
                    constrained::outside::through_by_position(&product, &params, lambda).unwrap();
                let through = &found.by_pair;
                assert!(
                    (found.optimum - optimum).abs() < 1e-6,
                    "{protein} at λ={lambda}"
                );
                multiloops += found.through_multiloops;
                for (at, &exact) in &oracle {
                    let got = through.get(at).copied().unwrap_or(f64::INFINITY);
                    assert!(
                        (got - exact).abs() < 1e-6,
                        "{protein} at λ={lambda}, pair {at:?}: outside pass {got}, enumeration {exact}"
                    );
                    pairs += 1;
                }
                for (at, &got) in through {
                    assert!(
                        oracle.contains_key(at),
                        "{protein} at λ={lambda}: pair {at:?} at {got} lies on no motif-free design"
                    );
                }
                cases += 1;
            }
        }
        eprintln!("cases {cases}, pairs {pairs}, outside through a multi-loop {multiloops}");
        assert!(cases > 0 && pairs > 0, "no feasible case with a pair");
        assert!(
            multiloops > 0,
            "no pair took its outside from a multi-loop rule"
        );
    }

    /// A pruned anchor's motif-free outside is a floor: on every pair it holds,
    /// at most the exact value, and equal where the pair has a design within
    /// the ceiling. The ceiling is the open chain's score, feasible by
    /// construction.
    #[test]
    fn a_pruned_motif_free_outside_is_a_floor() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let mut rng = 20_261_021_u64;
        let (mut checked, mut capped) = (0, 0);
        for _ in 0..4 {
            const ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
            let protein: String = (0..8)
                .map(|_| ALPHABET[next(&mut rng) % ALPHABET.len()] as char)
                .collect();
            let free = crate::design(&params, &protein, &usage, 0.02)
                .unwrap()
                .sequence;
            let at = next(&mut rng) % (free.len() - 5);
            let motifs = vec![free[at..at + 5].to_string()];
            let aa: Vec<usize> = protein
                .chars()
                .map(|c| derna_codon_dp::codon_table::aa_to_index(c).unwrap())
                .collect();
            let graph = crate::dfa::build(&aa, &usage);
            let ac = crate::ac::Ac::build(&motifs).unwrap();
            let product = ProductDfa::build(&graph, &ac);
            if !product.is_feasible() {
                continue;
            }
            let product = product.merge_equivalent();
            let open_chain = constrained::open_chain_cai(&product).unwrap();
            for lambda in [0.02, 0.3] {
                let (exact, _) =
                    constrained::outside::motif_free_outside(&product, &params, lambda, 1).unwrap();
                let a = anchor_at(
                    &params,
                    &graph,
                    lambda,
                    ScanConfig::default(),
                    &PairConstraint::inert(),
                    &MustPairConstraint::inert(),
                );
                let ceiling = -(1.0 - lambda) * 100.0 * open_chain + constrained::BOUND_SLACK;
                let pruned = constrained::outside::motif_free_outside_pruned(
                    &product,
                    &params,
                    lambda,
                    OutsideBound {
                        outside: &a.table,
                        chord: None,
                        through: true,
                        zero: &[],
                        motif_free: None,
                        aware: None,
                        ceiling,
                    },
                    1,
                )
                .unwrap();
                for (column, exact) in pruned.iter().zip(&exact) {
                    for (key, &v) in column {
                        let truth = exact.get(key).copied().unwrap_or(f64::INFINITY);
                        assert!(
                            v <= truth + 1e-6,
                            "{protein} at λ={lambda}: {v} above {truth}"
                        );
                        checked += 1;
                        capped += usize::from(v < truth - 1e-6);
                    }
                }
            }
        }
        eprintln!("pairs {checked}, below the exact value {capped}");
        assert!(checked > 0, "no feasible case");
    }

    /// The cap keeps the default where memory allows, takes the most workers
    /// whose peak fits where it does not, and never goes below one; reading
    /// the machine's memory only ever lowers the default.
    #[test]
    fn workers_fit_the_memory_available() {
        let edges = 1746;
        let area = 1746.0_f64 * 1746.0;
        // The memory at which `w` workers fill the budget exactly.
        let peak = |w: f64| {
            ((WALK_BYTES_PER_EDGE2 + w * ANCHOR_BYTES_PER_EDGE2) * area / MEMORY_BUDGET).ceil()
                as u64
        };
        assert_eq!(workers_within(6, u64::MAX, edges), 6);
        assert_eq!(workers_within(6, peak(6.0), edges), 6);
        assert_eq!(workers_within(6, peak(6.0) - 1, edges), 5);
        assert_eq!(workers_within(6, peak(2.5), edges), 2);
        assert_eq!(workers_within(6, peak(0.5), edges), 1);
        assert_eq!(workers_within(6, 0, edges), 1);
        let mut last = 0;
        for gb in 0..64 {
            let w = workers_within(11, gb << 30, edges);
            assert!(w >= last && (1..=11).contains(&w), "{gb} GB gave {w}");
            last = w;
        }
        let graph = crate::dfa::build(&[0, 1, 2, 3], &CodonUsage::load_default().unwrap());
        for anchors in [1, 5, 11] {
            assert!(memory_capped_workers(anchors, &graph) <= default_workers(anchors));
        }
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert!(physical_cores().is_none_or(|p| (1..=logical).contains(&p)));
    }

    /// An anchor from a values-only scan is the anchor from a full scan: the
    /// outside pass reads the tables' values alone, so both give the same
    /// outside and through tables, value for value.
    #[test]
    fn values_only_anchors_are_full_anchors() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let config = ScanConfig::default();
        let (region, must) = (PairConstraint::inert(), MustPairConstraint::inert());
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64 ^ 20_261_020;
        for _ in 0..4 {
            let (protein, _) = case(&mut rng, &params, &usage);
            let aa: Vec<usize> = protein
                .bytes()
                .map(|b| derna_codon_dp::codon_table::aa_to_index(b as char).unwrap())
                .collect();
            let graph = crate::dfa::build(&aa, &usage);
            for lambda in [0.0001, 0.3, 0.99] {
                let values = anchor_at(&params, &graph, lambda, config, &region, &must);
                let full = crate::scan::scan_with_constraints(
                    &params,
                    &graph,
                    lambda,
                    config,
                    region.clone(),
                    must.clone(),
                );
                let costs = OutsideScan::cost_to_go(
                    &params,
                    &graph,
                    &full,
                    &region,
                    config.single_max_len,
                    config.threads,
                    false,
                );
                let table = anchor_table(&full, costs.p, lambda);
                assert_eq!(
                    values.table, table,
                    "{protein} at λ={lambda}: outside and through"
                );
                assert_eq!(
                    values.free,
                    full.c[graph.ix(graph.end())].map(|s| s.brd(lambda)),
                    "{protein} at λ={lambda}: free optimum"
                );
            }
        }
    }

    /// The chain tables are the span fills from the lattice's start and to
    /// its end, state by state.
    #[test]
    fn chain_tables_are_the_span_fills_to_the_ends() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64 ^ 20_261_018;
        for _ in 0..4 {
            let (protein, motifs) = case(&mut rng, &params, &usage);
            let aa: Vec<usize> = protein
                .bytes()
                .map(|b| derna_codon_dp::codon_table::aa_to_index(b as char).unwrap())
                .collect();
            let graph = crate::dfa::build(&aa, &usage);
            let ac = crate::ac::Ac::build(&motifs).unwrap();
            let product = ProductDfa::build(&graph, &ac);
            if !product.is_feasible() {
                continue;
            }
            let product = product.merge_equivalent();
            let chain = ChainCai::build(&product);
            let (dfa, lac) = (product.dfa(), product.lattice_ac());
            for pos in 0..=dfa.seq_len {
                for &node in dfa.nodes_at(pos) {
                    let head = constrained::span_fill(dfa, lac, dfa.start(), lac.start(), node);
                    for &(q, cai) in &head {
                        let tail = constrained::span_fill(dfa, lac, node, q, dfa.end())
                            .into_iter()
                            .map(|(_, c)| c)
                            .reduce(f64::max);
                        let around = chain.around(node, q, node, q);
                        match tail {
                            Some(t) => {
                                let got = around.expect("a state on a motif-free path");
                                assert!((got - (cai + t)).abs() < 1e-9, "{protein} at {node:?}");
                            }
                            None => assert!(around.is_none(), "{protein} at {node:?}"),
                        }
                    }
                }
            }
        }
    }

    /// A fold pruning on the chords to λ = 0 reaches the optimum under the
    /// tightest ceiling there is, the optimum itself, where an inadmissible
    /// floor would cut a cell of the optimal design. And the chords change
    /// what some fold discards, so the agreement is not the two folds being
    /// the same fold.
    #[test]
    fn zero_chords_keep_the_optimum_and_change_the_discards() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let config = ScanConfig::default();
        let (region, must) = (PairConstraint::inert(), MustPairConstraint::inert());
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64 ^ 20_261_019;
        let (mut chorded, mut sharper) = (0, 0);
        for _ in 0..10 {
            let (protein, motifs) = case(&mut rng, &params, &usage);
            let aa: Vec<usize> = protein
                .bytes()
                .map(|b| derna_codon_dp::codon_table::aa_to_index(b as char).unwrap())
                .collect();
            let graph = crate::dfa::build(&aa, &usage);
            let ac = crate::ac::Ac::build(&motifs).unwrap();
            let product = ProductDfa::build(&graph, &ac);
            if !product.is_feasible() {
                continue;
            }
            let product = product.merge_equivalent();
            let chain = ChainCai::build(&product);
            let open_chain = constrained::open_chain_cai(&product);
            // An end on its own anchor, and a λ inside a low bracket.
            for (lambda, lams) in [(0.0001, [0.0001, 0.99]), (0.01, [0.0001, 0.05])] {
                let anchors: Vec<Anchor> = lams
                    .iter()
                    .map(|&l| anchor_at(&params, &graph, l, config, &region, &must))
                    .collect();
                let exact = constrained::fold_beamed(
                    &product,
                    &params,
                    lambda,
                    region.clone(),
                    must.clone(),
                    1,
                    crate::Beam::exact(),
                )
                .expect("a feasible product");
                let score = |mfe: Energy, cai: f64| {
                    lambda * f64::from(mfe.0) - (1.0 - lambda) * cai * 100.0
                };
                let optimum = score(exact.mfe, exact.cai);
                let (h, l, w) = bracket(&lams, lambda);
                let free = free_chord(anchors[h].free, anchors[l].free, w);
                let zero = zero_chords(&chain, open_chain, &lams, h, lambda, free, |u| {
                    (&anchors[u].table[..], anchors[u].free)
                });
                let fold = |zero: &[ZeroChord]| {
                    let bound = OutsideBound {
                        outside: &anchors[h].table,
                        chord: (h != l).then(|| (&anchors[l].table[..], w)),
                        through: true,
                        zero,
                        motif_free: None,
                        aware: None,
                        ceiling: optimum,
                    };
                    constrained::fold_bounded(
                        &product,
                        &params,
                        lambda,
                        region.clone(),
                        must.clone(),
                        1,
                        bound,
                        &mut None,
                    )
                };
                let (plain, plain_counts) = fold(&[]);
                let (with, counts) = fold(&zero);
                for f in [plain, with] {
                    let f = f.expect("every cell of the optimum survives its own ceiling");
                    assert!(
                        (score(f.mfe, f.cai) - optimum).abs() < 1e-6,
                        "{protein} at λ={lambda}: {} against the optimum {optimum}",
                        score(f.mfe, f.cai)
                    );
                }
                if !zero.is_empty() {
                    chorded += 1;
                    sharper += usize::from(counts != plain_counts);
                }
            }
        }
        assert!(chorded > 0, "no fold took a chord to λ = 0");
        assert!(
            sharper > 0,
            "the chords never changed what a fold discarded"
        );
    }
}
