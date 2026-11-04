//! The modularity check behind [`super::first_modular`].
//!
//! The walk hands designs, in walk order, to a bounded queue; `threads`
//! workers take them from the front. A worker first offers its design to the
//! shared [`WitnessPool`] and folds it exactly only when no witness rejects it;
//! a fold that rejects adds its free structure to the pool. A worker scores
//! against a snapshot of the pool, taken under its lock and released before
//! scoring, so an addition never waits for another worker's scoring. Because a design
//! meets the pool when a worker takes it rather than when the walk produces
//! it, a design queued behind one that fell through to a fold still sees that
//! fold's structure if it finished first.
//!
//! The answer does not depend on the scheduling. A modular design
//! found at rank `r` stops the walk, and the workers still settle every queued
//! design ranked below `r`; the least-ranked one found is the answer, the same
//! as a one-at-a-time check in walk order. Which designs a witness rejects,
//! and so the counters, vary from run to run.
//!
//! Under an adaptation window ([`super::window`]) the check collects instead:
//! every modular design is kept and none stops the walk, which
//! the window's own rule ends. A design is skipped only when one found has
//! at most its constrained energy and at least its adaptation, and so dominates
//! it.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use super::window::Floor;
use super::{ExactFold, LazyError, Verify};
use crate::fold_cut::{CachedFold, FoldCut};
use crate::free_fold::{
    sequence_free_fold, sequence_free_fold_zuker, sequence_free_mfe, sequence_gap, CdsWindow,
    FreeFoldError,
};
use crate::region::RegionConstraints;
use crate::witness::{pairs_of, Pairs, Scorer, WitnessPool};
use crate::Params;
use rna_mfe::dp::{fold_sparse, Cut, SparseFold};

/// What checking a design needs besides the design.
#[derive(Clone, Copy)]
pub(super) struct Context<'a> {
    pub params: &'a Params,
    pub cds: CdsWindow,
    pub constraints: RegionConstraints,
    /// Whether the walk's constrained MFE is the one the constrained fold
    /// would return; when not, a design costs both folds and no witness.
    pub walk_energy_usable: bool,
    /// The exact free fold, and the cut's lattice when it is
    /// [`ExactFold::SparseCut`]. The lattice may land while the check runs
    /// ([`super::CutSchedule::Background`]); until it does, a fold under the
    /// cut is the exact uncut Zuker fold.
    pub fold: ExactFold,
    pub cut: &'a OnceLock<FoldCut>,
    /// How far, in dekacalories, a design's free MFE may sit below its
    /// constrained MFE and the design still pass: `0` passes the modular
    /// designs only. Every rejection — a witness, a windowed fold, the
    /// exact fold — tests against the constrained MFE less this, so the
    /// check passes exactly the designs whose gap is at most it.
    pub slack: i32,
}

impl Context<'_> {
    /// The energy a free structure of a design with constrained MFE `e` must
    /// lie below to reject it.
    fn threshold(&self, e: i32) -> i32 {
        e - self.slack
    }
}

/// How the check went: the answer, and where the designs and seconds went.
pub(super) struct Outcome {
    pub modular: Option<(f64, String, u64)>,
    pub witness_rejects: u64,
    pub folds: u64,
    /// Designs ranked past an answer already found, never checked.
    pub skipped: u64,
    /// Worker-seconds offering designs to the pool.
    pub witness_s: f64,
    /// Worker-seconds folding.
    pub fold_s: f64,
    /// Seconds the walk waited on a full queue.
    pub wait_s: f64,
    /// Worker-seconds outside `witness_s` and `fold_s`: taking a design off the
    /// queue and asking whether it is already passed (`queue_s`), waiting there
    /// for the walk to queue one (`starve_s`), and adding a rejecting fold's
    /// structure to the pool (`pool_add_s`). Of `witness_s`, the seconds spent
    /// taking a snapshot of the pool (`pool_read_s`) and adding a rejecting
    /// structure to it (`pool_write_s`).
    pub queue_s: f64,
    pub starve_s: f64,
    pub pool_add_s: f64,
    pub pool_read_s: f64,
    pub pool_write_s: f64,
    /// Witness rejections by the rejecting candidate's index in scoring order
    /// ([`WitnessPool::reject_ranked`]), in buckets `0, 1, 2–3, 4–7, …,
    /// 64–127`; and how many of them were a witness spliced into the design's
    /// own structure rather than the witness alone.
    pub witness_rank: [u64; RANK_BUCKETS],
    pub witness_spliced: u64,
    /// Under [`Verify::fold_reuse`], folds that copied cells from a recent
    /// fold, and the cells they copied.
    pub reuse_hits: u64,
    pub cells_copied: u64,
    /// Under [`Verify::fold_slots`], designs that waited for a slot, and of
    /// them those a finished fold's structure rejected.
    pub deferred: u64,
    pub deferred_rejected: u64,
    /// Folds under [`ExactFold::SparseCut`] made before its lattice landed,
    /// by the uncut Zuker fold.
    pub uncut_folds: u64,
    /// Under [`Verify::window`], designs a windowed fold rejected (also in
    /// `witness_rejects`), and worker-seconds in windowed folds.
    pub window_rejects: u64,
    pub window_s: f64,
    /// Under [`Verify::window_reuse`], cells windowed folds copied from
    /// earlier ones, and windowed structures an exact re-score did not
    /// confirm below the design's constrained MFE (expected none).
    pub window_cells_copied: u64,
    pub window_unconfirmed: u64,
    /// Under [`Verify::window_auto`], the design (by rank) whose full fold
    /// switched the UTR pairs on, and the designs checked when window reuse
    /// switched on; `None` when a switch never fired (or was forced on).
    pub window_utr_from: Option<u64>,
    pub window_reuse_from: Option<u64>,
    /// Under an adaptation window, every modular design found,
    /// in the order the workers found them.
    pub found: Vec<Found>,
}

/// A modular design a collecting check found.
#[derive(Debug, Clone)]
pub(super) struct Found {
    /// Derivations the bound had cut when the walk reached the design.
    pub cut_before: u64,
    pub seq: String,
    pub energy: Option<i32>,
    pub cai: f64,
    pub rank: u64,
}

/// Buckets of [`Outcome::witness_rank`]: index `0`, then one per power of two
/// up to the pool's 64 witnesses × 2 candidates.
pub(super) const RANK_BUCKETS: usize = 8;

fn rank_bucket(index: usize) -> usize {
    (usize::BITS - index.leading_zeros()).min(RANK_BUCKETS as u32 - 1) as usize
}

/// Closes [`Shared`]'s queue when dropped, unwinding included.
struct Closing<'s, 'a>(&'s Shared<'a>);

impl Drop for Closing<'_, '_> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// The walk's visitor, as the check sees it: returns `true` to stop.
pub(super) type Offer<'o> = dyn FnMut(f64, &str, Option<i32>, &[(u32, u32)]) -> bool + Send + 'o;

/// [`Offer`], with the design's adaptation and the derivations the bound
/// had cut when the walk reached the design.
pub(super) type OfferCai<'o> =
    dyn FnMut(f64, &str, Option<i32>, &[(u32, u32)], f64, u64) -> bool + Send + 'o;

/// A design waiting for a worker, with its rank in the walk.
struct Design {
    objective: f64,
    seq: String,
    energy: Option<i32>,
    own: Pairs,
    rank: u64,
    /// Its adaptation, `NaN` outside a window.
    cai: f64,
    /// Under a window, derivations the bound had cut when the walk reached it.
    cut_before: u64,
}

/// The fold slots of [`Verify::fold_slots`]: how many are folding, and the
/// designs waiting for one, by rank. A slot is given up only when nothing
/// waits, under the same lock a design takes to wait, so no design is left
/// waiting with every slot free.
struct Slots {
    slots: usize,
    folding: usize,
    waiting: BTreeMap<u64, Design>,
}

#[derive(Default)]
struct Queue {
    items: VecDeque<Design>,
    closed: bool,
    modular: Option<(f64, String, u64)>,
    error: Option<FreeFoldError>,
}

impl Queue {
    /// Whether the walk should stop producing designs.
    fn settled(&self) -> bool {
        self.modular.is_some() || self.error.is_some()
    }
}

struct Shared<'a> {
    ctx: Context<'a>,
    /// The pool behind a lock held only to take a snapshot or to add.
    pool: Mutex<Arc<WitnessPool>>,
    witness_stop: usize,
    /// Move a rejecting candidate's source witness to the pool's front.
    witness_promote: bool,
    /// Recent folds to copy from, under [`Verify::fold_reuse`].
    recent: Option<Recent>,
    /// Recent windowed folds to copy from, under [`Verify::window_reuse`]
    /// or [`Verify::window_auto`]: one cache per window predicate (without
    /// and with the UTR pairs), since a fold copies only from folds that
    /// admitted the same pairs.
    recent_window: Option<[Recent; 2]>,
    /// Whether windowed folds copy from `recent_window` now.
    window_reuse_on: AtomicBool,
    window_reuse_from: AtomicU64,
    queue: Mutex<Queue>,
    /// The rank of the best answer found (`u64::MAX` for none), and whether
    /// a fold failed: [`Queue::modular`] and [`Queue::error`] as a worker
    /// reads them before each design, without the queue's lock.
    modular_rank: AtomicU64,
    failed: AtomicBool,
    /// Designs moved between the walk and the queue, and between the queue
    /// and a worker, per lock ([`Verify::handoff_batch`]).
    batch: usize,
    /// Under [`Verify::fold_slots`].
    slots: Option<Mutex<Slots>>,
    deferred: AtomicU64,
    deferred_rejected: AtomicU64,
    /// Signalled when a design is queued or the queue closes.
    ready: Condvar,
    /// Signalled when a design leaves the queue or the check settles.
    room: Condvar,
    cap: usize,
    witness_rejects: AtomicU64,
    folds: AtomicU64,
    skipped: AtomicU64,
    witness_ns: AtomicU64,
    fold_ns: AtomicU64,
    queue_ns: AtomicU64,
    starve_ns: AtomicU64,
    pool_add_ns: AtomicU64,
    pool_read_ns: AtomicU64,
    pool_write_ns: AtomicU64,
    witness_rank: [AtomicU64; RANK_BUCKETS],
    witness_spliced: AtomicU64,
    uncut_folds: AtomicU64,
    /// Under [`Verify::window`]; whether windowed folds admit the UTR pairs
    /// now ([`Verify::window_utr`], or switched on by [`Verify::window_auto`]);
    /// windowed folds tried.
    window: Option<(usize, usize)>,
    window_auto: bool,
    window_utr_on: AtomicBool,
    window_utr_from: AtomicU64,
    window_calls: AtomicU64,
    window_rejects: AtomicU64,
    window_ns: AtomicU64,
    window_unconfirmed: AtomicU64,
    /// Under a window: the designs found, and the best adaptation among them,
    /// which the walk reads to raise its floor.
    collect: Option<&'a Floor>,
    found: Mutex<Vec<Found>>,
}

fn nanos(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

impl Shared<'_> {
    /// Queue the designs in `batch`, waiting while the queue is full, and
    /// empty it. `true` once the check has settled, so the walk can stop.
    fn offer(&self, batch: &mut Vec<Design>, wait_ns: &mut u64) -> bool {
        if batch.is_empty() {
            return false;
        }
        let t = Instant::now();
        let mut q = self.queue.lock().expect("the queue");
        while q.items.len() >= self.cap && !q.settled() {
            q = self.room.wait(q).expect("the queue");
        }
        *wait_ns += nanos(t);
        if q.settled() {
            self.skipped.fetch_add(batch.len() as u64, Relaxed);
            batch.clear();
            return true;
        }
        let many = batch.len() > 1;
        q.items.extend(batch.drain(..));
        drop(q);
        if many {
            self.ready.notify_all();
        } else {
            self.ready.notify_one();
        }
        false
    }

    /// No more designs: let the workers drain the queue and stop.
    fn close(&self) {
        self.queue.lock().expect("the queue").closed = true;
        self.ready.notify_all();
    }

    /// The next queued designs, up to a quarter of [`Self::batch`] and at
    /// least one, into `into`; `false` once the queue is closed and empty.
    fn next(&self, into: &mut VecDeque<Design>) -> bool {
        let t = Instant::now();
        let mut starved = 0u64;
        let mut q = self.queue.lock().expect("the queue");
        let take = (self.batch / 4).max(1);
        let got = loop {
            if !q.items.is_empty() {
                let n = take.min(q.items.len());
                into.extend(q.items.drain(..n));
                break true;
            }
            if q.closed {
                break false;
            }
            let w = Instant::now();
            q = self.ready.wait(q).expect("the queue");
            starved += nanos(w);
        };
        drop(q);
        if got {
            self.room.notify_one();
        }
        self.starve_ns.fetch_add(starved, Relaxed);
        self.queue_ns.fetch_add(nanos(t) - starved, Relaxed);
        got
    }

    /// Whether an answer ranked before `rank` is already known, or a fold
    /// failed.
    fn passed(&self, rank: u64) -> bool {
        self.failed.load(Relaxed) || self.modular_rank.load(Relaxed) < rank
    }

    /// Whether `d` need not be checked: [`Self::passed`], or under a window
    /// a design found has at most its constrained energy and at least its
    /// adaptation.
    fn skip(&self, d: &Design) -> bool {
        self.passed(d.rank)
            || self
                .collect
                .is_some_and(|floor| d.energy.is_some_and(|e| floor.dominates(e, d.cai)))
    }

    fn work(&self) {
        let mut mine = VecDeque::new();
        while self.next(&mut mine) {
            while let Some(d) = mine.pop_front() {
                if self.skip(&d) {
                    self.skipped.fetch_add(1, Relaxed);
                } else if !self.witness_rejects(&d) && !self.window_rejects(&d) {
                    self.settle(d);
                }
            }
        }
    }

    /// Fold `d`, or under [`Verify::fold_slots`] with every slot folding,
    /// leave it waiting; a slot folds the waiting designs until none is left.
    fn settle(&self, d: Design) {
        let Some(slots) = &self.slots else {
            self.fold(d);
            return;
        };
        {
            let mut s = slots.lock().expect("the fold slots");
            if s.folding >= s.slots {
                s.waiting.insert(d.rank, d);
                self.deferred.fetch_add(1, Relaxed);
                return;
            }
            s.folding += 1;
        }
        let mut next = d;
        loop {
            if let Some(structure) = self.fold(next) {
                self.offer_waiting(slots, &structure);
            }
            let mut s = slots.lock().expect("the fold slots");
            loop {
                let Some((_, d)) = s.waiting.pop_first() else {
                    s.folding -= 1;
                    return;
                };
                if self.skip(&d) {
                    self.skipped.fetch_add(1, Relaxed);
                } else {
                    next = d;
                    break;
                }
            }
        }
    }

    /// Offer a finished fold's structure to every waiting design, outside the
    /// lock; those it rejects are settled, the rest wait again.
    fn offer_waiting(&self, slots: &Mutex<Slots>, structure: &Pairs) {
        let waiting = std::mem::take(&mut slots.lock().expect("the fold slots").waiting);
        if waiting.is_empty() {
            return;
        }
        let mut one = WitnessPool::new(1);
        one.add(structure.clone());
        let mut kept = BTreeMap::new();
        for (rank, d) in waiting {
            let rejected = d
                .energy
                .filter(|_| self.ctx.walk_energy_usable)
                .and_then(|e| {
                    let t = self.ctx.threshold(e);
                    one.reject_ranked(self.ctx.params, d.seq.as_bytes(), t, &d.own, 1)
                });
            match rejected {
                Some((w, pairs, _)) => {
                    debug_assert!(
                        sequence_free_mfe(self.ctx.params, &d.seq).is_ok_and(|f| f.0 <= w),
                        "a fold's structure scored below the exact free MFE"
                    );
                    self.add(pairs);
                    self.witness_rejects.fetch_add(1, Relaxed);
                    self.deferred_rejected.fetch_add(1, Relaxed);
                }
                None => {
                    kept.insert(rank, d);
                }
            }
        }
        slots
            .lock()
            .expect("the fold slots")
            .waiting
            .append(&mut kept);
    }

    /// Under [`Verify::window`], whether a fold restricted to the window's
    /// pairs and `d`'s own pairs finds a free structure below `d`'s
    /// constrained MFE less the slack; it joins the pool.
    fn window_rejects(&self, d: &Design) -> bool {
        let Some(window) = self.window else {
            return false;
        };
        let Some(e) = d.energy.filter(|_| self.ctx.walk_energy_usable) else {
            return false;
        };
        let t = Instant::now();
        self.maybe_start_window_reuse();
        let utr = self.window_utr_on.load(Relaxed);
        let recent = self
            .recent_window
            .as_ref()
            .filter(|_| self.window_reuse_on.load(Relaxed))
            .map(|caches| &caches[usize::from(utr)]);
        let e = self.ctx.threshold(e);
        let found = window_certificate(&self.ctx, &d.seq, &d.own, (window, utr), e, recent);
        self.window_calls.fetch_add(1, Relaxed);
        // A fold that copied cells is confirmed by scoring its structure on
        // the design exactly: only a structure truly below `e` rejects.
        let found = found.filter(|(_, pairs)| {
            if recent.is_none() {
                return true;
            }
            let ok = Scorer::new(self.ctx.params, d.seq.as_bytes())
                .score_constructed(pairs)
                .is_some_and(|x| x < e);
            debug_assert!(ok, "a reusing windowed fold's structure is not below e");
            if !ok {
                self.window_unconfirmed.fetch_add(1, Relaxed);
            }
            ok
        });
        self.window_ns.fetch_add(nanos(t), Relaxed);
        let Some((w, pairs)) = found else {
            return false;
        };
        debug_assert!(
            sequence_free_mfe(self.ctx.params, &d.seq).is_ok_and(|f| f.0 <= w && w < e),
            "a windowed fold's structure scored below the exact free MFE, or not below e"
        );
        self.add(pairs);
        self.witness_rejects.fetch_add(1, Relaxed);
        self.window_rejects.fetch_add(1, Relaxed);
        true
    }

    /// Whether a witness in the pool rejects `d`: one scores below its
    /// constrained MFE less the slack. A rejecting structure joins the pool.
    fn witness_rejects(&self, d: &Design) -> bool {
        let Some(e) = d.energy.filter(|_| self.ctx.walk_energy_usable) else {
            return false;
        };
        WITNESS_LOG.write(|| {
            let own = crate::witness::dot_bracket(d.seq.len(), &d.own);
            format!("{}\t{e}\t{}\t{own}\n", d.rank, d.seq)
        });
        let t = Instant::now();
        let pool = Arc::clone(&self.pool.lock().expect("the pool"));
        self.pool_read_ns.fetch_add(nanos(t), Relaxed);
        let (params, seq) = (self.ctx.params, d.seq.as_bytes());
        let found = pool.reject_ranked(
            params,
            seq,
            self.ctx.threshold(e),
            &d.own,
            self.witness_stop,
        );
        let source = found
            .as_ref()
            .filter(|_| self.witness_promote)
            .and_then(|(_, _, index)| pool.witness(index / 2));
        drop(pool);
        let hit = found.is_some();
        if let Some((w, pairs, index)) = found {
            self.witness_rank[rank_bucket(index)].fetch_add(1, Relaxed);
            self.witness_spliced.fetch_add((index % 2) as u64, Relaxed);
            debug_assert!(
                sequence_free_mfe(self.ctx.params, &d.seq).is_ok_and(|f| f.0 <= w),
                "a witness scored below the exact free MFE"
            );
            let w = Instant::now();
            self.add_promoting(pairs, source.as_ref());
            self.pool_write_ns.fetch_add(nanos(w), Relaxed);
            self.witness_rejects.fetch_add(1, Relaxed);
        }
        self.witness_ns.fetch_add(nanos(t), Relaxed);
        hit
    }

    /// Add a rejecting structure to the pool, copying the pool's list of
    /// shared witnesses when a snapshot of it is still being scored.
    fn add(&self, pairs: Pairs) {
        Arc::make_mut(&mut self.pool.lock().expect("the pool")).add(pairs);
    }

    /// [`Self::add`], first moving the witness that produced `pairs` to the
    /// front when [`Verify::witness_promote`] names it.
    fn add_promoting(&self, pairs: Pairs, source: Option<&Arc<Pairs>>) {
        let mut constraint = self.pool.lock().expect("the pool");
        let pool = Arc::make_mut(&mut constraint);
        if let Some(s) = source {
            pool.promote(s);
        }
        pool.add(pairs);
    }

    /// Under [`Verify::window_auto`], start copying windowed folds once they
    /// are frequent: past [`REUSE_AFTER`] designs checked, when more than one
    /// in [`REUSE_SHARE`] reached a windowed fold.
    fn maybe_start_window_reuse(&self) {
        if !self.window_auto || self.recent_window.is_none() || self.window_reuse_on.load(Relaxed) {
            return;
        }
        let checked = self.witness_rejects.load(Relaxed) + self.folds.load(Relaxed);
        if reuse_due(checked, self.window_calls.load(Relaxed))
            && !self.window_reuse_on.swap(true, Relaxed)
        {
            self.window_reuse_from.store(checked, Relaxed);
        }
    }

    /// Under [`Verify::window_auto`], switch the UTR pairs on when a design
    /// the window let through was rejected by a structure that needs one.
    fn note_full_rejection(&self, d: &Design, rejecting: &[(u32, u32)]) {
        let Some(window) = self.window.filter(|_| self.window_auto) else {
            return;
        };
        if self.window_utr_on.load(Relaxed) {
            return;
        }
        if utr_miss(self.ctx.cds, d.seq.len(), window, &d.own, rejecting)
            && !self.window_utr_on.swap(true, Relaxed)
        {
            self.window_utr_from.store(d.rank, Relaxed);
        }
    }

    /// Settle `d` with an exact fold; the free structure that rejected it,
    /// when one did.
    fn fold(&self, d: Design) -> Option<Pairs> {
        let t = Instant::now();
        let cut = self.ctx.cut.get();
        if cut.is_none()
            && matches!(
                self.ctx.fold,
                ExactFold::SparseCut | ExactFold::SparseCutInside
            )
        {
            self.uncut_folds.fetch_add(1, Relaxed);
        }
        let verdict = verdict(&self.ctx, cut, self.recent.as_ref(), &d);
        let ns = nanos(t);
        self.fold_ns.fetch_add(ns, Relaxed);
        self.folds.fetch_add(1, Relaxed);
        log_fold(&d, ns);
        match verdict {
            Ok((false, witness)) => {
                let pairs = witness?;
                self.note_full_rejection(&d, &pairs);
                let w = Instant::now();
                self.add(pairs.clone());
                self.pool_add_ns.fetch_add(nanos(w), Relaxed);
                Some(pairs)
            }
            Ok((true, _)) if self.collect.is_some() => {
                let cai = d.cai;
                self.found.lock().expect("the found designs").push(Found {
                    cut_before: d.cut_before,
                    seq: d.seq,
                    energy: d.energy,
                    cai,
                    rank: d.rank,
                });
                if let Some(e) = d.energy {
                    self.collect.expect("a collecting check").record(e, cai);
                }
                None
            }
            Ok((true, _)) => {
                let mut q = self.queue.lock().expect("the queue");
                if q.modular.as_ref().is_none_or(|h| d.rank < h.2) {
                    self.modular_rank.store(d.rank, Relaxed);
                    q.modular = Some((d.objective, d.seq, d.rank));
                }
                self.room.notify_all();
                None
            }
            Err(e) => {
                self.queue.lock().expect("the queue").error.get_or_insert(e);
                self.failed.store(true, Relaxed);
                self.room.notify_all();
                None
            }
        }
    }
}

/// The last folds under the cut, shared by the workers, for a new fold to
/// copy the spans it shares with the most similar one
/// ([`FoldCut::fold_reusing`]).
struct Recent {
    folds: Mutex<VecDeque<Arc<CachedFold>>>,
    cap: usize,
    /// Copy each row from whichever recent fold reaches furthest.
    per_row: bool,
    /// Folds that copied cells, and the cells they copied.
    hits: AtomicU64,
    copied: AtomicU64,
}

impl Recent {
    fn new(cap: usize, per_row: bool) -> Self {
        Self {
            folds: Mutex::new(VecDeque::with_capacity(cap)),
            cap,
            per_row,
            hits: AtomicU64::new(0),
            copied: AtomicU64::new(0),
        }
    }

    /// Fold `seq` under `cut` at `threshold`, copying from a recent fold,
    /// and file this fold as the newest.
    fn fold(
        &self,
        cut: &FoldCut,
        params: &Params,
        seq: &[u8],
        threshold: i32,
    ) -> Option<SparseFold> {
        let earlier: Vec<Arc<CachedFold>> = self
            .folds
            .lock()
            .expect("the recent folds")
            .iter()
            .cloned()
            .collect();
        let (verdict, cached, copied) =
            cut.fold_reusing(params, seq, threshold, &earlier, self.per_row)?;
        drop(earlier);
        self.hits.fetch_add(u64::from(copied > 0), Relaxed);
        self.copied.fetch_add(copied, Relaxed);
        let mut folds = self.folds.lock().expect("the recent folds");
        folds.push_front(Arc::new(cached));
        folds.truncate(self.cap);
        Some(verdict)
    }

    /// A windowed fold of `seq` under `cut` over the pairs `allowed` admits,
    /// copying from a recent windowed fold the spans where the two designs
    /// spell alike and their own structures (`mate`, each position's
    /// partner) agree, and filing this fold as the newest.
    fn fold_window(
        &self,
        cut: &FoldCut,
        params: &Params,
        seq: &[u8],
        threshold: i32,
        allowed: &dyn Fn(usize, usize) -> bool,
        mate: &[u32],
    ) -> Option<SparseFold> {
        let earlier: Vec<Arc<CachedFold>> = self
            .folds
            .lock()
            .expect("the recent folds")
            .iter()
            .cloned()
            .collect();
        let (verdict, cached, copied) = cut.fold_restricted_reusing(
            params,
            seq,
            threshold,
            allowed,
            mate,
            &earlier,
            self.per_row,
        )?;
        drop(earlier);
        self.hits.fetch_add(u64::from(copied > 0), Relaxed);
        self.copied.fetch_add(copied, Relaxed);
        let mut folds = self.folds.lock().expect("the recent folds");
        folds.push_front(Arc::new(cached));
        folds.truncate(self.cap);
        Some(verdict)
    }
}

/// An opt-in record of the designs the check sees, for profiling it offline
/// on the stream only a walk produces: the file the environment variable
/// `var` names, its first `{var}_MAX` lines (default `max`).
struct Log {
    var: &'static str,
    max: u64,
    file: std::sync::OnceLock<Option<(Mutex<std::fs::File>, u64, u64)>>,
    lines: AtomicU64,
}

impl Log {
    const fn new(var: &'static str, max: u64) -> Self {
        Self {
            var,
            max,
            file: std::sync::OnceLock::new(),
            lines: AtomicU64::new(0),
        }
    }

    /// Append `line()` for the lines past `<VAR>_SKIP` (default 0) and under
    /// the cap `<VAR>_MAX`; nothing when the variable is unset.
    fn write(&self, line: impl FnOnce() -> String) {
        use std::io::Write;
        let open = self.file.get_or_init(|| {
            let path = std::env::var_os(self.var)?;
            let cap = std::env::var(format!("{}_MAX", self.var))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.max);
            let skip = std::env::var(format!("{}_SKIP", self.var))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()?;
            Some((Mutex::new(file), skip, cap))
        });
        let Some((file, skip, cap)) = open else {
            return;
        };
        let line_no = self.lines.fetch_add(1, Relaxed);
        if line_no < *skip || line_no - *skip >= *cap {
            return;
        }
        let _ = file.lock().expect("the log").write_all(line().as_bytes());
    }
}

/// `DERNA_FOLD_LOG`: each folded design as `rank<TAB>constrained MFE<TAB>fold
/// ns<TAB>sequence` — the designs the witness pool let through.
static FOLD_LOG: Log = Log::new("DERNA_FOLD_LOG", 2_000);

/// `DERNA_WITNESS_LOG`: each design offered to the witness pool as
/// `rank<TAB>constrained MFE<TAB>sequence<TAB>its constrained structure`.
static WITNESS_LOG: Log = Log::new("DERNA_WITNESS_LOG", 20_000);

fn log_fold(d: &Design, ns: u64) {
    FOLD_LOG.write(|| {
        let energy = d.energy.map_or("NA".to_string(), |e| e.to_string());
        format!("{}\t{energy}\t{ns}\t{}\n", d.rank, d.seq)
    });
}

/// Designs checked before [`Verify::window_auto`] may start window reuse.
pub(super) const REUSE_AFTER: u64 = 100_000;
/// Reuse starts when more than one checked design in this many reached a
/// windowed fold (0.5%).
pub(super) const REUSE_SHARE: u64 = 200;

/// Whether windowed folds are frequent enough to start copying them.
fn reuse_due(checked: u64, windowed: u64) -> bool {
    checked >= REUSE_AFTER && windowed * REUSE_SHARE > checked
}

/// Whether `rejecting`, a free structure that beat a design the window let
/// through, uses a pair the UTR pairs admit and the window does not: one
/// with an end in a UTR and the other outside the window, not the design's
/// own.
fn utr_miss(
    cds: CdsWindow,
    n: usize,
    (w5, w3): (usize, usize),
    own: &[(u32, u32)],
    rejecting: &[(u32, u32)],
) -> bool {
    let lo = (cds.start + w5).min(n);
    let hi = cds.end.saturating_sub(w3).max(lo);
    let near = |i: usize| i < lo || i >= hi;
    let in_utr = |i: usize| i < cds.start || i >= cds.end;
    rejecting.iter().any(|&(i, j)| {
        let (i, j) = (i as usize, j as usize);
        (in_utr(i) || in_utr(j)) && !(near(i) && near(j)) && !own.contains(&(i as u32, j as u32))
    })
}

/// One design's exact verdict, and its free structure when it is rejected.
/// A free structure of `seq` below `threshold`, and its energy, using only
/// pairs with both ends in the window — the UTRs, the first `w5` and the
/// last `w3` CDS positions — or among `own`'s pairs; with the cut's lattice
/// in hand its outside bound prunes them too; `None` when the restricted MFE is
/// at or above `threshold`. Every such structure is a free structure, so one
/// below a design's constrained MFE rejects it. A design's free optimum
/// usually differs from its own structure only near the flanks, so this
/// sparse fold, which visits only the allowed pairs, often finds a
/// rejection a full fold would.
fn window_certificate(
    ctx: &Context<'_>,
    seq: &str,
    own: &[(u32, u32)],
    ((w5, w3), utr): ((usize, usize), bool),
    threshold: i32,
    recent: Option<&Recent>,
) -> Option<(i32, Pairs)> {
    const OPEN: i32 = -(1 << 28);
    const SHUT: i32 = 1 << 28;
    let params = ctx.params;
    let nt = rna_mfe::encode_seq(seq.as_bytes()).ok()?;
    let n = nt.len();
    let lo = (ctx.cds.start + w5).min(n);
    let hi = ctx.cds.end.saturating_sub(w3).max(lo);
    let mut mate = vec![u32::MAX; n];
    for &(i, j) in own {
        mate[i as usize] = j;
        mate[j as usize] = i;
    }
    let near = |i: usize| i < lo || i >= hi;
    let in_utr = |i: usize| i < ctx.cds.start || i >= ctx.cds.end;
    let allowed = |i: usize, j: usize| {
        (near(i) && near(j)) || mate[i] as usize == j || (utr && (in_utr(i) || in_utr(j)))
    };
    if let Some(cut) = ctx.cut.get() {
        let verdict = match recent {
            Some(r) => r.fold_window(cut, params, seq.as_bytes(), threshold, &allowed, &mate)?,
            None => cut.fold_restricted(params, seq.as_bytes(), threshold, &allowed)?,
        };
        return match verdict {
            SparseFold::Below(f, db) => Some((f.0, pairs_of(&db))),
            SparseFold::AtOrAbove => None,
        };
    }
    let inside = |i: usize, j: usize| if allowed(i, j) { OPEN } else { SHUT };
    let outside = |_: usize, _: usize| OPEN;
    let cut = Cut {
        threshold,
        outside: &outside,
        inside: Some(&inside),
    };
    match fold_sparse(params, &nt, Some(&cut)) {
        SparseFold::Below(f, db) => Some((f.0, pairs_of(&db))),
        SparseFold::AtOrAbove => None,
    }
}

fn verdict(
    ctx: &Context<'_>,
    cut: Option<&FoldCut>,
    recent: Option<&Recent>,
    d: &Design,
) -> Result<(bool, Option<Pairs>), FreeFoldError> {
    match d.energy {
        Some(e) if ctx.walk_energy_usable => {
            let free = match (ctx.fold, cut) {
                (ExactFold::LinearFold, _) => Some(sequence_free_fold(ctx.params, &d.seq)?),
                (ExactFold::Sparse, _) => {
                    let nt = rna_mfe::encode_seq(d.seq.as_bytes())?;
                    match rna_mfe::dp::fold_sparse(ctx.params, &nt, None) {
                        SparseFold::Below(f, db) => Some((db, f)),
                        SparseFold::AtOrAbove => unreachable!("no cut, so below"),
                    }
                }
                (ExactFold::SparseCut | ExactFold::SparseCutInside, Some(cut)) => {
                    let seq = d.seq.as_bytes();
                    let t = ctx.threshold(e);
                    let folded = match recent {
                        Some(r) => r.fold(cut, ctx.params, seq, t),
                        None => cut.fold(ctx.params, seq, t),
                    };
                    match folded {
                        Some(SparseFold::Below(f, db)) => Some((db, f)),
                        Some(SparseFold::AtOrAbove) => None,
                        None => unreachable!("the walk's design is on its lattice"),
                    }
                }
                _ => Some(sequence_free_fold_zuker(ctx.params, &d.seq)?),
            };
            // `None`: under the cut, nothing lies below the threshold, so the
            // free MFE is between it and `e` (the constrained structure is a
            // free structure).
            debug_assert!(
                sequence_free_mfe(ctx.params, &d.seq).is_ok_and(|f| free
                    .as_ref()
                    .map_or((ctx.threshold(e)..=e).contains(&f.0), |(_, x)| *x == f)),
                "the exact folds disagree on the free MFE"
            );
            // The gap `e − free` is at most the slack, and not negative.
            let modular = free
                .as_ref()
                .is_none_or(|(_, f)| (ctx.threshold(e)..=e).contains(&f.0));
            Ok((
                modular,
                free.filter(|_| !modular).map(|(db, _)| pairs_of(&db)),
            ))
        }
        _ => sequence_gap(ctx.params, &d.seq, ctx.cds, ctx.constraints)
            .map(|g| (g.is_within(f64::from(ctx.slack) / 100.0), None)),
    }
}

/// Run `walk` with the check behind its visitor, and report the first
/// modular design in walk order among those it visits.
pub(super) fn check_while<R>(
    ctx: Context<'_>,
    verify: Verify,
    walk: impl FnOnce(&mut Offer<'_>) -> Result<R, LazyError>,
) -> Result<(R, Outcome), LazyError> {
    run(ctx, verify, None, |offer| {
        walk(&mut |objective, seq, energy, own| offer(objective, seq, energy, own, f64::NAN, 0))
    })
}

/// Run `walk` with the check behind its visitor, keeping every modular
/// design it is offered ([`Outcome::found`]) and raising `floor` to
/// the best adaptation among them; nothing found stops the walk.
pub(super) fn check_collecting<R>(
    ctx: Context<'_>,
    verify: Verify,
    floor: &Floor,
    walk: impl FnOnce(&mut OfferCai<'_>) -> Result<R, LazyError>,
) -> Result<(R, Outcome), LazyError> {
    run(ctx, verify, Some(floor), walk)
}

fn run<R>(
    ctx: Context<'_>,
    verify: Verify,
    collect: Option<&Floor>,
    walk: impl FnOnce(&mut OfferCai<'_>) -> Result<R, LazyError>,
) -> Result<(R, Outcome), LazyError> {
    let threads = verify.threads.max(1);
    let shared = Shared {
        ctx,
        pool: Mutex::new(Arc::new(WitnessPool::new(verify.witnesses))),
        witness_stop: verify.witness_stop,
        witness_promote: verify.witness_promote,
        recent: (verify.fold_reuse > 0)
            .then(|| Recent::new(verify.fold_reuse, verify.fold_reuse_rows)),
        recent_window: ((verify.window_reuse || verify.window_auto) && verify.fold_reuse > 0)
            .then(|| [0, 1].map(|_| Recent::new(verify.fold_reuse, verify.fold_reuse_rows))),
        window_reuse_on: AtomicBool::new(verify.window_reuse && verify.fold_reuse > 0),
        window_reuse_from: AtomicU64::new(u64::MAX),
        window_unconfirmed: AtomicU64::new(0),
        queue: Mutex::new(Queue::default()),
        modular_rank: AtomicU64::new(u64::MAX),
        failed: AtomicBool::new(false),
        batch: verify.handoff_batch.max(1),
        slots: (verify.fold_slots > 0).then(|| {
            Mutex::new(Slots {
                slots: verify.fold_slots,
                folding: 0,
                waiting: BTreeMap::new(),
            })
        }),
        deferred: AtomicU64::new(0),
        deferred_rejected: AtomicU64::new(0),
        ready: Condvar::new(),
        room: Condvar::new(),
        cap: 4 * threads,
        witness_rejects: AtomicU64::new(0),
        folds: AtomicU64::new(0),
        skipped: AtomicU64::new(0),
        witness_ns: AtomicU64::new(0),
        fold_ns: AtomicU64::new(0),
        queue_ns: AtomicU64::new(0),
        starve_ns: AtomicU64::new(0),
        pool_add_ns: AtomicU64::new(0),
        pool_read_ns: AtomicU64::new(0),
        pool_write_ns: AtomicU64::new(0),
        witness_rank: Default::default(),
        witness_spliced: AtomicU64::new(0),
        uncut_folds: AtomicU64::new(0),
        window: verify.window,
        window_auto: verify.window_auto,
        window_utr_on: AtomicBool::new(verify.window_utr),
        window_utr_from: AtomicU64::new(u64::MAX),
        window_calls: AtomicU64::new(0),
        window_rejects: AtomicU64::new(0),
        window_ns: AtomicU64::new(0),
        collect,
        found: Mutex::new(Vec::new()),
    };
    let mut wait_ns = 0u64;
    let result = std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| shared.work());
        }
        // Closes the queue however the walk ends, so that a walk that
        // panics lets the workers drain and the panic reach the caller
        // rather than leaving every worker waiting on an open queue.
        let closing = Closing(&shared);
        let mut rank = 0u64;
        let mut batch = Vec::with_capacity(shared.batch);
        let result = walk(&mut |objective, seq, energy, own, cai, cut_before| {
            rank += 1;
            batch.push(Design {
                objective,
                seq: seq.to_string(),
                energy,
                own: own.to_vec(),
                rank,
                cai,
                cut_before,
            });
            batch.len() >= shared.batch && shared.offer(&mut batch, &mut wait_ns)
        });
        shared.offer(&mut batch, &mut wait_ns);
        drop(closing);
        result
    });
    let q = shared.queue.into_inner().expect("the queue");
    if let Some(e) = q.error {
        return Err(e.into());
    }
    let secs = |a: AtomicU64| a.into_inner() as f64 * 1e-9;
    Ok((
        result?,
        Outcome {
            modular: q.modular,
            witness_rejects: shared.witness_rejects.into_inner(),
            folds: shared.folds.into_inner(),
            skipped: shared.skipped.into_inner(),
            witness_s: secs(shared.witness_ns),
            fold_s: secs(shared.fold_ns),
            wait_s: wait_ns as f64 * 1e-9,
            queue_s: secs(shared.queue_ns),
            starve_s: secs(shared.starve_ns),
            pool_add_s: secs(shared.pool_add_ns),
            pool_read_s: secs(shared.pool_read_ns),
            pool_write_s: secs(shared.pool_write_ns),
            witness_rank: shared.witness_rank.map(AtomicU64::into_inner),
            witness_spliced: shared.witness_spliced.into_inner(),
            reuse_hits: shared.recent.as_ref().map_or(0, |r| r.hits.load(Relaxed)),
            cells_copied: shared.recent.as_ref().map_or(0, |r| r.copied.load(Relaxed)),
            deferred: shared.deferred.into_inner(),
            deferred_rejected: shared.deferred_rejected.into_inner(),
            uncut_folds: shared.uncut_folds.into_inner(),
            window_rejects: shared.window_rejects.into_inner(),
            window_s: secs(shared.window_ns),
            window_cells_copied: shared
                .recent_window
                .as_ref()
                .map_or(0, |r| r.iter().map(|c| c.copied.load(Relaxed)).sum()),
            window_unconfirmed: shared.window_unconfirmed.into_inner(),
            window_utr_from: Some(shared.window_utr_from.into_inner()).filter(|&r| r != u64::MAX),
            window_reuse_from: Some(shared.window_reuse_from.into_inner())
                .filter(|&r| r != u64::MAX),
            found: shared.found.into_inner().expect("the found designs"),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// UTRs `0..10` and `90..100`, CDS `10..90`.
    const CDS: CdsWindow = CdsWindow { start: 10, end: 90 };

    #[test]
    fn a_utr_pair_into_the_cds_middle_is_a_miss_and_the_window_s_own_pairs_are_not() {
        let w = (5, 5);
        // 5'UTR to the CDS middle: outside the window.
        assert!(utr_miss(CDS, 100, w, &[], &[(8, 50)]));
        // The same pair as the design's own: the window admits it already.
        assert!(!utr_miss(CDS, 100, w, &[(8, 50)], &[(8, 50)]));
        // 5'UTR to the CDS start, inside the window.
        assert!(!utr_miss(CDS, 100, w, &[], &[(8, 12)]));
        // 3'UTR to the CDS middle.
        assert!(utr_miss(CDS, 100, w, &[], &[(40, 95)]));
        // The clamp, UTR to UTR: inside every window.
        assert!(!utr_miss(CDS, 100, w, &[], &[(2, 97)]));
        // CDS to CDS outside the window, but no UTR end.
        assert!(!utr_miss(CDS, 100, w, &[], &[(30, 60)]));
    }

    #[test]
    fn reuse_starts_only_past_its_count_and_share() {
        assert!(!reuse_due(REUSE_AFTER - 1, REUSE_AFTER));
        assert!(!reuse_due(REUSE_AFTER, REUSE_AFTER / REUSE_SHARE));
        assert!(reuse_due(REUSE_AFTER, REUSE_AFTER / REUSE_SHARE + 1));
    }
}
