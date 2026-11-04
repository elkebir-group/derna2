//! A single-writer fill of one `P` column: the `Lz` default (`DERNA_PUSH_SCAN`
//! selects the push scan), and opt-in in `Naive` mode (`DERNA_PULL_FILL`).
//!
//! The default scan produces a `P[pos]` cell from three directions: hairpins
//! seeded up front, the stack / bulge / tabulated-small closures **pushed** from
//! every finalized inner pair to later columns, and the generic internal loops
//! **pulled** at `pos`. Pushes and the direct pull family are dealt by inner
//! position, so several workers produce candidates for one target cell and the
//! candidates are buffered and merged serially.
//!
//! Here the unit of work is one **target**: a closing side `(qn, eq)` at
//! `q = pos − 1` and an opening position `p`. The unit produces every
//! single-branch candidate of every cell it owns — the inverted push, the
//! inverted direct family and the existing skewed / one-sided reductions — and
//! keeps one best per cell in a fixed order. Units own disjoint cells, so
//! nothing is merged across workers, and each cell's tie order is the unit's
//! own: the column is the same at every thread count.
//!
//! The inner pairs are read from the one-sided column index, whose entries are
//! every finalized `P` cell sorted by 5′ base `k`; for a fixed 3′ flank the
//! eligible inner pairs are one contiguous slice.
//!
//! In `Naive` mode the unit scores every flank shape within the loop cap
//! directly, as the `Naive` push does, and no reduction runs: the fill there
//! changes who writes a cell, not how a candidate is scored.

use super::*;

/// One target's best candidates, at most one per cell it owns.
pub(super) type CellBest = Vec<(Node, PairKey, State)>;

impl<const S: usize> Scan<'_, S> {
    /// Run `work(i)` for participants `i = 0..n` and return what each that ran
    /// produced, in participant order. On the scan's pool the caller is
    /// participant 0 and a worker not scheduled before the caller's share is
    /// done does not run, so `work` must take its units off a shared counter;
    /// without a pool every participant runs on a fresh scoped thread.
    pub(super) fn fan_collect<R: Send>(
        &self,
        n: usize,
        work: impl Fn(usize) -> R + Sync,
    ) -> Vec<R> {
        let out: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
        let job = |i: usize| {
            let r = work(i);
            *out[i].lock().unwrap() = Some(r);
        };
        match self.fan_pool {
            Some(pool) => pool.run_joined(n, &job),
            None => std::thread::scope(|scope| {
                for i in 0..n {
                    let job = &job;
                    crate::task_clock::spawn(scope, move || job(i));
                }
            }),
        }
        out.into_iter()
            .filter_map(|m| m.into_inner().unwrap())
            .collect()
    }

    /// Fill `P[pos]`'s single-branch closures, one owner per target cell.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pull_fill_column(
        &self,
        t: &mut Tables,
        hx: &HIndex,
        osx: &OneSideIndex,
        okx: &OneSideKIndex,
        pos: u32,
        prof: &mut Profile,
        pool: &mut [Writes],
    ) {
        if pos == 0 {
            return;
        }
        let q = pos - 1;
        let sides: Vec<(Node, Edge)> = self
            .dfa
            .nodes_at(q)
            .iter()
            .flat_map(|&qn| self.dfa.right(qn).iter().map(move |&eq| (qn, eq)))
            .collect();
        if sides.is_empty() || q == 0 {
            return;
        }
        // Targets are `(side, p)` for every opening position before `q`.
        let n_p = q;
        let total = sides.len() as u32 * n_p;
        let m = prof.mark();
        let rights = RightCorners::build(self, q);
        prof.record(Phase::FillRights, m);
        let unit = |w: &mut Writes, out: &mut CellBest, i: u32| {
            let (qn, eq) = sides[(i / n_p) as usize];
            self.fill_target(t, w, hx, osx, okx, &rights, qn, eq, i % n_p);
            reduce_into(w, out, t.lambda());
        };
        if self.threads <= 1 || (!self.fan_all && total < PULL_PARALLEL_MIN) {
            let m = prof.mark();
            let w = &mut pool[0];
            let mut out = CellBest::new();
            for i in 0..total {
                unit(w, &mut out, i);
            }
            prof.record(Phase::FanPullCompute, m);
            let m = prof.mark();
            for (o, key, st) in out {
                t.update_p(o, key, st);
            }
            prof.record(Phase::FanPullApply, m);
            return;
        }
        let counter = AtomicU32::new(0);
        let n_workers = self.fan_workers(total.div_ceil(PULL_CHUNK) as usize);
        let m = prof.mark();
        let timing = prof.is_on();
        let buffers: Vec<Mutex<&mut Writes>> =
            pool[..n_workers].iter_mut().map(Mutex::new).collect();
        // Each participant returns its chunks' best cells tagged by chunk
        // start, so the insert below runs in unit order whatever the dealing
        // was.
        let (clocks, parts): (Vec<WorkerClock>, Vec<_>) = self
            .fan_collect(n_workers, |me| {
                let mut w = buffers[me].lock().unwrap();
                let mut clock = WorkerClock::start(timing);
                let mut mine = Vec::new();
                loop {
                    let start = counter.fetch_add(PULL_CHUNK, Ordering::Relaxed);
                    if start >= total {
                        break;
                    }
                    let began = timing.then(Instant::now);
                    let mut out = CellBest::new();
                    for i in start..(start + PULL_CHUNK).min(total) {
                        unit(&mut w, &mut out, i);
                    }
                    mine.push((start, out));
                    clock.chunk(began, false);
                }
                (clock.stop(), mine)
            })
            .into_iter()
            .unzip();
        prof.record(Phase::FanPullCompute, m);
        record_fan(prof, FanSite::Pull, &[], &clocks);
        let mut parts: Vec<(u32, CellBest)> = parts.into_iter().flatten().collect();
        let m = prof.mark();
        parts.sort_unstable_by_key(|&(start, _)| start);
        for (_, out) in parts {
            for (o, key, st) in out {
                t.update_p(o, key, st);
            }
        }
        prof.record(Phase::FanPullApply, m);
    }

    /// Every single-branch candidate of the cells opening at `p` and closing at
    /// `(qn, eq)`, appended to `w.p` in a fixed order.
    #[allow(clippy::too_many_arguments)]
    fn fill_target(
        &self,
        t: &Tables,
        w: &mut Writes,
        hx: &HIndex,
        osx: &OneSideIndex,
        okx: &OneSideKIndex,
        rights: &RightCorners,
        qn: Node,
        eq: Edge,
        p: u32,
    ) {
        let q = qn.pos;
        let ls = self.single_max_len;
        // The inverted push and direct family: per 3′ flank `n2`, the inner
        // pairs closing at `l = q − 1 − n2` whose 5′ base `k = p + 1 + n1`
        // falls in the 5′ flank window, gathered and then scored grouped by
        // their 5′ boundary node, so each opening edge's flank is weighed once
        // per group as the push weighs it once per inner pair.
        GATHER.with_borrow_mut(|gathered| {
            gathered.clear();
            for n2 in 0..=ls {
                let Some(l) = q.checked_sub(1 + n2 as u32) else {
                    break;
                };
                if l <= p + 1 || !self.must.free_span(l + 1, q) {
                    continue;
                }
                // No shape with a longer 5′ flank is written at this `n2`.
                let Some(n1_hi) = self.fill_n1_hi[n2] else {
                    continue;
                };
                let khi = (p + 1 + n1_hi as u32).min(l);
                for e in osx.window(l, p + 1, khi) {
                    let n1 = (e.k - p - 1) as usize;
                    let shape = self.fill_shapes[n1 * (ls + 1) + n2];
                    if shape == 0 {
                        continue;
                    }
                    let pushed = shape & SHAPE_PUSHED != 0;
                    if !self.must.free_span(p + 1, e.k) {
                        continue;
                    }
                    gathered.push(Gathered {
                        io: e.io,
                        ikey: e.ikey,
                        st: e.st,
                        n1,
                        n2,
                        pushed,
                    });
                }
            }
            let Some(&first) = gathered.first() else {
                return;
            };
            // Group by (n1, lane) with a stable counting sort: both are small,
            // so this is linear where a comparison sort per target was not.
            GROUPED.with_borrow_mut(|(grouped, counts)| {
                let stride = self.dfa.stride();
                let buckets = stride * (ls + 1);
                counts.clear();
                counts.resize(buckets + 1, 0);
                for g in gathered.iter() {
                    counts[bucket(g, stride) + 1] += 1;
                }
                for b in 0..buckets {
                    counts[b + 1] += counts[b];
                }
                grouped.clear();
                grouped.resize(gathered.len(), first);
                for g in gathered.iter() {
                    let slot = &mut counts[bucket(g, stride)];
                    grouped[*slot] = *g;
                    *slot += 1;
                }
                self.close_gathered(t, w, rights, qn, eq, p, grouped);
            });
        });
        if self.pull_generic() && p <= q.saturating_sub(13) {
            self.skewed_pull_at(t, w, hx, qn, eq, p);
            self.oneside_pull_at(t, w, osx, qn, eq, p);
            self.oneside_pull3_at(t, w, okx, qn, eq, p);
        }
    }

    /// Score the gathered inner pairs against every opening edge at `p` that
    /// pairs with `eq`, through the push's and the direct family's own
    /// scorers. A pushed shape is dropped where its 5′ flank has no path, as
    /// the push drops it; the direct family does not ask.
    #[allow(clippy::too_many_arguments)]
    fn close_gathered(
        &self,
        t: &Tables,
        w: &mut Writes,
        rights: &RightCorners,
        qn: Node,
        eq: Edge,
        p: u32,
        gathered: &[Gathered],
    ) {
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                for group in gathered.chunk_by(|a, b| a.ikey.open == b.ikey.open) {
                    let open = group[0].ikey.open;
                    let left_cai = self.dfa.broken_codon_cai_s::<S>(ep.to, open);
                    let mut corners_ready = false;
                    LEFTS.with_borrow_mut(|lefts| {
                        for g in group {
                            if g.pushed && left_cai.is_none() {
                                continue;
                            }
                            let ctx = SingleCtx {
                                inner: PairRef {
                                    outer: g.io,
                                    key: g.ikey,
                                },
                                st: g.st,
                                ep: *ep,
                                pn,
                                qn,
                                eq,
                                outer_pair,
                                inner_pair: pair_from_nucleotides(
                                    g.ikey.nuc_open(),
                                    g.ikey.nuc_close,
                                ),
                            };
                            if g.n1 == 0 || g.n2 == 0 {
                                let left_cai = left_cai.expect("checked for a pushed shape");
                                self.stack_or_bulge(w, &ctx, left_cai, g.n1, g.n2);
                                continue;
                            }
                            if !corners_ready {
                                corner_scores_into(self.dfa, ep.to, open, lefts);
                                corners_ready = true;
                            }
                            if let Some(cand) = self.internal_loop_cand_with(
                                t.lambda(),
                                &ctx,
                                lefts,
                                rights.get(g.io, qn),
                                g.n1,
                                g.n2,
                            ) {
                                w.p.push(cand);
                            }
                        }
                    });
                }
            }
        }
    }
}

/// A flank shape the push writes (see [`pushed_shape`]).
const SHAPE_PUSHED: u8 = 1;
/// A flank shape the direct pull family writes (see [`direct_shape`]).
const SHAPE_DIRECT: u8 = 2;

/// Which of the two inverted families writes each flank shape `(n1, n2)`,
/// `n1, n2 ≤ ls`, at index `n1 · (ls + 1) + n2`: a gathered inner pair is
/// classified by one load rather than by both predicates. In `Naive` mode the
/// push writes every shape within the loop cap, generic ones included.
pub(super) fn fill_shapes(mode: InternalLoopMode, ls: usize, dskew: usize) -> Vec<u8> {
    let mut table = vec![0; (ls + 1) * (ls + 1)];
    for n1 in 0..=ls {
        for n2 in 0..=ls {
            let mut shape = 0;
            let naive = matches!(mode, InternalLoopMode::Naive);
            if (naive && n1 + n2 <= ls) || (!naive && pushed_shape(ls, n1, n2)) {
                shape |= SHAPE_PUSHED;
            }
            if !naive && direct_shape(ls, dskew, n1, n2) {
                shape |= SHAPE_DIRECT;
            }
            table[n1 * (ls + 1) + n2] = shape;
        }
    }
    table
}

/// Per 3′ flank `n2`, the longest 5′ flank any shape in `shapes` (a
/// [`fill_shapes`] table) writes, or `None` when it writes none.
pub(super) fn fill_n1_hi(ls: usize, shapes: &[u8]) -> Vec<Option<usize>> {
    (0..=ls)
        .map(|n2| (0..=ls).rev().find(|&n1| shapes[n1 * (ls + 1) + n2] != 0))
        .collect()
}

/// Whether the push writes shape `(n1, n2)`: a non-generic shape whose
/// unpaired total is within the loop cap `ls` and, for `n1 ≥ 2`, whose 3′
/// flank is at most three — the push's own special-shape window.
fn pushed_shape(ls: usize, n1: usize, n2: usize) -> bool {
    if n1 + n2 > ls || is_generic(n1, n2) {
        return false;
    }
    n1 < 2 || n2 <= 3
}

/// Whether the direct pull family writes shape `(n1, n2)`: a generic shape
/// with `n2 ∈ [2, ls − 2]` and `n1` in the window the direct family scores for
/// that `n2`, given the ninio saturation threshold `dskew`.
fn direct_shape(ls: usize, dskew: usize, n1: usize, n2: usize) -> bool {
    if !is_generic(n1, n2) || !(2..=ls - 2).contains(&n2) {
        return false;
    }
    let (lo, hi) = if n2 <= 4 {
        (2, (n2 + dskew - 1).min(ls - n2))
    } else {
        ((n2.saturating_sub(dskew) + 1).max(2), 4)
    };
    (lo..=hi).contains(&n1)
}

/// The 3′ flank corner scores from every inner node an internal loop closing
/// at column `q` can enclose to each closing node at `q`. They depend on the
/// two nodes only, not on the opening position or edges, so one table serves
/// every target of the column.
pub(super) struct RightCorners {
    /// The lattice's dense node layout, which inner nodes are indexed by.
    layout: crate::dfa::NodeLayout,
    /// Node index of the first inner node held.
    lo: usize,
    /// Per closing-node lane, the corners per inner node index from `lo`.
    by_lane: [Vec<Vec<Corner>>; 3],
}

impl RightCorners {
    fn build<const S: usize>(scan: &Scan<'_, S>, q: u32) -> Self {
        let from = q.saturating_sub(scan.single_max_len as u32);
        let layout = scan.dfa.layout();
        let lo = layout.ix(Node::new(from, 0));
        let span = layout.ix(Node::new(q, 0)) - lo;
        // A third lane only on a stop-labelled lattice: an empty `Vec`
        // allocates nothing.
        let lane = || vec![Vec::new(); span];
        let mut by_lane = [
            lane(),
            lane(),
            if layout.stride() > 2 {
                lane()
            } else {
                Vec::new()
            },
        ];
        for &qn in scan.dfa.nodes_at(q) {
            for pos in from..q {
                for &io in scan.dfa.nodes_at(pos) {
                    let out = &mut by_lane[usize::from(qn.lane)][layout.ix(io) - lo];
                    corner_scores_into(scan.dfa, io, qn, out);
                }
            }
        }
        RightCorners {
            layout,
            lo,
            by_lane,
        }
    }

    /// The corners from inner node `io` to closing node `qn`.
    fn get(&self, io: Node, qn: Node) -> &[Corner] {
        &self.by_lane[usize::from(qn.lane)][self.layout.ix(io) - self.lo]
    }
}

/// One inner pair a target gathered: its cell, and the flank shape it would
/// close with, and whether that shape is one the push writes.
#[derive(Clone, Copy)]
struct Gathered {
    io: Node,
    ikey: PairKey,
    st: Value,
    n1: usize,
    n2: usize,
    pushed: bool,
}

thread_local! {
    /// Reused per-target scratch for the gathered inner pairs.
    static GATHER: RefCell<Vec<Gathered>> = const { RefCell::new(Vec::new()) };
    /// Reused per-target scratch for the grouping: the gathered pairs in
    /// (n1, lane) order, and the bucket offsets.
    static GROUPED: RefCell<(Vec<Gathered>, Vec<usize>)> =
        const { RefCell::new((Vec::new(), Vec::new())) };
}

/// A gathered pair's grouping bucket: its 5′ flank length, then its 5′ lane.
fn bucket(g: &Gathered, stride: usize) -> usize {
    g.n1 * stride + usize::from(g.ikey.open.lane)
}

/// Fold `w.p` down to one best candidate per cell, in production order — the
/// min-BRD merge with the first arrival kept on a tie — append them to `out`,
/// and empty `w`.
fn reduce_into(w: &mut Writes, out: &mut CellBest, lambda: f64) {
    let from = out.len();
    for &(o, key, st) in &w.p {
        match out[from..]
            .iter_mut()
            .find(|(oo, kk, _)| *oo == o && *kk == key)
        {
            Some((_, _, best)) if st.brd(lambda) < best.brd(lambda) => *best = st,
            Some(_) => {}
            None => out.push((o, key, st)),
        }
    }
    w.clear();
}

/// One opening position's results from the `M2` column: the best shared
/// opening per `(open node, opening nucleotide)` and the best short-flank
/// `Multi` cell per target.
#[derive(Default)]
struct OpeningBest {
    opens: Vec<(MultiOpenKey, State)>,
    multis: Vec<(Node, PairKey, State)>,
}

/// The column-wide facts every `M2` core at one position shares: the codon
/// boundary its 3′ stretch runs to, whether the shared opening is admissible,
/// and how far a short flank may reach.
#[derive(Clone, Copy)]
struct M2Column {
    pos: u32,
    boundary: u32,
    open_ok: bool,
    short_hi: u32,
}

impl<const S: usize> Scan<'_, S> {
    /// Process `M2[pos]` with one owner per opening position: `M = M2` for
    /// every cell, then every multi-loop opening the column's cores seed —
    /// the shared pool entries and the short-flank `Multi` cells — computed by
    /// the unit that owns its opening position and inserted once.
    pub(super) fn m2_fill_column(
        &self,
        t: &mut Tables,
        pos: u32,
        prof: &mut Profile,
        lead: Option<&Lead>,
    ) {
        let m = prof.mark();
        let cols: Vec<(Node, Vec<(Node, State)>)> = self
            .dfa
            .nodes_at(pos)
            .iter()
            .map(|&o| (o, t.m2_states(o)))
            .collect();
        for (o, cells) in &cols {
            for &(inner, st) in cells {
                let back = Back::MFromM2 {
                    m2: SpanRef { outer: *o, inner },
                };
                t.update_m(*o, inner, State::new(st.mfe, st.cai, back));
            }
        }
        prof.record(Phase::M2Copy, m);
        let m = prof.mark();
        let ls = self.single_max_len as u32;
        let inners = cols.iter().flat_map(|(_, c)| c.iter().map(|(i, _)| i.pos));
        let (Some(lo), Some(hi)) = (inners.clone().min(), inners.max()) else {
            prof.record(Phase::M2Units, m);
            return;
        };
        let lambda = t.lambda();
        let boundary = self.dfa.next_codon_boundary(pos);
        let col = M2Column {
            pos,
            boundary,
            open_ok: self.must.free_span(pos, boundary),
            short_hi: boundary.min(self.must.max_right(pos).saturating_add(1)),
        };
        let p_lo = lo.saturating_sub(ls);
        let units: Vec<u32> = (p_lo..hi).collect();
        let parts: Vec<OpeningBest> = if self.threads <= 1 || (!self.fan_all && units.len() < 64) {
            units
                .iter()
                .map(|&p| self.m2_opening(&cols, col, p, lambda))
                .collect()
        } else {
            let counter = AtomicUsize::new(0);
            let n_workers = self.fan_workers(units.len().div_ceil(8));
            let timing = prof.is_on();
            let (clocks, tagged): (Vec<WorkerClock>, Vec<_>) = self
                .fan_collect(n_workers, |_| {
                    lead.map(run_lead);
                    let mut clock = WorkerClock::start(timing);
                    let mut mine = Vec::new();
                    loop {
                        let i = counter.fetch_add(1, Ordering::Relaxed);
                        if i >= units.len() {
                            break;
                        }
                        let began = timing.then(Instant::now);
                        mine.push((i, self.m2_opening(&cols, col, units[i], lambda)));
                        clock.chunk(began, false);
                    }
                    (clock.stop(), mine)
                })
                .into_iter()
                .unzip();
            record_fan(prof, FanSite::M2, &[], &clocks);
            let mut tagged: Vec<(usize, OpeningBest)> = tagged.into_iter().flatten().collect();
            tagged.sort_unstable_by_key(|&(i, _)| i);
            tagged.into_iter().map(|(_, b)| b).collect()
        };
        prof.record(Phase::M2Units, m);
        let m = prof.mark();
        for part in parts {
            for (key, st) in part.opens {
                if col.boundary == col.pos {
                    t.multi_open_insert(key, st);
                } else {
                    t.multi_open_defer(col.boundary, key, st);
                }
            }
            for (outer, key, st) in part.multis {
                t.update_multi(outer, key, st);
            }
        }
        // Left flanks of `SINGLE_MAX_LEN` or longer: the units stop short of
        // them, and each core end extends them leftward in one sweep, as the
        // push scan does (see `far_multi_open`).
        for (o, cells) in &cols {
            if cells.is_empty() {
                continue;
            }
            let end = self.core_end(*o);
            let mut far = FxHashMap::default();
            for &(inner, st) in cells {
                self.offer_far(&end, inner, st, lambda, &mut far);
            }
            self.far_multi_open(t, &end, far);
        }
        prof.record(Phase::M2Apply, m);
    }

    /// Every opening at position `p` over the column's `M2` cores, reduced to
    /// one best per pool key and per short-flank cell, in the cores' column
    /// order — the order the per-core scan offered them.
    fn m2_opening(
        &self,
        cols: &[(Node, Vec<(Node, State)>)],
        col: M2Column,
        p: u32,
        lambda: f64,
    ) -> OpeningBest {
        let mut best = OpeningBest::default();
        let ls = self.single_max_len as u32;
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                for (o, cells) in cols {
                    // Cores whose inner node lies in `(p, p + L]`, still in
                    // ascending inner order.
                    let from = cells.partition_point(|(i, _)| i.pos <= p);
                    let to = cells.partition_point(|(i, _)| i.pos <= p + ls);
                    for &(inner, st) in &cells[from..to] {
                        if p < self.must.min_left(inner.pos) {
                            continue;
                        }
                        let core = SpanRef { outer: *o, inner };
                        self.m2_offer(&mut best, col, core, st, pn, *ep, lambda);
                    }
                }
            }
        }
        best
    }

    /// One core's opening at `(pn, ep)`: its short-flank `Multi` cells and its
    /// shared pool entry, each merged into `best` by min-BRD, first kept on a
    /// tie.
    #[allow(clippy::too_many_arguments)]
    fn m2_offer(
        &self,
        best: &mut OpeningBest,
        col: M2Column,
        core: SpanRef,
        st: State,
        pn: Node,
        ep: Edge,
        lambda: f64,
    ) {
        let Some(flank_cai) = self.dfa.broken_codon_cai_s::<S>(ep.to, core.inner) else {
            return;
        };
        let left_cai = ep.cai + flank_cai;
        for q in col.pos..col.short_hi {
            for &qn in self.dfa.nodes_at(q) {
                let Some(right_cai) = self.dfa.broken_codon_cai_s::<S>(core.outer, qn) else {
                    continue;
                };
                for eq in self.dfa.right(qn) {
                    if canonical(ep.nuc, eq.nuc).is_none() {
                        continue;
                    }
                    let key = PairKey {
                        open: pn,
                        close: qn,
                        opening: ep.opening,
                        nuc_close: eq.nuc,
                    };
                    let cand = State::new(
                        st.mfe,
                        st.cai + left_cai + right_cai + eq.cai,
                        Back::MultiWrap { core },
                    );
                    keep_best(&mut best.multis, (eq.to, key), cand, lambda);
                }
            }
        }
        if !col.open_ok {
            return;
        }
        let Some(c_outer) = self.dfa.span_head_cai(core.outer) else {
            return;
        };
        let key = MultiOpenKey {
            open: pn,
            nuc_open: ep.nuc,
            lane_open: ep.to.lane,
            reach: self.reach_of(core.outer),
        };
        let cand = State::new(
            st.mfe,
            st.cai + left_cai + c_outer,
            Back::MultiWrap { core },
        );
        keep_best(&mut best.opens, key, cand, lambda);
    }
}

/// Merge `cand` into a short list keyed by `key`: min-BRD, the first arrival
/// kept on a tie — the table merge's rule.
fn keep_best<K: PartialEq + Copy, E: KeyedBest<K>>(
    list: &mut Vec<E>,
    key: K,
    cand: State,
    lambda: f64,
) {
    match list.iter_mut().find(|e| e.key() == key) {
        Some(e) if cand.brd(lambda) < e.state().brd(lambda) => e.set(cand),
        Some(_) => {}
        None => list.push(E::new(key, cand)),
    }
}

/// An entry of a [`keep_best`] list.
trait KeyedBest<K> {
    fn key(&self) -> K;
    fn state(&self) -> State;
    fn set(&mut self, st: State);
    fn new(key: K, st: State) -> Self;
}

impl KeyedBest<MultiOpenKey> for (MultiOpenKey, State) {
    fn key(&self) -> MultiOpenKey {
        self.0
    }
    fn state(&self) -> State {
        self.1
    }
    fn set(&mut self, st: State) {
        self.1 = st;
    }
    fn new(key: MultiOpenKey, st: State) -> Self {
        (key, st)
    }
}

impl KeyedBest<(Node, PairKey)> for (Node, PairKey, State) {
    fn key(&self) -> (Node, PairKey) {
        (self.0, self.1)
    }
    fn state(&self) -> State {
        self.2
    }
    fn set(&mut self, st: State) {
        self.2 = st;
    }
    fn new(key: (Node, PairKey), st: State) -> Self {
        (key.0, key.1, st)
    }
}

/// One `M2` slot's best bifurcation: the slot index, the merged `(mfe, cai)`
/// and the closed pair that gave it.
type SlotBest = (usize, (Energy, f64, PairKey));

/// One closed pair as the bifurcation reads it: the `M` column it extends and
/// what it adds.
#[derive(Clone, Copy)]
struct BranchCell {
    key: PairKey,
    add_mfe: Energy,
    add_cai: f64,
}

impl<const S: usize> Scan<'_, S> {
    /// `M2 = M + P` for every closed pair at `pos`, one owner per `M2` slot:
    /// each slot `inner` takes the least of `M[open][inner] + P` over the
    /// column's closed pairs, read in the order the push offered them, so a
    /// slot's tie keeps the same first arrival. The `M` columns read are all
    /// finalized (`open < pos`), and `M2[pos]` has no other writer.
    pub(super) fn bifurcate_fill_column(
        &self,
        t: &mut Tables,
        pos: u32,
        prof: &mut Profile,
        lead: Option<&Lead>,
    ) {
        let lambda = t.lambda();
        for &o in self.dfa.nodes_at(pos) {
            let m = prof.mark();
            let cells: Vec<BranchCell> = t
                .p_states(o)
                .into_iter()
                .map(|(key, st)| BranchCell {
                    key,
                    add_mfe: st.mfe
                        + self.m_stem_cost(pair_from_nucleotides(key.nuc_open(), key.nuc_close)),
                    add_cai: st.cai,
                })
                .collect();
            if cells.is_empty() {
                prof.record(Phase::BifPrep, m);
                continue;
            }
            let slots = cells
                .iter()
                .map(|c| t.m[self.dfa.ix(c.key.open)].slots())
                .max()
                .unwrap_or(0);
            prof.record(Phase::BifPrep, m);
            let m = prof.mark();
            let shared: &Tables = t;
            // One block of `M2` slots: each cell's `M` column is read as one
            // contiguous slice, and every slot still meets the cells in the
            // push's order, so its tie keeps the push's first arrival.
            let best_block = |from: usize, to: usize| -> Vec<SlotBest> {
                let len = to - from;
                // Per slot: the best `(mfe, cai)`, its cell, and its BRD, kept
                // so each compare evaluates only the candidate's.
                let mut bm = [i32::MAX; BIF_BLOCK];
                let mut bc = [0.0; BIF_BLOCK];
                let mut bk = [0; BIF_BLOCK];
                let mut bb = [f64::INFINITY; BIF_BLOCK];
                for (ci, c) in cells.iter().enumerate() {
                    let (ms, cs) = shared.m[self.dfa.ix(c.key.open)].hot_slices(from, to);
                    for k in 0..ms.len() {
                        if ms[k] == i32::MAX {
                            continue;
                        }
                        let (mfe, cai) = (ms[k] + c.add_mfe.0, cs[k] + c.add_cai);
                        let brd = hot_brd(Energy(mfe), cai, lambda);
                        if brd < bb[k] {
                            (bm[k], bc[k], bk[k], bb[k]) = (mfe, cai, ci, brd);
                        }
                    }
                }
                (0..len)
                    .filter(|&k| bm[k] != i32::MAX)
                    .map(|k| (from + k, (Energy(bm[k]), bc[k], cells[bk[k]].key)))
                    .collect()
            };
            let chunk = BIF_BLOCK;
            let found: Vec<SlotBest> = if self.threads <= 1 || (!self.fan_all && slots < 512) {
                (0..slots)
                    .step_by(chunk)
                    .flat_map(|from| best_block(from, (from + chunk).min(slots)))
                    .collect()
            } else {
                let counter = AtomicUsize::new(0);
                let n_workers = self.fan_workers(slots.div_ceil(chunk));
                let timing = prof.is_on();
                let (clocks, parts): (Vec<WorkerClock>, Vec<_>) = self
                    .fan_collect(n_workers, |_| {
                        lead.map(run_lead);
                        let mut clock = WorkerClock::start(timing);
                        let mut mine = Vec::new();
                        loop {
                            let from = counter.fetch_add(chunk, Ordering::Relaxed);
                            if from >= slots {
                                break;
                            }
                            let began = timing.then(Instant::now);
                            let to = (from + chunk).min(slots);
                            mine.push((from, best_block(from, to)));
                            clock.chunk(began, false);
                        }
                        (clock.stop(), mine)
                    })
                    .into_iter()
                    .unzip();
                record_fan(prof, FanSite::Bif, &[], &clocks);
                let mut parts: Vec<(usize, Vec<SlotBest>)> = parts.into_iter().flatten().collect();
                parts.sort_unstable_by_key(|&(from, _)| from);
                parts.into_iter().flat_map(|(_, v)| v).collect()
            };
            prof.record(Phase::BifUnits, m);
            let m = prof.mark();
            for (i, (mfe, cai, key)) in found {
                let inner = self.dfa.node_of(i);
                let back = Back::M2Bifurcation {
                    left: SpanRef {
                        outer: key.open,
                        inner,
                    },
                    right: PairRef { outer: o, key },
                };
                t.update_m2(o, inner, State::new(mfe, cai, back));
            }
            prof.record(Phase::BifApply, m);
        }
    }
}

/// `M2` slots per bifurcation block.
const BIF_BLOCK: usize = 64;

/// [`State::brd`] on a bare `(mfe, cai)`, as the dense columns rank them.
fn hot_brd(mfe: Energy, cai: f64, lambda: f64) -> f64 {
    lambda * f64::from(mfe.0) - (1.0 - lambda) * (cai * 100.0)
}
