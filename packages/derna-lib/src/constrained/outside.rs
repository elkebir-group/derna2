//! The motif-free outside of the constrained fold's closed pairs.
//!
//! The fold's rules turn tail cells into a head candidate. Through one rule
//! instance a tail's best completion is the head's plus what the instance adds,
//! `outside(head) + brd(candidate) − brd(tail)`. So the outside pass re-runs
//! the forward enumeration over the finished tables, from each tail, and keeps
//! the least such value per tail instead of merging a candidate into the head.
//! Columns are visited from the 3′ end, and within a column in the reverse of
//! the forward order (the unpaired extensions into the next column, the
//! multi-loop wraps and `M2`, the closed pairs, the multi-loop closings): every
//! head of a cell lies in a later column, or later in the same column, so its
//! outside is final when the cell's is taken. A multi-loop wrap is taken in one
//! sweep per frontier state, mirroring the forward fold's long-flank sweep, and
//! each column's closed pairs are spread over a pool.
//!
//! The pass runs over the constrained lattice's own tables, keyed by the
//! automaton state on either side of each pair, so every completion it scores
//! is motif-free: unlike an outside pass over the unconstrained lattice it
//! never credits a completion the motif set forbids. A walk takes it at a few
//! weights and prunes the folds between them on the chord: the hull walk
//! under an incumbent's ceiling ([`motif_free_outside_pruned`]), the front
//! walk under a ceiling it certifies itself ([`aware_pass`]).

use super::*;

mod aware_file;

/// A closed pair by position: `(open, close, nuc_open, nuc_close, q_open,
/// q_outer)`, the automaton states before the opening base and after the
/// closing one.
#[cfg(test)]
pub(crate) type PairAt = (u32, u32, u8, u8, u32, u32);

/// What [`through_by_position`] found: each closed pair's through value, the
/// optimum, and how many pairs take their outside from a multi-loop rule.
#[cfg(test)]
pub(crate) struct Through {
    pub(crate) by_pair: FxHashMap<PairAt, f64>,
    pub(crate) optimum: f64,
    pub(crate) through_multiloops: usize,
}

/// The exact constrained fold at `lambda` and the motif-free through value
/// (inside plus outside, BRD units) of each of its closed pairs that some
/// complete design passes through, keyed by position and taking the least over
/// the lattice's lanes there. `None` where no motif-free design exists.
#[cfg(test)]
pub(crate) fn through_by_position(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
) -> Option<Through> {
    let o = outside_of(product, params, lambda, None, Wraps::Sweep, 1)?;
    let score = |c: &Cell| brd(c.mfe, c.cai, lambda);
    let mut by_pair: FxHashMap<PairAt, f64> = FxHashMap::default();
    for (column, outs) in o.fold.p.iter().zip(&o.p) {
        for (key, outside) in outs {
            let at = (
                key.open.node().pos,
                key.close.node().pos,
                key.nuc_open,
                key.nuc_close,
                key.q_open,
                key.q_outer,
            );
            let slot = by_pair.entry(at).or_insert(f64::INFINITY);
            *slot = slot.min(score(&column[key]) + outside);
        }
    }
    Some(Through {
        by_pair,
        optimum: o.optimum,
        through_multiloops: o.through_multiloops,
    })
}

/// The motif-free outside of every closed pair at `lambda`, indexed by outer
/// node and keyed as the fold's `P`, and the optimum. A pair on no motif-free
/// design has no entry.
#[cfg(test)]
pub(crate) fn motif_free_outside(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    threads: usize,
) -> Option<(Vec<FxHashMap<PKey, f64>>, f64)> {
    let o = outside_of(product, params, lambda, None, Wraps::Sweep, threads)?;
    Some((o.p, o.optimum))
}

/// [`motif_free_outside`] from a fold pruned by `bound` (an unconstrained
/// anchor at `lambda` and a feasible ceiling): exact for every pair with a
/// design within the ceiling, and a floor for the rest. A pair the prune never
/// built has no entry, which says nothing about it.
pub(crate) fn motif_free_outside_pruned<'a>(
    product: &'a ProductDfa<'a>,
    params: &'a Params,
    lambda: f64,
    bound: OutsideBound<'a>,
    threads: usize,
) -> Option<Vec<FxHashMap<PKey, f64>>> {
    Some(outside_of(product, params, lambda, Some(bound), Wraps::Sweep, threads)?.p)
}

/// The motif-free floors at one weight, from the outside pass over the
/// constrained lattice: what a fold between two such weights prunes on
/// ([`AwareSide::Pass`]).
pub(crate) struct AwarePass {
    /// The lattice's dense node layout, which the tables are indexed by.
    layout: NodeLayout,
    /// The weight.
    lambda: f64,
    /// The constrained optimum at `lambda`, on the fold's BRD scale.
    pub optimum: f64,
    /// The ceiling the tables were taken under. A closed pair with a
    /// motif-free design at or below it holds its exact value; any other holds
    /// a floor on it, or no entry. A cell or pair with no entry was never
    /// built, so every motif-free design through it scores above the ceiling.
    pub ceiling: f64,
    /// Bounded folds the search for the ceiling took; the last one's tables
    /// are kept.
    pub folds: usize,
    /// Seconds for the folds and the outside pass.
    pub secs: f64,
    /// A motif-free design at the optimum, when the pass was asked to keep
    /// one: the hull vertex at `lambda`, which seeds a walk.
    pub design: Option<super::ConstrainedFold>,
    /// The motif-free outside of each closed pair, by outer node, keyed as
    /// the fold's `P`.
    outside: Vec<FxHashMap<PKey, f64>>,
    /// The best motif-free design through each pair of nodes and
    /// nucleotides, over the automaton states on either side of it, by outer
    /// node.
    through: Vec<FxHashMap<crate::state::PairKey, f64>>,
}

impl AwarePass {
    /// A floor on every motif-free completion of `P[on][key]`, whose inside
    /// `cell` holds at some other weight.
    ///
    /// A cell the pass's fold never built is on no design within its
    /// ceiling. `cell`'s interior completed by the best completion at this
    /// weight is such a design, so that completion scores above the ceiling
    /// less the interior's score here.
    pub(super) fn cell_floor(&self, on: Node, key: &PKey, cell: Cell) -> f64 {
        match self
            .outside
            .get(self.layout.ix(on))
            .and_then(|col| col.get(key))
        {
            Some(&v) => v,
            None => self.ceiling - brd(cell.mfe, cell.cai, self.lambda),
        }
    }

    /// A floor on every motif-free design through `pair`, closing into outer
    /// node `o`: its least through value over its cells, or the ceiling
    /// where the pass's fold built none of them.
    pub(super) fn pair_floor(&self, o: usize, pair: &crate::state::PairKey) -> f64 {
        self.through
            .get(o)
            .and_then(|col| col.get(pair))
            .copied()
            .unwrap_or(self.ceiling)
    }

    /// The pass's weight.
    pub(super) fn lambda(&self) -> f64 {
        self.lambda
    }

    /// [`Self::cell_floor`] before an interior is known: the cell's entry, or
    /// the ceiling the interior's score is taken from.
    pub(super) fn cell_term(&self, on: Node, key: &PKey) -> SideTerm {
        match self
            .outside
            .get(self.layout.ix(on))
            .and_then(|col| col.get(key))
        {
            Some(&v) => SideTerm::Fixed(v),
            None => SideTerm::Above(self.ceiling),
        }
    }

    /// Entries held: cells and pairs, for sizing the pass.
    pub(crate) fn entries(&self) -> (usize, usize) {
        (
            self.outside.iter().map(FxHashMap::len).sum(),
            self.through.iter().map(FxHashMap::len).sum(),
        )
    }
}

/// Where an [`AwarePass`] runs: the constrained lattice, the energy
/// parameters and the region constraint.
pub(crate) struct AwareLattice<'a> {
    pub product: &'a ProductDfa<'a>,
    pub params: &'a Params,
    pub region: PairConstraint,
}

/// The [`AwarePass`] at `lambda` over `on`.
///
/// Its ceiling comes from the bounded search an end of the walk runs: the
/// first ceiling sits `margin` above `floor` (a floor on the constrained
/// optimum), each fold prunes on `bound` at its ceiling, and the margin
/// doubles until a fold certifies, a design found above its ceiling capping
/// the next. The outside pass then runs over the certifying fold's tables, so
/// the ceiling is never below the optimum: every pair on an optimal design
/// holds its exact value. A pair whose every design scores above the ceiling
/// may have lost the cells that carry its outside; it holds the ceiling less
/// its inside, still a floor. `None` where the lattice has no motif-free
/// design.
///
/// A search that has seen a fold fail lifts each later ceiling by
/// `headroom`, so the fold that certifies usually sits that far above the
/// optimum already. A certifying fold whose ceiling sits less than half the
/// `headroom` above the optimum is folded once more at the optimum plus
/// `headroom`, and the outside pass runs over that fold instead. A fold that certifies close above
/// the optimum floors every pair off the optimal designs at the ceiling, which
/// prunes nothing for a fold whose own ceiling lies further above; the
/// headroom gives those pairs their exact values. The kept design is the
/// first certifying fold's, whatever the headroom.
///
/// The pass's floors read a cell or pair its fold never built as above the
/// ceiling ([`AwarePass::cell_floor`], [`AwarePass::pair_floor`]). That
/// holds because the fold is exact and runs to its end: it has no beam, no
/// deadline and no memory stop, and prunes only on `bound`, whose floors are
/// admissible, so every cell of every design within the ceiling is built.
pub(crate) fn aware_pass<'a>(
    on: &AwareLattice<'a>,
    lambda: f64,
    bound: impl Fn(f64) -> OutsideBound<'a>,
    (floor, margin, headroom): (f64, f64, f64),
    threads: usize,
    keep_design: bool,
) -> Option<AwarePass> {
    let started = std::time::Instant::now();
    let (mut cap, mut margin, mut folds) = (f64::INFINITY, margin, 0);
    let mut spans = None;
    let mut lift = 0.0;
    loop {
        let ceiling = (floor + margin).min(cap) + lift;
        folds += 1;
        let (mut f, end_q) = inside(on, lambda, Some(bound(ceiling)), spans.take());
        let found = end_q.map(|q| {
            let c = f.c[f.dfa.ix(f.dfa.end())][&q];
            brd(c.mfe, c.cai, lambda)
        });
        if let Some(end_q) =
            end_q.filter(|_| found.is_some_and(|v| v <= ceiling + BOUND_SLACK) || ceiling >= cap)
        {
            // The fold is exact under an admissible bound and its optimum is
            // under the ceiling, so its traceback is an optimal design.
            let design = keep_design.then(|| {
                let cell = f.c[f.dfa.ix(f.dfa.end())][&end_q];
                let (sequence, structure) = f.traceback(end_q);
                super::ConstrainedFold {
                    mfe: cell.mfe,
                    cai: cell.cai,
                    sequence,
                    structure,
                }
            });
            let optimum = found.expect("a certifying fold has an optimum");
            let (f, end_q, ceiling) = if ceiling < optimum + headroom / 2.0 {
                let wider = optimum + headroom;
                folds += 1;
                let spans = SpanMemo {
                    fills: f.fills,
                    corners: f.corners,
                };
                let (f, end_q) = inside(on, lambda, Some(bound(wider)), Some(spans));
                let end_q = end_q.expect("a wider ceiling keeps the optimum");
                (f, end_q, wider)
            } else {
                (f, end_q, ceiling)
            };
            let how = Reverse {
                ceiling: Some(ceiling),
                wraps: Wraps::Sweep,
                threads,
                inside_s: 0.0,
                skip: sweep_skip(),
            };
            let o = reverse(f, end_q, lambda, how);
            let through = through_of(&o, lambda);
            return Some(AwarePass {
                layout: o.fold.dfa.layout(),
                lambda,
                optimum: o.optimum,
                ceiling,
                folds,
                secs: started.elapsed().as_secs_f64(),
                design,
                outside: o.p,
                through,
            });
        }
        if ceiling >= cap || !ceiling.is_finite() {
            return None;
        }
        if let Some(v) = found {
            cap = cap.min(v);
        }
        margin *= 2.0;
        lift = headroom;
        spans = Some(SpanMemo {
            fills: f.fills,
            corners: f.corners,
        });
    }
}

/// Each pair's least through value over its cells: inside plus outside,
/// keyed without the automaton states. A cell's outside is a floor on its
/// every completion, so the least over the states on either side of a pair
/// is a floor on every motif-free design through the pair.
fn through_of(o: &Outside<'_>, lambda: f64) -> Vec<FxHashMap<crate::state::PairKey, f64>> {
    o.fold
        .p
        .iter()
        .zip(&o.p)
        .map(|(cells, outs)| {
            let mut column: FxHashMap<crate::state::PairKey, f64> = FxHashMap::default();
            for (key, &out) in outs {
                let Some(cell) = cells.get(key) else { continue };
                let v = brd(cell.mfe, cell.cai, lambda) + out;
                let slot = column.entry(key.pair_key()).or_insert(f64::INFINITY);
                *slot = slot.min(v);
            }
            column
        })
        .collect()
}

/// How the outside pass takes the multi-loop wraps: one sweep per frontier
/// state ([`wrap_outsides`]), or every opening of every core, the reference
/// the sweep is tested against.
#[derive(Clone, Copy)]
enum Wraps {
    Sweep,
    /// Every opening whose 5′ flank is at most `max_flank` long.
    #[cfg(test)]
    Enumerate {
        max_flank: u32,
    },
}

/// What [`outside_of`] leaves: the finished fold, the outside of every closed
/// pair and of every multi-loop core (`M2R`), each indexed by its node as the
/// fold's tables are, the optimum, and how many pairs take their outside from
/// a multi-loop rule.
struct Outside<'a> {
    fold: Folder<'a>,
    p: Vec<FxHashMap<PKey, f64>>,
    #[cfg_attr(not(test), allow(dead_code))]
    m2r: Vec<FxHashMap<MKey, f64>>,
    optimum: f64,
    #[cfg_attr(not(test), allow(dead_code))]
    through_multiloops: usize,
    /// Left positions the sweep passed over.
    #[cfg_attr(not(test), allow(dead_code))]
    skipped: usize,
}

/// The pass itself over the lattice without a region constraint: the inside
/// fold, serial, then the reverse pass with its closed pairs spread over
/// `threads`.
fn outside_of<'a>(
    product: &'a ProductDfa<'a>,
    params: &'a Params,
    lambda: f64,
    prune: Option<OutsideBound<'a>>,
    wraps: Wraps,
    threads: usize,
) -> Option<Outside<'a>> {
    let ceiling = prune.as_ref().map(|b| b.ceiling);
    let clock = std::time::Instant::now();
    let on = AwareLattice {
        product,
        params,
        region: PairConstraint::inert(),
    };
    let (f, end_q) = inside(&on, lambda, prune, None);
    let how = Reverse {
        ceiling,
        wraps,
        threads,
        inside_s: clock.elapsed().as_secs_f64(),
        skip: sweep_skip(),
    };
    Some(reverse(f, end_q?, lambda, how))
}

/// The inside fold at `lambda` over `on`, pruned by `prune` when given,
/// serial: its fan-outs cost more than they save at a pass's size. Returns
/// the fold and the automaton state its best design ends in, `None` where
/// no design survives.
fn inside<'a>(
    on: &AwareLattice<'a>,
    lambda: f64,
    prune: Option<OutsideBound<'a>>,
    spans: Option<SpanMemo>,
) -> (Folder<'a>, Option<u32>) {
    let mut f = Folder::with_spans(
        on.product,
        on.params,
        lambda,
        Beam::exact(),
        on.region.clone(),
        MustPairConstraint::inert(),
        1,
        spans,
    );
    f.bound = prune;
    // An unbuilt cell reads as above the ceiling only from a fold that drops
    // nothing but what an admissible floor rules out.
    assert!(f.beam.is_exact(), "an aware pass needs an exact fold");
    let end_q = f.run();
    (f, end_q)
}

/// How [`reverse`] runs: the ceiling the fold was pruned under, if any; how
/// it takes the multi-loop wraps; the threads its closed pairs are spread
/// over; and the fold's seconds, for the timing line a development build can
/// ask for.
struct Reverse {
    ceiling: Option<f64>,
    wraps: Wraps,
    threads: usize,
    inside_s: f64,
    /// Whether the sweep passes over dead left positions under the ceiling
    /// ([`LiveOpens`]).
    skip: bool,
}

/// The reverse pass over the finished fold `f`, whose best design ends in
/// automaton state `end_q`.
fn reverse<'a>(mut f: Folder<'a>, end_q: u32, lambda: f64, how: Reverse) -> Outside<'a> {
    let Reverse {
        ceiling,
        wraps,
        threads,
        inside_s,
        skip,
    } = how;
    // The single-loop enumeration below reads the span tables without
    // filling them, as a threaded fold's workers do, so the spans a serial
    // fold fills lazily are filled here first.
    let (dfa, reachable) = (f.dfa, f.product);
    for pos in 0..=dfa.seq_len {
        for &from in dfa.nodes_at(pos) {
            for &q in reachable.reachable_states(from) {
                f.prefill_from(from, q);
            }
        }
    }
    let (mut wraps_s, mut pairs_s) = (0.0, 0.0);
    let end = f.dfa.end();
    let best = f.c[f.dfa.ix(end)][&end_q];
    let optimum = brd(best.mfe, best.cai, lambda);
    let score = |c: &Cell| brd(c.mfe, c.cai, lambda);
    let edge = |cai: f64| -(1.0 - lambda) * cai * 100.0;
    let size = f.c.len();
    let mut out_c: Vec<FxHashMap<u32, f64>> = vec![FxHashMap::default(); size];
    let mut out_p: Vec<FxHashMap<PKey, f64>> = vec![FxHashMap::default(); size];
    let mut out_multi: Vec<FxHashMap<PKey, f64>> = vec![FxHashMap::default(); size];
    let mut out_m: Vec<FxHashMap<MKey, f64>> = vec![FxHashMap::default(); size];
    let mut out_m2: Vec<FxHashMap<MKey, f64>> = vec![FxHashMap::default(); size];
    let mut out_m2r: Vec<FxHashMap<MKey, f64>> = vec![FxHashMap::default(); size];
    for &q in f.c[f.dfa.ix(end)].keys() {
        out_c[f.dfa.ix(end)].insert(q, 0.0);
    }
    fn lower<K: std::hash::Hash + Eq>(map: &mut FxHashMap<K, f64>, key: K, v: f64) {
        let slot = map.entry(key).or_insert(f64::INFINITY);
        *slot = slot.min(v);
    }
    let mut through_multiloops = 0;
    let skip = ceiling.filter(|_| skip);
    let mut live = LiveOpens(vec![Vec::new(); f.dfa.seq_len as usize + 1]);
    let mut skipped = 0;
    // Each node's `M` cells by their outer automaton state: a closed pair
    // opening at the node reads as bifurcation partners only those whose
    // outer state is its own opening state.
    let m_by_outer: Vec<FxHashMap<u32, Vec<(MKey, Cell)>>> =
        f.m.iter()
            .map(|column| {
                let mut by: FxHashMap<u32, Vec<(MKey, Cell)>> = FxHashMap::default();
                for (&mk, &cell) in column {
                    by.entry(mk.q_outer).or_default().push((mk, cell));
                }
                by
            })
            .collect();
    let mut pass = |pool: Option<&Pool>| {
        for pos in (0..=f.dfa.seq_len).rev() {
            // One unpaired base into the next column: `C`, `M` and `M2R` alike add
            // the edge's adaptation alone.
            if pos < f.dfa.seq_len {
                for &node in f.dfa.nodes_at(pos) {
                    for &e in f.dfa.right(node) {
                        for &q in f.c[f.dfa.ix(node)].keys() {
                            let Some(q2) = f.ac.step_edge(node, q, e) else {
                                continue;
                            };
                            if let Some(&head) = out_c[f.dfa.ix(e.to)].get(&q2) {
                                lower(&mut out_c[f.dfa.ix(node)], q, head + edge(e.cai));
                            }
                        }
                        for (table, out) in [(&f.m, &mut out_m), (&f.m2r, &mut out_m2r)] {
                            for &mk in table[f.dfa.ix(node)].keys() {
                                let Some(q2) = f.ac.step_edge(node, mk.q_outer, e) else {
                                    continue;
                                };
                                let next = MKey { q_outer: q2, ..mk };
                                if let Some(&head) = out[f.dfa.ix(e.to)].get(&next) {
                                    lower(&mut out[f.dfa.ix(node)], mk, head + edge(e.cai));
                                }
                            }
                        }
                    }
                }
            }
            // `Multi = (closing pair) [5′ flank] M2R`.
            let clock = std::time::Instant::now();
            for &on in f.dfa.nodes_at(pos) {
                let on_ix = f.dfa.ix(on);
                match wraps {
                    Wraps::Sweep => wrap_outsides(&f, on, &out_multi, &mut out_m2r[on_ix], lambda),
                    #[cfg(test)]
                    Wraps::Enumerate { max_flank } => wrap_outsides_enumerated(
                        &mut f,
                        on,
                        &out_multi,
                        &mut out_m2r[on_ix],
                        lambda,
                        max_flank,
                    ),
                }
            }
            wraps_s += clock.elapsed().as_secs_f64();
            for &on in f.dfa.nodes_at(pos) {
                // `M2` is also an `M`, and starts an `M2R`, unchanged.
                for &mk in f.m2[f.dfa.ix(on)].keys() {
                    let via_m = out_m[f.dfa.ix(on)].get(&mk).copied();
                    let via_r = out_m2r[f.dfa.ix(on)].get(&mk).copied();
                    let v = via_m
                        .unwrap_or(f64::INFINITY)
                        .min(via_r.unwrap_or(f64::INFINITY));
                    if v.is_finite() {
                        out_m2[f.dfa.ix(on)].insert(mk, v);
                    }
                }
            }
            let clock = std::time::Instant::now();
            // Closed pairs ending here: an exterior branch (`C = C + P`), the inner
            // pair of a single-branch loop, a multi-loop branch (`M = P`), or the
            // right branch of a bifurcation (`M2 = M + P`). Every head of a pair
            // lies in a later column or is a multi-loop list at this one, so the
            // column's pairs are independent; what they hand to earlier columns
            // (`C` and `M` at their opening) is merged after them by `min`.
            // A pair the bound rules out has every design through it above
            // the ceiling, so all it would hand on loses to the ceiling caps
            // below; it is skipped, as the fold skips it.
            let cells: Vec<(Node, PKey, Cell)> = f
                .dfa
                .nodes_at(pos)
                .iter()
                .flat_map(|&on| f.p[f.dfa.ix(on)].iter().map(move |(&k, &c)| (on, k, c)))
                .filter(|&(on, k, c)| !f.beyond_bound(on, &k, c))
                .collect();
            let heads = Heads {
                c: &out_c,
                p: &out_p,
                m: &out_m,
                m2: &out_m2,
                m_by_outer: &m_by_outer,
            };
            let opens = skip.map(|_| &live);
            let sinks = fan_pairs(pool, threads, &cells, |sink, w, (on, key, cell)| {
                pair_outside(&f, &heads, (lambda, opens), sink, w, (on, key, cell));
            });
            for sink in sinks {
                through_multiloops += sink.multiloops;
                skipped += sink.skipped;
                for (on, key, v) in sink.p {
                    out_p[on].insert(key, v);
                }
                for (open, q, v) in sink.c {
                    lower(&mut out_c[open], q, v);
                }
                for (open, mk, v) in sink.m {
                    lower(&mut out_m[open], mk, v);
                }
            }
            if let Some(c) = skip {
                for &on in f.dfa.nodes_at(pos) {
                    let ix = f.dfa.ix(on);
                    for (key, &out) in &out_p[ix] {
                        let Some(cell) = f.p[ix].get(key) else {
                            continue;
                        };
                        if out + score(cell) <= c + BOUND_SLACK {
                            let closes = &mut live.0[key.open.node().pos as usize];
                            if closes.last() != Some(&pos) {
                                closes.push(pos);
                            }
                        }
                    }
                }
            }
            for &on in f.dfa.nodes_at(pos) {
                // A multi-loop closes into the pair with the same key.
                for &key in f.multi[f.dfa.ix(on)].keys() {
                    let Some(&head) = out_p[f.dfa.ix(on)].get(&key) else {
                        continue;
                    };
                    let added = lambda * f64::from(f.multi_close_cost(key.pair()).0);
                    out_multi[f.dfa.ix(on)].insert(key, head + added);
                }
            }
            pairs_s += clock.elapsed().as_secs_f64();
        }
    };
    if threads > 1 {
        Pool::scoped(threads, |pool| pass(Some(pool)));
    } else {
        pass(None);
    }
    if crate::dev_env::var_os(crate::dev_env::Setting::MaoTimes).is_some() {
        eprintln!("motif-free outside at λ {lambda}: inside {inside_s:.2} s, wraps {wraps_s:.2} s, pairs {pairs_s:.2} s, sweep skipped {skipped} positions");
    }
    // Under a prune, a pair whose every design scores above the ceiling may
    // have lost the instances that carry its outside. Its through value is
    // above the ceiling, so the ceiling less its inside is still a floor; a
    // pair whose best design is within the ceiling kept every cell on it, and
    // its value is exact.
    if let Some(ceiling) = ceiling {
        for (column, outs) in f.p.iter().zip(out_p.iter_mut()) {
            for (key, cell) in column {
                let cap = ceiling - score(cell);
                let slot = outs.entry(*key).or_insert(cap);
                *slot = slot.min(cap);
            }
        }
    }
    Outside {
        fold: f,
        p: out_p,
        m2r: out_m2r,
        optimum,
        through_multiloops,
        skipped,
    }
}

/// A multi-loop opening: the node, the nucleotide it spells and the AC state
/// before it.
type Opening = (Node, u8, u32);

/// The outside of every `M2R` core at frontier `on` through a multi-loop
/// wrap, lowered into `out_m2r` (the frontier's own outside map).
///
/// A wrap adds the opening edge, the 5′ flank and the closing edge to its core
/// and nothing else, so it depends on the core only through the frontier's AC
/// state and the core's 5′ boundary `(inner, q_inner)`. Per frontier state the
/// closing edges fold into one value per opening (node, nucleotide, AC state),
/// and one left-to-right sweep carries the least opening-plus-flank value to
/// every boundary node and state; a core reads its own. This mirrors the
/// forward fold's long-flank sweep, which keeps the best core per opening, and
/// covers every flank length in one pass instead of an enumeration of every
/// opening per core.
fn wrap_outsides(
    f: &Folder,
    on: Node,
    out_multi: &[FxHashMap<PKey, f64>],
    out_m2r: &mut FxHashMap<MKey, f64>,
    lambda: f64,
) {
    let edge = |cai: f64| -(1.0 - lambda) * cai * 100.0;
    let mut by_state: FxHashMap<u32, Vec<MKey>> = FxHashMap::default();
    for &core in f.m2r[f.dfa.ix(on)].keys() {
        by_state.entry(core.q_outer).or_default().push(core);
    }
    // The multi-loop outsides each closing edge out of `on` reaches, by the
    // outer automaton state they were closed in.
    let closing: Vec<_> = f
        .dfa
        .right(on)
        .iter()
        .map(|&eq| {
            let mut by: FxHashMap<u32, Vec<(PKey, f64)>> = FxHashMap::default();
            for (key, &head) in &out_multi[f.dfa.ix(eq.to)] {
                if key.close.node() == on && key.nuc_close == eq.nuc as u8 {
                    by.entry(key.q_outer).or_default().push((*key, head));
                }
            }
            (eq, by)
        })
        .collect();
    for (q_frontier, mut cores) in by_state {
        // Each opening's least outside over the closing edges.
        let mut openings: FxHashMap<Opening, f64> = FxHashMap::default();
        for (eq, by) in &closing {
            let Some(q_outer) = f.ac.step_edge(on, q_frontier, *eq) else {
                continue;
            };
            for &(key, head) in by.get(&q_outer).into_iter().flatten() {
                let slot = openings
                    .entry((key.open.node(), key.nuc_open, key.q_open))
                    .or_insert(f64::INFINITY);
                *slot = slot.min(head + edge(eq.cai));
            }
        }
        let Some(first) = openings.keys().map(|o| o.0.pos).min() else {
            continue;
        };
        let mut seeds: Vec<Vec<(Opening, f64)>> = vec![Vec::new(); on.pos as usize];
        for (&o, &v) in &openings {
            seeds[o.0.pos as usize].push((o, v));
        }
        cores.sort_unstable_by_key(|c| c.inner.node().pos);
        let last = cores.last().map_or(0, |c| c.inner.node().pos);
        let mut next_core = cores.partition_point(|c| c.inner.node().pos <= first);
        // The least value reaching each (node, AC state) at the sweep's
        // position: an opening, its edge and the flank spelled so far.
        let mut reach: FxHashMap<(Node, u32), f64> = FxHashMap::default();
        for t in first + 1..=last {
            let mut ahead: FxHashMap<(Node, u32), f64> = FxHashMap::default();
            let mut lower = |at: (Node, u32), v: f64| {
                let slot = ahead.entry(at).or_insert(f64::INFINITY);
                *slot = slot.min(v);
            };
            // The flank grows by the unpaired base `t − 1`.
            if !f.must.is_must(t - 1) {
                for (&(x, q), &v) in &reach {
                    for &e in f.dfa.right(x) {
                        if let Some(q2) = f.ac.step_edge(x, q, e) {
                            lower((e.to, q2), v + edge(e.cai));
                        }
                    }
                }
            }
            // Openings at `t − 1` start an empty flank.
            for &((pn, nuc, q_open), v) in &seeds[(t - 1) as usize] {
                for &ep in f.dfa.right(pn) {
                    if ep.nuc as u8 != nuc {
                        continue;
                    }
                    if let Some(q_ae) = f.ac.step_edge(pn, q_open, ep) {
                        lower((ep.to, q_ae), v + edge(ep.cai));
                    }
                }
            }
            reach = ahead;
            while next_core < cores.len() && cores[next_core].inner.node().pos == t {
                let core = cores[next_core];
                next_core += 1;
                if let Some(&v) = reach.get(&(core.inner.node(), core.q_inner)) {
                    let slot = out_m2r.entry(core).or_insert(f64::INFINITY);
                    *slot = slot.min(v);
                }
            }
        }
    }
}

/// [`wrap_outsides`] by enumeration: every opening of every core whose 5′
/// flank is at most `max_flank` long, its flank filled per core.
#[cfg(test)]
fn wrap_outsides_enumerated(
    f: &mut Folder,
    on: Node,
    out_multi: &[FxHashMap<PKey, f64>],
    out_m2r: &mut FxHashMap<MKey, f64>,
    lambda: f64,
    max_flank: u32,
) {
    let edge = |cai: f64| -(1.0 - lambda) * cai * 100.0;
    let cores: Vec<MKey> = f.m2r[f.dfa.ix(on)].keys().copied().collect();
    for core in cores {
        let inner = core.inner.node();
        let shortest = inner.pos.saturating_sub(max_flank.saturating_add(1));
        for p in f.must.min_left(inner.pos).max(shortest)..inner.pos {
            for &pn in f.dfa.nodes_at(p) {
                for &ep in f.dfa.right(pn) {
                    for &q_open in f.product.reachable_states(pn) {
                        let Some(q_ae) = f.ac.step_edge(pn, q_open, ep) else {
                            continue;
                        };
                        let (start, len) = f.fill_span(ep.to, q_ae, inner);
                        let Some((_, left_cai)) = (start..start + len)
                            .map(|i| f.fills.at(i))
                            .find(|&(q, _)| q == core.q_inner)
                        else {
                            continue;
                        };
                        for &eq in f.dfa.right(on) {
                            let Some(q_outer) = f.ac.step_edge(on, core.q_outer, eq) else {
                                continue;
                            };
                            let key = PKey::of((pn, ep, q_open), (on, eq, q_outer));
                            let Some(&head) = out_multi[f.dfa.ix(eq.to)].get(&key) else {
                                continue;
                            };
                            let v = head + edge(ep.cai + left_cai + eq.cai);
                            let slot = out_m2r.entry(core).or_insert(f64::INFINITY);
                            *slot = slot.min(v);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodonUsage, Params};

    fn next(state: &mut u64) -> usize {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as usize
    }

    /// An aware pass's floors are floors, and read what the motif set costs.
    ///
    /// Every cell's floor is at most its exact motif-free outside, every
    /// pair's at most the best motif-free design through it, and the optimum
    /// the search certified is the exact one — for the cells and pairs the
    /// pass's fold built and, through the ceiling, for those it did not. Some
    /// cell's floor lies above the unconstrained outside of its pair, which no
    /// plain floor reaches; under a motif no spelling contains, where the two
    /// lattices are one, none does.
    #[test]
    fn an_aware_pass_is_a_floor() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let region = PairConstraint::inert();
        let mut rng = 20_261_023_u64;
        let (mut cells, mut exact, mut below, mut pairs) = (0, 0, 0, 0);
        let (mut unbuilt_cells, mut unbuilt_pairs) = (0, 0);
        let mut tighter = [0, 0];
        let mut refolded = 0;
        let mut skipped = 0;
        for _ in 0..6 {
            const ALPHABET: &[u8] = b"ACDEGHIKLMNPQRSTVWY";
            let protein: String = (0..9)
                .map(|_| ALPHABET[next(&mut rng) % ALPHABET.len()] as char)
                .collect();
            let free = crate::design(&params, &protein, &usage, 0.04)
                .unwrap()
                .sequence;
            let at = next(&mut rng) % (free.len() - 5);
            let aa: Vec<usize> = protein
                .chars()
                .map(|c| derna_codon_dp::codon_table::aa_to_index(c).unwrap())
                .collect();
            let graph = crate::dfa::build(&aa, &usage);
            // A motif the free design spells, and one longer than any spelling.
            let sets = [free[at..at + 5].to_string(), "A".repeat(40)];
            for (control, motif) in sets.iter().enumerate() {
                let ac = crate::ac::Ac::build(std::slice::from_ref(motif)).unwrap();
                let product = ProductDfa::build(&graph, &ac);
                if !product.is_feasible() {
                    continue;
                }
                let product = product.merge_equivalent();
                // A headroom refolds a pass that certified close to its
                // optimum: its floors stay floors and its design is kept.
                for (lambda, headroom) in [(0.02, 0.0), (0.06, 0.0), (0.02, 8.0), (0.06, 8.0)] {
                    let scan = crate::scan::scan_values_with_constraints(
                        &params,
                        &graph,
                        lambda,
                        crate::ScanConfig::default(),
                        region.clone(),
                        MustPairConstraint::inert(),
                    );
                    let plain = crate::outside_scan::OutsideScan::cost_to_go(
                        &params,
                        &graph,
                        &scan,
                        &region,
                        crate::ScanConfig::default().single_max_len,
                        1,
                        false,
                    );
                    let plain_table = pair_table(plain.p.clone(), |o, k| {
                        scan.p[o].get(k).map(|s| s.brd(lambda))
                    });
                    let free_optimum = scan.c[graph.ix(graph.end())].unwrap().brd(lambda);
                    let on = AwareLattice {
                        product: &product,
                        params: &params,
                        region: region.clone(),
                    };
                    let aware = aware_pass(
                        &on,
                        lambda,
                        |ceiling| OutsideBound {
                            outside: &plain_table,
                            chord: None,
                            through: true,
                            zero: &[],
                            motif_free: None,
                            aware: None,
                            ceiling,
                        },
                        (free_optimum, 1.0, headroom),
                        1,
                        true,
                    )
                    .unwrap();
                    // The sweep passing over dead positions leaves every
                    // outside the cap leaves, bit for bit.
                    let fold_at = |skip: bool| {
                        let bound = OutsideBound {
                            outside: &plain_table,
                            chord: None,
                            through: true,
                            zero: &[],
                            motif_free: None,
                            aware: None,
                            ceiling: aware.ceiling,
                        };
                        let (f, end_q) = inside(&on, lambda, Some(bound), None);
                        let how = Reverse {
                            ceiling: Some(aware.ceiling),
                            wraps: Wraps::Sweep,
                            threads: 1,
                            inside_s: 0.0,
                            skip,
                        };
                        reverse(
                            f,
                            end_q.expect("the ceiling keeps the optimum"),
                            lambda,
                            how,
                        )
                    };
                    let (whole, swept) = (fold_at(false), fold_at(true));
                    assert_eq!(whole.skipped, 0);
                    assert!(
                        whole.p == swept.p,
                        "{protein} at λ={lambda}: passing over dead positions moved an outside"
                    );
                    skipped += swept.skipped;
                    let truth =
                        outside_of(&product, &params, lambda, None, Wraps::Sweep, 1).unwrap();
                    let design = aware.design.as_ref().expect("a kept design");
                    assert!(
                        (brd(design.mfe, design.cai, lambda) - truth.optimum).abs() < 1e-6,
                        "{protein} at λ={lambda}: the kept design scores {} against {}",
                        brd(design.mfe, design.cai, lambda),
                        truth.optimum
                    );
                    assert!(
                        (aware.optimum - truth.optimum).abs() < 1e-6,
                        "{protein} at λ={lambda}: optimum {} against {}",
                        aware.optimum,
                        truth.optimum
                    );
                    assert!(aware.ceiling >= truth.optimum + headroom / 2.0 - BOUND_SLACK);
                    // Folded again at the headroom, or certified on a lifted
                    // ceiling after a failed fold.
                    refolded += usize::from(
                        headroom > 0.0
                            && ((aware.ceiling - truth.optimum - headroom).abs() < 1e-6
                                || aware.folds > 1),
                    );
                    let mut truth_through: Vec<FxHashMap<crate::state::PairKey, f64>> =
                        vec![FxHashMap::default(); truth.p.len()];
                    for (o, (column, outs)) in truth.fold.p.iter().zip(&truth.p).enumerate() {
                        for (key, &out) in outs {
                            let v = brd(column[key].mfe, column[key].cai, lambda) + out;
                            let slot = truth_through[o]
                                .entry(key.pair_key())
                                .or_insert(f64::INFINITY);
                            *slot = slot.min(v);
                        }
                    }
                    for (o, column) in aware.outside.iter().enumerate() {
                        for (key, &v) in column {
                            let t = truth.p[o].get(key).copied().unwrap_or(f64::INFINITY);
                            assert!(v <= t + 1e-6, "{protein} at λ={lambda}: cell {v} above {t}");
                            cells += 1;
                            exact += usize::from((v - t).abs() < 1e-6);
                            below += usize::from(v < t - 1e-6);
                            let p = plain.p[o].get(&key.pair_key()).copied();
                            tighter[control] += usize::from(p.is_some_and(|p| v > p + 1e-6));
                        }
                    }
                    for (o, column) in aware.through.iter().enumerate() {
                        for (pair, &v) in column {
                            let t = truth_through[o].get(pair).copied().unwrap_or(f64::INFINITY);
                            assert!(v <= t + 1e-6, "{protein} at λ={lambda}: pair {v} above {t}");
                            pairs += 1;
                        }
                    }
                    for (o, (cells_at, outs)) in truth.fold.p.iter().zip(&truth.p).enumerate() {
                        let on = graph.node_of(o);
                        for (key, &t) in outs {
                            if aware.outside[o].contains_key(key) {
                                continue;
                            }
                            let v = aware.cell_floor(on, key, cells_at[key]);
                            assert!(
                                v <= t + 1e-6,
                                "{protein} at λ={lambda}: unbuilt cell {v} above {t}"
                            );
                            unbuilt_cells += 1;
                        }
                    }
                    for (o, column) in truth_through.iter().enumerate() {
                        for (pair, &t) in column {
                            if aware.through[o].contains_key(pair) {
                                continue;
                            }
                            let v = aware.pair_floor(o, pair);
                            assert!(
                                v <= t + 1e-6,
                                "{protein} at λ={lambda}: unbuilt pair {v} above {t}"
                            );
                            unbuilt_pairs += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "cells {cells} ({exact} exact, {below} capped below), pairs {pairs}; \
             unbuilt cells {unbuilt_cells}, pairs {unbuilt_pairs}; \
             above the plain outside: {} binding, {} control",
            tighter[0], tighter[1]
        );
        assert!(cells > 0 && pairs > 0, "no feasible case");
        assert!(skipped > 0, "the sweep passed over no position");
        assert!(
            refolded > 0,
            "no pass reached its headroom past its first fold"
        );
        assert!(
            unbuilt_cells > 0 && unbuilt_pairs > 0,
            "the ceiling left every cell or every pair built"
        );
        assert!(
            exact > 0 && below > 0,
            "the ceiling never capped or never kept"
        );
        assert!(
            tighter[0] > 0,
            "no aware floor reached past the plain outside"
        );
        assert_eq!(tighter[1], 0, "an unconstraining motif tightened a floor");
    }

    /// The multi-loop wrap sweep, on three threads, equals the serial
    /// enumeration of every opening of every core, on proteins long enough for 5′ flanks past the forward
    /// fold's per-core window, in every core's and every pair's outside. The
    /// enumeration capped at that window must leave some core's outside
    /// higher, or no long flank carried a value and the sweep's long-flank
    /// half went untested.
    #[test]
    fn the_wrap_sweep_is_the_wrap_enumeration() {
        let (params, usage) = (
            Params::load_default().unwrap(),
            CodonUsage::load_default().unwrap(),
        );
        let mut rng = 20_261_020_u64;
        let (mut keys, mut long_carried) = (0, 0);
        for _ in 0..3 {
            const ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
            let protein: String = (0..36)
                .map(|_| ALPHABET[next(&mut rng) % ALPHABET.len()] as char)
                .collect();
            let free = crate::design(&params, &protein, &usage, 0.3)
                .unwrap()
                .sequence;
            let at = next(&mut rng) % (free.len() - 6);
            let motifs = vec![free[at..at + 6].to_string()];
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
            for lambda in [0.02, 0.3] {
                let run = |wraps, threads| {
                    outside_of(&product, &params, lambda, None, wraps, threads).unwrap()
                };
                let swept = run(Wraps::Sweep, 3);
                let every = run(
                    Wraps::Enumerate {
                        max_flank: u32::MAX,
                    },
                    1,
                );
                let short = run(
                    Wraps::Enumerate {
                        max_flank: SINGLE_MAX_LEN - 1,
                    },
                    1,
                );
                for (s, e) in swept.p.iter().zip(&every.p) {
                    assert_eq!(
                        s.len(),
                        e.len(),
                        "{protein} at λ={lambda}: pair keys differ"
                    );
                    for (key, &v) in e {
                        let got = s.get(key).copied().unwrap_or(f64::INFINITY);
                        assert!(
                            (got - v).abs() < 1e-6,
                            "{protein} at λ={lambda}: pair outside {got} by the sweep, {v} enumerated"
                        );
                        keys += 1;
                    }
                }
                for ((s, e), c) in swept.m2r.iter().zip(&every.m2r).zip(&short.m2r) {
                    assert_eq!(
                        s.len(),
                        e.len(),
                        "{protein} at λ={lambda}: core keys differ"
                    );
                    for (key, &v) in e {
                        let got = s.get(key).copied().unwrap_or(f64::INFINITY);
                        assert!(
                            (got - v).abs() < 1e-6,
                            "{protein} at λ={lambda}: core outside {got} by the sweep, {v} enumerated"
                        );
                        keys += 1;
                        let capped = c.get(key).copied().unwrap_or(f64::INFINITY);
                        long_carried += usize::from(capped > v + 1e-6);
                    }
                }
            }
        }
        eprintln!("keys {keys}, core outsides carried by a long flank {long_carried}");
        assert!(keys > 0, "no feasible case");
        assert!(
            long_carried > 0,
            "no core took its outside through a 5′ flank of 30 or more"
        );
    }
}

/// The outside tables a closed pair's heads are read from, as the reverse
/// pass has finished them for every later column.
struct Heads<'t> {
    c: &'t [FxHashMap<u32, f64>],
    p: &'t [FxHashMap<PKey, f64>],
    m: &'t [FxHashMap<MKey, f64>],
    m2: &'t [FxHashMap<MKey, f64>],
    /// The fold's `M` cells at each node, by their outer automaton state.
    m_by_outer: &'t [FxHashMap<u32, Vec<(MKey, Cell)>>],
}

/// What one worker's closed pairs found: each pair's outside at its node
/// index, the lowered `C` and `M` outsides at their opening's node index, and
/// how many pairs took their outside from a multi-loop rule.
#[derive(Default)]
struct PairSink {
    p: Vec<(usize, PKey, f64)>,
    c: Vec<(usize, u32, f64)>,
    m: Vec<(usize, MKey, f64)>,
    multiloops: usize,
    /// Left positions the single-loop sweep passed over ([`LiveOpens`]).
    skipped: usize,
}

/// For each left position, the positions its closed pairs close into whose
/// through value, inside plus the finished outside, is within a pass's
/// ceiling: the *live* pairs, newest (leftmost) last.
///
/// The reverse pass finishes columns right to left, so when a column's pairs
/// are read every pair that could enclose one of them in a single-branch loop
/// is finished. A left position with no live pair closing inside the loop's
/// window offers only pairs above the ceiling, each of whose contributions
/// to the enclosed pair puts it above the ceiling too, where the cap after
/// the pass takes over; and a pair whose best design is within the ceiling
/// has every pair enclosing it on that design live, its value read
/// unchanged. So the sweep passes such a position over, and every outside
/// the cap leaves is the same.
pub(crate) struct LiveOpens(Vec<Vec<u32>>);

impl LiveOpens {
    /// Whether a live pair opening at `p` closes into a position in
    /// `(lo, hi]`.
    pub(crate) fn closes_within(&self, p: u32, lo: u32, hi: u32) -> bool {
        let closes = &self.0[p as usize];
        let at = closes.partition_point(|&q| q > hi);
        closes.get(at).is_some_and(|&q| q > lo)
    }
}

/// Whether the reverse pass of a pruned fold passes over dead left
/// positions ([`LiveOpens`]): yes, unless a development build turns it off.
fn sweep_skip() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::SweepSkip).map_or(true, |v| v != "0")
}

/// Run `body` over `items` on `pool` (serially without one), each worker
/// claiming small chunks into its own [`PairSink`].
fn fan_pairs<T: Copy + Send + Sync>(
    pool: Option<&Pool>,
    threads: usize,
    items: &[T],
    body: impl Fn(&mut PairSink, &mut Writes, T) + Sync,
) -> Vec<PairSink> {
    let pool = match pool {
        Some(pool) if items.len() >= PARALLEL_MIN => pool,
        _ => {
            let (mut sink, mut w) = (PairSink::default(), Writes::default());
            for &item in items {
                body(&mut sink, &mut w, item);
            }
            return vec![sink];
        }
    };
    let n_workers = threads.min(items.len()).max(1);
    let chunk = items.len().div_ceil(n_workers * CHUNKS_PER_WORKER).max(1);
    let next = AtomicUsize::new(0);
    let sinks: Mutex<Vec<PairSink>> = Mutex::new(Vec::with_capacity(n_workers));
    pool.run(n_workers, &|_| {
        let (mut sink, mut w) = (PairSink::default(), Writes::default());
        loop {
            let start = next.fetch_add(1, Ordering::Relaxed) * chunk;
            if start >= items.len() {
                break;
            }
            for &item in &items[start..(start + chunk).min(items.len())] {
                body(&mut sink, &mut w, item);
            }
        }
        sinks.lock().unwrap().push(sink);
    });
    sinks.into_inner().unwrap()
}

/// The outside of the closed pair `key` at `on` with inside `cell`, and what
/// it hands to the exterior prefix and the multi-loop list on its left.
#[allow(clippy::too_many_arguments)]
fn pair_outside(
    f: &Folder,
    heads: &Heads,
    (lambda, live): (f64, Option<&LiveOpens>),
    sink: &mut PairSink,
    w: &mut Writes,
    (on, key, cell): (Node, PKey, Cell),
) {
    let score = |c: &Cell| brd(c.mfe, c.cai, lambda);
    let mut outside = f64::INFINITY;
    let open = key.open.node();
    if let (Some(&prefix), Some(&head)) = (
        f.c[f.dfa.ix(open)].get(&key.q_open),
        heads.c[f.dfa.ix(on)].get(&key.q_outer),
    ) {
        let cand = Cell {
            mfe: prefix.mfe + cell.mfe + f.terminal_au(key.pair()),
            cai: prefix.cai + cell.cai,
        };
        outside = outside.min(head + score(&cand) - score(&cell));
        sink.c.push((
            f.dfa.ix(open),
            key.q_open,
            head + score(&cand) - score(&prefix),
        ));
    }
    w.reset(1);
    sink.skipped += f.enumerate_outer_pairs_ro(w, (on, key, cell), live);
    for &(outer, okey, cand) in &w.p.buckets[0] {
        if let Some(&head) = heads.p[f.dfa.ix(outer.node())].get(&okey) {
            outside = outside.min(head + score(&cand) - score(&cell));
        }
    }
    let plain = outside;
    let stem = f.m_stem_cost(key.pair());
    let branch = MKey {
        inner: key.open,
        q_inner: key.q_open,
        q_outer: key.q_outer,
    };
    if let Some(&head) = heads.m[f.dfa.ix(on)].get(&branch) {
        outside = outside.min(head + lambda * f64::from(stem.0));
    }
    let partners = heads.m_by_outer[f.dfa.ix(open)].get(&key.q_open);
    for &(mk, ref mcell) in partners.into_iter().flatten() {
        let joined = MKey {
            q_outer: key.q_outer,
            ..mk
        };
        let Some(&head) = heads.m2[f.dfa.ix(on)].get(&joined) else {
            continue;
        };
        let cand = Cell {
            mfe: mcell.mfe + cell.mfe + stem,
            cai: mcell.cai + cell.cai,
        };
        outside = outside.min(head + score(&cand) - score(&cell));
        sink.m
            .push((f.dfa.ix(open), mk, head + score(&cand) - score(mcell)));
    }
    if outside.is_finite() {
        sink.multiloops += usize::from(outside < plain - BOUND_SLACK);
        sink.p.push((f.dfa.ix(on), key, outside));
    }
}
