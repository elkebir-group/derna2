//! The 5′→3′ CKY scan over DFA nodes.
//!
//! A direct generalization of `rna-linearfold`'s position-indexed Zuker
//! DP to the codon DFA: the same six tables, the same Turner-2004 loop
//! energies (from `rna-mfe`, so a λ=1 single-path scan reproduces
//! `rna_linearfold::fold` exactly), now traversing DFA edges instead of
//! incrementing a sequence position. Each closed-pair cell additionally
//! keys on its boundary nucleotides, and a CAI accumulator rides
//! alongside the energy. Loop interiors that the grammar jumps over are
//! filled by the best-CAI DFA path between their boundaries (the
//! broken-codon bookkeeping); on a single-path DFA that path is unique,
//! so the energies are exact.
//!
//! By default the scan runs the exact lattice (`b → ∞`); per-column pruning
//! is applied only under an opt-in finite beam (see `scan`).

use rna_mfe::loops::{bulge_energy, pair_from_nucleotides, stack_energy, BulgeLoop, StackPair};
use rna_mfe::params::BasePair;
use rna_mfe::{Energy, Nucleotide, Params};

use crate::dfa::{Dfa, Edge, Node};
use crate::fold_pool::Pool;
use crate::hash::FxHashMap;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::profile::{FanSite, Phase, Profile};

use crate::hash::FxHashSet;
use crate::interior::{
    corner_scores_into, score_hairpin, score_internal_from, Corner, LoopPairs, OneByNRows,
    ScoredStrand,
};
use crate::lz_index::{HEntry, HIndex, OneSideEntry, OneSideIndex, OneSideKIndex};
use crate::must_pair::MustPairConstraint;
use crate::region::PairConstraint;
use crate::state::{Back, MultiOpenKey, PairKey, PairRef, SpanRef, State, Value};
use crate::tables::{IndexSizes, ScanCut, Tables};
use crate::{InternalLoopMode, ScanConfig};

mod pull_fill;
use pull_fill::CellBest;

/// Smallest number of unpaired bases inside a hairpin (the "no sharp
/// turn" rule): `b - a - 1 >= 3`.
const MIN_HAIRPIN_LOOP: u32 = 3;

// The single-branch loop cap is per-scan rather than a constant: see
// [`ScanConfig::single_max_len`] and [`crate::DEFAULT_SINGLE_MAX_LEN`]. It is
// read through `Scan::single_max_len` at every site that bounds a flank.

/// Outer-cell indices a `single_close_pull` worker claims per atomic step —
/// large enough to amortize the `fetch_add`, small enough to balance the tail.
const PULL_CHUNK: u32 = 32;

/// Direct-family outer cells a `single_close_pull` worker claims per atomic
/// step. Each scores a whole window of flank shapes, so they are dealt singly
/// to keep one worker from holding a long run of them at the end.
const PULL_DIRECT_CHUNK: u32 = 1;

/// Below this many outer cells at a position the parallel pull runs serially:
/// short positions carry too little work to repay thread-spawn + fold overhead.
const PULL_PARALLEL_MIN: u32 = 256;

/// Closed pairs a `p_beam` push worker claims per atomic step.
const PUSH_CHUNK: usize = 16;

/// Below this many closed pairs at a position the parallel push runs serially.
const PUSH_PARALLEL_MIN: usize = 64;

/// Run the CKY scan over `dfa` at fixed `lambda`, returning the filled
/// tables. The optimum is read from `C[end]`.
///
/// `beam` is the per-column pruning policy: a finite [`Beam`](crate::Beam) prunes the
/// lattice to a beam-search heuristic (an opt-in speed knob, which can
/// change the optimum), while [`Beam::exact`](crate::Beam::exact) runs the exact (`b → ∞`) lattice.
/// Each column is pruned exactly once, immediately before it is consumed, so
/// every cell a surviving derivation points back to also survives — the
/// traceback never dereferences a pruned cell.
pub fn scan(params: &Params, dfa: &Dfa, lambda: f64, config: ScanConfig) -> Tables {
    scan_with_region(params, dfa, lambda, config, PairConstraint::inert())
}

/// Run the CKY scan as [`scan`], enforcing the region-pair `region` constraint at the
/// single `P`-write choke point ([`Tables::update_p`]). [`PairConstraint::inert`]
/// is byte-for-byte [`scan`]; an active constraint forbids or penalizes the pairs its
/// families reject. The constraint's verdict on an outer pair is constant within each
/// interior-loop reduction (Lyngsø–Zuker skewed, one-sided, `multi_open` `C+D`),
/// so those reductions still separate and the constrained fold stays exact — a
/// forbidden pair simply never materializes a `P` cell for the reduction to read.
pub(crate) fn scan_with_region(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    config: ScanConfig,
    region: PairConstraint,
) -> Tables {
    scan_with_constraints(
        params,
        dfa,
        lambda,
        config,
        region,
        MustPairConstraint::inert(),
    )
}

/// Run the CKY scan as [`scan_with_region`], additionally requiring every
/// position `must_pair` names to carry a base pair.
///
/// The predicate is enforced where a base is *left unpaired* rather than where a
/// pair is written: each site tests that the range it is about to declare
/// unpaired holds no required position. Because those ranges are delimited by
/// coordinates the enclosing closure already enumerates, the test lands as a
/// **narrowing of an enumeration range** — the hairpin's closing scan, the
/// single-branch flank scans, and the `k`-windows the interior reductions
/// binary-search — rather than as a filter over an already-minimized set. The
/// reductions therefore separate exactly as they do unconstrained, and a
/// constrained fold explores strictly fewer candidates.
///
/// [`MustPairConstraint::inert`] is byte-for-byte [`scan_with_region`].
pub(crate) fn scan_with_constraints(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    config: ScanConfig,
    region: PairConstraint,
    must_pair: MustPairConstraint,
) -> Tables {
    scan_tables(
        params,
        dfa,
        lambda,
        config,
        Constraints {
            region,
            must_pair,
            cut: None,
            pull_push: true,
            values_only: false,
        },
    )
}

/// [`scan_with_constraints`] with values-only `M` and `M2` columns
/// ([`Tables::values_only`]): the same values in every table, for a scan whose
/// tables only feed an outside pass. No traceback may read them.
pub(crate) fn scan_values_with_constraints(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    config: ScanConfig,
    region: PairConstraint,
    must_pair: MustPairConstraint,
) -> Tables {
    scan_tables(
        params,
        dfa,
        lambda,
        config,
        Constraints {
            region,
            must_pair,
            cut: None,
            pull_push: true,
            values_only: true,
        },
    )
}

/// [`scan_with_constraints`] with the single-branch closures **pushed** rather than
/// pulled at their outer position: the route every scan took before the pull,
/// kept so the pull can be checked against it on the same tables. The two
/// agree cell for cell; the pull is the faster of the two.
pub(crate) fn scan_with_constraints_pushing(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    config: ScanConfig,
    region: PairConstraint,
    must_pair: MustPairConstraint,
) -> Tables {
    scan_tables(
        params,
        dfa,
        lambda,
        config,
        Constraints {
            region,
            must_pair,
            cut: None,
            pull_push: false,
            values_only: false,
        },
    )
}

/// [`scan_with_region`] with every column cut at `cut` (see [`ScanCut`]): it
/// finds the optimum whenever the optimum meets the cut's threshold, and its
/// tables are not exact.
pub(crate) fn scan_cut(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    config: ScanConfig,
    cut: ScanCut,
    region: PairConstraint,
) -> Tables {
    scan_tables(
        params,
        dfa,
        lambda,
        config,
        Constraints {
            region,
            must_pair: MustPairConstraint::inert(),
            cut: Some(cut),
            pull_push: true,
            values_only: false,
        },
    )
}

/// What a scan enforces and how it routes its single-branch closures, beside
/// the lattice and weight every scan takes.
struct Constraints {
    region: PairConstraint,
    must_pair: MustPairConstraint,
    cut: Option<ScanCut>,
    /// Whether the single-branch closures are pulled at their enclosing
    /// position rather than pushed from each enclosed pair's own.
    pull_push: bool,
    /// Whether the tables keep only values ([`Tables::values_only`]).
    values_only: bool,
}

/// Whether a scan in `mode` runs the column fill: the `Lz` default
/// (`DERNA_PUSH_SCAN` selects the push scan instead); in `Naive` mode the push
/// is the default and `DERNA_PULL_FILL` selects the fill. Independent of the
/// lattice, so it is read before one is monomorphized over.
fn fills(mode: InternalLoopMode) -> bool {
    match mode {
        InternalLoopMode::Lz => std::env::var_os("DERNA_PUSH_SCAN").is_none(),
        InternalLoopMode::Naive => std::env::var_os("DERNA_PULL_FILL").is_some(),
    }
}

fn scan_tables(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    config: ScanConfig,
    constraints: Constraints,
) -> Tables {
    let Constraints {
        region,
        must_pair,
        cut,
        pull_push,
        values_only,
    } = constraints;
    let run = |fan_pool: Option<&Pool>| {
        fn at<const S: usize>(
            (params, dfa, lambda, config): (&Params, &Dfa, f64, ScanConfig),
            (must_pair, region, cut, values_only): (
                MustPairConstraint,
                PairConstraint,
                Option<ScanCut>,
                bool,
            ),
            (pull_push, fan_pool): (bool, Option<&Pool>),
        ) -> Tables {
            let mut s = Scan::<S>::new(params, dfa, config, must_pair, fan_pool);
            s.pull_push = pull_push;
            let mut lane = Lane::new(&s, lambda, config, region, cut, values_only);
            s.seed_lane(&mut lane);
            for pos in 0..=dfa.seq_len {
                s.step(&mut lane, pos);
            }
            lane.finish()
        }
        let args = (
            (params, dfa, lambda, config),
            (must_pair.clone(), region.clone(), cut, values_only),
            (pull_push, fan_pool),
        );
        match dfa.stride() {
            2 => at::<2>(args.0, args.1, args.2),
            3 => at::<3>(args.0, args.1, args.2),
            n => unreachable!("a lattice of lane stride {n}"),
        }
    };
    // The column fill fans out up to three times per position; one pool for
    // the whole scan replaces a thread start per worker per fan-out.
    let push_pool = std::env::var_os("DERNA_PUSH_POOL").is_some();
    if (fills(config.mode) || push_pool) && config.threads > 1 {
        Pool::scoped(config.threads - 1, |p| run(Some(p)))
    } else {
        run(None)
    }
}

/// Two weights' scans run position by position in one loop: at each position
/// every phase runs for the first weight and then for the second, so both
/// weights' tables are live together, as they would be in a scan that carried
/// two weights through one traversal. Each returned table is byte-identical to
/// [`scan`] at its weight: the two lanes share no mutable state.
///
/// An instrument for pricing a two-weight scan's memory traffic against two
/// scans run one after the other; it saves no work.
pub fn scan_lockstep(
    params: &Params,
    dfa: &Dfa,
    lambdas: [f64; 2],
    config: ScanConfig,
) -> [Tables; 2] {
    let s = Scan::<2>::new(params, dfa, config, MustPairConstraint::inert(), None);
    let mut lanes =
        lambdas.map(|lambda| Lane::new(&s, lambda, config, PairConstraint::inert(), None, false));
    for lane in &mut lanes {
        s.seed_lane(lane);
    }
    for pos in 0..=dfa.seq_len {
        for lane in &mut lanes {
            s.step(lane, pos);
        }
    }
    lanes.map(Lane::finish)
}

/// One weight's mutable scan state: its tables, its profile, its per-worker
/// candidate buffers and its pull indexes.
struct Lane {
    t: Tables,
    prof: Profile,
    // One candidate buffer per worker, reused at every position and both
    // fanned-out sites: each is cleared after its fold and keeps its capacity,
    // so the buffers grow once per scan instead of once per position.
    pool: Vec<Writes>,
    // Per-size-diagonal `H` index for the `Lz` skewed-loop reduction, and the
    // per-column index for the one-sided asymmetric-family reduction; both stay
    // empty (and untouched) in `Naive` mode, except that the column fill reads
    // the per-column index there too.
    hx: HIndex,
    osx: OneSideIndex,
    okx: OneSideKIndex,
}

impl Lane {
    fn new<const S: usize>(
        s: &Scan<'_, S>,
        lambda: f64,
        config: ScanConfig,
        region: PairConstraint,
        cut: Option<ScanCut>,
        values_only: bool,
    ) -> Self {
        let mut t = if values_only {
            Tables::values_only(s.dfa, lambda, config.beam, region)
        } else {
            Tables::new(s.dfa, lambda, config.beam, region)
        };
        if let Some(cut) = cut {
            t.set_cut(cut);
        }
        let mut prof = Profile::from_env(s.threads);
        prof.begin();
        t.update_c(
            s.dfa.start(),
            State::new(Energy::ZERO, 0.0, Back::ExteriorEmpty),
        );
        Self {
            t,
            prof,
            pool: (0..s.threads).map(|_| Writes::default()).collect(),
            hx: HIndex::new(),
            osx: OneSideIndex::new(s.dfa.seq_len),
            okx: OneSideKIndex::new(s.dfa.seq_len),
        }
    }

    fn finish(self) -> Tables {
        let Self {
            mut t,
            mut prof,
            hx,
            osx,
            okx,
            ..
        } = self;
        prof.end();
        t.set_profile(prof);
        t.set_index_sizes(IndexSizes {
            h_bytes: hx.approx_bytes(),
            h_entries: hx.entries(),
            one_side_bytes: osx.approx_bytes(),
            one_side_k_bytes: okx.approx_bytes(),
            one_side_entries: osx.entries(),
            one_side_k_entries: okx.entries(),
        });
        t
    }
}

impl<'a, const S: usize> Scan<'a, S> {
    fn new(
        params: &'a Params,
        dfa: &'a Dfa,
        config: ScanConfig,
        must: MustPairConstraint,
        fan_pool: Option<&'a Pool>,
    ) -> Self {
        let ninio_skew = {
            let n = &params.ninio;
            ((n.max_ninio + n.ninio_37 - 1) / n.ninio_37) as usize
        };
        let fill_shapes = pull_fill::fill_shapes(config.mode, config.single_max_len, ninio_skew);
        debug_assert_eq!(
            S,
            dfa.stride(),
            "a scan monomorphized over the lattice's stride"
        );
        Scan {
            params,
            dfa,
            mode: config.mode,
            threads: config.threads.max(1),
            validate_oneside: std::env::var_os("DERNA_ONESIDE_VALIDATE").is_some(),
            pull_fill: fills(config.mode),
            fan_pool,
            fan_all: std::env::var_os("DERNA_FILL_FAN_ALL").is_some(),
            serial_index: std::env::var_os("DERNA_FILL_SERIAL_INDEX").is_some(),
            ninio_skew,
            fill_n1_hi: pull_fill::fill_n1_hi(config.single_max_len, &fill_shapes),
            fill_shapes,
            one_by_n: OneByNRows::of(params),
            fan_workers_cap: std::env::var("DERNA_FAN_WORKERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&cap| cap >= 1),
            must,
            single_max_len: config.single_max_len,
            pull_push: true,
        }
    }

    fn seed_lane(&self, lane: &mut Lane) {
        let m = lane.prof.mark();
        self.seed_hairpins(&mut lane.t);
        lane.prof.record(Phase::SeedHairpins, m);
    }

    /// Run every phase of the scan at `pos` on one lane.
    fn step(&self, lane: &mut Lane, pos: u32) {
        let s = self;
        let dfa = self.dfa;
        let Lane {
            t,
            prof,
            pool,
            hx,
            osx,
            okx,
        } = lane;
        // A shared multi-loop opening carries its 3′ unpaired stretch from the
        // core's right boundary to whichever frontier closes it. Once the scan
        // has passed a required position, every opening still pooled would have
        // to skip over it to reach any later frontier, so the whole pool is
        // evicted — the opening-side half of the `C + D` split's constraint. The
        // frontier-side half needs no test afterwards: what survives cannot span
        // a required position. Ordered before activation, so an opening whose
        // boundary is this position (its stretch starts here, past the required
        // one) is not caught by the eviction.
        if pos > 0 && s.must.is_must(pos - 1) {
            t.multi_open_clear();
        }
        t.multi_open_activate(pos);
        // `Multi[pos]` is complete (all writers were earlier positions) and
        // is consumed next by `multi_close`.
        s.timed_nodes(prof, pos, Phase::PruneGrouped, |jn| t.prune_multi(jn));
        // The single-branch push into `P[pos]` is pulled here, ahead of
        // `multi_close`, which is where the push's last fold left it.
        if s.pulls_push() && pos > 0 {
            let m = prof.mark();
            s.push_pull(t, (osx, okx), pos);
            prof.record(Phase::PToSingle, m);
        }
        s.timed_nodes(prof, pos, Phase::MultiClose, |jn| {
            s.multi_close(t, jn);
        });
        // In `Lz` mode, the generic internal-loop closure is pulled here (it
        // is disabled on the push path): each `P[pos]` outer cell reads the
        // already-finalized inner pairs at earlier positions — directly for
        // the small/balanced shapes, and via the `H` index `hx` for the
        // skewed-large shapes. Ordered after `multi_close` and before
        // `prune_p`. A no-op in `Naive` mode. The column fill produces every
        // single-branch closure of the column here instead.
        if s.pull_fill {
            s.pull_fill_column(t, hx, osx, okx, pos, prof, pool);
        } else {
            s.single_close_pull(t, hx, osx, okx, pos, prof, pool);
        }
        // `P[pos]` is now complete (hairpins, earlier single-closures, and
        // this position's `multi_close`); it is consumed next by `p_beam`.
        s.timed_nodes(prof, pos, Phase::PruneGrouped, |jn| t.prune_p(jn));
        if s.pull_fill && s.fan_pool.is_some() && !s.serial_index && s.pull_generic() && pos > 0 {
            // `P[pos]` is final, and nothing the bifurcation or the `M2` fill
            // reads is in the pull indexes, so the column is indexed — its two
            // independent halves as two tasks — while the pool fills those two
            // (see `Lead`).
            let cells = s.finalized_cells(t, pos);
            let lambda = t.lambda();
            let spent = Mutex::new(Duration::ZERO);
            let (cells, spent, l) = (&cells, &spent, pos - 1);
            let lead = Lead::new(vec![
                Box::new(move || {
                    let began = Instant::now();
                    s.index_h(cells, lambda, hx, l);
                    *spent.lock().unwrap() += began.elapsed();
                }),
                Box::new(move || {
                    let began = Instant::now();
                    s.index_oneside(cells, lambda, osx, okx, l);
                    *spent.lock().unwrap() += began.elapsed();
                }),
            ]);
            s.p_beam_all(t, pos, prof, pool, Some(&lead));
            s.timed_nodes(prof, pos, Phase::PruneGrouped, |jn| t.prune_m2(jn));
            s.m2_fill_column(t, pos, prof, Some(&lead));
            let m = prof.mark();
            run_lead(&lead);
            prof.record(Phase::IndexFinalizedP, m);
            drop(lead);
            prof.add_overlapped(*spent.lock().unwrap());
        } else {
            s.p_beam_all(t, pos, prof, pool, None);
            // `P[pos]` is now final: record each cell's reduction value `H` on
            // its size diagonal so later positions' pulls can range-min over it
            // (`Lz`).
            s.index_finalized_p(t, hx, osx, okx, pos, prof);
            // `M2[pos]` is filled only by this position's `p_beam`; consumed
            // next by `m2_beam`.
            s.timed_nodes(prof, pos, Phase::PruneGrouped, |jn| t.prune_m2(jn));
            if s.pull_fill {
                s.m2_fill_column(t, pos, prof, None);
            } else {
                s.timed_nodes(prof, pos, Phase::M2Beam, |jn| s.m2_beam(t, jn));
            }
        }
        // `M[pos]`'s last writer is this position's `m2_beam`; it is consumed
        // by `m_extend` below and by later positions' bifurcations.
        s.timed_nodes(prof, pos, Phase::PruneGrouped, |jn| t.prune_m(jn));
        // Close every active opening against this position's frontier nodes.
        // Snapshot once per position — after `m2_beam` has seeded any aligned
        // openings here, and before the borrow conflict of reading
        // `multi_open` while writing `Multi`.
        let openings = t.multi_open_states();
        s.timed_nodes(prof, pos, Phase::MultiOpenClose, |jn| {
            s.multi_open_close(t, jn, &openings);
        });
        if pos < dfa.seq_len {
            s.timed_nodes(prof, pos, Phase::MExtend, |jn| s.m_extend(t, jn));
            s.timed_nodes(prof, pos, Phase::CExtend, |jn| s.c_extend(t, jn));
        }
    }
}

/// Immutable scan context (energies + DFA + the selected internal-loop
/// closure); the mutable tables are threaded through each beam call.
struct Scan<'a, const S: usize> {
    params: &'a Params,
    dfa: &'a Dfa,
    mode: InternalLoopMode,
    /// Worker-thread count for the two fat inner phases (`single_close_pull`,
    /// `p_beam`). `1` runs the single-threaded path; `>1` fans the independent
    /// outer-cell work across scoped threads and folds the per-worker partials
    /// back with the min-BRD merge (opt-in speed mode). Every cell's BRD is
    /// byte-identical to single-thread. On the push scan an exact BRD tie goes
    /// to the first arrival, so the tied `(mfe, cai)` split, the back-pointer,
    /// and the design traced back through it may differ run to run (at `λ = 1`
    /// the BRD is the energy alone, so a cell's stored CAI is order-dependent
    /// there); the column fill keeps one tie order at every thread count.
    threads: usize,
    /// Dev-only gate (`DERNA_ONESIDE_VALIDATE`): when set, every skewed
    /// one-sided generic closure asserts its factored BRD equals the full
    /// corner enumeration — the de-risk check for the lever-3 reduction, read
    /// once so the hot-path branch stays predictable.
    validate_oneside: bool,
    /// Fill each `P` column with one owner per target cell ([`pull_fill`])
    /// instead of the push and the inner-dealt pull: on by default in `Lz`
    /// mode (off under `DERNA_PUSH_SCAN`), off by default in `Naive` mode (on
    /// under `DERNA_PULL_FILL`). Read once, so the hot path never touches the
    /// environment.
    pull_fill: bool,
    /// The scan's workers, kept for the whole scan; `None` at one thread and
    /// on the push scan unless `DERNA_PUSH_POOL` is set.
    fan_pool: Option<&'a Pool>,
    /// Dev-only gate (`DERNA_FILL_FAN_ALL`): every column-fill site fans out
    /// above one thread whatever its size, so a test on small lattices still
    /// runs the fanned paths. Read once.
    fan_all: bool,
    /// Dev-only gate (`DERNA_FILL_SERIAL_INDEX`): index each finalized column
    /// serially after the bifurcation instead of beside it, so the overlap
    /// can be timed against its absence in one binary. Read once.
    serial_index: bool,
    /// [`Self::ninio_skew_threshold`], computed once: it is read per inner
    /// pair on the hot paths, and is an integer division.
    ninio_skew: usize,
    /// The column fill's flank-shape table ([`pull_fill::fill_shapes`]).
    fill_shapes: Vec<u8>,
    /// Per 3′ flank, the longest 5′ flank the table admits
    /// ([`pull_fill::fill_n1_hi`]): the fill's gather window ends there.
    fill_n1_hi: Vec<Option<usize>>,
    /// Which closing pairs' `1×n` loops score the same at every corner.
    one_by_n: OneByNRows,
    /// Dev-only cap (`DERNA_FAN_WORKERS`) on the workers a fan-out starts,
    /// below `threads`. At `1` the threaded code path runs on one worker,
    /// which separates what the parallel decomposition costs from what the
    /// workers cost each other. Read once, so the hot path never touches the
    /// environment.
    fan_workers_cap: Option<usize>,
    /// Positions required to carry a base pair. Consulted wherever the scan
    /// declares a range unpaired; [`MustPairConstraint::inert`] leaves every such
    /// site on its unconstrained path.
    must: MustPairConstraint,
    /// Largest total unpaired count a single-branch loop may carry, bounding
    /// every flank scan in this module. See [`ScanConfig::single_max_len`];
    /// validated by the caller, so the arithmetic here may assume it is at
    /// least [`crate::MIN_SINGLE_MAX_LEN`].
    single_max_len: usize,
    /// Whether the single-branch closures are pulled at their outer position
    /// ([`Self::push_pull`]) rather than pushed from their inner one. Both
    /// routes leave every cell the same; the pull is the faster, and the push
    /// is kept so the pull can be checked against it
    /// ([`scan_with_constraints_pushing`]).
    pull_push: bool,
}

/// One `H`-index answer at an outer `(p, q)`: the window's least `H`, the
/// weighted loop-size term its candidate adds, and where its entry is, read
/// only if it wins an outer cell.
struct SkewCand {
    h: f64,
    add: f64,
    /// The entry's slot in the `H` index ([`Scan::h_slot`]).
    delta: i64,
    idx: u32,
}

/// One finalized closed pair, as the three `P` closures consume it: the outer
/// boundary node, the pair's key, and its merged state.
#[derive(Clone, Copy)]
struct PCell {
    o: Node,
    key: PairKey,
    st: State,
}

/// The fixed coordinates of one single-branch closure: the enclosed inner
/// pair and its state, the outer pair's opening/closing edges and boundary
/// nodes, and the two closing pairs. Shared by the stack/bulge and
/// internal-loop closures so they agree on the cell they write.
struct SingleCtx {
    inner: PairRef,
    st: Value,
    ep: Edge,
    pn: Node,
    qn: Node,
    eq: Edge,
    outer_pair: BasePair,
    inner_pair: BasePair,
}

impl SingleCtx {
    /// The outer pair's closed-pair key `(pn, qn, nuc_open, nuc_close)`.
    fn out_key(&self) -> PairKey {
        PairKey {
            open: self.pn,
            close: self.qn,
            opening: self.ep.opening,
            nuc_close: self.eq.nuc,
        }
    }

    /// The traceback manner for a single-branch closure around `inner`.
    fn back(&self) -> Back {
        Back::Single { inner: self.inner }
    }
}

/// The loop-invariant 5′ (left) side of a single-branch closure: the inner
/// pair and its state, the opening edge/node, the 5′-flank length and its
/// best-CAI weight, and (for internal loops) the flank's terminal-mismatch
/// corners. Fixed across the 3′-pair (`q`) scan, so it is computed once and
/// passed by reference into each closure candidate.
struct LeftFlank<'a> {
    /// The closure's context, its 5′ side and inner pair fixed; the 3′ side
    /// (`qn`, `eq`, `outer_pair`) is set for each closure.
    ctx: SingleCtx,
    left_cai: f64,
    left_corners: Option<&'a [Corner]>,
    n1: usize,
    o: Node,
}

/// One enclosed inner pair as the generic pull sees it: its `P`-cell
/// coordinates (table node `io`, key) and merged state. The inner 5′ base is
/// at `ikey.open.pos`, the 3′ base at `io.pos - 1`.
struct InnerCell {
    io: Node,
    ikey: PairKey,
    st: State,
}

/// One outer cell the skewed-loop reduction writes into: its 5′ side
/// (open node `pn`, opening edge `ep`) and 3′ side (close node `qn`, closing
/// edge `eq`), with the canonical outer pair. The written cell lives at
/// `eq.to` keyed by `(pn, qn, ep.nuc, eq.nuc)`.
struct OuterCell {
    pn: Node,
    ep: Edge,
    qn: Node,
    eq: Edge,
    outer_pair: BasePair,
}

/// One multi-loop opening's invariants: the wrapped `M2` core span and its
/// state, plus the 5′ flank (opening node `pn`, opening edge `ep`, and the
/// flank's best-CAI weight). Shared by the short-flank direct closures and
/// the cross-codon `multi_open` entry that one opening produces.
struct MultiSeed {
    core: SpanRef,
    st: State,
    pn: Node,
    ep: Edge,
    left_cai: f64,
}

/// What a multi-loop opening reads off its core's 3′ end `core_outer` — the
/// same for every core ending there and every opening around it, so it is
/// derived once per end: the frontiers inside `core_outer`'s codon that close
/// directly, each with its broken-codon CAI from `core_outer`; the codon
/// boundary; and `C(core_outer)` when the shared `multi_open` path is
/// admissible (its stretch `[core_outer.pos, boundary)` is free).
struct CoreEnd {
    outer: Node,
    closes: Vec<(Node, f64)>,
    boundary: u32,
    c_outer: Option<f64>,
}

/// A multi-loop opening with a long left flank, carried leftward by
/// [`Scan::far_multi_open`]: the best `M2` core for it and that core's state,
/// the opening edge, and the opening's adaptation (core plus flank, used only
/// to choose between cores).
#[derive(Clone, Copy)]
struct FarOpening {
    core: SpanRef,
    st: State,
    ep: Edge,
    cai: f64,
}

impl FarOpening {
    /// Keep this opening at `(pn, ep.nuc)` unless one already there scores better.
    fn merge_into(self, far: &mut FxHashMap<MultiOpenKey, FarOpening>, pn: Node, lambda: f64) {
        let key = MultiOpenKey {
            open: pn,
            nuc_open: self.ep.nuc,
            lane_open: self.ep.to.lane,
            reach: 0,
        };
        let score = |f: &FarOpening| State::new(f.st.mfe, f.cai, f.st.back).brd(lambda);
        match far.get(&key) {
            Some(old) if score(old) <= score(&self) => {}
            _ => {
                far.insert(key, self);
            }
        }
    }
}

/// Table writes buffered from one (possibly parallel) phase, folded into the
/// shared tables afterwards. Collecting then folding lets the fat inner phases
/// run their independent outer-cell work on worker-local buffers with no shared
/// mutation in the hot path; the min-BRD [`Tables`] merges make the folded
/// result independent of collection order for MFE + CAI (only equal-BRD tie
/// back-pointers depend on order). Applying per-table in collection order
/// reproduces the single-threaded sequence exactly.
#[derive(Default)]
struct Writes {
    p: Vec<(Node, PairKey, State)>,
    m: Vec<(Node, Node, State)>,
    m2: Vec<(Node, Node, State)>,
    c: Vec<(Node, State)>,
}

impl Writes {
    /// Fold the buffered candidates into `t` with the standard min-BRD merges,
    /// each table in collection order.
    fn apply(&self, t: &mut Tables) {
        for &(outer, key, cand) in &self.p {
            t.update_p(outer, key, cand);
        }
        for &(outer, inner, cand) in &self.m {
            t.update_m(outer, inner, cand);
        }
        for &(outer, inner, cand) in &self.m2 {
            t.update_m2(outer, inner, cand);
        }
        for &(node, cand) in &self.c {
            t.update_c(node, cand);
        }
    }

    /// Candidates buffered, across every table.
    fn len(&self) -> usize {
        self.p.len() + self.m.len() + self.m2.len() + self.c.len()
    }

    /// Empty the buffer for reuse, keeping its capacity.
    fn clear(&mut self) {
        self.p.clear();
        self.m.clear();
        self.m2.clear();
        self.c.clear();
    }
}

/// One fan-out worker's own clock, read only while the profile records: its
/// whole loop, its longest single chunk, and the time in chunks of the site's
/// heavy family (the pull's direct closures; nothing at the push).
#[derive(Clone, Copy, Default)]
struct WorkerClock {
    started: Option<Instant>,
    busy: Duration,
    chunk_max: Duration,
    heavy: Duration,
}

impl WorkerClock {
    fn start(timing: bool) -> Self {
        Self {
            started: timing.then(Instant::now),
            ..Self::default()
        }
    }

    /// Charge the chunk begun at `began`; a no-op when not timing.
    fn chunk(&mut self, began: Option<Instant>, heavy: bool) {
        let Some(began) = began else {
            return;
        };
        let spent = began.elapsed();
        self.chunk_max = self.chunk_max.max(spent);
        if heavy {
            self.heavy += spent;
        }
    }

    fn stop(mut self) -> Self {
        if let Some(started) = self.started {
            self.busy = started.elapsed();
        }
        self
    }
}

/// Record one fanned-out position's workers — each one's busy time, the
/// longest chunk, the heavy family's time and the candidates they buffered —
/// into `prof`.
fn record_fan(prof: &mut Profile, site: FanSite, buffers: &[Writes], clocks: &[WorkerClock]) {
    if !prof.is_on() {
        return;
    }
    let busy: Vec<Duration> = clocks.iter().map(|c| c.busy).collect();
    let chunk_max = clocks.iter().map(|c| c.chunk_max).max().unwrap_or_default();
    let heavy = clocks.iter().map(|c| c.heavy).sum();
    let candidates = buffers.iter().map(Writes::len).sum();
    prof.record_fan(site, &busy, chunk_max, heavy, candidates);
}

/// Independent tasks the participants of a fanned-out site take before its
/// units, overlapping the units of those that arrive later. Each task runs
/// once, on whichever participant takes it first; the pool's caller always
/// takes part, so every task has run when the first fanned site returns, and
/// the driver runs any left if no site fanned out.
struct Lead<'a> {
    tasks: Vec<Mutex<Option<LeadTask<'a>>>>,
    next: AtomicUsize,
}

type LeadTask<'a> = Box<dyn FnOnce() + Send + 'a>;

impl<'a> Lead<'a> {
    fn new(tasks: Vec<LeadTask<'a>>) -> Self {
        Lead {
            tasks: tasks.into_iter().map(|t| Mutex::new(Some(t))).collect(),
            next: AtomicUsize::new(0),
        }
    }
}

/// Take and run `lead`'s tasks until none is left.
fn run_lead(lead: &Lead) {
    loop {
        let i = lead.next.fetch_add(1, Ordering::Relaxed);
        let Some(slot) = lead.tasks.get(i) else {
            return;
        };
        let job = slot.lock().unwrap().take();
        if let Some(job) = job {
            job();
        }
    }
}

thread_local! {
    /// Reused 5′-flank corner scratch for the single-branch closure, held
    /// thread-locally so the multi-threaded scan gives each worker its own
    /// buffer with no contention and no `RefCell` on the shared [`Scan`] (which
    /// must be `Sync` to cross scoped threads). Serial callers use the main
    /// thread's copy — the exact single-thread behaviour.
    static LEFTS: RefCell<Vec<Corner>> = const { RefCell::new(Vec::new()) };
    /// Reused 3′-flank corner scratch (see [`LEFTS`]).
    static RIGHTS: RefCell<Vec<Corner>> = const { RefCell::new(Vec::new()) };
    /// Reused `H`-index answers for one outer `(p, q)` (see
    /// [`Scan::skewed_pull_at`]).
    static SKEWED: RefCell<Vec<SkewCand>> = const { RefCell::new(Vec::new()) };
    /// The 3′ corners every closure of one pull row shares (see
    /// [`Scan::pull_close_side_at`]).
    static ROW_RIGHTS: RefCell<Vec<Corner>> = const { RefCell::new(Vec::new()) };
    /// The push's 3′ corners by closing node, for one inner pair's outer node
    /// (see [`CornerCache`]).
    static PUSH_RIGHTS: RefCell<CornerCache> = RefCell::new(CornerCache::default());
    /// The push's 5′ corners by opening edge target, into one inner pair's
    /// opening node (see [`CornerCache`]).
    static PUSH_LEFTS: RefCell<CornerCache> = RefCell::new(CornerCache::default());
}

/// A pull row's outer 3′ side: the closing node and edge, and the 3′ corners
/// from the row's inner node to it.
#[derive(Clone, Copy)]
struct RowSide<'a> {
    qn: Node,
    eq: Edge,
    rights: &'a [Corner],
}

/// One cell the pull won, as [`Scan::push_pull`] carries it out of its
/// workers: the `P`-write the push's fold would have kept, and the arrival key
/// of the *first* candidate offered to that cell — where the push would have
/// inserted it, had the column not already held it.
type PullWin = ((Node, PairKey, State), (u32, u32));

/// The candidate [`Scan::pulled_single`] has kept so far: its merge score and
/// where the push would have sent it (enclosed position, rank within it).
struct PulledBest {
    score: f64,
    arrival: (u32, u32),
    /// The least arrival key over every candidate offered, whether it won or
    /// not: the moment the push's fold would have inserted this cell, had the
    /// column not already held it.
    first: (u32, u32),
    cand: Option<(Node, PairKey, State)>,
}

impl Default for PulledBest {
    fn default() -> Self {
        PulledBest {
            score: 0.0,
            arrival: (0, 0),
            first: (u32::MAX, u32::MAX),
            cand: None,
        }
    }
}

/// Strand corners with one end fixed, enumerated on first use for each other
/// end: the 3′-flank corners from an inner pair's outer node `o` to each
/// closing node, or the 5′-flank corners from each opening edge's target into
/// the inner pair's opening node. Every push closure around one inner pair
/// that shares a strand reads the same corners, whatever its other side.
/// Cleared for each inner pair ([`Scan::p_to_single`]), so it never outlives
/// the lattice it was filled from.
#[derive(Default)]
struct CornerCache {
    anchor: Option<Node>,
    /// By `(distance from the anchor, other end's lane)`; a slot holds its
    /// strand's corners when its node is that other end.
    slots: Vec<(Option<Node>, Vec<Corner>)>,
}

impl CornerCache {
    fn clear(&mut self) {
        self.anchor = None;
    }

    /// The corners `o → qn`, anchored at `o`.
    fn rights(&mut self, dfa: &Dfa, o: Node, qn: Node) -> &[Corner] {
        self.get(o, qn, (qn.pos - o.pos, dfa.stride()), |out| {
            corner_scores_into(dfa, o, qn, out)
        })
    }

    /// The corners `from → to`, anchored at `to`.
    fn lefts(&mut self, dfa: &Dfa, from: Node, to: Node) -> &[Corner] {
        self.get(to, from, (to.pos - from.pos, dfa.stride()), |out| {
            corner_scores_into(dfa, from, to, out);
        })
    }

    fn get(
        &mut self,
        anchor: Node,
        end: Node,
        (distance, stride): (u32, usize),
        fill: impl FnOnce(&mut Vec<Corner>),
    ) -> &[Corner] {
        if self.anchor != Some(anchor) {
            self.slots.iter_mut().for_each(|slot| slot.0 = None);
            self.anchor = Some(anchor);
        }
        let i = distance as usize * stride + usize::from(end.lane);
        if self.slots.len() <= i {
            self.slots.resize_with(i + 1, || (None, Vec::new()));
        }
        let slot = &mut self.slots[i];
        if slot.0 != Some(end) {
            fill(&mut slot.1);
            slot.0 = Some(end);
        }
        &slot.1
    }
}

impl<const S: usize> Scan<'_, S> {
    /// The stops a node still reaches ([`Dfa::reach_class`]); every one of
    /// them on a lattice whose positions hold no stop lane, where the test
    /// compiles away.
    #[inline]
    fn reach_of(&self, node: Node) -> u8 {
        if S == 2 {
            0
        } else {
            self.dfa.reach_class(node)
        }
    }

    /// Workers for a fan-out with `chunks` chunks of work: one per chunk up to
    /// `threads`, and never above the dev-only cap.
    fn fan_workers(&self, chunks: usize) -> usize {
        let workers = self.threads.min(chunks);
        self.fan_workers_cap
            .map_or(workers, |cap| workers.min(cap))
            .max(1)
    }

    /// Run `body` over every DFA node at `pos`, charging the whole sweep to
    /// `phase`.
    ///
    /// One clock reading per position per phase rather than per node: the
    /// per-position sweep is the unit the driver already works in, and timing
    /// finer would cost more readings than the phases it separates are worth.
    fn timed_nodes<F: FnMut(Node)>(&self, prof: &mut Profile, pos: u32, phase: Phase, mut body: F) {
        let m = prof.mark();
        for &jn in self.dfa.nodes_at(pos) {
            body(jn);
        }
        prof.record(phase, m);
    }

    /// Seed every hairpin-closed `P` cell. A hairpin spans an opening
    /// edge at slot `a`, the loop interior, and a closing edge at slot
    /// `b` with `b - a - 1 >= 3`; its energy is exact on the
    /// reconstructed interior.
    fn seed_hairpins(&self, t: &mut Tables) {
        if self.fan_pool.is_some() {
            self.seed_hairpins_fanned(t);
            return;
        }
        let lambda = t.lambda();
        for a in 0..self.dfa.seq_len {
            for &an in self.dfa.nodes_at(a) {
                for eo in self.dfa.right(an) {
                    self.seed_hairpins_from(lambda, an, *eo, &mut |o, key, st| {
                        t.update_p(o, key, st);
                    });
                }
            }
        }
    }

    /// [`Self::seed_hairpins`] on the scan's pool: each opening position's
    /// hairpins are scored by one participant, a batch of positions at a
    /// time, and inserted in the serial order, so a tie keeps the same first
    /// arrival.
    fn seed_hairpins_fanned(&self, t: &mut Tables) {
        const BATCH: u32 = 64;
        let lambda = t.lambda();
        let n = self.dfa.seq_len;
        let mut from = 0;
        while from < n {
            let to = (from + BATCH).min(n);
            let next = AtomicU32::new(from);
            let mut parts: Vec<(u32, CellBest)> = self
                .fan_collect(self.threads, |_| {
                    let mut mine = Vec::new();
                    loop {
                        let a = next.fetch_add(1, Ordering::Relaxed);
                        if a >= to {
                            break;
                        }
                        let mut out = CellBest::new();
                        for &an in self.dfa.nodes_at(a) {
                            for eo in self.dfa.right(an) {
                                self.seed_hairpins_from(lambda, an, *eo, &mut |o, key, st| {
                                    out.push((o, key, st));
                                });
                            }
                        }
                        mine.push((a, out));
                    }
                    mine
                })
                .into_iter()
                .flatten()
                .collect();
            parts.sort_unstable_by_key(|&(a, _)| a);
            for (_, out) in parts {
                for (o, key, st) in out {
                    t.update_p(o, key, st);
                }
            }
            from = to;
        }
    }

    fn seed_hairpins_from(
        &self,
        lambda: f64,
        an: Node,
        eo: Edge,
        emit: &mut impl FnMut(Node, PairKey, State),
    ) {
        // The whole interior `[an.pos + 1, b)` is unpaired, so a required
        // position inside it caps how far the closing base may sit — the
        // hairpin's `b`-scan is truncated rather than filtered.
        let b_hi = self
            .dfa
            .seq_len
            .min(self.must.max_right(an.pos + 1).saturating_add(1));
        // Jump over closing positions with no partner for `eo.nuc` rather
        // than visiting and discarding them (next-pair sparsity).
        let mut b = self.dfa.next_partner(eo.nuc, an.pos + MIN_HAIRPIN_LOOP + 1);
        while b < b_hi {
            for &bn in self.dfa.nodes_at(b) {
                for ec in self.dfa.right(bn) {
                    self.seed_hairpin_close(lambda, an, eo, bn, *ec, emit);
                }
            }
            b = self.dfa.next_partner(eo.nuc, b + 1);
        }
    }

    /// Seed the hairpin-closed `P` cell for one opening edge `eo` (out of
    /// `an`) and one closing edge `ec` (out of `bn`), when they pair.
    fn seed_hairpin_close(
        &self,
        lambda: f64,
        an: Node,
        eo: Edge,
        bn: Node,
        ec: Edge,
        emit: &mut impl FnMut(Node, PairKey, State),
    ) {
        let Some(closing) = canonical(eo.nuc, ec.nuc) else {
            return;
        };
        let Some((energy, interior_cai)) =
            score_hairpin(self.params, self.dfa, lambda, eo.to, bn, closing)
        else {
            return;
        };
        let cai = eo.cai + interior_cai + ec.cai;
        let key = PairKey {
            open: an,
            close: bn,
            opening: eo.opening,
            nuc_close: ec.nuc,
        };
        emit(ec.to, key, State::new(energy, cai, Back::Hairpin));
    }

    /// Close a multi-loop: `Multi[o][key]` becomes `P[o][key]`, paying
    /// the loop-closing cost on the closing pair.
    fn multi_close(&self, t: &mut Tables, o: Node) {
        for (key, st) in t.multi_states(o) {
            let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
            let mfe = st.mfe + self.multi_close_cost(closing);
            let back = Back::MultiClose {
                multi: PairRef { outer: o, key },
            };
            t.update_p(o, key, State::new(mfe, st.cai, back));
        }
    }

    /// Process every closed pair at `pos`: extend the exterior (`C = C + P`),
    /// open a multi-branch (`M = P`), bifurcate (`M2 = M + P`), and grow stacks
    /// / bulges / internal loops into larger `P` cells. Two passes so the push
    /// (`p_to_single`) — which reads no `P`/`M`/`M2`/`C` cell, only the closed
    /// pair and the DFA/energies — can fan out over the independent closed pairs:
    /// pass 1 runs the exterior + multibranch serially (their `M2` merge cannot
    /// buffer the `O(n³)` bifurcation candidates), pass 2 the push in parallel.
    fn p_beam_all(
        &self,
        t: &mut Tables,
        pos: u32,
        prof: &mut Profile,
        pool: &mut [Writes],
        lead: Option<&Lead>,
    ) {
        // Under the column fill every single-branch closure is pulled, so only
        // the exterior and multibranch closures run here.
        if self.pull_fill {
            let m = prof.mark();
            for &jn in self.dfa.nodes_at(pos) {
                for (key, st) in t.p_states(jn) {
                    self.p_to_exterior(t, jn, key, st);
                    let stem =
                        self.m_stem_cost(pair_from_nucleotides(key.nuc_open(), key.nuc_close));
                    let back = Back::MFromP {
                        pair: PairRef { outer: jn, key },
                    };
                    t.update_m(jn, key.open, State::new(st.mfe + stem, st.cai, back));
                }
            }
            prof.record(Phase::FillExterior, m);
            self.bifurcate_fill_column(t, pos, prof, lead);
            return;
        }
        // Single-thread: one snapshot per node, all three closures per closed
        // pair (push buffered, folded once). Byte-identical to the interleaved
        // pre-parallel path with no extra per-position copies.
        if self.threads <= 1 {
            let w = &mut pool[0];
            for &jn in self.dfa.nodes_at(pos) {
                for (key, st) in t.p_states(jn) {
                    self.p_closures(t, w, PCell { o: jn, key, st }, prof);
                }
            }
            // The buffer holds this column's `p_to_single` candidates and
            // nothing else, so its fold belongs to that phase.
            let m = prof.mark();
            w.apply(t);
            w.clear();
            prof.record(Phase::PToSingle, m);
            return;
        }
        // Multi-thread pass 1 (serial): C, M, M2. Reads finalized cells (`< pos`);
        // writes the `pos` columns not read again this phase.
        let m = prof.mark();
        for &jn in self.dfa.nodes_at(pos) {
            for (key, st) in t.p_states(jn) {
                self.p_to_exterior(t, jn, key, st);
                self.p_to_multibranch(t, jn, key, st);
            }
        }
        // Pass 2: push specials → `P` at later columns (never read this phase),
        // unless they are pulled at their own positions instead.
        if self.pulls_push() {
            prof.record(Phase::FanPushSerial, m);
            return;
        }
        let cells: Vec<(Node, PairKey, State)> = self
            .dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&jn| t.p_states(jn).into_iter().map(move |(k, s)| (jn, k, s)))
            .collect();
        prof.record(Phase::FanPushSerial, m);
        if cells.is_empty() {
            return;
        }
        if cells.len() < PUSH_PARALLEL_MIN {
            let m = prof.mark();
            let w = &mut pool[0];
            for &(o, key, st) in &cells {
                self.p_to_single_dispatch(t, w, o, key, st);
            }
            w.apply(t);
            w.clear();
            prof.record(Phase::FanPushSmall, m);
            return;
        }
        let counter = AtomicUsize::new(0);
        let shared: &Tables = t;
        let cells_ref = &cells;
        let n_workers = self.fan_workers(cells.len().div_ceil(PUSH_CHUNK));
        let timing = prof.is_on();
        let m = prof.mark();
        let buffers: Vec<Mutex<&mut Writes>> =
            pool[..n_workers].iter_mut().map(Mutex::new).collect();
        let clocks: Vec<WorkerClock> = self.fan_collect(n_workers, |me| {
            let mut w = buffers[me].lock().unwrap();
            let mut clock = WorkerClock::start(timing);
            loop {
                let start = counter.fetch_add(PUSH_CHUNK, Ordering::Relaxed);
                if start >= cells_ref.len() {
                    break;
                }
                let chunk = timing.then(Instant::now);
                let end = (start + PUSH_CHUNK).min(cells_ref.len());
                for &(o, key, st) in &cells_ref[start..end] {
                    self.p_to_single_dispatch(shared, &mut w, o, key, st);
                }
                clock.chunk(chunk, false);
            }
            clock.stop()
        });
        drop(buffers);
        let buffers = &mut pool[..n_workers];
        prof.record(Phase::FanPushCompute, m);
        record_fan(prof, FanSite::Push, buffers, &clocks);
        let m = prof.mark();
        for w in buffers.iter_mut() {
            w.apply(t);
            w.clear();
        }
        prof.record(Phase::FanPushApply, m);
    }

    /// Run all three closures of one finalized closed pair, in the order the
    /// scan defines: exterior, then multibranch, then the single-branch push.
    ///
    /// When the profile is recording, the three are separated by clock readings
    /// taken between them — four per closed pair, and the only per-cell timing
    /// in the scan. Timing them at this granularity rather than per position is
    /// what lets an attribution say which of the three carries the column's
    /// cost; the readings are the instrument's dominant overhead, which is why
    /// the whole-scan elapsed time is recorded beside them.
    fn p_closures(&self, t: &mut Tables, w: &mut Writes, cell: PCell, prof: &mut Profile) {
        let PCell { o, key, st } = cell;
        if !prof.is_on() {
            self.p_to_exterior(t, o, key, st);
            self.p_to_multibranch(t, o, key, st);
            self.p_to_single_dispatch(t, w, o, key, st);
            return;
        }
        let t0 = Instant::now();
        self.p_to_exterior(t, o, key, st);
        let t1 = Instant::now();
        self.p_to_multibranch(t, o, key, st);
        let t2 = Instant::now();
        self.p_to_single_dispatch(t, w, o, key, st);
        let t3 = Instant::now();
        prof.add(Phase::PToExterior, t1 - t0);
        prof.add(Phase::PToMultibranch, t2 - t1);
        prof.add(Phase::PToSingle, t3 - t2);
    }

    /// Dispatch the push to the active internal-loop mode.
    fn p_to_single_dispatch(&self, t: &Tables, w: &mut Writes, o: Node, key: PairKey, st: State) {
        if self.pulls_push() {
            return;
        }
        match self.mode {
            InternalLoopMode::Naive => self.p_to_single(t, w, o, key, st),
            InternalLoopMode::Lz => self.p_to_single_lz(t, w, o, key, st),
        }
    }

    /// The push half of the Lyngsø–Zuker `O(n²·L)` reformulation
    /// ([`InternalLoopMode::Lz`]). It runs the same push path as
    /// [`Self::p_to_single`], but [`Self::single_close`] skips the **generic**
    /// internal-loop branch (both flanks `≥ 2`, not a tabulated small shape)
    /// in `Lz` mode — those cells are filled by [`Self::single_close_pull`].
    /// The stack/bulge, `int11/21/22`, `1×n`/`n×1`, and `2×3`/`3×2` special
    /// cases stay here: they are already `O(L)` and byte-exact.
    fn p_to_single_lz(&self, t: &Tables, w: &mut Writes, o: Node, key: PairKey, st: State) {
        self.p_to_single(t, w, o, key, st);
    }

    /// `C = C + P` and `M = P`, plus the `M2 = M + P` bifurcation.
    fn p_to_multibranch(&self, t: &mut Tables, o: Node, key: PairKey, st: State) {
        let inner = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let stem = self.m_stem_cost(inner);
        let back_m = Back::MFromP {
            pair: PairRef { outer: o, key },
        };
        t.update_m(o, key.open, State::new(st.mfe + stem, st.cai, back_m));
        t.bifurcate_m_into_m2(key.open, o, st.mfe + stem, st.cai, |m_inner| {
            Back::M2Bifurcation {
                left: SpanRef {
                    outer: key.open,
                    inner: m_inner,
                },
                right: PairRef { outer: o, key },
            }
        });
    }

    /// `C = C + P`: attach a closed pair to the exterior prefix.
    fn p_to_exterior(&self, t: &mut Tables, o: Node, key: PairKey, st: State) {
        let Some(prefix) = t.c[self.dfa.ix(key.open)] else {
            return;
        };
        let inner = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let mfe = prefix.mfe + st.mfe + self.terminal_au(inner);
        let back = Back::ExteriorPair {
            prefix: key.open,
            pair: PairRef { outer: o, key },
        };
        t.update_c(o, State::new(mfe, prefix.cai + st.cai, back));
    }

    /// Grow `P[o]` (inner pair `a..b`) into every outer pair `(p, q)`
    /// whose unpaired flanks fit within `SINGLE_MAX_LEN` — the stack,
    /// bulge, and internal-loop closures.
    fn p_to_single(&self, t: &Tables, w: &mut Writes, o: Node, key: PairKey, st: State) {
        let a = key.open.pos;
        // n1 = a − p − 1 ≤ L, so p ≥ a − L − 1. The 5′ flank `[p + 1, a)` is
        // unpaired, so a required position inside it floors `p`: the flank scan
        // is truncated, not filtered.
        let p_lo = a
            .saturating_sub(self.single_max_len as u32 + 1)
            .max(self.must.min_left(a));
        let inner = PairRef { outer: o, key };
        // The cache lives on the thread, across scans and lattices: start it
        // empty for each inner pair.
        PUSH_RIGHTS.with_borrow_mut(CornerCache::clear);
        PUSH_LEFTS.with_borrow_mut(CornerCache::clear);
        for p in p_lo..a {
            for &pn in self.dfa.nodes_at(p) {
                for ep in self.dfa.right(pn) {
                    self.single_from_left(t, w, inner, st, pn, *ep);
                }
            }
        }
    }

    fn single_from_left(
        &self,
        t: &Tables,
        w: &mut Writes,
        inner: PairRef,
        st: State,
        pn: Node,
        ep: Edge,
    ) {
        let (o, key) = (inner.outer, inner.key);
        let inner_pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let Some(left_cai) = self.dfa.broken_codon_cai_s::<S>(ep.to, key.open) else {
            return;
        };
        // 5′-flank length and (for internal loops) its terminal-mismatch
        // corners, both fixed across the 3′-pair scan below.
        let n1 = (key.open.pos - ep.to.pos) as usize;
        PUSH_LEFTS.with_borrow_mut(|cache| {
            let left_corners = (n1 > 0).then(|| cache.lefts(self.dfa, ep.to, key.open));
            let b = o.pos - 1;
            // In `Lz` mode the generic closures are pulled, so the push scan only
            // needs the **special** 3′ flanks. For `n1 >= 2` the only non-generic
            // shapes are `n2 <= 3` (`(2,2)`/`(2,3)`/`(3,2)` plus the `n2 <= 1`
            // bulge/`int`/`1×n` cases); every longer 3′ flank is generic. Capping
            // the `q`-scan there turns the push from `O(L²)` to `O(L)` per inner
            // pair. `n1 <= 1` keeps the full scan (all its shapes are special).
            // Either way n2 = q − b − 1 ≤ L − n1, so q < b + 2 + (L − n1): the
            // short-flank window is capped too, since `n1` reaches `L`.
            let q_cap = b + 2 + (self.single_max_len - n1) as u32;
            let q_hi = if self.pull_generic() && n1 >= 2 {
                (b + 5).min(q_cap).min(self.dfa.seq_len)
            } else {
                q_cap.min(self.dfa.seq_len)
            };
            // The 3′ flank `[o.pos, q)` is unpaired, so a required position inside
            // it caps `q` — the same truncation the 5′ side applies to `p`.
            let q_hi = q_hi.min(self.must.max_right(o.pos).saturating_add(1));
            // Jump over 3′-pair positions with no partner for `ep.nuc`
            // (next-pair sparsity); the flank-length cap keeps the scan bounded.
            let mut flank = LeftFlank {
                ctx: SingleCtx {
                    inner,
                    st: st.into(),
                    ep,
                    pn,
                    qn: o,
                    eq: ep,
                    outer_pair: inner_pair,
                    inner_pair,
                },
                left_cai,
                left_corners,
                n1,
                o,
            };
            let mut q = self.dfa.next_partner(ep.nuc, b + 1);
            while q < q_hi {
                for &qn in self.dfa.nodes_at(q) {
                    for eq in self.dfa.right(qn) {
                        self.single_close(t, w, &mut flank, qn, *eq);
                    }
                }
                q = self.dfa.next_partner(ep.nuc, q + 1);
            }
        });
    }

    /// Close one single-branch candidate: outer pair `(pn, qn)` around the
    /// inner pair, dispatching to the internal-loop or stack/bulge scorer.
    fn single_close(
        &self,
        t: &Tables,
        w: &mut Writes,
        flank: &mut LeftFlank<'_>,
        qn: Node,
        eq: Edge,
    ) {
        let Some(outer_pair) = canonical(flank.ctx.ep.nuc, eq.nuc) else {
            return;
        };
        let n2 = (qn.pos - flank.o.pos) as usize;
        let lefts = flank.left_corners.filter(|_| n2 > 0);
        // Both flanks non-empty: a true internal loop, whose energy reads the
        // four terminal mismatches — chosen jointly with their CAI rather than
        // reconstructed. In `Lz` mode the generic shape (both flanks `≥ 2`,
        // not a tabulated small case) is closed by the `O(L)` pull pass
        // instead; the special internal-loop shapes stay on this push path.
        if lefts.is_some() && self.pull_generic() && is_generic(flank.n1, n2) {
            return;
        }
        flank.ctx.qn = qn;
        flank.ctx.eq = eq;
        flank.ctx.outer_pair = outer_pair;
        let ctx = &flank.ctx;
        match lefts {
            Some(lefts) => {
                PUSH_RIGHTS.with_borrow_mut(|cache| {
                    let rights = cache.rights(self.dfa, flank.o, qn);
                    if let Some(cand) =
                        self.internal_loop_cand_with(t.lambda(), ctx, lefts, rights, flank.n1, n2)
                    {
                        w.p.push(cand);
                    }
                });
            }
            // Stack or bulge: energy is independent of the interior
            // nucleotides, so the best-CAI flank is jointly optimal.
            None => self.stack_or_bulge(w, ctx, flank.left_cai, flank.n1, n2),
        }
    }

    /// Whether the generic internal-loop closure is pulled (computed in
    /// [`Self::single_close_pull`]) rather than pushed — true exactly in
    /// [`InternalLoopMode::Lz`].
    fn pull_generic(&self) -> bool {
        matches!(self.mode, InternalLoopMode::Lz)
    }

    /// Close a stack or bulge: the interior nucleotides do not reach the
    /// energy model (only the flank sizes do), so the best-CAI 5′/3′
    /// flanks are jointly optimal.
    fn stack_or_bulge(&self, w: &mut Writes, ctx: &SingleCtx, left_cai: f64, n1: usize, n2: usize) {
        if let Some(cand) = self.stack_or_bulge_cand(ctx, left_cai, n1, n2) {
            w.p.push(cand);
        }
    }

    /// The `P`-write candidate of a stack or bulge closure, or `None` when its
    /// 3′ flank has no path.
    fn stack_or_bulge_cand(
        &self,
        ctx: &SingleCtx,
        left_cai: f64,
        n1: usize,
        n2: usize,
    ) -> Option<(Node, PairKey, State)> {
        let right_cai = self.dfa.broken_codon_cai_s::<S>(ctx.inner.outer, ctx.qn)?;
        let energy = stack_or_bulge_energy(self.params, ctx.outer_pair, ctx.inner_pair, n1, n2);
        let cai = ctx.st.cai + ctx.ep.cai + left_cai + right_cai + ctx.eq.cai;
        Some((
            ctx.eq.to,
            ctx.out_key(),
            State::new(ctx.st.mfe + energy, cai, ctx.back()),
        ))
    }

    /// Whether the single-branch push runs as [`Self::push_pull`]: on the
    /// push scan in `Lz` mode (the column fill produces those closures
    /// itself), unless the scan asked to push ([`scan_with_constraints_pushing`]).
    /// Without required positions every closed pair the push writes to
    /// already holds its hairpin, so the pull only improves cells that exist;
    /// with them a blocked hairpin can leave a target cell absent, and the
    /// pull creates it (see [`Self::pull_targets`]).
    fn pulls_push(&self) -> bool {
        !self.pull_fill && self.pull_generic() && self.pull_push
    }

    /// The single-branch push ([`Self::p_to_single`]) as a pull into the
    /// closed pairs at `pos`, merged before anything else writes them there.
    ///
    /// The push sends each closed pair one candidate from every enclosed pair
    /// it closes around, folded position by position, each cell keeping the
    /// first candidate to reach its least value. Nothing reads a cell before
    /// its own position, so merging only that candidate — found here over the
    /// same enclosed pairs, scored by the same functions, ties broken by the
    /// push's order (enclosed position, then rank within it) — leaves every
    /// cell as the push leaves it.
    fn push_pull(&self, t: &mut Tables, idx: (&OneSideIndex, &OneSideKIndex), pos: u32) {
        let cells = self.pull_targets(t, pos);
        let shared: &Tables = t;
        let pull = |chunk: &[(Node, PairKey)]| -> Vec<PullWin> {
            // The caches live on the thread, across scans and lattices.
            PUSH_LEFTS.with_borrow_mut(CornerCache::clear);
            PUSH_RIGHTS.with_borrow_mut(CornerCache::clear);
            chunk
                .iter()
                .filter_map(|&(jn, key)| {
                    let best = self.pulled_single(shared, idx, jn, key);
                    Some((best.cand?, best.first))
                })
                .collect()
        };
        // Cells are independent, so workers take contiguous chunks and the
        // merges run afterwards in cell order, whatever the worker count.
        let workers = self.fan_workers(cells.len().div_ceil(PUSH_PARALLEL_MIN));
        let wins: Vec<PullWin> = if workers <= 1 {
            pull(&cells)
        } else {
            let per = cells.len().div_ceil(workers);
            std::thread::scope(|scope| {
                let handles: Vec<_> = cells
                    .chunks(per)
                    .map(|chunk| crate::task_clock::spawn(scope, move || pull(chunk)))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("a pull worker panicked"))
                    .collect()
            })
        };
        // Every cell the push can write already holds its hairpin unless a
        // required position blocked it, so without them no win inserts and
        // they all merge in cell order — no cell needs looking up first.
        if !self.must.is_active() {
            for ((jn, key, cand), _) in wins {
                t.update_p(jn, key, cand);
            }
            return;
        }
        // A win on a cell the column already holds only changes its value, so
        // those merge in cell order. A win on an absent cell inserts it, and
        // the column's iteration order depends on the order of its inserts —
        // so those go in the order the push would have created them: by the
        // position and rank of the *first* candidate to reach the cell, which
        // is the moment the push's fold inserted it.
        let (mut creates, updates): (Vec<_>, Vec<_>) = wins
            .into_iter()
            .partition(|&((jn, key, _), _)| t.p_at(jn, key).is_none());
        for ((jn, key, cand), _) in updates {
            t.update_p(jn, key, cand);
        }
        creates.sort_by_key(|&(_, first)| first);
        for ((jn, key, cand), _) in creates {
            t.update_p(jn, key, cand);
        }
    }

    /// Every closed-pair cell the push could write at `pos`: the column's own
    /// cells, and — under required positions, where a hairpin can be blocked
    /// while the pair it closes around is allowed — every canonical pair
    /// closing there whether the column holds it or not. Unconstrained, a
    /// hairpin is seeded for every such pair before the scan, so the column's
    /// cells are already all of them and the enumeration is skipped.
    fn pull_targets(&self, t: &Tables, pos: u32) -> Vec<(Node, PairKey)> {
        let held = |jn: Node| t.p_states(jn).into_iter().map(move |(key, _)| (jn, key));
        if !self.must.is_active() {
            return self
                .dfa
                .nodes_at(pos)
                .iter()
                .flat_map(|&jn| held(jn))
                .collect();
        }
        let mut targets: Vec<(Node, PairKey)> = Vec::new();
        for &jn in self.dfa.nodes_at(pos) {
            targets.extend(held(jn));
        }
        let mut seen: FxHashSet<(Node, PairKey)> = targets.iter().copied().collect();
        for &qn in self.dfa.nodes_at(pos - 1) {
            for eq in self.dfa.right(qn) {
                if eq.to.pos != pos {
                    continue;
                }
                // Every opening a canonical pair could sit at: the loop cap
                // bounds the two flanks, not the enclosed pair's own span, so
                // the enclosing pair's opening is not bounded from below.
                for p in 0..qn.pos.saturating_sub(MIN_HAIRPIN_LOOP) {
                    for &pn in self.dfa.nodes_at(p) {
                        for ep in self.dfa.right(pn) {
                            if canonical(ep.nuc, eq.nuc).is_none() {
                                continue;
                            }
                            let key = PairKey {
                                open: pn,
                                close: qn,
                                opening: ep.opening,
                                nuc_close: eq.nuc,
                            };
                            if seen.insert((eq.to, key)) {
                                targets.push((eq.to, key));
                            }
                        }
                    }
                }
            }
        }
        targets
    }

    /// The candidate the push's fold keeps for the closed pair `key` at `jn`,
    /// over the enclosed pairs of every shape the push closes: a 5′ flank of
    /// at most one, a 3′ flank of at most one, and the tabulated `2×2`, `2×3`,
    /// `3×2` loops.
    fn pulled_single(
        &self,
        t: &Tables,
        (osx, okx): (&OneSideIndex, &OneSideKIndex),
        jn: Node,
        key: PairKey,
    ) -> PulledBest {
        let mut best = PulledBest::default();
        let (pn, qn) = (key.open, key.close);
        let Some(outer) = key
            .open_edge(self.dfa)
            .zip(key.close_edge(self.dfa, jn))
            .and_then(|(ep, eq)| {
                Some(OuterCell {
                    pn,
                    ep,
                    qn,
                    eq,
                    outer_pair: canonical(ep.nuc, eq.nuc)?,
                })
            })
        else {
            return best;
        };
        let windows =
            Self::pulled_windows((osx, okx), (pn.pos, qn.pos), self.single_max_len as u32);
        PUSH_LEFTS.with_borrow_mut(|lefts| {
            PUSH_RIGHTS.with_borrow_mut(|rights| {
                for entry in windows.into_iter().flatten() {
                    self.offer_pulled(t, &outer, entry, (lefts, rights), &mut best);
                }
            });
        });
        best
    }

    /// The enclosed pairs the push closes an outer pair `(p, q)` around, as
    /// index windows: a 5′ flank of at most one (bucket `k = p + 1 + n1`, every
    /// 3′ flank up to the size cap `l_max`), a 3′ flank of at most one with a
    /// 5′ flank of two or more (column `l = q − 1 − n2`, every opening up to
    /// the cap), and the tabulated small loops. The cap is on the total:
    /// `n1 + n2 ≤ l_max`.
    fn pulled_windows<'x>(
        (osx, okx): (&'x OneSideIndex, &'x OneSideKIndex),
        (p, q): (u32, u32),
        l_max: u32,
    ) -> Vec<&'x [OneSideEntry]> {
        let mut windows = Vec::with_capacity(7);
        for n1 in 0..=1u32 {
            let k = p + 1 + n1;
            if k < q {
                // n2 = q − l − 1 ≤ l_max − n1 ⇒ l ≥ q + n1 − l_max − 1.
                windows.push(okx.window(k, (q + n1).saturating_sub(l_max + 1), q - 1));
            }
        }
        for n2 in 0..=1u32 {
            if q > n2 {
                // n1 = k − p − 1 ≤ l_max − n2 ⇒ k ≤ p + 1 + l_max − n2.
                windows.push(osx.window(q - 1 - n2, p + 3, p + 1 + l_max - n2));
            }
        }
        for (n1, n2) in [(2u32, 2u32), (2, 3), (3, 2)] {
            if q > n2 {
                let k = p + 1 + n1;
                windows.push(osx.window(q - 1 - n2, k, k));
            }
        }
        windows
    }

    /// Score the push's closure of `outer` around one enclosed pair and keep
    /// it in `best` if the fold would; `caches` hold the 5′ corners by
    /// opening target and the 3′ corners by enclosed node.
    fn offer_pulled(
        &self,
        t: &Tables,
        outer: &OuterCell,
        entry: &OneSideEntry,
        caches: (&mut CornerCache, &mut CornerCache),
        best: &mut PulledBest,
    ) {
        let Some(left_cai) = self
            .dfa
            .broken_codon_cai_s::<S>(outer.ep.to, entry.ikey.open)
        else {
            return;
        };
        // The two flanks are declared unpaired, so a required position in
        // either forbids this closure — the push truncates its flank scans by
        // the same two bounds rather than filtering.
        if !self.must.free_span(outer.pn.pos + 1, entry.k)
            || !self.must.free_span(entry.io.pos, outer.qn.pos)
        {
            return;
        }
        let n1 = (entry.k - outer.pn.pos - 1) as usize;
        let n2 = (outer.qn.pos - entry.io.pos) as usize;
        let ctx = SingleCtx {
            inner: PairRef {
                outer: entry.io,
                key: entry.ikey,
            },
            st: entry.st,
            ep: outer.ep,
            pn: outer.pn,
            qn: outer.qn,
            eq: outer.eq,
            outer_pair: outer.outer_pair,
            inner_pair: pair_from_nucleotides(entry.ikey.nuc_open(), entry.ikey.nuc_close),
        };
        let cand = if n1 > 0 && n2 > 0 {
            let (lefts, rights) = caches;
            let lefts = lefts.rights(self.dfa, outer.ep.to, entry.ikey.open);
            let rights = rights.lefts(self.dfa, entry.io, outer.qn);
            self.internal_loop_cand_with(t.lambda(), &ctx, lefts, rights, n1, n2)
        } else {
            self.stack_or_bulge_cand(&ctx, left_cai, n1, n2)
        };
        let Some(cand) = cand else {
            return;
        };
        let Some(score) = t.p_score(cand.1, cand.2) else {
            return;
        };
        let arrival = (entry.io.pos, entry.rank);
        // Only a scan that can leave a target cell absent needs to know when
        // the push would have created it, and the walk's scans never do: this
        // is one compare per candidate offered, and the push offers hundreds
        // of millions of them at length.
        if self.must.is_active() {
            best.first = best.first.min(arrival);
        }
        if best
            .cand
            .is_none_or(|_| score < best.score || (score == best.score && arrival < best.arrival))
        {
            *best = PulledBest {
                score,
                arrival,
                first: best.first,
                cand: Some(cand),
            };
        }
    }

    /// The `P`-write candidate an internal-loop closure produces (or `None`) —
    /// the pure-scoring core, jointly optimizing the terminal-mismatch
    /// nucleotides on both flanks against their CAI. Shared by the serial
    /// [`Self::internal_loop`] and the parallel pull; the 3′-flank scratch is
    /// the thread-local [`RIGHTS`] so parallel workers never contend.
    fn internal_loop_cand(
        &self,
        lambda: f64,
        ctx: &SingleCtx,
        lefts: &[Corner],
        n1: usize,
        n2: usize,
    ) -> Option<(Node, PairKey, State)> {
        RIGHTS.with_borrow_mut(|rights| {
            corner_scores_into(self.dfa, ctx.inner.outer, ctx.qn, rights);
            self.internal_loop_cand_with(lambda, ctx, lefts, rights, n1, n2)
        })
    }

    /// [`Self::internal_loop_cand`] with the 3′ flank's corners already
    /// enumerated: `rights` must be [`corner_scores_into`] from the inner
    /// pair's outer node to `ctx.qn`, which closures sharing those two nodes
    /// share.
    fn internal_loop_cand_with(
        &self,
        lambda: f64,
        ctx: &SingleCtx,
        lefts: &[Corner],
        rights: &[Corner],
        n1: usize,
        n2: usize,
    ) -> Option<(Node, PairKey, State)> {
        if self.validate_oneside {
            self.check_oneside_factor(ctx, lefts, rights, n1, n2, lambda);
        }
        let pairs = LoopPairs {
            outer: ctx.outer_pair,
            inner: ctx.inner_pair,
        };
        let (energy, si_cai) = score_internal_from(
            self.params,
            lambda,
            &pairs,
            (
                &ScoredStrand {
                    corners: lefts,
                    len: n1,
                },
                &ScoredStrand {
                    corners: rights,
                    len: n2,
                },
            ),
            &self.one_by_n,
        )?;
        let cai = ctx.st.cai + ctx.ep.cai + si_cai + ctx.eq.cai;
        Some((
            ctx.eq.to,
            ctx.out_key(),
            State::new(ctx.st.mfe + energy, cai, ctx.back()),
        ))
    }

    /// De-risk gate for the lever-3 one-sided reduction
    /// (`DERNA_ONESIDE_VALIDATE`): assert the **skewed one-sided** closure's
    /// factored corner minimum equals the full `lefts × rights` enumeration.
    ///
    /// When exactly one flank is large (`≥ 5`, so it separates over a codon
    /// boundary into `CL/DL` halves) and the loop is skewed (`d ≥ T`, saturated
    /// ninio), the corner minimisation factors: the separated large flank's
    /// outer half (`l5` or `r3`) minimises independently of its inner half, so
    /// the joint `lefts × rights` min collapses to one min over the *small*
    /// flank's corners times `O(1)` per-large-flank-edge sub-minima. This is the
    /// algebra the reduction relies on; the assertion catches any derivation or
    /// implementation drift before it can perturb the optimum, mirroring H4's
    /// env-gated separation check.
    fn check_oneside_factor(
        &self,
        ctx: &SingleCtx,
        lefts: &[Corner],
        rights: &[Corner],
        n1: usize,
        n2: usize,
        lambda: f64,
    ) {
        let dskew = self.ninio_skew_threshold();
        let large5 = n1 >= 5 && (2..=4).contains(&n2) && n1 - n2 >= dskew;
        let large3 = n2 >= 5 && (2..=4).contains(&n1) && n2 - n1 >= dskew;
        if !(large5 || large3) {
            return;
        }
        let kk = (1.0 - lambda) * 100.0;
        let mm = &self.params.mismatch_interior;
        let o = ctx.outer_pair;
        let irev = ctx.inner_pair.reversed();
        // Direct: full corner enumeration (the value the reduction must match).
        let mut direct = f64::INFINITY;
        for l in lefts {
            for r in rights {
                let e = lambda
                    * (f64::from(mm.get(o, l.first.to_nuc5(), r.last.to_nuc5()))
                        + f64::from(mm.get(irev, r.first.to_nuc5(), l.last.to_nuc5())))
                    - kk * (l.cai + r.cai);
                direct = direct.min(e);
            }
        }
        // Factored: the large (separated) flank's two halves minimise
        // independently; iterate only the small flank's corners.
        let factored = if large5 {
            // 5′ flank large: A(r3) over `l5`, B(r5) over `l3`; small flank = 3′.
            let mut best = f64::INFINITY;
            for r in rights {
                let (r5, r3) = (r.first.to_nuc5(), r.last.to_nuc5());
                let mut a = f64::INFINITY;
                for l5 in self.dfa.right(ctx.ep.to) {
                    let cl = l5.cai + self.start_contrib(l5.to);
                    a = a.min(lambda * f64::from(mm.get(o, l5.nuc.to_nuc5(), r3)) - kk * cl);
                }
                let mut b = f64::INFINITY;
                for l3 in self.dfa.left(ctx.inner.key.open) {
                    let dl = self.end_contrib(l3.to) + l3.cai;
                    b = b.min(lambda * f64::from(mm.get(irev, r5, l3.nuc.to_nuc5())) - kk * dl);
                }
                best = best.min(a + b - kk * r.cai);
            }
            best
        } else {
            // 3′ flank large: Am(l5) over `r3`, Bm(l3) over `r5`; small flank = 5′.
            let mut best = f64::INFINITY;
            for l in lefts {
                let (l5, l3) = (l.first.to_nuc5(), l.last.to_nuc5());
                let mut am = f64::INFINITY;
                for r3 in self.dfa.left(ctx.qn) {
                    let cr = r3.cai + self.end_contrib(r3.to);
                    am = am.min(lambda * f64::from(mm.get(o, l5, r3.nuc.to_nuc5())) - kk * cr);
                }
                let mut bm = f64::INFINITY;
                for r5 in self.dfa.right(ctx.inner.outer) {
                    let dr = r5.cai + self.start_contrib(r5.to);
                    bm = bm.min(lambda * f64::from(mm.get(irev, r5.nuc.to_nuc5(), l3)) - kk * dr);
                }
                best = best.min(am + bm - kk * l.cai);
            }
            best
        };
        let diff = (direct - factored).abs();
        assert!(
            diff < 1e-6,
            "oneside factor mismatch: n1={n1} n2={n2} large5={large5} direct={direct} factored={factored} diff={diff}",
        );
    }

    /// Pull the generic internal-loop closures into the `P[pos]` column
    /// ([`InternalLoopMode::Lz`]). For every outer pair whose 3′ base sits at
    /// `pos - 1`, the small / balanced shapes are scored directly
    /// ([`Self::pull_close_side`]) while the skewed-large shapes are reduced to
    /// an `O(1)` per-size-diagonal range-min over the precomputed `H` index
    /// ([`Self::skewed_pull`]). Both write the same `(energy, cai,
    /// Back::Single)` state the push path would, via [`Self::internal_loop`].
    /// A no-op in `Naive` mode.
    #[allow(clippy::too_many_arguments)]
    fn single_close_pull(
        &self,
        t: &mut Tables,
        hx: &HIndex,
        osx: &OneSideIndex,
        okx: &OneSideKIndex,
        pos: u32,
        prof: &mut Profile,
        pool: &mut [Writes],
    ) {
        if !self.pull_generic() || pos == 0 {
            return;
        }
        let q = pos - 1;
        let sides: Vec<(Node, Edge)> = self
            .dfa
            .nodes_at(q)
            .iter()
            .flat_map(|&qn| self.dfa.right(qn).iter().map(move |&eq| (qn, eq)))
            .collect();
        if sides.is_empty() {
            return;
        }
        // The two independent outer axes shared by all sides (`qn.pos == q`):
        // the outer 5′ position `p ∈ [0, p_hi]` for the skewed / one-sided
        // families, and the inner 3′ position `l ∈ [l_lo, l_hi]` for the direct
        // family. Every closure reads finalized cells (`< pos`) and appends a
        // `P[pos]` candidate, so the outer cells are independent.
        let p_hi = q.saturating_sub(13);
        let n_p = p_hi + 1;
        let l_hi = q.saturating_sub(3);
        // n2 = q − l − 1 ≤ L − 2 (n1 ≥ 2, n1 + n2 ≤ L), so l ≥ q − (L − 1).
        let l_lo = q.saturating_sub((self.single_max_len - 1) as u32);
        let n_l = (l_hi + 1).saturating_sub(l_lo);
        let scan_total = sides.len() as u32 * n_p;
        let total = scan_total + sides.len() as u32 * n_l;

        // Small positions (and single-thread) run serially into one buffer,
        // folded in production order — byte-identical to the pre-parallel path.
        if self.threads <= 1 || total < PULL_PARALLEL_MIN {
            let w = &mut pool[0];
            for &(qn, eq) in &sides {
                let m = prof.mark();
                self.pull_close_side(t, w, qn, eq);
                prof.record(Phase::PullDirect, m);
                let m = prof.mark();
                self.skewed_pull(t, w, hx, qn, eq);
                prof.record(Phase::PullSkewed, m);
                let m = prof.mark();
                self.oneside_pull(t, w, osx, qn, eq);
                prof.record(Phase::PullOneside, m);
                let m = prof.mark();
                self.oneside_pull3(t, w, okx, qn, eq);
                prof.record(Phase::PullOneside3, m);
            }
            let m = prof.mark();
            w.apply(t);
            w.clear();
            prof.record(Phase::PullApply, m);
            return;
        }

        // Parallel: workers take the outer cells from two shared counters, each
        // accumulating a private `Writes`; the buffers are then folded in worker
        // order. The min-BRD merge makes every cell's BRD independent of that
        // order (on an exact tie the first arrival wins; see `Scan::threads`).
        //
        // The direct family's outer cells come first and one at a time: each
        // scores a whole window of flank shapes, so a run of them is the most
        // expensive work at the position, and dealt in the skewed families'
        // chunk size one such run could outlast every other worker. The
        // skewed and one-sided cells, several hundred cheap ones per side, then
        // fill in behind them in `PULL_CHUNK`s.
        let direct_total = total - scan_total;
        let direct = AtomicU32::new(0);
        let skewed = AtomicU32::new(0);
        let n_workers = self.fan_workers(total.div_ceil(PULL_CHUNK) as usize);
        // Reborrow the tables immutably so every worker can read them concurrently;
        // the `&mut` is reclaimed for the fold once the scope ends.
        let shared: &Tables = t;
        let timing = prof.is_on();
        let m = prof.mark();
        let buffers: Vec<Mutex<&mut Writes>> =
            pool[..n_workers].iter_mut().map(Mutex::new).collect();
        let clocks: Vec<WorkerClock> = self.fan_collect(n_workers, |me| {
            let mut w = buffers[me].lock().unwrap();
            let (direct, skewed, sides) = (&direct, &skewed, &sides);
            let mut clock = WorkerClock::start(timing);
            let mut run = |i| {
                self.run_pull_index(
                    shared, &mut w, hx, osx, okx, sides, i, n_p, n_l, l_lo, scan_total,
                );
            };
            loop {
                let j = direct.fetch_add(PULL_DIRECT_CHUNK, Ordering::Relaxed);
                if j >= direct_total {
                    break;
                }
                let chunk = timing.then(Instant::now);
                for j in j..(j + PULL_DIRECT_CHUNK).min(direct_total) {
                    run(scan_total + j);
                }
                clock.chunk(chunk, true);
            }
            loop {
                let start = skewed.fetch_add(PULL_CHUNK, Ordering::Relaxed);
                if start >= scan_total {
                    break;
                }
                let chunk = timing.then(Instant::now);
                for i in start..(start + PULL_CHUNK).min(scan_total) {
                    run(i);
                }
                clock.chunk(chunk, false);
            }
            clock.stop()
        });
        drop(buffers);
        let buffers = &mut pool[..n_workers];
        prof.record(Phase::FanPullCompute, m);
        record_fan(prof, FanSite::Pull, buffers, &clocks);
        let m = prof.mark();
        for w in buffers.iter_mut() {
            w.apply(t);
            w.clear();
        }
        prof.record(Phase::FanPullApply, m);
    }

    /// Run the single flat outer-cell index `i` of the parallel pull into `w`:
    /// the low `scan_total` indices are `(side, p)` for the skewed / one-sided
    /// families, the rest `(side, l)` for the direct family.
    #[allow(clippy::too_many_arguments)]
    fn run_pull_index(
        &self,
        t: &Tables,
        w: &mut Writes,
        hx: &HIndex,
        osx: &OneSideIndex,
        okx: &OneSideKIndex,
        sides: &[(Node, Edge)],
        i: u32,
        n_p: u32,
        n_l: u32,
        l_lo: u32,
        scan_total: u32,
    ) {
        if i < scan_total {
            let (qn, eq) = sides[(i / n_p) as usize];
            let p = i % n_p;
            self.skewed_pull_at(t, w, hx, qn, eq, p);
            self.oneside_pull_at(t, w, osx, qn, eq, p);
            self.oneside_pull3_at(t, w, okx, qn, eq, p);
        } else {
            let j = i - scan_total;
            let (qn, eq) = sides[(j / n_l) as usize];
            self.pull_close_side_at(t, w, qn, eq, l_lo + j % n_l);
        }
    }

    /// Score the **direct** generic closures sharing the outer 3′ side
    /// `(qn, eq)`: the small-flank shapes (`n1 ≤ 4` or `n2 ≤ 4`) and the
    /// balanced-large shapes (`n1, n2 ≥ 5`, `|n1 − n2|` below the ninio
    /// saturation threshold). These either do not separate over a codon
    /// boundary or carry a varying ninio term, so they keep the full corner
    /// enumeration; the skewed-large remainder is handled by
    /// [`Self::skewed_pull`].
    fn pull_close_side(&self, t: &Tables, w: &mut Writes, qn: Node, eq: Edge) {
        let q = qn.pos;
        // n2 = q - l - 1 ∈ [2, SINGLE_MAX_LEN - 2] (n1 ≥ 2, n1 + n2 ≤ SINGLE_MAX_LEN).
        let l_hi = q.saturating_sub(3);
        let l_lo = q.saturating_sub((self.single_max_len - 1) as u32);
        for l in l_lo..=l_hi {
            self.pull_close_side_at(t, w, qn, eq, l);
        }
    }

    /// One inner-3′ row `l` of [`Self::pull_close_side`] — the unit of parallel
    /// work for the direct family. Independent across `l` (each reads finalized
    /// `P[l+1]` and appends `P[pos]` candidates to `w`).
    fn pull_close_side_at(&self, t: &Tables, w: &mut Writes, qn: Node, eq: Edge, l: u32) {
        let q = qn.pos;
        if l + 1 >= q {
            return;
        }
        let n2 = (q - l - 1) as usize;
        if !(2..=self.single_max_len - 2).contains(&n2) {
            return;
        }
        // The whole `l`-row shares the 3′ flank `[l + 1, q)`, so a required
        // position inside it drops the row before any inner cell is read.
        if !self.must.free_span(l + 1, q) {
            return;
        }
        // Every closure of the row around inner node `io` shares its 3′
        // corners, `io → qn`, so they are enumerated once per node.
        ROW_RIGHTS.with_borrow_mut(|rights| {
            for &io in self.dfa.nodes_at(l + 1) {
                let mut ready = false;
                for (ikey, st) in t.p_states(io) {
                    if !ready {
                        corner_scores_into(self.dfa, io, qn, rights);
                        ready = true;
                    }
                    let cell = InnerCell { io, ikey, st };
                    let side = RowSide { qn, eq, rights };
                    self.pull_inner_direct(t, w, &side, &cell, n2);
                }
            }
        });
    }

    /// Score the **small-flank** generic closures of one inner pair `(k, l)`
    /// (`cell`) against the fixed outer 3′ side `(qn, eq)`: every `n1` when the
    /// 3′ flank is itself small (`n2 ≤ 4`), otherwise only `n1 ∈ {2, 3, 4}`.
    /// These do not separate over a codon boundary, so they keep the full
    /// corner enumeration. Every closure with both flanks `≥ 5` (balanced or
    /// skewed) is reduced by [`Self::skewed_pull`] instead.
    fn pull_inner_direct(
        &self,
        t: &Tables,
        w: &mut Writes,
        side: &RowSide,
        cell: &InnerCell,
        n2: usize,
    ) {
        let dskew = self.ninio_skew_threshold();
        // Skip the skewed one-sided shapes (saturated ninio, one flank `≥ 5`
        // that separates over a codon boundary) — they are reduced by
        // [`Self::oneside_pull`] (large 5′) / [`Self::oneside_pull3`] (large 3′).
        // Only the balanced / short-flank shapes stay on the direct path.
        let (lo, hi) = if n2 <= 4 {
            // Small 3′ flank: large 5′ is `n1 ≥ n2 + T`; n1 + n2 ≤ L.
            (2, (n2 + dskew - 1).min(self.single_max_len - n2))
        } else {
            // Large 3′ flank (`n1 ∈ {2,3,4}`): large 3′ is `n1 ≤ n2 − T`.
            ((n2.saturating_sub(dskew) + 1).max(2), 4)
        };
        for n1 in lo..=hi {
            self.direct_close(t, w, side, cell, n1, n2);
        }
    }

    /// Score one direct generic closure: outer pair opening at `p = k − n1 − 1`
    /// and closing at `(qn, eq)`, around inner pair `cell`. Writes through the
    /// shared [`Self::internal_loop`] (full corner enumeration), so the stored
    /// `(energy, cai)` is exact.
    fn direct_close(
        &self,
        t: &Tables,
        w: &mut Writes,
        side: &RowSide,
        cell: &InnerCell,
        n1: usize,
        n2: usize,
    ) {
        let RowSide { qn, eq, rights } = *side;
        let k = cell.ikey.open.pos;
        if !is_generic(n1, n2) || k < (n1 + 1) as u32 {
            return;
        }
        let p = k - n1 as u32 - 1;
        // 5′ flank `[p + 1, k)`; the 3′ flank was cleared for the whole row.
        if !self.must.free_span(p + 1, k) {
            return;
        }
        let inner = PairRef {
            outer: cell.io,
            key: cell.ikey,
        };
        let inner_pair = pair_from_nucleotides(cell.ikey.nuc_open(), cell.ikey.nuc_close);
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                let ctx = SingleCtx {
                    inner,
                    st: cell.st.into(),
                    ep: *ep,
                    pn,
                    qn,
                    eq,
                    outer_pair,
                    inner_pair,
                };
                LEFTS.with_borrow_mut(|lefts| {
                    corner_scores_into(self.dfa, ep.to, cell.ikey.open, lefts);
                    if let Some(cand) =
                        self.internal_loop_cand_with(t.lambda(), &ctx, lefts, rights, n1, n2)
                    {
                        w.p.push(cand);
                    }
                });
            }
        }
    }

    /// The asymmetry `d` at or above which the ninio term saturates to
    /// `MAX_NINIO` (`ceil(MAX_NINIO / ninio_37)`). Shapes with `d` below it are
    /// "balanced" (varying ninio, scored directly); at or above it the loop is
    /// "skewed" (constant ninio, eligible for the reduction).
    fn ninio_skew_threshold(&self) -> usize {
        self.ninio_skew
    }

    /// The `H`-index slot of size diagonal `delta` for inner pairs whose 3′
    /// node reaches the stops `class` ([`Dfa::reach_class`]). Each class has a
    /// slot range of its own, so a range-min never returns an inner pair the
    /// outer pair cannot reach; class `0`, the only one on a lattice that is
    /// not stop-labelled, keeps the diagonal itself.
    fn h_slot(&self, delta: i64, class: u8) -> i64 {
        delta + i64::from(class) * (i64::from(self.dfa.seq_len) + 1)
    }

    /// The inner-pair classes ([`Self::h_slot`]) an outer pair can enclose
    /// when its 5′ flank starts at `open` (the node after the opening
    /// nucleotide) and its 3′ flank ends at `qn`: every class reachable from
    /// `open` that reaches `qn`, in ascending order. Only class `0` unless the
    /// lattice is stop-labelled.
    fn h_classes(&self, open: Node, qn: Node) -> SmallVec<[u8; 5]> {
        if self.dfa.stop_exit().is_none() {
            return smallvec::smallvec![0];
        }
        let outer = self.reach_of(open);
        (0..8u8)
            .filter(|&c| c == 0 || (outer == 0 || c & !outer == 0) && self.dfa.reaches(c, qn))
            .collect()
    }

    /// 5′ broken-codon CAI contribution of a flank that *begins* at `node`:
    /// [`Dfa::span_head_cai`], the 5′ half of the cross-codon split
    /// `cai(a → b) = Start(a) + End(b)`.
    fn start_contrib(&self, node: Node) -> f64 {
        self.dfa.span_head_cai(node).unwrap_or(0.0)
    }

    /// 3′ broken-codon CAI contribution of a flank that *ends* at `node`:
    /// [`Dfa::span_tail_cai`], the 3′ half of the split.
    fn end_contrib(&self, node: Node) -> f64 {
        self.dfa.span_tail_cai(node).unwrap_or(0.0)
    }

    /// The outer-pair-independent reduction value of an inner pair: its own
    /// BRD plus the `O(1)` inner-corner minimum
    /// `min over (l3, r5) [ λ·MM_inner(I_rev, r5, l3) − K·(DL(l3) + DR(r5)) ]`,
    /// where `DL` / `DR` are the inner halves of the separated flank CAI. Valid
    /// as a skewed-loop contribution wherever both flanks span a codon boundary
    /// (`n1, n2 ≥ 5`), which is exactly where the reduction consults it.
    fn inner_h(&self, io: Node, ikey: PairKey, st: State, lambda: f64) -> f64 {
        let kk = (1.0 - lambda) * 100.0;
        let inner_rev = pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close).reversed();
        let mm = &self.params.mismatch_interior;
        let mut imin = f64::INFINITY;
        for l3 in self.dfa.left(ikey.open) {
            let dl = self.end_contrib(l3.to) + l3.cai;
            for r5 in self.dfa.right(io) {
                let e = mm.get(inner_rev, r5.nuc.to_nuc5(), l3.nuc.to_nuc5());
                let dr = r5.cai + self.start_contrib(r5.to);
                imin = imin.min(lambda * f64::from(e) - kk * (dl + dr));
            }
        }
        st.brd(lambda) + imin
    }

    /// The outer-corner minimum of a skewed closure for the outer cell `oc`:
    /// `min over (l5, r3) [ λ·MM_outer(O, l5, r3) − K·(CL(l5) + CR(r3)) ]` less
    /// the outer opening/closing edge CAI, where `CL` / `CR` are the outer
    /// halves of the separated flank CAI. Independent of the inner pair.
    fn outer_min(&self, oc: &OuterCell, lambda: f64) -> f64 {
        let kk = (1.0 - lambda) * 100.0;
        let mm = &self.params.mismatch_interior;
        let mut omin = f64::INFINITY;
        for l5 in self.dfa.right(oc.ep.to) {
            let cl = l5.cai + self.start_contrib(l5.to);
            for r3 in self.dfa.left(oc.qn) {
                let e = mm.get(oc.outer_pair, l5.nuc.to_nuc5(), r3.nuc.to_nuc5());
                let cr = r3.cai + self.end_contrib(r3.to);
                omin = omin.min(lambda * f64::from(e) - kk * (cl + cr));
            }
        }
        omin - kk * (oc.ep.cai + oc.eq.cai)
    }

    /// Record each finalized `P[pos]` cell's reduction value `H` on its size
    /// diagonal `Δ = (pos − 1) − k`, so later positions' [`Self::skewed_pull`]
    /// can range-min over it. `Lz`-only (and the column index alone for the
    /// `Naive` column fill); called once `P[pos]` is complete.
    ///
    /// Each cell's entry is a pure function of the cell and its rank in the
    /// column, so the entries are computed first and appended afterwards, in
    /// cell order.
    fn index_finalized_p(
        &self,
        t: &Tables,
        hx: &mut HIndex,
        osx: &mut OneSideIndex,
        okx: &mut OneSideKIndex,
        pos: u32,
        prof: &mut Profile,
    ) {
        if pos == 0 || !(self.pull_generic() || self.pull_fill) {
            return;
        }
        let m = prof.mark();
        let cells = self.finalized_cells(t, pos);
        prof.record(Phase::IndexCompute, m);
        if self.pull_generic() {
            let m = prof.mark();
            self.index_h(&cells, t.lambda(), hx, pos - 1);
            prof.record(Phase::IndexFinalizedP, m);
            let m = prof.mark();
            self.index_oneside(&cells, t.lambda(), osx, okx, pos - 1);
            prof.record(Phase::IndexOneSide, m);
        } else {
            // `Naive` under the column fill: the fill gathers its inner pairs
            // from the column index and needs no reduction's minima.
            let m = prof.mark();
            self.index_columns_only(&cells, t.lambda(), osx, pos - 1);
            prof.record(Phase::IndexOneSide, m);
        }
    }

    /// Every finalized `P[pos]` cell, in node and table order.
    fn finalized_cells(&self, t: &Tables, pos: u32) -> Vec<(Node, PairKey, State)> {
        self.dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&io| t.p_states(io).into_iter().map(move |(k, s)| (io, k, s)))
            .collect()
    }

    /// The size-diagonal half of [`Self::index_finalized_p`]: each cell's
    /// `H`, appended in cell order.
    fn index_h(&self, cells: &[(Node, PairKey, State)], lambda: f64, hx: &mut HIndex, l: u32) {
        for &(io, ikey, st) in cells {
            let k = ikey.open.pos;
            let h = self.inner_h(io, ikey, st, lambda);
            hx.insert(
                self.h_slot(l as i64 - k as i64, self.reach_of(io)),
                HEntry {
                    k,
                    h,
                    io,
                    ikey,
                    st: st.into(),
                },
            );
        }
    }

    /// The one-sided half of [`Self::index_finalized_p`]: each cell's entry,
    /// ranked by its place in the column, appended in cell order to the
    /// column index (then sorted by `k`) and to the `k` index (which stays
    /// `l`-sorted by construction).
    fn index_oneside(
        &self,
        cells: &[(Node, PairKey, State)],
        lambda: f64,
        osx: &mut OneSideIndex,
        okx: &mut OneSideKIndex,
        l: u32,
    ) {
        for (rank, &(io, ikey, st)) in (0u32..).zip(cells) {
            let entry = OneSideEntry {
                k: ikey.open.pos,
                io,
                ikey,
                st: st.into(),
                brd: st.brd(lambda),
                bmin: self.oneside_bmin(ikey, lambda),
                rank,
            };
            osx.push(l, entry);
            let bmin = self.oneside_bmin3(io, ikey, lambda);
            okx.push(entry.k, OneSideEntry { bmin, ..entry });
        }
        osx.sort_col(l);
    }

    /// The column index alone, for the `Naive` column fill: each cell's entry
    /// with its BRD and no reduction minima (the `Lz` reductions read those,
    /// and they do not run), appended in cell order and sorted by `k`.
    fn index_columns_only(
        &self,
        cells: &[(Node, PairKey, State)],
        lambda: f64,
        osx: &mut OneSideIndex,
        l: u32,
    ) {
        for (rank, &(io, ikey, st)) in (0u32..).zip(cells) {
            osx.push(
                l,
                OneSideEntry {
                    k: ikey.open.pos,
                    io,
                    ikey,
                    st: st.into(),
                    brd: st.brd(lambda),
                    bmin: [f64::INFINITY; 4],
                    rank,
                },
            );
        }
        osx.sort_col(l);
    }

    /// The one-sided reduction's inner-half minima for an inner pair, keyed by
    /// the 3′-flank 5′ terminal `r5` (`A/C/G/U → 0..4`):
    /// `B(r5) = min over l3 [ λ·MM_inner(I_rev, r5, l3) − K·DL(l3) ]`, where
    /// `DL(l3) = End(l3) + l3.cai`. Outer-independent — read once per inner
    /// `P`-cell and stored in [`OneSideEntry::bmin`]. The large-5′ flank (`≥ 5`)
    /// separates over a codon boundary, so `DL` is its inner CAI half.
    fn oneside_bmin(&self, ikey: PairKey, lambda: f64) -> [f64; 4] {
        let kk = (1.0 - lambda) * 100.0;
        let irev = pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close).reversed();
        let mm = &self.params.mismatch_interior;
        let mut bmin = [f64::INFINITY; 4];
        for l3 in self.dfa.left(ikey.open) {
            let dl = self.end_contrib(l3.to) + l3.cai;
            let l3n = l3.nuc.to_nuc5();
            for r5 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = lambda * f64::from(mm.get(irev, r5.to_nuc5(), l3n)) - kk * dl;
                let idx = r5 as usize;
                if e < bmin[idx] {
                    bmin[idx] = e;
                }
            }
        }
        bmin
    }

    /// The mirror of [`Self::oneside_bmin`] for the large-3′ family, keyed by
    /// the 5′-flank 3′ terminal `l3`:
    /// `Bm(l3) = min over r5 [ λ·MM_inner(I_rev, r5, l3) − K·DR(r5) ]`, where
    /// `DR(r5) = r5.cai + Start(r5)` is the large 3′ flank's inner CAI half.
    /// Outer-independent — stored in [`OneSideKIndex`]'s [`OneSideEntry::bmin`].
    fn oneside_bmin3(&self, io: Node, ikey: PairKey, lambda: f64) -> [f64; 4] {
        let kk = (1.0 - lambda) * 100.0;
        let irev = pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close).reversed();
        let mm = &self.params.mismatch_interior;
        let mut bmin = [f64::INFINITY; 4];
        for r5 in self.dfa.right(io) {
            let dr = r5.cai + self.start_contrib(r5.to);
            let r5n = r5.nuc.to_nuc5();
            for l3 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = lambda * f64::from(mm.get(irev, r5n, l3.to_nuc5())) - kk * dr;
                let idx = l3 as usize;
                if e < bmin[idx] {
                    bmin[idx] = e;
                }
            }
        }
        bmin
    }

    /// Reduce the skewed **large-5′** one-sided closures sharing the outer 3′
    /// side `(qn, eq)`: large 5′ flank (`n1 ≥ n2 + T`, separates over a codon
    /// boundary) and small 3′ flank (`n2 ∈ {2,3,4}`, explicit). Enumerate the
    /// outer 5′ side `(p, pn, ep)` and, for each outer cell, find its best
    /// large-5′ inner pair via [`Self::oneside_outer_cell`].
    fn oneside_pull(&self, t: &Tables, w: &mut Writes, osx: &OneSideIndex, qn: Node, eq: Edge) {
        // A large-5′ inner needs `n1 ≥ n2 + T ≥ 8` and `n2 ≥ 2`, so `k ≥ p + 9`
        // and `l ≤ q − 3`, hence `p ≤ q − 13`.
        let p_hi = qn.pos.saturating_sub(13);
        for p in 0..=p_hi {
            self.oneside_pull_at(t, w, osx, qn, eq, p);
        }
    }

    /// One outer 5′ position `p` of [`Self::oneside_pull`] — the unit of parallel
    /// work for the large-5′ family (independent across `p`).
    fn oneside_pull_at(
        &self,
        t: &Tables,
        w: &mut Writes,
        osx: &OneSideIndex,
        qn: Node,
        eq: Edge,
        p: u32,
    ) {
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                let oc = OuterCell {
                    pn,
                    ep: *ep,
                    qn,
                    eq,
                    outer_pair,
                };
                self.oneside_outer_cell(t, w, osx, &oc);
            }
        }
    }

    /// Find and write one outer cell's best large-5′ one-sided closure. For each
    /// small `n2 ∈ {2,3,4}`, the eligible inner pairs sit in column `l = q−1−n2`
    /// with `k` in `[p+1+n2+T, …]`; the factored candidate reads the outer half
    /// `A(r3)` (precomputed here over `l5`) and the inner half `B(r5)`
    /// (precomputed in the index) with the small 3′ flank enumerated explicitly.
    /// The single best candidate over all `n2` is re-scored by the exact
    /// [`Self::internal_loop`], so the stored `(energy, cai)` is byte-exact and
    /// the optimum is preserved.
    fn oneside_outer_cell(&self, t: &Tables, w: &mut Writes, osx: &OneSideIndex, oc: &OuterCell) {
        let (p, q) = (oc.pn.pos, oc.qn.pos);
        let lambda = t.lambda();
        let kk = (1.0 - lambda) * 100.0;
        let max_ninio = f64::from(self.params.ninio.max_ninio);
        let dskew = self.ninio_skew_threshold() as u32;
        let mm = &self.params.mismatch_interior;
        // A(r3) = min over l5∈right(ep.to) [ λ·MM_outer(O,l5,r3) − K·CL(l5) ],
        // keyed by `r3` nucleotide; independent of the inner pair.
        let mut a_by_r3 = [f64::INFINITY; 4];
        for l5 in self.dfa.right(oc.ep.to) {
            let cl = l5.cai + self.start_contrib(l5.to);
            let l5n = l5.nuc.to_nuc5();
            for r3 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = lambda * f64::from(mm.get(oc.outer_pair, l5n, r3.to_nuc5())) - kk * cl;
                let idx = r3 as usize;
                if e < a_by_r3[idx] {
                    a_by_r3[idx] = e;
                }
            }
        }
        if !a_by_r3.iter().any(|a| a.is_finite()) {
            return;
        }
        let edge = -kk * (oc.ep.cai + oc.eq.cai);
        // The large 5′ flank `[p + 1, k)` is the reduced side, the small 3′
        // flank `[l + 1, q)` the explicitly enumerated one. A required position
        // caps `k` on the reduced side — a truncation of the `k`-window the
        // column index is binary-searched over — and rejects a whole `n2` on the
        // explicit side, since that flank is fixed by `(q, n2)` alone. Neither
        // bound reaches the corner factorization, which minimizes over
        // *nucleotide* choices within the flanks and not over their extent.
        let k_cap = self.must.max_right(p + 1);
        let mut best: Option<(f64, &OneSideEntry, usize, usize)> = None;
        for n2 in 2..=4u32 {
            if q < n2 + 2 {
                continue;
            }
            let l = q - 1 - n2;
            if !self.must.free_span(l + 1, q) {
                continue;
            }
            // n1 = k − p − 1 ≥ n2 + T, and n1 + n2 ≤ L ⇒ k ≤ p + 1 + L − n2.
            let klo = p + 1 + n2 + dskew;
            let khi = (p + 1 + self.single_max_len as u32 - n2)
                .min(q - 2 - n2)
                .min(k_cap);
            if klo > khi {
                continue;
            }
            let labelled = self.dfa.stop_exit().is_some();
            for entry in osx.window(l, klo, khi) {
                // Entries share a window across stop lanes; only those both
                // flanks reach can be enclosed.
                if labelled
                    && (!self.dfa.connected(oc.ep.to, entry.ikey.open)
                        || !self.dfa.connected(entry.io, oc.qn))
                {
                    continue;
                }
                let n1 = (entry.k - p - 1) as usize;
                let cm = self.oneside_corner_min(entry, oc.qn, &a_by_r3, kk);
                if !cm.is_finite() {
                    continue;
                }
                let size = f64::from(self.params.interior.values[n1 + n2 as usize]);
                let cand = entry.brd + lambda * (size + max_ninio) + cm + edge;
                if best.as_ref().is_none_or(|b| cand < b.0) {
                    best = Some((cand, entry, n1, n2 as usize));
                }
            }
        }
        let Some((_, entry, n1, n2)) = best else {
            return;
        };
        let ctx = SingleCtx {
            inner: PairRef {
                outer: entry.io,
                key: entry.ikey,
            },
            st: entry.st,
            ep: oc.ep,
            pn: oc.pn,
            qn: oc.qn,
            eq: oc.eq,
            outer_pair: oc.outer_pair,
            inner_pair: pair_from_nucleotides(entry.ikey.nuc_open(), entry.ikey.nuc_close),
        };
        LEFTS.with_borrow_mut(|lefts| {
            corner_scores_into(self.dfa, oc.ep.to, entry.ikey.open, lefts);
            if let Some(cand) = self.internal_loop_cand(lambda, &ctx, lefts, n1, n2) {
                w.p.push(cand);
            }
        });
    }

    /// The factored corner minimum of a large-5′ candidate at the fixed outer
    /// cell: `min over (r5, r3) [ A(r3) + B(r5) − K·caiR(r5, r3) ]`, where the
    /// small 3′ flank's `(r5, r3, caiR)` are enumerated explicitly over the
    /// inner pair's right edges and the outer 3′ left edges (the corner
    /// enumeration of [`crate::interior::corner_scores`], inlined
    /// allocation-free). `A` is the precomputed outer half; `B` rides in
    /// [`OneSideEntry::bmin`].
    fn oneside_corner_min(
        &self,
        entry: &OneSideEntry,
        qn: Node,
        a_by_r3: &[f64; 4],
        kk: f64,
    ) -> f64 {
        let mut cm = f64::INFINITY;
        for r5 in self.dfa.right(entry.io) {
            let b = entry.bmin[r5.nuc as usize];
            if !b.is_finite() {
                continue;
            }
            for r3 in self.dfa.left(qn) {
                let Some(mid) = self.dfa.broken_codon_cai_s::<S>(r5.to, r3.to) else {
                    continue;
                };
                let a = a_by_r3[r3.nuc as usize];
                if a.is_finite() {
                    let cair = r5.cai + mid + r3.cai;
                    cm = cm.min(a + b - kk * cair);
                }
            }
        }
        cm
    }

    /// The mirror of [`Self::oneside_pull`] for the skewed **large-3′** family:
    /// large 3′ flank (`n2 ≥ n1 + T`, separates) and small 5′ flank
    /// (`n1 ∈ {2,3,4}`, explicit). The inner pair's 5′ base `k = p + 1 + n1` is
    /// pinned near the opening, so eligibility is a query on the `k`-index.
    fn oneside_pull3(&self, t: &Tables, w: &mut Writes, okx: &OneSideKIndex, qn: Node, eq: Edge) {
        let p_hi = qn.pos.saturating_sub(13);
        for p in 0..=p_hi {
            self.oneside_pull3_at(t, w, okx, qn, eq, p);
        }
    }

    /// One outer 5′ position `p` of [`Self::oneside_pull3`] — the unit of
    /// parallel work for the large-3′ family (independent across `p`).
    fn oneside_pull3_at(
        &self,
        t: &Tables,
        w: &mut Writes,
        okx: &OneSideKIndex,
        qn: Node,
        eq: Edge,
        p: u32,
    ) {
        for &pn in self.dfa.nodes_at(p) {
            for ep in self.dfa.right(pn) {
                let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                    continue;
                };
                let oc = OuterCell {
                    pn,
                    ep: *ep,
                    qn,
                    eq,
                    outer_pair,
                };
                self.oneside_outer_cell3(t, w, okx, &oc);
            }
        }
    }

    /// Find and write one outer cell's best large-3′ one-sided closure. For each
    /// small `n1 ∈ {2,3,4}`, the inner pair's 5′ base is `k = p+1+n1` and its 3′
    /// base `l` ranges over the large-`n2` window `[…, q−1−n1−T]`; the factored
    /// candidate reads the outer half `Am(l5)` (precomputed here over `r3`) and
    /// the inner half `Bm(l3)` (from the index) with the small 5′ flank
    /// enumerated explicitly. The best is re-scored by the exact
    /// [`Self::internal_loop`].
    fn oneside_outer_cell3(&self, t: &Tables, w: &mut Writes, okx: &OneSideKIndex, oc: &OuterCell) {
        let (p, q) = (oc.pn.pos, oc.qn.pos);
        let lambda = t.lambda();
        let kk = (1.0 - lambda) * 100.0;
        let max_ninio = f64::from(self.params.ninio.max_ninio);
        let dskew = self.ninio_skew_threshold() as u32;
        let mm = &self.params.mismatch_interior;
        // Am(l5) = min over r3∈left(qn) [ λ·MM_outer(O,l5,r3) − K·CR(r3) ],
        // keyed by `l5` nucleotide; independent of the inner pair.
        let mut am_by_l5 = [f64::INFINITY; 4];
        for r3 in self.dfa.left(oc.qn) {
            let cr = r3.cai + self.end_contrib(r3.to);
            let r3n = r3.nuc.to_nuc5();
            for l5 in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                let e = lambda * f64::from(mm.get(oc.outer_pair, l5.to_nuc5(), r3n)) - kk * cr;
                let idx = l5 as usize;
                if e < am_by_l5[idx] {
                    am_by_l5[idx] = e;
                }
            }
        }
        if !am_by_l5.iter().any(|a| a.is_finite()) {
            return;
        }
        let edge = -kk * (oc.ep.cai + oc.eq.cai);
        // The mirror of the large-5′ case: the large 3′ flank `[l + 1, q)` is
        // reduced, so a required position inside it floors `l` and truncates the
        // `l`-window; the small 5′ flank `[p + 1, k)` is fixed by `(p, n1)` and
        // so rejects a whole `n1`.
        let l_floor = self.must.min_left(q);
        let mut best: Option<(f64, &OneSideEntry, usize, usize)> = None;
        for n1 in 2..=4u32 {
            let k = p + 1 + n1;
            if !self.must.free_span(p + 1, k) {
                continue;
            }
            // n2 = q−l−1 ≥ n1+T  ⇒  l ≤ q−1−n1−T; size cap n1+n2 ≤ L ⇒
            // l ≥ q+n1−L−1; and l > k.
            let lhi = q.saturating_sub(1 + n1 + dskew);
            let llo = (q + n1)
                .saturating_sub(self.single_max_len as u32 + 1)
                .max(k + 1)
                .max(l_floor);
            if llo > lhi {
                continue;
            }
            let labelled = self.dfa.stop_exit().is_some();
            for entry in okx.window(k, llo, lhi) {
                // Entries share a window across stop lanes; only those both
                // flanks reach can be enclosed.
                if labelled
                    && (!self.dfa.connected(oc.ep.to, entry.ikey.open)
                        || !self.dfa.connected(entry.io, oc.qn))
                {
                    continue;
                }
                let n2 = (q - entry.io.pos) as usize;
                let cm = self.oneside_corner_min3(entry, oc.ep.to, &am_by_l5, kk);
                if !cm.is_finite() {
                    continue;
                }
                let size = f64::from(self.params.interior.values[n1 as usize + n2]);
                let cand = entry.brd + lambda * (size + max_ninio) + cm + edge;
                if best.as_ref().is_none_or(|b| cand < b.0) {
                    best = Some((cand, entry, n1 as usize, n2));
                }
            }
        }
        let Some((_, entry, n1, n2)) = best else {
            return;
        };
        let ctx = SingleCtx {
            inner: PairRef {
                outer: entry.io,
                key: entry.ikey,
            },
            st: entry.st,
            ep: oc.ep,
            pn: oc.pn,
            qn: oc.qn,
            eq: oc.eq,
            outer_pair: oc.outer_pair,
            inner_pair: pair_from_nucleotides(entry.ikey.nuc_open(), entry.ikey.nuc_close),
        };
        LEFTS.with_borrow_mut(|lefts| {
            corner_scores_into(self.dfa, oc.ep.to, entry.ikey.open, lefts);
            if let Some(cand) = self.internal_loop_cand(lambda, &ctx, lefts, n1, n2) {
                w.p.push(cand);
            }
        });
    }

    /// The factored corner minimum of a large-3′ candidate: `min over (l5, l3)
    /// [ Am(l5) + Bm(l3) − K·caiL(l5, l3) ]`, with the small 5′ flank's
    /// `(l5, l3, caiL)` enumerated explicitly (allocation-free) over the outer
    /// opening's right edges and the inner pair's left edges.
    fn oneside_corner_min3(
        &self,
        entry: &OneSideEntry,
        ep_to: Node,
        am_by_l5: &[f64; 4],
        kk: f64,
    ) -> f64 {
        let mut cm = f64::INFINITY;
        for l5 in self.dfa.right(ep_to) {
            let a = am_by_l5[l5.nuc as usize];
            if !a.is_finite() {
                continue;
            }
            for l3 in self.dfa.left(entry.ikey.open) {
                let Some(mid) = self.dfa.broken_codon_cai_s::<S>(l5.to, l3.to) else {
                    continue;
                };
                let bm = entry.bmin[l3.nuc as usize];
                if bm.is_finite() {
                    let cail = l5.cai + mid + l3.cai;
                    cm = cm.min(a + bm - kk * cail);
                }
            }
        }
        cm
    }

    /// Reduce the skewed-large generic closures sharing the outer 3′ side
    /// `(qn, eq)`: enumerate the outer 5′ side `(p, pn, ep)` and, for each
    /// outer cell, find its best skewed inner pair via the `H` index.
    fn skewed_pull(&self, t: &Tables, w: &mut Writes, hx: &HIndex, qn: Node, eq: Edge) {
        // A skewed inner needs k ≥ p+6 and l ≤ q−6 with k < l, so p ≤ q−13.
        let p_hi = qn.pos.saturating_sub(13);
        for p in 0..=p_hi {
            self.skewed_pull_at(t, w, hx, qn, eq, p);
        }
    }

    /// One outer 5′ position `p` of [`Self::skewed_pull`] — the unit of parallel
    /// work for the skewed-large family (independent across `p`).
    ///
    /// Every `H`-index query an outer cell makes depends only on `(p, q)`, so
    /// they are answered once here, on the first outer cell with a finite
    /// outer minimum, and every outer cell at `(p, q)` reads the answers.
    fn skewed_pull_at(&self, t: &Tables, w: &mut Writes, hx: &HIndex, qn: Node, eq: Edge, p: u32) {
        SKEWED.with_borrow_mut(|cands| {
            let mut ready: Option<u8> = None;
            for &pn in self.dfa.nodes_at(p) {
                for ep in self.dfa.right(pn) {
                    let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
                        continue;
                    };
                    let oc = OuterCell {
                        pn,
                        ep: *ep,
                        qn,
                        eq,
                        outer_pair,
                    };
                    let om = self.outer_min(&oc, t.lambda());
                    if !om.is_finite() {
                        continue;
                    }
                    let reach = self.reach_of(ep.to);
                    if ready != Some(reach) {
                        self.skewed_candidates(hx, (pn, ep.to, qn), t.lambda(), cands);
                        ready = Some(reach);
                    }
                    self.skewed_outer_cell(t, w, hx, cands, &oc, om);
                }
            }
        });
    }

    /// The `H`-index answers every outer cell at `(p, q)` scores, in the
    /// order it scores them: over each loop size `s` (both flanks `≥ 5`), the
    /// inner pair lies on the diagonal `Δ = (q − p − 2) − s`, and the index
    /// returns the diagonal's minimum over the eligible `k ∈ [p+6, q−6−Δ]` in
    /// `O(1)`. Each answer comes with the weighted loop-size term its
    /// candidate adds.
    fn skewed_candidates(
        &self,
        hx: &HIndex,
        (pn, open, qn): (Node, Node, Node),
        lambda: f64,
        cands: &mut Vec<SkewCand>,
    ) {
        if self.dfa.stop_exit().is_none() {
            self.skewed_candidates_in(hx, (pn.pos, qn.pos), &[0], lambda, cands);
        } else {
            let classes = self.h_classes(open, qn);
            self.skewed_candidates_in(hx, (pn.pos, qn.pos), &classes, lambda, cands);
        }
    }

    /// [`Self::skewed_candidates`] over the inner-pair `classes` the outer
    /// cell can enclose.
    #[inline(always)]
    fn skewed_candidates_in(
        &self,
        hx: &HIndex,
        (p, q): (u32, u32),
        classes: &[u8],
        lambda: f64,
        cands: &mut Vec<SkewCand>,
    ) {
        cands.clear();
        let max_ninio = f64::from(self.params.ninio.max_ninio);
        let ninio_37 = self.params.ninio.ninio_37;
        let dskew = self.ninio_skew_threshold();
        let span = q as i64 - p as i64 - 2;
        let klo = p + 6;
        // Both flanks of the closure are unpaired — `[p + 1, k)` on the 5′ side
        // and `[l + 1, q)` on the 3′, with `l = k + Δ`. A required position in
        // the 5′ flank caps `k`; one in the 3′ flank floors `l`, hence floors
        // `k` by the same amount shifted along the diagonal. Both are bounds on
        // the `[klo, khi]` window the `H` index is already binary-searched over,
        // so the range-min answers the narrowed query at unchanged cost.
        let k_cap = self.must.max_right(p + 1);
        let l_floor = self.must.min_left(q) as i64;
        for s in 10..=self.single_max_len {
            let delta = span - s as i64;
            let khi = q as i64 - 6 - delta;
            if delta < 1 || khi < klo as i64 {
                continue;
            }
            let khi = (khi as u32).min(k_cap);
            let klo = (klo as i64).max(l_floor - delta);
            if klo > khi as i64 {
                continue;
            }
            let klo = klo as u32;
            let size = f64::from(self.params.interior.values[s]);
            // Skewed band (d ≥ dskew): one range-min over the whole eligible
            // window with the saturated MAX_NINIO. Balanced cells swept in here
            // are dominated by their exact-ninio single-`k` score below, so the
            // window need not exclude them.
            for &class in classes {
                let slot = self.h_slot(delta, class);
                if let Some((h, idx)) = hx.window_min(slot, klo, khi) {
                    let add = lambda * (size + max_ninio);
                    cands.push(SkewCand {
                        h,
                        add,
                        delta: slot,
                        idx,
                    });
                }
            }
            // Balanced band (d < dskew): each `(n1, n2)` with `n1 + n2 = s`
            // pins a single `k = p + 1 + n1`, scored with its true ninio. The
            // window test below is the same narrowed one, so the required-
            // position bounds degenerate here to an `O(1)` admissibility test
            // on an already-pinned `k`.
            for d in 0..dskew {
                if (s + d) % 2 != 0 || (s - d) / 2 < 5 {
                    continue;
                }
                let nin = f64::from(d as i32 * ninio_37);
                for n1 in [(s + d) / 2, (s - d) / 2] {
                    let k = p + 1 + n1 as u32;
                    if k >= klo && k <= khi {
                        for &class in classes {
                            let slot = self.h_slot(delta, class);
                            if let Some((h, idx)) = hx.window_min(slot, k, k) {
                                let add = lambda * (size + nin);
                                cands.push(SkewCand {
                                    h,
                                    add,
                                    delta: slot,
                                    idx,
                                });
                            }
                        }
                    }
                    if d == 0 {
                        break;
                    }
                }
            }
        }
    }

    /// Find and write one outer cell's best skewed-large closure among
    /// `cands` ([`Self::skewed_candidates`] at its `(p, q)`), `om` its outer
    /// minimum. The winner is re-scored by the exact
    /// [`Self::internal_loop_cand`], so a balanced cell swept in under the
    /// constant `MAX_NINIO` upper bound is written with its true (smaller)
    /// energy — and the optimum is preserved.
    fn skewed_outer_cell(
        &self,
        t: &Tables,
        w: &mut Writes,
        hx: &HIndex,
        cands: &[SkewCand],
        oc: &OuterCell,
        om: f64,
    ) {
        let (p, q) = (oc.pn.pos, oc.qn.pos);
        let lambda = t.lambda();
        let mut best: Option<(f64, &SkewCand)> = None;
        for c in cands {
            let cand = om + c.h + c.add;
            if best.as_ref().is_none_or(|b| cand < b.0) {
                best = Some((cand, c));
            }
        }
        let Some((_, c)) = best else {
            return;
        };
        let entry = hx.entry(c.delta, c.idx);
        let n1 = (entry.k - p - 1) as usize;
        let n2 = (q - entry.io.pos) as usize;
        let ctx = SingleCtx {
            inner: PairRef {
                outer: entry.io,
                key: entry.ikey,
            },
            st: entry.st,
            ep: oc.ep,
            pn: oc.pn,
            qn: oc.qn,
            eq: oc.eq,
            outer_pair: oc.outer_pair,
            inner_pair: pair_from_nucleotides(entry.ikey.nuc_open(), entry.ikey.nuc_close),
        };
        LEFTS.with_borrow_mut(|lefts| {
            corner_scores_into(self.dfa, oc.ep.to, entry.ikey.open, lefts);
            if let Some(cand) = self.internal_loop_cand(lambda, &ctx, lefts, n1, n2) {
                w.p.push(cand);
            }
        });
    }

    /// Process `M2[o]`: collapse to `M` (`M = M2`) and wrap into a
    /// multi-loop in progress (`Multi`) by enclosing the `M2` core with a
    /// closing pair and unpaired flanks.
    fn m2_beam(&self, t: &mut Tables, o: Node) {
        let states = t.m2_states(o);
        if states.is_empty() {
            return;
        }
        let end = self.core_end(o);
        let mut far = FxHashMap::default();
        for (inner, st) in states {
            t.update_m(
                o,
                inner,
                State::new(
                    st.mfe,
                    st.cai,
                    Back::MFromM2 {
                        m2: SpanRef { outer: o, inner },
                    },
                ),
            );
            self.m2_to_multi(t, &end, inner, st, &mut far);
        }
        self.far_multi_open(t, &end, far);
    }

    /// The [`CoreEnd`] of the cores ending at `core_outer`.
    fn core_end(&self, core_outer: Node) -> CoreEnd {
        let boundary = self.dfa.next_codon_boundary(core_outer.pos);
        // The 3′ unpaired stretch is `[core_outer.pos, q)`, so a required
        // position inside it caps the frontier scan.
        let short_top = boundary.min(self.must.max_right(core_outer.pos).saturating_add(1));
        let mut closes = Vec::new();
        for q in core_outer.pos..short_top {
            for &qn in self.dfa.nodes_at(q) {
                if let Some(right_cai) = self.dfa.broken_codon_cai_s::<S>(core_outer, qn) {
                    closes.push((qn, right_cai));
                }
            }
        }
        // The shared opening carries the 3′ unpaired stretch from the core's
        // right boundary to whatever frontier closes it, so it is admissible
        // only if its own `C` stretch `[core_outer.pos, boundary)` is free. The
        // remainder of that stretch is the frontier's business, and the pool
        // eviction at each required position handles it.
        let c_outer = if self.must.free_span(core_outer.pos, boundary) {
            self.dfa.span_head_cai(core_outer)
        } else {
            None
        };
        CoreEnd {
            outer: core_outer,
            closes,
            boundary,
            c_outer,
        }
    }

    /// `Multi = (closing pair) [left flank] M2 [right flank]`. Both flanks
    /// are unbounded, as in the energy model (multi-loop unpaired is
    /// energy-free); the right flank is shared across closing frontiers via
    /// [`Self::seed_multi_open`] rather than re-scanned per core.
    ///
    /// Left flanks shorter than `SINGLE_MAX_LEN` are seeded here, per core.
    /// A longer one would make the work per core grow with the sequence, so
    /// each core instead offers its flank of exactly `SINGLE_MAX_LEN` to `far`,
    /// which [`Self::far_multi_open`] extends leftward once for all the cores
    /// sharing `core_outer`.
    fn m2_to_multi(
        &self,
        t: &mut Tables,
        end: &CoreEnd,
        core_inner: Node,
        st: State,
        far: &mut FxHashMap<MultiOpenKey, FarOpening>,
    ) {
        // The 5′ flank `[p + 1, core_inner.pos)` is unpaired, so a required
        // position inside it floors `p` — the same truncation the single-branch
        // 5′ flank applies.
        let p_lo = self.must.min_left(core_inner.pos);
        let near = core_inner
            .pos
            .saturating_sub(self.single_max_len as u32)
            .max(p_lo);
        let core = SpanRef {
            outer: end.outer,
            inner: core_inner,
        };
        for p in near..core_inner.pos {
            for &pn in self.dfa.nodes_at(p) {
                for ep in self.dfa.right(pn) {
                    self.seed_multi_open(t, end, core, st, pn, *ep);
                }
            }
        }
        self.offer_far(end, core_inner, st, t.lambda(), far);
    }

    /// Offer one core's left flank of exactly `SINGLE_MAX_LEN` to `far`, for
    /// [`Self::far_multi_open`] to extend leftward; nothing when a required
    /// position floors the flank first.
    fn offer_far(
        &self,
        end: &CoreEnd,
        core_inner: Node,
        st: State,
        lambda: f64,
        far: &mut FxHashMap<MultiOpenKey, FarOpening>,
    ) {
        let p_lo = self.must.min_left(core_inner.pos);
        let near = core_inner
            .pos
            .saturating_sub(self.single_max_len as u32)
            .max(p_lo);
        if near == p_lo {
            return;
        }
        let core = SpanRef {
            outer: end.outer,
            inner: core_inner,
        };
        for &pn in self.dfa.nodes_at(near - 1) {
            for ep in self.dfa.right(pn) {
                let Some(flank) = self.dfa.broken_codon_cai_s::<S>(ep.to, core_inner) else {
                    continue;
                };
                let cand = FarOpening {
                    core,
                    st,
                    ep: *ep,
                    cai: st.cai + ep.cai + flank,
                };
                cand.merge_into(far, pn, lambda);
            }
        }
    }

    /// Seed every multi-loop opening whose left flank is `SINGLE_MAX_LEN` or
    /// longer, for the cores sharing one `core_outer` (the keys of `far`).
    ///
    /// Every use an opening is put to — its `multi_open` entry and its
    /// short-flank closures — depends on its core only through the core's
    /// state, `core_outer` and the flank, and a left flank grows by one
    /// position at a fixed price: the unpaired base costs no energy and the
    /// new opening edge adds its own adaptation. So for each opening only the
    /// best core matters, and the best core one position further left is the
    /// best over the openings it extends. Sweeping right to left, each opening
    /// is seeded once (through [`Self::seed_multi_open`], which re-reads the
    /// flank's exact adaptation) and then offered one position left — never
    /// across a position required to pair, which the longer flank would leave
    /// unpaired.
    fn far_multi_open(
        &self,
        t: &mut Tables,
        end: &CoreEnd,
        mut far: FxHashMap<MultiOpenKey, FarOpening>,
    ) {
        let Some(top) = far.keys().map(|k| k.open.pos).max() else {
            return;
        };
        let lambda = t.lambda();
        for p in (0..=top).rev() {
            for &pn in self.dfa.nodes_at(p) {
                for ep in self.dfa.right(pn) {
                    let key = MultiOpenKey {
                        open: pn,
                        nuc_open: ep.nuc,
                        lane_open: ep.to.lane,
                        reach: 0,
                    };
                    let Some(f) = far.remove(&key) else {
                        continue;
                    };
                    self.seed_multi_open(t, end, f.core, f.st, pn, f.ep);
                    if p == 0 || self.must.is_must(p) {
                        continue;
                    }
                    for l in self.dfa.left(pn) {
                        let cand = FarOpening {
                            ep: Edge { to: pn, ..*l },
                            cai: f.cai + l.cai,
                            ..f
                        };
                        cand.merge_into(&mut far, l.to, lambda);
                    }
                }
            }
        }
    }

    /// Register one `(M2 core, left flank)` multi-loop opening. The right
    /// flank's broken-codon CAI separates as `C(core_outer) + D(frontier)`
    /// for frontiers at/after `core_outer`'s codon boundary, so the opening —
    /// with `C(core_outer)` folded in — is shared in `multi_open` and closed
    /// against every such frontier by adding `D(frontier)`. Frontiers inside
    /// `core_outer`'s own codon do not separate and are closed directly here.
    fn seed_multi_open(
        &self,
        t: &mut Tables,
        end: &CoreEnd,
        core: SpanRef,
        st: State,
        pn: Node,
        ep: Edge,
    ) {
        let Some(flank_cai) = self.dfa.broken_codon_cai_s::<S>(ep.to, core.inner) else {
            return;
        };
        let seed = MultiSeed {
            core,
            st,
            pn,
            ep,
            left_cai: ep.cai + flank_cai,
        };
        self.seed_short_flanks(t, &seed, end);
        let Some(c_outer) = end.c_outer else {
            return;
        };
        let key = MultiOpenKey {
            open: pn,
            nuc_open: ep.nuc,
            lane_open: ep.to.lane,
            reach: self.reach_of(core.outer),
        };
        let open_state = State::new(
            st.mfe,
            st.cai + seed.left_cai + c_outer,
            Back::MultiWrap { core },
        );
        if end.boundary == core.outer.pos {
            t.multi_open_insert(key, open_state);
        } else {
            t.multi_open_defer(end.boundary, key, open_state);
        }
    }

    /// Close the short (same-codon-as-`core_outer`) right flanks directly,
    /// using the exact broken-codon CAI — these frontiers do not admit the
    /// `C + D` separation the shared `multi_open` path relies on.
    fn seed_short_flanks(&self, t: &mut Tables, seed: &MultiSeed, end: &CoreEnd) {
        let core = seed.core;
        for &(qn, right_cai) in &end.closes {
            for eq in self.dfa.right(qn) {
                if canonical(seed.ep.nuc, eq.nuc).is_none() {
                    continue;
                }
                let cai = seed.st.cai + seed.left_cai + right_cai + eq.cai;
                let key = PairKey {
                    open: seed.pn,
                    close: qn,
                    opening: seed.ep.opening,
                    nuc_close: eq.nuc,
                };
                t.update_multi(
                    eq.to,
                    key,
                    State::new(seed.st.mfe, cai, Back::MultiWrap { core }),
                );
            }
        }
    }

    /// Close every active multi-loop opening against frontier node `qn`,
    /// paying the frontier's broken-codon CAI `D(qn)` plus the closing edge.
    /// Each `(opening, qn)` forms a `Multi` cell identical to the one the
    /// per-core scan produced, so `multi_close` and the traceback are
    /// unchanged.
    fn multi_open_close(&self, t: &mut Tables, qn: Node, openings: &[(MultiOpenKey, State)]) {
        let Some(d_qn) = self.dfa.span_tail_cai(qn) else {
            return;
        };
        for &(key, state) in openings {
            if !self.dfa.reaches(key.reach, qn) {
                continue;
            }
            for eq in self.dfa.right(qn) {
                if canonical(key.nuc_open, eq.nuc).is_none() {
                    continue;
                }
                let cai = state.cai + d_qn + eq.cai;
                let pk = PairKey {
                    open: key.open,
                    close: qn,
                    opening: crate::state::Opening::new(key.nuc_open, key.lane_open),
                    nuc_close: eq.nuc,
                };
                t.update_multi(eq.to, pk, State::new(state.mfe, cai, state.back));
            }
        }
    }

    /// `M` extends by one unpaired nucleotide on the right (free at
    /// `dangles=0`, base_37 = 0). Skipped outright when that nucleotide is
    /// required to pair.
    fn m_extend(&self, t: &mut Tables, jn: Node) {
        if self.must.is_must(jn.pos) {
            return;
        }
        for (inner, st) in t.m_states(jn) {
            for e in self.dfa.right(jn) {
                let back = Back::MUnpaired {
                    prev: SpanRef { outer: jn, inner },
                    nuc: e.nuc,
                };
                t.update_m(e.to, inner, State::new(st.mfe, st.cai + e.cai, back));
            }
        }
    }

    /// `C` extends by one unpaired nucleotide on the right. Skipped outright
    /// when that nucleotide is required to pair.
    fn c_extend(&self, t: &mut Tables, jn: Node) {
        if self.must.is_must(jn.pos) {
            return;
        }
        let Some(st) = t.c[self.dfa.ix(jn)] else {
            return;
        };
        for e in self.dfa.right(jn) {
            let back = Back::ExteriorUnpaired {
                prev: jn,
                nuc: e.nuc,
            };
            t.update_c(e.to, State::new(st.mfe, st.cai + e.cai, back));
        }
    }

    /// Per-branch multi-loop initiation: `intern_37` plus the per-stem
    /// terminal-AU correction.
    fn m_stem_cost(&self, pair: BasePair) -> Energy {
        Energy(self.params.ml.intern_37) + self.terminal_au(pair)
    }

    /// Multi-loop closing cost on the closing pair: `closing_37` plus the
    /// closing pair's stem contribution.
    fn multi_close_cost(&self, closing: BasePair) -> Energy {
        Energy(self.params.ml.closing_37) + self.m_stem_cost(closing)
    }

    /// Per-stem terminal-AU correction: `terminal_au_37` for AU / GU
    /// closures, zero for GC / CG.
    fn terminal_au(&self, pair: BasePair) -> Energy {
        if pair.applies_terminal_au() {
            Energy(self.params.misc.terminal_au_37)
        } else {
            Energy::ZERO
        }
    }
}

/// Stack or bulge closure energy, dispatching on the unpaired-flank
/// sizes: `(0, 0)` is a stack, one zero a bulge. Neither reads the
/// interior nucleotides, so the flank lengths alone determine the
/// energy. Internal loops (`n1 > 0 && n2 > 0`) are scored separately via
/// [`crate::interior::score_internal_from`], which reads their terminal
/// mismatches.
fn stack_or_bulge_energy(
    params: &Params,
    outer: BasePair,
    inner: BasePair,
    n1: usize,
    n2: usize,
) -> Energy {
    if n1 == 0 && n2 == 0 {
        return stack_energy(params, &StackPair { outer, inner });
    }
    bulge_energy(
        params,
        &BulgeLoop {
            outer,
            inner,
            size: n1 + n2,
        },
    )
}

/// Whether the `(n1, n2)` flank shape routes to the **generic**
/// internal-loop branch ([`rna_mfe::loops::internal_energy_scalar`]'s
/// fall-through arm): both unpaired strands `≥ 2` and the shape is not one
/// of the tabulated small cases `(2, 2)` / `(2, 3)` / `(3, 2)`. Only the
/// generic branch is two-dimensional in the flank lengths (`O(L²)`), so it
/// is the one the `Lz` pull reformulates; every other shape stays on the
/// `O(L)` push path.
fn is_generic(n1: usize, n2: usize) -> bool {
    n1 >= 2 && n2 >= 2 && !matches!((n1, n2), (2, 2) | (2, 3) | (3, 2))
}

/// Canonical Watson–Crick / wobble pair indicator; `None` for any
/// non-pairing combination.
fn canonical(a: Nucleotide, b: Nucleotide) -> Option<BasePair> {
    match pair_from_nucleotides(a, b) {
        BasePair::NN => None,
        canonical => Some(canonical),
    }
}
