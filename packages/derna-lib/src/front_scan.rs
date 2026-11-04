//! Exact un-reduced front-valued CKY over the codon DFA — the opt-in
//! full-Pareto-front designer.
//!
//! Where [`crate::scan`] keeps one BRD-optimal [`crate::state::State`] per CKY
//! cell (an argmin under a fixed λ), this fold keeps a [`Front`]: the
//! dominance-reduced staircase of *every* non-dominated `(mfe, cai)` trade-off
//! reaching that cell — including the off-hull points a linear λ-sweep provably
//! cannot reach. The recurrence structure is identical to the scalar scan (the
//! same six node-keyed tables and their merges); only the cell payload and its
//! two operators change, per the front lift:
//!
//! - scalar `min`-by-BRD (`⊕`) becomes [`Front::merge`] (dominance-merge);
//! - scalar `+` (`⊗`) becomes [`Front::compose`] (Minkowski-sum + eliminate)
//!   or its uniform-shift special case [`Front::shift_map`].
//!
//! The fold is **un-reduced** (naive internal-loop enumeration, naive `O(n³)`
//! multi-loop assembly) — the reductions are a scaling concern above the
//! tractability crossover, not a correctness one below it — and **opt-in**: it
//! is a separate entry point ([`crate::design_front`]), never the scalar path,
//! so a default (non-front) design is byte-identical to today.
//!
//! Each front point carries its own provenance handle, so recovering one Pareto
//! point walks that point's own derivation — a per-front-entry traceback.
//! To keep provenance memory proportional to the *stored* front rather than the
//! `O(n³·k̄)` shift/compose volume, a cell accumulates points holding [`Prov`]
//! **values** (referencing already-committed parents by arena index), and those
//! values are pushed into the [`Arena`] **only when the cell is consumed** —
//! i.e. only for the points that survived every dominance-merge and go on to
//! parent a larger structure. A shifted point that a later merge drops is freed
//! with its transient front and never reaches the arena.

use rna_mfe::loops::{
    bulge_energy, hairpin_energy, hairpin_energy_scalar, internal_energy_scalar,
    pair_from_nucleotides, stack_energy, BulgeLoop, HairpinLoop, InternalLoopScalar, StackPair,
};
use rna_mfe::params::BasePair;
use rna_mfe::{Energy, Nucleotide, Params};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::bounds::{retain_within, ExactCredit, Slack, Windows};
use crate::box_census::{BoxCuts, BoxTally, Lows, Survivors};
use crate::cost_to_go::CellRef;
use crate::dfa::{Dfa, Edge, Node};
use crate::fan_detail::{self, FanDetail, FanSite, WorkerTimes};
use crate::fold_pool::Pool;
use crate::front::{Front, Point, Prune, Screen};
use crate::front_probe::{CellTable, ComposeSite, FrontProbe, RetainedCounts};
use crate::hash::FxHashMap;
use crate::region::{PairConstraint, Verdict};
use crate::serial_detail::{self, ComposeCount, SerialDetail};
use crate::state::{MultiOpenKey, PairKey};

/// No-sharp-turn rule: a hairpin has `>= 3` unpaired bases.
const MIN_HAIRPIN_LOOP: u32 = 3;
/// Vienna `MAXLOOP`: the largest total unpaired count in a single-branch loop.
/// **This bound is fixed while [`crate::scan`]'s is not.** That module's
/// equivalent is [`crate::ScanConfig::single_max_len`], settable per run
/// (default [`crate::DEFAULT_SINGLE_MAX_LEN`]); this one is a constant because
/// the front-valued entry points take no `ScanConfig` and so cannot be asked
/// for another value. Anything that later
/// gives `design_front*` a `ScanConfig` has to thread the setting here too, or
/// one run will fold two energy models — the scalar scan's and this one's.
const SINGLE_MAX_LEN: u32 = 30;

/// Items a fat-phase worker claims per atomic step. Each claimed chunk gets its
/// own [`FrontWrites`] tagged with the chunk index, so the fold can replay the
/// partials in the exact serial production order regardless of steal order.
/// Large enough to amortize the `fetch_add`, small enough to balance the tail.
const FAN_OUT_CHUNK: usize = 8;
/// Below this many items a fat phase runs serially: too little work to repay
/// handing it to the pooled workers and folding their partials. The workers
/// live for the whole fold, so a fan-out costs a fraction of a millisecond and
/// the floor is low.
const FAN_OUT_MIN: usize = 8;
/// Closed pairs per window of a position's first pass: the most whose
/// composed bifurcations are held at once.
const BIFURCATION_WINDOW: usize = 128;
/// Below this many arrivals the shared openings are merged serially.
const OPEN_SHARD_MIN: usize = 64;

/// An index into the [`Prov`] arena — the committed per-front-entry back-pointer.
type ProvIdx = u32;
/// A front whose points hold un-committed [`Prov`] **values** — how a cell holds
/// its points while it is still accumulating (before it is consumed).
type Acc = Front<Prov>;
/// A front whose points hold committed arena indices — how a cell is read once
/// it is consumed, so children can reference its points as parents.
type Com = Front<ProvIdx>;
/// A `P` candidate bound for a merge: `(outer node, pair key, front)`.
type PCandidate = (Node, PairKey, Acc);

/// One recovered Pareto point: its objectives plus the coding sequence and
/// dot-bracket structure its provenance spells.
pub struct RecoveredPoint {
    /// Accumulated free energy (integer dekacalories; lower is better).
    pub mfe: Energy,
    /// Accumulated log-CAI, `Σ log(g / g_max)` (higher is better).
    pub cai_log: f64,
    /// Coding sequence as ASCII `A/C/G/U`.
    pub sequence: String,
    /// Dot-bracket secondary structure.
    pub structure: String,
}

/// The front of `(mfe, cai)` trade-offs for the protein encoded by `dfa`, each
/// point with a recovered `(sequence, structure)`. The points are the staircase
/// at the exterior end cell, `mfe`-ascending / `cai`-ascending.
///
/// `prune` selects the fold's fidelity: [`Prune::None`] recovers the exact
/// complete front (every non-dominated trade-off, validated byte-exact against
/// the brute-force oracle where it fits); a lossy [`Prune`] caps each cell's
/// front so the fold scales past the exact crossover, at the cost of the bound
/// that mode carries.
/// An optional read-only `probe` is written to and never read back, so the
/// recovered front is the same front — point for point and sequence for
/// sequence — whether or not one is supplied. What comes back with it is what
/// each of its drop rules would have removed had the fold been applying them.
///
/// `credit`, when given, adds each cell's exact cost to go to the window's drop
/// ([`crate::bounds::retain_within`]); it can empty cells, which the window
/// alone never does. `None` is the window alone, byte for byte.
pub fn fold_probed(
    params: &Params,
    dfa: &Dfa,
    prune: Prune,
    windows: Windows,
    credit: Option<&ExactCredit>,
    threads: usize,
    probe: Option<FrontProbe>,
) -> (Vec<RecoveredPoint>, Option<FrontProbe>, DropTally) {
    let (points, probe, dropped, _, _) = fold_inner(
        params,
        dfa,
        prune,
        windows,
        credit,
        None,
        threads,
        probe,
        None,
        &PairConstraint::inert(),
    );
    (points, probe, dropped)
}

/// [`fold_probed`] with no probe, returning where the fold's time went, with
/// `region` applied where a pair closes: a pair it forbids never enters a `P`
/// cell, so no recovered structure contains it.
pub(crate) fn fold_clocked(
    params: &Params,
    dfa: &Dfa,
    prune: Prune,
    windows: Windows,
    credit: Option<&ExactCredit>,
    threads: usize,
    region: &PairConstraint,
) -> (Vec<RecoveredPoint>, DropTally, FoldClock) {
    let (points, _, dropped, _, clock) = fold_inner(
        params, dfa, prune, windows, credit, None, threads, None, None, region,
    );
    (points, dropped, clock)
}

/// The exact bounded fold with `cuts` applied at every merge beside the
/// energy drop: the fold of the box they bound, not of its whole ceiling.
/// Every point the cuts keep that also meets the ceiling and the floor is on
/// the box's front; the caller applies those two to the output. `region` is
/// applied as in [`fold_clocked`].
pub(crate) fn fold_boxed(
    params: &Params,
    dfa: &Dfa,
    windows: Windows,
    credit: Option<&ExactCredit>,
    cuts: &BoxCuts,
    threads: usize,
    region: &PairConstraint,
) -> (Vec<RecoveredPoint>, DropTally, FoldClock) {
    let (points, _, dropped, _, clock) = fold_inner(
        params,
        dfa,
        Prune::None,
        windows,
        credit,
        Some(cuts),
        threads,
        None,
        None,
        region,
    );
    (points, dropped, clock)
}

/// The exact bounded fold with a [`BoxCuts`] attached: every finalized cell
/// is tested against each of its cuts and counted, and nothing is removed, so
/// the recovered front is the one [`fold_probed`] recovers without it.
pub(crate) fn fold_census(
    params: &Params,
    dfa: &Dfa,
    windows: Windows,
    credit: Option<&ExactCredit>,
    threads: usize,
    census: &BoxCuts,
) -> (Vec<RecoveredPoint>, DropTally, BoxTally) {
    let (points, _, dropped, tally, _) = fold_inner(
        params,
        dfa,
        Prune::None,
        windows,
        credit,
        None,
        threads,
        None,
        Some(census),
        &PairConstraint::inert(),
    );
    (points, dropped, tally.expect("a census was attached"))
}

type Folded = (
    Vec<RecoveredPoint>,
    Option<FrontProbe>,
    DropTally,
    Option<BoxTally>,
    FoldClock,
);

#[allow(clippy::too_many_arguments)]
fn fold_inner(
    params: &Params,
    dfa: &Dfa,
    prune: Prune,
    windows: Windows,
    credit: Option<&ExactCredit>,
    cuts: Option<&BoxCuts>,
    threads: usize,
    probe: Option<FrontProbe>,
    census: Option<&BoxCuts>,
    region: &PairConstraint,
) -> Folded {
    if threads > 1 {
        Pool::scoped(threads, |pool| {
            fold_on(
                params,
                dfa,
                prune,
                windows,
                credit,
                cuts,
                threads,
                probe,
                census,
                region,
                Some(pool),
            )
        })
    } else {
        fold_on(
            params, dfa, prune, windows, credit, cuts, threads, probe, census, region, None,
        )
    }
}

/// [`fold_inner`] on the given workers.
#[allow(clippy::too_many_arguments)]
fn fold_on(
    params: &Params,
    dfa: &Dfa,
    prune: Prune,
    windows: Windows,
    credit: Option<&ExactCredit>,
    cuts: Option<&BoxCuts>,
    threads: usize,
    probe: Option<FrontProbe>,
    census: Option<&BoxCuts>,
    region: &PairConstraint,
    pool: Option<&Pool>,
) -> Folded {
    let mut dp = Dp::fill(
        params, dfa, prune, windows, credit, cuts, threads, probe, census, region, pool,
    );
    dp.observe_shared_openings();
    // The final exterior cell is finalized here rather than by the per-position
    // extension, which stops one short of it, so it is observed here too.
    dp.observe_final_exterior();
    // Commit the final exterior front so the traceback walks committed arena
    // indices uniformly with the rest of the DAG.
    let t = Instant::now();
    let end = commit(&mut dp.arena, &dp.c[dfa.ix(dfa.end())]);
    let probe = dp.probe.map(|state| state.probe);
    let tally = dp.census.map(|state| state.tally);
    let dropped = DropTally {
        by_table: dp.dropped,
        emptied: dp.emptied,
        boxed: dp.boxed,
        skipped: dp.skipped,
    };
    let speller = Speller::new(dfa, &dp.arena);
    // A stopped fold's exterior holds only the designs it reached.
    let points = if dp.clock.stopped {
        Vec::new()
    } else {
        end.iter().map(|p| speller.recover(p)).collect()
    };
    let mut clock = dp.clock;
    clock.stored = dp.arena.len() as u64;
    clock.traceback_s = t.elapsed().as_secs_f64();
    if dp.detail {
        clock.serial.fanned = dp.fanned.load(Ordering::Relaxed) as u64;
    }
    (points, probe, dropped, tally, clock)
}

/// Where one front fold's time went, phase by phase, and what the shared
/// multi-loop openings held and produced.
///
/// A phase's seconds are the time charged to it, not necessarily the time its
/// own work took: the fat phases include applying their writes, and so the
/// merges (and any box cuts) into the tables they write. The counts are there
/// to say where the work is.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FoldClock {
    /// Seeding every hairpin-closed pair.
    pub hairpins_s: f64,
    /// Merging deferred openings into the shared table at their boundary.
    pub open_activate_s: f64,
    /// Closing multi-loops into pairs.
    pub multi_close_s: f64,
    /// A closed pair's exterior attach, branch open and bifurcation (serial).
    pub pair_serial_s: f64,
    /// A closed pair's stacks, bulges and internal loops (fanned out).
    pub pair_single_s: f64,
    /// Committing each two-branch core and collapsing it to one branch.
    pub m2_serial_s: f64,
    /// Seeding the shared openings from each core, their short flanks
    /// included (fanned out).
    pub open_seed_s: f64,
    /// Closing every held opening against each frontier into `Multi`
    /// (fanned out), the merges into `Multi` included.
    pub open_close_s: f64,
    /// Finalizing and extending the multi-branch cells.
    pub m_extend_s: f64,
    /// Finalizing and extending the exterior cells.
    pub c_extend_s: f64,
    /// Merging into the shared opening table; charged inside
    /// `open_activate_s` and `open_seed_s` already, reported apart.
    pub open_merge_s: f64,
    /// Committing the final exterior cell and spelling its points.
    pub traceback_s: f64,
    /// Points arriving at the shared opening table.
    pub open_arrivals: u64,
    /// Openings arriving at the shared table as recipes, and recipes built.
    pub open_recipes: u64,
    pub open_built: u64,
    /// Opening keys held, summed over the positions they are closed at.
    pub open_keys_held: u64,
    /// Opening points held, summed over the positions they are closed at.
    pub open_points_held: u64,
    /// Points the closes produce: `Σ |opening| × closing edges` per position.
    pub open_close_points: u64,
    /// Of those, the points the box's cuts removed arriving at `Multi`.
    pub open_close_boxed: u64,
    /// Energy-drop removals at `Multi` during the closes' merges.
    pub open_close_dropped: u64,
    /// Of `open_close_boxed`, the points never shifted, because a cut removed
    /// their opening's lowest-valued point at the closed cell.
    pub open_close_screened: u64,
    /// Hairpin-closed points seeded into `P`, and of those the points the
    /// box's cuts removed.
    pub hairpin_points: u64,
    pub hairpin_boxed: u64,
    /// Hairpins never enumerated because a cut removes every interior.
    pub hairpin_screened: u64,
    /// Inner-front points stacks, bulges and internal loops never closed,
    /// because a cut removes every point they would make.
    pub single_screened: u64,
    /// Core points the short-flank closes of the openings never wrote,
    /// because a cut removes them all at the closed cell.
    pub short_screened: u64,
    /// `pair_serial_s` split, and the bifurcation's points counted, where the
    /// split was enabled; see [`SerialDetail`].
    pub serial: SerialDetail,
    /// Inside the fanned-out phases, where the tally was enabled; see
    /// [`FanDetail`].
    pub fan: FanDetail,
    /// Points the fold committed to its arena.
    pub stored: u64,
    /// Whether the box's stop ([`BoxCuts::with_stop`]) ended the fold before
    /// its last position; such a fold returns no point.
    pub stopped: bool,
}

impl FoldClock {
    /// Add another fold's phase seconds and opening counts to this one, so
    /// a solve run as several folds (energy slices) reports them together.
    /// The split and fan-out details stay this fold's own.
    pub fn absorb(&mut self, other: &FoldClock) {
        self.hairpins_s += other.hairpins_s;
        self.stored += other.stored;
        self.open_activate_s += other.open_activate_s;
        self.multi_close_s += other.multi_close_s;
        self.pair_serial_s += other.pair_serial_s;
        self.pair_single_s += other.pair_single_s;
        self.m2_serial_s += other.m2_serial_s;
        self.open_seed_s += other.open_seed_s;
        self.open_close_s += other.open_close_s;
        self.m_extend_s += other.m_extend_s;
        self.c_extend_s += other.c_extend_s;
        self.open_merge_s += other.open_merge_s;
        self.traceback_s += other.traceback_s;
        self.open_arrivals += other.open_arrivals;
        self.open_recipes += other.open_recipes;
        self.open_built += other.open_built;
        self.open_keys_held += other.open_keys_held;
        self.open_points_held += other.open_points_held;
        self.open_close_points += other.open_close_points;
        self.open_close_boxed += other.open_close_boxed;
        self.open_close_dropped += other.open_close_dropped;
        self.open_close_screened += other.open_close_screened;
        self.hairpin_points += other.hairpin_points;
        self.hairpin_boxed += other.hairpin_boxed;
        self.hairpin_screened += other.hairpin_screened;
        self.single_screened += other.single_screened;
        self.short_screened += other.short_screened;
    }

    /// Seconds charged to the phases, the traceback included.
    pub fn total_s(&self) -> f64 {
        self.hairpins_s
            + self.open_activate_s
            + self.multi_close_s
            + self.pair_serial_s
            + self.pair_single_s
            + self.m2_serial_s
            + self.open_seed_s
            + self.open_close_s
            + self.m_extend_s
            + self.c_extend_s
            + self.traceback_s
    }
}

/// What the feasibility drop discarded over a fold, split by table.
///
/// Split rather than totalled because the tables are not interchangeable: the
/// multi-branch ones are where the front's cost is generated and where a
/// completion estimate built by decomposing the outside would be inadmissible,
/// so "the drop fired" is only meaningful once it says *where*. A total alone
/// can be carried entirely by the exterior table, which holds a few hundred
/// cells out of hundreds of thousands.
///
/// **These are counts of removals, not a measure of how hard a window bit.**
/// The two are not the same quantity and do not even move together: a narrower
/// window upstream removes points that would otherwise have reached a
/// downstream cell, so it leaves *fewer* there to remove and the downstream
/// count falls. A tighter window can therefore lower a table's tally, and a
/// table whose own window did not change can still see its tally move. Read
/// these to answer "was this table's drop exercised at all"; anything
/// comparative belongs on the recovered front.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DropTally {
    by_table: [u64; 6],
    emptied: [u64; 6],
    boxed: [u64; 6],
    skipped: [u64; 6],
}

impl DropTally {
    /// Merges at `table` that left the cell with no point at all, which only
    /// the exact credit can do. An emptied cell is removed and never composed,
    /// so this — not the point count — is the tally of the lever the window
    /// cannot pull. A cell merged into again after being emptied is emptied
    /// again and counted again, so this counts events, not distinct cells.
    pub fn emptied_at(&self, table: CellTable) -> u64 {
        self.emptied[table.index()]
    }

    /// Points a box's cuts discarded at `table`, beside the energy drop's
    /// [`Self::at`]. Zero unless a box was applied. Counted where they arrive,
    /// so a cut upstream leaves less to count downstream.
    pub fn boxed_at(&self, table: CellTable) -> u64 {
        self.boxed[table.index()]
    }

    /// Products a compose into `table` never generated, because its merge
    /// would have discarded them: above the cell's cutoff, or excluded by a
    /// cut. A unit apart from [`Self::at`] and [`Self::boxed_at`], which count
    /// points after the compose's elimination; a skipped product is never
    /// counted there too, so those two read lower where this is non-zero.
    pub fn skipped_at(&self, table: CellTable) -> u64 {
        self.skipped[table.index()]
    }

    /// [`Self::boxed_at`] summed over every table.
    pub fn boxed_total(&self) -> u64 {
        self.boxed.iter().sum()
    }

    /// [`Self::emptied_at`] summed over every table.
    pub fn emptied_total(&self) -> u64 {
        self.emptied.iter().sum()
    }

    /// Derivations discarded at `table`.
    pub fn at(&self, table: CellTable) -> u64 {
        self.by_table[table.index()]
    }

    /// Derivations discarded across every table.
    pub fn total(&self) -> u64 {
        self.by_table.iter().sum()
    }

    /// Derivations discarded at the four tables inside a multi-loop, where a
    /// completion estimate that factorizes the outside has nothing to
    /// factorize. A drop that never reaches these has only been exercised where
    /// even an inadmissible estimate would have been sound.
    pub fn multiloop_interior(&self) -> u64 {
        self.at(CellTable::Multi)
            + self.at(CellTable::MultiOpen)
            + self.at(CellTable::Branch)
            + self.at(CellTable::Branch2)
    }
}

// ===========================================================================
// Provenance: the per-front-entry derivation DAG.
// ===========================================================================

/// The bracket + boundary-nucleotide spelling of one closed pair. The interior
/// is spelled by the enclosing [`Prov`] variant; this records only the pair's
/// own two slots.
#[derive(Debug, Clone, Copy)]
struct PairSpell {
    /// 5′ paired position.
    a: u32,
    /// 3′ paired position.
    b: u32,
    /// Nucleotide at the 5′ slot.
    na: Nucleotide,
    /// Nucleotide at the 3′ slot.
    nb: Nucleotide,
}

impl PairSpell {
    fn of(key: &PairKey) -> Self {
        Self {
            a: key.open.pos,
            b: key.close.pos,
            na: key.nuc_open(),
            nb: key.nuc_close,
        }
    }
}

/// A best-CAI unpaired span between two DFA boundary nodes — the energy-free
/// interior of a stack/bulge flank or a multi-loop flank, respelled by the
/// best-CAI DFA path (Pareto-complete: among equal-energy interiors only the
/// max-CAI one is non-dominated).
#[derive(Debug, Clone, Copy)]
struct Span {
    from: Node,
    to: Node,
}

/// The energy-determining interior of a single loop (hairpin or an internal-loop
/// flank): the terminal-mismatch nucleotides that fix the loop energy, with the
/// energy-irrelevant middle left implicit (respelled best-CAI), or — for the
/// special-hairpin sizes — the whole enumerated loop sequence.
#[derive(Debug, Clone, Copy)]
enum LoopFill {
    /// Terminal-mismatch corner: write `l_nuc` at `from`, the best-CAI path
    /// `mid_from → mid_to` across the interior, and `r_nuc` at `to - 1`. A
    /// length-1 flank collapses to `from == to - 1` with `l_nuc == r_nuc` and an
    /// empty middle, which this spelling handles without a special case.
    Corner {
        from: u32,
        to: u32,
        l_nuc: Nucleotide,
        r_nuc: Nucleotide,
        mid_from: Node,
        mid_to: Node,
    },
    /// A fully-enumerated short loop sequence (the Triloop/Tetraloop/Hexaloop
    /// sizes, whose whole-sequence bonus is energy-relevant): write `nuc[..len]`
    /// from `start`.
    Special {
        start: u32,
        nuc: [Nucleotide; 6],
        len: u8,
    },
}

/// One node of the provenance DAG. A cell's front holds these as inline values
/// while it accumulates; each names its predecessor point(s) by committed arena
/// index and carries the local spelling the step introduces. Mirrors the scalar
/// [`crate::state::Back`], but referencing individual front points rather than
/// whole cells (a front cell holds many points, each with its own provenance).
#[derive(Debug, Clone, Copy)]
enum Prov {
    /// Exterior base case: the empty prefix at the start node.
    ExtEmpty,
    /// Exterior extends by one unpaired nucleotide.
    ExtUnpaired {
        prev: ProvIdx,
        pos: u32,
        nuc: Nucleotide,
    },
    /// Exterior absorbs a closed pair: `C = C ⊗ P`.
    ExtPair { prefix: ProvIdx, pair: ProvIdx },
    /// Hairpin closure.
    Hairpin { pair: PairSpell, fill: LoopFill },
    /// A true internal loop (both flanks non-empty) around one inner pair.
    Internal {
        pair: PairSpell,
        left: LoopFill,
        right: LoopFill,
        inner: ProvIdx,
    },
    /// A stack or bulge (energy-irrelevant best-CAI flanks) around one inner pair.
    StackBulge {
        pair: PairSpell,
        left: Span,
        right: Span,
        inner: ProvIdx,
    },
    /// Multi-loop closure: `P ← Multi`.
    MultiClose { pair: PairSpell, multi: ProvIdx },
    /// Multi-loop assembly: `Multi` wraps an `M2` core with best-CAI flanks.
    MultiWrap {
        left: Span,
        right: Span,
        core: ProvIdx,
    },
    /// A multi-loop opening in progress (the `multi_open` `C + D` sharing): the
    /// left flank and `M2` core are fixed and `C(core_outer)` is folded into the
    /// point's CAI; the right flank is not yet known. Transient — it lives only
    /// in the shared `multi_open` fronts and is rewritten to [`Prov::MultiWrap`]
    /// (right flank `core_outer → frontier`) the moment the opening is closed, so
    /// it never reaches the arena or the traceback.
    MultiOpen {
        left: Span,
        core: ProvIdx,
        core_outer: Node,
    },
    /// First multi-branch: `M ← P`.
    MFromP { pair: ProvIdx },
    /// `M ← M2`.
    MFromM2 { m2: ProvIdx },
    /// `M` extends by one unpaired nucleotide on the right.
    MUnpaired {
        prev: ProvIdx,
        pos: u32,
        nuc: Nucleotide,
    },
    /// Multi-branch bifurcation: `M2 ← M + P`.
    M2Bifurcation { left: ProvIdx, right: ProvIdx },
}

/// The append-only provenance arena. Each [`Self::push`] returns the new node's
/// index, which becomes a committed front point's back-pointer.
struct Arena {
    nodes: Vec<Prov>,
}

impl Arena {
    /// The points committed so far.
    fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Arena {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn push(&mut self, prov: Prov) -> ProvIdx {
        let idx = self.nodes.len() as ProvIdx;
        self.nodes.push(prov);
        idx
    }

    fn get(&self, idx: ProvIdx) -> Prov {
        self.nodes[idx as usize]
    }
}

/// Commit an accumulating front: push each surviving point's [`Prov`] value into
/// the arena, yielding a front of committed indices its children can reference.
/// Called once per cell, when the cell is consumed — so the arena holds one node
/// per stored point, not per transient shift/compose candidate.
fn commit(arena: &mut Arena, front: &Acc) -> Com {
    front.shift_map(Energy::ZERO, 0.0, |p| arena.push(p.back))
}

// ===========================================================================
// The front-valued DP.
// ===========================================================================

/// Table-write candidates buffered from one (possibly parallel) fat production
/// phase, folded into the shared [`Dp`] afterwards. Splitting the producer from
/// the fold lets the fat phases run their independent per-cell work on
/// worker-local buffers with no shared mutation and — crucially — with **no
/// arena access** (the arena is committed only in the serial consume phases, so
/// a producer holds [`Prov`] *values* that reference already-committed parents,
/// never mutating the arena). The candidate fronts own their points, so a
/// producer moves them into the buffer.
///
/// Folding per-table **in production order** reproduces the single-threaded
/// merge sequence exactly. The staircase `⊕` merge is order-independent for the
/// `(mfe, cai)` points, but which [`Prov`] a tied point keeps is fixed by apply
/// order ([`push_if_undominated`] keeps the first-seen among equal-cost twins),
/// so the parallel fold must apply the partials in the deterministic
/// chunk-index order the serial path would have produced them — this is what
/// keeps the *recovered sequences* byte-identical, not merely the objectives.
#[derive(Default)]
struct FrontWrites {
    /// `merge_p` candidates: `(outer node, pair key, front)`.
    p: Vec<(Node, PairKey, Acc)>,
    /// `merge_multi` candidates: `(outer node, pair key, front)`.
    multi: Vec<(Node, PairKey, Acc)>,
    /// `merge_multi_open` candidates: `(opening key, opening)`.
    multi_open: Vec<(MultiOpenKey, Opening)>,
    /// Openings deferred to a codon boundary: `(boundary position, key, opening)`.
    multi_open_pending: Vec<(usize, MultiOpenKey, Opening)>,
    /// A worker-local observer, present only while a probe is attached. Its
    /// counters are sums and maxima, so folding them back is order-independent
    /// and the totals do not depend on how the phase was divided.
    probe: Option<FrontProbe>,
    /// A worker-local census tally and the counts of the closed pair it is
    /// charging, present only while a census is attached.
    census: Option<(Survivors, BoxTally)>,
    /// The internal-loop corner's products, counted where the serial split is
    /// enabled; sums, so folding them back is order-independent.
    corner: ComposeCount,
    /// Products the corner's composes skipped, and points the box's cuts
    /// removed from their arrivals before buffering them; tallied at `P`.
    p_skipped: u64,
    p_boxed: u64,
    /// Opening points a close never shifted because a cut removes them all at
    /// the closed cell; tallied at `Multi` as boxed.
    multi_screened: u64,
    /// Inner-front points a single-branch close never moved because a cut
    /// removes them all at the closed pair.
    single_screened: u64,
    /// Core points a short-flank close never wrote because a cut removes them
    /// all at the closed `Multi` cell.
    short_screened: u64,
    /// Opening slots a close had to build from their recipes, to be built in
    /// place once the closes are applied.
    materialize: Vec<usize>,
    /// Points closes wrote from recipes they built, beyond the held points
    /// [`Dp::count_open_close`] counted.
    close_extra: u64,
}

impl FrontWrites {
    /// Candidates buffered, and the points they carry.
    fn held(&self) -> (u64, u64) {
        let candidates =
            self.p.len() + self.multi.len() + self.multi_open.len() + self.multi_open_pending.len();
        let points = self.p.iter().map(|(_, _, f)| f.len()).sum::<usize>()
            + self.multi.iter().map(|(_, _, f)| f.len()).sum::<usize>()
            + self.multi_open.iter().map(|(_, f)| f.len()).sum::<usize>()
            + self
                .multi_open_pending
                .iter()
                .map(|(_, _, f)| f.len())
                .sum::<usize>();
        (candidates as u64, points as u64)
    }

    /// Fold the buffered candidates into `dp` with the standard dominance
    /// merges, each table in collection (production) order. Different tables are
    /// independent, so the cross-table apply order is immaterial; within one
    /// table the production order reproduces the serial merge (and prune)
    /// schedule exactly.
    fn apply(self, dp: &mut Dp) {
        for (outer, key, f) in self.p {
            dp.merge_p(outer, key, f);
        }
        for (outer, key, f) in self.multi {
            dp.merge_multi(outer, key, f);
        }
        dp.merge_openings(self.multi_open);
        for (boundary, key, f) in self.multi_open_pending {
            dp.multi_open_pending[boundary].push((key, f));
        }
        if let (Some(worker), Some(state)) = (self.probe, dp.probe.as_mut()) {
            state.probe.absorb(&worker);
        }
        if let (Some((_, worker)), Some(state)) = (self.census, dp.census.as_mut()) {
            state.tally.absorb(&worker);
        }
        dp.clock.serial.corner.absorb(&self.corner);
        dp.skipped[CellTable::Pair.index()] += self.p_skipped;
        dp.boxed[CellTable::Pair.index()] += self.p_boxed;
        dp.boxed[CellTable::Multi.index()] += self.multi_screened;
        dp.to_materialize.extend(self.materialize);
        dp.clock.open_close_points += self.close_extra;
        dp.clock.open_close_screened += self.multi_screened;
        dp.clock.single_screened += self.single_screened;
        dp.clock.short_screened += self.short_screened;
    }
}

/// The fold's optional box census, with the per-cell counts it needs to charge
/// a product — the same shape as [`ProbeState`], for the same reason.
struct CensusState<'a> {
    census: &'a BoxCuts,
    tally: BoxTally,
    /// The closed pair the serial consume phases are composing.
    current: Survivors,
    /// Counts for each finalized exterior cell, parallel to `c_committed`.
    c: Vec<Survivors>,
    /// Counts for each finalized multi-branch cell, parallel to `m_committed`.
    m: Vec<FxHashMap<Node, Survivors>>,
}

/// The fold's optional read-only observer, with the per-cell retention counts
/// it needs to charge a front-by-front product without recomputing its
/// operands — each cell's counts are taken once, when the cell is finalized,
/// and read back every time the cell is composed.
struct ProbeState {
    probe: FrontProbe,
    /// The cell the serial consume phases are currently composing.
    current: RetainedCounts,
    /// Counts for each finalized exterior cell, parallel to `c_committed`.
    c: Vec<RetainedCounts>,
    /// Counts for each finalized multi-branch cell, parallel to `m_committed`.
    m: Vec<FxHashMap<Node, RetainedCounts>>,
}

/// The six front-valued CKY tables, their committed-parent caches, the
/// provenance arena, and the immutable scan context. Mirrors
/// [`crate::tables::Tables`] with a [`Front`] per cell.
struct Dp<'a> {
    params: &'a Params,
    dfa: &'a Dfa,
    /// For each closing pair, the terminal mismatch with the lowest hairpin
    /// energy at any size that reads one.
    hairpin_least: [(Nucleotide, Nucleotide); 7],
    /// Under box cuts, each internal-loop shape's least energy over its
    /// terminal mismatches.
    internal_least: Option<InternalLeast>,
    /// Worker threads for the fat production phases (`p_to_single`,
    /// `seed_multi_open`, `multi_open_close`). `1` (the default) runs the exact
    /// single-threaded path; `>1` fans the independent per-cell work across
    /// scoped threads and folds the per-chunk partials back in chunk-index
    /// order — an opt-in speed mode that stays byte-identical to single-thread
    /// (objectives *and* recovered sequences), since every commit stays serial
    /// and the fold order is deterministic. `0` is treated as `1`.
    threads: usize,
    /// Per-cell front-pruning policy. [`Prune::None`] leaves every stored front
    /// un-pruned (the exact fold); a lossy mode caps each cell as it is stored.
    prune: Prune,
    /// How far above its own cell's best a partial derivation may sit and still
    /// be carried. [`Windows::UNBOUNDED`] is the unconstrained fold, byte for
    /// byte. Unlike `prune` this is exact: what it discards provably cannot
    /// complete within the ceiling, so it never trades the front's guarantee.
    windows: Windows,
    /// Each cell's exact cost to go, consulted beside the window where given.
    credit: Option<&'a ExactCredit>,
    /// Derivations the feasibility drop discarded, by table. Per-table because
    /// the tables differ in whether they generate the front's cost, so a total
    /// alone cannot say the drop reached the ones that do.
    dropped: [u64; 6],
    /// Merges that left a cell empty, by table; see [`DropTally::emptied_at`].
    emptied: [u64; 6],
    arena: Arena,
    /// Exterior parses, indexed by [`Dfa::ix`].
    c: Vec<Acc>,
    /// Committed exterior fronts, filled as each `C` node is finalized so a
    /// later pair can read it as its 5′ prefix.
    c_committed: Vec<Com>,
    /// Closed pairs, indexed by outer node, keyed by [`PairKey`].
    p: Vec<FxHashMap<PairKey, Acc>>,
    /// The region constraint every `P` write consults; inert for an unconstrained
    /// fold.
    region: &'a PairConstraint,
    /// Multi-loops in progress, same shape as `p`.
    multi: Vec<FxHashMap<PairKey, Acc>>,
    /// Shared multi-loop openings (the `multi_open` reduction): one front per
    /// `(opening node, opening nucleotide)`, with the left flank + `M2` core +
    /// `C(core_outer)` folded in, closed against each frontier by a uniform CAI
    /// shift `D(frontier)`. The reduction that takes the naive `O(n³·k̄)`
    /// multi-loop assembly to `O(n²·k̄)`.
    ///
    /// The map holds each key's slot in `multi_open_fronts`. A shared opening
    /// has no exact cutoff, so a key, once inserted, is never removed. The
    /// map's iteration order, which decides the order the closes write `Multi`
    /// in and so tie-breaks downstream, therefore depends only on the order
    /// keys are first inserted. Inserting serially and merging the fronts
    /// elsewhere keeps that order while the merges fan out.
    multi_open: FxHashMap<MultiOpenKey, usize>,
    /// The shared openings' fronts, by slot.
    multi_open_fronts: Vec<Acc>,
    /// Under box cuts, a lower bound on each cut's lowest share of each
    /// slot's front, lowered by every arrival merged into it; empty without
    /// cuts. A close skips an opening whose bound a cut excludes.
    multi_open_lows: Vec<Lows>,
    /// Each slot's openings not yet built: recipes the slot's front is built
    /// from, with what it already holds, once a close might keep a point.
    multi_open_lazy: Vec<Vec<LazyOpening>>,
    /// Slots a close needed built, to build in place after the closes.
    to_materialize: Vec<usize>,
    /// Spare per-key arrival buffers for the sharded opening merges.
    open_shards: Vec<Vec<Acc>>,
    /// Openings deferred until their `M2` core's next codon boundary, indexed by
    /// that boundary position; merged into `multi_open` when the fold reaches it,
    /// so an opening is closed only against frontiers past the boundary (the
    /// frontiers whose CAI actually separates as `C + D`).
    multi_open_pending: Vec<Vec<(MultiOpenKey, Opening)>>,
    /// Spare candidate buffers for the sharded push apply's columns, kept
    /// with their capacity across positions.
    p_shards: Vec<Vec<PCandidate>>,
    /// Multi-branch components (`>= 1` branch), indexed by outer node, keyed by
    /// inner node.
    m: Vec<FxHashMap<Node, Acc>>,
    /// Committed `M` fronts, filled as each `M` cell is finalized so a later
    /// bifurcation can read it as its left branch.
    m_committed: Vec<FxHashMap<Node, Com>>,
    /// Multi-branch components (`>= 2` branches), same shape as `m`.
    m2: Vec<FxHashMap<Node, Acc>>,
    /// Read-only observation of per-cell front sizes and product volume. The
    /// recurrence never reads it back, so an attached probe leaves the
    /// recovered front byte-identical by construction.
    probe: Option<ProbeState>,
    /// Read-only box census; like `probe`, never read back by the recurrence.
    census: Option<CensusState<'a>>,
    /// The cuts of a box, applied beside the energy drop where given. Unlike
    /// `census` this changes what the fold holds.
    cuts: Option<&'a BoxCuts>,
    /// Points the box's cuts discarded, by table; see [`DropTally::boxed_at`].
    boxed: [u64; 6],
    /// Products composes never generated, by table; see
    /// [`DropTally::skipped_at`].
    skipped: [u64; 6],
    /// Where the fold's time went; read by nothing in the recurrence.
    clock: FoldClock,
    /// Whether to split the serial pair phase and count its bifurcation.
    detail: bool,
    /// Positions whose bifurcations were fanned out; see [`SerialDetail::fanned`].
    fanned: AtomicUsize,
    /// Whether to tally inside the fanned-out phases; see [`FanDetail`].
    fan_detail: bool,
    /// A cap on the workers a fan-out starts (`DERNA_FAN_WORKERS`).
    fan_cap: Option<usize>,
    /// The fold's workers, kept for its whole length; absent on one thread.
    pool: Option<&'a Pool>,
}

impl<'a> Dp<'a> {
    /// Fold `dfa` with front-valued cells. Mirrors the [`crate::scan`] phase
    /// order (hairpin seed, then per-position multi-close → P-consume → M2 →
    /// M/C extend), but every cell holds a [`Front`] and no runtime reduction
    /// is applied.
    #[allow(clippy::too_many_arguments)]
    fn fill(
        params: &'a Params,
        dfa: &'a Dfa,
        prune: Prune,
        windows: Windows,
        credit: Option<&'a ExactCredit>,
        cuts: Option<&'a BoxCuts>,
        threads: usize,
        probe: Option<FrontProbe>,
        census: Option<&'a BoxCuts>,
        region: &'a PairConstraint,
        pool: Option<&'a Pool>,
    ) -> Self {
        let nodes = dfa.slots();
        let mut dp = Dp {
            params,
            dfa,
            hairpin_least: least_hairpin_mismatches(params),
            internal_least: cuts.map(|_| InternalLeast::new(params)),
            threads: threads.max(1),
            prune,
            windows,
            credit,
            dropped: [0; 6],
            emptied: [0; 6],
            arena: Arena::new(),
            c: vec![Acc::empty(); nodes],
            c_committed: vec![Com::empty(); nodes],
            p: vec![FxHashMap::default(); nodes],
            region,
            multi: vec![FxHashMap::default(); nodes],
            multi_open: FxHashMap::default(),
            multi_open_fronts: Vec::new(),
            multi_open_lows: Vec::new(),
            multi_open_lazy: Vec::new(),
            to_materialize: Vec::new(),
            open_shards: Vec::new(),
            multi_open_pending: vec![Vec::new(); dfa.seq_len as usize + 1],
            p_shards: Vec::new(),
            m: vec![FxHashMap::default(); nodes],
            m_committed: vec![FxHashMap::default(); nodes],
            m2: vec![FxHashMap::default(); nodes],
            probe: probe.map(|probe| ProbeState {
                probe,
                current: RetainedCounts::default(),
                c: vec![RetainedCounts::default(); nodes],
                m: vec![FxHashMap::default(); nodes],
            }),
            census: census.map(|census| CensusState {
                census,
                tally: census.tally(),
                current: Survivors::new(),
                c: vec![Survivors::new(); nodes],
                m: vec![FxHashMap::default(); nodes],
            }),
            cuts,
            boxed: [0; 6],
            skipped: [0; 6],
            clock: FoldClock::default(),
            detail: serial_detail::enabled(),
            fanned: AtomicUsize::new(0),
            fan_detail: fan_detail::enabled(),
            fan_cap: fan_detail::workers_cap(),
            pool,
        };
        dp.c[dfa.ix(dfa.start())] = Acc::singleton(Energy::ZERO, 0.0, Prov::ExtEmpty);
        let t = Instant::now();
        let boxed = dp.boxed[CellTable::Pair.index()];
        dp.seed_hairpins();
        dp.clock.hairpin_boxed = dp.boxed[CellTable::Pair.index()] - boxed;
        let mut t = dp.lap(t, |c| &mut c.hairpins_s);
        for pos in 0..=dfa.seq_len {
            if dp.cuts.is_some_and(|cuts| cuts.stops_at(dp.arena.len())) {
                dp.clock.stopped = true;
                break;
            }
            dp.multi_open_activate(pos);
            t = dp.lap(t, |c| &mut c.open_activate_s);
            dp.multi_close(pos);
            dp.lap(t, |c| &mut c.multi_close_s);
            dp.p_consume(pos);
            dp.m2_consume(pos);
            t = Instant::now();
            dp.multi_open_close(pos);
            t = dp.lap(t, |c| &mut c.open_close_s);
            if pos < dfa.seq_len {
                dp.m_extend(pos);
                t = dp.lap(t, |c| &mut c.m_extend_s);
                dp.c_extend(pos);
                t = dp.lap(t, |c| &mut c.c_extend_s);
            }
        }
        dp
    }

    /// Charge the time since `since` to one of the clock's phases and restart.
    fn lap(&mut self, since: Instant, slot: impl FnOnce(&mut FoldClock) -> &mut f64) -> Instant {
        let now = Instant::now();
        *slot(&mut self.clock) += now.duration_since(since).as_secs_f64();
        now
    }

    // -- Hairpins ----------------------------------------------------------

    /// Seed every hairpin-closed `P` cell, mirroring `Scan::seed_hairpins`.
    fn seed_hairpins(&mut self) {
        // The lattice outlives the fold, so its edges are read in place while
        // the fold's tables change.
        let dfa = self.dfa;
        for a in 0..dfa.seq_len {
            for &an in dfa.nodes_at(a) {
                for &eo in dfa.right(an) {
                    self.seed_hairpins_from(an, eo);
                }
            }
        }
    }

    fn seed_hairpins_from(&mut self, an: Node, eo: Edge) {
        let dfa = self.dfa;
        let mut b = dfa.next_partner(eo.nuc, an.pos + MIN_HAIRPIN_LOOP + 1);
        while b < dfa.seq_len {
            for &bn in dfa.nodes_at(b) {
                for &ec in dfa.right(bn) {
                    self.seed_hairpin_close(an, eo, bn, ec);
                }
            }
            b = dfa.next_partner(eo.nuc, b + 1);
        }
    }

    fn seed_hairpin_close(&mut self, an: Node, eo: Edge, bn: Node, ec: Edge) {
        let Some(closing) = canonical(eo.nuc, ec.nuc) else {
            return;
        };
        let key = PairKey {
            open: an,
            close: bn,
            opening: eo.opening,
            nuc_close: ec.nuc,
        };
        if self.hairpin_boxed_out(eo, bn, ec, closing, key) {
            return;
        }
        let interior = self.hairpin_front(eo.to, bn, closing);
        if interior.is_empty() {
            return;
        }
        let pair = PairSpell::of(&key);
        let closed = interior.shift_map(Energy::ZERO, eo.cai + ec.cai, |p| Prov::Hairpin {
            pair,
            fill: p.back,
        });
        self.clock.hairpin_points += closed.len() as u64;
        self.merge_p(ec.to, key, closed);
    }

    /// Whether a box cut removes every point of the hairpin closed by
    /// `eo`…`ec` before its interiors are enumerated. No interior has a lower
    /// energy than the size's penalty with the closing pair's least terminal
    /// mismatch, nor a higher adaptation than the loop's best spelling, so a
    /// cut that fails that pair of bounds fails every interior. The special
    /// sizes (3, 4, 6), whose table bonuses read the whole loop, are not
    /// bounded.
    fn hairpin_boxed_out(
        &mut self,
        eo: Edge,
        bn: Node,
        ec: Edge,
        closing: BasePair,
        key: PairKey,
    ) -> bool {
        let Some(cuts) = self.cuts else {
            return false;
        };
        let size = (bn.pos - eo.to.pos) as usize;
        if matches!(size, 3 | 4 | 6) {
            return false;
        }
        let Some(best) = self.dfa.broken_codon_cai(eo.to, bn) else {
            return false;
        };
        let (first, last) = self.hairpin_least[closing as usize];
        let least = hairpin_energy_scalar(self.params, closing, size, first, last);
        let outs = cuts.outs_at(CellRef::P(ec.to, key));
        let out = !cuts.admits_before_elimination(&outs, least, best + eo.cai + ec.cai);
        self.clock.hairpin_screened += u64::from(out);
        out
    }

    /// The front of every non-dominated hairpin interior `(energy, cai)` for a
    /// loop closed by `closing` spanning `start..end`, each point tagged with
    /// the [`LoopFill`] that respells it. Special sizes (3/4/6) enumerate the
    /// full loop sequence (a table bonus reads it); other sizes enumerate the
    /// two terminal mismatches and leave the energy-irrelevant middle best-CAI.
    fn hairpin_front(&self, start: Node, end: Node, closing: BasePair) -> Front<LoopFill> {
        let size = (end.pos - start.pos) as usize;
        let points = if matches!(size, 3 | 4 | 6) {
            self.special_hairpin_points(start, end, closing)
        } else {
            corner_fills(self.dfa, start, end)
                .into_iter()
                .map(|c| {
                    let e = hairpin_energy_scalar(self.params, closing, size, c.first, c.last);
                    Point::new(e, c.cai, c.fill)
                })
                .collect()
        };
        Front::from_points(points)
    }

    /// Enumerate the special-hairpin (`3/4/6`) interiors: each distinct loop
    /// sequence is a point, since the whole-sequence bonus makes the interior
    /// energy-relevant.
    fn special_hairpin_points(
        &self,
        start: Node,
        end: Node,
        closing: BasePair,
    ) -> Vec<Point<LoopFill>> {
        self.dfa
            .paths_between(start, end)
            .into_iter()
            .map(|(cai, seq)| {
                let e = hairpin_energy(
                    self.params,
                    &HairpinLoop {
                        closing,
                        loop_seq: &seq,
                    },
                );
                Point::new(e, cai, special_fill(start.pos, &seq))
            })
            .collect()
    }

    // -- Multi-loop close: `Multi -> P` ------------------------------------

    fn multi_close(&mut self, pos: u32) {
        for &jn in self.dfa.nodes_at(pos) {
            for (key, f) in take_col(&mut self.multi[self.dfa.ix(jn)]) {
                let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                let delta = self.multi_close_cost(closing);
                let pair = PairSpell::of(&key);
                self.observe(CellTable::Multi, pair_span(key, jn), &f);
                self.census_cell(CellTable::Multi, Some(CellRef::Multi(jn, key)), &f);
                let multi = commit(&mut self.arena, &f);
                let closed = multi.shift_map(delta, 0.0, |p| Prov::MultiClose {
                    pair,
                    multi: p.back,
                });
                self.merge_p(jn, key, closed);
                // Re-insert the Multi cell: multi_close reads it, never consumes it.
                self.multi[self.dfa.ix(jn)].insert(key, f);
            }
        }
    }

    // -- Consume `P[pos]`: exterior, multibranch, and single-branch grow ----

    /// Process every closed pair whose outer boundary is `pos`: commit its
    /// front (so its points can parent larger structures), then extend the
    /// exterior, open / bifurcate a multibranch, and grow stacks / bulges /
    /// internal loops into larger `P` cells. Moves the columns out first (they
    /// are consumed here and never re-read; later single-branch growth writes
    /// only into `P` columns at greater positions) so the fronts are taken, not
    /// cloned.
    fn p_consume(&mut self, pos: u32) {
        let t = Instant::now();
        let mut raw: Vec<(Node, PairKey, Acc)> = Vec::new();
        for &jn in self.dfa.nodes_at(pos) {
            for (key, f) in std::mem::take(&mut self.p[self.dfa.ix(jn)]) {
                raw.push((jn, key, f));
            }
        }
        let cells: Vec<(Node, PairKey, Com)> = raw
            .iter()
            .map(|(jn, key, f)| (*jn, *key, commit(&mut self.arena, f)))
            .collect();
        // Pass 1: exterior (`C ⊗ P`), branch open (`M = P`) and the bifurcation
        // (`M2 = M ⊗ P`), a window of closed pairs at a time, so that only one
        // window's composed bifurcations are held at once. Each table's merges
        // still come in the closed pairs' order: the windows run in that order,
        // and no window's merges touch a table another phase of it reads.
        for window in cells.chunks(BIFURCATION_WINDOW) {
            self.pair_pass(window);
        }
        let t = self.lap(t, |c| &mut c.pair_serial_s);
        // Pass 2: the fat single-branch push. Reads only committed `P` fronts +
        // DFA / energies (no arena) and writes `P` columns at greater positions
        // (never re-read this phase), so the closed pairs are independent — the
        // fan-out folds them back in cell order.
        let size = |(_, _, f): &(Node, PairKey, Com)| f.len();
        self.fan_out(FanSite::Push, &cells, size, |dp, w, (jn, key, f)| {
            dp.hand_to_worker(w, pair_span(*key, *jn), f);
            dp.census_latch(w, CellRef::P(*jn, *key), f);
            dp.p_to_single(w, *jn, *key, f)
        });
        self.lap(t, |c| &mut c.pair_single_s);
    }

    /// One window of [`Self::p_consume`]'s first pass. The bifurcations'
    /// composes read only committed `M` columns, the credit and the cuts, so
    /// they are made first, fanned out; the serial pass then merges each in
    /// the order the closed pairs and their `M` columns give, which is the
    /// order an immediate merge would have used.
    fn pair_pass(&mut self, cells: &[(Node, PairKey, Com)]) {
        let composing = self.detail.then(Instant::now);
        let composed = self.compose_bifurcations(cells);
        if let Some(t) = composing {
            let d = &mut self.clock.serial;
            let s = t.elapsed().as_secs_f64();
            d.compose_s += s;
            d.bifurcation_s += s;
        }
        for ((jn, key, f), bifurcations) in cells.iter().zip(composed) {
            let counts = self.observe(CellTable::Pair, pair_span(*key, *jn), f);
            self.hold(counts);
            let survivors = self.census_cell(CellTable::Pair, Some(CellRef::P(*jn, *key)), f);
            self.census_hold(survivors);
            // The split's own counting runs inside these clocks and is taken
            // back out of each.
            let counting = self.clock.serial.counting_s;
            let t = self.detail.then(Instant::now);
            self.p_to_exterior(*jn, *key, f);
            let t = self.lap_serial(t, |d| &mut d.exterior_s);
            let d = &mut self.clock.serial;
            d.exterior_s -= d.counting_s - counting;
            self.p_open_branch(*jn, *key, f);
            let t = self.lap_serial(t, |d| &mut d.open_s);
            let counting = self.clock.serial.counting_s;
            self.merge_bifurcations(*jn, *key, f, bifurcations);
            self.lap_serial(t, |d| &mut d.bifurcation_s);
            let d = &mut self.clock.serial;
            d.bifurcation_s -= d.counting_s - counting;
        }
    }

    /// `C = C ⊗ P`: attach a closed pair to the exterior prefix before its 5′
    /// open node.
    ///
    /// The compose generates only what `merge_c` keeps. The exterior column is
    /// dense and never removes a cell, so an emptied arrival needs no replay;
    /// it is merged anyway, a no-op on the cell. The prefix is moved out and
    /// back rather than cloned, as in [`Self::bifurcate`].
    fn p_to_exterior(&mut self, o: Node, key: PairKey, f: &Com) {
        if self.c_committed[self.dfa.ix(key.open)].is_empty() {
            return;
        }
        let prefix = std::mem::take(&mut self.c_committed[self.dfa.ix(key.open)]);
        let inner = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let delta = self.terminal_au(inner);
        self.observe_absorb(key.open);
        self.census_absorb(key.open);
        if self.detail {
            self.count_exterior(o, &prefix, f, delta);
        }
        let cell = CellRef::C(o);
        let held = Some(&self.c[self.dfa.ix(o)]);
        let filter = self.compose_filter(CellTable::Exterior, cell, held, &prefix, f, delta);
        let (joined, generated) = prefix.compose_within(
            f,
            delta,
            0.0,
            filter.cutoff,
            filter.screen(),
            |mfe, cai| filter.admits(mfe, cai),
            |prefix, pair| Prov::ExtPair { prefix, pair },
        );
        self.skipped[CellTable::Exterior.index()] += (prefix.len() * f.len() - generated) as u64;
        self.c_committed[self.dfa.ix(key.open)] = prefix;
        self.merge_c(o, joined);
    }

    /// `M = P`: open a branch.
    fn p_open_branch(&mut self, o: Node, key: PairKey, f: &Com) {
        let stem = self.m_stem_cost(pair_from_nucleotides(key.nuc_open(), key.nuc_close));
        let as_branch = f.shift_map(stem, 0.0, |p| Prov::MFromP { pair: p.back });
        self.merge_m(o, key.open, as_branch);
    }

    /// Every closed pair's bifurcations, composed ahead of their merges: one
    /// list per closed pair, in `cells` order, fanned out where there are
    /// enough of them. Each compose reads only state the pass does not write,
    /// so the lists are the same under any worker count.
    fn compose_bifurcations(&mut self, cells: &[(Node, PairKey, Com)]) -> Vec<Vec<Bifurcation>> {
        let of = |dp: &Self, (o, key, f): &(Node, PairKey, Com)| dp.bifurcations_of(*o, *key, f);
        let t = Instant::now();
        let composed: Vec<Vec<Bifurcation>> = if self.threads <= 1 || cells.len() < FAN_OUT_MIN {
            let composed = cells.iter().map(|cell| of(self, cell)).collect();
            if self.fan_detail {
                let tally = self.clock.fan.site_mut(FanSite::Bifurcate);
                tally.serial += 1;
                tally.serial_s += t.elapsed().as_secs_f64();
            }
            composed
        } else {
            self.fanned.fetch_add(1, Ordering::Relaxed);
            // A closed pair's compose work is its front against every point
            // of the `M` column at its open node.
            let mut column_points: FxHashMap<usize, usize> = FxHashMap::default();
            for (_, key, _) in cells {
                column_points
                    .entry(self.dfa.ix(key.open))
                    .or_insert_with(|| {
                        self.m_committed[self.dfa.ix(key.open)]
                            .values()
                            .map(Front::len)
                            .sum()
                    });
            }
            let weight = |(_, key, f): &(Node, PairKey, Com)| {
                f.len() * column_points[&self.dfa.ix(key.open)]
            };
            let (parts, times) = self.produce_parallel(
                cells,
                &weight,
                &|dp, w: &mut Vec<Vec<Bifurcation>>, cell| {
                    w.push(of(dp, cell));
                },
            );
            let composed = parts.into_iter().flatten().collect();
            if self.fan_detail {
                let compute = t.elapsed();
                self.clock
                    .fan
                    .record_fanned(FanSite::Bifurcate, compute, &times);
            }
            composed
        };
        if self.fan_detail {
            self.hold_bifurcations(cells.len(), &composed);
        }
        composed
    }

    /// Tally what one position's bifurcation buffer holds before its merges.
    fn hold_bifurcations(&mut self, items: usize, composed: &[Vec<Bifurcation>]) {
        let arrivals = composed.iter().map(Vec::len).sum::<usize>() as u64;
        let points = composed
            .iter()
            .flatten()
            .map(|b| b.arrival.len() as u64)
            .sum::<u64>();
        let bytes = points * std::mem::size_of::<Point<Prov>>() as u64;
        let tally = self.clock.fan.site_mut(FanSite::Bifurcate);
        tally.items += items as u64;
        tally.candidates += arrivals;
        tally.points += points;
        self.clock.fan.hold_bifurcations(arrivals, points, bytes);
    }

    /// One closed pair's bifurcations: for each finalized `M[key.open][inner]`,
    /// in the column's order, its compose with `f`, shifted by the branch
    /// stem, bound for `M2[o][inner]`.
    fn bifurcations_of(&self, o: Node, key: PairKey, f: &Com) -> Vec<Bifurcation> {
        let stem = self.m_stem_cost(pair_from_nucleotides(key.nuc_open(), key.nuc_close));
        self.m_committed[self.dfa.ix(key.open)]
            .iter()
            .map(|(&inner, left)| self.bifurcation(CellRef::M2(o, inner), inner, left, f, stem))
            .collect()
    }

    /// The compose of `left ⊗ right` shifted by `stem` into `cell`, filtered
    /// to what its merge can keep and cut by the box, as it will arrive.
    ///
    /// The filter does not read the cell's held points, which the merges of
    /// earlier closed pairs are still changing; it is looser for that, and as
    /// sound, since the merge truncates what the filter lets through.
    fn bifurcation(
        &self,
        cell: CellRef,
        inner: Node,
        left: &Com,
        right: &Com,
        stem: Energy,
    ) -> Bifurcation {
        let filter = self.compose_filter(CellTable::Branch2, cell, None, left, right, stem);
        let (mut arrival, generated) = left.compose_within(
            right,
            stem,
            0.0,
            filter.cutoff,
            filter.screen(),
            |mfe, cai| filter.admits(mfe, cai),
            |left, right| Prov::M2Bifurcation { left, right },
        );
        let eliminated = arrival.len();
        let boxed = self
            .cuts
            .map_or(0, |cuts| cuts.retain_in_box(cell, &mut arrival));
        let replay = arrival.is_empty() && self.unfiltered_arrives(cell, left, right, stem);
        Bifurcation {
            inner,
            arrival,
            products: left.len() * right.len(),
            generated,
            eliminated,
            boxed,
            replay,
        }
    }

    /// Merge one closed pair's bifurcations into `M2[o]`, in the order they
    /// were composed.
    fn merge_bifurcations(
        &mut self,
        o: Node,
        key: PairKey,
        f: &Com,
        bifurcations: Vec<Bifurcation>,
    ) {
        let open = key.open;
        for b in bifurcations {
            self.observe_bifurcation(open, b.inner);
            self.census_bifurcation(open, b.inner);
            self.skipped[CellTable::Branch2.index()] += (b.products - b.generated) as u64;
            if self.detail {
                self.count_bifurcation(o, key, f, &b);
            }
            let (inner, kept) = (b.inner, b.eliminated - b.boxed);
            let t = self.detail.then(Instant::now);
            self.merge_bifurcation(o, b);
            self.lap_serial(t, |d| &mut d.merge_s);
            if self.detail {
                self.count_merged(o, inner, f, kept);
            }
        }
    }

    /// Merge a bifurcation's filtered arrival into `M2[o][inner]`, leaving the
    /// column exactly as the unfiltered arrival's merge would.
    ///
    /// The points agree by construction; what needs care is the column's
    /// hash table. The unfiltered merge inserts an absent cell before
    /// truncating it, and removes it again when the exact cutoff empties it.
    /// That insertion and removal can leave a tombstone that moves the next
    /// resize and so the column's iteration order, which decides ties
    /// downstream. So where the filtered arrival leaves nothing to merge into
    /// an absent cell, the insertion and removal are made anyway exactly when
    /// the unfiltered merge would have made them: always without a box (its
    /// arrival is never empty), and with one when some point of the unfiltered
    /// arrival passes the cuts.
    fn merge_bifurcation(&mut self, o: Node, b: Bifurcation) {
        let Bifurcation {
            inner,
            arrival,
            boxed,
            replay,
            ..
        } = b;
        let cell = CellRef::M2(o, inner);
        self.boxed[CellTable::Branch2.index()] += boxed as u64;
        let absent = !self.m2[self.dfa.ix(o)].contains_key(&inner);
        if arrival.is_empty() && !(absent && replay) {
            return;
        }
        // An empty arrival reaches here only to replay an insertion the exact
        // cutoff then empties; without one, the filter never empties it.
        debug_assert!(
            !arrival.is_empty() || self.exact(cell, self.windows.beyond_exterior).is_some()
        );
        if self.detail && arrival.is_empty() {
            self.clock.serial.replayed += 1;
        }
        self.merge_m2(o, inner, arrival);
    }

    /// Whether the unfiltered compose of `left ⊗ right` shifted by `dmfe`
    /// would reach the merge into `cell` with a point left after the box's
    /// cuts; always, without a box.
    fn unfiltered_arrives<B: Copy, C: Copy>(
        &self,
        cell: CellRef,
        left: &Front<B>,
        right: &Front<C>,
        dmfe: Energy,
    ) -> bool {
        let Some(cuts) = self.cuts else {
            return true;
        };
        unfiltered_arrives_at(cuts, &cuts.outs_at(cell), left, right, dmfe)
    }

    /// What a compose of `left ⊗ right` shifted by `dmfe`, bound for `cell` in
    /// `table`, may skip generating: the products its merge would discard.
    /// `held` is what the cell holds now, where the merge follows at once.
    ///
    /// The merge truncates the cell at its exact cutoff and at its best point
    /// plus the window. The best after the merge is at most the held cell's
    /// best and, where no cut can remove it first, at most the compose's own
    /// best, `left[0] + right[0] + dmfe`; so each of those plus the window
    /// bounds the truncation from above, and a product above the lowest of
    /// them is discarded. Nothing is skipped under a lossy prune, which reads
    /// the cell before it is truncated, or where the bounds turned the filter
    /// off.
    fn compose_filter<B, C>(
        &self,
        table: CellTable,
        cell: CellRef,
        held: Option<&Acc>,
        left: &Front<B>,
        right: &Front<C>,
        dmfe: Energy,
    ) -> ComposeFilter<'a> {
        let offset = self.windows.filter_offset(table);
        let Some(offset) = offset.filter(|_| self.prune.is_exact()) else {
            return ComposeFilter::default();
        };
        let slack = match table {
            CellTable::Exterior => self.windows.exterior,
            _ => self.windows.beyond_exterior,
        };
        let exact = self.exact(cell, slack);
        let held_best = held.and_then(|cell| cell.points().first()).map(|p| p.mfe);
        let own_best = match (left.points().first(), right.points().first()) {
            (Some(a), Some(b)) if self.cuts.is_none() => Some(a.mfe + b.mfe + dmfe),
            _ => None,
        };
        let windowed = slack.window().and_then(|window| {
            let best = held_best.into_iter().chain(own_best).min()?;
            Some(Energy(best.0.saturating_add(window.0)))
        });
        let cutoff = exact.into_iter().chain(windowed).min();
        ComposeFilter {
            cutoff: cutoff.map(|c| Energy(c.0.saturating_sub(offset.0))),
            cuts: self.cuts.map(|cuts| (cuts, cuts.outs_at(cell))),
        }
    }

    /// Charge the time since `since` to one of the serial split's slots and
    /// restart; `None`, where the split is off, stays `None`.
    fn lap_serial(
        &mut self,
        since: Option<Instant>,
        slot: impl FnOnce(&mut SerialDetail) -> &mut f64,
    ) -> Option<Instant> {
        let since = since?;
        let now = Instant::now();
        *slot(&mut self.clock.serial) += now.duration_since(since).as_secs_f64();
        Some(now)
    }

    /// Count one internal-loop corner compose into `cell` on the worker's count.
    fn count_corner<B>(&self, w: &mut FrontWrites, cell: CellRef, f: &Com, corner: &Front<B>) {
        let cutoff = self.exact(cell, self.windows.beyond_exterior);
        let within = serial_detail::within_cutoff(f, corner, Energy::ZERO, cutoff);
        let boxed = self.cuts.map_or(within, |cuts| {
            serial_detail::within_box(f, corner, Energy::ZERO, cutoff, cuts, &cuts.outs_at(cell))
        });
        w.corner.absorb(&ComposeCount {
            products: (f.len() * corner.len()) as u64,
            within_cutoff: within,
            within_box: boxed,
        });
    }

    /// Count one exterior attach's products and those within its cell's cutoff.
    fn count_exterior(&mut self, o: Node, prefix: &Com, f: &Com, delta: Energy) {
        let t = Instant::now();
        let cutoff = self.exact(CellRef::C(o), self.windows.exterior);
        let within = serial_detail::within_cutoff(prefix, f, delta, cutoff);
        let d = &mut self.clock.serial;
        d.exterior_products += (prefix.len() * f.len()) as u64;
        d.exterior_within_cutoff += within;
        d.counting_s += t.elapsed().as_secs_f64();
    }

    /// Count one bifurcation compose into `cell`: its products, those it
    /// generated, what it returned, and the products within the cell's cutoff
    /// and the box.
    #[allow(clippy::too_many_arguments)]
    fn count_bifurcation(&mut self, o: Node, key: PairKey, f: &Com, b: &Bifurcation) {
        let t = Instant::now();
        let cell = CellRef::M2(o, b.inner);
        let stem = self.m_stem_cost(pair_from_nucleotides(key.nuc_open(), key.nuc_close));
        let left = &self.m_committed[self.dfa.ix(key.open)][&b.inner];
        let cutoff = self.exact(cell, self.windows.beyond_exterior);
        let within = serial_detail::within_cutoff(left, f, stem, cutoff);
        let boxed = self.cuts.map_or(within, |cuts| {
            serial_detail::within_box(left, f, stem, cutoff, cuts, &cuts.outs_at(cell))
        });
        let d = &mut self.clock.serial;
        d.composes += 1;
        d.products += (left.len() * f.len()) as u64;
        d.generated += b.generated as u64;
        d.after_eliminate += b.eliminated as u64;
        d.within_cutoff += within;
        d.within_box += boxed;
        d.counting_s += t.elapsed().as_secs_f64();
    }

    /// Count what one bifurcation's arrival kept past the cuts, and how much of
    /// it `M2[o][inner]` still holds after the merge. The arrival's points are
    /// the ones whose right branch is a point of `f`, whose committed indices
    /// are contiguous.
    fn count_merged(&mut self, o: Node, inner: Node, f: &Com, after_cuts: usize) {
        let t = Instant::now();
        let (Some(first), Some(last)) = (f.points().first(), f.points().last()) else {
            return;
        };
        let range = first.back..=last.back;
        let kept = self.m2[self.dfa.ix(o)].get(&inner).map_or(0, |cell| {
            cell.iter()
                .filter(|p| matches!(p.back, Prov::M2Bifurcation { right, .. } if range.contains(&right)))
                .count()
        });
        let d = &mut self.clock.serial;
        d.after_cuts += after_cuts as u64;
        d.after_merge += kept as u64;
        d.counting_s += t.elapsed().as_secs_f64();
    }

    /// Grow `P[o]` (inner pair) into every outer pair `(p, q)` whose unpaired
    /// flanks total at most `SINGLE_MAX_LEN` — the naive stack / bulge /
    /// internal closure (no Lyngsø–Zuker or one-sided reduction).
    fn p_to_single(&self, w: &mut FrontWrites, o: Node, key: PairKey, f: &Com) {
        let lows = self.lows_of(f);
        let a = key.open.pos;
        // n1 = a − p − 1 ≤ SINGLE_MAX_LEN.
        let p_lo = a.saturating_sub(SINGLE_MAX_LEN + 1);
        let mut corners = CloseCorners::default();
        for p in p_lo..a {
            for &pn in self.dfa.nodes_at(p) {
                for &ep in self.dfa.right(pn) {
                    self.single_from_left(w, (o, key), (f, &lows), pn, ep, &mut corners);
                }
            }
        }
    }

    /// Each box cut's lowest share of `f`'s points; empty without cuts.
    fn lows_of<B>(&self, f: &Front<B>) -> Lows {
        let mut lows = Lows::new();
        if let Some(cuts) = self.cuts {
            cuts.lower_shares(&mut lows, f);
        }
        lows
    }

    fn single_from_left(
        &self,
        w: &mut FrontWrites,
        (o, key): (Node, PairKey),
        (f, lows): (&Com, &[f64]),
        pn: Node,
        ep: Edge,
        corners: &mut CloseCorners,
    ) {
        if self.dfa.broken_codon_cai(ep.to, key.open).is_none() {
            return;
        }
        let n1 = key.open.pos - ep.to.pos;
        // The 5′ corners are this flank's for every 3′ side below.
        if n1 > 0 {
            corners.lefts = corner_fills(self.dfa, ep.to, key.open);
        }
        let b = o.pos - 1;
        // n2 = q − b − 1 ≤ SINGLE_MAX_LEN − n1.
        let q_hi = (b + 2 + (SINGLE_MAX_LEN - n1)).min(self.dfa.seq_len);
        let mut q = self.dfa.next_partner(ep.nuc, b + 1);
        while q < q_hi {
            for &qn in self.dfa.nodes_at(q) {
                for &eq in self.dfa.right(qn) {
                    self.single_close(
                        w,
                        &SingleClose {
                            o,
                            key,
                            pn,
                            ep,
                            qn,
                            eq,
                            n1,
                        },
                        (f, lows),
                        corners,
                    );
                }
            }
            q = self.dfa.next_partner(ep.nuc, q + 1);
        }
    }

    fn single_close(
        &self,
        w: &mut FrontWrites,
        c: &SingleClose,
        (f, lows): (&Com, &[f64]),
        corners: &mut CloseCorners,
    ) {
        let Some(outer_pair) = canonical(c.ep.nuc, c.eq.nuc) else {
            return;
        };
        let inner_pair = pair_from_nucleotides(c.key.nuc_open(), c.key.nuc_close);
        let n2 = c.qn.pos - c.o.pos;
        let oc = OuterClose {
            key: PairKey {
                open: c.pn,
                close: c.qn,
                opening: c.ep.opening,
                nuc_close: c.eq.nuc,
            },
            outer_pair,
            inner_pair,
            n2,
        };
        if c.n1 > 0 && n2 > 0 {
            if self.internal_boxed_out(c, &oc, lows) {
                w.single_screened += f.len() as u64;
                return;
            }
            self.internal_close(w, c, f, &oc, corners);
        } else {
            self.stack_or_bulge_close(w, c, (f, lows), &oc);
        }
    }

    /// Whether a box cut removes every point an internal loop closing `f`
    /// into `oc` could make, before its corners are enumerated: no corner
    /// has a lower energy than the loop shape's least over its terminal
    /// mismatches, nor a higher adaptation than the two flanks' best
    /// spellings.
    fn internal_boxed_out(&self, c: &SingleClose, oc: &OuterClose, lows: &[f64]) -> bool {
        let (Some(cuts), Some(least)) = (self.cuts, self.internal_least.as_ref()) else {
            return false;
        };
        if lows.is_empty() {
            return false;
        }
        let (Some(left), Some(right)) = (
            self.dfa.broken_codon_cai(c.ep.to, c.key.open),
            self.dfa.broken_codon_cai(c.o, c.qn),
        ) else {
            return false;
        };
        let energy = least.at(oc.outer_pair, oc.inner_pair, c.n1, oc.n2);
        let dcai = c.ep.cai + c.eq.cai + left + right;
        let outs = cuts.outs_at(CellRef::P(c.eq.to, oc.key));
        cuts.excludes_moved(&outs, lows, energy, dcai)
    }

    /// A true internal loop (both flanks non-empty): its energy reads the four
    /// terminal mismatches, so those are enumerated as a corner set (a small
    /// front of `(loop energy, corner cai)`) and Minkowski-composed with the
    /// inner front. This is the `k̄²` step.
    fn internal_close(
        &self,
        w: &mut FrontWrites,
        c: &SingleClose,
        f: &Com,
        oc: &OuterClose,
        corners: &mut CloseCorners,
    ) {
        let lefts = &corners.lefts;
        let rights = corners.rights.get(self.dfa, c.o, c.qn);
        let edge_cai = c.ep.cai + c.eq.cai;
        let mut pts = Vec::with_capacity(lefts.len() * rights.len());
        for l in lefts {
            for r in rights {
                let energy = internal_energy_scalar(
                    self.params,
                    &InternalLoopScalar {
                        outer: oc.outer_pair,
                        inner: oc.inner_pair,
                        boundary_left5p: l.first,
                        boundary_left3p: l.last,
                        boundary_right5p: r.first,
                        boundary_right3p: r.last,
                        n1: c.n1 as usize,
                        n2: oc.n2 as usize,
                    },
                );
                pts.push(Point::new(
                    energy,
                    edge_cai + l.cai + r.cai,
                    (l.fill, r.fill),
                ));
            }
        }
        let corner_front = Front::from_points(pts);
        if corner_front.is_empty() {
            return;
        }
        let pair = PairSpell::of(&oc.key);
        if let Some(probe) = w.probe.as_mut() {
            probe.observe_latched_product(ComposeSite::InternalCorner, &corner_front);
        }
        if let Some((held, tally)) = w.census.as_mut() {
            tally.observe_uncut_product(ComposeSite::InternalCorner, held, corner_front.len());
        }
        if self.detail {
            self.count_corner(w, CellRef::P(c.eq.to, oc.key), f, &corner_front);
        }
        let target = CellRef::P(c.eq.to, oc.key);
        let filter = self.compose_filter(
            CellTable::Pair,
            target,
            None,
            f,
            &corner_front,
            Energy::ZERO,
        );
        let (mut joined, generated) = f.compose_within(
            &corner_front,
            Energy::ZERO,
            0.0,
            filter.cutoff,
            filter.screen(),
            |mfe, cai| filter.admits(mfe, cai),
            |inner, (left, right)| Prov::Internal {
                pair,
                left,
                right,
                inner,
            },
        );
        w.p_skipped += (f.len() * corner_front.len() - generated) as u64;
        // The filter looked the cuts up at `target` already, where it has them.
        match (&filter.cuts, self.cuts) {
            (Some((cuts, outs)), _) => w.p_boxed += cuts.retain_at(outs, &mut joined) as u64,
            (None, Some(cuts)) => w.p_boxed += cuts.retain_in_box(target, &mut joined) as u64,
            (None, None) => {}
        }
        // An emptied arrival is still buffered where the unfiltered one would
        // have reached the merge, which replays its insertion and removal of
        // an absent cell; see `merge_m2_filtered`.
        let arrives = || match &filter.cuts {
            Some((cuts, outs)) => unfiltered_arrives_at(cuts, outs, f, &corner_front, Energy::ZERO),
            None => self.unfiltered_arrives(target, f, &corner_front, Energy::ZERO),
        };
        if joined.is_empty() && !arrives() {
            return;
        }
        w.p.push((c.eq.to, oc.key, joined));
    }

    /// A stack (`n1 == n2 == 0`) or bulge (one flank empty): the interior
    /// nucleotides do not reach the energy model, so the best-CAI flanks are
    /// jointly optimal and the whole inner front slides by a single `Δ`.
    fn stack_or_bulge_close(
        &self,
        w: &mut FrontWrites,
        c: &SingleClose,
        (f, lows): (&Com, &[f64]),
        oc: &OuterClose,
    ) {
        let Some(left_cai) = self.dfa.broken_codon_cai(c.ep.to, c.key.open) else {
            return;
        };
        let Some(right_cai) = self.dfa.broken_codon_cai(c.o, c.qn) else {
            return;
        };
        let energy = stack_or_bulge_energy(self.params, oc.outer_pair, oc.inner_pair, c.n1, oc.n2);
        let dcai = c.ep.cai + left_cai + right_cai + c.eq.cai;
        // The whole front moves by one `Δ`, so a cut that removes its lowest
        // point at the closed pair removes every point there.
        if let Some(cuts) = self.cuts.filter(|_| !lows.is_empty()) {
            let outs = cuts.outs_at(CellRef::P(c.eq.to, oc.key));
            if cuts.excludes_moved(&outs, lows, energy, dcai) {
                w.single_screened += f.len() as u64;
                return;
            }
        }
        let pair = PairSpell::of(&oc.key);
        let left = Span {
            from: c.ep.to,
            to: c.key.open,
        };
        let right = Span {
            from: c.o,
            to: c.qn,
        };
        let closed = f.shift_map(energy, dcai, |p| Prov::StackBulge {
            pair,
            left,
            right,
            inner: p.back,
        });
        w.p.push((c.eq.to, oc.key, closed));
    }

    // -- Consume `M2[pos]`: collapse to `M`, wrap into `Multi` --------------

    fn m2_consume(&mut self, pos: u32) {
        let t = Instant::now();
        // Serial pass: commit each `M2` core (arena mutation) and collapse it to
        // `M`. Both must stay serial — the arena and the `M` table are shared
        // mutable state. Collect the committed cores for the fat seeding below.
        let mut cores: Vec<(Node, Node, Arc<Com>)> = Vec::new();
        for &jn in self.dfa.nodes_at(pos) {
            // The `M2[pos]` column is consumed once here and never re-read, so
            // move it out rather than cloning each front.
            let cells: Vec<(Node, Acc)> = std::mem::take(&mut self.m2[self.dfa.ix(jn)])
                .into_iter()
                .collect();
            for (inner, f) in &cells {
                self.observe(CellTable::Branch2, branch_span(*inner, jn), f);
                self.census_cell(CellTable::Branch2, Some(CellRef::M2(jn, *inner)), f);
                let core = commit(&mut self.arena, f);
                let as_m = core.shift_map(Energy::ZERO, 0.0, |p| Prov::MFromM2 { m2: p.back });
                self.merge_m(jn, *inner, as_m);
                cores.push((jn, *inner, Arc::new(core)));
            }
        }
        // Fat pass: seed the shared multi-loop openings from each committed core.
        // Reads only the committed core + DFA (no arena) and writes `multi` /
        // `multi_open` — independent across cores, so the fan-out folds them back
        // in core order.
        let t = self.lap(t, |c| &mut c.m2_serial_s);
        self.fan_out(
            FanSite::Seed,
            &cores,
            |(_, _, core): &(Node, Node, Arc<Com>)| core.len(),
            |dp, w, (core_outer, core_inner, core)| {
                dp.seed_multi_open(w, *core_outer, *core_inner, core)
            },
        );
        let groups = far_groups(&cores);
        self.fan_out(
            FanSite::Seed,
            &groups,
            |(_, group): &(Node, Arc<FarCores>)| group.cores.iter().map(|(_, c)| c.len()).sum(),
            |dp, w, (core_outer, group)| dp.seed_far_openings(w, *core_outer, group),
        );
        self.lap(t, |c| &mut c.open_seed_s);
    }

    /// `Multi = (closing pair) [left flank] M2 [right flank]` via the
    /// `multi_open` `C + D` sharing: register one opening per `(left-flank open
    /// node, opening nucleotide)` for every left flank of this `M2` core shorter
    /// than `SINGLE_MAX_LEN`, with `C(core_outer)` — the core's own partial-codon
    /// CAI up to its next boundary — folded in. Frontiers past the
    /// boundary are closed later against the shared opening by a uniform
    /// `D(frontier)` shift; frontiers inside `core_outer`'s codon do not
    /// separate and are closed here directly. Multi-loop unpaired bases are
    /// energy-free (dangles=0), so `Multi` carries only the core's energy.
    ///
    /// The leading run is unbounded, as in the energy model; the longer left
    /// flanks are seeded once for all the cores sharing `core_outer`, by
    /// [`Self::seed_far_openings`], so the work per core does not grow with
    /// the sequence.
    fn seed_multi_open(
        &self,
        w: &mut FrontWrites,
        core_outer: Node,
        core_inner: Node,
        core: &Arc<Com>,
    ) {
        let f: &Com = core;
        let p_lo = core_inner.pos.saturating_sub(SINGLE_MAX_LEN);
        let lows = self.lows_of(f);
        for p in p_lo..core_inner.pos {
            for &pn in self.dfa.nodes_at(p) {
                for &ep in self.dfa.right(pn) {
                    let Some(flank) = self.dfa.broken_codon_cai(ep.to, core_inner) else {
                        continue;
                    };
                    let seed = MultiSeed {
                        core_outer,
                        pn,
                        ep,
                        left_cai: ep.cai + flank,
                    };
                    let left = Span {
                        from: ep.to,
                        to: core_inner,
                    };
                    let source = || OpeningSource::Near {
                        core: Arc::clone(core),
                        inner: core_inner,
                    };
                    self.seed_one_opening(w, &seed, (f, &lows), |p| (left, p.back), source);
                }
            }
        }
    }

    /// The left flanks of `SINGLE_MAX_LEN` or more for every core sharing
    /// `core_outer`, `cores` by descending inner position.
    ///
    /// A flank that long crosses a codon boundary, so its adaptation separates
    /// as [`Dfa::span_head_cai`] of its start plus [`Dfa::span_tail_cai`] of
    /// the core's inner node: one part fixed by the opening, one by the core.
    /// The cores' fronts, each shifted by its own part, merge into one front,
    /// and every opening takes that front shifted by the opening's part — a
    /// uniform shift, so a point the merge drops is one every opening would
    /// have dropped. Sweeping the opening position right to left, a core joins
    /// the front once its flank reaches `SINGLE_MAX_LEN`.
    fn seed_far_openings(&self, w: &mut FrontWrites, core_outer: Node, group: &Arc<FarCores>) {
        let cores = &group.cores;
        let Some(top) = cores.first().map(|(inner, _)| inner.pos) else {
            return;
        };
        let mut far: Front<FarBack> = Front::empty();
        let mut far_lows = Lows::new();
        let mut joined = 0;
        for p in (0..top.saturating_sub(SINGLE_MAX_LEN)).rev() {
            while let Some((inner, core)) = cores.get(joined) {
                let inner = *inner;
                if inner.pos <= p + SINGLE_MAX_LEN {
                    break;
                }
                if let Some(tail) = self.dfa.span_tail_cai(inner) {
                    let shifted = core.shift_map(Energy::ZERO, tail, |c| FarBack {
                        core: c.back,
                        inner,
                    });
                    if let Some(cuts) = self.cuts {
                        cuts.lower_shares(&mut far_lows, &shifted);
                    }
                    far.merge_into(shifted);
                }
                joined += 1;
            }
            if far.is_empty() {
                continue;
            }
            for &pn in self.dfa.nodes_at(p) {
                for &ep in self.dfa.right(pn) {
                    // Every merged core ends at `core_outer`, so an opening
                    // that reaches it reaches them all.
                    if !self.dfa.connected(ep.to, core_outer) {
                        continue;
                    }
                    let Some(head) = self.dfa.span_head_cai(ep.to) else {
                        continue;
                    };
                    let seed = MultiSeed {
                        core_outer,
                        pn,
                        ep,
                        left_cai: ep.cai + head,
                    };
                    let from = ep.to;
                    let split = |c: &Point<FarBack>| {
                        let left = Span {
                            from,
                            to: c.back.inner,
                        };
                        (left, c.back.core)
                    };
                    let source = || OpeningSource::Far {
                        cores: Arc::clone(group),
                        above: p + SINGLE_MAX_LEN,
                    };
                    self.seed_one_opening(w, &seed, (&far, &far_lows), split, source);
                }
            }
        }
    }

    /// Register one `(M2 core, left flank)` opening: close the non-separating
    /// short flanks directly, then fold `C(core_outer)` into the shared opening
    /// front and either activate it now (the core already ends on a codon
    /// boundary) or defer it to that boundary. `split` reads each point's left
    /// flank and core; `lows` bounds `f`'s shares under the box's cuts.
    ///
    /// Under box cuts the opening is written as a recipe (`source`) with its
    /// bound, and built only if a close might keep one of its points.
    fn seed_one_opening<B>(
        &self,
        w: &mut FrontWrites,
        seed: &MultiSeed,
        (f, lows): (&Front<B>, &[f64]),
        split: impl Fn(&Point<B>) -> (Span, ProvIdx) + Copy,
        source: impl FnOnce() -> OpeningSource,
    ) {
        let boundary = self.dfa.next_codon_boundary(seed.core_outer.pos);
        self.seed_short_flanks(w, seed, (f, lows), boundary, split);
        let Some(c_outer) = self.dfa.span_head_cai(seed.core_outer) else {
            return;
        };
        let key = MultiOpenKey {
            open: seed.pn,
            nuc_open: seed.ep.nuc,
            lane_open: seed.ep.to.lane,
            reach: self.dfa.reach_class(seed.core_outer),
        };
        let core_outer = seed.core_outer;
        let dcai = seed.left_cai + c_outer;
        let opening = match self.cuts {
            Some(cuts) => {
                let mut bound = Lows::new();
                cuts.lower_moved(&mut bound, lows, dcai);
                Opening::Lazy(LazyOpening {
                    source: source(),
                    from: seed.ep.to,
                    core_outer,
                    dcai,
                    lows: bound,
                })
            }
            None => Opening::Built(f.shift_map(Energy::ZERO, dcai, |p| {
                let (left, core) = split(p);
                Prov::MultiOpen {
                    left,
                    core,
                    core_outer,
                }
            })),
        };
        if boundary == seed.core_outer.pos {
            w.multi_open.push((key, opening));
        } else {
            w.multi_open_pending.push((boundary as usize, key, opening));
        }
    }

    /// Close the short (same-codon-as-`core_outer`) right flanks directly with
    /// the exact broken-codon CAI — these frontiers do not admit the `C + D`
    /// separation the shared opening relies on. A close a cut removes whole at
    /// its `Multi` cell, read off `lows`, is skipped as the shared closes are.
    fn seed_short_flanks<B>(
        &self,
        w: &mut FrontWrites,
        seed: &MultiSeed,
        (f, lows): (&Front<B>, &[f64]),
        boundary: u32,
        split: impl Fn(&Point<B>) -> (Span, ProvIdx) + Copy,
    ) {
        for q in seed.core_outer.pos..boundary {
            for &qn in self.dfa.nodes_at(q) {
                let Some(right_cai) = self.dfa.broken_codon_cai(seed.core_outer, qn) else {
                    continue;
                };
                for &eq in self.dfa.right(qn) {
                    if canonical(seed.ep.nuc, eq.nuc).is_none() {
                        continue;
                    }
                    let key = PairKey {
                        open: seed.pn,
                        close: qn,
                        opening: seed.ep.opening,
                        nuc_close: eq.nuc,
                    };
                    if self.closes_forbidden(key) {
                        continue;
                    }
                    let right = Span {
                        from: seed.core_outer,
                        to: qn,
                    };
                    let dcai = seed.left_cai + right_cai + eq.cai;
                    if let Some(cuts) = self.cuts.filter(|_| !lows.is_empty()) {
                        let outs = cuts.outs_at(CellRef::Multi(eq.to, key));
                        if cuts.excludes_shifted(&outs, lows, dcai) {
                            w.short_screened += f.len() as u64;
                            continue;
                        }
                    }
                    let wrapped = f.shift_map(Energy::ZERO, dcai, |p| {
                        let (left, core) = split(p);
                        Prov::MultiWrap { left, right, core }
                    });
                    w.multi.push((eq.to, key, wrapped));
                }
            }
        }
    }

    /// Merge the openings deferred to `pos` into the shared `multi_open` table,
    /// so they become closeable against frontiers at/after `pos`.
    fn multi_open_activate(&mut self, pos: u32) {
        let pending = std::mem::take(&mut self.multi_open_pending[pos as usize]);
        self.merge_openings(pending);
    }

    /// Close every active opening against each frontier node at `pos`, paying
    /// the frontier's broken-codon CAI `D(qn)` plus the closing edge — a uniform
    /// shift over each opening front (`O(k̄)`, no Minkowski blow-up), completing
    /// the right flank `core_outer → qn` in each point's provenance.
    fn multi_open_close(&mut self, pos: u32) {
        if self.multi_open.is_empty() {
            return;
        }
        // The shared openings persist across frontier positions, so they cannot
        // be consumed — but they are only read here (the closes write into
        // `multi`, a different table). Move the whole table out for the duration
        // of the scan and restore it, so each opening front is borrowed in place
        // rather than cloned once per frontier node.
        let openings = std::mem::take(&mut self.multi_open);
        let fronts = std::mem::take(&mut self.multi_open_fronts);
        // Each frontier node's broken-codon CAI `D(qn)`, computed once.
        let frontier: Vec<(Node, f64)> = self
            .dfa
            .nodes_at(pos)
            .iter()
            .filter_map(|&qn| self.dfa.span_tail_cai(qn).map(|d_qn| (qn, d_qn)))
            .collect();
        let items: Vec<(MultiOpenKey, &Acc)> = openings
            .iter()
            .map(|(k, &slot)| (*k, &fronts[slot]))
            .collect();
        self.count_open_close(&items, &frontier);
        let lows = std::mem::take(&mut self.multi_open_lows);
        let lazy = std::mem::take(&mut self.multi_open_lazy);
        let closes: Vec<OpenClose> = openings
            .iter()
            .map(|(&key, &slot)| OpenClose {
                key,
                slot,
                opening: &fronts[slot],
                lows: &lows[slot],
                lazy: &lazy[slot],
            })
            .collect();
        let (boxed, dropped) = (
            self.boxed[CellTable::Multi.index()],
            self.dropped[CellTable::Multi.index()],
        );
        {
            // Fan out over the openings: each closes against every frontier node
            // into a **distinct** `multi` cell (the cell key carries the
            // opening's open node + the closing frontier, so no two openings and
            // no two frontiers collide), so the writes never contend. Their
            // order still matters: it is the order cells are inserted into the
            // `multi` columns, whose iteration order sets `multi_close`'s
            // arrival order at `P`. Reading the openings in key order instead
            // left every front identical but moved the drop tallies, so the
            // chunk order is kept.
            let size = |c: &OpenClose| c.opening.len();
            self.fan_out(FanSite::Close, &closes, size, |dp, w, c| {
                dp.close_opening_against(w, c, &frontier)
            });
        }
        self.clock.open_close_boxed += self.boxed[CellTable::Multi.index()] - boxed;
        self.clock.open_close_dropped += self.dropped[CellTable::Multi.index()] - dropped;
        self.multi_open = openings;
        self.multi_open_fronts = fronts;
        self.multi_open_lows = lows;
        self.multi_open_lazy = lazy;
        self.materialize_openings();
    }

    /// Count what the held openings are and what closing them against
    /// `frontier` produces: every opening point once per canonical closing
    /// edge whose nucleotide pairs with the opening's.
    fn count_open_close(&mut self, openings: &[(MultiOpenKey, &Acc)], frontier: &[(Node, f64)]) {
        let edges_for = |open: Nucleotide| -> u64 {
            frontier
                .iter()
                .flat_map(|&(qn, _)| self.dfa.right(qn))
                .filter(|eq| canonical(open, eq.nuc).is_some())
                .count() as u64
        };
        let edges = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U].map(edges_for);
        for (key, opening) in openings {
            let held = opening.len() as u64;
            self.clock.open_keys_held += 1;
            self.clock.open_points_held += held;
            self.clock.open_close_points += held * edges[key.nuc_open as usize];
        }
    }

    /// Whether a multi-loop closed by `key`'s pair is one the region constraint
    /// forbids. Such a `Multi` cell is read only by [`Self::multi_close`], whose
    /// merge into `P` drops a forbidden pair's whole front, so building it —
    /// copying the wrapped front, buffering it, committing its points to the
    /// arena — holds memory for points no front can contain. Behind a long
    /// flank most closes are of this kind: a pair from the coding sequence to
    /// the 3′ flank is forbidden under the default region constraint.
    fn closes_forbidden(&self, key: PairKey) -> bool {
        matches!(
            self.region.verdict(key.open.pos, key.close.pos),
            Verdict::Forbid
        )
    }

    /// Close one shared opening against every frontier node, appending the
    /// completed `Multi` wrap (`core_outer → qn` right flank) for each canonical
    /// closing edge. Every `(opening, frontier, edge)` targets a distinct `multi`
    /// cell, so these appends are order-independent.
    fn close_opening_against(&self, w: &mut FrontWrites, c: &OpenClose, frontier: &[(Node, f64)]) {
        let OpenClose { key, lazy, .. } = *c;
        // The slot's recipes, built only for a close the screen lets through,
        // and then built in place for later positions too.
        let mut built: Option<Acc> = None;
        for &(qn, d_qn) in frontier {
            if !self.dfa.reaches(key.reach, qn) {
                continue;
            }
            for &eq in self.dfa.right(qn) {
                if canonical(key.nuc_open, eq.nuc).is_none() {
                    continue;
                }
                let pk = PairKey {
                    open: key.open,
                    close: qn,
                    opening: crate::state::Opening::new(key.nuc_open, key.lane_open),
                    nuc_close: eq.nuc,
                };
                if self.closes_forbidden(pk) {
                    continue;
                }
                // A cut that removes the opening's lowest point at the closed
                // cell removes the whole shifted front there, so the close
                // writes nothing the merge would keep.
                if let Some(cuts) = self.cuts.filter(|_| !c.lows.is_empty()) {
                    let outs = cuts.outs_at(CellRef::Multi(eq.to, pk));
                    if cuts.excludes_shifted(&outs, c.lows, d_qn + eq.cai) {
                        w.multi_screened += c.opening.len() as u64;
                        continue;
                    }
                }
                let opening = if lazy.is_empty() {
                    c.opening
                } else {
                    built.get_or_insert_with(|| {
                        w.materialize.push(c.slot);
                        self.built_opening(c.opening, lazy).0
                    })
                };
                // The count before the closes read only what was built then.
                w.close_extra += opening.len().saturating_sub(c.opening.len()) as u64;
                let closed =
                    opening.shift_map(Energy::ZERO, d_qn + eq.cai, |p| close_opening(p, qn));
                w.multi.push((eq.to, pk, closed));
            }
        }
    }

    /// A shared opening pools openings from different cores and boundaries
    /// under one key, so it has no single cost to go and keeps the window alone
    /// — the sound fallback for a table the exact credit does not cover.
    ///
    /// `arrivals` are merged in their order: each key's slot is found or made
    /// serially, in that order, so the table's keys are inserted exactly as
    /// one merge at a time would insert them. A large batch then merges its
    /// fronts on the pooled workers, one key per worker at a time, each key's
    /// arrivals in order; a small one merges them here.
    fn merge_openings(&mut self, arrivals: Vec<(MultiOpenKey, Opening)>) {
        if arrivals.is_empty() {
            return;
        }
        let t = Instant::now();
        let mut built = Vec::new();
        for (key, opening) in arrivals {
            match opening {
                Opening::Built(f) => built.push((key, f)),
                Opening::Lazy(lazy) => {
                    let slot = self.opening_slot(key);
                    if let Some(cuts) = self.cuts {
                        cuts.lower_moved(&mut self.multi_open_lows[slot], &lazy.lows, 0.0);
                    }
                    self.clock.open_recipes += 1;
                    self.multi_open_lazy[slot].push(lazy);
                }
            }
        }
        self.merge_built_openings(built);
        self.lap(t, |c| &mut c.open_merge_s);
    }

    /// [`Self::merge_openings`]' built arrivals, merged into their slots.
    fn merge_built_openings(&mut self, arrivals: Vec<(MultiOpenKey, Acc)>) {
        if arrivals.is_empty() {
            return;
        }
        let rule = self.open_merge_rule();
        match self.pool {
            Some(pool) if arrivals.len() >= OPEN_SHARD_MIN => {
                self.merge_openings_sharded(pool, rule, arrivals);
            }
            _ => {
                for (key, f) in arrivals {
                    self.clock.open_arrivals += f.len() as u64;
                    let slot = self.opening_slot(key);
                    if let Some(cuts) = self.cuts {
                        cuts.lower_shares(&mut self.multi_open_lows[slot], &f);
                    }
                    let dropped = rule.merge(&mut self.multi_open_fronts[slot], f);
                    self.tally_opening(dropped);
                }
            }
        }
    }

    /// The merge every shared opening's front takes its arrivals by.
    fn open_merge_rule(&self) -> OpenMerge {
        OpenMerge {
            prune: self.prune,
            slack: self.windows.beyond_exterior,
        }
    }

    /// `front` with every recipe in `lazy` built and merged in, in order, as
    /// the arrivals would have been merged had they come built; and the points
    /// the recipes built.
    fn built_opening(&self, front: &Acc, lazy: &[LazyOpening]) -> (Acc, u64) {
        let rule = self.open_merge_rule();
        let mut built = front.clone();
        let mut arrivals = 0;
        for recipe in lazy {
            let f = recipe.build(self.dfa);
            arrivals += f.len() as u64;
            rule.merge(&mut built, f);
        }
        (built, arrivals)
    }

    /// Build in place every slot a close had to build from recipes.
    fn materialize_openings(&mut self) {
        let mut slots = std::mem::take(&mut self.to_materialize);
        slots.sort_unstable();
        slots.dedup();
        for slot in slots {
            self.materialize_slot(slot);
        }
    }

    fn materialize_slot(&mut self, slot: usize) {
        let lazy = std::mem::take(&mut self.multi_open_lazy[slot]);
        if lazy.is_empty() {
            return;
        }
        self.clock.open_built += lazy.len() as u64;
        let (built, arrivals) = self.built_opening(&self.multi_open_fronts[slot], &lazy);
        self.clock.open_arrivals += arrivals;
        self.multi_open_fronts[slot] = built;
    }

    /// The slot of `key`'s front, inserting the key with an empty front if it
    /// is new.
    fn opening_slot(&mut self, key: MultiOpenKey) -> usize {
        let (fronts, lows, lazy) = (
            &mut self.multi_open_fronts,
            &mut self.multi_open_lows,
            &mut self.multi_open_lazy,
        );
        *self.multi_open.entry(key).or_insert_with(|| {
            fronts.push(Acc::empty());
            lows.push(Lows::new());
            lazy.push(Vec::new());
            fronts.len() - 1
        })
    }

    fn tally_opening(&mut self, dropped: usize) {
        self.tally(
            CellTable::MultiOpen,
            Merged {
                dropped,
                emptied: false,
                boxed: 0,
            },
        );
    }

    /// [`Self::merge_openings`] on the pooled workers: the keys' slots made
    /// serially in arrival order, the arrivals bucketed by slot in that order,
    /// and the slots dealt heaviest first.
    fn merge_openings_sharded(
        &mut self,
        pool: &Pool,
        rule: OpenMerge,
        arrivals: Vec<(MultiOpenKey, Acc)>,
    ) {
        let mut free = std::mem::take(&mut self.open_shards);
        let mut job_of: FxHashMap<usize, usize> = FxHashMap::default();
        let mut buckets: Vec<(usize, Vec<Acc>)> = Vec::new();
        for (key, f) in arrivals {
            self.clock.open_arrivals += f.len() as u64;
            let slot = self.opening_slot(key);
            let job = *job_of.entry(slot).or_insert_with(|| {
                buckets.push((slot, free.pop().unwrap_or_default()));
                buckets.len() - 1
            });
            buckets[job].1.push(f);
        }
        buckets.sort_by_key(|(slot, items)| (std::cmp::Reverse(items.len()), *slot));
        let jobs: Vec<Mutex<OpenJob>> = buckets
            .into_iter()
            .map(|(slot, items)| {
                Mutex::new(OpenJob {
                    slot,
                    front: std::mem::take(&mut self.multi_open_fronts[slot]),
                    lows: std::mem::take(&mut self.multi_open_lows[slot]),
                    items,
                    dropped: 0,
                })
            })
            .collect();
        let next = AtomicUsize::new(0);
        let cuts = self.cuts;
        pool.run(self.threads.min(jobs.len()), &|_| loop {
            let j = next.fetch_add(1, Ordering::Relaxed);
            let Some(job) = jobs.get(j) else { break };
            let mut constraint = job.lock().unwrap();
            let OpenJob {
                front,
                lows,
                items,
                dropped,
                ..
            } = &mut *constraint;
            for f in items.drain(..) {
                if let Some(cuts) = cuts {
                    cuts.lower_shares(lows, &f);
                }
                *dropped += rule.merge(front, f);
            }
        });
        for job in jobs {
            let job = job.into_inner().unwrap();
            self.multi_open_fronts[job.slot] = job.front;
            self.multi_open_lows[job.slot] = job.lows;
            self.tally_opening(job.dropped);
            free.push(job.items);
        }
        self.open_shards = free;
    }

    // -- `M` / `C` unpaired extension --------------------------------------

    /// Finalize each `M[pos]` cell (commit it for later bifurcations to read as
    /// a left branch) and extend it by one unpaired 3′ nucleotide.
    fn m_extend(&mut self, pos: u32) {
        for &jn in self.dfa.nodes_at(pos) {
            // The `M[pos]` column is finalized here (committed into `m_committed`
            // for later bifurcations) and never read from `m` again, so move it
            // out rather than cloning each front.
            let cells: Vec<(Node, Acc)> = std::mem::take(&mut self.m[self.dfa.ix(jn)])
                .into_iter()
                .collect();
            let edges = self.dfa.right(jn).to_vec();
            for (inner, f) in &cells {
                let counts = self.observe(CellTable::Branch, branch_span(*inner, jn), f);
                self.keep_branch(jn, *inner, counts);
                let survivors =
                    self.census_cell(CellTable::Branch, Some(CellRef::M(jn, *inner)), f);
                if let Some(state) = &mut self.census {
                    state.m[self.dfa.ix(jn)].insert(*inner, survivors);
                }
                let committed = commit(&mut self.arena, f);
                self.m_committed[self.dfa.ix(jn)].insert(*inner, committed.clone());
                for e in &edges {
                    let ext = committed.shift_map(Energy::ZERO, e.cai, |p| Prov::MUnpaired {
                        prev: p.back,
                        pos: jn.pos,
                        nuc: e.nuc,
                    });
                    self.merge_m(e.to, *inner, ext);
                }
            }
        }
    }

    /// Finalize each `C[pos]` cell (commit it for later pairs to read as a 5′
    /// prefix) and extend it by one unpaired exterior nucleotide.
    fn c_extend(&mut self, pos: u32) {
        for &jn in self.dfa.nodes_at(pos) {
            // `C[pos]` is finalized into `c_committed` here and never read from
            // `c` again (the terminal `C[end]` is not extended, so it is never
            // taken), so move it out rather than cloning.
            let f = std::mem::take(&mut self.c[self.dfa.ix(jn)]);
            let counts = self.observe(CellTable::Exterior, jn.pos, &f);
            self.keep_exterior(jn, counts);
            let survivors = self.census_cell(CellTable::Exterior, Some(CellRef::C(jn)), &f);
            if let Some(state) = &mut self.census {
                state.c[self.dfa.ix(jn)] = survivors;
            }
            let committed = commit(&mut self.arena, &f);
            self.c_committed[self.dfa.ix(jn)] = committed.clone();
            if committed.is_empty() {
                continue;
            }
            for e in self.dfa.right(jn).to_vec() {
                let ext = committed.shift_map(Energy::ZERO, e.cai, |p| Prov::ExtUnpaired {
                    prev: p.back,
                    pos: jn.pos,
                    nuc: e.nuc,
                });
                self.merge_c(e.to, ext);
            }
        }
    }

    // -- Fat-phase fan-out -------------------------------------------------

    /// Run a fat production phase over `items`: each item produces table-write
    /// candidates into a [`FrontWrites`] via `produce`, and the buffers are then
    /// folded back into the tables. Serial when `threads <= 1` or the phase is
    /// small; otherwise the independent per-item work fans across scoped threads
    /// and the per-chunk partials are folded in **chunk-index order** — the exact
    /// serial production order, so the fold is byte-identical (objectives *and*
    /// recovered sequences) at every thread count.
    ///
    /// Every `produce` call reads shared-immutable [`Dp`] state (`dfa`, `params`,
    /// committed fronts passed inside `items`) and touches no arena — the arena is
    /// mutated only in the serial consume phases, before/between these fan-outs.
    fn fan_out<T: Sync>(
        &mut self,
        site: FanSite,
        items: &[T],
        weight: impl Fn(&T) -> usize + Sync,
        produce: impl Fn(&Self, &mut FrontWrites, &T) + Sync,
    ) {
        let t = Instant::now();
        if self.threads <= 1 || items.len() < FAN_OUT_MIN {
            let mut w = FrontWrites::default();
            for it in items {
                produce(self, &mut w, it);
            }
            let held = self.fan_detail.then(|| w.held());
            w.apply(self);
            if let Some(held) = held {
                let tally = self.clock.fan.site_mut(site);
                tally.serial += 1;
                tally.serial_s += t.elapsed().as_secs_f64();
                tally.hold(items.len(), held);
            }
            return;
        }
        let (parts, times) = self.produce_parallel(items, &weight, &produce);
        if self.fan_detail {
            self.clock.fan.record_fanned(site, t.elapsed(), &times);
            let tally = self.clock.fan.site_mut(site);
            for w in &parts {
                tally.hold(0, w.held());
            }
            tally.items += items.len() as u64;
        }
        let t = Instant::now();
        // The partials' openings, in partial order, merged as one batch after
        // the other tables, which they do not touch.
        let mut parts = parts;
        let openings: Vec<(MultiOpenKey, Opening)> = parts
            .iter_mut()
            .flat_map(|w| std::mem::take(&mut w.multi_open))
            .collect();
        match self.pool {
            Some(pool) if site == FanSite::Push && self.serial_apply_free() => {
                self.apply_push_sharded(pool, parts);
            }
            _ => {
                for w in parts {
                    w.apply(self);
                }
            }
        }
        self.merge_openings(openings);
        if self.fan_detail {
            self.clock.fan.site_mut(site).apply_s += t.elapsed().as_secs_f64();
        }
    }

    /// Whether a fan-out's apply may leave the driver thread: no probe or
    /// census is attached, since their observers count in the serial order.
    fn serial_apply_free(&self) -> bool {
        self.probe.is_none() && self.census.is_none()
    }

    /// Fan `items` across up to `threads` of the fold's pooled workers, each
    /// stealing [`FAN_OUT_CHUNK`] ranges, each chunk accumulating a private [`FrontWrites`] tagged with its
    /// chunk index. Returns the partials sorted by chunk index, so the caller
    /// folds them in the serial production order.
    fn produce_parallel<T: Sync, W: Default + Send>(
        &self,
        items: &[T],
        weight: &(impl Fn(&T) -> usize + Sync),
        produce: &(impl Fn(&Self, &mut W, &T) + Sync),
    ) -> (Vec<W>, WorkerTimes) {
        let counter = AtomicUsize::new(0);
        let units = deal_units(items, weight, self.threads);
        let n_chunks = units.len();
        let cap = self.fan_cap.unwrap_or(usize::MAX);
        let n_workers = self.threads.min(n_chunks).min(cap).max(1);
        let timed = self.fan_detail;
        let mut times = WorkerTimes {
            chunks: n_chunks,
            ..WorkerTimes::default()
        };
        type Worked<W> = (Vec<(usize, W)>, Option<(Duration, Duration, Duration)>);
        let slots: Vec<Mutex<Option<Worked<W>>>> =
            (0..n_workers).map(|_| Mutex::new(None)).collect();
        let work = |worker: usize| {
            let mut local: Vec<(usize, W)> = Vec::new();
            let began = timed.then(Instant::now);
            let (mut longest, mut heaviest) = (Duration::ZERO, Duration::ZERO);
            loop {
                let u = counter.fetch_add(1, Ordering::Relaxed);
                let Some(&(start, end)) = units.get(u) else {
                    break;
                };
                let chunk = start;
                let chunk_began = timed.then(Instant::now);
                let mut w = W::default();
                for it in &items[start..end] {
                    let item_began = timed.then(Instant::now);
                    produce(self, &mut w, it);
                    if let Some(t) = item_began {
                        heaviest = heaviest.max(t.elapsed());
                    }
                }
                if let Some(t) = chunk_began {
                    longest = longest.max(t.elapsed());
                }
                local.push((chunk, w));
            }
            let busy = began.map(|t| (t.elapsed(), longest, heaviest));
            *slots[worker].lock().unwrap() = Some((local, busy));
        };
        match self.pool {
            Some(pool) => pool.run(n_workers, &work),
            None => (0..n_workers).for_each(work),
        }
        let mut tagged = Vec::new();
        for slot in slots {
            let (local, busy) = slot.into_inner().unwrap().expect("every worker ran");
            tagged.extend(local);
            if let Some((busy, longest, heaviest)) = busy {
                times.busy.push(busy);
                times.chunk_max = times.chunk_max.max(longest);
                times.item_max = times.item_max.max(heaviest);
            }
        }
        tagged.sort_by_key(|(chunk, _)| *chunk);
        (tagged.into_iter().map(|(_, w)| w).collect(), times)
    }

    // -- Read-only observation ---------------------------------------------

    /// Record a finalized cell and return its per-budget retention counts,
    /// empty when no probe is attached.
    fn observe<B>(&mut self, table: CellTable, inside_nt: u32, front: &Front<B>) -> RetainedCounts {
        match &mut self.probe {
            Some(state) => state.probe.observe_cell(table, inside_nt, front),
            None => RetainedCounts::default(),
        }
    }

    /// Hold `counts` as the cell the serial consume phases are composing.
    fn hold(&mut self, counts: RetainedCounts) {
        if let Some(state) = &mut self.probe {
            state.current = counts;
        }
    }

    /// Keep a finalized exterior cell's counts for the pairs that will absorb
    /// into it.
    fn keep_exterior(&mut self, node: Node, counts: RetainedCounts) {
        if let Some(state) = &mut self.probe {
            state.c[self.dfa.ix(node)] = counts;
        }
    }

    /// Keep a finalized multi-branch cell's counts for the bifurcations that
    /// will read it as a left branch.
    fn keep_branch(&mut self, outer: Node, inner: Node, counts: RetainedCounts) {
        if let Some(state) = &mut self.probe {
            state.m[self.dfa.ix(outer)].insert(inner, counts);
        }
    }

    /// Charge the product that absorbs the held pair into the exterior prefix
    /// at `prefix`.
    fn observe_absorb(&mut self, prefix: Node) {
        let Some(ProbeState {
            probe, current, c, ..
        }) = &mut self.probe
        else {
            return;
        };
        probe.observe_product(
            ComposeSite::ExteriorAbsorb,
            &c[self.dfa.ix(prefix)],
            current,
        );
    }

    /// Charge the product that bifurcates the held pair against `M[outer][inner]`.
    fn observe_bifurcation(&mut self, outer: Node, inner: Node) {
        let Some(ProbeState {
            probe, current, m, ..
        }) = &mut self.probe
        else {
            return;
        };
        if let Some(left) = m[self.dfa.ix(outer)].get(&inner) {
            probe.observe_product(ComposeSite::Bifurcation, left, current);
        }
    }

    /// Hand a worker the held pair so its single-branch closures can charge
    /// their corner products against it.
    fn hand_to_worker(&self, w: &mut FrontWrites, inside_nt: u32, f: &Com) {
        let Some(state) = &self.probe else {
            return;
        };
        let probe = w.probe.get_or_insert_with(|| state.probe.split());
        probe.latch(CellTable::Pair, inside_nt, f);
    }

    /// Record the final exterior cell, whose front is the molecule's complete
    /// staircase and whose retention under a budget is therefore the feasible
    /// set itself — the one cell whose answer is known independently of the
    /// instrument.
    fn observe_final_exterior(&mut self) {
        let end = std::mem::take(&mut self.c[self.dfa.ix(self.dfa.end())]);
        if let Some(state) = &mut self.probe {
            state.probe.observe_root(&end);
        }
        let root = self.census_cell(CellTable::Exterior, Some(CellRef::C(self.dfa.end())), &end);
        if let Some(state) = &mut self.census {
            state.tally.observe_root(&root);
        }
        self.c[self.dfa.ix(self.dfa.end())] = end;
    }

    /// Record the shared multi-loop openings.
    ///
    /// Unlike every other table these persist to the end of the fold instead of
    /// being consumed at a position, so they are counted once here. An opening
    /// has no settled extent — it is closed later against a frontier that is not
    /// yet chosen — so it is charged as accounting for no nucleotides, which for
    /// [`DropRule::Relaxation`] credits the largest possible outside and so
    /// keeps the most points. The understatement is confined to that one rule on
    /// this one table; the window rules do not consult the extent at all.
    fn observe_shared_openings(&mut self) {
        if self.probe.is_none() && self.census.is_none() {
            return;
        }
        // An observer counts the openings as held, recipes built.
        for slot in 0..self.multi_open_lazy.len() {
            self.materialize_slot(slot);
        }
        let openings = std::mem::take(&mut self.multi_open);
        let fronts = std::mem::take(&mut self.multi_open_fronts);
        for &slot in openings.values() {
            self.observe(CellTable::MultiOpen, 0, &fronts[slot]);
            self.census_cell(CellTable::MultiOpen, None, &fronts[slot]);
        }
        self.multi_open = openings;
        self.multi_open_fronts = fronts;
    }

    /// Count a finalized cell against the census's cuts and return its
    /// counts, empty when no census is attached.
    fn census_cell<B>(
        &mut self,
        table: CellTable,
        cell: Option<CellRef>,
        front: &Front<B>,
    ) -> Survivors {
        let Some(state) = &mut self.census else {
            return Survivors::new();
        };
        let survivors = state.census.survivors(cell, front);
        state.tally.observe_cell(table, &survivors);
        if let Some(cell) = cell {
            state.census.check_inside(cell, front, &mut state.tally);
        }
        survivors
    }

    /// Hold `survivors` as the closed pair the serial consume phases compose.
    fn census_hold(&mut self, survivors: Survivors) {
        if let Some(state) = &mut self.census {
            state.current = survivors;
        }
    }

    /// Charge the absorption of the held pair into the exterior prefix.
    fn census_absorb(&mut self, prefix: Node) {
        if let Some(state) = &mut self.census {
            let left = &state.c[self.dfa.ix(prefix)];
            state
                .tally
                .observe_product(ComposeSite::ExteriorAbsorb, left, &state.current);
        }
    }

    /// Charge the bifurcation of the held pair against `M[outer][inner]`.
    fn census_bifurcation(&mut self, outer: Node, inner: Node) {
        if let Some(state) = &mut self.census {
            if let Some(left) = state.m[self.dfa.ix(outer)].get(&inner) {
                state
                    .tally
                    .observe_product(ComposeSite::Bifurcation, left, &state.current);
            }
        }
    }

    /// Hand a worker the counts of the closed pair its single-branch closures
    /// compose, recomputed there since the cell was counted in the serial pass.
    fn census_latch(&self, w: &mut FrontWrites, cell: CellRef, f: &Com) {
        let Some(state) = &self.census else {
            return;
        };
        let survivors = state.census.survivors(Some(cell), f);
        match &mut w.census {
            Some((held, _)) => *held = survivors,
            None => w.census = Some((survivors, state.census.tally())),
        }
    }

    // -- `⊕` accumulators into each table ----------------------------------

    fn merge_p(&mut self, outer: Node, key: PairKey, f: Acc) {
        let rule = self.p_merge();
        let merged = rule.merge(&mut self.p[self.dfa.ix(outer)], outer, key, f);
        self.tally(CellTable::Pair, merged);
    }

    /// What a merge into a `P` cell reads, apart from the cell's column.
    fn p_merge(&self) -> PMerge<'a> {
        PMerge {
            prune: self.prune,
            slack: self.windows.beyond_exterior,
            credit: self.credit,
            cuts: self.cuts,
            region: self.region,
        }
    }

    /// Fold the push's partials into the `P` columns on the pooled workers.
    /// The candidates are bucketed by target column in the partials' order
    /// and each partial's production order, and the workers take whole
    /// columns, heaviest first, from a shared counter. One worker merges all
    /// of a column's candidates in that order, so every column sees the serial
    /// sequence of inserts and removals and ends in the same state, hash table
    /// included. The other tables a partial carries are folded serially, as
    /// before.
    fn apply_push_sharded(&mut self, pool: &Pool, parts: Vec<FrontWrites>) {
        let bucketing = Instant::now();
        let mut free = std::mem::take(&mut self.p_shards);
        let mut slot_of = vec![usize::MAX; self.p.len()];
        let mut buckets: Vec<(usize, Vec<PCandidate>)> = Vec::new();
        for mut w in parts {
            for c in w.p.drain(..) {
                let column = self.dfa.ix(c.0);
                if slot_of[column] == usize::MAX {
                    slot_of[column] = buckets.len();
                    buckets.push((column, free.pop().unwrap_or_default()));
                }
                buckets[slot_of[column]].1.push(c);
            }
            w.apply(self);
        }
        buckets.sort_by_key(|(column, items)| (std::cmp::Reverse(items.len()), *column));
        let rule = self.p_merge();
        let mut maps: Vec<Option<&mut FxHashMap<PairKey, Acc>>> =
            self.p.iter_mut().map(Some).collect();
        let jobs: Vec<Mutex<PColumn>> = buckets
            .into_iter()
            .map(|(column, items)| {
                Mutex::new(PColumn {
                    column: maps[column].take().expect("each column is bucketed once"),
                    items,
                    merged: [0; 3],
                })
            })
            .collect();
        let workers = self.threads.min(jobs.len());
        let busy: Vec<Mutex<Duration>> = (0..workers).map(|_| Mutex::new(Duration::ZERO)).collect();
        let next = AtomicUsize::new(0);
        let bucket = bucketing.elapsed();
        let merging = Instant::now();
        pool.run(workers, &|w| {
            let began = Instant::now();
            loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(j) else { break };
                let mut constraint = job.lock().unwrap();
                let PColumn {
                    column,
                    items,
                    merged,
                } = &mut *constraint;
                for (outer, key, f) in items.drain(..) {
                    let m = rule.merge(column, outer, key, f);
                    merged[0] += m.dropped as u64;
                    merged[1] += u64::from(m.emptied);
                    merged[2] += m.boxed as u64;
                }
            }
            *busy[w].lock().unwrap() = began.elapsed();
        });
        let merge = merging.elapsed();
        let table = CellTable::Pair.index();
        for job in jobs {
            let job = job.into_inner().unwrap();
            self.dropped[table] += job.merged[0];
            self.emptied[table] += job.merged[1];
            self.boxed[table] += job.merged[2];
            free.push(job.items);
        }
        self.p_shards = free;
        if self.fan_detail {
            let busy: Vec<Duration> = busy.into_iter().map(|b| b.into_inner().unwrap()).collect();
            self.clock.fan.record_sharded_apply(bucket, merge, &busy);
        }
    }

    fn merge_multi(&mut self, outer: Node, key: PairKey, f: Acc) {
        let slack = self.windows.beyond_exterior;
        let cell = CellRef::Multi(outer, key);
        let exact = self.exact(cell, slack);
        let boxed = self.boxed(cell);
        let merged = merge_keyed(
            &mut self.multi[self.dfa.ix(outer)],
            key,
            f,
            self.prune,
            slack,
            exact,
            boxed,
        );
        self.tally(CellTable::Multi, merged);
    }

    fn merge_m(&mut self, outer: Node, inner: Node, f: Acc) {
        let slack = self.windows.beyond_exterior;
        let cell = CellRef::M(outer, inner);
        let exact = self.exact(cell, slack);
        let boxed = self.boxed(cell);
        let merged = merge_keyed(
            &mut self.m[self.dfa.ix(outer)],
            inner,
            f,
            self.prune,
            slack,
            exact,
            boxed,
        );
        self.tally(CellTable::Branch, merged);
    }

    fn merge_m2(&mut self, outer: Node, inner: Node, f: Acc) {
        let slack = self.windows.beyond_exterior;
        let cell = CellRef::M2(outer, inner);
        let exact = self.exact(cell, slack);
        let boxed = self.boxed(cell);
        let merged = merge_keyed(
            &mut self.m2[self.dfa.ix(outer)],
            inner,
            f,
            self.prune,
            slack,
            exact,
            boxed,
        );
        self.tally(CellTable::Branch2, merged);
    }

    /// The exterior cell stays in place when emptied: it is a dense column,
    /// and an empty front composes to nothing downstream.
    fn merge_c(&mut self, node: Node, mut f: Acc) {
        let slack = self.windows.exterior;
        let exact = self.exact(CellRef::C(node), slack);
        let boxed = match self.boxed(CellRef::C(node)) {
            Some((cuts, cell)) => cuts.retain_in_box(cell, &mut f),
            None => 0,
        };
        let cell = &mut self.c[self.dfa.ix(node)];
        cell.merge_into(f);
        prune_in_place(cell, self.prune);
        let dropped = retain_within(cell, slack, exact);
        let emptied = exact.is_some() && cell.is_empty();
        self.tally(
            CellTable::Exterior,
            Merged {
                dropped,
                emptied,
                boxed,
            },
        );
    }

    /// The exact cutoff for `cell` under `slack`, where a credit is carried.
    fn exact(&self, cell: CellRef, slack: Slack) -> Option<Energy> {
        self.credit.and_then(|credit| credit.cutoff(cell, slack))
    }

    /// The box's cuts and the cell they are tested at, where a box is applied.
    fn boxed(&self, cell: CellRef) -> Option<(&'a BoxCuts, CellRef)> {
        self.cuts.map(|cuts| (cuts, cell))
    }

    /// Record what one merge at `table` discarded.
    fn tally(&mut self, table: CellTable, merged: Merged) {
        self.dropped[table.index()] += merged.dropped as u64;
        self.emptied[table.index()] += u64::from(merged.emptied);
        self.boxed[table.index()] += merged.boxed as u64;
    }

    // -- Energy / cost helpers (mirror scan.rs) ----------------------------

    fn m_stem_cost(&self, pair: BasePair) -> Energy {
        Energy(self.params.ml.intern_37) + self.terminal_au(pair)
    }

    fn multi_close_cost(&self, closing: BasePair) -> Energy {
        Energy(self.params.ml.closing_37) + self.m_stem_cost(closing)
    }

    fn terminal_au(&self, pair: BasePair) -> Energy {
        if pair.applies_terminal_au() {
            Energy(self.params.misc.terminal_au_37)
        } else {
            Energy::ZERO
        }
    }
}

/// The invariant coordinates of one stack/bulge/internal closure candidate: the
/// inner pair's outer node `o` and key, the outer pair's 5′ boundary node `pn`
/// and opening edge `ep`, and the closing node/edge `(qn, eq)` with the
/// left-flank size `n1`. Bundled so the closure helpers stay within the
/// argument budget.
struct SingleClose {
    o: Node,
    key: PairKey,
    pn: Node,
    ep: Edge,
    qn: Node,
    eq: Edge,
    n1: u32,
}

/// The resolved outer pair of one single-branch closure: its closed-pair key,
/// the canonical outer / inner pairs, and the 3′-flank length `n2`. Computed once
/// in [`Dp::single_close`] and shared by the internal-loop and stack/bulge
/// closers so they agree on the cell they write.
struct OuterClose {
    key: PairKey,
    outer_pair: BasePair,
    inner_pair: BasePair,
    n2: u32,
}

/// The invariant coordinates of one multi-loop opening: the `M2` core's outer
/// node, the left-flank 5′ boundary node `pn` and opening edge `ep`, and the
/// left-flank CAI its points share. Shared by the short-flank close and the
/// deferred shared opening it seeds.
struct MultiSeed {
    core_outer: Node,
    pn: Node,
    ep: Edge,
    left_cai: f64,
}

/// A point of the merged front behind the long left flanks of the cores
/// sharing one outer node: its core and that core's inner node.
#[derive(Clone, Copy)]
struct FarBack {
    core: ProvIdx,
    inner: Node,
}

/// The committed cores sharing one outer node, by descending inner position
/// (ties in core order), for [`Dp::seed_far_openings`] and the openings it
/// leaves to be built later.
struct FarCores {
    cores: Vec<InnerCore>,
}

/// A committed core and the inner node its branch closes at.
type InnerCore = (Node, Arc<Com>);

impl FarCores {
    /// The merged front behind the long left flanks of the cores whose inner
    /// position exceeds `above`, each shifted by its flank's core part —
    /// built in the order [`Dp::seed_far_openings`] builds it.
    fn front(&self, dfa: &Dfa, above: u32) -> Front<FarBack> {
        let mut far = Front::empty();
        for (inner, core) in self.cores.iter().take_while(|(inner, _)| inner.pos > above) {
            if let Some(tail) = dfa.span_tail_cai(*inner) {
                let inner = *inner;
                far.merge_into(core.shift_map(Energy::ZERO, tail, |c| FarBack {
                    core: c.back,
                    inner,
                }));
            }
        }
        far
    }
}

/// The committed cores grouped by outer node, in first-seen order, each
/// group's cores by descending inner position (ties in core order).
fn far_groups(cores: &[(Node, Node, Arc<Com>)]) -> Vec<(Node, Arc<FarCores>)> {
    let mut groups: Vec<(Node, Vec<InnerCore>)> = Vec::new();
    for (outer, inner, core) in cores {
        let entry = (*inner, Arc::clone(core));
        match groups.iter_mut().find(|(o, _)| o == outer) {
            Some((_, group)) => group.push(entry),
            None => groups.push((*outer, vec![entry])),
        }
    }
    groups
        .into_iter()
        .map(|(outer, mut cores)| {
            cores.sort_by_key(|(inner, _)| std::cmp::Reverse(inner.pos));
            (outer, Arc::new(FarCores { cores }))
        })
        .collect()
}

/// A multi-loop opening as seeding writes it: built, or — under box cuts,
/// where a close keeps almost none of its points — the recipe the shared
/// table builds it from once a close might keep one.
#[derive(Clone)]
enum Opening {
    Built(Acc),
    Lazy(LazyOpening),
}

impl Opening {
    /// The points it holds built; a recipe holds none yet.
    fn len(&self) -> usize {
        match self {
            Self::Built(f) => f.len(),
            Self::Lazy(_) => 0,
        }
    }
}

/// An opening not yet built: its source front, the left flank's start, the
/// core's outer node, the opening's adaptation shift and a bound on each box
/// cut's lowest share of the opening it builds.
#[derive(Clone)]
struct LazyOpening {
    source: OpeningSource,
    from: Node,
    core_outer: Node,
    dcai: f64,
    lows: Lows,
}

/// Where a lazy opening's points come from: one core through a short left
/// flank, or the merged front of an outer node's cores through a long one.
#[derive(Clone)]
enum OpeningSource {
    Near { core: Arc<Com>, inner: Node },
    Far { cores: Arc<FarCores>, above: u32 },
}

impl LazyOpening {
    /// The opening front, exactly as seeding would have built it.
    fn build(&self, dfa: &Dfa) -> Acc {
        let (from, core_outer) = (self.from, self.core_outer);
        match &self.source {
            OpeningSource::Near { core, inner } => {
                let left = Span { from, to: *inner };
                core.shift_map(Energy::ZERO, self.dcai, |p| Prov::MultiOpen {
                    left,
                    core: p.back,
                    core_outer,
                })
            }
            OpeningSource::Far { cores, above } => {
                cores
                    .front(dfa, *above)
                    .shift_map(Energy::ZERO, self.dcai, |c| Prov::MultiOpen {
                        left: Span {
                            from,
                            to: c.back.inner,
                        },
                        core: c.back.core,
                        core_outer,
                    })
            }
        }
    }
}

/// One held opening closed against a frontier: its key, its front and, under
/// box cuts, a lower bound on each cut's lowest share of that front.
struct OpenClose<'f> {
    key: MultiOpenKey,
    slot: usize,
    opening: &'f Acc,
    lows: &'f [f64],
    /// The slot's recipes not yet built into `opening`.
    lazy: &'f [LazyOpening],
}

// ===========================================================================
// Traceback: walk one point's provenance DAG into (sequence, structure).
// ===========================================================================

/// Recovers `(sequence, structure)` for a front point by walking its
/// provenance DAG, writing nucleotides and brackets into position-indexed
/// buffers — the front analogue of [`crate::traceback`], per point.
struct Speller<'a> {
    dfa: &'a Dfa,
    arena: &'a Arena,
}

impl<'a> Speller<'a> {
    fn new(dfa: &'a Dfa, arena: &'a Arena) -> Self {
        Self { dfa, arena }
    }

    /// Walk `point`'s provenance into a [`RecoveredPoint`], carrying its
    /// objectives through unchanged.
    fn recover(&self, point: &Point<ProvIdx>) -> RecoveredPoint {
        let n = self.dfa.seq_len as usize;
        let mut w = Walk {
            speller: self,
            seq: vec![Nucleotide::A; n],
            dots: vec![b'.'; n],
        };
        w.exterior(point.back);
        RecoveredPoint {
            mfe: point.mfe,
            cai_log: point.cai,
            sequence: w.seq.iter().map(|&nt| nuc_char(nt)).collect(),
            structure: String::from_utf8(w.dots).expect("dot-bracket bytes are ASCII"),
        }
    }
}

/// One traceback walk's mutable buffers plus the immutable speller context.
struct Walk<'a> {
    speller: &'a Speller<'a>,
    seq: Vec<Nucleotide>,
    dots: Vec<u8>,
}

impl Walk<'_> {
    fn prov(&self, idx: ProvIdx) -> Prov {
        self.speller.arena.get(idx)
    }

    /// Walk an exterior chain (`C`), iterating the linear unpaired / empty spine
    /// and recursing into each absorbed pair.
    fn exterior(&mut self, mut idx: ProvIdx) {
        loop {
            match self.prov(idx) {
                Prov::ExtEmpty => return,
                Prov::ExtUnpaired { prev, pos, nuc } => {
                    self.seq[pos as usize] = nuc;
                    idx = prev;
                }
                Prov::ExtPair { prefix, pair } => {
                    self.closed_pair(pair);
                    idx = prefix;
                }
                _ => return,
            }
        }
    }

    /// Walk a closed pair (`P`): write its brackets and boundary nucleotides,
    /// fill its interior, and recurse into whatever it encloses.
    fn closed_pair(&mut self, idx: ProvIdx) {
        match self.prov(idx) {
            Prov::Hairpin { pair, fill } => {
                self.write_pair(pair);
                self.fill_loop(&fill);
            }
            Prov::Internal {
                pair,
                left,
                right,
                inner,
            } => {
                self.write_pair(pair);
                self.fill_loop(&left);
                self.fill_loop(&right);
                self.closed_pair(inner);
            }
            Prov::StackBulge {
                pair,
                left,
                right,
                inner,
            } => {
                self.write_pair(pair);
                self.fill_span(left);
                self.fill_span(right);
                self.closed_pair(inner);
            }
            Prov::MultiClose { pair, multi } => {
                self.write_pair(pair);
                self.multi(multi);
            }
            _ => {}
        }
    }

    /// Walk a `Multi` cell: fill its two best-CAI flanks and recurse into the
    /// `M2` core.
    fn multi(&mut self, idx: ProvIdx) {
        if let Prov::MultiWrap { left, right, core } = self.prov(idx) {
            self.fill_span(left);
            self.fill_span(right);
            self.m2(core);
        }
    }

    /// Walk an `M2` cell: recurse into both bifurcation branches.
    fn m2(&mut self, idx: ProvIdx) {
        if let Prov::M2Bifurcation { left, right } = self.prov(idx) {
            self.m(left);
            self.closed_pair(right);
        }
    }

    /// Walk an `M` chain: iterate the linear unpaired spine, terminating at the
    /// branch (`P`) or the collapsed `M2` it was built from.
    fn m(&mut self, mut idx: ProvIdx) {
        loop {
            match self.prov(idx) {
                Prov::MFromP { pair } => return self.closed_pair(pair),
                Prov::MFromM2 { m2 } => return self.m2(m2),
                Prov::MUnpaired { prev, pos, nuc } => {
                    self.seq[pos as usize] = nuc;
                    idx = prev;
                }
                _ => return,
            }
        }
    }

    /// Emit a closed pair's brackets and boundary nucleotides.
    fn write_pair(&mut self, pair: PairSpell) {
        self.seq[pair.a as usize] = pair.na;
        self.seq[pair.b as usize] = pair.nb;
        self.dots[pair.a as usize] = b'(';
        self.dots[pair.b as usize] = b')';
    }

    /// Respell a single-loop interior from its [`LoopFill`].
    fn fill_loop(&mut self, fill: &LoopFill) {
        match *fill {
            LoopFill::Corner {
                from,
                to,
                l_nuc,
                r_nuc,
                mid_from,
                mid_to,
            } => {
                self.seq[from as usize] = l_nuc;
                let mid = self.speller.dfa.broken_codon_seq(mid_from, mid_to);
                for (k, nt) in mid.into_iter().enumerate() {
                    self.seq[mid_from.pos as usize + k] = nt;
                }
                self.seq[(to - 1) as usize] = r_nuc;
            }
            LoopFill::Special { start, nuc, len } => {
                for (k, &nt) in nuc[..len as usize].iter().enumerate() {
                    self.seq[start as usize + k] = nt;
                }
            }
        }
    }

    /// Respell a best-CAI unpaired span.
    fn fill_span(&mut self, span: Span) {
        let interior = self.speller.dfa.broken_codon_seq(span.from, span.to);
        for (k, nt) in interior.into_iter().enumerate() {
            self.seq[span.from.pos as usize + k] = nt;
        }
    }
}

// ===========================================================================
// Free helpers.
// ===========================================================================

/// A terminal-mismatch corner of an unpaired strand `from → to`: its two
/// terminal nucleotides (the only interior the energy model reads), the total
/// CAI of the strand (best-CAI middle included), and the [`LoopFill`] that
/// respells it. Mirrors `interior::corner_scores`. Precondition: `from.pos <
/// to.pos`.
struct CornerFill {
    cai: f64,
    first: Nucleotide,
    last: Nucleotide,
    fill: LoopFill,
}

/// One inner pair's corner sets for its single-branch push: the current 5′
/// flank's corners, fixed across every 3′ side it closes with, and the 3′
/// corners by closing node, which every 5′ side shares.
#[derive(Default)]
struct CloseCorners {
    lefts: SmallVec<[CornerFill; 8]>,
    rights: RightFills,
}

/// The 3′ corners from one inner pair's outer node `o` to each closing node,
/// enumerated on first use; a slot holds `qn`'s when its node is `qn`.
#[derive(Default)]
struct RightFills {
    o: Option<Node>,
    slots: Vec<(Option<Node>, SmallVec<[CornerFill; 8]>)>,
}

impl RightFills {
    fn get(&mut self, dfa: &Dfa, o: Node, qn: Node) -> &[CornerFill] {
        if self.o != Some(o) {
            self.slots.iter_mut().for_each(|slot| slot.0 = None);
            self.o = Some(o);
        }
        let i = (qn.pos - o.pos) as usize * dfa.stride() + usize::from(qn.lane);
        if self.slots.len() <= i {
            self.slots.resize_with(i + 1, || (None, SmallVec::new()));
        }
        let slot = &mut self.slots[i];
        if slot.0 != Some(qn) {
            slot.1 = corner_fills(dfa, o, qn);
            slot.0 = Some(qn);
        }
        &slot.1
    }
}

/// Every terminal-mismatch corner of the strand `from → to`. A length-1 strand
/// has one nucleotide, so `first == last` and the [`LoopFill`] middle is empty;
/// longer strands enumerate `(5′ mismatch, 3′ mismatch)` pairs with the
/// energy-irrelevant middle left best-CAI.
/// The least internal-loop energy of each loop shape — outer and inner
/// pair, and two flanks of one or more nucleotides totalling at most
/// `SINGLE_MAX_LEN` — over its four terminal mismatches.
struct InternalLeast {
    energies: Vec<Energy>,
}

/// What [`InternalLeast::at`] gives a shape it holds no bound for: low enough
/// that no cut can fail it.
const NO_BOUND: Energy = Energy(i32::MIN / 4);

impl InternalLeast {
    const SIDE: usize = SINGLE_MAX_LEN as usize + 1;

    fn new(params: &Params) -> Self {
        let mut energies = vec![NO_BOUND; 7 * 7 * Self::SIDE * Self::SIDE];
        let pairs: Vec<BasePair> = BasePair::ALL
            .into_iter()
            .filter(|&p| p != BasePair::NN)
            .collect();
        let shapes = (1..Self::SIDE).flat_map(|n1| (1..Self::SIDE - n1).map(move |n2| (n1, n2)));
        for &outer in &pairs {
            for &inner in &pairs {
                for (n1, n2) in shapes.clone() {
                    energies[Self::index(outer, inner, n1, n2)] =
                        least_internal(params, outer, inner, n1, n2);
                }
            }
        }
        Self { energies }
    }

    fn index(outer: BasePair, inner: BasePair, n1: usize, n2: usize) -> usize {
        ((outer as usize * 7 + inner as usize) * Self::SIDE + n1) * Self::SIDE + n2
    }

    /// The least energy of the shape, or [`NO_BOUND`] outside the table.
    fn at(&self, outer: BasePair, inner: BasePair, n1: u32, n2: u32) -> Energy {
        let (n1, n2) = (n1 as usize, n2 as usize);
        if n1 >= Self::SIDE || n2 >= Self::SIDE {
            return NO_BOUND;
        }
        self.energies[Self::index(outer, inner, n1, n2)]
    }
}

/// The least energy of one internal-loop shape over all 256 choices of its
/// four terminal mismatch nucleotides (inconsistent ones included, which only
/// lowers the bound).
fn least_internal(
    params: &Params,
    outer: BasePair,
    inner: BasePair,
    n1: usize,
    n2: usize,
) -> Energy {
    const NUCS: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
    (0..256usize)
        .map(|m| {
            let shape = InternalLoopScalar {
                outer,
                inner,
                boundary_left5p: NUCS[m & 3],
                boundary_left3p: NUCS[(m >> 2) & 3],
                boundary_right5p: NUCS[(m >> 4) & 3],
                boundary_right3p: NUCS[(m >> 6) & 3],
                n1,
                n2,
            };
            internal_energy_scalar(params, &shape)
        })
        .min()
        .expect("256 mismatches")
}

/// For each closing pair (by its index), the terminal mismatch giving the
/// least hairpin energy. A loop of any size that reads a mismatch adds the
/// same table entry to its size's penalty, so one size settles every size.
fn least_hairpin_mismatches(params: &Params) -> [(Nucleotide, Nucleotide); 7] {
    const NUCS: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
    let mut least = [(Nucleotide::A, Nucleotide::A); 7];
    for closing in BasePair::ALL {
        let energy = |&(first, last): &(Nucleotide, Nucleotide)| {
            hairpin_energy_scalar(params, closing, 5, first, last)
        };
        least[closing as usize] = NUCS
            .iter()
            .flat_map(|&first| NUCS.iter().map(move |&last| (first, last)))
            .min_by_key(energy)
            .expect("sixteen mismatches");
    }
    least
}

fn corner_fills(dfa: &Dfa, from: Node, to: Node) -> SmallVec<[CornerFill; 8]> {
    if to.pos - from.pos == 1 {
        return dfa
            .right(from)
            .iter()
            .filter(|e| e.to == to)
            .map(|e| CornerFill {
                cai: e.cai,
                first: e.nuc,
                last: e.nuc,
                fill: LoopFill::Corner {
                    from: from.pos,
                    to: to.pos,
                    l_nuc: e.nuc,
                    r_nuc: e.nuc,
                    mid_from: e.to,
                    mid_to: e.to,
                },
            })
            .collect();
    }
    let mut out = SmallVec::new();
    for first in dfa.right(from) {
        for last in dfa.left(to) {
            let Some(mid) = dfa.broken_codon_cai(first.to, last.to) else {
                continue;
            };
            out.push(CornerFill {
                cai: first.cai + mid + last.cai,
                first: first.nuc,
                last: last.nuc,
                fill: LoopFill::Corner {
                    from: from.pos,
                    to: to.pos,
                    l_nuc: first.nuc,
                    r_nuc: last.nuc,
                    mid_from: first.to,
                    mid_to: last.to,
                },
            });
        }
    }
    out
}

/// A [`LoopFill::Special`] holding the fully-enumerated loop sequence, written
/// from `start`. The special-hairpin sizes are `<= 6`, so the sequence fits the
/// fixed inline buffer.
fn special_fill(start: u32, seq: &[Nucleotide]) -> LoopFill {
    let mut nuc = [Nucleotide::A; 6];
    nuc[..seq.len()].copy_from_slice(seq);
    LoopFill::Special {
        start,
        nuc,
        len: seq.len() as u8,
    }
}

/// Stack (`n1 == n2 == 0`) or bulge (one flank empty) energy — the interior
/// nucleotides do not enter, only the flank sizes and the two pairs. Mirrors
/// `scan::stack_or_bulge_energy`.
fn stack_or_bulge_energy(
    params: &Params,
    outer: BasePair,
    inner: BasePair,
    n1: u32,
    n2: u32,
) -> Energy {
    if n1 == 0 && n2 == 0 {
        return stack_energy(params, &StackPair { outer, inner });
    }
    bulge_energy(
        params,
        &BulgeLoop {
            outer,
            inner,
            size: (n1 + n2) as usize,
        },
    )
}

/// Rewrite a shared opening point's provenance into a completed multi-loop
/// wrap, filling the right flank `core_outer → qn` the close resolves. The
/// `multi_open` fronts hold only [`Prov::MultiOpen`]; any other variant is
/// carried through unchanged (unreachable in practice).
fn close_opening(p: &Point<Prov>, qn: Node) -> Prov {
    match p.back {
        Prov::MultiOpen {
            left,
            core,
            core_outer,
        } => Prov::MultiWrap {
            left,
            right: Span {
                from: core_outer,
                to: qn,
            },
            core,
        },
        other => other,
    }
}

/// How many nucleotides a closed pair or multi-loop cell accounts for: its 5′
/// boundary node to the outer node it is consumed at.
fn pair_span(key: PairKey, outer: Node) -> u32 {
    outer.pos.saturating_sub(key.open.pos)
}

/// How many nucleotides a multi-branch cell accounts for: its inner boundary
/// node to the outer node it is consumed at.
fn branch_span(inner: Node, outer: Node) -> u32 {
    outer.pos.saturating_sub(inner.pos)
}

/// Canonical Watson–Crick / wobble pair indicator; `None` for non-pairing
/// combinations. Mirrors `scan::canonical`.
fn canonical(a: Nucleotide, b: Nucleotide) -> Option<BasePair> {
    match pair_from_nucleotides(a, b) {
        BasePair::NN => None,
        canonical => Some(canonical),
    }
}

fn nuc_char(nt: Nucleotide) -> char {
    match nt {
        Nucleotide::A => 'A',
        Nucleotide::C => 'C',
        Nucleotide::G => 'G',
        Nucleotide::U => 'U',
    }
}

/// `⊕`-accumulate a front into a keyed column, dominance-merging on collision,
/// then applying `prune` so the stored cell stays capped and `slack` so the cell
/// carries no derivation that provably cannot complete within the ceiling.
/// Returns how many the latter discarded, and whether it emptied the cell — in
/// which case the cell is removed, so nothing downstream composes it.
///
/// The feasibility drop runs *beside* the prune rather than inside it, and
/// unconditionally. [`prune_in_place`] short-circuits four ways — on the policy
/// being off, on either cap's size constraint, and on [`PRUNE_SLACK`] — and the
/// configuration a ceiling is most wanted in is the exact fold, where the first
/// of those fires every time. A drop folded into the prune would therefore
/// filter nothing exactly where it matters most.
///
/// A box's cuts, where given, filter the incoming front before it is merged:
/// each test reads one point and the cell's key, so filtering what arrives is
/// the same as filtering the cell, and an arrival the cuts empty creates no
/// cell at all.
#[allow(clippy::too_many_arguments)]
fn merge_keyed<K: std::hash::Hash + Eq + Copy>(
    map: &mut FxHashMap<K, Acc>,
    key: K,
    mut f: Acc,
    prune: Prune,
    slack: Slack,
    exact: Option<Energy>,
    boxed: Option<(&BoxCuts, CellRef)>,
) -> Merged {
    let boxed = match boxed {
        Some((cuts, cell)) => cuts.retain_in_box(cell, &mut f),
        None => 0,
    };
    if boxed > 0 && f.is_empty() {
        return Merged {
            dropped: 0,
            emptied: false,
            boxed,
        };
    }
    let cell = map.entry(key).or_insert_with(Acc::empty);
    cell.merge_into(f);
    prune_in_place(cell, prune);
    let dropped = retain_within(cell, slack, exact);
    let emptied = exact.is_some() && cell.is_empty();
    if emptied {
        map.remove(&key);
    }
    Merged {
        dropped,
        emptied,
        boxed,
    }
}

/// One bifurcation, composed ahead of its merge into `M2[o][inner]`.
struct Bifurcation {
    inner: Node,
    /// The filtered compose, cut by the box.
    arrival: Acc,
    /// `|M|·|P|`, and the products the filter let the compose generate.
    products: usize,
    generated: usize,
    /// Points the compose returned, and of those the box's cuts removed.
    eliminated: usize,
    boxed: usize,
    /// Whether the unfiltered compose would have reached the merge with a
    /// point left after the cuts, where the filtered one did not; see
    /// [`Dp::merge_bifurcation`].
    replay: bool,
}

/// What a compose may skip generating, because its merge would discard it:
/// products above `cutoff`, and products a cut excludes by more than the
/// elimination's tolerance could matter to (see
/// [`BoxCuts::admits_before_elimination`]).
#[derive(Default)]
struct ComposeFilter<'c> {
    cutoff: Option<Energy>,
    cuts: Option<(&'c BoxCuts, SmallVec<[f64; 8]>)>,
}

impl ComposeFilter<'_> {
    /// Whether a product within the cutoff is generated.
    fn admits(&self, mfe: Energy, cai: f64) -> bool {
        self.cuts
            .as_ref()
            .is_none_or(|(cuts, outs)| cuts.admits_before_elimination(outs, mfe, cai))
    }

    /// A linear test [`Self::admits`] implies, for the compose to order its
    /// products by.
    fn screen(&self) -> Option<Screen> {
        self.cuts
            .as_ref()
            .and_then(|(cuts, outs)| cuts.screen(outs))
    }
}

/// Whether the unfiltered compose of `left ⊗ right` shifted by `dmfe` keeps a
/// point every cut admits at a cell whose costs to go are `outs`.
fn unfiltered_arrives_at<B: Copy, C: Copy>(
    cuts: &BoxCuts,
    outs: &[f64],
    left: &Front<B>,
    right: &Front<C>,
    dmfe: Energy,
) -> bool {
    cuts.product_reaches(outs, left, right, dmfe)
        .unwrap_or_else(|| {
            let mut all = left.compose(right, dmfe, 0.0, |a, b| (a, b));
            cuts.retain_at(outs, &mut all);
            !all.is_empty()
        })
}

/// What one merge's feasibility drop did to its cell.
/// How finely a fan-out's work is cut: about this many runs per worker.
const UNITS_PER_WORKER: usize = 16;

/// Cut `items` into runs of consecutive items to deal to `threads` workers,
/// heaviest first. A run closes at [`FAN_OUT_CHUNK`] items, or before an item
/// that would take it past an even share of the work cut
/// [`UNITS_PER_WORKER`] ways per worker, so a heavy item travels alone and no
/// run of light items is dealt last. Runs are returned as `(start, end)` in
/// dealing order; the caller replays their partials by `start`, which is the
/// items' order, whatever order they were dealt in.
fn deal_units<T>(items: &[T], weight: impl Fn(&T) -> usize, threads: usize) -> Vec<(usize, usize)> {
    let weights: Vec<usize> = items.iter().map(|it| weight(it).max(1)).collect();
    let total: usize = weights.iter().sum();
    let target = (total / (threads.max(1) * UNITS_PER_WORKER)).max(1);
    let mut units: Vec<(usize, usize, usize)> = Vec::new();
    let (mut start, mut held) = (0, 0);
    for (i, &w) in weights.iter().enumerate() {
        if i > start && (i - start == FAN_OUT_CHUNK || held + w > target) {
            units.push((start, i, held));
            start = i;
            held = 0;
        }
        held += w;
    }
    if start < items.len() {
        units.push((start, items.len(), held));
    }
    units.sort_by_key(|&(start, _, w)| (std::cmp::Reverse(w), start));
    units
        .into_iter()
        .map(|(start, end, _)| (start, end))
        .collect()
}

/// A merge into a shared opening's front: the window alone, since a shared
/// opening pools cores with different costs to go and has no exact cutoff.
/// Its cell is never emptied and never removed.
#[derive(Clone, Copy)]
struct OpenMerge {
    prune: Prune,
    slack: Slack,
}

impl OpenMerge {
    /// Merge `f` into `front`; the points the window dropped.
    fn merge(&self, front: &mut Acc, f: Acc) -> usize {
        front.merge_into(f);
        prune_in_place(front, self.prune);
        retain_within(front, self.slack, None)
    }
}

/// One key of a sharded opening merge: its slot, its front taken out of the
/// table, its arrivals in order, and what the window dropped.
struct OpenJob {
    slot: usize,
    front: Acc,
    lows: Lows,
    items: Vec<Acc>,
    dropped: usize,
}

/// A merge into a `P` cell, with everything it reads but the cell's column.
#[derive(Clone, Copy)]
struct PMerge<'a> {
    prune: Prune,
    slack: Slack,
    credit: Option<&'a ExactCredit>,
    cuts: Option<&'a BoxCuts>,
    region: &'a PairConstraint,
}

impl PMerge<'_> {
    fn merge(
        &self,
        column: &mut FxHashMap<PairKey, Acc>,
        outer: Node,
        key: PairKey,
        f: Acc,
    ) -> Merged {
        match self.region.verdict(key.open.pos, key.close.pos) {
            Verdict::Allow => {}
            Verdict::Forbid => {
                return Merged {
                    dropped: 0,
                    emptied: false,
                    boxed: 0,
                }
            }
            Verdict::Penalize(_) => {
                unreachable!("the front fold is given hard region constraints only")
            }
        }
        let cell = CellRef::P(outer, key);
        let exact = self
            .credit
            .and_then(|credit| credit.cutoff(cell, self.slack));
        let boxed = self.cuts.map(|cuts| (cuts, cell));
        merge_keyed(column, key, f, self.prune, self.slack, exact, boxed)
    }
}

/// One column of a sharded push apply: the `P` column, the candidates bound
/// for it in serial order, and what their merges dropped, emptied and boxed.
struct PColumn<'c> {
    column: &'c mut FxHashMap<PairKey, Acc>,
    items: Vec<PCandidate>,
    merged: [u64; 3],
}

struct Merged {
    dropped: usize,
    emptied: bool,
    /// Points the box's cuts removed from the arrival before the merge.
    boxed: usize,
}

/// The smallest cell size worth pruning. Below this a front is already at or
/// near any useful cap and the grid rarely drops a point, so the per-merge prune
/// (an allocation, plus a `ln` per point for the ε-grid) is skipped — the cell
/// is re-checked on its next merge. This bounds a stored cell to
/// `PRUNE_SLACK + (merge size)` between prunes, not the un-pruned `k̄`, so memory
/// stays capped while the hot merge path stays cheap.
const PRUNE_SLACK: usize = 48;

/// Prune a stored cell in place, a no-op under [`Prune::None`] (so the exact
/// fold never reallocates a front it would leave unchanged) and skipped for
/// fronts already small enough that pruning cannot help.
fn prune_in_place(cell: &mut Acc, prune: Prune) {
    match prune {
        Prune::None => {}
        Prune::Cap { max } if cell.len() <= max => {}
        Prune::EpsGridCapped { max, .. } if cell.len() <= max => {}
        Prune::EpsGridMfe { .. } if cell.len() <= PRUNE_SLACK => {}
        _ => *cell = cell.prune(prune),
    }
}

/// Drain a keyed column into an owned vector, so the caller can iterate it while
/// writing into the same table field.
fn take_col<K: std::hash::Hash + Eq>(map: &mut FxHashMap<K, Acc>) -> Vec<(K, Acc)> {
    std::mem::take(map).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small xorshift, so the generated cases are fixed.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, m: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % m
        }
    }

    #[test]
    fn dealt_runs_cover_the_items_and_isolate_heavy_ones() {
        let mut rng = Rng(0x51ed_270b_27e5_9a3f);
        for case in 0..200 {
            let n = rng.below(300) as usize;
            let weights: Vec<usize> = (0..n)
                .map(|_| {
                    if rng.below(20) == 0 {
                        10_000
                    } else {
                        rng.below(50) as usize
                    }
                })
                .collect();
            let units = deal_units(&weights, |&w| w, 8);
            let mut runs = units.clone();
            runs.sort();
            let mut next = 0;
            for &(start, end) in &runs {
                assert_eq!(start, next, "case {case}: runs are contiguous");
                assert!(end > start && end - start <= FAN_OUT_CHUNK, "case {case}");
                next = end;
            }
            assert_eq!(next, n, "case {case}: runs cover every item");
            let total: usize = weights.iter().map(|&w| w.max(1)).sum();
            let target = (total / (8 * UNITS_PER_WORKER)).max(1);
            for &(start, end) in &runs {
                if (start..end).any(|i| weights[i].max(1) > target) {
                    assert_eq!(end - start, 1, "case {case}: a heavy item travels alone");
                }
            }
            let load = |&(start, end): &(usize, usize)| {
                weights[start..end].iter().map(|&w| w.max(1)).sum::<usize>()
            };
            assert!(
                units.windows(2).all(|u| load(&u[0]) >= load(&u[1])),
                "case {case}: heaviest first"
            );
        }
    }
}
