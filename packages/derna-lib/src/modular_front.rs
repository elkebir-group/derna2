//! The modular front of a whole molecule, under a time budget.
//!
//! A design is modular when its free MFE equals its constrained
//! energy. The modular front is the Pareto set of (constrained
//! energy, adaptation) over those designs. It is computed here from the
//! constrained front, which bounds it from below everywhere:
//!
//! 1. **Seeds.** Every point of the constrained front whose own free fold
//!    respects the constraint is on the modular front: nothing
//!    has lower energy at its adaptation or above, modular or not.
//! 2. **Stretches.** Between two adaptations held by consecutive seeds, and
//!    below the first and above the last, the front is a window walk
//!    ([`window_front`](crate::outside::lazy::window::window_front)): designs in ascending `E − μ·A`, each checked, with
//!    the seed above the stretch capping it. Two seeds adjacent on the
//!    constrained front leave nothing between them — a design there would be
//!    dominated by the upper seed — so that stretch is not walked.
//!    The band above the best-adapted point found has no cap — the
//!    best-adapted modular design is on the front whatever its
//!    energy — so its designs are enumerated, most adapted first ([`top`]).
//! 3. **Rounds.** A walk stopped by its share of the time leaves regions open.
//!    Each round walks what the last left open, for twice as long, with every
//!    point found so far known: the nearest one above a region caps it, and
//!    the walk runs at the slope that minimizes the band of designs it must
//!    pass (`band_slope`). Alternate rounds add the energy bound, which closes
//!    regions spanning much energy that the band alone does not. Rounds stop
//!    when nothing is open or the budget is spent.
//! 4. **Merge.** The union of the seeds and every walk's points, less what
//!    another of them dominates (a stretch's point can be dominated by a
//!    design of a stretch above it).
//!
//! **Under a budget the answer is partial and says exactly how.** A stretch
//! the budget stops reports the region it left open: adaptations at which a
//! missing point may sit, each at energy at least `last_g + μ·A`, and never
//! below the constrained front. A point of the merged front is
//! [`ModularPoint::certified`] when no open region can hold a design dominating it;
//! otherwise it is provisional — modular, undominated by
//! everything found, but not yet proven on the front.
//!
//! **From a partial constrained front.** A walk stopped by its budget returns
//! exact points and the boxes between them it left open; every front point
//! between two of its points lies in their box. The box's lowest energy,
//! at its highest adaptation, then floors every design the box can hold, so
//! the search reads each box's corner beside the points wherever it reads the
//! constrained front as a floor, and two points with a box between them are
//! not adjacent. The answer stays exact; a wide box only lengthens the walks.
//!
//! **Over motif-free designs** ([`modular_motif_front`]) the
//! same search starts from the motif-free constrained front and drops every
//! design that spells a forbidden motif before it is checked. The motif-free
//! constrained front bounds the motif-free modular designs from
//! below as the plain one bounds all of them, so the seeds and the stretches
//! read the same way. The walks run over every codon the pins leave each
//! residue: each bound they prune by holds over all those designs and so over
//! the motif-free ones, and only motif-free finds cover or cap anything.
//!
//! **Within ε.** Given `epsilon > 0`, the search returns the front of a
//! different set: the designs whose gap is at most `epsilon`. Every step
//! above is a domination argument that holds for any set of designs, so long
//! as each point that caps, covers or certifies is itself in the set — a
//! design dominates nothing in a set it is not a member of, however small its
//! gap. So `epsilon` reaches every test of membership together: the check of
//! each constrained-front point, the window walk's check, and the top band's.
//! The answer is not the modular front with near misses added: a
//! modular design that a design within `epsilon` dominates is
//! not on it. At `epsilon = 0` it is the modular front.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

mod top;

use crate::ac::{Ac, MotifWindow, ScopedAc};
use crate::coords::Layout;
use crate::dev_env::Setting;
use crate::free_fold::{
    epsilon_dekacal, sequence_gap_given_structure_reusing, CdsWindow, FreeFoldError, FreeFoldGap,
    FreeFoldTables,
};
use crate::outside::lazy::window::{adaptation, window_front_sharing, WindowReport};
use crate::outside::lazy::{Build, CutSchedule, ExactFold, LazyError, Verify, Walk};
use crate::pins::Pins;
use crate::positional::Infeasible;
use crate::region::RegionConstraints;
use crate::{CodonUsage, FrontPoint, NucSet, Nucleotide, Params, ScanConfig};
use derna_codon_dp::codon_table::AMINO_ACIDS;

/// The stack a window walk runs on: its derivations nest as deep as the
/// molecule is long, and the walk recurses through them.
const WALK_STACK: usize = 4 << 30;

/// How close two adaptations are to count as equal.
const CAI_TOL: f64 = 1e-9;

/// The first round's share of the time, in seconds; each later round's is
/// twice the last's.
const FIRST_ROUND_S: f64 = 30.0;

/// The fewest threads a region walked beside others is given: a round walks
/// at most `threads / MIN_LANE_THREADS` regions at once, so each still checks
/// designs on several workers.
const MIN_LANE_THREADS: usize = 2;

/// Regions a round walks side by side: `threads / MIN_LANE_THREADS` unless
/// [`Setting::ModularLanes`] names another count, and never more than the
/// regions open.
///
/// One lane holds one window walk's tables, so the count sets what the search
/// resides in as much as the molecule's length does. A molecule long enough
/// that one walk's tables already fill most of a memory ceiling cannot hold
/// two, and the ceiling is reached before either walk has run long enough to
/// find a design. Setting this to 1 trades each region's share of a round --
/// the regions then take the round in turn rather than together -- for a
/// resident set that one walk's tables bound.
fn lane_count(threads: usize) -> usize {
    crate::dev_env::var(Setting::ModularLanes)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(threads / MIN_LANE_THREADS)
        .max(1)
}

/// The seconds the first round gives a region: `FIRST_ROUND_S` unless
/// [`Setting::ModularFirstRoundS`] names another span.
///
/// Rounds double, so the default schedule has spent `2^k - 1` times the first
/// round's span by the end of round `k` and has never given one region more
/// than half of that. A region whose walk needs longer than the whole span to
/// reach its first design finds nothing in any of those rounds, and a run cut
/// short before the rounds grow has paid for the doubling without reaching the
/// span its regions need. Where that span is known from a walk of comparable
/// length, naming it here puts it in the first round instead.
fn first_round_s() -> f64 {
    crate::dev_env::var(Setting::ModularFirstRoundS)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(FIRST_ROUND_S)
}

/// The energy bound's slopes on the rounds that use it, as multiples of the
/// walk's own slope: a spread below and above it, and the pure-energy slope.
const ENERGY_SLOPES: [f64; 5] = [0.0, 0.25, 0.5, 2.0, 4.0];

/// How far an open region's energy floor must sit below a point's energy for
/// the region to hold a dominating design, in dekacalories: energies are
/// whole dekacalories, so a floor within rounding of the point cannot.
const ENERGY_TOL: f64 = 1e-6;

/// Why the modular front could not be computed.
#[derive(Debug, thiserror::Error)]
pub enum ModularFrontError {
    #[error(transparent)]
    Walk(#[from] LazyError),
    #[error(transparent)]
    Verdict(#[from] FreeFoldError),
    #[error(transparent)]
    Beam(#[from] crate::BeamError),
    /// A constrained-front point the molecule's lattice does not spell: the
    /// front was computed over a different molecule.
    #[error("a constrained-front point is not a design of this molecule: {0}")]
    NotOnLattice(String),
    /// Progress settled at another `epsilon`: its verdicts and finds are
    /// members of another set, and would cap this search unsoundly.
    #[error("the progress was settled within {prior} kcal/mol, and this search is within {run}")]
    OtherEpsilon { prior: f64, run: f64 },
}

/// One point of the modular front.
#[derive(Debug, Clone, PartialEq)]
pub struct ModularPoint {
    /// The whole molecule.
    pub sequence: String,
    /// Constrained energy in kcal/mol: the free MFE at `epsilon` 0, and at
    /// most `epsilon` above it otherwise.
    pub mfe_kcal_mol: f64,
    /// Coding log-adaptation, on the constrained front's scale.
    pub cai_log: f64,
    /// Whether the point is proven on the front: no region a budget left open
    /// can hold a design that dominates it.
    pub certified: bool,
    /// The check this point already passed, when it is a seed: the
    /// constrained-front point's own free fold against its own constrained
    /// fold, run to decide it was modular. A caller that reports
    /// each point's check reads it here instead of folding the point again.
    /// `None` for a point a walk found.
    pub check: Option<FreeFoldGap>,
}

/// Where a missing point may sit: adaptations in `[cai_log_lo, cai_log_hi]`
/// (`None` for an open end) at energy at least the higher of the two floors,
/// `mfe_floor_kcal_mol + slope·(cai_log − cai_log_lo)` and
/// `mfe_floor_constrained_kcal_mol`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRegion {
    pub cai_log_lo: Option<f64>,
    pub cai_log_hi: Option<f64>,
    /// The least energy the walks that reached the region leave a design in
    /// it, at `cai_log_lo`; `None` when no walk reached it.
    pub mfe_floor_kcal_mol: Option<f64>,
    /// How fast that floor rises with adaptation, kcal/mol per unit.
    pub slope_kcal_mol: f64,
    /// The constrained front's own floor over the region: no design has a
    /// constrained energy below it, and a modular design's
    /// energy *is* its constrained energy, so no missing point sits below it
    /// either. Flat, because the constrained front rises with adaptation and
    /// this is its value at `cai_log_lo` — or, where that end is open, the
    /// front's own minimum. `None` only for a region lying above every
    /// constrained-front point, which no design's adaptation reaches.
    ///
    /// It is the floor that binds where no walk has reached the region, and
    /// often where one has: a walk's floor is `last_g + μ·A`, which at the
    /// region's lowest adaptation can sit far below any design the lattice
    /// spells.
    pub mfe_floor_constrained_kcal_mol: Option<f64>,
}

impl OpenRegion {
    /// Whether a design at `(mfe, cai_log)` lies in the region, as far as its
    /// floors say.
    pub fn admits(&self, mfe_kcal_mol: f64, cai_log: f64) -> bool {
        let lo = self.cai_log_lo.unwrap_or(f64::NEG_INFINITY);
        let hi = self.cai_log_hi.unwrap_or(f64::INFINITY);
        let walked = match (self.mfe_floor_kcal_mol, self.cai_log_lo) {
            (Some(f), Some(lo)) => f + self.slope_kcal_mol * (cai_log - lo),
            _ => f64::NEG_INFINITY,
        };
        let floor = walked.max(
            self.mfe_floor_constrained_kcal_mol
                .unwrap_or(f64::NEG_INFINITY),
        );
        cai_log >= lo - CAI_TOL && cai_log <= hi + CAI_TOL && mfe_kcal_mol >= floor - 1e-6
    }

    /// Whether the region could hold a design dominating `(mfe, cai_log)`:
    /// adaptation at least as high, at energy no higher.
    ///
    /// This is the question certification asks, and it is not [`Self::admits`],
    /// which asks whether one particular point could lie in the region. A
    /// dominating design may sit anywhere in the region at or above `cai_log`,
    /// so the floor is read at the lowest such adaptation — floors rise with
    /// adaptation, so that is where the region is cheapest and the test
    /// therefore strongest.
    ///
    /// Sound, but not as tight as the search's own test, by one term: the
    /// constrained floor is reported flat at the region's low end, where the
    /// search evaluates the constrained front's staircase at the adaptation in
    /// question. The flat value is the lower of the two, so this predicate
    /// fires at least as often as the search's, and a point it leaves
    /// uncontested really is uncontested.
    pub fn could_dominate(&self, mfe_kcal_mol: f64, cai_log: f64) -> bool {
        // A region lying wholly below the point holds nothing better adapted
        // than it, at any energy.
        if self.cai_log_hi.is_some_and(|hi| hi < cai_log - CAI_TOL) {
            return false;
        }
        let at = self.cai_log_lo.unwrap_or(f64::NEG_INFINITY).max(cai_log);
        let walked = match (self.mfe_floor_kcal_mol, self.cai_log_lo) {
            (Some(f), Some(lo)) => f + self.slope_kcal_mol * (at - lo),
            _ => f64::NEG_INFINITY,
        };
        let floor = walked.max(
            self.mfe_floor_constrained_kcal_mol
                .unwrap_or(f64::NEG_INFINITY),
        );
        floor < mfe_kcal_mol - ENERGY_TOL
    }
}

/// One window walked: a stretch between two seeds (or a seed and an end) in
/// the first round, an open region in a later one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stretch {
    /// The round it was walked in, from 0.
    pub round: usize,
    /// The stretch's adaptation bounds on the constrained front's scale;
    /// `None` for an open end.
    pub cai_log_lo: Option<f64>,
    pub cai_log_hi: Option<f64>,
    /// Whether the stretch was walked at all: two seeds adjacent on the
    /// constrained front leave it empty.
    pub walked: bool,
    pub complete: bool,
    /// Modular points the stretch's walk returned.
    pub points: usize,
    /// Modular designs the stretch's walk found, on its front or
    /// not. A window walk prunes a design another it already found dominates,
    /// so it finds more than it returns and `modular_found - points` is how
    /// many modular designs it left off its front; only the front's molecules
    /// leave the walk, so that difference is the only trace of them. A band
    /// returns every design it found, so the two are equal there.
    pub modular_found: usize,
    /// Designs the top band checked here; zero for a window walk, whose
    /// designs are counted by its own report.
    pub designs: u64,
    /// The walk's slope, dekacalories per unit of adaptation.
    pub slope: f64,
    /// Wall-clock seconds of the visit. The visit runs on `threads` workers,
    /// so the thread-time it held is `wall_s × threads`; summing `wall_s`
    /// over stretches that ran side by side counts their overlap twice.
    pub wall_s: f64,
    /// The workers the visit ran on: its lane's share of the search's threads.
    pub threads: usize,
    /// Wall-clock seconds of the walk's setup, paid whole before its first
    /// design however short its share: the fold cut and the energy bound's
    /// outside passes. Zero for the top band and an unwalked stretch.
    pub cut_s: f64,
    pub energy_s: f64,
    /// Wall-clock seconds the walk spent checking designs (witnesses and
    /// folds), the folds it ran, and the designs it reached and offered to
    /// the check.
    pub verify_s: f64,
    pub folds: u64,
    pub reached: u64,
    pub offered: u64,
    /// The walk's own split: wall-clock seconds building its grammar (the
    /// inside sweep among them) and walking it, the derivations it took off
    /// the root, and why it ended.
    pub build_s: f64,
    pub sweep_s: f64,
    pub walk_s: f64,
    pub derivations: u64,
    pub stopped: &'static str,
}

/// The modular front, whole or as far as the budget reached.
#[derive(Debug, Clone, PartialEq)]
pub struct ModularFront {
    /// Energy ascending.
    pub points: Vec<ModularPoint>,
    /// Every region a stopped stretch left open.
    pub open: Vec<OpenRegion>,
    pub stretches: Vec<Stretch>,
    /// How many constrained-front points were modular.
    pub seeds: usize,
    /// How many constrained-front points the budget left unchecked: not seeds,
    /// and the stretches around them left open.
    pub unchecked: usize,
    /// Whether every stretch completed, so the front is the whole front.
    pub complete: bool,
    /// Seconds spent checking the constrained front's points, before any
    /// stretch was walked.
    ///
    /// Certification is the dominant cost at length, so this is the number that
    /// says what a search over one lattice saved against one search per stop
    /// codon: a lattice carrying the stop as a slot checks one front once.
    pub certify_s: f64,
    /// The largest gap, in kcal/mol, a point may have: `0` for the
    /// modular front.
    pub epsilon: f64,
    /// Regions the rounds walked side by side, and the seconds the first
    /// round gave a region.
    ///
    /// Both bound what a budget could have reached: one lane takes the
    /// regions in turn, and a round's span is the first round's doubled once
    /// per round, so a front stopped early is read against the span its
    /// regions were actually given rather than the budget they were allowed.
    pub schedule: (usize, f64),
    /// Seconds spent building the check's fold cut, once, before the first
    /// window walk; outside every stretch's `wall_s`.
    pub cut_build_s: f64,
    /// Set when [`SearchPolicy::give_up_empty`] stopped the search: the round
    /// it would have walked next and the derivations its walks had taken. The
    /// front is then empty and incomplete; it is not shown empty.
    pub gave_up: Option<(usize, u64)>,
}

/// When the search stops or holds back a region before its budget runs out.
/// The default does neither: every region is walked until the budget ends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchPolicy {
    /// `(rounds, derivations)`: a search that has found no point once `rounds`
    /// rounds are over and its walks have taken at least `derivations`
    /// derivations stops there. It costs a point found later; on the cells
    /// measured, a walk that found anything found its first point within
    /// four rounds.
    pub give_up_empty: Option<(usize, u64)>,
    /// A region re-entered with the bounds its last visit had, after a visit
    /// that found nothing, is given no more time than that visit while some
    /// other region is open. The time goes to the regions after it, so this
    /// changes what a budget finds.
    pub hold_barren: bool,
}

/// What an earlier search of the same molecule and constrained front settled,
/// for [`modular_front_resuming`] to skip.
#[derive(Debug, Clone, Default)]
pub struct ModularProgress {
    /// Constrained-front points already checked, by whole molecule: whether
    /// each is modular.
    pub checked: std::collections::HashMap<String, bool>,
    /// Modular designs already found: constrained energy in
    /// kcal/mol, coding log-adaptation on the constrained front's scale, and
    /// the whole molecule.
    pub found: Vec<(f64, f64, String)>,
    /// The log-adaptation, on the constrained front's scale, down to which the
    /// top band has checked every design.
    pub band_down_to: Option<f64>,
    /// The largest gap the search that settled this admitted, in kcal/mol.
    pub epsilon: f64,
}

/// One thing the search has settled, reported as it settles.
#[derive(Debug, Clone, Copy)]
pub enum ModularEvent<'a> {
    /// A constrained-front point was checked.
    Checked {
        sequence: &'a str,
        modular: bool,
        /// `E_iso − E_free` for this design, when this run folded it. `None`
        /// for a verdict carried over from an earlier run, whose progress file
        /// kept the verdict and not the number.
        gap_kcal_mol: Option<f64>,
        /// The pairs of this design's free fold that the constraints forbid,
        /// as `(i, j)` over the whole molecule. What stands between the design
        /// and modularity, and the only part of the verdict a user
        /// can act on: a pair is removed by respelling a codon at a coding
        /// endpoint, and a pair with no coding endpoint cannot be removed at
        /// all. Empty on a design that passed, and also on one that fails by
        /// energy while the recovered free structure is a co-optimal clean
        /// one.
        blocking: &'a [(usize, usize)],
        /// The free fold `blocking` and `gap_kcal_mol` were read from, in
        /// dot-bracket over the whole molecule. Recorded so a progress log
        /// carries the whole per-design check, and a separate `derna check` of
        /// the same front does not fold every design again to recover it.
        /// `None` for a carried verdict, as `gap_kcal_mol`.
        free_structure: Option<&'a str>,
    },
    /// A modular design was found off the constrained front.
    Found {
        mfe_kcal_mol: f64,
        cai_log: f64,
        sequence: &'a str,
    },
    /// The top band has checked every design down to this log-adaptation.
    Band { down_to: f64 },
}

/// Where a search reports what it settles.
pub type Record<'a> = &'a (dyn Fn(ModularEvent<'_>) + Sync);

/// A box a stopped constrained walk left open between two of its points: every
/// front point between them has energy at least `mfe_lo_kcal_mol` and coding
/// log-adaptation in `cai_log_lo..=cai_log_hi`, on the front's own scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpenBox {
    pub mfe_lo_kcal_mol: f64,
    pub cai_log_lo: f64,
    pub cai_log_hi: f64,
}

/// A design on the lattice's own axes: energy in dekacalories and adaptation
/// as the window walk sums it.
#[derive(Debug, Clone, Copy)]
struct Axes {
    energy: f64,
    adaptation: f64,
}

/// What every stretch's walk shares.
#[derive(Clone)]
struct Molecule<'a> {
    params: &'a Params,
    protein: &'a str,
    usage: &'a CodonUsage,
    allowed: Vec<Vec<[u8; 3]>>,
    utr5: &'a str,
    utr3: &'a str,
    constraints: RegionConstraints,
    threads: usize,
    /// How a region nothing caps is searched.
    top: TopSearch,
    /// The largest gap a design may have, in kcal/mol.
    epsilon: f64,
    /// The check's fold cut: a fact about the lattice alone, built once and
    /// shared by every window the search walks, side by side or not.
    cut: Arc<OnceLock<crate::fold_cut::FoldCut>>,
    /// Keeps a whole molecule as a design; `None` keeps every one.
    admit: Option<&'a Admit<'a>>,
    /// When the search stops or holds back a region early.
    policy: SearchPolicy,
}

/// How the search visits a region that no found point caps from above.
///
/// A window walk is the default: it meets the region's designs in order of
/// the walk's slope, so it reaches low-energy modular designs
/// early and leaves an energy floor on whatever it has not covered. The band
/// checks every design from the most adapted down, and above a well-adapted
/// seed that is mostly designs that fail their check.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TopSearch {
    /// Every design there, most adapted first ([`top`]).
    Band,
    /// A window walk, as between found points: designs in order of the
    /// walk's slope, with the region's open end left to its report.
    #[default]
    Window,
}

/// A test a whole molecule must pass to count as a design.
type Admit<'a> = dyn Fn(&str) -> bool + Sync + 'a;

/// The modular front of the molecule `utr5 · protein · utr3`
/// under `constraints`, from its constrained front `constrained` (energy
/// ascending, as [`crate::design_front_walk_with_utr`] returns it), within
/// `max_seconds` of walking. `threads` workers check designs.
///
/// # Errors
///
/// A malformed flank or residue, or a design whose verdict could not be
/// decided.
#[allow(clippy::too_many_arguments)]
pub fn modular_front(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    constrained: &[FrontPoint],
    max_seconds: f64,
    threads: usize,
) -> Result<ModularFront, ModularFrontError> {
    modular_front_resuming(
        params,
        protein,
        usage,
        (utr5, utr3),
        constraints,
        (constrained, &[]),
        (max_seconds, threads),
        (None, None),
        (TopSearch::Band, 0.0, SearchPolicy::default()),
    )
}

/// [`modular_front`] from a constrained front that may be
/// partial, continuing from `prior`, what an earlier search of the same
/// molecule and constrained front settled, and reporting to `record` what this
/// one settles as it settles it.
///
/// `boxes` are the boxes a stopped constrained walk left open between the
/// points of `constrained`; they floor the search beside the points (see the
/// module docs), and are empty for a whole front.
///
/// A point `prior` checked is not checked again, a design it found is found,
/// and the top band starts where it stopped. Everything else is searched as
/// a fresh run would: found designs only cap and prune, so the front a
/// resumed search returns is one a fresh search could return.
///
/// `top` is how a region nothing caps is visited, `epsilon` the largest
/// gap, in kcal/mol, a design on the returned front may have (see the module
/// docs), and `policy` when the search stops or holds back a region early;
/// `prior` must have been settled at the same `epsilon`.
#[allow(clippy::too_many_arguments)]
pub fn modular_front_resuming(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    (utr5, utr3): (&str, &str),
    constraints: RegionConstraints,
    (constrained, boxes): (&[FrontPoint], &[OpenBox]),
    (max_seconds, threads): (f64, usize),
    (prior, record): (Option<&ModularProgress>, Option<Record<'_>>),
    (top, epsilon, policy): (TopSearch, f64, SearchPolicy),
) -> Result<ModularFront, ModularFrontError> {
    let started = Instant::now();
    let m = Molecule {
        params,
        protein,
        usage,
        allowed: synonymous_codons(protein)?,
        utr5,
        utr3,
        constraints,
        threads: threads.max(1),
        top,
        epsilon,
        cut: Arc::new(OnceLock::new()),
        admit: None,
        policy,
    };
    search(
        &m,
        (constrained, boxes),
        (max_seconds, started),
        (prior, record),
    )
}

/// The modular front over the designs of `utr5 · protein ·
/// utr3` that spell none of `motifs` where each pattern's own window forbids
/// it and spell what `pins` pins, from their
/// constrained front `constrained` (energy ascending, as
/// [`crate::motif_front::motif_front_walk_run`] returns it) and the boxes its
/// walk left open, within `max_seconds` of walking. `threads` workers check
/// designs.
///
/// # Errors
///
/// A malformed flank, residue or motif, pins no design spells, or a design
/// whose verdict could not be decided.
#[allow(clippy::too_many_arguments)]
pub fn modular_motif_front<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    (motifs, pins): (&[(S, MotifWindow)], Option<&Pins>),
    (constrained, boxes): (&[FrontPoint], &[OpenBox]),
    max_seconds: f64,
    threads: usize,
) -> Result<ModularFront, ModularFrontError> {
    modular_motif_front_resuming(
        params,
        protein,
        usage,
        (utr5, utr3),
        constraints,
        (motifs, pins),
        (constrained, boxes),
        (max_seconds, threads),
        (None, None),
        (TopSearch::Band, SearchPolicy::default()),
    )
}

/// [`modular_motif_front`] continuing from `prior` and reporting
/// to `record`, as [`modular_front_resuming`] does.
///
/// # Errors
///
/// As [`modular_motif_front`].
#[allow(clippy::too_many_arguments)]
pub fn modular_motif_front_resuming<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    (utr5, utr3): (&str, &str),
    constraints: RegionConstraints,
    (motifs, pins): (&[(S, MotifWindow)], Option<&Pins>),
    (constrained, boxes): (&[FrontPoint], &[OpenBox]),
    (max_seconds, threads): (f64, usize),
    (prior, record): (Option<&ModularProgress>, Option<Record<'_>>),
    (top, policy): (TopSearch, SearchPolicy),
) -> Result<ModularFront, ModularFrontError> {
    let started = Instant::now();
    let (ac, scope) = Ac::for_windows(motifs).map_err(crate::BeamError::from)?;
    let scoped = ScopedAc::new(&ac, scope);
    let admit = |whole: &str| {
        let nucs: Option<Vec<Nucleotide>> = whole.bytes().map(Nucleotide::from_byte).collect();
        nucs.is_some_and(|nucs| scoped.is_clean(&nucs))
    };
    let m = Molecule {
        params,
        protein,
        usage,
        allowed: pinned_codons(protein, (utr5, utr3), pins)?,
        utr5,
        utr3,
        constraints,
        threads: threads.max(1),
        top,
        epsilon: 0.0,
        cut: Arc::new(OnceLock::new()),
        admit: Some(&admit),
        policy,
    };
    search(
        &m,
        (constrained, boxes),
        (max_seconds, started),
        (prior, record),
    )
}

/// The search behind both fronts, over the designs `m` admits.
fn search(
    m: &Molecule<'_>,
    (constrained, open_boxes): (&[FrontPoint], &[OpenBox]),
    (max_seconds, started): (f64, Instant),
    (prior, record): (Option<&ModularProgress>, Option<Record<'_>>),
) -> Result<ModularFront, ModularFrontError> {
    if let Some(p) = prior.filter(|p| p.epsilon != m.epsilon) {
        return Err(ModularFrontError::OtherEpsilon {
            prior: p.epsilon,
            run: m.epsilon,
        });
    }
    let (utr5, protein) = (m.utr5, m.protein);
    let cds = CdsWindow {
        start: utr5.len(),
        end: utr5.len() + 3 * protein.len(),
    };
    let floor = constrained
        .iter()
        .map(|p| m.axes(p))
        .collect::<Result<Vec<_>, _>>()?;
    let offset = match (constrained.first(), floor.first()) {
        (Some(p), Some(a)) => p.cai_log - a.adaptation,
        _ => 0.0,
    };
    // The open boxes as adaptation ranges, and every floor the search reads:
    // the points with each box's corner.
    let boxes: Vec<(f64, f64)> = open_boxes
        .iter()
        .map(|b| (b.cai_log_lo - offset, b.cai_log_hi - offset))
        .collect();
    let mut bound = floor.clone();
    bound.extend(open_boxes.iter().map(|b| Axes {
        energy: (b.mfe_lo_kcal_mol * 100.0).floor(),
        adaptation: b.cai_log_hi - offset,
    }));
    bound.sort_by(|a, b| {
        a.adaptation
            .total_cmp(&b.adaptation)
            .then(a.energy.total_cmp(&b.energy))
    });
    let deadline = max_seconds
        .is_finite()
        .then(|| started + Duration::from_secs_f64(max_seconds.max(0.0)));
    let checks = certify(m, constrained, cds, deadline, prior, record)?;
    let certify_s = started.elapsed().as_secs_f64();
    let unchecked = checks.iter().filter(|c| c.is_none()).count();
    let seeds: Vec<usize> = (0..constrained.len())
        .filter(|&i| checks[i].as_ref().is_some_and(|c| c.0))
        .collect();
    // Each seed's check, kept so the point carries the folds it was decided
    // by rather than leaving a caller to run them again. A seed `prior`
    // checked has none.
    let mut seed_checks: std::collections::HashMap<String, FreeFoldGap> = seeds
        .iter()
        .filter_map(|&i| {
            let gap = checks[i].as_ref()?.1.clone()?;
            Some((constrained[i].sequence.clone(), gap))
        })
        .collect();
    drop(checks);

    let mut found: Vec<(Axes, String)> = seeds
        .iter()
        .map(|&i| (floor[i], constrained[i].sequence.clone()))
        .collect();
    if let Some(prior) = prior {
        found.extend(prior.found.iter().map(|(mfe, cai_log, sequence)| {
            let axes = Axes {
                energy: (mfe * 100.0).round(),
                adaptation: cai_log - offset,
            };
            (axes, sequence.clone())
        }));
    }
    let plan = plan_stretches(&seeds, &floor, &boxes);
    let mut stretches: Vec<Stretch> = plan
        .iter()
        .filter(|p| !p.walk)
        .map(|p| Stretch::unwalked(p, offset))
        .collect();
    // The first round's windows are the stretches between seeds, as regions
    // nothing has walked yet.
    let mut open: Vec<Open> = plan
        .iter()
        .filter(|p| p.walk)
        .map(|p| Open::unwalked(p.lo, p.hi))
        .collect();
    // The top band is the region above the best-adapted seed; what `prior`
    // checked of it is not open.
    if let Some(down_to) = prior.and_then(|p| p.band_down_to) {
        let d = down_to - offset;
        for region in open.iter_mut().filter(|r| r.hi > d) {
            region.hi = d;
        }
        open.retain(|r| r.lo < r.hi);
    }
    let mut share = first_round_s();
    // What a window spends before its walk begins, learnt from the windows
    // this run has already walked. A window pays its fold cut and outside
    // passes whole however little time it is given, so a region whose share
    // cannot cover that would pay it and walk nothing.
    let mut setup_s = 0.0f64;
    let mut cut_build_s = 0.0f64;
    let mut gave_up = None;
    // The regions whose last visit found nothing and left them as they were,
    // by their bounds, with the seconds that visit was given.
    let mut barren: HashMap<(u64, u64), f64> = HashMap::new();
    for round in 0.. {
        if let Some((rounds, least)) = m.policy.give_up_empty {
            let derivations: u64 = stretches.iter().map(|s| s.derivations).sum();
            if round >= rounds && found.is_empty() && derivations >= least {
                gave_up = Some((round, derivations));
                break;
            }
        }
        let remaining = max_seconds - started.elapsed().as_secs_f64();
        // A caller watching memory stops the walks rather than the clock
        // ([`crate::outside::lazy::stop_walks`]). A stopped walk returns at
        // once and finds nothing, so without this a run given a memory
        // budget and no time budget would round forever over regions that
        // can no longer shrink.
        if open.is_empty() || remaining <= 0.0 || crate::outside::lazy::walks_stopped() {
            break;
        }
        let ends = round_ends(started, max_seconds, share);
        // An uncapped region goes first. It is the only one that can hand the
        // round a cap — which bounds every other region's walk — and it holds
        // the best-adapted designs; left last behind many narrow windows it
        // can go a whole budget without being visited at all.
        open.sort_by_key(|r| cap_above(&found, r.hi).is_some());
        let mut next = Vec::new();
        // The round's regions go in waves of up to `lanes`, side by side, the
        // threads split between them; each wave sees what the waves before it
        // found. One lane is one region at a time on every thread.
        let lanes = lane_count(m.threads).clamp(1, open.len());
        let mut walked_any = false;
        for (w, wave) in open.chunks(lanes).enumerate() {
            let waves_left = (open.len() - w * lanes).div_ceil(lanes) as f64;
            // A stop can fire inside a round, and the round's test at the top
            // is then already behind it. Each region of a wave pays its fold
            // cut and its outside passes whole before a stop can reach the
            // walk itself, so a wave starts only while the walks may run; the
            // rest of the round stays open as it is.
            if crate::outside::lazy::walks_stopped() {
                next.extend(wave.iter().cloned());
                continue;
            }
            // The cut is built once, on every thread, before the first wave
            // that walks a window; a search the band alone visits never
            // pays for it.
            if wave.iter().any(|r| !m.goes_to_band(r, &found)) {
                let t = Instant::now();
                m.build_cut()?;
                cut_build_s += t.elapsed().as_secs_f64();
            }
            let seconds = (ends - started.elapsed().as_secs_f64()).max(0.0) / waves_left;
            // A barren region is held to its last visit's time only while
            // another region is open to take the rest.
            let hold = m.policy.hold_barren && open.iter().any(|r| !barren.contains_key(&r.key()));
            let given: Vec<f64> = wave
                .iter()
                .map(|region| match barren.get(&region.key()) {
                    Some(&last) if hold => seconds.min(last),
                    _ => seconds,
                })
                .collect();
            // A wave that cannot pay for its own setup is left for the next
            // round, whose share is twice as long, rather than spending the
            // budget on setup and walking nothing. One wave per round always
            // runs, so a round always makes progress and the loop terminates.
            if walked_any && setup_s > 0.0 && seconds < setup_s {
                next.extend(wave.iter().cloned());
                continue;
            }
            walked_any = true;
            let steps: Vec<Result<Step, ModularFrontError>> = if wave.len() == 1 {
                let seconds = given[0];
                vec![m.visit(round, &wave[0], (&found, &bound), seconds, offset, record)]
            } else {
                let lane = m.with_threads(m.threads / wave.len());
                let (lane, found, bound) = (&lane, &found, &bound);
                std::thread::scope(|scope| {
                    let handles: Vec<_> = wave
                        .iter()
                        .zip(&given)
                        .map(|(region, &seconds)| {
                            scope.spawn(move || {
                                lane.visit(round, region, (found, bound), seconds, offset, record)
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("a region's visit panicked"))
                        .collect()
                })
            };
            for ((region, step), seconds) in wave.iter().zip(steps).zip(given) {
                let step = step?;
                if step.found.is_empty() && step.open.iter().any(|r| r.key() == region.key()) {
                    barren.insert(region.key(), seconds);
                } else {
                    barren.remove(&region.key());
                }
                setup_s = setup_s.max(step.setup_s);
                found.extend(step.found);
                next.extend(step.open);
                stretches.extend(step.stretch);
            }
        }
        open = next;
        share *= 2.0;
    }

    let mut points = merge(found, &open, &bound, offset);
    for p in &mut points {
        p.check = seed_checks.remove(&p.sequence);
    }
    Ok(ModularFront {
        complete: open.is_empty(),
        // Reported against every floor the search read, the boxes' corners
        // with the points: a region starting inside a box is floored by the
        // box, not by the point past it.
        open: open.iter().map(|r| r.on_scale(offset, &bound)).collect(),
        points,
        stretches,
        seeds: seeds.len(),
        unchecked,
        certify_s,
        epsilon: m.epsilon,
        schedule: (lane_count(m.threads), first_round_s()),
        cut_build_s,
        gave_up,
    })
}

/// Every synonymous codon of each residue, and every stop codon at a designed
/// stop ([`crate::residues`]): the window walk's lattice is the whole design
/// space.
fn synonymous_codons(protein: &str) -> Result<Vec<Vec<[u8; 3]>>, ModularFrontError> {
    Ok(crate::residues(protein)?
        .into_iter()
        .map(|a| AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect())
        .collect())
}

/// The codons each residue may take: those `pins` leave it, or every
/// synonymous codon. A pin on a flank must admit the flank's own nucleotide.
fn pinned_codons(
    protein: &str,
    (utr5, utr3): (&str, &str),
    pins: Option<&Pins>,
) -> Result<Vec<Vec<[u8; 3]>>, ModularFrontError> {
    let every = synonymous_codons(protein)?;
    let Some(pins) = pins else {
        return Ok(every);
    };
    let pinned = |e| crate::BeamError::from(Infeasible::Pins(e));
    let layout = Layout::new(utr5.len() as u32, every.len() as u32, utr3.len() as u32);
    for (flank, base) in [(utr5, 0), (utr3, layout.cds_end())] {
        let given = flank
            .bytes()
            .map(|b| Nucleotide::from_byte(b).map(NucSet::of))
            .collect::<Option<Vec<_>>>()
            .ok_or(crate::BeamError::Infeasible)?;
        pins.flank_mask(&layout, base, &given).map_err(pinned)?;
    }
    let aa = crate::residues(protein)?;
    Ok(pins
        .cds_codon_sets(&layout, &aa)
        .map_err(pinned)?
        .unwrap_or(every))
}

impl Molecule<'_> {
    /// The same molecule, its walks and band on `threads` workers.
    fn with_threads(&self, threads: usize) -> Self {
        Self {
            threads: threads.max(1),
            ..self.clone()
        }
    }

    /// Whether `region` is visited by the top band: no found point at or
    /// above it caps it, its lower end is known, and the band is asked for.
    fn goes_to_band(&self, region: &Open, found: &[(Axes, String)]) -> bool {
        cap_above(found, region.hi).is_none()
            && region.lo.is_finite()
            && self.top == TopSearch::Band
    }

    /// Visit one open region in `round` for at most `seconds`: the top band
    /// when nothing found caps it, a window walk otherwise. A window's
    /// designs are reported to `record` here, the band's as each of its
    /// slices finishes.
    fn visit(
        &self,
        round: usize,
        region: &Open,
        (found, floor): (&[(Axes, String)], &[Axes]),
        seconds: f64,
        offset: f64,
        record: Option<Record<'_>>,
    ) -> Result<Step, ModularFrontError> {
        if self.goes_to_band(region, found) {
            return self.band(round, region, offset, seconds, record);
        }
        // A window stopped before it starts would pay its fold cut and its
        // outside passes and walk nothing, so the region stays open as it is.
        // The wave this visit belongs to read the stop before it started;
        // this is the region that loses the race against a stop firing
        // between the two.
        if crate::outside::lazy::walks_stopped() {
            return Ok(Step {
                open: vec![region.clone()],
                ..Step::default()
            });
        }
        let step = self.window(round, region, found, floor, seconds, offset)?;
        if let Some(record) = record {
            for (axes, sequence) in &step.found {
                record(ModularEvent::Found {
                    mfe_kcal_mol: axes.energy / 100.0,
                    cai_log: axes.adaptation + offset,
                    sequence,
                });
            }
        }
        Ok(step)
    }

    /// A constrained-front point on the lattice's axes.
    fn axes(&self, p: &FrontPoint) -> Result<Axes, ModularFrontError> {
        let adaptation = adaptation(
            self.protein,
            self.usage,
            &self.allowed,
            self.utr5,
            self.utr3,
            &p.sequence,
        )?
        .ok_or_else(|| ModularFrontError::NotOnLattice(p.sequence.clone()))?;
        Ok(Axes {
            energy: (p.mfe_kcal_mol * 100.0).round(),
            adaptation,
        })
    }

    /// Visit every design of the uncapped band above a found point, most
    /// adapted first (see [`top`]).
    fn band(
        &self,
        round: usize,
        region: &Open,
        offset: f64,
        seconds: f64,
        record: Option<Record<'_>>,
    ) -> Result<Step, ModularFrontError> {
        let started = Instant::now();
        let band = top::Band::new(
            self.params,
            self.protein,
            self.usage,
            (self.utr5, self.utr3),
            (self.constraints, &self.allowed),
            (self.threads, self.admit),
        )
        .within(self.epsilon);
        let deadline = started + Duration::from_secs_f64(seconds.min(1e9));
        // Each finished slice is reported with what it found, so a search
        // stopped mid-band loses at most the slice in flight.
        let mut settled = |down_to: f64, kept: &[top::Kept]| {
            if let Some(record) = record {
                for (energy, cai_log, sequence) in kept {
                    record(ModularEvent::Found {
                        mfe_kcal_mol: energy / 100.0,
                        cai_log: *cai_log,
                        sequence,
                    });
                }
                record(ModularEvent::Band { down_to });
            }
        };
        let e = band.enumerate(
            region.lo + offset,
            region.hi + offset,
            deadline,
            &mut settled,
        )?;
        let found = e
            .found
            .into_iter()
            .map(|(energy, cai_log, sequence)| {
                let adaptation = cai_log - offset;
                (Axes { energy, adaptation }, sequence)
            })
            .collect::<Vec<_>>();
        let open: Vec<Open> = e
            .open_below
            .map(|hi| Open {
                lo: region.lo,
                hi: hi - offset,
                lines: region.lines.clone(),
            })
            .into_iter()
            .collect();
        let stretch = Stretch {
            round,
            cai_log_lo: on_scale(region.lo, offset),
            cai_log_hi: None,
            walked: true,
            complete: open.is_empty(),
            points: found.len(),
            modular_found: found.len(),
            designs: e.checked,
            slope: 0.0,
            wall_s: started.elapsed().as_secs_f64(),
            threads: self.threads,
            ..Stretch::default()
        };
        Ok(Step {
            found,
            open,
            stretch: Some(stretch),
            setup_s: 0.0,
        })
    }

    /// Walk one open region in `round`, with every point found so far known.
    fn window(
        &self,
        round: usize,
        region: &Open,
        found: &[(Axes, String)],
        floor: &[Axes],
        seconds: f64,
        offset: f64,
    ) -> Result<Step, ModularFrontError> {
        let known: Vec<(i32, f64)> = found
            .iter()
            .map(|(a, _)| (a.energy as i32, a.adaptation))
            .collect();
        let slope = window_slope(region.lo, region.hi, cap_above(found, region.hi), floor);
        let energy: Vec<f64> = if round % 2 == 1 {
            ENERGY_SLOPES.iter().map(|x| x * slope).collect()
        } else {
            Vec::new()
        };
        let Some(r) = self.walk((region.lo, region.hi), slope, &known, &energy, seconds)? else {
            return Ok(Step::default());
        };
        Ok(Step {
            found: r
                .front
                .iter()
                .map(|p| {
                    let axes = Axes {
                        energy: f64::from(p.energy),
                        adaptation: p.cai,
                    };
                    (axes, p.sequence.clone())
                })
                .collect(),
            open: open_regions(&r, region),
            stretch: Some(Stretch::walked(round, region, &r, offset, self.threads)),
            setup_s: r.cut_s + r.energy_s,
        })
    }

    /// Build the check's fold cut on the search's threads, once, before the
    /// first window walk: every window would otherwise build the same cut
    /// alone, single-threaded, before its walk and outside its time share.
    /// The build is outside the round's share too, and a budget does not stop
    /// it.
    fn build_cut(&self) -> Result<(), ModularFrontError> {
        if self.cut.get().is_some() {
            return Ok(());
        }
        let config = ScanConfig {
            threads: self.threads,
            ..ScanConfig::default()
        };
        let fold = verify(self.threads).fold;
        if !matches!(fold, ExactFold::SparseCut | ExactFold::SparseCutInside) {
            return Ok(());
        }
        let inside = fold == ExactFold::SparseCutInside;
        if let Some(cut) = crate::fold_cut::FoldCut::build(
            self.params,
            self.protein,
            self.usage,
            &self.allowed,
            self.utr5,
            self.utr3,
            config,
            inside,
        )? {
            let _ = self.cut.set(cut);
        }
        Ok(())
    }

    /// Walk one window for at most `seconds`, on a thread with room for the
    /// walk's recursion.
    ///
    /// The memory the previous window's walk freed is handed back first: the
    /// system allocator keeps what a finished walk freed rather than returning
    /// it, and resident memory, which the memory budget reads, still counts
    /// it. On a 1,273-residue window walk the process held 7.0 GB at a
    /// window's start and 1.0 GB once it was handed back.
    fn walk(
        &self,
        window: (f64, f64),
        slope: f64,
        known: &[(i32, f64)],
        energy_slopes: &[f64],
        seconds: f64,
    ) -> Result<Option<WindowReport>, ModularFrontError> {
        crate::motif_front::release_freed();
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(WALK_STACK)
                .spawn_scoped(scope, || {
                    window_front_sharing(
                        self.params,
                        self.protein,
                        self.usage,
                        &self.allowed,
                        self.utr5,
                        self.utr3,
                        self.constraints,
                        // The walk's own scan runs on the region's threads; the
                        // scan reaches the same optimum on any number of them.
                        ScanConfig {
                            threads: self.threads,
                            ..ScanConfig::default()
                        },
                        window,
                        slope,
                        known,
                        energy_slopes,
                        u64::MAX,
                        seconds,
                        verify(self.threads),
                        walk(),
                        (Some(self.cut.as_ref()), epsilon_dekacal(self.epsilon)),
                        self.admit,
                    )
                })
                .expect("spawn a window walk")
                .join()
                .expect("a window walk panicked")
        })
        .map_err(ModularFrontError::from)
    }
}

/// What one region's visit found, what it left open, and its record.
#[derive(Default)]
struct Step {
    found: Vec<(Axes, String)>,
    open: Vec<Open>,
    stretch: Option<Stretch>,
    /// What this visit spent before its walk could begin: the fold cut and
    /// the energy bound's outside passes. A window pays it whole however
    /// little time it is given, so a later region whose share is below it
    /// would pay it and walk nothing.
    setup_s: f64,
}

/// When a round's share of the time runs out, in seconds from the start: its
/// share, or what is left of the budget when that is less.
fn round_ends(started: Instant, max_seconds: f64, share: f64) -> f64 {
    (started.elapsed().as_secs_f64() + share).min(max_seconds)
}

/// The found point that caps a region whose top is `hi`: the least adapted
/// one at or above it, which is also the most stable.
fn cap_above(found: &[(Axes, String)], hi: f64) -> Option<Axes> {
    found
        .iter()
        .map(|(a, _)| *a)
        .filter(|a| a.adaptation >= hi - CAI_TOL)
        .min_by(|a, b| a.adaptation.total_cmp(&b.adaptation))
}

/// The slope a window is walked at. Capped, the one minimizing the band of
/// designs a walk must pass to cover it (`band_slope`); uncapped (no seed at
/// all, so one window spans everything), the chord of the constrained front,
/// so the walk starts where designs sit rather than at the lattice's best
/// energy. An uncapped band above a found point is not walked but enumerated
/// (see [`top`]).
fn window_slope(lo: f64, hi: f64, cap: Option<Axes>, floor: &[Axes]) -> f64 {
    let lo_seen = if lo.is_finite() {
        lo
    } else {
        floor.first().map_or(0.0, |p| p.adaptation)
    };
    if let Some(cap) = cap {
        if cap.adaptation > lo_seen + CAI_TOL {
            return band_slope(floor, lo_seen, cap.energy);
        }
    }
    let inside: Vec<&Axes> = floor
        .iter()
        .filter(|p| p.adaptation >= lo_seen - CAI_TOL && p.adaptation <= hi + CAI_TOL)
        .collect();
    let pool: Vec<&Axes> = if inside.is_empty() {
        floor.iter().collect()
    } else {
        inside
    };
    let (Some(first), Some(last)) = (pool.first(), pool.last()) else {
        return 0.0;
    };
    // The floor carries each open box's corner beside the front's points, at
    // the adaptation of the point that closes the box from above and at the
    // box's least energy. A chord ending on that corner instead of the point
    // runs flat across the box, and a flat slope orders the walk by energy
    // alone, so it must pass every design down the box's whole energy span
    // before it reaches one in the box. Each end takes the front's own point:
    // the most stable entry at the low end and the least stable at the high.
    let a = pool
        .iter()
        .filter(|p| p.adaptation <= first.adaptation + CAI_TOL)
        .min_by(|x, y| x.energy.total_cmp(&y.energy))
        .copied()
        .copied()
        .unwrap_or(**first);
    let b = pool
        .iter()
        .filter(|p| p.adaptation >= last.adaptation - CAI_TOL)
        .max_by(|x, y| x.energy.total_cmp(&y.energy))
        .copied()
        .copied()
        .unwrap_or(**last);
    if b.adaptation > a.adaptation + CAI_TOL {
        ((b.energy - a.energy) / (b.adaptation - a.adaptation)).max(0.0)
    } else {
        0.0
    }
}

/// The slope that minimizes the band a walk capped at energy `cap` must
/// traverse to cover adaptations from `lo`: `cap − μ·lo` less the least
/// `E − μ·A` over the constrained front, which bounds every design. The band
/// is convex in `μ` (a line less a concave envelope), so a golden-section
/// search on `ln μ` over `[1, 10⁶]` finds it.
fn band_slope(floor: &[Axes], lo: f64, cap: f64) -> f64 {
    let band = |mu: f64| {
        cap - mu * lo
            - floor
                .iter()
                .map(|p| p.energy - mu * p.adaptation)
                .fold(f64::INFINITY, f64::min)
    };
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (0.0f64, 1e6f64.ln());
    for _ in 0..200 {
        let (c, d) = (b - r * (b - a), a + r * (b - a));
        if band(c.exp()) <= band(d.exp()) {
            b = d;
        } else {
            a = c;
        }
    }
    ((a + b) / 2.0).exp()
}

/// The check's settings: every lever exact, the lazy probe's defaults.
fn verify(threads: usize) -> Verify {
    Verify {
        threads,
        witnesses: 64,
        witness_stop: 1,
        witness_promote: true,
        fold_reuse: 512,
        fold_reuse_rows: true,
        fold_slots: 8,
        handoff_batch: 16,
        fold: ExactFold::SparseCut,
        cut: CutSchedule::Before,
        window: Some((150, 150)),
        window_utr: false,
        window_reuse: false,
        window_auto: false,
        incumbent: None,
    }
}

/// The walk's settings. [`Walk::dedup_index`] is on: it answers the same
/// membership question as a set of fingerprints, since a probe reads a
/// derivation back only once the hashes agree and then compares the whole
/// fingerprint, so the walk visits the same designs in the same order while
/// each node's dedup set takes about a third fewer bytes. [`Walk::tight_lists`]
/// stays off: the growth slack it removes is capacity that was never written,
/// so it shrinks the lists' reserved bytes without shrinking resident memory.
fn walk() -> Walk {
    Walk {
        build: Build::Production,
        node_dedup: true,
        incremental_spell: true,
        pipelined_visit: false,
        spellers: 0,
        dedup_index: true,
        tight_lists: false,
    }
}

/// One point's check: whether it passed, and its gap where it was folded here.
type Verdict = (bool, Option<FreeFoldGap>);

/// Each constrained-front point's check — its own free fold against its own
/// constrained energy — on up to `m.threads` workers, until `deadline`. A point
/// is a seed when its check passes, its gap at most `m.epsilon`; `None` is a point the deadline left
/// unchecked. The constrained energy is the point's own, once its structure is
/// re-scored and found allowed
/// ([`sequence_gap_given_structure`](crate::free_fold::sequence_gap_given_structure)); a point
/// whose structure does not hold is folded under the constraint as well.
///
/// The budget governs this stage like any other: at length it is a fold of
/// the whole molecule per constrained point, hours on its own, and nothing is
/// written until it ends. An unchecked point is simply not a seed: the
/// stretch around it stays open, which only widens what is left to search.
/// The points are taken in runs of consecutive points, the runs in
/// bit-reversed order, so a deadline that stops the stage part-way leaves
/// seeds spread across the whole front rather than a prefix of it. Neighbours
/// on the front differ in few codons, so each worker's free fold copies every
/// cell it can from its last few folds
/// ([`sequence_gap_given_structure_reusing`]); the verdicts are the fresh
/// folds'.
///
/// A point `prior` checked keeps that verdict, with no check; every point
/// checked here is reported to `record` as its verdict lands.
fn certify(
    m: &Molecule,
    points: &[FrontPoint],
    cds: CdsWindow,
    deadline: Option<Instant>,
    prior: Option<&ModularProgress>,
    record: Option<Record<'_>>,
) -> Result<Vec<Option<Verdict>>, ModularFrontError> {
    type Checked = Result<Verdict, FreeFoldError>;
    let one = |p: &FrontPoint, pool: &mut VecDeque<FreeFoldTables>| -> Checked {
        if let Some(&passed) = prior.and_then(|prior| prior.checked.get(&p.sequence)) {
            return Ok((passed, None));
        }
        let held: Vec<&FreeFoldTables> = pool.iter().collect();
        let (gap, tables) = sequence_gap_given_structure_reusing(
            m.params,
            &p.sequence,
            &p.structure,
            p.mfe_kcal_mol,
            cds,
            m.constraints,
            &held,
        )?;
        if let Some(tables) = tables {
            if pool.len() == certify_pool(p.sequence.len()) {
                pool.pop_front();
            }
            pool.push_back(tables);
        }
        let passed = gap.is_within(m.epsilon);
        if let Some(record) = record {
            record(ModularEvent::Checked {
                sequence: &p.sequence,
                modular: passed,
                gap_kcal_mol: Some(gap.gap_kcal_mol),
                blocking: &gap.reinstated_pairs,
                free_structure: Some(&gap.free_structure),
            });
        }
        Ok((passed, Some(gap)))
    };
    let workers = m.threads.min(points.len()).max(1);
    let run = certify_run(points.len(), workers);
    let runs = spread_order(points.len().div_ceil(run));
    let mut verdicts: Vec<Option<Checked>> = (0..points.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let runs = &runs;
                let one = &one;
                scope.spawn(move || {
                    let mut pool = VecDeque::new();
                    runs.iter()
                        .skip(w)
                        .step_by(workers)
                        .flat_map(|&r| r * run..((r + 1) * run).min(points.len()))
                        .take_while(|_| deadline.is_none_or(|d| Instant::now() < d))
                        .map(|i| (i, one(&points[i], &mut pool)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, v) in h.join().expect("a certifying worker") {
                verdicts[i] = Some(v);
            }
        }
    });
    verdicts.into_iter().map(|v| Ok(v.transpose()?)).collect()
}

/// How many consecutive front points one certifying worker takes at a time:
/// up to 16, so a run's later points copy from its earlier ones, but short
/// enough that every worker gets about four runs.
fn certify_run(points: usize, workers: usize) -> usize {
    (points / (4 * workers)).clamp(1, 16)
}

/// How many of its last free folds a certifying worker keeps to copy from: up
/// to 8, and about 256 MiB of tables (three `i32` triangles per fold).
fn certify_pool(len: usize) -> usize {
    let bytes = 12 * len * (len + 1) / 2;
    ((256usize << 20) / bytes.max(1)).clamp(1, 8)
}

/// `0..n` in bit-reversed order: every prefix is spread evenly over the
/// range, so stopping part-way leaves no long run untouched.
fn spread_order(n: usize) -> Vec<usize> {
    let bits = usize::BITS - n.saturating_sub(1).leading_zeros();
    let mut order: Vec<usize> = (0..(1usize << bits))
        .map(|i| i.reverse_bits() >> (usize::BITS - bits.max(1)))
        .filter(|&i| i < n)
        .collect();
    if bits == 0 {
        order = (0..n).collect();
    }
    order
}

/// One stretch to walk: adaptations `(lo, hi]` on the lattice's axes, capped
/// by the seed at `hi` when there is one.
#[derive(Debug, Clone)]
struct Plan {
    lo: f64,
    hi: f64,
    walk: bool,
}

/// The stretches between consecutive seeds, and below the first and above
/// the last. With no seed, one stretch over the whole range.
fn plan_stretches(seeds: &[usize], floor: &[Axes], boxes: &[(f64, f64)]) -> Vec<Plan> {
    let boxed = |lo: f64, hi: f64| {
        boxes
            .iter()
            .any(|&(b_lo, b_hi)| b_hi > lo + CAI_TOL && b_lo < hi - CAI_TOL)
    };
    let mut bounds: Vec<Option<usize>> = vec![None];
    bounds.extend(seeds.iter().map(|&s| Some(s)));
    bounds.push(None);
    bounds
        .windows(2)
        .map(|w| {
            let (below, above) = (w[0], w[1]);
            // Adjacent on the constrained front, with no open box between:
            // a design between them would be dominated by the upper seed, so
            // there is nothing to find.
            let adjacent = matches!((below, above), (Some(a), Some(b))
                if b == a + 1 && !boxed(floor[a].adaptation, floor[b].adaptation));
            // Nothing lies below the most stable point of the constrained
            // front's own end unless that end failed its check.
            let empty_end = matches!((below, above), (None, Some(0)))
                && !boxed(f64::NEG_INFINITY, floor[0].adaptation);
            Plan {
                lo: below.map_or(f64::NEG_INFINITY, |i| floor[i].adaptation),
                hi: above.map_or(f64::INFINITY, |i| floor[i].adaptation),
                walk: !adjacent && !empty_end,
            }
        })
        .collect()
}

impl Stretch {
    fn unwalked(p: &Plan, offset: f64) -> Self {
        Self {
            round: 0,
            cai_log_lo: on_scale(p.lo, offset),
            cai_log_hi: on_scale(p.hi, offset),
            walked: false,
            complete: true,
            ..Self::default()
        }
    }

    fn walked(round: usize, region: &Open, r: &WindowReport, offset: f64, threads: usize) -> Self {
        Self {
            round,
            cai_log_lo: on_scale(region.lo, offset),
            cai_log_hi: on_scale(region.hi, offset),
            walked: true,
            complete: r.complete,
            points: r.front.len(),
            modular_found: r.found,
            designs: 0,
            slope: r.slope,
            wall_s: r.wall_s,
            threads,
            cut_s: r.cut_s,
            energy_s: r.energy_s,
            verify_s: r.verify_s,
            folds: r.folds,
            reached: r.reached,
            offered: r.offered,
            build_s: r.stats.build_s,
            sweep_s: r.stats.sweep_s,
            walk_s: r.stats.walk_s,
            derivations: r.stats.derivations,
            stopped: r.stats.stopped,
        }
    }
}

/// An adaptation on the constrained front's scale, or `None` for an open end.
fn on_scale(a: f64, offset: f64) -> Option<f64> {
    a.is_finite().then_some(a + offset)
}

/// An open region on the lattice's axes.
#[derive(Debug, Clone)]
struct Open {
    lo: f64,
    hi: f64,
    /// Every walk that reached the region leaves a floor `last_g + μ·A` under
    /// it; each is a bound, so the highest holds. Empty when no walk reached
    /// it, and the constrained front is then the only floor.
    lines: Vec<(f64, f64)>,
}

impl Open {
    /// The region's bounds, exactly: a region re-entered with the same key
    /// is the one its last visit left.
    fn key(&self) -> (u64, u64) {
        (self.lo.to_bits(), self.hi.to_bits())
    }

    /// A region nothing has walked yet.
    fn unwalked(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            lines: Vec::new(),
        }
    }

    /// The highest of the walks' floors at adaptation `x`.
    fn walked_floor(&self, x: f64) -> f64 {
        self.lines
            .iter()
            .map(|&(g, mu)| g + mu * x)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The least energy a design in the region with adaptation at least `a`
    /// can have: the walks' floors, and never below the constrained front.
    /// Every floor rises with adaptation, so the least is at the region's
    /// lowest adaptation at or above `a`.
    fn floor_from(&self, a: f64, floor: &[Axes]) -> Option<f64> {
        if self.hi < a - CAI_TOL {
            return None;
        }
        let x = self.lo.max(a);
        Some(self.walked_floor(x).max(staircase(floor, x)))
    }

    /// The region as reported: of the walks' floors the one that is highest at
    /// its lowest adaptation, which is a bound wherever it is not the highest,
    /// and the constrained front's floor beside it.
    fn on_scale(&self, offset: f64, floor: &[Axes]) -> OpenRegion {
        let best = self
            .lines
            .iter()
            .filter(|_| self.lo.is_finite())
            .max_by(|a, b| (a.0 + a.1 * self.lo).total_cmp(&(b.0 + b.1 * self.lo)));
        let constrained = staircase(floor, self.lo);
        OpenRegion {
            cai_log_lo: on_scale(self.lo, offset),
            cai_log_hi: on_scale(self.hi, offset),
            mfe_floor_kcal_mol: best.map(|&(g, mu)| (g + mu * self.lo) / 100.0),
            slope_kcal_mol: best.map_or(0.0, |&(_, mu)| mu / 100.0),
            // Unlike the walks' line, this floor does not need a finite `lo`
            // to evaluate: over a region open at its lowest adaptation it is
            // the constrained front's own minimum. That is the region no walk
            // has reached, where this is the only floor there is.
            mfe_floor_constrained_kcal_mol: constrained.is_finite().then_some(constrained / 100.0),
        }
    }
}

/// The least constrained energy of any design with adaptation at least `a`:
/// the energy of the first constrained-front point at or above it.
fn staircase(floor: &[Axes], a: f64) -> f64 {
    floor
        .iter()
        .find(|p| p.adaptation >= a - CAI_TOL)
        .map_or(f64::INFINITY, |p| p.energy)
}

/// What a walk of `parent` left open, each region keeping the parent's floors
/// beside the walk's own.
fn open_regions(r: &WindowReport, parent: &Open) -> Vec<Open> {
    if r.complete {
        return Vec::new();
    }
    let mut lines = parent.lines.clone();
    lines.extend(r.last_g.map(|g| (g, r.slope)));
    r.open
        .iter()
        .map(|&(lo, hi)| Open {
            lo,
            hi,
            lines: lines.clone(),
        })
        .collect()
}

/// The non-dominated points among `found`, energy ascending, each marked
/// certified unless an open region can hold a design dominating it.
fn merge(
    mut found: Vec<(Axes, String)>,
    open: &[Open],
    floor: &[Axes],
    offset: f64,
) -> Vec<ModularPoint> {
    // The sequence breaks a tie in both axes, so which of several co-optimal
    // designs represents a point does not depend on the order they were
    // found in: a resumed search reports the same molecules as an
    // uninterrupted one.
    found.sort_by(|(a, x), (b, y)| {
        a.energy
            .total_cmp(&b.energy)
            .then(b.adaptation.total_cmp(&a.adaptation))
            .then_with(|| x.cmp(y))
    });
    let mut front: Vec<(Axes, String)> = Vec::new();
    for (axes, sequence) in found {
        // Energy ascending, so a point is on the front exactly when it is
        // better adapted than every point kept before it.
        if front
            .last()
            .is_none_or(|(kept, _)| axes.adaptation > kept.adaptation + CAI_TOL)
        {
            front.push((axes, sequence));
        }
    }
    front
        .into_iter()
        .map(|(axes, sequence)| ModularPoint {
            certified: !open.iter().any(|o| {
                o.floor_from(axes.adaptation, floor)
                    .is_some_and(|e| e < axes.energy - ENERGY_TOL)
            }),
            sequence,
            mfe_kcal_mol: axes.energy / 100.0,
            cai_log: axes.adaptation + offset,
            check: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(energy: f64, adaptation: f64) -> Axes {
        Axes { energy, adaptation }
    }

    #[test]
    fn an_uncapped_window_slope_runs_to_the_point_that_closes_the_box_not_its_corner() {
        // A front stopped with one box open between its points at adaptations
        // -98.293 and 0: the floor holds both points and the box's corner at
        // the upper point's adaptation (shifted by rounding) and the box's
        // least energy. The chord is the points', not the corner's.
        let floor = vec![
            at(-214_060.0, -98.293),
            at(-141_960.0, 0.0),
            at(-214_060.0, 1.1e-13),
        ];
        let slope = window_slope(-98.293, f64::INFINITY, None, &floor);
        let chord = (-141_960.0 - -214_060.0) / 98.293;
        assert!((slope - chord).abs() < 1e-6, "{slope} vs {chord}");
        assert!(slope > 0.0, "a flat chord orders the walk by energy alone");
    }

    #[test]
    fn the_spread_order_visits_every_point_once_and_spreads_its_prefixes() {
        for n in [0, 1, 2, 3, 5, 8, 13, 100, 757] {
            let order = spread_order(n);
            let mut seen = order.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n = {n}");
        }
        // A quarter of the way through, no stretch of the front is untouched
        // for longer than about four points.
        let order = spread_order(757);
        let mut first: Vec<usize> = order[..757 / 4].to_vec();
        first.sort_unstable();
        let widest = first.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        assert!(widest <= 8, "widest untouched run {widest}");
        assert!(first[0] == 0 && *first.last().unwrap() > 700);
    }

    /// A constrained front: energies rise with adaptation.
    fn floor() -> Vec<Axes> {
        vec![at(-500.0, -2.0), at(-450.0, -1.0), at(-400.0, 0.0)]
    }

    fn open(lo: f64, hi: f64, last_g: Option<f64>, slope: f64) -> Open {
        Open {
            lo,
            hi,
            lines: last_g.map(|g| (g, slope)).into_iter().collect(),
        }
    }

    fn certified(found: Vec<(Axes, String)>, regions: &[Open]) -> Vec<bool> {
        merge(found, regions, &floor(), 0.0)
            .iter()
            .map(|p| p.certified)
            .collect()
    }

    #[test]
    fn a_point_is_provisional_only_where_an_open_region_can_dominate_it() {
        let point = || vec![(at(-420.0, -1.5), "p".to_string())];
        // No open region: proven.
        assert_eq!(certified(point(), &[]), [true]);
        // A region above it whose floor reaches below its energy: a design
        // there could dominate it.
        assert_eq!(
            certified(point(), &[open(-1.2, -0.5, Some(-500.0), 50.0)]),
            [false]
        );
        // The same region with its floor lifted above the point: it cannot.
        assert_eq!(
            certified(point(), &[open(-1.2, -0.5, Some(-300.0), 50.0)]),
            [true]
        );
        // A region wholly below its adaptation cannot dominate it.
        assert_eq!(
            certified(point(), &[open(-3.0, -1.8, Some(-900.0), 50.0)]),
            [true]
        );
        // A walk that reached nothing leaves the constrained front as the
        // only floor: above the point it reaches -450 < -420.
        assert_eq!(certified(point(), &[open(-1.2, -0.5, None, 0.0)]), [false]);
    }

    #[test]
    fn a_region_inside_an_open_box_reports_the_boxs_floor() {
        // A constrained front stopped with a box open between its points at
        // adaptations -2 and 0: the box's corner floors every design in it at
        // -500, the energy of its stable end. A region starting inside the box
        // is floored there, not at -400, the point past the box — a floor that
        // would clear a point a design in the box could dominate.
        let points = vec![at(-500.0, -2.0), at(-400.0, 0.0)];
        let bound = vec![at(-500.0, -2.0), at(-500.0, 0.0), at(-400.0, 0.0)];
        let region = open(-1.0, 0.0, None, 0.0);
        let reported = region.on_scale(0.0, &bound);
        assert_eq!(reported.mfe_floor_constrained_kcal_mol, Some(-5.0));
        assert_eq!(
            region.on_scale(0.0, &points).mfe_floor_constrained_kcal_mol,
            Some(-4.0),
            "the points alone overstate the floor, so the case discriminates"
        );
        let point = at(-420.0, -0.5);
        let found = vec![(point, "p".to_string())];
        let certified = merge(found, std::slice::from_ref(&region), &bound, 0.0)[0].certified;
        assert!(!certified, "a design in the box could dominate the point");
        assert!(reported.could_dominate(point.energy / 100.0, point.adaptation));
    }

    #[test]
    fn could_dominate_agrees_with_the_searchs_own_certification() {
        // `OpenRegion::could_dominate` exists so that a caller merging several
        // searches can ask another search's leftover regions the question this
        // one asks its own. The risk is that the two predicates drift, so this
        // pins the public one to `merge` over the same cases rather than
        // restating them: wherever the region cannot dominate, the search must
        // have certified the point.
        //
        // The implication holds one way only. The reported region carries the
        // constrained floor flat at its low end, where `merge` evaluates the
        // staircase at the adaptation asked about, so `could_dominate` reads a
        // floor no higher and fires no less often. It may therefore contest a
        // point the search certified; it may never clear one the search did
        // not.
        let cases = [
            open(-1.2, -0.5, Some(-500.0), 50.0),
            open(-1.2, -0.5, Some(-300.0), 50.0),
            open(-3.0, -1.8, Some(-900.0), 50.0),
            open(-1.2, -0.5, None, 0.0),
            open(f64::NEG_INFINITY, -0.5, None, 0.0),
            open(-1.2, f64::INFINITY, Some(-500.0), 50.0),
        ];
        let point = at(-420.0, -1.5);
        let (mut cleared, mut contested) = (0, 0);
        for region in &cases {
            let certified =
                certified(vec![(point, "p".to_string())], std::slice::from_ref(region))[0];
            let reported = region.on_scale(0.0, &floor());
            let dominates = reported.could_dominate(point.energy / 100.0, point.adaptation);
            if dominates {
                contested += 1;
            } else {
                cleared += 1;
                assert!(
                    certified,
                    "could_dominate cleared a point the search left provisional: {reported:?}"
                );
            }
        }
        // Neither arm of the implication is vacuous: some case is contested and
        // some is cleared, so the assertion above was actually reached.
        assert!(
            cleared > 0 && contested > 0,
            "{cleared} cleared, {contested} contested"
        );
    }

    #[test]
    fn a_region_keeps_the_highest_floor_of_every_walk_that_reached_it() {
        let point = || vec![(at(-420.0, -1.5), "p".to_string())];
        // One walk's floor alone admits a dominating design; a second walk's,
        // higher there, rules it out, and the region holds both.
        let mut region = open(-1.2, -0.5, Some(-500.0), 50.0);
        assert_eq!(certified(point(), std::slice::from_ref(&region)), [false]);
        region.lines.push((-300.0, 50.0));
        assert_eq!(certified(point(), &[region]), [true]);
    }

    #[test]
    fn a_seed_is_always_certified() {
        // A seed sits on the constrained front, which floors every open region.
        let seed = vec![(at(-450.0, -1.0), "s".to_string())];
        assert_eq!(certified(seed, &[open(-1.0, 0.0, None, 0.0)]), [true]);
    }

    #[test]
    fn the_merge_keeps_only_undominated_points() {
        let found = vec![
            (at(-450.0, -1.0), "a".to_string()),
            (at(-440.0, -1.5), "dominated".to_string()),
            (at(-400.0, 0.0), "b".to_string()),
        ];
        let kept: Vec<String> = merge(found, &[], &floor(), 0.0)
            .into_iter()
            .map(|p| p.sequence)
            .collect();
        assert_eq!(kept, ["a", "b"]);
    }

    #[test]
    fn seeds_adjacent_on_the_constrained_front_leave_nothing_to_walk() {
        let floor = vec![at(-5.0, 0.0), at(-4.0, 1.0), at(-3.0, 2.0), at(-2.0, 3.0)];
        let plan = plan_stretches(&[0, 1, 3], &floor, &[]);
        let walked: Vec<bool> = plan.iter().map(|p| p.walk).collect();
        // Below the most stable seed, 0..1 (adjacent), 1..3 (a gap), above 3.
        assert_eq!(walked, [false, false, true, true]);
        // No seed: one stretch over everything.
        let none = plan_stretches(&[], &floor, &[]);
        assert_eq!(none.len(), 1);
        assert!(none[0].walk);
        // An open box between two adjacent seeds leaves a stretch to walk.
        let boxed = plan_stretches(&[0, 1, 3], &floor, &[(0.2, 0.8)]);
        let walked: Vec<bool> = boxed.iter().map(|p| p.walk).collect();
        assert_eq!(walked, [false, true, true, true]);
    }
}
