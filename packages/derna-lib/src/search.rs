//! Bounded best-first search for the optimal **modular** design.
//!
//! Modularity relates two optima of one sequence — the constrained fold and the
//! unconstrained one — so it is a property of a finished design and cannot enter the
//! design DP as a cost. What a search *can* do is decide which parts of the
//! design space are worth solving at all, and that is what this module does:
//! branch-and-bound over **regions** of the space, where a region pins some
//! positions to chosen codons and forbids particular codons at others.
//!
//! # Why it is exact
//!
//! Three properties carry it.
//!
//! **Soundness of the stop.** The heap key of a region is a lower bound on the
//! objective over it and the heap is min-ordered, so a key at or above the
//! incumbent means every region still queued is worse. Popping such a key ends
//! the search with the incumbent proven optimal.
//!
//! **Soundness of the prune.** The bound is the *constrained* restricted
//! optimum, `lb_iso(R) = minᵥ∈ᵣ λ·E_iso(v) − (1−λ)·CAI(v)`. For a modular `v`,
//! `E_free(v) = E_iso(v)`, so the objective the search compares incumbents by —
//! computed on the free fold — equals the constrained objective, and is
//! therefore at least `lb_iso(R)`. Only modular candidates ever become the
//! incumbent, so discarding a region whose bound reaches the incumbent discards
//! nothing that could win. The argument holds under soft enforcement too: a
//! design is modular only if some minimum-energy structure incurs no penalty at
//! all, so the penalty term cannot lift `E_iso` above `E_free` for a candidate
//! that could win.
//!
//! **Tiling.** The regions a partition produces cover everything in the parent
//! except the parent's own optimum, each exactly once. Any sequence in the
//! parent other than that optimum differs from it at some position; the first
//! such position *in the branch order* selects the one part it lies in. The
//! order is free — which is what leaves room for the refinement below.
//!
//! # The two refinements
//!
//! Without both of these it is generate-and-test with extra bookkeeping.
//!
//! **Lazy bound refinement.** A child is a subset of its parent, so the parent's
//! bound is already valid for it. Children enter the heap carrying that
//! inherited bound and are solved only when popped, at which point the region is
//! re-pushed under its own refined bound. Cost then accrues per **popped**
//! region rather than per pushed one, and a region whose refined bound reaches
//! the incumbent is discarded without ever being verified or branched.
//!
//! **Violation-guided branching.** When a region's optimum verifies dirty, the
//! free fold names the pairs the molecule insisted on. The positions spelling
//! those pairs lead the branch order, so the parts that vary them are cut out
//! first. Tiling — and hence exactness — is unchanged; only the order is. This
//! is the one feedback available *before* an incumbent exists, since the prune
//! cannot fire until then.
//!
//! # The two things a caller can hand it
//!
//! Both are off by default and neither can change the answer.
//!
//! **A seed** (`SearchInputs::seed`). Until the first modular design is found the
//! incumbent is `+∞` and no region is ever discarded, so in that phase this is a
//! region-granular filter: one sequence tested per region solved. A seed removes
//! that phase by starting the incumbent at a modular design found elsewhere —
//! any modular design of this space will do, because the bound's lemma constrains
//! what an incumbent *is* and not where it came from. It is verified rather than
//! trusted, since an incumbent taken on the caller's word is the one way to make
//! this search report `Proved` over a wrong answer.
//!
//! **A width** (`SearchInputs::search_threads`). The regions on the heap are
//! independent whole-mRNA solves, so a batch of them can be examined side by
//! side. Examining is purely functional; every decision the results feed is made
//! afterwards in the order the entries were popped, so a wide walk passes through
//! exactly the incumbents a narrow one does.
//!
//! The division of labour between them is not a preference. A seed changes what a
//! **budget-limited** run returns and cannot shorten a **proof**: a best-first
//! walk must open every region whose bound falls below the optimum, and that set
//! is fixed by `lb` and by the answer, not by when the incumbent arrived. A width
//! is the lever on the proof, and the only other ones are a tighter bound — which
//! shrinks the set — or cheaper regions.
//!
//! # What it is not
//!
//! Its advantage over walking candidates in objective order is granularity — a
//! prune rejects a whole region unopened rather than one candidate at a time.
//! Against a genuinely empty modular set no traversal order, seed or width helps,
//! and emptiness can only be established by exhausting the heap.

use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

use derna_codon_dp::codon_table::AMINO_ACIDS;
use thiserror::Error;

use crate::batch_probe::{BatchProbeReport, BatchRow};
use crate::bound_probe::{
    BoundProbeReport, BoundProbeRow, MfeFloors, ProbeInstant, Probed, RelaxedPrice,
};
use crate::dfa;
use crate::free_fold::{sequence_gap, CdsWindow, FreeFoldError, FreeFoldGap};
use crate::region::RegionConstraints;
use crate::{design_restricted_with_region, BeamError, CodonUsage, Design, Params, ScanConfig};

/// One region of the design space: `pinned[i]` fixes protein position `i` to a
/// codon when set, `excluded[i]` lists codons that position may not take, and
/// anything else is free.
///
/// Exclusions are held **per position** rather than at one distinguished branch
/// position, and that is what buys the freedom to reorder branching. A partition
/// that visits positions in an order of its choosing produces parts that leave
/// the parent's other branch positions free; those parts must still refuse what
/// the parent refused, or they cover more than their parent did and the same
/// sequences are enumerated again and again.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DesignRegion {
    pinned: Vec<Option<[u8; 3]>>,
    excluded: Vec<Vec<[u8; 3]>>,
}

impl DesignRegion {
    /// The whole design space: every position free, nothing excluded.
    fn root(m: usize) -> Self {
        Self {
            pinned: vec![None; m],
            excluded: vec![Vec::new(); m],
        }
    }

    /// The singleton region admitting exactly one encoding: every position
    /// pinned to the codon it spells.
    fn singleton(codons: &[[u8; 3]]) -> Self {
        Self {
            pinned: codons.iter().copied().map(Some).collect(),
            excluded: vec![Vec::new(); codons.len()],
        }
    }

    /// The per-position codon sets this region admits. A pinned position is its
    /// one codon; a free one is every synonymous codon its exclusions leave.
    fn allowed(&self, aa: &[usize]) -> Vec<Vec<[u8; 3]>> {
        aa.iter()
            .enumerate()
            .map(|(i, &a)| match &self.pinned[i] {
                Some(codon) => vec![*codon],
                None => AMINO_ACIDS[a]
                    .codons
                    .iter()
                    .map(|c| c.nuc)
                    .filter(|c| !self.excluded[i].contains(c))
                    .collect(),
            })
            .collect()
    }

    /// The positions this region leaves free, ascending.
    fn free_positions(&self) -> Vec<usize> {
        (0..self.pinned.len())
            .filter(|&i| self.pinned[i].is_none())
            .collect()
    }

    /// `self` with one more codon excluded at position `q` — the set a
    /// partition's part is a subset of when `q` is the position it excludes at.
    fn excluding(&self, q: usize, codon: [u8; 3]) -> DesignRegion {
        let mut excluded = self.excluded.clone();
        excluded[q].push(codon);
        DesignRegion {
            pinned: self.pinned.clone(),
            excluded,
        }
    }

    /// The regions tiling everything in `self` except `solution`, branching on
    /// the free positions in `order`.
    ///
    /// `order` must list every free position of `self` exactly once. Any such
    /// ordering tiles correctly: a sequence in `self` other than `solution`
    /// differs from it somewhere, and the *first* differing position in `order`
    /// names the one part it lies in. Every part keeps all of `self`'s
    /// exclusions, so no part reaches outside its parent.
    fn partition(&self, solution: &[[u8; 3]], order: &[usize]) -> Vec<DesignRegion> {
        debug_assert_eq!(
            order.len(),
            self.free_positions().len(),
            "the branch order must cover every free position exactly once"
        );
        let mut pinned = self.pinned.clone();
        let mut parts = Vec::with_capacity(order.len());
        for &q in order {
            let mut excluded = self.excluded.clone();
            excluded[q].push(solution[q]);
            parts.push(DesignRegion {
                pinned: pinned.clone(),
                excluded,
            });
            pinned[q] = Some(solution[q]);
        }
        parts
    }
}

/// What a heap entry knows about its region's bound.
#[derive(Debug)]
enum Bound {
    /// The parent's bound, valid but not yet tightened. The region has not been
    /// solved, so it has cost nothing.
    Inherited,
    /// The region's own constrained optimum, and hence its exact bound — with
    /// the tables it was read from, when the walk keys children on marginals.
    Refined(Box<Design>, Option<Box<Kept>>),
}

/// A solved region's lattice, tables and constraint, kept until it branches so its
/// children's keys can be read off one outside pass over them.
struct Kept {
    dfa: crate::dfa::Dfa,
    tables: crate::tables::Tables,
    constraint: crate::region::PairConstraint,
    /// What these tables were charged against the budget: their estimated
    /// size while they sit on the heap, zero for tables rebuilt at branching.
    charged: usize,
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept").finish_non_exhaustive()
    }
}

/// A solved region: its bound, its optimum and, when kept, its tables.
type Solved = (f64, Design, Option<Box<Kept>>);

/// A region queued for examination, ordered so the heap yields the least bound
/// first.
#[derive(Debug)]
struct Entry {
    key: f64,
    /// Insertion order, breaking key ties deterministically so a run is
    /// reproducible.
    seq: usize,
    region: DesignRegion,
    bound: Bound,
    /// Where this region's row sits in [`BoundProbeReport::rows`], when the
    /// walk is being probed. A region keeps one row across both of its pops.
    probe: Option<usize>,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.seq == other.seq
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    /// Reversed: [`BinaryHeap`] is a max-heap and the search wants the minimum.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .key
            .total_cmp(&self.key)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// The design problem a bounded search runs over: the inputs of a constrained
/// design, plus the budget that bounds the walk.
///
/// Both flanks are given sequences. A designed flank has no codon lattice to
/// partition, so it is not a region of *this* space — the degeneracy a designed
/// flank shows is a different problem with a different answer.
#[derive(Clone, Copy)]
pub struct SearchInputs<'a> {
    pub params: &'a Params,
    pub protein: &'a str,
    /// The protein as amino-acid indices — the same residues as `protein`.
    pub aa: &'a [usize],
    pub usage: &'a CodonUsage,
    pub lambda: f64,
    pub utr5: &'a str,
    pub utr3: &'a str,
    pub constraints: RegionConstraints,
    pub config: ScanConfig,
    /// Wall-clock budget. Exhausting it returns the incumbent with an explicit
    /// unproven verdict rather than no answer.
    pub budget: Duration,
    /// Which order a partition branches in. Both are exact; the choice is what
    /// the search reaches first, not what it can reach.
    pub branch_order: BranchOrder,
    /// Record the incumbent's walk into [`SearchReport::trajectory`], so any
    /// budget below [`Self::budget`] can be read off this one run instead of
    /// costing a run of its own.
    ///
    /// Off by default. It has to be optional rather than always-on because the
    /// property the read-off rests on — that a logged run and an unlogged one
    /// return the same verdict — is only testable if both can be asked for.
    pub trajectory: bool,
    /// A modular design to start the incumbent at, as a whole-mRNA sequence
    /// over this same protein and these same flanks.
    ///
    /// Until the first modular design is found the incumbent is `+∞` and **no
    /// region can be discarded**, however tight its bound — so a walk that
    /// never finds one prunes nothing and degenerates to a region-granular
    /// filter. A seed removes that phase: the bound can fire from the first
    /// pop, and a seed already at or below the root's bound proves itself
    /// optimal in one solve. Where the seed comes from is the caller's
    /// business; a cut loop's rescue is the obvious source, since it returns a
    /// design that is modular or nothing.
    ///
    /// Nothing about the guarantee changes. The bound's lemma asks only that
    /// every incumbent be modular, so a seed is accepted only after it is
    /// solved over this space and verified like any other candidate — a
    /// sequence that is not a design of this space, or that folds in_violation,
    /// is a [`SearchError::Seed`] rather than a silently ignored argument.
    /// Passing an *optimal* seed changes what the walk certifies not at all:
    /// it still has to exhaust or reach the stop rule to say `Proved`.
    ///
    /// [`cegar_search`](crate::cegar_search) ignores this: the cut loop has no
    /// incumbent to improve on, only a lattice to shrink.
    pub seed: Option<&'a str>,
    /// How many regions the walk examines side by side. `1` (and `0`) is the
    /// serial walk, byte for byte.
    ///
    /// The regions on the heap are independent problems — each is one restricted
    /// constrained solve over the whole mRNA, and the walk's own bookkeeping is
    /// nothing beside it — so the width buys close to itself wherever the heap
    /// holds that many entries worth examining. It is also the *only* lever on
    /// the wall-clock of a **proof**: a best-first walk must open every region
    /// whose bound falls below the optimum, a set fixed by the bound and the
    /// answer, so no incumbent and no ordering can shrink it — only a tighter
    /// bound, or cheaper regions, can.
    ///
    /// The answer does not depend on the width. Examining is purely functional
    /// and every decision the results feed — the incumbent, the prune, the
    /// partition — is made afterwards in the order the entries were popped, so a
    /// wide walk passes through exactly the incumbents a narrow one does. What
    /// *does* change is the counters: a width of `k` may solve regions a serial
    /// walk would have stopped before reaching, which land in
    /// [`SearchStats::regions_discarded_after_refine`]. That is why this is
    /// opt-in rather than a default.
    ///
    /// Each solve stays single-threaded whatever this is set to
    /// ([`crate::design_restricted_with_region`] forces it), so the two kinds of
    /// concurrency do not contend: this one is for many small solves, the scan's
    /// own `threads` for one large one.
    pub search_threads: usize,
    /// Record what each batch of the width cost, into
    /// [`SearchReport::batch_probe`].
    ///
    /// Off by default, and free of anything the walk decides: it times the
    /// examining that already happens and adds one row per batch. Meaningless
    /// below a width of two, where every batch holds one entry.
    pub batch_probe: bool,
    /// Price a cheap region bound against the exact ones this walk pays for,
    /// into [`SearchReport::bound_probe`].
    ///
    /// Off by default, and it changes nothing the walk decides — no region is
    /// rejected on a cheap bound, and the counters, the answer and the verdict
    /// come back identical either way. What it costs is one pure-energy solve
    /// per branching, which is why it is opt-in rather than always on.
    pub bound_probe: bool,
    /// Key each child on its parent's codon max-marginal rather than on the
    /// parent's own bound.
    ///
    /// A child excludes its parent's codon at one position, so it is a subset
    /// of the parent with that one exclusion, and that set's optimum — read off
    /// one outside pass over the parent's tables for all of its children at
    /// once — is a floor under every design the child holds. A best-first walk
    /// folds a region because its key sorts ahead of the answer, so a child
    /// whose floor already reaches the answer is never folded at all.
    ///
    /// The answer and verdict do not change: every key stays a lower bound on
    /// its region. What changes is which regions are folded, and what the
    /// tables cost — see [`MarginalKey`]. `None` is the walk keyed on
    /// inherited bounds alone.
    pub marginal_key: Option<MarginalKey>,
}

/// How a walk keyed on marginals pays for its parents' tables.
///
/// A region's outside pass reads the tables its own solve filled, so a solved
/// region either keeps them until it branches or has them rebuilt then, at the
/// price of one more fold. Kept tables are held for as long as the region sits
/// on the heap, and a walk that finds no incumbent accumulates them without
/// limit. So tables are kept while their estimated total stays within
/// [`Self::kept_bytes`], and rebuilt at branching once it does not.
///
/// The keys are the same either way — a rebuilt table is the same table — so
/// the budget moves the cost of a walk and nothing it decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarginalKey {
    /// The most table memory, in estimated bytes, kept alive at once. `0`
    /// keeps nothing and rebuilds every parent at branching.
    pub kept_bytes: usize,
}

impl Default for MarginalKey {
    /// Two gibibytes: several hundred sets of tables at the lengths measured,
    /// and a small share of the hosts this runs on.
    fn default() -> Self {
        Self {
            kept_bytes: 2 << 30,
        }
    }
}

/// How a partition orders the positions it branches on.
///
/// The tiling is a partition for *any* ordering of a region's free positions,
/// so every setting returns the same optimum. It does **not** follow that the
/// walk is the same walk: part `k` of a region restricts the first `k + 1`
/// positions of this order, so a different order builds a different tree, with
/// different regions carrying different bounds. The seed's inertness argument
/// — that the regions a best-first walk must open are fixed by the bound and
/// the answer — is about *when the incumbent arrived* and does not transfer
/// here. Whether an order moves the node set is a measurement, and each of
/// these was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BranchOrder {
    /// Positions spelling the pairs the free fold insisted on come first — the
    /// one feedback available before an incumbent exists.
    #[default]
    ViolationGuided,
    /// Ascending position order, ignoring the verdict. This exists to *price*
    /// the guided order rather than to be used: a refinement asserted to help
    /// and never measured against its absence is a claim, not a result.
    Positional,
    /// Descending position order.
    ///
    /// The mirror of [`Self::Positional`], and not a symmetric one. A part
    /// pins a prefix of the order, so ascending leaves every region a free set
    /// that is a *suffix* and descending leaves one that is a *prefix* — which
    /// is to say the two spend the design space from opposite ends of the
    /// coding sequence, and the restricted runs they hand a fold drift in
    /// opposite directions as the tree deepens.
    Descending,
    /// Positions nearest an end of the **molecule** first: the key is a
    /// position's distance to whichever end of the mRNA is closer, ascending,
    /// with ties on ascending position.
    ///
    /// A CKY cell can be inherited from a less-restricted fold when its span
    /// misses every restricted codon, and the spans that miss a position are
    /// most numerous when that position sits near an end of the molecule — so
    /// the order a fold would prefer is this one. The end that matters is the
    /// molecule's and not the coding sequence's, because the UTRs are part of
    /// the folded span; with a short 5′ UTR and a long 3′ one the two are far
    /// apart.
    MoleculeEnds,
}

/// How a search ended — and therefore what its answer certifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOutcome {
    /// The stop rule fired or the heap emptied. The incumbent, if any, is the
    /// **exact optimum over modular designs**; no incumbent means the
    /// modular set is empty over the whole design space.
    Proved,
    /// The budget ran out holding a modular incumbent that is not proven
    /// optimal — something better may remain unexamined.
    BudgetModular,
    /// The budget ran out before any modular design was found. The best design
    /// is a not-modular best effort, and carries no optimality claim of any kind.
    BudgetNotModular,
}

impl SearchOutcome {
    /// Whether the answer is proven optimal over modular designs.
    pub fn is_proved(self) -> bool {
        self == SearchOutcome::Proved
    }
}

/// What the walk cost, in the units that decide whether it is affordable.
///
/// The two headline quantities are [`Self::solves_per_region_popped`] — the
/// price lazy refinement pays per region examined, against one solve per region
/// pushed without it — and [`Self::regions_pruned_unsolved`], the regions the
/// bound rejected before they cost anything at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SearchStats {
    /// Regions entered into the heap over the whole walk, the root included.
    pub regions_pushed: usize,
    /// Heap pops. A region is popped once carrying an inherited bound and again
    /// once refined, so this exceeds the number of distinct regions examined.
    pub regions_popped: usize,
    /// Restricted constrained solves — the cost that matters.
    pub solves: usize,
    /// Modularity verdicts issued, each two folds of a single sequence.
    pub verifies: usize,
    /// Regions the **bound** rejected without ever solving them: those still
    /// queued when the stop rule fired, every one of which carries a key at or
    /// above the incumbent.
    ///
    /// A budget exit contributes nothing here. Regions left queued when time
    /// runs out are unexamined, not proven irrelevant, and counting them would
    /// turn a walk that simply stopped into a walk that pruned.
    pub regions_pruned_unsolved: usize,
    /// Whether the walk started from a caller's seed. A seeded walk examines
    /// one region — the singleton the seed spells — before the heap is touched,
    /// and that region contributes one to [`Self::regions_pushed`],
    /// [`Self::regions_popped`], [`Self::solves`] and [`Self::verifies`], so
    /// this is what separates the seed's cost from the walk's own.
    pub seeded: bool,
    /// Regions whose codon lattice turned out empty — some position's
    /// synonymous set, after exclusions, admits nothing — so [`Walk::solve`]
    /// returned `None` without any DP work: the DFA build rejects the region
    /// before a scan is ever run. Counted apart from [`Self::solves`], which
    /// [`Walk::refine`] increments unconditionally, and from
    /// [`Self::regions_discarded_after_refine`], whose regions *did* solve.
    pub regions_empty: usize,
    /// Regions whose own refined bound reached the incumbent, so they were
    /// discarded without being verified or branched. These *did* cost their one
    /// solve — the tightening is what revealed them as irrelevant — so they are
    /// counted apart from [`Self::regions_pruned_unsolved`].
    pub regions_discarded_after_refine: usize,
    /// Outside passes run, one per branching of a walk keyed on marginals.
    pub marginal_passes: usize,
    /// Children whose marginal key rose above the bound they would otherwise
    /// have inherited.
    pub children_lifted: usize,
    /// Parents whose tables were rebuilt at branching because keeping them
    /// would have exceeded [`MarginalKey::kept_bytes`]. Each is one more fold,
    /// counted apart from [`Self::solves`].
    pub tables_rebuilt: usize,
}

impl SearchStats {
    /// Solves spent per region popped. Without lazy refinement every pushed
    /// region is solved before it enters the heap, so the comparable figure
    /// there is one solve per region.
    pub fn solves_per_region_popped(&self) -> f64 {
        if self.regions_popped == 0 {
            return 0.0;
        }
        self.solves as f64 / self.regions_popped as f64
    }

    /// The share of pushed regions that never cost a solve.
    pub fn pruned_fraction(&self) -> f64 {
        if self.regions_pushed == 0 {
            return 0.0;
        }
        self.regions_pruned_unsolved as f64 / self.regions_pushed as f64
    }

    /// The share of solves that hit an empty codon lattice rather than a real
    /// design — a counter tick, not DP work.
    pub fn empty_fraction(&self) -> f64 {
        if self.solves == 0 {
            return 0.0;
        }
        self.regions_empty as f64 / self.solves as f64
    }
}

/// What put one instant on the incumbent's walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryEvent {
    /// The root solve returned — the first instant at which the walk holds
    /// anything at all, and what a read-off before any improvement reports.
    RootSolved,
    /// The incumbent strictly improved.
    Incumbent,
}

/// One instant on the incumbent's walk.
///
/// A trajectory recorded at budget `B` answers *what would a run given budget
/// `T` have returned* for every `T ≤ B`, because the walk is budget-independent:
/// nothing in this module randomizes, the budget is consulted at one place and
/// only to return, and the incumbent has a single strictly-improving update
/// site. A run at `B` therefore passes through exactly the states a run at any
/// `T < B` would have ended in, and separate runs per budget re-walk an
/// identical prefix.
///
/// [`SearchReport::at_budget`] is that reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryRecord {
    pub event: TrajectoryEvent,
    /// Seconds from the walk's start to this instant.
    pub elapsed_s: f64,
    /// The objective the walk was holding, on the region bounds' scale. For
    /// [`TrajectoryEvent::RootSolved`] this is the root's own key — the
    /// constrained optimum over the whole space, which is the best effort a
    /// walk holding no modular design reports.
    pub objective: f64,
    /// Whether the design behind [`Self::objective`] verifies free-fold modular.
    /// True at every [`TrajectoryEvent::Incumbent`] instant, since only an
    /// modular design ever becomes the incumbent, and false at the root.
    pub modular: bool,
    /// Heap pops so far, so the walk's progress can be read in work as well as
    /// in seconds.
    pub regions_popped: usize,
}

/// What a run given a budget below the one actually spent would have returned.
///
/// The distinction the variants draw is the one a column of read-offs has to
/// preserve: a budget that was never honoured is not a budget the search failed
/// under.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetReadoff {
    /// The budget falls below the instant the root solve completed. [`Walk::run`]
    /// solves the root before entering the loop and never consults the budget
    /// during it, so a run given this budget would still have paid for the whole
    /// root solve and returned whatever it returned — the budget was never
    /// honoured. That is a fact about the root solve's cost, not about the
    /// search failing in the time given.
    NotHonoured,
    /// What a run at this budget would have held, and what it would have
    /// certified.
    Held {
        outcome: SearchOutcome,
        /// The incumbent's objective, on the same terms as
        /// [`SearchReport::best_objective`]: `None` when nothing modular was
        /// held, so a read-off and a terminal report answer alike.
        objective: Option<f64>,
        modular: bool,
        regions_popped: usize,
    },
}

/// The result of a bounded search: the best design it found, what that design
/// is certified to be, and what the walk cost.
#[derive(Debug, Clone)]
pub struct SearchReport {
    /// The incumbent — the best modular design found — or, when none was found,
    /// the constrained optimum over the whole space as a not-modular best effort
    /// ([`Self::modular`] says which). `None` only when the design space admits
    /// no encoding at all.
    pub best: Option<Design>,
    /// The incumbent's objective, on the same scale as the region bounds.
    pub best_objective: Option<f64>,
    /// Whether [`Self::best`] verifies free-fold modular.
    pub modular: bool,
    pub outcome: SearchOutcome,
    pub stats: SearchStats,
    pub elapsed_s: f64,
    /// The incumbent's walk, in the order it happened, when the run asked for
    /// it via [`SearchInputs::trajectory`]. Empty otherwise, and empty in any
    /// case for a design space that admits no encoding at all.
    pub trajectory: Vec<TrajectoryRecord>,
    /// The cheap bounds a probed walk composed, beside the exact ones it paid
    /// for. `None` unless [`SearchInputs::bound_probe`] asked for them.
    pub bound_probe: Option<BoundProbeReport>,
    /// What each batch of the width cost. `None` unless
    /// [`SearchInputs::batch_probe`] asked for it.
    pub batch_probe: Option<BatchProbeReport>,
    /// Seconds spent on outside passes, when the walk keys on marginals.
    pub marginal_s: f64,
}

impl SearchReport {
    /// What a run given `budget_s` would have returned, read off this walk.
    ///
    /// Equivalent to re-running the search at that budget — see
    /// [`TrajectoryRecord`] for why — and requires the report to carry a
    /// trajectory for any budget below [`Self::elapsed_s`].
    ///
    /// [`SearchOutcome::Proved`] is reachable only at or above
    /// [`Self::elapsed_s`]. It is a statement about the whole space, known only
    /// when the stop rule fires or the heap empties, so no instant mid-walk can
    /// report it however good the incumbent it holds.
    pub fn at_budget(&self, budget_s: f64) -> BudgetReadoff {
        if budget_s >= self.elapsed_s {
            return BudgetReadoff::Held {
                outcome: self.outcome,
                objective: self.best_objective,
                modular: self.modular,
                regions_popped: self.stats.regions_popped,
            };
        }
        // Instants are recorded from one monotone clock in the order they
        // occur, so the prefix at or before the budget is a `take_while`.
        let held = self
            .trajectory
            .iter()
            .take_while(|r| r.elapsed_s <= budget_s)
            .last();
        match held {
            None => BudgetReadoff::NotHonoured,
            Some(r) => BudgetReadoff::Held {
                outcome: if r.modular {
                    SearchOutcome::BudgetModular
                } else {
                    SearchOutcome::BudgetNotModular
                },
                objective: r.modular.then_some(r.objective),
                modular: r.modular,
                regions_popped: r.regions_popped,
            },
        }
    }
}

/// Errors surfaced by [`bounded_search`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SearchError {
    #[error("solving a region failed: {0}")]
    Design(#[from] BeamError),
    #[error("verifying a region's optimum failed: {0}")]
    Verdict(#[from] FreeFoldError),
    #[error("the seed cannot start the incumbent: {0}")]
    Seed(#[from] SeedRejected),
}

/// Why a [`SearchInputs::seed`] cannot be an incumbent.
///
/// Refusing is the point. The bound is a lower bound only over modular members
/// of a region, and it is the restriction of incumbents to modular designs that
/// licenses it — so an incumbent taken on trust is the one way to make the
/// search return a wrong answer while still reporting `Proved`. Each variant
/// below is a precondition of that lemma, checked rather than assumed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SeedRejected {
    /// The seed is not a whole-mRNA sequence of this instance: the design space
    /// spells `want` nucleotides — both flanks plus three per residue — and the
    /// seed has `got`.
    #[error("the seed is {got} nt; this design space spells {want}")]
    Length { got: usize, want: usize },
    /// The seed holds a byte that is not `A`, `C`, `G`, `U` or `T`.
    #[error("the seed holds byte {0:#04x}, which is not a nucleotide")]
    Nucleotide(u8),
    /// The seed's codon at this protein position does not translate to the
    /// residue the protein has there, so the seed is not an encoding of this
    /// protein at all.
    #[error("the seed's codon at protein position {at} is not synonymous for the residue there")]
    Codon { at: usize },
    /// The seed's CDS is an encoding of this protein, but the sequence around it
    /// is not this instance's flanks — the seed belongs to a different problem
    /// and its energy is not comparable to this one's bounds.
    #[error("the seed's flanks are not this instance's")]
    Flank,
    /// The seed folds below its own constrained optimum, so it is exactly the
    /// kind of design the search exists to avoid returning: taking it as an
    /// incumbent would compare a free-fold objective against bounds computed on
    /// the constrained one, and could prune the true optimum away.
    #[error("the seed is not free-fold modular: it folds {gap:.2} kcal/mol below its constrained optimum")]
    Violating { gap: f64 },
}

/// What examining one region produced, before any of it is acted on.
///
/// The split is by what the entry carried, and both variants hold the expensive
/// result and nothing else: a region with an inherited bound needs a solve, a
/// region already solved needs its optimum verified. Keeping the result separate
/// from the decision it feeds is what lets a batch of them be computed side by
/// side while the decisions stay in pop order.
enum Examined {
    /// A restricted constrained solve: the region's own bound and optimum, or
    /// `None` where some position's codon set is empty.
    Solved(Option<Solved>),
    /// The modularity verdict on a solved region's optimum — and, for a walk
    /// keyed on marginals, the outside pass over its tables, which is as pure
    /// as the verdict and so runs beside it.
    Verified(FreeFoldGap, Option<Keyed>),
}

/// A verified region's codon marginals, computed where it was examined.
struct Keyed {
    marginals: crate::outside_scan::Marginals,
    /// Tables rebuilt because the region had none kept, carrying the lattice
    /// the marginals are read against.
    rebuilt: Option<Box<Kept>>,
    secs: f64,
}

/// The outside pass over a verified region's tables — its kept ones, or ones
/// rebuilt by solving it again.
fn keyed(inputs: &SearchInputs<'_>, entry: &Entry) -> Result<Option<Keyed>, SearchError> {
    if inputs.marginal_key.is_none() {
        return Ok(None);
    }
    let started = Instant::now();
    let Bound::Refined(_, kept) = &entry.bound else {
        return Ok(None);
    };
    let rebuilt = match kept {
        Some(_) => None,
        None => {
            let Some((_, _, Some(kept))) = solve_region_kept(inputs, &entry.region)? else {
                unreachable!("a region that solved once solves again, keeping its tables")
            };
            Some(kept)
        }
    };
    let tables = kept
        .as_deref()
        .or(rebuilt.as_deref())
        .expect("kept or rebuilt");
    let marginals = crate::outside_scan::OutsideScan::new(
        inputs.params,
        &tables.dfa,
        &tables.tables,
        &tables.constraint,
        inputs.config.single_max_len,
    )
    .run();
    Ok(Some(Keyed {
        marginals,
        rebuilt,
        secs: started.elapsed().as_secs_f64(),
    }))
}

/// Examine one entry — the whole of what a worker does.
fn examine_one(
    inputs: &SearchInputs<'_>,
    cds: CdsWindow,
    entry: &Entry,
) -> Result<Examined, SearchError> {
    match &entry.bound {
        Bound::Inherited => Ok(Examined::Solved(solve_region_kept(inputs, &entry.region)?)),
        Bound::Refined(design, _) => {
            let gap = sequence_gap(inputs.params, &design.sequence, cds, inputs.constraints)?;
            // A modular region becomes the incumbent at its own key, and every
            // child's key is at least that, so the stop rule discards the
            // children before any of their keys is read: the pass would be
            // spent on nothing.
            let keyed = if gap.is_modular() {
                None
            } else {
                keyed(inputs, entry)?
            };
            Ok(Examined::Verified(gap, keyed))
        }
    }
}

/// Examine a batch across `workers` scoped threads, returning the results in
/// batch order.
///
/// The partition is **static**: worker `i` takes the `i`-th contiguous slice, and
/// the slices are concatenated back in index order. So the result vector is the
/// same sequence of values for any worker count, and [`Walk::apply`] consumes it
/// in the order the entries were popped — the width cannot reach the answer,
/// by construction rather than by observation.
/// Examine one entry, and say how long it took — the quantity a batch's
/// critical path is made of.
fn examine_timed(
    inputs: &SearchInputs<'_>,
    cds: CdsWindow,
    entry: &Entry,
) -> (Result<Examined, SearchError>, f64) {
    let started = Instant::now();
    let examined = examine_one(inputs, cds, entry);
    (examined, started.elapsed().as_secs_f64())
}

fn examine_fanned(
    inputs: &SearchInputs<'_>,
    cds: CdsWindow,
    batch: &[Entry],
    workers: usize,
) -> Vec<(Result<Examined, SearchError>, f64)> {
    let chunk = batch.len().div_ceil(workers.max(1));
    std::thread::scope(|scope| {
        let handles: Vec<_> = batch
            .chunks(chunk)
            .map(|slice| {
                crate::task_clock::spawn(scope, move || {
                    slice
                        .iter()
                        .map(|entry| examine_timed(inputs, cds, entry))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("an examining worker panicked"))
            .collect()
    })
}

/// Solve one region under the active constraints: its exact bound and its
/// optimum, or `None` when some position's codon set is empty.
///
/// Free of the walk's state so it can run on a worker thread.
fn solve_region(
    inputs: &SearchInputs<'_>,
    region: &DesignRegion,
) -> Result<Option<(f64, Design)>, SearchError> {
    let allowed = region.allowed(inputs.aa);
    let solved = design_restricted_with_region(
        inputs.params,
        inputs.protein,
        inputs.usage,
        inputs.lambda,
        &allowed,
        inputs.utr5,
        inputs.utr3,
        inputs.constraints,
        inputs.config,
    )?;
    Ok(solved.map(|design| {
        let key = objective(inputs.lambda, design.mfe_kcal_mol, design.cai_log);
        (key, design)
    }))
}

/// [`solve_region`], keeping the region's tables when the walk keys children
/// on marginals.
fn solve_region_kept(
    inputs: &SearchInputs<'_>,
    region: &DesignRegion,
) -> Result<Option<Solved>, SearchError> {
    if inputs.marginal_key.is_none() {
        return Ok(solve_region(inputs, region)?.map(|(key, design)| (key, design, None)));
    }
    let allowed = region.allowed(inputs.aa);
    let kept = crate::design_restricted_keeping(
        inputs.params,
        inputs.protein,
        inputs.usage,
        inputs.lambda,
        &allowed,
        inputs.utr5,
        inputs.utr3,
        inputs.constraints,
        inputs.config,
    )?;
    Ok(kept.map(|(design, dfa, tables, constraint)| {
        let key = objective(inputs.lambda, design.mfe_kcal_mol, design.cai_log);
        let kept = Kept {
            dfa,
            tables,
            constraint,
            charged: 0,
        };
        (key, design, Some(Box::new(kept)))
    }))
}

/// The codons a whole-mRNA sequence spells over the `m` residues starting at
/// `start`, as nucleotide-index triples — the alphabet the codon table and a
/// region's pins are written in.
fn cds_codons(sequence: &str, start: usize, m: usize) -> Vec<[u8; 3]> {
    let bytes = sequence.as_bytes();
    (0..m)
        .map(|i| {
            let at = |k: usize| {
                NUCLEOTIDES
                    .iter()
                    .position(|&n| n == bytes[start + 3 * i + k])
                    .expect("a whole-mRNA sequence spells A/C/G/U") as u8
            };
            [at(0), at(1), at(2)]
        })
        .collect()
}

/// The nucleotides in index order, which is the order the codon table's triples
/// are written in.
const NUCLEOTIDES: [u8; 4] = [b'A', b'C', b'G', b'U'];

/// A seed as the design DP spells sequences: ASCII `A`/`C`/`G`/`U`, with `T`
/// read as `U` so a DNA-spelled seed is not refused over an alphabet.
fn normalize(seed: &str) -> Result<String, SeedRejected> {
    seed.bytes()
        .map(|b| {
            let b = if b == b'T' { b'U' } else { b };
            NUCLEOTIDES
                .contains(&b)
                .then_some(b as char)
                .ok_or(SeedRejected::Nucleotide(b))
        })
        .collect()
}

/// The objective the search minimizes: `λ·MFE − (1−λ)·CAI_log`, on the design
/// DP's dekacalorie scale so a region bound and an incumbent are the same
/// quantity.
fn objective(lambda: f64, mfe_kcal_mol: f64, cai_log: f64) -> f64 {
    lambda * (mfe_kcal_mol * 100.0).round() - (1.0 - lambda) * (cai_log * 100.0).round()
}

/// Search for the optimal modular design, bounded by `inputs.budget`.
///
/// Returns the incumbent with an outcome saying what it certifies: proven
/// optimal over modular designs, modular but unproven, or a not-modular best
/// effort. See the module docs for why the walk is exact.
pub fn bounded_search(inputs: &SearchInputs<'_>) -> Result<SearchReport, SearchError> {
    let mut walk = Walk::new(inputs);
    walk.run()?;
    Ok(walk.finish())
}

/// The mutable state of one walk — the heap, the incumbent, and the counters.
struct Walk<'a> {
    inputs: &'a SearchInputs<'a>,
    cds: CdsWindow,
    heap: BinaryHeap<Entry>,
    incumbent: f64,
    best: Option<Design>,
    /// The root region's optimum, returned when no modular design was found. It
    /// is the best effort by construction — the root is the whole space, so
    /// nothing the walk reaches later has a lower constrained objective — which
    /// is why it is captured once and never revised.
    fallback: Option<Design>,
    modular: bool,
    stats: SearchStats,
    pushed_seq: usize,
    started: Instant,
    /// Seconds spent on outside passes.
    marginal_s: f64,
    /// Estimated bytes of tables kept alive on the heap right now.
    kept_bytes: usize,
    outcome: SearchOutcome,
    trajectory: Vec<TrajectoryRecord>,
    /// The rows a probed walk is filling, `None` when it is not being probed.
    probe: Option<BoundProbeReport>,
    /// What each batch cost, for a walk probing the width rather than the bound.
    batches: Option<BatchProbeReport>,
    /// A floor under the free energy of every encoding the whole space admits,
    /// so every region inherits it. Vacuous until a probed walk solves for it.
    space_floor: f64,
}

impl<'a> Walk<'a> {
    fn new(inputs: &'a SearchInputs<'a>) -> Self {
        Self {
            inputs,
            cds: CdsWindow::new(inputs.utr5.len(), inputs.aa.len()),
            heap: BinaryHeap::new(),
            incumbent: f64::INFINITY,
            best: None,
            fallback: None,
            modular: false,
            stats: SearchStats::default(),
            pushed_seq: 0,
            started: Instant::now(),
            marginal_s: 0.0,
            kept_bytes: 0,
            outcome: SearchOutcome::Proved,
            trajectory: Vec::new(),
            probe: inputs.bound_probe.then(BoundProbeReport::default),
            batches: inputs.batch_probe.then(BatchProbeReport::default),
            space_floor: f64::NEG_INFINITY,
        }
    }

    /// Record one instant of the walk, when the run asked for a trajectory.
    ///
    /// Every field is a value the caller already holds: the elapsed clock, the
    /// objective last compared, and a counter. Nothing here recomputes,
    /// re-folds or re-scores, which is what keeps a logged run's verdict
    /// identical to an unlogged one's — the property the whole read-off rests
    /// on.
    fn record(&mut self, event: TrajectoryEvent, objective: f64, modular: bool) {
        if !self.inputs.trajectory {
            return;
        }
        self.trajectory.push(TrajectoryRecord {
            event,
            elapsed_s: self.started.elapsed().as_secs_f64(),
            objective,
            modular,
            regions_popped: self.stats.regions_popped,
        });
    }

    /// Walk until the stop rule fires, the heap empties, or the budget runs out.
    fn run(&mut self) -> Result<(), SearchError> {
        let root = DesignRegion::root(self.inputs.aa.len());
        self.stats.regions_pushed += 1;
        // The root has no parent to inherit from, so it is solved up front;
        // every region after it enters lazily. Its optimum is also the best
        // effort by construction — it minimizes the constrained objective over
        // the whole space, so nothing the walk finds later can beat it on that
        // scale — which is what an exhausted budget with no modular design falls
        // back to.
        match self.solve(&root)? {
            Some((key, design, kept)) => {
                // Recorded because the budget is not consulted during this
                // solve, so a budget below this instant was never honoured and
                // has to be reported as such rather than as a failure.
                self.record(TrajectoryEvent::RootSolved, key, false);
                if self.probe.is_some() {
                    self.space_floor = self.energy_floor(&root)?.unwrap_or(f64::NEG_INFINITY);
                }
                self.fallback = Some(design.clone());
                let kept = self.charge(kept);
                self.push(key, root, Bound::Refined(Box::new(design), kept), None);
            }
            None => return Ok(()),
        }
        // A seed is adopted *after* the root is solved, not before. The root
        // solve runs either way and the budget is never consulted during it, so
        // a seed taken first would put a modular instant on the trajectory at an
        // elapsed time no run could have been given — and a read-off below the
        // root's cost has to stay `NotHonoured`. The stop rule is consulted at
        // the first peek below, so an incumbent adopted here is in place before
        // any region of the heap is examined.
        if let Some(seed) = self.inputs.seed {
            self.adopt_seed(seed)?;
        }
        // One batch at a time. At width 1 — the default — `take_batch` peeks,
        // applies the stop rule, checks the budget and pops exactly one entry,
        // which is the serial walk verbatim; wider, it takes several and
        // examines them side by side. Either way every *decision* is made below
        // in pop order, which is what keeps the answer independent of the width.
        loop {
            let batch = self.take_batch();
            if batch.is_empty() {
                // The stop rule fired, the heap emptied, or the budget ran out —
                // `take_batch` has recorded which. Whatever ended the walk left
                // the heap holding exactly the regions never examined, which is
                // what lets the stop rule's leftovers be counted as pruned and a
                // budget exit's not.
                return Ok(());
            }
            for (entry, examined, secs) in self.examine(batch)? {
                self.apply(entry, examined, secs)?;
            }
        }
    }

    /// Take the next entries to examine: as many as the width allows, each one
    /// past the stop rule, in heap order.
    ///
    /// The three ways a walk ends are all "no entry to take", and each leaves
    /// the heap in the state its own accounting needs. The budget is consulted
    /// once per batch rather than once per entry, because a batch's solves run
    /// side by side and overrun it by one solve either way; at width 1 that is
    /// the per-entry check.
    fn take_batch(&mut self) -> Vec<Entry> {
        let width = self.inputs.search_threads.max(1);
        let mut batch = Vec::with_capacity(width);
        while batch.len() < width {
            let Some(top) = self.heap.peek() else {
                break;
            };
            if top.key >= self.incumbent {
                // Min-ordered: every region still queued is at least this bad.
                break;
            }
            if batch.is_empty() && self.over_budget() {
                self.outcome = if self.modular {
                    SearchOutcome::BudgetModular
                } else {
                    SearchOutcome::BudgetNotModular
                };
                break;
            }
            batch.push(self.heap.pop().expect("just peeked"));
            self.stats.regions_popped += 1;
        }
        batch
    }

    /// Do the expensive half of examining a batch — one restricted solve per
    /// region carrying an inherited bound, one modularity verdict per region
    /// already solved — and hand the results back paired with their entries, in
    /// pop order.
    ///
    /// Nothing here reads or writes the incumbent, the heap or the trajectory.
    /// That is what lets the batch cross threads, and it is also why the width
    /// cannot change the answer: [`Walk::apply`] makes every decision
    /// afterwards, against the same incumbent it would have held serially,
    /// because the entries reach it in the order they were popped.
    fn examine(&mut self, batch: Vec<Entry>) -> Result<Vec<(Entry, Examined, f64)>, SearchError> {
        for entry in &batch {
            match entry.bound {
                Bound::Inherited => self.stats.solves += 1,
                Bound::Refined(..) => self.stats.verifies += 1,
            }
        }
        let width = self.inputs.search_threads.max(1);
        let started = Instant::now();
        let results = if width <= 1 || batch.len() <= 1 {
            batch
                .iter()
                .map(|e| examine_timed(self.inputs, self.cds, e))
                .collect()
        } else {
            examine_fanned(self.inputs, self.cds, &batch, width)
        };
        self.record_batch(
            batch.len(),
            width,
            &results,
            started.elapsed().as_secs_f64(),
        );
        batch
            .into_iter()
            .zip(results)
            .map(|(entry, (examined, secs))| examined.map(|e| (entry, e, secs)))
            .collect()
    }

    /// Act on one examined entry: tighten its bound, or verify and branch it.
    /// `secs` is what examining it cost, which only a probed walk reads.
    fn apply(&mut self, entry: Entry, examined: Examined, secs: f64) -> Result<(), SearchError> {
        match examined {
            Examined::Solved(solved) => {
                if solved.is_some() {
                    self.price_relaxed(&entry.region, entry.probe, secs)?;
                }
                self.refine(entry.region, entry.probe, solved);
                Ok(())
            }
            Examined::Verified(gap, keyed) => {
                let Bound::Refined(design, kept) = entry.bound else {
                    unreachable!("a verdict is only ever computed for a solved region")
                };
                self.expand(entry.key, entry.region, *design, kept, keyed, gap)
            }
        }
    }

    /// Take the caller's seed as the starting incumbent, or say why it cannot
    /// be one.
    ///
    /// The seed is solved as the singleton region it spells and verified
    /// exactly as any region's optimum is, so it joins the walk by the same
    /// path, on the same scale, under the same modularity test as every other
    /// incumbent — nothing about it is taken on the caller's word. Solving is
    /// also what settles admissibility without a second comparison: a singleton
    /// region's design is built from *this* instance's flanks and codon table,
    /// so a seed belonging to some other problem is exactly the seed whose
    /// solved sequence comes back different from the one handed in.
    fn adopt_seed(&mut self, seed: &str) -> Result<(), SearchError> {
        let sequence = normalize(seed)?;
        let codons = self.seed_codons(&sequence)?;
        self.stats.seeded = true;
        self.stats.regions_pushed += 1;
        self.stats.regions_popped += 1;
        let Some((_, design, _)) = self.solve(&DesignRegion::singleton(&codons))? else {
            // Unreachable: `seed_codons` has already established that every
            // position's codon is one this protein admits, so no set is empty.
            return Err(SeedRejected::Codon { at: 0 }.into());
        };
        if design.sequence != sequence {
            return Err(SeedRejected::Flank.into());
        }
        self.stats.verifies += 1;
        let gap = sequence_gap(
            self.inputs.params,
            &design.sequence,
            self.cds,
            self.inputs.constraints,
        )?;
        if !gap.is_modular() {
            return Err(SeedRejected::Violating {
                gap: gap.gap_kcal_mol,
            }
            .into());
        }
        let obj = objective(self.inputs.lambda, gap.free_mfe_kcal_mol, design.cai_log);
        self.incumbent = obj;
        self.modular = true;
        self.best = Some(design);
        self.record(TrajectoryEvent::Incumbent, obj, true);
        Ok(())
    }

    /// The seed's CDS codons, checked to be an encoding of *this* protein.
    ///
    /// The length check is over the whole mRNA rather than the CDS alone,
    /// because a seed is a finished molecule: it is the flanks plus three
    /// nucleotides per residue, and anything else is a sequence from a
    /// different instance.
    fn seed_codons(&self, sequence: &str) -> Result<Vec<[u8; 3]>, SeedRejected> {
        let m = self.inputs.aa.len();
        let want = self.cds.start + 3 * m + self.inputs.utr3.len();
        if sequence.len() != want {
            return Err(SeedRejected::Length {
                got: sequence.len(),
                want,
            });
        }
        let codons = cds_codons(sequence, self.cds.start, m);
        for (at, codon) in codons.iter().enumerate() {
            let synonymous = AMINO_ACIDS[self.inputs.aa[at]]
                .codons
                .iter()
                .any(|c| c.nuc == *codon);
            if !synonymous {
                return Err(SeedRejected::Codon { at });
            }
        }
        Ok(codons)
    }

    /// Tighten an inherited bound: back onto the heap under the region's own
    /// bound — or discarded unopened when that bound already reaches the
    /// incumbent.
    ///
    /// The solve itself happened in [`Walk::examine`]; what is left is the
    /// comparison, and it is made here so that it is made against the incumbent
    /// a serial walk would have been holding.
    fn refine(&mut self, region: DesignRegion, probe: Option<usize>, solved: Option<Solved>) {
        let Some((key, design, kept)) = solved else {
            // An empty codon set: the region holds no sequence.
            self.stats.regions_empty += 1;
            self.close_probe(probe, Probed::Empty);
            return;
        };
        self.close_probe(
            probe,
            Probed::Solved {
                exact: key,
                mfe: design.mfe_kcal_mol,
                cai: design.cai_log,
            },
        );
        if key >= self.incumbent {
            self.stats.regions_discarded_after_refine += 1;
            return;
        }
        let kept = self.charge(kept);
        self.push(key, region, Bound::Refined(Box::new(design), kept), probe);
    }

    /// Take a verified region's optimum as the incumbent if it is modular and
    /// better, and branch — violating pairs first.
    ///
    /// The verdict was computed in [`Walk::examine`]; the incumbent comparison
    /// and the partition happen here, in pop order.
    fn expand(
        &mut self,
        key: f64,
        region: DesignRegion,
        design: Design,
        kept: Option<Box<Kept>>,
        keyed: Option<Keyed>,
        gap: FreeFoldGap,
    ) -> Result<(), SearchError> {
        if gap.is_modular() {
            // A modular design's free fold *is* its constrained fold, so scoring
            // the incumbent on the free energy puts it on the bounds' scale and
            // stays correct at λ = 0, where the design's own returned structure
            // is arbitrary.
            let obj = objective(self.inputs.lambda, gap.free_mfe_kcal_mol, design.cai_log);
            if obj < self.incumbent {
                self.incumbent = obj;
                self.modular = true;
                self.best = Some(design.clone());
                self.record(TrajectoryEvent::Incumbent, obj, true);
            }
        }
        let codons = self.codons(&design);
        let order = self.branch_order(&region, &gap.reinstated_pairs);
        let floors = self.child_floors(&region, &design)?;
        let marginals = self.marginals(&region, &codons, &order)?;
        let keys = self.marginal_keys(key, kept, keyed, &codons, &order);
        for (part, child) in region.partition(&codons, &order).into_iter().enumerate() {
            self.stats.regions_pushed += 1;
            let probe = self.open_probe(&child, keys[part], floors, part, &order[..=part]);
            if let (Some(probe), Some(handle)) = (self.probe.as_mut(), probe) {
                probe.rows[handle].marginal = marginals[part];
            }
            self.push(keys[part], child, Bound::Inherited, probe);
        }
        Ok(())
    }

    /// The free positions of `region`, in the configured branch order.
    ///
    /// Every arm returns a permutation of exactly the free set: positions the
    /// region has already pinned are dropped, because they are not branchable,
    /// and none is added. That is what makes the partition a partition whatever
    /// the order says, and it is asserted in `partition` itself.
    fn branch_order(&self, region: &DesignRegion, violating: &[(usize, usize)]) -> Vec<usize> {
        let free = region.free_positions();
        match self.inputs.branch_order {
            BranchOrder::Positional => free,
            BranchOrder::Descending => free.into_iter().rev().collect(),
            BranchOrder::MoleculeEnds => self.nearest_molecule_end_first(free),
            BranchOrder::ViolationGuided => Self::violation_guided(free, violating, self.cds),
        }
    }

    /// `free` ordered by each position's distance to the nearer end of the
    /// molecule, ascending.
    ///
    /// The sort is stable, so positions equidistant from their nearer end keep
    /// ascending order between them and the walk stays reproducible.
    fn nearest_molecule_end_first(&self, mut free: Vec<usize>) -> Vec<usize> {
        let last = self.cds.end + self.inputs.utr3.len();
        free.sort_by_key(|&q| {
            let opens = self.cds.start + 3 * q;
            opens.min(last - (opens + 3))
        });
        free
    }

    /// `free` with the positions spelling `violating` promoted to the front.
    ///
    /// A violating pair's endpoints are whole-mRNA positions; only those inside
    /// the CDS name a codon, and a pair joining a UTR to the CDS contributes
    /// the one endpoint that does. The remaining free positions follow in
    /// ascending order, so the result is a permutation of exactly the free set
    /// whatever the pairs say.
    fn violation_guided(
        free: Vec<usize>,
        violating: &[(usize, usize)],
        cds: CdsWindow,
    ) -> Vec<usize> {
        let mut order: Vec<usize> = Vec::with_capacity(free.len());
        let endpoints = violating
            .iter()
            .flat_map(|&(i, j)| [i, j])
            .filter(|&p| p >= cds.start && p < cds.end)
            .map(|p| (p - cds.start) / 3);
        for position in endpoints {
            if free.contains(&position) && !order.contains(&position) {
                order.push(position);
            }
        }
        let rest: Vec<usize> = free
            .iter()
            .copied()
            .filter(|p| !order.contains(p))
            .collect();
        order.extend(rest);
        order
    }

    /// Solve one region on the walk's own thread, counting it — the root and a
    /// seed, which are examined before the loop begins.
    fn solve(&mut self, region: &DesignRegion) -> Result<Option<Solved>, SearchError> {
        self.stats.solves += 1;
        solve_region_kept(self.inputs, region)
    }

    /// Each child's key: its parent's bound, lifted to the parent's codon
    /// max-marginal at the position the child excludes a codon at, when the
    /// walk keys on marginals and the parent's tables were kept.
    ///
    /// The marginal is the least `λ·MFE − (1−λ)·100·log CAI` over the parent's
    /// designs avoiding that codon, and the child is a subset of them. A key
    /// rounds its adaptation term to whole dekacalories, which can lower a
    /// design's key by at most `(1−λ)/2` below that objective, so the marginal
    /// less that much — and a float's worth more — is a floor under every key
    /// in the child, modular or not.
    fn marginal_keys(
        &mut self,
        key: f64,
        kept: Option<Box<Kept>>,
        keyed: Option<Keyed>,
        codons: &[[u8; 3]],
        order: &[usize],
    ) -> Vec<f64> {
        if let Some(kept) = &kept {
            self.kept_bytes -= kept.charged;
        }
        let Some(keyed) = keyed else {
            return vec![key; order.len()];
        };
        if keyed.rebuilt.is_some() {
            self.stats.tables_rebuilt += 1;
        }
        self.stats.marginal_passes += 1;
        self.marginal_s += keyed.secs;
        let kept = kept.or(keyed.rebuilt).expect("kept or rebuilt");
        let marginals = keyed.marginals;
        let lambda = self.inputs.lambda;
        let report = -(1.0 - lambda) * 100.0 * kept.dfa.codon_offset();
        let slack = 0.5 * (1.0 - lambda) + 1e-6;
        order
            .iter()
            .map(|&q| {
                let floor = match marginals.excluding(&kept.dfa, q, codons[q]) {
                    Some(m) => m + report - slack,
                    // Nothing in the parent avoids that codon, so the child
                    // holds nothing: any key is a bound, and the largest sorts
                    // it past every answer.
                    None => f64::INFINITY,
                };
                if floor > key {
                    self.stats.children_lifted += 1;
                }
                key.max(floor)
            })
            .collect()
    }

    /// Keep `kept` on the heap if its tables fit what the budget has left,
    /// charging them against it; drop them otherwise, to be rebuilt at
    /// branching.
    fn charge(&mut self, kept: Option<Box<Kept>>) -> Option<Box<Kept>> {
        let mut kept = kept?;
        let budget = self.inputs.marginal_key.map_or(0, |m| m.kept_bytes);
        let bytes = kept.tables.approx_bytes();
        if self.kept_bytes + bytes > budget {
            return None;
        }
        self.kept_bytes += bytes;
        kept.charged = bytes;
        Some(kept)
    }

    fn push(&mut self, key: f64, region: DesignRegion, bound: Bound, probe: Option<usize>) {
        self.pushed_seq += 1;
        self.heap.push(Entry {
            key,
            seq: self.pushed_seq,
            region,
            bound,
            probe,
        });
    }

    /// The least constrained free energy any encoding of `region` folds to:
    /// the same solve the walk makes, at the objective's pure-energy end, where
    /// adaptation carries no weight and the optimum is an energy floor over the
    /// whole region. `None` for a region admitting no encoding.
    ///
    /// This is the probe's entire cost, and it is kept out of
    /// [`SearchStats::solves`] so a probed walk reports the same price as an
    /// unprobed one.
    fn energy_floor(&mut self, region: &DesignRegion) -> Result<Option<f64>, SearchError> {
        let started = Instant::now();
        let energy_only = SearchInputs {
            lambda: 1.0,
            ..*self.inputs
        };
        let floor = solve_region(&energy_only, region)?.map(|(_, design)| design.mfe_kcal_mol);
        if let Some(probe) = self.probe.as_mut() {
            probe.floor_solves += 1;
            probe.floor_s += started.elapsed().as_secs_f64();
        }
        Ok(floor)
    }

    /// The energies a probed walk offers a region's children as floors. One
    /// pure-energy solve per branching, amortised over every child.
    fn child_floors(
        &mut self,
        region: &DesignRegion,
        design: &Design,
    ) -> Result<Option<MfeFloors>, SearchError> {
        if self.probe.is_none() {
            return Ok(None);
        }
        let parent = self.energy_floor(region)?.unwrap_or(f64::NEG_INFINITY);
        Ok(Some(MfeFloors {
            space: self.space_floor,
            parent,
            parent_optimum: design.mfe_kcal_mol,
        }))
    }

    /// Record what was knowable about a region's bound before its solve.
    ///
    /// The adaptation ceiling is one pass over the positions; the floors were
    /// settled once for all of this region's siblings.
    fn open_probe(
        &mut self,
        region: &DesignRegion,
        inherited: f64,
        floors: Option<MfeFloors>,
        part: usize,
        restricted: &[usize],
    ) -> Option<usize> {
        let floors = floors?;
        let aa = self.inputs.aa;
        let cai_ceiling = dfa::cai_ceiling(aa, self.inputs.usage, &region.allowed(aa));
        let pinned = aa.len() - region.free_positions().len();
        Some(self.probe.as_mut()?.open(BoundProbeRow {
            pinned,
            part,
            restricted: restricted.to_vec(),
            inherited,
            cai_ceiling,
            floors,
            probed: Probed::Unsolved,
            closed_at: None,
            relaxed: None,
            marginal: None,
        }))
    }

    /// Record what one batch cost, for a walk probing the width.
    ///
    /// Every worker takes exactly one entry, so the batch's critical path is
    /// its slowest entry and the only interesting comparison is against a pool
    /// that could have placed the same work continuously.
    fn record_batch(
        &mut self,
        taken: usize,
        width: usize,
        results: &[(Result<Examined, SearchError>, f64)],
        wall_s: f64,
    ) {
        let Some(probe) = self.batches.as_mut() else {
            return;
        };
        probe.push(BatchRow {
            taken,
            width,
            slowest_s: results.iter().map(|(_, s)| *s).fold(0.0, f64::max),
            total_s: results.iter().map(|(_, s)| *s).sum(),
            wall_s,
        });
    }

    /// Solve `region` again with the region constraint dropped and record it beside
    /// the exact solve, for a probed walk. The time is added to the probe's
    /// own clock so a probed walk's net time stays comparable to an unprobed
    /// one's.
    fn price_relaxed(
        &mut self,
        region: &DesignRegion,
        handle: Option<usize>,
        exact_s: f64,
    ) -> Result<(), SearchError> {
        let (Some(_), Some(handle)) = (self.probe.as_ref(), handle) else {
            return Ok(());
        };
        let started = Instant::now();
        let constraint_free = SearchInputs {
            constraints: RegionConstraints::default(),
            ..*self.inputs
        };
        let solved = solve_region(&constraint_free, region)?;
        let relaxed_s = started.elapsed().as_secs_f64();
        let probe = self.probe.as_mut().expect("checked above");
        probe.floor_s += relaxed_s;
        if let Some((_, design)) = solved {
            probe.price_relaxed(
                handle,
                RelaxedPrice {
                    mfe: design.mfe_kcal_mol,
                    cai: design.cai_log,
                    relaxed_s,
                    exact_s,
                },
            );
        }
        Ok(())
    }

    /// For a probed walk, the parent's max-marginal at each position `order`
    /// branches on, solved by brute force: the parent with the solution's codon
    /// excluded there. Charged to the probe's clock. Empty for an unprobed walk.
    fn marginals(
        &mut self,
        region: &DesignRegion,
        codons: &[[u8; 3]],
        order: &[usize],
    ) -> Result<Vec<Option<(f64, f64)>>, SearchError> {
        if self.probe.is_none() {
            return Ok(vec![None; order.len()]);
        }
        let mut out = Vec::with_capacity(order.len());
        for &q in order {
            let started = Instant::now();
            let solved = solve_region(self.inputs, &region.excluding(q, codons[q]))?;
            let secs = started.elapsed().as_secs_f64();
            out.push(solved.map(|(key, _)| (key, secs)));
            if let Some(probe) = self.probe.as_mut() {
                probe.floor_s += secs;
            }
        }
        Ok(out)
    }

    /// Record what a region's solve turned out to be, for a probed walk.
    fn close_probe(&mut self, handle: Option<usize>, probed: Probed) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let popped = self.stats.regions_popped;
        if let (Some(probe), Some(handle)) = (self.probe.as_mut(), handle) {
            let at = ProbeInstant {
                regions_popped: popped,
                walk_s: elapsed - probe.floor_s,
            };
            probe.close(handle, probed, at);
        }
    }

    fn over_budget(&self) -> bool {
        self.started.elapsed() >= self.inputs.budget
    }

    /// The CDS codons a design spells.
    fn codons(&self, design: &Design) -> Vec<[u8; 3]> {
        cds_codons(&design.sequence, self.cds.start, self.inputs.aa.len())
    }

    /// Close the walk: settle what the bound rejected, and report.
    fn finish(self) -> SearchReport {
        let mut stats = self.stats;
        // Only a walk that ended by proving something can claim a prune. The
        // stop rule leaves the heap holding regions whose keys all reach the
        // incumbent; exhaustion leaves it empty; a budget exit leaves regions
        // that are merely unexamined, and those are nobody's pruning yield.
        stats.regions_pruned_unsolved = if self.outcome.is_proved() {
            self.heap
                .iter()
                .filter(|e| matches!(e.bound, Bound::Inherited))
                .count()
        } else {
            0
        };
        debug_assert!(
            stats.solves + stats.regions_pruned_unsolved <= stats.regions_pushed,
            "a region cannot be both solved and pruned unsolved"
        );
        debug_assert!(
            stats.regions_empty <= stats.solves,
            "an empty-lattice result is one particular kind of solve() call"
        );
        let modular = self.modular;
        SearchReport {
            best_objective: modular.then_some(self.incumbent),
            best: self.best.or(self.fallback),
            modular,
            outcome: self.outcome,
            stats,
            elapsed_s: self.started.elapsed().as_secs_f64(),
            trajectory: self.trajectory,
            bound_probe: self.probe,
            batch_probe: self.batches,
            marginal_s: self.marginal_s,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Region-algebra coverage: that a partition tiles its parent exactly once
    //! whatever order it branches in, which is what makes violation-guided
    //! branching free. The end-to-end search is exercised against the
    //! enumerator at `tests/bounded_search.rs`.

    use super::*;
    use derna_codon_dp::codon_table::aa_to_index;

    /// Every codon assignment a region admits, by brute force.
    fn members(region: &DesignRegion, aa: &[usize]) -> Vec<Vec<[u8; 3]>> {
        region
            .allowed(aa)
            .into_iter()
            .fold(vec![Vec::new()], |acc, choices| {
                acc.iter()
                    .flat_map(|prefix| {
                        choices.iter().map(move |c| {
                            let mut next = prefix.clone();
                            next.push(*c);
                            next
                        })
                    })
                    .collect()
            })
    }

    fn indices(protein: &str) -> Vec<usize> {
        protein.chars().map(|c| aa_to_index(c).unwrap()).collect()
    }

    /// The parent's members, minus the solution, must equal the disjoint union
    /// of the parts' members — for any branch order.
    fn assert_tiles(parent: &DesignRegion, aa: &[usize], solution: &[[u8; 3]], order: &[usize]) {
        let mut covered: Vec<Vec<[u8; 3]>> = Vec::new();
        for part in parent.partition(solution, order) {
            covered.extend(members(&part, aa));
        }
        let mut expected: Vec<Vec<[u8; 3]>> = members(parent, aa)
            .into_iter()
            .filter(|m| m != solution)
            .collect();
        let n_covered = covered.len();
        covered.sort();
        covered.dedup();
        assert_eq!(covered.len(), n_covered, "the parts overlap");
        expected.sort();
        assert_eq!(covered, expected, "the parts do not tile the parent");
    }

    #[test]
    fn a_partition_tiles_the_root_in_natural_order() {
        // L (6 codons) and S (6) give a 36-member space, small enough to check.
        let aa = indices("LS");
        let root = DesignRegion::root(aa.len());
        let solution = vec![
            AMINO_ACIDS[aa[0]].codons[0].nuc,
            AMINO_ACIDS[aa[1]].codons[0].nuc,
        ];
        assert_tiles(&root, &aa, &solution, &[0, 1]);
    }

    #[test]
    fn a_partition_tiles_the_root_in_reversed_order_too() {
        let aa = indices("LS");
        let root = DesignRegion::root(aa.len());
        let solution = vec![
            AMINO_ACIDS[aa[0]].codons[2].nuc,
            AMINO_ACIDS[aa[1]].codons[3].nuc,
        ];
        assert_tiles(&root, &aa, &solution, &[1, 0]);
    }

    /// The parent's own exclusions must survive into the part that leaves its
    /// branch position free — including when the order visits that position
    /// last, which is the case a natural-order partition never produces.
    #[test]
    fn a_partition_carries_the_parents_exclusions_through_any_order() {
        let aa = indices("LS");
        let root = DesignRegion::root(aa.len());
        let first = AMINO_ACIDS[aa[0]].codons[0].nuc;
        let mut excluded = vec![Vec::new(); aa.len()];
        excluded[0].push(first);
        let parent = DesignRegion {
            pinned: vec![None; aa.len()],
            excluded,
        };
        assert!(
            members(&parent, &aa).iter().all(|m| m[0] != first),
            "the parent excludes its branch codon"
        );
        let solution = vec![
            AMINO_ACIDS[aa[0]].codons[1].nuc,
            AMINO_ACIDS[aa[1]].codons[0].nuc,
        ];
        for order in [vec![0, 1], vec![1, 0]] {
            assert_tiles(&parent, &aa, &solution, &order);
            for part in parent.partition(&solution, &order) {
                assert!(
                    members(&part, &aa).iter().all(|m| m[0] != first),
                    "a part reinstated a codon the parent excluded"
                );
            }
        }
        // Nothing above depends on `root`, but a partition of it under the same
        // solution must still cover strictly more.
        assert!(members(&root, &aa).len() > members(&parent, &aa).len());
    }

    #[test]
    fn a_pinned_region_admits_exactly_one_sequence() {
        let aa = indices("LS");
        let region = DesignRegion {
            pinned: vec![
                Some(AMINO_ACIDS[aa[0]].codons[1].nuc),
                Some(AMINO_ACIDS[aa[1]].codons[2].nuc),
            ],
            excluded: vec![Vec::new(); 2],
        };
        assert_eq!(members(&region, &aa).len(), 1);
        assert!(region.free_positions().is_empty());
    }

    /// Excluding every codon of a position leaves an empty set, which is how a
    /// region reports itself infeasible rather than returning a garbage optimum.
    #[test]
    fn excluding_every_codon_empties_the_region() {
        let aa = indices("M"); // Methionine: one codon.
        let region = DesignRegion {
            pinned: vec![None],
            excluded: vec![vec![AMINO_ACIDS[aa[0]].codons[0].nuc]],
        };
        assert!(region.allowed(&aa)[0].is_empty());
    }

    #[test]
    fn the_heap_yields_the_least_bound_first() {
        let mut heap = BinaryHeap::new();
        for (i, key) in [3.0_f64, 1.0, 2.0].into_iter().enumerate() {
            heap.push(Entry {
                key,
                seq: i,
                region: DesignRegion::root(1),
                bound: Bound::Inherited,
                probe: None,
            });
        }
        let order: Vec<f64> = std::iter::from_fn(|| heap.pop()).map(|e| e.key).collect();
        assert_eq!(order, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn equal_bounds_break_ties_by_insertion_order() {
        let mut heap = BinaryHeap::new();
        for seq in [2, 0, 1] {
            heap.push(Entry {
                key: 1.0,
                seq,
                region: DesignRegion::root(1),
                bound: Bound::Inherited,
                probe: None,
            });
        }
        let order: Vec<usize> = std::iter::from_fn(|| heap.pop()).map(|e| e.seq).collect();
        assert_eq!(order, [0, 1, 2]);
    }
}
