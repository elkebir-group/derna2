//! Read-only observation of the front-valued fold under energy-budget drop
//! rules.
//!
//! The front fold's cost is governed by how large a cell's Pareto front grows:
//! every table merge is linear in it and every front-by-front product is
//! quadratic. A **completion bound** would shrink those fronts exactly — it
//! discards partial points that provably cannot reach a molecule within an
//! energy budget, which is a different thing from [`crate::front::Prune`]'s
//! lossy policies. Whether that shrinking is worth building is a question about
//! *growth rates*, and this module answers it without changing the fold:
//! it counts what each drop rule **would** have removed while removing nothing.
//!
//! Attaching a probe leaves the recovered front byte-identical by construction.
//! Nothing here is consulted by the recurrence; the counters are written and
//! never read back.
//!
//! # What is counted, and why both
//!
//! Two aggregates of per-cell front size are reported, because they disagree
//! and the disagreement is informative:
//!
//! - the **plain mean** over stored cells, which weights the many tiny cells
//!   near the base of the recursion equally with the few large ones; and
//! - the **product-weighted** size `sqrt(Σ k_left · k_right / calls)` over
//!   front-by-front products, which is the quantity the `wall ∝ structural · k̄²`
//!   cost model actually refers to.
//!
//! A rule that empties small cells and leaves large ones alone moves the first
//! and not the second. A report that quotes one alone cannot tell the two
//! situations apart.
//!
//! # Controls
//!
//! Two columns are controls rather than results and are always populated. A
//! drop rate is meaningless unless the budget was shown to forbid something, so
//! [`CellTally::bound_cells`] counts the cells where the rule removed at least
//! one point; and a rate is worth what its magnitude control is worth, so
//! [`CellTally::given_up_log_cai_sum`] records how much adaptation the rule cost
//! on exactly those cells.

use rna_mfe::Energy;

use crate::front::{Front, Point};

/// Which of the six CKY tables a cell belongs to.
///
/// The split matters because a left/right decomposition of a cell's completion
/// is exactly valid only for [`Self::Exterior`] — everywhere else the enclosing
/// pair's partner lies on the far side of the span, so "everything outside" is
/// one entangled quantity. A rule restricted to the exterior table is sound
/// where the fold's cost is *not* generated, so its reach is reported
/// separately rather than folded into a total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CellTable {
    /// Exterior parses (`C`).
    Exterior,
    /// Closed pairs (`P`).
    Pair,
    /// Multi-loops in progress (`Multi`).
    Multi,
    /// Shared multi-loop openings (`MultiOpen`).
    MultiOpen,
    /// Multi-branch components with at least one branch (`M`).
    Branch,
    /// Multi-branch components with at least two branches (`M2`).
    Branch2,
}

impl CellTable {
    /// Every table, in reporting order.
    pub const ALL: [CellTable; 6] = [
        CellTable::Exterior,
        CellTable::Pair,
        CellTable::Multi,
        CellTable::MultiOpen,
        CellTable::Branch,
        CellTable::Branch2,
    ];

    /// The short name used in emitted records.
    pub fn name(self) -> &'static str {
        match self {
            CellTable::Exterior => "exterior",
            CellTable::Pair => "pair",
            CellTable::Multi => "multi",
            CellTable::MultiOpen => "multi_open",
            CellTable::Branch => "m",
            CellTable::Branch2 => "m2",
        }
    }

    /// Position in [`CellTable::ALL`], for tallies stored as a flat array.
    pub fn index(self) -> usize {
        match self {
            CellTable::Exterior => 0,
            CellTable::Pair => 1,
            CellTable::Multi => 2,
            CellTable::MultiOpen => 3,
            CellTable::Branch => 4,
            CellTable::Branch2 => 5,
        }
    }
}

/// Where the fold forms a front-by-front product.
///
/// These three sites carry the whole of the quadratic term, and they are not
/// interchangeable: the exterior absorption is a small fraction of the volume
/// while the bifurcation and the single-loop corner closure carry the rest, so
/// a rule that reaches only one of them is bounded by that site's share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComposeSite {
    /// Attaching a closed pair to the exterior prefix before it.
    ExteriorAbsorb,
    /// The multi-branch bifurcation.
    Bifurcation,
    /// The single-branch (stack / bulge / internal) corner closure.
    InternalCorner,
}

impl ComposeSite {
    /// Every site, in reporting order.
    pub const ALL: [ComposeSite; 3] = [
        ComposeSite::ExteriorAbsorb,
        ComposeSite::Bifurcation,
        ComposeSite::InternalCorner,
    ];

    /// The short name used in emitted records.
    pub fn name(self) -> &'static str {
        match self {
            ComposeSite::ExteriorAbsorb => "exterior_absorb",
            ComposeSite::Bifurcation => "bifurcation",
            ComposeSite::InternalCorner => "internal_corner",
        }
    }

    /// Position in [`ComposeSite::ALL`], for tallies stored as a flat array.
    pub fn index(self) -> usize {
        match self {
            ComposeSite::ExteriorAbsorb => 0,
            ComposeSite::Bifurcation => 1,
            ComposeSite::InternalCorner => 2,
        }
    }
}

/// How a budget decides which of a cell's points could still reach a molecule
/// within it.
///
/// The rules are **ordered by how much they prune**, and the order runs one
/// way only. [`Self::CellWindow`] uses the tightest completion estimate
/// obtainable from a cell's own front, and the true cost-to-go can only be
/// tighter still — so an exact bound prunes **at least** as hard as the window,
/// and the window's reading is a floor under what any exact bound achieves, not
/// a ceiling over it.
///
/// What follows from that is asymmetric, and worth stating because it is easy
/// to read backwards. A window that *does* move a growth rate settles the
/// question affirmatively for every tighter rule. A window that *does not*
/// leaves the tighter rules open, and they can only be measured by computing
/// the true cost-to-go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropRule {
    /// Retain a point within the budget's slack of the cell's **own** best
    /// energy.
    ///
    /// This is a sound feasibility drop in its own right, at every table, and
    /// it costs one comparison against a value the cell already holds. Inside
    /// and outside a keyed cell are independent given the key, so the cell's
    /// best point plus its true best completion is the best whole molecule
    /// routed through the cell — which is never better than the molecule's
    /// unconstrained optimum. The true completion is therefore at least
    /// `optimum − best`, and crediting exactly that never overestimates it.
    ///
    /// Two things follow that a left/right completion term does not get. The
    /// argument never factorizes the outside into a left piece and a right
    /// piece, so it holds for a cell inside a multi-loop exactly as it does for
    /// an exterior branch. And the estimate only loosens if `best` is read
    /// before a cell has finished accumulating, since further merges can only
    /// lower it — so applying the rule mid-accumulation stays sound.
    ///
    /// It resembles a spread cap on the stored staircase and is not one: a cap
    /// discards points that could still have been optimal, while this discards
    /// only points no completion can rescue.
    CellWindow,
    /// [`Self::CellWindow`] applied to exterior cells only, every other table
    /// retained whole.
    ///
    /// Isolates how much of a rule's effect is reachable from the one table
    /// where a left/right decomposition of the completion is exactly valid,
    /// which is the table the fold's cost is *not* generated in.
    ExteriorWindow,
    /// A relaxation: credit the cell's outside with the most favourable energy
    /// the model admits for every loop it could possibly close, and compare the
    /// result against the optimum plus the slack.
    ///
    /// Deliberately generous — no real fold reaches the model's best loop at
    /// every one of its loops — so it underestimates the completion cost by a
    /// wide margin, and its looseness bounds every tighter linear-pass
    /// relaxation from below. See [`ProbeSpec::loop_floor`] for the counting
    /// argument that makes it sound.
    Relaxation,
}

impl DropRule {
    /// The short name used in emitted records.
    pub fn name(self) -> &'static str {
        match self {
            DropRule::CellWindow => "cell_window",
            DropRule::ExteriorWindow => "exterior_window",
            DropRule::Relaxation => "relaxation",
        }
    }
}

/// One drop rule at one energy slack — a single column of the report.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Which points the rule would keep.
    pub rule: DropRule,
    /// How far above the achievable optimum a molecule may fold and still be
    /// wanted.
    pub slack: Energy,
}

/// The immutable configuration a probe observes against.
#[derive(Debug, Clone)]
pub struct ProbeSpec {
    /// The columns to report, each an independent counting pass over the same
    /// fold.
    pub budgets: Vec<Budget>,
    /// The molecule's achievable minimum free energy, supplied by the caller
    /// (the energy-priority corner the scalar path already computes).
    pub optimum: Energy,
    /// The molecule's length in nucleotides.
    pub molecule_nt: u32,
    /// The most favourable energy any single loop can contribute — the most
    /// negative entry in the energy model — used by [`DropRule::Relaxation`].
    ///
    /// A structure on `m` nucleotides closes at most `m / 2` pairs and so sums
    /// at most `m / 2 + 1` loop terms, each no lower than this. Crediting every
    /// one of them the floor is therefore a genuine underestimate of any
    /// achievable completion, which is what makes the rule a relaxation rather
    /// than a subset minimum.
    pub loop_floor: Energy,
}

impl ProbeSpec {
    /// The retention threshold for `budget` in a cell of the given table whose
    /// best point sits at `best` and which already accounts for `inside_nt`
    /// nucleotides, or `None` when the rule does not apply to that table.
    fn threshold(
        &self,
        budget: Budget,
        table: CellTable,
        best: Energy,
        inside_nt: u32,
    ) -> Option<Energy> {
        match budget.rule {
            DropRule::CellWindow => Some(best + budget.slack),
            DropRule::ExteriorWindow if table == CellTable::Exterior => Some(best + budget.slack),
            DropRule::ExteriorWindow => None,
            DropRule::Relaxation => {
                let credit = self.completion_credit(inside_nt);
                Some(Energy(self.optimum.0 + budget.slack.0 - credit))
            }
        }
    }

    /// What [`DropRule::Relaxation`] credits the completion of a cell
    /// accounting for `inside_nt` nucleotides: the model's best loop, once per
    /// loop the rest of the molecule could possibly close.
    fn completion_credit(&self, inside_nt: u32) -> i32 {
        let outside = self.molecule_nt.saturating_sub(inside_nt);
        let loops = (outside / 2 + 1) as i32;
        self.loop_floor.0.saturating_mul(loops)
    }

    /// A lower bound, in energy, on how far `rule`'s completion estimate falls
    /// below the true best completion of a cell whose best point sits at
    /// `best`.
    ///
    /// The true completion is at least `optimum − best`, so any estimate more
    /// generous than that is loose by at least the difference. The window rules
    /// use that bound as their estimate, so their looseness is zero here by
    /// construction rather than by measurement.
    fn looseness(&self, rule: DropRule, best: Energy, inside_nt: u32) -> f64 {
        match rule {
            DropRule::CellWindow | DropRule::ExteriorWindow => 0.0,
            DropRule::Relaxation => {
                let true_floor = self.optimum.0 - best.0;
                f64::from(true_floor - self.completion_credit(inside_nt)) / 100.0
            }
        }
    }
}

/// How many points of a front a budget retains, and what that costs.
#[derive(Debug, Clone, Copy, Default)]
pub struct CellTally {
    /// Cells observed.
    pub cells: u64,
    /// Points stored across them, before any rule is applied.
    pub stored: u64,
    /// Points the rule would have retained.
    pub retained: u64,
    /// Cells where the rule removed at least one point — the control that says
    /// the budget forbade anything at all.
    pub bound_cells: u64,
    /// Cells the rule would have emptied outright.
    pub emptied_cells: u64,
    /// Summed log-CAI given up on the cells where the rule bound: the best
    /// adaptation in the cell less the best among the points it would keep.
    pub given_up_log_cai_sum: f64,
    /// The largest single such loss.
    pub given_up_log_cai_max: f64,
    /// Summed lower bound on the rule's **looseness**: how far its completion
    /// estimate falls below the true best completion, in energy.
    ///
    /// The true completion is at least the molecule's optimum less the cell's
    /// own best point, so a rule that credits the completion more generously
    /// than that is loose by at least the difference. Looseness matters because
    /// the window a rule really leaves open in a cell is the budget **plus** its
    /// looseness: a rule whose looseness grows with length gives back in slack
    /// what the budget bought in tightness. Zero for the window rules, whose
    /// estimate is that bound exactly.
    pub looseness_sum: f64,
    /// The largest single such gap.
    pub looseness_max: f64,
}

impl CellTally {
    fn absorb(&mut self, other: &CellTally) {
        self.looseness_sum += other.looseness_sum;
        self.looseness_max = self.looseness_max.max(other.looseness_max);
        self.cells += other.cells;
        self.stored += other.stored;
        self.retained += other.retained;
        self.bound_cells += other.bound_cells;
        self.emptied_cells += other.emptied_cells;
        self.given_up_log_cai_sum += other.given_up_log_cai_sum;
        self.given_up_log_cai_max = self.given_up_log_cai_max.max(other.given_up_log_cai_max);
    }
}

/// The volume of one front-by-front product site, with and without a rule.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairTally {
    /// Products formed.
    pub calls: u64,
    /// Summed `k_left · k_right` as the fold runs it.
    pub pairs: u128,
    /// Summed `k_left · k_right` over the points the rule would have kept.
    pub retained_pairs: u128,
}

impl PairTally {
    fn absorb(&mut self, other: &PairTally) {
        self.calls += other.calls;
        self.pairs += other.pairs;
        self.retained_pairs += other.retained_pairs;
    }
}

/// Stored and retained point counts bucketed by how much of the molecule a
/// cell spans, which is where a rule's reach separates: cells near the base of
/// the recursion hold fronts narrower than any useful slack and cannot be cut,
/// so an aggregate that mixes them in dilutes whatever happens higher up.
pub const SPAN_BUCKETS: usize = 16;

/// One budget's full accounting.
#[derive(Debug, Clone)]
pub struct BudgetTally {
    /// Per table.
    pub tables: [CellTally; 6],
    /// Per front-by-front product site.
    pub sites: [PairTally; 3],
    /// Cells and points by span bucket, as `(cells, stored, retained)`.
    pub by_span: [(u64, u64, u64); SPAN_BUCKETS],
}

impl Default for BudgetTally {
    fn default() -> Self {
        Self {
            tables: [CellTally::default(); 6],
            sites: [PairTally::default(); 3],
            by_span: [(0, 0, 0); SPAN_BUCKETS],
        }
    }
}

impl BudgetTally {
    fn absorb(&mut self, other: &BudgetTally) {
        for (dst, src) in self.tables.iter_mut().zip(&other.tables) {
            dst.absorb(src);
        }
        for (dst, src) in self.sites.iter_mut().zip(&other.sites) {
            dst.absorb(src);
        }
        for (dst, src) in self.by_span.iter_mut().zip(&other.by_span) {
            dst.0 += src.0;
            dst.1 += src.1;
            dst.2 += src.2;
        }
    }

    /// The plain per-cell mean front size, unconstrained and retained.
    pub fn mean_sizes(&self) -> (f64, f64) {
        let cells: u64 = self.tables.iter().map(|t| t.cells).sum();
        if cells == 0 {
            return (0.0, 0.0);
        }
        let stored: u64 = self.tables.iter().map(|t| t.stored).sum();
        let retained: u64 = self.tables.iter().map(|t| t.retained).sum();
        (stored as f64 / cells as f64, retained as f64 / cells as f64)
    }

    /// The product-weighted front size `sqrt(Σ k_left · k_right / calls)`,
    /// unconstrained and retained — the size the quadratic cost model means.
    pub fn product_sizes(&self) -> (f64, f64) {
        let calls: u64 = self.sites.iter().map(|s| s.calls).sum();
        if calls == 0 {
            return (0.0, 0.0);
        }
        let pairs: u128 = self.sites.iter().map(|s| s.pairs).sum();
        let kept: u128 = self.sites.iter().map(|s| s.retained_pairs).sum();
        (
            (pairs as f64 / calls as f64).sqrt(),
            (kept as f64 / calls as f64).sqrt(),
        )
    }
}

/// How many of each cell's points are optimal at some weight: the vertices of
/// its λ-envelope `min λ·E − (1 − λ)·100·A` over `λ ∈ [0, 1]`, which are the
/// strict vertices of the front's concave majorant from its most stable point
/// to its best-adapted one. A fold that carried each cell's envelope in place
/// of one scalar per weight would hold this many lines per cell.
#[derive(Debug, Clone, Default)]
pub struct HullTally {
    /// Cells observed.
    pub cells: u64,
    /// Envelope vertices summed over them.
    pub vertices: u64,
    /// Front points summed over them.
    pub points: u64,
    /// The largest envelope.
    pub max: u32,
    /// `(cells, vertices, points)` by the span bucket of the cell's inside.
    pub by_span: [(u64, u64, u64); SPAN_BUCKETS],
    /// Cells by envelope size: 1, 2, 3–4, 5–8, …, in powers of two.
    pub histogram: [u64; 16],
}

impl HullTally {
    fn observe<B>(&mut self, bucket: usize, points: &[Point<B>]) {
        let h = envelope_vertices(points);
        self.cells += 1;
        self.vertices += h as u64;
        self.points += points.len() as u64;
        self.max = self.max.max(h);
        let span = &mut self.by_span[bucket];
        span.0 += 1;
        span.1 += h as u64;
        span.2 += points.len() as u64;
        let class = (u32::BITS - (h.max(1) - 1).leading_zeros()) as usize;
        self.histogram[class.min(self.histogram.len() - 1)] += 1;
    }

    fn absorb(&mut self, other: &HullTally) {
        self.cells += other.cells;
        self.vertices += other.vertices;
        self.points += other.points;
        self.max = self.max.max(other.max);
        for (dst, src) in self.by_span.iter_mut().zip(&other.by_span) {
            dst.0 += src.0;
            dst.1 += src.1;
            dst.2 += src.2;
        }
        for (dst, src) in self.histogram.iter_mut().zip(&other.histogram) {
            *dst += src;
        }
    }
}

/// The strict vertices of the concave majorant of `points` in `(E, 100·A)`,
/// from the most stable point (best adapted among ties) to the best-adapted
/// one: the points some weight in `[0, 1]` makes uniquely optimal.
pub(crate) fn envelope_vertices<B>(points: &[Point<B>]) -> u32 {
    let mut xy: Vec<(f64, f64)> = points
        .iter()
        .map(|p| (f64::from(p.mfe.0), 100.0 * p.cai))
        .collect();
    xy.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut hull: Vec<(f64, f64)> = Vec::new();
    for p in xy {
        // Only points better adapted than every more stable one can be
        // optimal at a weight below 1.
        if hull.last().is_some_and(|q| p.1 <= q.1) {
            continue;
        }
        while hull.len() >= 2 {
            let (o, a) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            let cross = (a.0 - o.0) * (p.1 - o.1) - (a.1 - o.1) * (p.0 - o.0);
            if cross >= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    hull.len() as u32
}

/// One front's size and its per-budget retention, held while the front is an
/// operand of one or more products so each is computed once rather than once
/// per product.
#[derive(Debug, Clone, Default)]
pub struct RetainedCounts {
    len: u32,
    kept: Vec<u32>,
}

impl RetainedCounts {
    /// A front no rule touches — every budget keeps all `len` of its points.
    fn unfiltered(len: usize, budgets: usize) -> Self {
        Self {
            len: len as u32,
            kept: vec![len as u32; budgets],
        }
    }

    /// The front's unconstrained size.
    fn len(&self) -> u128 {
        self.len as u128
    }

    /// How many of its points survive `budget`.
    fn get(&self, budget: usize) -> u128 {
        self.kept.get(budget).copied().unwrap_or(0) as u128
    }
}

/// The accumulating observer.
///
/// Cloned into worker-local copies for the fold's parallel production phases
/// and folded back with [`Self::absorb`]; every counter is a sum or a maximum,
/// so the merge is order-independent and the totals do not depend on how the
/// work was divided.
#[derive(Debug, Clone)]
pub struct FrontProbe {
    spec: ProbeSpec,
    tallies: Vec<BudgetTally>,
    /// Retention counts for the cell currently being pushed through the
    /// single-branch closures, which composes against many transient corner
    /// fronts in a row.
    current: RetainedCounts,
    /// Cells observed, counted once rather than per budget.
    cells: u64,
    /// Points stored across them.
    stored: u64,
    /// Retention at the final exterior cell, per budget.
    root: Vec<u32>,
    /// Envelope sizes over every observed cell.
    hulls: HullTally,
}

impl FrontProbe {
    /// A probe that will report every budget in `spec`.
    pub fn new(spec: ProbeSpec) -> Self {
        let tallies = vec![BudgetTally::default(); spec.budgets.len()];
        Self {
            spec,
            tallies,
            current: RetainedCounts::default(),
            cells: 0,
            stored: 0,
            root: Vec::new(),
            hulls: HullTally::default(),
        }
    }

    /// Envelope sizes over every observed cell.
    pub fn hulls(&self) -> &HullTally {
        &self.hulls
    }

    /// What this probe was configured with.
    pub fn spec(&self) -> &ProbeSpec {
        &self.spec
    }

    /// One tally per budget, in the order the budgets were supplied.
    pub fn tallies(&self) -> &[BudgetTally] {
        &self.tallies
    }

    /// Cells observed, and points stored across them — the unconstrained
    /// control, independent of any budget.
    pub fn totals(&self) -> (u64, u64) {
        (self.cells, self.stored)
    }

    /// How many points each budget would keep at the **final exterior cell**,
    /// whose front is the molecule's complete Pareto staircase.
    ///
    /// This is the one cell where the window rule's answer is known
    /// independently: the cell's own best point *is* the achievable optimum, so
    /// the rule keeps exactly the designs that fold within the budget of it —
    /// the feasible set itself. A report that cannot reproduce that count from
    /// the emitted front has an instrument fault, not a finding, which is why
    /// it is carried separately rather than folded into the exterior table's
    /// total.
    pub fn root(&self) -> &[u32] {
        &self.root
    }

    /// A drained copy for a worker to accumulate into, sharing the
    /// configuration but none of the counts.
    pub fn split(&self) -> Self {
        Self::new(self.spec.clone())
    }

    /// Fold a worker's counts back in.
    pub fn absorb(&mut self, other: &FrontProbe) {
        self.cells += other.cells;
        self.stored += other.stored;
        self.hulls.absorb(&other.hulls);
        if self.root.is_empty() {
            self.root = other.root.clone();
        }
        for (dst, src) in self.tallies.iter_mut().zip(&other.tallies) {
            dst.absorb(src);
        }
    }

    /// Record the final exterior cell, whose front is the complete staircase.
    ///
    /// Counted as an ordinary exterior cell as well, so no cell is missing from
    /// the aggregates; the separate copy is what [`Self::root`] reports.
    pub fn observe_root<B>(&mut self, front: &Front<B>) {
        let counts = self.observe_cell(CellTable::Exterior, self.spec.molecule_nt, front);
        self.root = counts.kept.clone();
    }

    /// Record a stored cell and return its per-budget retention counts.
    ///
    /// `inside_nt` is how many nucleotides the cell's points already account
    /// for; only [`DropRule::Relaxation`] consults it.
    pub fn observe_cell<B>(
        &mut self,
        table: CellTable,
        inside_nt: u32,
        front: &Front<B>,
    ) -> RetainedCounts {
        let points = front.points();
        if points.is_empty() {
            return RetainedCounts::default();
        }
        self.cells += 1;
        self.stored += points.len() as u64;
        let bucket = span_bucket(inside_nt, self.spec.molecule_nt);
        self.hulls.observe(bucket, points);
        let counts = self.counts_only(table, inside_nt, front);
        let best = points[0].mfe;
        for (idx, tally) in self.tallies.iter_mut().enumerate() {
            let kept = counts.kept[idx] as usize;
            let slack = self
                .spec
                .looseness(self.spec.budgets[idx].rule, best, inside_nt);
            record_cell(&mut tally.tables[table.index()], points, kept, slack);
            let span = &mut tally.by_span[bucket];
            span.0 += 1;
            span.1 += points.len() as u64;
            span.2 += kept as u64;
        }
        counts
    }

    /// Record a front-by-front product between two fronts whose retention
    /// counts are already known.
    pub fn observe_product(
        &mut self,
        site: ComposeSite,
        left: &RetainedCounts,
        right: &RetainedCounts,
    ) {
        let idx = site.index();
        let volume = left.len() * right.len();
        for (budget, tally) in self.tallies.iter_mut().enumerate() {
            let entry = &mut tally.sites[idx];
            entry.calls += 1;
            entry.pairs += volume;
            entry.retained_pairs += left.get(budget) * right.get(budget);
        }
    }

    /// Latch the retention counts of the cell about to be composed against a
    /// run of transient fronts, so they are computed once for the run rather
    /// than once per product.
    pub fn latch<B>(&mut self, table: CellTable, inside_nt: u32, front: &Front<B>) {
        self.current = self.counts_only(table, inside_nt, front);
    }

    /// Record a product between the latched cell and a transient front.
    ///
    /// The other operand is charged **unfiltered**, and deliberately: a drop
    /// rule applies to the fronts the fold stores, and this one is never
    /// stored. Its points reach a stored cell only as part of the product, and
    /// are filtered there.
    pub fn observe_latched_product<B>(&mut self, site: ComposeSite, other: &Front<B>) {
        if self.current.len == 0 {
            return;
        }
        let right = RetainedCounts::unfiltered(other.len(), self.spec.budgets.len());
        let latched = std::mem::take(&mut self.current);
        self.observe_product(site, &latched, &right);
        self.current = latched;
    }

    /// Size and retention without recording the front as a stored cell — for
    /// the transient products a cell is composed against.
    fn counts_only<B>(&self, table: CellTable, inside_nt: u32, front: &Front<B>) -> RetainedCounts {
        let points = front.points();
        if points.is_empty() {
            return RetainedCounts::default();
        }
        let best = points[0].mfe;
        let kept = self
            .spec
            .budgets
            .iter()
            .map(
                |budget| match self.spec.threshold(*budget, table, best, inside_nt) {
                    Some(limit) => points.partition_point(|p| p.mfe <= limit) as u32,
                    None => points.len() as u32,
                },
            )
            .collect();
        RetainedCounts {
            len: points.len() as u32,
            kept,
        }
    }
}

/// Add one cell's contribution to a table's tally, including the two controls:
/// whether the rule bound at all, and what it cost where it did.
fn record_cell<B>(
    tally: &mut CellTally,
    points: &[crate::front::Point<B>],
    kept: usize,
    looseness: f64,
) {
    tally.cells += 1;
    tally.stored += points.len() as u64;
    tally.retained += kept as u64;
    tally.looseness_sum += looseness;
    tally.looseness_max = tally.looseness_max.max(looseness);
    if kept == points.len() {
        return;
    }
    tally.bound_cells += 1;
    if kept == 0 {
        tally.emptied_cells += 1;
        return;
    }
    // The staircase is stored energy-ascending, which forces adaptation-
    // ascending, so the best adaptation is the last point and the best the rule
    // keeps is the last it keeps.
    let lost = points[points.len() - 1].cai - points[kept - 1].cai;
    tally.given_up_log_cai_sum += lost;
    tally.given_up_log_cai_max = tally.given_up_log_cai_max.max(lost);
}

/// Which span bucket a cell accounting for `inside_nt` of `molecule_nt` falls
/// in.
fn span_bucket(inside_nt: u32, molecule_nt: u32) -> usize {
    if molecule_nt == 0 {
        return 0;
    }
    let scaled = (inside_nt as u64 * SPAN_BUCKETS as u64) / (molecule_nt as u64 + 1);
    (scaled as usize).min(SPAN_BUCKETS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::front::Front;

    fn front(mfes: &[i32]) -> Front<u32> {
        let points = mfes
            .iter()
            .enumerate()
            .map(|(i, &m)| crate::front::Point::new(Energy(m), i as f64, i as u32))
            .collect();
        Front::from_points(points)
    }

    fn spec(rule: DropRule, slack: i32) -> ProbeSpec {
        ProbeSpec {
            budgets: vec![Budget {
                rule,
                slack: Energy(slack),
            }],
            optimum: Energy(-1000),
            molecule_nt: 100,
            loop_floor: Energy(-340),
        }
    }

    #[test]
    fn cell_window_keeps_the_prefix_within_slack_of_the_cells_own_best() {
        let mut probe = FrontProbe::new(spec(DropRule::CellWindow, 250));
        let f = front(&[-500, -400, -300, -200, -100]);
        let counts = probe.observe_cell(CellTable::Pair, 30, &f);
        // Best is -500, so the window admits everything at or below -250.
        assert_eq!(counts.get(0), 3);
        let tally = &probe.tallies()[0].tables[CellTable::Pair.index()];
        assert_eq!(tally.stored, 5);
        assert_eq!(tally.retained, 3);
        assert_eq!(tally.bound_cells, 1);
    }

    #[test]
    fn a_window_wider_than_the_front_removes_nothing_and_does_not_count_as_bound() {
        let mut probe = FrontProbe::new(spec(DropRule::CellWindow, 100_000));
        let f = front(&[-500, -400, -300]);
        assert_eq!(probe.observe_cell(CellTable::Pair, 30, &f).get(0), 3);
        let tally = &probe.tallies()[0].tables[CellTable::Pair.index()];
        assert_eq!(tally.bound_cells, 0);
        assert_eq!(tally.given_up_log_cai_sum, 0.0);
    }

    #[test]
    fn the_exterior_rule_leaves_every_other_table_whole() {
        let mut probe = FrontProbe::new(spec(DropRule::ExteriorWindow, 50));
        let f = front(&[-500, -400, -300]);
        assert_eq!(probe.observe_cell(CellTable::Pair, 30, &f).get(0), 3);
        assert_eq!(probe.observe_cell(CellTable::Exterior, 30, &f).get(0), 1);
        assert_eq!(
            probe.tallies()[0].tables[CellTable::Pair.index()].bound_cells,
            0
        );
        assert_eq!(
            probe.tallies()[0].tables[CellTable::Exterior.index()].bound_cells,
            1
        );
    }

    #[test]
    fn the_relaxation_credits_every_outside_nucleotide_so_it_drops_less_than_the_window() {
        let f = front(&[-500, -400, -300, -200]);
        let mut window = FrontProbe::new(spec(DropRule::CellWindow, 150));
        let mut relaxed = FrontProbe::new(spec(DropRule::Relaxation, 150));
        let kept_window = window.observe_cell(CellTable::Pair, 30, &f).get(0);
        let kept_relaxed = relaxed.observe_cell(CellTable::Pair, 30, &f).get(0);
        assert!(
            kept_relaxed >= kept_window,
            "a relaxation underestimates the completion cost, so it can only keep more: \
             {kept_relaxed} < {kept_window}"
        );
    }

    #[test]
    fn the_adaptation_given_up_is_the_gap_to_the_best_point_the_rule_drops() {
        let mut probe = FrontProbe::new(spec(DropRule::CellWindow, 150));
        let f = front(&[-500, -400, -300, -200]);
        probe.observe_cell(CellTable::Pair, 30, &f);
        // Points carry adaptation 0,1,2,3; the window keeps the first two.
        let tally = &probe.tallies()[0].tables[CellTable::Pair.index()];
        assert_eq!(tally.retained, 2);
        assert_eq!(tally.given_up_log_cai_sum, 2.0);
        assert_eq!(tally.given_up_log_cai_max, 2.0);
    }

    #[test]
    fn product_volume_is_charged_before_and_after_the_rule() {
        let mut probe = FrontProbe::new(spec(DropRule::CellWindow, 150));
        let left = front(&[-500, -400, -300, -200]);
        let right = front(&[-90, -80, -70]);
        let lc = probe.observe_cell(CellTable::Branch, 30, &left);
        let rc = probe.observe_cell(CellTable::Pair, 20, &right);
        probe.observe_product(ComposeSite::Bifurcation, &lc, &rc);
        let site = &probe.tallies()[0].sites[ComposeSite::Bifurcation.index()];
        assert_eq!(site.calls, 1);
        assert_eq!(site.pairs, 12);
        assert_eq!(site.retained_pairs, 2 * 3);
    }

    #[test]
    fn absorbing_a_workers_counts_sums_them() {
        let mut main = FrontProbe::new(spec(DropRule::CellWindow, 150));
        let mut worker = main.split();
        let f = front(&[-500, -400, -300, -200]);
        main.observe_cell(CellTable::Pair, 30, &f);
        worker.observe_cell(CellTable::Pair, 30, &f);
        main.absorb(&worker);
        assert_eq!(main.totals(), (2, 8));
        assert_eq!(
            main.tallies()[0].tables[CellTable::Pair.index()].retained,
            4
        );
    }

    #[test]
    fn the_window_rules_carry_no_looseness_and_the_relaxation_carries_a_lot() {
        let f = front(&[-500, -400, -300]);
        let mut window = FrontProbe::new(spec(DropRule::CellWindow, 150));
        let mut relaxed = FrontProbe::new(spec(DropRule::Relaxation, 150));
        window.observe_cell(CellTable::Pair, 30, &f);
        relaxed.observe_cell(CellTable::Pair, 30, &f);
        assert_eq!(
            window.tallies()[0].tables[CellTable::Pair.index()].looseness_max,
            0.0
        );
        // The cell's best is -5.00 against an optimum of -10.00, so the true
        // completion is at least -5.00; the relaxation credits 36 loops of
        // -3.40, so it is loose by at least 117.40 kcal/mol.
        let measured = relaxed.tallies()[0].tables[CellTable::Pair.index()].looseness_max;
        assert!(
            (measured - 117.40).abs() < 1e-6,
            "looseness was {measured}, expected 117.40"
        );
    }

    #[test]
    fn envelope_counts_the_concave_majorants_strict_vertices() {
        let at = |e: i32, a: f64| Point::new(Energy(e), a, 0u32);
        // (2, 6) sits under the chord from (1, 5) to (3, 10); (2, 4) and
        // (0, 0) are dominated; (4, 12) is collinear with (3, 10)..(5, 14).
        let points = [
            at(0, 0.0),
            at(0, 0.01),
            at(1, 0.05),
            at(2, 0.06),
            at(2, 0.04),
            at(3, 0.10),
            at(4, 0.12),
            at(5, 0.14),
        ];
        assert_eq!(envelope_vertices(&points), 4);
        assert_eq!(envelope_vertices(&points[..1]), 1);
        assert_eq!(envelope_vertices::<u32>(&[]), 0);
    }

    #[test]
    fn span_buckets_stay_in_range_at_both_ends() {
        assert_eq!(span_bucket(0, 300), 0);
        assert_eq!(span_bucket(300, 300), SPAN_BUCKETS - 1);
        assert_eq!(span_bucket(9_999, 300), SPAN_BUCKETS - 1);
        assert_eq!(span_bucket(5, 0), 0);
    }
}
