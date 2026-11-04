//! The feasibility box a front fold is solved under, and the in-fold drop that
//! enforces it.
//!
//! A caller can ask for the best-adapted design that still folds at least as
//! stably as some threshold. Recovering the whole Pareto front and truncating it
//! answers that correctly, and pays in full for every derivation the threshold
//! already excludes. Carrying the threshold *into* the fold and discarding those
//! derivations as they are formed reaches the same answer for less work —
//! provided the discard is exact.
//!
//! # Why the drop is exact
//!
//! A cell's stored front holds the inside energy of every derivation reaching
//! that cell's key. Everything outside the cell sees only the key, so the set of
//! completions — and each completion's energy — is a property of the key alone,
//! shared by every point the cell holds. Write `C` for the cheapest of them.
//!
//! Combining the cell's own cheapest point with `C` yields a whole derivation of
//! a real molecule, whose energy is therefore no lower than `optimum`, the
//! minimum over every encoding and every structure. So
//!
//! ```text
//! C >= optimum - cell_min
//! ```
//!
//! and crediting a point's completion exactly `optimum - cell_min` can never
//! claim the rest of the molecule is cheaper than it really is. Dropping a point
//! when `point.mfe + (optimum - cell_min) > ceiling` therefore discards only
//! derivations that no completion can bring within the ceiling. Rearranged, the
//! survivors are the points within `ceiling - optimum` of their own cell's best,
//! which is why the fold carries a single [`Slack`] window rather than a
//! per-cell bound.
//!
//! What is discarded is a *derivation*, never a sequence: a sequence folding
//! within the ceiling keeps its own minimum-energy derivation, whose total is
//! that sequence's energy and so passes the test wherever it is applied.
//!
//! Three properties follow, and each shapes the implementation rather than
//! merely justifying it.
//!
//! **The argument never splits the outside into a left piece and a right
//! piece.** It needs the completion to be *some* real derivation, not a
//! decomposable one, so it holds for a cell inside a multi-loop exactly as for
//! an exterior branch, and applies at all six tables. An estimate taken as the
//! best completion over one *usage* of a cell would be a minimum over a subset
//! of the usages, which overestimates the completion cost and so drops feasible
//! derivations. This is not that, and the difference is why no table is exempt.
//!
//! **It stays sound mid-accumulation.** A cell read before it has finished
//! accumulating has a `cell_min` no lower than its final value, so the window
//! applied is at least as wide as the final one — the rule applied early is a
//! relaxation of itself. The surviving set is the same either way, since
//! anything a wider window excludes a narrower one excludes too. That is what
//! keeps the fold's output independent of merge order and of worker count.
//!
//! **It can never empty a cell**, a cell's own best point being trivially within
//! any window of itself. Infeasibility is therefore not something the drop
//! detects; it is settled before the fold, by comparing the ceiling against the
//! achievable minimum ([`Bounds::resolve`]).
//!
//! # The one unsound direction
//!
//! All of the above rests on `optimum` being no higher than the true minimum
//! over the space the fold searches. An `optimum` that is too *low* only shrinks
//! the credit: the drop stays admissible and is merely looser. An `optimum` that
//! is too *high* — a corner minimized under a constraint the fold does not
//! apply, or over a shorter molecule than the fold assembles — inflates the
//! credit and discards feasible designs, making the answer wrong rather than the
//! bound loose. [`Bounds::with_credit_offset`] exists to make that direction
//! reachable deliberately, so the exactness tests can show they would catch it.

use rna_mfe::Energy;

use crate::front::Front;
use crate::front_probe::CellTable;

/// How a caller states an energy ceiling.
///
/// Energy is not scale-free — it grows with length and differs between proteins
/// — so a bare threshold names no result on its own and both forms are offered.
/// They resolve to one absolute [`Energy`] before the fold starts, so the
/// solver, the report and any oracle compare the same integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ceiling {
    /// A threshold in the fold's own energy units.
    Absolute(Energy),
    /// A non-negative offset above the achievable minimum: `optimum + offset`.
    AboveOptimum(Energy),
}

impl Ceiling {
    /// The absolute threshold this states, given the achievable minimum.
    pub fn resolve(self, optimum: Energy) -> Energy {
        match self {
            Ceiling::Absolute(sigma) => sigma,
            Ceiling::AboveOptimum(offset) => Energy(optimum.0.saturating_add(offset.0)),
        }
    }
}

/// The feasibility box a fold is solved under.
///
/// One parameter rather than one per constraint, because every table merge
/// funnels through the same choke point and a box query costs no more there than
/// a single-sided one. Only the energy ceiling is filled; a bound on the
/// adaptation axis belongs in this type too and is deliberately absent rather
/// than present and ignored, so a caller cannot set something the fold does not
/// read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    ceiling: Option<Ceiling>,
    credit_offset: Energy,
    credit_offset_scope: CreditScope,
    exact_credit: bool,
    filter_offset: Option<Energy>,
    filter_offset_at: Option<CellTable>,
    floor_cut: bool,
}

/// Which cells a [`Bounds::with_credit_offset`] applies to.
///
/// Two scopes rather than one because the two express different failures. An
/// offset everywhere is a corner computed too high, which is wrong at every
/// cell at once. An offset at every table *but* the exterior is the shape of a
/// completion estimate that decomposes the outside into a left piece and a right
/// piece: exactly valid where a cell is definitionally an exterior branch, and
/// inadmissible at the five tables where it is not, because there "everything
/// else" is one entangled quantity with no left and right to it. Only a mutation
/// confined to those five can show that the exactness comparison is sensitive
/// there, rather than being killed at the one table that was never in doubt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CreditScope {
    /// Every table.
    #[default]
    Everywhere,
    /// Every table except the exterior.
    BeyondExterior,
}

impl Default for Bounds {
    /// [`Bounds::UNBOUNDED`] — a fold with no box asked for is the
    /// unconstrained one.
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

impl Bounds {
    /// No constraint at all — the fold is the unconstrained one, byte for byte.
    pub const UNBOUNDED: Bounds = Bounds {
        ceiling: None,
        credit_offset: Energy(0),
        credit_offset_scope: CreditScope::Everywhere,
        exact_credit: false,
        filter_offset: Some(Energy(0)),
        filter_offset_at: None,
        floor_cut: false,
    };

    /// Solve under an energy ceiling.
    pub fn with_ceiling(ceiling: Ceiling) -> Self {
        Self {
            ceiling: Some(ceiling),
            ..Self::UNBOUNDED
        }
    }

    /// The ceiling this box carries, if any.
    pub fn ceiling(&self) -> Option<Ceiling> {
        self.ceiling
    }

    /// Credit each cell's completion with its exact cost to go
    /// ([`crate::cost_to_go::Outside`]) wherever one exists, instead of the
    /// window alone.
    ///
    /// The answer is the same either way; both credits are admissible and the
    /// larger is used. What differs is what can be discarded. The window can
    /// never empty a cell, since a cell's own best point is within any window of
    /// itself; the exact credit empties every cell whose best molecule misses
    /// the ceiling, and an emptied cell is never composed. The price is one
    /// scalar `λ = 1` fold and one backward pass before the front fold starts.
    ///
    /// A credit offset applies to the exact credit exactly as to the window, so
    /// the exactness tests' deliberate inadmissibility reaches both.
    pub fn with_exact_credit(mut self) -> Self {
        self.exact_credit = true;
        self
    }

    /// Whether the exact cost to go is credited where it exists.
    pub fn exact_credit(&self) -> bool {
        self.exact_credit
    }

    /// Also cut a box's cells at its floor alone (`λ = 0`), with no pass: on a
    /// lattice whose every edge weight is at most zero, no completion adds
    /// adaptation, so a point already below the floor has no completion in
    /// the box. Read only where a box is given.
    pub fn with_floor_cut(mut self) -> Self {
        self.floor_cut = true;
        self
    }

    /// Whether a box's floor is cut at every cell ([`Self::with_floor_cut`]).
    pub fn floor_cut(&self) -> bool {
        self.floor_cut
    }

    /// Add `offset` to the completion credit every drop consults.
    ///
    /// Zero on every path that solves anything, and the only way to reach the
    /// unsound direction on purpose. A positive offset credits the completion
    /// more than the rest of the molecule can actually deliver, so the drop
    /// starts discarding derivations that could still have completed within the
    /// ceiling — which is exactly the failure a too-high achievable minimum
    /// would cause. It exists so the exactness proof can demonstrate it is
    /// capable of failing, rather than asserting a property that holds by
    /// construction and calling that a test.
    pub fn with_credit_offset(mut self, offset: Energy) -> Self {
        self.credit_offset = offset;
        self.credit_offset_scope = CreditScope::Everywhere;
        self
    }

    /// Add `offset` to the completion credit at every table except the
    /// exterior — the inadmissibility a completion estimate that factorizes the
    /// outside would have, in the only place it would have it.
    ///
    /// Zero on every path that solves anything, and present for the same reason
    /// as [`Bounds::with_credit_offset`]: an exactness comparison that cannot be
    /// made to fail is not evidence of anything. This is the narrower of the
    /// two, and the one that says the comparison is sensitive at the cells
    /// inside a multi-loop rather than only at the one table where a decomposed
    /// estimate would have been sound anyway.
    pub fn with_credit_offset_beyond_exterior(mut self, offset: Energy) -> Self {
        self.credit_offset = offset;
        self.credit_offset_scope = CreditScope::BeyondExterior;
        self
    }

    /// Lower the energy a filtered compose generates products up to by
    /// `offset`, below the cutoff its merge truncates at.
    ///
    /// Zero on every path that solves anything. A compose skips only products
    /// its merge would discard, so the answer cannot tell whether it skips them;
    /// this makes it skip ones the merge would keep, so a test can show the
    /// skipping is read and that a filter one unit too tight changes an answer.
    #[doc(hidden)]
    pub fn with_filter_offset(mut self, offset: Energy) -> Self {
        self.filter_offset = Some(offset);
        self.filter_offset_at = None;
        self
    }

    /// [`Bounds::with_filter_offset`] at the composes into `table` alone, so a
    /// test can show each filtered site is read on its own.
    #[doc(hidden)]
    pub fn with_filter_offset_at(mut self, table: CellTable, offset: Energy) -> Self {
        self.filter_offset = Some(offset);
        self.filter_offset_at = Some(table);
        self
    }

    /// Generate every product of every compose, as the fold did before its
    /// composes skipped what their merges discard. The answer is the same
    /// either way; this exists so a test can show it is byte for byte.
    #[doc(hidden)]
    pub fn without_compose_filter(mut self) -> Self {
        self.filter_offset = None;
        self.filter_offset_at = None;
        self
    }

    /// Resolve against the achievable minimum into the window the fold carries,
    /// or report that nothing folds within the ceiling.
    ///
    /// This is where feasibility is decided, and it is decided from the ceiling
    /// and the achievable minimum alone — never from the credit offset, so a
    /// deliberately inflated credit perturbs the drop without also moving the
    /// feasibility verdict underneath it.
    pub fn resolve(&self, optimum: Energy) -> Result<Windows, Unreachable> {
        let Some(ceiling) = self.ceiling else {
            return Ok(Windows::UNBOUNDED);
        };
        let sigma = ceiling.resolve(optimum);
        if sigma.0 < optimum.0 {
            return Err(Unreachable {
                ceiling: sigma,
                achievable: optimum,
            });
        }
        let modular = Slack(Some(Energy(sigma.0 - optimum.0)));
        let offset = Slack(Some(Energy(sigma.0 - optimum.0 - self.credit_offset.0)));
        Ok(match self.credit_offset_scope {
            CreditScope::Everywhere => Windows {
                exterior: offset,
                beyond_exterior: offset,
                filter_offset: self.filter_offset,
                filter_offset_at: self.filter_offset_at,
            },
            CreditScope::BeyondExterior => Windows {
                exterior: modular,
                beyond_exterior: offset,
                filter_offset: self.filter_offset,
                filter_offset_at: self.filter_offset_at,
            },
        })
    }
}

/// No design folds within the requested ceiling.
///
/// Carries the achievable minimum alongside, since a caller who set a threshold
/// too low needs to know what the molecule can actually reach in order to set a
/// reachable one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unreachable {
    /// The absolute ceiling that was asked for.
    pub ceiling: Energy,
    /// The achievable minimum free energy, which exceeds it.
    pub achievable: Energy,
}

/// The windows a fold carries, one for the exterior table and one for the five
/// where a completion estimate that decomposes the outside would be
/// inadmissible.
///
/// They are equal on every path that solves anything — the shipped credit makes
/// no distinction between tables, which is the whole of why it sidesteps the
/// asymmetry. They differ only under
/// [`Bounds::with_credit_offset_beyond_exterior`], so that the exactness
/// comparison can be shown to notice an estimate that is sound at the exterior
/// and wrong everywhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Windows {
    /// The window applied at the exterior table.
    pub exterior: Slack,
    /// The window applied at the other five.
    pub beyond_exterior: Slack,
    /// How far below its merge's cutoff a bifurcation's compose stops
    /// generating products: zero except under [`Bounds::with_filter_offset`],
    /// and `None`, generating every product, under
    /// [`Bounds::without_compose_filter`].
    pub filter_offset: Option<Energy>,
    /// The one table the offset applies at, or every filtered one; see
    /// [`Bounds::with_filter_offset_at`].
    pub filter_offset_at: Option<CellTable>,
}

impl Windows {
    /// The offset a filtered compose into `table` applies, or `None` where
    /// the filter is off.
    pub(crate) fn filter_offset(&self, table: CellTable) -> Option<Energy> {
        let offset = self.filter_offset?;
        let applies = self.filter_offset_at.is_none_or(|at| at == table);
        Some(if applies { offset } else { Energy(0) })
    }
}

impl Default for Windows {
    /// [`Windows::UNBOUNDED`].
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

impl Windows {
    /// No window anywhere — the unconstrained fold.
    pub const UNBOUNDED: Windows = Windows {
        exterior: Slack::UNBOUNDED,
        beyond_exterior: Slack::UNBOUNDED,
        filter_offset: Some(Energy(0)),
        filter_offset_at: None,
    };

    /// Whether either window constrains anything.
    pub fn is_bounded(self) -> bool {
        self.exterior.is_bounded() || self.beyond_exterior.is_bounded()
    }
}

/// How far above its own cell's best point a partial derivation may sit and
/// still be worth carrying — the resolved form of a [`Bounds`], one integer wide.
///
/// The window is the same everywhere: `ceiling - optimum`, a molecule-level
/// quantity, compared against a cell-level best. That is what makes the test one
/// comparison against a value the cell already holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Slack(Option<Energy>);

impl Slack {
    /// No window — every derivation is carried, and [`retain_feasible`] is a
    /// no-op. The fold under this is the unconstrained fold.
    pub const UNBOUNDED: Slack = Slack(None);

    /// The window, or `None` when unbounded.
    pub fn window(self) -> Option<Energy> {
        self.0
    }

    /// Whether this constrains anything.
    pub fn is_bounded(self) -> bool {
        self.0.is_some()
    }
}

/// Discard every point in `cell` that no completion can bring within the
/// window, returning how many were removed.
///
/// Applied beside each cell's own reduction and never inside it: a lossy
/// per-cell policy is free to short-circuit on cell size or on being switched
/// off, and this must run whether or not one did — the configuration that
/// matters most is an exact fold under a ceiling, where every such
/// short-circuit fires.
///
/// The test is monotone in `mfe` over a staircase stored `mfe`-ascending, so the
/// survivors are a prefix: one binary search and a truncation, with no
/// elimination sweep, no re-sort and no reallocation.
pub fn retain_feasible<B>(cell: &mut Front<B>, slack: Slack) -> usize {
    retain_within(cell, slack, None)
}

/// [`retain_feasible`] with an exact per-cell cutoff beside the window: keep a
/// point only if `point.mfe <= exact` as well, where `exact` is the ceiling
/// less the cell's true cost to go (see [`ExactCredit::cutoff`]).
///
/// Unlike the window alone this **can empty a cell**: the exact cutoff falls
/// below a cell's own best exactly when the best molecule through the cell
/// misses the ceiling. `None` is the window alone, byte for byte.
pub fn retain_within<B>(cell: &mut Front<B>, slack: Slack, exact: Option<Energy>) -> usize {
    let Some(window) = slack.window() else {
        return 0;
    };
    let Some(best) = cell.points().first() else {
        return 0;
    };
    let mut cutoff = Energy(best.mfe.0.saturating_add(window.0));
    if let Some(exact) = exact {
        cutoff = cutoff.min(exact);
    }
    cell.truncate_mfe(cutoff)
}

/// The exact completion credit a front fold consults: each cell's true cost to
/// go, and the energy the fold's windows were resolved against.
///
/// Stated as a cutoff rather than a credit so it composes with the window by a
/// minimum. A window `w` was resolved as `ceiling - anchor - offset`, so
/// `anchor + w - cost_to_go` is `ceiling - offset - cost_to_go` whatever the
/// anchor was: a credit offset inflates this credit exactly as it inflates the
/// window, and the per-table scope of an offset carries over with `w`.
pub struct ExactCredit {
    outside: crate::cost_to_go::Outside,
    anchor: Energy,
}

impl ExactCredit {
    /// The credit from `outside`, for windows resolved against `anchor`.
    pub fn new(outside: crate::cost_to_go::Outside, anchor: Energy) -> Self {
        Self { outside, anchor }
    }

    /// The exact `λ = 1` cost to go the credit reads.
    pub fn outside(&self) -> &crate::cost_to_go::Outside {
        &self.outside
    }

    /// The largest energy a point of `cell` may carry and still complete within
    /// the ceiling, under the table's window `slack`. `None` where the fold is
    /// unbounded or the cell has no cost to go, in which case the window alone
    /// applies — abstaining is the sound direction.
    pub fn cutoff(&self, cell: crate::cost_to_go::CellRef, slack: Slack) -> Option<Energy> {
        let window = slack.window()?;
        let out = self.outside.at(cell)?;
        Some(Energy(
            self.anchor.0.saturating_add(window.0).saturating_sub(out.0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn front(mfes: &[i32]) -> Front<()> {
        Front::from_points(
            mfes.iter()
                .enumerate()
                .map(|(i, &e)| crate::front::Point::new(Energy(e), i as f64, ()))
                .collect(),
        )
    }

    /// The window a box resolves to, when both of its windows agree.
    fn window_of(bounds: Bounds, optimum: Energy) -> Slack {
        let windows = bounds.resolve(optimum).unwrap();
        assert_eq!(windows.exterior, windows.beyond_exterior);
        windows.exterior
    }

    #[test]
    fn an_unbounded_box_resolves_to_no_window_and_drops_nothing() {
        let slack = window_of(Bounds::UNBOUNDED, Energy(-500));
        assert_eq!(slack, Slack::UNBOUNDED);
        let mut cell = front(&[-500, -400, -100]);
        assert_eq!(retain_feasible(&mut cell, slack), 0);
        assert_eq!(cell.len(), 3);
    }

    #[test]
    fn the_window_is_measured_from_the_cells_own_best_not_from_the_optimum() {
        // A cell whose best sits well above the molecule's optimum keeps the
        // same span of points as one sitting at it: the completion credit
        // absorbs the difference, which is the whole of the rule.
        let slack = window_of(
            Bounds::with_ceiling(Ceiling::AboveOptimum(Energy(300))),
            Energy(-1000),
        );
        assert_eq!(slack.window(), Some(Energy(300)));

        let mut at_optimum = front(&[-1000, -800, -700, -600]);
        assert_eq!(retain_feasible(&mut at_optimum, slack), 1);
        assert_eq!(at_optimum.len(), 3);

        let mut well_above = front(&[-400, -200, -100, 0]);
        assert_eq!(retain_feasible(&mut well_above, slack), 1);
        assert_eq!(well_above.len(), 3);
    }

    #[test]
    fn the_drop_never_empties_a_cell_however_tight_the_window() {
        let slack = window_of(
            Bounds::with_ceiling(Ceiling::AboveOptimum(Energy(0))),
            Energy(-1000),
        );
        let mut cell = front(&[-400, -399, -1]);
        assert_eq!(retain_feasible(&mut cell, slack), 2);
        assert_eq!(cell.len(), 1);
        assert_eq!(cell.points()[0].mfe, Energy(-400));
    }

    #[test]
    fn applying_the_window_early_and_again_later_lands_where_applying_it_once_does() {
        // Merge order must not change what survives, which is what lets the
        // drop run at a partially accumulated cell and under any worker count.
        let slack = window_of(
            Bounds::with_ceiling(Ceiling::AboveOptimum(Energy(200))),
            Energy(-1000),
        );

        let mut early = front(&[-300, -250, -150, -50]);
        retain_feasible(&mut early, slack);
        let mut merged = Front::from_points(
            early
                .points()
                .iter()
                .map(|p| crate::front::Point::new(p.mfe, p.cai, ()))
                .chain([crate::front::Point::new(Energy(-500), -1.0, ())])
                .collect(),
        );
        retain_feasible(&mut merged, slack);

        let mut once = front(&[-500, -300, -250, -150, -50]);
        retain_feasible(&mut once, slack);

        let left: Vec<i32> = merged.points().iter().map(|p| p.mfe.0).collect();
        let right: Vec<i32> = once.points().iter().map(|p| p.mfe.0).collect();
        assert_eq!(left, right);
    }

    #[test]
    fn a_ceiling_under_the_achievable_minimum_is_infeasible_and_says_what_is_reachable() {
        let err = Bounds::with_ceiling(Ceiling::Absolute(Energy(-1200)))
            .resolve(Energy(-1000))
            .unwrap_err();
        assert_eq!(
            err,
            Unreachable {
                ceiling: Energy(-1200),
                achievable: Energy(-1000),
            }
        );
    }

    #[test]
    fn an_offset_scoped_beyond_the_exterior_leaves_the_exterior_window_modular() {
        let windows = Bounds::with_ceiling(Ceiling::AboveOptimum(Energy(300)))
            .with_credit_offset_beyond_exterior(Energy(7))
            .resolve(Energy(-1000))
            .unwrap();
        assert_eq!(windows.exterior.window(), Some(Energy(300)));
        assert_eq!(windows.beyond_exterior.window(), Some(Energy(293)));
    }

    #[test]
    fn a_credit_offset_narrows_the_window_without_moving_the_feasibility_verdict() {
        let bounds =
            Bounds::with_ceiling(Ceiling::AboveOptimum(Energy(300))).with_credit_offset(Energy(1));
        let slack = window_of(bounds, Energy(-1000));
        assert_eq!(slack.window(), Some(Energy(299)));

        // The offset is inadmissible by exactly one unit: a point sitting on the
        // unmutated boundary is dropped, and nothing else is.
        let mut cell = front(&[-1000, -701, -700, -699]);
        assert_eq!(retain_feasible(&mut cell, slack), 2);
        assert_eq!(cell.len(), 2);
    }
}
