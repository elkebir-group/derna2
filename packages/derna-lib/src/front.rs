//! Pareto-front primitives for the front-valued algebraic DP.
//!
//! Where the scalar scan keeps one BRD-optimal [`crate::state::State`] per
//! table cell, the front-valued DP keeps a [`Front`]: the dominance-reduced
//! staircase of every non-dominated `(mfe, cai)` trade-off reaching that cell,
//! each with its own provenance handle. The two scalar cell operators lift to
//! two front operators:
//!
//! - **`⊕` — [`Front::merge`]** combines alternative derivations of one cell
//!   (in place of scalar `min` by BRD): union two staircases, drop every
//!   dominated point.
//! - **`⊗` — [`Front::compose`]** composes two sub-derivations into a larger
//!   structure (in place of scalar `+`): the Minkowski sum of two fronts
//!   shifted by a fixed loop `Δ`, then dominance-eliminated. Its cheap special
//!   case, [`Front::shift`], slides a whole front by one fixed `Δ` (the
//!   unpaired-extension case) in `O(k̄)` with no size growth.
//!
//! Dominance minimizes `mfe` (integer dekacalories, lower is better) and
//! maximizes `cai` (accumulated log-CAI, higher is better), matching the scalar
//! path's [`crate::state::State`] units and its fixed-λ objective semantics:
//! point `A` dominates `B` iff `A.mfe ≤ B.mfe && A.cai ≥ B.cai` with at least
//! one inequality strict. A reduced front is stored sorted by `mfe` ascending,
//! which after elimination forces `cai` strictly ascending — a staircase.
//!
//! Because `mfe`/`cai` are both additive over a parse and `⊕`/`⊗` are monotone,
//! a dominated sub-derivation can never be part of a non-dominated whole, so the
//! per-cell elimination is exact, not lossy (the multi-objective Bellman
//! principle). The `back` slot is per front entry, so recovering one Pareto
//! point walks that point's own provenance.

use std::collections::BinaryHeap;

use rna_mfe::Energy;

/// Log-CAI dominance tolerance. `cai` is an `f64` sum, and additive
/// accumulation is not associative, so two derivations of the *same* underlying
/// trade-off — the same codon multiset reached along different DP paths — can
/// carry log-CAI values that disagree at the ULP scale (`~1e-12` over a
/// protein). Comparing `cai` with strict `>=` would let the worse-MFE twin of
/// such a pair spuriously survive as "non-dominated," inflating the front with
/// float artifacts. Treating two `cai` values within this tolerance as equal
/// collapses those twins (keeping the lower-MFE one) while leaving genuinely
/// distinct trade-offs — which differ by `>= ~1e-3`, the granularity of the
/// per-codon adaptation weights — untouched. It sits three orders of magnitude
/// above the accumulation noise and four below any real CAI gap, so the front
/// is well-defined modulo float noise and no genuine Pareto point is lost.
const CAI_DOMINANCE_EPS: f64 = 1e-9;

/// One Pareto point on a cell's front: an `(mfe, cai)` trade-off and the
/// provenance handle that recovers the sub-derivation reaching it.
///
/// `B` is the back-pointer type, left generic so the front primitives stay
/// independent of how a particular DP records provenance (a lightweight index,
/// or a full traceback manner). Dominance reads only `mfe` and `cai`; `back`
/// rides along untouched.
#[derive(Debug, Clone, Copy)]
pub struct Point<B> {
    /// Accumulated free energy (integer dekacalories; lower is better).
    pub mfe: Energy,
    /// Accumulated log-CAI (natural log; higher is better).
    pub cai: f64,
    /// How this point was derived — the per-point traceback handle.
    pub back: B,
}

impl<B> Point<B> {
    /// Construct a point with the given accumulators and back-pointer.
    pub fn new(mfe: Energy, cai: f64, back: B) -> Self {
        Self { mfe, cai, back }
    }

    /// Whether `self` weakly dominates `other`: no worse on either objective
    /// (lower-or-equal `mfe`, and `cai` no lower within [`CAI_DOMINANCE_EPS`]).
    /// Strict domination is this plus at least one strict inequality; the
    /// staircase construction never needs the strict form separately, since ties
    /// — including the ULP-scale `cai` twins the tolerance absorbs — are resolved
    /// by keeping the first-seen (lower-`mfe`) point.
    fn weakly_dominates<C>(&self, other: &Point<C>) -> bool {
        self.mfe.0 <= other.mfe.0 && self.cai >= other.cai - CAI_DOMINANCE_EPS
    }
}

/// An opt-in front-pruning policy — the lever that caps per-cell front size
/// `k̄` so the quadratic Minkowski factor stops compounding at scale. The exact
/// front-DP uses [`Prune::None`] (the default; nothing is dropped, so the fold
/// stays byte-identical to the un-pruned front); the approximate front-DP
/// selects a lossy mode and carries the bound that mode guarantees.
///
/// Both lossy modes preserve the staircase (still `mfe`-ascending / `cai`-
/// ascending) and are **monotone**: a point they drop is one an already-kept
/// point approximately dominates, so extending both by the same suffix keeps the
/// approximation — the loss composes predictably over a derivation rather than
/// drifting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Prune {
    /// No pruning — the exact front. A genuine no-op: [`Front::prune`] returns
    /// the front untouched, so the exact front-DP is byte-identical whether or
    /// not it routes through the prune call.
    None,
    /// FPTAS ε-dominance grid on the MFE axis (Papadimitriou–Yannakakis): bucket
    /// the staircase into geometric intervals of ratio `1 + delta` in the
    /// energy-magnitude offset from the front's minimum MFE, and keep the
    /// maximum-CAI point per bucket (plus the minimum-MFE endpoint). The kept
    /// point of a bucket approximately dominates every exact point in it — no
    /// worse in CAI, and within a `(1 + delta)` factor in energy-magnitude
    /// offset — so one prune loses at most a `(1 + delta)` relative-MFE factor;
    /// over a derivation of depth `d` the recovered front is a
    /// `(1 + delta)^d`-approximation. Per-cell `k̄` is capped at
    /// `ceil(log_{1+delta}(mfe span)) + 1`, which grows only logarithmically in
    /// the span — the memory-provable route. `delta = 0` is exact.
    EpsGridMfe { delta: f64 },
    /// Hard front-size cap: keep at most `max` points, always retaining the two
    /// endpoints (min MFE, max CAI) and spreading the rest evenly along the
    /// staircase. Per-cell `k̄` is a constant `max` independent of `n`, so memory
    /// is `O(n² · max)` regardless of span — the memory-viable route when the
    /// FPTAS grid's logarithmically-growing bucket count still busts budget. The
    /// loss is a per-cell spread/window bound (the largest CAI gap between kept
    /// neighbours), not a provable global `(1 + ε)`.
    Cap { max: usize },
    /// ε-dominance grid under a hard ceiling on the kept count: bucket as
    /// [`Prune::EpsGridMfe`] does, but widen the grid until at most `max` points
    /// survive.
    ///
    /// The two lossy modes above fail in opposite ways. The grid's error is
    /// *composable* — a kept point ε-dominates everything it replaces, so the
    /// loss accumulates predictably over a derivation — but its bucket count
    /// grows with the MFE span, so it does not bound `k̄` and buys no change in
    /// the scaling exponent. The cap bounds `k̄`, and so does change the
    /// exponent, but selects by staircase *index*, which carries only a per-cell
    /// spread bound and does not compose: the same cap applied per cell through
    /// a fold loses far more than one application to the finished front.
    ///
    /// This mode takes the grid's selection and the cap's guarantee. Below the
    /// ceiling it is a no-op, so a cell keeps every point the ceiling allows;
    /// above it, `delta` is coarsened — never abandoned for index spacing — so
    /// the survivors are still bucket representatives that ε-dominate what they
    /// replace. The price is that the effective ratio is whatever the ceiling
    /// forced, so the `(1 + δ)^d` guarantee holds for that coarser value rather
    /// than the requested one; `delta` sets the floor, `max` the ceiling.
    EpsGridCapped { delta: f64, max: usize },
}

impl Prune {
    /// Whether this policy drops nothing — the exact front. A ceiling can always
    /// bind, so [`Prune::EpsGridCapped`] is never exact whatever its `delta`.
    pub fn is_exact(&self) -> bool {
        matches!(self, Prune::None) || matches!(self, Prune::EpsGridMfe { delta } if *delta <= 0.0)
    }
}

/// A dominance-reduced staircase of Pareto points, stored sorted by `mfe`
/// ascending (hence `cai` strictly ascending). The empty front is the identity
/// for [`Front::merge`].
#[derive(Debug, Clone)]
pub struct Front<B> {
    points: Vec<Point<B>>,
}

impl<B> Default for Front<B> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<B> Front<B> {
    /// The empty front — no derivation reaches the cell (the `⊕` identity).
    pub fn empty() -> Self {
        Self { points: Vec::new() }
    }

    /// A front holding one point.
    pub fn singleton(mfe: Energy, cai: f64, back: B) -> Self {
        Self {
            points: vec![Point::new(mfe, cai, back)],
        }
    }

    /// The staircase points, sorted by `mfe` ascending / `cai` ascending.
    pub fn points(&self) -> &[Point<B>] {
        &self.points
    }

    /// Number of Pareto points on the front — the per-cell `k̄` that governs the
    /// front fold's cost.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the front is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Iterate the staircase points in `mfe`-ascending order.
    pub fn iter(&self) -> impl Iterator<Item = &Point<B>> {
        self.points.iter()
    }

    /// The staircase's accumulated-CAI extent, `(lowest, highest)`, or `None`
    /// when the front is empty. Because the staircase is `cai`-ascending these
    /// are its first and last points, so the call is `O(1)`.
    ///
    /// Worth consulting before reading anything into an empty
    /// [`Front::slice_cai`]: a band whose upper bound lies below this range
    /// excludes the whole front, which is a different statement from "no
    /// design achieves that CAI". The front's lowest-CAI point is the
    /// unconstrained MFE optimum, and designs below it exist but are dominated,
    /// so no Pareto front can contain them.
    pub fn cai_range(&self) -> Option<(f64, f64)> {
        Some((self.points.first()?.cai, self.points.last()?.cai))
    }

    /// Drop every point for which `keep` returns `false`, in place.
    ///
    /// Filtering a staircase preserves the staircase: removing points cannot
    /// disturb the `mfe`-ascending order of the survivors, and cannot make a
    /// survivor dominated by another survivor that did not already dominate it.
    /// So the result needs neither a re-sort nor an elimination sweep — unlike
    /// [`Front::from_points`], which must assume an arbitrary bag. `O(|self|)`
    /// comparisons and no allocation.
    pub fn retain(&mut self, keep: impl FnMut(&Point<B>) -> bool) {
        self.points.retain(keep);
    }

    /// Keep the points for which `keep` holds, in order, letting it rewrite
    /// each kept point's back-pointer, and release the room the dropped
    /// points held. Energies and adaptations are not touched, so the
    /// staircase stays one, as under [`Front::retain`].
    pub fn retain_backs(&mut self, mut keep: impl FnMut(&mut B) -> bool) {
        self.points.retain_mut(|p| keep(&mut p.back));
        self.points.shrink_to_fit();
    }

    /// Drop every point above `cutoff`, keeping the `mfe`-ascending prefix at or
    /// below it, and return how many were removed.
    ///
    /// The cheapest filter the staircase admits, and the reason a threshold on
    /// `mfe` is worth applying inside a fold rather than once at the end. Because
    /// the points are stored `mfe`-ascending and the test is monotone in `mfe`,
    /// the survivors are a contiguous prefix — so the boundary is one binary
    /// search and the removal is a truncation, with no per-point scan (unlike
    /// [`Front::retain`]), no elimination sweep and no reallocation.
    pub fn truncate_mfe(&mut self, cutoff: Energy) -> usize {
        let keep = self.points.partition_point(|p| p.mfe.0 <= cutoff.0);
        let removed = self.points.len() - keep;
        self.points.truncate(keep);
        removed
    }

    /// `⊗` with a per-point back rewrite: slide the whole front by one fixed
    /// `Δ = (dmfe, dcai)` and replace each point's back with `f(point)`. Like
    /// [`Front::shift`] the uniform shift is monotone and so preserves the
    /// staircase order and size (`O(k̄)`, no elimination); unlike it, the back
    /// may change value or type, so a DP step can record the provenance the
    /// shift introduces (the closed pair a `P` cell wraps, the nucleotide an
    /// exterior extension spells). With `Δ = 0` it is the identity-shift back
    /// remap used to promote a compose's surviving pairs into real
    /// back-pointers. `f` is applied once per surviving point, never per
    /// eliminated candidate.
    pub fn shift_map<B2>(
        &self,
        dmfe: Energy,
        dcai: f64,
        mut f: impl FnMut(&Point<B>) -> B2,
    ) -> Front<B2> {
        Front {
            points: self
                .points
                .iter()
                .map(|p| Point::new(p.mfe + dmfe, p.cai + dcai, f(p)))
                .collect(),
        }
    }
}

impl<B: Copy> Front<B> {
    /// Build a reduced front from an arbitrary bag of points: sort by `mfe`
    /// ascending / `cai` descending, then sweep out every dominated point. Ties
    /// on both objectives keep the first point in the (stable) sort order, so
    /// the result is deterministic given the input order.
    pub fn from_points(mut points: Vec<Point<B>>) -> Self {
        sort_for_sweep(&mut points);
        Self {
            points: eliminate_sorted(points),
        }
    }

    /// The contiguous run of the staircase whose accumulated CAI lies in
    /// `[lo, hi]` inclusive — the `CAI ∈ [c₁, c₂]` query, answered over a
    /// completed front rather than as a constraint inside the fold.
    ///
    /// **Units.** `lo` and `hi` are *accumulated* log-CAI, the same quantity
    /// [`Point::cai`] carries, not a CAI value. A caller holding a CAI
    /// threshold `τ` over `m` codons converts it the way the scalar budget
    /// does: `m * τ.ln()`. Passing a raw `τ` compiles and silently selects the
    /// wrong band.
    ///
    /// Both bounds are monotone in `cai` on a staircase, so the survivors are a
    /// contiguous run located by two binary searches: `O(log |self|)` plus the
    /// copy. The result is still sorted and still mutually non-dominated, so it
    /// needs no re-reduction. `lo > hi` yields the empty front.
    ///
    /// Slicing is exact and order-preserving, which is what makes it composable
    /// for partitioning — see [`Front::partition_cai`].
    pub fn slice_cai(&self, lo: f64, hi: f64) -> Front<B> {
        let first = self.points.partition_point(|p| p.cai < lo);
        let end = self.points.partition_point(|p| p.cai <= hi);
        Front {
            points: if first < end {
                self.points[first..end].to_vec()
            } else {
                Vec::new()
            },
        }
    }

    /// Cut the staircase into `cuts.len() + 1` consecutive bands at the given
    /// **ascending** accumulated-log-CAI boundaries, each cut point falling in
    /// the band above it.
    ///
    /// The bands are disjoint and their concatenation is the original front, so
    /// this partitions rather than samples — the shape wanted for fanning a
    /// front out across workers, or for taking one representative per CAI band.
    /// An empty band is returned as an empty front rather than skipped, so the
    /// result always lines up positionally with `cuts`.
    ///
    /// Same units as [`Front::slice_cai`]. `cuts` must be ascending; a
    /// `debug_assert` catches the mistake in test builds, and out-of-order cuts
    /// otherwise yield empty bands rather than overlapping ones.
    pub fn partition_cai(&self, cuts: &[f64]) -> Vec<Front<B>> {
        debug_assert!(
            cuts.windows(2).all(|w| w[0] <= w[1]),
            "partition_cai expects ascending cuts, got {cuts:?}"
        );
        let mut out = Vec::with_capacity(cuts.len() + 1);
        let mut lo = f64::NEG_INFINITY;
        for &cut in cuts {
            let end = self.points.partition_point(|p| p.cai < cut);
            let first = self.points.partition_point(|p| p.cai < lo);
            out.push(Front {
                points: if first < end {
                    self.points[first..end].to_vec()
                } else {
                    Vec::new()
                },
            });
            lo = cut;
        }
        let first = self.points.partition_point(|p| p.cai < lo);
        out.push(Front {
            points: self.points[first..].to_vec(),
        });
        out
    }

    /// Apply a [`Prune`] policy, returning a pruned copy of the staircase (still
    /// `mfe`-ascending / `cai`-ascending). [`Prune::None`] and a front already at
    /// or below a [`Prune::Cap`] limit return a cheap clone; the lossy modes
    /// select a subset that approximately dominates the whole, per the policy's
    /// documented bound.
    pub fn prune(&self, policy: Prune) -> Front<B> {
        match policy {
            Prune::None => self.clone(),
            _ if self.points.len() <= 2 => self.clone(),
            Prune::EpsGridMfe { delta } if delta <= 0.0 => self.clone(),
            Prune::EpsGridMfe { delta } => Front {
                points: self.eps_grid_mfe(delta),
            },
            Prune::Cap { max } if self.points.len() <= max.max(2) => self.clone(),
            Prune::Cap { max } => Front {
                points: self.cap(max.max(2)),
            },
            Prune::EpsGridCapped { max, .. } if self.points.len() <= max.max(2) => self.clone(),
            Prune::EpsGridCapped { delta, max } => Front {
                points: self.eps_grid_capped(delta, max.max(2)),
            },
        }
    }

    /// Keep bucket representatives under a hard ceiling: widen the geometric
    /// grid until at most `max` survive.
    ///
    /// The starting ratio is the closed form that would put the whole
    /// energy-magnitude span in `max - 1` buckets, floored at `delta`, so the
    /// usual case needs one grid pass. Bucket boundaries land where they land,
    /// so the result can still exceed the ceiling; widening then repeats.
    /// Termination is guaranteed because a wide enough ratio collapses every
    /// point into bucket zero, leaving the two endpoints, and `max >= 2`.
    fn eps_grid_capped(&self, delta: f64, max: usize) -> Vec<Point<B>> {
        let span = (self.points[self.points.len() - 1].mfe.0 - self.points[0].mfe.0) as f64;
        let floor = ((span + 1.0).ln() / (max - 1) as f64).exp() - 1.0;
        let mut ratio = delta.max(floor).max(f64::MIN_POSITIVE);
        loop {
            let kept = self.eps_grid_mfe(ratio);
            if kept.len() <= max {
                return kept;
            }
            ratio *= 2.0;
        }
    }

    /// Keep the maximum-CAI point per geometric MFE bucket (plus the min-MFE
    /// endpoint). Points arrive `mfe`-ascending / `cai`-ascending, so a bucket is
    /// a contiguous run and its max-CAI point is the last one before the bucket
    /// index increments; the min-MFE endpoint (index 0) is retained so the
    /// energy-optimal vertex is never lost.
    fn eps_grid_mfe(&self, delta: f64) -> Vec<Point<B>> {
        let ln_ratio = (1.0 + delta).ln();
        let mfe_min = self.points[0].mfe.0 as f64;
        let bucket =
            |m: &Point<B>| ((m.mfe.0 as f64 - mfe_min + 1.0).ln() / ln_ratio).floor() as i64;
        let mut out = Vec::new();
        for (i, p) in self.points.iter().enumerate() {
            let last_of_bucket = self
                .points
                .get(i + 1)
                .is_none_or(|next| bucket(next) != bucket(p));
            if i == 0 || last_of_bucket {
                out.push(*p);
            }
        }
        out
    }

    /// Keep `max` points, retaining both endpoints and spreading the interior
    /// picks evenly across the staircase indices. Deterministic given the input.
    fn cap(&self, max: usize) -> Vec<Point<B>> {
        let n = self.points.len();
        let last = n - 1;
        let mut keep: Vec<usize> = (0..max)
            .map(|k| (k * last + (max - 1) / 2) / (max - 1))
            .collect();
        keep.dedup();
        keep.into_iter().map(|i| self.points[i]).collect()
    }

    /// `⊕` — dominance-merge two staircases into one, dropping every dominated
    /// point. `O(|self| + |other|)` on the two sorted inputs. Associative and
    /// commutative (an idempotent commutative monoid with [`Front::empty`] as
    /// identity), so alternative derivations of a cell can be accumulated in any
    /// order.
    pub fn merge(&self, other: &Front<B>) -> Front<B> {
        let (a, b) = (&self.points, &other.points);
        let mut out = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        // Merge both `mfe`-ascending staircases into one ascending stream,
        // keeping a point only when its `cai` beats the best kept so far.
        while i < a.len() || j < b.len() {
            let next = if take_left(a.get(i), b.get(j)) {
                i += 1;
                a[i - 1]
            } else {
                j += 1;
                b[j - 1]
            };
            push_if_undominated(&mut out, next);
        }
        Front { points: out }
    }

    /// Accumulate `other` into `self` via `⊕` (see [`Front::merge`]). Takes
    /// `other` by value so a merge into an empty cell — the common first write to
    /// a fresh cell — moves the incoming staircase in place of re-allocating and
    /// copying it, cutting an allocation from every cell's first accumulation.
    pub fn merge_into(&mut self, other: Front<B>) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            self.points = other.points;
            return;
        }
        *self = self.merge(&other);
    }

    /// The cheap `⊗` special case: slide the whole front by one fixed
    /// `Δ = (dmfe, dcai)`, back-pointers unchanged. A uniform shift is monotone
    /// and so preserves the staircase order and size — `O(k̄)`, no elimination.
    /// This is the unpaired-extension / stack / bulge case, and by far the most
    /// common `⊗`.
    pub fn shift(&self, dmfe: Energy, dcai: f64) -> Front<B> {
        self.shift_map(dmfe, dcai, |p| p.back)
    }

    /// `⊗` — Minkowski-compose two fronts into a larger structure: every pair
    /// `(a, b)` contributes `(a.mfe + b.mfe + dmfe, a.cai + b.cai + dcai)` with a
    /// back combined by `combine`, then the `|self|·|other|` products are
    /// dominance-eliminated. The `O(k̄²·log k̄)` step — the bifurcations and loop
    /// closures where the front size can grow. Use [`Front::shift`] for the
    /// singleton case (one fixed operand), which stays `O(k̄)`.
    pub fn compose<C: Copy, D: Copy>(
        &self,
        other: &Front<C>,
        dmfe: Energy,
        dcai: f64,
        mut combine: impl FnMut(B, C) -> D,
    ) -> Front<D> {
        let mut products = Vec::with_capacity(self.points.len() * other.points.len());
        for a in &self.points {
            for b in &other.points {
                products.push(Point::new(
                    a.mfe + b.mfe + dmfe,
                    a.cai + b.cai + dcai,
                    combine(a.back, b.back),
                ));
            }
        }
        Front::from_points(products)
    }

    /// [`Front::compose`] generating only the products with energy at most
    /// `cutoff` that `keep` admits, returning the front and how many products
    /// it generated.
    ///
    /// Both operands are `mfe`-ascending, so for each left point the right
    /// points within `cutoff` are a prefix, found by one binary search that
    /// shrinks as the left point's energy grows, and the left points past the
    /// first with an empty prefix contribute nothing. The result equals
    /// `compose`'s with the skipped products removed before the elimination.
    /// That equals `compose` followed by the caller's filter only where a
    /// skipped product can dominate nothing the filter keeps; for `cutoff` that
    /// holds by construction, since everything a product dominates has energy
    /// at least its own, and `keep` must be chosen so it holds too.
    ///
    /// With a `screen` that `keep` implies, a left point's products are tested
    /// in ascending order of the right point's [`Screen`] share and stop at the
    /// first share the screen rules out: the share is linear in a product's
    /// accumulators, so the products past it fail the screen and `keep` would
    /// reject them all. Only the order within one left point's products moves,
    /// and those never tie (their energies differ), so the stable sort keeps
    /// the result, and the count, exactly as without the screen.
    ///
    /// From [`MERGE_MIN_PRODUCTS`] products within the cutoff, the products
    /// are not built and sorted whole but merged row by row
    /// ([`Front::compose_merged`]), which keeps the same points and
    /// derivations and counts a product it drops as dominated as generated.
    #[allow(clippy::too_many_arguments)]
    pub fn compose_within<C: Copy, D: Copy>(
        &self,
        other: &Front<C>,
        dmfe: Energy,
        dcai: f64,
        cutoff: Option<Energy>,
        screen: Option<Screen>,
        mut keep: impl FnMut(Energy, f64) -> bool,
        mut combine: impl FnMut(B, C) -> D,
    ) -> (Front<D>, usize) {
        let within = self
            .prefixes_within(other, dmfe, cutoff)
            .map(|(_, end)| end)
            .sum();
        if within == 0 {
            return (Front::empty(), 0);
        }
        if within >= MERGE_MIN_PRODUCTS {
            return self.compose_merged(other, dmfe, dcai, cutoff, within, keep, combine);
        }
        let screen = screen.filter(|_| within >= SCREEN_MIN_PRODUCTS);
        let mut products = Vec::with_capacity(if screen.is_some() { 0 } else { within });
        let mut take = |a: &Point<B>, b: &Point<C>| {
            let (mfe, cai) = (a.mfe + b.mfe + dmfe, a.cai + b.cai + dcai);
            if keep(mfe, cai) {
                products.push(Point::new(mfe, cai, combine(a.back, b.back)));
            }
        };
        match screen {
            Some(screen) => {
                let longest = self
                    .prefixes_within(other, dmfe, cutoff)
                    .map(|(_, end)| end)
                    .max()
                    .unwrap_or(0);
                let fixed = screen.share(dmfe, dcai);
                let mut order: Vec<(f64, usize)> = other.points[..longest]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| (screen.share(b.mfe, b.cai) + fixed, j))
                    .collect();
                order.sort_by(|x, y| x.0.total_cmp(&y.0));
                for (a, end) in self.prefixes_within(other, dmfe, cutoff) {
                    let room = screen.limit - screen.share(a.mfe, a.cai);
                    for &(share, j) in &order {
                        if share > room {
                            break;
                        }
                        if j < end {
                            take(a, &other.points[j]);
                        }
                    }
                }
            }
            None => {
                for (a, end) in self.prefixes_within(other, dmfe, cutoff) {
                    for b in &other.points[..end] {
                        take(a, b);
                    }
                }
            }
        }
        let generated = products.len();
        (Front::from_points(products), generated)
    }

    /// [`Front::compose_within`] by merging the rows instead of building and
    /// sorting every product, for `within` products within the cutoff.
    ///
    /// Each left point's row of products is `mfe`-ascending and
    /// `cai`-ascending, so the rows are merged in the elimination's own order
    /// (`mfe` ascending, `cai` descending, then row-major, which is the stable
    /// sort's order) and each product is checked against the last one kept as
    /// it is reached. A product that point dominates is dropped unbuilt, and
    /// so is the rest of its row up to the first product whose `cai` clears
    /// it, found by one binary search: every later kept point has `cai` at
    /// least as high and energy no higher, so it dominates them too. `keep` is
    /// asked only about products not already dominated, and a product dropped
    /// as dominated counts as generated, as the whole compose would have built
    /// it and eliminated it.
    #[allow(clippy::too_many_arguments)]
    fn compose_merged<C: Copy, D: Copy>(
        &self,
        other: &Front<C>,
        dmfe: Energy,
        dcai: f64,
        cutoff: Option<Energy>,
        within: usize,
        mut keep: impl FnMut(Energy, f64) -> bool,
        mut combine: impl FnMut(B, C) -> D,
    ) -> (Front<D>, usize) {
        let rows: Vec<(&Point<B>, usize)> = self.prefixes_within(other, dmfe, cutoff).collect();
        let product = |row: usize, j: usize| {
            let (a, b) = (rows[row].0, &other.points[j]);
            (a.mfe + b.mfe + dmfe, a.cai + b.cai + dcai)
        };
        let mut heads: BinaryHeap<Head> = (0..rows.len())
            .map(|row| Head::new(product(row, 0), row, 0))
            .collect();
        let mut out: Vec<Point<D>> = Vec::new();
        let mut rejected = 0;
        while let Some(Head { mfe, cai, row, j }) = heads.pop() {
            let end = rows[row].1;
            let next = match out.last() {
                Some(last) if last.mfe.0 <= mfe.0 && last.cai >= cai - CAI_DOMINANCE_EPS => {
                    let a = rows[row].0;
                    let floor = last.cai;
                    j + 1
                        + other.points[j + 1..end]
                            .partition_point(|b| floor >= a.cai + b.cai + dcai - CAI_DOMINANCE_EPS)
                }
                _ => {
                    if keep(mfe, cai) {
                        let back = combine(rows[row].0.back, other.points[j].back);
                        out.push(Point::new(mfe, cai, back));
                    } else {
                        rejected += 1;
                    }
                    j + 1
                }
            };
            if next < end {
                heads.push(Head::new(product(row, next), row, next));
            }
        }
        (Front { points: out }, within - rejected)
    }

    /// Each left point with the length of `other`'s prefix whose products
    /// with it, shifted by `dmfe`, are within `cutoff`, stopping at the first
    /// empty prefix; every left point with the whole of `other` without one.
    fn prefixes_within<'s, C>(
        &'s self,
        other: &'s Front<C>,
        dmfe: Energy,
        cutoff: Option<Energy>,
    ) -> impl Iterator<Item = (&'s Point<B>, usize)> + 's {
        let mut end = other.points.len();
        self.points
            .iter()
            .map(move |a| {
                if let Some(cutoff) = cutoff {
                    let base = a.mfe + dmfe;
                    end = other.points[..end].partition_point(|b| (base + b.mfe).0 <= cutoff.0);
                }
                (a, end)
            })
            .take_while(|&(_, end)| end > 0)
    }
}

/// A linear test every product a compose's `keep` admits passes:
/// `lambda·mfe − kk·cai ≤ limit`. [`Front::compose_within`] reads it to skip
/// products without asking `keep`; it may admit more than `keep` does, never
/// less.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Screen {
    /// The weight on energy, per the fold's integer unit.
    pub lambda: f64,
    /// The weight on accumulated log-CAI.
    pub kk: f64,
    /// The largest value a product `keep` admits can have.
    pub limit: f64,
}

/// What [`Screen`]'s share may lose to rounding when a product's is taken as
/// the sum of its two operands' and the shift's: far above the few ulps the
/// regrouping of `lambda·mfe − kk·cai` costs at the fold's magnitudes, so a
/// product the screen skips is one `keep` rejects.
const SCREEN_ROUNDING: f64 = 1e-6;

/// The fewest products within the cutoff at which [`Front::compose_within`]
/// orders a compose by its [`Screen`]; below it the order costs more than the
/// tests it saves.
const SCREEN_MIN_PRODUCTS: usize = 64;

/// The fewest products within the cutoff at which [`Front::compose_within`]
/// merges a compose's rows instead of building and sorting every product.
/// Merging every compose cost 6.8 % more on a whole n = 500 walk, where
/// composes are small; a large compose is where building products that the
/// elimination then drops costs most.
const MERGE_MIN_PRODUCTS: usize = 1 << 16;

/// The next product of one row of a compose, ordered so a max-heap pops it in
/// the elimination's order: lower `mfe` first, then higher `cai`, then the
/// earlier row and the earlier product in it.
struct Head {
    mfe: Energy,
    cai: f64,
    row: usize,
    j: usize,
}

impl Head {
    fn new((mfe, cai): (Energy, f64), row: usize, j: usize) -> Self {
        Self { mfe, cai, row, j }
    }
}

impl PartialEq for Head {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Head {}

impl PartialOrd for Head {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Head {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .mfe
            .0
            .cmp(&self.mfe.0)
            .then_with(|| self.cai.total_cmp(&other.cai))
            .then_with(|| other.row.cmp(&self.row))
            .then_with(|| other.j.cmp(&self.j))
    }
}

impl Screen {
    /// The screen at `lambda` and `kk` admitting values up to `limit`, with
    /// [`SCREEN_ROUNDING`] more room.
    pub fn new(lambda: f64, kk: f64, limit: f64) -> Self {
        Self {
            lambda,
            kk,
            limit: limit + SCREEN_ROUNDING,
        }
    }

    /// One operand's share of a product's value.
    fn share(&self, mfe: Energy, cai: f64) -> f64 {
        self.lambda * f64::from(mfe.0) - self.kk * cai
    }
}

/// Whether the left staircase yields the next point in the `⊕` merge order:
/// smaller `mfe` first, and on an `mfe` tie the larger `cai` first (so the
/// dominating point is emitted before the one it dominates and the elimination
/// sweep drops the loser). An exhausted left side never yields; an exhausted
/// right side always does. The caller's loop constraint rules out both being empty.
fn take_left<B, C>(left: Option<&Point<B>>, right: Option<&Point<C>>) -> bool {
    match (left, right) {
        (Some(l), Some(r)) => (l.mfe.0, r.cai).le(&(r.mfe.0, l.cai)),
        (Some(_), None) => true,
        _ => false,
    }
}

/// Sort a bag of points the way [`Front::from_points`] does before its sweep:
/// `mfe` ascending, then `cai` descending, stable.
fn sort_for_sweep<B>(points: &mut [Point<B>]) {
    points.sort_by(|a, b| a.mfe.0.cmp(&b.mfe.0).then_with(|| b.cai.total_cmp(&a.cai)));
}

/// The staircase [`Front::from_points`] builds from `points`, written into
/// `out` in place of a new front, so that a caller building many small fronts
/// in turn reuses one buffer. `points` is left sorted.
pub(crate) fn staircase_into<B: Copy>(points: &mut [Point<B>], out: &mut Vec<Point<B>>) {
    sort_for_sweep(points);
    out.clear();
    for &p in points.iter() {
        push_if_undominated(out, p);
    }
}

/// Sweep an `(mfe asc, cai desc)`-sorted bag into a reduced staircase, keeping a
/// point only when its `cai` strictly exceeds the best `cai` kept so far. Points
/// tied on both objectives are dropped after the first (the earlier one in sort
/// order weakly dominates them).
fn eliminate_sorted<B: Copy>(sorted: Vec<Point<B>>) -> Vec<Point<B>> {
    let mut out: Vec<Point<B>> = Vec::with_capacity(sorted.len());
    for p in sorted {
        push_if_undominated(&mut out, p);
    }
    // A staircase is often a small part of its candidates, and a stored
    // front should not hold their room.
    out.shrink_to_fit();
    out
}

/// Push `p` onto an `mfe`-ascending staircase iff it is not dominated by the
/// last kept point. Because points arrive with `mfe` non-decreasing, the only
/// candidate dominator already kept is the last one: `p` survives exactly when
/// its `cai` strictly exceeds every kept point's, i.e. the running maximum,
/// which the last kept point holds.
fn push_if_undominated<B: Copy>(out: &mut Vec<Point<B>>, p: Point<B>) {
    match out.last() {
        Some(last) if last.weakly_dominates(&p) => {}
        _ => out.push(p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point with a `u32` back tag, so tests can assert which derivation
    /// survived a merge / compose as well as its `(mfe, cai)`.
    fn pt(mfe: i32, cai: f64, back: u32) -> Point<u32> {
        Point::new(Energy(mfe), cai, back)
    }

    /// The `(mfe, cai)` staircase of a front, dropping back tags — the shape
    /// most assertions compare against.
    fn shape<B>(f: &Front<B>) -> Vec<(i32, f64)> {
        f.iter().map(|p| (p.mfe.0, p.cai)).collect()
    }

    /// Assert a front's staircase equals `expected`: `mfe` exactly, `cai` within
    /// float-addition slop (composed `cai` values are sums, so `0.4 + 0.2` is not
    /// bit-equal to the literal `0.6`).
    fn assert_shape<B>(f: &Front<B>, expected: &[(i32, f64)]) {
        let got = shape(f);
        assert_eq!(got.len(), expected.len(), "front size: got {got:?}");
        for (g, e) in got.iter().zip(expected) {
            assert_eq!(g.0, e.0, "mfe: got {got:?} expected {expected:?}");
            assert!(
                (g.1 - e.1).abs() < 1e-9,
                "cai: got {got:?} expected {expected:?}"
            );
        }
    }

    /// A reduced front is sorted by `mfe` strictly ascending and `cai` strictly
    /// ascending — the staircase invariant every operator must preserve.
    fn assert_staircase<B: std::fmt::Debug>(f: &Front<B>) {
        for w in f.points().windows(2) {
            assert!(w[0].mfe.0 < w[1].mfe.0, "mfe not strictly ascending: {f:?}");
            assert!(w[0].cai < w[1].cai, "cai not strictly ascending: {f:?}");
        }
    }

    /// A compose ordered by a screen keeps exactly the points, the
    /// derivations and the count of the compose that tests every product —
    /// on long staircases with exact and near ties in `cai`, under cutoffs,
    /// with a filter that implies the screen and rejects more — and the screen
    /// spares the filter some products.
    #[test]
    fn a_screened_compose_equals_the_tested_one() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let (mut asked, mut within) = (0usize, 0usize);
        for case in 0..400u32 {
            // Monotone walks with zero steps (exact ties) and 1e-10 nudges
            // (near ties), so the staircases are long.
            let mut front = |tag: u32| {
                let size = 1 + next(40);
                let (mut mfe, mut cai) = (next(20) as i32 - 30, 0.0);
                let points = (0..size)
                    .map(|i| {
                        mfe += next(4) as i32;
                        cai += next(3) as f64 * 0.125;
                        let near = if next(4) == 0 { 1e-10 } else { 0.0 };
                        pt(mfe, cai + near, tag + i as u32)
                    })
                    .collect();
                Front::from_points(points)
            };
            let left = front(0);
            let right = front(1000);
            let cutoff = (case % 3 == 0).then(|| Energy(next(60) as i32 - 10));
            let dmfe = Energy(next(5) as i32 - 2);
            let dcai = [0.0, 0.1, -0.375][(case % 3) as usize];
            let lambda = 0.05 + next(90) as f64 / 100.0;
            let kk = 100.0 * (1.0 - lambda);
            let limit = next(400) as f64 - 300.0;
            let value = |mfe: Energy, cai: f64| lambda * f64::from(mfe.0) - kk * cai;
            let filtered = case % 2 == 1;
            let keep = |mfe: Energy, cai: f64| {
                value(mfe, cai) <= limit && (!filtered || (mfe.0 + (cai * 8.0) as i32) % 3 != 0)
            };
            let tested = left.compose_within(&right, dmfe, dcai, cutoff, None, keep, |a, b| (a, b));
            let calls = std::cell::Cell::new(0usize);
            let screened = left.compose_within(
                &right,
                dmfe,
                dcai,
                cutoff,
                Some(Screen::new(lambda, kk, limit)),
                |mfe, cai| {
                    calls.set(calls.get() + 1);
                    keep(mfe, cai)
                },
                |a, b| (a, b),
            );
            let bits = |f: &Front<(u32, u32)>| {
                f.iter()
                    .map(|p| (p.mfe.0, p.cai.to_bits(), p.back))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&screened.0), bits(&tested.0), "case {case}");
            assert_eq!(screened.1, tested.1, "case {case}");
            assert_staircase(&screened.0);
            asked += calls.get();
            within += left
                .iter()
                .flat_map(|a| right.iter().map(move |b| a.mfe + b.mfe + dmfe))
                .filter(|m| cutoff.is_none_or(|c| m.0 <= c.0))
                .count();
        }
        // Without the screen the filter is asked about every product within
        // the cutoff.
        assert!(
            asked < within,
            "the screen spared the filter nothing of {within}"
        );
    }

    /// `compose_within` as the whole compose computes it: every product within
    /// the cutoff that `keep` admits built, stably sorted and eliminated.
    fn compose_within_whole(
        left: &Front<u32>,
        right: &Front<u32>,
        dmfe: Energy,
        dcai: f64,
        cutoff: Option<Energy>,
        keep: impl Fn(Energy, f64) -> bool,
    ) -> (Front<(u32, u32)>, usize) {
        let mut products = Vec::new();
        for a in left.iter() {
            for b in right.iter() {
                let (mfe, cai) = (a.mfe + b.mfe + dmfe, a.cai + b.cai + dcai);
                if cutoff.is_none_or(|c| mfe.0 <= c.0) && keep(mfe, cai) {
                    products.push(Point::new(mfe, cai, (a.back, b.back)));
                }
            }
        }
        let generated = products.len();
        (Front::from_points(products), generated)
    }

    /// The merged compose keeps exactly the points, and the derivations, that
    /// building every product and eliminating them keeps — on fronts with
    /// exact and near ties in `cai`, under cutoffs and a filter — and it drops
    /// dominated products without asking the filter about them.
    #[test]
    fn a_merged_compose_equals_the_whole_compose() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let (mut asked, mut within_all) = (0usize, 0usize);
        for case in 0..400u32 {
            let mut front = |tag: u32, most: u64| {
                let size = 1 + next(most);
                let points = (0..size)
                    .map(|i| {
                        let near = if next(4) == 0 { 1e-10 } else { 0.0 };
                        pt(
                            next(60) as i32 - 30,
                            next(12) as f64 * 0.125 + near,
                            tag + i as u32,
                        )
                    })
                    .collect();
                Front::from_points(points)
            };
            let left = front(0, 30);
            let right = front(1000, 30);
            let cutoff = (case % 3 == 0).then(|| Energy(next(40) as i32 - 20));
            let dmfe = Energy(next(5) as i32 - 2);
            let dcai = [0.0, 0.1, -0.375][(case % 3) as usize];
            let filtered = case % 2 == 1;
            let keep = |mfe: Energy, cai: f64| !filtered || (mfe.0 + (cai * 8.0) as i32) % 3 != 0;
            let (whole, whole_generated) =
                compose_within_whole(&left, &right, dmfe, dcai, cutoff, keep);
            let within: usize = left
                .prefixes_within(&right, dmfe, cutoff)
                .map(|(_, end)| end)
                .sum();
            if within == 0 {
                assert!(whole.is_empty(), "case {case}");
                continue;
            }
            let calls = std::cell::Cell::new(0usize);
            let (merged, generated) = left.compose_merged(
                &right,
                dmfe,
                dcai,
                cutoff,
                within,
                |mfe, cai| {
                    calls.set(calls.get() + 1);
                    keep(mfe, cai)
                },
                |a, b| (a, b),
            );
            let bits = |f: &Front<(u32, u32)>| {
                f.iter()
                    .map(|p| (p.mfe.0, p.cai.to_bits(), p.back))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&merged), bits(&whole), "case {case}");
            assert_staircase(&merged);
            if filtered {
                assert!(generated >= whole_generated, "case {case}");
            } else {
                assert_eq!(generated, whole_generated, "case {case}");
            }
            assert!(generated <= within, "case {case}");
            asked += calls.get();
            within_all += within;
        }
        // The whole compose asks the filter about every product within the
        // cutoff; the merge must have dropped some unasked.
        assert!(
            asked < within_all,
            "the merge asked the filter about all {within_all} products"
        );
    }

    #[test]
    fn from_points_eliminates_and_sorts() {
        // (−10, 0.5) and (−5, 0.9) are non-dominated; (−5, 0.3) is dominated by
        // (−10, 0.5); (−8, 0.5) is dominated by (−10, 0.5) (equal cai, worse mfe).
        let f = Front::from_points(vec![
            pt(-5, 0.3, 0),
            pt(-10, 0.5, 1),
            pt(-8, 0.5, 2),
            pt(-5, 0.9, 3),
        ]);
        assert_eq!(shape(&f), vec![(-10, 0.5), (-5, 0.9)]);
        assert_eq!(f.iter().map(|p| p.back).collect::<Vec<_>>(), vec![1, 3]);
        assert_staircase(&f);
    }

    #[test]
    fn retain_keeps_the_staircase_without_rebuilding() {
        // A CAI floor drops a prefix of the staircase (cai is ascending), which
        // is the shape `retain` exists for: survivors stay in order and stay
        // mutually non-dominated, so no re-sort and no elimination are needed.
        let mut f = Front::from_points(vec![
            pt(-10, 0.2, 0),
            pt(-8, 0.4, 1),
            pt(-6, 0.6, 2),
            pt(-4, 0.8, 3),
        ]);
        f.retain(|p| p.cai >= 0.5);
        assert_eq!(shape(&f), vec![(-6, 0.6), (-4, 0.8)]);
        assert_eq!(f.iter().map(|p| p.back).collect::<Vec<_>>(), vec![2, 3]);
        assert_staircase(&f);

        // A non-contiguous predicate is still safe: survivors of a staircase
        // can never dominate one another.
        let mut g = Front::from_points(vec![pt(-10, 0.2, 0), pt(-8, 0.4, 1), pt(-6, 0.6, 2)]);
        g.retain(|p| p.back != 1);
        assert_eq!(shape(&g), vec![(-10, 0.2), (-6, 0.6)]);
        assert_staircase(&g);

        // Retaining nothing yields the merge identity.
        let mut h = Front::from_points(vec![pt(-10, 0.2, 0)]);
        h.retain(|_| false);
        assert!(h.is_empty());
    }

    /// A CAI band is a contiguous run of the staircase, and the run is still a
    /// staircase — the property that lets the query layer skip re-reduction.
    #[test]
    fn slice_cai_selects_the_band_and_stays_reduced() {
        let f = Front::from_points(vec![
            pt(-10, 0.2, 0),
            pt(-8, 0.4, 1),
            pt(-6, 0.6, 2),
            pt(-4, 0.8, 3),
            pt(-2, 1.0, 4),
        ]);
        let band = f.slice_cai(0.4, 0.8);
        assert_eq!(shape(&band), vec![(-8, 0.4), (-6, 0.6), (-4, 0.8)]);
        assert_staircase(&band);
        // Both bounds are inclusive.
        assert_eq!(f.slice_cai(0.6, 0.6).len(), 1);
        // A band wider than the front returns the whole front.
        assert_eq!(f.slice_cai(-1.0, 2.0).len(), f.len());
        // An inverted band is empty, not a panic and not a wrap-around.
        assert!(f.slice_cai(0.8, 0.4).is_empty());
    }

    /// A band lying wholly outside the front's CAI extent is empty, and
    /// `cai_range` is what tells a caller which side it fell off. The
    /// low-CAI case is the one that matters: the front's lowest point is the
    /// unconstrained MFE optimum, so an empty result there means "below the
    /// front", not "unachievable".
    #[test]
    fn slice_cai_outside_the_range_is_empty_and_diagnosable() {
        let f = Front::from_points(vec![pt(-10, 0.5, 0), pt(-6, 0.7, 1), pt(-2, 0.9, 2)]);
        assert_eq!(f.cai_range(), Some((0.5, 0.9)));

        let below = f.slice_cai(0.0, 0.4);
        assert!(below.is_empty());
        assert!(0.4 < f.cai_range().unwrap().0, "band sits below the front");

        let above = f.slice_cai(1.0, 1.5);
        assert!(above.is_empty());
        assert!(1.0 > f.cai_range().unwrap().1, "band sits above the front");

        assert_eq!(Front::<u32>::empty().cai_range(), None);
    }

    /// Partitioning is a partition: disjoint bands whose concatenation is the
    /// original front, positionally aligned with the cuts even where a band is
    /// empty.
    #[test]
    fn partition_cai_is_disjoint_and_covers_the_front() {
        let f = Front::from_points(vec![
            pt(-10, 0.2, 0),
            pt(-8, 0.4, 1),
            pt(-6, 0.6, 2),
            pt(-4, 0.8, 3),
        ]);
        let parts = f.partition_cai(&[0.4, 0.7]);
        assert_eq!(parts.len(), 3);
        assert_eq!(shape(&parts[0]), vec![(-10, 0.2)]);
        assert_eq!(shape(&parts[1]), vec![(-8, 0.4), (-6, 0.6)]);
        assert_eq!(shape(&parts[2]), vec![(-4, 0.8)]);
        for p in &parts {
            assert_staircase(p);
        }
        // Concatenation reconstructs the original, in order.
        let rejoined: Vec<_> = parts.iter().flat_map(shape).collect();
        assert_eq!(rejoined, shape(&f));

        // A cut outside the range yields an empty band, kept in position.
        let edge = f.partition_cai(&[0.1, 0.4]);
        assert_eq!(edge.len(), 3);
        assert!(edge[0].is_empty());
        assert_eq!(shape(&edge[1]), vec![(-10, 0.2)]);
        assert_eq!(edge[2].len(), 3);

        // No cuts is the identity partition.
        assert_eq!(shape(&f.partition_cai(&[])[0]), shape(&f));
    }

    /// The band is stated in accumulated log-CAI, which is what `Point::cai`
    /// carries; a caller converts a CAI threshold the way the scalar budget
    /// does. This pins the conversion so the units cannot drift silently.
    #[test]
    fn slice_cai_takes_accumulated_log_cai_not_a_cai_value() {
        let m = 10.0_f64;
        // Three designs at CAI 0.6 / 0.8 / 0.95 over ten codons.
        let f = Front::from_points(vec![
            pt(-30, m * 0.6_f64.ln(), 0),
            pt(-20, m * 0.8_f64.ln(), 1),
            pt(-10, m * 0.95_f64.ln(), 2),
        ]);
        let band = f.slice_cai(m * 0.7_f64.ln(), m * 0.9_f64.ln());
        assert_eq!(band.len(), 1);
        assert!((band.points()[0].cai / m).exp() - 0.8 < 1e-12);

        // Passing raw CAI instead of the accumulated log selects nothing here,
        // which is the failure this signature's doc warns about.
        assert!(f.slice_cai(0.7, 0.9).is_empty());
    }

    #[test]
    fn from_points_ties_keep_first_seen() {
        // Two points equal on both objectives: the first in input order wins.
        let f = Front::from_points(vec![pt(-7, 0.4, 10), pt(-7, 0.4, 20)]);
        assert_eq!(f.len(), 1);
        assert_eq!(f.points()[0].back, 10);
    }

    #[test]
    fn empty_is_merge_identity() {
        let f = Front::from_points(vec![pt(-10, 0.5, 0), pt(-5, 0.9, 1)]);
        assert_eq!(shape(&f.merge(&Front::empty())), shape(&f));
        assert_eq!(shape(&Front::<u32>::empty().merge(&f)), shape(&f));
        assert!(Front::<u32>::empty().merge(&Front::empty()).is_empty());
    }

    #[test]
    fn merge_is_commutative_and_associative() {
        let a = Front::from_points(vec![pt(-10, 0.5, 0), pt(-4, 0.95, 1)]);
        let b = Front::from_points(vec![pt(-8, 0.7, 2), pt(-6, 0.8, 3)]);
        let c = Front::from_points(vec![pt(-9, 0.55, 4), pt(-3, 0.99, 5)]);
        assert_eq!(shape(&a.merge(&b)), shape(&b.merge(&a)));
        let left = a.merge(&b).merge(&c);
        let right = a.merge(&b.merge(&c));
        assert_eq!(shape(&left), shape(&right));
        assert_staircase(&left);
    }

    #[test]
    fn merge_drops_dominated_across_fronts() {
        // b's (−7, 0.4) is dominated by a's (−10, 0.6); a's (−2, 0.7) is
        // dominated by b's (−5, 0.9). The union keeps only the frontier.
        let a = Front::from_points(vec![pt(-10, 0.6, 0), pt(-2, 0.7, 1)]);
        let b = Front::from_points(vec![pt(-7, 0.4, 2), pt(-5, 0.9, 3)]);
        assert_eq!(shape(&a.merge(&b)), vec![(-10, 0.6), (-5, 0.9)]);
    }

    #[test]
    fn merge_equal_mfe_keeps_higher_cai() {
        let a = Front::singleton(Energy(-6), 0.5, 0);
        let b = Front::singleton(Energy(-6), 0.8, 1);
        let m = a.merge(&b);
        assert_eq!(shape(&m), vec![(-6, 0.8)]);
        assert_eq!(m.points()[0].back, 1);
    }

    #[test]
    fn shift_slides_uniformly_and_preserves_size() {
        let f = Front::from_points(vec![pt(-10, 0.5, 0), pt(-5, 0.9, 1)]);
        let s = f.shift(Energy(3), 0.1);
        assert_eq!(shape(&s), vec![(-7, 0.6), (-2, 1.0)]);
        assert_eq!(s.len(), f.len());
        assert_staircase(&s);
    }

    #[test]
    fn shift_map_slides_and_rewrites_back() {
        // Same uniform slide as `shift`, but the back is rewritten from the
        // point's own `(mfe, cai)` — the shape the arena-backed front scan uses.
        let f = Front::from_points(vec![pt(-10, 0.5, 0), pt(-5, 0.9, 1)]);
        let s = f.shift_map(Energy(3), 0.1, |p| (p.back, p.mfe.0));
        assert_eq!(shape(&s), vec![(-7, 0.6), (-2, 1.0)]);
        assert_eq!(
            s.iter().map(|p| p.back).collect::<Vec<_>>(),
            vec![(0, -10), (1, -5)]
        );
        assert_staircase(&s);
        // Δ = 0 is a pure back remap, preserving the staircase verbatim.
        let remap = f.shift_map(Energy(0), 0.0, |p| p.back + 100);
        assert_eq!(shape(&remap), shape(&f));
        assert_eq!(
            remap.iter().map(|p| p.back).collect::<Vec<_>>(),
            vec![100, 101]
        );
    }

    #[test]
    fn compose_hand_checked_small_case() {
        // A = {(−10, 0.4), (−6, 0.9)}, B = {(−3, 0.2), (−1, 0.5)}, Δ = (−2, 0.0).
        // Products (mfe = a+b−2, cai = a+b):
        //   (−15, 0.6) (−13, 0.9) (−11, 1.1) (−9, 1.4)
        // (−13, 0.9) is dominated by (−15, 0.6)? no — higher cai, so kept.
        // Staircase: (−15,0.6),(−13,0.9),(−11,1.1),(−9,1.4) all non-dominated.
        let a = Front::from_points(vec![pt(-10, 0.4, 0), pt(-6, 0.9, 1)]);
        let b = Front::from_points(vec![pt(-3, 0.2, 2), pt(-1, 0.5, 3)]);
        let c = a.compose(&b, Energy(-2), 0.0, |x, y| (x, y));
        assert_shape(&c, &[(-15, 0.6), (-13, 0.9), (-11, 1.1), (-9, 1.4)]);
        assert_staircase(&c);
        // The lowest-mfe product traces back to the lowest-mfe pair of each side.
        assert_eq!(c.points()[0].back, (0, 2));
    }

    #[test]
    fn compose_eliminates_dominated_products() {
        // A = {(−10, 0.3), (−4, 0.8)}, B = {(−2, 0.1), (0, 0.2)}, Δ = 0.
        // Products: (−12,0.4)(−10,0.5)(−6,0.9)(−4,1.0).
        // (−10, 0.5) survives (higher cai than (−12,0.4)); all four form a
        // staircase — but flip B so a dominated product appears:
        // B = {(−2, 0.5), (0, 0.1)} → (−12,0.8)(−10,0.4)(−6,1.3)(−4,0.9).
        // Sorted: (−12,0.8)(−10,0.4)(−6,1.3)(−4,0.9); (−10,0.4) dominated by
        // (−12,0.8); (−4,0.9) dominated by (−6,1.3).
        let a = Front::from_points(vec![pt(-10, 0.3, 0), pt(-4, 0.8, 1)]);
        let b = Front::from_points(vec![pt(-2, 0.5, 2), pt(0, 0.1, 3)]);
        let c = a.compose(&b, Energy(0), 0.0, |x, y| (x, y));
        assert_shape(&c, &[(-12, 0.8), (-6, 1.3)]);
        assert_staircase(&c);
    }

    /// A geometric MFE grid keeps the max-CAI point per bucket plus the min-MFE
    /// endpoint, and the exact limit (`delta = 0`) and short fronts are no-ops.
    #[test]
    fn eps_grid_mfe_buckets_and_reduces_to_exact() {
        let f = Front::from_points(vec![
            pt(-100, 0.10, 0),
            pt(-90, 0.20, 1),
            pt(-80, 0.30, 2),
            pt(-40, 0.60, 3),
            pt(-20, 0.90, 4),
        ]);
        // A coarse grid collapses adjacent same-bucket points (keeping each
        // bucket's max-CAI representative) while keeping the endpoints.
        let coarse = f.prune(Prune::EpsGridMfe { delta: 3.0 });
        assert!(coarse.len() < f.len(), "coarse grid should drop points");
        assert_eq!(coarse.points()[0].mfe.0, -100, "min-MFE endpoint kept");
        assert_eq!(coarse.points().last().unwrap().mfe.0, -20, "max-CAI kept");
        assert_staircase(&coarse);
        // delta = 0 and a front of <= 2 points are exact no-ops.
        assert_eq!(shape(&f.prune(Prune::EpsGridMfe { delta: 0.0 })), shape(&f));
        let short = Front::from_points(vec![pt(-10, 0.5, 0), pt(-5, 0.9, 1)]);
        assert_eq!(
            shape(&short.prune(Prune::EpsGridMfe { delta: 5.0 })),
            shape(&short)
        );
    }

    /// A hard cap keeps exactly `max` points including both endpoints, and a
    /// front already within the cap is returned unchanged.
    #[test]
    fn cap_keeps_endpoints_and_spreads() {
        let f = Front::from_points(vec![
            pt(-100, 0.1, 0),
            pt(-90, 0.2, 1),
            pt(-80, 0.3, 2),
            pt(-70, 0.4, 3),
            pt(-60, 0.5, 4),
            pt(-50, 0.6, 5),
        ]);
        let capped = f.prune(Prune::Cap { max: 3 });
        assert_eq!(capped.len(), 3);
        assert_eq!(capped.points()[0].mfe.0, -100, "min-MFE endpoint kept");
        assert_eq!(capped.points().last().unwrap().mfe.0, -50, "max-CAI kept");
        assert_staircase(&capped);
        // Within the cap: unchanged.
        assert_eq!(shape(&f.prune(Prune::Cap { max: 10 })), shape(&f));
        // Prune::None is always a no-op.
        assert_eq!(shape(&f.prune(Prune::None)), shape(&f));
    }

    /// The capped grid honours its ceiling however fine the requested `delta`,
    /// keeps both endpoints, and is a no-op below the ceiling — the three
    /// properties the mode exists to combine.
    #[test]
    fn eps_grid_capped_honours_ceiling_and_keeps_endpoints() {
        let f = Front::from_points(
            (0..40)
                .map(|i| pt(-400 + i * 10, 0.01 * f64::from(i), i as u32))
                .collect(),
        );
        assert_eq!(f.len(), 40);
        // A ceiling below the front size binds, whatever `delta` is asked for:
        // the grid widens rather than giving up and spacing by index.
        for max in [2, 4, 9, 17] {
            let capped = f.prune(Prune::EpsGridCapped { delta: 0.0, max });
            assert!(
                capped.len() <= max,
                "ceiling {max} exceeded: got {}",
                capped.len()
            );
            assert_eq!(capped.points()[0].mfe.0, -400, "min-MFE endpoint kept");
            assert_eq!(
                capped.points().last().unwrap().mfe.0,
                -10,
                "max-CAI endpoint kept"
            );
            assert_staircase(&capped);
        }
        // Above the front size the ceiling cannot bind, so nothing is dropped —
        // the mode never prunes for its own sake.
        let loose = f.prune(Prune::EpsGridCapped {
            delta: 0.5,
            max: 100,
        });
        assert_eq!(shape(&loose), shape(&f));
    }

    #[test]
    fn compose_with_empty_is_empty() {
        let a = Front::from_points(vec![pt(-10, 0.3, 0)]);
        let empty = Front::<u32>::empty();
        assert!(a.compose(&empty, Energy(0), 0.0, |x, y| (x, y)).is_empty());
        assert!(empty.compose(&a, Energy(0), 0.0, |x, y| (x, y)).is_empty());
    }
}
