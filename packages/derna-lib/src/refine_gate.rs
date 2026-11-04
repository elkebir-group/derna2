//! Refine a pass grid's loose brackets on the hull's own evidence.
//!
//! A seeded walk refines each pair of consecutive grid optima on its own, so
//! the gaps whose tie weights fall between two grid weights are known, with
//! their chord ends, once that pair's probes are done, and before their folds
//! start: probes outrank folds in the walk's queue. A gap's chord is loose at
//! the root by the hull's concavity over its bracket,
//! `[g(λ*) − (t·g(λ_lo) + (1 − t)·g(λ_hi))]/λ*` with `g` the scalarized optimum
//! over the hull's vertices, and a fold cut by a loose chord keeps more cells.
//! [`Gate`] holds a bracket's gaps until its probes are done, and when the
//! width-weighted mean of their root looseness exceeds a threshold queues the
//! candidate grid's weights inside the bracket as exact passes and holds the
//! gaps until those are pooled; a released gap reads the tightened chord when
//! its fold starts. Added passes only tighten a chord, so the front's
//! objectives do not change.
//!
//! [`place`] is the priced alternative to the threshold: it predicts each held
//! gap's fold work in exact passes, under its chord now and under the chord an
//! added pass would leave, and adds a pass at the tie weight of a held gap
//! while the predicted saving exceeds the pass's price.

/// What [`Gate`] decided for one bracket of the grid.
#[derive(Debug, Clone, PartialEq)]
pub struct BracketRefinement {
    /// The bracket's lower grid weight.
    pub lo: f64,
    /// The bracket's upper grid weight.
    pub hi: f64,
    /// The candidate weights strictly inside it.
    pub candidates: Vec<f64>,
    /// Seconds from the walk's start until its probes were done and it was
    /// decided; `None` when no pair was refined inside it.
    pub decided_s: Option<f64>,
    /// Gaps held for the decision.
    pub gaps: usize,
    /// The width-weighted mean root chord looseness, kcal/mol, over the held
    /// gaps that read a chord; `None` when none did.
    pub looseness_kcal: Option<f64>,
    /// The weights queued as exact passes: empty when the bracket was kept.
    pub added: Vec<f64>,
    /// With [`place`], the held gaps' predicted fold work in exact passes
    /// before and after the added passes; `None` under the threshold.
    pub predicted_passes: Option<(f64, f64)>,
    /// Seconds from the walk's start until the last added pass was pooled
    /// and the bracket's gaps released.
    pub released_s: Option<f64>,
}

/// The width-weighted mean of `(looseness, width)` items: `Σ l·w / Σ w`, or
/// `None` without positive width.
pub(crate) fn weighted_looseness(items: &[(f64, f64)]) -> Option<f64> {
    let width: f64 = items.iter().map(|&(_, w)| w).sum();
    (width > 0.0).then(|| items.iter().map(|&(l, w)| l * w).sum::<f64>() / width)
}

/// Whether a bracket whose gaps read `items` refines at `threshold`.
pub(crate) fn refines(items: &[(f64, f64)], threshold: f64) -> bool {
    weighted_looseness(items).is_some_and(|l| l > threshold)
}

/// A held gap as [`place`] reads it: its tie weight, energy width in
/// kcal/mol and slice count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HeldGap {
    pub lambda: f64,
    pub width_kcal: f64,
    pub slices: usize,
}

/// The chord of a gap at `lambda`: the nearest of `passes` (ascending) at or
/// below it and at or above it, one pass when it holds `lambda` itself.
pub(crate) fn chord_of(passes: &[f64], lambda: f64) -> Option<(f64, f64)> {
    let i = passes.partition_point(|&w| w < lambda);
    let hi = *passes.get(i)?;
    let lo = if hi == lambda {
        hi
    } else {
        *passes.get(i.checked_sub(1)?)?
    };
    Some((lo, hi))
}

/// The root chord looseness at `lambda` of the chord `(lo, hi)` under the
/// scalarized optimum `g`, kcal/mol: zero for a pass at `lambda` itself.
pub(crate) fn root_looseness(g: impl Fn(f64) -> f64, lambda: f64, (lo, hi): (f64, f64)) -> f64 {
    if hi > lo {
        let s = (hi - lambda) / (hi - lo);
        (g(lambda) - (s * g(lo) + (1.0 - s) * g(hi))) / lambda
    } else {
        0.0
    }
}

/// What [`place`] chose for a bracket.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Placement {
    /// The weights to add as exact passes, in the order chosen.
    pub added: Vec<f64>,
    /// The held gaps' predicted fold work, in passes, under `passes`.
    pub before: f64,
    /// The same with `added` pooled too.
    pub after: f64,
}

/// Choose up to `max_passes` weights among the held `gaps`' tie weights to add
/// as exact passes to `passes` (ascending, pooled or queued), one at a time,
/// each the one whose predicted saving is largest, while that saving exceeds
/// `margin` passes. `cost(gap, looseness)` predicts a fold's work in passes;
/// `g` is the scalarized optimum, kcal/mol, exact wherever a gap lies.
pub(crate) fn place(
    gaps: &[HeldGap],
    passes: &[f64],
    g: impl Fn(f64) -> f64,
    cost: impl Fn(&HeldGap, f64) -> f64,
    margin: f64,
    max_passes: usize,
) -> Placement {
    let work = |pool: &[f64]| -> f64 {
        gaps.iter()
            .filter_map(|gap| {
                let chord = chord_of(pool, gap.lambda)?;
                Some(cost(gap, root_looseness(&g, gap.lambda, chord).max(0.0)))
            })
            .sum()
    };
    let mut pool: Vec<f64> = passes.to_vec();
    let before = work(&pool);
    let (mut now, mut added) = (before, Vec::new());
    while added.len() < max_passes {
        let best = gaps
            .iter()
            .map(|gap| gap.lambda)
            .filter(|w| !pool.contains(w))
            .map(|w| {
                let mut with = pool.clone();
                with.insert(with.partition_point(|&p| p < w), w);
                (w, work(&with))
            })
            .min_by(|x, y| x.1.total_cmp(&y.1));
        match best {
            Some((w, after)) if now - after > margin => {
                pool.insert(pool.partition_point(|&p| p < w), w);
                added.push(w);
                now = after;
            }
            _ => break,
        }
    }
    Placement {
        added,
        before,
        after: now,
    }
}

/// A bracket's progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Its pair is still being refined; its gaps are held.
    Probing,
    /// Its added passes are queued, this many still to be pooled.
    Refining(usize),
    /// Decided and released; later gaps pass straight through.
    Open,
}

/// A held fold with its queue priority.
pub(crate) type Held<T> = ((u8, u64), T);

struct Bracket<T> {
    record: BracketRefinement,
    outstanding: usize,
    seeded: bool,
    held: Vec<Held<T>>,
    phase: Phase,
}

impl<T> Bracket<T> {
    fn new(lo: f64, hi: f64, candidates: Vec<f64>) -> Self {
        Self {
            record: BracketRefinement {
                lo,
                hi,
                candidates,
                decided_s: None,
                gaps: 0,
                looseness_kcal: None,
                added: Vec::new(),
                predicted_passes: None,
                released_s: None,
            },
            outstanding: 0,
            seeded: false,
            held: Vec::new(),
            phase: Phase::Probing,
        }
    }
}

/// What the walk does once a bracket is decided.
pub(crate) enum Decision<T> {
    /// Queue these folds now.
    Release(Vec<Held<T>>),
    /// Queue exact passes at these weights; the folds stay held.
    Refine(Vec<f64>),
}

/// The brackets of a seeded walk's grid with candidate weights inside them,
/// their probes outstanding and their held folds.
pub(crate) struct Gate<T> {
    brackets: Vec<Bracket<T>>,
}

impl<T> Gate<T> {
    /// Brackets between consecutive `slots` (the seed weights, ascending) with
    /// at least one of `candidates` strictly between them.
    pub(crate) fn new(slots: &[f64], candidates: &[f64]) -> Self {
        let brackets = slots
            .windows(2)
            .filter_map(|w| {
                let inside: Vec<f64> = candidates
                    .iter()
                    .copied()
                    .filter(|&c| w[0] < c && c < w[1])
                    .collect();
                (!inside.is_empty()).then(|| Bracket::new(w[0], w[1], inside))
            })
            .collect();
        Self { brackets }
    }

    /// Every bracket between consecutive `slots` (the seed weights,
    /// ascending), for [`place`] to choose weights inside.
    pub(crate) fn every(slots: &[f64]) -> Self {
        let brackets = slots
            .windows(2)
            .map(|w| Bracket::new(w[0], w[1], Vec::new()))
            .collect();
        Self { brackets }
    }

    /// The bracket a seed pair with tie weight `lambda` refines inside.
    pub(crate) fn bracket_of(&self, lambda: f64) -> Option<usize> {
        self.brackets
            .iter()
            .position(|b| b.record.lo <= lambda && lambda < b.record.hi)
    }

    /// A probe tagged `tag` was queued; `root` when it is its seed pair's.
    pub(crate) fn probe_queued(&mut self, tag: Option<usize>, root: bool) {
        if let Some(b) = tag.and_then(|i| self.brackets.get_mut(i)) {
            b.outstanding += 1;
            b.seeded |= root;
        }
    }

    /// A probe tagged `tag` finished, after queueing its children and its
    /// gap: the bracket to decide when that was its last.
    pub(crate) fn probe_done(&mut self, tag: Option<usize>) -> Option<usize> {
        let i = tag?;
        let b = &mut self.brackets[i];
        b.outstanding -= 1;
        (b.outstanding == 0 && b.seeded && b.phase == Phase::Probing).then_some(i)
    }

    /// Hold a fold tagged `tag` while its bracket is undecided or refining,
    /// or hand it back to be queued.
    pub(crate) fn hold(&mut self, tag: Option<usize>, held: Held<T>) -> Option<Held<T>> {
        match tag.and_then(|i| self.brackets.get_mut(i)) {
            Some(b) if b.phase != Phase::Open => {
                b.held.push(held);
                None
            }
            _ => Some(held),
        }
    }

    /// The folds held in bracket `i`.
    pub(crate) fn held(&self, i: usize) -> impl Iterator<Item = &T> {
        self.brackets[i].held.iter().map(|(_, job)| job)
    }

    /// Decide bracket `i` at `at_s` from its gaps' `items` against
    /// `threshold`, skipping candidates `pooled` already holds.
    pub(crate) fn decide(
        &mut self,
        i: usize,
        items: &[(f64, f64)],
        threshold: f64,
        at_s: f64,
        pooled: impl Fn(f64) -> bool,
    ) -> Decision<T> {
        let added: Vec<f64> = if refines(items, threshold) {
            self.brackets[i]
                .record
                .candidates
                .iter()
                .copied()
                .filter(|&w| !pooled(w))
                .collect()
        } else {
            Vec::new()
        };
        self.settle(i, weighted_looseness(items), added, None, at_s)
    }

    /// Decide bracket `i` at `at_s` by `placement`, with `looseness` its
    /// gaps' weighted mean for the record.
    pub(crate) fn decide_placed(
        &mut self,
        i: usize,
        looseness: Option<f64>,
        placement: Placement,
        at_s: f64,
    ) -> Decision<T> {
        let predicted = Some((placement.before, placement.after));
        self.settle(i, looseness, placement.added, predicted, at_s)
    }

    fn settle(
        &mut self,
        i: usize,
        looseness: Option<f64>,
        added: Vec<f64>,
        predicted: Option<(f64, f64)>,
        at_s: f64,
    ) -> Decision<T> {
        let b = &mut self.brackets[i];
        b.record.decided_s = Some(at_s);
        b.record.gaps = b.held.len();
        b.record.looseness_kcal = looseness;
        b.record.predicted_passes = predicted;
        if added.is_empty() {
            b.phase = Phase::Open;
            b.record.released_s = Some(at_s);
            return Decision::Release(std::mem::take(&mut b.held));
        }
        b.record.added.clone_from(&added);
        b.phase = Phase::Refining(added.len());
        Decision::Refine(added)
    }

    /// An exact pass at `lambda` was pooled at `at_s`: the folds it releases,
    /// when it was a refining bracket's last added pass.
    pub(crate) fn pooled(&mut self, lambda: f64, at_s: f64) -> Vec<Held<T>> {
        for b in &mut self.brackets {
            if let Phase::Refining(left) = b.phase {
                if b.record.added.contains(&lambda) {
                    if left > 1 {
                        b.phase = Phase::Refining(left - 1);
                        return Vec::new();
                    }
                    b.phase = Phase::Open;
                    b.record.released_s = Some(at_s);
                    return std::mem::take(&mut b.held);
                }
            }
        }
        Vec::new()
    }

    /// Every bracket's record, by ascending weight.
    pub(crate) fn records(self) -> Vec<BracketRefinement> {
        self.brackets.into_iter().map(|b| b.record).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        chord_of, place, refines, root_looseness, weighted_looseness, Decision, Gate, HeldGap,
    };

    const SLOTS: [f64; 5] = [0.12, 0.14, 0.16, 0.21, 0.26];
    const CANDIDATES: [f64; 9] = [0.15, 0.17, 0.18, 0.19, 0.2, 0.22, 0.23, 0.24, 0.25];

    /// The weighted mean counts a wide gap more, and the threshold is strict.
    #[test]
    fn the_rule_weighs_looseness_by_gap_width() {
        let items = [(4.0, 3.0), (1.0, 1.0)];
        assert_eq!(weighted_looseness(&items), Some(3.25));
        assert!(refines(&items, 3.2));
        assert!(!refines(&items, 3.25));
        assert_eq!(weighted_looseness(&[]), None);
        assert!(!refines(&[], 0.0));
    }

    /// Only brackets with a candidate inside are watched.
    #[test]
    fn brackets_are_the_slots_around_candidates() {
        let gate: Gate<u32> = Gate::new(&SLOTS, &CANDIDATES);
        assert_eq!(gate.bracket_of(0.13), None);
        assert_eq!(gate.bracket_of(0.21), Some(2));
        assert_eq!(gate.bracket_of(0.26), None);
        let spans: Vec<_> = gate.records().iter().map(|r| (r.lo, r.hi)).collect();
        assert_eq!(spans, vec![(0.14, 0.16), (0.16, 0.21), (0.21, 0.26)]);
    }

    /// A loose bracket holds its folds through its probes and its added
    /// passes, then releases them; a later fold in it is not held.
    #[test]
    fn a_loose_bracket_holds_its_folds_until_its_passes_are_pooled() {
        let mut gate: Gate<u32> = Gate::new(&SLOTS, &CANDIDATES);
        gate.probe_queued(Some(2), true);
        gate.probe_queued(Some(2), false);
        assert!(gate.hold(Some(2), ((0, 5), 7)).is_none());
        assert_eq!(gate.probe_done(Some(2)), None);
        assert!(gate.hold(Some(2), ((0, 9), 8)).is_none());
        assert_eq!(gate.probe_done(Some(2)), Some(2));
        assert_eq!(gate.held(2).copied().collect::<Vec<_>>(), vec![7, 8]);
        let added = match gate.decide(2, &[(4.0, 1.0)], 3.2, 10.0, |w| w == 0.23) {
            Decision::Refine(added) => added,
            Decision::Release(_) => panic!("a loose bracket refines"),
        };
        assert_eq!(added, vec![0.22, 0.24, 0.25]);
        assert!(gate.pooled(0.22, 11.0).is_empty());
        assert!(gate.pooled(0.17, 11.5).is_empty());
        assert!(gate.pooled(0.24, 12.0).is_empty());
        let released: Vec<u32> = gate.pooled(0.25, 13.0).into_iter().map(|h| h.1).collect();
        assert_eq!(released, vec![7, 8]);
        assert_eq!(gate.hold(Some(2), ((0, 1), 9)), Some(((0, 1), 9)));
        let record = &gate.records()[2];
        assert_eq!(record.gaps, 2);
        assert_eq!(record.looseness_kcal, Some(4.0));
        assert_eq!(
            (record.decided_s, record.released_s),
            (Some(10.0), Some(13.0))
        );
    }

    /// A tight bracket releases its folds at once and adds nothing; folds
    /// outside every bracket are never held.
    #[test]
    fn a_tight_bracket_releases_its_folds_and_adds_nothing() {
        let mut gate: Gate<u32> = Gate::new(&SLOTS, &CANDIDATES);
        assert_eq!(gate.hold(None, ((0, 1), 3)), Some(((0, 1), 3)));
        gate.probe_queued(Some(1), true);
        assert!(gate.hold(Some(1), ((0, 2), 4)).is_none());
        assert_eq!(gate.probe_done(Some(1)), Some(1));
        match gate.decide(1, &[(2.0, 5.0)], 3.2, 4.0, |_| false) {
            Decision::Release(held) => assert_eq!(held, vec![((0, 2), 4)]),
            Decision::Refine(_) => panic!("a tight bracket is kept"),
        }
        let records = gate.records();
        assert!(records[1].added.is_empty());
        assert_eq!(records[0].decided_s, None);
    }

    /// A chord is the nearest pass on each side, or one pass at the weight.
    #[test]
    fn a_chord_is_the_nearest_passes_around_a_weight() {
        let passes = [0.31, 0.36, 0.41];
        assert_eq!(chord_of(&passes, 0.33), Some((0.31, 0.36)));
        assert_eq!(chord_of(&passes, 0.36), Some((0.36, 0.36)));
        assert_eq!(chord_of(&passes, 0.30), None);
        assert_eq!(chord_of(&passes, 0.42), None);
        let g = |w: f64| -(w - 0.35) * (w - 0.35);
        assert_eq!(root_looseness(g, 0.36, (0.36, 0.36)), 0.0);
        assert!(root_looseness(g, 0.35, (0.31, 0.41)) > 0.0);
    }

    /// The pass goes where it saves most, at the heaviest gap's tie weight,
    /// and only while the saving exceeds the margin.
    #[test]
    fn a_priced_pass_goes_to_the_heaviest_gap_while_it_pays() {
        let g = |w: f64| -100.0 * (w - 0.35) * (w - 0.35);
        let gap = |lambda, width_kcal| HeldGap {
            lambda,
            width_kcal,
            slices: 1,
        };
        let gaps = [gap(0.33, 10.0), gap(0.38, 1.0), gap(0.40, 0.5)];
        let cost = |gap: &HeldGap, loose: f64| gap.width_kcal * loose;
        let passes = [0.31, 0.41];
        let one = place(&gaps, &passes, g, cost, 0.0, 1);
        assert_eq!(one.added, vec![0.33]);
        assert!(one.after < one.before);
        let two = place(&gaps, &passes, g, cost, 0.0, 2);
        assert_eq!(two.added.len(), 2);
        assert_eq!(two.added[0], 0.33);
        assert!(two.after < one.after);
        let none = place(&gaps, &passes, g, cost, one.before, 3);
        assert!(none.added.is_empty());
        assert_eq!(none.after, none.before);
    }
}
