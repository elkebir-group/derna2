//! The best design whose structure crosses a segment's 3′ end cheaply enough
//! to survive a refold certificate.
//!
//! # The certificate and its margin
//!
//! A segment certificate refolds a segment `S` of a design's constrained structure
//! `G` freely: `F` is `G` without its pairs touching `S` (the *S-pairs*), plus
//! the segment's free fold. `E(F) < E(G)` proves the design is not modular
//! . At `dangles = 0` with a linear multi-loop, `E(G) − E(F)` is a sum
//! over `G`'s loops once each pair is labelled by position, and the part a
//! coding-sequence end of `S` contributes reads only a short **chain**: the
//! S-pairs opening in the designed end of `S` (positions `top ..= last`), each
//! nested in the one before through a single-branch loop, ending either in a
//! hairpin or in the first enclosed pair that is not an S-pair. The chain's
//! margin is
//!
//! ```text
//! Σ loop energies the chain's pairs close + β(top) − β(terminal pair, if any)
//! ```
//!
//! with `β(p)` the terminal-AU term of `p` plus, when the loop holding `S` is a
//! multi-loop, the multi-loop per-branch term — the branch term `p` pays where
//! it sits in `G`, against what the pair the certificate promotes pays in `F`.
//!
//! # What this computes
//!
//! The exact optimum, over every design admitting a structure whose chain
//! opens at `top` and whose chain margin is at most `budget`, of the scan's
//! own objective: the chain's outermost pair's outside value, each chain loop
//! scored as the scan scores it (stacks and bulges on their fills, internal
//! loops and hairpins over their terminal-mismatch corners or, at the
//! special-hairpin sizes, their spellings), and the terminal pair's inside
//! value. Margins are integer energies tracked per loop, never folded into the
//! objective, so the restriction is a hard one and not a penalty: no
//! multiplier is involved.
//!
//! A chain pair closing a **multi-loop** pays the multi-loop's closing and
//! branch terms, which no chain of the few designed positions can repay; the
//! pass computes the least margin any such chain could reach from the energy
//! model's own tables and reports it, so a caller can confirm it exceeds the
//! budget rather than assume it.
//!
//! # Preconditions
//!
//! The tables are the exact fold (a beam drops consumers), and every position
//! of the designed end before `top` is required unpaired, so the chain's
//! outermost pair has no S-pair around it.

use std::collections::BTreeMap;

use rna_mfe::loops::{
    bulge_energy, hairpin_energy, hairpin_energy_scalar, internal_energy_scalar,
    pair_from_nucleotides, stack_energy, BulgeLoop, HairpinLoop, InternalLoopScalar, StackPair,
};
use rna_mfe::params::BasePair;
use rna_mfe::{Energy, Nucleotide, Params};

use super::{OutsideScan, MIN_HAIRPIN_LOOP};
use crate::dfa::{Dfa, Node};
use crate::region::PairConstraint;
use crate::state::PairKey;
use crate::tables::Tables;

/// Which chains to optimize over.
#[derive(Debug, Clone)]
pub struct CrossingSpec {
    /// The chain's outermost pair opens here (0-based).
    pub top: u32,
    /// The last position (0-based, inclusive) whose pairs are S-pairs; pairs
    /// opening after it end the chain.
    pub last: u32,
    /// Largest admissible chain margin, dcal/mol.
    pub budget: i32,
    /// Whether the loop holding the segment is a multi-loop.
    pub multiloop: bool,
    /// When non-empty, only the chain whose pairs, terminal pair included, are
    /// exactly these — a control that names one configuration.
    pub only: Vec<(u32, u32)>,
    /// Score only chains whose design lies at most this far above the scan's
    /// optimum. A cell whose best design is already above it is skipped, which
    /// is exact for every chain at or below it: a chain through a cell is one
    /// design through that cell. Infinite scores every chain.
    pub depth_ceiling: f64,
}

/// How a chain ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainEnd {
    /// In the first enclosed pair that is not an S-pair (the last of `pairs`).
    Pair,
    /// In a hairpin closed by its last S-pair.
    Hairpin,
}

/// One scored chain.
#[derive(Debug, Clone)]
pub struct CrossingChain {
    /// The best whole design through the chain, on the tables' scale.
    pub value: f64,
    /// The chain's margin, dcal/mol.
    pub margin: i32,
    /// The chain's pairs outermost first, terminal pair included.
    pub pairs: Vec<(u32, u32)>,
    /// How it ends.
    pub end: ChainEnd,
}

/// The best chains of one shape: one position set of chain pairs, terminal
/// pair included.
#[derive(Debug, Clone)]
pub struct ShapeBest {
    /// The shape's pairs, outermost first.
    pub pairs: Vec<(u32, u32)>,
    /// How the shape ends.
    pub end: ChainEnd,
    /// Its best value at any margin, and that chain's margin.
    pub value: f64,
    /// See [`Self::value`].
    pub margin_at_value: i32,
    /// The least margin any of its chains reached.
    pub least_margin: i32,
    /// Its best value within the budget, infinite when none is.
    pub admitted_value: f64,
}

/// What the pass found, and how much of the chain space it scored.
#[derive(Debug, Clone, Default)]
pub struct CrossingReport {
    /// The best chain within the budget.
    pub best: Option<CrossingChain>,
    /// The best chain at any margin: a control, and the value the budget
    /// moved the answer from.
    pub best_any: Option<CrossingChain>,
    /// Outermost-pair cells the pass started from.
    pub tops: usize,
    /// Chains scored, and those within the budget.
    pub chains: u64,
    /// Chains within the budget.
    pub admitted: u64,
    /// Distinct position sets of chain pairs scored, and of those admitted.
    pub shapes: usize,
    /// Distinct position sets admitted.
    pub shapes_admitted: usize,
    /// Every shape scored, in position order.
    pub by_shape: Vec<ShapeBest>,
    /// Top cells whose best scored chain equals their own best inside, and
    /// those whose best inside no chain reaches (a multi-loop interior, or a
    /// chain pair's own multi-loop). A chain scoring *below* its top's inside
    /// is a scoring error and panics.
    pub tops_exact: usize,
    /// See [`Self::tops_exact`].
    pub tops_above: usize,
    /// The least margin a chain through a multi-loop-closing S-pair could
    /// reach, over every depth the chain can have.
    pub multiloop_floor: i32,
    /// The scan's optimum: depths are read against it.
    pub optimum: f64,
    /// The constant added to every value above to put it on the unrestricted
    /// objective's scale: a residue restricted to some of its codons is priced
    /// against its best remaining codon inside the lattice, and this restores
    /// the standard weights. Zero for an unrestricted lattice.
    pub objective_offset: f64,
}

impl CrossingReport {
    /// Shift every value by `offset`, recording it.
    pub(crate) fn rebase(&mut self, offset: f64) {
        for c in [&mut self.best, &mut self.best_any].into_iter().flatten() {
            c.value += offset;
        }
        for b in &mut self.by_shape {
            b.value += offset;
            b.admitted_value += offset;
        }
        self.optimum += offset;
        self.objective_offset += offset;
    }

    /// Fold another worker's report into this one.
    fn merge(&mut self, other: CrossingReport, shapes: &mut BTreeMap<Vec<(u32, u32)>, ShapeBest>) {
        for (keep, take) in [
            (&mut self.best, other.best),
            (&mut self.best_any, other.best_any),
        ] {
            if let Some(c) = take {
                if keep.as_ref().is_none_or(|k| c.value < k.value) {
                    *keep = Some(c);
                }
            }
        }
        self.tops += other.tops;
        self.chains += other.chains;
        self.admitted += other.admitted;
        self.tops_exact += other.tops_exact;
        self.tops_above += other.tops_above;
        for b in other.by_shape {
            match shapes.get_mut(&b.pairs) {
                Some(a) => a.absorb(&b),
                None => {
                    shapes.insert(b.pairs.clone(), b);
                }
            }
        }
    }
}

impl ShapeBest {
    fn absorb(&mut self, b: &ShapeBest) {
        if b.value < self.value {
            self.value = b.value;
            self.margin_at_value = b.margin_at_value;
        }
        self.least_margin = self.least_margin.min(b.least_margin);
        self.admitted_value = self.admitted_value.min(b.admitted_value);
    }
}

/// A closed pair as the chain reads it: outer node, key, inside value.
type Cell = (Node, PairKey, f64);

/// A chain under construction.
struct Walker<'s, 'a> {
    scan: &'s OutsideScan<'a>,
    spec: &'s CrossingSpec,
    /// Every closed pair at or below the ceiling, by its closing position,
    /// each list ordered by opening position.
    by_close: &'s [Vec<Cell>],
    top_beta: i32,
    report: CrossingReport,
    shapes: BTreeMap<Vec<(u32, u32)>, ShapeBest>,
    chain: Vec<(u32, u32)>,
    /// The best value any chain from the current top reached.
    top_best: f64,
}

impl<'a> OutsideScan<'a> {
    /// Run the backward sweep over exact `tables`, then score every crossing
    /// chain `spec` names.
    pub(crate) fn crossing(
        params: &'a Params,
        dfa: &'a Dfa,
        tables: &'a Tables,
        constraint: &'a PairConstraint,
        single_max_len: usize,
        threads: usize,
        spec: &CrossingSpec,
    ) -> CrossingReport {
        let mut scan = Self::with_recording(
            params,
            dfa,
            tables,
            constraint,
            single_max_len,
            false,
            threads,
        );
        let mut clock = super::Phases::from_env();
        clock.time("index_outers", || scan.index_outers());
        scan.sweep(&mut clock);
        clock.report();
        scan.chains(spec)
    }

    fn chains(&self, spec: &CrossingSpec) -> CrossingReport {
        let optimum = self
            .dfa
            .completion_ends()
            .iter()
            .map(|&end| self.in_c(end))
            .fold(f64::INFINITY, f64::min);
        let ceiling = optimum + spec.depth_ceiling;
        let mut by_close: Vec<Vec<Cell>> = vec![Vec::new(); self.dfa.seq_len as usize];
        for pos in 1..=self.dfa.seq_len {
            for (o, key, inside) in self.p_cells(pos) {
                let out = self.out_p[self.dfa.ix(o)]
                    .get(&key)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                if key.open.pos >= spec.top && out + inside <= ceiling {
                    by_close[(pos - 1) as usize].push((o, key, inside));
                }
            }
        }
        for cells in &mut by_close {
            cells.sort_by_key(|(_, key, _)| key.open.pos);
        }
        let tops: Vec<Cell> = by_close
            .iter()
            .flatten()
            .filter(|(_, key, _)| key.open.pos == spec.top)
            .filter(|(_, key, _)| {
                spec.only
                    .first()
                    .is_none_or(|&p| p == (key.open.pos, key.close.pos))
            })
            .copied()
            .collect();
        let workers = self.threads.clamp(1, tops.len().max(1));
        let chunk = tops.len().div_ceil(workers).max(1);
        let parts: Vec<CrossingReport> = std::thread::scope(|sc| {
            let handles: Vec<_> = tops
                .chunks(chunk)
                .map(|mine| {
                    let by_close = &by_close;
                    crate::task_clock::spawn(sc, move || self.walk_tops(spec, by_close, mine))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a crossing worker"))
                .collect()
        });
        let mut report = CrossingReport {
            multiloop_floor: self.multiloop_floor(spec),
            optimum,
            ..CrossingReport::default()
        };
        let mut shapes = BTreeMap::new();
        for part in parts {
            report.merge(part, &mut shapes);
        }
        report.shapes = shapes.len();
        report.by_shape = shapes.into_values().collect();
        report.shapes_admitted = report
            .by_shape
            .iter()
            .filter(|b| b.admitted_value.is_finite())
            .count();
        report
    }

    /// Score every chain from `tops`, one worker's share.
    fn walk_tops(
        &self,
        spec: &CrossingSpec,
        by_close: &[Vec<Cell>],
        tops: &[Cell],
    ) -> CrossingReport {
        let mut w = Walker {
            scan: self,
            spec,
            by_close,
            top_beta: 0,
            report: CrossingReport::default(),
            shapes: BTreeMap::new(),
            chain: Vec::new(),
            top_best: f64::INFINITY,
        };
        for &(o, key, inside) in tops {
            let Some(&out) = self.out_p[self.dfa.ix(o)].get(&key) else {
                continue;
            };
            let Some(stat) = self.stat(o, key) else {
                continue;
            };
            w.report.tops += 1;
            w.top_beta = self.beta(key, spec.multiloop);
            w.top_best = f64::INFINITY;
            w.chain.clear();
            w.walk(key, out + stat, 0);
            let floor = out + inside;
            let tol = 1e-6 * floor.abs().max(1.0);
            assert!(
                w.top_best >= floor - tol,
                "a crossing chain scored {} below its outermost pair's best {}",
                w.top_best,
                floor
            );
            if (w.top_best - floor).abs() <= tol {
                w.report.tops_exact += 1;
            } else {
                w.report.tops_above += 1;
            }
        }
        w.report.by_shape = w.shapes.into_values().collect();
        w.report
    }

    /// A pair's own terms, closing into `o`: the constraint's adjustment and
    /// its two edges' adaptation, or `None` where the constraint forbids it.
    fn stat(&self, o: Node, key: PairKey) -> Option<f64> {
        let constraint = self.constraint(key.open.pos, key.close.pos)?;
        let ep = self.open_edge(key);
        let eq = key
            .close_edge(self.dfa, o)
            .expect("a pair's edge is in the lattice");
        Some(constraint - self.kk * (ep.cai + eq.cai))
    }

    /// The edge that opens the pair `key`.
    fn open_edge(&self, key: PairKey) -> crate::dfa::Edge {
        key.open_edge(self.dfa)
            .expect("a pair's edge is in the lattice")
    }

    /// The branch term a pair pays in the loop holding the segment.
    fn beta(&self, key: PairKey, multiloop: bool) -> i32 {
        let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let ml = if multiloop {
            self.params.ml.intern_37
        } else {
            0
        };
        self.terminal_au(pair).0 + ml
    }

    /// The least margin of a chain one of whose S-pairs closes a multi-loop.
    ///
    /// That pair's loop pays the closing and two branch terms (its own, and the
    /// one the chain's outermost pair pays); every further S-position can at
    /// best add a branch term and its own least loop, and every S-pair above
    /// it its least loop. All terms come from the energy model's tables.
    ///
    /// Only a multi-loop holding the segment gives a floor: there, each pair
    /// the certificate promotes repays the branch term it paid inside the
    /// chain's multi-loop. In the exterior loop it does not, and a multi-loop
    /// with many branches has no floor at all.
    fn multiloop_floor(&self, spec: &CrossingSpec) -> i32 {
        if !spec.multiloop {
            return i32::MIN;
        }
        let ml = &self.params.ml;
        let loop_floor = self.single_loop_floor().min(self.hairpin_floor()).min(0);
        let branch = ml.intern_37.min(0);
        let span = (spec.last - spec.top) as i32;
        (0..=span)
            .map(|above| {
                let below = span - above;
                above * loop_floor
                    + ml.closing_37
                    + ml.intern_37
                    + branch
                    + below * (branch + loop_floor)
            })
            .min()
            .expect("at least one depth")
    }

    /// The least energy of any stack, bulge or internal loop the model scores
    /// within the loop bound, over every pair type and boundary nucleotide.
    fn single_loop_floor(&self) -> i32 {
        let pairs = all_pairs();
        let nucs = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
        let mut floor = i32::MAX;
        for &outer in &pairs {
            for &inner in &pairs {
                floor = floor.min(stack_energy(self.params, &StackPair { outer, inner }).0);
                for size in 1..=self.single_max_len {
                    let b = BulgeLoop { outer, inner, size };
                    floor = floor.min(bulge_energy(self.params, &b).0);
                }
                for n1 in 1..self.single_max_len {
                    for n2 in 1..=(self.single_max_len - n1) {
                        for &l5 in &nucs {
                            for &l3 in &nucs {
                                for &r5 in &nucs {
                                    for &r3 in &nucs {
                                        let e = internal_energy_scalar(
                                            self.params,
                                            &InternalLoopScalar {
                                                outer,
                                                inner,
                                                boundary_left5p: l5,
                                                boundary_left3p: l3,
                                                boundary_right5p: r5,
                                                boundary_right3p: r3,
                                                n1,
                                                n2,
                                            },
                                        );
                                        floor = floor.min(e.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        floor
    }

    /// The least hairpin energy the model gives any closing pair and loop up
    /// to the loop bound, special loops spelled out.
    fn hairpin_floor(&self) -> i32 {
        let nucs = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
        let mut floor = i32::MAX;
        for &closing in &all_pairs() {
            for size in MIN_HAIRPIN_LOOP as usize..=self.single_max_len {
                if matches!(size, 3 | 4 | 6) {
                    for code in 0..4usize.pow(size as u32) {
                        let seq: Vec<Nucleotide> =
                            (0..size).map(|k| nucs[(code >> (2 * k)) & 3]).collect();
                        let h = HairpinLoop {
                            closing,
                            loop_seq: &seq,
                        };
                        floor = floor.min(hairpin_energy(self.params, &h).0);
                    }
                } else {
                    for &first in &nucs {
                        for &last in &nucs {
                            let e = hairpin_energy_scalar(self.params, closing, size, first, last);
                            floor = floor.min(e.0);
                        }
                    }
                }
            }
        }
        floor
    }
}

fn all_pairs() -> Vec<BasePair> {
    let nucs = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
    let mut out = Vec::new();
    for &a in &nucs {
        for &b in &nucs {
            let p = pair_from_nucleotides(a, b);
            if p != BasePair::NN {
                out.push(p);
            }
        }
    }
    out
}

impl Walker<'_, '_> {
    /// Extend the chain from S-pair `key`. `acc` holds the outermost
    /// pair's outside and every chain term so far, this pair's own included;
    /// `margin` the chain's loop energies so far.
    fn walk(&mut self, key: PairKey, acc: f64, margin: i32) {
        self.chain.push((key.open.pos, key.close.pos));
        self.hairpins(key, acc, margin);
        self.singles(key, acc, margin);
        self.chain.pop();
    }

    fn score(&mut self, value: f64, margin: i32, end: ChainEnd) {
        if value < self.top_best {
            self.top_best = value;
        }
        let pairs = self.chain.clone();
        if !self.spec.only.is_empty() && self.spec.only != pairs {
            return;
        }
        self.report.chains += 1;
        let admitted = margin <= self.spec.budget;
        let slot = self
            .shapes
            .entry(pairs.clone())
            .or_insert_with(|| ShapeBest {
                pairs: pairs.clone(),
                end,
                value: f64::INFINITY,
                margin_at_value: margin,
                least_margin: margin,
                admitted_value: f64::INFINITY,
            });
        if value < slot.value {
            slot.value = value;
            slot.margin_at_value = margin;
        }
        slot.least_margin = slot.least_margin.min(margin);
        if admitted && value < slot.admitted_value {
            slot.admitted_value = value;
        }
        let chain = CrossingChain {
            value,
            margin,
            pairs,
            end,
        };
        if self
            .report
            .best_any
            .as_ref()
            .is_none_or(|b| value < b.value)
        {
            self.report.best_any = Some(chain.clone());
        }
        if admitted {
            self.report.admitted += 1;
            if self.report.best.as_ref().is_none_or(|b| value < b.value) {
                self.report.best = Some(chain);
            }
        }
    }

    /// The chain ends in a hairpin `key` closes, scored as the outside pass
    /// scores hairpins: every spelling at the special sizes, every corner
    /// otherwise.
    fn hairpins(&mut self, key: PairKey, acc: f64, margin: i32) {
        let s = self.scan;
        let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let eo = s.open_edge(key);
        let bn = key.close;
        if bn.pos < eo.to.pos + MIN_HAIRPIN_LOOP {
            return;
        }
        let size = (bn.pos - eo.to.pos) as usize;
        if !self.spec.only.is_empty() && self.spec.only.len() != self.chain.len() {
            return;
        }
        let base = margin + self.top_beta;
        if matches!(size, 3 | 4 | 6) {
            for (cai, seq) in s.dfa.paths_between(eo.to, bn) {
                let h = HairpinLoop {
                    closing,
                    loop_seq: &seq,
                };
                let e = hairpin_energy(s.params, &h);
                self.score(acc + s.e(e) - s.kk * cai, base + e.0, ChainEnd::Hairpin);
            }
            return;
        }
        for c in s.corners(eo.to, bn) {
            let e = hairpin_energy_scalar(s.params, closing, size, c.first, c.last);
            self.score(acc + s.e(e) - s.kk * c.cai, base + e.0, ChainEnd::Hairpin);
        }
    }

    /// Every single-branch loop `key` closes, as the outside pass scores them.
    fn singles(&mut self, key: PairKey, acc: f64, margin: i32) {
        let s = self.scan;
        let l = s.single_max_len as u32;
        let outer = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let left_from = s.open_edge(key).to;
        let right_to = key.close;
        let q_hi = right_to.pos.saturating_sub(1);
        let q_lo = right_to.pos.saturating_sub(1 + l);
        for q in (q_lo..=q_hi).rev() {
            let n2 = right_to.pos - 1 - q;
            let by_close = self.by_close;
            let cells = &by_close[q as usize];
            let from = cells.partition_point(|(_, k, _)| k.open.pos < left_from.pos);
            for &(xo, xkey, inside) in &cells[from..] {
                let p = xkey.open.pos;
                let n1 = p - left_from.pos;
                if n1 + n2 > l || p >= q {
                    break;
                }
                self.loop_to(
                    (outer, left_from, right_to),
                    (xo, xkey, inside),
                    (n1 as usize, n2 as usize),
                    acc,
                    margin,
                );
            }
        }
    }

    /// The loop from the chain's current pair to enclosed pair `x`, at every
    /// corner choice; recurse when `x` is an S-pair, score the chain otherwise.
    fn loop_to(
        &mut self,
        (outer, left_from, right_to): (BasePair, Node, Node),
        (xo, xkey, inside): (Node, PairKey, f64),
        (n1, n2): (usize, usize),
        acc: f64,
        margin: i32,
    ) {
        let s = self.scan;
        if let Some(&want) = self.spec.only.get(self.chain.len()) {
            if want != (xkey.open.pos, xkey.close.pos) {
                return;
            }
        } else if !self.spec.only.is_empty() {
            return;
        }
        let inner = pair_from_nucleotides(xkey.nuc_open(), xkey.nuc_close);
        let mut options: Vec<(Energy, f64)> = Vec::new();
        if n1 == 0 || n2 == 0 {
            let (Some(lc), Some(rc)) = (s.fill(left_from, xkey.open), s.fill(xo, right_to)) else {
                return;
            };
            let e = if n1 == 0 && n2 == 0 {
                stack_energy(s.params, &StackPair { outer, inner })
            } else {
                bulge_energy(
                    s.params,
                    &BulgeLoop {
                        outer,
                        inner,
                        size: n1 + n2,
                    },
                )
            };
            options.push((e, lc + rc));
        } else {
            let lefts = s.corners(left_from, xkey.open);
            let rights = s.corners(xo, right_to);
            for lc in &lefts {
                for rc in &rights {
                    let e = internal_energy_scalar(
                        s.params,
                        &InternalLoopScalar {
                            outer,
                            inner,
                            boundary_left5p: lc.first,
                            boundary_left3p: lc.last,
                            boundary_right5p: rc.first,
                            boundary_right3p: rc.last,
                            n1,
                            n2,
                        },
                    );
                    options.push((e, lc.cai + rc.cai));
                }
            }
        }
        let crossing = xkey.open.pos <= self.spec.last;
        for (e, cai) in options {
            let v = acc + s.e(e) - s.kk * cai;
            if crossing {
                let Some(stat) = s.stat(xo, xkey) else {
                    continue;
                };
                self.walk(xkey, v + stat, margin + e.0);
            } else {
                let m = margin + e.0 + self.top_beta - s.beta(xkey, self.spec.multiloop);
                self.chain.push((xkey.open.pos, xkey.close.pos));
                self.score(v + inside, m, ChainEnd::Pair);
                self.chain.pop();
            }
        }
    }
}
