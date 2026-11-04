//! Tail-cut rivals as a filter on the exterior recurrence of
//! [`crate::rival_cert::rival_optimum`].
//!
//! # The rival
//!
//! For an item `x` of `len` symbols with own configuration `d` and a cut
//! `t >= 1`, the **tail-cut rival** removes from `d` every pair whose 3′ end
//! lies in the last `t` symbols and folds those `t` symbols together with the
//! suffix context block. The kept part of `d` and the refolded suffix are
//! disjoint spans side by side in the additive outermost loop, so the rival
//! is a genuine configuration of the whole word, and its slack is
//!
//! ```text
//! slack_t = delta_t − gain_t(w_t)
//! ```
//!
//! with `w_t` the last `t` symbols, `gain_t(w) = MFE(suffix) − MFE(w + suffix)`,
//! and `delta_t` the cost of what remains of `d` in the outermost loop less
//! the cost of `d`. A pair `(x, d)` with `slack_t < 0` for some `t` cannot
//! witness tightness.
//!
//! # The recurrence
//!
//! Only the last domain of `d` is touched, and only along its outer chain.
//! For a closed pair `p` and its own sub-configuration, let `delta_t(p)` be the
//! cost of what remains of it in the outermost loop less its cost there
//! (its loops plus its terminal term). If `p` is kept, `delta_t(p) = 0`; if it
//! is removed and closes a degree-2 loop of cost `L` around the pair `q`,
//!
//! ```text
//! delta_t(p) = delta_t(q) + term(q) − term(p) − L
//! ```
//!
//! because `q`'s remains take `p`'s place beside the rest. A removed pair
//! closing a terminal or branching loop is not priced: its class is never
//! killed (`delta = +inf`), which only weakens the filter.
//!
//! The scan's closed-pair cells keep one best derivation each, which cannot
//! carry `delta`. So every cell whose closing symbol lies in the last `T`
//! symbols and whose opening symbol does not (a **tail cell**) is recomputed
//! here from its derivations: terminal loops, branching loops, and degree-2
//! loops around every inner cell of the exact scan. Each derivation becomes a
//! **class**: a lower bound on the scalarized value of every `(x, d)` it
//! stands for, an upper bound on each `delta_t`, and the spelling of the cell's
//! symbols in the tail (unknown loop symbols left open). Classes of one
//! spelling are kept on their Pareto front (lower value, higher `delta`).
//! Lower values and higher `delta`s are both safe: a class is killed only when
//! every `(x, d)` it stands for is.

use std::collections::HashMap;

use rna_mfe::loops::{bulge_energy, pair_from_nucleotides, stack_energy, BulgeLoop, StackPair};
use rna_mfe::params::BasePair;
use rna_mfe::{Energy, Nucleotide, Params};

use crate::dfa::{Dfa, Node, NodeLayout};
use crate::interior::{
    corner_scores, score_hairpin, score_internal_from, Corner, LoopPairs, OneByNRows, ScoredStrand,
};
use crate::state::{Back, PairKey, PairRef, State};
use crate::tables::Tables;
use crate::DEFAULT_SINGLE_MAX_LEN;

/// `delta` of a class that is never killed.
pub const UNKILLABLE: i32 = i32::MAX;

/// Why a class's `delta_t` is [`UNKILLABLE`] at one depth: the innermost step
/// of its derivation that left the removal unpriced. Diagnostic only — no
/// kill test reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unpriced {
    /// `delta_t` is finite.
    Priced,
    /// A removed pair closes a terminal loop.
    Terminal,
    /// A removed pair closes a branching loop whose branches are not counted
    /// (the 3' cut, or a cell with no branch-count table).
    Branching,
    /// A branching loop with [`KMAX`] or more branches.
    BranchCount,
    /// A branching loop whose first branch also lies in the cut region.
    BranchInCut,
    /// No spelling of the cut's `t` symbols gains as much as the removal costs.
    Ceiling,
    /// An inner cell lying wholly in the cut region, taken whole.
    Whole,
    /// The recomputed cell read above the scan's value; kept at the scan's.
    Fallback,
    /// A group over its cap made this class its never-killed floor.
    Floor,
}

impl Unpriced {
    /// Short name for reports.
    pub fn name(self) -> &'static str {
        match self {
            Unpriced::Priced => "priced",
            Unpriced::Terminal => "terminal",
            Unpriced::Branching => "branching",
            Unpriced::BranchCount => "branch-count",
            Unpriced::BranchInCut => "branch-in-cut",
            Unpriced::Ceiling => "ceiling",
            Unpriced::Whole => "whole",
            Unpriced::Fallback => "fallback",
            Unpriced::Floor => "floor",
        }
    }
}

/// A tail symbol not yet spelled: an unmatched symbol inside the cell.
pub const OPEN: u8 = 4;
/// A tail symbol past the cell's closing symbol (spelled by the trailing run).
pub const PAST: u8 = 5;

/// One class of derivations of a tail cell.
#[derive(Debug, Clone)]
pub struct TailClass {
    /// Lower bound on the cell's value; its back pointer is the derivation.
    pub st: State,
    /// Upper bound on `delta_t` for `t = 1..=T` (index `t − 1`).
    pub delta: Vec<i32>,
    /// Per depth, why `delta_t` is [`UNKILLABLE`] ([`Unpriced::Priced`] where
    /// it is finite).
    pub why: Vec<Unpriced>,
    /// Per depth, the upper bound on `delta_t` before the gain ceiling is
    /// applied: equal to `delta_t` except where [`Unpriced::Ceiling`] made it
    /// unkillable, where it says by how much the removal outruns every
    /// spelling's gain. Diagnostic only.
    pub raw: Vec<i32>,
    /// The tail symbols `len − T..len`: a nucleotide, [`OPEN`] or [`PAST`].
    pub spell: Vec<u8>,
    /// For a degree-2 derivation around another tail cell: that cell's class.
    pub inner_class: Option<usize>,
    /// 5' cut only: the invasion rivals of the domain's outer chain (empty: none).
    pub inv: Vec<Inv>,
}

/// One invasion rival of a first domain: its outer chain opened down to a kept
/// pair opening at `a`; the prefix block refolds with the leading run, the freed
/// 5' strand (the item's symbols `[i0, a)`), a stand-in on the kept pair and the
/// freed 3' strand `three`. Kill when `delta` (an upper bound on the domain's
/// break cost) is below that refold's gain.
#[derive(Debug, Clone, PartialEq)]
pub struct Inv {
    /// Upper bound on (kept part's cost in the outermost loop) − (domain's cost).
    pub delta: i32,
    /// Opening position of the kept pair.
    pub a: u32,
    /// The kept pair's symbols.
    pub inner: (u8, u8),
    /// The freed 3' strand, 5' to 3' (from immediately after the kept pair to the
    /// domain's closing symbol).
    pub three: Vec<u8>,
    /// Whether the 3' side is **sealed**: the rival keeps structure between the
    /// kept pair and the domain's closing symbol, so nothing there is free and
    /// the probe carries none of it. A sealed rival scores a probe shorter than
    /// the strand it could have used, which only lowers its gain, and it passes
    /// unchanged through a loop leaving unmatched symbols on its 3' side —
    /// which an open rival cannot, its strand no longer reaching the kept pair.
    pub sealed: bool,
    /// A split whose `delta` holds only where the refold encloses the
    /// branches in a pair ([`TailCut::kills_split`]): removing a branching
    /// loop of `k` branches saves `k` times `ML.intern` that the enclosing
    /// branching loop charges back, so `delta` leaves them both out and holds
    /// for every branch count, the open-ended bucket included.
    pub ml: bool,
}

/// A split refold's gain from `(five, first branch's outer pair, enclosed)`:
/// with `enclosed`, only a refold that encloses the stand-ins in a pair counts.
pub type SplitGain<'a> = dyn FnMut(&[Nucleotide], (Nucleotide, Nucleotide), bool) -> i32 + 'a;

/// An invasion refold's gain from `(five, kept pair, three)`.
pub type InvasionGain<'a> =
    dyn FnMut(&[Nucleotide], (Nucleotide, Nucleotide), &[Nucleotide]) -> i32 + 'a;

/// Branch counts tracked exactly: `0..KMAX`; `KMAX` stands for `KMAX` or more.
const KMAX: usize = 10;

/// The interiors of branching loops opening immediately after one node: for every
/// end node, the least scalarized value of an interior (unmatched symbols and
/// top-level branches, each branch charged its stem cost) by its number of
/// branches `c` and the opening position of its first branch, bucketed
/// `0..depth` (`depth`: at or past the cut region, or no branch yet).
pub struct BranchCounts {
    /// Per node: every live slot.
    best: Vec<Vec<Slot>>,
}

/// One branching-loop interior: how many branches it holds, where its first
/// branch opens (bucketed), that branch's symbols and the interior's least
/// value.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Branch count, exact below [`KMAX`], else [`KMAX`] for "that or more".
    c: u8,
    /// The first branch's opening position, bucketed at the cut depth.
    f: u8,
    /// The first branch's `(opening, closing)` symbols — known only where `f`
    /// is an exact position rather than the past-the-region bucket, which is
    /// also where an invasion rival can be spelled.
    first: Option<(u8, u8)>,
    /// The interior's least value at this count and first branch.
    st: State,
}

/// Keep `cand` unless another slot has at least as many branches, a
/// first branch no later and no higher value: such a slot is at least as hard
/// to kill and stays so after any further branch or unmatched symbol (the count
/// grows in step, the first branch never moves), so the dominated slot can only
/// give classes the dominating one covers.
///
/// A slot that can carry an invasion rival is compared only against slots
/// carrying the same one. That rival's gain is read off the first branch, so a
/// slot whose first branch differs is not *harder* to kill, only differently
/// killed, and dropping this one could drop a class that survives.
fn offer(list: &mut Vec<Slot>, cand: Slot, lambda: f64) {
    let v = cand.st.brd(lambda);
    let same_rival = |s: &Slot| s.first == cand.first && (cand.first.is_none() || s.f == cand.f);
    if list
        .iter()
        .any(|s| same_rival(s) && s.c >= cand.c && s.f <= cand.f && s.st.brd(lambda) <= v + 1e-9)
    {
        return;
    }
    list.retain(|s| {
        !(same_rival(s) && cand.c >= s.c && cand.f <= s.f && v <= s.st.brd(lambda) + 1e-9)
    });
    list.push(cand);
}

impl BranchCounts {
    /// Every live slot of the interior ending at `node`.
    fn slots(&self, node: usize) -> &[Slot] {
        &self.best[node]
    }

    /// The recurrence from `start` over the rest of the item. First-branch
    /// positions are bucketed relative to `anchor`, the cut region's first
    /// position.
    fn new(
        params: &Params,
        dfa: &Dfa,
        tables: &Tables,
        start: Node,
        depth: usize,
        anchor: u32,
    ) -> Self {
        let lambda = tables.lambda();
        let mut best: Vec<Vec<Slot>> = vec![Vec::new(); tables.p.len()];
        best[dfa.ix(start)].push(Slot {
            c: 0,
            f: depth as u8,
            first: None,
            st: State::new(Energy::ZERO, 0.0, Back::ExteriorEmpty),
        });
        for pos in start.pos..=dfa.seq_len {
            for &o in dfa.nodes_at(pos) {
                // Branches closing at `o`.
                let mut add = Vec::new();
                for (key, st) in &tables.p[dfa.ix(o)] {
                    if key.open.pos < start.pos {
                        continue;
                    }
                    let prevs = &best[dfa.ix(key.open)];
                    if prevs.is_empty() {
                        continue;
                    }
                    let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                    let stem = Energy(params.ml.intern_37 + term(params, pair));
                    for &prev in prevs {
                        let (c, f, first) = if prev.c == 0 {
                            let p = (key.open.pos.saturating_sub(anchor) as usize).min(depth);
                            let syms = (key.nuc_open() as u8, key.nuc_close as u8);
                            (1, p as u8, (p < depth).then_some(syms))
                        } else {
                            ((prev.c as usize + 1).min(KMAX) as u8, prev.f, prev.first)
                        };
                        add.push(Slot {
                            c,
                            f,
                            first,
                            st: State::new(
                                prev.st.mfe + st.mfe + stem,
                                prev.st.cai + st.cai,
                                Back::ExteriorEmpty,
                            ),
                        });
                    }
                }
                for cand in add {
                    offer(&mut best[dfa.ix(o)], cand, lambda);
                }
                // Unmatched symbols (no cost in a branching loop).
                let here = best[dfa.ix(o)].clone();
                for e in dfa.right(o) {
                    for &prev in &here {
                        let cand = Slot {
                            st: State::new(prev.st.mfe, prev.st.cai + e.cai, Back::ExteriorEmpty),
                            ..prev
                        };
                        offer(&mut best[dfa.ix(e.to)], cand, lambda);
                    }
                }
            }
        }
        BranchCounts { best }
    }

    /// The 3' mirror: the recurrence run backward from `end` (the node before
    /// a branching loop's closing symbol), so each slot at a node `x` is an
    /// interior spanning `x..end`. Its fixed branch is the **last** one —
    /// the first met going backward — and `f` buckets how deep a 3' cut may
    /// go and keep it: it closes on symbol `p`, a cut of depth `t` removes it
    /// when `p >= len − t`, so it stays for `t <= len − 1 − p` (capped at
    /// `depth`). With `f` read that way, [`offer`]'s dominance and the
    /// removal pricing are the 5' side's unchanged.
    fn new_back(params: &Params, dfa: &Dfa, tables: &Tables, end: Node, depth: usize) -> Self {
        let lambda = tables.lambda();
        let len = dfa.seq_len as usize;
        let mut best: Vec<Vec<Slot>> = vec![Vec::new(); tables.p.len()];
        best[dfa.ix(end)].push(Slot {
            c: 0,
            f: depth as u8,
            first: None,
            st: State::new(Energy::ZERO, 0.0, Back::ExteriorEmpty),
        });
        for pos in (0..=end.pos).rev() {
            for &o in dfa.nodes_at(pos) {
                // Every contribution to `o` comes from a later node, so its
                // slots are final here.
                let here = best[dfa.ix(o)].clone();
                if here.is_empty() {
                    continue;
                }
                // Branches closing at `o`: the interior then starts before them.
                let mut add = Vec::new();
                for (key, st) in &tables.p[dfa.ix(o)] {
                    let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                    let stem = Energy(params.ml.intern_37 + term(params, pair));
                    for &next in &here {
                        let (c, f) = if next.c == 0 {
                            let keep = (len - 1).saturating_sub(key.close.pos as usize);
                            (1, keep.min(depth) as u8)
                        } else {
                            ((next.c as usize + 1).min(KMAX) as u8, next.f)
                        };
                        add.push((
                            key.open,
                            Slot {
                                c,
                                f,
                                first: None,
                                st: State::new(
                                    next.st.mfe + st.mfe + stem,
                                    next.st.cai + st.cai,
                                    Back::ExteriorEmpty,
                                ),
                            },
                        ));
                    }
                }
                for (at, cand) in add {
                    offer(&mut best[dfa.ix(at)], cand, lambda);
                }
                // An unmatched symbol before `o` (no cost in a branching loop).
                for e in dfa.left(o) {
                    for &next in &here {
                        let cand = Slot {
                            st: State::new(next.st.mfe, next.st.cai + e.cai, Back::ExteriorEmpty),
                            ..next
                        };
                        offer(&mut best[dfa.ix(e.to)], cand, lambda);
                    }
                }
            }
        }
        BranchCounts { best }
    }
}

/// The most unspelled symbols a cut window may hold and still be tried.
pub const DEFAULT_MAX_OPEN: usize = 3;

/// The most window spellings a split rival fills (one refold each).
const SPLIT_FILLS: usize = 64;

/// The longest freed 3' strand an invasion rival carries.
const INV_MAX: usize = 16;

/// The most parts one refinement may make ([`TailCut::refine`]).
pub const REFINE_PART_CAP: usize = 256;

/// Why [`TailCut::refine`] left a class whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefineRefusal {
    /// The cell is unpriced or has no such class.
    Missing,
    /// Only the 5' cut is refined: its window starts at the cell's opening
    /// node, so the window's spellings are the paths out of that node.
    TailSide,
    /// The window holds no unspelled symbol: nothing to resolve.
    Spelled,
    /// More window spellings than [`TailCut::refine_cap`]. Refused rather
    /// than truncated: a truncated part set drops real spellings, the one
    /// unsound outcome available here.
    OverCap,
}

/// The tail cells of one scan and their classes.
pub struct TailCut {
    /// The lattice's dense node layout, which the cell map's outer nodes are
    /// indexed by.
    pub layout: NodeLayout,
    /// Cut depth `T`.
    pub depth: usize,
    /// The 5′ mirror: cut the first `t` symbols (pairs whose 5′ end lies there).
    pub head: bool,
    /// Also try each invasion rival against **no context block**, as an own
    /// rival ([`crate::rival_cert::EndGains::own_invasion_gain`]).
    pub own_repair: bool,
    /// Carry the rival that continues past a branching loop: its closing pair
    /// removed, every branch kept where it is and the first branch the
    /// stand-in. **Off by default**, and off no class carries one and nothing
    /// else changes: it is sound, and it binds at no cell yet measured while
    /// costing well over half as much again in wall clock.
    pub branch_rival: bool,
    /// 3' cut: price a removed pair closing a branching loop by its branch
    /// count, as the 5' cut does ([`BranchCounts::new_back`]); off, such a
    /// class is never killed. Set before pricing.
    pub tail_branch: bool,
    /// 5' cut from the item's first symbol: every spelling the item set allows
    /// of the first `depth` symbols (the split rival fills a window's unspelled
    /// symbols from these).
    pub heads: Vec<Vec<u8>>,
    /// 5' cut, with [`Self::branch_rival`]: also kill a first-domain class by
    /// its **split** rival ([`Self::kills_split`]): the domain's outer chain
    /// and its branching loop's closing pair removed, every branch kept, and
    /// both context blocks refolded with the leading run, the freed 5' window
    /// and the branches (on a one-domain item closing in the 3' cut, also the
    /// trailing run). Also records the branch-count-free `ml` split. Off by
    /// default.
    pub split: bool,
    /// 5' cut: on a domain priced by both cuts, also kill a (head class,
    /// tail class) pair when one removal frees both end windows and the two
    /// refolds together gain more than it costs (see
    /// [`crate::rival_cert::rival_optimum`]). Off by default.
    pub joint_ends: bool,
    /// 3' cut: the longest run of exterior unmatched symbols immediately 5' of
    /// the last domain that the **wrap** rival reads (`0`: off). The suffix
    /// block pairs with that run and wraps the whole last domain, which
    /// stays as it is: nothing is removed, so the rival kills whenever the
    /// refold gains anything ([`crate::rival_cert::EndGains::wrap_gain`]).
    pub wrap: usize,
    /// With [`Self::wrap`]: where the domain before the junction is a
    /// 5'-priced first domain, the wrap may also take that domain's freed
    /// closing strand (an unsealed invasion of its class).
    pub wrap_free: bool,
    /// With [`Self::wrap`]: price the domain before the junction by a 3'
    /// cut of this depth anchored at the junction (`0`: off).
    pub wrap_cut: usize,
    /// With [`Self::wrap`] and the junction cut: the **5' wrap**, the mirror
    /// across the same junction. Where the domain before the junction is a
    /// 5'-priced first domain, the prefix block folds with the leading run,
    /// that domain (kept whole, a stand-in on its outer pair) and the junction
    /// run of at most this many symbols (`0`: off); nothing is removed, so the
    /// rival kills whenever the refold gains anything.
    pub wrap5: usize,
    /// With [`Self::wrap5`]'s junctions: price the domain after the junction
    /// by a 5' cut of this depth anchored at the junction's end
    /// ([`Self::lazy_head_at`]); a class dies when its removal costs less
    /// than the 5' wrap refold with its freed window appended (`0`: off).
    pub wrap5_cut: usize,
    /// With [`Self::wrap5`]'s candidates (two domains, every exterior symbol
    /// spelled): also fold **both** context blocks with the whole item, each
    /// domain a stand-in on its outer pair
    /// ([`crate::rival_cert::EndGains::two_domain_gain`]). Nothing is removed,
    /// so any gain kills.
    pub wrap5_both: bool,
    /// With [`Self::wrap5_cut`]: a junction-cut class of the last domain also
    /// dies to its own invasion rivals (outer chain opened to a kept pair),
    /// the freed strands refolded with the leading run, the first domain and
    /// the junction run against the prefix block or both blocks.
    pub wrap5_inv: bool,
    /// 5' cut: invasion rivals also open degree-2 loops that leave unmatched
    /// symbols on their 3' side, recording those symbols as [`OPEN`] in the
    /// freed 3' strand. Such a rival is scored only where another cut spells
    /// them ([`Self::kills_slip`]). 3' cut: keep every spelled window symbol
    /// in the class key, so that cut can spell them. Off by default.
    pub slip: bool,
    /// The most unspelled symbols a cut window may hold and still be tried.
    /// A window with more is refused outright, so no fill of it is ever
    /// scored and the class cannot be killed however cheap its removal is.
    /// Each extra one costs a factor of four in fills.
    pub max_open: usize,
    /// Override the per-spelling class cap ([`GROUP_CAP`] / [`HEAD_GROUP_CAP`]).
    /// A group over the cap turns its highest-valued killable class into a
    /// never-killed floor, so the cap decides how many classes stay killable:
    /// raising it is the direct test of whether a bound is held up by class
    /// width rather than by a missing rival.
    pub group_cap: Option<usize>,
    /// The most parts one refinement may make.
    pub refine_cap: usize,
    /// 5' cut: the most [`OPEN`] symbols an invasion rival's freed 3'
    /// strand may hold and still be recorded (one with more is never tested,
    /// and keeping it splits classes for nothing).
    pub inv_open_cap: usize,
    /// 5' cut: price a removed pair closing a branching loop by its branch
    /// count ([`BranchCounts::new`]); off, such a class is never killed.
    /// Set before pricing.
    pub head_branch: bool,
    ceiling: Vec<i32>,
    len: u32,
    /// 5' cut: the cut region's first position (`0`, or a junction's end for
    /// a cut anchored there, [`Self::lazy_head_at`]).
    start: u32,
    cells: HashMap<(usize, PairKey), Vec<TailClass>>,
    by_open: HashMap<(usize, u32), Vec<(PairKey, State)>>,
    /// 5' cut: branching-loop interiors by branch count, per start node.
    multi: HashMap<usize, BranchCounts>,
    /// Cells whose recomputed least value exceeded the scan's (kept as one
    /// never-killed class at the scan's value).
    pub fallbacks: usize,
    /// Cells recomputed.
    pub recomputed: usize,
    /// Branching-loop invasion rivals built.
    pub sealed_rivals: usize,
    /// Of those, the ones whose stand-in the cell's symbols reach.
    pub sealed_spellable: usize,
}

fn term(params: &Params, pair: BasePair) -> i32 {
    if pair.applies_terminal_au() {
        params.misc.terminal_au_37
    } else {
        0
    }
}

/// `b` is at least as good as `a` in every coordinate (value no higher,
/// every `delta` no lower), so `b` may stand for `a`: every way `b` can be
/// killed must kill `a` too. An invasion record is such a way, so each of
/// `b`'s records needs one of `a`'s on the same probe that is no harder to
/// kill (`delta` no higher); records `a` has beyond those only make `a`
/// easier to kill.
fn dominates(b: &TailClass, a: &TailClass, lambda: f64) -> bool {
    b.st.brd(lambda) <= a.st.brd(lambda) + 1e-9
        && b.delta.iter().zip(&a.delta).all(|(x, y)| x >= y)
        && b.inv.iter().all(|y| {
            a.inv.iter().any(|x| {
                x.a == y.a
                    && x.inner == y.inner
                    && x.three == y.three
                    && x.sealed == y.sealed
                    && x.ml == y.ml
                    && x.delta <= y.delta
            })
        })
}

/// Classes of one cell, grouped by spelling, each group on its Pareto front.
type Groups = HashMap<Vec<u8>, Vec<TailClass>>;

/// The most killable classes a group keeps.
const GROUP_CAP: usize = 16;

/// The cap on the lazily priced 5' side, whose few cells carry branching-loop
/// classes by branch count and first-branch position.
const HEAD_GROUP_CAP: usize = 256;

/// Add `c` to its spelling's group. A group over [`GROUP_CAP`] turns its
/// highest-valued killable class into a never-killed **floor**: its value is
/// the least of every class dropped so far, a lower bound on all of them,
/// and as a floor it dominates (prunes) every class above it.
fn push_pareto(cap: usize, groups: &mut Groups, c: TailClass, lambda: f64) {
    let list = groups.entry(c.spell.clone()).or_default();
    if list.iter().any(|o| dominates(o, &c, lambda)) {
        return;
    }
    list.retain(|o| !dominates(&c, o, lambda));
    list.push(c);
    if list.len() > cap {
        let worst = (0..list.len())
            .filter(|&i| list[i].delta.iter().any(|&d| d != UNKILLABLE) || !list[i].inv.is_empty())
            .max_by(|&a, &b| list[a].st.brd(lambda).total_cmp(&list[b].st.brd(lambda)));
        if let Some(w) = worst {
            let mut floor = list.swap_remove(w);
            // The floor stands for every class it dropped, not its own
            // derivation, so no depth keeps that derivation's reason.
            floor.delta.iter_mut().for_each(|d| *d = UNKILLABLE);
            floor.why.iter_mut().for_each(|w| *w = Unpriced::Floor);
            floor.raw.iter_mut().for_each(|r| *r = UNKILLABLE);
            floor.inv.clear();
            list.retain(|o| !dominates(&floor, o, lambda));
            list.push(floor);
        }
    }
}

impl TailCut {
    /// Whether the cell `(o, key)` is a tail cell: closing symbol in the last
    /// `T` symbols, opening symbol before them.
    pub fn is_tail(&self, key: &PairKey) -> bool {
        if self.head {
            let d = self.start + self.depth as u32;
            key.open.pos >= self.start && key.open.pos < d && key.close.pos >= d
        } else {
            let first = self.len - self.depth as u32;
            key.close.pos >= first && key.open.pos < first
        }
    }

    /// Whether the cut of depth `t` removes the pair opening at `open` and
    /// closing at `close`.
    fn removed(&self, open: u32, close: u32, t: usize) -> bool {
        if self.head {
            open < self.start + t as u32
        } else {
            close >= self.len - t as u32
        }
    }

    /// Whether the cell `(o, key)` has been priced (its classes stand in for
    /// the scan's cell in the recurrence).
    pub fn priced(&self, o: Node, key: &PairKey) -> bool {
        self.cells.contains_key(&(self.layout.ix(o), *key))
    }

    /// The classes of a tail cell.
    pub fn classes(&self, o: Node, key: &PairKey) -> Option<&[TailClass]> {
        self.cells
            .get(&(self.layout.ix(o), *key))
            .map(Vec::as_slice)
    }

    /// The upper bound on any spelling's gain at depth `t` (`i32::MAX`:
    /// unknown).
    pub fn ceiling(&self, t: usize) -> i32 {
        self.ceiling[t - 1]
    }

    /// The first position of the cut region: `len − T` (3′) or its anchor,
    /// `0` unless anchored at a junction (5′).
    pub fn first(&self) -> u32 {
        if self.head {
            self.start
        } else {
            self.len - self.depth as u32
        }
    }

    /// Recompute every tail cell of `tables` (an exact scan of `dfa`).
    ///
    /// `ceiling[t − 1]` is an upper bound on the gain of any spelling of the
    /// cut's `t` symbols (`i32::MAX` when unknown): a class whose `delta_t`
    /// reaches it can never be killed at `t` and is recorded as such, which
    /// merges classes that differ only there.
    pub fn new(
        params: &Params,
        dfa: &Dfa,
        tables: &Tables,
        depth: usize,
        head: bool,
        ceiling: &[i32],
    ) -> Self {
        let mut cut = Self::lazy(dfa, tables, depth, head, ceiling);
        cut.price_all(params, dfa, tables);
        cut
    }

    /// Price every cut cell of `tables` not priced yet.
    pub fn price_all(&mut self, params: &Params, dfa: &Dfa, tables: &Tables) {
        let o_lo = if self.head {
            self.start + self.depth as u32 + 1
        } else {
            self.first() + 1
        };
        for pos in o_lo..=self.len {
            for &o in dfa.nodes_at(pos) {
                let keys: Vec<PairKey> = tables.p[dfa.ix(o)]
                    .keys()
                    .filter(|k| self.is_tail(k))
                    .copied()
                    .collect();
                for key in keys {
                    self.price(params, dfa, tables, o, key);
                }
            }
        }
    }

    /// A cut with no cell priced yet: [`Self::price`] prices cells on demand
    /// (with every cut cell inside them), and the recurrence treats an
    /// unpriced cell as never killed.
    pub fn lazy(dfa: &Dfa, tables: &Tables, depth: usize, head: bool, ceiling: &[i32]) -> Self {
        Self::lazy_at(dfa, tables, depth, head, ceiling, dfa.seq_len)
    }

    /// [`Self::lazy`] with the item taken to end at `len` symbols: a 3' cut
    /// of the domains closing immediately before position `len` (a junction cut).
    pub fn lazy_at(
        dfa: &Dfa,
        tables: &Tables,
        depth: usize,
        head: bool,
        ceiling: &[i32],
        len: u32,
    ) -> Self {
        Self::build(dfa, tables, depth, head, ceiling, len, 0)
    }

    /// A 5' cut whose region starts at `start` rather than at the item's
    /// first symbol: the cut of the domains opening at the end of a junction
    /// run (a 5' junction cut), the mirror of [`Self::lazy_at`]. No gain
    /// ceiling, and a branching loop's removal is left unpriced
    /// ([`Self::head_branch`] off): its branch-count table would be a pass
    /// over the rest of the item per opening node.
    pub fn lazy_head_at(dfa: &Dfa, tables: &Tables, depth: usize, start: u32) -> Self {
        let mut cut = Self::build(dfa, tables, depth, true, &[], dfa.seq_len, start);
        cut.head_branch = false;
        cut
    }

    fn build(
        dfa: &Dfa,
        tables: &Tables,
        depth: usize,
        head: bool,
        ceiling: &[i32],
        len: u32,
        start: u32,
    ) -> Self {
        let depth = depth.min((len - start) as usize).max(1);
        let first = len - depth as u32;
        let cap = DEFAULT_SINGLE_MAX_LEN as u32;
        // Inner cells near the cut region, by (outer node, opening position).
        let lo = if head {
            start
        } else {
            first.saturating_sub(cap + 1)
        };
        let open_hi = if head {
            start + depth as u32 + cap + 1
        } else {
            len
        };
        let mut by_open: HashMap<(usize, u32), Vec<(PairKey, State)>> = HashMap::new();
        for pos in lo..=len {
            for &io in dfa.nodes_at(pos) {
                for (key, st) in &tables.p[dfa.ix(io)] {
                    if key.open.pos > open_hi {
                        continue;
                    }
                    by_open
                        .entry((dfa.ix(io), key.open.pos))
                        .or_default()
                        .push((*key, *st));
                }
            }
        }
        let heads: Vec<Vec<u8>> = if head && start == 0 {
            dfa.nodes_at(depth as u32)
                .iter()
                .flat_map(|&v| dfa.paths_between(dfa.start(), v))
                .map(|(_, n)| n.iter().map(|&x| x as u8).collect())
                .collect()
        } else {
            Vec::new()
        };
        TailCut {
            layout: dfa.layout(),
            ceiling: (0..depth)
                .map(|t| ceiling.get(t).copied().unwrap_or(i32::MAX))
                .collect(),
            depth,
            head,
            heads,
            own_repair: false,
            branch_rival: false,
            split: false,
            tail_branch: false,
            joint_ends: false,
            wrap: 0,
            wrap_free: false,
            wrap_cut: 0,
            wrap5: 0,
            wrap5_cut: 0,
            wrap5_both: false,
            wrap5_inv: false,
            slip: false,
            max_open: DEFAULT_MAX_OPEN,
            group_cap: None,
            refine_cap: REFINE_PART_CAP,
            inv_open_cap: usize::MAX,
            head_branch: true,
            len,
            start,
            cells: HashMap::new(),
            by_open,
            multi: HashMap::new(),
            fallbacks: 0,
            recomputed: 0,
            sealed_rivals: 0,
            sealed_spellable: 0,
        }
    }

    /// Price the cut cell `(o, key)` and, first, every cut cell inside it.
    pub fn price(&mut self, params: &Params, dfa: &Dfa, tables: &Tables, o: Node, key: PairKey) {
        if !self.is_tail(&key) || self.cells.contains_key(&(dfa.ix(o), key)) {
            return;
        }
        let Some(&best) = tables.p[dfa.ix(o)].get(&key) else {
            return;
        };
        for (io, ikey) in self.inner_cells(dfa, &key) {
            self.price(params, dfa, tables, io, ikey);
        }
        if self.head && tables.multi[dfa.ix(o)].contains_key(&key) {
            if let (true, Some(eo)) = (self.head_branch, key.open_edge(dfa)) {
                let (depth, anchor) = (self.depth, self.start);
                self.multi.entry(dfa.ix(eo.to)).or_insert_with(|| {
                    BranchCounts::new(params, dfa, tables, eo.to, depth, anchor)
                });
            }
        } else if self.tail_branch && tables.multi[dfa.ix(o)].contains_key(&key) {
            // Keyed by the loop's last interior node; one backward pass serves
            // every cell closing there.
            let depth = self.depth;
            self.multi
                .entry(dfa.ix(key.close))
                .or_insert_with(|| BranchCounts::new_back(params, dfa, tables, key.close, depth));
        }
        let lambda = tables.lambda();
        let mut list = self.derive(params, dfa, tables, o, key);
        let least = list
            .iter()
            .map(|c| c.st.brd(lambda))
            .fold(f64::INFINITY, f64::min);
        if least > best.brd(lambda) + 1e-6 {
            self.fallbacks += 1;
            list.push(TailClass {
                st: best,
                delta: vec![UNKILLABLE; self.depth],
                why: vec![Unpriced::Fallback; self.depth],
                raw: vec![UNKILLABLE; self.depth],
                spell: self.cell_spell(&key),
                inner_class: None,
                inv: Vec::new(),
            });
        }
        self.recomputed += 1;
        // Non-vacuity: a rival that is never built explains nothing, and one
        // built but never spellable explains no more. A sealed rival reads the
        // cell's symbols from its opening to the stand-in, which a branching
        // loop leaves [`OPEN`] wherever it holds unmatched symbols there, so
        // the two counts come apart and both are reported.
        for c in &list {
            for x in c.inv.iter().filter(|x| x.sealed) {
                self.sealed_rivals += 1;
                // Before the cell's opening the leading run spells it.
                let spelled = (key.open.pos as usize..x.a as usize)
                    .all(|p| c.spell.get(p).is_some_and(|&v| v < 4));
                self.sealed_spellable += usize::from(spelled);
            }
        }
        self.cells.insert((dfa.ix(o), key), list);
    }

    /// The cut cells a degree-2 loop closed by `key` may enclose.
    fn inner_cells(&self, dfa: &Dfa, key: &PairKey) -> Vec<(Node, PairKey)> {
        let cap = DEFAULT_SINGLE_MAX_LEN as u32;
        let (i, j) = (key.open.pos, key.close.pos);
        let mut out = Vec::new();
        for io_pos in (i + 2)..=j {
            let n2 = j - io_pos;
            if n2 > cap {
                continue;
            }
            for &io in dfa.nodes_at(io_pos) {
                for k in (i + 1)..io_pos {
                    if (k - (i + 1)) + n2 > cap {
                        break;
                    }
                    for (ikey, _) in self.by_open.get(&(dfa.ix(io), k)).into_iter().flatten() {
                        if self.is_tail(ikey) {
                            out.push((io, *ikey));
                        }
                    }
                }
            }
        }
        out
    }

    /// The tail spelling known from the cell's own pair alone: its closing
    /// symbol; every other tail symbol inside the cell open, past it [`PAST`].
    fn cell_spell(&self, key: &PairKey) -> Vec<u8> {
        let first = self.first();
        (0..self.depth as u32)
            .map(|i| {
                let p = first + i;
                if p == key.close.pos {
                    key.nuc_close as u8
                } else if p == key.open.pos {
                    key.nuc_open() as u8
                } else if p > key.close.pos || p < key.open.pos {
                    PAST
                } else {
                    OPEN
                }
            })
            .collect()
    }

    /// Every class of the tail cell `(o, key)`.
    fn derive(
        &self,
        params: &Params,
        dfa: &Dfa,
        tables: &Tables,
        o: Node,
        key: PairKey,
    ) -> Vec<TailClass> {
        let lambda = tables.lambda();
        // Which `1xn` loops score the same at every corner, so their corners
        // are compared by adaptation alone.
        let one_by_n = OneByNRows::of(params);
        let depth = self.depth;
        let first = self.first();
        let group_cap =
            self.group_cap
                .unwrap_or(if self.head { HEAD_GROUP_CAP } else { GROUP_CAP });
        let list = Vec::new();
        let mut groups = Groups::new();
        let (Some(eo), Some(ec)) = (key.open_edge(dfa), key.close_edge(dfa, o)) else {
            return list;
        };
        let outer = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        if outer == BasePair::NN {
            return list;
        }
        let base_spell = self.cell_spell(&key);
        let unkillable = |st: State, why: Unpriced| TailClass {
            st,
            delta: vec![UNKILLABLE; depth],
            why: vec![why; depth],
            raw: vec![UNKILLABLE; depth],
            spell: base_spell.clone(),
            inner_class: None,
            inv: Vec::new(),
        };
        // Terminal loop.
        if key.close.pos >= eo.to.pos + 3 {
            if let Some((e, icai)) = score_hairpin(params, dfa, lambda, eo.to, key.close, outer) {
                let c = self.normalized(unkillable(
                    State::new(e, eo.cai + icai + ec.cai, Back::Hairpin),
                    Unpriced::Terminal,
                ));
                push_pareto(group_cap, &mut groups, c, lambda);
            }
        }
        // Branching loop, by branch count (the 5' cut always, the 3' cut with
        // `tail_branch`) — removing the closing pair of a loop whose k branches
        // all stay changes the cost by
        // -closing - (k + 1) * intern - 2 * AU (the branches' AU terms cancel).
        // The interiors by branch count: the 5' table starts after the
        // opening symbol and is read at the loop's last node; the 3' table
        // ends there and is read after the opening symbol.
        let (counted, read_at) = if self.head {
            (self.multi.get(&dfa.ix(eo.to)), key.close)
        } else if self.tail_branch {
            (self.multi.get(&dfa.ix(key.close)), eo.to)
        } else {
            (None, key.close)
        };
        if let (Some(_), Some(bc)) = (tables.multi[dfa.ix(o)].get(&key), counted) {
            let close = Energy(params.ml.closing_37 + params.ml.intern_37 + term(params, outer));
            let t_out = term(params, outer);
            for slot in bc.slots(dfa.ix(read_at)) {
                let (c, f) = (slot.c as usize, slot.f as usize);
                if !(2..=KMAX).contains(&c) {
                    continue;
                }
                let st = State::new(
                    slot.st.mfe + close,
                    slot.st.cai + eo.cai + ec.cai,
                    Back::MultiClose {
                        multi: PairRef { outer: o, key },
                    },
                );
                // What removing the closing pair costs when every branch stays.
                let removal =
                    -params.ml.closing_37 - (c as i32 + 1) * params.ml.intern_37 - 2 * t_out;
                let (delta, why): (Vec<i32>, Vec<Unpriced>) = (1..=depth)
                    .map(|t| {
                        if !self.removed(key.open.pos, key.close.pos, t) {
                            (0, Unpriced::Priced)
                        } else if c >= KMAX {
                            (UNKILLABLE, Unpriced::BranchCount)
                        } else if t > f {
                            (UNKILLABLE, Unpriced::BranchInCut)
                        } else if removal >= self.ceiling[t - 1] {
                            // No spelling gains as much: never killed at t.
                            (UNKILLABLE, Unpriced::Ceiling)
                        } else {
                            (removal, Unpriced::Priced)
                        }
                    })
                    .unzip();
                let raw: Vec<i32> = delta
                    .iter()
                    .zip(&why)
                    .map(|(&d, &w)| if w == Unpriced::Ceiling { removal } else { d })
                    .collect();
                // The invasion that continues past this loop: remove the
                // closing pair, keep every branch where it is, and refold the
                // freed 5' side against the first branch as the stand-in. The
                // other branches lie between that one and the domain's closing
                // symbol, so the 3' side is sealed. Only where the count is
                // exact — one more branch costs another `intern` to remove, so
                // the open-ended bucket bounds nothing.
                let mut inv = match slot.first {
                    Some(inner) if c < KMAX && self.branch_rival => vec![Inv {
                        delta: removal,
                        a: f as u32,
                        inner,
                        three: Vec::new(),
                        sealed: true,
                        ml: false,
                    }],
                    _ => Vec::new(),
                };
                // With the branches enclosed again in a branching loop, each
                // pays back the `intern` its removal saved: `delta` at two
                // branches, whatever the count. The refold reads no branch's
                // symbols (a stand-in in a branching loop adds only its own
                // terminal term), so a first branch past the window is fine:
                // the window ends the spelled part and two GC stand-ins follow.
                if self.split {
                    let inner = slot
                        .first
                        .unwrap_or((Nucleotide::G as u8, Nucleotide::C as u8));
                    inv.push(Inv {
                        delta: -params.ml.closing_37 - 3 * params.ml.intern_37 - 2 * t_out,
                        a: f as u32,
                        inner,
                        three: Vec::new(),
                        sealed: true,
                        ml: true,
                    });
                }
                let class = self.normalized(TailClass {
                    st,
                    delta,
                    why,
                    raw,
                    spell: base_spell.clone(),
                    inner_class: None,
                    inv,
                });
                push_pareto(group_cap, &mut groups, class, lambda);
            }
        } else if let Some(m) = tables.multi[dfa.ix(o)].get(&key) {
            let close = Energy(params.ml.closing_37 + params.ml.intern_37 + term(params, outer));
            let c = self.normalized(unkillable(
                State::new(
                    m.mfe + close,
                    m.cai,
                    Back::MultiClose {
                        multi: PairRef { outer: o, key },
                    },
                ),
                Unpriced::Branching,
            ));
            push_pareto(group_cap, &mut groups, c, lambda);
        }
        // Degree-2 loops around every inner cell.
        let (i, j) = (key.open.pos, key.close.pos);
        let cap = DEFAULT_SINGLE_MAX_LEN as u32;
        let t_out = term(params, outer);
        for io_pos in (i + 2)..=j {
            let n2 = j - io_pos;
            if n2 > cap {
                continue;
            }
            for &io in dfa.nodes_at(io_pos) {
                let Some(right_cai) = dfa.broken_codon_cai(io, key.close) else {
                    continue;
                };
                let rights = (n2 > 0).then(|| corner_scores(dfa, io, key.close));
                for k in (i + 1)..io_pos {
                    let n1 = k - (i + 1);
                    if n1 + n2 > cap {
                        break;
                    }
                    let Some(inners) = self.by_open.get(&(dfa.ix(io), k)) else {
                        continue;
                    };
                    for &(ikey, ist) in inners {
                        let inner = pair_from_nucleotides(ikey.nuc_open(), ikey.nuc_close);
                        if inner == BasePair::NN {
                            continue;
                        }
                        let Some(left_cai) = dfa.broken_codon_cai(eo.to, ikey.open) else {
                            continue;
                        };
                        // A loop strand inside the cut region is spelled path by
                        // path (its symbols then read as known by the kill test, and
                        // the loop energy is exact per spelling); otherwise the
                        // strand's best corners stand for every spelling.
                        // The strand meets the cut region and is short enough to list.
                        const SPELL_MAX: u32 = 8;
                        let region = if self.head {
                            n1 > 0 && n1 <= SPELL_MAX && eo.to.pos < first + depth as u32
                        } else {
                            n2 > 0 && n2 <= SPELL_MAX && j > first
                        };
                        let strands: Vec<Option<(f64, Vec<Nucleotide>)>> = if region {
                            let paths = if self.head {
                                dfa.paths_between(eo.to, ikey.open)
                            } else {
                                dfa.paths_between(io, key.close)
                            };
                            paths.into_iter().map(Some).collect()
                        } else {
                            vec![None]
                        };
                        // (value energy, value cai, least loop energy, spelled strand)
                        let mut variants: Vec<(Energy, f64, Energy, Option<Vec<Nucleotide>>)> =
                            Vec::new();
                        for strand in strands {
                            let (lc, rc) = match &strand {
                                Some((pc, _)) if self.head => (*pc, right_cai),
                                Some((pc, _)) => (left_cai, *pc),
                                None => (left_cai, right_cai),
                            };
                            if n1 == 0 || n2 == 0 {
                                let e = if n1 == 0 && n2 == 0 {
                                    stack_energy(params, &StackPair { outer, inner })
                                } else {
                                    bulge_energy(
                                        params,
                                        &BulgeLoop {
                                            outer,
                                            inner,
                                            size: (n1 + n2) as usize,
                                        },
                                    )
                                };
                                variants.push((e, lc + rc, e, strand.map(|(_, n)| n)));
                                continue;
                            }
                            let spelled = strand.as_ref().map(|(pc, n)| {
                                vec![Corner {
                                    cai: *pc,
                                    first: n[0],
                                    last: n[n.len() - 1],
                                }]
                            });
                            let lefts = match (&spelled, self.head) {
                                (Some(c), true) => c.clone(),
                                _ => corner_scores(dfa, eo.to, ikey.open),
                            };
                            let rights_v = match (&spelled, self.head) {
                                (Some(c), false) => c.clone(),
                                _ => rights.clone().unwrap_or_default(),
                            };
                            let pairs = LoopPairs { outer, inner };
                            let ls = ScoredStrand {
                                corners: &lefts,
                                len: n1 as usize,
                            };
                            let rs = ScoredStrand {
                                corners: &rights_v,
                                len: n2 as usize,
                            };
                            let (Some((e, si_cai)), Some((e_min, _))) = (
                                score_internal_from(params, lambda, &pairs, (&ls, &rs), &one_by_n),
                                score_internal_from(params, 1.0, &pairs, (&ls, &rs), &one_by_n),
                            ) else {
                                continue;
                            };
                            variants.push((e, si_cai, e_min, strand.map(|(_, n)| n)));
                        }
                        for (e, cai, e_min, strand) in variants {
                            let back = Back::Single {
                                inner: PairRef {
                                    outer: io,
                                    key: ikey,
                                },
                            };
                            let loop_cai = eo.cai + cai + ec.cai;
                            let t_in = term(params, inner);
                            let l = ikey.close.pos;
                            // The outer pair's own spelling, with the loop's
                            // unmatched tail symbols open.
                            let mut spell = base_spell.clone();
                            for (idx, s) in spell.iter_mut().enumerate() {
                                let p = first + idx as u32;
                                if p >= ikey.open.pos && p <= l {
                                    *s = PAST; // filled from the inner below
                                }
                            }
                            if let Some(nucs) = &strand {
                                let from = if self.head { eo.to.pos } else { io.pos };
                                for (k, nuc) in nucs.iter().enumerate() {
                                    let p = from + k as u32;
                                    if p >= first && p < first + depth as u32 {
                                        spell[(p - first) as usize] = *nuc as u8;
                                    }
                                }
                            }
                            let combine = |inner_delta: &[i32],
                                           inner_why: &[Unpriced],
                                           inner_raw: &[i32],
                                           inner_spell: &[u8],
                                           st: State| {
                                let raw: Vec<i32> = (1..=depth)
                                    .map(|t| {
                                        if !self.removed(i, j, t) {
                                            0
                                        } else if inner_raw[t - 1] == UNKILLABLE {
                                            UNKILLABLE
                                        } else {
                                            inner_raw[t - 1] + t_in - t_out - e_min.0
                                        }
                                    })
                                    .collect();
                                let (delta, why): (Vec<i32>, Vec<Unpriced>) = (1..=depth)
                                    .map(|t| {
                                        if !self.removed(i, j, t) {
                                            (0, Unpriced::Priced)
                                        } else if inner_delta[t - 1] == UNKILLABLE {
                                            (UNKILLABLE, inner_why[t - 1])
                                        } else {
                                            let d = inner_delta[t - 1] + t_in - t_out - e_min.0;
                                            // No spelling gains as much: never killed at t.
                                            if d >= self.ceiling[t - 1] {
                                                (UNKILLABLE, Unpriced::Ceiling)
                                            } else {
                                                (d, Unpriced::Priced)
                                            }
                                        }
                                    })
                                    .unzip();
                                let mut s = spell.clone();
                                for (idx, v) in s.iter_mut().enumerate() {
                                    let p = first + idx as u32;
                                    if p >= ikey.open.pos && p <= l {
                                        *v = inner_spell[idx];
                                    }
                                }
                                (delta, why, raw, s, st)
                            };
                            // Invasions through this loop (5' cut only, no unmatched symbol
                            // on its 3' side, or any with `slip`, those symbols left
                            // open): open this pair and keep the inner one, or extend
                            // the inner class's invasions by this pair.
                            let step = t_in - t_out - e_min.0;
                            let through = n2 == 0 || self.slip;
                            // The loop's 3' side as the freed strand reads it.
                            let side = |three: &mut Vec<u8>| {
                                three.extend(std::iter::repeat_n(OPEN, n2 as usize));
                                three.push(key.nuc_close as u8);
                            };
                            let invasions = |inner_inv: &[Inv]| -> Vec<Inv> {
                                if !self.head {
                                    return Vec::new();
                                }
                                let mut out = Vec::new();
                                if through
                                    && ikey.open.pos <= first + depth as u32
                                    && (n2 as usize) < INV_MAX
                                {
                                    let mut three = Vec::new();
                                    side(&mut three);
                                    out.push(Inv {
                                        delta: step,
                                        a: ikey.open.pos,
                                        inner: (ikey.nuc_open() as u8, ikey.nuc_close as u8),
                                        three,
                                        sealed: false,
                                        ml: false,
                                    });
                                }
                                for x in inner_inv {
                                    if x.delta == UNKILLABLE {
                                        continue;
                                    }
                                    if x.sealed {
                                        // Nothing 3' of the kept pair is read,
                                        // so this loop's own unmatched symbols
                                        // there change nothing.
                                        out.push(Inv {
                                            delta: x.delta + step,
                                            a: x.a,
                                            inner: x.inner,
                                            three: Vec::new(),
                                            sealed: true,
                                            ml: x.ml,
                                        });
                                    } else if through && x.three.len() + (n2 as usize) < INV_MAX {
                                        let mut three = x.three.clone();
                                        side(&mut three);
                                        out.push(Inv {
                                            delta: x.delta + step,
                                            a: x.a,
                                            inner: x.inner,
                                            three,
                                            sealed: false,
                                            ml: false,
                                        });
                                    }
                                }
                                // A rival with more open symbols than its test
                                // fills could never kill: not recorded.
                                out.retain(|x| {
                                    x.three.iter().filter(|&&v| v == OPEN).count()
                                        <= self.inv_open_cap
                                });
                                out
                            };
                            // An unpriced inner cell is taken whole (never killed once removed).
                            let priced = self
                                .is_tail(&ikey)
                                .then(|| self.cells.get(&(dfa.ix(io), ikey)))
                                .flatten();
                            if let Some(classes) = priced {
                                for (ci, c) in classes.iter().enumerate() {
                                    let st = State::new(c.st.mfe + e, c.st.cai + loop_cai, back);
                                    let (delta, why, raw, s, st) =
                                        combine(&c.delta, &c.why, &c.raw, &c.spell, st);
                                    let inv = invasions(&c.inv);
                                    push_pareto(
                                        group_cap,
                                        &mut groups,
                                        self.normalized(TailClass {
                                            st,
                                            delta,
                                            why,
                                            raw,
                                            spell: s,
                                            inner_class: Some(ci),
                                            inv,
                                        }),
                                        lambda,
                                    );
                                }
                            } else {
                                // A kept inner cell, or one wholly in the tail
                                // (never killed once removed).
                                let (inner_delta, inner_why): (Vec<i32>, Vec<Unpriced>) = (1
                                    ..=depth)
                                    .map(|t| {
                                        if self.removed(ikey.open.pos, l, t) {
                                            (UNKILLABLE, Unpriced::Whole)
                                        } else {
                                            (0, Unpriced::Priced)
                                        }
                                    })
                                    .unzip();
                                let inner_spell = self.cell_spell(&ikey);
                                let st = State::new(ist.mfe + e, ist.cai + loop_cai, back);
                                let (delta, why, raw, s, st) = combine(
                                    &inner_delta,
                                    &inner_why,
                                    &inner_delta,
                                    &inner_spell,
                                    st,
                                );
                                let inv = invasions(&[]);
                                push_pareto(
                                    group_cap,
                                    &mut groups,
                                    self.normalized(TailClass {
                                        st,
                                        delta,
                                        why,
                                        raw,
                                        spell: s,
                                        inner_class: None,
                                        inv,
                                    }),
                                    lambda,
                                );
                            }
                        }
                    }
                }
            }
        }
        let mut list = list;
        list.extend(groups.into_values().flatten());
        list
    }

    /// `c` with every cut symbol no cut can still kill it at left open, so
    /// classes that differ only there share a group. Sound through the
    /// recurrence: an outer class is killable at `t` only where its inner
    /// class is.
    fn normalized(&self, mut c: TailClass) -> TailClass {
        let reach = (1..=self.depth)
            .rev()
            .find(|&t| c.delta[t - 1] != UNKILLABLE)
            .unwrap_or(0)
            .max(
                c.inv
                    .iter()
                    .map(|x| x.a.saturating_sub(self.first()) as usize)
                    .max()
                    .unwrap_or(0),
            );
        for (idx, v) in c.spell.iter_mut().enumerate() {
            // A cut of depth t reads positions 0..t (5') or depth - t..depth (3').
            // With `slip` a 3' cut's window is also read by the head side's
            // invasion rivals, at any depth.
            let read = if self.head {
                idx < reach
            } else {
                self.slip || idx >= self.depth - reach
            };
            if !read && *v != PAST {
                *v = OPEN;
            }
        }
        c
    }

    /// Per cut depth, why [`Self::kills`] does or does not kill `c`: the
    /// class's `delta_t`, the least gain over every fill of what the window
    /// leaves open, and the count of open symbols -- or the reason the depth
    /// was skipped before any gain was scored.
    ///
    /// `run` must be the cell's actual leading run. A shorter one makes the
    /// window read the spell where the run belongs, where it finds [`PAST`]
    /// and refuses, so every depth reports `past` and the class looks
    /// unkillable when it is not.
    pub fn kills_trace(
        &self,
        c: &TailClass,
        run: &[Nucleotide],
        gain: &mut dyn FnMut(&[Nucleotide]) -> i32,
        max_open: usize,
    ) -> Vec<(usize, i32, Option<i32>, usize, &'static str)> {
        let first = self.first();
        let mut out = Vec::new();
        for t in 1..=self.depth {
            let d = c.delta[t - 1];
            if d == UNKILLABLE {
                out.push((t, d, None, 0, "unkillable"));
                continue;
            }
            let mut w: Vec<u8> = Vec::with_capacity(t);
            if self.head {
                for p in 0..t {
                    w.push(if p < run.len() {
                        run[p] as u8
                    } else {
                        c.spell[p]
                    });
                }
            } else {
                let start = self.len as usize - t;
                let run_from = self.len as usize - run.len();
                for p in start..self.len as usize {
                    if p >= run_from {
                        w.push(run[p - run_from] as u8);
                    } else {
                        w.push(c.spell[p - first as usize]);
                    }
                }
            }
            if w.contains(&PAST) {
                out.push((t, d, None, 0, "past"));
                continue;
            }
            let open: Vec<usize> = (0..w.len()).filter(|&i| w[i] == OPEN).collect();
            if open.len() > max_open {
                out.push((t, d, None, open.len(), "too-open"));
                continue;
            }
            let mut least = i32::MAX;
            for fill in 0..(1usize << (2 * open.len())) {
                let mut sp = w.clone();
                for (b, &i) in open.iter().enumerate() {
                    sp[i] = ((fill >> (2 * b)) & 3) as u8;
                }
                let nucs: Vec<Nucleotide> = sp.iter().map(|&v| NUCS[v as usize]).collect();
                least = least.min(gain(&nucs));
                if least <= d {
                    break;
                }
            }
            let verdict = if d < least { "kills" } else { "gain-too-small" };
            out.push((t, d, Some(least), open.len(), verdict));
        }
        out
    }

    /// Per depth `t`, the least gain over every fill of the cut window of
    /// class `c` beside the end run `run` — `None` where no removal is priced
    /// at `t` (`raw` unkillable), the window reaches past the run, or it
    /// holds more than `max_open` unspelled symbols. Unlike [`Self::kills`]
    /// it never stops early, so each entry is the true least over fills.
    pub fn least_gains(
        &self,
        c: &TailClass,
        run: &[Nucleotide],
        gain: &mut dyn FnMut(&[Nucleotide]) -> i32,
        max_open: usize,
    ) -> Vec<Option<i32>> {
        let first = self.first();
        (1..=self.depth)
            .map(|t| {
                if c.raw[t - 1] == UNKILLABLE {
                    return None;
                }
                let mut w: Vec<u8> = Vec::with_capacity(t);
                if self.head {
                    for p in 0..t {
                        w.push(if p < run.len() {
                            run[p] as u8
                        } else {
                            c.spell[p]
                        });
                    }
                } else {
                    let start = self.len as usize - t;
                    let run_from = self.len as usize - run.len();
                    for p in start..self.len as usize {
                        if p >= run_from {
                            w.push(run[p - run_from] as u8);
                        } else {
                            w.push(c.spell[p - first as usize]);
                        }
                    }
                }
                if w.contains(&PAST) {
                    return None;
                }
                let open: Vec<usize> = (0..w.len()).filter(|&i| w[i] == OPEN).collect();
                if open.len() > max_open {
                    return None;
                }
                let mut least = i32::MAX;
                for fill in 0..(1usize << (2 * open.len())) {
                    let mut sp = w.clone();
                    for (b, &i) in open.iter().enumerate() {
                        sp[i] = ((fill >> (2 * b)) & 3) as u8;
                    }
                    let nucs: Vec<Nucleotide> = sp.iter().map(|&v| NUCS[v as usize]).collect();
                    least = least.min(gain(&nucs));
                }
                Some(least)
            })
            .collect()
    }

    /// Whether the class `c` of a domain closing at `close`, followed by the
    /// trailing run `trail`, is killed by some cut: `delta_t < gain_t`, the
    /// gain lower-bounded over every spelling of the open symbols (at most
    /// `max_open` of them; more give no kill).
    pub fn kills(
        &self,
        c: &TailClass,
        run: &[Nucleotide],
        gain: &mut dyn FnMut(&[Nucleotide]) -> i32,
        max_open: usize,
    ) -> Option<usize> {
        let first = self.first();
        for t in 1..=self.depth {
            let d = c.delta[t - 1];
            if d == UNKILLABLE {
                continue;
            }
            // A removal that lowers the cost kills with the freed symbols
            // left unmatched (gain 0), whatever they spell.
            if d < 0 {
                return Some(t);
            }
            // Spell the cut symbols: the last (3′) or first (5′) t, the
            // unmatched end run `run` beside the cell.
            let mut w: Vec<u8> = Vec::with_capacity(t);
            if self.head {
                for p in 0..t {
                    w.push(if p < run.len() {
                        run[p] as u8
                    } else {
                        c.spell[p]
                    });
                }
            } else {
                let start = self.len as usize - t;
                let run_from = self.len as usize - run.len();
                for p in start..self.len as usize {
                    if p >= run_from {
                        w.push(run[p - run_from] as u8);
                    } else {
                        w.push(c.spell[p - first as usize]);
                    }
                }
            }
            if w.contains(&PAST) {
                // The trailing run is shorter than the cell leaves: cannot happen
                // for a consistent parse; refuse to kill.
                continue;
            }
            let open: Vec<usize> = (0..w.len()).filter(|&i| w[i] == OPEN).collect();
            if open.len() > max_open {
                continue;
            }
            let mut least = i32::MAX;
            for fill in 0..(1usize << (2 * open.len())) {
                let mut s = w.clone();
                for (b, &i) in open.iter().enumerate() {
                    s[i] = ((fill >> (2 * b)) & 3) as u8;
                }
                let nucs: Vec<Nucleotide> = s.iter().map(|&v| NUCS[v as usize]).collect();
                least = least.min(gain(&nucs));
                if least <= d {
                    break;
                }
            }
            if d < least {
                return Some(t);
            }
        }
        None
    }

    /// Whether some invasion rival of the first-domain class `c` after the
    /// leading run `lead` is cheaper: `delta < gain`, every probe symbol known
    /// (an open one gives no kill). `gain(five, inner, three)` is the refold's
    /// gain ([`crate::rival_cert::EndGains::invasion_gain`]).
    pub fn kills_invasion(
        &self,
        c: &TailClass,
        lead: &[Nucleotide],
        gain: &mut InvasionGain<'_>,
    ) -> bool {
        let known = |v: u8| (v < 4).then(|| NUCS[v as usize]);
        // An anchored cut's spelling does not start at the item's first
        // symbol, so `lead` does not precede it.
        if self.start != 0 {
            return false;
        }
        c.inv.iter().any(|x| {
            if x.ml {
                return false;
            }
            // A negative removal bound kills with nothing refolded.
            if x.delta < 0 {
                return true;
            }
            let mut five: Vec<Nucleotide> = lead.to_vec();
            for p in lead.len()..x.a as usize {
                match c.spell.get(p).copied().and_then(known) {
                    Some(n) => five.push(n),
                    None => return false,
                }
            }
            let (Some(i), Some(j)) = (known(x.inner.0), known(x.inner.1)) else {
                return false;
            };
            let Some(three) = x
                .three
                .iter()
                .map(|&v| known(v))
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            x.delta < gain(&five, (i, j), &three)
        })
    }

    /// Whether the **split** rival kills the 5' class `c` of a one-domain
    /// item after the leading run `lead`: for each sealed invasion (the
    /// branching-loop rival of [`Self::branch_rival`]: the outer chain and the
    /// loop's closing pair removed for at most `delta`, every branch kept, the
    /// first opening at `a`), `gain(five, first)` refolds both context blocks
    /// with `five` (the leading run and the window's spelled symbols up to `a`,
    /// stopping at the first unspelled one), the branches and the trailing
    /// run. Every symbol between `five` and the trailing run that the probe
    /// does not carry is unmatched in a loop holding at least two branches,
    /// where it costs nothing, so the gain holds for every spelling of them.
    /// An `ml` record is scored only against a refold that encloses the
    /// stand-ins in a pair. Returns the rivals scored, whether one killed
    /// (`delta < gain`), and the largest `gain - delta` scored.
    pub fn kills_split(
        &self,
        c: &TailClass,
        lead: &[Nucleotide],
        gain: &mut SplitGain<'_>,
    ) -> (usize, bool, Option<i32>) {
        let known = |v: u8| (v < 4).then(|| NUCS[v as usize]);
        if !self.head || self.start != 0 {
            return (0, false, None);
        }
        let mut tests = 0;
        let mut best: Option<i32> = None;
        let killed = c.inv.iter().any(|x| {
            if !x.sealed || x.delta == UNKILLABLE {
                return false;
            }
            if x.delta < 0 && !x.ml {
                tests += 1;
                return true;
            }
            let (Some(i), Some(j)) = (known(x.inner.0), known(x.inner.1)) else {
                return false;
            };
            // The window up to `a`: the leading run, then the class's symbols.
            // Its unspelled ones are filled from the item set's spellings of
            // the window (those agreeing with every known symbol), the least
            // gain kept; with no list or more than `SPLIT_FILLS` agreeing
            // prefixes, the probe stops at the first unspelled symbol (blind).
            let a = x.a as usize;
            let mut w: Vec<u8> = lead.iter().map(|&n| n as u8).collect();
            for p in lead.len()..a {
                match c.spell.get(p).copied() {
                    Some(v) if v != PAST => w.push(v),
                    _ => break,
                }
            }
            let fills: Vec<&[u8]> = {
                let mut v: Vec<&[u8]> = self
                    .heads
                    .iter()
                    .filter(|h| h.len() >= w.len())
                    .map(|h| &h[..w.len()])
                    .filter(|h| h.iter().zip(&w).all(|(&s, &k)| k > 3 || s == k))
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            let g = if w.iter().all(|&v| v < 4) || fills.is_empty() || fills.len() > SPLIT_FILLS {
                let five: Vec<Nucleotide> = w
                    .iter()
                    .take_while(|&&v| v < 4)
                    .map(|&v| NUCS[v as usize])
                    .collect();
                gain(&five, (i, j), x.ml)
            } else {
                let mut least = i32::MAX;
                for f in fills {
                    let five: Vec<Nucleotide> = f.iter().map(|&v| NUCS[v as usize]).collect();
                    least = least.min(gain(&five, (i, j), x.ml));
                    if least <= x.delta {
                        break;
                    }
                }
                least
            };
            if g == i32::MIN {
                return false;
            }
            tests += 1;
            let m = g.saturating_sub(x.delta);
            best = Some(best.map_or(m, |b| b.max(m)));
            x.delta < g
        });
        (tests, killed, best)
    }

    /// The symbol a class spells at item position `pos`, if its window covers
    /// `pos` and spells it.
    pub fn spelled(&self, c: &TailClass, pos: u32) -> Option<Nucleotide> {
        let idx = pos.checked_sub(self.first())? as usize;
        c.spell
            .get(idx)
            .copied()
            .filter(|&v| v < 4)
            .map(|v| NUCS[v as usize])
    }

    /// [`Self::kills_invasion`] for the rivals whose freed 3' strand this
    /// 5' class leaves open (loops with unmatched 3' symbols, kept under
    /// [`Self::slip`]), the open symbols read from `other`, a class of a 3'
    /// cut `tail` of the same domain (closing at `close`). Every derivation
    /// in both classes spells those symbols as `other` does and breaks for at
    /// most `delta`, so a kill removes the pair of classes, not either alone.
    /// Returns the rivals scored, whether one killed, and the largest
    /// `gain − delta` scored (positive kills).
    pub fn kills_slip(
        &self,
        c: &TailClass,
        lead: &[Nucleotide],
        close: u32,
        tail: &TailCut,
        other: &TailClass,
        gain: &mut InvasionGain<'_>,
    ) -> (usize, bool, Option<i32>) {
        let known = |v: u8| (v < 4).then(|| NUCS[v as usize]);
        let mut tests = 0;
        let mut best: Option<i32> = None;
        let killed = c.inv.iter().any(|x| {
            if x.sealed || x.delta == UNKILLABLE || x.three.iter().all(|&v| v < 4) {
                return false;
            }
            let mut five: Vec<Nucleotide> = lead.to_vec();
            for p in lead.len()..x.a as usize {
                match c.spell.get(p).copied().and_then(known) {
                    Some(n) => five.push(n),
                    None => return false,
                }
            }
            let (Some(i), Some(j)) = (known(x.inner.0), known(x.inner.1)) else {
                return false;
            };
            let start = close + 1 - x.three.len() as u32;
            let Some(three) = x
                .three
                .iter()
                .enumerate()
                .map(|(k, &v)| known(v).or_else(|| tail.spelled(other, start + k as u32)))
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            tests += 1;
            let g = gain(&five, (i, j), &three);
            if g != i32::MIN {
                best = Some(best.map_or(g - x.delta, |b| b.max(g - x.delta)));
            }
            x.delta < g
        });
        (tests, killed, best)
    }

    /// Split the class `ci` of the 5' cut cell `(o, key)` by the spelling of
    /// the symbols its cut window leaves [`OPEN`], one **part** per spelling.
    /// Returns the parts' indices in the cell's list: the first takes the
    /// class's own slot, so an outer class's `inner_class` still names a part
    /// standing for the same derivation; the rest are appended.
    ///
    /// The spellings are every path out of the cell's opening node over the
    /// window, kept where they agree with the symbols the class already
    /// spells. Every derivation of the class follows such a path, so the
    /// parts cover the class: a part matching no derivation can only fail to
    /// be killed, and no real spelling is dropped.
    ///
    /// Each part keeps the class's value, `delta` and rivals, all properties
    /// of the derivations or the removal rather than of the spelling, so a
    /// lower bound stays one. What refinement changes is the kill test: a
    /// part's window is spelled, so a rival is scored against its own
    /// spelling rather than against the least gain over every fill.
    pub fn refine(
        &mut self,
        dfa: &Dfa,
        o: Node,
        key: &PairKey,
        ci: usize,
    ) -> Result<Vec<usize>, RefineRefusal> {
        if !self.head {
            return Err(RefineRefusal::TailSide);
        }
        let parent = self
            .cells
            .get(&(dfa.ix(o), *key))
            .and_then(|l| l.get(ci))
            .ok_or(RefineRefusal::Missing)?
            .clone();
        let from = key.open.pos as usize;
        let end = (self.depth).min(key.close.pos as usize);
        let Some(last) = (from..end).rev().find(|&p| parent.spell[p] == OPEN) else {
            return Err(RefineRefusal::Spelled);
        };
        let mut spellings: Vec<Vec<u8>> = Vec::new();
        let mut stack = vec![(key.open, parent.spell.clone())];
        while let Some((node, spell)) = stack.pop() {
            let p = node.pos as usize;
            if p > last {
                if !spellings.contains(&spell) {
                    if spellings.len() == self.refine_cap {
                        return Err(RefineRefusal::OverCap);
                    }
                    spellings.push(spell);
                }
                continue;
            }
            for e in dfa.right(node) {
                let nuc = e.nuc as u8;
                match spell[p] {
                    OPEN => {
                        let mut next = spell.clone();
                        next[p] = nuc;
                        stack.push((e.to, next));
                    }
                    v if v == nuc || v == PAST => stack.push((e.to, spell.clone())),
                    _ => {}
                }
            }
        }
        let list = self.cells.get_mut(&(dfa.ix(o), *key)).expect("priced cell");
        let mut parts = Vec::with_capacity(spellings.len());
        for (n, spell) in spellings.into_iter().enumerate() {
            let part = TailClass {
                spell,
                ..parent.clone()
            };
            if n == 0 {
                list[ci] = part;
                parts.push(ci);
            } else {
                parts.push(list.len());
                list.push(part);
            }
        }
        Ok(parts)
    }

    /// The overrides that make the scan's tables trace the class `ci` of
    /// the tail cell `(o, key)`: that cell and every tail cell down its chain
    /// set to the class's derivation.
    pub fn overrides(&self, o: Node, key: PairKey, ci: usize) -> Vec<(Node, PairKey, State)> {
        let mut out = Vec::new();
        let (mut o, mut key, mut ci) = (o, key, ci);
        while let Some(c) = self
            .cells
            .get(&(self.layout.ix(o), key))
            .and_then(|l| l.get(ci))
        {
            out.push((o, key, c.st));
            match (c.st.back, c.inner_class) {
                (Back::Single { inner }, Some(next)) => {
                    o = inner.outer;
                    key = inner.key;
                    ci = next;
                }
                _ => break,
            }
        }
        out
    }
}

const NUCS: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rival_cert::{rival_optimum, shape, EndGains};
    use crate::{dfa, scan::scan, ScanConfig};
    use derna_codon_dp::codon_table::aa_to_index;
    use derna_codon_dp::CodonUsage;

    fn setup(lambda: f64) -> (Params, Dfa, Tables) {
        let p = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
            .bytes()
            .map(|b| aa_to_index(b as char).unwrap())
            .collect();
        let graph = dfa::build(&aa, &usage);
        let tables = scan(&p, &graph, lambda, ScanConfig::default());
        (p, graph, tables)
    }

    /// Every recomputed cell's least class is the scan's own value: the
    /// derivations enumerated here are the scan's.
    #[test]
    fn recomputed_cells_reproduce_the_scan() {
        for lambda in [0.2, 0.5, 0.9] {
            let (p, graph, tables) = setup(lambda);
            for head in [false, true] {
                let cut = TailCut::new(&p, &graph, &tables, 6, head, &[]);
                assert!(cut.recomputed > 0);
                assert_eq!(cut.fallbacks, 0, "lambda {lambda} head {head}");
            }
        }
    }

    /// Every class names a reason exactly where its `delta_t` is
    /// unkillable; a terminal-loop class names the terminal loop wherever its
    /// own pair is removed (a fallback cell names itself instead), and branch
    /// counts appear only on the 5' cut, the one that counts branches.
    #[test]
    fn every_unkillable_depth_names_its_origin() {
        let mut below = false;
        for lambda in [0.2, 0.7] {
            let (p, graph, tables) = setup(lambda);
            for head in [false, true] {
                // A finite gain ceiling, so some removals outrun it.
                let cut = TailCut::new(&p, &graph, &tables, 6, head, &[200; 6]);
                let mut named = std::collections::HashSet::new();
                for ((_, key), list) in &cut.cells {
                    for c in list {
                        assert_eq!(c.why.len(), cut.depth);
                        for (t, (&d, &w)) in c.delta.iter().zip(&c.why).enumerate() {
                            assert_eq!(d == UNKILLABLE, w != Unpriced::Priced, "t {}", t + 1);
                            // The raw removal cost differs from delta only where the
                            // ceiling refused it, there or in an inner class: a
                            // removal refused inside can come out below the
                            // ceiling once this loop's own step is added.
                            let raw = c.raw[t];
                            if w == Unpriced::Ceiling {
                                assert_ne!(raw, UNKILLABLE);
                                below |= raw < cut.ceiling(t + 1);
                            } else {
                                assert_eq!(raw, d);
                            }
                            named.insert(w);
                            let removed = cut.removed(key.open.pos, key.close.pos, t + 1);
                            if matches!(c.st.back, Back::Hairpin) && removed {
                                assert!(matches!(w, Unpriced::Terminal | Unpriced::Floor));
                            }
                        }
                        if !head {
                            assert!(!c.why.iter().any(|w| matches!(
                                w,
                                Unpriced::BranchCount | Unpriced::BranchInCut
                            )));
                        }
                    }
                }
                assert!(named.contains(&Unpriced::Terminal), "head {head}");
                assert!(named.contains(&Unpriced::Ceiling), "head {head}");
            }
        }
        assert!(
            below,
            "no inherited ceiling refusal came out below the ceiling"
        );
    }

    /// The cuts only remove pairs, so the bound never falls; the traced
    /// optimum, with the class derivations written over the scan's cells,
    /// re-scores to the claimed value — including where a domain priced by
    /// both cuts took its head floor, which must then trace the head class.
    #[test]
    fn the_cut_optimum_is_no_lower_and_traces_to_its_value() {
        let (u5, u3) = ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG");
        let mut lifts = 0;
        for lambda in [0.3, 0.7] {
            let (p, graph, mut tables) = setup(lambda);
            let tail = TailCut::new(&p, &graph, &tables, 6, false, &[]);
            let head = TailCut::new(&p, &graph, &tables, 6, true, &[]);
            let mut ends = EndGains::new(&p, u5, u3);
            let plain = rival_optimum(&p, &graph, &tables, 1, None, &mut ends, 9, 6, None, None);
            let cut = rival_optimum(
                &p,
                &graph,
                &tables,
                1,
                None,
                &mut ends,
                9,
                6,
                Some(&tail),
                Some(&head),
            );
            lifts += cut.head_floor_lifts;
            let (Some(a), Some(b)) = (plain.opt.best, cut.opt.best) else {
                continue;
            };
            assert!(b.brd(lambda) >= a.brd(lambda) - 1e-9);
            let mut column = vec![None; tables.c.len()];
            for &(node, st) in &cut.opt.path {
                column[graph.ix(node)] = Some(st);
            }
            for (o, k, st) in &cut.overrides {
                tables.p[graph.ix(*o)].insert(*k, *st);
            }
            std::mem::swap(&mut tables.c, &mut column);
            let rec = crate::traceback::recover(&p, &graph, &tables);
            let energy = rna_mfe::evaluate(&p, rec.sequence.as_bytes(), &rec.structure).unwrap();
            assert_eq!(energy, b.mfe, "lambda {lambda}: {}", rec.structure);
            assert!(shape(&rec.structure).0 >= 1);
        }
        assert!(
            lifts > 0,
            "no domain priced by both cuts took its head floor"
        );
    }

    /// The branch-count interiors, closed on a pair, reach the scan's
    /// branching-loop cell exactly at their least: the recurrence is the
    /// scan's branching loop split by count.
    #[test]
    fn branch_counts_reproduce_the_scans_branching_loops() {
        let (p, graph, tables) = setup(0.5);
        let mut checked = 0;
        for o_pos in 6..=graph.seq_len {
            for &o in graph.nodes_at(o_pos) {
                for (key, m) in &tables.multi[graph.ix(o)] {
                    if key.open.pos >= 6 {
                        continue;
                    }
                    let eo = key.open_edge(&graph).unwrap();
                    let ec = key.close_edge(&graph, o).unwrap();
                    let bc = BranchCounts::new(&p, &graph, &tables, eo.to, 6, 0);
                    let least = bc
                        .slots(graph.ix(key.close))
                        .iter()
                        .filter(|slot| (2..=KMAX).contains(&(slot.c as usize)))
                        .map(|slot| {
                            let s = slot.st;
                            State::new(s.mfe, s.cai + eo.cai + ec.cai, s.back).brd(0.5)
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert!(
                        (least - m.brd(0.5)).abs() < 1e-6,
                        "{least} vs {}",
                        m.brd(0.5)
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }

    /// The joint two-end rival only kills more, so the bound with it is no
    /// lower than with the two cuts alone, and the optimum still traces to
    /// its value; and the pair rule itself: one removal at the higher of the
    /// two bounds, paid by the two gains together.
    #[test]
    fn the_joint_two_end_rival_only_raises_the_bound() {
        let (u5, u3) = ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG");
        for lambda in [0.3, 0.7] {
            let (p, graph, mut tables) = setup(lambda);
            let tail = TailCut::new(&p, &graph, &tables, 6, false, &[]);
            let head = TailCut::new(&p, &graph, &tables, 6, true, &[]);
            let mut joint = TailCut::new(&p, &graph, &tables, 6, true, &[]);
            joint.joint_ends = true;
            let mut ends = EndGains::new(&p, u5, u3);
            let run = |h: &TailCut, ends: &mut EndGains<'_>| {
                rival_optimum(
                    &p,
                    &graph,
                    &tables,
                    1,
                    None,
                    ends,
                    9,
                    6,
                    Some(&tail),
                    Some(h),
                )
            };
            let (a, cut) = (run(&head, &mut ends).opt.best, run(&joint, &mut ends));
            let (Some(a), Some(b)) = (a, cut.opt.best) else {
                continue;
            };
            assert!(b.brd(lambda) >= a.brd(lambda) - 1e-9);
            let mut column = vec![None; tables.c.len()];
            for &(node, st) in &cut.opt.path {
                column[graph.ix(node)] = Some(st);
            }
            for (o, k, st) in &cut.overrides {
                tables.p[graph.ix(*o)].insert(*k, *st);
            }
            std::mem::swap(&mut tables.c, &mut column);
            let rec = crate::traceback::recover(&p, &graph, &tables);
            let energy = rna_mfe::evaluate(&p, rec.sequence.as_bytes(), &rec.structure).unwrap();
            assert_eq!(energy, b.mfe, "lambda {lambda}: {}", rec.structure);
        }
        let st = State::new(Energy::ZERO, 0.0, Back::Hairpin);
        let class = |raw: Vec<i32>| TailClass {
            st,
            delta: raw.clone(),
            why: vec![Unpriced::Priced; raw.len()],
            raw,
            spell: Vec::new(),
            inner_class: None,
            inv: Vec::new(),
        };
        let (h, c) = (class(vec![0, 90]), class(vec![0, 60]));
        // Neither window alone pays 90; together they do, and only at the
        // depths where both removals are priced.
        assert!(crate::rival_cert::joint_kills(
            &h,
            &[None, Some(50)],
            &c,
            &[None, Some(45)]
        ));
        assert!(!crate::rival_cert::joint_kills(
            &h,
            &[None, Some(50)],
            &c,
            &[None, Some(40)]
        ));
        let unpriced = class(vec![0, UNKILLABLE]);
        assert!(!crate::rival_cert::joint_kills(
            &unpriced,
            &[None, Some(500)],
            &c,
            &[None, Some(500)]
        ));
    }

    /// The 3' wrap rival, alone and with the junction cut, only raises the
    /// cut optimum, keeps the best-first order (no expanded candidate below
    /// its group), and the optimum still traces to its value through the
    /// junction it took.
    #[test]
    fn the_wrap_rival_only_raises_the_bound_and_traces() {
        let (u5, u3) = ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG");
        let mut scored = 0;
        for lambda in [0.3, 0.7] {
            let (p, graph, tables) = setup(lambda);
            let head = TailCut::new(&p, &graph, &tables, 6, true, &[]);
            let plain = TailCut::new(&p, &graph, &tables, 6, false, &[]);
            let mut ends = EndGains::new(&p, u5, u3);
            let base = rival_optimum(
                &p,
                &graph,
                &tables,
                1,
                None,
                &mut ends,
                9,
                6,
                Some(&plain),
                Some(&head),
            );
            for cut in [0, 4] {
                let mut tail = TailCut::new(&p, &graph, &tables, 6, false, &[]);
                tail.wrap = 8;
                tail.wrap_cut = cut;
                let r = rival_optimum(
                    &p,
                    &graph,
                    &tables,
                    1,
                    None,
                    &mut ends,
                    9,
                    6,
                    Some(&tail),
                    Some(&head),
                );
                scored += r.wrap_scored.scored;
                assert_eq!(r.wrap_scored.order_breaks, 0);
                let (Some(a), Some(b)) = (base.opt.best, r.opt.best) else {
                    continue;
                };
                assert!(b.brd(lambda) >= a.brd(lambda) - 1e-9, "cut {cut}");
                let (_, _, mut traced) = setup(lambda);
                let mut column = vec![None; traced.c.len()];
                for &(node, st) in &r.opt.path {
                    column[graph.ix(node)] = Some(st);
                }
                for (o, k, st) in &r.overrides {
                    traced.p[graph.ix(*o)].insert(*k, *st);
                }
                std::mem::swap(&mut traced.c, &mut column);
                let rec = crate::traceback::recover(&p, &graph, &traced);
                let energy =
                    rna_mfe::evaluate(&p, rec.sequence.as_bytes(), &rec.structure).unwrap();
                assert_eq!(
                    energy, b.mfe,
                    "lambda {lambda} cut {cut}: {}",
                    rec.structure
                );
            }
        }
        assert!(scored > 0, "the wrap rival scored no junction run");
    }

    /// The 5' wrap across the same junction, alone and with the 5' junction
    /// cut, only raises the wrap's optimum, keeps the best-first order, and
    /// the optimum still traces to its value (through the 5' junction cut's
    /// class where its floor set it).
    #[test]
    fn the_five_prime_wrap_only_raises_the_bound_and_traces() {
        let (u5, u3) = ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG");
        let mut tested = 0;
        for lambda in [0.3, 0.7] {
            let (p, graph, tables) = setup(lambda);
            let head = TailCut::new(&p, &graph, &tables, 6, true, &[]);
            let mut ends = EndGains::new(&p, u5, u3);
            let optimum = |w5: usize, w5_cut: usize, ends: &mut EndGains<'_>| {
                let mut tail = TailCut::new(&p, &graph, &tables, 6, false, &[]);
                tail.wrap = 8;
                tail.wrap_cut = 4;
                tail.wrap5 = w5;
                tail.wrap5_cut = w5_cut;
                // With the junction cut, every rival on its candidates.
                tail.wrap5_both = w5_cut > 0;
                tail.wrap5_inv = w5_cut > 0;
                rival_optimum(
                    &p,
                    &graph,
                    &tables,
                    1,
                    None,
                    ends,
                    9,
                    6,
                    Some(&tail),
                    Some(&head),
                )
            };
            let base = optimum(0, 0, &mut ends);
            for w5_cut in [0, 4] {
                let r = optimum(8, w5_cut, &mut ends);
                tested += r.wrap_scored.w5_tests + r.wrap_scored.w5_cut_tests;
                assert_eq!(r.wrap_scored.order_breaks, 0);
                let (Some(a), Some(b)) = (base.opt.best, r.opt.best) else {
                    continue;
                };
                assert!(b.brd(lambda) >= a.brd(lambda) - 1e-9, "w5 cut {w5_cut}");
                let (_, _, mut traced) = setup(lambda);
                let mut column = vec![None; traced.c.len()];
                for &(node, st) in &r.opt.path {
                    column[graph.ix(node)] = Some(st);
                }
                for (o, k, st) in &r.overrides {
                    traced.p[graph.ix(*o)].insert(*k, *st);
                }
                std::mem::swap(&mut traced.c, &mut column);
                let rec = crate::traceback::recover(&p, &graph, &traced);
                let energy =
                    rna_mfe::evaluate(&p, rec.sequence.as_bytes(), &rec.structure).unwrap();
                assert_eq!(
                    energy, b.mfe,
                    "lambda {lambda} w5 cut {w5_cut}: {}",
                    rec.structure
                );
            }
        }
        assert!(tested > 0, "the 5' wrap tested no junction candidate");
    }

    /// The backward branch-count interiors, closed on a pair near the 3'
    /// end, reach the scan's branching-loop cell exactly at their least,
    /// and the cut optimum with them on is no lower than without any cut
    /// and still traces to its value.
    #[test]
    fn backward_branch_counts_reproduce_the_scan_and_keep_the_bound() {
        let (p, graph, tables) = setup(0.5);
        let first = graph.seq_len - 6;
        let mut checked = 0;
        for o_pos in first + 1..=graph.seq_len {
            for &o in graph.nodes_at(o_pos) {
                for (key, m) in &tables.multi[graph.ix(o)] {
                    if key.close.pos < first || key.open.pos >= first {
                        continue;
                    }
                    let eo = key.open_edge(&graph).unwrap();
                    let ec = key.close_edge(&graph, o).unwrap();
                    let bc = BranchCounts::new_back(&p, &graph, &tables, key.close, 6);
                    let least = bc
                        .slots(graph.ix(eo.to))
                        .iter()
                        .filter(|slot| (2..=KMAX).contains(&(slot.c as usize)))
                        .map(|slot| {
                            let s = slot.st;
                            State::new(s.mfe, s.cai + eo.cai + ec.cai, s.back).brd(0.5)
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert!(
                        (least - m.brd(0.5)).abs() < 1e-6,
                        "{least} vs {}",
                        m.brd(0.5)
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
        let (u5, u3) = ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG");
        let mut priced = 0;
        for lambda in [0.3, 0.7] {
            let (p, graph, mut tables) = setup(lambda);
            let mut tail = TailCut::lazy(&graph, &tables, 6, false, &[]);
            tail.tail_branch = true;
            tail.price_all(&p, &graph, &tables);
            priced += tail
                .cells
                .values()
                .flatten()
                .filter(|c| matches!(c.st.back, Back::MultiClose { .. }))
                .filter(|c| c.delta.iter().any(|&d| d != UNKILLABLE && d != 0))
                .count();
            let mut ends = EndGains::new(&p, u5, u3);
            let plain = rival_optimum(&p, &graph, &tables, 1, None, &mut ends, 9, 6, None, None);
            let cut = rival_optimum(
                &p,
                &graph,
                &tables,
                1,
                None,
                &mut ends,
                9,
                6,
                Some(&tail),
                None,
            );
            let (Some(a), Some(b)) = (plain.opt.best, cut.opt.best) else {
                continue;
            };
            assert!(b.brd(lambda) >= a.brd(lambda) - 1e-9);
            let mut column = vec![None; tables.c.len()];
            for &(node, st) in &cut.opt.path {
                column[graph.ix(node)] = Some(st);
            }
            for (o, k, st) in &cut.overrides {
                tables.p[graph.ix(*o)].insert(*k, *st);
            }
            std::mem::swap(&mut tables.c, &mut column);
            let rec = crate::traceback::recover(&p, &graph, &tables);
            let energy = rna_mfe::evaluate(&p, rec.sequence.as_bytes(), &rec.structure).unwrap();
            assert_eq!(energy, b.mfe, "lambda {lambda}: {}", rec.structure);
        }
        assert!(
            priced > 0,
            "no 3' branching-loop class got a finite removal"
        );
    }

    /// A branching-loop class carries the rival that continues past it: the
    /// closing pair removed, every branch kept where it is, the first branch
    /// the stand-in and nothing 3' of it free. Its break cost is the removal
    /// formula at the class's own branch count, and the open-ended count
    /// bucket carries none — one more branch costs another `intern` to remove,
    /// so "ten or more" bounds nothing.
    #[test]
    fn a_branching_loop_class_carries_a_sealed_rival_priced_by_its_count() {
        let depth = 6;
        let (p, graph, tables) = setup(0.5);
        // Opt in before pricing: the rival is off by default, and a cut built
        // with it off holds classes that never carried one.
        let mut cut = TailCut::lazy(&graph, &tables, depth, true, &[]);
        cut.branch_rival = true;
        let mut checked = 0;
        for o_pos in 0..=graph.seq_len {
            for &o in graph.nodes_at(o_pos) {
                let keys: Vec<PairKey> = tables.p[graph.ix(o)]
                    .keys()
                    .filter(|k| cut.is_tail(k))
                    .copied()
                    .collect();
                for key in keys {
                    cut.price(&p, &graph, &tables, o, key);
                }
                for key in tables.p[graph.ix(o)].keys() {
                    let Some(classes) = cut.classes(o, key) else {
                        continue;
                    };
                    let outer = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                    let t_out = term(&p, outer);
                    let priced: Vec<i32> = (2..KMAX)
                        .map(|c| -p.ml.closing_37 - (c as i32 + 1) * p.ml.intern_37 - 2 * t_out)
                        .collect();
                    for class in classes {
                        // Where the chain starts. Carried outward through a
                        // loop each step adds its own cost to the break, so
                        // only here is the break the count's own term.
                        let origin = matches!(class.st.back, Back::MultiClose { .. });
                        for x in class.inv.iter().filter(|x| x.sealed) {
                            assert!(x.three.is_empty(), "a sealed rival reads no 3' strand");
                            assert!((x.a as usize) < depth, "the stand-in must be spellable");
                            assert_ne!(x.delta, UNKILLABLE, "a carried rival stays finite");
                            if origin {
                                assert!(
                                    priced.contains(&x.delta),
                                    "delta {} is no priced count's removal",
                                    x.delta
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "no branching-loop rival was built");
    }

    /// A 5' cut over the setup's item with every cut cell priced, and the
    /// classes leaving an open symbol in the window of a cell short enough
    /// to enumerate every path through.
    fn refinable(depth: usize) -> (Dfa, TailCut, Vec<(Node, PairKey, usize)>) {
        let (p, graph, tables) = setup(0.5);
        let mut cut = TailCut::lazy(&graph, &tables, depth, true, &[]);
        let mut found = Vec::new();
        for o_pos in 0..=graph.seq_len {
            for &o in graph.nodes_at(o_pos) {
                let keys: Vec<PairKey> = tables.p[graph.ix(o)]
                    .keys()
                    .filter(|k| cut.is_tail(k))
                    .copied()
                    .collect();
                for key in keys {
                    cut.price(&p, &graph, &tables, o, key);
                    if key.close.pos - key.open.pos > 14 {
                        continue;
                    }
                    for (ci, c) in cut.classes(o, &key).unwrap_or(&[]).iter().enumerate() {
                        let end = depth.min(key.close.pos as usize);
                        if (key.open.pos as usize..end).any(|q| c.spell[q] == OPEN) {
                            found.push((o, key, ci));
                        }
                    }
                }
            }
        }
        (graph, cut, found)
    }

    /// Refinement drops no spelling: every path through the cell, read over
    /// the window and agreeing with what the class already spells, is some
    /// part's spelling — and every part keeps the class's value, `delta` and
    /// rivals. Built where some class splits into differing spellings, or it
    /// would pass with nothing split.
    #[test]
    fn refinement_parts_cover_every_spelling_and_keep_the_value() {
        let depth = 9;
        let (graph, mut cut, found) = refinable(depth);
        let (mut checked, mut split) = (0, 0);
        for (o, key, ci) in found {
            let parent = cut.classes(o, &key).unwrap()[ci].clone();
            let Ok(parts) = cut.refine(&graph, o, &key, ci) else {
                continue;
            };
            let list = cut.classes(o, &key).unwrap();
            let from = key.open.pos as usize;
            let end = depth.min(key.close.pos as usize);
            for (_, path) in graph.paths_between(key.open, o) {
                let agrees = (from..end).all(|q| {
                    let v = parent.spell[q];
                    v >= 4 || v == path[q - from] as u8
                });
                if !agrees {
                    continue;
                }
                let covered = parts.iter().any(|&i| {
                    (from..end).all(|q| {
                        parent.spell[q] != OPEN || list[i].spell[q] == path[q - from] as u8
                    })
                });
                assert!(covered, "a spelling of {key:?} class {ci} is in no part");
                checked += 1;
            }
            for &i in &parts {
                let part = &list[i];
                assert_eq!(part.st.brd(0.5), parent.st.brd(0.5));
                assert_eq!(part.delta, parent.delta);
                assert_eq!(part.inv, parent.inv);
                assert!((from..end).all(|q| part.spell[q] != OPEN));
            }
            split += usize::from(parts.len() > 1);
        }
        assert!(checked > 0 && split > 0, "checked {checked}, split {split}");
    }

    /// Over the part cap a class is refused and left whole, never truncated.
    #[test]
    fn refinement_over_the_cap_is_refused_not_truncated() {
        let (graph, mut cut, found) = refinable(9);
        let mut refused = 0;
        for (o, key, ci) in found {
            cut.refine_cap = 1;
            let before = cut.classes(o, &key).unwrap().to_vec();
            if cut.refine(&graph, o, &key, ci) == Err(RefineRefusal::OverCap) {
                let after = cut.classes(o, &key).unwrap();
                assert_eq!(after.len(), before.len());
                assert!(after.iter().zip(&before).all(|(a, b)| a.spell == b.spell));
                refused += 1;
            }
        }
        assert!(refused > 0);
    }

    /// Refining the class on the optimum's route only splits a lower bound,
    /// so the bound never falls, and a class with every window symbol
    /// spelled is refused as having nothing to resolve.
    #[test]
    fn refining_the_route_class_never_lowers_the_bound() {
        let contexts = [
            ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG"),
            ("GGGCGCCGCCAUGGCCAUG", "GCGGCCGCUUAAUAAAGC"),
        ];
        let mut rounds = 0;
        for lambda in [0.2, 0.3, 0.5, 0.7] {
            let (p, graph, tables) = setup(lambda);
            for (u5, u3) in contexts {
                for depth in [6, 9, 12] {
                    rounds += refine_route(&p, &graph, &tables, (u5, u3), depth);
                }
            }
        }
        assert!(rounds > 0, "no class on the route was refined");
    }

    /// Price and refine the optimum's first domain until neither applies,
    /// checking the bound never falls; the rounds that refined something.
    fn refine_route(
        p: &Params,
        graph: &Dfa,
        tables: &Tables,
        (u5, u3): (&str, &str),
        depth: usize,
    ) -> usize {
        let lambda = tables.lambda();
        let mut ends = EndGains::new(p, u5, u3);
        let mut head = TailCut::lazy(graph, tables, depth, true, &[]);
        let solve = |ends: &mut EndGains<'_>, h: &TailCut| {
            rival_optimum(p, graph, tables, 1, None, ends, 12, 6, None, Some(h))
        };
        let mut cut = solve(&mut ends, &head);
        let mut rounds = 0;
        for _ in 0..60 {
            let Some(pair) = cut.opt.path.iter().find_map(|(_, st)| match st.back {
                Back::ExteriorPair { pair, .. } => Some(pair),
                _ => None,
            }) else {
                break;
            };
            if head.is_tail(&pair.key) && !head.priced(pair.outer, &pair.key) {
                head.price(p, graph, tables, pair.outer, pair.key);
            } else {
                let Some(pick) = cut.head_pick.clone() else {
                    break;
                };
                match head.refine(graph, pick.outer, &pick.key, pick.class) {
                    Ok(_) => rounds += 1,
                    Err(why) => {
                        assert_eq!(why, RefineRefusal::Spelled);
                        break;
                    }
                }
            }
            let before = cut.opt.best.map(|b| b.brd(lambda));
            cut = solve(&mut ends, &head);
            if let (Some(a), Some(b)) = (before, cut.opt.best) {
                assert!(b.brd(lambda) >= a - 1e-9);
            }
        }
        rounds
    }

    /// Removing a branching loop's closing pair changes the cost by
    /// `-closing - (k + 1) * intern - 2 * AU`, the branches' AU terms cancel.
    #[test]
    fn opening_a_branching_loop_costs_what_the_formula_says() {
        let p = Params::load_default().unwrap();
        // A domain closed on (0, 31), a G-C pair, around two hairpin stems: k = 2.
        let seq = "GAAGCGCGAAAGCGCAAGCGCGAAAGCGCAAC";
        let with_pair = "(..((((....))))..((((....))))..)";
        let opened = "...((((....))))..((((....))))...";
        let with = rna_mfe::evaluate(&p, seq.as_bytes(), with_pair).unwrap().0;
        let without = rna_mfe::evaluate(&p, seq.as_bytes(), opened).unwrap().0;
        let au = term(&p, pair_from_nucleotides(Nucleotide::G, Nucleotide::C));
        let k = 2;
        let formula = -p.ml.closing_37 - (k + 1) * p.ml.intern_37 - 2 * au;
        assert_eq!(without - with, formula);
    }

    /// A class may stand for another only if every way it can be killed kills
    /// the other too: an invasion record it carries needs one in the other
    /// that is no harder to kill.
    #[test]
    fn a_class_with_a_record_its_rival_lacks_does_not_stand_for_it() {
        let class = |e: i32, inv: Vec<Inv>| TailClass {
            st: State::new(Energy(e), 0.0, Back::Hairpin),
            delta: vec![10],
            why: vec![Unpriced::Priced],
            raw: vec![10],
            spell: vec![PAST],
            inner_class: None,
            inv,
        };
        let record = |delta: i32| Inv {
            delta,
            a: 3,
            inner: (2, 1),
            three: Vec::new(),
            sealed: true,
            ml: false,
        };
        // `b` is lower and as hard by `delta`, but its record can kill it
        // where `a` has no rival: it may not stand for `a`.
        assert!(!dominates(
            &class(0, vec![record(5)]),
            &class(10, Vec::new()),
            0.5
        ));
        // `a`'s extra record only makes `a` easier to kill.
        assert!(dominates(
            &class(0, Vec::new()),
            &class(10, vec![record(5)]),
            0.5
        ));
        // A shared record: `b`'s must be no easier to kill than `a`'s.
        assert!(dominates(
            &class(0, vec![record(5)]),
            &class(10, vec![record(3)]),
            0.5
        ));
        assert!(!dominates(
            &class(0, vec![record(3)]),
            &class(10, vec![record(5)]),
            0.5
        ));
    }

    /// A kill needs `delta_t < gain_t`: an unkillable class is never killed,
    /// a cheap removal is killed by a gaining run, and an open symbol takes
    /// the least gain over its fills.
    #[test]
    fn kills_compare_delta_with_the_least_gain() {
        let cut = TailCut {
            layout: NodeLayout::PLAIN,
            ceiling: vec![i32::MAX; 2],
            depth: 2,
            head: false,
            own_repair: false,
            branch_rival: true,
            split: false,
            heads: Vec::new(),
            tail_branch: false,
            joint_ends: false,
            wrap: 0,
            wrap_free: false,
            wrap_cut: 0,
            wrap5: 0,
            wrap5_cut: 0,
            wrap5_both: false,
            wrap5_inv: false,
            slip: false,
            max_open: DEFAULT_MAX_OPEN,
            group_cap: None,
            refine_cap: REFINE_PART_CAP,
            inv_open_cap: usize::MAX,
            head_branch: true,
            len: 10,
            start: 0,
            cells: HashMap::new(),
            by_open: HashMap::new(),
            multi: HashMap::new(),
            fallbacks: 0,
            recomputed: 0,
            sealed_rivals: 0,
            sealed_spellable: 0,
        };
        let st = State::new(Energy::ZERO, 0.0, Back::Hairpin);
        let class = |delta: Vec<i32>, spell: Vec<u8>| TailClass {
            st,
            why: vec![Unpriced::Priced; delta.len()],
            raw: delta.clone(),
            delta,
            spell,
            inner_class: None,
            inv: Vec::new(),
        };
        let mut gain = |w: &[Nucleotide]| {
            if w.first() == Some(&Nucleotide::G) {
                50
            } else {
                0
            }
        };
        let g = Nucleotide::G as u8;
        let run = [Nucleotide::A];
        let kill =
            |c: TailClass, gain: &mut dyn FnMut(&[Nucleotide]) -> i32| cut.kills(&c, &run, gain, 3);
        assert_eq!(
            kill(class(vec![0, UNKILLABLE], vec![g, PAST]), &mut gain),
            None
        );
        assert_eq!(kill(class(vec![0, 10], vec![g, PAST]), &mut gain), Some(2));
        assert_eq!(kill(class(vec![0, 60], vec![g, PAST]), &mut gain), None);
        assert_eq!(kill(class(vec![0, 10], vec![OPEN, PAST]), &mut gain), None);
    }

    /// The slip test fills a rival's open 3' symbols from the paired 3' cut
    /// class: it kills only when that spelling gains more than `delta`, and
    /// scores nothing when the cut class leaves the symbol unspelled, the
    /// rival is sealed, or its strand is fully spelled already.
    #[test]
    fn slip_fills_open_symbols_from_the_paired_cut_class() {
        let cut = |head: bool| TailCut {
            layout: NodeLayout::PLAIN,
            ceiling: vec![i32::MAX; 4],
            depth: 4,
            head,
            own_repair: false,
            branch_rival: true,
            split: false,
            heads: Vec::new(),
            tail_branch: false,
            joint_ends: false,
            wrap: 0,
            wrap_free: false,
            wrap_cut: 0,
            wrap5: 0,
            wrap5_cut: 0,
            wrap5_both: false,
            wrap5_inv: false,
            slip: true,
            max_open: DEFAULT_MAX_OPEN,
            group_cap: None,
            refine_cap: REFINE_PART_CAP,
            inv_open_cap: usize::MAX,
            head_branch: true,
            len: 10,
            start: 0,
            cells: HashMap::new(),
            by_open: HashMap::new(),
            multi: HashMap::new(),
            fallbacks: 0,
            recomputed: 0,
            sealed_rivals: 0,
            sealed_spellable: 0,
        };
        let (five_cut, three_cut) = (cut(true), cut(false));
        let st = State::new(Energy::ZERO, 0.0, Back::Hairpin);
        let class = |spell: Vec<u8>, inv: Vec<Inv>| TailClass {
            st,
            why: vec![Unpriced::Priced; 4],
            raw: vec![0; 4],
            delta: vec![0; 4],
            spell,
            inner_class: None,
            inv,
        };
        let (a, c, g, u) = (
            Nucleotide::A as u8,
            Nucleotide::C as u8,
            Nucleotide::G as u8,
            Nucleotide::U as u8,
        );
        let rival = |three: Vec<u8>, sealed: bool| Inv {
            delta: 10,
            a: 2,
            inner: (g, c),
            three,
            sealed,
            ml: false,
        };
        // The cut window of the 3' class starts at item 6: item 8 is its
        // third symbol.
        let other = |at8: u8| class(vec![a, a, at8, a], Vec::new());
        let mut gain = |_: &[Nucleotide], _: (Nucleotide, Nucleotide), t: &[Nucleotide]| {
            if t.first() == Some(&Nucleotide::U) {
                50
            } else {
                0
            }
        };
        let slip = |inv: Inv, at8: u8, gain: &mut InvasionGain<'_>| {
            let head = class(vec![g, g, g, c], vec![inv]);
            five_cut.kills_slip(&head, &[], 9, &three_cut, &other(at8), gain)
        };
        assert_eq!(
            slip(rival(vec![OPEN, a], false), u, &mut gain),
            (1, true, Some(40))
        );
        assert_eq!(
            slip(rival(vec![OPEN, a], false), a, &mut gain),
            (1, false, Some(-10))
        );
        assert_eq!(
            slip(rival(vec![OPEN, a], false), OPEN, &mut gain),
            (0, false, None)
        );
        assert_eq!(
            slip(rival(vec![OPEN, a], true), u, &mut gain),
            (0, false, None)
        );
        assert_eq!(
            slip(rival(vec![u, a], false), u, &mut gain),
            (0, false, None)
        );
    }
}
