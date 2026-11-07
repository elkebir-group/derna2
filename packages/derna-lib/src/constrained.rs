//! The motif-constrained folding scan: an AC-threaded CKY over the product DFA.
//!
//! This is the constrained counterpart to [`crate::scan`], run only when the
//! forbidden-motif set genuinely constrains — when the product lift removed a
//! transition this protein's lattice offers, not merely when the automaton has
//! something to match ([`crate::product::ProductDfa::dropped_edge`]). It
//! optimizes the same fixed-λ BRD objective over exactly the motif-free feasible
//! set: every boundary a CKY cell keys on carries an Aho–Corasick state `q`, and
//! every nucleotide the scan spells — including the loop interiors the grammar
//! jumps over — threads `δ_AC(q, x)`, so a path that would spell a forbidden
//! motif does not exist in the lattice.
//!
//! Unlike [`crate::scan`], this folder does **not** use the Lyngsø–Zuker
//! internal-loop reduction, the one-sided corner factorizations, or the shared
//! `multi_open` `C + D` CAI-separation. Those three reductions minimize over
//! interior nucleotide choices independently of the far boundary; the AC state
//! couples that choice to the boundary it threads to, so the reductions no
//! longer separate. The constrained path therefore runs the un-reduced direct
//! enumeration for single-branch loops (correct, slower).
//!
//! The multi-loop's 3′ unpaired flank is the exception. It touches the energy
//! model nowhere, and it couples to the closing pair only through the AC state
//! it exits in, so it is folded into a table of its own (`M2R`: an `M2`
//! followed by the flank, keyed on that exit state) grown one nucleotide at a
//! time. The wrap then closes each `M2R` cell at its own frontier instead of
//! walking every frontier to the end of the molecule from every `M2` cell.
//!
//! The exact unconstrained default is untouched: a run whose product dropped no
//! edge — the empty set among them — never enters this module and takes the base
//! scan, which returns the same optimum over the same lattice (see
//! [`crate::design_with_motifs`]).
//!
//! Enforcement may be scoped to a window of positions
//! ([`crate::ac::MotifWindow`]); the folder reads the scoped automaton and needs
//! no other change, since a scoped state is still a function of the linear
//! prefix. Independently, the folder consults the region-pair constraint at its
//! single `P`-write choke point, exactly as [`crate::tables::Tables::update_p`]
//! does on the unconstrained path — which is what lets a motif set coexist with
//! a class table.
//!
//! ## The span primitives
//!
//! The substrate scores a loop interior by its closed-form energy plus a
//! best-CAI fill of the energy-irrelevant middle, jumping over the interior
//! positions. Under the constraint each jumped-over nucleotide threads the AC
//! state and must avoid motifs, so the fill becomes a small forward DP over the
//! product for that span, reporting the best result *per reachable exit AC
//! state* rather than one global best. The three primitives below
//! ([`span_fill`], [`span_corners`], [`span_spellings`]) are that DP in the
//! three shapes the interior scorers need. [`SpanTable`] memoizes the first two
//! over a fold's lifetime, since the fold asks the same tuple hundreds of times.
//!
//! ## The opt-in parallel fan-out
//!
//! [`Folder::p_push`] and [`Folder::m2_beam`] are the fold's two fat phases
//! (`p_push` by far, once the wrap reads `M2R`). When [`fold_with_beam`]'s
//! `threads > 1`, each fans its expensive pass — in `p_push`, the
//! single-branch push and the `M2` bifurcation — across a worker pool kept for
//! the whole fold ([`crate::fold_pool`]), and the chunks' survivors are merged
//! back one column bucket per worker. The wrinkle `crate::scan`
//! does not have: [`SpanTable`] is a lazily-filled memo shared for the whole
//! fold, and many workers cannot fill it through `&mut self` concurrently. So
//! a `threads > 1` fold forces [`Folder::prefill_spans`] eager first — still
//! single-threaded — and the workers read it through [`Folder::fill_span_ro`] /
//! [`Folder::corner_span_ro`], plain `&self` borrows that `.expect()` a slot is
//! already there rather than filling one. [`Folder::fan_out`]'s workers claim
//! contiguous chunks of the flat item list dynamically, but each chunk's
//! buffer is folded back in **chunk** order, which reproduces the
//! single-threaded write order exactly for any thread count — so the
//! byte-exactness obligation `crate::scan`'s fan-out only meets empirically
//! (equal-BRD ties may differ) holds here by construction: every thread count
//! is byte-identical to `threads = 1`, sequence and structure included.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use rna_mfe::Nucleotide;

use crate::dfa::{Dfa, Node, NodeLayout};
use crate::fold_pool::Pool;
use crate::hash::FxHashMap;
use crate::must_pair::MustPairConstraint;
use crate::product::LatticeAc;
use crate::region::{PairConstraint, Verdict};
use crate::Beam;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;

/// Sub-phase instrument for the span primitives, off unless asked for.
///
/// [`PhaseClock`] attributes the fold to six steps, and `p_push` is one of
/// them. It cannot say whether that step's time is the `(n1, n2)` enumeration
/// or the AC-threaded span work inside it — a distinction any lever on the span
/// primitives turns on, and one a phase clock is the wrong shape to make. This
/// meter makes it, in two modes that are **mutually exclusive by
/// construction**, because each destroys what the other measures:
///
/// - **Timing**, on the phase clock's development setting: splits the
///   single-branch closure sweep into time inside the span primitives and time
///   around them, and reports the whole fold's in-span total beside it.
/// - **Counting**, on its own development setting: counts calls against distinct
///   `(from, q_from, to)` tuples and takes no clock reading at all. Inserting
///   into a hash set on every call makes the timings meaningless, so a cell
///   wanting both numbers is run twice.
///
/// Calls per distinct key are exact and machine-independent where wall clock on
/// a shared host is neither, so a claim about how much work is recomputed
/// belongs in that column and not in a stopwatch reading.
///
/// A nested call is charged to the outermost primitive on the stack — a
/// [`span_fill`] inside a [`span_corners`] is the corner's cost, not a second
/// entry.
///
/// The tallies are thread-local and the fold is single-threaded, so one fold's
/// report is one fold's work. Should the fold ever fan out, the report would
/// describe the calling thread alone and the per-worker tallies would have to be
/// summed.
pub(crate) mod outside;

mod span_meter {
    use super::Node;
    use crate::hash::FxHashSet;
    use std::cell::RefCell;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    /// The primitives the meter distinguishes, in the index order every array
    /// below is in.
    pub(super) const NAMES: [&str; 3] = ["span_fill", "span_corners", "span_spellings"];
    /// Index of [`super::span_fill`] in [`NAMES`].
    pub(super) const FILL: usize = 0;
    /// Index of [`super::span_corners`] in [`NAMES`].
    pub(super) const CORNERS: usize = 1;
    /// Index of [`super::span_spellings`] in [`NAMES`].
    pub(super) const SPELLINGS: usize = 2;

    /// One fold's span-primitive tallies, in whichever mode is on.
    #[derive(Default)]
    struct Meter {
        /// Primitives currently on the stack, so only the outermost is charged.
        depth: u32,
        /// Calls at any depth, and calls made from outside every primitive.
        calls: [u64; 3],
        top_calls: [u64; 3],
        /// Spans the fold asked the memo for, whether or not a call followed.
        /// The demand `calls` used to equal and no longer does, so the
        /// recomputation removed is readable as `asks - calls` rather than only
        /// as a ratio that fell.
        asks: [u64; 3],
        /// Distinct `(from, q_from, to)` tuples behind those two call counts.
        keys: [FxHashSet<(u32, u32, u32)>; 3],
        top_keys: [FxHashSet<(u32, u32, u32)>; 3],
        /// Time inside outermost calls, per primitive and summed.
        spent: [Duration; 3],
        total: Duration,
        /// The single-branch sweep's split: span work, and the enumeration and
        /// corner cross-products around it.
        single_span: Duration,
        single_other: Duration,
    }

    thread_local! {
        static METER: RefCell<Meter> = RefCell::new(Meter::default());
    }

    /// Read a development flag once per process; the primitives are hot enough
    /// that a `var_os` per call would itself be the measurement.
    fn env_flag(setting: crate::dev_env::Setting, cache: &'static OnceLock<bool>) -> bool {
        *cache.get_or_init(|| crate::dev_env::var_os(setting).is_some())
    }

    /// Whether calls are being counted — exact, machine-independent, and
    /// incompatible with any timing.
    pub(super) fn counting() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        env_flag(crate::dev_env::Setting::SpanCount, &ON)
    }

    /// Whether time is being charged. Shares the phase clock's gate and yields
    /// to counting, so the two modes cannot be on together.
    pub(super) fn timing() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        env_flag(crate::dev_env::Setting::Profile, &ON) && !counting()
    }

    /// A live measurement of one span-primitive call: charges the call on
    /// construction and the elapsed time on drop, so the primitive's body needs
    /// no change and every early return is still accounted for.
    pub(super) struct Scope {
        kind: usize,
        started: Option<Instant>,
        armed: bool,
    }

    impl Scope {
        /// Enter primitive `kind` called on the span `[from, to]` from state
        /// `q_from`. A no-op — and no clock reading — when neither mode is on.
        pub(super) fn enter(kind: usize, from: Node, q_from: u32, to: Node) -> Self {
            if !(timing() || counting()) {
                return Self {
                    kind,
                    started: None,
                    armed: false,
                };
            }
            let key = (
                super::NodeIx::from(from).0,
                q_from,
                super::NodeIx::from(to).0,
            );
            let outermost = METER.with_borrow_mut(|m| {
                let outermost = m.depth == 0;
                m.depth += 1;
                if counting() {
                    count(m, kind, key, outermost);
                }
                outermost
            });
            Self {
                kind,
                started: (outermost && timing()).then(Instant::now),
                armed: true,
            }
        }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            if !self.armed {
                return;
            }
            METER.with_borrow_mut(|m| {
                m.depth -= 1;
                if let Some(started) = self.started {
                    let elapsed = started.elapsed();
                    m.spent[self.kind] += elapsed;
                    m.total += elapsed;
                }
            });
        }
    }

    /// Tally one span the fold asked the memo for. Counted at the memo rather
    /// than inside the primitive, because a memo hit never enters the
    /// primitive and the demand is what a redundancy claim is a ratio of.
    pub(super) fn ask(kind: usize) {
        if counting() {
            METER.with_borrow_mut(|m| m.asks[kind] += 1);
        }
    }

    /// Tally one call and its argument tuple, at both depths.
    fn count(m: &mut Meter, kind: usize, key: (u32, u32, u32), outermost: bool) {
        m.calls[kind] += 1;
        m.keys[kind].insert(key);
        if outermost {
            m.top_calls[kind] += 1;
            m.top_keys[kind].insert(key);
        }
    }

    /// Run one single-branch closure sweep, splitting its wall clock into the
    /// span primitives it called and everything else it did.
    ///
    /// The span half is read off the meter's running total rather than timed
    /// again, so a call is never charged twice and the two halves sum to the
    /// sweep.
    pub(super) fn in_single<T>(body: impl FnOnce() -> T) -> T {
        if !timing() {
            return body();
        }
        let before = METER.with_borrow(|m| m.total);
        let started = Instant::now();
        let out = body();
        let elapsed = started.elapsed();
        METER.with_borrow_mut(|m| {
            let span = m.total - before;
            m.single_span += span;
            m.single_other += elapsed.saturating_sub(span);
        });
        out
    }

    /// Clear the tallies so each fold reports its own.
    pub(super) fn reset() {
        if timing() || counting() {
            METER.with_borrow_mut(|m| *m = Meter::default());
        }
    }

    /// Emit whichever mode's numbers were collected, on stderr beside the phase
    /// report.
    pub(super) fn report() {
        METER.with_borrow(|m| {
            if timing() {
                report_time(m);
            }
            if counting() {
                report_counts(m);
            }
        });
    }

    /// The timing report: the single-branch split, then where the in-span time
    /// went. The share is of the sweep, not of the whole process.
    fn report_time(m: &Meter) {
        let share = percent(m.single_span, m.single_span + m.single_other);
        eprintln!(
            "span primitives (time, total {:.3}s):",
            m.total.as_secs_f64()
        );
        eprintln!(
            "  p_to_single      span {:>9.3}s  other {:>9.3}s  span {share:>5.1}%",
            m.single_span.as_secs_f64(),
            m.single_other.as_secs_f64(),
        );
        for (name, spent) in NAMES.iter().zip(&m.spent) {
            eprintln!("  {name:<16} {:>9.3}s", spent.as_secs_f64());
        }
    }

    /// The counting report: calls against distinct argument tuples, at both
    /// depths. `redundancy` is how many times the average tuple was recomputed.
    fn report_counts(m: &Meter) {
        eprintln!("span primitives (counts):");
        for (i, name) in NAMES.iter().enumerate() {
            let (calls, distinct) = (m.calls[i], m.keys[i].len() as u64);
            let (top, top_distinct) = (m.top_calls[i], m.top_keys[i].len() as u64);
            if m.asks[i] > 0 {
                eprintln!(
                    "  {name:<16} asks  {:>12} served {:>12} = {:>7.2}% from the table",
                    m.asks[i],
                    m.asks[i].saturating_sub(top),
                    percent_of(m.asks[i].saturating_sub(top), m.asks[i]),
                );
            }
            eprintln!(
                "  {name:<16} calls {calls:>12} distinct {distinct:>9} \
redundancy {:>8.1}x  top {top:>12} / {top_distinct:>9} = {:>8.1}x",
                ratio(calls, distinct),
                ratio(top, top_distinct),
            );
        }
    }

    /// `part` as a percentage of `whole`, and zero when there is no whole.
    fn percent(part: Duration, whole: Duration) -> f64 {
        let whole = whole.as_secs_f64();
        if whole > 0.0 {
            part.as_secs_f64() / whole * 100.0
        } else {
            0.0
        }
    }

    /// `part` as a percentage of `whole`, and zero when there is no whole.
    fn percent_of(part: u64, whole: u64) -> f64 {
        if whole > 0 {
            part as f64 / whole as f64 * 100.0
        } else {
            0.0
        }
    }

    /// `calls / distinct`, and zero when nothing was called.
    fn ratio(calls: u64, distinct: u64) -> f64 {
        if distinct > 0 {
            calls as f64 / distinct as f64
        } else {
            0.0
        }
    }
}

/// The best-CAI motif-free fill of the span `[from, to]` per reachable exit AC
/// state: for each AC state `q_to` some motif-free DFA path from `(from,
/// q_from)` reaches `to` in, the maximum accumulated CAI log-weight over those
/// paths. The constrained analogue of [`Dfa::broken_codon_cai`] — which returns
/// one global best because the unconstrained fill has a single AC context;
/// here different exit states lead to different downstream feasibility, so each
/// is kept.
///
/// Returns `[(q_from, 0.0)]` for an empty span (`from == to`) and empty when no
/// motif-free path connects the boundaries (an infeasible or non-descending
/// span). The result is unordered.
pub(crate) fn span_fill(
    dfa: &Dfa,
    ac: LatticeAc<'_>,
    from: Node,
    q_from: u32,
    to: Node,
) -> Vec<(u32, f64)> {
    let _measured = span_meter::Scope::enter(span_meter::FILL, from, q_from, to);
    if from == to {
        return vec![(q_from, 0.0)];
    }
    if from.pos >= to.pos {
        return Vec::new();
    }
    // Forward DP one position at a time: `cur` holds the best CAI to reach each
    // `(node, q)` at the current position. Summing edge weights and taking the
    // max reproduces `broken_codon_cai` in the unconstrained case (the cost-to-go
    // weights telescope so the best whole codon contributes zero), now pruned to
    // motif-free transitions and split by exit state.
    let mut cur: FxHashMap<(Node, u32), f64> = FxHashMap::default();
    cur.insert((from, q_from), 0.0);
    for _ in from.pos..to.pos {
        let mut next: FxHashMap<(Node, u32), f64> = FxHashMap::default();
        for (&(node, q), &cai) in &cur {
            for e in dfa.right(node) {
                let Some(q2) = ac.step(node, q, e.nuc) else {
                    continue;
                };
                relax(&mut next, (e.to, q2), cai + e.cai);
            }
        }
        cur = next;
    }
    cur.into_iter()
        .filter(|&((node, _), _)| node == to)
        .map(|((_, q), cai)| (q, cai))
        .collect()
}

/// One terminal-mismatch corner of an unpaired strand under the constraint: the
/// strand's first (5′) and last (3′) nucleotides — the only interior the energy
/// model reads — the AC state after threading the whole strand, and the total
/// CAI (terminals plus the best-CAI motif-free middle). The constrained
/// analogue of [`crate::interior::corner_scores`], split by exit AC state.
#[derive(Clone, Copy)]
pub(crate) struct CornerAc {
    /// First (5′) nucleotide of the strand.
    pub first: Nucleotide,
    /// Last (3′) nucleotide of the strand.
    pub last: Nucleotide,
    /// AC state after threading the whole strand (its 3′-side exit state).
    pub q_to: u32,
    /// Total CAI log-weight of the strand (terminals plus best-CAI middle).
    pub cai: f64,
}

/// The terminal-mismatch corners of the strand `[from, to]` (length ≥ 1),
/// threading the AC state from `q_from`: every `(first, last)` nucleotide pair
/// the strand can take, with the energy-irrelevant middle filled best-CAI, kept
/// as the highest CAI per `(first, last, exit state)`. A length-1 strand's
/// single nucleotide is both terminals.
pub(crate) fn span_corners(
    dfa: &Dfa,
    ac: LatticeAc<'_>,
    from: Node,
    q_from: u32,
    to: Node,
) -> Vec<CornerAc> {
    let _measured = span_meter::Scope::enter(span_meter::CORNERS, from, q_from, to);
    if to.pos == from.pos + 1 {
        return dfa
            .right(from)
            .iter()
            .filter(|e| e.to == to)
            .filter_map(|e| {
                let q_to = ac.step(from, q_from, e.nuc)?;
                Some(CornerAc {
                    first: e.nuc,
                    last: e.nuc,
                    q_to,
                    cai: e.cai,
                })
            })
            .collect();
    }
    // Length ≥ 2: enumerate the first and last nucleotides (the terminal
    // mismatches) and fill the middle `[first.to, last.to]` per exit state. The
    // dedup key uses nucleotide indices (`Nucleotide` is not `Hash`); the
    // nucleotides are reconstructed from those indices when the corners are
    // materialized.
    let mut best: CornerBest = FxHashMap::default();
    for first in dfa.right(from) {
        let Some(q1) = ac.step(from, q_from, first.nuc) else {
            continue;
        };
        for last in dfa.left(to) {
            // `dfa.left` reverses the edge: `last.to` is the *source* node, so
            // `last.nuc` sits at position `last.to.pos` (one before `to`).
            for (q_mid, mid_cai) in span_fill(dfa, ac, first.to, q1, last.to) {
                let Some(q_to) = ac.step(last.to, q_mid, last.nuc) else {
                    continue;
                };
                let key = (first.nuc as u8, last.nuc as u8, q_to);
                relax(&mut best, key, first.cai + mid_cai + last.cai);
            }
        }
    }
    corners_of(best)
}

/// The best CAI per corner key `(first, last, exit state)`, nucleotides as
/// alphabet indices.
type CornerBest = FxHashMap<(u8, u8, u32), f64>;

/// The corners in `best`, keyed `(first, last, exit state)`, in key order —
/// an order fixed by the corners themselves rather than by the map's history,
/// so every way of computing one span lists it identically.
fn corners_of(best: CornerBest) -> Vec<CornerAc> {
    let mut keyed: Vec<_> = best.into_iter().collect();
    keyed.sort_unstable_by_key(|&(key, _)| key);
    keyed
        .into_iter()
        .map(|((first, last, q_to), cai)| CornerAc {
            first: nuc_from_index(first),
            last: nuc_from_index(last),
            q_to,
            cai,
        })
        .collect()
}

/// [`span_corners`] for every strand `[from, to]` with `to.pos` in
/// `from.pos + 1 ..= last`, as `(to, corners)` in position then node order,
/// from one forward sweep.
///
/// Computing each strand on its own refills the middle from scratch, so a
/// row of them costs the square of its length; a hairpin's closing base
/// ranges to the end of the molecule, which made the hairpin seeds cubic. The
/// sweep keeps, per first nucleotide, the best middle reaching each `(node,
/// state)` and closes it with every last edge at each step. The CAI of a
/// corner is summed exactly as [`span_corners`] sums it — the first edge, plus
/// the middle accumulated from zero, plus the last edge — so the values agree
/// bit for bit, not only to rounding.
pub(crate) fn span_corners_row(
    dfa: &Dfa,
    ac: LatticeAc<'_>,
    from: Node,
    q_from: u32,
    last: u32,
) -> Vec<(Node, Vec<CornerAc>)> {
    let _measured = span_meter::Scope::enter(span_meter::CORNERS, from, q_from, Node::new(last, 0));
    let last = last.min(dfa.seq_len);
    let mut row = Vec::new();
    if from.pos >= last {
        return row;
    }
    for &to in dfa.nodes_at(from.pos + 1) {
        row.push((to, span_corners(dfa, ac, from, q_from, to)));
    }
    // The middle so far, per (first nucleotide, node, state): its best CAI,
    // accumulated from zero after the first edge.
    let mut first_cai = [0.0; 4];
    let mut cur: FxHashMap<(u8, Node, u32), f64> = FxHashMap::default();
    for first in dfa.right(from) {
        let Some(q1) = ac.step_edge(from, q_from, *first) else {
            continue;
        };
        first_cai[first.nuc as usize] = first.cai;
        cur.insert((first.nuc as u8, first.to, q1), 0.0);
    }
    for pos in from.pos + 2..=last {
        let mut ends: Vec<(Node, CornerBest)> = dfa
            .nodes_at(pos)
            .iter()
            .map(|&to| (to, FxHashMap::default()))
            .collect();
        let mut next: FxHashMap<(u8, Node, u32), f64> = FxHashMap::default();
        for (&(first, node, q), &mid) in &cur {
            for e in dfa.right(node) {
                let Some(q_to) = ac.step_edge(node, q, *e) else {
                    continue;
                };
                if let Some((_, best)) = ends.iter_mut().find(|(to, _)| *to == e.to) {
                    let cai = first_cai[first as usize] + mid + e.cai;
                    relax(best, (first, e.nuc as u8, q_to), cai);
                }
                if pos < last {
                    relax(&mut next, (first, e.to, q_to), mid + e.cai);
                }
            }
        }
        row.extend(ends.into_iter().map(|(to, best)| (to, corners_of(best))));
        cur = next;
    }
    row
}

/// The nucleotide at alphabet index `i` (`A=0, C=1, G=2, U=3`), matching
/// `Nucleotide as u8`. Inverts the index used to key the corner dedup map.
fn nuc_from_index(i: u8) -> Nucleotide {
    match i {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        _ => Nucleotide::U,
    }
}

/// Whether every span slot is filled before the fold rather than on first
/// touch, read once per process because the fill sites are hot enough that a
/// `var_os` per call would itself be the measurement.
fn eager_spans() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::dev_env::var_os(crate::dev_env::Setting::SpanEager).is_some())
}

/// Whether internal loops with both flanks free score their corners
/// separately ([`Folder::separated_internal`]): off unless a development build
/// turns it on (`Setting::ConstrainedSeparate`), read once per process.
fn separate_free_loops() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::dev_env::var_os(crate::dev_env::Setting::ConstrainedSeparate)
            .is_some_and(|v| v == "1")
    })
}

/// The shortest flank whose corners separate: its first and last nucleotides
/// sit in different codons, so the codons strictly between take their best
/// whatever the corners are ([`Dfa::span_head_cai`]).
const SEPARABLE_FLANK: usize = 5;

/// For each layer, how many edges a free run starting there spans: a layer is
/// free when every base node on it holds one automaton state and keeps every
/// out-edge under it, and an edge is in a run when both its layers are. Over
/// such a run the motif set constrains nothing, so a flank inside one is
/// spelled exactly as on the plain lattice.
fn free_runs(product: &ProductDfa<'_>) -> Vec<u32> {
    let (dfa, ac) = (product.dfa(), product.lattice_ac());
    let layers = dfa.seq_len as usize + 1;
    let single: Vec<bool> = (0..layers)
        .map(|p| {
            dfa.nodes_at(p as u32)
                .iter()
                .all(|&n| product.reachable_states(n).len() == 1)
        })
        .collect();
    let kept = |p: usize| {
        dfa.nodes_at(p as u32).iter().all(|&v| {
            let q = product.reachable_states(v)[0];
            dfa.right(v).iter().all(|&e| {
                ac.step_edge(v, q, e)
                    .is_some_and(|q2| product.product_id(e.to, q2).is_some())
            })
        })
    };
    let mut run = vec![0u32; layers + 1];
    for p in (0..layers.saturating_sub(1)).rev() {
        if single[p] && single[p + 1] && kept(p) {
            run[p] = run[p + 1] + 1;
        }
    }
    run
}

/// Whether skewed internal loops on free flanks are taken from their size
/// diagonal ([`Folder::pull_skewed`]) rather than pushed pair by pair:
/// off unless a development build turns it on (`Setting::ConstrainedLz`), read
/// once per process. It scores corners as
/// [`separate_free_loops`] does.
fn diagonal_free_loops() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::dev_env::var_os(crate::dev_env::Setting::ConstrainedLz).is_some_and(|v| v == "1")
    })
}

/// For each layer, how many edges the free run ending there spans: [`free_runs`]
/// read from the other end.
fn free_backs(run: &[u32]) -> Vec<u32> {
    let mut back = vec![0u32; run.len()];
    for q in 1..run.len() {
        if run[q - 1] >= 1 {
            back[q] = back[q - 1] + 1;
        }
    }
    back
}

/// A closed pair whose two boundary layers end and start free runs, as the
/// diagonal pull reads it: its opening base's position, its value with its
/// best inner corner (outer-independent on free flanks), and what the write
/// needs.
#[derive(Clone, Copy)]
struct DiagEntry {
    k: u32,
    h: f64,
    key: PKey,
    cell: Cell,
    corner: Corner,
}

/// [`DiagEntry`]s by size diagonal `Δ = close − open` (the closed pair's two
/// base positions), each diagonal in ascending opening position, since pairs
/// are filed as their closing column finalizes; and the opening positions
/// filed per closing position, for the pull's candidate openings.
#[derive(Default)]
struct DiagIndex {
    by_diag: Vec<Vec<DiagEntry>>,
    opens_by_close: Vec<Vec<u32>>,
}

impl DiagIndex {
    fn insert(&mut self, close: u32, e: DiagEntry) {
        let d = (close - e.k) as usize;
        if self.by_diag.len() <= d {
            self.by_diag.resize_with(d + 1, Vec::new);
        }
        debug_assert!(self.by_diag[d].last().is_none_or(|l| l.k <= e.k));
        self.by_diag[d].push(e);
        let c = close as usize;
        if self.opens_by_close.len() <= c {
            self.opens_by_close.resize_with(c + 1, Vec::new);
        }
        let opens = &mut self.opens_by_close[c];
        if opens.last() != Some(&e.k) && !opens.contains(&e.k) {
            opens.push(e.k);
        }
    }

    /// The entry of least value on diagonal `delta` whose opening lies in
    /// `[klo, khi]`; the first of equal values wins.
    fn best(&self, delta: usize, klo: u32, khi: u32) -> Option<&DiagEntry> {
        let diag = self.by_diag.get(delta)?;
        let from = diag.partition_point(|e| e.k < klo);
        let mut best: Option<&DiagEntry> = None;
        for e in diag[from..].iter().take_while(|e| e.k <= khi) {
            if best.is_none_or(|b| e.h < b.h) {
                best = Some(e);
            }
        }
        best
    }

    /// The opening positions of the pairs filed as closing at `close`.
    fn opens_closing_at(&self, close: u32) -> &[u32] {
        self.opens_by_close
            .get(close as usize)
            .map_or(&[], Vec::as_slice)
    }
}

/// A dense-indexed memo for one span primitive over one fold.
///
/// [`span_fill`] and [`span_corners`] are pure functions of `(dfa, ac, from,
/// q_from, to)`, every argument of which is fixed for a fold's lifetime, and
/// the fold asks for the same tuple hundreds of times because the call sites
/// sit inside loops the tuple does not depend on. This gives every tuple one
/// slot, so each is computed at most once and read back as a slice afterwards.
///
/// **The slot holds what the primitive returned, verbatim, in its own order.**
/// Both primitives collect out of an [`FxHashMap`], so the order they return is
/// hash-iteration order — deterministic, arbitrary, and load-bearing:
/// [`Folder::close_internal`] feeds every corner pair to [`Folder::put_p`],
/// which resolves a tie by keeping the incumbent, so a reordered result moves
/// the emitted design while leaving every energy identical. Caching the vector
/// rather than refilling a slot by some tidier traversal is what keeps that
/// order.
///
/// The slot index is the product node `(from, q_from)` — the dense addressing
/// [`ProductDfa::product_id`] already provides — crossed with `to`'s node
/// index. Both endpoints are carried in full because a span is **not** bounded
/// by [`SINGLE_MAX_LEN`]: the multi-loop 3′ wrap sweeps its closing frontier to
/// the end of the molecule.
struct SpanTable<T> {
    /// Node-index stride of the `to` dimension — one past the largest
    /// [`Dfa::ix`].
    stride: usize,
    /// The lattice's dense node layout, which the `to` dimension is indexed by.
    layout: NodeLayout,
    /// `(start, len)` into `payload` per slot, or [`Self::UNFILLED`] as the
    /// start of a slot not yet computed.
    slots: Vec<(u32, u32)>,
    /// Every stored vector, concatenated in the order the slots were first
    /// filled. Append-only, so a handle taken before a later fill stays valid.
    payload: Vec<T>,
    /// How many slots have been filled — the demand, against `slots.len()` as
    /// the dense bound.
    filled: usize,
}

/// A product lattice's span memos ([`span_fill`] and [`span_corners`]
/// results), which depend on the lattice and the automaton but not on λ or
/// any bound: a sweep or walk folds the same product many times, and each
/// fold after the first starts from the slots the earlier ones filled.
pub(crate) struct SpanMemo {
    fills: SpanTable<(u32, f64)>,
    corners: SpanTable<CornerAc>,
}

impl<T: Copy> SpanTable<T> {
    /// The `start` of a slot that has not been computed.
    const UNFILLED: u32 = u32::MAX;

    /// An empty table addressing `product_nodes × stride` slots.
    fn new(product_nodes: usize, layout: NodeLayout, stride: usize) -> Self {
        Self {
            stride,
            layout,
            slots: vec![(Self::UNFILLED, 0); product_nodes * stride],
            payload: Vec::new(),
            filled: 0,
        }
    }

    /// The slot of the span ending at `to` from product node `origin`.
    fn slot(&self, origin: usize, to: Node) -> usize {
        origin * self.stride + self.layout.ix(to)
    }

    /// The stored `(start, len)` handle of `slot`, or `None` when it is empty.
    fn stored(&self, slot: usize) -> Option<(usize, usize)> {
        let (start, len) = self.slots[slot];
        (start != Self::UNFILLED).then_some((start as usize, len as usize))
    }

    /// Append `values` to the payload and record them at `slot`.
    fn store(&mut self, slot: usize, values: Vec<T>) -> (usize, usize) {
        let handle = self.append(values);
        self.slots[slot] = (handle.0 as u32, handle.1 as u32);
        self.filled += 1;
        handle
    }

    /// Append `values` without recording a slot — the path for a span whose
    /// origin is not a reachable product node and so has no address.
    fn append(&mut self, values: Vec<T>) -> (usize, usize) {
        let start = self.payload.len();
        self.payload.extend(values);
        (start, self.payload.len() - start)
    }

    /// The payload entry at `at`, copied out so no borrow outlives the read.
    fn at(&self, at: usize) -> T {
        self.payload[at]
    }
}

/// Every motif-free spelling of the small span `[from, to]`, threading the AC
/// state: each as its accumulated CAI, the nucleotides spelled, and the exit AC
/// state. Used for the special-hairpin sizes (3/4/6), where the whole loop
/// sequence selects a table bonus so no closed form over terminals suffices;
/// the spans are ≤ 6 nt with bounded fan-out, so the full enumeration is cheap.
/// The constrained analogue of [`Dfa::paths_between`].
pub(crate) fn span_spellings(
    dfa: &Dfa,
    ac: LatticeAc<'_>,
    from: Node,
    q_from: u32,
    to: Node,
) -> Vec<(f64, Vec<Nucleotide>, u32)> {
    let _measured = span_meter::Scope::enter(span_meter::SPELLINGS, from, q_from, to);
    let mut out = Vec::new();
    let mut stack = vec![(from, q_from, 0.0f64, Vec::new())];
    while let Some((node, q, cai, seq)) = stack.pop() {
        if node == to {
            out.push((cai, seq, q));
            continue;
        }
        if node.pos >= to.pos {
            continue;
        }
        for e in dfa.right(node) {
            let Some(q2) = ac.step(node, q, e.nuc) else {
                continue;
            };
            let mut next = seq.clone();
            next.push(e.nuc);
            stack.push((e.to, q2, cai + e.cai, next));
        }
    }
    out
}

/// Reconstruct a maximum-CAI motif-free path spelling the span `[from, to]`
/// that lands exactly at exit AC state `q_to`, as the nucleotides along it.
/// The path-returning companion to [`span_fill`]: where `span_fill` reports the
/// best CAI *per* exit state, this respells one path achieving that best value
/// for a chosen exit (any maximal path on ties — all share the reported CAI, and
/// the interior it fills is energy-irrelevant). The traceback uses it to respell
/// the loop interiors the forward scan jumped over. Returns an empty path for an
/// empty span (`from == to`) and when no motif-free path reaches `(to, q_to)`.
pub(crate) fn span_fill_path(
    dfa: &Dfa,
    ac: LatticeAc<'_>,
    from: Node,
    q_from: u32,
    to: Node,
    q_to: u32,
) -> Vec<Nucleotide> {
    if from.pos >= to.pos {
        return Vec::new();
    }
    // Forward max-CAI DP carrying, at each reached `(node, q)`, the predecessor
    // and the nucleotide that reached it. Each `(node, q)` belongs to a single
    // position (`node.pos`) and is finalized once that position's frontier is
    // swept, so one flat map holds a unique back-link per state; the path is the
    // links walked back from `(to, q_to)`.
    type Link = (f64, Option<(Node, u32, Nucleotide)>);
    let mut best: FxHashMap<(Node, u32), Link> = FxHashMap::default();
    best.insert((from, q_from), (0.0, None));
    let mut frontier: Vec<(Node, u32)> = vec![(from, q_from)];
    for _ in from.pos..to.pos {
        let mut next: Vec<(Node, u32)> = Vec::new();
        for (node, q) in std::mem::take(&mut frontier) {
            let cai = best[&(node, q)].0;
            for e in dfa.right(node) {
                let Some(q2) = ac.step(node, q, e.nuc) else {
                    continue;
                };
                let cand = cai + e.cai;
                use std::collections::hash_map::Entry;
                match best.entry((e.to, q2)) {
                    Entry::Occupied(mut o) => {
                        if cand > o.get().0 {
                            o.insert((cand, Some((node, q, e.nuc))));
                        }
                    }
                    Entry::Vacant(v) => {
                        v.insert((cand, Some((node, q, e.nuc))));
                        next.push((e.to, q2));
                    }
                }
            }
        }
        frontier = next;
    }
    let mut out = Vec::with_capacity((to.pos - from.pos) as usize);
    let mut at = (to, q_to);
    while let Some(&(_, back)) = best.get(&at) {
        match back {
            Some((pnode, pq, nuc)) => {
                out.push(nuc);
                at = (pnode, pq);
            }
            None => break,
        }
    }
    out.reverse();
    out
}

/// One [`span_corners`] corner with its full spelling recovered: the strand's
/// nucleotides — the two terminal mismatches plus the best-CAI motif-free middle
/// — for the winning `(first, last, exit state)`. The traceback companion to
/// [`span_corners`]; the value scan selects a corner by its terminals and exit
/// state, and this respells the strand that realized it.
pub(crate) struct CornerSpelled {
    pub first: Nucleotide,
    pub last: Nucleotide,
    pub q_to: u32,
    pub cai: f64,
    pub seq: Vec<Nucleotide>,
}

/// The [`span_corners`] corners of `[from, to]`, each carrying the spelled
/// strand (via [`span_fill_path`] for the middle). Mirrors [`span_corners`]'s
/// per-`(first, last, exit state)` best-CAI selection, additionally recording
/// the coordinates needed to respell the middle that achieved it.
pub(crate) fn span_corners_spelled(
    dfa: &Dfa,
    ac: LatticeAc<'_>,
    from: Node,
    q_from: u32,
    to: Node,
) -> Vec<CornerSpelled> {
    if to.pos == from.pos + 1 {
        return dfa
            .right(from)
            .iter()
            .filter(|e| e.to == to)
            .filter_map(|e| {
                let q_to = ac.step(from, q_from, e.nuc)?;
                Some(CornerSpelled {
                    first: e.nuc,
                    last: e.nuc,
                    q_to,
                    cai: e.cai,
                    seq: vec![e.nuc],
                })
            })
            .collect();
    }
    // The winning provenance per `(first, last, exit state)`: the best CAI and
    // the middle-span coordinates (`first_to → last_src` landing at `q_mid`).
    struct Won {
        cai: f64,
        first_to: Node,
        q1: u32,
        last_src: Node,
        q_mid: u32,
    }
    let mut best: FxHashMap<(u8, u8, u32), Won> = FxHashMap::default();
    for first in dfa.right(from) {
        let Some(q1) = ac.step(from, q_from, first.nuc) else {
            continue;
        };
        for last in dfa.left(to) {
            for (q_mid, mid_cai) in span_fill(dfa, ac, first.to, q1, last.to) {
                let Some(q_to) = ac.step(last.to, q_mid, last.nuc) else {
                    continue;
                };
                let total = first.cai + mid_cai + last.cai;
                let slot = best
                    .entry((first.nuc as u8, last.nuc as u8, q_to))
                    .or_insert(Won {
                        cai: f64::NEG_INFINITY,
                        first_to: first.to,
                        q1,
                        last_src: last.to,
                        q_mid,
                    });
                if total > slot.cai {
                    *slot = Won {
                        cai: total,
                        first_to: first.to,
                        q1,
                        last_src: last.to,
                        q_mid,
                    };
                }
            }
        }
    }
    best.into_iter()
        .map(|((f, l, q_to), won)| {
            let first = nuc_from_index(f);
            let last = nuc_from_index(l);
            let mut seq = Vec::with_capacity((to.pos - from.pos) as usize);
            seq.push(first);
            seq.extend(span_fill_path(
                dfa,
                ac,
                won.first_to,
                won.q1,
                won.last_src,
                won.q_mid,
            ));
            seq.push(last);
            CornerSpelled {
                first,
                last,
                q_to,
                cai: won.cai,
                seq,
            }
        })
        .collect()
}

/// Relax `map[key]` upward to `cai`, keeping the larger CAI (a max-plus DP
/// step). Inserts on first sight.
fn relax<K: std::hash::Hash + Eq>(map: &mut FxHashMap<K, f64>, key: K, cai: f64) {
    map.entry(key)
        .and_modify(|w| {
            if cai > *w {
                *w = cai;
            }
        })
        .or_insert(cai);
}

// ===========================================================================
// The constrained CKY driver (optimum value; sequence traceback is separate).
// ===========================================================================

use rna_mfe::loops::{
    bulge_energy, internal_energy_scalar, pair_from_nucleotides, stack_energy, BulgeLoop,
    InternalLoopScalar, StackPair,
};
use rna_mfe::params::BasePair;
use rna_mfe::{hairpin_energy, hairpin_energy_scalar, Energy, HairpinLoop, Params};

use crate::dfa::Edge;
use crate::product::ProductDfa;

/// Smallest number of unpaired bases inside a hairpin loop (the "no sharp turn"
/// rule), matching [`crate::scan`].
const MIN_HAIRPIN_LOOP: u32 = 3;

/// Largest total unpaired count in a single-branch (bulge / internal) loop,
/// matching [`crate::scan`]'s `SINGLE_MAX_LEN`.
const SINGLE_MAX_LEN: u32 = 30;

/// A closed-pair (`P` / `Multi`) cell key under the constraint: the base
/// [`crate::state::PairKey`] fields plus the AC state at each of the two
/// boundary nodes. `q_open` is the AC state entering at the 5′ open node;
/// `q_outer` the state leaving at the 3′ outer node (after the closing edge).
/// The cell lives in the table column indexed by its outer node, so `outer` is
/// not stored here. Nucleotides are held as alphabet indices so the key derives
/// `Hash` ([`Nucleotide`] does not).
///
/// Each of the pair's two edges is named by its nucleotide **and** the lane it
/// lands in. A designed stop codon's first node has two out-edges spelling `U`
/// (the `UA` and `UG` lanes), so the nucleotide alone does not say which edge a
/// pair took there, and a traceback that re-derived the edge from it could
/// resume on the other lane.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PKey {
    open: NodeIx,
    close: NodeIx,
    nuc_open: u8,
    nuc_close: u8,
    lane_open: u8,
    lane_close: u8,
    q_open: u32,
    q_outer: u32,
}

impl PKey {
    /// The closed pair opened at `open` by edge `ep` from automaton state
    /// `q_open`, and closed at `close` by edge `eq` into state `q_outer`.
    pub(crate) fn of(
        (open, ep, q_open): (Node, Edge, u32),
        (close, eq, q_outer): (Node, Edge, u32),
    ) -> Self {
        Self {
            open: open.into(),
            close: close.into(),
            nuc_open: ep.nuc as u8,
            nuc_close: eq.nuc as u8,
            lane_open: ep.to.lane,
            lane_close: eq.to.lane,
            q_open,
            q_outer,
        }
    }

    /// The edge that opens the pair.
    fn open_edge(&self, dfa: &Dfa) -> Option<Edge> {
        edge_into(dfa, self.open.node(), self.nuc_open, self.lane_open)
    }

    /// The edge that closes the pair.
    fn close_edge(&self, dfa: &Dfa) -> Option<Edge> {
        edge_into(dfa, self.close.node(), self.nuc_close, self.lane_close)
    }

    /// The unconstrained lattice's key for this closed pair: the same nodes
    /// and nucleotides without the automaton states, which is how an outside
    /// table computed on that lattice is addressed.
    fn pair_key(&self) -> crate::state::PairKey {
        crate::state::PairKey {
            open: self.open.node(),
            close: self.close.node(),
            opening: crate::state::Opening::new(nuc_from_index(self.nuc_open), self.lane_open),
            nuc_close: nuc_from_index(self.nuc_close),
        }
    }

    /// The closing base pair (`nuc_open` · `nuc_close`).
    fn pair(&self) -> BasePair {
        pair_from_nucleotides(
            nuc_from_index(self.nuc_open),
            nuc_from_index(self.nuc_close),
        )
    }
}

// The two lanes ride in what was padding: the key is the fold's hash-map key,
// and its width is most of the table's memory.
const _: () = assert!(std::mem::size_of::<PKey>() == 20);

/// The out-edge of `from` spelling the nucleotide `nuc` (an alphabet index) and
/// landing in lane `lane`: one edge, even where two out-edges spell `nuc`.
fn edge_into(dfa: &Dfa, from: Node, nuc: u8, lane: u8) -> Option<Edge> {
    dfa.right(from)
        .iter()
        .find(|e| e.nuc as u8 == nuc && e.to.lane == lane)
        .copied()
}

/// A [`Node`] packed into its dense index, for the table keys: half the size
/// of the node itself, and the keys are most of what the fold hashes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct NodeIx(u32);

impl NodeIx {
    /// The flag marking a node on a lane past the second: a stop-labelled
    /// lattice's third stop lane. Every other node packs as `pos * 2 + lane`,
    /// the value a two-lane lattice's dense index gives it, so a key's hash and
    /// therefore a table's iteration order do not depend on whether the lattice
    /// could hold a third lane.
    const WIDE: u32 = 1 << 31;

    /// The node this index packs.
    fn node(self) -> Node {
        if self.0 & Self::WIDE == 0 {
            Node::new(self.0 / 2, (self.0 % 2) as u8)
        } else {
            let rest = self.0 & !Self::WIDE;
            Node::new(rest >> 2, (rest & 3) as u8)
        }
    }
}

impl From<Node> for NodeIx {
    fn from(node: Node) -> Self {
        debug_assert!(
            node.pos < 1 << 29,
            "a node's position fits the packed index"
        );
        if node.lane < 2 {
            NodeIx(node.pos * 2 + u32::from(node.lane))
        } else {
            NodeIx(Self::WIDE | node.pos << 2 | u32::from(node.lane))
        }
    }
}

/// A multi-branch (`M` / `M2`) cell key under the constraint: the 5′ inner
/// boundary node with its AC state and the 3′ outer boundary's AC state. The
/// cell lives in the column indexed by its outer node.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MKey {
    inner: NodeIx,
    q_inner: u32,
    q_outer: u32,
}

/// One CKY cell's accumulators: minimum free energy and log-CAI. The
/// AC-threaded analogue of [`crate::state::State`] without its provenance —
/// the traceback re-derives how each cell on the optimal path was reached
/// (see [`Folder::derive_p`] and its siblings), so the tables, which are most
/// of the fold's memory traffic, carry values only.
///
/// Packed to 4-byte alignment: with 8-byte alignment the `f64` would round
/// every map entry up by 4 bytes. Fields are only ever read by value.
#[derive(Clone, Copy)]
#[repr(C, packed(4))]
struct Cell {
    mfe: Energy,
    cai: f64,
}

/// A multi-loop opening with a long left flank, keyed for
/// [`Folder::far_multi`]: the cores' shared frontier node and AC state there,
/// and the opening's node, nucleotide and AC state before it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FarKey {
    outer: Node,
    q_outer: u32,
    open: Node,
    nuc: u8,
    q_open: u32,
}

/// The best core for one [`FarKey`] opening: the `M2R` core and its cell, the
/// opening edge, and the left flank's adaptation.
#[derive(Clone, Copy)]
struct FarWrap {
    core: MKey,
    cell: Cell,
    ep: Edge,
    left_cai: f64,
}

impl FarWrap {
    /// Keep this wrap at `key` unless one already there scores at least as
    /// well; a key seen for the first time is queued at its position.
    fn merge_into(
        self,
        far: &mut FxHashMap<FarKey, FarWrap>,
        at: &mut [Vec<FarKey>],
        key: FarKey,
        lambda: f64,
    ) {
        let score = |f: &FarWrap| brd(f.cell.mfe, f.cell.cai + f.ep.cai + f.left_cai, lambda);
        match far.get(&key) {
            Some(old) if score(old) <= score(&self) => {}
            Some(_) => {
                far.insert(key, self);
            }
            None => {
                far.insert(key, self);
                at[key.open.pos as usize].push(key);
            }
        }
    }
}

/// A reference to a closed-pair (`P` / `Multi`) cell: the outer column node and
/// the key that addresses it. The AC-threaded analogue of
/// [`crate::state::PairRef`].
#[derive(Clone, Copy)]
struct PCellRef {
    outer: NodeIx,
    key: PKey,
}

/// A reference to a multi-branch (`M` / `M2`) cell: the outer column node and
/// its key. The AC-threaded analogue of [`crate::state::SpanRef`].
#[derive(Clone, Copy)]
struct MCellRef {
    outer: NodeIx,
    key: MKey,
}

/// How a constrained CKY cell was derived — the predecessor cell(s), locally
/// chosen nucleotides, and the AC exit states the traceback needs to respell the
/// jumped-over loop interiors. The AC-threaded analogue of [`crate::state::Back`],
/// recovered by the traceback for the cells on the optimal path (see
/// [`Folder::derive_p`]) rather than stored in the tables.
///
/// It names only what the cell's own column and key cannot supply: a
/// predecessor in the same column under the same key carries no payload, a
/// closed pair's column is the node its closing edge leads to, and a
/// bifurcation's left `M` cell is pinned by the right pair's opening.
#[derive(Clone, Copy)]
enum CBack {
    /// Exterior base case: the empty prefix at the start node.
    ExtEmpty,
    /// Exterior extends by one unpaired nucleotide: `C[to] ← C[prev]`.
    ExtUnpaired {
        prev: NodeIx,
        prev_q: u32,
        /// Nucleotide (alphabet index) spelled at slot `prev.pos`.
        nuc: u8,
    },
    /// Exterior absorbs a closed pair: `C ← C[prefix] + P`. The pair sits in
    /// this cell's column; the prefix ends at its opening node, in its
    /// opening state.
    ExtPair { pair: PKey },
    /// Hairpin closure. `interior_q` is the AC state the loop interior lands in
    /// (before the closing nucleotide), pinning which per-exit interior spelling
    /// the seed chose.
    Hairpin { interior_q: u32 },
    /// Stack / bulge / internal closure around the inner pair `inner`, which
    /// sits in the column its closing edge leads to. `right_q` is the right
    /// flank's exit AC state (unused for an empty right flank).
    Single { inner: PKey, right_q: u32 },
    /// Multi-loop closure: `P ← Multi` at the same coordinates.
    MultiClose,
    /// Multi-loop assembly: `Multi` wraps an `M2R` core (an `M2` with its 3′
    /// unpaired flank already absorbed) with the 5′ unpaired flank, which is
    /// respelled best-CAI to the core's 5′ state. The core sits in the column
    /// of the multi-loop's closing node.
    MultiWrap { core: MKey },
    /// First multi-branch: `M ← P`, the pair in the same column.
    MFromP { pair: PKey },
    /// `M ← M2` at the same coordinates.
    MFromM2,
    /// `M` extends by one unpaired nucleotide on the right: the predecessor is
    /// `M[prev]` under this key with 3′ state `prev_q`.
    MUnpaired {
        prev: NodeIx,
        prev_q: u32,
        /// Nucleotide (alphabet index) spelled at slot `prev.pos`.
        nuc: u8,
    },
    /// Multi-branch bifurcation: `M2 ← M + P` at the shared split node. The
    /// right pair sits in this cell's column; the left `M` cell is in the
    /// column of the pair's opening node, under this key's 5′ boundary and the
    /// pair's opening state.
    M2Bifurcation { right: PKey },
    /// `M2R ← M2` at the same coordinates: the 3′ unpaired flank starts empty.
    M2RFromM2,
    /// `M2R` extends its 3′ unpaired flank by one nucleotide on the right, from
    /// `M2R[prev]` under this key with 3′ state `prev_q`.
    M2RUnpaired {
        prev: NodeIx,
        prev_q: u32,
        /// Nucleotide (alphabet index) spelled at slot `prev.pos`.
        nuc: u8,
    },
}

/// The constrained optimum: the `(mfe, cai)` of the best motif-free design. The
/// value-only projection the brute-force oracle checks against.
#[cfg(test)]
pub(crate) struct COptimum {
    pub mfe: Energy,
    pub cai: f64,
}

/// The AC-threaded CKY folder over one product DFA at a fixed λ.
struct Folder<'a> {
    product: &'a ProductDfa<'a>,
    dfa: &'a Dfa,
    ac: LatticeAc<'a>,
    params: &'a Params,
    lambda: f64,
    /// Region-pair constraint constraint, consulted at the single `P`-write choke
    /// point [`Self::put_p`] — the constrained analogue of the constraint
    /// [`crate::tables::Tables::update_p`] consults. Inert
    /// ([`PairConstraint::inert`]) on the CDS-only motif path, which is what keeps
    /// every previously shipped motif result byte-identical.
    region: PairConstraint,
    /// Positions required to carry a base pair, consulted wherever this folder
    /// declares a range unpaired. [`MustPairConstraint::inert`] leaves every such
    /// site on its unconstrained path, byte for byte.
    ///
    /// This folder skips the three interior reductions, so every range it
    /// declares is enumerated directly and each constraint call is a plain narrowing
    /// of a loop bound — there is no pooled `multi_open` here and so no
    /// eviction to mirror.
    must: MustPairConstraint,
    /// Per-column beam-pruning policy; [`Beam::exact`] prunes nothing (the exact
    /// `b → ∞` lattice, byte-for-byte).
    beam: Beam,
    /// Worker threads for the two fat inner phases (`p_push`, `m2_beam`). `1`
    /// (the default) runs the single-threaded path unchanged; `>1` fans each
    /// phase's expensive pass across scoped threads — see [`Self::fan_out`] —
    /// and folds the per-chunk buffers back, in chunk order, through the
    /// ordinary min-BRD merges. Because that order is the single-threaded
    /// write order for any thread count, the merges (including tie-breaks) are
    /// byte-identical to `threads = 1`, not merely objective-identical the way
    /// [`crate::scan`]'s atomic-counter fan-out is.
    threads: usize,
    /// The CAI-half completion bound per node index ([`Dfa::span_tail_cai`] at
    /// the node), used only by [`Beam::cai_bound`].
    /// Empty otherwise, so the exact and flat-BRD paths pay nothing for it.
    end_cai: Vec<f64>,
    /// Closed pairs, indexed by outer node ([`Dfa::ix`]).
    p: Vec<FxHashMap<PKey, Cell>>,
    /// Multi-loops in progress, same shape as `p`.
    multi: Vec<FxHashMap<PKey, Cell>>,
    /// Multi-branch components (≥1 branch), indexed by outer node.
    m: Vec<FxHashMap<MKey, Cell>>,
    /// Multi-branch components (≥2 branches), indexed by outer node.
    m2: Vec<FxHashMap<MKey, Cell>>,
    /// `M2` components followed by the multi-loop's 3′ unpaired flank, indexed
    /// by the node that ends the flank (the closing base's node). Keyed as `M2`
    /// is, with `q_outer` the state after the flank.
    m2r: Vec<FxHashMap<MKey, Cell>>,
    /// Scratch for [`Self::right_closings`], reused across inner pairs.
    rights: Rights,
    /// Emptied [`Writes`] kept for the next fan-out's chunks.
    spare: Mutex<Vec<Writes>>,
    /// The outside bound [`fold_bounded`] prunes on; `None` prunes nothing.
    bound: Option<OutsideBound<'a>>,
    /// Closed pairs the bound ruled out.
    pruned: AtomicUsize,
    /// Closures the bound's pair floor skipped before scoring them.
    cut: AtomicUsize,
    /// Of `pruned` and `cut`, those only [`OutsideBound::aware`] ruled out.
    aware_pruned: AtomicUsize,
    aware_cut: AtomicUsize,
    /// Memoized [`Self::left_openings`], keyed by a core's 5′ boundary.
    lefts: FxHashMap<(Node, u32), Vec<LeftOpen>>,
    /// [`Self::mark_cut_sides`]'s marks for one inner column node.
    cut_marks: CutMarks,
    /// The long-flank wraps' pair-floor verdicts at the current frontier.
    far_cuts: WrapCuts,
    /// Exterior parses, indexed by node then keyed by AC state.
    c: Vec<FxHashMap<u32, Cell>>,
    /// Memoized [`span_fill`] results — see [`SpanTable`].
    fills: SpanTable<(u32, f64)>,
    /// Memoized [`span_corners`] results — see [`SpanTable`].
    corners: SpanTable<CornerAc>,
    /// [`free_runs`] of the product when internal loops on free flanks score
    /// their corners separately; empty otherwise.
    free_run: Vec<u32>,
    /// Internal-loop closures scored by [`Self::separated_internal`].
    separated: AtomicUsize,
    /// Whether skewed internal loops on free flanks are pulled from [`Self::diag`]
    /// ([`Self::pull_skewed`]) and skipped by the push. Off under a must-pair
    /// constraint, whose unpaired ranges the pull does not consult.
    lz: bool,
    /// [`free_backs`] of `free_run` when `lz`; empty otherwise.
    free_back: Vec<u32>,
    /// The asymmetry at which the ninio term saturates.
    skew: usize,
    /// The closed pairs the pull reads, by size diagonal.
    diag: DiagIndex,
    /// Outer pairs the pull wrote.
    pulled: AtomicUsize,
    /// Where this fold's time went, when a development build asks.
    clock: PhaseClock,
}

/// Per-phase wall clock for one constrained fold, on in development builds
/// that ask (`Setting::Profile`).
///
/// The folder's six steps differ in what they iterate over: `seed_hairpins` and
/// `p_push` sweep span endpoints, while `m_extend` / `c_extend` / `m2_beam`
/// walk the surviving table cells. An attribution that merged them could not
/// say whether the folder's cost grows because more cells materialize or
/// because each cell costs more to close, which are different problems with
/// different fixes.
#[derive(Default)]
struct PhaseClock {
    on: bool,
    /// Accumulated time per step, in the order [`PHASE_NAMES`] lists them.
    spent: [Duration; 6],
}

/// The steps `PhaseClock` attributes to, in index order.
const PHASE_NAMES: [&str; 6] = [
    "seed_hairpins",
    "multi_close",
    "p_push",
    "m2_beam",
    "m_extend",
    "c_extend",
];

impl PhaseClock {
    fn from_env() -> Self {
        Self {
            on: crate::dev_env::var_os(crate::dev_env::Setting::Profile).is_some(),
            spent: Default::default(),
        }
    }

    /// Run `body`, charging its elapsed time to phase `idx`. Takes no clock
    /// reading when disabled.
    fn time<T>(&mut self, idx: usize, body: impl FnOnce() -> T) -> T {
        if !self.on {
            return body();
        }
        let started = Instant::now();
        let out = body();
        self.spent[idx] += started.elapsed();
        out
    }

    /// Emit the breakdown on stderr. A share is a fraction of the summed phase
    /// time, not of the whole fold: the prunes and the terminal minimum sit
    /// outside every phase, so the shares answer "where did the folding go"
    /// rather than "what fraction of the process was this".
    fn report(&self) {
        if !self.on {
            return;
        }
        let total: Duration = self.spent.iter().sum();
        let secs = total.as_secs_f64();
        eprintln!("constrained fold phases (total {secs:.3}s):");
        for (name, spent) in PHASE_NAMES.iter().zip(&self.spent) {
            let share = if secs > 0.0 {
                spent.as_secs_f64() / secs * 100.0
            } else {
                0.0
            };
            eprintln!("  {name:<14} {:>9.3}s  {share:>5.1}%", spent.as_secs_f64());
        }
    }
}

/// The best log-CAI of a motif-free spelling of the whole lattice, or `None`
/// when there is none.
///
/// Left wholly unpaired, that spelling is a design the constrained fold can
/// derive (the exterior absorbs one unpaired base at a time), with free energy
/// zero. So `-(1 - λ) · cai · 100` is an attainable objective at every λ: a
/// ceiling on the constrained optimum that costs one pass over the lattice.
/// It is a poor one except near λ = 0, where the objective is almost all
/// adaptation and it sits within `λ · |MFE|` of the optimum.
pub(crate) fn open_chain_cai(product: &ProductDfa) -> Option<f64> {
    let dfa = product.dfa();
    let ac = product.lattice_ac();
    span_fill(dfa, ac, dfa.start(), ac.start(), dfa.end())
        .into_iter()
        .map(|(_, cai)| cai)
        .reduce(f64::max)
}

/// The best log-CAI of a motif-free spelling from the lattice start to every
/// `(node, AC state)`, and from every `(node, AC state)` to the lattice end.
///
/// At λ = 0 the objective is adaptation alone and a completion's structure
/// costs nothing, so the best motif-free completion of a closed pair is the
/// best motif-free prefix into the pair's 5′ boundary state plus the best
/// motif-free suffix out of its 3′ state: these two tables give the **exact**
/// constrained outside at λ = 0, per AC state, for one pass over the product
/// each way. See [`ZeroChord`] for how a fold at λ > 0 uses it.
pub(crate) struct ChainCai {
    /// The lattice's dense node layout, which every table here is indexed by.
    layout: NodeLayout,
    /// Indexed by node ([`Dfa::ix`]), keyed by the AC state there.
    head: Vec<FxHashMap<u32, f64>>,
    tail: Vec<FxHashMap<u32, f64>>,
    /// The best log-CAI of a whole motif-free spelling that takes an edge
    /// spelling each nucleotide out of each node, indexed by node then
    /// nucleotide; `-∞` where none does.
    through_edge: Vec<[f64; 4]>,
}

impl ChainCai {
    pub(crate) fn build(product: &ProductDfa) -> Self {
        let dfa = product.dfa();
        let ac = product.lattice_ac();
        let (start, end) = (dfa.start(), dfa.end());
        let size = dfa.ix(end) + 2;
        let mut head: Vec<FxHashMap<u32, f64>> = vec![FxHashMap::default(); size];
        head[dfa.ix(start)].insert(ac.start(), 0.0);
        for pos in start.pos..end.pos {
            for &node in dfa.nodes_at(pos) {
                let here = std::mem::take(&mut head[dfa.ix(node)]);
                for (&q, &cai) in &here {
                    for e in dfa.right(node) {
                        if let Some(q2) = ac.step(node, q, e.nuc) {
                            relax(&mut head[dfa.ix(e.to)], q2, cai + e.cai);
                        }
                    }
                }
                head[dfa.ix(node)] = here;
            }
        }
        let mut tail: Vec<FxHashMap<u32, f64>> = vec![FxHashMap::default(); size];
        tail[dfa.ix(end)] = head[dfa.ix(end)].keys().map(|&q| (q, 0.0)).collect();
        for pos in (start.pos..end.pos).rev() {
            for &node in dfa.nodes_at(pos) {
                let mut best = FxHashMap::default();
                for &q in head[dfa.ix(node)].keys() {
                    for e in dfa.right(node) {
                        let Some(q2) = ac.step(node, q, e.nuc) else {
                            continue;
                        };
                        if let Some(&rest) = tail[dfa.ix(e.to)].get(&q2) {
                            relax(&mut best, q, e.cai + rest);
                        }
                    }
                }
                tail[dfa.ix(node)] = best;
            }
        }
        let mut through_edge = vec![[f64::NEG_INFINITY; 4]; size];
        for pos in start.pos..end.pos {
            for &node in dfa.nodes_at(pos) {
                for (&q, &before) in &head[dfa.ix(node)] {
                    for e in dfa.right(node) {
                        let Some(q2) = ac.step(node, q, e.nuc) else {
                            continue;
                        };
                        if let Some(&after) = tail[dfa.ix(e.to)].get(&q2) {
                            let slot = &mut through_edge[dfa.ix(node)][e.nuc as usize];
                            *slot = slot.max(before + e.cai + after);
                        }
                    }
                }
            }
        }
        Self {
            layout: dfa.layout(),
            head,
            tail,
            through_edge,
        }
    }

    /// An upper bound on the log-CAI of any whole motif-free spelling through
    /// `pair`: every such spelling takes both of the pair's edges.
    fn through_pair(&self, pair: &crate::state::PairKey) -> f64 {
        let at = |node: Node, nuc: Nucleotide| {
            self.through_edge
                .get(self.layout.ix(node))
                .map_or(f64::NEG_INFINITY, |row| row[nuc as usize])
        };
        at(pair.open, pair.nuc_open()).min(at(pair.close, pair.nuc_close))
    }

    /// The best log-CAI of a motif-free prefix reaching `open` in state
    /// `q_open` plus a motif-free suffix leaving `outer` in state `q_outer`;
    /// `None` where either side has no motif-free spelling.
    pub(crate) fn around(&self, open: Node, q_open: u32, outer: Node, q_outer: u32) -> Option<f64> {
        let before = self.head.get(self.layout.ix(open))?.get(&q_open)?;
        let after = self.tail.get(self.layout.ix(outer))?.get(&q_outer)?;
        Some(before + after)
    }
}

/// The chord between the exact motif-free outside at λ = 0 ([`ChainCai`]) and
/// an unconstrained outside table taken at an anchor `ν ≥ λ`, with
/// `weight = λ / ν` on the latter.
///
/// A constrained cell's best motif-free completion is a minimum of affine
/// functions of λ, so it is concave in λ and lies above the chord between its
/// values at 0 and at `ν`. Its value at 0 is the chain table's, exactly; its
/// value at `ν` is at least the unconstrained outside there. So
/// `weight · outside_ν − (1 − weight) · 100 · around` is a floor. Near λ = 0,
/// where the unconstrained outside counts adaptation the motif set forbids and
/// prunes almost nothing, this floor reads the motif-free adaptation exactly
/// and the energy at its full weight λ.
///
/// The same chord over whole designs gives a floor on a pair before any of
/// its cells is scored: at 0 every motif-free design through the pair scores
/// at least `−100 ·` [`ChainCai::through_pair`], and at `ν` at least the
/// unconstrained through table's value.
pub(crate) struct ZeroChord<'a> {
    pub chain: &'a ChainCai,
    /// The unconstrained anchor at `ν`: outside and best design through each
    /// pair.
    pub upper: &'a PairTable,
    pub weight: f64,
}

impl ZeroChord<'_> {
    /// Whether no motif-free design through `pair`, closing into outer node
    /// `o`, can score `ceiling` or better. `false` where the anchor's table
    /// does not cover the pair.
    fn rules_out_pair(&self, o: usize, pair: &crate::state::PairKey, ceiling: f64) -> bool {
        let Some(upper) = self.upper.get(o).and_then(|col| col.get(pair)) else {
            return false;
        };
        let upper = upper.through;
        if upper == f64::NEG_INFINITY {
            return false;
        }
        let at_zero = -100.0 * self.chain.through_pair(pair);
        self.weight * upper + (1.0 - self.weight) * at_zero > ceiling + BOUND_SLACK
    }

    fn floor(&self, on: Node, key: &PKey) -> Option<f64> {
        let upper = self
            .upper
            .get(self.chain.layout.ix(on))?
            .get(&key.pair_key())?
            .outside;
        let around = self
            .chain
            .around(key.open.node(), key.q_open, on, key.q_outer)?;
        Some(self.weight * upper - (1.0 - self.weight) * 100.0 * around)
    }
}

/// Compute the constrained BRD optimum over the motif-free feasible set. Returns
/// `None` when no motif-free encoding reaches the terminal (infeasible). The
/// value-only entry point the brute-force oracle checks; [`fold`] adds the
/// traceback.
#[cfg(test)]
pub(crate) fn optimum(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    region: PairConstraint,
) -> Option<COptimum> {
    let mut f = Folder::new(
        product,
        params,
        lambda,
        Beam::exact(),
        region,
        MustPairConstraint::inert(),
        1,
    );
    let end_q = f.run()?;
    let cell = f.c[f.dfa.ix(f.dfa.end())][&end_q];
    Some(COptimum {
        mfe: cell.mfe,
        cai: cell.cai,
    })
}

/// One exact constrained fold's optimum at each terminal of the lattice, in
/// [`Dfa::ends`] order, `None` at a terminal no motif-free design reaches.
#[cfg(test)]
pub(crate) fn fold_ends(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    region: PairConstraint,
) -> Vec<Option<ConstrainedFold>> {
    let mut f = Folder::new(
        product,
        params,
        lambda,
        Beam::exact(),
        region,
        MustPairConstraint::inert(),
        1,
    );
    f.run();
    let ends = f.dfa.ends().to_vec();
    ends.into_iter()
        .map(|end| {
            let q = f.best_state_at(end)?;
            let cell = f.c[f.dfa.ix(end)][&q];
            let (sequence, structure) = f.traceback_at(end, q);
            Some(ConstrainedFold {
                mfe: cell.mfe,
                cai: cell.cai,
                sequence,
                structure,
            })
        })
        .collect()
}

/// A folded constrained design: the optimum `(mfe, cai)` plus the spelled
/// motif-free coding sequence and its dot-bracket structure.
#[derive(Clone)]
pub(crate) struct ConstrainedFold {
    pub mfe: Energy,
    pub cai: f64,
    pub sequence: String,
    pub structure: String,
}

/// Fold the exact (`b → ∞`) constrained optimum and recover its sequence and
/// structure. Returns `None` when no motif-free encoding reaches the terminal
/// (infeasible). The exact-lattice shorthand for [`fold_with_beam`], used by the
/// value-and-sequence oracle tests; production callers reach it via
/// [`fold_with_beam`] with [`Beam::exact`].
#[cfg(test)]
pub(crate) fn fold(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    region: PairConstraint,
) -> Option<ConstrainedFold> {
    fold_with_beam(
        product,
        params,
        lambda,
        Beam::exact(),
        region,
        MustPairConstraint::inert(),
        1,
    )
}

/// Fold the constrained optimum under a beam-pruning policy and recover its
/// sequence and structure. [`Beam::exact`] runs the exact lattice byte-for-byte
/// (identical to [`fold`]); a finite [`Beam`] prunes each span column
/// (`P` / `Multi` / `M` / `M2`) to its lowest-BRD cells, an opt-in speed↔quality
/// knob that composes with the motif constraint. Feasibility is preserved by
/// construction — the AC product already removed the infeasible edges, and the
/// exterior spine `C` is never pruned — so a finite beam trades only quality,
/// never motif-freeness. Returns `None` when the feasible set is empty.
///
/// `threads` is the opt-in parallel fan-out over `p_push` and `m2_beam`; `1`
/// (and `0`) is byte-for-byte the single-threaded path above.
pub(crate) fn fold_with_beam(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    beam: Beam,
    region: PairConstraint,
    must: MustPairConstraint,
    threads: usize,
) -> Option<ConstrainedFold> {
    let mut f = Folder::new(product, params, lambda, beam, region, must, threads);
    let end_q = f.run()?;
    let cell = f.c[f.dfa.ix(f.dfa.end())][&end_q];
    let (sequence, structure) = f.traceback(end_q);
    Some(ConstrainedFold {
        mfe: cell.mfe,
        cai: cell.cai,
        sequence,
        structure,
    })
}

/// The materialized CKY cell counts (a memory measurement): the cells each
/// constrained table materializes, against the AC automaton's state count
/// `|A_M|` and the product-node count they are compared to. The exterior `C`
/// table is single-boundary (one cell per reachable `(node, q)`); the rest are
/// span tables. A characterization harness times a fold around this value-only
/// run to pair the counts with wall-clock and memory.
pub(crate) struct SpanCellCounts {
    pub p_cells: usize,
    pub multi_cells: usize,
    pub m_cells: usize,
    pub m2_cells: usize,
    pub c_cells: usize,
    pub ac_states: usize,
    pub product_nodes: usize,
}

/// Fold the exact (`b → ∞`) constrained lattice and report the materialized
/// cell counts against `|A_M|` — the layout measurement. Returns `None` for an
/// infeasible set.
pub(crate) fn span_cell_counts(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    region: PairConstraint,
) -> Option<SpanCellCounts> {
    let mut f = Folder::new(
        product,
        params,
        lambda,
        Beam::exact(),
        region,
        MustPairConstraint::inert(),
        1,
    );
    f.run()?;
    Some(SpanCellCounts {
        p_cells: f.p.iter().map(FxHashMap::len).sum(),
        multi_cells: f.multi.iter().map(FxHashMap::len).sum(),
        m_cells: f.m.iter().map(FxHashMap::len).sum(),
        m2_cells: f.m2.iter().map(FxHashMap::len).sum(),
        c_cells: f.c.iter().map(FxHashMap::len).sum(),
        ac_states: f.ac.num_states(),
        product_nodes: product.num_product_nodes(),
    })
}

/// The motif-free outside of each closed pair of the constrained lattice at
/// one λ, indexed by outer node and keyed as `P`; a pair on no motif-free
/// design has no entry.
pub(crate) type MotifFreeTable = [FxHashMap<PKey, f64>];

/// The chord between two [`MotifFreeTable`]s taken at anchors bracketing a
/// fold's λ, `weight` on the upper one: a floor on every motif-free completion,
/// since each pair's best one is concave in λ.
#[derive(Clone, Copy)]
pub(crate) struct MotifFreeChord<'a> {
    pub lo: &'a MotifFreeTable,
    pub hi: &'a MotifFreeTable,
    pub weight: f64,
    /// Whether both tables come from exact folds, so a pair missing from
    /// either lies on no motif-free design. A pruned table's missing pair says
    /// nothing.
    pub complete: bool,
}

/// One side of an [`AwareChord`]: the exact motif-free completions at λ = 0,
/// or a pass over the constrained lattice at a weight above it.
#[derive(Clone, Copy)]
pub(crate) enum AwareSide<'a> {
    /// At λ = 0 a completion's structure costs nothing, so [`ChainCai`] gives
    /// the best motif-free completion of every cell exactly, and bounds the
    /// adaptation of every design through a pair.
    Zero(&'a ChainCai),
    Pass(&'a outside::AwarePass),
}

/// What one side of an [`AwareChord`] floors a closed pair's completion
/// at, for a point the side's own pass may not have built.
#[derive(Clone, Copy)]
enum SideTerm {
    /// A floor on every completion, whatever the interior.
    Fixed(f64),
    /// The pass never built the cell, so the completion scores above this
    /// ceiling less the interior's score at the side's weight.
    Above(f64),
}

/// What an [`AwareChord`] holds the points of one closed pair to.
pub(crate) enum PairLines {
    /// No motif-free design through the pair scores within the ceiling.
    Out,
    /// Each point's score at each weight must be at most its limit.
    Hold(SmallVec<[(f64, f64); 2]>),
}

impl AwareSide<'_> {
    /// The side's weight.
    fn lambda(&self) -> f64 {
        match self {
            Self::Zero(_) => 0.0,
            Self::Pass(pass) => pass.lambda(),
        }
    }

    /// A floor on every motif-free completion of `P[on][key]`, whose inside
    /// at the fold's weight is `cell`, at this side's weight; `None` where the
    /// side does not cover the cell.
    fn cell(&self, on: Node, key: &PKey, cell: Cell) -> Option<f64> {
        match self {
            Self::Zero(chain) => {
                let around = chain.around(key.open.node(), key.q_open, on, key.q_outer)?;
                Some(-100.0 * around)
            }
            Self::Pass(pass) => Some(pass.cell_floor(on, key, cell)),
        }
    }

    /// The side's floor on completions of `P[on][key]` before an interior is
    /// known; `None` where no motif-free design passes through the cell.
    fn term(&self, on: Node, key: &PKey) -> Option<SideTerm> {
        match self {
            Self::Zero(chain) => {
                let around = chain.around(key.open.node(), key.q_open, on, key.q_outer)?;
                Some(SideTerm::Fixed(-100.0 * around))
            }
            Self::Pass(pass) => Some(pass.cell_term(on, key)),
        }
    }

    /// A floor on every motif-free design through `pair`, closing into outer
    /// node `o`, at this side's weight; `None` where the side does not cover
    /// the pair.
    fn pair(&self, o: usize, pair: &crate::state::PairKey) -> Option<f64> {
        match self {
            Self::Zero(chain) => Some(-100.0 * chain.through_pair(pair)),
            Self::Pass(pass) => Some(pass.pair_floor(o, pair)),
        }
    }
}

/// The chord of two [`AwareSide`]s bracketing a fold's λ: a floor under every
/// motif-free completion of a cell and every motif-free design through a
/// pair.
///
/// Each side's value is a floor on a minimum over motif-free designs of a
/// quantity affine in λ, which is concave in λ, so the chord of the two lies
/// below the value between them. Unlike [`OutsideBound::outside`], the sides
/// read the constrained lattice, so a completion the motif set forbids does
/// not lower them.
#[derive(Clone, Copy)]
pub(crate) struct AwareChord<'a> {
    pub near: AwareSide<'a>,
    /// The other side and the weight on `near`, unless λ is `near`'s own.
    pub far: Option<(AwareSide<'a>, f64)>,
}

impl AwareChord<'_> {
    /// The limits the chord puts on the points of `P[on][key]` at weight
    /// `lambda`, for a fold that holds points to lines rather than reading
    /// one cell's inside, under `ceiling`; also [`PairLines::Out`] where its
    /// pair floor rules the pair out.
    ///
    /// A point `p` through the cell scores at least
    /// `brd_λ(p) + w·s_near + (1−w)·s_far`, each side term a fixed floor or
    /// that side's ceiling less `brd_side(p)`. With `brd_λ = w·brd_near +
    /// (1−w)·brd_far`, a fixed pair holds `p` at `λ`, one side above holds it
    /// at the other side's weight, and two sides above leave no point term:
    /// the pair is out or untouched.
    pub(crate) fn pair_lines(
        &self,
        (on, key): (Node, &PKey),
        layout: NodeLayout,
        lambda: f64,
        ceiling: f64,
    ) -> PairLines {
        if self
            .pair_floor(layout.ix(on), &key.pair_key())
            .is_some_and(|f| f > ceiling + BOUND_SLACK)
        {
            return PairLines::Out;
        }
        let Some(near) = self.near.term(on, key) else {
            return PairLines::Out;
        };
        let mut lines = SmallVec::new();
        let Some((far_side, w)) = self.far else {
            match near {
                SideTerm::Fixed(v) => lines.push((lambda, ceiling - v)),
                SideTerm::Above(c) if c > ceiling + BOUND_SLACK => return PairLines::Out,
                SideTerm::Above(_) => {}
            }
            return PairLines::Hold(lines);
        };
        let Some(far) = far_side.term(on, key) else {
            return PairLines::Out;
        };
        let (l_near, l_far) = (self.near.lambda(), far_side.lambda());
        match (near, far) {
            (SideTerm::Fixed(a), SideTerm::Fixed(b)) => {
                lines.push((lambda, ceiling - w * a - (1.0 - w) * b));
            }
            (SideTerm::Fixed(a), SideTerm::Above(c)) if w > 0.0 => {
                lines.push((l_near, (ceiling - (1.0 - w) * c) / w - a));
            }
            (SideTerm::Above(c), SideTerm::Fixed(b)) if w < 1.0 => {
                lines.push((l_far, (ceiling - w * c) / (1.0 - w) - b));
            }
            (SideTerm::Above(c), SideTerm::Above(d))
                if w * c + (1.0 - w) * d > ceiling + BOUND_SLACK =>
            {
                return PairLines::Out;
            }
            _ => {}
        }
        PairLines::Hold(lines)
    }

    fn chord(&self, read: impl Fn(&AwareSide<'_>) -> Option<f64>) -> Option<f64> {
        let near = read(&self.near)?;
        match &self.far {
            None => Some(near),
            Some((far, weight)) => Some(weight * near + (1.0 - weight) * read(far)?),
        }
    }

    fn cell_floor(&self, on: Node, key: &PKey, cell: Cell) -> Option<f64> {
        self.chord(|side| side.cell(on, key, cell))
    }

    fn pair_floor(&self, o: usize, pair: &crate::state::PairKey) -> Option<f64> {
        self.chord(|side| side.pair(o, pair))
    }
}

/// What an unconstrained anchor knows about one closed pair: its best
/// completion (the outside) and the best whole design through it (its inside
/// plus its outside), `−∞` where the pair has no inside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairBound {
    pub outside: f64,
    pub through: f64,
}

/// Per closed pair on the unconstrained lattice, indexed by outer node and
/// keyed by the pair: one map for both values, so an anchor keeps its key set
/// once.
pub(crate) type PairTable = [PairColumn];

/// One outer node's [`PairBound`]s, keyed by the pair packed into one word.
/// A [`crate::state::PairKey`] pads to 20 bytes and an entry to 40; packed,
/// an entry is 24, and the anchors' tables are most of a long walk's peak.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PairColumn(FxHashMap<u64, StoredBound>);

/// A [`PairBound`] as a column holds it: each value in single precision,
/// rounded down, so it stays a floor. A pair ruled out at full precision
/// within a few thousandths of the ceiling may be kept; none is cut that
/// should not be.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StoredBound {
    outside: f32,
    through: f32,
}

/// The largest single-precision value not above `x`.
fn floor_f32(x: f64) -> f32 {
    let y = x as f32;
    if f64::from(y) > x {
        y.next_down()
    } else {
        y
    }
}

impl From<PairBound> for StoredBound {
    fn from(b: PairBound) -> Self {
        Self {
            outside: floor_f32(b.outside),
            through: floor_f32(b.through),
        }
    }
}

impl From<StoredBound> for PairBound {
    fn from(b: StoredBound) -> Self {
        Self {
            outside: f64::from(b.outside),
            through: f64::from(b.through),
        }
    }
}

/// Bits per node position in a packed pair: positions run to `3m`, so up to
/// 22 million residues.
const PACK_POS: u32 = 26;

/// A node and nucleotide in the low [`PACK_SIDE`] bits, position lowest. Only
/// the lane's low bit is here; [`pack_pair`] keeps the high bit of each of its
/// three lanes above both sides, where a two-lane lattice's keys leave zeros.
fn pack_node(node: Node, nuc: Nucleotide) -> u64 {
    debug_assert!(node.pos < 1 << PACK_POS && node.lane < 4);
    u64::from(node.pos) | (u64::from(node.lane & 1) << PACK_POS) | ((nuc as u64) << (PACK_POS + 1))
}

#[cfg(test)]
fn unpack_node(bits: u64) -> (Node, Nucleotide) {
    let nuc = match bits >> (PACK_POS + 1) & 3 {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        _ => Nucleotide::U,
    };
    let node = Node::new(
        bits as u32 & ((1 << PACK_POS) - 1),
        (bits >> PACK_POS & 1) as u8,
    );
    (node, nuc)
}

/// The bit above both sides of a packed pair holding the opening edge's lane's
/// low bit; the three lanes' high bits follow it — opening, open node, close
/// node.
const PACK_LANES: u32 = 2 * PACK_SIDE;

const PACK_SIDE: u32 = PACK_POS + 3;

/// The open side in the low bits: an FxHash of one word is a multiply, so a
/// bucket index reads only the key's low bits, and within a column the opening
/// is what varies (the closing node takes a few values). Close side low put
/// whole columns in a handful of buckets.
///
/// The opening edge's lane sits above both sides: a designed stop's first node
/// has two `U` edges, so the open node and nucleotide alone do not name it.
fn pack_pair(pair: &crate::state::PairKey) -> u64 {
    let (opening, open, close) = (pair.lane_open(), pair.open.lane, pair.close.lane);
    debug_assert!(opening < 4);
    let high = |lane: u8, at: u32| u64::from(lane >> 1) << (PACK_LANES + at);
    (u64::from(opening & 1) << PACK_LANES)
        | high(opening, 1)
        | high(open, 2)
        | high(close, 3)
        | (pack_node(pair.close, pair.nuc_close) << PACK_SIDE)
        | pack_node(pair.open, pair.nuc_open())
}

#[cfg(test)]
fn unpack_pair(bits: u64) -> crate::state::PairKey {
    let side = (1 << PACK_SIDE) - 1;
    let high = |at: u32| ((bits >> (PACK_LANES + at) & 1) as u8) << 1;
    let (mut close, nuc_close) = unpack_node(bits >> PACK_SIDE & side);
    let (mut open, nuc_open) = unpack_node(bits & side);
    open.lane |= high(2);
    close.lane |= high(3);
    let opening = (bits >> PACK_LANES & 1) as u8 | high(1);
    crate::state::PairKey {
        open,
        close,
        opening: crate::state::Opening::new(nuc_open, opening),
        nuc_close,
    }
}

impl PairColumn {
    pub(crate) fn get(&self, pair: &crate::state::PairKey) -> Option<PairBound> {
        self.0.get(&pack_pair(pair)).map(|&b| b.into())
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (crate::state::PairKey, PairBound)> + '_ {
        self.0.iter().map(|(&k, &b)| (unpack_pair(k), b.into()))
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Every entry under its packed pair, with the bits of its stored
    /// outside and through, for a copy that [`Self::from_packed`] rebuilds
    /// bit for bit.
    pub(crate) fn packed(&self) -> impl Iterator<Item = (u64, [u32; 2])> + '_ {
        self.0
            .iter()
            .map(|(&k, b)| (k, [b.outside.to_bits(), b.through.to_bits()]))
    }

    pub(crate) fn from_packed(entries: impl IntoIterator<Item = (u64, [u32; 2])>) -> Self {
        Self(
            entries
                .into_iter()
                .map(|(k, [outside, through])| {
                    let bound = StoredBound {
                        outside: f32::from_bits(outside),
                        through: f32::from_bits(through),
                    };
                    (k, bound)
                })
                .collect(),
        )
    }
}

/// [`PairTable`] from an outside pass's per-pair values and, for each outer
/// node and pair, the pair's inside where it has one.
pub(crate) fn pair_table(
    outside: Vec<FxHashMap<crate::state::PairKey, f64>>,
    inside: impl Fn(usize, &crate::state::PairKey) -> Option<f64>,
) -> Vec<PairColumn> {
    outside
        .into_iter()
        .enumerate()
        .map(|(o, column)| {
            PairColumn(
                column
                    .into_iter()
                    .map(|(key, out)| {
                        let through = inside(o, &key).map_or(f64::NEG_INFINITY, |s| s + out);
                        (
                            pack_pair(&key),
                            PairBound {
                                outside: out,
                                through,
                            }
                            .into(),
                        )
                    })
                    .collect(),
            )
        })
        .collect()
}

/// An admissible bound on what surrounds each closed pair, and the objective
/// a design must reach to matter.
///
/// `outside[o][pair]` is the best completion of `pair` (closing into outer
/// node `o`) over the **unconstrained** lattice. A motif-free completion is
/// one of those, so it is never better: `brd(cell) + outside` is a floor
/// under every motif-free design through the cell. A cell whose floor is
/// above `ceiling` is on no design scoring `ceiling` or better.
pub(crate) struct OutsideBound<'a> {
    pub outside: &'a PairTable,
    /// A second outside table and the weight on the first, when the bound is
    /// the chord between two λ anchors rather than one exact table. The
    /// objective is affine in λ, so each cell's outside value is a minimum of
    /// affine functions and therefore concave in λ: the chord between any two
    /// anchors bracketing the target lies **below** the true value, which is
    /// what a floor must do. `None` reads `outside` alone.
    pub chord: Option<(&'a PairTable, f64)>,
    /// Whether to cut on the best whole design through each pair on the
    /// unconstrained lattice ([`PairBound::through`]), at the anchor `outside`
    /// came from and, when `chord` is set, at the far one. Like the outside, it
    /// is a minimum of affine functions of λ and so concave, and the chord of
    /// the two is a floor under every design through the pair whatever its
    /// interior. That lets a fold skip **producing** a cell whose pair it rules
    /// out, where `outside` can only rule a cell out once its interior has
    /// been scored.
    pub through: bool,
    /// Further floors under each constrained cell, from the exact motif-free
    /// outside at λ = 0 (see [`ZeroChord`]); a cell is ruled out when any
    /// floor rules it out. Empty adds none.
    pub zero: &'a [ZeroChord<'a>],
    /// A further floor from motif-free outside tables at two anchors (see
    /// [`MotifFreeChord`]); `None` adds none.
    pub motif_free: Option<MotifFreeChord<'a>>,
    /// Floors from passes over the constrained lattice itself, which read
    /// what the motif set costs a completion rather than only what the
    /// unconstrained lattice allows: a cell floor and a pair floor, consulted
    /// after every other floor. `None` adds none.
    pub aware: Option<AwareChord<'a>>,
    pub ceiling: f64,
}

/// Which floor ruled a cell or pair out: one the bound holds besides
/// [`OutsideBound::aware`], or that one alone.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Floor {
    Plain,
    Aware,
}

impl OutsideBound<'_> {
    /// Whether no design through `pair`, closing into outer node `o`, can
    /// score `ceiling` or better, and by which floor: the zero chords', the
    /// [`Self::through`] floor, or [`Self::aware`]'s pair floor. `None` where
    /// no floor covering the pair rules it out.
    fn rules_out_pair(&self, o: usize, pair: &crate::state::PairKey) -> Option<Floor> {
        if self
            .zero
            .iter()
            .any(|z| z.rules_out_pair(o, pair, self.ceiling))
            || self.through_rules_out(o, pair)
        {
            return Some(Floor::Plain);
        }
        self.aware
            .and_then(|a| a.pair_floor(o, pair))
            .is_some_and(|f| f > self.ceiling + BOUND_SLACK)
            .then_some(Floor::Aware)
    }

    /// Whether the [`Self::through`] floor rules `pair` out; `false` where
    /// it does not cover the pair.
    fn through_rules_out(&self, o: usize, pair: &crate::state::PairKey) -> bool {
        if !self.through {
            return false;
        }
        let Some(here) = self.outside.get(o).and_then(|col| col.get(pair)) else {
            return false;
        };
        let floor = match self.chord {
            Some((far, weight)) => {
                let Some(there) = far.get(o).and_then(|col| col.get(pair)) else {
                    return false;
                };
                weight * here.through + (1.0 - weight) * there.through
            }
            None => here.through,
        };
        // A pair without an inside at either anchor has no through floor;
        // −∞ there makes this false (or NaN at a zero weight, also false).
        floor > self.ceiling + BOUND_SLACK
    }

    /// The floor this bound puts under completions of `pair` closing into
    /// outer node `o`, or `None` where it covers the pair in neither table.
    fn floor(&self, o: usize, pair: &crate::state::PairKey) -> Option<f64> {
        let near = self.outside.get(o)?.get(pair)?.outside;
        match self.chord {
            None => Some(near),
            Some((far, weight)) => {
                let far = far.get(o)?.get(pair)?.outside;
                Some(weight * near + (1.0 - weight) * far)
            }
        }
    }
}

/// Slack in the bound's comparison: the outside and the fold sum the same
/// terms in different orders, so equal objectives can differ in the last
/// bits. Far below any real difference in objective.
pub(crate) const BOUND_SLACK: f64 = 1e-6;

/// Fold the exact constrained lattice, skipping every closed pair the bound
/// rules out. Exact for any design scoring `bound.ceiling` or better: every
/// cell on such a design's derivation passes the bound. So when the returned
/// design scores at or below the ceiling it is the constrained optimum; above
/// it (or `None`), the ceiling was below the optimum and nothing is
/// certified. Also returns what the bound ruled out ([`BoundCounts`]).
///
/// `spans` carries the span memos from fold to fold over the same product:
/// taken here if present, and left holding this fold's on return.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_bounded(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    region: PairConstraint,
    must: MustPairConstraint,
    threads: usize,
    bound: OutsideBound,
    spans: &mut Option<SpanMemo>,
) -> (Option<ConstrainedFold>, BoundCounts) {
    let mut f = Folder::with_spans(
        product,
        params,
        lambda,
        Beam::exact(),
        region,
        must,
        threads,
        spans.take(),
    );
    f.bound = Some(bound);
    let fold = f.run().map(|end_q| {
        let cell = f.c[f.dfa.ix(f.dfa.end())][&end_q];
        let (sequence, structure) = f.traceback(end_q);
        ConstrainedFold {
            mfe: cell.mfe,
            cai: cell.cai,
            sequence,
            structure,
        }
    });
    let counts = f.bound_counts();
    *spans = Some(SpanMemo {
        fills: f.fills,
        corners: f.corners,
    });
    (fold, counts)
}

/// What a bounded fold's floors ruled out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BoundCounts {
    /// Closed pairs the bound ruled out once scored.
    pub pruned: usize,
    /// Closures its pair floors skipped before scoring them.
    pub cut: usize,
    /// Of `pruned` and `cut`, those only [`OutsideBound::aware`]'s floors
    /// ruled out: what it adds to the bound without it.
    pub aware_pruned: usize,
    pub aware_cut: usize,
}

impl std::ops::AddAssign for BoundCounts {
    fn add_assign(&mut self, o: Self) {
        self.pruned += o.pruned;
        self.cut += o.cut;
        self.aware_pruned += o.aware_pruned;
        self.aware_cut += o.aware_cut;
    }
}

/// Fold the constrained lattice under a finite `beam` and no bound: a
/// motif-free design, but only an approximate one — a beam drops cells, so the
/// result is not a minimum on the exact lattice.
///
/// Its use is as a **ceiling**. Every design the beam can emit threads the
/// automaton, so it is feasible, so its objective is at or above the exact
/// optimum and bounds it from above. `None` where the beam pruned away every
/// completion, which says nothing about feasibility.
pub(crate) fn fold_beamed(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    region: PairConstraint,
    must: MustPairConstraint,
    threads: usize,
    beam: Beam,
) -> Option<ConstrainedFold> {
    let mut f = Folder::new(product, params, lambda, beam, region, must, threads);
    let end_q = f.run()?;
    let cell = f.c[f.dfa.ix(f.dfa.end())][&end_q];
    let (sequence, structure) = f.traceback(end_q);
    Some(ConstrainedFold {
        mfe: cell.mfe,
        cai: cell.cai,
        sequence,
        structure,
    })
}

/// How many constrained `P` cells an outside bound would rule out: the
/// measurement that decides whether pruning on one pays before it is built.
pub(crate) struct PruneCensus {
    pub p_cells: usize,
    /// Cells whose inside plus the bound exceeds `ceiling`.
    pub prunable: usize,
    /// Cells whose pair the `through` floor rules out on its own, without
    /// reading the cell's interior. Counted only when a `through` table is
    /// given, over the same cells as `prunable` so the two are comparable.
    pub prunable_through: usize,
    /// Cells neither floor rules out: what a tighter bound has left to take.
    /// This, not `p_cells - prunable`, is the headroom, because production
    /// consults both floors.
    pub prunable_either: usize,
    /// Cells the **exact constrained** completion at λ = 0 rules out, from
    /// [`ChainCai`]. The caller must pass the chain **only at λ = 0**; this
    /// is a diagnostic and **not a sound floor at λ > 0**:
    /// at λ = 0 a completion's structure costs nothing, so the chain gives the
    /// exact constrained outside, but above it the energy term is missing and
    /// production chords the value against an unconstrained anchor rather than
    /// reading it alone ([`ZeroChord`]). The column answers one question only:
    /// how much reach a constrained bound would have, measured where it is
    /// computable without a product-lattice outside pass.
    pub prunable_chain: usize,
    /// Cells any of the three floors rules out.
    pub prunable_any: usize,
    /// Cells with no bound (their pair is absent from `outside`).
    pub unbounded: usize,
    /// The constrained optimum, on the folder's BRD scale.
    pub optimum: f64,
}

/// Fold exactly, then count the `P` cells whose BRD plus the outside value of
/// their base pair in `outside` (indexed by outer node, as `P` is) exceeds
/// `ceiling`, or the fold's own optimum when `ceiling` is `None`. Measurement
/// only; nothing is pruned.
pub(crate) fn prune_census(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    outside: &[FxHashMap<crate::state::PairKey, f64>],
    through: Option<&[FxHashMap<crate::state::PairKey, f64>]>,
    chain: Option<&ChainCai>,
    ceiling: Option<f64>,
) -> Option<PruneCensus> {
    let mut f = Folder::new(
        product,
        params,
        lambda,
        Beam::exact(),
        PairConstraint::inert(),
        MustPairConstraint::inert(),
        1,
    );
    let end_q = f.run()?;
    let end = f.c[f.dfa.ix(f.dfa.end())][&end_q];
    let optimum = brd(end.mfe, end.cai, lambda);
    let ceiling = ceiling.unwrap_or(optimum);
    let mut census = PruneCensus {
        p_cells: 0,
        prunable: 0,
        prunable_through: 0,
        prunable_either: 0,
        prunable_chain: 0,
        prunable_any: 0,
        unbounded: 0,
        optimum,
    };
    for (o, column) in f.p.iter().enumerate() {
        for (key, cell) in column {
            census.p_cells += 1;
            let pair = key.pair_key();
            // The `through` floor is a property of the pair alone, so it is
            // read whether or not the outside table covers the pair: that is
            // how a fold can skip producing the cell at all.
            let by_through = through
                .and_then(|t| t.get(o))
                .and_then(|col| col.get(&pair))
                .is_some_and(|&v| v > ceiling + 1e-6);
            if by_through {
                census.prunable_through += 1;
            }
            let by_outside = match outside.get(o).and_then(|col| col.get(&pair)) {
                None => {
                    census.unbounded += 1;
                    false
                }
                Some(&out) => brd(cell.mfe, cell.cai, lambda) + out > ceiling + 1e-6,
            };
            if by_outside {
                census.prunable += 1;
            }
            if by_outside || by_through {
                census.prunable_either += 1;
            }
            let by_chain = chain.is_some_and(|c| -100.0 * c.through_pair(&pair) > ceiling + 1e-6);
            if by_chain {
                census.prunable_chain += 1;
            }
            if by_outside || by_through || by_chain {
                census.prunable_any += 1;
            }
        }
    }
    Some(census)
}

/// Below this many top-level items, a multi-threaded phase's fat pass runs
/// serially into one buffer, folded immediately — too little work to repay
/// thread-spawn overhead. Matches the order of [`crate::scan`]'s
/// `PUSH_PARALLEL_MIN` / `PULL_PARALLEL_MIN`.
const PARALLEL_MIN: usize = 64;

/// Chunks per worker in [`Folder::fan_out`]: enough that a slow chunk near the
/// end of a column does not leave the other workers idle, few enough that
/// claiming one stays cheap against the work in it.
const CHUNKS_PER_WORKER: usize = 8;

/// Table writes buffered from one chunk of a multi-threaded phase (`p_push`'s
/// single-branch push into `P`, `m2_beam`'s wrap into `Multi`), applied
/// afterwards through the ordinary min-BRD merges.
///
/// Like [`crate::scan::Writes`], [`Folder::fan_out`]'s workers claim chunks
/// from a shared atomic counter, so which worker walks which chunk depends on
/// how the work happened to divide at runtime. Unlike it, each chunk is a
/// contiguous, order-preserving run of the flat item list with a buffer of its
/// own, and the buffers are concatenated in **chunk** order once every worker
/// has finished. That reproduces exactly the sequence a single serial walk over
/// the whole list would have produced, for **any** number of workers and any
/// claiming order — so the final merge order this phase feeds `P` / `Multi` is
/// independent of `threads`, and the byte-exactness obligation
/// ([`crate::scan`]'s "only equal-BRD traceback structure may differ") is met
/// by construction here rather than only empirically.
#[derive(Default)]
struct Writes {
    /// `P` candidates, region verdict already applied.
    p: Pending<PKey>,
    multi: Pending<PKey>,
    /// `M2` bifurcation candidates.
    m2: Pending<MKey>,
    /// Scratch for [`Folder::right_closings_ro`], reused across the chunk's
    /// inner pairs; never applied.
    rights: Rights,
    /// The chunk's multi-loop wraps' pair-floor verdicts
    /// ([`Folder::pair_out_cached`]).
    wrap_cuts: WrapCuts,
}

impl Writes {
    /// Empty the buffers, keeping their capacity for the next chunk, and set
    /// how many column buckets the next chunk sorts its survivors into.
    fn reset(&mut self, buckets: usize) {
        self.p.reset(buckets);
        self.multi.reset(buckets);
        self.m2.reset(buckets);
        self.wrap_cuts.clear();
    }
}

/// Pair-floor verdicts of multi-loop wraps closing at one frontier node:
/// every core closing there asks about the same openings, so one verdict
/// serves them all. Dense over (opening node, opening edge's lane and
/// nucleotide, closing nucleotide), since the closing node is fixed; the cuts found are tallied
/// here and added to the folder's count once ([`Folder::flush_cuts`]).
#[derive(Default)]
struct WrapCuts {
    close: Option<Node>,
    /// 0 unknown, 1 kept, 2 cut, 3 cut by the aware pair floor alone.
    verdicts: Vec<u8>,
    cuts: usize,
    aware_cuts: usize,
}

impl WrapCuts {
    /// Verdict slots per opening node: the opening edge's lane (up to three)
    /// and nucleotide, times the closing nucleotide.
    const PER_OPEN: usize = 48;

    /// Forget every verdict (the tally is kept until flushed).
    fn clear(&mut self) {
        self.close = None;
    }

    /// The slot for an opening, after starting afresh if the frontier moved.
    fn slot(
        &mut self,
        layout: NodeLayout,
        close: Node,
        open: Node,
        ep: Edge,
        nuc_close: Nucleotide,
    ) -> &mut u8 {
        if self.close != Some(close) {
            self.close = Some(close);
            self.verdicts.clear();
            self.verdicts.resize(layout.ix(close) * Self::PER_OPEN, 0);
        }
        let opening = usize::from(ep.to.lane) * 4 + ep.nuc as usize;
        &mut self.verdicts[layout.ix(open) * Self::PER_OPEN + opening * 4 + nuc_close as usize]
    }
}

/// One chunk's candidates for one table, reduced to the best per cell as they
/// arrive.
///
/// Buffering every candidate and merging them all afterwards would make the
/// serial merge as long as the parallel walk that produced them, and the
/// buffers alone run to gigabytes on a long molecule. Reducing here keeps one
/// candidate per cell. It changes nothing the table ends up holding: the
/// survivor is the first minimum in arrival order (a later candidate replaces
/// it only when strictly better, as [`merge`] does), and survivors are applied
/// in the order their cells first appeared, so the table sees new keys in the
/// order a serial walk would have inserted them.
///
/// Survivors are kept per **bucket** of outer columns (column index modulo the
/// bucket count). Columns are separate hash maps, so buckets can be merged in
/// parallel; the order that matters is within one column, and a column lives
/// in one bucket.
struct Pending<K> {
    buckets: Vec<Vec<(NodeIx, K, Cell)>>,
    at: FxHashMap<(NodeIx, K), (u32, u32)>,
}

impl<K> Default for Pending<K> {
    fn default() -> Self {
        Self {
            buckets: Vec::new(),
            at: FxHashMap::default(),
        }
    }
}

impl<K: Copy + Eq + std::hash::Hash> Pending<K> {
    fn reset(&mut self, buckets: usize) {
        self.buckets.resize_with(buckets.max(1), Vec::new);
        self.buckets.truncate(buckets.max(1));
        self.buckets.iter_mut().for_each(Vec::clear);
        self.at.clear();
    }

    /// Keep `cand` for cell `(outer, key)` if it is the first there or
    /// strictly better than the one kept.
    fn offer(&mut self, outer: Node, key: K, cand: Cell, lambda: f64) {
        let outer = NodeIx::from(outer);
        match self.at.entry((outer, key)) {
            Entry::Occupied(at) => {
                let (b, i) = *at.get();
                let kept = &mut self.buckets[b as usize][i as usize].2;
                if brd(cand.mfe, cand.cai, lambda) < brd(kept.mfe, kept.cai, lambda) {
                    *kept = cand;
                }
            }
            Entry::Vacant(at) => {
                let b = outer.0 as usize % self.buckets.len();
                at.insert((b as u32, self.buckets[b].len() as u32));
                self.buckets[b].push((outer, key, cand));
            }
        }
    }
}

impl<'a> Folder<'a> {
    /// Allocate the empty CKY tables for one product DFA at a fixed λ under the
    /// beam-pruning policy `beam` and the region-pair `region` constraint (pass
    /// [`PairConstraint::inert`] for a fold with no class table).
    fn new(
        product: &'a ProductDfa<'a>,
        params: &'a Params,
        lambda: f64,
        beam: Beam,
        region: PairConstraint,
        must: MustPairConstraint,
        threads: usize,
    ) -> Self {
        Self::with_spans(product, params, lambda, beam, region, must, threads, None)
    }

    /// [`Self::new`] over span memos an earlier fold of the same product
    /// left ([`SpanMemo`]), or fresh ones when `spans` is `None`.
    #[allow(clippy::too_many_arguments)]
    fn with_spans(
        product: &'a ProductDfa<'a>,
        params: &'a Params,
        lambda: f64,
        beam: Beam,
        region: PairConstraint,
        must: MustPairConstraint,
        threads: usize,
        spans: Option<SpanMemo>,
    ) -> Self {
        let dfa = product.dfa();
        let n_nodes = dfa.slots();
        let product_nodes = product.num_product_nodes();
        let spans = spans.unwrap_or_else(|| SpanMemo {
            fills: SpanTable::new(product_nodes, dfa.layout(), n_nodes),
            corners: SpanTable::new(product_nodes, dfa.layout(), n_nodes),
        });
        Folder {
            product,
            dfa,
            ac: product.lattice_ac(),
            params,
            lambda,
            region,
            must,
            beam,
            threads: threads.max(1),
            end_cai: end_cai_table(dfa, beam),
            p: vec![FxHashMap::default(); n_nodes],
            multi: vec![FxHashMap::default(); n_nodes],
            m: vec![FxHashMap::default(); n_nodes],
            m2: vec![FxHashMap::default(); n_nodes],
            m2r: vec![FxHashMap::default(); n_nodes],
            lefts: FxHashMap::default(),
            cut_marks: CutMarks::default(),
            far_cuts: WrapCuts::default(),
            rights: Rights::default(),
            spare: Mutex::default(),
            bound: None,
            pruned: AtomicUsize::new(0),
            cut: AtomicUsize::new(0),
            aware_pruned: AtomicUsize::new(0),
            aware_cut: AtomicUsize::new(0),
            c: vec![FxHashMap::default(); n_nodes],
            fills: spans.fills,
            corners: spans.corners,
            free_run: if separate_free_loops() || diagonal_free_loops() {
                free_runs(product)
            } else {
                Vec::new()
            },
            separated: AtomicUsize::new(0),
            lz: false,
            free_back: Vec::new(),
            skew: {
                let n = &params.ninio;
                ((n.max_ninio + n.ninio_37 - 1) / n.ninio_37) as usize
            },
            diag: DiagIndex::default(),
            pulled: AtomicUsize::new(0),
            clock: PhaseClock::default(),
        }
    }

    /// Fill every span slot a worker can address — those within the
    /// single-branch loop window, see [`Self::prefill_from`] — before the fold
    /// starts, when a development build asks (`Setting::SpanEager`), or
    /// unconditionally when
    /// `self.threads > 1`.
    ///
    /// The multi-threaded phases read the span memo through [`Self::fill_span_ro`]
    /// / [`Self::corner_span_ro`], which take `&self` so many workers can share
    /// it — and which therefore cannot fill a slot on a miss the way
    /// [`Self::fill_span`] does. Forcing the eager fill first (still
    /// single-threaded, before any worker spawns) is what lets those reads be
    /// plain shared borrows with no lock: every slot a worker could ask for is
    /// already stored by the time one exists. An eager fill at
    /// `threads = 1` is unaffected — that path never calls the `_ro` accessors.
    fn prefill_spans(&mut self) {
        if !eager_spans() && self.threads <= 1 {
            return;
        }
        // The topology is borrowed from the referents rather than from `self`,
        // so the shared references are copied out before the loop and the
        // `&mut self` calls inside it borrow nothing the iteration holds.
        let (dfa, product) = (self.dfa, self.product);
        for pos in 0..=dfa.seq_len {
            for &from in dfa.nodes_at(pos) {
                for &q in product.reachable_states(from) {
                    self.prefill_from(from, q);
                }
            }
        }
    }

    /// Fill every span leaving the product node `(from, q)` that a worker can
    /// address, **including** the zero-length `to == from` span:
    /// `close_bulge`'s empty-far-flank case addresses one (a real call with
    /// `from.pos == to.pos`, sometimes the same node), so a prefill starting
    /// past `from.pos` would leave exactly the slots [`Self::fill_span_ro`] /
    /// [`Self::corner_span_ro`] need most reliably unfilled.
    ///
    /// The workers only run the single-branch closures, whose flanks are at
    /// most `SINGLE_MAX_LEN` long, so nothing past that window is filled: a
    /// prefill to the end of the molecule would cost a cubic number of span
    /// steps for slots no worker reads. Longer spans (hairpin interiors) are
    /// read only on the serial path, which fills them on demand.
    fn prefill_from(&mut self, from: Node, q: u32) {
        let dfa = self.dfa;
        let last = dfa.seq_len.min(from.pos + SINGLE_MAX_LEN + 1);
        for to_pos in from.pos..=last {
            for &to in dfa.nodes_at(to_pos) {
                self.fill_span(from, q, to);
                self.corner_span(from, q, to);
            }
        }
    }

    /// How much of each span table the fold actually addressed, beside the
    /// meter's counts and under the same gate.
    ///
    /// The dense bound is what an eager fill would pay for and the filled count
    /// is what a fill-on-first-touch one does; printing both is what lets the
    /// gap between them be read off a run rather than argued.
    fn report_span_tables(&self) {
        if !span_meter::counting() {
            return;
        }
        let slots = self.fills.slots.len();
        eprintln!(
            "span tables (dense bound {slots} slots = {} product nodes x {} nodes):",
            self.product.num_product_nodes(),
            self.fills.stride,
        );
        for (name, filled, payload) in [
            ("span_fill", self.fills.filled, self.fills.payload.len()),
            (
                "span_corners",
                self.corners.filled,
                self.corners.payload.len(),
            ),
        ] {
            let occupancy = if slots > 0 {
                filled as f64 / slots as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  {name:<16} filled {filled:>9} of {slots:>9} = {occupancy:>6.2}%  \
values {payload:>10}"
            );
        }
    }

    // --- span primitives, memoized ----------------------------------------

    /// The memoized [`span_fill`] of `[from, to]` entered at `q_from`, as a
    /// `(start, len)` handle into [`Self::fills`] read back with
    /// [`SpanTable::at`].
    ///
    /// A handle rather than a slice because the callers write into the CKY
    /// tables while they iterate, which a live borrow of `self` would forbid.
    /// The payload is append-only, so a handle taken here survives any fill a
    /// caller triggers before it finishes reading.
    fn fill_span(&mut self, from: Node, q_from: u32, to: Node) -> (usize, usize) {
        span_meter::ask(span_meter::FILL);
        let Some(origin) = self.product.product_id(from, q_from) else {
            // No dense address, so no slot: an origin the product's forward
            // pass never reached. The value is still the primitive's, computed
            // and appended so the caller reads it the same way.
            let values = span_fill(self.dfa, self.ac, from, q_from, to);
            return self.fills.append(values);
        };
        let slot = self.fills.slot(origin, to);
        if let Some(handle) = self.fills.stored(slot) {
            return handle;
        }
        let values = span_fill(self.dfa, self.ac, from, q_from, to);
        self.fills.store(slot, values)
    }

    /// The memoized [`span_corners`] of `[from, to]` entered at `q_from`, as a
    /// `(start, len)` handle into [`Self::corners`]. See [`Self::fill_span`]
    /// for why it is a handle.
    fn corner_span(&mut self, from: Node, q_from: u32, to: Node) -> (usize, usize) {
        span_meter::ask(span_meter::CORNERS);
        let Some(origin) = self.product.product_id(from, q_from) else {
            let values = span_corners(self.dfa, self.ac, from, q_from, to);
            return self.corners.append(values);
        };
        let slot = self.corners.slot(origin, to);
        if let Some(handle) = self.corners.stored(slot) {
            return handle;
        }
        if to.pos <= from.pos {
            let values = span_corners(self.dfa, self.ac, from, q_from, to);
            return self.corners.store(slot, values);
        }
        // Fill the whole row out to a horizon in one sweep: at least the
        // loop-length window the single-branch closures read, and twice the
        // span asked for, so a hairpin walking its closing base outward pays
        // for a doubling series of rows rather than one strand per position.
        let reach = (to.pos - from.pos).max(SINGLE_MAX_LEN + 1) * 2;
        for (end, values) in span_corners_row(self.dfa, self.ac, from, q_from, from.pos + reach) {
            let at = self.corners.slot(origin, end);
            if self.corners.stored(at).is_none() {
                self.corners.store(at, values);
            }
        }
        self.corners
            .stored(slot)
            .expect("the row covers the span it was filled for")
    }

    /// The read-only twin of [`Self::fill_span`], for the multi-threaded
    /// phases: many workers hold `&self` at once, so a miss cannot be filled
    /// here the way it can under exclusive access. [`Self::prefill_spans`] is
    /// forced eager whenever `self.threads > 1`, which is the invariant this
    /// relies on — every `(from, q_from)` the fold ever threads is a product
    /// node the forward pass reached (checked by comparing the filled count
    /// against the distinct demanded-key count), so
    /// both `expect`s below hold by construction rather than by luck.
    fn fill_span_ro(&self, from: Node, q_from: u32, to: Node) -> (usize, usize) {
        let origin = self
            .product
            .product_id(from, q_from)
            .expect("threads > 1: every threaded (from, q) is a reachable product node");
        let slot = self.fills.slot(origin, to);
        self.fills
            .stored(slot)
            .expect("threads > 1: eager prefill must have filled every reachable slot")
    }

    /// The read-only twin of [`Self::corner_span`]; see [`Self::fill_span_ro`].
    fn corner_span_ro(&self, from: Node, q_from: u32, to: Node) -> (usize, usize) {
        let origin = self
            .product
            .product_id(from, q_from)
            .expect("threads > 1: every threaded (from, q) is a reachable product node");
        let slot = self.corners.slot(origin, to);
        self.corners
            .stored(slot)
            .expect("threads > 1: eager prefill must have filled every reachable slot")
    }

    /// Apply `writes` in chunk order and keep the buffers for the next
    /// fan-out: a chunk's buffers grow to its column's size, and freeing and
    /// regrowing them for every chunk of every column is what a multi-threaded
    /// fold would otherwise spend its time on.
    ///
    /// With a pool, the `P` survivors are merged one column bucket per worker
    /// (see [`Pending`]): on a long molecule there are a hundred million of
    /// them, and merging them on one thread was a third of the fold's wall
    /// clock at eight threads. Each column still sees its candidates in chunk
    /// order, so the result is the serial one.
    fn apply_all(&mut self, writes: Vec<Writes>, pool: Option<&Pool>) {
        let lambda = self.lambda;
        let buckets = writes.first().map_or(1, |w| w.p.buckets.len());
        match pool {
            Some(pool) if buckets > 1 => {
                // Each worker owns the columns of one bucket and merges that
                // bucket of every chunk, in chunk order.
                let mut owned: Vec<Vec<&mut FxHashMap<PKey, Cell>>> =
                    (0..buckets).map(|_| Vec::new()).collect();
                for (i, column) in self.p.iter_mut().enumerate() {
                    owned[i % buckets].push(column);
                }
                let owned: Vec<Mutex<Vec<&mut FxHashMap<PKey, Cell>>>> =
                    owned.into_iter().map(Mutex::new).collect();
                pool.run(buckets, &|b| {
                    let mut columns = owned[b].lock().unwrap();
                    for w in &writes {
                        for &(outer, key, cand) in &w.p.buckets[b] {
                            let outer = outer.node();
                            merge(columns[self.dfa.ix(outer) / buckets], key, cand, lambda);
                        }
                    }
                });
            }
            _ => {
                for w in &writes {
                    for bucket in &w.p.buckets {
                        for &(outer, key, cand) in bucket {
                            let outer = outer.node();
                            merge(&mut self.p[self.dfa.ix(outer)], key, cand, lambda);
                        }
                    }
                }
            }
        }
        for w in &writes {
            for bucket in &w.multi.buckets {
                for &(outer, key, cand) in bucket {
                    let outer = outer.node();
                    self.put_multi(outer, key, cand);
                }
            }
            for bucket in &w.m2.buckets {
                for &(outer, key, cand) in bucket {
                    let outer = outer.node();
                    self.put_m2(outer, key, cand);
                }
            }
        }
        let mut writes = writes;
        for w in &mut writes {
            self.flush_cuts(&mut w.wrap_cuts);
        }
        self.spare.get_mut().unwrap().extend(writes);
    }

    /// Run `body` over `items`, returning the buffered [`Writes`] the caller
    /// folds back into `self`'s tables (see [`Writes`] for why buffering and
    /// folding this way is byte-identical to a serial walk, for any thread
    /// count). `self.threads <= 1` or fewer than [`PARALLEL_MIN`] items runs
    /// `body` serially into one buffer; otherwise `items` is cut into
    /// contiguous chunks, which the workers claim one at a time, each walking
    /// its chunk in order into that chunk's own buffer.
    fn fan_out<T: Copy + Send + Sync>(
        &self,
        pool: Option<&Pool>,
        items: &[T],
        body: impl Fn(&Self, &mut Writes, T) + Sync,
    ) -> Vec<Writes> {
        let pool = match pool {
            Some(pool) if items.len() >= PARALLEL_MIN => pool,
            _ => {
                let mut w = self.spare.lock().unwrap().pop().unwrap_or_default();
                w.reset(self.threads);
                for &item in items {
                    body(self, &mut w, item);
                }
                return vec![w];
            }
        };
        // Small chunks claimed from a shared counter keep the workers busy when
        // the items differ in cost (an inner pair near the 5′ end has far fewer
        // openings than one in the middle). Each chunk's buffer is tagged with
        // its index and the buffers are returned in index order, so the merge
        // order is the serial walk's whatever the claiming order was.
        let n_workers = self.threads.min(items.len()).max(1);
        let chunk = items.len().div_ceil(n_workers * CHUNKS_PER_WORKER).max(1);
        let n_chunks = items.len().div_ceil(chunk);
        let next = AtomicUsize::new(0);
        let parts: Mutex<Vec<(usize, Writes)>> = Mutex::new(Vec::with_capacity(n_chunks));
        pool.run(n_workers, &|_| {
            let mut done = Vec::new();
            loop {
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c >= n_chunks {
                    break;
                }
                let mut w = self.spare.lock().unwrap().pop().unwrap_or_default();
                w.reset(self.threads);
                let end = ((c + 1) * chunk).min(items.len());
                for &item in &items[c * chunk..end] {
                    body(self, &mut w, item);
                }
                done.push((c, w));
            }
            parts.lock().unwrap().extend(done);
        });
        let mut parts = parts.into_inner().unwrap();
        parts.sort_unstable_by_key(|&(c, _)| c);
        parts.into_iter().map(|(_, w)| w).collect()
    }

    /// Fill the tables 5′→3′ and return the AC state of the lowest-BRD exterior
    /// parse reaching the terminal, or `None` when none does (infeasible).
    fn run(&mut self) -> Option<u32> {
        let start = self.dfa.start();
        self.c[self.dfa.ix(start)].insert(
            self.ac.start(),
            Cell {
                mfe: Energy::ZERO,
                cai: 0.0,
            },
        );
        self.clock = PhaseClock::from_env();
        span_meter::reset();
        self.prefill_spans();
        if diagonal_free_loops() {
            self.enable_diagonal();
        }
        let mut clock = std::mem::take(&mut self.clock);
        clock.time(0, || self.seed_hairpins());
        // One pool for the whole fold: the fan-outs run twice per position, and
        // starting threads for each would cost more than a short column's work.
        if self.threads > 1 {
            Pool::scoped(self.threads, |pool| {
                self.fold_columns(&mut clock, Some(pool))
            });
        } else {
            self.fold_columns(&mut clock, None);
        }
        clock.report();
        span_meter::report();
        self.report_span_tables();
        self.clock = clock;
        self.best_state_at(self.dfa.end())
    }

    /// The AC state of the lowest-BRD exterior parse reaching `end`, over every
    /// state it can end in, or `None` when none does.
    fn best_state_at(&self, end: Node) -> Option<u32> {
        self.c[self.dfa.ix(end)]
            .iter()
            .min_by(|a, b| {
                brd(a.1.mfe, a.1.cai, self.lambda).total_cmp(&brd(b.1.mfe, b.1.cai, self.lambda))
            })
            .map(|(&q, _)| q)
    }

    /// Fill the tables column by column, 5′→3′, after the hairpin seeds;
    /// `pool` runs the two fan-outs when the fold is multi-threaded.
    fn fold_columns(&mut self, clock: &mut PhaseClock, pool: Option<&Pool>) {
        for pos in 0..=self.dfa.seq_len {
            // Each span column is pruned exactly once, immediately before the
            // step that consumes it — after its last writer, mirroring
            // [`crate::scan`] — so a surviving derivation never points back to a
            // pruned cell. The exterior `C` is never pruned (its spine keeps
            // feasibility intact). All prunes are no-ops at [`Beam::exact`].
            self.prune_multi_col(pos);
            clock.time(1, || self.multi_close(pos));
            clock.time(2, || self.pull_skewed(pos));
            self.prune_p_col(pos);
            clock.time(2, || self.p_push(pos, pool));
            self.prune_m2_col(pos);
            clock.time(3, || self.m2_beam(pos, pool));
            if pos < self.dfa.seq_len {
                self.prune_m_col(pos);
                clock.time(4, || {
                    self.m_extend(pos);
                    self.m2r_extend(pos);
                });
                clock.time(5, || self.c_extend(pos));
            }
        }
    }

    // --- merges -----------------------------------------------------------

    /// Merge `cand` into `P[outer][key]`, keeping the lower-BRD cell, after
    /// consulting the region-pair constraint on the pair's two endpoint positions
    /// (`key.open.node().pos`, `key.close.node().pos`).
    ///
    /// The single choke point every `P`-write funnels through — hairpin seeds,
    /// `multi_close`, and the stack / bulge / internal-loop closures — so no
    /// site can bypass the constraint. A hard-forbidden pair is dropped; a soft
    /// penalty is added to the candidate's folding energy before the merge,
    /// identically to [`crate::tables::Tables::update_p`], so a soft table means
    /// the same thing with a motif set as without one. The penalty is a constant
    /// offset for a given pair and is applied once per candidate derivation, so
    /// it is credited exactly once into the cell's stored value and cannot
    /// reorder its merges.
    fn put_p(&mut self, outer: Node, key: PKey, cand: Cell) {
        if let Some(cand) = self.penalized(key, cand) {
            merge(&mut self.p[self.dfa.ix(outer)], key, cand, self.lambda);
        }
    }

    /// `cand` under the region-pair verdict on its pair's endpoints: as is,
    /// with the soft penalty added to its energy, or `None` when the pair is
    /// forbidden. What [`Self::put_p`] stores, and what a buffered write is
    /// reduced on before it gets there.
    fn penalized(&self, key: PKey, cand: Cell) -> Option<Cell> {
        match self
            .region
            .verdict(key.open.node().pos, key.close.node().pos)
        {
            Verdict::Allow => Some(cand),
            Verdict::Forbid => None,
            Verdict::Penalize(pen) => Some(Cell {
                mfe: cand.mfe + pen,
                ..cand
            }),
        }
    }

    /// The soft penalty [`Self::put_p`] added to the `P` cell keyed on `key`.
    ///
    /// A stored `P` energy is the derivation's energy **plus** this, so the
    /// traceback — which recovers a jumped-over interior by matching the energy
    /// the forward scan recorded — has to take it back off before comparing
    /// against a freshly scored loop. Zero for an allowed pair, and a forbidden
    /// pair has no cell to walk back to.
    fn pair_penalty(&self, key: PKey) -> Energy {
        match self
            .region
            .verdict(key.open.node().pos, key.close.node().pos)
        {
            Verdict::Penalize(pen) => pen,
            Verdict::Allow | Verdict::Forbid => Energy::ZERO,
        }
    }

    /// Merge `cand` into `Multi[outer][key]`.
    fn put_multi(&mut self, outer: Node, key: PKey, cand: Cell) {
        merge(&mut self.multi[self.dfa.ix(outer)], key, cand, self.lambda);
    }

    /// Merge `cand` into `M[outer][key]`.
    fn put_m(&mut self, outer: Node, key: MKey, cand: Cell) {
        merge(&mut self.m[self.dfa.ix(outer)], key, cand, self.lambda);
    }

    /// Merge `cand` into `M2[outer][key]`.
    fn put_m2(&mut self, outer: Node, key: MKey, cand: Cell) {
        merge(&mut self.m2[self.dfa.ix(outer)], key, cand, self.lambda);
    }

    /// Merge `cand` into `M2R[outer][key]`.
    fn put_m2r(&mut self, outer: Node, key: MKey, cand: Cell) {
        merge(&mut self.m2r[self.dfa.ix(outer)], key, cand, self.lambda);
    }

    /// Merge `cand` into `C[node][q]`.
    fn put_c(&mut self, node: Node, q: u32, cand: Cell) {
        merge(&mut self.c[self.dfa.ix(node)], q, cand, self.lambda);
    }

    // --- beam pruning -----------------------------------------------------

    /// Prune every `Multi[pos]` column before [`multi_close`] consumes it.
    fn prune_multi_col(&mut self, pos: u32) {
        if self.beam.is_exact() {
            return;
        }
        for &node in self.dfa.nodes_at(pos) {
            let (beam, lambda) = (self.beam, self.lambda);
            prune_span_column(
                &mut self.multi[self.dfa.ix(node)],
                beam,
                lambda,
                &self.end_cai,
                &self.c,
                self.dfa.layout(),
                |k| (k.open.node(), k.q_open),
            );
        }
    }

    /// Prune every `P[pos]` column before [`p_push`] consumes it.
    fn prune_p_col(&mut self, pos: u32) {
        if self.beam.is_exact() {
            return;
        }
        for &node in self.dfa.nodes_at(pos) {
            let (beam, lambda) = (self.beam, self.lambda);
            prune_span_column(
                &mut self.p[self.dfa.ix(node)],
                beam,
                lambda,
                &self.end_cai,
                &self.c,
                self.dfa.layout(),
                |k| (k.open.node(), k.q_open),
            );
        }
    }

    /// Prune every `M2[pos]` column before [`m2_beam`] consumes it.
    fn prune_m2_col(&mut self, pos: u32) {
        if self.beam.is_exact() {
            return;
        }
        for &node in self.dfa.nodes_at(pos) {
            let (beam, lambda) = (self.beam, self.lambda);
            prune_span_column(
                &mut self.m2[self.dfa.ix(node)],
                beam,
                lambda,
                &self.end_cai,
                &self.c,
                self.dfa.layout(),
                |k| (k.inner.node(), k.q_inner),
            );
        }
    }

    /// Prune every `M[pos]` column before [`m_extend`] and the later
    /// bifurcations consume it.
    fn prune_m_col(&mut self, pos: u32) {
        if self.beam.is_exact() {
            return;
        }
        for &node in self.dfa.nodes_at(pos) {
            let (beam, lambda) = (self.beam, self.lambda);
            prune_span_column(
                &mut self.m[self.dfa.ix(node)],
                beam,
                lambda,
                &self.end_cai,
                &self.c,
                self.dfa.layout(),
                |k| (k.inner.node(), k.q_inner),
            );
        }
    }

    // --- hairpins ---------------------------------------------------------

    /// Seed every hairpin-closed `P` cell across all opening/closing edges and
    /// AC states, threading `q` through the opening base, the loop interior, and
    /// the closing base.
    fn seed_hairpins(&mut self) {
        for a in 0..self.dfa.seq_len {
            for &an in self.dfa.nodes_at(a) {
                for &q_open in self.product.reachable_states(an) {
                    for &eo in self.dfa.right(an) {
                        self.seed_hairpins_from(an, q_open, eo);
                    }
                }
            }
        }
    }

    /// Seed hairpins for one opening `(an, q_open, eo)` over every closing base
    /// and closing edge, threading `q` through the loop interior.
    fn seed_hairpins_from(&mut self, an: Node, q_open: u32, eo: Edge) {
        let Some(q_ao) = self.ac.step_edge(an, q_open, eo) else {
            return;
        };
        let first_b = eo.to.pos + MIN_HAIRPIN_LOOP;
        // The interior `[eo.to.pos, b)` is unpaired, so the closing base may not
        // reach past the first required position after the loop starts.
        let b_end = self
            .dfa
            .seq_len
            .min(self.must.max_right(eo.to.pos).saturating_add(1));
        for b in first_b..b_end {
            for &bn in self.dfa.nodes_at(b) {
                for &ec in self.dfa.right(bn) {
                    let Some(closing) = canonical(eo.nuc, ec.nuc) else {
                        continue;
                    };
                    if self.pair_out(ec.to, an, bn, eo, ec.nuc) {
                        continue;
                    }
                    let scored = self.score_hairpin(eo.to, q_ao, bn, closing);
                    for (energy, interior_cai, q_close) in scored {
                        let Some(q_outer) = self.ac.step_edge(bn, q_close, ec) else {
                            continue;
                        };
                        let key = PKey::of((an, eo, q_open), (bn, ec, q_outer));
                        let cai = eo.cai + interior_cai + ec.cai;
                        self.put_p(ec.to, key, Cell { mfe: energy, cai });
                    }
                }
            }
        }
    }

    /// Per-exit-state hairpin interior scores for the loop `[from, to]` closed
    /// by `closing`, entered at AC state `q_from`: `(energy, interior_cai,
    /// q_close)` triples, the lowest-BRD per exit state. The special-hairpin
    /// sizes (3/4/6) enumerate whole loop spellings (the whole sequence selects
    /// a table bonus); the rest read the two terminal mismatches. The AC-threaded
    /// analogue of [`crate::interior::score_hairpin`].
    fn score_hairpin(
        &mut self,
        from: Node,
        q_from: u32,
        to: Node,
        closing: BasePair,
    ) -> Vec<(Energy, f64, u32)> {
        let size = (to.pos - from.pos) as usize;
        // The per-exit-state lowest-BRD interior.
        let cell = |mfe, cai| Cell { mfe, cai };
        let mut best: FxHashMap<u32, Cell> = FxHashMap::default();
        if matches!(size, 3 | 4 | 6) {
            for (cai, seq, q_close) in span_spellings(self.dfa, self.ac, from, q_from, to) {
                let energy = hairpin_energy(
                    self.params,
                    &HairpinLoop {
                        closing,
                        loop_seq: &seq,
                    },
                );
                merge(&mut best, q_close, cell(energy, cai), self.lambda);
            }
        } else {
            let (start, len) = self.corner_span(from, q_from, to);
            for i in start..start + len {
                let c = self.corners.at(i);
                let energy = hairpin_energy_scalar(self.params, closing, size, c.first, c.last);
                merge(&mut best, c.q_to, cell(energy, c.cai), self.lambda);
            }
        }
        best.into_iter()
            .map(|(q, cell)| (cell.mfe, cell.cai, q))
            .collect()
    }

    // --- exterior ---------------------------------------------------------

    /// `C[node]` extends by one unpaired nucleotide on the right, threading `q`.
    fn c_extend(&mut self, pos: u32) {
        // The extension declares `pos` unpaired on the exterior loop.
        if self.must.is_must(pos) {
            return;
        }
        for &node in self.dfa.nodes_at(pos) {
            let states: Vec<(u32, Cell)> = self.c[self.dfa.ix(node)]
                .iter()
                .map(|(&q, &cell)| (q, cell))
                .collect();
            for (q, cell) in states {
                for &e in self.dfa.right(node) {
                    let Some(q2) = self.ac.step_edge(node, q, e) else {
                        continue;
                    };
                    self.put_c(
                        e.to,
                        q2,
                        Cell {
                            mfe: cell.mfe,
                            cai: cell.cai + e.cai,
                        },
                    );
                }
            }
        }
    }

    // --- closed-pair consumption -----------------------------------------

    /// Close every in-progress multi-loop `Multi[pos]` into `P[pos]`, paying the
    /// loop-closing cost on the closing pair.
    fn multi_close(&mut self, pos: u32) {
        for &on in self.dfa.nodes_at(pos) {
            let cells: Vec<(PKey, Cell)> = self.multi[self.dfa.ix(on)]
                .iter()
                .map(|(&k, &c)| (k, c))
                .collect();
            for (key, cell) in cells {
                let closing = key.pair();
                let mfe = cell.mfe + self.multi_close_cost(closing);
                self.put_p(on, key, Cell { mfe, cai: cell.cai });
            }
        }
    }

    /// Consume every finalized `P[pos]` cell: extend the exterior (`C = C + P`),
    /// open / bifurcate the multi-branch tables (`M = P`, `M2 = M + P`), and grow
    /// stacks / bulges / internal loops into larger `P` cells.
    ///
    /// `threads > 1` splits the work in two: the exterior extension and
    /// `M = P` stay serial (cheap writes at `on`), and the `M2 = M + P`
    /// bifurcation and the single-branch push fan out via [`Self::fan_out`]
    /// over every closed pair at `pos`. The bifurcation reads only `M` columns
    /// left of `on` and writes `M2` at `on`, which nothing in the fan-out
    /// reads; the push writes `P` only past `pos`. So the workers read nothing
    /// any of them writes.
    fn p_push(&mut self, pos: u32, pool: Option<&Pool>) {
        if self.threads <= 1 {
            for &on in self.dfa.nodes_at(pos) {
                let cells: Vec<(PKey, Cell)> = self.p[self.dfa.ix(on)]
                    .iter()
                    .filter(|&(k, c)| !self.ruled_out(on, k, c))
                    .map(|(&k, &c)| (k, c))
                    .collect();
                for (key, cell) in cells {
                    self.p_to_exterior(on, key, cell);
                    self.p_to_multibranch(on, key, cell);
                    self.p_to_single(on, key, cell);
                    self.file_diagonal(on, key, cell);
                }
            }
            return;
        }
        let cells: Vec<(Node, PKey, Cell)> = self
            .dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&on| {
                self.p[self.dfa.ix(on)]
                    .iter()
                    .map(move |(&k, &c)| (on, k, c))
            })
            .filter(|(on, k, c)| !self.ruled_out(*on, k, c))
            .collect();
        for &(on, key, cell) in &cells {
            self.p_to_exterior(on, key, cell);
            self.p_to_m(on, key, cell);
            self.file_diagonal(on, key, cell);
        }
        if cells.is_empty() {
            return;
        }
        let writes = self.fan_out(pool, &cells, |f, w, (on, key, cell)| {
            f.bifurcate_ro(w, on, key, cell);
            span_meter::in_single(|| f.enumerate_outer_pairs_ro(w, (on, key, cell), None));
        });
        self.apply_all(writes, pool);
    }

    /// Whether the outside bound rules `P[on][key]` out: no design through it
    /// can score `bound.ceiling` or better, so nothing built on it is worth
    /// building. Never true without a bound, or for a pair the bound does not
    /// cover.
    fn ruled_out(&self, on: Node, key: &PKey, cell: &Cell) -> bool {
        let Some(by) = self.floor_beyond(on, key, *cell) else {
            return false;
        };
        self.pruned.fetch_add(1, Ordering::Relaxed);
        if by == Floor::Aware {
            self.aware_pruned.fetch_add(1, Ordering::Relaxed);
        }
        true
    }

    /// What the bound's floors have ruled out so far.
    fn bound_counts(&self) -> BoundCounts {
        BoundCounts {
            pruned: self.pruned.load(Ordering::Relaxed),
            cut: self.cut.load(Ordering::Relaxed),
            aware_pruned: self.aware_pruned.load(Ordering::Relaxed),
            aware_cut: self.aware_cut.load(Ordering::Relaxed),
        }
    }

    /// Count a cut by the pair floor `by`.
    fn count_cut(&self, by: Floor) {
        self.cut.fetch_add(1, Ordering::Relaxed);
        if by == Floor::Aware {
            self.aware_cut.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether the bound's pair floor rules out every cell closing the pair
    /// `open`·`close` (opened by edge `ep`, closed on nucleotide `nuc_close`)
    /// into outer node `outer`, before any of them is scored. Counted as a cut.
    fn pair_out(
        &self,
        outer: Node,
        open: Node,
        close: Node,
        ep: Edge,
        nuc_close: Nucleotide,
    ) -> bool {
        let by = self.pair_floor_out(outer, open, close, ep, nuc_close);
        if let Some(by) = by {
            self.count_cut(by);
        }
        by.is_some()
    }

    /// [`Self::pair_out`]'s test, without counting: which floor rules the
    /// pair out, if any.
    fn pair_floor_out(
        &self,
        outer: Node,
        open: Node,
        close: Node,
        ep: Edge,
        nuc_close: Nucleotide,
    ) -> Option<Floor> {
        let bound = self.bound.as_ref()?;
        let pair = crate::state::PairKey {
            open,
            close,
            opening: ep.opening,
            nuc_close,
        };
        bound.rules_out_pair(self.dfa.ix(outer), &pair)
    }

    /// [`Self::pair_out`] through `memo`, for the multi-loop wraps, where
    /// every core at a frontier asks about the same openings. A cut is
    /// tallied on every call, so the count does not depend on how the calls
    /// were shared out among memos.
    fn pair_out_cached(
        &self,
        memo: &mut WrapCuts,
        outer: Node,
        open: Node,
        close: Node,
        ep: Edge,
        nuc_close: Nucleotide,
    ) -> bool {
        if self.bound.is_none() {
            return false;
        }
        let slot = memo.slot(self.dfa.layout(), close, open, ep, nuc_close);
        if *slot == 0 {
            *slot = match self.pair_floor_out(outer, open, close, ep, nuc_close) {
                None => 1,
                Some(Floor::Plain) => 2,
                Some(Floor::Aware) => 3,
            };
        }
        let verdict = *slot;
        memo.cuts += usize::from(verdict >= 2);
        memo.aware_cuts += usize::from(verdict == 3);
        verdict >= 2
    }

    /// Add `memo`'s tallied cuts to the fold's count.
    fn flush_cuts(&self, memo: &mut WrapCuts) {
        self.cut
            .fetch_add(std::mem::take(&mut memo.cuts), Ordering::Relaxed);
        self.aware_cut
            .fetch_add(std::mem::take(&mut memo.aware_cuts), Ordering::Relaxed);
    }

    /// Whether the bound carries a pair floor at all.
    fn has_pair_floor(&self) -> bool {
        self.bound
            .as_ref()
            .is_some_and(|b| b.through || b.aware.is_some())
    }

    /// Mark which of `rights`' sides close an outer pair, opened at `pn` by
    /// `ep`, that the bound's pair floor rules out; `false` when every side
    /// is ruled out, so the opening need not be swept at all. Leaves the
    /// marks empty (nothing cut) without a pair floor.
    fn mark_cut_sides(&self, pn: Node, ep: Edge, rights: &mut Rights) -> bool {
        let Rights { sides, cut, .. } = rights;
        cut.clear();
        if !self.has_pair_floor() {
            return true;
        }
        let mut any = false;
        for side in sides.iter() {
            let out = canonical(ep.nuc, side.eq.nuc).is_some()
                && self.pair_out(side.eq.to, pn, side.qn, ep, side.eq.nuc);
            cut.push(out);
            any |= !out;
        }
        any
    }

    /// [`Self::mark_cut_sides`] through [`Self::cut_marks`]: the marks are
    /// computed once per (column node, opening) and copied for every later
    /// inner pair of the column, the cuts they stand for counted each time so
    /// the count matches the threaded path's, which marks afresh.
    fn marked_cut_sides(&mut self, on: Node, pn: Node, ep: Edge, rights: &mut Rights) -> bool {
        if !self.has_pair_floor() {
            rights.cut.clear();
            return true;
        }
        if self.cut_marks.column != Some(on) {
            self.cut_marks.column = Some(on);
            self.cut_marks.by_opening.clear();
        }
        let key = (NodeIx::from(pn), ep.nuc as u8);
        if let Some((marks, cuts, aware)) = self.cut_marks.by_opening.get(&key) {
            rights.cut.clone_from(marks);
            self.cut.fetch_add(*cuts, Ordering::Relaxed);
            self.aware_cut.fetch_add(*aware, Ordering::Relaxed);
            return marks.iter().any(|&c| !c);
        }
        let aware_before = self.aware_cut.load(Ordering::Relaxed);
        let any = self.mark_cut_sides(pn, ep, rights);
        let cuts = rights.cut.iter().filter(|&&c| c).count();
        let aware = self.aware_cut.load(Ordering::Relaxed) - aware_before;
        self.cut_marks
            .by_opening
            .insert(key, (rights.cut.clone(), cuts, aware));
        any
    }

    /// The test [`Self::ruled_out`] makes, without counting: whether the bound
    /// kept `P[on][key]` from being pushed.
    fn beyond_bound(&self, on: Node, key: &PKey, cell: Cell) -> bool {
        self.floor_beyond(on, key, cell).is_some()
    }

    /// Which of the bound's cell floors puts `P[on][key]` above the ceiling:
    /// the unconstrained outside, a zero chord or the motif-free chord, or
    /// else the aware chord.
    fn floor_beyond(&self, on: Node, key: &PKey, cell: Cell) -> Option<Floor> {
        let bound = self.bound.as_ref()?;
        let beyond =
            |floor: f64| brd(cell.mfe, cell.cai, self.lambda) + floor > bound.ceiling + BOUND_SLACK;
        if bound
            .floor(self.dfa.ix(on), &key.pair_key())
            .is_some_and(beyond)
            || bound
                .zero
                .iter()
                .any(|z| z.floor(on, key).is_some_and(beyond))
            || bound.motif_free.is_some_and(|c| {
                // In exact tables a pair no motif-free design passes through
                // has no entry at any λ.
                match (
                    c.lo[self.dfa.ix(on)].get(key),
                    c.hi[self.dfa.ix(on)].get(key),
                ) {
                    (Some(&a), Some(&b)) => beyond((1.0 - c.weight) * a + c.weight * b),
                    _ => c.complete,
                }
            })
        {
            return Some(Floor::Plain);
        }
        bound
            .aware
            .and_then(|a| a.cell_floor(on, key, cell))
            .is_some_and(beyond)
            .then_some(Floor::Aware)
    }

    /// `C = C + P`: absorb a closed pair into the exterior prefix.
    fn p_to_exterior(&mut self, on: Node, key: PKey, cell: Cell) {
        let Some(&prefix) = self.c[self.dfa.ix(key.open.node())].get(&key.q_open) else {
            return;
        };
        let mfe = prefix.mfe + cell.mfe + self.terminal_au(key.pair());
        self.put_c(
            on,
            key.q_outer,
            Cell {
                mfe,
                cai: prefix.cai + cell.cai,
            },
        );
    }

    /// `M = P` (a single multi-branch) and `M2 = M + P` (bifurcation at the
    /// shared split node `key.open.node()`, carrying the single AC state `key.q_open`).
    fn p_to_multibranch(&mut self, on: Node, key: PKey, cell: Cell) {
        self.p_to_m(on, key, cell);
        let stem = self.m_stem_cost(key.pair());
        let lambda = self.lambda;
        let (m, m2) = (&self.m, &mut self.m2);
        bifurcate(
            &m[self.dfa.ix(key.open.node())],
            key,
            cell,
            stem,
            |mk, cand| {
                merge(&mut m2[self.dfa.ix(on)], mk, cand, lambda);
            },
        );
    }

    /// `M = P`: a closed pair as a multi-branch component of one branch.
    fn p_to_m(&mut self, on: Node, key: PKey, cell: Cell) {
        let stem = self.m_stem_cost(key.pair());
        self.put_m(
            on,
            MKey {
                inner: key.open,
                q_inner: key.q_open,
                q_outer: key.q_outer,
            },
            Cell {
                mfe: cell.mfe + stem,
                cai: cell.cai,
            },
        );
    }

    /// The `M2` half of [`Self::p_to_multibranch`] for a threaded push: the
    /// bifurcations of `P[on][key]`, offered to the chunk's buffer. It reads
    /// only `M` columns left of `on`, which no worker writes.
    fn bifurcate_ro(&self, w: &mut Writes, on: Node, key: PKey, cell: Cell) {
        let stem = self.m_stem_cost(key.pair());
        let lambda = self.lambda;
        bifurcate(
            &self.m[self.dfa.ix(key.open.node())],
            key,
            cell,
            stem,
            |mk, cand| {
                w.m2.offer(on, mk, cand, lambda);
            },
        );
    }

    /// Grow `P` (an inner pair) into every outer pair whose unpaired flanks fit
    /// within `SINGLE_MAX_LEN` — the stack, bulge, and internal-loop closures,
    /// threading `q` through both flanks.
    ///
    /// This is the densest span-primitive caller in the folder, so it is where
    /// the sub-phase meter splits the enumeration from the span work; see
    /// [`span_meter`] for what that separation is for.
    fn p_to_single(&mut self, on: Node, key: PKey, cell: Cell) {
        span_meter::in_single(|| self.enumerate_outer_pairs(on, key, cell));
    }

    /// Sweep every outer 5′ opening the flank bound admits, dispatching each to
    /// [`Self::single_from_left`].
    fn enumerate_outer_pairs(&mut self, on: Node, key: PKey, cell: Cell) {
        let mut rights = std::mem::take(&mut self.rights);
        self.right_closings(on, key, &mut rights);
        let inner_open = key.open.node();
        let inner_pair = key.pair();
        let a = inner_open.pos;
        // The 5′ flank `[p + 1, a)` is unpaired, which bounds how far left the
        // outer opening may sit: n1 = a − p − 1 ≤ SINGLE_MAX_LEN.
        let p_lo = a
            .saturating_sub(SINGLE_MAX_LEN + 1)
            .max(self.must.min_left(a));
        for p in p_lo..a {
            let n1 = a - p - 1;
            for &pn in self.dfa.nodes_at(p) {
                for &ep in self.dfa.right(pn) {
                    if !self.marked_cut_sides(on, pn, ep, &mut rights) {
                        continue;
                    }
                    for &q_open in self.product.reachable_states(pn) {
                        let Some(q_ae) = self.ac.step_edge(pn, q_open, ep) else {
                            continue;
                        };
                        self.single_from_left(
                            on, key, cell, inner_pair, pn, ep, q_open, q_ae, n1, &rights,
                        );
                    }
                }
            }
        }
        self.rights = rights;
    }

    /// Close every outer 3′ pair against one fixed outer 5′ opening
    /// `(pn, ep, q_open)`, threading both flanks and dispatching on the flank
    /// shape (stack / bulge / internal).
    #[allow(clippy::too_many_arguments)]
    fn single_from_left(
        &mut self,
        on: Node,
        key: PKey,
        cell: Cell,
        inner_pair: BasePair,
        pn: Node,
        ep: Edge,
        q_open: u32,
        q_ae: u32,
        n1: u32,
        rights: &Rights,
    ) {
        // The 3′ flank `[on.pos, q)` is unpaired, which bounds how far right
        // the outer closing base may sit.
        let lefts = self.left_corners(ep.to, q_ae, key.open.node(), key.q_open, n1);
        // Past an empty right flank every closure is an internal loop, which
        // needs a left corner reaching the inner pair's entry state.
        let q_hi = if n1 > 0 && lefts.is_empty() {
            on.pos
        } else {
            (on.pos + SINGLE_MAX_LEN - n1)
                .min(self.dfa.seq_len - 1)
                .min(self.must.max_right(on.pos))
        };
        for (i, side) in rights.sides.iter().enumerate() {
            if on.pos + side.n2 > q_hi {
                break;
            }
            if rights.cut.get(i).copied().unwrap_or(false)
                || self.pulled_shape(ep.to.pos, n1 as usize, on.pos, side.n2 as usize)
            {
                continue;
            }
            let Some(outer_pair) = canonical(ep.nuc, side.eq.nuc) else {
                continue;
            };
            let ctx = SingleCtx {
                on,
                key,
                cell,
                inner_pair,
                outer_pair,
                pn,
                ep,
                q_open,
                q_ae,
                qn: side.qn,
                eq: side.eq,
                n1: n1 as usize,
                n2: side.n2 as usize,
            };
            self.close_single(&ctx, &lefts, rights.corners_of(side));
        }
    }

    /// Score and write the single-branch closure for one fully-fixed outer pair,
    /// dispatching on the flank sizes.
    fn close_single(&mut self, ctx: &SingleCtx, lefts: &[CornerAc], rights: &[(CornerAc, u32)]) {
        match (ctx.n1, ctx.n2) {
            (0, 0) => self.close_stack(ctx),
            (_, _) if ctx.n1 == 0 || ctx.n2 == 0 => self.close_bulge(ctx),
            _ => self.close_internal(ctx, lefts, rights),
        }
    }

    /// Stack closure (both flanks empty): the outer pair sits directly on the
    /// inner pair.
    fn close_stack(&mut self, ctx: &SingleCtx) {
        // Empty left flank: the outer opening base must land on the inner open
        // node at the inner entry state.
        if ctx.ep.to != ctx.key.open.node() || ctx.q_ae != ctx.key.q_open || ctx.qn != ctx.on {
            return;
        }
        let Some(q_outer) = self.ac.step_edge(ctx.qn, ctx.key.q_outer, ctx.eq) else {
            return;
        };
        let energy = stack_energy(
            self.params,
            &StackPair {
                outer: ctx.outer_pair,
                inner: ctx.inner_pair,
            },
        );
        self.write_single(ctx, ctx.cell.cai + ctx.ep.cai + ctx.eq.cai, energy, q_outer);
    }

    /// Bulge closure (exactly one flank empty): the interior nucleotides do not
    /// reach the energy model, so each non-empty flank is filled best-CAI.
    fn close_bulge(&mut self, ctx: &SingleCtx) {
        let energy = bulge_energy(
            self.params,
            &BulgeLoop {
                outer: ctx.outer_pair,
                inner: ctx.inner_pair,
                size: ctx.n1 + ctx.n2,
            },
        );
        if ctx.n1 == 0 {
            // Empty left flank; fill the right flank `[on, qn]`.
            if ctx.ep.to != ctx.key.open.node() || ctx.q_ae != ctx.key.q_open {
                return;
            }
            let (start, len) = self.fill_span(ctx.on, ctx.key.q_outer, ctx.qn);
            for i in start..start + len {
                let (q_qn, right_cai) = self.fills.at(i);
                let Some(q_outer) = self.ac.step_edge(ctx.qn, q_qn, ctx.eq) else {
                    continue;
                };
                let cai = ctx.cell.cai + ctx.ep.cai + right_cai + ctx.eq.cai;
                self.write_single(ctx, cai, energy, q_outer);
            }
        } else {
            // Empty right flank (`qn == on`); fill the left flank
            // `[ep.to, key.open.node()]`, which must reach the inner entry state.
            if ctx.qn != ctx.on {
                return;
            }
            let (start, len) = self.fill_span(ctx.ep.to, ctx.q_ae, ctx.key.open.node());
            let Some((_, left_cai)) = (start..start + len)
                .map(|i| self.fills.at(i))
                .find(|&(q, _)| q == ctx.key.q_open)
            else {
                return;
            };
            let Some(q_outer) = self.ac.step_edge(ctx.qn, ctx.key.q_outer, ctx.eq) else {
                return;
            };
            let cai = ctx.cell.cai + ctx.ep.cai + left_cai + ctx.eq.cai;
            self.write_single(ctx, cai, energy, q_outer);
        }
    }

    /// Internal-loop closure (both flanks non-empty): the four terminal
    /// mismatches reach the energy model, chosen jointly with their CAI over the
    /// AC-threaded corners.
    fn close_internal(&mut self, ctx: &SingleCtx, lefts: &[CornerAc], rights: &[(CornerAc, u32)]) {
        if lefts.is_empty() {
            return;
        }
        let mut best = SmallVec::new();
        self.reduce_internal(ctx, lefts, rights, &mut best);
        for r in &best {
            self.write_single(ctx, r.cai, r.energy, r.q_outer);
        }
    }

    /// Every right closing of the single-branch loops around the inner pair
    /// `(on, key)`, computed once per inner pair rather than once per left
    /// opening: each `(closing position, node, closing edge)` in the order the
    /// closures visit them, and for the internal-loop ones (a non-empty right
    /// flank) the right corners that reach the closing base with the state
    /// the closing edge leaves them in. A corner the closing edge cannot leave
    /// (it would complete a motif) is dropped here once.
    ///
    /// Written into `rights`, cleared first, so one buffer serves every inner
    /// pair: an allocation per inner pair is what the multi-threaded path
    /// would otherwise contend on.
    fn right_closings(&mut self, on: Node, key: PKey, rights: &mut Rights) {
        rights.clear();
        let dfa = self.dfa;
        for q in on.pos..=self.closing_limit(on) {
            let n2 = q - on.pos;
            for &qn in dfa.nodes_at(q) {
                let handle = match n2 {
                    0 if qn != on => continue,
                    0 => 0..0,
                    _ => {
                        let (start, len) = self.corner_span(on, key.q_outer, qn);
                        start..start + len
                    }
                };
                self.push_sides(rights, n2, qn, handle);
            }
        }
    }

    /// The `&self` twin of [`Self::right_closings`].
    fn right_closings_ro(&self, on: Node, key: PKey, rights: &mut Rights) {
        rights.clear();
        for q in on.pos..=self.closing_limit(on) {
            let n2 = q - on.pos;
            for &qn in self.dfa.nodes_at(q) {
                let handle = match n2 {
                    0 if qn != on => continue,
                    0 => 0..0,
                    _ => {
                        let (start, len) = self.corner_span_ro(on, key.q_outer, qn);
                        start..start + len
                    }
                };
                self.push_sides(rights, n2, qn, handle);
            }
        }
    }

    /// The last position a single-branch loop around an inner pair ending at
    /// `on` may close at: `SINGLE_MAX_LEN` past `on`, within the molecule, and
    /// short of the first must-pair position the right flank would cover.
    /// Below `on.pos` when no closing fits.
    fn closing_limit(&self, on: Node) -> u32 {
        (on.pos + SINGLE_MAX_LEN)
            .min(self.dfa.seq_len - 1)
            .min(self.must.max_right(on.pos))
    }

    /// Record one closing node's sides — one per out-edge — with the memoized
    /// right corners in `handle` each edge can close.
    fn push_sides(&self, rights: &mut Rights, n2: u32, qn: Node, handle: std::ops::Range<usize>) {
        for &eq in self.dfa.right(qn) {
            let start = rights.corners.len();
            for i in handle.clone() {
                let rc = self.corners.at(i);
                if let Some(q_outer) = self.ac.step_edge(qn, rc.q_to, eq) {
                    rights.corners.push((rc, q_outer));
                }
            }
            rights.sides.push(RightSide {
                n2,
                qn,
                eq,
                corners: start..rights.corners.len(),
            });
        }
    }

    /// The left corners of an internal loop opened at `from` (state `q_from`)
    /// and closed by the inner pair at `to`: the memoized [`span_corners`]
    /// entries that reach the inner pair's entry state `q_to`, copied out once
    /// per opening rather than filtered again at every closing frontier. Empty
    /// for an empty left flank (`n1 == 0`), which has no corner.
    fn left_corners(
        &mut self,
        from: Node,
        q_from: u32,
        to: Node,
        q_to: u32,
        n1: u32,
    ) -> SmallVec<[CornerAc; 8]> {
        if n1 == 0 {
            return SmallVec::new();
        }
        let (start, len) = self.corner_span(from, q_from, to);
        self.matching_corners(start..start + len, q_to)
    }

    /// The `&self` twin of [`Self::left_corners`].
    fn left_corners_ro(
        &self,
        from: Node,
        q_from: u32,
        to: Node,
        q_to: u32,
        n1: u32,
    ) -> SmallVec<[CornerAc; 8]> {
        if n1 == 0 {
            return SmallVec::new();
        }
        let (start, len) = self.corner_span_ro(from, q_from, to);
        self.matching_corners(start..start + len, q_to)
    }

    /// The corners in `handle` that exit in state `q_to`, in memo order.
    fn matching_corners(
        &self,
        handle: std::ops::Range<usize>,
        q_to: u32,
    ) -> SmallVec<[CornerAc; 8]> {
        handle
            .map(|i| self.corners.at(i))
            .filter(|c| c.q_to == q_to)
            .collect()
    }

    /// The internal-loop candidates of one closure, reduced to the best per
    /// `P` key before any of them reaches the table.
    ///
    /// Every (left corner, right corner) pair writes the key fixed by the
    /// right corner's exit state, so all left corners — and every right corner
    /// exiting to the same state — compete for one cell. Merging each into the
    /// hash table separately is most of what a closure costs; this keeps the
    /// winner per key locally and writes it once. The comparison is the one
    /// [`Self::put_p`]'s merge would make (soft region penalty included, which
    /// is constant for the closure's endpoints), and ties go to the candidate
    /// the left-major walk would have reached first, so the surviving cell is
    /// the one the per-candidate merges would have kept.
    fn reduce_internal(
        &self,
        ctx: &SingleCtx,
        lefts: &[CornerAc],
        rights: &[(CornerAc, u32)],
        out: &mut SmallVec<[Reduced; 4]>,
    ) {
        let pen = match self.region.verdict(ctx.pn.pos, ctx.qn.pos) {
            Verdict::Allow => Energy::ZERO,
            Verdict::Forbid => return,
            Verdict::Penalize(pen) => pen,
        };
        if let Some(cand) = self.separated_internal(ctx, rights, pen) {
            out.push(cand);
            return;
        }
        for &(rc, q_outer) in rights {
            let mut best: Option<Reduced> = None;
            for (li, lc) in lefts.iter().enumerate() {
                let energy = internal_energy_scalar(
                    self.params,
                    &InternalLoopScalar {
                        outer: ctx.outer_pair,
                        inner: ctx.inner_pair,
                        boundary_left5p: lc.first,
                        boundary_left3p: lc.last,
                        boundary_right5p: rc.first,
                        boundary_right3p: rc.last,
                        n1: ctx.n1,
                        n2: ctx.n2,
                    },
                );
                let cai = ctx.cell.cai + ctx.ep.cai + lc.cai + rc.cai + ctx.eq.cai;
                let score = brd(ctx.cell.mfe + energy + pen, cai, self.lambda);
                if best.is_none_or(|b| score < b.score) {
                    best = Some(Reduced {
                        q_outer,
                        right_q: rc.q_to,
                        energy,
                        cai,
                        score,
                        left: li,
                    });
                }
            }
            let Some(cand) = best else {
                continue;
            };
            match out.iter_mut().find(|r| r.q_outer == q_outer) {
                Some(r) => {
                    if cand.score < r.score || (cand.score == r.score && cand.left < r.left) {
                        *r = cand;
                    }
                }
                None => out.push(cand),
            }
        }
    }

    /// [`Self::reduce_internal`] for an internal loop whose two flanks are at
    /// least [`SEPARABLE_FLANK`] long and lie in free runs ([`free_runs`]),
    /// or `None` for any other.
    ///
    /// On such flanks every spelling the plain lattice has is motif-free, the
    /// states are fixed, and each flank's CAI splits into a half at its outer
    /// end and a half at its inner end. The loop's energy splits the same way:
    /// the outer mismatch reads the 5′ flank's first and the 3′ flank's last
    /// nucleotide, the inner mismatch the other two, and the size and ninio
    /// terms read no nucleotide. So the best corner pair is the best outer
    /// corner plus the best inner corner, 2 × 16 terms in place of 16 × 16.
    fn separated_internal(
        &self,
        ctx: &SingleCtx,
        rights: &[(CornerAc, u32)],
        pen: Energy,
    ) -> Option<Reduced> {
        if self.free_run.is_empty()
            || ctx.n1 < SEPARABLE_FLANK
            || ctx.n2 < SEPARABLE_FLANK
            || (self.free_run[ctx.ep.to.pos as usize] as usize) < ctx.n1
            || (self.free_run[ctx.on.pos as usize] as usize) < ctx.n2
        {
            return None;
        }
        // One state at the flank's end, so every right corner leads to the
        // same outer state.
        let &(rc0, q_outer) = rights.first()?;
        let outer = self.outer_corner(ctx.ep.to, ctx.qn, ctx.outer_pair)?;
        let inner = self.inner_corner(ctx.key.open.node(), ctx.on, ctx.inner_pair)?;
        let (
            Corner {
                a: l5,
                b: r3,
                cai: c_out,
                ..
            },
            Corner {
                a: l3,
                b: r5,
                cai: c_in,
                ..
            },
        ) = (outer, inner);
        let lambda = self.lambda;
        let energy = internal_energy_scalar(
            self.params,
            &InternalLoopScalar {
                outer: ctx.outer_pair,
                inner: ctx.inner_pair,
                boundary_left5p: l5,
                boundary_left3p: l3,
                boundary_right5p: r5,
                boundary_right3p: r3,
                n1: ctx.n1,
                n2: ctx.n2,
            },
        );
        let cai = ctx.cell.cai + ctx.ep.cai + c_out + c_in + ctx.eq.cai;
        self.separated.fetch_add(1, Ordering::Relaxed);
        Some(Reduced {
            q_outer,
            right_q: rc0.q_to,
            energy,
            cai,
            score: brd(ctx.cell.mfe + energy + pen, cai, lambda),
            left: 0,
        })
    }

    /// The best outer corner of an internal loop whose flanks are free: the
    /// 5′ flank's first nucleotide (an edge from `from`, the node after the
    /// opening base) and the 3′ flank's last (an edge into `qn`, the closing
    /// base's node), scored by the outer mismatch and the flanks' outer CAI
    /// halves. The first of equal scores wins, in edge order.
    fn outer_corner(&self, from: Node, qn: Node, outer: BasePair) -> Option<Corner> {
        let (lambda, kk) = (self.lambda, (1.0 - self.lambda) * 100.0);
        let mm = &self.params.mismatch_interior;
        let mut best: Option<Corner> = None;
        for l5 in self.dfa.right(from) {
            let cl = l5.cai + self.dfa.span_head_cai(l5.to).unwrap_or(0.0);
            for r3 in self.dfa.left(qn) {
                let cr = r3.cai + self.dfa.span_tail_cai(r3.to).unwrap_or(0.0);
                let e = mm.get(outer, l5.nuc.to_nuc5(), r3.nuc.to_nuc5());
                let score = lambda * f64::from(e) - kk * (cl + cr);
                if best.is_none_or(|b| score < b.score) {
                    best = Some(Corner {
                        score,
                        a: l5.nuc,
                        b: r3.nuc,
                        cai: cl + cr,
                    });
                }
            }
        }
        best
    }

    /// The best inner corner of an internal loop whose flanks are free: the
    /// 5′ flank's last nucleotide (an edge into `open`, the inner pair's
    /// opening node) and the 3′ flank's first (an edge from `on`, the node
    /// after the inner closing base), scored by the inner mismatch and the
    /// flanks' inner CAI halves. The first of equal scores wins, in edge order.
    fn inner_corner(&self, open: Node, on: Node, inner: BasePair) -> Option<Corner> {
        let (lambda, kk) = (self.lambda, (1.0 - self.lambda) * 100.0);
        let mm = &self.params.mismatch_interior;
        let inner_rev = inner.reversed();
        let mut best: Option<Corner> = None;
        for l3 in self.dfa.left(open) {
            let dl = self.dfa.span_tail_cai(l3.to).unwrap_or(0.0) + l3.cai;
            for r5 in self.dfa.right(on) {
                let dr = r5.cai + self.dfa.span_head_cai(r5.to).unwrap_or(0.0);
                let e = mm.get(inner_rev, r5.nuc.to_nuc5(), l3.nuc.to_nuc5());
                let score = lambda * f64::from(e) - kk * (dl + dr);
                if best.is_none_or(|b| score < b.score) {
                    best = Some(Corner {
                        score,
                        a: l3.nuc,
                        b: r5.nuc,
                        cai: dl + dr,
                    });
                }
            }
        }
        best
    }

    /// Turn on the diagonal pull for this fold, unless a must-pair constraint
    /// is active.
    fn enable_diagonal(&mut self) {
        if self.must.is_active() || self.free_run.is_empty() {
            return;
        }
        self.lz = true;
        self.free_back = free_backs(&self.free_run);
    }

    /// Whether the pull, not the push, scores the internal loop whose 5′
    /// flank of `n1` starts at layer `from` and whose 3′ flank of `n2`
    /// starts at layer `on`: both flanks at least [`SEPARABLE_FLANK`] and free,
    /// and the ninio term saturated. The pull covers exactly these.
    fn pulled_shape(&self, from: u32, n1: usize, on: u32, n2: usize) -> bool {
        self.lz
            && n1 >= SEPARABLE_FLANK
            && n2 >= SEPARABLE_FLANK
            && n1.abs_diff(n2) >= self.skew
            && self.free_run[from as usize] as usize >= n1
            && self.free_run[on as usize] as usize >= n2
    }

    /// File the closed pair `P[on][key]` on its size diagonal when its
    /// boundary layers end and start free runs at least [`SEPARABLE_FLANK`]
    /// long, so that some outer pair can close a pulled loop around it.
    fn file_diagonal(&mut self, on: Node, key: PKey, cell: Cell) {
        if !self.lz {
            return;
        }
        let (open, close) = (key.open.node(), key.close.node());
        if (self.free_back[open.pos as usize] as usize) < SEPARABLE_FLANK
            || (self.free_run[on.pos as usize] as usize) < SEPARABLE_FLANK
        {
            return;
        }
        let Some(corner) = self.inner_corner(open, on, key.pair()) else {
            return;
        };
        let h = brd(cell.mfe, cell.cai, self.lambda) + corner.score;
        self.diag.insert(
            close.pos,
            DiagEntry {
                k: open.pos,
                h,
                key,
                cell,
                corner,
            },
        );
    }

    /// Write every skewed internal loop on free flanks whose outer pair
    /// closes into column `pos`, from the size diagonals (the Lyngsø–Zuker
    /// reduction, restricted to free flanks).
    ///
    /// With the ninio term saturated, a loop of size `s` scores
    /// `λ · (size(s) + MAX_NINIO)` plus its outer corner plus its inner pair's
    /// value with its inner corner, and on free flanks the last depends on the
    /// inner pair alone. So for each outer pair and each size the best inner
    /// pair is a minimum over one diagonal, within the openings that keep both
    /// flanks free and the asymmetry saturated. Each outer pair is written
    /// once, with its best over every size; its corners are those
    /// [`Self::separated_internal`] would choose, so the traceback rederives
    /// it.
    fn pull_skewed(&mut self, pos: u32) {
        if !self.lz || pos == 0 {
            return;
        }
        let q = pos - 1;
        let back_q = self.free_back[q as usize] as usize;
        if back_q < SEPARABLE_FLANK {
            return;
        }
        let side_max = SINGLE_MAX_LEN as usize - SEPARABLE_FLANK;
        // The outer openings some filed pair can sit inside: a pair closing at
        // `c` with `n2 = q − 1 − c` and opening at `k` puts the outer opening at
        // `k − 1 − n1`.
        let mut opens: Vec<u32> = Vec::new();
        let c_lo = q.saturating_sub(1 + side_max as u32);
        for c in c_lo..q.saturating_sub(SEPARABLE_FLANK as u32) {
            opens.extend_from_slice(self.diag.opens_closing_at(c));
        }
        if opens.is_empty() {
            return;
        }
        let mut ps: Vec<u32> = opens
            .iter()
            .flat_map(|&k| {
                let lo = k.saturating_sub(1 + side_max as u32);
                let hi = k.saturating_sub(1 + SEPARABLE_FLANK as u32);
                lo..=hi
            })
            .filter(|&p| self.free_run[p as usize + 1] as usize >= SEPARABLE_FLANK)
            .collect();
        ps.sort_unstable();
        ps.dedup();
        let lambda = self.lambda;
        let max_ninio = Energy(self.params.ninio.max_ninio);
        let (dfa, product) = (self.dfa, self.product);
        for &qn in dfa.nodes_at(q) {
            let q_qn = product.reachable_states(qn)[0];
            for &eq in dfa.right(qn) {
                let Some(q_outer) = self.ac.step_edge(qn, q_qn, eq) else {
                    continue;
                };
                for &p in &ps {
                    let run_p = self.free_run[p as usize + 1] as usize;
                    let span = (q - p - 2) as usize;
                    for &pn in dfa.nodes_at(p) {
                        for &ep in dfa.right(pn) {
                            self.pull_outer(
                                (pn, ep, qn, eq, q_outer),
                                (run_p, back_q, span),
                                (lambda, max_ninio),
                            );
                        }
                    }
                }
            }
        }
    }

    /// [`Self::pull_skewed`] for one outer pair: its best pulled loop over
    /// every size, written under every opening state that reaches the pair.
    fn pull_outer(
        &mut self,
        (pn, ep, qn, eq, q_outer): (Node, Edge, Node, Edge, u32),
        (run_p, back_q, span): (usize, usize, usize),
        (lambda, max_ninio): (f64, Energy),
    ) {
        let Some(outer_pair) = canonical(ep.nuc, eq.nuc) else {
            return;
        };
        // A soft penalty is the same for every loop of this outer pair, so it
        // does not change which one wins; [`Self::put_p`] adds it on write.
        if self.region.verdict(pn.pos, qn.pos) == Verdict::Forbid
            || self.pair_out(eq.to, pn, qn, ep, eq.nuc)
        {
            return;
        }
        let Some(outer) = self.outer_corner(ep.to, qn, outer_pair) else {
            return;
        };
        let mut best: Option<(f64, DiagEntry, usize, usize)> = None;
        let s_hi = (SINGLE_MAX_LEN as usize).min(span.saturating_sub(1));
        for size in 2 * SEPARABLE_FLANK..=s_hi {
            let delta = span - size;
            let lo = SEPARABLE_FLANK.max(size.saturating_sub(back_q));
            let hi = (size - SEPARABLE_FLANK).min(run_p);
            if lo > hi {
                continue;
            }
            let fixed = lambda * f64::from(self.params.interior.values[size] + max_ninio.0);
            // Saturated asymmetry: |2 n1 − s| ≥ skew, two runs of n1.
            let low_hi = size.saturating_sub(self.skew) / 2;
            let high_lo = (size + self.skew).div_ceil(2);
            for (a, b) in [(lo, hi.min(low_hi)), (lo.max(high_lo), hi)] {
                if a > b {
                    continue;
                }
                let (klo, khi) = (pn.pos + 1 + a as u32, pn.pos + 1 + b as u32);
                let Some(e) = self.diag.best(delta, klo, khi) else {
                    continue;
                };
                let score = fixed + outer.score + e.h;
                if best.is_none_or(|b| score < b.0) {
                    let n1 = (e.k - pn.pos - 1) as usize;
                    best = Some((score, *e, n1, size - n1));
                }
            }
        }
        let Some((_, e, n1, n2)) = best else {
            return;
        };
        let energy = internal_energy_scalar(
            self.params,
            &InternalLoopScalar {
                outer: outer_pair,
                inner: e.key.pair(),
                boundary_left5p: outer.a,
                boundary_left3p: e.corner.a,
                boundary_right5p: e.corner.b,
                boundary_right3p: outer.b,
                n1,
                n2,
            },
        );
        let cell = Cell {
            mfe: e.cell.mfe + energy,
            cai: e.cell.cai + ep.cai + outer.cai + e.corner.cai + eq.cai,
        };
        let q_ae = self.product.reachable_states(ep.to)[0];
        let opens: SmallVec<[u32; 4]> = self
            .product
            .reachable_states(pn)
            .iter()
            .copied()
            .filter(|&q| self.ac.step_edge(pn, q, ep) == Some(q_ae))
            .collect();
        for q_open in opens {
            let key = PKey::of((pn, ep, q_open), (qn, eq, q_outer));
            self.put_p(eq.to, key, cell);
            self.pulled.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Write one single-branch closure into the outer `P` cell.
    fn write_single(&mut self, ctx: &SingleCtx, cai: f64, energy: Energy, q_outer: u32) {
        let key = PKey::of((ctx.pn, ctx.ep, ctx.q_open), (ctx.qn, ctx.eq, q_outer));
        self.put_p(
            ctx.eq.to,
            key,
            Cell {
                mfe: ctx.cell.mfe + energy,
                cai,
            },
        );
    }

    // --- multi-threaded twins of the single-branch push -------------------
    //
    // Each mirrors the method of the same name above exactly, with three
    // differences: `&self` rather than `&mut self` (many workers hold `&self`
    // at once), the read-only [`Self::fill_span_ro`] / [`Self::corner_span_ro`]
    // in place of the memoizing pair, and a candidate pushed onto a `&mut
    // Writes` buffer rather than merged into `self` directly via
    // [`Self::put_p`] — the merge happens later, in [`Writes::apply`], when the
    // buffered slice is folded back in worker order (see [`Self::fan_out`]).
    // Used only from [`Self::p_push`]'s `threads > 1` path; `threads <= 1`
    // keeps calling the methods above, unchanged.

    /// The `&self` twin of [`Self::enumerate_outer_pairs`]. With `live`, a
    /// left position no outer pair could close from within the window of a
    /// single-branch loop around `on` is passed over
    /// ([`outside::LiveOpens`]); returns how many were.
    fn enumerate_outer_pairs_ro(
        &self,
        w: &mut Writes,
        (on, key, cell): (Node, PKey, Cell),
        live: Option<&outside::LiveOpens>,
    ) -> usize {
        let mut skipped = 0;
        let mut rights = std::mem::take(&mut w.rights);
        self.right_closings_ro(on, key, &mut rights);
        let inner_open = key.open.node();
        let inner_pair = key.pair();
        let a = inner_open.pos;
        let p_lo = a
            .saturating_sub(SINGLE_MAX_LEN + 1)
            .max(self.must.min_left(a));
        for p in p_lo..a {
            if live.is_some_and(|l| !l.closes_within(p, on.pos, on.pos + SINGLE_MAX_LEN + 1)) {
                skipped += 1;
                continue;
            }
            let n1 = a - p - 1;
            for &pn in self.dfa.nodes_at(p) {
                for &ep in self.dfa.right(pn) {
                    if !self.mark_cut_sides(pn, ep, &mut rights) {
                        continue;
                    }
                    for &q_open in self.product.reachable_states(pn) {
                        let Some(q_ae) = self.ac.step_edge(pn, q_open, ep) else {
                            continue;
                        };
                        self.single_from_left_ro(
                            w, on, key, cell, inner_pair, pn, ep, q_open, q_ae, n1, &rights,
                        );
                    }
                }
            }
        }
        w.rights = rights;
        skipped
    }

    /// The `&self` twin of [`Self::single_from_left`].
    #[allow(clippy::too_many_arguments)]
    fn single_from_left_ro(
        &self,
        w: &mut Writes,
        on: Node,
        key: PKey,
        cell: Cell,
        inner_pair: BasePair,
        pn: Node,
        ep: Edge,
        q_open: u32,
        q_ae: u32,
        n1: u32,
        rights: &Rights,
    ) {
        let lefts = self.left_corners_ro(ep.to, q_ae, key.open.node(), key.q_open, n1);
        // Past an empty right flank every closure is an internal loop, which
        // needs a left corner reaching the inner pair's entry state.
        let q_hi = if n1 > 0 && lefts.is_empty() {
            on.pos
        } else {
            (on.pos + SINGLE_MAX_LEN - n1)
                .min(self.dfa.seq_len - 1)
                .min(self.must.max_right(on.pos))
        };
        for (i, side) in rights.sides.iter().enumerate() {
            if on.pos + side.n2 > q_hi {
                break;
            }
            if rights.cut.get(i).copied().unwrap_or(false)
                || self.pulled_shape(ep.to.pos, n1 as usize, on.pos, side.n2 as usize)
            {
                continue;
            }
            let Some(outer_pair) = canonical(ep.nuc, side.eq.nuc) else {
                continue;
            };
            let ctx = SingleCtx {
                on,
                key,
                cell,
                inner_pair,
                outer_pair,
                pn,
                ep,
                q_open,
                q_ae,
                qn: side.qn,
                eq: side.eq,
                n1: n1 as usize,
                n2: side.n2 as usize,
            };
            self.close_single_ro(w, &ctx, &lefts, rights.corners_of(side));
        }
    }

    /// The `&self` twin of [`Self::close_single`].
    fn close_single_ro(
        &self,
        w: &mut Writes,
        ctx: &SingleCtx,
        lefts: &[CornerAc],
        rights: &[(CornerAc, u32)],
    ) {
        match (ctx.n1, ctx.n2) {
            (0, 0) => self.close_stack_ro(w, ctx),
            (_, _) if ctx.n1 == 0 || ctx.n2 == 0 => self.close_bulge_ro(w, ctx),
            _ => self.close_internal_ro(w, ctx, lefts, rights),
        }
    }

    /// The `&self` twin of [`Self::close_stack`].
    fn close_stack_ro(&self, w: &mut Writes, ctx: &SingleCtx) {
        if ctx.ep.to != ctx.key.open.node() || ctx.q_ae != ctx.key.q_open || ctx.qn != ctx.on {
            return;
        }
        let Some(q_outer) = self.ac.step_edge(ctx.qn, ctx.key.q_outer, ctx.eq) else {
            return;
        };
        let energy = stack_energy(
            self.params,
            &StackPair {
                outer: ctx.outer_pair,
                inner: ctx.inner_pair,
            },
        );
        self.write_single_ro(
            w,
            ctx,
            ctx.cell.cai + ctx.ep.cai + ctx.eq.cai,
            energy,
            q_outer,
        );
    }

    /// The `&self` twin of [`Self::close_bulge`].
    fn close_bulge_ro(&self, w: &mut Writes, ctx: &SingleCtx) {
        let energy = bulge_energy(
            self.params,
            &BulgeLoop {
                outer: ctx.outer_pair,
                inner: ctx.inner_pair,
                size: ctx.n1 + ctx.n2,
            },
        );
        if ctx.n1 == 0 {
            if ctx.ep.to != ctx.key.open.node() || ctx.q_ae != ctx.key.q_open {
                return;
            }
            let (start, len) = self.fill_span_ro(ctx.on, ctx.key.q_outer, ctx.qn);
            for i in start..start + len {
                let (q_qn, right_cai) = self.fills.at(i);
                let Some(q_outer) = self.ac.step_edge(ctx.qn, q_qn, ctx.eq) else {
                    continue;
                };
                let cai = ctx.cell.cai + ctx.ep.cai + right_cai + ctx.eq.cai;
                self.write_single_ro(w, ctx, cai, energy, q_outer);
            }
        } else {
            if ctx.qn != ctx.on {
                return;
            }
            let (start, len) = self.fill_span_ro(ctx.ep.to, ctx.q_ae, ctx.key.open.node());
            let Some((_, left_cai)) = (start..start + len)
                .map(|i| self.fills.at(i))
                .find(|&(q, _)| q == ctx.key.q_open)
            else {
                return;
            };
            let Some(q_outer) = self.ac.step_edge(ctx.qn, ctx.key.q_outer, ctx.eq) else {
                return;
            };
            let cai = ctx.cell.cai + ctx.ep.cai + left_cai + ctx.eq.cai;
            self.write_single_ro(w, ctx, cai, energy, q_outer);
        }
    }

    /// The `&self` twin of [`Self::close_internal`].
    fn close_internal_ro(
        &self,
        w: &mut Writes,
        ctx: &SingleCtx,
        lefts: &[CornerAc],
        rights: &[(CornerAc, u32)],
    ) {
        if lefts.is_empty() {
            return;
        }
        let mut best = SmallVec::new();
        self.reduce_internal(ctx, lefts, rights, &mut best);
        for r in &best {
            self.write_single_ro(w, ctx, r.cai, r.energy, r.q_outer);
        }
    }

    /// The `&self` twin of [`Self::write_single`]: buffers the candidate into
    /// `w.p` instead of merging it in directly. The region-constraint verdict is
    /// applied here, before the chunk reduces its candidates, exactly as
    /// [`Self::put_p`] would apply it; [`Self::apply_all`] then merges the
    /// survivors without applying it again.
    fn write_single_ro(
        &self,
        w: &mut Writes,
        ctx: &SingleCtx,
        cai: f64,
        energy: Energy,
        q_outer: u32,
    ) {
        let key = PKey::of((ctx.pn, ctx.ep, ctx.q_open), (ctx.qn, ctx.eq, q_outer));
        let cand = Cell {
            mfe: ctx.cell.mfe + energy,
            cai,
        };
        if let Some(cand) = self.penalized(key, cand) {
            w.p.offer(ctx.eq.to, key, cand, self.lambda);
        }
    }

    // --- multi-branch assembly -------------------------------------------

    /// Consume `M2[pos]`: collapse to `M` (`M = M2`) and seed the
    /// right-extended table (`M2R = M2`, an empty 3′ flank); then wrap every
    /// `M2R[pos]` cell into a multi-loop in progress (`Multi`).
    ///
    /// The wrap closes against the frontier `pos` only. The multi-loop's 3′
    /// unpaired stretch is already inside the `M2R` cell, grown one nucleotide
    /// at a time by [`Self::m2r_extend`], so no wrap ever walks a flank. That is
    /// the separation the AC state permits: the 3′ flank couples to the closing
    /// frontier only through its exit state, which the `M2R` key carries.
    ///
    /// `threads > 1` splits the same way [`Self::p_push`] does: the two
    /// same-column writes stay serial, and the wrap fans out via
    /// [`Self::fan_out`], since it only ever writes `Multi` at positions past
    /// `pos`. The left openings it reads are computed serially first, so the
    /// workers read [`Self::lefts`] without filling it.
    fn m2_beam(&mut self, pos: u32, pool: Option<&Pool>) {
        for &on in self.dfa.nodes_at(pos) {
            let cells: Vec<(MKey, Cell)> = self.m2[self.dfa.ix(on)]
                .iter()
                .map(|(&k, &c)| (k, c))
                .collect();
            for (mk, cell) in cells {
                self.put_m(on, mk, cell);
                self.put_m2r(on, mk, cell);
            }
        }
        let cells: Vec<(Node, MKey, Cell)> = self
            .dfa
            .nodes_at(pos)
            .iter()
            .flat_map(|&on| {
                self.m2r[self.dfa.ix(on)]
                    .iter()
                    .map(move |(&k, &c)| (on, k, c))
            })
            .collect();
        for &(_, mk, _) in &cells {
            self.left_openings(mk.inner.node(), mk.q_inner);
        }
        let writes = self.fan_out(pool, &cells, |f, w, (on, mk, cell)| {
            f.wrap_m2r(w, on, mk, cell)
        });
        self.apply_all(writes, pool);
        // Serial on both paths, after the per-core wraps, so a threaded fold
        // stays identical to the serial one.
        self.far_multi(pos);
    }

    /// The multi-loop wraps of `M2R[pos]` whose 5′ flank is `SINGLE_MAX_LEN`
    /// or longer. A branching loop's leading unpaired run is unbounded in the
    /// energy model; [`Self::left_openings`] covers the shorter flanks per
    /// core, and this covers the rest in one sweep.
    ///
    /// An `M2R` cell's 3′ flank is already inside it, so a wrap depends on its
    /// core only through the frontier node and the AC state there, and a 5′
    /// flank grows by one position at a fixed price: the unpaired base is
    /// free, the new opening edge adds its adaptation, and the AC state
    /// threads back through it. So among the cores sharing a frontier and a
    /// state, only the best one per opening (node, nucleotide, AC state)
    /// matters, and the best one an opening further left is the best over the
    /// openings it extends. Each core offers its flank of exactly
    /// `SINGLE_MAX_LEN`; one right-to-left sweep extends every opening to
    /// position 0 — never across a position required to pair, which the
    /// longer flank would leave unpaired — and wraps each once, instead of a
    /// scan per core.
    ///
    /// The traceback needs nothing new: it rebuilds a wrap's 5′ flank as the
    /// best path through the span, which is the flank the sweep priced.
    fn far_multi(&mut self, pos: u32) {
        self.far_cuts.clear();
        self.far_wraps(pos, |f, key, wrap| f.wrap_far(key, wrap));
        let mut cuts = std::mem::take(&mut self.far_cuts);
        self.flush_cuts(&mut cuts);
        self.far_cuts = cuts;
    }

    /// The sweep behind [`Self::far_multi`], handing each opening's best core
    /// to `emit` in the order the sweep settles them. The traceback replays it
    /// to find which core a long-flank wrap took.
    fn far_wraps(&mut self, pos: u32, mut emit: impl FnMut(&mut Self, FarKey, FarWrap)) {
        let l = SINGLE_MAX_LEN;
        let mut far: FxHashMap<FarKey, FarWrap> = FxHashMap::default();
        let mut at: Vec<Vec<FarKey>> = vec![Vec::new(); pos as usize + 1];
        for &on in self.dfa.nodes_at(pos) {
            let cells: Vec<(MKey, Cell)> = self.m2r[self.dfa.ix(on)]
                .iter()
                .map(|(&k, &c)| (k, c))
                .collect();
            for (core, cell) in cells {
                let inner = core.inner.node();
                let min_left = self.must.min_left(inner.pos);
                let near = inner.pos.saturating_sub(l).max(min_left);
                if near == min_left {
                    continue;
                }
                let p = near - 1;
                for &pn in self.dfa.nodes_at(p) {
                    for &ep in self.dfa.right(pn) {
                        for &q_open in self.product.reachable_states(pn) {
                            let Some(q_ae) = self.ac.step_edge(pn, q_open, ep) else {
                                continue;
                            };
                            let (start, len) = self.fill_span(ep.to, q_ae, inner);
                            let Some((_, left_cai)) = (start..start + len)
                                .map(|i| self.fills.at(i))
                                .find(|&(q, _)| q == core.q_inner)
                            else {
                                continue;
                            };
                            let key = FarKey {
                                outer: on,
                                q_outer: core.q_outer,
                                open: pn,
                                nuc: ep.nuc as u8,
                                q_open,
                            };
                            let cand = FarWrap {
                                core,
                                cell,
                                ep,
                                left_cai,
                            };
                            cand.merge_into(&mut far, &mut at, key, self.lambda);
                        }
                    }
                }
            }
        }
        for p in (0..at.len()).rev() {
            for key in std::mem::take(&mut at[p]) {
                let Some(f) = far.remove(&key) else {
                    continue;
                };
                emit(self, key, f);
                if p == 0 || self.must.is_must(p as u32) {
                    continue;
                }
                let pn = key.open;
                for &le in self.dfa.left(pn) {
                    let ep = Edge { to: pn, ..le };
                    for &q in self.product.reachable_states(le.to) {
                        if self.ac.step_edge(le.to, q, ep) != Some(key.q_open) {
                            continue;
                        }
                        let cand = FarWrap {
                            ep,
                            left_cai: f.ep.cai + f.left_cai,
                            ..f
                        };
                        let next = FarKey {
                            open: le.to,
                            nuc: le.nuc as u8,
                            q_open: q,
                            ..key
                        };
                        cand.merge_into(&mut far, &mut at, next, self.lambda);
                    }
                }
            }
        }
    }

    /// Close one long-flank opening against every closing edge out of its
    /// frontier: the [`Self::wrap_m2r`] write, for a wrap [`Self::far_multi`]
    /// found.
    fn wrap_far(&mut self, key: FarKey, f: FarWrap) {
        let on = key.outer;
        let mut cuts = std::mem::take(&mut self.far_cuts);
        for &eq in self.dfa.right(on) {
            let Some(q_outer) = self.ac.step_edge(on, key.q_outer, eq) else {
                continue;
            };
            if canonical(f.ep.nuc, eq.nuc).is_none()
                || self.pair_out_cached(&mut cuts, eq.to, key.open, on, f.ep, eq.nuc)
            {
                continue;
            }
            let pkey = PKey::of((key.open, f.ep, key.q_open), (on, eq, q_outer));
            let cand = Cell {
                mfe: f.cell.mfe,
                cai: f.ep.cai + f.left_cai + f.cell.cai + eq.cai,
            };
            self.put_multi(eq.to, pkey, cand);
        }
        self.far_cuts = cuts;
    }

    /// Every multi-loop opening that can enclose a core whose 5′ boundary is
    /// `(inner, q_inner)`: the opening node and nucleotide, the AC state before
    /// it, and the CAI of the opening edge plus the best 5′ flank reaching
    /// `q_inner` — memoized in [`Self::lefts`], since every `M2R` cell sharing
    /// that boundary asks the same question.
    fn left_openings(&mut self, inner: Node, q_inner: u32) {
        if self.lefts.contains_key(&(inner, q_inner)) {
            return;
        }
        let mut found = Vec::new();
        // The multi-loop's 5′ unpaired stretch is `[p + 1, inner.pos)`. It is
        // unbounded; flanks of `SINGLE_MAX_LEN` and longer are wrapped by
        // [`Self::far_multi`].
        let p_lo = inner
            .pos
            .saturating_sub(SINGLE_MAX_LEN)
            .max(self.must.min_left(inner.pos));
        for p in p_lo..inner.pos {
            for &pn in self.dfa.nodes_at(p) {
                for &ep in self.dfa.right(pn) {
                    for &q_open in self.product.reachable_states(pn) {
                        let Some(q_ae) = self.ac.step_edge(pn, q_open, ep) else {
                            continue;
                        };
                        let (start, len) = self.fill_span(ep.to, q_ae, inner);
                        let Some((_, left_cai)) = (start..start + len)
                            .map(|i| self.fills.at(i))
                            .find(|&(q, _)| q == q_inner)
                        else {
                            continue;
                        };
                        found.push(LeftOpen {
                            pn,
                            ep,
                            q_open,
                            cai: ep.cai + left_cai,
                        });
                    }
                }
            }
        }
        self.lefts.insert((inner, q_inner), found);
    }

    /// `Multi = (closing pair) [5′ flank] M2R`: close one `M2R` cell at its own
    /// frontier `on` against every opening [`Self::left_openings`] recorded,
    /// buffering each candidate for [`Self::apply_all`].
    fn wrap_m2r(&self, w: &mut Writes, on: Node, core: MKey, cell: Cell) {
        let lefts = &self.lefts[&(core.inner.node(), core.q_inner)];
        for &eq in self.dfa.right(on) {
            let Some(q_outer) = self.ac.step_edge(on, core.q_outer, eq) else {
                continue;
            };
            for l in lefts {
                if canonical(l.ep.nuc, eq.nuc).is_none()
                    || self.pair_out_cached(&mut w.wrap_cuts, eq.to, l.pn, on, l.ep, eq.nuc)
                {
                    continue;
                }
                let key = PKey::of((l.pn, l.ep, l.q_open), (on, eq, q_outer));
                w.multi.offer(
                    eq.to,
                    key,
                    Cell {
                        mfe: cell.mfe,
                        cai: l.cai + cell.cai + eq.cai,
                    },
                    self.lambda,
                );
            }
        }
    }

    /// `M2R` extends by one unpaired nucleotide on the right — the multi-loop's
    /// 3′ unpaired stretch, energy-free like [`Self::m_extend`]'s, threading `q`.
    fn m2r_extend(&mut self, pos: u32) {
        // The extension declares `pos` unpaired inside the multi-loop.
        if self.must.is_must(pos) {
            return;
        }
        for &on in self.dfa.nodes_at(pos) {
            let cells: Vec<(MKey, Cell)> = self.m2r[self.dfa.ix(on)]
                .iter()
                .map(|(&k, &c)| (k, c))
                .collect();
            for (mk, cell) in cells {
                for &e in self.dfa.right(on) {
                    let Some(q2) = self.ac.step_edge(on, mk.q_outer, e) else {
                        continue;
                    };
                    self.put_m2r(
                        e.to,
                        MKey { q_outer: q2, ..mk },
                        Cell {
                            mfe: cell.mfe,
                            cai: cell.cai + e.cai,
                        },
                    );
                }
            }
        }
    }

    /// `M` extends by one unpaired nucleotide on the right (energy-free at
    /// `dangles = 0`), threading `q` on the 3′ side.
    fn m_extend(&mut self, pos: u32) {
        // The extension declares `pos` unpaired inside the multi-loop.
        if self.must.is_must(pos) {
            return;
        }
        for &on in self.dfa.nodes_at(pos) {
            let cells: Vec<(MKey, Cell)> = self.m[self.dfa.ix(on)]
                .iter()
                .map(|(&k, &c)| (k, c))
                .collect();
            for (mk, cell) in cells {
                for &e in self.dfa.right(on) {
                    let Some(q2) = self.ac.step_edge(on, mk.q_outer, e) else {
                        continue;
                    };
                    self.put_m(
                        e.to,
                        MKey {
                            inner: mk.inner,
                            q_inner: mk.q_inner,
                            q_outer: q2,
                        },
                        Cell {
                            mfe: cell.mfe,
                            cai: cell.cai + e.cai,
                        },
                    );
                }
            }
        }
    }

    // --- energy helpers ---------------------------------------------------

    /// Per-stem terminal-AU correction: `terminal_au_37` for AU / GU closures,
    /// zero otherwise.
    fn terminal_au(&self, pair: BasePair) -> Energy {
        if pair.applies_terminal_au() {
            Energy(self.params.misc.terminal_au_37)
        } else {
            Energy::ZERO
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

    // --- provenance, re-derived ---------------------------------------------
    //
    // The tables keep values only; the traceback recovers how each cell on the
    // optimal path was derived by re-running the producers that can write it,
    // in the order the fill ran them, and taking the first candidate that
    // reproduces the stored value exactly. A cell is the first candidate to
    // reach its minimum (a later one replaces it only when strictly better),
    // and an earlier candidate with the same value would have been kept in its
    // place, so the first exact match is the derivation the fill kept, tie
    // included. Each producer is re-run on the tables as the fill left them,
    // which are the tables it read: no column is written after it is consumed.

    /// How `C[node][q]` (holding `cell`) was derived.
    fn derive_c(&self, node: Node, q: u32, cell: Cell) -> Option<CBack> {
        if node == self.dfa.start() {
            return Some(CBack::ExtEmpty);
        }
        let pos = node.pos;
        // `c_extend` at the previous position.
        if !self.must.is_must(pos - 1) {
            for &prev in self.dfa.nodes_at(pos - 1) {
                for (&pq, &pc) in &self.c[self.dfa.ix(prev)] {
                    for &e in self.dfa.right(prev) {
                        if e.to != node || self.ac.step_edge(prev, pq, e) != Some(q) {
                            continue;
                        }
                        if reproduces(cell, pc.mfe, pc.cai + e.cai) {
                            return Some(CBack::ExtUnpaired {
                                prev: prev.into(),
                                prev_q: pq,
                                nuc: e.nuc as u8,
                            });
                        }
                    }
                }
            }
        }
        // `p_push` here: the exterior absorbs a closed pair.
        for (&key, &pc) in &self.p[self.dfa.ix(node)] {
            if key.q_outer != q || self.beyond_bound(node, &key, pc) {
                continue;
            }
            let Some(&prefix) = self.c[self.dfa.ix(key.open.node())].get(&key.q_open) else {
                continue;
            };
            let mfe = prefix.mfe + pc.mfe + self.terminal_au(key.pair());
            if reproduces(cell, mfe, prefix.cai + pc.cai) {
                return Some(CBack::ExtPair { pair: key });
            }
        }
        None
    }

    /// How `P[outer][key]` (holding `cell`) was derived: the hairpin seed, then
    /// the single-branch closures in push order, then the multi-loop closure.
    fn derive_p(&mut self, outer: Node, key: PKey, cell: Cell) -> Option<CBack> {
        let dfa = self.dfa;
        let pn = key.open.node();
        let ep = key.open_edge(dfa)?;
        let eq = key.close_edge(dfa)?;
        debug_assert_eq!(
            eq.to, outer,
            "a pair's cell sits at its closing edge's target"
        );
        let pen = self.pair_penalty(key);
        let opened = self.ac.step_edge(pn, key.q_open, ep);
        if let (Some(q_ae), Some(outer_pair)) = (opened, canonical(ep.nuc, eq.nuc)) {
            if let Some(back) = self.derive_hairpin(key, ep, eq, q_ae, outer_pair, cell, pen) {
                return Some(back);
            }
            if let Some(back) = self.derive_single(key, ep, eq, q_ae, outer_pair, cell, pen) {
                return Some(back);
            }
        }
        let &mc = self.multi[self.dfa.ix(outer)].get(&key)?;
        let mfe = mc.mfe + self.multi_close_cost(key.pair());
        reproduces(cell, mfe + pen, mc.cai).then_some(CBack::MultiClose)
    }

    /// The single-branch closures' candidates for `key`, inner pairs in the
    /// order [`Self::p_push`] consumed them.
    #[allow(clippy::too_many_arguments)]
    fn derive_single(
        &mut self,
        key: PKey,
        ep: Edge,
        eq: Edge,
        q_ae: u32,
        outer_pair: BasePair,
        cell: Cell,
        pen: Energy,
    ) -> Option<CBack> {
        let dfa = self.dfa;
        let qn = key.close.node();
        for ipos in qn.pos.saturating_sub(SINGLE_MAX_LEN)..=qn.pos {
            for &on in dfa.nodes_at(ipos) {
                let inners: Vec<(PKey, Cell)> = self.p[self.dfa.ix(on)]
                    .iter()
                    .filter(|&(ik, &ic)| !self.beyond_bound(on, ik, ic))
                    .map(|(&ik, &ic)| (ik, ic))
                    .collect();
                for (ikey, icell) in inners {
                    let found = self.single_from(on, ikey, icell, key, ep, eq, q_ae, outer_pair);
                    if let Some(&(_, _, right_q)) = found
                        .iter()
                        .find(|&&(mfe, cai, _)| reproduces(cell, mfe + pen, cai))
                    {
                        return Some(CBack::Single {
                            inner: ikey,
                            right_q,
                        });
                    }
                }
            }
        }
        None
    }

    /// The hairpin seed's candidates for `key`, as [`Self::seed_hairpins_from`]
    /// scored them.
    #[allow(clippy::too_many_arguments)]
    fn derive_hairpin(
        &mut self,
        key: PKey,
        eo: Edge,
        ec: Edge,
        q_ao: u32,
        closing: BasePair,
        cell: Cell,
        pen: Energy,
    ) -> Option<CBack> {
        let bn = key.close.node();
        let b_end = self
            .dfa
            .seq_len
            .min(self.must.max_right(eo.to.pos).saturating_add(1));
        if bn.pos < eo.to.pos + MIN_HAIRPIN_LOOP || bn.pos >= b_end {
            return None;
        }
        for (energy, interior_cai, q_close) in self.score_hairpin(eo.to, q_ao, bn, closing) {
            if self.ac.step_edge(bn, q_close, ec) != Some(key.q_outer) {
                continue;
            }
            if reproduces(cell, energy + pen, eo.cai + interior_cai + ec.cai) {
                return Some(CBack::Hairpin {
                    interior_q: q_close,
                });
            }
        }
        None
    }

    /// The single-branch closures of the inner pair `P[on][ikey]` that write
    /// the outer pair `target`, as `(energy, cai, right_q)` before the outer
    /// pair's soft penalty, in the order [`Self::enumerate_outer_pairs`] wrote
    /// them. Empty where the push never reached that outer pair.
    #[allow(clippy::too_many_arguments)]
    fn single_from(
        &mut self,
        on: Node,
        ikey: PKey,
        icell: Cell,
        target: PKey,
        ep: Edge,
        eq: Edge,
        q_ae: u32,
        outer_pair: BasePair,
    ) -> SmallVec<[(Energy, f64, u32); 4]> {
        let mut out = SmallVec::new();
        let (pn, qn) = (target.open.node(), target.close.node());
        let a = ikey.open.node().pos;
        // The 5′ opening the push sweeps from, and the closings it visits.
        let p_lo = a
            .saturating_sub(SINGLE_MAX_LEN + 1)
            .max(self.must.min_left(a));
        if pn.pos >= a || pn.pos < p_lo || qn.pos < on.pos || qn.pos > self.closing_limit(on) {
            return out;
        }
        let n1 = a - pn.pos - 1;
        let n2 = qn.pos - on.pos;
        if n2 == 0 && qn != on {
            return out;
        }
        let lefts = self.left_corners(ep.to, q_ae, ikey.open.node(), ikey.q_open, n1);
        let q_hi = if n1 > 0 && lefts.is_empty() {
            on.pos
        } else {
            (on.pos + SINGLE_MAX_LEN - n1)
                .min(self.dfa.seq_len - 1)
                .min(self.must.max_right(on.pos))
        };
        if on.pos + n2 > q_hi {
            return out;
        }
        let ctx = SingleCtx {
            on,
            key: ikey,
            cell: icell,
            inner_pair: ikey.pair(),
            outer_pair,
            pn,
            ep,
            q_open: target.q_open,
            q_ae,
            qn,
            eq,
            n1: n1 as usize,
            n2: n2 as usize,
        };
        for (cai, energy, q_outer, right_q) in self.closures_of(&ctx, &lefts, target.q_outer) {
            if q_outer == target.q_outer {
                out.push((ctx.cell.mfe + energy, cai, right_q));
            }
        }
        out
    }

    /// Every candidate one fully-fixed closure writes, as `(cai, energy,
    /// q_outer, right_q)` in write order: the bodies of [`Self::close_stack`],
    /// [`Self::close_bulge`] and [`Self::close_internal`] with the write
    /// collected. Internal-loop right corners not leading to `q_outer` are
    /// skipped, which leaves the reduction for `q_outer` as it was.
    fn closures_of(
        &mut self,
        ctx: &SingleCtx,
        lefts: &[CornerAc],
        q_outer_wanted: u32,
    ) -> SmallVec<[(f64, Energy, u32, u32); 4]> {
        let mut out = SmallVec::new();
        let (ikey, on, qn, eq, outer_pair) = (ctx.key, ctx.on, ctx.qn, ctx.eq, ctx.outer_pair);
        let mut keep = |cai: f64, energy: Energy, q_outer: u32, right_q: u32| {
            out.push((cai, energy, q_outer, right_q));
        };
        match (ctx.n1, ctx.n2) {
            (0, 0) => {
                if ctx.ep.to != ikey.open.node() || ctx.q_ae != ikey.q_open {
                    return SmallVec::new();
                }
                let Some(q_outer) = self.ac.step_edge(qn, ikey.q_outer, eq) else {
                    return SmallVec::new();
                };
                let energy = stack_energy(
                    self.params,
                    &StackPair {
                        outer: outer_pair,
                        inner: ctx.inner_pair,
                    },
                );
                keep(
                    ctx.cell.cai + ctx.ep.cai + ctx.eq.cai,
                    energy,
                    q_outer,
                    ikey.q_outer,
                );
            }
            (n1, n2) if n1 == 0 || n2 == 0 => {
                let energy = bulge_energy(
                    self.params,
                    &BulgeLoop {
                        outer: outer_pair,
                        inner: ctx.inner_pair,
                        size: n1 + n2,
                    },
                );
                if n1 == 0 {
                    if ctx.ep.to != ikey.open.node() || ctx.q_ae != ikey.q_open {
                        return SmallVec::new();
                    }
                    let (start, len) = self.fill_span(on, ikey.q_outer, qn);
                    for i in start..start + len {
                        let (q_qn, right_cai) = self.fills.at(i);
                        let Some(q_outer) = self.ac.step_edge(qn, q_qn, eq) else {
                            continue;
                        };
                        let cai = ctx.cell.cai + ctx.ep.cai + right_cai + ctx.eq.cai;
                        keep(cai, energy, q_outer, q_qn);
                    }
                } else {
                    let (start, len) = self.fill_span(ctx.ep.to, ctx.q_ae, ikey.open.node());
                    let Some((_, left_cai)) = (start..start + len)
                        .map(|i| self.fills.at(i))
                        .find(|&(q, _)| q == ikey.q_open)
                    else {
                        return SmallVec::new();
                    };
                    let Some(q_outer) = self.ac.step_edge(qn, ikey.q_outer, eq) else {
                        return SmallVec::new();
                    };
                    let cai = ctx.cell.cai + ctx.ep.cai + left_cai + ctx.eq.cai;
                    keep(cai, energy, q_outer, ikey.q_outer);
                }
            }
            _ => {
                if lefts.is_empty() {
                    return SmallVec::new();
                }
                let (start, len) = self.corner_span(on, ikey.q_outer, qn);
                let rights: Vec<(CornerAc, u32)> = (start..start + len)
                    .map(|i| self.corners.at(i))
                    .filter_map(|rc| {
                        let q_outer = self.ac.step_edge(qn, rc.q_to, eq)?;
                        (q_outer == q_outer_wanted).then_some((rc, q_outer))
                    })
                    .collect();
                let mut best = SmallVec::new();
                self.reduce_internal(ctx, lefts, &rights, &mut best);
                for r in &best {
                    keep(r.cai, r.energy, r.q_outer, r.right_q);
                }
            }
        }
        out
    }

    /// How `Multi[outer][key]` (holding `cell`) was derived: the wrap of which
    /// `M2R` core, short flanks first and then the long-flank sweep, as
    /// [`Self::m2_beam`] ran them.
    fn derive_multi(&mut self, key: PKey, cell: Cell) -> Option<CBack> {
        let on = key.close.node();
        let eq = key.close_edge(self.dfa)?;
        let cores: Vec<(MKey, Cell)> = self.m2r[self.dfa.ix(on)]
            .iter()
            .map(|(&k, &c)| (k, c))
            .collect();
        for &(core, mc) in &cores {
            if self.ac.step_edge(on, core.q_outer, eq) != Some(key.q_outer) {
                continue;
            }
            let Some(lefts) = self.lefts.get(&(core.inner.node(), core.q_inner)) else {
                continue;
            };
            let open = lefts.iter().find(|l| {
                l.pn == key.open.node()
                    && l.ep.nuc as u8 == key.nuc_open
                    && l.ep.to.lane == key.lane_open
                    && l.q_open == key.q_open
            });
            if let Some(l) = open {
                if canonical(l.ep.nuc, eq.nuc).is_some()
                    && reproduces(cell, mc.mfe, l.cai + mc.cai + eq.cai)
                {
                    return Some(CBack::MultiWrap { core });
                }
            }
        }
        let mut found = None;
        self.far_wraps(on.pos, |f, fk, w| {
            if found.is_some()
                || fk.outer != on
                || fk.open != key.open.node()
                || fk.nuc != key.nuc_open
                || w.ep.to.lane != key.lane_open
                || fk.q_open != key.q_open
                || canonical(w.ep.nuc, eq.nuc).is_none()
                || f.ac.step_edge(on, fk.q_outer, eq) != Some(key.q_outer)
            {
                return;
            }
            if reproduces(
                cell,
                w.cell.mfe,
                w.ep.cai + w.left_cai + w.cell.cai + eq.cai,
            ) {
                found = Some(CBack::MultiWrap { core: w.core });
            }
        });
        found
    }

    /// How `M[on][mk]` (holding `cell`) was derived: the unpaired extension
    /// from the previous position, then `M = P`, then `M = M2`.
    fn derive_m(&self, on: Node, mk: MKey, cell: Cell) -> Option<CBack> {
        if let Some(back) = self.derive_unpaired(&self.m, on, mk, cell) {
            return Some(CBack::MUnpaired {
                prev: back.0,
                prev_q: back.1,
                nuc: back.2,
            });
        }
        for (&key, &pc) in &self.p[self.dfa.ix(on)] {
            if key.open != mk.inner
                || key.q_open != mk.q_inner
                || key.q_outer != mk.q_outer
                || self.beyond_bound(on, &key, pc)
            {
                continue;
            }
            if reproduces(cell, pc.mfe + self.m_stem_cost(key.pair()), pc.cai) {
                return Some(CBack::MFromP { pair: key });
            }
        }
        let &m2 = self.m2[self.dfa.ix(on)].get(&mk)?;
        reproduces(cell, m2.mfe, m2.cai).then_some(CBack::MFromM2)
    }

    /// How `M2[on][mk]` (holding `cell`) was derived: the bifurcation of which
    /// closed pair, in push order.
    fn derive_m2(&self, on: Node, mk: MKey, cell: Cell) -> Option<CBack> {
        for (&key, &pc) in &self.p[self.dfa.ix(on)] {
            if key.q_outer != mk.q_outer || self.beyond_bound(on, &key, pc) {
                continue;
            }
            let left = MKey {
                q_outer: key.q_open,
                ..mk
            };
            let Some(&lc) = self.m[self.dfa.ix(key.open.node())].get(&left) else {
                continue;
            };
            let stem = self.m_stem_cost(key.pair());
            if reproduces(cell, lc.mfe + pc.mfe + stem, lc.cai + pc.cai) {
                return Some(CBack::M2Bifurcation { right: key });
            }
        }
        None
    }

    /// How `M2R[on][mk]` (holding `cell`) was derived: the unpaired extension
    /// from the previous position, then `M2R = M2`.
    fn derive_m2r(&self, on: Node, mk: MKey, cell: Cell) -> Option<CBack> {
        if let Some(back) = self.derive_unpaired(&self.m2r, on, mk, cell) {
            return Some(CBack::M2RUnpaired {
                prev: back.0,
                prev_q: back.1,
                nuc: back.2,
            });
        }
        let &m2 = self.m2[self.dfa.ix(on)].get(&mk)?;
        reproduces(cell, m2.mfe, m2.cai).then_some(CBack::M2RFromM2)
    }

    /// The one-nucleotide extension into `table[on][mk]` from the previous
    /// position that [`Self::m_extend`] / [`Self::m2r_extend`] kept, as
    /// `(prev node, prev 3′ state, nucleotide)`.
    fn derive_unpaired(
        &self,
        table: &[FxHashMap<MKey, Cell>],
        on: Node,
        mk: MKey,
        cell: Cell,
    ) -> Option<(NodeIx, u32, u8)> {
        if on.pos == 0 || self.must.is_must(on.pos - 1) {
            return None;
        }
        for &prev in self.dfa.nodes_at(on.pos - 1) {
            for (&pk, &pc) in &table[self.dfa.ix(prev)] {
                if pk.inner != mk.inner || pk.q_inner != mk.q_inner {
                    continue;
                }
                for &e in self.dfa.right(prev) {
                    if e.to != on || self.ac.step_edge(prev, pk.q_outer, e) != Some(mk.q_outer) {
                        continue;
                    }
                    if reproduces(cell, pc.mfe, pc.cai + e.cai) {
                        return Some((prev.into(), pk.q_outer, e.nuc as u8));
                    }
                }
            }
        }
        None
    }

    /// Walk the filled tables back from the terminal exterior parse ending in AC
    /// state `end_q`, respelling the `(sequence, structure)` pair. Mirrors
    /// [`crate::traceback`] with product keys and `q` threading; the jumped-over
    /// loop interiors are respelled by the AC-threaded span primitives.
    fn traceback(&mut self, end_q: u32) -> (String, String) {
        self.traceback_at(self.dfa.end(), end_q)
    }

    /// [`Self::traceback`] from the terminal `end`.
    fn traceback_at(&mut self, end: Node, end_q: u32) -> (String, String) {
        let n = self.dfa.seq_len as usize;
        let mut w = Walk {
            f: self,
            seq: vec![Nucleotide::A; n],
            dots: vec![b'.'; n],
        };
        w.walk_c(end, end_q);
        let sequence: String = w.seq.iter().map(|&nt| nuc_char(nt)).collect();
        let structure = String::from_utf8(w.dots).expect("dot-bracket bytes are ASCII");
        (sequence, structure)
    }
}

/// The backward walk over one filled [`Folder`]: it writes nucleotides and
/// brackets into position-indexed buffers as it visits cells, respelling the
/// jumped-over loop interiors from each cell's re-derived provenance. Every CKY coordinate
/// lands at a unique nucleotide slot, so the writes never collide.
struct Walk<'w, 'a> {
    f: &'w mut Folder<'a>,
    seq: Vec<Nucleotide>,
    dots: Vec<u8>,
}

impl Walk<'_, '_> {
    /// Follow the exterior chain from `(node, q)`, absorbing unpaired bases and
    /// closed pairs.
    fn walk_c(&mut self, node: Node, q: u32) {
        let mut cursor = Some((node, q));
        while let Some((node, q)) = cursor {
            let Some(&cell) = self.f.c[self.f.dfa.ix(node)].get(&q) else {
                return;
            };
            let back = self
                .f
                .derive_c(node, q, cell)
                .expect("a stored cell is reproduced by one of its producers");
            match back {
                CBack::ExtEmpty => return,
                CBack::ExtUnpaired { prev, prev_q, nuc } => {
                    let prev = prev.node();
                    self.seq[prev.pos as usize] = nuc_from_index(nuc);
                    cursor = Some((prev, prev_q));
                }
                CBack::ExtPair { pair } => {
                    self.walk_p(PCellRef {
                        outer: node.into(),
                        key: pair,
                    });
                    cursor = Some((pair.open.node(), pair.q_open));
                }
                _ => return,
            }
        }
    }

    /// Emit a closed pair's brackets and boundary nucleotides, then spell its
    /// interior per the closure kind.
    fn walk_p(&mut self, pref: PCellRef) {
        let key = pref.key;
        let a = key.open.node().pos as usize;
        let b = key.close.node().pos as usize;
        self.seq[a] = nuc_from_index(key.nuc_open);
        self.seq[b] = nuc_from_index(key.nuc_close);
        self.dots[a] = b'(';
        self.dots[b] = b')';
        let Some(&cell) = self.f.p[self.f.dfa.ix(pref.outer.node())].get(&key) else {
            return;
        };
        let back = self
            .f
            .derive_p(pref.outer.node(), key, cell)
            .expect("a stored cell is reproduced by one of its producers");
        match back {
            CBack::Hairpin { interior_q } => self.spell_hairpin(key, cell, interior_q),
            CBack::Single { inner, right_q } => {
                let inner = PCellRef {
                    outer: self.after_close(inner).into(),
                    key: inner,
                };
                self.spell_single(pref, inner, right_q, cell);
                self.walk_p(inner);
            }
            CBack::MultiClose => self.walk_multi(pref),
            _ => {}
        }
    }

    /// Respell a hairpin loop interior, reproducing the per-exit-state choice the
    /// seed made (`interior_q` pins the exit; the `(energy, cai)` pins which
    /// interior at that exit).
    fn spell_hairpin(&mut self, key: PKey, cell: Cell, interior_q: u32) {
        let nuc_open = nuc_from_index(key.nuc_open);
        let nuc_close = nuc_from_index(key.nuc_close);
        let from = self.after_open(key);
        let entry_q = self.open_state(key);
        let closing = pair_from_nucleotides(nuc_open, nuc_close);
        let size = (key.close.node().pos - from.pos) as usize;
        let (ep_cai, eq_cai) = self.edge_cais(key);
        let target_cai = cell.cai - ep_cai - eq_cai;
        // The stored energy carries any soft region penalty; the loop scored
        // below does not, so the target is the cell's energy net of it.
        let target_mfe = Energy(cell.mfe.0 - self.f.pair_penalty(key).0);
        let seq = self.hairpin_interior(
            from,
            entry_q,
            key.close.node(),
            closing,
            size,
            interior_q,
            target_mfe,
            target_cai,
        );
        self.write_span(from.pos, &seq);
    }

    /// The hairpin interior spelling for exit state `interior_q` matching the
    /// seed's `(energy, cai)`. Special sizes (3/4/6) enumerate whole spellings;
    /// the rest select the terminal-mismatch corner.
    #[allow(clippy::too_many_arguments)]
    fn hairpin_interior(
        &self,
        from: Node,
        entry_q: u32,
        to: Node,
        closing: BasePair,
        size: usize,
        interior_q: u32,
        target_mfe: Energy,
        target_cai: f64,
    ) -> Vec<Nucleotide> {
        if matches!(size, 3 | 4 | 6) {
            for (cai, seq, q_close) in span_spellings(self.f.dfa, self.f.ac, from, entry_q, to) {
                if q_close != interior_q {
                    continue;
                }
                let energy = hairpin_energy(
                    self.f.params,
                    &HairpinLoop {
                        closing,
                        loop_seq: &seq,
                    },
                );
                if energy == target_mfe && (cai - target_cai).abs() < CAI_TOL {
                    return seq;
                }
            }
        } else {
            for c in span_corners_spelled(self.f.dfa, self.f.ac, from, entry_q, to) {
                if c.q_to != interior_q {
                    continue;
                }
                let energy = hairpin_energy_scalar(self.f.params, closing, size, c.first, c.last);
                if energy == target_mfe && (c.cai - target_cai).abs() < CAI_TOL {
                    return c.seq;
                }
            }
        }
        debug_assert!(false, "hairpin interior not recovered in traceback");
        Vec::new()
    }

    /// Respell the flanks of a stack / bulge / internal closure around the inner
    /// pair.
    fn spell_single(&mut self, outer: PCellRef, inner: PCellRef, right_q: u32, cell: Cell) {
        let okey = outer.key;
        let ikey = inner.key;
        let left_from = self.after_open(okey);
        let q_ae = self.open_state(okey);
        let inner_outer = inner.outer.node();
        let qn = okey.close.node();
        let n1 = ikey.open.node().pos - left_from.pos;
        let n2 = qn.pos - inner_outer.pos;
        if n1 == 0 && n2 == 0 {
            return; // Stack: the inner pair sits directly on the outer, no interior.
        }
        if n1 > 0 && n2 > 0 {
            self.spell_internal(outer, inner, cell, right_q);
        } else if n1 > 0 {
            // Bulge with empty right flank; fill the left flank best-CAI.
            let seq = span_fill_path(
                self.f.dfa,
                self.f.ac,
                left_from,
                q_ae,
                ikey.open.node(),
                ikey.q_open,
            );
            self.write_span(left_from.pos, &seq);
        } else {
            // Bulge with empty left flank; fill the right flank best-CAI.
            let seq = span_fill_path(
                self.f.dfa,
                self.f.ac,
                inner_outer,
                ikey.q_outer,
                qn,
                right_q,
            );
            self.write_span(inner_outer.pos, &seq);
        }
    }

    /// Respell both flanks of an internal loop, selecting the terminal-mismatch
    /// corners whose joint `(energy, cai)` matches the recorded closure. The
    /// flank coordinates are recomputed from the outer/inner cell refs.
    fn spell_internal(&mut self, outer: PCellRef, inner: PCellRef, cell: Cell, right_q: u32) {
        let okey = outer.key;
        let ikey = inner.key;
        let left_from = self.after_open(okey);
        let q_ae = self.open_state(okey);
        let inner_outer = inner.outer.node();
        let n1 = (ikey.open.node().pos - left_from.pos) as usize;
        let n2 = (okey.close.node().pos - inner_outer.pos) as usize;
        let inner_cell = self.f.p[self.f.dfa.ix(inner.outer.node())][&ikey];
        // `cell` carries the outer pair's soft penalty and `inner_cell` its own;
        // the loop energy scored below carries neither, so only the outer one is
        // taken back off here.
        let target_energy = Energy(cell.mfe.0 - inner_cell.mfe.0 - self.f.pair_penalty(okey).0);
        let (ep_cai, eq_cai) = self.edge_cais(okey);
        let target_cai = cell.cai - inner_cell.cai - ep_cai - eq_cai;
        let outer_pair = pair_from_nucleotides(
            nuc_from_index(okey.nuc_open),
            nuc_from_index(okey.nuc_close),
        );
        let inner_pair = ikey.pair();
        let lefts: Vec<CornerSpelled> =
            span_corners_spelled(self.f.dfa, self.f.ac, left_from, q_ae, ikey.open.node())
                .into_iter()
                .filter(|c| c.q_to == ikey.q_open)
                .collect();
        let rights: Vec<CornerSpelled> = span_corners_spelled(
            self.f.dfa,
            self.f.ac,
            inner_outer,
            ikey.q_outer,
            okey.close.node(),
        )
        .into_iter()
        .filter(|c| c.q_to == right_q)
        .collect();
        for lc in &lefts {
            for rc in &rights {
                let energy = internal_energy_scalar(
                    self.f.params,
                    &InternalLoopScalar {
                        outer: outer_pair,
                        inner: inner_pair,
                        boundary_left5p: lc.first,
                        boundary_left3p: lc.last,
                        boundary_right5p: rc.first,
                        boundary_right3p: rc.last,
                        n1,
                        n2,
                    },
                );
                if energy == target_energy && (lc.cai + rc.cai - target_cai).abs() < CAI_TOL {
                    self.write_span(left_from.pos, &lc.seq);
                    self.write_span(inner_outer.pos, &rc.seq);
                    return;
                }
            }
        }
        debug_assert!(false, "internal-loop flanks not recovered in traceback");
    }

    /// Respell a multi-loop wrap: fill the 5′ unpaired flank best-CAI and
    /// descend into the `M2R` core, which spells the 3′ flank.
    fn walk_multi(&mut self, m: PCellRef) {
        let key = m.key;
        let Some(&cell) = self.f.multi[self.f.dfa.ix(m.outer.node())].get(&key) else {
            return;
        };
        let back = self
            .f
            .derive_multi(key, cell)
            .expect("a stored cell is reproduced by one of its producers");
        let CBack::MultiWrap { core } = back else {
            return;
        };
        let core = MCellRef {
            outer: key.close,
            key: core,
        };
        let left_from = self.after_open(key);
        let q_ae = self.open_state(key);
        let left = span_fill_path(
            self.f.dfa,
            self.f.ac,
            left_from,
            q_ae,
            core.key.inner.node(),
            core.key.q_inner,
        );
        self.write_span(left_from.pos, &left);
        self.walk_m2r(core);
    }

    /// Follow an `M2R` chain: spell the multi-loop's 3′ unpaired flank, then
    /// descend into the `M2` core it extends.
    fn walk_m2r(&mut self, span: MCellRef) {
        let mut cursor = span;
        loop {
            let Some(&cell) = self.f.m2r[self.f.dfa.ix(cursor.outer.node())].get(&cursor.key)
            else {
                return;
            };
            let back = self
                .f
                .derive_m2r(cursor.outer.node(), cursor.key, cell)
                .expect("a stored cell is reproduced by one of its producers");
            match back {
                CBack::M2RFromM2 => return self.walk_m2(cursor),
                CBack::M2RUnpaired { prev, prev_q, nuc } => {
                    self.seq[prev.node().pos as usize] = nuc_from_index(nuc);
                    cursor = MCellRef {
                        outer: prev,
                        key: MKey {
                            q_outer: prev_q,
                            ..cursor.key
                        },
                    };
                }
                _ => return,
            }
        }
    }

    /// Split an `M2` bifurcation into its left `M` and right `P` at the shared
    /// split node.
    fn walk_m2(&mut self, m: MCellRef) {
        let Some(&cell) = self.f.m2[self.f.dfa.ix(m.outer.node())].get(&m.key) else {
            return;
        };
        let back = self
            .f
            .derive_m2(m.outer.node(), m.key, cell)
            .expect("a stored cell is reproduced by one of its producers");
        if let CBack::M2Bifurcation { right } = back {
            self.walk_m(MCellRef {
                outer: right.open,
                key: MKey {
                    q_outer: right.q_open,
                    ..m.key
                },
            });
            self.walk_p(PCellRef {
                outer: m.outer,
                key: right,
            });
        }
    }

    /// Follow an `M` chain: absorb unpaired bases, then descend into the branch
    /// (`M ← P`) or the sub-bifurcation (`M ← M2`).
    fn walk_m(&mut self, span: MCellRef) {
        let mut cursor = Some(span);
        while let Some(s) = cursor {
            let Some(&cell) = self.f.m[self.f.dfa.ix(s.outer.node())].get(&s.key) else {
                return;
            };
            let back = self
                .f
                .derive_m(s.outer.node(), s.key, cell)
                .expect("a stored cell is reproduced by one of its producers");
            match back {
                CBack::MFromP { pair } => {
                    return self.walk_p(PCellRef {
                        outer: s.outer,
                        key: pair,
                    })
                }
                CBack::MFromM2 => return self.walk_m2(s),
                CBack::MUnpaired { prev, prev_q, nuc } => {
                    self.seq[prev.node().pos as usize] = nuc_from_index(nuc);
                    cursor = Some(MCellRef {
                        outer: prev,
                        key: MKey {
                            q_outer: prev_q,
                            ..s.key
                        },
                    });
                }
                _ => return,
            }
        }
    }

    /// Write a spelled interior into the sequence buffer from `start` (a
    /// nucleotide position), one nucleotide per slot.
    fn write_span(&mut self, start: u32, seq: &[Nucleotide]) {
        for (k, &nt) in seq.iter().enumerate() {
            self.seq[start as usize + k] = nt;
        }
    }

    /// The AC state immediately inside a recorded closure's opening base: `key.q_open`
    /// threaded across the opening nucleotide at `key.open.node().pos`.
    ///
    /// The forward scan wrote the cell only after taking this transition, so it
    /// is not one the automaton removes; a `None` here would mean the tables
    /// disagree with the automaton that filled them.
    fn open_state(&self, key: PKey) -> u32 {
        let ep = key
            .open_edge(self.f.dfa)
            .expect("a pair's opening edge is in the lattice");
        self.f
            .ac
            .step_edge(key.open.node(), key.q_open, ep)
            .expect("the opening transition of a recorded closure is not a removed one")
    }

    /// The node past the pair's opening edge.
    fn after_open(&self, key: PKey) -> Node {
        key.open_edge(self.f.dfa)
            .expect("a pair's opening edge is in the lattice")
            .to
    }

    /// The node past the pair's closing edge: the column its cell sits in.
    fn after_close(&self, key: PKey) -> Node {
        key.close_edge(self.f.dfa)
            .expect("a pair's closing edge is in the lattice")
            .to
    }

    /// The adaptation weights of the pair's opening and closing edges.
    fn edge_cais(&self, key: PKey) -> (f64, f64) {
        let dfa = self.f.dfa;
        let cai = |e: Option<Edge>| e.expect("a pair's edges are in the lattice").cai;
        (cai(key.open_edge(dfa)), cai(key.close_edge(dfa)))
    }
}

/// Tolerance for matching a recovered interior's CAI log-weight against the
/// recorded cell during traceback: distinct interior choices differ by far more
/// than this, so it only absorbs summation-order float noise.
const CAI_TOL: f64 = 1e-6;

/// The ASCII character for a nucleotide.
fn nuc_char(nt: Nucleotide) -> char {
    match nt {
        Nucleotide::A => 'A',
        Nucleotide::C => 'C',
        Nucleotide::G => 'G',
        Nucleotide::U => 'U',
    }
}

/// The fully-fixed coordinates of one single-branch closure — the inner cell,
/// the outer pair's opening/closing edges and boundary states, and the flank
/// sizes — shared by the stack / bulge / internal scorers so they agree on the
/// cell they write.
struct SingleCtx {
    on: Node,
    key: PKey,
    cell: Cell,
    inner_pair: BasePair,
    outer_pair: BasePair,
    pn: Node,
    ep: Edge,
    q_open: u32,
    q_ae: u32,
    qn: Node,
    eq: Edge,
    n1: usize,
    n2: usize,
}

/// [`Folder::mark_cut_sides`]'s marks for the inner column node `column`,
/// keyed by outer opening (node, nucleotide), each with how many sides it
/// cuts. A side's cut depends only on that column node, the opening and the
/// side, not on which inner pair is pushed, so every cell of a column shares
/// them.
#[derive(Default)]
struct CutMarks {
    column: Option<Node>,
    by_opening: FxHashMap<(NodeIx, u8), (Vec<bool>, usize, usize)>,
}

/// The right closings of one inner pair — see [`Folder::right_closings`].
#[derive(Default)]
struct Rights {
    /// One per `(closing node, closing edge)`, in the order the closures visit
    /// them (right flank length ascending).
    sides: Vec<RightSide>,
    /// The right corners of every side, each with the AC state its closing
    /// edge leaves it in; a side addresses its run by [`RightSide::corners`].
    corners: Vec<(CornerAc, u32)>,
    /// Per side, for the opening being swept: whether the bound's pair floor
    /// rules its outer pair out ([`Folder::mark_cut_sides`]). Empty when
    /// nothing is cut.
    cut: Vec<bool>,
}

impl Rights {
    /// Empty both lists, keeping their capacity.
    fn clear(&mut self) {
        self.sides.clear();
        self.corners.clear();
    }

    /// The right corners `side` can close with.
    fn corners_of(&self, side: &RightSide) -> &[(CornerAc, u32)] {
        &self.corners[side.corners.clone()]
    }
}

/// One right closing: the right flank's length, the closing node and edge, and
/// the run of [`Rights::corners`] it carries (empty for an empty flank).
struct RightSide {
    n2: u32,
    qn: Node,
    eq: Edge,
    corners: std::ops::Range<usize>,
}

/// One corner of an internal loop on free flanks: its score (mismatch energy
/// at the fold's weight less the flanks' CAI halves), the two nucleotides it
/// reads, and those CAI halves.
#[derive(Clone, Copy)]
struct Corner {
    score: f64,
    a: Nucleotide,
    b: Nucleotide,
    cai: f64,
}

/// The surviving internal-loop candidate for one `P` key, kept by
/// [`Folder::reduce_internal`]: the key's exit state, the right flank's exit
/// state for the traceback, the loop energy and CAI to write, the BRD it was
/// ranked on, and the left corner's position in the memo (the tie-break).
#[derive(Clone, Copy)]
struct Reduced {
    q_outer: u32,
    right_q: u32,
    energy: Energy,
    cai: f64,
    score: f64,
    left: usize,
}

/// One multi-loop opening recorded by [`Folder::left_openings`]: the opening
/// node and edge, the AC state entering it, and the CAI of the opening edge
/// plus the best 5′ flank reaching the core.
struct LeftOpen {
    pn: Node,
    ep: Edge,
    q_open: u32,
    cai: f64,
}

/// Bifurcation: every cell of `lefts` (the `M` column at `key.open.node()`) whose 3′
/// state matches this pair's entry state `key.q_open` is a left multi-branch
/// context for `P[on][key]`; the shared split node `(key.open.node(), key.q_open)` is
/// the single product node both sub-derivations agree on. Each resulting `M2`
/// candidate goes to `emit`, in the column's iteration order.
fn bifurcate(
    lefts: &FxHashMap<MKey, Cell>,
    key: PKey,
    cell: Cell,
    stem: Energy,
    mut emit: impl FnMut(MKey, Cell),
) {
    for (&mk, &mcell) in lefts.iter().filter(|(mk, _)| mk.q_outer == key.q_open) {
        emit(
            MKey {
                inner: mk.inner,
                q_inner: mk.q_inner,
                q_outer: key.q_outer,
            },
            Cell {
                mfe: mcell.mfe + cell.mfe + stem,
                cai: mcell.cai + cell.cai,
            },
        );
    }
}

/// Whether a candidate `(mfe, cai)` is exactly the value `cell` holds: the
/// traceback's test for which producer wrote a cell.
fn reproduces(cell: Cell, mfe: Energy, cai: f64) -> bool {
    let (stored_mfe, stored_cai) = (cell.mfe, cell.cai);
    stored_mfe == mfe && stored_cai.to_bits() == cai.to_bits()
}

/// Fixed-λ BRD score `λ·MFE − (1−λ)·CAI` (lower is better), matching
/// [`crate::state::State::brd`].
fn brd(mfe: Energy, cai: f64, lambda: f64) -> f64 {
    lambda * f64::from(mfe.0) - (1.0 - lambda) * (cai * 100.0)
}

/// Merge `cand` into `map[key]`, keeping the lower-BRD cell (ties keep the
/// incumbent).
fn merge<K: std::hash::Hash + Eq>(map: &mut FxHashMap<K, Cell>, key: K, cand: Cell, lambda: f64) {
    match map.get_mut(&key) {
        Some(existing)
            if brd(cand.mfe, cand.cai, lambda) < brd(existing.mfe, existing.cai, lambda) =>
        {
            *existing = cand;
        }
        Some(_) => {}
        None => {
            map.insert(key, cand);
        }
    }
}

/// The CAI-half completion bound per node index — [`Dfa::span_tail_cai`], the
/// best still-achievable adaptation of the prefix ending at each node: the
/// whole units before it and the partial codon straddling it. Empty unless [`Beam::cai_bound`]
/// is set. The AC-threaded analogue of [`crate::tables`]'s `end_cai_table`; the
/// bound is the *unconstrained* best fill, which stays an admissible optimistic
/// bound under a motif constraint (the constraint can only lower achievable
/// CAI, never raise it).
fn end_cai_table(dfa: &Dfa, beam: Beam) -> Vec<f64> {
    if !beam.cai_bound {
        return Vec::new();
    }
    let n = dfa.slots();
    (0..n)
        .map(|i| dfa.span_tail_cai(dfa.node_of(i)).unwrap_or(0.0))
        .collect()
}

/// The exterior-MFE completion estimate for a boundary node at absolute AC state
/// `q`: the best left-exterior parse `C[node][q]`, or `0.0` when none is
/// recorded yet (a neutral optimistic constraint, matching [`crate::tables`]'s
/// `ext_completion`). Read live from the exterior table, whose entry for every
/// node left of a pruned column is finalized before that column is pruned.
fn ext_completion(
    c: &[FxHashMap<u32, Cell>],
    layout: NodeLayout,
    node: Node,
    q: u32,
    lambda: f64,
) -> f64 {
    c[layout.ix(node)]
        .get(&q)
        .map_or(0.0, |cell| brd(cell.mfe, cell.cai, lambda))
}

/// Prune one span column under the beam policy, grouping on its 5′ boundary
/// node — the AC-threaded analogue of [`crate::tables`]'s `prune_grouped`.
/// `boundary` extracts each key's `(open node, AC state)`: `(key.open.node(),
/// key.q_open)` for the closed-pair (`P` / `Multi`) tables, `(key.inner,
/// key.q_inner)` for the multi-branch (`M` / `M2`) tables. Each cell is ranked
/// by its partial BRD, optionally less the CAI-half completion bound and/or plus
/// the exterior-MFE completion estimate at that boundary; at most `per_open`
/// cells survive per open node, then at most `cap` overall. The exact policy is a
/// no-op (the caller already short-circuits it). Unlike the dense unconstrained
/// `M` table (one cell per open node, where the per-open cap is a structural
/// no-op), each constrained column holds several AC states per node, so the
/// per-open cap genuinely bites on all four tables.
fn prune_span_column<K: Copy + Eq + std::hash::Hash>(
    map: &mut FxHashMap<K, Cell>,
    beam: Beam,
    lambda: f64,
    end_cai: &[f64],
    c: &[FxHashMap<u32, Cell>],
    layout: NodeLayout,
    boundary: impl Fn(&K) -> (Node, u32),
) {
    if beam.per_open == usize::MAX && map.len() <= beam.cap {
        return;
    }
    let kk = (1.0 - lambda) * 100.0;
    let scored: Vec<(f64, K, usize)> = map
        .iter()
        .map(|(&k, cell)| {
            let (open, q) = boundary(&k);
            let mut r = brd(cell.mfe, cell.cai, lambda);
            if beam.cai_bound {
                r -= kk * end_cai[layout.ix(open)];
            }
            if beam.mfe_bound {
                r += ext_completion(c, layout, open, q, lambda);
            }
            (r, k, layout.ix(open))
        })
        .collect();
    let keep = beam_keep(&scored, beam);
    for (i, &kept) in keep.iter().enumerate() {
        if !kept {
            map.remove(&scored[i].1);
        }
    }
}

/// The keep-mask for a beam prune of `scored` (each `(rank, key, open_index)`):
/// at most `beam.per_open` cells survive per distinct open-node group, then at
/// most `beam.cap` overall, lowest-rank first. The AC-threaded analogue of
/// [`crate::tables`]'s `cap_per_open` + `cap_flat`, generic over the key type.
fn beam_keep<K: Copy>(scored: &[(f64, K, usize)], beam: Beam) -> Vec<bool> {
    let mut keep = vec![true; scored.len()];
    if beam.per_open != usize::MAX {
        let mut by_open: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (i, cell) in scored.iter().enumerate() {
            by_open.entry(cell.2).or_default().push(i);
        }
        for idxs in by_open.values_mut() {
            if idxs.len() > beam.per_open {
                idxs.select_nth_unstable_by(beam.per_open, |&a, &b| {
                    scored[a].0.total_cmp(&scored[b].0)
                });
                for &i in &idxs[beam.per_open..] {
                    keep[i] = false;
                }
            }
        }
    }
    if keep.iter().filter(|&&k| k).count() > beam.cap {
        let mut idxs: Vec<usize> = (0..scored.len()).filter(|&i| keep[i]).collect();
        idxs.select_nth_unstable_by(beam.cap, |&a, &b| scored[a].0.total_cmp(&scored[b].0));
        for &i in &idxs[beam.cap..] {
            keep[i] = false;
        }
    }
    keep
}

/// Canonical Watson–Crick / wobble pair, or `None` for a non-pairing pair —
/// matching [`crate::scan`]'s `canonical`.
fn canonical(a: Nucleotide, b: Nucleotide) -> Option<BasePair> {
    match pair_from_nucleotides(a, b) {
        BasePair::NN => None,
        pair => Some(pair),
    }
}

/// One constrained `P` cell beside the outside value of its base pair under
/// each supplied outside table: the per-cell record a pruning-margin survival
/// curve is built from. Measurement only; nothing is pruned.
pub(crate) struct MarginRow {
    /// Position of the opening nucleotide.
    pub open: u32,
    /// Position of the closing nucleotide.
    pub close: u32,
    pub mfe: Energy,
    pub cai: f64,
    /// Per outside table, in order; `NAN` where the table lacks the pair.
    pub outside: Vec<f64>,
}

/// The exact fold's optimum beside every `P` cell's [`MarginRow`].
pub(crate) struct MarginDump {
    pub mfe: Energy,
    pub cai: f64,
    pub rows: Vec<MarginRow>,
}

/// Fold exactly, then record every `P` cell with the outside value of its
/// base pair under each table in `outsides` (each indexed by outer node, as
/// `P` is). The caller turns these into margins against whatever ceiling it
/// chooses.
pub(crate) fn margin_rows(
    product: &ProductDfa,
    params: &Params,
    lambda: f64,
    outsides: &[&[FxHashMap<crate::state::PairKey, f64>]],
) -> Option<MarginDump> {
    let mut f = Folder::new(
        product,
        params,
        lambda,
        Beam::exact(),
        PairConstraint::inert(),
        MustPairConstraint::inert(),
        1,
    );
    let end_q = f.run()?;
    let end = f.c[f.dfa.ix(f.dfa.end())][&end_q];
    let mut rows = Vec::new();
    for (o, column) in f.p.iter().enumerate() {
        for (key, cell) in column {
            let pair = key.pair_key();
            let outside = outsides
                .iter()
                .map(|t| {
                    t.get(o)
                        .and_then(|col| col.get(&pair))
                        .copied()
                        .unwrap_or(f64::NAN)
                })
                .collect();
            rows.push(MarginRow {
                open: key.open.node().pos,
                close: key.close.node().pos,
                mfe: cell.mfe,
                cai: cell.cai,
                outside,
            });
        }
    }
    Some(MarginDump {
        mfe: end.mfe,
        cai: end.cai,
        rows,
    })
}

#[cfg(test)]
mod tests {
    //! The span primitives against brute-force enumeration.
    //!
    //! [`span_spellings`] enumerates *every* motif-free spelling of a span, so
    //! it is the reference the two summarizing primitives are checked against:
    //! [`span_fill`] must equal the best CAI per exit state over that
    //! enumeration, and [`span_corners`] the best CAI per `(first, last, exit
    //! state)`. A separate check pins the unconstrained fill to
    //! [`Dfa::broken_codon_cai`], so the AC-threaded DP reduces to the substrate
    //! it generalizes when the automaton imposes no constraint.

    use std::collections::BTreeMap;

    use super::*;
    use crate::ac::{Ac, MotifWindow, ScopedAc};
    use crate::dfa::{self};
    use crate::CodonUsage;
    use derna_codon_dp::codon_table::aa_to_index;

    /// On a stop-labelled lattice, each terminal of one motif-constrained fold
    /// is the fold with that stop written into the 3′ flank: the same optimum,
    /// a design spelling its own stop, motif-free, scoring as reported. A motif
    /// that spells a stop leaves that terminal unreachable, as it leaves the
    /// pinned lattice infeasible.
    ///
    /// This is coverage of the labelled lattice through the constrained fold,
    /// not a test of [`Dfa::connected`]: the fold threads every span through
    /// the product lattice's own edges, and the cases pass with connectivity
    /// across stop lanes forced true.
    #[test]
    fn each_terminal_is_the_pinned_stop_constrained_optimum() {
        use crate::dfa::{UtrChain, STOP_CODON_NUCS};
        use crate::UtrProfile;
        use derna_codon_dp::codon_table::STOP;
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let nucs = |s: &str| -> Vec<Nucleotide> {
            s.bytes()
                .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
                .collect()
        };
        let cases: &[(&str, &str, &str, &[&str])] = &[
            ("MKF", "GGAC", "CUCCAGG", &["GG"]),
            ("MKL", "ACGC", "AGCGUCCGCUAAA", &["UAG", "CCC"]),
            ("MFR", "GCGA", "UCGCAAGGCGC", &["GAU"]),
            ("MAKF", "CC", "AUGGCAUUUGCCAUA", &["UAA"]),
            ("MWE", "GAAA", "GGGAAACCC", &["AGG", "UGAG"]),
            ("MGS", "", "CCUUCGGGAAAG", &["GN"]),
            ("MKRL", "GGC", "GGGCGCAAGCGCCCAAAGGGCGCAAGCGCCC", &["CGCG"]),
            ("MCWA", "CA", "AGCCGCAAGCGGCUAGCCUUCGGGCUAA", &["AAA"]),
        ];
        let mut reached = [0usize; 3];
        for &(protein, utr5, utr3, motifs) in cases {
            let mut aa: Vec<usize> = protein
                .chars()
                .map(|c| aa_to_index(c).expect("standard amino acid"))
                .collect();
            let (u5, u3) = (nucs(utr5), nucs(utr3));
            let ac = Ac::build(motifs).expect("valid motifs");
            let pinned: Vec<Dfa> = STOP_CODON_NUCS
                .iter()
                .map(|stop| {
                    let tail: Vec<Nucleotide> = stop.iter().chain(&u3).copied().collect();
                    dfa::build_with_utr(&aa, &usage, &u5, &tail)
                })
                .collect();
            aa.push(STOP);
            let graph = dfa::build_stop_labelled(
                &aa,
                &usage,
                UtrChain::Fixed(&u5),
                UtrChain::Fixed(&u3),
                UtrProfile::inert(),
            )
            .expect("a designed stop");
            let product = ProductDfa::build(&graph, &ac);
            for lambda in [0.05, 0.5, 1.0] {
                let ends = fold_ends(&product, &params, lambda, PairConstraint::inert());
                for (k, (got, pin)) in ends.iter().zip(&pinned).enumerate() {
                    let want = fold(
                        &ProductDfa::build(pin, &ac),
                        &params,
                        lambda,
                        PairConstraint::inert(),
                    );
                    let tag = format!("{protein} × {motifs:?} λ={lambda} stop {k}");
                    match (got, want) {
                        (None, None) => {}
                        (Some(g), Some(w)) => {
                            reached[k] += 1;
                            let b = |f: &ConstrainedFold| {
                                lambda * f64::from(f.mfe.0) - (1.0 - lambda) * f.cai * 100.0
                            };
                            assert!((b(g) - b(&w)).abs() < 1e-6, "{tag}: {} vs {}", b(g), b(&w));
                            let seq = nucs(&g.sequence);
                            let at = graph.cds_end() as usize - 3;
                            assert_eq!(&seq[at..at + 3], &STOP_CODON_NUCS[k], "{tag}: stop");
                            assert!(is_motif_free(&ac, &seq), "{tag}: motif in {}", g.sequence);
                            let reeval = evaluate(&params, g.sequence.as_bytes(), &g.structure)
                                .expect("recovered structure scores");
                            assert_eq!(reeval.dekacal(), g.mfe.0, "{tag}: {}", g.structure);
                        }
                        (g, w) => panic!(
                            "{tag}: labelled feasible {} pinned feasible {}",
                            g.is_some(),
                            w.is_some()
                        ),
                    }
                }
            }
        }
        assert!(
            reached.iter().all(|&n| n > 0),
            "every stop reached somewhere: {reached:?}"
        );
    }

    /// A packed pair reads back as the pair it was packed from, at the
    /// largest position the packing holds and on both lanes.
    #[test]
    fn packed_pair_round_trips() {
        let top = (1 << PACK_POS) - 1;
        let nucs = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
        for (open, close) in [(0, 0), (3, 17), (top - 1, top), (top, 5)] {
            for lanes in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                for (a, b) in nucs.iter().zip(nucs.iter().rev()) {
                    let pair = crate::state::PairKey {
                        open: Node::new(open, lanes.0),
                        close: Node::new(close, lanes.1),
                        opening: crate::state::Opening::new(*a, lanes.1),
                        nuc_close: *b,
                    };
                    assert_eq!(unpack_pair(pack_pair(&pair)), pair);
                }
            }
        }
    }

    /// The table entries' sizes, pinned: the fold is bound by the traffic
    /// through these maps, so growing an entry is a slowdown and should be
    /// a visible decision rather than a side effect of adding a field.
    #[test]
    fn table_entries_stay_small() {
        use std::mem::size_of;
        assert_eq!(size_of::<Cell>(), 12);
        assert_eq!(size_of::<(PKey, Cell)>(), 32);
        assert_eq!(size_of::<(MKey, Cell)>(), 24);
        assert_eq!(size_of::<(u32, Cell)>(), 16);
    }

    /// At λ = 0 the objective is adaptation alone, so the constrained
    /// optimum's CAI is the best motif-free spelling's — the value
    /// [`open_chain_cai`] claims, found here by a full fold instead. Its
    /// unpaired objective must also cap the optimum at every λ, which is what
    /// makes it a ceiling.
    #[test]
    fn open_chain_is_the_zero_lambda_optimum_and_caps_every_lambda() {
        let params = Params::load_default().unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("KFKF", &["GUU", "AAG"]),
            ("MWFDKE", &["CC"]),
            ("NYARTRRFDK", &["GCC", "AAG"]),
            // Ending on alanine (GCN) against motifs starting `CG` / `UC`
            // leaves the automaton in several states at the end, so the
            // best spelling is a choice among exit states.
            ("KWDA", &["CGA", "UCU"]),
            ("MKSA", &["CGG", "UCG"]),
        ];
        let (mut checked, mut several_exits) = (0, 0);
        for &(protein, motifs) in cases {
            let (dfa, ac) = build(protein, motifs);
            let product = ProductDfa::build(&dfa, &ac);
            let Some(open) = open_chain_cai(&product) else {
                assert!(optimum(&product, &params, 0.5, PairConstraint::inert()).is_none());
                continue;
            };
            let exits = span_fill(
                product.dfa(),
                product.lattice_ac(),
                dfa.start(),
                product.lattice_ac().start(),
                dfa.end(),
            );
            several_exits += usize::from(exits.len() > 1);
            let best = optimum(&product, &params, 0.0, PairConstraint::inert()).expect("feasible");
            assert!(
                (open - best.cai).abs() < 1e-9,
                "{protein}: open chain {open}, λ=0 optimum {}",
                best.cai
            );
            for lambda in [0.0001, 0.1, 0.5, 0.99] {
                let o =
                    optimum(&product, &params, lambda, PairConstraint::inert()).expect("feasible");
                let ceiling = -(1.0 - lambda) * open * 100.0;
                assert!(
                    ceiling >= brd(o.mfe, o.cai, lambda) - 1e-9,
                    "{protein} λ={lambda}: open chain {ceiling} below the optimum"
                );
            }
            checked += 1;
        }
        assert!(checked > 0, "no case was feasible");
        assert!(
            several_exits > 0,
            "every case ended in one automaton state, so the choice among exits went untested"
        );
    }

    /// `ac` enforced over the whole molecule — the span primitives' input where
    /// a case is not about scoping.
    fn global<'a>(dfa: &'a Dfa, ac: &'a Ac) -> LatticeAc<'a> {
        LatticeAc::unmerged(ScopedAc::new(ac, MotifWindow::all()), dfa)
    }

    fn build(protein: &str, motifs: &[&str]) -> (Dfa, Ac) {
        let usage = CodonUsage::load_default().expect("default codon usage");
        let aa: Vec<usize> = protein
            .chars()
            .map(|c| aa_to_index(c).expect("standard amino acid"))
            .collect();
        (
            dfa::build(&aa, &usage),
            Ac::build(motifs).expect("valid motifs"),
        )
    }

    /// Best CAI per exit state over the full enumeration — the reference
    /// [`span_fill`] summarizes. Floats are bucketed to a stable key so ties
    /// compare exactly.
    fn enumerated_best_per_exit(
        dfa: &Dfa,
        ac: LatticeAc<'_>,
        from: Node,
        q_from: u32,
        to: Node,
    ) -> BTreeMap<u32, i64> {
        let mut best: BTreeMap<u32, f64> = BTreeMap::new();
        for (cai, _seq, q_to) in span_spellings(dfa, ac, from, q_from, to) {
            best.entry(q_to)
                .and_modify(|w| {
                    if cai > *w {
                        *w = cai;
                    }
                })
                .or_insert(cai);
        }
        best.into_iter().map(|(q, cai)| (q, bucket(cai))).collect()
    }

    /// A stable integer key for a CAI log-weight (rounded to 1e-9), so the
    /// max-CAI comparison is exact despite summation-order float noise.
    fn bucket(cai: f64) -> i64 {
        (cai * 1e9).round() as i64
    }

    /// Every `(from, to)` node span of a few lengths, across all lanes.
    fn spans(dfa: &Dfa, lens: &[u32]) -> Vec<(Node, Node)> {
        let mut out = Vec::new();
        for p in 0..dfa.seq_len {
            for &len in lens {
                let q = p + len;
                if q > dfa.seq_len {
                    continue;
                }
                for &from in dfa.nodes_at(p) {
                    for &to in dfa.nodes_at(q) {
                        out.push((from, to));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn fill_matches_enumerated_best_per_exit_state() {
        let (dfa, ac) = build("KFKF", &["GUU", "AAG", "UUC"]);
        for (from, to) in spans(&dfa, &[1, 2, 3, 5, 7]) {
            for q_from in 0..ac.num_states() as u32 {
                if ac.is_accept(q_from) {
                    continue;
                }
                let got: BTreeMap<u32, i64> = span_fill(&dfa, global(&dfa, &ac), from, q_from, to)
                    .into_iter()
                    .map(|(q, cai)| (q, bucket(cai)))
                    .collect();
                let want = enumerated_best_per_exit(&dfa, global(&dfa, &ac), from, q_from, to);
                assert_eq!(
                    got, want,
                    "span_fill diverged from enumeration: {from:?}→{to:?} q_from={q_from}",
                );
            }
        }
    }

    #[test]
    fn corners_match_enumeration() {
        let (dfa, ac) = build("KFKF", &["GUU", "AAG", "UUC"]);
        for (from, to) in spans(&dfa, &[1, 2, 3, 5]) {
            for q_from in 0..ac.num_states() as u32 {
                if ac.is_accept(q_from) {
                    continue;
                }
                // Reference: best CAI per (first, last, exit state) over the
                // full enumeration.
                let mut want: BTreeMap<(u8, u8, u32), f64> = BTreeMap::new();
                for (cai, seq, q_to) in span_spellings(&dfa, global(&dfa, &ac), from, q_from, to) {
                    let key = (seq[0] as u8, seq[seq.len() - 1] as u8, q_to);
                    want.entry(key)
                        .and_modify(|w| {
                            if cai > *w {
                                *w = cai;
                            }
                        })
                        .or_insert(cai);
                }
                let want: BTreeMap<_, i64> =
                    want.into_iter().map(|(k, cai)| (k, bucket(cai))).collect();
                let got: BTreeMap<_, i64> = span_corners(&dfa, global(&dfa, &ac), from, q_from, to)
                    .into_iter()
                    .map(|c| ((c.first as u8, c.last as u8, c.q_to), bucket(c.cai)))
                    .collect();
                assert_eq!(
                    got, want,
                    "span_corners diverged from enumeration: {from:?}→{to:?} q_from={q_from}",
                );
            }
        }
    }

    #[test]
    fn unconstrained_fill_matches_broken_codon_cai() {
        // With no motifs the automaton is a single non-accepting root, so the
        // fill has exactly one exit state (the root) whose CAI is the
        // substrate's `broken_codon_cai`.
        let (dfa, ac) = build("MARNDKF", &[]);
        let root = ac.start();
        for (from, to) in spans(&dfa, &[1, 2, 3, 4, 6, 9]) {
            let fill = span_fill(&dfa, global(&dfa, &ac), from, root, to);
            match dfa.broken_codon_cai(from, to) {
                Some(want) => {
                    assert_eq!(fill.len(), 1, "single AC state expected: {from:?}→{to:?}");
                    assert_eq!(fill[0].0, root);
                    assert_eq!(
                        bucket(fill[0].1),
                        bucket(want),
                        "unconstrained fill diverged from broken_codon_cai: {from:?}→{to:?}",
                    );
                }
                None => assert!(
                    fill.is_empty(),
                    "fill found a path where broken_codon_cai reports none: {from:?}→{to:?}",
                ),
            }
        }
    }

    /// The premise a re-derived reduction would rest on, measured in both
    /// directions.
    ///
    /// The substrate's interior reductions minimize over a loop's interior
    /// nucleotides *independently of the far boundary*; the AC axis couples that
    /// choice to the boundary it threads to, which is why the constrained folder
    /// is un-reduced. Scoping is argued to restore the independence outside the
    /// window, on the grounds that one state per node leaves the aggregation
    /// intact — an argument, not a measurement.
    ///
    /// Measured here at the level the reductions actually act on: a span lying
    /// wholly outside the window must collapse to a **single** exit state whose
    /// value is exactly [`Dfa::broken_codon_cai`], the unconstrained closed form
    /// the reductions call. A span overlapping the window must not, or the
    /// outside result says nothing about scoping.
    #[test]
    fn the_interior_fill_reduces_to_the_closed_form_outside_the_window_only() {
        let usage = CodonUsage::load_default().expect("default codon usage");
        let aa: Vec<usize> = "MKWDEYKF"
            .chars()
            .map(|c| aa_to_index(c).expect("standard amino acid"))
            .collect();
        let flank: Vec<Nucleotide> = "GGCAUGGCAUGG"
            .bytes()
            .map(|b| Nucleotide::from_byte(b).expect("A/C/G/U"))
            .collect();
        let graph = dfa::build_with_utr_chain(
            &aa,
            &usage,
            dfa::UtrChain::Fixed(&flank),
            dfa::UtrChain::Designed(12),
        );
        let ac = Ac::build(&["GGC", "CCA", "GAU"]).expect("valid motifs");
        // The window is the designed 3′ flank, so the given flank and the whole
        // CDS lie outside it.
        let window = MotifWindow::new(graph.cds_end(), graph.seq_len);
        let scoped = LatticeAc::unmerged(ScopedAc::new(&ac, window), &graph);

        let (mut outside, mut inside) = (0usize, 0usize);
        for (from, to) in spans(&graph, &[1, 2, 3, 5, 8]) {
            let fill = span_fill(&graph, scoped, from, scoped.start(), to);
            if to.pos <= window.lo() || from.pos >= window.hi() {
                outside += 1;
                assert_eq!(
                    fill.len(),
                    1,
                    "a span wholly outside the window carried a split axis: \
                     {from:?}→{to:?}",
                );
                let want = graph
                    .broken_codon_cai(from, to)
                    .expect("a spannable interval has a closed-form fill");
                assert_eq!(
                    bucket(fill[0].1),
                    bucket(want),
                    "the scoped fill diverged from the closed form outside the \
                     window: {from:?}→{to:?}",
                );
            } else if fill.len() > 1 {
                inside += 1;
            }
        }
        assert!(outside > 0, "no span lay outside the window");
        assert!(
            inside > 0,
            "no span inside the window split the axis, so the outside result \
             cannot distinguish scoping from an automaton that does nothing",
        );
        println!(
            "spans collapsing to the closed form outside the window: {outside}; \
             spans splitting inside it: {inside}",
        );
    }

    // -----------------------------------------------------------------------
    // The constrained optimum against brute-force enumeration.
    // -----------------------------------------------------------------------

    use crate::product::ProductDfa;
    use crate::{fold_single_path, Params};
    use rna_mfe::evaluate;

    /// The brute-force constrained optimum: enumerate every encoding, keep the
    /// motif-free ones, and take the minimum BRD. Each motif-free sequence's MFE
    /// is its true minimum free energy (folded via [`fold_single_path`], the λ=1
    /// exact-lattice fold of the fixed sequence), and its CAI is the cost-to-go
    /// log-weight the DFA path accumulates. Returns the minimum BRD in
    /// dekacalorie units, or `None` when no encoding is motif-free.
    fn brute_force_brd(protein: &str, motifs: &[&str], lambda: f64) -> Option<f64> {
        let params = Params::load_default().unwrap();
        let (dfa, ac) = build(protein, motifs);
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        let mut best: Option<f64> = None;
        for (cai, seq) in dfa.paths_between(dfa.start(), dfa.end()) {
            if !is_motif_free(&ac, &seq) {
                continue;
            }
            let codons: Vec<[u8; 3]> = seq
                .chunks_exact(3)
                .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
                .collect();
            let mfe_dekacal = fold_single_path(&params, &aa, &codons).mfe_kcal_mol * 100.0;
            let brd = lambda * mfe_dekacal - (1.0 - lambda) * cai * 100.0;
            best = Some(best.map_or(brd, |b: f64| b.min(brd)));
        }
        best
    }

    /// Whether running the automaton over `seq` never enters an accepting state.
    fn is_motif_free(ac: &Ac, seq: &[Nucleotide]) -> bool {
        let mut q = ac.start();
        for &x in seq {
            q = ac.step(q, x);
            if ac.is_accept(q) {
                return false;
            }
        }
        true
    }

    /// Whether `seq` (ASCII `A/C/G/U`) is a valid coding sequence for `protein`.
    fn translates(seq: &str, protein: &str) -> bool {
        use derna_codon_dp::codon_table::AMINO_ACIDS;
        let b = seq.as_bytes();
        if b.len() != protein.len() * 3 {
            return false;
        }
        let code = |c: u8| match c {
            b'A' => 0u8,
            b'C' => 1,
            b'G' => 2,
            _ => 3,
        };
        protein.chars().enumerate().all(|(i, ch)| {
            let aa = aa_to_index(ch).unwrap();
            let trip = [code(b[3 * i]), code(b[3 * i + 1]), code(b[3 * i + 2])];
            AMINO_ACIDS[aa].codons.iter().any(|c| c.nuc == trip)
        })
    }

    /// The folder's grammar at the single-branch cap, against the reference
    /// minimizer: degree-2 loops of exactly `SINGLE_MAX_LEN` unpaired at several
    /// splits, and branching loops whose leading run reaches and passes it (the
    /// run is unbounded in the model; [`Folder::far_multi`] wraps the long
    /// ones). Each sequence is a fixed lattice under a motif it never spells, so
    /// the folder's minimum is the sequence's own.
    #[test]
    fn the_folder_reaches_the_reference_minimizer_at_the_loop_cap() {
        const CORE: &str = "GGGCGCGAAAGCGCCC";
        let params = Params::load_default().unwrap();
        let ac = Ac::build(&["UUUUUUUUUU"]).expect("valid motif");
        let mut seqs = Vec::new();
        for n1 in [0, 2, 15, 28, 30] {
            let n2 = SINGLE_MAX_LEN as usize - n1;
            seqs.push(format!(
                "GGGGG{}{CORE}{}CCCCC",
                "A".repeat(n1),
                "A".repeat(n2)
            ));
        }
        for lead in [29, 30, 31, 40] {
            seqs.push(format!("GGGGG{}{CORE}AA{CORE}AACCCCC", "A".repeat(lead)));
        }
        for s in seqs {
            let nts: Vec<Nucleotide> = s
                .bytes()
                .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
                .collect();
            let dfa = dfa::build_fixed_mrna(&nts);
            let product = ProductDfa::build(&dfa, &ac);
            let got = fold(&product, &params, 0.5, PairConstraint::inert()).expect("feasible");
            let (_, want) = rna_mfe::fold(&params, s.as_bytes()).expect("reference folds");
            assert_eq!(
                got.mfe.0, want.0,
                "{s}: folder {} reference {}",
                got.mfe.0, want.0
            );
        }
    }

    /// The spelled constrained fold must be a valid, motif-free, phantom-free
    /// design that achieves the brute-force optimal BRD: end-to-end, the same
    /// matrix as the value oracle. Where the value oracle pins the BRD, this pins
    /// the recovered sequence and structure — the traceback follows the folder.
    #[test]
    fn constrained_fold_is_valid_and_optimal() {
        let params = Params::load_default().unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("KF", &["GG"]),
            ("KFK", &["GUU"]),
            ("MFF", &["UUC"]),
            ("DFK", &["GN"]),
            ("KFKF", &["GUU", "AAG"]),
            ("MWF", &["CC"]),
        ];
        for &(protein, motifs) in cases {
            for &lambda in &[0.0001, 0.5, 0.99] {
                let (dfa, ac) = build(protein, motifs);
                let product = ProductDfa::build(&dfa, &ac);
                let want = brute_force_brd(protein, motifs, lambda);
                let got = fold(&product, &params, lambda, PairConstraint::inert());
                let got_feasible = got.is_some();
                match (want, got) {
                    (Some(w), Some(f)) => {
                        let seq: Vec<Nucleotide> = f
                            .sequence
                            .bytes()
                            .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
                            .collect();
                        assert!(
                            is_motif_free(&ac, &seq),
                            "spelled a forbidden motif: {protein} × {motifs:?} λ={lambda}: {}",
                            f.sequence,
                        );
                        assert!(
                            translates(&f.sequence, protein),
                            "spelled a non-coding sequence: {protein} × {motifs:?} λ={lambda}: {}",
                            f.sequence,
                        );
                        let reeval = evaluate(&params, f.sequence.as_bytes(), &f.structure)
                            .expect("recovered structure scores")
                            .dekacal() as f64
                            / 100.0;
                        assert!(
                            (reeval - f.mfe.to_kcal_mol()).abs() <= 0.005,
                            "phantom energy: {protein} × {motifs:?} λ={lambda}: \
                             reported={:.2} reeval={reeval:.2} structure={}",
                            f.mfe.to_kcal_mol(),
                            f.structure,
                        );
                        let got_brd =
                            lambda * f64::from(f.mfe.0) - (1.0 - lambda) * (f.cai * 100.0);
                        assert!(
                            (w - got_brd).abs() < 1e-6,
                            "constrained fold sub-optimal: {protein} × {motifs:?} λ={lambda}: \
                             brute={w} fold={got_brd}",
                        );
                    }
                    (None, None) => {}
                    _ => panic!(
                        "feasibility mismatch: {protein} × {motifs:?} λ={lambda}: \
                         brute={want:?} fold feasible={got_feasible}",
                    ),
                }
            }
        }
    }

    /// Internal loops on free flanks scored by their separated corners
    /// ([`Folder::separated_internal`]), and the skewed ones among them taken
    /// from their size diagonals ([`Folder::pull_skewed`]), leave every closed
    /// pair's score as the corner-pair enumeration has it, on one worker and
    /// on several, and the optimum's design valid: motif-free, coding, and
    /// scoring what the fold reports. On these cases both fire, and without
    /// free runs neither does.
    #[test]
    fn separated_corners_on_free_flanks_keep_every_closed_pair() {
        let params = Params::load_default().unwrap();
        let cases: &[(&str, &[&str])] = &[
            (
                "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDK",
                &["GAAUUC", "GGAUCC"],
            ),
            (
                "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDK",
                &["GGGGG", "CCCCC", "GAAUUC", "GGAUCC"],
            ),
            ("NWWFQWMMWHWMCMMFMQDQNWWFQWMMWHWMCMMFMQDQ", &["AAU"]),
        ];
        let (mut fired, mut pulled) = (0, 0);
        for &(protein, motifs) in cases {
            let (dfa, ac) = build(protein, motifs);
            let product = ProductDfa::build(&dfa, &ac).merge_equivalent();
            assert!(product.is_feasible(), "{protein} must be feasible");
            for &lambda in &[0.05, 0.5, 0.95] {
                let fold = |separate: bool, diagonal: bool, threads: usize| {
                    let mut f = Folder::new(
                        &product,
                        &params,
                        lambda,
                        Beam::exact(),
                        PairConstraint::inert(),
                        MustPairConstraint::inert(),
                        threads,
                    );
                    f.free_run = if separate {
                        free_runs(&product)
                    } else {
                        Vec::new()
                    };
                    if diagonal {
                        f.enable_diagonal();
                    }
                    let end_q = f.run().expect("feasible");
                    (f, end_q)
                };
                let (off, _) = fold(false, false, 1);
                assert_eq!(off.separated.load(Ordering::Relaxed), 0);
                for (diagonal, threads) in [(false, 1), (false, 3), (true, 1), (true, 3)] {
                    let (mut on, end_q) = fold(true, diagonal, threads);
                    fired += on.separated.load(Ordering::Relaxed);
                    pulled += on.pulled.load(Ordering::Relaxed);
                    assert_eq!(diagonal, on.lz);
                    for (a, b) in off.p.iter().zip(&on.p) {
                        assert_eq!(
                            a.len(),
                            b.len(),
                            "{protein} λ={lambda}: closed pairs differ"
                        );
                        for (key, ca) in a {
                            let cb = b.get(key).expect("the same closed pairs");
                            let (sa, sb) =
                                (brd(ca.mfe, ca.cai, lambda), brd(cb.mfe, cb.cai, lambda));
                            assert!(
                                (sa - sb).abs() < 1e-9,
                                "{protein} × {motifs:?} λ={lambda} threads={threads} \
                                 diagonal={diagonal}: a closed pair scores {sb}, {sa} enumerated",
                            );
                        }
                    }
                    let cell = on.c[dfa.ix(on.dfa.end())][&end_q];
                    let (sequence, structure) = on.traceback(end_q);
                    let seq: Vec<Nucleotide> = sequence
                        .bytes()
                        .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
                        .collect();
                    assert!(is_motif_free(&ac, &seq), "{protein}: spelled a motif");
                    assert!(translates(&sequence, protein), "{protein}: does not code");
                    let reeval = evaluate(&params, sequence.as_bytes(), &structure)
                        .expect("recovered structure scores")
                        .dekacal();
                    assert_eq!(reeval, cell.mfe.0, "{protein}: phantom energy");
                }
            }
        }
        assert!(fired > 0, "no internal loop had both flanks free");
        assert!(pulled > 0, "no skewed loop was taken from a diagonal");
    }

    /// Byte-exactness of the threaded fold: `threads ∈ {2, 4, 8}` must reproduce
    /// `threads = 1` exactly — sequence, structure, MFE *and* CAI — not merely
    /// the objective. A protein long enough to force a genuine multibranch
    /// junction under a several-pattern motif set gives both fat phases (and,
    /// under the junction, a real `M2` bifurcation) more than [`PARALLEL_MIN`]
    /// closed pairs at its busiest columns, so this exercises
    /// [`Folder::fan_out`]'s actual `std::thread::scope` path rather than only
    /// its small-input serial fallback.
    #[test]
    fn parallel_fan_out_matches_single_thread() {
        let params = Params::load_default().unwrap();
        let protein = &format!("{MULTILOOP_PROTEIN}{MULTILOOP_PROTEIN}");
        let motifs: &[&str] = &["AAU"];
        let (dfa, ac) = build(protein, motifs);
        let product = ProductDfa::build(&dfa, &ac);
        assert!(product.is_feasible(), "test case must be feasible");
        for &lambda in &[0.0001, 0.99] {
            let want = fold_with_beam(
                &product,
                &params,
                lambda,
                Beam::exact(),
                PairConstraint::inert(),
                MustPairConstraint::inert(),
                1,
            )
            .expect("feasible at threads = 1");
            for &threads in &[2usize, 4, 8] {
                let got = fold_with_beam(
                    &product,
                    &params,
                    lambda,
                    Beam::exact(),
                    PairConstraint::inert(),
                    MustPairConstraint::inert(),
                    threads,
                )
                .expect("feasible at threads > 1");
                assert_eq!(
                    got.mfe.0, want.mfe.0,
                    "MFE diverged at threads={threads} λ={lambda}",
                );
                assert_eq!(
                    got.cai, want.cai,
                    "CAI diverged at threads={threads} λ={lambda}",
                );
                assert_eq!(
                    got.sequence, want.sequence,
                    "sequence diverged at threads={threads} λ={lambda}",
                );
                assert_eq!(
                    got.structure, want.structure,
                    "structure diverged at threads={threads} λ={lambda}",
                );
            }
        }
    }

    /// The multi-loop's 3′ unpaired flank, which the fold grows one nucleotide
    /// at a time in `M2R`, checked against the unconstrained scan — separate
    /// code, with its own multi-loop recurrences.
    ///
    /// A motif longer than the molecule can never be spelled, so the product
    /// carries no constraint and the constrained optimum must equal the
    /// unconstrained one. The cases were chosen by measurement: each one's
    /// unconstrained optimum closes a multi-loop with a **non-empty** 3′ flank
    /// (6 and 3 unpaired), so the flank extension is on the optimum's
    /// derivation and a fold that dropped or mis-threaded it would miss.
    /// [`MULTILOOP_PROTEIN`]'s junction has an empty 3′ flank and cannot
    /// discriminate this.
    #[test]
    fn multiloop_three_prime_flank_matches_unconstrained_scan() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let cases: &[(&str, f64)] = &[
            ("AYMEGMVESKPPRRGYEVSGASWVKPM", 0.5),
            ("QHWWPNLIMYTSDFYNLKELVKRYEHVKWPRKWDCQDT", 0.99),
        ];
        for &(protein, lambda) in cases {
            let never = "A".repeat(3 * protein.len() + 1);
            let (dfa, ac) = build(protein, &[never.as_str()]);
            let product = ProductDfa::build(&dfa, &ac).merge_equivalent();
            let got = fold(&product, &params, lambda, PairConstraint::inert()).expect("feasible");
            let want = crate::design(&params, protein, &usage, lambda).expect("designs");
            assert!(
                has_three_prime_multiloop_flank(&want.structure),
                "{protein}: the case no longer exercises a 3′ flank: {}",
                want.structure,
            );
            assert!(
                (got.mfe.to_kcal_mol() - want.mfe_kcal_mol).abs() < 1e-9,
                "{protein} λ={lambda}: MFE {} against the scan's {}",
                got.mfe.to_kcal_mol(),
                want.mfe_kcal_mol,
            );
            assert!(
                (got.cai - want.cai_log).abs() < 1e-9,
                "{protein} λ={lambda}: log-CAI {} against the scan's {}",
                got.cai,
                want.cai_log,
            );
            assert!(translates(&got.sequence, protein), "{protein}: non-coding");
            let reeval = evaluate(&params, got.sequence.as_bytes(), &got.structure)
                .expect("recovered structure scores")
                .dekacal();
            assert_eq!(reeval, got.mfe.0, "{protein}: phantom energy");
        }
    }

    /// The 3′ flank threads the automaton: a motif cut from the unconstrained
    /// optimum's own 3′ multi-loop flank must not reappear in the constrained
    /// design. The motif is spelled by the scan's optimum, so the constraint
    /// binds on the optimum's derivation; a flank extension that skipped the AC
    /// step would let the design spell it again, which the motif scan catches.
    #[test]
    fn multiloop_three_prime_flank_threads_the_automaton() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let (protein, lambda) = ("AYMEGMVESKPPRRGYEVSGASWVKPM", 0.5);
        let free = crate::design(&params, protein, &usage, lambda).expect("designs");
        let (start, len) = three_prime_multiloop_flank(&free.structure).expect("a 3′ flank");
        let motif = &free.sequence[start..start + len.min(6)];
        let (dfa, ac) = build(protein, &[motif]);
        let product = ProductDfa::build(&dfa, &ac).merge_equivalent();
        let got = fold(&product, &params, lambda, PairConstraint::inert()).expect("feasible");
        let seq: Vec<Nucleotide> = got
            .sequence
            .bytes()
            .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
            .collect();
        assert!(
            is_motif_free(&ac, &seq),
            "spelled {motif}: {} {}",
            got.sequence,
            got.structure,
        );
        assert!(translates(&got.sequence, protein), "non-coding");
        let reeval = evaluate(&params, got.sequence.as_bytes(), &got.structure)
            .expect("recovered structure scores")
            .dekacal();
        assert_eq!(reeval, got.mfe.0, "phantom energy");
        let brd =
            |mfe_kcal: f64, cai: f64| lambda * mfe_kcal * 100.0 - (1.0 - lambda) * cai * 100.0;
        assert!(
            brd(got.mfe.to_kcal_mol(), got.cai) > brd(free.mfe_kcal_mol, free.cai_log) - 1e-9,
            "the constrained optimum beat the unconstrained one",
        );
    }

    /// The first multi-loop 3′ flank in `structure` with at least one unpaired
    /// nucleotide, as `(start, length)`.
    fn three_prime_multiloop_flank(structure: &str) -> Option<(usize, usize)> {
        let b = structure.as_bytes();
        let partner = partners(b);
        (0..b.len()).filter(|&i| b[i] == b'(').find_map(|i| {
            let (mut k, mut branches, mut last_end) = (i + 1, 0, i);
            while k < partner[i] {
                if b[k] == b'(' {
                    branches += 1;
                    last_end = partner[k];
                    k = partner[k] + 1;
                } else {
                    k += 1;
                }
            }
            (branches >= 2 && partner[i] > last_end + 1)
                .then(|| (last_end + 1, partner[i] - last_end - 1))
        })
    }

    /// The pairing partner of each paired position in a dot-bracket string.
    fn partners(b: &[u8]) -> Vec<usize> {
        let mut partner = vec![usize::MAX; b.len()];
        let mut stack = Vec::new();
        for (i, &c) in b.iter().enumerate() {
            match c {
                b'(' => stack.push(i),
                b')' => {
                    let j = stack.pop().expect("balanced");
                    partner[i] = j;
                    partner[j] = i;
                }
                _ => {}
            }
        }
        partner
    }

    /// Whether some multi-loop in `structure` leaves at least one unpaired
    /// nucleotide between its last branch and its closing base.
    fn has_three_prime_multiloop_flank(structure: &str) -> bool {
        three_prime_multiloop_flank(structure).is_some()
    }

    /// The longest unpaired run between a multi-loop's closing base and its
    /// first branch, over every multi-loop in `structure`.
    fn longest_multiloop_leading_run(structure: &str) -> Option<usize> {
        let b = structure.as_bytes();
        let partner = partners(b);
        (0..b.len())
            .filter(|&i| b[i] == b'(')
            .filter_map(|i| {
                let (mut k, mut branches, mut first) = (i + 1, 0, None);
                while k < partner[i] {
                    if b[k] == b'(' {
                        branches += 1;
                        first.get_or_insert(k);
                        k = partner[k] + 1;
                    } else {
                        k += 1;
                    }
                }
                (branches >= 2).then(|| first.expect("a branch") - i - 1)
            })
            .max()
    }

    /// A multi-loop whose 5′ leading run is longer than [`SINGLE_MAX_LEN`] —
    /// the wraps only [`Folder::far_multi`] builds — on the optimum of a real
    /// codon lattice under a binding motif set, at every thread count.
    ///
    /// Forbidding `U` leaves only G–C pairs, and forbidding `AAG` pins every
    /// Lys to `AAA`, so the Lys run is 33 nucleotides that cannot pair and
    /// the optimum carries them as a 34-unpaired leading run inside a
    /// three-way junction. Both motifs bind: without `AAG` the optimum has no
    /// multi-loop. The case was chosen by measurement: with the far sweep
    /// removed the fold misses this design at every λ checked (BRD −4185.6 →
    /// −4090.2 at λ = 0.5) and returns a structure with no long leading run,
    /// and at `threads = 4` the fold takes the pooled path 59 times, so the
    /// thread counts below are not all the serial fallback.
    #[test]
    fn long_leading_run_multiloop_survives_motifs_and_threads() {
        let params = Params::load_default().unwrap();
        let protein = "GGGGKKKKKKKKKKKRGGRERPPRERRGGRERPPRPPPP";
        let motifs: &[&str] = &["U", "AAG"];
        let (dfa, ac) = build(protein, motifs);
        let product = ProductDfa::build(&dfa, &ac).merge_equivalent();
        for &lambda in &[0.5, 0.99] {
            let fold = |threads| {
                fold_with_beam(
                    &product,
                    &params,
                    lambda,
                    Beam::exact(),
                    PairConstraint::inert(),
                    MustPairConstraint::inert(),
                    threads,
                )
                .expect("feasible")
            };
            let want = fold(1);
            let lead = longest_multiloop_leading_run(&want.structure);
            assert!(
                lead.is_some_and(|run| run > SINGLE_MAX_LEN as usize),
                "λ={lambda}: the optimum no longer carries a long leading run \
                 ({lead:?}): {}",
                want.structure,
            );
            assert!(
                translates(&want.sequence, protein),
                "λ={lambda}: non-coding"
            );
            let seq: Vec<Nucleotide> = want
                .sequence
                .bytes()
                .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
                .collect();
            assert!(is_motif_free(&ac, &seq), "λ={lambda}: spelled a motif");
            let reeval = evaluate(&params, want.sequence.as_bytes(), &want.structure)
                .expect("recovered structure scores")
                .dekacal();
            assert_eq!(reeval, want.mfe.0, "λ={lambda}: phantom energy");
            let (_, reference) =
                rna_mfe::fold(&params, want.sequence.as_bytes()).expect("reference folds");
            assert_eq!(
                want.mfe.0, reference.0,
                "λ={lambda}: the design's structure is not its own minimum",
            );
            for threads in [2usize, 4, 8] {
                let got = fold(threads);
                assert_eq!(
                    (got.mfe.0, got.cai, &got.sequence, &got.structure),
                    (want.mfe.0, want.cai, &want.sequence, &want.structure),
                    "λ={lambda}: threads={threads} diverged from threads=1",
                );
            }
        }
    }

    /// The threaded push reduces each chunk's candidates before they reach the
    /// table, with the region constraint's soft penalty applied first; a soft
    /// window therefore has to reach the same cells on every thread count. The
    /// window is checked to change the answer against an inert constraint, so the
    /// penalized path is on the optimum's derivation rather than idle.
    #[test]
    fn soft_window_threads_match_single_thread() {
        use crate::region::{AccessibilityWindow, Enforcement, RegionConstraints};
        let params = Params::load_default().unwrap();
        let protein = &format!("{MULTILOOP_PROTEIN}{MULTILOOP_PROTEIN}");
        let (dfa, ac) = build(protein, &["AAU"]);
        let product = ProductDfa::build(&dfa, &ac).merge_equivalent();
        let soft = RegionConstraints::default().with_window(AccessibilityWindow {
            upstream: 0,
            downstream: 24,
            enforcement: Enforcement::Penalize(Energy(150)),
        });
        let lambda = 0.99;
        let fold = |region: PairConstraint, threads| {
            fold_with_beam(
                &product,
                &params,
                lambda,
                Beam::exact(),
                region,
                MustPairConstraint::inert(),
                threads,
            )
            .expect("feasible")
        };
        let inert = fold(PairConstraint::inert(), 1);
        let want = fold(PairConstraint::new(&soft, &dfa), 1);
        assert_ne!(
            (want.mfe.0, &want.structure),
            (inert.mfe.0, &inert.structure),
            "the soft window changed nothing; the case does not exercise it",
        );
        for threads in [2usize, 4, 8] {
            let got = fold(PairConstraint::new(&soft, &dfa), threads);
            assert_eq!(got.mfe.0, want.mfe.0, "MFE diverged at threads={threads}");
            assert_eq!(got.cai, want.cai, "CAI diverged at threads={threads}");
            assert_eq!(
                got.sequence, want.sequence,
                "sequence diverged at threads={threads}"
            );
            assert_eq!(
                got.structure, want.structure,
                "structure diverged at threads={threads}"
            );
        }
    }

    /// The threaded push reads the must-pair constraint through the closing window
    /// and the opening floor; a constraint that binds has to give the
    /// single-threaded bytes on every thread count. The required positions are
    /// ones the unconstrained optimum leaves unpaired, so the constraint changes it.
    #[test]
    fn must_pair_threads_match_single_thread() {
        let params = Params::load_default().unwrap();
        let protein = &format!("{MULTILOOP_PROTEIN}{MULTILOOP_PROTEIN}");
        let (dfa, ac) = build(protein, &["AAU"]);
        let product = ProductDfa::build(&dfa, &ac).merge_equivalent();
        let lambda = 0.99;
        let fold = |must: MustPairConstraint, threads| {
            fold_with_beam(
                &product,
                &params,
                lambda,
                Beam::exact(),
                PairConstraint::inert(),
                must,
                threads,
            )
        };
        let free = fold(MustPairConstraint::inert(), 1).expect("feasible");
        let unpaired: Vec<u32> = free
            .structure
            .bytes()
            .enumerate()
            .filter(|&(_, c)| c == b'.')
            .map(|(i, _)| i as u32)
            .collect();
        let required: Vec<u32> = unpaired.iter().step_by(7).copied().take(4).collect();
        let constraint = || MustPairConstraint::new(&required, dfa.seq_len);
        let want = fold(constraint(), 1).expect("feasible under the constraint");
        assert_ne!(
            want.structure, free.structure,
            "the constraint changed nothing"
        );
        for threads in [2usize, 4, 8] {
            let got = fold(constraint(), threads).expect("feasible");
            assert_eq!(got.mfe.0, want.mfe.0, "MFE diverged at threads={threads}");
            assert_eq!(got.cai, want.cai, "CAI diverged at threads={threads}");
            assert_eq!(
                got.sequence, want.sequence,
                "sequence diverged at threads={threads}"
            );
            assert_eq!(
                got.structure, want.structure,
                "structure diverged at threads={threads}"
            );
        }
    }

    /// A protein forcing a genuine 3-way multibranch junction — shared with the
    /// blacklist-fuzz oracle's dedicated multiloop case, so both suites exercise
    /// the same traceback shape.
    const MULTILOOP_PROTEIN: &str = "NWWFQWMMWHWMCMMFMQDQ";

    /// The memory measurement: the materialized `P` / `Multi` two-boundary
    /// span-cell counts against `|A_M|`. Recorded (not gated) so the layout story
    /// is measured on the filled tables; the assertion only constraints that the fold
    /// materialized cells and stayed within the nominal `|nodes| · |A_M|` bound.
    #[test]
    fn d2_materialized_span_cell_counts() {
        let params = Params::load_default().unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("KFKF", &["GUU"]),
            ("MKTAYIAKQR", &["ACGACG"]),
            ("DEYWCNQHKR", &["GAU", "CAG"]),
        ];
        for &(protein, motifs) in cases {
            let (dfa, ac) = build(protein, motifs);
            let product = ProductDfa::build(&dfa, &ac);
            let counts = span_cell_counts(&product, &params, 0.5, PairConstraint::inert())
                .expect("feasible");
            let n_nodes = dfa.slots();
            assert!(counts.p_cells > 0, "no P cells materialized for {protein}");
            // The nominal dense `P` bound is `|nodes|² · |A_M|²` (two boundaries,
            // each node × state); the sparse fold stays far under it.
            let nominal = n_nodes * n_nodes * counts.ac_states * counts.ac_states;
            assert!(counts.p_cells <= nominal && counts.multi_cells <= nominal);
            println!(
                "layout {protein} × {motifs:?}: |A_M|={} product_nodes={} \
                 P_cells={} Multi_cells={}",
                counts.ac_states, counts.product_nodes, counts.p_cells, counts.multi_cells,
            );
        }
    }

    /// The constrained CKY optimum must match the brute-force minimum BRD across
    /// a matrix of tiny proteins, forbidden-motif sets that leave the protein
    /// encodable, and λ values. Motifs are chosen to exclude some (but not all)
    /// encodings — a straddle, an interior motif, and an IUPAC code — so the
    /// constraint genuinely bites.
    #[test]
    fn constrained_optimum_matches_brute_force() {
        let params = Params::load_default().unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("KF", &["GG"]),
            ("KFK", &["GUU"]),
            ("MFF", &["UUC"]),
            ("DFK", &["GN"]),
            ("KFKF", &["GUU", "AAG"]),
            ("MWF", &["CC"]),
        ];
        for &(protein, motifs) in cases {
            for &lambda in &[0.0001, 0.5, 0.99] {
                let (dfa, ac) = build(protein, motifs);
                let product = ProductDfa::build(&dfa, &ac);
                let want = brute_force_brd(protein, motifs, lambda);
                let got = optimum(&product, &params, lambda, PairConstraint::inert())
                    .map(|o| lambda * f64::from(o.mfe.0) - (1.0 - lambda) * (o.cai * 100.0));
                match (want, got) {
                    (Some(w), Some(g)) => assert!(
                        (w - g).abs() < 1e-6,
                        "constrained optimum diverged: {protein} × {motifs:?} λ={lambda}: \
                         brute={w} folder={g}",
                    ),
                    (None, None) => {}
                    _ => panic!(
                        "feasibility mismatch: {protein} × {motifs:?} λ={lambda}: \
                         brute={want:?} folder={got:?}",
                    ),
                }
            }
        }
    }

    /// Folding over future-equivalence classes searches the same feasible set:
    /// the merged fold must reach the brute-force optimum, spell no motif, and
    /// agree with the unmerged fold.
    ///
    /// Cases are random small proteins with two or three motifs cut from their
    /// own encodings, plus IUPAC wildcards, so partial matches often die at
    /// codon boundaries and merging has states to remove. The test asserts that
    /// merging removed states on most cases, so it cannot pass on products the
    /// quotient left alone. A quotient that over-merges — every live state at a
    /// node made one class — fails it.
    #[test]
    fn merged_fold_is_optimal() {
        const ALPHABET: &[u8] = b"MWCDEFHKNQYAGIPTV";
        let params = Params::load_default().unwrap();
        let mut x: u64 = 0x5EED_2026_1003;
        let mut next = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as usize % n
        };
        let (mut cases, mut shrank) = (0, 0);
        for _ in 0..60 {
            let len = 3 + next(3);
            let protein: String = (0..len)
                .map(|_| ALPHABET[next(ALPHABET.len())] as char)
                .collect();
            let (dfa, _) = build(&protein, &["A"]);
            let encoding: Vec<u8> = (0..dfa.seq_len)
                .map(|pos| {
                    let edges = dfa.right(*dfa.nodes_at(pos).first().unwrap());
                    b"ACGU"[edges[next(edges.len())].nuc as usize]
                })
                .collect();
            let motifs: Vec<String> = (0..2 + next(2))
                .map(|_| {
                    let k = 2 + next(4);
                    let start = next(encoding.len() - k + 1);
                    let mut m = encoding[start..start + k].to_vec();
                    if next(3) == 0 {
                        m[next(k)] = b'N';
                    }
                    String::from_utf8(m).unwrap()
                })
                .collect();
            let motifs: Vec<&str> = motifs.iter().map(String::as_str).collect();
            let (dfa, ac) = build(&protein, &motifs);
            let plain = ProductDfa::build(&dfa, &ac);
            let merged = ProductDfa::build(&dfa, &ac).merge_equivalent();
            assert!(merged.is_merged(), "{protein}: DFA not deterministic");
            assert!(merged.num_product_nodes() <= plain.num_product_nodes());
            cases += 1;
            shrank += usize::from(merged.num_product_nodes() < plain.num_product_nodes());
            for &lambda in &[0.0001, 0.5, 0.99] {
                let ctx = format!("{protein} × {motifs:?} λ={lambda}");
                let want = brute_force_brd(&protein, &motifs, lambda);
                let unmerged = optimum(&plain, &params, lambda, PairConstraint::inert());
                let got = fold(&merged, &params, lambda, PairConstraint::inert());
                match (want, got, unmerged) {
                    (None, None, None) => {}
                    (Some(w), Some(f), Some(u)) => {
                        let seq: Vec<Nucleotide> = f
                            .sequence
                            .bytes()
                            .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
                            .collect();
                        assert!(is_motif_free(&ac, &seq), "{ctx}: spelled a motif");
                        assert!(translates(&f.sequence, &protein), "{ctx}: non-coding");
                        let brd = |mfe: Energy, cai: f64| {
                            lambda * f64::from(mfe.0) - (1.0 - lambda) * (cai * 100.0)
                        };
                        let got_brd = brd(f.mfe, f.cai);
                        assert!(
                            (w - got_brd).abs() < 1e-6,
                            "{ctx}: brute={w} merged={got_brd}"
                        );
                        assert!(
                            (brd(u.mfe, u.cai) - got_brd).abs() < 1e-6,
                            "{ctx}: unmerged differs"
                        );
                    }
                    (w, f, u) => panic!(
                        "{ctx}: feasibility differs: brute={} merged={} unmerged={}",
                        w.is_some(),
                        f.is_some(),
                        u.is_some()
                    ),
                }
            }
        }
        assert!(
            shrank * 2 > cases,
            "merging removed states on only {shrank} of {cases} cases"
        );
    }
}
