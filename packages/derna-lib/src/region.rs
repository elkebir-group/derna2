//! Region-pair structural constraints: one shared constraint over a base pair's two
//! endpoint positions.
//!
//! Enforcement is a **table over pair-classes**. Each nucleotide position lies in
//! one of three [`Region`]s — 5′ UTR, CDS, 3′ UTR — as a function of its index
//! alone, so a candidate pair falls in one of the six unordered [`PairClass`]es,
//! and the table says what to do with it. Alongside the table sits one
//! distinguished family:
//!
//! - **Start-proximal accessibility window** — enforce on every pair with an
//!   endpoint inside a window straddling the start codon, *regardless of the
//!   partner's region*, keeping that window unpaired (a hard MFE-structural proxy
//!   for ribosome accessibility). It is not a class because its members are
//!   defined by position rather than by which regions they join: a hairpin folding
//!   the start codon back into the CDS is in it, and in no crossing class.
//!
//! Every entry supports a **hard** variant (forbid the pair — no closed-pair cell
//! forms) and a **soft** variant (add a positive MFE penalty, discouraging but not
//! excluding the pair). Because a position's region is an index compare, the constraint
//! needs no DP state: it stays a per-cell `O(1)` lookup consulted at the scan's
//! single `P`-write choke point.
//!
//! Named tables cover the useful sets — [`RegionConstraints::no_clamp`] (the
//! shipped default: `5′↔CDS` and `5′UTR↔3′UTR`),
//! [`RegionConstraints::five_prime_isolation`] (`5′↔CDS` alone),
//! [`RegionConstraints::symmetric_isolation`] (both CDS boundaries) and
//! [`RegionConstraints::full_mutual_isolation`] (every crossing class) — and
//! [`RegionConstraints::with`] reaches any entry individually.
//!
//! A default-constructed [`RegionConstraints`] is **inert**: the constraint returns
//! [`Verdict::Allow`] for every pair, so the fold is byte-identical to the
//! unconstrained substrate.
//!
//! # What a hard constraint claims
//!
//! A hard family constrains the **designed structure**, not the molecule. The
//! design returned under hard isolation is the exact optimum among
//! (sequence, isolation-respecting structure) pairs — but the same sequence
//! folded free may well reinstate the forbidden pairs, since the energy model
//! that would form them was never consulted about them. Selecting a sequence
//! because it *admits* a good isolated structure is a weaker statement than
//! selecting one that *prefers* to fold that way.
//!
//! [`crate::free_fold_gap`] measures the difference for a finished design:
//! it folds the emitted sequence with no constraint and reports both the energy the
//! constraint cost and the constrained pairs the free fold puts back. A soft
//! family is a different problem statement rather than a fix for this — it
//! prices the pair instead of excluding it, so the returned energy carries the
//! penalty (see [`crate::free_fold::FreeFoldGap::emitted`] for the
//! penalty-free energy of the same structure).

use std::sync::Arc;

use rna_mfe::Energy;

use crate::dfa::Dfa;
use crate::structure::PairMask;

/// How a region-pair family enforces a violating pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enforcement {
    /// Forbid the pair outright — no closed-pair (`P`) cell materializes for it.
    Forbid,
    /// Add `weight` (integer dekacalories) to the pair's folding energy. A
    /// positive weight discourages the pair without excluding it; the penalty
    /// enters the MFE half of the objective exactly once per formed pair.
    Penalize(Energy),
}

/// The start-proximal accessibility window: the positions
/// `[start − upstream, start + downstream)` straddling the start codon —
/// `upstream` nucleotides into the 5′ UTR through `downstream` into the CDS. A
/// pair with either endpoint inside this window is enforced against, keeping the
/// window unpaired (hard) or penalizing structure over it (soft). `upstream` and
/// `downstream` are parameters; the placement (straddling the start) is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessibilityWindow {
    /// Nucleotides upstream of the start codon (into the 5′ UTR). Clamped at the
    /// 5′ end of the mRNA, so a window wider than the 5′ UTR simply starts at 0.
    pub upstream: u32,
    /// Nucleotides downstream of (and including) the start codon (into the CDS).
    pub downstream: u32,
    /// Hard forbid or a soft penalty.
    pub enforcement: Enforcement,
}

impl AccessibilityWindow {
    /// The resolved default: a hard-forbidden window `upstream` nt into the 5′
    /// UTR through `downstream` nt into the CDS.
    pub fn hard(upstream: u32, downstream: u32) -> Self {
        Self {
            upstream,
            downstream,
            enforcement: Enforcement::Forbid,
        }
    }
}

/// One of the three regions of an mRNA. A position's region is a function of its
/// index alone, which is what keeps the constraint free of DP state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Region {
    /// Before the CDS.
    Utr5,
    /// The coding sequence, `[cds_start, cds_end)`.
    Cds,
    /// At or after `cds_end`.
    Utr3,
}

/// The unordered pair of regions a base pair joins — the unit enforcement is
/// expressed over. Six classes: three intra-region and three crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PairClass {
    /// Both endpoints in the 5′ UTR.
    Utr5Utr5,
    /// One endpoint in the 5′ UTR, one in the CDS.
    Utr5Cds,
    /// One endpoint in each UTR, spanning the whole CDS.
    Utr5Utr3,
    /// Both endpoints in the CDS.
    CdsCds,
    /// One endpoint in the CDS, one in the 3′ UTR.
    CdsUtr3,
    /// Both endpoints in the 3′ UTR.
    Utr3Utr3,
}

impl PairClass {
    /// Every class, in declaration order.
    pub const ALL: [PairClass; 6] = [
        Self::Utr5Utr5,
        Self::Utr5Cds,
        Self::Utr5Utr3,
        Self::CdsCds,
        Self::CdsUtr3,
        Self::Utr3Utr3,
    ];

    /// The classes whose pairs span a region boundary. Forbidding all three is
    /// what makes the joint fold separable at `dangles = 0`.
    pub const CROSSING: [PairClass; 3] = [Self::Utr5Cds, Self::Utr5Utr3, Self::CdsUtr3];

    /// The class a pair joining `a` and `b` falls in. Order-independent.
    pub fn of(a: Region, b: Region) -> Self {
        match (a, b) {
            (Region::Utr5, Region::Utr5) => Self::Utr5Utr5,
            (Region::Cds, Region::Cds) => Self::CdsCds,
            (Region::Utr3, Region::Utr3) => Self::Utr3Utr3,
            (Region::Utr5, Region::Cds) | (Region::Cds, Region::Utr5) => Self::Utr5Cds,
            (Region::Utr5, Region::Utr3) | (Region::Utr3, Region::Utr5) => Self::Utr5Utr3,
            (Region::Cds, Region::Utr3) | (Region::Utr3, Region::Cds) => Self::CdsUtr3,
        }
    }

    /// Whether the class spans a region boundary.
    pub fn is_crossing(self) -> bool {
        Self::CROSSING.contains(&self)
    }

    /// The class's spelling on the command line and in every report.
    ///
    /// One spelling, shared by the flag that names a class and the header that
    /// echoes what ran, so a caller can read a table back in the vocabulary they
    /// wrote it in.
    pub fn cli_name(self) -> &'static str {
        match self {
            Self::Utr5Cds => "5-cds",
            Self::CdsUtr3 => "cds-3",
            Self::Utr5Utr3 => "5-3",
            Self::Utr5Utr5 => "5-5",
            Self::CdsCds => "cds-cds",
            Self::Utr3Utr3 => "3-3",
        }
    }

    /// The class a spelling names, or `None` for a name no class has.
    pub fn from_cli_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|&c| c.cli_name() == name)
    }

    /// The two regions the class joins, 5′ → 3′. Equal for an intra-region
    /// class.
    ///
    /// Naming a class names both endpoints of every pair in it, which is what
    /// lets one class list state both halves of a constraint — which region a
    /// sequence-space guarantee derives *from* and which it is enforced *over*.
    pub fn endpoints(self) -> (Region, Region) {
        match self {
            Self::Utr5Utr5 => (Region::Utr5, Region::Utr5),
            Self::Utr5Cds => (Region::Utr5, Region::Cds),
            Self::Utr5Utr3 => (Region::Utr5, Region::Utr3),
            Self::CdsCds => (Region::Cds, Region::Cds),
            Self::CdsUtr3 => (Region::Cds, Region::Utr3),
            Self::Utr3Utr3 => (Region::Utr3, Region::Utr3),
        }
    }
}

/// Why a class list could not be read.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ClassListError {
    /// A name no class has.
    #[error("unknown region class {0:?} (5-cds|cds-3|5-3|5-5|cds-cds|3-3)")]
    UnknownClass(String),
    /// An empty entry — a stray comma, or a list with nothing in it. Refused
    /// rather than read as "no class", because a list that names nothing and a
    /// list that was never written are different things and only one of them is
    /// a typo.
    #[error(
        "empty region class in {0:?} — name at least one of          5-cds|cds-3|5-3|5-5|cds-cds|3-3, or state no list at all for no constraint"
    )]
    Empty(String),
}

/// The classes a comma-separated list names, in class order and deduplicated.
///
/// Shared by everything that accepts a class list, so a table written in one
/// place and read in another cannot disagree about what a name means. The result
/// is a set: a list composed from parts is the union of them, and the table it
/// selects does not depend on the order or the repeats it was spelled with.
pub fn parse_class_list(list: &str) -> Result<Vec<PairClass>, ClassListError> {
    let mut out = Vec::new();
    for name in list.split(',') {
        let name = name.trim();
        if name.is_empty() {
            return Err(ClassListError::Empty(list.to_string()));
        }
        out.push(
            PairClass::from_cli_name(name)
                .ok_or_else(|| ClassListError::UnknownClass(name.to_string()))?,
        );
    }
    out.sort_by_key(|&c| c as usize);
    out.dedup();
    Ok(out)
}

/// Enforcement per pair-class, plus the initiation window — the region
/// constraints passed to [`crate::design_with_utr`].
///
/// A default-constructed value is **inert** (every class allowed, no window),
/// leaving the design byte-identical to the unconstrained fold. The table is
/// orthogonal to the beam/mode axis, and every entry is consulted through one
/// endpoint-position constraint.
///
/// Build one from a named preset, or entry by entry:
///
/// ```
/// use derna_lib::{AccessibilityWindow, Enforcement, PairClass, RegionConstraints};
///
/// // The shipped default: forbid 5′↔CDS and the 5′↔3′ flank clamp.
/// let default = RegionConstraints::no_clamp(Enforcement::Forbid);
///
/// // Anything else, entry by entry.
/// let custom = RegionConstraints::default()
///     .with(PairClass::Utr5Cds, Enforcement::Forbid)
///     .with(PairClass::CdsUtr3, Enforcement::Penalize(derna_lib::Energy(50)))
///     .with_window(AccessibilityWindow::hard(15, 15));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegionConstraints {
    /// Enforcement per class, indexed by [`PairClass`] declaration order.
    /// `None` allows the class.
    classes: [Option<Enforcement>; 6],
    /// Start-proximal accessibility window. `None` disables it.
    pub accessibility: Option<AccessibilityWindow>,
    /// Fixed nucleotides past the designed codons that region classification
    /// counts as coding sequence — 3 for a stop codon spelled at the head of
    /// the 3′ flank, 0 otherwise. It moves only the `cds-3` boundary the
    /// classes see; the codon lanes, CAI and the accessibility window still
    /// end at the last designed codon.
    pub cds_suffix: u32,
}

impl RegionConstraints {
    /// Forbid pairs joining the 5′ UTR to the CDS, and nothing else.
    ///
    /// This is 5′-*boundary* isolation, not initiation-region accessibility: it
    /// says no pair joins the 5′ UTR to the coding sequence, and says nothing
    /// about a start codon sequestered by a partner inside the CDS. It is the one
    /// region predicate whose *whole* free-fold set is measured attainable across
    /// the length ladder — a design under it is modular at every measured point —
    /// so a clean verdict under it is the expected outcome rather than a lucky
    /// one, and it remains the weakest table anyone should run.
    ///
    /// It is **not** the default, because it permits the 5′↔3′ clamp: a designed
    /// flank satisfies it by pairing the two UTRs to each other around the whole
    /// CDS. [`Self::no_clamp`] is the default and forbids that as well.
    pub fn five_prime_isolation(enforcement: Enforcement) -> Self {
        Self::default().with(PairClass::Utr5Cds, enforcement)
    }

    /// **The shipped default.** [`Self::five_prime_isolation`] plus a forbidden
    /// `5′UTR↔3′UTR` class, so the 5′ UTR pairs only within itself and the two
    /// flanks cannot clamp to each other around the coding sequence.
    ///
    /// The clamp is what this buys. Permitting `5′↔3′` lets a designed flank —
    /// which earns no CAI and is therefore priced by folding energy alone —
    /// satisfy isolation by pairing base-for-base to the opposite flank, a
    /// structure every modularity test calls clean because no table objected to it.
    /// Forbidding the class means no run emits one by default.
    ///
    /// **Two consequences a caller should know**, both measured rather than
    /// argued:
    ///
    /// - **A clean verdict is no longer automatic.** `5′↔CDS` alone is modular at
    ///   every measured point; adding `5′↔3′` is not, because a pair of *given*
    ///   flanks may be complementary to each other and no codon assignment can
    ///   spell such a pair away. [`crate::flank_precheck`] is live under this
    ///   table for exactly that reason, and reports those pairs before any design
    ///   work — under [`Self::five_prime_isolation`] it can only ever report
    ///   *not applicable*.
    /// - **On a designed flank the degeneracy relocates rather than closing.**
    ///   Denied the opposite flank, the 3′ side pairs into the CDS instead and
    ///   the flank stays almost fully paired. Pairing is closed by forbidding
    ///   every class with a flank endpoint; the flank's *composition* is priced
    ///   by [`crate::UtrProfile`] and by nothing else.
    pub fn no_clamp(enforcement: Enforcement) -> Self {
        Self::five_prime_isolation(enforcement).with(PairClass::Utr5Utr3, enforcement)
    }

    /// Forbid every pair with an endpoint in the 5′ UTR, so the 5′ UTR is
    /// wholly unpaired in the designed structure. [`Self::no_clamp`] plus the
    /// 5′ UTR's own pairs.
    ///
    /// A given 5′ UTR that folds on its own is penalized by nothing here, so
    /// its free fold may pair where the constraint forbids, and few designs, or
    /// none, need be modular under this table.
    pub fn five_prime_unpaired(enforcement: Enforcement) -> Self {
        Self::no_clamp(enforcement).with(PairClass::Utr5Utr5, enforcement)
    }

    /// Both CDS boundaries: forbid every pair with **exactly one** endpoint in
    /// the CDS. The symmetric CDS↔UTR isolation, reachable as an explicit
    /// opt-in.
    pub fn symmetric_isolation(enforcement: Enforcement) -> Self {
        Self::default()
            .with(PairClass::Utr5Cds, enforcement)
            .with(PairClass::CdsUtr3, enforcement)
    }

    /// Every crossing class, so no pair spans any region boundary. At
    /// `dangles = 0` this makes the joint fold **separable** —
    /// `E = E₅′ + E_CDS + E₃′` over disjoint variables — so the design reduces
    /// to CDS-only design plus independent flank folds and the joint fold does
    /// no work.
    pub fn full_mutual_isolation(enforcement: Enforcement) -> Self {
        PairClass::CROSSING
            .iter()
            .fold(Self::default(), |rc, &c| rc.with(c, enforcement))
    }

    /// The table enforcing exactly these classes and nothing else — what a
    /// class list names.
    pub fn enforcing(classes: &[PairClass], enforcement: Enforcement) -> Self {
        classes
            .iter()
            .fold(Self::default(), |rc, &class| rc.with(class, enforcement))
    }

    /// This table with `class` enforced by `enforcement`.
    pub fn with(mut self, class: PairClass, enforcement: Enforcement) -> Self {
        self.classes[class as usize] = Some(enforcement);
        self
    }

    /// This table with `class` allowed — the inverse of [`Self::with`], for
    /// carving one class out of a preset.
    pub fn allowing(mut self, class: PairClass) -> Self {
        self.classes[class as usize] = None;
        self
    }

    /// This table with the start-proximal accessibility window active.
    pub fn with_window(mut self, window: AccessibilityWindow) -> Self {
        self.accessibility = Some(window);
        self
    }

    /// This table with `nucleotides` of fixed sequence past the designed codons
    /// classified as coding sequence. See [`Self::cds_suffix`].
    pub fn with_cds_suffix(mut self, nucleotides: u32) -> Self {
        self.cds_suffix = nucleotides;
        self
    }

    /// How `class` is enforced, or `None` if it is allowed.
    pub fn enforcement(&self, class: PairClass) -> Option<Enforcement> {
        self.classes[class as usize]
    }

    /// Whether every class is allowed and no window is set — the constraint is a no-op
    /// and the fold is byte-identical to the unconstrained substrate.
    pub fn is_inert(&self) -> bool {
        self.classes.iter().all(Option::is_none) && self.accessibility.is_none()
    }

    /// Whether every constraint set here forbids rather than penalizes, so
    /// that a pair is either allowed or never formed. An inert table is hard.
    pub fn is_hard(&self) -> bool {
        let hard = |e: Option<Enforcement>| matches!(e, None | Some(Enforcement::Forbid));
        self.classes.iter().all(|&e| hard(e)) && hard(self.accessibility.map(|w| w.enforcement))
    }

    /// Whether this table forbids a pair `(i, j)` (absolute, 0-based) on a
    /// molecule whose CDS is `[cds_start, cds_end)`: the same verdict the scan
    /// applies, for callers that fold an already-spelled sequence outside it.
    /// A penalized pair is not forbidden.
    pub fn pair_forbidden(&self, cds_start: u32, cds_end: u32) -> impl Fn(u32, u32) -> bool + Sync {
        let constraint = PairConstraint::for_cds(self, cds_start, cds_end);
        move |i, j| constraint.verdict(i.min(j), i.max(j)) == Verdict::Forbid
    }
}

/// The verdict for one candidate pair at the `P`-write choke point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Keep the candidate unchanged.
    Allow,
    /// Drop the candidate — the pair violates a hard constraint.
    Forbid,
    /// Keep the candidate with `Energy` added to its folding energy.
    Penalize(Energy),
}

/// A [`RegionConstraints`] resolved against a concrete DFA's CDS window,
/// together with the positional [`PairMask`]: the endpoint-position constraint the
/// scan consults at [`crate::tables::Tables::update_p`].
///
/// The two live in one constraint because they answer the same question at the same
/// choke point — *may this pair form?* — differing only in what the answer is
/// keyed on: a pair-class for the table, an absolute position for the mask. One
/// constraint is also what makes the positional half reach both folders, since the
/// AC-threaded constrained folder already consults this one.
///
/// Cheap `Copy` integers plus a refcounted mask, so it crosses the scan's scoped
/// worker threads freely. An inert constraint returns [`Verdict::Allow`] for every
/// pair.
#[derive(Debug, Clone)]
pub(crate) struct PairConstraint {
    /// First CDS position (`= |UTR₅′|`).
    cds_start: u32,
    /// First position past the CDS.
    cds_end: u32,
    /// Enforcement per class, indexed by [`PairClass`] declaration order.
    classes: [Option<Enforcement>; 6],
    /// Resolved absolute window `[lo, hi)` and its enforcement, when active.
    window: Option<(u32, u32, Enforcement)>,
    /// The positional half — `None` when no positional constraint was given, so
    /// the class-only paths pay nothing for it.
    mask: Option<Arc<PairMask>>,
    /// Fast path: skip the whole constraint when the table and the mask are inert.
    active: bool,
}

impl PairConstraint {
    /// The inert constraint — every pair allowed. Used by the CDS-only /
    /// no-constraint scan paths, keeping them byte-identical.
    pub(crate) fn inert() -> Self {
        Self {
            cds_start: 0,
            cds_end: 0,
            classes: [None; 6],
            window: None,
            mask: None,
            active: false,
        }
    }

    /// This constraint with `mask` also consulted at the choke point. An inert mask
    /// is dropped rather than stored, so a constraint that gains nothing from the
    /// positional axis stays byte-identical to the one without it.
    pub(crate) fn with_mask(mut self, mask: PairMask) -> Self {
        if mask.is_inert() {
            return self;
        }
        self.active = true;
        self.mask = Some(Arc::new(mask));
        self
    }

    /// Resolve `rc` against `dfa`'s CDS window. The window's `upstream` extent is
    /// clamped at position 0 (the 5′ end).
    pub(crate) fn new(rc: &RegionConstraints, dfa: &Dfa) -> Self {
        Self::for_cds(rc, dfa.cds_start(), dfa.cds_end())
    }

    /// Resolve `rc` against a CDS window `[cds_start, cds_end)` given directly,
    /// for callers that hold the window without a DFA — scoring an already-folded
    /// sequence, say. The window's `upstream` extent is clamped at position 0
    /// (the 5′ end). `cds_end` is the end of the designed codons; `rc`'s
    /// [`RegionConstraints::cds_suffix`] extends the classified CDS past it.
    pub(crate) fn for_cds(rc: &RegionConstraints, cds_start: u32, cds_end: u32) -> Self {
        let cds_end = cds_end + rc.cds_suffix;
        let window = rc.accessibility.map(|w| {
            let lo = cds_start.saturating_sub(w.upstream);
            let hi = cds_start + w.downstream;
            (lo, hi, w.enforcement)
        });
        Self {
            cds_start,
            cds_end,
            classes: rc.classes,
            window,
            mask: None,
            active: !rc.is_inert(),
        }
    }

    /// The region position `p` lies in.
    fn region(&self, p: u32) -> Region {
        if p < self.cds_start {
            Region::Utr5
        } else if p < self.cds_end {
            Region::Cds
        } else {
            Region::Utr3
        }
    }

    /// The class a candidate pair with endpoints at `i` and `j` falls in. Two
    /// index compares and no table lookup, so it is meaningful under an inert
    /// constraint too — which is what lets a report describe a pair distribution the
    /// table has nothing to say about.
    pub(crate) fn class_of(&self, i: u32, j: u32) -> PairClass {
        PairClass::of(self.region(i), self.region(j))
    }

    /// Whether either endpoint lies in the start-proximal window. False when no
    /// window is set.
    pub(crate) fn in_window(&self, i: u32, j: u32) -> bool {
        self.window.is_some_and(|(lo, hi, _)| {
            let inside = |p: u32| p >= lo && p < hi;
            inside(i) || inside(j)
        })
    }

    /// Whether a **hard** accessibility window forbids `pos` from pairing at
    /// all. The window is the one family keyed on a single endpoint, so it is
    /// the one that can contradict "this position must pair" without naming a
    /// partner.
    pub(crate) fn window_forbids(&self, pos: u32) -> bool {
        self.window
            .is_some_and(|(lo, hi, enf)| enf == Enforcement::Forbid && pos >= lo && pos < hi)
    }

    /// The verdict for a candidate pair with 5′ endpoint at position `i` and 3′
    /// endpoint at `j`. A hard forbid from the class table, the window or the
    /// positional mask dominates (the pair never forms, so any soft penalty is
    /// moot); otherwise soft penalties sum. An inert constraint always returns
    /// [`Verdict::Allow`].
    ///
    /// The positional half is hard-only by construction: a soft *forbid* is the
    /// class table's penalty, and D6 defers soft variants of the positional
    /// grammar entirely.
    pub(crate) fn verdict(&self, i: u32, j: u32) -> Verdict {
        if !self.active {
            return Verdict::Allow;
        }
        if self.mask.as_ref().is_some_and(|m| !m.allows(i, j)) {
            return Verdict::Forbid;
        }
        let mut penalty = 0i32;
        // The pair's class: two index compares and a table lookup.
        if let Some(enf) = self.classes[self.class_of(i, j) as usize] {
            match enf {
                Enforcement::Forbid => return Verdict::Forbid,
                Enforcement::Penalize(w) => penalty += w.0,
            }
        }
        // Accessibility window: either endpoint inside the window.
        if let Some((_, _, enf)) = self.window.filter(|_| self.in_window(i, j)) {
            match enf {
                Enforcement::Forbid => return Verdict::Forbid,
                Enforcement::Penalize(w) => penalty += w.0,
            }
        }
        if penalty != 0 {
            Verdict::Penalize(Energy(penalty))
        } else {
            Verdict::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    //! Constraint-predicate unit coverage: inertness, the isolation cross-region
    //! predicate, the window straddle, and hard-dominates-soft.

    use super::*;
    use derna_codon_dp::codon_table::aa_to_index;
    use derna_codon_dp::CodonUsage;
    use rna_mfe::Nucleotide;

    fn dfa_with_utr(protein: &str, utr5_len: usize, utr3_len: usize) -> Dfa {
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        let flank = vec![Nucleotide::A; utr5_len.max(utr3_len)];
        crate::dfa::build_with_utr(&aa, &usage, &flank[..utr5_len], &flank[..utr3_len])
    }

    #[test]
    fn inert_constraint_allows_everything() {
        let dfa = dfa_with_utr("MK", 4, 4);
        let constraint = PairConstraint::new(&RegionConstraints::default(), &dfa);
        for i in 0..dfa.seq_len {
            for j in (i + 1)..dfa.seq_len {
                assert_eq!(constraint.verdict(i, j), Verdict::Allow);
            }
        }
    }

    #[test]
    fn symmetric_isolation_forbids_exactly_cross_region_pairs() {
        // cds_start = 4, cds_end = 4 + 6 = 10.
        let dfa = dfa_with_utr("MK", 4, 4);
        let rc = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
        let constraint = PairConstraint::new(&rc, &dfa);
        // UTR (2) ↔ CDS (6): crosses → forbid.
        assert_eq!(constraint.verdict(2, 6), Verdict::Forbid);
        // CDS (5) ↔ CDS (9): both in CDS → allow.
        assert_eq!(constraint.verdict(5, 9), Verdict::Allow);
        // 5′ UTR (1) ↔ 3′ UTR (11): both outside CDS → allow.
        assert_eq!(constraint.verdict(1, 11), Verdict::Allow);
        // CDS (9) ↔ 3′ UTR (11): crosses → forbid.
        assert_eq!(constraint.verdict(9, 11), Verdict::Forbid);
    }

    #[test]
    fn five_prime_isolation_leaves_the_three_prime_boundary_alone() {
        let dfa = dfa_with_utr("MK", 4, 4);
        let rc = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
        let constraint = PairConstraint::new(&rc, &dfa);
        // 5′ UTR (2) ↔ CDS (6): the one forbidden class.
        assert_eq!(constraint.verdict(2, 6), Verdict::Forbid);
        // CDS (9) ↔ 3′ UTR (11): permitted, unlike under symmetric isolation.
        assert_eq!(constraint.verdict(9, 11), Verdict::Allow);
        // 5′ UTR (1) ↔ 3′ UTR (11): permitted.
        assert_eq!(constraint.verdict(1, 11), Verdict::Allow);
        // Intra-region pairs are untouched.
        assert_eq!(constraint.verdict(5, 9), Verdict::Allow);
    }

    #[test]
    fn a_zero_cds_suffix_classifies_every_pair_as_without_one() {
        let dfa = dfa_with_utr("MK", 4, 7);
        let rc = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
        let (plain, zero) = (
            PairConstraint::new(&rc, &dfa),
            PairConstraint::new(&rc.with_cds_suffix(0), &dfa),
        );
        for i in 0..dfa.seq_len {
            for j in (i + 1)..dfa.seq_len {
                assert_eq!(plain.class_of(i, j), zero.class_of(i, j), "({i}, {j})");
                assert_eq!(plain.verdict(i, j), zero.verdict(i, j), "({i}, {j})");
            }
        }
    }

    #[test]
    fn a_stop_codon_suffix_moves_the_three_prime_boundary_past_it() {
        // cds_start = 4, designed codons end at 10, the stop spans 10..13 at
        // the head of the 3′ flank, and the real 3′ UTR starts at 13.
        let dfa = dfa_with_utr("MK", 4, 7);
        let rc = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
        let (without, with) = (
            PairConstraint::new(&rc, &dfa),
            PairConstraint::new(&rc.with_cds_suffix(3), &dfa),
        );
        // A stop nucleotide paired to the 5′ UTR crosses the 5′ boundary once
        // the stop counts as coding sequence, and only then is it forbidden.
        assert_eq!(without.class_of(1, 11), PairClass::Utr5Utr3);
        assert_eq!(with.class_of(1, 11), PairClass::Utr5Cds);
        assert_eq!(without.verdict(1, 11), Verdict::Allow);
        assert_eq!(with.verdict(1, 11), Verdict::Forbid);
        // The stop is CDS to a codon and 3′-boundary-crossing to the real UTR.
        assert_eq!(without.class_of(8, 12), PairClass::CdsUtr3);
        assert_eq!(with.class_of(8, 12), PairClass::CdsCds);
        assert_eq!(without.class_of(11, 14), PairClass::Utr3Utr3);
        assert_eq!(with.class_of(11, 14), PairClass::CdsUtr3);
        // Past the stop nothing moves.
        assert_eq!(with.class_of(1, 13), PairClass::Utr5Utr3);
        assert_eq!(with.verdict(1, 13), Verdict::Allow);
    }

    #[test]
    fn full_mutual_isolation_forbids_every_crossing_class() {
        let dfa = dfa_with_utr("MK", 4, 4);
        let rc = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
        let constraint = PairConstraint::new(&rc, &dfa);
        assert_eq!(constraint.verdict(2, 6), Verdict::Forbid); // 5′↔CDS
        assert_eq!(constraint.verdict(9, 11), Verdict::Forbid); // CDS↔3′
        assert_eq!(constraint.verdict(1, 11), Verdict::Forbid); // 5′↔3′
                                                                // Every intra-region class stays allowed — that is what leaves the three
                                                                // regions free to fold independently.
        assert_eq!(constraint.verdict(0, 3), Verdict::Allow); // 5′↔5′
        assert_eq!(constraint.verdict(5, 9), Verdict::Allow); // CDS↔CDS
        assert_eq!(constraint.verdict(10, 13), Verdict::Allow); // 3′↔3′
    }

    #[test]
    fn each_class_is_independently_reachable() {
        let dfa = dfa_with_utr("MK", 4, 4);
        // One representative pair per class, at cds = [4, 10) over 14 positions.
        let sample = [
            (PairClass::Utr5Utr5, (0u32, 3u32)),
            (PairClass::Utr5Cds, (2, 6)),
            (PairClass::Utr5Utr3, (1, 11)),
            (PairClass::CdsCds, (5, 9)),
            (PairClass::CdsUtr3, (9, 11)),
            (PairClass::Utr3Utr3, (10, 13)),
        ];
        for (class, _) in sample {
            let rc = RegionConstraints::default().with(class, Enforcement::Forbid);
            let constraint = PairConstraint::new(&rc, &dfa);
            for (other, (oi, oj)) in sample {
                let expected = if other == class {
                    Verdict::Forbid
                } else {
                    Verdict::Allow
                };
                assert_eq!(
                    constraint.verdict(oi, oj),
                    expected,
                    "forbidding {class:?} gave the wrong verdict for {other:?}"
                );
            }
        }
    }

    #[test]
    fn pair_class_is_order_independent() {
        for a in [Region::Utr5, Region::Cds, Region::Utr3] {
            for b in [Region::Utr5, Region::Cds, Region::Utr3] {
                assert_eq!(PairClass::of(a, b), PairClass::of(b, a));
            }
        }
        assert!(PairClass::CROSSING.iter().all(|c| c.is_crossing()));
        assert_eq!(
            PairClass::ALL.iter().filter(|c| c.is_crossing()).count(),
            PairClass::CROSSING.len()
        );
    }

    #[test]
    fn window_straddles_start_and_hard_dominates_soft() {
        // cds_start = 4; window upstream 2, downstream 3 → [2, 7).
        let dfa = dfa_with_utr("MK", 4, 4);
        // Soft isolation so a hard window can dominate it on a shared pair.
        let rc = RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(50)))
            .with_window(AccessibilityWindow::hard(2, 3));
        let constraint = PairConstraint::new(&rc, &dfa);
        // Pair (3 in 5′UTR window, 6 in CDS window): window hard-forbids, and it
        // is also a cross-region soft pair — hard wins.
        assert_eq!(constraint.verdict(3, 6), Verdict::Forbid);
        // Pair (1 outside window, 8 in CDS outside window) but cross-region:
        // only soft isolation applies.
        assert_eq!(constraint.verdict(1, 8), Verdict::Penalize(Energy(50)));
        // Pair (7 immediately past window, 9): both CDS, neither in window → allow.
        assert_eq!(constraint.verdict(7, 9), Verdict::Allow);
    }
}
