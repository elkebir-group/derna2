//! Deriving a forbidden-motif set from the **given** regions of a molecule, so
//! that its **designed** regions cannot spell a duplex with them.
//!
//! An alphabet restriction reaches the case where both endpoints of the
//! prevented pairs are designed. It provably cannot reach the case where one
//! side is supplied: against a sequence containing all four nucleotides every
//! candidate nucleotide pairs with something present, so no subset survives.
//! What does reach it is a pattern-level constraint — forbid the *segments* that
//! could pair with the given sequence, leaving isolated pairs (which are
//! energetically irrelevant) legal.
//!
//! # Source and target
//!
//! The derivation is keyed on regions rather than on flank roles: **a source
//! must be given and a target must be designed.** A molecule has three regions —
//! 5′ UTR, coding sequence, 3′ UTR — and the coding sequence is designed by
//! construction, so which regions play which part follows from the flanks alone:
//!
//! | given | derives from | enforces over |
//! | --- | --- | --- |
//! | 5′ only | 5′ | CDS + 3′ |
//! | 3′ only | 3′ | 5′ + CDS |
//! | both | 5′ and 3′ | CDS |
//! | neither | — | *refused: nothing to derive from* |
//!
//! The one refusal is the alphabet-restriction case above — the `k = 1` limit,
//! where the derived set is one literal per nucleotide — and it is a refusal
//! because a derivation with no supplied source has nothing to be a complement
//! *of*.
//!
//! # Coverage is a class list
//!
//! The table above is the universal rule, and it is what a caller naming every
//! [`PairClass`] gets. A caller naming fewer restricts the derivation to those:
//! **a class names both of its endpoints**, so it says on its own which region a
//! set derives from and which it is enforced over — `5-cds` with the 5′ flank
//! given means *derive from the 5′ flank, enforce over the coding sequence*.
//! Coverage is therefore stated once, in the same vocabulary the structural
//! constraint is stated in, and cannot disagree with itself.
//!
//! A named class the derivation cannot reach — both endpoints given, or neither
//! given — contributes nothing and is not an error. The structural constraint is a
//! separate mechanism and the caller may have named the class for that alone;
//! what they need is a report of where the derivation *did* reach, which is
//! [`DerivedMotifs::sources`] and [`DerivedMotifs::targets`].
//!
//! The targets are always contiguous, which is what lets one enforcement window
//! cover them. That is a property of the class table rather than an assumption:
//! a flank becomes a target only through a class joining it to a **given**
//! region, so for both flanks to be targets at once each would have to be
//! designed while the other is given — and the span therefore never skips a
//! region it does not cover.
//!
//! # The derivation
//!
//! Pairing is antiparallel. A stacked run of `k` pairs joining a given window
//! `w₀…w_{k−1}` to a designed segment `d₀…d_{k−1}` pairs `w₀` with `d_{k−1}`,
//! `w₁` with `d_{k−2}`, and so on — so `d_s` is a partner of `w_{k−1−s}`, which
//! is to say `d` is the **reverse** complement of `w`. Reversing the wrong side
//! yields a set that is the same size, constrains the same positions, and is
//! wrong, which is why [`can_pair`] states the underlying relation separately
//! and the derivation is checked against it rather than read.
//!
//! The complement is degenerate under [`Complement::Wobble`], because `G` pairs
//! `C` *or* `U`. The map degenerates the **given** base (`G → Y`, `U → R`),
//! which reads as though it only caught a given `G` opposite a designed `U`. It
//! catches the mirror too: a given `U` maps to `R = {A, G}`, so a designed `G`
//! opposite a given `U` is forbidden as well. Enumerating over the given
//! sequence's bases is what makes the relation symmetric in effect —
//! [`can_pair`] is exactly the canonical pair relation, in both argument orders.
//!
//! Two sources contribute one set: the patterns are unioned and deduplicated, so
//! a segment forbidden on account of either flank is forbidden once.
//!
//! # What the derived set guarantees
//!
//! Applied over the designed regions, the set makes this true of every sequence
//! the design can spell:
//!
//! > **No ungapped, perfectly-paired duplex of length `≥ k` joins a given region
//! > to a designed one** — no run of `k` consecutive stacked pairs each of which
//! > has one end in given sequence and the other in designed positions.
//!
//! It is stronger than the enforcement window's own predicate needs it to be:
//! the designed side of such a run lies wholly inside the designed regions,
//! which is the window, so the entirely-inside reading catches it with nothing
//! straddling — including a run whose designed side crosses from the coding
//! sequence into a designed flank, since the two are one contiguous window.
//!
//! Three things it does **not** guarantee, each on purpose:
//!
//! - **A bulged or interior-looped helix is not forbidden.** Two runs of `k−1`
//!   pairs either side of a one-nucleotide bulge total `2k − 2` paired
//!   positions and contain no ungapped run of `k`. A `k` chosen against ungapped
//!   runs is a different `k` from one chosen against bulged ones.
//! - **Runs shorter than `k` stay legal**, which is the point: an isolated pair
//!   costs nothing energetically, and forbidding it would forbid most sequences.
//! - **Two designed regions pairing with each other are untouched.** The set is
//!   derived from given sequence alone, so a duplex with both ends among the
//!   designed positions — the coding sequence with itself, or a designed flank
//!   with the coding sequence — stays with the region table.

use std::collections::BTreeSet;

use rna_mfe::Nucleotide;

use crate::ac::{expand_motif, iupac_class, MotifWindow};
use crate::region::{PairClass, Region};
use crate::{BeamError, UtrSpec};

/// Which pairing relation a derived motif set covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Complement {
    /// `G` pairs with `C` **or** `U`, so the partner of a given base is a
    /// degenerate IUPAC class. The default, because a `G`-rich flank clamp is
    /// made of exactly the `G·U` helices the strict variant misses.
    #[default]
    Wobble,
    /// Watson–Crick only — one literal per window. Cheaper (a `k`-mer expands to
    /// a single pattern) and strictly weaker: it misses every `G·U` helix, so it
    /// is a measured control rather than a safe default.
    Strict,
}

impl Complement {
    /// The spelling used on the command line and in reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Wobble => "wobble",
            Self::Strict => "strict",
        }
    }
}

/// Derive the forbidden set from the **given** regions of a run: every segment
/// that could form an ungapped duplex of `k` pairs with one of their windows,
/// enforced over the run's **designed** regions.
///
/// The window is not a separate parameter. A derived set has source regions, and
/// enforcing it over one of them is the failure position scoping exists to fix —
/// the set is near-certain to forbid the sequence that generated it, so the
/// design comes back infeasible rather than expensive. Sources and targets are
/// disjoint by construction here, which is what makes that unreachable rather
/// than merely documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlankComplement {
    /// The duplex length forbidden: no ungapped run of `k` pairs joining given
    /// sequence to designed positions survives. Measured rather than assumed —
    /// the cost and the feasibility floor both move with it.
    pub k: usize,
    /// Which pairing relation the partner segments cover.
    pub complement: Complement,
}

impl FlankComplement {
    /// A wobble-inclusive derivation at duplex length `k`.
    pub fn wobble(k: usize) -> Self {
        Self {
            k,
            complement: Complement::Wobble,
        }
    }
}

/// A resolved derivation: the patterns, the regions at both ends of it, the
/// positions they are enforced over, and what the expansion cost.
///
/// Reported as well as consumed. The pattern count is nearly flat in `k` — it is
/// the number of deduplicated windows across the sources — so it is not where
/// the cost is; [`DerivedMotifs::literals`] is the number that sizes the
/// automaton, and neither is the number that sizes the fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedMotifs {
    /// The forbidden patterns, as IUPAC strings, sorted and deduplicated. One
    /// set across every source: a segment forbidden on account of either given
    /// region appears once.
    pub motifs: Vec<String>,
    /// The positions they are enforced over — the span of the targets, which is
    /// contiguous.
    pub window: MotifWindow,
    /// The given regions they were derived from, 5′ → 3′.
    pub sources: Vec<Region>,
    /// The designed regions the window covers, 5′ → 3′.
    pub targets: Vec<Region>,
    /// The region-pair classes the derivation actually acted across — those with
    /// one endpoint given and the other designed, among the classes it covered.
    ///
    /// Reported rather than recomputed by the caller, because a report that says
    /// which classes carry a sequence-space guarantee and which carry only the
    /// structural constraint is the whole of what makes skipping an unreachable class
    /// safe, and two derivations of the same fact drift.
    pub classes: Vec<PairClass>,
    /// How many nucleotides the sources hold in total.
    pub source_len: usize,
    /// Windows across the sources, before deduplication (`len − k + 1` per
    /// source, or zero for a source shorter than `k`).
    pub windows: usize,
    /// **Distinct** literal `ACGU` patterns the IUPAC set expands to — the count
    /// that sizes the automaton.
    pub literals: usize,
    /// The expansion summed per pattern (`Σ 2^(#G + #U)` under wobble), before
    /// distinct literals are merged. The naive cost estimate; the gap to
    /// [`DerivedMotifs::literals`] is how much the patterns overlap.
    pub expansion_sum: usize,
}

impl DerivedMotifs {
    /// Whether the derivation produced no patterns, which happens exactly when
    /// every source is shorter than `k`. Such a run is the run without the
    /// derivation, byte for byte.
    pub fn is_empty(&self) -> bool {
        self.motifs.is_empty()
    }
}

/// Whether a **given** base and a **designed** one can form a base pair.
///
/// The relation the derivation is derived from, stated on its own so it can be
/// checked rather than inferred from the pattern set. Under
/// [`Complement::Wobble`] this is exactly the canonical pair relation — `A·U`,
/// `G·C` and `G·U`, in either argument order — and under
/// [`Complement::Strict`] it is that relation minus the two `G·U` orientations.
pub fn can_pair(given: Nucleotide, designed: Nucleotide, complement: Complement) -> bool {
    iupac_class(partner_code(given, complement)).contains(&designed)
}

/// The IUPAC code for what a given base can pair with.
pub fn partner_code(given: Nucleotide, complement: Complement) -> u8 {
    match (given, complement) {
        (Nucleotide::A, _) => b'U',
        (Nucleotide::C, _) => b'G',
        (Nucleotide::G, Complement::Strict) => b'C',
        (Nucleotide::G, Complement::Wobble) => b'Y',
        (Nucleotide::U, Complement::Strict) => b'A',
        (Nucleotide::U, Complement::Wobble) => b'R',
    }
}

/// Every segment that could form an ungapped duplex of `k` pairs with a window
/// of `flank`, as IUPAC patterns — sorted and deduplicated.
///
/// The **reverse** complement, because pairing is antiparallel: the partner of
/// `w₀…w_{k−1}` is `partner(w_{k−1}) … partner(w₀)`. A flank shorter than `k`
/// has no window and yields no patterns.
pub fn pairing_motifs(flank: &[Nucleotide], k: usize, complement: Complement) -> Vec<String> {
    if k == 0 || flank.len() < k {
        return Vec::new();
    }
    let mut out: Vec<String> = flank
        .windows(k)
        .map(|w| {
            w.iter()
                .rev()
                .map(|&n| partner_code(n, complement) as char)
                .collect()
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// What one region contributes to a derivation.
///
/// The two roles that carry it are the constraint's two ends: a **given** region
/// is a source, a **designed** one is a target. An absent region is neither, and
/// the coding sequence is always designed, so only the flanks need classifying.
enum Role {
    /// A supplied sequence — the set is derived from it.
    Given(Vec<Nucleotide>),
    /// At least one position the designer chooses — the set is enforced over it.
    Designed,
    /// No positions at all.
    Absent,
}

impl Role {
    /// The sequence a given region holds, or `None` where it supplies none.
    fn given(&self) -> Option<&[Nucleotide]> {
        match self {
            Self::Given(nts) => Some(nts),
            Self::Designed | Self::Absent => None,
        }
    }

    /// Whether the designer chooses any position of this region.
    fn is_designed(&self) -> bool {
        matches!(self, Self::Designed)
    }
}

/// Classify a flank. A [`UtrSpec::Mixed`] flank whose every code names a single
/// nucleotide is a given sequence written the long way; one with any real choice
/// in it is designed.
fn role(spec: &UtrSpec) -> Result<Role, BeamError> {
    match spec {
        _ if spec.is_empty() => Ok(Role::Absent),
        UtrSpec::Fixed(s) => Ok(Role::Given(parse_literals(s)?)),
        UtrSpec::Designed(_) => Ok(Role::Designed),
        UtrSpec::Mixed(s) => match parse_literals(s) {
            Ok(nts) => Ok(Role::Given(nts)),
            Err(_) => Ok(Role::Designed),
        },
    }
}

/// Read a flank string as literal nucleotides (`T` as `U`), failing on any code
/// that names more than one.
fn parse_literals(s: &str) -> Result<Vec<Nucleotide>, BeamError> {
    s.bytes()
        .map(|b| {
            let b = if b == b'T' { b'U' } else { b };
            Nucleotide::from_byte(b).ok_or(BeamError::InvalidNucleotide(b))
        })
        .collect()
}

/// The refusal for the one arrangement no coverage can rescue.
const NOTHING_TO_DERIVE_FROM: &str =
    "no region of this run holds given sequence, so there is nothing to derive a complement \
     of; supply the sequence the design has to avoid pairing with, or — where every region is \
     designed — hold them over one mutually non-pairing alphabet, which is this constraint's \
     k = 1 limit";

/// What the covered classes reach on one arrangement.
#[derive(Default)]
struct Reach<'a> {
    sources: Vec<Source<'a>>,
    targets: Vec<Region>,
    classes: Vec<PairClass>,
}

/// One region the set derives from.
struct Source<'a> {
    region: Region,
    seq: &'a [Nucleotide],
}

/// A run's regions, classified and placed: which hold given sequence, which the
/// designer chooses, and where each sits on the molecule.
///
/// Everything the derivation needs about an arrangement is read off this, from
/// one rule — **a source must be given and a target must be designed** — rather
/// than by dispatching on the pair of flank roles.
struct Arrangement {
    utr5: Role,
    utr3: Role,
    cds_start: u32,
    cds_end: u32,
    seq_len: u32,
}

impl Arrangement {
    fn resolve(utr5: &UtrSpec, utr3: &UtrSpec, protein_len: usize) -> Result<Self, BeamError> {
        let cds_start = utr5.len() as u32;
        let cds_end = cds_start + 3 * protein_len as u32;
        Ok(Self {
            utr5: role(utr5)?,
            utr3: role(utr3)?,
            cds_start,
            cds_end,
            seq_len: cds_end + utr3.len() as u32,
        })
    }

    /// The given sequence a region holds. The coding sequence never holds any:
    /// it is what the design chooses.
    fn given(&self, region: Region) -> Option<&[Nucleotide]> {
        match region {
            Region::Utr5 => self.utr5.given(),
            Region::Utr3 => self.utr3.given(),
            Region::Cds => None,
        }
    }

    /// Whether the design chooses any position of a region. The coding sequence
    /// always qualifies: its synonymous codons are what the design picks.
    fn is_designed(&self, region: Region) -> bool {
        match region {
            Region::Utr5 => self.utr5.is_designed(),
            Region::Cds => true,
            Region::Utr3 => self.utr3.is_designed(),
        }
    }

    /// Whether any region of the run holds given sequence.
    fn has_given(&self) -> bool {
        self.utr5.given().is_some() || self.utr3.given().is_some()
    }

    /// Which regions the named classes make sources and which they make targets.
    ///
    /// A class names both endpoints, so it is read directly: the endpoint that
    /// is **given** is a source of that class, the endpoint that is **designed**
    /// is its target. A class with no given endpoint, or none designed, reaches
    /// nothing and contributes nothing — the caller may have named it for the
    /// structural constraint alone, which is a separate mechanism.
    fn reach(&self, classes: &[PairClass]) -> Reach<'_> {
        let mut reach = Reach::default();
        for region in [Region::Utr5, Region::Cds, Region::Utr3] {
            if classes.iter().any(|c| self.derives_from(*c, region)) {
                if let Some(seq) = self.given(region) {
                    reach.sources.push(Source { region, seq });
                }
            }
            if classes.iter().any(|c| self.enforces_over(*c, region)) {
                reach.targets.push(region);
            }
        }
        reach.classes = PairClass::ALL
            .into_iter()
            .filter(|class| classes.contains(class) && self.reaches(*class))
            .collect();
        reach
    }

    /// Whether a class has one given endpoint and one designed one, which is
    /// what it takes for the derivation to act across it.
    fn reaches(&self, class: PairClass) -> bool {
        let (a, b) = class.endpoints();
        (self.given(a).is_some() && self.is_designed(b))
            || (self.given(b).is_some() && self.is_designed(a))
    }

    /// Whether `class` makes `region` a source: `region` is a given endpoint of
    /// it and the other endpoint is designed.
    fn derives_from(&self, class: PairClass, region: Region) -> bool {
        let (a, b) = class.endpoints();
        let other = if a == region { b } else { a };
        (a == region || b == region) && self.given(region).is_some() && self.is_designed(other)
    }

    /// Whether `class` makes `region` a target: `region` is a designed endpoint
    /// of it and the other endpoint is given.
    fn enforces_over(&self, class: PairClass, region: Region) -> bool {
        let (a, b) = class.endpoints();
        let other = if a == region { b } else { a };
        (a == region || b == region) && self.is_designed(region) && self.given(other).is_some()
    }

    /// The positions a region occupies.
    fn bounds(&self, region: Region) -> (u32, u32) {
        match region {
            Region::Utr5 => (0, self.cds_start),
            Region::Cds => (self.cds_start, self.cds_end),
            Region::Utr3 => (self.cds_end, self.seq_len),
        }
    }

    /// The positions the targets occupy, as one window.
    ///
    /// One window suffices because the targets are always **contiguous**, which
    /// is a property of the class table rather than an assumption: a flank
    /// becomes a target only through a class joining it to a *given* region, and
    /// for the two flanks to be targets at once each would have to be designed
    /// while the other is given and the coding sequence — the only region
    /// between them — is designed too, so the span never skips a region it does
    /// not cover. Empty targets give an empty window, which enforces nothing.
    fn window_over(&self, targets: &[Region]) -> MotifWindow {
        let bounds = targets.iter().map(|&region| self.bounds(region));
        let lo = bounds.clone().map(|(lo, _)| lo).min().unwrap_or(0);
        let hi = bounds.map(|(_, hi)| hi).max().unwrap_or(0);
        MotifWindow::new(lo, hi)
    }
}

/// Resolve a derivation for a whole run: [`derive_from_flanks`] plus the refusal
/// of every combination that would need two enforcement windows.
///
/// `hand_supplied` and `window` are the run's own motif set and window. A
/// derived set pins the window itself, so either of those being non-neutral is a
/// second, conflicting source of truth rather than an addition.
///
/// This is the entry the design takes, and a front end that wants to refuse a
/// bad combination before doing any work should take the same one — the two
/// resolving a run differently is the drift this exists to prevent.
pub fn derive_for_run(
    fc: FlankComplement,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    protein_len: usize,
    classes: Option<&[PairClass]>,
    hand_supplied: bool,
    window: MotifWindow,
) -> Result<DerivedMotifs, BeamError> {
    if hand_supplied {
        return Err(BeamError::Derivation(
            "a derived set is enforced over the designed regions and a hand-supplied set \
             over its own window, and one run enforces one window; design them as \
             separate runs, or write the reverse complements out and pass the whole \
             set with an explicit window",
        ));
    }
    if window != MotifWindow::all() {
        return Err(BeamError::Derivation(
            "a derived set pins its own enforcement window to the designed regions; \
             drop the explicit window, or supply the patterns by hand instead",
        ));
    }
    derive_over(
        fc,
        utr5,
        utr3,
        protein_len,
        classes.unwrap_or(&PairClass::ALL),
    )
}

/// Derive the forbidden set for a run over these flanks: the patterns, the
/// designed regions they are enforced over, and what the expansion cost.
///
/// A source must be **given** and a target must be **designed**, which resolves
/// every arrangement a run can present. The coding sequence is designed by
/// construction, so it is a target in all of them, and the one arrangement left
/// without a source — no given region anywhere — is
/// [`BeamError::Derivation`] rather than an empty set.
///
/// The window is derived alongside the set rather than taken as an argument, so
/// a set cannot be pointed at the region it came from.
pub fn derive_from_flanks(
    fc: FlankComplement,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    protein_len: usize,
) -> Result<DerivedMotifs, BeamError> {
    derive_over(fc, utr5, utr3, protein_len, &PairClass::ALL)
}

/// [`derive_from_flanks`] restricted to the region-pair classes named.
///
/// A class determines both ends of the constraint, so one list states the whole
/// coverage: the derivation acts across a class whose one endpoint is given and
/// whose other is designed, and only across those. `&PairClass::ALL` is the
/// universal rule and is what [`derive_from_flanks`] passes.
///
/// A named class that reaches nothing — both endpoints given, or none given —
/// contributes nothing and is **not** an error: the caller may have named it for
/// the structural constraint, which is a different mechanism enforced elsewhere. What
/// the caller needs in that case is a report of which classes the derivation
/// actually reached, which is [`DerivedMotifs::sources`] and
/// [`DerivedMotifs::targets`].
///
/// The one refusal is the arrangement no class list can rescue: no region of the
/// run holds given sequence, so there is nothing to be a complement of.
pub fn derive_over(
    fc: FlankComplement,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    protein_len: usize,
    classes: &[PairClass],
) -> Result<DerivedMotifs, BeamError> {
    if fc.k == 0 {
        return Err(BeamError::Derivation(
            "the duplex length k must be at least 1; a zero-length duplex names every \
             position at once, which is not a constraint — pick the shortest run of \
             stacked pairs worth forbidding",
        ));
    }
    let arrangement = Arrangement::resolve(utr5, utr3, protein_len)?;
    if !arrangement.has_given() {
        return Err(BeamError::Derivation(NOTHING_TO_DERIVE_FROM));
    }
    let reach = arrangement.reach(classes);
    let window = arrangement.window_over(&reach.targets);
    Ok(build(&reach, window, fc)?)
}

/// Assemble the report for a derivation that is known to be well-posed.
///
/// Two sources yield one set: the patterns are unioned before deduplication, so
/// a segment that both given regions forbid is forbidden once and the automaton
/// is sized by the union rather than by the sum.
fn build(
    reach: &Reach<'_>,
    window: MotifWindow,
    fc: FlankComplement,
) -> Result<DerivedMotifs, crate::ac::MotifError> {
    let sources = &reach.sources;
    let mut motifs: Vec<String> = sources
        .iter()
        .flat_map(|source| pairing_motifs(source.seq, fc.k, fc.complement))
        .collect();
    motifs.sort();
    motifs.dedup();
    let (literals, expansion_sum) = expansion(&motifs)?;
    Ok(DerivedMotifs {
        window,
        sources: sources.iter().map(|source| source.region).collect(),
        targets: reach.targets.clone(),
        classes: reach.classes.clone(),
        source_len: sources.iter().map(|source| source.seq.len()).sum(),
        windows: sources
            .iter()
            .map(|source| (source.seq.len() + 1).saturating_sub(fc.k))
            .sum(),
        literals,
        expansion_sum,
        motifs,
    })
}

/// What the IUPAC set expands to: the **distinct** literal patterns, and the
/// per-pattern expansion summed before they are merged.
fn expansion(motifs: &[String]) -> Result<(usize, usize), crate::ac::MotifError> {
    let mut distinct: BTreeSet<Vec<u8>> = BTreeSet::new();
    let mut expansion_sum = 0usize;
    for motif in motifs {
        let expansions = expand_motif(motif)?;
        expansion_sum += expansions.len();
        distinct.extend(
            expansions
                .into_iter()
                .map(|p| p.into_iter().map(|n| n as u8).collect()),
        );
    }
    Ok((distinct.len(), expansion_sum))
}
