//! The two positional axes together: what the sequence must spell, and what the
//! structure must do.
//!
//! [`Pins`] and [`StructureConstraints`] are independent — either is meaningful
//! without the other, and neither is expressed in terms of the other — but a
//! caller supplies them as one statement about one molecule, and the checks that
//! matter are the ones that read *both*. This module is where they meet.
//!
//! # The three feasibility tiers
//!
//! Infeasibility here is genuine, unlike the region families' modularity gap: a
//! required pair really can have no satisfying design. Which tier catches it
//! decides how strong the resulting claim is, so the report says which one did.
//!
//! 1. **Construction.** A pin set that leaves a residue no codon. Found while
//!    the lattice is built, before any folding.
//! 2. **Pre-check.** A required pair whose two positions offer no complementary
//!    nucleotide combination at all. `O(1)` per pair against the lattice's
//!    per-position nucleotide sets, so it is cheap enough to always run. It is a
//!    one-way test: it proves infeasibility and never feasibility, because it
//!    reads each position's set independently and a codon correlates its three.
//! 3. **The scan.** Joint feasibility across a whole element — the question the
//!    first two tiers cannot answer. On the exact `b → ∞` lattice an empty scan
//!    is a **proof**; under a finite beam it is not, because pruning can discard
//!    the only feasible completion. The two are different sentences and the
//!    error type keeps them apart.

use std::sync::LazyLock;

use rna_mfe::{loops::pair_from_nucleotides, params::BasePair, Nucleotide};

use crate::coords::{Coord, CoordError, Layout};
use crate::dfa::Dfa;
use crate::nuc_set::NucSet;
use crate::pins::{PinError, Pins};
use crate::region::{PairConstraint, Verdict};
use crate::structure::{StructureConstraints, StructureError};

/// Everything a caller has already decided about one molecule: the nucleotides
/// at chosen positions, and the fold at chosen positions.
///
/// [`Default`] is the empty statement over a zero-length mRNA, which is inert at
/// every position — the value a run that names no positional constraint carries,
/// and the one the byte-identical floors are measured against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionalConstraints {
    /// The sequence axis.
    pub pins: Pins,
    /// The structure axis.
    pub structure: StructureConstraints,
}

impl Default for PositionalConstraints {
    fn default() -> Self {
        Self::none(0)
    }
}

impl PositionalConstraints {
    /// The empty statement over an `mrna_len`-nucleotide mRNA.
    pub fn none(mrna_len: u32) -> Self {
        Self {
            pins: Pins::none(mrna_len),
            structure: StructureConstraints::none(mrna_len),
        }
    }

    /// Whether nothing at all is constrained — the floor that routes a run to
    /// the unconstrained substrate verbatim.
    pub fn is_inert(&self) -> bool {
        self.pins.is_inert() && self.structure.is_inert()
    }

    /// The shared inert value, borrowed for as long as the program runs.
    ///
    /// A `CompositeSpec` holds a reference, so a caller that *returns* one
    /// cannot point it at a temporary. Every such caller wants the same empty
    /// statement, so there is one.
    pub fn empty() -> &'static Self {
        static EMPTY: LazyLock<PositionalConstraints> =
            LazyLock::new(PositionalConstraints::default);
        &EMPTY
    }
}

/// The constraint statement **as written** — in the caller's coordinates,
/// before a layout exists to resolve it against.
///
/// A caller writes constraints against a protein and two flanks, and the
/// absolute positions only exist once those lengths are known. Keeping the two
/// apart is what lets `cds:12` survive a flank-length edit, and it is why every
/// coordinate is resolved exactly once, in [`Self::resolve`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PositionalSpec {
    /// `--pin <coord>:<letters>` — anchored subsequences.
    pins: Vec<(Coord, String)>,
    /// `--structure <string>` — the general whole-mRNA form.
    structure: Option<String>,
    /// `--helix <coord>:<coord>:<len>` — sugar over the string.
    helices: Vec<(Coord, Coord, u32)>,
    /// `--unpaired <coord>..<coord>` — sugar over the string.
    unpaired: Vec<(Coord, Coord)>,
    /// `--forbid-pair <coord>:<coord>` — the negative direction.
    forbidden: Vec<(Coord, Coord)>,
}

impl PositionalSpec {
    /// Whether nothing was written at all.
    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
            && self.structure.is_none()
            && self.helices.is_empty()
            && self.unpaired.is_empty()
            && self.forbidden.is_empty()
    }

    /// Record `<coord>:<letters>` — a pin anchored at `coord`, spelling
    /// consecutive positions from there (`cds:4:G`, `62:GAAUUC`, `5utr:-6:GCCACC`).
    pub fn add_pin(&mut self, s: &str) -> Result<(), ConstraintError> {
        let (head, letters) = s.rsplit_once(':').ok_or_else(|| ConstraintError::Syntax {
            what: "--pin",
            found: s.to_owned(),
            expected: "<coord>:<letters>, e.g. cds:4:G or 5utr:-6:GCCACC",
        })?;
        self.pins.push((Coord::parse(head)?, letters.to_owned()));
        Ok(())
    }

    /// Record the general whole-mRNA constraint string.
    pub fn set_structure(&mut self, s: String) {
        self.structure = Some(s);
    }

    /// Record `<coord>:<coord>:<len>` — a required helix pairing
    /// `(open + s, close − s)` for `s = 0 … len − 1`.
    pub fn add_helix(&mut self, s: &str) -> Result<(), ConstraintError> {
        let syntax = || ConstraintError::Syntax {
            what: "--helix",
            found: s.to_owned(),
            expected: "<coord>:<coord>:<len>, e.g. cds:121:cds:201:8",
        };
        let (rest, len) = s.rsplit_once(':').ok_or_else(syntax)?;
        let len: u32 = len.parse().map_err(|_| syntax())?;
        if len == 0 {
            return Err(syntax());
        }
        let (open, close) = two_coords(rest).ok_or_else(syntax)?;
        self.helices.push((open?, close?, len));
        Ok(())
    }

    /// Record `<coord>..<coord>` — an inclusive range required to carry no pair.
    pub fn add_unpaired(&mut self, s: &str) -> Result<(), ConstraintError> {
        let (from, to) = s.split_once("..").ok_or_else(|| ConstraintError::Syntax {
            what: "--unpaired",
            found: s.to_owned(),
            expected: "<coord>..<coord>, e.g. 5utr:-4..cds:12",
        })?;
        self.unpaired.push((Coord::parse(from)?, Coord::parse(to)?));
        Ok(())
    }

    /// Record `<coord>:<coord>` — one pair forbidden outright.
    pub fn add_forbidden_pair(&mut self, s: &str) -> Result<(), ConstraintError> {
        let syntax = || ConstraintError::Syntax {
            what: "--forbid-pair",
            found: s.to_owned(),
            expected: "<coord>:<coord>, e.g. cds:10:cds:40",
        };
        let (open, close) = two_coords(s).ok_or_else(syntax)?;
        self.forbidden.push((open?, close?));
        Ok(())
    }

    /// Resolve every coordinate against `layout` and build the constraint set,
    /// raising any contradiction here rather than as an infeasible solve.
    ///
    /// The general string is applied first and the sugar over it, so a run that
    /// gives both gets their intersection with the same errors it would get
    /// from either alone.
    pub fn resolve(&self, layout: &Layout) -> Result<PositionalConstraints, ConstraintError> {
        let mut pc = PositionalConstraints::none(layout.len());
        if let Some(s) = &self.structure {
            pc.structure = StructureConstraints::parse(layout, s)?;
        }
        for (from, to) in &self.unpaired {
            let (a, b) = (layout.resolve(*from)?, layout.resolve(*to)?);
            if a > b {
                return Err(ConstraintError::EmptyRange {
                    from: from.to_string(),
                    to: to.to_string(),
                });
            }
            pc.structure.require_unpaired(layout, a, b)?;
        }
        for (open, close, len) in &self.helices {
            let (a, b) = (layout.resolve(*open)?, layout.resolve(*close)?);
            pc.structure.require_helix(layout, a, b, *len)?;
        }
        for (open, close) in &self.forbidden {
            let (a, b) = (layout.resolve(*open)?, layout.resolve(*close)?);
            pc.structure.forbid_pair(layout, a, b)?;
        }
        for (at, letters) in &self.pins {
            pc.pins.pin(layout, layout.resolve(*at)?, letters)?;
        }
        Ok(pc)
    }
}

/// Split a colon-separated string into exactly two coordinates.
///
/// A coordinate is either `N` or `<region>:N`, so the token count alone is
/// ambiguous; the split is decided by whether a token is a signed integer,
/// which the region names never are.
fn two_coords(s: &str) -> Option<(Result<Coord, CoordError>, Result<Coord, CoordError>)> {
    let tokens: Vec<&str> = s.split(':').collect();
    let width = |i: usize| -> usize {
        if tokens.get(i).is_some_and(|t| t.parse::<i64>().is_ok()) {
            1
        } else {
            2
        }
    };
    let first = width(0);
    if tokens.len() != first + width(first) {
        return None;
    }
    Some((
        Coord::parse(&tokens[..first].join(":")),
        Coord::parse(&tokens[first..].join(":")),
    ))
}

/// A statement that contradicts another statement, or one no design can satisfy.
///
/// Every variant names the flag or coordinate that resolves it, because a
/// constraint whose exact meaning lives only in the implementation is one a
/// caller will get wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConstraintError {
    /// A pin could not be recorded or could not reach the lattice.
    #[error(transparent)]
    Pin(#[from] PinError),
    /// A structure constraint contradicts another, or the grammar.
    #[error(transparent)]
    Structure(#[from] StructureError),
    /// A coordinate could not be read, or lands off the molecule.
    #[error(transparent)]
    Coord(#[from] CoordError),
    /// A flag's value does not match its grammar.
    #[error("{what} expects {expected}, got {found:?}")]
    Syntax {
        /// The flag.
        what: &'static str,
        /// What was written.
        found: String,
        /// The grammar, with an example.
        expected: &'static str,
    },
    /// An unpaired range written 3′-to-5′, which names no positions. Refused
    /// rather than silently constraining nothing.
    #[error("--unpaired {from}..{to} runs backwards and names no position")]
    EmptyRange {
        /// The range's first coordinate as written.
        from: String,
        /// Its second.
        to: String,
    },
    /// A required pair the region-class table forbids. The two statements are
    /// both hard, so neither can give way silently.
    #[error("required pair ({open}, {close}) is forbidden by the region-class table — allow the class with --region, or drop the pair")]
    RegionTableForbidsPair {
        /// The 5′ endpoint, region-qualified.
        open: String,
        /// The 3′ endpoint, region-qualified.
        close: String,
    },
    /// A required position inside the start-proximal accessibility window,
    /// which exists to keep exactly that window unpaired.
    #[error("{position} must pair but lies inside the accessibility window, which forbids pairing there — narrow --access, or drop the requirement")]
    AccessWindowForbidsPair {
        /// The position, region-qualified.
        position: String,
    },
}

/// A required pair no design can form, and the evidence for saying so.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("required pair ({open}, {close}) cannot form: {open} may spell {open_set} and {close} may spell {close_set}, and no combination of those is a canonical pair")]
pub struct UnpairablePair {
    /// The 5′ endpoint, region-qualified.
    pub open: String,
    /// The 3′ endpoint, region-qualified.
    pub close: String,
    /// The nucleotides the 5′ endpoint may spell, as IUPAC letters.
    pub open_set: String,
    /// The nucleotides the 3′ endpoint may spell.
    pub close_set: String,
}

/// Why no design satisfying the constraints exists — and, decisively, **which
/// tier found out**.
///
/// The distinction is not bookkeeping. [`Self::ExactScan`] is a proof over the
/// whole design space; [`Self::Beam`] is a statement about one search, and a
/// caller who reads it as the former will discard a feasible problem.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Infeasible {
    /// Construction tier: the pins left a residue with no codon.
    #[error("infeasible at construction: {0}")]
    Pins(#[from] PinError),
    /// Pre-check tier: a required pair with no spellable partner combination.
    #[error("infeasible before search: {0}")]
    PreCheck(#[from] UnpairablePair),
    /// The scan tier on the **exact** `b → ∞` lattice: no derivation exists.
    /// This is a proof — every design was considered.
    #[error("no design satisfies the constraints (proved on the exact lattice)")]
    ExactScan,
    /// The scan tier **under a beam**: no derivation survived pruning. This is
    /// *not* a proof of infeasibility — the beam may have discarded the only
    /// feasible completion. Re-run with `--mode exact` to decide.
    #[error("no feasible design found at this beam width — the beam may have pruned the only feasible completion; re-run with --mode exact for a decisive answer")]
    Beam,
}

/// The nucleotides position `pos` of `dfa` can spell: the union over every lane
/// at that position of its out-edges' nucleotides.
///
/// Reading it off the graph rather than off the constraints is what makes the
/// pre-check uniform — a CDS position restricted by a pin, a designed flank
/// position restricted by an alphabet subset, and a given flank position all
/// answer through the same query.
pub(crate) fn spellable_at(dfa: &Dfa, pos: u32) -> Option<NucSet> {
    let nts: Vec<Nucleotide> = dfa
        .nodes_at(pos)
        .iter()
        .flat_map(|&n| dfa.right(n).iter().map(|e| e.nuc))
        .collect();
    NucSet::from_nucleotides(&nts)
}

/// The pre-check (tier 2): every required pair, against what its two positions
/// can actually spell.
///
/// Cheap enough to always run — 16 nucleotide combinations per pair — and worth
/// running because the alternative is an empty scan the caller cannot diagnose.
/// A pair that passes is *not* thereby feasible: the sets are read
/// independently, and a codon correlates the three positions it spells.
pub(crate) fn precheck_required_pairs(
    dfa: &Dfa,
    layout: &Layout,
    structure: &StructureConstraints,
) -> Result<(), UnpairablePair> {
    for (open, close) in structure.required_pairs() {
        let (Some(a), Some(b)) = (spellable_at(dfa, open), spellable_at(dfa, close)) else {
            continue;
        };
        if a.iter().any(|x| {
            b.iter()
                .any(|y| pair_from_nucleotides(x, y) != BasePair::NN)
        }) {
            continue;
        }
        return Err(UnpairablePair {
            open: layout.describe(open),
            close: layout.describe(close),
            open_set: set_letters(a),
            close_set: set_letters(b),
        });
    }
    Ok(())
}

/// Refuse a required pair the class table or the accessibility window already
/// forbids (D5). Both statements are hard, so resolving one silently in favour
/// of the other is exactly the mistake the region work was promoted to fix.
pub(crate) fn check_against_region_table(
    constraint: &PairConstraint,
    layout: &Layout,
    structure: &StructureConstraints,
) -> Result<(), ConstraintError> {
    // A position required to pair *at all* is unsatisfiable inside a hard
    // accessibility window, whatever partner it would have taken — the window
    // is the one family keyed on a single endpoint. Tested before the class
    // table because it also explains a *pair* the window forbids, and names the
    // flag that actually resolves it.
    for pos in structure.required_positions() {
        if constraint.window_forbids(pos) {
            return Err(ConstraintError::AccessWindowForbidsPair {
                position: layout.describe(pos),
            });
        }
    }
    for (open, close) in structure.required_pairs() {
        if constraint.verdict(open, close) == Verdict::Forbid {
            return Err(ConstraintError::RegionTableForbidsPair {
                open: layout.describe(open),
                close: layout.describe(close),
            });
        }
    }
    Ok(())
}

/// A set as IUPAC letters, listing members rather than naming the code — an
/// error is read once and `{A, G}` needs no lookup table where `R` does.
fn set_letters(set: NucSet) -> String {
    let names: Vec<String> = set.iter().map(|nt| format!("{nt:?}")).collect();
    format!("{{{}}}", names.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{AccessibilityWindow, Enforcement, RegionConstraints};
    use derna_codon_dp::codon_table::aa_to_index;
    use derna_codon_dp::CodonUsage;

    fn dfa_for(protein: &str) -> Dfa {
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        crate::dfa::build(&aa, &usage)
    }

    #[test]
    fn the_empty_statement_is_inert() {
        assert!(PositionalConstraints::default().is_inert());
        assert!(PositionalConstraints::none(60).is_inert());
    }

    /// The slot-feasibility rule, checked against the lattice rather than
    /// asserted: valine is `GUN` and isoleucine is `AUH`, so both have `U`
    /// fixed at slot 2 by the protein — and `U · U` is not a pair. No
    /// synonymous choice repairs it, and the pre-check says so before any
    /// search runs.
    #[test]
    fn a_pair_between_two_protein_fixed_slot_twos_is_refused_before_the_search() {
        let dfa = dfa_for("VAAAAIAA");
        let l = Layout::new(0, 8, 0);
        // Valine at residue 1 → slot 2 is position 1; isoleucine at residue 6 →
        // slot 2 is position 16.
        assert_eq!(spellable_at(&dfa, 1), Some(NucSet::of(Nucleotide::U)));
        assert_eq!(spellable_at(&dfa, 16), Some(NucSet::of(Nucleotide::U)));
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 1, 16).unwrap();
        let err = precheck_required_pairs(&dfa, &l, &sc).unwrap_err();
        assert!(err.to_string().contains("cannot form"), "{err}");
    }

    /// The control the case above needs: a wobble-to-wobble pair over the same
    /// construct passes the pre-check, so the refusal is a statement about the
    /// slots rather than about the pre-check refusing everything.
    #[test]
    fn a_pair_with_a_spellable_combination_passes_the_precheck() {
        let dfa = dfa_for("VAAAAIAA");
        let l = Layout::new(0, 8, 0);
        let mut sc = StructureConstraints::none(l.len());
        // Alanine is `GCN`: slot 3 is a free wobble at both ends.
        sc.require_pair(&l, 8, 20).unwrap();
        assert!(precheck_required_pairs(&dfa, &l, &sc).is_ok());
    }

    #[test]
    fn a_required_pair_the_class_table_forbids_is_refused() {
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = "AAAA".chars().map(|c| aa_to_index(c).unwrap()).collect();
        let flank = vec![Nucleotide::A; 8];
        let dfa = crate::dfa::build_with_utr(&aa, &usage, &flank, &[]);
        let l = Layout::new(8, 4, 0);
        let rc = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
        let constraint = PairConstraint::new(&rc, &dfa);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 2, 12).unwrap(); // 5′ UTR ↔ CDS
        assert!(matches!(
            check_against_region_table(&constraint, &l, &sc),
            Err(ConstraintError::RegionTableForbidsPair { .. })
        ));
        // An intra-CDS pair over the same table is untouched.
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 9, 15).unwrap();
        assert!(check_against_region_table(&constraint, &l, &sc).is_ok());
    }

    #[test]
    fn a_required_position_inside_a_hard_access_window_is_refused() {
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = "AAAA".chars().map(|c| aa_to_index(c).unwrap()).collect();
        let flank = vec![Nucleotide::A; 8];
        let dfa = crate::dfa::build_with_utr(&aa, &usage, &flank, &[]);
        let l = Layout::new(8, 4, 0);
        let rc = RegionConstraints::default().with_window(AccessibilityWindow::hard(4, 4));
        let constraint = PairConstraint::new(&rc, &dfa);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 5, 17).unwrap(); // 5 is inside [4, 12)
        assert!(matches!(
            check_against_region_table(&constraint, &l, &sc),
            Err(ConstraintError::AccessWindowForbidsPair { .. })
        ));
    }
}
