//! The sequence axis: nucleotides the caller has already decided.
//!
//! A pin names a position and the nucleotides it may spell — one letter, or an
//! IUPAC degenerate code standing for a subset. Pins are **nucleotide-granular**
//! and need not align to codon boundaries: a restriction site, a cloning
//! overhang or a Kozak consensus lands wherever it lands.
//!
//! # A pin constrains the nucleotide, and the codon set is its projection
//!
//! Inside the CDS a position is spelled by a codon, so a nucleotide pin has to
//! reach the lattice through one. The reading this crate implements is that the
//! pin constrains the **nucleotide**: the codon set kept at that residue is
//! exactly the codons spelling an admitted nucleotide at the pinned offset, and
//! pins landing on the same codon **intersect**. Nothing is relaxed to a
//! codon-level approximation, and nothing is widened to "some codon that nearly
//! fits" — a residue left with no codon is [`PinError::NoCodon`], reported
//! before any search runs.
//!
//! The consequence worth stating, because it is the one a caller meets: two
//! pins that are each individually spellable can be jointly unspellable when
//! they share a codon, and the failure names the **residue**, not either pin.
//! `AUG` is the extreme case — one codon, so any pin over methionine that is
//! not `AUG`'s own letter empties it.
//!
//! # CAI accounting
//!
//! A residue pinned to a single codon is compiled by `dfa::CodonUnit::single`,
//! whose edges carry **zero** adaptation — deliberately, so no parse can claim
//! adaptation it never spelled. That is a constant across every parse and so
//! cannot change the argmin, but it does change the *report*, which adds it back
//! (`crate::restore_pinned_cai`). Reported CAI is therefore the whole CDS's,
//! pins included.

use derna_codon_dp::codon_table::AMINO_ACIDS;
use rna_mfe::Nucleotide;

use crate::coords::Layout;
use crate::nuc_set::NucSet;

/// The nucleotides each position of an mRNA may spell.
///
/// A default-constructed value over any length is **inert**: every position
/// holds [`NucSet::ALL`], so the design is byte-identical to the unpinned one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pins {
    /// One entry per absolute position; [`NucSet::ALL`] is unconstrained.
    sets: Vec<NucSet>,
    /// Fast path and floor test: whether any position is narrower than `ALL`.
    active: bool,
}

/// Why a pin could not be recorded, or could not reach the lattice.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PinError {
    /// A letter in a pin string is not an IUPAC nucleotide code.
    #[error("pin letter {0:?} is not an IUPAC nucleotide code")]
    NotIupac(char),
    /// A pin string is empty, so it names no position.
    #[error("a pin needs at least one letter")]
    Empty,
    /// The pin runs off the 3′ end of the mRNA.
    #[error("a {len}-letter pin at position {at} runs past the {mrna_len}-nt mRNA")]
    PastEnd {
        /// Absolute 0-based position of the pin's first letter.
        at: u32,
        /// How many letters the pin carries.
        len: usize,
        /// The mRNA's length.
        mrna_len: u32,
    },
    /// Two pins over one position admit no common nucleotide. Caught when the
    /// second is recorded — at parse time, not as an infeasible solve.
    #[error("pins conflict at {position}: {held} and {added} share no nucleotide")]
    Conflict {
        /// The position, region-qualified.
        position: String,
        /// The set already held there, as IUPAC letters.
        held: String,
        /// The set the new pin asks for.
        added: String,
    },
    /// No codon of the residue covering a pinned position spells an admitted
    /// nucleotide at every pinned offset of that codon.
    #[error(
        "no codon for residue {residue} ({amino_acid}) at {position} spells the pinned pattern {pattern}"
    )]
    NoCodon {
        /// 1-based residue index within the protein.
        residue: u32,
        /// The residue's one-letter code.
        amino_acid: char,
        /// The codon's first position, region-qualified.
        position: String,
        /// The three pinned sets over that codon, as IUPAC letters.
        pattern: String,
    },
    /// A pin excludes the nucleotide a **given** flank already supplies there.
    /// A given flank is not a design space, so this cannot be solved — it is a
    /// contradiction in the input.
    #[error("pin at {position} admits {pinned} but the given flank supplies {supplied}")]
    GivenFlank {
        /// The position, region-qualified.
        position: String,
        /// The set the pin asks for.
        pinned: String,
        /// The nucleotide the flank string already fixes.
        supplied: char,
    },
}

impl Pins {
    /// An inert pin set over an `mrna_len`-nucleotide mRNA.
    pub fn none(mrna_len: u32) -> Self {
        Self {
            sets: vec![NucSet::ALL; mrna_len as usize],
            active: false,
        }
    }

    /// Whether no position is constrained — the byte-identical floor.
    pub fn is_inert(&self) -> bool {
        !self.active
    }

    /// The nucleotides position `pos` may spell. [`NucSet::ALL`] past the end.
    pub fn at(&self, pos: u32) -> NucSet {
        self.sets.get(pos as usize).copied().unwrap_or(NucSet::ALL)
    }

    /// Record an **anchored subsequence**: `letters` spells consecutive
    /// positions starting at absolute `at`, each intersected with what is
    /// already held there.
    ///
    /// `layout` is read only to phrase errors in the caller's own coordinates.
    pub fn pin(&mut self, layout: &Layout, at: u32, letters: &str) -> Result<(), PinError> {
        if letters.is_empty() {
            return Err(PinError::Empty);
        }
        let bytes = letters.as_bytes();
        if at as usize + bytes.len() > self.sets.len() {
            return Err(PinError::PastEnd {
                at,
                len: bytes.len(),
                mrna_len: self.sets.len() as u32,
            });
        }
        for (offset, &code) in bytes.iter().enumerate() {
            let want = NucSet::from_iupac(code.to_ascii_uppercase())
                .ok_or(PinError::NotIupac(code as char))?;
            let pos = at + offset as u32;
            let held = self.sets[pos as usize];
            let merged = intersect(held, want).ok_or_else(|| PinError::Conflict {
                position: layout.describe(pos),
                held: iupac_of(held).to_string(),
                added: iupac_of(want).to_string(),
            })?;
            self.sets[pos as usize] = merged;
            self.active |= !merged.is_full();
        }
        Ok(())
    }

    /// The codon subsets the pins leave each residue — the `allowed` argument
    /// `dfa::build_restricted_with_utr` takes.
    ///
    /// `None` when no CDS position is pinned, which routes the caller to the
    /// unrestricted lattice verbatim rather than to an all-codons restriction
    /// that would merely reproduce it.
    pub(crate) fn cds_codon_sets(
        &self,
        layout: &Layout,
        protein: &[usize],
    ) -> Result<Option<Vec<Vec<[u8; 3]>>>, PinError> {
        let cds_start = layout.cds_start();
        if (0..layout.cds).all(|k| self.at(cds_start + k).is_full()) {
            return Ok(None);
        }
        let sets = protein
            .iter()
            .enumerate()
            .map(|(i, &aa)| self.codon_set_for(layout, i as u32, aa))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(sets))
    }

    /// The codons residue `i` may still take: those spelling an admitted
    /// nucleotide at all three offsets.
    fn codon_set_for(&self, layout: &Layout, i: u32, aa: usize) -> Result<Vec<[u8; 3]>, PinError> {
        let base = layout.cds_start() + 3 * i;
        let masks = [self.at(base), self.at(base + 1), self.at(base + 2)];
        let kept: Vec<[u8; 3]> = AMINO_ACIDS[aa]
            .codons
            .iter()
            .map(|c| c.nuc)
            .filter(|nuc| {
                nuc.iter()
                    .zip(masks)
                    .all(|(&b, mask)| mask.contains(nucleotide_of(b)))
            })
            .collect();
        if kept.is_empty() {
            return Err(PinError::NoCodon {
                residue: i + 1,
                amino_acid: AMINO_ACIDS[aa].code,
                position: layout.describe(base),
                pattern: masks.iter().map(|&m| iupac_of(m)).collect(),
            });
        }
        Ok(kept)
    }

    /// The pins over one flank, as the per-position mask a designed flank is
    /// spliced from.
    ///
    /// `supplied` is the flank's own per-position sets — what the caller wrote
    /// as `Fixed`, `Designed` or `Mixed`. A pin narrowing a **given** position
    /// away from its supplied nucleotide is [`PinError::GivenFlank`]: a given
    /// flank offers nothing to choose, so the two statements simply contradict.
    pub(crate) fn flank_mask(
        &self,
        layout: &Layout,
        base: u32,
        supplied: &[NucSet],
    ) -> Result<Vec<NucSet>, PinError> {
        supplied
            .iter()
            .enumerate()
            .map(|(i, &have)| {
                let pos = base + i as u32;
                let want = self.at(pos);
                intersect(have, want).ok_or_else(|| given_flank_error(layout, pos, have, want))
            })
            .collect()
    }

    /// Whether any pin lands on `[base, base + len)` — the test that decides
    /// whether a flank needs a mask at all, keeping an untouched flank on its
    /// current path byte for byte.
    pub(crate) fn touches(&self, base: u32, len: u32) -> bool {
        (0..len).any(|i| !self.at(base + i).is_full())
    }
}

/// A pin against a given flank: the supplied nucleotide is the one the caller
/// already fixed, so the message names it rather than reporting an empty set.
fn given_flank_error(layout: &Layout, pos: u32, have: NucSet, want: NucSet) -> PinError {
    if have.is_pinned() {
        return PinError::GivenFlank {
            position: layout.describe(pos),
            pinned: iupac_of(want).to_string(),
            supplied: iupac_of(have),
        };
    }
    PinError::Conflict {
        position: layout.describe(pos),
        held: iupac_of(have).to_string(),
        added: iupac_of(want).to_string(),
    }
}

/// The intersection of two sets, or `None` when they share no nucleotide.
/// [`NucSet`] has no empty value by construction, so the emptiness shows up
/// here rather than as an unrepresentable set.
fn intersect(a: NucSet, b: NucSet) -> Option<NucSet> {
    let kept: Vec<Nucleotide> = a.iter().filter(|nt| b.contains(*nt)).collect();
    NucSet::from_nucleotides(&kept)
}

/// The IUPAC code denoting `set` — the spelling errors and run headers report
/// sets in, so a caller reads back what they wrote.
fn iupac_of(set: NucSet) -> char {
    const CODES: [u8; 15] = *b"ACGURYSWKMBDHVN";
    CODES
        .iter()
        .copied()
        .find(|&code| NucSet::from_iupac(code) == Some(set))
        .map_or('?', |code| code as char)
}

/// DERNA's codon-table nucleotide indices (`A=0, C=1, G=2, U=3`).
fn nucleotide_of(b: u8) -> Nucleotide {
    match b {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        _ => Nucleotide::U,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use derna_codon_dp::codon_table::aa_to_index;

    fn aa(protein: &str) -> Vec<usize> {
        protein.chars().map(|c| aa_to_index(c).unwrap()).collect()
    }

    /// 6 nt of 5′ UTR, a 4-residue CDS, 6 nt of 3′ UTR.
    fn layout() -> Layout {
        Layout::new(6, 4, 6)
    }

    #[test]
    fn an_unpinned_set_is_inert_and_needs_no_restriction() {
        let l = layout();
        let pins = Pins::none(l.len());
        assert!(pins.is_inert());
        assert_eq!(pins.cds_codon_sets(&l, &aa("MKVL")), Ok(None));
    }

    #[test]
    fn a_pin_is_an_anchored_subsequence() {
        let l = layout();
        let mut pins = Pins::none(l.len());
        pins.pin(&l, 0, "GCCACC").unwrap();
        for (i, code) in b"GCCACC".iter().enumerate() {
            assert_eq!(pins.at(i as u32), NucSet::from_iupac(*code).unwrap());
        }
        assert!(pins.at(6).is_full());
        assert!(!pins.is_inert());
    }

    /// The pin constrains the nucleotide; the codon set is what is left of the
    /// residue after projecting it. Leucine has six codons; requiring `C` at the
    /// first offset keeps exactly the four `CUN` ones.
    #[test]
    fn a_pin_projects_onto_the_codons_that_spell_it() {
        let l = Layout::new(0, 1, 0);
        let mut pins = Pins::none(l.len());
        pins.pin(&l, 0, "C").unwrap();
        let sets = pins.cds_codon_sets(&l, &aa("L")).unwrap().unwrap();
        assert_eq!(sets[0].len(), 4);
        assert!(sets[0].iter().all(|c| c[0] == 1));
    }

    /// A degenerate code is the same intersection over a set: `Y` is `{C, U}`,
    /// so over leucine it keeps `CUN` and `UUR` together.
    #[test]
    fn a_degenerate_code_admits_every_nucleotide_it_denotes() {
        let l = Layout::new(0, 1, 0);
        let mut pins = Pins::none(l.len());
        pins.pin(&l, 0, "Y").unwrap();
        let sets = pins.cds_codon_sets(&l, &aa("L")).unwrap().unwrap();
        assert_eq!(sets[0].len(), 6, "every leucine codon starts C or U");
    }

    /// The failure a caller actually meets: methionine has one codon, so any
    /// pin over it that `AUG` does not spell empties the residue — and the error
    /// names the residue rather than the pin, because either pin alone is fine.
    #[test]
    fn a_residue_with_no_surviving_codon_is_named_at_parse_time() {
        let l = Layout::new(0, 1, 0);
        let mut pins = Pins::none(l.len());
        pins.pin(&l, 0, "C").unwrap();
        let err = pins.cds_codon_sets(&l, &aa("M")).unwrap_err();
        assert!(
            matches!(
                &err,
                PinError::NoCodon {
                    residue: 1,
                    amino_acid: 'M',
                    ..
                }
            ),
            "{err}"
        );
    }

    /// Two pins over one position that share no nucleotide are refused where
    /// the second is recorded — D5's parse-time error, not an infeasible solve.
    #[test]
    fn contradicting_pins_are_refused_when_the_second_is_recorded() {
        let l = layout();
        let mut pins = Pins::none(l.len());
        pins.pin(&l, 3, "M").unwrap(); // {A, C}
        let err = pins.pin(&l, 3, "G").unwrap_err();
        assert!(matches!(err, PinError::Conflict { .. }), "{err}");
        // …and a compatible second pin narrows rather than replaces.
        pins.pin(&l, 3, "A").unwrap();
        assert_eq!(pins.at(3), NucSet::of(Nucleotide::A));
    }

    #[test]
    fn a_pin_past_the_three_prime_end_is_refused() {
        let l = layout();
        let mut pins = Pins::none(l.len());
        let err = pins.pin(&l, l.len() - 2, "ACGU").unwrap_err();
        assert!(matches!(err, PinError::PastEnd { .. }), "{err}");
    }

    #[test]
    fn junk_letters_are_refused() {
        let l = layout();
        let mut pins = Pins::none(l.len());
        assert_eq!(pins.pin(&l, 0, "AZ"), Err(PinError::NotIupac('Z')));
        assert_eq!(pins.pin(&l, 0, ""), Err(PinError::Empty));
    }

    /// A pin over a given flank position either agrees with the supplied
    /// nucleotide or contradicts it; there is no design freedom to spend.
    #[test]
    fn a_pin_contradicting_a_given_flank_names_the_supplied_nucleotide() {
        let l = layout();
        let mut pins = Pins::none(l.len());
        pins.pin(&l, 1, "G").unwrap();
        let supplied = vec![NucSet::of(Nucleotide::A); 6];
        let err = pins.flank_mask(&l, 0, &supplied).unwrap_err();
        assert!(
            matches!(&err, PinError::GivenFlank { supplied: 'A', .. }),
            "{err}"
        );
    }

    /// A pin on a designed flank position is exactly an alphabet subset there —
    /// the shape the designed-flank restriction already ships.
    #[test]
    fn a_pin_on_a_designed_flank_narrows_that_position() {
        let l = layout();
        let mut pins = Pins::none(l.len());
        pins.pin(&l, 2, "Y").unwrap();
        let mask = pins
            .flank_mask(&l, 0, &[NucSet::ALL; 6])
            .expect("designed positions have freedom to narrow");
        assert_eq!(mask[2], NucSet::from_iupac(b'Y').unwrap());
        assert!(mask[0].is_full());
        assert!(pins.touches(0, 6));
        assert!(!pins.touches(l.cds_end(), 6));
    }

    #[test]
    fn every_iupac_code_names_itself_back() {
        for code in *b"ACGURYSWKMBDHVN" {
            let set = NucSet::from_iupac(code).unwrap();
            assert_eq!(iupac_of(set), code as char);
        }
    }
}
