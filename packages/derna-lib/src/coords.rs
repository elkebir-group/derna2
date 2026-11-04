//! Where a constraint lands: the whole-mRNA coordinate system and the
//! region-qualified spelling that survives a flank-length edit.
//!
//! Every constraint in this crate is a predicate on an **absolute** nucleotide
//! position, 0-based over `UTR₅′ + CDS + UTR₃′`. That is the only coordinate the
//! constraints ever see. A caller writing constraints by hand wants something else:
//! positions that keep their meaning when a flank is re-measured, and the
//! start-codon-relative numbering biology already uses. [`Coord`] is that
//! spelling, and [`Layout`] resolves it — at parse time, once, so nothing
//! downstream carries two coordinate systems.
//!
//! # The two readings a signed coordinate could have
//!
//! A positive index counts forward from a region's 5′ end and a negative one
//! counts **backward from that same region's 3′ end**, with `-1` naming the
//! region's last nucleotide. One rule for all three regions, and in the 5′ UTR
//! it is exactly the numbering a Kozak consensus is quoted in: the six
//! nucleotides immediately upstream of the start codon are `5utr:-6 … 5utr:-1`.
//!
//! The rejected reading is "negative counts backward from the *molecule's* 3′
//! end regardless of region", which agrees with this one only for `3utr` and
//! would put `5utr:-6` past the CDS.
//!
//! Index `0` names nothing under either reading and is refused rather than
//! silently taken as `1` or as the last position.

use std::fmt;

/// The three lengths a whole mRNA is divided into — everything needed to turn a
/// [`Coord`] into an absolute position.
///
/// Held in nucleotides. The CDS length is `3 ×` the protein length; a caller
/// with a protein and two flanks has all three without folding anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Nucleotides in the 5′ UTR. Zero when there is none.
    pub utr5: u32,
    /// Nucleotides in the coding sequence — three per residue, stop excluded.
    pub cds: u32,
    /// Nucleotides in the 3′ UTR. Zero when there is none.
    pub utr3: u32,
}

/// Which of the three regions a [`Coord`] is written against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordRegion {
    /// The whole mRNA, 5′ end to 3′ end.
    Mrna,
    /// The 5′ UTR alone. Negative indices are the start-codon-relative
    /// numbering: `-1` is the nucleotide immediately before the `A` of `AUG`.
    Utr5,
    /// The coding sequence alone. `cds:1` is the `A` of the start codon.
    Cds,
    /// The 3′ UTR alone. `3utr:1` is the first nucleotide past the CDS.
    Utr3,
}

impl CoordRegion {
    /// The prefix this region is written with, without the separating colon.
    /// [`CoordRegion::Mrna`] has none — a bare index is whole-mRNA.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Mrna => "",
            Self::Utr5 => "5utr",
            Self::Cds => "cds",
            Self::Utr3 => "3utr",
        }
    }

    fn parse_prefix(s: &str) -> Option<Self> {
        match s {
            "mrna" => Some(Self::Mrna),
            "5utr" | "utr5" => Some(Self::Utr5),
            "cds" => Some(Self::Cds),
            "3utr" | "utr3" => Some(Self::Utr3),
            _ => None,
        }
    }
}

/// A caller-written position: a region and a **1-based signed** index into it.
///
/// Resolved to an absolute 0-based position by [`Layout::resolve`], which is the
/// only thing that ever reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord {
    /// The region the index counts within.
    pub region: CoordRegion,
    /// 1-based from the region's 5′ end when positive; `-1`-based from its 3′
    /// end when negative. Never zero — [`Coord::parse`] refuses it.
    pub index: i64,
}

impl Coord {
    /// A whole-mRNA coordinate at 1-based `index`.
    pub fn mrna(index: i64) -> Self {
        Self {
            region: CoordRegion::Mrna,
            index,
        }
    }

    /// Parse `<region>:<index>` or a bare `<index>` (whole-mRNA).
    ///
    /// `5utr` / `utr5`, `cds`, `3utr` / `utr3` and `mrna` name the regions; the
    /// index is 1-based and signed, and `0` is refused.
    pub fn parse(s: &str) -> Result<Self, CoordError> {
        let (region, digits) = match s.rsplit_once(':') {
            Some((prefix, rest)) => (
                CoordRegion::parse_prefix(&prefix.to_ascii_lowercase())
                    .ok_or_else(|| CoordError::UnknownRegion(prefix.to_owned()))?,
                rest,
            ),
            None => (CoordRegion::Mrna, s),
        };
        let index: i64 = digits
            .parse()
            .map_err(|_| CoordError::NotAnIndex(digits.to_owned()))?;
        if index == 0 {
            return Err(CoordError::ZeroIndex);
        }
        Ok(Self { region, index })
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.region {
            CoordRegion::Mrna => write!(f, "{}", self.index),
            r => write!(f, "{}:{}", r.prefix(), self.index),
        }
    }
}

/// Why a written coordinate could not become an absolute position.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CoordError {
    /// The prefix before the colon is not one of the four region names.
    #[error("unknown coordinate region {0:?} — expected 5utr, cds, 3utr or mrna")]
    UnknownRegion(String),
    /// The text after the colon is not a signed integer.
    #[error("coordinate index {0:?} is not an integer")]
    NotAnIndex(String),
    /// Index `0`, which names no nucleotide under a 1-based system.
    #[error(
        "coordinate index 0 has no meaning — indices are 1-based, and -1 is the last position"
    )]
    ZeroIndex,
    /// The index resolves outside the region it is written against.
    #[error("{coord} is outside the {region} ({len} nt)")]
    OutOfRange {
        /// The coordinate as written.
        coord: Coord,
        /// The region name it was written against.
        region: &'static str,
        /// How many nucleotides that region has.
        len: u32,
    },
}

impl Layout {
    /// The layout of a design over `protein_len` residues between two flanks.
    pub fn new(utr5: u32, protein_len: u32, utr3: u32) -> Self {
        Self {
            utr5,
            cds: 3 * protein_len,
            utr3,
        }
    }

    /// Total nucleotides in the mRNA.
    pub fn len(&self) -> u32 {
        self.utr5 + self.cds + self.utr3
    }

    /// Whether the mRNA has no positions at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The first CDS position, absolute and 0-based.
    pub fn cds_start(&self) -> u32 {
        self.utr5
    }

    /// One past the last CDS position, absolute and 0-based.
    pub fn cds_end(&self) -> u32 {
        self.utr5 + self.cds
    }

    /// The absolute 0-based position `coord` names, or why it names none.
    pub fn resolve(&self, coord: Coord) -> Result<u32, CoordError> {
        let (base, len, name) = match coord.region {
            CoordRegion::Mrna => (0, self.len(), "mRNA"),
            CoordRegion::Utr5 => (0, self.utr5, "5′ UTR"),
            CoordRegion::Cds => (self.cds_start(), self.cds, "CDS"),
            CoordRegion::Utr3 => (self.cds_end(), self.utr3, "3′ UTR"),
        };
        // Positive counts forward from the region's 5′ end, negative back from
        // its 3′ end; both land in `0 .. len` or the coordinate names nothing.
        let offset = if coord.index > 0 {
            coord.index - 1
        } else {
            len as i64 + coord.index
        };
        if offset < 0 || offset >= len as i64 {
            return Err(CoordError::OutOfRange {
                coord,
                region: name,
                len,
            });
        }
        Ok(base + offset as u32)
    }

    /// How an absolute position reads back in region-qualified form — for error
    /// messages, which are the one place a caller meets a position they did not
    /// write themselves.
    pub fn describe(&self, pos: u32) -> String {
        if pos < self.cds_start() {
            format!("5utr:{}", pos as i64 - self.cds_start() as i64)
        } else if pos < self.cds_end() {
            format!("cds:{}", pos - self.cds_start() + 1)
        } else {
            format!("3utr:{}", pos - self.cds_end() + 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 12 nt of 5′ UTR, a 5-residue CDS, 9 nt of 3′ UTR.
    fn layout() -> Layout {
        Layout::new(12, 5, 9)
    }

    #[test]
    fn a_bare_index_is_whole_mrna_and_one_based() {
        let l = layout();
        assert_eq!(l.resolve(Coord::parse("1").unwrap()), Ok(0));
        assert_eq!(l.resolve(Coord::parse("36").unwrap()), Ok(35));
        assert_eq!(l.resolve(Coord::parse("mrna:2").unwrap()), Ok(1));
    }

    /// The Kozak spelling the region-qualified form exists for: `5utr:-6 …
    /// 5utr:-1` are the six nucleotides immediately upstream of the start codon.
    #[test]
    fn a_negative_five_prime_index_counts_back_from_the_start_codon() {
        let l = layout();
        for (k, want) in [(-1, 11), (-6, 6)] {
            assert_eq!(
                l.resolve(Coord::parse(&format!("5utr:{k}")).unwrap()),
                Ok(want),
                "5utr:{k}"
            );
        }
        // …and the last of them is immediately before cds:1.
        assert_eq!(l.resolve(Coord::parse("cds:1").unwrap()), Ok(12));
    }

    /// The same rule in every region, which is what makes it one reading rather
    /// than a 5′-UTR special case.
    #[test]
    fn negative_counts_back_from_each_regions_own_three_prime_end() {
        let l = layout();
        assert_eq!(l.resolve(Coord::parse("cds:-1").unwrap()), Ok(26));
        assert_eq!(l.resolve(Coord::parse("3utr:-1").unwrap()), Ok(35));
        assert_eq!(l.resolve(Coord::parse("3utr:1").unwrap()), Ok(27));
    }

    #[test]
    fn out_of_range_names_the_region_and_its_length() {
        let l = layout();
        let err = l.resolve(Coord::parse("5utr:-13").unwrap()).unwrap_err();
        assert!(matches!(
            err,
            CoordError::OutOfRange {
                region: "5′ UTR",
                len: 12,
                ..
            }
        ));
        assert!(l.resolve(Coord::parse("cds:16").unwrap()).is_err());
    }

    /// A CDS-only design is the degenerate case D4 asks for: the coordinate
    /// system collapses to CDS-relative and whole-mRNA indices coincide with it.
    #[test]
    fn a_cds_only_layout_makes_mrna_and_cds_coordinates_agree() {
        let l = Layout::new(0, 5, 0);
        for k in 1..=15 {
            assert_eq!(
                l.resolve(Coord::parse(&format!("cds:{k}")).unwrap()),
                l.resolve(Coord::parse(&k.to_string()).unwrap()),
            );
        }
        assert!(l.resolve(Coord::parse("5utr:-1").unwrap()).is_err());
    }

    #[test]
    fn zero_and_junk_are_refused_rather_than_guessed_at() {
        assert_eq!(Coord::parse("cds:0"), Err(CoordError::ZeroIndex));
        assert!(matches!(
            Coord::parse("intron:4"),
            Err(CoordError::UnknownRegion(_))
        ));
        assert!(matches!(
            Coord::parse("cds:x"),
            Err(CoordError::NotAnIndex(_))
        ));
    }

    #[test]
    fn describe_round_trips_through_resolve() {
        let l = layout();
        for pos in 0..l.len() {
            let back = Coord::parse(&l.describe(pos)).expect("describe writes a parseable coord");
            assert_eq!(l.resolve(back), Ok(pos), "position {pos}");
        }
    }

    #[test]
    fn display_round_trips_through_parse() {
        for s in ["4", "cds:12", "5utr:-6", "3utr:5"] {
            assert_eq!(Coord::parse(s).unwrap().to_string(), s);
        }
    }
}
