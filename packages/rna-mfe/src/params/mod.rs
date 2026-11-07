//! Turner 2004 parameter loading.
//!
//! The built-in table is `data/turner2004.params`: the Turner 2004 free
//! energies in this crate's canonical layout (the output of
//! [`Params::dump_canonical`]), compiled into the binary.
//! [`Params::from_file`] loads a table from a file instead, in that layout
//! or in the `.par` format.
//!
//! # Coverage
//!
//! - **Read** (energy-only, 37°C): `stack`, `mismatch_hairpin`,
//!   `mismatch_interior`, `mismatch_interior_1n`, `mismatch_interior_23`,
//!   `int11`, `int21`, `int22`, `hairpin`, `bulge`, `interior`,
//!   `ML_params`, `NINIO`, `Misc`, `Triloops`, `Tetraloops`, `Hexaloops`.
//! - **Skipped** (dangles=0 only — a codon-level DP cell does not record
//!   the flanking codon a dangle energy would read): `dangle3`, `dangle5`,
//!   `mismatch_multi`, `mismatch_exterior`. Sections are passed over without
//!   being stored.
//! - **Skipped** (37°C only): all `*_enthalpies` sections. The Misc and
//!   ML_params blocks interleave 37°C and dH values, so the parser keeps both
//!   for round-trip but only the 37°C entries are read downstream.
//!
//! # Round-trip
//!
//! [`Params::dump_canonical`] emits the read sections in the order above,
//! in the canonical layout. The parser is its own inverse: parsing the dump
//! yields a `Params` equal to the input, and the built-in table is a fixed
//! point of parse-then-dump. `tests/round_trip.rs` checks both.

mod dump;
mod lex;
mod parse;
mod special;
mod tables;

pub use parse::ParseError;
pub use special::SpecialHairpinTable;
pub use tables::{
    BasePair, Int11Key, Int11Table, Int21Key, Int21Table, Int22Key, Int22Table, MiscParams,
    MultiLoopParams, NinioParams, Nuc5, SizeTable, Stack37, Triplet37, INF,
};

/// Parsed Turner 2004 parameters at 37°C.
///
/// All energies are integer dekacal/mol (kcal/mol × 100). The built-in
/// table is loaded with [`Params::load_default`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Params {
    pub stack: Stack37,
    pub mismatch_hairpin: Triplet37,
    pub mismatch_interior: Triplet37,
    pub mismatch_interior_1n: Triplet37,
    pub mismatch_interior_23: Triplet37,
    pub int11: Int11Table,
    pub int21: Int21Table,
    pub int22: Int22Table,
    pub hairpin: SizeTable,
    pub bulge: SizeTable,
    pub interior: SizeTable,
    pub ml: MultiLoopParams,
    pub ninio: NinioParams,
    pub misc: MiscParams,
    pub triloops: SpecialHairpinTable,
    pub tetraloops: SpecialHairpinTable,
    pub hexaloops: SpecialHairpinTable,
    /// Values derived from the tables on first use, such as
    /// [`crate::dp`]'s interior-loop lower bounds; never part of equality.
    pub derived: Derived,
}

/// Values a fold derives from [`Params`] once and reuses: filled on first
/// use, cloned with the parameters, and ignored by equality, since they
/// follow from the tables.
#[derive(Debug, Clone, Default)]
pub struct Derived {
    /// Per `(u1, u2)` unpaired lengths, a lower bound on every stack, bulge
    /// or interior loop with those lengths ([`crate::dp`]).
    pub interior_bounds: std::sync::OnceLock<Vec<i32>>,
}

impl PartialEq for Derived {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Derived {}

const TURNER_2004: &str = include_str!("../../data/turner2004.params");

impl Params {
    /// The built-in Turner 2004 table (compiled in).
    ///
    /// Returns `Err` only if the embedded table is malformed — i.e. never,
    /// barring a deliberate corruption of `data/turner2004.params`. The
    /// `Result` exists so the same code path can parse arbitrary inputs
    /// in tests without `unwrap()` litter.
    pub fn load_default() -> Result<Self, ParseError> {
        Self::parse(TURNER_2004)
    }

    /// Load a parameter table from the file at `path`, written either in the
    /// canonical layout [`Self::dump_canonical`] writes or in the `.par`
    /// format.
    ///
    /// # Errors
    ///
    /// [`ParseError::Read`] when the file cannot be read, and the parser's
    /// errors when its contents are not a parameter table.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, ParseError> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|source| ParseError::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::parse(&source)
    }

    /// Parse a parameter table from a string, in either layout
    /// [`Self::from_file`] reads.
    pub fn parse(source: &str) -> Result<Self, ParseError> {
        parse::parse_params(source)
    }

    /// Emit a canonical dump of the parsed parameters. The format
    /// preserves the section order and includes only the sections this
    /// crate reads (skipped sections are omitted). The dump is its own
    /// input — `Self::parse` round-trips through it.
    pub fn dump_canonical(&self) -> String {
        dump::dump_canonical(self)
    }
}
