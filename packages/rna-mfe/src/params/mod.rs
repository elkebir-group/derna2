//! Turner 2004 parameter loading.
//!
//! Parses ViennaRNA's `rna_turner2004.par` (vendored at `data/`) into
//! typed lookup tables.
//!
//! # Coverage
//!
//! - **Parsed** (energy-only, 37°C): `stack`, `mismatch_hairpin`,
//!   `mismatch_interior`, `mismatch_interior_1n`, `mismatch_interior_23`,
//!   `int11`, `int21`, `int22`, `hairpin`, `bulge`, `interior`,
//!   `ML_params`, `NINIO`, `Misc`, `Triloops`, `Tetraloops`, `Hexaloops`.
//! - **Skipped** (dangles=0 only — a codon-level DP cell does not record
//!   the flanking codon a dangle energy would read): `dangle3`, `dangle5`, `mismatch_multi`,
//!   `mismatch_exterior`. Sections are passed over without being stored.
//! - **Skipped** (37°C only): all `*_enthalpies` sections. The Misc and
//!   ML_params blocks interleave 37°C and dH values, so the parser keeps both
//!   for round-trip but only consume the 37°C entries downstream.
//!
//! # Round-trip
//!
//! [`Params::dump_canonical`] emits the parsed sections in upstream order
//! using a simplified canonical format. The parser is its own inverse:
//! parsing the dump yields a `Params` equal to the input. See the
//! integration test in `tests/round_trip.rs` for the load-bearing
//! validation against the vendored upstream file.

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
/// All energies are integer dekacalories per Vienna's internal
/// representation (10 × kcal/mol). Loaded once via [`Params::load_default`]
/// from a copy of `rna_turner2004.par` baked into the binary.
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

const RNA_TURNER_2004_PAR: &str = include_str!("../../data/rna_turner2004.par");

impl Params {
    /// Parse the vendored `rna_turner2004.par` (baked in at compile time).
    ///
    /// Returns `Err` only if the embedded file is malformed — i.e. never,
    /// barring a deliberate corruption of `data/rna_turner2004.par`. The
    /// `Result` exists so the same code path can parse arbitrary inputs
    /// in tests without `unwrap()` litter.
    pub fn load_default() -> Result<Self, ParseError> {
        Self::parse(RNA_TURNER_2004_PAR)
    }

    /// Parse a parameter file from a string. Intended for the round-trip
    /// test (`parse(dump_canonical(parse(file))) == parse(file)`); not
    /// part of the production hot path.
    pub fn parse(source: &str) -> Result<Self, ParseError> {
        parse::parse_params(source)
    }

    /// Emit a canonical dump of the parsed parameters. The format
    /// preserves the upstream section order and includes only the
    /// sections this crate parses (skipped sections are omitted). The
    /// dump is its own input — `Self::parse` round-trips through it.
    pub fn dump_canonical(&self) -> String {
        dump::dump_canonical(self)
    }
}
