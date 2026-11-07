//! Per-loop Turner 2004 free-energy evaluators at `dangles=0`.
//!
//! Each submodule owns one motif kind: [`hairpin`], [`stack`], [`bulge`],
//! [`internal`], [`multi`], [`exterior`]. Special-hairpin bonus lookup
//! (`Triloops` / `Tetraloops` / `Hexaloops`) lives in
//! [`special_hairpin`] and is invoked from inside [`hairpin_energy`]
//! when the loop size is 3, 4, or 6.
//!
//! Energies flow as integer dekacalories (`Energy(i32)`) internally, so
//! sums and roundings are exact integer arithmetic; convert to `f64 kcal/mol`
//! at the public boundary via [`Energy::to_kcal_mol`].

use crate::params::{BasePair, Nuc5};

pub mod bulge;
pub mod exterior;
pub mod hairpin;
pub mod internal;
pub mod multi;
mod special_hairpin;
pub mod stack;

pub use bulge::{bulge_energy, BulgeLoop};
pub use exterior::exterior_loop_energy;
pub use hairpin::{hairpin_energy, hairpin_energy_scalar, HairpinLoop};
pub use internal::{internal_energy, internal_energy_scalar, InternalLoop, InternalLoopScalar};
pub use multi::{multi_loop_energy, MultiLoop};
pub use stack::{stack_energy, StackPair};

/// Logarithmic factor for size-table extrapolation of internal /
/// hairpin / bulge loops with size > 30: `lxc37 = 107.856` (kcal/mol
/// per ln-unit; integer dekacal after multiply), a compiled-in constant
/// of the Turner 2004 model. It is *not* parsed
/// from the parameter file; see [`crate::params::MiscParams`] for
/// context on where the parser draws the file/constant boundary.
pub const LXC_37: f64 = 107.856;

/// Largest size with an explicit entry in the size-keyed tables
/// (`hairpin`, `bulge`, `interior`). Sizes > 30 use [`LXC_37`].
pub const MAX_TABULATED_SIZE: usize = 30;

/// Loop sizes [`extrapolation`] tabulates.
const EXTRAPOLATED_SIZES: usize = 8192;

/// `LXC_37 · ln(size / MAX_TABULATED_SIZE)` truncated to dekacalories: the
/// growth of a hairpin, bulge or interior loop past the tabulated sizes.
/// Computed once per size up to [`EXTRAPOLATED_SIZES`] and read from a table,
/// so a fold that meets long loops does not take a logarithm per loop; the
/// table holds exactly what the formula gives. The formula is kept out of
/// line: inlined, its logarithm is speculatable and gets hoisted above the
/// table lookup, computed for every loop and then discarded.
pub(crate) fn extrapolation(size: usize) -> i32 {
    #[cold]
    #[inline(never)]
    fn formula(size: usize) -> i32 {
        (LXC_37 * (size as f64 / MAX_TABULATED_SIZE as f64).ln()) as i32
    }
    static TABLE: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| (0..EXTRAPOLATED_SIZES).map(|n| formula(n.max(1))).collect());
    table.get(size).copied().unwrap_or_else(|| formula(size))
}

/// Free energy in integer dekacalories (10 × kcal/mol). Construct via
/// per-motif evaluators; convert at the boundary to floating-point kcal/mol when comparing against fixture
/// records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Energy(pub i32);

impl Energy {
    pub const ZERO: Energy = Energy(0);

    /// Convert to kcal/mol as a floating-point number. The `i32` →
    /// `f64` cast is lossless within the representable range; the
    /// `/100.0` divide introduces at most one ULP of slop, which sits
    /// well under the `0.001 kcal/mol` fixture tolerance.
    pub fn to_kcal_mol(self) -> f64 {
        f64::from(self.0) / 100.0
    }

    pub const fn dekacal(self) -> i32 {
        self.0
    }
}

impl core::ops::Add for Energy {
    type Output = Energy;

    fn add(self, rhs: Energy) -> Energy {
        Energy(self.0 + rhs.0)
    }
}

impl core::ops::AddAssign for Energy {
    fn add_assign(&mut self, rhs: Energy) {
        self.0 += rhs.0;
    }
}

impl core::iter::Sum for Energy {
    fn sum<I: Iterator<Item = Energy>>(iter: I) -> Energy {
        Energy(iter.map(|e| e.0).sum())
    }
}

/// Standard four nucleotides. Index 0..3 maps to A, C, G, U for the
/// reduced-axis tables (`int22`'s 4-nucleotide axes); the
/// [`Nucleotide::to_nuc5`] conversion lifts to the 5-slot encoding the
/// other mismatch tables use (with the `N=0` sentinel reserved).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Nucleotide {
    A = 0,
    C = 1,
    G = 2,
    U = 3,
}

impl Nucleotide {
    /// Map from an ASCII byte (`b'A' | b'C' | b'G' | b'U'`); returns
    /// `None` for any other byte. Callers handling fixture sequences
    /// should treat `None` as a malformed input.
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            b'A' => Some(Nucleotide::A),
            b'C' => Some(Nucleotide::C),
            b'G' => Some(Nucleotide::G),
            b'U' => Some(Nucleotide::U),
            _ => None,
        }
    }

    /// Lift to the 5-slot encoding used by `mismatch_*` / `int11` /
    /// `int21`, where index 0 is reserved for the `N` sentinel.
    pub const fn to_nuc5(self) -> Nuc5 {
        match self {
            Nucleotide::A => Nuc5::A,
            Nucleotide::C => Nuc5::C,
            Nucleotide::G => Nuc5::G,
            Nucleotide::U => Nuc5::U,
        }
    }

    /// 4-slot index used by `int22`'s reduced axes.
    pub const fn as_int22_index(self) -> usize {
        self as usize
    }
}

/// Convert a sequence-order nucleotide pair to the canonical
/// [`BasePair`] enum. Returns `BasePair::NN` for non-canonical
/// combinations (e.g., `AA`, `GG`); fixture inputs only carry
/// canonical pairs.
#[inline]
pub fn pair_from_nucleotides(a: Nucleotide, b: Nucleotide) -> BasePair {
    match (a, b) {
        (Nucleotide::C, Nucleotide::G) => BasePair::CG,
        (Nucleotide::G, Nucleotide::C) => BasePair::GC,
        (Nucleotide::G, Nucleotide::U) => BasePair::GU,
        (Nucleotide::U, Nucleotide::G) => BasePair::UG,
        (Nucleotide::A, Nucleotide::U) => BasePair::AU,
        (Nucleotide::U, Nucleotide::A) => BasePair::UA,
        _ => BasePair::NN,
    }
}

#[cfg(test)]
mod extrapolation_tests {
    use super::*;

    /// The table holds exactly what the formula gives, inside it and past it.
    #[test]
    fn tabulated_extrapolation_is_the_formula() {
        let sizes = (MAX_TABULATED_SIZE + 1..20_000)
            .step_by(7)
            .chain([EXTRAPOLATED_SIZES - 1, EXTRAPOLATED_SIZES]);
        for size in sizes {
            let want = (LXC_37 * (size as f64 / MAX_TABULATED_SIZE as f64).ln()) as i32;
            assert_eq!(extrapolation(size), want, "size {size}");
        }
    }
}
