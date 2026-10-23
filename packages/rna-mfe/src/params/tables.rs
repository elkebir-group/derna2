//! Typed storage for the Turner 2004 lookup tables.
//!
//! Index conventions match ViennaRNA's `rna_turner2004.par` layout, with
//! the implicit "no pair" / "no nucleotide" slot at index 0 dropped.
//!
//! - [`BasePair`] (7): `CG, GC, GU, UG, AU, UA, NN` — same order as the
//!   `# stack` block's columns.
//! - [`Nuc5`] (5): `N, A, C, G, U` — index 0 is Vienna's "any/unknown"
//!   placeholder used in mismatch tables.
//!
//! Multi-dimensional tables store row-major flat arrays internally;
//! typed accessors compute the offset. This keeps the storage layout
//! aligned with the on-disk file (round-trip parsing/dumping is trivial)
//! and avoids the deep-nested-array types that hurt code-health
//! complexity scores.

/// `INF` token value used in size tables for sizes below the minimum.
///
/// Vienna stores it as `INT_MAX/10` (compiled-in macro). On round-trip
/// the dumper recognises this exact value and emits the literal token `INF` to
/// match the upstream file's textual representation.
pub const INF: i32 = 10_000_000;

/// Number of canonical base pairs covered by the parameter tables.
pub const N_PAIRS: usize = 7;

/// Index space for nucleotide-keyed dimensions in mismatch / interior
/// loop tables. Index 0 is Vienna's "any" sentinel (`N`); 1..=4 are
/// `A, C, G, U`.
pub const N_NUCS: usize = 5;

/// `int22` covers only canonical-canonical pair combinations (dropping
/// the `NN` row + col) and `A/C/G/U` mismatches (dropping the `N`
/// sentinel) — see ViennaRNA `src/ViennaRNA/params/io.c` `int22_dim`
/// at v2.7.0 line 64–72.
pub const N_PAIRS_22: usize = 6;
pub const N_NUCS_22: usize = 4;

/// Length of the size-keyed energy tables (`hairpin`, `bulge`,
/// `interior`). Sizes 0..=30 are stored explicitly; ≥31 uses the
/// `lxc` logarithmic extrapolation in [`crate::loops`].
pub const SIZE_TABLE_LEN: usize = 31;

pub const STACK_LEN: usize = N_PAIRS * N_PAIRS;
pub const TRIPLET_LEN: usize = N_PAIRS * N_NUCS * N_NUCS;
pub const INT11_LEN: usize = N_PAIRS * N_PAIRS * N_NUCS * N_NUCS;
pub const INT21_LEN: usize = N_PAIRS * N_PAIRS * N_NUCS * N_NUCS * N_NUCS;
pub const INT22_LEN: usize =
    N_PAIRS_22 * N_PAIRS_22 * N_NUCS_22 * N_NUCS_22 * N_NUCS_22 * N_NUCS_22;

/// Canonical base-pair index for the parameter file's row/column order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasePair {
    CG = 0,
    GC = 1,
    GU = 2,
    UG = 3,
    AU = 4,
    UA = 5,
    NN = 6,
}

impl BasePair {
    pub const ALL: [BasePair; N_PAIRS] = [
        BasePair::CG,
        BasePair::GC,
        BasePair::GU,
        BasePair::UG,
        BasePair::AU,
        BasePair::UA,
        BasePair::NN,
    ];

    pub const fn as_index(self) -> usize {
        self as usize
    }

    /// Reverse the orientation of the pair: CG ↔ GC, GU ↔ UG, AU ↔ UA,
    /// NN unchanged. Used by stack / internal-loop lookups: Vienna's
    /// `eval_internal.c` indexes the inner pair as
    /// `vrna_get_ptype_md(S2[l], S2[k], md)` — i.e., from the loop side
    /// rather than sequence-order — so callers reverse the inner pair
    /// before consulting `stack` / `int*` / `mismatch_*` tables.
    pub const fn reversed(self) -> BasePair {
        match self {
            BasePair::CG => BasePair::GC,
            BasePair::GC => BasePair::CG,
            BasePair::GU => BasePair::UG,
            BasePair::UG => BasePair::GU,
            BasePair::AU => BasePair::UA,
            BasePair::UA => BasePair::AU,
            BasePair::NN => BasePair::NN,
        }
    }

    /// True for pairs that incur the `Misc.terminal_au_37` penalty when
    /// they end an interior/exterior/multi-loop branch — i.e., everything
    /// other than `CG` / `GC`. Mirrors Vienna's `type > 2` predicate
    /// (Vienna pair indices start at 1; `as_index() >= 2` is the
    /// equivalent on the 0-based encoding).
    pub const fn applies_terminal_au(self) -> bool {
        self.as_index() >= BasePair::GU.as_index()
    }

    /// Index in the reduced-pair space used by `int22` (which drops the
    /// `NN` row + column on disk). Returns `None` for `NN`.
    pub const fn as_int22_index(self) -> Option<usize> {
        match self {
            BasePair::NN => None,
            other => Some(other.as_index()),
        }
    }
}

/// Nucleotide index for mismatch / interior loop dimensions. Index 0
/// is Vienna's "any" sentinel (`N`); 1..=4 are `A, C, G, U`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Nuc5 {
    N = 0,
    A = 1,
    C = 2,
    G = 3,
    U = 4,
}

impl Nuc5 {
    pub const ALL: [Nuc5; N_NUCS] = [Nuc5::N, Nuc5::A, Nuc5::C, Nuc5::G, Nuc5::U];

    pub const fn as_index(self) -> usize {
        self as usize
    }
}

/// 7×7 stack-energy table indexed by (closing pair, inner pair).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stack37 {
    pub flat: [i32; STACK_LEN],
}

impl Stack37 {
    pub fn get(&self, closing: BasePair, inner: BasePair) -> i32 {
        self.flat[closing.as_index() * N_PAIRS + inner.as_index()]
    }
}

/// 7×5×5 mismatch table indexed by (closing pair, 5' mismatch, 3' mismatch).
/// Used for `mismatch_hairpin` / `mismatch_interior` / `mismatch_interior_1n`
/// / `mismatch_interior_23`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Triplet37 {
    pub flat: [i32; TRIPLET_LEN],
}

impl Triplet37 {
    pub fn get(&self, closing: BasePair, n5: Nuc5, n3: Nuc5) -> i32 {
        let idx = closing.as_index() * N_NUCS * N_NUCS + n5.as_index() * N_NUCS + n3.as_index();
        self.flat[idx]
    }
}

/// 7×7×5×5 internal-loop 1×1 table indexed by
/// (closing, inner, mismatch_5', mismatch_3'). The inner pair is
/// stored in reversed orientation (Vienna's loop-side view); callers
/// pass sequence-order pairs through [`BasePair::reversed`] first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int11Table {
    pub flat: Box<[i32; INT11_LEN]>,
}

/// Lookup key for [`Int11Table::get`]. `inner_reversed` carries the
/// inner pair in Vienna's loop-side orientation, matching the layout
/// of the on-disk `int11` block.
#[derive(Debug, Clone, Copy)]
pub struct Int11Key {
    pub outer: BasePair,
    pub inner_reversed: BasePair,
    pub m5: Nuc5,
    pub m3: Nuc5,
}

impl Int11Table {
    pub fn get(&self, key: Int11Key) -> i32 {
        let idx = ((key.outer.as_index() * N_PAIRS + key.inner_reversed.as_index()) * N_NUCS
            + key.m5.as_index())
            * N_NUCS
            + key.m3.as_index();
        self.flat[idx]
    }
}

/// 7×7×5×5×5 internal-loop 2×1 table. Innermost three axes are the
/// three loop nucleotide positions; ordering follows Vienna's
/// `rd_5dim` walk in `src/ViennaRNA/params/io.c` (v2.7.0 line 957–997)
/// and `eval_internal.c::vrna_E_internal` (`int21[type][type_2][si1][sq1][sj1]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int21Table {
    pub flat: Box<[i32; INT21_LEN]>,
}

/// Lookup key for [`Int21Table::get`]. The `(1, 2)` and `(2, 1)`
/// shapes share a code path by binding `pair_a` / `pair_b` to "the
/// pair on the unpaired-singleton side" / "the pair on the
/// two-unpaired side" respectively (per
/// `eval_internal.c::vrna_E_internal` `case 1` `nl == 2`).
#[derive(Debug, Clone, Copy)]
pub struct Int21Key {
    pub pair_a: BasePair,
    pub pair_b: BasePair,
    pub single: Nuc5,
    pub double_inner: Nuc5,
    pub double_outer: Nuc5,
}

impl Int21Table {
    /// `int21[pair_a][pair_b][single][double_inner][double_outer]`.
    /// See [`Int21Key`] for axis-binding semantics.
    pub fn get(&self, key: Int21Key) -> i32 {
        let idx = (((key.pair_a.as_index() * N_PAIRS + key.pair_b.as_index()) * N_NUCS
            + key.single.as_index())
            * N_NUCS
            + key.double_inner.as_index())
            * N_NUCS
            + key.double_outer.as_index();
        self.flat[idx]
    }
}

/// 6×6×4×4×4×4 internal-loop 2×2 table. The reduced dimensions reflect
/// the upstream file's omission of the `NN` row + column and of the
/// `N` nucleotide slot — Vienna fills the missing entries on load via
/// `update_nst`. Axis order: `int22[outer][inner_reversed][si1][sp1][sq1][sj1]`
/// per `eval_internal.c::vrna_E_internal` (the 2×2 case).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int22Table {
    pub flat: Box<[i32; INT22_LEN]>,
}

/// Lookup key for [`Int22Table::get`] using the reduced-range indices
/// (`pair` in `0..6`, nucleotides in `0..4`). The `mismatches` array
/// holds `[si1, sp1, sq1, sj1]` per Vienna's
/// `int22[type][type_2][si1][sp1][sq1][sj1]` axis order.
#[derive(Debug, Clone, Copy)]
pub struct Int22Key {
    pub outer: usize,
    pub inner_reversed: usize,
    pub mismatches: [usize; 4],
}

impl Int22Table {
    /// Lookup with reduced-range indices. Callers pre-screen via
    /// [`BasePair::as_int22_index`] / the reduced `Nuc5 → 0..3`
    /// mapping; non-canonical inputs lie outside the table footprint.
    pub fn get(&self, key: Int22Key) -> i32 {
        let [si1, sp1, sq1, sj1] = key.mismatches;
        let idx = ((((key.outer * N_PAIRS_22 + key.inner_reversed) * N_NUCS_22 + si1) * N_NUCS_22
            + sp1)
            * N_NUCS_22
            + sq1)
            * N_NUCS_22
            + sj1;
        self.flat[idx]
    }
}

/// 31-entry size-keyed table for hairpin / bulge / interior sizes 0..=30.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeTable {
    pub values: [i32; SIZE_TABLE_LEN],
}

/// Multi-loop parameters at dangles=0.
///
/// Field order resolved from ViennaRNA 2.7.0
/// `src/ViennaRNA/params/io.c` lines 654–664 (`case ML:`):
/// `[ML_BASE37, ML_BASEdH, ML_closing37, ML_closingdH, ML_intern37,
/// ML_interndH]`. The dump comment in the same file (line 1589) is:
/// `/*\t    cu\t cu_dH\t    cc\t cc_dH\t    ci\t ci_dH  */`. So:
/// - `cu` = `base` = per-unpaired nucleotide
/// - `cc` = `closing` = multi-loop initiation
/// - `ci` = `intern` = per-branch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiLoopParams {
    pub base_37: i32,
    pub base_dh: i32,
    pub closing_37: i32,
    pub closing_dh: i32,
    pub intern_37: i32,
    pub intern_dh: i32,
}

/// NINIO parameters: per ViennaRNA 2.7.0 `src/ViennaRNA/params/io.c`
/// lines 666–673 (`case NIN:`), the three values are
/// `[ninio37, niniodH, MAX_NINIO]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NinioParams {
    pub ninio_37: i32,
    pub ninio_dh: i32,
    pub max_ninio: i32,
}

/// Misc block — the four interleaved 37°C/dH values for `DuplexInit`
/// and `TerminalAU`.
///
/// **OQ2 resolution.** Field order is from ViennaRNA 2.7.0
/// `src/ViennaRNA/params/io.c` `case MISC:` (lines 675–684):
///
/// ```text
/// DuplexInit37  = values[0];   // 410
/// DuplexInitdH  = values[1];   // 360
/// TerminalAU37  = values[2];   // 50
/// TerminalAUdH  = values[3];   // 370
/// ```
///
/// Note: Vienna's *dumper* (same file, line 1606–1614) emits a 6-token
/// row that also includes `lxc37` — but `lxc37` is a hardcoded
/// `default.c` constant, **not** parsed from the file. The on-disk
/// Misc block contains only these four integers; interior-loop
/// size extrapolation reads `lxc` from a separate hardcoded constant
/// (see [`super::super::loops::LXC_37`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiscParams {
    pub duplex_init_37: i32,
    pub duplex_init_dh: i32,
    pub terminal_au_37: i32,
    pub terminal_au_dh: i32,
}
