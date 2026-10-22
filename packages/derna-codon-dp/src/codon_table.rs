//! Canonical amino-acid and codon ordering for the codon DP.
//!
//! [`AMINO_ACIDS`] is the source of truth: AA index, single-letter code, and
//! synonymous codons in the order the DERNA paper / C++ reference expects.
//! Derived tables ([`N_CODON`], [`aa_to_index`], [`index_to_aa`],
//! [`get_codon_index`]) are computed at initialisation time.
//!
//! AA index order matches `docs/reference/_cpp/derna-main/src/default.cpp`
//! (`n_codon` line 106 and `codon_usage[20][6]` line 510):
//! A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, M=9,
//! I=10, L=11, K=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19.
//! Drift here breaks the Kazusa codon-usage loader and the C++ parity tests.
//!
//! One more entry follows the twenty: the stop codons, [`STOP`] = 20, code
//! `*`, as a codon family of their own. It is not an amino acid. It is what
//! lets a lattice carry the stop as a designed slot — one position whose three
//! codons `UAA`, `UAG` and `UGA` the design chooses between — rather than as
//! fixed sequence after the last residue. A stop codon carries no adaptation:
//! CAI proxies tRNA abundance, and stops are read by release factors, so the
//! codon usage table has no stop rows and every stop codon weighs the same.
//! [`aa_to_index`] still refuses `*`: a protein is the twenty, and a stop slot
//! is asked for separately.

use std::sync::OnceLock;

use thiserror::Error;

/// One codon: nucleotide-index triple plus its RNA string.
/// Encoding: A=0, C=1, G=2, U=3 — matches `rna_mfe::Nucleotide`'s `repr(u8)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Codon {
    pub nuc: [u8; 3],
    pub name: &'static str,
}

/// One amino acid: single-letter code, DERNA index, and synonymous codons.
#[derive(Debug, Clone, Copy)]
pub struct AminoAcid {
    pub code: char,
    pub index: usize,
    pub codons: &'static [Codon],
}

// ---------------------------------------------------------------------------
// Static codon data (A=0, C=1, G=2, U=3)
// ---------------------------------------------------------------------------

macro_rules! codon {
    ($name:expr, $a:expr, $b:expr, $c:expr) => {
        Codon {
            nuc: [$a, $b, $c],
            name: $name,
        }
    };
}

static A_CODONS: &[Codon] = &[
    codon!("GCU", 2, 1, 3),
    codon!("GCC", 2, 1, 1),
    codon!("GCA", 2, 1, 0),
    codon!("GCG", 2, 1, 2),
];
static R_CODONS: &[Codon] = &[
    codon!("CGU", 1, 2, 3),
    codon!("CGC", 1, 2, 1),
    codon!("CGA", 1, 2, 0),
    codon!("CGG", 1, 2, 2),
    codon!("AGA", 0, 2, 0),
    codon!("AGG", 0, 2, 2),
];
static N_CODONS: &[Codon] = &[codon!("AAU", 0, 0, 3), codon!("AAC", 0, 0, 1)];
static D_CODONS: &[Codon] = &[codon!("GAU", 2, 0, 3), codon!("GAC", 2, 0, 1)];
static C_CODONS: &[Codon] = &[codon!("UGU", 3, 2, 3), codon!("UGC", 3, 2, 1)];
static Q_CODONS: &[Codon] = &[codon!("CAA", 1, 0, 0), codon!("CAG", 1, 0, 2)];
static E_CODONS: &[Codon] = &[codon!("GAA", 2, 0, 0), codon!("GAG", 2, 0, 2)];
static G_CODONS: &[Codon] = &[
    codon!("GGU", 2, 2, 3),
    codon!("GGC", 2, 2, 1),
    codon!("GGA", 2, 2, 0),
    codon!("GGG", 2, 2, 2),
];
static H_CODONS: &[Codon] = &[codon!("CAU", 1, 0, 3), codon!("CAC", 1, 0, 1)];
static M_CODONS: &[Codon] = &[codon!("AUG", 0, 3, 2)];
static I_CODONS: &[Codon] = &[
    codon!("AUU", 0, 3, 3),
    codon!("AUC", 0, 3, 1),
    codon!("AUA", 0, 3, 0),
];
static L_CODONS: &[Codon] = &[
    codon!("CUU", 1, 3, 3),
    codon!("CUC", 1, 3, 1),
    codon!("CUA", 1, 3, 0),
    codon!("CUG", 1, 3, 2),
    codon!("UUA", 3, 3, 0),
    codon!("UUG", 3, 3, 2),
];
static K_CODONS: &[Codon] = &[codon!("AAA", 0, 0, 0), codon!("AAG", 0, 0, 2)];
static F_CODONS: &[Codon] = &[codon!("UUU", 3, 3, 3), codon!("UUC", 3, 3, 1)];
static P_CODONS: &[Codon] = &[
    codon!("CCU", 1, 1, 3),
    codon!("CCC", 1, 1, 1),
    codon!("CCA", 1, 1, 0),
    codon!("CCG", 1, 1, 2),
];
static S_CODONS: &[Codon] = &[
    codon!("UCU", 3, 1, 3),
    codon!("UCC", 3, 1, 1),
    codon!("UCA", 3, 1, 0),
    codon!("UCG", 3, 1, 2),
    codon!("AGU", 0, 2, 3),
    codon!("AGC", 0, 2, 1),
];
static T_CODONS: &[Codon] = &[
    codon!("ACU", 0, 1, 3),
    codon!("ACC", 0, 1, 1),
    codon!("ACA", 0, 1, 0),
    codon!("ACG", 0, 1, 2),
];
static W_CODONS: &[Codon] = &[codon!("UGG", 3, 2, 2)];
static Y_CODONS: &[Codon] = &[codon!("UAU", 3, 0, 3), codon!("UAC", 3, 0, 1)];
static V_CODONS: &[Codon] = &[
    codon!("GUU", 2, 3, 3),
    codon!("GUC", 2, 3, 1),
    codon!("GUA", 2, 3, 0),
    codon!("GUG", 2, 3, 2),
];
static STOP_CODONS: &[Codon] = &[
    codon!("UAA", 3, 0, 0),
    codon!("UAG", 3, 0, 2),
    codon!("UGA", 3, 2, 0),
];

/// The index of the stop codons' family, past the twenty amino acids.
pub const STOP: usize = 20;

/// The stop family's code: the conventional `*` of a translated stop.
pub const STOP_CODE: char = '*';

/// The amino acids and the stop family: the row count of every table indexed
/// by family.
pub const FAMILIES: usize = 21;

/// Source of truth. Entry order pins AA indices; codon order within each
/// entry pins the row layout in the Kazusa CSV (the Kazusa *H. sapiens* table, transcribed
/// from the DERNA C++ source). The last entry is
/// the stop family ([`STOP`]), which the CSV has no rows for.
pub static AMINO_ACIDS: [AminoAcid; FAMILIES] = [
    AminoAcid {
        code: 'A',
        index: 0,
        codons: A_CODONS,
    },
    AminoAcid {
        code: 'R',
        index: 1,
        codons: R_CODONS,
    },
    AminoAcid {
        code: 'N',
        index: 2,
        codons: N_CODONS,
    },
    AminoAcid {
        code: 'D',
        index: 3,
        codons: D_CODONS,
    },
    AminoAcid {
        code: 'C',
        index: 4,
        codons: C_CODONS,
    },
    AminoAcid {
        code: 'Q',
        index: 5,
        codons: Q_CODONS,
    },
    AminoAcid {
        code: 'E',
        index: 6,
        codons: E_CODONS,
    },
    AminoAcid {
        code: 'G',
        index: 7,
        codons: G_CODONS,
    },
    AminoAcid {
        code: 'H',
        index: 8,
        codons: H_CODONS,
    },
    AminoAcid {
        code: 'M',
        index: 9,
        codons: M_CODONS,
    },
    AminoAcid {
        code: 'I',
        index: 10,
        codons: I_CODONS,
    },
    AminoAcid {
        code: 'L',
        index: 11,
        codons: L_CODONS,
    },
    AminoAcid {
        code: 'K',
        index: 12,
        codons: K_CODONS,
    },
    AminoAcid {
        code: 'F',
        index: 13,
        codons: F_CODONS,
    },
    AminoAcid {
        code: 'P',
        index: 14,
        codons: P_CODONS,
    },
    AminoAcid {
        code: 'S',
        index: 15,
        codons: S_CODONS,
    },
    AminoAcid {
        code: 'T',
        index: 16,
        codons: T_CODONS,
    },
    AminoAcid {
        code: 'W',
        index: 17,
        codons: W_CODONS,
    },
    AminoAcid {
        code: 'Y',
        index: 18,
        codons: Y_CODONS,
    },
    AminoAcid {
        code: 'V',
        index: 19,
        codons: V_CODONS,
    },
    AminoAcid {
        code: STOP_CODE,
        index: STOP,
        codons: STOP_CODONS,
    },
];

/// Number of synonymous codons per AA, indexed by AA index, then the stop
/// family's three.
pub const N_CODON: [usize; FAMILIES] = [
    4, 6, 2, 2, 2, 2, 2, 4, 2, 1, 3, 6, 2, 2, 4, 6, 4, 1, 2, 4, 3,
];

// ---------------------------------------------------------------------------
// Lookup tables (initialised once)
// ---------------------------------------------------------------------------

type CodonIndexMap = std::collections::HashMap<(usize, [u8; 3]), usize>;
type AaIndexMap = std::collections::HashMap<char, usize>;

/// The twenty amino acids' codes. The stop family is left out: a protein never
/// names it, so `*` in one is refused rather than read as a slot.
fn build_aa_index() -> AaIndexMap {
    AMINO_ACIDS[..STOP]
        .iter()
        .map(|aa| (aa.code, aa.index))
        .collect()
}

fn build_codon_index() -> CodonIndexMap {
    let mut map = CodonIndexMap::new();
    for aa in &AMINO_ACIDS {
        for (ci, codon) in aa.codons.iter().enumerate() {
            map.insert((aa.index, codon.nuc), ci);
        }
    }
    map
}

static AA_INDEX: OnceLock<AaIndexMap> = OnceLock::new();
static CODON_INDEX: OnceLock<CodonIndexMap> = OnceLock::new();

fn aa_index() -> &'static AaIndexMap {
    AA_INDEX.get_or_init(build_aa_index)
}

fn codon_index() -> &'static CodonIndexMap {
    CODON_INDEX.get_or_init(build_codon_index)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return the DERNA index for a single-letter amino acid code. `*` is not one:
/// the stop family's index is [`STOP`].
pub fn aa_to_index(code: char) -> Result<usize, AminoAcidError> {
    aa_index()
        .get(&code)
        .copied()
        .ok_or(AminoAcidError::UnknownCode(code))
}

/// Return the single-letter amino acid code for a DERNA index.
pub fn index_to_aa(idx: usize) -> Result<char, AminoAcidError> {
    if idx < AMINO_ACIDS.len() {
        Ok(AMINO_ACIDS[idx].code)
    } else {
        Err(AminoAcidError::IndexOutOfRange(idx))
    }
}

/// Return the codon position within an AA's synonym list.
///
/// Returns `Err` if `codon` is not a synonym of AA `aa_idx`.
pub fn get_codon_index(aa_idx: usize, codon: [u8; 3]) -> Result<usize, CodonError> {
    codon_index()
        .get(&(aa_idx, codon))
        .copied()
        .ok_or(CodonError::NotASynonym { aa_idx, codon })
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AminoAcidError {
    #[error("unknown amino acid code: {0:?}")]
    UnknownCode(char),
    #[error("amino acid index out of range: {0}")]
    IndexOutOfRange(usize),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CodonError {
    #[error("codon {codon:?} is not a synonym of amino acid index {aa_idx}")]
    NotASynonym { aa_idx: usize, codon: [u8; 3] },
}
