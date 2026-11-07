//! Kazusa codon-usage table loader.
//!
//! Loads a `amino_acid,codon,fraction` CSV into a `[21 × 6]` f64 table: the
//! twenty amino acids' rows from the file, and the stop family's row (see
//! [`crate::codon_table::STOP`]) filled with one weight for every stop codon,
//! since a stop codon carries no adaptation. The file has no stop rows.
//! The default human codon-usage table is baked into the binary via
//! `include_str!`; no runtime file I/O is needed for the common path.

use std::path::Path;

use thiserror::Error;

use crate::codon_table::{aa_to_index, get_codon_index, AMINO_ACIDS, FAMILIES, N_CODON, STOP};

const CODON_USAGE_CSV: &str = include_str!("../data/codon_usage_human.csv");

/// The weight every stop codon carries in the stop family's row: one value
/// for all three, so each one's log weight against the family's best is
/// `ln 1 = 0` and a stop slot adds nothing to any design's adaptation.
pub const STOP_WEIGHT: f64 = 1.0;

/// Codon-usage frequency table. `table[aa_idx][codon_idx]` is the Kazusa
/// fraction for that codon. Cells beyond `N_CODON[aa]` are `0.0`. The stop
/// family's row holds [`STOP_WEIGHT`] for each stop codon.
#[derive(Debug, Clone)]
pub struct CodonUsage {
    table: [[f64; 6]; FAMILIES],
}

impl CodonUsage {
    /// Load the vendored human codon-usage table (baked in at compile time).
    pub fn load_default() -> Result<Self, CodonUsageError> {
        Self::load_from_str(CODON_USAGE_CSV)
    }

    /// Load from an arbitrary file path.
    pub fn load_from_path(path: &Path) -> Result<Self, CodonUsageError> {
        let text = std::fs::read_to_string(path).map_err(|e| CodonUsageError::Io(e.to_string()))?;
        Self::load_from_str(&text)
    }

    /// Load from a CSV string.
    pub fn load_from_str(csv: &str) -> Result<Self, CodonUsageError> {
        let mut table = [[0.0f64; 6]; FAMILIES];
        let mut seen: [u8; FAMILIES] = [0; FAMILIES];

        let mut lines = csv.lines();
        let header = lines.next().ok_or(CodonUsageError::MissingHeader)?;
        if header.trim() != "amino_acid,codon,fraction" {
            return Err(CodonUsageError::BadHeader(header.to_string()));
        }

        for (line_no, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, ',').collect();
            if parts.len() != 3 {
                return Err(CodonUsageError::BadRow {
                    line_no: line_no + 2,
                    row: line.to_string(),
                });
            }
            let aa_char = parse_aa_char(parts[0])
                .ok_or_else(|| CodonUsageError::UnknownAminoAcid(parts[0].to_string()))?;
            let aa_idx = aa_to_index(aa_char)
                .map_err(|_| CodonUsageError::UnknownAminoAcid(parts[0].to_string()))?;

            let codon_nuc = parse_codon_nuc(parts[1])
                .ok_or_else(|| CodonUsageError::BadCodon(parts[1].to_string()))?;
            let ci = get_codon_index(aa_idx, codon_nuc).map_err(|_| {
                CodonUsageError::CodonNotSynonym {
                    aa: aa_char,
                    codon: parts[1].to_string(),
                }
            })?;

            let fraction: f64 = parts[2]
                .trim()
                .parse()
                .map_err(|_| CodonUsageError::BadFraction(parts[2].to_string()))?;

            table[aa_idx][ci] = fraction;
            seen[aa_idx] |= 1 << ci;
        }

        check_complete(&seen)?;
        table[STOP][..N_CODON[STOP]].fill(STOP_WEIGHT);
        Ok(CodonUsage { table })
    }

    /// Return the frequency for `(aa_idx, codon_idx)`.
    pub fn fraction(&self, aa_idx: usize, ci: usize) -> f64 {
        self.table[aa_idx][ci]
    }

    /// Raw table access for the CAI module.
    pub(crate) fn table(&self) -> &[[f64; 6]; FAMILIES] {
        &self.table
    }
}

fn parse_aa_char(s: &str) -> Option<char> {
    let s = s.trim();
    if s.len() == 1 {
        s.chars().next()
    } else {
        None
    }
}

fn parse_codon_nuc(s: &str) -> Option<[u8; 3]> {
    let s = s.trim();
    if s.len() != 3 {
        return None;
    }
    let bytes: Vec<u8> = s.chars().map(nuc_index).collect::<Option<Vec<_>>>()?;
    Some([bytes[0], bytes[1], bytes[2]])
}

fn nuc_index(c: char) -> Option<u8> {
    match c {
        'A' => Some(0),
        'C' => Some(1),
        'G' => Some(2),
        'U' => Some(3),
        _ => None,
    }
}

/// Every amino acid's codons were given. The stop family is not read from the
/// file, so it is not asked for.
fn check_complete(seen: &[u8; FAMILIES]) -> Result<(), CodonUsageError> {
    let mut missing = Vec::new();
    for aa in &AMINO_ACIDS[..STOP] {
        let nc = N_CODON[aa.index];
        for ci in 0..nc {
            if seen[aa.index] & (1 << ci) == 0 {
                missing.push(format!("{}/{}", aa.code, aa.codons[ci].name));
            }
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(CodonUsageError::MissingCodons(missing))
    }
}

/// Convenience wrapper: load a codon-usage table from `path`.
pub fn load_codon_usage(path: &Path) -> Result<CodonUsage, CodonUsageError> {
    CodonUsage::load_from_path(path)
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CodonUsageError {
    #[error("CSV is missing a header line")]
    MissingHeader,
    #[error("unexpected CSV header: {0:?} (expected \"amino_acid,codon,fraction\")")]
    BadHeader(String),
    #[error("malformed row at line {line_no}: {row:?}")]
    BadRow { line_no: usize, row: String },
    #[error("unknown amino acid: {0:?}")]
    UnknownAminoAcid(String),
    #[error("unrecognised codon: {0:?}")]
    BadCodon(String),
    #[error("codon {codon:?} is not a synonym of amino acid {aa:?}")]
    CodonNotSynonym { aa: char, codon: String },
    #[error("unreadable fraction: {0:?}")]
    BadFraction(String),
    #[error("CSV missing codons: {}", .0.join(", "))]
    MissingCodons(Vec<String>),
    #[error("I/O error: {0}")]
    Io(String),
}
