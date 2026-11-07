//! Codon Adaptation Index (CAI) scoring.
//!
//! Log-scaled and standard CAI from Kazusa codon-usage frequencies.
//! The log-scaled objective is consumed directly by the codon DP; the
//! standard CAI (geometric mean of relative adaptiveness) is the
//! conventional metric reported alongside DP results.

use crate::codon_table::{get_codon_index, FAMILIES, N_CODON, STOP};
use crate::codon_usage::CodonUsage;

/// Floor for `log(0)` in CAI: a Kazusa-zero codon that is still a valid
/// synonym maps to `log((usage + EPSILON) / max_freq)` rather than `-inf`,
/// keeping the score finite without distorting non-zero weights.
pub const EPSILON: f64 = 1e-5;

/// Log-scaled CAI lookup tables derived from a [`CodonUsage`].
#[derive(Debug, Clone)]
pub struct CaiTables {
    /// `[aa_idx][codon_idx]` log-CAI. Valid codons: `log(usage / max_usage)`.
    /// Valid codons with zero usage: `log((usage + EPSILON) / max_usage)`.
    /// Invalid slots (`ci >= N_CODON[aa]`) are `f64::NEG_INFINITY`. The stop
    /// family's row ([`STOP`]) is zero for every stop codon.
    pub log_cai: [[f64; 6]; FAMILIES],
    /// Per-AA index of the most frequent codon.
    pub max_cai_pos: [usize; FAMILIES],
}

/// Compute log-scaled CAI lookups from raw codon-usage frequencies.
pub fn compute_cai_tables(codon_usage: &CodonUsage) -> CaiTables {
    let mut log_cai = [[f64::NEG_INFINITY; 6]; FAMILIES];
    let mut max_cai_pos = [0usize; FAMILIES];
    let table = codon_usage.table();

    for aa in 0..FAMILIES {
        let nc = N_CODON[aa];
        let row = &table[aa][..nc];

        let max_idx = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        max_cai_pos[aa] = max_idx;
        let max_freq = row[max_idx];
        if max_freq > 0.0 {
            for ci in 0..nc {
                log_cai[aa][ci] = (row[ci].max(EPSILON) / max_freq).ln();
            }
        }
    }

    CaiTables {
        log_cai,
        max_cai_pos,
    }
}

/// Sum of `log_cai` across the protein's codons. The raw DP objective.
///
/// `rna` is a flat slice of nucleotide indices (length `3 * protein.len()`);
/// `protein` is a slice of AA indices.
pub fn score_cai(
    rna: &[u8],
    protein: &[usize],
    log_cai: &[[f64; 6]; FAMILIES],
) -> Result<f64, CaiError> {
    let n = protein.len();
    if rna.len() != 3 * n {
        return Err(CaiError::LengthMismatch {
            rna_len: rna.len(),
            protein_len: n,
        });
    }
    let mut total = 0.0f64;
    for (i, &aa) in protein.iter().enumerate() {
        let codon = [rna[3 * i], rna[3 * i + 1], rna[3 * i + 2]];
        let ci = get_codon_index(aa, codon).map_err(|_| CaiError::InvalidCodon { aa, codon })?;
        total += log_cai[aa][ci];
    }
    Ok(total)
}

/// Conventional CAI: `exp(score_cai / m)`, in [0, 1], where `m` counts the
/// protein's sense codons ([`sense_codons`]). Returns `0.0` when there are
/// none.
pub fn standard_cai(
    rna: &[u8],
    protein: &[usize],
    log_cai: &[[f64; 6]; FAMILIES],
) -> Result<f64, CaiError> {
    let m = sense_codons(protein);
    if m == 0 {
        return Ok(0.0);
    }
    Ok((score_cai(rna, protein, log_cai)? / m as f64).exp())
}

/// How many of `protein`'s positions are amino acids rather than a stop slot
/// ([`STOP`]): the count conventional CAI's geometric mean is taken over. A
/// stop slot adds zero to the log sum, so leaving it out of the count is what
/// keeps the reported CAI the same whichever stop a design ends in, and the
/// same as a design whose stop is fixed sequence.
pub fn sense_codons(protein: &[usize]) -> usize {
    protein.iter().filter(|&&aa| aa != STOP).count()
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CaiError {
    #[error("rna length {rna_len} != 3 * protein length {protein_len}")]
    LengthMismatch { rna_len: usize, protein_len: usize },
    #[error("codon {codon:?} is not valid for amino acid index {aa}")]
    InvalidCodon { aa: usize, codon: [u8; 3] },
}
