//! λ=1 fold-parity cross-check.
//!
//! A pure-MFE beam (λ=1) over a single-path DFA folds a fixed coding
//! sequence; it must reproduce `rna_linearfold::fold` (the CKY scan
//! template) at `b → ∞` on that sequence. This isolates the
//! grammar/energy-model port from the codon-DFA's multi-path CAI
//! bookkeeping, so a regression in the port surfaces here on its own.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::fold_single_path;
use rna_linearfold::fold;
use rna_mfe::{evaluate, Params};

/// b → ∞: a beam wide enough that `rna-linearfold` runs the exact lattice.
const EXACT: usize = 1 << 20;

const GLPE78: &str =
    "MYGKIIFVLLLSGIVSISASSTTGVAMHTSTSSSVTKSYISSQTNGITLINWWAMARVIFEVMLVVVGMIILISYCIR";

fn nuc_byte(n: u8) -> u8 {
    [b'A', b'C', b'G', b'U'][n as usize]
}

/// Build `(aa indices, codon triples, RNA bytes)` for `protein`, picking
/// each residue's codon by `pick(aa_index) mod synonym_count`.
fn coding(protein: &str, pick: impl Fn(usize) -> usize) -> (Vec<usize>, Vec<[u8; 3]>, Vec<u8>) {
    let mut aa = Vec::new();
    let mut triples = Vec::new();
    let mut rna = Vec::new();
    for ch in protein.chars() {
        let idx = aa_to_index(ch).expect("standard amino acid");
        let codons = AMINO_ACIDS[idx].codons;
        let codon = codons[pick(idx) % codons.len()].nuc;
        aa.push(idx);
        triples.push(codon);
        rna.extend(codon.iter().map(|&n| nuc_byte(n)));
    }
    (aa, triples, rna)
}

fn assert_parity(protein: &str, pick: impl Fn(usize) -> usize, label: &str) {
    let params = Params::load_default().expect("Turner 2004");
    let (aa, triples, rna) = coding(protein, pick);
    let design = fold_single_path(&params, &aa, &triples);
    let (structure, energy) = fold(&params, &rna, EXACT).expect("linearfold folds");

    let beam_dekacal = (design.mfe_kcal_mol * 100.0).round() as i32;
    // (1) MFE parity: the λ=1 beam reproduces LinearFold's exact-lattice
    // free energy. This is the substantive fold-parity invariant.
    assert_eq!(
        beam_dekacal,
        energy.dekacal(),
        "energy mismatch [{label}]: beam={} linearfold={} seq={}\n  beam {}\n  lf   {}",
        beam_dekacal,
        energy.dekacal(),
        String::from_utf8_lossy(&rna),
        design.structure,
        structure,
    );
    // (2) Traceback self-consistency: the recovered structure, scored
    // independently by the Zuker evaluator, equals the reported MFE — so
    // it is a genuine co-optimal MFE fold (tie-break choices among
    // equal-energy structures may differ from LinearFold's, which is
    // expected and immaterial to the energy gate).
    let reeval = evaluate(&params, &rna, &design.structure).expect("recovered structure scores");
    assert_eq!(
        reeval.dekacal(),
        beam_dekacal,
        "traceback inconsistency [{label}]: structure={} reeval={} reported={}",
        design.structure,
        reeval.dekacal(),
        beam_dekacal,
    );
    // The single path must spell exactly the chosen coding sequence.
    assert_eq!(
        design.sequence.as_bytes(),
        &rna[..],
        "sequence mismatch [{label}]"
    );
}

#[test]
fn fold_parity_first_codon() {
    for protein in ["MWMWMW", "GGGGCCCC", "ACDEFGHIKLMNPQRSTVWY", GLPE78] {
        assert_parity(protein, |_| 0, "first-codon");
    }
}

#[test]
fn fold_parity_varied_codons() {
    for protein in ["GGGGCCCC", "ACDEFGHIKLMNPQRSTVWY", GLPE78] {
        assert_parity(protein, |i| i, "by-index");
        assert_parity(protein, |i| i + 1, "by-index+1");
        assert_parity(protein, |i| 2 * i + 1, "by-index*2+1");
    }
}

#[test]
fn fold_parity_glpe78_codon_offsets() {
    for offset in [0, 2, 4] {
        assert_parity(GLPE78, move |i| i + offset, "glpe78-offset");
    }
}
