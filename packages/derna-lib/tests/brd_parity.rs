//! Multi-path BRD parity: the codon-DFA beam against the exact interval DP.
//!
//! The beam designs over every synonymous coding sequence at once; the
//! `derna-codon-dp` `solve_brd` oracle enumerates the exact optimum by
//! interval DP. Both are exact for the same fixed-λ objective, so on every
//! instance their optima must agree in value — the comparison is two-sided
//! and tolerates floating-point accumulation only.
//!
//! It was not always so. The oracle used to quantize its objective at every
//! transition, so the two minimized subtly different quantities and this
//! assertion was one-sided with a half-dekacal tolerance. That shape could
//! not see an oracle returning a *worse* design than the beam, which is
//! what it was in fact doing.
//!
//! Two harder invariants admit no tolerance at all: the design must
//! translate to the target protein, and its reported MFE must equal an
//! independent re-evaluation of the recovered structure (no phantom energy
//! from an interior the traceback spelled inconsistently with the scan).

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_codon_dp::{solve_brd, BrdOptions};
use derna_lib::{design, CodonUsage, Params};
use rna_mfe::evaluate;

/// BRD tolerance in dekacalories, covering floating-point accumulation
/// only. Both solvers are exact for the same objective, so their optima
/// must agree in value; they reach it by different summation orders, which
/// is all this absorbs.
///
/// It was once `0.5`, one-sided, to cover an oracle that quantized its
/// objective per transition. That tolerance was both larger than the drift
/// it was meant to describe and blind to its sign: the oracle could return
/// a design worse than the beam's without the assertion noticing, because
/// the assertion only looked one way.
const BRD_TOL: f64 = 1e-6;

/// Fixed-λ BRD score in dekacalories. Nothing is rounded: this is the
/// quantity both sides minimize, and rounding it here would reintroduce
/// the blur the comparison exists to detect.
fn brd(lambda: f64, mfe_kcal: f64, cai_log_sum: f64) -> f64 {
    lambda * mfe_kcal * 100.0 - (1.0 - lambda) * cai_log_sum * 100.0
}

/// Whether `seq` is a valid coding sequence for `protein` (every triple a
/// synonymous codon of the corresponding residue).
fn translates(seq: &str, protein: &str) -> bool {
    let b = seq.as_bytes();
    if b.len() != protein.len() * 3 {
        return false;
    }
    let code = |c: u8| match c {
        b'A' => 0u8,
        b'C' => 1,
        b'G' => 2,
        _ => 3,
    };
    protein.chars().enumerate().all(|(i, ch)| {
        let aa = aa_to_index(ch).unwrap();
        let trip = [code(b[3 * i]), code(b[3 * i + 1]), code(b[3 * i + 2])];
        AMINO_ACIDS[aa].codons.iter().any(|c| c.nuc == trip)
    })
}

/// Assert the beam matches the oracle on `protein` across the λ sweep:
/// translatable, phantom-free, and BRD-optimal within [`BRD_TOL`].
fn assert_parity(protein: &str, lambdas: &[f64]) {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let opts = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    let m = protein.len() as f64;
    for &lambda in lambdas {
        let d = design(&params, protein, &usage, lambda).unwrap();
        let o = solve_brd(protein, &usage, lambda, &opts).unwrap();

        assert!(
            translates(&d.sequence, protein),
            "design does not encode {protein} at λ={lambda}: seq={}",
            d.sequence,
        );
        let reeval = evaluate(&params, d.sequence.as_bytes(), &d.structure)
            .expect("recovered structure scores")
            .dekacal() as f64
            / 100.0;
        assert!(
            (reeval - d.mfe_kcal_mol).abs() <= 0.005,
            "phantom energy [{protein} λ={lambda}]: reported={:.2} reeval={reeval:.2} \
             structure={}",
            d.mfe_kcal_mol,
            d.structure,
        );
        let delta =
            brd(lambda, d.mfe_kcal_mol, d.cai_log) - brd(lambda, o.mfe_kcal_mol, o.cai.ln() * m);
        assert!(
            delta.abs() <= BRD_TOL,
            "beam and oracle disagree [{protein} λ={lambda}]: Δbrd={delta:.6} \
             beam(mfe={:.2},cai={:.4}) oracle(mfe={:.2},cai={:.4})",
            d.mfe_kcal_mol,
            d.cai,
            o.mfe_kcal_mol,
            o.cai,
        );
    }
}

/// The full λ sweep used by the parity checks.
const LAMBDAS: [f64; 7] = [0.0001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];

/// Short proteins that still exercise every loop kind — hairpins,
/// bulges, internal loops, and multi-loops — at sizes the default
/// (debug) test run can fold quickly. Longer sequences live behind the
/// ignored corpus-length check below.
#[test]
fn brd_parity_curated() {
    for protein in [
        "MWMW",
        "GGGGCCCC",
        "FFLLSSPPHHRR",
        "MKTAYIAKQR",
        "DEYWCNQHKR",
    ] {
        assert_parity(protein, &LAMBDAS);
    }
}

/// The corpus-length acceptance check: a 77-residue human-derived protein
/// (231 nt) folded against the exact oracle. Ignored by default because a
/// full multi-path design at this length runs in seconds, not
/// milliseconds; run explicitly with `--ignored --release`.
#[test]
#[ignore = "corpus-length acceptance check; run with --ignored --release"]
fn brd_parity_glpe78() {
    const GLPE78: &str =
        "MYGKIIFVLLLSGIVSISASSTTGVAMHTSTSSSVTKSYISSQTNGITLINWWAMARVIFEVMLVVVGMIILISYCIR";
    assert_parity(GLPE78, &[0.0001, 0.5, 0.99]);
}
