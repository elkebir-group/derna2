//! Fixed-UTR joint-fold parity.
//!
//! The full mRNA `utr5 + CDS + utr3` folds as one structure, with the UTR
//! nucleotides fixed and only the CDS designed. Three invariants pin the
//! substrate:
//!
//! 1. **Zero-length parity (the floor).** With empty UTRs and no other change,
//!    [`design_with_utr`] is byte-identical to [`design`] — same sequence,
//!    structure, MFE, and CAI. The scope must never break this.
//! 2. **Fold parity.** The whole-mRNA MFE the joint scan reports for its chosen
//!    sequence equals an independent full fold of that spelled sequence, and the
//!    reported structure re-scores to the reported energy (phantom-free). The
//!    seam between UTR and CDS introduces no energetics error.
//! 3. **Brute-force BRD optimum.** At a small protein and short UTRs, enumerate
//!    every CDS encoding, concatenate the fixed UTR, fold each, and take the
//!    BRD-argmin; the joint design matches it within the objective
//!    tolerance (byte-exact on MFE + sequence, tolerance only at flat fronts).

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design, design_with_utr, linearfold_fold, CodonUsage, Design, Params, RegionConstraints,
    ScanConfig,
};
use rna_mfe::evaluate;

/// b → ∞: a beam wide enough that `rna-linearfold` runs the exact lattice.
const EXACT: usize = 1 << 20;

/// BRD tolerance in dekacalories — absorbs the rounded-integer vs continuous
/// objective gap, exactly as `tests/brd_parity.rs`.
const BRD_TOL: f64 = 0.5;

/// The λ values swept by every parity check (low / mid / high).
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

fn nuc_byte(n: u8) -> u8 {
    [b'A', b'C', b'G', b'U'][n as usize]
}

fn protein_indices(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}

/// Fixed-λ BRD on rounded dekacalories, matching how the oracle and the design
/// both round at the scalarization boundary (`tests/brd_parity.rs`).
fn brd(lambda: f64, mfe_kcal: f64, cai_log: f64) -> f64 {
    let mfe_dk = (mfe_kcal * 100.0).round();
    let cai_dk = (cai_log * 100.0).round();
    lambda * mfe_dk - (1.0 - lambda) * cai_dk
}

/// Whether `cds` (ASCII `A/C/G/U`) is a valid coding sequence for `protein`.
fn cds_translates(cds: &[u8], protein: &str) -> bool {
    if cds.len() != protein.len() * 3 {
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
        let trip = [code(cds[3 * i]), code(cds[3 * i + 1]), code(cds[3 * i + 2])];
        AMINO_ACIDS[aa].codons.iter().any(|c| c.nuc == trip)
    })
}

/// Assert the design's whole-mRNA sequence really is `utr5 + CDS + utr3` for
/// `protein`, returning the spelled full-mRNA bytes.
fn assert_layout(d: &Design, protein: &str, utr5: &str, utr3: &str, label: &str) -> Vec<u8> {
    let full = d.sequence.as_bytes();
    let l5 = utr5.len();
    let l3 = utr3.len();
    assert_eq!(
        full.len(),
        l5 + protein.len() * 3 + l3,
        "full-mRNA length wrong [{label}]",
    );
    assert_eq!(
        &full[..l5],
        utr5.as_bytes(),
        "5′ UTR not preserved [{label}]"
    );
    assert_eq!(
        &full[full.len() - l3..],
        utr3.as_bytes(),
        "3′ UTR not preserved [{label}]",
    );
    assert!(
        cds_translates(&full[l5..full.len() - l3], protein),
        "CDS does not encode {protein} [{label}]",
    );
    full.to_vec()
}

/// Fold parity + phantom-free for one `(protein, utr5, utr3, λ)`: the joint
/// scan's whole-mRNA MFE equals an independent fold of its chosen sequence, and
/// the reported structure re-scores to that energy.
fn assert_fold_parity(protein: &str, utr5: &str, utr3: &str) {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for &lambda in &LAMBDAS {
        let label = format!("{protein} 5′={utr5:?} 3′={utr3:?} λ={lambda}");
        let d = design_with_utr(
            &params,
            protein,
            &usage,
            lambda,
            utr5,
            utr3,
            RegionConstraints::default(),
            ScanConfig::default(),
        )
        .unwrap();
        let full = assert_layout(&d, protein, utr5, utr3, &label);

        // (1) Fold parity: the joint MFE equals an independent full fold of the
        // spelled whole mRNA — the seam introduces no energetics error.
        let (structure, energy) = linearfold_fold(&params, &full, EXACT).expect("folds");
        let reported_dk = (d.mfe_kcal_mol * 100.0).round() as i32;
        assert_eq!(
            reported_dk,
            energy.dekacal(),
            "fold-parity MFE mismatch [{label}]: joint={reported_dk} independent={}\n  joint {}\n  indep {}",
            energy.dekacal(),
            d.structure,
            structure,
        );
        // (2) Phantom-free: the reported structure re-scores to the reported MFE.
        let reeval = evaluate(&params, &full, &d.structure).expect("structure scores");
        assert_eq!(
            reeval.dekacal(),
            reported_dk,
            "phantom energy [{label}]: structure={} reeval={}",
            d.structure,
            reeval.dekacal(),
        );
    }
}

/// Every CDS encoding of `protein` as a list of per-residue codon choices,
/// enumerated by mixed-radix counting over the synonym counts.
fn encodings(aa: &[usize]) -> Vec<Vec<usize>> {
    let counts: Vec<usize> = aa.iter().map(|&a| AMINO_ACIDS[a].codons.len()).collect();
    let total: usize = counts.iter().product();
    (0..total)
        .map(|mut n| {
            counts
                .iter()
                .map(|&c| {
                    let ci = n % c;
                    n /= c;
                    ci
                })
                .collect()
        })
        .collect()
}

/// The whole-mRNA bytes for one CDS encoding: `utr5 + CDS + utr3`.
fn full_bytes(aa: &[usize], encoding: &[usize], utr5: &str, utr3: &str) -> Vec<u8> {
    let mut out = utr5.as_bytes().to_vec();
    for (&a, &ci) in aa.iter().zip(encoding) {
        out.extend(AMINO_ACIDS[a].codons[ci].nuc.iter().map(|&n| nuc_byte(n)));
    }
    out.extend_from_slice(utr3.as_bytes());
    out
}

/// CDS-only additive log-CAI of one encoding: `Σ log(g(ci) / g_max)` per codon,
/// with the same `EPSILON` floor the DFA's edge weights use (so it reproduces
/// the design's `cai_log` for the same encoding).
fn cds_cai_log(usage: &CodonUsage, aa: &[usize], encoding: &[usize]) -> f64 {
    aa.iter()
        .zip(encoding)
        .map(|(&a, &ci)| {
            let g = |c: usize| usage.fraction(a, c).max(EPSILON);
            let g_max = (0..AMINO_ACIDS[a].codons.len())
                .map(g)
                .fold(EPSILON, f64::max);
            (g(ci) / g_max).ln()
        })
        .sum()
}

/// Brute-force BRD oracle for one small `(protein, utr5, utr3)`: enumerate every
/// CDS encoding, fold `utr5 + CDS + utr3`, and compare the design to the
/// BRD-argmin across the λ sweep.
fn assert_brute_force(protein: &str, utr5: &str, utr3: &str) {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let aa = protein_indices(protein);
    let all = encodings(&aa);
    // Pre-fold every candidate once; the MFE is λ-independent, the CAI too.
    let scored: Vec<(f64, f64)> = all
        .iter()
        .map(|enc| {
            let full = full_bytes(&aa, enc, utr5, utr3);
            let (_, energy) = linearfold_fold(&params, &full, EXACT).expect("folds");
            (
                energy.dekacal() as f64 / 100.0,
                cds_cai_log(&usage, &aa, enc),
            )
        })
        .collect();

    for &lambda in &LAMBDAS {
        let label = format!("{protein} 5′={utr5:?} 3′={utr3:?} λ={lambda}");
        let oracle = scored
            .iter()
            .map(|&(mfe, cai)| brd(lambda, mfe, cai))
            .fold(f64::INFINITY, f64::min);
        let d = design_with_utr(
            &params,
            protein,
            &usage,
            lambda,
            utr5,
            utr3,
            RegionConstraints::default(),
            ScanConfig::default(),
        )
        .unwrap();
        assert_layout(&d, protein, utr5, utr3, &label);
        let design_brd = brd(lambda, d.mfe_kcal_mol, d.cai_log);
        assert!(
            design_brd - oracle <= BRD_TOL,
            "design suboptimal vs brute force [{label}]: Δbrd={:.1} design={design_brd:.1} oracle={oracle:.1}",
            design_brd - oracle,
        );
    }
}

#[test]
fn zero_length_utr_is_byte_identical_to_cds_only() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in ["MWMW", "GGGGCCCC", "MKTAYIAKQR", "DEYWCNQHKR"] {
        for &lambda in &LAMBDAS {
            let base = design(&params, protein, &usage, lambda).unwrap();
            let utr = design_with_utr(
                &params,
                protein,
                &usage,
                lambda,
                "",
                "",
                RegionConstraints::default(),
                ScanConfig::default(),
            )
            .unwrap();
            assert_eq!(base, utr, "zero-length UTR diverged [{protein} λ={lambda}]");
        }
    }
}

#[test]
fn fixed_utr_fold_parity() {
    // 5′-only (Kozak-like context), 3′-only, both flanks, and UTRs that can
    // pair with the CDS to force genuine cross-region structure.
    assert_fold_parity("MKTAYIAKQR", "GCCACCAUGG", "");
    assert_fold_parity("DEYWCNQHKR", "", "AAUAAAGCUU");
    assert_fold_parity("MKTAYIAKQR", "GGGACC", "UUUAAACCC");
    assert_fold_parity("FFLLSSPPHHRR", "GGGCCCGGG", "CCCGGGCCC");
}

#[test]
fn fixed_utr_brute_force_optimum() {
    // Small proteins + short palindromic UTRs (cross-region pairing) so every
    // CDS encoding folds fast and the enumeration stays bounded.
    assert_brute_force("CDKN", "GGCC", "GGUU");
    assert_brute_force("LS", "GGGCCC", "GGGCCC");
    assert_brute_force("RF", "GCGCGC", "GCGCGC");
}
