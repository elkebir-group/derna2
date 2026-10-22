use derna_codon_dp::cai::{compute_cai_tables, score_cai, sense_codons, standard_cai, CaiError};
use derna_codon_dp::codon_table::{FAMILIES, N_CODON, STOP};
use derna_codon_dp::codon_usage::CodonUsage;

fn default_tables() -> (CodonUsage, derna_codon_dp::cai::CaiTables) {
    let usage = CodonUsage::load_default().expect("load default");
    let tables = compute_cai_tables(&usage);
    (usage, tables)
}

// ---------------------------------------------------------------------------
// max_cai_pos — must match C++ default.cpp:554 exactly
// ---------------------------------------------------------------------------

#[test]
fn max_cai_pos_matches_cpp() {
    let (_, tables) = default_tables();
    assert_eq!(
        tables.max_cai_pos[..STOP],
        [1, 3, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 5, 1, 0, 1, 3]
    );
}

#[test]
fn every_stop_codon_weighs_nothing() {
    let (_, tables) = default_tables();
    assert_eq!(&tables.log_cai[STOP][..N_CODON[STOP]], &[0.0, 0.0, 0.0]);
    // So "the most adapted stop codon" names nothing: the three tie, and which
    // index the argmax lands on carries no meaning. Asserted as the tie rather
    // than as that index, which nothing reads.
    let row = &tables.log_cai[STOP][..N_CODON[STOP]];
    assert!(row.iter().all(|&w| w == row[0]));
}

// ---------------------------------------------------------------------------
// log_cai invariants
// ---------------------------------------------------------------------------

#[test]
fn log_cai_invalid_slots_are_neg_infinity() {
    let (_, tables) = default_tables();
    for (aa, &nc) in N_CODON.iter().enumerate() {
        for ci in nc..6 {
            assert!(
                tables.log_cai[aa][ci].is_infinite() && tables.log_cai[aa][ci] < 0.0,
                "expected NEG_INFINITY at aa={aa} ci={ci}, got {}",
                tables.log_cai[aa][ci]
            );
        }
    }
}

#[test]
fn log_cai_max_codon_is_zero() {
    let (_, tables) = default_tables();
    for (aa, _) in N_CODON.iter().enumerate() {
        let max_ci = tables.max_cai_pos[aa];
        assert_eq!(
            tables.log_cai[aa][max_ci], 0.0,
            "log_cai[{aa}][{max_ci}] should be 0.0 (max codon)"
        );
    }
}

#[test]
fn log_cai_non_max_valid_codons_are_negative() {
    let (_, tables) = default_tables();
    // The stop family is left out: its codons tie, so none is below the best.
    for (aa, &nc) in N_CODON.iter().enumerate().take(STOP) {
        let max_ci = tables.max_cai_pos[aa];
        for ci in 0..nc {
            if ci != max_ci {
                assert!(
                    tables.log_cai[aa][ci] < 0.0,
                    "log_cai[{aa}][{ci}] should be < 0.0, got {}",
                    tables.log_cai[aa][ci]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// standard_cai == 1.0 on all-max-codon RNA
// ---------------------------------------------------------------------------

fn build_max_rna(protein: &[usize], max_cai_pos: &[usize; FAMILIES]) -> Vec<u8> {
    use derna_codon_dp::codon_table::AMINO_ACIDS;
    let mut rna = Vec::with_capacity(protein.len() * 3);
    for &aa in protein {
        let ci = max_cai_pos[aa];
        rna.extend_from_slice(&AMINO_ACIDS[aa].codons[ci].nuc);
    }
    rna
}

fn protein_indices(s: &str) -> Vec<usize> {
    use derna_codon_dp::codon_table::aa_to_index;
    s.chars().map(|c| aa_to_index(c).unwrap()).collect()
}

#[test]
fn standard_cai_one_on_max_rna() {
    let (_, tables) = default_tables();
    for protein_str in &["MA", "GLPE", "MAVKVHRPSCN"] {
        let protein = protein_indices(protein_str);
        let rna = build_max_rna(&protein, &tables.max_cai_pos);
        let s = score_cai(&rna, &protein, &tables.log_cai).unwrap();
        let std = standard_cai(&rna, &protein, &tables.log_cai).unwrap();
        assert_eq!(s, 0.0, "{protein_str}: score_cai should be 0.0");
        assert_eq!(std, 1.0, "{protein_str}: standard_cai should be 1.0");
    }
}

// ---------------------------------------------------------------------------
// Empty protein → standard_cai == 0.0
// ---------------------------------------------------------------------------

#[test]
fn standard_cai_empty_protein() {
    let (_, tables) = default_tables();
    let result = standard_cai(&[], &[], &tables.log_cai).unwrap();
    assert_eq!(result, 0.0);
}

// ---------------------------------------------------------------------------
// A stop slot moves neither the log sum nor the conventional CAI
// ---------------------------------------------------------------------------

#[test]
fn a_stop_slot_changes_no_cai() {
    let (_, tables) = default_tables();
    use derna_codon_dp::codon_table::AMINO_ACIDS;
    let protein = protein_indices("MAVKVHRPSCN");
    // A spelling below the best at every residue that has a choice, so the
    // geometric mean is not 1 and a changed denominator would show.
    let rna: Vec<u8> = protein
        .iter()
        .flat_map(|&aa| AMINO_ACIDS[aa].codons.last().unwrap().nuc)
        .collect();
    let score = score_cai(&rna, &protein, &tables.log_cai).unwrap();
    let cai = standard_cai(&rna, &protein, &tables.log_cai).unwrap();
    assert!(cai < 1.0);
    let mut with_stop = protein.clone();
    with_stop.push(STOP);
    assert_eq!(sense_codons(&with_stop), protein.len());
    for stop in AMINO_ACIDS[STOP].codons {
        let mut spelled = rna.clone();
        spelled.extend_from_slice(&stop.nuc);
        assert_eq!(
            score_cai(&spelled, &with_stop, &tables.log_cai).unwrap(),
            score,
            "{}",
            stop.name
        );
        assert_eq!(
            standard_cai(&spelled, &with_stop, &tables.log_cai).unwrap(),
            cai,
            "{}",
            stop.name
        );
    }
}

// ---------------------------------------------------------------------------
// Wrong-length RNA returns error
// ---------------------------------------------------------------------------

#[test]
fn score_cai_wrong_length_returns_error() {
    let (_, tables) = default_tables();
    let protein = protein_indices("MA");
    let short_rna: Vec<u8> = vec![0, 3, 2]; // 3 != 3*2 = 6
    assert_eq!(
        score_cai(&short_rna, &protein, &tables.log_cai),
        Err(CaiError::LengthMismatch {
            rna_len: 3,
            protein_len: 2
        })
    );
}

// ---------------------------------------------------------------------------
// Parity vs Python v1 — exact f64 on standard_cai and score_cai
// ---------------------------------------------------------------------------

#[test]
fn parity_vs_python_v1() {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct TestCase {
        protein: Vec<usize>,
        rna: Vec<u8>,
        score_cai: f64,
        standard_cai: f64,
    }

    #[derive(Deserialize)]
    struct Dump {
        test_cases: Vec<TestCase>,
    }

    let json_str = include_str!("data/v1_cai_dump.json");
    let dump: Dump = serde_json::from_str(json_str).expect("parse v1 cai dump");
    let (_, tables) = default_tables();

    for (i, tc) in dump.test_cases.iter().enumerate() {
        let rust_score = score_cai(&tc.rna, &tc.protein, &tables.log_cai).unwrap();
        let rust_std = standard_cai(&tc.rna, &tc.protein, &tables.log_cai).unwrap();
        assert_eq!(
            rust_score, tc.score_cai,
            "test_case[{i}]: score_cai mismatch: rust={rust_score} py={}",
            tc.score_cai
        );
        assert_eq!(
            rust_std, tc.standard_cai,
            "test_case[{i}]: standard_cai mismatch: rust={rust_std} py={}",
            tc.standard_cai
        );
    }
}
