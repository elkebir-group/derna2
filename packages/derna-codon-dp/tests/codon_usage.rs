use derna_codon_dp::codon_usage::{CodonUsage, CodonUsageError};

// ---------------------------------------------------------------------------
// CSV vendoring
// ---------------------------------------------------------------------------

const VENDORED_CSV: &str = include_str!("../data/codon_usage_human.csv");

#[test]
fn csv_byte_count() {
    assert_eq!(VENDORED_CSV.len(), 695);
}

#[test]
fn csv_sha256() {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(VENDORED_CSV.as_bytes());
    let hex = hex::encode(hash);
    assert_eq!(
        hex,
        "f1df2640cbb9fde4273f1a7ffe2c4a173ac4abfe2cff96607c1fbff9a8a50d56"
    );
}

// ---------------------------------------------------------------------------
// Loader round-trip parity vs Python
// ---------------------------------------------------------------------------

#[test]
fn load_default_succeeds() {
    CodonUsage::load_default().expect("default CSV loads");
}

#[test]
fn parity_vs_python_v1() {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Dump {
        table: Vec<Vec<f64>>,
    }

    let json_str = include_str!("data/v1_codon_usage_dump.json");
    let dump: Dump = serde_json::from_str(json_str).expect("parse v1 dump");
    let usage = CodonUsage::load_default().expect("load default");

    for aa in 0..20 {
        for ci in 0..6 {
            let rust_val = usage.fraction(aa, ci);
            let py_val = dump.table[aa][ci];
            assert_eq!(
                rust_val, py_val,
                "mismatch at aa={aa} ci={ci}: rust={rust_val} py={py_val}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Spot checks against the Kazusa H. sapiens table
// ---------------------------------------------------------------------------

#[test]
fn spot_checks_ala_leu_arg() {
    let usage = CodonUsage::load_default().expect("load");
    // A GCC = 0.40 (aa=0, ci=1)
    assert_eq!(usage.fraction(0, 1), 0.40);
    // L CUG = 0.41 (aa=11, ci=3)
    assert_eq!(usage.fraction(11, 3), 0.41);
    // R CGG = 0.21 (aa=1, ci=3)
    assert_eq!(usage.fraction(1, 3), 0.21);
}

// ---------------------------------------------------------------------------
// Cells beyond N_CODON[aa] are zero
// ---------------------------------------------------------------------------

#[test]
fn cells_beyond_n_codon_are_zero() {
    use derna_codon_dp::codon_table::N_CODON;
    let usage = CodonUsage::load_default().expect("load");
    for (aa, &nc) in N_CODON.iter().enumerate() {
        for ci in nc..6 {
            assert_eq!(
                usage.fraction(aa, ci),
                0.0,
                "aa={aa} ci={ci} beyond N_CODON[{aa}]={nc} should be 0"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Negative paths
// ---------------------------------------------------------------------------

#[test]
fn missing_header_returns_error() {
    let csv = "A,GCU,0.26\n";
    assert!(matches!(
        CodonUsage::load_from_str(csv),
        Err(CodonUsageError::BadHeader(_))
    ));
}

#[test]
fn empty_string_returns_error() {
    assert!(matches!(
        CodonUsage::load_from_str(""),
        Err(CodonUsageError::MissingHeader)
    ));
}

#[test]
fn unknown_aa_returns_error() {
    let csv = "amino_acid,codon,fraction\nZ,GCU,0.26\n";
    assert!(matches!(
        CodonUsage::load_from_str(csv),
        Err(CodonUsageError::UnknownAminoAcid(_))
    ));
}

#[test]
fn codon_mismatch_returns_error() {
    // GGC is Gly, not Ala — should fail with CodonNotSynonym.
    let csv = "amino_acid,codon,fraction\nA,GGC,0.26\n";
    assert!(matches!(
        CodonUsage::load_from_str(csv),
        Err(CodonUsageError::CodonNotSynonym { .. })
    ));
}

#[test]
fn missing_codon_rows_returns_error() {
    // Only one row for Ala (missing GCC, GCA, GCG and all other AAs).
    let csv = "amino_acid,codon,fraction\nA,GCU,0.26\n";
    assert!(matches!(
        CodonUsage::load_from_str(csv),
        Err(CodonUsageError::MissingCodons(_))
    ));
}

// ---------------------------------------------------------------------------
// The stop family: one weight for all three, and never read from the file
// ---------------------------------------------------------------------------

#[test]
fn every_stop_codon_carries_one_weight() {
    use derna_codon_dp::codon_table::STOP;
    use derna_codon_dp::codon_usage::STOP_WEIGHT;
    let usage = CodonUsage::load_default().expect("load");
    for ci in 0..3 {
        assert_eq!(usage.fraction(STOP, ci), STOP_WEIGHT, "ci={ci}");
    }
}

#[test]
fn a_stop_row_is_refused() {
    // A stop codon's frequency is not a codon usage: weighting the stops would
    // give the adaptation axis a term no amino acid has.
    let csv = format!("{VENDORED_CSV}*,UAA,0.30\n");
    assert!(matches!(
        CodonUsage::load_from_str(&csv),
        Err(CodonUsageError::UnknownAminoAcid(_))
    ));
}
