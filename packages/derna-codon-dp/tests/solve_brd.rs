//! Full-solve parity vs v1 for `solve_brd`.
//!
//! Loads `v1_brd_toy_lambda_{0,25,50,75,100}.json` (regenerable via
//! `tests/data/dump_v1_brd.py`) and asserts the Rust port's
//! `BrdSolution` matches v1's field-by-field on every λ corner of the
//! toy 6-aa `GLAGPR` protein. `rna_seq` and `structure` compare as
//! string equality; `mfe_kcal_mol`, `cai`, `lambda_`, and
//! `brd_value_kcal_mol` compare as exact `f64` equality at the
//! dekacal-int boundary.

use std::fs;
use std::path::PathBuf;

use derna_codon_dp::codon_usage::CodonUsage;
use derna_codon_dp::{solve_brd, BrdOptions, BrdSolution};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct V1Brd {
    protein: String,
    dangling_5_prime: bool,
    lambda_: f64,
    rna_seq: String,
    structure: String,
    mfe_kcal_mol_bits: u64,
    cai_bits: u64,
    brd_value_kcal_mol_bits: u64,
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn load_v1(lambda_suffix: u32) -> V1Brd {
    let path = data_dir().join(format!("v1_brd_toy_lambda_{lambda_suffix}.json"));
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn rust_solve(protein: &str, lambda: f64) -> BrdSolution {
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    solve_brd(protein, &codon_usage, lambda, &options).expect("solve_brd")
}

fn run_parity(lambda: f64, suffix: u32) {
    let v1 = load_v1(suffix);
    assert_eq!(v1.protein, "GLAGPR");
    assert!((v1.lambda_ - lambda).abs() < 1e-12);
    assert!(!v1.dangling_5_prime);

    let rust = rust_solve(&v1.protein, lambda);
    assert_eq!(rust.rna_seq, v1.rna_seq, "rna_seq mismatch at λ={lambda}");
    assert_eq!(
        rust.structure, v1.structure,
        "structure mismatch at λ={lambda}"
    );
    assert_eq!(
        rust.mfe_kcal_mol.to_bits(),
        v1.mfe_kcal_mol_bits,
        "mfe_kcal_mol mismatch at λ={lambda}: rust={} v1.bits=0x{:016x}",
        rust.mfe_kcal_mol,
        v1.mfe_kcal_mol_bits,
    );
    assert_eq!(
        rust.cai.to_bits(),
        v1.cai_bits,
        "cai mismatch at λ={lambda}: rust={} v1.bits=0x{:016x}",
        rust.cai,
        v1.cai_bits,
    );
    assert_eq!(rust.lambda_.to_bits(), v1.lambda_.to_bits());
    assert_eq!(
        rust.brd_value_kcal_mol.to_bits(),
        v1.brd_value_kcal_mol_bits,
        "brd_value_kcal_mol mismatch at λ={lambda}: rust={} v1.bits=0x{:016x}",
        rust.brd_value_kcal_mol,
        v1.brd_value_kcal_mol_bits,
    );
}

#[test]
fn full_solve_parity_lambda_0() {
    run_parity(0.0, 0);
}

#[test]
fn full_solve_parity_lambda_25() {
    run_parity(0.25, 25);
}

#[test]
fn full_solve_parity_lambda_50() {
    run_parity(0.5, 50);
}

#[test]
fn full_solve_parity_lambda_75() {
    run_parity(0.75, 75);
}

#[test]
fn full_solve_parity_lambda_100() {
    run_parity(1.0, 100);
}
