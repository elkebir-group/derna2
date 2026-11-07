//! Per-cell agreement with the reference tables for the four single-loop
//! recurrences and the `E` reduction.
//!
//! Loads `reference_tables_toy_lambda_10.json` and asserts the `TablesSnapshot`
//! matches `{E_h, E_s, E_b, E_i, E, E_m, M, N, O}` cell by cell.
//!
//! **Only λ = 1 is checked, and that is the whole of what the reference can
//! still say.** It was produced by an implementation that stored each cell's
//! objective quantized to whole dekacal. At λ = 1 the CAI term carries zero
//! weight and the energies are already integral, so no fractional quantity
//! exists and the two agree exactly — which makes this a real invariant
//! rather than a leftover. At any other weight the reference holds a sum of
//! separately-rounded pieces, which is not the rounding of this solver's
//! value and cannot be recovered from it by any tolerance; those weights are
//! anchored against reference designs in `reference_anchor.rs` instead.
//!
//! `E_m`, `M`, `N`, `O` should all be untouched (`INF`) here; the
//! reduction's `min` over `E_m` is a no-op because `E_m == INF`
//! everywhere.

use std::fs;
use std::path::PathBuf;

use derna_codon_dp::codon_usage::CodonUsage;
use derna_codon_dp::testing::{fill_single_loop_tables, FillInputs, TablesSnapshot, INF};
use derna_codon_dp::BrdOptions;
use rna_mfe::Params;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ReferenceTables {
    protein: String,
    lambda_: f64,
    dangling_5_prime: bool,
    n_aa: usize,
    max_codons: usize,
    shape: [usize; 6],
    #[serde(rename = "E")]
    e: Vec<i32>,
    #[serde(rename = "E_h")]
    e_h: Vec<i32>,
    #[serde(rename = "E_s")]
    e_s: Vec<i32>,
    #[serde(rename = "E_b")]
    e_b: Vec<i32>,
    #[serde(rename = "E_i")]
    e_i: Vec<i32>,
    #[serde(rename = "E_m")]
    e_m: Vec<i32>,
    #[serde(rename = "M")]
    m: Vec<i32>,
    #[serde(rename = "N")]
    n: Vec<i32>,
    #[serde(rename = "O")]
    o: Vec<i32>,
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn load_reference(lambda_suffix: u32) -> ReferenceTables {
    let path = data_dir().join(format!("reference_tables_toy_lambda_{lambda_suffix}.json"));
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn rust_snapshot(protein: &str, lambda_: f64) -> TablesSnapshot {
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let params = Params::load_default().expect("load default Turner 2004 params");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    fill_single_loop_tables(FillInputs {
        protein,
        codon_usage: &codon_usage,
        lambda_,
        options: &options,
        params: &params,
    })
    .expect("fill single-loop tables")
}

#[derive(Debug, Clone, Copy)]
struct TableShape {
    n_aa: usize,
    max_codons: usize,
}

fn assert_table_eq(name: &str, rust: &[f64], reference: &[i32], shape: TableShape) {
    assert_eq!(
        rust.len(),
        reference.len(),
        "{name}: rust len {} != reference len {}",
        rust.len(),
        reference.len()
    );
    let mut mismatches: Vec<(usize, f64, i32)> = Vec::new();
    for (i, (&r, &v)) in rust.iter().zip(reference.iter()).enumerate() {
        // The reference tables were quantized to whole dekacal; these are
        // not. Equality therefore holds through the rounding, and any cell
        // failing it is one where dropping the quantization changed which
        // candidate the recurrence selected -- a structural difference
        // rather than a resolution one.
        if r.round_ties_even() != f64::from(v) {
            mismatches.push((i, r, v));
        }
    }
    if !mismatches.is_empty() {
        let (i, r, v) = mismatches[0];
        let idx = decode_idx(i, shape);
        panic!(
            "{name}: {} cells mismatch (of {}); first at flat={} idx=(a={},b={},s={},t={},x={},y={}): rust={} reference={}",
            mismatches.len(),
            rust.len(),
            i,
            idx.0,
            idx.1,
            idx.2,
            idx.3,
            idx.4,
            idx.5,
            r,
            v,
        );
    }
}

fn decode_idx(flat: usize, shape: TableShape) -> (usize, usize, usize, usize, usize, usize) {
    let mc = shape.max_codons;
    let y = flat % mc;
    let x = (flat / mc) % mc;
    let t = (flat / (mc * mc)) % 3;
    let s = (flat / (mc * mc * 3)) % 3;
    let b = (flat / (mc * mc * 3 * 3)) % shape.n_aa;
    let a = flat / (mc * mc * 3 * 3 * shape.n_aa);
    (a, b, s, t, x, y)
}

fn assert_inf_fixture(name: &str, table: &[i32]) {
    for (i, &v) in table.iter().enumerate() {
        assert!(
            v == 1_000_000_000,
            "{name}[{i}] = {v}, expected the fixture INF sentinel"
        );
    }
}

fn assert_inf(name: &str, table: &[f64]) {
    for (i, &v) in table.iter().enumerate() {
        assert!(
            v == INF,
            "{name}[{i}] = {v}, expected INF (table should be untouched by the single-loop driver)"
        );
    }
}

fn run_parity_for_lambda(lambda_: f64, suffix: u32) {
    let reference = load_reference(suffix);
    assert_eq!(reference.protein, "GLAGPR");
    assert!((reference.lambda_ - lambda_).abs() < 1e-12);
    assert!(!reference.dangling_5_prime);

    let rust = rust_snapshot(&reference.protein, lambda_);
    assert_eq!(rust.n_aa, reference.n_aa);
    assert_eq!(rust.max_codons, reference.max_codons);
    assert_eq!(
        reference.shape,
        [
            reference.n_aa,
            reference.n_aa,
            3,
            3,
            reference.max_codons,
            reference.max_codons
        ]
    );

    let shape = TableShape {
        n_aa: rust.n_aa,
        max_codons: rust.max_codons,
    };
    assert_table_eq("E_h", &rust.e_h, &reference.e_h, shape);
    assert_table_eq("E_s", &rust.e_s, &reference.e_s, shape);
    assert_table_eq("E_b", &rust.e_b, &reference.e_b, shape);
    assert_table_eq("E_i", &rust.e_i, &reference.e_i, shape);
    assert_table_eq("E", &rust.e, &reference.e, shape);

    // Untouched by the single-loop driver: E_m / M / N / O stay at INF on both sides.
    assert_inf("E_m (rust)", &rust.e_m);
    assert_inf("M (rust)", &rust.m);
    assert_inf("N (rust)", &rust.n);
    assert_inf("O (rust)", &rust.o);
    assert_inf_fixture("E_m (reference)", &reference.e_m);
    assert_inf_fixture("M (reference)", &reference.m);
    assert_inf_fixture("N (reference)", &reference.n);
    assert_inf_fixture("O (reference)", &reference.o);
}

#[test]
fn per_cell_parity_lambda_1() {
    run_parity_for_lambda(1.0, 10);
}
