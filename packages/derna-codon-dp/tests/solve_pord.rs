//! Hull-only Pareto-front parity vs brute-force enumeration fixtures.
//!
//! Loads `tests/data/codon_pareto_{toy,reduced}.json`
//! (the brute-force *Pareto front*) and asserts the Rust port's
//! `solve_pord` output recovers every Pareto-front cell that lies on the
//! convex hull of `(MFE, CAI)`. A λ-sweep of a scalar objective can only
//! recover hull points — concave-up Pareto-front
//! cells are documented-as-missed. The reduced fixture has 3 Pareto-front
//! cells, only 2 of which lie on the hull; the test computes that subset
//! at fixture load time and asserts recovery only against it.
//!
//! Co-optimal-tie cells (different `rna_seq` at the same `(MFE, CAI)`)
//! log informationally and do not fail the test.
//!
//! Comparison: `mfe_kcal_mol` matches at the dekacal-int level (`round(mfe
//! * 100)`); `cai` matches at a 2-ULP tolerance. The 2-ULP tolerance
//! covers `serde_json`'s 1-ULP drift on half-way decimals — JSON's
//! float round-trip is not bit-exact at every half-way input, and the
//! brute-force corpus is not bit-equal across `json.dumps` ↔
//! `serde_json::from_str`. The DP's f64 CAI accumulation matches
//! Python's at the underlying float level (see the
//! `tests/solve_brd.rs` parity tests, which dump CAI bits explicitly
//! and assert bit-equality against v1's `solve_brd`).

use std::fs;
use std::path::PathBuf;

use derna_codon_dp::codon_usage::CodonUsage;
use derna_codon_dp::{solve_pord, BrdOptions, ParetoPoint};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PointFixture {
    rna_seq: String,
    structure: String,
    mfe_kcal_mol: f64,
    cai: f64,
}

#[derive(Debug, Deserialize)]
struct FrontFixture {
    dangling_5_prime: bool,
    front: Vec<PointFixture>,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    codon_mask: Option<Vec<Vec<usize>>>,
}

#[derive(Debug, Deserialize)]
struct CorpusFixture {
    protein: String,
    fronts: Vec<FrontFixture>,
    metadata: FixtureMetadata,
}

fn corpus_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/{name}"))
}

fn load_corpus(name: &str) -> CorpusFixture {
    let path = corpus_path(name);
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn rust_pord(protein: &str, codon_mask: Option<Vec<Vec<usize>>>) -> Vec<ParetoPoint> {
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask,
    };
    solve_pord(protein, &codon_usage, &options).expect("solve_pord")
}

fn mfe_dekacal(x: f64) -> i32 {
    (x * 100.0).round() as i32
}

/// Deduplicate `front` by `(mfe_dekacal, cai_bits)` and sort by `mfe`
/// ascending — pre-pass for the convex-hull scan.
fn dedup_and_sort(front: &[PointFixture]) -> Vec<&PointFixture> {
    let mut seen: std::collections::HashSet<(i32, u64)> = std::collections::HashSet::new();
    let mut unique: Vec<&PointFixture> = Vec::new();
    for cell in front {
        let key = (mfe_dekacal(cell.mfe_kcal_mol), cell.cai.to_bits());
        if seen.insert(key) {
            unique.push(cell);
        }
    }
    unique.sort_by(|a, b| {
        a.mfe_kcal_mol
            .partial_cmp(&b.mfe_kcal_mol)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    unique
}

/// Convex-hull subset of a Pareto front. The front is collapsed to one
/// representative per `(mfe_dekacal, cai_bits)` key first (co-optimal
/// ties collapse), then the upper convex hull in `(mfe, cai)` is
/// retained — i.e., a point is dropped if a strictly convex
/// combination of two front neighbours dominates it under any linear
/// scoring function `λ·MFE − (1−λ)·CAI`. A λ-sweep recovers exactly this
/// subset.
fn convex_hull_subset(front: &[PointFixture]) -> Vec<&PointFixture> {
    let unique = dedup_and_sort(front);
    let mut hull: Vec<&PointFixture> = Vec::new();
    for cell in unique {
        while hull.len() >= 2 && !concave_turn(hull[hull.len() - 2], hull[hull.len() - 1], cell) {
            hull.pop();
        }
        hull.push(cell);
    }
    hull
}

/// `true` iff `O → A → B` is a concave-down (right) turn in `(MFE, CAI)`.
/// Equivalent to keeping points whose CAI lies strictly above the line
/// between the previous and next hull points.
fn concave_turn(o: &PointFixture, a: &PointFixture, b: &PointFixture) -> bool {
    let (ox, oy) = (o.mfe_kcal_mol, o.cai);
    let (ax, ay) = (a.mfe_kcal_mol, a.cai);
    let (bx, by) = (b.mfe_kcal_mol, b.cai);
    let cross = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
    cross < 0.0
}

/// Two `f64`s match if their bits differ by at most this many ULPs.
/// Covers `serde_json`'s 1-ULP drift on half-way decimals; the DP's
/// f64 CAI itself matches Python's bit-for-bit (see `solve_brd.rs`).
const CAI_TOLERANCE_ULPS: u64 = 2;

fn cai_matches(rust_cai: f64, fixture_cai: f64) -> bool {
    let rust_bits = rust_cai.to_bits();
    let fixt_bits = fixture_cai.to_bits();
    rust_bits.abs_diff(fixt_bits) <= CAI_TOLERANCE_ULPS
}

/// Hull recovery: every convex-hull cell of the brute-force Pareto
/// front must appear in the Rust output (CAI matched at the 2-ULP
/// tolerance). The Rust output may carry additional points (co-optimal
/// ties) — those log informationally.
fn assert_hull_recovered(corpus: &CorpusFixture, rust_points: &[ParetoPoint], label: &str) {
    let d0_front = corpus
        .fronts
        .iter()
        .find(|f| !f.dangling_5_prime)
        .expect("d0 front exists");
    let hull = convex_hull_subset(&d0_front.front);
    let mut missing: Vec<&PointFixture> = Vec::new();
    for cell in &hull {
        let cell_mfe = mfe_dekacal(cell.mfe_kcal_mol);
        let found = rust_points
            .iter()
            .any(|p| mfe_dekacal(p.mfe_kcal_mol) == cell_mfe && cai_matches(p.cai, cell.cai));
        if !found {
            missing.push(cell);
        }
    }
    if !missing.is_empty() {
        panic!(
            "[{label}] {} convex-hull cells missing from Rust output (of {} hull cells, \
             {} Pareto-front cells). First: mfe={} cai={} rna={} structure={}. Rust hull size = {}.",
            missing.len(),
            hull.len(),
            d0_front.front.len(),
            missing[0].mfe_kcal_mol,
            missing[0].cai,
            missing[0].rna_seq,
            missing[0].structure,
            rust_points.len(),
        );
    }
    log_co_optimal_ties(&hull, rust_points, label);
}

fn log_co_optimal_ties(hull: &[&PointFixture], rust_points: &[ParetoPoint], label: &str) {
    for p in rust_points {
        let p_mfe = mfe_dekacal(p.mfe_kcal_mol);
        let same_key_cells: Vec<&PointFixture> = hull
            .iter()
            .copied()
            .filter(|c| mfe_dekacal(c.mfe_kcal_mol) == p_mfe && cai_matches(p.cai, c.cai))
            .collect();
        if same_key_cells.is_empty() {
            eprintln!(
                "[{label}] Rust hull point not in brute-force convex-hull subset: mfe={} cai={} rna={}",
                p.mfe_kcal_mol, p.cai, p.rna_seq,
            );
        } else if !same_key_cells.iter().any(|c| c.rna_seq == p.rna_seq) {
            eprintln!(
                "[{label}] co-optimal tie at mfe={} cai={}: rust rna={} not in brute-force set ({:?})",
                p.mfe_kcal_mol,
                p.cai,
                p.rna_seq,
                same_key_cells.iter().map(|c| &c.rna_seq).collect::<Vec<_>>(),
            );
        }
    }
}

#[test]
fn hull_recovery_toy() {
    let corpus = load_corpus("codon_pareto_toy.json");
    assert_eq!(corpus.protein, "GLAGPR");
    let rust_points = rust_pord(&corpus.protein, corpus.metadata.codon_mask.clone());
    assert_hull_recovered(&corpus, &rust_points, "toy");
}

#[test]
fn hull_recovery_reduced() {
    let corpus = load_corpus("codon_pareto_reduced.json");
    assert_eq!(corpus.protein, "MYGKIIFVLLLSGIV");
    assert!(corpus.metadata.codon_mask.is_some());
    let rust_points = rust_pord(&corpus.protein, corpus.metadata.codon_mask.clone());
    assert_hull_recovered(&corpus, &rust_points, "reduced");
}
