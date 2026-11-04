//! Differential-fuzz oracle for the internal-loop reformulation.
//!
//! The fixed parity fixtures (`brd_parity` GLPE-78 + the curated λ sweep,
//! `fold_parity`) exercise only the loop topologies those sequences happen
//! to hit. The `O(n²·L)` Lyngsø–Zuker reformulation of the generic
//! internal-loop closure can pass every fixture and still diverge on an
//! internal-loop shape — an asymmetric bulge, a near-`SINGLE_MAX_LEN`
//! interior, a broken-codon flank alignment — that the fixtures never touch.
//!
//! This harness runs random protein × random λ through *both*
//! [`InternalLoopMode::Naive`] (the validated reference) and
//! [`InternalLoopMode::Lz`] (the reformulation under test). It asserts the
//! **BRD optimum is byte-exact** — the spelled RNA (translatable), the MFE,
//! and the CAI all match Naive — and treats the dot-bracket **structure as a
//! diagnostic**, not a hard failure.
//!
//! The structure is relaxed deliberately. The `O(n²·L)` Lyngsø–Zuker
//! reduction aggregates inner pairs by loop-size diagonal, an order that
//! cannot reproduce Naive's first-writer (p_beam processing-order) tie-break
//! on **degenerate equal-BRD folds** without disproportionate cost — and the
//! property it would protect is only *which* of several equal-`(MFE, CAI)`
//! folds gets drawn, never the optimum itself. The contract
//! is therefore BRD byte-exact, translatable and phantom-free, with the
//! structure as a diagnostic. A unified single-branch pull with
//! tie-break-preserving minima would restore full structure parity.
//!
//! Set `DERNA_LZ_STRICT_STRUCTURE=1` to re-arm the full structure assertion —
//! a sharp dev-time detector for coverage gaps in the reduction (a dropped
//! co-optimal candidate is invisible to the BRD check but flips a tie's
//! structure).

use derna_lib::{design_with_beam_mode, CodonUsage, Design, InternalLoopMode, Params};

/// The 20 standard amino acids, single-letter. Mixing two- and one-codon
/// residues exercises both single- and dual-lane DFA units.
const AMINO_ACIDS: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

/// λ values spanning the BRD front: CAI-dominated, balanced, and
/// MFE-dominated (the MFE-priority anchor the corpus benchmarks use).
const LAMBDAS: &[f64] = &[0.0001, 0.3, 0.5, 0.9, 0.99999];

/// A small deterministic xorshift64 PRNG — keeps the harness reproducible
/// without a dev-dependency, and lets a failing case be replayed from its
/// seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the zero fixed point.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }

    fn amino_acid(&mut self) -> char {
        AMINO_ACIDS[(self.next_u64() as usize) % AMINO_ACIDS.len()] as char
    }
}

/// A random protein of length `len` over the 20-aa alphabet.
fn random_protein(rng: &mut Rng, len: usize) -> String {
    (0..len).map(|_| rng.amino_acid()).collect()
}

/// The oracle comparison: BRD optimum byte-exact (sequence + MFE + CAI),
/// structure as a diagnostic (re-armed to a hard assertion under
/// `DERNA_LZ_STRICT_STRUCTURE`).
fn assert_byte_exact(naive: &Design, lz: &Design, protein: &str, lambda: f64, seed: u64) {
    let ctx = format!("seed={seed} λ={lambda} protein={protein}");
    // Translatable: identical to the validated reference's spelled CDS.
    assert_eq!(naive.sequence, lz.sequence, "sequence mismatch: {ctx}");
    // BRD byte-exact: the optimum the scan minimizes.
    assert_eq!(
        naive.mfe_kcal_mol, lz.mfe_kcal_mol,
        "MFE mismatch: {ctx} (naive {} vs lz {})",
        naive.mfe_kcal_mol, lz.mfe_kcal_mol,
    );
    assert_eq!(
        naive.cai_log, lz.cai_log,
        "CAI-log mismatch: {ctx} (naive {} vs lz {})",
        naive.cai_log, lz.cai_log,
    );
    assert_eq!(naive.cai, lz.cai, "CAI mismatch: {ctx}");
    // Structure: equal-BRD folds may tie and resolve differently (diagnostic).
    if naive.structure != lz.structure {
        eprintln!(
            "structure divergence on equal-BRD tie: {ctx}\n  naive {}\n  lz    {}",
            naive.structure, lz.structure,
        );
        assert!(
            std::env::var_os("DERNA_LZ_STRICT_STRUCTURE").is_none(),
            "structure mismatch under DERNA_LZ_STRICT_STRUCTURE: {ctx}",
        );
    }
}

/// Run one protein × all λ through both modes and assert byte-exact parity.
fn check_protein(params: &Params, usage: &CodonUsage, protein: &str, seed: u64) {
    for &lambda in LAMBDAS {
        let naive = design_with_beam_mode(
            params,
            protein,
            usage,
            lambda,
            usize::MAX,
            InternalLoopMode::Naive,
        )
        .expect("naive design");
        let lz = design_with_beam_mode(
            params,
            protein,
            usage,
            lambda,
            usize::MAX,
            InternalLoopMode::Lz,
        )
        .expect("lz design");
        assert_byte_exact(&naive, &lz, protein, lambda, seed);
    }
}

/// Sweep `cases` random proteins with lengths drawn from `[lo, hi]` aa,
/// each across the λ front, asserting Naive ≡ Lz byte-for-byte.
fn fuzz_sweep(base_seed: u64, cases: usize, lo: usize, hi: usize) {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for i in 0..cases {
        let seed = base_seed.wrapping_add(i as u64);
        let mut rng = Rng::new(seed);
        let len = rng.range(lo, hi);
        let protein = random_protein(&mut rng, len);
        check_protein(&params, &usage, &protein, seed);
    }
}

/// Fast default gate: a spread of short/medium proteins across the λ front.
/// Runs in the standard `cargo test` pass.
#[test]
fn lz_matches_naive_quick() {
    fuzz_sweep(0xA5A5_0001, 24, 8, 26);
}

/// Heavier sweep — more cases and longer, more loop-rich proteins — behind
/// `--ignored`, mirroring the `brd_parity` rigorous oracle. Run with
/// `cargo test -p derna-lib --release --test lz_fuzz -- --ignored`.
#[test]
#[ignore = "rigorous differential-fuzz sweep; run with --ignored --release"]
fn lz_matches_naive_rigorous() {
    fuzz_sweep(0xC0FF_EE01, 120, 20, 60);
}
