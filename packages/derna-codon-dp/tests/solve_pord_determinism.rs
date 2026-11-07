//! Bit-for-bit reproducibility of `solve_pord`.
//!
//! Two invocations of `solve_pord` with identical arguments must
//! produce identical `Vec<ParetoPoint>` results — field-by-field,
//! float bits included. This is load-bearing for downstream parity
//! measurements against reference fronts: non-deterministic output would
//! make hull-diff regression tests flaky.
//!
//! Likely sources of non-determinism (none of which the implementation
//! should reach): hash-based memo, non-FIFO bisection queue, unstable
//! sort, parallelism. The implementation uses
//! [`std::collections::VecDeque`] (FIFO), a hand-rolled
//! insertion-ordered memo keyed by `λ.to_bits()`, and a stable sort by
//! `(mfe_kcal_mol, cai, rna_seq)`. Toy 6-aa `GLAGPR` is enough to
//! catch any regression — fast (~ms scale) and definitive.

use derna_codon_dp::codon_usage::CodonUsage;
use derna_codon_dp::{
    solve_pord, solve_pord_with_telemetry, BrdOptions, ParetoPoint, SweepBracketStopReason,
};

fn run_once() -> Vec<ParetoPoint> {
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    solve_pord("GLAGPR", &codon_usage, &options).expect("solve_pord")
}

fn run_once_with_telemetry() -> Vec<ParetoPoint> {
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    let (points, telemetry) =
        solve_pord_with_telemetry("GLAGPR", &codon_usage, &options).expect("solve_pord");
    assert_eq!(telemetry.total_solves(), telemetry.brd_solves.len());
    assert_eq!(telemetry.memo_misses, telemetry.brd_solves.len());
    assert!(telemetry.total_solves() > 0);
    assert_eq!(
        telemetry.sweep_brackets.len(),
        telemetry.total_solves().saturating_sub(2) * 2 + 1
    );
    for solve in &telemetry.brd_solves {
        assert!(solve.lambda_.is_finite());
        assert!(solve.elapsed_seconds.is_finite());
        assert!(solve.elapsed_seconds >= 0.0);
        assert!(solve.mfe_kcal_mol.is_finite());
        assert!(solve.cai.is_finite());
    }
    for bracket in &telemetry.sweep_brackets {
        assert!(bracket.lo.is_finite());
        assert!(bracket.hi.is_finite());
        assert!(bracket.lo < bracket.hi);
        assert!(bracket.width > 0.0);
        assert!(bracket.width_threshold > 0.0);
        assert!(bracket.lo_mfe_kcal_mol.is_finite());
        assert!(bracket.lo_cai.is_finite());
        assert!(bracket.hi_mfe_kcal_mol.is_finite());
        assert!(bracket.hi_cai.is_finite());
        match bracket.stop_reason {
            SweepBracketStopReason::Bisect => {
                let mid = bracket.mid.expect("bisect bracket records mid");
                assert!(mid > bracket.lo);
                assert!(mid < bracket.hi);
            }
            _ => assert!(bracket.mid.is_none()),
        }
    }
    points
}

fn bit_equal(a: &ParetoPoint, b: &ParetoPoint) -> bool {
    a.rna_seq == b.rna_seq
        && a.structure == b.structure
        && a.mfe_kcal_mol.to_bits() == b.mfe_kcal_mol.to_bits()
        && a.cai.to_bits() == b.cai.to_bits()
        && a.lambda_.to_bits() == b.lambda_.to_bits()
}

#[test]
fn two_runs_match_bit_for_bit() {
    let a = run_once();
    let b = run_once();
    assert_eq!(
        a.len(),
        b.len(),
        "solve_pord output length differs: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            bit_equal(pa, pb),
            "ParetoPoint[{i}] differs:\n  a: rna={} struct={} mfe.bits={:016x} cai.bits={:016x} lambda.bits={:016x}\n  b: rna={} struct={} mfe.bits={:016x} cai.bits={:016x} lambda.bits={:016x}",
            pa.rna_seq,
            pa.structure,
            pa.mfe_kcal_mol.to_bits(),
            pa.cai.to_bits(),
            pa.lambda_.to_bits(),
            pb.rna_seq,
            pb.structure,
            pb.mfe_kcal_mol.to_bits(),
            pb.cai.to_bits(),
            pb.lambda_.to_bits(),
        );
    }
}

#[test]
fn telemetry_does_not_change_output() {
    let plain = run_once();
    let measured = run_once_with_telemetry();
    assert_eq!(
        plain.len(),
        measured.len(),
        "telemetry changed solve_pord output length"
    );
    for (i, (pa, pb)) in plain.iter().zip(measured.iter()).enumerate() {
        assert!(bit_equal(pa, pb), "telemetry changed ParetoPoint[{i}]");
    }
}
