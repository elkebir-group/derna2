use derna_codon_dp::codon_usage::CodonUsage;
use derna_codon_dp::{
    solve_pord, solve_pord_with_telemetry, BrdOptions, ParetoPoint, SweepBracketStopReason,
};

fn bit_equal(a: &ParetoPoint, b: &ParetoPoint) -> bool {
    a.rna_seq == b.rna_seq
        && a.structure == b.structure
        && a.mfe_kcal_mol.to_bits() == b.mfe_kcal_mol.to_bits()
        && a.cai.to_bits() == b.cai.to_bits()
        && a.lambda_.to_bits() == b.lambda_.to_bits()
}

#[test]
fn solve_pord_deterministic_and_telemetry_matches_points() {
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };

    let plain = solve_pord("GLAGPR", &codon_usage, &options).expect("solve_pord");
    let (measured, telemetry) =
        solve_pord_with_telemetry("GLAGPR", &codon_usage, &options).expect("solve_pord");

    assert_eq!(plain.len(), measured.len());
    for (left, right) in plain.iter().zip(&measured) {
        assert!(bit_equal(left, right));
    }

    assert_eq!(telemetry.total_solves(), telemetry.brd_solves.len());
    assert_eq!(telemetry.memo_misses, telemetry.total_solves());
    assert_eq!(
        telemetry.sweep_brackets.len(),
        telemetry.total_solves().saturating_sub(2) * 2 + 1
    );
    assert!(telemetry
        .sweep_brackets
        .iter()
        .any(|bracket| bracket.stop_reason == SweepBracketStopReason::Bisect));
    for bracket in &telemetry.sweep_brackets {
        assert!(bracket.lo < bracket.hi);
        assert_eq!(
            bracket.mid.is_some(),
            bracket.stop_reason == SweepBracketStopReason::Bisect
        );
    }
}
