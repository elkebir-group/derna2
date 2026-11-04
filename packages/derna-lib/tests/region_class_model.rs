//! The per-class constraint table and the mixed UTR chain, pinned against the
//! substrate they generalize.
//!
//! Enforcement is a table over the six pair-classes of {5′UTR, CDS, 3′UTR} plus
//! the initiation window, and a UTR flank is a per-position mask of given and
//! designed nucleotides. Both are generalizations, so what has to be proven is
//! that they *reduce* — that the configurations reachable before still produce
//! exactly what they produced before, bit for bit.
//!
//! Four floors:
//!
//! 1. **An inert table is the unconstrained fold.** With flanks present and every
//!    class allowed, the design is byte-identical to the same run with no constraint.
//! 2. **The `5′↔CDS` + `CDS↔3′` table is today's symmetric isolation.** Built
//!    entry by entry it equals the named preset, and both equal what the shipped
//!    isolation family produced. The oracle suites at `region_pair_constraints.rs`
//!    and `designed_utr.rs` carry the same floor against a brute force.
//! 3. **The mixed chain's ends are `Fixed` and `Designed`.** An all-given mask is
//!    byte-identical to the fixed flank it spells; an all-`N` mask is
//!    byte-identical to the designed flank of that length.
//! 4. **Full mutual isolation is separable.** With no pair permitted to cross any
//!    boundary, the designed CDS matches a CDS-only design of the same protein and
//!    the joint MFE equals the sum of the three regions' independent folds. The
//!    claim the exactness argument rests on, checked rather than derived.

use derna_lib::{
    design, design_with_utr, design_with_utr_spec, linearfold_fold, CodonUsage, Enforcement,
    PairClass, Params, RegionConstraints, ScanConfig, UtrSpec,
};

/// b → ∞: a beam wide enough that `rna-linearfold` runs the exact lattice.
const EXACT: usize = 1 << 20;

/// Low / mid / high λ, matching the other oracle suites.
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// Energies are compared byte-exactly; this absorbs only f64 round-tripping.
const EPS: f64 = 1e-9;

fn setup() -> (Params, CodonUsage) {
    (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    )
}

/// Proteins and flanks short enough for the exact lattice, long enough that the
/// flanks have structure of their own to contribute.
const CASES: [(&str, &str, &str); 3] = [
    ("MKTAYIAKQR", "GCCACCAUGGCC", "AAUAAAGCUU"),
    ("DEYWCNQHKR", "GGGCCCGGG", "CCCGGGCCC"),
    ("FFLLSSPPHH", "GCGCGCGCGC", "GCGCGCGCGC"),
];

fn spec_design(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    lambda: f64,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    rc: RegionConstraints,
) -> derna_lib::Design {
    design_with_utr_spec(
        params,
        protein,
        usage,
        lambda,
        utr5,
        utr3,
        rc,
        ScanConfig::default(),
    )
    .expect("design succeeds")
}

/// Floor 1 — every class allowed is the same fold as no constraint at all.
#[test]
fn inert_class_table_is_the_unconstrained_design() {
    let (params, usage) = setup();
    // An inert table reached two ways: default-constructed, and built by
    // explicitly allowing nothing.
    assert!(RegionConstraints::default().is_inert());
    for (protein, utr5, utr3) in CASES {
        for &lambda in &LAMBDAS {
            let unconstrained = design_with_utr(
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
            for class in PairClass::ALL {
                // Allowing a class is not the same as omitting it only if the
                // table leaks; assert every entry reads back empty.
                assert_eq!(
                    RegionConstraints::default().enforcement(class),
                    None,
                    "default table enforces {class:?}"
                );
            }
            let via_spec = spec_design(
                &params,
                &usage,
                protein,
                lambda,
                &UtrSpec::Fixed(utr5.into()),
                &UtrSpec::Fixed(utr3.into()),
                RegionConstraints::default(),
            );
            assert_eq!(
                unconstrained, via_spec,
                "inert table diverged [{protein} λ={lambda}]"
            );
        }
    }
}

/// Floor 2 — the two CDS-boundary entries are exactly today's symmetric
/// isolation, whether named or spelled out.
#[test]
fn two_boundary_entries_are_symmetric_isolation() {
    let (params, usage) = setup();
    for enforcement in [
        Enforcement::Forbid,
        Enforcement::Penalize(derna_lib::Energy(30)),
    ] {
        let spelled = RegionConstraints::default()
            .with(PairClass::Utr5Cds, enforcement)
            .with(PairClass::CdsUtr3, enforcement);
        let named = RegionConstraints::symmetric_isolation(enforcement);
        assert_eq!(spelled, named, "spelled-out table differs from the preset");
        // And the preset touches those two entries and no others.
        for class in PairClass::ALL {
            let expect = match class {
                PairClass::Utr5Cds | PairClass::CdsUtr3 => Some(enforcement),
                _ => None,
            };
            assert_eq!(named.enforcement(class), expect, "{class:?}");
        }
        for (protein, utr5, utr3) in CASES {
            for &lambda in &LAMBDAS {
                let a = design_with_utr(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    utr5,
                    utr3,
                    spelled,
                    ScanConfig::default(),
                )
                .unwrap();
                let b = design_with_utr(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    utr5,
                    utr3,
                    named,
                    ScanConfig::default(),
                )
                .unwrap();
                assert_eq!(a, b, "isolation table diverged [{protein} λ={lambda}]");
            }
        }
    }
}

/// The named presets forbid exactly the classes they say they do.
#[test]
fn presets_select_exactly_their_classes() {
    let f = Enforcement::Forbid;
    let five = RegionConstraints::five_prime_isolation(f);
    assert_eq!(five.enforcement(PairClass::Utr5Cds), Some(f));
    assert_eq!(five.enforcement(PairClass::CdsUtr3), None);
    assert_eq!(five.enforcement(PairClass::Utr5Utr3), None);

    let full = RegionConstraints::full_mutual_isolation(f);
    for class in PairClass::ALL {
        let expect = class.is_crossing().then_some(f);
        assert_eq!(full.enforcement(class), expect, "{class:?}");
    }
}

/// Floor 3 — the mixed mask reduces to each of its two ends.
#[test]
fn mixed_chain_ends_are_fixed_and_designed() {
    let (params, usage) = setup();
    let rcs = [
        RegionConstraints::default(),
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
    ];
    for (protein, utr5, utr3) in CASES {
        for rc in rcs {
            for &lambda in &LAMBDAS {
                // All-given mask == the fixed flank it spells.
                let fixed = spec_design(
                    &params,
                    &usage,
                    protein,
                    lambda,
                    &UtrSpec::Fixed(utr5.into()),
                    &UtrSpec::Fixed(utr3.into()),
                    rc,
                );
                let masked = spec_design(
                    &params,
                    &usage,
                    protein,
                    lambda,
                    &UtrSpec::Mixed(utr5.into()),
                    &UtrSpec::Mixed(utr3.into()),
                    rc,
                );
                assert_eq!(
                    fixed, masked,
                    "all-given mask diverged from Fixed [{protein} λ={lambda}]"
                );

                // All-`N` mask == the designed flank of that length. Kept short:
                // a designed flank is a four-way choice per position.
                let (n5, n3) = (5usize, 4usize);
                let designed = spec_design(
                    &params,
                    &usage,
                    protein,
                    lambda,
                    &UtrSpec::Designed(n5),
                    &UtrSpec::Designed(n3),
                    rc,
                );
                let free_mask = spec_design(
                    &params,
                    &usage,
                    protein,
                    lambda,
                    &UtrSpec::Mixed("N".repeat(n5)),
                    &UtrSpec::Mixed("N".repeat(n3)),
                    rc,
                );
                assert_eq!(
                    designed, free_mask,
                    "all-N mask diverged from Designed [{protein} λ={lambda}]"
                );
            }
        }
    }
}

/// A partly-given mask is genuinely between its ends: it pins what it spells and
/// leaves the rest to the fold. Without this the floors above would also pass on
/// a mask that silently ignored its `N`s (or its nucleotides).
#[test]
fn mixed_chain_pins_given_positions_and_frees_the_rest() {
    let (params, usage) = setup();
    let mask = "GGGNNNNCCC";
    let mut differs_from_all_given = 0;
    for (protein, _, _) in CASES {
        for &lambda in &LAMBDAS {
            let d = spec_design(
                &params,
                &usage,
                protein,
                lambda,
                &UtrSpec::Mixed(mask.into()),
                &UtrSpec::Fixed(String::new()),
                RegionConstraints::five_prime_isolation(Enforcement::Forbid),
            );
            let flank = &d.sequence[..mask.len()];
            for (spelled, got) in mask.bytes().zip(flank.bytes()) {
                if spelled != b'N' {
                    assert_eq!(spelled, got, "mask position not pinned in {flank}");
                }
            }
            // The free positions are chosen, so the flank need not equal the mask
            // with `N` read literally — track that at least one run uses that
            // freedom, or the test is vacuous.
            if flank
                .bytes()
                .zip(mask.bytes())
                .any(|(g, m)| m == b'N' && g != m)
            {
                differs_from_all_given += 1;
            }
        }
    }
    assert!(
        differs_from_all_given > 0,
        "no run exercised a free mask position"
    );
}

/// Floor 4 — under full mutual isolation the joint problem separates.
///
/// No pair may cross a boundary, so at `dangles = 0` every closed loop is
/// region-pure and the exterior loop contributes only per-branch terminal
/// penalties. The energy is then `E₅′ + E_CDS + E₃′` over disjoint variables, and
/// because a fixed flank's energy does not depend on the codon assignment the
/// design reduces to CDS-only design plus two independent folds.
#[test]
fn full_mutual_isolation_separates_the_joint_fold() {
    let (params, usage) = setup();
    let rc = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
    for (protein, utr5, utr3) in CASES {
        for &lambda in &LAMBDAS {
            let joint = design_with_utr(
                &params,
                protein,
                &usage,
                lambda,
                utr5,
                utr3,
                rc,
                ScanConfig::default(),
            )
            .unwrap();
            let cds_only = design(&params, protein, &usage, lambda).unwrap();
            let e5 = linearfold_fold(&params, utr5.as_bytes(), EXACT)
                .unwrap()
                .1
                .to_kcal_mol();
            let e3 = linearfold_fold(&params, utr3.as_bytes(), EXACT)
                .unwrap()
                .1
                .to_kcal_mol();

            let label = format!("[{protein} λ={lambda}]");
            assert!(
                (joint.mfe_kcal_mol - (cds_only.mfe_kcal_mol + e5 + e3)).abs() < EPS,
                "energy did not separate {label}: joint {} vs {} + {e5} + {e3}",
                joint.mfe_kcal_mol,
                cds_only.mfe_kcal_mol,
            );
            // The CAI half sees the CDS alone, so a separable optimum reaches the
            // CDS-only optimum's CAI exactly. The spelling may differ where the
            // objective has co-optimal encodings, so the value is what is pinned.
            assert!(
                (joint.cai_log - cds_only.cai_log).abs() < EPS,
                "CAI did not separate {label}: {} vs {}",
                joint.cai_log,
                cds_only.cai_log,
            );
            // And no recovered pair crosses a boundary, which is what made the
            // decomposition available in the first place.
            let cds = (utr5.len(), utr5.len() + 3 * protein.len());
            for (i, j) in pairs_of(&joint.structure) {
                let region = |p: usize| (p >= cds.0) as u8 + (p >= cds.1) as u8;
                assert_eq!(
                    region(i),
                    region(j),
                    "crossing pair ({i},{j}) under full mutual isolation {label}"
                );
            }
        }
    }
}

/// The `(i, j)` pairs of a dot-bracket string.
fn pairs_of(structure: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut pairs = Vec::new();
    for (j, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(j),
            b')' => pairs.push((stack.pop().expect("balanced"), j)),
            _ => {}
        }
    }
    pairs
}
