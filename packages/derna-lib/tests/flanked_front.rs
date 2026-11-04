//! The front-valued fold, and an energy ceiling over it, on a molecule that
//! carries 5' and 3' flanks — supplied, designed, or per-position mixed, with or
//! without a profile pricing the designed positions.
//!
//! A ceiling on such a molecule is well defined because the energy accumulator
//! spans the whole folded molecule; the adaptation axis stays the coding
//! sequence's alone, since a flank position credits no CAI however it is
//! spelled. Both halves of that are asserted here against an enumeration that
//! folds the same molecules by a different route.

mod common;

use common::{eliminate, enumerate_all, front_objectives, Objective};
use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::free_fold::{sequence_gap, CdsWindow};
use derna_lib::{
    design_front_bounded, design_front_bounded_with_utr, design_front_walk_with_utr,
    design_with_utr, dfa, Bounded, Bounds, Ceiling, CodonUsage, Energy, NucSet, Nucleotide, Params,
    Prune, RegionConstraints, ScanConfig, UtrProfile, UtrSpec, WalkOptions,
};

const FLANK5: &str = "GCAUUG";
const FLANK3: &str = "AUCGA";

fn fixed(s: &str) -> UtrSpec {
    UtrSpec::Fixed(s.to_string())
}

fn fixtures() -> (Params, CodonUsage) {
    (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    )
}

fn feasible(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    utr5: &str,
    utr3: &str,
    bounds: Bounds,
) -> derna_lib::BoundedFront {
    spec_feasible(
        params,
        usage,
        protein,
        &fixed(utr5),
        &fixed(utr3),
        UtrProfile::inert(),
        bounds,
    )
}

fn spec_feasible(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
    bounds: Bounds,
) -> derna_lib::BoundedFront {
    match design_front_bounded_with_utr(
        params,
        protein,
        usage,
        utr5,
        utr3,
        profile,
        Prune::None,
        bounds,
        1,
    )
    .expect("the flanked fold answers")
    {
        Bounded::Feasible(f) => f,
        Bounded::Unreachable { .. } => panic!("expected a feasible answer"),
    }
}

/// Whole-molecule re-fold on a path that is not the one under test.
fn refold_molecule(params: &Params, sequence: &str, utr5_len: usize, residues: usize) -> i32 {
    let report = sequence_gap(
        params,
        sequence,
        CdsWindow::new(utr5_len, residues),
        RegionConstraints::default(),
    )
    .expect("an inert constraint folds any sequence");
    (report.free_mfe_kcal_mol * 100.0).round() as i32
}

#[test]
fn empty_flanks_reproduce_the_coding_only_front() {
    let (params, usage) = fixtures();
    for protein in common::CASES {
        let flanked = feasible(&params, &usage, protein, "", "", Bounds::UNBOUNDED);
        let plain =
            match design_front_bounded(&params, protein, &usage, Prune::None, Bounds::UNBOUNDED, 1)
                .unwrap()
            {
                Bounded::Feasible(f) => f,
                Bounded::Unreachable { .. } => unreachable!(),
            };
        assert_eq!(flanked.front, plain.front, "{protein}: front differs");
        assert_eq!(flanked.optimum_kcal_mol, plain.optimum_kcal_mol);
    }
}

#[test]
fn the_recovered_molecule_carries_the_flanks_and_the_protein() {
    let (params, usage) = fixtures();
    let protein = "MYGKIIFVLM";
    let front = feasible(&params, &usage, protein, FLANK5, FLANK3, Bounds::UNBOUNDED);
    assert!(!front.front.is_empty());
    for point in &front.front {
        assert_eq!(
            point.sequence.len(),
            FLANK5.len() + 3 * protein.len() + FLANK3.len(),
            "the recovered molecule spans flanks and coding sequence",
        );
        assert!(point.sequence.starts_with(FLANK5));
        assert!(point.sequence.ends_with(FLANK3));
        assert_eq!(point.structure.len(), point.sequence.len());
    }
}

#[test]
fn every_flanked_point_refolds_to_its_reported_energy() {
    let (params, usage) = fixtures();
    let protein = "MYGKIIFVLM";
    let front = feasible(&params, &usage, protein, FLANK5, FLANK3, Bounds::UNBOUNDED);
    for point in &front.front {
        let reported = (point.mfe_kcal_mol * 100.0).round() as i32;
        assert_eq!(
            reported,
            refold_molecule(&params, &point.sequence, FLANK5.len(), protein.len()),
            "traceback energy is not the molecule's own",
        );
    }
}

#[test]
fn the_flanked_corner_agrees_with_the_scalar_path() {
    let (params, usage) = fixtures();
    let protein = "MYGKIIFVLM";
    let corner = design_with_utr(
        &params,
        protein,
        &usage,
        1.0,
        FLANK5,
        FLANK3,
        RegionConstraints::default(),
        ScanConfig::default(),
    )
    .unwrap();
    let front = feasible(&params, &usage, protein, FLANK5, FLANK3, Bounds::UNBOUNDED);
    assert_eq!(front.optimum_kcal_mol, corner.mfe_kcal_mol);
    assert_eq!(front.front[0].mfe_kcal_mol, corner.mfe_kcal_mol);
}

#[test]
fn a_ceiling_below_the_flanked_minimum_is_unreachable_without_searching() {
    let (params, usage) = fixtures();
    let protein = "MYGKIIFVLM";
    let front = feasible(&params, &usage, protein, FLANK5, FLANK3, Bounds::UNBOUNDED);
    let below = Energy((front.optimum_kcal_mol * 100.0).round() as i32 - 1);
    match design_front_bounded_with_utr(
        &params,
        protein,
        &usage,
        &fixed(FLANK5),
        &fixed(FLANK3),
        UtrProfile::inert(),
        Prune::None,
        Bounds::with_ceiling(Ceiling::Absolute(below)),
        1,
    )
    .unwrap()
    {
        Bounded::Unreachable {
            achievable_kcal_mol,
            ..
        } => assert_eq!(achievable_kcal_mol, front.optimum_kcal_mol),
        Bounded::Feasible(_) => panic!("a ceiling below the achievable minimum answered"),
    }
}

/// The bar the Definition of Done states: exact against an enumeration that
/// folds the same molecules by a different route.
#[test]
fn the_flanked_front_is_exact_against_enumeration() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";

    let bag: Vec<Objective> = enumerate_all(&params, &usage, protein)
        .into_iter()
        .zip(coding_sequences(&usage, protein))
        .map(|(objective, cds)| Objective {
            mfe: refold_molecule(
                &params,
                &format!("{FLANK5}{cds}{FLANK3}"),
                FLANK5.len(),
                protein.len(),
            ),
            cai_log: objective.cai_log,
        })
        .collect();
    let expected = eliminate(bag);

    let front = feasible(&params, &usage, protein, FLANK5, FLANK3, Bounds::UNBOUNDED);
    common::assert_same("flanked front", &expected, &front_objectives(&front.front));

    for cut in [1, expected.len() / 2, expected.len() - 1] {
        let ceiling = Energy(expected[cut].mfe);
        let feasible_here = feasible(
            &params,
            &usage,
            protein,
            FLANK5,
            FLANK3,
            Bounds::with_ceiling(Ceiling::Absolute(ceiling)),
        );
        let want: Vec<Objective> = expected
            .iter()
            .copied()
            .filter(|o| o.mfe <= ceiling.0)
            .collect();
        common::assert_same(
            &format!("flanked sub-front at {}", ceiling.0),
            &want,
            &front_objectives(&feasible_here.front),
        );
    }
}

/// Flanks whose lengths are multiples of three keep every position of the
/// molecule codon-aligned.
///
/// **This case does not discriminate, and is kept as a floor rather than as a
/// check.** A boundary rule stated in absolute positions — one that assumes the
/// coding sequence starts at zero — returns this exact front too: its
/// disagreement with the molecule's real codon lattice falls only on flank
/// positions, which credit no adaptation, so it moves when a derivation is
/// scheduled and not what the fold answers. What such a rule does instead is
/// derive an index past the end of the molecule, which is why the cases that
/// discriminate are the ones whose total length is not a multiple of three.
#[test]
fn the_flanked_front_is_exact_on_codon_aligned_flanks() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";
    let (utr5, utr3) = ("GCAUUG", "AUCGAU");

    let bag: Vec<Objective> = enumerate_all(&params, &usage, protein)
        .into_iter()
        .zip(coding_sequences(&usage, protein))
        .map(|(objective, cds)| Objective {
            mfe: refold_molecule(
                &params,
                &format!("{utr5}{cds}{utr3}"),
                utr5.len(),
                protein.len(),
            ),
            cai_log: objective.cai_log,
        })
        .collect();

    let front = feasible(&params, &usage, protein, utr5, utr3, Bounds::UNBOUNDED);
    common::assert_same(
        "codon-aligned flanks",
        &eliminate(bag),
        &front_objectives(&front.front),
    );
}

/// A 5' flank whose length is not a multiple of three offsets the whole codon
/// lattice from the molecule's own coordinates, while a total length that *is* a
/// multiple of three keeps every derived index in range.
///
/// Built to catch a boundary rule reading the wrong lattice rather than merely
/// running off the end of it. **It found no such rule**: the absolute-position
/// rule this replaced returns this front unchanged, for the reason recorded
/// above. Kept because the property is worth holding, and recorded as
/// non-discriminating so it is not later mistaken for the check.
#[test]
fn the_flanked_front_is_exact_on_an_offset_coding_sequence() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";
    let (utr5, utr3) = ("GCAU", "AU");

    let bag: Vec<Objective> = enumerate_all(&params, &usage, protein)
        .into_iter()
        .zip(coding_sequences(&usage, protein))
        .map(|(objective, cds)| Objective {
            mfe: refold_molecule(
                &params,
                &format!("{utr5}{cds}{utr3}"),
                utr5.len(),
                protein.len(),
            ),
            cai_log: objective.cai_log,
        })
        .collect();

    let front = feasible(&params, &usage, protein, utr5, utr3, Bounds::UNBOUNDED);
    common::assert_same(
        "offset coding sequence",
        &eliminate(bag),
        &front_objectives(&front.front),
    );
}

/// The coding sequences `enumerate_all` folds, in its own order.
fn coding_sequences(usage: &CodonUsage, protein: &str) -> Vec<String> {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = dfa::build(&aa, usage);
    graph
        .paths_between(graph.start(), graph.end())
        .into_iter()
        .map(|(_, seq)| {
            seq.iter()
                .map(|&nt| match nt as u8 {
                    0 => 'A',
                    1 => 'C',
                    2 => 'G',
                    _ => 'U',
                })
                .collect()
        })
        .collect()
}

/// Every molecule the flanked lattice admits, folded — coding choices and flank
/// spellings together, with the adaptation the fold accumulates over both.
///
/// Enumerated off the same lattice builder the solver uses, so the two agree on
/// what the design space *is*; the energy of each molecule is then measured on a
/// path the solver does not run.
fn enumerate_molecules(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
) -> Vec<Objective> {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = spec_graph(usage, &aa, utr5, utr3, profile);
    let offsets = graph.utr_profile_offset(profile) + graph.codon_offset();
    graph
        .paths_between(graph.start(), graph.end())
        .into_iter()
        .map(|(adaptation, seq)| {
            let sequence: String = seq.iter().map(|&nt| nuc_char(nt)).collect();
            Objective {
                mfe: refold_molecule(params, &sequence, utr5.len(), protein.len()),
                cai_log: adaptation + offsets,
            }
        })
        .collect()
}

/// The lattice for a flank pair, built the way the solver builds it.
fn spec_graph(
    usage: &CodonUsage,
    aa: &[usize],
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
) -> dfa::Dfa {
    let (five, three) = (Chain::of(utr5), Chain::of(utr3));
    dfa::build_with_utr_profile(aa, usage, five.borrow(), three.borrow(), profile)
}

/// A [`UtrSpec`] resolved to the chain the lattice builder takes, owning what
/// that chain borrows.
enum Chain {
    Fixed(Vec<Nucleotide>),
    Designed(u32),
    Mixed(Vec<NucSet>),
}

impl Chain {
    fn of(spec: &UtrSpec) -> Self {
        match spec {
            UtrSpec::Fixed(s) => Self::Fixed(
                s.bytes()
                    .map(|b| Nucleotide::from_byte(if b == b'T' { b'U' } else { b }).unwrap())
                    .collect(),
            ),
            UtrSpec::Designed(n) => Self::Designed(*n as u32),
            UtrSpec::Mixed(s) => {
                Self::Mixed(s.bytes().map(|b| NucSet::from_iupac(b).unwrap()).collect())
            }
        }
    }

    fn borrow(&self) -> dfa::UtrChain<'_> {
        match self {
            Self::Fixed(v) => dfa::UtrChain::Fixed(v),
            Self::Designed(n) => dfa::UtrChain::Designed(*n),
            Self::Mixed(v) => dfa::UtrChain::Mixed(v),
        }
    }
}

/// A recovered front in the oracle's units, on the **total** adaptation the
/// fold's single accumulator carries — `cai_log + utr_profile_log`, which is the
/// axis the front's dominance is actually over.
fn total_adaptation(points: &[derna_lib::FrontPoint]) -> Vec<Objective> {
    let mut objs: Vec<Objective> = points
        .iter()
        .map(|p| Objective {
            mfe: (p.mfe_kcal_mol * 100.0).round() as i32,
            cai_log: p.cai_log + p.utr_profile_log,
        })
        .collect();
    objs.sort_by_key(|o| o.mfe);
    objs
}

#[test]
fn a_designed_flank_front_is_exact_against_enumeration() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";
    let (utr5, utr3) = (UtrSpec::Designed(2), fixed("AU"));
    let profile = UtrProfile::inert();

    let expected = eliminate(enumerate_molecules(
        &params, &usage, protein, &utr5, &utr3, profile,
    ));
    let front = spec_feasible(
        &params,
        &usage,
        protein,
        &utr5,
        &utr3,
        profile,
        Bounds::UNBOUNDED,
    );
    common::assert_same("designed flank", &expected, &total_adaptation(&front.front));
}

#[test]
fn a_priced_designed_flank_front_is_exact_against_enumeration() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";
    let (utr5, utr3) = (UtrSpec::Designed(2), fixed("AU"));
    let profile = au_rich();

    let expected = eliminate(enumerate_molecules(
        &params, &usage, protein, &utr5, &utr3, profile,
    ));
    let front = spec_feasible(
        &params,
        &usage,
        protein,
        &utr5,
        &utr3,
        profile,
        Bounds::UNBOUNDED,
    );
    common::assert_same("priced flank", &expected, &total_adaptation(&front.front));

    assert!(
        front.front.iter().any(|p| p.utr_profile_log != 0.0),
        "an active profile over designed positions prices nothing",
    );
}

/// The profile changes the answer, so the test above is reading the priced
/// lattice rather than passing on one where the price is inert in effect.
#[test]
fn pricing_a_designed_flank_moves_the_front() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";
    let (utr5, utr3) = (UtrSpec::Designed(2), fixed("AU"));

    let inert = spec_feasible(
        &params,
        &usage,
        protein,
        &utr5,
        &utr3,
        UtrProfile::inert(),
        Bounds::UNBOUNDED,
    );
    let priced = spec_feasible(
        &params,
        &usage,
        protein,
        &utr5,
        &utr3,
        au_rich(),
        Bounds::UNBOUNDED,
    );
    assert_ne!(
        total_adaptation(&inert.front),
        total_adaptation(&priced.front),
        "the profile left the front unmoved",
    );
}

#[test]
fn a_ceiling_over_a_priced_designed_flank_is_the_filtered_front() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";
    let (utr5, utr3) = (UtrSpec::Designed(2), fixed("AU"));
    let profile = au_rich();

    let whole = spec_feasible(
        &params,
        &usage,
        protein,
        &utr5,
        &utr3,
        profile,
        Bounds::UNBOUNDED,
    );
    let expected = total_adaptation(&whole.front);

    for cut in [1, expected.len() / 2, expected.len() - 1] {
        let ceiling = Energy(expected[cut].mfe);
        let under = spec_feasible(
            &params,
            &usage,
            protein,
            &utr5,
            &utr3,
            profile,
            Bounds::with_ceiling(Ceiling::Absolute(ceiling)),
        );
        let want: Vec<Objective> = expected
            .iter()
            .copied()
            .filter(|o| o.mfe <= ceiling.0)
            .collect();
        common::assert_same(
            &format!("priced sub-front at {}", ceiling.0),
            &want,
            &total_adaptation(&under.front),
        );
    }
}

/// `Mixed` reduces to `Fixed` and to `Designed` at its two ends, which the spec
/// states and which decides whether one code path serves all three.
#[test]
fn mixed_flanks_reduce_to_their_ends() {
    let (params, usage) = fixtures();
    let protein = "MYGKIIFVLM";

    let as_fixed = spec_feasible(
        &params,
        &usage,
        protein,
        &fixed("GCAU"),
        &fixed("AU"),
        UtrProfile::inert(),
        Bounds::UNBOUNDED,
    );
    let as_mixed = spec_feasible(
        &params,
        &usage,
        protein,
        &UtrSpec::Mixed("GCAU".into()),
        &UtrSpec::Mixed("AU".into()),
        UtrProfile::inert(),
        Bounds::UNBOUNDED,
    );
    assert_eq!(as_fixed.front, as_mixed.front, "all-nucleotide Mixed");

    let designed = spec_feasible(
        &params,
        &usage,
        protein,
        &UtrSpec::Designed(3),
        &fixed(""),
        UtrProfile::inert(),
        Bounds::UNBOUNDED,
    );
    let all_n = spec_feasible(
        &params,
        &usage,
        protein,
        &UtrSpec::Mixed("NNN".into()),
        &fixed(""),
        UtrProfile::inert(),
        Bounds::UNBOUNDED,
    );
    assert_eq!(designed.front, all_n.front, "all-N Mixed");
}

/// A profile weighting `A` and `U` above `G` and `C`.
///
/// **The direction is the whole of the choice.** A profile that agrees with what
/// folding already wants prices almost nothing on the recovered front, because
/// every surviving point spells the profile's argmax anyway and pays a weight of
/// zero: measured on this case, a `G`/`C`-weighted profile over two designed
/// positions leaves **0 of 2** front points priced, and over three, **1 of 3**.
/// Weighting the nucleotides that pair *worst* puts the two axes in opposition,
/// which is the configuration under which the flank term is observable at all —
/// **3 of 5** points priced, at both flank lengths. A test written over the
/// agreeing direction would hold whatever the flank term did.
fn au_rich() -> UtrProfile {
    UtrProfile::new([0.4, 0.1, 0.1, 0.4], 2.0).expect("a valid profile")
}

fn nuc_char(nt: Nucleotide) -> char {
    match nt as u8 {
        0 => 'A',
        1 => 'C',
        2 => 'G',
        _ => 'U',
    }
}

/// The walk options the flanked walk is checked under: refined to adjacency,
/// and with every lever that changes how gaps are cut — slices, no cut near the
/// optimum, chords between other gaps' passes, and probes cut by those chords.
fn walk_plans() -> [(usize, WalkOptions); 3] {
    let levers = WalkOptions {
        stop_slack: None,
        slice_slack: Some(5.0),
        slice_width: None,
        uncut_within: Some(30),
        chord_bracket: Some(1.0),
        cut_probes: true,
        ..WalkOptions::default()
    };
    [
        (1, WalkOptions::default()),
        (4, WalkOptions::default()),
        (4, levers),
    ]
}

fn walk_flanked(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    (threads, options): (usize, WalkOptions),
) -> derna_lib::WalkedFront {
    design_front_walk_with_utr(
        params,
        protein,
        usage,
        &fixed(FLANK5),
        &fixed(FLANK3),
        RegionConstraints::default(),
        1e-4,
        0.999,
        options,
        threads,
    )
    .expect("the flanked walk answers")
}

/// The walk over a flanked lattice recovers the flanked fold's whole front,
/// point for point, on any worker count and under every walk option, and each
/// point is a real molecule carrying both flanks whose re-fold is the energy it
/// reports. The flanks' total length is not a multiple of three, so the coding
/// sequence sits off the molecule's codon phase.
#[test]
fn the_flanked_walk_is_the_flanked_front() {
    let (params, usage) = fixtures();
    let mut walked_gaps = 0;
    for protein in common::CASES.iter().chain(common::WIDE_CASES.iter()) {
        let want = feasible(&params, &usage, protein, FLANK5, FLANK3, Bounds::UNBOUNDED);
        let want = front_objectives(&want.front);
        for plan in walk_plans() {
            let walked = walk_flanked(&params, &usage, protein, plan.clone());
            let label = format!("{protein} walked with flanks, {plan:?}");
            common::assert_same(&label, &want, &front_objectives(&walked.front));
            walked_gaps = walked_gaps.max(walked.gaps.len());
            for point in &walked.front {
                assert!(point.sequence.starts_with(FLANK5), "{label}");
                assert!(point.sequence.ends_with(FLANK3), "{label}");
                assert_eq!(
                    refold_molecule(&params, &point.sequence, FLANK5.len(), protein.len()),
                    (point.mfe_kcal_mol * 100.0).round() as i32,
                    "{label}: {}",
                    point.sequence,
                );
            }
        }
    }
    assert!(
        walked_gaps > 1,
        "no case walked more than one gap, so no hull sweep was exercised"
    );
}

/// The flanked walk against an oracle that shares none of its code: every
/// coding sequence enumerated, each whole molecule re-folded by the free folder,
/// then Pareto-reduced.
#[test]
fn the_flanked_walk_is_exact_against_enumeration() {
    let (params, usage) = fixtures();
    let protein = "MWWKFYCLMW";
    let bag: Vec<Objective> = enumerate_all(&params, &usage, protein)
        .into_iter()
        .zip(coding_sequences(&usage, protein))
        .map(|(objective, cds)| Objective {
            mfe: refold_molecule(
                &params,
                &format!("{FLANK5}{cds}{FLANK3}"),
                FLANK5.len(),
                protein.len(),
            ),
            cai_log: objective.cai_log,
        })
        .collect();
    let expected = eliminate(bag);
    for plan in walk_plans() {
        let walked = walk_flanked(&params, &usage, protein, plan.clone());
        common::assert_same(
            &format!("flanked walk, {plan:?}"),
            &expected,
            &front_objectives(&walked.front),
        );
    }
}
