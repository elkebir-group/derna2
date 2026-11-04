//! The positional constraint surface: floors, the shipped-feature cross-check,
//! the two normalization traps, and what an unsatisfiable statement reports.
//!
//! The brute-force correctness gate lives in `positional_oracle.rs`. This file
//! covers everything that is a claim about the *surface* rather than about the
//! optimum:
//!
//! 1. **Byte-identical floors.** An empty positional statement reproduces the
//!    unconstrained substrate in every UTR configuration, and a constraint every
//!    design already satisfies changes nothing.
//! 2. **`--access` re-expressed.** The shipped accessibility window is
//!    "keep this window unpaired", and saying that in the new grammar must give
//!    byte-for-byte the same design. A shipped feature re-expressed in a new
//!    surface is the strongest available test of the new one.
//! 3. **Both normalization offsets are exercised, and asserted directly.** A
//!    restriction that excludes a table's argmax leaves the best available
//!    weight below zero; if every test case happened to keep the argmax, both
//!    offsets would be zero, the whole path would be a no-op, and the suite
//!    would pass while testing nothing. Each is asserted `< 0` on the graph
//!    itself, with a control where it is exactly zero.
//! 4. **Typed infeasibility and parse-time conflicts.**

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::{
    design_composite, design_must_paired, design_with_config, design_with_utr_spec, dfa,
    AccessibilityWindow, Beam, BeamError, CodonUsage, CompositeSpec, ConstraintError, Design,
    Enforcement, Infeasible, Layout, MotifWindow, NucSet, Params, PinError, PositionalConstraints,
    PositionalSpec, RegionConstraints, ScanConfig, StructureConstraints, StructureError,
    UtrProfile, UtrSpec,
};
use rna_mfe::Nucleotide;

const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

const PROTEIN: &str = "MLWKCWMKY";

fn exact() -> ScanConfig {
    ScanConfig {
        beam: Beam::exact(),
        ..ScanConfig::default()
    }
}

fn params() -> Params {
    Params::load_default().expect("params")
}

fn usage() -> CodonUsage {
    CodonUsage::load_default().expect("usage")
}

fn protein_indices(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}

/// Design through the composite entry under one positional statement.
fn design(
    protein: &str,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    constraints: RegionConstraints,
    profile: UtrProfile,
    pc: &PositionalConstraints,
    lambda: f64,
) -> Result<Design, BeamError> {
    design_composite(
        &params(),
        protein,
        &usage(),
        lambda,
        &CompositeSpec::<String> {
            utr5,
            utr3,
            constraints,
            profile,
            motifs: &[],
            motif_window: MotifWindow::all(),
            flank_complement: None,
            complement_classes: None,
            positional: pc,
        },
        exact(),
    )
}

/// The four fields a floor compares. Comparing the whole design rather than its
/// objective is what makes "byte-identical" mean it.
fn fields(d: &Design) -> (String, String, f64, f64) {
    (
        d.sequence.clone(),
        d.structure.clone(),
        d.mfe_kcal_mol,
        d.cai_log,
    )
}

/// An empty positional statement leaves every UTR configuration exactly where
/// it was — the additivity floor, one per configuration.
#[test]
fn an_empty_statement_is_byte_identical_in_every_utr_configuration() {
    let flank = "GCAUGCAU";
    let configs: [(UtrSpec, UtrSpec); 4] = [
        (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new())),
        (UtrSpec::Fixed(flank.into()), UtrSpec::Fixed(flank.into())),
        (UtrSpec::Designed(8), UtrSpec::Designed(8)),
        (UtrSpec::Fixed(flank.into()), UtrSpec::Designed(8)),
    ];
    for (utr5, utr3) in configs {
        let len = (utr5.len() + PROTEIN.len() * 3 + utr3.len()) as u32;
        let empty = PositionalConstraints::none(len);
        assert!(empty.is_inert());
        for lambda in LAMBDAS {
            let want = design_with_utr_spec(
                &params(),
                PROTEIN,
                &usage(),
                lambda,
                &utr5,
                &utr3,
                RegionConstraints::default(),
                exact(),
            )
            .expect("the unconstrained design exists");
            let got = design(
                PROTEIN,
                &utr5,
                &utr3,
                RegionConstraints::default(),
                UtrProfile::inert(),
                &empty,
                lambda,
            )
            .expect("an empty statement constrains nothing");
            assert_eq!(fields(&want), fields(&got), "λ={lambda}");
        }
    }
}

/// A constraint every design already satisfies changes nothing — the spec's own
/// floor, and the one that catches a constraint which narrows more than it says.
///
/// Built from the unconstrained optimum itself: require exactly the positions it
/// already pairs, and forbid nothing.
#[test]
fn a_constraint_the_optimum_already_satisfies_changes_nothing() {
    let (no5, no3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    let layout = Layout::new(0, PROTEIN.len() as u32, 0);
    let mut satisfied_something = false;
    for lambda in LAMBDAS {
        let free = design_with_config(&params(), PROTEIN, &usage(), lambda, exact())
            .expect("the unconstrained design exists");
        let paired: Vec<u32> = free
            .structure
            .bytes()
            .enumerate()
            .filter(|(_, c)| *c != b'.')
            .map(|(i, _)| i as u32)
            .collect();
        if paired.is_empty() {
            continue;
        }
        satisfied_something = true;
        // Require exactly what the optimum already does, so the feasible set
        // still contains it and the answer may not move.
        let mut pc = PositionalConstraints::none(layout.len());
        let marks: String = (0..layout.len())
            .map(|p| if paired.contains(&p) { '|' } else { '.' })
            .collect();
        pc.structure = StructureConstraints::parse(&layout, &marks).expect("valid");
        let got = design(
            PROTEIN,
            &no5,
            &no3,
            RegionConstraints::default(),
            UtrProfile::inert(),
            &pc,
            lambda,
        )
        .expect("the optimum itself is feasible");
        assert_eq!(fields(&free), fields(&got), "λ={lambda}");
    }
    assert!(
        satisfied_something,
        "no λ produced a paired optimum, so this floor tested nothing"
    );
}

/// The shipped accessibility window says "no pair has an endpoint in this
/// window". Saying the same thing with `x` over the same positions must produce
/// the same design, byte for byte.
///
/// This is the strongest available test of the mask: a feature that already
/// works, re-expressed through the new grammar.
#[test]
fn the_accessibility_window_re_expressed_as_x_is_byte_identical() {
    let utr5 = UtrSpec::Fixed("GCAUGCAUGCAU".into());
    let utr3 = UtrSpec::Fixed(String::new());
    let (upstream, downstream) = (6u32, 9u32);
    let layout = Layout::new(utr5.len() as u32, PROTEIN.len() as u32, 0);
    let window = layout.cds_start() - upstream..layout.cds_start() + downstream;

    let mut pc = PositionalConstraints::none(layout.len());
    pc.structure
        .require_unpaired(&layout, window.start, window.end - 1)
        .expect("valid window");

    let mut differed = false;
    for lambda in LAMBDAS {
        let via_access = design(
            PROTEIN,
            &utr5,
            &utr3,
            RegionConstraints::default()
                .with_window(AccessibilityWindow::hard(upstream, downstream)),
            UtrProfile::inert(),
            PositionalConstraints::empty(),
            lambda,
        )
        .expect("feasible");
        let via_mask = design(
            PROTEIN,
            &utr5,
            &utr3,
            RegionConstraints::default(),
            UtrProfile::inert(),
            &pc,
            lambda,
        )
        .expect("feasible");
        assert_eq!(fields(&via_access), fields(&via_mask), "λ={lambda}");

        // The control: the window has to have removed something, or the two
        // agreeing says nothing about either.
        let free = design(
            PROTEIN,
            &utr5,
            &utr3,
            RegionConstraints::default(),
            UtrProfile::inert(),
            PositionalConstraints::empty(),
            lambda,
        )
        .expect("feasible");
        differed |= fields(&free) != fields(&via_access);
    }
    assert!(
        differed,
        "the window forbade nothing at any λ, so the cross-check compared two \
         unconstrained designs"
    );
}

/// Must-pair reaches the surface through `design_composite` and agrees with the
/// entry point that shipped it — so the composite path is the same mechanism,
/// not a second one.
#[test]
fn must_pair_through_the_composite_entry_matches_the_direct_entry() {
    let layout = Layout::new(0, PROTEIN.len() as u32, 0);
    let (no5, no3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    for required in [vec![10u32], vec![8, 9, 10], vec![4, 20]] {
        let marks: String = (0..layout.len())
            .map(|p| if required.contains(&p) { '|' } else { '.' })
            .collect();
        let mut pc = PositionalConstraints::none(layout.len());
        pc.structure = StructureConstraints::parse(&layout, &marks).expect("valid");
        for lambda in LAMBDAS {
            let direct =
                design_must_paired(&params(), PROTEIN, &usage(), lambda, &required, exact())
                    .expect("no beam error");
            let composite = design(
                PROTEIN,
                &no5,
                &no3,
                RegionConstraints::default(),
                UtrProfile::inert(),
                &pc,
                lambda,
            );
            match (direct, composite) {
                (Some(a), Ok(b)) => assert_eq!(fields(&a), fields(&b), "{required:?} λ={lambda}"),
                (None, Err(BeamError::NoFeasibleDesign(_))) => {}
                (a, b) => panic!(
                    "{required:?} λ={lambda}: direct {:?} vs composite {:?}",
                    a.is_some(),
                    b.is_ok()
                ),
            }
        }
    }
}

/// The codon-table normalization, asserted on the graph rather than inferred
/// from a design.
///
/// A restricted codon unit's edges are shifted so the best **available** codon
/// sits at weight zero, because the `O(1)` broken-codon closed form credits a
/// whole interior codon with nothing. If every case pinned a residue to a set
/// containing its family argmax the shift would be zero everywhere and the path
/// would be untested — so the excluding case asserts `< 0` directly, against a
/// control that is exactly `0`.
#[test]
fn a_codon_restriction_that_excludes_the_argmax_carries_a_negative_offset() {
    let usage = usage();
    let aa = protein_indices("MLWKC");
    let all_of = |i: usize| -> Vec<[u8; 3]> {
        derna_codon_dp::codon_table::AMINO_ACIDS[i]
            .codons
            .iter()
            .map(|c| c.nuc)
            .collect()
    };
    let build = |leu: Vec<[u8; 3]>| {
        let sets: Vec<Vec<[u8; 3]>> = vec![
            all_of(aa[0]),
            leu,
            all_of(aa[2]),
            all_of(aa[3]),
            all_of(aa[4]),
        ];
        dfa::build_composite(
            &aa,
            &usage,
            Some(&sets),
            dfa::UtrChain::Fixed(&[]),
            dfa::UtrChain::Fixed(&[]),
            UtrProfile::inert(),
        )
        .expect("every residue keeps a codon")
    };
    // `CUG` is human leucine's argmax; `{UUA, UUG}` excludes it.
    let excluding = build(vec![[3, 3, 0], [3, 3, 2]]);
    assert!(
        excluding.codon_offset() < 0.0,
        "a restriction excluding the argmax must move adaptation out of the edges, got {}",
        excluding.codon_offset()
    );
    // The control: a restriction *containing* the argmax leaves nothing to move.
    let containing = build(vec![[1, 3, 2], [3, 3, 2]]);
    assert_eq!(containing.codon_offset(), 0.0);
    // And the unrestricted lattice, which every shipped path runs on.
    let free = dfa::build_composite(
        &aa,
        &usage,
        None,
        dfa::UtrChain::Fixed(&[]),
        dfa::UtrChain::Fixed(&[]),
        UtrProfile::inert(),
    )
    .expect("unrestricted");
    assert_eq!(free.codon_offset(), 0.0);

    // The whole point of the shift: with it, a whole restricted codon really is
    // free to cross, which is what the closed form assumes.
    assert_eq!(
        excluding.broken_codon_cai(dfa::Node::new(3, 0), dfa::Node::new(6, 0)),
        Some(0.0)
    );
}

/// The designed-UTR normalization (D4a), reached through a **pin** rather than
/// through the alphabet spelling — and asserted the same way, `< 0` directly
/// with a control at exactly zero.
#[test]
fn a_pin_on_a_designed_flank_position_carries_a_negative_profile_offset() {
    let usage = usage();
    let aa = protein_indices("MLWKC");
    // A `G`-favouring profile, pinned away from `G`.
    let g_favouring = UtrProfile::new([0.1, 0.1, 0.7, 0.1], 1.0).expect("valid profile");
    let a_favouring = UtrProfile::new([0.7, 0.1, 0.1, 0.1], 1.0).expect("valid profile");
    let mask: Vec<NucSet> = vec![
        NucSet::from_iupac(b'M').expect("M is {A, C}"),
        NucSet::ALL,
        NucSet::ALL,
        NucSet::ALL,
    ];
    let graph = |profile| {
        dfa::build_composite(
            &aa,
            &usage,
            None,
            dfa::UtrChain::Mixed(&mask),
            dfa::UtrChain::Fixed(&[]),
            profile,
        )
        .expect("unrestricted codons")
    };
    assert!(
        graph(g_favouring).utr_profile_offset(g_favouring) < 0.0,
        "a pin excluding the profile's argmax must move weight out of the edges"
    );
    // The control: the same pin under a profile whose argmax the pin admits.
    assert_eq!(graph(a_favouring).utr_profile_offset(a_favouring), 0.0);
}

/// A pin on a designed flank position **is** that position's alphabet subset, so
/// the two spellings must produce the same design. The floor that says the pin
/// grammar generalizes the shipped flank restriction rather than shadowing it.
#[test]
fn a_pin_on_a_designed_flank_equals_writing_the_subset() {
    let layout = Layout::new(6, PROTEIN.len() as u32, 0);
    let mut pc = PositionalConstraints::none(layout.len());
    pc.pins.pin(&layout, 0, "MMR").expect("valid pin");
    let designed = UtrSpec::Designed(6);
    let spelled = UtrSpec::Mixed("MMRNNN".into());
    let none = UtrSpec::Fixed(String::new());
    let profile = UtrProfile::new([0.1, 0.1, 0.7, 0.1], 1.0).expect("valid profile");
    for lambda in LAMBDAS {
        let via_pin = design(
            PROTEIN,
            &designed,
            &none,
            RegionConstraints::default(),
            profile,
            &pc,
            lambda,
        )
        .expect("feasible");
        let via_subset = design(
            PROTEIN,
            &spelled,
            &none,
            RegionConstraints::default(),
            profile,
            PositionalConstraints::empty(),
            lambda,
        )
        .expect("feasible");
        assert_eq!(fields(&via_pin), fields(&via_subset), "λ={lambda}");
    }
}

/// Construction tier: a pin no codon of the covered residue can spell is
/// reported before any search, naming the residue rather than the pin — because
/// either pin alone may be perfectly spellable.
#[test]
fn an_unspellable_pin_is_reported_at_construction() {
    let layout = Layout::new(0, PROTEIN.len() as u32, 0);
    let mut pc = PositionalConstraints::none(layout.len());
    pc.pins.pin(&layout, 0, "C").expect("recording succeeds");
    let (no5, no3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    let err = design(
        PROTEIN,
        &no5,
        &no3,
        RegionConstraints::default(),
        UtrProfile::inert(),
        &pc,
        0.5,
    )
    .expect_err("methionine has one codon and it starts A");
    assert!(
        matches!(
            err,
            BeamError::NoFeasibleDesign(Infeasible::Pins(PinError::NoCodon { residue: 1, .. }))
        ),
        "{err}"
    );
}

/// Pre-check tier: a required pair between two positions the protein fixes to
/// non-complementary letters is refused before the search, with the message
/// naming both positions' spellable sets.
#[test]
fn an_unpairable_required_pair_is_reported_by_the_precheck() {
    // Valine and isoleucine both fix `U` at slot 2, and `U · U` is not a pair.
    let protein = "VAAAAIAA";
    let layout = Layout::new(0, protein.len() as u32, 0);
    let mut pc = PositionalConstraints::none(layout.len());
    pc.structure
        .require_pair(&layout, 1, 16)
        .expect("a well-formed pair");
    let (no5, no3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    let err = design(
        protein,
        &no5,
        &no3,
        RegionConstraints::default(),
        UtrProfile::inert(),
        &pc,
        0.5,
    )
    .expect_err("no synonymous choice repairs a U·U pair");
    assert!(
        matches!(err, BeamError::NoFeasibleDesign(Infeasible::PreCheck(_))),
        "{err}"
    );
    assert!(
        err.to_string().contains("infeasible before search"),
        "{err}"
    );
}

/// Scan tier, and the distinction the typed infeasibility exists for: on the exact lattice an empty
/// scan is a **proof**, under a beam it is only a statement about one search.
/// The two must not report the same sentence.
#[test]
fn the_scan_tier_says_whether_it_proved_anything() {
    // A pair whose endpoints have no legal partner at all: the no-sharp-turn
    // rule leaves position 1 of a 12-nt molecule unable to reach position 3.
    let protein = "MWCW";
    let layout = Layout::new(0, protein.len() as u32, 0);
    let mut pc = PositionalConstraints::none(layout.len());
    // Every position paired is unsatisfiable at this length — a hairpin needs
    // three unpaired nucleotides.
    pc.structure = StructureConstraints::parse(&layout, &"|".repeat(12)).expect("valid");
    let (no5, no3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    let run = |beam| {
        design_composite(
            &params(),
            protein,
            &usage(),
            0.5,
            &CompositeSpec::<String> {
                utr5: &no5,
                utr3: &no3,
                constraints: RegionConstraints::default(),
                profile: UtrProfile::inert(),
                motifs: &[],
                motif_window: MotifWindow::all(),
                flank_complement: None,
                complement_classes: None,
                positional: &pc,
            },
            ScanConfig {
                beam,
                ..ScanConfig::default()
            },
        )
    };
    let exact_err = run(Beam::exact()).expect_err("no structure pairs every position");
    assert!(
        matches!(
            exact_err,
            BeamError::NoFeasibleDesign(Infeasible::ExactScan)
        ),
        "{exact_err}"
    );
    assert!(exact_err.to_string().contains("proved"), "{exact_err}");

    let beam_err = run(Beam::flat(1)).expect_err("nor under a beam");
    assert!(
        matches!(beam_err, BeamError::NoFeasibleDesign(Infeasible::Beam)),
        "{beam_err}"
    );
    assert!(
        beam_err.to_string().contains("--mode exact"),
        "the beam's weaker claim must say how to get a decisive answer: {beam_err}"
    );
}

/// A required pair the region-class table forbids is refused at parse time,
/// naming the flag that resolves it — not silently resolved in favour of either.
#[test]
fn a_required_pair_the_region_table_forbids_is_refused() {
    let utr5 = UtrSpec::Fixed("GCAUGCAU".into());
    let no3 = UtrSpec::Fixed(String::new());
    let layout = Layout::new(8, PROTEIN.len() as u32, 0);
    let mut pc = PositionalConstraints::none(layout.len());
    pc.structure
        .require_pair(&layout, 2, 14)
        .expect("a well-formed 5′UTR↔CDS pair");
    let err = design(
        PROTEIN,
        &utr5,
        &no3,
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        UtrProfile::inert(),
        &pc,
        0.5,
    )
    .expect_err("both statements are hard");
    assert!(
        matches!(
            err,
            BeamError::Constraint(ConstraintError::RegionTableForbidsPair { .. })
        ),
        "{err}"
    );
    assert!(err.to_string().contains("--region"), "{err}");
}

/// The written form: region-qualified coordinates and the sugar, resolved once
/// against a layout. The Kozak spelling is the case the region qualification
/// exists for.
#[test]
fn the_written_spec_resolves_its_sugar_against_the_layout() {
    let layout = Layout::new(12, 20, 9);
    let mut spec = PositionalSpec::default();
    spec.add_pin("5utr:-6:GCCACC").expect("Kozak consensus");
    spec.add_pin("cds:4:G").expect("Kozak +4");
    spec.add_unpaired("5utr:-4..cds:12").expect("start window");
    spec.add_helix("cds:20:cds:50:4")
        .expect("a CDS-internal element");
    spec.add_forbidden_pair("3utr:1:3utr:9").expect("one pair");
    let pc = spec.resolve(&layout).expect("no contradiction");

    // The consensus lands on the six nucleotides before the start codon.
    for (offset, code) in b"GCCACC".iter().enumerate() {
        assert_eq!(
            pc.pins.at(6 + offset as u32),
            NucSet::from_iupac(*code).expect("IUPAC"),
        );
    }
    assert_eq!(pc.pins.at(15), NucSet::of(Nucleotide::G), "cds:4 is +4");
    // The start window spans the UTR/CDS boundary, which is why a per-region
    // string could not express it.
    for p in 8..=23 {
        assert_eq!(pc.structure.mark(p), derna_lib::Mark::Unpaired, "{p}");
    }
    assert_eq!(
        pc.structure.required_pairs(),
        vec![(31, 61), (32, 60), (33, 59), (34, 58)]
    );
}

/// Malformed and contradictory written statements are refused where they are
/// written, naming what has to change.
#[test]
fn contradictions_are_refused_before_any_design_work() {
    let layout = Layout::new(12, 20, 9);
    let refuse = |build: &dyn Fn(&mut PositionalSpec) -> Result<(), ConstraintError>| {
        let mut spec = PositionalSpec::default();
        build(&mut spec)
            .and_then(|()| spec.resolve(&layout).map(|_| ()))
            .expect_err("must be refused")
    };
    // Two pins with no common nucleotide.
    assert!(matches!(
        refuse(&|s| {
            s.add_pin("cds:4:M")?;
            s.add_pin("cds:4:G")
        }),
        ConstraintError::Pin(PinError::Conflict { .. })
    ));
    // A required pair inside a window required unpaired.
    assert!(matches!(
        refuse(&|s| {
            s.add_unpaired("cds:1..cds:9")?;
            s.add_helix("cds:2:cds:30:2")
        }),
        ConstraintError::Structure(StructureError::Contradiction { .. })
    ));
    // A pseudoknot: the grammar is nested-only, so it is refused rather than
    // silently dropped.
    assert!(matches!(
        refuse(&|s| {
            s.add_helix("cds:1:cds:20:2")?;
            s.add_helix("cds:10:cds:40:2")
        }),
        ConstraintError::Structure(StructureError::Crossing { .. })
    ));
    // A coordinate off the molecule.
    assert!(matches!(
        refuse(&|s| s.add_pin("3utr:20:A")),
        ConstraintError::Coord(_)
    ));
    // A range written backwards names no position.
    assert!(matches!(
        refuse(&|s| s.add_unpaired("cds:9..cds:1")),
        ConstraintError::EmptyRange { .. }
    ));
    // Flag grammar.
    assert!(matches!(
        refuse(&|s| s.add_helix("cds:1:cds:20")),
        ConstraintError::Syntax {
            what: "--helix",
            ..
        }
    ));
}
