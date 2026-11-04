//! The report half of the closed-form normalization.
//!
//! A restriction that excludes an alphabet's best choice is compiled by shifting
//! its edges so the best *available* choice weighs zero, and the report adds the
//! removed constant back. The span oracle in `dfa::closed_form_oracle` checks
//! the shift; it cannot see a report that forgets the add-back, because the
//! search runs entirely on shifted weights. This file closes that gap: on every
//! report path that runs over a shifted graph, the reported `cai_log` must equal
//! `Σ ln(g / g_max)` recomputed from the emitted coding sequence.
//!
//! Each case asserts its shift is strictly non-zero on the graph it solves, or
//! the add-back would be zero and the check vacuous.

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_composite, design_front_bounded_with_utr, design_restricted,
    design_restricted_with_placed_motifs, dfa, Beam, Bounded, Bounds, CodonUsage, CompositeSpec,
    Design, Layout, MotifWindow, NucSet, Params, PositionalConstraints, Prune, RegionConstraints,
    ScanConfig, UtrProfile, UtrSpec,
};

const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// Leucine and lysine twice each, so a restriction on them sits both at a span
/// end and wholly inside spans.
const PROTEIN: &str = "MLWKCLKY";

const TOLERANCE: f64 = 1e-9;

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

fn residues(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}

/// Every residue keeps every codon except leucine, held to `{UUA, UUG}`, and
/// lysine, held to `AAA` — both exclude the human argmax (`CUG`, `AAG`).
fn restriction(aa: &[usize]) -> Vec<Vec<[u8; 3]>> {
    let (leu, lys) = (aa_to_index('L').unwrap(), aa_to_index('K').unwrap());
    aa.iter()
        .map(|&a| match a {
            _ if a == leu => vec![[3, 3, 0], [3, 3, 2]],
            _ if a == lys => vec![[0, 0, 0]],
            _ => AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect(),
        })
        .collect()
}

/// `Σ ln(g / g_max)` over the coding sequence at `cds_start` of an emitted mRNA,
/// `EPSILON`-floored as the lattice's own edge weights are.
fn recomputed_cai_log(usage: &CodonUsage, aa: &[usize], sequence: &str, cds_start: usize) -> f64 {
    let index = |b: u8| match b {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'U' => 3,
        other => panic!("not a nucleotide: {other}"),
    };
    let bytes = sequence.as_bytes();
    aa.iter()
        .enumerate()
        .map(|(i, &a)| {
            let at = cds_start + 3 * i;
            let nuc = [index(bytes[at]), index(bytes[at + 1]), index(bytes[at + 2])];
            let codons = AMINO_ACIDS[a].codons;
            let ci = codons
                .iter()
                .position(|c| c.nuc == nuc)
                .expect("the emitted codon encodes its residue");
            let g = |c: usize| usage.fraction(a, c).max(EPSILON);
            let g_max = (0..codons.len()).map(g).fold(EPSILON, f64::max);
            (g(ci) / g_max).ln()
        })
        .sum()
}

fn assert_reports(label: &str, design: &Design, aa: &[usize], cds_start: usize) {
    let expected = recomputed_cai_log(&usage(), aa, &design.sequence, cds_start);
    assert!(
        (design.cai_log - expected).abs() < TOLERANCE,
        "{label}: reported cai_log {} against {expected} from the sequence",
        design.cai_log
    );
}

/// A restricted lattice through the plain restricted entry, CDS-only and
/// flanked: the scan report adds the codon offset back.
#[test]
fn a_restricted_scan_reports_the_adaptation_it_spelled() {
    let (params, usage) = (params(), usage());
    let aa = residues(PROTEIN);
    let allowed = restriction(&aa);
    for (utr5, utr3) in [("", ""), ("GGAUC", "CAUGC")] {
        let graph =
            dfa::build_restricted_with_utr(&aa, &usage, &allowed, &parse(utr5), &parse(utr3))
                .expect("non-empty");
        assert!(graph.codon_offset() < 0.0, "the restriction must shift");
        for lambda in LAMBDAS {
            let design = design_restricted(
                &params,
                PROTEIN,
                &usage,
                lambda,
                &allowed,
                utr5,
                utr3,
                exact(),
            )
            .expect("valid input")
            .expect("feasible");
            assert_reports(&format!("λ={lambda} {utr5:?}"), &design, &aa, utr5.len());
        }
    }
}

/// The same restriction through the constrained folder, which carries its own
/// traceback and its own report: a placed motif that the restricted optimum
/// spells forces it onto that path.
#[test]
fn a_restricted_constrained_fold_reports_the_adaptation_it_spelled() {
    let (params, usage) = (params(), usage());
    let aa = residues(PROTEIN);
    let allowed = restriction(&aa);
    let mut solved = 0;
    for lambda in LAMBDAS {
        let free = design_restricted(&params, PROTEIN, &usage, lambda, &allowed, "", "", exact())
            .expect("valid input")
            .expect("feasible");
        // The leucine codon the free optimum chose binds by construction, and
        // leaves the other one of `{UUA, UUG}` to spell both leucines with.
        let motif = free.sequence[3..6].to_string();
        let placed = [(motif.clone(), MotifWindow::all())];
        let Some(design) = design_restricted_with_placed_motifs(
            &params,
            PROTEIN,
            &usage,
            lambda,
            &allowed,
            &placed,
            "",
            "",
            RegionConstraints::default(),
            exact(),
        )
        .expect("valid input") else {
            continue;
        };
        assert!(
            !design.sequence.contains(&motif),
            "λ={lambda}: the cut held"
        );
        assert_reports(&format!("λ={lambda} placed {motif}"), &design, &aa, 0);
        solved += 1;
    }
    assert!(
        solved > 0,
        "every cut emptied the space; nothing was checked"
    );
}

/// Pins lowered by the composite entry — CDS pins to a codon subset excluding
/// the argmax, a flank pin to a mask excluding the profile's argmax — under an
/// active profile: both offsets are added back, and the flank term stays out of
/// `cai_log`.
#[test]
fn a_pinned_composite_reports_the_adaptation_it_spelled() {
    let (params, usage) = (params(), usage());
    let aa = residues(PROTEIN);
    let flank = 6;
    let layout = Layout::new(flank, aa.len() as u32, 0);
    let mut pc = PositionalConstraints::none(layout.len());
    // `MM` on the flank's first two positions excludes the profile's `G`; `UU`
    // on residue 2 (leucine) leaves `{UUA, UUG}`.
    pc.pins.pin(&layout, 0, "MM").expect("valid pin");
    pc.pins
        .pin(&layout, layout.cds_start() + 3, "UU")
        .expect("valid pin");
    let profile = UtrProfile::new([0.1, 0.1, 0.7, 0.1], 1.0).expect("valid profile");

    let mask: Vec<NucSet> = [b'M', b'M', b'N', b'N', b'N', b'N']
        .iter()
        .map(|&c| NucSet::from_iupac(c).expect("iupac"))
        .collect();
    let mut allowed: Vec<Vec<[u8; 3]>> = aa
        .iter()
        .map(|&a| AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect())
        .collect();
    allowed[1] = vec![[3, 3, 0], [3, 3, 2]];
    let graph = dfa::build_composite(
        &aa,
        &usage,
        Some(&allowed),
        dfa::UtrChain::Mixed(&mask),
        dfa::UtrChain::Fixed(&[]),
        profile,
    )
    .expect("non-empty");
    assert!(graph.codon_offset() < 0.0);
    assert!(graph.utr_profile_offset(profile) < 0.0);

    let (designed, none) = (
        UtrSpec::Designed(flank as usize),
        UtrSpec::Fixed(String::new()),
    );
    for lambda in LAMBDAS {
        let design = design_composite(
            &params,
            PROTEIN,
            &usage,
            lambda,
            &CompositeSpec::<String> {
                utr5: &designed,
                utr3: &none,
                constraints: RegionConstraints::default(),
                profile,
                motifs: &[],
                motif_window: MotifWindow::all(),
                flank_complement: None,
                complement_classes: None,
                positional: &pc,
            },
            exact(),
        )
        .expect("feasible");
        assert!(design.sequence[..2].bytes().all(|b| b == b'A' || b == b'C'));
        assert_reports(&format!("λ={lambda}"), &design, &aa, flank as usize);
    }
}

/// A front over a flank masked away from the profile's argmax: every point's
/// `cai_log` is its coding sequence's, with the flank offset added back.
#[test]
fn a_masked_flank_front_reports_the_adaptation_each_point_spelled() {
    let (params, usage) = (params(), usage());
    let protein = "MLWKC";
    let aa = residues(protein);
    let profile = UtrProfile::new([0.1, 0.1, 0.7, 0.1], 1.0).expect("valid profile");
    let utr5 = UtrSpec::Mixed("MMNN".into());
    let mask = [b'M', b'M', b'N', b'N'].map(|c| NucSet::from_iupac(c).expect("iupac"));
    let graph = dfa::build_with_utr_profile(
        &aa,
        &usage,
        dfa::UtrChain::Mixed(&mask),
        dfa::UtrChain::Fixed(&[]),
        profile,
    );
    assert!(graph.utr_profile_offset(profile) < 0.0);
    let front = match design_front_bounded_with_utr(
        &params,
        protein,
        &usage,
        &utr5,
        &UtrSpec::Fixed(String::new()),
        profile,
        Prune::None,
        Bounds::UNBOUNDED,
        1,
    )
    .expect("the flanked fold answers")
    {
        Bounded::Feasible(f) => f.front,
        Bounded::Unreachable { .. } => unreachable!("no ceiling was set"),
    };
    assert!(front.len() > 1, "a front with one point checks one report");
    for point in &front {
        let expected = recomputed_cai_log(&usage, &aa, &point.sequence, 4);
        assert!(
            (point.cai_log - expected).abs() < TOLERANCE,
            "{}: reported {} against {expected}",
            point.sequence,
            point.cai_log
        );
    }
}

fn parse(s: &str) -> Vec<derna_lib::Nucleotide> {
    s.bytes()
        .map(|b| derna_lib::Nucleotide::from_byte(b).expect("nucleotide"))
        .collect()
}
