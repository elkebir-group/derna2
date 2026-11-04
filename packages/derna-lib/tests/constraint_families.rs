//! The three standalone constraint cases, and the one that runs them together.
//!
//! The census sweeps a synthetic element to produce curves. These are the cases
//! a caller actually writes, run as independent gates:
//!
//! 1. **A manufacturing pin** — keep a restriction site in the coding sequence.
//! 2. **Accessibility** — keep the initiation window unpaired.
//! 3. **All families on one construct** — pins, a required element, an unpaired
//!    window, a forbidden pair, the region-class table and a motif set together.
//!    This is the integration case and the conflict-detection case; nothing else
//!    exercises the composition end to end.
//!
//! Every case asserts the thing no other test does: **the emitted coding
//! sequence still translates to the input protein.** A constraint surface that
//! reaches the codon lattice through arc removal cannot change the protein, and
//! that is exactly the kind of invariant that is assumed rather than checked.

use derna_codon_dp::codon_table::AMINO_ACIDS;
use derna_lib::{
    design_composite, design_with_config, AccessibilityWindow, Beam, BeamError, CodonUsage,
    CompositeSpec, ConstraintError, Design, Enforcement, Layout, MotifWindow, Params,
    PositionalSpec, RegionConstraints, ScanConfig, UtrProfile, UtrSpec,
};

/// A 78-residue human protein from the corpus — long enough for a CDS-internal
/// element, short enough to design in seconds.
const PROTEIN: &str =
    "MYGKIIFVLLLSGIVSISASSTTGVAMHTSTSSSVTKSYISSQTNGITLINWWAMARVIFEVMLVVVGMIILISYCIR";

/// The composed case runs on a prefix of it. The AC-threaded folder the motif
/// family routes through carries no interior reductions, so the integration
/// case is placed where that fold is affordable rather than where the sweep is.
const SHORT: &str = "MYGKIIFVLLLSGIVSISASSTTGVAMHTSTSSS";

const LAMBDA: f64 = 0.5;

fn params() -> Params {
    Params::load_default().expect("params")
}

fn usage() -> CodonUsage {
    CodonUsage::load_default().expect("usage")
}

fn exact() -> ScanConfig {
    ScanConfig {
        beam: Beam::exact(),
        ..ScanConfig::default()
    }
}

/// Design under one written positional statement, with whatever flank and
/// region table the case needs.
fn design_under(
    utr5: &UtrSpec,
    constraints: RegionConstraints,
    spec: &PositionalSpec,
    motifs: &[String],
) -> (Layout, Result<Design, BeamError>) {
    design_windowed(PROTEIN, utr5, constraints, spec, motifs, None)
}

/// [`design_under`] with the motif set scoped to a window.
///
/// A window is not only a semantic choice — it is what keeps the AC-threaded
/// folder affordable, since the product carries its extra dimension only over
/// the scoped positions.
fn design_windowed(
    protein: &str,
    utr5: &UtrSpec,
    constraints: RegionConstraints,
    spec: &PositionalSpec,
    motifs: &[String],
    window: Option<(u32, u32)>,
) -> (Layout, Result<Design, BeamError>) {
    let layout = Layout::new(utr5.len() as u32, protein.len() as u32, 0);
    let pc = match spec.resolve(&layout) {
        Ok(pc) => pc,
        Err(e) => panic!("the statement should resolve: {e}"),
    };
    let no3 = UtrSpec::Fixed(String::new());
    let got = design_composite(
        &params(),
        protein,
        &usage(),
        LAMBDA,
        &CompositeSpec {
            utr5,
            utr3: &no3,
            constraints,
            profile: UtrProfile::inert(),
            motifs,
            motif_window: window.map_or_else(MotifWindow::all, |(lo, hi)| MotifWindow::new(lo, hi)),
            flank_complement: None,
            complement_classes: None,
            positional: &pc,
        },
        exact(),
    );
    (layout, got)
}

/// The protein an emitted CDS spells, read straight off the codon table.
fn translate(sequence: &str, layout: &Layout) -> String {
    let cds = &sequence.as_bytes()[layout.cds_start() as usize..layout.cds_end() as usize];
    cds.chunks(3)
        .map(|c| {
            let nuc: [u8; 3] = std::array::from_fn(|k| match c[k] {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                _ => 3,
            });
            AMINO_ACIDS
                .iter()
                .find(|aa| aa.codons.iter().any(|cd| cd.nuc == nuc))
                .map(|aa| aa.code)
                .expect("every codon spells a residue")
        })
        .collect()
}

/// The invariant every case shares: a constraint narrows the design space and
/// never the protein.
fn assert_protein_preserved(design: &Design, layout: &Layout) {
    assert_protein(design, layout, PROTEIN);
}

/// [`assert_protein_preserved`] over any protein.
fn assert_protein(design: &Design, layout: &Layout, protein: &str) {
    assert_eq!(
        translate(&design.sequence, layout),
        protein,
        "the emitted CDS no longer spells the input protein"
    );
}

fn pairs_of(structure: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut pairs = Vec::new();
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => pairs.push((stack.pop().expect("balanced"), i)),
            _ => {}
        }
    }
    pairs
}

/// **Case 1 — a manufacturing pin.** Keep a SalI site in the coding sequence
/// without changing the protein.
///
/// The site straddles codon boundaries wherever it is placed, so this is the
/// case only a nucleotide-granular pin can express: a codon-granular one cannot
/// say "these six nucleotides, starting here".
///
/// **The placement is not free to choose.** Over this protein `GUCGAC` is
/// spellable at exactly two positions, and `GAAUUC`, `GGAUCC`, `AAGCUU` and
/// `CUGCAG` at none at all — whether a site fits anywhere is a property of the
/// protein, decided before any search. That is the same fact the census reports
/// as a feasibility rate, met here in the form a caller meets it.
#[test]
fn a_manufacturing_pin_keeps_the_site_and_the_protein() {
    const SITE: &str = "GUCGAC";
    let no5 = UtrSpec::Fixed(String::new());
    let mut spec = PositionalSpec::default();
    spec.add_pin(&format!("cds:60:{SITE}")).expect("valid pin");

    let (layout, got) = design_under(&no5, RegionConstraints::default(), &spec, &[]);
    let design = got.expect("EcoRI is spellable across these residues");
    assert_protein_preserved(&design, &layout);

    let at = layout.cds_start() as usize + 59;
    assert_eq!(
        &design.sequence[at..at + SITE.len()],
        SITE,
        "the site is not where it was pinned: {}",
        design.sequence
    );

    // The control: the unconstrained design does *not* already carry the site
    // there, so the pin removed something.
    let free = design_with_config(&params(), PROTEIN, &usage(), LAMBDA, exact())
        .expect("the unconstrained design exists");
    assert_ne!(
        &free.sequence[at..at + SITE.len()],
        SITE,
        "the site was already there, so this case measured the instance"
    );
    // …and it cost adaptation, which is the number a caller wants.
    assert!(
        design.cai_log < free.cai_log,
        "pinning six nucleotides was free, which would be surprising: {} vs {}",
        design.cai_log,
        free.cai_log
    );
}

/// **Case 2 — accessibility.** Keep the initiation window unpaired, written in
/// the coordinates a biologist uses.
///
/// The window straddles the UTR/CDS boundary, which is the class a per-region
/// constraint string cannot express and the reason positional coordinates are
/// whole-mRNA with region-qualified sugar.
#[test]
fn an_accessible_start_window_is_unpaired_in_the_design() {
    let utr5 = UtrSpec::Fixed("GCCUCCAUCGCAUGCAUCGA".into());
    let mut spec = PositionalSpec::default();
    spec.add_unpaired("5utr:-6..cds:12").expect("valid window");

    let (layout, got) = design_under(&utr5, RegionConstraints::default(), &spec, &[]);
    let design = got.expect("an unpaired window is always satisfiable");
    assert_protein_preserved(&design, &layout);

    let window = layout.cds_start() as usize - 6..layout.cds_start() as usize + 12;
    for (i, j) in pairs_of(&design.structure) {
        assert!(
            !window.contains(&i) && !window.contains(&j),
            "pair ({i}, {j}) has an endpoint in the initiation window: {}",
            design.structure
        );
    }

    // The control: without the window the design *does* pair inside it, so the
    // constraint removed something.
    let (_, free) = design_under(
        &utr5,
        RegionConstraints::default(),
        &PositionalSpec::default(),
        &[],
    );
    let free = free.expect("feasible");
    assert!(
        pairs_of(&free.structure)
            .iter()
            .any(|&(i, j)| window.contains(&i) || window.contains(&j)),
        "the unconstrained design already leaves the window open, so this case \
         measured the instance: {}",
        free.structure
    );
}

/// **Case 3 — every family on one construct.** Pins, a required element, an
/// unpaired window, a forbidden pair, the region-class table and a motif set,
/// all at once.
///
/// The integration case. Each family reaches the lattice somewhere different —
/// arc removal, a positional verdict at the `P`-write, a bound on an unpaired
/// range, a class lookup, an automaton on node identity — so this is the only
/// place they are shown to compose rather than merely coexist.
#[test]
fn every_constraint_family_composes_on_one_construct() {
    // A *designed* flank, so the Kozak pin has freedom to act on. A given flank
    // is not a design space, and a pin contradicting one is refused rather than
    // solved — which the conflict case below covers.
    let utr5 = UtrSpec::Designed(20);
    let motifs = ["GGGGG".to_string()];
    let mut spec = PositionalSpec::default();
    spec.add_pin("5utr:-6:GCCACC").expect("Kozak consensus");
    // A wobble pin. `cds:4` — the Kozak `+4` position — is residue 2's slot 1,
    // which this protein's tyrosine fixes to `U`; see the case below.
    spec.add_pin("cds:9:C").expect("a wobble pin");
    spec.add_unpaired("5utr:-4..cds:9").expect("start window");
    // A placement the protein admits — most do not, which is the census's
    // headline and is why this one is chosen rather than picked at round numbers.
    spec.add_helix("cds:31:cds:71:3")
        .expect("a CDS-internal element");
    spec.add_forbidden_pair("cds:20:cds:60").expect("one pair");

    // The motif set is scoped to the designed flank — the region it constrains,
    // and the scoping that keeps the AC-threaded fold affordable.
    let (layout, got) = design_windowed(
        SHORT,
        &utr5,
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        &spec,
        &motifs,
        Some((0, 20)),
    );
    let design = got.expect("the families are jointly satisfiable here");
    assert_protein(&design, &layout, SHORT);

    let cds = layout.cds_start() as usize;
    // Every family, checked on the emitted design rather than trusted.
    assert_eq!(&design.sequence[cds - 6..cds], "GCCACC", "the Kozak pin");
    assert_eq!(design.sequence.as_bytes()[cds + 8], b'C', "the wobble pin");
    // The motif set is forbidden *inside its window and nowhere else* — an
    // occurrence straddling either edge is permitted, and one past the window
    // entirely is none of the set's business.
    assert!(
        !design.sequence[..20].contains("GGGGG"),
        "the motif set inside its window: {}",
        design.sequence
    );
    let pairs = pairs_of(&design.structure);
    for (i, j) in &pairs {
        assert!(
            !(cds - 4..cds + 9).contains(i) && !(cds - 4..cds + 9).contains(j),
            "the unpaired window: ({i}, {j}) in {}",
            design.structure
        );
        assert!(
            !(*i < cds && *j >= cds),
            "5′ isolation: ({i}, {j}) joins the flank to the CDS"
        );
    }
    assert!(
        !pairs.contains(&(cds + 19, cds + 59)),
        "the forbidden pair formed anyway"
    );
    for t in 0..3usize {
        assert!(
            pairs.contains(&(cds + 30 + t, cds + 70 - t)),
            "the required helix is missing pair {t}: {}",
            design.structure
        );
    }
}

/// The conflict half of case 3: making one family contradict another is refused
/// **at parse time**, naming the flag that resolves it — not returned as an
/// infeasible solve the caller has to diagnose.
#[test]
fn a_family_contradicting_another_is_refused_by_name() {
    let utr5 = UtrSpec::Fixed("GCCUCCAUCGCAUGCAUCGA".into());
    let layout = Layout::new(utr5.len() as u32, PROTEIN.len() as u32, 0);

    // A required pair inside the window another family requires unpaired.
    let mut spec = PositionalSpec::default();
    spec.add_unpaired("5utr:-4..cds:9").expect("valid");
    spec.add_helix("cds:2:cds:60:2").expect("valid on its own");
    assert!(
        matches!(spec.resolve(&layout), Err(ConstraintError::Structure(_))),
        "a required pair inside an unpaired window must be refused"
    );

    // A required pair the region-class table forbids. This one survives parsing
    // — the table is not part of the written statement — and is refused when the
    // two meet at the design entry.
    let mut spec = PositionalSpec::default();
    spec.add_helix("5utr:2:cds:60:2").expect("valid on its own");
    let (_, got) = design_under(
        &utr5,
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        &spec,
        &[],
    );
    let err = got.expect_err("a required 5′↔CDS pair under 5′ isolation");
    assert!(
        matches!(
            err,
            BeamError::Constraint(ConstraintError::RegionTableForbidsPair { .. })
        ),
        "{err}"
    );
    assert!(err.to_string().contains("--region"), "{err}");
}

/// A pin the protein itself forbids is refused before any search, naming the
/// **residue** and its amino acid.
///
/// This is the slot determinism the scope rests on, met in the form a caller
/// meets it: the Kozak `+4` position is residue 2's slot 1, and slot 1 is fixed
/// by the amino acid in every family but leucine, arginine and serine. Here
/// residue 2 is a tyrosine, so `+4` cannot be a `G` under any synonymous choice
/// — a fact about the protein, available before the search rather than after it.
#[test]
fn a_pin_the_protein_forbids_names_the_residue_before_any_search() {
    let no5 = UtrSpec::Fixed(String::new());
    let mut spec = PositionalSpec::default();
    spec.add_pin("cds:4:G").expect("well-formed as a request");
    let (_, got) = design_under(&no5, RegionConstraints::default(), &spec, &[]);
    let err = got.expect_err("tyrosine fixes slot 1 to U");
    assert!(
        err.to_string().contains("residue 2")
            && err.to_string().contains("(Y)")
            && err.to_string().contains("cds:4"),
        "the message must name the residue, its amino acid and the position: {err}"
    );
}

/// A hard accessibility window and a requirement to pair inside it contradict,
/// and the message names the window rather than the pair — because narrowing
/// `--access` is what resolves it.
#[test]
fn the_accessibility_window_and_a_requirement_inside_it_are_refused() {
    let utr5 = UtrSpec::Fixed("GCCUCCAUCGCAUGCAUCGA".into());
    let mut spec = PositionalSpec::default();
    spec.add_helix("cds:2:cds:60:2").expect("valid on its own");
    let (_, got) = design_under(
        &utr5,
        RegionConstraints::default().with_window(AccessibilityWindow::hard(6, 12)),
        &spec,
        &[],
    );
    let err = got.expect_err("a required pair inside a hard window");
    assert!(
        matches!(
            err,
            BeamError::Constraint(ConstraintError::AccessWindowForbidsPair { .. })
        ),
        "{err}"
    );
    assert!(err.to_string().contains("--access"), "{err}");
}

/// The floor these cases sit on: an empty statement over the same construct is
/// byte-identical to the design without one.
#[test]
fn the_composed_construct_reduces_to_its_unconstrained_design() {
    let no5 = UtrSpec::Fixed(String::new());
    let (layout, got) = design_under(
        &no5,
        RegionConstraints::default(),
        &PositionalSpec::default(),
        &[],
    );
    let got = got.expect("feasible");
    let free = design_with_config(&params(), PROTEIN, &usage(), LAMBDA, exact()).expect("feasible");
    assert_eq!(
        (&got.sequence, &got.structure, got.mfe_kcal_mol, got.cai_log),
        (
            &free.sequence,
            &free.structure,
            free.mfe_kcal_mol,
            free.cai_log
        )
    );
    assert_protein_preserved(&got, &layout);
}
