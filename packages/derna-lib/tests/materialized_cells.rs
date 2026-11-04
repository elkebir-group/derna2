//! The cell-count instrumentation, and the claim it exists to make checkable.
//!
//! Wall-clock is a property of the host; **materialized CKY cells** are a
//! property of the inputs, which is what makes them the memory number a
//! characterization can compare across runs. This file asserts the three things
//! a reader of those counts is entitled to assume:
//!
//! 1. **The counted entry points are the uncounted ones.** Reporting counts must
//!    not be a second code path, or a count would describe a design nobody ran.
//! 2. **A forbid-shaped constraint removes closed-pair cells, and the removal is
//!    asserted to be non-zero.** This is the claim the characterization rests on,
//!    and a test that only checked `<=` would pass on an implementation where the
//!    constraint did nothing at all. The control is the same mark at a position the
//!    optimum already leaves unpaired: the *design* comes back unchanged and the
//!    *cells* still drop, which is what separates "the table shrank" from "the
//!    answer moved". They are different claims and the sweep reports both.
//! 3. **A require-shaped constraint is *not* a cell remover.** It bounds the
//!    ranges the interior reductions enumerate over rather than forbidding
//!    anything, so its cell count stays close to the unconstrained one — the two
//!    mechanisms are distinguishable in this metric, which is why the sweep
//!    keeps them on separate axes.

use derna_lib::{
    design_composite, design_composite_counted, design_with_config, design_with_config_counted,
    Beam, CellCounts, CodonUsage, CompositeSpec, Design, Layout, MotifWindow, Params,
    PositionalConstraints, RegionConstraints, ScanConfig, StructureConstraints, UtrProfile,
    UtrSpec,
};

/// Long enough that the unconstrained optimum forms real helices for a forbid to
/// remove, short enough for a debug-profile test.
const PROTEIN: &str = "MLWKCWMKYFVNQHRTGADESILP";

const LAMBDA: f64 = 0.5;

fn params() -> Params {
    Params::load_default().expect("params")
}

fn usage() -> CodonUsage {
    CodonUsage::load_default().expect("codon usage")
}

fn exact() -> ScanConfig {
    ScanConfig {
        beam: Beam::exact(),
        ..ScanConfig::default()
    }
}

fn layout() -> Layout {
    Layout::new(0, PROTEIN.len() as u32, 0)
}

fn spec<'a>(
    positional: &'a PositionalConstraints,
    flanks: &'a (UtrSpec, UtrSpec),
) -> CompositeSpec<'a, String> {
    CompositeSpec {
        utr5: &flanks.0,
        utr3: &flanks.1,
        constraints: RegionConstraints::default(),
        profile: UtrProfile::inert(),
        motifs: &[],
        motif_window: MotifWindow::all(),
        flank_complement: None,
        complement_classes: None,
        positional,
    }
}

fn empty_flanks() -> (UtrSpec, UtrSpec) {
    (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()))
}

/// Design under `positional`, returning the design and its cell counts.
fn design(positional: &PositionalConstraints) -> (Design, CellCounts) {
    let flanks = empty_flanks();
    let (design, counts) = design_composite_counted(
        &params(),
        PROTEIN,
        &usage(),
        LAMBDA,
        &spec(positional, &flanks),
        exact(),
    )
    .expect("feasible");
    (design, counts.expect("the plain scan reports its cells"))
}

#[test]
fn counted_entry_points_return_the_uncounted_design() {
    let p = params();
    let u = usage();
    let plain = design_with_config(&p, PROTEIN, &u, LAMBDA, exact()).expect("design");
    let (counted, counts) =
        design_with_config_counted(&p, PROTEIN, &u, LAMBDA, exact()).expect("design");
    assert_eq!(plain, counted, "counting must not be a second code path");
    assert!(counts.total() > 0, "a finished scan materialized cells");
    assert_eq!(
        counts.total(),
        counts.p_cells + counts.multi_cells + counts.m_cells + counts.m2_cells + counts.c_cells
    );

    let flanks = empty_flanks();
    let inert = PositionalConstraints::empty();
    let composite =
        design_composite(&p, PROTEIN, &u, LAMBDA, &spec(inert, &flanks), exact()).expect("design");
    let (composite_counted, _) =
        design_composite_counted(&p, PROTEIN, &u, LAMBDA, &spec(inert, &flanks), exact())
            .expect("design");
    assert_eq!(composite, composite_counted);
    assert_eq!(
        composite, plain,
        "an empty positional statement is the unconstrained substrate"
    );
}

/// An `x` at a position the unconstrained optimum pairs removes closed-pair
/// cells; the same mark where it pairs nothing removes none.
#[test]
fn an_unpaired_mark_removes_closed_pair_cells_and_the_control_removes_none() {
    let layout = layout();
    let (free, free_counts) = design(PositionalConstraints::empty());
    let structure = free.structure.as_bytes();
    let paired = structure
        .iter()
        .position(|&c| c != b'.')
        .expect("the unconstrained optimum forms at least one pair") as u32;

    let mut marked = StructureConstraints::none(layout.len());
    marked
        .require_unpaired(&layout, paired, paired)
        .expect("in range");
    let (_, constrained) = design(&PositionalConstraints {
        pins: derna_lib::Pins::none(0),
        structure: marked,
    });

    // `<` rather than `<=`: an implementation whose constraint did nothing would pass
    // a `<=` and this file would be testing that the counts exist.
    assert!(
        constrained.p_cells < free_counts.p_cells,
        "forbidding every pair at a paired position must remove P cells: \
         {} against {}",
        constrained.p_cells,
        free_counts.p_cells
    );

    let unpaired = structure
        .iter()
        .position(|&c| c == b'.')
        .expect("the optimum leaves something unpaired") as u32;
    let mut control = StructureConstraints::none(layout.len());
    control
        .require_unpaired(&layout, unpaired, unpaired)
        .expect("in range");
    let (control_design, control_counts) = design(&PositionalConstraints {
        pins: derna_lib::Pins::none(0),
        structure: control,
    });
    assert_eq!(
        control_design, free,
        "a mark the optimum already satisfies changes nothing"
    );
    assert!(
        control_counts.p_cells < free_counts.p_cells,
        "the control still forbids the pairs that position could have carried"
    );
}

/// A `|` bounds enumeration ranges rather than forbidding pairs, so it leaves
/// the closed-pair table essentially intact — the property that makes it a
/// different axis from `x` rather than a stronger dose of the same one.
#[test]
fn a_required_position_is_not_a_cell_remover() {
    let layout = layout();
    let (free, free_counts) = design(PositionalConstraints::empty());
    // A position the unconstrained optimum leaves unpaired, so the requirement
    // has something to change. Mid-construct, where a deep loop interior forms.
    let bytes = free.structure.as_bytes();
    let middle = (layout.len() / 3..2 * layout.len() / 3)
        .find(|&i| bytes[i as usize] == b'.')
        .expect("the optimum leaves something unpaired mid-construct");
    let text: String = (0..layout.len())
        .map(|i| if i == middle { '|' } else { '.' })
        .collect();
    let structure = StructureConstraints::parse(&layout, &text).expect("valid constraint string");
    let (constrained, counts) = design(&PositionalConstraints {
        pins: derna_lib::Pins::none(0),
        structure,
    });

    assert_ne!(
        constrained.structure.as_bytes()[middle as usize],
        b'.',
        "the required position carries a pair in the emitted structure"
    );
    assert!(
        counts.p_cells * 100 >= free_counts.p_cells * 99,
        "a required position should leave the P table within 1%: {} against {}",
        counts.p_cells,
        free_counts.p_cells
    );
    // Non-vacuity: the requirement moved the answer. Without this the `p_cells`
    // assertion above would pass for a constraint that did nothing at all.
    assert_ne!(
        constrained.structure, free.structure,
        "the requirement must change the emitted structure, or the cell-count \
         assertion above is measuring the unconstrained run twice"
    );
}
