//! Can any design satisfy the complement, and what is the smallest `k` that
//! admits one?
//!
//! A region table can never be infeasible — the all-unpaired structure satisfies
//! every one of them — so a complement is the only constraint that can empty the
//! space. The check is pure reachability over the Aho–Corasick × codon-graph
//! product, which is what makes it affordable enough to run on every complement
//! run rather than on request.
//!
//! **The bisection is the part that needs checking.** It returns the true
//! boundary rather than a grid's resolution only if feasibility is monotone in
//! `k`, so that monotonicity is asserted directly and the bisection is compared
//! against a linear scan over the same range. Assuming the property and testing
//! only the answer would pass on an instance where the boundary happens to be
//! where a scan and a bisection agree.

use derna_lib::{
    check_complement, removed_codons, CodonUsage, CompositeSpec, FlankComplement, MotifWindow,
    PositionalConstraints, RegionConstraints, UtrProfile, UtrSpec,
};

/// A `G`-rich given flank against a designed one: the complement it derives is
/// pyrimidine runs, which the coding sequence has to avoid.
const GIVEN: &str = "GGGAGGGAGGGA";
const PROTEIN: &str = "MSLLKW";

fn usage() -> CodonUsage {
    CodonUsage::load_default().expect("codon usage")
}

fn spec<'a>(
    utr5: &'a UtrSpec,
    utr3: &'a UtrSpec,
    fc: Option<FlankComplement>,
    motifs: &'a [String],
) -> CompositeSpec<'a, String> {
    CompositeSpec {
        utr5,
        utr3,
        constraints: RegionConstraints::default(),
        profile: UtrProfile::inert(),
        motifs,
        motif_window: MotifWindow::all(),
        flank_complement: fc,
        complement_classes: None,
        positional: PositionalConstraints::empty(),
    }
}

/// Feasibility is **monotone in `k`**: a longer window matches fewer segments,
/// so a `k` that admits a design admits one at every larger `k` too.
///
/// This is what the bisection rests on. Asserted over the whole range rather
/// than argued, because a bisection over a non-monotone predicate returns a
/// boundary that is not the boundary.
#[test]
fn feasibility_is_monotone_in_k() {
    let usage = usage();
    let (utr5, utr3) = (UtrSpec::Fixed(GIVEN.into()), UtrSpec::Designed(9));
    let none: Vec<String> = Vec::new();
    let mut seen_infeasible = false;
    let mut seen_feasible = false;
    let mut previous = false;
    for k in 1..=GIVEN.len() + 1 {
        let fc = FlankComplement::wobble(k);
        let report = check_complement(PROTEIN, &usage, &spec(&utr5, &utr3, Some(fc), &none))
            .expect("the check runs")
            .expect("a complement was asked for");
        assert!(
            !previous || report.feasible,
            "k={k} is infeasible although k={} was feasible, so feasibility is \
             not monotone and a bisection over it returns nothing meaningful",
            k - 1,
        );
        previous = report.feasible;
        seen_infeasible |= !report.feasible;
        seen_feasible |= report.feasible;
    }
    assert!(
        seen_infeasible && seen_feasible,
        "every k gave the same answer, so monotonicity held vacuously",
    );
}

/// The bisection returns the same boundary a linear scan does.
///
/// The scan is the definition; the bisection is the implementation. They are
/// compared on an instance whose boundary is interior — asserted, since a
/// boundary at either end of the range is where the two agree for free.
#[test]
fn the_bisection_returns_the_boundary_a_scan_finds() {
    let usage = usage();
    let (utr5, utr3) = (UtrSpec::Fixed(GIVEN.into()), UtrSpec::Designed(9));
    let none: Vec<String> = Vec::new();
    let ceiling = GIVEN.len() + 1;

    let feasible_at = |k: usize| {
        check_complement(
            PROTEIN,
            &usage,
            &spec(&utr5, &utr3, Some(FlankComplement::wobble(k)), &none),
        )
        .expect("the check runs")
        .expect("a complement was asked for")
        .feasible
    };
    let scanned = (1..=ceiling)
        .find(|&k| feasible_at(k))
        .expect("some k admits a design");
    assert!(
        scanned > 1 && scanned < ceiling,
        "the boundary is at an end of the range ({scanned}), where a scan and a \
         bisection agree without the bisection being right",
    );

    // Asked from both sides of the boundary, since the two enter the bisection
    // by different branches.
    for k in [1, scanned, ceiling] {
        let report = check_complement(
            PROTEIN,
            &usage,
            &spec(&utr5, &utr3, Some(FlankComplement::wobble(k)), &none),
        )
        .expect("the check runs")
        .expect("a complement was asked for");
        assert_eq!(
            report.smallest_feasible_k,
            Some(scanned),
            "asked at k={k}, the bisection disagreed with the scan",
        );
        assert!(
            report.checks <= ceiling,
            "the bisection spent {} checks over a range of {ceiling}, which is \
             a scan wearing a bisection's name",
            report.checks,
        );
    }
}

/// A run with no complement has nothing that can empty its space, and says so
/// by returning nothing rather than by returning feasible.
#[test]
fn a_run_without_a_complement_is_not_checked() {
    let usage = usage();
    let (utr5, utr3) = (UtrSpec::Fixed(GIVEN.into()), UtrSpec::Designed(9));
    let none: Vec<String> = Vec::new();
    assert!(
        check_complement(PROTEIN, &usage, &spec(&utr5, &utr3, None, &none))
            .expect("the check runs")
            .is_none(),
    );
}

/// The lattice reading of *binds*: absent where there is no sequence-space
/// constraint, and a count where there is.
///
/// Absent rather than `false`, because a `false` for a constraint the run does
/// not carry is a claim the run did not make — and it is the claim that makes a
/// vacuity rate unreadable.
#[test]
fn the_lattice_reading_is_absent_without_a_constraint() {
    let usage = usage();
    let (utr5, utr3) = (UtrSpec::Fixed(GIVEN.into()), UtrSpec::Designed(9));
    let none: Vec<String> = Vec::new();
    assert_eq!(
        removed_codons(PROTEIN, &usage, &spec(&utr5, &utr3, None, &none)).expect("runs"),
        None,
        "a run with no motif set and no derivation reported a lattice reading",
    );

    // A tight complement removes codons; a loose one removes none. Both are
    // present, and the difference between them is the whole point of the count.
    let tight = removed_codons(
        PROTEIN,
        &usage,
        &spec(&utr5, &utr3, Some(FlankComplement::wobble(3)), &none),
    )
    .expect("runs")
    .expect("a complement was asked for");
    let loose = removed_codons(
        PROTEIN,
        &usage,
        &spec(&utr5, &utr3, Some(FlankComplement::wobble(9)), &none),
    )
    .expect("runs")
    .expect("a complement was asked for");
    assert!(
        tight.0 > loose.0,
        "a tighter complement removed no more codons than a looser one \
         ({} vs {}), so the count is not reading the constraint",
        tight.0,
        loose.0,
    );
    assert_eq!(
        tight.1, loose.1,
        "the codon total is a property of the protein"
    );
}
