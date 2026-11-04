//! Which lattice a forbidden-motif set routes to, and why the question is about
//! the product rather than the automaton.
//!
//! The AC-threaded folder is the slower, un-reduced substrate; a run reaches it
//! only when the motif set genuinely constrains. "Genuinely" has two candidate
//! readings and they disagree on real inputs. The automaton reading — *does this
//! pattern set match anything at all* — sends a run down the folder whenever the
//! set is non-empty, including when the protein cannot spell a single one of its
//! patterns, and pays the whole AC axis to rediscover that. The product reading
//! — *did the lift remove any transition this protein's lattice offers* — is the
//! question the routing decision actually turns on.
//!
//! The two claims below are the contract, and they pull in opposite directions
//! on purpose:
//!
//! 1. **A set that removed nothing routes to the base scan**, and its design is
//!    the unconstrained one character for character. The cell counts are the
//!    observable: they are reported by the scan and are `None` from the folder,
//!    so they say which lattice ran without timing anything.
//! 2. **A set that removed even one transition routes to the folder**, however
//!    little it removed. This direction is the one that must not be relaxed: a
//!    genuinely constrained run sent to the base scan emits a sequence the set
//!    forbids, with no error and no diagnostic. The case here is the one a
//!    per-codon census gets wrong — a pattern straddling a codon boundary that
//!    leaves every codon usable somewhere.

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::ac::Ac;
use derna_lib::dfa;
use derna_lib::product::ProductDfa;
use derna_lib::{
    design, design_composite_counted, design_with_placed_motifs, Beam, CellCounts, CodonUsage,
    CompositeSpec, Design, MotifWindow, Params, PositionalConstraints, RegionConstraints,
    ScanConfig, UtrProfile, UtrSpec,
};

/// Short enough for a debug-profile test on the folder, long enough to form the
/// helices a fold has to choose between.
const PROTEIN: &str = "MLWKCWMKYFVN";

/// A pattern no encoding of [`PROTEIN`] spells: it needs `ACGUACGUACGU`
/// contiguously, and the codon lattice offers no such run.
const INERT: &str = "ACGUACGUACGU";

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

/// Design `protein` under `motifs` and the given flanks, returning the design
/// and whichever cell counts the run produced. `None` counts mean the run
/// folded the AC-threaded lattice.
fn route(protein: &str, motifs: &[&str], utr5: &str, utr3: &str) -> (Design, Option<CellCounts>) {
    let (utr5, utr3) = (
        UtrSpec::Fixed(utr5.to_string()),
        UtrSpec::Fixed(utr3.to_string()),
    );
    let positional = &PositionalConstraints::empty();
    design_composite_counted(
        &params(),
        protein,
        &usage(),
        LAMBDA,
        &CompositeSpec {
            utr5: &utr5,
            utr3: &utr3,
            constraints: RegionConstraints::default(),
            profile: UtrProfile::inert(),
            motifs,
            motif_window: MotifWindow::all(),
            flank_complement: None,
            complement_classes: None,
            positional,
        },
        exact(),
    )
    .expect("feasible")
}

/// A pattern set the protein cannot spell takes the base scan and returns the
/// unconstrained design verbatim — the CDS-only early return, which is where
/// most of the cost of getting this wrong sits.
#[test]
fn unspellable_motif_set_returns_the_unconstrained_design() {
    let plain = design(&params(), PROTEIN, &usage(), LAMBDA).expect("design");
    let (routed, counts) = route(PROTEIN, &[INERT], "", "");
    assert_eq!(
        plain, routed,
        "a set that forbade nothing must not move the design",
    );
    assert!(
        counts.is_some_and(|c| c.total() > 0),
        "the run should have taken the base scan, which reports its cells",
    );
}

/// The same reroute with flanks present. The early return needs an empty
/// molecule, but the routing decision does not: a run that keeps its UTRs still
/// takes the base scan rather than the AC-threaded lattice.
#[test]
fn unspellable_motif_set_with_flanks_takes_the_base_scan() {
    let (with_motif, counts) = route(PROTEIN, &[INERT], "GGGAGGCGGCGGAGGCGG", "GGAGGCGGAGGCGGA");
    let (without, _) = route(PROTEIN, &[], "GGGAGGCGGCGGAGGCGG", "GGAGGCGGAGGCGGA");
    assert_eq!(with_motif, without, "an inert set must not move the design");
    assert!(
        counts.is_some_and(|c| c.total() > 0),
        "the run should have taken the base scan, which reports its cells",
    );
}

/// The direction that must stay conservative. `FF` spells `UUU`/`UUC` twice, so
/// forbidding `UUUUUU` removes exactly one of its four encodings while leaving
/// every codon usable in a survivor — a per-codon census reads inert here. The
/// run must still reach the folder, and its design must avoid the pattern.
#[test]
fn a_single_removed_edge_still_routes_to_the_folder() {
    let (design, counts) = route("FF", &["UUUUUU"], "", "");
    assert!(
        counts.is_none(),
        "a set that removed a transition must fold the constrained lattice",
    );
    assert!(
        !design.sequence.contains("UUUUUU"),
        "the design spells a forbidden pattern: {}",
        design.sequence,
    );
}

/// The gate reads the product for *this* protein, so one pattern set can route
/// two proteins two different ways. Without that, a set is either always cheap
/// or always dear regardless of what it meets.
#[test]
fn the_same_motif_set_routes_two_proteins_differently() {
    let (_, inert_here) = route("FF", &[INERT], "", "");
    let (_, binding_here) = route("FF", &["UUUUUU"], "", "");
    assert!(
        inert_here.is_some(),
        "ACGUACGUACGU forbids nothing FF spells"
    );
    assert!(
        binding_here.is_none(),
        "UUUUUU forbids one of FF's encodings"
    );
}

/// The same question on the **placed** path, where a walk accumulates one cut
/// per counterexample and each cut holds only at the place it occurred.
///
/// A cut is likelier to be inert than a hand-written motif set, because it is
/// pinned to a window: it constrains only if the lattice can spell that pattern
/// *there*. The placed entry point returns a design and no cell counts, so the
/// routing is asserted on the product and its consequence on the design.
#[test]
fn a_placed_cut_that_forbids_nothing_returns_the_unplaced_design() {
    let (params, usage) = (params(), usage());
    let plain = placed_design(&params, &usage, PROTEIN, &[]);

    let inert = [(INERT.to_string(), MotifWindow::new(0, 36))];
    let graph = dfa::build(&residues(PROTEIN), &usage);
    let ac = Ac::build_placed(&inert).expect("valid placement");
    assert!(
        !ac.is_pass_through(),
        "a placed pattern should build a real automaton",
    );
    assert!(
        !ProductDfa::build(&graph, &ac).dropped_edge(),
        "the premise: this lattice cannot spell that pattern in that window",
    );

    assert_eq!(
        plain,
        placed_design(&params, &usage, PROTEIN, &inert),
        "a cut that forbade nothing must not move the design",
    );
}

/// The conservative direction on the placed path: a cut the lattice *can* spell
/// inside its window still routes to the folder, and the design avoids it.
#[test]
fn a_placed_cut_that_forbids_something_is_still_enforced() {
    let (params, usage) = (params(), usage());
    let cut = [("UUUUUU".to_string(), MotifWindow::new(0, 6))];
    let graph = dfa::build(&residues("FF"), &usage);
    assert!(
        ProductDfa::build(&graph, &Ac::build_placed(&cut).expect("valid")).dropped_edge(),
        "the premise: FF can spell UUUUUU inside [0, 6)",
    );

    let design = placed_design(&params, &usage, "FF", &cut);
    assert!(
        !design.sequence.contains("UUUUUU"),
        "the design spells the cut pattern: {}",
        design.sequence,
    );
}

/// Design `protein` under a placed cut set, with no flanks and no region table.
fn placed_design(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    placed: &[(String, MotifWindow)],
) -> Design {
    design_with_placed_motifs(
        params,
        protein,
        usage,
        LAMBDA,
        placed,
        "",
        "",
        RegionConstraints::default(),
        exact(),
    )
    .expect("feasible")
    .expect("a non-empty design space")
}

/// The residue indices of a single-letter amino-acid protein.
fn residues(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}
