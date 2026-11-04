//! What a less-restricted fold's CKY table already holds for a more-restricted
//! one, and the two properties that make the measurement readable.
//!
//! The counts in [`derna_lib::reuse_probe`] exist to answer a go/no-go: is
//! enough of a child region's table a value its parent already computed to be
//! worth a warm start? A share read off an instrument nobody showed could
//! register one is not an answer, so this file asserts both directions:
//!
//! 1. **The instrument matches when nothing changed.** Folding one lattice
//!    against itself must report every cell identical and no cell dirty. A
//!    comparator that always said "different" would make every share below
//!    zero and look like a finding.
//! 2. **The instrument separates when something did.** A restriction that
//!    removes a codon the optimum was using must leave *some* cell disagreeing,
//!    or the first assertion is vacuous — a comparator that always said "same"
//!    would pass it.
//!
//! On top of those sits the claim the whole measurement rests on: a cell whose
//! span does not reach into a restricted codon accumulates over an identical
//! sub-lattice, so it must come back identical **every time**, not merely
//! usually. That is asserted as an equality rather than a bound, because a
//! bound would pass on an implementation whose spans were computed wrongly.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::reuse_probe::{table_reuse, ReuseInputs, ReuseReport};
use derna_lib::state::{PairKey, State};
use derna_lib::{CodonUsage, Enforcement, Params, RegionConstraints, ScanConfig};

/// Long enough to form real helices over several codons — a restriction has to
/// have spans to be inside — and short enough for a debug-profile test.
const PROTEIN: &str = "MLWKCWMKYFVNQHRTGADESILP";

const LAMBDA: f64 = 0.5;

fn aa() -> Vec<usize> {
    PROTEIN
        .chars()
        .map(|c| aa_to_index(c).expect("a residue"))
        .collect()
}

/// The unrestricted lattice: every synonymous codon at every position.
fn full(aa: &[usize]) -> Vec<Vec<[u8; 3]>> {
    aa.iter()
        .map(|&a| AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect())
        .collect()
}

/// `parent` with position `q`'s first codon dropped — the one exclusion a
/// branch-and-bound child carries.
fn without_first(parent: &[Vec<[u8; 3]>], q: usize) -> Vec<Vec<[u8; 3]>> {
    let mut child = parent.to_vec();
    child[q].remove(0);
    child
}

/// `parent` with position `q` pinned to its first codon — the deepest
/// restriction one position admits.
fn pinned(parent: &[Vec<[u8; 3]>], q: usize) -> Vec<Vec<[u8; 3]>> {
    let mut child = parent.to_vec();
    child[q].truncate(1);
    child
}

/// The first position offering more than two codons, so that both a single
/// exclusion and a pin leave a non-empty set and the two are different regions.
fn branchable(aa: &[usize]) -> usize {
    (0..aa.len())
        .find(|&i| AMINO_ACIDS[aa[i]].codons.len() > 2)
        .expect("some position has three or more synonymous codons")
}

fn report(
    params: &Params,
    usage: &CodonUsage,
    aa: &[usize],
    constraints: RegionConstraints,
    parent: &[Vec<[u8; 3]>],
    child: &[Vec<[u8; 3]>],
) -> ReuseReport {
    table_reuse(
        &ReuseInputs {
            params,
            aa,
            usage,
            lambda: LAMBDA,
            utr5: "",
            utr3: "",
            constraints,
            config: ScanConfig::default(),
        },
        parent,
        child,
    )
    .expect("the folds run")
    .expect("both lattices admit an encoding")
}

fn fixtures() -> (Params, CodonUsage, Vec<usize>) {
    (
        Params::load_default().expect("params"),
        CodonUsage::load_default().expect("codon usage"),
        aa(),
    )
}

/// What a parent's tables would cost to keep, against what the region that
/// names them costs.
///
/// A warm start has to hold a parent's tables until its last child is solved,
/// and a best-first walk interleaves children from many parents, so the cache
/// is the whole frontier rather than one table. The gap between the two
/// quantities is the whole of that objection — a region is a pin vector, its
/// tables are a fold — and it is asserted rather than described, because the
/// arithmetic in the write-up is only as good as the layout it assumes.
#[test]
fn a_regions_tables_are_orders_larger_than_the_region_naming_them() {
    use std::mem::size_of;
    let (params, usage, aa) = fixtures();
    let parent = full(&aa);
    let r = report(
        &params,
        &usage,
        &aa,
        RegionConstraints::default(),
        &parent,
        &parent,
    );
    // What a `DesignRegion` carries: one optional codon and an exclusion list
    // per position. The exclusions are empty at the root, which is the cheapest
    // a region ever is and so the conservative side of the comparison.
    let region_bytes = aa.len() * (size_of::<Option<[u8; 3]>>() + size_of::<Vec<[u8; 3]>>());
    // What its fold carries: the cells it materialized, at the payload each
    // table's key and state occupy. Hash-table and dense-column slack is left
    // out, so this too is the conservative side.
    let cell_bytes = |name: &str| match name {
        "P" | "Multi" => size_of::<PairKey>() + size_of::<State>(),
        "C" => size_of::<u32>() + size_of::<State>(),
        _ => size_of::<State>(),
    };
    let table_bytes: usize = r
        .tables
        .iter()
        .map(|t| t.parent_cells * cell_bytes(t.name))
        .sum();
    println!(
        "region {region_bytes} B, tables {table_bytes} B over {} cells, ratio {:.0}x",
        r.total.parent_cells,
        table_bytes as f64 / region_bytes as f64
    );
    // A loose floor on purpose. The gap widens with length — the tables grow
    // with the molecule squared while the region grows with the protein — so
    // pinning it tightly at the one length a debug-profile test can afford
    // would make the test about this protein rather than about the shape.
    assert!(
        table_bytes > 100 * region_bytes,
        "tables {table_bytes} B against a region's {region_bytes} B"
    );
}

/// Folding a lattice against itself: the instrument's zero point.
#[test]
fn an_unchanged_lattice_matches_itself_everywhere() {
    let (params, usage, aa) = fixtures();
    let parent = full(&aa);
    let r = report(
        &params,
        &usage,
        &aa,
        RegionConstraints::default(),
        &parent,
        &parent,
    );
    assert!(r.total.child_cells > 0, "the fold materialized no cells");
    assert_eq!(r.restricted, Vec::<usize>::new());
    assert_eq!(
        r.total.dirty, 0,
        "nothing was restricted, so no cell can be dirty"
    );
    assert_eq!(
        r.total.identical(),
        r.total.child_cells,
        "a fold disagreed with itself"
    );
    assert_eq!(r.total.absent, 0, "a cell went missing against itself");
    assert_eq!(r.lane_divergent_nodes, 0);
}

/// The other direction: a restriction the optimum was using must move
/// something, or the match above proves nothing.
#[test]
fn a_restriction_leaves_some_cell_disagreeing() {
    let (params, usage, aa) = fixtures();
    let parent = full(&aa);
    let q = branchable(&aa);
    let r = report(
        &params,
        &usage,
        &aa,
        RegionConstraints::default(),
        &parent,
        &without_first(&parent, q),
    );
    assert_eq!(r.restricted, vec![q]);
    assert!(
        r.total.dirty > 0,
        "no cell's span reached the restricted codon"
    );
    assert!(
        r.total.dirty > r.total.dirty_identical,
        "every dirty cell agreed — the comparator cannot separate the two folds"
    );
}

/// The claim the share rests on: a span that misses every restricted codon
/// accumulates over an identical sub-lattice, so it agrees without exception.
#[test]
fn every_cell_clear_of_the_restriction_is_identical() {
    let (params, usage, aa) = fixtures();
    let parent = full(&aa);
    let q = branchable(&aa);
    for child in [without_first(&parent, q), pinned(&parent, q)] {
        let r = report(
            &params,
            &usage,
            &aa,
            RegionConstraints::default(),
            &parent,
            &child,
        );
        assert!(r.total.clean > 0, "no cell was clear of the restriction");
        assert_eq!(
            r.total.clean, r.total.clean_identical,
            "a cell clear of the restriction changed value"
        );
        assert_eq!(
            r.total.absent_clean, 0,
            "a cell clear of the restriction has no parent coordinate"
        );
        assert!(
            r.comparable(),
            "the two folds address different cells outside the restricted codon"
        );
    }
}

/// The same, under the region-pair constraint a bounded search actually runs with:
/// the constraint drops `P` cells, and it must drop the same ones on both sides.
#[test]
fn a_region_constraint_does_not_break_the_addressing() {
    let (params, usage, aa) = fixtures();
    let parent = full(&aa);
    let q = branchable(&aa);
    let r = report(
        &params,
        &usage,
        &aa,
        RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        &parent,
        &without_first(&parent, q),
    );
    assert!(r.total.clean > 0, "no cell was clear of the restriction");
    assert_eq!(r.total.clean, r.total.clean_identical);
    assert_eq!(r.total.absent_clean, 0);
    assert!(r.comparable());
}
