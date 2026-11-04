//! The flank-versus-flank pre-check's contracts.
//!
//! The instrument folds two *fixed* flanks against each other with no CDS
//! present, so what it finds is a property of the flanks alone. What is pinned
//! here:
//!
//! 1. **It cannot fire for `5′↔CDS` alone.** Every pair of that class has one
//!    endpoint in the designed CDS, so no flank-only fold can reach it. The
//!    pre-check reports that it is not live rather than reporting a clean
//!    result, which would be a claim it has no evidence for. That is the
//!    `five-prime` table's case.
//! 2. **It fires under the shipped default**, whose `5′UTR↔3′UTR` entry does
//!    reach a pair of flank positions. Two complementary flanks duplex; those
//!    pairs are refused, and both of their endpoints are in fixed sequence.
//! 3. **A live table can still come back clear**, so "live" is not "blocked".
//! 4. **Coordinates map back to the whole molecule** — the fold sees the flanks
//!    adjacent, and the mRNA does not.

use derna_lib::{flank_precheck, CdsWindow, Enforcement, PairClass, Params, RegionConstraints};

/// Complementary flanks: whatever the CDS turns out to be, these two can pair
/// with each other base-for-base.
const CLAMP_UTR5: &str = "GGGGGGGGGGGG";
const CLAMP_UTR3: &str = "CCCCCCCCCCCC";

fn params() -> Params {
    Params::load_default().expect("default Turner 2004 parameters")
}

/// The shipped default — `5′↔CDS` plus the clamp class — which is the smallest
/// table the instrument is live for.
fn no_clamp() -> RegionConstraints {
    RegionConstraints::no_clamp(Enforcement::Forbid)
}

/// (1) Under `--regions five-prime` there is nothing for this fold to test, and
/// it says so: a table whose only entry has an endpoint in the designed CDS is
/// beyond a CDS-free instrument's reach.
#[test]
fn five_prime_isolation_alone_cannot_fire() {
    let report = flank_precheck(
        &params(),
        CLAMP_UTR5,
        CLAMP_UTR3,
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
    )
    .expect("the flanks fold");

    assert!(!report.is_live());
    assert!(report.live_classes.is_empty());
    assert!(
        report.fold.is_none(),
        "a table with nothing to test earns no fold"
    );
    assert!(
        !report.blocks(),
        "not-live must never read as a clean verdict"
    );
}

/// (2) The case the instrument exists for: flanks that duplex under a table
/// that forbids the duplex. Every pair it reports has both endpoints in fixed
/// sequence, which is what distinguishes it from every other diagnostic here.
#[test]
fn complementary_flanks_block_a_table_that_forbids_the_clamp() {
    let report = flank_precheck(&params(), CLAMP_UTR5, CLAMP_UTR3, no_clamp()).expect("folds");

    assert_eq!(report.live_classes, [PairClass::Utr5Utr3]);
    assert!(
        report.blocks(),
        "poly-G against poly-C must duplex; free fold was {:?}",
        report.fold.as_ref().map(|f| &f.free_structure)
    );
    for &(i, j) in report.blocking_pairs() {
        assert!(
            i < report.utr5_len && j >= report.utr5_len,
            "a blocking pair under this table joins the two flanks, got ({i}, {j})"
        );
    }
    let fold = report.fold.as_ref().expect("a live pre-check folds");
    assert!(
        fold.gap_kcal_mol > 0.0,
        "forbidding the flanks' preferred structure must cost energy, got {}",
        fold.gap_kcal_mol
    );
}

/// (3) Live is not blocked: the same table over flanks with nothing to say to
/// each other comes back clear.
#[test]
fn flanks_that_cannot_pair_come_back_clear() {
    let report =
        flank_precheck(&params(), "AAAAAAAAAAAA", "AAAAAAAAAAAA", no_clamp()).expect("folds");

    assert!(report.is_live(), "the table still reaches these positions");
    assert!(!report.blocks());
    assert_eq!(
        report.fold.expect("a live pre-check folds").gap_kcal_mol,
        0.0,
        "poly-A has no structure for the constraint to take away"
    );
}

/// (4) The fold sees the flanks adjacent; the molecule has a CDS between them.
/// A reported pair is only interpretable once mapped back.
#[test]
fn blocking_pairs_map_back_to_whole_mrna_coordinates() {
    let report = flank_precheck(&params(), CLAMP_UTR5, CLAMP_UTR3, no_clamp()).expect("folds");
    let cds = CdsWindow::new(CLAMP_UTR5.len(), 26);
    let cds_nt = cds.end - cds.start;

    for &pair in report.blocking_pairs() {
        let (i, j) = report.in_mrna(pair, cds_nt);
        assert_eq!(i, pair.0, "a 5′ endpoint keeps its position");
        assert_eq!(j, pair.1 + cds_nt, "a 3′ endpoint clears the whole CDS");
        assert!(
            i < cds.start && j >= cds.end,
            "the mapped pair must span the CDS, got ({i}, {j})"
        );
    }
}
