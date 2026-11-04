//! The walk's floor cut ([`WalkOptions::floor_cut`]): each gap slice is also cut
//! at its box's floor alone (`λ = 0`) with a cost to go of zero, which is exact
//! because no edge of the plain lattice adds adaptation. The front must be the
//! same design for design; the folds must hold fewer points, or the cut was
//! never exercised.

use derna_lib::{
    design_front_walk_run, CodonUsage, Params, PassGrid, RegionConstraints, UtrSpec, WalkOptions,
    WalkRun, WalkedFront,
};

/// Long enough for gaps of several slices, where a slice's floor sits above
/// the gap's lower vertex.
const PROTEIN: &str = "MKVLAKGIVLAGQWERSTYLMKVLAKG";
const FLANK5: &str = "ACAUUUGCUU";
const FLANK3: &str = "GGCCAAUUGG";

fn walk(floor_cut: bool, threads: usize) -> WalkedFront {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let options = WalkOptions {
        pass_grid: Some(PassGrid::for_residues(PROTEIN.len())),
        seed_sweep: true,
        cut_probes: true,
        // Narrow slices, so most slices have a floor above their gap's.
        slice_width: Some(50.0),
        floor_cut,
        ..WalkOptions::default()
    };
    design_front_walk_run(
        &params,
        PROTEIN,
        &usage,
        &UtrSpec::Fixed(FLANK5.into()),
        &UtrSpec::Fixed(FLANK3.into()),
        RegionConstraints::full_mutual_isolation(derna_lib::Enforcement::Forbid),
        2f64.powi(-23),
        1.0 - 2f64.powi(-23),
        options,
        threads,
        WalkRun::default(),
    )
    .unwrap()
}

fn designs(w: &WalkedFront) -> Vec<(String, f64, f64)> {
    w.front
        .iter()
        .map(|p| (p.sequence.clone(), p.mfe_kcal_mol, p.cai_log))
        .collect()
}

/// Points the gap folds held at their open and close sites, summed.
fn held(w: &WalkedFront) -> u64 {
    w.gaps
        .iter()
        .map(|g| g.clock.fold.open_points_held + g.clock.fold.open_close_points)
        .sum()
}

#[test]
fn the_floor_cut_keeps_the_front_and_holds_fewer_points() {
    let (plain, cut) = (walk(false, 1), walk(true, 1));
    assert!(
        plain.gaps.iter().any(|g| g.slices > 1),
        "a gap of several slices"
    );
    assert_eq!(designs(&cut), designs(&plain));
    assert_eq!(cut.vertices, plain.vertices);
    assert!(
        held(&cut) < held(&plain),
        "the floor cut removed nothing: {} against {}",
        held(&cut),
        held(&plain)
    );
}

#[test]
fn the_floor_cut_keeps_the_front_on_several_workers() {
    let objectives = |w: &WalkedFront| {
        w.front
            .iter()
            .map(|p| (p.mfe_kcal_mol, p.cai_log))
            .collect::<Vec<_>>()
    };
    assert_eq!(objectives(&walk(true, 3)), objectives(&walk(false, 1)));
}

/// A slice whose fold stores more points than the walk allows is halved, its
/// lower half first, as often as it takes: the front is the same design for
/// design, and the halving fired.
#[test]
fn halving_a_slice_whose_fold_stores_too_much_keeps_the_front() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let run = |slice_points: Option<usize>| {
        let options = WalkOptions {
            pass_grid: Some(PassGrid::for_residues(PROTEIN.len())),
            seed_sweep: true,
            cut_probes: true,
            slice_points,
            ..WalkOptions::default()
        };
        design_front_walk_run(
            &params,
            PROTEIN,
            &usage,
            &UtrSpec::Fixed(FLANK5.into()),
            &UtrSpec::Fixed(FLANK3.into()),
            RegionConstraints::full_mutual_isolation(derna_lib::Enforcement::Forbid),
            2f64.powi(-23),
            1.0 - 2f64.powi(-23),
            options,
            1,
            WalkRun::default(),
        )
        .unwrap()
    };
    let whole = run(None);
    let most = whole
        .gaps
        .iter()
        .map(|g| g.clock.fold.stored)
        .max()
        .expect("a gap");
    let halved = run(Some((most / 3) as usize));
    let count: usize = halved.gaps.iter().map(|g| g.clock.halved).sum();
    assert!(
        count > 0,
        "no slice was halved under a third of the largest fold"
    );
    assert_eq!(designs(&halved), designs(&whole));
    assert!(halved.open.is_empty());
}
