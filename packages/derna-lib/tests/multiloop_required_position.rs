//! A required position **inside a multi-loop's unpaired stretch** — the case the
//! reduction screen could not reach.
//!
//! The shared multi-loop opening is what turns the `O(n³)` multi-loop scan into
//! `O(n²)`: an opening wraps an `M2` core, carries the 5′ flank, and is pooled
//! keyed only on `(opening node, opening nucleotide)` — the core's right
//! boundary is folded into the value and dropped from the key. Any later
//! frontier closes it, leaving `[core_end, frontier)` unpaired. Requiring a
//! position inside that stretch is therefore a joint condition on a coordinate
//! the pool has already minimized away and one it has not seen yet, and it is a
//! *truncation* only because admissibility is monotone in the scan direction:
//! once the scan passes a required position every pooled opening is permanently
//! unusable, so the pool is evicted whole.
//!
//! # Why this case needed building rather than sweeping
//!
//! Three independent reasons it falls outside the screen's evidence:
//!
//! - The screen's floor only proves the **preserving** direction. It requires
//!   the positions the unconstrained optimum already pairs, so its target is
//!   feasible by construction and only over-eviction shows up.
//! - The reduced-vs-un-reduced differential is **blind** here: both
//!   internal-loop closures run the same shared-opening path, so they would be
//!   wrong together.
//! - Brute force **cannot reach it**: a construct where a multi-loop is
//!   *optimal* rather than merely possible is far past enumerable size.
//!
//! Getting it wrong is silent. An eviction that rejects too little returns a
//! design violating the requirement — visible, and the post-condition below
//! catches it. One that rejects on the wrong condition returns a **worse**
//! optimum with every post-condition satisfied, which is why the ground truth
//! here is an independent implementation rather than a post-condition.
//!
//! # The independent arm
//!
//! `#[ignore]`d and release-only for cost, [`the_pooled_and_direct_multi_loop_closures_agree`]
//! folds the same problem through the **AC-threaded** folder, which uses no
//! shared-opening separation at all: it closes every frontier directly against
//! each wrap, so its multi-loop stretches are enumerated rather than pooled and
//! evicted. The two share no multi-loop code, so agreement is evidence in a way
//! the reduced-vs-un-reduced differential is not. It is reached by naming a
//! motif set that forbids nothing — an automaton with states, but a window too
//! short to hold an occurrence — so the fold is the same problem, taken through
//! the other closure.

use derna_lib::{
    design_composite, design_with_config, Beam, BeamError, CodonUsage, CompositeSpec, Design,
    Layout, MotifWindow, Params, PositionalConstraints, RegionConstraints, ScanConfig,
    StructureConstraints, UtrProfile, UtrSpec,
};
use rna_mfe::evaluate;

/// The construct whose exact optimum carries a genuine multi-branch junction.
/// A common expression tag; nothing about it is special beyond that its optimum
/// branches, which the test asserts rather than assumes.
const PROTEIN: &str = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS";

/// The λ at which that optimum branches.
const LAMBDA: f64 = 0.5;

/// Objective tolerance in dekacalories (absorbs the rounded-integer vs continuous objective gap).
const BRD_TOL: f64 = 0.5;

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

fn brd(lambda: f64, mfe_kcal: f64, cai_log: f64) -> f64 {
    lambda * (mfe_kcal * 100.0).round() - (1.0 - lambda) * cai_log * 100.0
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

/// The unpaired stretches of every multi-loop in `structure`, as
/// `(5′ stretch, 3′ stretch)` half-open ranges.
///
/// Read off the emitted dot-bracket rather than from the scan's tables, so the
/// test's notion of "inside a multi-loop's unpaired stretch" is independent of
/// the machinery it is checking.
fn multiloop_stretches(structure: &str) -> Vec<((usize, usize), (usize, usize))> {
    let pairs = pairs_of(structure);
    let mut out = Vec::new();
    for &(i, j) in &pairs {
        let mut children: Vec<(usize, usize)> = pairs
            .iter()
            .copied()
            .filter(|&(a, b)| a > i && b < j)
            .filter(|&(a, b)| !pairs.iter().any(|&(c, e)| c > i && e < j && c < a && b < e))
            .collect();
        if children.len() < 2 {
            continue;
        }
        children.sort();
        out.push((
            (i + 1, children[0].0),
            (children.last().expect("non-empty").1 + 1, j),
        ));
    }
    out
}

/// Design requiring every position in `required` to pair, through the composite
/// entry and its pooled `multi_open` path.
fn require(required: &[u32]) -> Result<Design, BeamError> {
    let layout = Layout::new(0, PROTEIN.len() as u32, 0);
    let marks: String = (0..layout.len())
        .map(|p| if required.contains(&p) { '|' } else { '.' })
        .collect();
    let mut pc = PositionalConstraints::none(layout.len());
    pc.structure = StructureConstraints::parse(&layout, &marks).expect("valid marks");
    let (no5, no3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    design_composite(
        &params(),
        PROTEIN,
        &usage(),
        LAMBDA,
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
        exact(),
    )
}

fn unconstrained() -> Design {
    design_with_config(&params(), PROTEIN, &usage(), LAMBDA, exact())
        .expect("the unconstrained design exists")
}

/// The unconstrained optimum's two multi-loop unpaired stretches, as one
/// interior position each.
///
/// Asserted rather than hard-coded: if the substrate's optimum ever stops
/// branching, this fails loudly instead of leaving the eviction untested.
fn stretch_positions() -> (u32, u32) {
    let free = unconstrained();
    let stretches = multiloop_stretches(&free.structure);
    let ((five_lo, five_hi), (three_lo, three_hi)) = *stretches
        .first()
        .unwrap_or_else(|| panic!("the optimum no longer branches: {}", free.structure));
    assert!(
        five_hi > five_lo && three_hi > three_lo,
        "both multi-loop stretches must hold at least one unpaired position, got \
         [{five_lo}, {five_hi}) and [{three_lo}, {three_hi}) in {}",
        free.structure
    );
    (five_lo as u32, three_lo as u32)
}

/// The setup itself, checked before anything is concluded from it.
#[test]
fn the_unconstrained_optimum_branches_with_both_stretches_occupied() {
    let free = unconstrained();
    let (five, three) = stretch_positions();
    assert_eq!(
        free.structure.as_bytes()[five as usize],
        b'.',
        "the 5′ stretch position must be unpaired in the unconstrained optimum"
    );
    assert_eq!(
        free.structure.as_bytes()[three as usize],
        b'.',
        "the 3′ stretch position must be unpaired in the unconstrained optimum"
    );
}

/// Requiring a position inside the **3′** stretch is the case the pool eviction
/// exists for: every opening whose stretch would span it is permanently
/// inadmissible, so the eviction must *reject* it.
///
/// Requiring one inside the **5′** stretch exercises the other half of the
/// split — the `O(1)` test at opening creation, where an opening that fails is
/// simply never pooled.
#[test]
fn a_required_position_in_a_multi_loop_stretch_is_honoured_and_costs_something() {
    let free = unconstrained();
    let free_brd = brd(LAMBDA, free.mfe_kcal_mol, free.cai_log);
    let (five, three) = stretch_positions();
    for (label, pos) in [("5′ stretch", five), ("3′ stretch", three)] {
        let got = require(&[pos]).expect("a design pairing this position exists");

        // The post-condition, read off what was emitted.
        assert_ne!(
            got.structure.as_bytes()[pos as usize],
            b'.',
            "{label}: position {pos} left unpaired: {}",
            got.structure
        );
        // The design changed, so the constraint removed the derivation it was
        // pointed at rather than being satisfied by the instance.
        assert_ne!(
            (&free.sequence, &free.structure),
            (&got.sequence, &got.structure),
            "{label}: the requirement changed nothing, so nothing was evicted"
        );
        // Constraining can only cost. A *better* objective would mean the
        // eviction let through a derivation the unconstrained scan did not have.
        let got_brd = brd(LAMBDA, got.mfe_kcal_mol, got.cai_log);
        assert!(
            got_brd >= free_brd - BRD_TOL,
            "{label}: the constrained optimum beat the unconstrained one \
             ({got_brd} < {free_brd})"
        );
        // The emitted structure independently re-scores to the reported energy.
        // A traceback that matched a stored value carrying something the loop
        // scorer does not would show up here and nowhere else.
        let scored = evaluate(&params(), got.sequence.as_bytes(), &got.structure)
            .expect("the emitted structure scores")
            .to_kcal_mol();
        assert!(
            (scored - got.mfe_kcal_mol).abs() < 1e-9,
            "{label}: reported {} but the emitted structure scores {scored}",
            got.mfe_kcal_mol
        );
    }
}

/// The over-eviction control. Requiring every position the unconstrained
/// optimum already pairs leaves that structure feasible by construction, and no
/// pooled opening in it spans a required position — so it must survive the
/// eviction untouched.
///
/// An eviction that rejects too much shows up here as a strictly worse optimum,
/// which no post-condition would catch.
#[test]
fn requiring_what_the_optimum_already_pairs_evicts_nothing() {
    let free = unconstrained();
    let paired: Vec<u32> = free
        .structure
        .bytes()
        .enumerate()
        .filter(|(_, c)| *c != b'.')
        .map(|(i, _)| i as u32)
        .collect();
    assert!(
        !paired.is_empty() && !multiloop_stretches(&free.structure).is_empty(),
        "the control is only meaningful over a branched optimum"
    );
    let got = require(&paired).expect("the optimum itself is feasible");
    assert_eq!(
        (
            free.sequence,
            free.structure,
            free.mfe_kcal_mol,
            free.cai_log
        ),
        (got.sequence, got.structure, got.mfe_kcal_mol, got.cai_log),
    );
}

/// The independent arm: the same constrained problem through the AC-threaded
/// folder, which closes each multi-loop frontier directly and pools nothing.
///
/// This is the check the screen could not make. Both internal-loop closures in
/// the main scan share the pooled `multi_open` path, so a differential between
/// them says nothing about the eviction; this folder does not have that path at
/// all, so an eviction rejecting on the wrong condition diverges here.
///
/// `#[ignore]`d and release-only: the unreduced folder is ~75× slower on this
/// construct, which is the price of it being a genuinely separate implementation.
#[test]
#[ignore = "release-only: the AC-threaded folder is ~75x slower at this length"]
fn the_pooled_and_direct_multi_loop_closures_agree() {
    let (five, three) = stretch_positions();
    let layout = Layout::new(0, PROTEIN.len() as u32, 0);
    // A motif set with a window too short to hold an occurrence entirely inside
    // it forbids nothing, so the two arms fold the same problem — but the
    // automaton has states, so the run takes the AC-threaded folder.
    let inert_motifs = ["GGGGGGGG".to_string()];
    for required in [vec![], vec![five], vec![three], vec![five, three]] {
        let marks: String = (0..layout.len())
            .map(|p| if required.contains(&p) { '|' } else { '.' })
            .collect();
        let mut pc = PositionalConstraints::none(layout.len());
        pc.structure = StructureConstraints::parse(&layout, &marks).expect("valid marks");
        let (no5, no3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
        let direct = design_composite(
            &params(),
            PROTEIN,
            &usage(),
            LAMBDA,
            &CompositeSpec {
                utr5: &no5,
                utr3: &no3,
                constraints: RegionConstraints::default(),
                profile: UtrProfile::inert(),
                motifs: &inert_motifs,
                motif_window: MotifWindow::new(0, 4),
                flank_complement: None,
                complement_classes: None,
                positional: &pc,
            },
            exact(),
        )
        .expect("feasible");
        let pooled = require(&required).expect("feasible");
        assert_eq!(
            (pooled.mfe_kcal_mol, pooled.cai_log),
            (direct.mfe_kcal_mol, direct.cai_log),
            "required={required:?}: the pooled and direct multi-loop closures disagree\n\
             pooled {}\ndirect {}",
            pooled.structure,
            direct.structure,
        );
        for &p in &required {
            assert_ne!(
                direct.structure.as_bytes()[p as usize],
                b'.',
                "required={required:?}: the direct closure left {p} unpaired: {}",
                direct.structure
            );
        }
    }
}
