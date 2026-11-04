//! Cutting a family out of the best-first walk.
//!
//! The walk is exact without cuts: its regions tile the design space, so a
//! caller testing each candidate in objective order finds the optimum with any
//! property it likes. A cut is a **speed** move on top of that — it stops the
//! walk re-deriving a family the caller has already rejected — and a speed move
//! that removes the wrong sequences silently changes the answer.
//!
//! So what is checked here is what a cut removes, exhaustively, against a filter
//! applied to the uncut walk: the same designs in the same order, minus exactly
//! those spelling the cut pattern inside the cut window. Two further properties
//! carry the mechanism:
//!
//! - **A cut lands on the heap, not only on later solves.** Regions solved
//!   before it was placed hold optima it may have removed, and a walk that
//!   emitted those would leak a cut design.
//! - **A cut that forbids nothing changes nothing.** Cutting routes the solves
//!   through the AC-threaded constrained folder rather than the plain scan, so a
//!   vacuous cut is the check that the two agree — design for design, in order.

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::{CodonUsage, Enumerator, MotifWindow, Params, Problem};

/// A three-residue protein with several encodings at two of its positions:
/// small enough to enumerate completely and wide enough that a cut removes some
/// designs and leaves others.
const PROTEIN: &str = "MPA";
const UTR5: &str = "GG";
const UTR3: &str = "CC";
const LAMBDA: f64 = 0.5;
/// Objectives are integer dekacalories under the hood; this absorbs the float
/// division only.
const TOL: f64 = 1e-6;

fn setup() -> (Params, CodonUsage) {
    (
        Params::load_default().expect("params"),
        CodonUsage::load_default().expect("usage"),
    )
}

/// Every design the walk emits, with its objective, under whatever cuts `place`
/// installs.
///
/// `place` is called with the walk and the rank about to be emitted, so a case
/// can cut before the first design or partway through.
fn walk(
    params: &Params,
    usage: &CodonUsage,
    mut place: impl FnMut(&mut Enumerator<'_>, usize),
) -> Vec<(f64, String)> {
    let aa: Vec<usize> = PROTEIN
        .chars()
        .map(|c| aa_to_index(c).expect("an amino acid"))
        .collect();
    let mut e = Enumerator::new(
        params,
        Problem {
            protein: PROTEIN,
            aa: &aa,
            usage,
            lambda: LAMBDA,
            utr5: UTR5,
            utr3: UTR3,
            threads: 2,
        },
    )
    .expect("enumerator builds")
    .expect("protein has encodings");

    let mut out = Vec::new();
    loop {
        place(&mut e, out.len());
        let Some(d) = e.next_design().expect("walk advances") else {
            break;
        };
        let mfe_dk = (d.mfe_kcal_mol * 100.0).round();
        let cai_dk = (d.cai_log * 100.0).round();
        out.push((LAMBDA * mfe_dk - (1.0 - LAMBDA) * cai_dk, d.sequence));
    }
    out
}

/// Whether `sequence` spells `pattern` somewhere the occurrence lies **entirely
/// inside** `window` — the reading a placed motif is forbidden under, restated
/// here so the filter cannot inherit the implementation's answer.
fn spells(sequence: &str, pattern: &str, window: MotifWindow) -> bool {
    let lo = window.lo() as usize;
    let hi = (window.hi() as usize).min(sequence.len());
    (lo..hi.saturating_sub(pattern.len() - 1))
        .any(|at| &sequence[at..at + pattern.len()] == pattern)
}

fn assert_same(got: &[(f64, String)], want: &[(f64, String)], label: &str) {
    assert_eq!(
        got.iter().map(|(_, s)| s).collect::<Vec<_>>(),
        want.iter().map(|(_, s)| s).collect::<Vec<_>>(),
        "[{label}] the walk emitted a different set of designs, or in a different order",
    );
    for (rank, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g.0 - w.0).abs() < TOL,
            "[{label}] rank {rank}: objective {} != {}",
            g.0,
            w.0,
        );
    }
}

/// A cut placed before the walk starts removes exactly the designs spelling the
/// pattern inside the window, and leaves every other design in its place.
#[test]
fn a_cut_removes_exactly_the_designs_that_spell_it() {
    let (params, usage) = setup();
    let pattern = "CCC";
    let window = MotifWindow::new(2, 11); // the coding sequence
    let uncut = walk(&params, &usage, |_, _| {});
    let cut = walk(&params, &usage, |e, rank| {
        if rank == 0 && e.cuts() == 0 {
            e.cut(pattern.into(), window);
        }
    });
    let want: Vec<(f64, String)> = uncut
        .iter()
        .filter(|(_, s)| !spells(s, pattern, window))
        .cloned()
        .collect();
    assert!(
        want.len() < uncut.len(),
        "the cut forbids nothing on this instance, so the case tests nothing",
    );
    assert!(!want.is_empty(), "the cut empties the space");
    assert_same(&cut, &want, "cut before the walk starts");
}

/// The same, cutting partway through. Regions solved before the cut landed hold
/// optima it removes, so a walk applying cuts only to later solves would emit
/// them anyway — which is the failure this case exists to catch.
#[test]
fn a_cut_placed_partway_through_reaches_the_designs_already_solved_for() {
    let (params, usage) = setup();
    let pattern = "CCC";
    let window = MotifWindow::new(2, 11);
    let uncut = walk(&params, &usage, |_, _| {});
    let at = 2;
    let cut = walk(&params, &usage, |e, rank| {
        if rank == at && e.cuts() == 0 {
            e.cut(pattern.into(), window);
        }
    });
    let mut want: Vec<(f64, String)> = uncut[..at].to_vec();
    want.extend(
        uncut[at..]
            .iter()
            .filter(|(_, s)| !spells(s, pattern, window))
            .cloned(),
    );
    assert!(
        want.len() < uncut.len(),
        "nothing after rank {at} spells the pattern, so the case tests nothing",
    );
    assert_same(&cut, &want, "cut partway through");
}

/// A cut whose pattern cannot fit inside its window forbids nothing, and the
/// walk is the uncut walk design for design.
///
/// The floor that matters: a cut routes every solve through the AC-threaded
/// constrained folder instead of the plain scan, so this is the case that says
/// the two return the same designs in the same order when nothing is forbidden.
#[test]
fn a_cut_that_cannot_occur_leaves_the_walk_unchanged() {
    let (params, usage) = setup();
    let uncut = walk(&params, &usage, |_, _| {});
    let cut = walk(&params, &usage, |e, rank| {
        if rank == 0 && e.cuts() == 0 {
            // Three nucleotides forbidden over a two-wide window: no occurrence
            // lies entirely inside it, so the automaton is built and rejects
            // nothing.
            e.cut("CCC".into(), MotifWindow::new(4, 6));
        }
    });
    assert_same(&cut, &uncut, "a cut that cannot occur");
}
