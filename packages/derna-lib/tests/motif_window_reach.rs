//! A window's two reaches, and the one occurrence they disagree about.
//!
//! The entirely-inside reading and the overlapping one agree on every
//! occurrence lying wholly inside the window and on every occurrence lying
//! wholly outside it. They differ on exactly one case — an occurrence
//! straddling a boundary — so that case is what these tests pin, from both
//! sides and at both edges. A test that only checked an interior occurrence
//! would pass against either reading.
//!
//! The left straddler is the one that needs the mechanism. An automaton held
//! at its root until the window starts has no memory of the positions before
//! it, so it cannot see an occurrence that begins there; the overlapping reach
//! is therefore enforced through the placed path, which steps at every
//! position and scopes the accept test per pattern. The last test here is what
//! would fail if the reach were changed without the mechanism.

use derna_lib::ac::{Ac, MotifWindow, ScopedAc, WindowReach};
use rna_mfe::Nucleotide;

/// Whether `seq` survives `window`'s enforcement of the single motif `motif`.
///
/// An overlapping window is built placed, which is how such a window is
/// enforced; an inside one is built either way and is built unplaced here, the
/// shape every caller of a shared window uses.
fn spellable(motif: &str, window: MotifWindow, seq: &str) -> bool {
    let ac = match window.reach() {
        WindowReach::Inside => Ac::build(&[motif]).expect("valid motif"),
        WindowReach::Overlapping => Ac::build_placed(&[(motif, window)]).expect("valid motif"),
    };
    let scoped = ScopedAc::new(&ac, window);
    let nucleotides: Vec<Nucleotide> = seq
        .bytes()
        .map(|b| Nucleotide::from_byte(b).expect("ACGU"))
        .collect();
    scoped.is_clean(&nucleotides)
}

/// `GGG` placed so that each case is one occurrence at a known offset.
/// Positions:            0123456789
const SEQ_INTERIOR: &str = "AAAGGGAAAA";
/// The occurrence spans positions 2..5, so a window starting at 3 has it
/// straddling the left edge.
const SEQ_LEFT: &str = "AAGGGAAAAA";
/// The occurrence spans 6..9, straddling the right edge of a window ending
/// at 7.
const SEQ_RIGHT: &str = "AAAAAAGGGA";

#[test]
fn the_two_reaches_agree_on_an_occurrence_wholly_inside_the_window() {
    let inside = MotifWindow::new(3, 7);
    let overlapping = MotifWindow::overlapping(3, 7);
    assert!(
        !spellable("GGG", inside, SEQ_INTERIOR),
        "inside: 3..6 is in 3..7"
    );
    assert!(
        !spellable("GGG", overlapping, SEQ_INTERIOR),
        "overlapping: 3..6 is in 3..7"
    );
}

#[test]
fn the_two_reaches_agree_on_an_occurrence_wholly_outside_the_window() {
    // The window sits past the occurrence entirely.
    let inside = MotifWindow::new(7, 10);
    let overlapping = MotifWindow::overlapping(7, 10);
    assert!(
        spellable("GGG", inside, SEQ_INTERIOR),
        "inside: 3..6 misses 7..10"
    );
    assert!(
        spellable("GGG", overlapping, SEQ_INTERIOR),
        "overlapping: 3..6 misses 7..10"
    );
}

#[test]
fn only_the_overlapping_reach_forbids_a_left_straddler() {
    // The occurrence spans 2..5; the window starts at 3, so one nucleotide of
    // the occurrence is outside it and two are inside.
    let inside = MotifWindow::new(3, 10);
    let overlapping = MotifWindow::overlapping(3, 10);
    assert!(
        spellable("GGG", inside, SEQ_LEFT),
        "the inside reading permits a straddler: the occurrence is not wholly within 3..10"
    );
    assert!(
        !spellable("GGG", overlapping, SEQ_LEFT),
        "the overlapping reading forbids it: positions 3 and 4 are in the window"
    );
}

#[test]
fn only_the_overlapping_reach_forbids_a_right_straddler() {
    // The occurrence spans 6..9; the window ends at 7.
    let inside = MotifWindow::new(0, 7);
    let overlapping = MotifWindow::overlapping(0, 7);
    assert!(
        spellable("GGG", inside, SEQ_RIGHT),
        "the inside reading permits a straddler at the right edge"
    );
    assert!(
        !spellable("GGG", overlapping, SEQ_RIGHT),
        "the overlapping reading forbids it: position 6 is in the window"
    );
}

#[test]
fn a_whole_molecule_window_is_the_same_constraint_under_either_reach() {
    // With nothing outside the window there is no straddler to disagree about,
    // so a caller who widens a window to the whole molecule cannot see which
    // reach it asked for. This is the floor the CLI's no-flank case rests on.
    for seq in [SEQ_INTERIOR, SEQ_LEFT, SEQ_RIGHT, "AAAAAAAAAA"] {
        let whole = MotifWindow::overlapping(0, seq.len() as u32);
        assert_eq!(
            spellable("GGG", MotifWindow::all(), seq),
            spellable("GGG", whole, seq),
            "{seq}"
        );
    }
}

#[test]
fn the_reach_is_part_of_the_windows_identity() {
    // Two windows over the same range are not the same constraint, so a
    // comparison must separate them — a header recording only `lo` and `hi`
    // would describe both.
    assert_ne!(MotifWindow::new(3, 7), MotifWindow::overlapping(3, 7));
    assert_eq!(MotifWindow::new(3, 7).reach(), WindowReach::Inside);
    assert_eq!(
        MotifWindow::overlapping(3, 7).reach(),
        WindowReach::Overlapping
    );
    assert_eq!(MotifWindow::all().reach(), WindowReach::Inside);
}
