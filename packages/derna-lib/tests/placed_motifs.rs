//! Several forbidden patterns, each holding at **its own place**.
//!
//! One window per automaton is enough for a set that means the same thing
//! everywhere. It is not enough for a loop that cuts a counterexample where the
//! counterexample occurred: a second cut would inherit the first's scope, so
//! either every cut becomes global — a two-nucleotide motif banned across a
//! whole coding sequence — or the cuts stop being positional.
//!
//! A placed set is stepped at every position and scoped only at the accept, per
//! pattern. That is what these tests pin: the same pattern is rejected inside
//! its window and permitted outside it, **while a different pattern with a
//! different window is doing the same thing on the same sequence**. A test with
//! one placed pattern would pass on an automaton that ignored placements and
//! used the shared window instead.

use derna_lib::ac::{Ac, MotifWindow, ScopedAc};
use rna_mfe::Nucleotide;

/// Read `seq` through a scoped automaton, returning the position of the first
/// rejected step, or `None` when the whole sequence survives.
fn first_rejection(ac: &Ac, seq: &str) -> Option<u32> {
    let scoped = ScopedAc::new(ac, MotifWindow::all());
    let mut state = scoped.start();
    for (pos, byte) in seq.bytes().enumerate() {
        let nuc = Nucleotide::from_byte(byte).expect("ACGU");
        match scoped.step_at(pos as u32, state, nuc) {
            Some(next) => state = next,
            None => return Some(pos as u32),
        }
    }
    None
}

/// Each pattern is forbidden inside its own window and permitted outside it,
/// with two patterns placed at different ranges of one sequence.
#[test]
fn a_placed_pattern_is_forbidden_only_inside_its_own_window() {
    //           0123456789...
    let seq = "AACCAAGGAACC";
    // `CC` at [2, 4) and `GG` at [6, 8).
    let ac = Ac::build_placed(&[
        ("CC", MotifWindow::new(2, 4)),
        ("GG", MotifWindow::new(6, 8)),
    ])
    .expect("placed patterns build");

    // The first `CC` lies inside its window and is rejected on its last
    // nucleotide.
    assert_eq!(first_rejection(&ac, seq), Some(3));

    // The same pattern, outside its window, survives — and so does the other
    // one, whose window this sequence does not put it in.
    assert_eq!(first_rejection(&ac, "AAAAAACCAAAA"), None);
    assert_eq!(first_rejection(&ac, "AAGGAAAAAAAA"), None);

    // `GG` inside *its* window is rejected, which is the half a shared window
    // could not express alongside the first.
    assert_eq!(first_rejection(&ac, "AAAAAAGGAAAA"), Some(7));
}

/// The placements are per pattern, not per automaton: the two windows above
/// cannot be replaced by any single one.
///
/// Asserted by exhibiting a sequence each window accepts and the other rejects.
/// Without this the two-pattern test could pass on an automaton that used the
/// union of the windows, or the first.
#[test]
fn no_single_window_reproduces_a_placed_set() {
    let ac = Ac::build_placed(&[
        ("CC", MotifWindow::new(2, 4)),
        ("GG", MotifWindow::new(6, 8)),
    ])
    .expect("placed patterns build");

    // `CC` at [2,4) rejected, `GG` at [2,4) permitted.
    assert_eq!(first_rejection(&ac, "AACCAAAAAAAA"), Some(3));
    assert_eq!(first_rejection(&ac, "AAGGAAAAAAAA"), None);
    // ...and the mirror at [6,8), which is what makes one window impossible.
    assert_eq!(first_rejection(&ac, "AAAAAAGGAAAA"), Some(7));
    assert_eq!(first_rejection(&ac, "AAAAAACCAAAA"), None);
}

/// An occurrence is forbidden only where it lies **entirely** inside its
/// window — the same reading a shared window has, applied per
/// pattern.
///
/// The straddling cases are the ones that separate this from "the last
/// nucleotide falls inside", the coherent alternative reading, which would
/// leave part of a source flank inside the constraint it generated.
#[test]
fn a_placed_occurrence_must_lie_entirely_inside_its_window() {
    let ac = Ac::build_placed(&[("CCC", MotifWindow::new(4, 8))]).expect("builds");

    // Entirely inside [4, 8): positions 4..7 and 5..8.
    assert_eq!(first_rejection(&ac, "AAAACCCAAAAA"), Some(6));
    assert_eq!(first_rejection(&ac, "AAAAACCCAAAA"), Some(7));
    // Straddling the left edge (3..6) and the right edge (6..9): permitted.
    assert_eq!(first_rejection(&ac, "AAACCCAAAAAA"), None);
    assert_eq!(first_rejection(&ac, "AAAAAACCCAAA"), None);
}

/// A pattern placed at two windows is forbidden at both, and nowhere else.
///
/// This is the shape the cut loop produces when the same junction is cut twice
/// at different positions, and it is the case a per-state single placement would
/// silently lose.
#[test]
fn one_pattern_can_hold_at_more_than_one_place() {
    let ac = Ac::build_placed(&[
        ("CC", MotifWindow::new(0, 2)),
        ("CC", MotifWindow::new(6, 8)),
    ])
    .expect("builds");
    assert_eq!(first_rejection(&ac, "CCAAAAAAAAAA"), Some(1));
    assert_eq!(first_rejection(&ac, "AAAAAACCAAAA"), Some(7));
    assert_eq!(first_rejection(&ac, "AAACCAAAAAAA"), None);
}

/// An automaton built the ordinary way carries no placements, so the shared
/// window still governs it — the two enforcement rules do not bleed into each
/// other.
#[test]
fn an_unplaced_automaton_keeps_the_shared_window() {
    let ac = Ac::build(&["CC"]).expect("builds");
    assert!(!ac.is_placed());
    let scoped = ScopedAc::new(&ac, MotifWindow::new(4, 8));
    let step =
        |pos: u32, q: u32, b: u8| scoped.step_at(pos, q, Nucleotide::from_byte(b).expect("ACGU"));
    // Inside the shared window: rejected. Outside: the state is held.
    let mut q = scoped.start();
    q = step(4, q, b'C').expect("first C survives");
    assert!(step(5, q, b'C').is_none(), "CC inside the window survived");
    let outside = step(0, scoped.start(), b'C').expect("outside the window");
    assert!(
        step(1, outside, b'C').is_some(),
        "CC outside the window was rejected",
    );
}
