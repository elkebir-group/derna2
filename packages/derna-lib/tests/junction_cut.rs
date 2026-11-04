//! Does cutting a **junction** discard less than cutting a codon?
//!
//! The cut loop restricts its search each round by excluding what the last
//! counterexample used. The codon cut removes one synonym at one residue
//! outright. The junction cut forbids the sequence the design spells across a
//! codon boundary, at that place only — so it removes *combinations* of adjacent
//! codons and every codon involved stays available in every other context.
//!
//! **This is the control a null result needs.** A measurement found the same
//! false-negative rate for both rules, and a rate that does not move is worth
//! exactly what the evidence that the instrument was on is worth. So the
//! difference is asserted here on the lattice, where it is decidable: after a
//! junction cut the codon the design used is still spellable at that residue,
//! and after the codon cut it is not.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::{Ac, MotifWindow};
use derna_lib::dfa::{self, UtrChain};
use derna_lib::product::ProductDfa;
use derna_lib::CodonUsage;
use rna_mfe::Nucleotide;

/// Proline then leucine. The pair is chosen so the junction is a real choice on
/// **both** sides: every proline codon is `CCN` and every leucine codon starts
/// `U` or `C`, so forbidding one junction spelling leaves the proline free to
/// keep its codon and the leucine free to start the other way.
///
/// A pair whose second residue has one starting letter would not show this — the
/// junction would be forced, the codon would go with it, and the cut would look
/// exactly like a codon cut on that instance while being a different mechanism.
const PROTEIN: &str = "MPLP";

fn residues() -> Vec<usize> {
    PROTEIN
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}

fn nucleotides(s: &str) -> Vec<Nucleotide> {
    s.bytes()
        .map(|b| Nucleotide::from_byte(b).expect("ACGU"))
        .collect()
}

/// Which codons survive at each residue under a placed motif set.
fn surviving(placed: &[(String, MotifWindow)], utr5: &str) -> Vec<Vec<bool>> {
    let usage = CodonUsage::load_default().expect("codon usage");
    let aa = residues();
    let five = nucleotides(utr5);
    let graph =
        dfa::build_with_utr_chain(&aa, &usage, UtrChain::Fixed(&five), UtrChain::Fixed(&[]));
    let ac = Ac::build_placed(placed).expect("placed patterns build");
    ProductDfa::build_scoped(&graph, &ac, MotifWindow::all()).surviving_codons()
}

/// The codons of residue `i`, as the strings the table spells them with.
fn codons_of(i: usize) -> Vec<String> {
    AMINO_ACIDS[residues()[i]]
        .codons
        .iter()
        .map(|c| {
            c.nuc
                .iter()
                .map(|&n| "ACGU".as_bytes()[n as usize] as char)
                .collect()
        })
        .collect()
}

/// **A junction cut leaves both codons available; a codon cut removes one.**
///
/// The junction is the boundary between residue 1 (proline) and residue 2
/// (leucine). Forbidding one spelling across it removes the *pair*: proline
/// keeps `CCG` as long as the leucine after it does not start `U`, and leucine
/// keeps `UUA`/`UUG` as long as the proline before it does not end `G`.
///
/// Both halves are asserted. Only the neighbour surviving would also hold for a
/// cut that removed the codon outright and nothing else, and only the codon
/// surviving would hold for a cut that did nothing at all.
#[test]
fn a_junction_cut_keeps_both_codons_a_codon_cut_would_remove_one() {
    let utr5 = "AAA";
    let cds_start = utr5.len();
    // Residue 1 occupies [cds_start + 3, +6); the junction with residue 2 is at
    // cds_start + 6, so the crossing window is its last nucleotide and residue
    // 2's first.
    let lo = cds_start + 5;
    let window = MotifWindow::new(lo as u32, (lo + 2) as u32);
    // `GU`: proline ending `G` (`CCG`) followed by leucine starting `U`
    // (`UUA`/`UUG`). Neither residue is forced into it — that is the point.
    let placed = vec![("GU".to_string(), window)];

    let alive = surviving(&placed, utr5);
    let kept = |i: usize| -> Vec<String> {
        codons_of(i)
            .into_iter()
            .zip(&alive[i])
            .filter(|(_, keep)| **keep)
            .map(|(name, _)| name)
            .collect()
    };

    // The codon the cut was aimed at is still spellable, which is exactly what
    // a codon cut at this residue would have taken away.
    assert!(
        kept(1).contains(&"CCG".to_string()),
        "CCG went at residue 1, so the cut removed a codon rather than a pair: {:?}",
        kept(1),
    );
    // And so is the neighbour's, on the other side of the same junction.
    assert!(
        kept(2).contains(&"UUA".to_string()),
        "UUA went at residue 2, so the cut removed a codon rather than a pair: {:?}",
        kept(2),
    );

    // Non-vacuity: the cut is on. With both codons kept, what it removed is the
    // *combination*, which shows up as neither residue losing anything and the
    // pair being unspellable — so the check is that a codon cut would have
    // removed something here and this did not.
    assert_eq!(
        kept(1).len(),
        codons_of(1).len(),
        "the junction cut removed a proline codon outright",
    );
    assert_eq!(
        kept(2).len(),
        codons_of(2).len(),
        "the junction cut removed a leucine codon outright",
    );
}

/// The combination itself is gone, which is what makes the test above a cut
/// rather than a no-op.
///
/// `surviving_codons` is a per-residue reading and cannot see a pair, so the
/// pair is checked where it is visible: a sequence spelling it is rejected by
/// the automaton, and one differing only in the neighbour is not.
#[test]
fn the_combination_the_junction_cut_forbids_is_unspellable() {
    let utr5 = "AAA";
    let lo = utr5.len() + 5;
    let window = MotifWindow::new(lo as u32, (lo + 2) as u32);
    let ac = Ac::build_placed(&[("GU".to_string(), window)]).expect("builds");

    let rejects = |cds: &str| {
        let seq = format!("{utr5}{cds}");
        let scoped = derna_lib::ac::ScopedAc::new(&ac, MotifWindow::all());
        let mut state = scoped.start();
        for (pos, byte) in seq.bytes().enumerate() {
            let nuc = Nucleotide::from_byte(byte).expect("ACGU");
            match scoped.step_at(pos as u32, state, nuc) {
                Some(next) => state = next,
                None => return true,
            }
        }
        false
    };

    // AUG CCG UUA CCU — the forbidden junction, at the place it was cut.
    assert!(rejects("AUGCCGUUACCU"), "the cut combination was spellable");
    // The same proline, a leucine starting the other way: permitted.
    assert!(
        !rejects("AUGCCGCUACCU"),
        "changing the neighbour did not help"
    );
    // The same leucine, a proline ending the other way: permitted.
    assert!(!rejects("AUGCCAUUACCU"), "changing the codon did not help");
}

/// The junction cut is scoped: the same pattern elsewhere in the coding sequence
/// is untouched.
///
/// Without this the cut would be a global motif ban — the thing a shared window
/// forces and the reason placed patterns exist.
#[test]
fn a_junction_cut_holds_at_one_junction_only() {
    let utr5 = "AAA";
    let cds_start = utr5.len();
    let first = cds_start + 5;
    let placed = vec![(
        "GU".to_string(),
        MotifWindow::new(first as u32, (first + 2) as u32),
    )];
    let alive = surviving(&placed, utr5);

    // Residue 3 is proline again, so the same `G`-ending codon exists at a
    // junction the cut was not placed at — and is untouched.
    let names = codons_of(3);
    let kept: Vec<&String> = names
        .iter()
        .zip(&alive[3])
        .filter(|(_, keep)| **keep)
        .map(|(name, _)| name)
        .collect();
    assert!(
        kept.iter().any(|c| *c == "CCG"),
        "CCG went at a residue the cut was not placed at, so the cut is global",
    );
}
