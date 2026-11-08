//! A cap on homopolymer run length reduces to four forbidden motifs, and the
//! reduction is proven rather than argued.
//!
//! The claim: *no run of one nucleotide longer than `k`* is the same constraint
//! as *none of `A^(k+1)`, `C^(k+1)`, `G^(k+1)`, `U^(k+1)` occurs*. The argument
//! is that a run of length `m > k` contains `m - k` occurrences of the
//! `(k + 1)`-run of its own letter, and conversely an occurrence of a
//! `(k + 1)`-run **is** a run of at least `k + 1`. Neither direction mentions
//! codon boundaries, so the cap needs no lattice machinery of its own: it is
//! the exact motif exclusion that already ships, under four patterns.
//!
//! Three things have to be true for that to be usable, and each has a test
//! here.
//!
//! 1. **The two predicates select the same encodings.** The reference is an
//!    independent run-length scan ([`longest_run`]) over every encoding of a
//!    small protein — never the automaton, which is the thing under test. A
//!    reference that read the automaton would share any defect in it.
//! 2. **The equivalence survives the fold.** The optimum the constrained
//!    folder returns under the four patterns must be the brute-force optimum
//!    over the run-length-capped encodings, byte-exact. This is the same shape
//!    as `motif_blacklist_fuzz`'s oracle, with its feasible set defined by the
//!    cap instead of by the automaton.
//! 3. **The comparison can fail.** A cap of `k` compared against the four
//!    `(k + 2)`-runs must *disagree*, or neither of the first two tests had any
//!    discriminating power.
//!
//! The fourth test is about a window: `derna` forbids every occurrence
//! **overlapping** the enforced span, so a narrowed span is not a narrowed cap,
//! and the test measures the difference rather than reasoning about it.

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::ac::{Ac, ScopedAc};
use derna_lib::dfa;
use derna_lib::{
    design, design_with_motifs, fold_single_path, CodonUsage, MotifWindow, Nucleotide, Params,
    ScanConfig,
};

/// λ values spanning the front: CAI-dominated, balanced, MFE-dominated. The
/// same three `motif_blacklist_fuzz` sweeps.
const LAMBDAS: &[f64] = &[0.0001, 0.5, 0.99];

/// The longest run of one nucleotide in `seq`.
///
/// The reference the reduction is checked against, and deliberately not an
/// automaton: it counts consecutive equal letters and knows nothing about
/// patterns, windows or codons. Mirrors the `max_homopolymer` metric the CLI
/// reports, which is the same quantity read off an emitted design.
fn longest_run(seq: &[Nucleotide]) -> usize {
    let mut best = 0;
    let mut run = 0;
    let mut last = None;
    for &x in seq {
        run = if last == Some(x as u8) { run + 1 } else { 1 };
        last = Some(x as u8);
        best = best.max(run);
    }
    best
}

/// The four patterns a cap of `k` forbids: each nucleotide repeated `k + 1`
/// times.
///
/// Spelled here rather than imported, so the library test and the `derna` flag
/// that expands to the same four are independent statements of the reduction.
fn cap_patterns(k: usize) -> Vec<String> {
    ['A', 'C', 'G', 'U']
        .iter()
        .map(|c| c.to_string().repeat(k + 1))
        .collect()
}

/// Whether running `ac` over `seq` never enters an accepting state.
fn is_motif_free(ac: &Ac, seq: &[Nucleotide]) -> bool {
    let mut q = ac.start();
    for &x in seq {
        q = ac.step(q, x);
        if ac.is_accept(q) {
            return false;
        }
    }
    true
}

/// `seq` as an `A/C/G/U` string.
fn spell(seq: &[Nucleotide]) -> String {
    seq.iter()
        .map(|n| match n {
            Nucleotide::A => 'A',
            Nucleotide::C => 'C',
            Nucleotide::G => 'G',
            Nucleotide::U => 'U',
        })
        .collect()
}

/// Every encoding of `protein`, each with its CAI log-score.
fn encodings(usage: &CodonUsage, protein: &str) -> Vec<(f64, Vec<Nucleotide>)> {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = dfa::build(&aa, usage);
    graph.paths_between(graph.start(), graph.end())
}

/// The proteins the set comparison runs over. Short, with enough one- and
/// two-letter-run residues (`K` = `AAA`/`AAG`, `P` = `CCN`, `G` = `GGN`) that
/// long runs are reachable and the cap genuinely bites.
const PROTEINS: &[&str] = &["KK", "MKKW", "PPG", "GGK", "MKPGW", "KFKF"];

#[test]
fn the_four_runs_admit_exactly_the_encodings_whose_runs_fit_the_cap() {
    let usage = CodonUsage::load_default().unwrap();
    let (mut admitted, mut refused) = (0usize, 0usize);
    for protein in PROTEINS {
        for k in 2..=5usize {
            let patterns = cap_patterns(k);
            let ac = Ac::build(&patterns).unwrap();
            for (_cai, seq) in encodings(&usage, protein) {
                let capped = longest_run(&seq) <= k;
                let motif_free = is_motif_free(&ac, &seq);
                assert_eq!(
                    capped,
                    motif_free,
                    "cap k={k} and {patterns:?} disagree on {} (longest run {})",
                    spell(&seq),
                    longest_run(&seq),
                );
                if capped {
                    admitted += 1;
                } else {
                    refused += 1;
                }
            }
        }
    }
    // Both classes have to be non-empty, or the agreement above is agreement
    // about nothing: a sweep over instances no cap binds on would pass with the
    // automaton wired to accept everything.
    assert!(refused > 0, "no encoding was refused by any cap");
    assert!(admitted > 0, "no encoding was admitted by any cap");
}

#[test]
fn a_run_spanning_a_codon_boundary_is_refused_and_one_inside_a_codon_is_not() {
    // `K` spells `AAA` or `AAG`, so at a cap of 4 no single codon violates the
    // cap on its own — `AAA` is a run of three. Every violation on `KK` is
    // therefore a run assembled across the codon boundary, which is the case a
    // per-codon filter would miss and the case the claim rests on.
    let usage = CodonUsage::load_default().unwrap();
    let k = 4;
    let ac = Ac::build(&cap_patterns(k)).unwrap();
    let (mut admitted, mut refused) = (Vec::new(), Vec::new());
    for (_cai, seq) in encodings(&usage, "KK") {
        assert!(
            longest_run(&seq[..3]) <= k && longest_run(&seq[3..]) <= k,
            "a single codon already violates the cap: {}",
            spell(&seq),
        );
        if is_motif_free(&ac, &seq) {
            admitted.push(spell(&seq));
        } else {
            refused.push(spell(&seq));
        }
    }
    admitted.sort();
    refused.sort();
    assert_eq!(admitted, ["AAGAAA", "AAGAAG"]);
    assert_eq!(refused, ["AAAAAA", "AAAAAG"]);
}

/// The proteins the fold-level oracle runs over, and why these.
///
/// `P`'s most-adapted codon is `CCC`, so an unconstrained design of `PP` or
/// `GPP` spells a run of 6 or 7 and a cap of 3 removes that design from the
/// feasible set — which is what makes the equality below a statement about the
/// constraint. The first pick for this sweep was a set on which the cap
/// forbade nothing at any weight, and only the binding count caught it: the
/// equality passed on all of it, measuring an unconstrained run against
/// itself.
const FOLDED: &[&str] = &["PP", "GPP", "PK", "KK", "GGK", "MKPGW"];

#[test]
fn the_capped_optimum_is_the_brute_force_optimum_over_capped_encodings() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let k = 3;
    let patterns = cap_patterns(k);
    let mut bound = 0usize;
    let mut pp_bound = 0usize;
    for protein in FOLDED {
        // The feasible set is defined by the run-length reference alone; the
        // folder is given the four patterns and has to land on the same
        // optimum.
        let capped: Vec<(f64, Vec<Nucleotide>)> = encodings(&usage, protein)
            .into_iter()
            .filter(|(_, seq)| longest_run(seq) <= k)
            .collect();
        assert!(
            !capped.is_empty(),
            "{protein}: no encoding fits a cap of {k}"
        );
        for &lambda in LAMBDAS {
            let want = capped
                .iter()
                .map(|(cai, seq)| {
                    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
                    let codons: Vec<[u8; 3]> = seq
                        .chunks_exact(3)
                        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
                        .collect();
                    let mfe = fold_single_path(&params, &aa, &codons).mfe_kcal_mol * 100.0;
                    lambda * mfe - (1.0 - lambda) * cai * 100.0
                })
                .fold(f64::INFINITY, f64::min);
            let got = design_with_motifs(
                &params,
                protein,
                &usage,
                lambda,
                &patterns,
                ScanConfig::default(),
            )
            .unwrap_or_else(|e| panic!("{protein} λ={lambda}: {e}"));
            let got_brd =
                lambda * (got.mfe_kcal_mol * 100.0) - (1.0 - lambda) * (got.cai_log * 100.0);
            assert!(
                (want - got_brd).abs() < 1e-6,
                "{protein} λ={lambda}: brute={want} folder={got_brd}",
            );
            assert!(
                longest_run(
                    &got.sequence
                        .bytes()
                        .map(|c| Nucleotide::from_byte(c).unwrap())
                        .collect::<Vec<_>>()
                ) <= k,
                "{protein} λ={lambda}: the emitted design breaks the cap: {}",
                got.sequence,
            );
            // Whether the cap bound at all: the unconstrained optimum breaking
            // it is what makes the equality above a statement about the
            // constraint rather than about an unconstrained run.
            let free = design(&params, protein, &usage, lambda).unwrap();
            let free_run = longest_run(
                &free
                    .sequence
                    .bytes()
                    .map(|c| Nucleotide::from_byte(c).unwrap())
                    .collect::<Vec<_>>(),
            );
            if free_run > k {
                bound += 1;
                if *protein == "PP" {
                    pp_bound += 1;
                }
            }
        }
    }
    assert!(
        bound > 0,
        "the cap bound on no case: every unconstrained optimum already fitted it, \
         so the comparison priced nothing",
    );
    // A named case rather than a count alone: a count can stay positive while
    // the one instance the sweep was built around stops binding, and the
    // instance is the thing the reasoning above rests on.
    assert_eq!(
        pp_bound,
        LAMBDAS.len(),
        "the cap stopped binding on PP, whose unconstrained design is the run of \
         6 this sweep is built around",
    );
}

#[test]
fn the_equivalence_is_specific_to_the_cap_it_is_stated_for() {
    // The mirror of the first test: were the comparison blind — an automaton
    // that accepted everything, a reference that read the automaton — it would
    // pass against the *wrong* pattern set too. A cap of `k` against the four
    // `(k + 2)`-runs must disagree on some encoding, namely one whose longest
    // run is exactly `k + 1`.
    let usage = CodonUsage::load_default().unwrap();
    let k = 3;
    let wrong = Ac::build(&cap_patterns(k + 1)).unwrap();
    let mut disagreements = 0usize;
    for protein in PROTEINS {
        for (_cai, seq) in encodings(&usage, protein) {
            if (longest_run(&seq) <= k) != is_motif_free(&wrong, &seq) {
                disagreements += 1;
            }
        }
    }
    assert!(
        disagreements > 0,
        "a cap of {k} and the {}-runs agreed everywhere, so the comparison \
         cannot distinguish a correct pattern set from a wrong one",
        k + 2,
    );
}

#[test]
fn a_narrowed_window_is_not_a_narrowed_cap() {
    // `derna` forbids every occurrence *overlapping* the enforced span, so
    // narrowing the span with `--motif-window` does not narrow the cap to that
    // span: a run that straddles the span's edge is still
    // forbidden, even where the part inside the span fits the cap. The
    // difference is measured here on one sequence rather than argued, because
    // it is the reason `--max-homopolymer` refuses `--motif-window` instead of
    // quietly meaning one of the two.
    let k = 3;
    let patterns = cap_patterns(k);
    let seq: Vec<Nucleotide> = "GGAAAAGG"
        .bytes()
        .map(|c| Nucleotide::from_byte(c).unwrap())
        .collect();
    // The whole molecule breaks the cap: a run of four A at positions 2..6.
    assert_eq!(longest_run(&seq), 4);
    // Positions 5..8 hold `AGG` — longest run 2, inside the cap. A windowed
    // *cap* would admit this molecule; the overlapping window forbids it,
    // because the `AAAA` occurrence at 2..6 touches position 5.
    assert_eq!(longest_run(&seq[5..]), 2);
    let window = MotifWindow::overlapping(5, 8);
    let placed: Vec<(&str, MotifWindow)> = patterns.iter().map(|p| (p.as_str(), window)).collect();
    let ac = Ac::build_placed(&placed).unwrap();
    assert!(
        !ScopedAc::new(&ac, window).is_clean(&seq),
        "an overlapping window admitted a run that only straddles it",
    );
    // The library's other reading is the complement of the finding: the
    // entirely-inside window does admit it, so the two readings disagree on
    // exactly this molecule and a caller cannot be told that a window narrows
    // the cap.
    let inside = MotifWindow::new(5, 8);
    let ac_inside = Ac::for_window(&patterns, inside).unwrap();
    assert!(
        ScopedAc::new(&ac_inside, inside).is_clean(&seq),
        "the entirely-inside reading should admit a straddling run",
    );
}

#[test]
fn four_run_patterns_cost_four_states_per_run_length() {
    // What the reduction costs the lattice: the AC node multiplier is the
    // automaton's state count, and four patterns of length `k + 1` share only
    // the root, so it is `4(k + 1) + 1` and grows linearly in the cap. A cap
    // of 9 — a typical synthesis limit — is 41 states.
    for k in 1..=12usize {
        let ac = Ac::build(&cap_patterns(k)).unwrap();
        assert_eq!(ac.num_states(), 4 * (k + 1) + 1, "cap k={k}");
    }
}
