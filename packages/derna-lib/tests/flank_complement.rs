//! Deriving the forbidden set from a molecule's given regions, so its designed
//! regions cannot spell a duplex with them.
//!
//! Two things are under test and they fail differently. **The arithmetic** — the
//! reverse complement, the degenerate expansion, the pairing relation — is what
//! sections 1 and 2 pin, against oracles written from the relation rather than
//! from the patterns. **The entry contract** is section 3: which regions a run's
//! flank arrangement makes sources, which it makes targets, and where that puts
//! the enforcement window. The arithmetic is the same in every arrangement, so a
//! suite that exercised one of them would leave the contract untested while
//! looking complete.
//!
//! The window this set is enforced over is tested in `motif_scoped_window.rs`,
//! deliberately on a hand-written set so that a failure there is unambiguous
//! between the window and the derivation. Here the roles swap: the window is
//! held fixed and known-good, and only the derivation varies.
//!
//! **A design-level test cannot tell a correct derivation from a wrong one.** A
//! reversed reverse complement is the same number of patterns, constrains the
//! same positions, and yields a feasible design; so does a complement that
//! degenerates the designed base instead of the given one. Every such set is
//! self-consistent — the automaton excludes what the set says, and the design is
//! optimal over what is left. What separates them is the *pairing relation*, so
//! that is what the first section enumerates:
//!
//! 1. **The set is the reverse complement**, checked against an independent
//!    re-derivation, with the reversal asserted to be load-bearing on the flank
//!    used (a palindromic flank would make the check vacuous).
//! 2. **The automaton permits exactly the segments that cannot pair.** For small
//!    `k`, every one of the `4^k` designed segments: rejected iff it forms an
//!    ungapped duplex with some window of the given flank, under a pairing
//!    relation written here from the six canonical pairs.
//! 3. **Wobble is symmetric in effect.** The map degenerates the *given* base,
//!    which reads as though it only caught a given `G` opposite a designed `U`.
//!    A designed `G` opposite a given `U` is caught too, and that case is tested
//!    on its own — otherwise half of the wobble relation is untested and the
//!    strict control would pass the same suite.
//!
//! The rest tests what the set *means* for a molecule. The guarantee is narrower
//! than "no helix of length ≥ k" and the difference is asserted rather than
//! glossed: a **bulged** helix of more than `k` total pairs is permitted, on
//! purpose, and a test asserting zero flank-to-flank pairing would be testing a
//! claim the instrument does not make.
//!
//! The pairing relation, the antiparallel index mapping, the structure
//! enumerator, the CAI formula and the run-length scanner are all re-implemented
//! here rather than shared with the crate, so a bug in the derivation cannot
//! hide inside its own oracle.

use std::collections::HashMap;

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::Ac;
use derna_lib::PositionalConstraints;
use derna_lib::{
    derive_from_flanks, derive_over, design_composite, design_with_utr_profile, sequence_gap,
    CdsWindow, CodonUsage, Complement, CompositeSpec, Design, FlankComplement, MotifWindow,
    PairClass, Params, Region, RegionConstraints, ScanConfig, UtrProfile, UtrSpec,
};
use rna_mfe::{evaluate, Nucleotide};

/// One secondary structure as its list of `(i, j)` base pairs.
type Pairs = Vec<(usize, usize)>;
/// Per-interval memo for the structure enumeration.
type Memo = HashMap<(usize, usize), Vec<Pairs>>;

/// BRD tolerance in dekacalories — the rounded-integer vs continuous
/// objective gap, exactly as `tests/brd_parity.rs`.
const BRD_TOL: f64 = 0.5;

/// The λ values the parity checks sweep (low / mid / high).
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

fn params_usage() -> (Params, CodonUsage) {
    (
        Params::load_default().expect("Turner 2004 parameters"),
        CodonUsage::load_default().expect("codon usage"),
    )
}

fn empty() -> UtrSpec {
    UtrSpec::Fixed(String::new())
}

fn nuc_byte(n: usize) -> u8 {
    b"ACGU"[n]
}

fn protein_indices(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}

// ===========================================================================
// The oracle's own pairing relation
// ===========================================================================

/// Whether two nucleotides form a base pair — the six canonical pairs, written
/// out. `wobble` adds the two `G·U` orientations to the four Watson–Crick ones.
///
/// Symmetric in its arguments by construction, which is the property the
/// derivation's one-sided-looking map has to reproduce.
fn pairs(a: u8, b: u8, wobble: bool) -> bool {
    let wc = matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'C') | (b'C', b'G')
    );
    wc || (wobble && matches!((a, b), (b'G', b'U') | (b'U', b'G')))
}

/// Whether a given window `w` and a designed segment `d` of equal length form an
/// **ungapped** duplex.
///
/// Pairing is antiparallel: whichever of the two lies 5′ of the other, the
/// stacked run pairs `w[t]` with `d[len − 1 − t]`. That the same mapping serves
/// both sidednesses is why one derivation covers both, and it is asserted
/// against the substrate rather than assumed
/// (`the_same_set_serves_both_sidednesses`).
fn duplex(w: &[u8], d: &[u8], wobble: bool) -> bool {
    w.len() == d.len() && (0..w.len()).all(|t| pairs(w[t], d[w.len() - 1 - t], wobble))
}

/// Whether `segment` can form an ungapped duplex with **some** `k`-window of
/// `given`.
fn duplexes_with_flank(given: &[u8], segment: &[u8], wobble: bool) -> bool {
    given
        .windows(segment.len())
        .any(|w| duplex(w, segment, wobble))
}

/// Whether `seq` contains, anywhere entirely inside `[lo, hi)`, a `k`-segment
/// that duplexes with the given flank.
///
/// **The oracle's statement of what the derivation forbids**, written from the
/// pairing relation and never from the derived patterns — so a wrong set cannot
/// satisfy it by being consistently wrong.
fn spells_partner_inside(
    seq: &[u8],
    given: &[u8],
    k: usize,
    lo: usize,
    hi: usize,
    wobble: bool,
) -> bool {
    (lo..hi.saturating_sub(k - 1)).any(|s| duplexes_with_flank(given, &seq[s..s + k], wobble))
}

// ===========================================================================
// 1. The derivation is the reverse complement
// ===========================================================================

/// The derived set, independently re-derived here and compared as a set.
///
/// The re-derivation is deliberately literal — reverse the window, map each base
/// to the IUPAC code for what it pairs with — because that is the definition the
/// shipped derivation is supposed to implement, and the two agreeing is the only
/// evidence that it does. The check would be vacuous on a flank whose reverse
/// complement equals its forward complement, so the last assertion pins that it
/// is not.
#[test]
fn the_derived_set_is_the_reverse_complement_of_every_window() {
    let partner = |b: u8, wobble: bool| match (b, wobble) {
        (b'A', _) => b'U',
        (b'C', _) => b'G',
        (b'G', false) => b'C',
        (b'G', true) => b'Y',
        (b'U', false) => b'A',
        (b'U', true) => b'R',
        _ => unreachable!("A/C/G/U"),
    };
    // Every flank here is asymmetric under reversal at every `k` swept, which is
    // what makes the last assertion in the loop meaningful. A periodic flank such
    // as `GGGAGGGAGGGA` has a reversal-symmetric window set and would pass a
    // forward complement equally well.
    for flank in ["ACGUGGCAUGCA", "AAACCCUUUGGG", "AUUUCUUCUGACACAACUGUG"] {
        for k in 2..=6usize {
            for (wobble, complement) in [(true, Complement::Wobble), (false, Complement::Strict)] {
                let mut want: Vec<String> = flank
                    .as_bytes()
                    .windows(k)
                    .map(|w| {
                        w.iter()
                            .rev()
                            .map(|&b| partner(b, wobble) as char)
                            .collect()
                    })
                    .collect();
                want.sort();
                want.dedup();

                let got = derive(flank, 9, FlankComplement { k, complement });
                assert_eq!(
                    got.motifs,
                    want,
                    "the derived set is not the reverse complement [{flank} k={k} \
                     {}]",
                    complement.label(),
                );
                assert_eq!(got.windows, flank.len() - k + 1);
                assert_eq!(got.sources, [Region::Utr5]);

                // Without this the check passes on a flank where reversing
                // changes nothing, which is the case it exists to rule out.
                let forward: Vec<String> = flank
                    .as_bytes()
                    .windows(k)
                    .map(|w| w.iter().map(|&b| partner(b, wobble) as char).collect())
                    .collect();
                let mut forward = forward;
                forward.sort();
                forward.dedup();
                assert_ne!(
                    got.motifs, forward,
                    "reversing the window changes nothing on this flank, so it cannot \
                     tell a reverse complement from a forward one [{flank} k={k}]",
                );
            }
        }
    }
}

/// **The check no design-level test can make.** For every one of the `4^k`
/// segments a designed flank could spell, the automaton must reject it exactly
/// when it forms an ungapped duplex with some window of the given flank.
///
/// A reversed reverse complement, or one that degenerates the designed base
/// rather than the given one, produces a set that is the same size and equally
/// self-consistent; both fail here and neither fails anywhere else.
#[test]
fn the_automaton_permits_exactly_the_segments_that_cannot_pair() {
    for flank in ["ACGUGGCAUGCA", "GGGAGGGAGGGA", "AAACCCUUUGGG"] {
        for k in 2..=5usize {
            for (wobble, complement) in [(true, Complement::Wobble), (false, Complement::Strict)] {
                let derived = derive(flank, 9, FlankComplement { k, complement });
                let ac = Ac::build(&derived.motifs).expect("valid derived patterns");
                let (mut rejected, mut permitted) = (0usize, 0usize);
                for segment in every_segment(k) {
                    // Feeding exactly `k` nucleotides from the root: since every
                    // derived pattern has length `k`, an accepting state here
                    // means this whole segment is one of them.
                    let mut q = ac.start();
                    let mut hit = false;
                    for &b in &segment {
                        q = ac.step(q, Nucleotide::from_byte(b).expect("A/C/G/U"));
                        hit |= ac.is_accept(q);
                    }
                    let want = duplexes_with_flank(flank.as_bytes(), &segment, wobble);
                    assert_eq!(
                        hit,
                        want,
                        "automaton and pairing relation disagree on {} [{flank} k={k} {}]",
                        String::from_utf8_lossy(&segment),
                        complement.label(),
                    );
                    if want {
                        rejected += 1;
                    } else {
                        permitted += 1;
                    }
                }
                // Both counts non-zero, or the agreement above is agreement on a
                // constant.
                assert!(
                    rejected > 0 && permitted > 0,
                    "the derived set is trivial ({rejected} rejected, {permitted} \
                     permitted) [{flank} k={k} {}]",
                    complement.label(),
                );
            }
        }
    }
}

/// Wobble is not one-directional, and the half that looks like it might be
/// missing is tested on its own.
///
/// The map degenerates the **given** base (`G → Y`, `U → R`), which reads as
/// catching only a given `G` opposite a designed `U`. The mirror — a designed
/// `G` opposite a given `U` — is caught through `U → R = {A, G}`. If it were
/// not, half of the `G·U` relation would be unenforced and the strict control
/// would be indistinguishable from the default on exactly those cases.
#[test]
fn wobble_catches_a_designed_g_opposite_a_given_u() {
    let k = 4;
    // Every base of the given flank is `U`, so the only `G·U` pairs available
    // have the **designed** `G` opposite a **given** `U`. Under strict the flank
    // pairs only `A`.
    let derived = derive(
        "UUUUUUUU",
        8,
        FlankComplement {
            k,
            complement: Complement::Wobble,
        },
    );
    let strict = derive(
        "UUUUUUUU",
        8,
        FlankComplement {
            k,
            complement: Complement::Strict,
        },
    );
    assert_eq!(derived.motifs, vec!["RRRR".to_string()]);
    assert_eq!(strict.motifs, vec!["AAAA".to_string()]);

    let wobble_ac = Ac::build(&derived.motifs).expect("valid");
    let strict_ac = Ac::build(&strict.motifs).expect("valid");
    // A designed segment of four `G`s duplexes with `UUUU` on the wobble
    // relation alone.
    assert!(
        rejects(&wobble_ac, "GGGG"),
        "a designed G opposite a given U was permitted under wobble, so half of \
         the G·U relation is unenforced",
    );
    assert!(
        !rejects(&strict_ac, "GGGG"),
        "the strict control forbade a G·U duplex, so it is not the control it is \
         reported as",
    );
    // And the other orientation, on a flank of `G`s: a designed `U` opposite a
    // given `G`, which is the direction the map reads as covering.
    let given_g = derive(
        "GGGGGGGG",
        8,
        FlankComplement {
            k,
            complement: Complement::Wobble,
        },
    );
    assert_eq!(given_g.motifs, vec!["YYYY".to_string()]);
    assert!(rejects(&Ac::build(&given_g.motifs).expect("valid"), "UUUU"));
    // The pairing relation the oracle states agrees with both readings.
    assert!(duplex(b"UUUU", b"GGGG", true));
    assert!(duplex(b"GGGG", b"UUUU", true));
    assert!(!duplex(b"UUUU", b"GGGG", false));
}

/// The same derived set serves both sidednesses, because the antiparallel index
/// mapping is the same whichever flank lies 5′ of the other.
///
/// Asserted rather than argued: a derivation that quietly depended on the given
/// flank being the 5′ one would be wrong on the mirror and right on the case
/// everything else here uses.
#[test]
fn the_same_set_serves_both_sidednesses() {
    let flank = "ACGUGGCAUGCA";
    let fc = FlankComplement::wobble(4);
    let given5 = derive_from_flanks(fc, &UtrSpec::Fixed(flank.into()), &UtrSpec::Designed(9), 3)
        .expect("given 5′, designed 3′");
    let given3 = derive_from_flanks(fc, &UtrSpec::Designed(9), &UtrSpec::Fixed(flank.into()), 3)
        .expect("designed 5′, given 3′");
    assert_eq!(given5.motifs, given3.motifs);
    assert_eq!(given5.sources, [Region::Utr5]);
    assert_eq!(given3.sources, [Region::Utr3]);
    // Each window covers the regions the designer fills — the coding sequence
    // and the designed flank — and never the source.
    assert_eq!(given5.targets, [Region::Cds, Region::Utr3]);
    assert_eq!(given3.targets, [Region::Utr5, Region::Cds]);
    assert_eq!(given5.window, MotifWindow::new(12, 12 + 9 + 9));
    assert_eq!(given3.window, MotifWindow::new(0, 9 + 9));
}

// ===========================================================================
// 2. What the set guarantees, and what it does not
// ===========================================================================

/// The guarantee is about **ungapped** runs, and a bulged helix carrying more
/// than `k` total pairs is permitted.
///
/// This is the difference between the spec's shorthand — *"no helix of length
/// ≥ k"* — and what forbidding reverse complements of `k`-mers actually
/// delivers. Stating it as a passing test rather than as a caveat is what stops
/// a later reader assuming the stronger reading: a `k` chosen against ungapped
/// runs is a different `k` from one chosen against bulged ones, and that choice
/// belongs to the caller.
#[test]
fn a_bulged_duplex_of_more_than_k_pairs_is_permitted() {
    let k = 4;
    let given = "GGGGGGGGGGGG";
    let derived = derive(given, 12, FlankComplement::wobble(k));
    let ac = Ac::build(&derived.motifs).expect("valid");
    // Three runs of three `C`s, separated by `A`s, which cannot pair a `G`.
    let designed = "CCCACCCACCCA";
    assert!(
        !rejects(&ac, designed),
        "the designed flank was excluded, so this case says nothing about bulges",
    );
    // Yet it pairs nine of the given flank's twelve positions, in three stacked
    // runs of three — more total pairs than `k`, and no ungapped run of `k`.
    let (total, longest) = bulged_duplex(given.as_bytes(), designed.as_bytes());
    assert!(
        total > k && longest < k,
        "this case does not exhibit a bulged duplex: {total} pairs, longest \
         ungapped run {longest}",
    );
}

/// **Modular by construction, with the region table inert.**
///
/// The emitted sequence's own **free** fold — folded with no constraint at all — must
/// contain no ungapped run of `k` pairs joining the given flank to the designed
/// one. Checked against the free fold, never against the designed structure.
///
/// The instance is chosen so the control can fail. Under an inert table with a
/// long CDS available, a designed flank does not clamp to the *other* flank
/// anyway — it pairs into the CDS, which offers more — and the unrestricted
/// control reads zero too, leaving the cell unable to distinguish the instrument
/// from the instance. Here the CDS is short enough for the flanks to outweigh
/// it, and **the control is asserted to form the run**: without that assertion
/// this test measures the instance.
#[test]
fn the_free_fold_of_a_derived_design_spells_no_flank_to_flank_duplex() {
    let (params, usage) = params_usage();
    let protein = "MK";
    let k = 4;
    let given = "GGGAGGGAGGGA";
    let utr5 = UtrSpec::Fixed(given.into());
    let utr3 = UtrSpec::Designed(12);
    let (cds_start, cds_end) = (utr5.len(), utr5.len() + 3 * protein.len());
    let seq_len = cds_end + utr3.len();
    let cds = CdsWindow::new(utr5.len(), protein.len());
    let mut controls_clamped = 0usize;

    for &lambda in &[0.5, 0.99] {
        let free_run = |design: &Design| {
            let gap = sequence_gap(&params, &design.sequence, cds, RegionConstraints::default())
                .expect("the emitted sequence folds");
            longest_flank_run(&gap.free_structure, cds_start, cds_end, seq_len)
        };

        let control = design_with_utr_profile(
            &params,
            protein,
            &usage,
            lambda,
            &utr5,
            &utr3,
            RegionConstraints::default(),
            UtrProfile::inert(),
            ScanConfig::default(),
        )
        .expect("the control designs");
        let control_run = free_run(&control);
        controls_clamped += usize::from(control_run >= k);

        let derived = design_composite(
            &params,
            protein,
            &usage,
            lambda,
            &derived_spec(&utr5, &utr3, FlankComplement::wobble(k)),
            ScanConfig::default(),
        )
        .expect("the derived design is feasible");
        // Strictly shorter than `k`, **not** zero. The instrument leaves short
        // complementary runs legal on purpose, and the designs here do use that
        // — asserting zero would fail on a working instrument.
        let run = free_run(&derived);
        assert!(
            run < k,
            "the free fold of a derived design joins the flanks with an ungapped \
             run of {run} ≥ {k} [λ={lambda}]: {}",
            derived.sequence,
        );
        println!("λ={lambda}: derived longest flank-to-flank run {run}, control {control_run}",);
        // The sequence-level statement, independent of any fold: the designed
        // flank spells no partner of the given one at all.
        assert!(
            !spells_partner_inside(
                derived.sequence.as_bytes(),
                given.as_bytes(),
                k,
                cds_end,
                seq_len,
                true,
            ),
            "the designed flank spells a partner of a given window [λ={lambda}]: {}",
            derived.sequence,
        );
    }
    assert!(
        controls_clamped > 0,
        "with the derivation off this case never formed a flank-to-flank run of \
         {k}, so it says nothing about the derivation — it measures the instance",
    );
}

// ===========================================================================
// 3. The entry contract: sources, targets and the window between them
// ===========================================================================

/// A derived set enforced over the region it came from is the failure position
/// scoping exists to fix, and it surfaces as *infeasible* — which reads like a
/// hard constraint working rather than like a misconfiguration.
///
/// So the coupling is not left to the caller: the derivation computes its own
/// window, from the rule that a target must be designed. This pins that the two
/// are genuinely different outcomes on the same inputs, which is what makes
/// deriving the window worth doing.
#[test]
fn the_derivation_pins_a_window_that_excludes_its_own_source() {
    let (params, usage) = params_usage();
    let protein = "MK";
    // A flank holding both a `G` run and a `C` run spells its own complement, so
    // the derived set applied globally rejects the sequence that generated it —
    // which is the failure being guarded against, in its sharpest form.
    let utr5 = UtrSpec::Fixed("GGGGACCCCAGG".into());
    let utr3 = UtrSpec::Designed(12);
    let fc = FlankComplement::wobble(4);
    let derived = derive_from_flanks(fc, &utr5, &utr3, protein.len()).expect("well-posed");
    assert_eq!(
        derived.window,
        MotifWindow::new(
            utr5.len() as u32,
            (utr5.len() + 3 * protein.len() + utr3.len()) as u32,
        ),
        "the derived window is not the designed regions",
    );

    let run = |spec: CompositeSpec<'_, String>| {
        design_composite(&params, protein, &usage, 0.5, &spec, ScanConfig::default())
    };
    // The same patterns, hand-supplied and enforced globally — the arrangement
    // the derivation makes unreachable.
    assert!(
        run(CompositeSpec {
            utr5: &utr5,
            utr3: &utr3,
            constraints: RegionConstraints::default(),
            profile: UtrProfile::inert(),
            motifs: &derived.motifs,
            motif_window: MotifWindow::all(),
            flank_complement: None,
            complement_classes: None,
            positional: PositionalConstraints::empty(),
        })
        .is_err(),
        "the globally-enforced derived set was feasible, so this case cannot show \
         what pinning the window buys",
    );
    let scoped = run(derived_spec(&utr5, &utr3, fc)).expect("the derived design is feasible");
    // The source flank is outside the window and comes back verbatim.
    assert!(
        scoped.sequence.starts_with("GGGGACCCCAGG"),
        "the given flank did not survive: {}",
        scoped.sequence,
    );
}

/// Every configuration the derivation cannot serve is refused with a message,
/// and none of them is allowed to become an empty or self-defeating set.
///
/// **One arrangement is here and only one** — no region of the run holds given
/// sequence, so there is nothing to be a complement *of*. Every other
/// arrangement has a given source and is served; the rows that used to be
/// refused for want of a designed *flank* are the ones the coding sequence
/// answers, and they are asserted as accepted in
/// [`the_arrangement_table_holds_in_every_row`]. What remains beside it is the
/// two combinations that would need a second enforcement window, and the
/// degenerate `k`.
#[test]
fn the_configurations_the_derivation_cannot_serve_are_refused() {
    let (params, usage) = params_usage();
    let protein = "MKW";
    let fixed = UtrSpec::Fixed("GGGAGGGAGGGA".into());
    let designed = UtrSpec::Designed(12);
    let motifs = vec!["GGGG".to_string()];
    let none: Vec<String> = Vec::new();

    /// One configuration the derivation has to refuse.
    struct Case<'a> {
        label: &'a str,
        utr5: UtrSpec,
        utr3: UtrSpec,
        motifs: &'a [String],
        window: MotifWindow,
        fc: FlankComplement,
    }
    // The well-posed case every entry perturbs in exactly one way.
    let base = |label, utr5: &UtrSpec, utr3: &UtrSpec| Case {
        label,
        utr5: utr5.clone(),
        utr3: utr3.clone(),
        motifs: &none,
        window: MotifWindow::all(),
        fc: FlankComplement::wobble(4),
    };

    let cases = vec![
        base("both flanks designed", &designed, &designed),
        base("one designed flank and no other", &designed, &empty()),
        base("no flanks at all", &empty(), &empty()),
        Case {
            motifs: &motifs,
            ..base("alongside a hand-supplied set", &fixed, &designed)
        },
        Case {
            window: MotifWindow::new(0, 12),
            ..base("alongside an explicit window", &fixed, &designed)
        },
        Case {
            fc: FlankComplement::wobble(0),
            ..base("k = 0", &fixed, &designed)
        },
    ];

    for case in cases {
        let label = case.label;
        let err = design_composite(
            &params,
            protein,
            &usage,
            0.5,
            &CompositeSpec {
                utr5: &case.utr5,
                utr3: &case.utr3,
                constraints: RegionConstraints::default(),
                profile: UtrProfile::inert(),
                motifs: case.motifs,
                motif_window: case.window,
                flank_complement: Some(case.fc),
                complement_classes: None,
                positional: PositionalConstraints::empty(),
            },
            ScanConfig::default(),
        )
        .expect_err(&format!("{label} was accepted"));
        let message = err.to_string();
        assert!(
            message.starts_with("flank-complement derivation:"),
            "{label} failed for an unrelated reason: {message}",
        );
        // A refusal that does not say what to do instead is a case that happens
        // to error rather than a decision.
        assert!(
            message.len() > 60,
            "{label} was refused without saying what to do instead: {message}",
        );
    }
}

/// The whole arrangement table, row by row: which regions the set derives from,
/// which regions it is enforced over, and where that window sits.
///
/// One rule generates every row — **a source must be given and a target must be
/// designed** — and the coding sequence is designed in all of them. That is why
/// only one row refuses: no region holds given sequence, so there is nothing for
/// the set to be a complement *of*, and the case belongs to an alphabet
/// restriction rather than to this instrument.
///
/// Asserted on the report rather than through a design, because a design-level
/// check cannot separate *enforced over the wrong regions* from *enforced over
/// the right ones and feasible anyway*.
#[test]
fn the_arrangement_table_holds_in_every_row() {
    // Both flanks are 12 nt and neither is a reverse-complement palindrome, so
    // the union row below is a real union rather than one set twice.
    let flank5 = "ACGUGGCAUGCA";
    let flank3 = "AUUUCUUCUGAC";
    let protein_len = 3;
    let fc = FlankComplement::wobble(4);
    let given5 = || UtrSpec::Fixed(flank5.into());
    let given3 = || UtrSpec::Fixed(flank3.into());

    struct Row<'a> {
        label: &'a str,
        utr5: UtrSpec,
        utr3: UtrSpec,
        sources: &'a [Region],
        targets: &'a [Region],
        window: MotifWindow,
    }

    let rows = vec![
        Row {
            label: "5\u{2032} given, 3\u{2032} designed",
            utr5: given5(),
            utr3: UtrSpec::Designed(9),
            sources: &[Region::Utr5],
            targets: &[Region::Cds, Region::Utr3],
            window: MotifWindow::new(12, 30),
        },
        Row {
            label: "5\u{2032} given, no 3\u{2032}",
            utr5: given5(),
            utr3: empty(),
            sources: &[Region::Utr5],
            targets: &[Region::Cds],
            window: MotifWindow::new(12, 21),
        },
        Row {
            label: "3\u{2032} given, 5\u{2032} designed",
            utr5: UtrSpec::Designed(9),
            utr3: given3(),
            sources: &[Region::Utr3],
            targets: &[Region::Utr5, Region::Cds],
            window: MotifWindow::new(0, 18),
        },
        Row {
            label: "3\u{2032} given, no 5\u{2032}",
            utr5: empty(),
            utr3: given3(),
            sources: &[Region::Utr3],
            targets: &[Region::Cds],
            window: MotifWindow::new(0, 9),
        },
        Row {
            label: "both given",
            utr5: given5(),
            utr3: given3(),
            sources: &[Region::Utr5, Region::Utr3],
            targets: &[Region::Cds],
            window: MotifWindow::new(12, 21),
        },
    ];

    let mut by_label: HashMap<&str, Vec<String>> = HashMap::new();
    for row in &rows {
        let got = derive_from_flanks(fc, &row.utr5, &row.utr3, protein_len)
            .unwrap_or_else(|e| panic!("{} was refused: {e}", row.label));
        assert_eq!(got.sources, row.sources, "sources [{}]", row.label);
        assert_eq!(got.targets, row.targets, "targets [{}]", row.label);
        assert_eq!(got.window, row.window, "window [{}]", row.label);
        // Sources and targets are what makes enforcing a set over its own source
        // unreachable, so their disjointness is asserted rather than read off the
        // rows above.
        for source in &got.sources {
            assert!(
                !got.targets.contains(source),
                "{:?} is both a source and a target [{}]",
                source,
                row.label,
            );
        }
        assert!(!got.is_empty(), "no patterns derived [{}]", row.label);
        by_label.insert(row.label, got.motifs);
    }

    // Two sources contribute one set: the both-given row is the union of the two
    // single-source rows over the same flanks, deduplicated.
    let mut union = by_label["5\u{2032} given, no 3\u{2032}"].clone();
    union.extend(by_label["3\u{2032} given, no 5\u{2032}"].clone());
    union.sort();
    union.dedup();
    assert_eq!(
        by_label["both given"], union,
        "the two-source derivation is not the union of its sources",
    );
    // Which is only evidence of a union where the two sources disagree.
    assert_ne!(
        by_label["5\u{2032} given, no 3\u{2032}"], by_label["3\u{2032} given, no 5\u{2032}"],
        "the two flanks derive the same set, so the union row is one set twice",
    );

    // The fourth row: no given region anywhere, so no source exists.
    let refused = derive_from_flanks(
        fc,
        &UtrSpec::Designed(9),
        &UtrSpec::Designed(9),
        protein_len,
    )
    .expect_err("a run with no given region has nothing to derive from");
    assert!(
        refused.to_string().contains("k = 1 limit"),
        "the refusal does not name the alphabet restriction it defers to: {refused}",
    );
}

/// **Coverage is a class list, and a class states both halves of it.** Naming
/// fewer classes narrows which given region the set derives from and which
/// designed region it is enforced over, together.
///
/// The three cases below hold the arrangement fixed — both flanks given, the
/// coding sequence designed — so what varies is only which classes were named.
/// That is the axis a caller measuring one given flank against the other needs,
/// and it is what dissolves a separate "which arm" parameter: `5-cds` *is* the
/// 5′ arm, because the class names its own endpoints.
#[test]
fn coverage_is_a_class_list_and_a_class_names_both_of_its_ends() {
    let utr5 = UtrSpec::Fixed("ACGUGGCAUGCA".into());
    let utr3 = UtrSpec::Fixed("AUUUCUUCUGAC".into());
    let fc = FlankComplement::wobble(4);
    let derive = |classes: &[PairClass]| {
        derive_over(fc, &utr5, &utr3, 3, classes).expect("both flanks are given")
    };

    let both = derive(&[PairClass::Utr5Cds, PairClass::CdsUtr3]);
    let five = derive(&[PairClass::Utr5Cds]);
    let three = derive(&[PairClass::CdsUtr3]);

    for restricted in [&five, &three] {
        assert_eq!(restricted.window, both.window, "the target moved");
        assert_eq!(restricted.targets, both.targets, "the target moved");
        assert!(
            restricted.motifs.len() < both.motifs.len(),
            "naming one class did not narrow the set",
        );
    }
    assert_eq!(five.sources, [Region::Utr5]);
    assert_eq!(five.classes, [PairClass::Utr5Cds]);
    assert_eq!(three.sources, [Region::Utr3]);
    assert_eq!(three.classes, [PairClass::CdsUtr3]);
    assert_eq!(five.source_len, 12);
    assert_eq!(both.source_len, 24);
    // The universal rule is the whole class list, so it has to agree with the
    // list that names the two reachable classes — otherwise "every class" and
    // "every class that reaches" are two different defaults.
    assert_eq!(
        derive_from_flanks(fc, &utr5, &utr3, 3).expect("both flanks are given"),
        both,
        "the universal derivation is not the union over every class",
    );

    // A class the derivation cannot reach contributes nothing and is not an
    // error: both endpoints of `5-3` are given here, so there is no designed
    // target, and the caller may have named it for the structural constraint alone.
    let unreachable = derive(&[PairClass::Utr5Utr3]);
    assert!(
        unreachable.is_empty(),
        "a class with no designed endpoint still derived something",
    );
    assert!(unreachable.classes.is_empty(), "it was reported as reached");
    assert_eq!(unreachable.window, MotifWindow::new(0, 0));
}

/// The enforcement window is exactly the positions of the regions the coverage
/// made targets — never a span that swallows a region in between.
///
/// One window can only carry the constraint while the targets are contiguous,
/// and that is a claim about *every* arrangement and *every* class list rather
/// than about the ones a test happened to pick. So both are enumerated: three
/// spellings of each flank against all 64 subsets of the six classes.
#[test]
fn the_window_is_the_targets_and_nothing_between_them() {
    let flanks = [
        ("given", UtrSpec::Fixed("ACGUGGCAUGCA".into())),
        ("designed", UtrSpec::Designed(12)),
        ("absent", empty()),
    ];
    let protein_len = 3;
    let fc = FlankComplement::wobble(4);
    let mut nonempty = 0usize;
    for (label5, utr5) in &flanks {
        for (label3, utr3) in &flanks {
            let cds_start = utr5.len() as u32;
            let cds_end = cds_start + 3 * protein_len as u32;
            let seq_len = cds_end + utr3.len() as u32;
            let bounds = |region| match region {
                Region::Utr5 => (0, cds_start),
                Region::Cds => (cds_start, cds_end),
                Region::Utr3 => (cds_end, seq_len),
            };
            for mask in 0u8..64 {
                let classes: Vec<PairClass> = PairClass::ALL
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, c)| c)
                    .collect();
                let Ok(got) = derive_over(fc, utr5, utr3, protein_len, &classes) else {
                    continue;
                };
                let want = got
                    .targets
                    .iter()
                    .map(|&r| bounds(r))
                    .fold(None::<(u32, u32)>, |acc, (lo, hi)| match acc {
                        Some((a, b)) => Some((a.min(lo), b.max(hi))),
                        None => Some((lo, hi)),
                    })
                    .unwrap_or((0, 0));
                assert_eq!(
                    got.window,
                    MotifWindow::new(want.0, want.1),
                    "the window is not the target span [5\u{2032}={label5} 3\u{2032}={label3} \
                     mask={mask:06b}]",
                );
                // The span is the union only when the targets leave no gap, so
                // the width has to add up region by region.
                let covered: u32 = got
                    .targets
                    .iter()
                    .map(|&r| {
                        let (lo, hi) = bounds(r);
                        hi - lo
                    })
                    .sum();
                assert_eq!(
                    got.window.hi() - got.window.lo(),
                    covered,
                    "the target span swallowed a region that is not a target \
                     [5\u{2032}={label5} 3\u{2032}={label3} mask={mask:06b}]",
                );
                nonempty += usize::from(!got.targets.is_empty());
            }
        }
    }
    assert!(
        nonempty > 0,
        "no arrangement produced a target, so the enumeration proved nothing",
    );
}

// ===========================================================================
// 4. Floors
// ===========================================================================

/// A source flank shorter than `k` derives nothing, and nothing is not an empty
/// automaton — it is the run without the flag, byte for byte, through the
/// pass-through path.
#[test]
fn an_empty_derivation_is_byte_identical_to_no_derivation() {
    let (params, usage) = params_usage();
    let protein = "MKW";
    let utr5 = UtrSpec::Fixed("GGGA".into());
    let utr3 = UtrSpec::Designed(9);
    let fc = FlankComplement::wobble(5);
    assert!(
        derive_from_flanks(fc, &utr5, &utr3, protein.len())
            .expect("well-posed")
            .is_empty(),
        "the derivation was expected to be empty at k > |flank|",
    );
    for &lambda in &LAMBDAS {
        let want = design_with_utr_profile(
            &params,
            protein,
            &usage,
            lambda,
            &utr5,
            &utr3,
            RegionConstraints::default(),
            UtrProfile::inert(),
            ScanConfig::default(),
        )
        .expect("designs");
        let got = design_composite(
            &params,
            protein,
            &usage,
            lambda,
            &derived_spec(&utr5, &utr3, fc),
            ScanConfig::default(),
        )
        .expect("designs");
        assert_eq!(
            got, want,
            "an empty derivation perturbed the design [λ={lambda}]",
        );
    }
}

/// The derivation is the set plus the window and nothing else: handing the same
/// patterns to the hand-supplied surface under the same window reproduces it
/// byte for byte, **in every arrangement that has a given source**.
///
/// Which is also the floor that keeps a surface above this where it is — the
/// derivation adds a way to *obtain* a set, not a second way to enforce one.
///
/// The arrangements are swept rather than sampled because they reach different
/// windows: a designed flank puts one end of the span at the molecule's edge and
/// a given one leaves it at the coding boundary, so a run agreeing with its own
/// hand-supplied twin on one arrangement says nothing about the others.
///
/// The two are compared on their **outcome**, infeasibility included, because
/// the same set under the same window has to empty the space in the same places.
/// And the agreement is only worth having where the set forbids something —
/// against an inert one, two identical runs agree for no reason — so every
/// arrangement is required to **change** the outcome relative to the run without
/// the derivation, and to change it at least once while still returning a
/// design, which infeasibility alone would not show.
#[test]
fn a_derived_run_equals_the_same_set_supplied_by_hand() {
    let (params, usage) = params_usage();
    // Chosen so the derivation bites in every arrangement: the protein's coding
    // options are pyrimidine-rich and the given flanks are `G`-runs, whose
    // partners are exactly those. Without that the two routes agree because
    // neither constrains anything, which is indistinguishable from agreement.
    let protein = "MSLLK";
    let flank5 = "GGGAGGGA";
    let flank3 = "AUUUCUUC";
    let arrangements: Vec<(&str, UtrSpec, UtrSpec)> = vec![
        (
            "5\u{2032} given, 3\u{2032} designed",
            UtrSpec::Fixed(flank5.into()),
            UtrSpec::Designed(4),
        ),
        (
            "5\u{2032} given, no 3\u{2032}",
            UtrSpec::Fixed(flank5.into()),
            empty(),
        ),
        (
            "3\u{2032} given, 5\u{2032} designed",
            UtrSpec::Designed(4),
            UtrSpec::Fixed(flank3.into()),
        ),
        (
            "3\u{2032} given, no 5\u{2032}",
            empty(),
            UtrSpec::Fixed(flank3.into()),
        ),
        (
            "both given",
            UtrSpec::Fixed(flank5.into()),
            UtrSpec::Fixed(flank3.into()),
        ),
    ];

    for (label, utr5, utr3) in &arrangements {
        // The run without the derivation, once per λ: what "moved" is measured
        // against, and the same for every k and relation below.
        let free: Vec<Design> = LAMBDAS
            .iter()
            .map(|&lambda| {
                design_with_utr_profile(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    utr5,
                    utr3,
                    RegionConstraints::default(),
                    UtrProfile::inert(),
                    ScanConfig::default(),
                )
                .expect("the run without the derivation designs")
            })
            .collect();
        let mut moved_with_a_design = false;
        for k in [3usize, 4] {
            for complement in [Complement::Wobble, Complement::Strict] {
                let fc = FlankComplement { k, complement };
                let derived =
                    derive_from_flanks(fc, utr5, utr3, protein.len()).expect("well-posed");
                for (index, &lambda) in LAMBDAS.iter().enumerate() {
                    let run = |spec: CompositeSpec<'_, String>| {
                        design_composite(
                            &params,
                            protein,
                            &usage,
                            lambda,
                            &spec,
                            ScanConfig::default(),
                        )
                        .map_err(|e| e.to_string())
                    };
                    let by_hand = run(CompositeSpec {
                        utr5,
                        utr3,
                        constraints: RegionConstraints::default(),
                        profile: UtrProfile::inert(),
                        motifs: &derived.motifs,
                        motif_window: derived.window,
                        flank_complement: None,
                        complement_classes: None,
                        positional: PositionalConstraints::empty(),
                    });
                    let got = run(derived_spec(utr5, utr3, fc));
                    assert_eq!(
                        got,
                        by_hand,
                        "the derived run diverged from the same set supplied by hand \
                         [{label} k={k} {} λ={lambda}]",
                        complement.label(),
                    );
                    moved_with_a_design |= got.is_ok_and(|d| d.sequence != free[index].sequence);
                }
            }
        }
        assert!(
            moved_with_a_design,
            "the derivation never moved the design on [{label}] while still returning \
             one, so the agreement above is between two runs of an inert constraint",
        );
    }
}

/// **The exactness oracle.** With a derived set active, the design must be the
/// BRD-argmin over flank spellings × CDS encodings × nested structures,
/// restricted to the sequences whose designed flank spells no partner of any
/// window of the given one.
///
/// The filter is written from the pairing relation rather than from the derived
/// patterns, so a wrong derivation cannot pass by being wrong in the oracle too.
#[test]
fn the_constrained_optimum_is_exact_under_a_derived_set() {
    let (params, usage) = params_usage();
    let protein = "MK";
    let k = 3;
    let utr5_literal = "GGCA";
    let utr5 = UtrSpec::Fixed(utr5_literal.into());
    let utr3 = UtrSpec::Designed(3);
    let aa = protein_indices(protein);
    // The filter runs over the **designed** positions, which is the coding
    // sequence and the designed flank together — the same span the derivation
    // pins its window to, written here from the arrangement rather than read off
    // the report.
    let cds_start = utr5.len();
    let seq_len = cds_start + 3 * protein.len() + utr3.len();

    let mut scored: Vec<(f64, f64)> = Vec::new();
    let mut rejected = 0usize;
    for flank in every_segment(utr3.len()) {
        for enc in encodings(&aa) {
            let full = full_bytes(&aa, &enc, utr5_literal.as_bytes(), &flank);
            if spells_partner_inside(&full, utr5_literal.as_bytes(), k, cds_start, seq_len, true) {
                rejected += 1;
                continue;
            }
            scored.push((
                f64::from(free_mfe_dk(&params, &full)) / 100.0,
                cds_cai_log(&usage, &aa, &enc),
            ));
        }
    }
    assert!(
        rejected > 0,
        "the derived set excluded nothing, so this case is an unconstrained \
         problem wearing a constraint's name",
    );

    for &lambda in &LAMBDAS {
        let got = design_composite(
            &params,
            protein,
            &usage,
            lambda,
            &derived_spec(&utr5, &utr3, FlankComplement::wobble(k)),
            ScanConfig::default(),
        )
        .expect("designs");
        let want = scored
            .iter()
            .map(|&(mfe, cai)| brd(lambda, mfe, cai))
            .fold(f64::INFINITY, f64::min);
        let got_brd = brd(lambda, got.mfe_kcal_mol, got.cai_log);
        assert!(
            (got_brd - want).abs() <= BRD_TOL,
            "the derived optimum diverged from enumeration [λ={lambda}]: \
             design={got_brd} oracle={want} seq={}",
            got.sequence,
        );
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// The spec for a derived run: no hand-supplied set, no explicit window.
fn derived_spec<'a>(
    utr5: &'a UtrSpec,
    utr3: &'a UtrSpec,
    fc: FlankComplement,
) -> CompositeSpec<'a, String> {
    CompositeSpec {
        utr5,
        utr3,
        constraints: RegionConstraints::default(),
        profile: UtrProfile::inert(),
        motifs: &[],
        motif_window: MotifWindow::all(),
        flank_complement: Some(fc),
        complement_classes: None,
        positional: PositionalConstraints::empty(),
    }
}

/// The derivation from a given 5′ flank against a designed 3′ one.
fn derive(flank: &str, designed: usize, fc: FlankComplement) -> derna_lib::DerivedMotifs {
    derive_from_flanks(
        fc,
        &UtrSpec::Fixed(flank.into()),
        &UtrSpec::Designed(designed),
        3,
    )
    .expect("a given 5′ flank and a designed 3′ one is well-posed")
}

/// Whether `ac` rejects `segment` read as a whole from the root.
fn rejects(ac: &Ac, segment: &str) -> bool {
    let mut q = ac.start();
    for b in segment.bytes() {
        q = ac.step(q, Nucleotide::from_byte(b).expect("A/C/G/U"));
        if ac.is_accept(q) {
            return true;
        }
    }
    false
}

/// Every `ACGU` segment of length `k`.
fn every_segment(k: usize) -> Vec<Vec<u8>> {
    mixed_radix(&vec![4; k])
        .into_iter()
        .map(|d| d.into_iter().map(nuc_byte).collect())
        .collect()
}

/// The best antiparallel alignment of `designed` against `given` as
/// `(total pairs, longest ungapped run)`, over the one alignment that puts
/// position `t` of the given flank opposite position `len − 1 − t` of the
/// designed one.
fn bulged_duplex(given: &[u8], designed: &[u8]) -> (usize, usize) {
    let n = given.len().min(designed.len());
    let (mut total, mut longest, mut run) = (0usize, 0usize, 0usize);
    for t in 0..n {
        if pairs(given[t], designed[n - 1 - t], true) {
            total += 1;
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    (total, longest)
}

/// The longest ungapped run of stacked pairs in `structure` every pair of which
/// joins the 5′ flank to the 3′ flank.
///
/// The predicate the derivation actually makes, read off a dot-bracket: a
/// stacked run is a maximal chain `(i, j), (i+1, j−1), …`, and what is forbidden
/// is such a chain of length ≥ `k` with one end in each flank. A count of
/// flank-to-flank *pairs* would be testing a stronger claim the instrument does
/// not make.
fn longest_flank_run(structure: &str, cds_start: usize, cds_end: usize, seq_len: usize) -> usize {
    let mut partner = vec![usize::MAX; seq_len];
    for (i, j) in pairs_of(structure) {
        partner[i] = j;
        partner[j] = i;
    }
    let joins = |i: usize| i < cds_start && partner[i] != usize::MAX && partner[i] >= cds_end;
    let mut best = 0usize;
    for i in 0..cds_start {
        // Count only from the 5′ end of a maximal chain, so each run is measured
        // once.
        if !joins(i) || (i > 0 && joins(i - 1) && partner[i - 1] == partner[i] + 1) {
            continue;
        }
        let mut len = 0usize;
        let (mut a, mut b) = (i, partner[i]);
        while a < cds_start && joins(a) && partner[a] == b {
            len += 1;
            if b == 0 {
                break;
            }
            a += 1;
            b -= 1;
        }
        best = best.max(len);
    }
    best
}

/// The `(i, j)` pairs of a dot-bracket string.
fn pairs_of(structure: &str) -> Pairs {
    let mut stack = Vec::new();
    let mut out = Vec::new();
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => out.push((stack.pop().expect("balanced brackets"), i)),
            _ => {}
        }
    }
    out
}

/// Every digit vector of the mixed-radix counter with the given per-place radix.
fn mixed_radix(radix: &[usize]) -> Vec<Vec<usize>> {
    let total: usize = radix.iter().product();
    (0..total)
        .map(|mut n| {
            radix
                .iter()
                .map(|&r| {
                    let d = n % r;
                    n /= r;
                    d
                })
                .collect()
        })
        .collect()
}

/// Every CDS encoding of `aa` as per-residue codon choices.
fn encodings(aa: &[usize]) -> Vec<Vec<usize>> {
    mixed_radix(
        &aa.iter()
            .map(|&a| AMINO_ACIDS[a].codons.len())
            .collect::<Vec<_>>(),
    )
}

/// The whole-mRNA bytes for one flank spelling and CDS encoding.
fn full_bytes(aa: &[usize], encoding: &[usize], utr5: &[u8], utr3: &[u8]) -> Vec<u8> {
    let mut out = utr5.to_vec();
    for (&a, &ci) in aa.iter().zip(encoding) {
        out.extend(
            AMINO_ACIDS[a].codons[ci]
                .nuc
                .iter()
                .map(|&n| nuc_byte(n as usize)),
        );
    }
    out.extend_from_slice(utr3);
    out
}

/// CDS-only additive log-CAI of one encoding, `EPSILON`-floored as the DFA's own
/// edge weights are.
fn cds_cai_log(usage: &CodonUsage, aa: &[usize], encoding: &[usize]) -> f64 {
    aa.iter()
        .zip(encoding)
        .map(|(&a, &ci)| {
            let g = |c: usize| usage.fraction(a, c).max(EPSILON);
            let g_max = (0..AMINO_ACIDS[a].codons.len())
                .map(g)
                .fold(EPSILON, f64::max);
            (g(ci) / g_max).ln()
        })
        .sum()
}

/// The unconstrained MFE (dekacalories) of one spelled sequence, by enumerating
/// every nested canonical structure.
fn free_mfe_dk(params: &Params, full: &[u8]) -> i32 {
    let mut memo = Memo::new();
    structures(full, 0, full.len(), &mut memo)
        .into_iter()
        .map(|s| {
            let db = dot_bracket(full.len(), &s);
            evaluate(params, full, &db)
                .expect("structure scores")
                .dekacal()
        })
        .min()
        .expect("the all-unpaired structure is always present")
}

/// Enumerate every nested canonical secondary structure of `seq[lo..hi)` with
/// minimum hairpin loop 3, memoized per interval.
fn structures(seq: &[u8], lo: usize, hi: usize, memo: &mut Memo) -> Vec<Pairs> {
    if lo >= hi {
        return vec![Vec::new()];
    }
    if let Some(cached) = memo.get(&(lo, hi)) {
        return cached.clone();
    }
    let mut out = structures(seq, lo + 1, hi, memo);
    for k in (lo + 4)..hi {
        if !pairs(seq[lo], seq[k], true) {
            continue;
        }
        let inners = structures(seq, lo + 1, k, memo);
        let outers = structures(seq, k + 1, hi, memo);
        for inner in &inners {
            for outer in &outers {
                let mut s = Vec::with_capacity(1 + inner.len() + outer.len());
                s.push((lo, k));
                s.extend_from_slice(inner);
                s.extend_from_slice(outer);
                out.push(s);
            }
        }
    }
    memo.insert((lo, hi), out.clone());
    out
}

/// The dot-bracket string for a pair list over a length-`n` sequence.
fn dot_bracket(n: usize, pairs: &[(usize, usize)]) -> String {
    let mut s = vec![b'.'; n];
    for &(i, j) in pairs {
        s[i] = b'(';
        s[j] = b')';
    }
    String::from_utf8(s).expect("ascii brackets")
}

/// Fixed-λ BRD on rounded dekacalories, matching how the oracle and the design
/// both round at the scalarization boundary.
fn brd(lambda: f64, mfe_kcal: f64, adaptation_log: f64) -> f64 {
    lambda * (mfe_kcal * 100.0).round() - (1.0 - lambda) * (adaptation_log * 100.0).round()
}
