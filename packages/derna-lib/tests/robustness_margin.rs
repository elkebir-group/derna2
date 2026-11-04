//! The robustness margin: what a molecule gives up to form the pairing it was
//! constrained against.
//!
//! The quantity is a minimum over the structures containing **at least one**
//! forbidden duplex — a disjunction the folding recurrence cannot express. The
//! claim under test is that on a fixed sequence it is nevertheless exact,
//! because the disjunction is finite: enumerate the duplexes, require each in
//! turn, and take the least. Everything here checks one of the three things that
//! claim rests on — that the enumeration is the right set, that the minimum over
//! it is the true minimum, and that a truncated budget degrades in the direction
//! it says it does.
//!
//! The brute-force oracle enumerates every nested structure of a short sequence
//! and scores it under the energy model directly, so it shares no code with the
//! folder it checks. It is used only after it has reproduced the folder's
//! **unconstrained** optimum on the same instance — an oracle that disagreed
//! there would be measuring a different grammar, and its verdict on the
//! constrained one would mean nothing.

use derna_lib::complement::pairing_motifs;
use derna_lib::coords::Layout;
use derna_lib::structure::StructureConstraints;
use derna_lib::{
    cross_region_duplexes, honouring_mfe, linearfold_fold, robustness_margin, stacked_runs, Beam,
    CdsWindow, Complement, Duplex, Params,
};
use rna_mfe::{evaluate, Nucleotide};

/// Energies are integer dekacalories under the model, so equality is exact up to
/// the `/100.0` float division.
const EXACT: f64 = 1e-9;

/// A 22-nucleotide molecule with two competing cross-region duplexes: a
/// `G·C` five-pair stem the sequence prefers, and a `U·A` five-pair stem that
/// crosses it and therefore cannot coexist with it.
///
/// Ten nucleotides of flank followed by four codons, so the flank is
/// `0..10` and the coding window `10..22`.
const TWO_STEMS: &str = "GGGGGUUUUUCCCCCAAAAAAA";
const FLANK: std::ops::Range<u32> = 0..10;
const WINDOW: std::ops::Range<u32> = 10..22;

fn params() -> Params {
    Params::load_default().expect("params")
}

fn layout() -> Layout {
    Layout::new(10, 4, 0)
}

fn nucleotides(s: &str) -> Vec<Nucleotide> {
    s.bytes()
        .map(|b| Nucleotide::from_byte(b).expect("ACGU"))
        .collect()
}

/// The margin over the whole violating set of `run`-pair duplexes.
fn margin_at(seq: &str, run: u32) -> derna_lib::Margin {
    let candidates =
        cross_region_duplexes(&nucleotides(seq), FLANK, WINDOW, run, Complement::Strict);
    robustness_margin(
        &params(),
        seq,
        &layout(),
        &candidates,
        Beam::exact(),
        usize::MAX,
    )
    .expect("margin")
}

// ─── the enumeration is the right set ────────────────────────────────────────

/// Both stems are found, and the enumeration reports them 5′-endpoint outward.
#[test]
fn the_enumeration_finds_every_ungapped_duplex_it_can_spell() {
    let found = cross_region_duplexes(
        &nucleotides(TWO_STEMS),
        FLANK,
        WINDOW,
        5,
        Complement::Strict,
    );
    let ends: Vec<(u32, u32)> = found.iter().map(|d| (d.open, d.close)).collect();
    assert!(
        ends.contains(&(0, 14)),
        "the G·C stem joining flank 0..5 to window 10..15 is missing: {ends:?}"
    );
    assert!(
        ends.contains(&(5, 19)),
        "the U·A stem joining flank 5..10 to window 15..20 is missing: {ends:?}"
    );
    assert!(
        found.iter().all(|d| d.len == 5),
        "every duplex carries the run length it was enumerated at"
    );
}

/// A helix interrupted by a bulge is not an ungapped duplex, and a set stated
/// over ungapped runs does not forbid it. Asserting otherwise is the harness's
/// error and not the code's.
#[test]
fn a_bulged_helix_is_not_an_ungapped_duplex() {
    // Flank `GGGGG`; the window spells `CCC` `A` `CC`, so the longest ungapped
    // run against the flank is three even though six positions could pair.
    let seq = "GGGGGAAAAACCCACCAAAAAA";
    let bulged = nucleotides(seq);
    let runs: Vec<u32> = (3..=5)
        .filter(|&r| {
            !cross_region_duplexes(&bulged, FLANK, WINDOW, r, Complement::Strict).is_empty()
        })
        .collect();
    assert_eq!(
        runs,
        vec![3],
        "only the three-pair run survives the bulge; a four-pair one would mean \
         the enumeration is stitching across it"
    );
}

/// An occurrence has to lie entirely inside the window it is enforced over. One
/// straddling the edge is permitted, so it is not enumerated.
#[test]
fn a_duplex_straddling_the_window_edge_is_not_enumerated() {
    let seq = nucleotides(TWO_STEMS);
    let inside = cross_region_duplexes(&seq, FLANK, WINDOW, 5, Complement::Strict);
    // Narrowing the window past the G·C stem's coding end drops it, and drops
    // nothing else, because that stem occupies exactly 10..15.
    let narrowed = cross_region_duplexes(&seq, FLANK, 11..22, 5, Complement::Strict);
    assert!(
        inside.iter().any(|d| d.open == 0 && d.close == 14),
        "the G·C stem is inside the full window"
    );
    assert!(
        !narrowed.iter().any(|d| d.open == 0 && d.close == 14),
        "a window starting one nucleotide in no longer contains it entirely"
    );
    assert!(
        !narrowed.is_empty(),
        "the narrowed window still contains something, so the check above is not \
         passing on an empty set"
    );
}

/// The derived set and the enumeration are two statements of one relation: the
/// window spells a forbidden pattern exactly when an ungapped duplex of that
/// length exists against the flank.
#[test]
fn a_derived_set_is_spelled_exactly_when_a_duplex_exists() {
    let params = [(5usize, Complement::Strict), (4, Complement::Wobble)];
    let mut agreed = 0;
    let mut with_duplex = 0;
    for seed in 0..48u64 {
        let seq = pseudo_random(seed, 22);
        for (k, complement) in params {
            let motifs = pairing_motifs(
                &seq[FLANK.start as usize..FLANK.end as usize],
                k,
                complement,
            );
            let spelled = window_spells_any(&seq, &motifs, k);
            let duplexes =
                !cross_region_duplexes(&seq, FLANK, WINDOW, k as u32, complement).is_empty();
            assert_eq!(
                spelled, duplexes,
                "seed {seed}, k={k}: the window spells a derived pattern ({spelled}) but \
                 the duplex enumeration says {duplexes}"
            );
            agreed += 1;
            with_duplex += usize::from(duplexes);
        }
    }
    assert_eq!(agreed, 96);
    assert!(
        with_duplex > 0,
        "no instance had a duplex at all, so the agreement is over one answer only"
    );
}

/// Whether the coding window spells any of `motifs`, read literally.
fn window_spells_any(seq: &[Nucleotide], motifs: &[String], k: usize) -> bool {
    let window = &seq[WINDOW.start as usize..WINDOW.end as usize];
    window
        .windows(k)
        .any(|segment| motifs.iter().any(|motif| matches_iupac(motif, segment)))
}

/// Whether one IUPAC pattern matches a literal segment.
fn matches_iupac(motif: &str, segment: &[Nucleotide]) -> bool {
    motif.bytes().zip(segment).all(|(code, &nuc)| {
        let letter = match nuc {
            Nucleotide::A => b'A',
            Nucleotide::C => b'C',
            Nucleotide::G => b'G',
            Nucleotide::U => b'U',
        };
        iupac(code).contains(&letter)
    })
}

fn iupac(code: u8) -> &'static [u8] {
    match code {
        b'A' => b"A",
        b'C' => b"C",
        b'G' => b"G",
        b'U' => b"U",
        b'R' => b"AG",
        b'Y' => b"CU",
        b'S' => b"CG",
        b'W' => b"AU",
        b'K' => b"GU",
        b'M' => b"AC",
        b'B' => b"CGU",
        b'D' => b"AGU",
        b'H' => b"ACU",
        b'V' => b"ACG",
        _ => b"ACGU",
    }
}

/// A deterministic sequence, so a failure is reproducible from its seed alone.
fn pseudo_random(seed: u64, n: usize) -> Vec<Nucleotide> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U][(x % 4) as usize]
        })
        .collect()
}

// ─── the minimum over it is the true minimum ─────────────────────────────────

/// The free arm of the margin is the sequence's own optimum, and it agrees with
/// the reference folder — the matched-pair floor the difference rests on.
#[test]
fn the_free_arm_reproduces_the_reference_folder() {
    let inert = StructureConstraints::none(layout().len());
    let ours =
        honouring_mfe(&params(), TWO_STEMS, &layout(), &inert, Beam::exact()).expect("free fold");
    let (_, reference) =
        linearfold_fold(&params(), TWO_STEMS.as_bytes(), 1 << 20).expect("reference fold");
    assert!(
        (ours - reference.to_kcal_mol()).abs() < EXACT,
        "free arm {ours} against reference {}",
        reference.to_kcal_mol()
    );
}

/// The margin equals the true minimum over violating structures, checked
/// against exhaustive enumeration of the structure space.
#[test]
fn the_margin_is_the_least_a_violating_structure_costs() {
    let params = params();
    let seq = nucleotides(TWO_STEMS);
    let structures = every_nested_structure(&seq);
    let scored: Vec<(String, f64)> = structures
        .into_iter()
        .filter_map(|s| {
            evaluate(&params, TWO_STEMS.as_bytes(), &s)
                .ok()
                .map(|e| (s, e.to_kcal_mol()))
        })
        .collect();
    let free = scored
        .iter()
        .map(|(_, e)| *e)
        .reduce(f64::min)
        .expect("some structure");
    let measured = margin_at(TWO_STEMS, 5);
    assert!(
        (free - measured.free_mfe_kcal_mol).abs() < EXACT,
        "the oracle's own optimum {free} must reproduce the folder's {} before its \
         verdict on the constrained problem means anything",
        measured.free_mfe_kcal_mol
    );
    let violating = scored
        .iter()
        .filter(|(s, _)| contains_cross_region_run(s, 5))
        .map(|(_, e)| *e)
        .reduce(f64::min)
        .expect("some structure violates");
    let expected = violating - free;
    let got = measured.least().expect("the violating set is non-empty");
    assert!(
        (expected - got).abs() < EXACT,
        "margin {got} against exhaustive {expected}"
    );
    assert!(
        measured.is_exact(),
        "an uncapped budget folds the whole violating set"
    );
}

/// Whether a dot-bracket contains an ungapped run of `run` stacked pairs joining
/// the flank to the window — stated over positions, independently of the
/// enumeration under test.
fn contains_cross_region_run(structure: &str, run: u32) -> bool {
    let partner = partners(structure);
    (FLANK.start..=FLANK.end - run).any(|i| {
        (0..run).all(|s| {
            partner[(i + s) as usize].is_some_and(|j| {
                WINDOW.contains(&j) && partner[i as usize].is_some_and(|first| j + s == first)
            })
        })
    })
}

/// Partner index per position, `None` where unpaired.
fn partners(structure: &str) -> Vec<Option<u32>> {
    let mut out = vec![None; structure.len()];
    let mut stack = Vec::new();
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => {
                let open = stack.pop().expect("balanced");
                out[open] = Some(i as u32);
                out[i] = Some(open as u32);
            }
            _ => {}
        }
    }
    out
}

/// Every nested structure of `seq` over canonical pairs with a hairpin of at
/// least three unpaired nucleotides.
fn every_nested_structure(seq: &[Nucleotide]) -> Vec<String> {
    let n = seq.len();
    let mut out = Vec::new();
    let mut current = vec![b'.'; n];
    extend(seq, 0, &mut current, &mut out);
    out
}

/// Place or skip a pair opening at `from`, depth-first over the whole sequence.
fn extend(seq: &[Nucleotide], from: usize, current: &mut Vec<u8>, out: &mut Vec<String>) {
    if from >= seq.len() {
        out.push(String::from_utf8(current.clone()).expect("ascii"));
        return;
    }
    extend(seq, from + 1, current, out);
    for close in from + 4..seq.len() {
        if current[close] != b'.' || !canonical(seq[from], seq[close]) {
            continue;
        }
        if !nests(current, from, close) {
            continue;
        }
        current[from] = b'(';
        current[close] = b')';
        extend(seq, from + 1, current, out);
        current[from] = b'.';
        current[close] = b'.';
    }
}

/// Whether adding `(from, close)` keeps the structure nested — every pair inside
/// the span stays inside it.
fn nests(current: &[u8], from: usize, close: usize) -> bool {
    let partner = partners(&String::from_utf8_lossy(current));
    (from + 1..close).all(|p| partner[p].is_none_or(|q| (from as u32) < q && q < close as u32))
}

fn canonical(a: Nucleotide, b: Nucleotide) -> bool {
    use Nucleotide::{A, C, G, U};
    matches!((a, b), (A, U) | (U, A) | (C, G) | (G, C) | (G, U) | (U, G))
}

// ─── the instrument is on, and degrades where it says it does ────────────────

/// A duplex the sequence's own optimum already forms costs it nothing, and one
/// that crosses it costs it something. Both directions are asserted, because a
/// margin instrument that only ever returned zero would pass the first alone.
#[test]
fn the_instrument_registers_both_a_free_duplex_and_a_paid_one() {
    let measured = margin_at(TWO_STEMS, 5);
    let least = measured.least().expect("violating set is non-empty");
    let greatest = measured.greatest().expect("violating set is non-empty");
    assert!(
        least.abs() < EXACT,
        "the G·C stem is the sequence's own optimum, so some duplex is free: {least}"
    );
    assert!(
        greatest > EXACT,
        "the U·A stem crosses the G·C one, so some duplex is paid for: {greatest}"
    );
}

/// The violating set shrinks as the run lengthens, so the margin cannot fall.
#[test]
fn a_longer_run_cannot_lower_the_margin() {
    let short = margin_at(TWO_STEMS, 4);
    let long = margin_at(TWO_STEMS, 5);
    assert!(
        short.candidates >= long.candidates,
        "every five-pair duplex contains a four-pair one: {} against {}",
        short.candidates,
        long.candidates
    );
    let (Some(a), Some(b)) = (short.least(), long.least()) else {
        panic!("both run lengths have a violating set here");
    };
    assert!(a <= b + EXACT, "margin at run 4 ({a}) exceeds run 5 ({b})");
}

/// A budget too small to fold the whole violating set reports an upper bound,
/// says so, and never reports a value below the exact one.
#[test]
fn a_capped_budget_reports_an_upper_bound_and_says_so() {
    let candidates = cross_region_duplexes(
        &nucleotides(TWO_STEMS),
        FLANK,
        WINDOW,
        5,
        Complement::Strict,
    );
    assert!(
        candidates.len() > 1,
        "the cap is only meaningful with something to drop"
    );
    let capped = robustness_margin(
        &params(),
        TWO_STEMS,
        &layout(),
        &candidates,
        Beam::exact(),
        1,
    )
    .expect("capped margin");
    let full = margin_at(TWO_STEMS, 5);
    assert!(!capped.is_exact(), "a capped run is not an exact margin");
    assert!(
        capped.dropped > 0,
        "the dropped count names what was skipped"
    );
    assert!(
        capped.least().expect("one fold ran") >= full.least().expect("exact") - EXACT,
        "a truncated minimum can only be at or above the exact one"
    );
}

/// An empty violating set is an absent margin rather than a zero one — the
/// separation is unbounded, and averaging a zero in would understate it.
#[test]
fn an_empty_violating_set_has_no_margin_rather_than_a_zero_one() {
    let none: Vec<Duplex> = Vec::new();
    let measured = robustness_margin(
        &params(),
        TWO_STEMS,
        &layout(),
        &none,
        Beam::exact(),
        usize::MAX,
    )
    .expect("margin");
    assert_eq!(measured.candidates, 0);
    assert!(measured.least().is_none());
    assert!(measured.is_exact());
}

/// A duplex a fold **formed** is one the sequence can **spell**, and the two
/// censuses name it identically.
///
/// The distinction the pairing carries is what each is a fact about:
/// [`cross_region_duplexes`] reads a sequence and reports what it could pair;
/// [`stacked_runs`] reads a structure and reports what a fold did pair. A
/// constraint is stated over the first and a residue is measured on the second,
/// so a report that puts them in one column depends on them being the same
/// object — position for position and stability for stability.
#[test]
fn a_run_a_fold_formed_is_one_the_sequence_could_spell() {
    // 5′ UTR of four, a two-codon coding sequence, a 3′ UTR of four. The two G's
    // opening the coding sequence pair with the two C's opening the 3′ UTR.
    let seq: Vec<Nucleotide> = "AAAAGGAAAACCAA"
        .bytes()
        .map(|b| Nucleotide::from_byte(b).expect("A/C/G/U"))
        .collect();
    let structure = "....((....))..";
    let cds = CdsWindow::new(4, 2);

    let formed: Vec<Duplex> = stacked_runs(structure, cds)
        .into_iter()
        .map(|run| Duplex::from_run(&seq, run))
        .collect();
    assert_eq!(
        formed,
        [Duplex {
            open: 4,
            close: 11,
            len: 2,
            stability: 6,
        }],
        "the fold forms one two-pair coding-to-3′ duplex of two G·C pairs",
    );

    let spellable = cross_region_duplexes(&seq, 4..10, 10..14, 2, Complement::Wobble);
    assert!(
        spellable.contains(&formed[0]),
        "the formed duplex {:?} is absent from the spellable census {spellable:?}",
        formed[0],
    );
}
