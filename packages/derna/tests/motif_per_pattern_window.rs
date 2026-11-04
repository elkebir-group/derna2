//! A window per pattern: `--motif PATH@LO..HI` and the `@window LO..HI` line
//! inside a set file.
//!
//! The behaviour the surface exists for is one run saying *this blacklist over
//! the whole coding sequence, that one only from coding position 5 onward*.
//! The case that needs it is a pattern a blacklist means **internally** which
//! the molecule also spells across the 5′ boundary: a strong initiation
//! context ends `ACC`, the start codon follows, and `+4` is fixed by the
//! residue after the initiator methionine — so the pattern is spelled by
//! sequence no codon assignment can touch, and forbidding it over the whole
//! coding sequence empties the design space.
//!
//! Every test here therefore runs both arms. Unscoped the run is refused;
//! scoped it returns a front, and the pattern that carries no window is still
//! enforced everywhere.

use derna_cli::front;
use derna_cli::molecule::Molecule;

/// A 5′ flank ending in the `ACC` of a strong initiation context, so `RCC` is
/// spelled by fixed sequence.
const UTR5: &str = "GGGAAAUAAGCCACC";
const UTR3: &str = "GGCCAAUUGG";
/// Residue 2 is `A`, every codon of which begins with `G`, so `+4 = G` is
/// forced and `RCCAUGG` is spelled whatever the design does.
const PROTEIN: &str = "MAKVLAKGIV";
/// The initiation pattern, as a blacklist of internal Kozak-like contexts
/// writes it: `R` is `A` or `G`, and `T` is read as `U`.
const INITIATION: &str = "RCCATGG";
/// A pattern with no window of its own, enforced over the whole coding
/// sequence in every arm below.
const EVERYWHERE: &str = "GCC";

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// A set file written for one test, removed when it returns.
struct SetFile(std::path::PathBuf);

impl SetFile {
    fn new(name: &str, text: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("derna-window-{name}-{}.txt", std::process::id()));
        std::fs::write(&path, text).unwrap();
        Self(path)
    }

    fn path(&self) -> String {
        self.0.to_str().unwrap().to_string()
    }
}

impl Drop for SetFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `derna front` over `PROTEIN` between the two flanks, under `extra`.
fn front_under(extra: &[String]) -> (Molecule, front::FrontArgs) {
    let mut argv = words(&[
        "--protein",
        PROTEIN,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
    ]);
    argv.extend(extra.iter().cloned());
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    (m, args)
}

/// The front's points, or the reason the run was refused.
fn walk(extra: &[String]) -> Result<Vec<String>, String> {
    let (m, args) = front_under(extra);
    match front::motif_walk(&m, &args.sequence, 1, Default::default(), None) {
        Ok(walked) => Ok(walked.front.iter().map(|p| p.sequence.clone()).collect()),
        Err(e) => Err(e.to_string()),
    }
}

/// Whether `sequence` spells `motif` — `R` read as `A` or `G`, `T` as `U` —
/// with a position in `[lo, hi)`.
fn spells_within(sequence: &str, motif: &str, (lo, hi): (usize, usize)) -> bool {
    let classes: Vec<&[u8]> = motif
        .bytes()
        .map(|code| match code {
            b'R' => b"AG".as_slice(),
            b'T' | b'U' => b"U".as_slice(),
            b'A' => b"A".as_slice(),
            b'C' => b"C".as_slice(),
            b'G' => b"G".as_slice(),
            other => panic!("the test's matcher does not read {}", other as char),
        })
        .collect();
    let text = sequence.as_bytes();
    (0..=text.len().saturating_sub(classes.len())).any(|at| {
        at < hi
            && at + classes.len() > lo
            && classes
                .iter()
                .zip(&text[at..at + classes.len()])
                .all(|(allowed, letter)| allowed.contains(letter))
    })
}

/// The coding sequence's span in the molecule, as the tests read it.
fn cds() -> (usize, usize) {
    (UTR5.len(), UTR5.len() + 3 * PROTEIN.len())
}

#[test]
fn a_window_in_the_set_file_is_what_makes_the_initiation_pattern_satisfiable() {
    // The whole point of the surface, as one file holding both patterns. The
    // run that forbids the initiation pattern everywhere has no design to
    // return, because `RCC` is fixed flank sequence and `+4 = G` is forced by
    // residue 2. Scoped to coding position 5 onward it does, and the pattern
    // carrying no window is still forbidden over the whole coding sequence.
    let unscoped = SetFile::new("unscoped", &format!("{EVERYWHERE}\n{INITIATION}\n"));
    let e = walk(&words(&["--motif", &unscoped.path()])).expect_err("the space is empty");
    // The refusal names the set, and this is the string a harness reads to
    // tell an infeasible rung from a crashed one, so it is asserted verbatim
    // rather than by a word that happens to appear in it.
    assert!(
        e.contains("no encoding of the protein avoids every motif"),
        "{e}"
    );
    assert!(e.contains(INITIATION), "{e}");

    let scoped = SetFile::new(
        "scoped",
        &format!("{EVERYWHERE}\n@window 5..-1\n{INITIATION}\n"),
    );
    let front = walk(&words(&["--motif", &scoped.path()])).expect("the scoped run has a front");
    assert!(!front.is_empty(), "the scoped run returned no point");
    for sequence in &front {
        assert!(
            !spells_within(sequence, EVERYWHERE, cds()),
            "{sequence} spells {EVERYWHERE}, which carries no window"
        );
        // Scoping gives up exactly the junction occurrence: every design still
        // spells the initiation pattern there, and none spells it inside the
        // window it is now forbidden over.
        let (lo, hi) = cds();
        assert!(
            spells_within(sequence, INITIATION, (lo, hi)),
            "{sequence} does not spell the junction occurrence the window excluded"
        );
        assert!(
            !spells_within(sequence, INITIATION, (lo + 4, hi)),
            "{sequence} spells {INITIATION} inside the window it is forbidden over"
        );
    }
}

#[test]
fn a_value_suffix_scopes_what_that_value_names() {
    // The same scope written on the command line instead of in the file, which
    // is the arm a caller reaching for one blacklist of one pattern has.
    let front = walk(&words(&[
        "--motif",
        EVERYWHERE,
        "--motif",
        &format!("{INITIATION}@5..-1"),
    ]))
    .expect("the scoped run has a front");
    assert!(!front.is_empty());
    for sequence in &front {
        assert!(!spells_within(sequence, EVERYWHERE, cds()));
    }
    // Without the suffix the same two patterns empty the space.
    walk(&words(&["--motif", EVERYWHERE, "--motif", INITIATION]))
        .expect_err("the space is empty without the scope");
}

#[test]
fn the_header_records_which_pattern_carried_which_window() {
    // `motifs` and one `motif_window` no longer determine the constraint once
    // two patterns carry different windows, so the header pairs each pattern
    // with its own window. Two runs differing only in which pattern is scoped
    // must therefore differ in the header.
    let fields = |extra: &[String]| {
        let (m, args) = front_under(extra);
        args.sequence.header_fields(&m).unwrap()
    };

    let ours = fields(&words(&[
        "--motif",
        EVERYWHERE,
        "--motif",
        &format!("{INITIATION}@5..-1"),
    ]));
    let swapped = fields(&words(&[
        "--motif",
        &format!("{EVERYWHERE}@5..-1"),
        "--motif",
        INITIATION,
    ]));
    assert_ne!(
        ours["motif_scopes"], swapped["motif_scopes"],
        "two runs scoping different patterns share a motif_scopes"
    );
    // The two fields that used to carry the whole constraint are identical
    // across that pair, which is why the pairing needs a field of its own.
    assert_eq!(ours["motifs"], swapped["motifs"]);
    assert_eq!(ours["motif_window"], swapped["motif_window"]);

    // The field says so item by item, one pairing per item.
    let scopes = ours["motif_scopes"].as_array().unwrap();
    assert_eq!(scopes.len(), 2);
    assert_eq!(scopes[0]["motif"], EVERYWHERE);
    assert_eq!(scopes[0]["window"], "the coding sequence");
    assert_eq!(scopes[1]["motif"], INITIATION);
    assert_eq!(scopes[1]["window"], "coding positions 5..-1");

    // A run with no per-pattern window writes no such field: there is one
    // window and `motif_window` is it.
    let plain = fields(&words(&["--motif", EVERYWHERE]));
    assert!(plain.get("motif_scopes").is_none());
    assert_eq!(plain["motif_window"], "the coding sequence");
}

#[test]
fn a_scope_composes_with_the_flags_a_combined_run_uses() {
    // The surface has to hold with everything else a combined design run asks
    // for at once, since that is the only configuration it was added for.
    let scoped = SetFile::new(
        "composed",
        &format!("{EVERYWHERE}\n@window 5..-1\n{INITIATION}\n"),
    );
    let mut argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--stop",
        "UAA",
        "--forbid",
        "5-cds,cds-3",
        "--access",
        "4,30",
        "--motif",
        &scoped.path(),
    ]);
    argv.extend(words(&["--pin", "-6:GCCACC"]));
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let placed = args.sequence.placed(&m).unwrap();
    assert_eq!(placed.len(), 2);
    // The unscoped pattern spans the whole coding sequence and the scoped one
    // starts four nucleotides in, which is what excludes the junction.
    let cds = m.cds();
    assert_eq!(
        (placed[0].1.lo(), placed[0].1.hi()),
        (cds.start as u32, cds.end as u32)
    );
    assert_eq!(placed[1].1.lo(), cds.start as u32 + 4);
    assert_eq!(placed[1].1.hi(), cds.end as u32);
}

#[test]
fn design_refuses_a_window_it_cannot_enforce() {
    // `derna design` forbids its whole set over one window. A pattern carrying
    // its own is refused there rather than quietly widened to the set's, and
    // the message names the command that does enforce it.
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--lambda",
        "0.5",
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        &format!("{INITIATION}@5..-1"),
    ]);
    let args = derna_cli::design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let e = derna_cli::design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None)
        .expect_err("design refuses a per-pattern window")
        .to_string();
    assert!(e.contains("derna front"), "{e}");
    assert!(e.contains("--motif-window"), "{e}");
}
