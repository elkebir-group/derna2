//! `--stop UAA|UAG|UGA`: a **given** stop codon after the last residue.
//!
//! The codon is spelled at the head of the 3′ flank, so it is fixed sequence
//! with no synonymous choice and no part of CAI, and the constraint counts it
//! as coding sequence. Both halves are tested, because each alone passes
//! vacuously:
//!
//! - under `--forbid none` the given stop codon designs exactly what spelling
//!   it into `--utr3` designs;
//! - under the constraint the two differ in the one place they should — a
//!   pair from a codon to the stop is coding-to-coding and allowed, where the
//!   same pair with the stop spelled into the 3′ UTR crosses the `cds-3`
//!   boundary and is forbidden.
//!
//! `--stop none` asks for no stop codon, which with a flank given is **not**
//! what omitting the flag means: that designs one (`stop_slot.rs`). So every
//! comparison here spells its intent, and the one place the two defaults meet
//! — a molecule with no flank, where a stop codon could change no answer — is
//! tested for the equality that still holds.

use derna_cli::design;
use derna_cli::molecule::{Molecule, Stop};
use derna_cli::result::molecule_fields;

const PROTEIN: &str = "MKVLAKGIVSFA";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";
const WEIGHTS: [&str; 3] = ["0.05", "0.5", "0.95"];

fn molecule(extra: &[&str]) -> (Molecule, Vec<f64>) {
    let mut argv: Vec<String> = ["--protein", PROTEIN, "--utr5", UTR5]
        .iter()
        .map(|s| s.to_string())
        .collect();
    argv.extend(["--lambda".to_string(), WEIGHTS.join(",")]);
    argv.extend(extra.iter().map(|s| s.to_string()));
    let args = design::parse(argv).unwrap();
    (Molecule::resolve(&args.shared).unwrap(), args.lambdas)
}

/// Every design's sequence, structure, energy and adaptation, at each weight.
fn designs(m: &Molecule, lambdas: &[f64]) -> Vec<(String, String, f64, f64)> {
    design::solve(m, lambdas, 1)
        .unwrap()
        .into_iter()
        .map(|s| {
            let d = s.design;
            (d.sequence, d.structure, d.mfe_kcal_mol, d.cai_log)
        })
        .collect()
}

#[test]
fn stop_none_asks_for_no_stop_codon() {
    let (none, lambdas) = molecule(&["--utr3", UTR3, "--stop", "none"]);
    assert!(molecule_fields(&none)["stop"].is_null());
    assert_eq!(none.stop, Stop::Absent);
    // Every design ends at the 3′ UTR, with no codon between.
    for (sequence, ..) in designs(&none, &lambdas) {
        let coding = UTR5.len() + 3 * PROTEIN.len();
        assert_eq!(sequence.len(), coding + UTR3.len());
        assert!(sequence.ends_with(UTR3));
    }
    // And with a flank given it is not what omitting the flag means: that
    // designs the codon, so the two molecules are different design spaces.
    let (omitted, _) = molecule(&["--utr3", UTR3]);
    assert_eq!(omitted.stop, Stop::Free);
    assert_ne!(molecule_fields(&omitted), molecule_fields(&none));
}

#[test]
fn with_no_flank_an_omitted_stop_is_none() {
    // A stop codon can only matter through a pair the constraint forbids, and
    // a forbidden pair needs a flank, so the bare default is unchanged.
    let mut argv: Vec<String> = ["--protein", PROTEIN, "--lambda", "0.5"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let bare = design::parse(argv.clone()).unwrap();
    let bare = Molecule::resolve(&bare.shared).unwrap();
    assert_eq!(bare.stop, Stop::Absent);
    argv.extend(["--stop".to_string(), "none".to_string()]);
    let none = design::parse(argv).unwrap();
    let none = Molecule::resolve(&none.shared).unwrap();
    assert_eq!(molecule_fields(&bare), molecule_fields(&none));
}

#[test]
fn without_a_constraint_the_stop_is_the_head_of_the_three_prime_flank() {
    let spelled = format!("UAA{UTR3}");
    let (stop, lambdas) = molecule(&["--utr3", UTR3, "--stop", "UAA", "--forbid", "none"]);
    // The comparison arm spells the codon into the flank itself, so it must ask
    // for no stop codon of its own.
    let (prefixed, _) = molecule(&["--utr3", &spelled, "--forbid", "none", "--stop", "none"]);
    let got = designs(&stop, &lambdas);
    assert_eq!(got, designs(&prefixed, &lambdas));
    for (sequence, ..) in &got {
        let coding = UTR5.len() + 3 * PROTEIN.len();
        assert_eq!(&sequence[coding..coding + 3], "UAA");
        assert!(sequence.ends_with(UTR3));
    }
}

#[test]
fn under_the_constraint_the_stop_is_coding_sequence() {
    let spelled = format!("UAA{UTR3}");
    let (stop, _) = molecule(&["--utr3", UTR3, "--stop", "UAA"]);
    // `--stop none`, so the three nucleotides are the flank's own and the
    // molecule has no stop codon of its own to classify.
    let (prefixed, _) = molecule(&["--utr3", &spelled, "--stop", "none"]);
    assert_eq!(
        stop.constraint, prefixed.constraint,
        "the same classes forbidden"
    );
    let cds = stop.cds();
    let (start, end) = (cds.start as u32, cds.end as u32);
    let (codon, stop_nt, utr3) = (end - 2, end + 1, end + 4);
    let with = stop.table().pair_forbidden(start, end);
    let without = prefixed.table().pair_forbidden(start, end);
    // A codon to the stop: coding-to-coding with it, cds-3 without.
    assert!(!with(codon, stop_nt));
    assert!(without(codon, stop_nt));
    // The stop to the real 3′ UTR: cds-3 with it, within the flank without.
    assert!(with(stop_nt, utr3));
    assert!(!without(stop_nt, utr3));
}

#[test]
fn a_constrained_design_never_pairs_the_stop_into_a_forbidden_class() {
    let (m, lambdas) = molecule(&["--utr3", UTR3, "--stop", "UGA"]);
    let cds = m.cds();
    let forbidden = m.table().pair_forbidden(cds.start as u32, cds.end as u32);
    for s in design::solve(&m, &lambdas, 1).unwrap() {
        let mut open = Vec::new();
        for (j, c) in s.design.structure.chars().enumerate() {
            match c {
                '(' => open.push(j),
                ')' => {
                    let i = open.pop().unwrap();
                    assert!(
                        !forbidden(i as u32, j as u32),
                        "({i}, {j}) at λ={}",
                        s.lambda
                    );
                }
                _ => {}
            }
        }
        assert!(s.checked.is_some(), "a constrained run is checked");
    }
}

#[test]
fn the_stop_survives_the_header() {
    let (m, _) = molecule(&["--utr3", UTR3, "--stop", "uag"]);
    let header = serde_json::Value::Object(molecule_fields(&m));
    assert_eq!(header["stop"], "UAG");
    let back = Molecule::from_header(&header).unwrap();
    assert_eq!(back.stop, Stop::Given("UAG".to_string()));
    assert_eq!(back.table(), m.table());
}

#[test]
fn an_unknown_stop_is_refused() {
    let argv: Vec<String> = ["--protein", PROTEIN, "--lambda", "0.5", "--stop", "UAC"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let args = design::parse(argv).unwrap();
    let e = Molecule::resolve(&args.shared).unwrap_err().to_string();
    // What the message must carry, not the prose carrying it: the value that
    // was refused, so the reader knows which argument is at fault, and every
    // spelling that would have been accepted. Asserting the whole sentence
    // breaks whenever a new spelling is offered, which is when this test
    // most needs to still run.
    assert!(e.contains("UAC"), "the refused value is not named: {e}");
    for legal in ["UAA", "UAG", "UGA", "free", "each", "none"] {
        assert!(e.contains(legal), "{legal} is not offered: {e}");
    }
}
