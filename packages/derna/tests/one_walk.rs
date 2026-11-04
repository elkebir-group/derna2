//! The motif walk with an empty motif set, or one that removes nothing from
//! the protein's lattice, is the plain walk: one walk serves both, and it
//! takes the plain lattice's scan whenever the product lattice removed no
//! edge. Checked under the options `derna front` walks with
//! ([`front::walk_run`]), not the library's defaults.

use derna_cli::front;
use derna_cli::molecule::Molecule;
use derna_lib::motif_front::{motif_front_walk_run, MotifFrontSpec, MotifRun};
use derna_lib::{FrontPoint, MotifWindow, Params, UtrSpec, WalkRun};

const PROTEIN: &str = "MKVLAKGIVLAGQWERS";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";

fn molecule() -> Molecule {
    let words = [
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--forbid",
        "5-cds,cds-3",
        // A fixed stop: with flanks and no stop given, the plain walk designs
        // one, and a motif walk refuses a designed stop.
        "--stop",
        "UAA",
    ];
    let args = front::parse(words.iter().map(|w| w.to_string())).unwrap();
    Molecule::resolve(&args.shared).unwrap()
}

fn fixed(spec: UtrSpec) -> String {
    match spec {
        UtrSpec::Fixed(s) => s,
        _ => unreachable!("the molecule's flanks are fixed"),
    }
}

/// The front as `(sequence, mfe, cai_log)` in emission order, compared with
/// no tolerance.
fn designs(front: &[FrontPoint]) -> Vec<(String, f64, f64)> {
    front
        .iter()
        .map(|p| (p.sequence.clone(), p.mfe_kcal_mol, p.cai_log))
        .collect()
}

fn motif_walk(m: &Molecule, motifs: &[&str], threads: usize) -> Vec<(String, f64, f64)> {
    let (utr5, utr3) = (fixed(m.utr5_spec()), fixed(m.utr3_spec()));
    let placed: Vec<(&str, MotifWindow)> =
        motifs.iter().map(|m| (*m, MotifWindow::all())).collect();
    let spec = MotifFrontSpec {
        utr5: &utr5,
        utr3: &utr3,
        constraints: m.table(),
        motifs: &placed,
        pins: None,
    };
    let walked = motif_front_walk_run(
        &Params::load_default().unwrap(),
        &m.protein.sequence,
        &m.codons.usage,
        &spec,
        threads,
        MotifRun::default(),
    )
    .unwrap();
    assert!(walked.open.is_empty());
    designs(&walked.front)
}

fn plain_walk(m: &Molecule, threads: usize) -> Vec<(String, f64, f64)> {
    let walked = front::walk_run(m, threads, Default::default(), WalkRun::default()).unwrap();
    assert!(walked.open.is_empty());
    designs(&walked.front)
}

#[test]
fn an_empty_motif_set_walks_the_plain_front() {
    let m = molecule();
    let plain = plain_walk(&m, 1);
    assert!(plain.len() > 2, "a front with gaps to walk");
    assert_eq!(motif_walk(&m, &[], 1), plain);
}

#[test]
fn a_motif_set_that_removes_nothing_walks_the_plain_front() {
    let m = molecule();
    // Longer than the molecule, so no design can spell it.
    let absent = "A".repeat(200);
    assert_eq!(motif_walk(&m, &[&absent], 1), plain_walk(&m, 1));
}
