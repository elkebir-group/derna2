//! What a design must satisfy whatever the optimum is.
//!
//! The equalities compare this command against the binary it replaces, which
//! cannot catch an error the two share — they run the same lattice. These
//! properties are about the answer's own shape, so they hold against no
//! reference at all:
//!
//! - the emitted molecule is the fixed 5′ UTR, then exactly three nucleotides
//!   per residue, then the fixed 3′ UTR;
//! - translating those codons gives the requested protein back, read through
//!   the run's own codon table;
//! - the structure is a balanced dot-bracket of the molecule's length, and
//!   every pair it forms is one the energy model allows;
//! - CAI lies in (0, 1], and the accumulated log-CAI is ≤ 0 with equality
//!   exactly when CAI is 1;
//! - under a constraint the check's gap is non-negative — the constrained
//!   optimum cannot beat the free one.
//!
//! The proteins are fixed rather than random: a test that picks its own inputs
//! each run reports a different thing each time it fails.

use derna_cli::design;
use derna_cli::free_check::Checked;
use derna_cli::molecule::Molecule;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// Proteins chosen to cover the awkward residues rather than at random:
/// methionine and tryptophan have one codon each, leucine and serine have six,
/// and a run of the same residue exercises a repeated codon choice.
const PROTEINS: [&str; 5] = ["MW", "MLLLLL", "MSSSSS", "MWFYCGP", "MKVLAKGIVSFA"];
const WEIGHTS: [&str; 3] = ["0.05", "0.5", "0.95"];

/// `codon -> amino acid`, from the run's own table: the design is scored
/// against that table, so it is the one that says what a codon means here.
fn translation(m: &Molecule) -> HashMap<String, char> {
    m.codons
        .rows
        .iter()
        .map(|(aa, codon, _)| (codon.clone(), aa.chars().next().expect("an amino acid")))
        .collect()
}

fn balanced(structure: &str) -> bool {
    let mut open = 0i64;
    for c in structure.chars() {
        match c {
            '(' => open += 1,
            ')' => open -= 1,
            '.' => {}
            _ => return false,
        }
        if open < 0 {
            return false;
        }
    }
    open == 0
}

/// Every pair the structure forms, as `(i, j)` indices.
fn pairs(structure: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut found = Vec::new();
    for (i, c) in structure.chars().enumerate() {
        match c {
            '(' => stack.push(i),
            ')' => found.push((stack.pop().expect("balanced"), i)),
            _ => {}
        }
    }
    found
}

#[test]
fn a_design_satisfies_its_own_shape_whatever_the_optimum_is() {
    let (utr5, utr3) = ("ACAUUUGCUU", "GGCCAAUUGG");
    for protein in PROTEINS {
        for weight in WEIGHTS {
            let argv = words(&[
                "--protein",
                protein,
                "--lambda",
                weight,
                "--utr5",
                utr5,
                "--utr3",
                utr3,
            ]);
            let args = design::parse(argv).unwrap();
            let m = Molecule::resolve(&args.shared).unwrap();
            let solved = design::solve(&m, &args.lambdas, 1).unwrap();
            let design = &solved[0].design;
            let codon_of = translation(&m);
            let case = format!("{protein} at λ={weight}");

            // The molecule is the flanks around three nucleotides per residue
            // and a designed stop codon: the flanks are given and no `--stop`
            // is, so the search chooses the coding sequence's last codon among
            // the three stops.
            let coding = 3 * protein.len();
            assert_eq!(
                design.sequence.len(),
                utr5.len() + coding + 3 + utr3.len(),
                "{case}: length"
            );
            let stop = &design.sequence[utr5.len() + coding..utr5.len() + coding + 3];
            assert!(
                ["UAA", "UAG", "UGA"].contains(&stop),
                "{case}: coding sequence ends in {stop}"
            );
            assert!(design.sequence.starts_with(utr5), "{case}: 5′ UTR");
            assert!(design.sequence.ends_with(utr3), "{case}: 3′ UTR");

            // Those codons translate back to the protein that was asked for.
            let cds = &design.sequence[utr5.len()..utr5.len() + coding];
            let read: String = (0..protein.len())
                .map(|i| {
                    let codon = &cds[3 * i..3 * i + 3];
                    *codon_of
                        .get(codon)
                        .unwrap_or_else(|| panic!("{case}: {codon} is not a sense codon"))
                })
                .collect();
            assert_eq!(
                read, protein,
                "{case}: the design does not encode the protein"
            );

            // The structure is a balanced dot-bracket over the whole molecule,
            // and every pair it forms is one the model allows.
            assert_eq!(
                design.structure.len(),
                design.sequence.len(),
                "{case}: structure length"
            );
            assert!(balanced(&design.structure), "{case}: {}", design.structure);
            let bases: Vec<char> = design.sequence.chars().collect();
            for (i, j) in pairs(&design.structure) {
                let pair = (bases[i], bases[j]);
                let allowed = matches!(
                    pair,
                    ('A', 'U') | ('U', 'A') | ('C', 'G') | ('G', 'C') | ('G', 'U') | ('U', 'G')
                );
                assert!(
                    allowed,
                    "{case}: {pair:?} at ({i}, {j}) is not a pair the model allows"
                );
                assert!(
                    j > i + 3,
                    "{case}: ({i}, {j}) is a hairpin shorter than the minimum"
                );
            }

            // CAI is a geometric mean of ratios to the best codon, so it lies
            // in (0, 1], and its log is ≤ 0 with equality only at 1.
            assert!(
                design.cai > 0.0 && design.cai <= 1.0 + 1e-12,
                "{case}: cai {}",
                design.cai
            );
            assert!(
                design.cai_log <= 1e-12,
                "{case}: cai_log {}",
                design.cai_log
            );
            assert_eq!(
                design.cai_log.abs() < 1e-12,
                (design.cai - 1.0).abs() < 1e-12,
                "{case}: cai and cai_log disagree about the best possible adaptation"
            );

            // Under a constraint, the constrained optimum cannot beat the free
            // one, so the gap cannot be negative.
            match solved[0]
                .checked
                .as_ref()
                .expect("a constrained run is checked")
            {
                Checked::Exact(gap) => assert!(
                    gap.gap_kcal_mol >= -1e-9,
                    "{case}: a negative gap {}",
                    gap.gap_kcal_mol
                ),
                Checked::Refuted { gap_at_least, .. } => {
                    assert!(*gap_at_least > 0.0, "{case}: a refutation with no gap")
                }
            }
        }
    }
}

#[test]
fn a_weight_nearer_one_does_not_prefer_adaptation() {
    // λ weights the energy against the adaptation, so raising it cannot make
    // the design *more* adapted at a worse energy: along the weights the MFE
    // is non-increasing and the CAI non-increasing with it. This is a property
    // of the objective, and it is what makes a front a front.
    let fasta = root().join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta");
    let argv = words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.05,0.3,0.6,0.95",
    ]);
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let solved = design::solve(&m, &args.lambdas, 1).unwrap();
    for pair in solved.windows(2) {
        let (lower, higher) = (&pair[0], &pair[1]);
        assert!(
            higher.design.mfe_kcal_mol <= lower.design.mfe_kcal_mol + 1e-9,
            "λ={} gave a worse energy ({:+.2}) than λ={} ({:+.2})",
            higher.lambda,
            higher.design.mfe_kcal_mol,
            lower.lambda,
            lower.design.mfe_kcal_mol
        );
        assert!(
            higher.design.cai_log <= lower.design.cai_log + 1e-9,
            "λ={} gave a better adaptation than λ={}, which the objective forbids",
            higher.lambda,
            lower.lambda
        );
    }
}

/// The default constraint is the two classes that reach the coding sequence,
/// and leaving `5-3` out of it changes what a run returns.
///
/// Two halves, and the second is the one that can rot: the first pins the
/// default to the list it stands for, and the second shows that list is not
/// the same as all three — on complementary flanks, where a flank-to-flank
/// pair is there to be forbidden. Without it, a default that quietly went
/// back to forbidding every class would still pass the first half on any
/// molecule whose flanks never pair with each other.
#[test]
fn the_default_forbids_the_classes_that_reach_the_coding_sequence() {
    // Reverse complements of each other, so `5-3` is a class with something
    // in it rather than one that is empty whatever the default says.
    let (utr5, utr3) = ("GGGCUUCGAAGCCC", "GGGCUUCGAAGCCC");
    let design_at = |forbid: Option<&str>| {
        let mut parts = vec![
            "--protein",
            "MKVLAKGIVSFA",
            "--lambda",
            "0.5",
            "--utr5",
            utr5,
            "--utr3",
            utr3,
        ];
        if let Some(list) = forbid {
            parts.extend(["--forbid", list]);
        }
        let args = design::parse(words(&parts)).unwrap();
        let m = Molecule::resolve(&args.shared).unwrap();
        let solved = design::solve(&m, &args.lambdas, 1).unwrap();
        let design = &solved[0].design;
        (
            m.constraint.forbid_names(),
            design.sequence.clone(),
            design.structure.clone(),
        )
    };
    let (names, _, _) = design_at(None);
    assert_eq!(names, ["5-cds", "cds-3"], "the default's classes");

    let (_, by_default, structure_by_default) = design_at(None);
    let (_, explicit, structure_explicit) = design_at(Some("5-cds,cds-3"));
    assert_eq!(
        by_default, explicit,
        "the default is the list it stands for"
    );
    assert_eq!(structure_by_default, structure_explicit);

    let (names_all, _, structure_all) = design_at(Some("5-cds,cds-3,5-3"));
    assert_eq!(names_all, ["5-cds", "cds-3", "5-3"]);
    assert_ne!(
        structure_by_default, structure_all,
        "forbidding 5-3 must change the structure these flanks fold into, or \
         this case cannot tell the two defaults apart"
    );
}
