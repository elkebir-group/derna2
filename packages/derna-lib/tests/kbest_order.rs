//! Best-first enumeration against brute force.
//!
//! The enumerator's whole value is that the *first* candidate passing a
//! predicate is provably optimal among passing candidates. That claim rests
//! entirely on two properties of the walk, and neither is self-evident from the
//! partitioning code:
//!
//! 1. **Order** — designs come out in non-decreasing objective.
//! 2. **Coverage** — every encoding appears exactly once: none skipped (which
//!    would let a better passing design go unseen) and none repeated (which
//!    would inflate the reported rank).
//!
//! Both are checked against exhaustive enumeration on proteins small enough to
//! enumerate completely, at several λ so the objective's shape varies.

use std::collections::HashSet;

use derna_codon_dp::cai::{compute_cai_tables, score_cai};
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{CodonUsage, Enumerator, Params, Problem};

const NUC: [u8; 4] = [b'A', b'C', b'G', b'U'];
const EXACT: usize = 1 << 20;
/// Objectives are integer dekacalories under the hood; this absorbs the float
/// division only.
const TOL: f64 = 1e-6;

fn setup() -> (Params, CodonUsage) {
    (
        Params::load_default().expect("params"),
        CodonUsage::load_default().expect("usage"),
    )
}

fn indices(protein: &str) -> Vec<usize> {
    protein.chars().map(|c| aa_to_index(c).unwrap()).collect()
}

/// Every encoding of `aa`, as flat nucleotide-index sequences.
fn all_encodings(aa: &[usize]) -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new()];
    for &a in aa {
        let mut next = Vec::new();
        for prefix in &out {
            for codon in AMINO_ACIDS[a].codons {
                let mut e = prefix.clone();
                e.extend_from_slice(&codon.nuc);
                next.push(e);
            }
        }
        out = next;
    }
    out
}

fn spell(utr5: &str, cds: &[u8], utr3: &str) -> String {
    let mut s = String::from(utr5);
    s.extend(cds.iter().map(|&n| NUC[n as usize] as char));
    s.push_str(utr3);
    s
}

/// `(objective, whole-mRNA sequence)` for every encoding, ascending.
fn brute_force(
    params: &Params,
    usage: &CodonUsage,
    aa: &[usize],
    lambda: f64,
    utr5: &str,
    utr3: &str,
) -> Vec<(f64, String)> {
    let cai = compute_cai_tables(usage);
    let mut out: Vec<(f64, String)> = all_encodings(aa)
        .into_iter()
        .map(|cds| {
            let seq = spell(utr5, &cds, utr3);
            let (_, energy) = derna_lib::linearfold_fold(params, seq.as_bytes(), EXACT).unwrap();
            let cai_log = score_cai(&cds, aa, &cai.log_cai).unwrap();
            let mfe_dk = (energy.to_kcal_mol() * 100.0).round();
            let cai_dk = (cai_log * 100.0).round();
            (lambda * mfe_dk - (1.0 - lambda) * cai_dk, seq)
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

fn enumerate_all(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    aa: &[usize],
    lambda: f64,
    utr5: &str,
    utr3: &str,
) -> Vec<(f64, String)> {
    let problem = Problem {
        protein,
        aa,
        usage,
        lambda,
        utr5,
        utr3,
        threads: 4,
    };
    let mut e = Enumerator::new(params, problem)
        .expect("enumerator builds")
        .expect("protein has encodings");
    let mut out = Vec::new();
    while let Some(d) = e.next_design().expect("walk advances") {
        let mfe_dk = (d.mfe_kcal_mol * 100.0).round();
        let cai_dk = (d.cai_log * 100.0).round();
        out.push((lambda * mfe_dk - (1.0 - lambda) * cai_dk, d.sequence));
    }
    out
}

/// The walk reproduces exhaustive enumeration: same count, same objective
/// sequence, no repeats, and every encoding present.
fn assert_matches_brute_force(protein: &str, utr5: &str, utr3: &str, lambda: f64) {
    let (params, usage) = setup();
    let aa = indices(protein);
    let want = brute_force(&params, &usage, &aa, lambda, utr5, utr3);
    let got = enumerate_all(&params, &usage, protein, &aa, lambda, utr5, utr3);
    let label = format!("{protein} λ={lambda} utr=({utr5},{utr3})");

    assert_eq!(
        got.len(),
        want.len(),
        "[{label}] the walk must emit every encoding exactly once"
    );
    for (rank, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g.0 - w.0).abs() < TOL,
            "[{label}] rank {rank}: objective {} != brute-force {}",
            g.0,
            w.0
        );
    }
    let distinct: HashSet<&String> = got.iter().map(|(_, s)| s).collect();
    assert_eq!(
        distinct.len(),
        got.len(),
        "[{label}] no sequence may be enumerated twice"
    );
    let wanted: HashSet<&String> = want.iter().map(|(_, s)| s).collect();
    assert_eq!(distinct, wanted, "[{label}] the same set of encodings");
}

/// Order and coverage, CDS-only, across λ. `MK` has 2×2 encodings, `MKT` 2×2×4,
/// `LR` 6×6 — the last exercising two six-fold-degenerate positions, where the
/// prefix-group lanes are widest.
#[test]
fn walk_reproduces_brute_force_cds_only() {
    for lambda in [0.0001, 0.5, 0.99] {
        assert_matches_brute_force("MK", "", "", lambda);
        assert_matches_brute_force("MKT", "", "", lambda);
    }
    assert_matches_brute_force("LR", "", "", 0.5);
}

/// The same with flanks present: the UTR is folded but not designed, so it
/// changes the objective without changing the space being enumerated.
#[test]
fn walk_reproduces_brute_force_with_flanks() {
    assert_matches_brute_force("MKT", "GGGACC", "UUUAAA", 0.5);
    assert_matches_brute_force("MKT", "GCCACCAUGG", "", 0.99);
}

/// A lattice pinned to one sequence must fold that sequence and nothing else:
/// same MFE as an independent fold, and the **same CAI at every λ**, since a
/// fixed sequence has one adaptation.
///
/// The regression this constraints: carrying a real CAI weight on a single-codon
/// unit lets the broken-codon cost-to-go credit adaptation to a parse that never
/// spells the codon, so the solver buys the phantom credit with a positive-energy
/// hairpin — visible only at λ small enough for CAI to dominate. It stayed hidden
/// while the only single-codon path zeroed its weights.
#[test]
fn a_pinned_lattice_folds_one_sequence_at_every_lambda() {
    let (params, usage) = setup();
    let protein = "MKT";
    // AUG AAA ACU — deliberately not the max-CAI codon at either free position,
    // so a phantom credit would be strictly attractive.
    let pinned = vec![vec![[0u8, 3, 2]], vec![[0, 0, 0]], vec![[0, 1, 3]]];
    let mut seen: Option<(String, f64)> = None;

    for lambda in [1.0, 0.5, 0.0001] {
        let d = derna_lib::design_restricted(
            &params,
            protein,
            &usage,
            lambda,
            &pinned,
            "",
            "",
            derna_lib::ScanConfig::default(),
        )
        .expect("solves")
        .expect("the pinned lattice is feasible");

        let (_, energy) =
            derna_lib::linearfold_fold(&params, d.sequence.as_bytes(), EXACT).expect("folds");
        assert!(
            (d.mfe_kcal_mol - energy.to_kcal_mol()).abs() < TOL,
            "λ={lambda}: pinned MFE {} must equal an independent fold {}",
            d.mfe_kcal_mol,
            energy.to_kcal_mol()
        );
        match &seen {
            None => seen = Some((d.sequence.clone(), d.cai_log)),
            Some((seq, cai)) => {
                assert_eq!(seq, &d.sequence, "λ={lambda}: the pinned sequence is fixed");
                assert!(
                    (d.cai_log - cai).abs() < TOL,
                    "λ={lambda}: one sequence has one CAI, got {} vs {cai}",
                    d.cai_log
                );
            }
        }
    }
}

/// Restricting every position to *all* of its codons is the unrestricted design,
/// exactly — the floor that says the restriction machinery adds nothing when it
/// restricts nothing.
#[test]
fn restricting_to_everything_reproduces_the_plain_design() {
    let (params, usage) = setup();
    for protein in ["MKT", "LR", "MKTAYIAKQR"] {
        let all: Vec<Vec<[u8; 3]>> = indices(protein)
            .iter()
            .map(|&a| AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect())
            .collect();
        for lambda in [1.0, 0.5, 0.0001] {
            let r = derna_lib::design_restricted(
                &params,
                protein,
                &usage,
                lambda,
                &all,
                "",
                "",
                derna_lib::ScanConfig::default(),
            )
            .expect("solves")
            .expect("feasible");
            let b = derna_lib::design(&params, protein, &usage, lambda).expect("designs");
            assert_eq!(r.sequence, b.sequence, "[{protein} λ={lambda}] same design");
            assert!((r.mfe_kcal_mol - b.mfe_kcal_mol).abs() < TOL);
            assert!((r.cai_log - b.cai_log).abs() < TOL);
        }
    }
}

/// The first design satisfying a predicate is optimal among those satisfying it
/// — the property the whole search exists for. Checked against brute force with
/// an arbitrary predicate over the sequence.
#[test]
fn first_accepted_is_optimal_among_accepted() {
    let (params, usage) = setup();
    let (protein, lambda) = ("MKT", 0.5);
    let aa = indices(protein);
    // A predicate with no relation to the objective, so it cannot be satisfied
    // by accident of ordering.
    let accept = |seq: &str| seq.contains("ACG");

    let want = brute_force(&params, &usage, &aa, lambda, "", "")
        .into_iter()
        .find(|(_, s)| accept(s))
        .expect("some encoding contains ACG");
    let got = enumerate_all(&params, &usage, protein, &aa, lambda, "", "")
        .into_iter()
        .find(|(_, s)| accept(s))
        .expect("the walk finds it too");

    assert!(
        (got.0 - want.0).abs() < TOL,
        "first accepted objective {} must equal the best accepted {}",
        got.0,
        want.0
    );
}
