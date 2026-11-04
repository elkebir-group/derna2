//! The inside–outside codon marginals against one constrained fold per
//! exclusion.
//!
//! [`edge_marginals`] claims two **equalities**: its inside optimum is the
//! production scan's optimum, and its value for excluding codon `c` at position
//! `q` is the optimum of the lattice with `c` removed there. Both are checked
//! here as equalities against [`design_restricted_with_region`], which computes
//! each one by an independent fold. A marginal that came out even slightly above
//! the fold would discard designs if used as a search bound, so the tolerance is
//! float noise and nothing more.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::outside::{edge_marginals, edge_marginals_fast, EdgeMarginals};
use derna_lib::{
    design_restricted_with_region, CodonUsage, Design, Enforcement, InternalLoopMode, PairClass,
    Params, RegionConstraints, ScanConfig, DEFAULT_SINGLE_MAX_LEN, MIN_SINGLE_MAX_LEN,
};

/// Float noise in a sum of a few dozen log-weights, and nothing more.
const TOL: f64 = 1e-6;

struct Case<'a> {
    protein: &'a str,
    utr5: &'a str,
    utr3: &'a str,
    lambda: f64,
    constraints: RegionConstraints,
}

fn tables() -> [(&'static str, RegionConstraints); 3] {
    let five = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
    [
        ("none", RegionConstraints::default()),
        (
            "five-prime+no-clamp",
            five.with(PairClass::Utr5Utr3, Enforcement::Forbid),
        ),
        (
            "symmetric",
            RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        ),
    ]
}

/// The design's objective on the scale the marginals report:
/// `λ·MFE − (1−λ)·100·log CAI`, energies in dekacalories.
fn brd(lambda: f64, d: &Design) -> f64 {
    lambda * (d.mfe_kcal_mol * 100.0) - (1.0 - lambda) * 100.0 * d.cai_log
}

fn full_sets(protein: &str) -> Vec<Vec<[u8; 3]>> {
    protein
        .chars()
        .map(|c| {
            let a = aa_to_index(c).expect("a standard residue");
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

/// Either implementation of the marginals, by its signature.
type Pass = fn(
    &Params,
    &str,
    &CodonUsage,
    f64,
    &[Vec<[u8; 3]>],
    &str,
    &str,
    RegionConstraints,
    ScanConfig,
) -> Result<Option<EdgeMarginals>, derna_lib::BeamError>;

struct Oracle {
    params: Params,
    usage: CodonUsage,
    /// The single-loop size cap every fold and pass here runs at.
    cap: usize,
}

impl Oracle {
    fn new() -> Self {
        Self {
            params: Params::load_default().expect("Turner 2004 parameters"),
            usage: CodonUsage::load_default().expect("the canonical human table"),
            cap: DEFAULT_SINGLE_MAX_LEN,
        }
    }

    fn with_cap(cap: usize) -> Self {
        Self { cap, ..Self::new() }
    }

    fn fold(
        &self,
        case: &Case<'_>,
        allowed: &[Vec<[u8; 3]>],
        mode: InternalLoopMode,
    ) -> Option<f64> {
        design_restricted_with_region(
            &self.params,
            case.protein,
            &self.usage,
            case.lambda,
            allowed,
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig {
                mode,
                single_max_len: self.cap,
                ..ScanConfig::default()
            },
        )
        .expect("a valid instance")
        .map(|d| brd(case.lambda, &d))
    }

    fn marginals_by(
        &self,
        how: Pass,
        case: &Case<'_>,
        allowed: &[Vec<[u8; 3]>],
    ) -> Option<EdgeMarginals> {
        how(
            &self.params,
            case.protein,
            &self.usage,
            case.lambda,
            allowed,
            case.utr5,
            case.utr3,
            case.constraints,
            ScanConfig {
                single_max_len: self.cap,
                ..ScanConfig::default()
            },
        )
        .expect("a valid instance")
    }

    /// Check every claim on one lattice, for both the recorded reference and the
    /// pass over the production tables, returning how many exclusions moved the
    /// optimum — the count a non-vacuity assertion reads.
    fn check(&self, case: &Case<'_>, allowed: &[Vec<[u8; 3]>]) -> usize {
        self.check_pair_outsides(case, allowed);
        let reference = self.check_by("reference", edge_marginals, case, allowed);
        let fast = self.check_by("fast", edge_marginals_fast, case, allowed);
        assert_eq!(reference, fast, "the two passes moved different exclusions");
        fast
    }

    /// Every closed pair's outside value, from both passes, as an equality —
    /// the same pairs and the same values.
    fn check_pair_outsides(&self, case: &Case<'_>, allowed: &[Vec<[u8; 3]>]) {
        let reference = self
            .marginals_by(edge_marginals, case, allowed)
            .expect("the lattice is non-empty");
        let fast = self
            .marginals_by(edge_marginals_fast, case, allowed)
            .expect("the lattice is non-empty");
        let (r, f) = (reference.pair_outsides(), fast.pair_outsides());
        assert_eq!(
            r.len(),
            f.len(),
            "{} λ={}: the passes give {} and {} pairs a finite outside",
            case.protein,
            case.lambda,
            r.len(),
            f.len()
        );
        for (x, y) in r.iter().zip(f) {
            assert!(
                x.0 == y.0 && x.1 == y.1 && (x.2 - y.2).abs() < TOL,
                "{} λ={}: pair outside {x:?} != {y:?}",
                case.protein,
                case.lambda
            );
        }
        let (r, f) = (reference.cds_edges(), fast.cds_edges());
        assert_eq!(r.len(), f.len(), "{}: edge counts differ", case.protein);
        for (x, y) in r.iter().zip(&f) {
            assert!(
                x.0 == y.0 && x.1 == y.1 && (x.2 - y.2).abs() < TOL,
                "{} λ={}: edge marginal {x:?} != {y:?}",
                case.protein,
                case.lambda
            );
        }
    }

    fn check_by(&self, name: &str, how: Pass, case: &Case<'_>, allowed: &[Vec<[u8; 3]>]) -> usize {
        let label = format!(
            "{name}: {} λ={} utr=({},{})",
            case.protein, case.lambda, case.utr5, case.utr3
        );
        let m = self
            .marginals_by(how, case, allowed)
            .expect("the lattice is non-empty");
        for mode in [InternalLoopMode::Lz, InternalLoopMode::Naive] {
            let folded = self
                .fold(case, allowed, mode)
                .expect("the lattice is non-empty");
            assert!(
                (m.optimum() - folded).abs() < TOL,
                "{label} {mode:?}: inside optimum {} != fold {folded}",
                m.optimum()
            );
        }
        let mut moved = 0;
        for (q, codons) in allowed.iter().enumerate() {
            if codons.len() < 2 {
                continue;
            }
            for &c in codons {
                let mut rest = allowed.to_vec();
                rest[q].retain(|&x| x != c);
                let exact = self.fold(case, &rest, InternalLoopMode::Lz);
                let got = m.excluding(q, c);
                match (got, exact) {
                    (Some(g), Some(e)) => {
                        assert!(
                            (g - e).abs() < TOL,
                            "{label}: excluding {c:?} at {q}: marginal {g} != fold {e}"
                        );
                        if g > m.optimum() + TOL {
                            moved += 1;
                        }
                    }
                    (None, None) => {}
                    other => panic!("{label}: excluding {c:?} at {q}: {other:?}"),
                }
            }
        }
        moved
    }
}

/// The inside optimum and every single-codon exclusion, over three proteins,
/// three weights and three constraint tables. Every exclusion is a separate
/// fold, so the proteins are short.
#[test]
fn every_exclusion_equals_its_own_fold() {
    let oracle = Oracle::new();
    let mut moved = 0;
    for (protein, utr5, utr3) in [
        ("LSKF", "GGGAUC", "GAUCCC"),
        ("MRSL", "ACGUAG", "CUACGU"),
        ("GLRV", "", ""),
    ] {
        for lambda in [0.2, 0.5, 0.9] {
            for (_, constraints) in tables() {
                let case = Case {
                    protein,
                    utr5,
                    utr3,
                    lambda,
                    constraints,
                };
                moved += oracle.check(&case, &full_sets(protein));
            }
        }
    }
    assert!(
        moved > 0,
        "no exclusion moved the optimum: the check is vacuous"
    );
}

/// The search's regions are restricted lattices, not whole ones: a position
/// pinned to one codon and another with codons already excluded. The marginals
/// are taken over that region, as a parent's would be.
#[test]
fn exclusions_are_exact_inside_a_restricted_region() {
    let oracle = Oracle::new();
    let protein = "LSRKL";
    let mut allowed = full_sets(protein);
    allowed[1].truncate(1);
    allowed[2].truncate(3);
    allowed[4].remove(0);
    let (_, constraints) = tables()[1];
    let case = Case {
        protein,
        utr5: "GGGAUC",
        utr3: "GAUCCC",
        lambda: 0.5,
        constraints,
    };
    let moved = oracle.check(&case, &allowed);
    assert!(
        moved > 0,
        "no exclusion moved the optimum: the check is vacuous"
    );
}

/// The cases above are too short for any optimum or exclusion to use a
/// multi-loop, so removing that rule from the reference leaves them passing.
/// These two are chosen because it does not: each flank carries a hairpin the
/// other cannot pair with, a closing helix wraps both, and the coding sequence
/// sits inside the loop as its unpaired flank. With the multi-loop rule removed
/// their optima move from −495 to −485 and from −660 to −630 dekacalories, so
/// the rule, and the fill it carries across the coding sequence, bear the answer.
#[test]
fn exclusions_are_exact_where_the_optimum_is_a_multiloop() {
    let oracle = Oracle::new();
    let mut moved = 0;
    for (protein, utr5, utr3) in [
        ("KK", "GAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUC"),
        ("MA", "GGAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUCC"),
    ] {
        for lambda in [0.2, 0.5, 0.9] {
            let case = Case {
                protein,
                utr5,
                utr3,
                lambda,
                constraints: RegionConstraints::default(),
            };
            moved += oracle.check(&case, &full_sets(protein));
        }
    }
    assert!(
        moved > 0,
        "no exclusion moved the optimum: the check is vacuous"
    );
}

/// Two stable hairpins that cannot pair with each other sit in the 5′ flank,
/// and a closing helix joins the flank's first bases to the molecule's last, so
/// the coding sequence is the multi-loop's 3′ unpaired flank — the stretch the
/// scan closes through its shared opening pool rather than one frontier at a
/// time. In the last case the closing strand ends mid-codon (`G` then Leu
/// `CUC`), so the frontier's partial codon carries part of the answer too.
/// Without these cases the pool's contribution to a codon marginal could be
/// deleted with every other test still passing.
#[test]
fn exclusions_are_exact_where_the_coding_sequence_closes_a_multiloop() {
    let oracle = Oracle::new();
    let (a, b) = ("GGGACCUUCGGGUCCC", "CAUGCGGAAACGCAUG");
    let tight = format!("GAGC{a}A{b}");
    let loose = format!("GAGCA{a}AA{b}");
    let mut moved = 0;
    for (protein, utr5, utr3) in [
        ("KKK", tight.as_str(), "GCUC"),
        ("KKK", loose.as_str(), "AGCUC"),
        ("KKKL", tight.as_str(), "A"),
    ] {
        for lambda in [0.5, 0.9] {
            let case = Case {
                protein,
                utr5,
                utr3,
                lambda,
                constraints: RegionConstraints::default(),
            };
            moved += oracle.check(&case, &full_sets(protein));
        }
    }
    assert!(
        moved > 0,
        "no exclusion moved the optimum: the check is vacuous"
    );
}

/// Loops at the size cap are the ones a window bound on the enclosing pair's
/// position can drop by one, and at the default cap of 30 unpaired bases no
/// case short enough to fold once per exclusion has one that matters. Run at
/// the smallest cap the scan accepts, the same cases reach it.
#[test]
fn exclusions_are_exact_at_the_loop_size_cap() {
    let oracle = Oracle::with_cap(MIN_SINGLE_MAX_LEN);
    let mut moved = 0;
    for (protein, utr5, utr3) in [
        ("LSKF", "GGGAUC", "GAUCCC"),
        ("MRSL", "ACGUAG", "CUACGU"),
        ("KK", "GAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUC"),
    ] {
        for lambda in [0.2, 0.5, 0.9] {
            for (_, constraints) in tables() {
                let case = Case {
                    protein,
                    utr5,
                    utr3,
                    lambda,
                    constraints,
                };
                moved += oracle.check(&case, &full_sets(protein));
            }
        }
    }
    assert!(
        moved > 0,
        "no exclusion moved the optimum: the check is vacuous"
    );
}

/// The cases above exercise generic internal loops whose strands cross codon
/// boundaries hundreds of thousands of times, and in none of them is such a
/// loop the best derivation through any coding edge — so the pass's split of
/// those loops into an outer and an inner half could hand its pieces nothing
/// and every assertion would still hold. Here the coding sequence *is* one
/// strand of such a loop: a stable outer helix from the flanks, a hairpin
/// inside it, and four lysine codons, which pair poorly, as the 5′ strand in
/// one case and the 3′ strand in the other — `(12, 5)` and `(5, 12)` loops.
#[test]
fn exclusions_are_exact_where_the_coding_sequence_is_an_internal_loop_strand() {
    let oracle = Oracle::new();
    let mut moved = 0;
    for (utr5, utr3) in [
        ("GGACUC", "GAGCGAAAGCUCAAAAAGAGUCC"),
        ("GGACUCAAAAAGAGCGAAAGCUC", "GAGUCC"),
    ] {
        for lambda in [0.2, 0.5, 0.9] {
            let case = Case {
                protein: "KKKK",
                utr5,
                utr3,
                lambda,
                constraints: RegionConstraints::default(),
            };
            moved += oracle.check(&case, &full_sets("KKKK"));
        }
    }
    assert!(
        moved > 0,
        "no exclusion moved the optimum: the check is vacuous"
    );
}
