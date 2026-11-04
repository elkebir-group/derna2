//! Region-pair constraint oracle.
//!
//! One shared constraint on a base pair's two endpoint *positions* serves both
//! constraint families — CDS↔UTR isolation and the start-proximal accessibility
//! window — in hard (forbid) and soft (penalize) variants. This oracle pins the
//! constraint byte-exact against a genuinely **constrained** brute force, and proves
//! the constraint composes with the substrate's interior-loop reductions.
//!
//! Five checks:
//!
//! 1. **Constrained-fold oracle (not fold-then-filter).** For each `(protein,
//!    UTR, constraints, λ)`, enumerate every CDS encoding, spell `utr5 + CDS +
//!    utr3`, enumerate **all** nested canonical structures (min hairpin 3), drop
//!    those violating a hard constraint (add the soft penalty to the rest), score each
//!    survivor with [`evaluate`], take the per-encoding constrained MFE, form BRD
//!    with the CDS-only CAI, and argmin over encodings. The design's BRD matches
//!    within the objective tolerance (`BRD_TOL`) — a hard constraint's optimum is
//!    frequently a structure that is *suboptimal* unconstrained, so folding under
//!    the constraint (not filtering afterward) is essential.
//! 2. **Phantom-free constrained MFE (byte-exact).** The design's reported MFE
//!    equals the plain re-score of its own recovered structure plus the soft
//!    penalties that structure incurs — so the penalty is credited exactly once.
//! 3. **The constraint bites.** A tracked count of cases where the constrained optimum
//!    differs from the unconstrained one is asserted non-zero, so the oracle is
//!    not vacuously green.
//! 4. **Structural post-conditions.** Hard isolation: no recovered pair crosses
//!    the CDS/UTR boundary. Hard window: every window position is unpaired.
//! 5. **Reduction composition.** The constrained fold under the Lyngsø–Zuker /
//!    one-sided / `multi_open` reductions ([`InternalLoopMode::Lz`]) equals the
//!    un-reduced constrained fold ([`InternalLoopMode::Naive`]) byte-exactly at
//!    lengths that engage every reduction — the Scope-17 composition trap, proven
//!    rather than assumed.

use std::collections::HashMap;

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_with_utr, linearfold_fold, AccessibilityWindow, CodonUsage, Design, Enforcement,
    InternalLoopMode, PairClass, Params, RegionConstraints, ScanConfig,
};
use rna_mfe::evaluate;

/// One secondary structure as its list of `(i, j)` base pairs.
type Pairs = Vec<(usize, usize)>;
/// A set of structures over one interval.
type Structs = Vec<Pairs>;
/// Per-interval memo for the structure enumeration.
type Memo = HashMap<(usize, usize), Structs>;

/// b → ∞: a beam wide enough that `rna-linearfold` runs the exact lattice.
const EXACT: usize = 1 << 20;

/// BRD tolerance in dekacalories — absorbs the rounded-integer vs continuous
/// objective gap, exactly as `tests/brd_parity.rs`.
const BRD_TOL: f64 = 0.5;

/// The λ values swept by every parity check (low / mid / high).
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

fn nuc_byte(n: u8) -> u8 {
    [b'A', b'C', b'G', b'U'][n as usize]
}

fn protein_indices(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}

/// Fixed-λ BRD on rounded dekacalories, matching how the oracle and the design
/// both round at the scalarization boundary (`tests/brd_parity.rs`).
fn brd(lambda: f64, mfe_kcal: f64, cai_log: f64) -> f64 {
    let mfe_dk = (mfe_kcal * 100.0).round();
    let cai_dk = (cai_log * 100.0).round();
    lambda * mfe_dk - (1.0 - lambda) * cai_dk
}

/// Whether ASCII nucleotides `a` and `b` form a canonical Watson–Crick / wobble
/// pair (the same six pairs `rna-mfe`'s `pair_from_nucleotides` admits).
fn canonical(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'C') | (b'C', b'G') | (b'G', b'U') | (b'U', b'G')
    )
}

/// Every CDS encoding of `aa` as per-residue codon choices (mixed-radix over
/// synonym counts).
fn encodings(aa: &[usize]) -> Vec<Vec<usize>> {
    let counts: Vec<usize> = aa.iter().map(|&a| AMINO_ACIDS[a].codons.len()).collect();
    let total: usize = counts.iter().product();
    (0..total)
        .map(|mut n| {
            counts
                .iter()
                .map(|&c| {
                    let ci = n % c;
                    n /= c;
                    ci
                })
                .collect()
        })
        .collect()
}

/// The whole-mRNA bytes for one CDS encoding: `utr5 + CDS + utr3`.
fn full_bytes(aa: &[usize], encoding: &[usize], utr5: &str, utr3: &str) -> Vec<u8> {
    let mut out = utr5.as_bytes().to_vec();
    for (&a, &ci) in aa.iter().zip(encoding) {
        out.extend(AMINO_ACIDS[a].codons[ci].nuc.iter().map(|&n| nuc_byte(n)));
    }
    out.extend_from_slice(utr3.as_bytes());
    out
}

/// CDS-only additive log-CAI of one encoding (the same `EPSILON`-floored formula
/// the DFA's edge weights use).
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

/// Enumerate every nested canonical secondary structure of `seq[lo..hi)` with
/// minimum hairpin loop 3, memoized per interval. Each structure is a list of
/// `(i, j)` pairs. This is the exact structure space the CKY grammar folds over,
/// so a min over it reproduces `rna-linearfold`'s energy (asserted separately).
fn structures(seq: &[u8], lo: usize, hi: usize, memo: &mut Memo) -> Structs {
    if lo >= hi {
        return vec![Vec::new()];
    }
    if let Some(cached) = memo.get(&(lo, hi)) {
        return cached.clone();
    }
    let mut out: Vec<Vec<(usize, usize)>> = Vec::new();
    // `lo` unpaired.
    for s in structures(seq, lo + 1, hi, memo) {
        out.push(s);
    }
    // `lo` pairs with `k` (min hairpin 3 ⇒ k ≥ lo + 4).
    for k in (lo + 4)..hi {
        if !canonical(seq[lo], seq[k]) {
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

/// All nested canonical structures of the whole sequence `seq`.
fn all_structures(seq: &[u8]) -> Structs {
    let mut memo = Memo::new();
    structures(seq, 0, seq.len(), &mut memo)
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

/// The pairs of a dot-bracket structure (a balanced-parens stack walk).
fn pairs_of(structure: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut pairs = Vec::new();
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => pairs.push((stack.pop().expect("balanced"), i)),
            _ => {}
        }
    }
    pairs
}

/// The resolved region spec for one `(constraints, cds window)`: the independent
/// re-implementation of the constraint predicate the oracle scores against (kept
/// separate from the crate's `PairConstraint` so a shared bug cannot hide).
struct Spec {
    cds_start: u32,
    cds_end: u32,
    /// Enforcement per class, keyed by the oracle's **own** region codes so the
    /// crate's table indexing is never shared with it.
    classes: Vec<((u8, u8), Enforcement)>,
    /// Resolved absolute window `[lo, hi)` and its enforcement.
    window: Option<(u32, u32, Enforcement)>,
}

impl Spec {
    fn new(rc: &RegionConstraints, cds_start: u32, cds_end: u32) -> Self {
        let window = rc.accessibility.map(|w| {
            let lo = cds_start.saturating_sub(w.upstream);
            (lo, cds_start + w.downstream, w.enforcement)
        });
        // The configuration is read through the public accessor; the region
        // codes it is re-keyed on, and every predicate below, are the oracle's.
        let code = |c: PairClass| match c {
            PairClass::Utr5Utr5 => (0, 0),
            PairClass::Utr5Cds => (0, 1),
            PairClass::Utr5Utr3 => (0, 2),
            PairClass::CdsCds => (1, 1),
            PairClass::CdsUtr3 => (1, 2),
            PairClass::Utr3Utr3 => (2, 2),
        };
        let classes = PairClass::ALL
            .iter()
            .filter_map(|&c| rc.enforcement(c).map(|e| (code(c), e)))
            .collect();
        Self {
            cds_start,
            cds_end,
            classes,
            window,
        }
    }

    /// Region code for a position: 0 = 5′ UTR, 1 = CDS, 2 = 3′ UTR.
    fn region_code(&self, p: usize) -> u8 {
        let p = p as u32;
        if p < self.cds_start {
            0
        } else if p < self.cds_end {
            1
        } else {
            2
        }
    }

    /// How the class table enforces the pair `(i, j)`, if at all.
    fn class_enforcement(&self, i: usize, j: usize) -> Option<Enforcement> {
        let (a, b) = (self.region_code(i), self.region_code(j));
        let key = (a.min(b), a.max(b));
        self.classes
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, e)| *e)
    }

    /// `(hard-forbidden, soft-penalty-dekacal)` for one pair `(i, j)`.
    fn pair(&self, i: usize, j: usize) -> (bool, i32) {
        let mut forbid = false;
        let mut pen = 0i32;
        let mut apply = |enf: Enforcement| match enf {
            Enforcement::Forbid => forbid = true,
            Enforcement::Penalize(w) => pen += w.dekacal(),
        };
        if let Some(enf) = self.class_enforcement(i, j) {
            apply(enf);
        }
        if let Some((lo, hi, enf)) = self.window {
            let in_win = |p: u32| p >= lo && p < hi;
            if in_win(i as u32) || in_win(j as u32) {
                apply(enf);
            }
        }
        (forbid, pen)
    }

    /// `(any-hard-violation, total-soft-penalty-dekacal)` for a whole structure.
    fn structure(&self, pairs: &[(usize, usize)]) -> (bool, i32) {
        let mut forbid = false;
        let mut pen = 0i32;
        for &(i, j) in pairs {
            let (f, p) = self.pair(i, j);
            forbid |= f;
            pen += p;
        }
        (forbid, pen)
    }
}

/// The constrained MFE (dekacalories) of one spelled sequence under `spec`: the
/// minimum over every nested structure of `evaluate + soft-penalty`, dropping
/// hard-violating structures. Always defined — the all-unpaired structure is
/// feasible for every region constraint (so a region constraint has no infeasibility path).
fn constrained_mfe_dk(params: &Params, full: &[u8], spec: &Spec) -> i32 {
    let mut best: Option<i32> = None;
    for s in all_structures(full) {
        let (forbid, pen) = spec.structure(&s);
        if forbid {
            continue;
        }
        let db = dot_bracket(full.len(), &s);
        let e = evaluate(params, full, &db)
            .expect("structure scores")
            .dekacal();
        let total = e + pen;
        best = Some(best.map_or(total, |b| b.min(total)));
    }
    best.expect("all-unpaired structure is always feasible")
}

/// One region-pair oracle case: the design must match the constrained brute
/// force, be phantom-free, and (for hard families) satisfy the structural
/// post-conditions. Returns `1` if the constraint **bit** (the constrained
/// optimum differs from the unconstrained one) at any swept λ, else `0`.
fn check_case(protein: &str, utr5: &str, utr3: &str, rc: RegionConstraints) -> u32 {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let aa = protein_indices(protein);
    let cds_start = utr5.len() as u32;
    let cds_end = cds_start + 3 * aa.len() as u32;
    let spec = Spec::new(&rc, cds_start, cds_end);

    // Per-encoding constrained (MFE, CAI) — both λ-independent, so precompute
    // once (a fixed encoding's best structure minimizes MFE for every λ > 0).
    let scored: Vec<(f64, f64)> = encodings(&aa)
        .iter()
        .map(|enc| {
            let full = full_bytes(&aa, enc, utr5, utr3);
            (
                f64::from(constrained_mfe_dk(&params, &full, &spec)) / 100.0,
                cds_cai_log(&usage, &aa, enc),
            )
        })
        .collect();

    let mut bit = 0u32;
    for &lambda in &LAMBDAS {
        let label = format!("{protein} 5′={utr5:?} 3′={utr3:?} λ={lambda} {rc:?}");
        let oracle = scored
            .iter()
            .map(|&(mfe, cai)| brd(lambda, mfe, cai))
            .fold(f64::INFINITY, f64::min);

        let d = design_with_utr(
            &params,
            protein,
            &usage,
            lambda,
            utr5,
            utr3,
            rc,
            ScanConfig::default(),
        )
        .unwrap();
        let full = d.sequence.as_bytes().to_vec();
        let reported_dk = (d.mfe_kcal_mol * 100.0).round() as i32;

        // (1) Constrained-fold optimality, both directions: the design cannot
        // beat the true constrained optimum (that would mean it violated the
        // constraint or the oracle missed a structure), nor fall short of it.
        let design_brd = brd(lambda, d.mfe_kcal_mol, d.cai_log);
        assert!(
            (design_brd - oracle).abs() <= BRD_TOL,
            "constrained-oracle mismatch [{label}]: design={design_brd:.1} oracle={oracle:.1}",
        );

        // (2) Phantom-free constrained MFE: the reported energy is the modular
        // re-score of the recovered structure plus its soft penalties, so a soft
        // penalty is credited exactly once.
        let recovered = pairs_of(&d.structure);
        let (forbid, pen) = spec.structure(&recovered);
        assert!(
            !forbid,
            "design emitted a hard-forbidden pair [{label}]: {}",
            d.structure
        );
        let reeval = evaluate(&params, &full, &d.structure)
            .expect("scores")
            .dekacal();
        assert_eq!(
            reported_dk,
            reeval + pen,
            "phantom constrained MFE [{label}]: reported={reported_dk} reeval={reeval} pen={pen}",
        );

        // (4) Structural post-conditions for the hard families.
        assert_hard_postconditions(&d, &spec, &recovered, &label);

        // (3) Bite tracking: constrained optimum differs from the unconstrained.
        let unconstrained = design_with_utr(
            &params,
            protein,
            &usage,
            lambda,
            utr5,
            utr3,
            RegionConstraints::default(),
            ScanConfig::default(),
        )
        .unwrap();
        if d.structure != unconstrained.structure || d.sequence != unconstrained.sequence {
            bit = 1;
        }
    }
    bit
}

/// Assert the recovered structure satisfies the hard entries' post-conditions,
/// read directly off the dot-bracket: no pair may fall in a hard-forbidden
/// class, and a hard window leaves every window position unpaired.
fn assert_hard_postconditions(d: &Design, spec: &Spec, pairs: &[(usize, usize)], label: &str) {
    for &(i, j) in pairs {
        assert!(
            spec.class_enforcement(i, j) != Some(Enforcement::Forbid),
            "hard class: forbidden pair ({i},{j}) [{label}]: {}",
            d.structure,
        );
    }
    if let Some((lo, hi, Enforcement::Forbid)) = spec.window {
        let bytes = d.structure.as_bytes();
        for p in lo..hi.min(bytes.len() as u32) {
            assert_eq!(
                bytes[p as usize], b'.',
                "hard window: position {p} paired [{label}]: {}",
                d.structure,
            );
        }
    }
}

/// The structure enumerator must span exactly the CKY grammar's space: the
/// minimum energy over every enumerated nested structure equals
/// `rna-linearfold`'s exact-lattice fold. De-risks the oracle itself.
#[test]
fn structure_enumerator_matches_linearfold() {
    let params = Params::load_default().unwrap();
    for seq in [
        b"GGGAAACCC".as_slice(),
        b"AAAAUUUUGGGG".as_slice(),
        b"GCGCAAAAGCGC".as_slice(),
        b"AUAUAUAUAUAU".as_slice(),
        b"GGGGAAAACCCCUUUU".as_slice(),
    ] {
        let mut best = i32::MAX;
        for s in all_structures(seq) {
            let db = dot_bracket(seq.len(), &s);
            best = best.min(evaluate(&params, seq, &db).expect("scores").dekacal());
        }
        let (_, energy) = linearfold_fold(&params, seq, EXACT).expect("folds");
        assert_eq!(
            best,
            energy.dekacal(),
            "enumerator min ≠ linearfold for {}",
            std::str::from_utf8(seq).unwrap(),
        );
    }
}

/// Cases whose unconstrained MFE optimum is a strong GC stem crossing the
/// CDS/UTR boundary (and straddling the start): a C-rich UTR complementary to a
/// G-rich stretch of the CDS, with a ≥3-nt loop between them. The constraint
/// forbids/penalizes exactly those cross-boundary, over-start pairs, so it bites
/// on both families — and the constrained optimum is a structure that is
/// *suboptimal* unconstrained, exercising the fold-under-constraint path.
const BITING_CASES: [(&str, &str, &str); 3] = [
    ("KGG", "CCCCC", ""), // 5-bp C·G stem: UTR CCCCC ↔ CDS GGGGG, loop AAA…
    ("GKG", "CCC", ""),   // 3-bp stem straddling the start, loop GGGAAA
    ("GGK", "", "CCCCC"), // 3′-side isolation: CDS GGGGG ↔ 3′ UTR CCCCC
];

/// A soft penalty heavy enough to flip a multi-bp GC stem, so the soft variant
/// demonstrably bites (the byte-exact oracle holds at any weight).
fn heavy_soft() -> Enforcement {
    Enforcement::Penalize(derna_lib::Energy(500))
}

/// Cases biting the classes no shipped family could express. Both flanks are
/// present and complementary, so left unconstrained the two UTRs pair with each
/// other across the whole CDS — the configuration the `5′↔3′` class names — and
/// the last flank can fold on itself, which is what an intra-region entry
/// forbids. Kept tiny: the oracle enumerates every nested structure.
const CROSSING_CASES: [(&str, &str, &str); 3] = [
    ("KG", "CCCC", "GGGG"),  // 5′ UTR ↔ 3′ UTR around the CDS
    ("KGG", "CCC", "GGG"),   // both CDS boundaries live alongside it
    ("KG", "", "CCCAAAGGG"), // 3′↔3′: the flank folds on itself
];

fn check_cases(cases: &[(&str, &str, &str)], rc: RegionConstraints, label: &str) {
    let bites: u32 = cases
        .iter()
        .map(|&(p, u5, u3)| check_case(p, u5, u3, rc))
        .sum();
    assert!(bites > 0, "{label} never bit — oracle vacuous");
}

fn check_biting_cases(rc: RegionConstraints, label: &str) {
    check_cases(&BITING_CASES, rc, label);
}

#[test]
fn isolation_hard_matches_constrained_oracle() {
    check_biting_cases(
        RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        "hard isolation",
    );
}

#[test]
fn isolation_soft_matches_constrained_oracle() {
    check_biting_cases(
        RegionConstraints::symmetric_isolation(heavy_soft()),
        "soft isolation",
    );
}

#[test]
fn accessibility_window_hard_matches_constrained_oracle() {
    // Window straddling the start: `upstream` into the 5′ UTR, `downstream` into
    // the CDS, hard-forbidden (the resolved default placement).
    check_biting_cases(
        RegionConstraints::default().with_window(AccessibilityWindow::hard(3, 3)),
        "hard window",
    );
}

#[test]
fn accessibility_window_soft_matches_constrained_oracle() {
    check_biting_cases(
        RegionConstraints::default().with_window(AccessibilityWindow {
            upstream: 3,
            downstream: 3,
            enforcement: heavy_soft(),
        }),
        "soft window",
    );
}

/// Combined families (soft isolation + hard window): penalties from one family
/// and a forbid from the other compose in the oracle exactly as in the constraint.
#[test]
fn combined_families_match_constrained_oracle() {
    check_biting_cases(
        RegionConstraints::symmetric_isolation(Enforcement::Penalize(derna_lib::Energy(30)))
            .with_window(AccessibilityWindow::hard(2, 3)),
        "combined families",
    );
}

/// The `5′↔3′` class: a pair with one endpoint in each UTR and none in the CDS.
/// No shipped family could express it — isolation asks whether *exactly one*
/// endpoint is in the CDS, and this pair has none — which is why it needs its own
/// oracle rather than inheriting one.
#[test]
fn utr_to_utr_class_matches_constrained_oracle() {
    check_cases(
        &CROSSING_CASES,
        RegionConstraints::default().with(PairClass::Utr5Utr3, Enforcement::Forbid),
        "5′↔3′ hard",
    );
    check_cases(
        &CROSSING_CASES,
        RegionConstraints::default().with(PairClass::Utr5Utr3, heavy_soft()),
        "5′↔3′ soft",
    );
}

/// The asymmetric table: one CDS boundary forbidden, the other open. The shipped
/// default, and the case a symmetric family cannot represent at all.
#[test]
fn five_prime_isolation_matches_constrained_oracle() {
    check_cases(
        &BITING_CASES,
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        "5′↔CDS hard",
    );
    check_cases(
        &CROSSING_CASES,
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        "5′↔CDS hard, both flanks",
    );
}

/// Every crossing class at once — the separability configuration, checked here
/// against the enumeration rather than against the decomposition.
#[test]
fn full_mutual_isolation_matches_constrained_oracle() {
    check_cases(
        &CROSSING_CASES,
        RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
        "full mutual isolation",
    );
}

/// An intra-region entry: the 3′ UTR may not pair with itself. Nothing in the
/// shipped families reaches an intra-region class, so it exercises a corner of
/// the table the generalization opened.
#[test]
fn intra_region_class_matches_constrained_oracle() {
    check_cases(
        &CROSSING_CASES,
        RegionConstraints::default().with(PairClass::Utr3Utr3, Enforcement::Forbid),
        "3′↔3′ hard",
    );
}

/// The Scope-17 composition trap, proven: the constrained fold under the
/// Lyngsø–Zuker / one-sided / `multi_open` reductions ([`InternalLoopMode::Lz`])
/// equals the un-reduced constrained fold ([`InternalLoopMode::Naive`])
/// byte-exactly (MFE + CAI) at lengths that engage the skewed/one-sided
/// reductions (loops with flanks ≥ 5). If a reduction dropped a constrained
/// pair the other kept, the two folds would diverge.
#[test]
fn constraint_composes_with_reductions_lz_vs_naive() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let cfg = |mode| ScanConfig {
        mode,
        ..ScanConfig::default()
    };
    // Long enough (n ≈ 45–60) that internal loops with both flanks ≥ 5 form and
    // the skewed / one-sided reductions run; the constraint is active and bites.
    let cases: &[(&str, &str, &str, RegionConstraints)] = &[
        (
            "MKTAYIAKQRLF",
            "GGGCACCAUGGCC",
            "AAUAAAGCUU",
            RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        ),
        (
            "DEYWCNQHKRFF",
            "GCGCGCGCGC",
            "GCGCGCGCGC",
            RegionConstraints::symmetric_isolation(Enforcement::Penalize(derna_lib::Energy(25)))
                .with_window(AccessibilityWindow::hard(4, 6)),
        ),
        (
            "FFLLSSPPHHRRKK",
            "GGGGGGCCCCCC",
            "",
            RegionConstraints::default().with_window(AccessibilityWindow::hard(5, 5)),
        ),
        // The 5′↔3′ class alone. Complementary flanks so the class has something
        // to bite on: without a constraint the two would pair straight through.
        (
            "MKTAYIAKQRLF",
            "GGGGGGCCCCCC",
            "GGGGGGCCCCCC",
            RegionConstraints::default().with(PairClass::Utr5Utr3, Enforcement::Forbid),
        ),
        // The asymmetric table: one CDS boundary forbidden and the other open.
        (
            "DEYWCNQHKRFF",
            "GCGCGCGCGC",
            "GCGCGCGCGC",
            RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        ),
        // Every crossing class at once, soft so the fold still has to price it
        // rather than route around it.
        (
            "FFLLSSPPHHRRKK",
            "GGGCACCAUGGCC",
            "AAUAAAGCUU",
            RegionConstraints::full_mutual_isolation(Enforcement::Penalize(derna_lib::Energy(25))),
        ),
        // An intra-region class, which no shipped family could express: the 3′
        // UTR may not pair with itself.
        (
            "MKTAYIAKQRLF",
            "",
            "GGGGGGCCCCCC",
            RegionConstraints::default().with(PairClass::Utr3Utr3, Enforcement::Forbid),
        ),
    ];
    for (protein, utr5, utr3, rc) in cases {
        for &lambda in &LAMBDAS {
            let lz = design_with_utr(
                &params,
                protein,
                &usage,
                lambda,
                utr5,
                utr3,
                *rc,
                cfg(InternalLoopMode::Lz),
            )
            .unwrap();
            let naive = design_with_utr(
                &params,
                protein,
                &usage,
                lambda,
                utr5,
                utr3,
                *rc,
                cfg(InternalLoopMode::Naive),
            )
            .unwrap();
            assert_eq!(
                (
                    (lz.mfe_kcal_mol * 100.0).round() as i32,
                    (lz.cai_log * 100.0).round() as i32,
                ),
                (
                    (naive.mfe_kcal_mol * 100.0).round() as i32,
                    (naive.cai_log * 100.0).round() as i32,
                ),
                "Lz vs Naive constrained fold diverged [{protein} λ={lambda} {rc:?}]",
            );
        }
    }
}
