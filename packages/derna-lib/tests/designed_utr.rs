//! Designed-UTR oracle.
//!
//! A designed UTR position is one DFA node with **four** parallel zero-CAI
//! out-edges (`A`/`C`/`G`/`U`), so the flanking sequence is optimized jointly
//! with the CDS instead of being supplied. The CKY scan, the six tables, and the
//! interior-loop reductions are unchanged — a closed pair already keys on its
//! endpoint nucleotides, so the pairing base is selected on the edge. These
//! checks pin that claim rather than assuming it.
//!
//! Six checks:
//!
//! 1. **Floors.** An all-`Fixed` spec is byte-identical to `design_with_utr` on
//!    the same strings; zero-length flanks with an inert constraint set are
//!    byte-identical to `design`. Neither may ever break.
//! 2. **Joint enumeration oracle (unconstrained).** For each case, enumerate
//!    every designed-flank assignment (`4^|UTR|`) **and** every CDS encoding,
//!    spell the full mRNA, enumerate all nested canonical structures (min
//!    hairpin 3), take the per-sequence MFE, form BRD with the CDS-only CAI, and
//!    argmin over *both* axes. The design matches within the objective tolerance.
//! 3. **Under region constraints.** The same oracle with an active constraint — hard
//!    and soft, both families — folding *under* the constraint (hard violators
//!    dropped, soft penalties added) rather than filtering an unconstrained fold.
//! 4. **The accessibility window bites on a designed 5′ UTR.** Global-MFE
//!    minimization drives a CAI-free UTR to maximal structure; the window is the
//!    only counter-term. A tracked, asserted-non-zero count of cases where the
//!    windowed optimum leaves the start region unpaired while the unconstrained
//!    optimum structures it.
//! 5. **Fold parity / phantom-free.** The joint scan's whole-mRNA MFE equals an
//!    independent full fold of the spelled designed sequence, and the reported
//!    structure re-scores to the reported energy (plus soft penalties, credited
//!    exactly once).
//! 6. **Reduction composition over designed nodes.** A four-edge UTR node is a
//!    new multi-edge configuration, so the Lyngsø–Zuker / one-sided /
//!    `multi_open` reductions are re-proven against the un-reduced fold at
//!    lengths that engage them, with and without constraints.
//!
//! The structure enumerator, the constraint predicate (`Spec`), and the CAI formula
//! are re-implemented here rather than shared with the crate, so a bug in the
//! substrate cannot hide inside its own oracle. `structure_enumerator_matches_linearfold`
//! de-risks this copy of the enumerator.

use std::collections::HashMap;

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design, design_with_utr, design_with_utr_spec, linearfold_fold, AccessibilityWindow,
    CodonUsage, Design, Energy, Enforcement, InternalLoopMode, PairClass, Params,
    RegionConstraints, ScanConfig, UtrSpec,
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
    mixed_radix(&counts)
}

/// Every assignment of a designed flank of `len` positions: all `4^len`
/// nucleotide strings, as ASCII bytes. A zero-length flank yields the single
/// empty assignment.
fn designed_assignments(len: usize) -> Vec<Vec<u8>> {
    mixed_radix(&vec![4usize; len])
        .into_iter()
        .map(|digits| digits.into_iter().map(|d| nuc_byte(d as u8)).collect())
        .collect()
}

/// Every digit vector of the mixed-radix counter with the given per-place radix
/// — the shared enumeration behind both the CDS encodings and the designed-flank
/// assignments. An empty `radix` yields one empty vector.
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

/// Every candidate spelling of one flank: the single fixed string, all `4^len`
/// designed assignments, or — for a mixed mask — the `4^k` spellings over its
/// `k` free positions with every given position held.
fn flank_choices(spec: &UtrSpec) -> Vec<Vec<u8>> {
    match spec {
        UtrSpec::Fixed(s) => vec![s.as_bytes().to_vec()],
        UtrSpec::Designed(len) => designed_assignments(*len),
        UtrSpec::Mixed(mask) => mixed_assignments(mask.as_bytes()),
    }
}

/// Every spelling of a mixed mask: each `N` ranges over the four nucleotides,
/// every other byte is held as written.
fn mixed_assignments(mask: &[u8]) -> Vec<Vec<u8>> {
    let radix: Vec<usize> = mask
        .iter()
        .map(|&b| if b == b'N' { 4 } else { 1 })
        .collect();
    mixed_radix(&radix)
        .into_iter()
        .map(|digits| {
            mask.iter()
                .zip(digits)
                .map(|(&b, d)| if b == b'N' { nuc_byte(d as u8) } else { b })
                .collect()
        })
        .collect()
}

/// The whole-mRNA bytes for one flank pair and CDS encoding.
fn full_bytes(aa: &[usize], encoding: &[usize], utr5: &[u8], utr3: &[u8]) -> Vec<u8> {
    let mut out = utr5.to_vec();
    for (&a, &ci) in aa.iter().zip(encoding) {
        out.extend(AMINO_ACIDS[a].codons[ci].nuc.iter().map(|&n| nuc_byte(n)));
    }
    out.extend_from_slice(utr3);
    out
}

/// CDS-only additive log-CAI of one encoding (the same `EPSILON`-floored formula
/// the DFA's edge weights use). A designed UTR contributes nothing — the point of
/// the zero-CAI edges.
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
    // `lo` unpaired.
    let mut out: Structs = structures(seq, lo + 1, hi, memo);
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
/// re-implementation of the constraint predicate the oracle scores against.
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
/// feasible for every region constraint.
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

/// One oracle case: a protein plus a spec for each flank.
struct Case {
    protein: &'static str,
    utr5: UtrSpec,
    utr3: UtrSpec,
}

impl Case {
    fn new(protein: &'static str, utr5: UtrSpec, utr3: UtrSpec) -> Self {
        Self {
            protein,
            utr5,
            utr3,
        }
    }

    fn cds_start(&self) -> u32 {
        self.utr5.len() as u32
    }

    fn cds_end(&self) -> u32 {
        self.cds_start() + 3 * self.protein.len() as u32
    }

    fn design(&self, lambda: f64, rc: RegionConstraints, cfg: ScanConfig) -> Design {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        design_with_utr_spec(
            &params,
            self.protein,
            &usage,
            lambda,
            &self.utr5,
            &self.utr3,
            rc,
            cfg,
        )
        .expect("designs")
    }

    /// Every `(constrained MFE kcal, CDS-only log-CAI)` the case admits: the
    /// joint product of designed-flank assignments and CDS encodings. Both
    /// coordinates are λ-independent, so one sweep serves every λ.
    fn scored(&self, spec: &Spec) -> Vec<(f64, f64)> {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa = protein_indices(self.protein);
        let mut out = Vec::new();
        for u5 in flank_choices(&self.utr5) {
            for u3 in flank_choices(&self.utr3) {
                for enc in encodings(&aa) {
                    let full = full_bytes(&aa, &enc, &u5, &u3);
                    out.push((
                        f64::from(constrained_mfe_dk(&params, &full, spec)) / 100.0,
                        cds_cai_log(&usage, &aa, &enc),
                    ));
                }
            }
        }
        out
    }
}

/// Whether `cds` (ASCII `A/C/G/U`) is a valid coding sequence for `protein`.
fn cds_translates(cds: &[u8], protein: &str) -> bool {
    if cds.len() != protein.len() * 3 {
        return false;
    }
    let code = |c: u8| match c {
        b'A' => 0u8,
        b'C' => 1,
        b'G' => 2,
        _ => 3,
    };
    protein.chars().enumerate().all(|(i, ch)| {
        let aa = aa_to_index(ch).unwrap();
        let trip = [code(cds[3 * i]), code(cds[3 * i + 1]), code(cds[3 * i + 2])];
        AMINO_ACIDS[aa].codons.iter().any(|c| c.nuc == trip)
    })
}

/// Assert the spelled mRNA has the case's layout: the right total length, a CDS
/// that encodes the protein, every flank position a real nucleotide, and any
/// `Fixed` flank preserved verbatim.
fn assert_layout(d: &Design, case: &Case, label: &str) {
    let full = d.sequence.as_bytes();
    let (l5, l3) = (case.utr5.len(), case.utr3.len());
    assert_eq!(
        full.len(),
        l5 + case.protein.len() * 3 + l3,
        "full-mRNA length wrong [{label}]",
    );
    assert!(
        full.iter().all(|b| b"ACGU".contains(b)),
        "non-nucleotide in spelled mRNA [{label}]: {}",
        d.sequence,
    );
    assert!(
        cds_translates(&full[l5..full.len() - l3], case.protein),
        "CDS does not encode {} [{label}]",
        case.protein,
    );
    if let UtrSpec::Fixed(s) = &case.utr5 {
        assert_eq!(&full[..l5], s.as_bytes(), "5′ UTR not preserved [{label}]");
    }
    if let UtrSpec::Fixed(s) = &case.utr3 {
        assert_eq!(
            &full[full.len() - l3..],
            s.as_bytes(),
            "3′ UTR not preserved [{label}]",
        );
    }
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

/// Run the joint enumeration oracle over one case under `rc`, across the λ
/// sweep: BRD optimality against the brute force, layout, phantom-free MFE, and
/// the hard structural post-conditions. Returns the number of swept λ at which
/// the constraint **bit** (the constrained optimum differs from the
/// unconstrained one).
fn check_case(case: &Case, rc: RegionConstraints) -> u32 {
    let params = Params::load_default().unwrap();
    let spec = Spec::new(&rc, case.cds_start(), case.cds_end());
    let scored = case.scored(&spec);

    let mut bites = 0;
    for &lambda in &LAMBDAS {
        let label = format!(
            "{} 5′={:?} 3′={:?} λ={lambda} {rc:?}",
            case.protein, case.utr5, case.utr3
        );
        let oracle = scored
            .iter()
            .map(|&(mfe, cai)| brd(lambda, mfe, cai))
            .fold(f64::INFINITY, f64::min);

        let d = case.design(lambda, rc, ScanConfig::default());
        assert_layout(&d, case, &label);

        // (2/3) Joint optimality over both axes, both directions: the design can
        // neither beat the true constrained optimum (which would mean it violated
        // the constraint or the oracle missed a structure) nor fall short of it.
        let design_brd = brd(lambda, d.mfe_kcal_mol, d.cai_log);
        assert!(
            (design_brd - oracle).abs() <= BRD_TOL,
            "joint-oracle mismatch [{label}]: design={design_brd:.1} oracle={oracle:.1}",
        );

        // (5) Phantom-free: the reported energy is the plain re-score of the
        // recovered structure plus the soft penalties that structure incurs.
        let recovered = pairs_of(&d.structure);
        let (forbid, pen) = spec.structure(&recovered);
        assert!(
            !forbid,
            "design emitted a hard-forbidden pair [{label}]: {}",
            d.structure,
        );
        let reeval = evaluate(&params, d.sequence.as_bytes(), &d.structure)
            .expect("structure scores")
            .dekacal();
        let reported_dk = (d.mfe_kcal_mol * 100.0).round() as i32;
        assert_eq!(
            reported_dk,
            reeval + pen,
            "phantom constrained MFE [{label}]: reported={reported_dk} reeval={reeval} pen={pen}",
        );

        assert_hard_postconditions(&d, &spec, &recovered, &label);

        let free = case.design(lambda, RegionConstraints::default(), ScanConfig::default());
        if d.sequence != free.sequence || d.structure != free.structure {
            bites += 1;
        }
    }
    bites
}

/// Cases small enough to enumerate `4^|designed UTR|` × encodings × structures,
/// and shaped so a CAI-free UTR wants to fold back onto the CDS: `M`/`W` have a
/// single codon each (the flank is then the whole design space) and `G`/`K` give
/// a G-rich CDS a designed 5′ UTR can pair into.
///
/// The last two add an A-rich (`K`) and a U-rich (`F`) CDS, so the flank faces
/// every partner base rather than only the G-rich stems the others offer.
///
/// This oracle does **not** pin the designed alphabet on its own: `A`/`U` stacks
/// are too weak to pay a hairpin penalty at these lengths, so no tractable
/// optimum spells a `U` at a *paired* designed position, and a node offering only
/// three of the four edges would still match. That claim is pinned deterministically
/// at the substrate instead, by `dfa::tests::designed_utr_node_offers_all_four_nucleotides`.
fn oracle_cases() -> Vec<Case> {
    vec![
        // Pure UTR design: one CDS encoding, five free 5′ positions.
        Case::new("MW", UtrSpec::Designed(5), UtrSpec::Fixed(String::new())),
        // Joint design: 16 encodings × 64 designed 5′ flanks.
        Case::new("GG", UtrSpec::Designed(3), UtrSpec::Fixed(String::new())),
        // The 3′ side.
        Case::new("GK", UtrSpec::Fixed(String::new()), UtrSpec::Designed(3)),
        // Mixed flanks: a supplied 5′ UTR with a designed 3′ one.
        Case::new("KG", UtrSpec::Fixed("CC".into()), UtrSpec::Designed(2)),
        // A-rich CDS: only a `U` flank can stem into it.
        Case::new("KK", UtrSpec::Designed(4), UtrSpec::Fixed(String::new())),
        // U-rich CDS: only an `A` flank can stem into it.
        Case::new("FF", UtrSpec::Designed(4), UtrSpec::Fixed(String::new())),
    ]
}

/// A soft penalty heavy enough to flip a multi-bp GC stem, so the soft variant
/// demonstrably bites (the byte-exact oracle holds at any weight).
fn heavy_soft() -> Enforcement {
    Enforcement::Penalize(Energy(500))
}

/// The window used wherever the accessibility family is exercised: two positions
/// into the 5′ UTR through three into the CDS, so it straddles the start of every
/// oracle case.
fn window(enforcement: Enforcement) -> AccessibilityWindow {
    AccessibilityWindow {
        upstream: 2,
        downstream: 3,
        enforcement,
    }
}

/// Run every oracle case under `rc` and assert the constraint bit somewhere, so
/// the check is not vacuously green.
fn check_all_cases(rc: RegionConstraints, label: &str) {
    let bites: u32 = oracle_cases().iter().map(|c| check_case(c, rc)).sum();
    assert!(bites > 0, "{label} never bit — oracle vacuous");
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

/// Floor 1: an all-`Fixed` spec reproduces `design_with_utr` on the same strings
/// byte-for-byte, so generalizing the API left the fixed-UTR substrate untouched.
#[test]
fn all_fixed_spec_is_byte_identical_to_fixed_utr() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let rcs = [
        RegionConstraints::default(),
        RegionConstraints::symmetric_isolation(Enforcement::Forbid)
            .with_window(window(Enforcement::Forbid)),
    ];
    for (protein, utr5, utr3) in [
        ("MKTAYIAKQR", "GCCACCAUGG", ""),
        ("DEYWCNQHKR", "", "AAUAAAGCUU"),
        ("FFLLSSPPHH", "GGGCCCGGG", "CCCGGGCCC"),
    ] {
        for rc in rcs {
            for &lambda in &LAMBDAS {
                let fixed = design_with_utr(
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
                let spec = design_with_utr_spec(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    &UtrSpec::Fixed(utr5.into()),
                    &UtrSpec::Fixed(utr3.into()),
                    rc,
                    ScanConfig::default(),
                )
                .unwrap();
                assert_eq!(
                    fixed, spec,
                    "all-fixed spec diverged [{protein} λ={lambda}]"
                );
            }
        }
    }
}

/// Floor 2: zero-length flanks — `Designed(0)` as much as `Fixed("")` — with an
/// inert constraint set collapse to the CDS-only design byte-for-byte.
#[test]
fn zero_length_spec_is_byte_identical_to_cds_only() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let empties = [UtrSpec::Fixed(String::new()), UtrSpec::Designed(0)];
    for protein in ["MWMW", "GGGGCCCC", "MKTAYIAKQR"] {
        for &lambda in &LAMBDAS {
            let base = design(&params, protein, &usage, lambda).unwrap();
            for utr5 in &empties {
                for utr3 in &empties {
                    let d = design_with_utr_spec(
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
                    assert_eq!(
                        base, d,
                        "zero-length spec diverged [{protein} λ={lambda} {utr5:?}/{utr3:?}]",
                    );
                }
            }
        }
    }
}

/// The joint `4^|UTR|` × encodings oracle, unconstrained: the design is the BRD
/// argmin over both the designed-flank assignments and the CDS encodings.
#[test]
fn designed_utr_matches_joint_oracle() {
    for case in oracle_cases() {
        check_case(&case, RegionConstraints::default());
    }
}

#[test]
fn designed_utr_under_hard_isolation_matches_oracle() {
    check_all_cases(
        RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        "hard isolation",
    );
}

#[test]
fn designed_utr_under_soft_isolation_matches_oracle() {
    check_all_cases(
        RegionConstraints::symmetric_isolation(heavy_soft()),
        "soft isolation",
    );
}

#[test]
fn designed_utr_under_hard_window_matches_oracle() {
    check_all_cases(
        RegionConstraints::default().with_window(window(Enforcement::Forbid)),
        "hard window",
    );
}

#[test]
fn designed_utr_under_soft_window_matches_oracle() {
    check_all_cases(
        RegionConstraints::default().with_window(window(heavy_soft())),
        "soft window",
    );
}

#[test]
fn designed_utr_under_combined_families_matches_oracle() {
    check_all_cases(
        RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(30)))
            .with_window(window(Enforcement::Forbid)),
        "combined families",
    );
}

/// The payoff composition: a UTR earns no CAI, so global-MFE minimization drives
/// a designed one to fold back onto the start region — and the accessibility
/// window is the only counter-term. Counts cases where the unconstrained designed
/// optimum structures the window while the windowed optimum leaves every window
/// position unpaired, and asserts that happens.
#[test]
fn accessibility_window_opens_a_designed_start_region() {
    let rc = RegionConstraints::default().with_window(window(Enforcement::Forbid));
    // 5′-designed cases only: the window straddles the start, so it can only
    // counteract a designed flank on the 5′ side.
    let cases = [
        Case::new("MW", UtrSpec::Designed(5), UtrSpec::Fixed(String::new())),
        Case::new("GG", UtrSpec::Designed(4), UtrSpec::Fixed(String::new())),
        Case::new("GK", UtrSpec::Designed(4), UtrSpec::Fixed(String::new())),
    ];
    let mut opened = 0;
    for case in &cases {
        let spec = Spec::new(&rc, case.cds_start(), case.cds_end());
        let (lo, hi, _) = spec.window.expect("window active");
        for &lambda in &LAMBDAS {
            let free = case.design(lambda, RegionConstraints::default(), ScanConfig::default());
            let held = case.design(lambda, rc, ScanConfig::default());
            let structured = |d: &Design| {
                d.structure.as_bytes()[lo as usize..(hi as usize).min(d.structure.len())]
                    .iter()
                    .any(|&b| b != b'.')
            };
            if structured(&free) && !structured(&held) {
                opened += 1;
            }
        }
    }
    assert!(
        opened > 0,
        "the accessibility window never opened a designed start region — the designed-start case is unexercised",
    );
}

/// Fold parity + phantom-free over a *designed* sequence: the joint scan's
/// whole-mRNA MFE equals an independent full fold of the sequence it spelled, and
/// the reported structure re-scores to the reported energy. Unconstrained, so the
/// independent fold is directly comparable.
#[test]
fn designed_utr_fold_parity() {
    let params = Params::load_default().unwrap();
    let cases = [
        Case::new(
            "MKTAYIAKQR",
            UtrSpec::Designed(8),
            UtrSpec::Fixed(String::new()),
        ),
        Case::new(
            "DEYWCNQHKR",
            UtrSpec::Fixed(String::new()),
            UtrSpec::Designed(8),
        ),
        Case::new("FFLLSSPPHH", UtrSpec::Designed(6), UtrSpec::Designed(6)),
        Case::new(
            "MKTAYIAKQR",
            UtrSpec::Fixed("GGGACC".into()),
            UtrSpec::Designed(5),
        ),
    ];
    for case in &cases {
        for &lambda in &LAMBDAS {
            let label = format!("{} λ={lambda}", case.protein);
            let d = case.design(lambda, RegionConstraints::default(), ScanConfig::default());
            assert_layout(&d, case, &label);
            let full = d.sequence.as_bytes();

            let (structure, energy) = linearfold_fold(&params, full, EXACT).expect("folds");
            let reported_dk = (d.mfe_kcal_mol * 100.0).round() as i32;
            assert_eq!(
                reported_dk,
                energy.dekacal(),
                "fold-parity MFE mismatch [{label}]: joint={reported_dk} independent={}\n  joint {}\n  indep {}",
                energy.dekacal(),
                d.structure,
                structure,
            );
            let reeval = evaluate(&params, full, &d.structure).expect("structure scores");
            assert_eq!(
                reeval.dekacal(),
                reported_dk,
                "phantom energy [{label}]: structure={} reeval={}",
                d.structure,
                reeval.dekacal(),
            );
        }
    }
}

/// Reduction composition over the designed nodes. A four-edge UTR node is a new
/// multi-edge configuration in the UTR region, so the Lyngsø–Zuker / one-sided /
/// `multi_open` reductions ([`InternalLoopMode::Lz`]) are re-proven against the
/// un-reduced fold ([`InternalLoopMode::Naive`]) byte-exactly (MFE + CAI) at
/// lengths that engage the skewed / one-sided reductions (loops with flanks ≥ 5),
/// with and without an active constraint. If a reduction dropped a candidate the other
/// kept over a designed node, the two folds would diverge.
#[test]
fn designed_utr_composes_with_reductions_lz_vs_naive() {
    let cfg = |mode| ScanConfig {
        mode,
        ..ScanConfig::default()
    };
    let cases = [
        (
            Case::new("MKTAYIAKQRLF", UtrSpec::Designed(8), UtrSpec::Designed(6)),
            RegionConstraints::default(),
        ),
        (
            Case::new(
                "DEYWCNQHKRFF",
                UtrSpec::Designed(10),
                UtrSpec::Fixed(String::new()),
            ),
            RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        ),
        (
            Case::new("FFLLSSPPHHRR", UtrSpec::Designed(6), UtrSpec::Designed(6)),
            RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(25)))
                .with_window(AccessibilityWindow::hard(4, 6)),
        ),
        (
            Case::new(
                "MKTAYIAKQRLF",
                UtrSpec::Fixed("GGGCACCAUGGCC".into()),
                UtrSpec::Designed(7),
            ),
            RegionConstraints::default().with_window(AccessibilityWindow::hard(5, 5)),
        ),
        // The 5′↔3′ class on two designed flanks — the configuration that,
        // unconstrained, clamps one flank to the other around the whole CDS.
        (
            Case::new("FFLLSSPPHHRR", UtrSpec::Designed(8), UtrSpec::Designed(8)),
            RegionConstraints::default().with(PairClass::Utr5Utr3, Enforcement::Forbid),
        ),
        // The asymmetric table over a mixed fixed/designed pair of flanks.
        (
            Case::new(
                "DEYWCNQHKRFF",
                UtrSpec::Fixed("GCGCGCGCGC".into()),
                UtrSpec::Designed(8),
            ),
            RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        ),
        // Every crossing class at once over designed flanks: separability's
        // configuration, where each region must fold alone.
        (
            Case::new("MKTAYIAKQRLF", UtrSpec::Designed(7), UtrSpec::Designed(7)),
            RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
        ),
    ];
    for (case, rc) in &cases {
        for &lambda in &LAMBDAS {
            let lz = case.design(lambda, *rc, cfg(InternalLoopMode::Lz));
            let naive = case.design(lambda, *rc, cfg(InternalLoopMode::Naive));
            let key = |d: &Design| {
                (
                    (d.mfe_kcal_mol * 100.0).round() as i32,
                    (d.cai_log * 100.0).round() as i32,
                )
            };
            assert_eq!(
                key(&lz),
                key(&naive),
                "Lz vs Naive diverged over designed UTR [{} λ={lambda} {rc:?}]",
                case.protein,
            );
        }
    }
}
