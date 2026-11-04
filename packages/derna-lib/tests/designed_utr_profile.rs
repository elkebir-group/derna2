//! Designed-UTR profile oracle.
//!
//! A designed UTR position earns no CAI, so without a term of its own nothing in
//! the objective mentions its nucleotide: it is settled by folding energy alone,
//! or — under a table forbidding the flank every pair it could form — by nothing
//! at all, leaving every spelling exactly co-optimal. [`UtrProfile`] is the term
//! that prices it, a per-position nucleotide log-odds riding the same edge
//! weights CAI does.
//!
//! Six checks:
//!
//! 1. **Floors.** A zero-strength profile, and a uniform one, are byte-identical
//!    to the same design without the term — on designed, fixed and mixed flanks,
//!    constrained and not, at every λ. A fully given flank is byte-identical with
//!    any profile on or off, since it has no position to price. Neither may ever
//!    break.
//! 2. **Joint enumeration oracle.** For each case, enumerate every flank
//!    spelling **and** every CDS encoding, spell the full mRNA, enumerate all
//!    nested canonical structures (min hairpin 3), take the per-sequence
//!    constrained MFE, form the objective with the CDS-only CAI **plus the
//!    profile's own total over the designed positions**, and argmin over both
//!    axes. The design matches within the objective tolerance.
//! 3. **The reported halves are the ones documented.** `cai_log` is the CDS
//!    codons alone and `utr_profile_log` is the profile scored on the emitted
//!    flank, both recomputed here from the emitted sequence. A term folded into
//!    the reported CAI would pass check 2 and fail this one.
//! 4. **The term bites.** A tracked, asserted-non-zero count of cases where the
//!    profiled optimum spells a different flank than the unprofiled one.
//! 5. **It prices what nothing else prices.** Under a table forbidding every
//!    class with a flank endpoint the flank cannot pair, so folding energy is
//!    indifferent to it and every spelling is co-optimal. The profile alone then
//!    decides, and the flank comes back as the profile's argmax homopolymer —
//!    which is both the property that makes the term worth having and the reason
//!    it is a preference rather than a composition target.
//! 6. **Fold parity.** The whole-mRNA MFE the scan reports is an independent full
//!    fold of the spelled sequence: the term rides the adaptation half and must
//!    not perturb the energy the design is credited with.
//!
//! The structure enumerator, the constraint predicate, the CAI formula and the
//! profile's own weights are re-implemented here rather than shared with the
//! crate, so a bug in the substrate cannot hide inside its own oracle.

use std::collections::HashMap;

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_with_utr_profile, design_with_utr_spec, linearfold_fold, CodonUsage, Design,
    Enforcement, PairClass, Params, RegionConstraints, ScanConfig, UtrProfile, UtrSpec,
};
use rna_mfe::evaluate;

/// One secondary structure as its list of `(i, j)` base pairs.
type Pairs = Vec<(usize, usize)>;
/// A set of structures over one interval.
type Structs = Vec<Pairs>;
/// Per-interval memo for the structure enumeration.
type Memo = HashMap<(usize, usize), Structs>;

/// BRD tolerance in dekacalories — absorbs the rounded-integer vs continuous
/// objective gap, exactly as `tests/brd_parity.rs`.
const BRD_TOL: f64 = 0.5;

/// The λ values swept by every parity check (low / mid / high).
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// Two profiles with **different, unique** argmax nucleotides. Two rather than
/// one so a check that the flank follows the profile cannot pass by coincidence:
/// the argmax has to move when the profile does. Unique rather than tied so the
/// argmax is the profile's doing and not the edge order's — a tie would let a
/// spelling that ignores the profile entirely satisfy the assertion.
const A_LEANING: [f64; 4] = [0.40, 0.15, 0.15, 0.30];
const C_LEANING: [f64; 4] = [0.15, 0.40, 0.30, 0.15];

fn nuc_byte(n: u8) -> u8 {
    [b'A', b'C', b'G', b'U'][n as usize]
}

fn params_usage() -> (Params, CodonUsage) {
    (
        Params::load_default().expect("Turner 2004 parameters"),
        CodonUsage::load_default().expect("codon usage"),
    )
}

fn protein_indices(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect()
}

/// Fixed-λ BRD on rounded dekacalories, matching how the oracle and the design
/// both round at the scalarization boundary (`tests/brd_parity.rs`). The
/// adaptation half is the CDS CAI **plus** the profile's total, which is the one
/// thing this objective adds to the one `designed_utr.rs` scores against.
fn brd(lambda: f64, mfe_kcal: f64, adaptation_log: f64) -> f64 {
    let mfe_dk = (mfe_kcal * 100.0).round();
    let adaptation_dk = (adaptation_log * 100.0).round();
    lambda * mfe_dk - (1.0 - lambda) * adaptation_dk
}

/// The oracle's own copy of the profile weights: `β·log(f / f_max)`, normalized
/// to a maximum of zero. Recomputed from the frequencies rather than read off
/// [`UtrProfile`], so the crate's arithmetic is never shared with the check.
fn oracle_weights(frequencies: [f64; 4], strength: f64) -> [f64; 4] {
    let max = frequencies.iter().copied().fold(f64::MIN, f64::max);
    let mut w = [0.0; 4];
    for (slot, f) in w.iter_mut().zip(frequencies) {
        *slot = strength * (f / max).ln();
    }
    w
}

/// The profile's total over a run of designed positions, from ASCII bytes.
fn score_flank(weights: [f64; 4], flank: &[u8]) -> f64 {
    flank
        .iter()
        .map(|&b| weights[b"ACGU".iter().position(|&n| n == b).expect("nucleotide")])
        .sum()
}

/// Whether ASCII nucleotides `a` and `b` form a canonical Watson–Crick / wobble
/// pair (the same six pairs `rna-mfe`'s `pair_from_nucleotides` admits).
fn canonical(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'C') | (b'C', b'G') | (b'G', b'U') | (b'U', b'G')
    )
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

/// Every candidate spelling of one flank: the single fixed string, all `4^len`
/// designed assignments, or the `4^k` spellings over a mixed mask's free
/// positions.
fn flank_choices(spec: &UtrSpec) -> Vec<Vec<u8>> {
    let mask: Vec<u8> = match spec {
        UtrSpec::Fixed(s) => return vec![s.as_bytes().to_vec()],
        UtrSpec::Designed(len) => vec![b'N'; *len],
        UtrSpec::Mixed(s) => s.as_bytes().to_vec(),
    };
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

/// The positions of a flank spelling that the designer chose — the ones a
/// profile prices. A `Fixed` flank has none.
fn designed_mask(spec: &UtrSpec) -> Vec<bool> {
    match spec {
        UtrSpec::Fixed(s) => vec![false; s.len()],
        UtrSpec::Designed(len) => vec![true; *len],
        UtrSpec::Mixed(s) => s.bytes().map(|b| b == b'N').collect(),
    }
}

/// The designed positions of a flank spelling, as bytes.
fn designed_bytes(spec: &UtrSpec, flank: &[u8]) -> Vec<u8> {
    designed_mask(spec)
        .into_iter()
        .zip(flank)
        .filter_map(|(free, &b)| free.then_some(b))
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

/// CDS-only additive log-CAI of one encoding, `EPSILON`-floored as the DFA's own
/// edge weights are. A UTR position contributes nothing here whatever a profile
/// says — that separation is what check 3 pins.
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
/// minimum hairpin loop 3, memoized per interval.
fn structures(seq: &[u8], lo: usize, hi: usize, memo: &mut Memo) -> Structs {
    if lo >= hi {
        return vec![Vec::new()];
    }
    if let Some(cached) = memo.get(&(lo, hi)) {
        return cached.clone();
    }
    let mut out: Structs = structures(seq, lo + 1, hi, memo);
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

/// The dot-bracket string for a pair list over a length-`n` sequence.
fn dot_bracket(n: usize, pairs: &[(usize, usize)]) -> String {
    let mut s = vec![b'.'; n];
    for &(i, j) in pairs {
        s[i] = b'(';
        s[j] = b')';
    }
    String::from_utf8(s).expect("ascii brackets")
}

/// The oracle's re-implementation of the region constraint, keyed on its **own**
/// region codes so the crate's table indexing is never shared with it.
struct Spec {
    cds_start: usize,
    cds_end: usize,
    classes: Vec<((u8, u8), Enforcement)>,
}

impl Spec {
    fn new(rc: &RegionConstraints, cds_start: usize, cds_end: usize) -> Self {
        let code = |c: PairClass| match c {
            PairClass::Utr5Utr5 => (0, 0),
            PairClass::Utr5Cds => (0, 1),
            PairClass::Utr5Utr3 => (0, 2),
            PairClass::CdsCds => (1, 1),
            PairClass::CdsUtr3 => (1, 2),
            PairClass::Utr3Utr3 => (2, 2),
        };
        Self {
            cds_start,
            cds_end,
            classes: PairClass::ALL
                .iter()
                .filter_map(|&c| rc.enforcement(c).map(|e| (code(c), e)))
                .collect(),
        }
    }

    fn region_code(&self, p: usize) -> u8 {
        if p < self.cds_start {
            0
        } else if p < self.cds_end {
            1
        } else {
            2
        }
    }

    /// `(any-hard-violation, total-soft-penalty-dekacal)` for a whole structure.
    fn structure(&self, pairs: &[(usize, usize)]) -> (bool, i32) {
        let mut forbid = false;
        let mut pen = 0i32;
        for &(i, j) in pairs {
            let (a, b) = (self.region_code(i), self.region_code(j));
            let key = (a.min(b), a.max(b));
            match self
                .classes
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, e)| *e)
            {
                Some(Enforcement::Forbid) => forbid = true,
                Some(Enforcement::Penalize(w)) => pen += w.dekacal(),
                None => {}
            }
        }
        (forbid, pen)
    }
}

/// The constrained MFE (dekacalories) of one spelled sequence under `spec`.
/// Always defined — the all-unpaired structure satisfies every region table.
fn constrained_mfe_dk(params: &Params, full: &[u8], spec: &Spec) -> i32 {
    let mut memo = Memo::new();
    let mut best: Option<i32> = None;
    for s in structures(full, 0, full.len(), &mut memo) {
        let (forbid, pen) = spec.structure(&s);
        if forbid {
            continue;
        }
        let db = dot_bracket(full.len(), &s);
        let e = evaluate(params, full, &db)
            .expect("structure scores")
            .dekacal()
            + pen;
        best = Some(best.map_or(e, |b| b.min(e)));
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

    fn cds_start(&self) -> usize {
        self.utr5.len()
    }

    fn cds_end(&self) -> usize {
        self.cds_start() + 3 * self.protein.len()
    }

    /// The design under `profile`, or without the term when it is inert.
    fn design(&self, lambda: f64, rc: RegionConstraints, profile: UtrProfile) -> Design {
        let (params, usage) = params_usage();
        design_with_utr_profile(
            &params,
            self.protein,
            &usage,
            lambda,
            &self.utr5,
            &self.utr3,
            rc,
            profile,
            ScanConfig::default(),
        )
        .expect("designs")
    }

    /// Every `(constrained MFE kcal, adaptation log)` the case admits under
    /// `weights`: the joint product of flank spellings and CDS encodings. Both
    /// coordinates are λ-independent, so one sweep serves every λ.
    fn scored(&self, spec: &Spec, weights: [f64; 4]) -> Vec<(f64, f64)> {
        let (params, usage) = params_usage();
        let aa = protein_indices(self.protein);
        let mut out = Vec::new();
        for u5 in flank_choices(&self.utr5) {
            for u3 in flank_choices(&self.utr3) {
                let flank = score_flank(weights, &designed_bytes(&self.utr5, &u5))
                    + score_flank(weights, &designed_bytes(&self.utr3, &u3));
                for enc in encodings(&aa) {
                    let full = full_bytes(&aa, &enc, &u5, &u3);
                    out.push((
                        f64::from(constrained_mfe_dk(&params, &full, spec)) / 100.0,
                        cds_cai_log(&usage, &aa, &enc) + flank,
                    ));
                }
            }
        }
        out
    }
}

/// The cases every check runs over: small enough that `4^|designed| ×
/// encodings × structures` is enumerable, and covering a designed flank on each
/// side alone, both at once, and a mixed chain.
fn cases() -> Vec<Case> {
    vec![
        Case::new("MK", UtrSpec::Designed(4), UtrSpec::Fixed(String::new())),
        Case::new("MK", UtrSpec::Fixed(String::new()), UtrSpec::Designed(4)),
        Case::new("MW", UtrSpec::Designed(3), UtrSpec::Designed(3)),
        Case::new(
            "MK",
            UtrSpec::Mixed("ACNN".into()),
            UtrSpec::Fixed("GU".into()),
        ),
    ]
}

/// The region tables every parity check sweeps: inert, the shipped default, and
/// the table that forbids a flank every pair it could form — the case where the
/// profile is the only thing left pricing the flank.
fn tables() -> Vec<(&'static str, RegionConstraints)> {
    vec![
        ("inert", RegionConstraints::default()),
        (
            "five-prime",
            RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        ),
        ("no-flank-pairs", no_flank_pairs()),
    ]
}

/// Forbid every class with an endpoint in a flank, leaving only `cds-cds`. A
/// designed flank under it cannot pair with anything, so folding energy is
/// indifferent to its spelling.
fn no_flank_pairs() -> RegionConstraints {
    [
        PairClass::Utr5Utr5,
        PairClass::Utr5Cds,
        PairClass::Utr5Utr3,
        PairClass::CdsUtr3,
        PairClass::Utr3Utr3,
    ]
    .iter()
    .fold(RegionConstraints::default(), |rc, &c| {
        rc.with(c, Enforcement::Forbid)
    })
}

/// The designed flank bytes of an emitted design, 5′ run then 3′ run.
fn emitted_designed(d: &Design, case: &Case) -> Vec<u8> {
    let full = d.sequence.as_bytes();
    let mut out = designed_bytes(&case.utr5, &full[..case.cds_start()]);
    out.extend(designed_bytes(&case.utr3, &full[case.cds_end()..]));
    out
}

// ---------------------------------------------------------------------------
// 1. Floors
// ---------------------------------------------------------------------------

#[test]
fn a_zero_strength_profile_is_byte_identical_to_no_term() {
    let (params, usage) = params_usage();
    for case in cases() {
        for (label, rc) in tables() {
            for &lambda in &LAMBDAS {
                let without = design_with_utr_spec(
                    &params,
                    case.protein,
                    &usage,
                    lambda,
                    &case.utr5,
                    &case.utr3,
                    rc,
                    ScanConfig::default(),
                )
                .expect("designs");
                let with = case.design(
                    lambda,
                    rc,
                    UtrProfile::new(C_LEANING, 0.0).expect("valid profile"),
                );
                assert_eq!(
                    without, with,
                    "zero-strength profile moved the design [{} {label} λ={lambda}]",
                    case.protein,
                );
            }
        }
    }
}

#[test]
fn a_uniform_profile_is_byte_identical_to_no_term() {
    let (params, usage) = params_usage();
    for case in cases() {
        for &lambda in &LAMBDAS {
            let rc = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
            let without = design_with_utr_spec(
                &params,
                case.protein,
                &usage,
                lambda,
                &case.utr5,
                &case.utr3,
                rc,
                ScanConfig::default(),
            )
            .expect("designs");
            let with = case.design(
                lambda,
                rc,
                UtrProfile::new([0.25; 4], 1.0).expect("valid profile"),
            );
            assert_eq!(
                without, with,
                "uniform profile moved the design [{} λ={lambda}]",
                case.protein,
            );
        }
    }
}

#[test]
fn a_fully_given_flank_is_unaffected_by_any_profile() {
    // Both flanks fixed: no position has a choice, so no profile can price
    // anything and every field must match the design without the term.
    let fixed = Case::new(
        "MKW",
        UtrSpec::Fixed("ACGUA".into()),
        UtrSpec::Fixed("GCAU".into()),
    );
    for (label, rc) in tables() {
        for &lambda in &LAMBDAS {
            let off = fixed.design(lambda, rc, UtrProfile::inert());
            for freqs in [A_LEANING, C_LEANING] {
                let on = fixed.design(lambda, rc, UtrProfile::new(freqs, 4.0).expect("valid"));
                assert_eq!(
                    off, on,
                    "a profile moved a fully given flank [{label} λ={lambda}]",
                );
                assert_eq!(on.utr_profile_log, 0.0, "a given flank earned a preference");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. The joint enumeration oracle
// ---------------------------------------------------------------------------

#[test]
fn matches_the_joint_enumeration_oracle() {
    for case in cases() {
        for (label, rc) in tables() {
            let spec = Spec::new(&rc, case.cds_start(), case.cds_end());
            for freqs in [A_LEANING, C_LEANING] {
                let weights = oracle_weights(freqs, 2.0);
                let scored = case.scored(&spec, weights);
                let profile = UtrProfile::new(freqs, 2.0).expect("valid profile");
                for &lambda in &LAMBDAS {
                    let d = case.design(lambda, rc, profile);
                    let best = scored
                        .iter()
                        .map(|&(mfe, adaptation)| brd(lambda, mfe, adaptation))
                        .fold(f64::INFINITY, f64::min);
                    let got = brd(lambda, d.mfe_kcal_mol, d.cai_log + d.utr_profile_log);
                    assert!(
                        (got - best).abs() <= BRD_TOL,
                        "design is not the joint optimum [{} {label} λ={lambda}]: \
                         got {got:.4}, oracle {best:.4}",
                        case.protein,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The reported halves
// ---------------------------------------------------------------------------

#[test]
fn the_reported_halves_are_cds_cai_and_flank_preference() {
    let (_, usage) = params_usage();
    for case in cases() {
        let aa = protein_indices(case.protein);
        let rc = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
        for freqs in [A_LEANING, C_LEANING] {
            let weights = oracle_weights(freqs, 2.0);
            let profile = UtrProfile::new(freqs, 2.0).expect("valid profile");
            for &lambda in &LAMBDAS {
                let d = case.design(lambda, rc, profile);
                let flank = score_flank(weights, &emitted_designed(&d, &case));
                assert!(
                    (d.utr_profile_log - flank).abs() < 1e-9,
                    "utr_profile_log is not the profile on the emitted flank [{} λ={lambda}]: \
                     {} vs {flank}",
                    case.protein,
                    d.utr_profile_log,
                );
                // The CAI half must still be the CDS codons alone: recover the
                // emitted encoding and score it independently.
                let cds = &d.sequence.as_bytes()[case.cds_start()..case.cds_end()];
                let enc = encoding_of(&aa, cds);
                assert!(
                    (d.cai_log - cds_cai_log(&usage, &aa, &enc)).abs() < 1e-9,
                    "cai_log is not the CDS-only adaptation [{} λ={lambda}]",
                    case.protein,
                );
            }
        }
    }
}

/// The per-residue codon choices an emitted CDS spells.
fn encoding_of(aa: &[usize], cds: &[u8]) -> Vec<usize> {
    let code = |c: u8| b"ACGU".iter().position(|&n| n == c).expect("nucleotide") as u8;
    aa.iter()
        .enumerate()
        .map(|(i, &a)| {
            let trip = [code(cds[3 * i]), code(cds[3 * i + 1]), code(cds[3 * i + 2])];
            AMINO_ACIDS[a]
                .codons
                .iter()
                .position(|c| c.nuc == trip)
                .expect("emitted codon encodes the residue")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 4. The term bites
// ---------------------------------------------------------------------------

#[test]
fn the_profile_moves_designs_the_term_would_otherwise_leave_alone() {
    // A term that never changes an answer would pass every parity check above
    // and be worthless, so the count of cases it moves is tracked and asserted
    // non-zero rather than assumed.
    let mut moved = 0;
    let mut total = 0;
    for case in cases() {
        for (_, rc) in tables() {
            for &lambda in &LAMBDAS {
                let off = case.design(lambda, rc, UtrProfile::inert());
                let on = case.design(
                    lambda,
                    rc,
                    UtrProfile::new(A_LEANING, 4.0).expect("valid profile"),
                );
                total += 1;
                if off.sequence != on.sequence {
                    moved += 1;
                }
            }
        }
    }
    assert!(
        moved > 0,
        "the profile changed no design in {total} cases — the term is inert in practice",
    );
}

// ---------------------------------------------------------------------------
// 5. It prices what nothing else prices
// ---------------------------------------------------------------------------

#[test]
fn with_every_flank_pair_forbidden_the_profile_alone_decides_the_spelling() {
    // Under this table a designed flank can pair with nothing, so the folding
    // energy is exactly indifferent to its spelling — measured directly below by
    // two profiles giving the same MFE and CAI on different flanks. Whatever the
    // flank comes back as is therefore the profile's doing alone, and for a sum
    // of independent per-position weights that is the argmax at every position.
    let case = Case::new("MW", UtrSpec::Designed(3), UtrSpec::Designed(3));
    let rc = no_flank_pairs();
    for (freqs, want) in [(A_LEANING, b'A'), (C_LEANING, b'C')] {
        for &lambda in &LAMBDAS {
            let d = case.design(lambda, rc, UtrProfile::new(freqs, 2.0).expect("valid"));
            let flank = emitted_designed(&d, &case);
            assert!(
                flank.iter().all(|&b| b == want),
                "flank is not the profile's argmax homopolymer [λ={lambda}]: {}",
                String::from_utf8_lossy(&flank),
            );
        }
    }
    // The indifference the paragraph above claims, measured rather than asserted:
    // two profiles that spell opposite flanks reach the same energy and the same
    // CDS adaptation, so nothing but the profile distinguished them.
    let a = case.design(0.5, rc, UtrProfile::new(A_LEANING, 2.0).expect("valid"));
    let c = case.design(0.5, rc, UtrProfile::new(C_LEANING, 2.0).expect("valid"));
    assert_eq!(a.mfe_kcal_mol, c.mfe_kcal_mol, "flank spelling moved MFE");
    assert_eq!(a.cai_log, c.cai_log, "flank spelling moved the CDS CAI");
}

// ---------------------------------------------------------------------------
// 6. Fold parity
// ---------------------------------------------------------------------------

#[test]
fn the_reported_energy_is_an_independent_fold_of_the_spelled_sequence() {
    // The term rides the adaptation half, so the energy the design is credited
    // with must be exactly the sequence's own constrained-optimal fold — checked
    // here against an independent folder on the inert table, where constrained
    // and free coincide.
    let (params, _) = params_usage();
    for case in cases() {
        for freqs in [A_LEANING, C_LEANING] {
            let profile = UtrProfile::new(freqs, 2.0).expect("valid profile");
            for &lambda in &LAMBDAS {
                let d = case.design(lambda, RegionConstraints::default(), profile);
                let independent = linearfold_fold(&params, d.sequence.as_bytes(), usize::MAX)
                    .expect("folds")
                    .1
                    .to_kcal_mol();
                assert!(
                    (d.mfe_kcal_mol - independent).abs() < 1e-9,
                    "reported MFE is not the sequence's own fold [{} λ={lambda}]: \
                     {} vs {independent}",
                    case.protein,
                    d.mfe_kcal_mol,
                );
            }
        }
    }
}
