//! Alphabet subsets on designed flank positions.
//!
//! A region constraint restricts the structure a design is **scored against**,
//! so a design can be the exact optimum of its constrained problem and still
//! fold differently once synthesized. A constraint on the **sequence** does not
//! have that gap: the fold is left unrestricted, so whatever sequence survives,
//! the returned structure is its true minimum. Restricting both flanks to one
//! mutually non-pairing subset — `{A,C}`, `{A,G}` or `{C,U}` — makes every
//! flank-to-flank helix unspellable, with no verdict to read afterwards.
//!
//! Five checks:
//!
//! 1. **Byte-identity floors.** An all-`N` mask is the designed flank it has
//!    always been, an all-literal mask is the fixed flank, and neither moves
//!    under any profile that did not move it before.
//! 2. **The re-normalization floor.** With a profile whose argmax the
//!    restriction **excludes**, the design equals an independent brute-force
//!    argmin over the restricted sequence set. This is the check that catches a
//!    span closed form left over-estimating what a restricted position can earn;
//!    no byte-identity test would, because both answers are self-consistent.
//! 3. **The reported halves survive it.** `cai_log` stays the CDS codons alone
//!    and `utr_profile_log` the profile's total on the emitted flank under the
//!    profile **as stated** — the re-normalization is an implementation detail
//!    of the bound and must not reach the report.
//! 4. **Modular by construction.** With a restriction active and the region table
//!    **inert**, the emitted sequence's own **free** fold contains zero `5-3`,
//!    `5-5` and `3-3` pairs. Checked against the free fold, never the designed
//!    structure — that distinction is the whole point of the instrument.
//! 5. **Constrained enumeration.** At tractable `n`, flank spellings over the
//!    restricted set × CDS encodings × nested structures, BRD-argmin,
//!    byte-exact — with the restriction asserted to bite.
//!
//! The IUPAC expansion, the structure enumerator, the CAI formula and the
//! profile's weights are re-implemented here rather than shared with the crate,
//! so a bug in the substrate cannot hide inside its own oracle.

use std::collections::HashMap;

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_with_utr_profile, design_with_utr_spec, sequence_gap, CdsWindow, CodonUsage, Design,
    Enforcement, PairClass, Params, RegionConstraints, ScanConfig, UtrProfile, UtrSpec,
};
use rna_mfe::evaluate;

/// One secondary structure as its list of `(i, j)` base pairs.
type Pairs = Vec<(usize, usize)>;
/// A set of structures over one interval.
type Structs = Vec<Pairs>;
/// Per-interval memo for the structure enumeration.
type Memo = HashMap<(usize, usize), Structs>;

/// BRD tolerance in dekacalories — the rounded-integer vs continuous
/// objective gap, exactly as `tests/brd_parity.rs`.
const BRD_TOL: f64 = 0.5;

/// The λ values every parity check sweeps (low / mid / high).
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// A profile whose argmax is `G` — the nucleotide **excluded** by both the
/// `{A,C}` and the `{C,U}` restriction, which is what makes the
/// re-normalization constant non-zero and the re-normalization path actually exercised. The
/// two survivors carry distinct weights, so the restriction leaves a real choice
/// rather than a tie.
const G_FAVOURING: [f64; 4] = [0.15, 0.20, 0.45, 0.20];
/// A profile whose argmax is `A`, which **is** available under `{A,C}`: the
/// control where the re-normalization constant is zero and every restricted edge
/// keeps its raw weight.
const A_FAVOURING: [f64; 4] = [0.45, 0.20, 0.15, 0.20];
/// The strength every profiled check uses.
const STRENGTH: f64 = 2.0;

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

fn nuc_byte(n: usize) -> u8 {
    b"ACGU"[n]
}

fn nuc_index(b: u8) -> usize {
    b"ACGU".iter().position(|&n| n == b).expect("nucleotide")
}

/// The oracle's own IUPAC table, over the codes these cases use.
fn codes_of(code: u8) -> &'static [u8] {
    match code {
        b'A' => b"A",
        b'C' => b"C",
        b'G' => b"G",
        b'U' => b"U",
        b'M' => b"AC",
        b'R' => b"AG",
        b'Y' => b"CU",
        b'N' => b"ACGU",
        _ => panic!("unhandled IUPAC code {:?}", code as char),
    }
}

/// Fixed-λ BRD on rounded dekacalories, matching how the oracle and the design
/// both round at the scalarization boundary.
fn brd(lambda: f64, mfe_kcal: f64, adaptation_log: f64) -> f64 {
    let mfe_dk = (mfe_kcal * 100.0).round();
    let adaptation_dk = (adaptation_log * 100.0).round();
    lambda * mfe_dk - (1.0 - lambda) * adaptation_dk
}

/// The oracle's own copy of the profile weights: `β·log(f / f_max)`, normalized
/// to a maximum of zero over the **full** alphabet. Deliberately *not* the
/// per-position re-normalization — the report is specified against these, so
/// scoring against them is what pins that the constant never leaks out.
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
    flank.iter().map(|&b| weights[nuc_index(b)]).sum()
}

/// Whether ASCII nucleotides `a` and `b` form a canonical Watson–Crick / wobble
/// pair.
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

/// The per-position IUPAC codes of a flank spec, one byte per position.
fn mask_of(spec: &UtrSpec) -> Vec<u8> {
    match spec {
        UtrSpec::Fixed(s) => s.as_bytes().to_vec(),
        UtrSpec::Designed(len) => vec![b'N'; *len],
        UtrSpec::Mixed(s) => s.as_bytes().to_vec(),
    }
}

/// Every spelling a flank spec admits — the product of each position's IUPAC
/// class, which is what makes this an enumeration over the **restricted**
/// sequence set rather than over all `4^len`.
fn flank_choices(spec: &UtrSpec) -> Vec<Vec<u8>> {
    let mask = mask_of(spec);
    let radix: Vec<usize> = mask.iter().map(|&b| codes_of(b).len()).collect();
    mixed_radix(&radix)
        .into_iter()
        .map(|digits| {
            mask.iter()
                .zip(digits)
                .map(|(&b, d)| codes_of(b)[d])
                .collect()
        })
        .collect()
}

/// The positions of a flank a profile prices: the ones with more than one
/// choice. A position pinned to a single code is a given one however it was
/// spelled, so it earns no preference — the same contract a literal always had.
fn designed_mask(spec: &UtrSpec) -> Vec<bool> {
    mask_of(spec)
        .iter()
        .map(|&b| codes_of(b).len() > 1)
        .collect()
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
        out.extend(
            AMINO_ACIDS[a].codons[ci]
                .nuc
                .iter()
                .map(|&n| nuc_byte(n as usize)),
        );
    }
    out.extend_from_slice(utr3);
    out
}

/// CDS-only additive log-CAI of one encoding, `EPSILON`-floored as the DFA's own
/// edge weights are.
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

/// The per-residue codon choices an emitted CDS spells.
fn encoding_of(aa: &[usize], cds: &[u8]) -> Vec<usize> {
    aa.iter()
        .enumerate()
        .map(|(i, &a)| {
            let trip = [
                nuc_index(cds[3 * i]) as u8,
                nuc_index(cds[3 * i + 1]) as u8,
                nuc_index(cds[3 * i + 2]) as u8,
            ];
            AMINO_ACIDS[a]
                .codons
                .iter()
                .position(|c| c.nuc == trip)
                .expect("emitted codon encodes the residue")
        })
        .collect()
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

/// The oracle's re-implementation of the region constraint, keyed on its own region
/// codes.
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
    /// `weights` — the joint product of restricted flank spellings and CDS
    /// encodings. Both coordinates are λ-independent, so one sweep serves every λ.
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

    /// The designed flank bytes of an emitted design, 5′ run then 3′ run.
    fn emitted_designed(&self, d: &Design) -> Vec<u8> {
        let full = d.sequence.as_bytes();
        let mut out = designed_bytes(&self.utr5, &full[..self.cds_start()]);
        out.extend(designed_bytes(&self.utr3, &full[self.cds_end()..]));
        out
    }
}

/// The cases the restricted checks run over: a restriction on each side alone,
/// both at once, and a mixed chain interleaving pinned and restricted positions.
/// Small enough that `|subset|^designed × encodings × structures` enumerates.
fn restricted_cases() -> Vec<Case> {
    vec![
        Case::new("MK", UtrSpec::Mixed("MMMM".into()), empty()),
        Case::new("MK", empty(), UtrSpec::Mixed("MMMM".into())),
        Case::new(
            "MW",
            UtrSpec::Mixed("MMM".into()),
            UtrSpec::Mixed("MMM".into()),
        ),
        Case::new(
            "MK",
            UtrSpec::Mixed("ACMM".into()),
            UtrSpec::Fixed("CA".into()),
        ),
    ]
}

fn empty() -> UtrSpec {
    UtrSpec::Fixed(String::new())
}

/// The region tables the parity checks sweep: inert, `five-prime`, and the table
/// forbidding a flank every pair it could form.
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

// ---------------------------------------------------------------------------
// 1. Byte-identity floors
// ---------------------------------------------------------------------------

#[test]
fn an_all_n_mask_is_the_designed_flank_it_has_always_been() {
    let (params, usage) = params_usage();
    let profiles = [
        UtrProfile::inert(),
        UtrProfile::new(G_FAVOURING, STRENGTH).expect("valid"),
    ];
    for (n5, n3) in [(4usize, 0usize), (0, 4), (3, 3)] {
        let masked = Case::new(
            "MK",
            UtrSpec::Mixed("N".repeat(n5)),
            UtrSpec::Mixed("N".repeat(n3)),
        );
        let designed = Case::new("MK", UtrSpec::Designed(n5), UtrSpec::Designed(n3));
        for (label, rc) in tables() {
            for &lambda in &LAMBDAS {
                for profile in profiles {
                    assert_eq!(
                        masked.design(lambda, rc, profile),
                        designed.design(lambda, rc, profile),
                        "all-N mask diverged from a designed flank [{n5}/{n3} {label} λ={lambda}]",
                    );
                }
            }
        }
    }
    // And the empty mask still collapses to the CDS-only substrate verbatim.
    let empty_mask = design_with_utr_spec(
        &params,
        "MKW",
        &usage,
        0.5,
        &UtrSpec::Mixed(String::new()),
        &UtrSpec::Mixed(String::new()),
        RegionConstraints::default(),
        ScanConfig::default(),
    )
    .expect("designs");
    let no_flank = design_with_utr_spec(
        &params,
        "MKW",
        &usage,
        0.5,
        &empty(),
        &empty(),
        RegionConstraints::default(),
        ScanConfig::default(),
    )
    .expect("designs");
    assert_eq!(empty_mask, no_flank);
}

#[test]
fn an_all_literal_mask_is_the_fixed_flank() {
    // Every position pinned: one out-edge each, no preference to price, and no
    // re-normalization constant — so a profile must leave it exactly alone, the
    // same contract a `Fixed` flank has always had.
    let masked = Case::new(
        "MKW",
        UtrSpec::Mixed("ACGUA".into()),
        UtrSpec::Mixed("GCAU".into()),
    );
    let fixed = Case::new(
        "MKW",
        UtrSpec::Fixed("ACGUA".into()),
        UtrSpec::Fixed("GCAU".into()),
    );
    for (label, rc) in tables() {
        for &lambda in &LAMBDAS {
            for freqs in [G_FAVOURING, A_FAVOURING] {
                let profile = UtrProfile::new(freqs, STRENGTH).expect("valid");
                let m = masked.design(lambda, rc, profile);
                assert_eq!(
                    m,
                    fixed.design(lambda, rc, profile),
                    "all-literal mask diverged from a fixed flank [{label} λ={lambda}]",
                );
                assert_eq!(m.utr_profile_log, 0.0, "a pinned flank earned a preference");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. The re-normalization floor
// ---------------------------------------------------------------------------

#[test]
fn a_restriction_excluding_the_argmax_matches_the_brute_force_argmin() {
    // The case re-normalization exists for. `{A, C}` excludes `G`, the profile's argmax, so
    // every restricted position's best available weight is strictly below zero
    // and the closed form would over-estimate without the per-position
    // re-normalization. Asserted rather than assumed: a case that degenerated to
    // a zero constant would read as coverage while testing nothing.
    let weights = oracle_weights(G_FAVOURING, STRENGTH);
    let best_available = weights[nuc_index(b'A')].max(weights[nuc_index(b'C')]);
    assert!(
        best_available < -1e-6,
        "the restriction must exclude the profile's argmax: c_i = {best_available}",
    );
    assert_eq!(weights.iter().copied().fold(f64::MIN, f64::max), 0.0);

    let profile = UtrProfile::new(G_FAVOURING, STRENGTH).expect("valid");
    for case in restricted_cases() {
        for (label, rc) in tables() {
            let spec = Spec::new(&rc, case.cds_start(), case.cds_end());
            let scored = case.scored(&spec, weights);
            for &lambda in &LAMBDAS {
                let d = case.design(lambda, rc, profile);
                let best = scored
                    .iter()
                    .map(|&(mfe, adaptation)| brd(lambda, mfe, adaptation))
                    .fold(f64::INFINITY, f64::min);
                let got = brd(lambda, d.mfe_kcal_mol, d.cai_log + d.utr_profile_log);
                assert!(
                    (got - best).abs() <= BRD_TOL,
                    "restricted design is not the argmin over the restricted set \
                     [{} {label} λ={lambda}]: got {got:.4}, oracle {best:.4}",
                    case.protein,
                );
            }
        }
    }
}

#[test]
fn a_restricted_design_spells_only_its_subset() {
    // The mechanism, checked on the emitted sequence rather than on the graph:
    // no position may spell outside its own IUPAC class, and the restriction
    // must actually bite — the unrestricted design of the same case reaches for
    // a nucleotide the subset forbids.
    let profile = UtrProfile::new(G_FAVOURING, STRENGTH).expect("valid");
    let mut bit = 0;
    for case in restricted_cases() {
        let masks = [mask_of(&case.utr5), mask_of(&case.utr3)];
        for (_, rc) in tables() {
            for &lambda in &LAMBDAS {
                let d = case.design(lambda, rc, profile);
                let full = d.sequence.as_bytes();
                let flanks = [&full[..case.cds_start()], &full[case.cds_end()..]];
                for (mask, flank) in masks.iter().zip(flanks) {
                    for (&code, &spelled) in mask.iter().zip(flank) {
                        assert!(
                            codes_of(code).contains(&spelled),
                            "position spelled {:?} outside its class {:?} [{} λ={lambda}]",
                            spelled as char,
                            code as char,
                            case.protein,
                        );
                    }
                }
                let free = Case::new(case.protein, unrestrict(&case.utr5), unrestrict(&case.utr3));
                let f = free.design(lambda, rc, profile).sequence;
                let f = f.as_bytes();
                let outside = |mask: &[u8], span: &[u8]| {
                    mask.iter()
                        .zip(span)
                        .any(|(&code, &b)| !codes_of(code).contains(&b))
                };
                if outside(&masks[0], &f[..case.cds_start()])
                    || outside(&masks[1], &f[case.cds_end()..])
                {
                    bit += 1;
                }
            }
        }
    }
    assert!(
        bit > 0,
        "no case's unrestricted design left the subset — the restriction never bit",
    );
}

/// The same flank with every restricted position freed, keeping pinned ones
/// pinned — the control the bite check compares against.
fn unrestrict(spec: &UtrSpec) -> UtrSpec {
    let freed: String = mask_of(spec)
        .iter()
        .map(|&b| if codes_of(b).len() > 1 { b'N' } else { b })
        .map(char::from)
        .collect();
    UtrSpec::Mixed(freed)
}

// ---------------------------------------------------------------------------
// 3. The reported halves survive the re-normalization
// ---------------------------------------------------------------------------

#[test]
fn the_reported_halves_survive_the_renormalization() {
    // The half that is easy to forget. The scan accumulates
    // `CAI_cds + U(v) − Σ c_i`, so a report that subtracts only `U(v)` leaves
    // `cai_log` short by `Σ c_i` — invisible on any case where the restriction
    // happens to keep the argmax, which is why the profile here is the one whose
    // argmax the restriction excludes.
    let (_, usage) = params_usage();
    let weights = oracle_weights(G_FAVOURING, STRENGTH);
    let profile = UtrProfile::new(G_FAVOURING, STRENGTH).expect("valid");
    for case in restricted_cases() {
        let aa = protein_indices(case.protein);
        for (label, rc) in tables() {
            for &lambda in &LAMBDAS {
                let d = case.design(lambda, rc, profile);
                let flank = score_flank(weights, &case.emitted_designed(&d));
                assert!(
                    (d.utr_profile_log - flank).abs() < 1e-9,
                    "utr_profile_log is not the stated profile on the emitted flank \
                     [{} {label} λ={lambda}]: {} vs {flank}",
                    case.protein,
                    d.utr_profile_log,
                );
                let cds = &d.sequence.as_bytes()[case.cds_start()..case.cds_end()];
                let enc = encoding_of(&aa, cds);
                assert!(
                    (d.cai_log - cds_cai_log(&usage, &aa, &enc)).abs() < 1e-9,
                    "cai_log is not the CDS-only adaptation [{} {label} λ={lambda}]: \
                     {} vs {}",
                    case.protein,
                    d.cai_log,
                    cds_cai_log(&usage, &aa, &enc),
                );
                // The constant is genuinely non-zero here, so the two assertions
                // above were exercised rather than trivially satisfied.
                assert!(
                    d.utr_profile_log < -1e-6,
                    "every designed position spelled the argmax — Σ c_i was inert \
                     [{} {label} λ={lambda}]",
                    case.protein,
                );
            }
        }
    }
}

#[test]
fn a_restriction_containing_the_argmax_needs_no_constant() {
    // The control for the check above: with `{A, C}` and an `A`-favouring
    // profile the re-normalization constant is exactly zero at every position,
    // so the flank comes back as the argmax homopolymer at its best possible
    // score. A report keyed on that score's *value* would go silent here, which
    // is why the CLI keys on whether a profile was supplied.
    let profile = UtrProfile::new(A_FAVOURING, STRENGTH).expect("valid");
    let case = Case::new(
        "MW",
        UtrSpec::Mixed("MMM".into()),
        UtrSpec::Mixed("MMM".into()),
    );
    for &lambda in &LAMBDAS {
        let d = case.design(lambda, no_flank_pairs(), profile);
        let flank = case.emitted_designed(&d);
        assert!(
            flank.iter().all(|&b| b == b'A'),
            "restricted flank is not the available argmax homopolymer [λ={lambda}]: {}",
            String::from_utf8_lossy(&flank),
        );
        assert_eq!(
            d.utr_profile_log, 0.0,
            "the best possible preference score is not zero [λ={lambda}]",
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Modular by construction — the point of the scope
// ---------------------------------------------------------------------------

/// With a restriction active and **the region table inert**, the emitted
/// sequence's own **free** fold contains zero `5-3`, `5-5` and `3-3` pairs.
///
/// Checked against the free fold, never the design's own structure. That is the
/// distinction between this instrument and the region constraint it complements: the
/// constraint constrains the structure a design is scored against and reports the gap
/// afterwards, while a sequence-space restriction leaves the fold unconstrained
/// and removes the sequences that could pair at all.
#[test]
fn a_non_pairing_restriction_leaves_no_flank_pair_in_the_free_fold() {
    let (params, _) = params_usage();
    let inert = RegionConstraints::default();
    let flank_classes = [
        PairClass::Utr5Utr3,
        PairClass::Utr5Utr5,
        PairClass::Utr3Utr3,
    ];
    // Both sidednesses, both flanks at once, and a mixed chain whose pinned
    // positions are themselves drawn from the subset — a pin outside it would
    // reintroduce exactly the partner the restriction removed.
    let cases = [
        (
            "5′ only",
            "MKWFYLA",
            UtrSpec::Mixed("MMMMMMMMMMMM".into()),
            empty(),
        ),
        (
            "3′ only",
            "MKWFYLA",
            empty(),
            UtrSpec::Mixed("MMMMMMMMMMMM".into()),
        ),
        (
            "both",
            "MKWFYLA",
            UtrSpec::Mixed("MMMMMMMMMMMM".into()),
            UtrSpec::Mixed("MMMMMMMMMMMM".into()),
        ),
        (
            "mixed chain",
            "MKWFYLA",
            UtrSpec::Mixed("ACMMMMMMMMCA".into()),
            UtrSpec::Mixed("MMMMCACAMMMM".into()),
        ),
        (
            "C,U both",
            "MKWFYLA",
            UtrSpec::Mixed("YYYYYYYYYYYY".into()),
            UtrSpec::Mixed("YYYYYYYYYYYY".into()),
        ),
    ];
    let mut control_pairs = 0usize;
    for (label, protein, utr5, utr3) in cases {
        let case = Case::new(protein, utr5, utr3);
        let cds = CdsWindow::new(case.cds_start(), protein.len());
        for &lambda in &[0.5, 0.99] {
            let d = case.design(lambda, inert, UtrProfile::inert());
            let gap = sequence_gap(&params, &d.sequence, cds, inert).expect("folds");
            for class in flank_classes {
                assert_eq!(
                    gap.classes.count(class),
                    0,
                    "{class:?} pair in the FREE fold of a restricted design \
                     [{label} λ={lambda}]\n  seq: {}\n  free: {}",
                    d.sequence,
                    gap.free_structure,
                );
            }

            // Non-vacuity: the same case with the restriction lifted does form
            // flank-to-flank pairs, so the zero above is the restriction's doing
            // and not an artifact of the instance being too short to pair.
            let free = Case::new(protein, unrestrict(&case.utr5), unrestrict(&case.utr3));
            let fd = free.design(lambda, inert, UtrProfile::inert());
            let fgap = sequence_gap(&params, &fd.sequence, cds, inert).expect("folds");
            control_pairs += flank_classes
                .iter()
                .map(|&c| fgap.classes.count(c))
                .sum::<usize>();
        }
    }
    assert!(
        control_pairs > 0,
        "no unrestricted control formed a flank-to-flank pair — the test cannot \
         distinguish the restriction from an instance that never pairs",
    );
}
