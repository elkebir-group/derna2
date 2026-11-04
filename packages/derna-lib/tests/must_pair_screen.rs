//! Must-pair (required-position) constraint oracle.
//!
//! A must-pair constraint is enforced nowhere near the `P`-write: it is a
//! predicate consulted wherever the scan declares a **range unpaired** — the two
//! single-base extensions, a hairpin's interior, a single-branch loop's two
//! flanks, and a multi-loop's two unpaired stretches. Those ranges are delimited
//! by coordinates the enclosing closure already enumerates, so the constraint is
//! expected to land as a *narrowing of an enumeration range* rather than as a
//! filter over an already-minimized set. This harness is what turns that
//! expectation into evidence.
//!
//! Six checks:
//!
//! 1. **Byte-identical empty-constraint floor.** With no required position, the
//!    constrained entry reproduces the unconstrained design byte-for-byte — sequence,
//!    MFE and CAI — under both internal-loop closures. If this fails nothing else
//!    measured here is meaningful.
//! 2. **Constrained-fold oracle (not fold-then-filter).** For each
//!    `(protein, required positions, λ)`, enumerate every encoding, enumerate all
//!    nested canonical structures, keep only those pairing every required
//!    position, score with [`evaluate`], take the per-encoding constrained MFE,
//!    form BRD with the CAI, and argmin over encodings. The constrained optimum is
//!    frequently a structure that is *suboptimal* unconstrained, so folding under
//!    the constraint rather than filtering afterwards is essential.
//! 3. **The constraint bites, and its post-condition holds.** Every required
//!    position is paired in the recovered structure, and a tracked count of cases
//!    where the constrained optimum differs from the unconstrained one is asserted
//!    non-zero, so the oracle is not vacuously green.
//! 4. **Infeasibility is reported, not papered over.** A required position with
//!    no legal partner yields `None`.
//! 5. **Reduction composition.** The constrained fold under the Lyngsø–Zuker /
//!    one-sided / `multi_open` reductions equals the un-reduced constrained fold
//!    byte-exactly, at lengths that engage every reduction and with the required
//!    positions placed **mid-sequence** so they land inside deep loop interiors
//!    rather than near an end. A reduction that dropped a candidate the direct
//!    enumeration kept would diverge here.
//! 6. **The `multi_open` eviction is not over-aggressive.** Both closures share
//!    that path, so (5) cannot see a fault in it, and no enumerable construct has
//!    a multi-loop optimum. Checked instead by requiring every position the
//!    unconstrained optimum already pairs and asserting nothing changes.

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_must_paired, design_must_paired_pushing, design_with_config, linearfold_fold,
    CodonUsage, InternalLoopMode, Params, ScanConfig,
};
use rna_mfe::evaluate;

/// One secondary structure as its list of `(i, j)` base pairs.
type Pairs = Vec<(usize, usize)>;

/// b → ∞: a beam wide enough that `rna-linearfold` runs the exact lattice.
const EXACT: usize = 1 << 20;

/// BRD tolerance in dekacalories — absorbs the rounded-integer vs continuous
/// objective gap, exactly as `tests/brd_parity.rs`.
const BRD_TOL: f64 = 0.5;

/// The λ values swept by every parity check (low / mid / high).
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// The 20 standard amino acids, single-letter — the alphabet the randomized
/// sweep draws proteins from.
const ONE_LETTER: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

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

/// The CDS bytes for one encoding.
fn cds_bytes(aa: &[usize], encoding: &[usize]) -> Vec<u8> {
    aa.iter()
        .zip(encoding)
        .flat_map(|(&a, &ci)| AMINO_ACIDS[a].codons[ci].nuc.iter().map(|&n| nuc_byte(n)))
        .collect()
}

/// Additive log-CAI of one encoding (the same `EPSILON`-floored formula the
/// DFA's edge weights use).
fn cai_log(usage: &CodonUsage, aa: &[usize], encoding: &[usize]) -> f64 {
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

/// Visit every nested canonical secondary structure of `seq` with minimum
/// hairpin loop 3, calling `visit` on each one's pair list — the exact structure
/// space the CKY grammar folds over (de-risked by
/// `structure_enumerator_matches_linearfold`).
///
/// Streaming rather than materializing. The structure count is exponential —
/// ~2·10⁵ at 27 nt and ~8·10⁶ at 33 nt — and the multi-loop cases need those
/// lengths, so holding the whole set would cost gigabytes for no reason. `open`
/// is the worklist of intervals still to be resolved and `pairs` the structure
/// under construction; both are restored before returning, so one buffer serves
/// the whole walk.
fn walk_structures(
    seq: &[u8],
    open: &mut Vec<(usize, usize)>,
    pairs: &mut Pairs,
    visit: &mut impl FnMut(&[(usize, usize)]),
) {
    let Some((lo, hi)) = open.pop() else {
        visit(pairs);
        return;
    };
    if lo >= hi {
        walk_structures(seq, open, pairs, visit);
        open.push((lo, hi));
        return;
    }
    // `lo` unpaired.
    open.push((lo + 1, hi));
    walk_structures(seq, open, pairs, visit);
    open.pop();
    // `lo` pairs with `k` (min hairpin 3 ⇒ k ≥ lo + 4), splitting the interval
    // into the enclosed part and the remainder.
    for k in (lo + 4)..hi {
        if !canonical(seq[lo], seq[k]) {
            continue;
        }
        pairs.push((lo, k));
        open.push((k + 1, hi));
        open.push((lo + 1, k));
        walk_structures(seq, open, pairs, visit);
        open.pop();
        open.pop();
        pairs.pop();
    }
    open.push((lo, hi));
}

/// [`walk_structures`] over a whole sequence.
fn for_each_structure(seq: &[u8], mut visit: impl FnMut(&[(usize, usize)])) {
    let mut open = vec![(0, seq.len())];
    let mut pairs = Pairs::new();
    walk_structures(seq, &mut open, &mut pairs, &mut visit);
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

/// Whether `pairs` leaves every position in `required` paired — the oracle's own
/// reading of the predicate, deliberately independent of the crate's constraint so a
/// shared bug cannot hide.
fn satisfies(pairs: &[(usize, usize)], required: &[u32]) -> bool {
    required.iter().all(|&r| {
        pairs
            .iter()
            .any(|&(i, j)| i == r as usize || j == r as usize)
    })
}

/// The constrained MFE (dekacalories) of one spelled sequence, or `None` when no
/// nested structure pairs every required position.
fn constrained_mfe_dk(params: &Params, seq: &[u8], required: &[u32]) -> Option<i32> {
    let mut best: Option<i32> = None;
    for_each_structure(seq, |s| {
        if !satisfies(s, required) {
            return;
        }
        let e = evaluate(params, seq, &dot_bracket(seq.len(), s))
            .expect("structure scores")
            .dekacal();
        best = Some(best.map_or(e, |b| b.min(e)));
    });
    best
}

/// Whether `pairs` contains a multi-loop: a pair directly enclosing two or more
/// sibling pairs. Read off the pair list so the oracle's own notion of the
/// topology is independent of the scan's tables.
fn has_multiloop(pairs: &[(usize, usize)]) -> bool {
    pairs.iter().any(|&(i, j)| {
        let children = pairs
            .iter()
            .filter(|&&(a, b)| a > i && b < j)
            .filter(|&&(a, b)| !pairs.iter().any(|&(c, d)| c > i && d < j && c < a && b < d))
            .count();
        children >= 2
    })
}

/// What a case demonstrated, beyond passing: whether the constraint changed the
/// optimum at all, and whether it drove the fold through a multi-loop.
#[derive(Default)]
struct Outcome {
    /// The constrained optimum differed from the unconstrained one at some λ, so
    /// the case is not vacuous.
    bit: bool,
    /// Some λ's constrained optimum is a multi-loop-bearing structure, so the
    /// `multi_open` path carried it.
    multiloop: bool,
}

/// One must-pair oracle case: the design must match the constrained brute force
/// and satisfy the post-condition.
fn check_case(protein: &str, required: &[u32]) -> Outcome {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let aa = protein_indices(protein);

    // Per-encoding constrained (MFE, CAI) — both λ-independent, so precompute
    // once (a fixed encoding's best feasible structure minimizes MFE for every
    // λ > 0). Encodings with no feasible structure drop out entirely.
    let scored: Vec<(f64, f64)> = encodings(&aa)
        .iter()
        .filter_map(|enc| {
            let seq = cds_bytes(&aa, enc);
            constrained_mfe_dk(&params, &seq, required)
                .map(|dk| (f64::from(dk) / 100.0, cai_log(&usage, &aa, enc)))
        })
        .collect();

    let mut outcome = Outcome::default();
    for &lambda in &LAMBDAS {
        let label = format!("{protein} required={required:?} λ={lambda}");
        let design = design_must_paired(
            &params,
            protein,
            &usage,
            lambda,
            required,
            ScanConfig::default(),
        )
        .unwrap();

        // (4) Feasibility agreement, both directions.
        assert_eq!(
            design.is_some(),
            !scored.is_empty(),
            "feasibility disagreement [{label}]",
        );
        let Some(d) = design else {
            continue;
        };

        // (2) Constrained-fold optimality, both directions: the design cannot
        // beat the true constrained optimum (that would mean it violated the
        // requirement or the oracle missed a structure), nor fall short of it.
        let oracle = scored
            .iter()
            .map(|&(mfe, cai)| brd(lambda, mfe, cai))
            .fold(f64::INFINITY, f64::min);
        let design_brd = brd(lambda, d.mfe_kcal_mol, d.cai_log);
        assert!(
            (design_brd - oracle).abs() <= BRD_TOL,
            "constrained-oracle mismatch [{label}]: design={design_brd:.1} oracle={oracle:.1}",
        );

        // (3) Post-condition, read straight off the emitted dot-bracket.
        let bytes = d.structure.as_bytes();
        for &r in required {
            assert_ne!(
                bytes[r as usize], b'.',
                "required position {r} left unpaired [{label}]: {}",
                d.structure,
            );
        }
        outcome.multiloop |= has_multiloop(&pairs_of(&d.structure));

        let free = design_with_config(&params, protein, &usage, lambda, ScanConfig::default())
            .expect("unconstrained design");
        outcome.bit |= free.structure != d.structure || free.sequence != d.sequence;
    }
    outcome
}

/// The pairs of a dot-bracket structure (a balanced-parens stack walk).
fn pairs_of(structure: &str) -> Pairs {
    let mut stack = Vec::new();
    let mut pairs = Pairs::new();
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => pairs.push((stack.pop().expect("balanced"), i)),
            _ => {}
        }
    }
    pairs
}

fn check_cases(cases: &[(&str, &[u32])], label: &str) -> Outcome {
    let mut total = Outcome::default();
    for &(p, r) in cases {
        let o = check_case(p, r);
        total.bit |= o.bit;
        total.multiloop |= o.multiloop;
    }
    assert!(total.bit, "{label} never bit — oracle vacuous");
    total
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
        b"GGGGAAAACCCCUUUU".as_slice(),
    ] {
        let mut best = i32::MAX;
        for_each_structure(seq, |s| {
            let db = dot_bracket(seq.len(), s);
            best = best.min(evaluate(&params, seq, &db).expect("scores").dekacal());
        });
        let (_, energy) = linearfold_fold(&params, seq, EXACT).expect("folds");
        assert_eq!(best, energy.dekacal());
    }
}

/// (1) The byte-identical empty-constraint floor, under both internal-loop
/// closures and across the λ front. The whole screen rests on this: an inert
/// constraint must leave the exact substrate untouched.
#[test]
fn empty_requirement_is_byte_identical_to_the_unconstrained_design() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in [
        "MKTAYIAKQRLF",
        "DEYWCNQHKRFF",
        "FFLLSSPPHHRRKKMWGGVVAA",
        "MASMTGGQQMGRGSHHHHHH",
    ] {
        for mode in [InternalLoopMode::Lz, InternalLoopMode::Naive] {
            let config = ScanConfig {
                mode,
                ..ScanConfig::default()
            };
            for &lambda in &LAMBDAS {
                let free =
                    design_with_config(&params, protein, &usage, lambda, config).expect("design");
                let constrained = design_must_paired(&params, protein, &usage, lambda, &[], config)
                    .expect("design")
                    .expect("empty requirement is always feasible");
                let ctx = format!("{protein} λ={lambda} {mode:?}");
                assert_eq!(free.sequence, constrained.sequence, "sequence [{ctx}]");
                assert_eq!(free.structure, constrained.structure, "structure [{ctx}]");
                assert_eq!(free.mfe_kcal_mol, constrained.mfe_kcal_mol, "MFE [{ctx}]");
                assert_eq!(free.cai_log, constrained.cai_log, "CAI [{ctx}]");
            }
        }
    }
}

/// (2)+(3) Cases small enough for full enumeration, with the required position
/// placed where the unconstrained optimum leaves it unpaired — so the constraint
/// has to move the fold rather than being satisfied for free.
const SINGLE_POSITION_CASES: &[(&str, &[u32])] = &[
    ("KGGK", &[1]),
    ("KGGK", &[6]),
    ("MGGK", &[4, 11]),
    ("KWGGK", &[2, 8]),
    ("MKGGW", &[3]),
];

#[test]
fn single_required_positions_match_the_constrained_oracle() {
    check_cases(SINGLE_POSITION_CASES, "single required positions");
}

/// Brute-force cases at lengths that can form a multi-loop at all (27 nt), which
/// the 12–15 nt cases above cannot.
///
/// The proteins are built from the two single-codon residues, so the encoding
/// axis stays at one member while the structure space carries the cost —
/// 2·10⁵–3·10⁵ nested structures per case, walked without materializing.
///
/// This widens the oracle's reach; it does **not** by itself exercise the
/// `multi_open` path's constrained behaviour, because at these lengths a
/// multi-loop is never the optimum (the closing penalty outweighs what two short
/// AU/GU-rich stems can pay for). `an_already_satisfied_requirement_changes_nothing`
/// is the test that reaches it.
const MULTILOOP_LENGTH_CASES: &[(&str, &[u32])] = &[
    ("MWMWMWMWM", &[13]),
    ("MWMWMWMWM", &[12, 13, 14]),
    ("WWWMMMWWW", &[13]),
    ("MMWWMMWWM", &[12, 14]),
];

#[test]
fn multiloop_length_cases_match_the_constrained_oracle() {
    check_cases(MULTILOOP_LENGTH_CASES, "multi-loop-capable lengths");
}

/// The `multi_open` site's own check, and the only evidence that its eviction is
/// not over-aggressive.
///
/// A shared multi-loop opening carries the 3′ unpaired stretch from the wrapped
/// core's right boundary to whichever frontier closes it, and is pooled *without*
/// that boundary in its key. A requirement is honoured there by evicting the pool
/// as the scan crosses a required position rather than by widening the key — an
/// argument, not an inspection, and the one place in this family where evicting
/// too much would silently lose optima instead of emitting a visible violation.
///
/// **Both internal-loop closures share that path**, so the reduced-vs-un-reduced
/// differential cannot see a fault in it: they would be wrong together. And the
/// brute force cannot reach it, because a construct long enough for a multi-loop
/// to be *optimal* is far past enumerable size.
///
/// So it is checked by a floor instead. Take the unconstrained optimum, require
/// every position it already pairs, and re-solve: that structure is still
/// feasible by construction, so the constrained optimum must equal it. Every
/// required position is paired in the target, hence none lies in any unpaired
/// stretch of it, hence the target's own multi-loop opening spans no required
/// position and must survive the eviction. An eviction that took too much would
/// show up as a worse constrained optimum. At least one case's optimum must
/// carry a multi-loop, or the check is vacuous.
#[test]
#[ignore = "corpus-scale floor; run with --ignored --release"]
fn an_already_satisfied_requirement_changes_nothing() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let mut multiloop_seen = false;
    let mut checked = 0u32;
    for protein in [
        "MASMTGGQQMGRGSHHHHHH",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS",
    ] {
        for &lambda in &LAMBDAS {
            let free = design_with_config(&params, protein, &usage, lambda, ScanConfig::default())
                .expect("unconstrained design");
            let paired: Vec<u32> = free
                .structure
                .bytes()
                .enumerate()
                .filter(|&(_, b)| b != b'.')
                .map(|(i, _)| i as u32)
                .collect();
            if paired.is_empty() {
                continue;
            }
            multiloop_seen |= has_multiloop(&pairs_of(&free.structure));
            checked += 1;
            let ctx = format!("{protein} λ={lambda} required={} positions", paired.len());
            let held = design_must_paired(
                &params,
                protein,
                &usage,
                lambda,
                &paired,
                ScanConfig::default(),
            )
            .unwrap()
            .expect("the unconstrained optimum satisfies its own pairing");
            let dk = |d: &derna_lib::Design| {
                (
                    (d.mfe_kcal_mol * 100.0).round() as i32,
                    (d.cai_log * 100.0).round() as i32,
                )
            };
            assert_eq!(
                dk(&held),
                dk(&free),
                "a requirement the optimum already satisfies changed it [{ctx}]",
            );
            for &r in &paired {
                assert_ne!(
                    held.structure.as_bytes()[r as usize],
                    b'.',
                    "required position {r} left unpaired [{ctx}]",
                );
            }
        }
    }
    assert!(checked > 0, "no case had a paired optimum");
    assert!(
        multiloop_seen,
        "no optimum carried a multi-loop — the multi_open path was never exercised",
    );
}

/// A required *adjacent run* — the shape a required helix lowers to — which
/// forces several loop-interior ranges to shrink at once rather than one.
/// The last case is infeasible at this length — three mutually distant partners
/// do not fit on 12 nucleotides — which exercises the feasibility agreement in
/// the direction the feasible cases cannot.
const RUN_CASES: &[(&str, &[u32])] = &[
    ("KWGGK", &[1, 2, 3]),
    ("KWGGK", &[2, 3, 4]),
    ("KGGK", &[0, 1, 2]),
];

#[test]
fn required_runs_match_the_constrained_oracle() {
    check_cases(RUN_CASES, "required runs");
}

/// (4) A position with no legal partner is infeasible, and is reported as such
/// rather than as a garbage optimum. On a 6-nt sequence the no-sharp-turn rule
/// admits a pair `(i, j)` only for `j ≥ i + 4`, so a position can pair at all
/// only if it is at index `≤ 1` (as the 5′ base) or `≥ 4` (as the 3′ base) —
/// positions 2 and 3 have no partner whatever the sequence spells.
#[test]
fn an_unpairable_requirement_is_reported_infeasible() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for pos in [2u32, 3] {
        assert!(
            design_must_paired(&params, "MK", &usage, 0.5, &[pos], ScanConfig::default())
                .unwrap()
                .is_none(),
            "position {pos} of a 6-nt sequence cannot pair",
        );
    }
}

/// (5) The reduction-composition screen, proven rather than argued: the
/// constrained fold under the Lyngsø–Zuker / one-sided / `multi_open` reductions
/// equals the un-reduced constrained fold byte-exactly (MFE + CAI).
///
/// The proteins are long enough (n ≈ 45–75) that internal loops with both flanks
/// ≥ 5 form and every reduced path runs, and the required positions sit **in the
/// middle third** of the sequence — where they land inside deep loop interiors.
/// A requirement near an end never reaches the flank ranges the reductions
/// enumerate over and would pass this check vacuously.
#[test]
fn reductions_compose_with_the_requirement_lz_vs_naive() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let cfg = |mode| ScanConfig {
        mode,
        ..ScanConfig::default()
    };
    let cases: &[&str] = &[
        "MKTAYIAKQRLF",
        "DEYWCNQHKRFFGG",
        "FFLLSSPPHHRRKKMW",
        "GGGCCCAAAGGGMKTAYIA",
        "MASMTGGQQMGRGSHHHHHH",
    ];
    let mut feasible = 0u32;
    for protein in cases {
        let n = 3 * protein.len() as u32;
        // Mid-fold placements at three densities, all inside the middle third.
        let mid = n / 3;
        let plans: [Vec<u32>; 3] = [
            vec![n / 2],
            vec![mid, mid + 7],
            (0..5).map(|i| mid + 3 * i).collect(),
        ];
        for required in &plans {
            for &lambda in &LAMBDAS {
                let ctx = format!("{protein} required={required:?} λ={lambda}");
                let lz = design_must_paired(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    required,
                    cfg(InternalLoopMode::Lz),
                )
                .unwrap();
                let naive = design_must_paired(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    required,
                    cfg(InternalLoopMode::Naive),
                )
                .unwrap();
                assert_eq!(
                    lz.is_some(),
                    naive.is_some(),
                    "feasibility diverged between closures [{ctx}]",
                );
                let (Some(lz), Some(naive)) = (lz, naive) else {
                    continue;
                };
                feasible += 1;
                let dk = |d: &derna_lib::Design| {
                    (
                        (d.mfe_kcal_mol * 100.0).round() as i32,
                        (d.cai_log * 100.0).round() as i32,
                    )
                };
                assert_eq!(
                    dk(&lz),
                    dk(&naive),
                    "reduced vs un-reduced constrained fold diverged [{ctx}]",
                );
                for &r in required {
                    assert_ne!(
                        lz.structure.as_bytes()[r as usize],
                        b'.',
                        "required position {r} left unpaired [{ctx}]",
                    );
                }
            }
        }
    }
    assert!(
        feasible > 0,
        "every mid-fold case was infeasible — the composition check never ran",
    );
}

/// A small deterministic xorshift64 PRNG, so a failing sweep case replays from
/// its seed — the same shape `lz_fuzz` uses.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

/// The pulled single-branch closure and the pushed one leave the same design,
/// under required positions as without them.
///
/// Unconstrained, every pair the push writes to already holds a hairpin, so the
/// pull only improves cells that exist. A required position inside an enclosed
/// pair blocks the *enclosing* pair's hairpin while allowing the pair itself,
/// so there the push's first candidate creates the cell — and the pull has to
/// create it too, at the moment the push would have, since a column's
/// iteration order follows the order of its inserts and every later tie is
/// broken in that order. This checks the two routes agree on the whole design,
/// sequence and structure included, over every fixed case and a randomized
/// sweep whose requirements land mid-fold.
#[test]
fn the_pulled_and_pushed_closures_agree_under_required_positions() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let mut created = 0u32;
    let mut cases: Vec<(String, Vec<u32>)> = Vec::new();
    for set in [SINGLE_POSITION_CASES, MULTILOOP_LENGTH_CASES, RUN_CASES] {
        cases.extend(set.iter().map(|&(p, r)| (p.to_string(), r.to_vec())));
    }
    for case in 0..12u64 {
        let mut rng = Rng::new(0x5055_4C4Cu64.wrapping_add(case));
        let m = rng.range(14, 34);
        let protein: String = (0..m)
            .map(|_| ONE_LETTER[rng.range(0, ONE_LETTER.len() - 1)] as char)
            .collect();
        let n = 3 * m as u32;
        let (lo, hi) = (n / 3, 2 * n / 3);
        let mut required: Vec<u32> = (0..rng.range(1, 4))
            .map(|_| rng.range(lo as usize, hi as usize) as u32)
            .collect();
        required.sort_unstable();
        required.dedup();
        cases.push((protein, required));
    }
    for (protein, required) in &cases {
        for &lambda in &LAMBDAS {
            let ctx = format!("{protein} required={required:?} λ={lambda}");
            let design = |pull: bool| {
                let route = if pull {
                    design_must_paired
                } else {
                    design_must_paired_pushing
                };
                route(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    required,
                    ScanConfig::default(),
                )
                .unwrap()
            };
            let (pulled, pushed) = (design(true), design(false));
            assert_eq!(
                pulled.is_some(),
                pushed.is_some(),
                "feasibility diverged between the two routes [{ctx}]",
            );
            let (Some(pulled), Some(pushed)) = (pulled, pushed) else {
                continue;
            };
            if !required.is_empty() {
                created += 1;
            }
            let shape = |d: &derna_lib::Design| {
                (
                    (d.mfe_kcal_mol * 100.0).round() as i64,
                    (d.cai_log * 1e9).round() as i64,
                    d.sequence.clone(),
                    d.structure.clone(),
                )
            };
            assert_eq!(
                shape(&pulled),
                shape(&pushed),
                "the pulled and pushed closures spelled different designs [{ctx}]",
            );
        }
    }
    assert!(
        created > 0,
        "every constrained case was infeasible — the comparison never reached a blocked hairpin",
    );
}

/// (5, extended) The randomized differential sweep behind the fixed cases: random
/// proteins, random mid-fold requirement sets, both closures, byte-exact.
///
/// Longer than the standard-pass cases (`n = 60 … 180`) and unseeded by any
/// hand-chosen placement, so it reaches loop topologies the fixed set does not.
/// Run with `cargo test -p derna-lib --release --test must_pair_screen --
/// --ignored`.
#[test]
#[ignore = "rigorous differential sweep; run with --ignored --release"]
fn reductions_compose_with_the_requirement_fuzz() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let cfg = |mode| ScanConfig {
        mode,
        ..ScanConfig::default()
    };
    let mut feasible = 0u32;
    for case in 0..40u64 {
        let mut rng = Rng::new(0x4D50_0001u64.wrapping_add(case));
        let m = rng.range(20, 60);
        let protein: String = (0..m)
            .map(|_| ONE_LETTER[rng.range(0, ONE_LETTER.len() - 1)] as char)
            .collect();
        let n = 3 * m as u32;
        // Requirements land strictly inside the middle third, where deep loop
        // interiors form.
        let (lo, hi) = (n / 3, 2 * n / 3);
        let count = rng.range(1, 6);
        let mut required: Vec<u32> = (0..count)
            .map(|_| rng.range(lo as usize, hi as usize) as u32)
            .collect();
        required.sort_unstable();
        required.dedup();
        for &lambda in &LAMBDAS {
            let ctx = format!("case={case} {protein} required={required:?} λ={lambda}");
            let lz = design_must_paired(
                &params,
                &protein,
                &usage,
                lambda,
                &required,
                cfg(InternalLoopMode::Lz),
            )
            .unwrap();
            let naive = design_must_paired(
                &params,
                &protein,
                &usage,
                lambda,
                &required,
                cfg(InternalLoopMode::Naive),
            )
            .unwrap();
            assert_eq!(
                lz.is_some(),
                naive.is_some(),
                "feasibility diverged between closures [{ctx}]",
            );
            let (Some(lz), Some(naive)) = (lz, naive) else {
                continue;
            };
            feasible += 1;
            let dk = |d: &derna_lib::Design| {
                (
                    (d.mfe_kcal_mol * 100.0).round() as i32,
                    (d.cai_log * 100.0).round() as i32,
                )
            };
            assert_eq!(
                dk(&lz),
                dk(&naive),
                "reduced vs un-reduced constrained fold diverged [{ctx}]",
            );
            for &r in &required {
                assert_ne!(
                    lz.structure.as_bytes()[r as usize],
                    b'.',
                    "required position {r} left unpaired [{ctx}]",
                );
            }
        }
    }
    assert!(feasible > 0, "every sweep case was infeasible");
}
