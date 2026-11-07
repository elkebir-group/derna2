//! Brute-force gate for the positional constraint surface.
//!
//! For each case the design is compared against an exhaustive enumeration:
//! every synonymous encoding admitted by the pins × every nested canonical
//! structure honouring the constraint characters, scored by `rna_mfe::evaluate`
//! and reduced to the fixed-λ BRD optimum. Byte-exact agreement in **both**
//! directions is the gate — the design may not beat the enumeration (which would
//! mean it violated a constraint, or the enumeration missed a structure) nor
//! fall short of it.
//!
//! # Why the oracle re-states the alphabet
//!
//! [`Want`] below writes out the constraint alphabet's semantics again rather than
//! consulting the crate's own `Mark`. A differential that shares the definition
//! with what it checks cannot see a misreading of it — and the `<` / `>`
//! directions are exactly the kind of thing that is misread, since `<` names a
//! partner *downstream*.
//!
//! # Why every case carries a control
//!
//! A constrained arm that satisfies a constraint the *unconstrained* arm already
//! satisfied has measured the instance, not the constraint. Every case here
//! therefore asserts that the unconstrained optimum **violates** what the
//! constrained one honours, so a case that stopped being a constraint fails
//! rather than passes.

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_composite, design_with_config, Beam, CodonUsage, CompositeSpec, Design, Layout,
    MotifWindow, Params, PositionalConstraints, RegionConstraints, ScanConfig,
    StructureConstraints, UtrProfile, UtrSpec,
};
use rna_mfe::evaluate;

/// BRD tolerance in dekacalories — absorbs the rounded-integer vs continuous
/// objective gap, exactly as `tests/brd_parity.rs`.
const BRD_TOL: f64 = 0.5;

/// The λ values every case is swept over (low / mid / high).
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// What one position's constraint character demands of a structure.
///
/// Re-stated from the constraint alphabet's definitions rather than imported, so
/// this oracle disagrees with an implementation that misread them.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Want {
    /// `.`
    Any,
    /// `x`
    Unpaired,
    /// `|`
    Paired,
    /// `<` — the partner lies at a **larger** index.
    PartnerDownstream,
    /// `>` — the partner lies at a **smaller** index.
    PartnerUpstream,
    /// `(` / `)` — this exact partner.
    Partner(usize),
}

/// One oracle case: a protein, the pins over it, the constraint string, and any
/// pair forbidden outright. Coordinates are absolute 0-based over a CDS-only
/// molecule, so the constraint string is exactly `3 × |protein|` characters.
struct Case {
    protein: &'static str,
    /// `(absolute position, IUPAC letters)` — anchored subsequences.
    pins: &'static [(u32, &'static str)],
    /// The whole-molecule `-C` string.
    marks: &'static str,
    /// Pairs forbidden outright.
    forbid: &'static [(u32, u32)],
}

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
    lambda * mfe_dk - (1.0 - lambda) * cai_log * 100.0
}

/// Whether ASCII nucleotides `a` and `b` form a canonical Watson–Crick / wobble
/// pair (the six pairs `rna-mfe`'s `pair_from_nucleotides` admits).
fn canonical(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'C', b'G') | (b'G', b'C') | (b'G', b'U') | (b'U', b'G')
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
/// hairpin loop 3 — the exact structure space the CKY grammar folds over.
/// Streaming rather than materializing; the count is exponential in length.
fn walk_structures(
    seq: &[u8],
    open: &mut Vec<(usize, usize)>,
    pairs: &mut Vec<(usize, usize)>,
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
    open.push((lo + 1, hi));
    walk_structures(seq, open, pairs, visit);
    open.pop();
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
    let mut pairs = Vec::new();
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

/// The `-C` string read into the oracle's own per-position demands.
fn wants_of(marks: &str) -> Vec<Want> {
    let mut wants = vec![Want::Any; marks.len()];
    let mut stack = Vec::new();
    for (i, c) in marks.bytes().enumerate() {
        wants[i] = match c {
            b'.' => Want::Any,
            b'x' => Want::Unpaired,
            b'|' => Want::Paired,
            b'<' => Want::PartnerDownstream,
            b'>' => Want::PartnerUpstream,
            b'(' => {
                stack.push(i);
                continue;
            }
            b')' => {
                let open = stack.pop().expect("balanced constraint string");
                wants[open] = Want::Partner(i);
                Want::Partner(open)
            }
            other => panic!("unexpected constraint character {:?}", other as char),
        };
    }
    assert!(stack.is_empty(), "balanced constraint string");
    wants
}

/// Whether a structure honours every demand — the oracle's independent reading.
fn honours(wants: &[Want], forbid: &[(u32, u32)], pairs: &[(usize, usize)]) -> bool {
    let partner = |k: usize| {
        pairs.iter().find_map(|&(i, j)| match k {
            _ if i == k => Some(j),
            _ if j == k => Some(i),
            _ => None,
        })
    };
    let honoured = wants.iter().enumerate().all(|(k, want)| match *want {
        Want::Any => true,
        Want::Unpaired => partner(k).is_none(),
        Want::Paired => partner(k).is_some(),
        Want::PartnerDownstream => partner(k).is_some_and(|p| p > k),
        Want::PartnerUpstream => partner(k).is_some_and(|p| p < k),
        Want::Partner(p) => partner(k) == Some(p),
    });
    honoured
        && !forbid
            .iter()
            .any(|&(i, j)| pairs.contains(&(i as usize, j as usize)))
}

/// Whether every pinned position of `seq` spells a letter its pin admits.
fn pins_admit(seq: &[u8], pins: &[(u32, &str)]) -> bool {
    pins.iter().all(|&(at, letters)| {
        letters
            .bytes()
            .enumerate()
            .all(|(off, code)| iupac_set(code).contains(&seq[at as usize + off]))
    })
}

/// The nucleotides an IUPAC code denotes, spelled out here rather than taken
/// from the crate for the same reason [`Want`] is.
fn iupac_set(code: u8) -> &'static [u8] {
    match code {
        b'A' => b"A",
        b'C' => b"C",
        b'G' => b"G",
        b'U' => b"U",
        b'R' => b"AG",
        b'Y' => b"CU",
        b'S' => b"CG",
        b'W' => b"AU",
        b'K' => b"GU",
        b'M' => b"AC",
        b'B' => b"CGU",
        b'D' => b"AGU",
        b'H' => b"ACU",
        b'V' => b"ACG",
        b'N' => b"ACGU",
        other => panic!("unexpected IUPAC code {:?}", other as char),
    }
}

/// The brute-force optimum: the lowest-BRD `(sequence, structure)` over every
/// admitted encoding and honouring structure, or `None` when the set is empty.
fn brute_force(
    params: &Params,
    usage: &CodonUsage,
    case: &Case,
    lambda: f64,
) -> Option<(f64, f64, String)> {
    let aa = protein_indices(case.protein);
    let wants = wants_of(case.marks);
    let mut best: Option<(f64, f64, String)> = None;
    for encoding in encodings(&aa) {
        let seq = cds_bytes(&aa, &encoding);
        if !pins_admit(&seq, case.pins) {
            continue;
        }
        let cai = cai_log(usage, &aa, &encoding);
        let mut best_dk: Option<i32> = None;
        for_each_structure(&seq, |pairs| {
            if !honours(&wants, case.forbid, pairs) {
                return;
            }
            let e = evaluate(params, &seq, &dot_bracket(seq.len(), pairs))
                .expect("structure scores")
                .dekacal();
            best_dk = Some(best_dk.map_or(e, |b| b.min(e)));
        });
        let Some(dk) = best_dk else { continue };
        let mfe = dk as f64 / 100.0;
        let score = brd(lambda, mfe, cai);
        let seq_string = String::from_utf8(seq).expect("ascii");
        if best.as_ref().is_none_or(|b| score < brd(lambda, b.0, b.1)) {
            best = Some((mfe, cai, seq_string));
        }
    }
    best
}

/// Design under `case`'s constraints through the composite entry point — the
/// only entry the surface is reachable from.
fn design_under(
    params: &Params,
    usage: &CodonUsage,
    case: &Case,
    lambda: f64,
) -> Result<Design, derna_lib::BeamError> {
    let layout = Layout::new(0, case.protein.len() as u32, 0);
    let mut pc = PositionalConstraints::none(layout.len());
    pc.structure = StructureConstraints::parse(&layout, case.marks).expect("valid constraints");
    for &(i, j) in case.forbid {
        pc.structure
            .forbid_pair(&layout, i, j)
            .expect("valid forbidden pair");
    }
    for &(at, letters) in case.pins {
        pc.pins.pin(&layout, at, letters).expect("valid pin");
    }
    let (no_utr5, no_utr3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    design_composite(
        params,
        case.protein,
        usage,
        lambda,
        &CompositeSpec::<String> {
            utr5: &no_utr5,
            utr3: &no_utr3,
            constraints: RegionConstraints::default(),
            profile: UtrProfile::inert(),
            motifs: &[],
            motif_window: MotifWindow::all(),
            flank_complement: None,
            complement_classes: None,
            positional: &pc,
        },
        ScanConfig {
            beam: Beam::exact(),
            ..ScanConfig::default()
        },
    )
}

/// Check one case at every λ: byte-exact against brute force, the post-condition
/// read off the emitted structure, and the control that the constraint bit.
fn check(case: &Case) {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    assert_eq!(
        case.marks.len(),
        case.protein.len() * 3,
        "the constraint string spans the whole molecule"
    );
    let wants = wants_of(case.marks);
    assert_control_violates(&params, &usage, case, &wants);

    for lambda in LAMBDAS {
        let want = brute_force(&params, &usage, case, lambda);
        let got = design_under(&params, &usage, case, lambda);
        match (want, got) {
            (None, Err(_)) => {}
            (Some((mfe, cai, _)), Ok(design)) => {
                let (want_brd, got_brd) = (
                    brd(lambda, mfe, cai),
                    brd(lambda, design.mfe_kcal_mol, design.cai_log),
                );
                assert!(
                    (want_brd - got_brd).abs() <= BRD_TOL,
                    "{} λ={lambda}: design BRD {got_brd} vs brute force {want_brd} \
                     (design {} {} mfe={} cai={})",
                    case.protein,
                    design.sequence,
                    design.structure,
                    design.mfe_kcal_mol,
                    design.cai_log,
                );
                // The post-condition, read off what was actually emitted rather
                // than trusted from the solver's own bookkeeping.
                assert!(
                    honours(&wants, case.forbid, &pairs_of(&design.structure)),
                    "{} λ={lambda}: emitted structure {} violates {}",
                    case.protein,
                    design.structure,
                    case.marks
                );
                assert!(
                    pins_admit(design.sequence.as_bytes(), case.pins),
                    "{} λ={lambda}: emitted sequence {} violates its pins",
                    case.protein,
                    design.sequence
                );
            }
            (want, got) => panic!(
                "{} λ={lambda}: feasibility disagreement — brute force {:?}, design {:?}",
                case.protein,
                want.is_some(),
                got.is_ok()
            ),
        }
    }
}

/// The control: at some λ the **unconstrained** optimum must violate the case's
/// constraints. Without it a green case is consistent with the constraint never
/// having removed anything.
fn assert_control_violates(params: &Params, usage: &CodonUsage, case: &Case, wants: &[Want]) {
    let violated = LAMBDAS.iter().any(|&lambda| {
        let free = design_with_config(
            params,
            case.protein,
            usage,
            lambda,
            ScanConfig {
                beam: Beam::exact(),
                ..ScanConfig::default()
            },
        )
        .expect("the unconstrained design always exists");
        !honours(wants, case.forbid, &pairs_of(&free.structure))
            || !pins_admit(free.sequence.as_bytes(), case.pins)
    });
    assert!(
        violated,
        "{}: the unconstrained optimum already satisfies {:?} / {} at every λ, \
         so this case measures the instance rather than the constraint",
        case.protein, case.pins, case.marks
    );
}

/// A pin removes DFA arcs and nothing else, so the optimum is the unconstrained
/// one restricted to the admitted encodings — including its reported CAI, which
/// must be the whole CDS's rather than the zero a pinned residue's edges
/// carry.
#[test]
fn pins_alone_match_brute_force() {
    for case in [
        // A degenerate pin leaving the residue two codons — the case whose
        // restricted unit has a non-zero best available weight, so its whole
        // codon may not be credited as free inside an enclosing span.
        Case {
            protein: "MLWKC",
            pins: &[(3, "UUR")],
            marks: "...............",
            forbid: &[],
        },
        // A pin narrowing a residue to one codon: its edges carry no adaptation
        // at all, so the reported CAI is right only if the constant is added
        // back.
        Case {
            protein: "MLWKC",
            pins: &[(12, "UGU")],
            marks: "...............",
            forbid: &[],
        },
        // Two pins on different residues, one of them degenerate.
        Case {
            protein: "MLWKC",
            pins: &[(3, "Y"), (14, "C")],
            marks: "...............",
            forbid: &[],
        },
    ] {
        check(&case);
    }
}

/// `x` forbids every pair with an endpoint at the position, whatever the
/// partner — the reading a span-based implementation would get wrong.
#[test]
fn unpaired_marks_match_brute_force() {
    for case in [
        // The unconstrained optimum closes a five-pair stem over this protein,
        // so an `x` inside it removes something.
        Case {
            protein: "MLWKC",
            pins: &[],
            marks: "..xxx..........",
            forbid: &[],
        },
        Case {
            protein: "MLWKC",
            pins: &[],
            marks: "xx...........xx",
            forbid: &[],
        },
    ] {
        check(&case);
    }
}

/// A named pair is required *and* excludes every other partner for both of its
/// endpoints.
#[test]
fn required_pairs_match_brute_force() {
    for case in [
        Case {
            protein: "MKWLC",
            pins: &[],
            marks: "..(.........)..",
            forbid: &[],
        },
        Case {
            protein: "MLWKC",
            pins: &[],
            marks: ".((.......))...",
            forbid: &[],
        },
    ] {
        check(&case);
    }
}

/// `|` requires pairing and forbids nothing, and `<` / `>` add the direction.
/// The directions are the load-bearing case: the oracle reads them from the
/// alphabet's definitions, so an implementation that swapped them fails here.
#[test]
fn bar_and_angle_marks_match_brute_force() {
    for case in [
        Case {
            protein: "MKWLC",
            pins: &[],
            marks: "..|............",
            forbid: &[],
        },
        Case {
            protein: "MKWLC",
            pins: &[],
            marks: "..<............",
            forbid: &[],
        },
        Case {
            protein: "MKWLC",
            pins: &[],
            marks: "............>..",
            forbid: &[],
        },
    ] {
        check(&case);
    }
}

/// The negative direction, keyed by position: exactly one pair is removed.
#[test]
fn forbidden_pairs_match_brute_force() {
    check(&Case {
        protein: "MLWKC",
        pins: &[],
        marks: "...............",
        forbid: &[(2, 12), (0, 14)],
    });
}

/// The two axes on one molecule. They reach the lattice in different places —
/// arc removal and a positional verdict — so composing them is the case that
/// would catch either one being applied to the other's coordinates.
#[test]
fn pins_and_structure_together_match_brute_force() {
    for case in [
        Case {
            protein: "MLWKC",
            pins: &[(3, "UUR")],
            marks: "..xxx..........",
            forbid: &[],
        },
        Case {
            protein: "MLWKC",
            pins: &[(12, "UGU")],
            marks: "..(.........)..",
            forbid: &[],
        },
    ] {
        check(&case);
    }
}
