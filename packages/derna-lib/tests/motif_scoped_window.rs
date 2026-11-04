//! Position-scoped motif enforcement, and the region constraint on the constrained
//! fold.
//!
//! Global motif enforcement applies a forbidden set to every position of the
//! molecule. That is the wrong constraint whenever the set was **derived from**
//! part of the molecule — a motif set built from a given flank rejects the flank
//! that generated it, and the design is infeasible rather than merely
//! expensive. A window says where the set applies.
//!
//! Six things are checked here, and they are separate on purpose:
//!
//! 1. **The predicate.** A window has two coherent readings — occurrences
//!    *entirely inside* it, or occurrences *ending* inside it — and they differ
//!    on real inputs. The shipped reading is the first. A test written against
//!    an unstated predicate tests whatever the implementation happens to do, so
//!    the reading is asserted from both sides: an occurrence entirely inside is
//!    rejected, and one straddling the left edge is **spellable**.
//! 2. **The axis collapses outside the window** — one reachable AC state per
//!    node below the window's start and at or above its end. Checked on **both
//!    sidednesses**, because a window at the 3′ end has nothing past it: the
//!    coding sequence reads one state there merely by *preceding* the window,
//!    and the mirror (given 3′ flank, designed 5′) is the arrangement where the
//!    whole CDS lies past the window's end and the collapse is what produces
//!    the reading.
//! 3. **Feasibility**, which the state census does not imply — a construction
//!    could collapse the axis outside the window and still admit no in-window
//!    spelling. A set that kills the design globally must leave it feasible when
//!    scoped off the sequence it came from.
//! 4. **Byte-identity floors**: a whole-molecule window is global enforcement,
//!    an empty motif set is the run without one, and an inert table leaves the
//!    constrained fold where it was.
//! 5. **Exactness under both new axes at once** — flank spellings × CDS
//!    encodings × nested structures, filtered by the window predicate and the
//!    class table, BRD-argmin, byte-exact. Hard *and* soft enforcement, since a
//!    soft table that priced differently on the constrained path would mean two
//!    different things depending on whether `--motif` was passed.
//! 6. **The reductions**, differentially — see the last check in this file for
//!    what that differential does and does not settle.
//!
//! The window predicate, the IUPAC expansion, the structure enumerator, the CAI
//! formula and the region table are re-implemented here rather than shared with
//! the crate, so a bug in the substrate cannot hide inside its own oracle.

use std::collections::HashMap;

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::{Ac, ScopedAc};
use derna_lib::dfa::{self, Node, UtrChain};
use derna_lib::product::ProductDfa;
use derna_lib::PositionalConstraints;
use derna_lib::{
    design_composite, design_with_motifs, design_with_utr_profile, CodonUsage, CompositeSpec,
    Design, Enforcement, InternalLoopMode, MotifWindow, PairClass, Params, RegionConstraints,
    ScanConfig, UtrProfile, UtrSpec,
};
use rna_mfe::{evaluate, Nucleotide};

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

fn params_usage() -> (Params, CodonUsage) {
    (
        Params::load_default().expect("Turner 2004 parameters"),
        CodonUsage::load_default().expect("codon usage"),
    )
}

fn empty() -> UtrSpec {
    UtrSpec::Fixed(String::new())
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

fn nucleotides(s: &str) -> Vec<Nucleotide> {
    s.bytes()
        .map(|b| Nucleotide::from_byte(b).expect("A/C/G/U"))
        .collect()
}

// ===========================================================================
// 1. The predicate
// ===========================================================================

/// Whether `seq` contains an occurrence of any motif lying **entirely inside**
/// `[lo, hi)` — the oracle's own statement of the predicate, written from the
/// definition rather than from the automaton.
fn has_occurrence_inside(seq: &[u8], motifs: &[&str], lo: usize, hi: usize) -> bool {
    motifs.iter().any(|m| {
        let m = m.as_bytes();
        (lo..=hi.saturating_sub(m.len())).any(|s| s + m.len() <= hi && &seq[s..s + m.len()] == m)
    })
}

/// The two readings of a window differ exactly on the occurrences that straddle
/// its left edge, and the shipped reading permits them.
///
/// This is the check the window's whole semantics rests on: it fails in one
/// direction if the automaton is primed from before `lo` (the *ending-inside*
/// reading) and in the other if the window's last position is not enforced.
#[test]
fn window_forbids_occurrences_entirely_inside_and_permits_straddlers() {
    let ac = Ac::build(&["GGG"]).expect("valid motif");
    // Positions:      0123456789
    let window = MotifWindow::new(4, 8);
    let scoped = ScopedAc::new(&ac, window);

    // Entirely inside `[4, 8)`, at 4 and at 5 — both rejected.
    for seq in ["AAAAGGGAA", "AAAAAGGGA"] {
        assert!(
            !scoped.is_clean(&nucleotides(seq)),
            "an occurrence entirely inside the window was permitted: {seq}",
        );
    }
    // Straddling the left edge (starts at 2, ends at 5) — permitted, because the
    // automaton is never fed a position below `lo`.
    assert!(
        scoped.is_clean(&nucleotides("AAGGGAAAA")),
        "an occurrence straddling the window's left edge was rejected",
    );
    // Straddling the right edge (starts at 7, ends at 10) — permitted.
    assert!(
        scoped.is_clean(&nucleotides("AAAAAAAGGG")),
        "an occurrence straddling the window's right edge was rejected",
    );
    // Ending exactly at `hi` (positions 5..8) is entirely inside, so the last
    // in-window position is enforced and the collapse happens after the test.
    assert!(
        !scoped.is_clean(&nucleotides("AAAAAGGGAA")),
        "an occurrence ending exactly at the window's end was permitted",
    );
    // Wholly outside on the 3′ side.
    assert!(
        scoped.is_clean(&nucleotides("AAAAAAAAGGG")),
        "an occurrence wholly past the window was rejected",
    );
    // And the oracle agrees on every one of those, from the definition.
    for seq in [
        "AAAAGGGAA",
        "AAAAAGGGA",
        "AAGGGAAAA",
        "AAAAAAAGGG",
        "AAAAAGGGAA",
        "AAAAAAAAGGG",
    ] {
        assert_eq!(
            scoped.is_clean(&nucleotides(seq)),
            !has_occurrence_inside(seq.as_bytes(), &["GGG"], 4, 8),
            "automaton and predicate disagree on {seq}",
        );
    }
}

/// A whole-molecule window is the unscoped automaton: every occurrence is
/// entirely inside it.
#[test]
fn a_whole_molecule_window_is_global_enforcement() {
    let ac = Ac::build(&["GGG"]).expect("valid motif");
    let global = ScopedAc::new(&ac, MotifWindow::all());
    for seq in ["AAAAGGGAA", "GGGAAAAAA", "AAAAAAGGG", "ACGUACGUA"] {
        let seq = nucleotides(seq);
        let mut q = 0u32;
        let mut clean = true;
        for &x in &seq {
            q = ac.step(q, x);
            if ac.is_accept(q) {
                clean = false;
                break;
            }
        }
        assert_eq!(global.is_clean(&seq), clean);
    }
}

// ===========================================================================
// 2. The collapse, on both sidednesses
// ===========================================================================

/// The reachable AC states per base node of one region, as
/// `(nodes, total states, max at one node, nodes carrying more than one)`.
#[derive(Default, Debug, PartialEq, Eq)]
struct RegionStat {
    nodes: usize,
    states: usize,
    max: usize,
    split: usize,
}

impl RegionStat {
    fn mean(&self) -> f64 {
        if self.nodes == 0 {
            0.0
        } else {
            self.states as f64 / self.nodes as f64
        }
    }
}

/// The reachable-state census of a product, split into `(5′ UTR, CDS, 3′ UTR)`.
fn census(graph: &dfa::Dfa, product: &ProductDfa) -> (RegionStat, RegionStat, RegionStat) {
    let (cs, ce) = (graph.cds_start(), graph.cds_end());
    let mut out = (
        RegionStat::default(),
        RegionStat::default(),
        RegionStat::default(),
    );
    for pos in 0..=graph.seq_len {
        let stat = if pos < cs {
            &mut out.0
        } else if pos < ce {
            &mut out.1
        } else {
            &mut out.2
        };
        for lane in 0..2u8 {
            let node = Node::new(pos, lane);
            if !graph.nodes_at(pos).contains(&node) {
                continue;
            }
            let n = product.reachable_states(node).len();
            stat.nodes += 1;
            stat.states += n;
            stat.max = stat.max.max(n);
            stat.split += usize::from(n > 1);
        }
    }
    out
}

/// A hand-written motif set. Deriving one from a flank's complement is a
/// separate concern, and deriving it here would make a failure ambiguous
/// between the window and the derivation.
const MOTIFS: [&str; 3] = ["GGC", "CCA", "GAU"];

/// **The load-bearing check, and the one easiest to pass without meaning it.**
///
/// The axis must carry exactly one reachable state at every node outside the
/// window — and that has to be shown on the sidedness where the *right*
/// boundary does the work. With the window on the 3′ flank nothing lies past
/// it, so a `1.000` reading over the CDS says only that the CDS precedes the
/// window. The mirror puts the whole coding sequence past the window's end,
/// where one state per node is produced by the collapse or not at all.
#[test]
fn the_axis_collapses_outside_the_window_on_both_sidednesses() {
    let (_, usage) = params_usage();
    let aa = protein_indices("MKWDEYKF");
    let ac = Ac::build(&MOTIFS).expect("valid motifs");
    let flank: Vec<Nucleotide> = nucleotides("GGCAUGGCAUGGCAUGGCAU");

    // (label, graph, window) — the two arrangements, each with the motif set
    // enforced over the *designed* flank alone.
    let given5 =
        dfa::build_with_utr_chain(&aa, &usage, UtrChain::Fixed(&flank), UtrChain::Designed(12));
    let w5 = MotifWindow::new(given5.cds_end(), given5.seq_len);
    let given3 =
        dfa::build_with_utr_chain(&aa, &usage, UtrChain::Designed(12), UtrChain::Fixed(&flank));
    let w3 = MotifWindow::new(0, given3.cds_start());

    for (label, graph, window, outside_first) in [
        ("given 5′ flank, designed 3′", &given5, w5, true),
        ("given 3′ flank, designed 5′", &given3, w3, false),
    ] {
        let scoped = ProductDfa::build_scoped(graph, &ac, window);
        assert!(
            scoped.is_feasible(),
            "the scoped product is infeasible [{label}]",
        );
        let (u5, cds, u3) = census(graph, &scoped);
        // Every region outside the window reads exactly one state per node.
        let (before, after) = if outside_first {
            (&u5, &u3)
        } else {
            (&u3, &u5)
        };
        assert_eq!(
            (before.mean(), before.split),
            (1.0, 0),
            "the region before the window split [{label}]: {before:?}",
        );
        assert_eq!(
            (cds.mean(), cds.split),
            (1.0, 0),
            "the CDS split under scoping [{label}]: {cds:?}",
        );
        // The window itself must genuinely carry the axis, or the whole census
        // is a statement about a constraint that never engaged.
        assert!(
            after.max > 1,
            "the window carried no split states, so the census is vacuous \
             [{label}]: {after:?}",
        );
        // And the global product on the same graph must read differently —
        // otherwise the collapse above is a property of the instance rather
        // than of scoping. It differs in one of the two ways the baseline
        // census recorded: the CDS splits, or the path dies outright.
        let global = ProductDfa::build(graph, &ac);
        let (_, global_cds, _) = census(graph, &global);
        assert!(
            !global.is_feasible() || global_cds.split > 0,
            "the unscoped product was feasible and left the CDS unsplit, so this \
             instance cannot distinguish scoping from doing nothing \
             [{label}]: {global_cds:?}",
        );
    }
}

// ===========================================================================
// 3. Feasibility
// ===========================================================================

/// A motif set that kills the design globally leaves it feasible when the
/// window excludes the sequence the set came from.
///
/// This is what scoping exists for, and it is not implied by the state census:
/// a construction could collapse the axis outside the window and still admit no
/// in-window spelling at all.
#[test]
fn scoping_recovers_feasibility_a_global_set_destroys() {
    let (params, usage) = params_usage();
    let protein = "MKWDEY";
    // A flank spelling every 3-mer the set forbids, so the *given* sequence is
    // itself unspellable under global enforcement.
    let utr5 = UtrSpec::Fixed("GGCCCAGAUGGC".into());
    let utr3 = UtrSpec::Designed(9);
    let motifs: Vec<String> = MOTIFS.iter().map(|m| (*m).to_string()).collect();
    let cds_end = utr5.len() as u32 + 3 * protein.len() as u32;
    let seq_len = cds_end + 9;

    let run = |window: MotifWindow| {
        design_composite(
            &params,
            protein,
            &usage,
            0.5,
            &CompositeSpec {
                utr5: &utr5,
                utr3: &utr3,
                constraints: RegionConstraints::default(),
                profile: UtrProfile::inert(),
                motifs: &motifs,
                motif_window: window,
                flank_complement: None,
                complement_classes: None,
                positional: PositionalConstraints::empty(),
            },
            ScanConfig::default(),
        )
    };

    assert!(
        run(MotifWindow::all()).is_err(),
        "the global set was expected to be infeasible — without that, scoping \
         has nothing to recover and this case proves nothing",
    );
    let scoped = run(MotifWindow::new(cds_end, seq_len)).expect("scoped design is feasible");
    // The source flank is outside the window and still spells its motifs.
    assert!(
        has_occurrence_inside(scoped.sequence.as_bytes(), &MOTIFS, 0, utr5.len()),
        "the given flank lost its motifs, so the window did not exclude it",
    );
    // Inside the window, nothing does.
    assert!(
        !has_occurrence_inside(
            scoped.sequence.as_bytes(),
            &MOTIFS,
            cds_end as usize,
            seq_len as usize,
        ),
        "the designed flank spells a forbidden motif inside the window: {}",
        scoped.sequence,
    );
}

// ===========================================================================
// 4. Byte-identity floors
// ===========================================================================

fn composite<'a>(
    utr5: &'a UtrSpec,
    utr3: &'a UtrSpec,
    constraints: RegionConstraints,
    motifs: &'a [String],
    motif_window: MotifWindow,
) -> CompositeSpec<'a, String> {
    CompositeSpec {
        utr5,
        utr3,
        constraints,
        profile: UtrProfile::inert(),
        motifs,
        motif_window,
        flank_complement: None,
        complement_classes: None,
        positional: PositionalConstraints::empty(),
    }
}

/// A window covering the whole molecule reproduces global enforcement, and an
/// explicit `[0, seq_len)` window does too — the second is the interesting one,
/// because it is the window whose right edge sits on the terminal node.
#[test]
fn a_whole_molecule_window_is_byte_identical_to_no_window() {
    let (params, usage) = params_usage();
    let protein = "MKWDEYK";
    let motifs: Vec<String> = MOTIFS.iter().map(|m| (*m).to_string()).collect();
    let seq_len = 3 * protein.len() as u32;
    let (u5, u3) = (empty(), empty());
    for &lambda in &LAMBDAS {
        let want = design_with_motifs(
            &params,
            protein,
            &usage,
            lambda,
            &motifs,
            ScanConfig::default(),
        )
        .expect("designs");
        for window in [MotifWindow::all(), MotifWindow::new(0, seq_len)] {
            let got = design_composite(
                &params,
                protein,
                &usage,
                lambda,
                &composite(&u5, &u3, RegionConstraints::default(), &motifs, window),
                ScanConfig::default(),
            )
            .expect("designs");
            assert_eq!(
                got, want,
                "a whole-molecule window diverged from global enforcement [λ={lambda}]",
            );
        }
    }
}

/// An empty motif set with flanks and a table present is the run without one,
/// byte for byte — the floor that keeps every shipped UTR result where it was.
#[test]
fn an_empty_motif_set_is_byte_identical_under_flanks_and_a_table() {
    let (params, usage) = params_usage();
    let protein = "MKW";
    let utr5 = UtrSpec::Fixed("ACGUAC".into());
    let utr3 = UtrSpec::Designed(5);
    let none: Vec<String> = Vec::new();
    for (label, rc) in tables() {
        for &lambda in &LAMBDAS {
            let want = design_with_utr_profile(
                &params,
                protein,
                &usage,
                lambda,
                &utr5,
                &utr3,
                rc,
                UtrProfile::inert(),
                ScanConfig::default(),
            )
            .expect("designs");
            let got = design_composite(
                &params,
                protein,
                &usage,
                lambda,
                &composite(&utr5, &utr3, rc, &none, MotifWindow::new(2, 7)),
                ScanConfig::default(),
            )
            .expect("designs");
            assert_eq!(
                got, want,
                "an empty motif set perturbed a UTR design [{label} λ={lambda}]",
            );
        }
    }
}

/// An inert table leaves the constrained fold where it was: the constraint call is
/// new on this path, so every previously shipped `--motif` result has to be
/// shown unchanged rather than assumed to be.
#[test]
fn an_inert_table_leaves_the_constrained_fold_byte_identical() {
    let (params, usage) = params_usage();
    let motifs: Vec<String> = MOTIFS.iter().map(|m| (*m).to_string()).collect();
    let (u5, u3) = (empty(), empty());
    for protein in ["MKW", "MKWDEYK", "DEYWCNQHKR"] {
        for &lambda in &LAMBDAS {
            let want = design_with_motifs(
                &params,
                protein,
                &usage,
                lambda,
                &motifs,
                ScanConfig::default(),
            )
            .expect("designs");
            let got = design_composite(
                &params,
                protein,
                &usage,
                lambda,
                &composite(
                    &u5,
                    &u3,
                    RegionConstraints::default(),
                    &motifs,
                    MotifWindow::all(),
                ),
                ScanConfig::default(),
            )
            .expect("designs");
            assert_eq!(
                got, want,
                "the constraint call moved a motif-only design [{protein} λ={lambda}]"
            );
        }
    }
}

// ===========================================================================
// 5. Exactness under the window and the constraint
// ===========================================================================

/// The region tables the parity check sweeps: inert, a hard crossing class, and
/// a soft one. The soft entry is not decoration — a penalty credited differently
/// on the constrained path would make a soft table mean two things.
fn tables() -> Vec<(&'static str, RegionConstraints)> {
    vec![
        ("inert", RegionConstraints::default()),
        (
            "5-cds:forbid",
            RegionConstraints::default().with(PairClass::Utr5Cds, Enforcement::Forbid),
        ),
        (
            "cds-3:soft",
            RegionConstraints::default().with(
                PairClass::CdsUtr3,
                Enforcement::Penalize(derna_lib::Energy(150)),
            ),
        ),
    ]
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

/// Every spelling a flank spec admits.
fn flank_choices(spec: &UtrSpec) -> Vec<Vec<u8>> {
    match spec {
        UtrSpec::Fixed(s) => vec![s.as_bytes().to_vec()],
        UtrSpec::Designed(n) => mixed_radix(&vec![4; *n])
            .into_iter()
            .map(|d| d.into_iter().map(nuc_byte).collect())
            .collect(),
        UtrSpec::Mixed(_) => panic!("this oracle covers fixed and designed flanks"),
    }
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

/// Whether ASCII nucleotides `a` and `b` form a canonical Watson–Crick / wobble
/// pair.
fn canonical(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'C') | (b'C', b'G') | (b'G', b'U') | (b'U', b'G')
    )
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

/// The oracle's own region table, keyed on its own region codes.
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

/// The constrained MFE (dekacalories) of one spelled sequence under `spec`, or
/// `None` when the table forbids every structure (which cannot happen — the
/// all-unpaired structure is always allowed).
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

/// Fixed-λ BRD on rounded dekacalories, matching how the oracle and the design
/// both round at the scalarization boundary.
fn brd(lambda: f64, mfe_kcal: f64, adaptation_log: f64) -> f64 {
    lambda * (mfe_kcal * 100.0).round() - (1.0 - lambda) * (adaptation_log * 100.0).round()
}

/// **The exactness oracle.** With a window *and* a class table active, the
/// design must be the BRD-argmin over the joint space of flank spellings, CDS
/// encodings and table-allowed nested structures, restricted to the sequences
/// with no forbidden occurrence inside the window.
///
/// Both new axes are on at once deliberately: each is exercised elsewhere alone,
/// and the composition is the thing neither of those checks reaches.
#[test]
fn the_constrained_optimum_is_exact_under_a_window_and_a_table() {
    let (params, usage) = params_usage();
    let protein = "MK";
    let motifs: Vec<String> = ["GG", "CA"].iter().map(|m| (*m).to_string()).collect();
    let literal: Vec<&str> = vec!["GG", "CA"];
    // Given 5′ flank spelling both motifs, designed 3′ flank, window over the
    // designed flank alone — the arrangement instrument 2 is for.
    let utr5_literal = "GGCA";
    let utr5 = UtrSpec::Fixed(utr5_literal.into());
    let utr3 = UtrSpec::Designed(3);
    let aa = protein_indices(protein);
    let (cds_start, cds_end) = (utr5.len(), utr5.len() + 3 * protein.len());
    let seq_len = cds_end + utr3.len();
    let window = MotifWindow::new(cds_end as u32, seq_len as u32);
    // Every table's design at every λ, so the non-inert ones can be shown to
    // have moved something. A table that never bites would let all three cases
    // below pass as the inert one three times over — which is what a constraint call
    // that was never reached would also look like.
    let mut by_table: Vec<(&str, Vec<Design>)> = Vec::new();

    for (label, rc) in tables() {
        let spec = Spec::new(&rc, cds_start, cds_end);
        // Score the whole feasible set once; both coordinates are λ-independent.
        let mut scored: Vec<(f64, f64)> = Vec::new();
        let mut rejected = 0usize;
        for u3 in flank_choices(&utr3) {
            for enc in encodings(&aa) {
                let full = full_bytes(&aa, &enc, utr5_literal.as_bytes(), &u3);
                if has_occurrence_inside(&full, &literal, cds_end, seq_len) {
                    rejected += 1;
                    continue;
                }
                scored.push((
                    f64::from(constrained_mfe_dk(&params, &full, &spec)) / 100.0,
                    cds_cai_log(&usage, &aa, &enc),
                ));
            }
        }
        assert!(
            rejected > 0,
            "the window excluded nothing, so this case cannot tell an enforced \
             window from an absent one [{label}]",
        );
        let mut designs = Vec::new();
        for &lambda in &LAMBDAS {
            let got = design_composite(
                &params,
                protein,
                &usage,
                lambda,
                &composite(&utr5, &utr3, rc, &motifs, window),
                ScanConfig::default(),
            )
            .expect("designs");
            designs.push(got.clone());
            let want = scored
                .iter()
                .map(|&(mfe, cai)| brd(lambda, mfe, cai))
                .fold(f64::INFINITY, f64::min);
            let got_brd = brd(lambda, got.mfe_kcal_mol, got.cai_log);
            assert!(
                (got_brd - want).abs() <= BRD_TOL,
                "constrained optimum diverged from enumeration [{label} λ={lambda}]: \
                 design={got_brd} oracle={want} seq={}",
                got.sequence,
            );
            // And the design honors the window from the outside in.
            assert!(
                !has_occurrence_inside(got.sequence.as_bytes(), &literal, cds_end, seq_len),
                "the design spells a forbidden motif inside the window [{label} λ={lambda}]: {}",
                got.sequence,
            );
            assert!(
                has_occurrence_inside(got.sequence.as_bytes(), &literal, 0, cds_start),
                "the given flank lost its motifs [{label} λ={lambda}]: {}",
                got.sequence,
            );
        }
        by_table.push((label, designs));
    }
    // The constraint has to be reached on this path, and the only evidence for that
    // is a table that moved the answer. Without it the three cases above are
    // the inert one three times, and a diff that added the field without
    // consulting it would pass.
    let inert = &by_table[0].1;
    let moved: Vec<&str> = by_table[1..]
        .iter()
        .filter(|(_, d)| d != inert)
        .map(|(l, _)| *l)
        .collect();
    assert!(
        !moved.is_empty(),
        "no table changed the motif-constrained design, so this case cannot \
         distinguish a consulted constraint from an ignored one",
    );
    println!("tables that moved the motif-constrained design: {moved:?}");
}

/// The `(i, j)` pairs of a dot-bracket string.
fn pairs_of(structure: &str) -> Pairs {
    let mut stack = Vec::new();
    let mut out = Vec::new();
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => out.push((stack.pop().expect("balanced brackets"), i)),
            _ => {}
        }
    }
    out
}

/// A soft penalty must enter the constrained fold's energy exactly as it enters
/// the unconstrained scan's: added once per formed pair of the penalized class,
/// into the reported MFE.
///
/// A soft table that priced differently here would make `--motif` change what a
/// soft table *means*, which no caller could be expected to anticipate. The
/// arithmetic is checked against a re-evaluation of the emitted structure rather
/// than against another design, so a penalty credited twice, or folded into CAI,
/// or dropped, each fails distinctly — and the pair count is asserted non-zero,
/// because a penalty on a class that never forms is satisfied by doing nothing.
#[test]
fn a_soft_penalty_is_credited_exactly_once_on_the_constrained_path() {
    let (params, usage) = params_usage();
    let protein = "MKWDEY";
    let utr5_literal = "CCCCCC";
    let utr5 = UtrSpec::Fixed(utr5_literal.into());
    let utr3 = UtrSpec::Designed(6);
    let motifs: Vec<String> = MOTIFS.iter().map(|m| (*m).to_string()).collect();
    let (cds_start, cds_end) = (utr5.len(), utr5.len() + 3 * protein.len());
    let window = MotifWindow::new(cds_end as u32, (cds_end + utr3.len()) as u32);
    let penalty_dk = 10;
    // `cds-cds` rather than a crossing class: the arithmetic is the point, and
    // a class that reliably forms pairs is what keeps the check from passing
    // vacuously on a design that simply never triggers the penalty.
    let rc = RegionConstraints::default().with(
        PairClass::CdsCds,
        Enforcement::Penalize(derna_lib::Energy(penalty_dk)),
    );

    let mut total_penalized = 0usize;
    for &lambda in &LAMBDAS {
        let got = design_composite(
            &params,
            protein,
            &usage,
            lambda,
            &composite(&utr5, &utr3, rc, &motifs, window),
            ScanConfig::default(),
        )
        .expect("designs");
        let penalized = pairs_of(&got.structure)
            .into_iter()
            .filter(|&(i, j)| i >= cds_start && j < cds_end)
            .count();
        total_penalized += penalized;
        let raw_dk = evaluate(&params, got.sequence.as_bytes(), &got.structure)
            .expect("emitted structure scores")
            .dekacal();
        let reported_dk = (got.mfe_kcal_mol * 100.0).round() as i32;
        assert_eq!(
            reported_dk,
            raw_dk + penalty_dk * penalized as i32,
            "the soft penalty was not credited once per formed pair [λ={lambda}]: \
             reported={reported_dk} raw={raw_dk} penalized_pairs={penalized}",
        );
    }
    assert!(
        total_penalized > 0,
        "no pair of the penalized class ever formed, so this case cannot tell a \
         correctly-credited penalty from one that is never applied",
    );
}

// ===========================================================================
// 6. Reduction composition
// ===========================================================================

/// The differential a new constraint axis owes the interior reductions, and a
/// plain statement of what it settles.
///
/// The internal-loop closure has two forms — the Lyngsø–Zuker reduction and the
/// naive enumeration — and they must agree wherever both run. Under an active
/// motif set the fold routes to the un-reduced AC-threaded folder, which
/// consults neither, so agreement here confirms **mode-independence** rather
/// than that a reduction survived the product. What bears on survival is
/// measured separately: whether the AC axis is single-valued outside the window,
/// which is the premise any re-derived reduction would rest on
/// (`the_axis_collapses_outside_the_window_on_both_sidednesses`, and the
/// span-level check in the crate's own `constrained` tests).
#[test]
fn reduction_composition_holds_under_a_window_and_a_constraint() {
    let (params, usage) = params_usage();
    let protein = "MKWDE";
    let motifs: Vec<String> = MOTIFS.iter().map(|m| (*m).to_string()).collect();
    let none: Vec<String> = Vec::new();
    let utr5 = UtrSpec::Fixed("GGCAUGGC".into());
    let utr3 = UtrSpec::Designed(6);
    let cds_end = utr5.len() as u32 + 3 * protein.len() as u32;
    let window = MotifWindow::new(cds_end, cds_end + 6);

    for (label, rc) in tables() {
        for &lambda in &LAMBDAS {
            for (set_label, set) in [("scoped motifs", &motifs), ("no motifs", &none)] {
                let run = |mode| {
                    design_composite(
                        &params,
                        protein,
                        &usage,
                        lambda,
                        &composite(&utr5, &utr3, rc, set, window),
                        ScanConfig {
                            mode,
                            ..ScanConfig::default()
                        },
                    )
                    .expect("designs")
                };
                assert_eq!(
                    run(InternalLoopMode::Lz),
                    run(InternalLoopMode::Naive),
                    "Lyngsø–Zuker and naive diverged [{set_label} {label} λ={lambda}]",
                );
            }
        }
    }
}
