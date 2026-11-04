//! An exact path oracle for [`Dfa::broken_codon_cai`].
//!
//! The closed form answers a span from its two partial end codons plus a
//! prefix-sum middle term for the whole units in between. The builders shift a
//! restricted codon unit (or a restricted flank position) so its best available
//! weight is zero and report the constant separately, which makes that middle
//! term exactly zero on every graph they build. A closed form that *assumed* the
//! zero — or a builder that forgot the shift under one — over-credits every
//! span across the unit, and since the recurrences minimize over that value the
//! exact optimum comes back wrong rather than a bound looser.
//!
//! The oracle here knows nothing about offsets. It relaxes the best edge-weight
//! sum forward over the same graph's edges, so any over-credit shows up as a
//! disagreement on a span that crosses the unit. Four things keep it modular:
//!
//! 1. **Coverage by constructor.** Every public builder is exercised — the plain
//!    lattice, single-path, fixed-mRNA, given / designed / IUPAC-masked flanks,
//!    an active profile, codon restrictions, and pins lowered the way the
//!    composite entry lowers them — over seeded random short proteins, on every
//!    node pair of every lane.
//! 2. **Non-vacuity.** The restricted cases carry offsets strictly below zero,
//!    beside controls at exactly zero, so the shift under test is not a no-op.
//! 3. **A mutant.** With the shift and the middle term both switched off the
//!    oracle disagrees on every graph the shift would have moved, and on no
//!    other.
//! 4. **Independence from the shift.** With the shift off and the middle term
//!    on, the closed form and every exact answer downstream of its splits stay
//!    right.

use super::*;
use crate::coords::Layout;
use crate::pins::Pins;
use derna_codon_dp::codon_table::aa_to_index;

/// The twenty standard residues, in a fixed order for the seeded draws.
const RESIDUES: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

/// A small deterministic generator: the draws only need to be reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

/// The best edge-weight sum over every DFA path from `a` to `b`, or `None`
/// when no path joins them — a forward relax over the positions between,
/// reading nothing but [`Dfa::right`].
fn best_path_cai(dfa: &Dfa, a: Node, b: Node) -> Option<f64> {
    if a == b {
        return Some(0.0);
    }
    if a.pos >= b.pos {
        return None;
    }
    let mut best = vec![f64::NEG_INFINITY; dfa.slots()];
    best[dfa.ix(a)] = 0.0;
    for pos in a.pos..b.pos {
        for &node in dfa.nodes_at(pos) {
            let here = best[dfa.ix(node)];
            if here == f64::NEG_INFINITY {
                continue;
            }
            for e in dfa.right(node) {
                let slot = &mut best[dfa.ix(e.to)];
                *slot = slot.max(here + e.cai);
            }
        }
    }
    let w = best[dfa.ix(b)];
    (w != f64::NEG_INFINITY).then_some(w)
}

/// Run `f` with the closed form's whole-unit middle term switched off on this
/// thread, so it credits every whole unit with zero.
fn with_middle_term_skipped<T>(f: impl FnOnce() -> T) -> T {
    SKIP_MIDDLE_TERM.with(|s| s.set(true));
    let out = f();
    SKIP_MIDDLE_TERM.with(|s| s.set(false));
    out
}

/// Run `f` with the per-unit normalization switched off on this thread.
fn with_normalization_skipped<T>(f: impl FnOnce() -> T) -> T {
    SKIP_NORMALIZATION.with(|s| s.set(true));
    let out = f();
    SKIP_NORMALIZATION.with(|s| s.set(false));
    out
}

/// Every node of `dfa`, every lane each position has.
fn all_nodes(dfa: &Dfa) -> Vec<Node> {
    (0..=dfa.seq_len)
        .flat_map(|pos| dfa.nodes_at(pos).iter().copied())
        .collect()
}

/// How a span sits against the unit boundaries — the three shapes the closed
/// form treats differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Shape {
    WithinOneUnit,
    AcrossOneBoundary,
    AcrossSeveral,
}

fn shape(dfa: &Dfa, a: Node, b: Node) -> Shape {
    let boundaries = (a.pos + 1..b.pos)
        .filter(|&p| dfa.codon_start(p) == p)
        .count();
    match boundaries {
        0 => Shape::WithinOneUnit,
        1 => Shape::AcrossOneBoundary,
        _ => Shape::AcrossSeveral,
    }
}

/// Whether two builds of one graph carry any different edge weight.
fn edges_differ(a: &Dfa, b: &Dfa) -> bool {
    all_nodes(a).into_iter().any(|n| {
        a.right(n)
            .iter()
            .zip(b.right(n))
            .any(|(x, y)| x.cai.to_bits() != y.cai.to_bits())
    })
}

/// The spans on which the closed form and the oracle disagree, `None`
/// included, compared to the bit.
///
/// To the bit is right on a shifted graph, where every middle term is exactly
/// zero and both sides add the same partial weights in the same order.
fn disagreements(dfa: &Dfa) -> Vec<(Node, Node, Option<f64>, Option<f64>)> {
    spans_apart(dfa, |x, y| x.to_bits() != y.to_bits())
}

/// [`disagreements`] beyond a rounding tolerance — for an unshifted graph,
/// where the middle term is a difference of prefix sums and the oracle a
/// running sum, so the two round differently in the last place. The errors the
/// mutants make are whole log-weights, many orders larger.
fn disagreements_beyond_rounding(dfa: &Dfa) -> Vec<(Node, Node, Option<f64>, Option<f64>)> {
    spans_apart(dfa, |x, y| (x - y).abs() > 1e-9)
}

fn spans_apart(
    dfa: &Dfa,
    apart: impl Fn(f64, f64) -> bool,
) -> Vec<(Node, Node, Option<f64>, Option<f64>)> {
    let nodes = all_nodes(dfa);
    let mut out = Vec::new();
    for &a in &nodes {
        for &b in nodes.iter().filter(|b| b.pos >= a.pos) {
            let closed = dfa.broken_codon_cai(a, b);
            let exact = best_path_cai(dfa, a, b);
            let differ = match (closed, exact) {
                (Some(x), Some(y)) => apart(x, y),
                (x, y) => x.is_some() != y.is_some(),
            };
            if differ {
                out.push((a, b, closed, exact));
            }
        }
    }
    out
}

/// One graph under test, the profile it was built under, and whether a
/// builder narrowed some alphabet in it.
struct Case {
    name: String,
    dfa: Dfa,
    profile: UtrProfile,
    restricted: bool,
}

/// Whether a dot-bracket structure holds a multi-loop: a pair that directly
/// encloses two or more pairs.
fn has_multiloop(structure: &str) -> bool {
    let mut branches: Vec<usize> = vec![0];
    for c in structure.chars() {
        match c {
            '(' => {
                *branches.last_mut().expect("the exterior level") += 1;
                branches.push(0);
            }
            ')' if branches.pop().is_some_and(|inner| inner >= 2) => return true,
            _ => {}
        }
    }
    false
}

fn usage() -> CodonUsage {
    CodonUsage::load_default().expect("the canonical human table")
}

fn random_protein(rng: &mut Rng, len: usize) -> Vec<usize> {
    (0..len)
        .map(|_| aa_to_index(RESIDUES[rng.below(RESIDUES.len())] as char).expect("standard"))
        .collect()
}

/// A random non-empty codon subset per residue. Each draw keeps a codon with
/// probability one half, so the family argmax is dropped about half the time.
fn random_restriction(rng: &mut Rng, protein: &[usize]) -> Vec<Vec<[u8; 3]>> {
    protein
        .iter()
        .map(|&aa| {
            let codons: Vec<[u8; 3]> = AMINO_ACIDS[aa].codons.iter().map(|c| c.nuc).collect();
            let kept: Vec<[u8; 3]> = codons.iter().copied().filter(|_| rng.coin()).collect();
            if kept.is_empty() {
                vec![codons[rng.below(codons.len())]]
            } else {
                kept
            }
        })
        .collect()
}

fn random_nucs(rng: &mut Rng, len: usize) -> Vec<Nucleotide> {
    (0..len).map(|_| nuc(rng.below(4) as u8)).collect()
}

/// A random non-empty nucleotide subset per position.
fn random_mask(rng: &mut Rng, len: usize) -> Vec<NucSet> {
    (0..len)
        .map(|_| {
            let picked: Vec<Nucleotide> = (0..4u8).filter(|_| rng.coin()).map(nuc).collect();
            NucSet::from_nucleotides(&picked).unwrap_or(NucSet::of(nuc(rng.below(4) as u8)))
        })
        .collect()
}

/// A profile favouring one nucleotide strongly, so a mask leaving it out
/// moves weight out of the edges.
fn favouring(nt: usize) -> UtrProfile {
    let mut f = [0.1; 4];
    f[nt] = 0.7;
    UtrProfile::new(f, 1.0).expect("a valid profile")
}

/// A pin set over a flanked layout, lowered exactly as the composite entry
/// lowers it: CDS pins to codon subsets, flank pins to per-position masks.
fn pinned(rng: &mut Rng, protein: &[usize], usage: &CodonUsage, flank: u32) -> Option<Dfa> {
    let layout = Layout::new(flank, protein.len() as u32, flank);
    let mut pins = Pins::none(layout.len());
    for _ in 0..4 {
        let at = rng.below(layout.len() as usize) as u32;
        let code = b"ACGURYMKSWN"[rng.below(11)];
        // A pin that conflicts with an earlier one is simply skipped: the draw
        // only needs some pins to land.
        let _ = pins.pin(&layout, at, std::str::from_utf8(&[code]).expect("ascii"));
    }
    let allowed = pins.cds_codon_sets(&layout, protein).ok()?;
    let full = vec![NucSet::ALL; flank as usize];
    let five = pins.flank_mask(&layout, 0, &full).ok()?;
    let three = pins.flank_mask(&layout, layout.cds_end(), &full).ok()?;
    build_composite(
        protein,
        usage,
        allowed.as_deref(),
        UtrChain::Mixed(&five),
        UtrChain::Mixed(&three),
        favouring(2),
    )
}

/// Every constructor, over a few seeded random proteins each.
fn cases(usage: &CodonUsage) -> Vec<Case> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut out = Vec::new();
    let mut push = |name: String, dfa: Dfa, profile: UtrProfile, restricted: bool| {
        out.push(Case {
            name,
            dfa,
            profile,
            restricted,
        })
    };
    let inert = UtrProfile::inert();
    for seed in 0..4 {
        let len = 3 + rng.below(4);
        let p = random_protein(&mut rng, len);
        let flank = 1 + rng.below(4);
        let (u5, u3) = (random_nucs(&mut rng, flank), random_nucs(&mut rng, flank));
        let (m5, m3) = (random_mask(&mut rng, flank), random_mask(&mut rng, flank));
        let allowed = random_restriction(&mut rng, &p);
        let codons: Vec<[u8; 3]> = allowed.iter().map(|set| set[0]).collect();
        let profile = favouring(rng.below(4));
        let tag = |what: &str| format!("{what} #{seed}");

        push(tag("build"), build(&p, usage), inert, false);
        push(
            tag("build_single_path"),
            build_single_path(&p, &codons),
            inert,
            false,
        );
        let mrna: Vec<Nucleotide> = u5.iter().chain(&u3).copied().collect();
        push(
            tag("build_fixed_mrna"),
            build_fixed_mrna(&mrna),
            inert,
            false,
        );
        push(
            tag("build_with_utr"),
            build_with_utr(&p, usage, &u5, &u3),
            inert,
            false,
        );
        push(
            tag("build_with_utr_chain designed"),
            build_with_utr_chain(
                &p,
                usage,
                UtrChain::Designed(flank as u32),
                UtrChain::Designed(flank as u32),
            ),
            inert,
            false,
        );
        push(
            tag("build_with_utr_chain mixed"),
            build_with_utr_chain(&p, usage, UtrChain::Mixed(&m5), UtrChain::Mixed(&m3)),
            inert,
            false,
        );
        push(
            tag("build_with_utr_profile designed"),
            build_with_utr_profile(
                &p,
                usage,
                UtrChain::Designed(flank as u32),
                UtrChain::Fixed(&u3),
                profile,
            ),
            profile,
            // A given flank under an active profile is a one-letter
            // restriction: its single edge is shifted to zero too.
            true,
        );
        push(
            tag("build_with_utr_profile mixed"),
            build_with_utr_profile(
                &p,
                usage,
                UtrChain::Mixed(&m5),
                UtrChain::Mixed(&m3),
                profile,
            ),
            profile,
            true,
        );
        push(
            tag("build_restricted_with_utr cds-only"),
            build_restricted_with_utr(&p, usage, &allowed, &[], &[]).expect("non-empty"),
            inert,
            true,
        );
        push(
            tag("build_restricted_with_utr flanked"),
            build_restricted_with_utr(&p, usage, &allowed, &u5, &u3).expect("non-empty"),
            inert,
            true,
        );
        push(
            tag("build_composite"),
            build_composite(
                &p,
                usage,
                Some(&allowed),
                UtrChain::Mixed(&m5),
                UtrChain::Designed(flank as u32),
                profile,
            )
            .expect("non-empty"),
            profile,
            true,
        );
        if let Some(dfa) = pinned(&mut rng, &p, usage, flank as u32) {
            push(tag("pins"), dfa, favouring(2), true);
        }
    }
    out
}

/// The closed form equals the best path on every span of every constructor's
/// graph, to the bit, `None` included — and the spans asked cover all three
/// shapes with non-zero answers, so agreement is not agreement on zeros.
#[test]
fn the_closed_form_is_the_best_path_on_every_constructor() {
    let usage = usage();
    let mut weighted: FxHashMap<Shape, usize> = FxHashMap::default();
    for case in cases(&usage) {
        let wrong = disagreements(&case.dfa);
        assert!(
            wrong.is_empty(),
            "{}: {} spans disagree, first {:?}",
            case.name,
            wrong.len(),
            wrong[0]
        );
        let nodes = all_nodes(&case.dfa);
        for &a in &nodes {
            for &b in nodes.iter().filter(|b| b.pos > a.pos) {
                if case.dfa.broken_codon_cai(a, b).is_some_and(|w| w != 0.0) {
                    *weighted.entry(shape(&case.dfa, a, b)).or_default() += 1;
                }
            }
        }
    }
    for s in [
        Shape::WithinOneUnit,
        Shape::AcrossOneBoundary,
        Shape::AcrossSeveral,
    ] {
        assert!(
            weighted.get(&s).copied().unwrap_or(0) > 0,
            "no weighted span of shape {s:?}"
        );
    }
}

/// A second, independent reference on short spans: brute-force enumeration of
/// every path. Constraints the oracle's own relax, which the test above trusts.
#[test]
fn the_oracle_matches_path_enumeration_on_short_spans() {
    let usage = usage();
    for case in cases(&usage) {
        let nodes = all_nodes(&case.dfa);
        for &a in &nodes {
            for &b in nodes
                .iter()
                .filter(|b| b.pos >= a.pos && b.pos <= a.pos + 7)
            {
                let enumerated = case
                    .dfa
                    .paths_between(a, b)
                    .into_iter()
                    .map(|(w, _)| w)
                    .reduce(f64::max);
                assert_eq!(
                    best_path_cai(&case.dfa, a, b).map(f64::to_bits),
                    enumerated.map(f64::to_bits),
                    "{}: {a:?} → {b:?}",
                    case.name
                );
            }
        }
    }
}

/// The restricted cases really move adaptation out of the edges — otherwise
/// the oracle would be testing a shift of zero. Fixed instances with a control
/// at exactly zero, plus a count over the random draws.
#[test]
fn the_restricted_cases_carry_negative_offsets() {
    let usage = usage();
    let p: Vec<usize> = "MLWKC".chars().map(|c| aa_to_index(c).unwrap()).collect();
    let mut sets: Vec<Vec<[u8; 3]>> = p
        .iter()
        .map(|&aa| AMINO_ACIDS[aa].codons.iter().map(|c| c.nuc).collect())
        .collect();
    // `CUG` is human leucine's argmax; `{UUA, UUG}` excludes it, `{CUG, UUG}`
    // keeps it.
    sets[1] = vec![[3, 3, 0], [3, 3, 2]];
    let excluding = build_restricted_with_utr(&p, &usage, &sets, &[], &[]).expect("non-empty");
    assert!(excluding.codon_offset() < 0.0);
    sets[1] = vec![[1, 3, 2], [3, 3, 2]];
    let keeping = build_restricted_with_utr(&p, &usage, &sets, &[], &[]).expect("non-empty");
    assert_eq!(keeping.codon_offset(), 0.0);
    assert_eq!(build(&p, &usage).codon_offset(), 0.0);

    // A `G`-favouring profile over `{A, C}` positions, against an
    // `A`-favouring control the mask admits.
    let mask = [NucSet::from_iupac(b'M').expect("M is {A, C}"); 3];
    let flanked = |profile| {
        build_with_utr_profile(
            &p,
            &usage,
            UtrChain::Mixed(&mask),
            UtrChain::Fixed(&[]),
            profile,
        )
    };
    assert!(flanked(favouring(2)).utr_profile_offset(favouring(2)) < 0.0);
    assert_eq!(flanked(favouring(0)).utr_profile_offset(favouring(0)), 0.0);

    let negative = cases(&usage)
        .iter()
        .filter(|c| c.restricted)
        .filter(|c| c.dfa.codon_offset() < 0.0 || c.dfa.utr_profile_offset(c.profile) < 0.0)
        .count();
    assert!(
        negative >= 4,
        "only {negative} random restricted cases shift"
    );
}

/// The mutant: on graphs built without the shift, a closed form that assumed
/// every whole unit's best is zero — its middle term switched off — is caught
/// on every graph whose edges the shift would have moved, and on no other. With
/// the middle term on, the closed form stays exact on those same graphs.
#[test]
fn the_oracle_catches_a_closed_form_that_assumes_the_shift() {
    let usage = usage();
    let p: Vec<usize> = "MLWKCL".chars().map(|c| aa_to_index(c).unwrap()).collect();
    let mut sets: Vec<Vec<[u8; 3]>> = p
        .iter()
        .map(|&aa| AMINO_ACIDS[aa].codons.iter().map(|c| c.nuc).collect())
        .collect();
    sets[1] = vec![[3, 3, 0], [3, 3, 2]];
    let mask = [NucSet::from_iupac(b'M').expect("M is {A, C}"); 3];
    let fixed = || {
        (
            build_restricted_with_utr(&p, &usage, &sets, &[], &[]).expect("non-empty"),
            build_with_utr_profile(
                &p,
                &usage,
                UtrChain::Mixed(&mask),
                UtrChain::Fixed(&[]),
                favouring(2),
            ),
        )
    };
    let (codons, flank) = with_middle_term_skipped(|| with_normalization_skipped(fixed));
    assert!(
        !disagreements_beyond_rounding(&codons).is_empty(),
        "missed the codon shift"
    );
    assert!(
        !disagreements_beyond_rounding(&flank).is_empty(),
        "missed the flank shift"
    );
    let (codons, flank) = with_normalization_skipped(fixed);
    assert!(disagreements_beyond_rounding(&codons).is_empty());
    assert!(disagreements_beyond_rounding(&flank).is_empty());

    // Across every constructor: the mutant is caught exactly when the shift
    // moved some edge, and the shift moves edges on restricted graphs only.
    let unshifted = with_normalization_skipped(|| cases(&usage));
    let mutants = with_middle_term_skipped(|| with_normalization_skipped(|| cases(&usage)));
    let mut caught = 0;
    for ((normal, raw), mutant) in cases(&usage).iter().zip(&unshifted).zip(&mutants) {
        let shifted = edges_differ(&normal.dfa, &raw.dfa);
        assert!(normal.restricted || !shifted, "{}: shifted", normal.name);
        assert!(
            disagreements_beyond_rounding(&raw.dfa).is_empty(),
            "{}: the closed form needs the shift",
            normal.name
        );
        assert_eq!(
            !disagreements_beyond_rounding(&mutant.dfa).is_empty(),
            shifted,
            "{}: the oracle and the knob disagree on whether anything moved",
            normal.name
        );
        caught += usize::from(shifted);
    }
    assert!(caught >= 8, "only {caught} cases shifted anything");
}

/// The answers the closed form's splits feed — not only the closed form
/// itself — do not depend on the shift. With it skipped the lattice carries raw
/// weights and nothing is added back, yet the scan's design, both backward
/// passes' marginals on a design's reported scale, and a flanked front (up to
/// the profile constant its report still adds) come out as they do with it.
/// What the shift still decides is the report's add-back, which
/// `tests/closed_form_report.rs` owns.
#[test]
fn the_exact_answers_do_not_depend_on_the_shift() {
    use crate::{Beam, Params, RegionConstraints, ScanConfig};
    let params = Params::load_default().expect("params");
    let usage = usage();
    let protein = "MLWKCLKYQ";
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let (leu, lys) = (aa_to_index('L').unwrap(), aa_to_index('K').unwrap());
    let allowed: Vec<Vec<[u8; 3]>> = aa
        .iter()
        .map(|&a| match a {
            _ if a == leu => vec![[3, 3, 0], [3, 3, 2]],
            _ if a == lys => vec![[0, 0, 0]],
            _ => AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect(),
        })
        .collect();
    let config = ScanConfig {
        beam: Beam::exact(),
        ..ScanConfig::default()
    };
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
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
    ) -> Result<Option<crate::outside::EdgeMarginals>, crate::BeamError>;
    let passes: [(&str, Pass); 2] = [
        ("rules", crate::outside::edge_marginals),
        ("scan", crate::outside::edge_marginals_fast),
    ];
    for lambda in [0.0001, 0.5, 0.99] {
        for (utr5, utr3) in [("", ""), ("GGAUC", "CAUGC")] {
            let design = || {
                crate::design_restricted(
                    &params, protein, &usage, lambda, &allowed, utr5, utr3, config,
                )
                .expect("valid input")
                .expect("feasible")
            };
            let (normal, mutant) = (design(), with_normalization_skipped(design));
            assert!(
                close(normal.mfe_kcal_mol, mutant.mfe_kcal_mol)
                    && close(normal.cai_log, mutant.cai_log),
                "λ={lambda} {utr5:?}: design {normal:?} became {mutant:?}"
            );
            for (name, pass) in passes {
                let marginals = || {
                    pass(
                        &params,
                        protein,
                        &usage,
                        lambda,
                        &allowed,
                        utr5,
                        utr3,
                        RegionConstraints::default(),
                        config,
                    )
                    .expect("valid input")
                    .expect("feasible")
                };
                let (normal, mutant) = (marginals(), with_normalization_skipped(marginals));
                assert!(
                    close(normal.optimum(), mutant.optimum()),
                    "λ={lambda} {utr5:?} {name}: optimum {} became {}",
                    normal.optimum(),
                    mutant.optimum()
                );
                let (ne, me) = (normal.cds_edges(), mutant.cds_edges());
                assert_eq!(ne.len(), me.len(), "λ={lambda} {utr5:?} {name}");
                for (x, y) in ne.iter().zip(&me) {
                    assert!(
                        x.0 == y.0 && x.1 == y.1 && close(x.2, y.2),
                        "λ={lambda} {utr5:?} {name}: edge {x:?} became {y:?}"
                    );
                }
            }
        }
    }

    // Long enough for the reductions that read the split rather than the whole
    // span — the one-sided corner factorization, the Lyngsø–Zuker index and
    // the multi-loop opening pool — every residue held away from its argmax
    // where it has a second codon.
    let long = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPD";
    let long_aa: Vec<usize> = long.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let all_but_best: Vec<Vec<[u8; 3]>> = long_aa
        .iter()
        .map(|&a| {
            let codons = AMINO_ACIDS[a].codons;
            let best = (0..codons.len())
                .max_by(|&x, &y| usage.fraction(a, x).total_cmp(&usage.fraction(a, y)))
                .expect("a residue has a codon");
            let rest: Vec<[u8; 3]> = (0..codons.len())
                .filter(|&c| c != best)
                .map(|c| codons[c].nuc)
                .collect();
            if rest.is_empty() {
                vec![codons[best].nuc]
            } else {
                rest
            }
        })
        .collect();
    for lambda in [0.0001, 0.3, 0.7] {
        let design = || {
            crate::design_restricted(&params, long, &usage, lambda, &all_but_best, "", "", config)
                .expect("valid input")
                .expect("feasible")
        };
        let (normal, mutant) = (design(), with_normalization_skipped(design));
        assert!(
            close(normal.mfe_kcal_mol, mutant.mfe_kcal_mol)
                && close(normal.cai_log, mutant.cai_log),
            "λ={lambda} long: design {normal:?} became {mutant:?}"
        );
    }

    // A front over a flank masked away from the profile's argmax. Its report
    // adds the profile constant back from the profile rather than from the
    // edges, so the mutant reads exactly that much higher.
    let profile = favouring(2);
    let mask = [b'M', b'M', b'N', b'N'].map(|c| NucSet::from_iupac(c).expect("iupac"));
    let front_aa: Vec<usize> = "MLWKC".chars().map(|c| aa_to_index(c).unwrap()).collect();
    let shift = build_with_utr_profile(
        &front_aa,
        &usage,
        UtrChain::Mixed(&mask),
        UtrChain::Fixed(&[]),
        profile,
    )
    .utr_profile_offset(profile);
    assert!(shift < 0.0);
    let front = || match crate::design_front_bounded_with_utr(
        &params,
        "MLWKC",
        &usage,
        &crate::UtrSpec::Mixed("MMNN".into()),
        &crate::UtrSpec::Fixed(String::new()),
        profile,
        crate::Prune::None,
        crate::Bounds::UNBOUNDED,
        1,
    )
    .expect("the flanked fold answers")
    {
        crate::Bounded::Feasible(f) => f.front,
        crate::Bounded::Unreachable { .. } => unreachable!("no ceiling was set"),
    };
    let (normal, mutant) = (front(), with_normalization_skipped(front));
    assert_eq!(normal.len(), mutant.len(), "the front changed size");
    for (x, y) in normal.iter().zip(&mutant) {
        assert!(
            close(x.mfe_kcal_mol, y.mfe_kcal_mol) && close(x.cai_log, y.cai_log - shift),
            "front point {x:?} became {y:?}"
        );
    }
}

/// The multi-loop opening pool decides an answer only where the optimum holds a
/// multi-loop, which no optimum in
/// [`the_exact_answers_do_not_depend_on_the_shift`] does. Here it does
/// (asserted), so a pool seed carrying the bare partial codon instead of
/// `span_head_cai` moves the design (the column fill's seed) or the fast
/// outside pass's edge values (its replay and its `M2` tails). The push scan's
/// seed is the same term on a path only `DERNA_PUSH_SCAN` selects, which a unit
/// test cannot set without leaking into its neighbours.
#[test]
fn the_multi_loop_pool_does_not_depend_on_the_shift() {
    use crate::{Beam, Params, RegionConstraints, ScanConfig};
    let params = Params::load_default().expect("params");
    let usage = usage();
    let config = ScanConfig {
        beam: Beam::exact(),
        ..ScanConfig::default()
    };
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
    let lambda = 0.0001;
    let protein = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV";
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let all_but_best: Vec<Vec<[u8; 3]>> = aa
        .iter()
        .map(|&a| {
            let codons = AMINO_ACIDS[a].codons;
            let best = (0..codons.len())
                .max_by(|&x, &y| usage.fraction(a, x).total_cmp(&usage.fraction(a, y)))
                .expect("a residue has a codon");
            let rest: Vec<[u8; 3]> = (0..codons.len())
                .filter(|&c| c != best)
                .map(|c| codons[c].nuc)
                .collect();
            if rest.is_empty() {
                vec![codons[best].nuc]
            } else {
                rest
            }
        })
        .collect();
    let design = || {
        crate::design_restricted(
            &params,
            protein,
            &usage,
            lambda,
            &all_but_best,
            "",
            "",
            config,
        )
        .expect("valid input")
        .expect("feasible")
    };
    let (normal, mutant) = (design(), with_normalization_skipped(design));
    assert!(
        has_multiloop(&normal.structure),
        "the optimum holds no multi-loop, so the pool decides nothing"
    );
    assert!(
        close(normal.mfe_kcal_mol, mutant.mfe_kcal_mol) && close(normal.cai_log, mutant.cai_log),
        "design {normal:?} became {mutant:?}"
    );
    let marginals = || {
        crate::outside::edge_marginals_fast(
            &params,
            protein,
            &usage,
            lambda,
            &all_but_best,
            "",
            "",
            RegionConstraints::default(),
            config,
        )
        .expect("valid input")
        .expect("feasible")
    };
    let (normal, mutant) = (marginals(), with_normalization_skipped(marginals));
    assert!(close(normal.optimum(), mutant.optimum()), "optimum");
    let (ne, me) = (normal.cds_edges(), mutant.cds_edges());
    assert_eq!(ne.len(), me.len());
    for (x, y) in ne.iter().zip(&me) {
        assert!(
            x.0 == y.0 && x.1 == y.1 && close(x.2, y.2),
            "edge {x:?} became {y:?}"
        );
    }
}
