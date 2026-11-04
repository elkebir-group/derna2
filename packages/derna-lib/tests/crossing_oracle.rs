//! Brute-force gate for the crossing-chain pass of a segment certificate.
//!
//! On a molecule small enough to enumerate, every synonymous encoding × every
//! nested canonical structure is read the way the pass's own definition reads
//! it — the chain of pairs opening in `top ..= last`, each the single branch of
//! the loop the one before closes, ending in a hairpin or in the first enclosed
//! pair opening past `last` — and the chain's margin is computed from loop
//! energies the oracle derives itself: each loop is the energy of the
//! standalone substructure its closing pair encloses, less its children's, with
//! the exterior terminal-AU terms put back. Nothing is taken from the pass's
//! loop scorers.
//!
//! The gate is agreement in both directions at several budgets, including one
//! that **binds** (the budgeted optimum is worse than the unbudgeted one) and
//! one no chain meets — so a pass that ignored the budget, or one that
//! admitted nothing, fails rather than passes.

use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    crossing_composite, design_composite, CodonUsage, CompositeSpec, CrossingReport, CrossingSpec,
    Layout, MotifWindow, Params, Pins, PositionalConstraints, RegionConstraints, ScanConfig,
    StructureConstraints, UtrProfile, UtrSpec,
};
use rna_mfe::evaluate;

const UTR5: &str = "GGCA";
const UTR3: &str = "UGCC";
const PROTEIN: &str = "MFVW";
/// The Phe codon's three positions: the designed end of the segment.
const TOP: u32 = 7;
const LAST: u32 = 9;

fn nuc_byte(n: u8) -> u8 {
    [b'A', b'C', b'G', b'U'][n as usize]
}

fn canonical(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'C', b'G') | (b'G', b'C') | (b'G', b'U') | (b'U', b'G')
    )
}

fn au_like(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'U') | (b'U', b'G')
    )
}

/// Every encoding of the protein: its whole-molecule bytes and log-CAI, by the
/// `EPSILON`-floored formula the lattice's edge weights use.
fn encodings(usage: &CodonUsage) -> Vec<(Vec<u8>, f64)> {
    encodings_between(usage, UTR3)
}

/// [`encodings`] with another 3' UTR.
fn encodings_between(usage: &CodonUsage, utr3: &str) -> Vec<(Vec<u8>, f64)> {
    let aa: Vec<usize> = PROTEIN.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let counts: Vec<usize> = aa.iter().map(|&a| AMINO_ACIDS[a].codons.len()).collect();
    let total: usize = counts.iter().product();
    (0..total)
        .map(|mut n| {
            let mut seq = UTR5.as_bytes().to_vec();
            let mut cai = 0.0;
            for (&a, &c) in aa.iter().zip(&counts) {
                let ci = n % c;
                n /= c;
                seq.extend(AMINO_ACIDS[a].codons[ci].nuc.iter().map(|&x| nuc_byte(x)));
                let g = |k: usize| usage.fraction(a, k).max(EPSILON);
                let top = (0..c).map(g).fold(EPSILON, f64::max);
                cai += (g(ci) / top).ln();
            }
            seq.extend_from_slice(utr3.as_bytes());
            (seq, cai)
        })
        .collect()
}

fn walk(
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
        walk(seq, open, pairs, visit);
        open.push((lo, hi));
        return;
    }
    open.push((lo + 1, hi));
    walk(seq, open, pairs, visit);
    open.pop();
    for k in (lo + 4)..hi {
        if !canonical(seq[lo], seq[k]) {
            continue;
        }
        pairs.push((lo, k));
        open.push((k + 1, hi));
        open.push((lo + 1, k));
        walk(seq, open, pairs, visit);
        open.pop();
        open.pop();
        pairs.pop();
    }
    open.push((lo, hi));
}

fn dot_bracket(n: usize, pairs: &[(usize, usize)]) -> String {
    let mut s = vec![b'.'; n];
    for &(i, j) in pairs {
        s[i] = b'(';
        s[j] = b')';
    }
    String::from_utf8(s).unwrap()
}

/// The pairs immediately inside `(i, j)`.
fn children(pairs: &[(usize, usize)], (i, j): (usize, usize)) -> Vec<(usize, usize)> {
    let inside: Vec<_> = pairs
        .iter()
        .copied()
        .filter(|&(a, b)| i < a && b < j)
        .collect();
    inside
        .iter()
        .copied()
        .filter(|&(a, b)| !inside.iter().any(|&(c, d)| c < a && b < d))
        .collect()
}

struct Oracle<'a> {
    params: &'a Params,
}

impl Oracle<'_> {
    fn au(&self, seq: &[u8], (i, j): (usize, usize)) -> i32 {
        if au_like(seq[i], seq[j]) {
            self.params.misc.terminal_au_37
        } else {
            0
        }
    }

    /// The energy of the standalone substructure `(i, j)` encloses, without
    /// its exterior terminal-AU term: that pair's loop and everything inside.
    fn closed(&self, seq: &[u8], pairs: &[(usize, usize)], (i, j): (usize, usize)) -> i32 {
        let inner: Vec<_> = pairs
            .iter()
            .filter(|&&(a, b)| i <= a && b <= j)
            .map(|&(a, b)| (a - i, b - i))
            .collect();
        let sub = &seq[i..=j];
        evaluate(self.params, sub, &dot_bracket(sub.len(), &inner))
            .unwrap()
            .0
            - self.au(seq, (i, j))
    }

    fn loop_energy(&self, seq: &[u8], pairs: &[(usize, usize)], p: (usize, usize)) -> i32 {
        self.closed(seq, pairs, p)
            - children(pairs, p)
                .iter()
                .map(|&c| self.closed(seq, pairs, c))
                .sum::<i32>()
    }

    /// The chain from `top` and its margin, or `None` when the structure has
    /// no chain the pass scores (`top` unpaired, or a chain pair closing a
    /// multi-loop).
    fn chain_margin(&self, seq: &[u8], pairs: &[(usize, usize)], multiloop: bool) -> Option<i32> {
        let beta = |p| {
            self.au(seq, p)
                + if multiloop {
                    self.params.ml.intern_37
                } else {
                    0
                }
        };
        let top = *pairs.iter().find(|&&(i, _)| i == TOP as usize)?;
        let mut margin = beta(top);
        let mut at = top;
        loop {
            margin += self.loop_energy(seq, pairs, at);
            match children(pairs, at).as_slice() {
                [] => return Some(margin),
                [c] if c.0 <= LAST as usize => at = *c,
                [c] => return Some(margin - beta(*c)),
                _ => return None,
            }
        }
    }

    /// The best `(BRD, margin)` per encoding-and-structure with a scored
    /// chain: every candidate the pass's minimum ranges over.
    fn candidates(&self, usage: &CodonUsage, lambda: f64, multiloop: bool) -> Vec<(f64, i32)> {
        let mut out = Vec::new();
        for (seq, cai) in encodings(usage) {
            let mut open = vec![(0, seq.len())];
            let mut pairs = Vec::new();
            walk(&seq, &mut open, &mut pairs, &mut |ps: &[(
                usize,
                usize,
            )]| {
                if let Some(m) = self.chain_margin(&seq, ps, multiloop) {
                    let e = evaluate(self.params, &seq, &dot_bracket(seq.len(), ps))
                        .unwrap()
                        .0;
                    out.push((lambda * f64::from(e) - (1.0 - lambda) * cai * 100.0, m));
                }
            });
        }
        out
    }
}

fn run(params: &Params, usage: &CodonUsage, lambda: f64, spec: &CrossingSpec) -> CrossingReport {
    let n = (UTR5.len() + 3 * PROTEIN.len() + UTR3.len()) as u32;
    run_with(params, usage, lambda, spec, &PositionalConstraints::none(n))
}

fn composite<'a>(
    utr5: &'a UtrSpec,
    utr3: &'a UtrSpec,
    positional: &'a PositionalConstraints,
) -> CompositeSpec<'a, &'a str> {
    CompositeSpec {
        utr5,
        utr3,
        constraints: RegionConstraints::default(),
        profile: UtrProfile::inert(),
        motifs: &[],
        motif_window: MotifWindow::all(),
        flank_complement: None,
        complement_classes: None,
        positional,
    }
}

fn run_with(
    params: &Params,
    usage: &CodonUsage,
    lambda: f64,
    spec: &CrossingSpec,
    positional: &PositionalConstraints,
) -> CrossingReport {
    let utr5 = UtrSpec::Fixed(UTR5.into());
    let utr3 = UtrSpec::Fixed(UTR3.into());
    let statement = composite(&utr5, &utr3, positional);
    crossing_composite(
        params,
        PROTEIN,
        usage,
        lambda,
        &statement,
        ScanConfig::default(),
        spec,
    )
    .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(1.0)
}

#[test]
fn crossing_matches_enumeration_at_binding_and_empty_budgets() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let oracle = Oracle { params: &params };
    for (lambda, multiloop) in [(0.5, true), (0.5, false), (0.2, true)] {
        let cands = oracle.candidates(&usage, lambda, multiloop);
        assert!(!cands.is_empty(), "the instance has crossing chains");
        let any = cands.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
        let mut margins: Vec<i32> = cands.iter().map(|c| c.1).collect();
        margins.sort_unstable();
        let least = margins[0];
        let at_any = cands
            .iter()
            .filter(|c| close(c.0, any))
            .map(|c| c.1)
            .min()
            .unwrap();
        // A budget below the unbudgeted optimum's margin binds; one below every
        // margin admits nothing.
        let binding = at_any - 1;
        assert!(binding >= least, "the instance has a binding budget");
        for budget in [i32::MAX, binding, least, least - 1] {
            let spec = CrossingSpec {
                top: TOP,
                last: LAST,
                budget,
                multiloop,
                only: Vec::new(),
                depth_ceiling: f64::INFINITY,
            };
            let r = run(&params, &usage, lambda, &spec);
            let want = cands
                .iter()
                .filter(|c| c.1 <= budget)
                .map(|c| c.0)
                .fold(f64::INFINITY, f64::min);
            let got = r.best.as_ref().map_or(f64::INFINITY, |c| c.value);
            assert!(
                (want.is_infinite() && got.is_infinite()) || close(want, got),
                "λ {lambda} multiloop {multiloop} budget {budget}: oracle {want}, pass {got}"
            );
            assert!(close(r.best_any.as_ref().unwrap().value, any));
            if let Some(best) = &r.best {
                assert!(best.margin <= budget);
            }
            if budget == binding {
                assert!(got > any + 1e-9, "the binding budget moved the answer");
            }
        }
    }
}

#[test]
fn naming_a_chain_scores_that_chain_alone() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let all = run(
        &params,
        &usage,
        0.5,
        &CrossingSpec {
            top: TOP,
            last: LAST,
            budget: i32::MAX,
            multiloop: true,
            only: Vec::new(),
            depth_ceiling: f64::INFINITY,
        },
    );
    let best = all.best_any.unwrap();
    let named = run(
        &params,
        &usage,
        0.5,
        &CrossingSpec {
            top: TOP,
            last: LAST,
            budget: i32::MAX,
            multiloop: true,
            only: best.pairs.clone(),
            depth_ceiling: f64::INFINITY,
        },
    );
    let got = named.best_any.unwrap();
    assert_eq!(got.pairs, best.pairs);
    assert!(close(got.value, best.value));
    assert_eq!(got.margin, best.margin);
    assert!(
        named.shapes == 1 && all.shapes > 1,
        "naming a chain narrows the shapes"
    );
}

#[test]
fn a_depth_ceiling_keeps_every_chain_at_or_below_it() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let spec = |depth_ceiling| CrossingSpec {
        top: TOP,
        last: LAST,
        budget: i32::MAX,
        multiloop: true,
        only: Vec::new(),
        depth_ceiling,
    };
    let free = run(&params, &usage, 0.5, &spec(f64::INFINITY));
    let best = free.best.clone().unwrap();
    let depth = best.value - free.optimum;
    assert!(
        depth > 1.0,
        "the chain's optimum is not the scan's, so the ceiling can cut it"
    );
    let kept = run(&params, &usage, 0.5, &spec(depth + 1e-6));
    assert!(close(kept.best.unwrap().value, best.value));
    assert!(
        kept.shapes < free.shapes,
        "a ceiling above the answer still prunes shapes"
    );
    let cut = run(&params, &usage, 0.5, &spec(depth - 1.0));
    assert!(
        cut.best.is_none(),
        "nothing scores below the chain's own optimum"
    );
}

#[test]
fn a_pinned_residue_is_reported_on_the_unrestricted_scale() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let layout = Layout::new(UTR5.len() as u32, PROTEIN.len() as u32, UTR3.len() as u32);
    let mut pins = Pins::none(layout.len());
    // Phe's weaker codon: the lattice prices it against itself, so the offset
    // is non-zero and the test is not a no-op.
    pins.pin(&layout, TOP, "UUU").unwrap();
    let positional = PositionalConstraints {
        pins,
        structure: StructureConstraints::none(layout.len()),
    };
    let lambda = 0.5;
    let spec = CrossingSpec {
        top: TOP,
        last: LAST,
        budget: i32::MAX,
        multiloop: true,
        only: Vec::new(),
        depth_ceiling: f64::INFINITY,
    };
    let report = run_with(&params, &usage, lambda, &spec, &positional);
    assert!(report.objective_offset > 1.0, "UUU is not Phe's best codon");
    let utr5 = UtrSpec::Fixed(UTR5.into());
    let utr3 = UtrSpec::Fixed(UTR3.into());
    let statement = composite(&utr5, &utr3, &positional);
    let design = design_composite(
        &params,
        PROTEIN,
        &usage,
        lambda,
        &statement,
        ScanConfig::default(),
    )
    .unwrap();
    assert!(design.sequence[TOP as usize..].starts_with("UUU"));
    let objective =
        lambda * (design.mfe_kcal_mol * 100.0).round() - (1.0 - lambda) * 100.0 * design.cai_log;
    assert!(
        close(report.optimum, objective),
        "the crossing pass's optimum {} against the design's own objective {}",
        report.optimum,
        objective
    );
}

/// The segment of the margin-identity test: the 5' UTR's last two bases and
/// the coding sequence through Phe, `[2, 10)`.
const SEG: (usize, usize) = (2, 10);

/// A 3' UTR with room for a hairpin of its own and a partner for the 5' UTR's
/// `GG`, so the loop holding the segment can be a multi-loop.
const SEG_UTR3: &str = "GCAAAGCC";

/// The certificate's margin, computed from its definition — `E(G) − E(F)`
/// with `F` = `G` without the pairs touching the segment plus the segment's
/// own free fold — against the per-loop sum the bound is built on: every
/// loop an S-pair closes, plus each S-pair's branch term where it hangs in
/// L*, less each pair the certificate promotes into L*, less the segment
/// fold's own energy and branch terms. Checked over every encoding and every
/// structure whose L* is the exterior loop or a multi-loop in both `G` and
/// `F`; both kinds must occur, so the test is not passing on one of them.
#[test]
fn the_certificate_margin_is_a_sum_over_loops() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let o = Oracle { params: &params };
    let (lo, cut) = SEG;
    let touches = |(i, j): (usize, usize)| (lo..cut).contains(&i) || (lo..cut).contains(&j);
    let ml = params.ml.intern_37;
    let (mut checked, mut ext, mut multi, mut promoted) = (0u64, 0u64, 0u64, 0u64);
    for (seq, _) in encodings_between(&usage, SEG_UTR3) {
        let (sfold, _) = rna_mfe::fold(&params, &seq[lo..cut]).unwrap();
        let spairs: Vec<(usize, usize)> = pairs_of(&sfold)
            .into_iter()
            .map(|(i, j)| (i + lo, j + lo))
            .collect();
        let s_alone = evaluate(&params, &seq[lo..cut], &sfold).unwrap().0;
        let s_top = spairs
            .iter()
            .filter(|&&(a, b)| !spairs.iter().any(|&(c, d)| c < a && b < d))
            .count();
        let mut open = vec![(0, seq.len())];
        let mut pairs = Vec::new();
        walk(&seq, &mut open, &mut pairs, &mut |g: &[(
            usize,
            usize,
        )]| {
            let kept: Vec<_> = g.iter().copied().filter(|&p| !touches(p)).collect();
            let f: Vec<_> = kept.iter().chain(&spairs).copied().collect();
            let lstar = kept
                .iter()
                .copied()
                .filter(|&(i, j)| i < lo && j >= cut)
                .max();
            let is_ml = lstar.is_some();
            if let Some(p) = lstar {
                if children(g, p).len() < 2 || children(&f, p).len() < 2 {
                    return;
                }
            }
            let n = seq.len();
            let e_g = evaluate(&params, &seq, &dot_bracket(n, g)).unwrap().0;
            let e_f = evaluate(&params, &seq, &dot_bracket(n, &f)).unwrap().0;
            let beta = |p| o.au(&seq, p) + if is_ml { ml } else { 0 };
            let parent =
                |p: (usize, usize)| g.iter().copied().filter(|&(a, b)| a < p.0 && p.1 < b).max();
            let mut sum = 0;
            for &p in g.iter().filter(|&&p| touches(p)) {
                sum += o.loop_energy(&seq, g, p);
                for c in children(g, p) {
                    if !touches(c) {
                        sum -= beta(c);
                        promoted += 1;
                    }
                }
                match parent(p) {
                    q if q == lstar => sum += beta(p),
                    Some(q) => assert!(touches(q), "an S-pair hangs in a loop other than L*"),
                    None => unreachable!("no enclosing pair, yet L* exists"),
                }
            }
            let t = s_alone + if is_ml { ml * s_top as i32 } else { 0 };
            assert_eq!(
                e_g - e_f,
                sum - t,
                "{} {} {}",
                String::from_utf8_lossy(&seq),
                dot_bracket(n, g),
                dot_bracket(n, &f)
            );
            checked += 1;
            if is_ml {
                multi += 1;
            } else {
                ext += 1;
            }
        });
    }
    eprintln!("{checked} structures: {ext} exterior, {multi} multi-loop L*; {promoted} promotions");
    assert!(
        ext > 1000 && multi > 0,
        "both kinds of L*: {ext} exterior, {multi} multi-loop"
    );
    assert!(promoted > 0, "some structure promotes a pair");
}

/// The pairs of a dot-bracket string.
fn pairs_of(db: &str) -> Vec<(usize, usize)> {
    let (mut st, mut out) = (Vec::new(), Vec::new());
    for (j, c) in db.bytes().enumerate() {
        match c {
            b'(' => st.push(j),
            b')' => out.push((st.pop().unwrap(), j)),
            _ => {}
        }
    }
    out
}
