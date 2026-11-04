//! Local search emits only modular designs, by a second pair of DPs.
//!
//! [`derna_lib::repair`] decides tightness with two Zuker folds. Here every
//! design it reports tight is re-certified independently: the constrained energy by
//! the scan under full mutual isolation, the free energy by LinearFold at an
//! exact beam, and the two must agree. A design whose gap only narrowed must
//! fail that certificate — the mutant a weaker acceptance rule ("the gap shrank")
//! would let through. A climb's every recorded design passes the same
//! certificate, and its path takes the best tight neighbour at each step.

use derna_lib::free_fold::{sequence_constrained_mfe, sequence_free_fold, CdsWindow};
use derna_lib::region::{Enforcement, RegionConstraints};
use derna_lib::repair::{Aim, Evaluated, Menus, Searcher, Toward};
use derna_lib::{CodonUsage, Params};

/// The fixture protein's first 20 residues.
const PROTEIN: &str = "MFVFLVLLPLVSSQCVNLTT";

/// A relaxed-front design of `PROTEIN` (gap 0.4 under `RICH_POOR`, 1.9 under `HBB`).
const START: &str = "AUGUUCGUGUUCCUGGUGCUGCUGCCUCUGGUGAGCAGCCAGUGCGUGAACCUGACCACC";

/// A synthetic, rarely pairing UTR, used on both sides.
const RICH_POOR: &str = "ACAACCACAACCAAACCACAACCAACACCAAACCACAACCAAACCACAAC";

/// A real UTR pair, under which no single change makes the start tight.
const HBB: (&str, &str) = (
    "ACAUUUGCUUCUGACACAACUGUGUUCACUAGCAACCUCAAACAGACACC",
    "GCUCGCUUUCUUGCUGUCCAAUUUCUAUUAAAGGUUCCUUUGUUCCCUAAGUCCAACUACUAAACUGGGGGAUAUUAUGAAGGGCCUUGAGCAUCUGGAUUCUGCCUAAUAAAAAACAUUUAUUUUCAUUGC",
);

/// Whether the scan and LinearFold, neither used by the search, agree that the
/// design is modular.
fn certified(params: &Params, (utr5, utr3): (&str, &str), cds: &str) -> bool {
    let molecule = format!("{utr5}{cds}{utr3}");
    let window = CdsWindow::new(utr5.len(), cds.len() / 3);
    let constrained = sequence_constrained_mfe(
        params,
        &molecule,
        window,
        RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
    )
    .expect("constrained fold");
    let free = sequence_free_fold(params, &molecule).expect("free fold").1;
    (constrained * 100.0).round() as i32 == free.0
}

fn searcher<'a>(params: &'a Params, utrs: (&'a str, &'a str)) -> Searcher<'a> {
    let usage = CodonUsage::load_default().expect("codon usage");
    let menus = Menus::new(PROTEIN, &usage).expect("menus");
    Searcher::new(params, utrs, menus, 4, Aim::Weight).expect("searcher")
}

#[test]
fn a_reported_tight_design_is_certified_by_independent_folds() {
    let params = Params::load_default().expect("parameters");
    let utrs = (RICH_POOR, RICH_POOR);
    let o = searcher(&params, utrs).run(START, 8, 0).expect("search");
    assert_eq!(o.start.gap(), 40, "the fixture starts 0.4 from tight");
    assert!(o.end.is_tight(), "the search reaches a tight design here");
    assert!(
        certified(&params, utrs, &o.end.cds),
        "and the scan and LinearFold agree"
    );
    assert!(o.weight <= 0.0);
}

#[test]
fn a_narrowed_gap_is_not_reported_tight() {
    let params = Params::load_default().expect("parameters");
    let o = searcher(&params, HBB).run(START, 8, 0).expect("search");
    assert!(
        o.end.gap() < o.start.gap(),
        "the search narrows the gap here"
    );
    assert!(o.end.gap() > 0, "but does not close it");
    assert!(!o.end.is_tight());
    // The mutant acceptance rule would report this design; the certificate refuses it.
    assert!(!certified(&params, HBB, &o.end.cds));
}

#[test]
fn a_cds_off_the_menus_is_refused() {
    let params = Params::load_default().expect("parameters");
    let s = searcher(&params, (RICH_POOR, RICH_POOR));
    let mut off = START.to_string();
    off.replace_range(0..3, "UUU");
    assert!(
        s.evaluate(&[off]).is_err(),
        "the first codon must code for M"
    );
    assert!(
        s.evaluate(&[START[..57].to_string()]).is_err(),
        "and the length must match"
    );
}

/// `PROTEIN` spelled with rare synonyms, far below the top weight.
const LOW: &str = "AUGUUUGUAUUUCUAGUACUACUACCGCUAGUAUCGUCGCAAUGUGUAAAUCUAACGACG";

/// A tight design to climb from: the low-weight spelling, repaired under `RICH_POOR`.
fn tight_start(params: &Params) -> String {
    let o = searcher(params, (RICH_POOR, RICH_POOR))
        .run(LOW, 8, 0)
        .expect("search");
    assert!(o.end.is_tight());
    o.end.cds
}

#[test]
fn a_weight_climb_records_only_certified_designs_and_takes_the_best_each_step() {
    let params = Params::load_default().expect("parameters");
    let utrs = (RICH_POOR, RICH_POOR);
    let s = searcher(&params, utrs);
    let start = tight_start(&params);
    let c = s.climb(&start, Toward::Weight, 4).expect("climb");
    assert!(c.steps >= 1, "the fixture climbs");
    assert!(c.met.iter().all(|m| certified(&params, utrs, &m.cds)));
    let mut w = s.menus().weight(&start).unwrap();
    for (step, next) in c.path.iter().enumerate() {
        let here = s.menus().weight(next).unwrap();
        assert!(here > w, "step {step}: the weight rises");
        let best = c
            .met
            .iter()
            .filter(|m| m.step == step)
            .map(|m| m.weight)
            .fold(f64::MIN, f64::max);
        assert_eq!(
            here, best,
            "step {step}: to the highest-weight tight neighbour"
        );
        w = here;
    }
}

#[test]
fn a_slope_climb_lowers_the_scalarized_cost_at_every_step() {
    let params = Params::load_default().expect("parameters");
    let utrs = (RICH_POOR, RICH_POOR);
    let s = searcher(&params, utrs);
    let start = tight_start(&params);
    let mu = 100.0 * 2.0;
    let c = s.climb(&start, Toward::Slope(mu), 4).expect("climb");
    assert!(c.steps >= 1, "the fixture takes a slope step");
    assert!(c.met.iter().all(|m| certified(&params, utrs, &m.cds)));
    let first = s.evaluate(std::slice::from_ref(&start)).unwrap().remove(0);
    let mut here = f64::from(first.constrained) - mu * s.menus().weight(&start).unwrap();
    for (step, next) in c.path.iter().enumerate() {
        let m = c.met.iter().find(|m| &m.cds == next).expect("a met design");
        let value = f64::from(m.constrained) - mu * m.weight;
        assert!(value < here, "step {step}: E − μA falls");
        here = value;
    }
}

#[test]
fn a_start_that_is_not_tight_is_not_climbed() {
    let params = Params::load_default().expect("parameters");
    let c = searcher(&params, (RICH_POOR, RICH_POOR))
        .climb(START, Toward::Weight, 4)
        .expect("climb");
    assert!(!c.start.is_tight());
    assert!(c.met.is_empty() && c.path.is_empty() && c.end == START);
}

#[test]
fn pairs_at_the_ends_never_widen_the_gap_or_report_an_uncertified_design() {
    let params = Params::load_default().expect("parameters");
    let usage = CodonUsage::load_default().expect("codon usage");
    for utrs in [HBB, (RICH_POOR, RICH_POOR)] {
        let menus = Menus::new(PROTEIN, &usage).expect("menus");
        let s = Searcher::new(&params, utrs, menus, 4, Aim::Weight)
            .expect("searcher")
            .with_pairs(4);
        let o = s.run(START, 4, 0).expect("search");
        assert!(o.end.gap() <= o.start.gap());
        assert_eq!(o.end.is_tight(), certified(&params, utrs, &o.end.cds));
    }
}

/// The fixture protein's first 40 residues.
const PROTEIN40: &str = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPD";

/// `PROTEIN40` spelled with rare synonyms.
const LOW40: &str = concat!(
    "AUGUUUGUAUUUCUAGUACUACUACCGCUAGUAUCGUCGCAAUGUGUAAAUCUAACGACG",
    "CGAACGCAACUACCGCCGGCGUAUACGAAUUCGUUUACGCGUGGUGUAUAUUAUCCGGAU",
);

fn searcher40(params: &Params, batch: usize) -> Searcher<'_> {
    let usage = CodonUsage::load_default().expect("codon usage");
    let menus = Menus::new(PROTEIN40, &usage).expect("menus");
    Searcher::new(params, (RICH_POOR, RICH_POOR), menus, 4, Aim::Weight)
        .expect("searcher")
        .with_batch(batch)
}

/// A tight design of `PROTEIN40` to climb from.
fn tight_start40(params: &Params) -> String {
    let o = searcher40(params, 0).run(LOW40, 12, 0).expect("search");
    assert!(o.end.is_tight(), "the fixture repairs to a tight design");
    o.end.cds
}

/// `E − μ·A` of a design.
fn scalarized(constrained: i32, weight: f64, mu: f64) -> f64 {
    f64::from(constrained) - mu * weight
}

/// A tight design a climb met: CDS, weight, constrained energy, step.
type MetRow = (String, f64, i32, usize);

/// A climb that folds every admitted neighbour from scratch and takes the
/// best tight one: every tight design met, and the path.
fn reference_climb(
    s: &Searcher<'_>,
    start: &str,
    toward: Toward,
    steps: usize,
) -> (Vec<MetRow>, Vec<String>) {
    let menus = s.menus();
    let mut cur = s.evaluate(&[start.to_string()]).unwrap().remove(0);
    let (mut met, mut path) = (Vec::new(), Vec::new());
    for step in 0..steps {
        let w = menus.weight(&cur.cds).unwrap();
        let ups: Vec<String> = menus
            .neighbours(&cur.cds)
            .into_iter()
            .filter(|c| match toward {
                Toward::Weight => menus.weight(c).unwrap() > w + 1e-12,
                Toward::Slope(_) => true,
            })
            .collect();
        let mut best: Option<(f64, Evaluated)> = None;
        for e in s.evaluate(&ups).unwrap() {
            if !e.is_tight() {
                continue;
            }
            let ew = menus.weight(&e.cds).unwrap();
            met.push((e.cds.clone(), ew, e.constrained, step));
            let better = match (toward, &best) {
                (Toward::Weight, None) => true,
                (Toward::Weight, Some((bw, b))) => {
                    ew > *bw || (ew == *bw && e.constrained < b.constrained)
                }
                (Toward::Slope(mu), b) => {
                    let v = scalarized(e.constrained, ew, mu);
                    v < scalarized(cur.constrained, w, mu) - 1e-9
                        && b.as_ref()
                            .is_none_or(|(bw, b)| v < scalarized(b.constrained, *bw, mu))
                }
            };
            if better {
                best = Some((ew, e));
            }
        }
        let Some((_, e)) = best else { break };
        path.push(e.cds.clone());
        cur = e;
    }
    (met, path)
}

#[test]
fn a_climb_over_the_whole_neighbourhood_matches_folding_every_neighbour() {
    let params = Params::load_default().expect("parameters");
    let start = tight_start40(&params);
    for toward in [Toward::Weight, Toward::Slope(100.0 * 2.0)] {
        let (met, path) = reference_climb(&searcher40(&params, 0), &start, toward, 4);
        assert!(!path.is_empty(), "{toward:?}: the fixture climbs");
        for batch in [0, usize::MAX, 10_000] {
            let c = searcher40(&params, batch)
                .climb(&start, toward, 4)
                .expect("climb");
            let got: Vec<MetRow> = c
                .met
                .iter()
                .map(|m| (m.cds.clone(), m.weight, m.constrained, m.step))
                .collect();
            assert_eq!(got, met, "{toward:?}, batch {batch}: the same designs met");
            assert_eq!(c.path, path, "{toward:?}, batch {batch}: the same path");
        }
    }
}

#[test]
fn a_batched_climb_meets_only_tight_designs_and_improves_at_every_step() {
    let params = Params::load_default().expect("parameters");
    let start = tight_start40(&params);
    let molecule_mfe = |cds: &str| {
        let molecule = format!("{RICH_POOR}{cds}{RICH_POOR}");
        rna_mfe::fold(&params, molecule.as_bytes()).unwrap().1 .0
    };
    let mu = 100.0 * 2.0;
    let orders = [(false, false), (true, false), (false, true)];
    for (batch, (descend, recall)) in (1..=4).flat_map(|b| orders.map(|o| (b, o))) {
        for toward in [Toward::Weight, Toward::Slope(mu)] {
            let s = searcher40(&params, batch)
                .with_descent(descend)
                .with_recall(recall);
            let c = s.climb(&start, toward, 6).expect("climb");
            assert!(
                c.steps >= 1,
                "batch {batch}, {toward:?}: the fixture climbs"
            );
            for m in &c.met {
                let own = rna_mfe::fold(&params, m.cds.as_bytes()).unwrap().1 .0;
                let constrained = s.flanks() + own;
                assert_eq!(constrained, m.constrained, "the recorded E is the fold's");
                assert_eq!(molecule_mfe(&m.cds), constrained, "E = F by a fresh fold");
            }
            let first = s.evaluate(std::slice::from_ref(&start)).unwrap().remove(0);
            let mut here = (s.menus().weight(&start).unwrap(), first.constrained);
            for (step, next) in c.path.iter().enumerate() {
                let m = c
                    .met
                    .iter()
                    .find(|m| &m.cds == next && m.step == step)
                    .expect("each step moves to a design met in it");
                match toward {
                    Toward::Weight => assert!(m.weight > here.0, "step {step}: weight rises"),
                    Toward::Slope(mu) => assert!(
                        scalarized(m.constrained, m.weight, mu) < scalarized(here.1, here.0, mu),
                        "step {step}: E − μA falls"
                    ),
                }
                here = (m.weight, m.constrained);
            }
        }
    }
}

#[test]
fn a_step_folds_at_most_its_batch_limit() {
    let params = Params::load_default().expect("parameters");
    let start = tight_start40(&params);
    let (batch, limit) = (2, 2);
    let s = searcher40(&params, batch).with_max_batches(limit);
    let mut folded = Vec::new();
    let c = s
        .climb_with(&start, Toward::Slope(100.0 * 2.0), 6, |found| {
            folded.push(found.len());
            Ok(())
        })
        .expect("climb");
    assert!(c.steps >= 1, "the fixture climbs");
    assert!(
        folded.iter().all(|&k| k <= batch * limit),
        "no step meets more than it folds: {folded:?}"
    );
    assert_eq!(folded.len(), c.steps + usize::from(c.steps < 6));
}

/// Every design two changes away from `cds` at two codons whose symbols its
/// own optimum (a fresh fold) pairs, by codon pair, each menu in its order.
fn stem_pair_moves(params: &Params, menus: &Menus, cds: &str) -> Vec<String> {
    let own = rna_mfe::fold(params, cds.as_bytes()).unwrap().0;
    let mut stack = Vec::new();
    let mut codons = std::collections::BTreeSet::new();
    for (j, ch) in own.bytes().enumerate() {
        match ch {
            b'(' => {
                stack.push(j);
            }
            b')' => {
                let i = stack.pop().unwrap();
                if i / 3 != j / 3 {
                    codons.insert((i / 3, j / 3));
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for (p, q) in codons {
        for one in alternatives(menus, cds, p) {
            out.extend(alternatives(menus, &one, q));
        }
    }
    out
}

/// Every design one change at `p` away from `cds`.
fn alternatives(menus: &Menus, cds: &str, p: usize) -> Vec<String> {
    menus
        .neighbours(cds)
        .into_iter()
        .filter(|c| c[..3 * p] == cds[..3 * p] && c[3 * p + 3..] == cds[3 * p + 3..])
        .collect()
}

/// Every design two changes away from `cds` inside its first or last `k`
/// codons, sorted.
fn end_pair_moves(menus: &Menus, cds: &str, k: usize) -> Vec<String> {
    let n = menus.len();
    let mut out = Vec::new();
    for window in [(0..k).collect::<Vec<_>>(), (n - k..n).collect()] {
        for (a, &p) in window.iter().enumerate() {
            for &q in &window[a + 1..] {
                for one in alternatives(menus, cds, p) {
                    out.extend(alternatives(menus, &one, q));
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// [`reference_climb`] over the single changes, then the stem pairs, then
/// the end pairs within `ends` codons (weight-raising ones only for a weight
/// climb), without repeats; each folded from scratch.
fn reference_pair_climb(
    params: &Params,
    s: &Searcher<'_>,
    start: &str,
    toward: Toward,
    (stems, ends): (bool, usize),
    steps: usize,
) -> (Vec<MetRow>, Vec<String>) {
    let menus = s.menus();
    let mut cur = s.evaluate(&[start.to_string()]).unwrap().remove(0);
    let (mut met, mut path) = (Vec::new(), Vec::new());
    for step in 0..steps {
        let w = menus.weight(&cur.cds).unwrap();
        let mut all = menus.neighbours(&cur.cds);
        if stems {
            all.extend(stem_pair_moves(params, menus, &cur.cds));
        }
        if ends > 0 {
            all.extend(end_pair_moves(menus, &cur.cds, ends));
        }
        let mut seen = std::collections::HashSet::new();
        let ups: Vec<String> = all
            .into_iter()
            .filter(|c| match toward {
                Toward::Weight => menus.weight(c).unwrap() > w + 1e-12,
                Toward::Slope(_) => true,
            })
            .filter(|c| seen.insert(c.clone()))
            .collect();
        let mut best: Option<(f64, Evaluated)> = None;
        for e in s.evaluate(&ups).unwrap() {
            if !e.is_tight() {
                continue;
            }
            let ew = menus.weight(&e.cds).unwrap();
            met.push((e.cds.clone(), ew, e.constrained, step));
            let better = match (toward, &best) {
                (Toward::Weight, None) => true,
                (Toward::Weight, Some((bw, b))) => {
                    ew > *bw || (ew == *bw && e.constrained < b.constrained)
                }
                (Toward::Slope(mu), b) => {
                    let v = scalarized(e.constrained, ew, mu);
                    v < scalarized(cur.constrained, w, mu) - 1e-9
                        && b.as_ref()
                            .is_none_or(|(bw, b)| v < scalarized(b.constrained, *bw, mu))
                }
            };
            if better {
                best = Some((ew, e));
            }
        }
        let Some((_, e)) = best else { break };
        path.push(e.cds.clone());
        cur = e;
    }
    (met, path)
}

#[test]
fn a_pair_climb_over_the_whole_neighbourhood_matches_folding_every_neighbour() {
    let params = Params::load_default().expect("parameters");
    let start = tight_start40(&params);
    for (toward, pairs) in [
        (Toward::Weight, (true, 4)),
        (Toward::Slope(100.0), (true, 0)),
        (Toward::Slope(100.0), (false, 4)),
        (Toward::Slope(100.0), (true, 4)),
    ] {
        let (met, path) =
            reference_pair_climb(&params, &searcher40(&params, 0), &start, toward, pairs, 3);
        assert!(!path.is_empty(), "{toward:?} {pairs:?}: the fixture climbs");
        for batch in [0, usize::MAX] {
            let c = searcher40(&params, batch)
                .with_climb_pairs(pairs.0, pairs.1)
                .climb(&start, toward, 3)
                .expect("climb");
            let got: Vec<MetRow> = c
                .met
                .iter()
                .map(|m| (m.cds.clone(), m.weight, m.constrained, m.step))
                .collect();
            assert_eq!(
                got, met,
                "{toward:?} {pairs:?}, batch {batch}: the same designs met"
            );
            assert_eq!(
                c.path, path,
                "{toward:?} {pairs:?}, batch {batch}: the same path"
            );
        }
    }
}

#[test]
fn a_batched_pair_climb_meets_only_tight_designs_and_improves_at_every_step() {
    let params = Params::load_default().expect("parameters");
    let start = tight_start40(&params);
    let molecule_mfe = |cds: &str| {
        let molecule = format!("{RICH_POOR}{cds}{RICH_POOR}");
        rna_mfe::fold(&params, molecule.as_bytes()).unwrap().1 .0
    };
    let mu = 100.0;
    // (stems, end window, estimate order alone)
    let kinds = [
        (true, 0, false),
        (false, 6, false),
        (true, 6, false),
        (false, 0, true),
    ];
    for (batch, (stems, ends, estimate)) in (1..=4).flat_map(|b| kinds.map(|k| (b, k))) {
        for toward in [Toward::Weight, Toward::Slope(mu)] {
            let s = searcher40(&params, batch)
                .with_climb_pairs(stems, ends)
                .with_estimate(estimate)
                .with_max_batches(6);
            let c = s.climb(&start, toward, 4).expect("climb");
            assert!(
                c.steps >= 1,
                "batch {batch}, {toward:?}: the fixture climbs"
            );
            for m in &c.met {
                let own = rna_mfe::fold(&params, m.cds.as_bytes()).unwrap().1 .0;
                let constrained = s.flanks() + own;
                assert_eq!(constrained, m.constrained, "the recorded E is the fold's");
                assert_eq!(molecule_mfe(&m.cds), constrained, "E = F by a fresh fold");
            }
            let first = s.evaluate(std::slice::from_ref(&start)).unwrap().remove(0);
            let mut here = (s.menus().weight(&start).unwrap(), first.constrained);
            for (step, next) in c.path.iter().enumerate() {
                let m = c
                    .met
                    .iter()
                    .find(|m| &m.cds == next && m.step == step)
                    .expect("each step moves to a design met in it");
                match toward {
                    Toward::Weight => assert!(m.weight > here.0, "step {step}: weight rises"),
                    Toward::Slope(mu) => assert!(
                        scalarized(m.constrained, m.weight, mu) < scalarized(here.1, here.0, mu),
                        "step {step}: E − μA falls"
                    ),
                }
                here = (m.weight, m.constrained);
            }
        }
    }
}

/// A tight design of `PROTEIN40` under `RICH_POOR` that no single change
/// improves at `μ = 1` (kcal/mol per unit weight): the end of a single-change
/// slope climb there.
const SINGLE_MINIMUM40: &str = concat!(
    "AUGUUCGUGUUCUUAGUGCUGUUACCACUGGUGAGCAGCCAGUGCGUGAAUCUGACAACA",
    "CGGACACAGUUACCCCCCGCGUAUACGAACUCGUUUACGCGGGGGGUGUACUACCCCGAC",
);

#[test]
fn a_stem_pair_escapes_a_single_change_local_minimum() {
    let params = Params::load_default().expect("parameters");
    let mu = 100.0;
    let toward = Toward::Slope(mu);
    let single = searcher40(&params, 0)
        .climb(SINGLE_MINIMUM40, toward, 2)
        .expect("climb");
    assert!(single.start.is_tight(), "the fixture is tight");
    assert_eq!(single.steps, 0, "no single change lowers E − μA");
    let s = searcher40(&params, 0).with_climb_pairs(true, 0);
    let c = s.climb(SINGLE_MINIMUM40, toward, 1).expect("climb");
    assert_eq!(c.steps, 1, "a stem pair does");
    let to = &c.path[0];
    let changed = (0..PROTEIN40.len())
        .filter(|&p| to[3 * p..3 * p + 3] != SINGLE_MINIMUM40[3 * p..3 * p + 3])
        .count();
    assert_eq!(changed, 2, "the move changes two codons");
    let m = c.met.iter().find(|m| &m.cds == to).expect("met");
    let w0 = s.menus().weight(SINGLE_MINIMUM40).unwrap();
    assert!(scalarized(m.constrained, m.weight, mu) < scalarized(single.start.constrained, w0, mu));
    let molecule = format!("{RICH_POOR}{to}{RICH_POOR}");
    let free = rna_mfe::fold(&params, molecule.as_bytes()).unwrap().1 .0;
    assert_eq!(free, m.constrained, "and lands on a tight design");
    // A batched step, in the estimate's order, reaches an improving pair too.
    let mut folded = 0;
    let b = searcher40(&params, 4)
        .with_climb_pairs(true, 0)
        .climb_with(SINGLE_MINIMUM40, toward, 1, |found| {
            folded += found.len();
            Ok(())
        })
        .expect("climb");
    println!(
        "batched: {} tight met before the move, path {:?}",
        folded, b.path
    );
    assert_eq!(b.steps, 1, "the batched step escapes too");
    let m = b.met.iter().find(|m| m.cds == b.path[0]).expect("met");
    assert!(scalarized(m.constrained, m.weight, mu) < scalarized(single.start.constrained, w0, mu));
}

/// The positions at which two equal-length CDSs spell different codons.
fn changed(a: &str, b: &str) -> Vec<usize> {
    (0..a.len() / 3)
        .filter(|p| a[3 * p..3 * p + 3] != b[3 * p..3 * p + 3])
        .collect()
}

#[test]
fn frozen_end_windows_are_never_re_spelled() {
    let params = Params::load_default().expect("parameters");
    let usage = CodonUsage::load_default().expect("codon usage");
    let n = PROTEIN.len();

    // The windows are taken from the unfrozen search's own changes — the widest
    // pair that still leaves the middle free — so the freeze provably forbids
    // something and the assertion below is not vacuous.
    let free = searcher(&params, HBB).run(START, 8, 0).expect("search");
    let touched = changed(START, &free.end.cds);
    let p = *touched
        .first()
        .expect("the unfrozen search re-spells something");

    // A tail window reaching back to `p`: the position the unfrozen search took
    // is inside it, so the freeze has something to forbid.
    let w3 = n - p;
    let mut menus = Menus::new(PROTEIN, &usage).expect("menus");
    menus
        .freeze_ends(START, 0, w3)
        .expect("the start spells the menus");
    let tail = Searcher::new(&params, HBB, menus, 4, Aim::Weight)
        .expect("searcher")
        .run(START, 8, 0)
        .expect("search");
    assert!(
        changed(START, &tail.end.cds).iter().all(|q| q + w3 < n),
        "no change the frozen search makes lies in the tail window"
    );
    assert!(tail.end.gap() <= tail.start.gap());

    // Overlapping windows freeze every position, so the search cannot move.
    let mut menus = Menus::new(PROTEIN, &usage).expect("menus");
    menus
        .freeze_ends(START, p + 1, n - p)
        .expect("the start spells the menus");
    let both = Searcher::new(&params, HBB, menus, 4, Aim::Weight)
        .expect("searcher")
        .run(START, 8, 0)
        .expect("search");
    assert_eq!(
        both.end.cds, START,
        "two overlapping windows freeze the design"
    );
}
