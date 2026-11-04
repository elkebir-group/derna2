//! The modular front against exhaustive enumeration.
//!
//! On molecules small enough to fold every design, the front is the Pareto set
//! of (constrained energy, coding log-adaptation) over the designs whose free
//! MFE equals their constrained energy. A complete run must return exactly
//! that set; a run its budget stopped must certify only points of it, and
//! every point of it it does not return must sit in a region it reported open.

use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::{aa_to_index, get_codon_index, AMINO_ACIDS};
use derna_lib::free_fold::{sequence_constrained_mfe, sequence_gap, CdsWindow};
use derna_lib::modular_front::{
    modular_front, modular_front_resuming, ModularEvent, ModularFront, ModularProgress, OpenBox,
    SearchPolicy, TopSearch,
};
use derna_lib::{
    design_front_walk_with_utr, CodonUsage, Enforcement, Params, RegionConstraints, UtrSpec,
    WalkOptions,
};

const EPSILON: f64 = 1.0 / (1u32 << 23) as f64;

/// A cell: protein, flanks and constraint.
struct Cell {
    protein: &'static str,
    utr5: &'static str,
    utr3: &'static str,
    constraints: RegionConstraints,
}

/// (energy in dekacalories, coding log-adaptation) of a design.
type Point = (i64, f64);

fn every_design(cell: &Cell) -> Vec<String> {
    let mut designs = vec![String::new()];
    for b in cell.protein.bytes() {
        let a = aa_to_index(b as char).unwrap();
        let codons: Vec<String> = AMINO_ACIDS[a]
            .codons
            .iter()
            .map(|c| c.nuc.iter().map(|&n| b"ACGU"[n as usize] as char).collect())
            .collect();
        designs = designs
            .iter()
            .flat_map(|d| codons.iter().map(move |c| format!("{d}{c}")))
            .collect();
    }
    designs
}

fn cai_log(cell: &Cell, usage: &CodonUsage, cds: &str) -> f64 {
    let tables = compute_cai_tables(usage);
    cell.protein
        .bytes()
        .enumerate()
        .map(|(i, b)| {
            let a = aa_to_index(b as char).unwrap();
            let codon: Vec<u8> = cds.as_bytes()[3 * i..3 * i + 3]
                .iter()
                .map(|&c| b"ACGU".iter().position(|&x| x == c).unwrap() as u8)
                .collect();
            tables.log_cai[a][get_codon_index(a, [codon[0], codon[1], codon[2]]).unwrap()]
        })
        .sum()
}

/// The modular front by folding every design.
fn exhaustive(cell: &Cell, params: &Params, usage: &CodonUsage) -> Vec<Point> {
    let cds = CdsWindow {
        start: cell.utr5.len(),
        end: cell.utr5.len() + 3 * cell.protein.len(),
    };
    let mut optimal: Vec<Point> = every_design(cell)
        .into_iter()
        .filter_map(|c| {
            let whole = format!("{}{c}{}", cell.utr5, cell.utr3);
            let gap = sequence_gap(params, &whole, cds, cell.constraints).unwrap();
            gap.is_modular().then(|| {
                let e = sequence_constrained_mfe(params, &whole, cds, cell.constraints).unwrap();
                ((e * 100.0).round() as i64, cai_log(cell, usage, &c))
            })
        })
        .collect();
    optimal.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut front: Vec<Point> = Vec::new();
    for p in optimal {
        if front.last().is_none_or(|q| p.1 > q.1 + 1e-9) {
            front.push(p);
        }
    }
    front
}

fn computed(cell: &Cell, params: &Params, usage: &CodonUsage, seconds: f64) -> ModularFront {
    computed_on(cell, params, usage, seconds, 2)
}

fn computed_on(
    cell: &Cell,
    params: &Params,
    usage: &CodonUsage,
    seconds: f64,
    threads: usize,
) -> ModularFront {
    let walked = design_front_walk_with_utr(
        params,
        cell.protein,
        usage,
        &UtrSpec::Fixed(cell.utr5.into()),
        &UtrSpec::Fixed(cell.utr3.into()),
        cell.constraints,
        EPSILON,
        1.0 - EPSILON,
        WalkOptions::default(),
        1,
    )
    .unwrap();
    modular_front(
        params,
        cell.protein,
        usage,
        cell.utr5,
        cell.utr3,
        cell.constraints,
        &walked.front,
        seconds,
        threads,
    )
    .unwrap()
}

/// On enough threads a round walks its regions side by side; the complete
/// front is still the exhaustive one, and some cell has a round of more than
/// one region, so the side-by-side waves ran.
#[test]
fn regions_walked_side_by_side_still_give_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut wide_round = false;
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage);
        let got = computed_on(&cell, &params, &usage, 600.0, 8);
        assert!(got.complete, "{}", cell.protein);
        assert!(
            same(&points(&got), &truth),
            "{}:\n got   {:?}\n truth {:?}",
            cell.protein,
            points(&got),
            truth
        );
        let walked: Vec<usize> = got
            .stretches
            .iter()
            .filter(|s| s.walked)
            .map(|s| s.round)
            .collect();
        wide_round |= walked
            .iter()
            .any(|r| walked.iter().filter(|q| *q == r).count() > 1);
    }
    assert!(
        wide_round,
        "no round had two regions: the waves went untested"
    );
}

fn points(front: &ModularFront) -> Vec<Point> {
    front
        .points
        .iter()
        .map(|p| ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log))
        .collect()
}

fn same(a: &[Point], b: &[Point]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-6)
}

/// Cells chosen by measuring what each exercises, since a front made of seeds
/// alone never walks a stretch: every point a seed (the first), points only a
/// walk finds beside seeds (the second), no seed so one walk covers the whole
/// range (the third), and no modular design at all (the fourth).
fn cells() -> Vec<Cell> {
    let five = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
    let symmetric = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    vec![
        Cell {
            protein: "MRLGSA",
            utr5: "GGCAUCCAC",
            utr3: "GUGGAUGCC",
            constraints: symmetric,
        },
        Cell {
            protein: "MPRGLA",
            utr5: "CCGCGG",
            utr3: "",
            constraints: five,
        },
        Cell {
            protein: "MLLRGA",
            utr5: "UCCGCGCAG",
            utr3: "",
            constraints: five,
        },
        Cell {
            protein: "MRPGAL",
            utr5: "GGGCGCCCG",
            utr3: "",
            constraints: five,
        },
    ]
}

#[test]
fn a_complete_run_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut shapes = Vec::new();
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage);
        let got = computed(&cell, &params, &usage, 600.0);
        shapes.push((got.seeds, got.points.len()));
        assert!(got.complete, "{}: the run did not complete", cell.protein);
        assert!(got.points.iter().all(|p| p.certified), "{}", cell.protein);
        assert!(
            same(&points(&got), &truth),
            "{}:\n got   {:?}\n truth {:?}",
            cell.protein,
            points(&got),
            truth
        );
    }
    // The cells still cover what they were chosen for.
    assert!(
        shapes.iter().any(|&(seeds, n)| seeds > 0 && n > seeds),
        "{shapes:?}"
    );
    assert!(
        shapes.iter().any(|&(seeds, n)| seeds == 0 && n > 0),
        "{shapes:?}"
    );
    assert!(shapes.iter().any(|&(_, n)| n == 0), "{shapes:?}");
}

#[test]
fn a_stopped_run_certifies_only_front_points_and_leaves_the_rest_open() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage);
        let got = computed(&cell, &params, &usage, 0.0);
        for p in got.points.iter().filter(|p| p.certified) {
            let at = ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log);
            assert!(
                truth.iter().any(|t| same(&[*t], &[at])),
                "{}: certified {at:?} is not on the front {truth:?}",
                cell.protein
            );
        }
        let returned = points(&got);
        for t in &truth {
            if returned.iter().any(|r| same(&[*r], &[*t])) {
                continue;
            }
            let open = got.open.iter().any(|o| o.admits(t.0 as f64 / 100.0, t.1));
            assert!(open, "{}: {t:?} is neither returned nor open", cell.protein);
        }
    }
}

/// What a search reported, owned, in the order it reported it.
#[derive(Debug, Clone)]
enum Settled {
    /// The design, its verdict, its gap, and the forbidden pairs its free fold
    /// reinstated.
    Checked(String, bool, Option<f64>, Vec<(usize, usize)>),
    Found(f64, f64, String),
    Band(f64),
}

/// Run the search to completion from `prior`, keeping every event it reports.
fn resumed(
    cell: &Cell,
    params: &Params,
    usage: &CodonUsage,
    prior: Option<&ModularProgress>,
) -> (ModularFront, Vec<Settled>) {
    let walked = design_front_walk_with_utr(
        params,
        cell.protein,
        usage,
        &UtrSpec::Fixed(cell.utr5.into()),
        &UtrSpec::Fixed(cell.utr3.into()),
        cell.constraints,
        EPSILON,
        1.0 - EPSILON,
        WalkOptions::default(),
        1,
    )
    .unwrap();
    let log = std::sync::Mutex::new(Vec::new());
    let record = |e: ModularEvent<'_>| {
        log.lock().unwrap().push(match e {
            ModularEvent::Checked {
                sequence,
                modular,
                gap_kcal_mol,
                blocking,
            } => Settled::Checked(
                sequence.to_string(),
                modular,
                gap_kcal_mol,
                blocking.to_vec(),
            ),
            ModularEvent::Found {
                mfe_kcal_mol,
                cai_log,
                sequence,
            } => Settled::Found(mfe_kcal_mol, cai_log, sequence.to_string()),
            ModularEvent::Band { down_to } => Settled::Band(down_to),
        });
    };
    let front = modular_front_resuming(
        params,
        cell.protein,
        usage,
        (cell.utr5, cell.utr3),
        cell.constraints,
        (&walked.front, &[]),
        (600.0, 2),
        (prior, Some(&record)),
        (TopSearch::Band, 0.0, SearchPolicy::default()),
    )
    .unwrap();
    (front, log.into_inner().unwrap())
}

/// The progress that `events`, reported in order, leave behind.
fn progress(events: &[Settled]) -> ModularProgress {
    let mut p = ModularProgress::default();
    for e in events {
        match e {
            Settled::Checked(s, so, ..) => {
                p.checked.insert(s.clone(), *so);
            }
            Settled::Found(mfe, cai, s) => p.found.push((*mfe, *cai, s.clone())),
            Settled::Band(d) => p.band_down_to = Some(*d),
        }
    }
    p
}

/// A verdict says why, not only whether: a design this run folded reports its
/// gap, zero exactly when it passed and positive when it did not, and a design
/// that fails reports the forbidden pairs its free fold reinstated — the part a
/// user can act on, since a pair is removed by respelling a codon at one of its
/// coding endpoints.
///
/// The pairs are deliberately not asserted non-empty per design. Free optima
/// are routinely co-optimal, so a design can fail by energy while the recovered
/// structure is one of the clean ones at that energy, and an empty list is then
/// the honest report. What is asserted is the implication that does hold: every
/// named pair crosses the coding boundary, which is what both tables these
/// cells use forbid.
#[test]
fn a_checked_design_reports_its_gap_and_what_blocks_it() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (mut passed, mut failed, mut named) = (0, 0, 0);
    for cell in cells() {
        let cds = cell.utr5.len()..cell.utr5.len() + 3 * cell.protein.len();
        let (_, events) = resumed(&cell, &params, &usage, None);
        for e in &events {
            let Settled::Checked(sequence, modular, gap, blocking) = e else {
                continue;
            };
            let gap = gap.expect("a design this run folded reports its gap");
            if *modular {
                passed += 1;
                assert!(
                    gap.abs() < 1e-9,
                    "{}: a passing design has gap {gap}",
                    cell.protein
                );
                assert!(
                    blocking.is_empty(),
                    "{}: a passing design named {blocking:?}",
                    cell.protein
                );
                continue;
            }
            failed += 1;
            named += usize::from(!blocking.is_empty());
            for &(i, j) in blocking {
                assert!(i < j, "{}: pair ({i}, {j}) is not ascending", cell.protein);
                assert!(
                    j < sequence.len(),
                    "{}: pair ({i}, {j}) is outside a molecule of {}",
                    cell.protein,
                    sequence.len()
                );
                assert!(
                    cds.contains(&i) != cds.contains(&j),
                    "{}: pair ({i}, {j}) does not cross the coding boundary {cds:?}",
                    cell.protein
                );
            }
        }
    }
    // Non-vacuity on each arm: a run whose designs all passed would satisfy
    // every assertion about failures, and the reverse.
    assert!(passed > 0, "no design passed, so that arm tested nothing");
    assert!(failed > 0, "no design failed, so that arm tested nothing");
    assert!(
        named > 0,
        "no failing design named a pair, so no pair was checked"
    );
}

/// A search resumed from any prefix of what a complete one reported — what a
/// kill at that moment leaves — returns the exhaustive front, and checks only
/// the constrained points the prefix had not.
#[test]
fn a_search_resumed_from_any_prefix_of_its_reports_is_the_whole_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut cut_past_checks = false;
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage);
        let (whole, events) = resumed(&cell, &params, &usage, None);
        assert!(same(&points(&whole), &truth), "{}", cell.protein);
        let checks = events
            .iter()
            .filter(|e| matches!(e, Settled::Checked(..)))
            .count();
        for cut in [0, events.len() / 3, 2 * events.len() / 3, events.len()] {
            let prior = progress(&events[..cut]);
            let (again, more) = resumed(&cell, &params, &usage, Some(&prior));
            assert!(again.complete, "{} cut {cut}", cell.protein);
            assert!(
                same(&points(&again), &truth),
                "{} cut {cut}:\n got   {:?}\n truth {:?}",
                cell.protein,
                points(&again),
                truth
            );
            let rechecked = more
                .iter()
                .filter(|e| matches!(e, Settled::Checked(..)))
                .count();
            assert_eq!(
                rechecked,
                checks - prior.checked.len(),
                "{} cut {cut}",
                cell.protein
            );
            cut_past_checks |= cut > checks;
        }
    }
    assert!(
        cut_past_checks,
        "no cut fell after certification: the search's own reports went untested"
    );
}

/// Searching the regions nothing caps by window walks instead of the band
/// still completes to the exhaustive front, on the cell with no seed too.
#[test]
fn window_walks_on_uncapped_regions_give_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage);
        let walked = design_front_walk_with_utr(
            &params,
            cell.protein,
            &usage,
            &UtrSpec::Fixed(cell.utr5.into()),
            &UtrSpec::Fixed(cell.utr3.into()),
            cell.constraints,
            EPSILON,
            1.0 - EPSILON,
            WalkOptions::default(),
            1,
        )
        .unwrap();
        let got = modular_front_resuming(
            &params,
            cell.protein,
            &usage,
            (cell.utr5, cell.utr3),
            cell.constraints,
            (&walked.front, &[]),
            (600.0, 2),
            (None, None),
            (TopSearch::Window, 0.0, SearchPolicy::default()),
        )
        .unwrap();
        assert!(got.complete, "{}", cell.protein);
        assert!(
            same(&points(&got), &truth),
            "{}:\n got   {:?}\n truth {:?}",
            cell.protein,
            points(&got),
            truth
        );
        // No region was handed to the band.
        assert!(
            got.stretches.iter().all(|s| s.designs == 0),
            "{}",
            cell.protein
        );
    }
}

/// A constrained front with every gap between neighbouring hull vertices put
/// back as an open box, its inner points dropped as a walk stopped before that
/// gap would return it, gives the exhaustive front when the search floors on
/// the boxes. Without them the same points give a wrong answer on some cell,
/// which is what the boxes are for.
#[test]
fn a_partial_constrained_front_with_its_boxes_gives_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (mut dropped, mut wrong_without) = (0, 0);
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage);
        let walked = design_front_walk_with_utr(
            &params,
            cell.protein,
            &usage,
            &UtrSpec::Fixed(cell.utr5.into()),
            &UtrSpec::Fixed(cell.utr3.into()),
            cell.constraints,
            EPSILON,
            1.0 - EPSILON,
            WalkOptions::default(),
            1,
        )
        .unwrap();
        let mut vertices = walked.vertices.clone();
        vertices.sort_by(|a, b| a.mfe_kcal_mol.total_cmp(&b.mfe_kcal_mol));
        let (mut front, mut boxes) = (walked.front.clone(), Vec::new());
        for w in vertices.windows(2) {
            let (lo, hi) = (w[0].mfe_kcal_mol, w[1].mfe_kcal_mol);
            let inner = |p: &derna_lib::FrontPoint| {
                p.mfe_kcal_mol > lo + 1e-9 && p.mfe_kcal_mol < hi - 1e-9
            };
            if walked.front.iter().any(inner) {
                front.retain(|p| !inner(p));
                boxes.push(OpenBox {
                    mfe_lo_kcal_mol: lo,
                    cai_log_lo: w[0].adaptation_log,
                    cai_log_hi: w[1].adaptation_log,
                });
            }
        }
        dropped += walked.front.len() - front.len();
        let search = |boxes: &[OpenBox]| {
            modular_front_resuming(
                &params,
                cell.protein,
                &usage,
                (cell.utr5, cell.utr3),
                cell.constraints,
                (&front, boxes),
                (600.0, 2),
                (None, None),
                (TopSearch::Window, 0.0, SearchPolicy::default()),
            )
            .unwrap()
        };
        let got = search(&boxes);
        assert!(got.complete, "{}", cell.protein);
        assert!(
            same(&points(&got), &truth),
            "{}:\n got   {:?}\n truth {:?}",
            cell.protein,
            points(&got),
            truth
        );
        let without = search(&[]);
        if !(without.complete && same(&points(&without), &truth)) {
            wrong_without += 1;
        }
    }
    assert!(dropped > 0, "no cell had a point inside a hull gap");
    assert!(wrong_without > 0, "the boxes decided nothing on any cell");
}

/// A stopped run's open regions report the constrained front's floor beside
/// the walks', it never rises above a front point the region holds, and it is
/// the binding one somewhere: a walk's floor is `last_g + μ·A`, which at a
/// region's lowest adaptation can sit far below anything the lattice spells.
#[test]
fn an_open_region_reports_the_constrained_floor_and_it_binds() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut regions = 0;
    let mut binding = 0;
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage);
        let got = computed(&cell, &params, &usage, 0.0);
        let returned = points(&got);
        for o in &got.open {
            let Some(constrained) = o.mfe_floor_constrained_kcal_mol else {
                continue;
            };
            regions += 1;
            let lo = o.cai_log_lo.unwrap_or(f64::NEG_INFINITY);
            let hi = o.cai_log_hi.unwrap_or(f64::INFINITY);
            for t in truth.iter().filter(|t| t.1 >= lo && t.1 <= hi) {
                if returned.iter().any(|r| same(&[*r], &[*t])) {
                    continue;
                }
                assert!(
                    t.0 as f64 / 100.0 >= constrained - 1e-6,
                    "{}: open floor {constrained} is above front point {t:?}",
                    cell.protein
                );
            }
            let walked = match (o.mfe_floor_kcal_mol, o.cai_log_lo) {
                (Some(f), Some(_)) => f,
                _ => f64::NEG_INFINITY,
            };
            if constrained > walked {
                binding += 1;
            }
        }
    }
    assert!(regions > 0, "no stopped run left a region with a floor");
    assert!(
        binding > 0,
        "the constrained floor bound nothing in {regions} region(s)"
    );
}

/// A round visits an uncapped region before the capped ones. The uncapped
/// region is the only one that can hand the round a cap — which bounds every
/// other region's walk — and it holds the best-adapted designs, so a budget
/// that runs out while it waits behind many narrow windows returns the worse
/// half of the front. A seeded cell at length showed 53 windows walked for
/// 2,022 s, each returning nothing, while the uncapped band got no time at all.
///
/// The uncapped region is the one open above the best-adapted seed, whose
/// stretch is the one with no upper bound.
#[test]
fn a_round_visits_an_uncapped_region_first() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut seen = 0;
    for cell in cells() {
        let got = computed(&cell, &params, &usage, 20.0);
        for round in 0..3 {
            let walked: Vec<&_> = got
                .stretches
                .iter()
                .filter(|s| s.round == round && s.walked)
                .collect();
            if walked.len() < 2 || !walked.iter().any(|s| s.cai_log_hi.is_none()) {
                continue;
            }
            seen += 1;
            assert!(
                walked[0].cai_log_hi.is_none(),
                "{}: round {round} walked {:?} before the uncapped region",
                cell.protein,
                walked[0].cai_log_hi
            );
        }
    }
    assert!(
        seen > 0,
        "no round had an uncapped region beside a capped one, so the order went untested"
    );
}

/// Every design of `cell` as (energy in dekacalories, coding log-adaptation,
/// gap in kcal/mol, whole molecule).
fn every_gap(cell: &Cell, params: &Params, usage: &CodonUsage) -> Vec<(i64, f64, f64, String)> {
    let cds = CdsWindow {
        start: cell.utr5.len(),
        end: cell.utr5.len() + 3 * cell.protein.len(),
    };
    every_design(cell)
        .into_iter()
        .map(|c| {
            let whole = format!("{}{c}{}", cell.utr5, cell.utr3);
            let gap = sequence_gap(params, &whole, cds, cell.constraints).unwrap();
            let e = sequence_constrained_mfe(params, &whole, cds, cell.constraints).unwrap();
            let energy = (e * 100.0).round() as i64;
            (energy, cai_log(cell, usage, &c), gap.gap_kcal_mol, whole)
        })
        .collect()
}

/// The Pareto front of the designs whose gap is at most `epsilon`.
fn front_within(designs: &[(i64, f64, f64, String)], epsilon: f64) -> Vec<Point> {
    let mut kept: Vec<Point> = designs
        .iter()
        .filter(|d| d.2 <= epsilon + 1e-9)
        .map(|d| (d.0, d.1))
        .collect();
    kept.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut front: Vec<Point> = Vec::new();
    for p in kept {
        if front.last().is_none_or(|q| p.1 > q.1 + 1e-9) {
            front.push(p);
        }
    }
    front
}

fn computed_within(
    cell: &Cell,
    params: &Params,
    usage: &CodonUsage,
    (top, epsilon): (TopSearch, f64),
    prior: Option<&ModularProgress>,
) -> Result<ModularFront, derna_lib::modular_front::ModularFrontError> {
    computed_with(
        cell,
        params,
        usage,
        (top, epsilon, SearchPolicy::default()),
        prior,
    )
}

fn computed_with(
    cell: &Cell,
    params: &Params,
    usage: &CodonUsage,
    (top, epsilon, policy): (TopSearch, f64, SearchPolicy),
    prior: Option<&ModularProgress>,
) -> Result<ModularFront, derna_lib::modular_front::ModularFrontError> {
    let walked = design_front_walk_with_utr(
        params,
        cell.protein,
        usage,
        &UtrSpec::Fixed(cell.utr5.into()),
        &UtrSpec::Fixed(cell.utr3.into()),
        cell.constraints,
        EPSILON,
        1.0 - EPSILON,
        WalkOptions::default(),
        1,
    )
    .unwrap();
    modular_front_resuming(
        params,
        cell.protein,
        usage,
        (cell.utr5, cell.utr3),
        cell.constraints,
        (&walked.front, &[]),
        (600.0, 2),
        (prior, None),
        (top, epsilon, policy),
    )
}

/// Within an epsilon, a complete run is the exhaustive front of the designs
/// whose gap is at most it, with either way of visiting an uncapped region.
///
/// Each side has a non-vacuity arm, since a run that admitted nothing passes
/// every assertion about what it admitted and one that rejected nothing every
/// assertion about rejection. Across the cells some point returned has a gap
/// strictly between zero and epsilon, and some design with a gap above
/// epsilon is undominated by everything returned: admitted, it would have
/// changed the answer. On the cell with no modular design
/// (`MRPGAL`, least gap 0.70) the front is empty at 0.5 and holds only
/// designs that are not modular at 1.0.
#[test]
fn a_run_within_epsilon_is_the_exhaustive_front_within_it() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (mut admitted, mut rejected) = (Vec::new(), Vec::new());
    for cell in cells() {
        let designs = every_gap(&cell, &params, &usage);
        let gap_of = |seq: &str| designs.iter().find(|d| d.3 == seq).unwrap().2;
        for epsilon in [0.1, 0.5, 1.0, 2.5] {
            let truth = front_within(&designs, epsilon);
            for top in [TopSearch::Band, TopSearch::Window] {
                let got = computed_within(&cell, &params, &usage, (top, epsilon), None).unwrap();
                let at = format!("{} within {epsilon} by {top:?}", cell.protein);
                assert!(got.complete, "{at}");
                assert_eq!(got.epsilon, epsilon, "{at}");
                assert!(got.points.iter().all(|p| p.certified), "{at}");
                assert!(
                    same(&points(&got), &truth),
                    "{at}:\n got   {:?}\n truth {:?}",
                    points(&got),
                    truth
                );
                for p in &got.points {
                    let gap = gap_of(&p.sequence);
                    assert!(gap <= epsilon + 1e-9, "{at}: {} has gap {gap}", p.sequence);
                    if gap > 1e-9 {
                        admitted.push(at.clone());
                    }
                }
                let undominated = |d: &&(i64, f64, f64, String)| {
                    !got.points.iter().any(|p| {
                        (p.mfe_kcal_mol * 100.0).round() as i64 <= d.0 && p.cai_log >= d.1 - 1e-9
                    })
                };
                if designs
                    .iter()
                    .filter(|d| d.2 > epsilon + 1e-9)
                    .any(|d| undominated(&d))
                {
                    rejected.push(at.clone());
                }
            }
        }
    }
    assert!(
        !admitted.is_empty(),
        "no point within epsilon was not tight"
    );
    assert!(
        !rejected.is_empty(),
        "no design above epsilon would have counted"
    );

    let cell = &cells()[3];
    let none = computed_within(cell, &params, &usage, (TopSearch::Band, 0.5), None).unwrap();
    assert!(none.complete && none.points.is_empty());
    let some = computed_within(cell, &params, &usage, (TopSearch::Band, 1.0), None).unwrap();
    assert!(!some.points.is_empty());
    let exact = computed_within(cell, &params, &usage, (TopSearch::Band, 0.0), None).unwrap();
    assert!(exact.complete && exact.points.is_empty());
}

/// At zero the search is the modular one: the same points, seeds
/// and stretches as the entry point that takes no epsilon.
#[test]
fn epsilon_zero_is_the_modular_search() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    for cell in cells() {
        let plain = computed(&cell, &params, &usage, 600.0);
        let zero = computed_within(&cell, &params, &usage, (TopSearch::Band, 0.0), None).unwrap();
        let key = |f: &ModularFront| -> Vec<(String, i64, bool)> {
            f.points
                .iter()
                .map(|p| {
                    let e = (p.mfe_kcal_mol * 100.0).round() as i64;
                    (p.sequence.clone(), e, p.certified)
                })
                .collect()
        };
        assert_eq!(key(&plain), key(&zero), "{}", cell.protein);
        assert_eq!(
            (
                plain.seeds,
                plain.unchecked,
                plain.complete,
                plain.open.len()
            ),
            (zero.seeds, zero.unchecked, zero.complete, zero.open.len()),
            "{}",
            cell.protein
        );
        assert_eq!(zero.epsilon, 0.0);
    }
}

/// Progress settled within one epsilon is refused by a search within
/// another: its verdicts and finds belong to another set and would cap this
/// one unsoundly.
#[test]
fn progress_from_another_epsilon_is_refused() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let cell = &cells()[1];
    let within = ModularProgress {
        epsilon: 0.5,
        ..ModularProgress::default()
    };
    let refused = computed_within(cell, &params, &usage, (TopSearch::Band, 0.0), Some(&within));
    assert!(
        matches!(
            refused,
            Err(derna_lib::modular_front::ModularFrontError::OtherEpsilon { .. })
        ),
        "{refused:?}"
    );
    let same = computed_within(cell, &params, &usage, (TopSearch::Band, 0.5), Some(&within));
    assert!(same.is_ok());
}

/// A search that has found nothing when its rounds are over stops there,
/// incomplete and empty; one that has found a point is not stopped, and
/// holding back barren regions leaves a complete front as it was.
///
/// The stop is asked for at round 0, before any walk, since these cells
/// complete within their first round. The non-vacuity arm is the cell with no
/// modular design (`MRPGAL`): without the stop its search walks a
/// stretch and completes, so a stop that never fired would fail the `gave_up`
/// assertion. A cell whose points all come from its walk would lose them, so
/// only seeded cells are compared whole.
#[test]
fn an_empty_search_gives_up_and_a_search_with_a_point_does_not() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let give_up = SearchPolicy {
        give_up_empty: Some((0, 0)),
        hold_barren: false,
    };
    let hold = SearchPolicy {
        give_up_empty: None,
        hold_barren: true,
    };
    // Compared on objectives, not spellings: a point with co-optimal
    // spellings may be represented by either, depending on the schedule.
    let key = |f: &ModularFront| -> (Vec<Point>, Vec<bool>) {
        (points(f), f.points.iter().map(|p| p.certified).collect())
    };
    let same_key =
        |a: &(Vec<Point>, Vec<bool>), b: &(Vec<Point>, Vec<bool>)| same(&a.0, &b.0) && a.1 == b.1;
    let mut gave_up = 0;
    for cell in cells() {
        let plain =
            computed_within(&cell, &params, &usage, (TopSearch::Window, 0.0), None).unwrap();
        let stopped = computed_with(
            &cell,
            &params,
            &usage,
            (TopSearch::Window, 0.0, give_up),
            None,
        )
        .unwrap();
        let held =
            computed_with(&cell, &params, &usage, (TopSearch::Window, 0.0, hold), None).unwrap();
        assert!(plain.complete, "{}", cell.protein);
        assert!(same_key(&key(&held), &key(&plain)), "{} held", cell.protein);
        assert!(held.gave_up.is_none());
        if plain.points.is_empty() && plain.stretches.iter().any(|s| s.walked) {
            assert_eq!(stopped.gave_up.map(|g| g.0), Some(0), "{}", cell.protein);
            assert!(
                !stopped.complete && stopped.points.is_empty(),
                "{}",
                cell.protein
            );
            gave_up += 1;
        } else if plain.seeds > 0 {
            assert!(same_key(&key(&stopped), &key(&plain)), "{}", cell.protein);
            assert!(stopped.gave_up.is_none(), "{}", cell.protein);
        }
    }
    assert!(gave_up > 0, "no cell exercised the stop");
}
