//! The motif-excluded front against exhaustive enumeration.
//!
//! On molecules small enough to fold every design, the front is the Pareto set
//! of (constrained energy, coding log-adaptation) over the designs that spell
//! no forbidden motif inside the window. The fold must return exactly that set,
//! and each point's sequence must be a motif-free design that scores what the
//! point says.

use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::{aa_to_index, get_codon_index, AMINO_ACIDS};
use derna_lib::ac::{expand_iupac_motif, MotifWindow};
use derna_lib::free_fold::{sequence_constrained_mfe, CdsWindow};
use derna_lib::motif_front::{
    motif_front_tallied, motif_front_walk, motif_front_walk_run, narrowed_resume, MotifFrontSpec,
    MotifResume, MotifRun, MotifWalk, RuleTally,
};
use derna_lib::{
    design_front_walk_with_utr, BeamError, CodonUsage, Enforcement, FrontPoint, Layout, Params,
    Pins, PositionalSpec, RegionConstraints, UtrSpec, WalkBudget, WalkOptions,
};

const EPSILON: f64 = 1.0 / (1u32 << 23) as f64;

/// A cell: protein, flanks, constraint and motif set, forbidden over the
/// coding sequence by the overlapping reading.
#[derive(Clone)]
struct Cell {
    protein: &'static str,
    utr5: &'static str,
    utr3: &'static str,
    constraints: RegionConstraints,
    motifs: &'static [&'static str],
}

impl Cell {
    fn cds(&self) -> CdsWindow {
        CdsWindow::new(self.utr5.len(), self.protein.len())
    }

    fn window(&self) -> MotifWindow {
        let cds = self.cds();
        MotifWindow::overlapping(cds.start as u32, cds.end as u32)
    }

    /// Every motif with the window it is forbidden inside, as a spec takes it:
    /// here one shared window, the whole coding sequence by the overlapping
    /// reading.
    fn placed(&self) -> Vec<(&'static str, MotifWindow)> {
        self.motifs.iter().map(|m| (*m, self.window())).collect()
    }
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

/// Whether `whole` spells a motif occurrence overlapping the coding sequence,
/// read directly off the string.
fn spells_motif(cell: &Cell, whole: &str) -> bool {
    let cds = cell.cds();
    cell.motifs.iter().any(|m| {
        expand_iupac_motif(m).unwrap().iter().any(|pattern| {
            (0..=whole.len().saturating_sub(pattern.len())).any(|s| {
                whole[s..].starts_with(pattern.as_str())
                    && s < cds.end
                    && s + pattern.len() > cds.start
            })
        })
    })
}

fn pareto(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut front: Vec<Point> = Vec::new();
    for p in points {
        if front.last().is_none_or(|q| p.1 > q.1 + 1e-9) {
            front.push(p);
        }
    }
    front
}

fn score(cell: &Cell, params: &Params, usage: &CodonUsage, cds: &str) -> Point {
    let whole = format!("{}{cds}{}", cell.utr5, cell.utr3);
    let e = sequence_constrained_mfe(params, &whole, cell.cds(), cell.constraints).unwrap();
    ((e * 100.0).round() as i64, cai_log(cell, usage, cds))
}

/// The plain front and the motif-free front, by folding every design.
fn exhaustive(cell: &Cell, params: &Params, usage: &CodonUsage) -> (Vec<Point>, Vec<Point>) {
    let mut all = Vec::new();
    let mut free = Vec::new();
    for c in every_design(cell) {
        let p = score(cell, params, usage, &c);
        all.push(p);
        if !spells_motif(cell, &format!("{}{c}{}", cell.utr5, cell.utr3)) {
            free.push(p);
        }
    }
    (pareto(all), pareto(free))
}

fn computed(
    cell: &Cell,
    params: &Params,
    usage: &CodonUsage,
) -> Result<(Vec<FrontPoint>, RuleTally), BeamError> {
    motif_front_tallied(
        params,
        cell.protein,
        usage,
        &MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        },
        None,
    )
}

fn plain_walk(cell: &Cell, params: &Params, usage: &CodonUsage) -> Vec<Point> {
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
    points(&walked.front)
}

fn points(front: &[FrontPoint]) -> Vec<Point> {
    front
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

/// Each returned design is motif-free and scores exactly its point.
fn check_designs(cell: &Cell, params: &Params, usage: &CodonUsage, front: &[FrontPoint]) {
    let (lo, hi) = (cell.utr5.len(), cell.utr5.len() + 3 * cell.protein.len());
    for p in front {
        assert!(p.sequence.starts_with(cell.utr5) && p.sequence.ends_with(cell.utr3));
        assert!(!spells_motif(cell, &p.sequence), "{}", p.sequence);
        let got = score(cell, params, usage, &p.sequence[lo..hi]);
        assert!(
            same(&[got], &points(std::slice::from_ref(p))),
            "{}: scores {got:?}",
            p.sequence
        );
    }
}

/// What a cell was chosen to exercise, measured with [`measure_cells`].
#[derive(Clone, Copy, PartialEq, Debug)]
enum Covers {
    /// The motifs remove points of the plain front.
    RemovesFrontPoints,
    /// The motifs remove designs but no point of the plain front, so the
    /// front equals the plain walk's.
    RemovesNoFrontPoint,
    /// Flanks and a pair constraint, with bulges and internal loops on the
    /// front.
    FlanksAndConstraint,
    /// A multi-loop on the front.
    MultiLoop,
    /// A hairpin whose whole spelling earns a tabulated hexaloop bonus, in
    /// designed positions.
    SpecialHexaloop,
    /// No encoding avoids the motifs.
    Infeasible,
}

/// Whether `structure` closes a 6-base hairpin whose closing pair and loop
/// spell one of the energy table's hexaloops.
fn has_tabulated_hexaloop(sequence: &str, structure: &str) -> bool {
    const HEXALOOPS: [&str; 4] = ["ACAGUACU", "ACAGUGAU", "ACAGUGCU", "ACAGUGUU"];
    let mut open = Vec::new();
    structure.bytes().enumerate().any(|(j, c)| match c {
        b'(' => {
            open.push(j);
            false
        }
        b')' => {
            let i = open.pop().unwrap();
            j - i == 7
                && structure[i + 1..j].bytes().all(|c| c == b'.')
                && HEXALOOPS.contains(&&sequence[i..=j])
        }
        _ => false,
    })
}

/// The partly solved boxes of `walk` as a [`MotifResume`] reads them.
fn rests_of(walk: &MotifWalk) -> Vec<((f64, f64), (f64, f64))> {
    walk.rests
        .iter()
        .map(|r| {
            (
                (r.lo.to_kcal_mol(), r.hi.to_kcal_mol()),
                (r.from.to_kcal_mol(), r.floor),
            )
        })
        .collect()
}

fn cells() -> Vec<(Cell, Covers)> {
    let five = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
    let symmetric = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    let none = RegionConstraints::default();
    vec![
        (
            Cell {
                protein: "MRLGSA",
                utr5: "",
                utr3: "",
                constraints: none,
                motifs: &["GGC"],
            },
            Covers::RemovesFrontPoints,
        ),
        (
            Cell {
                protein: "MCHFYKNDEQ",
                utr5: "GGGAAACCC",
                utr3: "GGGUUUCCC",
                constraints: five,
                motifs: &["AAA"],
            },
            Covers::RemovesNoFrontPoint,
        ),
        (
            Cell {
                protein: "MRLGSA",
                utr5: "GGCAUCCAC",
                utr3: "GUGGAUGCC",
                constraints: symmetric,
                motifs: &["CCA", "UCC"],
            },
            Covers::FlanksAndConstraint,
        ),
        (
            Cell {
                protein: "MPVKW",
                utr5: "GGAACCGAGGAAACCUCG",
                utr3: "GUCAGUUCGCUGACGUUCC",
                constraints: five,
                motifs: &["CCC"],
            },
            Covers::MultiLoop,
        ),
        (
            Cell {
                protein: "MTVF",
                utr5: "CCG",
                utr3: "AUCGG",
                constraints: none,
                motifs: &["ACC"],
            },
            Covers::SpecialHexaloop,
        ),
        (
            Cell {
                protein: "MWWWWM",
                utr5: "",
                utr3: "",
                constraints: none,
                motifs: &["UGG"],
            },
            Covers::Infeasible,
        ),
    ]
}

#[test]
fn the_fold_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    for (cell, covers) in cells() {
        let (plain, truth) = exhaustive(&cell, &params, &usage);
        let name = format!("{} {:?}", cell.protein, cell.motifs);
        let got = match computed(&cell, &params, &usage) {
            Ok(got) => got,
            Err(BeamError::Infeasible) => {
                assert!(truth.is_empty(), "{name}: infeasible, yet designs survive");
                assert_eq!(covers, Covers::Infeasible, "{name}");
                continue;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        let (front, tally) = got;
        check_designs(&cell, &params, &usage, &front);
        assert!(
            same(&points(&front), &truth),
            "{name}:\n got   {:?}\n truth {:?}",
            points(&front),
            truth
        );
        // The cell still covers what it was chosen for.
        match covers {
            Covers::RemovesFrontPoints => assert!(!same(&plain, &truth), "{name}"),
            Covers::RemovesNoFrontPoint => {
                let spelled = every_design(&cell)
                    .iter()
                    .filter(|c| spells_motif(&cell, &format!("{}{c}{}", cell.utr5, cell.utr3)))
                    .count();
                assert!(spelled > 0, "{name}: the motifs remove no design");
                assert!(same(&plain, &truth), "{name}");
                assert!(same(&plain_walk(&cell, &params, &usage), &truth), "{name}");
            }
            Covers::FlanksAndConstraint => {
                assert!(!cell.utr5.is_empty() && !cell.utr3.is_empty(), "{name}");
                assert!(!cell.constraints.is_inert(), "{name}");
                assert!(tally.bulges > 0 && tally.internals > 0, "{name}: {tally:?}");
                // And the constraint binds. A non-inert table forbids nothing
                // on a molecule whose optimal structures never cross a
                // boundary, and such a cell would pass with no table at all:
                // the front it covers has to move when the table is removed.
                let unconstrained = Cell {
                    constraints: RegionConstraints::default(),
                    ..cell.clone()
                };
                let (_, free) = exhaustive(&unconstrained, &params, &usage);
                assert!(
                    !same(&free, &truth),
                    "{name}: the constraint forbids nothing here\n  with    {:?}\n  without {free:?}",
                    truth
                );
            }
            Covers::MultiLoop => assert!(tally.multis > 0, "{name}: {tally:?}"),
            Covers::SpecialHexaloop => assert!(
                front
                    .iter()
                    .any(|p| has_tabulated_hexaloop(&p.sequence, &p.structure)),
                "{name}"
            ),
            Covers::Infeasible => panic!("{name}: expected no motif-free encoding"),
        }
        assert!(
            tally.hairpins + tally.spelled_hairpins > 0 && tally.stacks > 0,
            "{name}: {tally:?}"
        );
    }
}

/// The walk returns the same front as the whole fold: hull vertices from the
/// scalar product folds, each gap from the front fold under its cut.
#[test]
fn the_walk_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut gaps = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let name = format!("{} {:?}", cell.protein, cell.motifs);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let walked = match motif_front_walk(&params, cell.protein, &usage, &spec, 2) {
            Ok(walked) => walked,
            Err(BeamError::Infeasible) => {
                assert!(truth.is_empty(), "{name}");
                continue;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        check_designs(&cell, &params, &usage, &walked.front);
        assert!(
            same(&points(&walked.front), &truth),
            "{name}:\n got   {:?}\n truth {:?}",
            points(&walked.front),
            truth
        );
        gaps += walked.gaps.len();
    }
    assert!(gaps > 5, "the cells walk too few gaps: {gaps}");
}

/// Every checkpoint the walk hands over, resumed, returns the exhaustive
/// front: a checkpoint's points are front points and its open boxes hold all
/// the rest.
#[test]
fn every_checkpoint_resumes_to_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut resumed_open = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let name = format!("{} {:?}", cell.protein, cell.motifs);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let taken = std::sync::Mutex::new(Vec::new());
        let hand = |so_far: &MotifWalk| taken.lock().unwrap().push(so_far.clone());
        let run = MotifRun {
            checkpoint: Some(&hand),
            ..MotifRun::default()
        };
        match motif_front_walk_run(&params, cell.protein, &usage, &spec, 2, run) {
            Ok(_) => {}
            Err(BeamError::Infeasible) => {
                assert!(truth.is_empty(), "{name}");
                continue;
            }
            Err(e) => panic!("{name}: {e}"),
        }
        let taken = taken.into_inner().unwrap();
        // The first is handed out once the ends are folded, before any pair
        // settles: the ends, with the box between them open.
        let first = &taken[0];
        assert!(first.gaps.is_empty() && first.probes == 0, "{name}");
        assert_eq!(first.open.len(), 1, "{name}");
        assert_eq!(
            first.open[0],
            (first.vertices[0], *first.vertices.last().unwrap()),
            "{name}"
        );
        for (k, so_far) in taken.into_iter().enumerate() {
            let vertex = |p: &FrontPoint| {
                so_far.vertices.iter().any(|v| {
                    (v.mfe.to_kcal_mol() - p.mfe_kcal_mol).abs() < 1e-9
                        && (v.cai - p.cai_log).abs() < 1e-9
                })
            };
            let resume = MotifResume {
                points: so_far
                    .front
                    .iter()
                    .map(|p| (p.clone(), vertex(p)))
                    .collect(),
                open: so_far
                    .open
                    .iter()
                    .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
                    .collect(),
                rests: rests_of(&so_far),
                rehull: false,
                moves: Vec::new(),
            };
            resumed_open += resume.open.len();
            let run = MotifRun {
                resume: Some(&resume),
                ..MotifRun::default()
            };
            let resumed =
                motif_front_walk_run(&params, cell.protein, &usage, &spec, 2, run).unwrap();
            if k == 0 {
                // A fresh walk counts its two end folds; this one folded none.
                assert_eq!(resumed.hull_folds, resumed.probes, "{name}");
            }
            check_designs(&cell, &params, &usage, &resumed.front);
            assert!(
                same(&points(&resumed.front), &truth),
                "{name} from {} open:\n got   {:?}\n truth {:?}",
                resume.open.len(),
                points(&resumed.front),
                truth
            );
        }
    }
    assert!(
        resumed_open > 10,
        "the checkpoints leave too little open: {resumed_open}"
    );
}

/// A walk stopped by coverage holds only front points, says so, and leaves
/// open boxes spanning no more than its share of the ends' energy span; at
/// share zero it is the whole front.
#[test]
fn a_walk_stopped_by_coverage_keeps_front_points_and_its_share_open() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut stopped = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let walk = |share| {
            let run = MotifRun {
                stop_open_share: Some(share),
                ..MotifRun::default()
            };
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run)
        };
        let (Ok(half), Ok(whole)) = (walk(0.5), walk(0.0)) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(same(&points(&whole.front), &truth), "{}", cell.protein);
        let on_front = |p: &Point| truth.iter().any(|t| t.0 == p.0 && (t.1 - p.1).abs() < 1e-6);
        assert!(points(&half.front).iter().all(on_front), "{}", cell.protein);
        if half.stopped_by == Some("coverage") {
            stopped += 1;
        } else {
            assert!(half.stopped_by.is_none(), "{}", cell.protein);
        }
    }
    assert!(stopped > 0, "no walk stopped by coverage");
}

/// The walk holding at most three passes without a memory budget returns the
/// same front, and takes some pass again.
#[test]
fn a_walk_holding_three_passes_without_a_budget_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut retaken = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let run = MotifRun {
            resident_passes: Some(3),
            ..MotifRun::default()
        };
        let Ok(walked) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(walked.stopped_by.is_none(), "{}", cell.protein);
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
        let mut weights: Vec<u64> = walked.passes.iter().map(|p| p.lambda.to_bits()).collect();
        weights.sort_unstable();
        weights.dedup();
        retaken += walked.passes.len() - weights.len();
    }
    assert!(retaken > 0, "no pass was dropped and taken again");
}

/// The walk under a memory budget it never reaches returns the same front: it
/// folds the ends one after the other and, past its share of the budget, drops
/// passes no fold is reading; a pass taken again is the same pass.
#[test]
fn a_walk_holding_few_passes_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut retaken = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        // Past a share of nothing the walk drops passes at once; under the
        // default share, a budget it never nears drops none.
        let run = |share| MotifRun {
            budget: WalkBudget {
                max_secs: None,
                max_rss_gb: Some(1e6),
            },
            resident_share: share,
            ..MotifRun::default()
        };
        let walk =
            |share| motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run(share));
        let (Ok(walked), Ok(roomy)) = (walk(Some(0.0)), walk(None)) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(walked.stopped_by.is_none(), "{}", cell.protein);
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
        let again = |w: &MotifWalk| {
            let mut weights: Vec<u64> = w.passes.iter().map(|p| p.lambda.to_bits()).collect();
            weights.sort_unstable();
            weights.dedup();
            w.passes.len() - weights.len()
        };
        retaken += again(&walked);
        assert_eq!(again(&roomy), 0, "{}", cell.protein);
    }
    assert!(retaken > 0, "no pass was dropped and taken again");
}

/// The adapted end folded over the most adapted codons alone is the end the
/// whole lattice gives: some walk takes it that way, and every walk is still
/// the exhaustive front, whose most adapted point it must be.
#[test]
fn the_adapted_end_from_the_most_adapted_codons_is_the_exhaustive_end() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut narrow = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let Ok(walked) =
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, MotifRun::default())
        else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
        narrow += usize::from(walked.adapted_narrow);
    }
    assert!(
        narrow > 0,
        "no walk took its adapted end from the most adapted codons"
    );
}

/// Threads inside the folds change no answer: a walk folding each end, pass,
/// probe and gap on three threads returns the one-thread walk's front, designs
/// included, and both are the exhaustive front. On cells this small the
/// threaded phases may run on one thread, so this checks the result, not the
/// speed.
#[test]
fn a_walk_folding_on_three_threads_is_the_one_thread_walk() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let designs = |front: &[FrontPoint]| -> Vec<(String, String)> {
        front
            .iter()
            .map(|p| (p.sequence.clone(), p.structure.clone()))
            .collect()
    };
    let mut walks = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let walk = |fold_threads| {
            let run = MotifRun {
                fold_threads,
                ..MotifRun::default()
            };
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run)
        };
        let (Ok(one), Ok(three)) = (walk(1), walk(3)) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(same(&points(&one.front), &truth), "{}", cell.protein);
        assert_eq!(points(&three.front), points(&one.front), "{}", cell.protein);
        assert_eq!(
            designs(&three.front),
            designs(&one.front),
            "{}",
            cell.protein
        );
        walks += 1;
    }
    assert!(walks > 0, "no cell walked");
}

/// A gap fold on three threads gives up in the span the one-thread fold does,
/// with as many points stored by the end of each span before it: with every
/// gap forced into energy slices, each gap takes as many folds and finds as
/// many points, and the front and its designs are the same.
#[test]
fn a_sliced_gap_on_three_threads_slices_as_on_one() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut folds = 0;
    let mut gaps = 0;
    for (cell, _) in cells() {
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let walk = |fold_threads| {
            let run = MotifRun {
                gap_budget: Some(20),
                slice_grow: Some(0),
                fold_threads,
                ..MotifRun::default()
            };
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run)
        };
        let (Ok(one), Ok(three)) = (walk(1), walk(3)) else {
            continue;
        };
        let slicing = |walked: &MotifWalk| -> Vec<(u64, usize, usize)> {
            let mut gaps: Vec<_> = walked
                .gaps
                .iter()
                .map(|g| (g.lambda.to_bits(), g.folds, g.points))
                .collect();
            gaps.sort_unstable();
            gaps
        };
        assert_eq!(slicing(&three), slicing(&one), "{}", cell.protein);
        assert_eq!(points(&three.front), points(&one.front), "{}", cell.protein);
        let designs = |w: &MotifWalk| -> Vec<(String, String)> {
            w.front
                .iter()
                .map(|p| (p.sequence.clone(), p.structure.clone()))
                .collect()
        };
        assert_eq!(designs(&three), designs(&one), "{}", cell.protein);
        folds += one.gaps.iter().map(|g| g.folds).sum::<usize>();
        gaps += one.gaps.len();
    }
    assert!(
        folds > 2 * gaps,
        "the budget sliced too little: {folds} folds over {gaps} gaps"
    );
}

/// A probe searches up from a floor on the plain optimum rather than folding
/// once under its pair's own score: some probe certifies under a lower
/// ceiling, and the walk is still the exhaustive front.
#[test]
fn a_probe_certifies_under_a_ceiling_below_its_pairs_score() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (mut probes, mut under) = (0, 0);
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let Ok(walked) =
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, MotifRun::default())
        else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
        assert!(walked.probe_folds >= walked.probes, "{}", cell.protein);
        probes += walked.probes;
        under += walked.probes_under_chord;
    }
    assert!(probes > 0, "no walk probed");
    assert!(
        under > 0,
        "no probe of {probes} certified below its pair's score"
    );
}

/// A probe that folds at its pair's score at once where its doubled first
/// margin reaches that score takes fewer folds, and the walk is still the
/// exhaustive front.
#[test]
fn a_probe_skipping_a_spent_first_fold_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (mut folds, mut skipping_folds) = (0, 0);
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        // A learned first ceiling already spares the fold a skip would, so
        // the skip is measured with learning off.
        let run = |skip| MotifRun {
            probe_skip: Some(skip),
            probe_learn: Some(false),
            ..MotifRun::default()
        };
        let Ok(plain) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run(false))
        else {
            assert!(truth.is_empty());
            continue;
        };
        let skipping =
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run(true)).unwrap();
        assert!(same(&points(&skipping.front), &truth), "{}", cell.protein);
        assert_eq!(skipping.probes, plain.probes, "{}", cell.protein);
        folds += plain.probe_folds;
        skipping_folds += skipping.probe_folds;
    }
    assert!(
        skipping_folds < folds,
        "no probe skipped a fold: {skipping_folds} against {folds}"
    );
}

/// A probe whose ceilings learn from the walk's earlier probes takes other
/// folds, and the walk is still the exhaustive front.
#[test]
fn a_probe_learning_its_ceiling_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (mut folds, mut learning_folds) = (0, 0);
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let run = |learn| MotifRun {
            probe_learn: Some(learn),
            ..MotifRun::default()
        };
        let Ok(plain) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run(false))
        else {
            assert!(truth.is_empty());
            continue;
        };
        let learning =
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run(true)).unwrap();
        assert!(same(&points(&learning.front), &truth), "{}", cell.protein);
        assert_eq!(learning.probes, plain.probes, "{}", cell.protein);
        folds += plain.probe_folds;
        learning_folds += learning.probe_folds;
    }
    assert_ne!(learning_folds, folds, "no probe learned its ceiling");
}

/// Under a residency limit the pass at weight 1, which every box reads and
/// none drops, is taken before any other: taken after a box's bracket, it
/// would drop one of the passes that box is about to read, and the next box
/// near the same weight would take that pass again.
#[test]
fn a_walk_holding_few_passes_takes_the_pass_at_weight_one_first() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut walks = 0;
    for (cell, _) in cells() {
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let run = MotifRun {
            budget: WalkBudget {
                max_secs: None,
                max_rss_gb: Some(1e6),
            },
            ..MotifRun::default()
        };
        let Ok(walked) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run) else {
            continue;
        };
        let Some(first) = walked.passes.first() else {
            continue;
        };
        assert!(
            (first.lambda - 1.0).abs() < 1e-12,
            "{}: the first pass is at {}",
            cell.protein,
            first.lambda
        );
        walks += 1;
    }
    assert!(walks > 0, "no walk took a pass");
}

/// One worker settles boxes from the stable end toward the adapted one: each
/// box it takes has the highest tie weight of those waiting, and the box a new
/// vertex opens above it lies below every box already settled there. So a walk
/// stopped by its budget has spent it on the cheaper, stable end first.
#[test]
fn one_worker_settles_boxes_by_falling_weight() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut checked = 0;
    for (cell, _) in cells() {
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let Ok(walked) = motif_front_walk(&params, cell.protein, &usage, &spec, 1) else {
            continue;
        };
        let mut gaps = walked.gaps.clone();
        gaps.sort_by(|a, b| a.settled_s.total_cmp(&b.settled_s));
        for pair in gaps.windows(2) {
            assert!(
                pair[1].lambda <= pair[0].lambda + 1e-12,
                "{}: settled at λ {} after λ {}",
                cell.protein,
                pair[1].lambda,
                pair[0].lambda
            );
        }
        checked += gaps.len().saturating_sub(1);
    }
    assert!(checked > 5, "too few successive gaps to order: {checked}");
}

/// The constrained-lattice passes change what the probes and gap folds
/// prune and nothing the walk returns.
///
/// Walks with those passes at every grid weight, at the default low weights,
/// and at none give the same points and designs, the exhaustive front where it
/// is known. With them the probes rule out cells and pairs, and the gap folds
/// pairs and points, that no plain floor does; without them both counts are
/// exactly zero.
#[test]
fn aware_passes_prune_probes_and_gaps_and_keep_the_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let designs = |front: &[FrontPoint]| -> Vec<(String, String)> {
        front
            .iter()
            .map(|p| (p.sequence.clone(), p.structure.clone()))
            .collect()
    };
    let longer = Cell {
        protein: "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDK",
        utr5: "",
        utr3: "",
        constraints: RegionConstraints::default(),
        motifs: &["GGGGG", "CCCCC", "GAAUUC", "GGAUCC"],
    };
    let mut cases: Vec<(Cell, bool)> = cells().into_iter().map(|(c, _)| (c, true)).collect();
    cases.push((longer, false));
    let mut ruled_out = [0usize; 3];
    let mut gap_ruled_out = [0usize; 3];
    let mut walks = 0;
    for (cell, exhaustive_known) in cases {
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let walk = |upto: Option<f64>| {
            let run = MotifRun {
                aware_upto: upto,
                ..MotifRun::default()
            };
            motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run)
        };
        let arms = [walk(Some(0.0)), walk(None), walk(Some(1.0))];
        let [Ok(off), Ok(_), Ok(_)] = &arms else {
            assert!(arms.iter().all(Result::is_err), "{}", cell.protein);
            continue;
        };
        if exhaustive_known {
            let (_, truth) = exhaustive(&cell, &params, &usage);
            assert!(same(&points(&off.front), &truth), "{}", cell.protein);
        }
        for (arm, walked) in arms.iter().enumerate() {
            let walked = walked.as_ref().unwrap();
            assert_eq!(
                points(&walked.front),
                points(&off.front),
                "{}",
                cell.protein
            );
            assert_eq!(
                designs(&walked.front),
                designs(&off.front),
                "{}",
                cell.protein
            );
            ruled_out[arm] += walked
                .probe_records
                .iter()
                .map(|r| r.aware_pruned + r.aware_cut)
                .sum::<usize>();
            gap_ruled_out[arm] += walked
                .gaps
                .iter()
                .map(|g| g.aware_out + g.aware_dropped)
                .sum::<usize>();
            assert!(
                walked.gaps.iter().all(|g| g.aware_unkeyed == 0),
                "{}: a gap fold met a node no cell key names",
                cell.protein
            );
        }
        walks += 1;
    }
    eprintln!("cells and pairs only the aware floors ruled out (none, default, every weight): {ruled_out:?}");
    assert!(walks > 1, "too few cells walked: {walks}");
    eprintln!("pairs and points only the aware floors ruled out in gap folds: {gap_ruled_out:?}");
    assert_eq!(ruled_out[0], 0, "a walk without aware passes counted one");
    assert_eq!(
        gap_ruled_out[0], 0,
        "a gap fold without aware passes counted one"
    );
    assert!(
        gap_ruled_out[2] > 0,
        "no aware limit ruled out anything in a gap fold"
    );
    assert!(
        ruled_out[2] > 0,
        "no aware floor ruled out anything the plain floors kept"
    );
}

/// Seeding the sweep with the optima the aware passes certify changes how the
/// walk reaches its hull and nothing it returns.
///
/// Seeded walks, at the default low weights and at every grid weight, on one
/// worker and on two, give the unseeded walk's points, designs and hull
/// vertices. They seed at least one vertex; an unseeded walk seeds none.
#[test]
fn aware_seeds_keep_the_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let designs = |front: &[FrontPoint]| -> Vec<(String, String)> {
        front
            .iter()
            .map(|p| (p.sequence.clone(), p.structure.clone()))
            .collect()
    };
    let mut seeded = 0;
    let mut walks = 0;
    for (cell, _) in cells() {
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let walk = |seeds: bool, upto: Option<f64>, threads: usize| {
            let run = MotifRun {
                aware_upto: upto,
                aware_seeds: seeds,
                ..MotifRun::default()
            };
            motif_front_walk_run(&params, cell.protein, &usage, &spec, threads, run)
        };
        let Ok(plain) = walk(false, None, 1) else {
            continue;
        };
        assert_eq!(plain.seeded, 0, "{}: an unseeded walk seeded", cell.protein);
        for (upto, threads) in [(None, 1), (None, 2), (Some(1.0), 1), (Some(1.0), 2)] {
            let walked = walk(true, upto, threads).unwrap();
            let label = format!("{} (upto {upto:?}, {threads} threads)", cell.protein);
            assert_eq!(points(&walked.front), points(&plain.front), "{label}");
            assert_eq!(designs(&walked.front), designs(&plain.front), "{label}");
            assert_eq!(walked.vertices, plain.vertices, "{label}");
            seeded += walked.seeded;
        }
        walks += 1;
    }
    assert!(walks > 1, "too few cells walked: {walks}");
    assert!(seeded > 0, "no walk seeded a vertex");
}

/// A front narrowed from one under fewer motifs is the front walked from the
/// ends: the same points and hull vertices, each design free of every motif.
/// Each cell's front under its own motifs is narrowed to that set plus one
/// more motif; across the cells and motifs, at least one narrowing keeps
/// points and walks only some stretches, and at least one loses an end and
/// walks from the ends. Each narrowing keeps every wider design the narrower
/// front holds; narrowing from the wider walk stopped half way gives the same
/// front, and so does resuming a narrowed walk stopped half way.
#[test]
fn a_narrowed_front_is_the_walked_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let as_resume = |walk: &MotifWalk| MotifResume {
        points: walk
            .front
            .iter()
            .map(|p| {
                let e = (p.mfe_kcal_mol * 100.0).round() as i32;
                let on_hull = walk.vertices.iter().any(|v| v.mfe.0 == e);
                (p.clone(), on_hull)
            })
            .collect(),
        open: Vec::new(),
        rests: Vec::new(),
        rehull: false,
        moves: Vec::new(),
    };
    let (mut narrowed, mut fell_back) = (0, 0);
    for (cell, _) in cells() {
        let wider_spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let taken = std::sync::Mutex::new(Vec::new());
        let hand = |so_far: &MotifWalk| taken.lock().unwrap().push(so_far.clone());
        let run = MotifRun {
            checkpoint: Some(&hand),
            ..MotifRun::default()
        };
        let Ok(wider) = motif_front_walk_run(&params, cell.protein, &usage, &wider_spec, 1, run)
        else {
            continue;
        };
        let wider = as_resume(&wider);
        // The wider walk stopped half way: its open boxes and rests.
        let taken = taken.into_inner().unwrap();
        let stopped = taken.get(taken.len() / 2).map(|so_far| MotifResume {
            open: so_far
                .open
                .iter()
                .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
                .collect(),
            rests: rests_of(so_far),
            ..as_resume(so_far)
        });
        for extra in ["GCC", "CGG", "AUG", "UCU", "GAG", "CAC"] {
            let motifs: Vec<(&str, MotifWindow)> = cell
                .placed()
                .into_iter()
                .chain([(extra, cell.window())])
                .collect();
            let spec = MotifFrontSpec {
                motifs: &motifs,
                ..wider_spec
            };
            let Ok(walked) = motif_front_walk(&params, cell.protein, &usage, &spec, 1) else {
                continue;
            };
            let label = format!("{} + {extra}", cell.protein);
            match narrowed_resume(&spec, &wider).unwrap() {
                Some(r) => {
                    // Every wider design the narrower front holds is kept.
                    for p in walked
                        .front
                        .iter()
                        .filter(|p| wider.points.iter().any(|(q, _)| q.sequence == p.sequence))
                    {
                        assert!(
                            r.points.iter().any(|(q, _)| q.sequence == p.sequence),
                            "{label}: dropped {}",
                            p.sequence
                        );
                    }
                    if r.points.len() > 2 && r.open.len() < wider.points.len() {
                        narrowed += 1;
                    }
                }
                None => fell_back += 1,
            }
            let taken = std::sync::Mutex::new(Vec::new());
            let hand = |so_far: &MotifWalk| taken.lock().unwrap().push(so_far.clone());
            let run = MotifRun {
                narrow_from: Some(&wider),
                checkpoint: Some(&hand),
                ..MotifRun::default()
            };
            let narrow =
                motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run).unwrap();
            assert!(
                same(&points(&narrow.front), &points(&walked.front)),
                "{label}: points"
            );
            assert_eq!(narrow.vertices, walked.vertices, "{label}: vertices");
            // So is the front narrowed from the wider walk stopped half way.
            if let Some(stopped) = &stopped {
                let run = MotifRun {
                    narrow_from: Some(stopped),
                    ..MotifRun::default()
                };
                let narrow =
                    motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run).unwrap();
                assert!(
                    same(&points(&narrow.front), &points(&walked.front)),
                    "{label}: points from a stopped walk"
                );
                assert_eq!(
                    narrow.vertices, walked.vertices,
                    "{label}: vertices from a stopped walk"
                );
            }
            // A narrowed walk stopped half way resumes to the same front.
            let taken = taken.into_inner().unwrap();
            if let Some(so_far) = taken.get(taken.len() / 2) {
                let vertex = |mfe: f64, cai: f64| {
                    so_far.vertices.iter().any(|v| {
                        (v.mfe.to_kcal_mol() - mfe).abs() < 1e-9 && (v.cai - cai).abs() < 1e-9
                    })
                };
                let open: Vec<_> = so_far
                    .open
                    .iter()
                    .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
                    .collect();
                let resume = MotifResume {
                    points: so_far
                        .front
                        .iter()
                        .map(|p| (p.clone(), vertex(p.mfe_kcal_mol, p.cai_log)))
                        .collect(),
                    rehull: so_far.rehull,
                    open,
                    rests: rests_of(so_far),
                    moves: Vec::new(),
                };
                let run = MotifRun {
                    resume: Some(&resume),
                    ..MotifRun::default()
                };
                let resumed =
                    motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run).unwrap();
                assert!(
                    same(&points(&resumed.front), &points(&walked.front)),
                    "{label}: resumed points"
                );
                assert_eq!(
                    resumed.vertices, walked.vertices,
                    "{label}: resumed vertices"
                );
            }
            let patterns: Vec<&'static str> = motifs.iter().map(|(m, _)| *m).collect();
            let under: &'static [&'static str] = Box::leak(patterns.into_boxed_slice());
            let narrower = Cell {
                motifs: under,
                ..cell.clone()
            };
            check_designs(&narrower, &params, &usage, &narrow.front);
        }
    }
    assert!(
        narrowed > 0,
        "no narrowing kept points and walked part of the front"
    );
    assert!(fell_back > 0, "no narrowing lost an end");
}

/// The walk with every gap forced into energy slices returns the same front:
/// slicing changes only what each fold is asked, never the union. So does
/// the walk that folds the rest of a gap as one range once a slice too narrow
/// to split outgrows the budget, and it folds less. Both walks are checked
/// with the slice sizing firing, and the second with the density carry
/// firing, so neither passes with its lever idle.
#[test]
fn a_sliced_walk_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    // Small enough that every gap wider than the narrowest slice is split.
    let run = MotifRun {
        gap_budget: Some(20),
        slice_grow: Some(0),
        ..MotifRun::default()
    };
    let merging = MotifRun {
        slice_grow: None,
        ..run
    };
    let mut folds = 0;
    let mut merged_folds = 0;
    let mut gaps = 0;
    let (mut sized, mut merged_sized, mut carried, mut idle_carried) = (0, 0, 0, 0);
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let Ok(walked) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 2, run) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
        folds += walked.gaps.iter().map(|g| g.folds).sum::<usize>();
        gaps += walked.gaps.len();
        sized += walked.gaps.iter().map(|g| g.sized).sum::<usize>();
        idle_carried += walked.gaps.iter().map(|g| g.carried).sum::<usize>();
        let merged = motif_front_walk_run(&params, cell.protein, &usage, &spec, 2, merging)
            .expect("the walk that slices succeeded");
        assert!(same(&points(&merged.front), &truth), "{}", cell.protein);
        merged_folds += merged.gaps.iter().map(|g| g.folds).sum::<usize>();
        merged_sized += merged.gaps.iter().map(|g| g.sized).sum::<usize>();
        carried += merged.gaps.iter().map(|g| g.carried).sum::<usize>();
    }
    assert_eq!(
        idle_carried, 0,
        "the carry raised a budget with no slice growth"
    );
    assert!(
        sized > 0 && merged_sized > 0,
        "the slice sizing cut no range: {sized} cuts, {merged_sized} under the carry"
    );
    assert!(carried > 0, "the density carry raised no budget");
    assert!(
        folds > 2 * gaps,
        "the budget sliced too little: {folds} folds over {gaps} gaps"
    );
    assert!(
        merged_folds < folds,
        "no gap was folded again as one range: {merged_folds} folds against {folds}"
    );
}

/// Under a memory budget, a walk whose gaps' first ranges are cut to a
/// width returns the same front.
#[test]
fn a_walk_cutting_gaps_to_a_width_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let run = MotifRun {
        budget: WalkBudget {
            max_secs: None,
            max_rss_gb: Some(1e6),
        },
        gap_budget: Some(20),
        gap_fit: Some(3),
        ..MotifRun::default()
    };
    let (mut folds, mut gaps) = (0, 0);
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let Ok(walked) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 2, run) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(walked.stopped_by.is_none(), "{}", cell.protein);
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
        folds += walked.gaps.iter().map(|g| g.folds).sum::<usize>();
        gaps += walked.gaps.len();
    }
    assert!(
        folds > gaps,
        "no gap was cut: {folds} folds over {gaps} gaps"
    );
}

/// A walk whose slices are also cut by each pass bracketing the gap's weight
/// on its own returns the same front: each such cut is a floor, as the chord
/// is.
#[test]
fn a_walk_cut_by_each_bracketing_pass_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let run = MotifRun {
        gap_budget: Some(20),
        bracket_cuts: Some(true),
        ..MotifRun::default()
    };
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let Ok(walked) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 2, run) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(walked.stopped_by.is_none(), "{}", cell.protein);
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
    }
}

/// A walk stopped after a count of gap folds stops a gap between two of its
/// slices and keeps what the gap solved: its points, and as the open box's
/// rest the energy solved up to and the best adaptation there. Walks resumed
/// from each rest, under the same count, end at the exhaustive front, and a
/// resumed box with a rest runs no probe: its earlier one found no vertex.
#[test]
fn a_walk_resumed_from_gaps_stopped_part_way_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut rests = 0;
    let mut solved_part_way = 0;
    // Gaps solved in more than one fold, and resumes of rests alone.
    let mut sliced = 0;
    let mut rests_alone = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        for limit in [1, 3] {
            // Every gap wider than the narrowest slice is split.
            let run = MotifRun {
                gap_budget: Some(20),
                slice_grow: Some(0),
                gap_folds: Some(limit),
                ..MotifRun::default()
            };
            let Ok(mut walk) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run)
            else {
                assert!(truth.is_empty());
                break;
            };
            let mut rounds = 0;
            while !walk.open.is_empty() {
                assert_eq!(walk.stopped_by, Some("folds"), "{}", cell.protein);
                let on_front =
                    |p: &Point| truth.iter().any(|q| q.0 == p.0 && (q.1 - p.1).abs() < 1e-6);
                assert!(
                    points(&walk.front).iter().all(on_front),
                    "{} holds a point off the front: {:?} against {:?}",
                    cell.protein,
                    points(&walk.front),
                    truth
                );
                rests += walk.rests.len();
                solved_part_way += walk.rests.iter().filter(|r| r.from.0 >= r.lo.0).count();
                let vertex = |p: &FrontPoint| {
                    walk.vertices.iter().any(|v| {
                        (v.mfe.to_kcal_mol() - p.mfe_kcal_mol).abs() < 1e-9
                            && (v.cai - p.cai_log).abs() < 1e-9
                    })
                };
                let resume = MotifResume {
                    points: walk.front.iter().map(|p| (p.clone(), vertex(p))).collect(),
                    open: walk
                        .open
                        .iter()
                        .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
                        .collect(),
                    rests: rests_of(&walk),
                    rehull: false,
                    moves: Vec::new(),
                };
                let resumed = MotifRun {
                    resume: Some(&resume),
                    ..run
                };
                sliced += walk.gaps.iter().filter(|g| g.folds > 1).count();
                walk =
                    motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, resumed).unwrap();
                if resume.rests.len() == resume.open.len() {
                    assert_eq!(walk.probes, 0, "{} probed a box with a rest", cell.protein);
                    rests_alone += 1;
                }
                rounds += 1;
                assert!(rounds < 1000, "{} does not finish", cell.protein);
            }
            sliced += walk.gaps.iter().filter(|g| g.folds > 1).count();
            check_designs(&cell, &params, &usage, &walk.front);
            assert!(same(&points(&walk.front), &truth), "{}", cell.protein);
        }
    }
    assert!(
        solved_part_way > 10,
        "too few gaps stopped part way: {solved_part_way} of {rests} rests"
    );
    assert!(sliced > 0, "no gap was solved in more than one fold");
    assert!(rests_alone > 0, "no resume held only boxes with rests");
}

/// A gap's fold under a per-gap time cap of zero seconds moves on after its
/// first solved slice, keeps its rest and is taken up again behind every other
/// box, on one worker and on three; the completed front is the uncapped walk's
/// and the exhaustive front. Resumed from walks stopped part way, with the
/// boxes carrying a rest queued last, it ends at the same front.
#[test]
fn a_walk_whose_gaps_move_on_under_a_time_cap_is_the_uncapped_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut moved_on = 0;
    let mut resumed_moved_on = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        // Every gap wider than the narrowest slice is split.
        let uncapped = MotifRun {
            gap_budget: Some(20),
            slice_grow: Some(0),
            ..MotifRun::default()
        };
        let Ok(whole) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, uncapped)
        else {
            assert!(truth.is_empty());
            continue;
        };
        assert_eq!(whole.gaps_moved_on, 0, "{}", cell.protein);
        let capped = MotifRun {
            gap_secs: Some(0.0),
            ..uncapped
        };
        for threads in [1, 3] {
            let walk = motif_front_walk_run(&params, cell.protein, &usage, &spec, threads, capped)
                .unwrap();
            assert!(walk.stopped_by.is_none(), "{}", cell.protein);
            assert!(walk.open.is_empty(), "{}", cell.protein);
            assert!(
                same(&points(&walk.front), &points(&whole.front)),
                "{} on {threads} thread(s) under the cap",
                cell.protein
            );
            assert!(same(&points(&walk.front), &truth), "{}", cell.protein);
            moved_on += walk.gaps_moved_on;
        }
        // Resumed under the cap from walks stopped after a count of folds.
        let mut walk = motif_front_walk_run(
            &params,
            cell.protein,
            &usage,
            &spec,
            1,
            MotifRun {
                gap_folds: Some(2),
                ..capped
            },
        )
        .unwrap();
        let mut rounds = 0;
        while !walk.open.is_empty() {
            let vertex = |p: &FrontPoint| {
                walk.vertices.iter().any(|v| {
                    (v.mfe.to_kcal_mol() - p.mfe_kcal_mol).abs() < 1e-9
                        && (v.cai - p.cai_log).abs() < 1e-9
                })
            };
            let resume = MotifResume {
                points: walk.front.iter().map(|p| (p.clone(), vertex(p))).collect(),
                open: walk
                    .open
                    .iter()
                    .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
                    .collect(),
                rests: rests_of(&walk),
                rehull: walk.rehull,
                moves: Vec::new(),
            };
            let resumed = MotifRun {
                resume: Some(&resume),
                gap_folds: Some(2),
                ..capped
            };
            walk = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, resumed).unwrap();
            resumed_moved_on += walk.gaps_moved_on;
            rounds += 1;
            assert!(rounds < 1000, "{} does not finish", cell.protein);
        }
        assert!(
            same(&points(&walk.front), &truth),
            "{} resumed",
            cell.protein
        );
    }
    assert!(moved_on > 0, "no gap moved on under the cap");
    assert!(
        resumed_moved_on > 0,
        "no resumed gap moved on under the cap"
    );
}

/// How many times each of a walk's open boxes moved on, as a resume takes it.
fn moves_of(walk: &MotifWalk) -> Vec<((f64, f64), u32)> {
    walk.moves
        .iter()
        .map(|&(lo, hi, moves)| ((lo.to_kcal_mol(), hi.to_kcal_mol()), moves))
        .collect()
}

/// A resumed walk takes up each box's count of moves under the per-gap time
/// cap where the earlier walk left it: a gap that moved on `m` times before
/// is visited for the `m + 1`-th time, and moving on again records `m + 1`
/// or more. Without the counts every gap starts again from its first visit,
/// so the same round records only its own moves. Carried through every round
/// the walk still ends at the exhaustive front.
#[test]
fn a_resumed_walk_takes_up_each_gaps_count_of_moves() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (mut seeded_boxes, mut moved_again) = (0, 0);
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let capped = MotifRun {
            gap_budget: Some(20),
            slice_grow: Some(0),
            gap_secs: Some(0.0),
            gap_folds: Some(2),
            ..MotifRun::default()
        };
        let Ok(mut walk) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, capped)
        else {
            assert!(truth.is_empty());
            continue;
        };
        let mut rounds = 0;
        while !walk.open.is_empty() {
            let vertex = |p: &FrontPoint| {
                walk.vertices.iter().any(|v| {
                    (v.mfe.to_kcal_mol() - p.mfe_kcal_mol).abs() < 1e-9
                        && (v.cai - p.cai_log).abs() < 1e-9
                })
            };
            let resume = MotifResume {
                points: walk.front.iter().map(|p| (p.clone(), vertex(p))).collect(),
                open: walk
                    .open
                    .iter()
                    .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
                    .collect(),
                rests: rests_of(&walk),
                rehull: walk.rehull,
                moves: moves_of(&walk),
            };
            // Every box listed moved on at least once, each listed once.
            assert!(resume.moves.iter().all(|m| m.1 > 0), "{}", cell.protein);
            let fresh = MotifResume {
                moves: Vec::new(),
                ..resume.clone()
            };
            let run = |resume| MotifRun {
                resume: Some(resume),
                ..capped
            };
            let next = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run(&resume))
                .unwrap();
            let unseeded =
                motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run(&fresh)).unwrap();
            let count = |w: &MotifWalk, key: (f64, f64)| {
                moves_of(w).iter().find(|m| m.0 == key).map_or(0, |m| m.1)
            };
            let still_open = |w: &MotifWalk, key: (f64, f64)| {
                w.open
                    .iter()
                    .any(|(a, b)| (a.mfe.to_kcal_mol(), b.mfe.to_kcal_mol()) == key)
            };
            for &(key, before) in &resume.moves {
                seeded_boxes += 1;
                if !still_open(&next, key) {
                    continue;
                }
                let (after, own) = (count(&next, key), count(&unseeded, key));
                // A count is never lost across a resume.
                assert!(
                    after >= before,
                    "{} {key:?}: {before} -> {after}",
                    cell.protein
                );
                // Where the seeded and the fresh round moved the box on alike,
                // the seeded round's first move was its visit `before + 1`.
                if own > 0 && after == before + own {
                    moved_again += 1;
                }
            }
            walk = next;
            rounds += 1;
            assert!(rounds < 1000, "{} does not finish", cell.protein);
        }
        assert!(walk.moves.is_empty(), "{}", cell.protein);
        assert!(
            same(&points(&walk.front), &truth),
            "{} resumed with its moves",
            cell.protein
        );
    }
    assert!(seeded_boxes > 0, "no resume carried a count of moves");
    assert!(
        moved_again > 0,
        "no resumed gap moved on again after its count"
    );
}

/// The walk against the whole fold where the walk's cuts bite.
///
/// Exhaustive cells are too short for the cuts to prune much, and one cut that
/// dropped derivations it should have kept passed every one of them and 54
/// walk-against-fold cells up to 40 residues, then lost 4 of 240 points at
/// 100. This is that molecule. About 8 minutes in release, almost all of it the
/// whole fold.
#[test]
#[ignore = "whole fold at 100 residues; run with --ignored --release"]
fn the_walk_is_the_whole_fold_at_length() {
    use derna_lib::motif_front::motif_front_exact;
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let protein = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI";
    let (utr5, utr3) = ("GGGAAAUAAGAGAGAAAAGAAGAGU", "UGAUAAUAGGCUGGAGCCUCGGUGG");
    let spec = MotifFrontSpec {
        utr5,
        utr3,
        constraints: RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        motifs: &[
            (
                "GGGGG",
                MotifWindow::overlapping(25, 25 + 3 * protein.len() as u32),
            ),
            (
                "CCCCC",
                MotifWindow::overlapping(25, 25 + 3 * protein.len() as u32),
            ),
        ],
        pins: None,
    };
    let walked = motif_front_walk(&params, protein, &usage, &spec, 4).unwrap();
    let whole = motif_front_exact(&params, protein, &usage, &spec).unwrap();
    assert_eq!(whole.len(), 240, "the whole fold's front moved");
    assert!(same(&points(&walked.front), &points(&whole)));
}

/// Pins on coding positions, 1-based, each fixing one nucleotide.
type CdsPins = &'static [(usize, char)];

/// Cells whose pins take a residue's best codon away, so the lattice the walk
/// folds is shifted by a non-zero offset.
fn pinned_cells() -> Vec<(Cell, CdsPins)> {
    let symmetric = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    let none = RegionConstraints::default();
    vec![
        (
            Cell {
                protein: "MRLGSA",
                utr5: "",
                utr3: "",
                constraints: none,
                motifs: &["GGC"],
            },
            // Leucine's third base to U leaves CUU of its six codons.
            &[(9, 'U')],
        ),
        (
            Cell {
                protein: "MRLGSA",
                utr5: "GGCAUCCAC",
                utr3: "GUGGAUGCC",
                constraints: symmetric,
                motifs: &["CCA", "UCC"],
            },
            // Arginine's first base to A and alanine's third to A.
            &[(4, 'A'), (18, 'A')],
        ),
    ]
}

fn honours(cds: &str, pins: CdsPins) -> bool {
    pins.iter()
        .all(|&(at, letter)| cds.as_bytes()[at - 1] as char == letter)
}

fn resolved_pins(cell: &Cell, pins: CdsPins) -> Pins {
    let mut spec = PositionalSpec::default();
    for &(at, letter) in pins {
        spec.add_pin(&format!("cds:{at}:{letter}")).unwrap();
    }
    let layout = Layout::new(
        cell.utr5.len() as u32,
        cell.protein.len() as u32,
        cell.utr3.len() as u32,
    );
    spec.resolve(&layout).unwrap().pins
}

/// A pinned walk and a pinned fold return the front of the motif-free designs
/// that spell the pins, adaptation included: the restricted lattice's offset
/// is added back to every point.
#[test]
fn a_pinned_front_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    for (cell, pins) in pinned_cells() {
        let name = format!("{} {:?} {pins:?}", cell.protein, cell.motifs);
        let mut admitted = Vec::new();
        for c in every_design(&cell) {
            let whole = format!("{}{c}{}", cell.utr5, cell.utr3);
            if honours(&c, pins) && !spells_motif(&cell, &whole) {
                admitted.push(score(&cell, &params, &usage, &c));
            }
        }
        // The offset is the best adaptation the pins leave, and it must not be
        // zero, or adding it back is untested.
        let best = admitted.iter().map(|p| p.1).fold(f64::MIN, f64::max);
        assert!(
            best < -1e-3,
            "{name}: the pins remove no best codon ({best})"
        );
        let truth = pareto(admitted);
        let (_, unpinned) = exhaustive(&cell, &params, &usage);
        assert!(!same(&truth, &unpinned), "{name}: the pins move no point");
        let resolved = resolved_pins(&cell, pins);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: Some(&resolved),
        };
        let (folded, _) = motif_front_tallied(&params, cell.protein, &usage, &spec, None).unwrap();
        let walked = motif_front_walk(&params, cell.protein, &usage, &spec, 2).unwrap();
        // The vertices are reported in the front's units, offset included.
        for v in &walked.vertices {
            assert!(
                points(&walked.front)
                    .iter()
                    .any(|p| same(&[*p], &[(i64::from(v.mfe.0), v.cai)])),
                "{name}: vertex {v:?} is no front point"
            );
        }
        for front in [&folded, &walked.front] {
            check_designs(&cell, &params, &usage, front);
            let (lo, hi) = (cell.utr5.len(), cell.utr5.len() + 3 * cell.protein.len());
            for p in front.iter() {
                assert!(honours(&p.sequence[lo..hi], pins), "{name}: {}", p.sequence);
            }
            assert!(
                same(&points(front), &truth),
                "{name}:\n got   {:?}\n truth {:?}",
                points(front),
                truth
            );
        }
    }
}

/// A pin that contradicts a fixed flank, or leaves a residue no codon, is
/// refused rather than folded.
#[test]
fn a_contradicting_pin_is_refused() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let layout = Layout::new(3, 2, 3);
    for pin in ["5utr:1:C", "cds:1:C"] {
        let mut spec = PositionalSpec::default();
        spec.add_pin(pin).unwrap();
        let pins = spec.resolve(&layout).unwrap().pins;
        let spec = MotifFrontSpec {
            utr5: "GGA",
            utr3: "UCC",
            constraints: RegionConstraints::default(),
            motifs: &[("AAA", MotifWindow::overlapping(3, 9))],
            pins: Some(&pins),
        };
        let e = motif_front_walk(&params, "MK", &usage, &spec, 1).unwrap_err();
        assert!(matches!(e, BeamError::NoFeasibleDesign(_)), "{pin}: {e}");
    }
}

/// A walk folding each gap first in narrow even ranges, lowest first, returns
/// the exhaustive front: each range runs under its own ceiling and the floor
/// found below it. Some gap must have been folded in more than one range, or
/// the width was never applied.
#[test]
fn a_walk_folding_gaps_in_narrow_ranges_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut ranged = 0;
    for (cell, _) in cells() {
        let (_, truth) = exhaustive(&cell, &params, &usage);
        let spec = MotifFrontSpec {
            utr5: cell.utr5,
            utr3: cell.utr3,
            constraints: cell.constraints,
            motifs: &cell.placed(),
            pins: None,
        };
        let run = MotifRun {
            slice_width: Some(20),
            ..MotifRun::default()
        };
        let Ok(walked) = motif_front_walk_run(&params, cell.protein, &usage, &spec, 1, run) else {
            assert!(truth.is_empty());
            continue;
        };
        assert!(walked.stopped_by.is_none(), "{}", cell.protein);
        assert!(same(&points(&walked.front), &truth), "{}", cell.protein);
        ranged += walked.gaps.iter().filter(|g| g.folds > 1).count();
    }
    assert!(ranged > 0, "no gap was folded in more than one range");
}

/// Every spelling of a flank written in IUPAC: one string per choice of the
/// nucleotides each position allows.
fn flank_spellings(iupac: &str) -> Vec<String> {
    let mut out = vec![String::new()];
    for code in iupac.chars() {
        let letters = expand_iupac_motif(&code.to_string()).unwrap();
        out = out
            .iter()
            .flat_map(|s| letters.iter().map(move |l| format!("{s}{l}")))
            .collect();
    }
    out
}

/// With designed flank positions the walk is the exhaustive front over every
/// spelling of the flanks and every encoding, motif-free over the whole
/// molecule, flanks included. The flanks must matter: some designed spelling
/// spells a motif, and the front differs from the one with the flanks fixed.
#[test]
fn a_walk_with_designed_flanks_is_the_exhaustive_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let (protein, utr5, utr3) = ("MKVL", "NNG", "SN");
    let motifs: &[&str] = &["GCC", "AAU"];
    let constraints = RegionConstraints::default();
    let cds = CdsWindow::new(utr5.len(), protein.len());
    let cell = Cell {
        protein,
        utr5,
        utr3,
        constraints,
        motifs,
    };
    let spells = |whole: &str| {
        motifs.iter().any(|m| {
            expand_iupac_motif(m)
                .unwrap()
                .iter()
                .any(|p| whole.contains(p.as_str()))
        })
    };
    let (mut free, mut all) = (Vec::new(), Vec::new());
    let mut excluded_by_flank = 0;
    for five in flank_spellings(utr5) {
        for three in flank_spellings(utr3) {
            for c in every_design(&cell) {
                let whole = format!("{five}{c}{three}");
                let e = sequence_constrained_mfe(&params, &whole, cds, constraints).unwrap();
                let point = ((e * 100.0).round() as i64, cai_log(&cell, &usage, &c));
                all.push(point);
                if spells(&whole) {
                    excluded_by_flank += usize::from(!spells(&c));
                    continue;
                }
                free.push(point);
            }
        }
    }
    let truth = pareto(free);
    assert!(!truth.is_empty(), "every design spells a motif");
    assert!(excluded_by_flank > 0, "no flank spelling spelled a motif");
    let walk_under = |utr5: &str, utr3: &str, motifs: &[&str]| {
        let placed: Vec<(&str, MotifWindow)> =
            motifs.iter().map(|m| (*m, MotifWindow::all())).collect();
        let spec = MotifFrontSpec {
            utr5,
            utr3,
            constraints,
            motifs: &placed,
            pins: None,
        };
        motif_front_walk(&params, protein, &usage, &spec, 1).unwrap()
    };
    let walk = |utr5: &str, utr3: &str| walk_under(utr5, utr3, motifs);
    let walked = walk(utr5, utr3);
    assert!(walked.stopped_by.is_none());
    assert!(
        same(&points(&walked.front), &truth),
        "{:?}\nagainst\n{:?}",
        points(&walked.front),
        truth
    );
    // Each design is a whole molecule whose flanks spell what they allow.
    for p in &walked.front {
        let s = &p.sequence;
        assert_eq!(s.len(), utr5.len() + 3 * protein.len() + utr3.len());
        assert!(!spells(s), "{s} spells a motif");
    }
    let fixed = walk("GGG", "CA");
    assert!(
        !same(&points(&fixed.front), &truth),
        "the designed positions changed nothing"
    );
    // With no motif the walk takes the plain lattice, designed flanks and all.
    assert!(same(
        &points(&walk_under(utr5, utr3, &[]).front),
        &pareto(all)
    ));
}
