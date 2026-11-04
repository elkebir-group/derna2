//! The front walk's budget: a walk that runs out of time or memory returns
//! the front points it has settled and names the boxes it has not.
//!
//! What a budget may not do is return a point that is not on the front. The
//! walk solves each gap exactly, so the points of a stopped walk are a subset
//! of the whole walk's; the open boxes are what the subset may be missing, and
//! each names two hull points the walk already holds.

mod common;

use derna_lib::{
    design_front_walk_run, CodonUsage, Params, PassGrid, RegionConstraints, UtrSpec, WalkBudget,
    WalkOptions, WalkResume, WalkRun,
};

/// A protein whose walk settles several gaps, so a budget has something to
/// leave open.
const PROTEIN: &str = "MYGKIIFVLLL";
const FLANK5: &str = "GCAUUG";
const FLANK3: &str = "AUCGA";

fn fixed(s: &str) -> UtrSpec {
    UtrSpec::Fixed(s.to_string())
}

/// The walk `derna front` runs, under `budget` and `run`.
fn walk_run(budget: WalkBudget, run: WalkRun<'_>) -> derna_lib::WalkedFront {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let options = WalkOptions {
        pass_grid: Some(PassGrid::for_residues(PROTEIN.len())),
        seed_sweep: true,
        cut_probes: true,
        budget,
        ..WalkOptions::default()
    };
    design_front_walk_run(
        &params,
        PROTEIN,
        &usage,
        &fixed(FLANK5),
        &fixed(FLANK3),
        RegionConstraints::default(),
        1e-4,
        0.999,
        options,
        2,
        run,
    )
    .expect("the walk answers under a budget")
}

/// The walk under `budget`, from the start.
fn walk(budget: WalkBudget) -> derna_lib::WalkedFront {
    walk_run(budget, WalkRun::default())
}

/// The same front without the pass grid or the seeded sweep, so the walk
/// starts from its two ends and refines each pair by halving it.
///
/// The option set `derna front` runs seeds the sweep from a grid of optima,
/// and at this size every pair it hands out is already adjacent: no probe
/// splits, and the walk never reaches the code that replaces a pair by the two
/// it refines into. This set does reach it.
fn bisecting(budget: WalkBudget, run: WalkRun<'_>) -> derna_lib::WalkedFront {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let options = WalkOptions {
        budget,
        ..WalkOptions::default()
    };
    design_front_walk_run(
        &params,
        PROTEIN,
        &usage,
        &fixed(FLANK5),
        &fixed(FLANK3),
        RegionConstraints::default(),
        1e-4,
        0.999,
        options,
        2,
        run,
    )
    .expect("the bisecting walk answers under a budget")
}

/// What a walk leaves to resume from: its points with their hull flags, and
/// its open boxes, as the command's result file carries them.
fn to_resume(walked: &derna_lib::WalkedFront) -> WalkResume {
    let on_hull = |p: &derna_lib::FrontPoint| {
        walked.vertices.iter().any(|v| {
            v.mfe_kcal_mol == p.mfe_kcal_mol && (v.adaptation_log - p.cai_log).abs() < 1e-9
        })
    };
    WalkResume {
        points: walked
            .front
            .iter()
            .map(|p| (p.clone(), on_hull(p)))
            .collect(),
        open: walked
            .open
            .iter()
            .map(|(a, b)| {
                (
                    (a.mfe_kcal_mol, a.adaptation_log),
                    (b.mfe_kcal_mol, b.adaptation_log),
                )
            })
            .collect(),
        rests: walked
            .rests
            .iter()
            .map(|r| {
                (
                    (r.lo.to_kcal_mol(), r.hi.to_kcal_mol()),
                    (r.from.to_kcal_mol(), r.floor),
                )
            })
            .collect(),
    }
}

#[test]
fn a_budget_returns_front_points_and_names_what_is_open() {
    let whole = walk(WalkBudget::default());
    assert!(whole.stopped_by.is_none(), "{:?}", whole.stopped_by);
    assert!(whole.open.is_empty(), "{:?}", whole.open);
    // A whole front of one gap would leave a budget nothing to stop short of.
    assert!(whole.gaps.len() >= 2, "{} gap(s)", whole.gaps.len());

    let stopped = walk(WalkBudget {
        max_secs: Some(1e-6),
        ..WalkBudget::default()
    });
    assert_eq!(stopped.stopped_by, Some("time"));
    assert!(!stopped.open.is_empty());
    assert!(stopped.gaps.len() < whole.gaps.len());
    assert!(stopped.front.len() < whole.front.len());
    // No probe ran: a budget already spent when the ends finish skips the
    // sweep's first pair, so the walk refines nothing and every box it names
    // is a pair the seeds handed out.
    assert_eq!(stopped.rounds, 0, "{} round(s)", stopped.rounds);
    assert!(stopped.gaps.is_empty(), "{} gap(s)", stopped.gaps.len());

    // Every point the stopped walk returned is a point of the whole front.
    // A budget this small stops the walk at its first probe, so the check is
    // over however many gaps the queue had already handed out; what it rules
    // out is a partial walk reporting a point the whole walk does not have.
    let whole_points = common::front_objectives(&whole.front);
    for point in common::front_objectives(&stopped.front) {
        assert!(
            whole_points
                .iter()
                .any(|q| q.mfe == point.mfe && (q.cai_log - point.cai_log).abs() < 1e-9),
            "{point:?} is not a point of the whole front"
        );
    }

    // Each open box lies between two of the walk's own hull points, the more
    // stable first: that pair is what a later run would settle.
    for (a, b) in &stopped.open {
        assert!(a.mfe_kcal_mol < b.mfe_kcal_mol, "{a:?} is not below {b:?}");
        for end in [a, b] {
            assert!(
                stopped
                    .vertices
                    .iter()
                    .any(|v| v.mfe_kcal_mol == end.mfe_kcal_mol),
                "{end:?} is no vertex of the walk that left it open"
            );
        }
    }

    // Each box's stable end is a hull point folded exactly, so the lower of
    // the stopped front's least energy and the boxes' least is the whole
    // front's least energy.
    let least = |es: &mut dyn Iterator<Item = i32>| es.min().unwrap_or(i32::MAX);
    let whole_lo = least(&mut whole_points.iter().map(|p| p.mfe));
    let point_lo = least(
        &mut common::front_objectives(&stopped.front)
            .iter()
            .map(|p| p.mfe),
    );
    let box_lo = least(
        &mut stopped
            .open
            .iter()
            .map(|(a, _)| (a.mfe_kcal_mol * 100.0).round() as i32),
    );
    assert_eq!(point_lo.min(box_lo), whole_lo);
    // Without this the check passes on any stop that kept the stable end.
    assert!(point_lo > whole_lo, "the open boxes were not needed");
}

/// A memory cap below what the process already holds stops the walk too, and
/// says so: the two budgets are read by the same check and reported apart.
#[test]
fn a_memory_cap_stops_the_walk_by_memory() {
    let stopped = walk(WalkBudget {
        max_rss_gb: Some(1e-9),
        ..WalkBudget::default()
    });
    assert_eq!(stopped.stopped_by, Some("memory"));
    assert!(!stopped.open.is_empty());
}

/// A walk resumed from a stopped one returns the whole front: the boxes the
/// budget left are folded, the points already found are kept, and the two sets
/// together are the front the unbudgeted walk returns, point for point.
///
/// The hull a resumed walk is handed is the earlier walk's, so its ends are
/// not folded again — which is the saving — and a box resumed twice reads the
/// same two hull points.
#[test]
fn a_resumed_walk_returns_the_whole_front() {
    let whole = walk(WalkBudget::default());
    let stopped = walk(WalkBudget {
        max_secs: Some(1e-6),
        ..WalkBudget::default()
    });
    // A resume from a walk that settled everything would test nothing.
    assert!(!stopped.open.is_empty());
    assert!(stopped.front.len() < whole.front.len());

    let resume = to_resume(&stopped);
    let resumed = walk_run(
        WalkBudget::default(),
        WalkRun {
            resume: Some(&resume),
            ..WalkRun::default()
        },
    );
    assert!(resumed.stopped_by.is_none(), "{:?}", resumed.stopped_by);
    assert!(resumed.open.is_empty(), "{:?}", resumed.open);
    // No end was folded: the hull came from the file.
    assert_eq!(resumed.sweep_scans, stopped.open.len());
    // And the hull it reports is the whole walk's, the ends it was handed
    // among them, so a caller reading `on_hull` from a resumed front sees the
    // same vertices as from a front walked in one run.
    let hull = |w: &derna_lib::WalkedFront| {
        let mut energies: Vec<i64> = w
            .vertices
            .iter()
            .map(|v| (v.mfe_kcal_mol * 100.0).round() as i64)
            .collect();
        energies.sort_unstable();
        energies
    };
    assert_eq!(hull(&resumed), hull(&whole));

    common::assert_same(
        "resumed against whole",
        &common::front_objectives(&whole.front),
        &common::front_objectives(&resumed.front),
    );
}

/// Resuming a front with nothing open settles nothing and returns it
/// unchanged: the walk has no box to fold.
#[test]
fn a_resume_with_nothing_open_returns_what_it_was_given() {
    let whole = walk(WalkBudget::default());
    let resume = to_resume(&whole);
    assert!(resume.open.is_empty());
    let again = walk_run(
        WalkBudget::default(),
        WalkRun {
            resume: Some(&resume),
            ..WalkRun::default()
        },
    );
    assert!(again.gaps.is_empty(), "{} gap(s)", again.gaps.len());
    common::assert_same(
        "resumed from a whole front",
        &common::front_objectives(&whole.front),
        &common::front_objectives(&again.front),
    );
}

/// A resume that is handed points as well as boxes keeps them: the front it
/// returns is the whole front, the settled gap's interior points included.
///
/// A budget small enough to stop the walk deterministically stops it before
/// any gap settles, so that resume carries no points and cannot see them
/// dropped. This one is built from the whole walk instead — the first gap
/// settled, the rest left open — which is the shape of a walk stopped part
/// way and is the same every run.
#[test]
fn a_resume_keeps_the_points_it_was_given() {
    let whole = walk(WalkBudget::default());
    assert!(whole.gaps.len() >= 2, "{} gap(s)", whole.gaps.len());
    let (first, rest) = whole.gaps.split_first().expect("a walk with gaps");
    // The settled gap's own interior points: what only the resume can supply,
    // since the boxes left open do not cover them.
    let interior = first
        .front
        .iter()
        .filter(|p| p.mfe_kcal_mol > first.a.mfe_kcal_mol && p.mfe_kcal_mol < first.b.mfe_kcal_mol)
        .count();
    assert!(interior >= 1, "the first gap holds no interior point");
    let on_hull = |p: &derna_lib::FrontPoint| {
        [first.a, first.b]
            .iter()
            .any(|v| v.mfe_kcal_mol == p.mfe_kcal_mol)
    };
    let resume = WalkResume {
        points: first
            .front
            .iter()
            .map(|p| (p.clone(), on_hull(p)))
            .collect(),
        open: rest
            .iter()
            .map(|g| {
                (
                    (g.a.mfe_kcal_mol, g.a.adaptation_log),
                    (g.b.mfe_kcal_mol, g.b.adaptation_log),
                )
            })
            .collect(),

        rests: Vec::new(),
    };
    let resumed = walk_run(
        WalkBudget::default(),
        WalkRun {
            resume: Some(&resume),
            ..WalkRun::default()
        },
    );
    assert_eq!(resumed.gaps.len(), rest.len());
    // The front ascends in energy whichever walk found each point: a resumed
    // walk's own points lie inside the boxes the earlier one left, so the two
    // sets interleave rather than follow one another.
    for pair in resumed.front.windows(2) {
        assert!(
            pair[0].mfe_kcal_mol <= pair[1].mfe_kcal_mol,
            "{:?} comes before {:?}",
            pair[0].mfe_kcal_mol,
            pair[1].mfe_kcal_mol
        );
    }
    common::assert_same(
        "resumed from one settled gap",
        &common::front_objectives(&whole.front),
        &common::front_objectives(&resumed.front),
    );
}

/// Every energy range the walk reports, settled or open, is disjoint from
/// every other.
///
/// The walk hands out a pair, and either solves it as a gap or refines it into
/// two pairs that cover the same range. So at any moment the ranges it has
/// settled and the ranges it still owes tile parts of the hull without
/// overlapping: a range reported open while a gap inside it was solved means
/// the walk lost track of what it had settled, and a resume would fold that
/// range again and merge two answers for it.
///
/// This is what the resume's own equality cannot see. A resume tolerates a
/// range reported twice, because re-folding is exact and the merge drops the
/// repeated points, so the front comes back whole either way.
fn assert_ranges_are_disjoint(label: &str, walked: &derna_lib::WalkedFront) {
    let cents = |p: &derna_lib::HullPoint| (p.mfe_kcal_mol * 100.0).round() as i64;
    let mut ranges: Vec<(i64, i64, &str)> = walked
        .gaps
        .iter()
        .map(|g| (cents(&g.a), cents(&g.b), "settled"))
        .chain(
            walked
                .open
                .iter()
                .map(|(a, b)| (cents(a), cents(b), "open")),
        )
        .collect();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        let ((lo_a, hi_a, what_a), (lo_b, hi_b, what_b)) = (pair[0], pair[1]);
        assert!(
            hi_a <= lo_b,
            "{label}: the {what_a} range {lo_a}..{hi_a} overlaps the {what_b}              range {lo_b}..{hi_b}"
        );
    }
}

/// A walk stopped anywhere resumes to the whole front.
///
/// The budgets span the walk's phases, so between them they stop it before
/// the grid is pooled, while the sweep is refining pairs, and while the last
/// gaps are folding. Whatever is outstanding at each of those moments has to
/// be reported, or the resume comes back short: a pair the sweep had split, a
/// gap waiting for its fold and a pair the budget skipped are all open boxes,
/// and only the walk knows which it had handed out.
#[test]
fn a_walk_stopped_anywhere_resumes_to_the_whole_front() {
    let whole = walk(WalkBudget::default());
    let want = common::front_objectives(&whole.front);
    let mut stopped_short = 0;
    for secs in [1e-6, 1e-3, 5e-3, 2e-2, 5e-2, 1e-1] {
        let stopped = walk(WalkBudget {
            max_secs: Some(secs),
            ..WalkBudget::default()
        });
        if stopped.open.is_empty() {
            // The walk finished inside this budget: nothing to resume.
            assert!(stopped.stopped_by.is_none(), "{secs}: {:?}", stopped.open);
            common::assert_same(
                &format!("unstopped at {secs}s"),
                &want,
                &common::front_objectives(&stopped.front),
            );
            continue;
        }
        stopped_short += 1;
        assert_ranges_are_disjoint(&format!("stopped at {secs}s"), &stopped);
        let resume = to_resume(&stopped);
        let resumed = walk_run(
            WalkBudget::default(),
            WalkRun {
                resume: Some(&resume),
                ..WalkRun::default()
            },
        );
        assert!(resumed.open.is_empty(), "{secs}: {:?}", resumed.open);
        assert_ranges_are_disjoint(&format!("resumed from {secs}s"), &resumed);
        common::assert_same(
            &format!("resumed from a stop at {secs}s"),
            &want,
            &common::front_objectives(&resumed.front),
        );
    }
    assert!(
        stopped_short >= 2,
        "{stopped_short} budget(s) stopped short"
    );
}

/// A bisecting sweep refines pairs, and a walk stopped while it does resumes
/// to the whole front.
///
/// This is the test that reaches the refinement: the walk starts from its two
/// ends, so every pair between them is split before it is solved, and what the
/// walk owes at any moment is a mix of pairs it has refined and pairs it has
/// not. A pair dropped when it splits, or one of its halves never recorded,
/// loses a range of the hull that no later run is told about.
#[test]
fn a_bisecting_sweep_resumes_to_the_whole_front() {
    let whole = bisecting(WalkBudget::default(), WalkRun::default());
    // The sweep must actually refine, or the split is unreached and nothing
    // below tests it.
    assert!(whole.rounds >= 2, "the sweep never split a pair");
    assert!(whole.open.is_empty(), "{:?}", whole.open);
    assert_ranges_are_disjoint("bisecting, whole", &whole);
    let want = common::front_objectives(&whole.front);

    let mut stopped_short = 0;
    for secs in [1e-6, 1e-3, 5e-3, 2e-2, 5e-2, 1e-1, 2e-1] {
        let stopped = bisecting(
            WalkBudget {
                max_secs: Some(secs),
                ..WalkBudget::default()
            },
            WalkRun::default(),
        );
        if stopped.open.is_empty() {
            continue;
        }
        stopped_short += 1;
        assert_ranges_are_disjoint(&format!("bisecting, stopped at {secs}s"), &stopped);
        let resume = to_resume(&stopped);
        let resumed = bisecting(
            WalkBudget::default(),
            WalkRun {
                resume: Some(&resume),
                ..WalkRun::default()
            },
        );
        assert!(resumed.open.is_empty(), "{secs}: {:?}", resumed.open);
        common::assert_same(
            &format!("bisecting, resumed from a stop at {secs}s"),
            &want,
            &common::front_objectives(&resumed.front),
        );
    }
    assert!(
        stopped_short >= 2,
        "{stopped_short} budget(s) stopped short"
    );
}

/// The walk `walker` runs, and every checkpoint it handed over, in order.
fn checkpointed(
    walker: impl Fn(WalkRun<'_>) -> derna_lib::WalkedFront,
) -> (derna_lib::WalkedFront, Vec<derna_lib::WalkedFront>) {
    let taken = std::sync::Mutex::new(Vec::new());
    let hand = |so_far: &derna_lib::WalkedFront| taken.lock().unwrap().push(so_far.clone());
    let whole = walker(WalkRun {
        checkpoint: Some(&hand),
        ..WalkRun::default()
    });
    (whole, taken.into_inner().unwrap())
}

/// What every checkpoint of a walk must be: a state the walk can be resumed
/// from to the whole front, later than the one before it, and ending at the
/// whole front with nothing open.
///
/// Each checkpoint's ranges are disjoint, so no pair it reports open is also a
/// gap it has settled and no two open boxes overlap; each open box lies
/// between two of its own hull points; and its points are points of the whole
/// front. Returns how many checkpoints had a box open.
fn assert_every_checkpoint_resumes(
    label: &str,
    whole: &derna_lib::WalkedFront,
    taken: &[derna_lib::WalkedFront],
    resume: impl Fn(&WalkResume) -> derna_lib::WalkedFront,
) -> usize {
    let want = common::front_objectives(&whole.front);
    let last = taken
        .last()
        .expect("a walk that settles a pair checkpoints");
    // The last checkpoint is taken after the last pair settles, so it is the
    // whole front: a walk killed after it loses nothing.
    assert!(last.open.is_empty(), "{label}: last open {:?}", last.open);
    common::assert_same(
        &format!("{label}: the last checkpoint"),
        &want,
        &common::front_objectives(&last.front),
    );
    for pair in taken.windows(2) {
        assert!(
            pair[0].gaps.len() <= pair[1].gaps.len(),
            "{label}: a checkpoint settled fewer gaps than the one before it"
        );
    }
    for (i, so_far) in taken.iter().enumerate() {
        let at = format!("{label}: checkpoint {i}");
        assert_ranges_are_disjoint(&at, so_far);
        for (a, b) in &so_far.open {
            assert!(
                a.mfe_kcal_mol < b.mfe_kcal_mol,
                "{at}: {a:?} not below {b:?}"
            );
            for end in [a, b] {
                assert!(
                    so_far
                        .vertices
                        .iter()
                        .any(|v| v.mfe_kcal_mol == end.mfe_kcal_mol),
                    "{at}: {end:?} is no vertex of the checkpoint"
                );
            }
        }
        for point in common::front_objectives(&so_far.front) {
            assert!(
                want.iter()
                    .any(|q| q.mfe == point.mfe && (q.cai_log - point.cai_log).abs() < 1e-9),
                "{at}: {point:?} is not a point of the whole front"
            );
        }
        let resumed = resume(&to_resume(so_far));
        assert!(resumed.open.is_empty(), "{at}: {:?}", resumed.open);
        common::assert_same(
            &format!("{at}, resumed"),
            &want,
            &common::front_objectives(&resumed.front),
        );
    }
    taken.iter().filter(|c| !c.open.is_empty()).count()
}

/// A protein whose seeded walk settles enough gaps for several checkpoints
/// to leave boxes open; [`PROTEIN`]'s settles three.
const LONGER: &str = "MKVLAKGIVLAGQWERS";

/// The walk `derna front` runs, over [`LONGER`].
fn seeded_longer(run: WalkRun<'_>) -> derna_lib::WalkedFront {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let options = WalkOptions {
        pass_grid: Some(PassGrid::for_residues(LONGER.len())),
        seed_sweep: true,
        cut_probes: true,
        ..WalkOptions::default()
    };
    design_front_walk_run(
        &params,
        LONGER,
        &usage,
        &fixed("ACAUUUGCUU"),
        &fixed("GGCCAAUUGG"),
        RegionConstraints::default(),
        1e-4,
        0.999,
        options,
        2,
        run,
    )
    .expect("the seeded walk answers")
}

/// Every checkpoint of the walk `derna front` runs resumes to the whole front.
///
/// This walk seeds its sweep from the grid's optima, which are handed out as
/// pairs only as the grid's passes are pooled; a checkpoint taken before then
/// owes the ranges between the optima already known as well as the pairs
/// handed out.
#[test]
fn every_checkpoint_of_a_walk_resumes_to_the_whole_front() {
    let (whole, taken) = checkpointed(seeded_longer);
    assert!(whole.open.is_empty(), "{:?}", whole.open);
    // The first is taken as the second of the two ends lands, whichever
    // task folds it: it holds both.
    for end in [whole.vertices[0], *whole.vertices.last().unwrap()] {
        assert!(taken[0].vertices.contains(&end), "{end:?}");
    }
    let resume = |r: &WalkResume| {
        seeded_longer(WalkRun {
            resume: Some(r),
            ..WalkRun::default()
        })
    };
    let open = assert_every_checkpoint_resumes("seeded", &whole, &taken, resume);
    // A checkpoint with nothing open resumes to itself and tests nothing.
    assert!(
        open >= 3,
        "{open} of {} checkpoint(s) left a box open",
        taken.len()
    );
}

/// Every checkpoint of a bisecting walk resumes to the whole front, and one is
/// taken after every pair the walk settles.
///
/// A bisecting walk starts from its two ends and refines every pair between
/// them, so each pair either splits into two, adding one hull vertex, or is
/// solved as a gap: the checkpoints number one for the two ends, then the
/// gaps plus the vertices past the two ends. A checkpoint taken while a pair
/// is split, or before a gap's points are recorded, loses a range a resume is
/// never told about.
#[test]
fn every_checkpoint_of_a_bisecting_walk_resumes_to_the_whole_front() {
    let (whole, taken) = checkpointed(|run| bisecting(WalkBudget::default(), run));
    assert!(whole.rounds >= 2, "the sweep never split a pair");
    assert!(whole.open.is_empty(), "{:?}", whole.open);
    assert_eq!(
        taken.len(),
        1 + whole.gaps.len() + whole.vertices.len() - 2,
        "one checkpoint for the ends, then one per settled pair"
    );
    // The first is taken once the ends are folded and before any pair
    // settles: the two ends, with the box between them open.
    let first = &taken[0];
    assert!(first.gaps.is_empty() && first.sweep_scans == 0);
    assert_eq!(first.vertices.len(), 2);
    assert_eq!(first.open, vec![(first.vertices[0], first.vertices[1])]);
    // Each later one is taken once its pair has settled, so the kth shows at
    // least k settled pairs: one taken at a split before the new vertex is
    // recorded shows one fewer, and still resumes.
    for (k, c) in taken.iter().enumerate().skip(1) {
        let settled = c.gaps.len() + c.vertices.len().saturating_sub(2);
        assert!(
            settled >= k,
            "checkpoint {k} shows {settled} settled pair(s)"
        );
    }
    let resume = |r: &WalkResume| {
        bisecting(
            WalkBudget::default(),
            WalkRun {
                resume: Some(r),
                ..WalkRun::default()
            },
        )
    };
    let open = assert_every_checkpoint_resumes("bisecting", &whole, &taken, resume);
    assert!(
        open >= 3,
        "{open} of {} checkpoint(s) left a box open",
        taken.len()
    );
}

/// A resumed walk checkpoints too, and its checkpoints carry the points it was
/// handed: a walk killed while it continues another loses neither.
#[test]
fn every_checkpoint_of_a_resumed_walk_resumes_to_the_whole_front() {
    let whole = bisecting(WalkBudget::default(), WalkRun::default());
    let (first, rest) = whole.gaps.split_first().expect("a walk with gaps");
    let on_hull = |p: &derna_lib::FrontPoint| {
        [first.a, first.b]
            .iter()
            .any(|v| v.mfe_kcal_mol == p.mfe_kcal_mol)
    };
    let given = WalkResume {
        points: first
            .front
            .iter()
            .map(|p| (p.clone(), on_hull(p)))
            .collect(),
        open: rest
            .iter()
            .map(|g| {
                (
                    (g.a.mfe_kcal_mol, g.a.adaptation_log),
                    (g.b.mfe_kcal_mol, g.b.adaptation_log),
                )
            })
            .collect(),

        rests: Vec::new(),
    };
    assert!(given.open.len() >= 2, "{} box(es) open", given.open.len());
    let resume = |r: &WalkResume| {
        bisecting(
            WalkBudget::default(),
            WalkRun {
                resume: Some(r),
                ..WalkRun::default()
            },
        )
    };
    let (resumed, taken) = checkpointed(|run| {
        bisecting(
            WalkBudget::default(),
            WalkRun {
                resume: Some(&given),
                ..run
            },
        )
    });
    let open = assert_every_checkpoint_resumes("resumed", &resumed, &taken, resume);
    assert!(
        open >= 1,
        "{open} of {} checkpoint(s) left a box open",
        taken.len()
    );
    common::assert_same(
        "resumed against whole",
        &common::front_objectives(&whole.front),
        &common::front_objectives(&resumed.front),
    );
}

/// The walk `derna front` runs, stopped after `folds` gap slices, with slices
/// narrow enough that a gap holds several.
fn walk_folds(folds: Option<usize>, resume: Option<&WalkResume>) -> derna_lib::WalkedFront {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let options = WalkOptions {
        pass_grid: Some(PassGrid::for_residues(PROTEIN.len())),
        seed_sweep: true,
        cut_probes: true,
        slice_width: Some(30.0),
        gap_folds: folds,
        ..WalkOptions::default()
    };
    design_front_walk_run(
        &params,
        PROTEIN,
        &usage,
        &fixed(FLANK5),
        &fixed(FLANK3),
        RegionConstraints::default(),
        2f64.powi(-23),
        1.0 - 2f64.powi(-23),
        options,
        1,
        WalkRun {
            resume,
            ..WalkRun::default()
        },
    )
    .unwrap()
}

/// A walk stopped between two slices of a gap keeps the slices it solved, as
/// the gap's rest; resuming from it folds only the energies above the rest
/// and returns the whole front.
#[test]
fn a_walk_stopped_inside_a_gap_keeps_its_rest_and_resumes_to_the_whole_front() {
    let whole = walk_folds(None, None);
    let objectives = |w: &derna_lib::WalkedFront| {
        let mut o: Vec<(i64, i64)> = w
            .front
            .iter()
            .map(|p| {
                (
                    (p.mfe_kcal_mol * 100.0).round() as i64,
                    (p.cai_log * 1e9).round() as i64,
                )
            })
            .collect();
        o.sort_unstable();
        o
    };
    // The first count that stops a gap part way, among the first few.
    let (stopped, folds) = (1..40)
        .map(|k| (walk_folds(Some(k), None), k))
        .find(|(w, _)| !w.rests.is_empty())
        .expect("some count stops a gap between its slices");
    assert_eq!(stopped.stopped_by, Some("folds"));
    let rest = stopped.rests[0];
    assert!(
        rest.lo < rest.from && rest.from < rest.hi,
        "solved part way: {rest:?}"
    );
    // Every point it holds is a point of the whole front.
    let all = objectives(&whole);
    for p in objectives(&stopped) {
        assert!(
            all.contains(&p),
            "{p:?} after {folds} folds is not on the front"
        );
    }
    // Resumed with no budget, the walk reaches the whole front.
    let resumed = walk_folds(None, Some(&to_resume(&stopped)));
    assert!(resumed.open.is_empty() && resumed.rests.is_empty());
    assert_eq!(objectives(&resumed), all);
}
