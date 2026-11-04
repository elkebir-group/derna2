//! A stop codon left free is one more slot of the codon lattice.
//!
//! With flanks given and no `--stop`, a run designs its stop codon: the last
//! slot of the coding sequence chooses among `UAA`, `UAG` and `UGA` inside the
//! same search as every residue, and each point names the stop it ends in.
//!
//! The claim under test is that the one search returns what the three pinned
//! searches return re-dominated together. A design has exactly one stop codon,
//! so the free design space is the disjoint union of the three pinned ones: a
//! design is on the union's front exactly when no design of any stop dominates
//! it, and modularity is a property of a whole sequence, its stop
//! included, so no pinned run's verdict changes by being merged. The pinned
//! runs are the oracle; both directions of the equality are asserted, because
//! each alone passes vacuously.
//!
//! A stop codon carries no adaptation, so the slot adds nothing to any design's
//! `cai_log` and the fronts of the four runs share one adaptation axis.
//!
//! The cell is the one the modular front's own tests use, so every
//! run here is small and complete without a budget.

use serde_json::Value;

const STOPS: [&str; 3] = ["UAA", "UAG", "UGA"];
const PROTEIN: &str = "MPRGLA";
const UTR5: &str = "CCGCGG";

/// Run `derna front` on the cell and read its lines.
///
/// The tests run in one process on several threads and each deletes its own
/// output, so the file is named by a counter rather than by the flags.
fn run(extra: &[&str]) -> Vec<Value> {
    run_cell(PROTEIN, UTR5, extra)
}

/// Run `derna front` on `protein` behind the 5′ flank `utr5` and read its lines.
fn run_cell(protein: &str, utr5: &str, extra: &[&str]) -> Vec<Value> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!(
        "derna-stop-slot-{}-{nth}.jsonl",
        std::process::id()
    ));
    let mut argv: Vec<String> = [
        "front",
        "--protein",
        protein,
        "--utr5",
        utr5,
        "--force",
        // One worker: the front is the same at any thread count, and a test
        // that says so on a shared host costs its neighbours nothing.
        "--threads",
        "1",
        "--out",
        out.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    argv.extend(extra.iter().map(|s| s.to_string()));
    derna_cli::run(argv).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    std::fs::remove_file(&out).unwrap();
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// One point: its sequence and its two objectives.
#[derive(Debug, Clone, PartialEq)]
struct Point {
    sequence: String,
    mfe: f64,
    cai_log: f64,
}

fn points(lines: &[Value]) -> Vec<Point> {
    lines[1..].iter().map(point).collect()
}

fn point(p: &Value) -> Point {
    Point {
        sequence: p["sequence"].as_str().unwrap().to_string(),
        mfe: p["mfe"].as_f64().unwrap(),
        cai_log: p["cai_log"].as_f64().unwrap(),
    }
}

/// How close two adaptations must be to be one: a log sum's last bit can move
/// with the order its terms are added in.
const SAME_CAI: f64 = 1e-9;

/// Whether `q` is no worse than `p` in either objective and better in one.
fn dominates(q: &Point, p: &Point) -> bool {
    q.mfe <= p.mfe
        && q.cai_log >= p.cai_log - SAME_CAI
        && (q.mfe < p.mfe || q.cai_log > p.cai_log + SAME_CAI)
}

/// The points of `all` nothing in `all` dominates.
fn undominated(all: &[Point]) -> Vec<Point> {
    all.iter()
        .filter(|p| !all.iter().any(|q| dominates(q, p)))
        .cloned()
        .collect()
}

/// Whether two point sets hold the same objectives, each point of one matched
/// by a point of the other.
fn same_objectives(a: &[Point], b: &[Point]) -> bool {
    let covered = |x: &[Point], y: &[Point]| {
        x.iter().all(|p| {
            y.iter()
                .any(|q| q.mfe == p.mfe && (q.cai_log - p.cai_log).abs() <= SAME_CAI)
        })
    };
    covered(a, b) && covered(b, a)
}

/// The stop codon a whole molecule of this cell ends in: the cell has no 3′ UTR,
/// so the stop is its last three nucleotides.
fn stop_of(sequence: &str) -> &str {
    &sequence[sequence.len() - 3..]
}

#[test]
fn the_free_front_is_the_three_pinned_fronts_re_dominated() {
    let free = points(&run(&[]));
    let pinned: Vec<Vec<Point>> = STOPS.iter().map(|s| points(&run(&["--stop", s]))).collect();
    let union: Vec<Point> = pinned.iter().flatten().cloned().collect();
    let expected = undominated(&union);

    // Something to choose between: otherwise the equality below says nothing.
    // Not asserted as "the union is larger than the free front", which two arms
    // agreeing exactly in both objectives would fail while the equality held —
    // tying designs dominate neither each other nor anything else.
    for (stop, arm) in STOPS.iter().zip(&pinned) {
        assert!(!arm.is_empty(), "--stop {stop} returned no point");
    }
    assert!(!free.is_empty());
    assert!(
        same_objectives(&free, &expected),
        "free {free:?}\nunion {expected:?}"
    );
    // Every free point is a design of the union, ending in one of the three.
    for p in &free {
        assert!(STOPS.contains(&stop_of(&p.sequence)), "{p:?}");
        assert!(!union.iter().any(|q| dominates(q, p)), "{p:?} is dominated");
    }
}

/// The same relation where the fold pairs the designed stop. A 3′ flank
/// complementary to the coding sequence's last nucleotides, under no pair
/// constraint, opens a pair on the stop's `U`: the node where the `UA` and `UG`
/// lanes spell the same nucleotide, so one closed-pair cell stands for both.
/// The walk's pruning must still see the lane the best designs take, here
/// `UG`; the cell above never pairs its stop and would not notice.
#[test]
fn a_stop_paired_into_the_3_prime_flank_keeps_the_free_front() {
    const PROTEIN: &str = "MAKVLE";
    const UTR5: &str = "CCCUUUCCCUUU";
    let paired = ["--utr3", "UUUCACUCUUU", "--forbid", "none"];
    let lines = run_cell(PROTEIN, UTR5, &paired);
    let free = points(&lines);
    let pinned: Vec<Vec<Point>> = STOPS
        .iter()
        .map(|s| {
            points(&run_cell(
                PROTEIN,
                UTR5,
                &[&paired[..], &["--stop", s]].concat(),
            ))
        })
        .collect();
    let union: Vec<Point> = pinned.iter().flatten().cloned().collect();
    for (stop, arm) in STOPS.iter().zip(&pinned) {
        assert!(!arm.is_empty(), "--stop {stop} returned no point");
    }
    let expected = undominated(&union);
    assert!(
        same_objectives(&free, &expected),
        "free {free:?}\nunion {expected:?}"
    );
    // The stop is paired in what comes back, so the shared cell was on the
    // answer rather than beside it.
    let stop = UTR5.len() + 3 * PROTEIN.len();
    let paired_stop = lines[1..].iter().any(|p| {
        p["structure"].as_str().unwrap()[stop..stop + 3]
            .chars()
            .any(|c| c != '.')
    });
    assert!(paired_stop, "no design pairs its stop codon: {free:?}");
}

#[test]
fn the_free_modular_front_is_the_pinned_ones_re_dominated() {
    let free_lines = run(&["--modular"]);
    let free = points(&free_lines);
    let pinned: Vec<Vec<Point>> = STOPS
        .iter()
        .map(|s| points(&run(&["--modular", "--stop", s])))
        .collect();
    let union: Vec<Point> = pinned.iter().flatten().cloned().collect();
    let expected = undominated(&union);
    assert!(!free.is_empty());
    assert!(
        same_objectives(&free, &expected),
        "free {free:?}\nunion {expected:?}"
    );
    // By sequence, the exact comparison: each point the search returns is one a
    // pinned search returned, so the slot invents no design.
    for p in &free {
        assert!(
            union.iter().any(|q| q.sequence == p.sequence),
            "{} is on no pinned front",
            p.sequence
        );
    }
    // Every pinned run and the free one ran whole, so every point is proven.
    let header = &free_lines[0];
    assert_eq!(header["complete"], true, "{header}");
    for p in &free_lines[1..] {
        assert_eq!(p["status"], "certified", "{p}");
        assert_eq!(p["modular"], true, "{p}");
    }
}

#[test]
fn every_point_names_the_stop_it_ends_in() {
    for extra in [&[][..], &["--modular"][..]] {
        let lines = run(extra);
        let (header, rows) = lines.split_first().unwrap();
        assert_eq!(header["stop"], "free", "{header}");
        assert!(!rows.is_empty());
        for row in rows {
            let sequence = row["sequence"].as_str().unwrap();
            assert_eq!(sequence.len(), UTR5.len() + 3 * PROTEIN.len() + 3, "{row}");
            assert_eq!(row["stop"], stop_of(sequence), "{row}");
            assert!(STOPS.contains(&stop_of(sequence)), "{row}");
        }
    }
}

#[test]
fn a_pinned_run_names_no_stop_per_point() {
    let lines = run(&["--stop", "UGA"]);
    assert_eq!(lines[0]["stop"], "UGA");
    for row in &lines[1..] {
        assert!(row.get("stop").is_none(), "{row}");
        assert!(row["sequence"].as_str().unwrap().ends_with("UGA"));
    }
}

#[test]
fn the_slot_moves_no_reported_adaptation() {
    // A stop codon weighs nothing, and the conventional CAI is a mean over the
    // residues' codons, so a free point reports what the same design reports
    // with its stop pinned.
    let free = run(&[]);
    let pinned: Vec<Vec<Value>> = STOPS.iter().map(|s| run(&["--stop", s])).collect();
    for row in &free[1..] {
        let cai_log = row["cai_log"].as_f64().unwrap();
        let cai = row["cai"].as_f64().unwrap();
        assert!(
            (cai - (cai_log / PROTEIN.len() as f64).exp()).abs() < 1e-12,
            "{row}"
        );
        let sequence = row["sequence"].as_str().unwrap();
        for arm in &pinned {
            for other in &arm[1..] {
                if other["sequence"] == sequence {
                    assert_eq!(other["cai_log"], row["cai_log"], "{row}");
                    assert_eq!(other["cai"], row["cai"], "{row}");
                    assert_eq!(other["mfe"], row["mfe"], "{row}");
                }
            }
        }
    }
}

#[test]
fn a_front_walked_for_one_stop_cannot_seed_a_search_over_another() {
    // The guarantee a pipeline depends on. A front whose designs all end in one
    // given stop codon is not the union's front, so a modular
    // search reading it back would certify its points against a front missing
    // every design of the other two stops — and certify points that cannot be
    // on the union's front at all. `--front-from` compares the molecule the
    // file names against the run's, stop included, so the mistake is an error
    // rather than a quieter answer.
    let dir = std::env::temp_dir();
    let pinned = dir.join(format!("derna-stop-slot-seed-{}.jsonl", std::process::id()));
    let walk = |extra: &[&str]| {
        let mut argv: Vec<String> = [
            "front",
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--threads",
            "1",
            "--force",
            "--out",
            pinned.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        derna_cli::run(argv)
    };
    walk(&["--stop", "UAA"]).expect("a pinned front");

    let search = |extra: &[&str]| {
        let mut argv: Vec<String> = [
            "front",
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--threads",
            "1",
            "--modular",
            "--front-from",
            pinned.to_str().unwrap(),
            "--force",
            "--out",
            dir.join(format!(
                "derna-stop-slot-seeded-{}.jsonl",
                std::process::id()
            ))
            .to_str()
            .unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        derna_cli::run(argv)
    };
    // A free-stop search cannot read a UAA front…
    let e = search(&[])
        .expect_err("a free-stop search read a pinned front")
        .to_string();
    assert!(e.contains("stop"), "{e}");
    assert!(e.contains("another molecule"), "{e}");
    // …nor can a search over one stop read another's.
    let e = search(&["--stop", "UGA"])
        .expect_err("a UGA search read a UAA front")
        .to_string();
    assert!(e.contains("stop"), "{e}");
    // The same stop reads back, so the refusal is about the molecule and not
    // about `--front-from`.
    search(&["--stop", "UAA"]).expect("a UAA search reading a UAA front");
    std::fs::remove_file(&pinned).unwrap();
}

/// A motif constraint over a designed stop, on a cell whose fold pairs the
/// stop into the 3′ flank. The stop slot's first node has two out-edges
/// spelling `U`, so the constrained fold's traceback has to know which lane a
/// pair took there.
///
/// The oracle is the three pinned runs, pooled and kept to the designs whose
/// whole coding sequence, stop included, spells no motif, then re-dominated. A
/// pinned stop sits in the 3′ flank, where a motif straddling the last sense
/// codon and the stop is not enforced, so each pinned run is a relaxation and
/// its motif-spelling designs are filtered out here instead.
///
/// The three motifs are the `E|stop` junction under each stop. `AGUGA` is
/// taken from the unconstrained front's own designs, so the constraint removes
/// points; the other two remove none here but steer the constrained fold's
/// traceback through a pair on the slot's shared node, where a lookup by
/// nucleotide alone fails to reproduce the stored cell.
#[test]
fn a_motif_front_over_a_designed_stop_paired_into_the_flank() {
    const PROTEIN: &str = "MAKVLE";
    const UTR5: &str = "CCCUUUCCCUUU";
    const MOTIFS: [&str; 3] = ["AGUGA", "AGUAG", "AGUAA"];
    let flank = ["--utr3", "UUUCACUCUUU", "--forbid", "none"];
    let motifs = MOTIFS.map(|m| ["--motif", m]).concat();
    let cds = UTR5.len()..UTR5.len() + 3 * PROTEIN.len() + 3;
    let motif_free = |p: &Point| !MOTIFS.iter().any(|m| p.sequence[cds.clone()].contains(m));
    let front = |extra: &[&[&str]]| -> Vec<Value> {
        let argv: Vec<&str> = extra.iter().flat_map(|a| a.iter().copied()).collect();
        run_cell(PROTEIN, UTR5, &argv)
            .into_iter()
            .filter(|l| l["record"] == "point")
            .collect()
    };
    let lines = front(&[&flank, &motifs]);
    let free: Vec<Point> = lines.iter().map(point).collect();
    let pool: Vec<Point> = STOPS
        .iter()
        .flat_map(|s| front(&[&flank, &motifs, &["--stop", s]]))
        .map(|l| point(&l))
        .filter(|p| motif_free(p))
        .collect();
    let expected = undominated(&pool);
    assert!(
        same_objectives(&free, &expected),
        "free {free:?}\nexpected {expected:?}"
    );
    assert!(
        free.iter().all(motif_free),
        "a returned design spells a motif"
    );
    // The motifs bite: the unconstrained front holds designs they forbid.
    let unconstrained: Vec<Point> = front(&[&flank]).iter().map(point).collect();
    assert!(
        unconstrained.iter().any(|p| !motif_free(p)),
        "the motifs forbid nothing on this cell"
    );
    // And the stop is paired in what comes back, so a pair's edge was resolved
    // at the slot rather than beside it.
    let stop = cds.end - 3;
    let paired = lines.iter().any(|p| {
        p["structure"].as_str().unwrap()[stop..stop + 3]
            .chars()
            .any(|c| c != '.')
    });
    assert!(paired, "no design pairs its stop codon: {free:?}");
}

/// A pin over a designed stop, through `design`, on the same paired cell. A
/// free stop's design space is the disjoint union of the three pinned ones, so
/// the weighted optimum over it is the best of their three optima: the design
/// matches one of them in both objectives and none of them dominates it. A pin
/// restricts the lattice's edges rather than running the motif automaton, so
/// this checks that the two compose, not the constrained fold's traceback.
#[test]
fn a_pinned_design_over_a_designed_stop_is_one_of_the_pinned_optima() {
    const UTR5: &str = "CCCUUUCCCUUU";
    const PIN: &str = "GCC";
    let design = |extra: &[&str]| -> Value {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let out = std::env::temp_dir().join(format!(
            "derna-stop-slot-pin-{}-{nth}.jsonl",
            std::process::id()
        ));
        let mut argv: Vec<String> = [
            "design",
            "--protein",
            "MAKVLE",
            "--utr5",
            UTR5,
            "--utr3",
            "UUUCACUCUUU",
            "--forbid",
            "none",
            "--lambda",
            "0.5",
            "--pin",
            &format!("4:{PIN}"),
            "--threads",
            "1",
            "--force",
            "--out",
            out.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        derna_cli::run(argv).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        std::fs::remove_file(&out).unwrap();
        let last = text.lines().last().unwrap();
        serde_json::from_str(last).unwrap()
    };
    let line = design(&[]);
    let free = point(&line);
    let pinned: Vec<Point> = STOPS
        .iter()
        .map(|s| point(&design(&["--stop", s])))
        .collect();
    assert!(
        pinned
            .iter()
            .any(|p| p.mfe == free.mfe && (p.cai_log - free.cai_log).abs() <= SAME_CAI),
        "free {free:?} is none of the pinned optima {pinned:?}"
    );
    assert!(
        !pinned.iter().any(|q| dominates(q, &free)),
        "a pinned optimum dominates {free:?}: {pinned:?}"
    );
    assert_eq!(&free.sequence[UTR5.len() + 3..UTR5.len() + 6], PIN);
    // The stop is paired, so the constrained traceback resolved an edge at it.
    let stop = UTR5.len() + 18;
    let structure = line["structure"].as_str().unwrap();
    assert!(
        structure[stop..stop + 3].chars().any(|c| c != '.'),
        "the stop is unpaired: {structure}"
    );
}

/// A motif front narrowed from one under fewer motifs, over a designed stop on
/// the paired cell, is the front walked from the ends: the narrowed walk folds
/// through the same stop slot, so its kept and reopened points must come back
/// with the same objectives, each free of every motif.
///
/// `GCAAA` removes only an inner point of the `AGUAG` front, so the narrowing
/// keeps both ends and reopens one stretch; `AGUGA` is spelled at every point's
/// stop, so the narrowing loses its ends and walks from them.
#[test]
fn a_narrowed_motif_front_over_a_designed_stop_is_the_walked_front() {
    const PROTEIN: &str = "MAKVLE";
    const UTR5: &str = "CCCUUUCCCUUU";
    let flank = ["--utr3", "UUUCACUCUUU", "--forbid", "none"];
    let cds = UTR5.len()..UTR5.len() + 3 * PROTEIN.len() + 3;
    let wider_out = std::env::temp_dir().join(format!(
        "derna-stop-slot-narrow-{}.jsonl",
        std::process::id()
    ));
    let mut argv: Vec<String> = [
        "front",
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--motif",
        "AGUAG",
        "--threads",
        "1",
        "--force",
        "--out",
        wider_out.to_str().unwrap(),
    ]
    .iter()
    .chain(flank.iter())
    .map(|s| s.to_string())
    .collect();
    derna_cli::run(argv.clone()).unwrap();
    argv.clear();
    let narrow_from = ["--narrow-from", wider_out.to_str().unwrap()];
    for motifs in [["AGUAG", "GCAAA"], ["AGUAG", "AGUGA"]] {
        let flags = motifs
            .iter()
            .flat_map(|m| ["--motif", *m])
            .collect::<Vec<_>>();
        let walked: Vec<Point> = points(&run_cell(PROTEIN, UTR5, &[&flank[..], &flags].concat()));
        let narrowed: Vec<Point> = points(&run_cell(
            PROTEIN,
            UTR5,
            &[&flank[..], &flags, &narrow_from].concat(),
        ));
        assert!(
            same_objectives(&narrowed, &walked),
            "{motifs:?}: narrowed {narrowed:?}\nwalked {walked:?}"
        );
        assert!(
            narrowed
                .iter()
                .all(|p| !motifs.iter().any(|m| p.sequence[cds.clone()].contains(m))),
            "{motifs:?}: a narrowed design spells a motif"
        );
    }
    std::fs::remove_file(&wider_out).unwrap();
}

#[test]
fn without_flanks_an_absent_stop_is_no_stop() {
    // The stop can matter only through a flank's forbidden pair, so with none
    // the default is unchanged: no stop codon at all.
    let argv = |extra: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = ["--protein", PROTEIN]
            .iter()
            .map(|s| s.to_string())
            .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    let bare = derna_cli::front::parse(argv(&[])).unwrap();
    let m = derna_cli::molecule::Molecule::resolve(&bare.shared).unwrap();
    assert!(m.stop.is_absent(), "{:?}", m.stop);
    let flanked = derna_cli::front::parse(argv(&["--utr5", UTR5])).unwrap();
    let m = derna_cli::molecule::Molecule::resolve(&flanked.shared).unwrap();
    assert!(m.stop.is_free(), "{:?}", m.stop);
    let none = derna_cli::front::parse(argv(&["--utr5", UTR5, "--stop", "none"])).unwrap();
    let m = derna_cli::molecule::Molecule::resolve(&none.shared).unwrap();
    assert!(m.stop.is_absent(), "{:?}", m.stop);
}

#[test]
fn stop_free_spells_out_what_an_omitted_stop_means_at_a_flank() {
    // A driver that names every flag can ask for a designed stop by name, and
    // gets the run an omitted flag gives: the same molecule and the same front.
    let argv = |extra: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = ["--protein", PROTEIN]
            .iter()
            .map(|s| s.to_string())
            .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    };
    let named = derna_cli::front::parse(argv(&["--utr5", UTR5, "--stop", "free"])).unwrap();
    let m = derna_cli::molecule::Molecule::resolve(&named.shared).unwrap();
    assert!(m.stop.is_free(), "{:?}", m.stop);
    assert_eq!(points(&run(&["--stop", "free"])), points(&run(&[])));
    // Without a flank there is nothing a designed stop could change, and the
    // request is refused rather than quietly read as no stop.
    let bare = derna_cli::front::parse(argv(&["--stop", "free"])).unwrap();
    let e = derna_cli::molecule::Molecule::resolve(&bare.shared)
        .unwrap_err()
        .to_string();
    assert!(e.contains("needs a flank"), "{e}");
}

#[test]
fn a_free_stop_result_checks_as_one_molecule() {
    // `check` rebuilds the molecule from the header. A free stop is one slot of
    // three nucleotides wherever it is, so its constraint classifies every pair
    // as each point's own pinned molecule would, and the check certifies what
    // the search certified.
    let lines = run(&["--modular"]);
    let m = derna_cli::molecule::Molecule::from_header(&lines[0]).unwrap();
    assert!(m.stop.is_free());
    let certified = derna_cli::check::certify(&m, &lines[1..], None).unwrap();
    for (point, checked) in &certified {
        assert!(checked.modular(), "{point}");
        let stop = point["stop"].as_str().unwrap();
        let pinned_m = {
            let mut header = lines[0].clone();
            header["stop"] = stop.into();
            derna_cli::molecule::Molecule::from_header(&header).unwrap()
        };
        let pinned =
            derna_cli::check::certify(&pinned_m, std::slice::from_ref(point), None).unwrap();
        assert_eq!(
            derna_cli::check::certified_point(point, checked),
            derna_cli::check::certified_point(point, &pinned[0].1),
            "{point}"
        );
    }
}

#[test]
fn one_lattice_certifies_one_front_and_not_three() {
    // The point of the exercise. Certification is the dominant cost at length,
    // and one lattice pays it over the union's constrained front once where one
    // search per stop codon pays it over three fronts.
    //
    // Asserted on the **count** of constrained-front points certified rather
    // than on `certify_s`: a count is what the search did and is reproducible,
    // where a wall on a shared host is a reading about the host as much as
    // about the code. `certify_s` is asserted only to be present and finite,
    // so a caller measuring the saving has a field to read.
    let header = |lines: &[Value]| lines[0].clone();
    let free = header(&run(&["--modular"]));
    let pinned: Vec<Value> = STOPS
        .iter()
        .map(|s| header(&run(&["--modular", "--stop", s])))
        .collect();

    let count = |h: &Value| h["constrained_points"].as_u64().expect("a count");
    let largest = pinned.iter().map(count).max().expect("three arms");
    assert!(largest > 0, "{pinned:?}");
    // The bound is against the **largest arm**, not the three arms' sum.
    //
    // `union <= sum` would hold by the arithmetic of a Pareto merge whatever
    // the search did — dominated points are pruned and duplicates collapse, and
    // the arms share one adaptation scale so they overlap heavily. That
    // assertion passes on a union of 2.9x one arm, which would *refute* the
    // claim it looks like it is testing.
    //
    // The claim is that one lattice certifies a front the size of one arm's:
    // front size tracks synonymous freedom, and one three-codon slot barely
    // moves it. Measured at **1.00x** the largest arm on this cell, and
    // elsewhere at 0.94x (163 residues, hbb flanks) and 1.00x (239 residues,
    // bnt flanks) — so the slot does not widen the front even at six residues,
    // where it is a third of the design space rather than a rounding error.
    //
    // The 1.5x allowed below is therefore slack over a measured 1.00x, and it
    // is tight enough to fail on the case that would refute the claim: a union
    // of 2.9x one arm, which a bound against the three arms' sum would have
    // accepted.
    assert!(
        2 * count(&free) <= 3 * largest,
        "the union's front ({}) is more than 1.5x one arm's ({largest}) — \
         certifying it once is not what certifying one arm's costs",
        count(&free)
    );
    for h in std::iter::once(&free).chain(&pinned) {
        let s = h["certify_s"].as_f64().expect("certify seconds");
        assert!(s >= 0.0 && s.is_finite(), "{h}");
    }
}

#[test]
fn a_free_front_reads_back_and_reports_one_walk() {
    // One walk over one lattice: it has one report, and its front can be written
    // beside a search and read back by another, which three pinned walks could
    // not offer.
    let dir = std::env::temp_dir();
    let front = dir.join(format!(
        "derna-stop-slot-front-{}.jsonl",
        std::process::id()
    ));
    let report = dir.join(format!(
        "derna-stop-slot-report-{}.json",
        std::process::id()
    ));
    let first = run(&[
        "--modular",
        "--front-out",
        front.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]);
    let walk: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert!(
        walk["vertices"].as_array().is_some_and(|v| !v.is_empty()),
        "{walk}"
    );
    let again = run(&["--modular", "--front-from", front.to_str().unwrap()]);
    std::fs::remove_file(&front).unwrap();
    std::fs::remove_file(&report).unwrap();
    assert_eq!(points(&first), points(&again));
}
