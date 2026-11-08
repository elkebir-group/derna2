//! `derna front --motif` stopped with gaps solved part way: the header gives
//! each partly solved box the energy its gap is solved up to and the best
//! adaptation there, and `--resume` continues each box from that rest.

use derna_cli::front;
use derna_cli::show;

const PROTEIN: &str = "MKVLAKGIVLAGQWERS";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";
const MOTIF: &str = "GCC";

/// Run `derna front` on the molecule with these further words, writing
/// `name`, and read back what it wrote.
fn front_file(extra: &[&str], name: &str) -> (std::path::PathBuf, show::Rendered) {
    front_file_in("rest", extra, name)
}

/// [`front_file`] in a directory named by `tag`, so tests running at once do
/// not remove each other's files.
fn front_file_in(tag: &str, extra: &[&str], name: &str) -> (std::path::PathBuf, show::Rendered) {
    let dir = std::env::temp_dir().join(format!("derna-motif-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join(name);
    let _ = std::fs::remove_file(&out);
    let argv: Vec<String> = [
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--stop",
        "none",
        "--motif",
        MOTIF,
        "--threads",
        "1",
        "--out",
        out.to_str().unwrap(),
    ]
    .iter()
    .chain(extra)
    .map(|w| w.to_string())
    .collect();
    front::run(&front::parse(argv).unwrap()).unwrap();
    let read = show::read(Some(&out)).unwrap();
    (out, read)
}

/// A point's energy in integer dekacalories, its `cai_log`, and whether it
/// is a hull vertex.
fn objectives(p: &serde_json::Value) -> (i64, f64, bool) {
    (
        (p["mfe"].as_f64().unwrap() * 100.0).round() as i64,
        p["cai_log"].as_f64().unwrap(),
        p["on_hull"].as_bool().unwrap(),
    )
}

/// A walk stopped after every gap fold, each round resumed from the file the
/// last wrote, writes `mfe_from` and `cai_log_floor` on the boxes it solved
/// part way, and ends at the front of the walk run whole.
///
/// Each round solves one range, so a box solved part way is finished only by
/// resumes that read its rest: one that ignored it would fold the box from
/// its stable end again, solve the same first range, and never finish. A rest
/// never moves down from one round to the next, and some box must be solved
/// part way, or the rest is never read back.
#[test]
fn a_front_stopped_inside_its_gaps_resumes_from_the_header_to_the_whole_front() {
    let (_, whole) = front_file(&[], "whole.jsonl");
    let (mut from, mut walk) = front_file(&["--max-gap-folds", "1"], "round-0.jsonl");
    // The stop is real: the first round returned less than the whole front.
    assert!(
        walk.points.len() < whole.points.len(),
        "one gap fold returned {} of {} points",
        walk.points.len(),
        whole.points.len()
    );
    let dekacal = |kcal: f64| (kcal * 100.0).round() as i64;
    let mut rests_before = std::collections::HashMap::new();
    let mut part_way = 0;
    let mut rounds = 0;
    while walk.header["complete"] == false {
        assert_eq!(walk.header["stopped_by"], "folds");
        let mut rests = std::collections::HashMap::new();
        for open in walk.header["open"].as_array().unwrap() {
            let (Some(rest), Some(floor)) =
                (open["mfe_from"].as_f64(), open["cai_log_floor"].as_f64())
            else {
                continue;
            };
            let lo = open["mfe_lo"].as_f64().unwrap();
            let hi = open["mfe_hi"].as_f64().unwrap();
            // One hundredth below the stable end: probed, no slice solved.
            assert!(lo - 0.01 - 1e-9 <= rest && rest < hi, "{open}");
            assert!(
                floor >= open["cai_log_lo"].as_f64().unwrap() - 1e-9,
                "{open}"
            );
            let (key, rest) = ((dekacal(lo), dekacal(hi)), dekacal(rest));
            if let Some(&before) = rests_before.get(&key) {
                assert!(rest >= before, "{open} fell back from {before}");
            }
            if rest >= key.0 {
                part_way += 1;
            }
            rests.insert(key, rest);
        }
        rests_before = rests;
        rounds += 1;
        assert!(rounds < 500, "the rounds do not finish");
        let resume = ["--resume", from.to_str().unwrap(), "--max-gap-folds", "1"];
        let next = front_file(&resume, &format!("round-{rounds}.jsonl"));
        (from, walk) = next;
    }
    std::fs::remove_dir_all(from.parent().unwrap()).unwrap();
    assert!(
        part_way > 0,
        "no box was solved part way in {rounds} rounds"
    );
    assert_eq!(walk.points.len(), whole.points.len());
    for (a, b) in walk.points.iter().zip(&whole.points) {
        let (a, b) = (objectives(a), objectives(b));
        assert_eq!((a.0, a.2), (b.0, b.2));
        assert!((a.1 - b.1).abs() < 1e-9, "{a:?} against {b:?}");
    }
}

/// A box whose probe finds a design beyond one of its ends is settled as its
/// own gap, not split there, so the walk neither panics on a reversed gap nor
/// writes one that a resume refuses.
///
/// The box is made by marking a front point that lies strictly under the
/// chord of the hull vertices around it as a hull vertex, and leaving open
/// the box from the stable vertex to it. The probe at that box's tie weight
/// finds the next hull vertex, which is more adapted than the box's adapted
/// end. Splitting there leaves a box with its ends reversed: solving it
/// panics, and stopping before it is solved writes it into the header.
#[test]
fn a_probe_beyond_a_box_end_neither_panics_nor_writes_a_reversed_box() {
    let (whole_path, whole) = front_file_in("beyond", &[], "whole.jsonl");
    let points = &whole.points;
    let at = |p: &serde_json::Value| (p["mfe"].as_f64().unwrap(), p["cai_log"].as_f64().unwrap());
    let hull: Vec<usize> = (0..points.len())
        .filter(|&k| points[k]["on_hull"] == true)
        .collect();
    // Two adjacent hull vertices with a point strictly under their chord.
    let (lo, inner) = hull
        .windows(2)
        .find_map(|w| {
            let ((m0, c0), (m1, c1)) = (at(&points[w[0]]), at(&points[w[1]]));
            (w[0] + 1..w[1])
                .find(|&k| {
                    let (m, c) = at(&points[k]);
                    c < c0 + (c1 - c0) * (m - m0) / (m1 - m0) - 1e-6
                })
                .map(|k| (w[0], k))
        })
        .expect("a front point under a hull chord");
    let ((mlo, clo), (mhi, chi)) = (at(&points[lo]), at(&points[inner]));
    let mut header = whole.header.clone();
    header["complete"] = false.into();
    header["stopped_by"] = "secs".into();
    header["open"] = serde_json::json!([
        {"mfe_lo": mlo, "mfe_hi": mhi, "cai_log_lo": clo, "cai_log_hi": chi}
    ]);
    let mut lines = vec![header.to_string()];
    for (k, p) in points.iter().enumerate() {
        let mut p = p.clone();
        if k == inner {
            p["on_hull"] = true.into();
        }
        lines.push(p.to_string());
    }
    let crafted = whole_path.with_file_name("crafted.jsonl");
    std::fs::write(&crafted, lines.join("\n") + "\n").unwrap();
    let crafted = crafted.to_str().unwrap();
    // Run whole, the box's own gap is solved and the front is unchanged.
    let (_, resumed) = front_file_in("beyond", &["--resume", crafted], "resumed.jsonl");
    assert_eq!(resumed.header["complete"], true);
    assert_eq!(resumed.points.len(), points.len());
    for (a, b) in resumed.points.iter().zip(points) {
        let (a, b) = (objectives(a), objectives(b));
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-9, "{a:?} against {b:?}");
    }
    // Stopped after one gap fold, every box it leaves open can be resumed.
    let (stopped, read) = front_file_in(
        "beyond",
        &["--resume", crafted, "--max-gap-folds", "1"],
        "stopped.jsonl",
    );
    for open in read.header["open"].as_array().unwrap() {
        let f = |k: &str| open[k].as_f64().unwrap();
        assert!(
            f("mfe_lo") <= f("mfe_hi") && f("cai_log_lo") <= f("cai_log_hi"),
            "a reversed box was written: {open}"
        );
    }
    if read.header["complete"] == false {
        front_file_in(
            "beyond",
            &["--resume", stopped.to_str().unwrap()],
            "again.jsonl",
        );
    }
    std::fs::remove_dir_all(whole_path.parent().unwrap()).unwrap();
}
