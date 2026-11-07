//! `derna front` (no motifs) stopped with a gap solved part way: the header
//! gives the box the energy its gap is solved up to and the best adaptation
//! there, as a motif front's does, and `--resume` continues the box from that
//! rest to the whole front.

use derna_cli::front;
use derna_cli::show;

/// Long enough for a gap wider than one slice.
const PROTEIN: &str = "MKVLAKGIVLAGQWERSTYLMKVLAKGIVLAGQWERSTYL";

fn front_file(extra: &[&str], name: &str) -> show::Rendered {
    let dir = std::env::temp_dir().join(format!("derna-front-rest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join(name);
    let _ = std::fs::remove_file(&out);
    let argv: Vec<String> = ["--protein", PROTEIN, "--threads", "1", "--out"]
        .iter()
        .map(|w| w.to_string())
        .chain([out.to_str().unwrap().to_string()])
        .chain(extra.iter().map(|w| w.to_string()))
        .collect();
    front::run(&front::parse(argv).unwrap()).unwrap();
    show::read(Some(&out)).unwrap()
}

fn objectives(r: &show::Rendered) -> Vec<(i64, i64)> {
    let mut o: Vec<(i64, i64)> = r
        .points
        .iter()
        .map(|p| {
            (
                (p["mfe"].as_f64().unwrap() * 100.0).round() as i64,
                (p["cai_log"].as_f64().unwrap() * 1e9).round() as i64,
            )
        })
        .collect();
    o.sort_unstable();
    o
}

#[test]
fn a_plain_front_stopped_inside_a_gap_resumes_from_the_header_to_the_whole_front() {
    let whole = front_file(&[], "whole.jsonl");
    let part_way = |r: &show::Rendered| {
        r.header["open"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b.get("mfe_from").is_some())
    };
    let mut stopped = None;
    for k in 1..80 {
        let name = format!("stopped-{k}.jsonl");
        let r = front_file(&["--max-gap-folds", &k.to_string()], &name);
        if r.header["complete"] == true {
            break;
        }
        assert_eq!(r.header["stopped_by"], "folds");
        // The stop is real: the run returned less than the whole front.
        assert!(
            r.points.len() < whole.points.len(),
            "stopped after {k} gap folds with {} of {} points",
            r.points.len(),
            whole.points.len()
        );
        if part_way(&r) {
            stopped = Some(name);
            break;
        }
    }
    let name = stopped.expect("some count stops a gap between its slices");
    let dir = std::env::temp_dir().join(format!("derna-front-rest-{}", std::process::id()));
    let from = dir.join(&name);
    let resumed = front_file(&["--resume", from.to_str().unwrap()], "resumed.jsonl");
    assert_eq!(resumed.header["complete"], true);
    assert_eq!(objectives(&resumed), objectives(&whole));
}
