//! A development setting in the environment changes nothing unless the build
//! reads development settings.
//!
//! `DERNA_MOTIF_GAP_FOLDS` stops a walk after that many gap folds, so a build
//! that reads it returns a truncated front, and one that does not returns the
//! whole front, the same as with the variable unset. The test checks whichever
//! of the two this build is, so the setting is shown to bite where it is read
//! and to be ignored where it is not.
//!
//! A test binary of its own, because the setting is process-wide and every
//! test in one binary would share it.

use derna_cli::front;
use derna_cli::show;

/// The setting this test sets: a gap-fold limit.
const VAR: &str = "DERNA_MOTIF_GAP_FOLDS";

/// Long enough for a gap wider than one slice.
const PROTEIN: &str = "MKVLAKGIVLAGQWERSTYLMKVLAKGIVLAGQWERSTYL";

fn front_file(name: &str) -> show::Rendered {
    let dir = std::env::temp_dir().join(format!("derna-dev-env-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join(name);
    let _ = std::fs::remove_file(&out);
    let argv: Vec<String> = ["--protein", PROTEIN, "--threads", "1", "--out"]
        .iter()
        .map(|w| w.to_string())
        .chain([out.to_str().unwrap().to_string()])
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

/// The header less the run's own cost, which differs between any two runs.
fn without_cost(r: &show::Rendered) -> serde_json::Value {
    let mut h = r.header.clone();
    let o = h.as_object_mut().unwrap();
    o.remove("wall_s");
    o.remove("peak_rss_kb");
    h
}

#[test]
fn a_gap_fold_limit_in_the_environment_bites_only_where_it_is_read() {
    std::env::remove_var(VAR);
    let unset = front_file("unset.jsonl");
    assert_eq!(unset.header["complete"], true);
    std::env::set_var(VAR, "1");
    let set = front_file("set.jsonl");
    std::env::remove_var(VAR);
    assert_eq!(set.header["complete"], true);
    assert_eq!(without_cost(&set), without_cost(&unset));
    assert_eq!(objectives(&set), objectives(&unset));
    assert_eq!(set.points, unset.points);
}
