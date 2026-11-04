//! `derna front` reproduces the front recorded from the standalone walk it
//! was built over, with and without flanks and constraint, and marks exactly
//! the hull's vertices `on_hull`.
//!
//! Both sides state what they walk: the fixtures are generated with the range
//! `derna front` fixes ([2⁻²³, 1 − 2⁻²³]) and the automatic pass grid spelled
//! out on the instrument's command line, so a difference can only come from
//! the command, not from a default. The fixtures are frozen output of that
//! walk, which is not part of this distribution.
//!
//! The instrument writes its front with energies at two decimals and
//! adaptations at six, so the comparison is at the precision the file carries.

use std::path::{Path, PathBuf};

use derna_cli::front;
use derna_cli::molecule::Molecule;
use serde_json::Value;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_json(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(data(name)).unwrap()).unwrap()
}

fn read_jsonl(name: &str) -> Vec<Value> {
    std::fs::read_to_string(data(name))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn derna_words(case: &Value) -> Vec<String> {
    // Each case names its own protein: the first cell carries every
    // constraint spelling, and a second protein with a second flank pair
    // carries the default one, so the equality is not a property of one
    // molecule.
    let fasta = repo_root().join(case["fasta"].as_str().unwrap());
    let mut words = vec!["--protein".to_string(), fasta.to_str().unwrap().to_string()];
    words.extend(
        case["derna"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap().to_string()),
    );
    words
}

/// Energies are integer dekacalories under the model, printed at two decimals.
const ENERGY_TOL: f64 = 5e-3;
/// Adaptations are printed at six decimals.
const LOG_TOL: f64 = 5e-7;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn front_matches_the_walk_on_every_recorded_case() {
    let manifest = read_json("front-cases.json");
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let expected = read_jsonl(&format!("front-{name}.jsonl"));
        let report = read_json(&format!("front-{name}.report.json"));

        let args = front::parse(derna_words(case)).unwrap();
        let m = Molecule::resolve(&args.shared).unwrap();
        let walked = front::walk(&m, 1).unwrap();
        let hull = front::on_hull(&walked);

        assert_eq!(expected.len(), walked.front.len(), "{name}: point count");
        assert_eq!(
            report["points"].as_u64().unwrap() as usize,
            walked.front.len(),
            "{name}: the report's own count"
        );
        for (e, got) in expected.iter().zip(&walked.front) {
            let at = |field: &str| e[field].as_f64().unwrap();
            assert!(
                close(at("mfe_kcal_mol"), got.mfe_kcal_mol, ENERGY_TOL),
                "{name}: mfe {} vs {}",
                at("mfe_kcal_mol"),
                got.mfe_kcal_mol
            );
            assert!(
                close(at("cai_log"), got.cai_log, LOG_TOL),
                "{name}: cai_log {} vs {}",
                at("cai_log"),
                got.cai_log
            );
            assert!(close(at("cai"), got.cai, LOG_TOL), "{name}: cai");
            assert_eq!(
                e["sequence"].as_str().unwrap(),
                got.sequence,
                "{name}: sequence"
            );
            assert_eq!(
                e["structure"].as_str().unwrap(),
                got.structure,
                "{name}: structure"
            );
        }

        // `on_hull` marks exactly the vertices the walk's own report lists.
        let vertices = report["vertices"].as_array().unwrap();
        assert_eq!(
            vertices.len(),
            hull.iter().filter(|&&h| h).count(),
            "{name}: hull size"
        );
        for (p, &on) in walked.front.iter().zip(&hull) {
            let listed = vertices.iter().any(|v| {
                close(
                    v["mfe_kcal_mol"].as_f64().unwrap(),
                    p.mfe_kcal_mol,
                    ENERGY_TOL,
                ) && close(
                    v["adaptation_log"].as_f64().unwrap(),
                    p.cai_log + p.utr_profile_log,
                    LOG_TOL,
                )
            });
            assert_eq!(listed, on, "{name}: on_hull at mfe {}", p.mfe_kcal_mol);
        }
    }
}

#[test]
fn the_cases_are_not_all_the_same_front() {
    // Non-vacuity: the constraint has to change the front, or every case above
    // would pass against one walk. The constrained fronts differ from the free
    // one in point count or in their best energy.
    let free = read_jsonl("front-glpe-flanks-free.jsonl");
    let all = read_jsonl("front-glpe-flanks-all.jsonl");
    let best = |f: &[Value]| {
        f.iter()
            .map(|p| p["mfe_kcal_mol"].as_f64().unwrap())
            .fold(f64::MAX, f64::min)
    };
    assert!(
        free.len() != all.len() || best(&free) < best(&all),
        "the three-class constraint changed neither the count nor the best energy"
    );
    // And the front carries off-hull points, so `on_hull` is not constant.
    let report = read_json("front-glpe-flanks-all.report.json");
    assert!(
        report["vertices"].as_array().unwrap().len() < all.len(),
        "every point is a vertex, so on_hull is untested"
    );
}

#[test]
fn the_fixtures_were_walked_over_the_range_the_command_fixes() {
    let manifest = read_json("front-cases.json");
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let report = read_json(&format!("front-{name}.report.json"));
        assert_eq!(report["lo"].as_f64().unwrap(), front::EPSILON, "{name}: lo");
        assert_eq!(
            report["hi"].as_f64().unwrap(),
            1.0 - front::EPSILON,
            "{name}: hi"
        );
    }
}
