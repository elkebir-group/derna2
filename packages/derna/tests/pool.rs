//! `derna pool`: joining several certified fronts of one molecule.
//!
//! The command exists for one property the objectives cannot see. A design
//! that is modular under one stop codon can be dominated in
//! `(mfe, cai_log)` by a design under another, so re-dominating the pooled
//! points throws it away — and with it the only reason to have walked the
//! stops apart. `pool` therefore writes two fronts: the pooled points with the
//! dominated ones dropped, and the same filter applied *inside* the
//! modular designs alone. The second is the one that keeps them.
//!
//! The arms here are written by hand rather than walked. The join is the
//! behaviour under test, and a hand-written arm can state the case that makes
//! the two fronts differ — a modular design strictly dominated by a
//! non-modular one — on four points instead of several hundred, with no fold
//! and no dependence on what a walk happens to find.

use serde_json::{json, Value};

/// A 3′ flank, long enough that the stop codon is not at the sequence's end:
/// `pool` locates the stop by stepping back over this flank, so a test whose
/// flank were empty would pass for the wrong reason.
const UTR3: &str = "GGGCCC";

/// A temporary directory of this process, named by a counter: the tests run in
/// one process on several threads and each cleans up after itself.
fn work_dir(tag: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("derna-pool-{tag}-{}-{nth}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a work directory");
    dir
}

/// A front header of the molecule every arm in these tests is of.
fn header(stop: &str) -> Value {
    json!({
        "record": "header",
        // What marks the file as a result rather than any other JSONL: the
        // reader refuses a first line without these two. The version is taken
        // from the crate rather than written out, so these fixtures follow it
        // when it moves instead of going stale against it.
        "format": "derna-result",
        "format_version": derna_cli::result::FORMAT_VERSION,
        "command": "front",
        "protein": "MPRG",
        "utr5": {"sequence": ""},
        "utr3": {"sequence": UTR3},
        "forbid": ["5-cds", "cds-3"],
        "stop": stop,
        "complete": true,
        "stopped_by": Value::Null,
    })
}

/// One certified point: the coding sequence, then `stop`, then the 3′ flank,
/// so the codon `pool` reads back out of the sequence is `stop`.
///
/// `cds` only has to be a plausible coding sequence of the right shape; no
/// fold is taken over it here.
fn point(cds: &str, stop: &str, mfe: f64, cai_log: f64, modular: bool) -> Value {
    json!({
        "record": "point",
        "sequence": format!("{cds}{stop}{UTR3}"),
        "mfe": mfe,
        "cai_log": cai_log,
        "modular": modular,
    })
}

/// Write an arm and return its path.
fn arm(dir: &std::path::Path, name: &str, header: &Value, points: &[Value]) -> std::path::PathBuf {
    let path = dir.join(format!("{name}.jsonl"));
    let mut text = header.to_string();
    for p in points {
        text.push('\n');
        text.push_str(&p.to_string());
    }
    text.push('\n');
    std::fs::write(&path, text).expect("an arm is written");
    path
}

/// `derna pool` over `inputs`, writing fronts named from `out`.
fn pool(
    inputs: &[&std::path::Path],
    out: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut argv: Vec<String> = vec!["pool".to_string()];
    argv.extend(inputs.iter().map(|p| p.to_str().unwrap().to_string()));
    argv.push("--out".to_string());
    argv.push(out.to_str().unwrap().to_string());
    argv.push("--force".to_string());
    derna_cli::run(argv)
}

/// The lines of one written front.
fn read(path: &std::path::Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// `out`'s two fronts, as `(header, points)` each.
type Front = (Value, Vec<Value>);
fn written(out: &std::path::Path) -> (Front, Front) {
    let split = |name: &str| {
        let stem = out.file_stem().unwrap().to_string_lossy().into_owned();
        let lines = read(&out.with_file_name(format!("{stem}.{name}.jsonl")));
        let (h, p) = lines.split_first().expect("a header and its points");
        (h.clone(), p.to_vec())
    };
    (split("union"), split("modular"))
}

/// The sequences of a front's points, sorted so an ordering change does not
/// read as a different front.
fn sequences(points: &[Value]) -> Vec<String> {
    let mut v: Vec<String> = points
        .iter()
        .map(|p| p["sequence"].as_str().unwrap().to_string())
        .collect();
    v.sort();
    v
}

/// Two arms whose modular design is dominated by the other arm's non-modular
/// one: the case the command exists for.
///
/// `UAA` holds a modular design at `(-10.0, -2.0)`. `UAG` holds a non-modular
/// design at `(-11.0, -1.0)`, better on both axes, so it dominates it.
fn dominated_case(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let uaa = arm(
        dir,
        "uaa",
        &header("UAA"),
        &[
            point("AUGCCGCGCGGG", "UAA", -10.0, -2.0, true),
            point("AUGCCGCGCGGA", "UAA", -4.0, -0.5, false),
        ],
    );
    let uag = arm(
        dir,
        "uag",
        &header("UAG"),
        &[
            point("AUGCCGCGCGGG", "UAG", -11.0, -1.0, false),
            point("AUGCCGCGCGGC", "UAG", -3.0, -0.2, false),
        ],
    );
    (uaa, uag)
}

/// The modular front keeps a modular design that the pooled front drops.
///
/// This is the whole claim: the `UAA` arm's modular design is strictly
/// dominated by the `UAG` arm's non-modular one, so it is absent from the
/// union and present in the modular front. A `pool` that filtered before
/// separating the modular designs would lose it from both.
#[test]
fn the_modular_front_keeps_what_domination_takes() {
    let dir = work_dir("dominated");
    let (uaa, uag) = dominated_case(&dir);
    let out = dir.join("joined.jsonl");
    pool(&[&uaa, &uag], &out).expect("pool joins two certified arms");
    let ((_, union), (_, modular)) = written(&out);

    let kept = format!("AUGCCGCGCGGGUAA{UTR3}");
    assert!(
        !sequences(&union).contains(&kept),
        "the union should have dropped the dominated modular design, got {:?}",
        sequences(&union)
    );
    assert_eq!(
        sequences(&modular),
        vec![kept],
        "the modular front should hold exactly the modular design the union drops"
    );
}

/// Every pooled point is labelled with the stop its own sequence spells, so a
/// row's stop is readable after the arms are joined and no longer separate
/// files.
#[test]
fn a_pooled_point_carries_the_stop_it_spells() {
    let dir = work_dir("labels");
    let (uaa, uag) = dominated_case(&dir);
    let out = dir.join("joined.jsonl");
    pool(&[&uaa, &uag], &out).expect("pool runs");
    let ((_, union), _) = written(&out);
    for p in &union {
        let sequence = p["sequence"].as_str().unwrap();
        let spelled = &sequence[sequence.len() - UTR3.len() - 3..sequence.len() - UTR3.len()];
        assert_eq!(
            p["stop"].as_str().unwrap(),
            spelled,
            "a pooled point's stop disagrees with the codon it spells"
        );
    }
}

/// An arm whose header names one stop while its points spell another is
/// refused. The label is read from the sequence, so a mislabelled file would
/// otherwise be pooled under the wrong stop — silently, since nothing
/// downstream re-reads the codon.
#[test]
fn an_arm_whose_header_disagrees_with_its_points_is_refused() {
    let dir = work_dir("mislabelled");
    let good = arm(
        dir.as_path(),
        "good",
        &header("UAA"),
        &[point("AUGCCGCGCGGG", "UAA", -10.0, -2.0, true)],
    );
    // The header says UAG; every point spells UAA.
    let lying = arm(
        dir.as_path(),
        "lying",
        &header("UAG"),
        &[point("AUGCCGCGCGGA", "UAA", -9.0, -1.5, true)],
    );
    let out = dir.join("joined.jsonl");
    let e = pool(&[&good, &lying], &out)
        .expect_err("an arm whose header and rows disagree is refused")
        .to_string();
    assert!(e.contains("UAG"), "the declared stop is not named: {e}");
    assert!(e.contains("UAA"), "the spelled stop is not named: {e}");
    assert!(e.contains("lying"), "the file at fault is not named: {e}");
}

/// A front that has not been certified is refused by name: a modular front
/// cannot be taken over points whose own free fold was never folded, and
/// treating a missing verdict as a failing one would silently shrink the
/// front it is meant to preserve.
#[test]
fn an_uncertified_arm_is_refused_by_name() {
    let dir = work_dir("uncertified");
    let certified = arm(
        dir.as_path(),
        "certified",
        &header("UAA"),
        &[point("AUGCCGCGCGGG", "UAA", -10.0, -2.0, true)],
    );
    let mut bare = point("AUGCCGCGCGGA", "UAG", -9.0, -1.5, false);
    bare.as_object_mut().unwrap().remove("modular");
    let uncertified = arm(dir.as_path(), "uncertified", &header("UAG"), &[bare]);
    let out = dir.join("joined.jsonl");
    let e = pool(&[&certified, &uncertified], &out)
        .expect_err("an uncertified arm is refused")
        .to_string();
    assert!(
        e.contains("uncertified"),
        "the uncertified file is not named: {e}"
    );
    assert!(
        e.contains("check"),
        "the message does not say how to certify it: {e}"
    );
}

/// Fronts of different molecules are refused, and the message names the field
/// that differs. A union over two molecules is a front of neither, and the
/// stop is deliberately not part of that identity — differing stops are
/// exactly what this joins.
#[test]
fn fronts_of_different_molecules_are_refused_by_field() {
    let dir = work_dir("molecules");
    let mine = arm(
        dir.as_path(),
        "mine",
        &header("UAA"),
        &[point("AUGCCGCGCGGG", "UAA", -10.0, -2.0, true)],
    );
    let mut other = header("UAG");
    other["protein"] = json!("MPRGK");
    let theirs = arm(
        dir.as_path(),
        "theirs",
        &other,
        &[point("AUGCCGCGCGGA", "UAG", -9.0, -1.5, true)],
    );
    let out = dir.join("joined.jsonl");
    let e = pool(&[&mine, &theirs], &out)
        .expect_err("fronts of different molecules are refused")
        .to_string();
    assert!(e.contains("protein"), "the field is not named: {e}");
    assert!(
        e.contains("mine") && e.contains("theirs"),
        "both files are not named: {e}"
    );
}

/// Arms differing only in their stop are joined rather than refused, which is
/// the complement of the test above: the stop is not part of a molecule's
/// identity here.
#[test]
fn arms_differing_only_in_their_stop_are_joined() {
    let dir = work_dir("stops-differ");
    let (uaa, uag) = dominated_case(&dir);
    let out = dir.join("joined.jsonl");
    pool(&[&uaa, &uag], &out).expect("arms of different stops join");
    let ((header, _), _) = written(&out);
    assert_eq!(
        header["stop"],
        json!(["UAA", "UAG"]),
        "the pooled header should list every arm's stop"
    );
}

/// Each written front says what each arm contributed, per arm rather than
/// reduced to one flag: one settled arm beside an open one is not the same
/// claim as two open ones, and a pooled front can say no more than its least
/// complete arm.
#[test]
fn the_header_records_what_each_arm_contributed() {
    let dir = work_dir("provenance");
    let (uaa, uag) = dominated_case(&dir);
    let out = dir.join("joined.jsonl");
    pool(&[&uaa, &uag], &out).expect("pool runs");
    let ((header, union), _) = written(&out);
    let arms = header["arms"].as_array().expect("the arms are recorded");
    assert_eq!(arms.len(), 2, "one record per arm");
    assert_eq!(arms[0]["stop"], json!("UAA"));
    assert_eq!(arms[0]["points"], json!(2));
    assert_eq!(arms[0]["modular"], json!(1), "the UAA arm's modular count");
    assert_eq!(arms[1]["modular"], json!(0), "the UAG arm's modular count");
    assert_eq!(arms[0]["complete"], json!(true));
    assert_eq!(
        header["pooled"],
        json!(4),
        "every arm's points are pooled before the filter"
    );
    assert_eq!(
        header["points"],
        json!(union.len()),
        "the header's count is the front's own"
    );
}

/// An arm that stopped on a budget carries that forward: the pooled front
/// cannot claim to be more complete than the arm it was built from.
#[test]
fn an_incomplete_arm_is_carried_forward() {
    let dir = work_dir("incomplete");
    let whole = arm(
        dir.as_path(),
        "whole",
        &header("UAA"),
        &[point("AUGCCGCGCGGG", "UAA", -10.0, -2.0, true)],
    );
    let mut stopped = header("UAG");
    stopped["complete"] = json!(false);
    stopped["stopped_by"] = json!("max-secs");
    let partial = arm(
        dir.as_path(),
        "partial",
        &stopped,
        &[point("AUGCCGCGCGGA", "UAG", -9.0, -1.5, false)],
    );
    let out = dir.join("joined.jsonl");
    pool(&[&whole, &partial], &out).expect("pool joins a stopped arm");
    let ((header, _), _) = written(&out);
    let arms = header["arms"].as_array().unwrap();
    assert_eq!(arms[0]["complete"], json!(true));
    assert_eq!(arms[1]["complete"], json!(false));
    assert_eq!(
        arms[1]["stopped_by"],
        json!("max-secs"),
        "why the arm stopped should survive the join"
    );
}

/// One arm is a join too: `pool` over a single front is its non-dominated
/// points and its modular ones, not an error. The modular front of one arm is
/// what that arm's own walk could not give, since a walk filters before
/// certifying.
#[test]
fn one_arm_is_a_join_of_one() {
    let dir = work_dir("single");
    let only = arm(
        dir.as_path(),
        "only",
        &header("UAA"),
        &[
            point("AUGCCGCGCGGG", "UAA", -10.0, -2.0, true),
            point("AUGCCGCGCGGA", "UAA", -11.0, -1.0, false),
        ],
    );
    let out = dir.join("joined.jsonl");
    pool(&[&only], &out).expect("one front joins");
    let ((_, union), (_, modular)) = written(&out);
    assert_eq!(union.len(), 1, "the dominated point is dropped");
    assert_eq!(
        sequences(&modular),
        vec![format!("AUGCCGCGCGGGUAA{UTR3}")],
        "the modular design survives inside the modular front"
    );
}
