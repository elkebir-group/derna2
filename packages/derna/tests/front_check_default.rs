//! `derna front` certifies the front it writes by default, and declines to
//! where there is nothing to learn or the caller says so.
//!
//! A certified point carries its verdict, `modular`, beside the
//! energy the walk gave it; an uncertified point carries none. That field is
//! what these tests read, so they hold on the file a caller gets rather than on
//! which function ran.
//!
//! The modular search's own exemption is not visible in its file —
//! every point it writes carries a verdict of its own — and is covered by the
//! `check_points` unit test in `src/front.rs` instead.

use serde_json::Value;

/// The stop slot's cell: small, complete without a budget, and constrained by
/// its 5′ flank, so the default check has something to certify.
const PROTEIN: &str = "MPRGLA";
const UTR5: &str = "CCGCGG";

/// `derna front` with `args`, written to a temporary file and read back.
fn front(args: &[&str]) -> Vec<Value> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!(
        "derna-check-default-{}-{nth}.jsonl",
        std::process::id()
    ));
    let mut argv: Vec<String> = ["front", "--protein", PROTEIN, "--stop", "UAA"]
        .iter()
        .chain(args)
        .chain(&["--threads", "1", "--force", "--out"])
        .map(|s| s.to_string())
        .collect();
    argv.push(out.to_str().unwrap().to_string());
    derna_cli::run(argv).expect("the front runs");
    let text = std::fs::read_to_string(&out).unwrap();
    std::fs::remove_file(&out).unwrap();
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// How many of a front's points carry a verdict, and how many points it has.
fn verdicts(lines: &[Value]) -> (usize, usize) {
    let points = &lines[1..];
    let with = points.iter().filter(|p| p["modular"].is_boolean()).count();
    (with, points.len())
}

/// With a constraint and no flag, every point is certified.
#[test]
fn a_constrained_front_is_certified_by_default() {
    let (with, of) = verdicts(&front(&["--utr5", UTR5]));
    assert!(of > 0, "the front is empty, so the check proves nothing");
    assert_eq!(with, of, "{with} of {of} point(s) carry a verdict");
}

/// `--no-check` leaves the same front uncertified.
#[test]
fn no_check_leaves_the_front_uncertified() {
    let (with, of) = verdicts(&front(&["--utr5", UTR5, "--no-check"]));
    assert!(of > 0, "the front is empty, so the check proves nothing");
    assert_eq!(with, 0, "{with} of {of} point(s) carry a verdict");
}

/// With no constraint every design's free fold respects it, so the run does
/// not fold the points to confirm what is already known.
#[test]
fn an_unconstrained_front_is_not_certified() {
    let (with, of) = verdicts(&front(&[]));
    assert!(of > 0, "the front is empty, so the check proves nothing");
    assert_eq!(with, 0, "{with} of {of} point(s) carry a verdict");
}

/// A motif front under the same constraint is certified as a plain one is:
/// a motif limits what a design may spell, not which pairs its fold may form.
#[test]
fn a_constrained_motif_front_is_certified_by_default() {
    let (with, of) = verdicts(&front(&["--utr5", UTR5, "--motif", "GAAUUC"]));
    assert!(of > 0, "the front is empty, so the check proves nothing");
    assert_eq!(with, of, "{with} of {of} point(s) carry a verdict");
    let (with, of) = verdicts(&front(&["--utr5", UTR5, "--motif", "GAAUUC", "--no-check"]));
    assert!(of > 0, "the front is empty, so the check proves nothing");
    assert_eq!(
        with, 0,
        "--no-check: {with} of {of} point(s) carry a verdict"
    );
}

/// `--check-pool` copies cells between neighbouring points' free folds and
/// changes nothing a caller reads: every certified point is the one the
/// fresh folds write, field for field.
#[test]
fn a_check_pool_writes_the_fresh_folds_points() {
    let fresh = front(&["--utr5", UTR5]);
    let pooled = front(&["--utr5", UTR5, "--check-pool", "4"]);
    let (with, of) = verdicts(&pooled);
    assert!(of > 1, "one point has no neighbour to copy from");
    assert_eq!(with, of, "{with} of {of} point(s) carry a verdict");
    assert_eq!(pooled[1..], fresh[1..]);
}

/// The pool belongs to the default check, so it is refused where that check
/// does not run rather than accepted and ignored.
#[test]
fn a_check_pool_without_the_check_is_refused() {
    for flag in [&["--no-check"][..], &["--beam-first", "50"][..]] {
        let argv: Vec<String> = ["front", "--protein", PROTEIN, "--check-pool", "4"]
            .iter()
            .chain(flag)
            .map(|s| s.to_string())
            .collect();
        let err = derna_cli::run(argv).expect_err("refused");
        assert!(err.to_string().contains(flag[0]), "{flag:?}: {err}");
    }
}
