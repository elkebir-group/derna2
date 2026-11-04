//! The binary's own wiring: which stream carries what, and what the exit code
//! says.
//!
//! The result goes to stdout (or `--out`) and everything else — progress,
//! warnings, the summary — goes to stderr. That is what lets
//! `derna front … | derna check -` compose, and it cannot be checked by
//! calling library functions: it is a property of the process. These tests run
//! the built binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn derna() -> Command {
    Command::new(env!("CARGO_BIN_EXE_derna"))
}

fn run(args: &[&str]) -> Output {
    derna().args(args).output().expect("the binary runs")
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("utf-8 stdout")
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("utf-8 stderr")
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("derna-{}-{}", std::process::id(), name))
}

#[test]
fn a_result_goes_to_stdout_and_everything_else_to_stderr() {
    let out = run(&["design", "--protein", "MKV", "--lambda", "0.5"]);
    assert!(out.status.success(), "{}", stderr_of(&out));

    // Every stdout line is a JSON object of the result format, and nothing
    // else: a progress line on stdout would break the pipe into `check`.
    let text = stdout_of(&out);
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 2, "a header and one point: {lines:?}");
    let header: serde_json::Value = serde_json::from_str(lines[0]).expect("header is JSON");
    assert_eq!(header["record"], "header");
    assert_eq!(header["format"], "derna-result");
    let point: serde_json::Value = serde_json::from_str(lines[1]).expect("point is JSON");
    assert_eq!(point["record"], "point");

    // The human-readable progress is on stderr.
    let err = stderr_of(&out);
    assert!(err.contains("λ=0.5"), "{err}");
}

#[test]
fn out_writes_the_file_and_leaves_stdout_empty() {
    let path = temp("out.jsonl");
    let _ = std::fs::remove_file(&path);
    let first = run(&[
        "design",
        "--protein",
        "MKV",
        "--lambda",
        "0.5",
        "--out",
        path.to_str().unwrap(),
    ]);
    assert!(first.status.success(), "{}", stderr_of(&first));
    assert_eq!(
        stdout_of(&first),
        "",
        "the result went to the file and to stdout"
    );
    let written = std::fs::read_to_string(&path).unwrap();
    assert_eq!(written.lines().count(), 2);

    // A second run refuses rather than overwriting, and says how to overwrite.
    let second = run(&[
        "design",
        "--protein",
        "MKV",
        "--lambda",
        "0.5",
        "--out",
        path.to_str().unwrap(),
    ]);
    assert!(!second.status.success());
    assert_eq!(second.status.code(), Some(1));
    let message = stderr_of(&second);
    assert!(message.contains("--force"), "{message}");
    // The refused run left the file as it was.
    assert_eq!(std::fs::read_to_string(&path).unwrap(), written);

    // ...and --force overwrites it.
    let forced = run(&[
        "design",
        "--protein",
        "MKV",
        "--lambda",
        "0.9",
        "--out",
        path.to_str().unwrap(),
        "--force",
    ]);
    assert!(forced.status.success(), "{}", stderr_of(&forced));
    assert!(std::fs::read_to_string(&path).unwrap().contains("0.9"));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn out_dash_is_stdout_not_a_file_named_dash() {
    // `--protein -` means stdin, so `--out -` means stdout. Before this it
    // wrote a file called `-` in the working directory, which is a surprise
    // the command's own convention argues against.
    let out = run(&[
        "design",
        "--protein",
        "MKV",
        "--lambda",
        "0.5",
        "--out",
        "-",
    ]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let text = stdout_of(&out);
    assert_eq!(text.lines().filter(|l| !l.is_empty()).count(), 2);
    assert!(
        !Path::new("-").exists(),
        "a file named `-` was created in the working directory"
    );
}

#[test]
fn out_to_a_device_is_not_an_overwrite() {
    // `--out /dev/null` is how a caller says *discard this*. The constraint exists
    // to stop a run destroying a file's contents, and a character device has
    // none to destroy, so refusing it would make the constraint an obstacle.
    let out = run(&[
        "design",
        "--protein",
        "MKV",
        "--lambda",
        "0.5",
        "--out",
        "/dev/null",
    ]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(stdout_of(&out), "", "the result went to the device");
    // A regular file is still refused.
    let path = temp("guard.jsonl");
    std::fs::write(&path, "PRECIOUS").unwrap();
    let refused = run(&[
        "design",
        "--protein",
        "MKV",
        "--lambda",
        "0.5",
        "--out",
        path.to_str().unwrap(),
    ]);
    assert!(!refused.status.success());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "PRECIOUS");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn a_design_pipes_into_show() {
    let path = temp("pipe.jsonl");
    let _ = std::fs::remove_file(&path);
    let designed = run(&[
        "design",
        "--protein",
        "MKVLA",
        "--lambda",
        "0.3,0.7",
        "--out",
        path.to_str().unwrap(),
    ]);
    assert!(designed.status.success(), "{}", stderr_of(&designed));

    let shown = run(&["show", path.to_str().unwrap()]);
    std::fs::remove_file(&path).unwrap();
    assert!(shown.status.success(), "{}", stderr_of(&shown));
    let text = stdout_of(&shown);
    assert!(text.contains("design — 5 residue(s)"), "{text}");
    assert!(text.contains("2 point(s)"), "{text}");
    assert!(
        text.contains("0.300000") && text.contains("0.700000"),
        "{text}"
    );
}

#[test]
fn show_reads_stdin_so_the_commands_compose() {
    let designed = run(&["design", "--protein", "MKVLA", "--lambda", "0.5"]);
    assert!(designed.status.success());

    let mut child = derna()
        .args(["show", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("show runs");
    use std::io::Write;
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&designed.stdout)
        .expect("write the result in");
    let shown = child.wait_with_output().expect("show finishes");
    assert!(shown.status.success(), "{}", stderr_of(&shown));
    assert!(
        stdout_of(&shown).contains("1 point(s)"),
        "{}",
        stdout_of(&shown)
    );
}

#[test]
fn a_design_pipes_into_a_check_and_then_into_show() {
    // The pipeline the guide promises, as processes: the result travels on
    // stdout, the molecule travels in the header, and `show` renders what
    // comes out. A library-level test cannot catch a dispatch that never
    // reaches `check`, or a header the writer and the reader disagree about.
    let designed = run(&[
        "design",
        "--protein",
        "MKVLAKGIV",
        "--lambda",
        "0.2,0.8",
        "--utr5",
        "ACAUUUGCUUCUGACACAACUGUGU",
        "--utr3",
        "GCUCGCUUUCUUGCUGUCCAAUUUC",
    ]);
    assert!(designed.status.success(), "{}", stderr_of(&designed));

    let mut checking = derna()
        .args(["check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("check runs");
    use std::io::Write;
    checking
        .stdin
        .take()
        .expect("stdin")
        .write_all(&designed.stdout)
        .expect("write the design in");
    let checked = checking.wait_with_output().expect("check finishes");
    assert!(checked.status.success(), "{}", stderr_of(&checked));
    assert!(
        stderr_of(&checked).contains("modular"),
        "{}",
        stderr_of(&checked)
    );

    // Every point comes back with a verdict, and the header says what it was
    // checked from.
    let text = stdout_of(&checked);
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["command"], "check");
    assert_eq!(header["checked_from"]["command"], "design");
    assert_eq!(lines.len(), 3, "a header and two points");
    for line in &lines[1..] {
        let point: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(point["gap"].as_f64().is_some(), "{line}");
        assert!(point["modular"].as_bool().is_some(), "{line}");
    }

    // ...and `show` renders it.
    let mut show = derna()
        .args(["show", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("show runs");
    show.stdin
        .take()
        .expect("stdin")
        .write_all(&checked.stdout)
        .expect("write the check in");
    let shown = show.wait_with_output().expect("show finishes");
    assert!(shown.status.success(), "{}", stderr_of(&shown));
    assert!(
        stdout_of(&shown).contains("check — 9 residue(s)"),
        "{}",
        stdout_of(&shown)
    );
}

#[test]
fn a_bad_command_line_exits_one_with_the_reason_on_stderr() {
    for args in [
        vec!["design", "--protein", "MKV"],                   // no weight
        vec!["design", "--protein", "MKV", "--lambda", "1"],  // a closed end
        vec!["design", "--lambda", "0.5"],                    // no protein
        vec!["front", "--protein", "MKV", "--lambda", "0.5"], // a weight on the front
        vec!["wat"],                                          // no such command
        vec!["show", "/nonexistent/result.jsonl"],            // no such file
    ] {
        let out = run(&args);
        assert_eq!(out.status.code(), Some(1), "{args:?} did not exit 1");
        assert!(stdout_of(&out).is_empty(), "{args:?} wrote to stdout");
        assert!(
            stderr_of(&out).starts_with("error: "),
            "{args:?}: {}",
            stderr_of(&out)
        );
    }
}

#[test]
fn the_summary_s_energy_span_is_the_span_of_the_file_it_wrote() {
    // The summary reports the energy the front spans, because that is what the
    // walk's cost tracks. It has to be the span of the result it wrote,
    // not of some other field: the first version of this read the wrong one.
    let out_path = temp("span.jsonl");
    // Eleven residues with one flank: six points over 6.4 kcal/mol, in under
    // a second. A shorter protein's front is a single point, which would make
    // the comparison vacuous.
    let out = run(&[
        "front",
        "--protein",
        "MKVLAGSTRPQ",
        "--utr5",
        "ACGUACGUAC",
        "--out",
        out_path.to_str().unwrap(),
        "--force",
    ]);
    assert!(out.status.success(), "{}", stderr_of(&out));

    let reported: f64 = stderr_of(&out)
        .split("spanning ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .expect("the summary names the span")
        .parse()
        .expect("the span is a number");

    let text = std::fs::read_to_string(&out_path).unwrap();
    let energies: Vec<f64> = text
        .lines()
        .skip(1) // the header
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["mfe"]
                .as_f64()
                .unwrap()
        })
        .collect();
    assert!(energies.len() >= 2, "a front of one point tests nothing");
    let span = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - energies.iter().copied().fold(f64::INFINITY, f64::min);

    // The summary prints one decimal, so compare at that resolution.
    assert!(
        (reported - span).abs() < 0.05,
        "the summary says {reported} and the file spans {span}"
    );
    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn help_exits_zero_and_names_the_commands() {
    // Including the no-argument path, which is the same help on stdout: a
    // caller who redirects stdout gets the text, and nothing lands on stderr.
    for args in [vec!["--help"], vec![], vec!["help"]] {
        let out = run(&args);
        assert!(out.status.success(), "{args:?}");
        let text = stdout_of(&out);
        for command in ["design", "front", "check", "show"] {
            assert!(text.contains(command), "{args:?} omits {command}");
        }
        assert!(stderr_of(&out).is_empty(), "{args:?} wrote to stderr");
    }
    for args in [
        ["design", "--help"],
        ["front", "--help"],
        ["check", "--help"],
        ["show", "--help"],
    ] {
        let out = run(&args);
        assert!(out.status.success(), "{args:?}");
        assert!(stdout_of(&out).starts_with("derna "), "{args:?}");
    }
}

#[test]
fn no_edge_input_panics_and_every_refusal_names_its_flag() {
    // A command surface must not panic on an input a user can type. Each case
    // here is an edge of a flag's domain: either it designs, or it is refused
    // with `error: ` and the flag's own name. A panic is neither.
    let cases: [(&str, Vec<&str>); 16] = [
        (
            "one residue",
            vec!["design", "--protein", "M", "--lambda", "0.5"],
        ),
        (
            "one residue with flanks",
            vec![
                "design",
                "--protein",
                "M",
                "--utr5",
                "ACGUACGUAC",
                "--utr3",
                "GGGGCCCCAA",
                "--lambda",
                "0.5",
            ],
        ),
        (
            "a flank longer than the coding sequence",
            vec![
                "design",
                "--protein",
                "MK",
                "--utr5",
                "ACGUACGUACGUACGUACGUACGU",
                "--lambda",
                "0.5",
            ],
        ),
        (
            "a window wider than the flank",
            vec![
                "design",
                "--protein",
                "MK",
                "--utr5",
                "ACGU",
                "--access",
                "100,100",
                "--lambda",
                "0.5",
            ],
        ),
        (
            "a window inside the coding sequence alone",
            vec![
                "design",
                "--protein",
                "MKVLA",
                "--access",
                "0,9",
                "--lambda",
                "0.5",
            ],
        ),
        (
            "the smallest weight",
            vec!["design", "--protein", "MKV", "--lambda", "0.0000001"],
        ),
        (
            "the largest weight",
            vec!["design", "--protein", "MKV", "--lambda", "0.9999999"],
        ),
        ("a front of one residue", vec!["front", "--protein", "M"]),
        ("a front of two residues", vec!["front", "--protein", "MK"]),
        (
            "a motif longer than the molecule",
            vec![
                "design",
                "--protein",
                "MK",
                "--lambda",
                "0.5",
                "--motif",
                "ACGUACGUACGUACGU",
            ],
        ),
        (
            "a reversed window",
            vec![
                "design",
                "--protein",
                "MKVLA",
                "--lambda",
                "0.5",
                "--motif",
                "GGGG",
                "--motif-window",
                "9..3",
            ],
        ),
        (
            "a window at the coding sequence's last base",
            vec![
                "design",
                "--protein",
                "MKVLA",
                "--lambda",
                "0.5",
                "--motif",
                "GGGG",
                "--motif-window",
                "-1..-1",
            ],
        ),
        (
            "a pin on the last base",
            vec![
                "design",
                "--protein",
                "MKVLA",
                "--lambda",
                "0.5",
                "--pin",
                "-1:G",
            ],
        ),
        (
            "a pin past the end",
            vec![
                "design",
                "--protein",
                "MKVLA",
                "--lambda",
                "0.5",
                "--pin",
                "999:G",
            ],
        ),
        (
            "an empty protein",
            vec!["design", "--protein", "", "--lambda", "0.5"],
        ),
        ("an empty result file", vec!["show", "/dev/null"]),
    ];
    for (what, args) in cases {
        let out = run(&args);
        let err = stderr_of(&out);
        assert!(!err.contains("panicked"), "{what} panicked:\n{err}");
        if !out.status.success() {
            assert_eq!(out.status.code(), Some(1), "{what}");
            assert!(err.starts_with("error: "), "{what}: {err}");
        }
    }
}

#[test]
fn the_binary_says_which_version_wrote_a_result() {
    // A result header records the version that wrote it, so the binary has to
    // be able to say which one it is — otherwise a file's `version` field
    // cannot be matched to anything on disk.
    let designed = run(&["design", "--protein", "MKV", "--lambda", "0.5"]);
    let text = stdout_of(&designed);
    let header: serde_json::Value =
        serde_json::from_str(text.lines().next().expect("a header")).unwrap();
    let recorded = header["version"].as_str().expect("a version").to_string();

    for flag in ["--version", "-V", "version"] {
        let out = run(&[flag]);
        assert!(out.status.success(), "{flag}");
        assert_eq!(
            stdout_of(&out).trim(),
            format!("derna {recorded}"),
            "{flag}"
        );
    }
}

#[test]
fn a_motif_in_a_fixed_flank_warns_on_stderr_and_still_designs() {
    // The warning belongs on stderr: a result file is machine-read, and a
    // warning in it would be a line a reader has to skip.
    let out = run(&[
        "design",
        "--protein",
        "MKVLA",
        "--lambda",
        "0.5",
        "--utr5",
        "ACGUACGU",
        "--forbid",
        "none",
        // A sequence constraint is not carried over a designed stop codon, so
        // this run asks for none.
        "--stop",
        "none",
        // Wholly inside the fixed flank (positions 3-6 of ACGUACGU): a
        // pattern straddling the boundary would reach the start codon, which
        // the protein fixes, and the run would be refused instead.
        "--motif",
        "GUAC",
    ]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let err = stderr_of(&out);
    assert!(err.contains("warning:"), "{err}");
    assert!(err.contains("5′ UTR"), "{err}");
    // And the design is still there, on stdout, with the flank intact.
    let text = stdout_of(&out);
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let point: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert!(point["sequence"].as_str().unwrap().starts_with("ACGUACGU"));
}
