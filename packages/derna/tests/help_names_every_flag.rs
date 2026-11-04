//! `--help` names every flag the command accepts.
//!
//! A flag a parser takes but the help text omits is undiscoverable: the only
//! way to learn it is to read the source. The test is written the other way
//! round from a reflection test — it lists the flags each command accepts,
//! checks the help names them, and checks the parser *rejects* anything the
//! help does not name — so adding a flag without documenting it fails here.

use derna_cli::{design, front, pick, show};

/// Every flag `derna design` accepts.
const DESIGN_FLAGS: [&str; 18] = [
    "--quiet",
    "--protein",
    "--codon-usage",
    "--utr5",
    "--utr3",
    "--stop",
    "--forbid",
    "--access",
    "--threads",
    "--deterministic",
    "--beam-first",
    "--out",
    "--force",
    "--lambda",
    "--motif",
    "--motif-window",
    "--pin",
    "--checkpoint",
];

/// Every flag `derna front` accepts.
const FRONT_FLAGS: [&str; 36] = [
    "--quiet",
    "--walk-secs",
    "--check-pool",
    "--no-check",
    "--no-checkpoint",
    "--protein",
    "--codon-usage",
    "--utr5",
    "--utr3",
    "--stop",
    "--forbid",
    "--access",
    "--threads",
    "--deterministic",
    "--out",
    "--force",
    "--report",
    "--modular",
    "--max-secs",
    "--max-rss-gb",
    "--shed-on-memory",
    "--empty-stop-rounds",
    "--empty-stop-derivations",
    "--hold-barren",
    "--front-out",
    "--front-from",
    "--narrow-from",
    "--motif",
    "--motif-window",
    "--resume",
    "--checkpoint",
    "--resume-search",
    "--anchors-from",
    "--top-search",
    "--epsilon",
    "--fold-threads",
];

/// Every flag `derna check` accepts. The molecule's flags are not among
/// them — the result file's header carries those — and neither is
/// `--deterministic`, which a certification already is at any thread count.
const CHECK_FLAGS: [&str; 6] = [
    "--quiet",
    "--threads",
    "--beam-first",
    "--out",
    "--force",
    "--checkpoint",
];

/// Every flag `derna pool` accepts.
const POOL_FLAGS: [&str; 3] = ["--out", "--force", "--quiet"];

/// Every flag `derna show` accepts.
const SHOW_FLAGS: [&str; 12] = [
    "--sequences",
    "--min-cai",
    "--max-mfe",
    "--modular",
    "--tsv",
    "--fasta",
    "--min-gc",
    "--max-gc",
    "--max-homopolymer",
    "--dna",
    "--cds-only",
    "--plot",
];

/// Every flag `derna pick` accepts.
const PICK_FLAGS: [&str; 12] = [
    "--best-cai",
    "--best-mfe",
    "--knee",
    "--min-cai",
    "--max-mfe",
    "--modular",
    "--min-gc",
    "--max-gc",
    "--max-homopolymer",
    "--dna",
    "--cds-only",
    "--json",
];

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

#[test]
fn design_help_names_every_flag_it_takes() {
    let help = design::help();
    for flag in DESIGN_FLAGS {
        assert!(help.contains(flag), "design --help omits {flag}");
    }
    // `--motif-window` belongs to the motif surface and is named with it.
    assert!(help.contains("--motif-window"));
}

#[test]
fn front_help_names_every_flag_it_takes() {
    let help = front::help();
    for flag in FRONT_FLAGS {
        assert!(help.contains(flag), "front --help omits {flag}");
    }
}

#[test]
fn check_help_names_every_flag_it_takes() {
    let help = derna_cli::check::help();
    for flag in CHECK_FLAGS {
        assert!(help.contains(flag), "check --help omits {flag}");
    }
    // And it says what a failing point gets, so a caller does not expect a
    // replacement.
    assert!(help.contains("a failing"), "{help}");
}

#[test]
fn show_help_names_every_flag_it_takes() {
    let help = show::help();
    for flag in SHOW_FLAGS {
        assert!(help.contains(flag), "show --help omits {flag}");
    }
}

#[test]
fn pick_help_names_every_flag_it_takes() {
    let help = pick::help();
    for flag in PICK_FLAGS {
        assert!(help.contains(flag), "pick --help omits {flag}");
    }
}

/// Every long flag a help text names, read off the lines that document one.
fn flags_named(help: &str) -> Vec<String> {
    help.lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("--"))
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

#[test]
fn no_help_text_names_a_flag_its_command_refuses() {
    // The direction the lists above do not cover. They assert the help names
    // every flag the command takes; nothing asserted the converse, and
    // `front`'s help inherited `--beam-first` from the shared block for a
    // while — a flag it refuses, with a reason, after telling the reader to
    // try it. A help text that advertises a refusal is the same defect as an
    // undocumented flag, pointing the other way.
    for (command, help, accepted) in [
        ("design", design::help(), DESIGN_FLAGS.as_slice()),
        ("front", front::help(), FRONT_FLAGS.as_slice()),
        ("check", derna_cli::check::help(), CHECK_FLAGS.as_slice()),
        ("pool", derna_cli::pool::help(), POOL_FLAGS.as_slice()),
        ("show", show::help(), SHOW_FLAGS.as_slice()),
        ("pick", pick::help(), PICK_FLAGS.as_slice()),
    ] {
        let named = flags_named(&help);
        assert!(!named.is_empty(), "{command}'s help names no flag at all");
        for flag in named {
            assert!(
                accepted.contains(&flag.as_str()),
                "{command}'s help names {flag}, which {command} does not accept"
            );
        }
    }
}

#[test]
fn each_command_takes_the_flags_its_help_names_and_no_others() {
    // The other direction: a flag named for one command is not silently
    // accepted by another. `front` takes no weight and no pin; `design` takes
    // no walk report.
    for flag in ["--lambda", "--pin", "--beam-first"] {
        assert!(
            front::parse(words(&["--protein", "MK", flag, "1"])).is_err(),
            "front accepted {flag}, which its help does not name"
        );
    }
    assert!(
        design::parse(words(&[
            "--protein",
            "MK",
            "--lambda",
            "0.5",
            "--report",
            "r.json"
        ]))
        .is_err(),
        "design accepted --report, which its help does not name"
    );
    // A flag the spec gives a command but the command cannot run yet says what
    // it waits on, rather than reading as a typo.
    for (command, flag) in [
        ("design", "--strict"),
        ("design", "--max-secs"),
        ("check", "--report"),
        ("check", "--max-rss-gb"),
        ("front", "--strict"),
    ] {
        let argv = match command {
            "design" => words(&["--protein", "MK", "--lambda", "0.5", flag, "1"]),
            "check" => words(&["front.jsonl", flag, "1"]),
            _ => words(&["--protein", "MK", flag, "1"]),
        };
        let refused = match command {
            "design" => design::parse(argv),
            "check" => derna_cli::check::parse(argv).map(|_| unreachable!()),
            _ => front::parse(argv).map(|_| unreachable!()),
        };
        let message = refused
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| panic!("{command} accepted {flag}"));
        assert!(
            !message.contains("unknown flag"),
            "{command} {flag} reads as a typo: {message}"
        );
    }

    // And a flag no command takes is refused rather than ignored.
    assert!(design::parse(words(&["--protein", "MK", "--lambda", "0.5", "--wat"])).is_err());
    assert!(front::parse(words(&["--protein", "MK", "--wat"])).is_err());
    assert!(show::parse(words(&["--wat"])).is_err());
}

/// The modular front's flag was spelled `--structurally-optimal` before it was
/// `--modular`. No help names the old spelling, and both commands that took it
/// refuse it with the spelling that replaced it, rather than as a typo or by
/// quietly doing the same thing.
#[test]
fn the_old_spelling_of_modular_is_refused_with_the_new_one() {
    for (command, help) in [("front", front::help()), ("show", show::help())] {
        assert!(
            help.contains("--modular"),
            "{command} --help omits --modular"
        );
        assert!(
            !help.contains("--structurally-optimal"),
            "{command} --help still names --structurally-optimal"
        );
    }
    let front_refusal = front::parse(words(&["--protein", "MK", "--structurally-optimal"]))
        .unwrap_err()
        .to_string();
    let show_refusal = show::parse(words(&["r.jsonl", "--structurally-optimal"]))
        .unwrap_err()
        .to_string();
    for (command, e) in [("front", front_refusal), ("show", show_refusal)] {
        assert!(
            e.contains("--structurally-optimal is now --modular"),
            "{command}: {e}"
        );
    }
    // The new spelling is the one each command takes.
    assert!(front::parse(words(&["--protein", "MK", "--modular"])).is_ok_and(|a| a.modular));
    assert!(show::parse(words(&["r.jsonl", "--modular"])).is_ok_and(|a| a.selection.modular));
}

#[test]
fn the_help_text_states_the_unit_once_and_everywhere() {
    // Energies are kcal/mol on every flag; a flag in another unit is the drift
    // this command exists to end, so the help says the unit where it takes one.
    let design_help = design::help();
    assert!(
        design_help.contains("(0, 1)"),
        "the weight's range is unstated"
    );
    assert!(
        show::help().contains("kcal/mol"),
        "show's --max-mfe does not state its unit"
    );
}
