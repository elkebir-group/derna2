//! The `derna` command line: argument parsing, the result format and file I/O
//! over the `derna_lib` library.
//!
//! Subcommands:
//! - `design`: the best design at given trade-off weights λ.
//! - `front`: the exact Pareto front of (MFE, CAI).
//! - `check`: certify a result's points against their own free folds.
//! - `pool`: join several certified fronts of one molecule.
//! - `show`: render a result file for reading.
//! - `pick`: one design out of a result file, as a sequence to order.

pub mod args;
pub mod batch;
pub mod check;
pub mod constraint;
pub mod design;
pub mod free_check;
pub mod front;
pub mod given;
pub mod input;
pub mod metrics;
pub mod modular_front;
pub mod molecule;
pub mod pick;
pub mod plot;
pub mod pool;
pub mod result;
pub mod sequence_constraint;
pub mod show;
pub mod summary;

use std::path::Path;

/// Every fallible step reports a message for the user.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Top-level help.
pub const HELP: &str = "\
derna — exact Pareto-optimal mRNA design over (MFE, CAI)

Usage: derna <command> [options]

Commands
  design    the best design at given trade-off weights λ
  front     the exact Pareto front of (MFE, CAI)
  check     certify a result's points against their own free folds
  pool      join several certified fronts of one molecule
  show      render a result file for reading
  pick      one design out of a result file, as FASTA or its JSONL row

`derna <command> --help` lists a command's options, and `derna --version` says
which version wrote a result file.
";

/// Run the command line `words` (without the program name).
pub fn run(words: Vec<String>) -> Result<()> {
    let mut words = words.into_iter();
    let command = words.next();
    let rest: Vec<String> = words.collect();
    let wants_help = rest.iter().any(|w| w == "--help" || w == "-h");
    match command.as_deref() {
        None | Some("--help" | "-h" | "help") => {
            print!("{HELP}");
            Ok(())
        }
        // A result file records the version that wrote it, so the binary has
        // to be able to say which one it is.
        Some("--version" | "-V" | "version") => {
            println!("derna {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("design") if wants_help => {
            print!("{}", design::help());
            Ok(())
        }
        Some("design") => batch::run_design(&design::parse(rest)?),
        Some("front") if wants_help => {
            print!("{}", front::help());
            Ok(())
        }
        Some("front") => batch::run_front(&front::parse(rest)?),
        Some("check") if wants_help => {
            print!("{}", check::help());
            Ok(())
        }
        Some("check") => {
            let args = check::parse(rest)?;
            summary::start();
            check::run(&args)?;
            let out = result::destination(args.shared.out.as_deref()).map(Path::to_path_buf);
            summary::print_tables(&[out], args.shared.quiet);
            Ok(())
        }
        Some("pool") if wants_help => {
            print!("{}", pool::help());
            Ok(())
        }
        Some("pool") => {
            let args = pool::parse(rest)?;
            pool::run(&args)?;
            summary::print_tables(&pool::outputs(&args).map(Some), args.quiet);
            Ok(())
        }
        Some("show") if wants_help => {
            print!("{}", show::help());
            Ok(())
        }
        Some("show") => show::run(&show::parse(rest)?),
        Some("pick") if wants_help => {
            print!("{}", pick::help());
            Ok(())
        }
        Some("pick") => pick::run(&pick::parse(rest)?),
        Some(other) => Err(format!("unknown command {other:?}; see `derna --help`").into()),
    }
}
