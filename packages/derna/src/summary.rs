//! The table a computing command prints at the end of a run: `show`'s default
//! rendering of each result file it wrote, on **stderr**, so stdout keeps
//! only the result itself. `--quiet` turns it off.

use std::path::{Path, PathBuf};

use crate::{result, show};

/// Print the default table of each of `outputs` to stderr: a file read back
/// from its path, or `None` for the result this thread wrote to stdout.
///
/// A file that is not a readable result — `/dev/null`, a stop codon no design
/// reached, a run that wrote nothing — is passed over rather than reported,
/// since the run itself has already said what it wrote.
pub fn print_tables(outputs: &[Option<PathBuf>], quiet: bool) {
    let copy = result::take_stdout_copy();
    if quiet {
        return;
    }
    for output in outputs {
        let rendered = match output {
            Some(path) if is_regular(path) => show::read(Some(path)).ok(),
            Some(_) => None,
            None => std::str::from_utf8(&copy)
                .ok()
                .and_then(|text| show::parse_lines_of(text, "stdout").ok()),
        };
        let Some(rendered) = rendered else {
            continue;
        };
        let name = output
            .as_ref()
            .map_or_else(|| "stdout".to_string(), |p| p.display().to_string());
        let mut lines = vec![String::new(), format!("wrote {name}:")];
        lines.extend(show::default_table(&rendered));
        eprintln!("{}", lines.join("\n"));
    }
}

/// Forget what this thread has written to stdout so far, so the next run's
/// table renders only its own result.
pub fn start() {
    result::take_stdout_copy();
}

fn is_regular(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file())
}
