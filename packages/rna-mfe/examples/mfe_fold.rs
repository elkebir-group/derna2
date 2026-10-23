//! Stdin-driven CLI for `rna_mfe::fold`. Reads one RNA sequence per
//! line from stdin and prints `<structure>\t<energy_dekacal>` per line
//! to stdout. Empty lines and lines starting with `#` are ignored.
//!
//! Usage: `cargo run --release --example mfe_fold -p rna-mfe < seqs.txt`.

use std::io::{self, BufRead, Write};

use rna_mfe::{fold, Params};

fn main() -> io::Result<()> {
    let params = Params::load_default().expect("load Turner 2004");
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let seq = line?;
        let trimmed = seq.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let (structure, energy) = fold(&params, trimmed.as_bytes())
            .unwrap_or_else(|e| panic!("fold({trimmed:?}) failed: {e}"));
        writeln!(out, "{}\t{}", structure, energy.dekacal())?;
    }
    Ok(())
}
