//! Stdin-driven CLI for `rna_mfe::evaluate`. Reads `(sequence, structure)`
//! pairs as two consecutive non-blank lines from stdin and prints
//! `<energy_dekacal>` per pair to stdout. Empty lines and lines starting
//! with `#` are ignored when looking for the next sequence; once a
//! sequence is read, the immediately-following non-blank, non-comment
//! line is taken as its structure.
//!
//! Usage: `cargo run --release --example mfe_eval -p rna-mfe < seq_struct.txt`.

use std::io::{self, BufRead, Write};

use rna_mfe::{evaluate, Params};

fn main() -> io::Result<()> {
    let params = Params::load_default().expect("load Turner 2004");
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut pending_seq: Option<String> = None;
    for line in stdin.lock().lines() {
        let raw = line?;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        match pending_seq.take() {
            None => pending_seq = Some(trimmed.to_string()),
            Some(seq) => {
                let energy = evaluate(&params, seq.as_bytes(), trimmed)
                    .unwrap_or_else(|e| panic!("evaluate({seq:?}, {trimmed:?}) failed: {e}"));
                writeln!(out, "{}", energy.dekacal())?;
            }
        }
    }
    Ok(())
}
