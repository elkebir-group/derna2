//! Reading the molecule's inputs: the protein, the fixed UTRs and the codon
//! usage table.
//!
//! `--protein`, `--utr5` and `--utr3` share one rule. A value is `-` (stdin), a
//! path to an existing one-record FASTA file, or a literal sequence. A value
//! that is both an existing file name and a valid literal is refused, since
//! either reading could be the intended one; `./NAME` names the file.

use std::io::Read;
use std::path::{Path, PathBuf};

use derna_codon_dp::CodonUsage;

use crate::Result;

/// The 20 amino-acid one-letter codes.
const AMINO_ACIDS: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
/// IUPAC nucleotide codes that stand for more than one base.
const DEGENERATE: &[u8] = b"RYSWKMBDHVN";
/// The codon usage tables bundled into the binary, by the name `--codon-usage`
/// takes for each. All are from the Kazusa Codon Usage Database;
/// `PROVENANCE.md` beside the files records each one's source entry, corpus
/// size and snapshot. The order is the order the help text and a refusal list
/// them in, so `human` stays first.
const BUNDLED_CODON_USAGE: &[(&str, &str)] = &[
    (
        "human",
        include_str!("../../derna-codon-dp/data/codon_usage_human.csv"),
    ),
    (
        "mouse",
        include_str!("../../derna-codon-dp/data/codon_usage_mouse.csv"),
    ),
    (
        "cho",
        include_str!("../../derna-codon-dp/data/codon_usage_cho.csv"),
    ),
    (
        "ecoli",
        include_str!("../../derna-codon-dp/data/codon_usage_ecoli.csv"),
    ),
    (
        "yeast",
        include_str!("../../derna-codon-dp/data/codon_usage_yeast.csv"),
    ),
];

/// A sequence read from a flag, with where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedSequence {
    /// The FASTA record name, or `None` for a literal or a nameless record.
    pub name: Option<String>,
    /// The sequence, upper-case; nucleotides with `T` read as `U`.
    pub sequence: String,
}

/// Where a flag's value is read from.
enum Source {
    Stdin,
    File(PathBuf),
    Literal(String),
}

/// Which alphabet a sequence flag accepts.
#[derive(Clone, Copy)]
enum Alphabet {
    Protein,
    Nucleotide,
}

/// Read `--protein`: a FASTA file, a literal amino-acid sequence, or `-`.
pub fn read_protein(value: &str) -> Result<NamedSequence> {
    read_sequence("--protein", value, Alphabet::Protein)
}

/// Read `--utr5` or `--utr3`: a FASTA file, a literal nucleotide sequence, or
/// `-`.
pub fn read_utr(flag: &str, value: &str) -> Result<NamedSequence> {
    read_sequence(flag, value, Alphabet::Nucleotide)
}

/// Refuse more than one flag reading stdin, since only one can.
pub fn refuse_shared_stdin(values: &[(&str, Option<&str>)]) -> Result<()> {
    let readers: Vec<&str> = values
        .iter()
        .filter(|(_, v)| *v == Some("-"))
        .map(|(flag, _)| *flag)
        .collect();
    if readers.len() > 1 {
        return Err(format!(
            "{} all read stdin; at most one flag can",
            readers.join(", ")
        )
        .into());
    }
    Ok(())
}

fn read_sequence(flag: &str, value: &str, alphabet: Alphabet) -> Result<NamedSequence> {
    match classify(flag, value, alphabet)? {
        Source::Stdin => {
            let mut text = String::new();
            std::io::stdin().read_to_string(&mut text)?;
            parse_fasta_or_literal(flag, &text, alphabet)
        }
        Source::File(path) => {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("{flag}: cannot read {}: {e}", path.display()))?;
            parse_fasta(flag, &text, alphabet)
        }
        Source::Literal(seq) => Ok(NamedSequence {
            name: None,
            sequence: seq,
        }),
    }
}

fn classify(flag: &str, value: &str, alphabet: Alphabet) -> Result<Source> {
    decide(flag, value, alphabet, Path::new(value).is_file())
}

/// The path-or-literal rule, given whether `value` names an existing file.
fn decide(flag: &str, value: &str, alphabet: Alphabet, is_file: bool) -> Result<Source> {
    if value == "-" {
        return Ok(Source::Stdin);
    }
    match (is_file, validate(flag, value, alphabet)) {
        (true, Ok(_)) => Err(format!(
            "{flag}: {value:?} is both an existing file and a valid sequence; \
             write ./{value} for the file"
        )
        .into()),
        (true, Err(_)) => Ok(Source::File(PathBuf::from(value))),
        (false, Ok(seq)) => Ok(Source::Literal(seq)),
        (false, Err(e)) => Err(e),
    }
}

/// Stdin may hold FASTA or a bare sequence.
fn parse_fasta_or_literal(flag: &str, text: &str, alphabet: Alphabet) -> Result<NamedSequence> {
    if text.trim_start().starts_with('>') {
        return parse_fasta(flag, text, alphabet);
    }
    let joined: String = text.split_whitespace().collect();
    Ok(NamedSequence {
        name: None,
        sequence: validate(flag, &joined, alphabet)?,
    })
}

/// A FASTA text's records as `(header, sequence)`, the sequence unchecked.
fn split_fasta(flag: &str, text: &str) -> Result<Vec<(String, String)>> {
    let mut records: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('>') {
            records.push((header.trim().to_string(), String::new()));
        } else if !line.is_empty() {
            match records.last_mut() {
                Some((_, seq)) => seq.push_str(line),
                None => {
                    return Err(format!("{flag}: FASTA text before the first '>' header").into())
                }
            }
        }
    }
    Ok(records)
}

/// The records of a `--protein` FASTA file holding more than one, each
/// sequence as written and not yet checked; `None` for every other value,
/// which [`read_protein`] reads as it always has.
///
/// The sequences are left unchecked so that a record with a residue that
/// cannot be designed fails on its own, in its turn of a batch, rather than
/// refusing the whole file.
pub fn protein_records(value: &str) -> Result<Option<Vec<NamedSequence>>> {
    let Ok(Source::File(path)) = classify("--protein", value, Alphabet::Protein) else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("--protein: cannot read {}: {e}", path.display()))?;
    let records = split_fasta("--protein", &text)?;
    if records.len() < 2 {
        return Ok(None);
    }
    Ok(Some(
        records
            .into_iter()
            .map(|(name, sequence)| NamedSequence {
                name: (!name.is_empty()).then_some(name),
                sequence,
            })
            .collect(),
    ))
}

/// One record of a multi-record `--protein` file, checked as a one-record
/// file's sequence is.
pub fn checked_protein(record: &NamedSequence) -> Result<NamedSequence> {
    Ok(NamedSequence {
        name: record.name.clone(),
        sequence: validate("--protein", &record.sequence, Alphabet::Protein)?,
    })
}

/// Exactly one FASTA record.
fn parse_fasta(flag: &str, text: &str, alphabet: Alphabet) -> Result<NamedSequence> {
    let mut records = split_fasta(flag, text)?;
    match records.len() {
        0 => Err(format!("{flag}: the file holds no FASTA record").into()),
        1 => {
            let (name, seq) = records.remove(0);
            Ok(NamedSequence {
                name: (!name.is_empty()).then_some(name),
                sequence: validate(flag, &seq, alphabet)?,
            })
        }
        n => {
            let names: Vec<&str> = records.iter().map(|(name, _)| name.as_str()).collect();
            Err(format!(
                "{flag}: the file holds {n} FASTA records ({}); give exactly one",
                names.join(", ")
            )
            .into())
        }
    }
}

/// The sequence upper-cased (nucleotides with `T` as `U`), or a refusal naming
/// the first offending position (1-based) and letter.
fn validate(flag: &str, seq: &str, alphabet: Alphabet) -> Result<String> {
    if seq.is_empty() {
        return Err(format!("{flag}: empty sequence").into());
    }
    let mut out = String::with_capacity(seq.len());
    for (i, c) in seq.bytes().enumerate() {
        let u = c.to_ascii_uppercase();
        out.push(char::from(check_letter(flag, i + 1, u, alphabet)?));
    }
    Ok(out)
}

fn check_letter(flag: &str, pos: usize, u: u8, alphabet: Alphabet) -> Result<u8> {
    let letter = char::from(u);
    match alphabet {
        Alphabet::Protein if AMINO_ACIDS.contains(&u) => Ok(u),
        // One way to ask for a stop codon: the flag, which places it in the
        // constraint's coding sequence and out of CAI.
        Alphabet::Protein if u == b'*' => Err(format!(
            "{flag}: position {pos} is a stop ('*'); give the protein without it \
             and ask for the codon with --stop UAA|UAG|UGA"
        )
        .into()),
        Alphabet::Protein => Err(format!(
            "{flag}: position {pos} is {letter:?}, not one of the 20 amino acids"
        )
        .into()),
        Alphabet::Nucleotide if b"ACGU".contains(&u) => Ok(u),
        Alphabet::Nucleotide if u == b'T' => Ok(b'U'),
        Alphabet::Nucleotide if DEGENERATE.contains(&u) => Err(format!(
            "{flag}: position {pos} is the degenerate code {letter:?}; \
             designed UTR positions are not supported"
        )
        .into()),
        Alphabet::Nucleotide => {
            Err(format!("{flag}: position {pos} is {letter:?}, not a nucleotide").into())
        }
    }
}

/// A codon usage table with the text it was read from, so a result file can
/// carry the exact table its CAI was scored against.
#[derive(Debug, Clone)]
pub struct CodonTable {
    /// `human`, or the path given.
    pub source: String,
    /// The parsed table.
    pub usage: CodonUsage,
    /// `(amino acid, codon, fraction)` rows, as the file lists them.
    pub rows: Vec<(String, String, f64)>,
}

/// The bundled table names, as a refusal and the help text list them.
pub fn bundled_codon_usage_names() -> String {
    let names: Vec<&str> = BUNDLED_CODON_USAGE.iter().map(|(name, _)| *name).collect();
    names.join(", ")
}

/// Read `--codon-usage`: a bundled table's name, or a path in the
/// `amino_acid,codon,fraction` format.
///
/// A value that is not a bundled name is read as a path, and a value that is
/// neither is refused in one message naming both — a misspelled name and a
/// mistyped path are indistinguishable here, so the refusal gives the reader
/// what each reading would have needed rather than guessing which was meant.
pub fn read_codon_usage(value: Option<&str>) -> Result<CodonTable> {
    let name = value.unwrap_or("human");
    if let Some((_, csv)) = BUNDLED_CODON_USAGE.iter().find(|(n, _)| *n == name) {
        return codon_table_from_csv(name.to_string(), csv);
    }
    let text = std::fs::read_to_string(name).map_err(|e| {
        format!(
            "--codon-usage: {name:?} is not a bundled table ({}) and cannot be read as a file: {e}",
            bundled_codon_usage_names()
        )
    })?;
    codon_table_from_csv(name.to_string(), &text)
}

/// Parse a codon usage table from its CSV text.
///
/// The rows are kept beside the parsed table because a result file embeds
/// them: a later command scores CAI against exactly the table a run used, so
/// a row that cannot be read is a refusal rather than a gap. A `NaN` here
/// would reach the file as `null`, since that is what JSON has for it, and a
/// null fraction would read as a table with a hole in it.
pub fn codon_table_from_csv(source: String, text: &str) -> Result<CodonTable> {
    let usage = CodonUsage::load_from_str(text).map_err(|e| format!("--codon-usage: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let fraction = fields.get(2).and_then(|x| x.parse::<f64>().ok());
        match (fields.first(), fields.get(1), fraction) {
            (Some(aa), Some(codon), Some(fraction)) if fraction.is_finite() => {
                rows.push(((*aa).to_string(), (*codon).to_string(), fraction));
            }
            _ => {
                let what = "`amino_acid,codon,fraction`";
                return Err(format!(
                    "--codon-usage: line {} of {source} is not {what}: {line:?}",
                    i + 1
                )
                .into());
            }
        }
    }
    Ok(CodonTable {
        source,
        usage,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_literal_protein_is_upper_cased() {
        assert_eq!(read_protein("mkv").unwrap().sequence, "MKV");
    }

    #[test]
    fn a_stop_in_the_protein_points_at_the_stop_flag() {
        let e = read_protein("MK*").unwrap_err().to_string();
        assert!(e.contains("position 3") && e.contains("--stop"), "{e}");
    }

    #[test]
    fn an_unknown_letter_is_refused_by_position() {
        let e = read_protein("MKBV").unwrap_err().to_string();
        assert!(e.contains("position 3") && e.contains("'B'"), "{e}");
    }

    #[test]
    fn a_utr_reads_t_as_u_and_refuses_degenerate_codes() {
        assert_eq!(read_utr("--utr5", "acgt").unwrap().sequence, "ACGU");
        let e = read_utr("--utr5", "ACNU").unwrap_err().to_string();
        assert!(
            e.contains("designed UTR positions are not supported"),
            "{e}"
        );
    }

    #[test]
    fn a_fasta_file_must_hold_one_record() {
        let e = parse_fasta("--protein", ">a\nMK\n>b\nMV\n", Alphabet::Protein)
            .unwrap_err()
            .to_string();
        assert!(e.contains("2 FASTA records (a, b)"), "{e}");
        let one = parse_fasta("--protein", ">p1 desc\nMK\nVL\n", Alphabet::Protein).unwrap();
        assert_eq!(one.sequence, "MKVL");
        assert_eq!(one.name.as_deref(), Some("p1 desc"));
    }

    #[test]
    fn a_name_that_is_a_file_and_a_sequence_is_refused() {
        let nt = Alphabet::Nucleotide;
        let e = decide("--utr3", "ACGU", nt, true).map(|_| ()).unwrap_err();
        assert!(e.to_string().contains("write ./ACGU"), "{e}");
        assert!(matches!(
            decide("--utr3", "./ACGU", nt, true),
            Ok(Source::File(_))
        ));
        assert!(matches!(
            decide("--utr3", "ACGU", nt, false),
            Ok(Source::Literal(_))
        ));
        assert!(decide("--utr3", "utr.fa", nt, false).is_err());
    }

    #[test]
    fn a_fasta_file_is_read_through_the_flag() {
        let path = std::env::temp_dir().join(format!("derna-input-{}.fa", std::process::id()));
        std::fs::write(&path, ">hbb5\nacau\nuugc\n").unwrap();
        let read = read_utr("--utr5", path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        let read = read.unwrap();
        assert_eq!(read.sequence, "ACAUUUGC");
        assert_eq!(read.name.as_deref(), Some("hbb5"));
    }

    #[test]
    fn two_stdin_readers_are_refused() {
        let e = refuse_shared_stdin(&[("--protein", Some("-")), ("--utr5", Some("-"))]);
        assert!(e.is_err());
        assert!(refuse_shared_stdin(&[("--protein", Some("-")), ("--utr5", None)]).is_ok());
    }

    #[test]
    fn the_human_table_has_all_sixty_one_sense_codons() {
        let t = read_codon_usage(None).unwrap();
        assert_eq!(t.source, "human");
        assert_eq!(t.rows.len(), 61);
        assert!(t.rows.iter().all(|(_, _, f)| f.is_finite()));
    }

    #[test]
    fn a_row_that_cannot_be_read_is_refused_rather_than_recorded_as_null() {
        // A table a run scored CAI against is embedded in its result, so a row
        // that does not parse must not become a `null` fraction in the file —
        // which is what JSON does with a NaN.
        //
        // The library's own parser refuses this shape first, so the row scan
        // here is a backstop rather than the only constraint. The test asserts the
        // outcome that matters — a refusal naming the line — not which layer
        // produced it, because a change of layer is not a change of behaviour.
        let csv = "amino_acid,codon,fraction\nL,CUU,0.13\nL,CUC\n";
        let e = codon_table_from_csv("t.csv".into(), csv)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("--codon-usage"), "{e}");
        assert!(e.contains("line 3"), "{e}");
    }
}
