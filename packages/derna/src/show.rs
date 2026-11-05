//! `derna show`: render a result file for reading. It folds nothing: every
//! energy, CAI and verdict it prints comes from the file.
//!
//! What it derives is sequence arithmetic. A file written before points
//! carried their sequence metrics (`gc`, `gc_cds`, `u_cds`, `gc_window_min`,
//! `gc_window_max`, `max_homopolymer`) gets them from each point's `sequence`
//! and the coding window its header implies, computed as a run computes them
//! ([`crate::metrics`]); `--cds-only` cuts the same window out; `--dna` spells
//! U as T; and `--plot` draws the selected points.
//!
//! The default rendering is the header, a summary, and one row per point.
//! Filters select points; `--tsv` and `--fasta` change the shape of what is
//! written rather than what is selected.

use std::io::{Read, Write};
use std::path::PathBuf;

use serde_json::Value;

use crate::args::Flags;
use crate::modular_front;
use crate::Result;

/// The filters `show` and `pick` share: which points of a file are selected.
#[derive(Debug, Default, Clone)]
pub struct Selection {
    pub min_cai: Option<f64>,
    pub max_mfe: Option<f64>,
    pub modular: bool,
    pub min_gc: Option<f64>,
    pub max_gc: Option<f64>,
    pub max_homopolymer: Option<usize>,
}

/// Help lines for the [`Selection`] flags.
pub const SELECTION_HELP: &str = "  \
--min-cai X               only points with CAI ≥ X
  --max-mfe E               only points with MFE ≤ E (kcal/mol)
  --modular                 only the modular points, those that pass the
                            check
  --min-gc X                only points whose GC fraction is ≥ X
  --max-gc X                only points whose GC fraction is ≤ X
  --max-homopolymer N       only points whose longest single-nucleotide run
                            is at most N
";

/// One filter of a [`Selection`]: the flag that asks for it, and whether a
/// point passes it.
type Filter<'a> = (&'static str, Box<dyn Fn(&Value) -> bool + 'a>);

impl Selection {
    /// Consume `flag` (and its value) if it is a selection flag; `false` if not.
    pub fn take(&mut self, flag: &str, flags: &mut Flags) -> Result<bool> {
        match flag {
            "--modular" => self.modular = true,
            "--min-cai" => self.min_cai = Some(flags.parsed(flag)?),
            "--max-mfe" => self.max_mfe = Some(flags.parsed(flag)?),
            "--min-gc" => self.min_gc = Some(flags.parsed(flag)?),
            "--max-gc" => self.max_gc = Some(flags.parsed(flag)?),
            "--max-homopolymer" => self.max_homopolymer = Some(flags.parsed(flag)?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The filters asked for, in the order a refusal names the first that
    /// emptied the selection.
    fn filters(&self) -> Vec<Filter<'_>> {
        let at_least = |field: &'static str, x: f64| -> Box<dyn Fn(&Value) -> bool> {
            Box::new(move |p: &Value| p[field].as_f64().is_some_and(|v| v >= x))
        };
        let at_most = |field: &'static str, x: f64| -> Box<dyn Fn(&Value) -> bool> {
            Box::new(move |p: &Value| p[field].as_f64().is_some_and(|v| v <= x))
        };
        let mut out: Vec<Filter<'_>> = Vec::new();
        if self.modular {
            out.push((
                "--modular",
                Box::new(|p: &Value| p["modular"] == Value::Bool(true)),
            ));
        }
        if let Some(x) = self.min_cai {
            out.push(("--min-cai", at_least("cai", x)));
        }
        if let Some(e) = self.max_mfe {
            out.push(("--max-mfe", at_most("mfe", e)));
        }
        if let Some(x) = self.min_gc {
            out.push(("--min-gc", at_least("gc", x)));
        }
        if let Some(x) = self.max_gc {
            out.push(("--max-gc", at_most("gc", x)));
        }
        if let Some(n) = self.max_homopolymer {
            out.push(("--max-homopolymer", at_most("max_homopolymer", n as f64)));
        }
        out
    }

    /// Whether a point passes every filter. A point is read with its metrics
    /// filled in ([`crate::metrics::filled`]).
    pub fn keeps(&self, p: &Value) -> bool {
        self.filters().iter().all(|(_, keeps)| keeps(p))
    }

    /// The points that pass, with their 1-based index in the file, or the
    /// first filter after which none was left.
    pub fn select<'p>(
        &self,
        points: &'p [Value],
    ) -> std::result::Result<Vec<(usize, &'p Value)>, &'static str> {
        let mut kept: Vec<(usize, &Value)> =
            points.iter().enumerate().map(|(i, p)| (i + 1, p)).collect();
        for (flag, keeps) in self.filters() {
            kept.retain(|(_, p)| keeps(p));
            if kept.is_empty() && !points.is_empty() {
                return Err(flag);
            }
        }
        Ok(kept)
    }

    /// Whether `--modular` was asked of a file no point of which was checked.
    pub fn modular_unanswerable(&self, points: &[Value]) -> bool {
        self.modular && !points.iter().any(|p| p.get("modular").is_some())
    }
}

/// How a FASTA record is spelled.
#[derive(Debug, Default, Clone, Copy)]
pub struct FastaStyle {
    /// T for U.
    pub dna: bool,
    /// The coding sequence alone, start codon through stop codon.
    pub cds_only: bool,
}

impl FastaStyle {
    /// Consume `flag` if it is a FASTA style flag; `false` if not.
    pub fn take(&mut self, flag: &str) -> bool {
        match flag {
            "--dna" => self.dna = true,
            "--cds-only" => self.cds_only = true,
            _ => return false,
        }
        true
    }

    fn asked(&self) -> bool {
        self.dna || self.cds_only
    }
}

/// Help lines for the [`FastaStyle`] flags.
pub const FASTA_STYLE_HELP: &str = "  \
--dna                     write T for U
  --cds-only                write the coding sequence alone, start codon
                            through stop codon (through the last codon when
                            there is none), without the UTRs
";

/// `derna show` as parsed.
#[derive(Debug, Default)]
pub struct ShowArgs {
    /// The result file, or `None` for stdin.
    pub path: Option<PathBuf>,
    pub sequences: bool,
    pub selection: Selection,
    pub tsv: bool,
    pub fasta: bool,
    pub style: FastaStyle,
    /// Also draw the selected points as SVG here.
    pub plot: Option<PathBuf>,
    /// The plot's title, in place of the record's name.
    pub plot_title: Option<String>,
}

/// Help text for `derna show`.
pub fn help() -> String {
    format!(
        "derna show — render a result file for reading\n\n\
         Usage: derna show RESULT.jsonl|- [options]\n\n\
         Selection\n\
         {SELECTION_HELP}\n\
         Output\n  \
         --sequences               each point's sequence, structure and sequence\n                            \
         metrics under its row, or as more columns under\n                            \
         --tsv\n  \
         --tsv                     tab-separated, the same columns\n  \
         --fasta                   the selected points' sequences as FASTA\n\
         {FASTA_STYLE_HELP}  \
         --plot PATH.svg           also draw the selected points, CAI against MFE\n  \
         --plot-title TEXT         the plot's title [default: the record's name]\n"
    )
}

/// Parse the words after `show`.
pub fn parse(words: impl IntoIterator<Item = String>) -> Result<ShowArgs> {
    let mut flags = Flags::new(words);
    let mut args = ShowArgs::default();
    while let Some(word) = flags.next_word() {
        if args.selection.take(&word, &mut flags)? || args.style.take(&word) {
            continue;
        }
        match word.as_str() {
            "--sequences" => args.sequences = true,
            "--structurally-optimal" => return Err(crate::args::renamed(&word).into()),
            "--tsv" => args.tsv = true,
            "--fasta" => args.fasta = true,
            "--plot" => args.plot = Some(PathBuf::from(flags.value(&word)?)),
            "--plot-title" => args.plot_title = Some(flags.value(&word)?),
            flag if flag.starts_with("--") => {
                return Err(format!("derna show: unknown flag {flag:?} (see --help)").into())
            }
            "-" if args.path.is_none() => args.path = Some(PathBuf::from("-")),
            path if args.path.is_none() => args.path = Some(PathBuf::from(path)),
            extra => {
                return Err(format!("derna show: one result file at a time, got {extra:?}").into())
            }
        }
    }
    if args.tsv && args.fasta {
        return Err("derna show: --tsv and --fasta ask for two different outputs".into());
    }
    // `--fasta` is the sequences, so asking for them again says nothing;
    // under `--tsv` the request is meaningful and adds columns.
    if args.sequences && args.fasta {
        return Err("derna show: --fasta already writes the sequences".into());
    }
    if args.style.asked() && !args.fasta {
        return Err(
            "derna show: --dna and --cds-only spell FASTA records, so they need --fasta".into(),
        );
    }
    Ok(args)
}

/// A result file read back: its header and its points.
#[derive(Debug)]
pub struct Rendered {
    pub header: Value,
    pub points: Vec<Value>,
}

/// Read a result file, refusing one that is not a `derna` result.
pub fn read(path: Option<&std::path::Path>) -> Result<Rendered> {
    let text = match path.map(|p| p.to_str().unwrap_or("-")) {
        None | Some("-") => {
            let mut text = String::new();
            std::io::stdin().read_to_string(&mut text)?;
            text
        }
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("cannot read {p}: {e}"))?,
    };
    let name = match path {
        Some(p) if p.to_str() != Some("-") => p.display().to_string(),
        _ => "the result on stdin".to_string(),
    };
    parse_lines_of(&text, &name)
}

/// Split a result file's text into its header and points.
pub fn parse_lines(text: &str) -> Result<Rendered> {
    parse_lines_of(text, "this result")
}

/// [`parse_lines`] for the file `name`, which a refusal names.
pub fn parse_lines_of(text: &str, name: &str) -> Result<Rendered> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Value = match lines.next() {
        Some(first) => crate::result::parse_line(first).map_err(|e| format!("header line: {e}"))?,
        None => return Err("the result file is empty".into()),
    };
    if header["format"] != "derna-result" {
        return Err("not a derna result file: the first line is not a derna header".into());
    }
    // A reader that ignored the version would render a later format's fields
    // as absent — a point would look unchecked rather than unreadable.
    match header["format_version"].as_u64() {
        Some(v) if v <= u64::from(crate::result::FORMAT_VERSION) => {}
        Some(v) => {
            return Err(format!(
                "{name} is derna result format version {v}, and this derna reads up to \
                 version {}: the file is intact, only newer than this build. Use a newer \
                 derna, one that reads format version {v}, to read it",
                crate::result::FORMAT_VERSION
            )
            .into())
        }
        None => return Err("the header does not say which format version it is in".into()),
    }
    let points = lines
        .map(|l| crate::result::parse_line(l).map_err(|e| format!("point line: {e}").into()))
        .collect::<Result<Vec<Value>>>()?;
    Ok(Rendered { header, points })
}

/// Whether a filter was asked for that this file cannot answer.
///
/// `--modular` on a file nobody has checked selects nothing, and
/// "0 shown" reads as *no point passed* rather than *no point was asked*. The
/// two are different answers and a caller acts differently on them.
fn unanswerable(args: &ShowArgs, points: &[Value]) -> Option<String> {
    args.selection.modular_unanswerable(points).then(|| {
        "  note: --modular selects nothing here because no point in this file \
         carries a check — `derna check` adds them"
            .to_string()
    })
}

/// Every point of `rendered` with its sequence metrics, derived from its
/// sequence where the file does not carry them.
pub fn filled_points(rendered: &Rendered) -> Vec<Value> {
    rendered
        .points
        .iter()
        .map(|p| crate::metrics::filled(p, &rendered.header))
        .collect()
}

/// The points `args` selects, each with its 1-based index in the file.
fn selected<'p>(args: &ShowArgs, points: &'p [Value]) -> Vec<(usize, &'p Value)> {
    points
        .iter()
        .enumerate()
        .filter(|(_, p)| args.selection.keeps(p))
        .map(|(i, p)| (i + 1, p))
        .collect()
}

/// The lines `show` writes for `rendered` under `args`.
pub fn render(args: &ShowArgs, rendered: &Rendered) -> Vec<String> {
    let points = filled_points(rendered);
    let kept = selected(args, &points);
    // Three output shapes, one function each: the flags describe them that
    // way, and a single loop asking "not TSV?" three times did not.
    match (args.fasta, args.tsv) {
        (true, _) => kept
            .iter()
            .flat_map(|(i, p)| fasta_record(&rendered.header, *i, p, args.style))
            .collect(),
        (_, true) => tsv_lines(args, &kept),
        _ => table_lines(args, rendered, &points, &kept),
    }
}

/// The default rendering of `rendered`: what a run prints of the file it has
/// written.
pub fn default_table(rendered: &Rendered) -> Vec<String> {
    render(&ShowArgs::default(), rendered)
}

/// The SVG `show --plot` writes of the points `args` selects.
pub fn plot_of(args: &ShowArgs, rendered: &Rendered) -> String {
    let points = filled_points(rendered);
    let kept: Vec<&Value> = selected(args, &points)
        .into_iter()
        .map(|(_, p)| p)
        .collect();
    let title = args
        .plot_title
        .clone()
        .unwrap_or_else(|| record_name(&rendered.header));
    crate::plot::svg(&title, &kept)
}

/// One line per point, behind a header row naming the columns.
fn tsv_lines(args: &ShowArgs, kept: &[(usize, &Value)]) -> Vec<String> {
    let mut columns: Vec<&str> = TSV_COLUMNS.to_vec();
    if args.sequences {
        columns.extend(&crate::metrics::FIELDS[1..]);
        columns.extend(["sequence", "structure"]);
    }
    let mut out = vec![columns.join("\t")];
    out.extend(kept.iter().map(|(_, p)| row(p, true, args.sequences)));
    out
}

/// The header, the summary, and one row per point — with, on request, the
/// sequences and metrics under it.
fn table_lines(
    args: &ShowArgs,
    rendered: &Rendered,
    all: &[Value],
    kept: &[(usize, &Value)],
) -> Vec<String> {
    let mut out = header_lines(&rendered.header);
    out.push(summary_line(all, kept.len()));
    out.extend(epsilon_line(&rendered.header, all));
    out.extend(unanswerable(args, all));
    out.push(String::new());
    out.push(table_heading());
    for (_, p) in kept {
        out.push(row(p, false, args.sequences));
        if args.sequences {
            out.push(format!("    seq  {}", string_of(&p["sequence"])));
            out.push(format!("    dot  {}", string_of(&p["structure"])));
            out.push(format!(
                "    gc   cds {:.3}  u(cds) {:.3}  window {:.3}–{:.3}  homopolymer {}",
                number(&p["gc_cds"]),
                number(&p["u_cds"]),
                number(&p["gc_window_min"]),
                number(&p["gc_window_max"]),
                p["max_homopolymer"].as_u64().unwrap_or_default()
            ));
        }
    }
    out
}

/// The record a result is of: the first token of the protein's FASTA name,
/// or `derna` when the run was given a literal sequence.
pub fn record_name(header: &Value) -> String {
    header["protein"]["name"]
        .as_str()
        .and_then(|n| n.split_whitespace().next())
        .unwrap_or("derna")
        .to_string()
}

/// One point as a FASTA record: a header naming the record, the point's
/// 1-based `index` in its file, its energy, adaptation and GC fraction, and
/// the sequence spelled as `style` asks.
pub fn fasta_record(header: &Value, index: usize, p: &Value, style: FastaStyle) -> [String; 2] {
    let lambda = p["lambda"]
        .as_f64()
        .map_or_else(String::new, |l| format!(" lambda={l}"));
    let title = format!(
        ">{} point={index} mfe={:.2} cai={:.4} gc={:.2}{lambda}",
        record_name(header),
        number(&p["mfe"]),
        number(&p["cai"]),
        number(&p["gc"]),
    );
    let whole = string_of(&p["sequence"]);
    let mut sequence = if style.cds_only {
        let cds = crate::metrics::coding_window(header);
        whole
            .get(cds.start.min(whole.len())..cds.end.min(whole.len()))
            .unwrap_or_default()
            .to_string()
    } else {
        whole
    };
    if style.dna {
        sequence = sequence.replace('U', "T");
    }
    [title, sequence]
}

fn header_lines(h: &Value) -> Vec<String> {
    let protein = &h["protein"];
    let residues = string_of(&protein["sequence"]).len();
    let mut lines = vec![format!(
        "{} — {} residue(s){}",
        string_of(&h["command"]),
        residues,
        // A FASTA header's first token is its identifier; the rest is a
        // description, and the file keeps it whole either way.
        match protein["name"]
            .as_str()
            .and_then(|n| n.split_whitespace().next())
        {
            Some(id) => format!(" of {id}"),
            None => String::new(),
        }
    )];
    lines.push(format!(
        "  5′ UTR {}   3′ UTR {}   stop {}",
        flank(&h["utr5"]),
        flank(&h["utr3"]),
        h["stop"].as_str().unwrap_or("none")
    ));
    lines.push(format!(
        "  constraint {}   access {}",
        match h["forbid"].as_array() {
            Some(a) if a.is_empty() => "none".to_string(),
            Some(a) => a.iter().map(string_of).collect::<Vec<_>>().join(","),
            None => "none".to_string(),
        },
        match h["access"].as_array() {
            Some(w) if w.len() == 2 => format!("{},{}", number(&w[0]), number(&w[1])),
            _ => "off".to_string(),
        }
    ));
    lines.push(format!(
        "  codon usage {}   derna {}   {:.2}s   peak {}",
        string_of(&h["codon_usage"]["source"]),
        string_of(&h["version"]),
        number(&h["wall_s"]),
        match h["peak_rss_kb"].as_u64() {
            Some(kb) => peak_memory(kb),
            None => "n/a".to_string(),
        }
    ));
    lines
}

/// Peak resident memory in the unit that shows it: a walk that used 40 MB and
/// one that used 400 MB both read `0.0 GB`.
fn peak_memory(kb: u64) -> String {
    let mb = kb as f64 / 1024.0;
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

fn flank(f: &Value) -> String {
    match f["sequence"].as_str() {
        Some(s) => format!("{} nt", s.len()),
        None => "absent".to_string(),
    }
}

/// On a front within an epsilon above zero, the epsilon and how many points
/// sit at each gap: gaps clump, so this says which steps admitted the front.
fn epsilon_line(header: &Value, all: &[Value]) -> Option<String> {
    let epsilon = header["epsilon"].as_f64().filter(|&e| e > 0.0)?;
    let mut gaps = modular_front::GapHistogram::default();
    gaps.count(all);
    Some(format!("  within {epsilon} kcal/mol, by gap: {gaps}"))
}

fn summary_line(all: &[Value], kept: usize) -> String {
    let count = |f: &str| all.iter().filter(|p| p[f] == Value::Bool(true)).count();
    let checked = all.iter().filter(|p| p.get("modular").is_some()).count();
    let mut parts = vec![format!("{} point(s)", all.len())];
    if kept != all.len() {
        parts.push(format!("{kept} shown"));
    }
    parts.push(format!("{} on the hull", count("on_hull")));
    parts.push(match checked {
        0 => "not checked".to_string(),
        _ => format!("{} of {checked} modular", count("modular")),
    });
    format!("  {}", parts.join(", "))
}

/// The columns, in order. `--tsv` prints these names.
const TSV_COLUMNS: [&str; 8] = [
    "mfe", "cai", "gc", "gap", "modular", "on_hull", "lambda", "status",
];

fn table_heading() -> String {
    format!(
        "{:>10}  {:>8}  {:>5}  {:>7}  {:>7}  {:>5}  {:>8}  {}",
        "mfe", "cai", "gc", "gap", "modular", "hull", "lambda", "status"
    )
}

/// A point's status, as a checked result records it: `certified` for a point
/// that passed the check. A result nobody has checked carries none, and the
/// column reads `—`.
fn status(p: &Value) -> Option<&str> {
    p.get("status").and_then(Value::as_str)
}

/// The gap to print, and whether it is a bound.
///
/// A check refuted by a beam reports `gap_at_least` in place of `gap`. The
/// table marks it rather than printing a bound where a reader expects a value.
fn gap_cell(p: &Value) -> (Option<f64>, bool) {
    match p.get("gap").and_then(Value::as_f64) {
        Some(exact) => (Some(exact), false),
        None => (p.get("gap_at_least").and_then(Value::as_f64), true),
    }
}

fn row(p: &Value, tsv: bool, sequences: bool) -> String {
    let (gap, bounded) = gap_cell(p);
    let s_opt = p.get("modular").and_then(Value::as_bool);
    let lambda = p.get("lambda").and_then(Value::as_f64);
    if tsv {
        let cell = |v: Option<String>| v.unwrap_or_default();
        let metrics = crate::metrics::FIELDS[1..].iter().map(|f| match &p[*f] {
            Value::Null => String::new(),
            v => v.to_string(),
        });
        return [
            format!("{}", number(&p["mfe"])),
            format!("{}", number(&p["cai"])),
            format!("{}", number(&p["gc"])),
            cell(gap.map(|g| {
                if bounded {
                    format!(">={g}")
                } else {
                    g.to_string()
                }
            })),
            cell(s_opt.map(|b| b.to_string())),
            cell(p["on_hull"].as_bool().map(|b| b.to_string())),
            cell(lambda.map(|l| l.to_string())),
            status(p).unwrap_or_default().to_string(),
        ]
        .into_iter()
        .chain(sequences.then(|| metrics.collect::<Vec<_>>().join("\t")))
        .chain(
            sequences.then(|| [string_of(&p["sequence"]), string_of(&p["structure"])].join("\t")),
        )
        .collect::<Vec<_>>()
        .join("\t");
    }
    format!(
        "{:>10.2}  {:>8.4}  {:>5.2}  {:>7}  {:>7}  {:>5}  {:>8}  {}",
        number(&p["mfe"]),
        number(&p["cai"]),
        number(&p["gc"]),
        gap.map_or_else(
            || "—".to_string(),
            |g| if bounded {
                format!("≥{g:.2}")
            } else {
                format!("{g:.2}")
            }
        ),
        s_opt.map_or_else(
            || "—".to_string(),
            |b| if b { "yes" } else { "no" }.to_string()
        ),
        if p["on_hull"] == Value::Bool(true) {
            "yes"
        } else {
            "no"
        },
        lambda.map_or_else(|| "—".to_string(), |l| format!("{l:.6}")),
        status(p).unwrap_or("—")
    )
}

fn string_of(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_string()
}

fn number(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// Run `derna show`.
///
/// A reader is the end of a pipe, so a closed downstream (`derna show … |
/// head`) ends the rendering quietly rather than failing: there is nothing
/// left to report it to.
pub fn run(args: &ShowArgs) -> Result<()> {
    let rendered = read(args.path.as_deref())?;
    if let Some(path) = &args.plot {
        std::fs::write(path, plot_of(args, &rendered))
            .map_err(|e| format!("--plot {}: {e}", path.display()))?;
    }
    let mut out = std::io::stdout().lock();
    for line in render(args, &rendered) {
        match writeln!(out, "{line}") {
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => return Ok(()),
            other => other?,
        }
    }
    match out.flush() {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a front within an epsilon above zero gets the line.
    #[test]
    fn the_epsilon_line_is_for_a_front_within_one() {
        let points = [
            serde_json::json!({"gap": 0.0}),
            serde_json::json!({"gap": 0.2}),
        ];
        let within = |e: Value| epsilon_line(&serde_json::json!({ "epsilon": e }), &points);
        assert_eq!(within(Value::Null), None);
        assert_eq!(within(0.0.into()), None);
        assert_eq!(
            within(0.5.into()).as_deref(),
            Some("  within 0.5 kcal/mol, by gap: 0.00×1 0.20×1")
        );
    }

    const FILE: &str = concat!(
        r#"{"record":"header","format":"derna-result","format_version":1,"command":"design","#,
        r#""version":"0.1.0","protein":{"name":"p","sequence":"MKV"},"#,
        r#""utr5":{"name":null,"sequence":"ACGU"},"utr3":null,"stop":null,"#,
        r#""forbid":["5-cds"],"access":null,"codon_usage":{"source":"human","table":[]},"#,
        r#""wall_s":1.5,"peak_rss_kb":2097152}"#,
        "\n",
        r#"{"record":"point","sequence":"ACGUAUGAAGGUG","mfe":-2.5,"cai":0.95,"cai_log":-0.1,"#,
        r#""structure":".....((...)).","on_hull":true,"lambda":0.5,"free_mfe":-3.5,"#,
        r#""free_structure":"....(((...)))","gap":1.0,"modular":false,"#,
        r#""violations":[[5,13]]}"#,
        "\n",
        r#"{"record":"point","sequence":"ACGUAUGAAAGUA","mfe":-1.0,"cai":0.8,"cai_log":-0.3,"#,
        r#""structure":".............","on_hull":true,"lambda":0.9,"free_mfe":-1.0,"#,
        r#""free_structure":".............","gap":0.0,"modular":true,"#,
        r#""violations":[]}"#,
        "\n",
    );

    fn file() -> Rendered {
        parse_lines(FILE).unwrap()
    }

    fn rendered(words: &[&str]) -> Vec<String> {
        let args = parse(words.iter().map(|w| w.to_string())).unwrap();
        render(&args, &file())
    }

    #[test]
    fn the_default_rendering_names_the_inputs_and_every_point() {
        let out = rendered(&[]);
        let text = out.join("\n");
        assert!(text.contains("design — 3 residue(s) of p"), "{text}");
        assert!(text.contains("5′ UTR 4 nt   3′ UTR absent"), "{text}");
        assert!(text.contains("constraint 5-cds"), "{text}");
        assert!(
            text.contains("2 point(s), 2 on the hull, 1 of 2 modular"),
            "{text}"
        );
        assert_eq!(out.iter().filter(|l| l.contains("0.500000")).count(), 1);
        assert_eq!(out.iter().filter(|l| l.contains("0.900000")).count(), 1);
    }

    #[test]
    fn a_filter_selects_points_and_the_summary_says_how_many() {
        let out = rendered(&["--min-cai", "0.9"]).join("\n");
        assert!(out.contains("2 point(s), 1 shown"), "{out}");
        assert!(
            out.contains("0.500000") && !out.contains("0.900000"),
            "{out}"
        );
        let by_mfe = rendered(&["--max-mfe", "-2.0"]).join("\n");
        assert!(by_mfe.contains("1 shown"), "{by_mfe}");
        let checked = rendered(&["--modular"]).join("\n");
        assert!(
            checked.contains("1 shown") && checked.contains("0.900000"),
            "{checked}"
        );
    }

    #[test]
    fn sequences_go_under_their_row() {
        let out = rendered(&["--sequences"]).join("\n");
        assert!(out.contains("    seq  ACGUAUGAAGGUG"), "{out}");
        assert!(out.contains("    dot  .....((...))."), "{out}");
    }

    #[test]
    fn tsv_is_the_same_columns_with_no_header_block() {
        let out = rendered(&["--tsv"]);
        assert_eq!(
            out[0],
            "mfe\tcai\tgc\tgap\tmodular\ton_hull\tlambda\tstatus"
        );
        assert_eq!(out.len(), 3);
        // An unchecked point's status cell is empty rather than absent, so
        // every row carries the same field count. The file carries no
        // metrics, so `gc` is derived: 6 of ACGUAUGAAGGUG's 13 are G or C.
        assert_eq!(
            out[1],
            format!("-2.5\t0.95\t{}\t1\tfalse\ttrue\t0.5\t", 6.0 / 13.0)
        );
    }

    #[test]
    fn fasta_writes_the_selected_sequences() {
        let out = rendered(&["--fasta", "--modular"]);
        assert_eq!(out.len(), 2);
        // The record's id, the point's index in the file, and its numbers:
        // 4 of ACGUAUGAAAGUA's 13 are G or C.
        assert_eq!(out[0], ">p point=2 mfe=-1.00 cai=0.8000 gc=0.31 lambda=0.9");
        assert_eq!(out[1], "ACGUAUGAAAGUA");
    }

    #[test]
    fn fasta_can_spell_dna_and_cut_the_coding_sequence() {
        // The 5′ UTR is ACGU and there is no stop: the coding sequence is the
        // three residues' nine nucleotides after it.
        let cds = rendered(&["--fasta", "--modular", "--cds-only"]);
        assert_eq!(cds[1], "AUGAAAGUA");
        let dna = rendered(&["--fasta", "--modular", "--dna"]);
        assert_eq!(dna[1], "ACGTATGAAAGTA");
        let both = rendered(&["--fasta", "--modular", "--dna", "--cds-only"]);
        assert_eq!(both[1], "ATGAAAGTA");
        // Without --fasta they spell nothing, and are refused rather than
        // dropped.
        assert!(parse(["--dna"].map(String::from)).is_err());
        assert!(parse(["--cds-only", "--tsv"].map(String::from)).is_err());
    }

    #[test]
    fn the_table_names_the_verdict_column_modular_and_shows_gc() {
        let out = rendered(&[]);
        let heading = out
            .iter()
            .find(|l| l.trim_start().starts_with("mfe"))
            .unwrap();
        assert!(
            heading.contains("modular") && heading.contains(" gc "),
            "{heading}"
        );
        assert!(!heading.contains("s.opt"), "{heading}");
        // 6/13 and 4/13 to two places.
        assert!(out.iter().any(|l| l.contains(" 0.46 ")), "{out:?}");
        assert!(out.iter().any(|l| l.contains(" 0.31 ")), "{out:?}");
    }

    #[test]
    fn gc_and_homopolymer_filters_select_on_the_derived_metrics() {
        let count = |words: &[&str]| {
            let mut w = vec!["--tsv"];
            w.extend(words);
            rendered(&w).len() - 1
        };
        assert_eq!(count(&["--min-gc", "0.4"]), 1);
        assert_eq!(count(&["--max-gc", "0.4"]), 1);
        assert_eq!(count(&["--min-gc", "0.5"]), 0);
        // ACGUAUGAAGGUG's longest run is 2 (AA, GG); ACGUAUGAAAGUA's is 3.
        assert_eq!(count(&["--max-homopolymer", "2"]), 1);
        assert_eq!(count(&["--max-homopolymer", "3"]), 2);
    }

    #[test]
    fn a_point_that_carries_its_metrics_is_shown_with_its_own() {
        let text = FILE.replace(
            r#""on_hull":true,"lambda":0.5"#,
            r#""on_hull":true,"gc":0.99,"lambda":0.5"#,
        );
        let r = parse_lines(&text).unwrap();
        let args = parse(["--tsv"].map(String::from)).unwrap();
        assert!(render(&args, &r)[1].starts_with("-2.5\t0.95\t0.99\t"));
    }

    #[test]
    fn an_unchecked_point_renders_without_a_verdict() {
        let text = FILE.replace(r#","modular":false"#, "");
        let r = parse_lines(&text).unwrap();
        let out = render(&ShowArgs::default(), &r).join("\n");
        assert!(out.contains("1 of 1 modular"), "{out}");
    }

    #[test]
    fn peak_memory_shows_the_unit_it_fits_in() {
        assert_eq!(peak_memory(40 * 1024), "40 MB");
        assert_eq!(peak_memory(400 * 1024), "400 MB");
        assert_eq!(peak_memory(3 * 1024 * 1024), "3.0 GB");
        // The header line names one of them rather than rounding both to 0.0.
        let out = rendered(&[]).join("\n");
        assert!(out.contains("peak 2.0 GB"), "{out}");
    }

    #[test]
    fn the_header_line_names_the_record_id_not_its_description() {
        let text = FILE.replace(
            r#""name":"p""#,
            r#""name":"GLPE_HUMAN_20aa first 20 residues""#,
        );
        let r = parse_lines(&text).unwrap();
        let first = render(&ShowArgs::default(), &r)[0].clone();
        assert_eq!(first, "design — 3 residue(s) of GLPE_HUMAN_20aa");
    }

    /// A checked front: one certified point and one that failed. This is the
    /// shape `check` writes, and the rendering is its contract.
    const CHECKED: &str = concat!(
        r#"{"record":"header","format":"derna-result","format_version":1,"command":"check","#,
        r#""version":"0.1.0","protein":{"name":"p","sequence":"MKV"},"#,
        r#""utr5":{"name":null,"sequence":"ACGU"},"utr3":null,"stop":null,"#,
        r#""forbid":["5-cds"],"access":null,"codon_usage":{"source":"human","table":[]},"#,
        r#""wall_s":9.0,"peak_rss_kb":1048576}"#,
        "\n",
        r#"{"record":"point","sequence":"ACGUAUGAAAGUA","mfe":-1.0,"cai":0.8,"cai_log":-0.3,"#,
        r#""structure":".............","on_hull":true,"gap":0.0,"#,
        r#""modular":true,"violations":[],"status":"certified"}"#,
        "\n",
        r#"{"record":"point","sequence":"ACGUAUGAAGGUG","mfe":-2.5,"cai":0.95,"cai_log":-0.1,"#,
        r#""structure":".....((...)).","on_hull":false,"gap":1.0,"#,
        r#""modular":false,"violations":[[5,13]]}"#,
        "\n",
    );

    #[test]
    fn a_checked_result_shows_each_points_status() {
        let r = parse_lines(CHECKED).unwrap();
        let out = render(&ShowArgs::default(), &r);
        let text = out.join("\n");
        assert!(
            text.contains("2 point(s), 1 on the hull, 1 of 2 modular"),
            "{text}"
        );
        assert!(
            out.iter().any(|l| l.trim_end().ends_with("certified")),
            "certified is not on any row:\n{text}"
        );
    }

    #[test]
    fn a_bounded_gap_is_marked_rather_than_printed_as_a_value() {
        // A check refuted by a beam reports `gap_at_least`; the column says so.
        let text = FILE
            .replace(r#""gap":1.0"#, r#""gap_at_least":1.0"#)
            .replace(r#""free_mfe":-3.5"#, r#""free_mfe_at_most":-3.5"#);
        let r = parse_lines(&text).unwrap();
        let rows = render(&ShowArgs::default(), &r);
        assert!(rows.iter().any(|l| l.contains("≥1.00")), "{rows:?}");
        let args = parse(["--tsv"].map(String::from)).unwrap();
        let tsv = render(&args, &r);
        assert!(tsv[1].contains(">=1"), "{}", tsv[1]);
    }

    #[test]
    fn a_file_that_is_not_a_result_is_refused() {
        assert!(parse_lines("{\"a\":1}\n").is_err());
        assert!(parse_lines("").is_err());
        assert!(parse_lines("not json\n").is_err());
    }

    #[test]
    fn a_filter_the_file_cannot_answer_says_so() {
        // "0 shown" reads as *no point passed*; on an unchecked file the true
        // answer is *no point was asked*, and a caller acts differently on
        // the two.
        let unchecked = FILE
            .replace(r#","modular":false"#, "")
            .replace(r#","modular":true"#, "");
        let r = parse_lines(&unchecked).unwrap();
        let args = parse(["--modular"].map(String::from)).unwrap();
        let text = render(&args, &r).join("\n");
        assert!(
            text.contains("no point in this file carries a check"),
            "{text}"
        );
        assert!(text.contains("derna check"), "{text}");

        // On a checked file the note is absent: the count means what it says.
        let checked = render(&args, &file()).join("\n");
        assert!(!checked.contains("carries a check"), "{checked}");
    }

    #[test]
    fn a_later_format_is_refused_rather_than_half_read() {
        let later = FILE.replace(r#""format_version":1"#, r#""format_version":3"#);
        let e = parse_lines(&later).unwrap_err().to_string();
        assert!(e.contains("format version 3"), "{e}");
        assert!(e.contains("newer derna"), "{e}");
        // An absent version is refused too: a file that does not say cannot be
        // assumed to be this one.
        let silent = FILE.replace(r#""format_version":1,"#, "");
        assert!(parse_lines(&silent).is_err());
    }

    #[test]
    fn sequences_under_tsv_are_two_more_columns() {
        let out = rendered(&["--tsv", "--sequences"]);
        assert_eq!(
            out[0],
            "mfe\tcai\tgc\tgap\tmodular\ton_hull\tlambda\tstatus\tgc_cds\tu_cds\t\
             gc_window_min\tgc_window_max\tmax_homopolymer\tsequence\tstructure"
        );
        assert!(
            out[1].ends_with("\tACGUAUGAAGGUG\t.....((...))."),
            "{}",
            out[1]
        );
        // Under --fasta the request says nothing, so it is refused rather than
        // quietly dropped.
        assert!(parse(["--fasta", "--sequences"].map(String::from)).is_err());
    }

    #[test]
    fn every_tsv_row_has_the_header_s_columns() {
        // What makes a TSV machine-readable: a reader splitting on tabs gets
        // the same number of fields from every line. A point missing its check
        // has to leave its cells *empty*, not absent — and a row is built from
        // the point's own fields, so a missing one could shorten the row.
        for flags in [vec!["--tsv"], vec!["--tsv", "--sequences"]] {
            let out = rendered(&flags);
            let columns = out[0].split('\t').count();
            for (i, line) in out.iter().enumerate() {
                assert_eq!(
                    line.split('\t').count(),
                    columns,
                    "{flags:?} line {i} is ragged: {line}"
                );
            }
        }
    }

    #[test]
    fn two_output_shapes_at_once_are_refused() {
        assert!(parse(["--tsv", "--fasta"].map(String::from)).is_err());
        assert!(parse(["a.jsonl", "b.jsonl"].map(String::from)).is_err());
        assert!(parse(["--wat"].map(String::from)).is_err());
    }
}
