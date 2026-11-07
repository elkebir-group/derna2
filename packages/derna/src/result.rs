//! The result format every subcommand writes: JSONL, a header line recording
//! the inputs, then one line per design.
//!
//! Every energy is kcal/mol. `mfe` is the energy under the constraint, which is
//! the objective; under a constraint the check adds the free fold of the same
//! sequence. CAI is not repeated in the check, since both folds are of one
//! sequence and so share its codons.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use derna_lib::{Design, FreeFoldGap};
use serde_json::{json, Map, Value};

use crate::free_check::Checked;
use crate::molecule::Molecule;
use crate::Result;

/// The newest result format version this derna writes, and the newest it
/// reads; bumped on any change a reader must know of. Version 2 renamed
/// `structurally_optimal` to `modular` and `structurally_optimal_front` to
/// `modular_front`.
pub const FORMAT_VERSION: u32 = 2;

/// The version a file is written in when it carries none of the renamed
/// fields: its content is then exactly version 1's.
const UNRENAMED_FORMAT_VERSION: u32 = 1;

/// The format version a result file must declare: 2 when one of its lines
/// carries a renamed field, at any depth, and 1 otherwise.
///
/// A reader refuses any version above its own, so declaring 2 on a file a
/// version-1 reader would read correctly (a plain front, a motif front, an
/// unchecked design) would lock every derna built before the rename out of it
/// for nothing. A writer that puts the header out alone, before its points,
/// is versioned by the header: the two that do are modular fronts, whose
/// header carries `modular_front`.
pub fn format_version(lines: &[Value]) -> u32 {
    if lines.iter().any(carries_renamed) {
        FORMAT_VERSION
    } else {
        UNRENAMED_FORMAT_VERSION
    }
}

/// Whether `v` holds a field under a name the rename introduced.
fn carries_renamed(v: &Value) -> bool {
    match v {
        Value::Object(fields) => fields.iter().any(|(key, value)| {
            RENAMED_FIELDS.iter().any(|(_, new)| new == key) || carries_renamed(value)
        }),
        Value::Array(items) => items.iter().any(carries_renamed),
        _ => false,
    }
}

/// Fields that earlier versions of the format wrote under another name, as
/// (old name, current name).
pub const RENAMED_FIELDS: [(&str, &str); 2] = [
    ("structurally_optimal", "modular"),
    ("structurally_optimal_front", "modular_front"),
];

/// A line of a result or progress file with every field an earlier version
/// wrote under an old name moved to its current name, at any depth, so a
/// reader asks for one name whichever version wrote the file. A line that
/// carries both names keeps the current one.
pub fn upgraded(v: Value) -> Value {
    match v {
        Value::Object(fields) => {
            let mut out = Map::new();
            for (key, value) in fields {
                let value = upgraded(value);
                match RENAMED_FIELDS.iter().find(|(old, _)| *old == key) {
                    Some((_, new)) => {
                        out.entry(new.to_string()).or_insert(value);
                    }
                    None => {
                        out.insert(key, value);
                    }
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(upgraded).collect()),
        other => other,
    }
}

/// Parse one line of a result or progress file, reading old field names as
/// their current ones (see [`upgraded`]).
pub fn parse_line(line: &str) -> serde_json::Result<Value> {
    serde_json::from_str(line).map(upgraded)
}

/// The header line's fields that describe the molecule: protein, flanks, stop,
/// constraint and the full codon usage table.
pub fn molecule_fields(m: &Molecule) -> Map<String, Value> {
    let seq = |s: &crate::input::NamedSequence| json!({"name": s.name, "sequence": s.sequence});
    let mut fields = Map::new();
    fields.insert("protein".into(), seq(&m.protein));
    fields.insert("utr5".into(), m.utr5.as_ref().map_or(Value::Null, seq));
    fields.insert("utr3".into(), m.utr3.as_ref().map_or(Value::Null, seq));
    fields.insert("stop".into(), m.stop_field());
    fields.insert("forbid".into(), json!(m.constraint.forbid_names()));
    fields.insert(
        "access".into(),
        m.constraint
            .access
            .map_or(Value::Null, |(u, d)| json!([u, d])),
    );
    let table: Vec<Value> = m
        .codons
        .rows
        .iter()
        .map(|(aa, codon, f)| json!([aa, codon, f]))
        .collect();
    fields.insert(
        "codon_usage".into(),
        json!({"source": m.codons.source, "table": table}),
    );
    fields
}

/// A full header line: format, command, version, the molecule, then `extra`.
pub fn header(command: &str, m: &Molecule, extra: Map<String, Value>) -> Value {
    let mut h = Map::new();
    h.insert("record".into(), "header".into());
    h.insert("format".into(), "derna-result".into());
    h.insert("format_version".into(), FORMAT_VERSION.into());
    h.insert("command".into(), command.into());
    h.insert("version".into(), env!("CARGO_PKG_VERSION").into());
    h.extend(molecule_fields(m));
    h.extend(extra);
    Value::Object(h)
}

/// The check of one design against the free fold of its own sequence.
pub fn verdict_fields(gap: &FreeFoldGap) -> Map<String, Value> {
    let violations: Vec<Value> = gap
        .reinstated_pairs
        .iter()
        .map(|&(i, j)| json!([i + 1, j + 1]))
        .collect();
    let mut v = Map::new();
    v.insert("free_mfe".into(), gap.free_mfe_kcal_mol.into());
    v.insert("free_structure".into(), gap.free_structure.clone().into());
    v.insert("gap".into(), gap.gap_kcal_mol.into());
    v.insert("modular".into(), gap.is_modular().into());
    v.insert("violations".into(), violations.into());
    v
}

/// What a beam refutation establishes, in place of the exact pair: a bound,
/// named so it cannot be read as an exact value.
pub fn bound_fields(free_mfe_at_most: f64, gap_at_least: f64) -> Map<String, Value> {
    let mut v = Map::new();
    v.insert("free_mfe_at_most".into(), free_mfe_at_most.into());
    v.insert("gap_at_least".into(), gap_at_least.into());
    v.insert("modular".into(), false.into());
    v
}

/// The check's fields, exact or bounded.
pub fn check_fields(checked: &Checked) -> Map<String, Value> {
    match checked {
        Checked::Exact(gap) => verdict_fields(gap),
        Checked::Refuted {
            free_mfe_at_most,
            gap_at_least,
        } => bound_fields(*free_mfe_at_most, *gap_at_least),
    }
}

/// One design line. `lambda` is the weight it was asked at, on `design`
/// results only.
pub fn point(
    design: &Design,
    lambda: Option<f64>,
    on_hull: bool,
    gap: Option<&FreeFoldGap>,
) -> Value {
    let mut p = Map::new();
    p.insert("record".into(), "point".into());
    p.insert("sequence".into(), design.sequence.clone().into());
    p.insert("mfe".into(), design.mfe_kcal_mol.into());
    p.insert("cai".into(), design.cai.into());
    p.insert("cai_log".into(), design.cai_log.into());
    p.insert("structure".into(), design.structure.clone().into());
    p.insert("on_hull".into(), on_hull.into());
    if let Some(l) = lambda {
        p.insert("lambda".into(), l.into());
    }
    if let Some(g) = gap {
        p.extend(verdict_fields(g));
    }
    Value::Object(p)
}

/// One design line whose check may be a bound rather than the exact pair.
pub fn checked_point(
    design: &Design,
    lambda: Option<f64>,
    on_hull: bool,
    checked: Option<&Checked>,
) -> Value {
    let mut p = point(design, lambda, on_hull, None);
    if let (Value::Object(fields), Some(checked)) = (&mut p, checked) {
        fields.extend(check_fields(checked));
    }
    p
}

/// Where a result goes: `--out PATH` or stdout. An existing file is refused
/// unless `force`.
pub fn open_output(out: Option<&Path>, force: bool) -> Result<Box<dyn Write>> {
    check_output(out, force)?;
    match destination(out) {
        None => Ok(Box::new(BufWriter::new(Copied(std::io::stdout().lock())))),
        Some(path) => Ok(Box::new(BufWriter::new(File::create(path)?))),
    }
}

thread_local! {
    /// What this thread's commands wrote to stdout as a result, so the table
    /// at the end of a run can render it although stdout is not a file to
    /// read back.
    static STDOUT_COPY: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A writer to stdout that keeps a copy of what it writes.
struct Copied<W>(W);

impl<W: Write> Write for Copied<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.0.write(buf)?;
        STDOUT_COPY.with(|copy| copy.borrow_mut().extend_from_slice(&buf[..n]));
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

/// Take what this thread has written to stdout as a result since the last
/// take, leaving the copy empty.
pub fn take_stdout_copy() -> Vec<u8> {
    STDOUT_COPY.with(|copy| std::mem::take(&mut *copy.borrow_mut()))
}

/// Where a result goes, with `-` read as stdout.
///
/// `--protein -` means stdin, so `--out -` means stdout rather than a file
/// named `-`: a caller who writes the one and gets the other has been
/// surprised by a convention this command already uses.
pub fn destination(out: Option<&Path>) -> Option<&Path> {
    out.filter(|path| path.as_os_str() != "-")
}

/// Refuse an existing `--out` before any work, so a long run cannot end on it.
pub fn check_output(out: Option<&Path>, force: bool) -> Result<()> {
    match destination(out) {
        Some(path) if would_overwrite(path) && !force => Err(format!(
            "--out {}: the file exists; pass --force to overwrite it",
            path.display()
        )
        .into()),
        _ => Ok(()),
    }
}

/// Whether writing to `path` would destroy something.
///
/// Only a **regular file** holds content a run could lose. `/dev/null`,
/// `/dev/stdout` and a named pipe all exist and lose nothing, and refusing
/// them would turn the constraint into an obstacle — `--out /dev/null` is how a
/// caller says *discard this*.
fn would_overwrite(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file())
}

/// Write `lines` as JSONL. A header among them declares the format version
/// the lines need (see [`format_version`]).
///
/// Every point among them carries its sequence metrics
/// ([`crate::metrics`]), derived over the coding window the header among
/// them implies; a writer that puts its header out alone gives its points
/// theirs with [`with_metrics`].
pub fn write_lines(w: &mut dyn Write, lines: &[Value]) -> Result<()> {
    let version = format_version(lines);
    let cds = lines
        .iter()
        .find(|l| l["record"] == "header")
        .map(crate::metrics::coding_window);
    for line in lines {
        let mut line = std::borrow::Cow::Borrowed(line);
        if line["record"] == "header" && line.get("format_version").is_some() {
            line.to_mut()["format_version"] = version.into();
        }
        if let (Some(cds), true) = (&cds, line["record"] == "point") {
            crate::metrics::add_to(line.to_mut(), cds);
        }
        serde_json::to_writer(&mut *w, line.as_ref())?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

/// `points` with their sequence metrics, over the coding window `header`
/// implies, for a writer that writes them after a header written alone.
pub fn with_metrics(header: &Value, mut points: Vec<Value>) -> Vec<Value> {
    let cds = crate::metrics::coding_window(header);
    for p in points.iter_mut().filter(|p| p["record"] == "point") {
        crate::metrics::add_to(p, &cds);
    }
    points
}

/// Replace the file at `path` with `lines` in one step: they are written to a
/// file beside it, synced to disk and renamed over it, so a reader — or a
/// run killed part-way — finds either the old file whole or the new one.
pub fn replace_lines(path: &Path, lines: &[Value]) -> Result<()> {
    let mut beside = path.as_os_str().to_owned();
    beside.push(".partial");
    let beside = std::path::PathBuf::from(beside);
    let mut file = BufWriter::new(File::create(&beside)?);
    write_lines(&mut file, lines)?;
    file.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&beside, path)?;
    Ok(())
}

/// This process's peak resident set, in kB, or `None` where no platform
/// facility reports one.
///
/// The library's reading, not a second one. This read `/proc/self/status`
/// itself and so answered `None` on a platform that has no `/proc` — while
/// `task_clock` answered the same question from `getrusage` and returned a
/// figure. One run then printed `peak n/a` in a header and a real peak in its
/// own timeline.
pub fn peak_rss_kb() -> Option<u64> {
    derna_lib::task_clock::peak_rss_kb()
}
