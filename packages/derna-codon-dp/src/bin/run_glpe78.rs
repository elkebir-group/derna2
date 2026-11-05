//! Sweep `solve_pord` on a single protein FASTA at `dangling_5_prime = false`
//! and emit JSONL conformant to the [parity contract][contract].
//!
//! Pairs with the parity harness at `packages/derna-codon-dp/scripts/cpp_parity/`:
//! reads `--fasta`, `--codon-usage`, writes `--out` plus a sidecar
//! `<out>.machine.json` capturing the run's wall-clock and host snapshot,
//! and a telemetry sidecar with per-λ solve timing.
//!
//! `--stream <path>` appends one JSONL record per BRD solve as it finishes
//! (λ, design, solve time, elapsed time since the sweep started) and syncs
//! the file after each, so a run stopped at a time limit keeps every solve
//! it finished.
//!
//! [contract]: https://github.com/elkebir-group/derna2/blob/main/packages/derna-codon-dp/scripts/cpp_parity/README.md

use std::cell::RefCell;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use derna_codon_dp::{
    load_codon_usage, solve_pord_on_cai_scale, BrdOptions, BrdSolution, ParetoPoint, PordTelemetry,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

struct Args {
    fasta: PathBuf,
    codon_usage: PathBuf,
    out: PathBuf,
    machine: Option<String>,
    telemetry_out: Option<PathBuf>,
    stream: Option<PathBuf>,
    cai_scale: f64,
}

struct RunSummary<'a> {
    machine_label: Option<&'a str>,
    wall_clock_seconds: f64,
    n_pareto_points: usize,
    telemetry_path: &'a Path,
    telemetry: &'a PordTelemetry,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1))?;
    let protein = read_single_protein(&args.fasta)?;
    let codon_usage = load_codon_usage(&args.codon_usage)?;
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };

    let t0 = Instant::now();
    let stream = args
        .stream
        .as_deref()
        .map(SolveStream::create)
        .transpose()?;
    let report = |sol: &BrdSolution, solve_s: f64| {
        if let Some(stream) = &stream {
            stream.append(sol, solve_s, t0.elapsed().as_secs_f64());
        }
    };
    let (points, telemetry) = solve_pord_on_cai_scale(
        &protein,
        &codon_usage,
        &options,
        Some(&report),
        args.cai_scale,
    )?;
    let elapsed = t0.elapsed().as_secs_f64();
    if let Some(stream) = stream {
        stream.finish()?;
    }

    write_jsonl(&args.out, &points)?;
    let telemetry_path = args
        .telemetry_out
        .unwrap_or_else(|| telemetry_sidecar_path(&args.out));
    let summary = RunSummary {
        machine_label: args.machine.as_deref(),
        wall_clock_seconds: elapsed,
        n_pareto_points: points.len(),
        telemetry_path: &telemetry_path,
        telemetry: &telemetry,
    };
    write_telemetry_json(&summary)?;
    write_machine_snapshot(&args.out, &summary)?;

    eprintln!(
        "wrote {} pareto points ({} BRD solves, {:.2}s) → {}",
        points.len(),
        telemetry.total_solves(),
        elapsed,
        args.out.display()
    );
    eprintln!("wrote sweep telemetry → {}", telemetry_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Arg parsing (hand-rolled — three required flags + one optional)
// ---------------------------------------------------------------------------

fn parse_args(iter: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut fasta: Option<PathBuf> = None;
    let mut codon_usage: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut machine: Option<String> = None;
    let mut telemetry_out: Option<PathBuf> = None;
    let mut stream: Option<PathBuf> = None;
    let mut cai_scale = 1.0_f64;
    let mut it = iter.into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--fasta" => fasta = Some(PathBuf::from(require_value(&mut it, "--fasta")?)),
            "--codon-usage" => {
                codon_usage = Some(PathBuf::from(require_value(&mut it, "--codon-usage")?));
            }
            "--out" => out = Some(PathBuf::from(require_value(&mut it, "--out")?)),
            "--machine" => machine = Some(require_value(&mut it, "--machine")?),
            "--telemetry-out" => {
                telemetry_out = Some(PathBuf::from(require_value(&mut it, "--telemetry-out")?));
            }
            "--stream" => stream = Some(PathBuf::from(require_value(&mut it, "--stream")?)),
            "--cai-scale" => {
                cai_scale = require_value(&mut it, "--cai-scale")?
                    .parse()
                    .map_err(|e| format!("--cai-scale parse: {e}"))?;
                if !(cai_scale >= 1.0 && cai_scale.is_finite()) {
                    return Err(format!(
                        "--cai-scale must be a finite number at least 1; got {cai_scale}"
                    ));
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok(Args {
        fasta: fasta.ok_or("--fasta is required")?,
        codon_usage: codon_usage.ok_or("--codon-usage is required")?,
        out: out.ok_or("--out is required")?,
        machine,
        telemetry_out,
        stream,
        cai_scale,
    })
}

fn require_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "run_glpe78 --fasta <path> --codon-usage <path> --out <path>\n\
         \x20          [--machine <name>] [--telemetry-out <path>] [--stream <path>]\n\
         \x20          [--cai-scale <s>]\n\
         \n\
         Runs solve_pord on the protein in <fasta> and writes JSONL hull records\n\
         to <out>. Emits <out>.machine.json and <out-stem>.telemetry.json sidecars\n\
         with wall-clock, host snapshot, and per-λ solve timing.\n\
         \n\
         --cai-scale <s> places the sweep's λ values where a solver quantizing\n\
         CAI on scale <s> would place them: this solver uses 100x C++'s CAI\n\
         scale, so the same thresholds subdivide further on its own λ than on\n\
         C++'s. Pass 100 to sweep the weights DERNA C++ would sweep. Every\n\
         solve is this solver's either way; only the weights move."
    );
}

// ---------------------------------------------------------------------------
// Per-solve stream
// ---------------------------------------------------------------------------

/// Append-only JSONL record of every BRD solve, synced line by line.
///
/// The sweep reports through a shared reference, so the file sits behind a
/// `RefCell`. A write error is kept rather than raised mid-sweep and is
/// returned by [`SolveStream::finish`].
struct SolveStream {
    file: RefCell<fs::File>,
    error: RefCell<Option<std::io::Error>>,
}

impl SolveStream {
    fn create(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        Ok(Self {
            file: RefCell::new(fs::File::create(path)?),
            error: RefCell::new(None),
        })
    }

    fn append(&self, sol: &BrdSolution, solve_s: f64, elapsed_s: f64) {
        if self.error.borrow().is_some() {
            return;
        }
        let record = serde_json::json!({
            "lambda": sol.lambda_,
            "mfe_kcal_mol": sol.mfe_kcal_mol,
            "cai": sol.cai,
            "rna_seq": sol.rna_seq,
            "structure": sol.structure,
            "solve_s": solve_s,
            "elapsed_s": elapsed_s,
        });
        let mut file = self.file.borrow_mut();
        let written = writeln!(file, "{record}").and_then(|()| file.sync_data());
        if let Err(e) = written {
            *self.error.borrow_mut() = Some(e);
        }
    }

    fn finish(self) -> Result<(), std::io::Error> {
        self.error.into_inner().map_or(Ok(()), Err)
    }
}

// ---------------------------------------------------------------------------
// FASTA reader
// ---------------------------------------------------------------------------

fn read_single_protein(path: &Path) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut records: Vec<String> = Vec::new();
    let mut current: String = String::new();
    let mut in_record = false;
    for line in text.lines() {
        if let Some(stripped) = line.strip_prefix('>') {
            let _ = stripped;
            if in_record {
                records.push(std::mem::take(&mut current));
            }
            in_record = true;
        } else if in_record {
            current.push_str(line.trim());
        }
    }
    if in_record {
        records.push(current);
    }
    match records.len() {
        0 => Err(format!("FASTA {} contains no records", path.display())),
        1 => Ok(records.into_iter().next().unwrap()),
        n => Err(format!(
            "FASTA {} contains {n} records; expected 1",
            path.display()
        )),
    }
}

// ---------------------------------------------------------------------------
// JSONL writer (parity-contract schema)
// ---------------------------------------------------------------------------

fn write_jsonl(path: &Path, points: &[ParetoPoint]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut buf = String::with_capacity(points.len() * 256);
    for p in points {
        let record = serde_json::json!({
            "lambda": p.lambda_,
            "mfe_kcal_mol": p.mfe_kcal_mol,
            "cai": p.cai,
            "rna_seq": p.rna_seq,
            "structure": p.structure,
            "source": "rust",
        });
        buf.push_str(&serde_json::to_string(&record)?);
        buf.push('\n');
    }
    fs::write(path, buf)?;
    Ok(())
}

fn write_telemetry_json(summary: &RunSummary<'_>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = summary.telemetry_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let solves: Vec<serde_json::Value> = summary
        .telemetry
        .brd_solves
        .iter()
        .map(|solve| {
            serde_json::json!({
                "lambda": solve.lambda_,
                "elapsed_seconds": solve.elapsed_seconds,
                "mfe_kcal_mol": solve.mfe_kcal_mol,
                "cai": solve.cai,
            })
        })
        .collect();
    let sweep_brackets: Vec<serde_json::Value> = summary
        .telemetry
        .sweep_brackets
        .iter()
        .map(|bracket| {
            serde_json::json!({
                "lo": bracket.lo,
                "hi": bracket.hi,
                "mid": bracket.mid,
                "lo_cell": {
                    "mfe_kcal_mol": bracket.lo_mfe_kcal_mol,
                    "cai": bracket.lo_cai,
                },
                "hi_cell": {
                    "mfe_kcal_mol": bracket.hi_mfe_kcal_mol,
                    "cai": bracket.hi_cai,
                },
                "mfe_differs": bracket.mfe_differs,
                "cai_differs": bracket.cai_differs,
                "width": bracket.width,
                "width_threshold": bracket.width_threshold,
                "stop_reason": bracket.stop_reason.as_str(),
            })
        })
        .collect();
    let document = serde_json::json!({
        "machine_label": summary.machine_label.unwrap_or("unspecified"),
        "sweep_wall_clock_seconds": summary.wall_clock_seconds,
        "n_pareto_points": summary.n_pareto_points,
        "total_solves": summary.telemetry.total_solves(),
        "memo_hits": summary.telemetry.memo_hits,
        "memo_misses": summary.telemetry.memo_misses,
        "solves": solves,
        "sweep_brackets": sweep_brackets,
    });
    fs::write(
        summary.telemetry_path,
        serde_json::to_string_pretty(&document)?,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Machine snapshot sidecar
// ---------------------------------------------------------------------------

fn write_machine_snapshot(
    out_path: &Path,
    summary: &RunSummary<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = serde_json::json!({
        "cpu": detect_cpu(),
        "os": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "rust_version": run_capture(&["rustc", "--version"]).unwrap_or_else(|| "unknown".into()),
        "repo_sha": run_capture(&["git", "rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
        "machine_label": summary.machine_label.unwrap_or("unspecified"),
        "wall_clock_seconds": summary.wall_clock_seconds,
        "n_pareto_points": summary.n_pareto_points,
        "n_brd_solves": summary.telemetry.total_solves(),
        "memo_hits": summary.telemetry.memo_hits,
        "memo_misses": summary.telemetry.memo_misses,
        "telemetry_path": summary.telemetry_path.display().to_string(),
    });
    let path = machine_sidecar_path(out_path);
    fs::write(&path, serde_json::to_string_pretty(&snapshot)?)?;
    Ok(())
}

fn sidecar_path(out_path: &Path, kind: &str) -> PathBuf {
    let mut path = out_path.to_path_buf();
    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());
    let stem = match file_name.rsplit_once('.') {
        Some((s, _)) => s,
        None => &file_name,
    };
    path.set_file_name(format!("{stem}.{kind}.json"));
    path
}

fn machine_sidecar_path(out_path: &Path) -> PathBuf {
    sidecar_path(out_path, "machine")
}

fn telemetry_sidecar_path(out_path: &Path) -> PathBuf {
    sidecar_path(out_path, "telemetry")
}

fn detect_cpu() -> String {
    if std::env::consts::OS == "macos" {
        if let Some(brand) = run_capture(&["sysctl", "-n", "machdep.cpu.brand_string"]) {
            return brand;
        }
    }
    "unknown".to_string()
}

fn run_capture(argv: &[&str]) -> Option<String> {
    let (cmd, rest) = argv.split_first()?;
    let out = Command::new(cmd).args(rest).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}
