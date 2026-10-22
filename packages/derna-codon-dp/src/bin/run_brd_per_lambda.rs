//! Run `solve_brd` once per λ from an input JSONL and emit per-λ records
//! conformant to the parity-harness contract.
//!
//! Reads the `lambda` field from each record of `--lambdas` (typically the
//! C++ side's `pareto_points.jsonl`), calls
//! [`derna_codon_dp::solve_brd`] at each λ, and writes one JSONL record
//! per λ to `--out`. λ values exactly equal to `0.0` or `1.0` are skipped
//! (the sweep is open on `(0, 1)`; `solve_brd` is undefined at the
//! endpoints). The order of output records matches the order of input λs.
//!
//! `--cai-scale <s>` reweights the input λ for the Rust solver so that the
//! C++↔Rust BRD scalars select the same argmin at corresponding λs. v1
//! and the Rust port quantize CAI as `CAI_dekacal = round(log_CAI · 100)`;
//! DERNA C++ uses raw `log_CAI`. Two solvers minimizing
//! `λ·MFE − (1−λ)·CAI_term` with that 100× CAI scale select the same
//! `(MFE, CAI)` vertex iff `λ_R = s·λ_C / (1 + (s−1)·λ_C)` where `s` is
//! the CAI scale factor (100 for the v1↔C++ pairing). Pass `--cai-scale
//! 100` when feeding C++'s JSONL to compare apples-to-apples; the output
//! `lambda` field carries the original C++ λ so `diff_per_lambda.py` can
//! group by it.
//!
//! Pairs with `diff_per_lambda.py` in the parity harness — see
//! `packages/derna-codon-dp/scripts/cpp_parity/README.md` for the JSONL
//! schema. Emits a sidecar `<out-stem>.machine.json` capturing the run's
//! wall-clock and host snapshot.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use derna_codon_dp::{load_codon_usage, solve_brd, BrdOptions, BrdSolution, CodonUsage};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

struct Args {
    fasta: PathBuf,
    codon_usage: PathBuf,
    lambdas: PathBuf,
    out: PathBuf,
    machine: Option<String>,
    cai_scale: f64,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1))?;
    let protein = read_single_protein(&args.fasta)?;
    let codon_usage = load_codon_usage(&args.codon_usage)?;
    let lambdas = read_lambdas(&args.lambdas)?;
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };

    if args.cai_scale != 1.0 {
        eprintln!(
            "running solve_brd on {} λ values from {} (cai-scale={}, transform λ_R = s·λ / (1+(s−1)·λ))",
            lambdas.len(),
            args.lambdas.display(),
            args.cai_scale,
        );
    } else {
        eprintln!(
            "running solve_brd on {} λ values from {} (no λ transform)",
            lambdas.len(),
            args.lambdas.display(),
        );
    }

    let ctx = SweepInputs {
        protein: &protein,
        codon_usage: &codon_usage,
        lambdas: &lambdas,
        options: &options,
        cai_scale: args.cai_scale,
    };
    let t0 = Instant::now();
    let solutions = solve_per_lambda(&ctx)?;
    let elapsed = t0.elapsed().as_secs_f64();

    write_jsonl(&args.out, &solutions)?;
    write_machine_snapshot(&args.out, args.machine.as_deref(), elapsed, solutions.len())?;

    eprintln!(
        "wrote {} per-λ records ({:.2}s total) → {}",
        solutions.len(),
        elapsed,
        args.out.display()
    );
    Ok(())
}

/// Bundled inputs to the per-λ sweep — keeps `solve_per_lambda` at one
/// argument and lets call sites name what they're passing.
struct SweepInputs<'a> {
    protein: &'a str,
    codon_usage: &'a CodonUsage,
    lambdas: &'a [f64],
    options: &'a BrdOptions,
    cai_scale: f64,
}

/// Re-weight an input λ for the Rust solver so the argmin selection
/// matches a C++ solver that uses CAI scaled by `1/scale` relative to
/// v1/Rust. Identity when `scale == 1.0`. See module doc-comment for the
/// derivation.
fn transform_lambda(lambda: f64, scale: f64) -> f64 {
    if scale == 1.0 {
        return lambda;
    }
    scale * lambda / (1.0 + (scale - 1.0) * lambda)
}

fn solve_per_lambda(
    ctx: &SweepInputs<'_>,
) -> Result<Vec<(f64, BrdSolution)>, Box<dyn std::error::Error>> {
    let mut solutions = Vec::with_capacity(ctx.lambdas.len());
    for (i, &lambda) in ctx.lambdas.iter().enumerate() {
        if lambda == 0.0 || lambda == 1.0 {
            eprintln!(
                "  [{:>3}/{}] λ={lambda} — skipped (endpoint excluded by bisection range)",
                i + 1,
                ctx.lambdas.len()
            );
            continue;
        }
        let lambda_solver = transform_lambda(lambda, ctx.cai_scale);
        let t_lambda = Instant::now();
        let solution = solve_brd(ctx.protein, ctx.codon_usage, lambda_solver, ctx.options)?;
        let elapsed = t_lambda.elapsed().as_secs_f64();
        eprintln!(
            "  [{:>3}/{}] λ_in={:.6} λ_solver={:.6} mfe={:+.4} cai={:.6} ({:.2}s)",
            i + 1,
            ctx.lambdas.len(),
            lambda,
            lambda_solver,
            solution.mfe_kcal_mol,
            solution.cai,
            elapsed
        );
        solutions.push((lambda, solution));
    }
    Ok(solutions)
}

// ---------------------------------------------------------------------------
// Arg parsing (hand-rolled — four required flags + one optional)
// ---------------------------------------------------------------------------

fn parse_args(iter: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut fasta: Option<PathBuf> = None;
    let mut codon_usage: Option<PathBuf> = None;
    let mut lambdas: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut machine: Option<String> = None;
    let mut cai_scale: f64 = 1.0;
    let mut it = iter.into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--fasta" => fasta = Some(PathBuf::from(require_value(&mut it, "--fasta")?)),
            "--codon-usage" => {
                codon_usage = Some(PathBuf::from(require_value(&mut it, "--codon-usage")?));
            }
            "--lambdas" => lambdas = Some(PathBuf::from(require_value(&mut it, "--lambdas")?)),
            "--out" => out = Some(PathBuf::from(require_value(&mut it, "--out")?)),
            "--machine" => machine = Some(require_value(&mut it, "--machine")?),
            "--cai-scale" => {
                cai_scale = require_value(&mut it, "--cai-scale")?
                    .parse()
                    .map_err(|e| format!("--cai-scale parse: {e}"))?;
                if cai_scale <= 0.0 || !cai_scale.is_finite() {
                    return Err(format!(
                        "--cai-scale must be a positive finite number; got {cai_scale}"
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
        lambdas: lambdas.ok_or("--lambdas is required")?,
        out: out.ok_or("--out is required")?,
        machine,
        cai_scale,
    })
}

fn require_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "run_brd_per_lambda --fasta <path> --codon-usage <path>\n\
         \x20                  --lambdas <jsonl> --out <jsonl>\n\
         \x20                  [--cai-scale <float>] [--machine <label>]\n\
         \n\
         Reads the `lambda` field from each JSONL record in <lambdas>, runs\n\
         solve_brd once per λ, and writes one parity-contract record per λ to\n\
         <out>. Skips λ ∈ {{0, 1}} (sweep is open on (0, 1)).\n\
         \n\
         --cai-scale <s> (default 1.0) reweights each input λ as\n\
         λ_solver = s·λ / (1 + (s−1)·λ) before feeding the Rust solver, so\n\
         that v1/Rust (s=100 dekacal CAI) compares apples-to-apples against\n\
         DERNA C++ (raw log_CAI) at the same input λ. The output `lambda`\n\
         field carries the *original* λ for cross-side comparison."
    );
}

// ---------------------------------------------------------------------------
// FASTA reader (single record)
// ---------------------------------------------------------------------------

fn read_single_protein(path: &Path) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut records: Vec<String> = Vec::new();
    let mut current: String = String::new();
    let mut in_record = false;
    for line in text.lines() {
        if line.starts_with('>') {
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
// λ-list reader — pulls the `lambda` field from each JSONL line
// ---------------------------------------------------------------------------

fn read_lambdas(path: &Path) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut lambdas = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: invalid JSON: {e}", path.display(), line_no + 1))?;
        let lambda = value
            .get("lambda")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                format!(
                    "{}:{}: missing or non-numeric `lambda` field",
                    path.display(),
                    line_no + 1
                )
            })?;
        lambdas.push(lambda);
    }
    if lambdas.is_empty() {
        return Err(format!("{} contains no λ records", path.display()).into());
    }
    Ok(lambdas)
}

// ---------------------------------------------------------------------------
// JSONL writer (parity-contract schema)
// ---------------------------------------------------------------------------

fn write_jsonl(
    path: &Path,
    solutions: &[(f64, BrdSolution)],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut buf = String::with_capacity(solutions.len() * 256);
    for (lambda, sol) in solutions {
        let record = serde_json::json!({
            "lambda": lambda,
            "mfe_kcal_mol": sol.mfe_kcal_mol,
            "cai": sol.cai,
            "rna_seq": sol.rna_seq,
            "structure": sol.structure,
            "source": "rust",
        });
        buf.push_str(&serde_json::to_string(&record)?);
        buf.push('\n');
    }
    fs::write(path, buf)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Machine snapshot sidecar
// ---------------------------------------------------------------------------

fn write_machine_snapshot(
    out_path: &Path,
    machine_label: Option<&str>,
    wall_clock_seconds: f64,
    n_solutions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = serde_json::json!({
        "cpu": detect_cpu(),
        "os": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "rust_version": run_capture(&["rustc", "--version"]).unwrap_or_else(|| "unknown".into()),
        "repo_sha": run_capture(&["git", "rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
        "machine_label": machine_label.unwrap_or("unspecified"),
        "wall_clock_seconds": wall_clock_seconds,
        "n_solutions": n_solutions,
    });
    let path = sidecar_path(out_path);
    fs::write(&path, serde_json::to_string_pretty(&snapshot)?)?;
    Ok(())
}

fn sidecar_path(out_path: &Path) -> PathBuf {
    let mut path = out_path.to_path_buf();
    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());
    let stem = match file_name.rsplit_once('.') {
        Some((s, _)) => s,
        None => &file_name,
    };
    path.set_file_name(format!("{stem}.machine.json"));
    path
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
