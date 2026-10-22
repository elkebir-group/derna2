//! Single-`solve_brd` profile harness. Reads a FASTA + codon-usage table,
//! invokes [`derna_codon_dp::solve_brd`] once at the requested λ, and prints
//! `(mfe_kcal_mol, cai, wall_clock_seconds)` to stdout. Designed for samply /
//! cargo-flamegraph runs that want hot-path attribution without the full
//! λ-sweep wall-clock.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use derna_codon_dp::testing::{count_o_prefix_winning_cases, count_outer_gate_cells, FillInputs};
use derna_codon_dp::{load_codon_usage, solve_brd, BrdOptions};
use rna_mfe::Params;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

struct Args {
    fasta: PathBuf,
    codon_usage: PathBuf,
    lambda: f64,
    cai_scale: f64,
    diag_outer_gate: bool,
    diag_o_prefix_winners: bool,
}

impl Args {
    fn lambda_solver(&self) -> f64 {
        transform_lambda(self.lambda, self.cai_scale)
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1))?;
    let protein = read_single_protein(&args.fasta)?;
    let codon_usage = load_codon_usage(&args.codon_usage)?;
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    let lambda_solver = args.lambda_solver();

    if args.diag_outer_gate {
        let params = Params::load_default()?;
        let counts = count_outer_gate_cells(FillInputs {
            protein: &protein,
            codon_usage: &codon_usage,
            lambda_: lambda_solver,
            options: &options,
            params: &params,
        })?;
        println!(
            "diag_outer_gate: lambda_in={:.8} lambda_solver={:.8} gated={} admitted={} total={} gated_fraction={:.4}",
            args.lambda,
            lambda_solver,
            counts.gated,
            counts.admitted,
            counts.total(),
            counts.gated_fraction(),
        );
        return Ok(());
    }

    if args.diag_o_prefix_winners {
        let params = Params::load_default()?;
        let counts = count_o_prefix_winning_cases(FillInputs {
            protein: &protein,
            codon_usage: &codon_usage,
            lambda_: lambda_solver,
            options: &options,
            params: &params,
        })?;
        print_o_prefix_winner_counts(args.lambda, lambda_solver, &counts);
        return Ok(());
    }

    let t0 = Instant::now();
    let solution = solve_brd(&protein, &codon_usage, lambda_solver, &options)?;
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "lambda_in={:.8} lambda_solver={:.8} mfe_kcal_mol={:+.4} cai={:.6} wall_clock_seconds={:.3}",
        args.lambda, lambda_solver, solution.mfe_kcal_mol, solution.cai, elapsed,
    );

    Ok(())
}

fn parse_args(iter: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut fasta: Option<PathBuf> = None;
    let mut codon_usage: Option<PathBuf> = None;
    let mut lambda: Option<f64> = None;
    let mut cai_scale: f64 = 1.0;
    let mut diag_outer_gate = false;
    let mut diag_o_prefix_winners = false;
    let mut it = iter.into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--fasta" => fasta = Some(PathBuf::from(require_value(&mut it, "--fasta")?)),
            "--codon-usage" => {
                codon_usage = Some(PathBuf::from(require_value(&mut it, "--codon-usage")?));
            }
            "--lambda" => {
                lambda = Some(
                    require_value(&mut it, "--lambda")?
                        .parse()
                        .map_err(|e| format!("--lambda parse: {e}"))?,
                );
            }
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
            "--diag-outer-gate" => diag_outer_gate = true,
            "--diag-o-prefix-winners" => diag_o_prefix_winners = true,
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
        lambda: lambda.ok_or("--lambda is required")?,
        cai_scale,
        diag_outer_gate,
        diag_o_prefix_winners,
    })
}

fn transform_lambda(lambda: f64, scale: f64) -> f64 {
    if scale == 1.0 {
        return lambda;
    }
    scale * lambda / (1.0 + (scale - 1.0) * lambda)
}

fn require_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "profile_solve_brd --fasta <path> --codon-usage <path> --lambda <float>\n\
         \n\
         Runs a single solve_brd call at the requested lambda and prints\n\
         (mfe_kcal_mol, cai, wall_clock_seconds) to stdout. Use under samply /\n\
         cargo-flamegraph for hot-path attribution.\n\
         \n\
         Diagnostics:\n\
           --diag-outer-gate\n\
           --diag-o-prefix-winners\n\
           --cai-scale <float>  transform input lambda before solving"
    );
}

fn print_o_prefix_winner_counts(
    lambda_in: f64,
    lambda_solver: f64,
    counts: &derna_codon_dp::testing::OPrefixWinningCaseCounts,
) {
    println!(
        "diag_o_prefix_winners: lambda_in={:.8} lambda_solver={:.8} total={} finite={} inf={} no_winner={}",
        lambda_in,
        lambda_solver,
        counts.total_cells,
        counts.finite_cells,
        counts.inf_cells,
        counts.no_winner_cells,
    );
    print!("diag_o_prefix_case_at_min:");
    for case in 1..=9 {
        print!(" case{}={}", case, counts.case_at_min(case));
    }
    println!();
    print!("diag_o_prefix_strict_winner:");
    for case in 1..=9 {
        print!(" case{}={}", case, counts.strict_winner_case(case));
    }
    println!();
    if counts.tie_sets.is_empty() {
        println!("diag_o_prefix_tie_sets: none");
    } else {
        for tie_set in &counts.tie_sets {
            let cases = tie_set
                .cases
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join("+");
            println!(
                "diag_o_prefix_tie_set: cases={} count={}",
                cases, tie_set.count
            );
        }
    }
}

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
