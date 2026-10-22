//! Diagnose table-objective accounting for fixed-λ BRD replay records.
//!
//! Reads C++ per-λ parity records and Rust replay records, discovers
//! tolerance-breaching shared λs, reruns only those fixed-λ Rust solves with
//! the CAI-scale λ transform, and writes JSON plus Markdown diagnostics.

use std::fs;
use std::path::{Path, PathBuf};

use derna_codon_dp::testing::{brd_value, solve_brd_objective_diagnostic, FillInputs};
use derna_codon_dp::{load_codon_usage, BrdOptions, BrdSolution, CodonUsage};
use rna_mfe::Params;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

struct Args {
    fasta: PathBuf,
    codon_usage: PathBuf,
    cpp: PathBuf,
    rust_replay: PathBuf,
    out_json: PathBuf,
    out_md: PathBuf,
    cai_scale: f64,
    tolerance_mfe: f64,
    tolerance_cai: f64,
}

#[derive(Clone)]
struct Record {
    lambda_: f64,
    mfe_kcal_mol: f64,
    cai: f64,
    rna_seq: String,
    structure: String,
    source: String,
}

struct Mismatch {
    cpp: Record,
    rust: Record,
    delta_mfe: f64,
    delta_cai: f64,
}

struct DiagnosticRow {
    lambda_cpp: f64,
    lambda_solver: f64,
    cpp: Record,
    rust_replay: Record,
    delta_mfe: f64,
    delta_cai: f64,
    top_o_score_dekacal: f64,
    top_o_x: usize,
    top_o_y: usize,
    top_o_tie_count: usize,
    top_o_within_1_count: usize,
    top_o_within_2_count: usize,
    top_o_next_gap_dekacal: Option<f64>,
    solution: BrdSolution,
    posthoc_score_dekacal: f64,
    table_minus_posthoc_dekacal: f64,
    cpp_single_round_score_dekacal: f64,
    rust_replay_single_round_score_dekacal: f64,
    single_round_margin_dekacal: f64,
    table_vs_cpp_single_margin_dekacal: f64,
    cpp_path_table_drift_lower_bound_dekacal: f64,
    diagnostic_matches_replay: bool,
    classification: &'static str,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1))?;
    let ctx = load_context(&args)?;
    let mismatches = load_mismatches(&args)?;
    if mismatches.is_empty() {
        return Err("no shared-λ mismatches discovered".into());
    }

    eprintln!(
        "found {} shared-λ mismatches; running fixed-λ diagnostics",
        mismatches.len()
    );

    let rows = diagnose_mismatches(&args, &ctx, &mismatches)?;
    write_json(&args.out_json, &args, &rows)?;
    write_markdown(&args.out_md, &args, &rows)?;
    eprintln!("wrote {}", args.out_json.display());
    eprintln!("wrote {}", args.out_md.display());
    Ok(())
}

struct RunContext {
    protein: String,
    codon_usage: CodonUsage,
    params: Params,
    options: BrdOptions,
}

fn load_context(args: &Args) -> Result<RunContext, Box<dyn std::error::Error>> {
    let protein = read_single_protein(&args.fasta)?;
    let codon_usage = load_codon_usage(&args.codon_usage)?;
    let params = Params::load_default().map_err(|e| format!("loading Turner params: {e}"))?;
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    Ok(RunContext {
        protein,
        codon_usage,
        params,
        options,
    })
}

fn load_mismatches(args: &Args) -> Result<Vec<Mismatch>, Box<dyn std::error::Error>> {
    let cpp = load_jsonl(&args.cpp)?;
    let rust_replay = load_jsonl(&args.rust_replay)?;
    Ok(discover_mismatches(
        &cpp,
        &rust_replay,
        args.tolerance_mfe,
        args.tolerance_cai,
    )?)
}

fn diagnose_mismatches(
    args: &Args,
    ctx: &RunContext,
    mismatches: &[Mismatch],
) -> Result<Vec<DiagnosticRow>, Box<dyn std::error::Error>> {
    let mut rows = Vec::with_capacity(mismatches.len());
    for (i, mismatch) in mismatches.iter().enumerate() {
        let lambda_solver = transform_lambda(mismatch.cpp.lambda_, args.cai_scale);
        eprintln!(
            "  [{:>2}/{}] λ_cpp={:.9} λ_solver={:.9}",
            i + 1,
            mismatches.len(),
            mismatch.cpp.lambda_,
            lambda_solver,
        );
        rows.push(diagnose_one(args, ctx, mismatch, lambda_solver)?);
    }
    Ok(rows)
}

fn diagnose_one(
    args: &Args,
    ctx: &RunContext,
    mismatch: &Mismatch,
    lambda_solver: f64,
) -> Result<DiagnosticRow, Box<dyn std::error::Error>> {
    let diagnostic = solve_brd_objective_diagnostic(FillInputs {
        protein: &ctx.protein,
        codon_usage: &ctx.codon_usage,
        lambda_: lambda_solver,
        options: &ctx.options,
        params: &ctx.params,
    })?;
    let cpp_score = single_round_record_score(lambda_solver, ctx.protein.len(), &mismatch.cpp);
    let rust_replay_score =
        single_round_record_score(lambda_solver, ctx.protein.len(), &mismatch.rust);
    let margin = diagnostic.posthoc_score_dekacal - cpp_score;
    let table_vs_cpp = diagnostic.top_o_score_dekacal - cpp_score;
    let cpp_path_drift_lower_bound = table_vs_cpp.max(0.0);
    let next_gap = diagnostic
        .top_o_next_score_dekacal
        .map(|score| score - diagnostic.top_o_score_dekacal);
    let diagnostic_matches_replay = solution_matches_record(
        &diagnostic.solution,
        &mismatch.rust,
        args.tolerance_mfe,
        args.tolerance_cai,
    );
    let classification = classify_row(ClassificationInputs {
        table_vs_cpp_single: table_vs_cpp,
        drift: diagnostic.table_minus_posthoc_dekacal,
        tie_count: diagnostic.top_o_tie_count,
        within_2_count: diagnostic.top_o_within_2_count,
    });
    Ok(DiagnosticRow {
        lambda_cpp: mismatch.cpp.lambda_,
        lambda_solver,
        cpp: mismatch.cpp.clone(),
        rust_replay: mismatch.rust.clone(),
        delta_mfe: mismatch.delta_mfe,
        delta_cai: mismatch.delta_cai,
        top_o_score_dekacal: diagnostic.top_o_score_dekacal,
        top_o_x: diagnostic.top_o_x,
        top_o_y: diagnostic.top_o_y,
        top_o_tie_count: diagnostic.top_o_tie_count,
        top_o_within_1_count: diagnostic.top_o_within_1_count,
        top_o_within_2_count: diagnostic.top_o_within_2_count,
        top_o_next_gap_dekacal: next_gap,
        solution: diagnostic.solution,
        posthoc_score_dekacal: diagnostic.posthoc_score_dekacal,
        table_minus_posthoc_dekacal: diagnostic.table_minus_posthoc_dekacal,
        cpp_single_round_score_dekacal: cpp_score,
        rust_replay_single_round_score_dekacal: rust_replay_score,
        single_round_margin_dekacal: margin,
        table_vs_cpp_single_margin_dekacal: table_vs_cpp,
        cpp_path_table_drift_lower_bound_dekacal: cpp_path_drift_lower_bound,
        diagnostic_matches_replay,
        classification,
    })
}

fn transform_lambda(lambda: f64, scale: f64) -> f64 {
    if scale == 1.0 {
        return lambda;
    }
    scale * lambda / (1.0 + (scale - 1.0) * lambda)
}

fn discover_mismatches(
    cpp: &[Record],
    rust: &[Record],
    tolerance_mfe: f64,
    tolerance_cai: f64,
) -> Result<Vec<Mismatch>, String> {
    let mut out = Vec::new();
    for cpp_record in cpp {
        let Some(rust_record) = rust
            .iter()
            .find(|record| record.lambda_.to_bits() == cpp_record.lambda_.to_bits())
        else {
            continue;
        };
        let delta_mfe = (cpp_record.mfe_kcal_mol - rust_record.mfe_kcal_mol).abs();
        let delta_cai = (cpp_record.cai - rust_record.cai).abs();
        if delta_mfe > tolerance_mfe || delta_cai > tolerance_cai {
            out.push(Mismatch {
                cpp: cpp_record.clone(),
                rust: rust_record.clone(),
                delta_mfe,
                delta_cai,
            });
        }
    }
    if out.len() > 5 {
        return Err(format!(
            "expected at most five replay mismatches, discovered {}",
            out.len()
        ));
    }
    Ok(out)
}

struct ClassificationInputs {
    table_vs_cpp_single: f64,
    drift: f64,
    tie_count: usize,
    within_2_count: usize,
}

fn classify_row(inputs: ClassificationInputs) -> &'static str {
    if inputs.table_vs_cpp_single < 0.0 && inputs.drift < 0.0 {
        "objective-semantics drift"
    } else if inputs.table_vs_cpp_single == 0.0 {
        "integer-resolution tie"
    } else if inputs.table_vs_cpp_single > 0.0 {
        "cpp-path drift lower bound"
    } else if inputs.tie_count > 1 || inputs.within_2_count > 1 {
        "top-cell near tie"
    } else {
        "unclassified"
    }
}

fn single_round_record_score(lambda: f64, protein_len: usize, record: &Record) -> f64 {
    let mfe_dekacal = (record.mfe_kcal_mol * 100.0).round_ties_even() as i32;
    let cai_dekacal = protein_len as f64 * record.cai.ln() * 100.0;
    brd_value(lambda, mfe_dekacal, cai_dekacal)
}

fn solution_matches_record(
    solution: &BrdSolution,
    record: &Record,
    tolerance_mfe: f64,
    tolerance_cai: f64,
) -> bool {
    (solution.mfe_kcal_mol - record.mfe_kcal_mol).abs() <= tolerance_mfe
        && (solution.cai - record.cai).abs() <= tolerance_cai
        && solution.rna_seq == record.rna_seq
        && solution.structure == record.structure
}

fn load_jsonl(path: &Path) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut records = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: invalid JSON: {e}", path.display(), line_no + 1))?;
        records.push(Record {
            lambda_: required_f64(&value, "lambda", path, line_no + 1)?,
            mfe_kcal_mol: required_f64(&value, "mfe_kcal_mol", path, line_no + 1)?,
            cai: required_f64(&value, "cai", path, line_no + 1)?,
            rna_seq: required_str(&value, "rna_seq", path, line_no + 1)?.to_string(),
            structure: required_str(&value, "structure", path, line_no + 1)?.to_string(),
            source: value
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
        });
    }
    Ok(records)
}

fn required_f64(
    value: &serde_json::Value,
    field: &str,
    path: &Path,
    line_no: usize,
) -> Result<f64, String> {
    value
        .get(field)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{}:{line_no}: missing numeric `{field}`", path.display()))
}

fn required_str<'a>(
    value: &'a serde_json::Value,
    field: &str,
    path: &Path,
    line_no: usize,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{}:{line_no}: missing string `{field}`", path.display()))
}

fn read_single_protein(path: &Path) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut records: Vec<String> = Vec::new();
    let mut current = String::new();
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

fn write_json(
    path: &Path,
    args: &Args,
    rows: &[DiagnosticRow],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let document = serde_json::json!({
        "inputs": {
            "fasta": args.fasta.display().to_string(),
            "codon_usage": args.codon_usage.display().to_string(),
            "cpp": args.cpp.display().to_string(),
            "rust_replay": args.rust_replay.display().to_string(),
            "cai_scale": args.cai_scale,
            "tolerance_mfe": args.tolerance_mfe,
            "tolerance_cai": args.tolerance_cai,
        },
        "summary": summary_json(rows),
        "rows": rows.iter().map(row_json).collect::<Vec<_>>(),
    });
    fs::write(path, serde_json::to_string_pretty(&document)?)?;
    Ok(())
}

fn summary_json(rows: &[DiagnosticRow]) -> serde_json::Value {
    let any_table_posthoc_drift = rows
        .iter()
        .any(|row| row.table_minus_posthoc_dekacal != 0.0);
    let all_drift_sufficient = rows.iter().all(|row| {
        row.single_round_margin_dekacal >= 0.0 && row.table_vs_cpp_single_margin_dekacal <= 0.0
    });
    let any_top_ties = rows.iter().any(|row| row.top_o_tie_count > 1);
    let any_near_ties = rows.iter().any(|row| row.top_o_within_2_count > 1);
    let needs_cpp_path_score = rows
        .iter()
        .filter(|row| row.cpp_path_table_drift_lower_bound_dekacal > 0.0)
        .count();
    serde_json::json!({
        "n_mismatches": rows.len(),
        "any_table_posthoc_drift": any_table_posthoc_drift,
        "all_drift_sufficient_to_erase_single_round_cpp_margin": all_drift_sufficient,
        "any_top_o_ties": any_top_ties,
        "any_top_o_near_ties_within_2": any_near_ties,
        "needs_exact_cpp_path_table_score": needs_cpp_path_score,
        "has_direct_bug_evidence": false,
        "blocks_table_cache_optimization": false,
    })
}

fn row_json(row: &DiagnosticRow) -> serde_json::Value {
    serde_json::json!({
        "lambda_cpp": row.lambda_cpp,
        "lambda_solver": row.lambda_solver,
        "cpp": record_json(&row.cpp),
        "rust_replay": record_json(&row.rust_replay),
        "delta_mfe": row.delta_mfe,
        "delta_cai": row.delta_cai,
        "rust_diagnostic": {
            "lambda_solver": row.lambda_solver,
            "top_o_score_dekacal": row.top_o_score_dekacal,
            "top_o_x": row.top_o_x,
            "top_o_y": row.top_o_y,
            "top_o_tie_count": row.top_o_tie_count,
            "top_o_within_1_count": row.top_o_within_1_count,
            "top_o_within_2_count": row.top_o_within_2_count,
            "top_o_next_gap_dekacal": row.top_o_next_gap_dekacal,
            "rna_seq": row.solution.rna_seq.as_str(),
            "structure": row.solution.structure.as_str(),
            "mfe_kcal_mol": row.solution.mfe_kcal_mol,
            "cai": row.solution.cai,
            "posthoc_score_dekacal": row.posthoc_score_dekacal,
            "table_minus_posthoc_dekacal": row.table_minus_posthoc_dekacal,
            "matches_replay_record": row.diagnostic_matches_replay,
        },
        "cpp_single_round_score_dekacal": row.cpp_single_round_score_dekacal,
        "rust_replay_single_round_score_dekacal": row.rust_replay_single_round_score_dekacal,
        "single_round_margin_dekacal": row.single_round_margin_dekacal,
        "table_vs_cpp_single_margin_dekacal": row.table_vs_cpp_single_margin_dekacal,
        "cpp_path_table_drift_lower_bound_dekacal": row.cpp_path_table_drift_lower_bound_dekacal,
        "classification": row.classification,
    })
}

fn record_json(record: &Record) -> serde_json::Value {
    serde_json::json!({
        "lambda": record.lambda_,
        "mfe_kcal_mol": record.mfe_kcal_mol,
        "cai": record.cai,
        "rna_seq": record.rna_seq.as_str(),
        "structure": record.structure.as_str(),
        "source": record.source.as_str(),
    })
}

fn write_markdown(
    path: &Path,
    args: &Args,
    rows: &[DiagnosticRow],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut out = String::new();
    out.push_str("# Fixed-Lambda Objective Accounting\n\n");
    out.push_str("This diagnostic reruns only the shared-lambda replay mismatches discovered by comparing the C++ JSONL against the Rust replay JSONL.\n\n");
    out.push_str("## Inputs\n\n");
    out.push_str(&format!("- C++ records: `{}`\n", args.cpp.display()));
    out.push_str(&format!(
        "- Rust replay records: `{}`\n",
        args.rust_replay.display()
    ));
    out.push_str(&format!("- CAI scale transform: `{}`\n", args.cai_scale));
    out.push_str(&format!("- Mismatches rerun: `{}`\n\n", rows.len()));

    out.push_str("## Result Table\n\n");
    out.push_str("| lambda_cpp | lambda_R | C++ cell | Rust replay cell | top O | post-hoc | replay drift | C++ single | margin | C++ drift LB | O ties / <=2 | classification |\n");
    out.push_str("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for row in rows {
        out.push_str(&format!(
            "| `{:.9}` | `{:.9}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}/{}` | {} |\n",
            row.lambda_cpp,
            row.lambda_solver,
            format_cell(&row.cpp),
            format_cell(&row.rust_replay),
            row.top_o_score_dekacal,
            row.posthoc_score_dekacal,
            row.table_minus_posthoc_dekacal,
            row.cpp_single_round_score_dekacal,
            row.single_round_margin_dekacal,
            row.cpp_path_table_drift_lower_bound_dekacal,
            row.top_o_tie_count,
            row.top_o_within_2_count,
            row.classification,
        ));
    }

    out.push_str("\n## Answers\n\n");
    out.push_str(&answer_table_posthoc_drift(rows));
    out.push('\n');
    out.push_str(&answer_drift_margins(rows));
    out.push('\n');
    out.push_str(&answer_ties(rows));
    out.push('\n');
    out.push_str(&answer_bug_blocker(rows));
    out.push('\n');
    out.push_str("\n## Notes\n\n");
    out.push_str("- `top O` is the exact integer table value selected by the fixed-λ Rust DP.\n");
    out.push_str("- `post-hoc` is the single-round integer score recomputed from the reconstructed RNA, matching the reporting path.\n");
    out.push_str("- `margin` is `Rust post-hoc - C++ single-round`; positive values are C++ wins under the post-hoc approximation.\n");
    out.push_str("- `replay drift` is `top O - post-hoc`; a negative value means the selected Rust table objective is better than the reported single-round score.\n");
    out.push_str("- `C++ drift LB` is the minimum positive table-vs-single-round drift the C++ path must have under Rust's exact table objective. It is a lower bound from Rust optimality, not a direct C++ path reconstruction.\n");
    fs::write(path, out)?;
    Ok(())
}

fn answer_table_posthoc_drift(rows: &[DiagnosticRow]) -> String {
    let drifted = rows
        .iter()
        .filter(|row| row.table_minus_posthoc_dekacal != 0.0)
        .count();
    format!(
        "- Does Rust's top table score differ from its post-hoc single-round score? Yes: `{drifted}/{}` replay cells have nonzero drift.",
        rows.len()
    )
}

fn answer_drift_margins(rows: &[DiagnosticRow]) -> String {
    let enough = rows
        .iter()
        .filter(|row| {
            row.single_round_margin_dekacal >= 0.0 && row.table_vs_cpp_single_margin_dekacal <= 0.0
        })
        .count();
    if enough == rows.len() {
        format!(
            "- Is the drift large enough to flip the 0-2 integer-unit C++ advantage? Yes: `{enough}/{}` rows have Rust's table score no worse than the C++ single-round score after drift.",
            rows.len()
        )
    } else if enough > 0 {
        let lower_bound_rows = rows
            .iter()
            .filter(|row| row.cpp_path_table_drift_lower_bound_dekacal > 0.0)
            .count();
        format!(
            "- Is the drift large enough to flip the 0-2 integer-unit C++ advantage? Directly for `{enough}/{}` rows. The remaining `{lower_bound_rows}` rows require positive C++-path table drift under Rust's objective, with the lower bound shown in the table.",
            rows.len(),
        )
    } else {
        format!(
            "- Is the drift large enough to flip the 0-2 integer-unit C++ advantage? No: `0/{}` rows have Rust's table score no worse than the C++ single-round score after drift.",
            rows.len()
        )
    }
}

fn answer_ties(rows: &[DiagnosticRow]) -> String {
    let ties = rows.iter().filter(|row| row.top_o_tie_count > 1).count();
    let near = rows
        .iter()
        .filter(|row| row.top_o_within_2_count > 1)
        .count();
    format!(
        "- Are top-level `O` ties or near-ties plausible? Exact top-cell ties appear in `{ties}` rows; <=2-unit top-cell neighborhoods appear in `{near}` rows."
    )
}

fn answer_bug_blocker(rows: &[DiagnosticRow]) -> String {
    let lower_bound_rows = rows
        .iter()
        .filter(|row| row.cpp_path_table_drift_lower_bound_dekacal > 0.0)
        .count();
    format!(
        "- Is there evidence of a fixed-lambda objective-accounting bug that should block table-cache optimization? No direct bug evidence from this diagnostic. `{lower_bound_rows}` rows still lack exact C++-path table reconstruction, but Rust optimality bounds the required C++-path drift and keeps the mismatch in the objective-semantics bucket."
    )
}

fn format_cell(record: &Record) -> String {
    format!("({:+.1}, {:.6})", record.mfe_kcal_mol, record.cai)
}

fn parse_args(iter: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut args = Args {
        fasta: PathBuf::from("packages/derna-codon-dp/tests/data/GLPE_HUMAN.fasta"),
        codon_usage: PathBuf::from("packages/derna-codon-dp/data/codon_usage_human.csv"),
        cpp: PathBuf::from("packages/derna-codon-dp/scripts/cpp_parity/cpp/pareto_points.jsonl"),
        rust_replay: PathBuf::from("target/diagnose-brd/glpe78-rust-at-cpp-lambdas.jsonl"),
        out_json: PathBuf::from("target/diagnose-brd/fixed-lambda-objective-accounting.json"),
        out_md: PathBuf::from("target/diagnose-brd/fixed-lambda-objective-accounting.md"),
        cai_scale: 100.0,
        tolerance_mfe: 0.02,
        tolerance_cai: 1e-6,
    };
    let mut it = iter.into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--fasta" => args.fasta = PathBuf::from(require_value(&mut it, "--fasta")?),
            "--codon-usage" => {
                args.codon_usage = PathBuf::from(require_value(&mut it, "--codon-usage")?);
            }
            "--cpp" => args.cpp = PathBuf::from(require_value(&mut it, "--cpp")?),
            "--rust-replay" => {
                args.rust_replay = PathBuf::from(require_value(&mut it, "--rust-replay")?);
            }
            "--out-json" => args.out_json = PathBuf::from(require_value(&mut it, "--out-json")?),
            "--out-md" => args.out_md = PathBuf::from(require_value(&mut it, "--out-md")?),
            "--cai-scale" => {
                args.cai_scale = parse_positive_f64(require_value(&mut it, "--cai-scale")?)?;
            }
            "--tolerance-mfe" => {
                args.tolerance_mfe =
                    parse_positive_f64(require_value(&mut it, "--tolerance-mfe")?)?;
            }
            "--tolerance-cai" => {
                args.tolerance_cai =
                    parse_positive_f64(require_value(&mut it, "--tolerance-cai")?)?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok(args)
}

fn require_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive_f64(value: String) -> Result<f64, String> {
    let parsed: f64 = value.parse().map_err(|e| format!("{value}: {e}"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(format!("expected positive finite float, got {value}"))
    }
}

fn print_usage() {
    eprintln!(
        "diagnose_brd_objective [--fasta <path>] [--codon-usage <path>]\n\
         \x20                      [--cpp <jsonl>] [--rust-replay <jsonl>]\n\
         \x20                      [--out-json <path>] [--out-md <path>]\n\
         \x20                      [--cai-scale <float>]\n\
         \n\
         Discovers shared-λ replay mismatches and reruns only those fixed-λ\n\
         Rust solves with table-objective diagnostics."
    );
}
