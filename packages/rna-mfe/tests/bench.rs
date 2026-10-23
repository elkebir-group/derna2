//! Throughput benchmarks for `rna_mfe`.
//!
//! Run with:
//!
//! ```text
//! cargo test --release -p rna-mfe --test bench -- --ignored --nocapture
//! ```
//!
//! All tests are `#[ignore]` so they don't run as part of `cargo test`. Numbers
//! produced here aren't asserted on — they're informational.
//!
//! Methodology: load fixtures once into memory, then run in a tight loop over
//! all of them, repeated until the wall-clock budget is consumed. Per-call
//! mean is reported. The wall-clock budget is generous enough (≥1 s) that the
//! loop is dominated by the evaluator/folder, not by `Instant::now()` overhead.

use std::io::Write as IoWrite;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use rna_mfe::{encode_seq, evaluate, evaluate_encoded, fold, EvalScratch, Nucleotide, Params};
use serde::{Deserialize, Serialize};

const WARMUP: Duration = Duration::from_millis(200);
const MEASURE: Duration = Duration::from_millis(1500);

// ---------------------------------------------------------------------------
// evaluate bench
// ---------------------------------------------------------------------------

#[test]
#[ignore = "throughput benchmark; run with --ignored"]
fn bench_evaluate_dangles_zero() {
    let params = Params::load_default().expect("load Turner 2004");
    let fixtures = load_curated_fixtures();
    println!("loaded {} dangles=0 curated fixtures", fixtures.len());

    let inputs: Vec<(Vec<u8>, &str)> = fixtures
        .iter()
        .map(|f| (f.seq.as_bytes().to_vec(), f.structure.as_str()))
        .collect();

    let long_form = inputs
        .iter()
        .find(|(_, s)| s.len() > 200)
        .expect("long-form fixture (>200 nt) missing");

    bench("long_form_only", |budget| {
        run_for(budget, || {
            evaluate(&params, &long_form.0, long_form.1).unwrap();
            1
        })
    });

    bench("all_47_dangles_zero", |budget| {
        run_for(budget, || {
            for (seq, structure) in &inputs {
                evaluate(&params, seq, structure).unwrap();
            }
            inputs.len()
        })
    });
}

// ---------------------------------------------------------------------------
// fold bench
// ---------------------------------------------------------------------------

#[test]
#[ignore = "throughput benchmark; run with --ignored"]
fn bench_fold_dangles_zero() {
    let params = Params::load_default().expect("load Turner 2004");
    let curated = load_curated_fixtures();
    println!("loaded {} dangles=0 curated fixtures", curated.len());

    let seqs: Vec<Vec<u8>> = curated.iter().map(|f| f.seq.as_bytes().to_vec()).collect();

    let long_form_seq = seqs
        .iter()
        .find(|s| s.len() > 200)
        .expect("long-form fixture (>200 nt) missing");
    println!("long-form sequence: {} nt", long_form_seq.len());

    // Headline number: ms/fold on the longest fixture.
    bench_ms("long_form_only", |budget| {
        run_for(budget, || {
            fold(&params, long_form_seq).unwrap();
            1
        })
    });

    // Length ladder: pick one sequence near each target length to show O(n³) shape.
    for &target_nt in &[20usize, 50, 100, 200] {
        let seq = seqs
            .iter()
            .min_by_key(|s| s.len().abs_diff(target_nt))
            .expect("non-empty corpus");
        let label = format!("ladder_{}nt", seq.len());
        bench_ms(&label, |budget| {
            run_for(budget, || {
                fold(&params, seq).unwrap();
                1
            })
        });
    }

    // Corpus mean: mean ms/fold across all curated sequences.
    bench_ms("all_curated_corpus_mean", |budget| {
        run_for(budget, || {
            for seq in &seqs {
                fold(&params, seq).unwrap();
            }
            seqs.len()
        })
    });
}

// ---------------------------------------------------------------------------
// Shared infrastructure
// ---------------------------------------------------------------------------

fn bench(label: &str, f: impl Fn(Duration) -> Stats) {
    let _ = f(WARMUP);
    let stats = f(MEASURE);
    println!(
        "{label}: {} evaluations in {:.3}s → {:.2} µs/eval, {:.2} M evals/s",
        stats.calls,
        stats.elapsed.as_secs_f64(),
        stats.elapsed.as_secs_f64() * 1e6 / (stats.calls as f64),
        (stats.calls as f64) / stats.elapsed.as_secs_f64() / 1e6,
    );
}

fn bench_ms(label: &str, f: impl Fn(Duration) -> Stats) {
    let _ = f(WARMUP);
    let stats = f(MEASURE);
    println!(
        "{label}: {} folds in {:.3}s → {:.3} ms/fold",
        stats.calls,
        stats.elapsed.as_secs_f64(),
        stats.elapsed.as_secs_f64() * 1e3 / (stats.calls as f64),
    );
}

struct Stats {
    calls: usize,
    elapsed: Duration,
}

fn run_for(budget: Duration, mut iter: impl FnMut() -> usize) -> Stats {
    let start = Instant::now();
    let mut calls = 0usize;
    while start.elapsed() < budget {
        calls += iter();
    }
    Stats {
        calls,
        elapsed: start.elapsed(),
    }
}

fn load_curated_fixtures() -> Vec<CuratedFixture> {
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mfe_fixtures.json",);
    let raw = std::fs::read_to_string(PATH).expect("read mfe_fixtures.json");
    let parsed: CuratedFixtureFile = serde_json::from_str(&raw).expect("parse mfe_fixtures.json");
    parsed
        .fixtures
        .into_iter()
        .filter(|f| !f.dangling_5_prime)
        .collect()
}

#[derive(Debug, Deserialize)]
struct CuratedFixtureFile {
    fixtures: Vec<CuratedFixture>,
}

#[derive(Debug, Deserialize)]
struct CuratedFixture {
    dangling_5_prime: bool,
    seq: String,
    structure: String,
}

// ============================================================================
// Perf-baseline ladder benchmarks
// ============================================================================
//
// Subprocess wrappers time the full call including fork/exec/parse.
//
// Evaluator batching: rust_mfe::evaluate is fast enough at small lengths that
// a single call's wall-clock may be dominated by Instant::now() overhead.
// EVAL_BATCH calls are run per trial; the stored time is divided to yield
// per-call ms. Subprocess calls are never batched (floor is part of the
// measurement contract).
//
// Seq generation: xorshift64 with seed 42 — NOT byte-equivalent to Python's
// random.Random(42). Sequences at each length always start from seed 42, so
// shorter sequences are prefixes of longer ones.
//
// Subprocess floor characterisation: rnafold_d0 / rnaeval_d0 on a 12-nt
// hairpin (FLOOR_SEQ). Ratios at lengths < ~400 nt are floor-distorted;
// see README § Performance.

const LADDER_LENGTHS: &[usize] = &[100, 200, 400, 800, 1500, 3000];
const FOLD_TRIALS: usize = 5;
const EVAL_TRIALS: usize = 5;
const EVAL_BATCH: usize = 100;
const FLOOR_TRIALS: usize = 10;
const FLOOR_SEQ: &str = "GGGGAAAACCCC";
const FLOOR_STRUCT: &str = "((((....))))";

// -- JSON schema --------------------------------------------------------------

#[derive(Serialize, Clone)]
struct PerfRow {
    label: &'static str,
    #[serde(rename = "fn")]
    fn_name: &'static str,
    tool: &'static str,
    len: usize,
    n_trials: usize,
    median_ms: f64,
    p25_ms: f64,
    p75_ms: f64,
}

#[derive(Serialize, Clone)]
struct MfeFloor {
    rnafold_ms: f64,
    rnaeval_ms: f64,
}

#[derive(Serialize)]
struct TrialsPerCell {
    standard_ladder: usize,
    extended_lf_ladder: usize,
    evaluator_ladder: usize,
}

#[derive(Serialize)]
struct PerfSummaryMfe {
    schema_version: u32,
    generated_at: String,
    host: &'static str,
    rust_toolchain: String,
    git_commit: String,
    rng_seed: u64,
    trials_per_cell: TrialsPerCell,
    subprocess_floor_ms: MfeFloor,
    rows: Vec<PerfRow>,
}

// -- subprocess helpers -------------------------------------------------------

fn run_proc(cmd: &str, args: &[&str], stdin_data: &[u8]) -> String {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {cmd}: {e}"));
    child.stdin.take().unwrap().write_all(stdin_data).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{cmd} exit {:?}", out.status);
    String::from_utf8(out.stdout).unwrap()
}

// Both ViennaRNA outputs "<seq>\n<structure> ( <energy>)\n".
// rfind('(') finds the energy-notation '(' which always follows the structure.
// Hand-verified output: printf 'GGGGAAAACCCC\n' | RNAfold -d 0 --noPS
//   → "GGGGAAAACCCC\n((((....)))) ( -5.40)\n"
fn parse_dot_bracket(output: &str) -> (String, f64) {
    let line = output
        .lines()
        .find(|l| !l.trim().is_empty() && l.trim_start().starts_with(['(', '.']))
        .expect("no dot-bracket line in output");
    let i = line.rfind('(').expect("no '(' in dot-bracket line");
    let structure = line[..i].trim().to_string();
    let energy: f64 = line[i + 1..line.len() - 1]
        .trim()
        .parse()
        .expect("parse energy");
    (structure, energy)
}

fn rnafold_d0(seq: &str) -> (String, f64) {
    // Invoked as `RNAfold -d 0 --noPS` with the sequence on stdin.
    let cmd = std::env::var("RNAFOLD").unwrap_or_else(|_| "RNAfold".into());
    let out = run_proc(&cmd, &["-d", "0", "--noPS"], format!("{seq}\n").as_bytes());
    parse_dot_bracket(&out)
}

// Hand-verified output format:
//   printf 'GGGGAAAACCCC\n((((....))))\n' | RNAeval -d 0
//   → "GGGGAAAACCCC\n((((....)))) ( -5.40)\n"
fn rnaeval_d0(seq: &str, structure: &str) -> f64 {
    let cmd = std::env::var("RNAEVAL").unwrap_or_else(|_| "RNAeval".into());
    let stdin_data = format!("{seq}\n{structure}\n");
    let out = run_proc(&cmd, &["-d", "0"], stdin_data.as_bytes());
    let (_, energy) = parse_dot_bracket(&out);
    energy
}

// -- sequence generation ------------------------------------------------------

fn synth_rna(len: usize) -> Vec<u8> {
    const BASES: &[u8] = b"ACGU";
    let mut s: u64 = 42;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            BASES[(s % 4) as usize]
        })
        .collect()
}

// -- trial protocol -----------------------------------------------------------

// Runs n timed trials (1 untimed warmup first). Each trial executes `batch`
// calls and returns the per-call time in ms.
fn time_n_trials<F: FnMut()>(n: usize, batch: usize, mut f: F) -> Vec<f64> {
    f(); // untimed warmup
    (0..n)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..batch {
                f();
            }
            t.elapsed().as_secs_f64() * 1e3 / batch as f64
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac
}

fn aggregate(mut times: Vec<f64>) -> (f64, f64, f64) {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        percentile(&times, 0.5),
        percentile(&times, 0.25),
        percentile(&times, 0.75),
    )
}

fn time_cell(
    label: &'static str,
    fn_name: &'static str,
    tool: &'static str,
    len: usize,
    n: usize,
    batch: usize,
    f: impl FnMut(),
) -> PerfRow {
    let (median, p25, p75) = aggregate(time_n_trials(n, batch, f));
    println!(
        "{label}[{len:>5}nt]:  {tool:<10} N={n}  median {median:8.2} ms  IQR [{p25:.2}, {p75:.2}]"
    );
    PerfRow {
        label,
        fn_name,
        tool,
        len,
        n_trials: n,
        median_ms: median,
        p25_ms: p25,
        p75_ms: p75,
    }
}

// -- ladder runners -----------------------------------------------------------

fn fold_ladder_rows(params: &Params) -> Vec<PerfRow> {
    let mut rows = Vec::new();
    for &len in LADDER_LENGTHS {
        let seq = synth_rna(len);
        let seq_str = String::from_utf8(seq.clone()).unwrap();
        rows.push(time_cell(
            "ladder_4way",
            "fold",
            "vienna",
            len,
            FOLD_TRIALS,
            1,
            || {
                rnafold_d0(&seq_str);
            },
        ));
        rows.push(time_cell(
            "ladder_4way",
            "fold",
            "rust_mfe",
            len,
            FOLD_TRIALS,
            1,
            || {
                fold(params, &seq).unwrap();
            },
        ));
    }
    rows
}

fn eval_ladder_rows(params: &Params) -> Vec<PerfRow> {
    let mut rows = Vec::new();
    for &len in LADDER_LENGTHS {
        let seq = synth_rna(len);
        let seq_str = String::from_utf8(seq.clone()).unwrap();
        let (structure, _) = fold(params, &seq).unwrap();
        // Pre-encode once; reuse scratch across all trials — this is the
        // zero-alloc hot path that a codon-design DP uses.
        let nt_seq: Vec<Nucleotide> = encode_seq(&seq).expect("valid synth sequence");
        let mut scratch = EvalScratch::with_capacity(len);
        // rnaeval_d0: 1 call per trial (subprocess floor dominates at small lengths).
        // rust_mfe evaluate_encoded: EVAL_BATCH calls per trial; time is divided to per-call ms.
        rows.push(time_cell(
            "ladder_evaluator",
            "evaluate",
            "rnaeval",
            len,
            EVAL_TRIALS,
            1,
            || {
                rnaeval_d0(&seq_str, &structure);
            },
        ));
        rows.push(time_cell(
            "ladder_evaluator",
            "evaluate",
            "rust_mfe",
            len,
            EVAL_TRIALS,
            EVAL_BATCH,
            || {
                evaluate_encoded(params, &nt_seq, &structure, &mut scratch).unwrap();
            },
        ));
    }
    rows
}

fn measure_floor<F: FnMut()>(n: usize, f: F) -> f64 {
    let (median, _, _) = aggregate(time_n_trials(n, 1, f));
    median
}

fn measure_mfe_floors() -> MfeFloor {
    let rnafold_ms = measure_floor(FLOOR_TRIALS, || {
        rnafold_d0(FLOOR_SEQ);
    });
    println!(
        "RNAfold floor ({} nt):  {rnafold_ms:.2} ms",
        FLOOR_SEQ.len()
    );
    let rnaeval_ms = measure_floor(FLOOR_TRIALS, || {
        rnaeval_d0(FLOOR_SEQ, FLOOR_STRUCT);
    });
    println!(
        "RNAeval floor ({} nt):  {rnaeval_ms:.2} ms",
        FLOOR_SEQ.len()
    );
    MfeFloor {
        rnafold_ms,
        rnaeval_ms,
    }
}

// -- metadata helpers ---------------------------------------------------------

fn cmd_output_trimmed(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn git_short_sha() -> String {
    cmd_output_trimmed("git", &["rev-parse", "--short", "HEAD"])
}

fn rust_toolchain_version() -> String {
    cmd_output_trimmed("rustc", &["--version"])
}

fn today_iso() -> String {
    cmd_output_trimmed("date", &["+%Y-%m-%d"])
}

// -- JSON write ---------------------------------------------------------------

fn write_perf_baseline_mfe(floor: &MfeFloor, rows: &[PerfRow]) {
    let summary = PerfSummaryMfe {
        schema_version: 1,
        generated_at: today_iso(),
        host: "Apple M-series, macOS",
        rust_toolchain: rust_toolchain_version(),
        git_commit: git_short_sha(),
        rng_seed: 42,
        trials_per_cell: TrialsPerCell {
            standard_ladder: FOLD_TRIALS,
            extended_lf_ladder: 0,
            evaluator_ladder: EVAL_TRIALS,
        },
        subprocess_floor_ms: floor.clone(),
        rows: rows.to_vec(),
    };
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/perf_baseline.json");
    std::fs::write(PATH, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
}

// -- test ---------------------------------------------------------------------

#[test]
#[ignore = "perf baseline; writes tests/data/perf_baseline.json; run with: cargo test --release -p rna-mfe --test bench -- --ignored --nocapture"]
fn bench_perf_baseline() {
    let params = Params::load_default().expect("load Turner 2004");
    println!("\n=== ladder_4way (fold: vienna vs rust_mfe) ===");
    let mut rows = fold_ladder_rows(&params);
    println!("\n=== ladder_evaluator (evaluate: rnaeval vs rust_mfe) ===");
    rows.extend(eval_ladder_rows(&params));
    println!("\n=== subprocess floors ===");
    let floor = measure_mfe_floors();
    write_perf_baseline_mfe(&floor, &rows);
    println!(
        "\nwrote tests/data/perf_baseline.json ({} rows)",
        rows.len()
    );
}
