use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("derna-codon-dp-{}-{}", std::process::id(), name));
    path
}

#[test]
fn glpe78_runner_writes_telemetry_sidecar() {
    let fasta = temp_path("glagpr.fasta");
    let out = temp_path("pareto_points.jsonl");
    let telemetry = temp_path("pareto_points.telemetry.json");
    let codon_usage = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("codon_usage_human.csv");
    fs::write(&fasta, ">toy\nGLAGPR\n").expect("write toy FASTA");

    let status = Command::new(env!("CARGO_BIN_EXE_run_glpe78"))
        .args([
            "--fasta",
            fasta.to_str().expect("fasta path utf8"),
            "--codon-usage",
            codon_usage.to_str().expect("codon usage path utf8"),
            "--out",
            out.to_str().expect("out path utf8"),
            "--telemetry-out",
            telemetry.to_str().expect("telemetry path utf8"),
        ])
        .status()
        .expect("run_glpe78 starts");
    assert!(status.success());

    let document: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&telemetry).expect("read telemetry sidecar"))
            .expect("parse telemetry sidecar");
    assert_eq!(
        document["total_solves"].as_u64(),
        Some(document["solves"].as_array().expect("solves array").len() as u64)
    );
    assert!(!document["sweep_brackets"]
        .as_array()
        .expect("sweep_brackets array")
        .is_empty());
    assert!(document["sweep_brackets"]
        .as_array()
        .expect("sweep_brackets array")
        .iter()
        .all(|bracket| bracket["stop_reason"].as_str().is_some()));

    let _ = fs::remove_file(fasta);
    let _ = fs::remove_file(out);
    let _ = fs::remove_file(telemetry);
}
