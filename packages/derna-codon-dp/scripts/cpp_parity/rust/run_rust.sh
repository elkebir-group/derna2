#!/usr/bin/env bash
# Idempotent driver: build derna-codon-dp in release mode, run the GLPE-78
# sweep, emit pareto_points.jsonl conformant to the parity-harness contract
# (`packages/derna-codon-dp/scripts/cpp_parity/README.md`). Pairs with
# `cpp/run_cpp.sh` as the other source in the harness.
#
# Run from the project root:
#   bash packages/derna-codon-dp/scripts/cpp_parity/rust/run_rust.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SCRIPT_DIR="packages/derna-codon-dp/scripts/cpp_parity/rust"
FASTA="packages/derna-codon-dp/tests/data/GLPE_HUMAN.fasta"
CODON_USAGE="packages/derna-codon-dp/data/codon_usage_human.csv"
OUT="$SCRIPT_DIR/pareto_points.jsonl"
MACHINE_LABEL="${MACHINE_LABEL:-$(uname -n)}"

# Override the input FASTA via `--fasta <path>`; output paths are unchanged.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --fasta)
      FASTA="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Building derna-codon-dp (release)…"
cargo build --workspace --release

echo "Running solve_pord sweep on GLPE_HUMAN…"
START=$(date +%s)
./target/release/run_glpe78 \
    --fasta "$FASTA" \
    --codon-usage "$CODON_USAGE" \
    --out "$OUT" \
    --machine "$MACHINE_LABEL"
END=$(date +%s)
echo "Sweep complete in $((END - START))s. JSONL at $OUT"
