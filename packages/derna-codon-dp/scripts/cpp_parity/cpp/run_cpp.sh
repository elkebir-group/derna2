#!/usr/bin/env bash
# Idempotent driver: build DERNA C++, run lambda_sweep_2 on GLPE_HUMAN, project CSV → JSONL.
#
# Pinned upstream commit: adecda8277add8c0b75512434bd729f30b8fd7d6
# Repo: https://github.com/elkebir-group/derna
#
# Run from the project root:
#   bash packages/derna-codon-dp/scripts/cpp_parity/cpp/run_cpp.sh
set -euo pipefail

CPP_REPO="https://github.com/elkebir-group/derna"
CPP_COMMIT="adecda8277add8c0b75512434bd729f30b8fd7d6"
CPP_DIR="docs/reference/_cpp/derna-main"
CPP_BUILD="$CPP_DIR/build"
CPP_BIN="$CPP_BUILD/derna"
SCRIPT_DIR="packages/derna-codon-dp/scripts/cpp_parity/cpp"
CODON_USAGE_LONG="packages/derna-codon-dp/data/codon_usage_human.csv"
CODON_USAGE_CPP="$SCRIPT_DIR/codon_usage_human_cpp.csv"
FASTA="packages/derna-codon-dp/tests/data/GLPE_HUMAN.fasta"
RUN_LOG="$SCRIPT_DIR/run_log.txt"
SWEEP_BASE="$SCRIPT_DIR/sweep"

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

# ── 1. Clone (skip if already present) ──────────────────────────────────────
if [ ! -d "$CPP_DIR/.git" ]; then
  echo "Cloning DERNA C++ at $CPP_COMMIT …"
  git clone "$CPP_REPO" "$CPP_DIR"
  (cd "$CPP_DIR" && git checkout "$CPP_COMMIT")
else
  echo "C++ source already present at $CPP_DIR — skipping clone."
fi

# ── 2. Build (skip if binary is newer than all sources) ──────────────────────
NEED_BUILD=false
if [ ! -f "$CPP_BIN" ]; then
  NEED_BUILD=true
else
  # Rebuild if any source file is newer than the binary
  if find "$CPP_DIR/src" -name "*.cpp" -newer "$CPP_BIN" | grep -q .; then
    NEED_BUILD=true
  fi
fi

if [ "$NEED_BUILD" = true ]; then
  echo "Building DERNA C++ …"
  cmake -S "$CPP_DIR" -B "$CPP_BUILD" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  cmake --build "$CPP_BUILD"
else
  echo "Binary $CPP_BIN is up-to-date — skipping build."
fi

# ── 3. Translate codon usage (cheap, always re-run) ──────────────────────────
echo "Translating codon usage …"
python3 packages/derna-codon-dp/scripts/cpp_parity/translate_codon_usage.py \
  --input "$CODON_USAGE_LONG" \
  --output "$CODON_USAGE_CPP"

# ── 4. Run the sweep ──────────────────────────────────────────────────────────
echo "Running lambda_sweep_2 on GLPE_HUMAN …"
START=$(date +%s)
"$CPP_BIN" \
  -i "$FASTA" \
  -o "$RUN_LOG" \
  -O "$SWEEP_BASE" \
  -m 1 -s 3 \
  -t 0.0025 -p 0.00075 \
  -c "$CODON_USAGE_CPP"
END=$(date +%s)
echo "Sweep complete in $((END - START))s. CSV at ${SWEEP_BASE}.csv"

# ── 5. Project CSV → JSONL ────────────────────────────────────────────────────
echo "Projecting CSV to JSONL …"
python3 "$SCRIPT_DIR/project_csv_to_jsonl.py" \
  --input "${SWEEP_BASE}.csv" \
  --output "$SCRIPT_DIR/pareto_points.jsonl"

echo "Done. pareto_points.jsonl at $SCRIPT_DIR/pareto_points.jsonl"
