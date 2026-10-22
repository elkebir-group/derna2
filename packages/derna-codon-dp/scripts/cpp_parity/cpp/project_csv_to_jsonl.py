"""Project the DERNA C++ lambda_sweep_2 CSV output to the parity JSONL format.

Reads sweep.csv produced by the C++ binary, drops rows where gamma != "None"
(suboptimal samples), and emits pareto_points.jsonl with schema:
  {"lambda", "mfe_kcal_mol", "cai", "rna_seq", "structure", "source": "cpp"}

Column mapping:
  C++ lambda  -> lambda
  C++ mfe     -> mfe_kcal_mol   (dekacal int → kcal/mol: divide by 100)
  C++ sCAI    -> cai            (standard CAI in [0, 1]; NOT the log-form CAI column)
  C++ seq     -> rna_seq
  C++ bp      -> structure

Note: ZukerAlgorithm.calculate_W() returns energies in dekacal units (0.01 kcal/mol per
term), consistent with Turner 2004 integer parameters. v1 also uses dekacal internally
but converts to true kcal/mol in ParetoPoint.mfe_kcal_mol. This script applies the same /100
conversion here so both JSONL files share the same unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def project(input_path: Path, output_path: Path) -> int:
    written = 0
    with input_path.open(newline="") as f_in, output_path.open("w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if row["gamma"] != "None":
                continue
            record = {
                "lambda": float(row["lambda"]),
                "mfe_kcal_mol": float(row["mfe"]) / 100.0,
                "cai": float(row["sCAI"]),
                "rna_seq": row["seq"],
                "structure": row["bp"],
                "source": "cpp",
            }
            f_out.write(json.dumps(record) + "\n")
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("packages/derna-codon-dp/scripts/cpp_parity/cpp/sweep.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("packages/derna-codon-dp/scripts/cpp_parity/cpp/pareto_points.jsonl"),
    )
    args = parser.parse_args()
    n = project(args.input, args.output)
    print(f"Wrote {n} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
