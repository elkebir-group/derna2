#!/usr/bin/env python3
"""Generate fixed-lambda anchor fixtures from DERNA C++.

The solver is checked against the published implementation rather than
against a reimplementation of it. Each fixture records what the C++ returns
at one weight, and the Rust test asserts `solve_brd` returns the same
molecule.

**The weight is reparametrized, not copied.** The C++ pairs dekacal energies
with raw log-CAI, where this solver scales log-CAI by 100, so the same
nominal lambda names a different trade-off. Equating the coefficient ratios
gives `lam_cpp = lam / (100 - 99*lam)`, which leaves the argmin unchanged --
the same transform the solver's own `--cai-scale` flag applies in reverse.
A fixture therefore carries both weights, and the Rust-coordinate one is
what the test passes to `solve_brd`.

**What is exact and what is not.** The C++ prints CAI to six significant
digits, so `cai` is a tolerance check. The sequence and the structure are
exact strings and the MFE is read from an integer dekacal field, so those
are exact -- and since CAI is a function of the sequence, asserting the
sequence subsumes it.

`lambda = 0` is deliberately absent. There the MFE term carries zero weight,
so no structure is preferred over any other and the traceback is free to
return any of them; both implementations emit one whose pair count is
inconsistent with the energy they report beside it. That is a real defect,
but it is not this objective's, and pinning a fixture to either answer would
be pinning an arbitrary choice.

Usage:
    python3 gen_anchor_fixtures.py --derna-bin /path/to/derna [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
DEFAULT_OUT = PACKAGE / "tests/data"
CODON_CSV = PACKAGE / "data/codon_usage_human.csv"

# (name, protein-or-fasta, lambda in THIS solver's coordinates)
CASES = [
    ("toy_lambda_25", "GLAGPR", 0.25),
    ("toy_lambda_50", "GLAGPR", 0.5),
    ("toy_lambda_75", "GLAGPR", 0.75),
    ("toy_lambda_100", "GLAGPR", 1.0),
    # The three weights at which the quantized objective returned a design
    # the published implementation does not: the regression this anchor
    # exists to hold.
    ("glpe78_lambda_0234", None, 0.02344703125),
    ("glpe78_lambda_0391", None, 0.03907171875),
    ("glpe78_lambda_1016", None, 0.10157046875),
]
GLPE_FASTA = PACKAGE / "tests/data/GLPE_HUMAN.fasta"


def remap(lam: float) -> float:
    """This solver's weight -> the equivalent weight for the C++."""
    return lam / (100.0 - 99.0 * lam)


def cpp_codon_table(work: Path) -> Path:
    """The canonical Kazusa table in the matrix form the C++ reads.

    Both sides must weigh the same table or the comparison is vacuous, so
    this is a translation of the committed one rather than a second copy.
    """
    out = work / "codon_usage_cpp.csv"
    subprocess.run(
        [sys.executable, str(HERE / "translate_codon_usage.py"),
         "--input", str(CODON_CSV), "--output", str(out)],
        capture_output=True, text=True, check=True,
    )
    return out


def read_protein(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if not line.startswith(">")
    )


def run_cpp(derna_bin: Path, fasta: Path, codon_csv: Path, lam_cpp: float, work: Path) -> dict:
    out = work / "cpp.txt"
    proc = subprocess.run(
        [str(derna_bin), "-i", str(fasta), "-m", "1", "-s", "2",
         "-l", repr(lam_cpp), "-c", str(codon_csv), "-o", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"derna failed at lambda_cpp={lam_cpp!r}: {proc.stderr[-400:]}")
    txt = out.read_text()

    def grab(pattern: str) -> str:
        m = re.search(pattern, txt)
        if not m:
            raise RuntimeError(f"pattern {pattern!r} absent from C++ output")
        return m.group(1)

    return {
        "rna_seq": grab(r"zuker cai rna: ([ACGU]+)"),
        "structure": grab(r"zuker cai bp: ([.()]+)"),
        "mfe_dekacal": int(grab(r"mfe: (-?\d+)")),
        "cai": float(grab(r"Codon Adaptation Index: ([\d.]+)")),
    }


def build_fixture(name: str, protein: str, lam: float, result: dict) -> dict:
    return {
        "source": "derna-cpp",
        "name": name,
        "protein": protein,
        "dangling_5_prime": False,
        "lambda_": lam,
        "lambda_cpp": remap(lam),
        "rna_seq": result["rna_seq"],
        "structure": result["structure"],
        "mfe_kcal_mol": result["mfe_dekacal"] / 100.0,
        "cai": result["cai"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derna-bin", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.derna_bin.exists():
        sys.exit(f"C++ binary not found: {args.derna_bin}")
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        codon_csv = cpp_codon_table(work)
        for name, inline_protein, lam in CASES:
            if inline_protein is None:
                fasta, protein = GLPE_FASTA, read_protein(GLPE_FASTA)
            else:
                protein = inline_protein
                fasta = work / f"{name}.fasta"
                fasta.write_text(f">{name}\n{protein}\n")
            lam_cpp = remap(lam)
            print(f"  {name}: n={len(protein)} lambda={lam} lambda_cpp={lam_cpp:.9f}", flush=True)
            result = run_cpp(args.derna_bin, fasta, codon_csv, lam_cpp, work)
            fixture = build_fixture(name, protein, lam, result)
            path = args.out / f"cpp_anchor_{name}.json"
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
            print(f"    -> {path.name}  mfe={fixture['mfe_kcal_mol']} cai={fixture['cai']}")


if __name__ == "__main__":
    main()
