#!/usr/bin/env python3
"""Summarize Rust/C++ lambda-sweep solve-count attribution.

The script is intentionally stdlib-only so it can run in a fresh checkout. It
accepts both current telemetry sidecars with explicit bracket traces and older
sidecars that only contain per-lambda solve records; for older files it
reconstructs the FIFO Rust bracket queue from the solve sequence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RUST_TELEMETRY = Path(
    "packages/derna-codon-dp/scripts/cpp_parity/rust/pareto_points.telemetry.json"
)
DEFAULT_CPP_JSONL = Path("packages/derna-codon-dp/scripts/cpp_parity/cpp/pareto_points.jsonl")
DEFAULT_CPP_LOG = Path("packages/derna-codon-dp/scripts/cpp_parity/cpp/run_log.txt")

EPSILON = 1e-5
MFE_TOLERANCE = 0.02
CAI_TOLERANCE = 1e-6
THRESHOLD = 0.0025
THRESHOLD2 = 0.00075


@dataclass(frozen=True)
class Cell:
    mfe_kcal_mol: float
    cai: float


@dataclass(frozen=True)
class Solve:
    lambda_: float
    cell: Cell


@dataclass(frozen=True)
class BracketTrace:
    lo: float
    hi: float
    mid: float | None
    lo_cell: Cell
    hi_cell: Cell
    mfe_differs: bool
    cai_differs: bool
    width: float
    width_threshold: float
    stop_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-telemetry", type=Path, default=DEFAULT_RUST_TELEMETRY)
    parser.add_argument("--cpp-jsonl", type=Path, default=DEFAULT_CPP_JSONL)
    parser.add_argument("--cpp-log", type=Path, default=DEFAULT_CPP_LOG)
    parser.add_argument(
        "--write-brackets",
        type=Path,
        help="write reconstructed current Rust bracket trace JSON",
    )
    return parser.parse_args()


def load_rust_solves(path: Path) -> tuple[list[Solve], list[dict]]:
    document = json.loads(path.read_text())
    solves = [
        Solve(
            lambda_=float(item["lambda"]),
            cell=Cell(float(item["mfe_kcal_mol"]), float(item["cai"])),
        )
        for item in document["solves"]
    ]
    return solves, list(document.get("sweep_brackets", []))


def load_cpp_records(path: Path) -> list[Solve]:
    records: list[Solve] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        records.append(
            Solve(
                lambda_=float(item["lambda"]),
                cell=Cell(float(item["mfe_kcal_mol"]), float(item["cai"])),
            )
        )
    return records


def count_cpp_dp_solves(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if "Time taken by DP" in line)


def cell_key(cell: Cell) -> tuple[float, float]:
    return (cell.mfe_kcal_mol, cell.cai)


def unique_cell_count(cells: Iterable[Cell]) -> int:
    return len({cell_key(cell) for cell in cells})


def same_cell(a: Cell, b: Cell) -> bool:
    return (
        abs(a.mfe_kcal_mol - b.mfe_kcal_mol) <= MFE_TOLERANCE
        and abs(a.cai - b.cai) <= CAI_TOLERANCE
    )


def lambda_key(lambda_: float) -> str:
    return float(lambda_).hex()


def solve_lookup(solves: list[Solve]) -> dict[str, Solve]:
    return {lambda_key(solve.lambda_): solve for solve in solves}


def get_solve(lookup: dict[str, Solve], lambda_: float) -> Solve:
    direct = lookup.get(lambda_key(lambda_))
    if direct is not None:
        return direct
    for solve in lookup.values():
        if abs(solve.lambda_ - lambda_) < 1e-15:
            return solve
    raise KeyError(f"telemetry does not contain lambda {lambda_:.17g}")


def bracket_decision(
    lo_solve: Solve,
    hi_solve: Solve,
    mode: str,
) -> tuple[float | None, bool, bool, float, str]:
    mfe_differs = abs(lo_solve.cell.mfe_kcal_mol - hi_solve.cell.mfe_kcal_mol) > EPSILON
    cai_differs = abs(lo_solve.cell.cai - hi_solve.cell.cai) > EPSILON
    differs = (mfe_differs or cai_differs) if mode == "or" else (mfe_differs and cai_differs)
    width = abs(hi_solve.lambda_ - lo_solve.lambda_)
    width_threshold = THRESHOLD2 if hi_solve.lambda_ < THRESHOLD else THRESHOLD
    if not differs:
        reason = "same_solution" if not (mfe_differs or cai_differs) else "single_axis_diff"
        return None, mfe_differs, cai_differs, width_threshold, reason
    if width > width_threshold:
        return 0.5 * (lo_solve.lambda_ + hi_solve.lambda_), mfe_differs, cai_differs, width_threshold, "bisect"
    reason = "near_zero_width_threshold" if hi_solve.lambda_ < THRESHOLD else "width_threshold"
    return None, mfe_differs, cai_differs, width_threshold, reason


def replay_rust(solves: list[Solve], mode: str) -> tuple[list[Solve], list[BracketTrace]]:
    lookup = solve_lookup(solves)
    queue: list[tuple[float, float]] = [(EPSILON, 1.0 - EPSILON)]
    selected: dict[str, Solve] = {
        lambda_key(EPSILON): get_solve(lookup, EPSILON),
        lambda_key(1.0 - EPSILON): get_solve(lookup, 1.0 - EPSILON),
    }
    brackets: list[BracketTrace] = []
    while queue:
        lo, hi = queue.pop(0)
        lo_solve = get_solve(lookup, lo)
        hi_solve = get_solve(lookup, hi)
        mid, mfe_differs, cai_differs, width_threshold, reason = bracket_decision(
            lo_solve, hi_solve, mode
        )
        brackets.append(
            BracketTrace(
                lo=lo_solve.lambda_,
                hi=hi_solve.lambda_,
                mid=mid,
                lo_cell=lo_solve.cell,
                hi_cell=hi_solve.cell,
                mfe_differs=mfe_differs,
                cai_differs=cai_differs,
                width=abs(hi_solve.lambda_ - lo_solve.lambda_),
                width_threshold=width_threshold,
                stop_reason=reason,
            )
        )
        if mid is None:
            continue
        selected[lambda_key(mid)] = get_solve(lookup, mid)
        queue.append((lo, mid))
        queue.append((mid, hi))
    ordered = sorted(selected.values(), key=lambda solve: solve.lambda_)
    return ordered, brackets


def reason_counts(brackets: Iterable[BracketTrace]) -> Counter[str]:
    return Counter(bracket.stop_reason for bracket in brackets)


def missing_cpp_cells(cpp_records: list[Solve], rust_solves: list[Solve]) -> list[Cell]:
    missing: list[Cell] = []
    rust_cells = [solve.cell for solve in rust_solves]
    seen_missing: set[tuple[float, float]] = set()
    for cpp in cpp_records:
        if any(same_cell(cpp.cell, rust_cell) for rust_cell in rust_cells):
            continue
        key = cell_key(cpp.cell)
        if key not in seen_missing:
            missing.append(cpp.cell)
            seen_missing.add(key)
    return missing


def shared_prefix(a: list[float], b: list[float]) -> int:
    length = 0
    for left, right in zip(a, b):
        if abs(left - right) > 1e-12:
            break
        length += 1
    return length


def remap_cpp_lambda_to_rust(lambda_cpp: float, cai_scale: float = 100.0) -> float:
    return cai_scale * lambda_cpp / (1.0 + (cai_scale - 1.0) * lambda_cpp)


def write_brackets(path: Path, brackets: list[BracketTrace]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "lo": bracket.lo,
            "hi": bracket.hi,
            "mid": bracket.mid,
            "lo_cell": {
                "mfe_kcal_mol": bracket.lo_cell.mfe_kcal_mol,
                "cai": bracket.lo_cell.cai,
            },
            "hi_cell": {
                "mfe_kcal_mol": bracket.hi_cell.mfe_kcal_mol,
                "cai": bracket.hi_cell.cai,
            },
            "mfe_differs": bracket.mfe_differs,
            "cai_differs": bracket.cai_differs,
            "width": bracket.width,
            "width_threshold": bracket.width_threshold,
            "stop_reason": bracket.stop_reason,
        }
        for bracket in brackets
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def format_lambdas(lambdas: Iterable[float]) -> str:
    return " ".join(f"{lambda_:.12g}" for lambda_ in lambdas)


def main() -> None:
    args = parse_args()
    rust_solves, bracket_telemetry = load_rust_solves(args.rust_telemetry)
    cpp_records = load_cpp_records(args.cpp_jsonl)
    cpp_dp_solves = count_cpp_dp_solves(args.cpp_log)

    rust_or_solves, rust_or_brackets = replay_rust(rust_solves, "or")
    rust_and_solves, rust_and_brackets = replay_rust(rust_solves, "and")
    actual_bracket_count = len(bracket_telemetry) if bracket_telemetry else len(rust_or_brackets)

    cpp_missing_from_or = missing_cpp_cells(cpp_records, rust_or_solves)
    cpp_missing_from_and = missing_cpp_cells(cpp_records, rust_and_solves)
    prefix = shared_prefix(
        [solve.lambda_ for solve in rust_solves],
        [solve.lambda_ for solve in cpp_records],
    )

    print("Rust existing telemetry")
    print(f"  solves: {len(rust_solves)}")
    print(f"  unique solve cells: {unique_cell_count(solve.cell for solve in rust_solves)}")
    print(
        "  duplicate cell visits: "
        f"{len(rust_solves) - unique_cell_count(solve.cell for solve in rust_solves)}"
    )
    print(f"  bracket records: {actual_bracket_count}")
    print(f"  OR stop reasons: {dict(reason_counts(rust_or_brackets))}")
    print()
    print("C++ existing artifacts")
    print(f"  DP solves from run_log: {cpp_dp_solves}")
    print(f"  JSONL records: {len(cpp_records)}")
    print(f"  unique cells: {unique_cell_count(solve.cell for solve in cpp_records)}")
    print()
    print("C++ AND endpoint-difference replay on Rust telemetry")
    print(f"  solves: {len(rust_and_solves)}")
    print(f"  unique solve cells: {unique_cell_count(solve.cell for solve in rust_and_solves)}")
    print(
        "  duplicate cell visits: "
        f"{len(rust_and_solves) - unique_cell_count(solve.cell for solve in rust_and_solves)}"
    )
    print(f"  stop reasons: {dict(reason_counts(rust_and_brackets))}")
    print(f"  missing C++ cells under OR replay: {len(cpp_missing_from_or)}")
    print(f"  missing C++ cells under AND replay: {len(cpp_missing_from_and)}")
    print()
    print("Lambda sequence divergence")
    print(f"  shared raw prefix length: {prefix}")
    if prefix < min(len(rust_solves), len(cpp_records)):
        print(f"  next Rust lambda: {rust_solves[prefix].lambda_:.12g}")
        print(f"  next C++ lambda: {cpp_records[prefix].lambda_:.12g}")
    print(f"  C++ lambdas: {format_lambdas(solve.lambda_ for solve in cpp_records)}")
    print(
        "  C++ lambdas remapped to Rust scale: "
        f"{format_lambdas(remap_cpp_lambda_to_rust(solve.lambda_) for solve in cpp_records)}"
    )

    if args.write_brackets is not None:
        write_brackets(args.write_brackets, rust_or_brackets)
        print()
        print(f"wrote reconstructed Rust OR bracket trace: {args.write_brackets}")


if __name__ == "__main__":
    main()
