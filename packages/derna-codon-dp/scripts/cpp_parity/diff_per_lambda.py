"""Compare two parity-harness JSONL outputs per λ.

Both sides emit one record per λ with the schema ``{lambda, mfe_kcal_mol,
cai, rna_seq, structure, source}`` (see the README in this directory).
The C++ side is the reference; the other side (typically Rust) is fed the
same λ list and produces one record per λ. Per-λ checks isolate cell-level
correctness from bisection-schedule convergence and hull dedup — the
failure mode is a *specific λ value* with non-matching ``(mfe, cai)``
rather than a hull-set mismatch.

For each λ that appears on both sides the script asserts ``|Δ mfe_kcal_mol| ≤
tol_mfe`` and ``|Δ cai| ≤ tol_cai``. When those hold and the
``(rna_seq, structure)`` pair matches exactly the cell is "exact";
otherwise it is a co-optimal tie and logged informationally — both
solvers found a valid argmin at the same ``(mfe, cai)`` cell but picked
different representatives. Failure is restricted to (a) λs present on
one side but not the other and (b) per-λ tolerance breaches.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    lambda_: float
    mfe_kcal_mol: float
    cai: float
    rna_seq: str
    structure: str
    source: str


def _load_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                Record(
                    lambda_=float(obj["lambda"]),
                    mfe_kcal_mol=float(obj["mfe_kcal_mol"]),
                    cai=float(obj["cai"]),
                    rna_seq=obj["rna_seq"],
                    structure=obj["structure"],
                    source=obj.get("source", path.stem),
                )
            )
    return records


def _index_by_lambda(records: list[Record]) -> dict[float, Record]:
    """First record per λ value, in input order.

    Keys on the float's bit pattern so two records with the same
    floating-point λ collide exactly. Both sides receive the same λ list,
    so exact-bit equality is the expectation; this also avoids the cost
    of a tolerance-based grouping when no tolerance is needed.
    """

    index: dict[float, Record] = {}
    for record in records:
        if record.lambda_ not in index:
            index[record.lambda_] = record
    return index


def _format_cell(record: Record) -> str:
    return f"mfe={record.mfe_kcal_mol:+.4f} cai={record.cai:.6f}"


def _report_orphans(label: str, orphans: list[Record]) -> None:
    print(f"  {label} ({len(orphans)}):")
    for rec in orphans:
        print(f"    λ={rec.lambda_:.6f} {_format_cell(rec)} {rec.source}")


@dataclass(frozen=True)
class IndexedSides:
    """The two JSONL inputs reduced to (λ → record) lookups."""

    cpp: dict[float, Record]
    other: dict[float, Record]
    n_cpp_records: int
    n_other_records: int


@dataclass(frozen=True)
class Tolerances:
    mfe: float
    cai: float


@dataclass(frozen=True)
class Categorized:
    """Outcome buckets for matched λ values, in shared-λ order."""

    exact: list[float]
    co_optimal: list[tuple[Record, Record]]
    mismatched: list[tuple[Record, Record, float, float]]


def _load_sides(cpp_path: Path, other_path: Path) -> IndexedSides:
    cpp_records = _load_jsonl(cpp_path)
    other_records = _load_jsonl(other_path)
    return IndexedSides(
        cpp=_index_by_lambda(cpp_records),
        other=_index_by_lambda(other_records),
        n_cpp_records=len(cpp_records),
        n_other_records=len(other_records),
    )


def _categorize_shared(
    sides: IndexedSides, shared: list[float], tol: Tolerances
) -> Categorized:
    """Bucket each shared λ into exact / co-optimal / mismatched."""

    exact: list[float] = []
    co_optimal: list[tuple[Record, Record]] = []
    mismatched: list[tuple[Record, Record, float, float]] = []
    for lam in shared:
        cpp_rec = sides.cpp[lam]
        other_rec = sides.other[lam]
        delta_mfe = abs(cpp_rec.mfe_kcal_mol - other_rec.mfe_kcal_mol)
        delta_cai = abs(cpp_rec.cai - other_rec.cai)
        if delta_mfe > tol.mfe or delta_cai > tol.cai:
            mismatched.append((cpp_rec, other_rec, delta_mfe, delta_cai))
            continue
        if (
            cpp_rec.rna_seq == other_rec.rna_seq
            and cpp_rec.structure == other_rec.structure
        ):
            exact.append(lam)
        else:
            co_optimal.append((cpp_rec, other_rec))
    return Categorized(exact=exact, co_optimal=co_optimal, mismatched=mismatched)


def _print_mismatches(
    mismatches: list[tuple[Record, Record, float, float]]
) -> None:
    print("\nPER-λ MISMATCH:")
    for cpp_rec, other_rec, delta_mfe, delta_cai in mismatches:
        print(f"  λ={cpp_rec.lambda_:.6f}: Δmfe={delta_mfe:+.6f} Δcai={delta_cai:+.2e}")
        print(f"    cpp:   {_format_cell(cpp_rec)}")
        print(f"           seq={cpp_rec.rna_seq}")
        print(f"           bp ={cpp_rec.structure}")
        print(f"    other: {_format_cell(other_rec)}")
        print(f"           seq={other_rec.rna_seq}")
        print(f"           bp ={other_rec.structure}")


def _print_ties(ties: list[tuple[Record, Record]]) -> None:
    print("\nco-optimal ties (informational; not a failure):")
    for cpp_rec, other_rec in ties[:5]:
        print(f"  λ={cpp_rec.lambda_:.6f} cell {_format_cell(cpp_rec)}")
        print(f"    cpp:   seq={cpp_rec.rna_seq[:32]}… struct={cpp_rec.structure[:32]}…")
        print(f"    other: seq={other_rec.rna_seq[:32]}… struct={other_rec.structure[:32]}…")
    if len(ties) > 5:
        print(f"  … and {len(ties) - 5} more")


@dataclass(frozen=True)
class DiffReport:
    """Inputs to the FAIL / PASS rendering — bundles the three failure modes."""

    cpp_only: list[Record]
    other_only: list[Record]
    result: Categorized

    @property
    def has_orphans(self) -> bool:
        return bool(self.cpp_only) or bool(self.other_only)

    @property
    def is_failure(self) -> bool:
        return self.has_orphans or bool(self.result.mismatched)


def _report_failure(report: DiffReport) -> None:
    if report.has_orphans:
        print("\nλ-SET MISMATCH:")
        _report_orphans("cpp-only", report.cpp_only)
        _report_orphans("other-only", report.other_only)
    if report.result.mismatched:
        _print_mismatches(report.result.mismatched)
    print("\nresult: FAIL")


def diff_per_lambda(
    cpp_path: Path, other_path: Path, tol_mfe: float, tol_cai: float
) -> int:
    sides = _load_sides(cpp_path, other_path)
    cpp_keys = set(sides.cpp)
    other_keys = set(sides.other)
    cpp_only = sorted(
        (sides.cpp[k] for k in cpp_keys - other_keys), key=lambda r: r.lambda_
    )
    other_only = sorted(
        (sides.other[k] for k in other_keys - cpp_keys), key=lambda r: r.lambda_
    )
    shared = sorted(cpp_keys & other_keys)

    result = _categorize_shared(sides, shared, Tolerances(mfe=tol_mfe, cai=tol_cai))

    print(f"cpp: {sides.n_cpp_records} records ({len(sides.cpp)} unique λs)")
    print(f"other: {sides.n_other_records} records ({len(sides.other)} unique λs)")
    print(
        f"shared λs: {len(shared)} "
        f"({len(result.exact)} exact, {len(result.co_optimal)} co-optimal-tie, "
        f"{len(result.mismatched)} mismatched)"
    )

    report = DiffReport(cpp_only=cpp_only, other_only=other_only, result=result)
    if report.is_failure:
        _report_failure(report)
        return 1

    if result.co_optimal:
        _print_ties(result.co_optimal)

    print(
        f"\nresult: PASS — {len(result.exact)} exact "
        f"+ {len(result.co_optimal)} co-optimal-tie / {len(shared)} λs"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpp",
        required=True,
        type=Path,
        help="Path to the C++ JSONL output (the reference side).",
    )
    parser.add_argument(
        "--other",
        required=True,
        type=Path,
        help="Path to the other JSONL output (v1 or Rust side).",
    )
    parser.add_argument(
        "--tolerance-mfe",
        type=float,
        default=0.02,
        help="Tolerance on mfe_kcal_mol (default: 0.02).",
    )
    parser.add_argument(
        "--tolerance-cai",
        type=float,
        default=1e-6,
        help="Tolerance on cai (default: 1e-6).",
    )
    args = parser.parse_args()
    return diff_per_lambda(args.cpp, args.other, args.tolerance_mfe, args.tolerance_cai)


if __name__ == "__main__":
    sys.exit(main())
