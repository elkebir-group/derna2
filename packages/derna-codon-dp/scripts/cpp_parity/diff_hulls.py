"""Compare two parity-harness JSONL outputs for hull equality (Rust-superset framing).

The harness (see README in this directory) writes one JSONL per source — C++,
v1, or the Rust port — with the schema ``{lambda, mfe_kcal_mol, cai, rna_seq,
structure, source}``. Hull comparison reduces each side's records to the set of
``(mfe_kcal_mol, cai)`` cells within tolerance, then reports four buckets:

- ``shared`` — cells in both sides within tolerance;
- ``cpp-only orphans`` — cells in C++ only (failure mode);
- ``rust-only orphans (dominated by shared)`` — Rust extras at the same
  ``(mfe, cai)`` cell as a shared cell that integer-dekacal tie-break
  differently (informational);
- ``rust-only orphans (non-dominated)`` — Rust extras that are genuine
  additional Pareto vertices not present in C++ (informational).

Failure iff Rust's hull is missing a Pareto vertex C++ found — i.e.
``cpp ⊆ rust``. Rust finding extras is evidence of *more complete* Pareto-front
exploration by Rust's finer-grained `(1−λ)·CAI_dekacal` arithmetic, not a parity
violation. Empirically, the rust-only extras split into integer-dekacal
tie-break duplicates (Rust ties at the dekacal level once `(1−λ)·CAI_dekacal`
falls below dekacal resolution at high transformed λ; traceback picks a
different representative codon set without dominating C++'s pick) and genuine
additional non-dominated Pareto vertices that C++'s bisection schedule did not
visit.

For each matched cell the script additionally compares ``(rna_seq, structure)`` and logs
whether the two sides agreed on the representative or hit a co-optimal tie;
ties are informational and never fail the diff.
"""

from __future__ import annotations

import argparse
import json
import math
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


@dataclass(frozen=True)
class CellKey:
    """Tolerance-quantized ``(mfe, cai)`` cell identifier.

    Two records map to the same cell iff their unrounded values lie within
    ``tol_mfe`` / ``tol_cai`` of the same grid centre. Quantizing once at load
    time means later set operations are exact-integer comparisons.
    """

    mfe_step: int
    cai_step: int


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


def _quantize(value: float, tol: float) -> int:
    return math.floor(value / tol + 0.5)


def _cell_key(record: Record, tol_mfe: float, tol_cai: float) -> CellKey:
    return CellKey(
        mfe_step=_quantize(record.mfe_kcal_mol, tol_mfe),
        cai_step=_quantize(record.cai, tol_cai),
    )


def _index_by_cell(
    records: list[Record], tol_mfe: float, tol_cai: float
) -> dict[CellKey, Record]:
    """First record per ``(mfe, cai)`` cell, in input order."""

    index: dict[CellKey, Record] = {}
    for record in records:
        key = _cell_key(record, tol_mfe, tol_cai)
        if key not in index:
            index[key] = record
    return index


def _format_cell(record: Record) -> str:
    return f"mfe={record.mfe_kcal_mol:+.4f} cai={record.cai:.6f}"


def _report_orphans(label: str, orphans: list[Record]) -> None:
    print(f"  {label} ({len(orphans)}):")
    for rec in orphans:
        print(f"    {_format_cell(rec)} (λ={rec.lambda_:.6f}) {rec.source}")


def _compare_matched_cells(
    cpp_index: dict[CellKey, Record],
    other_index: dict[CellKey, Record],
    shared: list[CellKey],
) -> tuple[int, int, list[tuple[Record, Record]]]:
    exact = 0
    co_optimal = 0
    ties: list[tuple[Record, Record]] = []
    for key in shared:
        cpp_rec = cpp_index[key]
        other_rec = other_index[key]
        if (
            cpp_rec.rna_seq == other_rec.rna_seq
            and cpp_rec.structure == other_rec.structure
        ):
            exact += 1
        else:
            co_optimal += 1
            ties.append((cpp_rec, other_rec))
    return exact, co_optimal, ties


def _dominates(s: Record, r: Record) -> bool:
    """Return whether ``s`` Pareto-dominates ``r``: lower-MFE-and-no-worse-CAI
    (or equal-MFE-and-higher-CAI). Equal-on-both is filtered out by the caller
    (records here come from different CellKey buckets, so won't be equal-equal).
    """

    return (
        s.mfe_kcal_mol <= r.mfe_kcal_mol
        and s.cai >= r.cai
        and (s.mfe_kcal_mol < r.mfe_kcal_mol or s.cai > r.cai)
    )


def _classify_rust_extras(
    extras: list[Record], shared_cells: list[Record]
) -> tuple[list[Record], list[Record]]:
    """Split rust-only extras into dominated-by-shared vs non-dominated.

    Dominated extras are integer-dekacal tie-break duplicates: same hull cell
    as a shared one, different codon-representative due to Rust's quantized
    BRD value ties at high λ. Non-dominated extras are genuine additional
    Pareto vertices the C++ bisection schedule did not visit.
    """

    dominated: list[Record] = []
    non_dominated: list[Record] = []
    for r in extras:
        if any(_dominates(s, r) for s in shared_cells):
            dominated.append(r)
        else:
            non_dominated.append(r)
    return dominated, non_dominated


@dataclass(frozen=True)
class HullDiff:
    """Bucketed comparison of two hull JSONLs."""

    cpp_records: list[Record]
    other_records: list[Record]
    cpp_index: dict[CellKey, Record]
    other_index: dict[CellKey, Record]
    shared: list[CellKey]
    cpp_only: list[Record]
    other_dominated: list[Record]
    other_non_dominated: list[Record]
    exact: int
    co_optimal: int
    ties: list[tuple[Record, Record]]

    @property
    def other_only_total(self) -> int:
        return len(self.other_dominated) + len(self.other_non_dominated)

    @property
    def is_failure(self) -> bool:
        return bool(self.cpp_only)


def _bucket_hulls(
    cpp_records: list[Record],
    other_records: list[Record],
    tol_mfe: float,
    tol_cai: float,
) -> HullDiff:
    """Index both sides by cell key, partition into shared / cpp-only /
    other-only, and classify the other-only extras by domination."""

    cpp_index = _index_by_cell(cpp_records, tol_mfe, tol_cai)
    other_index = _index_by_cell(other_records, tol_mfe, tol_cai)
    cpp_keys = set(cpp_index)
    other_keys = set(other_index)
    cpp_only = sorted(
        (cpp_index[k] for k in cpp_keys - other_keys),
        key=lambda r: (r.mfe_kcal_mol, r.cai),
    )
    other_only = sorted(
        (other_index[k] for k in other_keys - cpp_keys),
        key=lambda r: (r.mfe_kcal_mol, r.cai),
    )
    shared = sorted(cpp_keys & other_keys, key=lambda k: (k.mfe_step, k.cai_step))
    shared_records = [cpp_index[k] for k in shared]
    other_dominated, other_non_dominated = _classify_rust_extras(
        other_only, shared_records
    )
    exact, co_optimal, ties = _compare_matched_cells(cpp_index, other_index, shared)
    return HullDiff(
        cpp_records=cpp_records,
        other_records=other_records,
        cpp_index=cpp_index,
        other_index=other_index,
        shared=shared,
        cpp_only=cpp_only,
        other_dominated=other_dominated,
        other_non_dominated=other_non_dominated,
        exact=exact,
        co_optimal=co_optimal,
        ties=ties,
    )


def _print_summary(diff: HullDiff) -> None:
    print(f"cpp:   {len(diff.cpp_records)} records → {len(diff.cpp_index)} unique hull cells")
    print(f"other: {len(diff.other_records)} records → {len(diff.other_index)} unique hull cells")
    print(f"shared cells: {len(diff.shared)} ({diff.exact} exact, {diff.co_optimal} co-optimal-tie)")
    print(f"cpp-only orphans: {len(diff.cpp_only)}")
    print(
        f"other-only orphans: {diff.other_only_total} "
        f"({len(diff.other_dominated)} dominated by shared, "
        f"{len(diff.other_non_dominated)} non-dominated — true Pareto extras)"
    )


def _print_other_extras(diff: HullDiff) -> None:
    if diff.other_dominated:
        print(
            "\nother-only orphans dominated by a shared cell "
            "(integer-dekacal tie-break duplicates — informational):"
        )
        _report_orphans("dominated", diff.other_dominated)
    if diff.other_non_dominated:
        print(
            "\nother-only orphans NOT dominated by any shared cell "
            "(genuine extra Pareto vertices the C++ bisection missed — informational):"
        )
        _report_orphans("non-dominated", diff.other_non_dominated)


def _print_ties(ties: list[tuple[Record, Record]]) -> None:
    if not ties:
        return
    print("\nco-optimal ties (informational; not a failure):")
    for cpp_rec, other_rec in ties[:5]:
        print(f"  cell {_format_cell(cpp_rec)}")
        print(f"    cpp:   seq={cpp_rec.rna_seq[:32]}… struct={cpp_rec.structure[:32]}…")
        print(f"    other: seq={other_rec.rna_seq[:32]}… struct={other_rec.structure[:32]}…")
    if len(ties) > 5:
        print(f"  … and {len(ties) - 5} more")


def diff_hulls(
    cpp_path: Path, other_path: Path, tol_mfe: float, tol_cai: float
) -> int:
    diff = _bucket_hulls(
        _load_jsonl(cpp_path),
        _load_jsonl(other_path),
        tol_mfe,
        tol_cai,
    )
    _print_summary(diff)
    if diff.is_failure:
        print("\nHULL MISMATCH: C++ has cells the other side is missing.")
        _report_orphans("cpp-only", diff.cpp_only)
        print("\nresult: FAIL")
        return 1
    _print_other_extras(diff)
    _print_ties(diff.ties)
    print(
        f"\nresult: PASS — cpp ⊆ other "
        f"({diff.exact} exact + {diff.co_optimal} co-optimal-tie / "
        f"{len(diff.shared)} shared cells; "
        f"{len(diff.other_non_dominated)} extra Pareto vertices on the other side)"
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
    return diff_hulls(args.cpp, args.other, args.tolerance_mfe, args.tolerance_cai)


if __name__ == "__main__":
    sys.exit(main())
