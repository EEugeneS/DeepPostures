#!/usr/bin/env python3
"""Create a subject-level train/validation/test manifest for Rise.

The supplied P2 randomization file already defines train_set and test_set.
This script preserves every test_set subject and deterministically divides
only train_set subjects into train and validation subsets. When BL and FV raw
directories are supplied, it stratifies this division by visit availability:
BL-only, FV-only, and BL+FV.
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def _normalise(value: str) -> str:
    return value.strip().strip('"').lower()


def _find_column(
    fieldnames: List[str], requested: Optional[str], kind: str
) -> str:
    if not fieldnames:
        raise ValueError("The source CSV has no header row.")

    if requested is not None:
        for fieldname in fieldnames:
            if _normalise(fieldname) == _normalise(requested):
                return fieldname
        raise ValueError(
            f"Could not find {kind} column {requested!r}. Available columns: {fieldnames}"
        )

    if kind == "ID":
        for fieldname in fieldnames:
            if _normalise(fieldname) in {"id", "subject_id", "subject"}:
                return fieldname
        return fieldnames[0]

    if len(fieldnames) < 2:
        raise ValueError("The source CSV needs an ID column and a source-split column.")
    return fieldnames[1]


def _visit_subject_ids(raw_dir: Path, visit: str) -> set:
    """Extract numeric subject IDs from names such as 123456BL.csv.gz."""
    suffixes = (f"{visit}.csv.gz", f"{visit}.csv")
    subject_ids = set()
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        suffix = next((item for item in suffixes if path.name.endswith(item)), None)
        if suffix is None:
            continue
        subject_id = path.name[:-len(suffix)]
        if not subject_id:
            raise ValueError(f"Could not extract an ID from {path.name!r}.")
        subject_ids.add(subject_id)
    if not subject_ids:
        raise ValueError(
            f"No files matching <numeric-ID>{visit}.csv[.gz] were found in {raw_dir}."
        )
    return subject_ids


def _visit_strata(
    source_train_ids: List[str],
    bl_ids: set,
    fv_ids: set,
) -> Dict[str, List[str]]:
    strata: Dict[str, List[str]] = {"BL-only": [], "FV-only": [], "BL+FV": [], "neither": []}
    for subject_id in source_train_ids:
        has_bl = subject_id in bl_ids
        has_fv = subject_id in fv_ids
        if has_bl and has_fv:
            strata["BL+FV"].append(subject_id)
        elif has_bl:
            strata["BL-only"].append(subject_id)
        elif has_fv:
            strata["FV-only"].append(subject_id)
        else:
            strata["neither"].append(subject_id)
    return strata


def _validation_ids_by_stratum(
    strata: Dict[str, List[str]], validation_fraction: float, seed: int
) -> set:
    """Select validation subjects independently within every availability stratum."""
    validation_ids = set()
    rng = random.Random(seed)
    for stratum in sorted(strata):
        subject_ids = sorted(strata[stratum])
        if len(subject_ids) < 2:
            # A one-subject stratum cannot be represented in both train and validation.
            continue
        validation_count = round(len(subject_ids) * validation_fraction)
        validation_count = min(max(1, validation_count), len(subject_ids) - 1)
        rng.shuffle(subject_ids)
        validation_ids.update(subject_ids[:validation_count])
    return validation_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve Rise P2 test subjects and split only its train subjects "
            "into subject-level train/validation sets."
        )
    )
    parser.add_argument("--source-csv", required=True, help="P2_train_test_rand.csv")
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output manifest with CHAP2-compatible subject_id,split columns.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.25,
        help="Fraction of source train_set subjects assigned to validation (default: 0.25).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional source ID column name; defaults to ID/subject_id/first column.",
    )
    parser.add_argument(
        "--source-split-column",
        default=None,
        help="Optional source split column name; defaults to the second column.",
    )
    parser.add_argument(
        "--source-train-value",
        default="train_set",
        help="Value denoting an original training subject (default: train_set).",
    )
    parser.add_argument(
        "--source-test-value",
        default="test_set",
        help="Value denoting an original test subject (default: test_set).",
    )
    parser.add_argument(
        "--bl-raw-dir",
        default=None,
        help="AG/BL directory. Together with --fv-raw-dir, enables BL/FV-stratified validation splitting.",
    )
    parser.add_argument(
        "--fv-raw-dir",
        default=None,
        help="AG/FV directory. Together with --bl-raw-dir, enables BL/FV-stratified validation splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be strictly between 0 and 1.")
    if (args.bl_raw_dir is None) != (args.fv_raw_dir is None):
        raise ValueError("Provide both --bl-raw-dir and --fv-raw-dir, or neither.")

    source_path = Path(args.source_csv)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source_path}")

    source_train_value = _normalise(args.source_train_value)
    source_test_value = _normalise(args.source_test_value)
    if source_train_value == source_test_value:
        raise ValueError("Source train and test values must be different.")

    assignments: Dict[str, str] = {}
    with source_path.open(newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        id_column = _find_column(reader.fieldnames or [], args.id_column, "ID")
        split_column = _find_column(
            reader.fieldnames or [], args.source_split_column, "source split"
        )

        for row_number, row in enumerate(reader, start=2):
            subject_id = (row.get(id_column) or "").strip().strip('"')
            source_split = _normalise(row.get(split_column) or "")
            if not subject_id:
                raise ValueError(f"Missing subject ID on row {row_number}.")
            if source_split not in {source_train_value, source_test_value}:
                raise ValueError(
                    f"Unexpected source split {source_split!r} on row {row_number}; "
                    f"expected {source_train_value!r} or {source_test_value!r}."
                )
            if subject_id in assignments:
                raise ValueError(
                    f"Duplicate subject ID {subject_id!r} on row {row_number}. "
                    "Resolve duplicate IDs before splitting."
                )
            assignments[subject_id] = source_split

    source_train_ids = sorted(
        subject_id
        for subject_id, source_split in assignments.items()
        if source_split == source_train_value
    )
    source_test_ids = sorted(
        subject_id
        for subject_id, source_split in assignments.items()
        if source_split == source_test_value
    )
    if len(source_train_ids) < 2:
        raise ValueError("At least two source training subjects are required.")

    bl_ids, fv_ids = set(), set()
    if args.bl_raw_dir is not None:
        bl_dir = Path(args.bl_raw_dir)
        fv_dir = Path(args.fv_raw_dir)
        if not bl_dir.is_dir() or not fv_dir.is_dir():
            raise NotADirectoryError("--bl-raw-dir and --fv-raw-dir must both be directories.")
        bl_ids = _visit_subject_ids(bl_dir, "BL")
        fv_ids = _visit_subject_ids(fv_dir, "FV")
        raw_ids_without_p2_assignment = (bl_ids | fv_ids) - set(assignments)
        if raw_ids_without_p2_assignment:
            raise ValueError(
                "Raw IDs missing from the P2 randomization: "
                f"{sorted(raw_ids_without_p2_assignment)}"
            )
        strata = _visit_strata(source_train_ids, bl_ids, fv_ids)
        validation_ids = _validation_ids_by_stratum(
            strata, args.validation_fraction, args.seed
        )
    else:
        strata = {"unstratified": source_train_ids}
        validation_ids = _validation_ids_by_stratum(
            strata, args.validation_fraction, args.seed
        )

    output_rows = []
    for subject_id in sorted(assignments):
        if assignments[subject_id] == source_test_value:
            split = "test"
        elif subject_id in validation_ids:
            split = "validation"
        else:
            split = "train"
        output_rows.append({"subject_id": subject_id, "split": split})

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["subject_id", "split"])
        writer.writeheader()
        writer.writerows(output_rows)

    final_counts = Counter(row["split"] for row in output_rows)
    total = len(output_rows)
    print(f"Read {total} unique subjects from {source_path}")
    print(
        "Source split: "
        f"{len(source_train_ids)} train_set, {len(source_test_ids)} test_set"
    )
    for split in ("train", "validation", "test"):
        count = final_counts[split]
        print(f"{split}: {count} ({100 * count / total:.1f}%)")
    if args.bl_raw_dir is not None:
        print("Source-train visit-availability strata (train / validation):")
        for stratum in ("BL-only", "FV-only", "BL+FV", "neither"):
            members = set(strata[stratum])
            validation_count = len(members & validation_ids)
            print(f"  {stratum}: {len(members) - validation_count} / {validation_count}")

        final_split_by_id = {row["subject_id"]: row["split"] for row in output_rows}
        print("Visit records by final split (BL / FV):")
        for split in ("train", "validation", "test"):
            print(
                f"  {split}: "
                f"{sum(final_split_by_id[item] == split for item in bl_ids)} / "
                f"{sum(final_split_by_id[item] == split for item in fv_ids)}"
            )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
