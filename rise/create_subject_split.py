#!/usr/bin/env python3
"""Create a subject-level train/validation/test manifest for Rise.

The supplied P2 randomization file already defines train_set and test_set.
This script preserves every test_set subject and deterministically divides
only train_set subjects into train and validation subsets.
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be strictly between 0 and 1.")

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

    validation_count = round(len(source_train_ids) * args.validation_fraction)
    validation_count = max(1, validation_count)
    if validation_count >= len(source_train_ids):
        raise ValueError("Validation split would leave no subjects in the training split.")

    shuffled_train_ids = source_train_ids.copy()
    random.Random(args.seed).shuffle(shuffled_train_ids)
    validation_ids = set(shuffled_train_ids[:validation_count])

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
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
