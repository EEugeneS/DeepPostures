#!/usr/bin/env python3
"""Create a reproducible, age-stratified subject split for MoCA wrist.

The dataset's top-level train/test assignment is preserved. Only source-train
subjects are randomly divided into final train and validation sets. Every
wrist recording belonging to a subject receives the same final split.

Example:
  python create_subject_split.py \\
    --data-root /data/MOCA \\
    --output-csv splits/moca_wrist_subject_split.csv \\
    --validation-fraction 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

LOGGER = logging.getLogger(__name__)
FILENAME_RE = re.compile(
    r"^(?P<subject>[^_]+)_(?P<environment>[CEHS])_(?P<location>[HW])\.csv(?:\.gz)?$",
    re.IGNORECASE,
)


def parse_recording_name(path: Path) -> tuple[str, str, str] | None:
    """Return subject, environment and location encoded in an input filename."""
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    return (
        match.group("subject"),
        match.group("environment").upper(),
        match.group("location").upper(),
    )


def discover_subjects(data_root: Path, source_split: str) -> dict[str, str]:
    """Return subject -> age group for wrist recordings in one source split."""
    raw_root = data_root / source_split / "80Hz_RAW"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_root}")

    subjects: dict[str, str] = {}
    for raw_path in sorted(raw_root.rglob("*.csv.gz")):
        parsed = parse_recording_name(raw_path)
        if parsed is None:
            LOGGER.warning("Skipping unrecognised raw filename: %s", raw_path)
            continue
        subject, _environment, location = parsed
        if location != "W":
            continue
        age_group = raw_path.parent.relative_to(raw_root).as_posix()
        previous_age_group = subjects.setdefault(subject, age_group)
        if previous_age_group != age_group:
            raise ValueError(
                f"Subject {subject} occurs in two age groups: {previous_age_group}, {age_group}"
            )
    if not subjects:
        raise RuntimeError(f"No wrist recordings found in {raw_root}")
    return subjects


def assign_validation_subjects(
    train_subjects: dict[str, str], validation_fraction: float, seed: int
) -> set[str]:
    """Choose validation subjects independently in each age group."""
    by_age_group: dict[str, list[str]] = defaultdict(list)
    for subject, age_group in train_subjects.items():
        by_age_group[age_group].append(subject)

    rng = random.Random(seed)
    validation_subjects: set[str] = set()
    for age_group, subjects in sorted(by_age_group.items()):
        subjects = sorted(subjects)
        if len(subjects) < 2:
            n_validation = 0
            LOGGER.warning("Age group %s has one source-train subject; it remains in train.", age_group)
        else:
            n_validation = round(len(subjects) * validation_fraction)
            n_validation = min(len(subjects) - 1, max(1, n_validation))
        selected = rng.sample(subjects, k=n_validation) if n_validation else []
        validation_subjects.update(selected)
        LOGGER.info("%s: %d train, %d validation subjects", age_group, len(subjects) - n_validation, n_validation)
    return validation_subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing train/ and test/.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.25,
                        help="Per-age-group fraction of source-train subjects assigned to validation.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed RNG seed recorded in the output CSV.")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be strictly between 0 and 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_subjects = discover_subjects(args.data_root, "train")
    test_subjects = discover_subjects(args.data_root, "test")
    overlap = set(train_subjects).intersection(test_subjects)
    if overlap:
        raise ValueError(f"Subjects occur in both source train and test: {sorted(overlap)}")

    validation_subjects = assign_validation_subjects(train_subjects, args.validation_fraction, args.seed)
    rows = []
    for subject, age_group in train_subjects.items():
        rows.append({
            "subject": subject,
            "age_group": age_group,
            "split": "validation" if subject in validation_subjects else "train",
            "source_split": "train",
            "seed": args.seed,
        })
    for subject, age_group in test_subjects.items():
        rows.append({
            "subject": subject,
            "age_group": age_group,
            "split": "test",
            "source_split": "test",
            "seed": args.seed,
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "age_group", "split", "source_split", "seed"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["split"], row["age_group"], row["subject"])))

    counts = defaultdict(int)
    for row in rows:
        counts[row["split"]] += 1
    LOGGER.info("Wrote %s (train=%d, validation=%d, test=%d)", args.output_csv,
                counts["train"], counts["validation"], counts["test"])


if __name__ == "__main__":
    main()
