#!/usr/bin/env python3
"""Create the reproducible age-stratified MoCA wrist subject split.

The cleaned dataset stores every recording under all/ and preserves the
study-provided train/test assignment in one randomization CSV per age group.
This script keeps every source-test subject in test and selects validation
subjects only from source train, independently within each age group.

Expected layout:
  <data-root>/all/80Hz_RAW/<age-group>/<subject>_<env>_W.csv.gz
  <data-root>/support_files/randomization/<age-group>.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger(__name__)
FILENAME_RE = re.compile(
    r"^(?P<subject>[^_]+)_(?P<environment>E(?:1|2)?|[CHS])_"
    r"(?P<location>[HW])\.csv(?:\.gz)?$",
    re.IGNORECASE,
)


def parse_recording_name(path: Path) -> tuple[str, str, str] | None:
    """Return subject, environment, and location encoded in a filename."""
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    return (
        match.group("subject"),
        match.group("environment").upper(),
        match.group("location").upper(),
    )


def discover_wrist_subjects(data_root: Path) -> dict[str, str]:
    """Return subject -> age group from the cleaned all/80Hz_RAW tree."""
    raw_root = data_root / "all" / "80Hz_RAW"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_root}")

    subject_age: dict[str, str] = {}
    recognised_recordings = 0
    for raw_path in sorted(raw_root.rglob("*.csv.gz")):
        parsed = parse_recording_name(raw_path)
        if parsed is None:
            LOGGER.warning("Skipping unrecognised raw filename: %s", raw_path)
            continue
        subject, _environment, location = parsed
        if location != "W":
            continue
        recognised_recordings += 1
        age_group = raw_path.parent.relative_to(raw_root).as_posix()
        previous = subject_age.setdefault(subject, age_group)
        if previous != age_group:
            raise ValueError(
                f"Subject {subject} occurs in multiple age groups: "
                f"{previous!r} and {age_group!r}"
            )

    if not subject_age:
        raise RuntimeError(f"No wrist recordings found in {raw_root}")
    LOGGER.info(
        "Discovered %d wrist recordings from %d unique subjects.",
        recognised_recordings,
        len(subject_age),
    )
    return subject_age


def read_randomization(randomization_root: Path) -> dict[str, tuple[str, str]]:
    """Return subject -> (age group, source split) from per-age CSV files."""
    if not randomization_root.is_dir():
        raise FileNotFoundError(
            f"Missing randomization directory: {randomization_root}"
        )

    assignments: dict[str, tuple[str, str]] = {}
    csv_paths = sorted(randomization_root.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No randomization CSV files found in {randomization_root}"
        )

    for csv_path in csv_paths:
        age_group = csv_path.stem
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = {str(name).strip().lower(): name for name in (reader.fieldnames or [])}
            if "subject" not in fields or "random" not in fields:
                raise ValueError(
                    f"{csv_path} must contain subject and random columns."
                )
            for row_number, row in enumerate(reader, start=2):
                subject = str(row[fields["subject"]] or "").strip().strip('"')
                source_split = str(row[fields["random"]] or "").strip().lower()
                if not subject:
                    raise ValueError(
                        f"Missing subject in {csv_path} row {row_number}."
                    )
                if source_split not in {"train", "test"}:
                    raise ValueError(
                        f"Unsupported randomization value {source_split!r} "
                        f"in {csv_path} row {row_number}."
                    )
                if subject in assignments:
                    previous_age, previous_split = assignments[subject]
                    raise ValueError(
                        f"Subject {subject} occurs more than once in the "
                        f"randomization files: {previous_age}/{previous_split} "
                        f"and {age_group}/{source_split}."
                    )
                assignments[subject] = (age_group, source_split)
    return assignments


def assign_validation_subjects(
    subjects_by_age: dict[str, list[str]], validation_fraction: float, seed: int
) -> set[str]:
    """Select source-train validation subjects independently by age group."""
    validation_subjects: set[str] = set()
    for age_index, age_group in enumerate(sorted(subjects_by_age)):
        subjects = sorted(subjects_by_age[age_group])
        if len(subjects) < 2:
            LOGGER.warning(
                "Age group %s has one source-train subject; it remains in train.",
                age_group,
            )
            continue
        target = min(
            len(subjects) - 1,
            max(1, round(len(subjects) * validation_fraction)),
        )
        rng = random.Random(f"{seed}:{age_index}:{age_group}")
        rng.shuffle(subjects)
        selected = subjects[:target]
        validation_subjects.update(selected)
        LOGGER.info(
            "%s: %d source-train subjects -> %d train, %d validation",
            age_group,
            len(subjects),
            len(subjects) - target,
            target,
        )
    return validation_subjects


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing all/ and support_files/.",
    )
    parser.add_argument(
        "--randomization-root",
        type=Path,
        default=None,
        help=(
            "Directory containing per-age randomization CSVs; "
            "default: <data-root>/support_files/randomization."
        ),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.25,
        help="Fraction of source-train subjects assigned to validation in each age group.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Fixed RNG seed recorded in the output CSV."
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be strictly between 0 and 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    randomization_root = (
        args.randomization_root
        if args.randomization_root is not None
        else args.data_root / "support_files" / "randomization"
    )
    subject_age = discover_wrist_subjects(args.data_root)
    randomization = read_randomization(randomization_root)

    raw_subjects = set(subject_age)
    randomized_subjects = set(randomization)
    missing_randomization = raw_subjects - randomized_subjects
    if missing_randomization:
        raise ValueError(
            "Wrist subjects missing from randomization files: "
            f"{sorted(missing_randomization)}"
        )

    age_mismatches = {
        subject: (subject_age[subject], randomization[subject][0])
        for subject in raw_subjects
        if subject_age[subject] != randomization[subject][0]
    }
    if age_mismatches:
        raise ValueError(
            "Raw-folder and randomization age groups disagree: "
            f"{age_mismatches}"
        )

    randomization_without_wrist = randomized_subjects - raw_subjects
    if randomization_without_wrist:
        LOGGER.warning(
            "%d randomized subjects have no wrist recording and will be excluded: %s",
            len(randomization_without_wrist),
            sorted(randomization_without_wrist),
        )

    source_train_by_age: dict[str, list[str]] = {}
    for subject in sorted(raw_subjects):
        age_group, source_split = randomization[subject]
        if source_split == "train":
            source_train_by_age.setdefault(age_group, []).append(subject)

    validation_subjects = assign_validation_subjects(
        source_train_by_age, args.validation_fraction, args.seed
    )

    rows: list[dict[str, str | int]] = []
    for subject in sorted(raw_subjects):
        age_group, source_split = randomization[subject]
        if source_split == "test":
            final_split = "test"
        elif subject in validation_subjects:
            final_split = "validation"
        else:
            final_split = "train"
        rows.append(
            {
                "subject": subject,
                "age_group": age_group,
                "split": final_split,
                "source_split": source_split,
                "seed": args.seed,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "age_group", "split", "source_split", "seed"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["split"]) for row in rows)
    LOGGER.info(
        "Wrote %s (train=%d, validation=%d, test=%d)",
        args.output_csv,
        counts["train"],
        counts["validation"],
        counts["test"],
    )


if __name__ == "__main__":
    main()
