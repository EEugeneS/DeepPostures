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


def discover_subject_age_groups(data_root: Path, source_split: str) -> dict[str, set[str]]:
    """Return subject -> all age groups represented in its wrist recordings."""
    raw_root = data_root / source_split / "80Hz_RAW"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_root}")

    subjects: dict[str, set[str]] = defaultdict(set)
    for raw_path in sorted(raw_root.rglob("*.csv.gz")):
        parsed = parse_recording_name(raw_path)
        if parsed is None:
            LOGGER.warning("Skipping unrecognised raw filename: %s", raw_path)
            continue
        subject, _environment, location = parsed
        if location != "W":
            continue
        age_group = raw_path.parent.relative_to(raw_root).as_posix()
        subjects[subject].add(age_group)
    if not subjects:
        raise RuntimeError(f"No wrist recordings found in {raw_root}")
    return dict(subjects)


def assign_validation_subjects(
    train_subjects: dict[str, set[str]], validation_fraction: float, seed: int
) -> set[str]:
    """Balance age groups while assigning every subject to one final split.

    A subject may contribute recordings to multiple age groups.  Independent
    within-age-group randomisation would then put that person in both train and
    validation, which leaks subject-specific signal.  This greedy assignment
    therefore chooses validation *subjects* while aiming for the requested
    validation fraction in every represented age group.
    """
    by_age_group: dict[str, set[str]] = defaultdict(set)
    for subject, age_groups in train_subjects.items():
        for age_group in age_groups:
            by_age_group[age_group].add(subject)

    total = {age_group: len(subjects) for age_group, subjects in by_age_group.items()}
    target: dict[str, int] = {}
    for age_group, count in total.items():
        if count < 2:
            target[age_group] = 0
            LOGGER.warning("Age group %s has one source-train subject; it remains in train.", age_group)
        else:
            target[age_group] = min(count - 1, max(1, round(count * validation_fraction)))
    rng = random.Random(seed)
    tie_breaker = {subject: rng.random() for subject in train_subjects}
    validation_subjects: set[str] = set()
    actual = {age_group: 0 for age_group in by_age_group}

    while True:
        needed = [age_group for age_group in by_age_group if actual[age_group] < target[age_group]]
        if not needed:
            break
        # Serve the currently most under-filled age group first.  A candidate
        # contributing to other under-filled groups is preferred.
        age_group = max(needed, key=lambda group: ((target[group] - actual[group]) / target[group], -total[group]))
        candidates = []
        for subject in by_age_group[age_group]:
            if subject in validation_subjects:
                continue
            subject_groups = train_subjects[subject]
            # Preserve at least one source-train subject in every age group.
            if any(actual[group] + 1 > total[group] - 1 for group in subject_groups):
                continue
            score = sum(
                max(target[group] - actual[group], 0) / max(target[group], 1)
                for group in subject_groups
            )
            candidates.append((score, tie_breaker[subject], subject))
        if not candidates:
            LOGGER.warning(
                "Could not reach validation target for age group %s without splitting a subject across sets.",
                age_group,
            )
            target[age_group] = actual[age_group]
            continue
        _, _, selected = max(candidates)
        validation_subjects.add(selected)
        for group in train_subjects[selected]:
            actual[group] += 1

    for age_group in sorted(by_age_group):
        LOGGER.info(
            "%s: %d source-train subjects -> %d train, %d validation (target %d)",
            age_group, total[age_group], total[age_group] - actual[age_group], actual[age_group], target[age_group],
        )
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
    train_subjects = discover_subject_age_groups(args.data_root, "train")
    test_subjects = discover_subject_age_groups(args.data_root, "test")
    overlap = set(train_subjects).intersection(test_subjects)
    if overlap:
        raise ValueError(f"Subjects occur in both source train and test: {sorted(overlap)}")

    validation_subjects = assign_validation_subjects(train_subjects, args.validation_fraction, args.seed)
    rows = []
    for subject, age_groups in train_subjects.items():
        for age_group in sorted(age_groups):
            rows.append({
                "subject": subject,
                "age_group": age_group,
                "split": "validation" if subject in validation_subjects else "train",
                "source_split": "train",
                "seed": args.seed,
            })
    for subject, age_groups in test_subjects.items():
        for age_group in sorted(age_groups):
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

    counts = {
        "train": len(train_subjects) - len(validation_subjects),
        "validation": len(validation_subjects),
        "test": len(test_subjects),
    }
    LOGGER.info("Wrote %s (train=%d, validation=%d, test=%d)", args.output_csv,
                counts["train"], counts["validation"], counts["test"])


if __name__ == "__main__":
    main()
