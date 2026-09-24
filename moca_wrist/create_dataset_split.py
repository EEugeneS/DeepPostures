#!/usr/bin/env python3
"""Create CHAP-ready train/validation/test H5 files from MoCA wrist segments.

This is the MoCA/wrist counterpart of CHAP2/create_dataset_split.py.  It keeps
the original CHAP output schema (x, y, timestamp, subject_id, std), while
reading each recording H5 independently.  Consequently, a 42 x 10-second
model window never crosses a C/E/H/S recording-file boundary.

Example:
  python create_dataset_split.py \\
    --data-dir /data/MOCA/processed_wrist \\
    --split-csv splits/moca_wrist_subject_split.csv \\
    --output-dir /data/MOCA/chap_ready_wrist
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def _as_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def iter_valid_sequences(segment_path: Path) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield contiguous valid 10-second sequences from one segment H5.

    Invalid labels, sleeping, or non-wear rows end a sequence.  This applies
    to every final split so `label=-1` never reaches validation or test.
    """
    with h5py.File(segment_path, "r") as h5_file:
        required = {"data", "time", "label", "sleeping", "non_wear"}
        missing = required.difference(h5_file.keys())
        if missing:
            raise ValueError(f"{segment_path} is missing datasets: {sorted(missing)}")
        data = h5_file["data"][:]
        timestamps = h5_file["time"][:]
        labels = h5_file["label"][:]
        sleeping = h5_file["sleeping"][:]
        non_wear = h5_file["non_wear"][:]

    if not (len(data) == len(timestamps) == len(labels) == len(sleeping) == len(non_wear)):
        raise ValueError(f"Datasets have inconsistent lengths in {segment_path}")
    if data.ndim != 3 or data.shape[1:] != (100, 3):
        raise ValueError(f"Expected data shape (N, 100, 3) in {segment_path}; got {data.shape}")

    x_buffer: list[np.ndarray] = []
    time_buffer: list[float] = []
    y_buffer: list[int] = []
    previous_timestamp: float | None = None
    for x, timestamp, label, is_sleeping, is_non_wear in zip(data, timestamps, labels, sleeping, non_wear):
        if label < 0 or is_sleeping == 1 or is_non_wear == 1:
            if x_buffer:
                yield np.stack(x_buffer), np.asarray(time_buffer), np.asarray(y_buffer)
            x_buffer.clear()
            time_buffer.clear()
            y_buffer.clear()
            previous_timestamp = None
            continue
        if previous_timestamp is not None and not np.isclose(
            float(timestamp) - previous_timestamp, 10.0, rtol=0.0, atol=1e-3
        ):
            if x_buffer:
                yield np.stack(x_buffer), np.asarray(time_buffer), np.asarray(y_buffer)
            x_buffer.clear()
            time_buffer.clear()
            y_buffer.clear()
        x_buffer.append(x)
        time_buffer.append(timestamp)
        y_buffer.append(label)
        previous_timestamp = float(timestamp)
    if x_buffer:
        yield np.stack(x_buffer), np.asarray(time_buffer), np.asarray(y_buffer)


def iter_segment_windows(
    subject_dir: Path, subject: str, window_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]]:
    """Yield non-overlapping CHAP windows, resetting at every segment H5."""
    segment_paths = sorted(path for path in subject_dir.glob("*.h5") if path.is_file())
    if not segment_paths:
        LOGGER.warning("No segment H5 files found for subject %s", subject)
        return

    for segment_path in segment_paths:
        with h5py.File(segment_path, "r") as h5_file:
            environment = _as_text(h5_file.attrs.get("environment", "unknown"))
            age_group = _as_text(h5_file.attrs.get("age_group", "unknown"))
        for x_sequence, timestamps, y_sequence in iter_valid_sequences(segment_path):
            complete_windows = len(y_sequence) // window_size
            for window_index in range(complete_windows):
                start = window_index * window_size
                end = start + window_size
                yield (
                    x_sequence[start:end],
                    timestamps[start:end],
                    y_sequence[start:end],
                    segment_path.stem,
                    environment,
                    age_group,
                )


class OutputWriter:
    """Buffered writer for an original-CHAP-compatible dataset H5."""

    def __init__(self, output_path: Path, window_size: int, flush_threshold: int):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(output_path, "w")
        self.window_size = window_size
        self.flush_threshold = flush_threshold
        text_dtype = h5py.string_dtype(encoding="utf-8")
        self.datasets = {
            "x": self.file.create_dataset(
                "x", shape=(0, window_size, 100, 3), maxshape=(None, window_size, 100, 3),
                dtype="f4", chunks=(1, window_size, 100, 3), compression="gzip",
            ),
            "y": self.file.create_dataset(
                "y", shape=(0, window_size), maxshape=(None, window_size), dtype="i4",
                chunks=(1, window_size), compression="gzip",
            ),
            "timestamp": self.file.create_dataset(
                "timestamp", shape=(0, window_size), maxshape=(None, window_size), dtype="f8",
                chunks=(1, window_size), compression="gzip",
            ),
            "subject_id": self.file.create_dataset(
                "subject_id", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=(1,), compression="gzip",
            ),
            "std": self.file.create_dataset(
                "std", shape=(0, window_size), maxshape=(None, window_size), dtype="f4",
                chunks=(1, window_size), compression="gzip",
            ),
            # Extra metadata is harmless to existing CHAP readers and enables environment analysis.
            "segment_id": self.file.create_dataset(
                "segment_id", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=(1,), compression="gzip",
            ),
            "environment": self.file.create_dataset(
                "environment", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=(1,), compression="gzip",
            ),
            "age_group": self.file.create_dataset(
                "age_group", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=(1,), compression="gzip",
            ),
        }
        self.x_buffer: list[np.ndarray] = []
        self.y_buffer: list[np.ndarray] = []
        self.time_buffer: list[np.ndarray] = []
        self.subject_buffer: list[str] = []
        self.segment_buffer: list[str] = []
        self.environment_buffer: list[str] = []
        self.age_group_buffer: list[str] = []

    def append(
        self, x: np.ndarray, y: np.ndarray, timestamp: np.ndarray,
        subject: str, segment_id: str, environment: str, age_group: str,
    ) -> None:
        self.x_buffer.append(x.astype(np.float32, copy=False))
        self.y_buffer.append(y.astype(np.int32, copy=False))
        self.time_buffer.append(timestamp.astype(np.float64, copy=False))
        self.subject_buffer.append(subject)
        self.segment_buffer.append(segment_id)
        self.environment_buffer.append(environment)
        self.age_group_buffer.append(age_group)
        if len(self.x_buffer) >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self.x_buffer:
            return
        count = len(self.x_buffer)
        old_size = self.datasets["x"].shape[0]
        new_size = old_size + count
        x = np.stack(self.x_buffer)
        payloads = {
            "x": x,
            "y": np.stack(self.y_buffer),
            "timestamp": np.stack(self.time_buffer),
            "subject_id": np.asarray(self.subject_buffer, dtype=h5py.string_dtype(encoding="utf-8")),
            "std": np.mean(np.std(x, axis=2), axis=2).astype(np.float32),
            "segment_id": np.asarray(self.segment_buffer, dtype=h5py.string_dtype(encoding="utf-8")),
            "environment": np.asarray(self.environment_buffer, dtype=h5py.string_dtype(encoding="utf-8")),
            "age_group": np.asarray(self.age_group_buffer, dtype=h5py.string_dtype(encoding="utf-8")),
        }
        for name, values in payloads.items():
            dataset = self.datasets[name]
            dataset.resize((new_size,) + dataset.shape[1:])
            dataset[old_size:new_size] = values
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.time_buffer.clear()
        self.subject_buffer.clear()
        self.segment_buffer.clear()
        self.environment_buffer.clear()
        self.age_group_buffer.clear()

    def close(self) -> int:
        self.flush()
        count = int(self.datasets["x"].shape[0])
        self.file.close()
        return count


def write_split(
    data_dir: Path, output_path: Path, subjects: list[str], window_size: int, flush_threshold: int
) -> int:
    """Write all valid segment windows for the supplied subject list."""
    writer = OutputWriter(output_path, window_size, flush_threshold)
    try:
        for subject in tqdm(subjects, desc=output_path.stem):
            subject_dir = data_dir / subject
            if not subject_dir.is_dir():
                LOGGER.warning("Subject directory not found: %s", subject_dir)
                continue
            for x, timestamps, y, segment_id, environment, age_group in iter_segment_windows(subject_dir, subject, window_size):
                writer.append(x, y, timestamps, subject, segment_id, environment, age_group)
        return writer.close()
    except Exception:
        writer.close()
        raise


def read_split_csv(split_csv: Path, age_group: str | None = None) -> dict[str, list[str]]:
    frame = pd.read_csv(split_csv, dtype=str)
    subject_column = "subject" if "subject" in frame.columns else "subject_id"
    required = {subject_column, "split"}
    if age_group is not None:
        required.add("age_group")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split_csv} is missing columns: {sorted(missing)}")
    if age_group is not None:
        available = set(frame["age_group"].dropna().astype(str))
        if age_group not in available:
            raise ValueError(
                f"Age group {age_group!r} is absent from {split_csv}; "
                f"available groups: {sorted(available)}"
            )
        frame = frame.loc[frame["age_group"].astype(str) == age_group]
    splits: dict[str, list[str]] = {}
    assignment_by_subject: dict[str, str] = {}
    for row in frame.loc[:, [subject_column, "split"]].dropna().itertuples(index=False):
        subject, split_name = str(row[0]), str(row[1])
        if split_name not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split {split_name!r} for subject {subject} in {split_csv}")
        previous = assignment_by_subject.setdefault(subject, split_name)
        if previous != split_name:
            raise ValueError(f"Subject {subject} occurs in both {previous} and {split_name} in {split_csv}")
    for split_name in ("train", "validation", "test"):
        splits[split_name] = sorted(
            subject for subject, assignment in assignment_by_subject.items() if assignment == split_name
        )
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory produced by pre_process_data.py.")
    parser.add_argument("--split-csv", type=Path, required=True, help="CSV produced by create_subject_split.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--age-group",
        help=("Build only one age group from the shared subject split. "
              "Omit to build the pooled full dataset."),
    )
    parser.add_argument("--window-size", type=int, default=42, help="Number of ten-second samples per CHAP input.")
    parser.add_argument("--flush-threshold", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing output H5 files.")
    args = parser.parse_args()
    if args.window_size <= 0 or args.flush_threshold <= 0:
        raise ValueError("--window-size and --flush-threshold must be positive")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Missing preprocessed data directory: {args.data_dir}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    splits = read_split_csv(args.split_csv, args.age_group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": args.output_dir / "10s_train.h5",
        "validation": args.output_dir / "10s_val.h5",
        "test": args.output_dir / "10s_test_complete.h5",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}. Use --overwrite only to regenerate it.")

    for split_name, output_path in outputs.items():
        count = write_split(
            args.data_dir, output_path, splits[split_name], args.window_size, args.flush_threshold
        )
        LOGGER.info("%s: %d subjects -> %d windows (%s)", split_name, len(splits[split_name]), count, output_path)


if __name__ == "__main__":
    main()
