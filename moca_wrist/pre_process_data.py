#!/usr/bin/env python3
"""Preprocess MoCA wrist ActiGraph recordings for CHAP.

One same-name raw/Event pair is one recording segment. The script never joins
files from different C/E/H/S environments and processes only wrist (_W) files.

Expected layout:
  <data-root>/train/80Hz_RAW/<age-group>/<subject>_<env>_W.csv.gz
  <data-root>/train/Event_file/<age-group>/<subject>_<env>_W.csv
  <data-root>/test/80Hz_RAW/<age-group>/...
  <data-root>/test/Event_file/<age-group>/...

Unknown-event files live below --unknown-root and have columns ID, unknown_DT,
int.sec, wearDate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
EXCEL_ORIGIN = "1899-12-30"
FILENAME_RE = re.compile(
    r"^(?P<subject>[^_]+)_(?P<environment>[CEHS])_(?P<location>[HW])\.csv(?:\.gz)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Recording:
    source_split: str
    age_group: str
    subject: str
    environment: str
    raw_path: Path
    event_path: Path

    @property
    def basename(self) -> str:
        return self.raw_path.name.removesuffix(".gz").removesuffix(".csv")


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


def find_recordings(data_root: Path) -> list[Recording]:
    """Find and validate every wrist raw/Event pair in train and test."""
    recordings: list[Recording] = []
    for source_split in ("train", "test"):
        raw_root = data_root / source_split / "80Hz_RAW"
        event_root = data_root / source_split / "Event_file"
        if not raw_root.is_dir():
            raise FileNotFoundError(f"Missing raw directory: {raw_root}")
        if not event_root.is_dir():
            raise FileNotFoundError(f"Missing Event directory: {event_root}")

        for raw_path in sorted(raw_root.rglob("*.csv.gz")):
            parsed = parse_recording_name(raw_path)
            if parsed is None:
                LOGGER.warning("Skipping unrecognised raw filename: %s", raw_path)
                continue
            subject, environment, location = parsed
            if location != "W":
                continue

            age_group = raw_path.parent.relative_to(raw_root).as_posix()
            event_path = event_root / age_group / raw_path.name.removesuffix(".gz")
            if not event_path.is_file():
                raise FileNotFoundError(
                    "No same-name Event file for raw recording: "
                    f"{raw_path} (expected {event_path})"
                )
            recordings.append(
                Recording(source_split, age_group, subject, environment, raw_path, event_path)
            )

    if not recordings:
        raise RuntimeError("No wrist (*.csv.gz ending in _W.csv.gz) recordings were found.")
    return recordings


def parse_unknown_events(unknown_root: Path | None) -> dict[str, list[tuple[datetime, datetime]]]:
    """Read all MOCA_unknownEvent.csv files below unknown_root."""
    unknown_by_subject: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    if unknown_root is None:
        LOGGER.warning("No --unknown-root supplied; no unknown intervals will be excluded.")
        return unknown_by_subject
    if not unknown_root.is_dir():
        raise FileNotFoundError(f"Unknown-event root does not exist: {unknown_root}")

    files = sorted(unknown_root.rglob("MOCA_unknownEvent.csv"))
    if not files:
        raise FileNotFoundError(f"No MOCA_unknownEvent.csv found below {unknown_root}")

    required = {"ID", "unknown_DT", "int.sec"}
    for csv_path in files:
        frame = pd.read_csv(csv_path, dtype={"ID": str})
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        for row in frame.loc[:, ["ID", "unknown_DT", "int.sec"]].dropna().itertuples(index=False):
            subject = str(row[0]).strip()
            start = pd.to_datetime(row[1], format="%Y-%m-%d %H:%M:%S", errors="raise").to_pydatetime()
            duration = float(row[2])
            if duration < 0:
                raise ValueError(f"Negative unknown duration in {csv_path}: {row}")
            unknown_by_subject[subject].append((start, start + timedelta(seconds=duration)))

    for intervals in unknown_by_subject.values():
        intervals.sort(key=lambda item: item[0])
    LOGGER.info("Read unknown intervals for %d subjects from %d file(s).", len(unknown_by_subject), len(files))
    return unknown_by_subject


def read_event_labels(event_path: Path) -> tuple[list[datetime], list[datetime], list[int]]:
    """Read ActivPAL Events and map 0 -> sitting, 1/2 -> non-sitting."""
    frame = pd.read_csv(event_path)
    activity_column = next((column for column in frame.columns if column.startswith("ActivityCode")), None)
    if activity_column is None:
        raise ValueError(f"No ActivityCode column in {event_path}")
    required = {"Time", "Interval (s)", activity_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{event_path} is missing columns: {sorted(missing)}")

    starts = pd.to_datetime(frame["Time"].astype(float), unit="D", origin=EXCEL_ORIGIN)
    durations = pd.to_timedelta(frame["Interval (s)"].astype(float), unit="s")
    labels: list[int] = []
    for value in frame[activity_column]:
        try:
            code = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ActivityCode {value!r} in {event_path}") from exc
        if code == 0:
            labels.append(0)
        elif code in (1, 2):
            labels.append(1)
        else:
            raise ValueError(f"Unsupported ActivityCode {code} in {event_path}")

    records = sorted(
        zip(starts.dt.to_pydatetime(), (starts + durations).dt.to_pydatetime(), labels),
        key=lambda item: item[0],
    )
    return [item[0] for item in records], [item[1] for item in records], [item[2] for item in records]


def read_raw_start_and_skip_header(raw_file: Iterable[str]) -> datetime:
    """Read the fixed eleven-line ActiGraph header and return its start time."""
    header = [next(raw_file, "").rstrip("\r\n") for _ in range(11)]
    start_time_line = next((line for line in header if line.startswith("Start Time")), None)
    start_date_line = next((line for line in header if line.startswith("Start Date")), None)
    if start_time_line is None or start_date_line is None:
        raise ValueError("Could not find Start Time and Start Date in the raw header.")
    start_time = start_time_line.split("Start Time", 1)[1].strip()
    start_date = start_date_line.split("Start Date", 1)[1].strip()
    return datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S")


class IntervalLookup:
    """Efficient sequential lookup for sorted, non-decreasing time intervals."""

    def __init__(self, starts: list[datetime], ends: list[datetime], values: list[int] | None = None):
        self.starts, self.ends, self.values = starts, ends, values
        self.pointer = 0

    def value_at(self, timestamp: datetime, default: int = -1) -> int:
        while self.pointer < len(self.ends) and timestamp >= self.ends[self.pointer]:
            self.pointer += 1
        if self.pointer < len(self.starts) and self.starts[self.pointer] <= timestamp < self.ends[self.pointer]:
            return default if self.values is None else self.values[self.pointer]
        return default

    def contains(self, timestamp: datetime) -> bool:
        while self.pointer < len(self.ends) and timestamp >= self.ends[self.pointer]:
            self.pointer += 1
        return (
            self.pointer < len(self.starts)
            and self.starts[self.pointer] <= timestamp < self.ends[self.pointer]
        )


class SegmentWriter:
    """Append complete ten-second CHAP samples to one independent H5 segment."""

    def __init__(self, destination: Path, metadata: dict[str, str], buffer_size: int = 256):
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(destination, "w")
        self.file.attrs.update(metadata)
        self.buffer_size = buffer_size
        self.data_buffer: list[np.ndarray] = []
        self.time_buffer: list[float] = []
        self.label_buffer: list[int] = []
        self.datasets = {
            "time": self.file.create_dataset("time", shape=(0,), maxshape=(None,), dtype="f8", chunks=True),
            "data": self.file.create_dataset(
                "data", shape=(0, 100, 3), maxshape=(None, 100, 3), dtype="f4",
                chunks=(1, 100, 3), compression="gzip",
            ),
            "non_wear": self.file.create_dataset("non_wear", shape=(0,), maxshape=(None,), dtype="i1", chunks=True),
            "sleeping": self.file.create_dataset("sleeping", shape=(0,), maxshape=(None,), dtype="i1", chunks=True),
            "label": self.file.create_dataset("label", shape=(0,), maxshape=(None,), dtype="i1", chunks=True),
        }

    def append(self, timestamp: datetime, data: np.ndarray, label: int) -> None:
        self.time_buffer.append(timestamp.timestamp())
        self.data_buffer.append(data.astype(np.float32, copy=False))
        self.label_buffer.append(label)
        if len(self.data_buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.data_buffer:
            return
        count = len(self.data_buffer)
        old_size = self.datasets["label"].shape[0]
        new_size = old_size + count
        payloads = {
            "time": np.asarray(self.time_buffer, dtype=np.float64),
            "data": np.stack(self.data_buffer),
            "non_wear": np.zeros(count, dtype=np.int8),
            "sleeping": np.zeros(count, dtype=np.int8),
            "label": np.asarray(self.label_buffer, dtype=np.int8),
        }
        for name, values in payloads.items():
            dataset = self.datasets[name]
            dataset.resize((new_size,) + dataset.shape[1:])
            dataset[old_size:new_size] = values
        self.data_buffer.clear()
        self.time_buffer.clear()
        self.label_buffer.clear()

    def close(self) -> int:
        self.flush()
        count = int(self.datasets["label"].shape[0])
        self.file.close()
        return count


def parse_acceleration_line(line: str, raw_path: Path) -> np.ndarray:
    values = np.fromstring(line.strip(), sep=",", dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected three numeric values in {raw_path}, got: {line[:100]!r}")
    return values


def preprocess_recording(
    recording: Recording,
    unknown_by_subject: dict[str, list[tuple[datetime, datetime]]],
    output_dir: Path,
    gt3x_frequency: int,
    downsample_frequency: int,
) -> tuple[Path, int]:
    """Downsample one recording and write complete non-overlapping ten-second samples."""
    if gt3x_frequency % downsample_frequency != 0:
        raise ValueError("--gt3x-frequency must be an integer multiple of --downsample-frequency")
    raw_per_resolution = gt3x_frequency // downsample_frequency
    resolution = 1 / downsample_frequency
    starts, ends, labels = read_event_labels(recording.event_path)
    label_lookup = IntervalLookup(starts, ends, labels)
    unknown_intervals = unknown_by_subject.get(recording.subject, [])
    unknown_lookup = IntervalLookup(
        [item[0] for item in unknown_intervals], [item[1] for item in unknown_intervals]
    )

    destination = output_dir / recording.subject / f"{recording.basename}.h5"
    metadata = {
        "subject_id": recording.subject,
        "environment": recording.environment,
        "wear_location": "W",
        "age_group": recording.age_group,
        "source_split": recording.source_split,
        "raw_filename": recording.raw_path.name,
        "event_filename": recording.event_path.name,
        "sample_rate_hz": str(downsample_frequency),
        "window_seconds": "10",
    }
    writer = SegmentWriter(destination, metadata)
    samples: list[np.ndarray] = []
    sample_labels: list[int] = []
    sample_start: datetime | None = None

    try:
        with gzip.open(recording.raw_path, mode="rt", newline="") as raw_file:
            current_time = read_raw_start_and_skip_header(raw_file)
            while True:
                raw_lines: list[str] = []
                for _ in range(raw_per_resolution):
                    line = raw_file.readline()
                    if not line:
                        break
                    if line.strip():
                        raw_lines.append(line)
                if len(raw_lines) != raw_per_resolution:
                    break

                acc = np.mean([parse_acceleration_line(line, recording.raw_path) for line in raw_lines], axis=0)
                if sample_start is None:
                    sample_start = current_time
                label = -1 if unknown_lookup.contains(current_time) else label_lookup.value_at(current_time)
                samples.append(acc)
                sample_labels.append(label)

                if len(samples) == 100:  # ten seconds at 10 Hz
                    # Preserve unknown/unlabelled intervals: one such tenth invalidates the H5 sample.
                    window_label = (
                        -1
                        if any(value < 0 for value in sample_labels)
                        else int(sum(sample_labels) > len(sample_labels) / 2)
                    )
                    writer.append(sample_start, np.stack(samples), window_label)
                    samples.clear()
                    sample_labels.clear()
                    sample_start = None
                current_time += timedelta(seconds=resolution)
    except Exception:
        writer.close()
        raise
    return destination, writer.close()


def write_manifest(rows: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject", "age_group", "source_split", "environment", "wear_location",
        "raw_path", "event_path", "processed_path", "num_10s_samples",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing train/ and test/.")
    parser.add_argument("--unknown-root", type=Path, default=None, help="Directory containing MOCA_unknownEvent.csv files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for per-recording H5 files.")
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Optional manifest; default: <output-dir>/processed_segments.csv.")
    parser.add_argument("--gt3x-frequency", type=int, default=80)
    parser.add_argument("--downsample-frequency", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing segment H5 files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    recordings = find_recordings(args.data_root)
    subjects_in_train = {item.subject for item in recordings if item.source_split == "train"}
    subjects_in_test = {item.subject for item in recordings if item.source_split == "test"}
    overlap = subjects_in_train.intersection(subjects_in_test)
    if overlap:
        raise ValueError(f"Subjects appear in both source train and test: {sorted(overlap)}")
    unknown_by_subject = parse_unknown_events(args.unknown_root)

    existing = [args.output_dir / item.subject / f"{item.basename}.h5" for item in recordings]
    if not args.overwrite and any(path.exists() for path in existing):
        first = next(path for path in existing if path.exists())
        raise FileExistsError(f"Output already exists: {first}. Use --overwrite only to regenerate it.")

    manifest_rows: list[dict[str, str | int]] = []
    for index, recording in enumerate(recordings, start=1):
        LOGGER.info("[%d/%d] %s", index, len(recordings), recording.raw_path.name)
        destination, count = preprocess_recording(
            recording, unknown_by_subject, args.output_dir, args.gt3x_frequency, args.downsample_frequency
        )
        manifest_rows.append({
            "subject": recording.subject,
            "age_group": recording.age_group,
            "source_split": recording.source_split,
            "environment": recording.environment,
            "wear_location": "W",
            "raw_path": str(recording.raw_path),
            "event_path": str(recording.event_path),
            "processed_path": str(destination),
            "num_10s_samples": count,
        })

    manifest_path = args.manifest_csv or args.output_dir / "processed_segments.csv"
    write_manifest(manifest_rows, manifest_path)
    LOGGER.info("Done: %d wrist recordings; manifest: %s", len(recordings), manifest_path)


if __name__ == "__main__":
    main()
