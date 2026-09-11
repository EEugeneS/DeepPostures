#!/usr/bin/env python3
"""Build subject-disjoint Rise tensors for MoCA.

The MoCA RISE loader expects tensors initially shaped (N, 1, T, 3).  This
script keeps the original 30 Hz acceleration signal, forms non-overlapping
10-second windows (T=300 by default), and applies the same 10 Hz label,
sleep, non-wear, and valid-day decisions as CHAP2 preprocessing.
"""

import argparse
import csv
import gzip
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_LABEL_MAP = {
    "0.0": 0, "1.0": 1, "2.0": 1, "2.1": 1, "3.1": -1,
    "3.2": 0, "4.0": -1, "5.0": 0, "0": 0, "1": 1,
    "2": 1, "4": -1, "5": 0, "-1.0": -1, "-1": -1,
}
DATE_FORMATS = ("%m/%d/%Y", "%Y/%m/%d", "%Y-%m-%d", "%m-%d-%Y")
DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%y %H:%M",
    "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
)


def normalise(value: str) -> str:
    return value.strip().strip('"').lower()


def parse_date(value: str) -> datetime.date:
    value = value.strip().strip('"')
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date {value!r}; expected one of {DATE_FORMATS}.")


def parse_datetime(value: str) -> datetime:
    value = value.strip().strip('"')
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported datetime {value!r}; expected one of {DATETIME_FORMATS}.")


def source_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".csv.gz"):
        return name[:-7]
    if name.endswith(".csv"):
        return name[:-4]
    raise ValueError(f"Unsupported raw filename: {path}")


def subject_id_from_stem(stem: str, args: argparse.Namespace) -> str:
    if args.n_start_id is not None:
        return stem[args.n_start_id - 1:args.n_end_id]
    if args.expression_after_id:
        for separator in args.expression_after_id:
            if separator in stem:
                return stem.split(separator, 1)[0]
    return stem


def read_split_manifest(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as split_file:
        reader = csv.DictReader(split_file)
        expected = {normalise(name): name for name in (reader.fieldnames or [])}
        if "subject_id" not in expected or "split" not in expected:
            raise ValueError("Split CSV must have subject_id and split columns.")
        assignments: Dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            subject_id = (row[expected["subject_id"]] or "").strip().strip('"')
            split = normalise(row[expected["split"]] or "")
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unexpected split {split!r} on row {row_number}.")
            if not subject_id or subject_id in assignments:
                raise ValueError(f"Missing or duplicate subject ID on row {row_number}.")
            assignments[subject_id] = split
    return assignments


def read_valid_days(path: Optional[Path]) -> Dict[str, set]:
    if path is None:
        return {}
    valid_days: Dict[str, set] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = {normalise(name): name for name in (reader.fieldnames or [])}
        if "id" not in fields or "date.valid.day" not in fields:
            raise ValueError("valid-day CSV must have ID and Date.Valid.Day columns.")
        for row in reader:
            valid_days[(row[fields["id"]] or "").strip().strip('"')].add(
                parse_date(row[fields["date.valid.day"]] or "")
            )
    return valid_days


def read_sleep_logs(path: Optional[Path]) -> Dict[str, List[Tuple[datetime, datetime]]]:
    if path is None:
        return {}
    sleep_logs: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = {normalise(name): name for name in (reader.fieldnames or [])}
        if not {"id", "startsleep", "endsleep"}.issubset(fields):
            raise ValueError("sleep-log CSV must have ID, startSleep, and endSleep columns.")
        for row in reader:
            subject_id = (row[fields["id"]] or "").strip().strip('"')
            sleep_logs[subject_id].append((
                parse_datetime(row[fields["startsleep"]] or ""),
                parse_datetime(row[fields["endsleep"]] or ""),
            ))
    return sleep_logs


def read_non_wear(path: Optional[Path]) -> Dict[str, List[Tuple[datetime, datetime]]]:
    if path is None:
        return {}
    non_wear: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = {normalise(name): name for name in (reader.fieldnames or [])}
        required = {"id", "date.nw.start", "time.nw.start", "date.nw.end", "time.nw.end"}
        if not required.issubset(fields):
            raise ValueError("non-wear CSV must have ID and Date/Time.Nw.Start/End columns.")
        for row in reader:
            subject_id = (row[fields["id"]] or "").strip().strip('"')
            start = parse_datetime(
                f"{row[fields['date.nw.start']] or ''} {row[fields['time.nw.start']] or ''}"
            )
            end = parse_datetime(
                f"{row[fields['date.nw.end']] or ''} {row[fields['time.nw.end']] or ''}"
            )
            non_wear[subject_id].append((start, end))
    return non_wear


class EventLookup:
    def __init__(self, path: Path, label_map: Dict[str, int]):
        events: List[Tuple[datetime, datetime, int]] = []
        with path.open(newline="", encoding="utf-8-sig") as input_file:
            reader = csv.DictReader(input_file)
            fields = {normalise(name): name for name in (reader.fieldnames or [])}
            time_column = fields.get("time")
            interval_column = fields.get("interval (s)")
            activity_column = next(
                (name for normalised, name in fields.items() if normalised.startswith("activitycode")),
                None,
            )
            if not time_column or not interval_column or not activity_column:
                raise ValueError(
                    f"Event CSV {path} needs Time, Interval (s), and ActivityCode columns."
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    excel_time = float(row[time_column])
                    start = datetime.utcfromtimestamp(
                        round((excel_time - 25569.0) * 86400.0 * 10.0) / 10.0
                    )
                    interval = timedelta(seconds=round(float(row[interval_column]) * 10.0) / 10.0)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid Event time on {path} row {row_number}.") from exc
                raw_label = str(row[activity_column]).strip()
                label = label_map.get(raw_label)
                if label is None:
                    try:
                        numeric = float(raw_label)
                        label = label_map.get(str(numeric), label_map.get(f"{numeric:g}"))
                    except ValueError:
                        pass
                if label is None:
                    raise ValueError(f"No label mapping for {raw_label!r} in {path} row {row_number}.")
                events.append((start, start + interval, label))
        self.events = events
        self.pointer = 0

    def label_at(self, timestamp: datetime) -> int:
        while self.pointer < len(self.events) and timestamp > self.events[self.pointer][1]:
            self.pointer += 1
        if self.pointer == len(self.events):
            return -1
        start, end, label = self.events[self.pointer]
        return label if start <= timestamp <= end else -1


def in_intervals(timestamp: datetime, intervals: Sequence[Tuple[datetime, datetime]]) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def mode(values: Sequence[int]) -> int:
    counts = Counter(values)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def parse_raw_start(raw_file) -> datetime:
    header = [raw_file.readline().rstrip() for _ in range(11)]
    if any(line == "" for line in header):
        raise ValueError("Raw file ended before its expected 11-line ActiGraph header.")
    start_time = header[3][11:].strip() + " " + header[2][11:].strip()
    return datetime.strptime(start_time + " UTC", "%m/%d/%Y %H:%M:%S %Z")


def raw_files(raw_dir: Path) -> List[Path]:
    return sorted(
        path for path in raw_dir.iterdir()
        if path.is_file() and (path.name.endswith(".csv") or path.name.endswith(".csv.gz"))
    )


def iter_windows(
    raw_dir: Path,
    event_dir: Path,
    assignments: Dict[str, str],
    valid_days: Dict[str, set],
    sleep_logs: Dict[str, List[Tuple[datetime, datetime]]],
    non_wear: Dict[str, List[Tuple[datetime, datetime]]],
    label_map: Dict[str, int],
    args: argparse.Namespace,
    include_signals: bool,
    counters: Counter,
) -> Iterator[Tuple[str, Optional[np.ndarray], int, float, str, str]]:
    samples_per_window = args.gt3x_frequency * args.window_seconds
    if args.gt3x_frequency % args.label_frequency != 0:
        raise ValueError("gt3x-frequency must be divisible by label-frequency.")

    for raw_path in raw_files(raw_dir):
        stem = source_stem(raw_path)
        subject_id = subject_id_from_stem(stem, args)
        if subject_id not in assignments:
            counters["raw_files_without_split_subject"] += 1
            continue
        event_path = event_dir / f"{stem}.csv"
        if not event_path.is_file():
            raise FileNotFoundError(f"Missing matching Event file for {raw_path}: {event_path}")

        opener = gzip.open if raw_path.name.endswith(".gz") else open
        with opener(raw_path, mode="rt") as input_file:
            current_time = parse_raw_start(input_file)
            events = EventLookup(event_path, label_map)
            while True:
                raw_lines = [input_file.readline().rstrip() for _ in range(samples_per_window)]
                if any(line == "" for line in raw_lines):
                    break

                tick_labels: List[int] = []
                tick_non_wear: List[int] = []
                tick_sleep: List[int] = []
                for tick in range(args.window_seconds * args.label_frequency):
                    tick_time = current_time + timedelta(seconds=tick / args.label_frequency)
                    tick_labels.append(events.label_at(tick_time))
                    valid_day_missing = bool(valid_days) and (
                        subject_id in valid_days and tick_time.date() not in valid_days[subject_id]
                    )
                    tick_non_wear.append(int(
                        valid_day_missing or in_intervals(tick_time, non_wear.get(subject_id, []))
                    ))
                    tick_sleep.append(int(in_intervals(tick_time, sleep_logs.get(subject_id, []))))

                label = mode(tick_labels)
                non_wear_label = mode(tick_non_wear)
                sleep_label = mode(tick_sleep)
                window_end = current_time + timedelta(seconds=args.window_seconds)

                if label == -1:
                    counters["dropped_unlabelled"] += 1
                elif non_wear_label:
                    counters["dropped_non_wear_or_invalid_day"] += 1
                elif sleep_label:
                    counters["dropped_sleep"] += 1
                else:
                    signals = None
                    if include_signals:
                        try:
                            signals = np.asarray(
                                [[float(value) for value in line.split(",")] for line in raw_lines],
                                dtype=np.float32,
                            )
                        except ValueError as exc:
                            raise ValueError(f"Invalid acceleration row in {raw_path} at {current_time}.") from exc
                        if signals.shape != (samples_per_window, 3):
                            raise ValueError(
                                f"Expected {samples_per_window} rows of xyz acceleration in {raw_path}; "
                                f"got shape {signals.shape}."
                            )
                    yield assignments[subject_id], signals, label, current_time.timestamp(), subject_id, stem

                current_time = window_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MoCA-ready Rise tensors from raw ActiGraph data.")
    parser.add_argument("--gt3x-dir", required=True, help="Flat directory containing Rise 30 Hz raw CSV/CSV.GZ files.")
    parser.add_argument("--activpal-dir", required=True, help="Directory containing matching Rise AP Event CSV files.")
    parser.add_argument("--split-csv", required=True, help="subject_id,split manifest created by create_subject_split.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for X_*.pt and y_*.pt tensors.")
    parser.add_argument("--valid-days-file", default=None)
    parser.add_argument("--sleep-logs-file", default=None)
    parser.add_argument("--non-wear-times-file", default=None)
    parser.add_argument("--gt3x-frequency", type=int, default=30)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--label-frequency", type=int, default=10,
                        help="Frequency used to reproduce CHAP's window label mode (default: 10 Hz).")
    parser.add_argument("--activpal-label-map", default=json.dumps(DEFAULT_LABEL_MAP),
                        help="JSON mapping from raw ActivPAL code to final class.")
    parser.add_argument("--n-start-id", type=int, default=None)
    parser.add_argument("--n-end-id", type=int, default=None)
    parser.add_argument("--expression-after-id", nargs="*", default=None,
                        help="Split raw filename stem at this separator to obtain a subject ID.")
    parser.add_argument("--keep-temporary-arrays", action="store_true",
                        help="Keep memory-mapped .npy staging arrays after writing tensors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gt3x_frequency <= 0 or args.window_seconds <= 0 or args.label_frequency <= 0:
        raise ValueError("Frequencies and window-seconds must be positive.")
    if (args.n_start_id is None) != (args.n_end_id is None):
        raise ValueError("--n-start-id and --n-end-id must be provided together.")
    if args.n_start_id is not None and (args.n_start_id <= 0 or args.n_start_id > args.n_end_id):
        raise ValueError("Invalid ID slice boundaries.")

    raw_dir = Path(args.gt3x_dir)
    event_dir = Path(args.activpal_dir)
    split_path = Path(args.split_csv)
    for required_path in (raw_dir, event_dir, split_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    label_map = {str(key): int(value) for key, value in json.loads(args.activpal_label_map).items()}
    assignments = read_split_manifest(split_path)
    valid_days = read_valid_days(Path(args.valid_days_file)) if args.valid_days_file else {}
    sleep_logs = read_sleep_logs(Path(args.sleep_logs_file)) if args.sleep_logs_file else {}
    non_wear = read_non_wear(Path(args.non_wear_times_file)) if args.non_wear_times_file else {}

    # Pass 1 counts valid windows so large studies need not reside in RAM.
    count_counters: Counter = Counter()
    counts: Counter = Counter()
    for split, _, _, _, _, _ in iter_windows(
        raw_dir, event_dir, assignments, valid_days, sleep_logs, non_wear,
        label_map, args, include_signals=False, counters=count_counters,
    ):
        counts[split] += 1
    if not sum(counts.values()):
        raise RuntimeError("No valid windows were produced; check file names, timestamps, and filters.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_per_window = args.gt3x_frequency * args.window_seconds
    staging: Dict[str, Tuple[Path, Path]] = {}
    for split in ("train", "validation", "test"):
        x_path = output_dir / f".X_{split}.npy"
        y_path = output_dir / f".y_{split}.npy"
        np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                  shape=(counts[split], 1, samples_per_window, 3))
        np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(counts[split],))
        staging[split] = (x_path, y_path)

    # Pass 2 writes raw 30 Hz signal windows into the preallocated arrays.
    fill_counters: Counter = Counter()
    positions: Counter = Counter()
    x_arrays = {split: np.load(paths[0], mmap_mode="r+") for split, paths in staging.items()}
    y_arrays = {split: np.load(paths[1], mmap_mode="r+") for split, paths in staging.items()}
    for split, signals, label, _, _, _ in iter_windows(
        raw_dir, event_dir, assignments, valid_days, sleep_logs, non_wear,
        label_map, args, include_signals=True, counters=fill_counters,
    ):
        index = positions[split]
        x_arrays[split][index, 0] = signals
        y_arrays[split][index] = label
        positions[split] += 1
    if counts != positions:
        raise RuntimeError(f"Window counts changed between passes: expected {dict(counts)}, got {dict(positions)}")

    for split in ("train", "validation", "test"):
        x_arrays[split].flush()
        y_arrays[split].flush()
        torch.save(torch.from_numpy(x_arrays[split]), output_dir / f"X_{split}.pt")
        torch.save(torch.from_numpy(y_arrays[split]), output_dir / f"y_{split}.pt")

    metadata = {
        "input_shape": [1, samples_per_window, 3],
        "sampling_hz": args.gt3x_frequency,
        "window_seconds": args.window_seconds,
        "label_frequency": args.label_frequency,
        "label_map": label_map,
        "window_counts": dict(counts),
        "first_pass_dropped_windows": dict(count_counters),
    }
    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    if not args.keep_temporary_arrays:
        for x_path, y_path in staging.values():
            os.remove(x_path)
            os.remove(y_path)
    print("Created MoCA-ready Rise tensors:")
    for split in ("train", "validation", "test"):
        print(f"  {split}: {counts[split]} windows; X_{split}.pt, y_{split}.pt")
    print(f"Metadata: {output_dir / 'dataset_metadata.json'}")


if __name__ == "__main__":
    main()
