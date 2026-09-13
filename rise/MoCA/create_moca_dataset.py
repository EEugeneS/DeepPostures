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
import multiprocessing
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


def iter_file_windows(
    raw_path: Path,
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
    """Yield valid windows from one raw ActiGraph file."""
    samples_per_window = args.gt3x_frequency * args.window_seconds
    if args.gt3x_frequency % args.label_frequency != 0:
        raise ValueError("gt3x-frequency must be divisible by label-frequency.")

    stem = source_stem(raw_path)
    subject_id = subject_id_from_stem(stem, args)
    if subject_id not in assignments:
        counters["raw_files_without_split_subject"] += 1
        return
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
    """Yield valid windows from every file in a single visit source."""
    for raw_path in raw_files(raw_dir):
        yield from iter_file_windows(
            raw_path, event_dir, assignments, valid_days, sleep_logs, non_wear,
            label_map, args, include_signals, counters,
        )


# Worker state is initialized once per process.  Raw files are independent, so
# workers can count or write separate files without sharing Python objects.
_WORKER_STATE: Dict[str, object] = {}


def initialise_worker(
    args: argparse.Namespace,
    assignments: Dict[str, str],
    label_map: Dict[str, int],
    sources: List[Dict[str, object]],
    staging_paths: Optional[Dict[str, Tuple[str, str]]] = None,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "args": args,
        "assignments": assignments,
        "label_map": label_map,
        "sources": sources,
        "staging_paths": staging_paths,
    }


def worker_windows(source_index: int, raw_path_string: str, include_signals: bool, counters: Counter):
    source = _WORKER_STATE["sources"][source_index]
    return iter_file_windows(
        Path(raw_path_string),
        source["event_dir"],
        _WORKER_STATE["assignments"],
        source["valid_days"],
        source["sleep_logs"],
        source["non_wear"],
        _WORKER_STATE["label_map"],
        _WORKER_STATE["args"],
        include_signals,
        counters,
    )


def count_raw_task(task: Tuple[int, int, str]) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    task_index, source_index, raw_path_string = task
    counts: Counter = Counter()
    dropped: Counter = Counter()
    for split, _, _, _, _, _ in worker_windows(source_index, raw_path_string, False, dropped):
        counts[split] += 1
    return task_index, dict(counts), dict(dropped)


def write_raw_task(
    task: Tuple[int, int, str, Dict[str, int]]
) -> Tuple[int, Dict[str, int]]:
    task_index, source_index, raw_path_string, offsets = task
    staging_paths = _WORKER_STATE["staging_paths"]
    arrays = _WORKER_STATE.get("x_arrays")
    labels = _WORKER_STATE.get("y_arrays")
    if arrays is None or labels is None:
        arrays = {
            split: np.load(x_path, mmap_mode="r+")
            for split, (x_path, _) in staging_paths.items()
        }
        labels = {
            split: np.load(y_path, mmap_mode="r+")
            for split, (_, y_path) in staging_paths.items()
        }
        _WORKER_STATE["x_arrays"] = arrays
        _WORKER_STATE["y_arrays"] = labels
    written: Counter = Counter()
    try:
        ignored: Counter = Counter()
        for split, signals, label, _, _, _ in worker_windows(source_index, raw_path_string, True, ignored):
            index = offsets[split] + written[split]
            arrays[split][index, 0] = signals
            labels[split][index] = label
            written[split] += 1
        return task_index, dict(written)
    finally:
        for array in arrays.values():
            array.flush()
        for array in labels.values():
            array.flush()


def run_parallel(
    worker_function,
    tasks: Sequence[tuple],
    args: argparse.Namespace,
    assignments: Dict[str, str],
    label_map: Dict[str, int],
    sources: List[Dict[str, object]],
    staging_paths: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Iterator[tuple]:
    initargs = (args, assignments, label_map, sources, staging_paths)
    if args.mp == 1:
        initialise_worker(*initargs)
        for task in tasks:
            yield worker_function(task)
        return

    with multiprocessing.Pool(
        processes=args.mp,
        initializer=initialise_worker,
        initargs=initargs,
    ) as pool:
        yield from pool.imap_unordered(worker_function, tasks, chunksize=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MoCA-ready Rise tensors from raw ActiGraph data.")
    parser.add_argument("--gt3x-dir", help="Legacy single-visit raw CSV/CSV.GZ directory.")
    parser.add_argument("--activpal-dir", help="Legacy single-visit matching Rise AP Event CSV directory.")
    parser.add_argument("--split-csv", required=True, help="subject_id,split manifest created by create_subject_split.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for X_*.pt and y_*.pt tensors.")
    parser.add_argument("--valid-days-file", default=None)
    parser.add_argument("--sleep-logs-file", default=None)
    parser.add_argument("--non-wear-times-file", default=None)
    parser.add_argument(
        "--visit",
        action="append",
        nargs=6,
        metavar=("NAME", "GT3X_DIR", "ACTIVPAL_DIR", "VALID_DAYS_CSV", "SLEEP_LOG_CSV", "NON_WEAR_CSV"),
        help=(
            "One Rise visit source. Repeat for BL and FV. When used, provide "
            "NAME, raw directory, Event directory, valid-day CSV, sleep-log CSV, and non-wear CSV."
        ),
    )
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
    parser.add_argument("--mp", type=int, default=1,
                        help="Number of raw-file preprocessing workers (default: 1).")
    parser.add_argument("--keep-temporary-arrays", action="store_true",
                        help="Keep memory-mapped .npy staging arrays after writing tensors.")
    return parser.parse_args()


def visit_sources(args: argparse.Namespace) -> List[Tuple[str, Path, Path, Optional[Path], Optional[Path], Optional[Path]]]:
    """Return one or more raw/Event/support-file groups to combine into one dataset."""
    sources = []
    if args.visit:
        if any((args.gt3x_dir, args.activpal_dir, args.valid_days_file,
                args.sleep_logs_file, args.non_wear_times_file)):
            raise ValueError("Use either repeated --visit sources or the legacy single-visit arguments, not both.")
        for name, raw_dir, event_dir, valid_days, sleep_logs, non_wear in args.visit:
            sources.append((
                name,
                Path(raw_dir),
                Path(event_dir),
                Path(valid_days),
                Path(sleep_logs),
                Path(non_wear),
            ))
    else:
        if not args.gt3x_dir or not args.activpal_dir:
            raise ValueError("Provide either --visit or both --gt3x-dir and --activpal-dir.")
        sources.append((
            "default",
            Path(args.gt3x_dir),
            Path(args.activpal_dir),
            Path(args.valid_days_file) if args.valid_days_file else None,
            Path(args.sleep_logs_file) if args.sleep_logs_file else None,
            Path(args.non_wear_times_file) if args.non_wear_times_file else None,
        ))

    for name, raw_dir, event_dir, valid_days, sleep_logs, non_wear in sources:
        for description, path in (("raw", raw_dir), ("Event", event_dir)):
            if not path.is_dir():
                raise NotADirectoryError(f"{name} {description} directory does not exist: {path}")
        for description, path in (("valid-day", valid_days), ("sleep-log", sleep_logs), ("non-wear", non_wear)):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"{name} {description} CSV does not exist: {path}")
    return sources


def main() -> None:
    args = parse_args()
    if args.gt3x_frequency <= 0 or args.window_seconds <= 0 or args.label_frequency <= 0:
        raise ValueError("Frequencies and window-seconds must be positive.")
    if args.mp <= 0:
        raise ValueError("--mp must be positive.")
    if (args.n_start_id is None) != (args.n_end_id is None):
        raise ValueError("--n-start-id and --n-end-id must be provided together.")
    if args.n_start_id is not None and (args.n_start_id <= 0 or args.n_start_id > args.n_end_id):
        raise ValueError("Invalid ID slice boundaries.")

    split_path = Path(args.split_csv)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    sources = visit_sources(args)
    label_map = {str(key): int(value) for key, value in json.loads(args.activpal_label_map).items()}
    assignments = read_split_manifest(split_path)

    worker_sources: List[Dict[str, object]] = []
    tasks: List[Tuple[int, int, str]] = []
    for source_index, (name, raw_dir, event_dir, valid_days_path, sleep_logs_path, non_wear_path) in enumerate(sources):
        worker_sources.append({
            "name": name,
            "event_dir": event_dir,
            "valid_days": read_valid_days(valid_days_path),
            "sleep_logs": read_sleep_logs(sleep_logs_path),
            "non_wear": read_non_wear(non_wear_path),
        })
        for raw_path in raw_files(raw_dir):
            tasks.append((len(tasks), source_index, str(raw_path)))
    if not tasks:
        raise RuntimeError("No raw CSV/CSV.GZ files were found in the supplied visit sources.")
    print(f"Processing {len(tasks)} raw files with {args.mp} worker(s).")

    # Pass 1 counts valid windows so large studies need not reside in RAM.
    count_counters: Counter = Counter()
    counts: Counter = Counter()
    task_counts: Dict[int, Counter] = {}
    for task_index, task_count, task_dropped in run_parallel(
        count_raw_task, tasks, args, assignments, label_map, worker_sources,
    ):
        task_counts[task_index] = Counter(task_count)
        counts.update(task_count)
        count_counters.update(task_dropped)
    if len(task_counts) != len(tasks):
        raise RuntimeError("The first pass did not return a count for every raw file.")
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

    # Pass 2 writes each raw file to a preassigned, non-overlapping memmap range.
    # This permits parallel writes without copying raw signals through the parent process.
    running_offsets: Counter = Counter()
    write_tasks = []
    for task_index, source_index, raw_path_string in tasks:
        offsets = {split: running_offsets[split] for split in ("train", "validation", "test")}
        write_tasks.append((task_index, source_index, raw_path_string, offsets))
        running_offsets.update(task_counts[task_index])

    positions: Dict[int, Counter] = {}
    staging_paths = {split: (str(x_path), str(y_path)) for split, (x_path, y_path) in staging.items()}
    for task_index, task_positions in run_parallel(
        write_raw_task, write_tasks, args, assignments, label_map, worker_sources, staging_paths,
    ):
        positions[task_index] = Counter(task_positions)
    if len(positions) != len(tasks):
        raise RuntimeError("The second pass did not return a write count for every raw file.")
    for task_index, expected in task_counts.items():
        actual = positions[task_index]
        if expected != actual:
            raise RuntimeError(
                f"Window counts changed for raw task {task_index}: expected {dict(expected)}, got {dict(actual)}"
            )

    for split in ("train", "validation", "test"):
        x_array = np.load(staging[split][0], mmap_mode="r")
        y_array = np.load(staging[split][1], mmap_mode="r")
        torch.save(torch.from_numpy(x_array), output_dir / f"X_{split}.pt")
        torch.save(torch.from_numpy(y_array), output_dir / f"y_{split}.pt")

    metadata = {
        "input_shape": [1, samples_per_window, 3],
        "sampling_hz": args.gt3x_frequency,
        "window_seconds": args.window_seconds,
        "label_frequency": args.label_frequency,
        "label_map": label_map,
        "visit_sources": [
            {"name": name, "raw_dir": str(raw_dir), "activpal_dir": str(event_dir)}
            for name, raw_dir, event_dir, _, _, _ in sources
        ],
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
