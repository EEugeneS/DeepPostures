from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

import create_dataset_split
import create_subject_split
import pre_process_data
import run_experiment
import run_tuning


class FilenameTests(unittest.TestCase):
    def test_supported_recording_names(self) -> None:
        expected = {
            "226_E_W.csv.gz": ("226", "E", "W"),
            "226_E1_W.csv.gz": ("226", "E1", "W"),
            "226_E2_W.csv.gz": ("226", "E2", "W"),
            "266_C_W.csv.gz": ("266", "C", "W"),
            "266_H_H.csv.gz": ("266", "H", "H"),
            "266_S_W.csv": ("266", "S", "W"),
        }
        for name, parsed in expected.items():
            self.assertEqual(pre_process_data.parse_recording_name(Path(name)), parsed)
            self.assertEqual(create_subject_split.parse_recording_name(Path(name)), parsed)

    def test_unsupported_environment_is_rejected(self) -> None:
        self.assertIsNone(pre_process_data.parse_recording_name(Path("226_E3_W.csv.gz")))


class EventTests(unittest.TestCase):
    def test_activity_mapping(self) -> None:
        self.assertEqual(pre_process_data.map_activity_code(0), 0)
        self.assertEqual(pre_process_data.map_activity_code("1"), 1)
        self.assertEqual(pre_process_data.map_activity_code(2.0), 1)
        self.assertEqual(pre_process_data.map_activity_code("unknown"), -1)
        self.assertEqual(pre_process_data.map_activity_code(""), -1)
        self.assertEqual(pre_process_data.map_activity_code(float("nan")), -1)
        self.assertEqual(pre_process_data.map_activity_code(9), -1)

    def test_unknown_interval_marks_resolution_samples(self) -> None:
        origin = datetime(2026, 1, 1)
        lookup = pre_process_data.IntervalLookup(
            [origin + timedelta(seconds=5)],
            [origin + timedelta(seconds=6)],
        )
        self.assertTrue(lookup.contains(origin + timedelta(seconds=5.5)))
        self.assertTrue(lookup.contains(origin + timedelta(seconds=6)))
        self.assertFalse(lookup.contains(origin + timedelta(seconds=6.1)))

    def test_event_boundary_belongs_to_earlier_interval_like_chap(self) -> None:
        origin = datetime(2026, 1, 1)
        boundary = origin + timedelta(seconds=10)
        lookup = pre_process_data.IntervalLookup(
            [origin, boundary],
            [boundary, origin + timedelta(seconds=20)],
            [0, 1],
        )
        self.assertEqual(lookup.value_at(boundary), 0)
        self.assertEqual(lookup.value_at(boundary + timedelta(seconds=0.1)), 1)

    def test_chap2_mode_includes_missing_labels(self) -> None:
        self.assertEqual(pre_process_data.chap2_mode([0] * 99 + [-1]), 0)
        self.assertEqual(pre_process_data.chap2_mode([1] * 60 + [-1] * 40), 1)
        self.assertEqual(pre_process_data.chap2_mode([0] * 40 + [-1] * 60), -1)

    def test_chap2_mode_uses_smallest_value_to_break_ties(self) -> None:
        self.assertEqual(pre_process_data.chap2_mode([0] * 50 + [-1] * 50), -1)
        self.assertEqual(pre_process_data.chap2_mode([0] * 50 + [1] * 50), 0)

    def test_excel_time_is_rounded_to_tenth_second(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Time", "Interval (s)", "ActivityCode (0=sedentary)"])
                writer.writerow([43615.6632523148, 10, 0])
            starts, ends, labels = pre_process_data.read_event_labels(path)
            self.assertEqual(starts, [datetime(2019, 5, 30, 15, 55, 5)])
            self.assertEqual(ends, [datetime(2019, 5, 30, 15, 55, 15)])
            self.assertEqual(labels, [0])

    def test_unknown_events_are_keyed_by_recording_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            support = Path(tmp) / "10to12.9"
            support.mkdir()
            path = support / "MOCA_unknownEvent.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ID", "unknown_DT", "int.sec", "wearDate"])
                writer.writerow(["226_E1_W", "2019-05-30 15:55:05", 2, "2019-05-30"])
            unknown = pre_process_data.parse_unknown_events(Path(tmp))
            self.assertIn("226_E1_W", unknown)
            self.assertNotIn("226", unknown)


class LayoutTests(unittest.TestCase):
    def test_cleaned_layout_and_e1_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "all" / "80Hz_RAW" / "10to12.9"
            event_dir = root / "all" / "Event_File" / "10to12.9"
            raw_dir.mkdir(parents=True)
            event_dir.mkdir(parents=True)
            (raw_dir / "226_E1_W.csv.gz").touch()
            (raw_dir / "226_E1_H.csv.gz").touch()
            (event_dir / "226_E1_W.csv").touch()
            recordings = pre_process_data.find_recordings(root)
            self.assertEqual(len(recordings), 1)
            self.assertEqual(recordings[0].environment, "E1")
            self.assertEqual(recordings[0].age_group, "10to12.9")

    def test_randomization_duplicate_subject_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for age in ("6to9.9", "10to12.9"):
                path = root / f"{age}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["subject", "random"])
                    writer.writerow(["266", "train"])
            with self.assertRaisesRegex(ValueError, "occurs more than once"):
                create_subject_split.read_randomization(root)


class SequenceTests(unittest.TestCase):
    def test_chap2_windows_are_non_overlapping_complete_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_dir = Path(tmp) / "101"
            subject_dir.mkdir()
            path = subject_dir / "101_C_W.h5"
            count = 85
            with h5py.File(path, "w") as output:
                output.attrs["environment"] = "C"
                output.attrs["age_group"] = "1.5to5.9"
                output.create_dataset("data", data=np.zeros((count, 100, 3), dtype=np.float32))
                output.create_dataset("time", data=np.arange(count, dtype=np.float64) * 10)
                output.create_dataset("label", data=np.zeros(count, dtype=np.int8))
                output.create_dataset("sleeping", data=np.zeros(count, dtype=np.int8))
                output.create_dataset("non_wear", data=np.zeros(count, dtype=np.int8))
            windows = list(create_dataset_split.iter_segment_windows(subject_dir, "101", 42))
            self.assertEqual(len(windows), 2)
            self.assertTrue(all(item[0].shape == (42, 100, 3) for item in windows))

    def test_read_split_csv_filters_age_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split_csv = Path(directory) / "split.csv"
            with split_csv.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["subject", "age_group", "split"])
                writer.writerow(["101", "1.5to5.9", "train"])
                writer.writerow(["102", "6to9.9", "validation"])
                writer.writerow(["103", "1.5to5.9", "test"])
            splits = create_dataset_split.read_split_csv(split_csv, "1.5to5.9")
            self.assertEqual(splits, {
                "train": ["101"], "validation": [], "test": ["103"]
            })

    def test_seventeen_experiment_index_mapping(self) -> None:
        experiments = [run_experiment.experiment_for_index(index) for index in range(17)]
        self.assertEqual(len(experiments), 17)
        self.assertEqual(experiments[0]["model"], "chap1")
        self.assertEqual(experiments[1]["model"], "solw")
        self.assertEqual(
            [item["scope"] for item in experiments[2:9]],
            list(run_experiment.SCOPES),
        )
        self.assertEqual(
            [item["scope"] for item in experiments[9:16]],
            list(run_experiment.SCOPES),
        )
        self.assertEqual(experiments[16], {
            "model": "scratch", "mode": "scratch", "scope": "full"
        })

    def test_nine_pooled_tuning_trials(self) -> None:
        trials = [run_tuning.trial_for_index(index) for index in range(9)]
        self.assertEqual([trial[0] for trial in trials], [
            "chap1", "chap1", "chap1", "solw", "solw", "solw",
            "scratch", "scratch", "scratch",
        ])
        self.assertTrue(all(trial[2] > 0 for trial in trials))

    def test_timestamp_gap_splits_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "segment.h5"
            with h5py.File(path, "w") as output:
                output.create_dataset("data", data=np.zeros((4, 100, 3), dtype=np.float32))
                output.create_dataset("time", data=np.asarray([0.0, 10.0, 30.0, 40.0]))
                output.create_dataset("label", data=np.zeros(4, dtype=np.int8))
                output.create_dataset("sleeping", data=np.zeros(4, dtype=np.int8))
                output.create_dataset("non_wear", data=np.zeros(4, dtype=np.int8))
            sequences = list(create_dataset_split.iter_valid_sequences(path))
            self.assertEqual([len(labels) for _, _, labels in sequences], [2, 2])


if __name__ == "__main__":
    unittest.main()
