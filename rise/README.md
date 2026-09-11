# Rise experiment utilities

```text
rise/
├── create_subject_split.py       # shared subject-level train/validation/test manifest
├── CHAP2.0/pre_process_data.py   # Rise copy of the CHAP2 preprocessing entry point
└── MoCA/create_moca_dataset.py   # raw 30 Hz Rise → MoCA tensors
```

`create_subject_split.py` converts the study-provided P2 train/test
randomization into the `subject_id,split` manifest consumed by
`CHAP2/create_dataset_split.py`.

It keeps every `test_set` participant as `test` and divides only `train_set`
participants into `train` and `validation`.  All files and visits belonging to
one subject must remain in the same final split.

For the current P2 list (311 `train_set`, 95 `test_set`), the default 25%
validation split yields approximately 233 train, 78 validation, and 95 test
subjects.

```bash
python rise/create_subject_split.py \
  --source-csv /path/to/P2_train_test_rand.csv \
  --output-csv /path/to/rise_subject_split.csv \
  --validation-fraction 0.25 \
  --seed 42
```

The result is compatible with:

```bash
python CHAP2/create_dataset_split.py \
  --data_dir /path/to/pre_processed \
  --split_csv /path/to/rise_subject_split.csv \
  --output_dir /path/to/rise_chap_split
```

## MoCA tensors

The original MoCA repository has a `RISE` Dataset class but no script that
creates its `X_*.pt` and `y_*.pt` inputs. `create_moca_dataset.py` fills that
gap without using CHAP's downsampled HDF5 files. It preserves the 30 Hz raw
signal and creates non-overlapping 10-second windows of shape `(1, 300, 3)`.

```bash
python rise/MoCA/create_moca_dataset.py \
  --gt3x-dir /path/to/AG_30Hz \
  --activpal-dir /path/to/AP_10s \
  --split-csv /path/to/rise_subject_split.csv \
  --valid-days-file /path/to/Valid_day.csv \
  --sleep-logs-file /path/to/sleepLog.csv \
  --non-wear-times-file /path/to/NonWear.csv \
  --output-dir /path/to/rise_moca_30hz
```

It writes `X_train.pt`, `X_validation.pt`, `X_test.pt`, their corresponding
labels, and `dataset_metadata.json`. The existing MoCA loader only knows
train/test names, so it must be updated before using a held-out validation set
and final test set separately.
