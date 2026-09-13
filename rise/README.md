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
participants into `train` and `validation`. All files and visits belonging to
one subject remain in the same final split. With the BL/FV raw directories,
validation selection is stratified by `BL-only`, `FV-only`, and `BL+FV`
availability, so both visits remain represented in train and validation.

For the current P2 list (311 `train_set`, 95 `test_set`), the default 25%
validation split yields approximately 233 train, 78 validation, and 95 test
subjects.

```bash
python rise/create_subject_split.py \
  --source-csv /path/to/P2_train_test_rand.csv \
  --bl-raw-dir /path/to/AG/BL \
  --fv-raw-dir /path/to/AG/FV \
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

For Rise, provide both BL and FV in one invocation. This prevents the second
visit from overwriting tensors created for the first visit:

```bash
python rise/MoCA/create_moca_dataset.py \
  --split-csv /path/to/rise_subject_split.csv \
  --output-dir /path/to/rise_moca_30hz \
  --expression-after-id BL FV \
  --visit BL /path/to/AG/BL /path/to/AP_10s/BL /path/to/P2_BL_valid_day.csv /path/to/P2_BL_sleepLog.csv /path/to/P2_BL_NonWear.csv \
  --visit FV /path/to/AG/FV /path/to/AP_10s/FV /path/to/P2_FV_valid_day.csv /path/to/P2_FV_sleepLog.csv /path/to/P2_FV_NonWear.csv
```

It writes `X_train.pt`, `X_validation.pt`, `X_test.pt`, their corresponding
labels, and `dataset_metadata.json`. The existing MoCA loader only knows
train/test names, so it must be updated before using a held-out validation set
and final test set separately.
