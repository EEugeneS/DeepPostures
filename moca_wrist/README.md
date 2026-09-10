# MoCA wrist preprocessing

This folder contains MoCA/wrist-specific data preparation scripts. Only
`*_W.csv.gz` wrist recordings are used; hip recordings are not mixed in. Each
same-name raw/Event pair is an independent segment, so no model sample crosses
a C/E/H/S file boundary.

## 1. Make the subject split

The original top-level `train`/`test` allocation is preserved. The script
selects a validation set from original-train subjects separately in each age
group.

```bash
python create_subject_split.py \\
  --data-root /path/to/MOCA \\
  --output-csv splits/moca_wrist_subject_split.csv \\
  --validation-fraction 0.25 \\
  --seed 42
```

The fixed output CSV has `subject`, `age_group`, `split`, `source_split`, and
`seed` columns. Use the same file for every CHAP experiment.

## 2. Preprocess wrist recordings

```bash
python pre_process_data.py \\
  --data-root /path/to/MOCA \\
  --unknown-root '/path/to/support files' \\
  --output-dir /path/to/MOCA/processed_wrist
```

The output form is `processed_wrist/<subject>/<subject>_<env>_W.h5`. H5
contains ten-second samples with `data` shape `(N, 100, 3)` at 10 Hz, and the
binary labels `0 = sitting`, `1 = non-sitting` (standing or stepping). Any
ten-second sample touching an absent Event label or an `unknownEvent` interval
gets `label = -1`; omit it during training and evaluation.

The preprocessing command refuses to overwrite existing generated segment H5
files. Add `--overwrite` only when deliberately regenerating them.

## 3. Make CHAP-ready datasets

```bash
python create_dataset_split.py \\
  --data-dir /path/to/MOCA/processed_wrist \\
  --split-csv splits/moca_wrist_subject_split.csv \\
  --output-dir /path/to/MOCA/chap_ready_wrist
```

This produces `10s_train.h5`, `10s_val.h5`, and `10s_test_complete.h5` with
the same core arrays as CHAP2. Windows are made separately within each segment
H5, and never cross a recording-file boundary. Extra `segment_id` and
`environment` datasets preserve the source context for later analysis.
