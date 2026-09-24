# MoCA wrist preprocessing

These scripts prepare the cleaned MoCA wrist data for CHAP. They use only
ActiGraph wrist recordings (files ending in _W), pair each raw file with its
same-name Event file, and keep every C/E/E1/E2/H/S recording as an independent
segment.

## Cleaned source layout

    /niddk-data-central/MOCA/
    ├── all/
    │   ├── 80Hz_RAW/<age-group>/*.csv.gz
    │   └── Event_File/<age-group>/*.csv
    └── support_files/
        ├── randomization/<age-group>.csv
        └── <age-group>/MOCA_unknownEvent.csv

The subject-split builder verifies that each wrist participant occurs in
exactly one age group, that the raw-folder age agrees with the randomization
CSV, and that every wrist participant has a source train/test assignment.

## Container image

The Rise container also packages this directory. Build and publish the Linux
AMD64 image from the repository root:

    bash rise/docker/build_and_push.sh \
      ghcr.io/eeugenes/deep-postures-rise:moca-wrist-chap-compatible-v1

Source code and Python dependencies are inside the image. The Kubernetes Jobs
mount CephFS only for source data and generated artifacts; no git pull or pip
installation is performed on the server.

## 1. Create the subject split

The study-provided per-age train/test assignment is preserved. Within each age
group, 25% of source-train participants are assigned to validation.

    python moca_wrist/create_subject_split.py \
      --data-root /niddk-data-central/MOCA \
      --output-csv /niddk-data-central/yuchen_workspace/moca_wrist_outputs/splits/moca_wrist_subject_split.csv \
      --validation-fraction 0.25 \
      --seed 42

The Kubernetes manifest is
rise/k8s/moca-wrist-subject-split-job.yaml.

## 2. Preprocess wrist recordings

    python moca_wrist/pre_process_data.py \
      --data-root /niddk-data-central/MOCA \
      --output-dir /niddk-data-central/yuchen_workspace/moca_wrist_outputs/processed_wrist

Outputs have the form
processed_wrist/<subject>/<subject>_<environment>_W.h5. Each H5 stores
non-overlapping 10-second samples with data shape (N, 100, 3) at 10 Hz.

Labels are 0 for sitting and 1 for non-sitting (standing or stepping). Missing
or unsupported Event labels become -1 at the 0.1-second resolution. Samples in
an unknown interval also receive -1. Following CHAP2, each 10-second label is
the mode of all 100 labels, including -1; tied counts select the smallest label,
matching `scipy.stats.mode`. Event and unknown intervals use CHAP1/2's closed
boundary convention (`start <= time <= end`). Unknown-event IDs are recording basenames (for
example, 221_C_W), so intervals are matched to the exact wrist segment rather
than only to the participant. No window crosses an input recording file.

## 3. Create CHAP-ready datasets

    python moca_wrist/create_dataset_split.py \
      --data-dir /niddk-data-central/yuchen_workspace/moca_wrist_outputs/processed_wrist \
      --split-csv /niddk-data-central/yuchen_workspace/moca_wrist_outputs/splits/moca_wrist_subject_split.csv \
      --output-dir /niddk-data-central/yuchen_workspace/moca_wrist_outputs/chap_ready_wrist

This produces 10s_train.h5, 10s_val.h5, and 10s_test_complete.h5. Each CHAP
sample contains 42 consecutive, non-overlapping 10-second windows. As in CHAP2,
a final 10-second label of -1 ends the valid run and incomplete trailing runs
are discarded. Sequences also reset at recording boundaries and timestamp gaps.

For the pooled plus six age-specific experiment inputs, submit
`rise/k8s/moca-wrist-chap-dataset-job.yaml`. After its seven indexed tasks
finish, `rise/k8s/moca-wrist-17-experiments-job.yaml` runs the frozen matrix:
two zero-shot evaluations, fourteen fine-tuning runs (pooled plus six age
groups for each checkpoint), and one pooled scratch run. The experiment jobs
write aggregate metric JSON only and do not emit per-window prediction CSVs.
Training selects checkpoints using validation balanced accuracy and does not
touch test. Test evaluation is submitted separately only after the tuning
protocol and all hyperparameters are frozen.
