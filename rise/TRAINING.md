# Rise training and prediction

Run from the DeepPostures repository root. A second MoCA clone is not needed.
MoCA's downstream encoder is vendored with attribution and its CC BY-NC
license in `rise/MoCA/vendor`. The Rise entry point reuses `CHAP2/chap_model.py`.
No pretrained checkpoint is newly uploaded by this change.

Install into a dedicated training environment (PyTorch >=2.2 for mmap loading):

```bash
python -m pip install -r rise/requirements-training.txt
rise_out=/niddk-data-central/yuchen_workspace/rise_outputs
chap_data="$rise_out/chap_dataset_split"
moca_data="$rise_out/moca_30hz"
chap1=CHAP2/SUBMIT_RESULT/iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth
solw=CHAP2/SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth
```

## CHAP

ZS evaluates a supplied binary CHAP checkpoint. Compare checkpoint candidates
on validation before selecting one. Repeat with `$chap1` or `$solw`:

```bash
python rise/run.py --family chap --mode predict --split validation \
  --data-dir "$chap_data" --checkpoint "$solw" \
  --output-dir "$rise_out/runs/chap/solw_zs"
python rise/run.py --family chap --mode ft --data-dir "$chap_data" \
  --checkpoint "$solw" --epochs 20 --batch-size 16 --lr 0.0001 \
  --output-dir "$rise_out/runs/chap/solw_ft/seed42"
python rise/run.py --family chap --mode scratch --data-dir "$chap_data" \
  --epochs 20 --batch-size 16 --lr 0.0001 \
  --output-dir "$rise_out/runs/chap/scratch/seed42"
```

Optional CHAP LP freezes both CNN and BiLSTM and trains `fc_bilstm` only.

## MoCA

Set `moca_ssl` to a trusted UCI SSL checkpoint on the PVC. Verify its axis
order from the source preprocessing: `--acc-rows 0 1 2` is appropriate ONLY
if the first three source grid rows are ACC xyz. Otherwise supply the actual
ACC row indices in xyz order. The loader infers tiny/base and patch width
from checkpoint weights and rejects incomplete encoder loads. Scratch must
use the SAME arch/patch as that source model (shown in the saved config).

```bash
moca_ssl=/path/to/verified/UCI_SSL_checkpoint.pth
python rise/run.py --family moca --mode lp --data-dir "$moca_data" \
  --checkpoint "$moca_ssl" --acc-rows 0 1 2 --lr 0.001 \
  --output-dir "$rise_out/runs/moca/uci_ssl_lp/seed42"
python rise/run.py --family moca --mode ft --data-dir "$moca_data" \
  --checkpoint "$moca_ssl" --acc-rows 0 1 2 --lr 0.0001 \
  --output-dir "$rise_out/runs/moca/uci_ssl_ft/seed42"
python rise/run.py --family moca --mode scratch --data-dir "$moca_data" \
  --arch base --patch 20 --lr 0.0001 \
  --output-dir "$rise_out/runs/moca/scratch/seed42"
```

This Rise LP uses a plain linear head (no upstream BatchNorm variant).
Position adaptation is performed once before training; LP freezes it, FT
updates it. No gyro signals are synthesized. Normalization and augmentation
are disabled in this initial baseline. MoCA SSL weights alone cannot do ZS.

## Final prediction and recovery

```bash
python rise/run.py --family moca --mode predict --split test \
  --data-dir "$moca_data" \
  --checkpoint "$rise_out/runs/moca/uci_ssl_ft/seed42/best.pt" \
  --output-dir "$rise_out/runs/moca/uci_ssl_ft/seed42"
```

For CHAP use `--family chap` and its data/checkpoint/output paths.
Each run saves config.json, log.jsonl, best.pt, last.pt. Prediction writes
`test_predictions.csv` and `test_metrics.json` (or validation equivalents).
Metrics use 0=sitting and 1=non-sitting with threshold 0.5 and macro F1.
Training saves the best validation balanced accuracy; it never reads test.
Metrics are pooled over windows, not subject-macro averages.

Resume with the original training mode/data/output arguments plus
`--resume /path/to/run/last.pt`. Restores model, optimizer and RNG states and
starts the next epoch; unfinished epochs are repeated. Use separate run
directories for separate experiments. Start with one GPU and `--workers 0`;
the script does not implement multi-GPU/DDP. `--device cpu` is for smoke tests.

CHAP exports subject IDs and timestamps. Current MoCA preprocessing exports
only X/y and aggregate metadata, so its prediction CSV leaves subject/time
blank and records stable tensor row indices. It supports LP/FT/scratch but
does NOT yet support temporal BiLSTM heads or CHAP timestamp-matched scoring.
Those require a per-window metadata/index export and window-boundary audit.
The 10-second start boundaries are not assumed identical between builders.

An existing CHAP checkpoint must match CHAP(2,42,2) exactly; loading fails on
missing/unexpected tensors instead of silently evaluating random layers.
