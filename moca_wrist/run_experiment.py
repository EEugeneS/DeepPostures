#!/usr/bin/env python3
"""Run one indexed MoCA wrist CHAP experiment.

Indices 0--16 implement the frozen 17-experiment matrix. Zero-shot entries
evaluate the pooled test set and all six age-specific test sets in one job.
Training entries select ``best.pt`` from validation balanced accuracy and stop
without touching test. Zero-shot entries also default to validation. A later,
separate final-evaluation job must explicitly request ``--evaluation-split
test`` after all hyperparameters are frozen. Only aggregate metric JSON is
written; per-window prediction CSV is disabled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


AGE_GROUPS = (
    "1.5to5.9", "6to9.9", "10to12.9",
    "13to14.9", "15to17.9", "18to20",
)
SCOPES = ("full",) + AGE_GROUPS


def experiment_for_index(index: int) -> dict[str, str | None]:
    if index == 0:
        return {"model": "chap1", "mode": "predict", "scope": None}
    if index == 1:
        return {"model": "solw", "mode": "predict", "scope": None}
    if 2 <= index <= 8:
        return {"model": "chap1", "mode": "ft", "scope": SCOPES[index - 2]}
    if 9 <= index <= 15:
        return {"model": "solw", "mode": "ft", "scope": SCOPES[index - 9]}
    if index == 16:
        return {"model": "scratch", "mode": "scratch", "scope": "full"}
    raise ValueError(f"Experiment index must be in 0..16, got {index}")


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def predict(
    python: str, runner: Path, data_root: Path, output_root: Path,
    model_name: str, checkpoint: Path, scope: str, batch_size: int,
    evaluation_split: str,
) -> None:
    run([
        python, str(runner), "--family", "chap", "--mode", "predict",
        "--data-dir", str(data_root / scope),
        "--output-dir", str(output_root / model_name / "zs" / scope),
        "--checkpoint", str(checkpoint), "--split", evaluation_split,
        "--batch-size", str(batch_size), "--workers", "0",
        "--device", "cuda", "--metrics-only",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chap1-checkpoint", type=Path, required=True)
    parser.add_argument("--solw-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chap1-learning-rate", type=float, default=1e-4)
    parser.add_argument("--solw-learning-rate", type=float, default=1e-4)
    parser.add_argument("--scratch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test"),
        default="validation",
        help="Zero-shot split. Use test only in the separately authorized final evaluation.",
    )
    args = parser.parse_args()

    index = args.index
    if index is None:
        value = os.environ.get("JOB_COMPLETION_INDEX")
        if value is None:
            parser.error("Provide --index or JOB_COMPLETION_INDEX")
        index = int(value)
    experiment = experiment_for_index(index)
    root = Path(__file__).resolve().parents[1]
    runner = root / "rise" / "run.py"
    checkpoints = {
        "chap1": args.chap1_checkpoint,
        "solw": args.solw_checkpoint,
    }
    print(json.dumps({"index": index, **experiment}, sort_keys=True), flush=True)

    model_name = str(experiment["model"])
    mode = str(experiment["mode"])
    if mode == "predict":
        checkpoint = checkpoints[model_name]
        for scope in SCOPES:
            predict(
                sys.executable, runner, args.data_root, args.output_root,
                model_name, checkpoint, scope, args.batch_size,
                args.evaluation_split,
            )
        return

    scope = str(experiment["scope"])
    output_dir = args.output_root / model_name / scope / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(runner), "--family", "chap", "--mode", mode,
        "--data-dir", str(args.data_root / scope),
        "--output-dir", str(output_dir), "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--lr", str({
            "chap1": args.chap1_learning_rate,
            "solw": args.solw_learning_rate,
            "scratch": args.scratch_learning_rate,
        }[model_name]),
        "--weight-decay", str(args.weight_decay),
        "--seed", str(args.seed), "--device", "cuda",
    ]
    last_checkpoint = output_dir / "last.pt"
    if last_checkpoint.is_file():
        command.extend(["--resume", str(last_checkpoint)])
    elif mode == "ft":
        command.extend(["--checkpoint", str(checkpoints[model_name])])
    run(command)

    best_checkpoint = output_dir / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training did not produce {best_checkpoint}")
    print(
        f"Validation-selected checkpoint ready: {best_checkpoint}. "
        "Test evaluation is intentionally deferred.",
        flush=True,
    )


if __name__ == "__main__":
    main()
