#!/usr/bin/env python3
"""Run one pooled-validation hyperparameter trial for MoCA wrist CHAP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


TRIALS = (
    ("chap1", "ft", 1e-5),
    ("chap1", "ft", 3e-5),
    ("chap1", "ft", 1e-4),
    ("solw", "ft", 1e-5),
    ("solw", "ft", 3e-5),
    ("solw", "ft", 1e-4),
    ("scratch", "scratch", 1e-4),
    ("scratch", "scratch", 3e-4),
    ("scratch", "scratch", 1e-3),
)


def trial_for_index(index: int) -> tuple[str, str, float]:
    try:
        return TRIALS[index]
    except IndexError as error:
        raise ValueError(f"Tuning index must be in 0..{len(TRIALS) - 1}, got {index}") from error


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
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    index = args.index
    if index is None:
        value = os.environ.get("JOB_COMPLETION_INDEX")
        if value is None:
            parser.error("Provide --index or JOB_COMPLETION_INDEX")
        index = int(value)

    model_name, mode, learning_rate = trial_for_index(index)
    root = Path(__file__).resolve().parents[1]
    runner = root / "rise" / "run.py"
    lr_name = format(learning_rate, ".0e").replace("-0", "-")
    output_dir = args.output_root / model_name / f"lr-{lr_name}" / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "index": index, "model": model_name, "mode": mode,
        "learning_rate": learning_rate, "selection_split": "validation",
    }, sort_keys=True), flush=True)
    command = [
        sys.executable, str(runner), "--family", "chap", "--mode", mode,
        "--data-dir", str(args.data_root / "full"),
        "--output-dir", str(output_dir), "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--lr", str(learning_rate), "--weight-decay", str(args.weight_decay),
        "--seed", str(args.seed), "--device", "cuda",
    ]
    last_checkpoint = output_dir / "last.pt"
    if last_checkpoint.is_file():
        command.extend(["--resume", str(last_checkpoint)])
    elif mode == "ft":
        checkpoint = args.chap1_checkpoint if model_name == "chap1" else args.solw_checkpoint
        command.extend(["--checkpoint", str(checkpoint)])
    subprocess.run(command, check=True)
    if not (output_dir / "best.pt").is_file():
        raise FileNotFoundError(f"Trial did not produce {output_dir / 'best.pt'}")
    print("Trial complete; test was not accessed.", flush=True)


if __name__ == "__main__":
    main()
