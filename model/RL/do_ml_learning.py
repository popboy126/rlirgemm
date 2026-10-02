"""Command line entry point for RL training.

The release entry point intentionally has no default task or output directory;
both must be supplied by the user so a run cannot accidentally write into the
source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from gemm_adaptive_trainer import GemmAdaptiveTrainer


def load_config(path: Path, output_dir: Path, debug: bool = False) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    def expand(value):
        if isinstance(value, str):
            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value
    config = expand(config)
    gemm = config["gemm_configs"]
    train = gemm.setdefault("train_config", {})
    # The trainer's checkpoint callbacks enumerate these directories before
    # the first checkpoint is written. Create them here so a fresh release
    # checkout can run without a pre-existing output tree.
    (output_dir / "model").mkdir(parents=True, exist_ok=True)
    (output_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    train["model_save_dir"] = str(output_dir / "model")
    train["logger_dir"] = str(output_dir / "tensorboard")
    gemm["is_train"] = True
    if debug:
        gemm["is_debug"] = True
        train["device"] = "cpu"
    return gemm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL GEMM policy")
    parser.add_argument("--config", type=Path, required=True, help="JSON under configs/rl_train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug", action="store_true", help="use the trainer's debug mode")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = GemmAdaptiveTrainer(**load_config(args.config, args.output_dir, args.debug))
    trainer.do_train()


if __name__ == "__main__":
    main()
