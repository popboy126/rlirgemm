"""Checkpoint based RL inference entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gemm_adaptive_trainer import GemmAdaptiveTrainer


def do_inference(name: str, model_save_dir: str, model_name: str,
                 gemm_shapes: list[tuple[int, int, int]], log_dir: str,
                 is_debug: bool = False) -> list[dict[str, Any]]:
    checkpoint = Path(model_save_dir) / model_name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}; run do_ml_learning.py first or pass a trained checkpoint"
        )
    trainer = GemmAdaptiveTrainer(
        inference_name=name, logger_dir=log_dir, is_debug=is_debug, is_train=False,
        device="cpu", model_save_dir=model_save_dir, checkpoint_name=model_name,
    )
    return trainer.do_inference(gemm_shapes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL inference for GEMM shapes")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shapes", type=Path, required=True, help="JSON list of [M,K,N]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true",
                        help="use a dummy vector environment (useful for local smoke tests)")
    args = parser.parse_args()
    raw = json.loads(args.shapes.read_text(encoding="utf-8"))
    shapes = [tuple(int(v) for v in shape) for shape in raw]
    result = do_inference("release-inference", str(args.checkpoint.parent), args.checkpoint.name,
                          shapes, str(args.output.parent), is_debug=args.debug)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def json_default(value: Any) -> Any:
        # Policy/environment metadata may contain NumPy scalars or arrays.
        # Convert those explicitly so the documented JSON output is usable by
        # the parameter-export tools.
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    args.output.write_text(
        json.dumps(result, indent=2, default=json_default) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
