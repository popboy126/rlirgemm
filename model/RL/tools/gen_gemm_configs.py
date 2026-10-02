"""Convert RL policy output into a performance task configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def export_parameters(performance_template: Path, output: Path,
                      inference_json: Path | None = None) -> None:
    """Copy a published performance template and optionally replace opt_gemm rows.

    ``inference_json`` is produced by ``do_ml_inference.py`` and must contain
    rows with ``matrix_size`` and runtime parameter fields.
    """
    task: dict[str, Any] = json.loads(performance_template.read_text(encoding="utf-8"))
    if inference_json is not None:
        rows = {tuple(row["matrix_size"]): row for row in json.loads(inference_json.read_text(encoding="utf-8"))}
        mapping = task["gemm_configs"]["params_mapping"]
        for shape in task["gemm_configs"]["shapes"]:
            row = rows.get(tuple(shape["common"]))
            if row is None:
                continue
            values = list(shape["configs"].get("opt_gemm", task["gemm_configs"]["default_params"]))
            for name, index in mapping.items():
                if name in row and index < len(values):
                    values[index] = row[name]
            shape["configs"]["opt_gemm"] = values
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference", type=Path)
    args = parser.parse_args()
    export_parameters(args.template, args.output, args.inference)


if __name__ == "__main__":
    main()
