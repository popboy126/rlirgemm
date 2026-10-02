"""Run published performance task JSON files against local binaries.

The binary directory, output directory and CPU affinity are command line
inputs.  No development-machine path or SSH credential is used here.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PERF_RE = re.compile(r"^\[output\] benchmark: (\S+) GFlops")


def run_task(config_path: Path, bin_root: Path, output_dir: Path, core: str | None) -> None:
    task: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    gemm = task["gemm_configs"]
    mapping = gemm["params_mapping"]
    default = gemm["default_params"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for shape in gemm["shapes"]:
        m, k, n = shape["common"]
        for binary in task.get("bins", []):
            label, name = binary["label"], binary["name"]
            params = list(default)
            params[:len(shape.get("configs", {}).get(label, []))] = shape.get("configs", {}).get(label, [])
            conf = output_dir / f"{label}_{m}_{k}_{n}.conf"
            values = [params[mapping[key]] for key in (
                "element_precision", "storage_layout", "loop_order", "packing_type",
                "m_thread", "k_thread", "n_thread", "mc", "kc", "nc",
                "gemm_mat_a_prfm_bytes_stride", "gemm_mat_b_prfm_bytes_stride",
                "gemm_mat_c_prfm_bytes_stride", "add_mat_c_prfm_bytes_stride",
                "add_partial_sum_prfm_bytes_stride")]
            values += [1, params[mapping["loops"]], m, k, n,
                       params[mapping["alpha"]], params[mapping["beta"]]]
            conf.write_text(" ".join(map(str, values)), encoding="utf-8")
            command = [str(bin_root / name), str(conf), str(params[mapping["loop_order"]])]
            if core:
                command = ["taskset", "-c", core, *command]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            performances = [float(match.group(1)) for line in completed.stdout.splitlines()
                            if (match := PERF_RE.search(line))]
            records.append({"shape": [m, k, n], "bin": label, "returncode": completed.returncode,
                            "performances": performances, "stderr": completed.stderr[-1000:]})
    (output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bin-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core")
    args = parser.parse_args()
    run_task(args.config, args.bin_root, args.output_dir, args.core)


if __name__ == "__main__":
    main()
