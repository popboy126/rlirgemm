"""Generate and validate the two random GEMM shape sets used by the paper.

The checked-in JSON files are the normative inputs.  Generation is provided for
experiments and writes explicit, sorted JSON so its output is independent of
the iteration order of Python's ``set`` implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from math import floor
from pathlib import Path
from typing import Iterable, Sequence

Shape = tuple[int, int, int]
LIMIT_BYTES = 4 * 1024 * 1024


def gen_rnd_shapes(shape_num: int, seed: int) -> list[Shape]:
    """Reproduce the historical sampler while normalising its output order."""
    rng = random.Random(seed)
    total_elements = LIMIT_BYTES // 4
    max_m = int((total_elements / 3) ** 0.5)
    shapes: set[Shape] = set()
    while len(shapes) < shape_num:
        m = rng.randint(1, max_m)
        max_b = int(((m * m + total_elements) ** 0.5) - m)
        if max_b < m:
            continue
        b = rng.randint(m, max_b)
        remaining = total_elements - m * b
        if remaining <= 0:
            continue
        max_c = (remaining - 1) // (m + b)
        if max_c < b:
            continue
        c = rng.randint(b, max_c)
        if m * b + b * c + m * c > total_elements or m > 30:
            continue
        # Historical code names (a, c, b) as (M, K, N).
        k, n = c, b
        if k % 8 or n % 16:
            continue
        shapes.add((m, k, n))
    return sorted(shapes)


def heldout_shapes(training: Iterable[Shape], count: int, seed: int) -> list[Shape]:
    """Generate rounds of candidates and exclude every training shape."""
    training_set = set(training)
    selected: set[Shape] = set()
    round_seed = seed
    request_size = count * 2
    while len(selected) < count:
        for shape in gen_rnd_shapes(request_size, round_seed):
            if shape not in training_set:
                selected.add(shape)
                if len(selected) == count:
                    break
        round_seed += 1
        request_size = max(500, count - len(selected))
    return sorted(selected)


def validate_shapes(name: str, shapes: Sequence[Shape], expected: int = 3000,
                    strict: bool = False) -> list[Shape]:
    if len(shapes) != expected or len(set(shapes)) != expected:
        raise ValueError(f"{name}: expected {expected} unique shapes, got {len(shapes)}")
    invalid = [shape for shape in shapes
               if shape[0] > 30 or not (shape[0] < shape[2] < shape[1])
               or shape[1] % 8 or shape[2] % 16
               or 4 * (shape[0] * shape[1] + shape[1] * shape[2] + shape[0] * shape[2]) > LIMIT_BYTES]
    if invalid and strict:
        raise ValueError(f"{name}: {len(invalid)} shapes violate paper constraints; first={invalid[0]}")
    return invalid


def load_training(path: Path) -> list[Shape]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["gemm_configs"]["gemm_env_config"]["gemm_shape_config"]["shape_list"]
    return [tuple(int(v) for v in shape) for shape in raw]


def load_heldout(path: Path) -> list[Shape]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(int(v) for v in item["common"]) for item in data["gemm_configs"]["shapes"]]


def write_shapes(path: Path, shapes: Sequence[Shape]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([list(s) for s in shapes], indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=Path("configs/rl_train/ft2000_rl_diff_rnd_seed_gemm_train_t1_task0.json"))
    parser.add_argument("--heldout-config", type=Path, default=Path("configs/performance_test/performance_with_rl_diff_rnd_seed_portable_on_rnd_shapes_t1_task1_for_ft2000.json"))
    parser.add_argument("--generate", action="store_true", help="write generated shape lists")
    parser.add_argument("--training-output", type=Path, default=Path("configs/training_shapes.json"))
    parser.add_argument("--heldout-output", type=Path, default=Path("configs/heldout_shapes.json"))
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--strict", action="store_true", help="fail when historical inputs violate a stated constraint")
    args = parser.parse_args()

    training = load_training(args.training_config)
    heldout = load_heldout(args.heldout_config)
    training_invalid = validate_shapes("training", training, strict=args.strict)
    heldout_invalid = validate_shapes("heldout", heldout, strict=args.strict)
    overlap = set(training) & set(heldout)
    if overlap:
        raise ValueError(f"training/heldout overlap contains {len(overlap)} shapes")
    if args.generate:
        generated_training = gen_rnd_shapes(len(training), args.seed)
        generated_heldout = heldout_shapes(generated_training, len(heldout), args.seed)
        if set(generated_training) != set(training):
            raise ValueError("generated training set does not match the normative static set")
        if set(generated_heldout) != set(heldout):
            raise ValueError("generated held-out set does not match the normative static set")
        write_shapes(args.training_output, generated_training)
        write_shapes(args.heldout_output, generated_heldout)
        print(f"generated training sha256={sha256(args.training_output)}")
        print(f"generated heldout sha256={sha256(args.heldout_output)}")
        print("generated sets match normative static sets")
    print(f"training: {len(training)} shapes; heldout: {len(heldout)} shapes; overlap: 0")
    if training_invalid or heldout_invalid:
        print(f"historical constraint deviations: training={len(training_invalid)}, heldout={len(heldout_invalid)} (inputs preserved)")


if __name__ == "__main__":
    main()
