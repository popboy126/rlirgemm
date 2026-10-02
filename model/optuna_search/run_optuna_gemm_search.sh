#!/usr/bin/env bash
set -euo pipefail

# Run from the release root.  The task list, output directory and optional
# external environment are explicit so this script never names a private host.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_DIR="${ROOT_DIR}/configs/optuna_search"
OUTPUT_DIR="${1:?usage: run_optuna_gemm_search.sh OUTPUT_DIR [task1.json,task2.json]}"
TASKS="${2:-ft2000_optuna_gemm_random_search_trainset_t1_task0.json,ft2000_optuna_gemm_tpe_search_trainset_t1_task0.json}"
mkdir -p "${OUTPUT_DIR}"
python "${ROOT_DIR}/model/optuna_search/do_optuna_gemm_tuning_tasks.py" \
  "${ROOT_DIR}" "${OUTPUT_DIR}" "${TASK_DIR}" "${TASKS}"
