# -*- coding: utf-8 -*-

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

try:
    from .tune_parameters_with_optuna import TuneParametersWithOptuna
except ImportError:  # Support the task runner's legacy top-level import.
    from tune_parameters_with_optuna import TuneParametersWithOptuna


def init_logs(log_dir: str):
    """初始化日志"""
    logging.basicConfig(
        filename=os.path.join(log_dir, "optuna_search.log"),
        level=logging.DEBUG,
        format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d]  %(message)s",
        encoding="utf-8",
    )


def run(work_dir: str | Path, task_name: str, optuna_configs: Dict[str, Any]) -> None:
    """运行调优过程 (方案F, 8分支独立精炼)"""
    init_logs(work_dir)

    logger = logging.getLogger(__name__).getChild(__name__)

    # Task JSON uses placeholders so credentials and hosts never enter the
    # release. Resolve them only from the caller's environment at runtime.
    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    optuna_configs = expand(optuna_configs)
    is_debug = optuna_configs.get("is_debug", False)
    if is_debug:
        ssh_env = []
    else:
        ssh_env = optuna_configs["ssh_env_list"]

    total_trials = optuna_configs["total_trails"]
    sampler_type = optuna_configs.get("sampler_type", "RandomSampler")
    sampler_seed = optuna_configs.get("sampler_seed", 42)

    optuna_database_uri = optuna_configs.get("optuna_database_uri", None)

    gemm_env_config = optuna_configs["gemm_env_config"]

    alpha = gemm_env_config["alpha"]
    beta = gemm_env_config["beta"]
    storage_layout = gemm_env_config["storage_layout"]
    element_precision = gemm_env_config["element_precision"]
    packing_types_space = gemm_env_config["packing_types"]
    loop_orders_space = gemm_env_config["loop_orders"]
    threads_space = [tuple(i) for i in gemm_env_config["threads"]]
    gemm_shape_list = [tuple(i) for i in gemm_env_config["gemm_shape_list"]]

    logger.info(
        f"即将使用以下参数进行矩阵乘法优化运行时参数优化: sampler_type={sampler_type}(sampler_seed={sampler_seed}), total_trials={total_trials}, optuna_database_uri={optuna_database_uri}, gemm_env_config={gemm_env_config}"
    )

    tunner = TuneParametersWithOptuna(
        is_debug=is_debug,
        work_dir=work_dir,
        study_base_name=task_name,
        alpha=alpha,
        beta=beta,
        storage_layout=storage_layout,
        element_precision=element_precision,
        packing_types_space=packing_types_space,
        loop_orders_space=loop_orders_space,
        threads_space=threads_space,
        ssh_env=ssh_env,
    )

    tune_results = tunner.run_complete_optimization(
        gemm_shape=gemm_shape_list,
        total_trials=total_trials,
        study_database_uri=optuna_database_uri,
        sampler_type=sampler_type,
        sampler_seed=sampler_seed,
    )

    result_json_path = (
        os.path.join(work_dir, "tune_results.json")
        if isinstance(work_dir, str)
        else work_dir.joinpath("tune_results.json")
    )
    with open(result_json_path, "w") as f:
        json.dump(tune_results, f)

    logger.info(f"搜索结束，优化结果已保存到 {result_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run one published Optuna task")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = json.loads(args.config.read_text(encoding="utf-8"))
    run(args.output_dir, task["name"], task["optuna_configs"])
