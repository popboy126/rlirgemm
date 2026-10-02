# -*- coding: utf-8 -*-
import logging
import math
import os
from random import randint
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
try:
    from .ssh_agent import SSHAgent
except ImportError:  # Support invocation by the task runner as a top-level module.
    from ssh_agent import SSHAgent


def setup_logger(log_file_path: str, logger_name: str):
    """
    配置日志记录器，包括日志文件路径、日志级别和格式。
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # 设置 logger 总体级别

    # 2. 避免重复添加 handler（多次调用同一 logger 时）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. 确保日志文件所在目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 4. 创建文件 handler，绑定日志文件路径
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # handler 级别可以单独控制

    # 5. 设置日志格式
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # 6. 将 handler 添加到 logger
    logger.addHandler(file_handler)
    return logger


class TuneParametersWithOptuna:
    """
    使用optuna进行参数调优
    """

    def __init__(
        self,
        is_debug: bool,
        work_dir: str,
        study_base_name: str,
        alpha: float,
        beta: float,
        storage_layout: int,
        element_precision: int,
        packing_types_space: List[int],
        loop_orders_space: List[int],
        threads_space: List[Tuple[int, int, int]],
        ssh_env: List[Dict[str, Any]],
    ):
        """
        初始化TuneParametersWithOptuna对象

        Args:
            is_debug (bool): 是否为调试模式
            study_base_name (str): 研究基础名称
            alpha (float): alpha参数
            beta (float): beta参数
            storage_layout (int): 存储布局参数
            element_precision (int): 元素精度参数
            packing_types_space (List[int]): 打包类型参数空间
            loop_orders_space (List[int]): 循环顺序参数空间
            threads_space (List[Tuple[int, int,int]]): 线程参数空间
            ssh_env (List[Dict]): SSH环境配置列表
        """
        self.is_debug = is_debug
        self.work_dir = work_dir
        self.study_base_name = study_base_name
        self.alpha = alpha
        self.beta = beta
        self.storage_layout = storage_layout
        self.element_precision = element_precision
        if element_precision == 0:
            self.element_byte_size = 4
        else:
            raise ValueError("element_precision must be 0")
        self.packing_types_space = packing_types_space
        self.loop_orders_space = loop_orders_space
        self.threads_space = threads_space
        self.ssh_env = ssh_env
        self.ssh_agent = []
        for idx, config in enumerate(ssh_env):
            ssh_config = config["ssh_config"]
            agent_name = f"gemm_agent_{idx}"
            agent_logger = setup_logger(
                os.path.join(self.work_dir, f"{agent_name}.log"), logger_name=agent_name
            )
            agent = SSHAgent(
                user=ssh_config["user"],
                ip=ssh_config["host"],
                port=ssh_config["port"],
                password=ssh_config["password"],
                gemm_agent_name=agent_name,
                logger=agent_logger,
            )
            self.ssh_agent.append(agent)
        self.logger = logging.getLogger(__name__)

    def do_agent_gemm(
        self,
        M: int,
        K: int,
        N: int,
        trail_id: int,
        loop_order: int,
        packing_type: int,
        thread: Tuple[int, int, int],
        mc: int,
        kc: int,
        nc: int,
        gemm_mat_a_prfm_stride: int,
        gemm_mat_b_prfm_stride: int,
        gemm_mat_c_prfm_stride: int,
        add_mat_c_prfm_stride: int,
        add_partial_sum_prfm_stride: int,
    ) -> float:
        """
        执行GEMM计算并返回性能分数
        """
        if self.is_debug:
            performance = [randint(0, 100) for _ in range(5)]
            agent_idx = -1
            bind_cores = "NULL"

        else:
            agent_idx = trail_id % len(self.ssh_agent)
            agent = self.ssh_agent[agent_idx]
            agent_config = self.ssh_env[agent_idx]
            agent_perf_config = agent_config["gemm_perf_config"]
            m_thread, k_thread, n_thread = thread
            ret, performance = agent.run_gemm(
                work_dir=agent_perf_config["work_dir"],
                bin_path=agent_perf_config["bin_path"],
                storage_layout=self.storage_layout,
                loop_order=loop_order,
                packing_type=packing_type,
                element_precision=self.element_precision,
                m_thread=m_thread,
                k_thread=k_thread,
                n_thread=n_thread,
                mc=mc,
                kc=kc,
                nc=nc,
                gemm_mat_a_prfm_stride=gemm_mat_a_prfm_stride,
                gemm_mat_b_prfm_stride=gemm_mat_b_prfm_stride,
                gemm_mat_c_prfm_stride=gemm_mat_c_prfm_stride,
                add_mat_c_prfm_stride=add_mat_c_prfm_stride,
                add_partial_sum_prfm_stride=add_partial_sum_prfm_stride,
                trail_repeat_times=agent_perf_config["trail_repeat"],
                loop_times=agent_perf_config["loop_times"],
                M=M,
                N=N,
                K=K,
                alpha=self.alpha,
                beta=self.beta,
                bind_cores=agent_perf_config["bind_cores"],
            )
            bind_cores = agent_perf_config["bind_cores"]

            if ret != 0:
                self.logger.warning(
                    f"Shape ({M}, {K}, {N}) in trail({trail_id}) with parameters loop_order={loop_order}, packing_type={packing_type}, threads={thread}, mc={mc}, kc={kc}, nc={nc}, gemm_mat_a_prfm_stride={gemm_mat_a_prfm_stride}, gemm_mat_b_prfm_stride={gemm_mat_b_prfm_stride}, gemm_mat_c_prfm_stride={gemm_mat_c_prfm_stride}, add_mat_c_prfm_stride={add_mat_c_prfm_stride}, add_partial_sum_prfm_stride={add_partial_sum_prfm_stride} run failed on agent({agent_idx}, cores: {bind_cores}) with return code {ret}"
                )
                performance_score = [0]

        performance_score = np.array(performance).mean().item()
        self.logger.debug(
            f"Shape ({M}, {K}, {N}) in trail({trail_id}) with parameters loop_order={loop_order}, packing_type={packing_type}, threads={thread}, mc={mc}, kc={kc}, nc={nc}, gemm_mat_a_prfm_stride={gemm_mat_a_prfm_stride}, gemm_mat_b_prfm_stride={gemm_mat_b_prfm_stride}, gemm_mat_c_prfm_stride={gemm_mat_c_prfm_stride}, add_mat_c_prfm_stride={add_mat_c_prfm_stride}, add_partial_sum_prfm_stride={add_partial_sum_prfm_stride} got mean performance score {performance_score}({performance}) on agent({agent_idx}, cores: {bind_cores})"
        )
        return performance_score

    def optimize_strategy(
        self,
        gemm_shape: Tuple[int, int, int],
        n_trials: int,
        n_jobs: int,
        study_database_uri: str = "",
        sampler_type: str = "TPESampler",
        sampler_seed: int = 42,
    ) -> Tuple[
        int,
        int,
        Tuple[int, int, int],
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        float,
        int,
        str,
    ]:
        """
        搜索矩阵形状gemm_shape能够取得最好性能的

        Returns:
            最优参数组合的元组(循环顺序，打包类型，线程类型，mc, kc, nc, 预取A的字节跨度，预取B的字节跨度，预取C的字节跨度，累加部分和时预取C的字节跨度，累加部分和时预取部分和的字节跨度, 性能分数, 最优trial的编号, 研究名称)
        """
        self.logger.info(
            "搜索矩阵形状 %s, 搜索次数 %r, 并行数 %r", gemm_shape, n_trials, n_jobs
        )

        if study_database_uri.startswith("postgresql+psycopg2://"):
            storage = optuna.storages.RDBStorage(
                study_database_uri,
                engine_kwargs={
                    "pool_size": 20,
                    "max_overflow": 10,
                    "pool_pre_ping": True,
                },
            )
        else:
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{study_database_uri}",
                engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
            )

        # 准备筛选范围数据
        M, K, N = gemm_shape
        study_name = f"{self.study_base_name}_{M}x{K}x{N}"
        loop_order_space = self.loop_orders_space
        packing_types_space = self.packing_types_space
        thread_type_space = (0, len(self.threads_space) - 1)
        mc_space = (1, M)
        kc_space = (1, K)
        nc_space = (1, N)
        prfm_a_skip_bytes_space = (
            0,
            M * K * self.element_byte_size,
        )
        prfm_b_skip_bytes_space = (
            0,
            K * N * self.element_byte_size,
        )
        prfm_c_skip_bytes_space = (
            0,
            M * N * self.element_byte_size,
        )

        prfm_add_c_skip_bytes_space = (
            0,
            M * N * self.element_byte_size,
        )

        prfm_add_partial_sum_skip_bytes_space = (
            0,
            M * N * self.element_byte_size,
        )

        def objective(trial):

            # 选择11个参数
            loop_order = trial.suggest_categorical("loop_order", loop_order_space)
            packing_type = trial.suggest_categorical(
                "packing_type", packing_types_space
            )
            thread_type_encode = trial.suggest_int(
                "thread_type_encode", thread_type_space[0], thread_type_space[1]
            )
            mc = trial.suggest_int("mc", mc_space[0], mc_space[1])
            kc = trial.suggest_int("kc", kc_space[0], kc_space[1])
            nc = trial.suggest_int("nc", nc_space[0], nc_space[1])
            prfm_a_skip_bytes = trial.suggest_int(
                "prfm_a_skip_bytes",
                prfm_a_skip_bytes_space[0],
                prfm_a_skip_bytes_space[1],
            )

            prfm_b_skip_bytes = trial.suggest_int(
                "prfm_b_skip_bytes",
                prfm_b_skip_bytes_space[0],
                prfm_b_skip_bytes_space[1],
            )

            prfm_c_skip_bytes = trial.suggest_int(
                "prfm_c_skip_bytes",
                prfm_c_skip_bytes_space[0],
                prfm_c_skip_bytes_space[1],
            )

            prfm_add_c_skip_bytes = trial.suggest_int(
                "prfm_add_c_skip_bytes",
                prfm_add_c_skip_bytes_space[0],
                prfm_add_c_skip_bytes_space[1],
            )

            prfm_add_partial_sum_skip_bytes = trial.suggest_int(
                "prfm_add_partial_sum_skip_bytes",
                prfm_add_partial_sum_skip_bytes_space[0],
                prfm_add_partial_sum_skip_bytes_space[1],
            )

            thread_type = self.threads_space[thread_type_encode]
            trial.set_user_attr("thread_type", thread_type)
            # 根据 trail_id选择agent
            trail_id = trial.number
            # 使用选定的agent执行任务
            score = self.do_agent_gemm(
                M=M,
                K=K,
                N=N,
                trail_id=trail_id,
                loop_order=loop_order,
                packing_type=packing_type,
                thread=thread_type,
                mc=mc,
                kc=kc,
                nc=nc,
                gemm_mat_a_prfm_stride=prfm_a_skip_bytes,
                gemm_mat_b_prfm_stride=prfm_b_skip_bytes,
                gemm_mat_c_prfm_stride=prfm_c_skip_bytes,
                add_mat_c_prfm_stride=prfm_add_c_skip_bytes,
                add_partial_sum_prfm_stride=prfm_add_partial_sum_skip_bytes,
            )
            return score

        try:
            self.study_search = optuna.load_study(
                study_name=study_name,
                storage=storage,
            )
        except (ValueError, KeyError):
            if sampler_type == "TPESampler":
                sampler = optuna.samplers.TPESampler(seed=sampler_seed)
            elif sampler_type == "RandomSampler":
                sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
            else:
                raise ValueError(f"Unsupported sampler type: {sampler_type}")

            self.study_search = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=sampler,
                storage=storage,
            )

        self.study_search.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        best_trial = self.study_search.best_trial
        best_params = best_trial.params
        required_param_names = (
            "loop_order",
            "packing_type",
            "thread_type_encode",
            "mc",
            "kc",
            "nc",
            "prfm_a_skip_bytes",
            "prfm_b_skip_bytes",
            "prfm_c_skip_bytes",
            "prfm_add_c_skip_bytes",
            "prfm_add_partial_sum_skip_bytes",
        )
        missing_param_names = [
            param_name
            for param_name in required_param_names
            if param_name not in best_params
        ]
        if missing_param_names:
            raise ValueError(
                f"Best trial({best_trial.number}) missing parameters: {missing_param_names}"
            )

        thread_type_encode = int(best_params["thread_type_encode"])
        if not 0 <= thread_type_encode < len(self.threads_space):
            raise ValueError(
                f"Best trial({best_trial.number}) has invalid thread_type_encode: {thread_type_encode}"
            )

        thread_type = tuple(
            int(thread) for thread in self.threads_space[thread_type_encode]
        )
        if len(thread_type) != 3:
            raise ValueError(
                f"Best trial({best_trial.number}) has invalid thread_type: {thread_type}"
            )

        best_score = float(self.study_search.best_value)

        return (
            int(best_params["loop_order"]),
            int(best_params["packing_type"]),
            thread_type,
            int(best_params["mc"]),
            int(best_params["kc"]),
            int(best_params["nc"]),
            int(best_params["prfm_a_skip_bytes"]),
            int(best_params["prfm_b_skip_bytes"]),
            int(best_params["prfm_c_skip_bytes"]),
            int(best_params["prfm_add_c_skip_bytes"]),
            int(best_params["prfm_add_partial_sum_skip_bytes"]),
            best_score,
            int(best_trial.number),
            study_name,
        )

    def run_complete_optimization(
        self,
        gemm_shape: List[Tuple[int, int, int]],
        total_trials: int,
        study_database_uri: str = "",
        sampler_type: str = "TPESampler",
        sampler_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        完成所有形状参数的搜索

        Returns:
            List[Dict[str, Any]]: 所有形状参数的搜索结果, 示例[{
                "M": M,
                "K": K,
                "N": N,
                "loop_order": 0,
                "packing_type": 1,
                "thread_type": (1, 1, 1),
                "mc": 1,
                "kc": 1,
                "nc": 1,
                "prfm_a_skip_bytes": 64,
                "prfm_b_skip_bytes": 64,
                "prfm_c_skip_bytes": 64,
                "prfm_add_c_skip_bytes": 64,
                "prfm_add_partial_sum_skip_bytes": 64,
                "performance": 10.0,
                "trail_id": 1,
                "study_name": "xxxxx"
            }]

        """
        if not study_database_uri.startswith("postgresql+psycopg2://"):
            study_database_uri = os.path.join(self.work_dir, study_database_uri)
        self.logger.info(
            f"开始矩阵乘法参数调优(采样器: {sampler_type}, 采样器种子: {sampler_seed})..."
        )
        results = list()

        if self.is_debug:
            n_trails = 1
            n_jobs = 1
        else:
            n_trails = math.ceil(total_trials / len(gemm_shape))
            n_jobs = len(self.ssh_agent)

        for shape_idx, shape in enumerate(gemm_shape):
            M, K, N = shape
            self.logger.info(
                "开始搜索第 %d/%d 个矩阵形状: %s",
                shape_idx + 1,
                len(gemm_shape),
                shape,
            )
            (
                loop_order,
                packing_type,
                thread_type,
                mc,
                kc,
                nc,
                prfm_a_skip_bytes,
                prfm_b_skip_bytes,
                prfm_c_skip_bytes,
                prfm_add_c_skip_bytes,
                prfm_add_partial_sum_skip_bytes,
                performance,
                trail_id,
                study_name,
            ) = self.optimize_strategy(
                gemm_shape=shape,
                n_trials=n_trails,
                n_jobs=n_jobs,
                study_database_uri=study_database_uri,
                sampler_type=sampler_type,
                sampler_seed=sampler_seed,
            )

            results.append(
                {
                    "M": M,
                    "K": K,
                    "N": N,
                    "loop_order": loop_order,
                    "packing_type": packing_type,
                    "thread_type": thread_type,
                    "mc": mc,
                    "kc": kc,
                    "nc": nc,
                    "prfm_a_skip_bytes": prfm_a_skip_bytes,
                    "prfm_b_skip_bytes": prfm_b_skip_bytes,
                    "prfm_c_skip_bytes": prfm_c_skip_bytes,
                    "prfm_add_c_skip_bytes": prfm_add_c_skip_bytes,
                    "prfm_add_partial_sum_skip_bytes": prfm_add_partial_sum_skip_bytes,
                    "performance": performance,
                    "trail_id": trail_id,
                    "study_name": study_name,
                }
            )

        self.logger.info(
            f"矩阵乘法参数调优(采样器: {sampler_type}, 采样器种子: {sampler_seed})完成"
        )
        return results
