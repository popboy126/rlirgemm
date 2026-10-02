# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/3/31 11:19
# Contact       : hucheng.lew@qq.com
# File          : two_step_gemm_env.py
# IDE           : PyCharm
# Description   : 两步矩阵乘法优化环境定义

import logging
import os
from copy import deepcopy
from logging.handlers import RotatingFileHandler
from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from action_space_transformer import ActionSpaceTransformer
from gemm_base_agent import GemmBaseAgent
from gemm_space import GEMMActionSpace, GEMMObservationSpace
from graviton2_gemm_agent import Graviton2GemmAgent
from KunPeng920GemmAgent import KunPeng920GemmAgent
from observation_space_transformer import ObservationSpaceTransformer
from reward_manager import RewardManager
from tianshou.env import BaseVectorEnv
from tianshou.highlevel.env import (
    ContinuousEnvironments,
    DiscreteEnvironments,
    EnvFactory,
    Environments,
    EnvMode,
    EnvType,
    VectorEnvType,
)


def get_reward_norm_range() -> Tuple[float, float]:
    """
    获取奖励归一化范围

    修复：放大奖励范围从 (-1, 1) 到 (-10, 10)
    原因：奖励信号过弱导致critic的TD error过小，进而导致梯度消失
    放大后奖励信号更强，能产生更大的TD error和有意义的梯度
    """
    return -1.0, 1.0


def get_gemm_agent(agent_type: str, **kwargs) -> GemmBaseAgent:
    """
    获取gemm执行代理
    """
    if agent_type == "KunPeng920":
        return KunPeng920GemmAgent(**kwargs)
    elif agent_type == "Graviton2":
        return Graviton2GemmAgent(**kwargs)
    else:
        raise ValueError(f"unknown gemm agent type: {agent_type}")


# --------------------------
# 1. 定义支持多初始配置的环境
# --------------------------
class GemmEnv(gym.Env):
    def __init__(self, env_name: str = "", env_mode: EnvMode = EnvMode.TEST, **kwargs):
        """
        初始化
        """
        self.env_name = env_name
        self.is_pse_inference = kwargs.get(
            "is_pse_inference", False
        )  # 是否为PSE推理模式
        self.logger = logging.getLogger(env_name)
        if ("log_config" in kwargs) and (self.is_pse_inference == False):  # 初始化日志
            log_config = kwargs["log_config"]
            self.init_logger(log_config)

        self.env_mode = env_mode

        self.shape_list, self.remained_shape_list = [], []
        self.min_matrix_bytes_size, self.max_matrix_bytes_size = 0, 0
        self.min_matrix_element, self.max_matrix_element = 0, 0
        self.min_m, self.max_m, self.m_bucket_width = 0, 0, 0
        self.min_k, self.max_k, self.k_bucket_width = 0, 0, 0
        self.min_n, self.max_n, self.n_bucket_width = 0, 0, 0
        self.mc_bucket_width, self.kc_bucket_width, self.nc_bucket_width = 0, 0, 0
        (
            self.min_prfm_a_skip_bytes,
            self.max_prfm_a_skip_bytes,
            self.prfm_a_skip_bytes_bucket_width,
        ) = 0, 0, 0
        (
            self.min_prfm_b_skip_bytes,
            self.max_prfm_b_skip_bytes,
            self.prfm_b_skip_bytes_bucket_width,
        ) = 0, 0, 0
        (
            self.min_prfm_c_skip_bytes,
            self.max_prfm_c_skip_bytes,
            self.prfm_c_skip_bytes_bucket_width,
        ) = 0, 0, 0
        (
            self.min_prfm_add_c_skip_bytes,
            self.max_prfm_add_c_skip_bytes,
            self.prfm_add_c_skip_bytes_bucket_width,
        ) = 0, 0, 0
        (
            self.min_prfm_add_partial_sum_skip_bytes,
            self.max_prfm_add_partial_sum_skip_bytes,
            self.prfm_add_partial_sum_skip_bytes_bucket_width,
        ) = 0, 0, 0

        self.kwargs = kwargs  # 初始配置池
        self.observation_space = (
            GEMMObservationSpace.get_gym_space()
        )  # 观测空间: m_norm, k_norm, n_norm (桶索引， 桶用偏移)

        self.loop_orders = kwargs["loop_orders"]  # 循环顺序数
        self.packing_types = kwargs["packing_types"]  # 打包类型数
        self.threads = kwargs["threads"]  # 最大线程数

        self.gemm_action_space = GEMMActionSpace(
            loop_order_type=self.loop_orders,
            packing_type=self.packing_types,
            thread_type=self.threads,
        )
        self.action_space = self.gemm_action_space.get_gym_space()  # loop_order, (tm, tk, tn), packing_type, mc, kc, nc, prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes

        self.alpha, self.beta = (
            kwargs.get("alpha", 1.0),
            kwargs.get("beta", 0.0),
        )  # 缩放因子
        self.storage_layout = kwargs.get("storage_layout", 0)  # 存储布局，0表示NNN
        self.element_precision = kwargs.get(
            "element_precision", 0
        )  # 元素精度，0表示FP32

        self.matrix_element_size = kwargs["matrix_element_size"]  # 矩阵元素大小

        if kwargs["gemm_config_type"] == "gemm_shape_gen_config":
            self.parse_gemm_shape_gen_config(kwargs["gemm_shape_gen_config"])
            self.gemm_config_type = "gemm_shape_gen_config"
        elif kwargs["gemm_config_type"] == "gemm_shape_config":
            self.parse_gemm_shape_config(kwargs["gemm_shape_config"])
            self.gemm_config_type = "gemm_shape_config"
        else:
            raise ValueError(
                "gemm_config_type must be gemm_shape_gen_config or gemm_shape_config"
            )

        self._observation_space_normalizer = ObservationSpaceTransformer(
            dim_m=(self.min_m, self.max_m),
            dim_k=(self.min_k, self.max_k),
            dim_n=(self.min_n, self.max_n),
        )

        self._action_space_transformer = ActionSpaceTransformer(
            loop_orders=self.loop_orders,
            packing_types=self.packing_types,
            thread_types=self.threads,
            matrix_element_size=self.matrix_element_size,
            action_space=self.gemm_action_space,
        )

        self.is_debug = kwargs.get("is_debug", False)

        self.M, self.K, self.N = 1, 4096, 64  # 当前矩阵维度
        self._np_rng = self.np_random  # 随机数生成器

        self._reward_min, self._reward_max = (
            kwargs.get("reward_min", -100),
            kwargs.get("reward_max", 100),
        )  # 奖励范围
        self._reward_norm_min, self._reward_norm_max = (
            get_reward_norm_range()
        )  # 归一化奖励范围
        self._theoretical_single_core_performance = kwargs.get(
            "theoretical_single_core_performance", 50
        )

        # 动态采样配置参数
        self._dynamic_sampling_config = kwargs.get("dynamic_sampling_config", {})
        self._dynamic_sampling_mode = self._dynamic_sampling_config.get(
            "sampling_mode", "adaptive_repeat"
        )
        if self._dynamic_sampling_mode not in {"adaptive_repeat", "uniform_random"}:
            raise ValueError(
                "dynamic_sampling_config.sampling_mode must be "
                "'adaptive_repeat' or 'uniform_random'"
            )
        self._dynamic_sampling_enable = self._dynamic_sampling_config.get(
            "enable", True
        )
        self._dynamic_sampling_gap_thresholds = self._dynamic_sampling_config.get(
            "gap_thresholds", [0.05, 0.1, 0.2]
        )
        self._dynamic_sampling_repeat_factors = self._dynamic_sampling_config.get(
            "repeat_factors", [1, 2, 3, 5]
        )
        self._dynamic_sampling_new_shape_repeat = self._dynamic_sampling_config.get(
            "new_shape_repeat", 10
        )
        self._dynamic_sampling_max_repeat_times = self._dynamic_sampling_config.get(
            "max_repeat_times", 20
        )
        self._dynamic_sampling_use_adaptive = self._dynamic_sampling_config.get(
            "use_adaptive", True
        )
        self._dynamic_sampling_complexity_weight = self._dynamic_sampling_config.get(
            "complexity_weight", 0.3
        )
        self._reward_break_max, self._reward_break_k, self._reward_break_threshold = (
            kwargs.get("reward_break_max", 1.0),
            kwargs.get("reward_break_k", 15),
            kwargs.get("reward_break_threshold", 0.05),
        )

        self._reward_manager = RewardManager(
            name=f"RM-{self.env_name}",
            loop_orders=self.loop_orders,
            packing_types=self.packing_types,
            thread_types=self.threads,
            action_space_transformer=self._action_space_transformer,
            shape_perf_cache_shape_list=kwargs["gemm_shape_config"]["shape_list"],
            shape_perf_cache_shm_name=kwargs["cache_manager"]["shape_perf_shm_name"],
            shape_perf_cache_lock=kwargs["cache_manager"]["shape_perf_lock"],
            action_stat_cache_shm_name=kwargs["cache_manager"]["action_stat_shm_name"],
            action_stat_cache_lock=kwargs["cache_manager"]["action_stat_lock"],
            element_byte_size=self.matrix_element_size,
            reward_min=self._reward_min,
            reward_max=self._reward_max,
            reward_norm_min=self._reward_norm_min,
            reward_norm_max=self._reward_norm_max,
            reward_break_max=self._reward_break_max,
            reward_break_k=self._reward_break_k,
            reward_break_threshold=self._reward_break_threshold,
            performance_ema_alpha=kwargs.get("performance_ema_alpha", 0.3),
            reward_perf_exploration_factor=kwargs.get(
                "reward_perf_exploration_factor", 1
            ),
            reward_perf_absolute_factor=kwargs.get("reward_perf_absolute_factor", 1),
            reward_perf_ema_factor=kwargs.get("reward_perf_ema_factor", 1),
            reward_perf_breakthrough_factor=kwargs.get(
                "reward_perf_breakthrough_factor", 1
            ),
            reward_perf_ema_penalty_threshold=kwargs.get(
                "reward_perf_ema_penalty_threshold", 0.5
            ),
            reward_perf_ema_penalty_factor=kwargs.get(
                "reward_perf_ema_penalty_factor", 3
            ),
            reward_mode=kwargs.get("reward_mode", "multi_level"),
            reward_shape_best_ratio_threshold=kwargs.get(
                "reward_shape_best_ratio_threshold", 0.8
            ),
            logger=self.logger,
        )

        use_real_agent = kwargs.get("use_real_agent", True)
        agent_type = kwargs.get("agent_type", "KunPeng920")
        if use_real_agent:
            self.agent = get_gemm_agent(
                agent_type,
                user=kwargs["ssh_config"]["user"],
                ip=kwargs["ssh_config"]["host"],
                port=kwargs["ssh_config"]["port"],
                password=kwargs["ssh_config"]["password"],
                agent_name=env_name,
                logger=self.logger,
            )
            self.logger.info(f"gemm agent with type {agent_type} is created!")
            self.bind_cores = kwargs["gemm_perf_config"].get("bind_cores", "0-63")
        else:
            self.logger.warning(
                f"GemmEnv is running in debug mode without a real gemm agent with type {agent_type}."
            )
            self.agent = None
            self.bind_cores = ""

        self.logger.info(
            f"gemm env({self.env_name}) initialized with parameters: gemm_config_type: {self.gemm_config_type}, min_m: {self.min_m}, max_m: {self.max_m}, m_bucket_num: {self.m_bucket_width}, "
            f" min_k: {self.min_k}, max_k: {self.max_k}, "
            f"min_n: {self.min_n}, max_n: {self.max_n}, "
            f"reward_break_max: {self._reward_break_max}, reward_break_k: {self._reward_break_k}, reward_break_threshold: {self._reward_break_threshold}, "
            f"reward_min: {self._reward_min}, reward_max: {self._reward_max}, reward_norm_min: {self._reward_norm_min}, reward_norm_max: {self._reward_norm_max}, "
            f"history_queue_size: {kwargs.get('history_queue_size')}, history_performance_dict_size: {kwargs.get('history_performance_dict_size')}, "
            f"loop_orders: {self.loop_orders}, packing_types: {self.packing_types}, threads: {self.threads}, "
            f"min_prfm_a_skip_bytes: {self.min_prfm_a_skip_bytes}, max_prfm_a_skip_bytes: {self.max_prfm_a_skip_bytes}, "
            f"min_prfm_b_skip_bytes: {self.min_prfm_b_skip_bytes}, max_prfm_b_skip_bytes: {self.max_prfm_b_skip_bytes}, "
            f"min_prfm_c_skip_bytes: {self.min_prfm_c_skip_bytes}, max_prfm_c_skip_bytes: {self.max_prfm_c_skip_bytes}, "
            f"min_prfm_add_c_skip_bytes: {self.min_prfm_add_c_skip_bytes}, max_prfm_add_c_skip_bytes: {self.max_prfm_add_c_skip_bytes}, "
            f"min_prfm_add_partial_sum_skip_bytes: {self.min_prfm_add_partial_sum_skip_bytes}, max_prfm_add_partial_sum_skip_bytes: {self.max_prfm_add_partial_sum_skip_bytes}, "
            f"min_matrix_element_size: {self.min_matrix_element}, max_matrix_element: {self.max_matrix_element}, "
            f"bind_cores: {self.bind_cores}, "
            f"dynamic_sampling_enable: {self._dynamic_sampling_enable}, dynamic_sampling_use_adaptive: {self._dynamic_sampling_use_adaptive}, "
            f"dynamic_sampling_max_repeat_times: {self._dynamic_sampling_max_repeat_times}, dynamic_sampling_complexity_weight: {self._dynamic_sampling_complexity_weight}, "
            f"dynamic_sampling_gap_thresholds: {self._dynamic_sampling_gap_thresholds}, dynamic_sampling_repeat_factors: {self._dynamic_sampling_repeat_factors}, "
            f"dynamic_sampling_new_shape_repeat: {self._dynamic_sampling_new_shape_repeat}, "
            f"dynamic_sampling_mode: {self._dynamic_sampling_mode}, "
            f"reward_mode: {kwargs.get('reward_mode', 'multi_level')}"
        )

    def _build_uniform_training_shape_list(self) -> List[Tuple[int, int, int]]:
        """构建均匀随机消融模式的一轮训练 shape 列表。"""
        shape_list = list(self.shape_list)
        self.np_random.shuffle(shape_list)
        return shape_list

    def get_loop_order_size(self) -> int:
        """
        获取循环顺序数量
        """
        return len(self.loop_orders)

    def get_packing_type_size(self) -> int:
        """
        获取打包类型数量
        """
        return len(self.packing_types)

    def get_thread_type_size(self) -> int:
        """
        获取线程类型数量
        """
        return len(self.threads)

    def get_continuous_action_dim(self) -> int:
        """
        获取连续动作维度
        """
        return self.action_space.shape[0] - 3

    def parse_gemm_shape_gen_config(self, config: Dict[str, Any]) -> None:
        """
        解析矩阵形状生成配置
        """
        self.min_matrix_bytes_size = config["min_matrix_bytes_size"]  # 最小矩阵字节数
        self.max_matrix_bytes_size = config["max_matrix_bytes_size"]  # 最大矩阵字节数
        self.max_matrix_element = (
            self.max_matrix_bytes_size / self.matrix_element_size
        )  # 三个矩阵最多元素数
        self.min_matrix_element = (
            self.min_matrix_bytes_size / self.matrix_element_size
        )  # 三个矩阵最少元素数
        # 维度空间范围
        self.min_m, self.max_m = config["min_m"], config["max_m"]
        self.min_k, self.max_k = config["min_k"], config["max_k"]
        self.min_n, self.max_n = config["min_n"], config["max_n"]

        # 预取跳过字节数范围
        self.min_prfm_a_skip_bytes, self.max_prfm_a_skip_bytes = (
            config["min_prfm_a_skip_bytes"],
            config["max_prfm_a_skip_bytes"],
        )
        self.min_prfm_b_skip_bytes, self.max_prfm_b_skip_bytes = (
            config["min_prfm_b_skip_bytes"],
            config["max_prfm_b_skip_bytes"],
        )
        self.min_prfm_c_skip_bytes, self.max_prfm_c_skip_bytes = (
            config["min_prfm_c_skip_bytes"],
            config["max_prfm_c_skip_bytes"],
        )
        self.min_prfm_add_c_skip_bytes, self.max_prfm_add_c_skip_bytes = (
            config["min_prfm_add_c_skip_bytes"],
            config["max_prfm_add_c_skip_bytes"],
        )
        (
            self.min_prfm_add_partial_sum_skip_bytes,
            self.max_prfm_add_partial_sum_skip_bytes,
        ) = (
            config["min_prfm_add_partial_sum_skip_bytes"],
            config["max_prfm_add_partial_sum_skip_bytes"],
        )

    def parse_gemm_shape_config(self, config: Dict[str, Any]) -> None:
        """
        解析矩阵形状配置
        """
        self.shape_list = [tuple(shape) for shape in config["shape_list"]]
        m_list = [shape[0] for shape in self.shape_list]
        self.min_m, self.max_m = min(m_list), max(m_list)
        k_list = [shape[1] for shape in self.shape_list]
        self.min_k, self.max_k = min(k_list), max(k_list)
        n_list = [shape[2] for shape in self.shape_list]
        self.min_n, self.max_n = min(n_list), max(n_list)
        # 预取跳过字节数范围
        matrix_a_bytes_size = [
            shape[0] * shape[1] * self.matrix_element_size for shape in self.shape_list
        ]
        self.min_prfm_a_skip_bytes, self.max_prfm_a_skip_bytes = (
            0,
            max(matrix_a_bytes_size),
        )
        matrix_b_bytes_size = [
            shape[1] * shape[2] * self.matrix_element_size for shape in self.shape_list
        ]
        self.min_prfm_b_skip_bytes, self.max_prfm_b_skip_bytes = (
            0,
            max(matrix_b_bytes_size),
        )
        matrix_c_bytes_size = [
            shape[0] * shape[2] * self.matrix_element_size for shape in self.shape_list
        ]
        self.min_prfm_c_skip_bytes, self.max_prfm_c_skip_bytes = (
            0,
            max(matrix_c_bytes_size),
        )
        self.min_prfm_add_c_skip_bytes, self.max_prfm_add_c_skip_bytes = (
            0,
            max(matrix_c_bytes_size),
        )
        (
            self.min_prfm_add_partial_sum_skip_bytes,
            self.max_prfm_add_partial_sum_skip_bytes,
        ) = 0, max(matrix_c_bytes_size)
        matrix_total_element_size = [
            shape[0] * shape[1] + shape[1] * shape[2] + shape[0] * shape[2]
            for shape in self.shape_list
        ]
        self.min_matrix_element, self.max_matrix_element = (
            min(matrix_total_element_size),
            max(matrix_total_element_size),
        )

    def init_logger(self, log_config: Dict[str, Any]) -> None:
        """
        初始化日志记录器
        """
        self.logger.setLevel(logging.DEBUG)
        # 创建Formatter对象
        formatter = logging.Formatter(
            "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d]  %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False  # 不传递给全局

        if "log_dir" in log_config:
            log_dir: str = log_config["log_dir"]
            log_file_name = log_config.get("env_log_file_name", f"{self.env_name}.log")
            max_bytes = log_config.get("max_bytes", 256 * 1024 * 1024)  # 默认256MB
            backup_count = log_config.get("backup_count", 4)  # 默认保留4个备份日志文件
            # 创建FileHandler对象
            fh = RotatingFileHandler(
                os.path.join(log_dir, log_file_name),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            # 将FileHandler对象添加到Logger对象中
            self.logger.addHandler(fh)

    def set_gemm_shape(self, m: int, k: int, n: int) -> None:
        """
        设置矩阵乘法形状
        """
        self.M, self.K, self.N = m, k, n

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[gym.core.ObsType, dict[str, Any]]:
        """
        重置环境状态
        """

        def gen_matrices():
            """
            生成矩阵维度
            """
            if self.gemm_config_type == "gemm_shape_gen_config":
                dim_m_tmp = floor(
                    (self.max_matrix_element - self.min_k * self.min_n)
                    / (self.min_k + self.min_n)
                )
                max_m = min(self.max_m, dim_m_tmp)
                m = self._np_rng.choice(
                    np.arange(self.min_m, max_m + 1, dtype=int), size=1
                ).item()

                dim_k_tmp = floor(
                    (self.max_matrix_element - m * self.min_n) / (m + self.min_n)
                )
                max_k = min(self.max_k, dim_k_tmp)
                k = self._np_rng.choice(
                    np.arange(self.min_k, max_k + 1, dtype=int), size=1
                ).item()

                dim_n_tmp = floor((self.max_matrix_element - m * k) / (m + k))
                max_n = min(self.max_n, dim_n_tmp)
                n = self._np_rng.choice(
                    np.arange(self.min_n, max_n + 1, dtype=int), size=1
                ).item()
                if m == 0 or k == 0 or n == 0:
                    raise ValueError("m, k, and n must not be 0")
                return m, k, n
            elif self.gemm_config_type == "gemm_shape_config":
                if len(self.remained_shape_list) == 0:
                    if self.env_mode.value == EnvMode.TRAINING.value:
                        if self._dynamic_sampling_mode == "uniform_random":
                            # 消融模式：每轮每个 shape 恰好一次，使用环境 RNG 保证可复现。
                            self.remained_shape_list = self._build_uniform_training_shape_list()
                            shape = self.remained_shape_list.pop()
                            return shape[0], shape[1], shape[2]
                        if self._dynamic_sampling_mode != "adaptive_repeat":
                            # 构造函数已校验模式；保留守卫避免未来新增模式误用自适应逻辑。
                            raise ValueError(
                                "unsupported dynamic sampling mode: "
                                f"{self._dynamic_sampling_mode}"
                            )
                        # 训练环境：根据收敛进度动态调整形状采样频率
                        # 未收敛的形状会被重复加入列表，从而获得更多训练机会
                        new_shape_list = []

                        for shape in self.shape_list:
                            # 获取该形状的历史性能数据
                            ema_perf = self._reward_manager.history_performance_dict.get_ema_performance(
                                shape
                            )
                            best_perf = self._reward_manager.history_performance_dict.get_shape_best_performance(
                                shape
                            )

                            # 根据性能差距决定重复训练次数
                            repeat_times = 1  # 默认至少训练一次

                            if ema_perf is None or best_perf is None:
                                # 从未训练过的形状
                                if self._dynamic_sampling_use_adaptive:
                                    # 自适应模式：根据形状复杂度估计初始重复次数
                                    m, k, n = shape
                                    complexity = m * k * n
                                    # 使用对数尺度映射到合理范围
                                    log_complexity = np.log10(complexity + 1)
                                    repeat_times = max(
                                        5,
                                        min(
                                            self._dynamic_sampling_max_repeat_times,
                                            ceil(
                                                log_complexity
                                                * self._dynamic_sampling_complexity_weight
                                                * 10
                                            ),
                                        ),
                                    )
                                else:
                                    # 固定模式：使用配置的初始重复次数
                                    repeat_times = (
                                        self._dynamic_sampling_new_shape_repeat
                                    )
                            else:
                                # 计算当前EMA性能与最佳性能的差距比例
                                # 差距越大说明越未收敛，需要更多训练
                                gap_ratio = (best_perf - ema_perf) / (best_perf + 1e-6)

                                if self._dynamic_sampling_use_adaptive:
                                    # 自适应模式：使用连续函数映射
                                    # gap_ratio 范围 [0, 1+]，映射到 [1, max_repeat_times]
                                    repeat_times = max(
                                        1,
                                        min(
                                            self._dynamic_sampling_max_repeat_times,
                                            int(
                                                self._dynamic_sampling_max_repeat_times
                                                * gap_ratio
                                            ),
                                        ),
                                    )
                                else:
                                    # 固定模式：使用配置的阈值和重复因子
                                    # 找到对应的重复因子（从高到低匹配）
                                    matched = False
                                    for i, threshold in enumerate(
                                        reversed(self._dynamic_sampling_gap_thresholds)
                                    ):
                                        if gap_ratio > threshold:
                                            repeat_times = (
                                                self._dynamic_sampling_repeat_factors[
                                                    -(i + 1)
                                                ]
                                            )
                                            matched = True
                                            break
                                    if not matched:
                                        repeat_times = (
                                            self._dynamic_sampling_repeat_factors[0]
                                        )

                            # 记录日志，便于调试和分析
                            self.logger.debug(
                                f"[DynamicSampling] shape={shape}, ema_perf={ema_perf}, "
                                f"best_perf={best_perf}, repeat_times={repeat_times}"
                            )

                            # 将形状重复加入列表
                            for _ in range(repeat_times):
                                new_shape_list.append(shape)

                        self.remained_shape_list = new_shape_list
                        self.np_random.shuffle(self.remained_shape_list)
                    else:
                        # 测试环境：使用固定采样顺序（简单轮询）
                        new_shape_list = deepcopy(self.shape_list)
                        self.np_random.shuffle(new_shape_list)
                        self.remained_shape_list = new_shape_list

                shape = self.remained_shape_list.pop()
                return shape[0], shape[1], shape[2]
            else:
                raise ValueError(
                    "gemm_config_type must be gemm_shape_gen_config or gemm_shape_config"
                )

        if not self.is_pse_inference:
            self.M, self.K, self.N = gen_matrices()  # 随机生成矩阵维度
            total_element_num = self.M * self.K + self.K * self.N + self.M * self.N
            while (
                (self.M <= 0)
                or (self.K <= 0)
                or (self.N <= 0)
                or (total_element_num > self.max_matrix_element)
            ):
                self.M, self.K, self.N = gen_matrices()  # 随机生成矩阵维度
                total_element_num = self.M * self.K + self.K * self.N + self.M * self.N

        self.logger.info(
            f"Resetting environment with parameters: M: {self.M}, K: {self.K}, N: {self.N}"
        )
        return self.get_shape_observation(), {}

    def get_shape_observation(self) -> np.ndarray:
        """
        获取当前矩阵形状的观测
        """
        m_norm, k_norm, n_norm = self._observation_space_normalizer.normalize(
            self.M, self.K, self.N
        )
        return np.array(
            [
                m_norm,
                k_norm,
                n_norm,
            ]
        )

    def decoding_action(
        self, action: gym.core.ActType
    ) -> Tuple[
        int,
        int,
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int],
    ]:
        """
        解码动作
        根据loop_order值处理无意义的维度：
        - loop_order=0: mc, kc, nc, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes 无意义
        - loop_order=1或2: kc, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes 无意义
        - 其他情况: 所有维度都有意义
        """
        loop_order = self.loop_orders[int(action[0].item())]
        tm, tk, tn = self.threads[int(action[1].item())]
        packing_type = self.packing_types[int(action[2].item())]
        (
            mc,
            kc,
            nc,
            prfm_a_skip_bytes,
            prfm_b_skip_bytes,
            prfm_c_skip_bytes,
            prfm_add_c_skip_bytes,
            prfm_add_partial_sum_skip_bytes,
        ) = self._action_space_transformer.denormalize(
            m=self.M,
            k=self.K,
            n=self.N,
            mc_norm=action[3],
            kc_norm=action[4],
            nc_norm=action[5],
            prfm_a_skip_bytes_norm=action[6],
            prfm_b_skip_bytes_norm=action[7],
            prfm_c_skip_bytes_norm=action[8],
            prfm_add_c_skip_bytes_norm=action[9],
            prfm_add_partial_sum_skip_bytes_norm=action[10],
        )

        # 根据loop_order值将无意义维度设置为默认值
        if loop_order == 0:
            # loop_order=0时，mc, kc, nc, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes 无意义
            # 设置blocking参数为1（最小有效值），预取参数为0
            mc, kc, nc = 1, 1, 1
            prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes = 0, 0
        elif loop_order in [1, 2]:
            # loop_order=1或2时，kc, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes 无意义
            # 设置kc为1，预取参数为0
            kc = 1
            prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes = 0, 0

        return (
            loop_order,
            packing_type,
            (tm, tk, tn),
            (mc, kc, nc),
            (prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes),
            (prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes),
        )

    def step(
        self, action: gym.core.ActType
    ) -> tuple[gym.core.ObsType, gym.core.SupportsFloat, bool, bool, dict[str, Any]]:
        """
        进行步进分析
        """
        # 解析动作

        (
            loop_order,
            packing_type,
            (tm, tk, tn),
            (mc, kc, nc),
            (prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes),
            (prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes),
        ) = self.decoding_action(action)

        reward, performance = self.execute(
            loop_order,
            packing_type,
            (tm, tk, tn),
            (mc, kc, nc),
            (prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes),
            (prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes),
        )
        return (
            self.get_shape_observation(),
            reward,
            True,
            False,
            {
                "storage_layout": self.storage_layout,
                "element_precision": self.element_precision,
                "loop_order": loop_order,
                "packing_type": packing_type,
                "threads": (tm, tk, tn),
                "blocks": (mc, kc, nc),
                "prfm_mat": (
                    prfm_a_skip_bytes,
                    prfm_b_skip_bytes,
                    prfm_add_c_skip_bytes,
                ),
                "prfm_add": (prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes),
                "reward": reward,
                "performance": performance,
                "action": action,
                "matrix_size": (self.M, self.K, self.N),
            },
        )

    def execute(
        self,
        loop_order: int,
        packing_type: int,
        threads: Tuple[int, int, int],
        blocks: Tuple[int, int, int],
        prfm_mat: Tuple[int, int, int],
        prfm_add: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        执行矩阵乘法
        """
        tm, tk, tn = threads
        mc, kc, nc = blocks
        prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes = prfm_mat
        prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes = prfm_add
        check_reward, performance = 0.0, 0.0

        check_reward = self._reward_manager.get_params_check_reward(
            gemm_shape=(self.M, self.K, self.N),
            loop_order=loop_order,
            packing_type=packing_type,
            thread=(tm, tk, tn),
            blocking=(mc, kc, nc),
            gemm_prfm=(prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes),
            add_prfm=(prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes),
        )
        if check_reward < 0:
            self.logger.warning(
                f"Invalid parameters with check_reward({check_reward}) for (M: {self.M}, K: {self.K}, N: {self.N}): loop_order={loop_order}, packing_type={packing_type}, threads={threads}, blocks={blocks}, prfm_mat={prfm_mat}, prfm_add={prfm_add}."
            )
            mean_performance = -1
        else:
            if self.is_debug:
                low, high = 1, 20
                performance = np.random.uniform(
                    low, high, (self.kwargs["gemm_perf_config"]["trail_repeat"], 1)
                ).tolist()
            elif (self.is_pse_inference) or (self.agent is None):
                performance = [0]
            else:
                # 计算运行时间
                ret, performance = self.agent.run_gemm(
                    work_dir=self.kwargs["gemm_perf_config"]["work_dir"],
                    bin_path=self.kwargs["gemm_perf_config"]["bin_path"],
                    storage_layout=self.storage_layout,
                    loop_order=loop_order,
                    packing_type=packing_type,
                    element_precision=self.element_precision,
                    m_thread=tm,
                    k_thread=tk,
                    n_thread=tn,
                    mc=mc,
                    kc=kc,
                    nc=nc,
                    gemm_mat_a_prfm_stride=prfm_a_skip_bytes,
                    gemm_mat_b_prfm_stride=prfm_b_skip_bytes,
                    gemm_mat_c_prfm_stride=prfm_c_skip_bytes,
                    add_mat_c_prfm_stride=prfm_add_c_skip_bytes,
                    add_partial_sum_prfm_stride=prfm_add_partial_sum_skip_bytes,
                    trail_repeat_times=self.kwargs["gemm_perf_config"]["trail_repeat"],
                    loop_times=self.kwargs["gemm_perf_config"]["loop_times"],
                    M=self.M,
                    K=self.K,
                    N=self.N,
                    alpha=self.alpha,
                    beta=self.beta,
                    bind_cores=self.bind_cores,
                )

                if ret != 0:
                    performance = [0]
                    self.logger.error(
                        f"GEMM with parameters (M: {self.M}, K: {self.K}, N: {self.N}, loop_order: {loop_order}, packing_type={packing_type}, (tm, tk, tn): {threads}, (mc, kc, nc): {blocks}, (prfm_a_sip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes): {prfm_mat}, (prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes): {prfm_add}) run failed with return code: {ret}"
                    )
            mean_performance = np.array(performance).mean().item()
        final_reward = self._reward_manager.compute_final_reward(
            check_reward,
            mean_performance,
            loop_order=loop_order,
            packing_type=packing_type,
            thread_type=threads,
            M=self.M,
            K=self.K,
            N=self.N,
        )
        self.logger.info(
            f"GEMM with parameters (M: {self.M}, K: {self.K}, N: {self.N}, loop_order: {loop_order}, packing_type={packing_type}, (tm, tk, tn): {threads}, (mc, kc, nc): {blocks}, (prfm_a_sip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes): {prfm_mat}, (prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes): {prfm_add}) has the average performance of {mean_performance}({performance}) GFlops on cores {self.bind_cores} and got final reward {final_reward}."
        )

        if self.env_mode.value == EnvMode.TRAINING.value:  # 只有训练环境才更新
            self._reward_manager.history_queue.add_sample(
                loop_order, packing_type, threads, final_reward, mean_performance
            )  # 记录历史奖励，待用
            if mean_performance > 0:
                self._reward_manager.history_performance_dict.update(
                    (self.M, self.K, self.N), mean_performance
                )
        return final_reward, mean_performance  # 返回运行性能，最小为10ms


import unittest


class TestGemmEnv(unittest.TestCase):
    def setUp(self):
        self.env = GemmEnv(
            env_name="test",
            env_mode=EnvMode.TRAINING,
            is_debug=True,
            use_real_agent=False,
            is_pse_inference=False,
            min_m=64,
            max_m=512,
            m_step=64,
            m_bucket_num=8,
            min_k=64,
            max_k=512,
            k_step=64,
            k_bucket_num=8,
            min_n=64,
            max_n=512,
            n_step=64,
            n_bucket_num=8,
            min_matrix_bytes_size=1024,
            max_matrix_bytes_size=1048576,
            min_prfm_a_skip_bytes=0,
            max_prfm_a_skip_bytes=1024,
            prfm_a_skip_bytes_step=64,
            prfm_a_skip_bytes_bucket_num=17,
            min_prfm_b_skip_bytes=0,
            max_prfm_b_skip_bytes=1024,
            prfm_b_skip_bytes_step=64,
            prfm_b_skip_bytes_bucket_num=17,
            min_prfm_c_skip_bytes=0,
            max_prfm_c_skip_bytes=1024,
            prfm_c_skip_bytes_step=64,
            prfm_c_skip_bytes_bucket_num=17,
            min_prfm_add_c_skip_bytes=0,
            max_prfm_add_c_skip_bytes=1024,
            prfm_add_c_skip_bytes_step=64,
            prfm_add_c_skip_bytes_bucket_num=17,
            min_prfm_add_partial_sum_skip_bytes=0,
            max_prfm_add_partial_sum_skip_bytes=1024,
            prfm_add_partial_sum_skip_bytes_step=64,
            prfm_add_partial_sum_skip_bytes_bucket_num=17,
            loop_orders=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            threads=[1, 2, 4, 8, 16, 32, 64],
            matrix_element_size=4,
            reward_min=-200,
            reward_max=200,
            reward_penalty_coef=0.1,
            log_config={
                "log_dir": os.path.join(os.path.dirname(__file__), "model", "log"),
                "train_env_log_file_name": "m707-kp920-env0-train.log",
                "test_env_log_file_name": "m707-kp920-env0-test.log",
                "train_env_agent_log_file_name": "m707-kp920-env0-agent-train.log",
                "test_env_agent_log_file_name": "m707-kp920-env0-agent-test.log",
            },
        )

    def test_check_thread_dim_blocking(self):
        """
        测试检查单个维度的线程数和分块大小是否合理
        """
        reward = self.env.check_thread_dim_blocking(5, 4, 15, True)
        self.assertLess(
            reward, 0, "Expected negative reward for unreasonable blocking size"
        )

    def test_check_params(self):
        """
        测试检查参数是否合理
        """
        self.env.set_gemm_shape(80, 5520, 272)
        reward = self.env.check_params(
            0, 1, 1, 3, 110, 58432, 184352, 6345920, 6538816, 7927104, 5662592, 431936
        )
        self.assertLess(
            reward, 1e-10, "Reward must be less than 1 for valid parameters"
        )

    def test_execute_reward_scale(self):
        """
        测试检查参数是否合理
        """
        reward = self.env.execute_reward_scale(40)
        self.assertLess(reward, 1e-10)


class GEMMEnvFactory(EnvFactory):
    """
    创建GEMM环境的工厂类
    """

    def __init__(
        self,
        task: str,
        venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
        **make_kwargs: Any,
    ) -> None:
        super().__init__(venv_type=venv_type)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.make_kwargs = make_kwargs
        self._train_env_configs, self._test_env_configs = self.get_env_configs()

    def get_env_configs(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Gets the environment configurations for training and test envs.

        :return: a tuple of (training env configs, test env configs)
        """
        env_list = self.make_kwargs.get("env_list", [])
        self.logger.info(f"Total {len(env_list)} environment configurations found.")
        train_env_configs = [
            config for config in env_list if config.get("env_type") == "train"
        ]
        test_env_configs = [
            config for config in env_list if config.get("env_type") == "test"
        ]
        return train_env_configs, test_env_configs

    def _create_configs(self, mode: EnvMode, env_id: int) -> dict:
        """Adapts the keyword arguments for the given mode.

        :param mode: the mode
        :return: adapted keyword arguments
        """
        if "cache_manager" in self.make_kwargs:
            cache_manager = self.make_kwargs.pop("cache_manager")
        else:
            cache_manager = {}
        kwargs = deepcopy(self.make_kwargs)
        kwargs["cache_manager"] = cache_manager
        self.make_kwargs["cache_manager"] = cache_manager
        if mode == EnvMode.TRAINING:
            if env_id == 0:
                kwargs.update(
                    {
                        "is_debug": False,
                        "use_real_agent": False,
                        "ssh_config": {},
                        "gemm_perf_config": {},
                        "log_config": {},
                    }
                )
            else:
                kwargs.update(self._train_env_configs[env_id - 1])
        elif mode == EnvMode.TEST:
            kwargs.update(self._test_env_configs[env_id])
        else:
            raise ValueError(f"Unsupported env mode: {mode}")
        return kwargs

    def _create_env(self, mode: EnvMode) -> gym.Env:
        """Creates a single environment for the given mode.

        :param mode: the mode
        :return: an environment
        """

    def _get_env_name(self, env_id: int, mode: EnvMode) -> str:
        """
        获取环境名称
        """
        if mode == EnvMode.TRAINING:
            return f"train-env-{env_id}" if env_id > 0 else "lazy-env"
        elif mode == EnvMode.TEST:
            return f"test-env-{env_id}"
        else:
            raise ValueError(f"Unsupported env mode: {mode}")

    def _create_gemm_env(
        self, env_id: int, mode: EnvMode, seed: int | None = None
    ) -> gym.Env:
        """
        创建矩阵乘法游戏环境
        """
        name = self._get_env_name(env_id, mode)
        self.logger.info(
            f"Creating GEMM environment '{name}' with mode '{mode.name}' and env_id '{env_id}'."
        )
        kwargs = self._create_configs(mode, env_id)
        env = GemmEnv(env_name=name, env_mode=mode, **kwargs)

        # initialize the environment with the given seed (if any)
        if seed is not None:
            rng = self._create_rng(seed)
            env.np_random = rng
            # also set the seed member within the environment such that it can be retrieved
            # (gymnasium's random seed handling is, unfortunately, broken)
            if hasattr(env, "_np_random_seed"):
                env._np_random_seed = seed

        return env

    def create_venv(
        self, num_envs: int, mode: EnvMode, seed: int | None = None
    ) -> BaseVectorEnv:
        """Create vectorized environments.

        :param num_envs: the number of environments
        :param mode: the mode for which to create.
            In `WATCH` mode the resulting venv will always be of type `DUMMY` with a single env.

        :return: the vectorized environments
        """
        rng = self._create_rng(seed)

        def create_factory_fn(env_id: int) -> Callable[[], gym.Env]:
            # create a factory function that uses a sampled random seed
            return lambda random_seed=self._next_seed(rng): self._create_gemm_env(
                env_id, mode, seed=random_seed
            )  # type: ignore

        # create the vectorized environment, seeded appropriately
        if mode == EnvMode.WATCH:
            raise NotImplementedError("WATCH mode is not implemented yet.")
        elif mode == EnvMode.TRAINING:
            venv = self.venv_type.create_venv(
                [create_factory_fn(i + 1) for i in range(num_envs)]
            )
        elif mode == EnvMode.TEST:
            venv = self.venv_type.create_venv(
                [create_factory_fn(i) for i in range(num_envs)]
            )
        else:
            raise ValueError(f"Unsupported env mode: {mode}")

        # seed the action samplers
        venv.seed([self._next_seed(rng) for _ in range(num_envs)])

        return venv

    def create_envs(
        self,
        num_training_envs: int,
        num_test_envs: int,
        create_watch_env: bool = False,
        seed: int | None = None,
    ) -> Environments:
        """Create environments for learning.

        :param num_training_envs: the number of training environments
        :param num_test_envs: the number of test environments
        :param create_watch_env: whether to create an environment for watching the agent
        :param seed: the random seed to use for environment creation
        :return: the environments
        """
        rng = self._create_rng(seed)
        env = self._create_gemm_env(0, EnvMode.TRAINING)
        training_envs = self.create_venv(
            num_training_envs, EnvMode.TRAINING, seed=self._next_seed(rng)
        )
        test_envs = self.create_venv(
            num_test_envs, EnvMode.TEST, seed=self._next_seed(rng)
        )
        watch_env = (
            self.create_venv(1, EnvMode.WATCH, seed=self._next_seed(rng))
            if create_watch_env
            else None
        )

        match EnvType.from_env(env):
            case EnvType.DISCRETE:
                return DiscreteEnvironments(env, training_envs, test_envs, watch_env)
            case EnvType.CONTINUOUS:
                return ContinuousEnvironments(env, training_envs, test_envs, watch_env)
            case _:
                raise ValueError


def make_gemm_env(
    task: str,
    seed: int,
    num_training_envs: int,
    num_test_envs: int,
    venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
    **make_kwargs: Any,
) -> tuple[gym.Env, BaseVectorEnv, BaseVectorEnv]:
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.

    :return: a tuple of (single env, training envs, test envs).
    """
    env_factory = GEMMEnvFactory(task, venv_type, **make_kwargs)
    envs = env_factory.create_envs(num_training_envs, num_test_envs, seed=seed)
    return envs.env, envs.training_envs, envs.test_envs
