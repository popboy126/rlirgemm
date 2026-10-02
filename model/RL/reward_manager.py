# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/9 23:47
# Contact       : hucheng.lew@qq.com
# File          : reward_manager.py
# IDE           : PyCharm
# Description   : 管理reward相关的工作

import logging
import unittest
from math import ceil, isclose
from multiprocessing.synchronize import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
from action_space_transformer import ActionSpaceTransformer
from exploration.history_performance import HistoryBestPerformance
from gemm_space import GEMMActionSpace
from history_samples_statistics import HistorySampleStatistics


class HistoryRewardContext:
    """
    历史奖励上下文
    """

    def __init__(
        self,
    ):
        """
        构造函数
        """
        pass


class RewardManager:
    """
    管理reward相关的工作
    """

    def __init__(
        self,
        name: str,
        loop_orders: List[int],
        packing_types: List[int],
        thread_types: List[List[int]],
        action_space_transformer: ActionSpaceTransformer,
        shape_perf_cache_shape_list: List[Tuple[int, int, int]] = None,
        shape_perf_cache_shm_name: str = None,
        shape_perf_cache_lock: Lock = None,
        action_stat_cache_shm_name: str = None,
        action_stat_cache_lock: Lock = None,
        element_byte_size: int = 4,
        reward_min: float = -40.0,
        reward_max: float = 40.0,
        reward_norm_min: float = -1.0,
        reward_norm_max: float = 1.0,
        reward_break_max: float = 1.0,
        reward_break_k: float = 15.0,
        reward_break_threshold: float = 0.05,
        performance_ema_alpha: float = 0.3,
        reward_perf_exploration_factor: float = 1,
        reward_perf_absolute_factor: float = 1,
        reward_perf_ema_factor: float = 1,
        reward_perf_breakthrough_factor: float = 1,
        reward_perf_ema_penalty_threshold: float = 0.85,
        reward_perf_ema_penalty_factor: float = 1,  # 适当增强EMA惩罚系数
        reward_shape_best_ratio_threshold: float = 0.8,  # 形状最优性能比率基准（方案C）
        logger: Union[None, logging.Logger] = None,
        reward_mode: str = "multi_level",
    ):
        """
        构造函数
        :param class_labels: 分类情况
        """
        self._action_space_transformer = action_space_transformer

        self._element_byte_size = element_byte_size

        self._reward_min = reward_min
        self._reward_max = reward_max
        self._reward_norm_min = reward_norm_min
        self._reward_norm_max = reward_norm_max
        # 跟突破奖励有关
        self._reward_break_max = (
            reward_break_max  # 一般取1, 如果希望突破奖励强，则可取2
        )
        self._reward_break_k = (
            reward_break_k  # 一般取10 ~ 20之间的数值，表示过渡区域宽度
        )
        self._reward_break_threshold = reward_break_threshold

        self._logger = logger
        if (
            loop_orders is None
            or packing_types is None
            or thread_types is None
            or action_stat_cache_shm_name is None
            or action_stat_cache_lock is None
        ):
            self._history_queue = None
        else:
            self._history_queue = HistorySampleStatistics(
                name=name,
                loop_orders=loop_orders,
                packing_types=packing_types,
                thread_types=thread_types,
                shm_name=action_stat_cache_shm_name,
                lock=action_stat_cache_lock,
                element_byte_size=element_byte_size,
                logger=self._logger,
            )

        if (
            shape_perf_cache_shape_list is None
            or shape_perf_cache_shm_name is None
            or shape_perf_cache_lock is None
        ):
            self._history_performance_dict = None
        else:
            self._history_performance_dict = HistoryBestPerformance(
                shape_list=shape_perf_cache_shape_list,
                shm_name=shape_perf_cache_shm_name,
                lock=shape_perf_cache_lock,
                ema_alpha=performance_ema_alpha,
                logger=self._logger,
            )

        # 参数不合法的reward
        self._reward_param_invalid = -0.9

        # 参数合法状态下的各项reward
        self._reward_param_valid = 0.0  # 参数合法不给奖励，避免干扰性能激励
        # 性能奖励: 最大探索奖励
        self._reward_perf_max_exploration = 0.2
        self._reward_perf_max_absolute = 0.6  # 提升2.4倍，增强绝对性能奖励权重
        self._reward_perf_max_ema = 0.2
        self._reward_perf_max_breakthrough = 0.15
        self._reward_perf_penalty_max = 0.283  # 提升以实现最大性能惩罚≈-0.85

        self._reward_perf_exploration_factor = reward_perf_exploration_factor
        self._reward_perf_absolute_factor = reward_perf_absolute_factor
        self._reward_perf_ema_factor = reward_perf_ema_factor
        self._reward_perf_breakthrough_factor = reward_perf_breakthrough_factor

        # 负向惩罚相关参数
        self._reward_perf_ema_penalty_threshold = reward_perf_ema_penalty_threshold
        self._reward_perf_ema_penalty_factor = reward_perf_ema_penalty_factor

        # 形状最优性能比率基准（方案C）
        self._reward_shape_best_ratio_threshold = reward_shape_best_ratio_threshold
        if reward_mode not in {"multi_level", "pure_performance"}:
            raise ValueError(
                "reward_mode must be 'multi_level' or 'pure_performance'"
            )
        self._reward_mode = reward_mode

    @property
    def history_queue(self) -> HistorySampleStatistics:
        """
        获取历史队列
        """
        return self._history_queue

    @property
    def history_performance_dict(self) -> HistoryBestPerformance:
        """
        获取历史性能字典
        """
        return self._history_performance_dict

    def get_params_check_reward(
        self,
        gemm_shape: Tuple[int, int, int],
        loop_order: int,
        packing_type: int,
        thread: Tuple[int, int, int],
        blocking: Tuple[int, int, int],
        gemm_prfm: Tuple[int, int, int],
        add_prfm: Tuple[int, int],
    ) -> float:
        """
        获取参数检查奖励
        """
        # 参数解包
        M, K, N = gemm_shape
        mc, kc, nc = blocking
        tm, tk, tn = thread
        gemm_prfm_a, gemm_prfm_b, gemm_prfm_c = gemm_prfm
        add_prfm_c, add_prfm_partial_sum = add_prfm

        # 三个维度的线程平处理的数量
        if loop_order < 3:
            gemm_parallel_factor, gemm_prfm_factor, add_prfm_factor = 0.67, 0.33, 0
        else:
            gemm_parallel_factor, gemm_prfm_factor, add_prfm_factor = 0.5, 0.25, 0.25
        raw_reward = (
            gemm_parallel_factor
            * self.check_params_parallel_tasks(
                M, K, N, loop_order, tm, tk, tn, mc, kc, nc
            )
            / 3
        )
        raw_reward += (
            gemm_prfm_factor
            * self.check_prfm_skip_bytes(M, K, N, gemm_prfm_a, gemm_prfm_b, gemm_prfm_c)
            / 3
        )
        raw_reward += (
            add_prfm_factor
            * self.check_prfm_add_skip_bytes(M, N, add_prfm_c, add_prfm_partial_sum)
            / 2
        )
        if isclose(raw_reward, 0.0, abs_tol=1e-6):
            reward = self._reward_param_valid
        else:
            reward = self._reward_param_invalid

        return reward

    def check_prfm_skip_bytes(
        self,
        m: int,
        k: int,
        n: int,
        prfm_mat_a_skip_bytes: int,
        prfm_mat_b_skip_bytes: int,
        prfm_mat_c_skip_bytes: int,
    ) -> int:
        """
        检查预取参数
        """
        reward = 0
        mat_a_bytes_size = m * k * self._element_byte_size
        gap = mat_a_bytes_size - prfm_mat_a_skip_bytes
        if gap < 0:
            reward += min((abs(gap) / mat_a_bytes_size), 1)

        mat_b_bytes_size = k * n * self._element_byte_size
        gap = mat_b_bytes_size - prfm_mat_b_skip_bytes
        if gap < 0:
            reward += min((abs(gap) / mat_b_bytes_size), 1)

        mat_c_bytes_size = m * n * self._element_byte_size
        gap = mat_c_bytes_size - prfm_mat_c_skip_bytes
        if gap < 0:
            reward += min((abs(gap) / mat_c_bytes_size), 1)

        return reward

    def check_prfm_add_skip_bytes(
        self,
        m: int,
        n: int,
        prfm_add_c_skip_bytes: int,
        prfm_add_partial_sum_skip_bytes: int,
    ) -> int:
        """
        检查预取参数
        """
        reward = 0
        mat_c_bytes_size = m * n * self._element_byte_size
        gap = mat_c_bytes_size - prfm_add_c_skip_bytes
        if gap < 0:
            reward += min((abs(gap) / mat_c_bytes_size), 1)

        gap = mat_c_bytes_size - prfm_add_partial_sum_skip_bytes
        if gap < 0:
            reward += min((abs(gap) / mat_c_bytes_size), 1)
        return reward

    def check_params_parallel_tasks(
        self,
        m: int,
        k: int,
        n: int,
        loop_order: int,
        tm: int,
        tk: int,
        tn: int,
        mc: int,
        kc: int,
        nc: int,
    ) -> int:
        """
        检查参数是否合理
        """
        reward = 0
        if loop_order == 0:  # LO_NONE
            total_threads = tm * tk * tn
            if total_threads > 1:
                reward = 3 * (total_threads - 1) / total_threads
            else:
                reward = 0
        elif 0 < loop_order < 3:  # 需要检查M和N维度的的单个线程的分块大小
            dim_m_reward = (1.5 * self.check_thread_dim_blocking(m, tm, mc, False)) / 2
            dim_n_reward = (1.5 * self.check_thread_dim_blocking(n, tn, nc, False)) / 2
            reward = dim_m_reward + dim_n_reward
        elif loop_order == 3:  # LO_NKM
            dim_n_reward = self.check_thread_dim_blocking(n, tn, nc, True) / 2
            dim_k_reward = self.check_thread_dim_blocking(k, tk, kc, True) / 2
            dim_m_reward = self.check_thread_dim_blocking(m, tm, mc, False) / 2
            reward = dim_n_reward + dim_k_reward + dim_m_reward
        elif loop_order in [4, 8]:  # LO_NMK, LO_MNK
            dim_m_reward = self.check_thread_dim_blocking(m, tm, mc, True) / 2
            dim_n_reward = self.check_thread_dim_blocking(n, tn, nc, True) / 2
            dim_k_reward = self.check_thread_dim_blocking(k, tk, kc, True) / 2
            reward = dim_m_reward + dim_n_reward + dim_k_reward
        elif loop_order in [5, 6]:  # LO_KNM, LO_KMN
            dim_m_reward = self.check_thread_dim_blocking(k, tk, kc, True) / 2
            dim_n_reward = self.check_thread_dim_blocking(n, tn, nc, False) / 2
            dim_k_reward = self.check_thread_dim_blocking(m, tm, mc, False) / 2
            reward = dim_m_reward + dim_n_reward + dim_k_reward
        elif loop_order == 7:  # LO_MKN
            dim_m_reward = self.check_thread_dim_blocking(m, tm, mc, True) / 2
            dim_k_reward = self.check_thread_dim_blocking(k, tk, kc, True) / 2
            dim_n_reward = self.check_thread_dim_blocking(n, tn, nc, False) / 2
            reward = dim_m_reward + dim_k_reward + dim_n_reward
        else:
            self._logger.error(f"Invalid loop order: {loop_order}")
            reward = 3

        return reward

    def check_thread_dim_blocking(
        self,
        dim_size: int,
        thread_num: int,
        block_size: int,
        must_same_loop_times: bool,
    ) -> float:
        """
        检查单个维度的线程数和分块大小是否合理
        """
        dim_per_thread = ceil(dim_size / thread_num)
        spliter_idx_set = {i for i in range(0, dim_size, dim_per_thread)}
        spliter_idx_set.add(dim_size)
        spliter_idx_list = sorted(list(spliter_idx_set))

        if len(spliter_idx_list) <= thread_num:  # 维度数比线程数少
            return 2 * min(abs(thread_num * dim_per_thread - dim_size) / dim_size, 1)

        task_num_list = [
            spliter_idx_list[i + 1] - spliter_idx_list[i]
            for i in range(0, len(spliter_idx_list) - 1, 1)
        ]
        dim_per_thread_min = min(task_num_list)
        dim_per_thread_max = max(task_num_list)

        if dim_per_thread_max < block_size:
            block_size = dim_per_thread_max

        if must_same_loop_times:
            loop_times_min = ceil(dim_per_thread_min / block_size)
            loop_times_max = ceil(dim_per_thread_max / block_size)
            if loop_times_min != loop_times_max:
                return 2 * min(abs(loop_times_min - loop_times_max) / loop_times_min, 1)

        return 0

    def _normalize_performance_reward(
        self, performance: float, shape: Tuple[int, int, int]
    ) -> float:
        """将当前性能映射为与多层级奖励相同尺度的单调奖励。

        纯性能消融只依赖当前性能和历史基准，不访问 action history、EMA
        惩罚或突破奖励。使用完整方法绝对性能奖励所采用的平滑函数和 clip。
        """
        history = self.history_performance_dict
        if history is None:
            # 允许脱离共享缓存的本地 smoke 测试安全运行。
            baseline = performance
            shape_best = None
        else:
            shape_best = history.get_shape_best_performance(shape)
            baseline = (
                shape_best if shape_best is not None and shape_best > 0 else None
            )
            if baseline is None:
                baseline = history.get_best_global_avg_performance()
        if baseline is None or baseline <= 0:
            # 没有可用历史基准时沿用完整方法的冷启动尺度。
            baseline = performance
        ratio = performance / baseline if baseline > 0 else 0.0
        center = self._reward_shape_best_ratio_threshold * 0.75
        if shape_best is None or shape_best <= 0:
            center = 0.6
        normalized = ratio / max(center, 1e-6)
        smooth = (normalized**6.0) / (1.0 + normalized**6.0)
        return float(
            np.clip(
                smooth * self._reward_perf_max_absolute,
                self._reward_norm_min,
                self._reward_norm_max,
            )
        )

    def compute_performance_reward(
        self,
        shape: Tuple[int, int, int],
        performance: float,
        loop_order: int,
        packing_type: int,
        thread_type: Tuple[int, int, int],
    ) -> float:
        """
        计算参数正确的情况下的reward
        """
        if performance <= 0:
            return 0.0

        if self._reward_mode == "pure_performance":
            reward = self._normalize_performance_reward(performance, shape)
            if self._logger is not None:
                self._logger.debug(
                    f"[Reward] mode=pure_performance, shape={shape}, "
                    f"perf={performance:.2f}, performance_reward={reward:.4f}"
                )
            return reward

        if self._reward_mode != "multi_level":
            # 构造函数已校验模式；保留守卫避免未来新增模式误用完整奖励逻辑。
            raise ValueError(f"unsupported reward mode: {self._reward_mode}")

        # multi_level 分支：保留原有探索、绝对性能、突破和惩罚分量。
        shape_ema_performance = self.history_performance_dict.get_ema_performance(shape)
        shape_best_performance = (
            self.history_performance_dict.get_shape_best_performance(shape)
        )
        # 使用历史最好的全局平均性能作为基准，避免基准随性能下降而降低
        best_global_avg = (
            self.history_performance_dict.get_best_global_avg_performance()
        )

        # 分量1: 探索奖励
        ideal_ratio, action_ratio = self._history_queue.get_discrete_action_ratio(
            loop_order, packing_type, thread_type
        )
        exploration_reward = 0.0
        if action_ratio < ideal_ratio:
            exploration_reward = (
                (ideal_ratio - action_ratio) / ideal_ratio
            ) * self._reward_perf_max_exploration

        # 分量2: 平滑绝对性能奖励（改进：Sigmoid式连续平滑函数）
        absolute_reward = 0.0
        if shape_best_performance is not None and shape_best_performance > 0:
            perf_ratio = performance / shape_best_performance

            # 使用Sigmoid式连续平滑函数，全程平滑无突变点
            # 核心思想：让奖励在整个perf_ratio区间连续平滑增长
            # perf_ratio低时激励很小，perf_ratio高时激励快速增长

            # 使用现有参数组合，不引入新参数
            # center作为sigmoid中心点，控制奖励增长的位置
            center = self._reward_shape_best_ratio_threshold * 0.75  # 0.6
            steepness = 6.0  # 固定值，控制曲线陡峭度

            # 归一化到center附近
            normalized_perf = perf_ratio / center

            # 使用幂函数作为sigmoid近似（无需exp函数）
            # sigmoid_like在perf_ratio=center时≈0.5，perf_ratio>center时快速增长
            sigmoid_like = (normalized_perf**steepness) / (
                1.0 + normalized_perf**steepness
            )

            # 计算平滑奖励，在perf_ratio=1.0时达到满额0.6
            absolute_reward = sigmoid_like * self._reward_perf_max_absolute
        elif best_global_avg is not None and best_global_avg > 0:
            # 冷启动兜底：新形状使用全局基准
            perf_ratio_global = performance / best_global_avg
            if perf_ratio_global > 0.6:  # 降低门槛
                # 使用相同的sigmoid式平滑函数
                normalized_perf = perf_ratio_global / 0.6
                sigmoid_like = (normalized_perf**6.0) / (1.0 + normalized_perf**6.0)
                absolute_reward = sigmoid_like * self._reward_perf_max_absolute

        # 分量3: 废弃ema_reward正向激励（避免与absolute_reward重叠）
        # 保留EMA作为惩罚基准，不再给予正向奖励
        ema_reward = 0.0

        # 分量4: 突破奖励
        breakthrough_reward = 0.0
        improvement_pct = 0.0
        if (shape_best_performance is not None) and (
            performance > shape_best_performance
        ):
            # 超过历史最佳threshold以上，给予额外奖励
            improvement_pct = (
                performance - shape_best_performance
            ) / shape_best_performance
            breakthrough_reward = min(
                self._reward_perf_max_breakthrough
                / (
                    1
                    + np.exp(
                        -self._reward_break_k
                        * (improvement_pct - self._reward_break_threshold)
                    )
                ),
                1,
            )

        # 负向惩罚计算 - 双基准惩罚机制（改进）
        # 基准1：EMA基准（保底机制，保护策略稳定性）
        # 基准2：best基准（激励机制，防止策略性能过低）
        negative_penalty = 0.0

        if shape_ema_performance is not None and shape_best_performance is not None:
            perf_ratio_to_ema = performance / shape_ema_performance
            perf_ratio_to_best = performance / shape_best_performance

            ema_threshold = self._reward_perf_ema_penalty_threshold  # 0.7
            best_threshold = self._reward_shape_best_ratio_threshold * 0.7  # 0.56

            ema_penalty = 0.0
            best_penalty = 0.0

            # 基准1：EMA基准（保底机制）
            # 检测策略性能掉落（相对于自身历史水平）
            if perf_ratio_to_ema < ema_threshold:
                ema_gap = (ema_threshold - perf_ratio_to_ema) / ema_threshold
                # 平滑指数惩罚：使用幂函数避免突变
                ema_penalty = (ema_gap**2) * self._reward_perf_penalty_max

            # 基准2：best基准（激励机制）
            # 检测策略性能过低（相对于最优性能）
            if perf_ratio_to_best < best_threshold:
                best_gap = (best_threshold - perf_ratio_to_best) / best_threshold
                # 平滑指数惩罚，权重降低以避免过度惩罚正常策略
                best_penalty = (best_gap**2) * self._reward_perf_penalty_max * 0.5

            # 双基准机制：取最大惩罚（更严厉的基准）
            negative_penalty = max(ema_penalty, best_penalty)

        exploration_reward = (
            0 if negative_penalty > 0 else exploration_reward
        )  # 当惩罚奖励大于0时，取消探索奖励
        performance_reward = (
            self._reward_perf_exploration_factor * exploration_reward
            + self._reward_perf_absolute_factor * absolute_reward
            + self._reward_perf_ema_factor * ema_reward
            + self._reward_perf_breakthrough_factor * breakthrough_reward
            - self._reward_perf_ema_penalty_factor * negative_penalty
        )

        # 调试日志：记录奖励计算详情（改进版：包含perf_ratio信息）
        shape_ema_performance = (
            0.0 if shape_ema_performance is None else shape_ema_performance
        )
        best_global_avg = 0.0 if best_global_avg is None else best_global_avg
        shape_best_performance = (
            0.0 if shape_best_performance is None else shape_best_performance
        )

        # 计算性能比率（用于调试）
        perf_ratio = 0.0
        smooth_factor = 0.0
        if shape_best_performance > 0:
            perf_ratio = performance / shape_best_performance
            baseline_ratio = self._reward_shape_best_ratio_threshold
            if perf_ratio > baseline_ratio:
                excess_ratio = (perf_ratio - baseline_ratio) / (1.0 - baseline_ratio)
                smooth_factor = min(excess_ratio**1.5, 1.0)

        self._logger.debug(
            f"[Reward] shape={shape}, loop_order={loop_order}, packing_type={packing_type}, thread_type={thread_type}, "
            f"perf={performance:.2f}, ema={shape_ema_performance:.2f}, "
            f"best={shape_best_performance:.2f}, perf_ratio={perf_ratio:.3f}, "
            f"best_global_avg={best_global_avg:.2f}, "
            f"exploration_reward={exploration_reward:.4f}, "
            f"absolute_reward={absolute_reward:.4f} (smooth_factor={smooth_factor:.3f}), "
            f"breakthrough_reward={breakthrough_reward:.4f}, improvement_pct={improvement_pct:.2f}, "
            f"negative_penalty={negative_penalty:.4f}, total_reward={performance_reward:.2f}"
        )

        return performance_reward

    def reward_normalize(self, raw_reward: float) -> float:
        """
        参数检测失败和最后的性能的奖励缩放
        """
        # 先算原来的比例
        ratio = (raw_reward - self._reward_min) / (self._reward_max - self._reward_min)
        # 再算新的值
        reward = (
            ratio * (self._reward_norm_max - self._reward_norm_min)
            + self._reward_norm_min
        )
        return reward

    def compute_final_reward(
        self, check_reward: float, performance: float, **kwargs
    ) -> float:
        """
        计算最终奖励

        关键改进：
        - 将线程配置传递给性能奖励计算和历史性能记录
        - 确保不同线程配置有独立的性能基准
        """
        # diversity_penalty = 0.0
        shape = (kwargs["M"], kwargs["K"], kwargs["N"])

        loop_order: int = kwargs["loop_order"]
        packing_type: int = kwargs["packing_type"]

        # 提取线程配置
        thread_type = kwargs.get("thread_type", (1, 1, 1))

        # thread_num = (
        #     kwargs["thread_type"][0]
        #     * kwargs["thread_type"][1]
        #     * kwargs["thread_type"][2]
        # )

        if check_reward < 0:
            performance_reward = 0.0
            raw_reward = (
                abs(check_reward) * self._reward_min
                + performance_reward * self._reward_max
            )
        else:
            performance_reward = self.compute_performance_reward(
                shape=shape,
                performance=performance,
                loop_order=loop_order,
                packing_type=packing_type,
                thread_type=thread_type,
            )
            raw_reward = (check_reward + performance_reward) * self._reward_max

        final_reward = self.reward_normalize(raw_reward)
        self._logger.debug(
            f"Reward for shape({kwargs['M']}, {kwargs['K']}, {kwargs['N']}) "
            f"thread_type={thread_type}, loop_order={loop_order}, packing_type: {packing_type}, "
            f"perf={performance:.2f} GFlops, "
            f"check_r={check_reward:.2f}, perf_r={performance_reward:.2f}, "
            f"raw_reward={raw_reward:.4f}, "
            f"final={final_reward:.4f}"
        )
        return final_reward


class TestRewardManager(unittest.TestCase):
    """
    测试reward_manager
    """

    def setUp(self) -> None:
        """
        构建测试的reward_manager
        """
        self.loop_orders = [1, 2, 3, 4, 5, 6, 7, 8]
        self.packing_types = [0, 1]
        self.thread_types = [[1, 2, 1], [1, 1, 2]]
        self._reward_manager = RewardManager(
            name="unit_test",
            loop_orders=self.loop_orders,
            packing_types=self.packing_types,
            thread_types=self.thread_types,
            action_space_transformer=ActionSpaceTransformer(
                loop_orders=self.loop_orders,
                packing_types=self.packing_types,
                thread_types=self.thread_types,
                matrix_element_size=4,
                action_space=GEMMActionSpace(
                    loop_order_type=self.loop_orders,
                    packing_type=self.packing_types,
                    thread_type=self.thread_types,
                ),
            ),
            element_byte_size=4,
            reward_min=-500,
            reward_max=500,
            reward_norm_min=-1.0,
            reward_norm_max=1.0,
            reward_break_max=1.0,
            reward_break_k=15.0,
            reward_break_threshold=0.05,
            performance_ema_alpha=0.05,
            logger=None,
        )

    def test_get_params_check_reward(self):
        """
        测试参数校验正确性
        """
        M, K, N = 1, 8, 64
        loop_order = 2
        packing_type = 1
        tm, tk, tn = 1, 1, 2
        mc, kc, nc = 1, 1, 52
        prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes = 23, 1358, 97
        prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes = 0, 0
        check_result = self._reward_manager.get_params_check_reward(
            gemm_shape=(M, K, N),
            loop_order=loop_order,
            packing_type=packing_type,
            thread=(tm, tk, tn),
            blocking=(mc, kc, nc),
            gemm_prfm=(prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes),
            add_prfm=(prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes),
        )

        self.assertGreater(check_result, 0)
