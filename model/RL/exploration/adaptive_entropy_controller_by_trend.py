# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/4/3 14:30
# Contact       : hucheng.lew@qq.com
# File          : adaptive_entropy_controller_by_trend.py
# IDE           : PyCharm
# Description   : 基于趋势的自适应熵系数控制器 (v5版本)

import logging
from typing import Any, Dict, List, Optional

import numpy as np


class AdaptiveEntropyControllerByTrend:
    """
    基于趋势的自适应熵系数控制器 (v5版本)

    核心特性：
    1. 趋势驱动：通过监测entropy loss下降速率和performance上升速率
    2. 完全不考虑entropy loss / base loss的比例
    3. 差异化调整：四个熵系数独立调整
    4. 阶段化策略：根据训练epoch动态调整策略

    算法流程：
    1. 监测并计算entropy loss的下降速率
    2. 监测并计算performance的上升速率
    3. 根据epoch阶段和趋势动态调整各成分熵系数
    4. 返回更新后的系数

    使用示例：
    ```python
    controller = AdaptiveEntropyControllerByTrend(
        trend_window_size=10,
        learning_rate=0.01,
        coef_range=[...],
        warmup_epochs=10,
        init_coefs={
            "loop_order": 0.03,
            "packing_type": 0.03,
            "thread_type": 0.06,
            "cont": 0.10,
        },
    )

    # 在训练循环中
    for epoch in range(max_epochs):
        current_perf = policy._last_mean_env_performance
        ent_loss = policy._last_mean_ent_loss
        raw_entropies = {
            'loop_order': policy._last_mean_raw_entropy_disc_loop_order,
            'packing_type': policy._last_mean_raw_entropy_disc_packing_type,
            'thread_type': policy._last_mean_raw_entropy_disc_thread_type,
            'cont': policy._last_mean_raw_entropy_cont,
        }

        # 更新并获取新系数
        new_coefs = controller.update(
            epoch=epoch,
            current_performance=current_perf,
            entropy_loss=ent_loss,
            raw_entropies=raw_entropies,
        )

        # 应用新系数
        policy._ent_coef_disc_loop_order = new_coefs['loop_order']
        policy._ent_coef_disc_packing_type = new_coefs['packing_type']
        policy._ent_coef_disc_thread_type = new_coefs['thread_type']
        policy._ent_coef_cont = new_coefs['cont']
    ```
    """

    def __init__(
        self,
        trend_window_size: int = 10,
        base_learning_rate: float = 0.01,
        warmup_epochs: int = 10,
        reset_interval: int = 500,
        init_coefs: Dict[str, float] | None = None,
        stage_config: List[Dict[str, Any]] | None = None,
        adaptive_learning_rate: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化趋势驱动自适应熵系数控制器

        Args:
            trend_window_size: 计算趋势的滑动窗口大小
            base_learning_rate: 基础学习率（自适应时作为默认值）
            warmup_epochs: 预热epoch数
            reset_interval: 全局最佳重置间隔
            init_coefs: 初始系数值
            stage_config: 阶段配置（包含策略、阈值、敏感度、learning_rate等）
            adaptive_learning_rate: 是否启用自适应learning_rate
            logger: 日志记录器
        """
        # 基础配置
        self.trend_window_size = trend_window_size
        self.base_learning_rate = base_learning_rate
        self.adaptive_learning_rate = adaptive_learning_rate
        self.warmup_epochs = warmup_epochs
        self.reset_interval = reset_interval

        # coef_range现在整合到stage_config中，不再独立配置

        # 状态变量
        self.global_best_raw = -float("inf")
        self.global_best_smoothed = None
        self.last_reset_epoch = 0

        # 各系数当前值
        if init_coefs is None:
            self.current_coefs = {
                "loop_order": 0.03,
                "packing_type": 0.03,
                "thread_type": 0.06,
                "cont": 0.10,
            }
        else:
            self.current_coefs = init_coefs

        # 历史记录
        self.performance_history: List[float] = []
        self.entropy_loss_history: List[float] = []
        self.coef_history: Dict[str, List[float]] = {
            k: [] for k in self.current_coefs.keys()
        }

        # 趋势历史记录（用于计算速率）
        self.perf_trend_history: List[float] = []
        self.ent_loss_trend_history: List[float] = []

        # 日志
        self.logger = logger or logging.getLogger("AdaptiveEntropyControllerByTrend")

        # 阶段配置（合并策略、阈值、敏感度）
        if stage_config is None:
            self.stage_config = [
                {
                    "epoch_range": [0, 50],
                    "strategy": "explore",  # 早期：探索策略
                    "learning_rate": 0.02,  # 早期阶段使用较大的学习率，快速探索
                    "coef_boost": 1.5,  # 系数提升因子
                    "coef_range": {  # 早期阶段允许更大的系数范围，激进探索
                        "loop_order": [0.001, 0.5],
                        "packing_type": [0.001, 0.5],
                        "thread_type": [0.001, 0.5],
                        "cont": [0.001, 0.5],
                    },
                    "ent_drop_sensitivity": 2.0,  # entropy下降敏感度
                    "perf_rise_sensitivity": 0.5,  # performance上升敏感度
                    "adjustment_intensity": {  # 调整强度系数（替代硬编码参数）
                        "ent_drop_fast": 1.0,  # entropy快速下降时的调整强度
                        "ent_drop_slow": 0.5,  # entropy下降慢时的调整强度
                        "perf_rise_fast": 0.5,  # performance快速上升时的调整强度
                        "perf_rise_slow": 1.0,  # performance上升慢时的调整强度
                    },
                    "trend_thresholds": {
                        "perf_rise_fast": 5.0,  # performance快速上升阈值
                        "perf_rise_slow": 1.0,  # performance慢速上升阈值
                        "ent_drop_fast": 0.1,  # entropy loss快速下降阈值
                        "ent_drop_slow": 0.02,  # entropy loss慢速下降阈值
                    },
                    "action_sensitivity": {
                        "loop_order": 0.15,  # 中等敏感度
                        "packing_type": 0.10,  # 较低敏感度
                        "thread_type": 0.25,  # 高敏感度
                        "cont": 0.50,  # 最高敏感度
                    },
                },
                {
                    "epoch_range": [50, 200],
                    "strategy": "balanced",  # 中期：平衡策略
                    "learning_rate": 0.01,  # 中期阶段使用标准学习率，平衡调整
                    "coef_boost": 1.2,
                    "coef_range": {  # 中期阶段略微缩小系数范围，平衡调整
                        "loop_order": [0.001, 0.4],
                        "packing_type": [0.001, 0.4],
                        "thread_type": [0.001, 0.4],
                        "cont": [0.001, 0.4],
                    },
                    "ent_drop_sensitivity": 1.0,
                    "perf_rise_sensitivity": 1.0,
                    "adjustment_intensity": {  # 调整强度系数
                        "ent_drop_fast": 0.5,  # entropy快速下降时适度调整
                        "ent_drop_slow": 0.3,  # entropy下降慢时轻微调整
                        "perf_rise_fast": 0.3,  # performance快速上升时轻微调整
                        "perf_rise_slow": 0.5,  # performance上升慢时适度调整
                    },
                    "trend_thresholds": {
                        "perf_rise_fast": 5.0,
                        "perf_rise_slow": 1.0,
                        "ent_drop_fast": 0.1,
                        "ent_drop_slow": 0.02,
                    },
                    "action_sensitivity": {
                        "loop_order": 0.15,
                        "packing_type": 0.10,
                        "thread_type": 0.25,
                        "cont": 0.50,
                    },
                },
                {
                    "epoch_range": [200, 9999],
                    "strategy": "exploit",  # 后期：利用策略
                    "learning_rate": 0.005,  # 后期阶段使用较小的学习率，精细调整
                    "coef_boost": 0.8,
                    "coef_range": {  # 后期阶段进一步缩小系数范围，精细调整
                        "loop_order": [0.001, 0.3],
                        "packing_type": [0.001, 0.3],
                        "thread_type": [0.001, 0.3],
                        "cont": [0.001, 0.3],
                    },
                    "ent_drop_sensitivity": 0.5,
                    "perf_rise_sensitivity": 1.5,
                    "adjustment_intensity": {  # 调整强度系数
                        "ent_drop_fast": 0.3,  # entropy快速下降时轻微调整
                        "ent_drop_slow": 0.5,  # entropy下降慢时适度调整
                        "perf_rise_fast": 1.0,  # performance快速上升时完全响应
                        "perf_rise_slow": 0.5,  # performance上升慢时适度调整
                    },
                    "trend_thresholds": {
                        "perf_rise_fast": 5.0,
                        "perf_rise_slow": 1.0,
                        "ent_drop_fast": 0.1,
                        "ent_drop_slow": 0.02,
                    },
                    "action_sensitivity": {
                        "loop_order": 0.15,
                        "packing_type": 0.10,
                        "thread_type": 0.25,
                        "cont": 0.50,
                    },
                },
            ]
        else:
            self.stage_config = stage_config

        self.logger.info(
            f"AdaptiveEntropyControllerByTrend initialized:\n"
            f"  trend_window_size={trend_window_size}\n"
            f"  base_learning_rate={base_learning_rate}\n"
            f"  adaptive_learning_rate={adaptive_learning_rate}\n"
            f"  warmup_epochs={warmup_epochs}\n"
            f"  init_coefs={self.current_coefs}\n"
            f"  coef_range integrated in stage_config\n"
            f"  stage_config={self.stage_config}"
        )

    @staticmethod
    def get_target_datablob_by_epoch(
        identity: int | float,
        parameters: List[Dict[str, Any]],
        range_key: str = "epoch_range",
    ) -> Any:
        """
        根据identity获取目标配置
        """
        target_blob = None
        for blob in parameters:
            if blob[range_key][0] <= identity < blob[range_key][1]:
                target_blob = blob
                break
        return target_blob

    def get_coef_range(self, epoch: int, action: str) -> tuple[float, float]:
        """
        获取系数的最小值和最大值（从stage_config中获取）
        """
        # 默认系数范围
        default_coef_range = {
            "loop_order": [0.001, 0.5],
            "packing_type": [0.001, 0.5],
            "thread_type": [0.001, 0.5],
            "cont": [0.001, 0.5],
        }

        blob = self.get_target_datablob_by_epoch(epoch, self.stage_config)
        if blob is None:
            return default_coef_range[action][0], default_coef_range[action][1]

        coef_range_dict = blob.get("coef_range", default_coef_range)
        return coef_range_dict[action][0], coef_range_dict[action][1]

    def get_action_sensitivity(self, epoch: int, action: str) -> float:
        """
        获取动作敏感性
        """
        default = 1.0
        blob = self.get_target_datablob_by_epoch(epoch, self.stage_config)
        if blob is None:
            return default
        action_sensitivity_dict = blob.get("action_sensitivity", {})
        return action_sensitivity_dict.get(action, default)

    def get_trend_thresholds(self, epoch: int) -> Dict[str, float]:
        """
        获取趋势阈值配置
        """
        default_thresholds = {
            "perf_rise_fast": 5.0,
            "perf_rise_slow": 1.0,
            "ent_drop_fast": 0.1,
            "ent_drop_slow": 0.02,
        }
        blob = self.get_target_datablob_by_epoch(epoch, self.stage_config)
        if blob is None:
            return default_thresholds
        trend_thresholds_dict = blob.get("trend_thresholds", {})
        return {
            k: trend_thresholds_dict.get(k, v) for k, v in default_thresholds.items()
        }

    def get_learning_rate(self, epoch: int) -> float:
        """
        获取当前阶段的learning_rate

        Args:
            epoch: 当前epoch

        Returns:
            当前阶段的learning_rate
        """
        # 如果不启用自适应learning_rate，使用基础learning_rate
        if not self.adaptive_learning_rate:
            return self.base_learning_rate

        # 从stage_config中获取当前阶段的learning_rate
        blob = self.get_target_datablob_by_epoch(epoch, self.stage_config)
        if blob is None:
            return self.base_learning_rate

        # 返回当前阶段的learning_rate，如果没有配置则使用基础learning_rate
        return blob.get("learning_rate", self.base_learning_rate)

    def get_stage_strategy(self, epoch: int) -> Dict[str, Any]:
        """
        获取阶段策略配置
        """
        default_strategy = {
            "strategy": "balanced",
            "learning_rate": self.base_learning_rate,
            "coef_boost": 1.0,
            "coef_range": {
                "loop_order": [0.001, 0.5],
                "packing_type": [0.001, 0.5],
                "thread_type": [0.001, 0.5],
                "cont": [0.001, 0.5],
            },
            "ent_drop_sensitivity": 1.0,
            "perf_rise_sensitivity": 1.0,
            "adjustment_intensity": {
                "ent_drop_fast": 0.5,
                "ent_drop_slow": 0.3,
                "perf_rise_fast": 0.3,
                "perf_rise_slow": 0.5,
            },
        }
        blob = self.get_target_datablob_by_epoch(epoch, self.stage_config)
        if blob is None:
            return default_strategy
        # 提取策略相关字段（不包括epoch_range、trend_thresholds、action_sensitivity）
        result = {}
        for key in [
            "strategy",
            "learning_rate",
            "coef_boost",
            "coef_range",
            "ent_drop_sensitivity",
            "perf_rise_sensitivity",
            "adjustment_intensity",
        ]:
            result[key] = blob.get(key, default_strategy.get(key))
        return result

    def compute_trend_rate(self, history: List[float], window_size: int) -> float:
        """
        计算历史数据的趋势速率（线性拟合斜率）

        Args:
            history: 历史数据列表
            window_size: 窗口大小

        Returns:
            趋势速率（斜率）
        """
        if len(history) < 2:
            return 0.0

        # 取最近的窗口数据
        recent_data = history[-min(window_size, len(history)) :]
        if len(recent_data) < 2:
            return 0.0

        # 线性拟合计算趋势
        x = np.arange(len(recent_data))
        y = np.array(recent_data)

        # 计算线性回归的斜率: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        n = len(x)
        sum_x = float(np.sum(x))
        sum_y = float(np.sum(y))
        sum_xy = float(np.sum(x * y))
        sum_x2 = float(np.sum(x * x))

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return float(slope)

    def update_global_best(self, current_perf: float, epoch: int) -> float:
        """
        更新全局最佳性能（带EMA平滑和重置机制）

        Args:
            current_perf: 当前性能
            epoch: 当前epoch

        Returns:
            平滑后的全局最佳性能
        """
        # 定期重置
        if epoch - self.last_reset_epoch >= self.reset_interval:
            self.global_best_raw = -float("inf")
            self.global_best_smoothed = None
            self.last_reset_epoch = epoch
            self.logger.info(f"[Epoch {epoch}] Global best performance reset")

        # 更新原始全局最佳
        if current_perf > self.global_best_raw:
            old_best = self.global_best_raw
            self.global_best_raw = current_perf
            if old_best > -float("inf"):
                improvement = current_perf - old_best
                self.logger.info(
                    f"[Epoch {epoch}] New global best: {old_best:.2f} -> {current_perf:.2f} "
                    f"(+{improvement:.2f} GFlops)"
                )

        # EMA平滑
        if self.global_best_smoothed is None:
            self.global_best_smoothed = self.global_best_raw
        else:
            smoothing_factor = 0.99
            self.global_best_smoothed = (
                smoothing_factor * self.global_best_smoothed
                + (1 - smoothing_factor) * self.global_best_raw
            )

        return self.global_best_smoothed

    def update(
        self,
        epoch: int,
        current_performance: float,
        entropy_loss: float,
        raw_entropies: Dict[str, float | None],
        current_coefs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        更新所有熵系数（基于趋势）

        Args:
            epoch: 当前epoch
            current_performance: 当前平均性能
            entropy_loss: 当前entropy loss值
            raw_entropies: 各动作空间原始熵值字典
            current_coefs: 当前各系数值（可选）

        Returns:
            更新后的各熵系数字典
        """
        # 更新当前系数值
        if current_coefs is not None:
            self.current_coefs.update(current_coefs)

        # 更新历史记录
        self.performance_history.append(current_performance)
        self.entropy_loss_history.append(entropy_loss)

        # 1. 预热期间保持初始值
        if epoch < self.warmup_epochs:
            self.logger.debug(
                f"[Epoch {epoch}] Warmup phase, keeping initial coefficients"
            )
            self._record_history(
                epoch, current_performance, entropy_loss, self.current_coefs
            )
            return self.current_coefs.copy()

        # 2. 更新全局最佳
        self.update_global_best(current_performance, epoch)

        # 3. 计算趋势速率
        perf_rise_rate = self.compute_trend_rate(
            self.performance_history, self.trend_window_size
        )
        ent_drop_rate = -self.compute_trend_rate(
            self.entropy_loss_history, self.trend_window_size
        )  # 负号表示下降

        # 记录趋势历史
        self.perf_trend_history.append(perf_rise_rate)
        self.ent_loss_trend_history.append(ent_drop_rate)

        # 4. 获取当前配置
        trend_thresholds = self.get_trend_thresholds(epoch)
        stage_strategy = self.get_stage_strategy(epoch)

        # 5. 根据趋势和阶段策略计算调整因子
        adjustment_factors = {}

        for action_type in ["loop_order", "packing_type", "thread_type", "cont"]:
            raw_entropy = raw_entropies.get(action_type)
            if raw_entropy is None:
                adjustment_factors[action_type] = 1.0
                continue

            # 基础调整因子
            base_factor = 1.0
            sensitivity = self.get_action_sensitivity(epoch, action_type)

            # 根据阶段策略调整
            # 获取当前阶段的learning_rate
            current_learning_rate = self.get_learning_rate(epoch)

            adjustment_intensity = stage_strategy.get(
                "adjustment_intensity",
                {
                    "ent_drop_fast": 0.5,
                    "ent_drop_slow": 0.3,
                    "perf_rise_fast": 0.3,
                    "perf_rise_slow": 0.5,
                },
            )

            if stage_strategy["strategy"] == "explore":
                # 探索阶段：entropy下降快时增加探索，performance上升慢时也增加探索
                if ent_drop_rate > trend_thresholds["ent_drop_fast"]:
                    # entropy快速下降，增加探索
                    base_factor += (
                        current_learning_rate
                        * adjustment_intensity["ent_drop_fast"]
                        * stage_strategy["ent_drop_sensitivity"]
                        * sensitivity
                    )
                elif perf_rise_rate < trend_thresholds["perf_rise_slow"]:
                    # performance上升慢，增加探索
                    base_factor += (
                        current_learning_rate
                        * adjustment_intensity["perf_rise_slow"]
                        * stage_strategy["perf_rise_sensitivity"]
                        * sensitivity
                    )
                else:
                    # 其他情况保持不变
                    pass

            elif stage_strategy["strategy"] == "balanced":
                # 平衡阶段：根据趋势动态调整
                if ent_drop_rate > trend_thresholds["ent_drop_fast"]:
                    # entropy快速下降，适度增加探索
                    base_factor += (
                        current_learning_rate
                        * adjustment_intensity["ent_drop_fast"]
                        * stage_strategy["ent_drop_sensitivity"]
                        * sensitivity
                    )
                elif ent_drop_rate < trend_thresholds["ent_drop_slow"]:
                    # entropy下降慢，可以适度减小探索
                    base_factor -= (
                        current_learning_rate
                        * adjustment_intensity["ent_drop_slow"]
                        * stage_strategy["ent_drop_sensitivity"]
                        * sensitivity
                    )
                else:
                    # entropy下降速度适中，保持不变
                    pass

                if perf_rise_rate > trend_thresholds["perf_rise_fast"]:
                    # performance快速上升，适度减小探索
                    base_factor -= (
                        current_learning_rate
                        * adjustment_intensity["perf_rise_fast"]
                        * stage_strategy["perf_rise_sensitivity"]
                        * sensitivity
                    )
                elif perf_rise_rate < trend_thresholds["perf_rise_slow"]:
                    # performance上升慢，适度增加探索
                    base_factor += (
                        current_learning_rate
                        * adjustment_intensity["perf_rise_slow"]
                        * stage_strategy["perf_rise_sensitivity"]
                        * sensitivity
                    )
                else:
                    # performance上升速度适中，保持不变
                    pass

            elif stage_strategy["strategy"] == "exploit":
                # 利用阶段：performance上升快时减小探索，entropy下降慢时也减小探索
                if perf_rise_rate > trend_thresholds["perf_rise_fast"]:
                    # performance快速上升，减小探索
                    base_factor -= (
                        current_learning_rate
                        * adjustment_intensity["perf_rise_fast"]
                        * stage_strategy["perf_rise_sensitivity"]
                        * sensitivity
                    )
                elif ent_drop_rate < trend_thresholds["ent_drop_slow"]:
                    # entropy下降慢，减小探索
                    base_factor -= (
                        current_learning_rate
                        * adjustment_intensity["ent_drop_slow"]
                        * stage_strategy["ent_drop_sensitivity"]
                        * sensitivity
                    )
                else:
                    # 其他情况保持不变
                    pass

            else:
                # 未知策略，保持base_factor不变
                self.logger.warning(
                    f"[Epoch {epoch}] Unknown strategy: {stage_strategy['strategy']}, keeping base_factor unchanged"
                )

            # 应用阶段boost因子
            base_factor *= stage_strategy["coef_boost"]

            # 结合raw_entropy的状态进行微调
            if raw_entropy < 0.1:  # 熵过低，增加探索
                base_factor *= 1.1
            elif raw_entropy > 2.0:  # 熵过高，减少探索
                base_factor *= 0.9
            else:
                # raw_entropy在正常范围内（0.1-2.0），保持不变
                pass

            adjustment_factors[action_type] = max(0.5, min(2.0, base_factor))

        # 6. 更新系数
        new_coefs = {}
        for action_type, current_coef in self.current_coefs.items():
            adjustment = adjustment_factors[action_type]
            new_coef = current_coef * adjustment
            coef_min, coef_max = self.get_coef_range(epoch, action_type)
            new_coef = max(coef_min, min(coef_max, new_coef))
            new_coefs[action_type] = new_coef

        # 7. 记录历史
        self.current_coefs = new_coefs.copy()
        self._record_history(epoch, current_performance, entropy_loss, new_coefs)

        # 8. 日志输出
        if epoch % 10 == 0:  # 每10个epoch记录详细日志
            current_learning_rate = self.get_learning_rate(epoch)
            self.logger.info(
                f"[Epoch {epoch}] Trend-driven entropy adjustment:\n"
                f"  Performance: current={current_performance:.2f}, rise_rate={perf_rise_rate:.4f}\n"
                f"  Entropy loss: current={entropy_loss:.4f}, drop_rate={ent_drop_rate:.4f}\n"
                f"  Strategy: {stage_strategy['strategy']}, learning_rate={current_learning_rate:.4f}, coef_boost={stage_strategy['coef_boost']:.2f}\n"
                f"  Global best: raw={self.global_best_raw:.2f}, smoothed={self.global_best_smoothed:.2f}\n"
                f"  Coefficients: loop_order={new_coefs['loop_order']:.4f}, "
                f"packing_type={new_coefs['packing_type']:.4f}, "
                f"thread_type={new_coefs['thread_type']:.4f}, "
                f"cont={new_coefs['cont']:.4f}"
            )

        return new_coefs.copy()

    def _record_history(
        self,
        epoch: int,
        performance: float,
        entropy_loss: float,
        coefs: Dict[str, float],
    ) -> None:
        """记录历史数据"""
        # 只更新系数历史（performance和entropy_loss历史已在update方法中更新）
        for action_type, coef in coefs.items():
            self.coef_history[action_type].append(coef)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats: Dict[str, Any] = {
            "global_best_raw": self.global_best_raw,
            "global_best_smoothed": self.global_best_smoothed,
            "current_coefs": self.current_coefs.copy(),
            "performance_history_len": len(self.performance_history),
            "entropy_loss_history_len": len(self.entropy_loss_history),
            "recent_perf_trend": self.perf_trend_history[-10:]
            if self.perf_trend_history
            else [],
            "recent_ent_trend": self.ent_loss_trend_history[-10:]
            if self.ent_loss_trend_history
            else [],
        }
        return stats

    def reset(self) -> None:
        """重置控制器状态"""
        self.global_best_raw = -float("inf")
        self.global_best_smoothed = None
        self.last_reset_epoch = 0
        self.performance_history.clear()
        self.entropy_loss_history.clear()
        self.perf_trend_history.clear()
        self.ent_loss_trend_history.clear()
        for k in self.coef_history.keys():
            self.coef_history[k].clear()
        self.logger.info("AdaptiveEntropyControllerByTrend reset")
