# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/4/3 14:30
# Contact       : hucheng.lew@qq.com
# File          : adaptive_entropy_controller_by_performance.py
# IDE           : PyCharm
# Description   : 基于性能的自适应熵系数控制器

import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np


class AdaptiveEntropyControllerByPerformance:
    """
    基于性能的自适应熵系数控制器

    核心特性：
    1. 性能驱动：以GFlops为优化目标
    2. 智能目标：混合目标（固定基准 + 平滑全局最佳）
    3. 差异化调整：四个熵系数独立调整
    4. 稳定性约束：继承loss比例约束机制

    算法流程：
    1. 更新全局最佳性能（EMA平滑）
    2. 计算混合目标：target = max(fixed_target, smoothed_best * best_weight)
    3. 计算性能误差：error = current_perf - target
    4. 根据误差方向和大小调整四个熵系数
    5. 应用loss比例约束（防止熵loss过大）
    6. 返回更新后的系数

    使用示例：
    ```python
    controller = AdaptiveEntropyControllerByPerformance(
        target_type="hybrid",
        fixed_target=60.0,
        global_best_weight=0.8,
        smoothing_factor=0.99,
        min_improvement=2.0,
    )

    # 在训练循环中
    for epoch in range(max_epochs):
        # 获取当前性能和各原始熵值
        current_perf = policy._last_mean_env_performance
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
            raw_entropies=raw_entropies,
            base_loss=policy._last_mean_base_loss,
            ent_loss=policy._last_mean_ent_loss,
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
        target_type: str = "hybrid",
        fixed_target: float = 60.0,
        global_best_weight: float = 0.8,
        smoothing_factor: float = 0.99,
        min_improvement: float = 2.0,
        learning_rate: float = 0.01,
        coef_range: List[Dict[str, Any]] | None = None,
        entropy_loss_ratio_thresholds: List[Dict[str, Any]] | float = 0.5,
        warmup_epochs: int = 10,
        reset_interval: int = 500,
        action_sensitivity: List[Dict[str, Any]] | None = None,
        init_coefs: Dict[str, float] | None = None,
        perf_adaptive_margin: List[Dict[str, Any]] | None = None,
        hybrid_best_weight: List[Dict[str, Any]] | None = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化性能驱动自适应熵系数控制器

        Args:
            target_type: 目标类型
            fixed_target: 固定目标性能
            global_best_weight: 全局最佳权重 (0.0-1.0)
            smoothing_factor: 全局最佳平滑因子
            min_improvement: 最小改进阈值
            learning_rate: 调整学习率
            entropy_loss_ratio_thresholds: 熵loss比例阈值
            warmup_epochs: 预热epoch数
            reset_interval: 全局最佳重置间隔
            logger: 日志记录器
        """
        # 目标配置
        self.target_type = target_type.lower()
        self.fixed_target = fixed_target
        self.global_best_weight = max(0.0, min(1.0, global_best_weight))
        self.smoothing_factor = smoothing_factor
        self.min_improvement = min_improvement

        # 调整参数
        self.learning_rate = learning_rate
        if coef_range is None:
            self.coef_range = [
                {
                    "epoch_range": [0, 9999],
                    "loop_order": [0.001, 0.5],
                    "packing_type": [0.001, 0.5],
                    "thread_type": [0.001, 0.5],
                    "cont": [0.001, 0.5],
                }
            ]
        else:
            self.coef_range = coef_range

        if isinstance(entropy_loss_ratio_thresholds, float):
            self.entropy_loss_ratio_thresholds = [
                {
                    "epoch_range": [0, 9999],
                    "threshold": entropy_loss_ratio_thresholds,
                }
            ]
        else:
            self.entropy_loss_ratio_thresholds = entropy_loss_ratio_thresholds
        self.warmup_epochs = warmup_epochs
        self.reset_interval = reset_interval

        # 状态变量
        self.global_best_raw = -float("inf")
        self.global_best_smoothed = None
        self.last_reset_epoch = 0

        # 各系数当前值
        if init_coefs is None:
            self.current_coefs = {
                "loop_order": 0.03,  # 默认值，后续会更新
                "packing_type": 0.03,
                "thread_type": 0.06,
                "cont": 0.10,
            }
        else:
            self.current_coefs = init_coefs

        # 历史记录
        self.performance_history = []
        self.target_history = []
        self.coef_history = {k: [] for k in self.current_coefs.keys()}

        # 日志
        self.logger = logger or logging.getLogger(
            "AdaptiveEntropyControllerByPerformance"
        )

        # 各动作空间敏感度权重（可配置）
        if action_sensitivity is None:
            self.action_sensitivity = [
                {  # 各个分量在熵中的权重
                    "epoch_range": [0, 9999],
                    "loop_order": 0.15,  # 中等敏感度
                    "packing_type": 0.10,  # 较低敏感度
                    "thread_type": 0.25,  # 高敏感度
                    "cont": 0.50,  # 最高敏感度
                }
            ]
        else:
            self.action_sensitivity = action_sensitivity

        if perf_adaptive_margin is None:
            self.perf_adaptive_margin = [
                {
                    "epoch_range": [0, 40],
                    "margin": 12,
                },
                {
                    "epoch_range": [40, 60],
                    "margin": 8,
                },
                {
                    "epoch_range": [60, 80],
                    "margin": 5,
                },
                {
                    "epoch_range": [80, 9999],
                    "margin": 2,
                }
            ]
        else:
            self.perf_adaptive_margin = perf_adaptive_margin

        if hybrid_best_weight is None:
            self.hybrid_best_weight = [
                {
                    "epoch_range": [0, 50],
                    "weight": 0.3,
                },
                {
                    "epoch_range": [50, 200],
                    "weight": 0.6,
                },
                {
                    "epoch_range": [200, 9999],
                    "weight": 0.8,
                }
            ]
        else:
            self.hybrid_best_weight = hybrid_best_weight

        self.logger.info(
            f"AdaptiveEntropyControllerByPerformance initialized:\n"
            f"  target_type={target_type}, fixed_target={fixed_target:.1f} GFlops\n"
            f"  global_best_weight={global_best_weight:.2f}, smoothing_factor={smoothing_factor:.3f}\n"
            f"  init_coef={self.current_coefs}\n"
            f"  perf_adaptive_margin={self.perf_adaptive_margin }\n"
            f"  coef_range={self.coef_range}\n"
            f"  action_sensitivity={self.action_sensitivity}"
        )

    @staticmethod
    def get_target_datablob_by_epoch(identity: int | float, parameters: List[Dict[str, Any]], range_key: str = "epoch_range") -> Any:
        """
        根据identity获取目标配置
        """
        target_blob = None
        for blob in parameters:
            if blob[range_key][0] <= identity < blob[range_key][1]:
                target_blob = blob
                break
        return target_blob

    def get_coef_range(self, epoch: int, action: str) -> Tuple[float, float]:
        """
        获取系数的最小值和最大值
        """
        blob = self.get_target_datablob_by_epoch(epoch, self.coef_range)
        return blob[action][0], blob[action][1]

    def get_hybrid_best_weight(self, epoch: int) -> float:
        """
        获取权重信息
        """
        default = 0.95
        blob = self.get_target_datablob_by_epoch(epoch, self.hybrid_best_weight)
        best_weight = default if blob is None else blob.get("weight", default)
        return best_weight

    def get_action_sensitivity(self, epoch: int, action: str) -> float:
        """
        获取动作敏感性
        """
        default = 1.0
        blob = self.get_target_datablob_by_epoch(epoch, self.action_sensitivity)
        sensitivity = default if blob is None else blob.get(action, default)
        return sensitivity

    def get_perf_margin(self, perf: float) -> float:
        """
        获取允许的performance漂移
        计算自适应容差范围

        原理：性能越接近极限，容差越小（调整更精细）

        Args:
            current_perf: 当前性能

        Returns:
            容差范围
        """
        default = 0
        blob = self.get_target_datablob_by_epoch(perf, self.perf_adaptive_margin, range_key="perf_range")
        margin = default if blob is None else blob.get("margin", default)
        return margin

    def get_entropy_loss_ratio_threshold(self, epoch: int) -> float:
        """
        获取threshold
        """
        default = 0.5
        blob = self.get_target_datablob_by_epoch(epoch, self.entropy_loss_ratio_thresholds)
        threshold = default if blob is None else blob.get("threshold", default)
        return threshold

    def update_global_best(self, current_perf: float, epoch: int) -> float:
        """
        更新全局最佳性能（带EMA平滑和重置机制）

        Args:
            current_perf: 当前性能
            epoch: 当前epoch

        Returns:
            平滑后的全局最佳性能
        """
        # 定期重置，避免早期峰值锁定
        if epoch - self.last_reset_epoch >= self.reset_interval:
            self.global_best_raw = -float("inf")
            self.global_best_smoothed = None
            self.last_reset_epoch = epoch
            self.logger.info(f"[Epoch {epoch}] Global best performance reset")

        # 更新原始全局最佳（需满足最小改进条件）
        if current_perf > self.global_best_raw + self.min_improvement:
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
            self.global_best_smoothed = (
                self.smoothing_factor * self.global_best_smoothed
                + (1 - self.smoothing_factor) * self.global_best_raw
            )

        return self.global_best_smoothed

    def compute_target_performance(self, epoch: int, current_perf: float) -> float:
        """
        计算目标性能（智能混合目标）

        Args:
            epoch: 当前epoch
            current_perf: 当前性能

        Returns:
            目标性能值
        """
        # 更新全局最佳
        global_best = self.update_global_best(current_perf, epoch)

        # 分阶段策略
        if self.target_type == "fixed":
            target = self.fixed_target

        elif self.target_type == "global_best":
            target = global_best * self.global_best_weight

        elif self.target_type == "hybrid":
            # 动态权重：训练后期更多依赖全局最佳
            best_weight = self.get_hybrid_best_weight(epoch)
            target = max(self.fixed_target, global_best * best_weight)

        else:
            self.logger.warning(f"Unknown target_type: {self.target_type}, using fixed")
            target = self.fixed_target

        return target

    def update(
        self,
        epoch: int,
        current_performance: float,
        raw_entropies: Dict[str, float | None],
        base_loss: Optional[float] = None,
        ent_loss: Optional[float] = None,
        current_coefs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        更新所有熵系数

        Args:
            epoch: 当前epoch
            current_performance: 当前平均性能
            raw_entropies: 各动作空间原始熵值字典
            base_loss: 基础loss值
            ent_loss: 熵loss值
            current_coefs: 当前各系数值（可选）

        Returns:
            更新后的各熵系数字典
        """
        # 更新当前系数值
        if current_coefs is not None:
            self.current_coefs.update(current_coefs)

        # 1. 预热期间保持初始值
        if epoch < self.warmup_epochs:
            self.logger.debug(
                f"[Epoch {epoch}] Warmup phase, keeping initial coefficients"
            )
            self._record_history(epoch, current_performance, self.current_coefs)
            return self.current_coefs.copy()

        # 2. 计算目标性能
        target_performance = self.compute_target_performance(epoch, current_performance)

        # 3. 计算性能误差和容差
        performance_error = current_performance - target_performance
        adaptive_margin = self.get_perf_margin(current_performance)

        # 4. 计算entropy_loss_ratio约束
        loss_ratio_constraint = 1.0
        if base_loss is not None and ent_loss is not None and abs(base_loss) > 1e-6:
            entropy_loss_ratio = abs(ent_loss / base_loss)
            entropy_loss_ratio_threshold = self.get_entropy_loss_ratio_threshold(epoch)
            if entropy_loss_ratio > entropy_loss_ratio_threshold:
                # 熵loss过大，强制降低所有系数
                reduction_factor = (
                        entropy_loss_ratio_threshold / entropy_loss_ratio
                )
                loss_ratio_constraint = min(loss_ratio_constraint, reduction_factor)
                self.logger.warning(
                    f"[Epoch {epoch}] Entropy loss ratio too high ({entropy_loss_ratio:.3f} > {entropy_loss_ratio_threshold:.3f}), "
                    f"applying constraint factor: {reduction_factor:.3f}"
                )

        # 5. 计算各系数的调整因子
        adjustment_factors = {}

        for action_type in ["loop_order", "packing_type", "thread_type", "cont"]:
            raw_entropy = raw_entropies.get(action_type)
            if raw_entropy is None:
                adjustment_factors[action_type] = 1.0
                continue

            # 基础调整因子
            if abs(performance_error) < adaptive_margin:
                # 在容差范围内：微调，维持探索
                if raw_entropy < 0.1:  # 熵过低
                    base_factor = 1.0 + self.learning_rate * 0.5
                elif raw_entropy > 2.0:  # 熵过高
                    base_factor = 1.0 - self.learning_rate * 0.3
                else:
                    base_factor = 1.0

            elif performance_error < -adaptive_margin:
                # 显著低于目标：增加探索
                urgency = min(abs(performance_error) / target_performance, 1.0)
                sensitivity = self.get_action_sensitivity(epoch, action_type)
                base_factor = 1.0 + self.learning_rate * urgency * sensitivity

            else:  # performance_error > adaptive_margin
                # 显著高于目标：减少探索
                urgency = min(performance_error / target_performance, 0.5)
                sensitivity = self.get_action_sensitivity(epoch, action_type)
                base_factor = 1.0 - self.learning_rate * urgency * sensitivity

            # 应用loss比例约束
            adjusted_factor = base_factor * loss_ratio_constraint
            adjustment_factors[action_type] = max(0.5, min(2.0, adjusted_factor))

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
        self._record_history(epoch, current_performance, new_coefs)

        # 8. 日志输出
        if epoch % 10 == 0:  # 每10个epoch记录详细日志
            self.logger.info(
                f"[Epoch {epoch}] Performance-driven entropy adjustment:\n"
                f"  Performance: current={current_performance:.2f}, target={target_performance:.2f}, "
                f"error={performance_error:.2f}, margin={adaptive_margin:.1f}\n"
                f"  Global best: raw={self.global_best_raw:.2f}, smoothed={self.global_best_smoothed:.2f}\n"
                f"  Coefficients: loop_order={new_coefs['loop_order']:.4f}, "
                f"packing_type={new_coefs['packing_type']:.4f}, "
                f"thread_type={new_coefs['thread_type']:.4f}, "
                f"cont={new_coefs['cont']:.4f}"
            )

        return new_coefs.copy()

    def _record_history(
        self, epoch: int, performance: float, coefs: Dict[str, float]
    ) -> None:
        """记录历史数据"""
        self.performance_history.append(performance)
        self.target_history.append(self.compute_target_performance(epoch, performance))
        for action_type, coef in coefs.items():
            self.coef_history[action_type].append(coef)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "global_best_raw": self.global_best_raw,
            "global_best_smoothed": self.global_best_smoothed,
            "current_coefs": self.current_coefs.copy(),
            "performance_history_len": len(self.performance_history),
            "last_performance": self.performance_history[-1]
            if self.performance_history
            else None,
            "last_target": self.target_history[-1] if self.target_history else None,
        }

        # 计算各系数的历史统计
        for action_type, history in self.coef_history.items():
            if history:
                stats[f"{action_type}_mean"] = float(np.mean(history[-100:]))
                stats[f"{action_type}_std"] = float(np.std(history[-100:]))
                stats[f"{action_type}_min"] = float(np.min(history[-100:]))
                stats[f"{action_type}_max"] = float(np.max(history[-100:]))

        return stats

    def reset(self) -> None:
        """重置控制器状态"""
        self.global_best_raw = -float("inf")
        self.global_best_smoothed = None
        self.last_reset_epoch = 0
        self.performance_history.clear()
        self.target_history.clear()
        for key in self.coef_history:
            self.coef_history[key].clear()
        self.logger.info("AdaptiveEntropyControllerByPerformance reset")
