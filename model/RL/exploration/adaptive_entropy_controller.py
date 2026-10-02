# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/4/1 10:57
# Contact       : hucheng.lew@qq.com
# File          : adaptive_entropy_controller.py
# IDE           : PyCharm
# Description   : 自适应熵系数控制器（包含改进版V2，支持loss比例监控）

import logging
from typing import Any, Dict, List, Optional

import numpy as np


class AdaptiveEntropyControllerCont:
    """
    自适应熵系数控制器（连续动作部分）

    基于目标熵自动调节连续动作的熵系数，实现以下功能：
    1. 监控实际熵与目标熵的比值
    2. 动态调整熵系数以维持目标探索水平
    3. 平滑更新，避免剧烈波动
    4. 支持渐进式衰减（训练后期减少探索）

    核心思想：
    - 连续动作空间：高质量区间窄，需要足够的探索来找到最优解
    - 离散动作空间：已经过筛选，组合有限，不需要过多探索

    算法：
    - 计算目标熵：target_entropy = target_ratio * max_entropy
    - 计算熵误差：entropy_error = target_entropy - actual_entropy
    - 更新熵系数：ent_coef *= exp(learning_rate * entropy_error / max_entropy)
    - 裁剪到合理范围：[ent_coef_min, ent_coef_max]

    使用示例：
    ```python
    controller = AdaptiveEntropyControllerCont(
        target_ratio=0.7,
        learning_rate=0.01,
        ent_coef_init=0.01,
        ent_coef_min=0.001,
        ent_coef_max=0.5,
    )

    # 在训练循环中
    for epoch in range(max_epochs):
        # ... 训练代码 ...

        # 获取实际熵和最大熵（连续动作部分）
        actual_entropy = policy._last_mean_cont_entropy
        max_entropy = compute_max_entropy(cont_action_dim, sigma_max)

        # 更新熵系数
        new_ent_coef = controller.update(
            epoch=epoch,
            actual_entropy=actual_entropy,
            max_entropy=max_entropy,
        )

        # 应用新的熵系数（连续动作部分）
        policy.ent_coef_cont = new_ent_coef
    ```
    """

    def __init__(
        self,
        target_ratio: float = 0.7,
        learning_rate: float = 0.01,
        ent_coef_init: float = 0.01,
        ent_coef_min: float = 0.001,
        ent_coef_max: float = 0.5,
        decay_rate: float = 0.999,
        decay_start_epoch: int = 100,
        update_interval: int = 1,
        warmup_epochs: int = 10,
        smoothing_factor: float = 0.9,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化自适应熵系数控制器

        Args:
            target_ratio: 目标熵比例（相对于最大熵），默认0.7表示维持最大熵的70%探索
                         - 较大值（0.8-0.9）：更强的探索，适合复杂环境
                         - 中等值（0.6-0.7）：平衡探索和利用
                         - 较小值（0.4-0.5）：更多利用，适合简单环境
            learning_rate: 熵系数更新学习率，默认0.01
                          - 较大值（0.02-0.05）：快速适应，但可能不稳定
                          - 中等值（0.01）：稳定更新
                          - 较小值（0.001-0.005）：缓慢调整，更稳定
            ent_coef_init: 初始熵系数，默认0.01
            ent_coef_min: 最小熵系数，默认0.001
            ent_coef_max: 最大熵系数，默认0.5
            decay_rate: 熵系数衰减率，默认0.999（每epoch乘以decay_rate）
                       - 1.0: 不衰减
                       - 0.999: 缓慢衰减
                       - 0.995: 中等衰减
            decay_start_epoch: 开始衰减的epoch，默认100
            update_interval: 更新间隔（每隔多少个epoch更新一次），默认1
            warmup_epochs: 预热epoch数，在此期间保持初始值，默认10
            smoothing_factor: 熵值平滑因子，默认0.9（EMA平滑）
                             - 较大值（0.95-0.99）：更平滑，反应慢
                             - 中等值（0.9）：平衡
                             - 较小值（0.7-0.8）：反应快，噪声多
            logger: 日志记录器
        """
        self.target_ratio = target_ratio
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef_init
        self.ent_coef_init = ent_coef_init
        self.ent_coef_min = ent_coef_min
        self.ent_coef_max = ent_coef_max
        self.decay_rate = decay_rate
        self.decay_start_epoch = decay_start_epoch
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.smoothing_factor = smoothing_factor
        self.logger = logger or logging.getLogger("AdaptiveEntropyControllerCont")

        # 状态变量
        self.current_epoch = 0
        self.entropy_history: List[float] = []
        self.ent_coef_history: List[float] = []
        self.smoothed_entropy: Optional[float] = None

        # 统计信息
        self.total_updates = 0
        self.last_adjustment = 0.0

        self.logger.info(
            f"AdaptiveEntropyControllerCont initialized (for continuous actions):\n"
            f"  target_ratio={target_ratio}\n"
            f"  learning_rate={learning_rate}\n"
            f"  ent_coef=[{ent_coef_min}, {ent_coef_init}, {ent_coef_max}]\n"
            f"  decay_rate={decay_rate}\n"
            f"  decay_start_epoch={decay_start_epoch}\n"
            f"  warmup_epochs={warmup_epochs}\n"
            f"  smoothing_factor={smoothing_factor}"
        )

    def update(
        self,
        epoch: int,
        actual_entropy: Optional[float] = None,
        max_entropy: Optional[float] = None,
    ) -> float:
        """
        更新熵系数

        Args:
            epoch: 当前epoch
            actual_entropy: 实际观察到的平均熵（连续动作部分）
            max_entropy: 最大可能熵（连续动作部分）

        Returns:
            更新后的熵系数
        """
        self.current_epoch = epoch

        # 1. 预热期间保持初始值
        if epoch < self.warmup_epochs:
            self.logger.debug(
                f"[Epoch {epoch}] Warmup phase, ent_coef={self.ent_coef:.6f}"
            )
            self.ent_coef_history.append(self.ent_coef)
            return self.ent_coef

        # 2. 检查是否需要更新（根据update_interval）
        if epoch % self.update_interval != 0:
            self.ent_coef_history.append(self.ent_coef)
            return self.ent_coef

        # 3. 如果没有提供熵信息，无法自适应更新，使用衰减策略
        if actual_entropy is None or max_entropy is None:
            if epoch >= self.decay_start_epoch:
                old_coef = self.ent_coef
                self.ent_coef = max(self.ent_coef_min, self.ent_coef * self.decay_rate)
                self.logger.debug(
                    f"[Epoch {epoch}] Decay mode (no entropy info): "
                    f"ent_coef {old_coef:.6f} -> {self.ent_coef:.6f}"
                )
            self.ent_coef_history.append(self.ent_coef)
            return self.ent_coef

        # 4. 平滑实际熵值（EMA）
        if self.smoothed_entropy is None:
            self.smoothed_entropy = actual_entropy
        else:
            self.smoothed_entropy = (
                self.smoothing_factor * self.smoothed_entropy
                + (1 - self.smoothing_factor) * actual_entropy
            )

        # 使用平滑后的熵值进行更新
        smoothed_entropy = self.smoothed_entropy

        # 5. 自适应更新：基于目标熵
        target_entropy = self.target_ratio * max_entropy
        entropy_error = target_entropy - smoothed_entropy

        # 记录熵历史
        self.entropy_history.append(actual_entropy)

        # 6. 计算熵系数调整量
        # 使用指数更新规则：coef *= exp(lr * error / max_entropy)
        # 这样可以保持更新幅度在合理范围内
        adjustment = np.exp(self.learning_rate * entropy_error / max_entropy)
        old_coef = self.ent_coef
        self.ent_coef *= adjustment
        self.last_adjustment = self.ent_coef - old_coef

        # 7. 裁剪到合理范围
        self.ent_coef = np.clip(self.ent_coef, self.ent_coef_min, self.ent_coef_max)

        # 8. 应用衰减（训练后期减少探索）
        if epoch >= self.decay_start_epoch:
            self.ent_coef *= self.decay_rate
            self.ent_coef = max(self.ent_coef_min, self.ent_coef)

        # 9. 记录日志
        self.total_updates += 1
        self.logger.info(
            f"[Epoch {epoch}] Entropy adaptation:\n"
            f"  actual_entropy={actual_entropy:.4f}\n"
            f"  smoothed_entropy={smoothed_entropy:.4f}\n"
            f"  max_entropy={max_entropy:.4f}\n"
            f"  target_entropy={target_entropy:.4f}\n"
            f"  error={entropy_error:.4f}\n"
            f"  ent_coef {old_coef:.6f} -> {self.ent_coef:.6f}"
        )

        self.ent_coef_history.append(self.ent_coef)
        return self.ent_coef

    def get_entropy_statistics(self) -> Dict[str, Any]:
        """
        获取熵统计信息

        Returns:
            包含熵历史和熵系数历史的字典
        """
        if not self.entropy_history:
            return {
                "mean_entropy": None,
                "std_entropy": None,
                "min_entropy": None,
                "max_entropy": None,
                "mean_ent_coef": None,
                "current_ent_coef": self.ent_coef,
                "total_updates": self.total_updates,
                "current_epoch": self.current_epoch,
            }

        entropy_array = np.array(self.entropy_history)
        ent_coef_array = np.array(self.ent_coef_history)

        return {
            "mean_entropy": float(np.mean(entropy_array)),
            "std_entropy": float(np.std(entropy_array)),
            "min_entropy": float(np.min(entropy_array)),
            "max_entropy": float(np.max(entropy_array)),
            "mean_ent_coef": float(np.mean(ent_coef_array)),
            "current_ent_coef": self.ent_coef,
            "entropy_history_len": len(self.entropy_history),
            "ent_coef_history_len": len(self.ent_coef_history),
            "total_updates": self.total_updates,
            "current_epoch": self.current_epoch,
            "smoothed_entropy": self.smoothed_entropy,
            "last_adjustment": self.last_adjustment,
        }

    def reset(self):
        """
        重置控制器状态
        """
        self.ent_coef = self.ent_coef_init
        self.entropy_history = []
        self.ent_coef_history = []
        self.smoothed_entropy = None
        self.total_updates = 0
        self.last_adjustment = 0.0
        self.current_epoch = 0
        self.logger.info("AdaptiveEntropyControllerCont reset to initial state.")

    def set_target_ratio(self, target_ratio: float):
        """
        动态设置目标熵比例

        Args:
            target_ratio: 新的目标熵比例
        """
        if not 0.0 < target_ratio <= 1.0:
            self.logger.warning(
                f"Invalid target_ratio={target_ratio}, should be in (0, 1]. Ignoring."
            )
            return

        old_ratio = self.target_ratio
        self.target_ratio = target_ratio
        self.logger.info(f"Target ratio changed: {old_ratio:.3f} -> {target_ratio:.3f}")

    def set_learning_rate(self, learning_rate: float):
        """
        动态设置学习率

        Args:
            learning_rate: 新的学习率
        """
        if learning_rate <= 0:
            self.logger.warning(
                f"Invalid learning_rate={learning_rate}, should be > 0. Ignoring."
            )
            return

        old_lr = self.learning_rate
        self.learning_rate = learning_rate
        self.logger.info(f"Learning rate changed: {old_lr:.6f} -> {learning_rate:.6f}")

    def force_ent_coef(self, ent_coef: float):
        """
        强制设置熵系数（覆盖自适应逻辑）

        Args:
            ent_coef: 新的熵系数
        """
        ent_coef = np.clip(ent_coef, self.ent_coef_min, self.ent_coef_max)
        old_coef = self.ent_coef
        self.ent_coef = ent_coef
        self.logger.info(
            f"Entropy coefficient forced: {old_coef:.6f} -> {ent_coef:.6f}"
        )


def compute_max_entropy_for_gaussian(
    action_dim: int,
    sigma_max: float,
) -> float:
    """
    计算高斯分布的最大熵

    对于高斯分布 N(mu, sigma^2)，熵为：
    H = 0.5 * ln(2 * pi * e * sigma^2)

    对于多维独立高斯分布：
    H = sum(0.5 * ln(2 * pi * e * sigma_i^2))

    Args:
        action_dim: 动作维度
        sigma_max: 最大标准差

    Returns:
        最大熵值
    """
    # 单维度高斯分布的熵
    single_dim_entropy = 0.5 * np.log(2 * np.pi * np.e * sigma_max**2)

    # 多维度独立高斯分布的总熵
    max_entropy = single_dim_entropy * action_dim

    return max_entropy


class AdaptiveEntropyControllerContV2:
    """
    改进版自适应熵系数控制器（连续动作部分）- V2版本

    【核心改进】：同时监控熵比例和loss比例，优先保证训练稳定性

    基于双目标自动调节连续动作的熵系数：
    1. **主要目标**：控制entropy_loss_ratio（熵loss占总loss的比例）
       - 确保训练稳定性，避免熵loss支配总loss
       - 目标：entropy_loss_ratio应该在合理范围内（如0.2-0.5）

    2. **次要目标**：控制entropy_ratio（实际熵/最大熵）
       - 保证足够的探索强度
       - 目标：维持合理的探索水平

    3. **优先级策略**：
       - 当entropy_loss_ratio过高时，优先降低系数（即使熵比例合理）
       - 当entropy_loss_ratio合理时，再调整熵比例
       - 这确保了训练稳定性优先于探索强度

    算法：
    ```
    1. 计算entropy_loss_ratio_error = target_loss_ratio - actual_loss_ratio
    2. 如果entropy_loss_ratio偏离目标：
       - 使用loss_ratio调整：ent_coef *= exp(lr * loss_ratio_error)
    3. 否则（loss_ratio合理）：
       - 使用entropy_ratio调整：ent_coef *= exp(lr * entropy_error / max_entropy)
    4. 裁剪到合理范围：[ent_coef_min, ent_coef_max]
    ```

    使用示例：
    ```python
    controller = AdaptiveEntropyControllerContV2(
        target_entropy_ratio=0.5,
        target_entropy_loss_ratio=0.35,
        learning_rate=0.02,
        ent_coef_init=0.001,
        ent_coef_min=0.0005,
        ent_coef_max=0.002,
    )

    # 在训练循环中
    for epoch in range(max_epochs):
        # ... 训练代码 ...

        # 获取必要信息
        actual_entropy = policy._last_mean_cont_entropy
        max_entropy = compute_max_entropy(cont_action_dim, sigma_max)
        entropy_loss_ratio = current_entropy_loss_ratio  # 从训练stats获取

        # 更新熵系数
        new_ent_coef = controller.update(
            epoch=epoch,
            actual_entropy=actual_entropy,
            max_entropy=max_entropy,
            entropy_loss_ratio=entropy_loss_ratio,
        )

        # 应用新的熵系数
        policy.ent_coef_cont = new_ent_coef
    ```
    """

    def __init__(
        self,
        target_entropy_ratio: float = 0.5,
        target_entropy_loss_ratio: float = 0.35,
        learning_rate: float = 0.02,
        ent_coef_init: float = 0.001,
        ent_coef_min: float = 0.0005,
        ent_coef_max: float = 0.002,
        decay_rate: float = 0.9992,
        decay_start_epoch: int = 150,
        update_interval: int = 1,
        warmup_epochs: int = 3,
        smoothing_factor: float = 0.9,
        loss_ratio_priority_threshold: float = 0.5,  # loss_ratio超过此值时，优先调整
        base_loss_min_threshold: float = 0.01,  # base_loss最小阈值，避免ratio异常
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化改进版自适应熵系数控制器

        Args:
            target_entropy_ratio: 目标熵比例（相对于最大熵），默认0.5
                                 - 控制探索强度
                                 - 建议范围：0.4-0.6

            target_entropy_loss_ratio: 目标熵loss比例，默认0.35
                                      - 控制训练稳定性
                                      - 建议范围：0.2-0.5
                                      - 过高会导致熵loss支配总loss
                                      - 过低会限制探索

            learning_rate: 熵系数更新学习率，默认0.02
                          - 建议范围：0.01-0.05
                          - V2版本建议稍高，以便快速响应loss比例变化

            ent_coef_init: 初始熵系数，默认0.001
                          - 建议从较低值开始，避免初期熵loss过大

            ent_coef_min: 最小熵系数，默认0.0005

            ent_coef_max: 最大熵系数，默认0.002
                          - 严格限制上限，防止熵系数过高

            decay_rate: 熵系数衰减率，默认0.9992

            decay_start_epoch: 开始衰减的epoch，默认150

            update_interval: 更新间隔，默认1（每个epoch都更新）

            warmup_epochs: 预热epoch数，默认3
                          - 缩短warmup，让控制器更早介入

            smoothing_factor: 熵值平滑因子，默认0.9

            loss_ratio_priority_threshold: loss比例优先级阈值，默认0.5
                                          - 当entropy_loss_ratio超过此值时，
                                            优先调整loss比例，忽略熵比例

            logger: 日志记录器
        """
        # 目标参数
        self.target_entropy_ratio = target_entropy_ratio
        self.target_entropy_loss_ratio = target_entropy_loss_ratio

        # 学习率和系数范围
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef_init
        self.ent_coef_init = ent_coef_init
        self.ent_coef_min = ent_coef_min
        self.ent_coef_max = ent_coef_max

        # 衰减参数
        self.decay_rate = decay_rate
        self.decay_start_epoch = decay_start_epoch

        # 更新策略参数
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.smoothing_factor = smoothing_factor
        self.loss_ratio_priority_threshold = loss_ratio_priority_threshold
        self.base_loss_min_threshold = base_loss_min_threshold

        # 日志
        self.logger = logger or logging.getLogger("AdaptiveEntropyControllerContV2")

        # 状态变量
        self.current_epoch = 0
        self.entropy_history: List[float] = []
        self.ent_coef_history: List[float] = []
        self.entropy_loss_ratio_history: List[float] = []
        self.smoothed_entropy: Optional[float] = None
        self.smoothed_loss_ratio: Optional[float] = None

        # 统计信息
        self.total_updates = 0
        self.last_adjustment = 0.0
        self.last_adjustment_mode = "none"  # "loss_ratio" or "entropy_ratio"

        self.logger.info(
            f"AdaptiveEntropyControllerContV2 initialized:\n"
            f"  target_entropy_ratio={target_entropy_ratio}\n"
            f"  target_entropy_loss_ratio={target_entropy_loss_ratio}\n"
            f"  learning_rate={learning_rate}\n"
            f"  ent_coef=[{ent_coef_min}, {ent_coef_init}, {ent_coef_max}]\n"
            f"  loss_ratio_priority_threshold={loss_ratio_priority_threshold}\n"
            f"  decay_rate={decay_rate}\n"
            f"  decay_start_epoch={decay_start_epoch}\n"
            f"  warmup_epochs={warmup_epochs}"
        )

    def update(
        self,
        epoch: int,
        actual_entropy: Optional[float] = None,
        max_entropy: Optional[float] = None,
        base_loss: Optional[float] = None,
        ent_loss: Optional[float] = None,
    ) -> float:
        """
        更新熵系数（改进版，支持loss比例监控）

        Args:
            epoch: 当前epoch
            actual_entropy: 实际观察到的平均熵（连续动作部分）
            max_entropy: 最大可能熵（连续动作部分）
            base_loss: 基础loss（clip_loss + vf_coef * vf_loss）
            ent_loss: 熵loss

        Returns:
            更新后的熵系数
        """
        self.current_epoch = epoch

        # 0. 计算entropy_loss_ratio（在函数内部计算，而不是外部传入）
        entropy_loss_ratio = None
        if base_loss is not None and ent_loss is not None:
            # 使用初始化时设置的base_loss最小阈值，避免因base_loss过小导致ratio异常
            if abs(base_loss) > self.base_loss_min_threshold:
                # 只有当base_loss不是异常小的时候，才计算entropy_loss_ratio
                entropy_loss_ratio = abs(ent_loss / base_loss)
                if self.logger:
                    self.logger.debug(
                        f"[Epoch {epoch}] Calculated entropy_loss_ratio: "
                        f"ent_loss={ent_loss:.6f}, base_loss={base_loss:.6f}, ratio={entropy_loss_ratio:.4f}"
                    )
            else:
                # base_loss异常小，放弃使用entropy_loss_ratio进行调整
                if self.logger:
                    self.logger.warning(
                        f"[Epoch {epoch}] Base loss is too small ({base_loss:.6f} < threshold {self.base_loss_min_threshold:.6f}), "
                        f"skip entropy_loss_ratio adjustment"
                    )

        # 1. 预热期间保持初始值
        if epoch < self.warmup_epochs:
            self.logger.debug(
                f"[Epoch {epoch}] Warmup phase, ent_coef={self.ent_coef:.6f}"
            )
            self.ent_coef_history.append(self.ent_coef)
            return self.ent_coef

        # 2. 检查是否需要更新（根据update_interval）
        if epoch % self.update_interval != 0:
            self.ent_coef_history.append(self.ent_coef)
            return self.ent_coef

        # 3. 如果没有提供任何信息，使用衰减策略
        if actual_entropy is None or max_entropy is None:
            if epoch >= self.decay_start_epoch:
                old_coef = self.ent_coef
                self.ent_coef = max(self.ent_coef_min, self.ent_coef * self.decay_rate)
                self.logger.debug(
                    f"[Epoch {epoch}] Decay mode (no entropy info): "
                    f"ent_coef {old_coef:.6f} -> {self.ent_coef:.6f}"
                )
            self.ent_coef_history.append(self.ent_coef)
            return self.ent_coef

        # 4. 平滑实际熵值和loss比例（EMA）
        if self.smoothed_entropy is None:
            self.smoothed_entropy = actual_entropy
        else:
            self.smoothed_entropy = (
                self.smoothing_factor * self.smoothed_entropy
                + (1 - self.smoothing_factor) * actual_entropy
            )

        if entropy_loss_ratio is not None:
            if self.smoothed_loss_ratio is None:
                self.smoothed_loss_ratio = entropy_loss_ratio
            else:
                self.smoothed_loss_ratio = (
                    self.smoothing_factor * self.smoothed_loss_ratio
                    + (1 - self.smoothing_factor) * entropy_loss_ratio
                )

        # 记录历史
        self.entropy_history.append(actual_entropy)
        if entropy_loss_ratio is not None:
            self.entropy_loss_ratio_history.append(entropy_loss_ratio)

        # 5. 双目标调整策略
        old_coef = self.ent_coef
        adjustment_mode = "entropy_ratio"  # 默认使用熵比例调整

        # 情况1：entropy_loss_ratio过高（优先处理）
        if entropy_loss_ratio is not None and self.smoothed_loss_ratio is not None:
            loss_ratio_error = self.target_entropy_loss_ratio - self.smoothed_loss_ratio

            # 如果loss_ratio偏离目标较大（绝对值），优先调整
            if (
                abs(loss_ratio_error) > 0.05
                or self.smoothed_loss_ratio > self.loss_ratio_priority_threshold
            ):
                # 使用loss比例调整
                # entropy_loss_ratio过高 -> loss_ratio_error < 0 -> 降低系数
                # entropy_loss_ratio过低 -> loss_ratio_error > 0 -> 增加系数
                adjustment = np.exp(self.learning_rate * loss_ratio_error)
                self.ent_coef *= adjustment
                adjustment_mode = "loss_ratio"

                self.logger.info(
                    f"[Epoch {epoch}] Loss ratio adjustment (priority):\n"
                    f"  entropy_loss_ratio={self.smoothed_loss_ratio:.4f}\n"
                    f"  target_loss_ratio={self.target_entropy_loss_ratio:.4f}\n"
                    f"  error={loss_ratio_error:.4f}\n"
                    f"  ent_coef {old_coef:.6f} -> {self.ent_coef:.6f}"
                )

        # 情况2：loss比例合理，使用熵比例调整
        if adjustment_mode == "entropy_ratio":
            target_entropy = self.target_entropy_ratio * max_entropy
            entropy_error = target_entropy - self.smoothed_entropy

            # 使用指数更新规则
            adjustment = np.exp(self.learning_rate * entropy_error / max_entropy)
            self.ent_coef *= adjustment

            self.logger.debug(
                f"[Epoch {epoch}] Entropy ratio adjustment:\n"
                f"  actual_entropy={actual_entropy:.4f}\n"
                f"  smoothed_entropy={self.smoothed_entropy:.4f}\n"
                f"  max_entropy={max_entropy:.4f}\n"
                f"  target_entropy={target_entropy:.4f}\n"
                f"  error={entropy_error:.4f}\n"
                f"  ent_coef {old_coef:.6f} -> {self.ent_coef:.6f}"
            )

        # 6. 裁剪到合理范围
        self.ent_coef = np.clip(self.ent_coef, self.ent_coef_min, self.ent_coef_max)
        self.last_adjustment = self.ent_coef - old_coef
        self.last_adjustment_mode = adjustment_mode

        # 7. 应用衰减（训练后期减少探索）
        if epoch >= self.decay_start_epoch:
            self.ent_coef *= self.decay_rate
            self.ent_coef = max(self.ent_coef_min, self.ent_coef)

        # 8. 记录统计信息
        self.total_updates += 1
        if self.total_updates % 10 == 0:  # 每10次更新记录一次完整日志
            if self.smoothed_loss_ratio:
                self.logger.info(
                    f"[Epoch {epoch}] Entropy adaptation summary:\n"
                    f"  mode={adjustment_mode}\n"
                    f"  ent_coef={self.ent_coef:.6f}\n"
                    f"  entropy_ratio={self.smoothed_entropy / max_entropy:.4f}\n"
                    f"  loss_ratio={self.smoothed_loss_ratio:.4f}"
                )
            else:
                self.logger.info(
                    f"[Epoch {epoch}] Entropy adaptation summary:\n"
                    f"  mode={adjustment_mode}\n"
                    f"  ent_coef={self.ent_coef:.6f}\n"
                    f"  entropy_ratio={self.smoothed_entropy / max_entropy:.4f}\n"
                    f"  loss_ratio=N/A"
                )

        self.ent_coef_history.append(self.ent_coef)
        return self.ent_coef

    def get_entropy_statistics(self) -> Dict[str, Any]:
        """
        获取熵统计信息
        """
        if not self.entropy_history:
            return {
                "mean_entropy": None,
                "std_entropy": None,
                "min_entropy": None,
                "max_entropy": None,
                "mean_ent_coef": None,
                "current_ent_coef": self.ent_coef,
                "mean_entropy_loss_ratio": None,
                "current_entropy_loss_ratio": self.smoothed_loss_ratio,
                "total_updates": self.total_updates,
                "current_epoch": self.current_epoch,
                "last_adjustment_mode": self.last_adjustment_mode,
            }

        entropy_array = np.array(self.entropy_history)
        ent_coef_array = np.array(self.ent_coef_history)

        stats = {
            "mean_entropy": float(np.mean(entropy_array)),
            "std_entropy": float(np.std(entropy_array)),
            "min_entropy": float(np.min(entropy_array)),
            "max_entropy": float(np.max(entropy_array)),
            "mean_ent_coef": float(np.mean(ent_coef_array)),
            "current_ent_coef": self.ent_coef,
            "entropy_history_len": len(self.entropy_history),
            "ent_coef_history_len": len(self.ent_coef_history),
            "total_updates": self.total_updates,
            "current_epoch": self.current_epoch,
            "smoothed_entropy": self.smoothed_entropy,
            "last_adjustment": self.last_adjustment,
            "last_adjustment_mode": self.last_adjustment_mode,
        }

        if self.entropy_loss_ratio_history:
            loss_ratio_array = np.array(self.entropy_loss_ratio_history)
            stats.update(
                {
                    "mean_entropy_loss_ratio": float(np.mean(loss_ratio_array)),
                    "std_entropy_loss_ratio": float(np.std(loss_ratio_array)),
                    "min_entropy_loss_ratio": float(np.min(loss_ratio_array)),
                    "max_entropy_loss_ratio": float(np.max(loss_ratio_array)),
                    "current_entropy_loss_ratio": self.smoothed_loss_ratio,
                }
            )

        return stats

    def reset(self):
        """重置控制器状态"""
        self.ent_coef = self.ent_coef_init
        self.entropy_history = []
        self.ent_coef_history = []
        self.entropy_loss_ratio_history = []
        self.smoothed_entropy = None
        self.smoothed_loss_ratio = None
        self.total_updates = 0
        self.last_adjustment = 0.0
        self.last_adjustment_mode = "none"
        self.current_epoch = 0
        self.logger.info("AdaptiveEntropyControllerContV2 reset to initial state.")

    def set_target_entropy_ratio(self, target_entropy_ratio: float):
        """动态设置目标熵比例"""
        if not 0.0 < target_entropy_ratio <= 1.0:
            self.logger.warning(
                f"Invalid target_entropy_ratio={target_entropy_ratio}, "
                f"should be in (0, 1]. Ignoring."
            )
            return

        old_ratio = self.target_entropy_ratio
        self.target_entropy_ratio = target_entropy_ratio
        self.logger.info(
            f"Target entropy ratio changed: {old_ratio:.3f} -> {target_entropy_ratio:.3f}"
        )

    def set_target_entropy_loss_ratio(self, target_entropy_loss_ratio: float):
        """动态设置目标熵loss比例"""
        if not 0.0 <= target_entropy_loss_ratio <= 2.0:
            self.logger.warning(
                f"Invalid target_entropy_loss_ratio={target_entropy_loss_ratio}, "
                f"should be in [0, 2.0]. Ignoring."
            )
            return

        old_ratio = self.target_entropy_loss_ratio
        self.target_entropy_loss_ratio = target_entropy_loss_ratio
        self.logger.info(
            f"Target entropy loss ratio changed: {old_ratio:.3f} -> {target_entropy_loss_ratio:.3f}"
        )

    def set_learning_rate(self, learning_rate: float):
        """动态设置学习率"""
        if learning_rate <= 0:
            self.logger.warning(
                f"Invalid learning_rate={learning_rate}, should be > 0. Ignoring."
            )
            return

        old_lr = self.learning_rate
        self.learning_rate = learning_rate
        self.logger.info(f"Learning rate changed: {old_lr:.6f} -> {learning_rate:.6f}")

    def force_ent_coef(self, ent_coef: float):
        """强制设置熵系数（覆盖自适应逻辑）"""
        ent_coef = np.clip(ent_coef, self.ent_coef_min, self.ent_coef_max)
        old_coef = self.ent_coef
        self.ent_coef = ent_coef
        self.logger.info(
            f"Entropy coefficient forced: {old_coef:.6f} -> {ent_coef:.6f}"
        )


class AdaptiveEntropyControllerDisc:
    """
    自适应熵系数控制器（离散动作部分）

    【核心思想】：
    基于双目标自动调节离散动作的熵系数：
    1. **主要目标1**：控制entropy_loss_ratio（熵loss占总loss的比例）
       - 确保训练稳定性，避免熵loss支配总loss
       - 目标：entropy_loss_ratio应该在合理范围内（如0.2-0.5）
    2. **主要目标2**：控制原始熵值
       - 确保探索充分，避免策略过早收敛
       - 目标：原始熵应该保持在合理范围内（如0.3-1.0）

    【算法逻辑】：
    - 当entropy_loss_ratio > 0.5时，降低熵系数（避免熵loss过高）
    - 当entropy_loss_ratio < 0.2时，增加熵系数（增加探索）
    - 当原始熵 < 0.3时，增加熵系数（避免过早收敛）
    - 当原始熵 > 1.0时，适当降低熵系数（避免过度探索）

    【适用场景】：
    - 离散动作空间（如thread_type, loop_order, packing_type）
    - 初期需要充分探索，后期允许收敛
    - 需要平衡探索和利用

    【使用示例】：
    ```python
    controller = AdaptiveEntropyControllerDisc(
        target_entropy_min=0.3,
        target_entropy_max=1.0,
        target_entropy_loss_ratio_min=0.2,
        target_entropy_loss_ratio_max=0.5,
        learning_rate=0.02,
        ent_coef_init=0.15,
        ent_coef_min=0.05,
        ent_coef_max=0.5,
        max_entropy=1.386,  # ln(4) for 4 thread types
    )

    # 在训练循环中
    for epoch in range(max_epochs):
        # ... 训练代码 ...

        # 获取原始熵和base_loss、ent_loss
        actual_entropy = policy._last_mean_raw_entropy_disc_thread_type
        base_loss = policy._last_mean_base_loss
        ent_loss = policy._last_mean_ent_loss

        # 更新熵系数
        new_ent_coef = controller.update(
            epoch=epoch,
            actual_entropy=actual_entropy,
            base_loss=base_loss,
            ent_loss=ent_loss,
        )

        # 应用新的熵系数
        policy._ent_coef_disc_thread_type = new_ent_coef
    ```
    """

    def __init__(
        self,
        target_entropy_min: float = 0.3,
        target_entropy_max: float = 1.0,
        target_entropy_loss_ratio_min: float = 0.2,
        target_entropy_loss_ratio_max: float = 0.5,
        learning_rate: float = 0.02,
        ent_coef_init: float = 0.15,
        ent_coef_min: float = 0.05,
        ent_coef_max: float = 0.5,
        max_entropy: float = 1.386,  # ln(num_actions)
        priority_threshold: float = 0.1,  # 当熵低于此值时优先级提高
        smoothing_factor: float = 0.9,
        warmup_epochs: int = 3,  # 预热epoch数
        decay_rate: float = 0.9995,  # 衰减率
        decay_start_epoch: int = 150,  # 开始衰减的epoch
        base_loss_min_threshold: float = 0.01,  # base_loss最小阈值，避免ratio异常
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化离散动作自适应熵系数控制器

        Args:
            target_entropy_min: 目标熵最小值，低于此值会增加系数
            target_entropy_max: 目标熵最大值，高于此值会降低系数
            target_entropy_loss_ratio_min: 目标entropy_loss_ratio最小值
            target_entropy_loss_ratio_max: 目标entropy_loss_ratio最大值
            learning_rate: 熵系数更新学习率
            ent_coef_init: 初始熵系数
            ent_coef_min: 最小熵系数
            ent_coef_max: 最大熵系数
            max_entropy: 最大可能熵值（对于num_actions个选项，max_entropy = ln(num_actions)）
            priority_threshold: 熵优先阈值，低于此值时优先处理
            smoothing_factor: 平滑因子（用于EMA）
            warmup_epochs: 预热epoch数，在此期间保持初始熵系数不变
            decay_rate: 熵系数衰减率，训练后期逐渐降低探索
            decay_start_epoch: 开始衰减的epoch
            logger: 日志记录器
        """
        self.target_entropy_min = target_entropy_min
        self.target_entropy_max = target_entropy_max
        self.target_entropy_loss_ratio_min = target_entropy_loss_ratio_min
        self.target_entropy_loss_ratio_max = target_entropy_loss_ratio_max
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef_init
        self.ent_coef_min = ent_coef_min
        self.ent_coef_max = ent_coef_max
        self.max_entropy = max_entropy
        self.priority_threshold = priority_threshold
        self.smoothing_factor = smoothing_factor
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.decay_start_epoch = decay_start_epoch
        self.base_loss_min_threshold = base_loss_min_threshold
        self.logger = logger

        # 历史记录（用于平滑和监控）
        self._entropy_history = []
        self._entropy_loss_ratio_history = []
        self._ent_coef_history = []

        # 统计信息
        self._update_count = 0
        self._last_entropy = None
        self._last_entropy_loss_ratio = None

        if self.logger:
            self.logger.info(
                f"AdaptiveEntropyControllerDisc initialized:\n"
                f"  target_entropy_range=[{target_entropy_min}, {target_entropy_max}]\n"
                f"  target_entropy_loss_ratio_range=[{target_entropy_loss_ratio_min}, {target_entropy_loss_ratio_max}]\n"
                f"  ent_coef_init={ent_coef_init}, range=[{ent_coef_min}, {ent_coef_max}]\n"
                f"  max_entropy={max_entropy}, learning_rate={learning_rate}\n"
                f"  warmup_epochs={warmup_epochs}, decay_rate={decay_rate}, decay_start_epoch={decay_start_epoch}"
            )

    def update(
        self,
        epoch: int,
        actual_entropy: float,
        base_loss: Optional[float] = None,
        ent_loss: Optional[float] = None,
    ) -> float:
        """
        更新熵系数

        Args:
            epoch: 当前epoch
            actual_entropy: 实际原始熵值
            base_loss: 基础loss（clip_loss + vf_coef * vf_loss）
            ent_loss: 熵loss

        Returns:
            新的熵系数
        """
        self._update_count += 1

        # 1. 预热期间保持初始值
        if epoch < self.warmup_epochs:
            if self.logger and self._update_count % 10 == 0:
                self.logger.debug(
                    f"[Epoch {epoch}] Warmup phase for discrete entropy, "
                    f"ent_coef={self.ent_coef:.6f}"
                )
            self._ent_coef_history.append(self.ent_coef)
            return self.ent_coef

        # 2. 如果没有提供熵值信息，应用衰减策略
        if actual_entropy is None:
            if epoch >= self.decay_start_epoch:
                old_coef = self.ent_coef
                self.ent_coef = max(self.ent_coef_min, self.ent_coef * self.decay_rate)
                if self.logger:
                    self.logger.debug(
                        f"[Epoch {epoch}] Decay mode (no entropy info) for discrete: "
                        f"ent_coef {old_coef:.6f} -> {self.ent_coef:.6f}"
                    )
            self._ent_coef_history.append(self.ent_coef)
            return self.ent_coef


        # 3 计算entropy_loss_ratio
        entropy_loss_ratio = None
        if base_loss is not None and ent_loss is not None:
            # 使用初始化时设置的base_loss最小阈值，避免因base_loss过小导致ratio异常
            if abs(base_loss) > self.base_loss_min_threshold:
                # 只有当base_loss不是异常小的时候，才计算entropy_loss_ratio
                entropy_loss_ratio = abs(ent_loss / base_loss)
                if self.logger:
                    self.logger.debug(
                        f"[Epoch {epoch}] Calculated entropy_loss_ratio: "
                        f"ent_loss={ent_loss:.6f}, base_loss={base_loss:.6f}, ratio={entropy_loss_ratio:.4f}"
                    )
            else:
                # base_loss异常小，放弃使用entropy_loss_ratio进行调整
                if self.logger:
                    self.logger.warning(
                        f"[Epoch {epoch}] Base loss is too small ({base_loss:.6f} < threshold {self.base_loss_min_threshold:.6f}), "
                        f"skip entropy_loss_ratio adjustment"
                    )


        # 4 记录历史
        self._last_entropy = actual_entropy
        self._last_entropy_loss_ratio = entropy_loss_ratio
        self._entropy_history.append(actual_entropy)
        if entropy_loss_ratio is not None:
            self._entropy_loss_ratio_history.append(entropy_loss_ratio)

        # 5 计算调整因子
        entropy_factor = 1.0
        loss_ratio_factor = 1.0

        # 5.1. 基于原始熵值的调整
        if actual_entropy < self.target_entropy_min:
            # 熵过低，需要增加系数
            urgency = (self.target_entropy_min - actual_entropy) / self.max_entropy
            # 当熵极低时（低于priority_threshold），更激进地增加
            if actual_entropy < self.priority_threshold:
                urgency *= 2.0
            entropy_factor = 1.0 + self.learning_rate * urgency
        elif actual_entropy > self.target_entropy_max:
            # 熵过高，适当降低系数
            urgency = (actual_entropy - self.target_entropy_max) / self.max_entropy
            entropy_factor = 1.0 - self.learning_rate * urgency * 0.5
        else:
            # 熵在合理范围内，保持稳定
            entropy_factor = 1.0

        # 5.2. 基于entropy_loss_ratio的调整
        if entropy_loss_ratio is not None:
            if entropy_loss_ratio > self.target_entropy_loss_ratio_max:
                # 熵loss占比过高，降低系数
                urgency = entropy_loss_ratio - self.target_entropy_loss_ratio_max
                loss_ratio_factor = 1.0 - self.learning_rate * urgency
            elif entropy_loss_ratio < self.target_entropy_loss_ratio_min:
                # 熵loss占比过低，增加系数
                urgency = self.target_entropy_loss_ratio_min - entropy_loss_ratio
                loss_ratio_factor = 1.0 + self.learning_rate * urgency
            else:
                # 占比在合理范围内，保持稳定
                loss_ratio_factor = 1.0

        # 5.3. 综合调整（熵值优先级更高）
        # 如果熵过低，优先增加系数
        if actual_entropy < self.target_entropy_min:
            # 熵过低时，忽略loss_ratio的降低建议
            if loss_ratio_factor < 1.0:
                loss_ratio_factor = 1.0
            combined_factor = entropy_factor * loss_ratio_factor
        else:
            # 正常情况，但需要考虑熵值接近下限的情况
            # 如果熵值接近下限（低于目标范围的中间值），优先使用熵值因子
            target_mid = (self.target_entropy_min + self.target_entropy_max) / 2.0
            if actual_entropy < target_mid:
                # 熵值接近下限，不应该根据ratio降低系数
                if loss_ratio_factor < 1.0:
                    loss_ratio_factor = 1.0
                combined_factor = entropy_factor * loss_ratio_factor
            else:
                # 熵值接近上限或在上限附近，可以综合调整
                combined_factor = (entropy_factor + loss_ratio_factor) / 2.0

        # 5.4. 更新熵系数
        old_ent_coef = self.ent_coef
        self.ent_coef *= combined_factor

        # 5.5. 裁剪到合理范围
        self.ent_coef = max(self.ent_coef_min, min(self.ent_coef_max, self.ent_coef))

        # 5.6. 应用衰减（训练后期减少探索）
        if epoch >= self.decay_start_epoch:
            self.ent_coef *= self.decay_rate
            self.ent_coef = max(self.ent_coef_min, self.ent_coef)

        # 5.7. 记录历史
        self._ent_coef_history.append(self.ent_coef)

        # 5.8. 日志记录
        if self.logger and self._update_count % 10 == 0:
            log_msg = (
                f"[Epoch {epoch}] Discrete entropy coefficient update:\n"
                f"  actual_entropy={actual_entropy:.4f} (target: [{self.target_entropy_min:.2f}, {self.target_entropy_max:.2f}])\n"
            )
            if entropy_loss_ratio is not None:
                log_msg += (
                    f"  entropy_loss_ratio={entropy_loss_ratio:.4f} "
                    f"(target: [{self.target_entropy_loss_ratio_min:.2f}, {self.target_entropy_loss_ratio_max:.2f}])\n"
                )
            log_msg += (
                f"  entropy_factor={entropy_factor:.4f}, loss_ratio_factor={loss_ratio_factor:.4f}\n"
                f"  ent_coef: {old_ent_coef:.4f} -> {self.ent_coef:.4f}"
            )
            if epoch >= self.decay_start_epoch:
                log_msg += f" (decay applied)"
            self.logger.info(log_msg)

        return self.ent_coef

    def get_entropy_statistics(self) -> Dict[str, Any]:
        """
        获取熵相关的统计信息

        Returns:
            包含统计信息的字典
        """
        stats = {
            "ent_coef": self.ent_coef,
            "update_count": self._update_count,
            "last_entropy": self._last_entropy,
            "last_entropy_loss_ratio": self._last_entropy_loss_ratio,
        }

        if self._entropy_history:
            stats["entropy_mean"] = np.mean(self._entropy_history[-100:])
            stats["entropy_std"] = np.std(self._entropy_history[-100:])
            stats["entropy_min"] = np.min(self._entropy_history[-100:])
            stats["entropy_max"] = np.max(self._entropy_history[-100:])

        if self._entropy_loss_ratio_history:
            stats["entropy_loss_ratio_mean"] = np.mean(
                self._entropy_loss_ratio_history[-100:]
            )
            stats["entropy_loss_ratio_std"] = np.std(
                self._entropy_loss_ratio_history[-100:]
            )

        if self._ent_coef_history:
            stats["ent_coef_mean"] = np.mean(self._ent_coef_history[-100:])
            stats["ent_coef_std"] = np.std(self._ent_coef_history[-100:])

        return stats

    def reset(self):
        """重置控制器状态"""
        self._entropy_history.clear()
        self._entropy_loss_ratio_history.clear()
        self._ent_coef_history.clear()
        self._update_count = 0
        self._last_entropy = None
        self._last_entropy_loss_ratio = None
        if self.logger:
            self.logger.info("AdaptiveEntropyControllerDisc reset")

    def force_ent_coef(self, ent_coef: float):
        """强制设置熵系数（覆盖自适应逻辑）"""
        ent_coef = max(self.ent_coef_min, min(self.ent_coef_max, ent_coef))
        old_coef = self.ent_coef
        self.ent_coef = ent_coef
        if self.logger:
            self.logger.info(
                f"Discrete entropy coefficient forced: {old_coef:.4f} -> {ent_coef:.4f}"
            )
