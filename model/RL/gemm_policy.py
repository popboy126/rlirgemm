# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/4/1 09:30
# Contact       : hucheng.lew@qq.com
# File          : gemm_policy.py
# IDE           : PyCharm
# Description   : GEMM采用的策略

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

import numpy as np
import torch
from exploration.random_network_distillation import RNDModule
from exploration.running_mean_std import RunningMeanStd
from gemm_env import get_reward_norm_range
from tianshou.algorithm import PPO, Algorithm
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer, SequenceSummaryStats
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol


@dataclass(kw_only=True)
class GEMMTrainingStats(A2CTrainingStats):
    rnd_loss: SequenceSummaryStats
    rew_env: SequenceSummaryStats
    rew_intrinsic: SequenceSummaryStats
    env_performance: SequenceSummaryStats
    ent_coef: SequenceSummaryStats
    base_loss: SequenceSummaryStats
    ent_loss_disc: SequenceSummaryStats
    ent_loss_disc_loop_order: SequenceSummaryStats
    ent_loss_disc_packing_type: SequenceSummaryStats
    ent_loss_disc_thread_type: SequenceSummaryStats
    ent_loss_cont: SequenceSummaryStats
    ent_coef_cont: SequenceSummaryStats
    ent_coef_disc_loop_order: SequenceSummaryStats
    ent_coef_disc_packing_type: SequenceSummaryStats
    ent_coef_disc_thread_type: SequenceSummaryStats
    ent_scale_factor_cont: SequenceSummaryStats
    ent_scale_factor_loop_order: SequenceSummaryStats
    ent_scale_factor_packing_type: SequenceSummaryStats
    ent_scale_factor_thread_type: SequenceSummaryStats
    raw_ent_loss_disc_loop_order: SequenceSummaryStats
    raw_ent_loss_disc_packing_type: SequenceSummaryStats
    raw_ent_loss_disc_thread_type: SequenceSummaryStats
    raw_ent_loss_cont: SequenceSummaryStats
    actor_loss_ratio: SequenceSummaryStats
    entropy_loss_ratio: SequenceSummaryStats


@dataclass(kw_only=True)
class InfoForEntropyAdaptiveController:
    """
    传递给自适应熵调节器的必要信息
    """

    last_mean_entropy: float | None = None
    last_mean_cont_entropy: float | None = None
    # 记录最近一次的平均base_loss和ent_loss，供自适应控制器使用
    last_mean_base_loss: float | None = None
    last_mean_ent_loss: float | None = None
    # 记录最近一次的离散动作原始熵值，供离散动作自适应控制器使用
    last_mean_raw_entropy_disc_loop_order: float | None = None
    last_mean_raw_entropy_disc_packing_type: float | None = None
    last_mean_raw_entropy_disc_thread_type: float | None = None
    # 记录最近一次的平均环境性能，供性能驱动自适应控制器使用
    last_mean_env_performance: float | None = None
    # 记录最近一次的连续动作原始熵值，供性能驱动自适应控制器使用
    last_mean_raw_entropy_cont: float | None = None


class GemmPPOPolicy(PPO):
    """
    GEMM PPO策略
    """

    def __init__(self, *args, **kwargs):
        # 取出rnd相关的配置
        rnd_net: RNDModule = kwargs.pop("rnd_net", None)
        rnd_optim = kwargs.pop("rnd_optim", None)
        rnd_intrinsic_weight: float = kwargs.pop("rnd_intrinsic_weight", None)
        # 分离式entropy系数：离散部分固定低值，连续部分自适应
        ent_coef_disc_loop_order: float = kwargs.pop("ent_coef_disc_loop_order", 0.03)
        ent_coef_disc_packing_type: float = kwargs.pop(
            "ent_coef_disc_packing_type", 0.03
        )
        ent_coef_disc_thread_type: float = kwargs.pop("ent_coef_disc_thread_type", 0.06)
        ent_coef_cont: float = kwargs.pop("ent_coef_cont", 0.10)
        use_adaptive_entropy_balancer: bool = kwargs.pop(
            "use_adaptive_entropy_balancer", False
        )
        # v8: Dual-Coefficient scale factors（粗调系数）
        scale_factor_loop_order: float = kwargs.pop("scale_factor_loop_order", 1.0)
        scale_factor_packing_type: float = kwargs.pop("scale_factor_packing_type", 1.0)
        scale_factor_thread_type: float = kwargs.pop("scale_factor_thread_type", 1.0)
        scale_factor_cont: float = kwargs.pop("scale_factor_cont", 1.0)
        super().__init__(*args, **kwargs)
        # self._indicator_extractor = IndicatorExtractor()
        self.logger = logging.getLogger("GemmPPOPolicy")
        self._rnd_net = rnd_net
        self._rnd_optim = rnd_optim
        self._rnd_intrinsic_weight = rnd_intrinsic_weight
        self._reward_norm_range = get_reward_norm_range()
        self._ent_coef_disc_loop_order = ent_coef_disc_loop_order
        self._ent_coef_disc_packing_type = ent_coef_disc_packing_type
        self._ent_coef_disc_thread_type = ent_coef_disc_thread_type
        self._ent_coef_cont = ent_coef_cont
        self._scale_factor_loop_order = scale_factor_loop_order
        self._scale_factor_packing_type = scale_factor_packing_type
        self._scale_factor_thread_type = scale_factor_thread_type
        self._scale_factor_cont = scale_factor_cont

        # 用于归一化内在奖励 (非常重要，否则训练不稳定)
        self.irew_rms = RunningMeanStd(
            (1,)
        )  # 可以在 process_fn 中实现简单的 RunningMeanStd

        self._info_for_entropy_adaptive_controller = InfoForEntropyAdaptiveController()
        self._use_adaptive_entropy_balancer = use_adaptive_entropy_balancer

        # 保存最后一次的batch数据，供按需计算loss张量使用
        # 用于解决trainer无法访问batch数据的问题
        self._last_batch_for_gradient_norm = None
        self._last_batch_size_for_gradient_norm = None

        # 上一次平均性能
        self._last_mean_env_performance = 0.0

    @property
    def last_mean_env_performance(self) -> float:
        """
        获取上一次性能
        """
        return self._last_mean_env_performance

    @property
    def info_for_entropy_adaptive_controller(self) -> InfoForEntropyAdaptiveController:
        """
        获取用于自适应熵控制器计算使用的方法
        """
        return self._info_for_entropy_adaptive_controller

    @property
    def ent_coef_cont(self):
        """连续动作entropy系数（供自适应控制器读写）"""
        return self._ent_coef_cont

    @ent_coef_cont.setter
    def ent_coef_cont(self, value):
        self._ent_coef_cont = value

    @property
    def scale_factor_loop_order(self):
        """离散动作loop_order的scale factor（供v8自适应控制器读写）"""
        return self._scale_factor_loop_order

    @scale_factor_loop_order.setter
    def scale_factor_loop_order(self, value):
        self._scale_factor_loop_order = value

    @property
    def scale_factor_packing_type(self):
        """离散动作packing_type的scale factor（供v8自适应控制器读写）"""
        return self._scale_factor_packing_type

    @scale_factor_packing_type.setter
    def scale_factor_packing_type(self, value):
        self._scale_factor_packing_type = value

    @property
    def scale_factor_thread_type(self):
        """离散动作thread_type的scale factor（供v8自适应控制器读写）"""
        return self._scale_factor_thread_type

    @scale_factor_thread_type.setter
    def scale_factor_thread_type(self, value):
        self._scale_factor_thread_type = value

    @property
    def scale_factor_cont(self):
        """连续动作的scale factor（供v8自适应控制器读写）"""
        return self._scale_factor_cont

    @scale_factor_cont.setter
    def scale_factor_cont(self, value):
        self._scale_factor_cont = value

    class GEMMOptimizer(Algorithm.Optimizer):
        """
        适用于GEMM的优化器
        """

        def __init__(self, *args, **kwargs):
            """
            初始化
            """
            super().__init__(*args, **kwargs)
            self.logger = logging.getLogger("GemmPPOPolicy")

        def step(
            self,
            loss: torch.Tensor,
            retain_graph: bool | None = None,
            create_graph: bool = False,
        ) -> None:
            """Performs an optimizer step, optionally applying gradient clipping (if configured at construction).

            :param loss: the loss to backpropagate
            :param retain_graph: passed on to `backward`
            :param create_graph: passed on to `backward`
            """
            self._optim.zero_grad()
            loss.backward(retain_graph=retain_graph, create_graph=create_graph)
            if self._max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._module.parameters(), max_norm=self._max_grad_norm
                )
            self._optim.step()

    def _create_optimizer(
        self,
        module: torch.nn.Module,
        factory: OptimizerFactory,
        max_grad_norm: float | None = None,
    ) -> Algorithm.Optimizer:
        optimizer, lr_scheduler = factory.create_instances(module)
        if lr_scheduler is not None:
            self.lr_schedulers.append(lr_scheduler)
        optim = self.GEMMOptimizer(optimizer, module, max_grad_norm=max_grad_norm)
        self._optimizers.append(optim)
        return optim

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        # 1. 计算新奇性奖励 (RND Error)
        obs = batch.obs
        act = batch.act  # 确保你的 env 输出的 act 包含具体的参数值

        intrinsic_rew = self._rnd_net.compute_intrinsic_reward(obs, act).cpu().numpy()

        # 2. 创建掩码：只对 performance > 0 的样本应用RND
        performance = batch.info["performance"]
        valid_mask = performance > 0

        # 3. 只用 performance > 0 的样本更新归一化统计量
        # 对于 valid_mask=False 的位置，用 0 替换
        intrinsic_rew_for_update = np.where(valid_mask, intrinsic_rew, 0.0)
        self.irew_rms.update(intrinsic_rew_for_update)

        # 4. 使用更新后的统计量归一化当前批次的内在奖励
        intrinsic_rew_normalized = self.irew_rms.normalize(
            intrinsic_rew, clip_range=self._reward_norm_range
        )

        intrinsic_rew_normalized_mask = intrinsic_rew_normalized > 0
        positive_intrinsic_normalized = np.where(
            intrinsic_rew_normalized_mask, intrinsic_rew_normalized, 0.0
        )

        # 5. 只对 performance > 0 的样本应用内在奖励
        total_intrinsic = np.where(
            valid_mask, self._rnd_intrinsic_weight * positive_intrinsic_normalized, 0.0
        )

        # 将内在奖励存入 batch，以便后续可能的分析 (info)
        batch.info["rew_intrinsic"] = total_intrinsic

        # 修改 batch.rew (PPO 将基于这个混合奖励进行优势估算)
        batch.rew += total_intrinsic

        # 调用父类方法计算 GAE (Advantage)
        return super()._preprocess_batch(batch, buffer, indices)

    def _update_rnd_with_batch(  # type: ignore[override]
        self, batch: LogpOldProtocol
    ) -> float:
        # 1. 更新 RND Predictor
        self._rnd_optim.zero_grad()
        rnd_loss = self._rnd_net(batch.obs, batch.act)
        rnd_loss.backward()
        self._rnd_optim.step()
        return rnd_loss.item()

    def _compute_mean_pure_env_performance(self, minibatch: LogpOldProtocol) -> float:
        # 计算纯环境奖励的平均值 (不包含内在奖励)
        return minibatch.info["performance"].mean().item()

    def _compute_losses_for_gradient_norm(
        self, minibatch: LogpOldProtocol
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算用于梯度范数计算的loss张量

        这个成员函数计算clip_loss和各个原始熵损失，供自适应熵系数调节器使用。
        这些loss张量将用于计算梯度范数，从而动态调整熵系数。

        Args:
            minibatch: 一个minibatch数据

        Returns:
            tuple: (clip_loss, raw_ent_loss_loop_order, raw_ent_loss_packing_type,
                   raw_ent_loss_thread_type, raw_ent_loss_cont)
        """
        # 计算前向传播
        dist = self.policy(minibatch).dist

        # 计算clip_loss (策略损失)
        advantages = minibatch.adv
        if self.advantage_normalization:
            mean, std = advantages.mean(), advantages.std()
            advantages = (advantages - mean) / (std + self._eps)

        log_ratios = dist.log_prob(minibatch.act) - minibatch.logp_old
        log_ratios = torch.clamp(log_ratios, min=-20.0, max=20.0)
        ratios = log_ratios.exp().float()
        ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
        surr1 = ratios * advantages
        surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

        if self.dual_clip:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
        else:
            clip_loss = -torch.min(surr1, surr2).mean()

        # 计算各个原始熵损失
        (
            ent_disc_loop_order,
            ent_disc_packing_type,
            ent_disc_thread_type,
            ent_cont,
        ) = dist.entropy_decomposed_for_actions(minibatch.act)

        # 计算原始熵损失（平均熵值）
        raw_ent_loss_loop_order = ent_disc_loop_order.mean()
        raw_ent_loss_packing_type = ent_disc_packing_type.mean()
        raw_ent_loss_thread_type = ent_disc_thread_type.mean()
        raw_ent_loss_cont = ent_cont.mean()

        return (
            clip_loss,
            raw_ent_loss_loop_order,
            raw_ent_loss_packing_type,
            raw_ent_loss_thread_type,
            raw_ent_loss_cont,
        )

    def compute_and_save_losses_for_gradient_norm(
        self, batch: LogpOldProtocol | None = None, batch_size: int | None = None
    ) -> (
        None
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        计算并保存用于梯度范数计算的loss张量（公共方法）

        该方法由trainer按 gradient_norm_balancer_update_frequency 频率调用，
        以避免每次 _update_with_batch 都计算loss张量造成不必要的计算开销。

        计算策略：遍历所有minibatches，计算平均loss张量，以提高代表性和稳定性。

        Args:
            batch: 训练batch数据（None时使用保存的_last_batch_for_gradient_norm）
            batch_size: minibatch大小（None表示使用整个batch作为一个minibatch）
        """
        if not self._use_adaptive_entropy_balancer:
            self.logger.warning(
                "compute_and_save_losses_for_gradient_norm called but adaptive entropy balancer is not enabled"
            )
            return None

        # 如果没有传入batch，使用保存的最后一次batch数据
        if batch is None:
            if self._last_batch_for_gradient_norm is None:
                self.logger.warning(
                    "compute_and_save_losses_for_gradient_norm called but no batch data available"
                )
                return None
            batch = self._last_batch_for_gradient_norm
            batch_size = self._last_batch_size_for_gradient_norm

        # 遍历所有minibatch，计算loss张量并累积
        split_batch_size = batch_size or -1
        all_minibatches = list(batch.split(split_batch_size, merge_last=True))

        clip_loss_sum = None
        raw_ent_loss_loop_order_sum = None
        raw_ent_loss_packing_type_sum = None
        raw_ent_loss_thread_type_sum = None
        raw_ent_loss_cont_sum = None

        for minibatch in all_minibatches:
            (
                clip_loss_tensor,
                raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont_tensor,
            ) = self._compute_losses_for_gradient_norm(minibatch)

            # 累积loss张量
            if clip_loss_sum is None:
                clip_loss_sum = clip_loss_tensor.clone()
                raw_ent_loss_loop_order_sum = raw_ent_loss_loop_order_tensor.clone()
                raw_ent_loss_packing_type_sum = raw_ent_loss_packing_type_tensor.clone()
                raw_ent_loss_thread_type_sum = raw_ent_loss_thread_type_tensor.clone()
                raw_ent_loss_cont_sum = raw_ent_loss_cont_tensor.clone()
            else:
                clip_loss_sum += clip_loss_tensor
                raw_ent_loss_loop_order_sum += raw_ent_loss_loop_order_tensor
                raw_ent_loss_packing_type_sum += raw_ent_loss_packing_type_tensor
                raw_ent_loss_thread_type_sum += raw_ent_loss_thread_type_tensor
                raw_ent_loss_cont_sum += raw_ent_loss_cont_tensor

        # 计算平均loss张量
        num_minibatches = len(all_minibatches)
        avg_clip_loss_tensor = clip_loss_sum / num_minibatches
        avg_raw_ent_loss_loop_order_tensor = (
            raw_ent_loss_loop_order_sum / num_minibatches
        )
        avg_raw_ent_loss_packing_type_tensor = (
            raw_ent_loss_packing_type_sum / num_minibatches
        )
        avg_raw_ent_loss_thread_type_tensor = (
            raw_ent_loss_thread_type_sum / num_minibatches
        )
        avg_raw_ent_loss_cont_tensor = raw_ent_loss_cont_sum / num_minibatches

        self.logger.debug(
            f"Computed and saved average loss tensors from {num_minibatches} minibatches"
        )
        return (
            avg_clip_loss_tensor,
            avg_raw_ent_loss_loop_order_tensor,
            avg_raw_ent_loss_packing_type_tensor,
            avg_raw_ent_loss_thread_type_tensor,
            avg_raw_ent_loss_cont_tensor,
        )

    def _update_with_batch(
        self,
        batch: LogpOldProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> A2CTrainingStats:
        (
            losses,
            clip_losses,
            vf_losses,
            ent_losses,
            rnd_losses,
            rew_env,
            rew_intrinsic,
            env_performance,
            ent_coefs,
        ) = [], [], [], [], [], [], [], [], []
        actor_loss_ratios, entropy_loss_ratios = [], []
        (
            ent_losses_disc_loop_order,
            ent_losses_disc_packing_type,
            ent_losses_disc_thread_type,
        ) = [], [], []
        ent_losses_disc, ent_losses_cont, ent_coefs_cont = [], [], []
        (
            raw_ent_losses_disc_loop_order,
            raw_ent_losses_disc_packing_type,
            raw_ent_losses_disc_thread_type,
            raw_ent_losses_cont,
        ) = [], [], [], []
        (
            ent_coefs_disc_loop_order,
            ent_coefs_disc_packing_type,
            ent_coefs_disc_thread_type,
        ) = [], [], []
        (
            ent_scale_factors_cont,
            ent_scale_factors_loop_order,
            ent_scale_factors_packing_type,
            ent_scale_factors_thread_type
        ) = [], [], [], []
        base_losses = []
        gradient_steps = 0
        split_batch_size = batch_size or -1
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = cast(
                    LogpOldProtocol,
                    self._add_returns_and_advantages(
                        batch, self._buffer, self._indices
                    ),
                )
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                # calculate loss for actor
                advantages = minibatch.adv
                dist = self.policy(minibatch).dist
                if self.advantage_normalization:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (
                        std + self._eps
                    )  # per-batch norm
                # 1. 计算 log_ratio
                log_ratios = dist.log_prob(minibatch.act) - minibatch.logp_old
                # 2. 钳位 log_ratio 防止 exp() 溢出
                # float32 的 exp 上限大约是 88，但为了安全起见，限制在 [-20, 20] 足够了
                # PPO 本身也会 clip ratio (通常是 1-eps 到 1+eps)，这里只是防止中间计算爆炸
                log_ratios = torch.clamp(log_ratios, min=-20.0, max=20.0)
                # 3. 安全计算 ratio
                ratios = log_ratios.exp().float()
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                surr1 = ratios * advantages
                surr2 = (
                    ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                )
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip,
                        self.eps_clip,
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                # 分离计算离散和连续entropy，使用不同系数
                # 根据每个样本的loop_order值动态计算有效维度的熵
                (
                    ent_disc_loop_order,
                    ent_disc_packing_type,
                    ent_disc_thread_type,
                    ent_cont,
                ) = dist.entropy_decomposed_for_actions(minibatch.act)
                raw_ent_loss_disc_loop_order = ent_disc_loop_order.mean()
                raw_ent_loss_disc_packing_type = ent_disc_packing_type.mean()
                raw_ent_loss_disc_thread_type = ent_disc_thread_type.mean()
                raw_ent_loss_cont = ent_cont.mean()
                ent_loss_disc_loop_order = (
                    self._ent_coef_disc_loop_order
                    * (self._scale_factor_loop_order * raw_ent_loss_disc_loop_order)
                )
                ent_loss_disc_packing_type = (
                    self._ent_coef_disc_packing_type
                    * (self._scale_factor_packing_type * raw_ent_loss_disc_packing_type)
                )
                ent_loss_disc_thread_type = (
                    self._ent_coef_disc_thread_type
                    * (self._scale_factor_thread_type * raw_ent_loss_disc_thread_type)
                )

                ent_loss_disc = (
                    ent_loss_disc_loop_order
                    + ent_loss_disc_packing_type
                    + ent_loss_disc_thread_type
                )
                ent_loss_cont = (
                    self._ent_coef_cont * (self._scale_factor_cont * raw_ent_loss_cont)
                )
                ent_loss = ent_loss_disc + ent_loss_cont  # 总entropy用于日志
                base_loss = clip_loss + self.vf_coef * vf_loss

                loss = base_loss - ent_loss

                self.optim.step(loss)
                rnd_loss = self._update_rnd_with_batch(minibatch)
                env_perf = self._compute_mean_pure_env_performance(minibatch)
                self._last_mean_env_performance = env_perf  # 保存当前批次的平均性能
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                raw_ent_losses_disc_loop_order.append(
                    raw_ent_loss_disc_loop_order.item()
                )
                raw_ent_losses_disc_packing_type.append(
                    raw_ent_loss_disc_packing_type.item()
                )
                raw_ent_losses_disc_thread_type.append(
                    raw_ent_loss_disc_thread_type.item()
                )
                ent_losses_disc_loop_order.append(ent_loss_disc_loop_order.item())
                ent_losses_disc_packing_type.append(ent_loss_disc_packing_type.item())
                ent_losses_disc_thread_type.append(ent_loss_disc_thread_type.item())
                raw_ent_losses_cont.append(raw_ent_loss_cont.item())
                ent_losses_disc.append(ent_loss_disc.item())
                ent_losses_cont.append(ent_loss_cont.item())
                base_losses.append(base_loss.item())
                losses.append(loss.item())
                # 使用不带entropy的loss作为基准，避免loss接近0时ratio异常
                actor_loss_ratios.append(abs(clip_loss.item() / base_loss.item()))
                entropy_loss_ratios.append(abs(ent_loss.item() / base_loss.item()))
                rnd_losses.append(rnd_loss)
                rew_env.append(minibatch.info["reward"].mean().item())
                rew_intrinsic.append(minibatch.info["rew_intrinsic"].mean().item())
                env_performance.append(env_perf)
                self.ent_coef = (
                    self._ent_coef_disc_loop_order
                    + self._ent_coef_disc_packing_type
                    + self._ent_coef_disc_thread_type
                    + self._ent_coef_cont
                )
                ent_coefs.append(self.ent_coef)
                ent_coefs_cont.append(self._ent_coef_cont)
                ent_coefs_disc_loop_order.append(self._ent_coef_disc_loop_order)
                ent_coefs_disc_packing_type.append(self._ent_coef_disc_packing_type)
                ent_coefs_disc_thread_type.append(self._ent_coef_disc_thread_type)
                ent_scale_factors_cont.append(self._scale_factor_cont)
                ent_scale_factors_loop_order.append(self._scale_factor_loop_order)
                ent_scale_factors_packing_type.append(self._scale_factor_packing_type)
                ent_scale_factors_thread_type.append(self._scale_factor_thread_type)

        # 保存最后一次的batch数据，供按需计算loss张量使用
        # 用于解决trainer无法访问batch数据的问题
        self._last_batch_for_gradient_norm = batch
        self._last_batch_size_for_gradient_norm = batch_size

        # 注意：loss张量的计算已移至公共方法 compute_and_save_losses_for_gradient_norm
        # 该方法由trainer按 gradient_norm_balancer_update_frequency 频率调用
        # 以避免每次 _update_with_batch 都计算loss张量造成不必要的开销

        # 记录最近一次的平均熵，供自适应控制器使用
        self._info_for_entropy_adaptive_controller.last_mean_entropy = (
            sum(ent_losses) / len(ent_losses) if ent_losses else None
        )
        self._info_for_entropy_adaptive_controller.last_mean_cont_entropy = (
            sum(ent_losses_cont) / len(ent_losses_cont) if ent_losses_cont else None
        )
        # 记录最近一次的平均base_loss和ent_loss，供自适应控制器使用
        self._info_for_entropy_adaptive_controller.last_mean_base_loss = (
            sum(base_losses) / len(base_losses) if base_losses else None
        )
        self._info_for_entropy_adaptive_controller.last_mean_ent_loss = (
            sum(ent_losses) / len(ent_losses) if ent_losses else None
        )
        # 记录最近一次的离散动作原始熵值，供离散动作自适应控制器使用
        self._info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_loop_order = (
            sum(raw_ent_losses_disc_loop_order) / len(raw_ent_losses_disc_loop_order)
            if raw_ent_losses_disc_loop_order
            else None
        )
        self._info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_packing_type = (
            sum(raw_ent_losses_disc_packing_type)
            / len(raw_ent_losses_disc_packing_type)
            if raw_ent_losses_disc_packing_type
            else None
        )
        self._info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_thread_type = (
            sum(raw_ent_losses_disc_thread_type) / len(raw_ent_losses_disc_thread_type)
            if raw_ent_losses_disc_thread_type
            else None
        )
        # 记录最近一次的平均环境性能，供性能驱动自适应控制器使用
        self._info_for_entropy_adaptive_controller.last_mean_env_performance = (
            sum(env_performance) / len(env_performance) if env_performance else None
        )
        # 记录最近一次的连续动作原始熵值，供性能驱动自适应控制器使用
        self._info_for_entropy_adaptive_controller.last_mean_raw_entropy_cont = (
            sum(raw_ent_losses_cont) / len(raw_ent_losses_cont)
            if raw_ent_losses_cont
            else None
        )

        return GEMMTrainingStats(
            loss=SequenceSummaryStats.from_sequence(losses),
            actor_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            gradient_steps=gradient_steps,
            rnd_loss=SequenceSummaryStats.from_sequence(rnd_losses),
            rew_env=SequenceSummaryStats.from_sequence(rew_env),
            rew_intrinsic=SequenceSummaryStats.from_sequence(rew_intrinsic),
            env_performance=SequenceSummaryStats.from_sequence(env_performance),
            ent_coef=SequenceSummaryStats.from_sequence(ent_coefs),
            base_loss=SequenceSummaryStats.from_sequence(base_losses),
            ent_loss_disc=SequenceSummaryStats.from_sequence(ent_losses_disc),
            ent_loss_disc_loop_order=SequenceSummaryStats.from_sequence(
                ent_losses_disc_loop_order
            ),
            ent_loss_disc_packing_type=SequenceSummaryStats.from_sequence(
                ent_losses_disc_packing_type
            ),
            ent_loss_disc_thread_type=SequenceSummaryStats.from_sequence(
                ent_losses_disc_thread_type
            ),
            ent_loss_cont=SequenceSummaryStats.from_sequence(ent_losses_cont),
            ent_coef_cont=SequenceSummaryStats.from_sequence(ent_coefs_cont),
            ent_coef_disc_loop_order=SequenceSummaryStats.from_sequence(
                ent_coefs_disc_loop_order
            ),
            ent_coef_disc_packing_type=SequenceSummaryStats.from_sequence(
                ent_coefs_disc_packing_type
            ),
            ent_coef_disc_thread_type=SequenceSummaryStats.from_sequence(
                ent_coefs_disc_thread_type
            ),
            ent_scale_factor_cont=SequenceSummaryStats.from_sequence(ent_scale_factors_cont),
            ent_scale_factor_loop_order=SequenceSummaryStats.from_sequence(ent_scale_factors_loop_order),
            ent_scale_factor_packing_type=SequenceSummaryStats.from_sequence(ent_scale_factors_packing_type),
            ent_scale_factor_thread_type=SequenceSummaryStats.from_sequence(ent_scale_factors_thread_type),
            raw_ent_loss_disc_loop_order=SequenceSummaryStats.from_sequence(
                raw_ent_losses_disc_loop_order
            ),
            raw_ent_loss_disc_packing_type=SequenceSummaryStats.from_sequence(
                raw_ent_losses_disc_packing_type
            ),
            raw_ent_loss_disc_thread_type=SequenceSummaryStats.from_sequence(
                raw_ent_losses_disc_thread_type
            ),
            raw_ent_loss_cont=SequenceSummaryStats.from_sequence(raw_ent_losses_cont),
            actor_loss_ratio=SequenceSummaryStats.from_sequence(actor_loss_ratios),
            entropy_loss_ratio=SequenceSummaryStats.from_sequence(entropy_loss_ratios),
        )

    def dump_rnd_module(self) -> Dict[str, Any]:
        """
        导出 RND 模块的状态字典
        :return: RND 模块的状态字典
        """
        return {
            "rnd_net": self._rnd_net.state_dict(),
            "rnd_optim": self._rnd_optim.state_dict(),
        }

    def load_rnd_module(self, state_dict: Dict[str, Any]) -> None:
        """
        加载 RND 模块的状态字典
        :param state_dict: RND 模块的状态字典
        """
        self._rnd_net.load_state_dict(state_dict["rnd_net"])
        self._rnd_optim.load_state_dict(state_dict["rnd_optim"])

    def reset_rnd_module(self, epoch: int, env_step: int):
        """
        重置随机网络蒸馏
        """
        if self._rnd_net.reset_predictor(epoch, env_step):
            self.irew_rms.reset()
            self._rnd_optim.state.clear()  # 重置优化器
            self.logger.info(
                f"[Epoch {epoch}] RND predictor reset to restore exploration capability."
            )

    def update_rnd_intrinsic_weight(self):
        """
        更新随机蒸馏网络的原生权重
        """
        self._rnd_intrinsic_weight = self._rnd_intrinsic_weight * 0.9
        if self._rnd_intrinsic_weight < 1e-9:
            self._rnd_intrinsic_weight = 0
