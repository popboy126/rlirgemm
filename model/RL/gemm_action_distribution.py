# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/4/1 09:17
# Contact       : hucheng.lew@qq.com
# File          : gemm_action_distribution.py
# IDE           : PyCharm
# Description   : 自定义混合动作分布，支持根据loop_order动态调整有效连续维度

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    Normal,
    constraints,
)
from torch.types import _size


def get_valid_continuous_indices_by_loop_order(
    loop_order: int, full_continuous_dim: int = 8
) -> List[int]:
    """
    根据loop_order值获取有效的连续维度索引

    连续动作向量索引映射：
    - 0: mc
    - 1: kc
    - 2: nc
    - 3: prfm_a_skip_bytes
    - 4: prfm_b_skip_bytes
    - 5: prfm_c_skip_bytes
    - 6: prfm_add_c_skip_bytes
    - 7: prfm_add_partial_sum_skip_bytes

    Args:
        loop_order: 循环顺序值
        full_continuous_dim: 完整的连续维度数，默认为8

    Returns:
        有效维度的索引列表
    """
    if loop_order == 0:
        # loop_order=0时，mc, kc, nc, prfm_add_c, prfm_add_partial_sum 无意义
        # 保留: prfm_a(3), prfm_b(4), prfm_c(5)
        return [3, 4, 5]
    elif loop_order in [1, 2]:
        # loop_order=1或2时，kc, prfm_add_c, prfm_add_partial_sum 无意义
        # 保留: mc(0), nc(2), prfm_a(3), prfm_b(4), prfm_c(5)
        return [0, 2, 3, 4, 5]
    else:
        # 其他情况，所有维度都有效
        return list(range(full_continuous_dim))


class GemmActionDistribution(Distribution):
    """
    动作分布类，包含离散和连续动作分布

    根据loop_order值动态调整有效的连续动作维度：
    - loop_order=0: mc, kc, nc, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes 无意义
    - loop_order=1或2: kc, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes 无意义
    - 其他情况: 所有维度都有意义
    """

    def __init__(
        self,
        loop_order_logits: Union[Tensor, float],
        thread_type_logits: Union[Tensor, float],
        packing_type_logits: Union[Tensor, float],
        mu: Union[Tensor, float],
        sigma: Union[Tensor, float],
        action_split_info: Dict[str, int],
        validate_args=None,
    ):
        batch_shape = mu.shape[:-1] if isinstance(mu, Tensor) else torch.Size()
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=False)

        self._loop_order_dist = Categorical(logits=loop_order_logits)
        self._thread_type_dist = Categorical(logits=thread_type_logits)
        self._packing_type_dist = Categorical(logits=packing_type_logits)

        self._eps = 1e-7

        # 存储完整的mu和sigma（所有8个连续维度）
        self._full_mu = mu if isinstance(mu, Tensor) else torch.tensor(mu)
        self._full_sigma = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma)
        self._full_continuous_dim = (
            self._full_mu.shape[-1] if isinstance(self._full_mu, Tensor) else 8
        )

        # 创建一个默认的分布（使用所有维度，会在具体方法中动态调整）
        self._other_base_dist = Normal(self._full_mu, self._full_sigma)
        self._other_dist = Independent(self._other_base_dist, 1)

        self.action_split_info = action_split_info

        # 从loop_order_logits中提取argmax作为默认的loop_order值，用于entropy计算
        if isinstance(loop_order_logits, Tensor):
            self._default_loop_order = int(loop_order_logits.argmax(dim=-1)[0].item())
        else:
            self._default_loop_order = 0

    def _get_distribution_for_loop_order(
        self, loop_order: int
    ) -> Tuple[Independent, List[int]]:
        """
        根据loop_order值创建对应的连续动作分布

        Args:
            loop_order: 循环顺序值

        Returns:
            (分布对象, 有效维度索引列表)
        """
        valid_indices = get_valid_continuous_indices_by_loop_order(
            loop_order, self._full_continuous_dim
        )

        if isinstance(self._full_mu, Tensor):
            valid_mu = self._full_mu[..., valid_indices]
            valid_sigma = self._full_sigma[..., valid_indices]
        else:
            valid_mu = self._full_mu[valid_indices]
            valid_sigma = self._full_sigma[valid_indices]

        base_dist = Normal(valid_mu, valid_sigma)
        dist = Independent(base_dist, 1)

        return dist, valid_indices

    @property
    def mean(self) -> Tensor:
        """
        返回均值（属性，匹配PyTorch Distribution接口）

        Returns:
            完整维度的均值向量，无意义维度填充为0
        """
        return self._mean_impl()

    def _mean_impl(self, loop_order: Optional[int] = None) -> Tensor:
        """
        返回均值的实现

        Args:
            loop_order: 循环顺序值。如果为None，使用默认值（基于loop_order_logits的argmax）

        Returns:
            完整维度的均值向量，无意义维度填充为0
        """
        if loop_order is None:
            loop_order = self._default_loop_order

        dis_means = []
        for dist in [
            self._loop_order_dist,
            self._thread_type_dist,
            self._packing_type_dist,
        ]:
            probs = dist.probs
            categories = torch.arange(
                probs.size(-1), device=probs.device, dtype=probs.dtype
            )
            mean = torch.sum(probs * categories, dim=-1)
            dis_means.append(mean)
        dis_mean_vec = torch.stack(dis_means, dim=-1)

        # 获取有效维度的分布
        dist, valid_indices = self._get_distribution_for_loop_order(loop_order)

        # 构建完整维度的均值，无意义维度填充为0
        batch_size = dis_mean_vec.shape[0]
        full_cont_mean = torch.zeros(
            batch_size,
            self._full_continuous_dim,
            device=dis_mean_vec.device,
            dtype=dis_mean_vec.dtype,
        )
        valid_cont_mean = dist.base_dist.loc
        full_cont_mean[..., valid_indices] = valid_cont_mean

        return torch.cat([dis_mean_vec, full_cont_mean], dim=-1)

    def mean_for_loop_order(self, loop_order: int) -> Tensor:
        """
        根据loop_order返回有效维度的均值

        Args:
            loop_order: 循环顺序值

        Returns:
            完整维度的均值向量，无意义维度填充为0
        """
        return self._mean_impl(loop_order=loop_order)

    @property
    def variance(self) -> torch.Tensor:
        """
        返回方差（属性，匹配PyTorch Distribution接口）

        Returns:
            完整维度的方差向量，无意义维度填充为0
        """
        return self._variance_impl()

    def _variance_impl(self, loop_order: Optional[int] = None) -> torch.Tensor:
        """
        返回方差的实现

        Args:
            loop_order: 循环顺序值。如果为None，使用默认值（基于loop_order_logits的argmax）

        Returns:
            完整维度的方差向量，无意义维度填充为0
        """
        if loop_order is None:
            loop_order = self._default_loop_order

        dis_vars = []
        for dist in [
            self._loop_order_dist,
            self._thread_type_dist,
            self._packing_type_dist,
        ]:
            probs = dist.probs
            categories = torch.arange(
                probs.size(-1), device=probs.device, dtype=probs.dtype
            )
            mean = torch.sum(probs * categories, dim=-1)
            mean_square = torch.sum(probs * categories**2, dim=-1)
            var = mean_square - mean**2
            dis_vars.append(var)
        dis_var_vec = torch.stack(dis_vars, dim=-1)

        # 获取有效维度的分布
        dist, valid_indices = self._get_distribution_for_loop_order(loop_order)

        # 构建完整维度的方差，无意义维度填充为0
        batch_size = dis_var_vec.shape[0]
        full_cont_var = torch.zeros(
            batch_size,
            self._full_continuous_dim,
            device=dis_var_vec.device,
            dtype=dis_var_vec.dtype,
        )
        valid_cont_var = dist.base_dist.scale**2
        full_cont_var[..., valid_indices] = valid_cont_var

        return torch.cat([dis_var_vec, full_cont_var], dim=-1)

    def variance_for_loop_order(self, loop_order: int) -> torch.Tensor:
        """
        根据loop_order返回有效维度的方差

        Args:
            loop_order: 循环顺序值

        Returns:
            完整维度的方差向量，无意义维度填充为0
        """
        return self._variance_impl(loop_order=loop_order)

    def log_prob(self, action: torch.Tensor) -> Tensor:
        """
        返回对数化的概率密度函数（根据action中的loop_order动态计算）

        Args:
            action: 动作张量，包含loop_order、thread_type、packing_type和连续动作

        Returns:
            log概率值
        """
        loop_order_action, thread_type_action, packing_type_action, other_actions = (
            action[..., self.action_split_info["loop_order"]],
            action[..., self.action_split_info["thread_type"]],
            action[..., self.action_split_info["packing_type"]],
            action[..., self.action_split_info["continues"] :],
        )

        # 计算离散部分的log_prob
        dis_logps = [
            self._loop_order_dist.log_prob(loop_order_action),
            self._thread_type_dist.log_prob(thread_type_action),
            self._packing_type_dist.log_prob(packing_type_action),
        ]
        total_dis_logp = torch.stack(dis_logps, dim=-1).sum(dim=-1)

        # 根据每个样本的loop_order值，计算对应的有效维度的log_prob
        # 需要为每个样本单独创建分布，以避免batch维度问题
        batch_size = action.shape[0]
        cont_logps = []

        for i in range(batch_size):
            loop_order_val = int(loop_order_action[i].item())

            # 获取有效维度索引
            valid_indices = get_valid_continuous_indices_by_loop_order(
                loop_order_val, self._full_continuous_dim
            )

            # 提取第i个样本的mu和sigma（在valid维度上）
            if isinstance(self._full_mu, Tensor):
                if self._full_mu.dim() > 1 and self._full_mu.shape[0] > 1:
                    # 有batch维度，提取第i个样本
                    sample_mu = self._full_mu[i : i + 1, valid_indices]
                    sample_sigma = self._full_sigma[i : i + 1, valid_indices]
                else:
                    # 没有batch维度或batch_size=1，直接使用
                    sample_mu = self._full_mu[..., valid_indices]
                    sample_sigma = self._full_sigma[..., valid_indices]
            else:
                sample_mu = self._full_mu[valid_indices]
                sample_sigma = self._full_sigma[valid_indices]

            # 创建只包含这一个样本的分布
            sample_dist = Independent(Normal(sample_mu, sample_sigma), 1)

            # 提取第i个样本的有效连续动作
            valid_other_actions = other_actions[i : i + 1, valid_indices]

            # 计算log_prob
            cont_logp = sample_dist.log_prob(valid_other_actions)
            cont_logps.append(cont_logp)

        # 合并所有样本的log_prob
        if batch_size == 1:
            cont_logp = cont_logps[0]
        else:
            cont_logp = torch.cat(cont_logps, dim=0)

        return total_dis_logp + cont_logp

    def entropy(self, loop_order: Optional[int] = None) -> Tensor:
        """
        计算熵

        Args:
            loop_order: 循环顺序值。如果为None，使用默认值（基于loop_order_logits的argmax）

        Returns:
            总熵值，包括离散和连续有效维度的熵
        """
        if loop_order is None:
            loop_order = self._default_loop_order

        loop_order_entropy = self._loop_order_dist.entropy()
        thread_type_entropy = self._thread_type_dist.entropy()
        packing_type_entropy = self._packing_type_dist.entropy()

        # 获取有效维度的分布
        dist, _ = self._get_distribution_for_loop_order(loop_order)
        cont_ent = dist.entropy()

        return (
            loop_order_entropy + thread_type_entropy + packing_type_entropy + cont_ent
        )

    def entropy_for_loop_order(self, loop_order: int) -> Tensor:
        """
        根据loop_order计算有效维度的熵

        Args:
            loop_order: 循环顺序值

        Returns:
            熵值
        """
        return self.entropy(loop_order=loop_order)

    def entropy_decomposed(
        self, loop_order: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        分别计算离散和连续部分的熵

        Args:
            loop_order: 循环顺序值。如果为None，使用默认值（基于loop_order_logits的argmax）

        Returns:
            (H_discrete, H_continuous): 离散部分熵和连续部分熵
        """
        if loop_order is None:
            loop_order = self._default_loop_order

        loop_order_entropy = self._loop_order_dist.entropy()
        thread_type_entropy = self._thread_type_dist.entropy()
        packing_type_entropy = self._packing_type_dist.entropy()
        h_discrete = loop_order_entropy + thread_type_entropy + packing_type_entropy

        # 获取有效维度的分布
        dist, _ = self._get_distribution_for_loop_order(loop_order)
        h_continuous = dist.entropy()

        return h_discrete, h_continuous

    def entropy_decomposed_for_loop_order(
        self, loop_order: int
    ) -> Tuple[Tensor, Tensor]:
        """
        根据loop_order分别计算离散和连续部分的熵

        Args:
            loop_order: 循环顺序值

        Returns:
            (H_discrete, H_continuous): 离散部分熵和连续部分熵
        """
        return self.entropy_decomposed(loop_order=loop_order)

    def entropy_decomposed_for_actions(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        根据actions中每个样本的loop_order值，批量计算熵
        用于训练时根据实际action计算熵

        Args:
            actions: 动作张量，包含loop_order信息

        Returns:
            (H_discrete, H_continuous): 离散部分熵和连续部分熵
        """
        loop_order_entropy = self._loop_order_dist.entropy()
        thread_type_entropy = self._thread_type_dist.entropy()
        packing_type_entropy = self._packing_type_dist.entropy()

        # 从actions中提取loop_order值
        loop_order_actions = actions[..., self.action_split_info["loop_order"]]
        batch_size = loop_order_actions.shape[0]

        # 为每个样本计算对应loop_order的连续熵
        h_continuous_list = []
        for i in range(batch_size):
            loop_order_val = int(loop_order_actions[i].item())
            dist, _ = self._get_distribution_for_loop_order(loop_order_val)
            h_cont = dist.entropy()
            # 提取第i个样本的熵
            if h_cont.dim() == 0:
                h_continuous_list.append(h_cont.unsqueeze(0))
            else:
                h_continuous_list.append(h_cont[i : i + 1])

        # 合并所有样本的连续熵
        if batch_size == 1:
            h_continuous = h_continuous_list[0]
        else:
            h_continuous = torch.cat(h_continuous_list, dim=0)

        return loop_order_entropy, packing_type_entropy, thread_type_entropy, h_continuous

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        采样动作（基于loop_order_logits的argmax确定有效维度）
        返回完整维度，无意义维度填充为默认值0.5

        Args:
            sample_shape: 采样形状

        Returns:
            采样的动作张量
        """
        # 先采样离散动作
        loop_order_samples = self._loop_order_dist.sample(sample_shape)
        thread_type_samples = self._thread_type_dist.sample(sample_shape)
        packing_type_samples = self._packing_type_dist.sample(sample_shape)
        dis_samples = torch.stack(
            [loop_order_samples, thread_type_samples, packing_type_samples], dim=-1
        )

        # 确定batch_size
        if len(sample_shape) > 0:
            batch_size = sample_shape[0]
        elif loop_order_samples.dim() == 0:
            batch_size = 1
        else:
            batch_size = loop_order_samples.shape[0]

        # 采样连续动作
        other_samples_list = []
        for i in range(batch_size):
            if batch_size > 1:
                loop_order_val = int(loop_order_samples[i].item())
            else:
                loop_order_val = int(loop_order_samples.item())

            # 获取有效维度索引
            valid_indices = get_valid_continuous_indices_by_loop_order(
                loop_order_val, self._full_continuous_dim
            )

            # 提取第i个样本的mu和sigma（在valid维度上）
            if isinstance(self._full_mu, Tensor):
                if self._full_mu.dim() > 1 and self._full_mu.shape[0] > 1:
                    # 有batch维度，提取第i个样本
                    sample_mu = self._full_mu[i : i + 1, valid_indices]
                    sample_sigma = self._full_sigma[i : i + 1, valid_indices]
                else:
                    # 没有batch维度或batch_size=1，直接使用
                    sample_mu = self._full_mu[..., valid_indices]
                    sample_sigma = self._full_sigma[..., valid_indices]
            else:
                sample_mu = self._full_mu[valid_indices]
                sample_sigma = self._full_sigma[valid_indices]

            # 创建只包含这一个样本的分布
            sample_dist = Independent(Normal(sample_mu, sample_sigma), 1)

            # 采样得到形状为[1, valid_dim]的样本
            valid_samples = sample_dist.sample(torch.Size())

            # 填充完整维度，无意义维度填充为默认值0.5
            full_samples = (
                torch.ones(
                    1,
                    self._full_continuous_dim,
                    device=valid_samples.device,
                    dtype=valid_samples.dtype,
                )
                * 0.5
            )
            full_samples[0, valid_indices] = valid_samples[0]
            other_samples_list.append(full_samples)

        if batch_size == 1:
            other_samples = other_samples_list[0]
        else:
            other_samples = torch.cat(other_samples_list, dim=0)

        return torch.cat([dis_samples, other_samples], dim=-1)

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        """
        重参数化采样动作（基于loop_order_logits的argmax确定有效维度）
        返回完整维度，无意义维度填充为默认值0.5

        Args:
            sample_shape: 采样形状

        Returns:
            重参数化采样的动作张量
        """
        # 先采样离散动作
        loop_order_samples = self._loop_order_dist.rsample(sample_shape)
        thread_type_samples = self._thread_type_dist.rsample(sample_shape)
        packing_type_samples = self._packing_type_dist.rsample(sample_shape)
        dis_samples = torch.stack(
            [loop_order_samples, thread_type_samples, packing_type_samples], dim=-1
        )

        # 确定batch_size
        if len(sample_shape) > 0:
            batch_size = sample_shape[0]
        elif loop_order_samples.dim() == 0:
            batch_size = 1
        else:
            batch_size = loop_order_samples.shape[0]

        # 重参数化采样连续动作
        other_samples_list = []
        for i in range(batch_size):
            if batch_size > 1:
                loop_order_val = int(loop_order_samples[i].item())
            else:
                loop_order_val = int(loop_order_samples.item())

            # 获取有效维度索引
            valid_indices = get_valid_continuous_indices_by_loop_order(
                loop_order_val, self._full_continuous_dim
            )

            # 提取第i个样本的mu和sigma（在valid维度上）
            if isinstance(self._full_mu, Tensor):
                if self._full_mu.dim() > 1 and self._full_mu.shape[0] > 1:
                    # 有batch维度，提取第i个样本
                    sample_mu = self._full_mu[i : i + 1, valid_indices]
                    sample_sigma = self._full_sigma[i : i + 1, valid_indices]
                else:
                    # 没有batch维度或batch_size=1，直接使用
                    sample_mu = self._full_mu[..., valid_indices]
                    sample_sigma = self._full_sigma[..., valid_indices]
            else:
                sample_mu = self._full_mu[valid_indices]
                sample_sigma = self._full_sigma[valid_indices]

            # 创建只包含这一个样本的分布
            sample_dist = Independent(Normal(sample_mu, sample_sigma), 1)

            # 重参数化采样得到形状为[1, valid_dim]的样本
            valid_samples = sample_dist.rsample(torch.Size())

            # 填充完整维度，无意义维度填充为默认值0.5
            full_samples = (
                torch.ones(
                    1,
                    self._full_continuous_dim,
                    device=valid_samples.device,
                    dtype=valid_samples.dtype,
                )
                * 0.5
            )
            full_samples[0, valid_indices] = valid_samples[0]
            other_samples_list.append(full_samples)

        if batch_size == 1:
            other_samples = other_samples_list[0]
        else:
            other_samples = torch.cat(other_samples_list, dim=0)

        return torch.cat([dis_samples, other_samples], dim=-1)

    def expand(self, batch_shape: _size, _instance=None):
        """
        扩展分布
        """
        self._loop_order_dist = self._loop_order_dist.expand(batch_shape, _instance)
        self._thread_type_dist = self._thread_type_dist.expand(batch_shape, _instance)
        self._packing_type_dist = self._packing_type_dist.expand(batch_shape, _instance)
        self._other_base_dist = self._other_base_dist.expand(batch_shape, _instance)
        self._other_dist = self._other_dist.expand(batch_shape, _instance)
        self._full_mu = self._full_mu.expand(batch_shape + (self._full_continuous_dim,))
        self._full_sigma = self._full_sigma.expand(
            batch_shape + (self._full_continuous_dim,)
        )
        return self

    @property
    def arg_constraints(self) -> dict[str, Any]:
        """
        返回约束
        """
        return {
            "loop_order_probs": constraints.real,
            "thread_type_probs": constraints.real,
            "packing_type_probs": constraints.real,
            "mu": constraints.real,
            "sigma": constraints.positive,
        }


def get_gemm_action_distribution(
    args: Tensor,
    loop_order_len: int,
    thread_type_len: int,
    packing_type_len: int,
    other_continues_len: int,
) -> Distribution:
    """
    获取GEMM动作分布

    Args:
        args: 网络输出的参数张量，包含:
            - loop_order_logits: loop_order的logits
            - thread_type_logits: thread_type的logits
            - packing_type_logits: packing_type的logits
            - mu: 连续动作的均值
            - sigma: 连续动作的标准差
        loop_order_len: loop_order的类别数
        thread_type_len: thread_type的类别数
        packing_type_len: packing_type的类别数
        other_continues_len: 连续动作维度数

    Returns:
        GemmActionDistribution实例
    """
    thread_type_start = loop_order_len
    packing_type_start = thread_type_start + thread_type_len
    mu_start = packing_type_start + packing_type_len
    sigma_start = mu_start + other_continues_len

    loop_order_logits, thread_type_logits, packing_type_logits, mu, sigma = (
        args[..., :loop_order_len],
        args[..., thread_type_start : thread_type_start + thread_type_len],
        args[..., packing_type_start : packing_type_start + packing_type_len],
        args[..., mu_start : mu_start + other_continues_len],
        args[..., sigma_start:],
    )

    return GemmActionDistribution(
        loop_order_logits,
        thread_type_logits,
        packing_type_logits,
        mu,
        sigma,
        action_split_info={
            "loop_order": 0,
            "thread_type": 1,
            "packing_type": 2,
            "continues": 3,
        },
    )
