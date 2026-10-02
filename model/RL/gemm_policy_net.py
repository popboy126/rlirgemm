# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/3/31 22:37
# Contact       : hucheng.lew@qq.com
# File          : gemm_policy_net.py
# IDE           : PyCharm
# Description   : 策略网络（方案2：简化结构，移除过度梯度限制）

import torch
import numpy as np
from tianshou.data.types import TObs
from tianshou.utils.torch_utils import torch_device
from tianshou.utils.net.common import MLP, ModuleType, Actor, ModuleWithVectorOutput
from tianshou.utils.net.continuous import T, ContinuousActorProbabilistic, ContinuousCritic
from typing import Any, Tuple, Sequence
from itertools import chain
import logging


def _init_linear(module, is_output=False, gain=1.0) -> None:
    """
    初始化线性层
    """
    if isinstance(module, torch.nn.Linear):
        if is_output:
            # 输出层：使用较小权重的正交初始化
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
        else:
            # 隐藏层：使用Kaiming初始化（适合ELU）
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_in", nonlinearity="relu"
            )
            # 调整增益
            with torch.no_grad():
                module.weight.data *= gain

        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class GemmAttentionNet(ModuleWithVectorOutput):
    """
    简化的注意力网络 - 移除过度clamp，优化梯度流
    """

    def __init__(
        self, input_dim: int, output_dim: int, use_attention: bool = False,
            attention_net_clamp_enable: bool = True,
            net_logits_min: float = -50.0,
            net_logits_max: float = 50.0,
            attention_net_use_layer_norm: bool = True,
    ) -> None:
        """
        初始化
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            use_attention: 是否使用attention（默认False，使用纯MLP更稳定）
        """
        super().__init__(output_dim)
        self._num_attention_heads = 4
        self._use_attention = use_attention
        self._logits_min = net_logits_min
        self._logits_max = net_logits_max
        self._attention_net_clamp_enable = attention_net_clamp_enable
        self._attention_net_use_layer_norm = attention_net_use_layer_norm
        self.input_dim = input_dim


        if use_attention:
            self._hidden_dim = 4 * output_dim  # 从4x增大到8x
            self._embedding_dim = self._hidden_dim * 2
            self._ff_hidden_dim = self._embedding_dim * 2
            (
                self._state_encoder,
                self._attention,
                self._attention_norm,
                self._attention_ff,
                self._ff_norm,
                self._feature_enhancer,
            ) = self.get_net_with_attention()
            self._state_encoder_output = None
        else:
            # 【MLP优化】增大隐藏层维度以弥补架构劣势
            self._hidden_dim = 32  # 从4x增大到8x
            self._embedding_dim = self._hidden_dim * 2
            self._ff_hidden_dim = self._embedding_dim * 2

            # 【修改】使用纯MLP替代Attention，更稳定且适合单步输入
            self._state_encoder, self._state_encoder_output = self.get_mlp_net()
            # 占位符，不使用
            self._attention = None
            self._attention_norm = None
            self._attention_ff = None
            self._ff_norm = None
            self._feature_enhancer = None

        self._init_weights()  # 初始化权重

    def get_mlp_net(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        【MLP优化】更深的网络结构，4层MLP替代3层
        """
        if self._attention_net_use_layer_norm:
            state_encoder = torch.nn.Sequential(
                # 第1层
                torch.nn.Linear(self.input_dim, self._hidden_dim),
                torch.nn.ELU(),
                torch.nn.LayerNorm(self._hidden_dim),
                torch.nn.Dropout(0.1),
                # 第2层
                torch.nn.Linear(self._hidden_dim, self._embedding_dim),
                torch.nn.ELU(),
                torch.nn.LayerNorm(self._embedding_dim),
                torch.nn.Dropout(0.1),
                # 第3层（新增）
                torch.nn.Linear(self._embedding_dim, self._ff_hidden_dim),
                torch.nn.ELU(),
                torch.nn.LayerNorm(self._ff_hidden_dim),
                torch.nn.Dropout(0.1),
                # 第4层（新增）
                torch.nn.Linear(self._ff_hidden_dim, self._ff_hidden_dim),
                torch.nn.ELU(),
                torch.nn.LayerNorm(self._ff_hidden_dim),
                torch.nn.Dropout(0.1),

            )
        else:
            state_encoder = torch.nn.Sequential(
                # 第1层
                torch.nn.Linear(self.input_dim, self._hidden_dim),
                torch.nn.ELU(),
                torch.nn.Dropout(0.1),
                # 第2层
                torch.nn.Linear(self._hidden_dim, self._embedding_dim),
                torch.nn.ELU(),
                torch.nn.Dropout(0.1),
                # 第3层（新增）
                torch.nn.Linear(self._embedding_dim, self._ff_hidden_dim),
                torch.nn.ELU(),
                torch.nn.Dropout(0.1),
                # 第4层（新增）
                torch.nn.Linear(self._ff_hidden_dim, self._ff_hidden_dim),
                torch.nn.ELU(),
                torch.nn.Dropout(0.1),
            )

        state_encoder_output = torch.nn.Sequential(
            # 输出层
            torch.nn.Linear(self._ff_hidden_dim, self.output_dim),
        )
        return state_encoder,  state_encoder_output

    def get_net_with_attention(
        self,
    ) -> Tuple[
        torch.nn.Module,
        torch.nn.Module,
        torch.nn.Module,
        torch.nn.Module,
        torch.nn.Module,
        torch.nn.Module,
    ]:
        """
        获取带Attention的网络结构（保留但简化）
        """
        # 特征编码层
        state_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self._hidden_dim),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self._hidden_dim),
            # 【修改】降低dropout
            torch.nn.Dropout(0.05),
            torch.nn.Linear(self._hidden_dim, self._embedding_dim),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self._embedding_dim),
            torch.nn.Dropout(0.05),
        )

        # 自注意力层
        attention = torch.nn.MultiheadAttention(
            embed_dim=self._embedding_dim,
            num_heads=self._num_attention_heads,
            dropout=0.05,  # 【修改】降低dropout
            batch_first=True,
            bias=True,
        )

        # 注意力后的归一化和前馈
        attention_norm = torch.nn.LayerNorm(self._embedding_dim)
        attention_ff = torch.nn.Sequential(
            torch.nn.Linear(self._embedding_dim, self._ff_hidden_dim),
            torch.nn.ELU(),
            torch.nn.Dropout(0.05),  # 【修改】降低dropout
            torch.nn.Linear(self._ff_hidden_dim, self._embedding_dim),
            torch.nn.ELU(),
        )
        ff_norm = torch.nn.LayerNorm(self._embedding_dim)

        # 特征增强层
        feature_enhancer = torch.nn.Sequential(
            torch.nn.Linear(self._embedding_dim, self._embedding_dim),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self._embedding_dim),
            torch.nn.Dropout(0.05),  # 【修改】降低dropout
            torch.nn.Linear(self._embedding_dim, self.output_dim),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.output_dim),
        )

        return (
            state_encoder,
            attention,
            attention_norm,
            attention_ff,
            ff_norm,
            feature_enhancer,
        )

    def _init_weights(self):
        """优化的权重初始化"""
        if self._use_attention and self._attention is not None:
            for module in self._state_encoder.modules():
                _init_linear(module, is_output=False, gain=1.0)
            # 注意力层（特殊处理）
            for name, param in self._attention.named_parameters():
                if "weight" in name:
                    if "in_proj_weight" in name:
                        # 【修改】增大初始化增益，帮助梯度传播
                        torch.nn.init.xavier_uniform_(param, gain=0.1)
                    elif "out_proj.weight" in name:
                        torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

            # 其他隐藏层
            if self._attention_ff is not None and self._feature_enhancer is not None:
                for module in chain(
                    self._attention_ff.modules(), self._feature_enhancer.modules()
                ):
                    _init_linear(module, is_output=False, gain=1.0)
        elif not self._use_attention and (self._state_encoder is not None) and (self._state_encoder_output is not None):
            for module in self._state_encoder.modules():
                _init_linear(module, is_output=False, gain=1.0)

            for module in self._state_encoder_output.modules():
                _init_linear(module, is_output=True, gain=0.01)
        else:
            pass

    def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 移除过度clamp
        """
        device = torch_device(self)
        obs = torch.as_tensor(
            obs,
            device=device,
            dtype=torch.float32,
        )

        if not self._use_attention:
            # 【修改】纯MLP前向，无多余clamp
            features = self._state_encoder(obs)
            features = self._state_encoder_output(features)
            # # 只在最终输出层进行宽松clamp
            # features = torch.clamp(features, min=self._logits_min, max=self._logits_max)
            return features
        else:
            # ============ 1. 编码输入状态 ============
            x = self._state_encoder(obs)  # [B, hidden_dim]
            # 【修改】减少clamp次数，只在关键位置使用
            # x = torch.clamp(x, min=self._logits_min, max=self._logits_max)

            # ============ 2. 自注意力处理 ============
            if self._use_attention and self._attention is not None:
                x_seq = x.unsqueeze(1)  # [B, 1, hidden_dim]

                # 应用多头注意力
                attn_output, attn_weights = self._attention(
                    query=x_seq, key=x_seq, value=x_seq, need_weights=False
                )
                # 【修改】移除clamp，保留梯度

                # 残差连接 + 层归一化
                x = x + attn_output.squeeze(1)
                if self._attention_norm is not None:
                    x = self._attention_norm(x)

                # 前馈网络 + 残差
                if self._attention_ff is not None:
                    ff_output = self._attention_ff(x)
                    x = x + ff_output
                if self._ff_norm is not None:
                    x = self._ff_norm(x)

                # ============ 3. 特征增强 ============
                if self._feature_enhancer is not None:
                    features = self._feature_enhancer(x)  # [B, final_dim]
                else:
                    features = x
            else:
                features = x

            if self._attention_net_clamp_enable:
                # 【修改】只在最终输出层进行宽松clamp
                features = torch.clamp(features, min=self._logits_min, max=self._logits_max)

            return features


class GaussianActor(ContinuousActorProbabilistic):
    """
    高斯actor - 优化梯度流
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化
        """
        sigma_min: float = kwargs.pop("sigma_min", 1e-5)
        sigma_max: float = kwargs.pop("sigma_max", 2.0)
        super().__init__(*args, **kwargs)

        if np.isnan(sigma_min) or np.isnan(sigma_max):
            raise ValueError(
                f"sigma_min and sigma_max must not be NaN, got sigma_min={sigma_min}, sigma_max={sigma_max}"
            )
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be greater than sigma_min, got sigma_min={sigma_min}, sigma_max={sigma_max}"
            )

        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(
        self,
        obs: TObs,
        state: T | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], T | None]:
        if info is None:
            info = {}
        logits = self.preprocess(obs)
        mu = self.mu(logits)

        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)

        if self._c_sigma:
            log_sigma = torch.clamp(
                self.sigma(logits),
                min=np.log(self._sigma_min),
                max=np.log(self._sigma_max),
            )
            sigma = torch.exp(log_sigma)
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            log_sigma = torch.clamp(
                self.sigma_param.view(shape) + torch.zeros_like(mu),
                min=np.log(self._sigma_min),
                max=np.log(self._sigma_max),
            )
            sigma = torch.exp(log_sigma)

        # # 【修改】放宽mu的clamp范围
        # mu = torch.clamp(mu, min=-20.0, max=20.0)  # 从-10,10放宽到-20,20
        # sigma = torch.clamp(sigma, min=self._sigma_min, max=self._sigma_max)
        return (mu, sigma), state


class ResidualBlock(ModuleWithVectorOutput):
    """简单的残差块：输入 x -> fc -> elu -> fc + x -> elu"""
    def __init__(self, input_dim: int, output_dim: int, activation: ModuleType, dropout: float = 0.1):
        super().__init__(output_dim)
        self.fc1 = torch.nn.Linear(input_dim, output_dim)
        self.fc2 = torch.nn.Linear(output_dim, output_dim)
        self.act = activation()
        # 投影短路：如果输入输出维度不同，需要一个线性层调整
        self.dropout = torch.nn.Dropout(dropout)
        self.shortcut = torch.nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        前向计算
        """
        device = torch_device(self)
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        out = self.act(self.fc1(obs))
        out = self.dropout(out)
        out = self.fc2(out)
        if self.shortcut is not None:
            obs = self.shortcut(obs)
        out = out + obs
        out = self.act(out)
        return out


# 构建带残差的离散头
class ResidualMLP(ModuleWithVectorOutput):
    """
    带残差连接的多层感知机，支持任意维度序列
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int]=(64, 64), activation: ModuleType =torch.nn.ELU):
        """
        初始化
        """
        super().__init__(output_dim)
        self.model = self.get_model(input_dim, output_dim, hidden_sizes, activation)

    def get_model(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int], activation: ModuleType) -> torch.nn.Sequential:
        """
        获取模型
        """
        layers = []
        prev_dim = input_dim
        # 遍历隐藏层维度
        for i, dim in enumerate(hidden_sizes):
            layers.append(ResidualBlock(prev_dim, dim, activation))
            # # 如果当前层与前一层维度相同，使用残差块；否则使用普通线性层
            # if dim == prev_dim:
            #     layers.append(ResidualBlock(prev_dim, dim, activation))
            # else:
            #     # 普通线性层（无残差）
            #     layers.append(torch.nn.Linear(prev_dim, dim))
            #     layers.append(activation())
            prev_dim = dim
        # 输出层
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        前向计算
        """
        device = torch_device(self)
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        return self.model(obs)


class GemmHyperActor(Actor):
    """
    混合动作空间的actor网络 - 优化版
    """

    def __init__(
        self,
        obs_dim: int,
        loop_order_size: int,
        packing_type_size: int,
        thread_type_size: int,
        continuous_action_dim: int,
        sigma_min: float,
        sigma_max: float,
        attention_net_output_dim: int,
        use_attention: bool = False,  # 【新增】控制是否使用attention
        attention_net_clamp_enable: bool = True,
        output_net_clamp_enable: bool = False,
        net_logits_min: float = -50.0,
        net_logits_max: float = 50.0,
        attention_net_use_layer_norm: bool = True,
    ):
        output_dim = (
            loop_order_size
            + packing_type_size
            + thread_type_size
            + continuous_action_dim
        )
        super().__init__(output_dim)
        self.logger = logging.getLogger("GemmHyperActor")
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        # 【修改】放宽clamp范围
        self._logits_min = net_logits_min  # 从-20放宽
        self._logits_max = net_logits_max  # 从20放宽
        self._attention_net = GemmAttentionNet(
            input_dim=obs_dim,
            output_dim=attention_net_output_dim,
            use_attention=use_attention,  # 【新增】
            attention_net_clamp_enable=attention_net_clamp_enable,
            net_logits_min=net_logits_min,
            net_logits_max=net_logits_max,
            attention_net_use_layer_norm=attention_net_use_layer_norm,

        )
        (
            self._loop_order_head,
            self._packing_type_head,
            self._thread_type_head,
            self._continuous_net,
        ) = self.get_actor_net(
            attention_net_output_dim,
            loop_order_size,
            packing_type_size,
            thread_type_size,
            continuous_action_dim,
        )
        dims = (
            self._loop_order_head.output_dim
            + self._packing_type_head.output_dim
            + self._thread_type_head.output_dim
            + self._continuous_net.output_dim
        )
        assert dims == output_dim, (
            f"网络输出维度 {dims} 与 期望输出维度 {output_dim} 不匹配"
        )
        self._init_weights()

    def get_preprocess_net(self) -> ModuleWithVectorOutput:
        """
        返回预处理网络
        """
        return self._attention_net

    def get_actor_net(
        self,
        attention_output_dim: int,
        loop_order_size: int,
        packing_type_size: int,
        thread_type_size: int,
        continuous_action_dim: int,
    ) -> Tuple[ResidualMLP, ResidualMLP, ResidualMLP, ContinuousActorProbabilistic]:
        """
        获取actor网络 - 增大网络容量
        """

        # 【修改】增大隐藏层尺寸，帮助学习更高精度策略
        loop_order_head = ResidualMLP(
            input_dim=attention_output_dim,
            output_dim=loop_order_size,
            hidden_sizes=[128, 128, 64],  # 从[128,128]增大到[256,256,128]
            activation=torch.nn.ELU,
        )

        packing_type_head = ResidualMLP(
            input_dim=attention_output_dim,
            output_dim=packing_type_size,
            hidden_sizes=[128, 128, 64],  # 增大
            activation=torch.nn.ELU,
        )

        thread_type_head = ResidualMLP(
            input_dim=attention_output_dim,
            output_dim=thread_type_size,
            hidden_sizes=[128, 128, 64],  # 增大
            activation=torch.nn.ELU,
        )

        # 【修改】增大预处理网络
        prfm_stride_pre_prepocess_net = MLP(
            input_dim=attention_output_dim,
            output_dim=32,  # 从256增大到512
            hidden_sizes=[128, 128, 64],  # 增加容量
            activation=[torch.nn.ELU, torch.nn.ELU, torch.nn.ELU],
        )

        prfm_stride_head = GaussianActor(
            preprocess_net=prfm_stride_pre_prepocess_net,
            action_shape=continuous_action_dim,
            unbounded=True,
            conditioned_sigma=True,
            sigma_min=self._sigma_min,
            sigma_max=self._sigma_max,
        )

        return loop_order_head, packing_type_head, thread_type_head, prfm_stride_head

    def _init_weights(self):
        """
        初始化网络权重 - 增大输出层初始化增益
        """
        # 离散动作头
        for head in [
            self._loop_order_head,
            self._packing_type_head,
            self._thread_type_head,
        ]:
            for i, module in enumerate(head.modules()):
                if isinstance(module, torch.nn.Linear):
                    is_output = i == len(list(head.modules())) - 1
                    # 【修改】输出层使用更大的gain，帮助梯度传播
                    _init_linear(
                        module, is_output=is_output, gain=0.1 if is_output else 1.0
                    )

        # 连续动作头
        for i, module in enumerate(self._continuous_net.get_preprocess_net().modules()):
            if isinstance(module, torch.nn.Linear):
                _init_linear(module, is_output=False, gain=1.0)

        for i, module in enumerate(self._continuous_net.mu.modules()):
            if isinstance(module, torch.nn.Linear):
                is_output = i == len(list(self._continuous_net.mu.modules())) - 1
                # 【修改】输出层使用更大的gain
                _init_linear(
                    module, is_output=is_output, gain=0.1 if is_output else 1.0
                )

    def forward(
        self,
        obs: TObs,
        state: T | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor | Sequence[torch.Tensor], T | None]:
        if info is None:
            info = {}

        attention_features = self._attention_net(obs)
        # # 【修改】使用更宽松的clamp范围
        # attention_features = torch.clamp(
        #     attention_features, min=self._logits_min, max=self._logits_max
        # )

        loop_order_logits = self._loop_order_head(attention_features)
        # 【修改】移除或放宽中间层的clamp
        # loop_order_logits_clamped = torch.clamp(loop_order_logits, min=self._logits_min, max=self._logits_max)
        loop_order_logits_clamped = loop_order_logits  # 不移除，让梯度自由流动

        packing_type_logits = self._packing_type_head(attention_features)
        # packing_type_logits_clamped = torch.clamp(packing_type_logits, min=self._logits_min, max=self._logits_max)
        packing_type_logits_clamped = packing_type_logits

        thread_type_logits = self._thread_type_head(attention_features)
        # thread_type_logits_clamped = torch.clamp(thread_type_logits, min=self._logits_min, max=self._logits_max)
        thread_type_logits_clamped = thread_type_logits

        (cont_mu, cont_sigma), state = self._continuous_net(
            attention_features, state, info
        )

        # cont_sigma = torch.clamp(cont_sigma, min=1e-5, max=self._sigma_max)

        out = torch.cat(
            [
                loop_order_logits_clamped,
                thread_type_logits_clamped,
                packing_type_logits_clamped,
                cont_mu,
                cont_sigma,
            ],
            dim=-1,
        )

        return out, state


class GemmContinuousCritic(ContinuousCritic):
    """
    critic网络 - 优化版
    """

    def __init__(
        self, obs_dim: int, attention_net_output_dim: int, use_attention: bool = False,
        attention_net_clamp_enable: bool = True,
        output_net_clamp_enable: bool = False,
        net_logits_min: float = -50.0,
        net_logits_max: float = 50.0,
        attention_net_use_layer_norm: bool = True,
    ):
        self.logger = logging.getLogger("GemmContinuousCritic")
        super().__init__(
            preprocess_net=self.get_critic_net(
                obs_dim, attention_net_output_dim, use_attention,
                attention_net_clamp_enable, net_logits_min, net_logits_max,
                attention_net_use_layer_norm
            ),
            apply_preprocess_net_to_obs_only=True,
        )
        self._output_net_clamp_enable = output_net_clamp_enable
        self._init_weights()

    def get_critic_net(
        self, obs_dim: int, attention_net_output_dim: int, use_attention: bool = False,
        attention_net_clamp_enable: bool = True,
        net_logits_min: float = -50.0,
        net_logits_max: float = 50.0,
        attention_net_use_layer_norm: bool = True,
    ) -> ModuleWithVectorOutput:
        """
        获取critic网络
        """
        output_dim = 16
        attention_net = torch.nn.Sequential(
            GemmAttentionNet(
                input_dim=obs_dim,
                output_dim=attention_net_output_dim,
                use_attention=use_attention,
                attention_net_clamp_enable=attention_net_clamp_enable,
                net_logits_min=net_logits_min,
                net_logits_max=net_logits_max,
                attention_net_use_layer_norm=attention_net_use_layer_norm,
            ),
            ResidualMLP(
                input_dim=attention_net_output_dim,
                output_dim=output_dim,
                hidden_sizes=[128, 128, 64, 32],  # 增大
                activation=torch.nn.ELU,
            )

        )

        return ModuleWithVectorOutput.from_module(attention_net, output_dim)

    def _init_weights(self):
        """
        初始化网络权重
        """
        for i, module in enumerate(self.last.modules()):
            if isinstance(module, torch.nn.Linear):
                is_output = i == len(list(self.last.modules())) - 1
                _init_linear(
                    module,
                    is_output=is_output,
                    gain=0.1 if is_output else 1.0,  # 【修改】增大gain
                )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""
        device = torch_device(self)
        obs_tensor = torch.as_tensor(
            obs,
            device=device,
            dtype=torch.float32,
        )
        if self.apply_preprocess_net_to_obs_only:
            obs_tensor = self.preprocess(obs_tensor)
        obs_tensor = obs_tensor.view(obs_tensor.size(0), -1)
        if act is not None:
            act_tensor = torch.as_tensor(
                act,
                device=device,
                dtype=torch.float32,
            )
            act_tensor = act_tensor.view(act_tensor.size(0), -1)
            obs_tensor = torch.cat([obs_tensor, act_tensor], dim=1)
        if not self.apply_preprocess_net_to_obs_only:
            obs_tensor = self.preprocess(obs_tensor)
        return self.last(obs_tensor)
