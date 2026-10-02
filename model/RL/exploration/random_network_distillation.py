# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/23 10:46
# Contact       : hucheng.lew@qq.com
# File          : random_network_distillation.py
# IDE           : PyCharm
# Description   : 随机网络蒸馏（RND）实现

from tianshou.data.types import TObs
import torch
import numpy as np
import logging



class RNDModule(torch.nn.Module):
    """
    随机网络蒸馏模块
    """
    def __init__(self, obs_dim: int, loop_order_size: int, packing_type_size: int, thread_type_size: int, continuous_action_dim: int, hidden_dim: int = 128, device='cpu', reset_interval: int = 1000):
        """
        构造函数
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param hidden_dim: 隐藏层维度
        """
        super(RNDModule, self).__init__()
        self.logger = logging.getLogger("RNDModule")
        self._embed_loop_order = torch.nn.Embedding(loop_order_size, loop_order_size.bit_length())
        self._embed_packing_type = torch.nn.Embedding(packing_type_size, packing_type_size.bit_length())
        self._embed_thread_type = torch.nn.Embedding(thread_type_size, thread_type_size.bit_length())
        self._other_continues_project = torch.nn.Sequential(
            torch.nn.Linear(continuous_action_dim, 64), # 升维
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64), # 输出
        )
        self._action_embed_dim = loop_order_size.bit_length() + packing_type_size.bit_length() + thread_type_size.bit_length() + 64
        self._input_dim = obs_dim + self._action_embed_dim
        # 定义目标网络（固定不变）
        self.target_network = torch.nn.Sequential(
            torch.nn.Linear(self._input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 64)
        )
        for param in self.target_network.parameters():
            param.requires_grad = False  # 冻结目标网络参数

        # 定义预测网络（需要训练）
        self.predictor_network = torch.nn.Sequential(
            torch.nn.Linear(self._input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 64)
        )
        self._device = device
        self._reset_interval = reset_interval
        self.to(self._device)

    @property
    def predictor(self)-> torch.nn.Module:
        return self.predictor_network

    def reset_predictor(self, epoch: int, env_step: int) -> bool:
        """
        重置预测网络参数，用于周期性恢复RND探索能力。
        当predictor过度拟合target后，内在奖励趋近于零，
        重置predictor可以重新激活探索信号。
        """
        if epoch > 0 and (self._reset_interval > 0) and (epoch % self._reset_interval == 0):
            for module in self.predictor_network.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.orthogonal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            return True
        return False

    def _process_inputs(self, obs: TObs, act: TObs) -> torch.Tensor:
        """
        处理输入
        """
        # act shape: [batch, 10]
        # act[:, 0]: A (离散 ID)
        # act[:, 1]: B (离散 ID)
        # act[:, 2:]: D-K (具体的数值，Float)

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self._device)

        # 这里需要注意：act 混合了 int 和 float，通常 Tianshou 的 batch.act 可能是 numpy array
        if isinstance(act, np.ndarray):
            act_tensor = torch.tensor(act, dtype=torch.float32, device=self._device)
        else:
            act_tensor = act.float()

        # --- [修改点 2] 处理逻辑变化 ---

        # 1. 处理 A 和 B (需要 Long 类型做索引)
        loop_order_type = act_tensor[:, 0].long()
        thread_type = act_tensor[:, 1].long()
        packing_type = act_tensor[:, 2].long()
        embed_loop_order_type = self._embed_loop_order(loop_order_type)
        embed_thread_type = self._embed_thread_type(thread_type)
        embed_packing_type = self._embed_packing_type(packing_type)

        # 2. 处理 D-K (直接使用 Float 数值)
        # 取出后8列
        vals_large = act_tensor[:, 3:]  # shape: [batch, 8]

        # 建议：对数值进行归一化处理，防止数值过大导致梯度爆炸
        # 如果你知道参数范围大概是 [0, 1000]，最好除以 1000
        # vals_large = vals_large / 1000.0

        other_continues_project = self._other_continues_project(vals_large) # shape: [batch, 64]

        # 3. 拼接
        action_embedding = torch.cat([embed_loop_order_type, embed_thread_type, embed_packing_type, other_continues_project], dim=1)

        obs = obs.flatten(start_dim=1)
        return torch.cat([obs, action_embedding], dim=1)

    def forward(self, obs: TObs, act: TObs) -> torch.Tensor:
        """
        计算 Predictor 的损失 (用于更新 RND)
        :param obs: 观察空间状态
        :param act: 动作空间状态
        :return: 损失张量
        """
        inputs = self._process_inputs(obs, act)
        target_feature = self.target_network(inputs)
        predict_feature = self.predictor_network(inputs)
        loss = torch.nn.functional.mse_loss(predict_feature, target_feature)
        return loss

    def compute_intrinsic_reward(self, obs, act) -> torch.Tensor:
        """
        计算内在奖励 (不进行梯度反向传播)
        :param x: 输入张量
        :return: 内在奖励张量
        """
        with torch.no_grad():
            inputs = self._process_inputs(obs, act)
            target_feature = self.target_network(inputs)
            predict_feature = self.predictor_network(inputs)
            dist = (target_feature - predict_feature).pow(2).mean(dim=1, keepdim=False)
            return dist



