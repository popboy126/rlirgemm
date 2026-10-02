# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/6 21:15
# Contact       : hucheng.lew@qq.com
# File          : indicator_monitor.py
# IDE           : PyCharm
# Description   : 指标监控器


import numpy as np
import torch
from tianshou.utils.logger.logger_base import BaseLogger
import logging
import json

class GradientsLogger:
    """梯度记录器"""
    def __init__(self, logger: BaseLogger, logger_interval: int = 500):
        """
        初始化
        @param logger: TensorboardLogger
        @param logger_interval: 梯度记录间隔
        """
        self._logger = logging.getLogger("GradientsLogger")
        self._gradients_logger = logger
        self._logger_interval = logger_interval
        self._last_logging_step = 0

    def do_logging(self, epoch: int, env_step: int, actor: torch.nn.Module, critic: torch.nn.Module, norm_type: int = 2):
        """
        记录梯度信息
        """
        if epoch == 0:
            self.log_gradients_layer_name(actor, critic)

        if env_step - self._last_logging_step >= self._logger_interval:
            self._logger.info(f"[Epoch {epoch}] Training step gradients logging triggered at env step {env_step}.")
            actor_grad_info = self.get_gradient_norms(actor, norm_type)
            actor_log_data = self._gradients_logger.prepare_dict_for_logging({
                'actor/total_norm': actor_grad_info['total_norm'],
                'actor/max_grad': actor_grad_info['max_grad'],
                'actor/min_grad': actor_grad_info['min_grad'],
                'actor/mean_grad': actor_grad_info['mean_grad'],
                'actor/zero_ratio': actor_grad_info['zero_grad_ratio'],
                **{f'actor/{layer_name}_norm': layer_norm
                   for layer_name, layer_norm in actor_grad_info['layer_norms'].items()}
            })
            self._gradients_logger.write("training_gradients/env_step", env_step, actor_log_data)

            critic_grad_info = self.get_gradient_norms(critic, norm_type)
            critic_log_data = self._gradients_logger.prepare_dict_for_logging({
                'critic/total_norm': critic_grad_info['total_norm'],
                'critic/max_grad': critic_grad_info['max_grad'],
                'critic/min_grad': critic_grad_info['min_grad'],
                'critic/mean_grad': critic_grad_info['mean_grad'],
                'critic/zero_ratio': critic_grad_info['zero_grad_ratio'],
                **{f'critic/{layer_name}_norm': layer_norm
                   for layer_name, layer_norm in critic_grad_info['layer_norms'].items()}
            })
            self._gradients_logger.write("training_gradients/env_step", env_step, critic_log_data)
            self._last_logging_step = env_step

    @staticmethod
    def get_gradient_norms(model, norm_type=2):
        """
        获取模型所有参数的梯度范数
        Args:
            model: PyTorch模型
            norm_type: 范数类型，2表示L2范数
        Returns:
            dict: 包含各层梯度信息的字典
        """
        total_norm = 0
        gradients_info = {
            'total_norm': 0,
            'layer_norms': {},
            'max_grad': 0,
            'min_grad': 0,
            'mean_grad': 0,
            'zero_grad_ratio': 0
        }

        grad_list = []
        zero_count = 0
        total_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type).item()
                total_norm += param_norm ** norm_type
                gradients_info['layer_norms'][name] = param_norm

                # 收集梯度值用于统计
                grad_flat = param.grad.data.cpu().numpy().flatten()
                grad_list.extend(grad_flat)

                zero_count += np.sum(grad_flat == 0)
                total_count += len(grad_flat)

        # 计算总范数
        total_norm = total_norm ** (1. / norm_type)
        gradients_info['total_norm'] = total_norm

        # 统计信息
        if grad_list:
            gradients_info['max_grad'] = np.max(np.abs(grad_list))
            gradients_info['min_grad'] = np.min(np.abs(grad_list))
            gradients_info['mean_grad'] = np.mean(np.abs(grad_list))
            gradients_info['zero_grad_ratio'] = zero_count / total_count if total_count > 0 else 0

        return gradients_info

    @staticmethod
    def get_weight_norms(model, norm_type=2):
        """获取权重范数"""
        weight_norms = {}
        for name, param in model.named_parameters():
            if param.data is not None:
                weight_norms[name] = param.data.norm(norm_type).item()
        return weight_norms

    def log_gradients_layer_name(self, actor: torch.nn.Module, critic: torch.nn.Module):
        """
        记录梯度层名称
        """
        layer_name = [
            'actor/total_norm',
            'actor/max_grad',
            'actor/min_grad',
            'actor/mean_grad',
            'actor/zero_ratio',
        ]

        for name, _ in actor.named_parameters():
            layer_name.append(f'actor/{name}_norm')

        layer_name.extend([
            'critic/total_norm',
            'critic/max_grad',
            'critic/min_grad',
            'critic/mean_grad',
            'critic/zero_ratio',
        ])

        for name, _ in critic.named_parameters():
            layer_name.append(f'critic/{name}_norm')

        self._logger.info(f"gradients layer names: {json.dumps(layer_name)}")


