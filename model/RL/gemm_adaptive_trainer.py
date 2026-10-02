# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/4/1 09:41
# Contact       : hucheng.lew@qq.com
# File          : gemm_train.py
# IDE           : PyCharm
# Description   : 控制训练的代码

import logging
import os.path
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from collectors.gemm_test_collect_stats import GEMMTestCollectStats
from exploration.adaptive_dual_coefficient_balancer import (
    AdaptiveDualCoefficientBalancer,
)
from exploration.adaptive_entropy_balancer import AdaptiveEntropyBalancer
from exploration.adaptive_entropy_beta_head_balancer import (
    AdaptiveEntropyBetaHeadBalancer,
)
from exploration.adaptive_entropy_controller import (
    AdaptiveEntropyControllerCont,
    AdaptiveEntropyControllerContV2,
    AdaptiveEntropyControllerDisc,
    compute_max_entropy_for_gaussian,
)
from exploration.adaptive_entropy_controller_by_performance import (
    AdaptiveEntropyControllerByPerformance,
)
from exploration.adaptive_entropy_controller_by_trend import (
    AdaptiveEntropyControllerByTrend,
)
from exploration.cache_manager import (
    DiscreteActionCacheManager,
    ShapePerformanceCacheManager,
)
from exploration.random_network_distillation import RNDModule
from gemm_action_distribution import get_gemm_action_distribution
from gemm_adam_optimizer import GemmAdamOptimizer
from gemm_env import make_gemm_env
from gemm_policy import GemmPPOPolicy
from gemm_policy_net import GemmContinuousCritic, GemmHyperActor
from gradients_logger import GradientsLogger
from gymnasium import Env
from tianshou.algorithm import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.highlevel.env import VectorEnvType
from tianshou.highlevel.logger import LoggerFactoryDefault, TLogger
from tianshou.trainer import OnPolicyTrainerParams


class GemmAdaptiveTrainer(object):
    """
    GEMM训练类
    """

    def __init__(self, **kwargs):
        """
        初始化
        """
        self.logger = logging.getLogger("GemmAdaptiveTrainer")
        self._is_debug = kwargs.get("is_debug", False)
        self._is_train = kwargs.get("is_train", True)
        if self._is_train:
            self.logger.info("Trainer initialized in training mode.")
            self.train_config: Dict[str, Any] = kwargs.get("train_config", {})
            self.gemm_env_configs = kwargs.get("gemm_env_config", {})
            self.model_save_dir: str = self.train_config.get("model_save_dir", "")
            self.restore_checkpoint_name: str = self.train_config.get(
                "restore_checkpoint_name", ""
            )

            checkpoint = self.load_checkpoint(
                self.model_save_dir, self.restore_checkpoint_name
            )
            if isinstance(checkpoint, dict):
                # train_config = checkpoint["train_configs"]
                # self.train_config.update(train_config)
                # self.model_save_dir: str = self.train_config.get("model_save_dir", "")
                # self.restore_checkpoint_name: str = self.train_config.get("restore_checkpoint_name", "")
                self.logger.info(
                    "Checkpoint loaded, training will resume from the loaded state."
                )
            else:
                self.logger.info(
                    "No valid checkpoint found, initializing new training session."
                )
        else:
            self.logger.info("Trainer initialized in inference mode.")
            model_save_dir: str = kwargs.get("model_save_dir", "")
            checkpoint_name: str = kwargs.get("checkpoint_name", "")
            checkpoint = self.load_checkpoint(model_save_dir, checkpoint_name)
            if not isinstance(checkpoint, dict):
                model_path = os.path.join(model_save_dir, checkpoint_name)
                raise ValueError(
                    f"Failed to load model from {model_path} for inference."
                )
            self.train_config: Dict[str, Any] = checkpoint["train_configs"]
            self.train_config.update(
                {  # 做一些推理时的配置覆盖
                    "logger_dir": kwargs.get("logger_dir", "./logs"),
                    "device": kwargs.get("device", "cpu"),
                    "trainer_name": kwargs.get("inference_name", "gemm_inference"),
                    "train_env_num": 1,
                    "test_env_num": 1,
                }
            )
            self.gemm_env_configs: Dict[str, Any] = checkpoint["gemm_env_configs"]
            self.gemm_env_configs.update(
                {
                    "use_real_agent": False,  # 不使用真实的agent进行推理
                    "is_pse_inference": True,  # 使用pse进行推理
                    "is_debug": False,
                }
            )
            self.logger.info("Checkpoint loaded for inference.")

        self.checkpoint_save_interval = self.train_config.get(
            "checkpoint_save_interval", 5
        )  # 每隔多少epoch保存一次checkpoint
        self.checkpoint_max_num = self.train_config.get(
            "checkpoint_max_num", 5
        )  # 最多保存多少个checkpoint，超过后覆盖最旧的
        self.best_policy_max_num = self.train_config.get(
            "best_policy_max_num", 5
        )  # 最多保存多少个best policy
        self.best_performance_max_num = self.train_config.get(
            "best_performance_max_num", 5
        )
        self.best_performance_improve_threshold = self.train_config.get(
            "best_performance_improve_threshold", 0.05
        )
        self._last_best_env_performance = 0.0  # 记录上一次保存的最好性能

        device = self.train_config.get("device", None)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.logger.info(f"Using device: {self.device}")
        self.trainer_name = self.train_config.get("trainer_name", "default_task")

        self.gemm_env_task = self.train_config.get("gemm_env_task", "gemm_env_task")
        self.gemm_env_seed = self.train_config.get("gemm_env_seed", 13289)
        self.gemm_env_train_num = self.train_config.get("train_env_num", 1)
        self.gemm_env_test_num = self.train_config.get("test_env_num", 1)
        if self._is_debug:
            self.gemm_env_venv_type = VectorEnvType.DUMMY
        else:
            self.gemm_env_venv_type = VectorEnvType.SUBPROC_SHARED_MEM_AUTO

        if isinstance(checkpoint, dict):
            shape_perf_cache_dumps = checkpoint.get("cache_manager", {}).get(
                "shape_performance", None
            )
            action_stat_cache_dumps = checkpoint.get("cache_manager", {}).get(
                "action_stat", None
            )
        else:
            shape_perf_cache_dumps = None
            action_stat_cache_dumps = None

        self._shape_perf_cache_manager, self._disc_action_stat_cache_manager = (
            self.create_cache_manager(
                shape_perf_ck_data=shape_perf_cache_dumps,
                action_stat_ck_data=action_stat_cache_dumps,
            )
        )

        cache_manager = {}
        if self._shape_perf_cache_manager:
            shm_name = self._shape_perf_cache_manager.get_shm_name()
            lock = self._shape_perf_cache_manager.get_lock()
            cache_manager.update(
                {
                    "shape_perf_shm_name": shm_name,
                    "shape_perf_lock": lock,
                }
            )

        if self._disc_action_stat_cache_manager:
            shm_name = self._disc_action_stat_cache_manager.get_shm_name()
            lock = self._disc_action_stat_cache_manager.get_lock()
            cache_manager.update(
                {
                    "action_stat_shm_name": shm_name,
                    "action_stat_lock": lock,
                }
            )

        if cache_manager:
            self.gemm_env_configs.update({"cache_manager": cache_manager})

        self.gemm_env, self.train_gemm_envs, self.test_gemm_envs = (
            self.create_gemm_env()
        )

        # 需要移除cache以保证dump操作能正常执行
        if "cache_manager" in self.gemm_env_configs:
            del self.gemm_env_configs["cache_manager"]

        self.state_shape = self.gemm_env.observation_space.shape[0]
        self.action_shape = self.gemm_env.action_space.shape
        self.logger.info(
            f"GEMM Env created with state shape: {self.state_shape}, action shape: {self.action_shape}"
        )

        # self.train_seed = kwargs.get("train_seed", 0)
        self.train_seed = self.train_config.get(
            "train_seed", kwargs.get("train_seed", 0)
        )
        # seed
        np.random.seed(self.train_seed)
        torch.manual_seed(self.train_seed)

        self.rnd_hidden_dim = self.train_config.get("rnd_hidden_dim", 128)
        self.rnd_intrinsic_weight = self.train_config.get("rnd_intrinsic_weight", 0.01)
        self.rnd_optim_learning_rate = self.train_config.get(
            "rnd_optim_learning_rate", 0.0001
        )
        self.rnd_reset_interval = self.train_config.get("rnd_reset_interval", 2000)
        self.rnd_net, self.rnd_optim = self.create_rnd_module()

        self.actor_net_sigma_min = self.train_config.get("actor_net_sigma_min", 1e-5)
        self.actor_net_sigma_max = self.train_config.get("actor_net_sigma_max", 2.0)
        self.attention_net_output_dim = self.train_config.get(
            "attention_net_output_dim", 64
        )
        # 【新增】控制是否使用attention网络，默认False使用MLP（更稳定）
        self.use_attention_net = self.train_config.get("use_attention_net", False)
        net_logits_clamp_loc = self.train_config.get(
            "net_logits_clamp_loc", "attention"
        )
        if net_logits_clamp_loc == "attention":
            self.attention_net_clamp_enable = True
            self.output_net_clamp_enable = False
        elif net_logits_clamp_loc == "output":
            self.output_net_clamp_enable = True
            self.attention_net_clamp_enable = False
        else:
            raise ValueError("logits_clamp_loc must be 'attention' or 'output'")
        self.net_logits_max = self.train_config.get("net_logits_max", 50.0)
        self.net_logits_min = self.train_config.get("net_logits_min", -50.0)
        self.attention_net_use_layer_norm = self.train_config.get(
            "attention_net_use_layer_norm", True
        )
        self.logger.info(
            f"Using net_logits_clamp_loc: {net_logits_clamp_loc}, net_logits range [{self.net_logits_min}, {self.net_logits_max}], attention_net_use_layer_norm: {self.attention_net_use_layer_norm}"
        )
        self.actor, self.critic = self.create_actor_critic()

        self.optim_learning_rate = self.train_config.get("optim_learning_rate", 0.0001)
        self.optim_weight_decay = self.train_config.get("optim_weight_decay", 0.0)
        self.optim = self.create_optimizer()

        self.actor_policy = self.create_actor_policy()

        self.ppo_gamma = self.train_config.get("ppo_gamma", 0.99)
        self.ppo_max_grad_norm = self.train_config.get("ppo_max_grad_norm", 3.0)
        self.ppo_dual_clip = self.train_config.get("ppo_dual_clip", None)
        self.ppo_gae_lambda = self.train_config.get("ppo_gae_lambda", 0.95)
        self.ppo_ent_coef: float = self.train_config.get(
            "ppo_ent_coef", 0.01
        )  # 设置前为默认值，设置后为实际值
        # 离散entropy系数（固定低值，允许离散动作正常收敛）, 用于
        self.ppo_ent_coef_disc_loop_order = self.train_config.get(
            "ppo_ent_coef_disc_loop_order", self.ppo_ent_coef
        )
        self.ppo_ent_coef_disc_packing_type = self.train_config.get(
            "ppo_ent_coef_disc_packing_type", self.ppo_ent_coef
        )  # 离散entropy系数（固定低值，允许离散动作正常收敛）
        self.ppo_ent_coef_disc_thread_type = self.train_config.get(
            "ppo_ent_coef_disc_thread_type", self.ppo_ent_coef
        )  # 离散entropy系数（固定低值，允许离散动作正常收敛）

        self.ppo_ent_coef_cont = self.train_config.get(
            "ppo_ent_coef_cont", self.ppo_ent_coef
        )  # 连续entropy初始系数（自适应管理）
        self.ppo_ent_coef = (
            self.ppo_ent_coef_disc_loop_order
            + self.ppo_ent_coef_disc_packing_type
            + self.ppo_ent_coef_disc_thread_type
            + self.ppo_ent_coef_cont
        )
        self.ppo_vf_coef = self.train_config.get("ppo_vf_coef", 0.5)
        self.ppo_max_batchsize = self.train_config.get("ppo_max_batchsize", 256)
        self.ppo_return_scaling = self.train_config.get("ppo_return_scaling", True)
        self.ppo_advantage_normalization = self.train_config.get(
            "ppo_advantage_normalization", True
        )
        self.ppo_recompute_advantage = self.train_config.get(
            "ppo_recompute_advantage", False
        )

        # 自适应熵系数控制器初始化（连续动作部分）
        # 控制器类型: "none" (禁用), "v1" (仅监控熵比例), "v2" (监控熵比例+loss比例), "v3" (v2的基础上增加对离散部分thread_type的熵调节)
        self.ppo_use_adaptive_ent_coef_type = self.train_config.get(
            "ppo_use_adaptive_ent_coef_type", "none"
        )

        # 离散动作自适应熵系数控制器配置（thread_type）相关的参数
        self.ppo_adaptive_ent_target_entropy_min_disc_thread_type = (
            self.train_config.get(
                "ppo_adaptive_ent_target_entropy_min_disc_thread_type", 0.3
            )
        )
        self.ppo_adaptive_ent_target_entropy_max_disc_thread_type = (
            self.train_config.get(
                "ppo_adaptive_ent_target_entropy_max_disc_thread_type", 1.0
            )
        )
        self.ppo_adaptive_ent_target_loss_ratio_min_disc_thread_type = (
            self.train_config.get(
                "ppo_adaptive_ent_target_loss_ratio_min_disc_thread_type", 0.2
            )
        )
        self.ppo_adaptive_ent_target_loss_ratio_max_disc_thread_type = (
            self.train_config.get(
                "ppo_adaptive_ent_target_loss_ratio_max_disc_thread_type", 0.5
            )
        )
        self.ppo_adaptive_ent_learning_rate_disc_thread_type = self.train_config.get(
            "ppo_adaptive_ent_learning_rate_disc_thread_type", 0.02
        )
        self.ppo_adaptive_ent_coef_init_disc_thread_type = self.train_config.get(
            "ppo_adaptive_ent_coef_init_disc_thread_type", 0.15
        )
        self.ppo_adaptive_ent_coef_min_disc_thread_type = self.train_config.get(
            "ppo_adaptive_ent_coef_min_disc_thread_type", 0.05
        )
        self.ppo_adaptive_ent_coef_max_disc_thread_type = self.train_config.get(
            "ppo_adaptive_ent_coef_max_disc_thread_type", 0.3
        )
        self.ppo_adaptive_ent_priority_threshold_disc_thread_type = (
            self.train_config.get(
                "ppo_adaptive_ent_priority_threshold_disc_thread_type", 0.1
            )
        )
        # 离散控制器的warmup和decay参数
        self.ppo_adaptive_ent_warmup_epochs_disc_thread_type = self.train_config.get(
            "ppo_adaptive_ent_warmup_epochs_disc_thread_type", 3
        )
        self.ppo_adaptive_ent_decay_rate_disc_thread_type = self.train_config.get(
            "ppo_adaptive_ent_decay_rate_disc_thread_type", 0.9995
        )
        self.ppo_adaptive_ent_decay_start_epoch_disc_thread_type = (
            self.train_config.get(
                "ppo_adaptive_ent_decay_start_epoch_disc_thread_type", 150
            )
        )

        self.ppo_adaptive_ent_target_ratio_cont = self.train_config.get(
            "ppo_adaptive_ent_target_ratio_cont", 0.72
        )
        self.ppo_adaptive_ent_coef_init_cont = self.train_config.get(
            "ppo_adaptive_ent_coef_init_cont", 0.008
        )
        self.ppo_adaptive_ent_coef_min_cont = self.train_config.get(
            "ppo_adaptive_ent_coef_min_cont", 0.0005
        )
        self.ppo_adaptive_ent_coef_max_cont = self.train_config.get(
            "ppo_adaptive_ent_coef_max_cont", 0.3
        )
        self.ppo_adaptive_ent_learning_rate_cont = self.train_config.get(
            "ppo_adaptive_ent_learning_rate_cont", 0.008
        )
        self.ppo_adaptive_ent_decay_rate_cont = self.train_config.get(
            "ppo_adaptive_ent_decay_rate_cont", 0.9992
        )
        self.ppo_adaptive_ent_decay_start_epoch_cont = self.train_config.get(
            "ppo_adaptive_ent_decay_start_epoch_cont", 150
        )
        self.ppo_adaptive_ent_warmup_epochs_cont = self.train_config.get(
            "ppo_adaptive_ent_warmup_epochs_cont", 3
        )
        self.ppo_adaptive_ent_update_interval_cont = self.train_config.get(
            "ppo_adaptive_ent_update_interval_cont", 1
        )

        # V2版本特有参数（支持loss比例监控）
        self.ppo_adaptive_ent_target_entropy_loss_ratio_cont = self.train_config.get(
            "ppo_adaptive_ent_target_entropy_loss_ratio_cont", 0.35
        )
        self.ppo_adaptive_ent_loss_ratio_priority_threshold_cont = (
            self.train_config.get(
                "ppo_adaptive_ent_loss_ratio_priority_threshold_cont", 0.5
            )
        )

        # 性能驱动自适应熵系数控制器专用参数
        self.ppo_adaptive_performance_target_type = self.train_config.get(
            "ppo_adaptive_performance_target_type", "hybrid"
        )
        self.ppo_adaptive_performance_fixed_target = self.train_config.get(
            "ppo_adaptive_performance_fixed_target", 60.0
        )
        self.ppo_adaptive_performance_global_best_weight = self.train_config.get(
            "ppo_adaptive_performance_global_best_weight", 0.8
        )
        self.ppo_adaptive_performance_smoothing_factor = self.train_config.get(
            "ppo_adaptive_performance_smoothing_factor", 0.99
        )
        self.ppo_adaptive_performance_min_improvement = self.train_config.get(
            "ppo_adaptive_performance_min_improvement", 2.0
        )
        self.ppo_adaptive_performance_learning_rate = self.train_config.get(
            "ppo_adaptive_performance_learning_rate", 0.01
        )

        self.ppo_adaptive_performance_coef_range = self.train_config.get(
            "ppo_adaptive_performance_coef_range",
            [
                {
                    "epoch_range": [0, 9999],
                    "loop_order": [0.001, 0.5],
                    "packing_type": [0.001, 0.5],
                    "thread_type": [0.001, 0.5],
                    "cont": [0.001, 0.5],
                }
            ],
        )

        self.ppo_adaptive_performance_entropy_loss_ratio_threshold = (
            self.train_config.get(
                "ppo_adaptive_performance_entropy_loss_ratio_threshold", 0.6
            )
        )

        self.ppo_adaptive_performance_action_sensitivity = self.train_config.get(
            "ppo_adaptive_performance_action_sensitivity",
            [
                {  # 各个分量在熵中的权重
                    "epoch_range": [0, 9999],
                    "loop_order": 0.15,  # 中等敏感度
                    "packing_type": 0.10,  # 较低敏感度
                    "thread_type": 0.25,  # 高敏感度
                    "cont": 0.50,  # 最高敏感度
                }
            ],
        )

        self.ppo_adaptive_performance_adaptive_margin = self.train_config.get(
            "ppo_adaptive_performance_adaptive_margin",
            [
                {
                    "perf_range": [0, 9999],
                    "margin": 2,
                }
            ],
        )

        self.ppo_adaptive_performance_hybrid_best_weight = self.train_config.get(
            "ppo_adaptive_performance_hybrid_best_weight",
            [
                {
                    "epoch_range": [0, 9999],
                    "weight": 0.8,
                }
            ],
        )

        self.ppo_adaptive_performance_warmup_epochs = self.train_config.get(
            "ppo_adaptive_performance_warmup_epochs", 5
        )

        self.ppo_adaptive_performance_reset_interval = self.train_config.get(
            "ppo_adaptive_performance_reset_interval", 500
        )

        # 趋势驱动自适应熵系数控制器（v5）专用参数
        self.ppo_adaptive_trend_window_size = self.train_config.get(
            "ppo_adaptive_trend_window_size", 10
        )
        self.ppo_adaptive_trend_base_learning_rate = self.train_config.get(
            "ppo_adaptive_trend_base_learning_rate", 0.01
        )
        self.ppo_adaptive_trend_warmup_epochs = self.train_config.get(
            "ppo_adaptive_trend_warmup_epochs", 10
        )
        self.ppo_adaptive_trend_reset_interval = self.train_config.get(
            "ppo_adaptive_trend_reset_interval", 500
        )
        self.ppo_adaptive_trend_adaptive_learning_rate = self.train_config.get(
            "ppo_adaptive_trend_adaptive_learning_rate", True
        )
        self.ppo_adaptive_trend_stage_config = self.train_config.get(
            "ppo_adaptive_trend_stage_config", None
        )

        # 熵系数错开调整策略配置
        # 策略选项: "none" (同时调整), "alternate" (交替调整), "priority" (优先级调整), "frequency" (频率调整)
        self.ppo_adaptive_ent_stagger_strategy = self.train_config.get(
            "ppo_adaptive_ent_stagger_strategy", "none"
        )
        # 频率调整策略的参数（仅在strategy="frequency"时使用）
        self.ppo_adaptive_ent_update_freq_cont = self.train_config.get(
            "ppo_adaptive_ent_update_freq_cont", 1
        )
        self.ppo_adaptive_ent_update_freq_disc = self.train_config.get(
            "ppo_adaptive_ent_update_freq_disc", 1
        )

        self.ppo_adaptive_ent_base_loss_min_threshold = self.train_config.get(
            "ppo_adaptive_ent_base_loss_min_threshold", 0.1
        )

        # 新的统一配置参数格式（推荐使用）
        self.gradient_norm_balancer_head_configs = self.train_config.get(
            "ppo_gradient_norm_balancer_head_configs", None
        )
        self.gradient_norm_balancer_ema_decay = self.train_config.get(
            "ppo_gradient_norm_balancer_ema_decay", 0.9
        )
        self.gradient_norm_balancer_update_frequency = self.train_config.get(
            "ppo_gradient_norm_balancer_update_frequency", 1
        )
        self.gradient_norm_balancer_balance_continuous = self.train_config.get(
            "ppo_gradient_norm_balancer_balance_continuous", False
        )

        # SAC 相关参数（用于双机制自适应熵系数调节）
        self.gradient_norm_balancer_sac_lr = self.train_config.get(
            "ppo_gradient_norm_balancer_sac_lr", 0.01
        )
        # Note: grad_safe_ratio_min, grad_safe_ratio_max, safe_tau have been moved to HeadConfig
        # Each action head can now have its own independent safety parameters
        self.gradient_norm_balancer_use_sac_for_discrete = self.train_config.get(
            "ppo_gradient_norm_balancer_use_sac_for_discrete", True
        )

        # Beta-Head 自适应熵系数调节器配置, v7 和 v8 版本的调节器（纯梯度范数比例平衡 / 双系数）
        self.use_adaptive_entropy_balancer = (
            self.ppo_use_adaptive_ent_coef_type.lower() in ["v6", "v7", "v8"]
        )

        # Beta-Head balancer 的统一配置参数格式（v7版本）
        self.beta_head_balancer_head_configs = self.train_config.get(
            "ppo_beta_head_balancer_head_configs", None
        )
        self.beta_head_balancer_ema_decay = self.train_config.get(
            "ppo_beta_head_balancer_ema_decay", 0.99
        )
        self.beta_head_balancer_warmup_steps = self.train_config.get(
            "ppo_beta_head_balancer_warmup_steps", 100
        )
        self.beta_head_balancer_max_change_ratio = self.train_config.get(
            "ppo_beta_head_balancer_max_change_ratio", 0.2
        )
        self.beta_head_balancer_update_frequency = self.train_config.get(
            "ppo_beta_head_balancer_update_frequency", 1
        )
        self.beta_head_balancer_use_safety_correction = self.train_config.get(
            "ppo_beta_head_balancer_use_safety_correction", True
        )
        self.beta_head_balancer_safety_factor = self.train_config.get(
            "ppo_beta_head_balancer_safety_factor", 5.0
        )
        self.beta_head_balancer_safety_tau = self.train_config.get(
            "ppo_beta_head_balancer_safety_tau", 0.5
        )
        self.beta_head_balancer_policy_grad_eps = self.train_config.get(
            "ppo_beta_head_balancer_policy_grad_eps", 1e-10
        )

        # Dual-Coefficient 自适应熵系数调节器配置, v8 版本（双系数：粗调 s_h + 细调 β_h）
        self.dual_coefficient_balancer_head_configs = self.train_config.get(
            "ppo_dual_coefficient_balancer_head_configs", None
        )
        self.dual_coefficient_balancer_ema_decay = self.train_config.get(
            "ppo_dual_coefficient_balancer_ema_decay", 0.99
        )
        self.dual_coefficient_balancer_warmup_steps = self.train_config.get(
            "ppo_dual_coefficient_balancer_warmup_steps", 100
        )
        self.dual_coefficient_balancer_max_change_ratio = self.train_config.get(
            "ppo_dual_coefficient_balancer_max_change_ratio", 0.2
        )
        self.dual_coefficient_balancer_update_frequency = self.train_config.get(
            "ppo_dual_coefficient_balancer_update_frequency", 1
        )
        self.dual_coefficient_balancer_scale_update_frequency = self.train_config.get(
            "ppo_dual_coefficient_balancer_scale_update_frequency", 5
        )
        self.dual_coefficient_balancer_use_safety_correction = self.train_config.get(
            "ppo_dual_coefficient_balancer_use_safety_correction", True
        )
        self.dual_coefficient_balancer_policy_grad_eps = self.train_config.get(
            "ppo_dual_coefficient_balancer_policy_grad_eps", 1e-10
        )
        self.dual_coefficient_balancer_scale_align_mode = self.train_config.get(
            "ppo_dual_coefficient_balancer_scale_align_mode", "magnitude"
        )

        self.algorithm = self.create_algorithm().to(self.device)
        if isinstance(checkpoint, dict):
            self.algorithm.load_state_dict(checkpoint["policy"])
            self.algorithm.load_rnd_module(checkpoint["rnd_module"])
            self.logger.info("algorithm restored from checkpoint successfully.")

        # 创建自适应熵控制器（连续动作部分）
        self.adaptive_ent_controller_cont = (
            self.create_adaptive_entropy_controller_cont()
        )

        # 创建自适应熵控制器（离散动作thread_type部分）
        self.adaptive_ent_controller_disc = (
            self.create_adaptive_entropy_controller_disc()
        )

        # 创建性能驱动自适应熵系数控制器
        self.adaptive_ent_controller_performance = (
            self.create_adaptive_entropy_controller_performance()
        )

        # 创建趋势驱动自适应熵系数控制器（v5版本）
        self.adaptive_ent_controller_trend = (
            self.create_adaptive_entropy_controller_trend()
        )

        # 创建梯度范数平衡自适应熵系数调节器
        self.adaptive_entropy_balancer_gradient_norm = (
            self.create_adaptive_entropy_balancer_gradient_norm()
        )

        # 创建 Beta-Head 自适应熵系数调节器（v7版本）
        self.adaptive_entropy_beta_head_balancer = (
            self.create_adaptive_entropy_beta_head_balancer()
        )

        # 创建 Dual-Coefficient 自适应熵系数调节器（v8版本）
        self.adaptive_entropy_dual_coefficient_balancer = (
            self.create_adaptive_dual_coefficient_balancer()
        )

        self.buffer_total_size = self.train_config.get("buffer_total_size", 40960)
        self.buffer = self.create_buffer()

        self.training_collector = self.create_training_collector()
        self.test_collector = self.create_test_collector()

        self.logger_save_interval = self.train_config.get("logger_save_interval", 4000)
        self.logger_update_interval = self.train_config.get(
            "logger_update_interval", 100
        )
        self.logger_dir: str = self.train_config.get("logger_dir", "./logs")
        self.train_logger = self.create_train_logger()
        self.train_gradients_logger = self.create_train_gradients_logger()

        self.train_max_epochs = self.train_config.get("train_max_epochs", 1000)
        self.train_epoch_num_steps = self.train_config.get(
            "train_epoch_num_steps", 1000
        )
        self.train_update_step_num_repetitions = self.train_config.get(
            "train_update_step_num_repetitions", 4
        )
        self.train_test_step_num_episodes = self.train_config.get(
            "train_test_step_num_episodes", 1
        )
        self.train_batch_size = self.train_config.get("train_batch_size", 64)
        if "train_collection_step_num_env_steps" in self.train_config:
            self.train_collection_step_num_env_steps = self.train_config[
                "train_collection_step_num_env_steps"
            ]
            collection_steps_source = "train_collection_step_num_env_steps"
        elif "collection_step_num_env_steps" in self.train_config:
            self.train_collection_step_num_env_steps = self.train_config[
                "collection_step_num_env_steps"
            ]
            collection_steps_source = "collection_step_num_env_steps"
        else:
            self.train_collection_step_num_env_steps = 2048
            collection_steps_source = "default"

        self.ablation_id = self.train_config.get("ablation_id", "base")
        dynamic_sampling_config = self.gemm_env_configs.get(
            "dynamic_sampling_config", {}
        )
        self.logger.info(
            "Training configuration resolved: "
            f"ablation_id={self.ablation_id}, "
            f"collection_step_num_env_steps={self.train_collection_step_num_env_steps} "
            f"(source={collection_steps_source}), "
            f"sampling_mode={dynamic_sampling_config.get('sampling_mode', 'adaptive_repeat')}, "
            f"reward_mode={self.gemm_env_configs.get('reward_mode', 'multi_level')}, "
            f"entropy_controller={self.ppo_use_adaptive_ent_coef_type}"
        )

    def __del__(self):
        """
        析构函数
        """
        if (
            hasattr(self, "_shape_perf_cache_manager")
            and self._shape_perf_cache_manager is not None
        ):
            log_str = self._shape_perf_cache_manager.get_global_stat_str()
            self.logger.info(log_str)
            self._shape_perf_cache_manager.release()  # 释放共享内存资源

        if (
            hasattr(self, "_disc_action_stat_cache_manager")
            and self._disc_action_stat_cache_manager is not None
        ):
            log_str = self._disc_action_stat_cache_manager.get_global_stat_str(True)
            self.logger.info(log_str)
            self._disc_action_stat_cache_manager.release()

    def create_cache_manager(
        self,
        shape_perf_ck_data: Dict[
            Tuple[int, int, int], Tuple[float, float, float, float, float]
        ] = None,
        action_stat_ck_data: Dict[Tuple[int, int, Tuple[int, int, int]], int] = None,
    ) -> Tuple[ShapePerformanceCacheManager, DiscreteActionCacheManager]:
        """
        创建基于共享内存的管理器
        """
        if self.gemm_env_configs.get("gemm_config_type", None) != "gemm_shape_config":
            raise ValueError("gemm_config_type must be 'gemm_shape_config'!")

        performance_ema_alpha = self.gemm_env_configs.get("performance_ema_alpha", 0.3)
        if shape_perf_ck_data is None:
            shape_list = self.gemm_env_configs.get("gemm_shape_config", {}).get(
                "shape_list", []
            )
            shape_list = [tuple(shape) for shape in shape_list]
            shape_cache_manager = ShapePerformanceCacheManager(
                ema_alpha=performance_ema_alpha, shape_list=shape_list, create=True
            )
        else:
            shape_cache_manager = ShapePerformanceCacheManager(
                ema_alpha=performance_ema_alpha,
                checkpoint_data=shape_perf_ck_data,
                create=True,
            )
            self.logger.info("shape cache manager is created with checkpoint data!!")

        if action_stat_ck_data is None:
            loop_orders = self.gemm_env_configs.get("loop_orders", [])
            packing_types = self.gemm_env_configs.get("packing_types", [])
            thread_types = self.gemm_env_configs.get("threads", [])
            thread_types = [tuple(thread) for thread in thread_types]
            action_stat_manager = DiscreteActionCacheManager(
                loop_orders=loop_orders,
                packing_types=packing_types,
                thread_types=thread_types,
                create=True,
            )
        else:
            action_stat_manager = DiscreteActionCacheManager(
                checkpoint_data=action_stat_ck_data, create=True
            )
            self.logger.info(
                "discrete action stat manager is created with checkpoint data!!"
            )

        return shape_cache_manager, action_stat_manager

    def create_gemm_env(self) -> Tuple[Env, BaseVectorEnv, BaseVectorEnv]:
        """
        创建GEMM环境
        """
        return make_gemm_env(
            task=self.gemm_env_task,
            seed=self.gemm_env_seed,
            num_training_envs=self.gemm_env_train_num,
            num_test_envs=self.gemm_env_test_num,
            venv_type=self.gemm_env_venv_type,
            **self.gemm_env_configs,
        )

    def create_rnd_module(self) -> Tuple[RNDModule, torch.optim.Adam]:
        """
        创建随机网络蒸馏模块
        """
        rnd_net = RNDModule(
            obs_dim=self.state_shape,
            loop_order_size=self.gemm_env.get_loop_order_size(),
            packing_type_size=self.gemm_env.get_packing_type_size(),
            thread_type_size=self.gemm_env.get_thread_type_size(),
            continuous_action_dim=self.gemm_env.get_continuous_action_dim(),
            hidden_dim=self.rnd_hidden_dim,
            device=self.device,
            reset_interval=self.rnd_reset_interval,
        )
        optim_rnd = torch.optim.Adam(
            rnd_net.predictor.parameters(), lr=self.rnd_optim_learning_rate
        )
        return rnd_net, optim_rnd

    def create_actor_critic(self) -> Tuple[GemmHyperActor, GemmContinuousCritic]:
        """
        创建actor和critic网络 - 支持MLP和Attention两种结构
        """
        # 从配置中读取是否使用attention（默认False，使用MLP更稳定）
        use_attention = self.train_config.get("use_attention", False)

        actor = GemmHyperActor(
            obs_dim=self.state_shape,
            loop_order_size=self.gemm_env.get_loop_order_size(),
            packing_type_size=self.gemm_env.get_packing_type_size(),
            thread_type_size=self.gemm_env.get_thread_type_size(),
            continuous_action_dim=self.gemm_env.get_continuous_action_dim(),
            sigma_min=self.actor_net_sigma_min,
            sigma_max=self.actor_net_sigma_max,
            attention_net_output_dim=self.attention_net_output_dim,
            use_attention=use_attention,  # 【新增】控制网络结构
            attention_net_clamp_enable=self.attention_net_clamp_enable,
            output_net_clamp_enable=self.output_net_clamp_enable,
            net_logits_min=self.net_logits_min,
            net_logits_max=self.net_logits_max,
            attention_net_use_layer_norm=self.attention_net_use_layer_norm,
        )
        critic = GemmContinuousCritic(
            self.state_shape,
            attention_net_output_dim=self.attention_net_output_dim,
            use_attention=use_attention,  # 【新增】
            attention_net_clamp_enable=self.attention_net_clamp_enable,
            output_net_clamp_enable=self.output_net_clamp_enable,
            net_logits_min=self.net_logits_min,
            net_logits_max=self.net_logits_max,
            attention_net_use_layer_norm=self.attention_net_use_layer_norm,
        )
        return actor, critic

    def create_optimizer(self) -> GemmAdamOptimizer:
        """
        创建优化器
        """
        return GemmAdamOptimizer(
            base_lr=self.optim_learning_rate, weight_decay=self.optim_weight_decay
        )

    def create_actor_policy(self) -> ProbabilisticActorPolicy:
        """
        创建actor策略
        """
        return ProbabilisticActorPolicy(
            actor=self.actor,
            dist_fn=partial(
                get_gemm_action_distribution,
                loop_order_len=self.gemm_env.get_loop_order_size(),
                thread_type_len=self.gemm_env.get_thread_type_size(),
                packing_type_len=self.gemm_env.get_packing_type_size(),
                other_continues_len=self.gemm_env.get_continuous_action_dim(),
            ),
            action_space=self.gemm_env.action_space,
            observation_space=self.gemm_env.observation_space,
            action_scaling=False,
            action_bound_method=None,
        )

    def create_algorithm(self) -> GemmPPOPolicy:
        """
        创建PPO算法
        """
        return GemmPPOPolicy(
            policy=self.actor_policy,
            critic=self.critic,
            optim=self.optim,
            dual_clip=self.ppo_dual_clip,
            advantage_normalization=self.ppo_advantage_normalization,
            recompute_advantage=self.ppo_recompute_advantage,
            ent_coef=self.ppo_ent_coef,
            vf_coef=self.ppo_vf_coef,
            max_grad_norm=self.ppo_max_grad_norm,
            gae_lambda=self.ppo_gae_lambda,
            max_batchsize=self.ppo_max_batchsize,
            gamma=self.ppo_gamma,
            return_scaling=self.ppo_return_scaling,
            rnd_net=self.rnd_net,
            rnd_optim=self.rnd_optim,
            rnd_intrinsic_weight=self.rnd_intrinsic_weight,
            ent_coef_disc_loop_order=self.ppo_ent_coef_disc_loop_order,
            ent_coef_disc_packing_type=self.ppo_ent_coef_disc_packing_type,
            ent_coef_disc_thread_type=self.ppo_ent_coef_disc_thread_type,
            ent_coef_cont=self.ppo_ent_coef_cont,
            use_adaptive_entropy_balancer=self.use_adaptive_entropy_balancer,
        )

    def create_adaptive_entropy_controller_cont(self):
        """
        创建熵自动化调节器（支持V1和V2版本）
        """
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        # 检查是否使用V2版本
        if controller_type in ["none", "v4", "v5", "v6", "v7", "v8"]:
            self.logger.info(
                "Adaptive entropy controller for continuous actions disabled."
            )
            return None
        elif controller_type in ["v2", "v3"]:
            # 创建V2版本控制器（支持loss比例监控）
            adaptive_ent_controller_cont = AdaptiveEntropyControllerContV2(
                target_entropy_ratio=self.ppo_adaptive_ent_target_ratio_cont,
                target_entropy_loss_ratio=self.ppo_adaptive_ent_target_entropy_loss_ratio_cont,
                learning_rate=self.ppo_adaptive_ent_learning_rate_cont,
                ent_coef_init=self.ppo_adaptive_ent_coef_init_cont,
                ent_coef_min=self.ppo_adaptive_ent_coef_min_cont,
                ent_coef_max=self.ppo_adaptive_ent_coef_max_cont,
                decay_rate=self.ppo_adaptive_ent_decay_rate_cont,
                decay_start_epoch=self.ppo_adaptive_ent_decay_start_epoch_cont,
                warmup_epochs=self.ppo_adaptive_ent_warmup_epochs_cont,
                update_interval=self.ppo_adaptive_ent_update_interval_cont,
                loss_ratio_priority_threshold=self.ppo_adaptive_ent_loss_ratio_priority_threshold_cont,
                base_loss_min_threshold=self.ppo_adaptive_ent_base_loss_min_threshold,
                logger=self.logger,
            )
            self.logger.info(
                f"Adaptive entropy controller V2 for continuous actions enabled:\n"
                f"  target_entropy_ratio={self.ppo_adaptive_ent_target_ratio_cont}\n"
                f"  target_entropy_loss_ratio={self.ppo_adaptive_ent_target_entropy_loss_ratio_cont}\n"
                f"  ent_coef_range=[{self.ppo_adaptive_ent_coef_min_cont}, {self.ppo_adaptive_ent_coef_max_cont}]\n"
                f"  base_loss_min_threshold={self.ppo_adaptive_ent_base_loss_min_threshold}"
            )
        elif controller_type == "v1":
            # 创建V1版本控制器（仅监控熵比例）
            adaptive_ent_controller_cont = AdaptiveEntropyControllerCont(
                target_ratio=self.ppo_adaptive_ent_target_ratio_cont,
                learning_rate=self.ppo_adaptive_ent_learning_rate_cont,
                ent_coef_init=self.ppo_adaptive_ent_coef_init_cont,
                ent_coef_min=self.ppo_adaptive_ent_coef_min_cont,
                ent_coef_max=self.ppo_adaptive_ent_coef_max_cont,
                decay_rate=self.ppo_adaptive_ent_decay_rate_cont,
                decay_start_epoch=self.ppo_adaptive_ent_decay_start_epoch_cont,
                warmup_epochs=self.ppo_adaptive_ent_warmup_epochs_cont,
                update_interval=self.ppo_adaptive_ent_update_interval_cont,
                logger=self.logger,
            )
            self.logger.info(
                f"Adaptive entropy controller V1 for continuous actions enabled with target_ratio={self.ppo_adaptive_ent_target_ratio_cont}, "
                f"ent_coef_range=[{self.ppo_adaptive_ent_coef_min_cont}, {self.ppo_adaptive_ent_coef_max_cont}]"
            )
        else:
            raise ValueError(
                f"Invalid ppo_use_adaptive_ent_coef_cont_type: {controller_type}. "
                f"Must be one of: 'none', 'v1', 'v2'"
            )

        return adaptive_ent_controller_cont

    def create_adaptive_entropy_controller_disc(self):
        """
        创建离散动作thread_type的自适应熵系数控制器
        """
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        if controller_type not in ["v3"]:
            self.logger.info(
                "Adaptive entropy controller for discrete thread_type disabled."
            )
            return None

        # 创建离散动作自适应熵系数控制器
        adaptive_ent_controller_disc_thread_type = AdaptiveEntropyControllerDisc(
            target_entropy_min=self.ppo_adaptive_ent_target_entropy_min_disc_thread_type,
            target_entropy_max=self.ppo_adaptive_ent_target_entropy_max_disc_thread_type,
            target_entropy_loss_ratio_min=self.ppo_adaptive_ent_target_loss_ratio_min_disc_thread_type,
            target_entropy_loss_ratio_max=self.ppo_adaptive_ent_target_loss_ratio_max_disc_thread_type,
            learning_rate=self.ppo_adaptive_ent_learning_rate_disc_thread_type,
            ent_coef_init=self.ppo_adaptive_ent_coef_init_disc_thread_type,
            ent_coef_min=self.ppo_adaptive_ent_coef_min_disc_thread_type,
            ent_coef_max=self.ppo_adaptive_ent_coef_max_disc_thread_type,
            max_entropy=np.log(self.gemm_env.get_thread_type_size()),
            priority_threshold=self.ppo_adaptive_ent_priority_threshold_disc_thread_type,
            warmup_epochs=self.ppo_adaptive_ent_warmup_epochs_disc_thread_type,
            decay_rate=self.ppo_adaptive_ent_decay_rate_disc_thread_type,
            decay_start_epoch=self.ppo_adaptive_ent_decay_start_epoch_disc_thread_type,
            base_loss_min_threshold=self.ppo_adaptive_ent_base_loss_min_threshold,
            logger=self.logger,
        )

        self.logger.info(
            f"Adaptive entropy controller for discrete thread_type enabled:\n"
            f"  target_entropy_range=[{self.ppo_adaptive_ent_target_entropy_min_disc_thread_type}, {self.ppo_adaptive_ent_target_entropy_max_disc_thread_type}]\n"
            f"  target_entropy_loss_ratio_range=[{self.ppo_adaptive_ent_target_loss_ratio_min_disc_thread_type}, {self.ppo_adaptive_ent_target_loss_ratio_max_disc_thread_type}]\n"
            f"  ent_coef_init={self.ppo_adaptive_ent_coef_init_disc_thread_type}, range=[{self.ppo_adaptive_ent_coef_min_disc_thread_type}, {self.ppo_adaptive_ent_coef_max_disc_thread_type}]\n"
            f"  warmup_epochs={self.ppo_adaptive_ent_warmup_epochs_disc_thread_type}, decay_rate={self.ppo_adaptive_ent_decay_rate_disc_thread_type}, decay_start_epoch={self.ppo_adaptive_ent_decay_start_epoch_disc_thread_type}\n"
            f"  base_loss_min_threshold={self.ppo_adaptive_ent_base_loss_min_threshold}"
        )

        return adaptive_ent_controller_disc_thread_type

    def create_adaptive_entropy_controller_performance(self):
        """
        创建性能驱动自适应熵系数控制器
        """
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        if controller_type != "v4":
            self.logger.info("Performance-driven adaptive entropy controller disabled.")
            return None

        controller = AdaptiveEntropyControllerByPerformance(
            target_type=self.ppo_adaptive_performance_target_type,
            fixed_target=self.ppo_adaptive_performance_fixed_target,
            global_best_weight=self.ppo_adaptive_performance_global_best_weight,
            smoothing_factor=self.ppo_adaptive_performance_smoothing_factor,
            min_improvement=self.ppo_adaptive_performance_min_improvement,
            learning_rate=self.ppo_adaptive_performance_learning_rate,
            coef_range=self.ppo_adaptive_performance_coef_range,
            entropy_loss_ratio_thresholds=self.ppo_adaptive_performance_entropy_loss_ratio_threshold,
            warmup_epochs=self.ppo_adaptive_performance_warmup_epochs,
            action_sensitivity=self.ppo_adaptive_performance_action_sensitivity,
            init_coefs={
                "loop_order": self.ppo_ent_coef_disc_loop_order,  # 默认值，后续会更新
                "packing_type": self.ppo_ent_coef_disc_packing_type,
                "thread_type": self.ppo_ent_coef_disc_thread_type,
                "cont": self.ppo_ent_coef_cont,
            },
            perf_adaptive_margin=self.ppo_adaptive_performance_adaptive_margin,
            hybrid_best_weight=self.ppo_adaptive_performance_hybrid_best_weight,
            reset_interval=self.ppo_adaptive_performance_reset_interval,
            logger=self.logger,
        )

        self.logger.info(
            f"Performance-driven adaptive entropy controller enabled:\n"
            f"  target_type={self.ppo_adaptive_performance_target_type}\n"
            f"  fixed_target={self.ppo_adaptive_performance_fixed_target:.1f} GFlops\n"
            f"  global_best_weight={self.ppo_adaptive_performance_global_best_weight:.2f}\n"
            f"  coef_range={self.ppo_adaptive_performance_coef_range}\n"
            f"  action_sensitivity={self.ppo_adaptive_performance_action_sensitivity}\n"
            f"  perf_adaptive_margin={self.ppo_adaptive_performance_adaptive_margin}\n"
            f"  hybrid_best_weight={self.ppo_adaptive_performance_hybrid_best_weight}"
        )

        return controller

    def create_adaptive_entropy_controller_trend(self):
        """
        创建趋势驱动自适应熵系数控制器（v5版本）
        """
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        if controller_type != "v5":
            self.logger.info("Trend-driven adaptive entropy controller disabled.")
            return None

        controller = AdaptiveEntropyControllerByTrend(
            trend_window_size=self.ppo_adaptive_trend_window_size,
            base_learning_rate=self.ppo_adaptive_trend_base_learning_rate,
            warmup_epochs=self.ppo_adaptive_trend_warmup_epochs,
            reset_interval=self.ppo_adaptive_trend_reset_interval,
            init_coefs={
                "loop_order": self.ppo_ent_coef_disc_loop_order,  # 默认值，后续会更新
                "packing_type": self.ppo_ent_coef_disc_packing_type,
                "thread_type": self.ppo_ent_coef_disc_thread_type,
                "cont": self.ppo_ent_coef_cont,
            },
            stage_config=self.ppo_adaptive_trend_stage_config,
            adaptive_learning_rate=self.ppo_adaptive_trend_adaptive_learning_rate,
            logger=self.logger,
        )

        self.logger.info(
            f"Trend-driven adaptive entropy controller enabled:\n"
            f"  trend_window_size={self.ppo_adaptive_trend_window_size}\n"
            f"  base_learning_rate={self.ppo_adaptive_trend_base_learning_rate}\n"
            f"  warmup_epochs={self.ppo_adaptive_trend_warmup_epochs}\n"
            f"  reset_interval={self.ppo_adaptive_trend_reset_interval}\n"
            f"  adaptive_learning_rate={self.ppo_adaptive_trend_adaptive_learning_rate}\n"
            f"  stage_config={self.ppo_adaptive_trend_stage_config}"
        )

        return controller

    def create_adaptive_entropy_balancer_gradient_norm(self):
        """
        创建梯度范数平衡自适应熵系数调节器

        支持两种配置方式：
        1. 新格式：使用 gradient_norm_balancer_head_configs 参数（推荐）

        如果提供了新的 head_configs 参数，直接使用；
        否则从旧参数自动构建配置。
        """
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        if controller_type != "v6":
            self.logger.info("Gradient norm entropy balancer disabled.")
            return None

        # 使用新的统一配置格式
        # Note: grad_safe_ratio_min, grad_safe_ratio_max, safe_tau are now per-head parameters in HeadConfig
        balancer = AdaptiveEntropyBalancer(
            head_configs=self.gradient_norm_balancer_head_configs,
            ema_decay=self.gradient_norm_balancer_ema_decay,
            epsilon=1e-8,
            update_frequency=self.gradient_norm_balancer_update_frequency,
            balance_continuous=self.gradient_norm_balancer_balance_continuous,
            sac_lr=self.gradient_norm_balancer_sac_lr,
            use_sac_for_discrete=self.gradient_norm_balancer_use_sac_for_discrete,
        )

        self.logger.info(
            "Gradient norm entropy balancer enabled with new head_configs format:\n"
            f"  head_configs={self.gradient_norm_balancer_head_configs}\n"
            f"  ema_decay={self.gradient_norm_balancer_ema_decay}\n"
            f"  update_frequency={self.gradient_norm_balancer_update_frequency}\n"
            f"  balance_continuous={self.gradient_norm_balancer_balance_continuous}\n"
            f"  SAC parameters: sac_lr={self.gradient_norm_balancer_sac_lr}, "
            f"use_sac_for_discrete={self.gradient_norm_balancer_use_sac_for_discrete}"
        )

        return balancer

    def create_adaptive_entropy_beta_head_balancer(self):
        """
        创建 Beta-Head 自适应熵系数调节器（v7版本）

        使用纯梯度范数比例平衡算法，统一处理离散和连续头。
        完全放弃 SAC 风格的目标熵调节。
        """
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        if controller_type != "v7":
            self.logger.info("Beta-Head entropy balancer disabled.")
            return None

        balancer = AdaptiveEntropyBetaHeadBalancer(
            head_configs=self.beta_head_balancer_head_configs,
            ema_decay=self.beta_head_balancer_ema_decay,
            warmup_steps=self.beta_head_balancer_warmup_steps,
            max_change_ratio=self.beta_head_balancer_max_change_ratio,
            update_frequency=self.beta_head_balancer_update_frequency,
            use_safety_correction=self.beta_head_balancer_use_safety_correction,
            safety_factor=self.beta_head_balancer_safety_factor,
            safety_tau=self.beta_head_balancer_safety_tau,
            policy_grad_eps=self.beta_head_balancer_policy_grad_eps,
            epsilon=1e-8,
        )

        self.logger.info(
            "Beta-Head entropy balancer enabled:\n"
            f"  head_configs={self.beta_head_balancer_head_configs}\n"
            f"  ema_decay={self.beta_head_balancer_ema_decay}\n"
            f"  warmup_steps={self.beta_head_balancer_warmup_steps}\n"
            f"  max_change_ratio={self.beta_head_balancer_max_change_ratio}\n"
            f"  update_frequency={self.beta_head_balancer_update_frequency}\n"
            f"  use_safety_correction={self.beta_head_balancer_use_safety_correction}\n"
            f"  safety_factor={self.beta_head_balancer_safety_factor}\n"
            f"  safety_tau={self.beta_head_balancer_safety_tau}\n"
            f"  policy_grad_eps={self.beta_head_balancer_policy_grad_eps}"
        )

        return balancer

    def create_adaptive_dual_coefficient_balancer(self):
        """
        创建 Dual-Coefficient 自适应熵系数调节器（v8版本）

        使用双系数调节算法：
        - β_h: 细调系数，快速响应梯度范数比例变化
        - s_h: 粗调系数，缓慢调整以稳定训练
        """
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        if controller_type != "v8":
            self.logger.info("Dual-Coefficient entropy balancer disabled.")
            return None

        balancer = AdaptiveDualCoefficientBalancer(
            head_configs=self.dual_coefficient_balancer_head_configs,
            ema_decay=self.dual_coefficient_balancer_ema_decay,
            warmup_steps=self.dual_coefficient_balancer_warmup_steps,
            max_change_ratio=self.dual_coefficient_balancer_max_change_ratio,
            update_frequency=self.dual_coefficient_balancer_update_frequency,
            scale_update_frequency=self.dual_coefficient_balancer_scale_update_frequency,
            use_safety_correction=self.dual_coefficient_balancer_use_safety_correction,
            policy_grad_eps=self.dual_coefficient_balancer_policy_grad_eps,
            scale_align_mode=self.dual_coefficient_balancer_scale_align_mode,
            epsilon=1e-8,
        )

        self.logger.info(
            "Dual-Coefficient entropy balancer enabled:\n"
            f"  head_configs={self.dual_coefficient_balancer_head_configs}\n"
            f"  ema_decay={self.dual_coefficient_balancer_ema_decay}\n"
            f"  warmup_steps={self.dual_coefficient_balancer_warmup_steps}\n"
            f"  max_change_ratio={self.dual_coefficient_balancer_max_change_ratio}\n"
            f"  update_frequency={self.dual_coefficient_balancer_update_frequency}\n"
            f"  scale_update_frequency={self.dual_coefficient_balancer_scale_update_frequency}\n"
            f"  use_safety_correction={self.dual_coefficient_balancer_use_safety_correction}\n"
            f"  policy_grad_eps={self.dual_coefficient_balancer_policy_grad_eps}\n"
            f"  scale_align_mode={self.dual_coefficient_balancer_scale_align_mode}"
        )

        return balancer

    def create_buffer(self) -> VectorReplayBuffer:
        """
        创建经验回放缓冲区
        """
        return VectorReplayBuffer(
            total_size=self.buffer_total_size, buffer_num=self.gemm_env_train_num
        )

    def create_training_collector(self) -> Collector:
        """
        创建训练采集器
        """
        return Collector[CollectStats](
            policy=self.algorithm,
            env=self.train_gemm_envs,
            buffer=self.buffer,
            exploration_noise=True,
        )

    def create_test_collector(self) -> Collector:
        """
        创建测试采集器
        """
        return Collector[GEMMTestCollectStats](
            policy=self.algorithm,
            env=self.test_gemm_envs,
            exploration_noise=False,
            collect_stats_class=GEMMTestCollectStats,
        )

    def create_train_logger(self) -> TLogger:
        """
        创建日志记录器
        """

        logger_factory = LoggerFactoryDefault(
            logger_type="tensorboard",
            save_interval=self.logger_save_interval,
        )

        now = datetime.now().strftime("%y%m%d_%H%M%S")
        tensorboard_fold_name = f"{self.trainer_name}_seed{self.train_seed}_{now}"
        log_path = os.path.join(self.logger_dir, tensorboard_fold_name)

        logger = logger_factory.create_logger(
            log_dir=log_path,
            experiment_name=self.trainer_name,
            run_id=None,
        )
        logger.update_interval = (
            self.logger_update_interval
        )  # 没有提供接口，只能直接设置
        return logger

    def create_train_gradients_logger(self) -> GradientsLogger | None:
        """
        创建训练过程中的梯度记录器
        """
        if self.train_logger is not None:
            return GradientsLogger(self.train_logger)
        else:
            return None

    def save_best_fn(self, policy: Algorithm):
        """
        保存最好的模型
        """
        # 列出目录中best policy的文件，按照修改时间排序，如果超过保存的数量限制，就删除最旧的
        file_list = filter(
            lambda x: x.startswith("best_policy") and x.endswith(".pth"),
            os.listdir(self.model_save_dir),
        )

        sorted_files = sorted(
            file_list,
            key=lambda x: os.path.getmtime(os.path.join(self.model_save_dir, x)),
            reverse=True,
        )

        if len(sorted_files) >= self.best_policy_max_num:
            for old_file in sorted_files[self.best_policy_max_num - 1 :]:
                old_file_path = os.path.join(self.model_save_dir, old_file)
                os.remove(old_file_path)
                self.logger.info(
                    f"Old best policy {old_file_path} removed to maintain max best policy limit."
                )

        self.save_policy_configs(policy, "best_policy")

    def load_checkpoint(
        self, model_dir: str, check_point_name: str
    ) -> bool | Dict[str, Any]:
        """
        检查模型的检查点是否存在
        """
        if (not os.path.exists(model_dir)) or (not os.path.isdir(model_dir)):
            self.logger.error(f"Model directory {model_dir} does not exist.")
            return False

        if check_point_name:
            checkpoint_path = os.path.join(model_dir, check_point_name)
            if not os.path.exists(checkpoint_path):
                self.logger.error(f"Checkpoint {checkpoint_path} does not exist.")
                return False
            return self.load_model_env(checkpoint_path)
        else:
            file_list = filter(lambda x: x.endswith(".pth"), os.listdir(model_dir))
            sorted_files = sorted(
                file_list,
                key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
                reverse=True,
            )
            if not sorted_files:
                self.logger.warning(f"No checkpoint files found in {model_dir}.")
                return False
            checkpoint_path = os.path.join(model_dir, sorted_files[0])

            return self.load_model_env(checkpoint_path)

    def load_model_env(self, model_path: str) -> Dict[str, Any]:
        """
        加载模型
        """
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        self.logger.info(f"Model from {model_path} loaded successfully")
        return checkpoint

    def save_policy_configs(
        self, policy: Algorithm, name_prefix: str, add_timestamp: bool = True
    ) -> str:
        """
        保存策略
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if add_timestamp:
            model_save_path = os.path.join(
                self.model_save_dir, f"{name_prefix}_{timestamp}_model.pth"
            )
        else:
            model_save_path = os.path.join(
                self.model_save_dir, f"{name_prefix}_model.pth"
            )
        torch.save(
            {
                "policy": policy.state_dict(),
                "train_configs": self.train_config,
                "gemm_env_configs": self.gemm_env_configs,
                "rnd_module": self.algorithm.dump_rnd_module(),
                "is_debug": self._is_debug,
                "is_train": self._is_train,
                "cache_manager": {
                    "shape_performance": self._shape_perf_cache_manager.dumps()
                    if self._shape_perf_cache_manager is not None
                    else None,
                    "action_stat": self._disc_action_stat_cache_manager.dumps()
                    if self._disc_action_stat_cache_manager is not None
                    else None,
                },
            },
            model_save_path,
        )
        self.logger.info(f"\nBest model saved at {model_save_path}")
        return model_save_path

    def stop_fn(self, score: float) -> bool:
        """
        判断什么时候停止
        """
        # 判断self.model_save_dir目录下是否存在finish_training.flag文件，如果存在就停止训练
        finish_file_path = os.path.join(self.model_save_dir, "finish_training.flag")
        if os.path.exists(finish_file_path):
            self.logger.info(
                f"Finish file {finish_file_path} detected, stopping training with score {score}."
            )
            return True
        return False

    def save_checkpoint_by_interval_fn(self, epoch: int) -> str | None:
        """
        保存checkpoint的函数，返回checkpoint的名字
        """
        # 1. 根据epoch数判断是否需要保存checkpoint
        if (self.checkpoint_save_interval > 0) and (
            epoch % self.checkpoint_save_interval != 0
        ):
            return None

        # 检查是否已保存过当前epoch的checkpoint，如果已保存过就不重复保存
        checkpoint_name = f"checkpoint_epoch{epoch}_model.pth"
        checkpoint_path = os.path.join(self.model_save_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            self.logger.info(
                f"Checkpoint for epoch {epoch} already exists at {checkpoint_path}, skipping save."
            )
            return None

        # 列出目录中checkpoint的文件，按照修改时间排序，如果超过保存的数量限制，就删除最旧的
        file_list = filter(
            lambda x: x.startswith("checkpoint_epoch") and x.endswith(".pth"),
            os.listdir(self.model_save_dir),
        )
        sorted_files = sorted(
            file_list,
            key=lambda x: os.path.getmtime(os.path.join(self.model_save_dir, x)),
            reverse=True,
        )
        if len(sorted_files) >= self.checkpoint_max_num:
            for old_file in sorted_files[self.checkpoint_max_num - 1 :]:
                old_file_path = os.path.join(self.model_save_dir, old_file)
                os.remove(old_file_path)
                self.logger.info(
                    f"Old checkpoint {old_file_path} removed to maintain max checkpoint limit."
                )

        # 调用保存函数
        model_save_path = self.save_policy_configs(
            self.algorithm, f"checkpoint_epoch{epoch}", add_timestamp=False
        )
        self.logger.info(f"Checkpoint for epoch {epoch} saved at {model_save_path}")
        return model_save_path

    def manual_save_checkpoint_fn(self, epoch: int) -> str | None:
        """
        手动触发保存checkpoint
        """
        save_ck_flag_file_path = os.path.join(self.model_save_dir, "save_ck.flag")
        if os.path.exists(save_ck_flag_file_path):
            Path(save_ck_flag_file_path).unlink(missing_ok=True)
            model_save_path = self.save_policy_configs(
                self.algorithm, f"manual_epoch{epoch}", add_timestamp=True
            )
            self.logger.info(
                f"manual save checkpoint file {save_ck_flag_file_path} detected. checkpoint is saved at {model_save_path}"
            )
            return model_save_path
        else:
            return None

    def _logging_agent_global_performance(self, epoch: int, env_step: int):
        """
        记录全局的统计信息到tensorboard上
        """
        global_stat_row = self._shape_perf_cache_manager.get_global_stat_row()
        if global_stat_row is not None:
            self.logger.info(
                f"[Epoch {epoch}] Training step agent global performance logging triggered at env step {env_step}."
            )
            global_mean = (global_stat_row[4] / global_stat_row[0]).item()
            global_min = global_stat_row[1].item()
            global_max = global_stat_row[2].item()
            global_mean_best = global_stat_row[3].item()
            global_total_shapes = global_stat_row[0].item()
            env_perf_log_data = self.train_logger.prepare_dict_for_logging(
                {
                    "perf/total_shapes": global_total_shapes,
                    "perf/global_min": global_min,
                    "perf/global_mean_perf": global_mean,
                    "perf/global_max": global_max,
                    "perf/global_mean_best": global_mean_best,
                }
            )
            self.train_logger.write("env/epoch", epoch, env_perf_log_data)

    def training_fn(self, epoch: int, env_step: int) -> None:
        """
        训练回调函数
        """
        self.logger.info(
            f"[Epoch {epoch}] Training step callback triggered at env step {env_step}."
        )

        # 自适应熵系数控制 - 根据错开策略调整
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()

        # 性能驱动自适应熵控制
        if controller_type == "v4":
            self._update_entropy_coef_by_performance(epoch, env_step)
        # 趋势驱动自适应熵控制（v5版本）
        elif controller_type == "v5":
            self._update_entropy_coef_by_trend(epoch, env_step)
        # 梯度范数平衡自适应熵控制（独立启用，可与其他控制器并存）
        elif controller_type == "v6":
            self._update_entropy_coef_by_gradient_norm(epoch, env_step)
        # Beta-Head 自适应熵控制（v7版本，纯梯度范数比例平衡）
        elif controller_type == "v7":
            self._update_entropy_coef_by_beta_head(epoch, env_step)
        # Dual-Coefficient 自适应熵控制（v8版本，双系数：粗调 s_h + 细调 β_h）
        elif controller_type == "v8":
            self._update_entropy_coef_by_dual_coefficient(epoch, env_step)
        elif controller_type in ["v2", "v3"]:
            stagger_strategy = self.ppo_adaptive_ent_stagger_strategy.lower()

            if stagger_strategy == "alternate":
                # 策略1: 交替调整（奇数epoch调整连续，偶数epoch调整离散）
                if epoch % 2 == 1:
                    self._update_entropy_coef_cont(epoch, env_step)
                else:
                    self._update_entropy_coef_disc_thread_type(epoch, env_step)

            elif stagger_strategy == "priority":
                # 策略2: 优先级调整（优先调整偏离目标更远的部分）
                # 获取连续和离散的熵值
                policy = self.algorithm
                cont_entropy = (
                    policy.info_for_entropy_adaptive_controller.last_mean_cont_entropy
                )
                disc_entropy = policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_thread_type

                # 计算偏离程度
                cont_distance = 0.0
                disc_distance = 0.0

                if cont_entropy is not None:
                    target_ratio = self.ppo_adaptive_ent_target_ratio_cont
                    # 计算偏离目标的比例（距离越远，优先级越高）
                    cont_distance = abs(cont_entropy - target_ratio) / target_ratio

                if disc_entropy is not None:
                    target_entropy_min = (
                        self.ppo_adaptive_ent_target_entropy_min_disc_thread_type
                    )
                    target_entropy_max = (
                        self.ppo_adaptive_ent_target_entropy_max_disc_thread_type
                    )
                    target_entropy_mid = (target_entropy_min + target_entropy_max) / 2.0
                    # 计算偏离目标的程度
                    disc_distance = (
                        abs(disc_entropy - target_entropy_mid) / target_entropy_mid
                    )

                # 优先调整偏离程度更大的部分
                if cont_distance > disc_distance:
                    self._update_entropy_coef_cont(epoch, env_step)
                else:
                    self._update_entropy_coef_disc_thread_type(epoch, env_step)

            elif stagger_strategy == "frequency":
                # 策略3: 频率调整（连续每N步，离散每M步）
                freq_cont = self.ppo_adaptive_ent_update_freq_cont
                freq_disc = self.ppo_adaptive_ent_update_freq_disc

                if epoch % freq_cont == 0:
                    self._update_entropy_coef_cont(epoch, env_step)
                if epoch % freq_disc == 0:
                    self._update_entropy_coef_disc_thread_type(epoch, env_step)

            else:
                # 默认: 同时调整（保持向后兼容）
                self._update_entropy_coef_cont(epoch, env_step)
                self._update_entropy_coef_disc_thread_type(epoch, env_step)

        # 每隔一段epoch重置RND predictor，防止RND探索失效
        self.algorithm.reset_rnd_module(epoch, env_step)

        # 对rnd权重进行衰减
        self.algorithm.update_rnd_intrinsic_weight()

        self.train_gradients_logger.do_logging(epoch, env_step, self.actor, self.critic)

        self._logging_agent_global_performance(
            epoch, env_step
        )  # 记录环境的全局平均性能和最好性能

        # 保存checkpoint
        self.save_checkpoint_fn(epoch, env_step)

    def _update_entropy_coef_cont(self, epoch: int, env_step: int) -> None:
        """
        更新熵系数（自适应控制，连续动作部分）
        支持V2版本控制器的loss比例监控
        """
        # 从policy中获取最近的连续动作熵
        policy = self.algorithm
        actual_entropy = (
            policy.info_for_entropy_adaptive_controller.last_mean_cont_entropy
        )

        # 计算最大熵（连续动作部分）
        try:
            # 获取连续动作维度
            cont_action_dim = self.gemm_env.get_continuous_action_dim()
            # 获取sigma范围
            sigma_max = self.actor_net_sigma_max
            # 计算最大熵
            max_entropy = compute_max_entropy_for_gaussian(cont_action_dim, sigma_max)
        except Exception as e:
            self.logger.warning(
                f"Failed to compute max_entropy: {e}, using default value"
            )
            max_entropy = 10.0  # 默认值

        # 获取base_loss和ent_loss（用于V2版本控制器）
        controller_type = self.ppo_use_adaptive_ent_coef_type.lower()
        base_loss = policy.info_for_entropy_adaptive_controller.last_mean_base_loss
        ent_loss = policy.info_for_entropy_adaptive_controller.last_mean_ent_loss

        # 更新熵系数（连续动作部分）
        if controller_type == "v2":
            # V2版本：传递base_loss和ent_loss
            new_ent_coef = self.adaptive_ent_controller_cont.update(
                epoch=epoch,
                actual_entropy=actual_entropy,
                max_entropy=max_entropy,
                base_loss=base_loss,
                ent_loss=ent_loss,
            )
        else:
            # V1版本：仅传递entropy信息
            new_ent_coef = self.adaptive_ent_controller_cont.update(
                epoch=epoch,
                actual_entropy=actual_entropy,
                max_entropy=max_entropy,
            )

        # 应用新的熵系数到policy（连续动作部分）
        if hasattr(policy, "ent_coef_cont"):
            old_ent_coef = policy.ent_coef_cont
            policy.ent_coef_cont = new_ent_coef
            if actual_entropy is not None:
                log_msg = (
                    f"[Epoch {epoch}] Entropy coefficient updated (continuous actions): "
                    f"ent_coef_cont {old_ent_coef:.6f} -> {new_ent_coef:.6f} "
                    f"(actual_entropy={actual_entropy:.4f}, max_entropy={max_entropy:.4f})"
                )
                if controller_type == "v2" and (
                    base_loss is not None and ent_loss is not None
                ):
                    entropy_loss_ratio = ent_loss / base_loss
                    log_msg += f", entropy_loss_ratio={entropy_loss_ratio:.4f}"
                self.logger.info(log_msg)
            else:
                self.logger.info(
                    f"[Epoch {epoch}] Entropy coefficient updated (continuous actions): "
                    f"ent_coef_cont {old_ent_coef:.6f} -> {new_ent_coef:.6f} "
                    f"(actual_entropy=None, max_entropy={max_entropy:.4f})"
                )
        else:
            self.logger.warning(
                f"[Epoch {epoch}] Policy does not have ent_coef_cont attribute"
            )

    def _update_entropy_coef_disc_thread_type(self, epoch: int, env_step: int) -> None:
        """
        更新熵系数（自适应控制，离散动作thread_type部分）
        """
        # 检查控制器是否启用
        if self.adaptive_ent_controller_disc is None:
            return

        # 从policy中获取最近的thread_type原始熵值
        policy = self.algorithm
        actual_entropy = policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_thread_type

        # 获取base_loss和ent_loss
        base_loss = policy.info_for_entropy_adaptive_controller.last_mean_base_loss
        ent_loss = policy.info_for_entropy_adaptive_controller.last_mean_ent_loss

        # 更新熵系数
        new_ent_coef = self.adaptive_ent_controller_disc.update(
            epoch=epoch,
            actual_entropy=actual_entropy,
            base_loss=base_loss,
            ent_loss=ent_loss,
        )

        # 应用新的熵系数到policy（离散动作thread_type部分）
        if hasattr(policy, "_ent_coef_disc_thread_type"):
            old_ent_coef = policy._ent_coef_disc_thread_type
            policy._ent_coef_disc_thread_type = new_ent_coef

            if actual_entropy is not None:
                log_msg = (
                    f"[Epoch {epoch}] Entropy coefficient updated (discrete thread_type): "
                    f"ent_coef_disc_thread_type {old_ent_coef:.6f} -> {new_ent_coef:.6f} "
                    f"(actual_entropy={actual_entropy:.4f})"
                )
                if base_loss is not None and ent_loss is not None:
                    # 计算entropy_loss_ratio用于日志
                    if abs(base_loss) > 1e-6:
                        entropy_loss_ratio = abs(ent_loss / base_loss)
                        log_msg += f", entropy_loss_ratio={entropy_loss_ratio:.4f}"
                self.logger.info(log_msg)
            else:
                self.logger.info(
                    f"[Epoch {epoch}] Entropy coefficient updated (discrete thread_type): "
                    f"ent_coef_disc_thread_type {old_ent_coef:.6f} -> {new_ent_coef:.6f} "
                    f"(actual_entropy=None)"
                )
        else:
            self.logger.warning(
                f"[Epoch {epoch}] Policy does not have _ent_coef_disc_thread_type attribute"
            )

    def _update_entropy_coef_by_performance(self, epoch: int, env_step: int) -> None:
        """
        基于性能更新所有熵系数
        """
        if self.adaptive_ent_controller_performance is None:
            return

        policy = self.algorithm

        # 获取当前性能和各原始熵值
        current_performance = (
            policy.info_for_entropy_adaptive_controller.last_mean_env_performance
        )
        if current_performance is None:
            self.logger.warning(
                f"[Epoch {epoch}] No environment performance data available"
            )
            return

        # 获取各原始熵值
        raw_entropies = {
            "loop_order": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_loop_order,
            "packing_type": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_packing_type,
            "thread_type": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_thread_type,
            "cont": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_cont,
        }

        # 获取loss信息（用于约束）
        base_loss = policy.info_for_entropy_adaptive_controller.last_mean_base_loss
        ent_loss = policy.info_for_entropy_adaptive_controller.last_mean_ent_loss

        # 获取当前系数值
        current_coefs = {}
        if hasattr(policy, "_ent_coef_disc_loop_order"):
            current_coefs["loop_order"] = policy._ent_coef_disc_loop_order
        if hasattr(policy, "_ent_coef_disc_packing_type"):
            current_coefs["packing_type"] = policy._ent_coef_disc_packing_type
        if hasattr(policy, "_ent_coef_disc_thread_type"):
            current_coefs["thread_type"] = policy._ent_coef_disc_thread_type
        if hasattr(policy, "_ent_coef_cont"):
            current_coefs["cont"] = policy._ent_coef_cont

        # 更新系数
        new_coefs = self.adaptive_ent_controller_performance.update(
            epoch=epoch,
            current_performance=current_performance,
            raw_entropies=raw_entropies,
            base_loss=base_loss,
            ent_loss=ent_loss,
            current_coefs=current_coefs,
        )

        # 应用新系数
        updated_count = 0
        for action_type, new_coef in new_coefs.items():
            if action_type == "loop_order" and hasattr(
                policy, "_ent_coef_disc_loop_order"
            ):
                policy._ent_coef_disc_loop_order = new_coef
                updated_count += 1
            elif action_type == "packing_type" and hasattr(
                policy, "_ent_coef_disc_packing_type"
            ):
                policy._ent_coef_disc_packing_type = new_coef
                updated_count += 1
            elif action_type == "thread_type" and hasattr(
                policy, "_ent_coef_disc_thread_type"
            ):
                policy._ent_coef_disc_thread_type = new_coef
                updated_count += 1
            elif action_type == "cont" and hasattr(policy, "_ent_coef_cont"):
                policy._ent_coef_cont = new_coef
                updated_count += 1

        if updated_count > 0 and epoch % 20 == 0:
            self.logger.info(
                f"[Epoch {epoch}] Performance-driven coefficients updated for {updated_count}/4 actions"
            )

    def _update_entropy_coef_by_trend(self, epoch: int, env_step: int) -> None:
        """
        基于趋势更新所有熵系数（v5版本）
        """
        if self.adaptive_ent_controller_trend is None:
            return

        policy = self.algorithm

        # 获取当前性能和entropy loss
        current_performance = (
            policy.info_for_entropy_adaptive_controller.last_mean_env_performance
        )
        if current_performance is None:
            self.logger.warning(
                f"[Epoch {epoch}] No environment performance data available"
            )
            return

        # 获取entropy loss
        ent_loss = policy.info_for_entropy_adaptive_controller.last_mean_ent_loss
        if ent_loss is None:
            self.logger.warning(f"[Epoch {epoch}] No entropy loss data available")
            return

        # 获取各原始熵值
        raw_entropies = {
            "loop_order": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_loop_order,
            "packing_type": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_packing_type,
            "thread_type": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_disc_thread_type,
            "cont": policy.info_for_entropy_adaptive_controller.last_mean_raw_entropy_cont,
        }

        # 获取当前系数值
        current_coefs = {}
        if hasattr(policy, "_ent_coef_disc_loop_order"):
            current_coefs["loop_order"] = policy._ent_coef_disc_loop_order
        if hasattr(policy, "_ent_coef_disc_packing_type"):
            current_coefs["packing_type"] = policy._ent_coef_disc_packing_type
        if hasattr(policy, "_ent_coef_disc_thread_type"):
            current_coefs["thread_type"] = policy._ent_coef_disc_thread_type
        if hasattr(policy, "_ent_coef_cont"):
            current_coefs["cont"] = policy._ent_coef_cont

        # 更新系数（注意：v5版本不需要base_loss）
        new_coefs = self.adaptive_ent_controller_trend.update(
            epoch=epoch,
            current_performance=current_performance,
            entropy_loss=ent_loss,
            raw_entropies=raw_entropies,
            current_coefs=current_coefs,
        )

        # 应用新系数
        updated_count = 0
        for action_type, new_coef in new_coefs.items():
            if action_type == "loop_order" and hasattr(
                policy, "_ent_coef_disc_loop_order"
            ):
                policy._ent_coef_disc_loop_order = new_coef
                updated_count += 1
            elif action_type == "packing_type" and hasattr(
                policy, "_ent_coef_disc_packing_type"
            ):
                policy._ent_coef_disc_packing_type = new_coef
                updated_count += 1
            elif action_type == "thread_type" and hasattr(
                policy, "_ent_coef_disc_thread_type"
            ):
                policy._ent_coef_disc_thread_type = new_coef
                updated_count += 1
            elif action_type == "cont" and hasattr(policy, "_ent_coef_cont"):
                policy._ent_coef_cont = new_coef
                updated_count += 1
            else:
                pass

        if updated_count > 0 and epoch % 5 == 0:
            self.logger.info(
                f"[Epoch {epoch}] Trend-driven coefficients updated for {updated_count}/4 actions"
            )

    def _update_entropy_coef_by_gradient_norm(self, epoch: int, env_step: int) -> None:
        """
        基于梯度范数平衡更新所有熵系数
        """
        if self.adaptive_entropy_balancer_gradient_norm is None:
            return None

        if epoch % self.gradient_norm_balancer_update_frequency != 0:
            return None

        policy = self.algorithm

        # 调用policy的方法计算并保存loss张量（使用保存的batch数据）
        # 这样就实现了频率控制：只有当需要更新系数时才计算loss张量
        losses = policy.compute_and_save_losses_for_gradient_norm()
        if losses is None:
            return None
        else:
            (
                clip_loss_tensor,
                raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont_tensor,
            ) = losses

        # 检查必要的loss张量是否可用
        if clip_loss_tensor is None:
            self.logger.warning(
                f"[Epoch {epoch}] No clip_loss tensor available for gradient norm computation"
            )
            return None

        if any(
            tensor is None
            for tensor in [
                raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont_tensor,
            ]
        ):
            self.logger.warning(
                f"[Epoch {epoch}] Some entropy loss tensors not available for gradient norm computation"
            )
            return None

        # 获取共享参数（attention_net的参数）
        try:
            shared_params = list(self.actor._attention_net.parameters())
        except AttributeError:
            self.logger.warning(
                f"[Epoch {epoch}] Cannot access _attention_net parameters from policy"
            )
            return None

        # 获取当前系数值
        current_coefs = [
            policy._ent_coef_disc_loop_order,
            policy._ent_coef_disc_packing_type,
            policy._ent_coef_disc_thread_type,
            policy._ent_coef_cont,
        ]

        # 更新系数（传递平均熵值用于 SAC 调节）
        updated_coefs = (
            self.adaptive_entropy_balancer_gradient_norm.update_coefficients(
                epoch=epoch,
                shared_params=shared_params,
                raw_ent_loss_loop_order=raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type=raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type=raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont=raw_ent_loss_cont_tensor,
                clip_loss=clip_loss_tensor,
                current_coefs=current_coefs,
            )
        )

        # 应用新系数
        if hasattr(policy, "_ent_coef_disc_loop_order"):
            policy._ent_coef_disc_loop_order = updated_coefs[0]
        if hasattr(policy, "_ent_coef_disc_packing_type"):
            policy._ent_coef_disc_packing_type = updated_coefs[1]
        if hasattr(policy, "_ent_coef_disc_thread_type"):
            policy._ent_coef_disc_thread_type = updated_coefs[2]
        if hasattr(policy, "_ent_coef_cont"):
            policy._ent_coef_cont = updated_coefs[3]

        # 记录调节器的统计信息
        grad_norms = self.adaptive_entropy_balancer_gradient_norm.get_last_grad_norms()
        rel_strengths = (
            self.adaptive_entropy_balancer_gradient_norm.get_last_relative_strengths()
        )

        self.logger.info(
            f"[Epoch {epoch}] Gradient norm entropy balancer update:\n"
            f"  Gradient norms (EMA):\n"
            f"    loop_order: {grad_norms.ent_loop_order:.6e}\n"
            f"    packing_type: {grad_norms.ent_packing_type:.6e}\n"
            f"    thread_type: {grad_norms.ent_thread_type:.6e}\n"
            f"    continuous: {grad_norms.ent_cont:.6e}\n"
            f"    policy: {grad_norms.policy:.6e}\n"
            f"  Relative strengths:\n"
            f"    loop_order: {rel_strengths.loop_order:.6f}\n"
            f"    packing_type: {rel_strengths.packing_type:.6f}\n"
            f"    thread_type: {rel_strengths.thread_type:.6f}\n"
            f"    continuous: {rel_strengths.cont:.6f}\n"
            f"  Updated coefficients:\n"
            f"    loop_order: {updated_coefs[0]:.6e}\n"
            f"    packing_type: {updated_coefs[1]:.6e}\n"
            f"    thread_type: {updated_coefs[2]:.6e}\n"
            f"    continuous: {updated_coefs[3]:.6e}"
        )

    def _update_entropy_coef_by_beta_head(self, epoch: int, env_step: int) -> None:
        """
        基于 Beta-Head 梯度范数比例平衡更新所有熵系数（v7版本）

        使用纯梯度范数比例平衡算法，统一处理离散和连续头。
        """
        if self.adaptive_entropy_beta_head_balancer is None:
            return None

        policy = self.algorithm

        # 调用policy的方法计算并保存loss张量（使用保存的batch数据）
        losses = policy.compute_and_save_losses_for_gradient_norm()
        if losses is None:
            return None
        else:
            (
                clip_loss_tensor,
                raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont_tensor,
            ) = losses

        # 检查必要的loss张量是否可用
        if clip_loss_tensor is None:
            self.logger.warning(
                f"[Epoch {epoch}] No clip_loss tensor available for beta-head balancer computation"
            )
            return None

        if any(
            tensor is None
            for tensor in [
                raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont_tensor,
            ]
        ):
            self.logger.warning(
                f"[Epoch {epoch}] Some entropy loss tensors not available for beta-head balancer computation"
            )
            return None

        # 获取共享参数（attention_net的参数）
        try:
            shared_params = list(self.actor._attention_net.parameters())
        except AttributeError:
            self.logger.warning(
                f"[Epoch {epoch}] Cannot access _attention_net parameters from policy"
            )
            return None

        # 构建熵损失字典
        ent_losses = {
            "loop_order": raw_ent_loss_loop_order_tensor,
            "packing_type": raw_ent_loss_packing_type_tensor,
            "thread_type": raw_ent_loss_thread_type_tensor,
            "continuous": raw_ent_loss_cont_tensor,
        }

        # 更新系数（使用新的 Beta-Head 接口）
        updated_coefs = self.adaptive_entropy_beta_head_balancer.update_coefficients(
            shared_params=shared_params,
            policy_loss=clip_loss_tensor,
            ent_losses=ent_losses,
            current_step=epoch,
        )

        # 应用新系数
        if hasattr(policy, "_ent_coef_disc_loop_order"):
            policy._ent_coef_disc_loop_order = updated_coefs["loop_order"]
        if hasattr(policy, "_ent_coef_disc_packing_type"):
            policy._ent_coef_disc_packing_type = updated_coefs["packing_type"]
        if hasattr(policy, "_ent_coef_disc_thread_type"):
            policy._ent_coef_disc_thread_type = updated_coefs["thread_type"]
        if hasattr(policy, "_ent_coef_cont"):
            policy._ent_coef_cont = updated_coefs["continuous"]

        # 记录调节器的统计信息
        stats = self.adaptive_entropy_beta_head_balancer.get_last_stats()
        ema_grad_norms = self.adaptive_entropy_beta_head_balancer.get_ema_grad_norms()

        self.logger.info(
            f"[Epoch {epoch}] Beta-Head entropy balancer update:\n"
            f"  Gradient norms (EMA):\n"
            f"    policy: {ema_grad_norms['policy']:.6e}\n"
            f"    loop_order: {ema_grad_norms['loop_order']:.6e}\n"
            f"    packing_type: {ema_grad_norms['packing_type']:.6e}\n"
            f"    thread_type: {ema_grad_norms['thread_type']:.6e}\n"
            f"    continuous: {ema_grad_norms['continuous']:.6e}\n"
            f"  Gradient ratios (w_h):\n"
            f"    loop_order: {stats.ratio_loop_order:.6f}\n"
            f"    packing_type: {stats.ratio_packing_type:.6f}\n"
            f"    thread_type: {stats.ratio_thread_type:.6f}\n"
            f"    continuous: {stats.ratio_continuous:.6f}\n"
            f"  Updated coefficients:\n"
            f"    loop_order: {updated_coefs['loop_order']:.6e}\n"
            f"    packing_type: {updated_coefs['packing_type']:.6e}\n"
            f"    thread_type: {updated_coefs['thread_type']:.6e}\n"
            f"    continuous: {updated_coefs['continuous']:.6e}"
        )

        if stats.triggered_safety_correction:
            self.logger.info(
                f"[Epoch {epoch}] Safety correction triggered for heads: {stats.triggered_safety_correction}"
            )

        if stats.triggered_emergency_reset:
            self.logger.warning(
                f"[Epoch {epoch}] Emergency reset triggered due to policy gradient vanishing"
            )

    def _update_entropy_coef_by_dual_coefficient(
        self, epoch: int, env_step: int
    ) -> None:
        """
        基于 Dual-Coefficient 梯度范数比例平衡更新所有熵系数（v8版本）

        使用双系数调节算法：
        - β_h: 细调系数，快速响应梯度范数比例变化
        - s_h: 粗调系数，缓慢调整以稳定训练
        有效系数 = β_h * s_h
        """
        if self.adaptive_entropy_dual_coefficient_balancer is None:
            return None

        policy = self.algorithm

        # 调用policy的方法计算并保存loss张量（使用保存的batch数据）
        losses = policy.compute_and_save_losses_for_gradient_norm()
        if losses is None:
            return None
        else:
            (
                clip_loss_tensor,
                raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont_tensor,
            ) = losses

        # 检查必要的loss张量是否可用
        if clip_loss_tensor is None:
            self.logger.warning(
                f"[Epoch {epoch}] No clip_loss tensor available for dual-coefficient balancer computation"
            )
            return None

        if any(
            tensor is None
            for tensor in [
                raw_ent_loss_loop_order_tensor,
                raw_ent_loss_packing_type_tensor,
                raw_ent_loss_thread_type_tensor,
                raw_ent_loss_cont_tensor,
            ]
        ):
            self.logger.warning(
                f"[Epoch {epoch}] Some entropy loss tensors not available for dual-coefficient balancer computation"
            )
            return None

        # 获取共享参数（attention_net的参数）
        try:
            shared_params = list(self.actor._attention_net.parameters())
        except AttributeError:
            self.logger.warning(
                f"[Epoch {epoch}] Cannot access _attention_net parameters from policy"
            )
            return None

        # 构建原始熵损失字典
        raw_ent_losses = {
            "loop_order": raw_ent_loss_loop_order_tensor,
            "packing_type": raw_ent_loss_packing_type_tensor,
            "thread_type": raw_ent_loss_thread_type_tensor,
            "continuous": raw_ent_loss_cont_tensor,
        }

        # 更新系数（使用 Dual-Coefficient 接口）
        beta_coefs, scale_factors = (
            self.adaptive_entropy_dual_coefficient_balancer.update(
                shared_params=shared_params,
                policy_loss=clip_loss_tensor,
                raw_ent_losses=raw_ent_losses,
                current_step=epoch,
            )
        )

        # 应用新系数：beta_coef 赋给 _ent_coef_*，scale_factor 赋给 _scale_factor_*
        # 注意：在 policy 的 _update_with_batch 中，会使用 beta * scale 作为有效系数
        # 这里分开赋值是为了保持系数的独立性，便于调试和统计
        head_names = ["loop_order", "packing_type", "thread_type", "continuous"]
        for head in head_names:
            if (
                hasattr(policy, f"_ent_coef_disc_{head}")
                if head != "continuous"
                else hasattr(policy, "_ent_coef_cont")
            ):
                if head == "continuous":
                    policy._ent_coef_cont = beta_coefs[head]
                    if hasattr(policy, "_scale_factor_cont"):
                        policy._scale_factor_cont = scale_factors[head]
                else:
                    attr_name = f"_ent_coef_disc_{head}"
                    setattr(policy, attr_name, beta_coefs[head])
                    scale_attr_name = f"_scale_factor_{head}"
                    if hasattr(policy, scale_attr_name):
                        setattr(policy, scale_attr_name, scale_factors[head])

        # 记录调节器的统计信息
        stats = self.adaptive_entropy_dual_coefficient_balancer.get_last_stats()
        log_str_list = [
            f"[Epoch {epoch}] Dual-Coefficient entropy balancer update:",
            f"  Gradient norms:",
            f"    policy: {stats.grad_norm_policy}",
            f"    loop_order: {stats.grad_norm_ent_loop_order}",
            f"    packing_type: {stats.grad_norm_ent_packing_type}",
            f"    thread_type: {stats.grad_norm_ent_thread_type}",
            f"    continuous: {stats.grad_norm_ent_continuous}",
        ]

        if stats.do_update:
            log_str_list.extend(
                [
                    f"  Gradient ratios (w_h):",
                    f"    loop_order: {stats.ratio_loop_order}",
                    f"    packing_type: {stats.ratio_packing_type}",
                    f"    thread_type: {stats.ratio_thread_type}",
                    f"    continuous: {stats.ratio_continuous}",
                    f"  Beta coefficients (β_h):",
                    f"    loop_order: {stats.beta_loop_order}",
                    f"    packing_type: {stats.beta_packing_type}",
                    f"    thread_type: {stats.beta_thread_type}",
                    f"    continuous: {stats.beta_continuous}",
                    f"  Scale factors (s_h):",
                    f"    loop_order: {stats.scale_loop_order}",
                    f"    packing_type: {stats.scale_packing_type}",
                    f"    thread_type: {stats.scale_thread_type}",
                    f"    continuous: {stats.scale_continuous}",
                    f"  Effective coefficients (β_h * s_h):",
                    f"    loop_order: {stats.effective_coef_loop_order}",
                    f"    packing_type: {stats.effective_coef_packing_type}",
                    f"    thread_type: {stats.effective_coef_thread_type}",
                    f"    continuous: {stats.effective_coef_continuous}"
                ]
            )

            self.logger.info("\n".join(log_str_list))

            if stats.triggered_safety_correction:
                self.logger.info(
                    f"[Epoch {epoch}] Safety correction triggered for heads: {stats.triggered_safety_correction}"
                )

            if stats.triggered_emergency_reset:
                self.logger.warning(
                    f"[Epoch {epoch}] Emergency reset triggered due to policy gradient vanishing"
                )
        else:
            self.logger.info("\n".join(log_str_list))

    def test_fn(self, epoch: int, env_step: int) -> None:
        """
        测试回调函数
        """
        pass

    def save_checkpoint_by_env_performance_fn(self, epoch: int) -> str | None:
        """
        按性能保存checkpoint
        """
        # 调用保存函数
        current_performance = self.algorithm.last_mean_env_performance
        save_threshold_performance = self._last_best_env_performance * (
            1 + self.best_performance_improve_threshold
        )
        if current_performance >= save_threshold_performance:
            # 列出目录中checkpoint的文件，按照修改时间排序，如果超过保存的数量限制，就删除最旧的
            file_list = filter(
                lambda x: x.startswith("best_performance_epoch") and x.endswith(".pth"),
                os.listdir(self.model_save_dir),
            )
            sorted_files = sorted(
                file_list,
                key=lambda x: os.path.getmtime(os.path.join(self.model_save_dir, x)),
                reverse=True,
            )
            if len(sorted_files) >= self.best_performance_max_num:
                for old_file in sorted_files[self.best_performance_max_num - 1 :]:
                    old_file_path = os.path.join(self.model_save_dir, old_file)
                    os.remove(old_file_path)
                    self.logger.info(
                        f"Old best performance checkpoint {old_file_path} removed to maintain max limit."
                    )

            model_save_path = self.save_policy_configs(
                self.algorithm,
                f"best_performance_epoch{epoch}_at_{current_performance:.3f}",
                add_timestamp=False,
            )
            self.logger.info(
                f"[Epoch {epoch}] Performance improved with {self._last_best_env_performance:.3f}GFlops -> {current_performance:.3f}GFlops and checkpoint is saved at {model_save_path}"
            )
            self._last_best_env_performance = current_performance
            return model_save_path
        return None

    def save_checkpoint_fn(self, epoch: int, env_step: int) -> str:
        """
        保存checkpoint, 需返回保存的文件路径
        """
        save_path = self.save_checkpoint_by_env_performance_fn(epoch)
        if save_path is not None:
            return save_path

        # 检查是否有手动保存的需求
        save_path = self.manual_save_checkpoint_fn(epoch)
        if save_path is not None:
            return save_path

        # 保存checkpoint
        save_path = self.save_checkpoint_by_interval_fn(epoch)
        if save_path is not None:
            return save_path

        return ""

    def do_train(self):
        """
        开始训练
        """
        self.logger.info("start training")

        result = self.algorithm.run_training(
            OnPolicyTrainerParams(
                training_collector=self.training_collector,
                test_collector=self.test_collector,
                max_epochs=self.train_max_epochs,
                epoch_num_steps=self.train_epoch_num_steps,
                update_step_num_repetitions=self.train_update_step_num_repetitions,
                test_step_num_episodes=self.train_test_step_num_episodes,
                batch_size=self.train_batch_size,
                collection_step_num_env_steps=self.train_collection_step_num_env_steps,
                stop_fn=self.stop_fn,
                save_best_fn=self.save_best_fn,
                logger=self.train_logger,
                test_in_training=False,
                training_fn=self.training_fn,
                test_fn=self.test_fn,
            )
        )

        model_save_path = self.save_policy_configs(self.algorithm, "final_policy")
        self.logger.info(
            "Final model is saved in %s, result: %s",
            model_save_path,
            result.pprints_asdict(),
        )

    def do_inference(self, shapes: List[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
        """
        执行推理
        """
        # 设置模型为推理模式
        self.algorithm.eval()
        result = []
        for shape in shapes:
            m, k, n = shape
            self.gemm_env.set_gemm_shape(m, k, n)
            obs, info = self.gemm_env.reset()
            with torch.no_grad():
                action = self.actor_policy.compute_action(obs, info=info, state=None)
                next_obs, reward, done, truncated, result_info = self.gemm_env.step(
                    action
                )
                if result_info:
                    result.append(result_info)
        return result
