# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025
# Contact       : hucheng.lew@qq.com
# File          : adaptive_entropy_balancer.py
# IDE           : PyCharm
# Description   : Adaptive entropy coefficient balancer with dual-mechanism regulation
#                 - SAC-style temperature adjustment for discrete heads
#                 - Gradient norm balancing (Scheme 3) as safety correction and for continuous head

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# Action head names constant
HEAD_NAMES = ["loop_order", "packing_type", "thread_type", "continuous"]


@dataclass
class HeadConfig:
    """Configuration for a single action head's entropy coefficient balancing."""

    w_target: float = (
        0.1  # Target relative strength (entropy gradient / policy gradient)
    )
    learning_rate: float = 0.2  # Learning rate for coefficient update (η)
    coef_min: float = 1e-5  # Minimum entropy coefficient (β_min)
    coef_max: float = 0.5  # Maximum entropy coefficient (β_max)
    target_entropy: Optional[Union[float, Dict[str, Any]]] = (
        None  # Target entropy for SAC-style adjustment (only for discrete heads)
        # Supports two formats:
        # - float: Fixed target entropy (no decay)
        # - Dict: {"start", "end", "decay_start_epoch", "decay_rate"} (dynamic decay)
    )
    # Safety correction parameters (per-head independent configuration)
    grad_safe_ratio_min: float = 0.02  # Minimum safe value for gradient norm ratio
    grad_safe_ratio_max: float = 0.5  # Maximum safe value for gradient norm ratio
    safe_tau: float = 0.1  # Smoothing coefficient for safety correction


# Predefined default configurations for each action head
# Target entropies are set to ~40% of max entropy for discrete heads:
# - loop_order: ln(9) ≈ 2.197, target ≈ 0.88
# - packing_type: ln(2) ≈ 0.693, target ≈ 0.28
# - thread_type: ln(7) ≈ 1.946, target ≈ 0.78
# Default configs use float format (fixed target entropy, no decay)
DEFAULT_HEAD_CONFIGS: Dict[str, HeadConfig] = {
    "loop_order": HeadConfig(
        w_target=0.10,
        learning_rate=0.20,
        coef_min=1e-5,
        coef_max=0.5,
        target_entropy=0.88,  # Fixed target entropy (no decay)
        grad_safe_ratio_min=0.02,
        grad_safe_ratio_max=0.5,
        safe_tau=0.1,
    ),
    "packing_type": HeadConfig(
        w_target=0.12,
        learning_rate=0.25,
        coef_min=1e-5,
        coef_max=0.5,
        target_entropy=0.28,  # Fixed target entropy (no decay)
        grad_safe_ratio_min=0.02,
        grad_safe_ratio_max=0.5,
        safe_tau=0.1,
    ),
    "thread_type": HeadConfig(
        w_target=0.10,
        learning_rate=0.20,
        coef_min=1e-5,
        coef_max=0.5,
        target_entropy=0.78,  # Fixed target entropy (no decay)
        grad_safe_ratio_min=0.02,
        grad_safe_ratio_max=0.5,
        safe_tau=0.1,
    ),
    "continuous": HeadConfig(
        w_target=0.03,
        learning_rate=0.10,
        coef_min=1e-6,
        coef_max=0.01,
        target_entropy=None,
        grad_safe_ratio_min=0.02,  # Continuous head also participates in safety correction
        grad_safe_ratio_max=5.0,  # Upper limit set to 5.0, exceeding triggers fuse
        safe_tau=0.3,  # Higher smoothing coefficient for faster correction
    ),
}


@dataclass
class GradientNormStats:
    """Statistics for gradient norms"""

    ent_loop_order: float = 0.0
    ent_packing_type: float = 0.0
    ent_thread_type: float = 0.0
    ent_cont: float = 0.0
    policy: float = 0.0


@dataclass
class RelativeStrengthStats:
    """Statistics for relative strength (w_i)"""

    loop_order: float = 0.0
    packing_type: float = 0.0
    thread_type: float = 0.0
    cont: float = 0.0


class AdaptiveEntropyBalancer:
    """
    Adaptive entropy coefficient balancer with dual-mechanism regulation.

    For discrete action heads (loop_order, packing_type, thread_type):
    - Primary regulator: SAC-style temperature auto-adjustment based on target entropy
    - Safety valve: Gradient norm ratio correction (Scheme 3) when ratio is out of safe range

    For continuous action head:
    - Pure gradient norm ratio balancing (Scheme 3)

    Core Algorithm:
    1. Compute gradient norms of each entropy loss and policy loss w.r.t. shared parameters
    2. Use EMA to smooth gradient norms and entropies, reduce noise
    3. For discrete heads:
       - SAC update: β_new = β_old - η_SAC * (H_target - H_current)
       - Safety correction: if weighted gradient ratio w is outside safe range,
         blend with suggested coefficient from Scheme 3
    4. For continuous head:
       - Multiplicative update: β_new = β_old * exp(η * (w_target - w_current))
    """

    def __init__(
        self,
        head_configs: Union[
            None,
            float,
            HeadConfig,
            Dict[str, Union[HeadConfig, Dict[str, float], float]],
        ] = None,
        ema_decay: float = 0.9,
        epsilon: float = 1e-8,
        update_frequency: int = 1,
        balance_continuous: bool = False,
        sac_lr: float = 0.01,
        use_sac_for_discrete: bool = True,
    ):
        """
        Initialize the adaptive entropy balancer.

        Args:
            head_configs: Configuration for action heads. Can be:
                - None: Use DEFAULT_HEAD_CONFIGS
                - float: Set w_target for all heads to this value, other fields use defaults
                - HeadConfig: All heads use this same configuration
                - Dict[str, ...]: Per-head configuration, where each value can be:
                    - HeadConfig: Complete configuration for that head
                    - Dict[str, float]: Partial configuration (missing fields use defaults)
                    - float: Only set w_target for that head
                  Heads not specified in dict use DEFAULT_HEAD_CONFIGS

            ema_decay: EMA decay factor for gradient norm smoothing (α)
                      Recommended value: 0.9
                      Higher values = more smoothing, slower adaptation

            epsilon: Small constant for numerical stability (ε)
                    Default: 1e-8

            update_frequency: Update coefficients every N calls
                             Default: 1 (update every call)
                             Higher values reduce computational overhead

            balance_continuous: Whether to balance continuous action head
                              Default: False (continuous head coefficient fixed)
                              When True, continuous head uses its own configuration

            sac_lr: Learning rate for SAC-style temperature adjustment (η_SAC)
                   Default: 0.01
                   Only used for discrete heads when use_sac_for_discrete=True

            use_sac_for_discrete: Whether to use SAC-style adjustment for discrete heads
                                 Default: True
                                 When False, discrete heads use pure gradient norm balancing
                                 Continuous head always uses gradient norm balancing
        """
        self.logger = logging.getLogger("AdaptiveEntropyBalancer")

        # Parse and store head configurations
        self._head_configs: Dict[str, HeadConfig] = self._parse_head_configs(
            head_configs
        )

        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.update_frequency = update_frequency
        self.balance_continuous = balance_continuous

        # SAC-related parameters
        self.sac_lr = sac_lr
        self.use_sac_for_discrete = use_sac_for_discrete

        # EMA values for gradient norms
        self._ema_grad_norm_ent_loop_order: Optional[float] = None
        self._ema_grad_norm_ent_packing_type: Optional[float] = None
        self._ema_grad_norm_ent_thread_type: Optional[float] = None
        self._ema_grad_norm_ent_cont: Optional[float] = None
        self._ema_grad_norm_policy: Optional[float] = None

        # EMA values for entropy (for SAC adjustment)
        self._ema_entropy_loop_order: Optional[float] = None
        self._ema_entropy_packing_type: Optional[float] = None
        self._ema_entropy_thread_type: Optional[float] = None

        # Update counter
        self._update_counter = 0

        # Statistics for logging and monitoring
        self._last_grad_norms = GradientNormStats()
        self._last_relative_strengths = RelativeStrengthStats()

        # Parse target_entropy configuration and initialize decay mechanism
        self._current_target_entropy = {}
        self._target_entropy_decay_configs = {}  # Store decay configs for dict format

        for name in HEAD_NAMES:
            target_entropy_cfg = self._head_configs[name].target_entropy

            if target_entropy_cfg is None:
                # No target entropy configured
                self._current_target_entropy[name] = None
                self._target_entropy_decay_configs[name] = None

            elif isinstance(target_entropy_cfg, float):
                # Float format: Fixed target entropy (no decay)
                self._current_target_entropy[name] = target_entropy_cfg
                self._target_entropy_decay_configs[name] = None

            elif isinstance(target_entropy_cfg, dict):
                # Dict format: Dynamic decay enabled
                # Required fields: "start", "end", "decay_start_epoch", "decay_rate"
                self._current_target_entropy[name] = target_entropy_cfg["start"]
                self._target_entropy_decay_configs[name] = target_entropy_cfg

        # Log the configuration
        self._log_configuration()

    def _parse_head_configs(
        self,
        configs: Union[
            None,
            float,
            HeadConfig,
            Dict[str, Union[HeadConfig, Dict[str, float], float]],
        ],
    ) -> Dict[str, HeadConfig]:
        """
        Parse head configurations from various input formats.

        Args:
            configs: Input configuration in various formats

        Returns:
            Dict[str, HeadConfig]: Parsed configuration for each head

        Raises:
            TypeError: If configs has invalid type
        """
        # Case 1: None - use defaults
        if configs is None:
            return DEFAULT_HEAD_CONFIGS.copy()

        # Case 2: float/int - set w_target for all heads
        if isinstance(configs, (int, float)):
            return {
                head: HeadConfig(
                    w_target=float(configs),
                    learning_rate=DEFAULT_HEAD_CONFIGS[head].learning_rate,
                    coef_min=DEFAULT_HEAD_CONFIGS[head].coef_min,
                    coef_max=DEFAULT_HEAD_CONFIGS[head].coef_max,
                    target_entropy=DEFAULT_HEAD_CONFIGS[head].target_entropy,
                    grad_safe_ratio_min=DEFAULT_HEAD_CONFIGS[head].grad_safe_ratio_min,
                    grad_safe_ratio_max=DEFAULT_HEAD_CONFIGS[head].grad_safe_ratio_max,
                    safe_tau=DEFAULT_HEAD_CONFIGS[head].safe_tau,
                )
                for head in HEAD_NAMES
            }

        # Case 3: HeadConfig - all heads use same config
        if isinstance(configs, HeadConfig):
            return {head: HeadConfig(**configs.__dict__) for head in HEAD_NAMES}

        # Case 4: Dict - per-head configuration
        if isinstance(configs, dict):
            parsed = {}
            for head in HEAD_NAMES:
                if head in configs:
                    cfg = configs[head]

                    # Sub-case 4a: HeadConfig object
                    if isinstance(cfg, HeadConfig):
                        parsed[head] = cfg

                    # Sub-case 4b: Partial dict
                    elif isinstance(cfg, dict):
                        default = DEFAULT_HEAD_CONFIGS[head]
                        merged = {
                            "w_target": cfg.get("w_target", default.w_target),
                            "learning_rate": cfg.get(
                                "learning_rate", default.learning_rate
                            ),
                            "coef_min": cfg.get("coef_min", default.coef_min),
                            "coef_max": cfg.get("coef_max", default.coef_max),
                            "target_entropy": cfg.get(
                                "target_entropy", default.target_entropy
                            ),
                            "grad_safe_ratio_min": cfg.get(
                                "grad_safe_ratio_min", default.grad_safe_ratio_min
                            ),
                            "grad_safe_ratio_max": cfg.get(
                                "grad_safe_ratio_max", default.grad_safe_ratio_max
                            ),
                            "safe_tau": cfg.get("safe_tau", default.safe_tau),
                        }
                        parsed[head] = HeadConfig(**merged)

                    # Sub-case 4c: float - only set w_target
                    elif isinstance(cfg, (int, float)):
                        default = DEFAULT_HEAD_CONFIGS[head]
                        parsed[head] = HeadConfig(
                            w_target=float(cfg),
                            learning_rate=default.learning_rate,
                            coef_min=default.coef_min,
                            coef_max=default.coef_max,
                            target_entropy=default.target_entropy,
                            grad_safe_ratio_min=default.grad_safe_ratio_min,
                            grad_safe_ratio_max=default.grad_safe_ratio_max,
                            safe_tau=default.safe_tau,
                        )

                    else:
                        raise TypeError(
                            f"Invalid config type for head '{head}': {type(cfg)}. "
                            f"Expected HeadConfig, dict, or float."
                        )
                else:
                    # Head not specified - use default
                    parsed[head] = DEFAULT_HEAD_CONFIGS[head]

            return parsed

        # Invalid type
        raise TypeError(
            f"head_configs must be None, float, HeadConfig, or dict. "
            f"Got {type(configs)}"
        )

    def _log_configuration(self):
        """Log the parsed head configurations."""
        self.logger.info("AdaptiveEntropyBalancer initialized with configurations:")
        for head, cfg in self._head_configs.items():
            # Format target_entropy display based on type
            if cfg.target_entropy is None:
                target_ent_str = ""
            elif isinstance(cfg.target_entropy, float):
                target_ent_str = (
                    f", target_entropy={cfg.target_entropy:.4f} (fixed, no decay)"
                )
            elif isinstance(cfg.target_entropy, dict):
                # Display dict format with decay information
                target_ent_str = (
                    f", target_entropy: start={cfg.target_entropy['start']:.4f}, "
                    f"end={cfg.target_entropy['end']:.4f}, "
                    f"decay_start_epoch={cfg.target_entropy['decay_start_epoch']}, "
                    f"decay_rate={cfg.target_entropy['decay_rate']:.4f} (dynamic decay)"
                )
            else:
                target_ent_str = f", target_entropy={cfg.target_entropy}"

            self.logger.info(
                f"  {head}: w_target={cfg.w_target:.4f}, "
                f"lr={cfg.learning_rate:.4f}, "
                f"bounds=[{cfg.coef_min:.6e}, {cfg.coef_max:.6e}], "
                f"safety_range=[{cfg.grad_safe_ratio_min:.2f}, {cfg.grad_safe_ratio_max:.2f}], "
                f"safe_tau={cfg.safe_tau:.2f}"
                f"{target_ent_str}"
            )

        self.logger.info(
            f"  balance_continuous={self.balance_continuous}, "
            f"ema_decay={self.ema_decay}, "
            f"update_frequency={self.update_frequency}"
        )
        self.logger.info(
            f"  SAC parameters: sac_lr={self.sac_lr}, "
            f"use_sac_for_discrete={self.use_sac_for_discrete}"
        )

    def compute_grad_norm(
        self,
        loss: torch.Tensor,
        shared_params: List[nn.Parameter],
    ) -> float:
        """
        Compute the L2 norm of gradients of loss w.r.t. shared parameters.

        This method uses torch.autograd.grad instead of .backward() to avoid
        interfering with the optimizer's gradient accumulation.

        Args:
            loss: The loss tensor to compute gradients for
            shared_params: List of shared parameters (from _attention_net)

        Returns:
            L2 norm of gradients (float), or 0.0 if computation fails
        """
        try:
            # Use torch.autograd.grad to compute gradients without affecting optimizer state
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,  # Keep computation graph for subsequent gradient computations
                create_graph=False,  # Don't create computation graph for gradient computation
                allow_unused=True,  # Handle parameters that don't contribute to loss
            )

            # Filter out None gradients (parameters not contributing to loss)
            valid_grads = [g.detach().view(-1) for g in grads if g is not None]

            if not valid_grads:
                self.logger.warning("No valid gradients found for loss computation")
                return 0.0

            # Concatenate all parameter gradients and compute global L2 norm
            concat_grad = torch.cat(valid_grads)
            grad_norm = torch.norm(concat_grad, p=2).item()

            # Check for numerical issues (NaN or Inf)
            if not torch.isfinite(torch.tensor(grad_norm)):
                self.logger.warning(f"Non-finite gradient norm detected: {grad_norm}")
                return 0.0

            return grad_norm

        except Exception as e:
            self.logger.error(f"Error computing gradient norm: {e}")
            return 0.0

    def _update_ema(self, current_value: float, ema_value: Optional[float]) -> float:
        """
        Update EMA (Exponential Moving Average) value.

        Formula: EMA_new = α * EMA_old + (1-α) * current_value

        For the first update, EMA is initialized with the current value.

        Args:
            current_value: Current measured value
            ema_value: Previous EMA value (None for first update)

        Returns:
            Updated EMA value
        """
        if ema_value is None:
            # Initialize EMA with first value
            return current_value
        return self.ema_decay * ema_value + (1 - self.ema_decay) * current_value

    def _update_target_entropy(self, epoch: int) -> None:
        """
        更新目标熵（动态衰减机制）

        Only applies decay when target_entropy is in dict format.
        Float format target_entropy remains fixed (no decay).
        """
        update_entropy_logs = []

        for name in HEAD_NAMES:
            # Only update if decay config exists (dict format)
            if self._target_entropy_decay_configs[name] is not None:
                decay_config = self._target_entropy_decay_configs[name]
                decay_start_epoch = decay_config["decay_start_epoch"]

                # Start decay when epoch >= decay_start_epoch (corrected logic)
                if epoch >= decay_start_epoch:
                    current_target_entropy = self._current_target_entropy[name]
                    end_target_entropy = decay_config["end"]
                    decay_rate = decay_config["decay_rate"]

                    # Apply exponential decay
                    new_target_entropy = max(
                        end_target_entropy, current_target_entropy * decay_rate
                    )

                    # Update current target entropy
                    self._current_target_entropy[name] = new_target_entropy

                    # Log the decay (corrected order: old -> new)
                    update_entropy_logs.append(
                        f"{name}({current_target_entropy:.4f} -> {new_target_entropy:.4f})"
                    )

        if update_entropy_logs:
            self.logger.info(
                f"[Epoch {epoch}] Target entropy decay: {', '.join(update_entropy_logs)}"
            )

    def update_coefficients(
        self,
        epoch: int,
        shared_params: List[nn.Parameter],
        raw_ent_loss_loop_order: torch.Tensor,
        raw_ent_loss_packing_type: torch.Tensor,
        raw_ent_loss_thread_type: torch.Tensor,
        raw_ent_loss_cont: torch.Tensor,
        clip_loss: torch.Tensor,
        current_coefs: List[float],
    ) -> List[float]:
        """
        Update entropy coefficients using dual-mechanism regulation.

        For discrete action heads:
        - Primary regulator: SAC-style temperature auto-adjustment based on target entropy
        - Safety valve: Gradient norm ratio correction (Scheme 3) when ratio is out of safe range

        For continuous action head:
        - Pure gradient norm ratio balancing (Scheme 3)

        This method performs the following steps:
        1. Compute gradient norms for each entropy loss and policy loss
        2. Update EMA values for gradient norms and entropies
        3. For discrete heads: SAC adjustment + safety correction
        4. For continuous head: Pure gradient norm ratio balancing

        Args:
            shared_params: List of shared parameters (from _attention_net)
                          Should be self.policy._attention_net.parameters()
            raw_ent_loss_loop_order: Raw entropy loss for loop order action head
                                    (before multiplying by coefficient)
            raw_ent_loss_packing_type: Raw entropy loss for packing type action head
            raw_ent_loss_thread_type: Raw entropy loss for thread type action head
            raw_ent_loss_cont: Raw entropy loss for continuous action head
            clip_loss: Policy loss (PPO clip loss)
            current_coefs: Current entropy coefficients [
                coef_disc_loop_order,
                coef_disc_packing_type,
                coef_disc_thread_type,
                coef_cont
            ]

        Returns:
            Updated entropy coefficients [new_coef_loop_order, new_coef_packing_type,
                                          new_coef_thread_type, new_coef_cont]
        """
        self._update_counter += 1
        self._update_target_entropy(epoch)

        # Check if we should update based on update frequency
        if self._update_counter % self.update_frequency != 0:
            return current_coefs

        avg_entropy_loop_order: float = raw_ent_loss_loop_order.item()
        avg_entropy_packing_type: float = raw_ent_loss_packing_type.item()
        avg_entropy_thread_type: float = raw_ent_loss_thread_type.item()

        # Step 1: Compute gradient norms for each entropy loss
        grad_norm_ent_loop_order = self.compute_grad_norm(
            raw_ent_loss_loop_order, shared_params
        )
        grad_norm_ent_packing_type = self.compute_grad_norm(
            raw_ent_loss_packing_type, shared_params
        )
        grad_norm_ent_thread_type = self.compute_grad_norm(
            raw_ent_loss_thread_type, shared_params
        )
        grad_norm_ent_cont = self.compute_grad_norm(raw_ent_loss_cont, shared_params)

        # Compute gradient norm for policy loss
        grad_norm_policy = self.compute_grad_norm(clip_loss, shared_params)

        # Store for logging and monitoring
        self._last_grad_norms.ent_loop_order = grad_norm_ent_loop_order
        self._last_grad_norms.ent_packing_type = grad_norm_ent_packing_type
        self._last_grad_norms.ent_thread_type = grad_norm_ent_thread_type
        self._last_grad_norms.ent_cont = grad_norm_ent_cont
        self._last_grad_norms.policy = grad_norm_policy

        # Check for policy gradient norm vanishing - critical safety mechanism
        # When policy gradient disappears, we must reset entropy coefficients to prevent
        # the entropy gradients from dominating and causing strategy collapse
        if grad_norm_policy < self.epsilon:
            self.logger.warning(
                f"[CRITICAL] Policy gradient norm vanished ({grad_norm_policy:.6e}). "
                f"This indicates potential training instability. "
                f"Resetting entropy coefficients to minimum safe values to prevent collapse."
            )

            # Emergency reset: reduce all coefficients to minimum to prevent entropy explosion
            safe_coefs = [
                self._head_configs["loop_order"].coef_min,
                self._head_configs["packing_type"].coef_min,
                self._head_configs["thread_type"].coef_min,
                self._head_configs["continuous"].coef_min,
            ]

            # Log the emergency reset
            self.logger.warning(
                f"Emergency coefficient reset:\n"
                f"  loop_order: {current_coefs[0]:.6f} -> {safe_coefs[0]:.6f}\n"
                f"  packing_type: {current_coefs[1]:.6f} -> {safe_coefs[1]:.6f}\n"
                f"  thread_type: {current_coefs[2]:.6f} -> {safe_coefs[2]:.6f}\n"
                f"  continuous: {current_coefs[3]:.6f} -> {safe_coefs[3]:.6f}"
            )

            return safe_coefs

        # Step 2: Update EMA values for gradient norms
        self._ema_grad_norm_ent_loop_order = self._update_ema(
            grad_norm_ent_loop_order, self._ema_grad_norm_ent_loop_order
        )
        self._ema_grad_norm_ent_packing_type = self._update_ema(
            grad_norm_ent_packing_type, self._ema_grad_norm_ent_packing_type
        )
        self._ema_grad_norm_ent_thread_type = self._update_ema(
            grad_norm_ent_thread_type, self._ema_grad_norm_ent_thread_type
        )
        self._ema_grad_norm_ent_cont = self._update_ema(
            grad_norm_ent_cont, self._ema_grad_norm_ent_cont
        )
        self._ema_grad_norm_policy = self._update_ema(
            grad_norm_policy, self._ema_grad_norm_policy
        )

        # Step 2.5: Update EMA values for entropies (for SAC adjustment)
        if avg_entropy_loop_order > 0:  # Only update if valid entropy provided
            self._ema_entropy_loop_order = self._update_ema(
                avg_entropy_loop_order, self._ema_entropy_loop_order
            )
        if avg_entropy_packing_type > 0:
            self._ema_entropy_packing_type = self._update_ema(
                avg_entropy_packing_type, self._ema_entropy_packing_type
            )
        if avg_entropy_thread_type > 0:
            self._ema_entropy_thread_type = self._update_ema(
                avg_entropy_thread_type, self._ema_entropy_thread_type
            )

        # Step 3: Compute relative strengths (raw, without coefficient weighting)
        # Formula: w_i_raw = g_i / (g_policy + ε)
        w_loop_order_raw = self._ema_grad_norm_ent_loop_order / (
            self._ema_grad_norm_policy + self.epsilon
        )
        w_packing_type_raw = self._ema_grad_norm_ent_packing_type / (
            self._ema_grad_norm_policy + self.epsilon
        )
        w_thread_type_raw = self._ema_grad_norm_ent_thread_type / (
            self._ema_grad_norm_policy + self.epsilon
        )
        w_cont_raw = self._ema_grad_norm_ent_cont / (
            self._ema_grad_norm_policy + self.epsilon
        )

        # Store for logging and monitoring
        self._last_relative_strengths.loop_order = w_loop_order_raw
        self._last_relative_strengths.packing_type = w_packing_type_raw
        self._last_relative_strengths.thread_type = w_thread_type_raw
        self._last_relative_strengths.cont = w_cont_raw

        # Unpack current coefficients
        (
            coef_disc_loop_order,
            coef_disc_packing_type,
            coef_disc_thread_type,
            coef_cont,
        ) = current_coefs

        # Step 4: Update coefficients for each head
        new_coefs = []

        # ==================== Discrete Heads ====================
        # Process each discrete head with dual-mechanism regulation
        discrete_head_data = [
            (
                "loop_order",
                coef_disc_loop_order,
                w_loop_order_raw,
                self._ema_entropy_loop_order,
                self._ema_grad_norm_ent_loop_order,
            ),
            (
                "packing_type",
                coef_disc_packing_type,
                w_packing_type_raw,
                self._ema_entropy_packing_type,
                self._ema_grad_norm_ent_packing_type,
            ),
            (
                "thread_type",
                coef_disc_thread_type,
                w_thread_type_raw,
                self._ema_entropy_thread_type,
                self._ema_grad_norm_ent_thread_type,
            ),
        ]

        for head_data in discrete_head_data:
            head_name = head_data[0]
            coef_old = head_data[1]
            # w_raw = head_data[2]  # Raw gradient ratio (unused in dual-mechanism)
            ema_entropy = head_data[3]
            ema_grad_norm_raw = head_data[4]

            cfg = self._head_configs[head_name]
            coef_new = coef_old  # Initialize with old coefficient

            # --- Mechanism 1: SAC Temperature Auto-Adjustment ---
            if (
                self.use_sac_for_discrete
                and cfg.target_entropy is not None
                and ema_entropy is not None
            ):
                # SAC update for PPO: β_new = β_old - η_SAC * (H_current - H_target)
                # Note: Gap definition is reversed compared to SAC because in PPO:
                # loss = base_loss - coef * entropy (encourages higher entropy when coef increases)
                # So when entropy > target, we need to decrease coef (reduce entropy bonus)
                target_entropy = self._current_target_entropy[head_name]
                entropy_gap = ema_entropy - target_entropy
                coef_sac = coef_old - self.sac_lr * entropy_gap
                # Clip to safe bounds
                coef_sac = max(cfg.coef_min, min(cfg.coef_max, coef_sac))
                coef_new = coef_sac

                self.logger.info(
                    f"SAC update for {head_name}: "
                    f"entropy={ema_entropy:.4f}, target={target_entropy:.4f}, "
                    f"gap={entropy_gap:.4f}, coef={coef_old:.6f}->{coef_sac:.6f}"
                )

            # --- Mechanism 2: Safety Correction (Gradient Norm Ratio) ---
            # Compute current weighted gradient norm ratio: w = β * g_raw / g_policy
            w_current = (
                coef_old
                * ema_grad_norm_raw
                / (self._ema_grad_norm_policy + self.epsilon)
            )

            # Check if safety correction is needed
            safety_triggered = False
            if (
                    (cfg.w_target > 0)
                    and
                    (
                            w_current < cfg.grad_safe_ratio_min
                            or
                            w_current > cfg.grad_safe_ratio_max
                    )
            ):
                safety_triggered = True

                # Compute scale factor to adjust current gradient ratio to target
                scale = cfg.w_target / (w_current + self.epsilon)

                # # Limit single-step change magnitude based on deviation direction
                # if w_current < cfg.grad_safe_ratio_min:
                #     scale = min(scale_unlimited, 5.0)  # Max 5x amplification per step
                # else:  # w_current > cfg.grad_safe_ratio_max
                #     scale = max(scale_unlimited, 0.2)  # Max 80% reduction per step

                beta_suggest = coef_old * scale

                # Smooth correction: blend with main regulator result
                coef_new = (1 - cfg.safe_tau) * coef_new + cfg.safe_tau * beta_suggest

                self.logger.info(
                    f"[Fuse Triggered] Safety correction for {head_name}: "
                    f"w_current={w_current:.6f} outside [{cfg.grad_safe_ratio_min}, {cfg.grad_safe_ratio_max}], "
                    f"scale={scale}, beta_suggest={beta_suggest:.6f}, "
                    f"coef={coef_old:.6f}->{coef_new:.6f}"
                )

            # Apply coefficient bounds
            # coef_new = max(cfg.coef_min, min(cfg.coef_max, coef_new))
            new_coefs.append(coef_new)

            # Log update details
            self.logger.info(
                f"Discrete head {head_name}: {coef_old:.6f} -> {coef_new:.6f} "
                f"(SAC={self.use_sac_for_discrete and cfg.target_entropy is not None}, "
                f"safety={safety_triggered})"
            )

        # ==================== Continuous Head ====================
        # Pure gradient norm ratio balancing (Scheme 3)
        if self.balance_continuous:
            cfg_cont = self._head_configs["continuous"]

            # Step 1: Multiplicative update (original logic)
            new_coef_cont = coef_cont * math.exp(
                cfg_cont.learning_rate * (cfg_cont.w_target - w_cont_raw)
            )
            new_coef_cont = max(
                cfg_cont.coef_min, min(cfg_cont.coef_max, new_coef_cont)
            )

            # Step 2: Safety correction (newly added)
            # Compute current weighted gradient norm ratio: w = β * g_raw / g_policy
            w_current = (
                coef_cont
                * self._ema_grad_norm_ent_cont
                / (self._ema_grad_norm_policy + self.epsilon)
            )

            # Check if safety correction is needed
            if (
                w_current < cfg_cont.grad_safe_ratio_min
                or w_current > cfg_cont.grad_safe_ratio_max
            ):
                # Compute scale factor to adjust current gradient ratio to target
                scale = cfg_cont.w_target / (w_current + self.epsilon)

                # # Limit single-step change magnitude based on deviation direction
                # if w_current < cfg_cont.grad_safe_ratio_min:
                #     scale = min(scale_unlimited, 5.0)  # Max 5x amplification per step
                # else:  # w_current > cfg_cont.grad_safe_ratio_max
                #     scale = max(scale_unlimited, 0.2)  # Max 80% reduction per step

                beta_suggest = coef_cont * scale

                # Smooth correction: blend with main regulator result
                new_coef_cont = (
                    1 - cfg_cont.safe_tau
                ) * new_coef_cont + cfg_cont.safe_tau * beta_suggest

                self.logger.warning(
                    f"[Fuse Triggered] Safety correction for continuous: w_current={w_current:.4f} outside "
                    f"[{cfg_cont.grad_safe_ratio_min}, {cfg_cont.grad_safe_ratio_max}]. "
                    f" scale={scale}, beta_suggest={beta_suggest:.6f}, "
                    f"coef adjusted: {coef_cont:.6f} -> {new_coef_cont:.6f}"
                )
            else:
                self.logger.info(
                    f"Continuous head: {coef_cont:.6f} -> {new_coef_cont:.6f} "
                    f"(w={w_cont_raw:.6f}, target={cfg_cont.w_target:.6f})"
                )
        else:
            new_coef_cont = coef_cont
            self.logger.info(f"Continuous head: fixed at {coef_cont:.6f}")

        new_coefs.append(new_coef_cont)

        # Log overall update
        self.logger.info(
            f"Entropy coefficient update (step {self._update_counter}):\n"
            f"  loop_order: {coef_disc_loop_order:.6f} -> {new_coefs[0]:.6f}\n"
            f"  packing_type: {coef_disc_packing_type:.6f} -> {new_coefs[1]:.6f}\n"
            f"  thread_type: {coef_disc_thread_type:.6f} -> {new_coefs[2]:.6f}\n"
            f"  continuous: {coef_cont:.6f} -> {new_coef_cont:.6f}"
        )

        # Log gradient norms (EMA)
        self.logger.info(
            f"Gradient norms (EMA):\n"
            f"  ent_loop_order: {self._ema_grad_norm_ent_loop_order:.6e}\n"
            f"  ent_packing_type: {self._ema_grad_norm_ent_packing_type:.6e}\n"
            f"  ent_thread_type: {self._ema_grad_norm_ent_thread_type:.6e}\n"
            f"  ent_cont: {self._ema_grad_norm_ent_cont:.6e}\n"
            f"  policy: {self._ema_grad_norm_policy:.6e}"
        )

        # Log entropy values (EMA) for SAC heads
        if self.use_sac_for_discrete:
            entropy_log = "Entropies (EMA):\n"
            if self._ema_entropy_loop_order is not None:
                target = self._current_target_entropy["loop_order"]
                entropy_log += f"  loop_order: {self._ema_entropy_loop_order:.4f}"
                if target is not None:
                    entropy_log += f" (target={target:.4f})"
                entropy_log += "\n"
            if self._ema_entropy_packing_type is not None:
                target = self._current_target_entropy["packing_type"]
                entropy_log += f"  packing_type: {self._ema_entropy_packing_type:.4f}"
                if target is not None:
                    entropy_log += f" (target={target:.4f})"
                entropy_log += "\n"
            if self._ema_entropy_thread_type is not None:
                target = self._current_target_entropy["thread_type"]
                entropy_log += f"  thread_type: {self._ema_entropy_thread_type:.4f}"
                if target is not None:
                    entropy_log += f" (target={target:.4f})"
                entropy_log += "\n"
            self.logger.info(entropy_log)

        return new_coefs

    def get_last_grad_norms(self) -> GradientNormStats:
        """
        Get the last computed gradient norms for logging and monitoring.

        Returns:
            GradientNormStats containing the last computed gradient norms
        """
        return self._last_grad_norms

    def get_last_relative_strengths(self) -> RelativeStrengthStats:
        """
        Get the last computed relative strengths for logging and monitoring.

        Returns:
            RelativeStrengthStats containing the last computed relative strengths
        """
        return self._last_relative_strengths

    def get_head_configs(self) -> Dict[str, HeadConfig]:
        """
        Get the current head configurations.

        Returns:
            Dict[str, HeadConfig]: Configuration for each action head
        """
        return self._head_configs.copy()

    def get_ema_grad_norms(self) -> dict:
        """
        Get the current EMA values of gradient norms.

        Returns:
            Dictionary containing EMA values for each gradient norm
        """
        return {
            "ent_loop_order": self._ema_grad_norm_ent_loop_order,
            "ent_packing_type": self._ema_grad_norm_ent_packing_type,
            "ent_thread_type": self._ema_grad_norm_ent_thread_type,
            "ent_cont": self._ema_grad_norm_ent_cont,
            "policy": self._ema_grad_norm_policy,
        }

    def get_ema_entropies(self) -> dict:
        """
        Get the current EMA values of entropies for SAC adjustment.

        Returns:
            Dictionary containing EMA values for each discrete head entropy
        """
        return {
            "loop_order": self._ema_entropy_loop_order,
            "packing_type": self._ema_entropy_packing_type,
            "thread_type": self._ema_entropy_thread_type,
        }

    def reset(self):
        """
        Reset all EMA values and counters.
        Call this when starting a new training session.
        """
        self._ema_grad_norm_ent_loop_order = None
        self._ema_grad_norm_ent_packing_type = None
        self._ema_grad_norm_ent_thread_type = None
        self._ema_grad_norm_ent_cont = None
        self._ema_grad_norm_policy = None
        self._ema_entropy_loop_order = None
        self._ema_entropy_packing_type = None
        self._ema_entropy_thread_type = None
        self._update_counter = 0
        self._last_grad_norms = GradientNormStats()
        self._last_relative_strengths = RelativeStrengthStats()
        self.logger.info("AdaptiveEntropyBalancer reset")
