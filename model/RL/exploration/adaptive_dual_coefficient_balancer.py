# -*- coding: utf-8 -*-
# Author        : AI Assistant
# Create Time   : 2025
# Contact       :
# File          : adaptive_dual_coefficient_balancer.py
# IDE           : PyCharm
# Description   : Adaptive dual-coefficient entropy balancer with scale normalization
#                 Introduces separate scale (s_h) and beta (β_h) coefficients to decouple
#                 magnitude normalization from ratio balancing

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Action head names constant
HEAD_NAMES = ["loop_order", "packing_type", "thread_type", "continuous"]


@dataclass
class HeadConfig:
    """Configuration for a single action head's dual-coefficient entropy balancing."""

    # Target gradient norm ratio (r_h)
    w_target: float = 0.05

    # ---------- β fine-tuning parameters ----------
    beta_lr: float = 0.2  # Multiplicative update learning rate (η_β)
    beta_init: float = 0.01  # Initial beta coefficient
    beta_min: float = 1e-5  # Minimum beta coefficient
    beta_max: float = 0.2  # Maximum beta coefficient
    # Safety correction (optional)
    safety_factor: float = 5.0  # Correction trigger factor (K)
    safety_tau: float = 0.5  # Correction strength (τ)

    # ---------- s coarse-tuning parameters ----------
    scale_init: float = 1.0  # Initial scale factor
    scale_lr: float = 0.1  # Learning rate for multiplicative mode
    scale_min: float = 1e-6  # Minimum scale factor
    scale_max: float = 1e6  # Maximum scale factor
    scale_update_mode: str = "ema"  # "ema" or "multiplicative"
    scale_ema_decay: float = 0.99  # EMA decay coefficient (α_scale)


# Predefined default configurations for each action head
DEFAULT_HEAD_CONFIGS: Dict[str, HeadConfig] = {
    "loop_order": HeadConfig(
        w_target=0.06,
        beta_lr=0.15,
        beta_init=0.01,
        beta_min=1e-5,
        beta_max=0.12,
        safety_factor=5.0,
        safety_tau=0.5,
        scale_init=1.0,
        scale_lr=0.1,
        scale_min=1e-6,
        scale_max=1e6,
        scale_update_mode="ema",
        scale_ema_decay=0.99,
    ),
    "packing_type": HeadConfig(
        w_target=0.025,
        beta_lr=0.12,
        beta_init=0.01,
        beta_min=1e-5,
        beta_max=0.06,
        safety_factor=5.0,
        safety_tau=0.5,
        scale_init=1.0,
        scale_lr=0.1,
        scale_min=1e-6,
        scale_max=1e6,
        scale_update_mode="ema",
        scale_ema_decay=0.99,
    ),
    "thread_type": HeadConfig(
        w_target=0.0,
        beta_lr=0.2,
        beta_init=0.01,
        beta_min=1e-5,
        beta_max=0.1,
        safety_factor=5.0,
        safety_tau=0.5,
        scale_init=1.0,
        scale_lr=0.1,
        scale_min=1e-6,
        scale_max=1e6,
        scale_update_mode="ema",
        scale_ema_decay=0.99,
    ),
    "continuous": HeadConfig(
        w_target=0.09,
        beta_lr=0.18,
        beta_init=0.001,
        beta_min=1e-6,
        beta_max=0.02,
        safety_factor=5.0,
        safety_tau=0.5,
        scale_init=1.0,
        scale_lr=0.1,
        scale_min=1e-6,
        scale_max=1e6,
        scale_update_mode="ema",
        scale_ema_decay=0.99,
    ),
}


@dataclass
class DualCoefficientStats:
    """Statistics for dual-coefficient balancer"""

    grad_norm_policy: float = 0.0
    grad_norm_ent_loop_order: float = 0.0
    grad_norm_ent_packing_type: float = 0.0
    grad_norm_ent_thread_type: float = 0.0
    grad_norm_ent_continuous: float = 0.0
    do_update: bool = False
    ratio_loop_order: float = 0.0
    ratio_packing_type: float = 0.0
    ratio_thread_type: float = 0.0
    ratio_continuous: float = 0.0
    beta_loop_order: float = 0.0
    beta_packing_type: float = 0.0
    beta_thread_type: float = 0.0
    beta_continuous: float = 0.0
    scale_loop_order: float = 0.0
    scale_packing_type: float = 0.0
    scale_thread_type: float = 0.0
    scale_continuous: float = 0.0
    effective_coef_loop_order: float = 0.0
    effective_coef_packing_type: float = 0.0
    effective_coef_thread_type: float = 0.0
    effective_coef_continuous: float = 0.0
    triggered_safety_correction: Optional[List[str]] = None
    triggered_emergency_reset: bool = False

    def __post_init__(self):
        if self.triggered_safety_correction is None:
            self.triggered_safety_correction = []


class AdaptiveDualCoefficientBalancer:
    """
    Adaptive dual-coefficient entropy balancer with scale normalization.

    This balancer introduces separate scale (s_h) and beta (β_h) coefficients to decouple
    magnitude normalization from ratio balancing, solving the conflict in the single-coefficient
    approach.

    Core Algorithm:
    1. Scale factor (s_h): Automatically normalizes raw entropy gradient magnitude
       - Updated using EMA or multiplicative rule based on ratio of target to current
    2. Beta coefficient (β_h): Precisely adjusts each head's ratio on normalized magnitude
       - Updated using multiplicative rule with gradient ratio feedback

    Key Concepts:
    - Raw entropy loss: H_h
    - Scaled entropy loss: H'_h = s_h · H_h
    - Effective regularizer: β_h · H'_h = (β_h · s_h) · H_h
    - Gradient ratio: w_h = ||∇(β_h · H'_h)|| / (||∇L_policy|| + ε)
    - Target ratio: r_h (user configured)

    Update Rules:
    - β_h: β_h ← β_h · exp(η_β · clamp(r_h - w_h, -δ, δ))
    - s_h (EMA): s_h ← α_scale · s_h + (1 - α_scale) · s_h · (r_h / w_h)
    - s_h (multiplicative): s_h ← s_h · (r_h / w_h)^{η_scale}

    Features:
    - Decoupled magnitude normalization and ratio balancing
    - Separate update frequencies for β and s
    - Safety mechanisms: coefficient bounds, safety correction, emergency reset
    - Comprehensive statistics for monitoring
    """

    def __init__(
        self,
        head_configs: Union[
            None,
            float,
            HeadConfig,
            Dict[str, Union[HeadConfig, Dict[str, float], float]],
        ] = None,
        ema_decay: float = 0.99,  # EMA decay for gradient norms (α)
        warmup_steps: int = 100,  # Number of steps before starting coefficient updates
        max_change_ratio: float = 0.2,  # Maximum single-step change rate for β (δ)
        update_frequency: int = 1,  # β update frequency
        scale_update_frequency: int = 5,  # s update frequency (slower than β)
        scale_align_mode: str = "magnitude",  # Global scale alignment mode: 'magnitude'(new decoupled) or 'legacy'(old ratio-based)
        use_safety_correction: bool = True,  # Enable safety correction for β
        policy_grad_eps: float = 1e-10,  # Policy gradient vanishing threshold
        epsilon: float = 1e-8,  # Small constant for numerical stability
    ):
        """
        Initialize the dual-coefficient entropy balancer.

        Args:
            head_configs: Configuration for action heads. Can be:
                - None: Use DEFAULT_HEAD_CONFIGS
                - float: Set w_target for all heads to this value
                - HeadConfig: All heads use this same configuration
                - Dict[str, ...]: Per-head configuration, where each value can be:
                    - HeadConfig: Complete configuration for that head
                    - Dict[str, float]: Partial configuration
                    - float: Only set w_target for that head

            ema_decay: EMA decay factor for gradient norm smoothing (α)
                      Recommended: 0.99 (higher = more smoothing)

            warmup_steps: Number of steps to wait before starting coefficient updates
                         During warmup, only EMA values are updated

            max_change_ratio: Maximum single-step change rate for β (δ)
                            Clamps the ratio difference to prevent sudden changes

            update_frequency: Update β coefficients every N calls
                            Higher values reduce computational overhead

            scale_update_frequency: Update s coefficients every N calls
                                   Recommended to be slower than β updates (e.g., 5)

            use_safety_correction: Enable safety correction mechanism for β
                                  When w_h deviates > K times from r_h, apply correction

            policy_grad_eps: Threshold for detecting policy gradient vanishing
                           When policy gradient < this value, trigger emergency reset

            epsilon: Small constant for numerical stability in divisions
        """
        self.logger = logging.getLogger("AdaptiveDualCoefficientBalancer")

        # Store global scale_align_mode first (needed for _parse_head_configs)
        self.scale_align_mode = scale_align_mode

        # Parse and store head configurations
        self._head_configs: Dict[str, HeadConfig] = self._parse_head_configs(
            head_configs
        )

        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.max_change_ratio = max_change_ratio
        self.update_frequency = update_frequency
        self.scale_update_frequency = scale_update_frequency
        self.use_safety_correction = use_safety_correction
        self.policy_grad_eps = policy_grad_eps
        self.epsilon = epsilon

        # EMA values for gradient norms (initialize as None, set on first observation)
        self._ema_grad_norm_policy: Optional[float] = None
        self._ema_grad_norm_ent_loop_order: Optional[float] = None
        self._ema_grad_norm_ent_packing_type: Optional[float] = None
        self._ema_grad_norm_ent_thread_type: Optional[float] = None
        self._ema_grad_norm_ent_cont: Optional[float] = None

        # Current β coefficients (initialized with beta_init for warmup phase)
        self._current_betas: Dict[str, float] = {
            name: config.beta_init for name, config in self._head_configs.items()
        }

        # Current s scale factors (initialized with scale_init)
        self._current_scales: Dict[str, float] = {
            name: config.scale_init for name, config in self._head_configs.items()
        }

        # Step counter
        self._step_counter = 0

        # Statistics for logging
        self._last_stats = DualCoefficientStats()

        # Log configuration
        self._log_configuration()

    def _parse_head_configs(
        self,
        configs: Union[
            None,
            float,
            HeadConfig,
            Dict[str, Union[HeadConfig, Dict[str, float], float]],
        ]
    ) -> Dict[str, HeadConfig]:
        """
        Parse head configurations from various input formats.

        Args:
            configs: Input configuration in various formats

        Returns:
            Dict[str, HeadConfig]: Parsed configuration for each head
        """
        # Case 1: None - use defaults
        if configs is None:
            return DEFAULT_HEAD_CONFIGS.copy()

        # Case 2: float/int - set w_target for all heads
        if isinstance(configs, (int, float)):
            return {
                head: HeadConfig(
                    w_target=float(configs),
                    beta_lr=DEFAULT_HEAD_CONFIGS[head].beta_lr,
                    beta_init=DEFAULT_HEAD_CONFIGS[head].beta_init,
                    beta_min=DEFAULT_HEAD_CONFIGS[head].beta_min,
                    beta_max=DEFAULT_HEAD_CONFIGS[head].beta_max,
                    safety_factor=DEFAULT_HEAD_CONFIGS[head].safety_factor,
                    safety_tau=DEFAULT_HEAD_CONFIGS[head].safety_tau,
                    scale_init=DEFAULT_HEAD_CONFIGS[head].scale_init,
                    scale_lr=DEFAULT_HEAD_CONFIGS[head].scale_lr,
                    scale_min=DEFAULT_HEAD_CONFIGS[head].scale_min,
                    scale_max=DEFAULT_HEAD_CONFIGS[head].scale_max,
                    scale_update_mode=DEFAULT_HEAD_CONFIGS[head].scale_update_mode,
                    scale_ema_decay=DEFAULT_HEAD_CONFIGS[head].scale_ema_decay,
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
                            "beta_lr": cfg.get("beta_lr", default.beta_lr),
                            "beta_init": cfg.get("beta_init", default.beta_init),
                            "beta_min": cfg.get("beta_min", default.beta_min),
                            "beta_max": cfg.get("beta_max", default.beta_max),
                            "safety_factor": cfg.get(
                                "safety_factor", default.safety_factor
                            ),
                            "safety_tau": cfg.get("safety_tau", default.safety_tau),
                            "scale_init": cfg.get("scale_init", default.scale_init),
                            "scale_lr": cfg.get("scale_lr", default.scale_lr),
                            "scale_min": cfg.get("scale_min", default.scale_min),
                            "scale_max": cfg.get("scale_max", default.scale_max),
                            "scale_update_mode": cfg.get(
                                "scale_update_mode", default.scale_update_mode
                            ),
                            "scale_ema_decay": cfg.get(
                                "scale_ema_decay", default.scale_ema_decay
                            ),
                        }
                        parsed[head] = HeadConfig(**merged)

                    # Sub-case 4c: float - only set w_target
                    elif isinstance(cfg, (int, float)):
                        default = DEFAULT_HEAD_CONFIGS[head]
                        parsed[head] = HeadConfig(
                            w_target=float(cfg),
                            beta_lr=default.beta_lr,
                            beta_init=default.beta_init,
                            beta_min=default.beta_min,
                            beta_max=default.beta_max,
                            safety_factor=default.safety_factor,
                            safety_tau=default.safety_tau,
                            scale_init=default.scale_init,
                            scale_lr=default.scale_lr,
                            scale_min=default.scale_min,
                            scale_max=default.scale_max,
                            scale_update_mode=default.scale_update_mode,
                            scale_ema_decay=default.scale_ema_decay,
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
        """Log the configuration for each head."""
        logs = []
        logs.append("AdaptiveDualCoefficientBalancer Configuration:")
        logs.append(f"  EMA decay: {self.ema_decay}")
        logs.append(f"  Warmup steps: {self.warmup_steps}")
        logs.append(f"  Max change ratio for β (δ): {self.max_change_ratio}")
        logs.append(f"  β update frequency: {self.update_frequency}")
        logs.append(f"  s update frequency: {self.scale_update_frequency}")
        logs.append(f"  Global scale align mode: {self.scale_align_mode}")
        logs.append(f"  Use safety correction: {self.use_safety_correction}")
        logs.append(f"  Policy gradient eps: {self.policy_grad_eps}")
        logs.append("")

        for head, config in self._head_configs.items():
            logs.append(f"  Head '{head}':")
            logs.append(f"    Target ratio (r_h): {config.w_target}")
            logs.append(f"    β learning rate (η_β): {config.beta_lr}")
            logs.append(f"    β initial: {config.beta_init}")
            logs.append(f"    β range: [{config.beta_min}, {config.beta_max}]")
            logs.append(f"    Safety factor (K): {config.safety_factor}")
            logs.append(f"    Safety tau (τ): {config.safety_tau}")
            logs.append(f"    s initial: {config.scale_init}")
            logs.append(f"    s learning rate: {config.scale_lr}")
            logs.append(f"    s range: [{config.scale_min}, {config.scale_max}]")
            logs.append(f"    s update mode: {config.scale_update_mode}")
            logs.append(f"    s EMA decay (α_scale): {config.scale_ema_decay}")

        self.logger.info("\n".join(logs))

    def compute_grad_norm(
        self, loss: torch.Tensor, params: List[nn.Parameter], retain_graph: bool = True
    ) -> float:
        """
        Compute the L2 norm of gradients of loss w.r.t. parameters.

        Args:
            loss: Loss tensor
            params: List of parameters
            retain_graph: Whether to retain computation graph

        Returns:
            Gradient norm as a float
        """
        # Filter out parameters that don't require gradient
        params = [p for p in params if p.requires_grad]

        if not params:
            return 0.0

        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )

        # Compute L2 norm
        grad_norm = 0.0
        for grad in grads:
            if grad is not None:
                grad_norm += grad.detach().pow(2).sum().item()

        return math.sqrt(grad_norm)

    def _update_ema(self, current: float, ema_value: Optional[float]) -> float:
        """
        Update EMA value.

        Args:
            current: Current observed value
            ema_value: Current EMA value (None if not initialized)

        Returns:
            Updated EMA value
        """
        if ema_value is None:
            # First observation: initialize EMA with current value
            return current
        else:
            return self.ema_decay * ema_value + (1 - self.ema_decay) * current

    def _update_beta(
        self,
        name: str,
        config: HeadConfig,
        beta: float,
        w_current: float,
        r_target: float,
    ) -> tuple[float, bool]:
        """
        Update β coefficient using multiplicative rule.

        Args:
            config: Head configuration
            beta: Current β value
            w_current: Current gradient ratio
            r_target: Target gradient ratio

        Returns:
            Tuple of (updated β, safety correction triggered)
        """
        # Multiplicative update: β_new = β_old * exp(η * clamp(r - w, -δ, δ))
        ratio_diff = r_target - w_current
        clamped_diff = max(
            -self.max_change_ratio, min(self.max_change_ratio, ratio_diff)
        )
        update_factor = math.exp(config.beta_lr * clamped_diff)
        new_beta = beta * update_factor

        safety_triggered = False

        # Optional safety correction
        if self.use_safety_correction:
            # Check if ratio deviates too much from target
            # Trigger correction if w_h > K * r_h or w_h < r_h / K
            upper_threshold = config.safety_factor * r_target
            lower_threshold = r_target / config.safety_factor

            if w_current > upper_threshold or w_current < lower_threshold:
                # Safety correction: β_new = β_old * (r / w)^τ
                # Ensure w_current is not too small to avoid division issues
                w_safe = max(w_current, self.epsilon)
                correction_factor = (r_target / w_safe) ** config.safety_tau
                new_beta = new_beta * correction_factor
                safety_triggered = True
                self.logger.info(
                    f"[Safety Correction] Head '{name}': ratio {w_current} "
                    f"outside safe range [{lower_threshold:.6f}, {upper_threshold:.6f}]. "
                    f"Applied correction factor {correction_factor} and get new_beta {new_beta}."
                )

        # Clip to [beta_min, beta_max]
        new_beta = max(config.beta_min, min(config.beta_max, new_beta))

        return new_beta, safety_triggered

    def _update_scale(
        self,
        name: str,
        config: HeadConfig,
        scale: float,
        w_current: float,
        r_target: float,
        ema_policy: Optional[float] = None,
        ema_ent: Optional[float] = None,
    ) -> float:
        """
        Update s scale factor using EMA or multiplicative rule.

        Supports two modes:
        - "magnitude": New decoupled mode using ideal_scale = g_policy / g_ent
        - "legacy": Old ratio-based mode using target_scale = s * (r / w)

        Args:
            name: Head name (for logging)
            config: Head configuration
            scale: Current s value
            w_current: Current gradient ratio (used in legacy mode)
            r_target: Target gradient ratio (used in legacy mode)
            ema_policy: EMA of policy gradient norm (used in magnitude mode)
            ema_ent: EMA of entropy gradient norm (used in magnitude mode)

        Returns:
            Updated s value
        """
        if self.scale_align_mode == "magnitude":
            # ---------- New Decoupled Mode: Magnitude Alignment ----------
            # Calculate ideal scale: s'_h = g_policy / g_ent
            if ema_ent is None or ema_policy is None:
                # Fallback to current scale if EMA values not provided
                self.logger.warning(
                    f"[Scale Update] Head '{name}': EMA values not provided for magnitude mode, "
                    f"keeping current scale {scale:.6e}"
                )
                return scale

            if ema_ent > self.epsilon:
                ideal_scale = ema_policy / ema_ent
            else:
                # If entropy gradient is too small, keep current scale
                ideal_scale = scale

            # Apply smooth update based on scale_update_mode
            if config.scale_update_mode == "ema":
                # EMA update: s_h ← α_scale · s_h + (1 - α_scale) · ideal_scale
                new_scale = (
                    config.scale_ema_decay * scale
                    + (1 - config.scale_ema_decay) * ideal_scale
                )
            elif config.scale_update_mode == "multiplicative":
                # Multiplicative update: s_h ← s_h · (ideal_scale / s_h)^{η_scale}
                if scale > self.epsilon:
                    update_factor = (ideal_scale / scale) ** config.scale_lr
                    new_scale = scale * update_factor
                else:
                    new_scale = ideal_scale
            else:
                raise ValueError(
                    f"Invalid scale_update_mode: {config.scale_update_mode}. "
                    f"Must be 'ema' or 'multiplicative'."
                )

            self.logger.info(
                f"[Scale Update - Magnitude] Head '{name}': "
                f"ema_policy={ema_policy:.6e}, ema_ent={ema_ent:.6e}, "
                f"ideal_scale={ideal_scale:.6e}, new_scale={new_scale:.6e}"
            )

        elif self.scale_align_mode == "legacy":
            # ---------- Old Coupled Mode: Ratio-Based Update ----------
            # Compute target scale: s_target = s_h * (r_h / w_h)
            w_safe = max(w_current, self.epsilon)
            target_scale = scale * (r_target / w_safe)

            if config.scale_update_mode == "ema":
                # EMA update: s_h ← α_scale · s_h + (1 - α_scale) · target_scale
                new_scale = (
                    config.scale_ema_decay * scale
                    + (1 - config.scale_ema_decay) * target_scale
                )
            elif config.scale_update_mode == "multiplicative":
                # Multiplicative update: s_h ← s_h · (r_h / w_h)^{η_scale}
                update_factor = (r_target / w_safe) ** config.scale_lr
                new_scale = scale * update_factor
            else:
                raise ValueError(
                    f"Invalid scale_update_mode: {config.scale_update_mode}. "
                    f"Must be 'ema' or 'multiplicative'."
                )

            self.logger.info(
                f"[Scale Update - Legacy] Head '{name}': "
                f"ratio={w_safe:.6e}, r_target={r_target:.6e}, new_scale={new_scale:.6e}"
            )

        else:
            raise ValueError(
                f"Invalid scale_align_mode: {self.scale_align_mode}. "
                f"Must be 'magnitude' or 'legacy'."
            )

        # Clip to [scale_min, scale_max]
        new_scale = max(config.scale_min, min(config.scale_max, new_scale))

        return new_scale

    def update(
        self,
        shared_params: List[nn.Parameter],
        policy_loss: torch.Tensor,
        raw_ent_losses: Dict[str, torch.Tensor],
        current_step: int,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Update dual coefficients (β and s) using gradient norm ratio balancing.

        This method performs the following steps:
        1. If in warmup phase, only update EMA values
        2. Compute gradient norms for policy loss and each entropy loss
        3. Update EMA values for gradient norms
        4. Compute current gradient ratios (w_h)
        5. Update β coefficients (multiplicative with safety correction)
        6. Update s scale factors (EMA or multiplicative, slower frequency)
        7. Check for policy gradient vanishing and emergency reset

        Args:
            shared_params: List of shared parameters (e.g., from _attention_net)
            policy_loss: Policy loss (PPO clip loss, without entropy term)
            raw_ent_losses: Dictionary of raw entropy losses for each head
                           {"loop_order": loss, "packing_type": loss,
                            "thread_type": loss, "continuous": loss}
            current_step: Current training step

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: (beta_coefs, scale_factors)
            Effective coefficient for head h = beta_coefs[h] * scale_factors[h]
        """
        self._step_counter += 1

        # Step 1: Check warmup phase
        is_warmup = current_step < self.warmup_steps

        # Step 2: Compute gradient norms for raw entropy losses
        grad_norm_ent_loop_order = self.compute_grad_norm(
            raw_ent_losses["loop_order"], shared_params
        )
        grad_norm_ent_packing_type = self.compute_grad_norm(
            raw_ent_losses["packing_type"], shared_params
        )
        grad_norm_ent_thread_type = self.compute_grad_norm(
            raw_ent_losses["thread_type"], shared_params
        )
        grad_norm_ent_cont = self.compute_grad_norm(
            raw_ent_losses["continuous"], shared_params
        )
        grad_norm_policy = self.compute_grad_norm(
            policy_loss, shared_params, retain_graph=False
        )

        # Step 3: Update EMA values for gradient norms
        self._ema_grad_norm_policy = self._update_ema(
            grad_norm_policy, self._ema_grad_norm_policy
        )
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

        # Store statistics
        self._last_stats.grad_norm_policy = grad_norm_policy
        self._last_stats.grad_norm_ent_loop_order = grad_norm_ent_loop_order
        self._last_stats.grad_norm_ent_packing_type = grad_norm_ent_packing_type
        self._last_stats.grad_norm_ent_thread_type = grad_norm_ent_thread_type
        self._last_stats.grad_norm_ent_continuous = grad_norm_ent_cont
        self._last_stats.triggered_safety_correction = []
        self._last_stats.triggered_emergency_reset = False
        self._last_stats.do_update = False

        if is_warmup:
            return self._current_betas.copy(), self._current_scales.copy()
        elif self._step_counter % self.update_frequency != 0:
            return self._current_betas.copy(), self._current_scales.copy()
        else:
            self._last_stats.do_update = True

        # Step 4: Check if EMA is initialized
        if self._ema_grad_norm_policy is None:
            self.logger.info(
                f"[Step {current_step}] Coefs will keep the same due to EMA not initialized."
            )
            return self._current_betas.copy(), self._current_scales.copy()

        # Step 5: Check for policy gradient vanishing - emergency reset
        if self._ema_grad_norm_policy < self.policy_grad_eps:
            self.logger.warning(
                f"[EMERGENCY RESET] Policy gradient norm vanished "
                f"(EMA: {self._ema_grad_norm_policy:.6e} < {self.policy_grad_eps:.6e}). "
                f"Resetting all β to minimum values. Keeping s unchanged."
            )
            for name in HEAD_NAMES:
                self._current_betas[name] = self._head_configs[name].beta_min
            self._last_stats.triggered_emergency_reset = True
            return self._current_betas.copy(), self._current_scales.copy()

        # Step 6: Prepare EMA policy value (used in both modes)
        ema_policy = self._ema_grad_norm_policy + self.epsilon

        # Step 7: Two-stage decoupled update for coefficients
        # New logic: For "magnitude" mode, update scale first, then compute ratio and update beta
        # Old logic: For "legacy" mode, compute ratio first, then update beta and scale

        updated_betas = {}
        updated_scales = {}
        ratios = {}

        update_scale = self._step_counter % self.scale_update_frequency == 0

        # Prepare EMA entropy gradient norms for each head
        ema_ents = {
            "loop_order": self._ema_grad_norm_ent_loop_order,
            "packing_type": self._ema_grad_norm_ent_packing_type,
            "thread_type": self._ema_grad_norm_ent_thread_type,
            "continuous": self._ema_grad_norm_ent_cont,
        }

        for head in HEAD_NAMES:
            config = self._head_configs[head]
            current_beta = self._current_betas[head]
            current_scale = self._current_scales[head]
            r_target = config.w_target

            if r_target == 0:  # Skip heads with zero target
                updated_betas[head] = 0
                updated_scales[head] = 0
                ratios[head] = 0
                continue

            if self.scale_align_mode == "magnitude":
                # ---------- New Decoupled Mode: Magnitude Alignment ----------
                # Stage 1: Update scale first using magnitude alignment
                if update_scale:
                    ema_ent = ema_ents[head] or 0.0
                    new_scale = self._update_scale(
                        head,
                        config,
                        current_scale,
                        w_current=0.0,  # Not used in magnitude mode
                        r_target=r_target,  # Not used in magnitude mode
                        ema_policy=self._ema_grad_norm_policy,
                        ema_ent=ema_ent,
                    )
                else:
                    new_scale = current_scale

                # Stage 2: Compute ratio using updated scale
                scaled_ent_grad = (ema_ents[head] or 0.0) * new_scale
                w_current = current_beta * scaled_ent_grad / ema_policy

                # Stage 3: Update beta using new ratio
                new_beta, safety_triggered = self._update_beta(
                    head, config, current_beta, w_current, r_target
                )

            else:  # "legacy" mode
                # ---------- Old Coupled Mode: Ratio-Based Update ----------
                # Compute ratio using current scale
                scaled_ent_grad = (ema_ents[head] or 0.0) * current_scale
                w_current = current_beta * scaled_ent_grad / ema_policy

                # Update beta
                new_beta, safety_triggered = self._update_beta(
                    head, config, current_beta, w_current, r_target
                )

                # Update scale (if it's time)
                if update_scale:
                    new_scale = self._update_scale(
                        head,
                        config,
                        current_scale,
                        w_current,
                        r_target,
                        ema_policy=None,  # Not used in legacy mode
                        ema_ent=None,
                    )
                else:
                    new_scale = current_scale

            updated_betas[head] = new_beta
            updated_scales[head] = new_scale
            ratios[head] = w_current

            if safety_triggered:
                self._last_stats.triggered_safety_correction.append(head)

        # Store ratios in stats
        self._last_stats.ratio_loop_order = ratios["loop_order"]
        self._last_stats.ratio_packing_type = ratios["packing_type"]
        self._last_stats.ratio_thread_type = ratios["thread_type"]
        self._last_stats.ratio_continuous = ratios["continuous"]

        # Update current coefficients
        self._current_betas = updated_betas
        self._current_scales = updated_scales

        # Store coefficient values in stats
        self._last_stats.beta_loop_order = updated_betas["loop_order"]
        self._last_stats.beta_packing_type = updated_betas["packing_type"]
        self._last_stats.beta_thread_type = updated_betas["thread_type"]
        self._last_stats.beta_continuous = updated_betas["continuous"]

        self._last_stats.scale_loop_order = updated_scales["loop_order"]
        self._last_stats.scale_packing_type = updated_scales["packing_type"]
        self._last_stats.scale_thread_type = updated_scales["thread_type"]
        self._last_stats.scale_continuous = updated_scales["continuous"]

        # Store effective coefficients (beta * scale) for logging
        # Note: This is for logging only, not for gradient computation
        self._last_stats.effective_coef_loop_order = (
            updated_betas["loop_order"] * updated_scales["loop_order"]
        )
        self._last_stats.effective_coef_packing_type = (
            updated_betas["packing_type"] * updated_scales["packing_type"]
        )
        self._last_stats.effective_coef_thread_type = (
            updated_betas["thread_type"] * updated_scales["thread_type"]
        )
        self._last_stats.effective_coef_continuous = (
            updated_betas["continuous"] * updated_scales["continuous"]
        )

        # Log update information
        self._log_update_info(current_step, ratios, updated_betas, updated_scales)

        return updated_betas, updated_scales

    def _log_update_info(
        self,
        current_step: int,
        ratios: Dict[str, float],
        updated_betas: Dict[str, float],
        updated_scales: Dict[str, float],
    ):
        """Log detailed update information."""
        log_msg = f"[Step {current_step}] Dual-coefficient balancer update:\n"
        log_msg += "  Gradient norms (EMA):\n"
        log_msg += f"    Policy: {self._ema_grad_norm_policy}\n"
        log_msg += f"    loop_order: {self._ema_grad_norm_ent_loop_order}\n"
        log_msg += f"    packing_type: {self._ema_grad_norm_ent_packing_type}\n"
        log_msg += f"    thread_type: {self._ema_grad_norm_ent_thread_type}\n"
        log_msg += f"    continuous: {self._ema_grad_norm_ent_cont}\n"
        log_msg += "  Gradient ratios (w_h):\n"

        for head in HEAD_NAMES:
            r_target = self._head_configs[head].w_target
            w_current = ratios[head]
            log_msg += f"    {head}: {w_current} (target: {r_target})\n"

        log_msg += "  β coefficients:\n"
        for head in HEAD_NAMES:
            log_msg += f"    {head}: {updated_betas[head]}\n"

        log_msg += "  s scale factors:\n"
        for head in HEAD_NAMES:
            log_msg += f"    {head}: {updated_scales[head]}\n"

        log_msg += "  Effective coefficients (β · s):\n"
        for head in HEAD_NAMES:
            effective = updated_betas[head] * updated_scales[head]
            log_msg += f"    {head}: {effective}\n"

        if self._last_stats.triggered_safety_correction:
            log_msg += f"  Safety correction triggered for: {self._last_stats.triggered_safety_correction}\n"

        if self._last_stats.triggered_emergency_reset:
            log_msg += "  EMERGENCY RESET triggered!\n"

        self.logger.info(log_msg)

    def get_last_stats(self) -> DualCoefficientStats:
        """
        Get statistics from the last update.

        Returns:
            DualCoefficientStats: Statistics including gradient norms, ratios, and coefficients
        """
        return self._last_stats

    def get_head_configs(self) -> Dict[str, HeadConfig]:
        """
        Get head configurations.

        Returns:
            Dict[str, HeadConfig]: Configuration for each head
        """
        return self._head_configs.copy()

    def get_ema_grad_norms(self) -> Dict[str, Optional[float]]:
        """
        Get EMA gradient norms.

        Returns:
            Dict with EMA gradient norms for policy and each entropy head
        """
        return {
            "policy": self._ema_grad_norm_policy,
            "loop_order": self._ema_grad_norm_ent_loop_order,
            "packing_type": self._ema_grad_norm_ent_packing_type,
            "thread_type": self._ema_grad_norm_ent_thread_type,
            "continuous": self._ema_grad_norm_ent_cont,
        }

    def get_current_coefficients(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Get current β and s coefficients.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: (beta_coefs, scale_factors)
        """
        return self._current_betas.copy(), self._current_scales.copy()

    def reset(self):
        """Reset the balancer to initial state."""
        self._ema_grad_norm_policy = None
        self._ema_grad_norm_ent_loop_order = None
        self._ema_grad_norm_ent_packing_type = None
        self._ema_grad_norm_ent_thread_type = None
        self._ema_grad_norm_ent_cont = None

        self._current_betas = {
            name: config.beta_init for name, config in self._head_configs.items()
        }

        self._current_scales = {
            name: config.scale_init for name, config in self._head_configs.items()
        }

        self._step_counter = 0
        self._last_stats = DualCoefficientStats()

        self.logger.info("AdaptiveDualCoefficientBalancer reset to initial state.")
