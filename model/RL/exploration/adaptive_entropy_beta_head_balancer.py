# -*- coding: utf-8 -*-
# Author        : AI Assistant
# Create Time   : 2025
# Contact       :
# File          : adaptive_entropy_beta_head_balancer.py
# IDE           : PyCharm
# Description   : Adaptive entropy coefficient balancer using pure gradient norm ratio balancing
#                 Unified approach for both discrete and continuous action heads

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

# Action head names constant
HEAD_NAMES = ["loop_order", "packing_type", "thread_type", "continuous"]


@dataclass
class BetaHeadConfig:
    """Configuration for a single action head's entropy coefficient balancing."""

    w_target: float = 0.05  # Target gradient norm ratio (r_h)
    learning_rate: float = 0.2  # Learning rate for multiplicative update (η)
    coef_init: float = 0.01  # Initial entropy coefficient for warmup phase
    coef_min: float = 1e-5  # Minimum entropy coefficient (β_min)
    coef_max: float = 0.2  # Maximum entropy coefficient (β_max)
    safety_factor: float = (
        5.0  # Safety correction factor (K), triggers when ratio deviates > K times
    )
    safety_tau: float = 0.5  # Safety correction smoothing coefficient (τ)


# Predefined default configurations for each action head
DEFAULT_BETA_HEAD_CONFIGS: Dict[str, BetaHeadConfig] = {
    "loop_order": BetaHeadConfig(
        w_target=0.06,
        learning_rate=0.20,
        coef_init=0.01,
        coef_min=1e-5,
        coef_max=0.2,
        safety_factor=5.0,
        safety_tau=0.5,
    ),
    "packing_type": BetaHeadConfig(
        w_target=0.06,
        learning_rate=0.25,
        coef_init=0.01,
        coef_min=1e-5,
        coef_max=0.08,
        safety_factor=5.0,
        safety_tau=0.5,
    ),
    "thread_type": BetaHeadConfig(
        w_target=0.06,
        learning_rate=0.20,
        coef_init=0.01,
        coef_min=1e-5,
        coef_max=0.1,
        safety_factor=5.0,
        safety_tau=0.5,
    ),
    "continuous": BetaHeadConfig(
        w_target=0.025,
        learning_rate=0.10,
        coef_init=0.001,
        coef_min=1e-6,
        coef_max=0.005,
        safety_factor=5.0,
        safety_tau=0.5,
    ),
}


@dataclass
class BetaHeadStats:
    """Statistics for beta head balancer"""

    grad_norm_policy: float = 0.0
    grad_norm_loop_order: float = 0.0
    grad_norm_packing_type: float = 0.0
    grad_norm_thread_type: float = 0.0
    grad_norm_continuous: float = 0.0
    ratio_loop_order: float = 0.0
    ratio_packing_type: float = 0.0
    ratio_thread_type: float = 0.0
    ratio_continuous: float = 0.0
    coef_loop_order: float = 0.0
    coef_packing_type: float = 0.0
    coef_thread_type: float = 0.0
    coef_continuous: float = 0.0
    triggered_safety_correction: Optional[List[str]] = None
    triggered_emergency_reset: bool = False

    def __post_init__(self):
        if self.triggered_safety_correction is None:
            self.triggered_safety_correction = []


class AdaptiveEntropyBetaHeadBalancer:
    """
    Adaptive entropy coefficient balancer using pure gradient norm ratio balancing.

    This balancer implements a unified approach for both discrete and continuous action heads,
    completely abandoning SAC-style target entropy adjustment. Instead, it uses gradient
    norm ratio balancing (Beta-Head logic) for all heads.

    Core Algorithm:
    1. Compute gradient norms of each entropy loss and policy loss w.r.t. shared parameters
    2. Use EMA to smooth gradient norms, reduce noise
    3. Compute current gradient ratio: w_h = g_ent,h / (g_policy + ε)
    4. Apply multiplicative update: β_h ← β_h · exp(η_h · clamp(r_h - w_h, -δ, δ))
    5. Optional safety correction when w_h deviates too much from r_h
    6. Emergency reset when policy gradient vanishes

    Key Features:
    - Unified logic for discrete and continuous heads
    - Simpler hyperparameters (only w_target and learning_rate needed)
    - Direct response to policy gradient changes
    - EMA smoothing for stability
    - Warmup mechanism
    - Safety mechanisms: coefficient bounds, safety correction, emergency reset
    """

    def __init__(
        self,
        head_configs: Union[
            None,
            float,
            BetaHeadConfig,
            Dict[str, Union[BetaHeadConfig, Dict[str, float], float]],
        ] = None,
        ema_decay: float = 0.99,  # EMA decay factor (α), recommended 0.99
        warmup_steps: int = 100,  # Number of steps before starting coefficient updates
        max_change_ratio: float = 0.2,  # Maximum single-step change rate (δ)
        update_frequency: int = 1,  # Update coefficients every N calls
        use_safety_correction: bool = True,  # Enable safety correction
        safety_factor: float = 5.0,  # Global safety correction factor (K)
        safety_tau: float = 0.5,  # Global safety correction smoothing (τ)
        policy_grad_eps: float = 1e-10,  # Policy gradient vanishing threshold
        epsilon: float = 1e-8,  # Small constant for numerical stability
    ):
        """
        Initialize the beta-head entropy balancer.

        Args:
            head_configs: Configuration for action heads. Can be:
                - None: Use DEFAULT_BETA_HEAD_CONFIGS
                - float: Set w_target for all heads to this value
                - BetaHeadConfig: All heads use this same configuration
                - Dict[str, ...]: Per-head configuration, where each value can be:
                    - BetaHeadConfig: Complete configuration for that head
                    - Dict[str, float]: Partial configuration
                    - float: Only set w_target for that head

            ema_decay: EMA decay factor for gradient norm smoothing (α)
                      Recommended: 0.99 (higher = more smoothing)

            warmup_steps: Number of steps to wait before starting coefficient updates
                         During warmup, only EMA values are updated

            max_change_ratio: Maximum single-step change rate (δ)
                            Clamps the ratio difference to prevent sudden changes

            update_frequency: Update coefficients every N calls
                            Higher values reduce computational overhead

            use_safety_correction: Enable safety correction mechanism
                                  When w_h deviates > K times from r_h, apply correction

            safety_factor: Global safety correction factor (K)
                          Per-head safety_factor can override this

            safety_tau: Global safety correction smoothing coefficient (τ)
                       Per-head safety_tau can override this

            policy_grad_eps: Threshold for detecting policy gradient vanishing
                           When policy gradient < this value, trigger emergency reset

            epsilon: Small constant for numerical stability in divisions
        """
        self.logger = logging.getLogger("AdaptiveEntropyBetaHeadBalancer")

        # Parse and store head configurations
        self._head_configs: Dict[str, BetaHeadConfig] = self._parse_head_configs(
            head_configs
        )

        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.max_change_ratio = max_change_ratio
        self.update_frequency = update_frequency
        self.use_safety_correction = use_safety_correction
        self.safety_factor = safety_factor
        self.safety_tau = safety_tau
        self.policy_grad_eps = policy_grad_eps
        self.epsilon = epsilon

        # EMA values for gradient norms (initialize as None, set on first observation)
        self._ema_grad_norm_policy: Optional[float] = None
        self._ema_grad_norm_ent_loop_order: Optional[float] = None
        self._ema_grad_norm_ent_packing_type: Optional[float] = None
        self._ema_grad_norm_ent_thread_type: Optional[float] = None
        self._ema_grad_norm_ent_cont: Optional[float] = None

        # Current coefficient values (initialized with coef_init for warmup phase)
        self._current_coefs: Dict[str, float] = {
            name: config.coef_init for name, config in self._head_configs.items()
        }

        # Step counter
        self._step_counter = 0

        # Statistics for logging
        self._last_stats = BetaHeadStats()

        # Log configuration
        self._log_configuration()

    def _parse_head_configs(
        self,
        configs: Union[
            None,
            float,
            BetaHeadConfig,
            Dict[str, Union[BetaHeadConfig, Dict[str, float], float]],
        ],
    ) -> Dict[str, BetaHeadConfig]:
        """
        Parse head configurations from various input formats.

        Args:
            configs: Input configuration in various formats

        Returns:
            Dict[str, BetaHeadConfig]: Parsed configuration for each head
        """
        # Case 1: None - use defaults
        if configs is None:
            return DEFAULT_BETA_HEAD_CONFIGS.copy()

        # Case 2: float/int - set w_target for all heads
        if isinstance(configs, (int, float)):
            return {
                head: BetaHeadConfig(
                    w_target=float(configs),
                    learning_rate=DEFAULT_BETA_HEAD_CONFIGS[head].learning_rate,
                    coef_init=DEFAULT_BETA_HEAD_CONFIGS[head].coef_init,
                    coef_min=DEFAULT_BETA_HEAD_CONFIGS[head].coef_min,
                    coef_max=DEFAULT_BETA_HEAD_CONFIGS[head].coef_max,
                    safety_factor=DEFAULT_BETA_HEAD_CONFIGS[head].safety_factor,
                    safety_tau=DEFAULT_BETA_HEAD_CONFIGS[head].safety_tau,
                )
                for head in HEAD_NAMES
            }

        # Case 3: BetaHeadConfig - all heads use same config
        if isinstance(configs, BetaHeadConfig):
            return {head: BetaHeadConfig(**configs.__dict__) for head in HEAD_NAMES}

        # Case 4: Dict - per-head configuration
        if isinstance(configs, dict):
            parsed = {}
            for head in HEAD_NAMES:
                if head in configs:
                    cfg = configs[head]

                    # Sub-case 4a: BetaHeadConfig object
                    if isinstance(cfg, BetaHeadConfig):
                        parsed[head] = cfg

                    # Sub-case 4b: Partial dict
                    elif isinstance(cfg, dict):
                        default = DEFAULT_BETA_HEAD_CONFIGS[head]
                        merged = {
                            "w_target": cfg.get("w_target", default.w_target),
                            "learning_rate": cfg.get(
                                "learning_rate", default.learning_rate
                            ),
                            "coef_init": cfg.get("coef_init", default.coef_init),
                            "coef_min": cfg.get("coef_min", default.coef_min),
                            "coef_max": cfg.get("coef_max", default.coef_max),
                            "safety_factor": cfg.get(
                                "safety_factor", default.safety_factor
                            ),
                            "safety_tau": cfg.get("safety_tau", default.safety_tau),
                        }
                        parsed[head] = BetaHeadConfig(**merged)

                    # Sub-case 4c: float - only set w_target
                    elif isinstance(cfg, (int, float)):
                        default = DEFAULT_BETA_HEAD_CONFIGS[head]
                        parsed[head] = BetaHeadConfig(
                            w_target=float(cfg),
                            learning_rate=default.learning_rate,
                            coef_init=default.coef_init,
                            coef_min=default.coef_min,
                            coef_max=default.coef_max,
                            safety_factor=default.safety_factor,
                            safety_tau=default.safety_tau,
                        )

                    else:
                        raise TypeError(
                            f"Invalid config type for head '{head}': {type(cfg)}. "
                            f"Expected BetaHeadConfig, dict, or float."
                        )
                else:
                    # Head not specified - use default
                    parsed[head] = DEFAULT_BETA_HEAD_CONFIGS[head]

            return parsed

        # Invalid type
        raise TypeError(
            f"head_configs must be None, float, BetaHeadConfig, or dict. "
            f"Got {type(configs)}"
        )

    def _log_configuration(self):
        """Log the configuration for each head."""
        logs = []
        logs.append("AdaptiveEntropyBetaHeadBalancer Configuration:")
        logs.append(f"  EMA decay: {self.ema_decay}")
        logs.append(f"  Warmup steps: {self.warmup_steps}")
        logs.append(f"  Max change ratio (δ): {self.max_change_ratio}")
        logs.append(f"  Update frequency: {self.update_frequency}")
        logs.append(f"  Use safety correction: {self.use_safety_correction}")
        logs.append(f"  Global safety factor (K): {self.safety_factor}")
        logs.append(f"  Global safety tau (τ): {self.safety_tau}")
        logs.append(f"  Policy gradient eps: {self.policy_grad_eps}")
        logs.append("")

        for head, config in self._head_configs.items():
            logs.append(f"  Head '{head}':")
            logs.append(f"    Target ratio (r_h): {config.w_target}")
            logs.append(f"    Learning rate (η): {config.learning_rate}")
            logs.append(f"    Initial coefficient: {config.coef_init}")
            logs.append(
                f"    Coefficient range: [{config.coef_min}, {config.coef_max}]"
            )
            logs.append(f"    Safety factor (K): {config.safety_factor}")
            logs.append(f"    Safety tau (τ): {config.safety_tau}")

        self.logger.info("\n".join(logs))

    def compute_grad_norm(
        self, loss: torch.Tensor, params: List[nn.Parameter], retain_graph: bool = True
    ) -> float:
        """
        Compute the L2 norm of gradients of loss w.r.t. parameters.

        Args:
            loss: Loss tensor
            params: List of parameters

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

    def update_coefficients(
        self,
        shared_params: List[nn.Parameter],
        policy_loss: torch.Tensor,
        ent_losses: Dict[str, torch.Tensor],
        current_step: int,
    ) -> Dict[str, float]:
        """
        Update entropy coefficients using pure gradient norm ratio balancing.

        This method performs the following steps:
        1. If in warmup phase, only update EMA values
        2. Compute gradient norms for policy loss and each entropy loss
        3. Update EMA values for gradient norms
        4. Compute current gradient ratios (w_h)
        5. Apply multiplicative update with change rate clipping
        6. Apply optional safety correction
        7. Clip coefficients to [coef_min, coef_max]
        8. Check for policy gradient vanishing and emergency reset

        Args:
            shared_params: List of shared parameters (e.g., from _attention_net)
            policy_loss: Policy loss (PPO clip loss, without entropy term)
            ent_losses: Dictionary of raw entropy losses for each head
                       {"loop_order": loss, "packing_type": loss,
                        "thread_type": loss, "continuous": loss}
            current_step: Current training step

        Returns:
            Dict[str, float]: Updated entropy coefficients for each head
        """
        self._step_counter += 1

        # Step 1: Check update frequency
        is_warmup = current_step < self.warmup_steps
        if not is_warmup and (self._step_counter % self.update_frequency != 0):
            return self._current_coefs.copy()

        # Step 2: Compute gradient norms
        grad_norm_ent_loop_order = self.compute_grad_norm(
            ent_losses["loop_order"], shared_params,
        )
        grad_norm_ent_packing_type = self.compute_grad_norm(
            ent_losses["packing_type"], shared_params,
        )
        grad_norm_ent_thread_type = self.compute_grad_norm(
            ent_losses["thread_type"], shared_params,
        )
        grad_norm_ent_cont = self.compute_grad_norm(
            ent_losses["continuous"],
            shared_params
        )
        grad_norm_policy = self.compute_grad_norm(
            policy_loss, shared_params, retain_graph=False
        )

        # Step 3: Update EMA values
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

        if is_warmup:
            return self._current_coefs.copy()

        # Store statistics
        self._last_stats.grad_norm_policy = grad_norm_policy
        self._last_stats.grad_norm_loop_order = grad_norm_ent_loop_order
        self._last_stats.grad_norm_packing_type = grad_norm_ent_packing_type
        self._last_stats.grad_norm_thread_type = grad_norm_ent_thread_type
        self._last_stats.grad_norm_continuous = grad_norm_ent_cont
        self._last_stats.triggered_safety_correction = []
        self._last_stats.triggered_emergency_reset = False

        # Step 4: Check if in warmup phase
        if current_step < self.warmup_steps:
            self.logger.info(
                f"[Step {current_step}] In warmup phase ({current_step}/{self.warmup_steps}). "
                f"Only updating EMA values."
            )
            return self._current_coefs.copy()

        if self._ema_grad_norm_policy is None:
            self.logger.info(
                f"[Step {current_step}] coefs will keep the same({self._current_coefs}) due to ema_grad_norm_policy is None. "
            )
            return self._current_coefs.copy()

        # Step 5: Check for policy gradient vanishing - emergency reset
        if self._ema_grad_norm_policy < self.policy_grad_eps:
            self.logger.warning(
                f"[EMERGENCY RESET] Policy gradient norm vanished "
                f"(EMA: {self._ema_grad_norm_policy:.6e} < {self.policy_grad_eps:.6e}). "
                f"Resetting all entropy coefficients to minimum values."
            )
            for name in HEAD_NAMES:
                self._current_coefs[name] = self._head_configs[name].coef_min
            self._last_stats.triggered_emergency_reset = True
            return self._current_coefs.copy()

        # Step 6: Compute current gradient ratios using EMA values
        # w_h = g_ent,h / (g_policy + epsilon)
        ema_policy = self._ema_grad_norm_policy + self.epsilon

        ratios = {
            "loop_order": (self._ema_grad_norm_ent_loop_order or 0.0) / ema_policy,
            "packing_type": (self._ema_grad_norm_ent_packing_type or 0.0) / ema_policy,
            "thread_type": (self._ema_grad_norm_ent_thread_type or 0.0) / ema_policy,
            "continuous": (self._ema_grad_norm_ent_cont or 0.0) / ema_policy,
        }

        # Store ratios in stats
        self._last_stats.ratio_loop_order = ratios["loop_order"]
        self._last_stats.ratio_packing_type = ratios["packing_type"]
        self._last_stats.ratio_thread_type = ratios["thread_type"]
        self._last_stats.ratio_continuous = ratios["continuous"]

        # Step 7: Update coefficients for each head
        updated_coefs = {}

        for head in HEAD_NAMES:
            config = self._head_configs[head]
            current_coef = self._current_coefs[head]
            w_current = ratios[head]
            r_target = config.w_target
            eta = config.learning_rate

            # Multiplicative update: β_new = β_old * exp(η * clamp(r - w, -δ, δ))
            ratio_diff = r_target - w_current
            clamped_diff = max(
                -self.max_change_ratio, min(self.max_change_ratio, ratio_diff)
            )
            update_factor = math.exp(eta * clamped_diff)
            new_coef = current_coef * update_factor

            # Step 8: Optional safety correction
            if self.use_safety_correction:
                # Check if ratio deviates too much from target
                # Trigger correction if w_h > K * r_h or w_h < r_h / K
                safety_factor = config.safety_factor
                upper_threshold = safety_factor * r_target
                lower_threshold = r_target / safety_factor

                if w_current > upper_threshold or w_current < lower_threshold:
                    # Safety correction: β_new = β_old * (r / w)^τ
                    # Ensure w_current is not too small to avoid division issues
                    w_safe = max(w_current, self.epsilon)
                    correction_factor = (r_target / w_safe) ** config.safety_tau
                    new_coef = new_coef * correction_factor

                    self._last_stats.triggered_safety_correction.append(head)
                    self.logger.info(
                        f"[Safety Correction] Head '{head}': ratio {w_current} "
                        f"outside safe range [{lower_threshold:.6f}, {upper_threshold:.6f}]. "
                        f"Applied correction factor {correction_factor} and get new_coef {new_coef}."
                    )

            # Step 9: Clip to [coef_min, coef_max]
            new_coef = max(config.coef_min, min(config.coef_max, new_coef))
            updated_coefs[head] = new_coef

        # Update current coefficients
        self._current_coefs = updated_coefs

        # Store coefficient values in stats
        self._last_stats.coef_loop_order = updated_coefs["loop_order"]
        self._last_stats.coef_packing_type = updated_coefs["packing_type"]
        self._last_stats.coef_thread_type = updated_coefs["thread_type"]
        self._last_stats.coef_continuous = updated_coefs["continuous"]

        # Log update information
        self._log_update_info(current_step, ratios, updated_coefs)

        return updated_coefs

    def _log_update_info(
        self,
        current_step: int,
        ratios: Dict[str, float],
        updated_coefs: Dict[str, float],
    ):
        """Log detailed update information."""
        log_msg = f"[Step {current_step}] Beta-head balancer update:\n"
        log_msg += "  Gradient norms (EMA):\n"
        log_msg += f"    Policy: {self._ema_grad_norm_policy:.6e}\n"
        log_msg += f"    loop_order: {self._ema_grad_norm_ent_loop_order:.6e}\n"
        log_msg += f"    packing_type: {self._ema_grad_norm_ent_packing_type:.6e}\n"
        log_msg += f"    thread_type: {self._ema_grad_norm_ent_thread_type:.6e}\n"
        log_msg += f"    continuous: {self._ema_grad_norm_ent_cont:.6e}\n"
        log_msg += "  Gradient ratios (w_h):\n"

        for head in HEAD_NAMES:
            r_target = self._head_configs[head].w_target
            w_current = ratios[head]
            log_msg += f"    {head}: {w_current:.6f} (target: {r_target:.6f})\n"

        log_msg += "  Updated coefficients (β_h):\n"
        for head in HEAD_NAMES:
            log_msg += f"    {head}: {updated_coefs[head]:.6e}\n"

        if self._last_stats.triggered_safety_correction:
            log_msg += f"  Safety correction triggered for: {self._last_stats.triggered_safety_correction}\n"

        if self._last_stats.triggered_emergency_reset:
            log_msg += "  EMERGENCY RESET triggered!\n"

        self.logger.info(log_msg)

    def get_last_stats(self) -> BetaHeadStats:
        """
        Get statistics from the last update.

        Returns:
            BetaHeadStats: Statistics including gradient norms, ratios, and coefficients
        """
        return self._last_stats

    def get_head_configs(self) -> Dict[str, BetaHeadConfig]:
        """
        Get head configurations.

        Returns:
            Dict[str, BetaHeadConfig]: Configuration for each head
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

    def get_current_coefficients(self) -> Dict[str, float]:
        """
        Get current entropy coefficients.

        Returns:
            Dict[str, float]: Current coefficients for each head
        """
        return self._current_coefs.copy()

    def reset(self):
        """Reset the balancer to initial state."""
        self._ema_grad_norm_policy = None
        self._ema_grad_norm_ent_loop_order = None
        self._ema_grad_norm_ent_packing_type = None
        self._ema_grad_norm_ent_thread_type = None
        self._ema_grad_norm_ent_cont = None

        self._current_coefs = {
            name: config.coef_init for name, config in self._head_configs.items()
        }

        self._step_counter = 0
        self._last_stats = BetaHeadStats()

        self.logger.info("AdaptiveEntropyBetaHeadBalancer reset to initial state.")
