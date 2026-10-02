"""PPO 消融开关的快速回归测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gemm_env import GemmEnv  # noqa: E402
from reward_manager import RewardManager  # noqa: E402


class _History:
    def __init__(self, best: float | None, global_best: float | None):
        self.best = best
        self.global_best = global_best

    def get_shape_best_performance(self, shape):
        return self.best

    def get_best_global_avg_performance(self):
        return self.global_best

    def get_ema_performance(self, shape):
        return 1.0


class _ActionHistory:
    def get_discrete_action_ratio(self, *args):
        return (1.0, 0.0)


class TestPPOAblation(unittest.TestCase):
    def test_uniform_sampling_is_permutation_and_seeded(self):
        shapes = [(1, 8, 64), (2, 16, 128), (3, 24, 192)]
        first = SimpleNamespace(shape_list=shapes, np_random=np.random.default_rng(7))
        second = SimpleNamespace(shape_list=shapes, np_random=np.random.default_rng(7))
        order_a = GemmEnv._build_uniform_training_shape_list(first)
        order_b = GemmEnv._build_uniform_training_shape_list(second)
        self.assertEqual(order_a, order_b)
        self.assertEqual(sorted(order_a), sorted(shapes))
        self.assertEqual(len(order_a), len(set(order_a)))

    def test_pure_performance_is_monotonic_and_history_independent(self):
        manager = RewardManager(
            name="unit_test_ablation",
            loop_orders=[0],
            packing_types=[0],
            thread_types=[[1, 1, 1]],
            action_space_transformer=None,
            logger=None,
            reward_mode="pure_performance",
        )
        manager._history_performance_dict = _History(10.0, 10.0)
        manager._history_queue = _ActionHistory()
        shape = (1, 8, 64)
        low = manager.compute_performance_reward(shape, 8.0, 0, 0, (1, 1, 1))
        high = manager.compute_performance_reward(shape, 12.0, 0, 0, (1, 1, 1))
        self.assertGreater(high, low)
        manager._history_queue = SimpleNamespace(
            get_discrete_action_ratio=lambda *args: (0.0, 1.0)
        )
        self.assertEqual(
            high,
            manager.compute_performance_reward(shape, 12.0, 0, 0, (1, 1, 1)),
        )
        self.assertEqual(0.0, manager.compute_performance_reward(shape, 0.0, 0, 0, (1, 1, 1)))

    def test_modes_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            RewardManager(
                name="invalid_reward_mode",
                loop_orders=[0],
                packing_types=[0],
                thread_types=[[1, 1, 1]],
                action_space_transformer=None,
                reward_mode="invalid",
            )
        with self.assertRaises(ValueError):
            GemmEnv(
                loop_orders=[0],
                packing_types=[0],
                threads=[[1, 1, 1]],
                matrix_element_size=4,
                gemm_config_type="invalid",
                dynamic_sampling_config={"sampling_mode": "invalid"},
            )

    def test_generated_task_modes(self):
        task_dir = Path(__file__).parents[2] / "configs" / "rl_train"
        expected = {
            "base": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4_ablate_plain.json",
            "a": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4_ablate_adaptive_repeated_sampling.json",
            "r": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4_ablate_multilevel_reward.json",
            "e": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4_ablate_adaptive_entropy.json",
            "a_r": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4_ablate_adaptive_repeated_sampling_multilevel_reward.json",
            "a_e": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4_ablate_adaptive_repeated_sampling_adaptive_entropy.json",
            "r_e": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4_ablate_multilevel_reward_adaptive_entropy.json",
            "a_r_e": "ft2000_rl_diff_rnd_seed_gemm_train_t1_task4.json",
        }
        self.assertEqual(8, len(expected))
        for variant, filename in expected.items():
            with self.subTest(variant=variant):
                path = task_dir / filename
                self.assertTrue(path.is_file(), path)
                task = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("gemm_env_config", task["gemm_configs"])
                self.assertIn("train_config", task["gemm_configs"])


if __name__ == "__main__":
    unittest.main()
