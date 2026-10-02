# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/3/29 19:21
# Contact       : hucheng.lew@qq.com
# File          : gemm_test_collect_stats.py
# IDE           : PyCharm
# Description   : 测试环境的收集器

from tianshou.data import CollectStats, SequenceSummaryStats
from tianshou.data.collector import CollectStepBatchProtocol, EpisodeBatchProtocol
from dataclasses import dataclass, field
import numpy as np

@dataclass(kw_only=True)
class GEMMTestCollectStats(CollectStats):
    """
    测试环境的收集器
    """
    env_performances: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    env_performance: SequenceSummaryStats | None = None

    def update_at_step_batch(
            self,
            step_batch: CollectStepBatchProtocol,
            refresh_sequence_stats: bool = False,
    ) -> None:
        super(GEMMTestCollectStats, self).update_at_step_batch(step_batch, refresh_sequence_stats)

        info = step_batch.info
        if info is not None:
            # 获取原始性能数据
            performance = info["performance"].mean().item()
            if self.env_performances.size == 0:
                self.env_performances = np.array([performance], dtype=float)
            else:
                self.env_performances = np.concatenate((self.env_performances, [performance]), dtype=float)  # type: ignore

        if refresh_sequence_stats:
            self.refresh_env_performance()

    def update_at_episode_done(
            self,
            episode_batch: EpisodeBatchProtocol,
            # NOTE: in the MARL setting this is not actually a float but rather an array or list, see todo below
            episode_return: float,
            refresh_sequence_stats: bool = False,
    ) -> None:
        super(GEMMTestCollectStats, self).update_at_episode_done(episode_batch, episode_return, refresh_sequence_stats)

    def refresh_env_performance(self) -> None:
        if self.env_performances.size > 0:
            self.env_performance = SequenceSummaryStats.from_sequence(self.env_performances)
        else:
            self.env_performance = None

    def refresh_all_sequence_stats(self) -> None:
        super(GEMMTestCollectStats, self).refresh_all_sequence_stats()
        self.refresh_env_performance()
