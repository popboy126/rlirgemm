# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/3/31 08:47
# Contact       : hucheng.lew@qq.com
# File          : cache_manager.py
# IDE           : PyCharm
# Description   : 缓存管理器，用于存储全局的

import multiprocessing
import multiprocessing.shared_memory as shared_memory
from multiprocessing.synchronize import Lock
from typing import List, Tuple, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
import itertools


class AbstractCacheManager(ABC):
    """
    管理跨agent的缓存共享
    """
    def __init__(self, shape_size: Tuple[int, int], shm_name: str = None, create: bool = True, lock: Lock = None):
        """
        初始化
        """
        self._shape_size = shape_size
        self._create = create
        self._shm_name = shm_name
        self._dtype = np.float64

        self._shm, self._arr = self._create_or_attach_shared_memory()  # 创建或挂载共享内存

        if self._create:
            self._lock = multiprocessing.Lock()
        elif lock is None:
            raise ValueError("lock must not be None")
        else:
            self._lock = lock

    def _create_or_attach_shared_memory(self) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
        """
        创建或挂载共享内存
        """
        if self._create:
            # 计算所需字节数
            bytes_needed = np.prod(self._shape_size) * np.dtype(self._dtype).itemsize
            # 创建共享内存（名称自动生成）
            shm = shared_memory.SharedMemory(create=True, size=int(bytes_needed))
            # 将共享内存映射为 numpy 数组
            arr = np.ndarray(self._shape_size, dtype=self._dtype, buffer=shm.buf)
        else:
            # 连接到已存在的共享内存
            shm = shared_memory.SharedMemory(name=self._shm_name)
            # 创建 numpy 数组视图
            arr = np.ndarray(self._shape_size , dtype=self._dtype, buffer=shm.buf)
        return shm, arr

    def get_lock(self) -> Lock | None:
        """
        获取锁
        """
        return self._lock

    def release(self):
        """
        释放资源
        """
        if self._arr is not None:
            self._arr = None   # 断开视图，方便垃圾回收
        if self._shm is not None:
            self._shm.close()    # 关闭当前进程对共享内存的引用
            # 如果是创建者（未提供 name），则删除共享内存块
            if self._create:
                self._shm.unlink()
            self._shm = None

    def get_shm_name(self) -> None | str:
        """
        获取shm名称
        """
        if self._create:
            if self._shm is not None:
                return self._shm.name
            else:
                return None
        elif self._shm_name is not None:
            return self._shm_name
        else:
            return None

    @abstractmethod
    def get_index(self, key: Any) -> int | None:
        """
        获取索引
        """
        raise NotImplementedError

    def get_value(self, key: Any) -> np.ndarray | None:
        """
        获取数据
        """
        index = self.get_index(key)
        if index is None:
            raise ValueError(f"key {key} is not in the index!")

        if self._lock is None:
            raise ValueError("lock must not be None")

        with self._lock:
            # 更新最佳记录
            row = self._arr[index].copy()
        return row


    def _init_shared_memory(self, checkpoint_data: Dict[Any, Any] = None) -> None:
        """
        初始化共享内存
        """
        if not self._create:  # 挂载的不允许初始化
            return

        if checkpoint_data is None:  # 没有checkpoint的话，全部初始化为0
            self._arr[:] = 0.0
        else:
            for key, value in checkpoint_data.items():
                # 根据键获取数组行索引
                idx = self.get_index(key)
                if idx is None:
                    continue
                # 将元组转换为 numpy 数组并赋值给对应行
                self._arr[idx, :] = np.array(value, dtype=self._dtype)

    @abstractmethod
    def update(
            self,
            **kwargs
    ) -> str:
        """
        更新历史性能记录
        """
        raise NotImplementedError

    @abstractmethod
    def dumps(self) -> Dict[Any, Any]:
        """
        保存为字典文件
        """
        raise NotImplementedError


class ShapePerformanceCacheManager(AbstractCacheManager):
    """
    管理跨agent的缓存共享
    """
    def __init__(self, ema_alpha:float, shape_list: List[Tuple[int, int, int]]= None,
                 checkpoint_data: Dict[Tuple[int, int, int], Tuple[float, float, float, float, float]] = None, shm_name: str = None, create: bool = True, lock: Lock = None):
        """
        初始化
        """
        self._ema_alpha = ema_alpha
        self._global_stat_key = (0, 0, 0)
        if shape_list is not None:
            shape_set = set(shape_list)
            shape_set.add(self._global_stat_key)
            shape_size = (len(shape_set), 5)
            shapes = sorted(list(shape_set), key=lambda x: x)
            self._shape_to_index = { key: idx for idx, key in enumerate(shapes) }
        elif checkpoint_data is not None:
            shape_size = (len(checkpoint_data), 5)
            shape_list = sorted(checkpoint_data.keys(), key=lambda x: x)
            self._shape_to_index = { key: idx for idx, key in enumerate(shape_list) }
        else:
            raise ValueError("checkpoint_data or shape_list must not be None")

        super(ShapePerformanceCacheManager, self).__init__(
            shape_size = shape_size,
            shm_name = shm_name,
            create = create,
            lock = lock
        )

        self._init_shared_memory(checkpoint_data=checkpoint_data)

    def get_index(self, key: Any) -> int | None:
        """
        获取索引
        """
        return self._shape_to_index.get(key, None)

    def update(
        self,
        **kwargs
    ) -> str:
        """
        更新历史性能记录
        :param shape: 输入形状 (M, K, N)
        :param performance: 当前性能 (GFlops)
        """
        shape: Tuple[int, int, int] | None = kwargs.get("shape", None)
        performance: float | None = kwargs.get("performance", None)

        index = self.get_index(shape)
        if index is None:
            raise ValueError(f"shape {shape} is not in the index!")

        if self._lock is None:
            raise ValueError("lock must not be None")

        global_index = self._shape_to_index.get(self._global_stat_key, None)
        if global_index is None:
            raise ValueError(f"global statkey {self._global_stat_key} is not in the index!")

        with self._lock:
            # 更新最佳记录
            row = self._arr[index].copy()
            row[0] = row[0] + 1    # total count
            if np.isclose(row[1], 0):
                row[1] = performance
            else:
                row[1] = row[1] if performance > row[1] else performance  # min performance
            row[2] = row[2] if performance < row[2] else performance  # max performance
            row[3] = self._ema_alpha * row[3] + (1 - self._ema_alpha) * performance # ema performance
            row[4] = row[4] + performance  # total performance
            self._arr[index] = row

            global_row = self._arr[global_index].copy()
            global_row[0] = global_row[0] + 1  # total count
            if np.isclose(global_row[1], 0):
                global_row[1] = performance
            else:
                global_row[1] = global_row[1] if performance > global_row[1] else performance  # global min performance
            global_row[2] = global_row[2] if performance < global_row[2] else performance  # global max performance
            global_row[4] = global_row[4] + performance  # global total performance
            current_global_avg = global_row[4] / global_row[0]
            global_row[3] = global_row[3] if current_global_avg < global_row[3] else current_global_avg  # global best average performance
            self._arr[global_index] = global_row

        return "" if int(global_row[0]) % 50 != 0 else self.get_logger_str(global_row)

    def get_logger_str(self, row: np.ndarray) -> str:
        """
        获取日志串
        """
        if row[0] > 0:
            mean_perf = row[4] / row[0]
        else:
            mean_perf = 0
        log_str = (f"History performance stats: global_min={row[1]:.2f}, "
                   f"global_max={row[2]:.2f}, global_avg={mean_perf:.2f}, "
                   f"global_avg_best={row[3]:.2f}, "
                   f"total_samples={row[0]}")
        return log_str

    def get_global_stat_row(self) -> np.ndarray:
        """
        获取全局统计数据
        """
        global_index = self._shape_to_index.get(self._global_stat_key, None)
        if global_index is None:
            raise ValueError(f"global statkey {self._global_stat_key} is not in the index!")

        with self._lock:
            global_row = self._arr[global_index].copy()
        return global_row

    def get_shape_performances(self, shape: Tuple[int, int, int]) -> Tuple[float, float, float, float] | None:
        """
        获取形状性能
        """
        performances = self.get_value(shape)

        if performances is not None:
            total_count, min_perf, max_perf, ema_perf, total_perf = performances[0].item(), performances[1].item(), performances[2].item(), performances[3].item(), performances[4].item()
            avg_performance = total_perf / total_count if total_count > 0 else 0
            return min_perf, max_perf, ema_perf, avg_performance
        else:
            return None

    def get_global_performances(self) -> Tuple[float, float, float, float] | None:
        """
        获取全局性能信息
        """
        performances = self.get_value(self._global_stat_key)

        if performances is not None:
            total_count, min_perf, max_perf, best_avg_perf, total_perf = performances[0].item(), performances[1].item(), performances[2].item(), performances[3].item(), performances[4].item()
            avg_performance = total_perf / total_count if total_count > 0 else 0
            return min_perf, max_perf, best_avg_perf, avg_performance
        else:
            return None

    def get_global_stat_str(self) -> str:
        """
        获取全局统计信息
        """
        global_row = self.get_global_stat_row()
        return self.get_logger_str(global_row)

    def dumps(self) -> Dict[Any, Any]:
        """
        保存为字典文件
        """
        dump_dict = {}
        with self._lock:
            for shape, idx in self._shape_to_index.items():
                row = self._arr[idx].copy()
                dump_dict[shape] = (row[0].item(), row[1].item(), row[2].item(), row[3].item(), row[4].item())
        return dump_dict




import unittest
import random
import time

class TestShapePerformanceCacheManager(unittest.TestCase):
    """
    测试ShapePerformanceCacheManager
    """
    def setUp(self) -> None:
        self._shape: List[Tuple[int, int, int]] = [(1, 2, 4), (2, 4, 8) ,(4, 4, 8)]
        # self._cache_manager = ShapePerformanceCacheManager(
        #     ema_alpha = 0.05,
        #     shape_list = [(1, 2, 4), (2, 4, 8) ,(4, 4, 8)],
        #     create = True
        # )

    @staticmethod
    def worker(worker_id:int, shm_name: str, shape: List[Tuple[int, int, int]], lock: Lock, ema_alpha: float = 0.05, iterations: int=1000):
        """
        用于测试的worker线程
        """
        cache_manager = ShapePerformanceCacheManager(
            ema_alpha = 0.05,
            shape_list = shape,
            shm_name = shm_name,
            create = False,
            lock = lock
        )

        for _ in range(iterations):
            single_shape = random.choice(shape)
            performance = random.uniform(0, 100)

            log_str = cache_manager.update(shape=single_shape, performance=performance, ema_alpha=ema_alpha)

            # 可选：读取验证
            if log_str != "":
                print(f"worker({worker_id}) : {log_str}")

            # 模拟工作负载
            time.sleep(random.uniform(0.001, 0.01))


    def test_update(self):
        """
        测试更新
        """
        cache_manager = ShapePerformanceCacheManager(
            ema_alpha = 0.05,
            shape_list = self._shape,
            create = True,
        )

        # 启动 4 个工作进程
        processes = []
        shm_name = cache_manager.get_shm_name()
        lock = cache_manager.get_lock()
        for i in range(4):
            p = multiprocessing.Process(target=self.worker, args=(i, shm_name, self._shape, lock, 0.05, 500))
            processes.append(p)
            p.start()

        # 等待所有进程结束
        for p in processes:
            p.join()

        global_stat_str = cache_manager.get_global_stat_str()
        print(f"Final global stat : {global_stat_str}")

        cache_manager.release()


class DiscreteActionCacheManager(AbstractCacheManager):
    """
    统计离散形状
    """
    def __init__(self,
                 loop_orders: List[int]= None,
                 packing_types: List[int] = None,
                 thread_types: List[Tuple[int, int, int]] = None,
                 checkpoint_data: Dict[Tuple[int, int, Tuple[int, int, int]], int] = None,
                 shm_name: str = None, create: bool = True, lock: Lock = None):
        """
        初始化
        """
        self._global_valid_stat_key = (-1, -1, (0, 0, 0))
        self._global_invalid_stat_key = (-1, -1, (-1, -1, -1))
        if (loop_orders is not None) and (packing_types is not None) and (thread_types is not None):
            combined_actions = set(itertools.product(loop_orders, packing_types, thread_types))
            combined_actions.add(self._global_valid_stat_key)
            combined_actions.add(self._global_invalid_stat_key)
            shape_size = (len(combined_actions), 1)
            disc_actions = sorted(list(combined_actions), key=lambda x: x)
            self._disc_actions_to_index = { key: idx for idx, key in enumerate(disc_actions) }
        elif checkpoint_data is not None:
            shape_size = (len(checkpoint_data), 1)
            disc_actions = sorted(checkpoint_data.keys(), key=lambda x: x)
            self._disc_actions_to_index = { key: idx for idx, key in enumerate(disc_actions) }
        else:
            raise ValueError("checkpoint_data or loop_orders, packing_types, thread_types  must not be None")

        self._disc_action_ideal_ratio = 1 / (len(disc_actions) - 2)  # 减去两个全局统计数据
        super(DiscreteActionCacheManager, self).__init__(
            shape_size = shape_size,
            shm_name = shm_name,
            create = create,
            lock = lock
        )

        self._init_shared_memory(checkpoint_data=checkpoint_data)

        self._global_invalid_stat_idx = self._disc_actions_to_index[self._global_invalid_stat_key]
        self._global_valid_stat_idx = self._disc_actions_to_index[self._global_valid_stat_key]

    def get_index(self, key: Any) -> int | None:
        """
        获取索引
        """
        return self._disc_actions_to_index.get(key, None)

    def _make_key(
            self,
            loop_order: int, packing_type: int, thread_type: Tuple[int, int, int]
    ) -> tuple:
        """
        生成性能记录的key，包含shape和线程配置
        :param shape: 输入形状 (M, K, N)
        :param thread_config: 线程配置 (tm, tk, tn)，如果为None则使用默认单线程(1,1,1)
        :return: 组合key
        """
        return (
            loop_order,
            packing_type,
            thread_type,
        )

    def update(
            self,
            **kwargs
    ) -> str:
        """
        更新历史性能记录
        :param shape: 输入形状 (M, K, N)
        :param performance: 当前性能 (GFlops)
        """
        loop_order: int | None = kwargs.get("loop_order", None)
        packing_type: int | None = kwargs.get("packing_type", None)
        thread_type: Tuple[int, int, int] | None = kwargs.get("thread_type", None)
        is_valid : bool = kwargs.get("is_valid", False)

        if loop_order is None or packing_type is None or thread_type is None:
            return ""

        if self._lock is None:
            raise ValueError("lock must not be None")

        if not is_valid:
            with self._lock:
                # 更新最佳记录
                self._arr[self._global_invalid_stat_idx] += 1
            return ""
        else:
            disc_action_key = self._make_key(loop_order, packing_type, thread_type)
            index = self.get_index(disc_action_key)
            if index is None:
                raise ValueError(f"discrete action {disc_action_key} is not in the index!")

            with self._lock:
                # 更新最佳记录
                self._arr[index] += 1
                valid_row = self._arr[self._global_valid_stat_idx].copy()
                valid_row[0] = valid_row[0] + 1
                self._arr[self._global_valid_stat_idx] = valid_row

            return "" if int(valid_row[0]) % 100 != 0 else self.get_global_stat_str()

    def dumps(self) -> Dict[Any, Any]:
        """
        保存为字典文件
        """
        dump_dict = {}
        with self._lock:
            for disc_action, idx in self._disc_actions_to_index.items():
                row = self._arr[idx].copy()
                dump_dict[disc_action] = row[0].item()
        return dump_dict


    def get_global_stat_str(self, print_all: bool = False) -> str:
        """
        获取统计信息
        """
        if self._lock is None:
            raise ValueError("lock must not be None")

        disc_action_stat = {}

        with self._lock:
            for disc_action, idx in self._disc_actions_to_index.items():
                disc_action_stat[disc_action] = self._arr[idx].copy().item()

        total_valid_count = disc_action_stat.pop(self._global_valid_stat_key, 0)
        total_invalid_count = disc_action_stat.pop(self._global_invalid_stat_key, 0)
        total_keys = len(disc_action_stat)

        stat = [""]
        if total_keys <= 10:
            stat.append(f"=== Full statistics at total valid steps: {total_valid_count}, total invalid steps: {total_invalid_count}, total steps: {total_valid_count + total_invalid_count}, total discrete actions: {total_keys} ===")
            sorted_items = sorted(disc_action_stat.items(), key=lambda x: x[1], reverse=True)
            for action, action_count in sorted_items:
                stat.append(f"  Loop Order: {action[0]}, Packing Type: {action[1]}, Thread Type: {action[2]} -> count: {action_count}, frequency: {action_count / (total_valid_count + 1e-10):.6f}")
        elif (total_valid_count % 1000 == 0) or (print_all == True):  # 全量输出
            stat.append(f"=== Full statistics at total valid steps: {total_valid_count}, total invalid steps: {total_invalid_count}, total steps: {total_valid_count + total_invalid_count}, total discrete actions: {total_keys} ===")
            sorted_items = sorted(disc_action_stat.items(), key=lambda x: x[1], reverse=True)
            for action, action_count in sorted_items:
                stat.append(f"  Loop Order: {action[0]}, Packing Type: {action[1]}, Thread Type: {action[2]} -> count: {action_count}, frequency: {action_count / (total_valid_count + 1e-10):.6f}")
        else:  # 只输出一部分
            stat.append(f"=== Important statistics at total valid steps: {total_valid_count}, total invalid steps: {total_invalid_count}, total steps: {total_valid_count + total_invalid_count}, total discrete actions: {total_keys} ===")
            sorted_items = sorted(disc_action_stat.items(), key=lambda x: x[1], reverse=True)
            for action, action_count in sorted_items[:5]:
                stat.append(f"  Loop Order: {action[0]}, Packing Type: {action[1]}, Thread Type: {action[2]} -> count: {action_count}, frequency: {action_count / (total_valid_count + 1e-10):.6f}")
            stat.append("  ...")

            for action, action_count in sorted_items[-5:]:
                stat.append(f"  Loop Order: {action[0]}, Packing Type: {action[1]}, Thread Type: {action[2]} -> count: {action_count}, frequency: {action_count / (total_valid_count + 1e-10):.6f}")

        stat.append("  ---")
        return "\n".join(stat)

    def get_discrete_action_ratio(self, loop_order: int, packing_type: int, thread_type: Tuple[int, int, int]) -> Tuple[float, float]:
        """
        获得指定离散动作的概率
        """
        disc_action_key = self._make_key(loop_order, packing_type, thread_type)

        index = self.get_index(disc_action_key)
        if index is None:
            raise ValueError(f"discrete action {disc_action_key} is not in the index!")

        with self._lock:
            # 更新最佳记录
            action = self._arr[index].copy().item()
            total_valid_count = self._arr[self._global_valid_stat_idx].copy().item()

        if total_valid_count == 0:
            ratio = 0.0
        else:
            ratio = action / total_valid_count
        return self._disc_action_ideal_ratio, ratio


class TestDiscreteActionCacheManager(unittest.TestCase):
    """
    测试DiscreteActionCacheManager
    """
    def setUp(self) -> None:
        self._loop_orders: List[int] = [0, 1, 2]
        self._packing_types: List[int] = [0, 1]
        self._thread_types: List[Tuple[int, int, int]] = [(1, 1, 1), (1, 2, 1)]

    @staticmethod
    def worker(worker_id:int, shm_name: str,
               loop_orders: List[int],
               packing_types: List[int],
               thread_types: List[Tuple[int, int, int]], lock: Lock, iterations: int=1000):
        """
        用于测试的worker线程
        """
        cache_manager = DiscreteActionCacheManager(
            loop_orders=loop_orders,
            packing_types=packing_types,
            thread_types=thread_types,
            shm_name = shm_name,
            create = False,
            lock = lock
        )

        for _ in range(iterations):
            loop_order = random.choice(loop_orders)
            packing_type = random.choice(packing_types)
            thread_type = random.choice(thread_types)
            valid = random.choice([True, False])

            log_str = cache_manager.update(loop_order=loop_order, packing_type=packing_type, thread_type=thread_type, is_valid=valid)

            # 可选：读取验证
            if log_str != "":
                print(f"worker({worker_id}) : {log_str}")

            # 模拟工作负载
            time.sleep(random.uniform(0.001, 0.01))


    def test_update(self):
        """
        测试更新
        """
        cache_manager = DiscreteActionCacheManager(
            loop_orders=self._loop_orders,
            packing_types=self._packing_types,
            thread_types=self._thread_types,
            create = True,
        )

        # 启动 4 个工作进程
        processes = []
        shm_name = cache_manager.get_shm_name()
        lock = cache_manager.get_lock()
        for i in range(4):
            p = multiprocessing.Process(target=self.worker, args=(i, shm_name, self._loop_orders, self._packing_types, self._thread_types, lock,  500))
            processes.append(p)
            p.start()

        # 等待所有进程结束
        for p in processes:
            p.join()

        global_stat_str = cache_manager.get_global_stat_str(True)
        print(f"Final global stat : {global_stat_str}")

        cache_manager.release()
