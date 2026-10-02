# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/7/17 20:11
# Contact       : hucheng.lew@qq.com
# File          : feature_normalize.py
# IDE           : PyCharm
# Description   : 归一化方案

import unittest
import numpy as np
from typing import Tuple, List, Any
from math import floor


class ContinuesRatioNormalizer:
    """
    用于对特征进行归一化处理
    """
    def __init__(self, input_min: int, input_max: int, output_min: float, output_max: float):
        """
        构造函数
        """
        self._input_min = input_min
        self._input_max = input_max
        self._output_min = output_min
        self._output_max = output_max

        # 提前计算范围
        self._input_range = self._input_max - self._input_min
        self._output_range = self._output_max - self._output_min

        # 标记是否为常数区间（使用整数比较避免浮点精度问题）
        self._input_is_constant = (self._input_range == 0)
        self._output_is_constant = (self._output_range == 0)

    def normalize(self, feature: float) -> float:
        """
        对特征进行归一化处理
        :param feature: 特征值
        :return: 归一化后的特征值
        """
        feature = np.clip(feature, self._input_min, self._input_max)
        if self._input_is_constant:
            # 输入范围恒定时，所有输入值均映射到输出最大值（与原逻辑一致，也可改为输出最小值）
            return self._output_max
        ratio = (feature - self._input_min) / (self._input_max - self._input_min)
        output_feature = self._output_min + ratio * (self._output_max - self._output_min)
        return output_feature

    def denormalize(self, action: float) -> int:
        """
        将归一化后的特征值反归一化为原始特征值
        """
        feature = np.clip(action, self._output_min, self._output_max)
        if self._output_is_constant:
            # 输出范围恒定时，所有输入动作均映射回原始输入的最大值（与原归一化逻辑对称）
            return int(round(self._input_max))
        ratio = (feature - self._output_min) / (self._output_max - self._output_min)
        output_feature = self._input_min + ratio * (self._input_max - self._input_min)
        output_feature = np.clip(output_feature, self._input_min, self._input_max)
        return int(round(output_feature))



class TestContinuesRatioNormalizer(unittest.TestCase):
    """
    测试按比例归一化
    """
    def test_normalize_denormalize(self):
        """
        测试区块生成函数
        """
        normalizer = ContinuesRatioNormalizer(0, 100, -1.0, 1.0)
        for test_feature in [-10, 0, 25, 50, 75, 100, 110]:
            norm_value = normalizer.normalize(test_feature)
            restored_feature = normalizer.denormalize(norm_value)
            expected_feature = np.clip(test_feature, 0, 100)
            self.assertEqual(expected_feature, restored_feature)

    def test_single_digit_denormalize(self):
        """
        测试区块生成函数
        """
        normalizer = ContinuesRatioNormalizer(1, 1, 0, 1.0)
        for test_feature in [1]:
            norm_value = normalizer.normalize(test_feature)
            restored_feature = normalizer.denormalize(norm_value)
            self.assertEqual(restored_feature, restored_feature)



class ContinuesBucketNormalizer(object):
    """
    用于对特征进行归一化处理
    """
    def __init__(self, input_min: int, input_max: int, bucket_width: int = 100, output_min: float = 0.0, output_max: float = 1.0):
        """
        构造函数
        """
        self._input_min = input_min
        self._input_max = input_max
        self._output_min = output_min
        self._output_max = output_max
        self._bucket_width = bucket_width
        bucket_num = (self._input_max - self._input_min) / self._bucket_width  # 共有多少个桶
        self._bucket_num = int(np.ceil(bucket_num))
        self._input_edges = np.linspace(self._input_min, self._input_max, self._bucket_num)
        self._eps = 1e-6
        self._input_edges[-1] = self._input_max + self._eps  # 确保最大值在最后一个桶内

        self._output_edges = np.linspace(self._output_min, self._output_max, self._bucket_num)
        self._output_edges[-1] = self._output_max + self._eps  # 确保最大值在最后一个桶内

    def get_rounded_input_feature(self, feature: int) -> int:
        """
        将特征值四舍五入到最近的输入步长
        :param feature: 特征值
        :return: 四舍五入后的特征值
        """
        feature = np.clip(feature, self._input_min, self._input_max)  # 确保特征值在有效范围内
        return feature.item()

    def get_rounded_output_feature(self, feature: float) -> float:
        """
        将特征值四舍五入到最近的输入步长
        :param feature: 特征值
        :return: 四舍五入后的特征值
        """
        feature = np.clip(feature, self._output_min, self._output_max)  # 确保特征值在有效范围内
        return feature.item()

    def normalize(self, feature: int) -> float:
        """
        对特征进行归一化处理
        :param feature: 特征值
        :return: 归一化后的特征值
        """
        rounded_feature = self.get_rounded_input_feature(feature)
        # 确定桶索引
        idx = np.digitize(rounded_feature, self._input_edges).astype(int) - 1  # 桶索引从0开始
        idx = np.clip(idx, -1, self._bucket_num - 1)  # 确保索引在有效范围内
        if hasattr(idx, 'astype'):
            idx = idx.astype(int)

        # 桶内归一化
        left = self._input_edges[idx] if idx >=0 else self._input_min
        right = self._input_edges[idx+1] if idx + 1 < self._bucket_num else self._input_max
        offset_ratio = (rounded_feature - left) / (right - left)  # (-1, 1)

        norm_left = self._output_edges[idx] if idx >=0 else self._output_min
        norm_right = self._output_edges[idx+1] if idx + 1 < self._bucket_num else self._output_max
        norm_out = norm_left + offset_ratio * (norm_right - norm_left)  # 映射到归一化区间内

        return norm_out # 返回两个特征

    def denormalize(self, norm_value: float) -> int:
        """
        将归一化后的特征值反归一化为原始特征值
        """
        # 先找到桶
        rounded_norm_value = self.get_rounded_output_feature(norm_value)
        idx = np.digitize(rounded_norm_value, self._output_edges).astype(int) - 1  # 桶索引从0开始
        idx = np.clip(idx, -1, self._bucket_num - 1)  # 确保索引在有效范围内
        if hasattr(idx, 'astype'):
            idx = idx.astype(int)

        # 桶内反归一化
        left = self._output_edges[idx] if idx >=0 else self._output_min
        right = self._output_edges[idx+1] if idx + 1 < self._bucket_num else self._output_max
        offset_ratio = (rounded_norm_value - left) / (right - left)  # 计算百分比

        output_left = self._input_edges[idx] if idx >=0 else self._input_min
        output_right = self._input_edges[idx+1] if idx + 1 < self._bucket_num else self._input_max
        restored_feature = output_left + offset_ratio * (output_right - output_left)  # 映射回原始区间内
        return self.get_rounded_input_feature(round(restored_feature))  # 还原为


class TestContinuesBucketNormalizer(unittest.TestCase):
    """
    测试用例
    """
    def test_normalize_denormalize_edges(self):
        """
        测试区块生成函数
        """
        normalizer = ContinuesBucketNormalizer(0, 100, bucket_width=10, output_min=0, output_max=1.0)
        for test_feature in [-1, 0, 5, 7, 10, 11, 35, 89, 94, 100, 110]:
            test_rounded_feature = normalizer.get_rounded_input_feature(test_feature)
            norm_value = normalizer.normalize(test_feature)
            restored_feature = normalizer.denormalize(norm_value)
            self.assertEqual(test_rounded_feature, restored_feature)

    def test_normalize_denormalize_random(self):
        """
        测试随机特征值的归一化和反归一化
        """
        normalizer = ContinuesBucketNormalizer(0, 200, bucket_width=10, output_min=0, output_max=1.0)
        for _ in range(400):
            test_feature = np.random.randint(0, 200)
            test_rounded_feature = normalizer.get_rounded_input_feature(test_feature)
            norm_value = normalizer.normalize(test_feature)
            restored_feature = normalizer.denormalize(norm_value)
            self.assertEqual(test_rounded_feature, restored_feature)

    def test_special_lowerbound_bucket_case(self):
        """
        测试特殊情况
        """
        normalizer = ContinuesBucketNormalizer(1, 3, bucket_width=10, output_min=0, output_max=1.0)
        norm_value = 0.57837135
        restored_feature = normalizer.denormalize(norm_value)
        self.assertGreater(restored_feature, 1)

    def test_special_init_bucket_case(self):
        """
        测试特殊情况
        """
        normalizer = ContinuesBucketNormalizer(1, 2, bucket_width=10, output_min=0, output_max=1)
        norm_value = 0.5
        restored_feature = normalizer.denormalize(norm_value)
        self.assertEqual(restored_feature, 0)


class DiscreteBucketNormalizer(object):
    """
    用于对特征进行离散化处理
    """
    def __init__(self, bucket_values: List[Any], output_min: float = 0.0, output_max: float = 1.0):
        """
        构造函数
        """
        self._output_min = output_min
        self._output_max = output_max
        self._bucket_values = bucket_values
        self._bucket_num = len(bucket_values) + 1
        # bucket_width = (self._output_max - self._output_min) / self._bucket_num  # 每个桶的宽度
        # self._edges = np.linspace(self._output_min, self._output_max + bucket_width, self._bucket_num + 1)
        self._edges = np.linspace(self._output_min, self._output_max, self._bucket_num)
        self._eps = 1e-6
        self._edges[0] -= self._eps
        self._edges[-1] += self._eps


    def normalize(self, feature: Any) -> float:
        """
        对特征进行归一化处理
        :param feature: 特征值
        :return: 归一化后的特征值
        """
        if feature not in self._bucket_values:
            raise Exception(f"Feature {feature} is not in the bucket values {self._bucket_values}")

        # 确定桶索引
        idx = self._bucket_values.index(feature)  # 获取特征值对应的桶索引
        left = max(self._edges[idx], self._output_min)  # 获取桶的左边界
        right = min(self._edges[idx + 1], self._output_max)  # 获取桶的右边界
        idx_norm = np.random.uniform(left, right, 1).item()  # 在桶内随机生成一个归一化值
        # # 桶索引归一化 (避免0值)
        # idx_norm = (idx + 1) / (self._bucket_num + 1)  # 映射到(0,1)区间

        return idx_norm  # 返回一个特征

    def denormalize(self, idx_norm: float) -> Any:
        """
        将归一化后的特征值反归一化为原始特征值
        """
        # 还原桶索引
        # idx = idx_norm * (self._bucket_num + 1) - 1
        idx_0 = np.clip(idx_norm, self._output_min, self._output_max).item() # 确保索引在有效范围内
        idx = np.digitize(idx_0, self._edges).item() - 1  # 桶索引从0开始
        # idx = np.clip(idx, 0, self._bucket_num - 1).item() # 确保索引在有效范围内
        # 还原桶的值
        try:
            result = self._bucket_values[idx]
        except IndexError:
            raise Exception(f"Index {idx} is out of bounds for bucket values {self._bucket_values}")
        return result


class TestDiscreteBucketNormalizer(unittest.TestCase):
    """
    测试用例
    """
    def test_normalize_denormalize_edges(self):
        """
        测试区块生成函数
        """
        features = [0, 10, 11, 12, 13, 14, 15]
        normalizer = DiscreteBucketNormalizer(features)
        for test_feature in features:
            idx_norm = normalizer.normalize(test_feature)
            restored_feature = normalizer.denormalize(idx_norm)
            self.assertEqual(test_feature, restored_feature)


    def test_normalize_edge_overflow(self):
        """
        测试边界情况
        """
        with self.assertRaises(Exception):
            features = [0, 10, 11, 12, 13, 14, 15]
            normalizer = DiscreteBucketNormalizer(features)
            idx_norm = normalizer.normalize(110)
            restored_feature = normalizer.denormalize(idx_norm)
            self.assertEqual(restored_feature, 110)

    def test_denormalize_edge(self):
        """
        测试反虚拟化
        """
        features = [0, 10, 11, 12, 13, 14, 15]
        normalizer = DiscreteBucketNormalizer(features)
        for featurn_norm, feature in [(1.2, 15), (-1.2, 0)]:
            restored_feature = normalizer.denormalize(featurn_norm)
            self.assertEqual(restored_feature, feature)

    def test_normalize_packing(self):
        """
        测试边界情况
        """
        features = [0, 1]
        normalizer = DiscreteBucketNormalizer(features)
        idx_norm = normalizer.normalize(1)
        restored_feature = normalizer.denormalize(idx_norm)
        self.assertEqual(restored_feature, 1)

    def test_normalize_threading(self):
        """
        测试边界情况
        """
        features = [(1, 1, 1)]
        normalizer = DiscreteBucketNormalizer(features)
        idx_norm = normalizer.normalize((1, 1, 1))
        restored_feature = normalizer.denormalize(idx_norm)
        self.assertEqual(restored_feature, (1, 1, 1))

        idx_norm_1 = 2
        restored_feature_1 = normalizer.denormalize(idx_norm_1)
        self.assertEqual(restored_feature_1, (1, 1, 1))
