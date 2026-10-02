# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/22 13:31
# Contact       : hucheng.lew@qq.com
# File          : precision_type.py
# IDE           : PyCharm
# Description   : 定义精度


from enum import Enum

class PrecisionType(Enum):
    """精度类型枚举"""
    FP8 = "fp8"        # 8位浮点数
    FP16 = "fp16"      # 半精度浮点数
    FP32 = "fp32"      # 单精度浮点数
    FP64 = "fp64"      # 双精度浮点数
    INT8 = "int8"      # 8位整数
    INT16 = "int16"    # 16位整数
    INT32 = "int32"    # 32位整数
    INT64 = "int64"    # 64位整数


class ElementPrecision:
    """元素精度定义"""
    def __init__(self, precision: PrecisionType):
        self._precision: PrecisionType = precision

    def get_dtype_c(self) -> str:
        """获取C语言数据类型"""
        dtype_map = {
            PrecisionType.FP8.value: 'uint8_t',      # FP8 (E4M3/E5M2)
            PrecisionType.FP16.value: '__fp16',      # FP16 (half precision)
            PrecisionType.FP32.value: 'float',
            PrecisionType.FP64.value: 'double',
            PrecisionType.INT8.value: 'int8_t',
            PrecisionType.INT16.value: 'int16_t',
            PrecisionType.INT32.value: 'int32_t',
            PrecisionType.INT64.value: 'int64_t',
        }
        return dtype_map.get(self.precision.value, 'float')

    def get_dtype_lsl_position(self) -> int:
        """获取数据类型左移位数(用于向量寄存器索引计算)"""
        lsl_map = {
            PrecisionType.FP8.value: 0,    # FP8
            PrecisionType.FP16.value: 1,  # FP16 (half)
            PrecisionType.FP32.value: 2,  # FP32 (float)
            PrecisionType.FP64.value: 3,  # FP64 (double)
            PrecisionType.INT8.value: 0,
            PrecisionType.INT16.value: 1,
            PrecisionType.INT32.value: 2,
            PrecisionType.INT64.value: 3,
        }
        return lsl_map.get(self.precision.value, 2)

    @property
    def short_name(self) -> str:
        """获取精度的简短名称"""
        return self.precision.value

    @property
    def precision(self) -> PrecisionType:
        """获取精度类型"""
        return self._precision

    def get_byte_size(self) -> int:
        """获取数据类型的字节大小"""
        byte_size_map = {
            PrecisionType.FP8.value: 1,    # FP8
            PrecisionType.FP16.value: 2,  # FP16 (half)
            PrecisionType.FP32.value: 4,  # FP32 (float)
            PrecisionType.FP64.value: 8,  # FP64 (double)
            PrecisionType.INT8.value: 1,
            PrecisionType.INT16.value: 2,
            PrecisionType.INT32.value: 4,
            PrecisionType.INT64.value: 8,
        }
        return byte_size_map.get(self.precision.value, 4)

    def get_bit_size(self) -> int:
        """获取数据类型的位大小"""
        return self.get_byte_size() * 8


def parse_precision(precision_str: str) -> ElementPrecision:
    """解析数据类型字符串"""
    precision_map = {
        'f8': PrecisionType.FP8,
        'f16': PrecisionType.FP16,
        'f32': PrecisionType.FP32,
        'f64': PrecisionType.FP64,
        'i8': PrecisionType.INT8,
        'i16': PrecisionType.INT16,
        'i32': PrecisionType.INT32,
        'i64': PrecisionType.INT64,
    }
    precision_type = precision_map.get(precision_str.lower())
    if not precision_type:
        raise ValueError(f"不支持的数据类型: {precision_str}")
    return ElementPrecision(precision_type)
