//
// Created by huche on 2026/1/28.
//

#ifndef APP_GEMM_ML_GEMM_COMMON_H
#define APP_GEMM_ML_GEMM_COMMON_H

#ifdef __cplusplus
    extern "C" {
#endif
        enum GemmLoopOrderAroundKernels {
        LO_NONE = 0,
        LO_NM = 1,
        LO_MN = 2,
        LO_NKM = 3,
        LO_NMK = 4,
        LO_KNM = 5,
        LO_KMN = 6,
        LO_MKN = 7,
        LO_MNK = 8,
        LO_COUNT = 9
    };

    /// 矩阵元素精度枚举
    enum ElementPrecision {
        EP_FLOAT32 = 0,
        EP_FLOAT16 = 1,
        EP_FLOAT64 = 2,
        EP_INT8 = 3,
        EP_COUNT = 4
    };

    enum StorageLayout {
        CR_AR_BR = 0,  // 矩阵C为行主序，矩阵A、B为行主序
        CR_AR_BC = 1,  // 矩阵C为行主序，矩阵A为行主序，矩阵B为列主序
        CC_AC_BC = 2,  // 矩阵C为列主序，矩阵A、B为列主序
        NotSupport = 3 // 不支持
    };

    enum PackingType {
        PACKING_TYPE_NONE = 0,      // 不打包
        PACKING_TYPE_B_FP = 1,      // 细粒度打包矩阵B
        PACKING_TYPE_COUNT = 2,     // 打包类型数量
    };

    inline char* get_loop_order_string(GemmLoopOrderAroundKernels loop_order) {
        switch (loop_order) {
            case LO_NONE:
                return (char *)"LO_NONE";
            case LO_NM:
                return (char *)"LO_NM";
            case LO_MN:
                return (char *)"LO_MN";
            case LO_NKM:
                return (char *)"LO_NKM";
            case LO_NMK:
                return (char *)"LO_NMK";
            case LO_KNM:
                return (char *)"LO_KNM";
            case LO_KMN:
                return (char *)"LO_KMN";
            case LO_MKN:
                return (char *)"LO_MKN";
            case LO_MNK:
                return (char *)"LO_MNK";
            default:
                return (char *)"UNKNOWN_LOOP_ORDER";
        }
    }

    inline char* get_storage_order_string(StorageLayout storage_order) {
        switch (storage_order) {
            case CR_AR_BR:
                return (char *)"CR_AR_BR";
            case CR_AR_BC:
                return (char *)"CR_AR_BC";
            case CC_AC_BC:
                return (char *)"CC_AC_BC";
            default:
                return (char *)"UNKNOWN_STORAGE_ORDER";
        }
    }

    inline char* get_element_precision_string(ElementPrecision element_precision) {
        switch (element_precision) {
            case EP_FLOAT32:
                return (char *)"FP32";
            case EP_FLOAT16:
                return (char *)"FP16";
            case EP_FLOAT64:
                return (char *)"FP64";
            case EP_INT8:
                return (char *)"INT8";
            default:
                return (char *)"UNKNOWN_ELEMENT_PRECISION";
        }
    }

    inline char* get_packing_type_string(PackingType packing_type) {
        switch (packing_type) {
            case PACKING_TYPE_NONE:
                return (char *)"None";
            case PACKING_TYPE_B_FP:
                return (char *)"B_FP";
            default:
                return (char *)"Unknow";
        }
    }
#ifdef __cplusplus
    }
#endif


#endif //APP_GEMM_ML_GEMM_COMMON_H
