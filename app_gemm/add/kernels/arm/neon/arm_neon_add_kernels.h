//
// Created by huche on 2025/12/30.
//

#ifndef APP_GEMM_ARM_NEON_ADD_KERNELS_H
#define APP_GEMM_ARM_NEON_ADD_KERNELS_H

#include <stdint.h>

namespace arm {
namespace neon {
#ifdef __cplusplus
    extern "C" {
#endif
        /// ARM NEON单精度浮点数矩阵加法内核，行主序C，部分和行主序, MR=2, NR=16, KR=2
        void add_kernel_arm_neon_fp32_r_mr2_nr16_kr2(
            float* C,
            int64_t ldc,
            float** partial_sum_ptr,
            int64_t ldp,
            int64_t m,
            int64_t n,
            int64_t k,
            float alpha,
            float beta,
            int64_t c_prefetch_stride,
            int64_t partial_sum_prefetch_stride
        );

#ifdef __cplusplus
    }
#endif

}
}

#endif //APP_GEMM_ARM_NEON_ADD_KERNELS_H
