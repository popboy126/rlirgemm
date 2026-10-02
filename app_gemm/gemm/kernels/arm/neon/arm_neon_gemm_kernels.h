//
// Created by huche on 2025/12/22.
//

#ifndef APP_GEMM_ARM_NEON_GEMM_KERNELS_H
#define APP_GEMM_ARM_NEON_GEMM_KERNELS_H
#include <stdint.h>

namespace arm {
namespace neon {
#ifdef __cplusplus
    extern "C" {
#endif
        /// ARM NEON单精度浮点数矩阵乘法内核，行主序C，行主序A，行主序B, AB均不打包, MR=1, NR=64, KR=8
        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr64_kr8(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr16_kr8(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr1_kr5(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr5_kr5(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr7_kr1(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr8_kr8(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta
        );

        // 以下是打包内核
        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr16_kr8(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta,
            float* bc
        );

        void mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr2_nr4_kr4(
            float* C,
            const float* A,
            const float* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            float alpha,
            float beta,
            float* bc
        );


#ifdef __cplusplus
    }
#endif

}
}

#endif //APP_GEMM_ARM_NEON_GEMM_KERNELS_H
