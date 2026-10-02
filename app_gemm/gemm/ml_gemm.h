//
// Created by huche on 2026/1/28.
//

#ifndef APP_GEMM_ML_GEMM_H
#define APP_GEMM_ML_GEMM_H

#include <stdint.h>
#include "ml_gemm_common.h"

#ifdef __cplusplus
    extern "C" {
#endif
        void do_gemm_compute_f32(
                unsigned short loop_order,
                float *C,
                int LDC,
                float *A,
                int LDA,
                float *B,
                int LDB,
                int M,
                int N,
                int K,
                float alpha,
                float beta,
                int mc, int kc, int nc,
                int m_thread, int k_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride,
                int add_mat_c_prfm_stride,
                int add_partial_sum_prfm_stride,
                unsigned short packing_type
        );

        void do_gemm_compute_simple_f32(
                float *C,
                int LDC,
                float *A,
                int LDA,
                float *B,
                int LDB,
                int M,
                int N,
                int K,
                float alpha,
                float beta
        );
#ifdef __cplusplus
    }
#endif


#endif //APP_GEMM_ML_GEMM_H
