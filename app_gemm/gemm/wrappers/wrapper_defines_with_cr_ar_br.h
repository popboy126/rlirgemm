//
// Created by huche on 2025/12/17.
//

#ifndef APP_GEMM_WRAPPER_DEFINES_H
#define APP_GEMM_WRAPPER_DEFINES_H
#include "ml_gemm_common.h"
// #include <cstdint>
#include <cassert>
// #include <cstring>
#include <float.h>
#include <omp.h>
#include <cmath>
#include <iostream>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "arm_neon_gemm_kernels.h"
#include "arm_neon_add_kernels.h"
#endif

namespace storage_layout_cr_ar_br {
        template <typename T>
        bool floats_are_equal_nextafter(T a, T b) {
            // 处理特殊情况，如 NaN 和 Inf
            if (a == b) return true;
            if (a == 0.0f || b == 0.0f) return std::fabs(a - b) < FLT_EPSILON;
            T diff = std::fabs(a - b);
            // 计算在 a 和 b 的数量级上，一个单位精度（ULP, Unit in the Last Place）的大小
            T next_diff = std::fabs(std::nextafter(a, b) - a);

            // 允许几个 ULP 的误差
            return diff < next_diff * 4; // 4 是一个经验值，可以根据需要调整
        }

        /// 获取GEMM内核的最大NR值
        inline unsigned get_gemm_kernel_max_nr() {
            return 16;
        }

        template <typename T>
        void gemm_kernel_scheduler_with_packing_none(
            T* C,
            const T* A,
            const T* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            T alpha,
            T beta
        ) {
            // 默认实现, 优化的版本通过函数重载实现
            if (floats_are_equal_nextafter(beta, 0.0f)) {
                for (int row = 0; row < m; row ++) {
                    for (int col = 0; col < n; col ++) {
                        int idx_c = row * ldc + col;
                        double sum = 0;
                        for (int i = 0; i < k; i ++) {
                            int idx_a = row * lda + i;
                            int idx_b = i * ldb + col;
                            sum += A[idx_a] * B[idx_b];
                        }
                        C[idx_c] = alpha * sum;
                    }
                }

            } else {
                for (int row = 0; row < m; row ++) {
                    for (int col = 0; col < n; col ++) {
                        int idx_c = row * ldc + col;
                        double sum = 0;
                        for (int i = 0; i < k; i ++) {
                            int idx_a = row * lda + i;
                            int idx_b = i * ldb + col;
                            sum += A[idx_a] * B[idx_b];
                        }
                        C[idx_c] = alpha * sum + beta * C[idx_c];
                    }
                }
            }
        }
#if defined(__aarch64__) && defined(__ARM_NEON)
    inline void gemm_kernel_scheduler_with_packing_none(
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
        ) {
            // if ((m < 5) && (n >= 64)) {
            //     arm::neon::mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr64_kr8(
            //         C,
            //         A,
            //         B,
            //         m, n, k,
            //         ldc, lda, ldb,
            //         c_prefetch_stride,
            //         a_prefetch_stride,
            //         b_prefetch_stride,
            //         alpha,
            //         beta
            //     );
            // } else {
            //     arm::neon::mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr16_kr8(
            //         C,
            //         A,
            //         B,
            //         m, n, k,
            //         ldc, lda, ldb,
            //         c_prefetch_stride,
            //         a_prefetch_stride,
            //         b_prefetch_stride,
            //         alpha,
            //         beta
            //     );
            // }

            arm::neon::mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr16_kr8(
                    C,
                    A,
                    B,
                    m, n, k,
                    ldc, lda, ldb,
                    c_prefetch_stride,
                    a_prefetch_stride,
                    b_prefetch_stride,
                    alpha,
                    beta
                );
        }
#endif

    template <typename T>
        void gemm_kernel_scheduler_with_packing_bfp(
            T* C,
            const T* A,
            const T* B,
            int64_t m, int64_t n, int64_t k,
            int64_t ldc, int64_t lda, int64_t ldb,
            int64_t c_prefetch_stride,
            int64_t a_prefetch_stride,
            int64_t b_prefetch_stride,
            T alpha,
            T beta,
            T* bc
        ) {
            gemm_kernel_scheduler_with_packing_none(
                    C,
                    A,
                    B,
                    m, n, k,
                    ldc, lda, ldb,
                    c_prefetch_stride,
                    a_prefetch_stride,
                    b_prefetch_stride,
                    alpha,
                    beta
            );
        }

#if defined(__aarch64__) && defined(__ARM_NEON)
    inline void gemm_kernel_scheduler_with_packing_bfp(
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
        ) {
            arm::neon::mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr16_kr8(
                    C,
                    A,
                    B,
                    m, n, k,
                    ldc, lda, ldb,
                    c_prefetch_stride,
                    a_prefetch_stride,
                    b_prefetch_stride,
                    alpha,
                    beta,
                    bc
            );
        }
#endif

    template <typename T>
    void add_kernel_scheduler(
        T* C,
        int64_t ldc,  // 矩阵C的真实列数
        T** partial_sum_ptr,
        int64_t ldp, // 部分和的真实列数
        int64_t m, // 部分和的行数
        int64_t n, // 部分和的列数
        int64_t k, // 部分和的个数
        T alpha,
        T beta,
        int64_t mat_c_prfm_stride,
        int64_t partial_sum_prfm_stride
    ) {
            // todo 后续需要替换这部分逻辑
            for (int row = 0; row < m; row ++) {
                for (int col = 0; col < n; col ++) {
                    int idx_c = row * ldc + col;
                    int idx_p = row * ldp + col;
                    double sum = 0;
                    for (int i = 0; i < k; i ++) {
                        auto partial_sum_ptr_i = partial_sum_ptr[i];
                        sum += partial_sum_ptr_i[idx_p];
                        // if (std::isnan(partial_sum_ptr_i[idx_p])) {
                        //     std::cout << "Error: PartialSUm[" << row << ", " << col << "] is NaN!" << std::endl;
                        // }
                    }
                    C[idx_c] = alpha * sum + beta * C[idx_c];
                }
            }
        }
#if defined(__aarch64__) && defined(__ARM_NEON)
    void add_kernel_scheduler(
        float* C,
        int64_t ldc,  // 矩阵C的真实列数
        float** partial_sum_ptr,
        int64_t ldp, // 部分和的真实列数
        int64_t m, // 部分和的行数
        int64_t n, // 部分和的列数
        int64_t k, // 部分和的个数
        float alpha,
        float beta,
        int64_t mat_c_prfm_stride,
        int64_t partial_sum_prfm_stride
    ) {
            arm::neon::add_kernel_arm_neon_fp32_r_mr2_nr16_kr2(
                 C, ldc,
                 partial_sum_ptr, ldp,
                m, n, k, alpha, beta,
                mat_c_prfm_stride, partial_sum_prfm_stride
            );
        }
#endif

    template <typename T>
    void gemm_compute_with_loop_order_none_packing_none(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mat_c_prfm_skip_bytes,
            int mat_a_prfm_skip_bytes,
            int mat_b_prfm_skip_bytes
    ) {
            gemm_kernel_scheduler_with_packing_none(
                    C,
                    A,
                    B,
                    M, N, K,
                    LDC, LDA, LDB,
                    mat_c_prfm_skip_bytes,
                    mat_a_prfm_skip_bytes,
                    mat_b_prfm_skip_bytes,
                    alpha,
                    beta
            );
    }

    template <typename T>
    void gemm_compute_with_loop_order_none_packing_bfp(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mat_c_prfm_skip_bytes,
            int mat_a_prfm_skip_bytes,
            int mat_b_prfm_skip_bytes
    ) {
            const unsigned max_nr = get_gemm_kernel_max_nr();
            T* bc = new T[K * max_nr]; // B packed buffer
            gemm_kernel_scheduler_with_packing_bfp(
                    C,
                    A,
                    B,
                    M, N, K,
                    LDC, LDA, LDB,
                    mat_c_prfm_skip_bytes,
                    mat_a_prfm_skip_bytes,
                    mat_b_prfm_skip_bytes,
                    alpha,
                    beta,
                    bc
            );
            delete [] bc;
        }

        template <typename T>
        void gemm_compute_with_loop_order_nm_packing_none(
                T *C,
                int LDC,
                T *A,
                int LDA,
                T *B,
                int LDB,
                int M,
                int N,
                int K,
                T alpha,
                T beta,
                int mc, int kc, int nc,
                int m_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * n_thread;
            #pragma omp parallel num_threads(thread_num) \
                        default(none) \
                        shared(C) \
                        firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                        alpha, beta, \
                        mc, kc, nc, \
                        m_thread, n_thread, \
                        gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride)
                {
                    int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                    int thread_id_m = thread_id / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                    int thread_id_n = thread_id % n_thread;  // N维度的索引

                    int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                    int dim_m_start = thread_id_m * dim_m_per_thread;
                    int dim_m_end = dim_m_start + dim_m_per_thread;
                    if (dim_m_end > M) {
                        dim_m_end = M;
                    }

                    int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                    int dim_n_start = thread_id_n * dim_n_per_thread;
                    int dim_n_end = dim_n_start + dim_n_per_thread;
                    if (dim_n_end > N) {
                        dim_n_end = N;
                    }

                    int current_dim_m = 0, current_dim_n = dim_n_start;
                    // Step 1: 先遍历N维度
                    int real_mc = 0, real_nc = 0;
                    float *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }

                        // Step 2: 再遍历M维度
                        current_dim_m = dim_m_start;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }

                            mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                            mat_a_block_ptr = A + current_dim_m * LDA;
                            mat_b_block_ptr = B + current_dim_n;

                            gemm_kernel_scheduler_with_packing_none(
                                mat_c_block_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, K,
                                LDC,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                alpha, beta
                            );

                            current_dim_m += real_mc;
                        }

                        current_dim_n += real_nc;
                    }
                }
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_nm_packing_bfp(
                T *C,
                int LDC,
                T *A,
                int LDA,
                T *B,
                int LDB,
                int M,
                int N,
                int K,
                T alpha,
                T beta,
                int mc, int kc, int nc,
                int m_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * n_thread;
            #pragma omp parallel num_threads(thread_num) \
                        default(none) \
                        shared(C) \
                        firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                        alpha, beta, \
                        mc, kc, nc, \
                        m_thread, n_thread, \
                        gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride)
                {
                    const unsigned max_nr = get_gemm_kernel_max_nr();
                    T* bc = new T[K * max_nr]; // B packed buffer
                    int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                    int thread_id_m = thread_id / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                    int thread_id_n = thread_id % n_thread;  // N维度的索引

                    int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                    int dim_m_start = thread_id_m * dim_m_per_thread;
                    int dim_m_end = dim_m_start + dim_m_per_thread;
                    if (dim_m_end > M) {
                        dim_m_end = M;
                    }

                    int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                    int dim_n_start = thread_id_n * dim_n_per_thread;
                    int dim_n_end = dim_n_start + dim_n_per_thread;
                    if (dim_n_end > N) {
                        dim_n_end = N;
                    }

                    int current_dim_m = 0, current_dim_n = dim_n_start;
                    // Step 1: 先遍历N维度
                    int real_mc = 0, real_nc = 0;
                    float *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }

                        // Step 2: 再遍历M维度
                        current_dim_m = dim_m_start;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }

                            mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                            mat_a_block_ptr = A + current_dim_m * LDA;
                            mat_b_block_ptr = B + current_dim_n;

                            gemm_kernel_scheduler_with_packing_bfp(
                                mat_c_block_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, K,
                                LDC,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                alpha, beta,
                                bc
                            );

                            current_dim_m += real_mc;
                        }

                        current_dim_n += real_nc;
                    }
                    delete[] bc;
                }
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_mn_packing_none(
                T *C,
                int LDC,
                T *A,
                int LDA,
                T *B,
                int LDB,
                int M,
                int N,
                int K,
                T alpha,
                T beta,
                int mc, int kc, int nc,
                int m_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * n_thread;
            #pragma omp parallel num_threads(thread_num) \
                        default(none) \
                        shared(C) \
                        firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                        alpha, beta, \
                        mc, kc, nc, \
                        m_thread, n_thread, \
                        gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride)
                {
                    int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                    int thread_id_m = thread_id / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                    int thread_id_n = thread_id % n_thread;  // N维度的索引

                    int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                    int dim_m_start = thread_id_m * dim_m_per_thread;
                    int dim_m_end = dim_m_start + dim_m_per_thread;
                    if (dim_m_end > M) {
                        dim_m_end = M;
                    }

                    int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                    int dim_n_start = thread_id_n * dim_n_per_thread;
                    int dim_n_end = dim_n_start + dim_n_per_thread;
                    if (dim_n_end > N) {
                        dim_n_end = N;
                    }

                    int current_dim_m = dim_m_start, current_dim_n = 0;
                    // Step 1: 先遍历N维度
                    int real_mc = 0, real_nc = 0;
                    float *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                    while(current_dim_m < dim_m_end) {
                        if (current_dim_m + mc > dim_m_end) {
                            real_mc = dim_m_end - current_dim_m;
                        } else {
                            real_mc = mc;
                        }


                        current_dim_n = dim_n_start;
                        while(current_dim_n < dim_n_end) {
                            if (current_dim_n + nc > dim_n_end) {
                                real_nc = dim_n_end - current_dim_n;
                            } else {
                                real_nc = nc;
                            }

                            mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                            mat_a_block_ptr = A + current_dim_m * LDA;
                            mat_b_block_ptr = B + current_dim_n;

                            gemm_kernel_scheduler_with_packing_none(
                                mat_c_block_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, K,
                                LDC,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                alpha, beta
                            );

                           current_dim_n += real_nc;
                        }
                        current_dim_m += real_mc;
                    }
                }
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_mn_packing_bfp(
                T *C,
                int LDC,
                T *A,
                int LDA,
                T *B,
                int LDB,
                int M,
                int N,
                int K,
                T alpha,
                T beta,
                int mc, int kc, int nc,
                int m_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * n_thread;
            #pragma omp parallel num_threads(thread_num) \
                        default(none) \
                        shared(C) \
                        firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                        alpha, beta, \
                        mc, kc, nc, \
                        m_thread, n_thread, \
                        gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride)
                {
                    const unsigned max_nr = get_gemm_kernel_max_nr();
                    T* bc = new T[K * max_nr]; // B packed buffer
                    int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                    int thread_id_m = thread_id / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                    int thread_id_n = thread_id % n_thread;  // N维度的索引

                    int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                    int dim_m_start = thread_id_m * dim_m_per_thread;
                    int dim_m_end = dim_m_start + dim_m_per_thread;
                    if (dim_m_end > M) {
                        dim_m_end = M;
                    }

                    int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                    int dim_n_start = thread_id_n * dim_n_per_thread;
                    int dim_n_end = dim_n_start + dim_n_per_thread;
                    if (dim_n_end > N) {
                        dim_n_end = N;
                    }

                    int current_dim_m = dim_m_start, current_dim_n = 0;
                    // Step 1: 先遍历N维度
                    int real_mc = 0, real_nc = 0;
                    float *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                    while(current_dim_m < dim_m_end) {
                        if (current_dim_m + mc > dim_m_end) {
                            real_mc = dim_m_end - current_dim_m;
                        } else {
                            real_mc = mc;
                        }


                        current_dim_n = dim_n_start;
                        while(current_dim_n < dim_n_end) {
                            if (current_dim_n + nc > dim_n_end) {
                                real_nc = dim_n_end - current_dim_n;
                            } else {
                                real_nc = nc;
                            }

                            mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                            mat_a_block_ptr = A + current_dim_m * LDA;
                            mat_b_block_ptr = B + current_dim_n;

                            gemm_kernel_scheduler_with_packing_bfp(
                                mat_c_block_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, K,
                                LDC,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                alpha, beta,
                                bc
                            );

                           current_dim_n += real_nc;
                        }
                        current_dim_m += real_mc;
                    }
                    delete [] bc;
                }
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_nkm_packing_none(
                T *C,
                int LDC,
                T *A,
                int LDA,
                T *B,
                int LDB,
                int M,
                int N,
                int K,
                T alpha,
                T beta,
                int mc, int kc, int nc,
                int m_thread, int k_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride,
                int add_mat_c_prfm_stride,
                int add_partial_sum_prfm_stride
        ) {
                // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [n_thread * k_thread]; // 用于存放部分和数组地址

        #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int km_thread = k_thread * m_thread;   // K维度和M维度的线程总数
                int thread_id_n = thread_id / km_thread;  // N维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., n_thread - 1
                int thread_id_km = thread_id % km_thread;  // K维度和M维度的索引
                int thread_id_k = thread_id_km % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                int thread_id_m = thread_id_km / k_thread;  // m维度的索引

                bool partial_sum_allocated_thread = (thread_id_m == 0); // 标计是否为分配部分和的线程, 每组k的第一个N维度为0的线程申请部分和空间
                const int partial_sum_array_index = thread_id_n * k_thread + thread_id_k;  // 部分和数组的索引

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = M;
                const int partial_sum_col = dim_n_per_thread > nc ? nc : dim_n_per_thread;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[partial_sum_array_index] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

        #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[partial_sum_array_index] + thread_id_m * dim_m_per_thread * partial_sum_col;

                int partial_sum_m_offset = 0;  // m维度的索引

                int current_dim_m = dim_m_start, current_dim_n = dim_n_start, current_dim_k = dim_k_start;
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;

                // Step 1: 先遍历N维度
                while(current_dim_n < dim_n_end) {
                    if (current_dim_n + nc > dim_n_end) {
                        real_nc = dim_n_end - current_dim_n;
                    } else {
                        real_nc = nc;
                    }

                    // Step 2: 再遍历K维度
                    current_dim_k = dim_k_start;
                    real_gemm_beta = 0;
                    while (current_dim_k < dim_k_end) {
                        if (current_dim_k + kc > dim_k_end) {
                            real_kc = dim_k_end - current_dim_k;
                        } else {
                            real_kc = kc;
                        }

                        mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                        // Step 3: 再遍历M维度
                        current_dim_m = dim_m_start;
                        partial_sum_m_offset = 0;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }


                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_m_offset * partial_sum_col;

                            gemm_kernel_scheduler_with_packing_none(
                                partital_sum_ptr,
                                mat_a_block_ptr,
                                mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col, LDA, LDB,
                                gemm_mat_c_prfm_stride,
                                gemm_mat_a_prfm_stride,
                                gemm_mat_b_prfm_stride,
                                1.0f,
                                real_gemm_beta
                            );
                            current_dim_m += real_mc;
                            partial_sum_m_offset += real_mc;
                        }
                        current_dim_k += real_kc;
                        real_gemm_beta = 1;
                    }

                    // 开始累加，等待所有线程都计算完
                    #pragma omp barrier
                    if (thread_id_km == 0) {
                        mat_c_block_ptr = C + thread_id_m * dim_m_per_thread * LDC + current_dim_n;
                        add_kernel_scheduler<T>(   // K维度并行的放缩交由累加内核来做
                            mat_c_block_ptr,
                            LDC,  // 矩阵C的真实列数
                            &(partial_sum_array[partial_sum_array_index]),
                            partial_sum_col, // 部分和的真实列数
                            partial_sum_row, // 部分和的行数
                            real_nc, // 部分和的列数
                            k_thread, // 部分和的个数
                            alpha,
                            beta,
                            add_mat_c_prfm_stride,
                            add_partial_sum_prfm_stride
                        );

                    }
                    #pragma omp barrier

                    // if (partial_sum_allocated_thread) {
                    //     memset((void *)(partial_sum_array[partial_sum_array_index]), 0, total_partial_sum_size * sizeof(T));
                    // }
                    // #pragma omp barrier

                    current_dim_n += real_nc;

                }

                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[partial_sum_array_index];
                }
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_nkm_packing_bfp(
                T *C,
                int LDC,
                T *A,
                int LDA,
                T *B,
                int LDB,
                int M,
                int N,
                int K,
                T alpha,
                T beta,
                int mc, int kc, int nc,
                int m_thread, int k_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride,
                int add_mat_c_prfm_stride,
                int add_partial_sum_prfm_stride
        ) {
                // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [n_thread * k_thread]; // 用于存放部分和数组地址

        #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                const unsigned max_nr = get_gemm_kernel_max_nr();
                T* bc = new T[kc * max_nr]; // B packed buffer
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int km_thread = k_thread * m_thread;   // K维度和M维度的线程总数
                int thread_id_n = thread_id / km_thread;  // N维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., n_thread - 1
                int thread_id_km = thread_id % km_thread;  // K维度和M维度的索引
                int thread_id_k = thread_id_km % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                int thread_id_m = thread_id_km / k_thread;  // m维度的索引

                bool partial_sum_allocated_thread = (thread_id_m == 0); // 标计是否为分配部分和的线程, 每组k的第一个N维度为0的线程申请部分和空间
                const int partial_sum_array_index = thread_id_n * k_thread + thread_id_k;  // 部分和数组的索引

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = M;
                const int partial_sum_col = dim_n_per_thread > nc ? nc : dim_n_per_thread;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[partial_sum_array_index] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

        #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[partial_sum_array_index] + thread_id_m * dim_m_per_thread * partial_sum_col;

                int partial_sum_m_offset = 0;  // m维度的索引

                int current_dim_m = dim_m_start, current_dim_n = dim_n_start, current_dim_k = dim_k_start;
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;

                // Step 1: 先遍历N维度
                while(current_dim_n < dim_n_end) {
                    if (current_dim_n + nc > dim_n_end) {
                        real_nc = dim_n_end - current_dim_n;
                    } else {
                        real_nc = nc;
                    }

                    // Step 2: 再遍历K维度
                    current_dim_k = dim_k_start;
                    real_gemm_beta = 0;
                    while (current_dim_k < dim_k_end) {
                        if (current_dim_k + kc > dim_k_end) {
                            real_kc = dim_k_end - current_dim_k;
                        } else {
                            real_kc = kc;
                        }

                        mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                        // Step 3: 再遍历M维度
                        current_dim_m = dim_m_start;
                        partial_sum_m_offset = 0;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }


                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_m_offset * partial_sum_col;

                            gemm_kernel_scheduler_with_packing_bfp(
                                partital_sum_ptr,
                                mat_a_block_ptr,
                                mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col, LDA, LDB,
                                gemm_mat_c_prfm_stride,
                                gemm_mat_a_prfm_stride,
                                gemm_mat_b_prfm_stride,
                                1.0f,
                                real_gemm_beta,
                                bc
                            );
                            current_dim_m += real_mc;
                            partial_sum_m_offset += real_mc;
                        }
                        current_dim_k += real_kc;
                        real_gemm_beta = 1;
                    }

                    // 开始累加，等待所有线程都计算完
                    #pragma omp barrier
                    if (thread_id_km == 0) {
                        mat_c_block_ptr = C + thread_id_m * dim_m_per_thread * LDC + current_dim_n;
                        add_kernel_scheduler<T>(   // K维度并行的放缩交由累加内核来做
                            mat_c_block_ptr,
                            LDC,  // 矩阵C的真实列数
                            &(partial_sum_array[partial_sum_array_index]),
                            partial_sum_col, // 部分和的真实列数
                            partial_sum_row, // 部分和的行数
                            real_nc, // 部分和的列数
                            k_thread, // 部分和的个数
                            alpha,
                            beta,
                            add_mat_c_prfm_stride,
                            add_partial_sum_prfm_stride
                        );

                    }
                    #pragma omp barrier

                    // if (partial_sum_allocated_thread) {
                    //     memset((void *)(partial_sum_array[partial_sum_array_index]), 0, total_partial_sum_size * sizeof(T));
                    // }
                    // #pragma omp barrier

                    current_dim_n += real_nc;

                }

                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[partial_sum_array_index];
                }
                delete [] bc;
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_nmk_packing_none(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [thread_num]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                        default(none) \
                                        shared(C, partial_sum_array) \
                                        firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                        alpha, beta, \
                                        mc, kc, nc, \
                                        m_thread, k_thread, n_thread, \
                                        gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                        add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
                {
                    int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                    int mk_thread = m_thread * k_thread;   // K维度和M维度的线程总数
                    int thread_id_n = thread_id / mk_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                    int thread_id_mk = thread_id % mk_thread;  // K维度和N维度的索引
                    int thread_id_k = thread_id_mk % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                    int thread_id_m = thread_id_mk / k_thread;  // M维度的索引

                    int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                    int dim_m_start = thread_id_m * dim_m_per_thread;
                    int dim_m_end = dim_m_start + dim_m_per_thread;
                    if (dim_m_end > M) {
                        dim_m_end = M;
                    }

                    int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                    int dim_n_start = thread_id_n * dim_n_per_thread;
                    int dim_n_end = dim_n_start + dim_n_per_thread;
                    if (dim_n_end > N) {
                        dim_n_end = N;
                    }

                    int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                    int dim_k_start = thread_id_k * dim_k_per_thread;
                    int dim_k_end = dim_k_start + dim_k_per_thread;
                    if (dim_k_end > K) {
                        dim_k_end = K;
                    }

                    T* partital_sum_ptr = new T [mc * nc];
                    partial_sum_array[thread_id] = partital_sum_ptr;

                    int current_dim_m = dim_m_start, current_dim_n = dim_n_start, current_dim_k = dim_k_start;
                    int real_mc = 0, real_kc = 0, real_nc = 0;
                    T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                    T real_gemm_beta = 0;

                    // Step 1: 先遍历N维度
                    current_dim_n = dim_n_start;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }

                        // Step 2: 再遍历M维度
                        current_dim_m = dim_m_start;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }

                            // Step 3: 最后遍历K维度
                            current_dim_k = dim_k_start;
                            real_gemm_beta = 0;
                            while (current_dim_k < dim_k_end) {
                                if (current_dim_k + kc > dim_k_end) {
                                    real_kc = dim_k_end - current_dim_k;
                                } else {
                                    real_kc = kc;
                                }
                                mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                                mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                                gemm_kernel_scheduler_with_packing_none(
                                    partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                    real_mc, real_nc, real_kc,
                                    nc,  LDA,  LDB,
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                    1.0f, real_gemm_beta
                                );

                                current_dim_k += real_kc;
                                real_gemm_beta = 1;
                            }

                            // 开始累加，等待所有线程都计算完
                            #pragma omp barrier
                            if (thread_id_k == 0) {
                                mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                                add_kernel_scheduler(
                                     mat_c_block_ptr,
                                     LDC,  // 矩阵C的真实列数
                                     &(partial_sum_array[thread_id]),
                                     nc, // 部分和的真实列数
                                     real_mc, // 部分和的行数
                                     real_nc, // 部分和的列数
                                     k_thread, // 部分和的个数
                                     alpha,
                                     beta,
                                     add_mat_c_prfm_stride,
                                     add_partial_sum_prfm_stride
                                );
                            }
                            #pragma omp barrier

                            current_dim_m += real_mc;
                        }
                        current_dim_n += real_nc;
                    }

                    delete[] partital_sum_ptr;
                }

            delete[] partial_sum_array;
            //#pragma omp flush

        }

        template <typename T>
        void gemm_compute_with_loop_order_nmk_packing_bfp(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [thread_num]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                        default(none) \
                                        shared(C, partial_sum_array) \
                                        firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                        alpha, beta, \
                                        mc, kc, nc, \
                                        m_thread, k_thread, n_thread, \
                                        gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                        add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
                {
                    const unsigned max_nr = get_gemm_kernel_max_nr();
                    T* bc = new T[kc * max_nr]; // B packed buffer
                    int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                    int mk_thread = m_thread * k_thread;   // K维度和M维度的线程总数
                    int thread_id_n = thread_id / mk_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                    int thread_id_mk = thread_id % mk_thread;  // K维度和N维度的索引
                    int thread_id_k = thread_id_mk % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                    int thread_id_m = thread_id_mk / k_thread;  // M维度的索引

                    int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                    int dim_m_start = thread_id_m * dim_m_per_thread;
                    int dim_m_end = dim_m_start + dim_m_per_thread;
                    if (dim_m_end > M) {
                        dim_m_end = M;
                    }

                    int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                    int dim_n_start = thread_id_n * dim_n_per_thread;
                    int dim_n_end = dim_n_start + dim_n_per_thread;
                    if (dim_n_end > N) {
                        dim_n_end = N;
                    }

                    int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                    int dim_k_start = thread_id_k * dim_k_per_thread;
                    int dim_k_end = dim_k_start + dim_k_per_thread;
                    if (dim_k_end > K) {
                        dim_k_end = K;
                    }

                    T* partital_sum_ptr = new T [mc * nc];
                    partial_sum_array[thread_id] = partital_sum_ptr;

                    int current_dim_m = dim_m_start, current_dim_n = dim_n_start, current_dim_k = dim_k_start;
                    int real_mc = 0, real_kc = 0, real_nc = 0;
                    T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                    T real_gemm_beta = 0;

                    // Step 1: 先遍历N维度
                    current_dim_n = dim_n_start;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }

                        // Step 2: 再遍历M维度
                        current_dim_m = dim_m_start;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }

                            // Step 3: 最后遍历K维度
                            current_dim_k = dim_k_start;
                            real_gemm_beta = 0;
                            while (current_dim_k < dim_k_end) {
                                if (current_dim_k + kc > dim_k_end) {
                                    real_kc = dim_k_end - current_dim_k;
                                } else {
                                    real_kc = kc;
                                }
                                mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                                mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                                gemm_kernel_scheduler_with_packing_bfp(
                                    partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                    real_mc, real_nc, real_kc,
                                    nc,  LDA,  LDB,
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                    1.0f, real_gemm_beta,
                                    bc
                                );

                                current_dim_k += real_kc;
                                real_gemm_beta = 1;
                            }

                            // 开始累加，等待所有线程都计算完
                            #pragma omp barrier
                            if (thread_id_k == 0) {
                                mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                                add_kernel_scheduler(
                                     mat_c_block_ptr,
                                     LDC,  // 矩阵C的真实列数
                                     &(partial_sum_array[thread_id]),
                                     nc, // 部分和的真实列数
                                     real_mc, // 部分和的行数
                                     real_nc, // 部分和的列数
                                     k_thread, // 部分和的个数
                                     alpha,
                                     beta,
                                     add_mat_c_prfm_stride,
                                     add_partial_sum_prfm_stride
                                );
                            }
                            #pragma omp barrier

                            current_dim_m += real_mc;
                        }
                        current_dim_n += real_nc;
                    }

                    delete[] partital_sum_ptr;
                    delete[] bc;
                }

            delete[] partial_sum_array;
            //#pragma omp flush

        }

        template <typename T>
        void gemm_compute_with_loop_order_knm_packing_none(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [k_thread]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int mn_thread = m_thread * n_thread;   // M维度和N维度的线程总数
                int thread_id_k = thread_id / mn_thread;  // K维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., k_thread - 1
                int thread_id_mn = thread_id % mn_thread; // M和N维度的索引
                int thread_id_m = thread_id_mn / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_n = thread_id_mn % n_thread;  // N维度的索引

                bool partial_sum_allocated_thread = ((thread_id % mn_thread) == 0); // 标计是否为分配部分和的线程

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = M;
                const int partial_sum_col = N;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[thread_id_k] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

        #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[thread_id_k] + thread_id_m * dim_m_per_thread * partial_sum_col + thread_id_n * dim_n_per_thread;

                int partial_sum_m_offset = 0, partial_sum_n_offset = 0;

                int current_dim_m = 0, current_dim_n = 0, current_dim_k = dim_k_start;
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                // Step 1: 先遍历K维度
                while (current_dim_k < dim_k_end) {
                    if (current_dim_k + kc > dim_k_end) {
                        real_kc = dim_k_end - current_dim_k;
                    } else {
                        real_kc = kc;
                    }

                    // Step 2: 再遍历N维度
                    current_dim_n = dim_n_start;
                    partial_sum_n_offset = 0;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }

                        mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                        // Step 3: 再遍历M维度
                        current_dim_m = dim_m_start;
                        partial_sum_m_offset = 0;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }


                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_m_offset * partial_sum_col + partial_sum_n_offset;

                            gemm_kernel_scheduler_with_packing_none(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta
                            );

                            current_dim_m += real_mc;
                            partial_sum_m_offset += real_mc;
                        }

                        current_dim_n += real_nc;
                        partial_sum_n_offset += real_nc;
                    }

                    current_dim_k += real_kc;
                    real_gemm_beta = 1;
                }

                // 开始累加，等待所有线程都计算完
        #pragma omp barrier
                if (thread_id == 0) {
                    mat_c_block_ptr = C;
                    add_kernel_scheduler(
                         mat_c_block_ptr,
                         LDC,  // 矩阵C的真实列数
                         &(partial_sum_array[thread_id]),
                         N, // 部分和的真实列数
                         M, // 部分和的行数
                         N, // 部分和的列数
                         k_thread, // 部分和的个数
                         alpha,
                         beta,
                         add_mat_c_prfm_stride,
                         add_partial_sum_prfm_stride
                    );
                }
        #pragma omp barrier
                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[thread_id_k];
                }
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_knm_packing_bfp(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [k_thread]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                const unsigned max_nr = get_gemm_kernel_max_nr();
                T* bc = new T[kc * max_nr]; // B packed buffer
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int mn_thread = m_thread * n_thread;   // M维度和N维度的线程总数
                int thread_id_k = thread_id / mn_thread;  // K维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., k_thread - 1
                int thread_id_mn = thread_id % mn_thread; // M和N维度的索引
                int thread_id_m = thread_id_mn / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_n = thread_id_mn % n_thread;  // N维度的索引

                bool partial_sum_allocated_thread = ((thread_id % mn_thread) == 0); // 标计是否为分配部分和的线程

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = M;
                const int partial_sum_col = N;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[thread_id_k] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

        #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[thread_id_k] + thread_id_m * dim_m_per_thread * partial_sum_col + thread_id_n * dim_n_per_thread;

                int partial_sum_m_offset = 0, partial_sum_n_offset = 0;

                int current_dim_m = 0, current_dim_n = 0, current_dim_k = dim_k_start;
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                // Step 1: 先遍历K维度
                while (current_dim_k < dim_k_end) {
                    if (current_dim_k + kc > dim_k_end) {
                        real_kc = dim_k_end - current_dim_k;
                    } else {
                        real_kc = kc;
                    }

                    // Step 2: 再遍历N维度
                    current_dim_n = dim_n_start;
                    partial_sum_n_offset = 0;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }

                        mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                        // Step 3: 再遍历M维度
                        current_dim_m = dim_m_start;
                        partial_sum_m_offset = 0;
                        while(current_dim_m < dim_m_end) {
                            if (current_dim_m + mc > dim_m_end) {
                                real_mc = dim_m_end - current_dim_m;
                            } else {
                                real_mc = mc;
                            }


                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_m_offset * partial_sum_col + partial_sum_n_offset;

                            gemm_kernel_scheduler_with_packing_bfp(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta,
                                bc
                            );

                            current_dim_m += real_mc;
                            partial_sum_m_offset += real_mc;
                        }

                        current_dim_n += real_nc;
                        partial_sum_n_offset += real_nc;
                    }

                    current_dim_k += real_kc;
                    real_gemm_beta = 1;
                }

                // 开始累加，等待所有线程都计算完
        #pragma omp barrier
                if (thread_id == 0) {
                    mat_c_block_ptr = C;
                    add_kernel_scheduler(
                         mat_c_block_ptr,
                         LDC,  // 矩阵C的真实列数
                         &(partial_sum_array[thread_id]),
                         N, // 部分和的真实列数
                         M, // 部分和的行数
                         N, // 部分和的列数
                         k_thread, // 部分和的个数
                         alpha,
                         beta,
                         add_mat_c_prfm_stride,
                         add_partial_sum_prfm_stride
                    );
                }
        #pragma omp barrier
                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[thread_id_k];
                }
                delete[] bc;
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_kmn_packing_none(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [k_thread]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int mn_thread = m_thread * n_thread;   // M维度和N维度的线程总数
                int thread_id_k = thread_id / mn_thread;  // K维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., k_thread - 1
                int thread_id_mn = thread_id % mn_thread; // M和N维度的索引
                int thread_id_m = thread_id_mn / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_n = thread_id_mn % n_thread;  // N维度的索引

                bool partial_sum_allocated_thread = ((thread_id % mn_thread) == 0); // 标计是否为分配部分和的线程

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = M;
                const int partial_sum_col = N;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[thread_id_k] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

                #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[thread_id_k] + thread_id_m * dim_m_per_thread * partial_sum_col + thread_id_n * dim_n_per_thread;

                int partial_sum_m_offset = 0, partial_sum_n_offset = 0;

                int current_dim_m = 0, current_dim_n = 0, current_dim_k = dim_k_start;
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                // Step 1: 先遍历K维度
                while (current_dim_k < dim_k_end) {
                    if (current_dim_k + kc > dim_k_end) {
                        real_kc = dim_k_end - current_dim_k;
                    } else {
                        real_kc = kc;
                    }

                    // Step 2: 再遍历M维度
                    current_dim_m = dim_m_start;
                    partial_sum_m_offset = 0;
                    while(current_dim_m < dim_m_end) {
                        if (current_dim_m + mc > dim_m_end) {
                            real_mc = dim_m_end - current_dim_m;
                        } else {
                            real_mc = mc;
                        }

                        // Step 2: 再遍历N维度
                        current_dim_n = dim_n_start;
                        partial_sum_n_offset = 0;
                        while(current_dim_n < dim_n_end) {
                            if (current_dim_n + nc > dim_n_end) {
                                real_nc = dim_n_end - current_dim_n;
                            } else {
                                real_nc = nc;
                            }

                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_m_offset * partial_sum_col + partial_sum_n_offset;

                            gemm_kernel_scheduler_with_packing_none(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta
                            );

                            current_dim_n += real_nc;
                            partial_sum_n_offset += real_nc;
                        }
                        current_dim_m += real_mc;
                        partial_sum_m_offset += real_mc;
                    }

                    current_dim_k += real_kc;
                    real_gemm_beta = 1;
                }

                // 开始累加，等待所有线程都计算完
        #pragma omp barrier
                if (thread_id == 0) {
                    mat_c_block_ptr = C;
                    add_kernel_scheduler(
                         mat_c_block_ptr,
                         LDC,  // 矩阵C的真实列数
                         &(partial_sum_array[thread_id]),
                         N, // 部分和的真实列数
                         M, // 部分和的行数
                         N, // 部分和的列数
                         k_thread, // 部分和的个数
                         alpha,
                         beta,
                         add_mat_c_prfm_stride,
                         add_partial_sum_prfm_stride
                    );
                }
        #pragma omp barrier
                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[thread_id_k];
                }
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_kmn_packing_bfp(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [k_thread]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                const unsigned max_nr = get_gemm_kernel_max_nr();
                T* bc = new T[kc * max_nr]; // B packed buffer
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int mn_thread = m_thread * n_thread;   // M维度和N维度的线程总数
                int thread_id_k = thread_id / mn_thread;  // K维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., k_thread - 1
                int thread_id_mn = thread_id % mn_thread; // M和N维度的索引
                int thread_id_m = thread_id_mn / n_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_n = thread_id_mn % n_thread;  // N维度的索引

                bool partial_sum_allocated_thread = ((thread_id % mn_thread) == 0); // 标计是否为分配部分和的线程

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = M;
                const int partial_sum_col = N;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[thread_id_k] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

                #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[thread_id_k] + thread_id_m * dim_m_per_thread * partial_sum_col + thread_id_n * dim_n_per_thread;

                int partial_sum_m_offset = 0, partial_sum_n_offset = 0;

                int current_dim_m = 0, current_dim_n = 0, current_dim_k = dim_k_start;
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                // Step 1: 先遍历K维度
                while (current_dim_k < dim_k_end) {
                    if (current_dim_k + kc > dim_k_end) {
                        real_kc = dim_k_end - current_dim_k;
                    } else {
                        real_kc = kc;
                    }

                    // Step 2: 再遍历M维度
                    current_dim_m = dim_m_start;
                    partial_sum_m_offset = 0;
                    while(current_dim_m < dim_m_end) {
                        if (current_dim_m + mc > dim_m_end) {
                            real_mc = dim_m_end - current_dim_m;
                        } else {
                            real_mc = mc;
                        }

                        // Step 2: 再遍历N维度
                        current_dim_n = dim_n_start;
                        partial_sum_n_offset = 0;
                        while(current_dim_n < dim_n_end) {
                            if (current_dim_n + nc > dim_n_end) {
                                real_nc = dim_n_end - current_dim_n;
                            } else {
                                real_nc = nc;
                            }

                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_m_offset * partial_sum_col + partial_sum_n_offset;

                            gemm_kernel_scheduler_with_packing_bfp(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta,
                                bc
                            );

                            current_dim_n += real_nc;
                            partial_sum_n_offset += real_nc;
                        }
                        current_dim_m += real_mc;
                        partial_sum_m_offset += real_mc;
                    }

                    current_dim_k += real_kc;
                    real_gemm_beta = 1;
                }

                // 开始累加，等待所有线程都计算完
        #pragma omp barrier
                if (thread_id == 0) {
                    mat_c_block_ptr = C;
                    add_kernel_scheduler(
                         mat_c_block_ptr,
                         LDC,  // 矩阵C的真实列数
                         &(partial_sum_array[thread_id]),
                         N, // 部分和的真实列数
                         M, // 部分和的行数
                         N, // 部分和的列数
                         k_thread, // 部分和的个数
                         alpha,
                         beta,
                         add_mat_c_prfm_stride,
                         add_partial_sum_prfm_stride
                    );
                }
        #pragma omp barrier
                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[thread_id_k];
                }

                delete[] bc;
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_mkn_packing_none(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [m_thread * k_thread]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int kn_thread = k_thread * n_thread;   // K维度和N维度的线程总数
                int thread_id_m = thread_id / kn_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_kn = thread_id % kn_thread;  // K维度和N维度的索引
                int thread_id_k = thread_id_kn % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                int thread_id_n = thread_id_kn / k_thread;  // N维度的索引

                bool partial_sum_allocated_thread = (thread_id_n == 0); // 标计是否为分配部分和的线程, 每组k的第一个N维度为0的线程申请部分和空间
                const int partial_sum_array_index = thread_id_m * k_thread + thread_id_k;  // 部分和数组的索引

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = dim_m_per_thread > mc ? mc : dim_m_per_thread;
                const int partial_sum_col = N;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[partial_sum_array_index] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

        #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[partial_sum_array_index] + thread_id_n * dim_n_per_thread;

                int partial_sum_n_offset = 0;  // m, n维度的索引

                int current_dim_m = dim_m_start, current_dim_n = 0, current_dim_k = dim_k_start;
                // Step 1: 先遍历M维度
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                while(current_dim_m < dim_m_end) {
                    if (current_dim_m + mc > dim_m_end) {
                        real_mc = dim_m_end - current_dim_m;
                    } else {
                        real_mc = mc;
                    }

                    // Step 2: 再遍历K维度
                    current_dim_k = dim_k_start;
                    real_gemm_beta = 0;
                    while (current_dim_k < dim_k_end) {
                        if (current_dim_k + kc > dim_k_end) {
                            real_kc = dim_k_end - current_dim_k;
                        } else {
                            real_kc = kc;
                        }
                        mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;

                        // Step 3: 最后遍历N维度
                        current_dim_n = dim_n_start;
                        partial_sum_n_offset = 0;
                        while(current_dim_n < dim_n_end) {
                            if (current_dim_n + nc > dim_n_end) {
                                real_nc = dim_n_end - current_dim_n;
                            } else {
                                real_nc = nc;
                            }

                            mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;
                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_n_offset;

                            gemm_kernel_scheduler_with_packing_none(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta
                            );

                            current_dim_n += real_nc;
                            partial_sum_n_offset += real_nc;

                        }
                        current_dim_k += real_kc;
                        real_gemm_beta = 1;
                    }

                    // 开始累加，等待所有线程都计算完
        #pragma omp barrier
                    if (thread_id_kn == 0) {
                        mat_c_block_ptr = C + current_dim_m * LDC + thread_id_n * dim_n_per_thread;
                        add_kernel_scheduler(
                             mat_c_block_ptr,
                             LDC,  // 矩阵C的真实列数
                             &(partial_sum_array[partial_sum_array_index]),
                             partial_sum_col, // 部分和的真实列数
                             real_mc, // 部分和的行数
                             partial_sum_col, // 部分和的列数
                             k_thread, // 部分和的个数
                             alpha,
                             beta,
                             add_mat_c_prfm_stride,
                             add_partial_sum_prfm_stride
                        );
                    }
        #pragma omp barrier
        //
        //             if (partial_sum_allocated_thread) {
        //                 memset((void *)(partial_sum_array[partial_sum_array_index]), 0, total_partial_sum_size * sizeof(T));
        //             }
        // #pragma omp barrier
                    current_dim_m += real_mc;
                }

                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[partial_sum_array_index];
                }
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_mkn_packing_bfp(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [m_thread * k_thread]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                const unsigned max_nr = get_gemm_kernel_max_nr();
                T* bc = new T[kc * max_nr]; // B packed buffer
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int kn_thread = k_thread * n_thread;   // K维度和N维度的线程总数
                int thread_id_m = thread_id / kn_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_kn = thread_id % kn_thread;  // K维度和N维度的索引
                int thread_id_k = thread_id_kn % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                int thread_id_n = thread_id_kn / k_thread;  // N维度的索引

                bool partial_sum_allocated_thread = (thread_id_n == 0); // 标计是否为分配部分和的线程, 每组k的第一个N维度为0的线程申请部分和空间
                const int partial_sum_array_index = thread_id_m * k_thread + thread_id_k;  // 部分和数组的索引

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                const int partial_sum_row = dim_m_per_thread > mc ? mc : dim_m_per_thread;
                const int partial_sum_col = N;
                const int total_partial_sum_size = partial_sum_row * partial_sum_col;
                T* partital_sum_base_ptr = nullptr, *partital_sum_ptr = nullptr;
                if (partial_sum_allocated_thread) {
                    partital_sum_base_ptr = new T [total_partial_sum_size];
                    partial_sum_array[partial_sum_array_index] = partital_sum_base_ptr;
                    // memset((void *)partital_sum_base_ptr, 0, total_partial_sum_size * sizeof(T));
                }

        #pragma omp barrier         // 等待K维度的线程分配好内存
                partital_sum_base_ptr = partial_sum_array[partial_sum_array_index] + thread_id_n * dim_n_per_thread;

                int partial_sum_n_offset = 0;  // m, n维度的索引

                int current_dim_m = dim_m_start, current_dim_n = 0, current_dim_k = dim_k_start;
                // Step 1: 先遍历M维度
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                while(current_dim_m < dim_m_end) {
                    if (current_dim_m + mc > dim_m_end) {
                        real_mc = dim_m_end - current_dim_m;
                    } else {
                        real_mc = mc;
                    }

                    // Step 2: 再遍历K维度
                    current_dim_k = dim_k_start;
                    real_gemm_beta = 0;
                    while (current_dim_k < dim_k_end) {
                        if (current_dim_k + kc > dim_k_end) {
                            real_kc = dim_k_end - current_dim_k;
                        } else {
                            real_kc = kc;
                        }
                        mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;

                        // Step 3: 最后遍历N维度
                        current_dim_n = dim_n_start;
                        partial_sum_n_offset = 0;
                        while(current_dim_n < dim_n_end) {
                            if (current_dim_n + nc > dim_n_end) {
                                real_nc = dim_n_end - current_dim_n;
                            } else {
                                real_nc = nc;
                            }

                            mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;
                            partital_sum_ptr = partital_sum_base_ptr + partial_sum_n_offset;

                            gemm_kernel_scheduler_with_packing_bfp(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                partial_sum_col,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta,
                                bc
                            );

                            current_dim_n += real_nc;
                            partial_sum_n_offset += real_nc;

                        }
                        current_dim_k += real_kc;
                        real_gemm_beta = 1;
                    }

                    // 开始累加，等待所有线程都计算完
        #pragma omp barrier
                    if (thread_id_kn == 0) {
                        mat_c_block_ptr = C + current_dim_m * LDC + thread_id_n * dim_n_per_thread;
                        add_kernel_scheduler(
                             mat_c_block_ptr,
                             LDC,  // 矩阵C的真实列数
                             &(partial_sum_array[partial_sum_array_index]),
                             partial_sum_col, // 部分和的真实列数
                             real_mc, // 部分和的行数
                             partial_sum_col, // 部分和的列数
                             k_thread, // 部分和的个数
                             alpha,
                             beta,
                             add_mat_c_prfm_stride,
                             add_partial_sum_prfm_stride
                        );
                    }
        #pragma omp barrier
        //
        //             if (partial_sum_allocated_thread) {
        //                 memset((void *)(partial_sum_array[partial_sum_array_index]), 0, total_partial_sum_size * sizeof(T));
        //             }
        // #pragma omp barrier
                    current_dim_m += real_mc;
                }

                if (partial_sum_allocated_thread) {
                    delete[] partial_sum_array[partial_sum_array_index];
                }

                delete [] bc;
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_mnk_packing_none(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [thread_num]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int kn_thread = k_thread * n_thread;   // K维度和N维度的线程总数
                int thread_id_m = thread_id / kn_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_kn = thread_id % kn_thread;  // K维度和N维度的索引
                int thread_id_k = thread_id_kn % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                int thread_id_n = thread_id_kn / k_thread;  // N维度的索引

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                T* partital_sum_ptr = new T [mc * nc];
                partial_sum_array[thread_id] = partital_sum_ptr;

                int current_dim_m = dim_m_start, current_dim_n = 0, current_dim_k = dim_k_start;
                // Step 1: 先遍历M维度
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                while(current_dim_m < dim_m_end) {
                    if (current_dim_m + mc > dim_m_end) {
                        real_mc = dim_m_end - current_dim_m;
                    } else {
                        real_mc = mc;
                    }

                    // Step 2: 再遍历N维度
                    current_dim_n = dim_n_start;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }
                        // Step 3: 最后遍历K维度
                        current_dim_k = dim_k_start;
                        real_gemm_beta = 0;
                        while (current_dim_k < dim_k_end) {
                            if (current_dim_k + kc > dim_k_end) {
                                real_kc = dim_k_end - current_dim_k;
                            } else {
                                real_kc = kc;
                            }
                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                            gemm_kernel_scheduler_with_packing_none(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                nc,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta
                            );

                            current_dim_k += real_kc;
                            real_gemm_beta = 1;
                        }
                        // 开始累加，等待所有线程都计算完
                        #pragma omp barrier
                        if (thread_id_k == 0) {
                            mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                            add_kernel_scheduler(
                                 mat_c_block_ptr,
                                 LDC,  // 矩阵C的真实列数
                                 &(partial_sum_array[thread_id]),
                                 nc, // 部分和的真实列数
                                 real_mc, // 部分和的行数
                                 real_nc, // 部分和的列数
                                 k_thread, // 部分和的个数
                                 alpha,
                                 beta,
                                 add_mat_c_prfm_stride,
                                 add_partial_sum_prfm_stride
                            );
                        }
                        #pragma omp barrier
                        current_dim_n += real_nc;
                    }
                    current_dim_m += real_mc;
                }

                delete[] partital_sum_ptr;
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void gemm_compute_with_loop_order_mnk_packing_bfp(
            T *C,
            int LDC,
            T *A,
            int LDA,
            T *B,
            int LDB,
            int M,
            int N,
            int K,
            T alpha,
            T beta,
            int mc, int kc, int nc,
            int m_thread, int k_thread, int n_thread,
            int gemm_mat_c_prfm_stride,
            int gemm_mat_a_prfm_stride,
            int gemm_mat_b_prfm_stride,
            int add_mat_c_prfm_stride,
            int add_partial_sum_prfm_stride
        ) {
            // 计算并行线程
            int thread_num = m_thread * k_thread * n_thread;
            T** partial_sum_array = new T* [thread_num]; // 用于存放部分和数组地址

            #pragma omp parallel num_threads(thread_num) \
                                    default(none) \
                                    shared(C, partial_sum_array) \
                                    firstprivate(LDC, A, LDA, B, LDB, M, N, K, \
                                    alpha, beta, \
                                    mc, kc, nc, \
                                    m_thread, k_thread, n_thread, \
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, \
                                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride)
            {
                const unsigned max_nr = get_gemm_kernel_max_nr();
                T* bc = new T[kc * max_nr]; // B packed buffer
                int thread_id = omp_get_thread_num();  // 线程分配依然采用行主序，即从一行来看，线程序号为 0, 1, 2, 3, ...,
                int kn_thread = k_thread * n_thread;   // K维度和N维度的线程总数
                int thread_id_m = thread_id / kn_thread;  // M维度的索引, 因为是整除的关系，一定会是 0, 1, 2, ..., m_thread - 1
                int thread_id_kn = thread_id % kn_thread;  // K维度和N维度的索引
                int thread_id_k = thread_id_kn % k_thread;  // K维度的索引, 线程grid呈现出列优化排列，方面方面做累加
                int thread_id_n = thread_id_kn / k_thread;  // N维度的索引

                int dim_m_per_thread = (M + m_thread - 1) / m_thread; // m维度可划分的块数，含不完整的块数
                int dim_m_start = thread_id_m * dim_m_per_thread;
                int dim_m_end = dim_m_start + dim_m_per_thread;
                if (dim_m_end > M) {
                    dim_m_end = M;
                }

                int dim_n_per_thread = (N + n_thread - 1) / n_thread; // n维度可划分的块数，含不完整的块数
                int dim_n_start = thread_id_n * dim_n_per_thread;
                int dim_n_end = dim_n_start + dim_n_per_thread;
                if (dim_n_end > N) {
                    dim_n_end = N;
                }

                int dim_k_per_thread = (K + k_thread - 1) / k_thread; // k维度可划分的块数，含不完整的块数
                int dim_k_start = thread_id_k * dim_k_per_thread;
                int dim_k_end = dim_k_start + dim_k_per_thread;
                if (dim_k_end > K) {
                    dim_k_end = K;
                }

                T* partital_sum_ptr = new T [mc * nc];
                partial_sum_array[thread_id] = partital_sum_ptr;

                int current_dim_m = dim_m_start, current_dim_n = 0, current_dim_k = dim_k_start;
                // Step 1: 先遍历M维度
                int real_mc = 0, real_kc = 0, real_nc = 0;
                T *mat_a_block_ptr = nullptr, *mat_b_block_ptr = nullptr, *mat_c_block_ptr = nullptr;
                T real_gemm_beta = 0;
                while(current_dim_m < dim_m_end) {
                    if (current_dim_m + mc > dim_m_end) {
                        real_mc = dim_m_end - current_dim_m;
                    } else {
                        real_mc = mc;
                    }

                    // Step 2: 再遍历N维度
                    current_dim_n = dim_n_start;
                    while(current_dim_n < dim_n_end) {
                        if (current_dim_n + nc > dim_n_end) {
                            real_nc = dim_n_end - current_dim_n;
                        } else {
                            real_nc = nc;
                        }
                        // Step 3: 最后遍历K维度
                        current_dim_k = dim_k_start;
                        real_gemm_beta = 0;
                        while (current_dim_k < dim_k_end) {
                            if (current_dim_k + kc > dim_k_end) {
                                real_kc = dim_k_end - current_dim_k;
                            } else {
                                real_kc = kc;
                            }
                            mat_a_block_ptr = A + current_dim_m * LDA + current_dim_k;
                            mat_b_block_ptr = B + current_dim_k * LDB + current_dim_n;

                            gemm_kernel_scheduler_with_packing_bfp(
                                partital_sum_ptr, mat_a_block_ptr, mat_b_block_ptr,
                                real_mc, real_nc, real_kc,
                                nc,  LDA,  LDB,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                1.0f, real_gemm_beta,
                                bc
                            );

                            current_dim_k += real_kc;
                            real_gemm_beta = 1;
                        }
                        // 开始累加，等待所有线程都计算完
                        #pragma omp barrier
                        if (thread_id_k == 0) {
                            mat_c_block_ptr = C + current_dim_m * LDC + current_dim_n;
                            add_kernel_scheduler(
                                 mat_c_block_ptr,
                                 LDC,  // 矩阵C的真实列数
                                 &(partial_sum_array[thread_id]),
                                 nc, // 部分和的真实列数
                                 real_mc, // 部分和的行数
                                 real_nc, // 部分和的列数
                                 k_thread, // 部分和的个数
                                 alpha,
                                 beta,
                                 add_mat_c_prfm_stride,
                                 add_partial_sum_prfm_stride
                            );
                        }
                        #pragma omp barrier
                        current_dim_n += real_nc;
                    }
                    current_dim_m += real_mc;
                }

                delete[] partital_sum_ptr;
                delete[] bc;
            }

            delete[] partial_sum_array;
            //#pragma omp flush
        }

        template <typename T>
        void do_gemm_compute(
                GemmLoopOrderAroundKernels loop_order,
                T *C,
                int LDC,
                T *A,
                int LDA,
                T *B,
                int LDB,
                int M,
                int N,
                int K,
                T alpha,
                T beta,
                int mc, int kc, int nc,
                int m_thread, int k_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride,
                int add_mat_c_prfm_stride,
                int add_partial_sum_prfm_stride,
                PackingType packing_type
        ) {
            switch (loop_order) {
                case GemmLoopOrderAroundKernels::LO_NONE: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_none_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                             gemm_mat_c_prfm_stride,
                            gemm_mat_a_prfm_stride,
                            gemm_mat_b_prfm_stride
                            );
                            break;
                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_none_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                             gemm_mat_c_prfm_stride,
                            gemm_mat_a_prfm_stride,
                            gemm_mat_b_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                case GemmLoopOrderAroundKernels::LO_NM: {
                        switch (packing_type) {
                            case PackingType::PACKING_TYPE_NONE: {
                                gemm_compute_with_loop_order_nm_packing_none(
                                    C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                    mc, kc, nc, m_thread, n_thread,
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride
                                );
                                break;
                            }
                            case PackingType::PACKING_TYPE_B_FP: {
                                gemm_compute_with_loop_order_nm_packing_bfp(
                                    C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                    mc, kc, nc, m_thread, n_thread,
                                    gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride
                                );
                                break;
                            }
                            default: {
                                assert(false);
                                break;
                            }
                        }
                        break;
                }
                case GemmLoopOrderAroundKernels::LO_MN: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_mn_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride
                            );
                            break;

                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_mn_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                case GemmLoopOrderAroundKernels::LO_NKM: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_nkm_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;

                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_nkm_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                case GemmLoopOrderAroundKernels::LO_NMK: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_nmk_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_nmk_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                case GemmLoopOrderAroundKernels::LO_KNM: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_knm_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;

                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_knm_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                case GemmLoopOrderAroundKernels::LO_KMN: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_kmn_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_kmn_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                case GemmLoopOrderAroundKernels::LO_MKN: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_mkn_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;

                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_mkn_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                case GemmLoopOrderAroundKernels::LO_MNK: {
                    switch (packing_type) {
                        case PackingType::PACKING_TYPE_NONE: {
                            gemm_compute_with_loop_order_mnk_packing_none(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        case PackingType::PACKING_TYPE_B_FP: {
                            gemm_compute_with_loop_order_mnk_packing_bfp(
                                C, LDC, A, LDA, B, LDB, M, N, K, alpha, beta,
                                mc, kc, nc, m_thread, k_thread, n_thread,
                                gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
                                add_mat_c_prfm_stride, add_partial_sum_prfm_stride
                            );
                            break;
                        }
                        default: {
                            assert(false);
                            break;
                        }
                    }
                    break;
                }
                default: {
                        assert(false);
                        break;
                }
            }
        }

}

#endif //APP_GEMM_WRAPPER_DEFINES_H
