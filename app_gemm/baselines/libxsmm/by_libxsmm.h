//
// Created by huche on 2023/5/10 0010.
//

#ifndef MATRIX_COMPUTATION_RESEARCH_BY_LIBXSMM_H
#define MATRIX_COMPUTATION_RESEARCH_BY_LIBXSMM_H

#include "gemm_perf_define.h"
#include "libxsmm.h"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <cassert>

benchmark_result performance_benchmark_cr_ar_br_float32(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) {
    float *matrix_a = nullptr, *matrix_b = nullptr, *matrix_c = nullptr;
    int i = 0;
    double start=0, end=0, cost=0;
    benchmark_result benchmark = {
            .total_compute_time = 0,
            .cost = 0,
            .ops = 0,
            .benchmark = 0,
            .types = 0
    };
    float alpha = (float)(extra_params.alpha);
    float beta = (float)(extra_params.beta);
    GemmLoopOrderAroundKernels loop_order = extra_params.loop_order;
    int mc = extra_params.mc, kc = extra_params.kc, nc = extra_params.nc;
    int m_thread = extra_params.m_thread, k_thread = extra_params.k_thread, n_thread = extra_params.n_thread;
    int gemm_mat_c_prfm_skip_bytes = extra_params.gemm_mat_c_prfm_skip_bytes;
    int gemm_mat_a_prfm_skip_bytes = extra_params.gemm_mat_a_prfm_skip_bytes;
    int gemm_mat_b_prfm_skip_bytes = extra_params.gemm_mat_b_prfm_skip_bytes;
    int add_mat_c_prfm_skip_bytes = extra_params.add_mat_c_prfm_skip_bytes;
    int add_partial_sum_prfm_skip_bytes = extra_params.add_partial_sum_prfm_skip_bytes;

    matrix_a = new float[m * k];
    random_matrix(m,k,matrix_a);
    matrix_b = new float[k * n];
    random_matrix(k, n,matrix_b);
    matrix_c = new float[m * n];

    omp_set_num_threads(thread_num);

    libxsmm_blasint blas_m = m;
    libxsmm_blasint blas_n = n;
    libxsmm_blasint blas_k = k;
    const char notransa = 'n';
    const char notransb = 'n';
    libxsmm_init();

    const int libxsmm_xgemm_flags_ = LIBXSMM_GEMM_FLAGS(notransa, notransb) | LIBXSMM_GEMM_FLAG_BETA_0;


    const libxsmm_gemm_shape libxsmm_xgemm_shape_ = libxsmm_create_gemm_shape(
            blas_n, blas_m, blas_k, blas_n, blas_k, blas_n,
            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
#ifdef EPPA_GEMM_PREFETCH_BL2
    const libxsmm_gemmfunction libxsmm_xgemm_function_ = libxsmm_dispatch_gemm(libxsmm_xgemm_shape_,
                                                                               (libxsmm_bitfield)libxsmm_xgemm_flags_, LIBXSMM_GEMM_PREFETCH_BL2);
#else
    const libxsmm_gemmfunction libxsmm_xgemm_function_ = libxsmm_dispatch_gemm(libxsmm_xgemm_shape_,
                                                                               (libxsmm_bitfield)libxsmm_xgemm_flags_, LIBXSMM_GEMM_PREFETCH_BL1);
#endif

    if (libxsmm_xgemm_function_ == nullptr) {
        for (i = 0; i < 5; i++) {  // 使缓存热起来
            libxsmm_gemm(
                    &notransa,
                    &notransb,
                    &blas_n,
                    &blas_m,
                    &blas_k,
                    &alpha,
                    matrix_b,
                    &blas_n,  // B的列，与是否转置无关
                    matrix_a,
                    &blas_k,  // A的列，与是否转置无关
                    &beta,
                    matrix_c,
                    &blas_n  // C的列，与是否转置无关
            );
        }

        start = dclock();
        for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
            libxsmm_gemm(
                    &notransa,
                    &notransb,
                    &blas_n,
                    &blas_m,
                    &blas_k,
                    &alpha,
                    matrix_b,
                    &blas_n,  // B的列，与是否转置无关
                    matrix_a,
                    &blas_k,  // A的列，与是否转置无关
                    &beta,
                    matrix_c,
                    &blas_n  // C的列，与是否转置无关
            );
        }
        end = dclock();
    } else {
        benchmark.types = 1;
        libxsmm_gemm_param libxsmm_xgemm_param_;
        libxsmm_xgemm_param_.a.primary = matrix_b;
        libxsmm_xgemm_param_.b.primary = matrix_a;
        libxsmm_xgemm_param_.c.primary = matrix_c;

        for (i = 0; i < 5; i++) {  // 使缓存热起来
            libxsmm_xgemm_function_(&libxsmm_xgemm_param_);
        }

        start = dclock();
        for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
            libxsmm_xgemm_function_(&libxsmm_xgemm_param_);
        }
        end = dclock();
    }

    libxsmm_finalize();
    benchmark.total_compute_time = end-start;
    benchmark.cost = benchmark.total_compute_time /iteration_times;
    benchmark.ops = (double)m * n  * k * 2 * 1.0e-09;
#if CHECK_GEMM_RESULT_VAL
    if (check_result(StorageLayout::CR_AR_BR, matrix_c, matrix_a, matrix_b, m, k, n) != 0) {
        cout << "Result error!" << endl;
    }
#endif
    delete [] matrix_a;
    delete [] matrix_b;
    delete [] matrix_c;
    benchmark.benchmark = benchmark.ops / benchmark.cost;
    return benchmark;
}

benchmark_result performance_benchmark_cr_ar_bc_float32(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) {
    float *matrix_a = nullptr, *matrix_b = nullptr, *matrix_c = nullptr;
    int i = 0;
    double start=0, end=0, cost=0;
    benchmark_result benchmark = {
            .total_compute_time = 0,
            .cost = 0,
            .ops = 0,
            .benchmark = 0,
            .types = 0
    };
    float alpha = (float)(extra_params.alpha);
    float beta = (float)(extra_params.beta);
    GemmLoopOrderAroundKernels loop_order = extra_params.loop_order;
    int mc = extra_params.mc, kc = extra_params.kc, nc = extra_params.nc;
    int m_thread = extra_params.m_thread, k_thread = extra_params.k_thread, n_thread = extra_params.n_thread;
    int gemm_mat_c_prfm_skip_bytes = extra_params.gemm_mat_c_prfm_skip_bytes;
    int gemm_mat_a_prfm_skip_bytes = extra_params.gemm_mat_a_prfm_skip_bytes;
    int gemm_mat_b_prfm_skip_bytes = extra_params.gemm_mat_b_prfm_skip_bytes;
    int add_mat_c_prfm_skip_bytes = extra_params.add_mat_c_prfm_skip_bytes;
    int add_partial_sum_prfm_skip_bytes = extra_params.add_partial_sum_prfm_skip_bytes;

    matrix_a = new float[m * k];
    random_matrix(m,k,matrix_a);
    matrix_b = new float[k * n];
    random_matrix(n, k,matrix_b);
    matrix_c = new float[m * n];

    omp_set_num_threads(thread_num);

    libxsmm_blasint blas_m = m;
    libxsmm_blasint blas_n = n;
    libxsmm_blasint blas_k = k;
    const char notrans = 'n';
    const char trans = 't';
    libxsmm_init();

    const int libxsmm_xgemm_flags_ = LIBXSMM_GEMM_FLAGS(trans, notrans) | LIBXSMM_GEMM_FLAG_BETA_0;


    const libxsmm_gemm_shape libxsmm_xgemm_shape_ = libxsmm_create_gemm_shape(
            blas_n, blas_m, blas_k, blas_k, blas_k, blas_n,
            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);

#ifdef EPPA_GEMM_PREFETCH_BL2
    const libxsmm_gemmfunction libxsmm_xgemm_function_ = libxsmm_dispatch_gemm(libxsmm_xgemm_shape_,
                                                                               (libxsmm_bitfield)libxsmm_xgemm_flags_, LIBXSMM_GEMM_PREFETCH_BL2);
#else
    const libxsmm_gemmfunction libxsmm_xgemm_function_ = libxsmm_dispatch_gemm(libxsmm_xgemm_shape_,
                                                                               (libxsmm_bitfield)libxsmm_xgemm_flags_, LIBXSMM_GEMM_PREFETCH_BL1);
#endif

    if (libxsmm_xgemm_function_ == nullptr) {
        for (i = 0; i < 5; i++) {  // 使缓存热起来
            libxsmm_gemm(
                    &trans,
                    &notrans,
                    &blas_n,
                    &blas_m,
                    &blas_k,
                    &alpha,
                    matrix_b,
                    &blas_k,  // B的列，与是否转置无关
                    matrix_a,
                    &blas_k,  // A的列，与是否转置无关
                    &beta,
                    matrix_c,
                    &blas_n  // C的列，与是否转置无关
            );
        }

        start = dclock();
        for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
            libxsmm_gemm(
                    &trans,
                    &notrans,
                    &blas_n,
                    &blas_m,
                    &blas_k,
                    &alpha,
                    matrix_b,
                    &blas_k,  // B的列，与是否转置无关
                    matrix_a,
                    &blas_k,  // A的列，与是否转置无关
                    &beta,
                    matrix_c,
                    &blas_n  // C的列，与是否转置无关
            );
        }
        end = dclock();
    } else {
        benchmark.types = 1;
        libxsmm_gemm_param libxsmm_xgemm_param_;
        libxsmm_xgemm_param_.a.primary = matrix_b;
        libxsmm_xgemm_param_.b.primary = matrix_a;
        libxsmm_xgemm_param_.c.primary = matrix_c;

        for (i = 0; i < 5; i++) {  // 使缓存热起来
            libxsmm_xgemm_function_(&libxsmm_xgemm_param_);
        }

        start = dclock();
        for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
            libxsmm_xgemm_function_(&libxsmm_xgemm_param_);
        }
        end = dclock();
    }

    libxsmm_finalize();
    benchmark.total_compute_time = end-start;
    benchmark.cost = benchmark.total_compute_time /iteration_times;
    benchmark.ops = (double)m * n  * k * 2 * 1.0e-09;
    delete [] matrix_a;
    delete [] matrix_b;
    delete [] matrix_c;
    benchmark.benchmark = benchmark.ops / benchmark.cost;
    return benchmark;
}

benchmark_result performance_benchmark(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) {
    if ((extra_params.element_precision == ElementPrecision::EP_FLOAT32) && (extra_params.storage_layout == StorageLayout::CR_AR_BR)) {
        return performance_benchmark_cr_ar_br_float32(thread_num, iteration_times, m, n, k, extra_params);
    } else if ((extra_params.element_precision == ElementPrecision::EP_FLOAT32) && (extra_params.storage_layout == StorageLayout::CR_AR_BC)) {
        return performance_benchmark_cr_ar_bc_float32(thread_num, iteration_times, m, n, k, extra_params);
    } else {
        assert(false);
    }
}

#endif //MATRIX_COMPUTATION_RESEARCH_BY_LIBXSMM_H


