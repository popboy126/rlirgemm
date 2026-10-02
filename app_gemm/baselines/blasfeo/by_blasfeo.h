//
// Created by huche on 2023/5/10 0010.
//

#ifndef MATRIX_COMPUTATION_RESEARCH_BY_BLASFEO_H
#define MATRIX_COMPUTATION_RESEARCH_BY_BLASFEO_H

#include "gemm_perf_define.h"
#include "blasfeo.h"
#include <cassert>

template <typename T>
void blasfeo_random_matrix( int m, int n, blasfeo_smat &a)
{
    double drand48();
    int i,j;

    for ( i=0; i<m; i++ )
        for ( j=0; j<n; j++ )
            BLASFEO_SMATEL(&a, i, j) = 2.0 * (T)drand48( ) - 1.0 ;
}

benchmark_result performance_benchmark_cr_ar_br_float32(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) {
    blasfeo_smat matrix_a, matrix_b, matrix_c;
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

    int matrix_a_size = (int)blasfeo_memsize_smat(m, k);
    void *matrix_a_mem_align = nullptr;
    v_zeros_align(&matrix_a_mem_align, matrix_a_size);
    blasfeo_create_smat(m, k, &matrix_a, matrix_a_mem_align);
    blasfeo_random_matrix<float>(m,k,matrix_a);

    int matrix_b_size = (int)blasfeo_memsize_smat(k, n);
    void *matrix_b_mem_align = nullptr;
    v_zeros_align(&matrix_b_mem_align, matrix_b_size);
    blasfeo_create_smat(k, n, &matrix_b, matrix_b_mem_align);
    blasfeo_random_matrix<float>(k, n,matrix_b);

    int matrix_c_size = (int)blasfeo_memsize_smat(m, n);
    void *matrix_c_mem_align = nullptr;
    v_zeros_align(&matrix_c_mem_align, matrix_c_size);
    blasfeo_create_smat(m, n, &matrix_c, matrix_c_mem_align);

    for (i = 0; i < 5; i++) {  // 使缓存热起来
        blasfeo_sgemm_nn(
                m,
                n,
                k,
                alpha,
                &matrix_a,
                0,
                0,
                &matrix_b,
                0,
                0,
                beta,
                &matrix_c,
                0,
                0,
                &matrix_c,
                0,
                0
        );
    }

    start = dclock();
    for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
        blasfeo_sgemm_nn(
                m,
                n,
                k,
                alpha,
                &matrix_a,
                0,
                0,
                &matrix_b,
                0,
                0,
                beta,
                &matrix_c,
                0,
                0,
                &matrix_c,
                0,
                0
        );
    }
    end = dclock();
    benchmark.total_compute_time = end-start;
    benchmark.cost = benchmark.total_compute_time /iteration_times;
    benchmark.ops = (double)m * n  * k * 2 * 1.0e-09;
    v_free_align(matrix_a_mem_align);
    v_free_align(matrix_b_mem_align);
    v_free_align(matrix_c_mem_align);
    benchmark.benchmark = benchmark.ops / benchmark.cost;
    return benchmark;
}

benchmark_result performance_benchmark_cr_ar_bc_float32(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) {
    blasfeo_smat matrix_a, matrix_b, matrix_c;
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

    int matrix_a_size = (int)blasfeo_memsize_smat(m, k);
    void *matrix_a_mem_align = nullptr;
    v_zeros_align(&matrix_a_mem_align, matrix_a_size);
    blasfeo_create_smat(m, k, &matrix_a, matrix_a_mem_align);
    blasfeo_random_matrix<float>(m,k,matrix_a);

    int matrix_b_size = (int)blasfeo_memsize_smat(n, k);
    void *matrix_b_mem_align = nullptr;
    v_zeros_align(&matrix_b_mem_align, matrix_b_size);
    blasfeo_create_smat(n, k, &matrix_b, matrix_b_mem_align);
    blasfeo_random_matrix<float>(n, k, matrix_b);

    int matrix_c_size = (int)blasfeo_memsize_smat(m, n);
    void *matrix_c_mem_align = nullptr;
    v_zeros_align(&matrix_c_mem_align, matrix_c_size);
    blasfeo_create_smat(m, n, &matrix_c, matrix_c_mem_align);

    for (i = 0; i < 5; i++) {  // 使缓存热起来
        blasfeo_sgemm_nt(
                m,
                n,
                k,
                alpha,
                &matrix_a,
                0,
                0,
                &matrix_b,
                0,
                0,
                beta,
                &matrix_c,
                0,
                0,
                &matrix_c,
                0,
                0
        );
    }

    start = dclock();
    for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
        blasfeo_sgemm_nt(
                m,
                n,
                k,
                alpha,
                &matrix_a,
                0,
                0,
                &matrix_b,
                0,
                0,
                beta,
                &matrix_c,
                0,
                0,
                &matrix_c,
                0,
                0
        );
    }
    end = dclock();
    benchmark.total_compute_time = end-start;
    benchmark.cost = benchmark.total_compute_time /iteration_times;
    benchmark.ops = (double)m * n  * k * 2 * 1.0e-09;
    v_free_align(matrix_a_mem_align);
    v_free_align(matrix_b_mem_align);
    v_free_align(matrix_c_mem_align);
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

#endif //MATRIX_COMPUTATION_RESEARCH_BY_BLASFEO_H


