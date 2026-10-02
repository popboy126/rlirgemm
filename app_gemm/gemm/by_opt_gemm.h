//
// Created by huche on 2025/12/18.
//

#ifndef APP_GEMM_BY_OPT_GEMM_H
#define APP_GEMM_BY_OPT_GEMM_H

#include "gemm_perf_define.h"
#include <cassert>

#include "wrapper_defines_with_cr_ar_br.h"
template <typename T>
benchmark_result performance_benchmark_cr_ar_br(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params){
    T *matrix_a = nullptr, *matrix_b = nullptr, *matrix_c = nullptr;
    int i = 0;

    T alpha = static_cast<T>(extra_params.alpha);
    T beta = static_cast<T>(extra_params.beta);
    GemmLoopOrderAroundKernels loop_order = extra_params.loop_order;
    PackingType packing_type = extra_params.packing_type;
    int mc = extra_params.mc, kc = extra_params.kc, nc = extra_params.nc;
    int m_thread = extra_params.m_thread, k_thread = extra_params.k_thread, n_thread = extra_params.n_thread;
    int gemm_mat_c_prfm_skip_bytes = extra_params.gemm_mat_c_prfm_skip_bytes;
    int gemm_mat_a_prfm_skip_bytes = extra_params.gemm_mat_a_prfm_skip_bytes;
    int gemm_mat_b_prfm_skip_bytes = extra_params.gemm_mat_b_prfm_skip_bytes;
    int add_mat_c_prfm_skip_bytes = extra_params.add_mat_c_prfm_skip_bytes;
    int add_partial_sum_prfm_skip_bytes = extra_params.add_partial_sum_prfm_skip_bytes;

    double start=0, end=0, cost=0;
    benchmark_result benchmark = {
            .total_compute_time = 0,
            .cost = 0,
            .ops = 0,
            .benchmark = 0,
            .types = 0
    };
    matrix_a = new T[m * k];
    random_matrix(m,k,matrix_a);
    matrix_b = new T[k * n];
    random_matrix(k, n,matrix_b);

    matrix_c = new T[m * n];
    for (i = 0; i < 5; i++) {  // 使缓存热起来
            storage_layout_cr_ar_br::do_gemm_compute<T>(
                    loop_order,
                    matrix_c,
                    n,
                    matrix_a,
                    k,
                    matrix_b,
                    n,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    mc, kc, nc,
                    m_thread, k_thread, n_thread,
                    gemm_mat_c_prfm_skip_bytes,
                    gemm_mat_a_prfm_skip_bytes,
                    gemm_mat_b_prfm_skip_bytes,
                    add_mat_c_prfm_skip_bytes,
                    add_partial_sum_prfm_skip_bytes,
                    packing_type
            );
    }

    start = dclock();
    for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
        storage_layout_cr_ar_br::do_gemm_compute<T>(
                    loop_order,
                    matrix_c,
                    n,
                    matrix_a,
                    k,
                    matrix_b,
                    n,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    mc, kc, nc,
                    m_thread, k_thread, n_thread,
                    gemm_mat_c_prfm_skip_bytes,
                    gemm_mat_a_prfm_skip_bytes,
                    gemm_mat_b_prfm_skip_bytes,
                    add_mat_c_prfm_skip_bytes,
                    add_partial_sum_prfm_skip_bytes,
                    packing_type
            );
    }
    end = dclock();
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

benchmark_result performance_benchmark(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) {
    if ((extra_params.element_precision == ElementPrecision::EP_FLOAT32) && (extra_params.storage_layout == StorageLayout::CR_AR_BR)) {
        return performance_benchmark_cr_ar_br<float>(thread_num, iteration_times, m, n, k, extra_params);
    } else {
        assert(false);
    }
}

#endif //APP_GEMM_BY_OPT_GEMM_H
