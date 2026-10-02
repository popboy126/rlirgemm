//
// Created by huche on 2023/5/10 0010.
//

#ifndef MATRIX_COMPUTATION_RESEARCH_BY_BLIS_H
#define MATRIX_COMPUTATION_RESEARCH_BY_BLIS_H

#include "gemm_perf_define.h"
#include "cake.h"
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

    cake_cntx_t* cake_cntx = cake_query_cntx();
    cake_cntx -> ncores = thread_num;  // 限制总线程数

    for (i = 0; i < 5; i++) {  // 使缓存热起来
        cake_sgemm(matrix_a, matrix_b, matrix_c, m, n, k, 1, cake_cntx);
    }

    start = dclock();
    for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
        cake_sgemm(matrix_a, matrix_b, matrix_c, m, n, k, 1, cake_cntx);
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
        return performance_benchmark_cr_ar_br_float32(thread_num, iteration_times, m, n, k, extra_params);
    }  else {
        assert(false);
    }
}

#endif //MATRIX_COMPUTATION_RESEARCH_BY_BLIS_H


