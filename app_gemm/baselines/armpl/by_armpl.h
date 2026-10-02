//
// Created by huche on 2023/5/10 0010.
//

#ifndef MATRIX_COMPUTATION_RESEARCH_BY_ARMPL_H
#define MATRIX_COMPUTATION_RESEARCH_BY_ARMPL_H

#include "gemm_perf_define.h"
#include "cblas.h"
#include <cassert>

#ifdef _ARMPL_USE_OMP
#include <omp.h>

void armpl_set_num_threads(int thread_num) {
    omp_set_num_threads(thread_num);
}
#endif

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
    matrix_a = new float[m * k];
    random_matrix(m,k,matrix_a);
    matrix_b = new float[k * n];
    random_matrix(k, n,matrix_b);
    armpl_set_num_threads(thread_num); // 设置线程数为1

    matrix_c = new float[m * n];
    for (i = 0; i < 5; i++) {  // 使缓存热起来
        cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m,
                n,
                k,
                1.0,
                matrix_a,
                k,  // A的列，与是否转置无关
                matrix_b,
                n,  // B的列，与是否转置无关
                0.0,
                matrix_c,
                n  // C的列，与是否转置无关
        );
    }

    start = dclock();
    for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
        cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m,
                n,
                k,
                1.0,
                matrix_a,
                k,  // A的列，与是否转置无关
                matrix_b,
                n,  // B的列，与是否转置无关
                0.0,
                matrix_c,
                n  // C的列，与是否转置无关
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
        return performance_benchmark_cr_ar_br_float32(thread_num, iteration_times, m, n, k, extra_params);
    } else {
        assert(false);
    }
}

#endif //MATRIX_COMPUTATION_RESEARCH_BY_ARMPL_H


