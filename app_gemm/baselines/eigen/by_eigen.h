//
// Created by huche on 2023/5/10 0010.
//

#ifndef MATRIX_COMPUTATION_RESEARCH_EIGEN_H
#define MATRIX_COMPUTATION_RESEARCH_EIGEN_H

#include "gemm_perf_define.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <cassert>

void eigen_random_matrix( int m, int n, Eigen::MatrixXd &a)
{
    double drand48();
    int i,j;

    for ( i=0; i<m; i++ )
        for ( j=0; j<n; j++ )
            a(i, j)= 2.0 * (float)drand48( ) - 1.0 ;
}

benchmark_result performance_benchmark_cr_ar_br_float32(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) {
    int i = 0;
    double start=0, end=0, cost=0;
    benchmark_result benchmark = {
            .total_compute_time = 0,
            .cost = 0,
            .ops = 0,
            .benchmark = 0,
            .types = 0
    };
    Eigen::setNbThreads(thread_num);; // 设置线程数为1

    Eigen::MatrixXd matrix_a(m, k);
    eigen_random_matrix(m,k,matrix_a);
    Eigen::MatrixXd matrix_b(k, n);
    eigen_random_matrix(k, n,matrix_b);

    Eigen::MatrixXd matrix_c(m, n);
    matrix_c.setZero();
    for (i = 0; i < 5; i++) {  // 使缓存热起来
        matrix_c.noalias() = matrix_a * matrix_b;
    }

    start = dclock();
    for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
        matrix_c.noalias() = matrix_a * matrix_b;
    }
    end = dclock();
    benchmark.total_compute_time = end-start;
    benchmark.cost = benchmark.total_compute_time /iteration_times;
    benchmark.ops = (double)m * n  * k * 2 * 1.0e-09;
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

#endif //MATRIX_COMPUTATION_RESEARCH_EIGEN_H


