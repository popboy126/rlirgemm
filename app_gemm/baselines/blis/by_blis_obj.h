//
// Created by huche on 2023/5/10 0010.
//

#ifndef MATRIX_COMPUTATION_RESEARCH_BY_BLIS_H
#define MATRIX_COMPUTATION_RESEARCH_BY_BLIS_H

#include "gemm_perf_define.h"
#include "blis.h"
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
    matrix_a = new float[m * k];
    random_matrix(m,k,matrix_a);
    matrix_b = new float[k * n];
    random_matrix(k, n,matrix_b);

    matrix_c = new float[m * n];
    dim_t mc = extra_params.mc;
    dim_t nc = extra_params.nc;
    dim_t kc = extra_params.kc;

    bli_thread_set_num_threads(thread_num); // 设置线程数为1

    // 按blis的方式初始对象
    obj_t mat_a_obj, mat_b_obj, mat_c_obj;
    bli_obj_create_with_attached_buffer(BLIS_FLOAT, m, k, matrix_a, k, 1, &mat_a_obj);
    bli_obj_create_with_attached_buffer(BLIS_FLOAT, k, n, matrix_b, n, 1, &mat_b_obj);
    bli_obj_create_with_attached_buffer(BLIS_FLOAT, m, n, matrix_c, n, 1, &mat_c_obj);

    // 创建自定义上下文
    const cntx_t* cntx_c = bli_gks_query_cntx();
//    cntx_t cntx;
//    memcpy(&cntx, cntx_c, sizeof(cntx_t)); // 先拷贝一份默认的上下文
//
//    bli_cntx_set_blksz_def_dt(BLIS_FLOAT, BLIS_MC, mc, &cntx);
//    bli_cntx_set_blksz_def_dt(BLIS_FLOAT, BLIS_KC, kc, &cntx);
//    bli_cntx_set_blksz_def_dt(BLIS_FLOAT, BLIS_NC, nc, &cntx);
//
//    if (BLIS_SUCCESS != bli_cntx_init(&cntx)) {
//        printf("Error: Context initialization failed.\n");
//        exit(1);
//    }


    // Set the scalars to use.
    const obj_t* alpha = &BLIS_ONE;
    const obj_t*  beta  = &BLIS_ZERO;
//
//    // Initialize the matrix operands.
//    bli_randm( &mat_a_obj );
//    bli_randm( &mat_b_obj );
    bli_setm( &BLIS_ZERO, &mat_c_obj );

    for (i = 0; i < 5; i++) {  // 使缓存热起来
        // 执行带自定义参数的 GEMM
        bli_gemm_ex(alpha, &mat_a_obj, &mat_b_obj, beta, &mat_c_obj, cntx_c, NULL);

    }

    start = dclock();
    for (i = 0; i < iteration_times; i ++) {  // 真实跑性能
        bli_gemm_ex(alpha, &mat_a_obj, &mat_b_obj, beta, &mat_c_obj, cntx_c, NULL);

    }
    end = dclock();
    benchmark.total_compute_time = end-start;
    benchmark.cost = benchmark.total_compute_time /iteration_times;
    benchmark.ops = (double)m * n  * k * 2 * 1.0e-09;

    // 4. 【重要】释放对象所占用的资源（但不会释放buffer）
//    bli_cntx_free(&cntx);
//    bli_obj_free( &mat_a_obj );
//    bli_obj_free( &mat_b_obj );
//    bli_obj_free( &mat_c_obj );

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

#endif //MATRIX_COMPUTATION_RESEARCH_BY_BLIS_H


