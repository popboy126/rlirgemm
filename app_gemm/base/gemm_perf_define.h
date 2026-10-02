//
// Created by huche on 24-9-24.
//

#ifndef MATRIX_COMPUTATION_RESEARCH_GEMM_DEFINE_H
#define MATRIX_COMPUTATION_RESEARCH_GEMM_DEFINE_H

#define _STORAGE_ORDER_NNN 0
#define _STORAGE_ORDER_NNT 1

#ifndef MAT_STORAGE_ORDER
    #define MAT_STORAGE_ORDER _STORAGE_ORDER_NNN   // 默认是NNN存储
#endif

#include "ml_gemm_common.h"

extern "C" {
    struct benchmark_params {
        int m_thread;
        int k_thread;
        int n_thread;
        int mc;
        int kc;
        int nc;
        double alpha;
        double beta;
        int gemm_mat_a_prfm_skip_bytes;
        int gemm_mat_b_prfm_skip_bytes;
        int gemm_mat_c_prfm_skip_bytes;
        int add_mat_c_prfm_skip_bytes;
        int add_partial_sum_prfm_skip_bytes;
        GemmLoopOrderAroundKernels loop_order;
        StorageLayout storage_layout;
        ElementPrecision element_precision;
        PackingType packing_type;
    };

    enum benchmark_result_type {
        BENCHMARK_RESULT_TYPE_NONE = 0,
        BENCHMARK_RESULT_TYPE_JIT = 1,
    };

    struct benchmark_result {
        double total_compute_time;
        double cost;
        double ops;
        double benchmark;
        int types; // 额外的信息
    };

};

#endif //MATRIX_COMPUTATION_RESEARCH_GEMM_DEFINE_H
