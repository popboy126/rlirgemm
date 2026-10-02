//
// Created by huche on 2025/12/18.
//

#include <iostream>
#include <random>
#include <limits>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "gtest/gtest.h"

#include "wrapper_defines_with_cr_ar_br.h"
#include<sys/time.h>

static double gtod_ref_time_sec = 0.0;

double dclock()
{
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday( &tv, NULL );

    if ( abs(gtod_ref_time_sec) < 1e-15 )
        gtod_ref_time_sec = ( double ) tv.tv_sec;

    norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

template <typename T>
void random_matrix( int m, int n, T *a)
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(0.0, 1.0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = dis(gen);
        }
    }
}

#include "by_opt_gemm.h"

using namespace std;

template <typename T>
bool AlmostEqual(T a, T b,
                 T abs_tol = std::numeric_limits<T>::epsilon(),
                 T rel_tol = 1e-5f)
{
    // 1. 快速检查：完全相等（包括正负零）
    if (a == b) return true;

    // 2. 处理 NaN（非数字）
    if (std::isnan(a) || std::isnan(b)) return false;

    // 3. 计算绝对误差
    T diff = std::fabs(a - b);

    // 4. 绝对容差检查：适用于接近零的数
    if (diff <= abs_tol) return true;

    // 5. 相对容差检查：适用于较大数值
    T max_val = std::max(std::fabs(a), std::fabs(b));
    return (diff <= rel_tol * max_val);
}

template <typename T>
string get_matrix_data(T* mat, int row, int col) {
    stringstream ss;
    ss << fixed << setprecision(6);
    ss << "[" << endl;
    for (int i = 0; i < row; i++) {
        ss << "\t[";
        for (int j = 0; j < col; j++) {
            ss << mat[i * col + j];
            if (j != col - 1) {
                ss << ", ";
            }
        }
        if (i != row - 1) {
            ss << "]," << endl;
        } else {
            ss <<"]"<< endl;
        }
    }
    ss << "]" << endl;
    return ss.str();
}

TEST(opt_gemm_with_cr_ar_br_float32, debug) {
    const int M = 1;
    const int N = 1;
    const int K = 1;
    const float alpha = 1;
    const float beta = 0;
    const GemmLoopOrderAroundKernels loop_order = GemmLoopOrderAroundKernels::LO_NONE;
    const PackingType packing_type = PackingType::PACKING_TYPE_B_FP;
    const int mc = 1;
    const int kc = 1;
    const int nc = 1;
    const int m_thread = 1;
    const int k_thread = 1;
    const int n_thread = 1;
    const int gemm_mat_c_prfm_stride = 128;
    const int gemm_mat_a_prfm_stride = 128;
    const int gemm_mat_b_prfm_stride = 128;
    const int add_mat_c_prfm_stride = 128;
    const int add_partial_sum_prfm_stride = 128;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    auto *C = new float[M * N];
    auto *C_check = new float[M * N];

    auto *A = new float[M * K];
    auto *B = new float[K * N];


    // 初始化A
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            A[m * K + k] = dis(gen);
            // A[m * K + k] = m * K + k;
        }
    }


    // 初始化B
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B[k * N + n] = dis(gen);
            // B[k * N + n] = k * N + n;
        }
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] = dis(gen);
            // C[m * N + n] = m * N + n;
        }
    }

        // 计算校验结果
        double result = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                result = 0;
                for (int k = 0; k < K; k++) {
                    result += double(A[m * K + k] * B[k * N + n]);
                }
                C_check[m * N + n] = alpha * result + beta * C[m * N + n];
            }
        }



    // 调用矩阵乘法
    storage_layout_cr_ar_br::do_gemm_compute<float>(
            loop_order,
            C, N, A, K, B, N,
            M, N, K,
            alpha, beta,
            mc, kc, nc,
            m_thread, k_thread, n_thread,
            gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, add_mat_c_prfm_stride, add_partial_sum_prfm_stride,
            packing_type
    );

    // 校验结果矩阵
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            ASSERT_NEAR(C[m * N + n], C_check[m * N + n], 1e-4)
                                        << "C[" << m * N + n << "](m=" << m
                                        << ", M=" << M << ", n=" << n << ", N=" << N
                                        << ", K=" << K << ", alpha=" << alpha << ", beta=" << beta <<", loop_order="<< get_loop_order_string(loop_order) << ", mc=" << mc << ", kc="
                                        << kc << ", nc=" << nc << ", m_thread="
                                        << m_thread << ", k_thread="<< k_thread << ", n_thread=" << n_thread
                                        << ") is abnormal!" << endl;
        }
    }

    delete[] A;
    delete[] B;
    cout <<"loop_order="<< get_loop_order_string(loop_order) << ", mc=" << mc << ", kc=" << kc << ", nc=" << nc << ", m_thread=" << m_thread
         << ", k_thread=" << k_thread << ", n_thread=" << n_thread
         << ", M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta << " is passed!" << endl;
    delete[] C;
    delete[] C_check;
}


TEST(opt_gemm_with_cr_ar_br_float32, correctness) {
    // 测试正确性, 矩阵A,B, C均为行主序存储
    const int mc_list[] = {1, 3, 4, 16};
    const int kc_list[] = {1, 3, 4, 16};
    const int nc_list[] = {1, 3, 4, 16, 80};
    const int m_thread_list[] = {1, 2};
    const int k_thread_list[] = {1, 2, 4};
    const int n_thread_list[] = {1, 2, 4};
    const int M_list[] = {1, 2, 4, 8, 16 ,32, 64};
    const int K_list[] = {1, 2, 4, 8, 32, 64, 128};
    const int N_list[] = {1, 2, 4, 8, 16, 32, 64};
    const float alpha_list[] = {1, 2, 0.5f};
    const float beta_list[] = {0, 1, 2, 0.5f};
    const GemmLoopOrderAroundKernels loop_order_list[] = {
        GemmLoopOrderAroundKernels::LO_NONE,
        GemmLoopOrderAroundKernels::LO_NM,
        GemmLoopOrderAroundKernels::LO_MN,
        GemmLoopOrderAroundKernels::LO_NKM,
        GemmLoopOrderAroundKernels::LO_NMK,
        GemmLoopOrderAroundKernels::LO_KNM,
        GemmLoopOrderAroundKernels::LO_KMN,
        GemmLoopOrderAroundKernels::LO_MKN,
        GemmLoopOrderAroundKernels::LO_MNK,
    };

    const PackingType packing_type_list[] = {
        PackingType::PACKING_TYPE_NONE,
        PackingType::PACKING_TYPE_B_FP
    };

    // 计算所有的测试项
    int total_tests = (sizeof(loop_order_list)/sizeof(loop_order_list[0])) *
                      (sizeof(packing_type_list)/sizeof(packing_type_list[0])) *
                      (sizeof(alpha_list)/sizeof(alpha_list[0])) *
                      (sizeof(beta_list)/sizeof(beta_list[0])) *
                      (sizeof(mc_list)/sizeof(mc_list[0])) *
                      (sizeof(kc_list)/sizeof(kc_list[0])) *
                      (sizeof(nc_list)/sizeof(nc_list[0])) *
                      (sizeof(m_thread_list)/sizeof(m_thread_list[0])) *
                      (sizeof(k_thread_list)/sizeof(k_thread_list[0])) *
                      (sizeof(n_thread_list)/sizeof(n_thread_list[0])) *
                      (sizeof(M_list)/sizeof(M_list[0])) *
                      (sizeof(N_list)/sizeof(N_list[0])) *
                      (sizeof(K_list)/sizeof(K_list[0]));
    cout << "Total test cases: " << total_tests << endl;

    int k_list_length = sizeof(k_thread_list)/sizeof(k_thread_list[0]);
    int m_list_length = sizeof(m_thread_list)/sizeof(m_thread_list[0]);
    int n_list_length = sizeof(n_thread_list)/sizeof(n_thread_list[0]);

    int tested_cases = 0;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (auto loop_order: loop_order_list) {
        for (auto packing_type: packing_type_list) {
            for (auto alpha: alpha_list) {
                for (auto beta: beta_list) {
                    for(auto mc: mc_list) {
                        for(auto kc: kc_list) {
                            for(auto nc: nc_list) {
                                for (auto k_thread: k_thread_list) {
                                    for (auto m_thread: m_thread_list) {
                                        for (auto n_thread: n_thread_list) {
                                            for (int m_idx =0; m_idx < m_list_length; m_idx++) {
                                                auto M = M_list[m_idx];
                                                if (m_thread > M) {
                                                    tested_cases += ((m_list_length - m_idx) * k_list_length * n_list_length);
                                                    continue;
                                                }
                                                for (int n_idx = 0; n_idx < n_list_length; n_idx ++) {
                                                    auto N = N_list[n_idx];
                                                    if (n_thread > N) {
                                                        tested_cases = tested_cases + (n_list_length - n_idx) * k_list_length;
                                                        continue;
                                                    }

                                                    auto *C = new float[M * N];
                                                    auto *C_check = new float[M * N];

                                                    for (int k_idx = 0; k_idx < k_list_length; k_idx++) {
                                                        auto K = K_list[k_idx];
                                                        if (k_thread > K) {
                                                            tested_cases = tested_cases + (k_list_length - k_idx);
                                                            continue;
                                                        }

                                                        cout <<"[ "<< tested_cases<<" / "<< total_tests << " ]: " <<"loop_order="<< get_loop_order_string(loop_order) <<", packing_type="<< get_packing_type_string(packing_type) <<", mc=" << mc << ", kc=" << kc << ", nc=" << nc << ", m_thread=" << m_thread
                                                             << ", k_thread=" << k_thread << ", n_thread=" << n_thread
                                                             << ", M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta << " is testing!" << endl;

                                                        auto *A = new float[M * K];
                                                        auto *B = new float[K * N];

                                                        // 初始化A
                                                        for (int m = 0; m < M; m++) {
                                                            for (int k = 0; k < K; k++) {
                                                                A[m * K + k] = dis(gen);
                                                            }
                                                        }

                                                        // 初始化B
                                                        for (int k = 0; k < K; k++) {
                                                            for (int n = 0; n < N; n++) {
                                                                B[k * N + n] = dis(gen);
                                                            }
                                                        }

                                                        // 初始化C
                                                        for (int m = 0; m < M; m++) {
                                                            for (int n = 0; n < N; n++) {
                                                                C[m * N + n] = dis(gen);
                                                            }
                                                        }

                                                        // 计算校验结果
                                                        double result = 0;
                                                        for (int m = 0; m < M; m++) {
                                                            for (int n = 0; n < N; n++) {
                                                                result = 0;
                                                                for (int k = 0; k < K; k++) {
                                                                    result += double(A[m * K + k] * B[k * N + n]);
                                                                }
                                                                C_check[m * N + n] = alpha * result + beta * C[m * N + n];
                                                            }
                                                        }


                                                        // 调用矩阵乘法
                                                        storage_layout_cr_ar_br::do_gemm_compute<float>(
                                                                loop_order,
                                                                C, N, A, K, B, N,
                                                                M, N, K,
                                                                alpha, beta,
                                                                mc, kc, nc,
                                                                m_thread, k_thread, n_thread,
                                                                128, 128, 128, 128, 128,
                                                                packing_type
                                                        );


                                                        // 校验结果矩阵
                                                        for (int m = 0; m < M; m++) {
                                                            for (int n = 0; n < N; n++) {
                                                                ASSERT_NEAR(C[m * N + n], C_check[m * N + n], 1e-5)
                                                                                            << "C[" << m * N + n << "](m=" << m
                                                                                            << ", M=" << M << ", n=" << n << ", N=" << N
                                                                                            << ", K=" << K << ", alpha=" << alpha << ", beta=" << beta <<", loop_order="<< get_loop_order_string(loop_order) << ", packing_type=" << get_packing_type_string(packing_type) << ", mc=" << mc << ", kc="
                                                                                            << kc << ", nc=" << nc << ", m_thread="
                                                                                            << m_thread << ", k_thread="<< k_thread << ", n_thread=" << n_thread
                                                                                            << ") is abnormal!" << endl
                                                                                            << "Matrix A: " << endl << get_matrix_data(A, M, K) << endl
                                                                                            << "Matrix B: " << endl << get_matrix_data(B, K, N) << endl
                                                                                            << "Matrix C (input): " << endl << get_matrix_data(C, M, N) << endl
                                                                                            << "Matrix C (expected): " << endl << get_matrix_data(C_check, M, N) << endl;
                                                            }
                                                        }

                                                        delete[] A;
                                                        delete[] B;
                                                        tested_cases += 1;
                                                        cout <<"[ "<< tested_cases<<" / "<< total_tests << " ]: " <<"loop_order="<< get_loop_order_string(loop_order) <<", packing_type="<< get_packing_type_string(packing_type) <<", mc=" << mc << ", kc=" << kc << ", nc=" << nc << ", m_thread=" << m_thread
                                                             << ", k_thread=" << k_thread << ", n_thread=" << n_thread
                                                             << ", M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta << " is passed!" << endl;
                                                    }
                                                    delete[] C;
                                                    delete[] C_check;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }

}


TEST(performance_benchmark_float, correctness) {
    const int M = 1, K = 256, N = 64;
    const unsigned iteration_times = 30;

    benchmark_params extra_params = benchmark_params {
        .m_thread = 1,
        .k_thread = 1,
        .n_thread = 1,
        .mc = 1,
        .kc = 128,
        .nc = 32,
        .alpha = 1.0f,
        .beta = 0.0f,
        .gemm_mat_a_prfm_skip_bytes = 128,
        .gemm_mat_b_prfm_skip_bytes = 64,
        .gemm_mat_c_prfm_skip_bytes = 64,
        .add_mat_c_prfm_skip_bytes = 64,
        .add_partial_sum_prfm_skip_bytes = 128,
        .loop_order = GemmLoopOrderAroundKernels::LO_NKM,
        .storage_layout = StorageLayout::CR_AR_BR,
        .element_precision = ElementPrecision::EP_FLOAT32,
        .packing_type = PackingType::PACKING_TYPE_B_FP,
    };

    const int thread_num = extra_params.m_thread * extra_params.k_thread * extra_params.n_thread;


    performance_benchmark(
        thread_num,
        iteration_times, M, N, K, extra_params);

}
