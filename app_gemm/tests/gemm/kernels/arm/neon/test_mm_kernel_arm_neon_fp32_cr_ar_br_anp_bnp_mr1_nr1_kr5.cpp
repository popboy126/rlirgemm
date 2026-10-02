//
// Created by huche on 2025/12/22.
//


#include <iostream>
#include <random>
#include <limits>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "gtest/gtest.h"

#include "arm_neon_gemm_kernels.h"

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

TEST(mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5, debug) {

    const int64_t M = 1;
    const int64_t N = 1;
    const int64_t K = 10;
    const int64_t ldc = N, lda = K, ldb = N;
    int64_t c_prefetch_stride = 64, a_prefetch_stride = 128, b_prefetch_stride = 256;

    const float alpha = 0.5;
    const float beta = 0;

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
            // A[m * K + k] = dis(gen);
            A[m * K + k] = 2;
            // A[m * K + k] = 1;
        }
    }


    // 初始化B
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            // B[k * N + n] = dis(gen);
            B[k * N + n] = k;
            // B[k * N + n] = n+1;
        }
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            // C[m * N + n] = dis(gen);
            C[m * N + n] = 1;
        }
    }

        // 计算校验结果
        float result = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                result = 0;
                for (int k = 0; k < K; k++) {
                    result += (A[m * K + k] * B[k * N + n]);
                }
                C_check[m * N + n] = alpha * result + beta * C[m * N + n];
            }
        }



    // 调用矩阵乘法
    arm::neon::mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5(
            C, A, B,
            M, N, K,
            ldc, lda, ldb,
            c_prefetch_stride, a_prefetch_stride, b_prefetch_stride,
            alpha, beta
            );

    // 校验结果矩阵
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            ASSERT_NEAR(C[m * N + n], C_check[m * N + n], 1e-5)
                                        << "C[" << m * N + n << "](m=" << m
                                        << ", M=" << M << ", n=" << n << ", N=" << N
                                        << ", K=" << K << ", alpha=" << alpha << ", beta=" << beta
                                        << ") is abnormal!" << endl;
        }
    }

    delete[] A;
    delete[] B;
    cout <<"M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta << " is passed!" << endl;
    delete[] C;
    delete[] C_check;
}

TEST(mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5, scale_test) {

    const int64_t M = 1;
    const int64_t N = 1;
    const int64_t K = 5;
    const int64_t ldc = N, lda = K, ldb = N;
    int64_t c_prefetch_stride = 64, a_prefetch_stride = 128, b_prefetch_stride = 256;

    const float alpha_list[] = {0.5f, 1.0f, 2.0f, -1.0f, -0.5f};
    const float beta_list[] = {-0.5f, 0.0f, 0.5f, 1.0f, 2.0f};

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    auto *C = new float[M * N];
    auto *C_check = new float[M * N];

    auto *A = new float[M * K];
    auto *B = new float[K * N];

    for (auto alpha : alpha_list) {
        for (auto beta : beta_list) {
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
                    // B[k * N + n] = K * N - (k * N + n);
                }
            }

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    C[m * N + n] = dis(gen);
                    // C[m * N + n] = m * N + n;
                }
            }

            // 计算校验结果
            float result = 0;
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    result = 0;
                    for (int k = 0; k < K; k++) {
                        result += (A[m * K + k] * B[k * N + n]);
                    }
                    C_check[m * N + n] = alpha * result + beta * C[m * N + n];
                }
            }



            // 调用矩阵乘法
            arm::neon::mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5(
                    C, A, B,
                    M, N, K,
                    ldc, lda, ldb,
                    c_prefetch_stride, a_prefetch_stride, b_prefetch_stride,
                    alpha, beta
                    );

            // 校验结果矩阵
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    ASSERT_NEAR(C[m * N + n], C_check[m * N + n], 1e-5)
                                                << "C[" << m * N + n << "](m=" << m
                                                << ", M=" << M << ", n=" << n << ", N=" << N
                                                << ", K=" << K << ", alpha=" << alpha << ", beta=" << beta
                                                << ") is abnormal!" << endl;
                }
            }

            cout <<"M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta << " is passed!" << endl;
        }
    }


    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_check;
}


TEST(mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5, all_scale_test) {

    const int64_t M_list[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int64_t N_list[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int64_t K_list[] = {5, 10, 15, 20, 25};
    int64_t ldc = 0, lda = 0, ldb = 0;
    int64_t c_prefetch_stride = 64, a_prefetch_stride = 128, b_prefetch_stride = 256;

    const float alpha_list[] = {0.5f, 1.0f, 2.0f, -1.0f, -0.5f};
    const float beta_list[] = {-0.5f, 0.0f, 0.5f, 1.0f, 2.0f};

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);


    for (auto M : M_list) {
        for (auto N : N_list) {
            for (auto K : K_list) {
                ldc = N;
                lda = K;
                ldb = N;

                auto *C = new float[M * N];
                auto *C_check = new float[M * N];

                auto *A = new float[M * K];
                auto *B = new float[K * N];

                for (auto alpha : alpha_list) {
                    for (auto beta : beta_list) {
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
                                // B[k * N + n] = K * N - (k * N + n);
                            }
                        }

                        for (int m = 0; m < M; m++) {
                            for (int n = 0; n < N; n++) {
                                C[m * N + n] = dis(gen);
                                // C[m * N + n] = m * N + n;
                            }
                        }

                        // 计算校验结果
                        float result = 0;
                        for (int m = 0; m < M; m++) {
                            for (int n = 0; n < N; n++) {
                                result = 0;
                                for (int k = 0; k < K; k++) {
                                    result += (A[m * K + k] * B[k * N + n]);
                                }
                                C_check[m * N + n] = alpha * result + beta * C[m * N + n];
                            }
                        }



                        // 调用矩阵乘法
                        arm::neon::mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5(
                                C, A, B,
                                M, N, K,
                                ldc, lda, ldb,
                                c_prefetch_stride, a_prefetch_stride, b_prefetch_stride,
                                alpha, beta
                                );

                        // 校验结果矩阵
                        for (int m = 0; m < M; m++) {
                            for (int n = 0; n < N; n++) {
                                ASSERT_NEAR(C[m * N + n], C_check[m * N + n], 1e-5)
                                                            << "C[" << m * N + n << "](m=" << m
                                                            << ", M=" << M << ", n=" << n << ", N=" << N
                                                            << ", K=" << K << ", alpha=" << alpha << ", beta=" << beta
                                                            << ") is abnormal!" << endl;
                            }
                        }

                        cout <<"M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta << " is passed!" << endl;
                    }
                }


                delete[] A;
                delete[] B;
                delete[] C;
                delete[] C_check;
            }
        }
    }
}
