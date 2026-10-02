//
// Created by huche on 2025/12/30.
//


#include <iostream>
#include <random>
#include <limits>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "gtest/gtest.h"

#include "arm_neon_add_kernels.h"

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

TEST(add_kernel_arm_neon_fp32_r_mr2_nr16_kr2, debug) {

    const int64_t M = 2;
    const int64_t N = 64;
    const int64_t K = 2;
    const int64_t ldc = 65, ldp = 70;
    int64_t c_prefetch_stride = 64, partial_sum_prefetch_stride = 64;

    const float alpha = 1;
    const float beta = 0;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    auto *C = new float[M * ldc];
    auto *C_check = new float[M * N];

    auto **partial_sum_ptr = new float*[K];
    for (int k = 0; k < K; k++) {
        partial_sum_ptr[k] = new float[M * ldp];
    }


    // 初始化部分和矩阵
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                // partial_sum_ptr[k][m * N + n] = dis(gen);
                // partial_sum_ptr[k][m * N + n] = k * N + n;
                partial_sum_ptr[k][m * ldp + n] = k;
            }
        }
    }

    // 初始化矩阵C
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            // C[m * ldc + n] = dis(gen);
            C[m * ldc + n] = 0;
        }
    }

    // 计算校验结果
    float result = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            result = 0;
            for (int k = 0; k < K; k++) {
                result += partial_sum_ptr[k][m * ldp + n];
            }
            C_check[m * N + n] = alpha * result + beta * C[m * ldc + n];
        }
    }



    // 调用矩阵乘法
    arm::neon::add_kernel_arm_neon_fp32_r_mr2_nr16_kr2(
            C, ldc,
            partial_sum_ptr, ldp,
            M, N, K,
            alpha, beta,
            c_prefetch_stride, partial_sum_prefetch_stride
            );

    // 校验结果矩阵
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            ASSERT_NEAR(C[m * ldc + n], C_check[m * N + n], 1e-5)
                                        << "C[" << m * N + n << "](m=" << m
                                        << ", M=" << M << ", n=" << n << ", N=" << N
                                        << ", K=" << K << ", alpha=" << alpha << ", beta=" << beta
                                        << ", ldc="<< ldc << ", ldp="<< ldp
                                        << ") is abnormal!" << endl;
        }
    }

    cout <<"M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta << ", ldc="<< ldc << ", ldp="<< ldp << " is passed!" << endl;
    for (int k = 0; k < K; k++) {
        delete[] partial_sum_ptr[k];
    }
    delete[] partial_sum_ptr;
    delete[] C;
    delete[] C_check;
}

TEST(add_kernel_arm_neon_fp32_r_mr2_nr16_kr2, all_scale_test) {
    const int64_t M_list[] = {2, 4, 6, 8, 16};
    const int64_t N_list[] = {16, 32, 48};
    const int64_t K_list[] = {2, 4, 8, 16};
    const int64_t ldc_list[] = {48, 64};
    const int64_t ldp_list[] = {56, 80};
    int64_t c_prefetch_stride = 64, partial_sum_prefetch_stride = 64;

    const float alpha_list[] = {0.5f, 1.0f, 2.0f, -1.0f, -0.5f};
    const float beta_list[] = {-0.5f, 0.0f, 0.5f, 1.0f, 2.0f};

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (auto M : M_list) {
        for (auto N : N_list) {
            for (auto ldc: ldc_list) {
                if (ldc < N) continue;
                for (auto ldp : ldp_list) {
                    if (ldp < N) continue;
                    for (auto K : K_list) {
                        for (auto alpha : alpha_list) {
                            for (auto beta : beta_list) {
                                auto *C = new float[M * ldc];
                                auto *C_check = new float[M * N];

                                auto **partial_sum_ptr = new float*[K];
                                for (int k = 0; k < K; k++) {
                                    partial_sum_ptr[k] = new float[M * ldp];
                                }

                                // 初始化部分和矩阵
                                for (int k = 0; k < K; k++) {
                                    for (int m = 0; m < M; m++) {
                                        for (int n = 0; n < N; n++) {
                                            partial_sum_ptr[k][m * ldp + n] = dis(gen);
                                        }
                                    }
                                }

                                // 初始化矩阵C
                                for (int m = 0; m < M; m++) {
                                    for (int n = 0; n < N; n++) {
                                        C[m * ldc + n] = dis(gen);
                                    }
                                }

                                // 计算校验结果
                                float result = 0;
                                for (int m = 0; m < M; m++) {
                                    for (int n = 0; n < N; n++) {
                                        result = 0;
                                        for (int k = 0; k < K; k++) {
                                            result += partial_sum_ptr[k][m * ldp + n];
                                        }
                                        C_check[m * N + n] = alpha * result + beta * C[m * ldc + n];
                                    }
                                }



                                // 调用矩阵乘法
                                arm::neon::add_kernel_arm_neon_fp32_r_mr2_nr16_kr2(
                                        C, ldc,
                                        partial_sum_ptr, ldp,
                                        M, N, K,
                                        alpha, beta,
                                        c_prefetch_stride, partial_sum_prefetch_stride
                                        );

                                // 校验结果矩阵
                                for (int m = 0; m < M; m++) {
                                    for (int n = 0; n < N; n++) {
                                        ASSERT_NEAR(C[m * ldc + n], C_check[m * N + n], 1e-5)
                                                                    << "C[" << m * N + n << "](m=" << m
                                                                    << ", M=" << M << ", n=" << n << ", N=" << N
                                                                    << ", K=" << K << ", ldc="<< ldc << ", ldp="<< ldp << ", alpha=" << alpha << ", beta=" << beta
                                                                    << ") is abnormal!" << endl;
                                    }
                                }

                                cout <<"M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta <<", ldc="<< ldc << ", ldp="<< ldp << " is passed!" << endl;
                                for (int k = 0; k < K; k++) {
                                    delete[] partial_sum_ptr[k];
                                }
                                delete[] partial_sum_ptr;
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


TEST(add_kernel_arm_neon_fp32_r_mr2_nr16_kr2, all_test) {
    const int64_t M_list[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int64_t N_list[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32};
    const int64_t K_list[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int64_t ldc_list[] = {32, 40, 64};
    const int64_t ldp_list[] = {32, 48, 56};
    int64_t c_prefetch_stride = 64, partial_sum_prefetch_stride = 64;

    const float alpha_list[] = {0.5f, 1.0f, 2.0f, -1.0f, -0.5f};
    const float beta_list[] = {-0.5f, 0.0f, 0.5f, 1.0f, 2.0f};

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (auto M : M_list) {
        for (auto N : N_list) {
            for (auto ldc: ldc_list) {
                if (ldc < N) continue;
                for (auto ldp : ldp_list) {
                    if (ldp < N) continue;
                    for (auto K : K_list) {
                        for (auto alpha : alpha_list) {
                            for (auto beta : beta_list) {
                                auto *C = new float[M * ldc];
                                auto *C_check = new float[M * N];

                                auto **partial_sum_ptr = new float*[K];
                                for (int k = 0; k < K; k++) {
                                    partial_sum_ptr[k] = new float[M * ldp];
                                }

                                // 初始化部分和矩阵
                                for (int k = 0; k < K; k++) {
                                    for (int m = 0; m < M; m++) {
                                        for (int n = 0; n < N; n++) {
                                            partial_sum_ptr[k][m * ldp + n] = dis(gen);
                                        }
                                    }
                                }

                                // 初始化矩阵C
                                for (int m = 0; m < M; m++) {
                                    for (int n = 0; n < N; n++) {
                                        C[m * ldc + n] = dis(gen);
                                    }
                                }

                                // 计算校验结果
                                float result = 0;
                                for (int m = 0; m < M; m++) {
                                    for (int n = 0; n < N; n++) {
                                        result = 0;
                                        for (int k = 0; k < K; k++) {
                                            result += partial_sum_ptr[k][m * ldp + n];
                                        }
                                        C_check[m * N + n] = alpha * result + beta * C[m * ldc + n];
                                    }
                                }



                                // 调用矩阵乘法
                                arm::neon::add_kernel_arm_neon_fp32_r_mr2_nr16_kr2(
                                        C, ldc,
                                        partial_sum_ptr, ldp,
                                        M, N, K,
                                        alpha, beta,
                                        c_prefetch_stride, partial_sum_prefetch_stride
                                        );

                                // 校验结果矩阵
                                for (int m = 0; m < M; m++) {
                                    for (int n = 0; n < N; n++) {
                                        ASSERT_NEAR(C[m * ldc + n], C_check[m * N + n], 1e-5)
                                                                    << "C[" << m * N + n << "](m=" << m
                                                                    << ", M=" << M << ", n=" << n << ", N=" << N
                                                                    << ", K=" << K << ", ldc="<< ldc << ", ldp="<< ldp << ", alpha=" << alpha << ", beta=" << beta
                                                                    << ") is abnormal!" << endl;
                                    }
                                }

                                cout <<"M=" << M << ", K=" << K << ", N=" << N << ", alpha=" << alpha << ", beta=" << beta <<", ldc="<< ldc << ", ldp="<< ldp << " is passed!" << endl;
                                for (int k = 0; k < K; k++) {
                                    delete[] partial_sum_ptr[k];
                                }
                                delete[] partial_sum_ptr;
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
