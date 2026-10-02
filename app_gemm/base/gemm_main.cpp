//
// Created by huche on 2022/8/10 0013.
// 用于计算方阵的性能

#include<sys/time.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include "gemm_perf_define.h"

using namespace std;

#define ToString(str) _ToString(str)
#define _ToString(str) #str


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

#if CHECK_GEMM_RESULT_VAL
#include <limits>
#include <algorithm>
#ifndef GEMM_RESULT_REL_TOL
#define GEMM_RESULT_REL_TOL 1e-5f
#endif
template <typename T>
bool AlmostEqual(T a, T b,
                 T abs_tol = std::numeric_limits<T>::epsilon(),
                 T rel_tol = static_cast<T>(GEMM_RESULT_REL_TOL))
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

// 比对列优先的矩阵乘法计算结果
template <typename T>
int check_result(StorageLayout layout, T* C, T* A, T* B, int m, int k, int n) {
    if (layout == StorageLayout::CR_AR_BR) {
        // CR_AR_BR stores all matrices in row-major order.  Keep the
        // leading dimensions used by the published adapters explicit so the
        // checker validates the same logical operation as the benchmark.
        T* c_check = new T[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c_check[i * n + j] = 0;
                for (int p = 0; p < k; p++) {
                    c_check[i * n + j] += A[i * k + p] * B[p * n + j];
                }
            }
        }
        // 比较结果
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!AlmostEqual<T>(C[i * n + j], c_check[i * n + j])) {
                    cout << "Error at (" << i << ", " << j << "): " << C[i * n + j] << " != " << c_check[i * n + j] << endl;
                    delete[] c_check;
                    return -1;
                }
            }
        }
        delete[] c_check;
        return 0;
    } else if (layout == StorageLayout::CC_AC_BC) {
        // 计算参考结果
        T* c_check = new T[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c_check[i * n + j] = 0;
                for (int p = 0; p < k; p++) {
                    c_check[i * n + j] += A[i * k + p] * B[p * n + j];
                }
            }
        }
        // 比较结果
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!AlmostEqual<T>(C[i * n + j], c_check[i * n + j])) {
                    cout << "Error at (" << i << ", " << j << "): " << C[i * n + j] << " != " << c_check[i * n + j] << endl;
                    delete[] c_check;
                    return -1;
                }
            }
        }
        delete[] c_check;
        return 0;
    } else {
        cout << "Not support check for this layout!" << endl;
        return -1;
    }
}
#endif

template <typename T>
void random_matrix( int m, int n, T *a)
{
    double drand48();
    int i,j;

    for ( i=0; i<m; i++ )
        for ( j=0; j<n; j++ )
            a[i*n+j]= 2.0 * (T)drand48( ) - 1.0 ;
}

#if BENCH_OPT_O0 > 0
benchmark_result performance_benchmark(int thread_num,unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params) __attribute__((optimize("-O0")));
#else
benchmark_result performance_benchmark(int thread_num, unsigned iteration_times, int m, int n, int k, benchmark_params& extra_params);
#endif

#if defined(GEMM_BY_LIBSHALOM2)
    #include "by_libshalom2.h"
#elif defined(GEMM_BY_LIBSHALOM2_MP)
    #include "by_libshalom2_mp.h"
#elif defined(GEMM_BY_EPPA)
    #include "by_eppa.h"
#elif defined(GEMM_BY_OPENBLAS)
    #include "by_openblas.h"
#elif defined(GEMM_BY_BLIS)
    #include "by_blis.h"
#elif defined(GEMM_BY_EIGEN)
    #include "by_eigen.h"
#elif defined(GEMM_BY_BLASFEO)
    #include "by_blasfeo.h"
#elif defined(GEMM_BY_ARMPL)
    #include "by_armpl.h"
#elif defined(GEMM_BY_LIBXSMM)
    #include "by_libxsmm.h"
#elif defined(GEMM_BY_CAKE)
    #include "by_cake.h"
#elif defined(GEMM_BY_COMM_HEADER)
    #include GEMM_BY_COMM_HEADER
#else
    ;
#endif

int main_test(int thread_num, int repeat_times, int loops, int m, int k, int n, benchmark_params& extra_params) {
    string line;
    string matrix_a_path, matrix_b_path;
    benchmark_result benchmark = {
        .total_compute_time = 0,
        .cost = 0,
        .ops = 0,
        .benchmark = 0,
        .types = 0
    };
    for (int i = 0; i < repeat_times; i++) {
        cout << "---" << endl;
        benchmark = performance_benchmark(thread_num, loops, m, n, k, extra_params);
        cout << "[output] test_id: " << i + 1 << endl;
        cout << "[output] ops: " << benchmark.ops << endl;
        cout << "[output] total_compute_time: " << benchmark.total_compute_time << " s" << endl;
        cout << "[output] one_call_cost: " << benchmark.cost << " s" << endl;
        cout << "[output] benchmark: " << benchmark.benchmark << " GFlops" << endl;
        cout << "[output] types: " << benchmark.types << endl;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "Usage: "<<argv[0]<<" /path/to/input/files"<<endl;
        return -1;
    }

    ifstream iofile(argv[1], ios::in);
    if (!iofile)  {
        return -2;
    }

    int m = 0, k = 0, n = 0;
    int repeat_times = 0, loops = 0, thread_num=0;
    int m_thread = 0, k_thread=0, n_thread = 0;
    unsigned loop_order = 0, storage_layout = 0, element_precision = 0, packing_type = 0;
    int gemm_mat_a_prfm_skip_bytes = 0, gemm_mat_b_prfm_skip_bytes = 0, gemm_mat_c_prfm_skip_bytes = 0, add_mat_c_prfm_skip_bytes = 0, add_partial_sum_prfm_skip_bytes = 0;
    int mc = 0, kc = 0, nc = 0;
    double alpha = 1.0f, beta = 0.0f;
#ifdef _TRAIL_LABEL
    cout << "[global] gemm_test_label: "<<ToString(_TRAIL_LABEL)<<endl;
#endif
    while(iofile >> element_precision >> storage_layout >>
        loop_order >> packing_type >>
        m_thread >> k_thread >> n_thread >>
        mc >> kc >> nc >>
        gemm_mat_a_prfm_skip_bytes>>gemm_mat_b_prfm_skip_bytes >> gemm_mat_c_prfm_skip_bytes >>
        add_mat_c_prfm_skip_bytes >> add_partial_sum_prfm_skip_bytes >>
        repeat_times >> loops >>
        m >> k >> n >> alpha >> beta) {  // 从文件按行读入
        cout << "------"<<endl;
		cout << "[input] element_precision: " << get_element_precision_string(static_cast<ElementPrecision>(element_precision)) << ", storage_layout: "<<get_storage_order_string(static_cast<StorageLayout>(storage_layout)) << endl;
        cout << "[input] loop_order: "<< get_loop_order_string(static_cast<GemmLoopOrderAroundKernels>(loop_order))<< ", packing_type: " << get_packing_type_string(static_cast<PackingType>(packing_type)) <<endl;
        cout << "[input] mc: "<<mc<<", kc: "<<kc<<", nc: "<<nc<<endl;
        thread_num = m_thread * k_thread * n_thread;
        cout << "[input] thread: "<< thread_num<<", m_thread: "<<m_thread<<", k_thread: "<<k_thread <<", n_thread: "<<n_thread<<endl;
        cout << "[input] gemm_mat_a_prfm_skip_bytes: "<<gemm_mat_a_prfm_skip_bytes<<", gemm_mat_b_prfm_skip_bytes: "<<gemm_mat_b_prfm_skip_bytes<<", gemm_mat_c_prfm_skip_bytes: "<<gemm_mat_c_prfm_skip_bytes << endl;
		cout << "[input] add_mat_c_prfm_skip_bytes: "<<add_mat_c_prfm_skip_bytes << ", add_partial_sum_prfm_skip_bytes: "<< add_partial_sum_prfm_skip_bytes <<endl;
        cout << "[input] repeat_times: "<<repeat_times<<", loops: "<<loops<<endl;
        cout << "[input] gemm_shape: ["<<m<<", "<<k<<"]x["<<k<<", "<<n<<"]"<<", alpha: "<< alpha << ", beta: "<< beta<<endl;


        benchmark_params params = {
            .m_thread = m_thread,
            .k_thread = k_thread,
            .n_thread = n_thread,
            .mc = mc,
            .kc = kc,
            .nc = nc,
            .alpha = alpha,
            .beta = beta,
            .gemm_mat_a_prfm_skip_bytes = gemm_mat_a_prfm_skip_bytes,
            .gemm_mat_b_prfm_skip_bytes = gemm_mat_b_prfm_skip_bytes,
            .gemm_mat_c_prfm_skip_bytes = gemm_mat_c_prfm_skip_bytes,
			.add_mat_c_prfm_skip_bytes = add_mat_c_prfm_skip_bytes,
            .add_partial_sum_prfm_skip_bytes = add_partial_sum_prfm_skip_bytes,
            .loop_order = static_cast<GemmLoopOrderAroundKernels>(loop_order),
			.storage_layout = static_cast<StorageLayout>(storage_layout),
            .element_precision = static_cast<ElementPrecision>(element_precision),
            .packing_type = static_cast<PackingType>(packing_type),
        };

        main_test(thread_num, repeat_times, loops, m, k, n, params);
    }
    cout << "------------" <<endl;
    cout << "finished!" <<endl;
    return 0;
}
