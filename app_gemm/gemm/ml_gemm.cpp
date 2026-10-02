//
// Created by huche on 2026/1/28.
//

#include <iostream>
#include <cstdlib>
#include <string>
#include "cblas.h"
#include "ml_gemm.h"
#include "wrapper_defines_with_cr_ar_br.h"
#ifdef __cplusplus
    extern "C" {
#endif
        void do_gemm_compute_f32(
                unsigned short loop_order,
                float *C,
                int LDC,
                float *A,
                int LDA,
                float *B,
                int LDB,
                int M,
                int N,
                int K,
                float alpha,
                float beta,
                int mc, int kc, int nc,
                int m_thread, int k_thread, int n_thread,
                int gemm_mat_c_prfm_stride,
                int gemm_mat_a_prfm_stride,
                int gemm_mat_b_prfm_stride,
                int add_mat_c_prfm_stride,
                int add_partial_sum_prfm_stride,
                unsigned short packing_type
        )
        {
            // 调用矩阵乘法
            storage_layout_cr_ar_br::do_gemm_compute<float>(
                static_cast<GemmLoopOrderAroundKernels>(loop_order),
                    C,
                    LDC,
                    A,
                    LDA,
                    B,
                    LDB,
                    M,
                    N,
                    K,
                    alpha,
                    beta,
                    mc, kc, nc,
                    m_thread, k_thread, n_thread,
                    gemm_mat_c_prfm_stride,
                    gemm_mat_a_prfm_stride,
                    gemm_mat_b_prfm_stride,
                    add_mat_c_prfm_stride,
                    add_partial_sum_prfm_stride,
                    static_cast<PackingType>(packing_type)
            );
        }

        void get_gemm_configs(int M, int K, int N,
            int &loop_order, int &packing_type,
            int & m_thread, int & k_thread, int& n_thread,
            int & mc, int & kc, int & nc,
            int &gemm_mat_a_prfm_stride, int &gemm_mat_b_prfm_stride, int &gemm_mat_c_prfm_stride,
            int &add_mat_c_prfm_stride, int &add_partial_sum_prfm_stride
        ) {
            int idx = (K << 18) + N;
            switch (idx) {
#define GEMM_CFG(K_val, N_val, loop_ord, pack, \
            m_thread_val, k_thread_val, n_thread_val, \
            mc_val, kc_val, nc_val, \
            gemm_mat_a_prfm_stride_val, gemm_mat_b_prfm_stride_val, gemm_mat_c_prfm_stride_val, \
            add_mat_c_prfm_stride_val, add_partial_sum_prfm_stride_val)  \
                case ((K_val << 18) + N_val): {                      \
                    loop_order = loop_ord; packing_type = pack;         \
                    m_thread = m_thread_val; k_thread = k_thread_val; n_thread = n_thread_val; \
                    mc = mc_val; kc = kc_val; nc = nc_val;         \
                    gemm_mat_c_prfm_stride = gemm_mat_c_prfm_stride_val;  \
                    gemm_mat_a_prfm_stride = gemm_mat_a_prfm_stride_val;  \
                    gemm_mat_b_prfm_stride = gemm_mat_b_prfm_stride_val;  \
                    add_mat_c_prfm_stride = add_mat_c_prfm_stride_val;  \
                    add_partial_sum_prfm_stride = add_partial_sum_prfm_stride_val;  \
                    break;                                            \
                }
                /*
                // KP920 32 线程
                GEMM_CFG(4096, 4096,6, 0, 1, 32, 1, 1, 1, 1337, 7865, 1, 16354, 16354, 1263)
                GEMM_CFG(2560, 6144,4, 0, 1, 32, 1, 1, 7, 968, 3152, 0, 678, 678, 1285)
                GEMM_CFG(3072, 16384,6, 1, 1, 32, 1, 1, 1, 15152, 7135, 0, 17551, 17551, 3757)
                GEMM_CFG(4096, 12288,7, 1, 1, 32, 1, 1, 1, 4848, 7025, 3, 21681, 21681, 3679)
                GEMM_CFG(640, 262144,7, 0, 1, 32, 1, 1, 1, 223111, 1535, 64, 889260, 889260, 71637)
                GEMM_CFG(8192, 3072,8, 0, 1, 32, 1, 1, 2, 2279, 8121, 0, 4822, 4822, 622)
                GEMM_CFG(11008, 4096,6, 0, 1, 32, 1, 1, 1, 2142, 21038, 1, 8105, 8105, 1336)
                GEMM_CFG(4096, 22016,7, 1, 1, 32, 1, 1, 1, 20578, 4295, 21, 17795, 17795, 2990)
                GEMM_CFG(640, 1024,2, 0, 1, 2, 16, 1, 1, 578, 907, 0, 0, 0, 0)
                GEMM_CFG(12288, 4096,5, 1, 1, 32, 1, 1, 1, 2095, 8671, 0, 15781, 15781, 2016)
                GEMM_CFG(4096, 102400,6, 1, 1, 32, 1, 1, 1172, 58264, 8216, 91, 140824, 140824, 124131)
                GEMM_CFG(4096, 6144,7, 1, 1, 32, 1, 1, 1, 3764, 7332, 0, 3921, 3921, 726)
                GEMM_CFG(1024, 640,2, 0, 1, 2, 16, 1, 1, 227, 1722, 0, 0, 0, 0)
                GEMM_CFG(3072, 3072,7, 1, 1, 32, 1, 1, 1, 1557, 3618, 2, 9329, 9329, 918)
                GEMM_CFG(2560, 151936,4, 0, 1, 32, 1, 1, 5, 110430, 7048, 34, 304612, 304612, 16507)
                GEMM_CFG(640, 4096,2, 1, 1, 2, 16, 1, 1, 1558, 682, 0, 0, 0, 0)
                GEMM_CFG(3072, 128256,5, 1, 1, 32, 1, 1, 726, 66071, 5702, 1, 445291, 445291, 68450)
                GEMM_CFG(2048, 640,5, 1, 1, 32, 1, 1, 1, 517, 1796, 4, 2094, 2094, 242)
                GEMM_CFG(4096, 24576,5, 0, 1, 32, 1, 1, 1, 11428, 4879, 0, 92908, 92908, 18492)
                GEMM_CFG(9728, 2560,7, 0, 1, 32, 1, 1, 17, 397, 17831, 0, 10087, 10087, 587)
                GEMM_CFG(3072, 5120,6, 0, 1, 32, 1, 1, 1, 1134, 2970, 0, 7324, 7324, 7736)
                GEMM_CFG(4096, 151936,5, 0, 1, 32, 1, 1, 1, 12312, 6833, 20, 604569, 604569, 9286)
                GEMM_CFG(4096, 2560,7, 1, 1, 32, 1, 1, 4, 158, 2762, 173, 9242, 9242, 291)
                GEMM_CFG(640, 256,2, 1, 1, 2, 16, 1, 1, 41, 1157, 0, 0, 0, 0)
                GEMM_CFG(2560, 19456,4, 1, 1, 32, 1, 1, 1, 11355, 2626, 16, 52436, 52436, 7497)
                */


                // KP920 128线程
                GEMM_CFG(640, 256,1, 0, 1, 8, 16, 1, 1, 160, 1945, 0, 902, 0, 0)
                GEMM_CFG(640, 1024,1, 0, 1, 4, 32, 1, 1, 638, 17, 0, 921, 0, 0)
                GEMM_CFG(640, 4096,1, 0, 1, 8, 16, 1, 1, 2550, 340, 0, 115, 0, 0)
                GEMM_CFG(640, 262144,4, 1, 1, 4, 32, 1, 3, 252141, 2492, 670896680, 655672, 1009330, 973110)
                GEMM_CFG(1024, 640,1, 0, 1, 8, 16, 1, 1, 399, 57, 48629, 1783, 0, 0)
                GEMM_CFG(2048, 640,1, 0, 1, 4, 32, 1, 1, 399, 8, 0, 3, 0, 0)
                GEMM_CFG(2560, 6144,4, 1, 1, 8, 16, 1, 7, 6075, 9610, 62869586, 12331, 24526, 8952)
                GEMM_CFG(2560, 19456,4, 1, 1, 8, 16, 1, 760, 12340, 1152, 71197774, 37459, 27903, 3201)
                GEMM_CFG(2560, 151936,4, 1, 1, 32, 4, 1, 2004, 22492, 851, 1299722806, 386865, 236842, 424412)
                GEMM_CFG(3072, 3072,4, 0, 1, 32, 4, 1, 356, 2486, 8216, 37747809, 7940, 522, 233)
                GEMM_CFG(3072, 5120,4, 0, 1, 8, 16, 1, 56, 4828, 6046, 62905365, 18544, 8171, 649)
                GEMM_CFG(3072, 16384,4, 0, 1, 32, 4, 1, 12, 9676, 7434, 81564150, 31427, 51271, 9306)
                GEMM_CFG(3072, 128256,4, 1, 1, 32, 4, 1, 12, 56401, 11431, 1269173687, 377479, 51232, 508092)
                GEMM_CFG(4096, 2560,4, 1, 1, 32, 4, 1, 3843, 1529, 13036, 6749371, 1707, 7471, 18)
                GEMM_CFG(4096, 4096,4, 1, 1, 32, 4, 1, 15, 3634, 12008, 67020724, 10541, 111, 9876)
                GEMM_CFG(4096, 6144,4, 1, 1, 32, 4, 1, 211, 3913, 5402, 81905310, 10264, 17340, 24537)
                GEMM_CFG(4096, 12288,4, 1, 1, 32, 4, 1, 3663, 5981, 5225, 141631872, 18333, 4115, 3869)
                GEMM_CFG(4096, 22016,4, 0, 1, 16, 8, 1, 2, 21944, 16158, 360706704, 75230, 87896, 26821)
                GEMM_CFG(4096, 24576,4, 0, 1, 16, 8, 1, 1, 23508, 11595, 402518880, 95851, 68927, 29037)
                GEMM_CFG(4096, 102400,4, 1, 1, 8, 16, 1, 1, 100039, 3425, 1672561500, 346023, 1457, 6361)
                GEMM_CFG(4096, 151936,4, 1, 1, 4, 32, 1, 22, 151750, 9044, 2048659473, 587253, 138280, 523)
                GEMM_CFG(8192, 3072,4, 1, 1, 64, 2, 1, 8101, 1073, 19353, 33899964, 2522, 1586, 117)
                GEMM_CFG(9728, 2560,4, 0, 1, 32, 4, 1, 235, 1090, 33186, 98876956, 3582, 6996, 278)
                GEMM_CFG(11008, 4096,4, 0, 1, 64, 2, 1, 9979, 1611, 14979, 159279418, 7605, 3510, 515)
                GEMM_CFG(12288, 4096,5, 1, 1, 64, 2, 1, 2416, 2370, 22155, 154173960, 2895, 9659, 1819)

                /*
                // graviton 64 线程
                GEMM_CFG(640, 256,8, 0, 1, 16, 4, 1, 105, 254, 23, 84, 885, 463, 638)
                GEMM_CFG(640, 1024,4, 0, 1, 16, 4, 1, 7, 1006, 251, 0, 96, 307, 52)
                GEMM_CFG(640, 4096,8, 0, 1, 16, 4, 1, 222, 4094, 914, 0, 15933, 3933, 256)
                GEMM_CFG(640, 262144,8, 0, 1, 2, 32, 1, 2, 201648, 0, 206, 595395, 5504, 630788)
                GEMM_CFG(1024, 640,4, 0, 1, 16, 4, 1, 298, 639, 4017, 0, 150, 1506, 1251)
                GEMM_CFG(2048, 640,4, 0, 1, 16, 4, 1, 61, 621, 4, 10328, 46, 710, 1225)
                GEMM_CFG(2560, 6144,8, 0, 1, 16, 4, 1, 44, 5922, 6492, 0, 7342, 139, 18)
                GEMM_CFG(2560, 19456,4, 0, 1, 16, 4, 1, 1, 18976, 8580, 3, 463, 1189, 24)
                GEMM_CFG(2560, 151936,4, 1, 1, 4, 16, 1, 2, 150009, 4356, 989, 405, 12041, 145318)
                GEMM_CFG(3072, 3072,4, 0, 1, 16, 4, 1, 15, 2817, 10548, 228, 13, 2069, 311)
                GEMM_CFG(3072, 5120,4, 0, 1, 16, 4, 1, 18, 4474, 11783, 0, 26, 277, 42)
                GEMM_CFG(3072, 16384,8, 0, 1, 16, 4, 1, 2, 15302, 12288, 0, 2683, 63290, 1)
                GEMM_CFG(3072, 128256,4, 0, 1, 2, 32, 1, 1, 123460, 12053, 303, 673, 30737, 98061)
                GEMM_CFG(4096, 2560,4, 0, 1, 16, 4, 1, 21, 2232, 0, 1, 2, 566, 42)
                GEMM_CFG(4096, 4096,4, 0, 1, 16, 4, 1, 29, 4086, 15934, 0, 1031, 7708, 1147)
                GEMM_CFG(4096, 6144,8, 0, 1, 16, 4, 1, 4, 6090, 14604, 0, 2996, 24347, 189)
                GEMM_CFG(4096, 12288,4, 0, 1, 16, 4, 1, 1, 12214, 16375, 2, 36, 31, 116)
                GEMM_CFG(4096, 22016,4, 0, 1, 16, 4, 1, 1, 21986, 16018, 0, 247, 4560, 18)
                GEMM_CFG(4096, 24576,4, 0, 1, 2, 32, 1, 5, 19024, 4247, 3578, 96795, 17061, 47825)
                GEMM_CFG(4096, 102400,8, 1, 1, 2, 32, 1, 1, 58261, 7004, 6890, 2949, 53539, 135165)
                GEMM_CFG(4096, 151936,4, 1, 1, 2, 32, 1, 2, 136144, 39, 345, 561366, 63508, 200471)
                GEMM_CFG(8192, 3072,8, 0, 1, 32, 2, 1, 21, 2804, 7747, 1024, 17, 1200, 3021)
                GEMM_CFG(9728, 2560,8, 0, 1, 16, 4, 1, 15, 2544, 14661, 99, 5, 20, 2)
                GEMM_CFG(11008, 4096,4, 0, 1, 16, 4, 1, 1, 4064, 30919, 0, 1340, 1475, 13577)
                GEMM_CFG(12288, 4096,4, 0, 1, 16, 4, 1, 1, 4083, 49146, 0, 21, 10370, 447)
                */
#undef GEMM_CFG
            }
        }

        // 判断是否应该使用cblas
        // 通过环境变量 ML_LLM_USE_CBLAS 控制，可选值: 1 表示使用cblas，0 表示使用自定义实现
        // 如果环境变量未设置，则使用默认逻辑 (M > 1)
        static bool should_use_cblas(int M) {
            // 使用静态变量缓存环境变量的值，只在第一次运行时读取
            static bool env_checked = false;
            static bool use_cblas_env = false;
            static bool env_is_set = false;

            if (!env_checked) {
                const char* env_use_cblas = std::getenv("ML_LLM_USE_CBLAS");
                if (env_use_cblas != nullptr) {
                    use_cblas_env = (std::string(env_use_cblas) == "1");
                    env_is_set = true;
                }
                env_checked = true;
            }

            if (env_is_set) {
                return use_cblas_env;
            }
            return (M > 1);  // 默认逻辑
        }

        // 获取OMP_NUM_THREADS的值
        // 通过环境变量 OMP_NUM_THREADS 控制
        // 如果环境变量未设置，则使用默认值
        static int get_omp_num_threads(int default_threads) {
            // 使用静态变量缓存环境变量的值，只在第一次运行时读取
            static bool env_checked = false;
            static int omp_num_threads = 0;
            static bool env_is_set = false;

            if (!env_checked) {
                const char* env_omp_threads = std::getenv("OMP_NUM_THREADS");
                if (env_omp_threads != nullptr) {
                    omp_num_threads = std::atoi(env_omp_threads);
                    env_is_set = true;
                }
                env_checked = true;
            }

            if (env_is_set) {
                return omp_num_threads;
            }
            return default_threads;  // 默认值
        }

        void do_gemm_compute_simple_f32(
                float *C,
                int LDC,
                float *A,
                int LDA,
                float *B,
                int LDB,
                int M,
                int N,
                int K,
                float alpha,
                float beta
        )
        {
            // 调用矩阵乘法
            int loop_order = 0;
            int packing_type = 0;
            int mc = M, kc = K, nc = N;
            int m_thread = 1, k_thread = 1, n_thread = 1;
            int gemm_mat_c_prfm_stride = 0;
            int gemm_mat_a_prfm_stride = 0;
            int gemm_mat_b_prfm_stride = 0;
            int add_mat_c_prfm_stride = 0;
            int add_partial_sum_prfm_stride = 0;

            bool use_cblas = should_use_cblas(M);

            if (use_cblas) {
                int thread_num = get_omp_num_threads(1);
                /*
                 std::cout << "use openblas with " << thread_num << " threads." << std::endl;
                 std::cout << "[input] gemm_shape: ["<<M<<", "<<K<<"]x["<<K<<", "<<N<<"]"<<", alpha: "<< alpha << ", beta: "<< beta<<std::endl;
                 */
                openblas_set_num_threads(thread_num);
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    M,
                    N,
                    K,
                    alpha,
                    A,
                    LDA,  // A的列，与是否转置无关
                    B,
                    LDB,  // B的列，与是否转置无关
                    beta,
                    C,
                    LDC  // C的列，与是否转置无关
                );
            } else {
                get_gemm_configs(M, K, N,
                    loop_order, packing_type,
                    m_thread, k_thread, n_thread,
                    mc, kc, nc,
                    gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride, gemm_mat_c_prfm_stride,
                    add_mat_c_prfm_stride, add_partial_sum_prfm_stride);
                /*
                std::cout << "[input] loop_order: "<< get_loop_order_string(static_cast<GemmLoopOrderAroundKernels>(loop_order))<< ", packing_type: " << get_packing_type_string(static_cast<PackingType>(packing_type)) <<std::endl;
                std::cout << "[input] mc: "<<mc<<", kc: "<<kc<<", nc: "<<nc<<std::endl;
                std::cout << "[input] m_thread: "<<m_thread<<", k_thread: "<<k_thread <<", n_thread: "<<n_thread<<std::endl;
                std::cout << "[input] gemm_mat_a_prfm_stride: "<<gemm_mat_a_prfm_stride<<", gemm_mat_b_prfm_stride: "<<gemm_mat_b_prfm_stride<<", gemm_mat_c_prfm_stride: "<<gemm_mat_c_prfm_stride << std::endl;
                std::cout << "[input] add_mat_c_prfm_stride: "<<add_mat_c_prfm_stride << ", add_partial_sum_prfm_stride: "<< add_partial_sum_prfm_stride <<std::endl;
                std::cout << "[input] gemm_shape: ["<<M<<", "<<K<<"]x["<<K<<", "<<N<<"]"<<", alpha: "<< alpha << ", beta: "<< beta<<std::endl;
                */
                storage_layout_cr_ar_br::do_gemm_compute<float>(
                    static_cast<GemmLoopOrderAroundKernels>(loop_order),
                        C,
                        LDC,
                        A,
                        LDA,
                        B,
                        LDB,
                        M,
                        N,
                        K,
                        alpha,
                        beta,
                        mc, kc, nc,
                        m_thread, k_thread, n_thread,
                        gemm_mat_c_prfm_stride,
                        gemm_mat_a_prfm_stride,
                        gemm_mat_b_prfm_stride,
                        add_mat_c_prfm_stride,
                        add_partial_sum_prfm_stride,
                        static_cast<PackingType>(packing_type)
                );
            }
        }
#ifdef __cplusplus
    }
#endif
