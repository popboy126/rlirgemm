//
// Created by huche on 2026/2/4.
//

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>
// #include <tvm/runtime/logging.h>

extern "C" {
    #include "ml_gemm.h"
}

#include "custom_gemm_runtime.h"
#include <tvm/runtime/tensor.h>

namespace tvm
{
    namespace contrib
    {
        using namespace runtime;

        struct CustomSgemmSimpleOp {
            typedef float TDatatype;
            void operator()(
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
            ) {
                // LOG(DEBUG) << "Call CustomSgemmSimpleOp with M=" << M << ", N=" << N << ", K=" << K << ", LDA=" << LDA << ", LDB=" << LDB << ", LDC=" << LDC << ", alpha=" << alpha << ", beta=" << beta;
                do_gemm_compute_simple_f32(
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
                    beta
            );
            }
        };

		struct CustomSgemmOp {
            typedef float TDatatype;
            void operator()(
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
            ) {
      //           LOG(DEBUG) << "Call CustomSgemmOp with M=" << M << ", N=" << N << ", K=" << K << ", LDA=" << LDA << ", LDB="
						// << LDB << ", LDC=" << LDC << ", alpha=" << alpha << ", beta=" << beta << ", loop_order=" << loop_order << ", mc=" << mc << ", kc=" << kc << ", nc=" << nc
      //                   << ", m_thread=" << m_thread << ", k_thread=" << k_thread << ", n_thread=" << n_thread << ", gemm_mat_c_prfm_stride=" << gemm_mat_c_prfm_stride
      //                   << ", gemm_mat_a_prfm_stride=" << gemm_mat_a_prfm_stride << ", gemm_mat_b_prfm_stride=" << gemm_mat_b_prfm_stride << ", add_mat_c_prfm_stride=" << add_mat_c_prfm_stride
      //                   << ", add_partial_sum_prfm_stride=" << add_partial_sum_prfm_stride << ", packing_type=" << packing_type;

                do_gemm_compute_f32(
                    loop_order,
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
                	packing_type
            );
            }
        };

    inline void CallTirCustomSimpleGemm(tvm::runtime::Tensor A, tvm::runtime::Tensor B, tvm::runtime::Tensor C, float alpha, float beta) {
	  TVM_FFI_ICHECK(TypeMatch(A->dtype, kDLFloat, 32)) << "Only float32 is supported in custom_gemm";
      int bit_depth = sizeof(CustomSgemmSimpleOp::TDatatype) * 8;
      TVM_FFI_ICHECK_EQ(A->ndim, 2);
      TVM_FFI_ICHECK_EQ(B->ndim, 2);
      TVM_FFI_ICHECK_EQ(C->ndim, 2);
      TVM_FFI_ICHECK_EQ(A->shape[0], C->shape[0]);  // 检查各维度是否相等
      TVM_FFI_ICHECK_EQ(B->shape[1], C->shape[1]);
      TVM_FFI_ICHECK_EQ(A->shape[1], B->shape[0]);

      int m = A->shape[0];
      int k = A->shape[1];
      int n = B->shape[1];

      TVM_FFI_ICHECK_EQ(ElementStride(A.GetDLTensorPtr()), 1);
      TVM_FFI_ICHECK_EQ(ElementStride(B.GetDLTensorPtr()), 1);
      TVM_FFI_ICHECK_EQ(ElementStride(C.GetDLTensorPtr()), 1);

      int lda = ColumnStride(A.GetDLTensorPtr());
      int ldb = ColumnStride(B.GetDLTensorPtr());
      int ldc = ColumnStride(C.GetDLTensorPtr());

      TVM_FFI_ICHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
      TVM_FFI_ICHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));

      auto c_ptr = reinterpret_cast<typename CustomSgemmSimpleOp::TDatatype*>(static_cast<char*>(C->data) + C->byte_offset);
      auto a_ptr = reinterpret_cast<typename CustomSgemmSimpleOp::TDatatype*>(static_cast<char*>(A->data) + A->byte_offset);
      auto b_ptr = reinterpret_cast<typename CustomSgemmSimpleOp::TDatatype*>(static_cast<char*>(B->data) + B->byte_offset);
	  CustomSgemmSimpleOp op;
      op(
            c_ptr,
            ldc,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            m,
            n,
            k,
            alpha,
            beta
        );
    }

	inline void CallTirCustomGemm(tvm::runtime::Tensor A, tvm::runtime::Tensor B, tvm::runtime::Tensor C, float alpha, float beta,
		int loop_order, int packing_type, int mc, int kc, int nc, int m_thread, int k_thread, int n_thread,
		int gemm_mat_c_prfm_stride, int gemm_mat_a_prfm_stride, int gemm_mat_b_prfm_stride, int add_mat_c_prfm_stride, int add_partial_sum_prfm_stride) {
	  TVM_FFI_ICHECK(TypeMatch(A->dtype, kDLFloat, 32)) << "Only float32 is supported in custom_gemm";
      int bit_depth = sizeof(CustomSgemmOp::TDatatype) * 8;
      TVM_FFI_ICHECK_EQ(A->ndim, 2);
      TVM_FFI_ICHECK_EQ(B->ndim, 2);
      TVM_FFI_ICHECK_EQ(C->ndim, 2);
      TVM_FFI_ICHECK_EQ(A->shape[0], C->shape[0]);  // 检查各维度是否相等
      TVM_FFI_ICHECK_EQ(B->shape[1], C->shape[1]);
      TVM_FFI_ICHECK_EQ(A->shape[1], B->shape[0]);

      int m = A->shape[0];
      int k = A->shape[1];
      int n = B->shape[1];

      TVM_FFI_ICHECK_EQ(ElementStride(A.GetDLTensorPtr()), 1);
      TVM_FFI_ICHECK_EQ(ElementStride(B.GetDLTensorPtr()), 1);
      TVM_FFI_ICHECK_EQ(ElementStride(C.GetDLTensorPtr()), 1);

      int lda = ColumnStride(A.GetDLTensorPtr());
      int ldb = ColumnStride(B.GetDLTensorPtr());
      int ldc = ColumnStride(C.GetDLTensorPtr());

      TVM_FFI_ICHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
      TVM_FFI_ICHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));

      auto c_ptr = reinterpret_cast<typename CustomSgemmOp::TDatatype*>(static_cast<char*>(C->data) + C->byte_offset);
      auto a_ptr = reinterpret_cast<typename CustomSgemmOp::TDatatype*>(static_cast<char*>(A->data) + A->byte_offset);
      auto b_ptr = reinterpret_cast<typename CustomSgemmOp::TDatatype*>(static_cast<char*>(B->data) + B->byte_offset);
	  CustomSgemmOp op;
      op(
        (unsigned short)loop_order,
        c_ptr,
        ldc,
        a_ptr,
        lda,
        b_ptr,
        ldb,
        m,
        n,
        k,
        alpha,
        beta,
        mc, kc, nc,
        m_thread, k_thread, n_thread,
        gemm_mat_c_prfm_stride, gemm_mat_a_prfm_stride, gemm_mat_b_prfm_stride,
        add_mat_c_prfm_stride, add_partial_sum_prfm_stride,
        (unsigned short)packing_type
        );
    }

        // matrix multiplication for row major
        TVM_FFI_STATIC_INIT_BLOCK() {
            namespace refl = tvm::ffi::reflection;
            refl::GlobalDef()
                .def_packed("tvm.contrib.custom.te_simple_matmul",
                            [](ffi::PackedArgs args, ffi::Any* ret) {
                                CallTeCustomSimpleGemm(args, ret, CustomSgemmSimpleOp());
                                })
				.def_packed("tvm.contrib.custom.te_matmul",
                            [](ffi::PackedArgs args, ffi::Any* ret) {
                                CallTeCustomGemm(args, ret, CustomSgemmOp());
                                })
				.def("tvm.contrib.custom.tir_simple_matmul", CallTirCustomSimpleGemm)
				.def("tvm.contrib.custom.tir_matmul", CallTirCustomGemm)
				.def_packed("tvm.contrib.custom.te_debug",
                            [](ffi::PackedArgs args, ffi::Any* ret) {
                                float alpha = args[0].cast<float>();
  							    float beta =  args[1].cast<float>();
						LOG(DEBUG) << "Call tvm.contrib.custom.te_debug successfully with alpha=" << alpha << ", beta=" << beta;
                })
				.def("tvm.contrib.custom.tir_debug",
                            [](float alpha, float beta) {
						LOG(DEBUG) << "Call tvm.contrib.custom.tir_debug successfully with alpha=" << alpha << ", beta=" << beta;
                });
        }
    }
}
