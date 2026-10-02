//
// Created by huche on 2026/2/4.
//

#ifndef APP_GEMM_CUSTOM_GEMM_RUNTIME_H
#define APP_GEMM_CUSTOM_GEMM_RUNTIME_H

#include <tvm/ffi/function.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

//#include <algorithm>
#include <string>

namespace tvm {
namespace contrib {

using ffi::Any;
using ffi::PackedArgs;
using runtime::TypeMatch;

inline int ColumnStride(const DLTensor* tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return tensor->strides[0];
  } else {
    return tensor->shape[1];
  }
}

inline int ElementStride(const DLTensor* tensor) {
  if (tensor->strides) {
    return tensor->strides[1];
  } else {
    return 1;
  }
}

// Call a row major blas.  Note that data is stored in tvm as row
// major
template <typename TGemmOp>
inline void CallTeCustomSimpleGemm(ffi::PackedArgs args, ffi::Any* ret, TGemmOp op) {
  auto A = args[0].cast<DLTensor*>();
  TVM_FFI_ICHECK(TypeMatch(A->dtype, kDLFloat, 32)) << "Only float32 is supported in custom_gemm";
  auto B = args[1].cast<DLTensor*>();
  auto C = args[2].cast<DLTensor*>();
  int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
  TVM_FFI_ICHECK_EQ(A->ndim, 2);
  TVM_FFI_ICHECK_EQ(B->ndim, 2);
  TVM_FFI_ICHECK_EQ(C->ndim, 2);
  TVM_FFI_ICHECK_EQ(A->shape[0], C->shape[0]);  // 检查各维度是否相等
  TVM_FFI_ICHECK_EQ(B->shape[1], C->shape[1]);
  TVM_FFI_ICHECK_EQ(A->shape[1], B->shape[0]);

  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  TVM_FFI_ICHECK_EQ(ElementStride(A), 1);
  TVM_FFI_ICHECK_EQ(ElementStride(B), 1);
  TVM_FFI_ICHECK_EQ(ElementStride(C), 1);

  int lda = ColumnStride(A);
  int ldb = ColumnStride(B);
  int ldc = ColumnStride(C);

  TVM_FFI_ICHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  TVM_FFI_ICHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));

  float alpha = args.size() > 3 ? args[3].cast<float>(): 1.0;
  float beta =  args.size() > 4 ? args[4].cast<float>(): 0.0;

  auto c_ptr = reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(C->data) + C->byte_offset);
  auto a_ptr = reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(A->data) + A->byte_offset);
  auto b_ptr = reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(B->data) + B->byte_offset);
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

template <typename TGemmOp>
inline void CallTeCustomGemm(ffi::PackedArgs args, ffi::Any* ret, TGemmOp op) {
  auto A = args[0].cast<DLTensor*>();
  TVM_FFI_ICHECK(TypeMatch(A->dtype, kDLFloat, 32)) << "Only float32 is supported in custom_gemm";
  auto B = args[1].cast<DLTensor*>();
  auto C = args[2].cast<DLTensor*>();
  int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
  TVM_FFI_ICHECK_EQ(A->ndim, 2);
  TVM_FFI_ICHECK_EQ(B->ndim, 2);
  TVM_FFI_ICHECK_EQ(C->ndim, 2);
  TVM_FFI_ICHECK_EQ(A->shape[0], C->shape[0]);  // 检查各维度是否相等
  TVM_FFI_ICHECK_EQ(B->shape[1], C->shape[1]);
  TVM_FFI_ICHECK_EQ(A->shape[1], B->shape[0]);

  int m = A->shape[0];
  int k = A->shape[1];
  int n = B->shape[1];

  TVM_FFI_ICHECK_EQ(ElementStride(A), 1);
  TVM_FFI_ICHECK_EQ(ElementStride(B), 1);
  TVM_FFI_ICHECK_EQ(ElementStride(C), 1);

  int lda = ColumnStride(A);
  int ldb = ColumnStride(B);
  int ldc = ColumnStride(C);

  TVM_FFI_ICHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  TVM_FFI_ICHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
  float alpha = args[3].cast<float>();
  float beta = args[4].cast<float>();

  unsigned short loop_order = args[5].cast<unsigned short>();
  unsigned short packing_type = args[6].cast<unsigned short>();

  int mc = args[7].cast<int>();
  int kc = args[8].cast<int>();
  int nc = args[9].cast<int>();

  int m_thread = args[10].cast<int>();
  int k_thread = args[11].cast<int>();
  int n_thread = args[12].cast<int>();

  int gemm_mat_c_prfm_stride = args[13].cast<int>();
  int gemm_mat_a_prfm_stride = args[14].cast<int>();
  int gemm_mat_b_prfm_stride = args[15].cast<int>();

  int add_mat_c_prfm_stride = args[16].cast<int>();
  int add_partial_sum_prfm_stride = args[17].cast<int>();


  auto c_ptr = reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(C->data) + C->byte_offset);
  auto a_ptr = reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(A->data) + A->byte_offset);
  auto b_ptr = reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(B->data) + B->byte_offset);
  op(
	loop_order,
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
    packing_type
    );
}

}  // namespace contrib
}  // namespace tvm

#endif //APP_GEMM_CUSTOM_GEMM_RUNTIME_H
