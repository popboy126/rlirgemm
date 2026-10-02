

set(GEMM_KERNELS_ARM_NEON_HOME "${GEMM_KERNELS_ARM_HOME}/neon")

set(GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr64_kr8.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr16_kr8.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr16_kr8.S"
)

set(GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS}
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr64_kr8.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr1_kr5.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr5_kr5.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr7_kr1.S"
    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr8_kr8.S"

    "${GEMM_KERNELS_ARM_NEON_HOME}/mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr2_nr4_kr4.S"
)


