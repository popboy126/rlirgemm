# Baseline dependencies

This release contains only baseline adapter entry points; it does not include
third-party source trees, archives, or build artifacts. Download each baseline
from the upstream project listed below and build/install it on the target host
under an independent prefix. CMake only discovers these prefixes; it never
downloads, extracts, or invokes `apt` or `apt-get`. For every experiment,
record `git rev-parse HEAD` (or the vendor package version), the compiler, and
the build configuration.

## Versions, upstream links, and licenses

| Baseline | Paper version | Upstream project | License/acquisition |
| --- | --- | --- | --- |
| OpenBLAS | 0.3.30 (`v0.3.30`) | <https://github.com/OpenMathLib/OpenBLAS> | BSD-3-Clause |
| ARMPL | 25.07 | <https://developer.arm.com/downloads/-/arm-performance-libraries> | Arm commercial license; download the official package and accept its EULA; not a system package |
| Eigen | 3.4.0 (`3.4.0`) | <https://gitlab.com/libeigen/eigen> | MPL-2.0 |
| BLIS | 2.0 (`2.0`) | <https://github.com/flame/blis> | BSD-3-Clause |
| LIBXSMM | `feature_sw_pipeline_panels-1.17-3834` | <https://github.com/libxsmm/libxsmm> | BSD-3-Clause |
| BLASFEO | 0.1.4.2 | <https://github.com/giaf/blasfeo> | BSD-3-Clause |
| LibShalom | Latest mainline version | <https://github.com/AnonymousYWL/TPDS> | See the repository `LICENSE` |
| CAKE | Latest mainline version | <https://github.com/vnatesh/CAKE_on_CPU> | See the repository `LICENSE` |
| autoGEMM | Latest mainline version | <https://github.com/wudu98/autoGEMM> | See the repository `LICENSE` |

## Installation

Use the official project links above and follow each project's installation
guide or README to obtain and install the specified version. Install ARMPL
according to Arm's official distribution instructions. After installation,
pass the actual prefix to the CMake parameters below. OpenBLAS is used only
when `gemm_by_openblas` or the `ml_gemm_static`/`ml_gemm_shared` integration
libraries are requested.

## CMake prefixes

The core `gemm_by_opt_gemm` target does not require OpenBLAS. OpenBLAS is
needed only for the optional `gemm_by_openblas` adapter and `ml_gemm`
integration libraries. The paths below are placeholders; replace them with
actual installation prefixes and omit optional baselines that are not used:

```bash
cmake -S app_gemm -B app_gemm/build -DCMAKE_BUILD_TYPE=Release \
  -DOPENBLAS_ROOT="/absolute/path/to/openblas-0.3.30" \
  -DARMPL_ROOT="/absolute/path/to/armpl-25.07" \
  -DEIGEN_ROOT="/absolute/path/to/eigen-3.4.0" \
  -DBLIS_ROOT="/absolute/path/to/blis-2.0" \
  -DLIBXSMM_ROOT="/absolute/path/to/libxsmm-1.17-3834" \
  -DBLASFEO_ROOT="/absolute/path/to/blasfeo-0.1.4.2" \
  -DLIBSHALOM_ROOT="/absolute/path/to/libshalom-prefix" \
  -DCAKE_ROOT="/absolute/path/to/cake-on-cpu-prefix"
```

`app_gemm/baselines/baselines.cmake` creates an adapter target for every
complete prefix that is supplied (for example, `gemm_by_blis`,
`gemm_by_blasfeo`, `gemm_by_libshalom2`, and `gemm_by_cake`). The aggregate
target `gemm_baselines` builds all discovered adapters. If an optional prefix
is not supplied, its target is skipped and the core target is unaffected.
`gemm_by_opt_gemm` uses the generated kernels and does not link OpenBLAS;
setting `OPENBLAS_ROOT` enables `gemm_by_openblas` and the `ml_gemm` libraries.

`autoGEMM` is listed only with its official address and version information;
the release does not provide an autoGEMM CMake target. Build it independently
according to its upstream README.

If the validation host cannot access the GoogleTest upstream repository, add
`-DBUILD_GEMM_TESTS=OFF` during configuration. This disables only the test
targets and does not affect baseline adapter compilation; when network access
is available or GoogleTest is prepared locally, keep the default `ON` and run
CTest.
