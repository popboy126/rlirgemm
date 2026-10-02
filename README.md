# RLirGEMM source release

This directory is the self-contained source package for the RLirGEMM paper.
It publishes the ARM NEON GEMM and vector-add generators, the CMake benchmark,
PPO training and inference code, the RNDS/TPES Optuna search logic, the paper's
training/performance configurations, and the two random shape datasets.

## Published contents

### Core reproducibility materials

- `gemm_codegen/`: configurable ARM GEMM and vector-add kernel generators,
  templates, and generator tests.
- `app_gemm/`: the ARM NEON kernels, wrappers, `gemm_by_opt_gemm` benchmark,
  correctness tests, CMake files, and external-prefix discovery for optional
  baseline and `ml_gemm` integrations.
- `model/RL/`: PPO training, inference, exploration, reward, and parameter
  export code.
- `model/optuna_search/`: RNDS and TPES tuning logic and task runner.
- `configs/rl_train/`, `configs/optuna_search/`, and
  `configs/performance_test/`: the published training, tuning, and RL
  performance configurations.
- `configs/gen_rnd_dataset.py`: generation and validation for the training and
  held-out random shape sets.
- `pyproject.toml`, `uv.lock`, and `.python-version`: the reproducible Python
  environment description.

### Optional integrations and baseline adapters

- `app_gemm/applications/`: TVM and MLC LLM integration scripts, patches,
  configuration generators, and build descriptions.
- `app_gemm/baselines/`: external-prefix CMake adapters, benchmark wrapper
  headers, and version, license, and official acquisition information for the
  listed baseline libraries.

The TVM, MLC LLM, and baseline projects themselves are obtained separately from
their official upstream sources; this release contains the integration material
listed above.

## Python environment

Use Python 3.13 and `uv`. The base `pyproject.toml` deliberately has no TVM or
MLC local-wheel dependency. From this directory run `uv sync --locked`; TVM and
MLC LLM are optional external integrations described below. The lock was
checked with `uv lock --check --offline` after removing the unpublished wheels;
network access is required for a fresh install.

The repository smoke checks for RL training, checkpoint inference and the
RNDS/TPES search used the already provisioned root `pyproject.toml` environment
(`.venv`). A standalone checkout may instead install the published lock file
with the command above; neither workflow includes checkpoints or experiment
results.

## Code generation and tests

From the release root:

```bash
uv run python -m unittest discover -s gemm_codegen/tests
uv run python gemm_codegen/kernel/gemm/run_gemm_kernel_codegen.py --help
uv run python gemm_codegen/kernel/add/run_add_kernel_codegen.py --help
```

Generated assembly should be written to a user-selected output directory; the
release tree is not an output directory.

## ARM benchmark

The supported execution target is Linux/aarch64 with GCC 13.3, CMake 3.28,
OpenMP and an ARMv8 CPU (FT-2000+, Kunpeng 920 or Graviton2). The core build
uses the generated kernels directly and does not require OpenBLAS. Baseline
and `ml_gemm` integrations discover only explicitly supplied prefixes; absent
prefixes are skipped while the core target and tests remain available:

```bash
cmake -S app_gemm -B app_gemm/build -DCMAKE_BUILD_TYPE=Release \
  -DCHECK_GEMM_RESULT=ON
cmake --build app_gemm/build -j
ctest --test-dir app_gemm/build --output-on-failure
```

The resulting `gemm_by_opt_gemm` accepts the configuration format used by
`configs/performance_test`. Thread affinity, warmup and repeat counts are
fields in each JSON configuration and should be reported with any measurement.
`gemm_by_opt_gemm` does not link OpenBLAS. If the optional `ml_gemm` libraries
or the `gemm_by_openblas` adapter are wanted, install OpenBLAS 0.3.30 from its
official project and pass `-DOPENBLAS_ROOT=/path/to/openblas` at configure time;
all other baselines are external and documented in
`app_gemm/baselines/README.md`.

If GoogleTest cannot be fetched in an offline build, configure with
`-DBUILD_GEMM_TESTS=OFF` to compile the benchmark targets only; leave the
option at its default `ON` for the normal CTest workflow.


## Baseline source builds

Every baseline is obtained from its upstream project and built into a user
prefix. The release never downloads, extracts, or installs a baseline during a
CMake configure, and no baseline is installed from `apt`/`apt-get`. Versions,
licenses and official project links are listed in
[`app_gemm/baselines/README.md`](app_gemm/baselines/README.md); follow each
project's official installation instructions. LibShalom and CAKE use the latest
mainline versions; record the actual commit used for each experiment.
ARMPL 25.07 is installed according to Arm's official distribution instructions.

After installing a prefix, pass it explicitly to CMake. Each supplied prefix
creates its corresponding optional adapter target. OpenBLAS additionally
enables the optional `ml_gemm_static`/`ml_gemm_shared` libraries; it is not a
dependency of `gemm_by_opt_gemm`:

```bash
cmake -S app_gemm -B app_gemm/build -DCMAKE_BUILD_TYPE=Release \
  -DOPENBLAS_ROOT=/absolute/path/to/openblas-0.3.30 \
  -DARMPL_ROOT=/absolute/path/to/armpl-25.07 \
  -DEIGEN_ROOT=/absolute/path/to/eigen-3.4.0 \
  -DBLIS_ROOT=/absolute/path/to/blis-2.0 \
  -DLIBXSMM_ROOT=/absolute/path/to/libxsmm-1.17-3834 \
  -DBLASFEO_ROOT=/absolute/path/to/blasfeo-0.1.4.2 \
  -DLIBSHALOM_ROOT=/absolute/path/to/libshalom-prefix \
  -DCAKE_ROOT=/absolute/path/to/cake-prefix
```

The adapter targets are `gemm_by_openblas`, `gemm_by_armpl`,
`gemm_by_eigen`, `gemm_by_blis`, `gemm_by_blis_obj`, `gemm_by_libxsmm`,
`gemm_by_blasfeo`, `gemm_by_libshalom2`,
`gemm_by_libshalom2_mp`, and `gemm_by_cake`. The aggregate target
`gemm_baselines` builds every adapter whose prefix was found. `baselines.cmake`
searches only explicitly supplied prefixes and does not fall back to a system
or package-manager library.

## RL training, inference and export

Five training seeds and the eight A/R/E ablations are in `configs/rl_train`.
The release entry point requires an explicit configuration and output path:

```bash
uv run python model/RL/do_ml_learning.py \
  --config configs/rl_train/ft2000_rl_diff_rnd_seed_gemm_train_t1_task4.json \
  --output-dir /tmp/rlirgemm-task4 --debug
```

Remote credentials and paths in the JSON files are placeholders. Set
`RLIRGEMM_SSH_USER`, `RLIRGEMM_SSH_HOST`, `RLIRGEMM_SSH_PASSWORD`,
`RLIRGEMM_WORK_DIR` and `RLIRGEMM_GEMM_BINARY` only in the local shell used for
an experiment. Do not commit those values. Inference requires a checkpoint
produced by training. First generate an explicit shape-list input (or provide
your own JSON list of `[M,K,N]` triples):

```bash
uv run python configs/gen_rnd_dataset.py --generate \
  --training-output /tmp/rlirgemm-training-shapes.json \
  --heldout-output /tmp/rlirgemm-heldout-shapes.json
```

```bash
uv run python model/RL/do_ml_inference.py --checkpoint /path/to/model.pth \
  --shapes /tmp/rlirgemm-heldout-shapes.json \
  --output /tmp/rlirgemm-inference.json
```

`model/RL/tools/gen_gemm_configs.py` maps inference fields to the 11 runtime
parameter positions in a published performance template. Run the performance
tool with an explicit binary root and output directory.

## Random datasets

The normative training set is the `shape_list` in the five RL training JSONs;
the normative held-out set is `common` in
`performance_with_rl_diff_rnd_seed_portable_on_rnd_shapes_t1_task1_for_ft2000.json`.
Both contain 3,000 `(M,K,N)` shapes. Validate them with:

```bash
uv run python configs/gen_rnd_dataset.py
```

The generator preserves the historical sampler (seed `20260410`), excludes
training shapes when producing held-out candidates, normalises output order,
and checks uniqueness, disjointness, `M <= 30`, `M < N < K`, `K % 8 == 0`,
`N % 16 == 0`, and `4*(M*K + K*N + M*N) <= 4 MiB`. The checked-in JSON remains
the paper input; generated files are optional outputs. The historical inputs
contain a small number of `N == K` shapes (62 in training and 50 in held-out),
which the validator reports without silently changing. Add `--strict` when a
new dataset must satisfy every constraint.
With `--generate`, the script also compares both generated sets with the
normative static sets and stops if either set differs.

## Optuna RNDS/TPES search

The four published task files in `configs/optuna_search` are the paper's
RandomSampler (RNDS) and TPESampler (TPES) trainset/held-out searches. They do
not contain a database URI, host, or result archive. Set
`RLIRGEMM_OPTUNA_DATABASE_URI` and the SSH placeholders locally, then run:

```bash
uv run python model/optuna_search/do_optuna_gemm_tuning_tasks.py \
  "$(pwd)" /tmp/rlirgemm-optuna configs/optuna_search \
  ft2000_optuna_gemm_random_search_trainset_t1_task0.json
```

Results are written to the chosen output directory. Optuna performance task
snapshots, historical results and databases are intentionally outside this
release.

## Optional TVM/MLC integration

`app_gemm/applications` contains only scripts and patches. Enable an integration
only after checking out the corresponding external TVM or MLC LLM source and
passing `-DTVM_SOURCE_DIR=...` or `-DMLC_LLM_SOURCE_DIR=...`. The `.tgz` files
and the paper's exact external commit are not bundled here; record the commit,
submodules, patch order and build command in the experiment record.

The task4 policy was selected as the training-set median (reported as 9.585
GFLOPS). Re-training must select the median by the same rule before held-out
evaluation; the package does not claim bitwise reproduction of paper numbers.
