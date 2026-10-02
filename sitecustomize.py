"""Make generator's legacy absolute imports work from the release root."""

from pathlib import Path
import sys

_CODEGEN_ROOT = Path(__file__).resolve().parent / "gemm_codegen"
if str(_CODEGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_ROOT))
