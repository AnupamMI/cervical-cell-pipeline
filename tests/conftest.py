import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "tinyseg_retrain"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

# Original (unmodified) tinyseg_retrain + nc_pipeline, used only by the parity / provenance tests.
# Override with NCMASK_TINYSEG_DIR / NCMASK_NC_PIPELINE_DIR if they live elsewhere.
TINYSEG_DIR = Path(os.environ.get("NCMASK_TINYSEG_DIR",
                                  ROOT.parent / "seg_audit/extracted/tinyseg_retrain/tinyseg_retrain"))
NC_PIPELINE_DIR = Path(os.environ.get("NCMASK_NC_PIPELINE_DIR", ROOT.parent / "nc_pipeline"))


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    from ncmask import config
    return config.load(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def tinyseg_dir():
    if not (TINYSEG_DIR / "tinyseg.py").is_file():
        pytest.skip(f"original tinyseg_retrain not found at {TINYSEG_DIR}")
    return TINYSEG_DIR
