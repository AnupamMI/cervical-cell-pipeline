#!/bin/bash
# Step 4: winner -> nc_mask_pipeline_v3, tests, 1-FOV + 5-FOV production runs with sheets
set -euo pipefail
"$PY" "$(cd "$(dirname "$0")" && pwd)/step4_pipeline_v3.py"
