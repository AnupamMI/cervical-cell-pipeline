#!/bin/bash
# Step 5: v0.2 commit + fresh-clone smoke test + push/tag (only if everything passes)
set -euo pipefail
"$PY" "$(cd "$(dirname "$0")" && pwd)/step5_push_v02.py"
