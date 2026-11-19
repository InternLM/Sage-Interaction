#!/usr/bin/env bash
set +x
set -euo pipefail

demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
demo_python="${SAGE_DEMO_PYTHON:-python3}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "$demo_python" "$demo_root/launch.py" "$@"
