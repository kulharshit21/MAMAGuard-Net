#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
VENV_PATH="${MAMAGUARD_VENV:-$REPO_ROOT/.venv}"
PYTHON_BIN="${MAMAGUARD_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No supported Python interpreter was found (requires Python 3.10-3.12)." >&2
  exit 69
fi
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Unsupported Python {sys.version.split()[0]}; requires 3.10-3.12")
print(f"Using Python {sys.version.split()[0]}")
PY
"$PYTHON_BIN" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REPO_ROOT/requirements/dgx-cu130.txt"
python -m pip install -e "$REPO_ROOT[dev]"
python "$REPO_ROOT/scripts/install_nnunet_plugin.py"
python -m pip check
python -m pip freeze > "$REPO_ROOT/reports/generated/environment.freeze.txt"
python - <<'PY'
import json, platform, torch
from pathlib import Path
payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
Path("reports/generated/torch_environment.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if not payload["cuda_available"]:
    raise SystemExit("CUDA is not available in the new environment")
PY

