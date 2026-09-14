#!/usr/bin/env bash
# Prepare the assets shared by both ABC-130K pi0.5 learning-rate runs.
# Called inside the training allocation, after the project venv is activated.
set -euo pipefail

REPO_ROOT="${1:?repository root is required}"
DATASET_ROOT="${2:?dataset root is required}"
FSDP_DEVICES="${3:?FSDP device count is required}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
NORM_STATS_PATH="${4:?normalization statistics file is required}"
NORM_STATS_DIR="$(dirname "${NORM_STATS_PATH}")"

[[ -r "${DATASET_ROOT}/meta/info.json" ]] || {
    echo "ERROR: dataset metadata is missing: ${DATASET_ROOT}/meta/info.json" >&2
    exit 2
}
[[ -s "${NORM_STATS_PATH}" ]] || {
    echo "ERROR: supplied normalization statistics are missing: ${NORM_STATS_PATH}" >&2
    exit 2
}

echo "Checking the allocated GPUs and video decoder."
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" - "${FSDP_DEVICES}" <<'PY'
import sys
import jax
import jax.numpy as jnp
import torchcodec

devices = jax.devices('gpu')
required = int(sys.argv[1])
if len(devices) < required:
    raise RuntimeError(f'FSDP needs {required} GPUs; JAX sees {devices}')
for device in devices:
    with jax.default_device(device):
        result = jax.jit(lambda x: x + 1)(jnp.asarray([1, 2, 3]))
        result.block_until_ready()
    print(f'GPU ready: {device}, {device.device_kind}', flush=True)
print(f'TorchCodec ready: {torchcodec.__version__}', flush=True)
PY

echo "Preparing the public pi0.5 checkpoint and tokenizer (cached after first use)."
"${PYTHON}" - <<'PY'
from openpi.shared import download

for url, options in (
    ('gs://big_vision/paligemma_tokenizer.model', {'gs': {'token': 'anon'}}),
    ('gs://openpi-assets/checkpoints/pi05_base/params', {}),
):
    print(f'Preparing {url}', flush=True)
    print(f'Asset ready: {download.maybe_download(url, **options)}', flush=True)
PY

"${PYTHON}" - "${NORM_STATS_DIR}" <<'PY'
import sys
import numpy as np
from openpi.shared import normalize

stats = normalize.load(sys.argv[1])
for name in ('state', 'actions'):
    for field in ('mean', 'std', 'q01', 'q99'):
        values = np.asarray(getattr(stats[name], field))
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid ABC-130K normalization statistics: {name}.{field}')
print(f'Normalization statistics ready: {sys.argv[1]}/norm_stats.json', flush=True)
PY
