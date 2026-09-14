# pi0.5 ABC bottle-bin checkpoint: loading and robot inference

This guide loads **step 16000** of `pi05_abc_put_the_plastic_bottles_in_the_bin_17183582`, trained for **`put the plastic bottles in the bin`**. It uses the OpenPI/JAX policy and YAM transforms in the author's `yam-abc-reproduce` checkout.

Hugging Face destination: [MikhailBird/pi05-abc-130k-task-finetune](https://huggingface.co/MikhailBird/pi05-abc-130k-task-finetune).

Original checkpoint:

```text
/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce/third_party/policy/openpi/checkpoints/pi05_abc130k_task/pi05_abc_put_the_plastic_bottles_in_the_bin_17183582/16000
```


## 1. Checkpoint format and configuration

| Setting | Value |
| --- | --- |
| Inference config | `pi05_abc130k` |
| Model | `Pi0Config(pi05=True)`, full weights, JAX/Orbax OCDBT |
| Task prompt | `put the plastic bottles in the bin` |
| Normalization file | `assets/put_the_plastic_bottles_in_the_bin/norm_stats.json` |
| State | `(14,)`, two YAM arms |
| Returned actions | `(50, 14)`, absolute joint/gripper position targets |
| Internal model action width | 32; the output transform removes padding |
| Dataset/control rate | 30 Hz |
| Existing evaluation's executed chunk | First 15 rows, then observe and infer again |

`pi05_abc130k_task` is constructed dynamically by the training launcher; it is not a registered inference config. `pi05_abc130k` has the same architecture and transforms. **Supply the task normalization file explicitly**: the base config otherwise looks for `abc130k_yam` assets.

The checkpoint contains 40 files totaling 44,656,448,068 bytes (about 41.59 GiB). Its directory layout is:

```text
CHECKPOINT_DIR/
  _CHECKPOINT_METADATA
  params/                  # Inference parameters: keep this entire directory
    _METADATA
    _sharding
    manifest.ocdbt
    array_metadatas/...
    d/...
    ocdbt.process_0/...
  assets/
    put_the_plastic_bottles_in_the_bin/norm_stats.json
  train_state/             # Training state; unnecessary for inference
```

Pass `CHECKPOINT_DIR` to the loader, not `CHECKPOINT_DIR/params`. Keep every file under `params/`, including all nested shard files. This is not a Transformers `from_pretrained` or PyTorch `.safetensors` checkpoint. The inference weights are self-contained; loading them does not require the training dataset or separately downloading the pi0.5 base weights. The tokenizer is a separate cached dependency, described below.

## 2. Prepare the GPU server

Use the author's fork, which includes `LeRobotYamDataConfig`, `YamInputs`, and `YamOutputs`. A plain upstream OpenPI checkout does not supply this task configuration. The inspected repository commit is `fa9c447e89beb978c44d9e06806ec161f61bba31`.

For a fresh checkout, with access to the fork and its i2rt submodule:

```bash
git clone --recurse-submodules git@github.com:ninghaolu/yam-abc-reproduce.git
cd yam-abc-reproduce
git checkout fa9c447e89beb978c44d9e06806ec161f61bba31
git submodule update --init --recursive
export YAM_REPO="$PWD"
uv sync --extra deploy --group openpi
```

For the existing cluster installation, simply use:

```bash
export YAM_REPO=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
```

Run inference on a Linux NVIDIA GPU machine with a driver compatible with the checkout's CUDA 12 JAX dependencies. The existing evaluation launcher requests one L40S. This guide does not establish a minimum VRAM requirement. The inspected environment has Python 3.12.12, JAX/JAXlib 0.5.3, Flax 0.10.2, Orbax-checkpoint 0.11.13, and NumPy 2.2.6. Use the repository's environment/lockfile; `uv sync` removes extras/groups omitted from its command, so preserve any additional groups needed in an existing environment.

Check the environment on the GPU host:

```bash
"$YAM_REPO/.venv/bin/python" -c 'import jax; from openpi.training import config; c=config.get_config("pi05_abc130k"); print(jax.devices()); print(c.model)'
```

The device list should contain a GPU. Installing `hf` on a Mac is useful for transferring files; it does not install this CUDA inference environment.

## 3. Obtain the checkpoint

Download the inference files from the repository above:

```bash
# Install the CLI if needed: brew install hf
# Or, in a separate CLI environment: python -m pip install -U huggingface_hub
# hf auth login  # Needed for a private repository; public downloads need no login.

export CKPT_DIR="$YAM_REPO/checkpoints/pi05-abc-130k-task-finetune-16000"
hf download MikhailBird/pi05-abc-130k-task-finetune \
  --local-dir "$CKPT_DIR" \
  --include 'params/*' 'assets/*' '_CHECKPOINT_METADATA' \
            'README.md' 'inference/*' 'checkpoint_info.json'
```

The Hub repository root corresponds directly to local step `16000`; do not append another `/16000` to the downloaded directory. The selective download skips roughly 30 GiB of training state. Omit `--include` to download the complete checkpoint.

To use the original cluster checkpoint instead:

```bash
export CKPT_DIR="$YAM_REPO/third_party/policy/openpi/checkpoints/pi05_abc130k_task/pi05_abc_put_the_plastic_bottles_in_the_bin_17183582/16000"
```

In either case:

```bash
export NORM_STATS_PATH="$CKPT_DIR/assets/put_the_plastic_bottles_in_the_bin/norm_stats.json"
test -s "$CKPT_DIR/params/manifest.ocdbt"
test -s "$NORM_STATS_PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME="$YAM_REPO/.cache/openpi-inference"
export FSSPEC_GS='{"session_kwargs":{"trust_env":true}}'
```

On the original cluster, reuse `OPENPI_DATA_HOME=/projects/work/yang-lab/projects/policy-finetuning/openpi-cache/openpi-assets` instead. On a fresh machine, the first policy creation downloads `gs://big_vision/paligemma_tokenizer.model` into `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`. Allow that download once, or copy the cached tokenizer to that exact location before running offline. `HF_HUB_OFFLINE=1` does not populate the OpenPI tokenizer cache.

## 4. Start the policy server

The handoff includes `inference/openpi_server.py`, a snapshot of the author's current server with `--norm-stats-path` support. Use that snapshot with the pinned source checkout: the author's working server has changes beyond the Git commit above.

```bash
# Downloaded handoff:
export SERVER_SCRIPT="$CKPT_DIR/inference/openpi_server.py"

# Original cluster checkout: use this instead.
# export SERVER_SCRIPT="$YAM_REPO/yam_abc_reproduce/deploy/servers/openpi_server.py"

cd "$YAM_REPO/third_party/policy/openpi"
"$YAM_REPO/.venv/bin/python" "$SERVER_SCRIPT" \
  --host 127.0.0.1 --port 8000 \
  --config pi05_abc130k \
  --checkpoint "$CKPT_DIR" \
  --norm-stats-path "$NORM_STATS_PATH" \
  --prompt 'put the plastic bottles in the bin' \
  --image-key-map 'top=image,left=left_wrist,right=right_wrist' \
  --flatten-prefix 'observation/' \
  --state-key 'observation/state'
```

Loading and the first JAX compilation can take a few minutes. The server performs a dummy warm-up. In another terminal:

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/healthz
```

Health checks confirm that the server is listening; the server catches warm-up failures, so also run the inference check below. For a separate robot computer, bind to the GPU host's private interface, or keep loopback binding and tunnel from the robot computer:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@gpu-host
```

The tunnel lets the robot client continue using `127.0.0.1:8000`.

## 5. Send one observation and inspect the action chunk

The server accepts unbatched NumPy arrays with this contract:

| Observation field | Meaning |
| --- | --- |
| `images["top"]` | Overhead/third-person RGB view |
| `images["left"]` | Left wrist RGB view |
| `images["right"]` | Right wrist RGB view |
| Each image | `(height, width, 3)` `uint8`, RGB values 0–255 |
| `state[0:6]` | Left arm joints, radians |
| `state[6]` | Left gripper: normalized 0 closed, 1 open |
| `state[7:13]` | Right arm joints, radians |
| `state[13]` | Right gripper: normalized 0 closed, 1 open |
| `prompt` | `put the plastic bottles in the bin` |

Send all three real views in their training roles. OpenCV camera frames usually need BGR-to-RGB conversion. OpenPI resizes images with padding to 224 × 224, normalizes state internally using the task quantiles, and unnormalizes predictions. Supply raw measured state, without manual normalization or 32-dimensional padding.

Run this hardware-free request on a machine with the repository's client dependencies installed:

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
"$YAM_REPO/.venv/bin/python" - <<'PY'
import numpy as np
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)
print("server:", client.get_server_metadata())
obs = {
    "images": {
        role: np.zeros((480, 640, 3), dtype=np.uint8)
        for role in ("top", "left", "right")
    },
    "state": np.zeros(14, dtype=np.float32),
    "prompt": "put the plastic bottles in the bin",
}
result = client.infer(obs)
actions = np.asarray(result["actions"], dtype=np.float32)
assert actions.shape == (50, 14), actions.shape
assert np.isfinite(actions).all()
print("actions:", actions.shape, actions.dtype)
print("first target:", actions[0])
# Synthetic observations test loading and transport only. Do not execute these targets.
PY
```

For actual inference, replace the zero images with synchronized camera frames and the zero state with current measured follower joints/grippers. Keep the policy/client alive between requests.

Each returned row has the same 14-dimensional ordering as the state. **The returned joint values are already absolute position targets.** Training uses joint deltas internally, but `create_trained_policy` applies the inverse transform using the observation state. Do not add the state again, integrate rows as velocities, or unnormalize the output again. Gripper outputs remain absolute normalized targets and can overshoot their nominal range; the robot adapter should enforce its calibrated limits.

## 6. Load the policy directly in Python

To run inference in the GPU process without WebSockets, keep the environment variables from section 3 and run:

```bash
"$YAM_REPO/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import numpy as np
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config

checkpoint = Path(os.environ["CKPT_DIR"]).resolve()
norm_path = checkpoint / "assets/put_the_plastic_bottles_in_the_bin/norm_stats.json"
policy = policy_config.create_trained_policy(
    config.get_config("pi05_abc130k"),
    checkpoint,
    default_prompt="put the plastic bottles in the bin",
    norm_stats=normalize.deserialize_json(norm_path.read_text()),
)

# Direct policy input uses the flattened model keys, unlike the WebSocket wrapper.
# Replace these synthetic inputs with real observations for deployment.
obs = {
    "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/state": np.zeros(14, dtype=np.float32),
    "prompt": "put the plastic bottles in the bin",
}
actions = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
assert actions.shape == (50, 14), actions.shape
assert np.isfinite(actions).all()
print(actions.shape, actions[0])
PY
```

This creates the complete input/output transform pipeline. Raw model sampling bypasses the YAM transforms and would require implementing them yourself.

## 7. Connect the existing YAM robot client

On a separate robot computer, install the client/camera dependencies in its own checkout:

```bash
cd "$YAM_REPO"
uv sync --extra camera --extra deploy
```

If server and robot share one environment, keep `--group openpi` in the sync command as well. Configure `configs/station_yam.yaml` for **left arm then right arm**, 30 Hz, and the robot's calibrated joints/grippers. Configure the camera serials and `top`, `left`, `right` roles in `configs/cameras.yaml`. Match the ABC training camera placements and robot coordinate conventions; shape compatibility alone does not establish that a different robot setup is compatible.

With the server ready, the existing client command is:

```bash
cd "$YAM_REPO"
.venv/bin/yam-abc-deploy \
  --station configs/station_yam.yaml \
  --host 127.0.0.1 --port 8000 \
  --prompt 'put the plastic bottles in the bin' \
  --open-loop-horizon 15 \
  --seconds 10 \
  --ramp-seconds 2 \
  --max-joint-speed 0.3
```

**This command moves the configured robot.** Inspect the live inputs and predictions first and use your station's normal supervised bring-up and stop procedure. `0.3 rad/s` is an example conservative arm-speed setting, not a validated operating limit for this task. The loop's speed clamp applies to arm joints; it does not clamp grippers or provide collision avoidance. No real-robot validation is claimed for this checkpoint by this guide.

The client executes up to 15 rows per request, then refreshes the observation. At 30 Hz that is 0.5 seconds of planned actions plus synchronous inference time. Its default client also blends the first four rows at chunk boundaries, so it is not identical to the simulator's direct action replay. Leave `--rtc` off: this OpenPI wrapper does not implement prefix-conditioned RTC. For another robot stack, split each row into `action[:7]` and `action[7:14]`, then map those absolute targets through that robot's calibrated controller, limits, and stop handling.

## 8. Reuse the original MuJoCo launcher

On the original cluster, to explicitly test this checkpoint with the existing simulator:

```bash
cd /projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
CHECKPOINT="$PWD/third_party/policy/openpi/checkpoints/pi05_abc130k_task/pi05_abc_put_the_plastic_bottles_in_the_bin_17183582/16000" \
CONFIG_NAME=pi05_abc130k \
NORM_STATS_ASSET_ID=put_the_plastic_bottles_in_the_bin \
PROMPT='put the plastic bottles in the bin' \
NUM_CHUNKS=2 sbatch slurm/eval_openpi_mujoco.bash
```

That launcher also requires the existing separate `abc/.venv` simulator environment. A successful synthetic inference or MuJoCo rollout does not measure real-robot task success.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Unknown config `pi05_abc130k_task` | Use `pi05_abc130k` with the explicit task normalization file. |
| Unknown config `pi05_abc130k` | Install the author's fork/environment containing the YAM config. |
| Missing `assets/abc130k_yam/norm_stats.json` | Pass `--norm-stats-path`, or `norm_stats=...` in Python. |
| Unrecognized `--norm-stats-path` | Use the provided server snapshot or the current original working checkout. |
| Missing `observation/image` or wrist key | Use the three server mapping flags and send all three camera roles. |
| State dimension mismatch | This checkpoint expects both arms, left first, 14 values including grippers. |
| Orbax missing manifest/shard | Download the whole `params/` tree and point the loader at its parent. |
| Tokenizer/network error during startup | Populate the PaliGemma tokenizer cache in section 3. |
| Only CPU devices / very slow inference | Check the GPU allocation, driver, and CUDA-enabled JAX environment. |
| HTTP/WebSocket proxy error | Set both `NO_PROXY` and `no_proxy` for the policy server address. |

Sources: the local `slurm/eval_openpi_mujoco.bash`, `slurm/finetune_abc_pi05_task.bash`, `yam_abc_reproduce/deploy/{servers/openpi_server.py,contract.py,client.py,loop.py,run.py}`, and vendored OpenPI `src/openpi/{training/config.py,policies/policy_config.py,policies/yam_policy.py,models/pi0_config.py,models/tokenizer.py}`. Transfer commands follow the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).
