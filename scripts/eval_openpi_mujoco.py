#!/usr/bin/env python3
"""Evaluate an OpenPI YAM policy server in the released ABC MuJoCo task.

This process intentionally contains no JAX/OpenPI model code. The policy is
served by ``openpi_server.py`` from the OpenPI environment, while this client
uses the independently validated ABC/MuJoCo-Warp environment.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
ABC_ROOT = WORKSPACE_ROOT / "abc"
OPENPI_CLIENT_SRC = REPO_ROOT / "third_party" / "policy" / "openpi" / "packages" / "openpi-client" / "src"
for source_root in (ABC_ROOT, OPENPI_CLIENT_SRC):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

from abc_minimal.config import PutBottlesSimConfig  # noqa: E402
from abc_minimal.eval_policy import PutBottlesEnv, jsonable, video_frame  # noqa: E402
from abc_minimal.gripper_diagnostics import GripperDiagnostics  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

CAMERA_KEYS = ("top", "left", "right")
STATE_DIM = 14


@dataclass
class OpenPIMujocoEvalConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 8001
    checkpoint: str = ""
    output_dir: str = str(REPO_ROOT / "outputs" / "pi05_mujoco" / "manual")
    num_worlds: int = 1
    seed: int = 20260511
    num_chunks: int = 120
    execute_chunk_dim: int = 15
    camera_height: int = 168
    camera_width: int = 224
    gpu_id: int | None = 0
    prompt: str = "throw the plastic bottles in the bin"
    save_video: bool = True
    video_fps: int = 30
    video_every_n_actions: int = 1
    vanilla_physics: bool = False
    log_grippers: bool = False
    scene: PutBottlesSimConfig = field(default_factory=PutBottlesSimConfig)


def _server_observation(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Convert ABC's CHW camera tensors to the shared HWC OpenPI contract."""
    images: dict[str, np.ndarray] = {}
    for name in CAMERA_KEYS:
        image = np.asarray(obs["images"][name])
        if image.ndim != 3:
            raise ValueError(f"camera {name!r} must be rank 3, got {image.shape}")
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3:
            raise ValueError(f"camera {name!r} must have 3 RGB channels, got {image.shape}")
        images[name] = np.ascontiguousarray(image, dtype=np.uint8)

    state = np.asarray(obs["state"], dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"expected {STATE_DIM}-D YAM state, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("simulator state contains non-finite values")
    return {"images": images, "state": state, "prompt": prompt}


def _parse_actions(response: dict[str, Any], execute_chunk_dim: int) -> np.ndarray:
    if "actions" not in response:
        raise KeyError(f"policy response has no 'actions' key: {sorted(response)}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != STATE_DIM:
        raise ValueError(f"expected policy actions shaped (H, {STATE_DIM}), got {actions.shape}")
    if actions.shape[0] < execute_chunk_dim:
        raise ValueError(
            f"policy returned only {actions.shape[0]} actions, fewer than "
            f"execute_chunk_dim={execute_chunk_dim}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("policy returned non-finite actions")
    return actions


def run_eval(config: OpenPIMujocoEvalConfig) -> dict[str, Any]:
    if config.num_worlds <= 0 or config.num_chunks <= 0:
        raise ValueError("num_worlds and num_chunks must be positive")
    if config.execute_chunk_dim <= 0:
        raise ValueError("execute_chunk_dim must be positive")
    if config.video_every_n_actions <= 0:
        raise ValueError("video_every_n_actions must be positive")

    client = websocket_client_policy.WebsocketClientPolicy(
        host=config.server_host,
        port=config.server_port,
    )
    metadata = client.get_server_metadata()
    expected_state_dim = metadata.get("state_dim")
    if expected_state_dim is not None and int(expected_state_dim) != STATE_DIM:
        raise ValueError(
            f"policy server reports state_dim={expected_state_dim}; this scene requires {STATE_DIM}"
        )

    env = PutBottlesEnv(
        height=config.camera_height,
        width=config.camera_width,
        camera_keys=CAMERA_KEYS,
        prompt=config.prompt,
        scene=config.scene,
        gpu_id=config.gpu_id,
    )
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o750)
    worlds: list[dict[str, Any]] = []

    try:
        for world_index in range(config.num_worlds):
            import imageio.v2 as imageio

            world_started = time.perf_counter()
            world_seed = config.seed + world_index
            obs = env.reset(seed=world_seed)
            final_eval = env.evaluate_vanilla() if config.vanilla_physics else env.evaluate()
            video = None
            video_path: Path | None = None
            steps = 0
            chunk_metrics: list[dict[str, Any]] = []
            if config.save_video:
                video_path = out_dir / f"world_{world_index:03d}.mp4"
                video = imageio.get_writer(
                    str(video_path), fps=config.video_fps, macro_block_size=1
                )
                video.append_data(video_frame(obs["images"], CAMERA_KEYS))

            grippers = None
            gripper_trace_path = out_dir / f"world_{world_index:03d}_grippers.csv"
            try:
                if config.log_grippers:
                    grippers = GripperDiagnostics(env, gripper_trace_path, vanilla_physics=config.vanilla_physics)
                obs_fn = env.obs_vanilla_state if config.vanilla_physics else env.obs
                eval_fn = env.evaluate_vanilla if config.vanilla_physics else env.evaluate
                step_fn = env.step_one_vanilla if config.vanilla_physics else env.step_one
                render_fn = (
                    env.render_cameras_vanilla_state
                    if config.vanilla_physics
                    else env.render_cameras
                )

                for chunk in range(config.num_chunks):
                    infer_started = time.perf_counter()
                    response = client.infer(_server_observation(obs, config.prompt))
                    infer_s = time.perf_counter() - infer_started
                    actions = _parse_actions(response, config.execute_chunk_dim)

                    if grippers is not None:
                        grippers.start_chunk(chunk)
                    step_started = time.perf_counter()
                    for action_index, action in enumerate(actions[: config.execute_chunk_dim]):
                        step_fn(action)
                        final_eval = eval_fn()
                        steps += 1
                        if grippers is not None:
                            grippers.record(action, action_index=action_index, step=steps)
                        if video is not None and steps % config.video_every_n_actions == 0:
                            video.append_data(video_frame(render_fn(), CAMERA_KEYS))
                        if final_eval["ever_success"]:
                            break
                    steps_s = time.perf_counter() - step_started

                    metric = {
                        "chunk": chunk,
                        "infer_s": infer_s,
                        "steps_s": steps_s,
                        "bottles": int(final_eval["num_bottles_in_bin"]),
                        "max_bottles": int(final_eval["max_bottles_in_bin_so_far"]),
                        "action_min": float(actions.min()),
                        "action_max": float(actions.max()),
                        "policy_timing": response.get("policy_timing"),
                        "server_timing": response.get("server_timing"),
                    }
                    if grippers is not None:
                        metric["grippers"] = grippers.finish_chunk(world_index)
                    chunk_metrics.append(metric)
                    print(
                        f"world={world_index:03d} chunk={chunk:03d} "
                        f"infer={infer_s * 1000:.0f}ms steps={steps_s * 1000:.0f}ms "
                        f"bottles={final_eval['num_bottles_in_bin']}/"
                        f"{final_eval['num_active_bottles']} "
                        f"success={final_eval['ever_success']}",
                        flush=True,
                    )
                    if final_eval["ever_success"]:
                        break
                    if chunk + 1 < config.num_chunks:
                        obs = obs_fn()
            finally:
                if grippers is not None:
                    grippers.close()
                if video is not None:
                    video.close()
                    if video_path is not None:
                        video_path.chmod(0o640)

            world = {
                "world_index": world_index,
                "world_seed": world_seed,
                "success": bool(final_eval["ever_success"]),
                "final_success": bool(final_eval["success"]),
                "reward": float(final_eval["reward"]),
                "steps": steps,
                "wall_s": time.perf_counter() - world_started,
                "chunk_metrics": chunk_metrics,
                "randomization": env.randomization,
                "final_task_eval": final_eval,
                "video_path": str(video_path) if video_path is not None else None,
                "gripper_trace_path": str(gripper_trace_path) if config.log_grippers else None,
            }
            worlds.append(world)
            print(
                f"world={world_index:03d} done success={world['success']} "
                f"bottles={final_eval['max_bottles_in_bin_so_far']}/"
                f"{final_eval['num_active_bottles']} steps={steps}",
                flush=True,
            )
    finally:
        env.close()

    successes = np.asarray([world["success"] for world in worlds], dtype=bool)
    rewards = np.asarray([world["reward"] for world in worlds], dtype=np.float32)
    max_bottles = np.asarray(
        [world["final_task_eval"]["max_bottles_in_bin_so_far"] for world in worlds],
        dtype=np.float32,
    )
    summary = {
        "format": "openpi_abc_put_bottles_eval/v1",
        "checkpoint": config.checkpoint,
        "prompt": config.prompt,
        "config": asdict(config),
        "server_metadata": metadata,
        "success_rate": float(successes.mean()),
        "num_success": int(successes.sum()),
        "num_worlds": len(worlds),
        "mean_reward": float(rewards.mean()),
        "mean_max_bottles_in_bin": float(max_bottles.mean()),
        "worlds": worlds,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    summary_path.chmod(0o640)
    print(
        f"summary: success_rate={summary['success_rate']} "
        f"num_success={summary['num_success']}/{summary['num_worlds']} "
        f"mean_reward={summary['mean_reward']} "
        f"mean_max_bottles={summary['mean_max_bottles_in_bin']}",
        flush=True,
    )
    print(f"wrote {summary_path}", flush=True)
    return summary


def main(config: OpenPIMujocoEvalConfig) -> None:
    run_eval(config)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(OpenPIMujocoEvalConfig))
