"""PHUMA G1 dict .npy -> whole_body_tracking BM motion.npz (same schema as soma2bm)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_REQUIRED_KEYS = ("root_trans", "root_ori", "dof_pos", "fps")


def _load_soma2bm_lib():
    soma_path = Path(__file__).resolve().parent.parent / "soma2bm" / "soma2bm_lib.py"
    spec = importlib.util.spec_from_file_location("_soma2bm_lib_phuma_reuse", soma_path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load soma2bm_lib from {soma_path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_soma = _load_soma2bm_lib()
_permute_mujoco_dof_to_isaac = _soma._permute_mujoco_csv_dof_rows_to_isaac_lab
_resample_linear = _soma._resample_linear
_resample_quat_wxyz = _soma._resample_quat_wxyz
_compute_angular_velocity_from_wxyz = _soma._compute_angular_velocity_from_wxyz


def load_phuma_g1_dict(path: str) -> tuple[dict, int]:
    """Load PHUMA/HF-format G1 motion dict and return (dict, fps)."""
    raw = np.load(path, allow_pickle=True)
    if raw.shape == ():
        data = raw.item()
    else:
        data = raw
    if not isinstance(data, dict):
        msg = f"Expected object array or dict in {path}, got {type(data)}"
        raise ValueError(msg)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        msg = f"Missing keys {missing} in {path}"
        raise ValueError(msg)
    fps_val = data["fps"]
    fps = int(fps_val) if not isinstance(fps_val, np.ndarray) else int(fps_val.flat[0])
    if fps < 1:
        msg = f"Invalid fps={fps} in {path}"
        raise ValueError(msg)
    return data, fps


def convert_phuma_npy_to_bm_npz(
    input_npy: str,
    output_npz: str,
    input_fps: int | None = None,
    output_fps: int = 50,
    num_bodies: int = 30,
    *,
    apply_mujoco_to_isaac_joint_reorder: bool = True,
) -> None:
    """Convert one PHUMA ``*.npy`` (dict with root_trans, root_ori xyzw, dof_pos rad, fps) to BM npz."""
    data, file_fps = load_phuma_g1_dict(input_npy)
    src_fps = int(input_fps) if input_fps is not None else file_fps

    root_trans = np.asarray(data["root_trans"], dtype=np.float64)
    root_ori_xyzw = np.asarray(data["root_ori"], dtype=np.float64)
    dof_mj = np.asarray(data["dof_pos"], dtype=np.float64)

    if root_trans.ndim != 2 or root_trans.shape[1] != 3:
        msg = f"root_trans must be (T,3), got {root_trans.shape} in {input_npy}"
        raise ValueError(msg)
    if root_ori_xyzw.shape != (root_trans.shape[0], 4):
        msg = f"root_ori must be (T,4) xyzw, got {root_ori_xyzw.shape} in {input_npy}"
        raise ValueError(msg)
    if dof_mj.shape != (root_trans.shape[0], 29):
        msg = f"dof_pos must be (T,29) for G1, got {dof_mj.shape} in {input_npy}"
        raise ValueError(msg)

    T = root_trans.shape[0]
    if T < 2:
        root_trans = np.vstack([root_trans, root_trans[-1:]])
        root_ori_xyzw = np.vstack([root_ori_xyzw, root_ori_xyzw[-1:]])
        dof_mj = np.vstack([dof_mj, dof_mj[-1:]])
        T = root_trans.shape[0]

    root_pos_m = root_trans
    quat_wxyz = root_ori_xyzw[:, [3, 0, 1, 2]]

    n = T
    src_t = np.arange(n, dtype=np.float64) / float(src_fps)
    duration = float(src_t[-1])
    out_frames = max(2, int(np.floor(duration * output_fps)) + 1)
    dst_t = np.linspace(0.0, duration, out_frames)

    root_pos_out = _resample_linear(root_pos_m, src_t, dst_t).astype(np.float32)
    root_quat_out = _resample_quat_wxyz(quat_wxyz, src_t, dst_t).astype(np.float32)
    dof_out = _resample_linear(dof_mj, src_t, dst_t).astype(np.float32)

    dt = 1.0 / float(output_fps)
    dof_vel = np.gradient(dof_out, dt, axis=0).astype(np.float32)
    if apply_mujoco_to_isaac_joint_reorder:
        dof_out = _permute_mujoco_dof_to_isaac(dof_out)
        dof_vel = _permute_mujoco_dof_to_isaac(dof_vel)
    root_lin_vel = np.gradient(root_pos_out, dt, axis=0).astype(np.float32)
    root_ang_vel = _compute_angular_velocity_from_wxyz(root_quat_out, dt)

    out_frames = root_pos_out.shape[0]
    body_pos_w = np.zeros((out_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((out_frames, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((out_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((out_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0

    body_pos_w[:, 0, :] = root_pos_out
    body_quat_w[:, 0, :] = root_quat_out
    body_lin_vel_w[:, 0, :] = root_lin_vel
    body_ang_vel_w[:, 0, :] = root_ang_vel

    out = {
        "fps": np.array([output_fps], dtype=np.int32),
        "joint_pos": dof_out,
        "joint_vel": dof_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **out)


def build_phuma_manifest(phuma_g1_root: str, split_file: str | None = None) -> tuple[list[dict], list[str]]:
    """List ``*.npy`` under ``phuma_g1_root`` (optionally restricted to paths in ``split_file``)."""
    root = Path(phuma_g1_root).resolve()
    if not root.is_dir():
        msg = f"Not a directory: {root}"
        raise FileNotFoundError(msg)

    allowed: set[str] | None = None
    if split_file:
        allowed = set()
        for line in Path(split_file).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rel = str(Path(s)).replace("\\", "/")
            while rel.startswith("./"):
                rel = rel[2:]
            if rel.startswith("g1/"):
                rel = rel[len("g1/") :]
            allowed.add(rel)
        if not allowed:
            allowed = None

    missing_split: list[str] = []
    records: list[dict] = []
    for npy in sorted(root.rglob("*.npy")):
        rel = str(npy.relative_to(root)).replace("\\", "/")
        if allowed is not None and rel not in allowed:
            continue
        records.append({"relative_npy_path": rel})
        if allowed is not None:
            allowed.discard(rel)

    if allowed:
        missing_split = sorted(allowed)

    return records, missing_split
