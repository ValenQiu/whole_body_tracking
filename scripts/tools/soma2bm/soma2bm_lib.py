import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

EXPECTED_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]


def clip_id_to_relative_csv_path(clip_id: str) -> str:
    if "-" not in clip_id:
        msg = f"Invalid clip_id without date prefix: {clip_id}"
        raise ValueError(msg)
    date_prefix, rest = clip_id.split("-", 1)
    motion_name = rest
    if motion_name.endswith("_smpl_params"):
        motion_name = motion_name[: -len("_smpl_params")]
    return f"{date_prefix}/{motion_name}.csv"


def build_filtered_manifest(csv_root: str, metadata_json: str) -> tuple[list[dict], list[dict]]:
    csv_root_path = Path(csv_root)
    with open(metadata_json, encoding="utf-8") as f:
        metadata = json.load(f)
    data_list = metadata.get("data_list", [])

    manifest: list[dict] = []
    missing: list[dict] = []
    seen: set[str] = set()
    for item in data_list:
        clip_id = item.get("clip_id", "")
        if not clip_id:
            continue
        try:
            relative_csv_path = clip_id_to_relative_csv_path(clip_id)
        except ValueError:
            continue
        if relative_csv_path.endswith("_M.csv"):
            continue
        if relative_csv_path in seen:
            continue
        seen.add(relative_csv_path)

        record = {
            "clip_id": clip_id,
            "relative_csv_path": relative_csv_path,
        }
        absolute_path = csv_root_path / relative_csv_path
        if absolute_path.exists():
            manifest.append(record)
        else:
            missing.append(record)
    return manifest, missing


def _resample_linear(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(dst_t, src_t, values)
    out = np.zeros((len(dst_t), values.shape[1]), dtype=np.float64)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, values[:, i])
    return out


def _resample_quat_wxyz(quat_wxyz: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if len(src_t) == 1:
        return np.repeat(quat_wxyz, len(dst_t), axis=0)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(quat_xyzw)
    slerp = Slerp(src_t, rotations)
    out_xyzw = slerp(dst_t).as_quat()
    return out_xyzw[:, [3, 0, 1, 2]]


def _compute_angular_velocity_from_wxyz(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(quat_xyzw)
    if len(quat_xyzw) < 2:
        return np.zeros((len(quat_xyzw), 3), dtype=np.float32)
    relative = rot[1:] * rot[:-1].inv()
    rotvec = relative.as_rotvec() / dt
    ang = np.zeros((len(quat_xyzw), 3), dtype=np.float64)
    ang[1:] = rotvec
    ang[0] = ang[1]
    return ang.astype(np.float32)


def convert_csv_to_bm_npz(
    input_csv: str,
    output_npz: str,
    input_fps: int = 120,
    output_fps: int = 50,
    num_bodies: int = 30,
) -> None:
    raw = np.genfromtxt(input_csv, delimiter=",", names=True, dtype=np.float64)
    if raw.dtype.names is None:
        msg = f"CSV has no header: {input_csv}"
        raise ValueError(msg)
    columns = list(raw.dtype.names)
    if columns != EXPECTED_COLUMNS:
        msg = (
            f"CSV columns mismatch.\nExpected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {columns}\nFile: {input_csv}"
        )
        raise ValueError(msg)

    T = raw.shape[0]
    if T < 2:
        msg = f"CSV must contain at least 2 frames: {input_csv}"
        raise ValueError(msg)

    root_pos_m = np.stack(
        [
            raw["root_translateX"],
            raw["root_translateY"],
            raw["root_translateZ"],
        ],
        axis=1,
    ) / 100.0
    root_euler_deg = np.stack(
        [
            raw["root_rotateX"],
            raw["root_rotateY"],
            raw["root_rotateZ"],
        ],
        axis=1,
    )
    root_quat_xyzw = Rotation.from_euler("xyz", root_euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    dof_deg = np.stack([raw[name] for name in EXPECTED_COLUMNS[7:]], axis=1)
    dof_rad = np.deg2rad(dof_deg)

    src_t = np.arange(T, dtype=np.float64) / float(input_fps)
    duration = src_t[-1]
    out_frames = max(2, int(np.floor(duration * output_fps)) + 1)
    dst_t = np.linspace(0.0, duration, out_frames)

    root_pos_out = _resample_linear(root_pos_m, src_t, dst_t).astype(np.float32)
    root_quat_out = _resample_quat_wxyz(root_quat_wxyz, src_t, dst_t).astype(np.float32)
    dof_out = _resample_linear(dof_rad, src_t, dst_t).astype(np.float32)

    dt = 1.0 / float(output_fps)
    dof_vel = np.gradient(dof_out, dt, axis=0).astype(np.float32)
    root_lin_vel = np.gradient(root_pos_out, dt, axis=0).astype(np.float32)
    root_ang_vel = _compute_angular_velocity_from_wxyz(root_quat_out, dt)

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
