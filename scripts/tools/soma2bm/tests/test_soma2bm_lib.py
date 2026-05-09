import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from soma2bm_lib import build_filtered_manifest, clip_id_to_relative_csv_path, convert_csv_to_bm_npz


def test_clip_id_to_relative_csv_path():
    clip_id = "230710-jog_ff_loop_225_R_001__A422_smpl_params"
    assert clip_id_to_relative_csv_path(clip_id) == "230710/jog_ff_loop_225_R_001__A422.csv"


def test_build_filtered_manifest_excludes_mirror_and_missing(tmp_path: Path):
    csv_root = tmp_path / "csv"
    (csv_root / "230710").mkdir(parents=True)
    (csv_root / "230710" / "walk_a.csv").write_text("dummy", encoding="utf-8")
    (csv_root / "230710" / "walk_a_M.csv").write_text("dummy", encoding="utf-8")

    metadata = {
        "data_list": [
            {"clip_id": "230710-walk_a_smpl_params"},
            {"clip_id": "230710-walk_a_M_smpl_params"},
            {"clip_id": "230710-not_exist_smpl_params"},
        ]
    }
    metadata_path = tmp_path / "meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    manifest, missing = build_filtered_manifest(str(csv_root), str(metadata_path))
    assert [item["relative_csv_path"] for item in manifest] == ["230710/walk_a.csv"]
    assert [item["relative_csv_path"] for item in missing] == ["230710/not_exist.csv"]


def test_convert_csv_to_bm_npz_generates_required_keys(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    header = (
        "Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,"
        "left_hip_pitch_joint_dof,left_hip_roll_joint_dof,left_hip_yaw_joint_dof,left_knee_joint_dof,"
        "left_ankle_pitch_joint_dof,left_ankle_roll_joint_dof,right_hip_pitch_joint_dof,right_hip_roll_joint_dof,"
        "right_hip_yaw_joint_dof,right_knee_joint_dof,right_ankle_pitch_joint_dof,right_ankle_roll_joint_dof,"
        "waist_yaw_joint_dof,waist_roll_joint_dof,waist_pitch_joint_dof,left_shoulder_pitch_joint_dof,"
        "left_shoulder_roll_joint_dof,left_shoulder_yaw_joint_dof,left_elbow_joint_dof,left_wrist_roll_joint_dof,"
        "left_wrist_pitch_joint_dof,left_wrist_yaw_joint_dof,right_shoulder_pitch_joint_dof,right_shoulder_roll_joint_dof,"
        "right_shoulder_yaw_joint_dof,right_elbow_joint_dof,right_wrist_roll_joint_dof,right_wrist_pitch_joint_dof,"
        "right_wrist_yaw_joint_dof"
    )
    rows = [
        "0,0,0,100,0,0,0," + ",".join(["0"] * 29),
        "1,1,0,100,0,0,10," + ",".join(["1"] * 29),
        "2,2,0,100,0,0,20," + ",".join(["2"] * 29),
    ]
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    out_npz = tmp_path / "out.npz"
    convert_csv_to_bm_npz(
        input_csv=str(csv_path),
        output_npz=str(out_npz),
        input_fps=120,
        output_fps=50,
        num_bodies=30,
    )

    data = np.load(out_npz)
    required_keys = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    assert required_keys.issubset(set(data.files))
    assert int(data["fps"][0]) == 50
    assert data["joint_pos"].shape[1] == 29
    assert data["body_pos_w"].shape[1:] == (30, 3)
    assert data["body_quat_w"].shape[1:] == (30, 4)
