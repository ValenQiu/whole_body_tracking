#!/usr/bin/env python3
import argparse

from soma2bm_lib import convert_csv_to_bm_npz


def parse_args():
    parser = argparse.ArgumentParser(description="Convert one SOMA/Bones G1 CSV to BM npz format.")
    parser.add_argument("--input_csv", required=True, help="Input csv path")
    parser.add_argument("--output_npz", required=True, help="Output npz path")
    parser.add_argument("--input_fps", type=int, default=120, help="Input csv fps")
    parser.add_argument("--output_fps", type=int, default=50, help="Output npz fps")
    parser.add_argument("--num_bodies", type=int, default=30, help="Body count for body_* arrays")
    parser.add_argument(
        "--assume_isaac_joint_order",
        action="store_true",
        help=(
            "Skip MuJoCo/CSV column -> IsaacLab joint index reorder. "
            "Use only if joint columns already match robot.data.joint_pos order."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_csv_to_bm_npz(
        input_csv=args.input_csv,
        output_npz=args.output_npz,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        num_bodies=args.num_bodies,
        apply_mujoco_csv_to_isaac_joint_reorder=not args.assume_isaac_joint_order,
    )
    print(f"[OK] converted: {args.input_csv} -> {args.output_npz}")


if __name__ == "__main__":
    main()
