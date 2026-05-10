#!/usr/bin/env python3
"""Convert a single PHUMA G1 ``*.npy`` to BM ``motion.npz`` (same keys as soma2bm output)."""

from __future__ import annotations

import argparse

from phuma2bm_lib import convert_phuma_npy_to_bm_npz


def parse_args():
    p = argparse.ArgumentParser(description="Convert one PHUMA dict .npy to BM npz.")
    p.add_argument("--input_npy", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument(
        "--input_fps",
        type=int,
        default=None,
        help="Override fps from file (default: use ``fps`` inside the npy dict).",
    )
    p.add_argument("--output_fps", type=int, default=50)
    p.add_argument("--num_bodies", type=int, default=30)
    p.add_argument(
        "--assume_isaac_joint_order",
        action="store_true",
        help="Skip MuJoCo -> IsaacLab joint reorder for dof_pos.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    convert_phuma_npy_to_bm_npz(
        args.input_npy,
        args.output_npz,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        num_bodies=args.num_bodies,
        apply_mujoco_to_isaac_joint_reorder=not args.assume_isaac_joint_order,
    )
    print(f"[OK] converted: {args.input_npy} -> {args.output_npz}")


if __name__ == "__main__":
    main()
