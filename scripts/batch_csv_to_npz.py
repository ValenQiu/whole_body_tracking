"""Batch convert CSV motions to NPZ using csv_to_npz.py.

Example:
    python scripts/batch_csv_to_npz.py \
        --input_dir /host_datasets/LAFAN1 \
        --output_dir /workspace/isaaclab/data_storage/lafan1_npz \
        --input_fps 30 --output_fps 50 --headless
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert motion CSV files into NPZ files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .csv motion files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated .npz files.")
    parser.add_argument("--input_fps", type=int, default=30, help="Input motion FPS.")
    parser.add_argument("--output_fps", type=int, default=50, help="Output motion FPS.")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional frame range passed to csv_to_npz.py.",
    )
    parser.add_argument("--pattern", type=str, default="*.csv", help="Filename glob pattern. Default: *.csv")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input directory.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already exist in output.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands without executing.")
    parser.add_argument("--device", type=str, default=None, help="Optional simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--headless", action="store_true", default=False, help="Run Isaac Sim in headless mode.")
    parser.add_argument(
        "--upload_wandb",
        action="store_true",
        default=False,
        help="If set, upload converted files to WandB registry (default: disabled).",
    )
    return parser.parse_args()


def collect_csv_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted([p for p in input_dir.rglob(pattern) if p.is_file()])
    else:
        files = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return files


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Invalid input directory: {input_dir}")
        return 2

    csv_files = collect_csv_files(input_dir, args.pattern, args.recursive)
    if not csv_files:
        print(f"[ERROR] No CSV files found in: {input_dir} (pattern={args.pattern}, recursive={args.recursive})")
        return 3

    output_dir.mkdir(parents=True, exist_ok=True)

    # In container/CI environments without a display server, default to headless mode.
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_headless = args.headless or not has_display
    if use_headless and not args.headless:
        print("[INFO] No display server detected. Enabling --headless automatically.")

    script_path = Path(__file__).resolve().parent / "csv_to_npz.py"
    success, failed, skipped = 0, 0, 0

    print(f"[INFO] Found {len(csv_files)} CSV files.")
    for idx, csv_file in enumerate(csv_files, start=1):
        rel = csv_file.relative_to(input_dir)
        output_file = (output_dir / rel).with_suffix(".npz")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and output_file.exists():
            skipped += 1
            print(f"[{idx}/{len(csv_files)}] [SKIP] {csv_file} -> {output_file}")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--input_file",
            str(csv_file),
            "--input_fps",
            str(args.input_fps),
            "--output_name",
            csv_file.stem,
            "--output_fps",
            str(args.output_fps),
            "--output_file",
            str(output_file),
        ]

        if use_headless:
            cmd.append("--headless")
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.frame_range is not None:
            cmd += ["--frame_range", str(args.frame_range[0]), str(args.frame_range[1])]
        if not args.upload_wandb:
            cmd.append("--disable_wandb")

        print(f"[{idx}/{len(csv_files)}] [RUN] {csv_file} -> {output_file}")
        if args.dry_run:
            print(" ".join(cmd))
            success += 1
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode == 0 and output_file.exists():
            success += 1
        else:
            failed += 1
            print(f"[ERROR] Failed on: {csv_file}")

    print("\n[SUMMARY]")
    print(f"  success: {success}")
    print(f"  skipped: {skipped}")
    print(f"  failed : {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
