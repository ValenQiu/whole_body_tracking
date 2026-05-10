#!/usr/bin/env python3
"""Batch PHUMA npy -> BM npz using ``filtered_phuma_npy_manifest.json``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phuma2bm_lib import convert_phuma_npy_to_bm_npz


def parse_args():
    p = argparse.ArgumentParser(description="Batch convert PHUMA npy to BM npz via manifest.")
    p.add_argument("--phuma_g1_root", required=True)
    p.add_argument("--metadata_manifest", required=True, help="``filtered_phuma_npy_manifest.json``")
    p.add_argument("--output_root", required=True)
    p.add_argument("--input_fps", type=int, default=None, help="Override per-file fps for all clips.")
    p.add_argument("--output_fps", type=int, default=50)
    p.add_argument("--num_bodies", type=int, default=30)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--assume_isaac_joint_order",
        action="store_true",
        help="Skip MuJoCo -> IsaacLab joint reorder.",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N records (debug).")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.phuma_g1_root).resolve()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.metadata_manifest).read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])

    converted, failed, skipped = [], [], []
    for i, item in enumerate(records):
        if args.limit is not None and i >= args.limit:
            break
        rel = item["relative_npy_path"]
        src = root / rel
        dst = out_root / Path(rel).with_suffix(".npz")
        if args.skip_existing and dst.exists():
            skipped.append({"relative_npy_path": rel, "output_npz": str(dst)})
            continue
        try:
            convert_phuma_npy_to_bm_npz(
                str(src),
                str(dst),
                input_fps=args.input_fps,
                output_fps=args.output_fps,
                num_bodies=args.num_bodies,
                apply_mujoco_to_isaac_joint_reorder=not args.assume_isaac_joint_order,
            )
            converted.append({"relative_npy_path": rel, "output_npz": str(dst)})
        except Exception as e:  # noqa: BLE001
            failed.append({"relative_npy_path": rel, "error": str(e)})

    report = {
        "phuma_g1_root": str(root),
        "output_root": str(out_root.resolve()),
        "metadata_manifest": str(Path(args.metadata_manifest).resolve()),
        "total_records": len(records),
        "converted_count": len(converted),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "converted": converted,
        "failed": failed,
        "skipped": skipped,
    }
    report_path = out_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report: {report_path}")
    print(f"[INFO] converted={len(converted)} failed={len(failed)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
