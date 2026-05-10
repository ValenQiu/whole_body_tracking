#!/usr/bin/env python3
"""Build ``filtered_phuma_npy_manifest.json`` (same role as soma2bm ``build_filtered_csv_manifest``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phuma2bm_lib import build_phuma_manifest


def parse_args():
    p = argparse.ArgumentParser(description="Scan PHUMA data/g1 tree and write npy manifest for batch convert.")
    p.add_argument(
        "--phuma_g1_root",
        required=True,
        help="Path to PHUMA ``data/g1`` (contains subfolders like humanml/, aist/, …).",
    )
    p.add_argument(
        "--split_file",
        default=None,
        help="Optional text file: one relative path per line (from g1 root), e.g. ``humanml/000000_chunk_0000.npy``.",
    )
    p.add_argument("--output_dir", required=True, help="Directory for manifest json + stats")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records, missing_split = build_phuma_manifest(args.phuma_g1_root, args.split_file)

    manifest_path = out / "filtered_phuma_npy_manifest.json"
    missing_path = out / "missing_split_lines.json"
    stats_path = out / "manifest_stats.json"

    manifest_payload = {
        "phuma_g1_root": str(Path(args.phuma_g1_root).resolve()),
        "split_file": str(Path(args.split_file).resolve()) if args.split_file else None,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_path.write_text(
        json.dumps({"missing_from_disk_or_split": missing_split}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(
            {
                "total_records": len(records),
                "split_lines_not_found_under_root": len(missing_split),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] stats   : {stats_path}")
    if args.split_file:
        print(f"[OK] split misses (paths in split but no file): {len(missing_split)} -> {missing_path}")
    print(f"[INFO] records={len(records)}")


if __name__ == "__main__":
    main()
