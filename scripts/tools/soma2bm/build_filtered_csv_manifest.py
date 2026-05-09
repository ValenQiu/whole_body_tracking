#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from soma2bm_lib import build_filtered_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build filtered relative CSV manifest from pre-filtered bones-seed metadata."
    )
    parser.add_argument("--csv_root", required=True, help="CSV root directory, e.g. .../g1_bones-seed/csv")
    parser.add_argument("--metadata_json", required=True, help="Pre-filtered metadata json path")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for filtered_csv_manifest.json and reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, missing = build_filtered_manifest(args.csv_root, args.metadata_json)
    manifest_path = output_dir / "filtered_csv_manifest.json"
    missing_path = output_dir / "missing_in_csv_root.json"
    stats_path = output_dir / "manifest_stats.json"

    manifest_payload = {
        "csv_root": str(Path(args.csv_root).resolve()),
        "metadata_json": str(Path(args.metadata_json).resolve()),
        "records": manifest,
    }
    missing_payload = {
        "csv_root": str(Path(args.csv_root).resolve()),
        "metadata_json": str(Path(args.metadata_json).resolve()),
        "records": missing,
    }
    stats_payload = {
        "total_manifest_records": len(manifest),
        "total_missing_records": len(missing),
        "total_input_records": len(manifest) + len(missing),
    }

    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_path.write_text(json.dumps(missing_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] missing : {missing_path}")
    print(f"[OK] stats   : {stats_path}")
    print(f"[INFO] kept={len(manifest)} missing={len(missing)}")


if __name__ == "__main__":
    main()
