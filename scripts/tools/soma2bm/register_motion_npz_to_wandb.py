#!/usr/bin/env python3
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Register converted motion npz to W&B artifact.")
    parser.add_argument("--project", default="csv_to_npz", help="W&B project")
    parser.add_argument("--artifact_name", required=True, help="Artifact collection name")
    parser.add_argument("--artifact_type", default="motions", help="Artifact type")
    parser.add_argument("--alias", action="append", default=["latest"], help="Artifact alias (repeatable)")
    parser.add_argument(
        "--input_npz",
        default=None,
        help="Single npz path to upload (copied as motion.npz in artifact)",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing converted npz files (for batch artifact)",
    )
    parser.add_argument(
        "--metadata_manifest",
        default=None,
        help="Optional manifest json to include in artifact",
    )
    return parser.parse_args()


def _add_single_npz(artifact: wandb.Artifact, input_npz: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dst = Path(tmp_dir) / "motion.npz"
        shutil.copy2(input_npz, dst)
        artifact.add_file(str(dst), name="motion.npz")


def _add_batch_dir(artifact: wandb.Artifact, input_dir: str):
    artifact.add_dir(input_dir, name="motions")


def _add_manifest(artifact: wandb.Artifact, metadata_manifest: str):
    manifest = Path(metadata_manifest)
    if not manifest.exists():
        return
    with open(manifest, encoding="utf-8") as f:
        payload = json.load(f)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "metadata_manifest.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact.add_file(str(out), name="metadata_manifest.json")


def main():
    args = parse_args()
    if not args.input_npz and not args.input_dir:
        raise ValueError("One of --input_npz or --input_dir is required.")
    if args.input_npz and args.input_dir:
        raise ValueError("Only one of --input_npz or --input_dir can be used.")

    run = wandb.init(project=args.project, name=f"register-{args.artifact_name}")
    artifact = wandb.Artifact(name=args.artifact_name, type=args.artifact_type)
    if args.input_npz:
        _add_single_npz(artifact, args.input_npz)
    else:
        _add_batch_dir(artifact, args.input_dir)
    if args.metadata_manifest:
        _add_manifest(artifact, args.metadata_manifest)
    run.log_artifact(artifact, aliases=args.alias)
    run.finish()
    print(f"[OK] logged artifact: {args.artifact_type}/{args.artifact_name} aliases={args.alias}")


if __name__ == "__main__":
    main()
