"""Fetch all runs from a W&B project and save to a JSON file.

Usage (run from whole_body_tracking/ root):

    python scripts/tools/auto_train/fetch_runs.py \\
        --project "myorg/g1_lafan1_motion_tracking" \\
        [--filter "walk.*"] \\
        [--state finished] \\
        [--output scripts/tools/auto_train/logs/runs.json] \\
        [--dry_run]

--project accepts the standard W&B entity/project path.
Output is saved to scripts/tools/auto_train/logs/ (gitignored).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch all W&B runs from a project and save to JSON."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help='W&B entity/project path, e.g. "myorg/g1_lafan1_motion_tracking".',
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional Python regex to keep only runs whose name matches.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        choices=["running", "finished", "crashed", "failed"],
        help="Only include runs with this state. Omit to include all states.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "scripts/tools/auto_train/logs/<project>.json"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print run list without writing the JSON file.",
    )
    return parser.parse_args()


def _safe(val):
    """Convert a wandb summary/config value to a JSON-serialisable type."""
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    return str(val)


def main() -> None:
    args = parse_args()

    import wandb

    api = wandb.Api()

    filters: dict = {}
    if args.state:
        filters["state"] = args.state

    print(f"[INFO] Querying W&B project: {args.project}")
    if args.state:
        print(f"[INFO] State filter: {args.state}")

    try:
        runs = api.runs(path=args.project, filters=filters)
    except Exception as e:
        print(f"[ERROR] Failed to fetch runs from '{args.project}': {e}")
        sys.exit(1)

    filter_re = re.compile(args.filter) if args.filter else None
    records = []
    skipped = 0

    for run in runs:
        if filter_re and not filter_re.search(run.name):
            skipped += 1
            continue

        records.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url
            }
        )

    print(
        f"[INFO] Found {len(records)} run(s)"
        + (f" ({skipped} filtered out)." if skipped else ".")
    )
    for r in records:
        print(f"  [{r['state']:8s}] {r['name']}  ({r['id']})  {r['url']}")

    if args.dry_run:
        print("[DRY RUN] JSON not written.")
        return

    if not records:
        print("[WARN] No runs found — JSON file will not be created.")
        return

    if args.output:
        output_path = args.output
    else:
        project_part = args.project.split("/")[-1]
        safe_project = re.sub(r"[^a-zA-Z0-9_\-]", "_", project_part)
        output_path = f"scripts/tools/auto_train/logs/{safe_project}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "project": args.project,
        "state_filter": args.state,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total": len(records),
        "runs": records,
    }

    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"[INFO] Saved {len(records)} run(s) to: {output_path}")


if __name__ == "__main__":
    main()
