"""Mark crashed (or other non-finished) W&B runs as finished.

The only reliable way to flip a run's state from 'crashed' to 'finished'
via the public SDK is to *resume* the run and immediately call wandb.finish().
This script does exactly that for every run in a project whose state matches
the given filter (default: crashed).

Usage (run from whole_body_tracking/ root):

    /isaac-sim/python.sh scripts/tools/auto_train/mark_runs_finished.py \\
        --project "entity/project_name" \\
        [--state crashed]              \\  # which states to fix (default: crashed)
        [--filter "walk.*"]            \\  # optional regex on run name
        [--ids d62rglx4 6awtzzua ...]  \\  # fix specific run IDs only
        [--dry_run]                        # print without changing anything

Example — fix all crashed runs in g1_lafan1_motion_tracking:

    /isaac-sim/python.sh scripts/tools/auto_train/mark_runs_finished.py \\
        --project "liuming-valen-qiu-the-hong-kong-polytechnic-university/g1_lafan1_motion_tracking"

"""

import argparse
import re
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume crashed W&B runs and call wandb.finish() to mark them as finished."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help='W&B entity/project path, e.g. "myorg/g1_lafan1_motion_tracking".',
    )
    parser.add_argument(
        "--state",
        type=str,
        nargs="+",
        default=["crashed"],
        choices=["crashed", "failed", "running"],
        help="State(s) to fix. Default: crashed.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional Python regex — only process runs whose name matches.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        default=None,
        help="Fix only runs with these specific run IDs (overrides --state / --filter).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print which runs would be fixed without actually resuming them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import wandb

    api = wandb.Api()
    entity, project_name = args.project.split("/", 1)

    # ---- collect runs to fix ------------------------------------------------
    if args.ids:
        runs_to_fix = []
        for run_id in args.ids:
            try:
                r = api.run(f"{args.project}/{run_id}")
                runs_to_fix.append(r)
            except Exception as e:
                print(f"[WARN] Could not fetch run {run_id}: {e}")
    else:
        filter_re = re.compile(args.filter) if args.filter else None
        runs_to_fix = []
        for state in args.state:
            for r in api.runs(path=args.project, filters={"state": state}):
                if filter_re and not filter_re.search(r.name):
                    continue
                runs_to_fix.append(r)

    if not runs_to_fix:
        print("[INFO] No matching runs found — nothing to do.")
        return

    print(f"[INFO] Found {len(runs_to_fix)} run(s) to mark as finished:")
    for r in runs_to_fix:
        print(f"  [{r.state:8s}] {r.name}  (id={r.id})  {r.url}")

    if args.dry_run:
        print("[DRY RUN] No changes made.")
        return

    # ---- resume each run and call finish() ----------------------------------
    ok = 0
    fail = 0
    for r in runs_to_fix:
        print(f"\n[....] Resuming {r.name} (id={r.id}) ...")
        try:
            resumed = wandb.init(
                entity=entity,
                project=project_name,
                id=r.id,
                resume="must",   # raises if run_id doesn't exist
            )
            wandb.finish(exit_code=0)
            print(f"[OK  ] {r.name} → finished")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {r.name}: {e}")
            fail += 1

    print(f"\n[SUMMARY] total={len(runs_to_fix)}  ok={ok}  failed={fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
