"""Fetch all motion artifact collections from a W&B registry and save to a JSON file.

The generated JSON is consumed by auto_train.sh to drive sequential policy training.

Usage (run from whole_body_tracking/ root):

    python scripts/tools/auto_train/fetch_registry.py \\
        --registry_name "liuming-valen-qiu-...-org/wandb-registry-Motions" \\
        [--filter "walk.*"] \\
        [--output scripts/tools/auto_train/motions.json] \\
        [--dry_run]

--registry_name accepts:
    entity/wandb-registry-Type                      (preferred — project path only)
    entity/wandb-registry-Type/collection           (collection part is ignored)
    entity/wandb-registry-Type/collection:version   (collection+version ignored)
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone


# Accepts the project path alone, or with an optional /collection[:version] suffix.
_REGISTRY_NAME_RE = re.compile(
    r"^(?P<entity>[^/]+)/(?P<project>wandb-registry-(?P<registry_type>[^/]+?))(?:/[^/].*)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate W&B registry motion artifacts and write a JSON schedule file."
    )
    parser.add_argument(
        "--registry_name",
        type=str,
        required=True,
        help=(
            'W&B registry project path, e.g. "myorg-org/wandb-registry-Motions". '
            "A trailing /collection[:version] is accepted but ignored — "
            "all collections in the registry are enumerated regardless."
        ),
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional Python regex pattern to keep only matching collection names.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/tools/auto_train/motions.json",
        help="Output JSON file path. Default: scripts/tools/auto_train/motions.json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the discovered collections without writing the JSON file.",
    )
    return parser.parse_args()


def parse_registry_name(registry_name: str) -> tuple[str, str, str]:
    """Parse a registry_name into (entity, project_path, registry_type).

    Accepts paths with or without a collection/version suffix:
        entity/wandb-registry-Type
        entity/wandb-registry-Type/collection
        entity/wandb-registry-Type/collection:version
    """
    m = _REGISTRY_NAME_RE.match(registry_name.strip())
    if not m:
        print(
            f"[ERROR] Cannot parse registry_name: {registry_name!r}\n"
            "        Expected format: entity/wandb-registry-Type[/collection[:version]]"
        )
        sys.exit(1)
    entity = m.group("entity")
    project_path = f"{entity}/{m.group('project')}"
    registry_type = m.group("registry_type")
    return entity, project_path, registry_type


def fetch_collections(api, project_path: str, registry_type: str) -> list:
    """Return ArtifactCollection objects from the registry project.

    The registry display name (e.g. "Motions" from wandb-registry-Motions) may differ
    in case from the actual artifact type field set at upload time (e.g. "motions").
    Strategy:
      1. Try the extracted registry_type as-is, then its lowercase variant.
      2. If both miss, enumerate every artifact type in the project and merge all collections.
    """
    for type_name in dict.fromkeys([registry_type, registry_type.lower()]):
        try:
            at = api.artifact_type(type_name, project_path)
            cols = list(at.collections())
            if cols:
                print(f"[INFO] Matched artifact type: {type_name!r}")
                return cols
        except Exception:
            pass

    # Fallback: enumerate all artifact types in the project.
    print(f"[WARN] Could not match type {registry_type!r} directly — enumerating all artifact types in project ...")
    try:
        all_collections = []
        for at in api.artifact_types(project_path):
            cols = list(at.collections())
            if cols:
                print(f"[INFO]   type {at.name!r}: {len(cols)} collection(s)")
                all_collections.extend(cols)
        if all_collections:
            return all_collections
    except Exception as err:
        print(f"[ERROR] artifact_types() enumeration failed: {err}")

    print(f"[ERROR] No artifact collections found in: {project_path}")
    sys.exit(1)


def sanitize_name(name: str) -> str:
    """Replace characters invalid in filesystem paths / wandb run names with underscores."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def main() -> None:
    args = parse_args()

    entity, project_path, registry_type = parse_registry_name(args.registry_name)
    print(f"[INFO] entity       : {entity}")
    print(f"[INFO] project_path : {project_path}")
    print(f"[INFO] registry_type: {registry_type}")

    import wandb

    api = wandb.Api()
    print(f"[INFO] Querying W&B registry: {project_path}")

    collections = fetch_collections(api, project_path, registry_type)
    filter_re = re.compile(args.filter) if args.filter else None

    motions: list[dict] = []
    skipped = 0
    for col in collections:
        name: str = col.name
        if filter_re and not filter_re.search(name):
            skipped += 1
            continue
        full_registry_name = f"{project_path}/{name}:latest"
        motions.append(
            {
                "name": name,
                "safe_name": sanitize_name(name),
                "registry_name": full_registry_name,
            }
        )

    print(f"[INFO] Found {len(motions)} collection(s)" + (f" ({skipped} filtered out)." if skipped else "."))
    for m in motions:
        print(f"  {m['registry_name']}")

    if args.dry_run:
        print("[DRY RUN] JSON not written.")
        return

    if not motions:
        print("[WARN] Nothing to write — JSON file will not be created.")
        return

    payload = {
        "entity": entity,
        "project_path": project_path,
        "registry_type": registry_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(motions),
        "motions": motions,
    }

    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"[INFO] Saved {len(motions)} entries to: {args.output}")


if __name__ == "__main__":
    main()
