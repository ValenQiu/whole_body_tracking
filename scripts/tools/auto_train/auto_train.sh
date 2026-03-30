#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# auto_train.sh  –  Sequentially train one policy per motion in a registry JSON.
#
# Usage (from whole_body_tracking/ root):
#   bash scripts/tools/auto_train/auto_train.sh --json <path_to_motions.json> [OPTIONS]
#
# Required:
#   --json PATH        Path to the motions JSON produced by fetch_registry.py.
#
# Optional training overrides:
#   --task  NAME       Isaac Lab task ID.            (default: Tracking-Flat-G1-v0)
#   --project NAME     W&B log project name.         (default: g1_motion_tracking)
#   --max_iter N       Override max training iters.  (default: 30000)
#   --num_envs N       Override parallel env count.  (default: from config)
#
# Flags:
#   --dry_run          Print commands without executing.
#
# State files  (written alongside the JSON):
#   completed.log      One motion name per line — skip on re-run.
#   failed.log         "<name>\t<ISO timestamp>\t<exit_code>" per failure.
# ---------------------------------------------------------------------------

set -uo pipefail   # intentionally no -e; failure isolation is handled per-job

# ---- defaults ---------------------------------------------------------------
TASK="Tracking-Flat-G1-v0"
LOG_PROJECT_NAME="g1_motion_tracking"
MAX_ITERATIONS="30000"
NUM_ENVS=""
MOTIONS_JSON=""
DRY_RUN=false

PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
TRAIN_SCRIPT="scripts/rsl_rl/train.py"

# ---- argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)       MOTIONS_JSON="$2"; shift 2 ;;
        --task)       TASK="$2";         shift 2 ;;
        --project)    LOG_PROJECT_NAME="$2"; shift 2 ;;
        --max_iter)   MAX_ITERATIONS="$2"; shift 2 ;;
        --num_envs)   NUM_ENVS="$2";     shift 2 ;;
        --dry_run)    DRY_RUN=true;      shift ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ---- sanity checks ----------------------------------------------------------
if [[ -z "$MOTIONS_JSON" ]]; then
    echo "[ERROR] --json <path> is required."
    exit 1
fi

if [[ ! -f "$MOTIONS_JSON" ]]; then
    echo "[ERROR] JSON file not found: $MOTIONS_JSON"
    exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[ERROR] Train script not found: $TRAIN_SCRIPT"
    echo "        Run this script from the whole_body_tracking/ root directory."
    exit 1
fi

if [[ ! "$DRY_RUN" == true && ! -f "$PYTHON" ]]; then
    echo "[ERROR] Python interpreter not found: $PYTHON"
    exit 1
fi

# ---- state files (alongside the JSON) --------------------------------------
STATE_DIR="$(cd "$(dirname "$MOTIONS_JSON")" && pwd)"
COMPLETED_LOG="$STATE_DIR/completed.log"
FAILED_LOG="$STATE_DIR/failed.log"
touch "$COMPLETED_LOG" "$FAILED_LOG"

# ---- parse JSON → pipe-delimited entries using the same Python --------------
ENTRIES_FILE="$(mktemp)"
"$PYTHON" - "$MOTIONS_JSON" > "$ENTRIES_FILE" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
for m in data["motions"]:
    # Format: name|safe_name|registry_name  (pipe is safe — names never contain it)
    print(f"{m['name']}|{m['safe_name']}|{m['registry_name']}")
PYEOF

mapfile -t ENTRIES < "$ENTRIES_FILE"
rm -f "$ENTRIES_FILE"

TOTAL=${#ENTRIES[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "[ERROR] No motions parsed from JSON: $MOTIONS_JSON"
    exit 1
fi

# ---- header -----------------------------------------------------------------
echo "============================================================"
echo "[INFO] Loaded $TOTAL motion(s) from: $MOTIONS_JSON"
echo "[INFO] Task:             $TASK"
echo "[INFO] W&B project:      $LOG_PROJECT_NAME"
echo "[INFO] Max iterations:   $MAX_ITERATIONS"
[[ -n "$NUM_ENVS" ]] && echo "[INFO] Num envs:         $NUM_ENVS"
echo "[INFO] completed.log:    $COMPLETED_LOG"
echo "[INFO] failed.log:       $FAILED_LOG"
$DRY_RUN && echo "[INFO] *** DRY RUN — commands will be printed but not executed ***"
echo "============================================================"

SUCCESS=0
FAILED_COUNT=0
SKIPPED=0
IDX=0

for entry in "${ENTRIES[@]}"; do
    IDX=$(( IDX + 1 ))
    IFS='|' read -r name safe_name registry_name <<< "$entry"

    echo ""
    echo "------------------------------------------------------------"
    echo "[$IDX/$TOTAL] $name"
    echo "  registry_name : $registry_name"
    echo "  run_name      : $safe_name"

    # ---- skip already completed ---------------------------------------------
    if grep -qxF "$name" "$COMPLETED_LOG" 2>/dev/null; then
        echo "[SKIP] Already in completed.log — skipping."
        (( SKIPPED++ )) || true
        continue
    fi

    # ---- build command array ------------------------------------------------
    CMD=(
        "$PYTHON" "$TRAIN_SCRIPT"
        "--task=$TASK"
        "--registry_name=$registry_name"
        "--run_name=$safe_name"
        "--headless"
        "--logger=wandb"
        "--log_project_name=$LOG_PROJECT_NAME"
        "--max_iterations=$MAX_ITERATIONS"
    )
    [[ -n "$NUM_ENVS" ]] && CMD+=("--num_envs=$NUM_ENVS")

    echo "[CMD ] ${CMD[*]}"

    if $DRY_RUN; then
        (( SKIPPED++ )) || true
        continue
    fi

    # ---- run training and capture exit code ---------------------------------
    TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    set +e
    "${CMD[@]}"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[OK  ] Training finished successfully."
        echo "$name" >> "$COMPLETED_LOG"
        (( SUCCESS++ )) || true
    else
        echo "[FAIL] Training exited with code $EXIT_CODE."
        printf "%s\t%s\t%s\n" "$name" "$TIMESTAMP" "$EXIT_CODE" >> "$FAILED_LOG"
        (( FAILED_COUNT++ )) || true
    fi
done

# ---- summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "[SUMMARY] total=$TOTAL  success=$SUCCESS  skipped=$SKIPPED  failed=$FAILED_COUNT"
if [[ $FAILED_COUNT -gt 0 ]]; then
    echo "[WARN ] ${FAILED_COUNT} motion(s) failed. Details in: $FAILED_LOG"
    exit 1
fi
exit 0
