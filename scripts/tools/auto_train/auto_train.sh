#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# auto_train.sh  –  Sequentially train one policy per motion in a W&B registry.
#
# Usage (from whole_body_tracking/ root):
#   bash scripts/tools/auto_train/auto_train.sh [OPTIONS]
#
# Auto-fetch args (W&B is queried fresh on every invocation):
#   --registry_name PATH   W&B registry path for fetch_registry.py.
#                          (default: liuming-valen-qiu-the-hong-kong-polytechnic-university-org/wandb-registry-Motions)
#   --wandb_project  PATH  W&B entity/project used to check existing run states.
#                          (default: liuming-valen-qiu-the-hong-kong-polytechnic-university/g1_lafan1_motion_tracking)
#
# Modes (pick at most one; default is resume):
#   (default)        Resume — skip finished + running, retry crashed/failed/absent.
#   --retrain        Retrain all motions from scratch, ignoring W&B history.
#   --motions N ...  Train only the named motions (space-separated safe_names or names).
#                    Resume skip logic still applies unless --retrain is also given.
#
# Training overrides:
#   --task  NAME       Isaac Lab task ID.            (default: Tracking-Flat-G1-v0)
#   --project NAME     W&B training log project.     (default: g1_lafan1_motion_tracking)
#   --max_iter N       Override max training iters.  (default: 30000)
#   --num_envs N       Override parallel env count.  (default: from config)
#
# Flags:
#   --dry_run          Print commands without executing.
#
# Legacy bypass (skips registry auto-fetch):
#   --json PATH        Use this motions JSON directly instead of fetching.
#                      W&B run state check is still performed via --wandb_project.
#
# State files (written to scripts/tools/auto_train/logs/<date_time>_auto_train/):
#   failed.log         "<name>\t<ISO timestamp>\t<exit_code>" per failure.
# ---------------------------------------------------------------------------

set -uo pipefail   # intentionally no -e; failure isolation is handled per-job

# ---- defaults ---------------------------------------------------------------
REGISTRY_NAME="liuming-valen-qiu-the-hong-kong-polytechnic-university-org/wandb-registry-Motions"
WANDB_PROJECT="liuming-valen-qiu-the-hong-kong-polytechnic-university/g1_lafan1_motion_tracking"
TASK="Tracking-Flat-G1-v0"
LOG_PROJECT_NAME="g1_lafan1_motion_tracking"
MAX_ITERATIONS="30000"
NUM_ENVS=""
MOTIONS_JSON=""       # set by --json to skip registry auto-fetch
RETRAIN=false
MOTIONS_FILTER=""     # space-separated names from --motions
DRY_RUN=false

PYTHON="/isaac-sim/python.sh"
TRAIN_SCRIPT="scripts/rsl_rl/train.py"
FETCH_REGISTRY_SCRIPT="scripts/tools/auto_train/fetch_registry.py"
FETCH_RUNS_SCRIPT="scripts/tools/auto_train/fetch_runs.py"
RUN_TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
LOG_DIR="scripts/tools/auto_train/logs/${RUN_TIMESTAMP}_auto_train"

# ---- argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry_name) REGISTRY_NAME="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --json)          MOTIONS_JSON="$2"; shift 2 ;;
        --task)          TASK="$2"; shift 2 ;;
        --project)       LOG_PROJECT_NAME="$2"; shift 2 ;;
        --max_iter)      MAX_ITERATIONS="$2"; shift 2 ;;
        --num_envs)      NUM_ENVS="$2"; shift 2 ;;
        --retrain)       RETRAIN=true; shift ;;
        --motions)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                MOTIONS_FILTER="$MOTIONS_FILTER $1"
                shift
            done
            MOTIONS_FILTER="${MOTIONS_FILTER# }"  # trim leading space
            ;;
        --dry_run)       DRY_RUN=true; shift ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ---- sanity checks ----------------------------------------------------------
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[ERROR] Train script not found: $TRAIN_SCRIPT"
    echo "        Run this script from the whole_body_tracking/ root directory."
    exit 1
fi

if [[ ! "$DRY_RUN" == true && ! -f "$PYTHON" ]]; then
    echo "[ERROR] Python interpreter not found: $PYTHON"
    exit 1
fi

mkdir -p "$LOG_DIR"

# ---- auto-fetch registry (skipped when --json is given) ---------------------
if [[ -z "$MOTIONS_JSON" ]]; then
    _reg_part="$(echo "$REGISTRY_NAME" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9_-]/_/g')"
    MOTIONS_JSON="$LOG_DIR/${_reg_part}.json"
    echo "[INFO] Fetching registry: $REGISTRY_NAME"
    if ! "$PYTHON" "$FETCH_REGISTRY_SCRIPT" --registry_name "$REGISTRY_NAME" --output "$MOTIONS_JSON"; then
        echo "[ERROR] fetch_registry.py failed."
        exit 1
    fi
else
    echo "[INFO] Using --json bypass: $MOTIONS_JSON"
fi

if [[ ! -f "$MOTIONS_JSON" ]]; then
    echo "[ERROR] Motions JSON not found: $MOTIONS_JSON"
    exit 1
fi

# ---- fetch runs status (always, for W&B-based coordination) -----------------
RUNS_JSON=""
if [[ -n "$WANDB_PROJECT" ]]; then
    _proj_part="$(echo "$WANDB_PROJECT" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9_-]/_/g')"
    RUNS_JSON="$LOG_DIR/${_proj_part}.json"
    echo "[INFO] Fetching runs status: $WANDB_PROJECT"
    if ! "$PYTHON" "$FETCH_RUNS_SCRIPT" --project "$WANDB_PROJECT" --output "$RUNS_JSON" > /dev/null 2>&1; then
        echo "[WARN] fetch_runs.py failed — all motions will be treated as untrained."
        RUNS_JSON=""
    fi
fi

# ---- state files ------------------------------------------------------------
FAILED_LOG="$LOG_DIR/failed.log"
touch "$FAILED_LOG"

# ---- resolve training queue -------------------------------------------------
# Write the resolver script to a temp file (avoids heredoc-in-subshell issues).
QUEUE_SCRIPT="$(mktemp /tmp/auto_train_queue_XXXXXX.py)"
cat > "$QUEUE_SCRIPT" << 'PYEOF'
import json, re, sys

motions_path   = sys.argv[1]
runs_path      = sys.argv[2]   # empty string when unavailable
retrain        = sys.argv[3].lower() == "true"
motions_filter = sys.argv[4].split() if sys.argv[4].strip() else []


def safe_name_from_run(run_name):
    """Strip the YYYY-MM-DD_HH-MM-SS_ timestamp prefix from a run name."""
    m = re.match(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$', run_name)
    return m.group(1) if m else None


with open(motions_path) as f:
    registry = json.load(f)

runs_finished = set()
runs_running  = set()

if runs_path:
    try:
        with open(runs_path) as f:
            runs_data = json.load(f)
        for r in runs_data.get("runs", []):
            sn = safe_name_from_run(r["name"])
            if sn is None:
                continue
            if r["state"] == "finished":
                runs_finished.add(sn)
            elif r["state"] == "running":
                runs_running.add(sn)
    except Exception as e:
        print(f"[WARN] Could not parse runs JSON: {e}", file=sys.stderr)

for m in registry["motions"]:
    name = m["name"]
    safe = m["safe_name"]
    reg  = m["registry_name"]

    # --motions filter: accept by name or safe_name
    if motions_filter and name not in motions_filter and safe not in motions_filter:
        continue

    if not retrain:
        if safe in runs_finished:
            print(f"[SKIP:finished] {name}", file=sys.stderr)
            continue
        if safe in runs_running:
            print(f"[SKIP:running ] {name}", file=sys.stderr)
            continue

    # Output line consumed by the shell training loop
    print(f"{name}|{safe}|{reg}")
PYEOF

QUEUE_FILE="$(mktemp)"
# Run once: stderr (skip messages) → terminal, stdout (queue) → file
"$PYTHON" "$QUEUE_SCRIPT" \
    "$MOTIONS_JSON" \
    "${RUNS_JSON}" \
    "$RETRAIN" \
    "$MOTIONS_FILTER" \
    > "$QUEUE_FILE"
rm -f "$QUEUE_SCRIPT"

mapfile -t ENTRIES < "$QUEUE_FILE"
rm -f "$QUEUE_FILE"

TOTAL=${#ENTRIES[@]}

# ---- per-motion re-fetch check script (written once, reused in loop) --------
RECHECK_SCRIPT="$(mktemp /tmp/auto_train_recheck_XXXXXX.py)"
cat > "$RECHECK_SCRIPT" << 'PYEOF'
import json, re, sys

runs_path    = sys.argv[1]
safe         = sys.argv[2]


def safe_name_from_run(n):
    m = re.match(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$', n)
    return m.group(1) if m else None


with open(runs_path) as f:
    data = json.load(f)

skip_states = {"finished", "running"}

for r in data.get("runs", []):
    if safe_name_from_run(r["name"]) == safe and r["state"] in skip_states:
        print(r["state"])
        break
PYEOF

# ---- header -----------------------------------------------------------------
MODE_DESC="resume"
if $RETRAIN; then
    MODE_DESC="retrain (ignore W&B history)"
elif [[ -n "$MOTIONS_FILTER" ]]; then
    MODE_DESC="motions: $MOTIONS_FILTER"
fi

echo "============================================================"
echo "[INFO] Registry:         $REGISTRY_NAME"
echo "[INFO] Runs project:     $WANDB_PROJECT"
echo "[INFO] Task:             $TASK"
echo "[INFO] Log project:      $LOG_PROJECT_NAME"
echo "[INFO] Max iterations:   $MAX_ITERATIONS"
[[ -n "$NUM_ENVS" ]] && echo "[INFO] Num envs:         $NUM_ENVS"
echo "[INFO] Mode:             $MODE_DESC"
echo "[INFO] Queue size:       $TOTAL"
echo "[INFO] failed.log:       $FAILED_LOG"
$DRY_RUN && echo "[INFO] *** DRY RUN — commands will be printed but not executed ***"
echo "============================================================"

if [[ $TOTAL -eq 0 ]]; then
    echo "[INFO] Nothing to train — all motions are finished or running on another server."
    rm -f "$RECHECK_SCRIPT"
    exit 0
fi

SUCCESS=0
FAILED_COUNT=0
SKIPPED=0
IDX=0

for entry in "${ENTRIES[@]}"; do
    IDX=$(( IDX + 1 ))
    IFS='|' read -r name safe_name registry_name_entry <<< "$entry"

    echo ""
    echo "------------------------------------------------------------"
    echo "[$IDX/$TOTAL] $name"
    echo "  registry_name : $registry_name_entry"
    echo "  run_name      : $safe_name"

    # ---- per-motion re-fetch (multi-server safety) --------------------------
    # Immediately before launching train.py, re-fetch runs from W&B and check
    # if another server has claimed this motion since the startup fetch.
    if [[ -n "$RUNS_JSON" ]] && ! $RETRAIN && ! $DRY_RUN; then
        if "$PYTHON" "$FETCH_RUNS_SCRIPT" --project "$WANDB_PROJECT" --output "$RUNS_JSON" > /dev/null 2>&1; then
            SKIP_REASON="$("$PYTHON" "$RECHECK_SCRIPT" "$RUNS_JSON" "$safe_name" 2>/dev/null || true)"
            if [[ -n "$SKIP_REASON" ]]; then
                echo "[SKIP] Re-fetch shows state=$SKIP_REASON on W&B — skipping."
                (( SKIPPED++ )) || true
                continue
            fi
        fi
    fi

    # ---- build command array ------------------------------------------------
    CMD=(
        "$PYTHON" "$TRAIN_SCRIPT"
        "--task=$TASK"
        "--registry_name=$registry_name_entry"
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
        (( SUCCESS++ )) || true
    else
        echo "[FAIL] Training exited with code $EXIT_CODE."
        printf "%s\t%s\t%s\n" "$name" "$TIMESTAMP" "$EXIT_CODE" >> "$FAILED_LOG"
        (( FAILED_COUNT++ )) || true
    fi
done

rm -f "$RECHECK_SCRIPT"

# ---- summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo "[SUMMARY] total=$TOTAL  success=$SUCCESS  skipped=$SKIPPED  failed=$FAILED_COUNT"
if [[ $FAILED_COUNT -gt 0 ]]; then
    echo "[WARN ] ${FAILED_COUNT} motion(s) failed. Details in: $FAILED_LOG"
    exit 1
fi
exit 0
