#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch convert all CSV motion files in INPUT_DIR to NPZ using csv_to_npz.py
#
# Usage (from whole_body_tracking/):
#   bash scripts/run_csv_to_npz_all.sh [--upload_wandb] [--skip_existing]
#
# Defaults:
#   INPUT_DIR  : /workspace/isaaclab/source/Datasets/lafan1_g1/g1
#   OUTPUT_DIR : /workspace/isaaclab/source/Datasets/lafan1_g1/g1_npz
#   INPUT_FPS  : 30
#   DISABLE_WANDB: true (pass --upload_wandb to enable)
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- configurable paths ----------------------------------------------------
INPUT_DIR="/workspace/isaaclab/source/Datasets/lafan1_g1/g1"
OUTPUT_DIR="/workspace/isaaclab/source/Datasets/lafan1_g1/g1_npz"
INPUT_FPS=30
OUTPUT_FPS=50
SCRIPT="scripts/csv_to_npz.py"

# Isaac Sim's Python wrapper (registered as alias python3 -> python.sh).
PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
echo "[INFO] Using Python: $PYTHON"

# ---- parse optional flags --------------------------------------------------
DISABLE_WANDB="--disable_wandb"
SKIP_EXISTING=false

for arg in "$@"; do
    case $arg in
        --upload_wandb)   DISABLE_WANDB="" ;;
        --skip_existing)  SKIP_EXISTING=true ;;
        *) echo "[WARN] Unknown argument: $arg" ;;
    esac
done

# ---- sanity checks ---------------------------------------------------------
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] Input directory not found: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t CSV_FILES < <(find "$INPUT_DIR" -maxdepth 1 -name "*.csv" | sort)
TOTAL=${#CSV_FILES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "[ERROR] No CSV files found in $INPUT_DIR"
    exit 1
fi

echo "[INFO] Found $TOTAL CSV files."
echo "[INFO] Output dir: $OUTPUT_DIR"
[[ -z "$DISABLE_WANDB" ]] && echo "[INFO] WandB upload: ENABLED" || echo "[INFO] WandB upload: disabled"
echo "------------------------------------------------------------"

SUCCESS=0
FAILED=0
SKIPPED=0

for csv_file in "${CSV_FILES[@]}"; do
    motion_name=$(basename "$csv_file" .csv)
    output_file="$OUTPUT_DIR/${motion_name}.npz"

    if [[ "$SKIP_EXISTING" == true && -f "$output_file" ]]; then
        echo "[SKIP] $motion_name"
        (( SKIPPED++ )) || true
        continue
    fi

    echo "[RUN ] $motion_name  ->  $output_file"

    if "$PYTHON" "$SCRIPT" \
        --input_file "$csv_file" \
        --input_fps "$INPUT_FPS" \
        --output_fps "$OUTPUT_FPS" \
        --output_name "$motion_name" \
        --output_file "$output_file" \
        --headless \
        $DISABLE_WANDB; then
        (( SUCCESS++ )) || true
    else
        echo "[ERROR] Failed: $motion_name"
        (( FAILED++ )) || true
    fi
done

echo "------------------------------------------------------------"
echo "[SUMMARY] success=$SUCCESS  skipped=$SKIPPED  failed=$FAILED"
[[ $FAILED -eq 0 ]] && exit 0 || exit 1
