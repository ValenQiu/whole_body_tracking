#!/usr/bin/env bash
# wandb_sync.sh — 将本地 .wandb 历史数据补传到 wandb 服务端
#
# 两种模式：
#   sync  (默认)  : 对 crashed run 补传完整历史曲线，带超时保护
#   check         : 对 running run 检查是否在正常同步（看心跳时间）
#
# 用法：
#   bash wandb_sync.sh                            # 自动查询 wandb，同步所有 crashed run
#   bash wandb_sync.sh run-20260406_xxx-yyy ...   # 只同步指定目录
#   bash wandb_sync.sh --timeout 1800             # 自定义超时（秒），默认 3600
#   bash wandb_sync.sh --check-running            # 仅检查 running run 的同步状态
#   bash wandb_sync.sh --wandb-project entity/project   # 指定要查询的 W&B project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WANDB_BASE="$REPO_ROOT/wandb"
PYTHON="/isaac-sim/python.sh"

# ---------- 参数解析 ----------
TIMEOUT=3600
MODE="sync"           # sync | check
WANDB_PROJECT=""
SYNC_TARGETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)         TIMEOUT="$2"; shift 2 ;;
        --timeout=*)       TIMEOUT="${1#*=}"; shift ;;
        --check-running)   MODE="check"; shift ;;
        --wandb-project)   WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
        *) SYNC_TARGETS+=("$1"); shift ;;
    esac
done

# ---------- 从 wandb API 查询状态 ----------
query_wandb_states() {
    local project="$1"
    $PYTHON - "$project" 2>/dev/null << 'PYEOF'
import sys, json
import wandb

project = sys.argv[1]
api = wandb.Api()
runs = api.runs(project, per_page=200)
result = {}
for r in runs:
    result[r.id] = {
        "state": r.state,
        "name": r.name,
        "heartbeat_at": getattr(r, "heartbeat_at", None),
    }
print(json.dumps(result))
PYEOF
}

# ---------- 模式：check running ----------
if [[ "$MODE" == "check" ]]; then
    echo "====== [CHECK] 检查 running run 的同步状态 ======"

    # 自动从本地目录猜 project（取最新 run 目录的 wandb 元数据）
    if [[ -z "$WANDB_PROJECT" ]]; then
        WANDB_PROJECT=$($PYTHON - "$WANDB_BASE" 2>/dev/null << 'PYEOF'
import sys, os, glob, json

base = sys.argv[1]
for d in sorted(glob.glob(os.path.join(base, "run-*")), reverse=True):
    meta = os.path.join(d, "files", "wandb-metadata.json")
    if os.path.exists(meta):
        with open(meta) as f:
            data = json.load(f)
        entity  = data.get("entity", "")
        project = data.get("project", "")
        if entity and project:
            print(f"{entity}/{project}")
            break
PYEOF
        )
    fi

    if [[ -z "$WANDB_PROJECT" ]]; then
        echo "[ERROR] 无法自动识别 wandb project，请用 --wandb-project entity/project"
        exit 1
    fi

    echo "[INFO] 查询 project: $WANDB_PROJECT"
    STATES_JSON=$($PYTHON - "$WANDB_PROJECT" 2>/dev/null << 'PYEOF'
import sys, json, datetime
import wandb

project = sys.argv[1]
api = wandb.Api()
runs = api.runs(project, per_page=200)
now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

result = []
for r in runs:
    if r.state != "running":
        continue
    hb = getattr(r, "heartbeat_at", None)
    if hb:
        try:
            from dateutil import parser as dp
            hb_dt = dp.parse(str(hb))
            lag = int((now - hb_dt).total_seconds())
            lag_str = f"{lag}s ago"
            ok = "OK  " if lag < 120 else "WARN"
        except Exception:
            lag_str = str(hb)
            ok = "????"
    else:
        lag_str = "N/A"
        ok = "WARN"
    result.append((ok, r.id, r.name, lag_str))

for ok, rid, name, lag in sorted(result):
    print(f"  [{ok}] {rid}  {name}  heartbeat={lag}")
PYEOF
    )

    if [[ -z "$STATES_JSON" ]]; then
        echo "[INFO] 当前没有 running 状态的 run，或查询失败"
    else
        echo "$STATES_JSON"
        echo ""
        echo "说明: [OK  ] 心跳 <2min 正常同步  |  [WARN] 心跳超时，可能已断连但 wandb 未判定 crashed"
    fi
    exit 0
fi

# ---------- 模式：sync crashed ----------
echo "====== [SYNC] 补传 crashed run 的历史曲线 ======"

# 自动发现：未指定目标时，通过 wandb API 查出 crashed run ID，再匹配本地目录
if [[ ${#SYNC_TARGETS[@]} -eq 0 ]]; then
    # 先猜 project
    if [[ -z "$WANDB_PROJECT" ]]; then
        WANDB_PROJECT=$($PYTHON - "$WANDB_BASE" 2>/dev/null << 'PYEOF'
import sys, os, glob, json

base = sys.argv[1]
for d in sorted(glob.glob(os.path.join(base, "run-*")), reverse=True):
    meta = os.path.join(d, "files", "wandb-metadata.json")
    if os.path.exists(meta):
        with open(meta) as f:
            data = json.load(f)
        entity  = data.get("entity", "")
        project = data.get("project", "")
        if entity and project:
            print(f"{entity}/{project}")
            break
PYEOF
        )
    fi

    if [[ -n "$WANDB_PROJECT" ]]; then
        echo "[INFO] 查询 project $WANDB_PROJECT 中的 crashed run..."
        CRASHED_IDS=$($PYTHON - "$WANDB_PROJECT" 2>/dev/null << 'PYEOF'
import sys, json
import wandb

project = sys.argv[1]
api = wandb.Api()
runs = api.runs(project, per_page=200)
for r in runs:
    if r.state == "crashed":
        print(r.id)
PYEOF
        )

        if [[ -z "$CRASHED_IDS" ]]; then
            echo "[INFO] wandb 上没有 crashed run，无需同步"
            exit 0
        fi

        # 匹配本地目录
        for run_id in $CRASHED_IDS; do
            matched=$(find "$WANDB_BASE" -maxdepth 1 -type d -name "run-*-${run_id}" 2>/dev/null | head -1)
            if [[ -n "$matched" ]]; then
                SYNC_TARGETS+=("$(basename "$matched")")
                echo "[INFO] 发现 crashed run: $run_id → $(basename "$matched")"
            else
                echo "[WARN] crashed run $run_id 在本地 wandb/ 目录中未找到对应目录，跳过"
            fi
        done
    else
        echo "[WARN] 无法自动识别 project，请用 --wandb-project entity/project 或直接指定目录名"
        exit 1
    fi
fi

if [[ ${#SYNC_TARGETS[@]} -eq 0 ]]; then
    echo "[INFO] 没有需要同步的 run"; exit 0
fi

echo "[INFO] 共 ${#SYNC_TARGETS[@]} 个 crashed run 待同步，超时=${TIMEOUT}s/个"
RESULT_LOG="$WANDB_BASE/sync_results.log"
mkdir -p "$WANDB_BASE"
echo "===== wandb_sync 开始 $(date) =====" >> "$RESULT_LOG"

# ---------- 并行 sync ----------
declare -A PIDS

for target in "${SYNC_TARGETS[@]}"; do
    run_dir="$WANDB_BASE/$target"
    [[ -d "$run_dir" ]] || { echo "[SKIP] 目录不存在：$run_dir"; continue; }

    run_log="$WANDB_BASE/sync_${target}.log"
    echo "[START] $target"
    echo "        进度日志: tail -f $run_log"

    (
        START=$(date +%s)
        if timeout "$TIMEOUT" "$PYTHON" -m wandb sync "$run_dir" > "$run_log" 2>&1; then
            ELAPSED=$(( $(date +%s) - START ))
            echo "[OK   ] $target  耗时 ${ELAPSED}s" | tee -a "$RESULT_LOG"
        else
            CODE=$?
            ELAPSED=$(( $(date +%s) - START ))
            if [[ $CODE -eq 124 ]]; then
                echo "[TIMEOUT] $target  超过 ${TIMEOUT}s 未完成，已中止" | tee -a "$RESULT_LOG"
            else
                echo "[FAIL ] $target  退出码=$CODE  耗时 ${ELAPSED}s" | tee -a "$RESULT_LOG"
            fi
        fi
    ) &
    PIDS["$target"]=$!
done

echo ""
echo "[INFO] 汇总结果实时查看: tail -f $RESULT_LOG"
echo ""
echo "后台 PID："
for t in "${!PIDS[@]}"; do
    echo "  ${PIDS[$t]}  →  $t"
done

wait
echo "[DONE] 所有同步完成 $(date)" | tee -a "$RESULT_LOG"
echo "---" >> "$RESULT_LOG"
