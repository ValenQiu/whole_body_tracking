# auto_train — 批量训练自动化工具

在 W&B Registry 中枚举所有 motion artifact，串行提交多个独立的 policy 训练任务。

W&B run 状态（`finished` / `running`）作为全局协调依据，支持多台服务器并行运行而不重复训练同一 motion。

## 文件说明

| 文件 | 作用 |
|------|------|
| `fetch_registry.py` | 从 W&B Registry 枚举所有 motion 集合，生成调度 JSON |
| `fetch_runs.py` | 从 W&B Project 拉取已有 run 的状态，生成状态 JSON |
| `auto_train.sh` | 主入口：自动 fetch → 计算队列 → 串行训练 |
| `logs/wandb-registry-Motions.json` | （运行后生成）registry 调度表 |
| `logs/g1_lafan1_motion_tracking.json` | （运行后生成）W&B runs 状态表 |
| `logs/completed.log` | （运行后生成）本地已完成记录，重跑时快速跳过 |
| `logs/failed.log` | （运行后生成）失败记录（名称、时间戳、退出码） |

---

## 快速开始

### 默认 resume 模式（推荐日常使用）

```bash
# 无需任何参数；自动 fetch registry + runs，跳过已 finished/running 的 motion
bash scripts/tools/auto_train/auto_train.sh
```

后台运行（防止 SSH 断开）：

```bash
nohup bash scripts/tools/auto_train/auto_train.sh \
    > logs/auto_train.log 2>&1 &
echo "PID: $!"
```

### 查找正在运行的训练进程

```bash
# 查看 auto_train.sh 的 PID
pgrep -fa "auto_train.sh"

# 查看当前正在运行的 train.py 进程（含完整参数）
pgrep -fa "train.py"

# 或者查看后台日志，确认当前正在训练哪个 motion
tail -f logs/auto_train.log
```

### 停止训练

```bash
# 方法一：只终止 auto_train.sh 调度器（当前 motion 训练完后不再启动下一个）
# 先找到 auto_train.sh 的 PID
pgrep -fa "auto_train.sh"
kill <PID>

# 方法二：同时终止调度器和当前正在运行的 train.py
pkill -f "auto_train.sh"
pkill -f "train.py"

# 方法三：强制立即终止所有相关进程（train.py 可能来不及调用 wandb.finish）
pkill -9 -f "auto_train.sh"
pkill -9 -f "train.py"
```

> **注意**：方法三（`kill -9`）会跳过 `wandb.finish()`，该 run 在 W&B 上会显示为 `crashed`。
> 如果只是想在当前 motion 训练完后优雅停止，推荐使用方法一。

---

## auto_train.sh 参数

### Auto-fetch 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--registry_name` | `liuming-valen-qiu-the-hong-kong-polytechnic-university-org/wandb-registry-Motions` | W&B registry 路径，传给 `fetch_registry.py` |
| `--wandb_project` | `liuming-valen-qiu-the-hong-kong-polytechnic-university/g1_lafan1_motion_tracking` | W&B entity/project，传给 `fetch_runs.py` 查询已有 run 状态 |

### 训练模式（三选一）

| 参数 | 说明 |
|------|------|
| （默认）resume | 跳过 `finished` + `running`，重训 `crashed`/`failed`/未训练 |
| `--retrain` | 忽略 W&B 历史，重训所有 motion |
| `--motions name1 name2 ...` | 只训练指定名称（支持 name 或 safe_name），仍受 resume 跳过规则约束，可与 `--retrain` 组合使用 |

### 训练覆盖参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `Tracking-Flat-G1-v0` | Isaac Lab 任务 ID |
| `--project` | `g1_lafan1_motion_tracking` | W&B 训练 log project 名 |
| `--max_iter` | `30000` | 覆盖最大迭代数 |
| `--num_envs` | （从配置读取） | 覆盖并行环境数 |
| `--dry_run` | — | 只打印命令不执行 |

### Legacy 参数

| 参数 | 说明 |
|------|------|
| `--json PATH` | 跳过 registry auto-fetch，直接使用指定的 motions JSON；W&B run 状态检查仍正常进行 |

---

## 使用示例

```bash
# 查看队列（不执行训练）
bash scripts/tools/auto_train/auto_train.sh --dry_run

# 重训所有 motion（忽略历史）
bash scripts/tools/auto_train/auto_train.sh --retrain

# 只训练指定 motion
bash scripts/tools/auto_train/auto_train.sh --motions walk3_subject2 run1_subject2

# 指定不同 registry 和训练项目
bash scripts/tools/auto_train/auto_train.sh \
    --registry_name "my-org/wandb-registry-Motions" \
    --wandb_project "my-entity/my_project" \
    --project my_project

# 快速冒烟测试（5 次迭代，写入独立 project 不污染生产）
bash scripts/tools/auto_train/auto_train.sh \
    --project test_auto_run \
    --max_iter 5
```

---

## Dry-run 模式测试各功能

在不启动任何训练的情况下，用 `--dry_run` 验证队列逻辑是否正确。

### 1. Resume 模式（默认）

```bash
bash scripts/tools/auto_train/auto_train.sh --dry_run
```

预期：已 `finished`/`running` 的 motion 显示 `[SKIP:finished]` / `[SKIP:running]`；`completed.log` 中的显示 `[SKIP] Already in completed.log`；其余全部显示 `[CMD]`。

### 2. Retrain 模式（强制重训所有）

```bash
bash scripts/tools/auto_train/auto_train.sh --retrain --dry_run
```

预期：所有 40 个 motion 全部显示 `[CMD]`，无任何 `[SKIP]`。

### 3. 指定单个 motion

```bash
bash scripts/tools/auto_train/auto_train.sh --motions walk3_subject2 --dry_run
```

预期：只显示 `walk3_subject2` 的结果（`[CMD]` 或 `[SKIP]` 取决于其当前 W&B 状态）。

### 4. 指定多个 motion

```bash
bash scripts/tools/auto_train/auto_train.sh \
    --motions walk3_subject2 run1_subject2 dance1_subject1 \
    --dry_run
```

预期：只显示这三个 motion，各自按 W&B 状态独立判断。

### 5. 指定 motion + retrain（强制，无视历史）

```bash
bash scripts/tools/auto_train/auto_train.sh \
    --motions walk3_subject2 run1_subject2 \
    --retrain --dry_run
```

预期：这两个 motion 强制显示 `[CMD]`，不受 `completed.log` 或 W&B 状态影响。

### 预期队列大小速查

| 命令 | 预期队列大小 |
|---|---|
| `--dry_run` | registry 总数 − finished − running |
| `--retrain --dry_run` | registry 总数（全部） |
| `--motions X --dry_run` | 0 或 1 |
| `--motions X Y --retrain --dry_run` | 2（强制） |

---

## 多服务器并行训练

多台服务器可以同时运行 `auto_train.sh`，W&B run 状态作为协调依据：

| W&B 状态 | 默认 resume | --retrain |
|---|---|---|
| `finished` | 跳过 | 重训 |
| `running` | 跳过（软锁） | 重训 |
| `crashed` / `failed` | 重训 | 重训 |
| 未出现 | 训练 | 训练 |

### 竞争窗口说明

训练进程调用 `wandb.init()` 之前（Isaac Sim 启动约需 2 分钟），W&B 尚未将该 motion 标记为 `running`。脚本在每个 motion 开始前都会重新 fetch runs 以缩短窗口，但无法完全消除：

```
Server A：fetch → motion X = untrained → 启动 train.py
  [~2 min Isaac Sim 初始化]
  wandb.init() → X 变为 running ✓

Server B（>2 min 后启动）：fetch → X = running → 跳过 ✓
Server B（<2 min 后启动）：fetch → X = 仍未出现 → 重复训练 ✗
```

**建议：多台服务器错开至少 5 分钟启动**，可最大限度避免重复训练。

---

## 进度监控

```bash
# 查看本地已完成
cat scripts/tools/auto_train/logs/completed.log

# 查看失败记录
cat scripts/tools/auto_train/logs/failed.log

# 监控后台日志
tail -f logs/auto_train.log
```
