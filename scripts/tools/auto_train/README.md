# auto_train — 批量训练自动化工具

在 W&B Registry 中枚举所有 motion artifact，串行提交多个独立的 policy 训练任务。

## 文件说明

| 文件 | 作用 |
|------|------|
| `fetch_registry.py` | 从 W&B Registry 枚举所有 motion 集合，生成 JSON 调度表 |
| `auto_train.sh` | 读取 JSON，串行调用 `train.py`，支持断点续传和失败隔离 |
| `motions.json` | （运行后生成）动作调度表 |
| `completed.log` | （运行后生成）已成功训练的动作名，重跑时自动跳过 |
| `failed.log` | （运行后生成）失败记录（名称、时间戳、退出码） |

---

## 使用流程

### 第一步：生成 motions.json

```bash
# 在容器内，从 whole_body_tracking/ 根目录运行
python scripts/tools/auto_train/fetch_registry.py \
    --registry_name "your-org/wandb-registry-Motions" \
    --output scripts/tools/auto_train/motions.json
```

可选参数：
- `--filter "walk.*"` — 正则过滤，只保留匹配的动作名
- `--dry_run` — 预览列表，不写文件

### 第二步：dry_run 预览训练命令

```bash
bash scripts/tools/auto_train/auto_train.sh \
    --json scripts/tools/auto_train/motions.json \
    --project g1_lafan1_motion_tracking \
    --dry_run
```

### 第三步：正式训练

```bash
# 前台运行（可见输出）
bash scripts/tools/auto_train/auto_train.sh \
    --json scripts/tools/auto_train/motions.json \
    --project g1_lafan1_motion_tracking

# 后台运行（推荐，防止 SSH 断开中断）
nohup bash scripts/tools/auto_train/auto_train.sh \
    --json scripts/tools/auto_train/motions.json \
    --project g1_lafan1_motion_tracking \
    > logs/auto_train.log 2>&1 &
echo "PID: $!"
```

---

## auto_train.sh 参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--json` | 是 | — | motions.json 路径 |
| `--task` | 否 | `Tracking-Flat-G1-v0` | Isaac Lab 任务 ID |
| `--project` | 否 | `g1_motion_tracking` | W&B 训练 project 名 |
| `--max_iter` | 否 | `30000` | 覆盖最大迭代数 |
| `--num_envs` | 否 | （从配置读取） | 覆盖并行环境数 |
| `--dry_run` | 否 | false | 只打印命令不执行 |

---

## 断点续传

脚本每完成一个动作就将名称写入 `completed.log`。中断后直接重新运行同一条命令，已完成的动作会被自动跳过：

```bash
# 查看进度
cat scripts/tools/auto_train/completed.log

# 查看失败记录
cat scripts/tools/auto_train/failed.log

# 监控后台日志
tail -f logs/auto_train.log
```

## 快速冒烟测试

用 `--max_iter 5` 跑完整流程验证通路，不会产生有效 checkpoint：

```bash
bash scripts/tools/auto_train/auto_train.sh \
    --json scripts/tools/auto_train/motions.json \
    --project g1_lafan1_motion_tracking \
    --max_iter 5
```
