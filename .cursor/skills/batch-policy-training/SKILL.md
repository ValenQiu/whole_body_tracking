---
name: batch-policy-training
description: >-
  Automate sequential policy training across an entire motion dataset in the
  whole_body_tracking project. Covers generating a W&B registry motion list (fetch_registry.py),
  running the batch training shell scheduler (auto_train.sh), breakpoint resumption via
  completed.log, and failure isolation. Use when the user asks about batch training,
  automating policy training, training all lafan1 motions, scheduling multiple train.py
  runs, or managing the auto_train tool.
---

# Batch Policy Training (whole_body_tracking)

Automates sequential policy training over all motions in a W&B Registry by coordinating
`fetch_registry.py` → `motions.json` → `auto_train.sh` → `train.py`.

## Tool location

```
scripts/tools/auto_train/
├── fetch_registry.py   # enumerate W&B registry → motions.json
├── auto_train.sh       # serial scheduler (reads motions.json)
├── motions.json        # generated schedule file
├── completed.log       # generated: successfully trained motions
├── failed.log          # generated: failures with timestamp + exit code
└── README.md           # full usage reference
```

## Workflow

### 1. Generate motions.json

```bash
python scripts/tools/auto_train/fetch_registry.py \
    --registry_name "your-org/wandb-registry-Motions" \
    --output scripts/tools/auto_train/motions.json
```

`--registry_name` accepts the project path (`entity/wandb-registry-Type`) or any full
artifact path — the collection/version suffix is ignored.  The script tries the extracted
registry type, then its lowercase variant, then enumerates all artifact types in the project
(handles case mismatches between the registry display name and the actual artifact `type` field).

### 2. Dry-run preview

```bash
bash scripts/tools/auto_train/auto_train.sh \
    --json scripts/tools/auto_train/motions.json \
    --project <wandb_project> --dry_run
```

### 3. Run training (background recommended)

```bash
nohup bash scripts/tools/auto_train/auto_train.sh \
    --json scripts/tools/auto_train/motions.json \
    --project g1_lafan1_motion_tracking \
    > logs/auto_train.log 2>&1 &
```

Key `auto_train.sh` options: `--task`, `--project`, `--max_iter`, `--num_envs`, `--dry_run`.

## State management

- `completed.log` — one motion name per line; re-runs skip these automatically.
- `failed.log` — tab-separated: `name \t ISO-timestamp \t exit_code`.
- A single failure does NOT stop the loop; the next motion is attempted regardless.

## Smoke test

```bash
bash scripts/tools/auto_train/auto_train.sh \
    --json scripts/tools/auto_train/motions.json \
    --project g1_lafan1_motion_tracking --max_iter 5
```

`save_interval = 2000`, so no checkpoint is written at 5 iterations — expected behaviour.

## Key constraints

- Each `train.py` call is a **separate process** (fresh Isaac Sim instance); runs are serial.
- `--registry_name` in `train.py` is **required**; motion is injected via W&B artifact download.
- `experiment_name` defaults to `g1_flat` (from `rsl_rl_ppo_cfg.py`); `run_name` is set to the sanitized motion name.
- `save_interval` is set to **2000** in `G1FlatPPORunnerCfg` (changed from 500).

## Full usage reference

See [README.md](README.md) in the same directory.
