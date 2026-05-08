---
name: env-setup
description: >-
  Docker 容器环境初始化与 whole_body_tracking 安装适配。覆盖以下场景：
  首次进入容器后的环境修复（/workspace/isaaclab 软链接、~/.bashrc alias 修复、wandb 兼容性）、
  whole_body_tracking 包安装（含 unitree_description 资产下载）、以及在无 wandb 登录时
  用本地 .npz 文件绕过 registry 直接运行 train.py。当用户遇到 "No module named 'isaaclab'"、
  python/pip alias 报错、train.py 运行失败、或需要从零配置环境时使用本 skill。
---

# 环境初始化与适配 (whole_body_tracking Docker 环境)

## 关键路径速查

| 名称 | 路径 |
|---|---|
| IsaacLab 源码 | `/workspace/qiulm@xiaopeng.com/source/IsaacLab` |
| Isaac Sim | `/isaac-sim/` |
| Isaac Sim Python | `/isaac-sim/python.sh` |
| whole_body_tracking | `/workspace/qiulm@xiaopeng.com/source/whole_body_tracking` |
| workspace 根目录软链接 | `/workspace/isaaclab` → IsaacLab 源码 |
| 本地 motion 文件 | `artifacts/jumps1_subject1:v1/motion.npz`（相对 whole_body_tracking/） |
| 环境初始化脚本 | `/workspace/qiulm@xiaopeng.com/source/setup_env.sh` |

---

## Step 1 — 基础环境修复（每次新容器必做）

### 1a. 运行 setup_env.sh

```bash
cd /workspace/qiulm@xiaopeng.com/source
bash setup_env.sh
source ~/.bashrc
```

脚本完成三件事：
1. 在 IsaacLab 目录下创建 `_isaac_sim -> /isaac-sim` 软链接
2. 修复 `~/.bashrc` 中 alias 的旧路径（`/workspace/isaaclab/_isaac_sim/...` → `/isaac-sim/python.sh`）
3. 检查并升级 wandb（修复 NumPy 2.0 兼容性）

### 1b. 创建 /workspace/isaaclab 软链接（关键！）

> **症状**：运行任何 Python 脚本时报 `ModuleNotFoundError: No module named 'isaaclab'`

**根本原因**：`isaaclab` 的 editable 安装路径写死为 `/workspace/isaaclab/...`，但该目录在容器中不存在。

**修复**：

```bash
ln -s "/workspace/qiulm@xiaopeng.com/source/IsaacLab" /workspace/isaaclab
```

验证：

```bash
/isaac-sim/python.sh -c "import isaaclab; print('OK:', isaaclab.__file__)"
# 预期输出：OK: /workspace/isaaclab/source/isaaclab/isaaclab/__init__.py
```

---

## Step 2 — 安装 whole_body_tracking

### 2a. 下载机器人描述文件

```bash
cd /workspace/qiulm@xiaopeng.com/source/whole_body_tracking

curl -L -o unitree_description.tar.gz \
    https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz \
    -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

解压后路径：`source/whole_body_tracking/whole_body_tracking/assets/unitree_description/`

### 2b. 安装包（editable 模式）

```bash
/isaac-sim/python.sh -m pip install -e source/whole_body_tracking
```

**已知依赖冲突**（不影响 whole_body_tracking 运行，但可能影响其他 IsaacLab 功能）：

| 包 | isaaclab 要求 | 安装后版本 |
|---|---|---|
| `onnx` | `==1.16.1` | `1.21.0` |
| `protobuf` | `<5.0.0` | `6.33.6` |
| `wandb` | `<0.13.0`（rl-games） | `0.25.1` |
| `prettytable` | `==3.3.0` | `3.17.0` |

---

## Step 3 — 运行 train.py（本地 motion 文件绕过 wandb）

### 问题背景

`train.py` 原本要求 `--registry_name`（必填），通过 wandb API 下载 motion artifact。
在未登录 wandb 的情况下会报：`wandb.errors.UsageError: No API key configured`。

### 修复方案：新增 --motion_file 参数

对 `scripts/rsl_rl/train.py` 做以下修改：

**1. 将 `--registry_name` 改为可选，新增 `--motion_file`**（第 27-28 行）：

```python
# 原代码
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

# 替换为
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Local path to a .npz motion file (bypasses wandb registry).")
```

**2. 在 `main()` 函数内替换 motion 文件加载逻辑**（原第 92-114 行）：

```python
# load the motion file: from local path or wandb registry
import pathlib

if args_cli.motion_file is not None:
    # Use local file directly, bypass wandb
    motion_file = pathlib.Path(args_cli.motion_file)
    if not motion_file.is_file():
        raise FileNotFoundError(f"Local motion file not found: {motion_file}")
    registry_name = args_cli.registry_name  # may be None when using local file
    print(f"[INFO]: Using local motion file (wandb bypassed): {motion_file}")
else:
    if args_cli.registry_name is None:
        raise ValueError("Either --registry_name or --motion_file must be provided.")
    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    artifact_dir = pathlib.Path(artifact.download())

    motion_file = artifact_dir / "motion.npz"
    if not motion_file.is_file():
        npz_files = sorted(artifact_dir.glob("*.npz"))
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No .npz motion file found under artifact directory: {artifact_dir}")
        motion_file = npz_files[0]
        if len(npz_files) > 1:
            print(
                f"[WARNING] Multiple .npz files found in artifact directory: {[p.name for p in npz_files]}. "
                f"Using: {motion_file.name}"
            )
```

### 修改后的启动命令

```bash
cd /workspace/qiulm@xiaopeng.com/source/whole_body_tracking/scripts/rsl_rl

# 本地 motion 文件（无需 wandb，适合测试）
/isaac-sim/python.sh train.py \
  --task=Tracking-Flat-G1-v0 \
  --motion_file=/workspace/qiulm@xiaopeng.com/source/whole_body_tracking/artifacts/jumps1_subject1:v1/motion.npz \
  --num_envs=16 \
  --max_iterations=5 \
  --headless

# wandb registry（需先 wandb login）
/isaac-sim/python.sh train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name {your-org}/wandb-registry-motions/{motion_name} \
  --num_envs=4096 \
  --headless --logger wandb --log_project_name {project_name}
```

---

## 完整初始化检查清单

每次进入新容器后按顺序执行：

```bash
# 1. 基础环境
cd /workspace/qiulm@xiaopeng.com/source
bash setup_env.sh && source ~/.bashrc

# 2. isaaclab 软链接（最常被忘记！）
[ -L /workspace/isaaclab ] || ln -s "/workspace/qiulm@xiaopeng.com/source/IsaacLab" /workspace/isaaclab

# 3. 验证 isaaclab 可导入
/isaac-sim/python.sh -c "import isaaclab; print('isaaclab OK')"

# 4. 验证 whole_body_tracking 可导入
/isaac-sim/python.sh -c "import whole_body_tracking; print('whole_body_tracking OK')"
```

---

## 常见报错速查

| 报错信息 | 原因 | 修复 |
|---|---|---|
| `No module named 'isaaclab'` | `/workspace/isaaclab` 软链接不存在 | `ln -s /workspace/qiulm@xiaopeng.com/source/IsaacLab /workspace/isaaclab` |
| `command not found: python` / `alias python` 报错 | `~/.bashrc` 未 source | `source ~/.bashrc` 或重开终端 |
| `wandb.errors.UsageError: No API key configured` | wandb 未登录 | 用 `--motion_file` 本地路径参数，或先执行 `wandb login` |
| `FileNotFoundError: unitree_description` | assets 未下载 | 执行 Step 2a 下载资产 |
| `Error: Not Found ... unitree_description/urdf/g1/main.urdf` | 机器人 URDF 资产缺失 | 检查并补齐 `source/whole_body_tracking/whole_body_tracking/assets/unitree_description/` |
| `X11 Forwarding is disabled from '.container.cfg'` | 容器 X11 转发被关闭 | 将 `X11_FORWARDING_ENABLED=1` 后重启容器，或删除 `.container.cfg` 重新选择启用 |
| `Cannot setup ExternalDragDrop without a default window` / `get_keyboard` | GUI 模式下默认窗口不可用（X11 cookie/display 不可用） | 容器环境先修复 X11，或直接使用 `--headless` |
| `Disabling key-value database because another kit process is locking it` | 仍有其他 Isaac Sim / replay 进程运行 | 先清理残留进程再启动新任务 |
| `DriverShaderCacheManager::init() called without a shutdown()` | GPU shader cache 警告 | 无需处理，不影响运行 |
| `GLFW initialization failed` | headless 模式无显示器 | 正常现象，加 `--headless` 运行即可 |
