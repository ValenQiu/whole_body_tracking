#!/usr/bin/env bash
# =============================================================================
# setup_env.sh  —  IsaacLab Docker 环境初始化脚本
#
# 问题背景：
#   Docker 镜像中 ~/.bashrc 里的 alias 硬编码了旧路径
#   /workspace/isaaclab/_isaac_sim/python.sh，但：
#     1. /workspace/isaaclab/ 目录不存在（未创建软链接）
#     2. IsaacLab 实际位于用户自己的 source 目录
#     3. Isaac Sim 实际安装在 /isaac-sim/
#   导致每次启动 kernel 后 python / pip 命令均报错。
#
# 本脚本解决的问题：
#   1. 在 IsaacLab 目录下创建 _isaac_sim -> /isaac-sim 软链接
#   2. 修复 ~/.bashrc 中的 alias，直接指向 /isaac-sim/python.sh
#   3. 升级 wandb，解决与 NumPy 2.0 的兼容性问题（np.float_ 已移除）
#
# 使用方式：
#   bash setup_env.sh
#   source ~/.bashrc   # 或重新开一个终端
# =============================================================================

set -euo pipefail

ISAACLAB_SOURCE="/workspace/qiulm@xiaopeng.com/source/IsaacLab"
ISAAC_SIM_ROOT="/isaac-sim"
BASHRC="$HOME/.bashrc"

echo "============================================================"
echo "[SETUP] IsaacLab Docker 环境初始化"
echo "============================================================"

# ------------------------------------------------------------
# Step 1: 创建 _isaac_sim 软链接
# ------------------------------------------------------------
echo ""
echo "[1/3] 检查 _isaac_sim 软链接 ..."

SYMLINK_PATH="$ISAACLAB_SOURCE/_isaac_sim"

if [[ ! -e "$ISAAC_SIM_ROOT" ]]; then
    echo "[ERROR] Isaac Sim 未找到: $ISAAC_SIM_ROOT"
    exit 1
fi

if [[ -L "$SYMLINK_PATH" && "$(readlink "$SYMLINK_PATH")" == "$ISAAC_SIM_ROOT" ]]; then
    echo "[SKIP]  _isaac_sim 软链接已存在且正确，跳过。"
elif [[ -e "$SYMLINK_PATH" ]]; then
    echo "[WARN]  $SYMLINK_PATH 已存在但指向不正确，重新创建 ..."
    rm -f "$SYMLINK_PATH"
    ln -s "$ISAAC_SIM_ROOT" "$SYMLINK_PATH"
    echo "[OK]    $SYMLINK_PATH -> $ISAAC_SIM_ROOT"
else
    ln -s "$ISAAC_SIM_ROOT" "$SYMLINK_PATH"
    echo "[OK]    已创建: $SYMLINK_PATH -> $ISAAC_SIM_ROOT"
fi

# ------------------------------------------------------------
# Step 2: 修复 ~/.bashrc 中的 alias 和 ISAACLAB_PATH
# ------------------------------------------------------------
echo ""
echo "[2/3] 检查并修复 ~/.bashrc 中的 alias ..."

PYTHON_SH="$ISAAC_SIM_ROOT/python.sh"

# 检查是否还有旧的错误路径
if grep -q "workspace/isaaclab/_isaac_sim" "$BASHRC" 2>/dev/null; then
    echo "[FIX]   检测到旧路径，正在修复 $BASHRC ..."

    # 用 sed 原地替换旧路径为新路径
    sed -i \
        -e "s|export ISAACLAB_PATH=.*|export ISAACLAB_PATH=$ISAACLAB_SOURCE|g" \
        -e "s|alias isaaclab=.*|alias isaaclab=$ISAACLAB_SOURCE/isaaclab.sh|g" \
        -e "s|alias python=.*_isaac_sim.*|alias python=$PYTHON_SH|g" \
        -e "s|alias python3=.*_isaac_sim.*|alias python3=$PYTHON_SH|g" \
        -e "s|alias pip=.*_isaac_sim.*|alias pip='$PYTHON_SH -m pip'|g" \
        -e "s|alias pip3=.*_isaac_sim.*|alias pip3='$PYTHON_SH -m pip'|g" \
        -e "s|alias tensorboard=.*_isaac_sim.*|alias tensorboard='$PYTHON_SH $ISAAC_SIM_ROOT/tensorboard'|g" \
        "$BASHRC"

    echo "[OK]    ~/.bashrc 修复完成。"
else
    echo "[SKIP]  ~/.bashrc 中未检测到旧路径，无需修复。"
fi

# 验证修复结果
echo "[INFO]  当前 alias 配置："
grep -E "^(export ISAACLAB_PATH|alias python|alias pip)" "$BASHRC" | sed 's/^/          /'

# ------------------------------------------------------------
# Step 3: 升级 wandb（修复 np.float_ 与 NumPy 2.0 的兼容性问题）
# ------------------------------------------------------------
echo ""
echo "[3/3] 检查 wandb 版本兼容性 ..."

# 用绝对路径调用，避免 alias 未生效的问题
WANDB_OK=$("$PYTHON_SH" -c "import wandb; print('ok')" 2>/dev/null || echo "fail")

if [[ "$WANDB_OK" == "ok" ]]; then
    WANDB_VER=$("$PYTHON_SH" -c "import wandb; print(wandb.__version__)" 2>/dev/null)
    echo "[SKIP]  wandb $WANDB_VER 可正常 import，无需升级。"
else
    echo "[FIX]   wandb import 失败（可能与 NumPy 2.0 不兼容），正在升级 ..."
    "$PYTHON_SH" -m pip install --upgrade wandb 2>&1 | grep -E "(Successfully|already|ERROR)" || true

    WANDB_VER=$("$PYTHON_SH" -c "import wandb; print(wandb.__version__)" 2>/dev/null || echo "unknown")
    echo "[OK]    wandb 升级完成，版本: $WANDB_VER"
fi

# ------------------------------------------------------------
# 完成
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "[DONE]  环境初始化完成！"
echo "        请执行以下命令使 alias 在当前终端生效："
echo ""
echo "            source ~/.bashrc"
echo ""
echo "        或直接重新开一个终端。"
echo "============================================================"
