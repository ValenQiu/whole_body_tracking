# PHUMA 信息蒸馏与 whole_body_tracking（BM）适配说明

> 文档位置：**`scripts/tools/phuma2bm/doc/`**（由 `scripts/tools/soma2bm/doc/` 迁入）。

本文汇总 [PHUMA 代码仓库](https://github.com/DAVIAN-Robotics/PHUMA)、[Hugging Face 数据集页](https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA)、论文 [arXiv:2510.26236](https://arxiv.org/pdf/2510.26236) 的公开描述，并结合本机数据 **`/home/qiulm/Dataset/PHUMA/data/g1`** 抽样校验结果；最后按本仓库 **`scripts/tools/soma2bm`** 既有流程，说明 **PHUMA → BM `motion.npz`** 的等价步骤与实现约定。

**命令速查**：见同目录上一级 [`README_phuma2bm.md`](../README_phuma2bm.md)。

---

## 1. PHUMA 是什么（目标与贡献）

- **全称**：PHUMA: Physically-Grounded Humanoid Locomotion Dataset（物理约束的人形运动数据集）。
- **机构**：DAVIAN Robotics, KAIST AI（论文与仓库署名一致）。
- **核心思路**（仓库 README 与论文摘要一致表述）：
  - 利用大规模人体运动数据；
  - 通过 **physics-aware curation** 筛除物理上不合理片段；
  - 再通过 **PhySINK**（physics-constrained retargeting）将人体运动重定向到人形机器人，并在优化中强调物理可信性。
- **机器人形态**：公开数据包含 **Unitree G1** 与 **H1-2**（HF 卡与仓库目录均为 `g1/`、`h1_2/`）；代码还支持从自定义 URDF/XML 生成配置后重定向。

---

## 2. 代码与数据发布渠道

| 资源 | URL / 路径 | 说明 |
|------|------------|------|
| 源码 | https://github.com/DAVIAN-Robotics/PHUMA | Python 管线：Motion-X 预处理 → SMPL-X curation → PhySINK 重定向；含 `setup_phuma.sh` 一键拉取预构建数据 |
| 数据集 | https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA | 压缩包形式；卡内说明解压后得到 `data/g1`、`data/h1_2`；许可证 **Apache-2.0** |
| 论文 PDF | https://arxiv.org/pdf/2510.26236 | 方法、实验与数据细节以论文为准 |
| 本机 G1 数据 | `/home/qiulm/Dataset/PHUMA/data/g1` | 已下载的 retarget 后 `.npy` 片段（见 §5） |

**许可**：数据集卡标注 Apache-2.0；使用论文与数据请按仓库给出的 BibTeX 引用（HF 卡与 GitHub README 一致）。

**已知发布限制**（GitHub FAQ）：因许可问题，**PHUMA Train/Test 对应的人体 SMPL-X pose 源文件**无法公开；`unseen_video` 相关 SMPL 人体 pose 计划后续发布。使用者主要接触的是 **已重定向到机器人的 `.npy`**。

---

## 3. 官方数据处理管线（仓库 README 摘要）

下列步骤仅帮助理解 PHUMA **从何而来**，BM 训练通常只需消费 **最终 retarget 输出**。

1. **人体 pose 预处理**  
   - Motion-X 等来源可为 `(N,322)` 的 `.npy` 或 stageii `.npz`；脚本可转 Y-up→Z-up、降采样等。  
   - 再提取为 PHUMA 所需的 **`(N,69)`**（`transl`, `global_orient`, `body_pose`，不含脸/手细节）。

2. **Physics-aware curation**  
   - 依赖 SMPL-X 模型资产（需自行注册下载放入 `asset/human_model/smplx/`）。  
   - 可调 `foot_contact_threshold` 等阈值：默认在「保留跳跃腾空」与「去浮空/穿模」之间折中，故数据集中仍可能出现 **轻微穿模/浮空**（FAQ 说明）。

3. **PhySINK 重定向到 G1**  
   - **Shape adaptation**（一次性）：得到 `asset/humanoid_model/g1/betas.npy` 等。  
   - **Motion adaptation**：单文件或 `motion_adaptation_multiprocess.py` 批量；输出即 §4 所述字典 `.npy`。

4. **评估与跟踪**  
   - README 指向 **MaskedMimic** 代码库做 motion tracking / path following；与 **本仓库 IsaacLab BM `motion.npz`** 不是同一条工具链，但数据语义相近（根位姿 + 关节角轨迹）。

---

## 4. Hugging Face 卡上的「标准」字段定义

HF 数据集说明中每个 `.npy` 为 **字典**（`np.load(..., allow_pickle=True).item()`），字段为：

| 字段 | 形状 | 含义 |
|------|------|------|
| `root_trans` | `(T, 3)` | 根平移 `(x, y, z)` |
| `root_ori` | `(T, 4)` | 根朝向四元数 **`(x, y, z, w)`** |
| `dof_pos` | `(T, num_dof)` | 关节位置；G1 为 **29** |
| `fps` | 标量 | 帧率 |

---

## 5. 本机 `data/g1` 实测（与 HF 定义一致）

- **目录结构**：按数据来源分子目录（如 `aist/`, `dance/`, `humanml/`, `idea400/` …），其下为大量 `*.npy` 片段文件名（多含 `clip` / `chunk` 语义）。
- **规模（统计）**：在 `/home/qiulm/Dataset/PHUMA/data/g1` 下约 **73,076** 个 `.npy` 文件（`find … -name '*.npy' | wc -l`，随下载版本可能变化）。
- **抽样文件**（示例）：`…/idea400/Wearing_Glasses_while_walking_clip2_chunk_0001.npy`  
  - `root_trans`: `(64, 3)`, `float64`，量级符合 **米**（非厘米）。  
  - `root_ori`: `(64, 4)`, `float64`，单位范数 1；按 **`(x,y,z,w)`** 代入 `scipy.spatial.transform.Rotation.from_quat` 得到合理欧拉角，与 HF 说明一致。  
  - `dof_pos`: `(64, 29)`, `float64`，量级为 **弧度**（非度）。  
  - `fps`: Python `int`，抽样为 **30**。

**`humanml/` 与 AMASS CMU 文件名（工程备注）**：该目录下常见 `XXXXXX_chunk_YYYY.npy`。对 **AMASS CMU** 子集 `CMU/{S}/{S}_{M}_poses.npz`，有对应关系 `XXXXXX = zero_pad_6(int(S)*100 + int(M))`（例如 `132_43` → `013243`）。顶层目录不会出现字面量 `amass` 或 `CMU/132/...`。

---

## 6. whole_body_tracking（BM）侧 `motion.npz` 要求

`MotionLoader`（`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`）期望键与形状要点：

| 键 | 说明 |
|----|------|
| `fps` | 输出帧率（标量或 `(1,)`） |
| `joint_pos` | `(T, N_joint)`，与仿真 **`robot.data.joint_pos` 同索引顺序**（IsaacLab / URDF 展开顺序） |
| `joint_vel` | 同上 |
| `body_pos_w` | `(T, N_bodies, 3)` 世界系位置 |
| `body_quat_w` | `(T, N_bodies, 4)`，**`w, x, y, z`** |
| `body_lin_vel_w` / `body_ang_vel_w` | 世界系线速度、角速度 |

**与 `soma2bm` 一致的重要约定**（见 `scripts/tools/soma2bm/soma2bm_lib.py` 与 `scripts/tools/soma2bm/doc/GR00T_SONIC_G1_数据到训练链路蒸馏.md` §9）：

- Bones flat CSV 的 29 维关节默认视为 **MuJoCo 列顺序**，写入 BM 前需按 **`MJ_TO_IL`** 置换为 IsaacLab 顺序。  
- PHUMA 由 **MuJoCo + PhySINK** 管线产出，**默认** `dof_pos` 为 **MuJoCo 顺序**；`phuma2bm_lib` 通过 **import 复用** `soma2bm_lib` 中同一套置换，避免两套映射漂移。

**四元数**：PHUMA 根为 **xyzw**；BM `body_quat_w` 为 **wxyz**，转换时需分量重排。

**根线速度 / 角速度 / 关节速度**：PHUMA 文件未提供；与 `soma2bm` 一样对重采样后序列用 **固定 dt 的 `np.gradient`** 得到 `joint_vel`、`body_lin_vel_w`、`body_ang_vel_w`（根体填入 `body_*[:,0,:]`，其余体为零位姿 + 单位四元数）。

**帧率**：PHUMA 片段多为 **30 fps**；训练常用 **50 fps**（默认 `output_fps=50`）时对 `root_trans`、`root_ori`（Slerp）、`dof_pos`（线性）重采样，并据此重算速度。

---

## 7. 对齐 `soma2bm` 流程的 PHUMA 实现（本仓库）

| soma2bm 步骤 | PHUMA 实现 |
|--------------|------------|
| 1) `build_filtered_csv_manifest.py` | **`build_phuma_npy_manifest.py`**：`--phuma_g1_root` + 可选 `--split_file` → `filtered_phuma_npy_manifest.json` |
| 2) `batch_convert_soma_bones_to_bm_npz.py` | **`batch_convert_phuma_to_bm_npz.py`**：manifest + 根路径 → 目录树镜像输出 `.npz` |
| 单条调试 | **`convert_phuma_npy_to_bm_npz.py`** |
| 3) `register_motion_npz_to_wandb.py` | **复用** `scripts/tools/soma2bm/register_motion_npz_to_wandb.py` |
| 4) `train.py` | 不变：`--motion_file` 指向生成的 `.npz` |

核心逻辑：**`phuma2bm_lib.py`**（运行时 `importlib` 加载 `soma2bm_lib.py` 以复用重采样与关节置换）。

---

## 8. 实现适配器时的校验清单（建议顺序）

1. **单条轨迹可视化**：用 `scripts/replay_npz.py` 播放 1～2 个转换后 npz，观察是否「拧关节」或整体漂移。  
2. **顺序对照**：任选一条 PHUMA 轨迹，与 `scripts/csv_to_npz.py` 从同语义 CSV（若有）或已知 MuJoCo 顺序参考对比第 0 帧各关节符号。  
3. **四元数**：确认仅做一次 **xyzw→wxyz**，且与 `joint_pos` 在同一根坐标约定下。  
4. **fps**：统计各文件 `fps` 是否恒为 30；若存在多 fps，可用 `--input_fps` 统一覆盖。  
5. **body 子集奖励**：若训练配置使用多 body 误差，仅有根+关节占位 npz 可能与「全 FK body」金标准有差异（`GR00T_SONIC_G1_…` §9 已说明）；需要时再走仿真 FK 补全 `body_*`。

---

## 9. 引用（BibTeX，与官方一致）

```bibtex
@article{lee2025phuma,
  title={PHUMA: Physically-Grounded Humanoid Locomotion Dataset},
  author={Kyungmin Lee and Sibeen Kim and Minho Park and Hyunseung Kim and Dongyoon Hwang and Hojoon Lee and Jaegul Choo},
  journal={arXiv preprint arXiv:2510.26236},
  year={2025},
}
```

---

## 10. 实现状态

`scripts/tools/phuma2bm/` 已提供与 §7 对应的脚本与 `phuma2bm_lib.py`；**H1-2** 需另写（当前仅校验 G1 的 29 DoF）。
