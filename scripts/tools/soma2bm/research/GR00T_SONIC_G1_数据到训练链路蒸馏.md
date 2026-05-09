# GR00T-WholeBodyControl（SONIC）G1 训练数据链路蒸馏

> 目标：聚焦 `NVlabs/GR00T-WholeBodyControl` 中 SONIC 训练部分，梳理 **G1 数据格式如何封装**，以及 **关节 / link 顺序与 CSV 列数对应关系**，形成从数据到训练的可执行参考。

## 1. 结论先看（TL;DR）

- SONIC 训练主入口使用 `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` 将 Bones-SEED G1 CSV 或 SOMA 导出 CSV/PKL 统一转成 `motion_lib` PKL。
- 训练真正吃的数据字段是：`root_trans_offset`、`pose_aa`、`dof`、`root_rot`、`fps`（外加 `smpl_joints` 占位）。
- **Bones-SEED flat CSV（单文件）** 关键是 36 列：
  - `Frame` 1 列
  - 根位姿 6 列（`root_translateXYZ` + `root_rotateXYZ`）
  - G1 关节 29 列（`*_dof`，MuJoCo/MJCF 顺序）
- `motion.yaml` 里训练跟踪 body 子集是 **14 个 link**，对应：
  - `body_pos.csv` 每帧 42 列（14×3）
  - `body_quat.csv` 每帧 56 列（14×4）
  - 对应 IsaacLab body index 为 `[0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]`

---

## 2. 数据 -> 训练总体链路

GR00T 文档给出的 SONIC 常规流程：

1. 下载 Bones-SEED（G1 CSV，120fps）
2. 执行转换脚本：
   - `convert_soma_csv_to_motion_lib.py`
   - 常见参数：`--fps_source 120 --fps 30 --individual`
3. 执行过滤脚本：
   - `filter_and_copy_bones_data.py`
4. 训练时在 `train_agent_trl.py` 里把 `motion_lib_cfg.motion_file` 指到过滤后的 robot PKL 目录

常见训练配置：

- `sonic_release`：G1 + teleop + SMPL（默认）
- `sonic_bones_seed`：在上面基础上再加 SOMA 编码器

---

## 3. SONIC 中 G1 数据格式如何封装

## 3.1 输入侧支持的 5 种模式（转换脚本）

`convert_soma_csv_to_motion_lib.py` 支持：

1. 单个 motion 目录（`joint_pos.csv` / `body_pos.csv` / `body_quat.csv`）
2. 多个 motion 子目录
3. deploy PKL（含 `joint_pos/body_pos_w/body_quat_w`）
4. Bones-SEED flat CSV 目录（每个动作一个 CSV）
5. session 目录批量（每个 session 下多个 flat CSV）

## 3.2 转换后的统一输出结构（motion_lib PKL）

每个动作条目为：

```python
{
  "motion_name": {
    "root_trans_offset": (T, 3),   # 根平移（米）
    "pose_aa": (T, 30, 3),         # 30个body轴角（body0是pelvis，1-29是受驱link）
    "dof": (T, 29),                # G1 29DoF（MuJoCo顺序）
    "root_rot": (T, 4),            # 根四元数（xyzw，scipy约定）
    "smpl_joints": (T, 24, 3),     # 占位或真实SMPL
    "fps": int
  }
}
```

---

## 4. G1 关节顺序与 CSV 列对应（核心）

## 4.1 Bones-SEED flat CSV 列布局（单文件）

脚本注释定义格式为：

- `Frame`
- `root_translateX/Y/Z`（cm）
- `root_rotateX/Y/Z`（deg, Euler xyz）
- 29 个 `*_dof`（deg）

因此每帧总列数 = `1 + 3 + 3 + 29 = 36`。

> 注意：代码里 `BONES_CSV_JOINT_NAMES` 给出了期望顺序，但实际读取时是 `joint_cols = [c for c in data.columns if c.endswith("_dof")]`。  
> 也就是说实际依赖 CSV 原有列顺序，未显式用该常量二次重排。

### 29 个 DOF 列（MuJoCo/MJCF顺序）

1. `left_hip_pitch_joint_dof`
2. `left_hip_roll_joint_dof`
3. `left_hip_yaw_joint_dof`
4. `left_knee_joint_dof`
5. `left_ankle_pitch_joint_dof`
6. `left_ankle_roll_joint_dof`
7. `right_hip_pitch_joint_dof`
8. `right_hip_roll_joint_dof`
9. `right_hip_yaw_joint_dof`
10. `right_knee_joint_dof`
11. `right_ankle_pitch_joint_dof`
12. `right_ankle_roll_joint_dof`
13. `waist_yaw_joint_dof`
14. `waist_roll_joint_dof`
15. `waist_pitch_joint_dof`
16. `left_shoulder_pitch_joint_dof`
17. `left_shoulder_roll_joint_dof`
18. `left_shoulder_yaw_joint_dof`
19. `left_elbow_joint_dof`
20. `left_wrist_roll_joint_dof`
21. `left_wrist_pitch_joint_dof`
22. `left_wrist_yaw_joint_dof`
23. `right_shoulder_pitch_joint_dof`
24. `right_shoulder_roll_joint_dof`
25. `right_shoulder_yaw_joint_dof`
26. `right_elbow_joint_dof`
27. `right_wrist_roll_joint_dof`
28. `right_wrist_pitch_joint_dof`
29. `right_wrist_yaw_joint_dof`

## 4.2 MuJoCo -> IsaacLab DOF 映射（用于 reorder）

转换逻辑：

- 若输入是 IsaacLab 顺序（`joint_order == "il"`），按 `MJ_TO_IL` 重排到 MuJoCo
- 若输入已是 MuJoCo（Bones flat CSV），直接使用

`MJ_TO_IL`（长度29）：

`[0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]`

可直观理解为：MuJoCo 第 `i` 个 DOF 对应 IsaacLab 第 `MJ_TO_IL[i]` 个 DOF。

---

## 5. link 顺序与 body CSV 列数

## 5.1 训练 tracking body 子集（14 links）

`gear_sonic/config/manager_env/commands/terms/motion.yaml` 的 `body_names` 顺序：

1. `pelvis`
2. `left_hip_roll_link`
3. `left_knee_link`
4. `left_ankle_roll_link`
5. `right_hip_roll_link`
6. `right_knee_link`
7. `right_ankle_roll_link`
8. `torso_link`
9. `left_shoulder_roll_link`
10. `left_elbow_link`
11. `left_wrist_yaw_link`
12. `right_shoulder_roll_link`
13. `right_elbow_link`
14. `right_wrist_yaw_link`

在 `commands.py` 中，这 14 个名字进一步映射为 IsaacLab 全 body 列表索引：

`body_indexes_data = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]`

这与部署文档示例 `metadata.txt` 中的 body index 列表一致。

## 5.2 对应 CSV 列数

若 body 文件按这 14 个 link 导出：

- `body_pos.csv`：`14 * 3 = 42` 列
- `body_quat.csv`：`14 * 4 = 56` 列

列组约定：

- `body_pos.csv`：每个 body 一组 `(x,y,z)`
- `body_quat.csv`：每个 body 一组 `(w,x,y,z)`
- 根（pelvis）必须是第 0 组（首组列）

---

## 6. 与 `joint_pos.csv/body_pos.csv/body_quat.csv` 三件套的关系

当输入是 SOMA 风格三文件目录时：

- `joint_pos.csv`：读取为 `(T,29)`，注释约定 IsaacLab 顺序
- `body_pos.csv`：读取后 reshape 为 `(T, N, 3)`
- `body_quat.csv`：读取后 reshape 为 `(T, N, 4)`（wxyz）

转换时仅强依赖：

- `body_pos_w[:,0,:]` 作为根平移
- `body_quat_w[:,0,:]` 作为根旋转

对 Bones flat CSV 路径，脚本会构造 dummy `body_pos_w/body_quat_w`（14 体，除根外为零/单位四元数）用于统一接口。

---

## 7. 训练相关关键约束（避免踩坑）

1. `dof` 与 `pose_aa` 在 motion_lib 中按 MuJoCo 顺序存储；训练中再按映射转回 IsaacLab 使用。
2. Bones flat CSV 是角度（deg）+ 位移（cm），脚本会转换到 `rad/m`。
3. 下采样是 stride 方式：`jump = int(fps_source / fps_target)`，典型 120 -> 30 为步长 4。
4. `BONES_CSV_JOINT_NAMES` 常量目前未参与实际重排，CSV 列顺序必须自洽。
5. 训练里用于 tracking 的 body 默认为 14-link 子集，不是完整 30 body。

---

## 8. 快速核对清单（给后续接入/排错）

- [ ] flat CSV 是否满足 36 列结构（1+6+29）
- [ ] 29 个 `*_dof` 列顺序是否与 MuJoCo 预期一致
- [ ] 若用三件套 CSV：`joint_pos` 是否 29 列；`body_pos/body_quat` 的 body 组数是否匹配
- [ ] `metadata/body_indexes_data` 是否与 `motion.yaml body_names` 对齐（14-link 时应为 `[0,4,10,18,5,11,19,9,16,22,28,17,23,29]`）
- [ ] 训练命令中 `motion_lib_cfg.motion_file` 是否指向过滤后的 PKL 目录

---

## 9. 主要依据文件（便于追溯）

- `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`
- `gear_sonic/envs/manager_env/robots/g1.py`
- `gear_sonic/config/manager_env/commands/terms/motion.yaml`
- `gear_sonic/envs/manager_env/mdp/commands.py`
- `docs/source/user_guide/training.md`
- `docs/source/user_guide/training_data.md`
- `docs/source/references/motion_reference.md`
- `docs/source/user_guide/new_embodiments.md`
