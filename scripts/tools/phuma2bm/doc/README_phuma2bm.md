# phuma2bm usage

将 [PHUMA](https://github.com/DAVIAN-Robotics/PHUMA) 发布的 G1 字典型 `*.npy`（`root_trans`, `root_ori` xyzw, `dof_pos`, `fps`）转为与本仓库 **`soma2bm`** 相同键名的 BM `motion.npz`，供 `MotionLoader` / `train.py` / `replay_npz.py` 使用。

实现要点：复用 `scripts/tools/soma2bm/soma2bm_lib.py` 中的重采样与 **MuJoCo → IsaacLab** 关节置换；根四元数 **xyzw → wxyz**；仅填充 `body_*[:,0,...]`（与 soma2bm 离线策略一致）。

## 1) 生成 manifest（可选按 split 列表过滤）

```bash
python scripts/tools/phuma2bm/build_phuma_npy_manifest.py \
  --phuma_g1_root /home/qiulm/Dataset/PHUMA/data/g1 \
  --output_dir /tmp/phuma2bm_manifest
```

可选 `--split_file my_list.txt`：每行一条相对 `g1` 根的路径，如 `humanml/000000_chunk_0000.npy`。空文件或仅注释则视为**不启用过滤**（扫描整棵树）。

## 2) 批量转换

```bash
python scripts/tools/phuma2bm/batch_convert_phuma_to_bm_npz.py \
  --phuma_g1_root /home/qiulm/Dataset/PHUMA/data/g1 \
  --metadata_manifest /tmp/phuma2bm_manifest/filtered_phuma_npy_manifest.json \
  --output_root /tmp/phuma2bm_npz \
  --output_fps 50 \
  --num_bodies 30
```

- 默认使用每个 npy 内的 `fps` 作为输入帧率；若需统一覆盖，加 `--input_fps 30`。
- 若确认 `dof_pos` 已是 Isaac 关节顺序：`--assume_isaac_joint_order`。
- 调试：`--limit 100`、`--skip_existing`。

## 3) 单文件转换

```bash
python scripts/tools/phuma2bm/convert_phuma_npy_to_bm_npz.py \
  --input_npy /home/qiulm/Dataset/PHUMA/data/g1/humanml/000000_chunk_0000.npy \
  --output_npz /tmp/phuma_one.npz
```

## 4) 注册到 W&B（与 soma2bm 相同脚本）

```bash
python scripts/tools/soma2bm/register_motion_npz_to_wandb.py \
  --project csv_to_npz \
  --artifact_name phuma-g1-bm \
  --artifact_type motions \
  --input_dir /tmp/phuma2bm_npz \
  --metadata_manifest /tmp/phuma2bm_manifest/filtered_phuma_npy_manifest.json \
  --alias latest
```

## 5) 训练

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /tmp/phuma_one.npz \
  --logger wandb
```

## 文档

- [PHUMA_信息蒸馏与BM适配.md](./PHUMA_信息蒸馏与BM适配.md)：数据格式与 BM 字段对照说明。
