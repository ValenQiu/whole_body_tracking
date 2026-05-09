# soma2bm usage

## 1) Build manifest from filtered metadata

```bash
python scripts/tools/soma2bm/build_filtered_csv_manifest.py \
  --csv_root /home/qiulm/Downloads/g1_bones-seed/csv \
  --metadata_json /home/qiulm/Downloads/bones-seed_ironmotion_metadata_0.9.json \
  --output_dir /tmp/soma2bm_manifest
```

## 2) Batch convert by manifest

```bash
python scripts/tools/soma2bm/batch_convert_soma_bones_to_bm_npz.py \
  --csv_root /home/qiulm/Downloads/g1_bones-seed/csv \
  --metadata_manifest /tmp/soma2bm_manifest/filtered_csv_manifest.json \
  --output_root /tmp/soma2bm_npz \
  --input_fps 120 \
  --output_fps 50 \
  --num_bodies 30
```

## 3) Register to W&B artifact

```bash
python scripts/tools/soma2bm/register_motion_npz_to_wandb.py \
  --project csv_to_npz \
  --artifact_name g1-bones-seed-0p9 \
  --artifact_type motions \
  --input_dir /tmp/soma2bm_npz \
  --metadata_manifest /tmp/soma2bm_manifest/filtered_csv_manifest.json \
  --alias latest
```

## 4) Train from local npz

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /tmp/soma2bm_npz/230710/jog_ff_loop_225_R_001__A422.npz \
  --logger wandb
```

## 5) Train from registry

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --registry_name g1-bones-seed-0p9 \
  --logger wandb
```
