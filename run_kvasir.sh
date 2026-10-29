#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -u train_polyp.py \
  --dataset_name Kvasir \
  --train_path ./data/polyp/target/Kvasir/train \
  --test_path ./data/polyp/target/Kvasir \
  --pretrained_dir ./pretrained_pth/pvt \
  --encoder pvt_v2_b0 \
  --batchsize 16 --test_batchsize 16 \
  --lr 0.0005 --epoch 200 --runs 5 --seed_base 2222 \
  --wandb_project EMCAD-polyp --wandb_mode online \
  "$@"
