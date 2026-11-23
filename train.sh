#!/usr/bin/env bash
lerobot-train --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --batch_size=8 --steps=30000 --save_freq=2000 \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_aloha
