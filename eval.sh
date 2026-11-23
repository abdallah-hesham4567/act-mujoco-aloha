#!/usr/bin/env bash
export MUJOCO_GL=egl
E="--env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=50 --eval.batch_size=1"
P=outputs/act_aloha/checkpoints/last/pretrained_model

lerobot-eval --policy.path=$P $E --output_dir=results/eval_50ep
lerobot-eval --policy.path=$P --policy.n_action_steps=10 $E --output_dir=results/eval_nact10
lerobot-eval --policy.path=$P --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 $E --output_dir=results/eval_tempens
