<div align="center">

# ACT on ALOHA Transfer Cube

**Imitation learning with an Action Chunking Transformer, trained and evaluated in MuJoCo**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.8-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.8.1-0A66C2?style=for-the-badge)
![LeRobot](https://img.shields.io/badge/LeRobot-0.7.0-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gymnasium](https://img.shields.io/badge/gym--aloha-0.1.5-00A3A3?style=for-the-badge)
![NVIDIA](https://img.shields.io/badge/CUDA-NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-training-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-evaluation-FCC624?style=for-the-badge&logo=linux&logoColor=black)

![Success rate](https://img.shields.io/badge/success_rate-96%25_(48%2F50)-2EA44F?style=flat-square)
![Episodes](https://img.shields.io/badge/eval_episodes-50-blue?style=flat-square)
![Params](https://img.shields.io/badge/parameters-52M-lightgrey?style=flat-square)

</div>

---
## Summary

**Task.** A simulated robot with two arms picks up a red block from a table with one arm and passes it to the other arm, which holds it in the air. This is the ALOHA "transfer cube" task in MuJoCo.

**Model.** I trained an Action Chunking Transformer (ACT). At each step it reads the camera image and the joint positions, then predicts the next 100 movements in one pass. Predicting in chunks gives smoother motion than choosing one action at a time.

**Training.** The model learned by imitation from 50 demonstrations recorded by a scripted expert. Training ran for 30,000 steps, about 2.5 hours on one Google Colab's T4 GPU.

## Demo

### Baseline (full 100-step chunk)

https://github.com/user-attachments/assets/a08833e7-b125-4869-b30b-8c794e9df289






## Results

30k-step checkpoint, 50 evaluation episodes per configuration on `AlohaTransferCube-v0` (400 steps max, eval seed 1000). Intervals are 95% Wilson.

| Configuration | Success | 95% CI |
|:---|:---:|:---:|
| Baseline (`n_action_steps=100`, full chunk executed) | **96%** (48/50) | 86.5-98.9% |
| Re-plan every 10 steps (`n_action_steps=10`) | 44% (22/50) | 31.2-57.7% |
| Temporal ensembling (`temporal_ensemble_coeff=0.01`) | **98%** (49/50) | 89.5-99.6% |

- Re-planning every 10 steps without ensembling lowers success substantially; the intervals do not overlap.
- Temporal ensembling is statistically indistinguishable from the baseline (overlapping intervals).
- All three rows use the same trained checkpoint. Only inference settings differ; nothing was retrained per setting.

### Failures

https://github.com/user-attachments/assets/65ebe820-a09d-4a4c-8754-42eb3a9de270


---

## Setup

| | |
|:---|:---|
| **Task** | `AlohaTransferCube-v0` (gym-aloha, MuJoCo, simulation) |
| **Data** | `lerobot/aloha_sim_transfer_cube_scripted`: 50 scripted demonstrations, 20,000 frames |
| **Policy** | ACT, ResNet18 backbone, 52M parameters |
| **Observations** | top camera (3x480x640) + 14-dim joint state |
| **Actions** | 14-dim joint targets, chunk size 100 |
| **Training** | 30k steps, batch size 8, lr 1e-5, KL weight 10, seed 1000 |
| **Training hardware** | Google Colab GPU |
| **Evaluation** | local Linux workstation, NVIDIA GPU, EGL rendering |

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "lerobot[aloha]"
sudo apt install -y libegl1 libgl1 libosmesa6
export MUJOCO_GL=egl    # use osmesa if EGL is unavailable
```

**Train**

```bash
lerobot-train --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --batch_size=8 --steps=30000 --save_freq=2000 \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_aloha
```

**Evaluate (50 episodes)**

```bash
E="--env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=50 --eval.batch_size=1"
P=outputs/act_aloha/checkpoints/last/pretrained_model

# baseline
lerobot-eval --policy.path=$P $E --output_dir=results/eval_50ep
# re-plan every 10 steps
lerobot-eval --policy.path=$P --policy.n_action_steps=10 $E --output_dir=results/eval_nact10
# temporal ensembling
lerobot-eval --policy.path=$P --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 $E --output_dir=results/eval_tempens
```

Evaluation needs the full `pretrained_model` folder (weights, config, and the pre/post-processor files holding the normalization statistics), not just `model.safetensors`.

## Repository layout

```
README.md
train.sh          # training command + train_config.json
eval.sh           # evaluation commands
results/          # eval logs and summaries
media/            # rollout videos
```

## Acknowledgements

ACT: Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (2023). Implementation and dataset from [LeRobot](https://github.com/huggingface/lerobot) and `gym-aloha`.

