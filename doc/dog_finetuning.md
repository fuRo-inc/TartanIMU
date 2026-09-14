# TartanIMU Go2 Fine-tuning

このドキュメントでは、Go2で収集・整形したIMUデータを使って、TartanIMUの事前学習済みモデルをfine-tuningする手順をまとめます。

## 1. 前提

TartanIMU repository:

```text
~/fuk_ws/TartanIMU
```

Go2用の整形済みTartanIMU dataset:

```text
/home/furo-inc/fuk_ws/unitree_rl_mjlab_test/furo_rl_locomotion_mjlab/locomotion_dataset/dataset/collection_locomotion_oshima_19500/tartan_imu
```

datasetは以下の構造を想定します。

```text
tartan_imu/
├── train/
│   ├── *.npz
│   └── ...
├── val/
│   ├── *.npz
│   └── ...
└── test/
    ├── *.npz
    └── ...
```

各 `.npz` は以下の4つのkeyを持ちます。

```text
retargetted_ts
retargetted_imu
retargetted_pos
retargetted_quat
```

想定フォーマット:

```text
retargetted_ts   : (N,)
retargetted_imu  : (N, 6)
retargetted_pos  : (N, 3)
retargetted_quat : (N, 4)
```

`retargetted_quat` は `xyzw` 順です。

IMUは200 Hzで保存します。

```text
dt = 0.005 s
imu_freq = 200 Hz
```

---

# 2. Fine-tuningの方針

最初のbaselineでは、TartanIMU本来の出力だけをfine-tuningします。

入力:

```text
6-axis IMU
├── gyroscope xyz
└── accelerometer xyz
```

出力:

```text
body-frame linear velocity
[vx, vy, vz]
```

今回は追加しません。

```text
Δposition
Δrotation
covariance
Legolas-style auxiliary output
```

これらはbaseline性能を確認した後に追加します。

転倒・衝突区間についても、最初のfine-tuningでは削除せず使用します。

---

# 3. Pretrained checkpoint

事前学習済みcheckpoint:

```text
./tartan_imu_weights/checkpoints/unified.pt
```

ただし、`unified.pt` には以下が含まれている場合があります。

```text
model_state_dict
optimizer_state_dict
epoch
scheduler_state_dict
scaler_state_dict
trainer_state
```

TartanIMUの現在のtraining codeでは、これをそのまま `--resume_from` に渡すと、モデルだけでなくoptimizer等も復元しようとします。

Go2へのfine-tuningでは、

```text
pretrained model weights : 使用する
optimizer state          : 使用しない
scheduler state          : 使用しない
epoch                    : 使用しない
AMP scaler               : 使用しない
trainer state            : 使用しない
```

とします。

そのため、weights-only checkpointを作成します。

---

# 4. Weights-only checkpoint作成スクリプト

以下のファイルを作成します。

```text
scripts/make_weights_only_checkpoint.py
```

内容:

```python
#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a weights-only PyTorch checkpoint for fine-tuning. "
            "Optimizer, scheduler, epoch, AMP scaler, and trainer state are removed."
        )
    )

    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source checkpoint path.",
    )

    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help=(
            "Output checkpoint path. "
            "If omitted, '<stem>_weights_only<suffix>' is used."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    src = args.src.expanduser().resolve()

    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    if args.dst is None:
        dst = src.with_name(
            f"{src.stem}_weights_only{src.suffix}"
        )
    else:
        dst = args.dst.expanduser().resolve()

    print(f"source: {src}")
    print(f"output: {dst}")

    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=False,
    )

    print("\noriginal checkpoint:")

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: "
            f"{type(checkpoint).__name__}"
        )

    for key in checkpoint.keys():
        print(f"  {key}")

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        state_dict = checkpoint
        print(
            "\n'model_state_dict' not found; "
            "treating checkpoint as state_dict."
        )

    output_checkpoint = {
        "model_state_dict": state_dict,
    }

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output_checkpoint,
        dst,
    )

    print("\nweights-only checkpoint created.")
    print(f"saved: {dst}")

    print("\nkept:")
    print("  model_state_dict")

    print("\nremoved if present:")
    print("  epoch")
    print("  optimizer_state_dict")
    print("  scheduler_state_dict")
    print("  scaler_state_dict")
    print("  trainer_state")


if __name__ == "__main__":
    main()
```

実行:

```bash
cd ~/fuk_ws/TartanIMU

python scripts/make_weights_only_checkpoint.py \
  --src ./tartan_imu_weights/checkpoints/unified.pt
```

生成物:

```text
./tartan_imu_weights/checkpoints/unified_weights_only.pt
```

確認:

```bash
ls -lh ./tartan_imu_weights/checkpoints/
```

---

# 5. Go2 Fine-tuning config

以下を作成します。

```text
config/datasets/tartanimu/go2_finetune.yaml
```

内容:

```yaml
port: 11570

seeds:
  use_seeds: True
  id: 42


# ============================================================
# Data
# ============================================================

data:
  dataset: AirLab

  data_path:
    car: null
    drone: null

    dog: /home/furo-inc/fuk_ws/unitree_rl_mjlab_test/furo_rl_locomotion_mjlab/locomotion_dataset/dataset/collection_locomotion_oshima_19500/tartan_imu

    human: null

  use_local_coord: True

  train_dir: train
  validation_dir: val
  test_dir: test

  random_partition: False

  data_rate: 1.0

  train_rate: 0.7
  valid_rate: 0.15
  test_rate: 0.15

  imu_freq: 200.0
  sample_freq: 40

  gravity_alignment:
    enabled: False
    magnitude: 9.8101
    direction:
      - 0
      - 0
      - -1

    enable_validation: False
    visualize: False


# ============================================================
# Augmentation
# ============================================================

augment:
  feat_acc_sigma: 0.0001
  feat_gyr_sigma: 0.00001

  add_bias_noise: True

  accel_bias_range: 0.1
  gyro_bias_range: 0.002

  add_gravity_noise: True
  gravity_noise_theta_range: 5


# ============================================================
# Model
# ============================================================

model:
  model_name: Foundation_Model

  model_yaml: "./config/resnet_lstm_multihead.yaml"

  pred_velocity: True

  adapter: False


# ============================================================
# Training
# ============================================================

train:
  experiment_name: go2_tartanimu_finetune_v1

  out_dir: ./exp_result/go2_tartanimu_finetune_v1

  use_pretrain_model: True

  use_multi_gpu: False

  use_amp: True

  # Smoke test
  batch_size: 128
  epochs: 2

  # Covariance training is disabled for the initial baseline.
  start_cov_epochs: 9999

  n_workers: 4

  seq_len: 10

  predict_start: 0
  predict_end: 10

  add_noise: True

  active_heads:
    - dog

  optimizer:
    learning_rate: 0.00001
    method: Adam
    weight_decay: 0.01

  scheduler:
    factor: 0.1
    patience: 8


# ============================================================
# Validation
# ============================================================

val:
  batch_size: 128
  n_workers: 2


# ============================================================
# Test
# ============================================================

test:
  out_dir: ./exp_result/go2_tartanimu_finetune_v1
  batch_size: 128
  n_workers: 2


# ============================================================
# Scheme
# ============================================================

schemes:
  train: True
  test: False
  online_adaption: False
```

---

# 6. Smoke test

最初は2 epochだけ実行します。

```bash
cd ~/fuk_ws/TartanIMU

WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=0 \
python main_net.py \
  --config ./config/datasets/tartanimu/go2_finetune.yaml \
  --resume_from ./tartan_imu_weights/checkpoints/unified_weights_only.pt
```

正常な場合、checkpoint読み込み時には概ね以下の状態になります。

```text
Starting Epoch:
0

Model Weights:
Loaded

Optimizer State:
Fresh optimizer
```

以下になっている場合は注意します。

```text
Starting Epoch:
200

Optimizer State:
Loaded from checkpoint
```

これはfine-tuningではなく、元checkpointのtraining stateをresumeしている可能性があります。

---

# 7. Dataset読み込み確認

現在のGo2 datasetでは、確認時点で以下のtrajectory数が読み込まれています。

```text
train : 2934
val   : 603
test  : 615
```

学習開始ログで以下のpathになっていることを確認します。

```text
dog/train
dog/val
dog/test
```

---

# 8. IMU入力について

元データは200 Hzです。

```text
200 samples / sec
dt = 0.005 sec
```

TartanIMUでは、

```yaml
imu_freq: 200
sample_freq: 40
window_time: 1.0
```

として使用します。

概念的には、

```text
200 Hz IMU
    ↓
1秒window
    ↓
200 raw samples
    ↓
5 sampleごとのstep
    ↓
40 Hz相当
    ↓
TartanIMU
```

となります。

---

# 9. Ground-truth velocity

`.npz` にはvelocityを直接保存していません。

以下からTartanIMU loader側で計算されます。

```text
retargetted_ts
retargetted_pos
retargetted_quat
```

流れ:

```text
position + quaternion
        ↓
calculate_velocity_from_poses()
        ↓
velocity_global
velocity_body
        ↓
1秒window平均
        ↓
GT body velocity
[vx, vy, vz]
```

`use_local_coord: True` のため、fine-tuning targetはbody-frame linear velocityです。

今後、これがGo2の

```text
base frame velocity
```

なのか、

```text
physical IMU frame velocity
```

なのかは、`prepare_dataset.py` が `retargetted_pos / retargetted_quat` にどのframeのposeを保存しているかを確認して厳密に整理します。

---

# 10. 転倒データについて

今回のdatasetには転倒・衝突trajectoryも含めます。

確認例:

```text
t=6.0 s
large acceleration impact

t=6.5 s
roll approximately 80 deg

t=6.6 s
roll approximately 96 deg

t=10 s
roll approximately 86 deg
```

このようなデータも今回は削除しません。

理由は、通常運用では転倒を避ける前提であっても、実機では転倒・衝突が起こる可能性があり、その際にもvelocity estimatorが完全に未知の入力を受ける状態を避けたいからです。

ただし、今後の評価では可能であれば、

```text
normal locomotion
fall / collision
post-fall
```

を分離して性能を確認します。

---

# 11. 本学習

2 epochのsmoke testが正常に通った後、

```yaml
train:
  batch_size: 512
  epochs: 30
```

などへ変更して本学習を開始します。

learning rateはfine-tuning用として、

```yaml
learning_rate: 0.00001
```

から開始します。

---

# 12. 最初に確認する評価指標

最初のbaselineではbody velocity predictionを評価します。

主に以下を確認します。

```text
vx MAE
vy MAE
vz MAE

vx RMSE
vy RMSE
vz RMSE

3D velocity error
```

さらにvelocityを積分してodometryとして評価する場合は、

```text
ATE
RPE
```

も確認します。

将来的には、

```text
normal locomotion
fall / collision
```

それぞれで分けて評価します。

---

# 13. 今後の拡張

最初のvelocity-only fine-tuningでbaselineを取得した後、以下を順次検討します。

```text
1. Velocity only

2. Velocity + covariance

3. Velocity + Δposition

4. Velocity + Δposition + Δrotation

5. Relative pose + covariance
```

Legolasのようなrelative pose outputを追加する場合は、

```text
Δposition : 3
Δrotation : rot6d 6
```

を補助headとして追加する構成を検討します。

最終的には、

```text
velocity
relative pose
uncertainty / covariance
```

を組み合わせたlegged odometry向けモデルへ拡張することを想定します。
