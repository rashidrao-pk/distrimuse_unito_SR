# DistriMuSe-UC3

> **University of Torino**  
> Distributed Multi-Sensor Systems for Human Safety and Health

---

# Use Case 3 — Safe Interaction with Robots

A **real-time anomaly detection pipeline** based on **VAE/VAE-GAN models** for monitoring industrial safety areas in collaborative robotics environments.

The system supports:

- Video preprocessing and masking  
- Safety-area specific model training  
- Threshold calibration  
- Live ROS-based inference  
- Rulex/ROS message publishing  

---

# Pipeline Overview

```text
Raw Video / ROS Stream
        ↓
Frame Extraction + Masking
        ↓
Safety Area Cropping / Resize
        ↓
VAE-GAN Training
        ↓
Threshold Calibration
        ↓
Live Inference / Alert Publishing
```

---

# Safety Areas

| ID | Description |
|---|---|
| `RoboArm` | Robot arm zone |
| `ConvBelt` | Conveyor belt zone |
| `PLeft` | Personnel left zone |
| `PRight` | Personnel right zone |
| `ALL` | All safety areas |

---

# Smart Robotics ROS Libraries

Used for ROS2 communication:

| Library | Purpose |
|---|---|
| `distrimuse-image-broadcaster` | Broadcast camera/image stream |
| `distrimuse-ros2-api` | Receive ROS detection messages |

Repositories:

- https://github.com/smart-robotics/distrimuse-image-broadcaster
- https://github.com/smart-robotics/distrimuse-ros2-api

---

# 1. Setup Environment

## Connect to Remote Device

```bash
ssh -X unito@distrimuse
```

---

## Install Pixi Environment

```bash
cd ~/advis/advis_distrimuse_unito_SR
pixi install
```

---

## Verify Installation

```bash
pixi run python -c "import rclpy; print('ROS OK')"
pixi run python -c "import cv2; print('CV OK')"
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

---

# 2. ROS2 Installation

<details>
<summary><strong>📦 Expand ROS2 Installation</strong></summary>

```bash
mkdir -p ~/ros2_ws/src

cd ~/ros2_ws/src

git clone https://github.com/smart-robotics/distrimuse-ros2-api.git

cd ~/ros2_ws

sudo apt install colcon

colcon build
```

</details>

---

# 3. Broadcast Image Stream

<details>
<summary><strong>📡 Expand Broadcast Commands</strong></summary>

```bash
cd ~/advis/distrimuse-image-broadcaster

export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### Verify Bag

```bash
pixi run ros2 bag info /home/unito/advis/bags/recording_20260313_133316
```

### Replay Bag

```bash
pixi run replay /home/unito/advis/bags/recording_20260313_133316/ \
--no-display \
--loop
```

</details>

---

# 4. Save Frames from ROS Broadcast - If Needed for Training Again

<details>
<summary><strong>🖼 Expand Save Frame Commands</strong></summary>

1. Verify Sensor Broadcast
    ```bash
    cd ~/advis/distrimuse-image-broadcaster
    pixi run ros2 topic list | grep camera
    pixi run ros2 topic hz /camera/back_view/image_raw
    ```
2. Save Frames
    ```bash
    pixi run python scripts/pixi/pixi_flow.py \
    --ros-args \
    -p save_dir:=/home/unito/advis/DS/SR/v3/train_processed/back_view \
    -p camera_topic:=/camera/back_view/image_raw \
    -p area_names:="['ConvBelt','PLeft','PRight','RoboArm']" \
    -p static_mask_paths:="[
    '/home/unito/advis/DS/SR/v2/masks/Mask Generation_ConvBelt_MASK.png',
    '/home/unito/advis/DS/SR/v2/masks/Mask Generation_PLeft_MASK.png',
    '/home/unito/advis/DS/SR/v2/masks/Mask Generation_PRight_MASK.png',
    '/home/unito/advis/DS/SR/v2/masks/Mask Generation_RoboArm_MASK.png'
    ]" \
    -p save_every_n:=5 \
    -p image_format:=png
    ```

</details>

---

# 5. Training Models

---

## Flush Existing Models

```bash
pixi run python scripts/flush_data.py \
--safety_area ALL \
--latent_dims 64 \
--dry_run
```

---

## Train on Old Dataset (v2)

```bash
python scripts/train.py \
--safety_area ALL \
--dataset_source SR \
--dataset_version v2 \
--dataset_cam_type refined \
--epochs 200 \
--batch_size 16 \
--latent_dims 64 \
--augmentation_type custom \
--save_figures
```

---

## Train on New ROS Dataset (v3)

```bash
python scripts/train.py \
--safety_area ALL \
--dataset_source SR \
--dataset_version v3 \
--dataset_cam_type back_view \
--epochs 200 \
--batch_size 16 \
--latent_dims 64 \
--augmentation_type custom
```

---

# 6. Threshold Calibration

---

## Validation Threshold Calibration

```bash
python scripts/calibrate_threshold.py \
--mode val \
--safety_area ALL \
--dataset_version v2 \
--dataset_type refined
```

---

## Test Threshold Calibration

```bash
python scripts/calibrate_threshold.py \
--mode test \
--safety_area ALL \
--gt_csv scripts/data/annotations.csv
```

---

# 7. Live ROS Inference

---

## Verify Topics

```bash
pixi run ros2 topic list | grep camera
pixi run ros2 topic hz /camera/back_view/image_raw
```

---

## Run Live Inference

<details>
<summary><strong>🚀 Expand Inference Command</strong></summary>

```bash
pixi run python scripts/infer_ros_live.py \
--camera_topic /camera/back_view/image_raw \
--safety_area ALL \
--area_names RoboArm ConvBelt PLeft PRight \
--static_mask_paths \
/home/unito/advis/DS/SR/v3/masks/Mask\ Generation_RoboArm_MASK.png \
/home/unito/advis/DS/SR/v3/masks/Mask\ Generation_ConvBelt_MASK.png \
/home/unito/advis/DS/SR/v3/masks/Mask\ Generation_PLeft_MASK.png \
/home/unito/advis/DS/SR/v3/masks/Mask\ Generation_PRight_MASK.png \
--threshold_dir scripts/results/thresholds \
--checkpoints scripts/results/models \
--latent_dims 64 \
--frame_stride 1
```

</details>

---

# 8. GUI Inference

```bash
pixi run python scripts/infer_ros_live_GUI.py \
--camera_topic /camera/back_view/image_raw \
--show_timeline
```

---

# 9. Publish Detection Response to Rulex

```bash
pixi run python scripts/infer_ros_live_MSG.py \
--publish_rulex \
--rulex_topic /rulex/detection_result
```

---

# Repository Structure

```text
advis_distrimuse_unito_SR/
│
├── scripts/
│   ├── train.py
│   ├── calibrate_threshold.py
│   ├── inference.py
│   ├── infer_ros_live.py
│   └── results/
│       ├── models/
│       ├── thresholds/
│       └── training/
│
├── pixi.toml
└── README.md
```

---

# Acknowledgements

Developed at the **University of Torino** under the  
**DistriMuSe Project**:

https://distrimuse.eu/

Focused on distributed multi-sensor systems for:

- Human safety  
- Collaborative robotics  
- Industrial anomaly detection  

---