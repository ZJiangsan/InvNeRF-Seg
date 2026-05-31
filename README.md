# InvNeRF-Seg

This repository contains code, pretrained model weights, and dataset for the paper:

"A Two-Stage Fine-Tuning Strategy for 3D Object Segmentation from Multi-View Images"

## Dataset

We provide a self-collected soybean dataset captured using an iPhone 12 Pro under natural outdoor lighting conditions. The data consists of multi-view images extracted from video recordings.

- Frames extracted at 5 FPS
- Camera poses estimated using COLMAP
- Masks generated using MobileSAM and manually verified

Tasks:
1. Binary instance-level segmentation (single pod)
2. Multi-instance segmentation (two pods)

## Contents

- `data/soybean_sample/`: sample images and masks
- `weights/`: pretrained model weights (if available)

## Usage

Detailed instructions for reproducing experiments will be added.

## Note

Full dataset and complete training pipeline will be released.
