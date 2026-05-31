#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:48:56 2026

@author: nibio
"""





import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 18

def smooth(values, window=5):
    return np.convolve(values, np.ones(window)/window, mode='valid')

colors = {
    "InvNeRF-Seg (full)": "#1f77b4",
    "Fine-tune density only": "#ff7f0e",
    "Fine-tune color only": "#2ca02c",
    "FruitNeRF (BCE)": "#d62728",
    "FruitNeRF (MSE)": "#9467bd",
}

# --- Apple (no legend) ---
conditions_apple = {
    "InvNeRF-Seg (full)": "iou_log_appleTreeSeg_optimal.pkl",
    "Fine-tune density only": "iou_log_appleTreeSeg_fineTuneDensityFreezeColor.pkl",
    "Fine-tune color only": "iou_log_appleTreeSeg_fineTuneColorFreezeDensity.pkl",
    "FruitNeRF (BCE)": "iou_log_fruitNerf_appleTree_BCELossOnly.pkl",
    "FruitNeRF (MSE)": "iou_log_fruitNerf_appleTree_mseLossOnly.pkl",
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for label, filepath in conditions_apple.items():
    try:
        with open(filepath, "rb") as f:
            iou_log = pickle.load(f)[:120]
        epochs = [x[0] for x in iou_log]
        ious = [x[1] for x in iou_log]
        epochs_smooth = epochs[:len(smooth(ious))]
        ax.plot(epochs_smooth, smooth(ious), linewidth=2, color=colors[label])
    except FileNotFoundError:
        print(f"File not found: {filepath}")

ax.set_xlabel("Epoch")
ax.set_ylabel("IoU")
ax.set_ylim(-0.1, 0.8)
plt.tight_layout()
plt.savefig("ablation_iou_convergence_apple.png", dpi=300, bbox_inches="tight")
plt.savefig("ablation_iou_convergence_apple.pdf", dpi=300, bbox_inches="tight")
plt.show()

# --- Peach (with legend) ---
conditions_peach = {
    "InvNeRF-Seg (full)": "iou_log_invNerf_peachTreeSeg_optimal.pkl",
    "Fine-tune density only": "iou_log_invNerf_peachTreeSeg_fineTuneDensityFreezeColor.pkl",
    "Fine-tune color only": "iou_log_invNerf_peachTreeSeg_fineTuneColorFreezeDensity.pkl",
    "FruitNeRF (BCE)": "iou_log_fruitNerf_peachTree_BCELossOnly.pkl",
    "FruitNeRF (MSE)": "iou_log_fruitNerf_peachTree_mseLossOnly.pkl",
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for label, filepath in conditions_peach.items():
    try:
        with open(filepath, "rb") as f:
            iou_log = pickle.load(f)[:120]
        epochs = [x[0] for x in iou_log]
        ious = [x[1] for x in iou_log]
        epochs_smooth = epochs[:len(smooth(ious))]
        ax.plot(epochs_smooth, smooth(ious), label=label, linewidth=2, color=colors[label])
    except FileNotFoundError:
        print(f"File not found: {filepath}")

ax.set_xlabel("Epoch")
ax.set_ylabel("IoU")
ax.set_ylim(-0.1, 0.8)
ax.legend(loc="lower right", fontsize=15, frameon=False)
plt.tight_layout()
plt.savefig("ablation_iou_convergence_peach.png", dpi=300, bbox_inches="tight")
plt.savefig("ablation_iou_convergence_peach.pdf", dpi=300, bbox_inches="tight")
plt.show()











# iou_log_invNerf_peachTreeSeg_maskOnly





import numpy as np

def smooth(values, window=5):
    return np.convolve(values, np.ones(window)/window, mode='valid')


import pickle
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

# Load all IoU logs — update filenames to match yours
conditions = {
    "InvNeRF-Seg (full)": "iou_log_invNerf_peachTreeSeg_optimal.pkl",
    "Fine-tune density only": "iou_log_invNerf_peachTreeSeg_fineTuneDensityFreezeColor.pkl",
    "Fine-tune color only": "iou_log_invNerf_peachTreeSeg_fineTuneColorFreezeDensity.pkl",
    "FruitNeRF (BCE)": "iou_log_fruitNerf_peachTree_BCELossOnly.pkl",
    "FruitNeRF (MSE)": "iou_log_fruitNerf_peachTree_mseLossOnly.pkl",
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for label, filepath in conditions.items():
    try:
        with open(filepath, "rb") as f:
            iou_log = pickle.load(f)[:120]
        epochs = [x[0] for x in iou_log]
        ious = [x[1] for x in iou_log]
        # ax.plot(epochs, ious, label=label, linewidth=2)
        epochs_smooth = epochs[:len(smooth(ious))]
        ax.plot(epochs_smooth, smooth(ious), label=label, linewidth=2)
        print(label)
        print(smooth(ious))
    except FileNotFoundError:
        print(f"File not found: {filepath}")

ax.set_xlabel("Epoch")
ax.set_ylabel("IoU")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ablation_iou_convergence_peach.png", dpi=300, bbox_inches="tight")
plt.savefig("ablation_iou_convergence_peach.pdf", dpi=300, bbox_inches="tight")
plt.show()














##################
conditions = {
    "InvNeRF-Seg (full)": "iou_log_appleTreeSeg_optimal.pkl",
    "Fine-tune density only": "iou_log_appleTreeSeg_fineTuneDensityFreezeColor.pkl",
    "Fine-tune color only": "iou_log_appleTreeSeg_fineTuneColorFreezeDensity.pkl",
    "FruitNeRF (BCE)": "iou_log_fruitNerf_appleTree_BCELossOnly.pkl",
    "FruitNeRF (MSE)": "iou_log_fruitNerf_appleTree_mseLossOnly.pkl",
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for label, filepath in conditions.items():
    try:
        with open(filepath, "rb") as f:
            iou_log = pickle.load(f)[:120]
        epochs = [x[0] for x in iou_log]
        ious = [x[1] for x in iou_log]
        # ax.plot(epochs, ious, label=label, linewidth=2)
        epochs_smooth = epochs[:len(smooth(ious))]
        print(label)
        print(smooth(ious))
        ax.plot(epochs_smooth, smooth(ious), label=label, linewidth=2)
    except FileNotFoundError:
        print(f"File not found: {filepath}")

ax.set_xlabel("Epoch")
ax.set_ylabel("IoU")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ablation_iou_convergence_apple.png", dpi=300, bbox_inches="tight")
plt.savefig("ablation_iou_convergence_apple.pdf", dpi=300, bbox_inches="tight")
plt.show()












