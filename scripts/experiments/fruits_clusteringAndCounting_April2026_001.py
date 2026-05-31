#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:18:10 2026

@author: nibio
"""






import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors


def cluster_volume(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return np.prod(maxs - mins)


# All point clouds to process: (filename, method_label)
pcd_list = [
    # Apple
    ("appleTreeSeg_invNerf_density_002_April2026", "InvNeRF-Seg", "apple"),
    ("appleTreeSeg_invNerf_density_001_April2026_fineTuneDensityFreezeColor", "InvNeRF-Seg-fineTuneDensityFreezeColor", "apple"),
    ("appleTreeSeg_invNerf_density_001_April2026_fineTuneColorFreezeDensity", "InvNeRF-SegfineTuneColorFreezeDensity", "apple"),
    ("appleTree_fruitNerf_density_001_April2026_BCEMaskLoss", "FruitNeRF_BCE", "apple"),
    ("appleTree_fruitNerf_density_001_April2026_mseMaskLoss", "FruitNeRF_MSE", "apple"),
    ("variantSA3D_appleTree_densitySeg_002_defaultConfig", "SA3D_density", "apple"),
    ("originalSA3D_appleTree_weightSeg_002_defaultConfig", "SA3D_weight", "apple"),
    # Peach
    ("peachTreeSeg_invNerf_density_002_April2026", "InvNeRF-Seg", "peach"),
    ("peachTreeSeg_invNerf_density_001_April2026_fineTuneDensityFreezeColor", "InvNeRF-Seg-fineTuneDensityFreezeColor", "peach"),
    ("peachTreeSeg_invNerf_density_001_April2026_fineTuneColorFreezeDensity", "InvNeRF-SegfineTuneColorFreezeDensity", "peach"),
    ("peachTree_fruitNerf_density_001_April2026_BCEMaskLoss", "FruitNeRF_BCE", "peach"),
    ("peachTree_fruitNerf_density_001_April2026_mseMaskLoss", "FruitNeRF_MSE", "peach"),
    ("variantSA3D_peachTree_densitySeg_002_defaultConfig", "SA3D_density", "peach"),
    ("originalSA3D_peachTree_weightSeg_002_defaultConfig", "SA3D_weight", "peach"),
]

# Ground truth counts
gt_counts = {
    "apple": 283,
    "peach": 152,
}

# Store results for final summary table
results = []

for pcd_filename, method_label, fruit_type in pcd_list:

    print("\n" + "=" * 60)
    print(f"Processing: {method_label} | {fruit_type}")
    print(f"File: {pcd_filename}.ply")
    print("=" * 60)

    # -----------------------------
    # Load point cloud
    # -----------------------------
    try:
        pcd = o3d.io.read_point_cloud(f"{pcd_filename}.ply")
    except Exception as e:
        print(f"SKIPPED — file not found: {e}")
        results.append((fruit_type, method_label, "N/A", gt_counts[fruit_type]))
        continue

    points_0 = np.asarray(pcd.points)
    colors_0 = np.asarray(pcd.colors)

    if points_0.shape[0] == 0:
        print("SKIPPED — empty point cloud")
        results.append((fruit_type, method_label, 0, gt_counts[fruit_type]))
        continue

    # -----------------------------
    # Color-based foreground filtering (handles non-white points if any)
    # -----------------------------
    colors_norm = colors_0 / 255.0 if colors_0.max() > 1.5 else colors_0
    mask_color = np.mean(colors_norm, axis=1) > 0.5
    points_fg = points_0[mask_color]
    print(f"After color filter: {points_fg.shape[0]}")

    if points_fg.shape[0] == 0:
        print("SKIPPED — no foreground points")
        results.append((fruit_type, method_label, 0, gt_counts[fruit_type]))
        continue

    # -----------------------------
    # Bounding box filtering
    # -----------------------------
    min_bound = np.array([-0.5, -0.5, -0.7])
    max_bound = np.array([0.5, 0.5, 0.2])
    mask_bbox = np.all((points_fg >= min_bound) & (points_fg <= max_bound), axis=1)
    points = points_fg[mask_bbox]
    print(f"After bbox filter: {points.shape[0]}")

    if points.shape[0] == 0:
        print("SKIPPED — no points in bbox")
        results.append((fruit_type, method_label, 0, gt_counts[fruit_type]))
        continue
    
    ####  downsampling if necessary
    # After bbox filtering, before denoising
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd_temp.voxel_down_sample(voxel_size=0.001)
    points = np.asarray(pcd_down.points)
    print(f"After voxel downsampling: {points.shape[0]}")
    # -----------------------------
    # Density-based denoising
    # -----------------------------
    nbrs = NearestNeighbors(n_neighbors=6).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    threshold = np.percentile(mean_distances, 90)
    denoised_points = points[mean_distances < threshold]
    print(f"After denoising: {denoised_points.shape[0]}")

    if denoised_points.shape[0] == 0:
        print("SKIPPED — no points after denoising")
        results.append((fruit_type, method_label, 0, gt_counts[fruit_type]))
        continue

    # -----------------------------
    # Initial DBSCAN clustering
    # -----------------------------
    db = DBSCAN(eps=0.0065, min_samples=3).fit(denoised_points)
    init_labels = db.labels_
    unique_init = [l for l in np.unique(init_labels) if l != -1]
    print(f"Initial clusters: {len(unique_init)}")

    if len(unique_init) == 0:
        print("SKIPPED — no clusters found")
        results.append((fruit_type, method_label, 0, gt_counts[fruit_type]))
        continue

    # -----------------------------
    # Compute size statistics
    # -----------------------------
    sizes = [np.sum(init_labels == l) for l in unique_init]
    non_dust = [s for s in sizes if s > 20]
    if not non_dust:
        non_dust = sizes

    median_size = np.median(non_dust)
    split_size_thresh = 1.0 * median_size
    min_accept_size = 0

    print(f"Median size: {median_size:.1f} | Split trigger: {split_size_thresh:.1f}")

    # -----------------------------
    # Estimate median volume
    # -----------------------------
    size_tol = 0.2 * median_size
    median_clusters = []
    for l in unique_init:
        idx = np.where(init_labels == l)[0]
        if abs(len(idx) - median_size) <= size_tol:
            median_clusters.append(idx)

    if not median_clusters:
        median_clusters = [
            np.where(init_labels == l)[0]
            for l in unique_init
            if len(np.where(init_labels == l)[0]) >= median_size
        ]

    if median_clusters:
        median_volumes = [cluster_volume(denoised_points[idx]) for idx in median_clusters]
        median_volume = np.median(median_volumes)
    else:
        median_volume = 1e-6

    split_volume_thresh = 2.0 * median_volume
    print(f"Median volume: {median_volume:.4e} | Split volume trigger: {split_volume_thresh:.4e}")

    # -----------------------------
    # Recursive splitting
    # -----------------------------
    queue = []
    for l in unique_init:
        idx = np.where(init_labels == l)[0]
        if len(idx) >= min_accept_size:
            queue.append(idx)

    final_labels = np.full(len(denoised_points), -1)
    next_id = 0

    while queue:
        current_idx = queue.pop(0)
        pts = denoised_points[current_idx]
        current_size = len(current_idx)
        current_volume = cluster_volume(pts)

        if current_size > split_size_thresh and current_volume > split_volume_thresh:
            try:
                km = KMeans(n_clusters=2, n_init=6, random_state=42).fit(pts)
                children = []
                valid_split = True
                for i in [0, 1]:
                    sub_idx = current_idx[km.labels_ == i]
                    if len(sub_idx) < min_accept_size:
                        valid_split = False
                        break
                    children.append(sub_idx)

                if valid_split:
                    queue.extend(children)
                else:
                    final_labels[current_idx] = next_id
                    next_id += 1
            except Exception:
                final_labels[current_idx] = next_id
                next_id += 1
        else:
            final_labels[current_idx] = next_id
            next_id += 1
    ##
    # -----------------------------
    # Filter tiny clusters
    # -----------------------------
    min_cluster_size = 40
    filtered_labels = np.full(len(denoised_points), -1)
    filtered_id = 0
    for lab in range(next_id):
        mask = final_labels == lab
        if mask.sum() >= min_cluster_size:
            filtered_labels[mask] = filtered_id
            filtered_id += 1
    print(f"Before filter: {next_id} | After filter: {filtered_id} (removed {next_id - filtered_id} tiny clusters)")
    next_id = filtered_id
    final_labels = filtered_labels

    gt = gt_counts[fruit_type]
    print(f"FINAL count: {next_id} (GT: {gt})")
    results.append((fruit_type, method_label, next_id, gt))

    # -----------------------------
    # Remove tiny clusters and save labeled point cloud
    # -----------------------------
    keep_mask = final_labels >= 0
    denoised_points = denoised_points[keep_mask]
    final_labels = final_labels[keep_mask]

    colors = np.zeros((len(denoised_points), 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    for lab in range(next_id):
        colors[final_labels == lab] = rng.random(3)

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(denoised_points)
    labeled_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f"labeled_{method_label}_{fruit_type}_002.ply", labeled_pcd)

# =====================================================
# SUMMARY TABLE
# =====================================================
print("\n" + "=" * 60)
print("COUNTING RESULTS SUMMARY")
print("=" * 60)
print(f"{'Fruit':<8} {'Method':<20} {'Predicted':<12} {'GT':<8}")
print("-" * 48)
for fruit_type, method, pred, gt in results:
    print(f"{fruit_type:<8} {method:<20} {str(pred):<12} {gt:<8}")













