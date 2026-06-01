# Soybean Training Scripts

These scripts reproduce the soybean experiments described in the manuscript using the local dataset layout:

```text
D:/InvNerf_seg_relatedScripts/soybeanRelated/
  soybeanRGB/
  soybeanSeg_singlePodMASK/
  soybeanSeg_doublePodMASK/
```

Run them from `D:/InvNerf_seg_relatedScripts/soybeanRelated` with the Anaconda `nerfstudio` environment:

```powershell
conda run -n nerfstudio python train_soybean_rgb_nerfacto_128.py
conda run -n nerfstudio python train_soybean_singlepod_invnerf_128.py
conda run -n nerfstudio python train_soybean_doublepod_invnerf_128.py
```

The RGB script trains the 128-hidden-dimension soybean Nerfacto model for 4000 epochs and saves `soybean_rgb_nerfacto_128.pth`. The single-pod and double-pod InvNeRF-Seg scripts load that RGB checkpoint, fine-tune for 6000 epochs on the corresponding mask-image dataset, and save `soybean_singlepod_invnerf_128.pth` or `soybean_doublepod_invnerf_128.pth`.

Learning rates follow the manuscript: RGB training uses `1e-2` for field/proposal networks and `1e-4` for the camera optimizer; fine-tuning uses `1e-4` and `1e-6`, respectively. The near/far planes are `0.05` and `2.0`.
