# Data Metadata

This directory contains `transforms.json` camera metadata copied from the local experiment datasets.

The corresponding images, masks, semantic labels, meshes, and generated point clouds are intentionally omitted from Git because of file size. Recreate the full dataset layout by placing the omitted folders back under each scene directory:

```text
data/
  appleTree/
    transforms.json
    images/
  appleTreeSeg/
    transforms.json
    images/
  peachTree/
    transforms.json
    images/
  peachTreeSeg/
    transforms.json
    images/
  soybeanRGB/
    transforms.json
    images/
  soybeanRGBSeg/
    transforms.json
    images/
```

The apple and peach scenes are derived from the publicly available FruitNeRF dataset:

- FruitNeRF paper: <https://arxiv.org/abs/2408.06190>
- FruitNeRF dataset archive: <https://zenodo.org/records/10869455>

The soybean scenes are self-collected for the InvNeRF-Seg study.
