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
