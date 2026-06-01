# External Artifacts

The local reproducibility folder contains trained model checkpoints, dense point clouds, source image sets, masks, and mesh files that are too large for a normal GitHub commit.

## Not Committed

- `*.pth`: trained PyTorch checkpoints, many around 100-245 MB each.
- `*.ply`: generated/labeled point clouds, some around 100-223 MB each.
- `*.obj` and `*.mtl`: scene meshes.
- Dataset image folders and mask/semantic folders.

Exception: the three soybean manuscript checkpoints are tracked with Git LFS in `weights/soybean/`.

## Local Source

The artifacts were inventoried from the local reproducibility workspace used to prepare this repository.

See [artifact_inventory.csv](artifact_inventory.csv) for file names and sizes.

## Recommended Release Options

- Put the checkpoint and point-cloud bundle in a GitHub Release if the total release size is acceptable.
- Use Zenodo, Figshare, OSF, institutional storage, or another DOI-backed archive for manuscript reproducibility.
- Keep this repository as the code and lightweight metadata entry point, then link the archive URL in the README.
