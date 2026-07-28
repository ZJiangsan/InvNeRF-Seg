#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InvNeRF-Seg two-stage training script.

Stage 1 trains a standard Nerfacto model from multi-view RGB images.
Stage 2 loads the best Stage-1 checkpoint and fine-tunes the unchanged
model using binary segmentation masks formatted as three-channel images.

Expected repository layout
--------------------------
InvNeRF-Seg/
├── data/
│   ├── appleTree/
│   ├── appleTreeSeg/
│   ├── peachTree/
│   ├── peachTreeSeg/
│   ├── soybeanRGB/
│   └── soybeanRGBSeg/
├── checkpoints/              # created automatically
└── invnerf_seg_train.py

Each dataset folder must contain ``transforms.json`` and the images referenced
by that file. The RGB and segmentation datasets for a scene must contain
matching camera views.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.classification import BinaryJaccardIndex

from nerfstudio.cameras.camera_optimizers import (
    CameraOptimizer,
    CameraOptimizerConfig,
)
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    NerfstudioDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolve paths relative to this script rather than the shell's current
# working directory. This allows the script to be launched from anywhere.
REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPOSITORY_ROOT / "data"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints"


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""
    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    'How far along the ray to start sampling.'
    far_plane: float = 1000.0
    'How far along the ray to stop sampling.'
    background_color: Literal['random', 'last_sample', 'black', 'white'] = 'last_sample'
    'Whether to randomize the background color.'
    hidden_dim: int = 64
    'Dimension of hidden layers'
    hidden_dim_color: int = 64
    'Dimension of hidden layers for color network'
    hidden_dim_transient: int = 64
    'Dimension of hidden layers for transient network'
    num_levels: int = 16
    'Number of levels of the hashmap for the base mlp.'
    base_res: int = 16
    'Resolution of the base grid for the hashgrid.'
    max_res: int = 2048
    'Maximum resolution of the hashmap for the base mlp.'
    log2_hashmap_size: int = 19
    'Size of the hashmap for the base mlp'
    features_per_level: int = 2
    'How many hashgrid features per level'
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    'Number of samples per ray for each proposal network.'
    num_nerf_samples_per_ray: int = 48
    'Number of samples per ray for the nerf network.'
    proposal_update_every: int = 5
    'Sample every n steps after the warmup'
    proposal_warmup: int = 5000
    'Scales n from 1 to proposal_update_every over this many steps'
    num_proposal_iterations: int = 2
    'Number of proposal network iterations.'
    use_same_proposal_network: bool = False
    'Use the same proposal network. Otherwise use different ones.'
    proposal_net_args_list: List[Dict] = field(default_factory=lambda: [{'hidden_dim': 16, 'log2_hashmap_size': 17, 'num_levels': 5, 'max_res': 128, 'use_linear': False}, {'hidden_dim': 16, 'log2_hashmap_size': 17, 'num_levels': 5, 'max_res': 256, 'use_linear': False}])
    'Arguments for the proposal density fields.'
    proposal_initial_sampler: Literal['piecewise', 'uniform'] = 'piecewise'
    'Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes.'
    interlevel_loss_mult: float = 1.0
    'Proposal loss multiplier.'
    distortion_loss_mult: float = 0.002
    'Distortion loss multiplier.'
    orientation_loss_mult: float = 0.0001
    'Orientation loss multiplier on computed normals.'
    pred_normal_loss_mult: float = 0.001
    'Predicted normal loss multiplier.'
    use_proposal_weight_anneal: bool = True
    'Whether to use proposal weight annealing.'
    use_appearance_embedding: bool = True
    'Whether to use an appearance embedding.'
    use_average_appearance_embedding: bool = True
    'Whether to use average appearance embedding or zeros for inference.'
    proposal_weights_anneal_slope: float = 10.0
    'Slope of the annealing function for the proposal weights.'
    proposal_weights_anneal_max_num_iters: int = 1000
    'Max num iterations for the annealing function.'
    use_single_jitter: bool = True
    'Whether use single jitter or not for the proposal networks.'
    predict_normals: bool = True
    'Whether to predict normals or not.'
    disable_scene_contraction: bool = False
    'Whether to disable scene contraction or not.'
    use_gradient_scaling: bool = False
    'Use gradient scaler where the gradients are lower for points closer to the camera.'
    implementation: Literal['tcnn', 'torch'] = 'tcnn'
    'Which implementation to use for the model.'
    appearance_embed_dim: int = 32
    'Dimension of the appearance embedding.'
    average_init_density: float = 1.0
    'Average initial density output from MLP. '
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode='SO3xR3'))
    'Config of the camera optimizer to use'
    pass_semantic_gradients: bool = False
    'Whether to pass semantic gradients.'
    use_semantics: bool = True
    'Whether to use semantics.'
    num_semantic_classes: int = 1
    'Number of semantic classes.'
    semantic_loss_weight: float = 1.0
    'Number of semantic classes.'


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """
    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float('inf'))
        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0
        self.field = NerfactoField(self.scene_box.aabb, hidden_dim=self.config.hidden_dim, num_levels=self.config.num_levels, max_res=self.config.max_res, base_res=self.config.base_res, features_per_level=self.config.features_per_level, log2_hashmap_size=self.config.log2_hashmap_size, hidden_dim_color=self.config.hidden_dim_color, hidden_dim_transient=self.config.hidden_dim_transient, spatial_distortion=scene_contraction, num_images=self.num_train_data, use_pred_normals=self.config.predict_normals, use_average_appearance_embedding=self.config.use_average_appearance_embedding, appearance_embedding_dim=appearance_embedding_dim, average_init_density=self.config.average_init_density, implementation=self.config.implementation)
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(num_cameras=self.num_train_data, device='cpu')
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, 'Only one proposal network is allowed.'
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args, average_init_density=self.config.average_init_density, implementation=self.config.implementation)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args, average_init_density=self.config.average_init_density, implementation=self.config.implementation)
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]), 1, self.config.proposal_update_every)
        initial_sampler = None
        if self.config.proposal_initial_sampler == 'uniform':
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray, num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray, num_proposal_network_iterations=self.config.num_proposal_iterations, single_jitter=self.config.use_single_jitter, update_sched=update_schedule, initial_sampler=initial_sampler)
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method='median')
        self.renderer_expected_depth = DepthRenderer(method='expected')
        self.renderer_normals = NormalsRenderer()
        self.normals_shader = NormalsShader()
        self.rgb_loss = MSELoss()
        self.step = 0
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups['proposal_networks'] = list(self.proposal_networks.parameters())
        param_groups['fields'] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)
            callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], update_every_num_iters=1, func=set_anneal))
            callbacks.append(TrainingCallback(where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION], update_every_num_iters=1, func=self.proposal_sampler.step_cb))
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        outputs = {'rgb': rgb, 'accumulation': accumulation, 'depth': depth, 'expected_depth': expected_depth}
        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs['normals'] = self.normals_shader(normals)
            outputs['pred_normals'] = self.normals_shader(pred_normals)
        if self.training:
            outputs['weights_list'] = weights_list
            outputs['ray_samples_list'] = ray_samples_list
        if self.training and self.config.predict_normals:
            outputs['rendered_orientation_loss'] = orientation_loss(weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions)
            outputs['rendered_pred_normal_loss'] = pred_normal_loss(weights.detach(), field_outputs[FieldHeadNames.NORMALS].detach(), field_outputs[FieldHeadNames.PRED_NORMALS])
        for i in range(self.config.num_proposal_iterations):
            outputs[f'prop_depth_{i}'] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch['image'].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        predicted_rgb = outputs['rgb']
        metrics_dict['psnr'] = self.psnr(predicted_rgb, gt_rgb)
        if self.training:
            metrics_dict['distortion'] = distortion_loss(outputs['weights_list'], outputs['ray_samples_list'])
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch['image'].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(pred_image=outputs['rgb'], pred_accumulation=outputs['accumulation'], gt_image=image)
        loss_dict['rgb_loss'] = 1 * self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict['interlevel_loss'] = self.config.interlevel_loss_mult * interlevel_loss(outputs['weights_list'], outputs['ray_samples_list'])
            assert metrics_dict is not None and 'distortion' in metrics_dict
            loss_dict['distortion_loss'] = self.config.distortion_loss_mult * metrics_dict['distortion']
            if self.config.predict_normals:
                loss_dict['orientation_loss'] = self.config.orientation_loss_mult * torch.mean(outputs['rendered_orientation_loss'])
                loss_dict['pred_normal_loss'] = self.config.pred_normal_loss_mult * torch.mean(outputs['rendered_pred_normal_loss'])
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch['image'].to(self.device)
        predicted_rgb = outputs['rgb']
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs['accumulation'])
        depth = colormaps.apply_depth_colormap(outputs['depth'], accumulation=outputs['accumulation'])
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)
        metrics_dict = {'psnr': float(psnr.item()), 'ssim': float(ssim)}
        metrics_dict['lpips'] = float(lpips)
        images_dict = {'img': combined_rgb, 'accumulation': combined_acc, 'depth': combined_depth}
        for i in range(self.config.num_proposal_iterations):
            key = f'prop_depth_{i}'
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key], accumulation=outputs['accumulation'])
            images_dict[key] = prop_depth_i
        return (metrics_dict, images_dict)



def save_checkpoint(
    model: torch.nn.Module,
    optimizers: dict,
    epoch: int,
    checkpoint_path: Path,
) -> None:
    """Save the model, optimizer states, and current epoch."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizers_state_dict": {
            name: (
                [optimizer.state_dict() for optimizer in optimizer_group]
                if isinstance(optimizer_group, list)
                else optimizer_group.state_dict()
            )
            for name, optimizer_group in optimizers.items()
        },
        "epoch": epoch,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")


def load_model_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
) -> int:
    """Load only the model parameters and return the saved epoch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    saved_epoch = int(checkpoint.get("epoch", -1))
    print(f"Loaded model checkpoint from epoch {saved_epoch}: {checkpoint_path}")
    return saved_epoch


def load_transforms(file_path: Path) -> dict:
    """Load a Nerfstudio transforms.json file."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_frames(transforms_data: dict) -> int:
    """Return the number of frames listed in transforms.json."""
    frames = transforms_data.get("frames", [])
    if not frames:
        raise ValueError("No frames were found in transforms.json.")
    return len(frames)


def zero_grad_all(optimizers: dict) -> None:
    """Clear gradients for every optimizer."""
    for optimizer_group in optimizers.values():
        if isinstance(optimizer_group, list):
            for optimizer in optimizer_group:
                optimizer.zero_grad(set_to_none=True)
        else:
            optimizer_group.zero_grad(set_to_none=True)


def step_all_optimizers(optimizers: dict) -> None:
    """Perform one optimization step with every optimizer."""
    for optimizer_group in optimizers.values():
        if isinstance(optimizer_group, list):
            for optimizer in optimizer_group:
                optimizer.step()
        else:
            optimizer_group.step()


def build_strided_dataset(source_dir: Path, stride: int = 1) -> Path:
    """Create a Nerfstudio dataset containing every ``stride``-th frame.

    The source dataset is left unchanged. Images are symlinked when possible,
    hard-linked as the second choice, and copied only as a fallback.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, received {stride}")

    source_dir = source_dir.resolve()
    source_json = source_dir / "transforms.json"

    if not source_json.exists():
        raise FileNotFoundError(f"Missing transforms.json: {source_json}")

    output_dir = source_dir.parent / f"{source_dir.name}_stride{stride}"
    output_json = output_dir / "transforms.json"

    transforms = load_transforms(source_json)
    all_frames = transforms.get("frames", [])
    selected_frames = all_frames[::stride]

    if not selected_frames:
        raise ValueError(f"No frames found in {source_json}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in selected_frames:
        relative_path = Path(frame["file_path"])
        candidates = [
            source_dir / relative_path,
            source_dir / relative_path.with_suffix(".png"),
            source_dir / relative_path.with_suffix(".jpg"),
            source_dir / relative_path.with_suffix(".jpeg"),
        ]
        source_image = next(
            (candidate for candidate in candidates if candidate.exists()),
            None,
        )

        if source_image is None:
            raise FileNotFoundError(
                f"Could not find image for frame path: {frame['file_path']}"
            )

        target_relative_path = relative_path
        if not target_relative_path.suffix:
            target_relative_path = target_relative_path.with_suffix(
                source_image.suffix
            )
            frame["file_path"] = target_relative_path.as_posix()

        target_image = output_dir / target_relative_path
        target_image.parent.mkdir(parents=True, exist_ok=True)

        if target_image.exists():
            continue

        try:
            target_image.symlink_to(source_image)
        except OSError:
            try:
                os.link(source_image, target_image)
            except OSError:
                shutil.copy2(source_image, target_image)

    reduced_transforms = dict(transforms)
    reduced_transforms["frames"] = selected_frames

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(reduced_transforms, file, indent=2)

    print(
        f"Prepared dataset: {len(all_frames)} -> {len(selected_frames)} "
        f"images (stride={stride})"
    )
    print(f"Dataset path: {output_dir}")
    return output_dir


def make_checkpoint_path(
    checkpoint_dir: Path,
    dataset_name: str,
    stage_name: str,
    stride: int,
) -> Path:
    """Return a unique checkpoint path for one dataset and stage."""
    return checkpoint_dir / (
        f"{dataset_name}_stride{stride}_stage_{stage_name}_July2026.pth"
    )


def create_optimizers(
    model: NerfactoModel,
    learning_rates: dict,
) -> tuple[dict, torch.optim.Optimizer]:
    """Create field, proposal-network, and camera optimizers."""
    field_optimizer = torch.optim.Adam(
        model.field.parameters(),
        lr=learning_rates["field"],
        eps=1e-8,
    )
    proposal_optimizers = [
        torch.optim.Adam(
            proposal_network.parameters(),
            lr=learning_rates["proposal"],
            eps=1e-8,
        )
        for proposal_network in model.proposal_networks
    ]
    camera_optimizer = torch.optim.Adam(
        model.camera_optimizer.parameters(),
        lr=learning_rates["camera"],
        eps=1e-8,
    )

    optimizers = {
        "field": field_optimizer,
        "camera": camera_optimizer,
        "proposal_networks": proposal_optimizers,
    }
    return optimizers, field_optimizer


def create_data_manager(
    dataset_path: Path,
    rays_per_batch: int,
) -> VanillaDataManager:
    """Create the Nerfstudio data manager used for both stages."""
    parser_config = NerfstudioDataParserConfig()
    parser_config.data = dataset_path
    parser_config.downscale_factor = 1
    parser_config.train_split_fraction = 0.8

    manager_config = VanillaDataManagerConfig()
    manager_config.dataparser = parser_config
    manager_config.train_num_rays_per_batch = rays_per_batch
    manager_config.eval_num_rays_per_batch = rays_per_batch
    manager_config.camera_res_scale_factor = 1

    return VanillaDataManager(manager_config)


def evaluate_full_image(
    model: NerfactoModel,
    data_manager: VanillaDataManager,
    epoch: int,
    height: int,
    width: int,
    near_plane: float,
    far_plane: float,
    iou_metric: BinaryJaccardIndex,
    dataset_name: str,
    stage_name: str,
) -> tuple[float, float]:
    """Render one evaluation image and report full-image PSNR and mask IoU."""
    model.eval()

    with torch.no_grad():
        eval_camera, eval_batch = data_manager.next_eval_image(epoch)
        eval_rays = eval_camera.generate_rays(0)

        image_index = eval_batch["image_idx"]
        print(f"Evaluation image index: {image_index}")

        ground_truth_rgb = eval_batch["image"][:, :, :3].float().cpu()

        flat_origins = eval_rays.origins.reshape(-1, 3)
        flat_directions = eval_rays.directions.reshape(-1, 3)
        flat_pixel_area = eval_rays.pixel_area.reshape(-1, 1)

        if eval_rays.camera_indices is None:
            camera_index = int(image_index)
            flat_camera_indices = torch.full(
                (height * width, 1),
                camera_index,
                dtype=torch.long,
            )
        else:
            flat_camera_indices = eval_rays.camera_indices.reshape(-1, 1)

        predicted_rgb_rows = []
        predicted_depth_rows = []

        for row_index in range(height):
            row_slice = slice(
                width * row_index,
                width * (row_index + 1),
            )

            row_bundle = RayBundle(
                origins=flat_origins[row_slice].float().to(DEVICE),
                directions=flat_directions[row_slice].float().to(DEVICE),
                nears=torch.full(
                    (width, 1),
                    near_plane,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                fars=torch.full(
                    (width, 1),
                    far_plane,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                pixel_area=flat_pixel_area[row_slice].float().to(DEVICE),
                camera_indices=(
                    flat_camera_indices[row_slice].long().to(DEVICE)
                ),
            )

            row_outputs = model.get_outputs(row_bundle)
            predicted_rgb_rows.append(
                row_outputs["rgb"].detach().cpu().reshape(1, width, 3)
            )
            predicted_depth_rows.append(
                row_outputs["depth"].detach().cpu().reshape(1, width)
            )

            del row_outputs, row_bundle

        predicted_rgb = torch.cat(predicted_rgb_rows, dim=0)
        predicted_depth = torch.cat(predicted_depth_rows, dim=0)

        mse = torch.mean((predicted_rgb - ground_truth_rgb) ** 2)
        psnr_value = (
            -10.0 * torch.log10(mse.clamp_min(1e-10))
        ).item()

        predicted_mask = (
            predicted_rgb.mean(dim=-1) > 0.5
        ).long()
        ground_truth_mask = (
            ground_truth_rgb.mean(dim=-1) > 0.5
        ).long()

        iou_metric.reset()
        iou_metric.update(
            predicted_mask.flatten().to(DEVICE),
            ground_truth_mask.flatten().to(DEVICE),
        )
        iou_value = iou_metric.compute().item()
        iou_metric.reset()

        depth_minimum = predicted_depth.min()
        depth_range = predicted_depth.max() - depth_minimum
        depth_visualization = (
            predicted_depth - depth_minimum
        ) / depth_range.clamp_min(1e-8)

        figure, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(predicted_rgb)
        axes[0].set_title("Rendered mask" if stage_name == "seg" else "Rendered RGB")
        axes[1].imshow(depth_visualization)
        axes[1].set_title("Rendered depth")
        axes[2].imshow(ground_truth_rgb)
        axes[2].set_title(
            "Ground-truth mask" if stage_name == "seg" else "Ground-truth RGB"
        )

        for axis in axes:
            axis.axis("off")

        figure.suptitle(f"{dataset_name}: stage {stage_name}")
        plt.tight_layout()
        plt.show()
        plt.close(figure)

        print(f"Full-image PSNR: {psnr_value:.4f}")
        print(f"Mask IoU: {iou_value:.4f}")

    return psnr_value, iou_value



# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name": "apple",
        "rgb_folder": "appleTree",
        "seg_folder": "appleTreeSeg",
        "far_plane": 1000.0,
        "hidden_dim": 64,
        "hidden_dim_color": 64,
    },
    {
        "name": "peach",
        "rgb_folder": "peachTree",
        "seg_folder": "peachTreeSeg",
        "far_plane": 1000.0,
        "hidden_dim": 64,
        "hidden_dim_color": 64,
    },
    {
        "name": "soybean",
        "rgb_folder": "soybeanRGB",
        "seg_folder": "soybeanRGBSeg",
        "far_plane": 2.0,
        "hidden_dim": 128,
        "hidden_dim_color": 128,
    },
]

STAGE_CONFIGS = {
    1: {
        "name": "rgb",
        "num_epochs": 4000,
        "learning_rates": {
            "field": 1e-2,
            "proposal": 1e-2,
            "camera": 1e-4,
        },
    },
    2: {
        "name": "seg",
        "num_epochs": 6000,
        "learning_rates": {
            "field": 1e-4,
            "proposal": 1e-4,
            "camera": 1e-6,
        },
    },
}

NEAR_PLANE = 0.01
EVAL_EVERY = 600
RAYS_PER_BATCH = 1024 * 20

# 1 uses every frame; 2 keeps frames 0, 2, 4, ...
IMAGE_STRIDE = 1


def train_stage(
    base_dir: Path,
    checkpoint_dir: Path,
    dataset_config: dict,
    stage_number: int,
    iou_metric: BinaryJaccardIndex,
) -> None:
    """Train one RGB or segmentation stage using a folder under ``data/``."""
    dataset_name = dataset_config["name"]
    stage_config = STAGE_CONFIGS[stage_number]
    stage_name = stage_config["name"]
    far_plane = float(dataset_config["far_plane"])

    rgb_checkpoint_path = make_checkpoint_path(
        checkpoint_dir,
        dataset_name,
        "rgb",
        IMAGE_STRIDE,
    )
    output_checkpoint_path = make_checkpoint_path(
        checkpoint_dir,
        dataset_name,
        stage_name,
        IMAGE_STRIDE,
    )

    source_folder_name = (
        dataset_config["rgb_folder"]
        if stage_number == 1
        else dataset_config["seg_folder"]
    )
    source_folder = base_dir / source_folder_name

    print("\n" + "=" * 78)
    print(f"Dataset: {dataset_name}")
    print(f"Stage: {stage_number} ({stage_name})")
    print(f"Input: {source_folder}")
    print(f"Near/far planes: {NEAR_PLANE} / {far_plane}")
    print(f"Output checkpoint: {output_checkpoint_path}")
    print("=" * 78)

    if not source_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {source_folder}"
        )

    dataset_path = build_strided_dataset(
        source_folder,
        stride=IMAGE_STRIDE,
    )
    transforms = load_transforms(dataset_path / "transforms.json")
    image_count = count_frames(transforms)

    scene_box = SceneBox(
        aabb=torch.tensor(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
    )

    model_config = NerfactoModelConfig(
        hidden_dim=dataset_config["hidden_dim"],
        hidden_dim_color=dataset_config["hidden_dim_color"],
        near_plane=NEAR_PLANE,
        far_plane=far_plane,
    )
    model = NerfactoModel(
        model_config,
        scene_box,
        num_train_data=image_count,
        training=True,
    ).to(DEVICE)

    if stage_number == 2:
        if not rgb_checkpoint_path.exists():
            raise FileNotFoundError(
                "Stage 2 requires the Stage-1 RGB checkpoint:\n"
                f"{rgb_checkpoint_path}"
            )
        load_model_checkpoint(rgb_checkpoint_path, model)

    optimizers, field_optimizer = create_optimizers(
        model,
        stage_config["learning_rates"],
    )
    data_manager = create_data_manager(
        dataset_path,
        RAYS_PER_BATCH,
    )

    height, width = data_manager.train_dataset[0]["image"].shape[:2]
    best_rgb_loss = np.inf
    iou_log: list[tuple[int, float]] = []

    try:
        for epoch in range(stage_config["num_epochs"]):
            model.train()
            ray_batch, image_batch = data_manager.next_train(epoch)
            number_of_rays = ray_batch.origins.shape[0]

            ray_bundle = RayBundle(
                origins=ray_batch.origins.float().to(DEVICE),
                directions=ray_batch.directions.float().to(DEVICE),
                nears=torch.full(
                    (number_of_rays, 1),
                    NEAR_PLANE,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                fars=torch.full(
                    (number_of_rays, 1),
                    far_plane,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                pixel_area=ray_batch.pixel_area.float().to(DEVICE),
                camera_indices=(
                    ray_batch.camera_indices.long().to(DEVICE)
                ),
            )
            batch = {
                "image": image_batch["image"][:, :3].float().to(DEVICE)
            }

            zero_grad_all(optimizers)

            outputs = model.get_outputs(ray_bundle)
            metrics = model.get_metrics_dict(outputs, batch)
            losses = model.get_loss_dict(outputs, batch, metrics)
            total_loss = sum(losses.values())

            total_loss.backward()
            step_all_optimizers(optimizers)

            rgb_loss = losses["rgb_loss"].item()
            displayed_loss_name = (
                "RGB loss" if stage_number == 1 else "mask MSE"
            )

            print(
                f"{dataset_name} | stage {stage_number} | "
                f"epoch {epoch:04d}/{stage_config['num_epochs'] - 1:04d} | "
                f"total {total_loss.item():.5f} | "
                f"{displayed_loss_name} {rgb_loss:.5f}"
            )

            if rgb_loss < best_rgb_loss:
                best_rgb_loss = rgb_loss
                save_checkpoint(
                    model,
                    optimizers,
                    epoch,
                    output_checkpoint_path,
                )

            del (
                outputs,
                metrics,
                losses,
                total_loss,
                ray_bundle,
                batch,
                ray_batch,
                image_batch,
            )

            if epoch % EVAL_EVERY == 0:
                print(
                    "Field learning rate: "
                    f"{field_optimizer.param_groups[0]['lr']}"
                )
                _, iou_value = evaluate_full_image(
                    model=model,
                    data_manager=data_manager,
                    epoch=epoch,
                    height=height,
                    width=width,
                    near_plane=NEAR_PLANE,
                    far_plane=far_plane,
                    iou_metric=iou_metric,
                    dataset_name=dataset_name,
                    stage_name=stage_name,
                )
                iou_log.append((epoch, iou_value))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(
            f"Finished {dataset_name}, stage {stage_number}. "
            f"Best RGB/mask loss: {best_rgb_loss:.6f}"
        )
        print(f"Saved checkpoint: {output_checkpoint_path}")

    finally:
        del model, optimizers, field_optimizer, data_manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    """Run Stage 1 and Stage 2 for every configured dataset."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            "The repository data directory was not found:\n"
            f"{DATA_ROOT}\n"
            "Place the six dataset folders inside InvNeRF-Seg/data/."
        )

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Data directory: {DATA_ROOT}")
    print(f"Checkpoint directory: {CHECKPOINT_ROOT}")

    iou_metric = BinaryJaccardIndex().to(DEVICE)

    for dataset_config in DATASETS:
        for stage_number in (1, 2):
            train_stage(
                base_dir=DATA_ROOT,
                checkpoint_dir=CHECKPOINT_ROOT,
                dataset_config=dataset_config,
                stage_number=stage_number,
                iou_metric=iou_metric,
            )


if __name__ == "__main__":
    main()
