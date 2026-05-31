#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 03:16:04 2026

@author: nibio
"""




import os


import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from pathlib import Path
import torch
from jaxtyping import Float
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
import os


# Nerfacto-Inspired Implementation in Python (Updated)
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nerfstudio.data.scene_box import SceneBox
import torch
import torch.optim as optim
import torch.nn.functional as F


## Nerfacto MLP based on nerfstudio components



from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
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
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses



@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = True
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    pass_semantic_gradients: bool = False
    """Whether to pass semantic gradients."""
    use_semantics: bool = True
    """Whether to use semantics."""
    num_semantic_classes: int = 1
    """Number of semantic classes."""
    semantic_loss_weight: float = 1.0
    """Number of semantic classes."""





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
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            # semantics
            use_semantics = self.config.use_semantics,
            num_semantic_classes = self.config.num_semantic_classes,
            pass_semantic_gradients = self.config.pass_semantic_gradients,
            #
            implementation=self.config.implementation,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()
        self.renderer_semantics = SemanticRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.step = 0
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
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
        # print("field_outputs[FieldHeadNames.RGB] {}".format(field_outputs[FieldHeadNames.RGB].shape))
        # print("weights {}".format(weights.shape))
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        # print("rgb {}".format(rgb.shape))
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        
        ## semantics
        semantic_weights = weights
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )
        # semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = semantic_labels
        return outputs
    
    def get_export_outputs(self, ray_bundle: RayBundle):
        ray_samples, _, _ = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
    
        field_outputs = self.field.forward(ray_samples, compute_normals=False)
    
        density = field_outputs[FieldHeadNames.DENSITY][..., 0]
        weights = ray_samples.get_weights(density.unsqueeze(-1))[..., 0]
    
        outputs = {
            "point_location": ray_samples.frustums.get_positions(),  # [R,S,3]
            "weights": weights,                                      # [R,S]
            "density": density,                                      # [R,S]
            "semantic_prob": torch.sigmoid(
                field_outputs[FieldHeadNames.SEMANTICS][..., 0]
            ),
            "rgb": field_outputs[FieldHeadNames.RGB],
        }
        return outputs

    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = 1*self.rgb_loss(gt_rgb, pred_rgb)
        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.binary_cross_entropy_loss(
            outputs["semantics"], batch["fruit_mask"]
        )
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i
        # semantics
        semantic_labels = torch.sigmoid(outputs["semantics"])
        images_dict[
            "semantics_colormap"] = semantic_labels

        # valid mask
        images_dict["fruit_mask"] = batch["fruit_mask"].repeat(1, 1, 3).to(self.device)
        # batch["fruit_mask"][batch["fruit_mask"] < 0.1] = 0
        # batch["fruit_mask"][batch["fruit_mask"] >= 0.1] = 1

        from torchmetrics.classification import BinaryJaccardIndex
        metric = BinaryJaccardIndex().to(self.device)
        semantic_labels = torch.nn.functional.softmax(outputs["semantics"])
        iou = metric(semantic_labels[..., 0], batch["fruit_mask"][..., 0])
        metrics_dict["iou"] = float(iou)
        
        return metrics_dict, images_dict


from typing import Optional, Tuple, List, Union, Callable

## 5. Ray Sampling
from typing import Tuple

from dataclasses import dataclass
from typing import Optional, Union
from torch import Tensor
from typing import List
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np


import torch


def load_checkpoint_multiple(checkpoint_path, model):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model's state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully.")

    # Optionally, you can print or log other details from the checkpoint
    epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {epoch}.")

    # Return the epoch to continue training
    return epoch



import json

def load_transforms(file_path):
    """Load camera poses from a transforms.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data



from pathlib import Path
from PIL import Image


import cv2

# convert 3 channel png mask to 1 channel till mask
# input_directory = Path('data/FruitNeRF_Syn/06_mango_tree_1024x1024_#300/semantics_old')
# output_directory = Path('data/FruitNeRF_Syn/06_mango_tree_1024x1024_#300/semantics')
# convert_png_to_tiff_binary(input_directory, output_directory)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def visualize_point_cloud(points, title="3D Point Cloud", colors=None):
    """
    points: (N, 3) numpy array
    colors: None or (N,) or (N,3) array
    """
    e_i_ls = [30, 0, 0]
    a_i_ls = [0, 90, 180]

    for elev, azim in zip(e_i_ls, a_i_ls):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        if colors is None:
            ax.scatter(x, y, z, s=1)
        else:
            ax.scatter(x, y, z, c=colors, s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title} (elev={elev}, azim={azim})")

        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()







# folder_ls = ["appleTree", "peachTree"]
folder_ls = ["peachTree"]


for folder_i in folder_ls:
    print("folder_i = {}".format(folder_i))


    img_num = len(load_transforms(f"{folder_i}/transforms.json")["frames"])
    
    
    nerfcfig = NerfactoModelConfig()
    
    # Create the SceneBox
    scale_factor = 1
    ab = 1
    min_bound = torch.tensor([-ab*scale_factor, -ab*scale_factor, -ab*scale_factor])
    max_bound = torch.tensor([ab*scale_factor, ab*scale_factor, ab*scale_factor])
    
    scene_box = SceneBox(aabb=torch.stack([1*min_bound, 1*max_bound], dim=0))
    ##
    
    ##
    near_plane = 0.01#nerfcfig.near_plane  # e.g., 0.05
    far_plane = 1000 #nerfcfig.far_plane    # e.g., 10.0
    
    nerfmodel = NerfactoModel(nerfcfig, scene_box, num_train_data=img_num, training = True).cuda()
    
    ##
    loss_type = "mseMaskLoss"# "mseMaskLoss", "BCEMaskLoss"
        
    checkpoint_path = f"best_checkpoint_nerfacto{folder_i}FruitNeRF_April2026_defaultConfig_001_{loss_type}.pth"
    load_checkpoint_multiple(checkpoint_path, nerfmodel)
    ##
    
    
    ##
    from pathlib import Path
    import random
    #
    # from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
    # they are two independent modules now, don't need to import anymore
    from nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
    data_config = NerfstudioDataParserConfig()
    data_config.data = Path('{}'.format(folder_i))
    data_config.downscale_factor = 1
    data_config.train_split_fraction = 0.9
    # data_config.eval_model = "all"
    #
    dataparser = Nerfstudio(data_config)
    ###
    
    # from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
    from parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
    
    dataManager_config = ParallelDataManagerConfig()
    dataManager_config.dataparser.data = Path('{}'.format(folder_i))
    # dataManager_config.dataparser.data = 0.5
    dataManager_config.dataparser = data_config
    dataManager_config.train_num_rays_per_batch = 1024*5
    dataManager_config.eval_num_rays_per_batch = 1024*5
    dataManager_config.camera_res_scale_factor = 1
    
    dataManager = ParallelDataManager(dataManager_config)
    
    
    
    
    # -------------------------------------------------
    # CONFIG
    # -------------------------------------------------
    NUM_BATCHES = 500
    near_plane = 0.01
    far_plane = 1000.0
    
    SEM_TH = 0.65
    W_TH   = 0.08
    DEN_TH = None

    MIN_BOUND = torch.tensor([-0.5, -0.5, -0.7])
    MAX_BOUND = torch.tensor([ 0.5,  0.5,  0.2])
    X_LIM = (-0.5, 0.5)
    Y_LIM = (-0.5, 0.5)
    Z_LIM = (-0.7, 0.7)
    nerfmodel.eval()
    device = nerfmodel.device
    
    fruit_points_all = []
    
    with torch.no_grad():
        for batch_i in range(NUM_BATCHES):
    
            # ---------------------------------------
            # GET RAYS
            # ---------------------------------------
            ray_bundle_train, _ = dataManager.next_train(0)
    
            ray_bundle = RayBundle(
                origins=ray_bundle_train.origins.float(),
                directions=ray_bundle_train.directions.float(),
                nears=torch.full(
                    (ray_bundle_train.origins.shape[0], 1),
                    near_plane,
                ),
                fars=torch.full(
                    (ray_bundle_train.origins.shape[0], 1),
                    far_plane,
                ),
                pixel_area=ray_bundle_train.pixel_area.float(),
                camera_indices=ray_bundle_train.camera_indices.long(),
            ).to(device)
    
            # ---------------------------------------
            # MODEL OUTPUTS
            # ---------------------------------------
            outputs = nerfmodel.get_export_outputs(ray_bundle)
    
            pos = outputs["point_location"]     # [R, S, 3]
            w   = outputs["weights"]             # [R, S]
            den = outputs["density"]             # [R, S]
            sem = outputs["semantic_prob"]       # [R, S]
    
            # ---------------------------------------
            # SURFACE + SEMANTIC MASK (NO ARGMAX)
            # ---------------------------------------
            mask = (w > W_TH) & (sem > SEM_TH)
    
            if DEN_TH is not None:
                mask &= (den > DEN_TH)
    
            # ---------------------------------------
            # FLATTEN
            # ---------------------------------------
            pts = pos[mask]
    
            if pts.numel() == 0:
                continue
    
            # ---------------------------------------
            # OPTIONAL BOUNDING BOX
            # ---------------------------------------
            bb_mask = (
                (pts[:, 0] >= MIN_BOUND[0].to(device)) &
                (pts[:, 0] <= MAX_BOUND[0].to(device)) &
                (pts[:, 1] >= MIN_BOUND[1].to(device)) &
                (pts[:, 1] <= MAX_BOUND[1].to(device)) &
                (pts[:, 2] >= MIN_BOUND[2].to(device)) &
                (pts[:, 2] <= MAX_BOUND[2].to(device))
            )
    
            pts = pts[bb_mask]
    
            if pts.numel() == 0:
                continue
    
            pts_np = pts.cpu().numpy()
    
            # show every 20 batches to avoid overload
            # =====================================================
            # 🔍 VISUALIZATION (PER BATCH)
            # =====================================================
            if batch_i % 20 == 0:
                e_i_ls = [30, 0, 0]
                a_i_ls = [0, 90, 180]
            
                for elev, azim in zip(e_i_ls, a_i_ls):
                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(111, projection='3d')
            
                    x, y, z = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2]
                    ax.scatter(x, y, z, s=1)
            
                    # ✅ FIXED AXIS LIMITS
                    ax.set_xlim(X_LIM)
                    ax.set_ylim(Y_LIM)
                    ax.set_zlim(Z_LIM)
            
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
            
                    ax.set_title(
                        f"FruitNeRF batch {batch_i} "
                        f"(elev={elev}, azim={azim})"
                    )
            
                    ax.view_init(elev=elev, azim=azim)
                    plt.tight_layout()
                    plt.show()
            
                
            # ---------------------------------------
            # COLLECT
            # ---------------------------------------
            fruit_points_all.append(pts_np)
    
    # ---------------------------------------
    # SAVE POINT CLOUD
    # ---------------------------------------
    fruit_points_all = np.concatenate(fruit_points_all, axis=0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fruit_points_all)
    
    white = np.ones((fruit_points_all.shape[0], 3), dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(white)
    
    o3d.io.write_point_cloud(
        f"{folder_i}_fruitNerf_density_001_April2026_{loss_type}.ply",
        pcd
    )
    
        
    






