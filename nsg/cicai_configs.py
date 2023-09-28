from __future__ import annotations

from pathlib import Path
from typing import Dict

import tyro

from nsg.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nsg.data.nsg_datamanager import NSGkittiDataManagerConfig
from nsg.data.nsg_dataparser import NSGkittiDataParserConfig
from nsg.data.nsg_plus_dataparser import NSGplusDataParserConfig
from nsg.data.nsg_vkitti_dataparser import NSGvkittiDataParserConfig
from nsg.models.car_nerf import CarNeRF, CarNeRFModelConfig
from nsg.models.mipnerf import MipNerfModel
from nsg.models.nerfacto import NerfactoModelConfig
from nsg.models.scene_graph import SceneGraphModelConfig
from nsg.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nsg.nsg_pipeline import NSGPipelineConfig

from nerfstudio.configs.base_config import ViewerConfig

MAX_NUM_ITERATIONS = 20000
# 600000
STEPS_PER_SAVE = 2000
STEPS_PER_EVAL_IMAGE = 50 
# 200
STEPS_PER_EVAL_ALL_IMAGES = 5000


VKITTI_Recon_NSG_Car_Depth_Semantic = MethodSpecification(
    config=TrainerConfig(
        method_name="nsg-vkitti-car-depth-recon-semantic",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        max_num_iterations=MAX_NUM_ITERATIONS,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    use_semantic=True,
                    semantic_mask_classes=['Van', 'Undefined'],
                    car_object_latents_path=Path(
                        "/mnt/intel/jupyterhub/lilu/code/mars/checkpoints/Vitural KITTI 2/car-object-latents/latent_codes18.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/mnt/intel/jupyterhub/lilu/code/mars/checkpoints/Vitural KITTI 2/car-nerf-state-dict/epoch_805.ckpt"),
                    semantic_path=Path("/mnt/intel/data/mrb/dataset/vkitti/Scene18/clone/frames/classSegmentation")
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=SemanticNerfWModelConfig(
                    num_proposal_iterations=1,
                    num_proposal_samples_per_ray=[48],
                    num_nerf_samples_per_ray=97,
                    use_single_jitter=False,
                    semantic_loss_weight=0.1
                ),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph with semantic learning for the backgruond model.",
)


PLUS_Recon_NSG_Car_Depth_Semantic = MethodSpecification(
    config=TrainerConfig(
        method_name="nsg-plus-car-depth-recon-semantic",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        max_num_iterations=MAX_NUM_ITERATIONS,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGplusDataParserConfig(
                    use_depth=True,
                    use_semantic=True,
                    semantic_mask_classes=['Undefined'],
                    split_setting="reconstruction",
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                # camera_optimizer=CameraOptimizerConfig(
                #     mode="SO3xR3",
                #     optimizer=RAdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                #     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                # ),
            ),
            model=SceneGraphModelConfig(
                background_model=SemanticNerfWModelConfig(
                    # num_proposal_iterations=1,
                    # num_proposal_samples_per_ray=[48],
                    # num_nerf_samples_per_ray=97,
                    use_single_jitter=False,
                    semantic_loss_weight=0.1
                ),
                # object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                # "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph with semantic learning for the backgruond model.",
)

KITTI_Recon_NSG_Car_Depth_Semantic = MethodSpecification(
    config=TrainerConfig(
        method_name="nsg-kitti-car-depth-recon-semantic",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        max_num_iterations=MAX_NUM_ITERATIONS,
        save_only_latest_checkpoint=False,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    use_semantic=True,
                    car_object_latents_path=Path(
                        "/mnt/intel/jupyterhub/lilu/code/mars/checkpoints/KITTI-MOT/car-object-latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/mnt/intel/jupyterhub/lilu/code/mars/checkpoints/KITTI-MOT/car-nerf-state-dict/epoch_670.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=SemanticNerfWModelConfig(
                    # num_proposal_iterations=1,
                    # num_proposal_samples_per_ray=[48],
                    # num_nerf_samples_per_ray=97,
                    # use_single_jitter=False,
                    semantic_loss_weight=0.1
                ),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph with semantic implementation with vanilla-NeRF model for backgruond and object models.",
)

KITTI_Recon_NSG_Car_Depth = MethodSpecification(
    config=TrainerConfig(
        method_name="nsg-kitti-car-depth-recon",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        max_num_iterations=MAX_NUM_ITERATIONS,
        save_only_latest_checkpoint=False,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/data1/chenjt/datasets/ckpts/pretrain/car_nerf/latent_codes_car_van_truck.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/data1/chenjt/datasets/ckpts/pretrain/car_nerf/epoch_670.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "learnable_global": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_no_depth_kitti = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-no-depth-kitti",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        max_num_iterations=MAX_NUM_ITERATIONS,
        save_only_latest_checkpoint=False,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=False,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/latent_codes_car_van_truck.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/epoch_670.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "learnable_global": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)


KITTI_NVS_NSG_Car_Depth = MethodSpecification(
    config=TrainerConfig(
        method_name="nsg-kitti-car-depth-nvs",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        max_num_iterations=MAX_NUM_ITERATIONS,
        save_only_latest_checkpoint=False,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/data41/luoly/kitti_mot/latents/latent_codes06.pt"
                    ),
                    split_setting="nvs-75",
                    car_nerf_state_dict_path=Path("/data1/chenjt/datasets/ckpts/pretrain/car_nerf/epoch_670.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=NerfactoModelConfig(),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "learnable_global": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

VKITTI_Recon_NSG_Car_Depth = MethodSpecification(
    config=TrainerConfig(
        method_name="nsg-vkitti-car-depth-recon",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                        # "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes02.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

VKITTI_NVS_NSG_Car_Depth = MethodSpecification(
    config=TrainerConfig(
        method_name="nsg-vkitti-car-depth-nvs",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="nvs-75",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_BG_NeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-background-nerf",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=VanillaModelConfig(
                    _target=NeRFModel,
                    num_coarse_samples=32,
                    num_importance_samples=64,
                ),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_BG_MipNeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-background-mip-nerf",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=VanillaModelConfig(
                    _target=MipNerfModel,
                    num_coarse_samples=48,
                    num_importance_samples=96,
                ),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_object_wise_NeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-object-wise-nerf",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=VanillaModelConfig(
                    _target=NeRFModel,
                    num_coarse_samples=32,
                    num_importance_samples=64,
                ),
                object_representation="object-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_object_wise_MipNeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-object-wise-mip-nerf",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=VanillaModelConfig(
                    _target=MipNerfModel,
                    num_coarse_samples=48,
                    num_importance_samples=96,
                ),
                object_representation="object-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_object_wise_NeRFacto = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-object-wise-nerfacto",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=False,
                    use_depth=True,
                    split_setting="reconstruction",
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=NerfactoModelConfig(),
                object_representation="object-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_class_wise_NeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-class-wise-nerf",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=VanillaModelConfig(
                    _target=NeRFModel,
                    num_coarse_samples=32,
                    num_importance_samples=64,
                ),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_class_wise_MipNeRF = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-class-wise-mip-nerf",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=VanillaModelConfig(
                    _target=MipNerfModel,
                    num_coarse_samples=48,
                    num_importance_samples=96,
                ),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_class_wise_NeRFacto = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-class-wise-nerfacto",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=NerfactoModelConfig(),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_warmup = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-warmup",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="warmup",
                object_warmup_steps=5000,
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_none_ray_sample = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-none-ray-sample",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=True,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="none",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)

Ablation_no_depth = MethodSpecification(
    config=TrainerConfig(
        method_name="ablation-no-depth",
        steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
        steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
        steps_per_save=STEPS_PER_SAVE,
        save_only_latest_checkpoint=False,
        max_num_iterations=MAX_NUM_ITERATIONS,
        mixed_precision=False,
        use_grad_scaler=True,
        log_gradients=True,
        pipeline=NSGPipelineConfig(
            datamanager=NSGkittiDataManagerConfig(
                dataparser=NSGvkittiDataParserConfig(
                    use_car_latents=True,
                    use_depth=False,
                    car_object_latents_path=Path(
                        "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/latents/latent_codes06.pt"
                    ),
                    split_setting="reconstruction",
                    car_nerf_state_dict_path=Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/vkitti/epoch_805.ckpt"),
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            model=SceneGraphModelConfig(
                background_model=NerfactoModelConfig(),
                object_model_template=CarNeRFModelConfig(_target=CarNeRF),
                object_representation="class-wise",
                object_ray_sample_strategy="remove-bg",
            ),
        ),
        optimizers={
            "background_model": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "object_model": {
                "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Neural Scene Graph implementation with vanilla-NeRF model for backgruond and object models.",
)
