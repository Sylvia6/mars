#!/usr/bin/env bash

set -aux
export CUDA_VISIBLE_DEVICES="0"
# ps -ef | grep nerf  | grep -v grep |  awk '{print "kill -9 "$2}' | sh
# latent_dir="/mnt/intel/data/mrb/nerf/mars/pretrain/VituralKITTI2/car-object-latents"
# latents_path="$latent_dir/latent_codes02.pt"
# car_nerf_state_dict_path="/mnt/intel/data/mrb/nerf/mars/pretrain/VituralKITTI2/car-nerf-state-dict/epoch_805.ckpt"
# semantic_path=/mnt/intel/data/mrb/dataset/vkitti/Scene02/clone/frames/classSegmentation
# python nerfstudio/nerfstudio/scripts/train.py  \
# nsg-vkitti-car-depth-recon-semantic \
# --data /mnt/intel/data/mrb/dataset/vkitti/Scene02/clone \
# --pipeline.datamanager.dataparser.car_object_latents_path  $latents_path \
# --pipeline.datamanager.dataparser.car_nerf_state_dict_path $car_nerf_state_dict_path \
# --pipeline.datamanager.dataparser.semantic_path $semantic_path \
# --experiment_name test \
# --machine.num-gpus 4

# --pipeline.datamanager.dataparser.first-frame  0 \
# --pipeline.datamanager.dataparser.last-frame 237 \
# --load-checkpoint pretrain/nerfstudio_models/step-000520000.ckpt
# --load-dir pretrain/nerfstudio_models
# --pipeline.datamanager.train_num_rays_per_batch 8092 \
# --pipeline.datamanager.eval_num_rays_per_batch 8092
# \--load-dir outputs/clone/nsg-vkitti-car-depth-recon/2023-08-01_104913/nerfstudio_models


# python nerfstudio/nerfstudio/scripts/train.py  \
# nsg-plus-car-depth-recon-semantic \
# --experiment_name trian_success
# --pipeline.datamanager.dataparser.first-frame  0 \
# --pipeline.datamanager.dataparser.last-frame 10

# --pipeline.model.debug_object_pose True \

# scale_factor=0.005

ns-train nsg-plus-car-depth-recon-semantic \
--vis viewer+wandb  \
--data /mnt/intel/data/mrb/dataset/nerf/pdb_b2_benchmark/20221228T111336_pdb-l4e-b0002_20_1to21.db \
--experiment_name pdb_all_image_lilu_left_cam_mean_mask_sky_lr_1e-3 \
# --pipeline.datamanager.dataparser.scale_factor $scale_factor
