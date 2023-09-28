"""Data# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. parser for nerual scene graph plus dataset"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, List

import imageio
import pickle
import numpy as np
import torch
from cv2 import sort
from rich.console import Console
import pandas as pd

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.colors import get_color

CONSOLE = Console(width=120)
_sem2label = {"Misc": -1, "Car": 0, "Van": 0, "Truck": 2, "Tram": 3, "Pedestrian": 4}
camera_ls = [2, 3]

opencv2opengl = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

kitti2vkitti = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_color(path):
    data = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("class="):
                line = line.split("=")[-1].strip()
                for c in "()''":
                    line = line.replace(c, "")
                categories = line.split(", ")
                data['Category'] = [cate[0].upper()+cate[1:] for cate in categories]
            if line.startswith("palette="):
                line = line.split("=")[-1].strip()
                palette_values = eval(line.strip().split("=")[-1])
                data['palette'] = palette_values
    df = pd.DataFrame(data)
    return df
   
def extract_object_information(args, visible_objects, objects_meta):
    """Get object and object network properties for the given sequence

    Args:
        args:
            args.object_setting are experimental settings for object networks inputs, set to 0 for current version
        visible_objects: Objects per frame + Pose and other dynamic properties + tracking ID
        objects_meta: Metadata with additional static object information sorted by tracking ID

    Retruns:
        obj_properties [n_input_frames, n_max_objects, n_object_properties, 0]: Object properties per frame
        add_input_rows: 2
        obj_meta_ls: List of object metadata
        scene_objects: List of objects per frame
        scene_classes: List of object classes per frame
    Notes:
        obj_properties: x,y,z,yaw_angle,track_id, 0
    """
    if args.dataset_type == "vkitti":
        # [n_frames, n_max_obj, xyz+track_id+ismoving+0]
        obj_state = visible_objects[:, :, [7, 8, 9, 2, -1]]

        obj_dir = visible_objects[:, :, 10][..., None]
        # [..., width+height+length]
        # obj_dim = visible_objects[:, :, 4:7]
        sh = obj_state.shape
    elif args.dataset_type == "waymo_od":
        obj_state = visible_objects[:, :, [7, 8, 9, 2, -1]]
        obj_dir = visible_objects[:, :, 10][..., None]
        sh = obj_state.shape
    elif args.dataset_type == "kitti":
        obj_state = visible_objects[:, :, [7, 8, 9, 2, 3]]  # [x,y,z,track_id,class_id]
        obj_dir = visible_objects[:, :, 10][..., None]  # yaw_angle
        sh = obj_state.shape

    # obj_state: [cam, n_obj, [x,y,z,track_id, class_id]]

    # [n_frames, n_max_obj]
    obj_track_id = obj_state[..., 3][..., None]
    obj_class_id = obj_state[..., 4][..., None]
    # Change track_id to row in list(objects_meta)
    obj_meta_ls = list(objects_meta.values())  # object_id, length, height, width, class_id
    # Add first row for no objects
    obj_meta_ls.insert(0, np.zeros_like(obj_meta_ls[0]))
    obj_meta_ls[0][0] = -1
    # Build array describing the relation between metadata IDs and where its located
    row_to_track_id = np.concatenate(
        [
            np.linspace(0, len(objects_meta.values()), len(objects_meta.values()) + 1)[:, None],
            np.array(obj_meta_ls)[:, 0][:, None],
        ],
        axis=1,
    ).astype(np.int32)
    # [n_frames, n_max_obj]
    track_row = np.zeros_like(obj_track_id)

    scene_objects = []
    scene_classes = list(np.unique(np.array(obj_meta_ls)[..., 4]))
    for i, frame_objects in enumerate(obj_track_id):
        for j, camera_objects in enumerate(frame_objects):
            track_row[i, j] = np.argwhere(row_to_track_id[:, 1] == camera_objects)
            if camera_objects >= 0 and not camera_objects in scene_objects:
                # print(camera_objects, "in this scene")
                scene_objects.append(camera_objects)
    CONSOLE.log(f"{scene_objects} in this scene.")

    obj_properties = np.concatenate([obj_state[..., :3], obj_dir, track_row], axis=2)

    if obj_properties.shape[-1] % 3 > 0:
        if obj_properties.shape[-1] % 3 == 1:
            obj_properties = np.concatenate([obj_properties, np.zeros([sh[0], sh[1], 2])], axis=2).astype(np.float32)
        else:
            obj_properties = np.concatenate([obj_properties, np.zeros([sh[0], sh[1], 1])], axis=2).astype(np.float32)

    add_input_rows = int(obj_properties.shape[-1] / 3)

    obj_meta_ls = [
        (obj * np.array([1.0, args.box_scale, 1.0, args.box_scale, 1.0])).astype(np.float32)
        if obj[4] != 4
        else obj * np.array([1.0, 1.2, 1.0, 1.2, 1.0])
        for obj in obj_meta_ls
    ]  
    # [n_obj, [track_id, length * scale, height, width * scale, class_id]]   scale: 1.2 for humans, box_scale for other objects

    return obj_properties, add_input_rows, obj_meta_ls, scene_objects, scene_classes

@dataclass
class PandaDataParserConfig(DataParserConfig):
    """nerual scene graph dataset parser config"""
    _target: Type = field(default_factory=lambda: NSGplus)
    """target class to instantiate"""
    data: Path = Path("/mnt/intel/data/mrb/dataset/nerf/pdb_b2_benchmark/20221228T111336_pdb-l4e-b0002_20_1to21.db") 
    """Directory specifying location of data."""
    scale_factor: float = 1
    depth_scale: float = 1
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    alpha_color: str = "white"
    """alpha color of ground"""
    first_frame: int = 0  
    """specifies the beginning of a sequence if not the complete scene is taken as Input"""
    last_frame: int = 99
    """specifies the end of a sequence"""
    use_object_properties: bool = True
    """ use pose and properties of visible objects as an input """
    object_setting: int = 0
    """specify wich properties are used"""
    obj_opaque: bool = True
    """Ray does stop after intersecting with the first object bbox if true"""
    box_scale: float = 1 
    # 1.5
    """Maximum scale for bboxes to include shadows"""
    novel_view: str = "left"
    use_obj: bool = True 
    render_only: bool = False
    bckg_only: bool = True 
    # False
    use_object_properties: bool = True
    near_plane: float = 0.5
    """specifies the distance from the last pose to the near plane"""
    far_plane: float = 150.0
    """specifies the distance from the last pose to the far plane"""
    dataset_type: str = "kitti"
    obj_only: bool = False
    """Train object models on rays close to the objects only"""
    netchunk: int = 1024 * 64
    """number of pts sent through network in parallel, decrease if running out of memory"""
    chunk: int = 1024 * 32
    """number of rays processed in parallel, decrease if running out of memory"""
    max_input_objects: int = -1
    """Max number of object poses considered by the network, will be set automatically"""
    add_input_rows: int = -1
    use_car_latents: bool = False
    car_object_latents_path: Optional[Path] = Path("pretrain/car_nerf/latent_codes.pt")
    """path of car object latent codes"""
    car_nerf_state_dict_path: Optional[Path] = Path("pretrain/car_nerf/car_nerf.ckpt")
    """path of car nerf state dicts"""
    use_depth: bool = True
    """whether the training loop contains depth"""
    split_setting: str = "reconstruction"
    use_semantic: bool = False
    """path of semantic inputs"""
    semantic_mask_classes: List[str] = field(default_factory=lambda: [])
    """semantic classes that do not generate gradient to the background model"""
    cameras: List[str] = field(default_factory=lambda: ["front_right"])
    # , "front_right"])



@dataclass
class NSGpanda(DataParser):
    """nerual scene graph kitti Dataset"""

    config: PandaDataParserConfig

    def __init__(self, config: NSGplusDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.bag_name = os.path.split(self.data)[-1]
        self.scale_factor: float = config.scale_factor
        self.depth_scale: float = config.depth_scale
        self.alpha_color = config.alpha_color
        self.selected_frames = [config.first_frame, config.last_frame + 1]        
        self.l_sequ = self.selected_frames[1] - self.selected_frames[0]
        self.novel_view = config.novel_view
        self.use_obj = config.use_obj
        self.use_time = False
        self.remove = -1
        self.max_input_objects = -1
        self.render_only = config.render_only
        self.near = config.near_plane
        self.far = config.far_plane
        self.use_object_properties = config.use_object_properties
        self.bckg_only = config.bckg_only
        self.dataset_type = config.dataset_type
        self.time_stamp = None
        self.obj_only = config.obj_only
        self.use_inst_segm = False
        self.netchunk = config.netchunk
        self.chunk = config.chunk
        self.remove_obj = None
        self.debug_local = False
        self.use_depth = config.use_depth
        self.use_semantic = config.use_semantic
        self.cameras = config.cameras

    def _generate_dataparser_outputs(self, split="train"):
        visible_objects_ls = []
        objects_meta_ls = []
        semantic_meta = []
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None
        
        # count frames
        timestamp_path = os.path.join(self.data, "timestamp")
        timestamps = [load_pickle(os.path.join(timestamp_path, name)) for name in os.listdir(timestamp_path)]
        
        # Total number of frames in the scene
        self.n_scene_frames = len(timestamps)

        # load calib info
        calibration_file = os.path.join(self.data, "calib", ("%06d"%0)+".pkl")
        calib = load_pickle(calibration_file)
        
        if self.use_semantic:
            semantic_names = []
            semantics = load_color(f"/mnt/intel/artifact_management/auto-labeling-multi-frame/j7_10-multi-frame_manual_all_bag/000210_20211111T153653_j7-00010_42_1to21.db/color.txt")
            semantics = semantics.loc[~semantics["Category"].isin(self.config.semantic_mask_classes)]
            semantic_meta = Semantics(
                filenames=[],
                classes=semantics["Category"].tolist(),
                colors=torch.tensor(semantics["palette"].tolist(),),
                mask_classes=self.config.semantic_mask_classes,
            )   
                
        # load imu pose        
        poses_imu_w_tracking = self.load_ego_pos() # (n_frames, 4, 4) imu2world
        pose0 = poses_imu_w_tracking[0].copy()
        poses_imu_w_tracking = np.linalg.inv(pose0) @ poses_imu_w_tracking
        
        # selected frame idx
        cam_poses_tracking = []
        intrinsics = []
        for cam_i in self.cameras:
            cam_i_imu = calib[f"Tr_cam_to_imu_{cam_i}"]
            cam_i_w = poses_imu_w_tracking @ cam_i_imu @ opencv2opengl
            cam_i_w = cam_i_w[range(*self.selected_frames)]
            cam_poses_tracking.append(cam_i_w)
            
            if cam_i == 'front_left':
                K = calib['P1'][None,...].repeat(len(cam_i_w),axis=0)   # cam_pose_num * 3 * 4
            elif cam_i == 'front_right':
                K = calib['P2'][None,...].repeat(len(cam_i_w),axis=0)   # cam_pose_num * 3 * 4
            else:
                K = calib[f'P_{cam_i}'][None,...].repeat(len(cam_i_w),axis=0)   # cam_pose_num * 3 * 4
            intrinsics.append(K)
            
        poses = np.concatenate(cam_poses_tracking)
        intrinsics = torch.from_numpy(np.concatenate(intrinsics).astype(np.float32))

        image_names = []
        for i, cam_i in enumerate(self.cameras):
            for frame_no in range(*self.selected_frames):
                image_names.append(os.path.join(self.data, cam_i, ("%06d"%frame_no)+".png"))
                if self.use_semantic:
                    semantic_names.append(os.path.join(self.data, f"{cam_i}_seg", ("%06d"%frame_no)+".png"))  
        
        if self.use_depth:            
            depth_names = [os.path.join(self.data, 'depth_npy', ("%06d"%frame)+".npy") for frame in range(*self.selected_frames)] 
        
        # Calculate the world pose of the object, and center scale
        shift = np.mean(poses[:, :3, 3], axis=0)
        
        visible_objects_, objects_meta_ = self.load_label(poses_imu_w_tracking, shift=shift)
        
        # Load visible_objects and objects_meta label 
        poses[:, :3, 3] -= shift        
        self.scale_factor = 1 / np.abs(poses[:, :3, 3]).max()
        self.config.scale_factor = self.scale_factor
        
        poses = kitti2vkitti @ poses
        visible_objects_[:, :, [9]] *= -1
        visible_objects_[:, :, [7, 8, 9]] = visible_objects_[:, :, [7, 9, 8]]
                
        visible_objects_ls.append(visible_objects_)
        objects_meta_ls.append(objects_meta_)
        
        objects_meta = objects_meta_ls[0]
        N_obj = np.array([len(seq_objs[0]) for seq_objs in visible_objects_ls]).max()
        for seq_i, visible_objects in enumerate(visible_objects_ls):
            diff = N_obj - len(visible_objects[0])
            if diff > 0:
                fill = np.ones([np.shape(visible_objects)[0], diff, np.shape(visible_objects)[2]]) * -1
                visible_objects = np.concatenate([visible_objects, fill], axis=1)
                visible_objects_ls[seq_i] = visible_objects

            if seq_i != 0:
                objects_meta.update(objects_meta_ls[seq_i])
        visible_objects = np.concatenate(visible_objects_ls)

        if visible_objects is not None:
            self.config.max_input_objects = visible_objects.shape[1]
        else:
            self.config.max_input_objects = 0
            

        counts = np.arange(len(visible_objects)).reshape(len(self.cameras), -1)
        i_test = np.array([(idx + 1) % 4 == 0 for idx in counts[0]])    # 1/4 test radio
        i_test = np.tile(i_test, len(self.cameras))
        if self.config.split_setting == "reconstruction":
            i_train = np.ones(len(poses), dtype=bool)
        elif self.config.split_setting == "nvs-75":
            i_train = ~i_test
        elif self.config.split_setting == "nvs-50":
            desired_length = np.shape(counts)[1]
            pattern = np.array([True, True, False, False])
            repetitions = (desired_length + len(pattern) - 1) // len(
                pattern
            )  # Calculate number of necessary repetitions
            repeated_pattern = np.tile(pattern, repetitions)
            i_train = repeated_pattern[:desired_length]  # Slice to the desired length
            i_train = np.tile(i_train, len(self.cameras))
        elif self.config.split_setting == "nvs-25":
            i_train = np.array([idx % 4 == 0 for idx in counts[0]])
            i_train = np.tile(i_train, len(self.cameras))
        else:
            raise ValueError("No such split method")
            
        counts = counts.reshape(-1)
        i_train = counts[i_train]
        i_test = counts[i_test]
        
        
        # novel_view = self.novel_view
        # shift_frame = None
        # n_oneside = int(poses.shape[0] / 2)
        # render_poses = poses[:1]
        # # Novel view middle between both cameras:
        # if novel_view == "mid":
        #     new_poses_o = ((poses[n_oneside:, :, -1] - poses[:n_oneside, :, -1]) / 2) + poses[:n_oneside, :, -1]
        #     new_poses = np.concatenate([poses[:n_oneside, :, :-1], new_poses_o[..., None]], axis=2)
        #     render_poses = new_poses

        # elif novel_view == "shift":
        #     render_poses = np.repeat(np.eye(4)[None], n_oneside, axis=0)
        #     l_poses = poses[:n_oneside, ...]
        #     r_poses = poses[n_oneside:, ...]
        #     render_poses[:, :3, :3] = (l_poses[:, :3, :3] + r_poses[:, :3, :3]) / 2.0
        #     render_poses[:, :3, 3] = (
        #         l_poses[:, :3, 3] + (r_poses[:, :3, 3] - l_poses[:, :3, 3]) * np.linspace(0, 1, n_oneside)[:, None]
        #     )
        #     if shift_frame is not None:
        #         visible_objects = np.repeat(visible_objects[shift_frame][None], len(visible_objects), axis=0)

        # elif novel_view == "left":
        #     render_poses = None
        #     start_i = 0
        #     # Render at trained left camera pose
        #     render_poses = (
        #         poses[start_i : start_i + self.l_sequ, ...]
        #         if render_poses is None
        #         else np.concatenate([render_poses, poses[start_i : start_i + self.l_sequ, ...]])
        #     )
        # elif novel_view == "right":
        #     # Render at trained left camera pose
        #     render_poses = poses[n_oneside:, ...]

        render_objects = None
        
        if self.use_obj:
            start_i = 0
            # Render at trained left camera pose
            render_objects = (
                visible_objects[start_i : start_i + self.l_sequ, ...]
                if render_objects is None
                else np.concatenate([render_objects, visible_objects[start_i : start_i + self.l_sequ, ...]])
            )
        
        if self.use_time:
            time_stamp = np.zeros([len(poses), 3])
            print("TIME ONLY WORKS FOR SINGLE SEQUENCES")
            time_stamp[:, 0] = np.repeat(
                np.linspace(*self.selected_frames, len(poses) // 2)[None], 2, axis=0
            ).flatten()
            render_time_stamp = time_stamp
        else:
            time_stamp = None
            render_time_stamp = None
            
        if visible_objects is not None:
            self.max_input_objects = visible_objects.shape[1]
        else:
            self.max_input_objects = 0

        if self.render_only:
            visible_objects = render_objects 
        
        
        H, W = imageio.imread(image_names[0]).shape[:2]
        
        
        # Extract objects positions and labels
        if self.use_object_properties or self.bckg_only:
            obj_nodes, add_input_rows, obj_meta_ls, scene_objects, scene_classes = extract_object_information(
                self.config, visible_objects, objects_meta
            )
            # obj_nodes: [n_frames, n_max_objects, [x,y,z,yaw_angle,track_id, 0]]
            n_input_frames = obj_nodes.shape[0]
            obj_nodes[..., :3] *= self.scale_factor
            obj_nodes = np.reshape(obj_nodes, [n_input_frames, self.max_input_objects * add_input_rows, 3])
        obj_meta_tensor = torch.from_numpy(np.array(obj_meta_ls, dtype="float32"))  # TODO
        
        
        obj_meta_tensor[..., 1:4] *= self.scale_factor
        poses[..., :3, 3] *= self.scale_factor
        
        
        self.config.add_input_rows = add_input_rows
        if split == "train":
            indices = i_train
        elif split == "val":
            indices = i_test
        elif split == "test":
            indices = i_test
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        
        
        ###############TODO############### Maybe change here, change generate rays
        # print("get rays")
        # # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # # for each pixel in the image. This stack() adds a new dimension.
        # rays = [get_rays_np(image_height, image_width, focal_X, p) for p in poses[:, :3, :4]]
        # rays = np.stack(rays, axis=0)  # [N, 2:ro+rd, H, W, 3]
        # print("done, concats")
        # # [N, ro+rd+rgb, H, W, 3]
        # rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)

        # print("adding object nodes to each ray")
        # rays_rgb_env = rays_rgb
        input_size = 0

        obj_nodes_tensor = torch.from_numpy(obj_nodes)
        # if self.config.fast_loading:
        #     obj_nodes_tensor = obj_nodes_tensor.cuda()
        obj_nodes_tensor = obj_nodes_tensor[:, :, None, ...].repeat_interleave(W, dim=2)
        obj_nodes_tensor = obj_nodes_tensor[:, :, None, ...].repeat_interleave(H, dim=2)

        obj_size = self.max_input_objects * add_input_rows
        input_size += obj_size
        # [N, ro+rd+rgb+obj_nodes, H, W, 3]
        # rays_rgb_env = np.concatenate([rays_rgb_env, obj_nodes], 1)

        # [N, H, W, ro+rd+rgb+obj_nodes*max_obj, 3]
        # with obj_nodes [(x+y+z)*max_obj + (obj_id+is_training+0)*max_obj]
        obj_nodes_tensor = obj_nodes_tensor.permute([0, 2, 3, 1, 4]).cpu()
        # obj_nodes = np.stack([obj_nodes[i] for i in i_train], axis=0)  # train images only
        obj_info = torch.cat([obj_nodes_tensor[i : i + 1] for i in indices], dim=0)

        # """
        # obj_info: n_images * image height * image width * (rays_o, rays_d, rgb, add_input_rows * n_max_obj) * 3
        # add_input_rows = 2 for kitti:
        #     the object info is represented as a 6-dim vector (~2*3, add_input_rows=2):
        #     0~2. x, y, z position of the object
        #     3. yaw angle of the object
        #     4. object id: not track id. track_id = obj_meta[object_id][0]
        #     5. 0 (no use, empty digit)
        # """
        # obj_info = torch.from_numpy(
        #     np.reshape(rays_rgb, [image_n, image_height, image_width, 3 + input_size, 3])[:, :, :, 3:, :]
        # )

                
        image_filenames = [image_names[i] for i in indices]
        depth_filenames = [depth_names[i] for i in indices[: len(indices)//len(self.cameras)]] if self.use_depth else None
        if self.use_semantic:
            semantic_meta.filenames = [semantic_names[i] for i in indices]
        poses = poses[indices]
        intrinsics = intrinsics[indices]
        
        
        
        if self.config.use_car_latents:
            if not self.config.car_object_latents_path.exists():
                CONSOLE.print("[yello]Error: latents not exist")
                exit()
            car_latents = torch.load(str(self.config.car_object_latents_path))
            track_car_latents = {}
            track_car_latents_mean = {}
            for k, idx in enumerate(car_latents["indices"]):
                if self.selected_frames[0] <= idx["fid"] <= self.selected_frames[1]:
                    if idx["oid"] in track_car_latents.keys():
                        track_car_latents[idx["oid"]] = torch.cat(
                            [track_car_latents[idx["oid"]], car_latents["latents"][k].unsqueeze(-1)], dim=-1
                        )
                    else:
                        track_car_latents[idx["oid"]] = car_latents["latents"][k].unsqueeze(-1)
            for k in track_car_latents.keys():
                track_car_latents_mean[k] = track_car_latents[k][..., -1]

        else:
            car_latents = None
        
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        
        cameras = Cameras(
            camera_to_worlds=torch.from_numpy(poses.astype(np.float32))[:, :3, :] ,
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=H,
            width=W,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            mask_filenames=None,
            dataparser_scale=self.scale_factor,
            depth_scale=self.depth_scale*self.scale_factor,
            metadata={
                "depth_filenames": depth_filenames,
                "obj_metadata": obj_meta_tensor if len(obj_meta_tensor) > 0 else None,
                "obj_class": scene_classes if len(scene_classes) > 0 else None,
                "scene_obj": scene_objects if len(scene_objects) > 0 else None,
                "obj_info": obj_info if len(obj_info) > 0 else None,
                "scale_factor": self.scale_factor,
                "semantics": semantic_meta,
            },
        )

        if self.config.use_car_latents:
            dataparser_outputs.metadata.update(
                {
                    "car_latents": track_car_latents_mean,
                    "car_nerf_state_dict_path": self.config.car_nerf_state_dict_path,
                }
            )
        dataparser_outputs.metadata.update(
            {
                "bckg_only": self.bckg_only,
            }
        )

        print("finished data parsing")
        return dataparser_outputs
    
    
    def load_ego_pos(self):
        poses_path = os.path.join(self.data, 'ego_pos_with_vel')
        poses = []
        for frame in range(self.n_scene_frames):            
            info = load_pickle(os.path.join(poses_path, ("%06d"%frame)+".pkl"))
            poses.append(info['ego_pose'])
            
        return np.array(poses).astype(np.float64)
    
    def load_label(self, poses_imu_w_trans, shift=np.zeros(3), moving_threshold=1.0):
        threshold = moving_threshold
        def roty_matrix(yaw):
            c = np.cos(yaw)
            s = np.sin(yaw)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Initialize dictionaries and lists to store object metadata and tracklets
        objects_meta_plus = {}
        objects_meta = {}
        tracklets_ls = []

        # Initialize an array to count the number of objects in each frame
        n_obj_in_frame = np.zeros(self.n_scene_frames)
        
        for frame in range(*self.selected_frames):
            objs = load_pickle(os.path.join(self.data, 'label', ("%06d"%frame)+".pkl"))
            for obj in objs:
                oid, name, meta = int(obj['track_id']), obj['name'], obj['box3d_lidar']
                if name in _sem2label:
                    type = _sem2label[name]
                    if not oid in objects_meta_plus:
                        l, w, h =  meta[3:6]
                        objects_meta_plus[oid] = np.array([float(oid), type, l, h, w])
                """
                The first two elements (frame number and object ID) as float64.
                The object type (converted from the semantic label) as a float.
                The remaining elements of the tracklet (3D position, rotation, and dimensions) as float64.
                """
                is_moving = float(np.linalg.norm(list(obj['velocity'].values())) > threshold)
                tr_array = np.array([
                    frame, oid, type, *meta[[0,1,2,-1]], is_moving
                ])
                tracklets_ls.append(tr_array)
                n_obj_in_frame[frame] += 1
        
        # Convert tracklets to a numpy array
        tracklets_array = np.array(tracklets_ls)
        # Find the maximum number of objects in a frame for the selected frames
        max_obj_per_frame = int(n_obj_in_frame[range(*self.selected_frames)].max())
        # Initialize an array to store visible objects with dimensions [2*(end_frame-start_frame+1), max_obj_per_frame, 14]
        visible_objects = np.ones([self.l_sequ * len(self.cameras), max_obj_per_frame, 14]) * -1.0
        
        # Iterate through the tracklets and process object data
        for tracklet in tracklets_array:
            # tracklet = [frame, oid, type, xyz,yaw, is_moving]
            frame_no = tracklet[0]
            if frame_no in range(*self.selected_frames): 
                obj_id = tracklet[1]
                id_int = int(obj_id)
                obj_type = objects_meta_plus[id_int][1]
                dim = objects_meta_plus[id_int][-3:].astype(np.float32)

                if id_int not in objects_meta:
                    objects_meta[id_int] = np.concatenate(
                        [
                            np.array([id_int]).astype(np.float32),
                            objects_meta_plus[id_int][2:].astype(np.float64),
                            np.array([objects_meta_plus[id_int][1]]).astype(np.float64),
                        ]
                    )
                    
                # Extract object pose data from tracklet
                # Initialize a 4x4 identity matrix for object pose in imu coordinates
                pose_obj_imu = np.eye(4)
                pose_obj_imu[:3, 3] = tracklet[-5:-2] # xyz
                pose_obj_imu[:3, :3] = roty_matrix(tracklet[-2]) # yaw
                
                # Get the IMU pose for the corresponding frame
                pose_obj_w_i = poses_imu_w_trans[int(frame_no)] @ pose_obj_imu 
                pose_obj_w_i[:3, 3] -= shift
                
                # Calculate the approximate yaw angle of the object in the world frame
                yaw_aprox = -np.arctan2(pose_obj_w_i[1, 0], pose_obj_w_i[0, 0])
                
                # Create a 7-element array representing the 3D pose of the object
                is_moving = tracklet[-1]
                
                pose_3d = np.array([pose_obj_w_i[0, 3], pose_obj_w_i[1, 3], pose_obj_w_i[2, 3], yaw_aprox, 0, 0, is_moving])
                

                # Iterate through the available cameras
                for i, cam_i in enumerate(self.cameras):
                    obj = np.array([frame_no, i, obj_id, obj_type, *dim, *pose_3d])
                    frame_cam_id = (int(frame_no) - self.selected_frames[0]) + i * self.l_sequ
                    obj_column = np.argwhere(visible_objects[frame_cam_id, :, 0] < 0).min()
                    visible_objects[frame_cam_id, obj_column] = obj

        return visible_objects, objects_meta


NSGplusDataParserConfigSpecification = DataParserSpecification(config=NSGplusDataParserConfig)

