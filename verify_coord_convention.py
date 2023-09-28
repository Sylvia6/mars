import os
from os import path as osp

import pickle
import numpy as np

import cv2


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')
    
    
class PlusData():
    def __init__(self, 
                 data_dir='/mnt/intel/artifact_management/auto-labeling-multi-frame/j7_10-multi-frame_manual_all_bag/000210_20211111T153653_j7-00010_42_1to21.db', 
                 bag_name=''):
        self.data_dir = data_dir
        self.bag_name = os.path.split(data_dir)[-1]
        self.seg_dir = osp.join("/mnt/intel/data/mrb/dataset/nerf/l4_origin/", bag_name)
        self.bag_dir = osp.join(data_dir, bag_name)
        self.camera_names = ['front_left', 'front_right',]
                            # 'side_left', 'side_right',
                            # 'rear_left', 'rear_right']
    
    def get_item(self, idx):
        # ego pose
        ego_pos_file = osp.join(self.bag_dir, "ego_pos_with_vel", ("%06d"%idx)+".pkl")
        ego_pos_info = load_pickle(ego_pos_file)
        ego_pos = ego_pos_info['ego_pose']
        
        # timestamp
        timestamp_file = osp.join(self.bag_dir, "timestamp", ("%06d"%idx)+".pkl")
        timestamp = load_pickle(timestamp_file)
        
        # lidar
        pts_filename = osp.join(self.bag_dir, "pointcloud", ("%06d"%idx)+".bin")
        
        # label
        # auto_label_path = osp.join(self.bag_dir, "auto_label", ("%06d"%idx)+".pkl")
        # auto_label = load_pickle(auto_label_path)       
        # auto_bboxs = np.array([l['box3d_lidar'] for l in auto_label])
        
        label_path = osp.join(self.bag_dir, "label", ("%06d"%idx)+".pkl")
        label = load_pickle(label_path)     
        bboxs = np.array([l['box3d_lidar'] for l in label])  
        
        # box_xyz = np.array([l['box3d_lidar'][:3] for l in label])
        # box_lwh = np.array([l['box3d_lidar'][3:6] for l in label])
        # box_yaw = np.array([l['box3d_lidar'][6] for l in label])
            
        # points = np.fromfile(pts_filename).reshape(-1, 4)
        # points[:, -1] = 1
        # plot_points(points, bboxs,'tmp_bev')
        # segmentation
        
        # calib
        calib_file  = osp.join(self.bag_dir, "calib", ("%06d"%idx)+".pkl")
        calib = load_pickle(calib_file)
        import pdb; pdb.set_trace()
        
        img_filenames_list = []
        imu2img_list = []
        imu2cam_list = []
        camera_intrinsics_list = []

        # iter cameras
        last_line = np.asarray([[0, 0, 0, 1]])
        for i, camera_name in enumerate(self.camera_names):
            img_filename = osp.join(self.bag_dir, camera_name, ("%06d"%idx)+".png")
            img_filenames_list.append(img_filename)

            imu2cam = np.linalg.inv(
                calib[f'Tr_cam_to_imu_{camera_name}']).astype(np.float64)
            K = np.concatenate([calib[f'P{i+1}'], last_line], axis=0).astype(np.float64)  # 内参
            imu2img = np.dot(K, imu2cam)  
            
            # camera2world = imu2world @ camera2imu
            
            imu2img_list.append(imu2img)
            imu2cam_list.append(imu2cam)
            camera_intrinsics_list.append(K)
            
        imu2img_list = np.stack(imu2img_list, axis=0)
        imu2cam_list = np.stack(imu2cam_list, axis=0)
        camera_intrinsics_list = np.stack(camera_intrinsics_list, axis=0)

        fine_boxs = {'Car':[], 'Truck':[]}
        for l in label:
            fine_boxs[l['name']].append(l['box3d_lidar'])
        # for l in auto_label: 
        #     name, box = l['name'], l['box3d_lidar']
        #     dis, same_idx = 2.0, -1
        #     for i, gt_box in enumerate(fine_boxs[name]):
        #         dis_tmp = np.linalg.norm(gt_box[:2] - box[:2])
        #         if dis_tmp <= dis:
        #             dis = dis_tmp
        #             same_idx = i
        #     if same_idx >= 0:
        #         fine_boxs[name][i][:6] += box[:6]
        #         fine_boxs[name][i][:6] /= 2.0
        #     # else:
        #     #     fine_boxs["Car"].append(box)
                    
        info = dict(
            ego_pos=ego_pos,
            timestamp=timestamp,
            pts_filename=pts_filename,
            img_info=img_filenames_list,
            imu2img=imu2img_list,
            imu2cam=imu2cam_list,
            camera_intrinsics=camera_intrinsics_list,
            camera_names=self.camera_names,
            # auto_bboxs=auto_bboxs,
            bboxs=bboxs,
        )
        fine_boxs = [*fine_boxs['Car'], *fine_boxs['Truck']]
        img = cv2.imread(info['img_info'][0])
        
        # img = plot_rect3d_on_img(img, get_bboxs(fine_boxs), imu2cam_list[0], camera_intrinsics_list[0])
        # img = plot_rect3d_on_img(img, get_bboxs(auto_bboxs), imu2cam_list[0], camera_intrinsics_list[0])
        img, obj_imu = plot_rect3d_on_img(img, bboxs, imu2cam_list[0], camera_intrinsics_list[0], (255,0,0))
        
        obj_w = ego_pos @ obj_imu[0]
        cv2.imwrite(f'tmp/{idx}.png', img)
        print(f'frame {idx}')
        # print([l['track_id'] for l in auto_label])
        return info
        
    
    def len(self):
        return len(os.listdir(osp.join(self.bag_dir, "timestamp")))


def cal_3dbbox(bbox, bev_range, steps):
    x, y, z, l, w, h, yaw = bbox
    yaw_rad = np.radians(yaw)
    
    corners = np.array([
        [l / 2, w / 2],
        [-l / 2, w / 2],
        [-l / 2, -w / 2],
        [l / 2, -w / 2]
    ])
    
    rotation_matrix = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad), np.cos(yaw_rad)]
    ])
    
    rotated_corners = np.dot(corners, rotation_matrix.T)
    x_corners = rotated_corners[:, 0] + x
    y_corners = rotated_corners[:, 1] + y
    
    loc_x = ((x_corners - bev_range[0])/ steps).astype(int)
    loc_y = ((y_corners - bev_range[1])/ steps).astype(int)
    loc_corners = np.array([loc_x, loc_y])
    return loc_corners


def plot_points(points, bboxs=None, file_name=None, bev_range=[-100, -50, 150, 50], canvas=None, color=[0, 255, 0]):
    # Configure the resolution
    steps = 0.1
    # Initialize the plotting canvas
    pixels_x = int((bev_range[2] - bev_range[0]) / steps) + 1
    pixels_y = int((bev_range[3] - bev_range[1]) / steps) + 1
    if canvas is None:
        canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
        canvas.fill(0)

    loc_x1 = ((points[:, 0] - bev_range[0]) / steps).astype(int)
    loc_x1 = np.clip(loc_x1, 0, pixels_x - 1)
    loc_y1 = ((points[:, 1] - bev_range[1]) / steps).astype(int)
    loc_y1 = np.clip(loc_y1, 0, pixels_y - 1)
    canvas[loc_x1, loc_y1] = color
    
    if bboxs is not None:
        for box in bboxs:
            loc_corners = cal_3dbbox(box, bev_range, steps)     
            for i in range(4):
                # cv2.line(canvas, tuple(loc_corners[:, i]), tuple(loc_corners[:, (i+1) % 4]),  [255,255,255], 3)   
                cv2.polylines(canvas, [loc_corners.T.reshape(4,2)[:,::-1]], isClosed=True, color=[0,0,255], thickness=1)
    if file_name is not None:
        cv2.imwrite("%s.png" % file_name, canvas)
    else:
        return canvas

def plot_rect3d_on_img(img,
                       bboxs, imu2cam, K,
                       color=(0, 255, 0),
                       thickness=1):
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    bboxs_imu, obj_imu = get_bboxs(bboxs)
    for bbox_imu in bboxs_imu:
        bbox_cam = bbox_imu @ imu2cam.T
        if (bbox_cam[:, 2] <= 0.0).any():
            continue
        bbox_img = bbox_cam @ K.T
        bbox_img = (bbox_img[:, :2] / bbox_img[:, 2][:,None]).astype(int)
        for start, end in line_indices:
            cv2.line(img, bbox_img[start], bbox_img[end],
                        color, thickness,
                        cv2.LINE_AA)

    return img.astype(np.uint8), obj_imu

def get_bboxs(bboxs):
    res = []
    obj_imu = []
    for bbox in bboxs:
        # Extract bbox parameters
        x, y, z, l, w, h, yaw = bbox
        # Define 3D box vertices
        corners = np.array([
            [l/2, w/2, h/2],
            [l/2, w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, -w/2, h/2],
            [-l/2, w/2, h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, -h/2],
            [-l/2, -w/2, h/2],
        ])
        corners = np.insert(corners, 3, 1, -1 )
        
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        obj_pose_imu = np.eye(4)
        obj_pose_imu[:3, 3] = np.array([x,y,z])
        obj_pose_imu[:3, :3] = rotation_matrix
    
        corners_imu = corners @ obj_pose_imu.T
    
        res.append(corners_imu)
        obj_imu.append(obj_pose_imu)
    return res, np.array(obj_imu)


if __name__ == '__main__':
    data = PlusData(data_dir = '/mnt/intel/data/mrb/dataset/nerf/pdb_b2_benchmark/20221228T111336_pdb-l4e-b0002_20_1to21.db')
    pcd = []
    
    # img = cv2.imread(data.get_item(80)['img_info'][0])[...,::-1]
    # H, W, _ = img.shape

    info = data.get_item(50)    

    # for frame in range(100, data.len(), 20):
    # os.makedirs('tmp', exist_ok=True)
    # for frame in range(0,data.len()):
    #     info = data.get_item(frame)    
        # img = cv2.imread(info['img_info'][0])
        # H, W, _ = img.shape
        # K = info['camera_intrinsics'][0] 
        # imu2world_pose = (np.linalg.inv(mid_pos) @ info['ego_pos'])
        
            
        # """
        #     lidar map verify
        # """
        # pose = info['imu2cam'][0]
        # points = np.fromfile(info['pts_filename']).reshape(-1, 4)
        # points[:, -1] = 1
        # points = points @ imu2world_pose.T
        # # @ info['ego_pos'].T @ np.linalg.inv(first_pos).T
        # # print(info['ego_pos'])
        # pcd.append(points)
        
    
    # pcd = np.concatenate(pcd, 0)
    # plot_points(pcd, None, 'tmp_bev')
    
    """
        camera verify 
    """
    
    # info = data.get_item(100)        
    # K = info['camera_intrinsics'][0] 
    # pose = info['imu2camera'][0] @ info['ego_pos'] # w2imu , imu2cam0
    # img = cv2.imread(info['img_info'][0])[...,::-1]
    
    # pts3d = pcd @ pose.T    # n*3
    # pts3d = pts3d[pts3d[:, 2] > 0.0]
    # pts2d = pts3d @ K.T

    # pts2d = pts2d[:, :2] / pts2d[:, 2][:, None]
    # condition = np.logical_and(
    #     (pts2d[:, 0] < W) & (pts2d[:, 0] > 0),
    #     (pts2d[:, 1] < H) & (pts2d[:, 1] > 0),
    # )
    # pts3d = pts3d[condition]
    # pts2d = pts2d[condition]
    
    
    # from matplotlib import pyplot as plt
    # import matplotlib.cm as cm
    # dis = np.sqrt(np.sum(np.square(pts3d), axis=-1))
    # import pdb;pdb.set_trace()
    # colors = cm.jet(dis / np.max(dis))[:,:3]
    # plt.imshow(img)
    # plt.gca().scatter(pts2d[:, 0], [pts2d[:, 1]], color=colors, s=1, alpha=0.3)
    # plt.savefig('tmp.png')
    # import pdb;pdb.set_trace()
    