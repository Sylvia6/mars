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
                 data_dir='/mnt/intel/artifact_management/auto-labeling-multi-frame/j7_10-multi-frame_manual_all_bag', 
                 bag_name='000210_20211111T153653_j7-00010_42_1to21.db'):
        self.data_dir = data_dir
        self.bag_name = bag_name
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
        label_path = osp.join(self.bag_dir, "auto_label", ("%06d"%idx)+".pkl")
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
        
        img_filenames_list = []
        lidar2img_list = []
        lidar2camera_list = []
        camera_intrinsics_list = []

        # iter cameras
        last_line = np.asarray([[0, 0, 0, 1]])
        for i, camera_name in enumerate(self.camera_names):
            img_filename = osp.join(self.bag_dir, camera_name, ("%06d"%idx)+".png")
            img_filenames_list.append(img_filename)

            rect = np.linalg.inv(
                calib[f'Tr_cam_to_imu_{camera_name}']).astype(np.float64)
            P2 = np.concatenate([calib[f'P_{camera_name}'], last_line], axis=0).astype(np.float64)  # 内参
            lidar2img = np.dot(P2, rect)  
            
            # camera2world = imu2world @ camera2imu
            
            lidar2img_list.append(lidar2img)
            lidar2camera_list.append(rect)
            camera_intrinsics_list.append(P2)
            
        lidar2img_list = np.stack(lidar2img_list, axis=0)
        lidar2camera_list = np.stack(lidar2camera_list, axis=0)
        camera_intrinsics_list = np.stack(camera_intrinsics_list, axis=0)


        info = dict(
            ego_pos=ego_pos,
            timestamp=timestamp,
            pts_filename=pts_filename,
            img_info=img_filenames_list,
            lidar2img=lidar2img_list,
            lidar2camera=lidar2camera_list,
            camera_intrinsics=camera_intrinsics_list,
            camera_names=self.camera_names,
            bboxs=bboxs,
        )
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
        import pdb;pdb.set_trace()
    else:
        return canvas
    
def get_bboxs(bboxs):
    res = []
    for bbox in bboxs:
        # Extract bbox parameters
        x, y, z, l, w, h, yaw = bbox
        # Define 3D box vertices
        corners = np.array([
            [l/2, w/2, h/2],
            [l/2, w/2, -h/2],
            [l/2, -w/2, h/2],
            [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [-l/2, -w/2, -h/2]
        ])
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        rotated_corners = corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([x, y, z])
        res.append(np.insert(translated_corners, 3, 1,-1))
    return res


if __name__ == '__main__':
    data = PlusData()
    pcd = []
    first_pos = data.get_item(80)['ego_pos']
    img = cv2.imread(data.get_item(80)['img_info'][0])[...,::-1]
    H, W, _ = img.shape

    for i in range(100, data.len(), 20):
        info = data.get_item(i)    
        K = info['camera_intrinsics'][0] 
        
        
        bboxs = get_bboxs(info['bboxs'])
        for bboxs_imu in bboxs:
            bboxs_cam = bboxs_imu @ info['ego_pos'].T @ np.linalg.inv(first_pos).T
            # pts3d = pts3d[pts3d[:, 2] > 0.0]
            bboxs_img = bboxs_cam @ K.T
            bboxs_img = (bboxs_img[:, :2] / bboxs_img[:, 2][:,None]).astype(int)
            import pdb;pdb.set_trace()
            for i in range(4):
                cv2.line(img, tuple(bboxs_img[i]), tuple(bboxs_img[(i+1) % 4]), (0, 255, 0), 2)
                cv2.line(img, tuple(bboxs_img[i+4]), tuple(bboxs_img[(i+1) % 4 + 4]), (0, 255, 0), 2)
                cv2.line(img, tuple(bboxs_img[i]), tuple(bboxs_img[i+4]), (0, 255, 0), 2)
        cv2.imwrite('tmp.png', img)
        import pdb;pdb.set_trace
        # bboxs_2d = bboxs_cam@
        
        # pose = info['lidar2camera'][0]
        # img = cv2.imread(info['img_info'][0])[...,::-1]
        # H, W, _ = img.shape
        # points = np.fromfile(info['pts_filename']).reshape(-1, 4)
        # points[:, -1] = 1
        # imu2world_pose = (np.linalg.inv(first_pos) @ info['ego_pos'])
        # points = points @ info['ego_pos'].T @ np.linalg.inv(first_pos).T
        # # print(info['ego_pos'])
        # pcd.append(points)
        
    # # lidar map
    # pcd = np.concatenate(pcd, 0)
    
    # plot_points(pcd, 'tmp_bev')
    
    # # camera
    
    # info = data.get_item(100)        
    # K = info['camera_intrinsics'][0] 
    # pose = info['lidar2camera'][0]
    # img = cv2.imread(info['img_info'][0])[...,::-1]
    
    # pts3d = points @ pose.T    # n*3
    # pts3d = pts3d[pts3d[:, 2] > 0.0]
    # pts2d = pts3d @ K.T
    
    # # pts2d = points @ info['lidar2img'][0].T

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
    # colors = cm.jet(dis / np.max(dis))[:,:3]
    # plt.imshow(img)
    # plt.gca().scatter(pts2d[:, 0], [pts2d[:, 1]], color=colors, s=1, alpha=0.3)
    # plt.savefig('tmp.png')
    