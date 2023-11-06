import numpy as np
import os
from scipy import io
import cv2
import time
import torch
from tqdm import tqdm

T_w_c_train_ = []
T_w_c_test_ = []

# pointcloud_dir = '/Users/david/Codes/dataset/train/pointcloud'

train_depth_dir = '/home/graham/datasets/fire/test/depth'
train_pose_dir = '/home/graham/datasets/fire/test/pose'

test_depth_dir = '/home/graham/datasets/fire/train/depth'
test_pose_dir = '/home/graham/datasets/fire/train/pose'

# data = io.loadmat('iou_merge.mat')
# print(data['iou_matrix'].shape)

# pointcloud_path = []
train_depth_img_list = []
test_depth_img_list = []

img_width = 640
img_height = 480

K = [[0.0, 0.0, 0.0] for _ in range(3)]
K[0][0] = 598.84
K[0][2] = 320.0
K[1][1] = 587.62
K[1][2] = 240.0
K[-1][-1] = 1

train_depth_dir_list = os.listdir(train_depth_dir)
test_depth_dir_list = os.listdir(test_depth_dir)

train_depth_dir_list.sort()
test_depth_dir_list.sort()

# depth_dir_list = depth_dir_list[:100]

# pointcloud_path = os.listdir(pointcloud_dir)
# pointcloud_path.sort()

train_pose_dir_list = os.listdir(train_pose_dir)
test_pose_dir_list = os.listdir(test_pose_dir)
train_pose_dir_list.sort()
test_pose_dir_list.sort()

start = time.time()
for pose in tqdm(train_pose_dir_list):
    T_w_c_train_.append(np.loadtxt(os.path.join(train_pose_dir, pose)))

for pose in tqdm(test_pose_dir_list):
    T_w_c_test_.append(np.loadtxt(os.path.join(test_pose_dir, pose)))


print(np.array(T_w_c_train_).shape)
for de_img in tqdm(train_depth_dir_list):
    train_depth_img_list.append(cv2.imread(os.path.join(train_depth_dir, de_img), cv2.IMREAD_UNCHANGED).astype(np.int16))

for de_img in tqdm(test_depth_dir_list):
    test_depth_img_list.append(cv2.imread(os.path.join(test_depth_dir, de_img), cv2.IMREAD_UNCHANGED).astype(np.int16))

end = time.time()
print('loading time: {:.6f}'.format(end - start))

def image_coords_ops_numpy(img_data):
    img_h, img_w = img_data.shape
    nx, ny = (img_w, img_h)
    #x = np.linspace(0, nx - 1, nx)
    #y = np.linspace(0, ny - 1, ny)
    #xv, yv = np.meshgrid(x, y)
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xv, yv = np.meshgrid(x, y)
    coords_uv = np.dstack((xv, yv)).reshape(-1, 2)
    coords_uv = coords_uv.astype(np.int32)
    depth_vec = img_data[coords_uv[:, 1], coords_uv[:, 0]] / 1000.0
    # filter out the zero depth region to avoid numerical error
    depth_vec[depth_vec == 0] = 1e-10
    depth_vec[depth_vec >= 20] = 1e-10

    coords_xy = coords_uv
    ext_c = np.ones((coords_xy.shape[0], 1))
    coords_xy = np.hstack((coords_xy, ext_c))
    coords_xyz = coords_xy * depth_vec.reshape(-1, 1)
    coords_xyz = np.matmul(np.linalg.inv(np.array(K)), coords_xyz.T).T
   # coords_xyc = np.matmul(np.linalg.inv(np.asarray(K)), coords_xy.T).T
   # coords_xyz = np.hstack((coords_xyc[:, :2], depth_vec.reshape(-1, 1)))
    return coords_xyz, coords_uv

def boundary_filter(key_points_uv):
    key_points_u = (key_points_uv[:, 0]>=0) & (key_points_uv[:, 0] < img_width)
    key_points_v = (key_points_uv[:, 1]>=0) & (key_points_uv[:, 1] < img_height)
    key_points_uv_selection = np.where((key_points_u == True) & (key_points_v == True))
    key_points_uv_in = key_points_uv[key_points_uv_selection[0], :]
    return key_points_uv_in, key_points_uv_selection

def z_buffer_filter(depth_img_ref, key_points_uv_in, key_points_xyz_proj, key_points_uv_selection):
    key_points_xyz_proj_transpose = key_points_xyz_proj.T
    key_points_xyz_proj_transpose = key_points_xyz_proj_transpose[key_points_uv_selection[0], :]
    depth_val_proj = key_points_xyz_proj_transpose[:, 2]
    depth_val_ref = depth_img_ref[key_points_uv_in[:, 1], key_points_uv_in[:, 0]]/1000.0
    depth_val_ref[depth_val_ref > 20.0] = 1e-10
    depth_val_ref[depth_val_ref == 0.0] = 1e-10

    depth_buffer_idx = np.where(depth_val_ref >= (depth_val_proj - 0.05))

    if len(depth_buffer_idx[0]) == 0:
        return np.empty(shape=(0, 3))
    else:
        return key_points_uv_in[depth_buffer_idx[0], :]

def image_reproject_and_show(index):
    depth_cur = test_depth_img_list[index]
    T_w_cur = T_w_c_test_[index]
    key_points_xyz, _ = image_coords_ops_numpy(depth_cur)
    depth_cur = depth_cur[(depth_cur > 0) & (depth_cur < 65535)]
    # key_points = np.fromfile(os.path.join(pointcloud_dir, pointcloud_path[index]))
    # key_points_xyz = key_points.reshape(-1, 3)
    ext_c = np.ones((key_points_xyz.shape[0], 1))
    key_points_xyz = np.hstack((key_points_xyz, ext_c))
    iou_list = []
    iou_non_empty_list = []
    intersect_area = 0


    for i in tqdm(range(len(train_depth_img_list))):
        depth_ref = train_depth_img_list[i]
        T_w_ref = T_w_c_train_[i]
        T_ref_cur = np.matmul(np.linalg.inv(T_w_ref), T_w_cur)
        key_points_xyz_proj = np.matmul(T_ref_cur, key_points_xyz.T).T
        key_points_xyz_idx = np.where(key_points_xyz_proj[:, 2] > 0)

        if(len(key_points_xyz_idx[0]) == 0):
            intersect_area = 0
            iou_non_empty_list.append(intersect_area)
            iou_list.append(intersect_area)
            continue
        key_points_xyz_proj = key_points_xyz_proj[key_points_xyz_idx[0], :].T
        key_points_xyz_norm = (key_points_xyz_proj[0:3, :]/key_points_xyz_proj[2, :])

        key_points_uv_proj = np.matmul(K, key_points_xyz_norm).T
        key_points_uv_proj = key_points_uv_proj.astype(np.int8)
        key_points_uv_in, key_points_uv_selection = boundary_filter(key_points_uv_proj)
        key_points_uv_in = z_buffer_filter(depth_ref, key_points_uv_in, key_points_xyz_proj, key_points_uv_selection)
        #if key_points_uv_in.shape[0] != 0:
        #    key_points_uv_in = np.unique(key_points_uv_in, axis=0)
        intersect_area = key_points_uv_in.shape[0]
        depth_ref =depth_ref[(depth_ref > 0) & (depth_ref < 65535)]
        union_area = 2 * img_width * img_height - intersect_area
        union_area_non_empty = depth_cur.shape[0] + depth_ref.shape[0] - intersect_area
        iou = intersect_area / union_area
        iou_non_empty = intersect_area / union_area_non_empty
        iou_list.append(iou)
        iou_non_empty_list.append(iou_non_empty)
    return np.array(iou_list), np.array(iou_non_empty_list)

iou_matrix = []
iou_non_empty_matrix = []
# iou_matrix = data['iou_matrix'].tolist()
# iou_non_empty_matrix = data['iou_non_empty_matrix'].tolist()

accumulate = 0
# start_i = data['iou_matrix'].shape[0]
# start_i = 1062
start_i = 0


for i in tqdm(range(0, len(test_depth_dir_list))):
# for i in range(1062, 1084):
    start = time.time()
    iou_list, iou_non_empty_list = image_reproject_and_show(i)
    iou_matrix.append(iou_list)
    iou_non_empty_matrix.append(iou_non_empty_list)
    end = time.time()
    diff = end - start
    accumulate += diff
    print('running time: {:.6f} average time: {:.6f} remain time:{:.6f}'.format(diff, accumulate / (i + 1 - start_i), accumulate /
                              (i + 1 - start_i) *
                              (len(test_depth_dir_list) - i - 1)))
    iou_matrix_np = np.array(iou_matrix)
    iou_non_empty_matrix_np = np.array(iou_non_empty_matrix)
    print(iou_matrix_np.shape)

    data = {'iou_matrix': iou_matrix_np, 'iou_non_empty_matrix':
            iou_non_empty_matrix_np}

    io.savemat('iou_final_re.mat',data)
