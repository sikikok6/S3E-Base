import numpy as np
import torch
import os
from scipy import io
import cv2
import tqdm
import time

T_w_c_ = []
pointcloud_dir = '/home/graham/datasets/apt2/luke/pointcloud'
depth_dir = '/home/graham/datasets/apt2/luke/depth'
pose_dir = '/home/graham/datasets/apt2/luke/pose'
pointcloud_path = []
depth_img_list = []

img_width = 640
img_height = 480

K = [[0, 0, 0] for _ in range(3)]
K[0][0] = 572.0
K[0][2] = 320.0
K[1][1] = 572.0
K[1][2] = 240.0
K[-1][-1] = 1
K = torch.Tensor(K).to('cuda')

depth_dir_list = os.listdir(depth_dir)
depth_dir_list.sort()
#  depth_dir_list = depth_dir_list[:100]
pointcloud_path = os.listdir(pointcloud_dir)
pointcloud_path.sort()
pose_dir_list = os.listdir(pose_dir)
pose_dir_list.sort()
print(len(pose_dir_list))

start = time.time()

for pose in pose_dir_list:
    T_w_c_.append(np.loadtxt(os.path.join(pose_dir, pose)))
print(len(T_w_c_))
T_w_c_ = torch.Tensor(T_w_c_)

print(T_w_c_.shape)
for de_img in depth_dir_list:
    depth_img_list.append(cv2.imread(os.path.join(depth_dir, de_img), cv2.IMREAD_UNCHANGED))
depth_img_list = torch.Tensor(depth_img_list)

end = time.time()
print('loading time: {:.6f}'.format(end - start))

def boundary_filter(key_points_uv):
    key_points_u = (key_points_uv[:, 0]>=0) & (key_points_uv[:, 0] < img_width)
    key_points_v = (key_points_uv[:, 1]>=0) & (key_points_uv[:, 1] < img_height)
    key_points_uv_selection = torch.where((key_points_u == True) & (key_points_v == True))
    key_points_uv_in = key_points_uv[key_points_uv_selection[0], :]
    return key_points_uv_in, key_points_uv_selection

def z_buffer_filter(depth_img_ref, key_points_uv_in, key_points_xyz_proj, key_points_uv_selection):
    key_points_xyz_proj_transpose = key_points_xyz_proj.T
    #print("The shape of key_points_xyz_proj is: ")
    #print(key_points_xyz_proj_transpose.shape)
    key_points_xyz_proj_transpose = key_points_xyz_proj_transpose[key_points_uv_selection[0], :]
    depth_val_proj = key_points_xyz_proj_transpose[:, 2]
    #print("The depth_val_proj is: ")
    #print(depth_val_proj)
    #print("The shape of depth_val_proj is: ")
    #print(depth_val_proj.shape)
    depth_val_ref = depth_img_ref[key_points_uv_in[:, 1], key_points_uv_in[:, 0]] / 1000.0
    depth_val_ref[depth_val_ref > 20.0] = 1e-10
    depth_val_ref[depth_val_ref == 0.0] = 1e-10
    #print("The depth val ref is: ")
    #print(depth_val_ref)
    #print("The shape of depth_val_ref is:")
    #print(depth_val_ref.shape)

    depth_buffer_idx = torch.where(depth_val_ref >= (depth_val_proj - 0.05))
    #depth_buffer_idx = np.where(depth_buffer_bool == True)
    #print("The depth_buffter_idx is: ")
    #print(depth_buffer_idx)

    if len(depth_buffer_idx[0]) == 0:
        return torch.empty(size=(0, 3))
    else:
        #print("The key_points_uv_in is: ")
        return key_points_uv_in[depth_buffer_idx[0], :]

def image_reproject_and_show(index):
    depth_cur = depth_img_list[index].to('cuda')
    # depth_cur = depth_img_list[index].copy()
    depth_cur =depth_cur[(depth_cur > 0) & (depth_cur < 65535)]
    # color_cur = self.color_img_list[num_iter].copy()
    T_w_cur = T_w_c_[index].to('cuda')
    # key_points_xyz, key_points_uv_ = self.image_coords_ops_numpy(depth_cur)
    #key_points_xyz = cp.asnumpy(key_points_xyz)
    key_points = torch.Tensor(np.fromfile(os.path.join(pointcloud_dir, pointcloud_path[index]))).to('cuda')
    key_points_xyz = key_points.reshape(-1, 3)
    ext_c = torch.ones((key_points_xyz.shape[0], 1)).to('cuda')
    key_points_xyz = torch.hstack((key_points_xyz, ext_c)).to('cuda')
    iou_list = []
    iou_non_empty_list = []
    intersect_area = 0


    for i in range(len(depth_img_list)):
        # depth_ref = depth_img_list[i].copy()
        depth_ref = depth_img_list[i].to('cuda')
        T_w_ref = T_w_c_[i].to('cuda')
        T_ref_cur = torch.matmul(torch.linalg.inv(T_w_ref), T_w_cur)

        key_points_xyz_proj = torch.matmul(T_ref_cur, key_points_xyz.T).T
        key_points_xyz_idx = torch.where(key_points_xyz_proj[:, 2] > 0)

        if(len(key_points_xyz_idx[0]) == 0):
            intersect_area = 0
            iou_list.append(intersect_area)
            iou_non_empty_list.append(intersect_area)
            continue
        key_points_xyz_proj = key_points_xyz_proj[key_points_xyz_idx[0], :].T
        key_points_xyz_norm = (key_points_xyz_proj[0:3, :]/key_points_xyz_proj[2, :])

        key_points_uv_proj = torch.matmul(K, key_points_xyz_norm).T
        key_points_uv_proj = key_points_uv_proj.long()
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
    #return iou_list
        #merge_img = self.key_points_show(key_points_uv_, key_points_uv_in, color_cur, color_ref, mode = "iou")
        #print("saving " + str(i) + " image !")
        #cv2.imwrite(file_dir + str(i) +".png", merge_img)
    return np.array(iou_list), np.array(iou_non_empty_list)

# data = io.loadmat('./iou_fire1_gates362_gpu.mat')
iou_matrix = []
iou_non_empty_matrix = []
# iou_matrix = data['iou_matrix'].tolist()
# iou_non_empty_matrix = data['iou_non_empty_matrix'].tolist()

accumulate = 0

for i in tqdm.tqdm(range(len(iou_matrix), len(depth_dir_list))):
    # start = time.time()
    iou_list, iou_non_empty_list = image_reproject_and_show(i)
    iou_matrix.append(iou_list)
    iou_non_empty_matrix.append(iou_non_empty_list)
    # end = time.time()
    # diff = end - start
    # accumulate += diff
    # print('running time: {:.6f} average time: {:.6f} remain time:{:.6f}'.format(diff, accumulate / (i + 1), accumulate / (i + 1) * (len(depth_dir_list) - i - 1)))
    iou_matrix_np = np.array(iou_matrix)
    iou_non_empty_matrix_np = np.array(iou_non_empty_matrix)

    data = {'iou_matrix': iou_matrix_np, 'iou_non_empty_matrix':
            iou_non_empty_matrix_np}

    io.savemat('iou_apt2_luke_gpu.mat',data)
