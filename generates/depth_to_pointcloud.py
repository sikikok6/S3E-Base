import cv2
import numpy as np
import random
import os
import tqdm

# parameters
root_dir = ['/home/graham/datasets/fire1/manolis/']
# depth_intrinsicis_file = 'intrinsics_depth.txt'
# depth_intrinsicis_file_path = os.path.join(root_dir, scenes[0], depth_intrinsicis_file)
scale_factor = 1000
img_width = 640
img_height = 480
cache = {}

# generate indices array
def indices_array_generic(m,n):
    r0 = np.arange(m) # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
    r1 = np.arange(n)
    out = np.empty((m,n,2),dtype=int)
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out

# read depth intrinsics from file
def read_depth_intrinsicis_file(file_name):
    K = np.zeros(shape=(3,3))
    with open(file_name) as f:
        txt = f.readlines()
        fx = float(txt[0].split(' ')[0])
        cx = float(txt[0].split(' ')[2])
        fy = float(txt[1].split(' ')[1])
        cy = float(txt[1].split(' ')[2])
    K[0][0] = fx
    K[0][2] = cx
    K[1][1] = fy
    K[1][2] = cy
    K[-1][-1] = 1
    return  K

K = [[0, 0, 0] for _ in range(3)]
K[0][0] = 572
K[0][2] = 320.0
K[1][1] = 572
K[1][2] = 240.0
K[-1][-1] = 1

def voxel_filter(point_cloud, leaf_size, random=False):
    filtered_points = []
    # 计算边界点
    x_min, y_min, z_min = np.amin(point_cloud, axis=0) #计算x y z 三个维度的最值
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
 
    # 计算 voxel grid维度
    Dx = (x_max - x_min)//leaf_size + 1
    Dy = (y_max - y_min)//leaf_size + 1
    Dz = (z_max - z_min)//leaf_size + 1
    # print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))
 
    # 计算每个点的voxel索引
    h = list()  #h 为保存索引的列表
    for i in range(len(point_cloud)):
        hx = (point_cloud[i][0] - x_min)//leaf_size
        hy = (point_cloud[i][1] - y_min)//leaf_size
        hz = (point_cloud[i][2] - z_min)//leaf_size
        h.append(hx + hy*Dx + hz*Dx*Dy)
    h = np.array(h)
 
    # 筛选点
    h_indice = np.argsort(h) # 返回h里面的元素按从小到大排序的索引
    h_sorted = h[h_indice]
    begin = 0
    for i in range(len(h_sorted)-1):   # 0~9999
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        else:
            point_idx = h_indice[begin: i + 1]
            filtered_points.append(np.mean(point_cloud[point_idx], axis=0))
            begin = i
 
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

# K = read_depth_intrinsicis_file(depth_intrinsicis_file_path)


def convert_depth_image_to_pointclouds(r, depth_image, pose_file, scene):
    img = cv2.imread(os.path.join(r, 'depth', depth_image), cv2.IMREAD_UNCHANGED)

    img_height, img_width = img.shape
    # z [u v 1] = k-1[X Y Z]
    uv = indices_array_generic(img_width, img_height)
    uv = uv.reshape(-1, 2)
    depth_vec = img[uv[:, 1], uv[:, 0]] / scale_factor
    # print(depth_vec[depth_vec > 0])
    depth_mask = depth_vec > 0
    uv = np.hstack([uv, np.ones((uv.shape[0], 1))])
    uv = depth_vec.reshape(-1, 1) * uv
    uv = uv[depth_mask]
    xyz = np.matmul(np.linalg.inv(np.array(K)), uv.T).T
    point_cloud = xyz
    point_cloud = np.hstack([point_cloud, np.ones((point_cloud.shape[0], 1))])

    points = point_cloud[:, :3]
    leaf_size = 0.05
    current_leaf_size = 0.05
    filtered_points = voxel_filter(points, leaf_size)
    init_points = filtered_points
    start = len(filtered_points)
    if start // 100 in cache:
        leaf_size = cache[start // 100]
        current_leaf_size = cache[start // 100]
        filtered_points = voxel_filter(points, leaf_size)
        init_points = filtered_points
    if len(init_points) > 4096:
        print("more")
        while(len(init_points) > 4096):
            current_leaf_size = leaf_size
            if(len(init_points) - 4096 > 700):
                leaf_size += 0.02
            else:
                leaf_size += 0.002
            filtered_points = init_points
            init_points = voxel_filter(points, leaf_size)
        if current_leaf_size - leaf_size == -0.02:
            current_leaf_size += 0.01
            init_points = voxel_filter(points, current_leaf_size)
        while(len(init_points) - 4096 > 300):
            current_leaf_size += 0.001
            filtered_points = init_points
            init_points = voxel_filter(points, current_leaf_size)
            if (len(init_points) < 4096):
                break

    else:
        print("less")
        while(len(init_points) < 4096):
            if leaf_size <= 0:
                leaf_size = current_leaf_size * 0.9
                init_points = voxel_filter(points, leaf_size)
            current_leaf_size = leaf_size
            if(len(init_points) - 4096 > -700):
                leaf_size -= 0.02
            else:
                leaf_size -= 0.002
            init_points = voxel_filter(points, leaf_size)
            filtered_points = init_points
        if current_leaf_size - leaf_size == 0.02:
            current_leaf_size -= 0.01
            init_points = voxel_filter(points, current_leaf_size)
        while(len(init_points) - 4096 > 300):
            current_leaf_size += 0.001
            filtered_points = init_points
            init_points = voxel_filter(points, current_leaf_size)
            if (len(init_points) < 4096):
                break
    print(len(filtered_points))
    cache[start // 100] = current_leaf_size
    choice = random.sample(range(len(filtered_points)), 4096)
    filtered_points = filtered_points[choice]
    filtered_points.tofile(os.path.join(r, "pointcloud_4096", depth_image.replace('depth.png', 'bin')))
    points.tofile(os.path.join(r, "pointcloud", depth_image.replace('depth.png', 'bin')))

for r in root_dir:
    depth_dir = os.path.join(r,  'depth')
    pose_dir = os.path.join(r, 'pose')
    depth_list = os.listdir(depth_dir)
    pose_list = os.listdir(pose_dir)
    depth_list.sort()
    pose_list.sort()

    for i in tqdm.tqdm(range(len(depth_list))):
        convert_depth_image_to_pointclouds(r, depth_list[i], pose_list[i], None)






