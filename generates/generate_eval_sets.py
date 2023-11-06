# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import scipy
import pandas as pd
from sklearn.neighbors import KDTree
import pickle

from sklearn.neighbors import KDTree
from datasets.ScanNetDataset import TrainingTuple

def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


mat = scipy.io.loadmat('./iou_final.mat')
mat_r = scipy.io.loadmat("./iou_final_re.mat")
iou_mat = mat['iou_matrix']
iou_mat_re = mat_r["iou_matrix"]
print(np.max(iou_mat))
train_iou_mat = np.load('/home/graham/datasets/fire/iou_heatmap/train_iou_heatmap.npy')
test_iou_mat = np.load('/home/graham/datasets/fire/iou_heatmap/val_iou_heatmap.npy')

root_dir = '/home/graham/datasets/fire/'

train_pose_dir = os.path.join(root_dir, 'train','pose')
test_pose_dir = os.path.join(root_dir, 'test','pose')


train_pose_list = os.listdir(train_pose_dir)
train_pose_list.sort()

train_pose_list = train_pose_list[:1000]

test_pose_list = os.listdir(test_pose_dir)
test_pose_list.sort()

test_pose_list = test_pose_list[:1000]

train_poses = []
test_poses = []


for i in range(len(train_pose_list)):
    R = np.loadtxt(os.path.join(root_dir, 'train' , 'pose', train_pose_list[i]))
    train_poses.append(R[:3, -1])
    # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
    # translations.append(R[:3, -1])
    
for i in range(len(test_pose_list)):
    R = np.loadtxt(os.path.join(root_dir, 'test' , 'pose', test_pose_list[i]))
    test_poses.append(R[:3, -1])

tree = KDTree(test_poses)


train_pointcloud_dir = root_dir + 'train/pointcloud_4096/'
test_pointcloud_dir = root_dir + 'test/pointcloud_4096/'
train_pointcloud_list = os.listdir(train_pointcloud_dir)
train_pointcloud_list.sort()
train_pointcloud_list = train_pointcloud_list[:1000]
test_pointcloud_list = os.listdir(test_pointcloud_dir)
test_pointcloud_list.sort()
# train_pointcloud_list = test_pointcloud_list
test_pointcloud_list = train_pointcloud_list[:1000]

def construct_query_and_database_sets():
    database_sets = []
    test_sets = []
    database = {}
    for i in range(len(train_pointcloud_list)):
        database[i] = {'query': os.path.join(train_pointcloud_dir,
                                             train_pointcloud_list[i])}
    print(len(database.keys()))
    database_sets.append(database)
    test = {}
    for i in range(len(test_pointcloud_list)):
        test[i] = {'query': os.path.join(test_pointcloud_dir,
                                             test_pointcloud_list[i])}
    test_sets.append(test)
    for i in range(len(database_sets)):
        for j in range(len(test_sets)):
            for key in range(len(test_sets[j].keys())):
                ind_p = tree.query_radius([test_poses[key].tolist()], r=0.15)
                # ind_non = tree.query_radius(translations, r=1.6)
                index = np.argsort(test_iou_mat[key])[::-1]
                count = 0
                ind = []
                for ii in index:
                    if key == ii:
                        continue
                    if count > 24:
                        break
                    if abs(test_iou_mat[key][ii]  - test_iou_mat[ii][key]) < 0.05:
                        count += 1
                        ind.append(ii)
                        # print(iou_mat[key][ii])


                # test_sets[j][key][i] = index.tolist()[:50]
                #print(ind_p[0].shape)
                # li = ind_p[0].tolist()
                # li.remove(key)
                # test_sets[j][key][i] = li[:20]
                test_sets[j][key][i] = ind
            for key in range(len(database_sets[i].keys())):
                # ind_p = tree.query_radius([test_poses[key].tolist()], r=0.15)
                # ind_non = tree.query_radius(translations, r=1.6)
                index = np.argsort(train_iou_mat[key])[::-1]
                count = 0
                ind = []
                for ii in index:
                    if key == ii:
                        continue
                    if count > 24:
                        break
                    if abs(train_iou_mat[key][ii]  - train_iou_mat[ii][key]) < 0.05:
                        count += 1
                        ind.append(ii)
                        # print(iou_mat[key][ii])


                # test_sets[j][key][i] = index.tolist()[:50]
                #print(ind_p[0].shape)
                # li = ind_p[0].tolist()
                # li.remove(key)
                # test_sets[j][key][i] = li[:20]
                database_sets[j][key][i] = ind
    # print(test_sets)
    output_to_file(database_sets, root_dir + 'pickle',
                   'fire_evaluation_database.pickle')
    output_to_file(test_sets, root_dir + 'pickle',
                   'fire_evaluation_query.pickle')
    

construct_query_and_database_sets()





