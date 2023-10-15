import numpy as np
# import quaternion
import pandas as pd
import pickle
import os
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from datasets.ScanNetDataset import TrainingTuple
import random

root_dir = '/home/graham/datasets/fire'
iou_heap = np.load('/home/graham/datasets/fire/iou_heatmap/train_iou_heatmap.npy')

scene = 'train'

pose_dir = os.path.join(root_dir, scene,'pose')
pose_list = os.listdir(pose_dir)
pose_list.sort()
train_pose_list = np.array(pose_list)
train_poses = []
train_translations = []

for i in range(len(train_pose_list)):
    R = np.loadtxt(os.path.join(root_dir,scene , 'pose', train_pose_list[i]))
    train_poses.append(R)
    # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
    train_translations.append(R[:3, -1])

def gen_tuple(scene):
    # depth_dir = os.path.join(root_dir, scene, 'pointcloud')
    # pose_dir = os.path.join(root_dir,'pose')

    pose_dir = os.path.join(root_dir, scene,'pose')
    pose_list = os.listdir(pose_dir)
    pose_list.sort()
    train_pose_list = np.array(pose_list)
    train_poses = []
    train_translations = []

    for i in range(len(train_pose_list)):
        R = np.loadtxt(os.path.join(root_dir,scene , 'pose', train_pose_list[i]))
        train_poses.append(R)
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        train_translations.append(R[:3, -1])

    pose_list = os.listdir(pose_dir)
    pose_list.sort()
    if scene == 'train':
        pose_list = np.array(pose_list)
        # pose_list = pose_list[:715]
    if scene == 'test':
        # pose_list = pose_list[715:]
        pose_list = np.array(pose_list)

    poses = []
    translations = []

    for i in range(len(pose_list)):
        R = np.loadtxt(os.path.join(root_dir, scene,'pose', pose_list[i]))
        poses.append(R)
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        translations.append(R[:3, -1])

    print(len(translations))
    print(len(train_translations))
    tree = KDTree(translations)
    train_tree = KDTree(train_translations)

    if scene == 'train':
        ind_p = tree.query_radius(translations, r=0.15)
        ind_non = tree.query_radius(translations, r=1)
    else:
        print('test')
        ind_p = train_tree.query_radius(translations, r=0.15)
        ind_non = train_tree.query_radius(translations, r=1)

    queries = {}

    labels = range(len(pose_list))
    for anchor_ndx in range(len(pose_list)):
        anchor_pos = poses[anchor_ndx]
        query = os.path.join('/home/graham/datasets/fire', scene ,'pointcloud_4096', pose_list[anchor_ndx].replace('pose.txt', 'bin'))
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        # timestamp = int(os.path.splitext(scan_filename)[0][6:])
        timestamp = int(os.path.splitext(scan_filename)[0])



        positives = []
        non_negatives = []
        most_positive =  []
        positives = ind_p[anchor_ndx]
        non_negatives = ind_non[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)



        negatives = np.setdiff1d(labels, non_negatives, True)
        if scene == 'train':
            # queries[anchor_ndx] = tmp
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
            #                                     # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][1:51])
            #                                     positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'fire_train_dist.pickle')
        else:
            # queries[anchor_ndx] = tmp
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
            #                                     # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][:50])
            #                                     positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'fire_test_dist.pickle')
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

gen_tuple('train')
gen_tuple('test')



