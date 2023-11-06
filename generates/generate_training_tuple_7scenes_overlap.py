import numpy as np
# import quaternion
import pandas as pd
import pickle
import os
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from datasets.ScanNetDataset import TrainingTuple
import random

root_dir = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire'

train_iou_heap = np.load('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/iou_/train_fire_iou.npy')
test_iou_heap = np.load('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/iou_/test_fire_iou.npy')
print(test_iou_heap.shape)


def gen_tuple(scene):
    pcl_dir = os.path.join(root_dir, scene, 'pointcloud_4096')
    pose_dir = os.path.join(root_dir, scene, 'pose')
    pose_list = os.listdir(pose_dir)
    pose_list.sort()
    poses = []
    for p in pose_list:
        R = np.loadtxt(os.path.join(root_dir, scene, 'pose', p))
        poses.append(R)
        # train_translations.append(R[:3, -1])

    pcl_list = os.listdir(pcl_dir)
    pcl_list.sort()
    if scene == 'train':
        iou_heap = train_iou_heap
    else:
        iou_heap = test_iou_heap

    queries = {}
    for anchor_ndx in range(len(pcl_list)):
        # print(anchor_ndx)
        anchor_pos = poses[anchor_ndx]
        query = os.path.join('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire', scene,
                             'pointcloud_4096', pose_list[anchor_ndx].replace('pose.txt', 'bin'))
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        # timestamp = int(os.path.splitext(scan_filename)[0][6:])
        timestamp = int(os.path.splitext(scan_filename)[0][6:])

        labels = list(range(len(iou_heap)))

        positives = []
        hard_ious = []
        hard_positives = []
        non_negatives = []
        most_positive = []
        most_iou = 0
        min_iou = 1
        min_index = -1
        single_iou = 0
        most_single_iou = []
        if scene == 'train':
            n = 1000
        else:
            n = 500

        for ndx in range(len(iou_heap)):

            if ndx == anchor_ndx or (ndx // n) == (anchor_ndx // n):
                non_negatives.append(ndx)
                continue

            if single_iou == 0:
                single_iou = iou_heap[anchor_ndx][ndx]
                most_single_iou.append(ndx)
            if single_iou < iou_heap[anchor_ndx][ndx]:
                single_iou = iou_heap[anchor_ndx][ndx]
                most_single_iou[0] = ndx
            iou_sub = iou_heap[anchor_ndx][ndx] - iou_heap[ndx][anchor_ndx]
            iou_sum = iou_heap[anchor_ndx][ndx] + iou_heap[ndx][anchor_ndx]
            if iou_sub < min_iou:
                min_iou = iou_sub
                min_index = ndx
            if abs(iou_sub) < 0.3:
                if iou_sum > 0.4:
                    if len(most_positive) == 0:
                        most_positive.append(ndx)
                        most_iou = iou_sum
                    if iou_sum > most_iou:
                        most_positive[0] = ndx
                        most_iou = iou_sum
                    positives.append(ndx)
                    hard_ious.append(iou_sum)
                    # if len(hard_positives) < 25:
                    #     hard_positives.append(ndx)
                    non_negatives.append(ndx)
                elif iou_sum > 0.05:
                    non_negatives.append(ndx)
            elif abs(iou_sub) < 0.3:
                non_negatives.append(ndx)
            if (iou_heap[anchor_ndx][ndx] > 0.01 or iou_heap[ndx][anchor_ndx] > 0.01) and (len(non_negatives) == 0 or non_negatives[-1] != ndx):
                non_negatives.append(ndx)

        if len(most_positive) == 0:
            most_positive = most_single_iou
        negatives = np.setdiff1d(labels, non_negatives, True).tolist()

        index = np.argsort(-np.array(hard_ious))
        hard_positives = np.array(positives)[index[:40]].tolist()

        # if len(positives) != 0:
        #     argsort = np.argsort(-iou[positives])
        #     positives = list(np.array(positives)[argsort])
        #     most_positive.append(positives[0])
        # tmp = {'query':query, 'positives':positives, 'negatives': negatives}

        anchor_pos = np.array([])
        if scene == 'train':
            # queries[anchor_ndx] = tmp
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, hard_positives=hard_positives, neighbours=None)
            #                                     # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][1:51])
            #                                     positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(
                root_dir, 'pickle', 'fire_train_overlap.pickle')
        else:
            # queries[anchor_ndx] = tmp
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, hard_positives=hard_positives, neighbours=None)
            #                                     # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][:50])
            #                                     positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(
                root_dir, 'pickle', 'fire_test_overlap.pickle')
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)


gen_tuple('train')
gen_tuple('test')
