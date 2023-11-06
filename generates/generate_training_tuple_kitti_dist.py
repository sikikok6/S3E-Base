import numpy as np
# import quaternion
import pandas as pd
import pickle
import os
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from datasets.ScanNetDataset import TrainingTuple

root_dir = '/home/graham/datasets/dataset/'

seqs = ['03', '04', '05', '06','07', '08','09', '10']

test_seqs = ['00', '02']

seq_num = []
test_seq_num = []



for s in seqs:
    num = len(os.listdir(os.path.join(root_dir, 'sequences', s, 'velodyne')))
    seq_num.append(num)
train_cum_sum = np.cumsum(seq_num).tolist()

for s in test_seqs:
    num = len(os.listdir(os.path.join(root_dir, 'sequences', s, 'velodyne')))
    test_seq_num.append(num)
# train_embeddings = np.load('./training/train_apt_embeddings.npy')[0]
# test_embeddings = np.load('./training/test_apt_embeddings.npy')[0]
iou_heaps = []
test_iou_heaps = []
test_cum_sum = np.cumsum(test_seq_num).tolist()



pose_list = os.listdir(os.path.join(root_dir, 'pose'))
pose_list.sort()
train_pose_list = pose_list[sum(test_seq_num):]
test_pose_list = pose_list[:test_cum_sum[0]]

train_poses = []
train_translations = []

test_poses = []
test_translations = []

for i in range(len(train_pose_list)):
    R = np.loadtxt(os.path.join(root_dir, 'pose', train_pose_list[i]))
    train_poses.append(R)
    # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
    train_translations.append(R[:3, -1])

for i in range(len(test_pose_list)):
    R = np.loadtxt(os.path.join(root_dir, 'pose', test_pose_list[i]))
    test_poses.append(R)
    # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
    test_translations.append(R[:3, -1])

# print(len(train_pcl_list))

def gen_tuple(scene):
    
    queries = {}
    count = 0
    count_non = 0
    if scene == 'train':
        # gen_pcl_list = train_pcl_list
        get_pose_list = train_pose_list
        trans = train_translations
        cum_sum = train_cum_sum
    else:
        # gen_pcl_list = test_pcl_list
        get_pose_list = test_pose_list
        trans = test_translations
        cum_sum = test_seq_num[1:]
    labels = list(range(len(trans)))
    print(len(trans))
    for anchor_ndx in range(len(trans)):
        # anchor_pos = poses[anchor_ndx]
        query = os.path.join(root_dir, 'pointcloud_4096_fps', train_pose_list[anchor_ndx].replace('pose.txt', 'bin'))
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        timestamp = int(os.path.splitext(scan_filename)[0])
        # timestamp = int(os.path.splitext(scan_filename)[0])



        positives = []
        non_negatives = []
        most_positive =  [anchor_ndx]
        seq_ind = 0
        for i in range(len(cum_sum)):
            if anchor_ndx < cum_sum[i]:
                seq_ind = i
                break
        
        
        if seq_ind == 0:
            start = 0
            end = cum_sum[seq_ind]
        else:
            start = cum_sum[seq_ind - 1]
            end = cum_sum[seq_ind]
        


        for i in range(0, start):
            non_negatives.append(i)
        for i in range(end, cum_sum[-1]):
            non_negatives.append(i)
        
        max_ = 0
        seq_pose = trans[start:end]
        tree = KDTree(seq_pose)
        # print(seq_ind, start, end, anchor_ndx)
        ind_p = tree.query_radius([seq_pose[anchor_ndx-start]], r=12.5) + start
        ind_non = tree.query_radius([seq_pose[anchor_ndx-start]], r=50) + start
        positives = ind_p[0].tolist()
        positives.remove(anchor_ndx)
        positives.sort()
        non_negatives.extend(ind_non[0].tolist())
        non_negatives = list(set(non_negatives))
        non_negatives.sort()
        # print(anchor_ndx - start)
        # print(ind_p)

        
        # for i in range(start, end):
        #     if i == anchor_ndx:
        #         non_negatives.append(i)
        #         continue
        #     if iou[anchor_ndx - start][i - start] > 0.3:
        #         positives.append(i)
        #         if iou[anchor_ndx - start][i - start] > max_:
        #             max_ = iou[anchor_ndx - start][i - start]
        #             most_positive[0] = i
        #         non_negatives.append(i)
        #     elif iou[anchor_ndx - start][i - start] > 0:
        #         non_negatives.append(i)
        
        negatives = np.setdiff1d(labels, non_negatives, True)
        # if len(positives) != 0:
        #     argsort = np.argsort(-iou[positives])
        #     positives = list(np.array(positives)[argsort])
        #     most_positive.append(positives[0])
# 
        if scene == 'train':
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
                                                positives=positives, non_negatives=non_negatives, pose=None, most_positive=most_positive, negatives=negatives, neighbours=None)
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'kitti_train_tuple_dist.pickle')
        else:
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
                                                positives=positives, non_negatives=non_negatives, pose=None, most_positive=most_positive, negatives=negatives, neighbours=None)
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'kitti_test_tuple_dist.pickle')

    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

gen_tuple('train')
gen_tuple('test')




