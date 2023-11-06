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
print(test_cum_sum)

for index, s in enumerate(seqs):
    print(s)
    iou_file = np.load('/home/graham/datasets/dataset/' + s + '-full.npz', allow_pickle=True)
    iou_file = iou_file[iou_file.files[0]]
    iou_num = seq_num[index]
    iou_heap = np.zeros((iou_num, iou_num))
    imgf1 = np.char.mod('%06d', iou_file[:, 0]).astype(np.uint)
    imgf2 = np.char.mod('%06d', iou_file[:, 1]).astype(np.uint)
    overlap = iou_file[:, 2]
    iou_heap[imgf1, imgf2] = overlap
    # iou_heap =[[0 for _ in range(iou_num)] for _ in range(iou_num)]
    # for i in iou_file:
    #     iou_heap[int(i[0])][int(i[1])] = i[2]
    iou_heaps.append(np.array(iou_heap) + np.array(iou_heap).T)

for index, s in enumerate(test_seqs):
    iou_file = np.load('/home/graham/datasets/dataset/' + s + '-full.npz', allow_pickle=True)
    iou_file = iou_file[iou_file.files[0]]
    iou_num = test_seq_num[index]

    iou_heap = np.zeros((iou_num, iou_num))
    imgf1 = np.char.mod('%06d', iou_file[:, 0]).astype(np.uint)
    imgf2 = np.char.mod('%06d', iou_file[:, 1]).astype(np.uint)
    overlap = iou_file[:, 2]
    iou_heap[imgf1, imgf2] = overlap
    # iou_heap =[[0 for _ in range(iou_num)] for _ in range(iou_num)]
    # for i in iou_file:
    #     iou_heap[int(i[0])][int(i[1])] = i[2]
    test_iou_heaps.append(np.array(iou_heap) + np.array(iou_heap).T)

pcl_list = os.listdir(os.path.join(root_dir, 'pointcloud_4096_fps'))
pcl_list.sort()
train_pcl_list = pcl_list[sum(test_seq_num):]
test_pcl_list = pcl_list[:test_cum_sum[0]]

pose_list = os.listdir(os.path.join(root_dir, 'pose'))
pose_list.sort()

train_pose_list = pose_list[sum(test_seq_num):]
test_pose_list = pose_list[test_cum_sum[0]:sum(test_seq_num)]

train_poses = []
train_translations = []

test_poses = []
test_translations = []



def gen_tuple(scene):
    
    queries = {}

    if scene == 'train':
        gen_pcl_list = train_pcl_list

        cum_sum = train_cum_sum
    else:
        gen_pcl_list = test_pcl_list
        cum_sum = test_seq_num[:1]
    labels = list(range(len(gen_pcl_list)))
    for anchor_ndx in range(len(gen_pcl_list)):
        # anchor_pos = poses[anchor_ndx]
        query = os.path.join(root_dir, 'pointcloud_4096_fps', gen_pcl_list[anchor_ndx])
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
        
        if scene == 'train':
            iou = iou_heaps[seq_ind]
        else:
            iou = test_iou_heaps[0]

        for i in range(0, start):
            non_negatives.append(i)
        for i in range(end, cum_sum[-1]):
            non_negatives.append(i)
        
        max_ = 0
    

        
        for i in range(start, end):
            if i == anchor_ndx:
                non_negatives.append(i)
                continue
            if iou[anchor_ndx - start][i - start] >= 0.25:
                if abs(anchor_ndx - i) < 10:
                    non_negatives.append(i)
                    continue
                positives.append(i)
                if iou[anchor_ndx - start][i - start] > max_:
                    max_ = iou[anchor_ndx - start][i - start]
                    most_positive[0] = i
                non_negatives.append(i)
            elif iou[anchor_ndx - start][i - start] > 0.000001:
                non_negatives.append(i)
        
        positives.sort()
        non_negatives.sort()
        
        negatives = np.setdiff1d(labels, non_negatives, True)
        # print(negatives)
        if scene == 'train':
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
                                                positives=positives, non_negatives=non_negatives, pose=None, most_positive=most_positive, negatives=negatives, neighbours=None)
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'kitti_train_tuple.pickle')
        else:
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
                                                positives=positives, non_negatives=non_negatives, pose=None, most_positive=most_positive, negatives=negatives, neighbours=None)
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'kitti_test_tuple.pickle')

    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

gen_tuple('train')
gen_tuple('test')




