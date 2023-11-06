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

root_dir = '/home/graham/datasets/dataset/'
pcl_dir = os.path.join(root_dir, 'pointcloud_4096_fps')

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
test_cum_sum = np.cumsum(test_seq_num).tolist()

pcl_list = os.listdir(os.path.join(root_dir, 'pointcloud_4096_fps'))
pcl_list.sort()
train_pcl_list = pcl_list[sum(test_seq_num):]
test_pcl_list = pcl_list[:test_cum_sum[0]]
print(len(train_pcl_list))


test_pickle = pickle.load(open('/home/graham/datasets/dataset/pickle/kitti_test_tuple.pickle', 'rb'))
train_pickle = pickle.load(open('/home/graham/datasets/dataset/pickle/kitti_train_tuple.pickle', 'rb'))
gt_ = np.load('/home/graham/datasets/dataset/gt_00.npz', allow_pickle=True)
gt_ = gt_['arr_0']
def construct_query_and_database_sets():
    database_sets = []
    test_sets = []
    database = {}
    for i in range(len(train_pcl_list)):
        database[i] = {'query': os.path.join(pcl_dir,
                                             train_pcl_list[i])}
    database_sets.append(database)
    test = {}
    for i in range(len(test_pcl_list)):
        test[i] = {'query': os.path.join(pcl_dir,
                                             test_pcl_list[i])}
    print(len(test.keys()))
    test_sets.append(test)
    for i in range(len(database_sets)):
        for j in range(len(test_sets)):
            for key in range(len(test_sets[j].keys())):
                # test_sets[j][key][i] = test_pickle[key].positives
                test_sets[j][key][i] = gt_[key]
            for key in range(len(database_sets[i].keys())):
                database_sets[j][key][i] = train_pickle[key].positives
    print(test_sets)
    output_to_file(database_sets, root_dir + '/pickle',
                   'kitti_evaluation_database.pickle')
    output_to_file(test_sets, root_dir + '/pickle',
                   'kitti_evaluation_query.pickle')
    

construct_query_and_database_sets()





