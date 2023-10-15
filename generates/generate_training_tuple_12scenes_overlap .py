import numpy as np
# import quaternion
import pandas as pd
import pickle
import os
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from datasets.ScanNetDataset import TrainingTuple
import random

root_dir = '/home/graham/datasets/'
ds_set = ['apt1','apt2','fire1','fire2']
scenes = [['kitchen','living'], ['living','bed', 'kitchen','luke'], ['gates362','gates381','lounge','manolis'],['5a','5b']]



def gen_tuple(ds, scene, s):
    print(ds, scene, s)
    if s == 'train':
        databse_iou = np.load(os.path.join('../iou','train_'+ds+'_'+scene+'.npy'))
        query_iou = np.load(os.path.join('../iou','train_'+ds+'_'+scene+'.npy'))
    else:
        databse_iou = np.load(os.path.join('../iou','database_'+ds+'_'+scene+'.npy'))
        query_iou = np.load(os.path.join('../iou','query_'+ds+'_'+scene+'.npy'))
        
    pcl_dir = os.path.join(root_dir, ds,scene,s, 'pointcloud_4096')
    pose_dir = os.path.join(root_dir, ds,scene,s,'pose')
    pose_list = os.listdir(pose_dir)
    pose_list.sort()
    poses = []
    for p in pose_list:
        R = np.loadtxt(os.path.join(root_dir,ds,scene,s, 'pose', p))
        poses.append(R)
        # train_translations.append(R[:3, -1])

    pcl_list = os.listdir(pcl_dir)
    pcl_list.sort()
        
    queries = {}
    for anchor_ndx in range(len(pcl_list)):
        anchor_pos = poses[anchor_ndx]
        query = os.path.join('/home/graham/datasets/',ds, scene,s ,'pointcloud_4096', pose_list[anchor_ndx].replace('pose.txt', 'bin'))
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        # timestamp = int(os.path.splitext(scan_filename)[0][6:])
        timestamp = int(os.path.splitext(scan_filename)[0][6:])


        labels = list(range(len(query_iou)))

        positives = []
        non_negatives = []
        most_positive =  []
        most_iou = 0
        min_iou = 1
        min_index = -1
        single_iou = 0
        most_single_iou = []


        for ndx in range(len(databse_iou)):

            if ndx == anchor_ndx or abs(ndx -anchor_ndx) < 30:
                non_negatives.append(ndx)
                continue

            iou_sub = query_iou[anchor_ndx][ndx] - databse_iou[ndx][anchor_ndx]
            iou_sum = query_iou[anchor_ndx][ndx] + databse_iou[ndx][anchor_ndx]
            if iou_sub < min_iou:
                min_iou = iou_sub
                min_index = ndx
            if abs(iou_sub) < 0.2:
                if iou_sum > 0.55:
                    if len(most_positive) == 0:
                        most_positive.append(ndx)
                        most_iou = iou_sum
                    if iou_sum > most_iou:
                         most_positive[0] = ndx
                         most_iou = iou_sum
                    positives.append(ndx)
                    non_negatives.append(ndx)
                elif iou_sum > 0.05:
                    non_negatives.append(ndx)
            elif abs(iou_sub) < 0.3 :
                    non_negatives.append(ndx)
            if (query_iou[anchor_ndx][ndx] > 0. or databse_iou[ndx][anchor_ndx] > 0.) and (len(non_negatives) == 0 or non_negatives[-1] != ndx) :
                    non_negatives.append(ndx)

        if len(most_positive) == 0:
            most_positive = most_single_iou
        negatives = np.setdiff1d(labels, non_negatives, True).tolist()

        # if len(positives) != 0:
        #     argsort = np.argsort(-iou[positives])
        #     positives = list(np.array(positives)[argsort])
        #     most_positive.append(positives[0])
        # tmp = {'query':query, 'positives':positives, 'negatives': negatives}

        if s== 'train':
            # queries[anchor_ndx] = tmp
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            #                                     # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][1:51])
            #                                     positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir,ds,scene, 'pickle', ds+'_'+scene+'_train_overlap.pickle')
        else:
            # queries[anchor_ndx] = tmp
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            #                                     # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][:50])
            #                                     positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir,ds,scene, 'pickle', ds+'_'+scene+'_test_overlap.pickle')
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

for i in range(len(ds_set)):
    for j in scenes[i]:
        gen_tuple(ds_set[i], j, 'train')
        gen_tuple(ds_set[i], j, 'test')



