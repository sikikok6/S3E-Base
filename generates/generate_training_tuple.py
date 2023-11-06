import numpy as np
# import quaternion
import pandas as pd
import pickle
import os
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from datasets.ScanNetDataset import TrainingTuple
import random

root_dir = '/home/graham/datasets/apt1/kitchen'
# root_dir = '/home/graham/datasets/fire/train'
# depth_intrinsicis_file = './intrinsics_depth.txt'

# def angular_diff(q_1, q_2):
#     delta_q = np.quaternion(q_1.conj() * q_2)
#     delta_q = delta_q.normalized()
#     delta_q_w = delta_q.w
#     thi = 2 * np.arccos(delta_q_w)
#     return thi / np.pi * 180
# train_mat = np.load('./mat.npy')
# test_mat = np.load('./test_mat.npy')
train_embeddings = np.load('./training/train_apt_embeddings.npy')[0]
test_embeddings = np.load('./training/test_apt_embeddings.npy')[0]

iou_heap = np.load('./apt1_iou_heatmap.npy')
# iou_heap = np.load('/home/graham/datasets/fire/iou_heatmap/train_iou_heatmap.npy')
random_list = np.load('./apt1_kitchen_random_list.npy')
test_list = random_list
test_list.sort()
train_list = list(set(range(len(iou_heap))) - set(list(test_list)))

pose_dir = os.path.join(root_dir,'pose')
pose_list = os.listdir(pose_dir)
pose_list.sort()
train_pose_list = np.array(pose_list)[train_list]
train_poses = []
train_translations = []

for i in range(len(train_pose_list)):
    print(train_pose_list[i])
    R = np.loadtxt(os.path.join(root_dir, 'pose', train_pose_list[i]))
    train_poses.append(R)
    # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
    train_translations.append(R[:3, -1])

def gen_tuple(scene):
    depth_dir = os.path.join(root_dir, 'pointcloud')
    pose_dir = os.path.join(root_dir,'pose')
    # depth_intrinsicis_file_path = os.path.join(root_dir, scene, depth_intrinsicis_file)
    # depth_intrinsicis_file_path = depth_intrinsicis_file
    scale_factor = 1000
    img_width = 640
    img_height = 480

    if scene == 'train':
        # mat = train_mat
        mat = np.matmul(train_embeddings, train_embeddings.T)
        # mat = distance.cdist(train_embeddings, train_embeddings, 'euclidean')
    else:
        # mat = test_mat
        mat = np.matmul(test_embeddings, train_embeddings.T)
        # mat = distance.cdist(test_embeddings, train_embeddings, 'euclidean')
    mat = np.argsort(-mat)



    pose_list = os.listdir(pose_dir)
    pose_list.sort()
    if scene == 'train':
        pose_list = np.array(pose_list)[train_list]
        # pose_list = pose_list[:715]
    if scene == 'test':
        # pose_list = pose_list[715:]
        pose_list = np.array(pose_list)[test_list]

    poses = []
    translations = []

    for i in range(len(pose_list)):
        R = np.loadtxt(os.path.join(root_dir, 'pose', pose_list[i]))
        poses.append(R)
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        translations.append(R[:3, -1])

    print(len(translations))
    tree = KDTree(translations)
    train_tree = KDTree(train_translations)

    if scene == 'train':
        ind_p = tree.query_radius(translations, r=0.15)
        ind_non = tree.query_radius(translations, r=1.5)
    else:
        ind_p = train_tree.query_radius(translations, r=0.15)
        ind_non = train_tree.query_radius(translations, r=1.5)

    # ind_p = tree.query_radius(translations, r=0.25)
    # ind_non = tree.query_radius(translations, r=1.6)


    # iou_heap = np.load('/home/graham/datasets/iou_heatmap/' + scene + '_iou_heatmap.npy')
    #  print(iou_heap.shape)
    #  iou_heap = iou_heap[:30, :30]
    #  print(iou_heap.shape)
    queries = {}
    count = 0
    count_non = 0
    labels = range(len(pose_list))
    for anchor_ndx in range(len(pose_list)):
        anchor_pos = poses[anchor_ndx]
        query = os.path.join(root_dir, 'pointcloud_4096', pose_list[anchor_ndx].replace('pose.txt', 'bin'))
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        timestamp = int(os.path.splitext(scan_filename)[0][6:])
        # timestamp = int(os.path.splitext(scan_filename)[0])


        if scene == 'train':
            # iou = iou[:715]
            iou = iou_heap[train_list[anchor_ndx]].copy()
            iou = np.array(iou)[train_list]
        if scene == 'test':
            # iou = iou_heap[anchor_ndx + 715].copy()
            # iou = iou[:715]
            iou = iou_heap[test_list[anchor_ndx]].copy()
            iou = np.array(iou)[train_list]


        positives = []
        non_negatives = []
        most_positive =  []
        # positives = ind_p[anchor_ndx]
        # non_negatives = ind_non[anchor_ndx]

        # positives = positives[positives != anchor_ndx]
        # # Sort ascending order
        # positives = np.sort(positives)
        # non_negatives = np.sort(non_negatives)

        if scene == 'train':
            for ndx in range(len(iou)):
                if ndx == anchor_ndx:
                    non_negatives.append(ndx)
                    continue
                if abs(iou[ndx] - iou_heap[train_list[ndx]][train_list[anchor_ndx]]) < 0.1:
                    if iou[ndx] + iou_heap[train_list[ndx]][train_list[anchor_ndx]] > 1.0:
                        if len(most_positive) == 0:
                            most_positive.append(ndx)
                        if iou[ndx] + iou_heap[train_list[ndx]][train_list[anchor_ndx]] > most_positive[0]:
                             most_positive[0] = ndx
                        positives.append(ndx)
                        non_negatives.append(ndx)
                    elif iou[ndx] + iou_heap[train_list[ndx]][train_list[anchor_ndx]] > 0.4:
                        non_negatives.append(ndx)
                elif abs(iou[ndx] - iou_heap[train_list[ndx]][train_list[anchor_ndx]]) < 0.3 :
                        non_negatives.append(ndx)
                elif iou[ndx] > 0.1:
                        non_negatives.append(ndx)
        else:
            for ndx in range(len(iou)):
                if ndx == anchor_ndx:
                    non_negatives.append(ndx)
                    continue
                if abs(iou[ndx] - iou_heap[train_list[ndx]][test_list[anchor_ndx]]) < 0.1:
                    if iou[ndx] + iou_heap[train_list[ndx]][test_list[anchor_ndx]] > 1.0:
                        if len(most_positive) == 0:
                            most_positive.append(ndx)
                        if iou[ndx] + iou_heap[train_list[ndx]][test_list[anchor_ndx]] > most_positive[0]:
                             most_positive[0] = ndx
                        positives.append(ndx)
                        non_negatives.append(ndx)
                    elif iou[ndx] + iou_heap[train_list[ndx]][test_list[anchor_ndx]] > 0.4:
                        non_negatives.append(ndx)
                elif abs(iou[ndx] - iou_heap[train_list[ndx]][test_list[anchor_ndx]]) < 0.3 :
                        non_negatives.append(ndx)
                elif iou[ndx] > 0.1:
                        non_negatives.append(ndx)
        
        '''
        # non_negatives = list(set(non_negatives))
        # positives = np.array(positives)
        # positives = np.sort(positives)
        # non_negatives = np.array(non_negatives)
        # non_negatives = np.sort(non_negatives)
        
        # count += len(positives)
        # count_non += len(non_negatives)
        # print(len(positives))
        # print(len(non_negatives))




        '''
        # argsort = np.argsort(iou)
        # # print(len(argsort))
        # positives = argsort[-200:]
        # positives = np.argwhere(iou > 0.55).squeeze()
        # # negatives = argsort[:100]
        # non_negatives = argsort[:-200]
        # non_negatives = np.argwhere(iou > 0.35).squeeze()
        # positives = positives[positives != anchor_ndx]
        # positives = np.sort(positives)
        # non_negatives = np.sort(non_negatives)
        # # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])

        negatives = np.setdiff1d(labels, non_negatives, True)
        # if len(positives) != 0:
        #     argsort = np.argsort(-iou[positives])
        #     positives = list(np.array(positives)[argsort])
        #     most_positive.append(positives[0])

        if scene == 'train':
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][1:51])
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'apt1_kitchen_train_random_iou.pickle')
        else:
            queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                                positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=None, negatives=negatives, neighbours=None)
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=mat[anchor_ndx][:50])
                                                # positives=positives, non_negatives=non_negatives, pose=anchor_pos, most_positive=most_positive, negatives=negatives, neighbours=None)
            file_path = os.path.join(root_dir, 'pickle', 'apt1_kitchen_test_random_iou.pickle')
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

gen_tuple('train')
gen_tuple('test')




# queries = {} 
# for anchor_ndx in range(len(ind_p)):
#     anchor_pos = np.array(poses[anchor_ndx])
#     query = os.path.join(root_dir, 'pointcloud', pose_list[anchor_ndx].replace('txt', 'bin'))
#     # Extract timestamp from the filename
#     scan_filename = os.path.split(query)[1]
#     assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
#     # timestamp = int(os.path.splitext(scan_filename)[0])
#     timestamp = anchor_ndx

#     print(anchor_ndx)
#     positives = ind_p[anchor_ndx]
#     print(positives.shape)
#     non_negatives = ind_non[anchor_ndx]
#     print(non_negatives.shape)

#     positives = positives[positives != anchor_ndx]
#     print(positives.shape)
#     # Sort ascending order
#     positives = np.sort(positives)
#     non_negatives = np.sort(non_negatives)

    # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
    # queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
    #                                     positives=positives, non_negatives=non_negatives, pose=anchor_pos)
# file_path = os.path.join(root_dir, 'pickle', 'test.pickle')
# with open(file_path, 'wb') as handle:
    # pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)


