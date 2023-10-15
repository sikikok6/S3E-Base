# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import os
import numpy as np
import pickle
from sklearn.neighbors import KDTree


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


root_dir = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/'
# pai_dir = '/root/7scenes/'

test_iou_heap = np.load('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/iou_/test_fire_iou.npy')
test_length = len(test_iou_heap)

database_iou_heap = test_iou_heap[:1000]
query_iou_heap = test_iou_heap[1000:]


train_pointcloud_dir = root_dir + 'train/pointcloud_4096/'
test_pointcloud_dir = root_dir + 'test/pointcloud_4096/'

test_pose_dir = root_dir + 'test/pose'
test_pose_list = os.listdir(test_pose_dir)
test_pose_list.sort()

train_pointcloud_list = os.listdir(train_pointcloud_dir)
test_pointcloud_list = os.listdir(test_pointcloud_dir)


train_pointcloud_list.sort()
test_pointcloud_list.sort()

train_pickle = pickle.load(
    open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_train_overlap.pickle', 'rb'))
test_pickle = pickle.load(
    open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_test_overlap.pickle', 'rb'))


def construct_query_and_database_sets():
    poses = []
    database_pose = []
    for i in range(2000):
        R = np.loadtxt(os.path.join(
            root_dir, 'test', 'pose', test_pose_list[i]))
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        poses.append(R[:3, -1])
    database_pose = poses[:1000]
    query_pose = poses[1000:]

    database_tree = KDTree(database_pose)

    database_sets = []
    test_sets = []
    database = {}
    for i in range(len(test_pointcloud_list[:1000])):
        database[i] = {'query': os.path.join(test_pointcloud_dir,
                                             test_pointcloud_list[:1000][i])}
    print(len(database))
    database_sets.append(database)
    test = {}
    for i in range(len(test_pointcloud_list[1000:])):
        test[i] = {'query': os.path.join(test_pointcloud_dir,
                                         test_pointcloud_list[1000:][i])}
    print(len(test.keys()))
    test_sets.append(test)
    for i in range(len(database_sets)):
        for j in range(len(test_sets)):
            for key in range(len(test_sets[j].keys())):
                # ind_p = database_tree.query_radius([query_pose[key].tolist()], r=0.25)
                key_list = []
                for ndx in range(len(database_iou_heap)):
                    iou_sum = query_iou_heap[key][ndx] + \
                        database_iou_heap[ndx][key+1000]
                    iou_sub = query_iou_heap[key][ndx] - \
                        database_iou_heap[ndx][key+1000]
                    if abs(iou_sub) < 0.3 and iou_sum > 0.4:
                        key_list.append(ndx)

                test_sets[j][key][i] = key_list
                # test_sets[j][key][i] = test_pickle[key]['positives']
            # for key in range(len(database_sets[i].keys())):

            #     database_sets[j][key][i] = train_pickle[key]['positives']
    print(test_sets)
    output_to_file(database_sets, root_dir + '/pickle',
                   'fire_evaluation_database.pickle')
    output_to_file(test_sets, root_dir + '/pickle',
                   'fire_evaluation_query.pickle')


construct_query_and_database_sets()
