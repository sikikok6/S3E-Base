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



# pai_dir = '/root/7scenes/'






import numpy as np

def construct_query_and_database_sets(ds, scene):

    root_dir = os.path.join('/home/graham/datasets/', ds, scene)
    query_iou_heap = np.load(os.path.join('/home/graham/Code/S3E_PreProcess/iou', 'query_'+ds+'_'+scene+'.npy'))
    database_iou_heap = np.load(os.path.join('/home/graham/Code/S3E_PreProcess/iou', 'database_'+ds+'_'+scene+'.npy'))

    train_pointcloud_dir = root_dir + '/train/pointcloud_4096/'
    test_pointcloud_dir = root_dir + '/test/pointcloud_4096/'

    train_pointcloud_list = os.listdir(train_pointcloud_dir)
    test_pointcloud_list = os.listdir(test_pointcloud_dir)

    train_pointcloud_list.sort()
    test_pointcloud_list.sort()

    database_sets = []
    test_sets = []
    database = {}
    for i in range(len(train_pointcloud_list)):
        database[i] = {'query': os.path.join(train_pointcloud_dir,
                                             train_pointcloud_list[i])}
    print(len(database))
    database_sets.append(database)
    test = {}
    for i in range(len(test_pointcloud_list)):
        test[i] = {'query': os.path.join(test_pointcloud_dir,
                                             test_pointcloud_list[i])}
    print(len(test.keys()))
    test_sets.append(test)
    for i in range(len(database_sets)):
        for j in range(len(test_sets)):
            for key in range(len(test_sets[j].keys())):
                # ind_p = database_tree.query_radius([query_pose[key].tolist()], r=0.25)
                key_list = []
                for ndx in range(len(database_iou_heap)):
                    iou_sum = query_iou_heap[key][ndx] + database_iou_heap[ndx][key]
                    iou_sub = query_iou_heap[key][ndx] - database_iou_heap[ndx][key]
                    if abs(iou_sub) < 0.3 and iou_sum > 0.4:
                        key_list.append(ndx)

                test_sets[j][key][i] = key_list
                # test_sets[j][key][i] = test_pickle[key]['positives']
            # for key in range(len(database_sets[i].keys())):
          
            #     database_sets[j][key][i] = train_pickle[key]['positives']
    print(test_sets)
    output_to_file(database_sets, root_dir + '/pickle',
                   ds+'_'+scene+'_evaluation_database.pickle')
    output_to_file(test_sets, root_dir + '/pickle',
                   ds+'_'+scene+'_evaluation_query.pickle')
    
ds_set = ['apt1','apt2','fire1','fire2']
scenes = [['kitchen','living'], ['living','bed', 'kitchen','luke'], ['gates362','gates381','lounge','manolis'],['5a','5b']]

for i in range(len(ds_set)):
    for j in scenes[i]:
        construct_query_and_database_sets(ds_set[i], j)





