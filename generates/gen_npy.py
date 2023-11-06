import os
from scipy import io
import numpy as np

root_dir = '/home/graham/datasets/'

iou_dir =  '../ioumat'
iou_list = os.listdir(iou_dir)
iou_list.sort()
print(iou_list)

for i in iou_list:
    print(i)
    iou_dst = os.path.join(iou_dir, i)
    iou_arr = i.split('_')
    ds = iou_arr[1]
    scene = iou_arr[2].replace('.mat', '')
    train_dir = os.path.join(root_dir, ds, scene, 'train', 'pointcloud_4096')
    train_count = len(os.listdir(train_dir))
    ioumat =  io.loadmat(os.path.join(iou_dir, i))['iou_non_empty_matrix']
    train_set = ioumat[:train_count, :train_count]
    database_set = ioumat[:train_count, train_count:]
    query_set = ioumat[train_count:, :train_count]
    np.save(os.path.join('../iou', 'train_'+ds+'_'+scene+'.npy'), train_set)
    np.save(os.path.join('../iou', 'database_'+ds+'_'+scene+'.npy'), train_set)
    np.save(os.path.join('../iou', 'query_'+ds+'_'+scene+'.npy'), train_set)

    print(train_count)