#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
from misc.utils import MinkLocParams
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import cv2 as cv

from models.model_factory import model_factory



rgb_config = '/home/student/code/S3E-rgb/config/config_baseline_multimodal.txt'
rgb_model_config = '/home/student/code/S3E-rgb/models/minklocrgb.txt'
rgb_weights =  '/home/student/code/S3E-rgb/weights/model_MinkLocRGB_20230805_16524_epoch_current_recall0.0_fire.pth'

class ValRGBTransform:
    def __init__(self):
        # 1 is default mode, no transform
        t = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        e = self.transform(e)
        return e

def ts_from_filename(filename):
    # Extract timestamp (as integer) from the file path/name
    temp = os.path.split(filename)[1]
    lidar_ts = os.path.splitext(temp)[0]        # LiDAR timestamp
    return lidar_ts

def image4lidar(filename, image_path, image_ext, lidar2image_ndx, k=None):
    # Return an image corresponding to the given lidar point cloud (given as a path to .bin file)
    # k: Number of closest images to randomly select from
    # lidar_ts, traversal = ts_from_filename(filename)
    lidar_ts= ts_from_filename(filename)
    image_file_path = os.path.join(image_path, filename)
    #image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'
    from PIL import Image
    img = Image.open(image_file_path)
    # import cv2
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # query_img = Image.open(query_filename).convert('RGB')
    return img

def load_data_item(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(file_name)

    result = {}
    if params.use_cloud:
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        result['coords'] = pc

    if params.use_rgb:
        # Get the first closest image for each LiDAR scan
        # assert os.path.exists(params.lidar2image_ndx_path), f"Cannot find lidar2image_ndx pickle: {params.lidar2image_ndx_path}"
        # lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))
        lidar2image_ndx = {}
        for i in range(len(os.listdir(params.dataset_folder))):
            lidar2image_ndx[i] = [i]
        img = image4lidar(file_name, params.image_path, '.color.png', lidar2image_ndx, k=1)
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result['image'] = transform(img)

    return result



if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"
print('Device: {}'.format(device))

rgb_params = MinkLocParams(rgb_config, rgb_model_config)


# # load minkloc
rgb_mink_model = model_factory(rgb_params)
rgb_mink_model.load_state_dict(torch.load(rgb_weights, map_location=device))
rgb_mink_model.eval()
img_dir = '/home/graham/datasets/fire/color'
img_list = os.listdir(img_dir)
img_path_m = '/home/graham/datasets/dataset/color/000009.png'
img_path = '/home/graham/datasets/dataset/color/000010.png'
img_path_p = '/home/graham/datasets/dataset/color/000011.png'

# img_path_m = '/home/graham/datasets/dataset/color/frame-000832.color.png'
# img_path = '/home/graham/datasets/dataset/color/frame-000833.color.png'
# img_path_p = '/home/graham/datasets/dataset/color/frame-000834.color.png'
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

model = resnet50(pretrained=True)
print(rgb_mink_model.image_fe)
target_layers = [rgb_mink_model.image_fe.fh_conv1x1['3']]
# target_layers = [rgb_mink_model.image_fe.fh_conv1x1['1'],rgb_mink_model.image_fe.fh_conv1x1['2'], rgb_mink_model.image_fe.fh_conv1x1['3'], rgb_mink_model.image_fe.fh_conv1x1['4'] ]
# target_layers = [model.layer4[-1]]

# img_path = os.path.join(img_dir, i)
img_m = load_data_item(img_path_m, rgb_params)
img = load_data_item(img_path, rgb_params)
img_p = load_data_item(img_path_p, rgb_params)
input_tensor = torch.stack([img_m['image'].to('cuda:1'), img['image'].to('cuda:1'),img_p['image'].to('cuda:1')]).unsqueeze(0).to('cuda:1')
# input_tensor = img['image'].unsqueeze(0).to('cuda')
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!
# 
# Construct the CAM object once, and then re-use it on many images:
cam = EigenCAM(model=rgb_mink_model, target_layers=target_layers, use_cuda=True)
# cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
# 
# 
targets = [ClassifierOutputTarget(127)]
# 
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
# 
# In this example grayscale_cam has only one image in the batch:
print(grayscale_cam.shape)
grayscale_cam = grayscale_cam[0, :]
img_ = cv2.imread(img_path)
img_ = np.float32(img_ / 255)
print(grayscale_cam.shape)
visualization = show_cam_on_image(img_, grayscale_cam)
cv2.imwrite('./viz_4.png',visualization)


    
