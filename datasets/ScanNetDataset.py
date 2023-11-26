# Author: Jacek Komorowski
# Warsaw University of Technology

# Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project
# For information on dataset see: https://github.com/mikacuy/pointnetvlad

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from typing import Dict
import torchvision.transforms as transforms
from scipy.linalg import expm, norm

from datasets.augmentation import ValRGBTransform

DEBUG = False


class ScanNetDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """

    def __init__(
        self,
        dataset_path: str,
        query_filename: str,
        image_path: str = None,
        lidar2image_ndx_path: str = None,
        transform=None,
        set_transform=None,
        image_transform=None,
        use_cloud: bool = True,
    ):
        assert os.path.exists(dataset_path), "Cannot access dataset path: {}".format(
            dataset_path
        )
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(
            self.query_filepath
        ), "Cannot access query file: {}".format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(
            open(self.query_filepath, "rb")
        )
        self.image_path = os.path.join(self.dataset_path, "color")
        self.lidar2image_ndx_path = lidar2image_ndx_path
        self.image_transform = image_transform
        self.n_points = (
            4096  # pointclouds in the dataset are downsampled to 4096 points
        )
        self.image_ext = ".color.png"
        self.use_cloud = use_cloud
        print("{} queries in the dataset".format(len(self)))

        # assert os.path.exists(self.lidar2image_ndx_path), f"Cannot access lidar2image_ndx: {self.lidar2image_ndx_path}"
        # self.lidar2image_ndx = pickle.load(open(self.lidar2image_ndx_path, 'rb'))
        self.lidar2image_ndx = {}
        for i in range(len(self)):
            self.lidar2image_ndx[i] = [i]

    def __len__(self):
        return 2000
        if "test" not in self.query_filepath:
            return len(self.queries)
        else:
            if len(self.queries) >= 4000:
                return 2000
            else:
                return 1000
        # return len(self.queries) if 'test' not in self.query_filepath else 2000

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.queries[ndx].rel_scan_filepath
        # print(f"scannetdataset_filename: {filename}")
        result = {"ndx": ndx}
        # if self.use_cloud:
        #     # Load point cloud and apply transform
        #     file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        #     query_pc = self.load_pc(file_pathname)
        #     if self.transform is not None:
        #         query_pc = self.transform(query_pc)
        #     result['cloud'] = query_pc

        if self.image_path is not None:
            # img = image4lidar(filename, self.image_path,
            #                   self.image_ext, self.lidar2image_ndx, k=None)
            img = Image.open(
                os.path.join(
                    self.image_path, filename.split("/")[-1].replace("bin", "color.png")
                )
            )
            # if self.image_transform is not None:
            transform = ValRGBTransform()
            img = transform(img)
            # query_img = Image.open(query_filename).convert('RGB')
            result["image"] = img

        # This result is a dict-type, result = {'ndx': ndx, 'cloud': query_pc, 'image': img}
        # return result
        return result, ndx

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.dataset_path, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert (
            pc.shape[0] == self.n_points * 3
        ), "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        return pc


def ts_from_filename(filename):
    # Extract timestamp (as integer) from the file path/name
    temp = os.path.split(filename)[1]
    lidar_ts = os.path.splitext(temp)[0]  # LiDAR timestamp
    # assert lidar_ts.isdigit(), 'Incorrect lidar timestamp: {}'.format(lidar_ts)

    # temp = os.path.split(filename)[0]
    # temp = os.path.split(temp)[0]
    # traversal = os.path.split(temp)[1]
    # assert len(traversal) == 19, 'Incorrect traversal name: {}'.format(traversal)

    # return int(lidar_ts), traversal
    return lidar_ts


def image4lidar(filename, image_path, image_ext, lidar2image_ndx, k=None):
    # Return an image corresponding to the given lidar point cloud (given as a path to .bin file)
    # k: Number of closest images to randomly select from
    # lidar_ts, traversal = ts_from_filename(filename)
    lidar_ts = ts_from_filename(filename)
    # # assert lidar_ts in lidar2image_ndx, 'Unknown lidar timestamp: {}'.format(lidar_ts)

    # # Randomly select one of images linked with the point cloud
    # # if k is None or k > len(lidar2image_ndx[lidar_ts]):
    # #     k = len(lidar2image_ndx[lidar_ts])

    # image_ts = random.choice(lidar2image_ndx[lidar_ts][:k])
    # image_ts = lidar2image_ndx[lidar_ts]
    # print(filename)
    # print(image_path, lidar_ts + image_ext)
    image_path = "/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/color"
    image_file_path = os.path.join(image_path, lidar_ts + image_ext)
    # image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'
    img = Image.open(image_file_path).convert("RGB")
    # query_img = Image.open(query_filename).convert('RGB')
    return img


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(
        self,
        id: int,
        timestamp: int,
        rel_scan_filepath: str,
        positives: np.ndarray,
        non_negatives: np.ndarray,
        pose: np.ndarray,
        most_positive: np.ndarray,
        negatives: np.ndarray,
        hard_positives: np.ndarray,
        neighbours: np.ndarray,
    ):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        # pose: [R, t]  4x4 matrix

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.hard_positives = hard_positives
        self.position = pose
        self.most_positive = most_positive
        self.negatives = negatives
        self.neighbours = neighbours


# class TrainingTuple:
#     # Tuple describing an element for training/validation
#     def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
#                  non_negatives: np.ndarray, position: np.ndarray):
#         # id: element id (ids start from 0 and are consecutive numbers)
#         # ts: timestamp
#         # rel_scan_filepath: relative path to the scan
#         # positives: sorted ndarray of positive elements id
#         # negatives: sorted ndarray of elements id
#         # position: x, y position in meters (northing, easting)
#         assert position.shape == (2,)

#         self.id = id
#         self.timestamp = timestamp
#         self.rel_scan_filepath = rel_scan_filepath
#         self.positives = positives
#         self.non_negatives = non_negatives
#         self.position = position


class TrainTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            t = [
                JitterPoints(sigma=0.001, clip=0.002),
                RemoveRandomPoints(r=(0.0, 0.1)),
                RandomTranslation(max_delta=0.01),
                RemoveRandomBlock(p=0.4),
            ]
        else:
            raise NotImplementedError("Unknown aug_mode: {}".format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class TrainSetTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        self.transform = None
        t = [
            RandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])),
            RandomFlip([0.25, 0.25, 0.0]),
        ]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, "sum(p) must be in (0, 1] range, is: {}".format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta  # Rotation around axis
        self.max_theta2 = max_theta2  # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(
            axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5)
        )
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(
                np.random.rand(3) - 0.5,
                (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5),
            )
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.0):
        assert 0 < p <= 1.0
        assert sigma > 0.0

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.0:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(
                probs=torch.tensor([1 - self.p, self.p])
            )
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64)

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(
            range(n), size=int(n * r), replace=False
        )  # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        import math

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            # Fronto-parallel cuboid to remove
            x, y, w, h = self.get_params(coords)
            mask = (
                (x < coords[..., 0])
                & (coords[..., 0] < x + w)
                & (y < coords[..., 1])
                & (coords[..., 1] < y + h)
            )
            coords[mask] = torch.zeros_like(coords[mask])
        return coords
