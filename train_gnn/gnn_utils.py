import torch
import tqdm
import os
import torch.nn as nn
import numpy as np
from models.model_factory import model_factory
from misc.utils import MinkLocParams
import MinkowskiEngine as ME
import torch.nn.functional as F
from PIL import Image

from torch import autograd
from scipy.spatial.transform import Rotation as R
from models.resnetrgb import resnet18
from datasets.augmentation import ValRGBTransform
import pytorch3d.transforms as p3dtrans


def calc_gradient_penalty(loss_module, embeddings, e, gt_iou, mask):
    logits = autograd.Variable(embeddings, requires_grad=True)
    e = autograd.Variable(e, requires_grad=True)
    # gt_iou = autograd.Variable(gt_iou, requires_grad=True)
    # mask = autograd.Variable(gt_iou, requires_grad=True)
    losses = loss_module(logits, e, gt_iou, mask)
    gradients = autograd.grad(
        outputs=losses,
        inputs=(logits, e),
        grad_outputs=torch.ones(losses.size(), device=losses.device),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
        only_inputs=True,
    )[0]
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def load_resnet_model():
    image_fe = resnet18()
    return image_fe


def load_minkLoc_model(
    config, model_config, pcl_weights=None, rgb_weights=None, project_args=None
):
    params = MinkLocParams(config, model_config, project_args)
    # params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # print('Device: {}'.format(device))

    mink_model = model_factory(params)

    if pcl_weights is not None:
        assert os.path.exists(pcl_weights), "Cannot open network weights: {}".format(
            pcl_weights
        )
        # print('Loading weights: {}'.format(weights))
        mink_model.cloud_fe.load_state_dict(
            torch.load(pcl_weights, map_location=device)
        )

    if rgb_weights is not None:
        assert os.path.exists(rgb_weights), "Cannot open network weights: {}".format(
            rgb_weights
        )
        # print('Loading weights: {}'.format(weights))
        mink_model.load_state_dict(
            torch.load(rgb_weights, map_location=device), strict=False
        )

    return mink_model, params


def load_pc(file_name):
    # returns Nx3 matrix
    pc = np.fromfile(file_name, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    # assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    return pc


def load_data_item(file_name, params, project_params, fp):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)

    result = {}
    if params.use_cloud:
        pc = np.fromfile(os.path.join(fp, file_name), dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert (
            pc.shape[0] == params.num_points * 3
        ), "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        result["coords"] = pc

    if params.use_rgb:
        # Get the first closest image for each LiDAR scan
        # assert os.path.exists(params.lidar2image_ndx_path), f"Cannot find lidar2image_ndx pickle: {params.lidar2image_ndx_path}"
        # lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))
        # lidar2image_ndx = {}
        # for i in range(len(os.listdir(params.dataset_folder))):
        #     lidar2image_ndx[i] = [i]
        # img = image4lidar(file_name, None,
        #                   None, None, k=1)
        img = Image.open(
            os.path.join(
                project_params.dataset_dir,
                project_params.scene,
                "color",
                file_name.replace("bin", "color.png"),
            )
        )
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result["image"] = transform(img)

    return result


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    if array == None or len(array) == 0:
        return False
    array = np.sort(array)
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return bool(array[pos] == e)
        # return True


def get_embeddings_3d(model, params, project_params, device, scene):
    model.eval()
    embeddings_l = []
    file_path = "{}/{}/{}/pointcloud_4096".format(
        project_params.dataset_dir, project_params.scene, scene
    )
    file_li = os.listdir(file_path)
    file_li.sort()

    for elem_ndx in tqdm.tqdm(range(len(file_li))):
        x = load_data_item(file_li[max(elem_ndx, 0)], params, project_params, file_path)

        with torch.no_grad():
            # coords are (n_clouds, num_points, channels) tensor
            batch = {}
            if params.use_cloud:
                coords = ME.utils.sparse_quantize(
                    coordinates=x["coords"],
                    quantization_size=params.model_params.mink_quantization_size,
                )
                bcoords = ME.utils.batched_coordinates([coords]).to(device)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(
                    device
                )
                batch["coords"] = bcoords
                batch["features"] = feats

            if params.use_rgb:
                # batch['images'] = torch.stack([x_m_1['image'].to(device), x['image'].to(
                #     device), x_p_1['image'].to(device)]).unsqueeze(0).to(device)
                batch["images"] = x["image"].to(device).unsqueeze(0).to(device)

            x = model(batch)
            embedding = x["embedding"]

            # embedding is (1, 256) tensor
            # if params.normalize_embeddings:
            embedding = torch.nn.functional.normalize(
                embedding, p=2, dim=1
            )  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_embeddings_resnet(model, params, project_params, device, scene):
    model.eval()
    embeddings_l = []
    feature_maps = []
    file_path = "{}/{}/{}/pointcloud_4096".format(
        project_params.dataset_dir, project_params.scene, scene
    )
    file_li = os.listdir(file_path)
    file_li.sort()

    for elem_ndx in tqdm.tqdm(range(len(file_li))):
        x = load_data_item(file_li[max(elem_ndx, 0)], params, project_params, file_path)

        with torch.no_grad():
            # coords are (n_clouds, num_points, channels) tensor
            batch = {}

            batch["images"] = x["image"].to(device).unsqueeze(0).to(device)

            x = model(batch)
            embedding = x[-1]
            feature_map = x[:-1]

            # embedding is (1, 256) tensor
            # if params.normalize_embeddings:
            embedding = torch.nn.functional.normalize(
                embedding, p=2, dim=1
            )  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)
        feature_maps.append(feature_map[-2].detach().cpu().numpy())

    embeddings = np.vstack(embeddings_l)
    feature_maps = np.vstack(feature_maps)
    print(feature_maps.shape)
    return embeddings, feature_maps


def process_poses(poses_in):
    """
    poses_in: 4x4
    poses_out: 0x7
    """
    poses_out = np.zeros((7))
    poses_out[0:3] = poses_in[:3, -1]
    q = R.from_matrix(poses_in[:3, :3]).as_quat()
    poses_out[3:] = q
    return poses_out


def process_poses_eular(poses_in):
    """
    poses_in: 4x4
    poses_out: 0x6
    """
    poses_out = np.zeros((6))
    poses_out[0:3] = poses_in[:3, -1]
    q = R.from_matrix(poses_in[:3, :3]).as_euler("zxy")
    poses_out[3:] = q
    return poses_out


def inverse_poses(poses_in):
    """
    poses_in: 0x7
    poses_out: 4x4
    """
    poses_out = np.eye(4)

    poses_out[:3, -1] = poses_in[:3]

    R_matrix = R.from_quat(poses_in[3:]).as_matrix()

    poses_out[:3, :3] = R_matrix

    return poses_out


def pose_delta(query_pose, target_pose):
    index_query = torch.tensor(
        [
            [[3, 0, 1, 2] for _ in range(query_pose.shape[1])]
            for _ in range(query_pose.shape[0])
        ]
    ).cuda()

    re_index = torch.tensor(
        [
            [[1, 2, 3, 0] for _ in range(query_pose.shape[1])]
            for _ in range(query_pose.shape[0])
        ]
    ).cuda()
    query_pos = query_pose[:, :, :3]
    query_rot = query_pose[:, :, 3:]
    delta_pos = query_pos - target_pose[:, :, :3]
    delta_rot = query_rot - target_pose[:, :, 3:]

    # delta_rot = p3dtrans.quaternion_multiply(
    #     F.normalize(query_rot, p=2, dim=2).gather(2, index_query),
    #     F.normalize(
    #         p3dtrans.quaternion_invert(target_pose[:, :, 3:]), p=2, dim=2
    #     ).gather(2, index_query),
    # )
    # return torch.cat((delta_pos, delta_rot.gather(2, re_index)), dim=2)

    return torch.cat((delta_pos, delta_rot), dim=2)


def get_poses(scene, project_params):
    file_path = "{}/{}/{}/pointcloud_4096".format(
        project_params.dataset_dir, project_params.scene, scene
    )
    file_li = os.listdir(file_path)
    file_li.sort()
    file_pose = [filename.replace(".bin", ".pose.txt") for filename in file_li]
    file_pose.sort()
    file_pose_path = file_path.replace("pointcloud_4096", "pose")
    embeddings_pose_l = []

    for elem_ndx in tqdm.tqdm(range(len(file_li))):
        # add for pose
        embeddings_pose = np.loadtxt(
            os.path.join(file_pose_path, file_pose[max(elem_ndx, 0)])
        )
        # trans pose to 3 + 4
        embeddings_pose = process_poses_eular(embeddings_pose)

        # pose
        embeddings_pose_l.append(embeddings_pose)

    embeddings_pose = np.vstack(embeddings_pose_l)

    return embeddings_pose


def get_embeddings(mink_model, params, device, scene):
    file_path = "/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/{}/pointcloud_4096".format(
        scene
    )
    file_li = os.listdir(file_path)
    file_li.sort()
    embeddings_l = []
    mink_model.eval()
    for elem in tqdm.tqdm(file_li):
        x = load_pc(os.path.join(file_path, elem))

        # coords are (n_clouds, num_points, channels) tensor
        with torch.no_grad():
            coords = ME.utils.sparse_quantize(
                coordinates=x,
                quantization_size=params.model_params.mink_quantization_size,
            )
            bcoords = ME.utils.batched_coordinates([coords])
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            b = {"coords": bcoords.to(device), "features": feats.to(device)}

            x = mink_model(b)
            embedding = x["embedding"]
            # embedding is (1, 1024) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(
                    embedding, p=2, dim=1
                )  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def cal_trans_rot_error(pred_pose, gt_pose):
    """
    Calculate both translation and rotation errors between two poses.
    :param pred_pose: Predicted pose as [tx, ty, tz, qx, qy, qz, qw]
    :param gt_pose: Ground truth pose as [tx, ty, tz, qx, qy, qz, qw]
    :return: Translation error and rotation error in degrees
    """
    pred_translation = pred_pose[:, :3]
    gt_translation = gt_pose[:, :3]

    pred_R_arr = [
        R.from_euler("zxy", pred_pose[i, 3:]).as_matrix()
        for i in range(len(pred_translation))
    ]
    gt_R_arr = [
        R.from_euler("zxy", gt_pose[i, 3:]).as_matrix()
        for i in range(len(pred_translation))
    ]

    cal_R_arr = [pred_R_arr[i].T @ gt_R_arr[i] for i in range(len(pred_R_arr))]

    r_arr = [R.from_matrix(cal_R_arr[i]).as_rotvec() for i in range(len(cal_R_arr))]
    rotation_error_degs = [
        np.linalg.norm(r_arr[i]) * 180 / np.pi for i in range(len(r_arr))
    ]

    translation_errors = [
        np.linalg.norm(pred_translation[i] - gt_translation[i])
        for i in range(len(pred_translation))
    ]

    return np.mean(translation_errors), np.mean(rotation_error_degs)
