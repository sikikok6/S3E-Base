from datasets.augmentation import ValRGBTransform
from datasets.oxford import image4lidar
import torch
import tqdm
from PIL import Image
import os
import copy
import random
import torch.nn as nn
import numpy as np
from models.model_factory import model_factory
from misc.utils import MinkLocParams
from datasets.ScanNetDataset import TrainTransform, ScanNetDataset, TrainSetTransform
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from datasets.ScanNetDataset import ScanNetDataset
import MinkowskiEngine as ME
from dgl.nn.pytorch import SAGEConv
import torch.nn.functional as F
import dgl.function as fn
# from sageconv_plus import SAGEConv_plus
from scipy.spatial import distance
from torch import autograd
import transforms3d.quaternions as txq
from scipy.spatial.transform import Rotation as R

train_sim_mat = []
query_sim_mat = []
database_sim_mat = []


def calc_gradient_penalty(loss_module, embeddings, e, gt_iou, mask):
    logits = autograd.Variable(embeddings, requires_grad=True)
    e = autograd.Variable(e, requires_grad=True)
    # gt_iou = autograd.Variable(gt_iou, requires_grad=True)
    # mask = autograd.Variable(gt_iou, requires_grad=True)
    losses = loss_module(logits, e, gt_iou, mask)
    gradients = autograd.grad(outputs=losses,
                              inputs=(logits, e),
                              grad_outputs=torch.ones(losses.size(),
                                                      device=losses.device),
                              create_graph=True,
                              retain_graph=True,
                              allow_unused=True,
                              only_inputs=True)[0]
    penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return penalty


class MLPModel(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLPModel, self).__init__()

        self.BN = nn.BatchNorm1d(256)
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.BN(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        x = F.normalize(x, dim=1)
        return x


def load_minkLoc_model(config, model_config, pcl_weights=None, rgb_weights=None, project_args=None):
    pw = pcl_weights
    rw = rgb_weights
    # print('Weights: {}'.format(w))
    # print('')

    params = MinkLocParams(config, model_config, project_args)
    # params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # print('Device: {}'.format(device))

    mink_model = model_factory(params)

    if pcl_weights is not None:
        assert os.path.exists(
            pcl_weights), 'Cannot open network weights: {}'.format(pcl_weights)
        # print('Loading weights: {}'.format(weights))
        mink_model.cloud_fe.load_state_dict(
            torch.load(pcl_weights, map_location=device))

    if rgb_weights is not None:
        assert os.path.exists(
            rgb_weights), 'Cannot open network weights: {}'.format(rgb_weights)
        # print('Loading weights: {}'.format(weights))
        mink_model.load_state_dict(torch.load(
            rgb_weights, map_location=device), strict=False)

    return mink_model, params


def load_pc(file_name):
    # returns Nx3 matrix
    pc = np.fromfile(file_name, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    # assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    return pc


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self, f):
        h = self.linear1(f)
        h = F.leaky_relu(h)
        h = self.linear2(h)
        h = F.leaky_relu(h)
        h = self.linear3(h)
        h = F.leaky_relu(h)
        h = self.linear4(h)
        # h = F.relu(h)
        h = torch.sigmoid(h)
        # h = F.relu(h)
        # return torch.sigmoid(h)
        return h


class myGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(myGNN, self).__init__()

        self.MLP = MLP(2*in_feats, 1)
        self.BN = nn.BatchNorm1d(2*in_feats)
        self.conv1 = SAGEConv(2*in_feats, 2*in_feats, 'mean')

        self.Encoder = nn.Sequential(nn.Linear(in_feats+7, 2*in_feats),
                                     nn.ReLU()
                                     )

        self.Decoder = nn.Sequential(nn.Linear(2*in_feats, in_feats+7),
                                     nn.Tanh())

    def apply_edges(self, edges):
        h_u = edges.src['x']
        h_v = edges.dst['x']
        # score = self.MLP(torch.cat((h_u, h_v), 1))
        score = self.MLP(h_u - h_v)
        return {'score': score}

    def forward(self, g, x):

        x = self.Encoder(x)

        x = self.BN(x)

        with g.local_scope():
            g.ndata['x'] = x
            g.apply_edges(self.apply_edges)
            e = g.edata['score']

        A = self.conv1(g, x, e)

        A = self.Decoder(A)

        est_pose = A[0, 512:]

        pos_out = est_pose[:3]
        ori_out = est_pose[3:]

        A = A[:, :512]

        A = F.leaky_relu(A)

        A = F.normalize(A, dim=1)
        # pred2, A2 = self.conv2(g, pred)
        return A, e, pos_out, ori_out


class ListDict(object):
    def __init__(self, items=None):
        if items is not None:
            self.items = copy.deepcopy(items)
            self.item_to_position = {
                item: ndx for ndx, item in enumerate(items)}
        else:
            self.items = []
            self.item_to_position = {}

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class BatchSampler(Sampler):
    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k=2 similar elements (positives)
    # Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    def __init__(self, dataset: ScanNetDataset, batch_size: int, type: str):
        self.batch_size = batch_size
        self.max_batches = batch_size
        self.dataset = dataset
        self.type = type
        self.k = 2

        # Index of elements in each batch (re-generated every epoch)
        self.batch_idx = []
        # List of point cloud indexes
        self.elems_ndx = list(self.dataset.queries)

    def __iter__(self):
        # Re-generate batches every epoch
        # self.generate_batches()
        # self.generate_most_only_batches()
        if self.type == 'train':
            # self.generate_top_batches()
            self.generate_smoothap_batches()
            # self.generate_smoothap_batches()
        else:
            self.generate_smoothap_val_batches()

        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_smoothap_batches(self):
        self.batch_idx = []
        for ndx in range(len(self.dataset)):
            current_batch = []
            current_batch.append(ndx)
            self.batch_idx.append(current_batch)
        random.shuffle(self.batch_idx)

    def generate_smoothap_val_batches(self):
        self.batch_idx = []
        for ndx in range(len(self.dataset)):
            current_batch = []
            current_batch.append(ndx)
            self.batch_idx.append(current_batch)

    def generate_batches(self):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        unused_elements_ndx = ListDict(self.elems_ndx)
        current_batch = []

        assert self.k == 2, 'sampler can sample only k=2 elements from the same class'

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2*self.k:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert len(current_batch) % self.k == 0, 'Incorrect bach size: {}'.format(
                        len(current_batch))
                    self.batch_idx.append(current_batch)
                    current_batch = []
                    if (self.max_batches is not None) and (len(self.batch_idx) >= self.max_batches):
                        break
                if len(unused_elements_ndx) == 0:
                    break

            # Add k=2 similar elements to the batch
            selected_element = unused_elements_ndx.choose_random()
            unused_elements_ndx.remove(selected_element)
            positives = self.dataset.get_positives(selected_element)
            if len(positives) == 0:
                # Broken dataset element without any positives
                continue

            unused_positives = [
                e for e in positives if e in unused_elements_ndx]
            # If there're unused elements similar to selected_element, sample from them
            # otherwise sample from all similar elements
            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_ndx.remove(second_positive)
            else:
                second_positive = random.choice(list(positives))

            current_batch += [selected_element, second_positive]

        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, 'Incorrect bach size: {}'.format(
                len(batch))


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


def make_collate_fn(dataset: ScanNetDataset, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        # Produces (batch_size, n_points, 3) tensor
        batch = torch.stack(clouds, dim=0)
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            batch = dataset.set_transform(batch)

        if mink_quantization_size is None:
            # Not a MinkowskiEngine based model
            batch = {'cloud': batch}
        else:
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                      for e in batch]
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': coords, 'features': feats}

        # Compute positives and negatives mask
        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(
            e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(
            e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        return batch, positives_mask, negatives_mask, labels

    return collate_fn


def make_smoothap_collate_fn(dataset: ScanNetDataset, mink_quantization_size=None, val=None):
    # set_transform: the transform to be applied to all batch elements

    def collate_fn(data_list):
        # Constructs a batch object
        global train_sim_mat
        global database_sim_mat
        global query_sim_mat
        num = 50
        positives_mask = []
        hard_positives_mask = []
        negatives_mask = []
        most_positives_mask = []
        labels = [e[1] for e in data_list]

        # dataset.queries[labels[0]]

        if val != 'val':
            labels.extend(train_sim_mat[labels[0]][1:num+1])
            positives_mask = [in_sorted_array(
                e, dataset.queries[labels[0]].positives) for e in labels]
            hard_positives_mask = [in_sorted_array(
                e, dataset.queries[labels[0]].hard_positives) for e in labels]
            positives_mask[0] = True
            negatives_mask = [not item for item in positives_mask]
            positives_mask = torch.tensor([positives_mask])
            negatives_mask = torch.tensor([negatives_mask])
            hard_positives_mask = torch.tensor([hard_positives_mask])
            most_positives_mask = [in_sorted_array(
                e, dataset.queries[labels[0]].most_positive) for e in labels]
            most_positives_mask = torch.tensor([most_positives_mask])
        else:

            labels.extend(query_sim_mat[labels[0]][:num])
            positives_mask = [in_sorted_array(
                e, dataset.queries[labels[0]].positives) for e in labels]
            # hard_positives_mask = [in_sorted_array(
            #     e, dataset.queries[labels[0]].hard_positives) for e in labels]
            positives_mask[0] = True
            negatives_mask = [not item for item in positives_mask]
            positives_mask = torch.tensor([positives_mask])
            negatives_mask = torch.tensor([negatives_mask])
            hard_positives_mask = torch.tensor([hard_positives_mask])
            most_positives_mask = [in_sorted_array(
                e, dataset.queries[labels[0]].most_positive) for e in labels]
            most_positives_mask = torch.tensor([most_positives_mask])

        neighbours = []
        if val == 'val':
            neighbours.append(query_sim_mat[labels[0]][:num])
            neighbours_temp = [database_sim_mat[item][1:num+1]
                               for item in labels[1:]]
            neighbours.extend(neighbours_temp)

        else:
            # neighbours = [dataset.get_neighbours(item)[:10] for item in labels]
            for i in labels:
                temp = train_sim_mat[i][1:num+1]

                neighbours.append(temp)
        return positives_mask, negatives_mask, hard_positives_mask, labels, neighbours, most_positives_mask, None

    return collate_fn


def make_dataloader(params, project_params):
    datasets = {}
    dataset_folder = os.path.join(
        project_params.dataset_dir, project_params.scene)
    train_file = 'pickle/' + project_params.scene + '_train_overlap.pickle'
    test_file = 'pickle/' + project_params.scene + '_test_overlap.pickle'

    train_transform = TrainTransform(1)
    train_set_transform = TrainSetTransform(1)

    train_embeddings = np.load('./gnn_pre_train_embeddings.npy')
    test_embeddings = np.load('./gnn_pre_test_embeddings.npy')

    database_len = len(
        test_embeddings) // 2 if len(test_embeddings) < 4000 else 3000
    database_embeddings = test_embeddings[:database_len]
    query_embeddings = test_embeddings[database_len:]
    global train_sim_mat
    global database_sim_mat
    global query_sim_mat

    train_sim = distance.cdist(train_embeddings, train_embeddings)
    database_sim = distance.cdist(database_embeddings, database_embeddings)
    query_sim = distance.cdist(query_embeddings, database_embeddings)
    print(query_sim.shape)

    # train_sim = np.matmul(train_embeddings, train_embeddings.T)
    # database_sim = np.matmul(test_embeddings, test_embeddings.T)
    # train_sim_mat = np.argsort(train_sim)
    # database_sim_mat = np.argsort(database_sim)

    train_sim_mat = np.argsort(train_sim).tolist()
    database_sim_mat = np.argsort(database_sim).tolist()
    query_sim_mat = np.argsort(query_sim).tolist()
    # train_sim_mat = []

    # for i in range(len(train_sim_mat_)):
    #     tmp = train_sim_mat_[i]
    #     mask = tmp // 1000 != i // 1000
    #     tmp = tmp[mask]
    #     train_sim_mat.append(tmp)

    t_ = []
    for i in range(len(train_sim_mat)):
        t = np.array(train_sim_mat[i])
        t_.append(list(t[(t != i) & (t // 1000 != i // 1000)]))
        # to_remove = []
        # for j in train_sim_mat[i]:
        #     if (i == j) or ((i // 1000) == (j // 1000)):
        #         to_remove.append(j)
        #
        #     # if abs(i - j) < 5:
        #     #     to_remove.append(j)
        # for j in to_remove:
        #     train_sim_mat[i].remove(j)
    train_sim_mat = t_

    # database_sim_mat = train_sim_mat.copy()

    datasets['train'] = ScanNetDataset(dataset_folder, train_file, train_transform,
                                       set_transform=train_set_transform)
    datasets['val'] = ScanNetDataset(dataset_folder, test_file, None)

    val_transform = None

    dataloaders = {}
    train_sampler = BatchSampler(
        datasets['train'], batch_size=100, type='train')
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_smoothap_collate_fn(datasets['train'],  0.01)
    dataloaders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                      num_workers=params.num_workers, pin_memory=False)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=100, type='val')
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_smoothap_collate_fn(datasets['val'], 0.01, 'val')
        dataloaders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                        num_workers=params.num_workers, pin_memory=True)
    return dataloaders


def load_data_item(file_name, params, project_params, fp):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)

    result = {}
    if params.use_cloud:
        pc = np.fromfile(os.path.join(fp, file_name), dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == params.num_points * \
            3, "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        result['coords'] = pc

    if params.use_rgb:
        # Get the first closest image for each LiDAR scan
        # assert os.path.exists(params.lidar2image_ndx_path), f"Cannot find lidar2image_ndx pickle: {params.lidar2image_ndx_path}"
        # lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))
        # lidar2image_ndx = {}
        # for i in range(len(os.listdir(params.dataset_folder))):
        #     lidar2image_ndx[i] = [i]
        # img = image4lidar(file_name, None,
        #                   None, None, k=1)
        img = Image.open(os.path.join(project_params.dataset_dir, project_params.scene,
                         'color', file_name.replace('bin', 'color.png')))
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result['image'] = transform(img)

    return result


def get_embeddings_3d(model, params, project_params, device, scene):

    model.eval()
    embeddings_l = []
    file_path = '{}/{}/{}/pointcloud_4096'.format(
        project_params.dataset_dir, project_params.scene, scene)
    file_li = os.listdir(file_path)
    file_li.sort()

    for elem_ndx in tqdm.tqdm(range(len(file_li))):

        x = load_data_item(
            file_li[max(elem_ndx, 0)], params, project_params, file_path)

        with torch.no_grad():
            # coords are (n_clouds, num_points, channels) tensor
            batch = {}
            if params.use_cloud:
                coords = ME.utils.sparse_quantize(coordinates=x['coords'],
                                                  quantization_size=params.model_params.mink_quantization_size)
                bcoords = ME.utils.batched_coordinates([coords]).to(device)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones(
                    (bcoords.shape[0], 1), dtype=torch.float32).to(device)
                batch['coords'] = bcoords
                batch['features'] = feats

            if params.use_rgb:
                # batch['images'] = torch.stack([x_m_1['image'].to(device), x['image'].to(
                #     device), x_p_1['image'].to(device)]).unsqueeze(0).to(device)
                batch['images'] = x['image'].to(device).unsqueeze(0).to(device)

            x = model(batch)
            embedding = x['embedding']

            # embedding is (1, 256) tensor
            # if params.normalize_embeddings:
            #     embedding = torch.nn.functional.normalize(
            #         embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


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


def get_poses(scene, project_params):
    file_path = '{}/{}/{}/pointcloud_4096'.format(project_params.dataset_dir, project_params.scene,
                                                  scene)
    file_li = os.listdir(file_path)
    file_li.sort()
    file_pose = [filename.replace('.bin', '.pose.txt') for filename in file_li]
    file_pose.sort()
    file_pose_path = file_path.replace('pointcloud_4096', 'pose')
    embeddings_pose_l = []

    for elem_ndx in tqdm.tqdm(range(len(file_li))):
        # add for pose
        embeddings_pose = np.loadtxt(os.path.join(
            file_pose_path, file_pose[max(elem_ndx, 0)]))
        # trans pose to 3 + 4
        embeddings_pose = process_poses(embeddings_pose)

        # pose
        embeddings_pose_l.append(embeddings_pose)

    embeddings_pose = np.vstack(embeddings_pose_l)

    return embeddings_pose


def cal_trans_rot_error(pred_pose, gt_pose):
    """
    Calculate both translation and rotation errors between two poses.
    :param pred_pose: Predicted pose as [tx, ty, tz, qx, qy, qz, qw]
    :param gt_pose: Ground truth pose as [tx, ty, tz, qx, qy, qz, qw]
    :return: Translation error and rotation error in degrees
    """
    pred_translation = pred_pose[:3]
    gt_translation = gt_pose[:3]

    pred_R = R.from_quat(pred_pose[3:]).as_matrix()
    gt_R = R.from_quat(gt_pose[3:]).as_matrix()

    cal_R = pred_R.T @ gt_R
    r = R.from_matrix(cal_R).as_rotvec()
    rotation_error_deg = np.linalg.norm(r) * 180 / np.pi
    translation_error = np.linalg.norm(pred_translation - gt_translation)

    # print(f"translation_error:{translation_error}")
    # print(f"rotation_error_deg:{rotation_error_deg}")

    return translation_error, rotation_error_deg


# def generate_test_poses(scene):
#     file_path = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/{}/pointcloud_4096'.format(scene)
#     file_li = os.listdir(file_path)
#     file_li.sort()
#     file_pose = [filename.replace('.bin','.pose.txt') for filename in file_li]
#     file_pose.sort()
#     file_pose_path = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pose/'
#     embeddings_pose_l = []

#     for elem_ndx in tqdm.tqdm(range(len(file_li))):
#         #add for pose
#         embeddings_pose = np.loadtxt(file_pose_path+file_pose[max(elem_ndx, 0)])[:3].ravel()
#         #pose
#         embeddings_pose_l.append(embeddings_pose)

#     embeddings_pose = np.vstack(embeddings_pose_l)
#     return embeddings_pose


def get_embeddings(mink_model, params, device, scene):
    file_path = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/{}/pointcloud_4096'.format(
        scene)
    file_li = os.listdir(file_path)
    file_li.sort()
    embeddings_l = []
    mink_model.eval()
    for elem in tqdm.tqdm(file_li):
        x = load_pc(os.path.join(file_path, elem))

        # coords are (n_clouds, num_points, channels) tensor
        with torch.no_grad():
            coords = ME.utils.sparse_quantize(coordinates=x,
                                              quantization_size=params.model_params.mink_quantization_size)
            bcoords = ME.utils.batched_coordinates([coords])
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            b = {'coords': bcoords.to(device), 'features': feats.to(device)}

            x = mink_model(b)
            embedding = x['embedding']
            # embedding is (1, 1024) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(
                    embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings
