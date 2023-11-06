# Copyright 2021 Ran Cheng <ran.cheng2@mail.mcgill.ca>
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import itertools

import scipy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME
import os
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import dgl
from dgl.data import DGLDataset
from itertools import product
from datasets.oxford import image4lidar
from datasets.augmentation import ValRGBTransform

def load_data_item(prefix, file_name):
    # returns Nx3 matrix
    file_path = os.path.join(prefix, 'pointcloud_4096', file_name)

    result = {}
    pc = np.fromfile(file_path, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    assert pc.shape[0] == 4096 * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    result['coords'] = pc

        # Get the first closest image for each LiDAR scan
        # assert os.path.exists(params.lidar2image_ndx_path), f"Cannot find lidar2image_ndx pickle: {params.lidar2image_ndx_path}"
        # lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, 'rb'))
    lidar2image_ndx = {}
    for i in range(len(os.listdir(prefix))):
        lidar2image_ndx[i] = [i]
    img = image4lidar(file_name, '/home/graham/darasets/fire/color', '.color.png', lidar2image_ndx, k=1)
    transform = ValRGBTransform()
    # Convert to tensor and normalize
    result['image'] = transform(img)

    return result

# dataset for the embedding training
class GNNDataset(Dataset):
    def __init__(
            self,
            data_root, scene, embed_model,iou_edge_th=0.01):
        """
        dataset initialization function, generate the graph data
        :param data_root: dataset root, which contains the torch_data folder
        :param model_cfgs: configures for the VIT model
        :param patch_size: torch patch size, there are three options: ["16", "32", "64"]
        :param iou_edge_th: the threshold for iou filter to build the edge between nodes of pose graph
        """
        self._data_root = data_root
        self._scene = scene
        self._embed_model = embed_model
        self._subgraph_size = 5
        self._pose_data_root = os.path.join(data_root, "pose")
        self._pointcloud_data_root = os.path.join(data_root, "pointcloud_4096")
        # self._dataset_size = len(self._torch_data_files)
        # iou heatmap
        iou_heatmap_fname = os.path.join('/home/graham/datasets/fire', 'iou_heatmap',scene + "_iou_heatmap.npy")
        if os.path.isfile(iou_heatmap_fname):
            # load the query map of iou data
            self._iou_map = np.load(iou_heatmap_fname)
            # self._iou_map = self._iou_map[:1000][:1000]
        else:
            raise Exception(
                "IoU heatmap is empty! Please train the embedding first.")
        # collect edge data for the pose graph
        self._g_edge_list = []
        self._iou_edge_th = iou_edge_th
        # cook the node embeddings offline
        self.emb_data_buffer = None
        # self._node_embedding_dir = os.path.join(self._data_root, "embeddings")
        # if not os.path.exists(self._node_embedding_dir):
        #     os.mkdir(self._node_embedding_dir)
        # if not len(os.listdir(self._node_embedding_dir)) > 1:
        #     # embed the frames with vit
        #     print("no embedding detected!")
        if os.path.exists('/home/graham/datasets/fire/' + self._scene + '/embeddings.mat'):
            self.emb_data_buffer = scipy.io.loadmat('/home/graham/datasets/fire/' + self._scene + '/embeddings.mat')['mat']
        else:
            self.embed_graph_nodes()
            torch.cuda.empty_cache()
        # else:
            # print("embedding data found, loading the embedding data!")
            # self._embedding_data_files = glob.glob(
            #     os.path.join(self._node_embedding_dir, "*.npy"))
            # self._embedding_data_files = sorted(self._embedding_data_files)
        # self.load_embedding_data()
        print("building graph from IoU map...")
        self.build_graph()
        print("builded graph from IoU map...")
        # construct the masked graph dataset
        self.masked_graph_edge_list = []
        self.emb_data_buffer_graph = None
        self.emb_data_buffer_query = None
        self.graph_idx = None
        self.node_idx = None
        self.graph_iou_map = None
        self.query_iou_map = None
        self.graph_idx_map = {}
        print("masking graph...")
        self.mask_graph()
        # self.train_query_idx, self.val_query_idx = train_test_split(np.arange(len(self.node_idx)), shuffle=True)

    def mask_graph(self):
        # mask the graph so that the map graph does not contain the query node
        idx_list = np.arange(len(self.emb_data_buffer))
        print(idx_list)
        # the portion of graph idx shold be large enough, to decrease the occurrence of orphan sub-graphs
        self.graph_idx, self.node_idx = train_test_split(
            idx_list, train_size=0.9, shuffle=True)
        # filter out the embedding and edges for graph
        self.graph_idx = sorted(self.graph_idx)
        #  print(len(self.graph_idx))
        # keep record the index mapping
        for gid, gd in enumerate(self.graph_idx):
            self.graph_idx_map[gd] = gid
        gidx_permute = np.array(
            [list(r) for r in
             product(range(len(self.graph_idx)),range(len(self.graph_idx)))])
        print(gidx_permute)
        self.graph_iou_map = np.reshape(self._iou_map[gidx_permute[:, 0], gidx_permute[:, 1]],
                                       (len(self.graph_idx), len(self.graph_idx)))
        qidx_permute = np.array(
            [list(r) for r in product(self.node_idx, self.graph_idx)])
        self.query_iou_map = np.reshape(self._iou_map[qidx_permute[:, 0], qidx_permute[:, 1]],
                                        #  (len(self.node_idx), len(self.graph_idx)))
                                        (len(self.node_idx), len(self.graph_idx)))
        print("building masked graph edge ...")
        for edge_item in tqdm(self._g_edge_list):
            if edge_item[0] not in self.node_idx and edge_item[1] not in self.node_idx:
                self.masked_graph_edge_list.append(
                    [self.graph_idx_map[edge_item[0]], self.graph_idx_map[edge_item[1]]])
        self.emb_data_buffer_graph = (self.emb_data_buffer[self.graph_idx]).astype(np.float16)
        self.emb_data_buffer_query = (self.emb_data_buffer[self.node_idx]).astype(np.float16)

    def get_graph(self):
        """
        retrieve the graph data
        :return: masked_edge: array of edge, emb_data_buffer_graph: graph node embedding, edge_features: concatenated node embeddings
        """
        masked_edge = np.array(self.masked_graph_edge_list)
        edge_features = np.linalg.norm(
            self.emb_data_buffer_graph[masked_edge[:, 0], :] - self.emb_data_buffer_graph[masked_edge[:, 1], :], axis=1)
        return torch.from_numpy(masked_edge), torch.from_numpy(self.emb_data_buffer_graph), torch.from_numpy(edge_features)

    def load_embedding_data(self):
        # return the embedding and edges
        self.emb_data_buffer = np.zeros(
            (len(self._embedding_data_files), np.load(self._embedding_data_files[0]).shape[0]))
        for emb_idx, emb_item in enumerate(self._embedding_data_files):
            self.emb_data_buffer[emb_idx] = np.load(emb_item)

    def build_graph(self):
        # build the graph data from the items based on the heatmap
        # | x - - - - x |
        # | - x - - x - |
        # | - - x x - - |
        # | x - - x x - |
        # | - - - - x x |
        # | - x - - - x |
        # search each row and build the edge when the iou exceed the threshold
        # row == column since it's fully connected graph
        print(self._iou_map.shape)
        row, column = self._iou_map.shape
        if row > 1000:
            count = 1000
        else:
            count = row
        for ri in range(count):
            ri_connections = list(
                np.where(self._iou_map[ri] >= self._iou_edge_th)[0])
            if len(ri_connections) < 1:
                self._g_edge_list.append([ri, ri])
            else:
                for ci in ri_connections:
                    self._g_edge_list.append([ri, ci])

    def __len__(self):
        return len(self.node_idx)

    def __getitem__(self, i):
        # return the query node information
        data_collections = {}
        node_idx = self.node_idx[i]
        # query node idx in original sequence
        data_collections['idx'] = node_idx
        data_collections['emb_vec'] = self.emb_data_buffer_query[i]
        data_collections['iou_label'] = self.query_iou_map[i]
        return data_collections

    # Graph nodes embedding
    def embed_graph_nodes(self,):
        embed_model = self._embed_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embed_model.to(device)
        embed_model.eval()
        # loop and embed
        print("generating the embeddings...")
        with torch.no_grad():
            embedding_list = []
            for td_item in tqdm(os.listdir(self._pointcloud_data_root)):
                x = load_data_item(self._data_root, td_item)
                batch = {}
                coords = ME.utils.sparse_quantize(coordinates=x['coords'],
                                                  quantization_size=0.01)
                bcoords = ME.utils.batched_coordinates([coords]).to(device)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(device)
                batch['coords'] = bcoords
                batch['features'] = feats

                batch['images'] = x['image'].unsqueeze(0).to(device)
                emb = embed_model(batch)
                embedding = emb['embedding']
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding = np.squeeze(embedding.cpu().numpy())
                # embedding_fname = td_item.split(
                #     "/")[-1].replace(".h5", ".npy")
                embedding_list.append(list(embedding))
                # np.save(os.path.join(self._node_embedding_dir,
                #         embedding_fname), embedding)
            self.emb_data_buffer = np.array(embedding_list)
            scipy.io.savemat('/home/graham/datasets/fire/' + self._scene + '/embeddings.mat', {'mat': self.emb_data_buffer})


class S3EGNN_Dataset(DGLDataset):
    def __init__(self, edges, node_feats, edge_feats, node_labels):
        super().__init__(name='karate_club')
        self.node_feats = node_feats
        self.node_labels = node_labels
        self.edge_feats = edge_feats
        self.edges = edges

    def process(self):
        node_features = torch.from_numpy(self.node_feats)
        node_labels = torch.from_numpy(self.node_labels)
        edge_features = torch.from_numpy(self.edge_feats)
        edges_src = torch.from_numpy(self.edges[:, 0])
        edges_dst = torch.from_numpy(self.edges[:, 1])

        self.graph = dgl.graph((edges_src, edges_dst),
                               num_nodes=self.node_feats.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.node_feats.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


if __name__ == "__main__":
    import yaml

    # "/mnt/Data/Shared/data_generate_2"
    data_root = "/mnt/Data/Datasets/S3E_SLAM/data_generate_1"
    patch_idx = 12
    m_configs = yaml.safe_load(open("config/vit_config.yaml", 'r'))
    dataset = GNNDataset(data_root, m_configs, str(
        m_configs['model']['patch_size']))
    data = dataset[30]
    idx = data['idx']
    print(data['emb_vec'].shape)
    print(data['iou_label'].shape)
