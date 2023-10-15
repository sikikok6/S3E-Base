import torch
import dgl
import numpy as np
import os
import scipy
import MinkowskiEngine as ME
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from misc.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.oxford import image4lidar
from datasets.augmentation import ValRGBTransform

from tqdm import tqdm


class S3EGraph():
    def __init__(self, args, root, scene):
        self.args = args
        self.root = root
        self.scene = scene
        self.pos_graph = None
        self.neg_graph = None
        self.embeddings = None
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print('Device: {}'.format(device))
        self.device = device
        self.generate_embeddings()
        self.construct_pos_graph()
    
    def load_data_item(self, prefix, file_name):
        # returns Nx3 matrix
        file_path = os.path.join(prefix, 'pointcloud_4096', file_name)

        result = {}
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == 4096 * 3, "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        result['coords'] = pc

        lidar2image_ndx = {}
        for i in range(len(os.listdir(prefix))):
            lidar2image_ndx[i] = [i]
        img = image4lidar(file_name, '/home/graham/fire/color', '.color.png', lidar2image_ndx, k=1)
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result['image'] = transform(img)

        return result
    
    def generate_embeddings(self):
      
        # if os.path.exists(os.path.join(self.root, self.scene, 'embeddings.mat')):
        if os.path.exists('./eval/embeddings.mat'):
            # self.embeddings = scipy.io.loadmat(os.path.join(self.root, self.scene, 'embeddings.mat'))['database'][0]
            self.embeddings = scipy.io.loadmat('./eval/embeddings.mat')
            self.embeddings = torch.tensor(self.embeddings['database'][0]).to(self.device)
            return
        params = MinkLocParams(self.args.config, self.args.model_config)
        params.print()


        model = model_factory(params)
        if self.args.weights is not None:
            assert os.path.exists(self.args.weights), 'Cannot open network weights: {}'.format(self.args.weights)
            print('Loading weights: {}'.format(self.args.weights))
            model.load_state_dict(torch.load(self.args.weights, map_location=self.device))
        print("generating the embeddings...")

        model.to(self.device)
        model.eval()
        

        with torch.no_grad():
            embedding_list = []
            pointcloud_list = os.listdir(os.path.join(self.root, self.scene, 'pointcloud_4096'))
            pointcloud_list.sort()
            for td_item in tqdm(pointcloud_list):
                x = self.load_data_item(os.path.join(self.root, self.scene), td_item)
                batch = {}
                coords = ME.utils.sparse_quantize(coordinates=x['coords'],
                                                  quantization_size=0.01)
                bcoords = ME.utils.batched_coordinates([coords]).to(self.device)
                # Assign a dummy feature equal to 1 to each point
                feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(self.device)
                batch['coords'] = bcoords
                batch['features'] = feats

                batch['images'] = x['image'].unsqueeze(0).to(self.device)
                emb = model(batch)
                embedding = emb['embedding']
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding = np.squeeze(embedding.cpu().numpy())
                # embedding_fname = td_item.split(
                #     "/")[-1].replace(".h5", ".npy")
                embedding_list.append(list(embedding))
                # np.save(os.path.join(self._node_embedding_dir,
                #         embedding_fname), embedding)
            self.embeddings = np.array(embedding_list)
            scipy.io.savemat('/home/graham/datasets/fire/' + self.scene + '/embeddings.mat', {'embeddings': self.embeddings})
            self.embeddings = torch.tensor(embedding_list).to(self.device)
        

    def construct_pos_graph(self):
        pose_dir = os.path.join(self.root, self.scene, 'pose')
        pose_list = os.listdir(pose_dir)
        pose_list.sort()
        poses = []
        translations = []

        for i in range(len(pose_list)):
            R = np.loadtxt(os.path.join(self.root, self.scene,'pose', pose_list[i]))
            poses.append(R)
            # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
            translations.append(R[:3, -1])
        tree = KDTree(translations)
        ind_p = tree.query_radius(translations, r=0.15)
        pos_src, pos_dst = [], []
        for index, item in enumerate(ind_p):
            for i in item:
                pos_src.append(index)
                pos_dst.append(i)
        self.pos_graph = dgl.graph((torch.tensor(pos_src), torch.tensor(pos_dst)), num_nodes=len(translations)).to(self.device)

    def construct_neg_graph(self):
        src, dst = self.pos_graph.cpu().edges()
        neg_src = src.repeat_interleave(16)
        neg_dst = torch.randint(0, self.pos_graph.num_nodes(), (len(src) * 16,))
        num_nodes = self.pos_graph.num_nodes()
        self.neg_graph = dgl.graph((neg_src, neg_dst), num_nodes=num_nodes).to(self.device)
