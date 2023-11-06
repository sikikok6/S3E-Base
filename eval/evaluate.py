# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import scipy
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import tqdm

from misc.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.oxford import image4lidar
from datasets.augmentation import ValRGBTransform

DEBUG = False


def evaluate(model, device, params, silent=True):
    # Run evaluation on all eval datasets
    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params,
                                database_sets, query_sets, silent=silent)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    # mat = scipy.io.loadmat('../gnn_embs.mat')
    # print(mat['graph'].shape)
    # database_embeddings = [mat['graph']]

    # mat = scipy.io.loadmat('./embeddings.mat')
    # database_embeddings = [mat['database'][0]]
    # query_embeddings = [mat['query'][0]]
    # print(mat['query'].shape)

    # for set in tqdm.tqdm(database_sets, disable=silent):
    #     database_embeddings.append(
    #         get_latent_vectors(model, set, device, params))
    # #
    # # data = {}
    # #
    # for set in tqdm.tqdm(query_sets, disable=silent):
    #     query_embeddings.append(get_latent_vectors(model, set, device, params))
    # # embedding_dir = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/npy'
    # emb_files = os.listdir(embedding_dir)
    # emb_files.sort()
    # embeddings = []
    # for e in tqdm.tqdm(emb_files):
    #     embeddings.append(np.load(os.path.join(embedding_dir, e))[0].tolist())
    # database_embeddings = [embeddings[:1000]]
    # query_embeddings = [embeddings[1000:]]

    for set in tqdm.tqdm(database_sets, disable=silent):
        database_embeddings.append(
            get_latent_vectors(model, set, device, params))

    data = {}

    for set in tqdm.tqdm(query_sets, disable=silent):
        query_embeddings.append(get_latent_vectors(model, set, device, params))

    # database_embeddings = query_embeddings
    # data['database'] = database_embeddings
    # data['query'] = query_embeddings
    # scipy.io.savemat('./embeddings_test.mat',data)
    # query_embeddings = database_embeddings

    for i in tqdm.tqdm(range(len(query_sets)), disable=silent):
        for j in range(len(query_sets)):
            # if i == j:
            #     continue
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    return stats


def load_data_item(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)

    result = {}
    if params.use_cloud:
        pc = np.fromfile(file_path.replace('lddf', 'david'), dtype=np.float64)
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
        lidar2image_ndx = {}
        for i in range(len(os.listdir(params.dataset_folder))):
            lidar2image_ndx[i] = [i]
        img = image4lidar(file_name, params.image_path,
                          '.color.png', lidar2image_ndx, k=1)
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result['image'] = transform(img)

    return result


def get_latent_vectors(model, set, device, params):
    # Adapted from original PointNetVLAD code

    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    model.eval()
    embeddings_l = []
    for elem_ndx in set:
        x = load_data_item(set[elem_ndx]["query"], params)

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
                batch['images'] = x['image'].unsqueeze(0).to(device)

            x = model(batch)
            embedding = x['embedding']

            # embedding is (1, 256) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(
                    embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


'''

def get_latent_vectors(model, set, device, params):
    # Adapted from original PointNetVLAD code
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((3000, 1))

    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    model.eval()
    embeddings_l = []
    for elem_ndx in tqdm.tqdm(set):
        x_m_4 = load_data_item(
            set[max(elem_ndx - 4, 0)]["query"].replace('graham', 'lddf'), params)
        x_m_3 = load_data_item(
            set[max(elem_ndx - 3, 0)]["query"].replace('graham', 'lddf'), params)
        x_m_2 = load_data_item(
            set[max(elem_ndx - 2, 0)]["query"].replace('graham', 'lddf'), params)
        x_m_1 = load_data_item(
            set[max(elem_ndx - 1, 0)]["query"].replace('graham', 'lddf'), params)
        x = load_data_item(set[elem_ndx]["query"].replace(
            'graham', 'lddf'), params)
        x_p_1 = load_data_item(
            set[min(elem_ndx + 1, len(set) - 1)]["query"].replace('graham', 'lddf'), params)

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
                batch['images'] = torch.stack([x_m_2['image'].to(device), x_m_1['image'].to(device), x['image'].to(
                    device)]).unsqueeze(0).to(device)

            starter.record()
            x = model(batch)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)
            timings[elem_ndx] = curr_time
            embedding = x['embedding']

            # embedding is (1, 256) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(
                    embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    print("avg={}\n".format(timings.mean()))
    return embeddings

'''


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in tqdm.tqdm(range(len(queries_output))):
        # i is query element ndx
        # {'query': path, 'northing': , 'easting': }
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]

        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=100)
        # true_indices = indices[0]

        # to_remove = []
        # for index, j in enumerate(indices[0]):
        #     if abs(i - j) <= 100 or j > i:
        #         to_remove.append(index)
        # true_indices= np.delete(true_indices, to_remove)
        # indices = np.array([true_indices])
        # if len(indices) == 0:
        # print("zero")

        for j in range(min(len(indices[0]), 25)):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    print(num_evaluated)
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'],
              stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model on RobotCar dataset')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False,
                        help='Trained model weights')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(
            args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    stats = evaluate(model, device, params, silent=False)
    print_eval_stats(stats)
