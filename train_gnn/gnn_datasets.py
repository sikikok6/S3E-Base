import random
from datasets.ScanNetDataset import ScanNetDataset
from datasets.ScanNetDataset import TrainTransform, ScanNetDataset, TrainSetTransform

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from datasets.ScanNetDataset import ScanNetDataset
from scipy.spatial import distance
import os
import numpy as np
import torch
from gnn_utils import get_poses, in_sorted_array

train_sim_mat = []
query_sim_mat = []
database_sim_mat = []


def split_batch(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


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
        if self.type == "train":
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
            # current_batch = []
            # current_batch.append(ndx)
            self.batch_idx.append(ndx)
        random.shuffle(self.batch_idx)
        self.batch_idx = split_batch(self.batch_idx, self.batch_size)

    def generate_smoothap_val_batches(self):
        self.batch_idx = []
        for ndx in range(len(self.dataset)):
            # current_batch = []
            # current_batch.append(ndx)
            self.batch_idx.append(ndx)
        self.batch_idx = split_batch(self.batch_idx, self.batch_size)


def make_smoothap_collate_fn(
    dataset: ScanNetDataset, mink_quantization_size=None, val=None
):
    # set_transform: the transform to be applied to all batch elements

    def collate_fn(data_list):
        # Constructs a batch object
        global train_sim_mat
        global database_sim_mat
        global query_sim_mat
        num = 20
        positives_mask = []
        hard_positives_mask = []
        negatives_mask = []
        most_positives_mask = []

        positives_masks = []
        hard_positives_masks = []
        negatives_masks = []
        most_positives_masks = []

        labels = [[e[1]] for e in data_list]

        # dataset.queries[labels[0]]

        if val != "val":
            for i in range(len(labels)):
                labels[i].extend(train_sim_mat[labels[i][0]][1 : num + 1])
                positives_mask = [
                    in_sorted_array(e, dataset.queries[labels[i][0]].positives)
                    for e in labels[i]
                ]
                hard_positives_mask = [
                    in_sorted_array(e, dataset.queries[labels[i][0]].hard_positives)
                    for e in labels[i]
                ]
                positives_mask[0] = True
                negatives_mask = [not item for item in positives_mask]
                positives_masks.append(positives_mask)
                negatives_masks.append(negatives_mask)
                hard_positives_masks.append(hard_positives_mask)
                # positives_mask = torch.tensor([positives_mask])
                # negatives_mask = torch.tensor([negatives_mask])
                # hard_positives_mask = torch.tensor([hard_positives_mask])
                most_positives_mask = [
                    in_sorted_array(e, dataset.queries[labels[i][0]].most_positive)
                    for e in labels[i]
                ]
                # most_positives_mask = torch.tensor([most_positives_mask])
                most_positives_masks.append(most_positives_mask)
        else:
            for i in range(len(labels)):
                labels[i].extend(query_sim_mat[labels[i][0]][:num])
                positives_mask = [
                    in_sorted_array(e, dataset.queries[labels[i][0]].positives)
                    for e in labels[i]
                ]
                # hard_positives_mask = [in_sorted_array(
                #     e, dataset.queries[labels[0]].hard_positives) for e in labels]
                positives_mask[0] = True
                negatives_mask = [not item for item in positives_mask]
                positives_masks.append(positives_mask)
                negatives_masks.append(negatives_mask)
                hard_positives_masks.append(hard_positives_mask)
                # positives_mask = torch.tensor([positives_mask])
                # negatives_mask = torch.tensor([negatives_mask])
                # hard_positives_mask = torch.tensor([hard_positives_mask])
                most_positives_mask = [
                    in_sorted_array(e, dataset.queries[labels[i][0]].most_positive)
                    for e in labels[i]
                ]
                # most_positives_mask = torch.tensor([most_positives_mask])
                most_positives_masks.append(most_positives_mask)

        neighbours = []
        if val == "val":
            for i in range(len(labels)):
                neighbours.append(query_sim_mat[labels[i][0]][:num])
            # neighbours_temp = [database_sim_mat[item][1:num+1]
            #                    for item in labels[1:]]
            # neighbours.extend(neighbours_temp)

        else:
            # neighbours = [dataset.get_neighbours(item)[:10] for item in labels]
            # for i in labels[:1]:
            #     temp = train_sim_mat[i][1:num+1]
            for i in range(len(labels)):
                neighbours.append(train_sim_mat[labels[i][0]][:num])

            # neighbours.append(temp)

        valid_mask = torch.sum(torch.tensor(positives_masks), -1) != 1
        if val == "val":
            valid_mask = torch.ones(valid_mask.shape, dtype=torch.bool)
        valid_mask = torch.ones(valid_mask.shape, dtype=torch.bool)

        positives_masks = torch.tensor(positives_masks)[valid_mask]
        negatives_masks = torch.tensor(negatives_masks)[valid_mask]
        hard_positives_masks = torch.tensor(hard_positives_masks)[valid_mask]
        labels = torch.tensor(labels)[valid_mask]
        neighbours = torch.tensor(neighbours)[valid_mask]
        most_positives_masks = torch.tensor(most_positives_masks)[valid_mask]

        return (
            positives_masks,
            negatives_masks,
            hard_positives_masks,
            labels,
            neighbours,
            most_positives_masks,
            None,
        )

        # return torch.tensor(positives_masks)[valid_mask], torch.tensor(negatives_masks)[valid_mask], torch.tensor(hard_positives_masks)[valid_mask], torch.tensor(labels)[valid_mask], torch.tensor(neighbours)[valid_mask], torch.tensor(most_positives_masks)[valid_mask], None

    return collate_fn


def make_dataloader(params, project_params):
    datasets = {}
    dataset_folder = os.path.join(project_params.dataset_dir, project_params.scene)
    train_file = "pickle/" + project_params.scene + "_train_overlap.pickle"
    test_file = "pickle/" + project_params.scene + "_test_overlap.pickle"

    train_transform = TrainTransform(1)
    train_set_transform = TrainSetTransform(1)

    train_poses = get_poses("train", project_params)
    test_poses = get_poses("test", project_params)
    np.save("./train_poses.npy", train_poses)
    np.save("./test_poses.npy", test_poses)

    train_embeddings = np.load("./embeddings/gnn_pre_train_embeddings.npy")
    test_embeddings = np.load("./embeddings/gnn_pre_test_embeddings.npy")

    database_len = len(test_embeddings) // 2 if len(test_embeddings) < 4000 else 3000
    database_embeddings = test_embeddings[:database_len]
    query_embeddings = test_embeddings[database_len:]

    # train_embeddings = train_poses[:, :3]
    # database_embeddings = test_poses[:database_len, :3]

    global train_sim_mat
    global database_sim_mat
    global query_sim_mat

    train_sim = distance.cdist(train_embeddings, train_embeddings)
    database_sim = distance.cdist(database_embeddings, database_embeddings)
    # database_sim = train_sim.copy()
    # query_sim = distance.cdist(database_embeddings, train_embeddings)
    query_sim = database_sim.copy()
    print(train_sim)
    print(query_sim.shape)

    train_sim_mat = np.argsort(train_sim).tolist()
    print(np.array(train_sim_mat))
    database_sim_mat = np.argsort(database_sim).tolist()
    query_sim_mat = np.argsort(query_sim).tolist()
    # train_sim_mat = []

    # t_ = []
    # for i in range(len(train_sim_mat)):
    #     t = np.array(train_sim_mat[i])
    #     t_.append(list(t[(t != i) & (t // 1000 != i // 1000)]))

    # train_sim_mat = t_

    # database_sim_mat = train_sim_mat.copy()

    datasets["train"] = ScanNetDataset(
        dataset_folder, train_file, train_transform, set_transform=train_set_transform
    )
    datasets["val"] = ScanNetDataset(dataset_folder, test_file, None)

    val_transform = None

    dataloaders = {}
    train_sampler = BatchSampler(datasets["train"], batch_size=16, type="train")
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_smoothap_collate_fn(datasets["train"], 0.01)
    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=params.num_workers,
        pin_memory=False,
    )

    if "val" in datasets:
        val_sampler = BatchSampler(datasets["val"], batch_size=16, type="val")
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_smoothap_collate_fn(datasets["val"], 0.01, "val")
        dataloaders["val"] = DataLoader(
            datasets["val"],
            batch_sampler=val_sampler,
            collate_fn=val_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True,
        )
    return dataloaders
