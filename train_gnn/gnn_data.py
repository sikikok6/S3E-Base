import torch
import pickle
import os
import numpy as np
from gnn_utils import get_poses


def get_gt(project_args):
    # evaluate_database_pickle = os.path.join(
    #     project_args.dataset_dir,
    #     project_args.scene,
    #     "pickle",
    #     project_args.scene + "_evaluation_database.pickle",
    # )
    # evaluate_query_pickle = os.path.join(
    #     project_args.dataset_dir,
    #     project_args.scene,
    #     "pickle",
    #     project_args.scene + "_evaluation_query.pickle",
    # )

    # load evaluate file
    # with open('/home/graham/datasets/fire/pickle/fire_evaluation_database.pickle', 'rb') as f:
    # with open('/home/graham/datasets/fire/pickle/fire_evaluation_query.pickle', 'rb') as f:
    # with open(evaluate_database_pickle, "rb") as f:
    #     database = pickle.load(f)
    #
    # with open(evaluate_query_pickle, "rb") as f:
    #     query = pickle.load(f)

    # train_iou_file = os.path.join(
    #     project_args.dataset_dir, "iou_", "train_" + project_args.scene + "_iou.npy"
    # )
    # test_iou_file = os.path.join(
    #     project_args.dataset_dir, "iou_", "test_" + project_args.scene + "_iou.npy"
    # )

    # iou = torch.tensor(np.load(train_iou_file), dtype=torch.float)
    # iou = torch.tensor(iou / np.linalg.norm(iou, axis=1, keepdims=True))
    # test_iou = torch.tensor(np.load(test_iou_file), dtype=torch.float)
    # test_iou = torch.tensor(
    #     test_iou / np.linalg.norm(test_iou, axis=1, keepdims=True))

    train_overlap = os.path.join(
        project_args.dataset_dir,
        project_args.scene,
        "pickle",
        project_args.scene + "_train_overlap.pickle",
    )
    test_overlap = os.path.join(
        project_args.dataset_dir,
        project_args.scene,
        "pickle",
        project_args.scene + "_test_overlap.pickle",
    )

    # load iou file
    with open(train_overlap, "rb") as f:
        train_iou = pickle.load(f)

    with open(test_overlap, "rb") as f:
        test_iou = pickle.load(f)

    gt = [[0.001 for _ in range(len(train_iou))] for _ in range(len(train_iou))]
    test_gt = [[0.0 for _ in range(3000)] for _ in range(2000)]

    gt = torch.tensor(gt)

    for i in train_iou:
        gt[i][i] = 1.0
        for p in train_iou[i].positives:
            gt[i][p] = 1.0
    return gt, test_gt


def get_embeddings(project_args):
    embs = np.load("./embeddings/gnn_resnet_train_embeddings.npy")
    pose_embs = get_poses("train", project_args)

    test_embs = np.load("./embeddings/gnn_resnet_test_embeddings.npy")
    test_pose_embs = get_poses("test", project_args)
    print(len(test_embs))
    database_len = len(test_embs) // 2 if len(test_embs) < 4000 else 3000
    database_embs = torch.tensor(test_embs[:database_len].copy())
    query_embs = torch.tensor(test_embs[database_len:].copy())
    database_embs = database_embs.to("cuda")
    query_embs = query_embs.to("cuda")

    embs = torch.tensor(embs).to("cuda")
    # Add Here For Pose
    pose_embs = torch.tensor(pose_embs, dtype=torch.float32).to("cuda")
    test_pose_embs = torch.tensor(test_pose_embs, dtype=torch.float32).to("cuda")
    return embs, test_embs, pose_embs, test_pose_embs, database_embs, query_embs
