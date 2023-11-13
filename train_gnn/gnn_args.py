import argparse
import os
import numpy as np
from gnn_utils import (
    load_minkLoc_model,
    load_resnet_model,
    get_embeddings_3d,
    get_embeddings_resnet,
)


def gnn_args():
    parser = argparse.ArgumentParser(description="gnn fine project")

    parser.add_argument(
        "-p",
        "--project_dir",
        type=str,
        help="project dir",
        default="/home/ubuntu-user/S3E-backup",
    )

    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        help="datasets dir",
        default="/home/ubuntu-user/S3E-backup/datasetfiles/datasets",
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="config file dir",
        default="config/config_baseline_multimodal.txt",
    )

    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        help="model file dir",
        default="models/minkloc3d.txt",
    )

    parser.add_argument(
        "-pw",
        "--pcl_weights",
        type=str,
        help="pcl_weights dir",
        default="weights_pcl/model_MinkFPN_GeM_20230515_1254fire_max_66.5.pth",
    )

    parser.add_argument(
        "-rw",
        "--rgb_weights",
        type=str,
        help="rgb_weights dir",
        default="weights_rgb/fire_rgb_best_weight/model_MinkLocRGB_20230515_05004_epoch_current_recall79.2_best.pth",
    )

    parser.add_argument("-s", "--scene", type=str, help="scene name", default="fire")
    project_args = parser.parse_args()
    return project_args


def gnn_config(project_args):
    config = os.path.join(project_args.project_dir, project_args.config_file)
    model_config = os.path.join(project_args.project_dir, project_args.model_file)
    rgb_weights = os.path.join(project_args.dataset_dir, project_args.rgb_weights)
    pcl_weights = os.path.join(project_args.dataset_dir, project_args.pcl_weights)

    print("config: ", config)
    print("model config: ", model_config)
    print("rgb weights: ", rgb_weights)
    print("pcl weights: ", pcl_weights)

    device = "cuda"

    # load minkloc
    mink_model, params = load_minkLoc_model(
        config, model_config, pcl_weights, rgb_weights, project_args
    )
    mink_model.to(device)
    # load resnet model
    resnet_model = load_resnet_model()
    resnet_model.to(device)
    return mink_model, resnet_model, params


def ap_embeddings(mink_model, params, project_args):
    embs = np.array(
        get_embeddings_3d(mink_model, params, project_args, "cuda", "train")
    )
    print(embs.shape)
    np.save("./embeddings/gnn_pre_train_embeddings.npy", embs)
    test_embs = np.array(
        get_embeddings_3d(mink_model, params, project_args, "cuda", "test")
    )
    np.save("./embeddings/gnn_pre_test_embeddings.npy", test_embs)


def resnet_embeddings(resnet_model, params, project_args):
    resnet_embs, resnet_feature_maps = get_embeddings_resnet(
        resnet_model, params, project_args, "cuda", "train"
    )
    print(resnet_embs.shape)
    np.save("./embeddings/gnn_resnet_train_embeddings.npy", np.array(resnet_embs))
    np.save(
        "./embeddings/gnn_resnet_train_feature_maps.npy", np.array(resnet_feature_maps)
    )
    resnet_test_embs, resnet_test_feature_maps = get_embeddings_resnet(
        resnet_model, params, project_args, "cuda", "test"
    )
    np.save("./embeddings/gnn_resnet_test_embeddings.npy", np.array(resnet_test_embs))
    np.save(
        "./embeddings/gnn_resnet_test_feature_maps.npy",
        np.array(resnet_test_feature_maps),
    )
