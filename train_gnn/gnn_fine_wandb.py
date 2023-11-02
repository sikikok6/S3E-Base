import datetime
import wandb
import torch.nn.functional as F
from loss import SmoothAP
import dgl
import pickle
import torch.nn as nn
import torch
import numpy as np
import tqdm
from gnn_utils import (
    load_minkLoc_model,
    make_dataloader,
    myGNN,
    get_embeddings_3d,
    get_embeddings,
    get_poses,
    cal_trans_rot_error,
    PoseLoss,
)
import os
import argparse


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

current_time = datetime.datetime.now()
time_string = current_time.strftime("%m%d_%H%M")


config = os.path.join(project_args.project_dir, project_args.config_file)
model_config = os.path.join(project_args.project_dir, project_args.model_file)
rgb_weights = os.path.join(project_args.dataset_dir, project_args.rgb_weights)
pcl_weights = os.path.join(project_args.dataset_dir, project_args.pcl_weights)

print("config: ", config)
print("model config: ", model_config)
print("rgb weights: ", rgb_weights)
print("pcl weights: ", pcl_weights)

run = wandb.init(project="SE3-Backup-V2-Model", name="Exp_" + time_string + "fineloss")

# config = '/home/ubuntu-user/S3E-backup/config/config_baseline_multimodal.txt'
# model_config = '//home/ubuntu-user/S3E-backup/models/minkloc3d.txt'
# pcl_weights = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/weights_pcl/model_MinkFPN_GeM_20230515_1254fire_max_66.5.pth'
# rgb_weights = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/weights_rgb/fire_rgb_best_weight/model_MinkLocRGB_20230515_05004_epoch_current_recall79.2_best.pth'


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device: {}".format(device))


# # load minkloc
mink_model, params = load_minkLoc_model(
    config, model_config, pcl_weights, rgb_weights, project_args
)
mink_model.to(device)

"""If Not Save,Run"""
# embs = np.array(get_embeddings_3d(
#     mink_model, params, project_args, 'cuda', 'train'))
# print(embs.shape)
# np.save('./gnn_pre_train_embeddings.npy', embs)
# test_embs = np.array(get_embeddings_3d(
#     mink_model, params, project_args, 'cuda', 'test'))
# np.save('./gnn_pre_test_embeddings.npy', test_embs)


# load dataloaders
dataloaders = make_dataloader(params, project_args)


evaluate_database_pickle = os.path.join(
    project_args.dataset_dir,
    project_args.scene,
    "pickle",
    project_args.scene + "_evaluation_database.pickle",
)
evaluate_query_pickle = os.path.join(
    project_args.dataset_dir,
    project_args.scene,
    "pickle",
    project_args.scene + "_evaluation_query.pickle",
)

# load evaluate file
# with open('/home/graham/datasets/fire/pickle/fire_evaluation_database.pickle', 'rb') as f:
with open(evaluate_database_pickle, "rb") as f:
    database = pickle.load(f)

# with open('/home/graham/datasets/fire/pickle/fire_evaluation_query.pickle', 'rb') as f:
with open(evaluate_query_pickle, "rb") as f:
    query = pickle.load(f)

train_iou_file = os.path.join(
    project_args.dataset_dir, "iou_", "train_" + project_args.scene + "_iou.npy"
)
test_iou_file = os.path.join(
    project_args.dataset_dir, "iou_", "test_" + project_args.scene + "_iou.npy"
)

iou = torch.tensor(np.load(train_iou_file), dtype=torch.float)
# iou = torch.tensor(iou / np.linalg.norm(iou, axis=1, keepdims=True))
test_iou = torch.tensor(np.load(test_iou_file), dtype=torch.float)
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


if params.model_params.model == "MinkLocMultimodal":
    in_feats = 512
else:
    in_feats = 256
# for save model
model_path = "./savemodel/" + time_string + "_my_dgl_model.pt"

model = myGNN(in_feats, 256, 128)
model.to("cuda")

opt = torch.optim.Adam(
    [{"params": model.parameters(), "lr": 0.0001, "weight_decay": 0.001}]
)
loss = None
recall = None
smoothap = SmoothAP()
d = {"loss": loss}


embs = np.load("./gnn_pre_train_embeddings.npy")
pose_embs = get_poses("train", project_args)

test_embs = np.load("./gnn_pre_test_embeddings.npy")
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

criterion = nn.MSELoss().to("cuda")

criterion_pose = nn.MSELoss().to("cuda")
pose_loss = PoseLoss().to("cuda")
pose_loss.eval()

pdist = nn.PairwiseDistance(p=2)
cos = nn.CosineSimilarity(dim=2).cuda()

max_ = 0.0
node_nums = 20
# labels = range(len(feat))
with tqdm.tqdm(range(200), position=0, desc="epoch", ncols=60) as tbar:
    for epoch in tbar:
        # loss status
        loss = 0.0
        batch_loss = None
        losses = []
        trans_loss = 0
        rotation_loss = 0
        train_loss_dic = {}
        train_loss_dic["ap"] = []
        train_loss_dic["mse1"] = []
        train_loss_dic["pos"] = []
        train_loss_dic["ori"] = []
        train_loss_dic["pos_ori"] = []
        train_loss_dic["train_tran_error"] = []
        train_loss_dic["train_ori_error"] = []

        with tqdm.tqdm(
            dataloaders["train"], position=1, desc="batch", ncols=60
        ) as tbar2:
            src = np.array(list(range(1, node_nums + 1)))
            dst = np.repeat(list(range(1)), node_nums)

            g = dgl.graph((src, dst))
            g = g.to("cuda")

            node = torch.tensor(list(range(node_nums + 1)))

            src = (
                torch.repeat_interleave(node.unsqueeze(0), node_nums + 1, 0)
                .view((1, -1))
                .squeeze()
            )

            dst = torch.repeat_interleave(node, node_nums + 1)

            g_fc = dgl.graph((src, dst))
            g_fc = g_fc.to("cuda")

            for (
                pos_mask,
                neg_mask,
                hard_pos_mask,
                labels,
                neighbours,
                most_pos_mask,
                batch,
            ) in tbar2:
                g_arr = [g] * len(pos_mask)
                g_batch = dgl.batch(g_arr)

                g_fc_arr = [g_fc] * len(pos_mask)
                g_fc_batch = dgl.batch(g_fc_arr)
                torch.cuda.empty_cache()
                model.train()

                with torch.enable_grad():
                    ind = labels.clone().detach()

                    """Key Here To Change Poses For Query"""
                    ind_pose = labels.clone().detach()
                    ind_pose[:, 0] = ind_pose[:, 1]
                    pose_embeddings = pose_embs[ind_pose[:, : node_nums + 1]]

                    trans_noise = (
                        torch.randn(
                            (pose_embeddings.shape[0], pose_embeddings.shape[1], 3)
                        )
                        .cuda()
                        .detach()
                    )
                    rot_noise = (
                        torch.randn(
                            (pose_embeddings.shape[0], pose_embeddings.shape[1], 4)
                        )
                        .cuda()
                        .detach()
                    )
                    pose_noise = torch.cat((trans_noise, rot_noise), dim=2)
                    pose_embeddings += pose_noise / 10

                    indx = ind[..., dst]
                    indy = ind[..., src]

                    embeddings = embs[ind]
                    gt_iou = gt[indx, indy].view((-1, 1))
                    gt_iou_ = iou[indx, indy].view((-1, 1)).cuda()

                    A, e, pos_pred, ori_pred, q2r = model(
                        g_fc_batch, g_batch, embeddings, pose_embeddings
                    )

                    query_embeddings = torch.repeat_interleave(
                        A[:, 0], node_nums, 0
                    ).view((len(pos_mask), node_nums, -1))

                    database_embeddings = A[:, 1 : node_nums + 1]

                    sim_mat = cos(query_embeddings, database_embeddings)

                    loss_affinity_1 = criterion(e, gt_iou.cuda())

                    # print(pos_mask.shape)
                    # hard_sim_mat = sim_mat[pos_mask[:, 1:]]
                    # print(hard_sim_mat.shape)
                    # hard_pos_mask[:, 0] = True
                    # hard_p_mask = hard_pos_mask[pos_mask].unsqueeze(0)

                    ap_coarse = smoothap(sim_mat, pos_mask)

                    # ap_fine = smoothap(hard_sim_mat, hard_p_mask)

                    # calculate poss loss
                    gt_pose = pose_embs[ind[:, :]].view((-1, 7))
                    pos_true = gt_pose[:, :3]
                    ori_true = gt_pose[:, 3:]

                    pos_q2r = q2r.view((-1, 7))[:, :3]
                    ori_q2r = F.normalize(q2r.view((-1, 7))[:, 3:], p=2, dim=1)

                    pos_q2r_true = (
                        pose_embs[ind[:, :]][:, 1:, :].contiguous().view((-1, 7))[:, :3]
                    )
                    ori_q2r_true = (
                        pose_embs[ind[:, :]][:, 1:, :].contiguous().view((-1, 7))[:, 3:]
                    )

                    ori_pred = F.normalize(ori_pred, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)
                    ori_q2r_true = F.normalize(ori_q2r_true, p=2, dim=1)

                    loss_pose, loss_pos, loss_ori = pose_loss(
                        pos_pred, ori_pred, pos_true, ori_true
                    )

                    loss_pose_q2r, loss_pos_q2r, loss_ori_q2r = pose_loss(
                        pos_q2r, ori_q2r, pos_q2r_true, ori_q2r_true
                    )

                    # alpha for mse1
                    alpha = 2
                    # beta for ori
                    beta = 1
                    # gamma for pos and ori
                    gamma = 2

                    # train_loss_ap = 1 - (0.7*ap_coarse + 0.3*ap_fine)

                    train_loss_ap = (1 - ap_coarse).mean()
                    train_loss_mse1 = loss_affinity_1 + train_loss_ap
                    train_loss_pos = loss_pos
                    # Here have beta
                    train_loss_ori = beta * loss_ori
                    # train_loss_pos_ori = train_loss_pos + train_loss_ori
                    train_loss_pos_ori = loss_pose + loss_pose_q2r

                    batch_loss = alpha * train_loss_mse1 + gamma * (loss_pose)
                    pred_pose = np.hstack(
                        (
                            pos_pred.detach().cpu().numpy(),
                            ori_pred.detach().cpu().numpy(),
                        )
                    )
                    pred_q2r_pose = np.hstack(
                        (pos_q2r.detach().cpu().numpy(), ori_q2r.detach().cpu().numpy())
                    )
                    true_q2r_pose = np.hstack(
                        (
                            pos_q2r_true.detach().cpu().numpy(),
                            ori_q2r_true.detach().cpu().numpy(),
                        )
                    )

                    trans_error, rot_error = cal_trans_rot_error(
                        pred_pose, gt_pose.detach().cpu().numpy()
                    )
                    # pred_q2r_pose, true_q2r_pose)

                    trans_loss += trans_error
                    rotation_loss += rot_error

                    train_loss_dic["ap"].append(train_loss_ap.item())
                    train_loss_dic["mse1"].append(alpha * train_loss_mse1.item())
                    train_loss_dic["pos"].append(train_loss_pos)
                    train_loss_dic["ori"].append(train_loss_ori)
                    train_loss_dic["pos_ori"].append(gamma * train_loss_pos_ori.item())
                    train_loss_dic["train_tran_error"].append(trans_error)
                    train_loss_dic["train_ori_error"].append(rot_error)

                    batch_loss.backward()
                    opt.step()
                    opt.zero_grad()

            count = tbar2.n
            sum_dict = {}
            for key, value in train_loss_dic.items():
                total_sum = sum(value)
                sum_dict[key] = total_sum

            print(f"Epoch {epoch}:Pos_Loss:{sum_dict['pos']/float(count)}")
            print(f"Epoch {epoch}:Ori_Loss:{sum_dict['ori']/float(count)}")
            print(
                f"\033[1;32mEpoch {epoch}:tran_error:{sum_dict['train_tran_error']/float(count)}\033[0m"
            )
            print(
                f"\033[1;32mEpoch {epoch}:ori_error:{sum_dict['train_ori_error']/float(count)}\033[0m"
            )

            print(f"Epoch {epoch}:Pos_And_Ori_Loss:{sum_dict['pos_ori']/float(count)}")
            print(f"Epoch {epoch}:Train_Average_Loss:{loss/float(count)}")

            # # print(f"Epoch {epoch}:Ap_Loss:{sum_dict['ap']/float(count)}")
            # wandb.log({'Ap_Loss': sum_dict['ap']/float(count)}, step=epoch)

            # print(f"Epoch {epoch}:Mse1_Loss:{sum_dict['mse1']/float(count)}")
            # wandb.log({'Mse1_Loss': sum_dict['mse1']/float(count)}, step=epoch)
            #
            wandb.log({"Pos_Loss": sum_dict["pos"] / float(count)}, step=epoch)
            #
            wandb.log({"Ori_Loss": sum_dict["ori"] / float(count)}, step=epoch)
            wandb.log(
                {"train_tran_error": sum_dict["train_tran_error"] / float(count)},
                step=epoch,
            )
            wandb.log(
                {"train_ori_error": sum_dict["train_ori_error"] / float(count)},
                step=epoch,
            )

            wandb.log(
                {"Pos_And_Ori_Loss": sum_dict["pos_ori"] / float(count)}, step=epoch
            )

            # wandb.log({'Train_Average_Loss': loss/float(count)}, step=epoch)

            with torch.no_grad():
                model.eval()
                t_loss = 0.0
                trans_loss = 0
                rotation_loss = 0

                src = np.array(list(range(1, node_nums + 1)))
                dst = np.repeat(list(range(1)), node_nums)

                g = dgl.graph((src, dst))
                g = g.to("cuda")

                src = (
                    torch.repeat_interleave(node.unsqueeze(0), node_nums + 1, 0)
                    .view((1, -1))
                    .squeeze()
                )

                dst = torch.repeat_interleave(node, node_nums + 1)

                g_fc = dgl.graph((src, dst))
                g_fc = g_fc.to("cuda")
                with tqdm.tqdm(
                    dataloaders["val"], position=1, desc="batch", ncols=60
                ) as tbar3:
                    for (
                        pos_mask,
                        neg_mask,
                        hard_pos_mask,
                        labels,
                        neighbours,
                        most_pos_mask,
                        batch,
                    ) in tbar3:
                        if len(labels) == 0:
                            continue

                        g_arr = [g] * len(pos_mask)
                        g_batch = dgl.batch(g_arr)

                        g_fc_arr = [g_fc] * len(pos_mask)
                        g_fc_batch = dgl.batch(g_fc_arr)

                        ind = labels.clone().detach()
                        # ind = labels

                        embeddings = torch.cat(
                            (
                                database_embs[ind[:, 0]].unsqueeze(1),
                                database_embs[ind[:, 1 : node_nums + 1]],
                            ),
                            dim=1,
                        )
                        # ind_pose = ind.copy()
                        ind_pose = labels.clone().detach()
                        ind_pose[:, 0] = ind_pose[:, 1]
                        test_pose_embeddings = test_pose_embs[
                            ind_pose[:, : node_nums + 1]
                        ]
                        trans_noise = (
                            torch.randn(
                                (
                                    test_pose_embeddings.shape[0],
                                    test_pose_embeddings.shape[1],
                                    3,
                                )
                            )
                            .cuda()
                            .detach()
                        )
                        rot_noise = (
                            torch.randn(
                                (
                                    test_pose_embeddings.shape[0],
                                    test_pose_embeddings.shape[1],
                                    4,
                                )
                            )
                            .cuda()
                            .detach()
                        )
                        pose_noise = torch.cat((trans_noise, rot_noise), dim=2)
                        test_pose_embeddings += pose_noise / 10

                        A, e, test_pos_pred, test_ori_pred, q2r = model(
                            g_fc_batch, g_batch, embeddings, test_pose_embeddings
                        )

                        """Pose Loss Cal"""
                        # test_gt_pose = pose_embs[ind[:, :]]
                        # test_gt_pose[:, 0] = test_pose_embs[ind[:, 0]]
                        # test_gt_pose = test_gt_pose.view((-1, 7))

                        test_gt_pose = test_pose_embs[ind[:, 0]]
                        # calculate poss loss

                        test_pos_true = test_gt_pose[:, :3]
                        test_ori_true = test_gt_pose[:, 3:]

                        test_ori_pred = F.normalize(
                            test_ori_pred.view(
                                (
                                    test_pose_embeddings.shape[0],
                                    test_pose_embeddings.shape[1],
                                    4,
                                )
                            )[:, 0],
                            p=2,
                            dim=1,
                        )
                        test_ori_true = F.normalize(test_ori_true, p=2, dim=1)

                        test_pos_pred = test_pos_pred.view(
                            (
                                test_pose_embeddings.shape[0],
                                test_pose_embeddings.shape[1],
                                3,
                            )
                        )[:, 0]

                        loss_pose, loss_pos, loss_ori = pose_loss(
                            test_pos_pred, test_ori_pred, test_pos_true, test_ori_true
                        )

                        # test_pos_pred, test_ori_pred

                        test_pred_pose = np.hstack(
                            (test_pos_pred.cpu().numpy(), test_ori_pred.cpu().numpy())
                        )

                        trans_error, rot_error = cal_trans_rot_error(
                            test_pred_pose, test_gt_pose.cpu().numpy()
                        )

                        trans_loss += trans_error
                        rotation_loss += rot_error

                    evanums = tbar3.n
                    # t_loss = t_loss.detach().cpu().numpy()
                    # print(f"Val_poss_loss:{t_loss/evanums}")
                    print(f"\033[1;33mVal_trans_loss:{trans_loss/evanums}\033[0m")
                    print(f"\033[1;33mVal_rotation_loss:{rotation_loss/evanums}\033[0m")
                    # wandb.log(
                    #     {'Val_Avg_Poss_Loss': t_loss/evanums}, step=epoch)

                    wandb.log({"Val_trans_loss": trans_loss / evanums}, step=epoch)
                    #
                    wandb.log(
                        {"Val_rotation_loss": rotation_loss / evanums}, step=epoch
                    )

        tbar.set_postfix({"train loss": loss / float(count)})

print("save model:")
model = model.to("cpu")
torch.save(model.state_dict(), model_path)
