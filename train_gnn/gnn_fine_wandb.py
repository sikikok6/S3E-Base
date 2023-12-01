import wandb
import torch.nn.functional as F
from loss import SmoothAP, PoseLoss
import dgl
import torch.nn as nn
import torch
import numpy as np
import tqdm
from gnn_utils import cal_trans_rot_error, pose_delta

from gnn_datasets import (
    make_dataloader,
)
from gnn_models import myGNN
from gnn_args import ap_embeddings, gnn_args, gnn_config, resnet_embeddings
from gnn_data import get_gt, get_embeddings

project_args = gnn_args()

mink_model, resnet_model, params = gnn_config(project_args)


# run = wandb.init(project="SE3-Backup-V2-Model", name="Exp_" + time_string + "fineloss")


"""If Not Save ap Embeddings,Run"""
# ap_embeddings(mink_model, params, project_args)

"""If Not Save resnet Embeddings, Run"""
# resnet_embeddings(mink_model, params, project_args)


# load dataloaders
datasets, dataloaders = make_dataloader(params, project_args)

gt, test_gt = get_gt(project_args)
(
    embs,
    test_embs,
    pose_embs,
    test_pose_embs,
    database_embs,
    query_embs,
) = get_embeddings(project_args)


if params.model_params.model == "MinkLocMultimodal":
    in_feats = 512
else:
    in_feats = 256
# for save model
# model_path = "./savemodel/" + time_string + "_my_dgl_model.pt"

model = myGNN(1000, 256, 128)
model.to("cuda")

loss = None
recall = None
smoothap = SmoothAP()
d = {"loss": loss}


criterion = nn.MSELoss().to("cuda")

criterion_pose = nn.MSELoss().to("cuda")
poseLoss = PoseLoss(learn_beta=True).to("cuda")
globalPoseLoss = PoseLoss(learn_beta=True).to("cuda")

relativePoseLoss = PoseLoss(learn_beta=True).to("cuda")

opt = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": 0.0001, "weight_decay": 0.0005},
        {"params": poseLoss.parameters()},
        {"params": globalPoseLoss.parameters()},
        {"params": relativePoseLoss.parameters()},
    ]
)

pdist = nn.PairwiseDistance(p=2)
cos = nn.CosineSimilarity(dim=2).cuda()


max_ = 0.0
node_nums = 8
# labels = range(len(feat))
with tqdm.tqdm(range(200), position=0, desc="epoch", ncols=100) as tbar:
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
        train_loss_dic["del_train_tran_error"] = []
        train_loss_dic["del_train_ori_error"] = []
        train_loss_dic["abs_train_tran_error"] = []
        train_loss_dic["abs_train_ori_error"] = []

        with tqdm.tqdm(
            dataloaders["train"], position=1, desc="batch", ncols=90
        ) as tbar2:
            # for images, edge_index, edge_attr, y in tbar2:
            for labels, images, edge_index, y in tbar2:
                src = edge_index[0, 1]
                dst = edge_index[0, 0]

                g_fc = dgl.graph((src, dst))
                g_fc = g_fc.to("cuda")

                g_fc_arr = [g_fc] * len(images)
                g_fc_batch = dgl.batch(g_fc_arr)
                torch.cuda.empty_cache()

                model.train()

                with torch.enable_grad():
                    A, deltaPose, globalPose, e = model(
                        g_fc_batch, images.view((-1, 3, 240, 320)).cuda()
                    )

                    # calculate poss loss

                    poses = y.cuda()

                    ind = torch.tensor(labels)

                    indx = ind[..., dst]
                    indy = ind[..., src]
                    gt_iou = gt[indx, indy].view((-1, 1))
                    loss_affinity = criterion(e, gt_iou.cuda())

                    gt_neighbour_pose = poses[:, src].view((images.shape[0], -1, 6))
                    gt_query_pose = poses[:, dst].view((images.shape[0], -1, 6))

                    gt_delta_pose = pose_delta(gt_neighbour_pose, gt_query_pose).view(
                        (-1, 6)
                    )

                    srcPose = globalPose[:, src].view((-1, 6))
                    relPose = srcPose - deltaPose

                    loss_pose_rel, _, _ = relativePoseLoss(
                        relPose[:, :3],
                        relPose[:, 3:],
                        gt_query_pose.view((-1, 6))[:, :3],
                        gt_query_pose.view((-1, 6))[:, 3:],
                    )

                    loss_pose_global, _, _ = globalPoseLoss(
                        globalPose.view((-1, 6))[:, :3],
                        globalPose.view((-1, 6))[:, 3:],
                        poses.view((-1, 6))[:, :3],
                        poses.view((-1, 6))[:, 3:],
                    )

                    gt_delta_pose_trans = gt_delta_pose[:, :3]
                    gt_delta_pose_ori = gt_delta_pose[:, 3:]
                    # target_x = edge_attr.view((-1, 6))[:, :3].cuda()
                    # target_q = edge_attr.view((-1, 6))[:, 3:].cuda()

                    deltaPose_trans = deltaPose[:, :3]
                    deltaPose_ori = deltaPose[:, 3:]

                    loss_pose_delta, loss_pos, loss_ori = poseLoss(
                        deltaPose_trans,
                        deltaPose_ori,
                        gt_delta_pose_trans,
                        gt_delta_pose_ori,
                        # target_x,
                        # target_q,
                    )

                    q_valid = edge_index[:, 0] == 0

                    train_loss_pos = loss_pos
                    # Here have beta
                    train_loss_ori = loss_ori

                    # batch_loss = alpha * train_loss_mse1 + gamma * (
                    batch_loss = (
                        loss_pose_delta
                        + loss_pose_global
                        + loss_pose_rel
                        + loss_affinity
                    )

                    deltaPose = (
                        deltaPose.view((images.shape[0], 56, 6))[q_valid]
                        .view((images.shape[0], -1, 6))
                        .view((-1, 6))
                    )

                    gt_delta_pose = (
                        gt_delta_pose.view((images.shape[0], 56, 6))[q_valid]
                        .view((images.shape[0], -1, 6))
                        .view((-1, 6))
                    )

                    absPose = globalPose.view((-1, 6))
                    gt_absPose = y.view((-1, 6))

                    relPose = (
                        relPose.view((images.shape[0], 56, 6))
                        .view((images.shape[0], -1, 6))
                        .view((-1, 6))
                    )

                    queryPose = (
                        gt_query_pose.view((images.shape[0], 56, 6))
                        .view((images.shape[0], -1, 6))
                        .view((-1, 6))
                    )

                    trans_error, rot_error = cal_trans_rot_error(
                        relPose.detach().cpu().numpy(),
                        queryPose.detach().cpu().numpy(),
                    )

                    del_trans_error, del_rot_error = cal_trans_rot_error(
                        deltaPose.detach().cpu().numpy(),
                        gt_delta_pose.detach().cpu().numpy(),
                    )

                    abs_trans_error, abs_rot_error = cal_trans_rot_error(
                        absPose.detach().cpu().numpy(),
                        gt_absPose.detach().cpu().numpy(),
                    )

                    # )
                    # pred_q2r_pose, true_q2r_pose)

                    # trans_loss += trans_error
                    # rotation_loss += rot_error

                    # train_loss_dic["ap"].append(train_loss_ap.item())
                    # train_loss_dic["mse1"].append(alpha * train_loss_mse1.item())
                    train_loss_dic["pos"].append(train_loss_pos)
                    train_loss_dic["ori"].append(train_loss_ori)
                    train_loss_dic["train_tran_error"].append(trans_error)
                    train_loss_dic["train_ori_error"].append(rot_error)

                    train_loss_dic["del_train_tran_error"].append(del_trans_error)
                    train_loss_dic["del_train_ori_error"].append(del_rot_error)

                    train_loss_dic["abs_train_tran_error"].append(abs_trans_error)
                    train_loss_dic["abs_train_ori_error"].append(abs_rot_error)

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
            print(
                f"\033[1;32mEpoch {epoch}:del_tran_error:{sum_dict['del_train_tran_error']/float(count)}\033[0m"
            )
            print(
                f"\033[1;32mEpoch {epoch}:del_ori_error:{sum_dict['del_train_ori_error']/float(count)}\033[0m"
            )
            print(
                f"\033[1;32mEpoch {epoch}:abs_tran_error:{sum_dict['abs_train_tran_error']/float(count)}\033[0m"
            )
            print(
                f"\033[1;32mEpoch {epoch}:abs_ori_error:{sum_dict['abs_train_ori_error']/float(count)}\033[0m"
            )

            print(f"Epoch {epoch}:Pos_And_Ori_Loss:{sum_dict['pos_ori']/float(count)}")
            print(f"Epoch {epoch}:Train_Average_Loss:{loss/float(count)}")

            # # print(f"Epoch {epoch}:Ap_Loss:{sum_dict['ap']/float(count)}")
            # wandb.log({'Ap_Loss': sum_dict['ap']/float(count)}, step=epoch)

            # print(f"Epoch {epoch}:Mse1_Loss:{sum_dict['mse1']/float(count)}")
            # wandb.log({'Mse1_Loss': sum_dict['mse1']/float(count)}, step=epoch)
            #

            # wandb.log({"Pos_Loss": sum_dict["pos"] / float(count)}, step=epoch)
            # wandb.log({"Ori_Loss": sum_dict["ori"] / float(count)}, step=epoch)
            # wandb.log(
            #     {"train_tran_error": sum_dict["train_tran_error"] / float(count)},
            #     step=epoch,
            # )
            # wandb.log(
            #     {"train_ori_error": sum_dict["train_ori_error"] / float(count)},
            #     step=epoch,
            # )
            #
            # wandb.log(
            #     {"Pos_And_Ori_Loss": sum_dict["pos_ori"] / float(count)}, step=epoch
            # )

            # wandb.log({'Train_Average_Loss': loss/float(count)}, step=epoch)

            with torch.no_grad():
                model.eval()
                t_loss = 0.0
                trans_loss = 0
                rotation_loss = 0
                del_trans_loss = 0
                del_rotation_loss = 0
                abs_trans_loss = 0
                abs_rotation_loss = 0

                with tqdm.tqdm(
                    dataloaders["val"], position=1, desc="batch", ncols=90
                ) as tbar3:
                    for labels, images, edge_index, y in tbar3:
                        # for images, edge_index, edge_attr, y in tbar3:
                        src = edge_index[0, 1]
                        dst = edge_index[0, 0]

                        g_fc = dgl.graph((src, dst))
                        g_fc = g_fc.to("cuda")

                        g_fc_arr = [g_fc] * len(images)
                        g_fc_batch = dgl.batch(g_fc_arr)

                        A, deltaPose, globalPose, e = model(
                            g_fc_batch, images.view((-1, 3, 240, 320)).cuda()
                        )

                        """Pose Loss Cal"""

                        # calculate poss loss
                        poses = y.cuda()
                        q_valid = edge_index[:, 0] == 0

                        srcPose = globalPose[:, src].view((-1, 6))
                        relPose = srcPose - deltaPose

                        deltaPose = (
                            deltaPose.view((8, 56, 6))[q_valid]
                            .view((8, -1, 6))
                            .view((-1, 6))
                        )
                        gt_neighbour_pose = poses[:, src].view((images.shape[0], -1, 6))
                        gt_query_pose = poses[:, dst].view((images.shape[0], -1, 6))

                        gt_delta_pose = pose_delta(
                            gt_neighbour_pose, gt_query_pose
                        ).view((-1, 6))
                        gt_delta_pose = (
                            gt_delta_pose.view((images.shape[0], 56, 6))[q_valid]
                            .view((images.shape[0], -1, 6))
                            .view((-1, 6))
                        )

                        relPose = (
                            relPose.view((images.shape[0], 56, 6))[q_valid]
                            .view((images.shape[0], -1, 6))
                            .view((-1, 6))
                        )

                        queryPose = (
                            gt_query_pose.view((images.shape[0], 56, 6))[q_valid]
                            .view((images.shape[0], -1, 6))
                            .view((-1, 6))
                        )

                        absPose = globalPose[:, 0].view((-1, 6))
                        gt_absPose = y[:, 0].view((-1, 6))

                        trans_error, rot_error = cal_trans_rot_error(
                            relPose.detach().cpu().numpy(),
                            queryPose.detach().cpu().numpy(),
                        )

                        del_trans_error, del_rot_error = cal_trans_rot_error(
                            deltaPose.detach().cpu().numpy(),
                            gt_delta_pose.detach().cpu().numpy(),
                        )

                        abs_trans_error, abs_rot_error = cal_trans_rot_error(
                            absPose.detach().cpu().numpy(),
                            gt_absPose.detach().cpu().numpy(),
                        )

                        trans_loss += trans_error
                        rotation_loss += rot_error

                        del_trans_loss += del_trans_error
                        del_rotation_loss += del_rot_error

                        abs_trans_loss += abs_trans_error
                        abs_rotation_loss += abs_rot_error

                    evanums = tbar3.n
                    # t_loss = t_loss.detach().cpu().numpy()
                    # print(f"Val_poss_loss:{t_loss/evanums}")
                    print(f"\033[1;33mVal_trans_loss:{trans_loss/evanums}\033[0m")
                    print(f"\033[1;33mVal_rotation_loss:{rotation_loss/evanums}\033[0m")

                    print(
                        f"\033[1;33mdel_Val_trans_loss:{del_trans_loss/evanums}\033[0m"
                    )
                    print(
                        f"\033[1;33mdel_Val_rotation_loss:{del_rotation_loss/evanums}\033[0m"
                    )

                    print(
                        f"\033[1;33mabs_Val_trans_loss:{abs_trans_loss/evanums}\033[0m"
                    )
                    print(
                        f"\033[1;33mabs_Val_rotation_loss:{abs_rotation_loss/evanums}\033[0m"
                    )
                    # wandb.log(
                    #     {'Val_Avg_Poss_Loss': t_loss/evanums}, step=epoch)

                    # wandb.log({"Val_trans_loss": trans_loss / evanums}, step=epoch)
                    # wandb.log(
                    #     {"Val_rotation_loss": rotation_loss / evanums}, step=epoch
                    # )

        tbar.set_postfix({"train loss": loss / float(count)})
