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
pose_loss = PoseLoss(learn_beta=False).to("cuda")
pose_loss.eval()

opt = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": 0.0001, "weight_decay": 0.001},
        # {"params": criterion_pose.parameters()},
    ]
)

pdist = nn.PairwiseDistance(p=2)
cos = nn.CosineSimilarity(dim=2).cuda()


max_ = 0.0
node_nums = 10
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

            valid = src != dst
            src = src[valid]
            dst = dst[valid]

            g_fc = dgl.graph((src, dst))
            g_fc = g_fc.to("cuda")

            for (
                pos_mask,
                neg_mask,
                hard_pos_mask,
                labels,
                images,
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
                    # if epoch > 40:
                    #     model.freeze_except_decoder()
                    ind = labels.clone().detach()

                    indx = ind[..., dst]
                    indy = ind[..., src]

                    A, deltaPose = model(
                        g_fc_batch, g_batch, images.view((-1, 3, 240, 320)).cuda()
                    )

                    query_embeddings = torch.repeat_interleave(
                        A[:, 0], node_nums, 0
                    ).view((len(pos_mask), node_nums, -1))

                    # calculate poss loss
                    gt_pose = pose_embs[ind[:, :]].view((-1, 6))

                    gt_neighbour_pose = pose_embs[indy]
                    gt_query_pose = pose_embs[indx]

                    gt_delta_pose = pose_delta(gt_neighbour_pose, gt_query_pose).view(
                        (-1, 6)
                    )
                    gt_delta_pose_trans = gt_delta_pose[:, :3]
                    gt_delta_pose_ori = gt_delta_pose[:, 3:]

                    deltaPose_trans = deltaPose[:, :3]
                    deltaPose_ori = deltaPose[:, 3:]

                    loss_pose_delta, loss_pos, loss_ori = pose_loss(
                        deltaPose_trans,
                        deltaPose_ori,
                        gt_delta_pose_trans,
                        gt_delta_pose_ori,
                    )

                    gt_neighbour_pose = pose_embs[ind[:, 1:]]
                    gt_query_pose = pose_embs[ind[:, 0]].unsqueeze(1).repeat((1, 10, 1))

                    gt_delta_pose = pose_delta(gt_neighbour_pose, gt_query_pose).view(
                        (-1, 6)
                    )

                    deltaPose = (
                        deltaPose.view((labels.shape[0], -1, deltaPose.shape[-1]))[
                            :, :10
                        ]
                        .contiguous()
                        .view((-1, 6))
                    )
                    # print(deltaPose.shape)

                    train_loss_pos = loss_pos
                    # Here have beta
                    train_loss_ori = loss_ori

                    # batch_loss = alpha * train_loss_mse1 + gamma * (
                    batch_loss = loss_pose_delta
                    # loss_pose + loss_pose_q2r

                    trans_error, rot_error = cal_trans_rot_error(
                        # pred_pose,
                        # gt_pose.detach().cpu().numpy()
                        deltaPose.detach().cpu().numpy(),
                        gt_delta_pose.detach().cpu().numpy(),
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
                valid = src != dst
                src = src[valid]
                dst = dst[valid]

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
                        images,
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

                        A, deltaPose = model(
                            g_fc_batch, g_batch, images.view((-1, 3, 240, 320)).cuda()
                        )

                        """Pose Loss Cal"""

                        # calculate poss loss
                        test_gt_neighbour_pose = pose_embs[ind[:, 1:]]
                        test_gt_query_pose = (
                            test_pose_embs[ind[:, 0]].unsqueeze(1).repeat((1, 10, 1))
                        )
                        gt_delta_pose = pose_delta(
                            test_gt_neighbour_pose, test_gt_query_pose
                        ).view((-1, 6))

                        deltaPose = (
                            deltaPose.view((labels.shape[0], -1, deltaPose.shape[-1]))[
                                :, :10
                            ]
                            .contiguous()
                            .view((-1, 6))
                        )

                        trans_error, rot_error = cal_trans_rot_error(
                            deltaPose.detach().cpu().numpy(),
                            gt_delta_pose.detach().cpu().numpy(),
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

                    # wandb.log({"Val_trans_loss": trans_loss / evanums}, step=epoch)
                    # wandb.log(
                    #     {"Val_rotation_loss": rotation_loss / evanums}, step=epoch
                    # )

        tbar.set_postfix({"train loss": loss / float(count)})
