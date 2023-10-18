import datetime
import wandb
import torch.nn.functional as F
from loss import SmoothAP, C2F
import dgl
import pickle
import torch.nn as nn
import torch
import numpy as np
import tqdm
from gnn_utils import load_minkLoc_model, make_dataloader, myGNN, get_embeddings_3d, get_embeddings, get_poses, cal_trans_rot_errorv1
import os
import argparse

parser = argparse.ArgumentParser(description="gnn fine project")

parser.add_argument('-p', '--project_dir', type=str,
                    help='project dir', default='/home/ubuntu-user/S3E-backup')

parser.add_argument('-d', '--dataset_dir', type=str,
                    help='datasets dir', default='/home/ubuntu-user/S3E-backup/datasetfiles/datasets')

parser.add_argument('-c', '--config_file', type=str,
                    help='config file dir', default='config/config_baseline_multimodal.txt')

parser.add_argument('-m', '--model_file', type=str,
                    help='model file dir', default='models/minkloc3d.txt')

parser.add_argument('-pw', '--pcl_weights', type=str,
                    help='pcl_weights dir', default='weights_pcl/model_MinkFPN_GeM_20230515_1254fire_max_66.5.pth')

parser.add_argument('-rw', '--rgb_weights', type=str,
                    help='rgb_weights dir', default='weights_rgb/fire_rgb_best_weight/model_MinkLocRGB_20230515_05004_epoch_current_recall79.2_best.pth')

parser.add_argument('-s', '--scene', type=str,
                    help='scene name', default='fire')
project_args = parser.parse_args()

current_time = datetime.datetime.now()
time_string = current_time.strftime("%m%d_%H%M")

run = wandb.init(project="SE3-Backup-V3-Model",
                 name="Exp_"+time_string +"only_pose_error_fullgnn")

config = os.path.join(project_args.project_dir, project_args.config_file)
model_config = os.path.join(project_args.project_dir, project_args.model_file)
rgb_weights = os.path.join(project_args.dataset_dir, project_args.rgb_weights)
pcl_weights = os.path.join(project_args.dataset_dir, project_args.pcl_weights)

print("config: ", config)
print("model config: ", model_config)
print("rgb weights: ", rgb_weights)
print("pcl weights: ", pcl_weights)

run = wandb.init(project="SE3-Backup-V2-Model",
                 name="Exp_"+time_string + "fineloss")

# config = '/home/ubuntu-user/S3E-backup/config/config_baseline_multimodal.txt'
# model_config = '//home/ubuntu-user/S3E-backup/models/minkloc3d.txt'
# pcl_weights = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/weights_pcl/model_MinkFPN_GeM_20230515_1254fire_max_66.5.pth'
# rgb_weights = '/home/ubuntu-user/S3E-backup/datasetfiles/datasets/weights_rgb/fire_rgb_best_weight/model_MinkLocRGB_20230515_05004_epoch_current_recall79.2_best.pth'


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print('Device: {}'.format(device))


# # load minkloc
mink_model, params = load_minkLoc_model(
    config, model_config, pcl_weights, rgb_weights, project_args)
mink_model.to(device)

'''If Not Save,Run'''
# embs = np.array(get_embeddings_3d(
#     mink_model, params, project_args, 'cuda', 'train'))
# np.save('./gnn_pre_train_embeddings.npy', embs)
# test_embs = np.array(get_embeddings_3d(
#     mink_model, params, project_args, 'cuda', 'test'))
# np.save('./gnn_pre_test_embeddings.npy', test_embs)


# load dataloaders
dataloaders = make_dataloader(params, project_args)


evaluate_database_pickle = os.path.join(project_args.dataset_dir, project_args.scene,
                                        'pickle', project_args.scene+'_evaluation_database.pickle')
evaluate_query_pickle = os.path.join(project_args.dataset_dir, project_args.scene,
                                     'pickle', project_args.scene+'_evaluation_query.pickle')

# load evaluate file
# with open('/home/graham/datasets/fire/pickle/fire_evaluation_database.pickle', 'rb') as f:
with open(evaluate_database_pickle, 'rb') as f:
    database = pickle.load(f)

# with open('/home/graham/datasets/fire/pickle/fire_evaluation_query.pickle', 'rb') as f:
with open(evaluate_query_pickle, 'rb') as f:
    query = pickle.load(f)

train_iou_file = os.path.join(
    project_args.dataset_dir, 'iou_', "train_" + project_args.scene + "_iou.npy")
test_iou_file = os.path.join(
    project_args.dataset_dir, 'iou_', "test_" + project_args.scene + "_iou.npy")

iou = torch.tensor(
    np.load(train_iou_file), dtype=torch.float)
# iou = torch.tensor(iou / np.linalg.norm(iou, axis=1, keepdims=True))
test_iou = torch.tensor(np.load(test_iou_file), dtype=torch.float)
# test_iou = torch.tensor(
#     test_iou / np.linalg.norm(test_iou, axis=1, keepdims=True))

train_overlap = os.path.join(project_args.dataset_dir, project_args.scene,
                             'pickle', project_args.scene+'_train_overlap.pickle')
test_overlap = os.path.join(project_args.dataset_dir, project_args.scene,
                            'pickle', project_args.scene+'_test_overlap.pickle')

# load iou file
with open(train_overlap, 'rb') as f:
    train_iou = pickle.load(f)

with open(test_overlap, 'rb') as f:
    test_iou = pickle.load(f)


gt = [[0.001 for _ in range(len(train_iou))] for _ in range(len(train_iou))]
test_gt = [[0. for _ in range(3000)] for _ in range(2000)]

gt = torch.tensor(gt)


for i in train_iou:
    gt[i][i] = 1.
    for p in train_iou[i].positives:
        gt[i][p] = 1.


if params.model_params.model == "MinkLocMultimodal":
    in_feats = 512
else:
    in_feats = 256
# for save model
model_path = './savemodel/'+time_string+'_my_dgl_model.pt'

model = myGNN(in_feats, 256, 128)
model.to('cuda')

opt = torch.optim.Adam(
    [{'params': model.parameters(), 'lr': 0.001, 'weight_decay': 0.001}])
loss = None
recall = None
smoothap = SmoothAP()
# c2f = C2F()
d = {'loss': loss}


embs = np.load('./gnn_pre_train_embeddings.npy')
pose_embs = get_poses("train", project_args)

test_embs = np.load('./gnn_pre_test_embeddings.npy')
test_pose_embs = get_poses("test", project_args)
print(len(test_embs))
# embs = np.hstack((embs, embs))
# test_embs = np.hstack((test_embs, test_embs))
database_len = len(test_embs) // 2 if len(test_embs) < 4000 else 3000
database_embs = torch.tensor(test_embs[:database_len].copy())
query_embs = torch.tensor(test_embs[database_len:].copy())
# test_embs = torch.tensor(test_embs).to('cuda')
database_embs = database_embs.to('cuda')
query_embs = query_embs.to('cuda')


embs = torch.tensor(embs).to('cuda')
# Add Here For Pose
pose_embs = torch.tensor(pose_embs, dtype=torch.float32).to('cuda')
test_pose_embs = torch.tensor(test_pose_embs, dtype=torch.float32).to('cuda')

criterion = nn.MSELoss().to('cuda')

criterion_pose = nn.MSELoss().to('cuda')

pdist = nn.PairwiseDistance(p=2)
cos = nn.CosineSimilarity(dim=1).cuda()

max_ = 0.
# labels = range(len(feat))
with tqdm.tqdm(range(200), position=0, desc='epoch', ncols=60) as tbar:
    for epoch in tbar:
        # loss status
        loss = 0.
        losses = []
        cnt = 0.
        num_evaluated = 0.
        recall = [0] * 50
        train_loss_dic = {}
        # train_loss_dic['ap'] = []
        # train_loss_dic['mse1'] = []
        train_loss_dic['pos'] = []
        train_loss_dic['ori'] = []
        train_loss_dic['pos_ori'] = []
        train_loss_dic['pos_error'] = []
        train_loss_dic['ori_error'] = []

        with tqdm.tqdm(
                dataloaders['train'], position=1, desc='batch', ncols=60) as tbar2:

            src = np.tile(np.arange(51), 51)
            dst = np.repeat(np.arange(51), 51)

            g = dgl.graph((src, dst))
            g = g.to('cuda')

            for pos_mask, neg_mask, hard_pos_mask, labels, neighbours, most_pos_mask, batch in tbar2:
                torch.cuda.empty_cache()
                cnt += 1
                model.train()

                with torch.enable_grad():
                    # batch = {e: batch[e].to(device) for e in batch}

                    ind = [labels[0]]
                    ind.extend(np.vstack(neighbours[0]).reshape((-1,)).tolist())

                    '''Key Here To Change Poses For Query'''
                    # ind_pose = ind.copy()
                    # ind_pose[0] = ind_pose[1]
                    pose_embeddings = pose_embs[ind]

                    # indx = torch.tensor(ind).view((-1,))[dst[:len(labels) - 1]]
                    # indy = torch.tensor(ind)[src[:len(labels) - 1]]
                    embeddings = embs[ind]
                    #embeddings = torch.cat((embs[ind], pose_embeddings), dim=1)

                    # gt_iou = gt[indx, indy].view((-1, 1))
                    # gt_iou_ = iou[indx, indy].view((-1, 1)).cuda()

                    A, e, pos_pred, ori_pred = model(g, embeddings)

                    # query_embeddings = torch.repeat_interleave(
                    #     A[0].unsqueeze(0), len(labels) - 1, 0)
                    # database_embeddings = A[1:len(labels)]
                    # sim_mat = cos(query_embeddings, database_embeddings)
                    # # sim_mat = nn.functional.normalize(sim_mat, 2, 0)
                    # d1 = database_embeddings.repeat(
                    #     1, len(database_embeddings), 1).squeeze()
                    # d2 = database_embeddings.repeat_interleave(
                    #     len(database_embeddings), 0)
                    # database_sim_mat = cos(d1, d2).view(
                    #     (len(database_embeddings), len(database_embeddings)))
                    # sim_mat[sim_mat < 0] = 0
                    # sim_mat = torch.matmul(query_embs, database_embs.T).squeeze()

                    # loss_affinity_1 = criterion(
                    #     e[:len(labels) - 1], gt_iou.cuda())

                    # hard_sim_mat = sim_mat[pos_mask.squeeze()[1:]]
                    # hard_pos_mask[0][0] = True
                    # hard_p_mask = hard_pos_mask[pos_mask].unsqueeze(0)

                    # ap_coarse = smoothap(sim_mat, pos_mask)
                    # ap_fine = smoothap(hard_sim_mat, hard_p_mask)

                    # calculate poss loss
                    #gt_pose = pose_embs[labels[0]]
                    gt_pose = pose_embeddings
                    pos_true = gt_pose[:,:3]
                    ori_true = gt_pose[:,3:]

                    ori_pred = F.normalize(ori_pred, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)

                    loss_pos = F.mse_loss(pos_pred, pos_true)
                    loss_ori = F.mse_loss(ori_pred, ori_true)


                    #delta for ap
                    delta = 1

                    #alpha for mse1
                    alpha = 1
                    #beta for ori
                    beta = 1
                    #cardi for pos
                    cardi = 1
                    #gamma for pos and ori
                    gamma = 1

                    # train_loss_ap = 1 - (0.7*ap_coarse + 0.3*ap_fine)

                    # train_loss_mse1 = loss_affinity_1
                    #Here have cardi
                    train_loss_pos = cardi * loss_pos
                    #Here have beta
                    train_loss_ori = beta * loss_ori

                    train_loss_pos_ori = train_loss_pos + train_loss_ori

                    # losses.append(
                    #     1 - (0.7*ap_coarse + 0.3*ap_fine) + (30 * loss_affinity_1))

                    losses.append(train_loss_pos_ori)
                    
                    # train_loss_dic['ap'].append(delta*train_loss_ap.item())
                    # train_loss_dic['mse1'].append(alpha * train_loss_mse1.item())
                    train_loss_dic['pos'].append(train_loss_pos.item())
                    train_loss_dic['ori'].append(train_loss_ori.item())
                    train_loss_dic['pos_ori'].append(gamma * train_loss_pos_ori.item())
                    
                    
                    pos_pred_np = pos_pred.detach().cpu().numpy()
                    ori_pred_np = ori_pred.detach().cpu().numpy()
                    gt_pred_np =  gt_pose.detach().cpu().numpy()

                    train_pred_pose = np.hstack((pos_pred_np, ori_pred_np))

                    trans_error, rot_error = cal_trans_rot_errorv1(
                            train_pred_pose[0], gt_pred_np[0])
                    
                    train_loss_dic['pos_error'].append(trans_error)
                    train_loss_dic['ori_error'].append(rot_error)

    
                    loss += losses[-1].item()
                    if cnt % 128 == 0 or cnt == len(train_iou):
                        a = torch.vstack(losses)
                        a = torch.where(torch.isnan(
                            a), torch.full_like(a, 0), a)
                        loss_smoothap = torch.mean(a)
                        loss_smoothap.backward()
                        opt.step()
                        opt.zero_grad()
                        losses = []
            count = tbar2.n
            # print(f"cnt is {cnt},count is {count}")
            sum_dict = {}
            for key, value in train_loss_dic.items():
                total_sum = sum(value)
                sum_dict[key] = total_sum

            # print(f"Epoch {epoch}:Ap_Loss:{sum_dict['ap']/float(count)}")
            # wandb.log({'Ap_Loss': sum_dict['ap']/float(count)}, step=epoch)

            # print(f"Epoch {epoch}:Mse1_Loss:{sum_dict['mse1']/float(count)}")
            # wandb.log({'Mse1_Loss': sum_dict['mse1']/float(count)}, step=epoch)

            print(f"Epoch {epoch}:Pos_Loss:{sum_dict['pos']/float(count)}")
            wandb.log({'Pos_Loss': sum_dict['pos']/float(count)}, step=epoch)

            print(f"Epoch {epoch}:Ori_Loss:{sum_dict['ori']/float(count)}")
            wandb.log({'Ori_Loss': sum_dict['ori']/float(count)}, step=epoch)

            print(
                f"Epoch {epoch}:Pos_And_Ori_Loss:{sum_dict['pos_ori']/float(count)}")
            wandb.log(
                {'Pos_And_Ori_Loss': sum_dict['pos_ori']/float(count)}, step=epoch)

            print(f"Epoch {epoch}:Train_Average_Loss:{loss/float(count)}")
            wandb.log({'Train_Average_Loss': loss/float(count)}, step=epoch)

            print(f"Epoch {epoch}:Pos_Error:{sum_dict['pos_error']/float(count)}")
            wandb.log({'Pos_Error': sum_dict['pos_error']/float(count)}, step=epoch)

            print(f"Epoch {epoch}:Ori_Error:{sum_dict['ori_error']/float(count)}")
            wandb.log({'Ori_Error': sum_dict['ori_error']/float(count)}, step=epoch)



            with torch.no_grad():
                recall = [0] * 50
                num_evaluated = 0
                top1_similarity_score = []
                one_percent_retrieved = 0
                threshold = max(int(round(2000/100.0)), 1)

                model.eval()
                t_loss = 0.
                trans_loss = 0
                rotation_loss = 0

                val_loss_dic = {}
                val_loss_dic['pos'] = []
                val_loss_dic['ori'] = []
                val_loss_dic['pos_error'] = []
                val_loss_dic['ori_error'] = []

                src = np.tile(np.arange(51), 51)
                dst = np.repeat(np.arange(51), 51)

                g = dgl.graph((src, dst))
                g = g.to('cuda')
                with tqdm.tqdm(dataloaders['val'], position=1, desc='batch', ncols=50) as tbar3:
                    for pos_mask, neg_mask, hard_pos_mask, labels, neighbours, most_pos_mask, batch in tbar3:
                        ind = [labels[0]]
                        # ind = labels
                        ind.extend(
                            np.vstack(neighbours[0]).reshape((-1,)).tolist())

                        embeddings = torch.vstack(
                            (query_embs[ind[0]], database_embs[ind[1:]]))
                        # ind_pose = ind.copy()
                        # ind_pose[0] = ind_pose[1]

                        test_pose_embeddings = test_pose_embs[ind]
                        # embeddings = torch.cat(
                        #     (embeddings, test_pose_embeddings), dim=1)
                        embeddings = embeddings

                        A, e, test_pos_pred, test_ori_pred = model(
                            g, embeddings)

                        '''Pose Loss Cal'''
                        test_gt_pose = test_pose_embeddings
                        # calculate poss loss

                        test_pos_true = test_gt_pose[:,:3]
                        test_ori_true = test_gt_pose[:,3:]
                        test_ori_pred = F.normalize(test_ori_pred, p=2, dim=1)
                        test_ori_true = F.normalize(test_ori_true, p=2, dim=1)
                        loss_pos = F.mse_loss(test_pos_pred, test_pos_true)
                        loss_ori = F.mse_loss(test_ori_pred, test_ori_true)

                        val_loss_dic['pos'].append(loss_pos.item())
                        val_loss_dic['ori'].append(loss_ori.item())
                        

                        test_pred_pose = np.hstack(
                            (test_pos_pred.cpu().numpy(), test_ori_pred.cpu().numpy()))

                        trans_error, rot_error = cal_trans_rot_errorv1(
                            test_pred_pose[0], test_gt_pose.cpu().numpy()[0])
                        
                        val_loss_dic['pos_error'].append(trans_error)
                        val_loss_dic['ori_error'].append(rot_error)


                        # database_embeddings = A[1:len(labels)]

                        # q = A[0].unsqueeze(0)

                        # query_embeddings = torch.repeat_interleave(
                        #     q, len(labels) - 1, 0)
                        # # sim_mat = torch.matmul(q, database_embs.T).squeeze()

                        # sim_mat = cos(query_embeddings, database_embeddings)

                        # rank = torch.argsort((-sim_mat).squeeze())

                        # # loss_smoothap = smoothap(sim_mat, pos_mask)
                        # # t_loss += loss_smoothap.item() / 2000

                        # # true_neighbors = gt_test
                        # true_neighbors = query[0][labels[0]][0]
                        # if len(true_neighbors) == 0:
                        #     continue
                        # num_evaluated += 1

                        # flag = 0
                        # for j in range(len(rank)):
                        #     # if rank[j] == 0:
                        #     #     flag = 1
                        #     #     continue
                        #     if labels[1:][rank[j]] in true_neighbors:
                        #         if j == 0:
                        #             similarity = sim_mat[rank[j]]
                        #             top1_similarity_score.append(similarity)
                        #         recall[j - flag] += 1
                        #         break
                    
                    val_sum_dict = {}
                    for key, value in val_loss_dic.items():
                        total_sum = sum(value)
                        val_sum_dict[key] = total_sum
                    evanums = tbar3.n
                    
                    # t_loss = t_loss.detach().cpu().numpy()
                    print(f"val_pos_loss:{val_sum_dict['pos']/evanums}")
                    wandb.log(
                        {'val_pos_loss': val_sum_dict['pos']/evanums}, step=epoch)
                    
                    print(f"val_ori_loss:{val_sum_dict['ori']/evanums}")
                    wandb.log(
                        {'val_ori_loss': val_sum_dict['ori']/evanums}, step=epoch)

                    print(f"val_trans_error:{val_sum_dict['pos_error']/evanums}")
                    wandb.log(
                        {'val_trans_error': val_sum_dict['pos_error']/evanums}, step=epoch)

                    print(f"val_rotation_error:{val_sum_dict['ori_error']/evanums}")
                    wandb.log(
                        {'val_rotation_error': val_sum_dict['ori_error']/evanums}, step=epoch)

                    # # one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
                    # recall = (np.cumsum(recall)/float(num_evaluated))*100
                    # max_ = max(max_, recall[0])
                    # print('recall\n', recall)

                    # print('max:', max_)
                    # wandb.log(
                    #     {'val_recall_Max': max_}, step=epoch)
                    # # print(gt_iou.view(-1,)[:len(pos_mask[0])])

        tbar.set_postfix({'train loss': loss/float(count)})

print("save model:")
model = model.to('cpu')
torch.save(model.state_dict(), model_path)
