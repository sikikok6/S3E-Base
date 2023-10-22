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
from gnn_utils import load_minkLoc_model, make_dataloader, PoseReg ,get_embeddings_3d, get_embeddings, get_poses, cal_trans_rot_errorv1,PoseLoss
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

run = wandb.init(project="SE3-Backup-2stage-Dataset-stage2",
                 name="Exp_"+time_string +"fineloss")

config = os.path.join(project_args.project_dir, project_args.config_file)
model_config = os.path.join(project_args.project_dir, project_args.model_file)
rgb_weights = os.path.join(project_args.dataset_dir, project_args.rgb_weights)
pcl_weights = os.path.join(project_args.dataset_dir, project_args.pcl_weights)

print("config: ", config)
print("model config: ", model_config)
print("rgb weights: ", rgb_weights)
print("pcl weights: ", pcl_weights)


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

model_path = '/home/ubuntu-user/S3E-backup/train_gnn/savemodel/'+time_string+'_stage2_dgl_model.pt'
model = PoseReg(128,7)
model.to('cuda')

pose_loss = PoseLoss().to('cuda')
pose_loss.eval()

print(model)

opt = torch.optim.Adam(
    [{'params': model.parameters(), 'lr': 0.001, 'weight_decay': 0.001}])
loss = None
recall = None
smoothap = SmoothAP()
# c2f = C2F()
d = {'loss': loss}


embs = np.load('./train_combine.npy')
pose_embs = get_poses("train", project_args)

test_embs = np.load('./test_combine.npy')
test_pose_embs = get_poses("test", project_args)

print(len(test_embs))

embs = torch.tensor(embs).to('cuda')
test_embs = torch.tensor(test_embs).to('cuda')

# Add Here For Pose
pose_embs = torch.tensor(pose_embs, dtype=torch.float32).to('cuda')
test_pose_embs = torch.tensor(test_pose_embs, dtype=torch.float32).to('cuda')


max_ = 0.
# labels = range(len(feat))
with tqdm.tqdm(range(100), position=0, desc='epoch', ncols=60) as tbar:
    for epoch in tbar:
        # loss status
        loss = 0.
        losses = []
        cnt = 0.
        num_evaluated = 0.
        recall = [0] * 50
        train_loss_dic = {}
        train_loss_dic['pos'] = []
        train_loss_dic['ori'] = []
        train_loss_dic['pos_ori'] = []
        train_loss_dic['pos_error'] = []
        train_loss_dic['ori_error'] = []

        with tqdm.tqdm(
                dataloaders['train'], position=1, desc='batch', ncols=60) as tbar2:
            for pos_mask, neg_mask, hard_pos_mask, labels, neighbours, most_pos_mask, batch in tbar2:
                torch.cuda.empty_cache()
                cnt += 1
                model.train()

                with torch.enable_grad():
                    # batch = {e: batch[e].to(device) for e in batch}

                    ind = labels[0]
                    pose_embeddings = pose_embs[ind]
                    embeddings = embs[ind]
                    pos_pred,ori_pred = model(embeddings)


                    # calculate poss loss
                    gt_pose = pose_embs[ind]
                    pos_true = gt_pose[:3]
                    ori_true = gt_pose[3:]

                    ori_pred = F.normalize(ori_pred, p=2, dim=0)
                    ori_true = F.normalize(ori_true, p=2, dim=0)

                    
                    loss_pose, loss_pos, loss_ori = pose_loss(pos_pred.unsqueeze(
                        0), ori_pred.unsqueeze(0), pos_true.unsqueeze(0), ori_true.unsqueeze(0))

                    alpha = 1
                    beta = 1
                    
                    train_loss_pos = loss_pos
                    #Here have beta
                    train_loss_ori = loss_ori

                    train_loss_pos_ori = loss_pose

                    # losses.append(
                    #     1 - (0.7*ap_coarse + 0.3*ap_fine) + (30 * loss_affinity_1))

                    losses.append(train_loss_pos_ori)
                    
                    
                    
                    train_loss_dic['pos'].append(train_loss_pos)
                    train_loss_dic['ori'].append(train_loss_ori)
                    train_loss_dic['pos_ori'].append(train_loss_pos_ori.item())
                    
                    pos_pred_np = pos_pred.detach().cpu().numpy()
                    ori_pred_np = ori_pred.detach().cpu().numpy()
                    gt_pred_np =  gt_pose.detach().cpu().numpy()

                    train_pred_pose = np.hstack((pos_pred_np, ori_pred_np))

                    trans_error, rot_error = cal_trans_rot_errorv1(
                            train_pred_pose, gt_pred_np)
                    
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

            print(
                f"\033[1;32mEpoch {epoch}:tran_error:{sum_dict['pos_error']/float(count)}\033[0m")
            print(
                f"\033[1;32mEpoch {epoch}:ori_error:{sum_dict['ori_error']/float(count)}\033[0m")


            with torch.no_grad():
                model.eval()
                t_loss = 0.
                trans_loss = 0
                rotation_loss = 0

                val_loss_dic = {}
                val_loss_dic['pos'] = []
                val_loss_dic['ori'] = []
                val_loss_dic['pos_error'] = []
                val_loss_dic['ori_error'] = []
                val_loss_dic['pos_and_ori'] = []

                with tqdm.tqdm(dataloaders['val'], position=1, desc='batch', ncols=50) as tbar3:
                    for pos_mask, neg_mask, hard_pos_mask, labels, neighbours, most_pos_mask, batch in tbar3:
                        ind = labels[0]
                        

                        
                        test_embeddings = test_embs[ind]
                        test_pos_pred,test_ori_pred = model(test_embeddings)

                        # calculate poss loss
                        test_gt_pose = test_pose_embs[ind]
                        test_pos_true = test_gt_pose[:3]
                        test_ori_true = test_gt_pose[3:]
                        test_ori_pred = F.normalize(test_ori_pred, p=2, dim=0)
                        test_ori_true = F.normalize(test_ori_true, p=2, dim=0)

                        loss_pos = F.mse_loss(test_pos_pred, test_pos_true)
                        loss_ori = F.mse_loss(test_ori_pred, test_ori_true)

                        loss_pose, loss_pos, loss_ori = pose_loss(test_pos_pred.unsqueeze(
                            0), test_ori_pred.unsqueeze(0), test_pos_true.unsqueeze(0), test_ori_true.unsqueeze(0))
                        

                        val_loss_dic['pos'].append(loss_pos)
                        val_loss_dic['ori'].append(loss_ori)
                        val_loss_dic['pos_and_ori'].append(loss_pose.item())
                        

                        test_pred_pose = np.hstack(
                            (test_pos_pred.cpu().numpy(), test_ori_pred.cpu().numpy()))

                        trans_error, rot_error = cal_trans_rot_errorv1(
                            test_pred_pose, test_gt_pose.cpu().numpy())
                        
                        val_loss_dic['pos_error'].append(trans_error)
                        val_loss_dic['ori_error'].append(rot_error)

                    
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
                    
                    print(f"val_pos_and_ori_loss:{val_sum_dict['pos_and_ori']/evanums}")
                    wandb.log(
                        {'val_pos_and_ori_loss': val_sum_dict['pos_and_ori']/evanums}, step=epoch)
                    
                    print(f"\033[1;33mVal_trans_error:{val_sum_dict['pos_error']/evanums}\033[0m")
                    print(f"\033[1;33mVal_rotation_error:{val_sum_dict['ori_error']/evanums}\033[0m")

            
        tbar.set_postfix({'train loss': loss/float(count)})

print("save model:")
model = model.to('cpu')
torch.save(model.state_dict(), model_path)
