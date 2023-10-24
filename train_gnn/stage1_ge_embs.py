import tqdm
import numpy as np
import torch
import torch.nn as nn
import pickle
import dgl
from loss import SmoothAP, C2F
import os


from gnn_utils import load_minkLoc_model, make_dataloader, myGNN, get_embeddings_3d, get_embeddings,get_poses,inverse_poses
from test_utils import save_array_to_txt
print("HELLO")
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

config = os.path.join(project_args.project_dir, project_args.config_file)
model_config = os.path.join(project_args.project_dir, project_args.model_file)
rgb_weights = os.path.join(project_args.dataset_dir, project_args.rgb_weights)
pcl_weights = os.path.join(project_args.dataset_dir, project_args.pcl_weights)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print('Device: {}'.format(device))


# # load minkloc
mink_model, params = load_minkLoc_model(config, model_config, pcl_weights, rgb_weights)
mink_model.to(device)

'''If Not Save,Run'''
#embs = np.array(get_embeddings_3d(mink_model, params,'cuda','train'))
#np.save('./gnn_pre_train_embeddings.npy', embs)
# test_embs = np.array(get_embeddings_3d(mink_model, params,'cuda', 'test'))
#np.save( './gnn_pre_test_embeddings.npy', test_embs)


# load dataloaders
dataloaders = make_dataloader(params, project_args)


# load evaluate file
# with open('/home/graham/datasets/fire/pickle/fire_evaluation_database.pickle', 'rb') as f:
with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_evaluation_database.pickle', 'rb') as f:
    database = pickle.load(f)

# with open('/home/graham/datasets/fire/pickle/fire_evaluation_query.pickle', 'rb') as f:
with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_evaluation_query.pickle', 'rb') as f:
    query = pickle.load(f)


iou = torch.tensor(
    np.load('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/iou_/train_fire_iou.npy'), dtype=torch.float)
# iou = torch.tensor(iou / np.linalg.norm(iou, axis=1, keepdims=True))
test_iou = np.load('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/iou_/test_fire_iou.npy')
# test_iou = torch.tensor(
#     test_iou / np.linalg.norm(test_iou, axis=1, keepdims=True))


# load iou file
# with open('/home/graham/datasets/fire/pickle/train_iou.pickle', 'rb+') as f:
with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_train_overlap.pickle', 'rb') as f:
    train_iou = pickle.load(f)

with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_test_overlap.pickle', 'rb') as f:
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

model_path = '/home/ubuntu-user/S3E-backup/train_gnn/savemodel/1023_1356_stage1_dgl_model.pt'
model = myGNN(in_feats, 256, 128)
model.load_state_dict(torch.load(model_path))
model.to('cuda')

opt = torch.optim.Adam(
    [{'params': model.parameters(), 'lr': 0.001, 'weight_decay': 0.001}])
loss = None
recall = None
smoothap = SmoothAP()
#c2f = C2F()
d = {'loss': loss}


embs = np.load('./gnn_pre_train_embeddings.npy')
pose_embs = get_poses("train",project_args)

test_embs = np.load('./gnn_pre_test_embeddings.npy')
test_pose_embs  = get_poses("test",project_args)
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
#Add Here For Pose
pose_embs = torch.tensor(pose_embs,dtype=torch.float32).to('cuda')
test_pose_embs = torch.tensor(test_pose_embs,dtype=torch.float32).to('cuda')

criterion = nn.MSELoss().to('cuda')
pdist = nn.PairwiseDistance(p=2)
cos = nn.CosineSimilarity(dim=1).cuda()

max_ = 0.


train_pred_data_list = []
test_pred_data_list = []

with torch.no_grad():
    recall = [0] * 50
    num_evaluated = 0
    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(2000/100.0)), 1)


    model.eval()
    t_loss = 0.

    src = np.array(
        list(range(1, 51 * (51 - 1) + 1)))
    dst = np.repeat(
        list(range(51)), 51 - 1)
    
    # src = np.tile(np.arange(51), 51)
    # dst = np.repeat(np.arange(51), 51)

    g = dgl.graph((src, dst))
    g = g.to('cuda')

    with tqdm.tqdm(dataloaders['train'], position=1, desc='batch', ncols=60) as tbar2:
        for pos_mask, neg_mask, hard_pos_mask, labels, neighbours, most_pos_mask, batch in tbar2:
            ind = [labels[0]]
            ind.extend(np.vstack(neighbours).reshape((-1,)).tolist())
            #ind.extend(np.vstack(neighbours[0]).reshape((-1,)).tolist())

            '''Key Here To Change Poses For Query'''
            ind_pose = ind.copy()
            ind_pose[0] = ind_pose[1]
            pose_embeddings = pose_embs[ind_pose]
            embeddings =  torch.cat((embs[ind],pose_embeddings),dim=1)
            A, e = model(g, embeddings)
            train_pred_data_list.append((labels[0], A[0].cpu().numpy()))

        sorted_data = sorted(train_pred_data_list, key=lambda x: x[0])
        train_combine = np.array([features for _, features in sorted_data])
        np.save('./train_combine.npy', train_combine)
        




    with tqdm.tqdm(dataloaders['val'], position=1, desc='batch', ncols=50) as tbar3:
        for pos_mask, neg_mask, hard_pos_mask, labels, neighbours, most_pos_mask, batch in tbar3:
            ind = [labels[0]]
            # ind = labels
            ind.extend(np.vstack(neighbours).reshape((-1,)).tolist())
            #ind.extend(np.vstack(neighbours[0]).reshape((-1,)).tolist())

            ind_pose = ind.copy()
            ind_pose[0]=ind_pose[1]

            embeddings = torch.vstack((database_embs[ind[0]], embs[ind[1:]]))
            test_pose_embeddings =  pose_embs[ind_pose]
            embeddings = torch.cat((embeddings,test_pose_embeddings),dim=1)
            A, e, = model(g, embeddings)

            test_pred_data_list.append((labels[0], A[0].cpu().numpy()))

        sorted_data = sorted(test_pred_data_list, key=lambda x: x[0])
        test_combine = np.array([features for _, features in sorted_data])
        np.save('./test_combine.npy', test_combine)

