
import os
from eval.evaluate import evaluate_dataset
import tqdm
import numpy as np
import torch
import torch.nn as nn
import pickle
from models.loss import BatchHardTripletLossWithMasks
import dgl
from loss import SmoothAP, Shrinkage_loss, Cali_loss
from datetime import datetime
# import warnings
# warnings.filterwarnings("ignore")


from gnn_utils import load_minkLoc_model, make_dataloader, myGNN, get_embeddings_3d 


config = '/home/david/code_1080/S3E-rgb/config/config_baseline_multimodal.txt'
model_config = '/home/david/code_1080/S3E-rgb/models/minklocrgb.txt'
weights = '/home/david/code_1080/S3E-rgb/weights/model_MinkLocRGB_20230821_173234_epoch_current_recall94.1_fire.pth'


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print('Device: {}'.format(device))


# # load minkloc
mink_model, params = load_minkLoc_model(config, model_config, weights)
mink_model.to(device)

# load dataloaders
dataloaders = make_dataloader(params)


# load evaluate file
# with open('/home/graham/datasets/fire/pickle/fire_evaluation_database.pickle', 'rb') as f:
with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_evaluation_database.pickle', 'rb') as f:
    database = pickle.load(f)

# with open('/home/graham/datasets/fire/pickle/fire_evaluation_query.pickle', 'rb') as f:
with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_evaluation_query.pickle', 'rb') as f:
    query = pickle.load(f)


# gt = torch.tensor(np.load('/home/graham/datasets/fire/iou_heatmap/train_iou_heatmap.npy'), dtype=torch.float)
# gt = torch.tensor(gt / np.linalg.norm(gt, axis=1, keepdims=True))
# test_gt = np.load('/home/graham/datasets/fire/iou_heatmap/test_iou_heatmap.npy')
# test_gt = torch.tensor(test_gt / np.linalg.norm(test_gt, axis=1, keepdims=True))


# load iou file
# with open('/home/graham/datasets/fire/pickle/train_iou.pickle', 'rb+') as f:
with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_train_overlap.pickle', 'rb') as f:
    train_iou = pickle.load(f)

with open('/home/ubuntu-user/S3E-backup/datasetfiles/datasets/fire/pickle/fire_test_overlap.pickle', 'rb') as f:
    test_iou = pickle.load(f)

# random_list = np.load('./random_list.npy')
# random_list.sort()
# test_list = random_list
# train_list = list(set(range(1101)) - set(test_list))

gt = [[0.001 for _ in range(len(train_iou))] for _ in range(len(train_iou))]
test_gt = [[0. for _ in range(1000)] for _ in range(1000)]

gt = torch.tensor(gt)
test_gt = torch.tensor(test_gt)

for i in train_iou:
    gt[i][i] = 1.
    for p in train_iou[i].positives:
        gt[i][p] = 1.

np.save('./gt_mat.npy', gt.numpy())

for i in test_iou:
    if i > 1000:
        # test_gt[i][i] = 1.
        for p in test_iou[i-1000].positives:
            if p < 1000:
                test_gt[i][p] = 1.

np.save('test_gt.npy', test_gt.numpy())

model = myGNN(128, 256, 128)
# model = torch.nn.DataParallel(model)
model.to('cuda')
# mlpmodel = MLPModel(128, 512, 256)
# mlpmodel.to('cuda')

opt = torch.optim.Adam(
    [{'params': model.parameters(), 'lr': 0.001, 'weight_decay': 0.001}])
# opt = torch.optim.Adam([{'params':mlpmodel.parameters(), 'lr':0.001, 'weight_decay': 0.0001}])
# opt = torch.optim.Adam([{'params':model.parameters(), 'lr':0.001}, {'params':edge_regression.parameters(), 'lr':0.001}])
loss = None
recall = None
smoothap = SmoothAP()
# caliloss = Cali_loss()
d = {'loss': loss}


#
# embeddings = get_embeddings_3d(mink_model, params, device, 'test')
# np.save('./gnn_pre_test_embeddings.npy', embeddings)


embs = np.load('./gnn_pre_train_embeddings.npy')
test_embs = np.load('./gnn_pre_test_embeddings.npy')
database_embs = torch.tensor(test_embs[:1000].copy())
query_embs = torch.tensor(test_embs[1000:].copy())
# test_embs = torch.tensor(test_embs).to('cuda')
database_embs = database_embs.to('cuda')
query_embs = query_embs.to('cuda')


embs = torch.tensor(embs).to('cuda')

# temp = torch.repeat_interleave(embs_numpy, len(embs_numpy), 0) - torch.tile(embs_numpy, (len(embs_numpy), 1))
criterion = nn.MSELoss().to('cuda')
# shrinkage_loss = Shrinkage_loss(5, 0.2).to('cuda')
pdist = nn.PairwiseDistance(p=2)
cos = nn.CosineSimilarity(dim=1).cuda()

max_ = 0.
# labels = range(len(feat))
with tqdm.tqdm(range(100), position=0, desc='epoch', ncols=60) as tbar:
    for i in tbar:
        # loss status
        loss = 0.
        losses = []
        cnt = 0.
        num_evaluated = 0.
        recall = [0] * 50
        opt.zero_grad()
        with tqdm.tqdm(dataloaders['train'], position=1, desc='batch', ncols=60) as tbar2:

            for pos_mask, neg_mask, _,labels, neighbours, most_pos_mask, batch in tbar2:
                torch.cuda.empty_cache()
                cnt += 1
                model.train()
                # mlpmodel.train()

                with torch.enable_grad():
                    # batch = {e: batch[e].to(device) for e in batch}

                    src = np.array(
                        list(range(1, len(labels) * (len(labels) - 1) + 1)))
                    dst = np.repeat(list(range(len(labels))), len(labels) - 1)
                    # src = np.array(list(range(len(labels))))
                    # dst = np.repeat([0], len(labels))
                    # src, dst = src+dst, dst+src

                    # src = np.tile(list(range(len(labels))), len(labels))
                    # dst = np.repeat(list(range(len(labels))), len(labels))
                    g = dgl.graph((src, dst))
                    g = g.to('cuda')
                    ind = [labels[0]]
                    # ind = labels
                    ind.extend(np.vstack(neighbours).reshape((-1,)).tolist())
                    indx = torch.tensor(ind).view((-1,))[dst[:len(labels) - 1]]
                    indy = torch.tensor(ind)[src[:len(labels) - 1]]
                    # indx = torch.tensor(ind).view((-1,))[dst[:]]
                    # indy = torch.tensor(ind)[src[:]]
                    embeddings = embs[ind]
                    gt_iou = gt[indx, indy].view((-1, 1))
                    A, e = model(g, embeddings)
                    # A = mlpmodel(embeddings)
                    # query_embs = A[0].unsqueeze(0)
                    query_embeddings = torch.repeat_interleave(
                        A[0].unsqueeze(0), len(labels) - 1, 0)
                    database_embeddings = A[1:len(labels)]
                    sim_mat = cos(query_embeddings, database_embeddings)
                    # sim_mat = torch.matmul(query_embs, database_embs.T).squeeze()
                    loss_affinity_1 = criterion(
                        e[:len(labels) - 1], gt_iou.cuda())
                    # cal = caliloss(e[:len(labels) - 1], pos_mask, labels)
                    # loss_affinity_1 = shrinkage_loss(A[:len(labels)], gt_iou[:len(labels)])
                    # # sim_mat = 1 / pdist(pred[0], pred)
                    # print("criterion ", loss_affinity_1)
                    # print("smooth ", smoothap(sim_mat, pos_mask))
                    # tri = tri_loss(A[:len(labels)], pos_mask, neg_mask)
                    # print(tri)

                    losses.append(
                        smoothap(sim_mat, pos_mask))
                        # + loss_affinity_1)
                    # losses.append(smoothap(sim_mat, pos_mask, most_pos_mask) + cal)
                    # losses.append(tri)
                    # smoothap(sim_mat[:3], pos_mask[:, :4]) + smoothap(sim_mat[:5], pos_mask[:, :6]))
                    # loss_smoothap = smoothap(sim_mat, pos_mask) + loss_affinity_1

                    # smoothap(sim_mat, most_pos_mask))
                    loss += losses[-1].item()
                    if cnt % 32 == 0 or cnt == len(train_iou):
                        a = torch.vstack(losses)
                        a = torch.where(torch.isnan(
                            a), torch.full_like(a, 0), a)
                        loss_smoothap = torch.mean(a)
                        loss_smoothap.backward()
                        opt.step()
                        opt.zero_grad()
                        losses = []
                    #
                    rank = np.argsort(-sim_mat.detach().cpu().numpy())
                    true_neighbors = train_iou[labels[0]].positives
                    if len(true_neighbors) == 0:
                        continue
                    num_evaluated += 1

                    flag = 0
                    for j in range(len(rank)):
                        # if rank[j] == 0:
                        #     flag = 1
                        #     continue
                        if labels[1:][rank[j]] in true_neighbors:
                            # if j == 0:
                            #     similarity = sim_mat[rank[j]]
                            #     top1_similarity_score.append(similarity)
                            recall[j - flag] += 1
                            break
            # tbar2.set_postfix({'loss': loss_smoothap.item()})
            print(loss / cnt)
            recall = (np.cumsum(recall)/float(num_evaluated))*100
            print('train recall\n', recall)
            # print(pos_mask)
            # print(gt_iou.view(-1,)[:len(pos_mask[0])])

            t_loss = 0.

            with torch.no_grad():
                recall = [0] * 50
                num_evaluated = 0
                top1_similarity_score = []
                one_percent_retrieved = 0
                threshold = max(int(round(2000/100.0)), 1)

                model.eval()
                t_loss = 0.
                with tqdm.tqdm(dataloaders['val'], position=1, desc='batch', ncols=50) as tbar3:
                    for pos_mask, neg_mask, _,labels, neighbours, most_pos_mask, batch in tbar3:
                        # mlpmodel.eval()
                        # batch = {e: batch[e].to(device) for e in batch}

                        src = np.array(
                            list(range(1, len(labels) * (len(labels) - 1) + 1)))
                        dst = np.repeat(
                            list(range(len(labels))), len(labels) - 1)

                        # src = np.array(list(range(len(labels))))
                        # dst = np.repeat([0], len(labels))
                        # src, dst = src+dst, dst+src

                        # src = np.tile(list(range(len(labels))), len(labels))
                        # dst = np.repeat(list(range(len(labels))), len(labels))
                        g = dgl.graph((src, dst))
                        g = g.to('cuda')
                        ind = [labels[0]]
                        # ind = labels
                        ind.extend(
                            np.vstack(neighbours).reshape((-1,)).tolist())

                        embeddings = torch.vstack(
                            (query_embs[ind[0]], database_embs[ind[1:]]))

                        A, e = model(g, embeddings)

                        # A = mlpmodel(embeddings)

                        database_embeddings = A[1:len(labels)]

                        q = A[0].unsqueeze(0)

                        query_embeddings = torch.repeat_interleave(
                            q, len(labels) - 1, 0)
                        # sim_mat = torch.matmul(q, database_embs.T).squeeze()

                        sim_mat = cos(query_embeddings, database_embeddings)
                        # print(pos_mask)
                        # t_loss += smoothap(sim_mat, pos_mask, most_pos_mask)
                        # t_loss += tri_loss(A[:len(labels)], pos_mask, neg_mask)

                        # src = np.tile(list(range(len(labels))), len(labels))
                        # dst = np.repeat(list(range(len(labels))), len(labels))
                        # # src = list(range(len(labels)))
                        # # dst = list(np.repeat([0], len(pos_mask[0])))
                        # # src, dst = src + dst, dst + src
                        # g = dgl.graph((src, dst))
                        # g = g.to('cuda')

                        # embeddings = test_embs[labels]
                        # embeddings = test_embs[labels]
                        # embeddings = model.minkloc(batch)

                        # A2 = model(g, embeddings)

                        # gt_test = test_gt[torch.tensor(labels)[dst], torch.tensor(labels)[src]].view((-1,1))
                        # test_loss = criterion(A2, gt_test.cuda())
                        # t_loss += test_loss.item()
                        # np.save('test_gt.npy', gt_test.numpy())
                        # np.save('test_A2.npy', A2.detach().cpu().numpy())

                        # A2 = A2[:len(labels)]
                        # A2 = pdist(pred2[0], pred2)
                        # A2 = pred2[:len(labels)]
                        # sim_mat = torch.matmul(pred3[0], pred3.T)
                        # test_pred = pred2.detach().cpu().numpy()
                        # np.save('test_gnn_2.npy', test_pred)
                        # print(evaluate_dataset(None, 'cuda', None,database, query)['ave_recall'])

                        rank = torch.argsort((-sim_mat).squeeze())

                        # loss_smoothap = smoothap(sim_mat, pos_mask)
                        # t_loss += loss_smoothap.item() / 2000

                        # true_neighbors = gt_test
                        true_neighbors = query[0][labels[0]][0]
                        if len(true_neighbors) == 0:
                            continue
                        num_evaluated += 1

                        flag = 0
                        for j in range(len(rank)):
                            # if rank[j] == 0:
                            #     flag = 1
                            #     continue
                            if labels[1:][rank[j]] in true_neighbors:
                                if j == 0:
                                    similarity = sim_mat[rank[j]]
                                    top1_similarity_score.append(similarity)
                                recall[j - flag] += 1
                                break

                        # if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
                        #     one_percent_retrieved += 1

                    # one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
                    recall = (np.cumsum(recall)/float(num_evaluated))*100
                    max_ = max(max_, recall[0])
                    print('recall\n', recall)
                    # print(t_loss / num_evaluated)
                    print('max:', max_)
                    # print(gt_iou.view(-1,)[:len(pos_mask[0])])

        tbar.set_postfix({'train loss': loss / cnt})
