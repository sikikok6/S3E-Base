import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
import pytorch3d.transforms as p3dtrans
from models.resnetrgb import resnet18


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self, f):
        h = self.linear1(f)
        h = F.leaky_relu(h)
        h = self.linear2(h)
        h = F.leaky_relu(h)
        h = self.linear3(h)
        h = F.leaky_relu(h)
        h = self.linear4(h)
        # h = F.relu(h)
        h = torch.sigmoid(h)
        # h = F.relu(h)
        # return torch.sigmoid(h)
        return h


class featureMapGNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(featureMapGNN, self).__init__(*args, **kwargs)


class myGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(myGNN, self).__init__()

        self.MLP = MLP(256, 1)
        # self.BN = nn.BatchNorm1d(2*in_feats)
        self.conv1 = SAGEConv(2048, 1024, "mean")
        self.conv_pos_1 = SAGEConv(256, 256, "mean")
        self.conv_pos_2 = SAGEConv(256, 512, "mean")
        self.conv_feat_1 = SAGEConv(256, 256, "mean")
        self.conv_feat_2 = SAGEConv(256, 512, "mean")

        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(in_feats),
            nn.Linear(in_feats, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mlp3 = nn.Sequential(
            nn.BatchNorm1d(7),
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.Encoder = nn.Sequential(
            nn.BatchNorm1d(1024), nn.Linear(1024, 2048), nn.ReLU()
        )

        self.TransDecoder = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 3)
            # nn.Tanh()
        )

        self.OriDecoder = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4)
            # nn.Tanh()
        )

        self.EdgePose = nn.Sequential(nn.Linear(2048, 7))

    def edge_score(self, edges):
        h_u = edges.src["x"]
        h_v = edges.dst["x"]
        # score = self.MLP(torch.cat((h_u, h_v), 1))
        score = self.MLP(h_u - h_v)
        return {"score": score}

    def edge_pose(self, edges):
        h_u = edges.src["x"]
        h_v = edges.dst["x"]
        pose = self.EdgePose(torch.cat((h_u, h_v), dim=1))
        return {"pose": pose}

    def pose_multipy(self, ori_pose, delta_pose):
        index_delta = torch.tensor(
            [
                [[3, 0, 1, 2] for _ in range(delta_pose.shape[1])]
                for _ in range(delta_pose.shape[0])
            ]
        ).cuda()
        index_ori = torch.tensor([[[3, 0, 1, 2]]]).cuda()
        re_index = torch.tensor(
            [
                [[1, 2, 3, 0] for _ in range(delta_pose.shape[1])]
                for _ in range(delta_pose.shape[0])
            ]
        ).cuda()
        # ori_pos = ori_pose[:, :3].unsqueeze(1)
        # ori_rot = ori_pose[:, 3:].unsqueeze(1)
        ori_pos = ori_pose[:, :, :3]
        ori_rot = ori_pose[:, :, 3:]
        delta_pos = delta_pose[:, :, :3] + ori_pos
        delta_rot = p3dtrans.quaternion_multiply(
            F.normalize(ori_rot, p=2, dim=2).gather(2, index_ori),
            F.normalize(delta_pose[:, :, 3:], p=2, dim=2).gather(2, index_delta),
        )
        return torch.cat((delta_pos, delta_rot.gather(2, re_index)), dim=2)

    def freeze_except_decoder(self):
        for name, param in self.named_parameters():
            if "Decoder" not in name or "EdgePose" not in name:
                param.requires_grad = False

    def forward(self, g_fc, g, x, x_pose):
        batch_size = len(x)
        x = self.mlp2(x.view((-1, 1000)))
        with g_fc.local_scope():
            g_fc.ndata["x"] = x
            g_fc.apply_edges(self.edge_score)
            e = g_fc.edata["score"]

        A_feat = self.conv_feat_1(g_fc, x, e)
        A_feat = self.conv_feat_2(g_fc, A_feat, e).view((batch_size, 21, -1))

        x_pose_fe = self.mlp3(x_pose.view((-1, 7)))
        A_pose = self.conv_pos_1(g_fc, x_pose_fe, e)
        A_pose = self.conv_pos_2(g_fc, A_pose, e).view((batch_size, 21, -1))

        x = self.Encoder(torch.cat((A_feat, A_pose), dim=2).view((-1, 1024)))
        # x = self.BN(x)

        # e_g = e.view((batch_size, -1, 1))[:, 1:11].reshape((-1, 1))
        A = self.conv1(g_fc, x, e)
        # .view((batch_size, 11, -1))
        # A = self.conv1(g, x)

        # A = F.leaky_relu(A)
        # est_pose = self.Decoder(A[:, 0]).unsqueeze(1)
        pos_out = self.TransDecoder(A).view((batch_size, 21, -1))
        ori_out = self.OriDecoder(A).view((batch_size, 21, -1))
        with g.local_scope():
            g.ndata["x"] = A
            g.apply_edges(self.edge_pose)
            deltaPose = g.edata["pose"]

        # q2r = self.pose_multipy(est_pose[:, 0], deltaPose.view((batch_size, 20, -1)))

        q2r = self.pose_multipy(
            torch.cat((pos_out, ori_out), dim=2)[:, 1:],
            # x_pose[:, 0],
            deltaPose.view((batch_size, 20, -1)),
        )

        # est_pose = A[0, 512:]

        # pos_out = est_pose[:, :, :3]
        # ori_out = est_pose[:, :, 3:]

        # A = A[:, :512]

        A = F.normalize(A, dim=1).view((batch_size, 21, -1))
        # pred2, A2 = self.conv2(g, pred)
        return A, e, pos_out.view((-1, 3)), ori_out.view((-1, 4)), q2r
