import torch
import torch.nn as nn
from loss import SmoothAP


class ReLoss(nn.Module):

    def __init__(self, hidden_dim=128):
        super(ReLoss, self).__init__()
        self.ap = SmoothAP().cuda()
        self.cos = nn.CosineSimilarity(dim=1).cuda()
        self.criterion = nn.MSELoss().cuda()
        self.labels_length = 51
        self.logits_fcs = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.loss = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings, e, gt_iou, mask):
        query_embeddings = torch.repeat_interleave(
            embeddings[0].unsqueeze(0), self.labels_length - 1, 0)
        database_embeddings = embeddings[1:self.labels_length]
        sim_mat = self.cos(query_embeddings, database_embeddings)
        aploss = self.ap(sim_mat, mask)
        mseloss = self.criterion(e[: self.labels_length -1], gt_iou)
        # loss_ = self.ap(sim_mat, mask) + \
        #     self.criterion(e[: self.labels_length - 1], gt_iou)
        loss_ = aploss + mseloss
        # probs = torch.softmax(logits, dim=1)
        # pos_probs = torch.gather(probs, 1, targets.unsqueeze(-1))
        hidden = self.logits_fcs(loss_.view(-1,))
        loss = self.loss(hidden).abs().mean()
        return loss
