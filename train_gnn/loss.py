import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_metric_learning import losses, distances


def vdot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: N x d
    :param v2: N x d
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, 1)
    return out


class QuaternionLoss(nn.Module):
    """
    Implements distance between quaternions as mentioned in
    D. Huynh. Metrics for 3D rotations: Comparison and analysis
    """

    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, q1, q2):
        """
        :param q1: N x 4
        :param q2: N x 4
        :return:
        """
        loss = 1 - torch.pow(vdot(q1, q2), 2)
        loss = torch.mean(loss)
        return loss


class PoseLoss(nn.Module):
    def __init__(self, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta
        self.quaternionLoss = QuaternionLoss()

        if not self.learn_beta:
            self.sx = 0.0
            self.sq = -6.25

        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)
        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        # pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        # loss_q = F.l1_loss(pred_q, target_q)
        loss_q = self.quaternionLoss(pred_q, target_q)

        loss = (
            torch.exp(-self.sx) * loss_x
            + self.sx
            + torch.exp(-self.sq) * loss_q
            + self.sq
        )

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()


class C2F(torch.nn.Module):
    def __init__(self):
        super(C2F, self).__init__()

    def coarse(self, sim_mat, pos_mask, neg_mask, iou):
        # pos_sim =
        iou_ = iou.squeeze(1)
        p_mask = pos_mask[:, 1:].squeeze()
        n_mask = neg_mask[:, 1:].squeeze()
        p_sim = sim_mat[p_mask]
        n_sim = sim_mat[n_mask]
        # p_iou = iou_[p_mask]
        # n_iou = iou_[n_mask]
        pos_sim_exp = torch.exp(p_sim)
        neg_sim_exp = torch.exp(n_sim)
        exp_sum = torch.sum(pos_sim_exp) + torch.sum(neg_sim_exp)
        iou_exp = torch.exp(iou_)
        iou_exp_sum = torch.sum(iou_exp)
        sigmoid_iou = iou_exp / iou_exp_sum
        sigmoid_snj = torch.exp(sim_mat) / exp_sum
        dce_sum = self.dce(sigmoid_iou, sigmoid_snj)

        return dce_sum

    def dce(self, gc, Sn):
        return torch.sum(1 - (gc * torch.log(Sn)))

    def fine(self, sim_mat, database_sim_mat, pos_mask, hard_pos_mask, iou):
        iou_ = iou.squeeze(1)
        p_mask = pos_mask[:, 1:].squeeze()
        hp_mask = hard_pos_mask[:, 1:].squeeze()
        hard_neg_mask = p_mask.logical_xor(hp_mask)
        if torch.sum(hp_mask) == 0:
            return torch.zeros(1).cuda()
        index = torch.tensor(list(range(len(sim_mat)))).cuda()

        hp_ind = index[hp_mask]
        hn_ind = index[hard_neg_mask]
        hp_inds = hp_ind.repeat_interleave(len(hn_ind))
        hn_inds = hn_ind.repeat(len(hp_ind))

        hp_sim = sim_mat[hp_mask]
        hp_sim_exp = torch.exp(hp_sim)
        hp_sim_exp_sum = torch.sum(hp_sim_exp)
        smk = database_sim_mat[hp_inds, hn_inds].view(len(hp_ind), len(hn_ind))
        smk_exp = torch.exp(smk)
        smk_exp_sum = torch.sum(smk_exp, 1)
        exp_sum = smk_exp_sum + hp_sim_exp_sum

        query_Sn = hp_sim_exp / exp_sum
        hn_Sn = smk_exp / exp_sum.repeat_interleave(len(hn_ind)).view(
            (len(hp_ind), len(hn_ind))
        )

        Sn = torch.concat((query_Sn.unsqueeze(1), hn_Sn), 1)
        hn_iou = iou_[hard_neg_mask]
        hn_iou = torch.cat((torch.tensor([1]).cuda(), hn_iou))

        return -torch.sum(Sn * torch.log(hn_iou / torch.sum(hn_iou)))

    def forward(
        self, sim_mat, database_sim_mat, pos_mask, hard_pos_mask, neg_mask, iou
    ):
        res = self.fine(sim_mat, database_sim_mat, pos_mask, hard_pos_mask, iou)
        return res
        # return self.coarse(sim_mat, pos_mask, neg_mask, iou)


def sigmoid(tensor, temp=1.0):
    """temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def sigmoid_t(tensor, temp=0.01):
    tensor_one = tensor * (tensor < 0)
    tensor_two = tensor * (tensor < 0.05)
    tensor_three = tensor * (tensor >= 0.05)
    return (
        sigmoid(tensor_one / 0.001)
        + (sigmoid(tensor_two / 0.001) + 0.5)
        + (10e3 * (tensor_three - 0.05) + sigmoid(tensor_three / 0.001) + 0.5)
    )


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self):
        """
        Parameters
        ----------

        """
        super(SmoothAP, self).__init__()

    def forward(self, sim_all, pos_mask_):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims)"""
        """_summary_

        Example: pred: [1, 0.9, 0.7, 0.6 0.3, 0.2]
        gt: [1, 1, 0, 1, 0, 0] smoothap = 0.833  forward = 0.833
        gt: [1, 1, 1, 1, 0, 0] smoothap = 1      forward = 1
        gt: [1, 1, 0, 1, 0, 1] smoothap = 0.7555 forward = 0.755

        Returns:
            _type_: _description_
        """

        sim_mask = sim_all
        pos_mask = pos_mask_[:, 1:].to("cuda")
        neg_mask = (~pos_mask).to(torch.float).to("cuda")
        # rel = most_pos[:, 1:].to(torch.float).to('cuda')
        rel = pos_mask_[:, 1:].to(torch.float).to("cuda")

        sort_ind = torch.argsort(-sim_mask)
        neg_mask = torch.gather(neg_mask, dim=1, index=sort_ind)
        # neg_mask = neg_mask.gather [sort_ind.unsqueeze(0)]
        # ndcg_neg = self.ndcg(neg_mask, neg_mask)
        rel = torch.gather(rel, dim=1, index=sort_ind)
        # rel[0] = rel[0][sort_ind]
        # ndcg = self.ndcg(rel, rel)
        if torch.sum(pos_mask) == 0:
            return torch.tensor(0.0001, requires_grad=True).cuda()

        # d = sim_mask.squeeze().unsqueeze(0)
        d = sim_mask
        d_repeat = d.repeat_interleave(sim_mask.shape[1], 0).view(
            sim_mask.shape[0], sim_mask.shape[1], sim_mask.shape[1]
        )
        D = d_repeat - d_repeat.transpose(1, 2)
        D = sigmoid(D, 0.001)
        D_ = D * (1 - torch.eye(sim_mask.shape[1])).to("cuda")
        pos_mask_repeat = pos_mask.repeat_interleave(sim_mask.shape[1], 0).view(
            sim_mask.shape[0], sim_mask.shape[1], sim_mask.shape[1]
        )
        D_pos = D_ * pos_mask_repeat

        R = 1 + torch.sum(D_, 2)
        R_pos = (1 + torch.sum(D_pos, 2)) * pos_mask
        R_neg = R - R_pos
        R = R_neg + R_pos

        ap = torch.zeros(sim_mask.shape[0], requires_grad=True).cuda()
        ap_ = (1 / torch.sum(pos_mask, 1)) * torch.sum(R_pos / R, 1)

        ap = ap + ap_

        return ap
        # sim_mask = sim_all[1:]
        """

        sim_mask = sim_all
        pos_mask = pos_mask_[:, 1:].to('cuda')
        neg_mask = (~pos_mask).to(torch.float).to('cuda')
        # rel = most_pos[:, 1:].to(torch.float).to('cuda')
        rel = pos_mask_[:, 1:].to(torch.float).to('cuda')

        sort_ind = torch.argsort(-sim_mask)
        neg_mask[0] = neg_mask[0][sort_ind]
        #ndcg_neg = self.ndcg(neg_mask, neg_mask)
        rel[0] = rel[0][sort_ind]
        #ndcg = self.ndcg(rel, rel)
        if torch.sum(pos_mask) == 0:
            return torch.tensor(0.0001, requires_grad=True).cuda()

        d = sim_mask.squeeze().unsqueeze(0)
        d_repeat = d.repeat(len(sim_mask), 1)
        D = d_repeat - d_repeat.T
        D = sigmoid(D, 0.001)
        D_ = D * (1 - torch.eye(len(sim_mask))).to('cuda')
        D_pos = D_ * pos_mask

        R = 1 + torch.sum(D_, 1)
        R_pos = (1 + torch.sum(D_pos, 1)) * pos_mask
        R_neg = R - R_pos
        R = R_neg + R_pos

        ap = torch.zeros(1, requires_grad=True).cuda()
        ap_ = (1 / torch.sum(pos_mask)) * torch.sum(R_pos / R)

        ap = ap + ap_

        return ap
        """
