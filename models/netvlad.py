import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.neighbors import NearestNeighbors
import numpy as np


class MinkNetVladWrapper(torch.nn.Module):
    # Wrapper around NetVlad class to process sparse tensors from Minkowski networks
    def __init__(self, feature_size, output_dim, cluster_size=64, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, cluster_size=cluster_size, output_dim=output_dim,
                                     gating=gating, add_batch_norm=True)

    def forward(self, x):
        # x is SparseTensor
        assert x.F.shape[1] == self.feature_size
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros

        x = self.net_vlad(features)
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.output_dim
        return x    # Return (batch_size, output_dim) tensor


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)

        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(torch.randn(
            cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        # vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        # vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad
    

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
    

class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


if __name__ == '__main__':
    net_vlad = NetVLADLoupe(feature_size=1024, max_samples=360, cluster_size=16,
                                 output_dim=20, gating=True, add_batch_norm=True,
                                 is_training=True)
    # input  (bs, 1024, 360, 1)
    torch.manual_seed(1234)
    input_tensor = F.normalize(torch.randn((1,1024,360,1)), dim=1)
    input_tensor2 = torch.zeros_like(input_tensor)
    input_tensor2[:, :, 2:, :] = input_tensor[:, :, 0:-2, :].clone()
    input_tensor2[:, :, :2, :]  = input_tensor[:, :, -2:, :].clone()
    input_tensor2= F.normalize(input_tensor2, dim=1)
    input_tensor_com = torch.cat((input_tensor, input_tensor2), dim=0)

    print(input_tensor.shape)
    print(input_tensor2.shape)
    print("==================================")

    with torch.no_grad():
        net_vlad.eval()
        # output_tensor = net_vlad(input_tensor_com)
        # print(output_tensor)
        out1 = net_vlad(input_tensor)
        print(out1.shape)
        net_vlad.eval()
        # input_tensor2[:, :, 20:, :] = 0.1
        input_tensor2 = F.normalize(input_tensor2, dim=1)
        print(input_tensor2.shape)
        out2 = net_vlad(input_tensor2)
        print(out2.shape)
        net_vlad.eval()
        input_tensor3 = torch.randn((1,1024,360,1))
        print(input_tensor3.shape)
        out3 = net_vlad(input_tensor3)
        print(out3.shape)


        print(((out1-out2)**2).sum(1))
        print(((out1-out3)**2).sum(1))


