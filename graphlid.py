import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import (
    GCNConv,
    SGConv,
    MessagePassing,
    knn_graph,
    DataParallel,
    GMMConv,
    knn,
)
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops, k_hop_subgraph
from torch_geometric.nn import GATConv, knn_graph
import torch_scatter as tscatter
from icecream import ic
from pprint import pprint

from tensorboardX import SummaryWriter
import time, random, os, sys, gc, copy, colorama, json, re, pretty_errors
from tabulate import tabulate
from gat_cls import *
from utils import *
from copy import copy

from icecream import ic
from tqdm import tqdm

from typing import Optional, List

class GraphAggr(MessagePassing):
    """
    Bilateral Filter Impl. under PyG
    In: (B * N) * FIN
    OUT: (B * N) * 1, the Node LID
    Since it aggreagets k nodes in the 1-hop, 
    it can only be used on fixed hop size graph
    """

    def __init__(self, k: int):
        super().__init__(aggr="add")
        self.k = k # hop size
        

    def message(self, x_i, x_j):
        return x_j
        
    def aggregate(neighb, x):
        """
        x ~ N * K * FIN
        """
        n_pts = x.shape[0]
        return neighb.view(n_pts, -1) # Flatten into N * (K * FIN)
            

    def forward(self, x, edge_index):
        """
        x ~ N * FIN
        edge_index ~ 2 * E
        """
        
        return self.propagate(edge_index, x=x)
    
class GraphRegularizer(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def message(self, x_i, x_j):
        r"""
        x_i, x_j ~ [E, FIN]
        """
        # batch outer prod.
        xdim = x_i.shape[-1]  # i.e. FIN
        res = torch.einsum("bi,bj->bij", x_i, x_j)
        # print(res.shape)
        return res
        
    def forward(self, x, edge_index=None):
        r"""
        Calculate graph LID term
        $R=X^T A X$
        """
        num_nodes = x.shape[-2]
        xdim = x.shape[-1]
        
        return  self.propagate(edge_index=edge_index, x=x)
        # => [N * N] dense
        # print(res.shape)  # [B, F * F]
        # Frobenius Norm (intrinstically same)
        # return (torch.norm(res, dim=-1, p="fro") ** 2).mean()

    
class LIDCalculater:
    def __init__(self, knn, khop, load_dir):
        # Cached K-hops
        self.khops = None
        self.knn = knn
        self.khop = khop
        self.load_dir = load_dir
        if os.path.exists(self.load_dir):
            self._load_subgraphs()
        
    def _load_subgraphs(self):
        self.khops = np.load(self.load_dir)
        self.khops = torch.from_numpy(self.khops)
    
    def _save_subgraphs(self):
        # if self.save_dir is not None:
        np.save(self.load_dir, self.khops.detach().cpu().numpy())
        
    def calcLID(self, x, edge_index):
        
        n_nodes = x.shape[0]
        ic(n_nodes)
        
        if self.khops is None:
            khops = []
            for idx, xi in tqdm(enumerate(x), total=x.shape[0]):
                khop_nidxs, *_ = k_hop_subgraph(idx, num_hops=self.khop, edge_index=edge_index) # [N, ]
                n_khop = khop_nidxs.shape[0]
                xs = (torch.ones(n_khop) * idx).view(1, -1).long().to(x) # [N, ]
                khop_edges = torch.cat([xs, khop_nidxs.view(1, -1)], dim=0) # [2, N]
                khops.append(khop_edges)
            khops = torch.cat(khops, dim=-1) # [2, N * N_NODES]
            khops, _ = tg.utils.remove_self_loops(khops)
            self.khops = khops.long()
            self._save_subgraphs()
        
        row, col = self.khops # [2, N * N_NODES]
        dist = (x[row] - x[col]).norm(dim=-1)
        dense_A = tg.utils.to_dense_adj(self.khops, edge_attr=dist).squeeze(0) # N * N
        
        dense_A = (dense_A == 0.).float() * 1e10 + dense_A
        
        topK = torch.topk(dense_A, k=self.knn, largest=False, dim=-1).values # N * K
        v_log = torch.log(topK.transpose(0, 1) / topK.transpose(0, 1)[-1] + 1e-8)
        v_log = v_log.transpose(0, 1).sum(dim=-1) # => [N, K] => [N]
        lid = - self.knn / v_log
        return lid.mean()
    
# Sadly, it's even slower
def calcLIDWithoutGrad(x, edge_index, khop: int, knn: int):
    n_nodes = x.shape[0]
    ic(n_nodes)
    # idxs = torch.arange(0, n_nodes).long()
    total_lid = torch.tensor(0.).to(x)

    for idx, xi in tqdm(enumerate(x)):
        khop_nidxs, _, mapping, _ = k_hop_subgraph(idx, num_hops=khop, edge_index=edge_index)
        khop_feat = x[khop_nidxs]
        
        # Not have to be differentiable! use tg.knn() method
        knn_idx = tg.nn.knn(khop_feat, xi.view(1, -1), k=knn + 1)[1][1:]
        # ic(knn_idx)
        knn_feat = khop_feat[knn_idx] # => [N]
        knn_dist = (knn_feat - xi).norm(dim=-1)
        # ic(knn_dist)
        lid = (knn_dist / torch.max(knn_dist)).log().sum() / knn
        lid = - lid ** -1.
        # ic(lid)
        total_lid += lid
    return total_lid / n_nodes
    
if __name__ == '__main__':
    x = torch.randn([100, 10])
    edge_index = tg.nn.knn_graph(x, loop=False, k=5)
    calc = LIDCalculater(khop=3, knn=10)
    total_lid = calc.calcLID(x, edge_index)
    ic(total_lid)
    total_lid = calcLIDWithoutGrad(x, edge_index, khop=3, knn=10)
    ic(total_lid)
    
    
    
    
    
    
    
    








"""
    aggregator = GraphAggr(knn)
    x0 = x.clone()
    for range(self.khop):
        # Aggregate for K times on 1-hop
        x = self.aggregator(x, edge_index) # => (B * N) * (knn * FIN)
    
    n_nodes, last_dim = x.shape
    fin = last_dim / self.knn
    
    x = x.view(n_nodes, self.knn, fin)
    for x_i_0, x_i in zip(x0, x):
        # x_i ~ knn^k * FIN
        knn_idx = 
        """
        
        
        
        
        
        
        
        
        
    
    