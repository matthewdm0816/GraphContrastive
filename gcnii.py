import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import (
    add_self_loops,
    degree,
)

from typing import Optional, Union, List
from icecream import ic
import pretty_errors
from gat_cls import BaseClassifier
from graphlid import LIDCalculater

# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020
# github: https://github.com/lessw2020/mish
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class GCNII(MessagePassing):
    def __init__(self, fin: int, alpha: float, beta: float, activation: str):
        super(GCNII, self).__init__(aggr="add")
        self.alpha = alpha
        self.beta = beta
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "mish":
            self.activation = Mish()

        self.fin = fin
        self.fc = nn.Linear(fin, fin)

    def forward(self, x, edge_index):
        # A~ = A + I
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # L~ = D~^-1/2 A~ D~^-1/2
        # Same with GCN
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # X = ((1-a) LH + aH)
        x = (
            self.propagate(edge_index, x=x, norm=norm) * (1 - self.alpha)
            + x * self.alpha
        )

        # ((1-b)I + b W)X
        x = self.fc(x) * self.beta + (1 - self.beta) * x

        # \simga(x)
        return self.activation(x)

    def message(self, x_i, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCNIIBackbone(nn.Module):
    def __init__(self, __C, fin):
        super().__init__()
        self.fc = nn.Linear(fin, __C.BACKBONE.HIDDEN_SIZE)
        self.layers = nn.ModuleList(
            [
                GCNII(
                    __C.BACKBONE.HIDDEN_SIZE,
                    alpha=__C.BACKBONE.ALPHA,
                    beta=__C.BACKBONE.LAMBDA / (idx + 1),
                    activation=__C.BACKBONE.ACTIVATION,
                )
                for idx in range(__C.BACKBONE.LAYERS)
            ]
        )
        ic(self.fc)
    
    def forward(self, x, edge_index):
        ic(x.shape)
        x = self.fc(x)
        ic(x.shape)
        for gcn in self.layers:
            x = gcn(x, edge_index=edge_index) 
        return x


class GCNIIClassifier(BaseClassifier):
    def __init__(self, fin, n_cls, __C):
        self.__C = __C
        super().__init__(
            fin=fin,  
            n_cls=n_cls, 
            hidden_layers=[1], # to make superclass happy
            dropout=__C.DROPOUT,
            criterion=nn.NLLLoss(),
            make_cls=__C.MAKE_CLS,
            knn=__C.GLID.KNN,
            khop=__C.GLID.KHOP,
        )
        # Clean unused components
        del self.filters
        del self.activations
        del self.cls

        self.fin = fin
        self.n_cls = n_cls
        self.lid_calc = LIDCalculater(knn=__C.GLID.KNN, khop=__C.GLID.KHOP, load_dir=__C.GLID.CACHE_PATH)
        self.backbone = self.get_backbone()
        self.cls = self.get_classifier(__C.BACKBONE.HIDDEN_SIZE, n_cls)
        

    # skip
    def get_layer(self, i, o):
        return nn.Identity()

    def process_fin(self, fin):
        return [fin]

    # skip
    def get_activation(self, o):
        return nn.Identity()

    def get_backbone(self):
        return GCNIIBackbone(self.__C, self.fin)

    # def get_classifier(self):
    #     return nn.Sequential(
    #         MLP(i, o, batchnorm=False, activation=nn.Identity), nn.Softmax(dim=-1)
    #     )

    def embed(self, x, edge_index, dummy=False):
        x = self.backbone(x, edge_index)
        # Calc LID
        if self.__C.GLID.ENABLE:
            with torch.no_grad():
                self.lid = self.lid_calc.calcLID(x, edge_index)
        else:
            self.lid = torch.tensor([-1])
        return x
    
    def predict(self, x, edge_index, dummy=False):
        # return softmaxed probs
        x_embed = self.embed(x, edge_index)
        return self.cls(x_embed) + 1e-10 # in case log0 => nan/inf

