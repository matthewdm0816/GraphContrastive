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
)
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
from torch_geometric.nn import GATConv, knn_graph
import torch_scatter as tscatter
from icecream import ic

from utils import layers


class BaseClassifier(nn.Module):
    r"""
    Generic Base Classifier Architecture
    x=>FILTER=>ACTIVATION(if not output layer)=>...=>y
    """

    def __init__(self, fin, n_cls, hidden_layers, criterion=nn.NLLLoss()):
        super().__init__()
        self.fin, self.hidden_layers = fin, self.process_fin(fin) + hidden_layers
        self.filters = nn.ModuleList(
            [self.get_layer(i, o) for i, o in layers(self.hidden_layers)]
        )
        self.cls = self.get_classifier(hidden_layers[-1], n_cls)
        self.criterion = criterion

    def get_layer(self, i, o):
        raise NotImplementedError

    def process_fin(self, fin):
        return NotImplementedError

    def get_classifier(self, i, o):
        return NotImplementedError

    # def calc_filter(x, filter, edge_index):
    #     return filter(x, edge_index=edge_index)

    def forward(self, data):
        # print(data)
        target, batch, x = data.target, data.batch, data.x
        for i, filter in enumerate(self.filters):
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            x = filter(x, edge_index=edge_index)

        x = self.cls(x)  # assume we have normalized/softmaxed prob here.
        loss = self.criterion(x.log(), target)
        return loss, x

class GATDenoiser(nn.Module):
    r"""
    Baseline GAT as a classifier
    examplar hidden_layers:
    [
        {"f":16, "heads":4}, # i.e. 64
        {"f":64, "heads":2}, # i.e. 128
        {"f":6, "heads":8, "concat":False, "negative_slope":0.5}
    ]
    """

    def __init__(self, fin, n_cls, hidden_layers: list, criterion=nn.NLLLoss()):
        super().__init__()
        hidden_layers = [{"f": fin, "heads": 1}] + hidden_layers
        self.gats = nn.ModuleList(
            [
                GATConv(
                    in_channels=i["f"] * i["heads"],
                    out_channels=o["f"],
                    heads=o["heads"],
                    concat=o["concat"] if "concat" in o.keys() else True,
                    negative_slope=o["negative_slope"]
                    if "negative_slope" in o.keys()
                    else 0.2, # by default 0.2
                )
                for i, o in layers(hidden_layers)
            ]
        )
        self.activation = nn.ModuleList(
            [
                nn.Sequential(
                    nn.PReLU(),
                    nn.BatchNorm1d(o["f"] * o["heads"]),
                )
                if idx != len(hidden_layers) - 2
                else nn.Identity()
                for idx, (i, o) in enumerate(layers(hidden_layers))
            ]
        )

        self.cls = self.get_classifier(hidden_layers[-1], n_cls)
        self.criterion = criterion

    def get_classifier(self, i, o):
        return nn.Sequential(
            nn.Linear(i, o),
        )

    def forward(self, data):
        # print(data)
        target, batch, x = data.target, data.batch, data.x
        for i, (layer, activation) in enumerate(zip(self.gats, self.activation)):
            # use dynamic graph
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            x = layer(x, edge_index=edge_index)
            x = activation(x)

        x = self.cls(x)  # assume we have normalized/softmaxed prob here.
        loss = self.criterion(x.log(), target)
        return loss, x
