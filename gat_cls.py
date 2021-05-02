import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import (
    GCNConv,
    SGConv,
    MessagePassing,
    knn_graph,
    DataParallel,
    GMMConv,
    DynamicEdgeConv,
    EdgeConv,
    SGConv
)
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
from torch_geometric.nn import GATConv, knn_graph
import torch_scatter as tscatter
from icecream import ic

from utils import layers, MultiSequential, entropy


class MLP(nn.Module):
    """
    Plain MLP with activation
    """

    def __init__(self, fin, fout, activation=nn.PReLU, dropout=None, batchnorm=True):
        super().__init__()
        if dropout is not None and batchnorm:
            assert isinstance(dropout, float)
            self.net = nn.Sequential(
                nn.Linear(fin, fout),
                nn.BatchNorm1d(fout),
                nn.Dropout(p=dropout),
                activation(),
            )
        elif batchnorm:
            self.net = nn.Sequential(
                nn.Linear(fin, fout), nn.BatchNorm1d(fout), activation()
            )
        else:
            self.net = nn.Sequential(nn.Linear(fin, fout), activation())

    def forward(self, x):
        return self.net(x)


class BaseClassifier(nn.Module):
    r"""
    Generic Base Classifier Architecture
    x=>FILTER=>ACTIVATION(if not output layer)=>...=>y
    """

    def __init__(
        self,
        fin,
        n_cls,
        hidden_layers: list,
        dropout: float = 0.3,
        criterion=nn.NLLLoss(),
        make_cls: bool=True
    ):
        super().__init__()
        self.fin, self.hidden_layers = fin, self.process_fin(fin) + hidden_layers
        self.dropout = dropout
        self.filters = nn.ModuleList(
            [self.get_layer(i, o) for i, o in layers(self.hidden_layers)]
        )
        self.activations = nn.ModuleList(
            [self.get_activation(o) for i, o in layers(self.hidden_layers)]
        )
        self.cls = self.get_classifier(hidden_layers[-1], n_cls)
        self.criterion = criterion
        self.make_cls = make_cls

    def get_layer(self, i, o):
        raise NotImplementedError

    def process_fin(self, fin):
        return NotImplementedError

    def get_activation(self, o):
        return nn.Sequential(
            nn.BatchNorm1d(o),
            nn.Dropout(p=self.dropout) if self.dropout > 1e-5 else nn.Identity(),
            nn.PReLU(),
        )

    def get_classifier(self, i, o):
        return nn.Sequential(
            MLP(i, o, batchnorm=False, activation=nn.Identity), 
            nn.Softmax(dim=-1)
        )

    # def calc_filter(x, filter, edge_index):
    #     return filter(x, edge_index=edge_index)

    def forward(self, data):
        # print(data)
        target, edge_index, clean_target, batch, x, mask = (
            data.y.long(),
            data.edge_index.long(),
            data.y0.long(),
            data.batch,
            data.x,
            data.mask.bool(),
        )
        # with torch.autograd.detect_anomaly():
        for i, (layer, activation) in enumerate(zip(self.filters, self.activations)):
            # use static edges
            # edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            if torch.any(torch.isnan(x)):
                ic(i, self.filters[i - 1], x.max(), x.median(), x.min())
                ic([param.data for param in self.filters[i - 1].parameters()])
                assert False, "NaN detected!"
            x = layer(x, edge_index=edge_index)
            x = activation(x)
        if self.make_cls:
            # assume we have normalized/softmaxed prob here.
            x = self.cls(x)  # [N, C]
            # ic(target)
            loss = self.criterion((x + 1e-8).log()[mask], target[mask])
            confidence, sel = x.max(dim=-1)  # [N, ]
            confidence = confidence
            # ic(sel[mask], target[mask])
            correct = sel[mask].eq(target[mask]).float().sum()  # [1, ]
            original_correct = sel[mask].eq(clean_target[mask]).float().sum()
            ent = entropy(x, dim=-1)
            return loss, x, confidence, correct, original_correct, ent
        else:
            return x


class GATClassifier(nn.Module):
    r"""
    Baseline GAT as a classifier
    examplar hidden_layers:
    [
        {"f":16, "heads":4}, # i.e. 64
        {"f":64, "heads":2}, # i.e. 128
        {"f":6, "heads":8, "concat":False, "negative_slope":0.5}
    ]
    """

    def __init__(
        self,
        fin,
        n_cls,
        hidden_layers: list,
        dropout: float = 0.3,
        criterion=nn.NLLLoss(),
    ):
        super().__init__()
        hidden_layers = [{"f": fin, "heads": 1}] + hidden_layers
        self.dropout = dropout
        self.gats = nn.ModuleList(
            [
                GATConv(
                    in_channels=i["f"] * i["heads"],
                    out_channels=o["f"],
                    heads=o["heads"],
                    concat=o["concat"] if "concat" in o.keys() else True,
                    negative_slope=o["negative_slope"]
                    if "negative_slope" in o.keys()
                    else 0.2,  # by default 0.2
                )
                for i, o in layers(hidden_layers)
            ]
        )
        self.activation = nn.ModuleList(
            [
                nn.Sequential(
                    nn.PReLU(),
                    nn.BatchNorm1d(o["f"] * o["heads"]),
                    nn.Dropout(p=self.dropout) if self.dropout > 1e-5 else nn.Identity,
                )
                for idx, (i, o) in enumerate(layers(hidden_layers))
            ]
        )

        self.cls = self.get_classifier(hidden_layers[-1], n_cls)
        self.criterion = criterion

    def get_classifier(self, i, o):
        return nn.Sequential(nn.Linear(i, o),)

    def forward(self, data):
        # print(data)
        target, batch, x = data.y, data.batch, data.x
        for i, (layer, activation) in enumerate(zip(self.gats, self.activation)):
            # use dynamic graph
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            x = layer(x, edge_index=edge_index)
            x = activation(x)

        x = self.cls(x)  # assume we have normalized/softmaxed prob here.
        loss = self.criterion((x + 1e-8).log(), target)
        return loss, x


class DGCNNClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_fin(self, fin):
        return [fin]

    def get_layer(self, i, o):
        mlp = MLP(2 * i, o)
        econv = EdgeConv(nn=mlp, aggr="mean")
        # ic(econv, i, o)
        return econv


class GCNClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_fin(self, fin):
        return [fin]

    def get_layer(self, i, o):
        return GCNConv(i, o)

class SGCClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_fin(self, fin):
        return [fin]
    
    def get_layer(self, i, o):
        return SGConv(i, o, K=3)
