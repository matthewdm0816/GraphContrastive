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
    SGConv,
)
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
from torch_geometric.nn import GATConv, knn_graph
import torch_scatter as tscatter
from icecream import ic

from utils import layers, MultiSequential, entropy
from vat.vat import VATLoss, VATGraphLoss
from activeloss.loss import NCEandRCEWithProbs, NCEandRCEWithLogits
from graphlid import LIDCalculater


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
        make_cls: bool = True,
        knn: int = 10,
        khop: int = 3,
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
        self.lid = None
        self.knn = knn
        self.khop = khop
        self.lid_calc = LIDCalculater(knn=self.knn, khop=self.khop, load_dir='subgraphs.npy')

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
            MLP(i, o, batchnorm=False, activation=nn.Identity), nn.Softmax(dim=-1)
        )
    
    def embed(self, x, edge_index, est_lid: bool=False):
        # return embeddings of x
        for i, (layer, activation) in enumerate(zip(self.filters, self.activations)):
            # use static edges
            if torch.any(torch.isnan(x)):
                ic(i, self.filters[i - 1], x.max(), x.median(), x.min())
                ic([param.data for param in self.filters[i - 1].parameters()])
                assert False, "NaN detected!"
            x = layer(x, edge_index=edge_index)
            x = activation(x)
            
        # Calc LID
        if est_lid:
            with torch.no_grad():
                self.lid = self.lid_calc.calcLID(x, edge_index)
        else:
            self.lid = torch.tensor([-1])
        return x
        
    def predict(self, x, edge_index, est_lid=False):
        # return softmaxed probs
        x_embed = self.embed(x, edge_index, est_lid=est_lid)
        pred_probs = self.cls(x_embed) + 1e-10 # in case log0 => nan/inf
        return pred_probs

    def forward(self, data, config):
        target, edge_index, clean_target, batch, x, mask, ln_mask, train_mask = (
            data.y.long(),
            data.edge_index.long(),
            data.y0.long(),
            data.batch,
            data.x,
            data.mask.bool(),
            data.ln_mask.bool(), # label noise mask
            data.train_mask.bool(),
        )
        is_noisy: bool = (ln_mask.sum().item() > 0)
        n_cls = clean_target.max().long() + 1
        if not self.make_cls:
            x = self.embed(x, edge_index, est_lid=config.est_lid)
            return x
        else:
            # predict on whole dataset
            vat_loss = VATGraphLoss(xi=config.vat_xi, eps=config.vat_eps, ip=config.vat_ip)
            lds = vat_loss(self, x, edge_index) # VAT loss calc.
            
            # calc labeled loss on train/test/val set
            x = self.predict(x, edge_index, est_lid=config.est_lid) # [N, C]
            if config.useNCEandRCE:
                self.criterion = NCEandRCEWithLogits(alpha=config.alpha_NCE, beta=config.beta_RCE, num_classes=n_cls)
            loss = self.criterion(x.log()[mask], target[mask])
            loss += config.lds_alpha * lds
            
            confidence, sel = x.max(dim=-1)  # [N, ]
            # record correct cls
            correct = sel[mask].eq(target[mask]).float().sum()  # [1, ]
            original_correct = sel[mask].eq(clean_target[mask]).float().sum()

            # record confidence/entropy
            if torch.all(mask.eq(train_mask)):
                # on train set, use clean samples
                real_confidence = confidence[train_mask][~ln_mask]
                real_ent = entropy(x[train_mask][~ln_mask], dim=-1)
            else:
                # on test/val set, use all samples
                real_confidence = confidence[mask]
                real_ent = entropy(x[mask], dim=-1)
            # selects noisy metrics
            if is_noisy:
                noise_confidence = confidence[train_mask][ln_mask]
                noise_ent = entropy(x[train_mask][ln_mask], dim=-1)
            else:
                noise_confidence = real_confidence
                noise_ent = real_ent

            return (
                loss,
                x,
                real_confidence,
                noise_confidence,
                correct,
                original_correct,
                real_ent,
                noise_ent,
            )


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

    def get_activation(self, o):
        return nn.Identity()

    def get_layer(self, i, o):
        return SGConv(i, o, K=3)
