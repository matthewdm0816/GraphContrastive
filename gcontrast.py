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
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
from torch_geometric.nn import GATConv, knn_graph
import torch_scatter as tscatter
from icecream import ic
from pprint import pprint

from tensorboardX import SummaryWriter
import time, random, os, sys, gc, copy, colorama, json, re
from tabulate import tabulate
from gat_cls import DGCNNClassifier, GATClassifier
from utils import *

# limit CPU usage
torch.set_num_threads(16)
torch.set_num_interop_threads(16)
print(
    colorama.Fore.GREEN
    + "Using %d/%d cores/threads of CPU"
    % (torch.get_num_threads(), torch.get_num_interop_threads())
)

config_file = "config.yml"
config = YAMLParser(config_file).data

# --------------------- config tests --------------------- #

assert config.optimizer_type in ["Adam", "SGD"]
assert config.dataset_type in ["Cora", "Citeseer", "Pubmed"]
assert config.dataset_type in ["DGCNN", "GAT"]
config.ngpu = len(config.gpu_ids)
config.parallel = config.ngpu > 1
config.batch_size = config.batch_size_single * config.ngpu

config.device = torch.device(
    "cuda:%d" % config.gpu_id if torch.cuda.is_available() else "cpu"
    )
if config.dataset_type in ["Cora", "Citeseer", "Pubmed"]:
    config.dataset = Planetoid(
        root='/home1/dataset/%s' % (config.dataset_type), name=config.dataset_type
    )
else:
    raise NotImplementedError("Only supports Cora/Citeseer/Pubmed dataser for now")

config.fin = config.dataset.num_node_features
config.n_cls = config.dataset.num_classes
config.model = DGCNNClassifier(
    config.fin, config.n_cls, hidden_layers=[256, 512, 256]
)

print(tabulate(config.items()))

# -------------------------------------------------------- #

# ----------------- train/eval functions ----------------- #

def train(config):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    # global dataset_type
    config.model.train()

    # show current lr
    current_lr = config.optimizer.param_groups[0]["lr"]
    ic(current_lr)

    for i, batch in tqdm(enumerate(config.train_loader), total=config.train_loader_length):
        # torch.cuda.empty_cache()
        batch = process_batch(batch, config.parallel, config.dataset_type)

        config.model.zero_grad()
        loss, x = model(batch)
        loss = loss.mean()

        loss.backward()
        config.optimizer.step()
        if i % 10 == 0:
            print(
                colorama.Fore.MAGENTA
                + "[%d/%d]MSE: %.3f, LOSS: %.3f, MSE-ORIG: %.3f, PSNR: %.3f, PSNR-ORIG: %.3f"
                % (
                    epoch,
                    i,
                    mse_loss.detach().item(),
                    loss.detach().item(),
                    orig_mse.detach().item(),
                    psnr_loss.detach().item(),
                    orig_psnr.detach().item(),
                )
            )
    scheduler.step()
    total_mse /= len(loader)
    total_psnr /= len(loader)
    total_orig_psnr /= len(loader)
    if return_lr:
        return total_mse, total_psnr, total_orig_psnr, current_lr
    else:
        return total_mse, total_psnr, total_orig_psnr

# -------------------------------------------------------- #
