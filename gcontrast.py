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

import torch
import torch.optim as optim
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
config = YAMLParser('config_file').data

# ================ Config Tests
assert config.optimizer_type in ["Adam", "SGD"]
assert config.dataset_type in ["Cora", "Citeseer", "Pubmed"]
config.ngpu = len(config.gpu_ids)
config.parallel = config.ngpu > 1
config.batch_size = config.batch_size_single * config.ngpu
tabulate(config.items)

device = torch.device("cuda:%d" % config.gpu_id if torch.cuda.is_available() else "cpu")

# ===============

