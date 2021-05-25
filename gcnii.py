import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as tgnn
from torch_geometric.nn import (
    MessagePassing
)
import torch_geometric as tg

from typing import Optional, Union, List
from icecream import ic
import pretty_errors

class GCNII(MessagePassing):
    def __init__(self, alpha: float, beta: float):
        super(GCNII, self).__init__(aggr='add')
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x, edge_index):
        pass
