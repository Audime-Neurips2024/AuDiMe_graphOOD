import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce, SparseTensor
import torch.optim as optim
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation

import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader

from modelNew.base_model import BaseModel

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor,learnable_aug=False, **kwargs):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.learnable_aug = learnable_aug
        self.learnable_edge_weight1 = None
        self.learnable_edge_weight2 = None


    def forward(self, x, edge_index, batch, edge_attr=None,label_emb = None):
        aug1, aug2 = self.augmentor
        if not self.learnable_aug:
            x1, edge_index1, edge_weight1 = aug1(x, edge_index)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index)
            z, g = self.encoder(x, edge_index, batch=batch,return_both_rep=True,edge_attr = None)
            z1, g1 = self.encoder(x1, edge_index1, batch=batch,return_both_rep=True,edge_attr = None)
            z2, g2 = self.encoder(x2, edge_index2, batch=batch,return_both_rep=True,edge_attr = None)
        else:
            assert self.learnable_edge_weight1 is not None
            z, g = self.encoder(x, edge_index, batch=batch,return_both_rep=True)
            z1, g1 = self.encoder(x, edge_index, batch=batch,edge_attr=edge_attr,edge_weight=self.learnable_edge_weight1,return_both_rep=True)
            z2, g2 = self.encoder(x, edge_index, batch=batch,edge_attr=edge_attr,edge_weight=self.learnable_edge_weight2,return_both_rep=True)
            if label_emb is not None:
                g1 = g1+ label_emb
                g2 = g2+ label_emb
        return z, g, z1, z2, g1, g2
    
    
    
    