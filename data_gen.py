import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from sklearn.metrics import matthews_corrcoef
from torch_geometric.data import DataLoader, Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm


from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from datasets.mutag_dataset import Mutag
from models.gnn_ib import GIB
from models.ciga import GNNERM, GNNEnv, CIGA, GSAT, GNNPooling, GALA
from models.losses import GeneralizedCELoss, get_contrast_loss, get_irm_loss
from utils.logger import Logger
from utils.util import args_print, set_seed, split_into_groups
from kmeans_pytorch import kmeans, kmeans_predict

from utils.helper import visualize_a_graph, get_viz_idx, plot_and_save_subgraph
from torch_geometric.transforms import BaseTransform

class FeatureSelect(BaseTransform):
    def __init__(self, nfeats):
        """
        Initialize the transformation with the number of features to retain.

        Parameters:
        - feats (int): The number of features to retain from the beginning of the feature matrix.
        """
        self.nfeats = nfeats

    def __call__(self, data):
        """
        Retain only the first 'feats' features of the node feature matrix 'data.x'.
        Parameters:
        - data (torch_geometric.data.Data): The graph data object.
        Returns:
        - torch_geometric.data.Data: The modified graph data object with the node feature matrix sliced.
        """
        
        # Check if 'data.x' exists and has enough features
        data.x = data.x[:, :self.nfeats]
        return data
    


class AddRandomFeatures(BaseTransform):
    def __init__(self, num_features=6):
        """
        Initialize the transformation with the desired number of features.
        Default is set to 6 random features.
        
        Parameters:
        - num_features (int): The number of random features to add to each node.
        """
        self.num_features = num_features

    def __call__(self, data):
        """
        Concatenate random features to each node in the graph. If node features
        do not exist, new random features are assigned as node features.

        Parameters:
        - data (torch_geometric.data.Data): The graph data object.

        Returns:
        - torch_geometric.data.Data: The modified graph data object with added features.
        """
        
        num_nodes = data.num_nodes
        # Generate random features for each node
        random_features = torch.randn((num_nodes, self.num_features))
        # Check if node features already exist
        if data.x is not None:
            # Concatenate the random features to the existing features
            data.x = torch.cat([data.x, random_features], dim=-1)
        else:
            # Assign the random features as the node features
            data.x = random_features
        return data

if __name__ == "__main__":
    
    train_dataset = SPMotif(root=f'data/SPMotif-0.5001/',mode='train',pre_transform=AddRandomFeatures(6))

    val_dataset = SPMotif(root=f'data/SPMotif-0.5001/',mode='val',pre_transform=AddRandomFeatures(6))

    test_dataset = SPMotif(root=f'data/SPMotif-0.5001/',mode='test',pre_transform=AddRandomFeatures(6))
    
    train_dataset = SPMotif(root=f'data/SPMotif-0.7001/',mode='train',pre_transform=AddRandomFeatures(6))

    val_dataset = SPMotif(root=f'data/SPMotif-0.7001/',mode='val',pre_transform=AddRandomFeatures(6))

    test_dataset = SPMotif(root=f'data/SPMotif-0.7001/',mode='test',pre_transform=AddRandomFeatures(6))
    
    print (train_dataset)
    print (train_dataset[0])



    