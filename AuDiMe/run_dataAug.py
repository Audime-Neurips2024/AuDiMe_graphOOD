import json
import math
import os
import os.path as osp
import sys
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
import shutil
import time
import json
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T
# # pytorch lightning
from torch_geometric.datasets import *
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
import argparse
import numpy as np
import higher
from termcolor import colored


from copy import deepcopy
from torch_geometric.loader import DataLoader
from termcolor import colored
from datetime import datetime
import networkx as nx
import scipy as sp
import pymetis
from tqdm import tqdm
import time
from torch_geometric.datasets import GNNBenchmarkDataset
import argparse
import warnings
from glob import glob


from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

from modelNew.model_learnableAugment_v2 import Model
from modelNew.utils import save_numpy_array_to_file,FeatureSelect

# from model.PL_Model import PL_Model

# from utils.functionalUtils import get_free_gpu,save_numpy_array_to_file,write_results_to_file,generate_string,pack_arrays_to_3D,last_K_unique_elements,get_svm_pred_proba,reweight_sample,dict_label_removed
# from utils.datasetInfo import DatasetInfo
# from utils.dataUtils import k_fold,generate_balanced_subsets_and_complements,load_graph_embeddings,CustomDataset,CustomKD_Dataset,FilteredDataset,generate_tensor_lists
# from utils.trainingUtils import train_and_evaluate_svm_with_proba,train_model_drugood,update_pseudo_samples,train_and_evaluate_svm_with_proba_synthetic,train_and_evaluate_mlp_synthetic
# from utils.NodePolicy import ALPolicy
from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from datasets.mutag_dataset import Mutag



def write_results_to_file(fpath, n, s):
    # Check if the directory exists, if not, create it
    # fpath: 存放路径
    # n: 文件名 （.txt结尾）
    # s: 内容
    if not os.path.exists(fpath):
        try:
            os.makedirs(fpath)
        except:
            pass

    # Construct full file path
    full_path = os.path.join(fpath, n)

    # Open the file in write mode, which will create the file if it does not exist
    # and overwrite it if it does. Then write the string to the file.
    with open(full_path, 'w') as f:
        f.write(s)


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments


parser.add_argument('--dataset', default='drugood_lbap_core_ec50_scaffold', type=str)
parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')

# Add arguments for model configuration. nfeats,nhid and nclass should be set manually
parser.add_argument('--nfeat', type=int, default=39, help='Number of features.') 
parser.add_argument('--nhid', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=2, help='Number of classes.')
parser.add_argument('--edge_dim', type=int, default=-1, help='dim of edge attr')
parser.add_argument('--nlayers', type=int, default=3, help='Number of GNN layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
parser.add_argument('--save_mem', action='store_true', default=True, help='Enable memory-saving mode.')
parser.add_argument('--jk', type=str, default='concat', choices=['last', 'concat'], help='Jumping knowledge mode.')
parser.add_argument('--node_cls', action='store_true', default=False, help='node classification or graph cls')
parser.add_argument('--pooling', type=str, default='sum', choices=['sum', 'attention'], help='Pooling strategy.')
parser.add_argument('--with_bn', action='store_true', default=False, help='Enable batch normalization.')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay rate.')
parser.add_argument('--with_bias', action='store_true', default=True, help='Include bias in layers.')
parser.add_argument('--base_gnn', type=str, default='gin', choices=['gin', 'gcn'], help='Base GNN model type.')
parser.add_argument('--edge_gnn', type=str, default='gin', choices=['gin', 'gcn'], help='data aug GNN model type.')
parser.add_argument('--edge_gnn_layers', type=int, default=2, help='No. layers')
parser.add_argument('--edge_budget', type=float, default=0.55, help='edge budget for edge removal')
parser.add_argument('--num_samples', type=int, default=1, help='No. of samples of graphs')
parser.add_argument('--edge_penalty', type=float, default=0., help='penalty for regularization of data aug')
parser.add_argument('--edge_uniform_penalty', type=float, default=0., help='penalty for edge sampling uniformity penalty')
parser.add_argument('--edge_prob_thres', type=int, default=50, help='edge prob thres of k in int(50%)')

parser.add_argument('--penalty', type=float, default=1e-1, help='SSL Penalty weight.')
parser.add_argument('--gradMatching_penalty', type=float, default=1.0, help='meta cls Penalty weight.') #! actually meta cls penalty, not gradient matching penalty
parser.add_argument('--featureMasking', action='store_true', default=False, help='mask input or not')
parser.add_argument('--useAutoAug', action='store_true', default=False, help='use learnable edge dropping')

# parser.add_argument('--useAutoAug', action='store_true', default=False, help='use learnable edge dropping')
# Add arguments for training configuration

parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
parser.add_argument('--adapt_lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_scheduler', action='store_false', default=True, help='Enable learning rate scheduler.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--early_stop_epochs', type=int, default=10)
parser.add_argument('--pretraining_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=50, help='Patience for learning rate scheduler.')
parser.add_argument('--lr_decay', type=float, default=0.75, help='Learning rate decay factor.')
parser.add_argument('--project_layer_num', type=int, default=2, help='Number of projection layers.')
parser.add_argument('--temp', type=float, default=0.2, help='Temperature parameter for contrastive loss.')
parser.add_argument('--valid_metric', type=str, default='auc', help='Validation metric.')

parser.add_argument('--erm', action='store_true', default=False, help='use erm')
parser.add_argument('--fname_str', type=str, default='', help='additional name for folder name')

parser.add_argument('--addRandomFeature', action='store_true', default=False, help='add random features for SPMotif datasets')

# test time adapt
parser.add_argument('--test_time_steps', type=int, default=20)
parser.add_argument('--adapt_option', type=str, default="")  # "" means no adaptation. "ssl" and "meta_loss" mean options
parser.add_argument('--adapt_params', type=str, default="edge_gnn")  # "" means no adaptation. "ssl" and "meta_loss" mean options

# System configuration
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--debug', action='store_true', default=False, help='Enable memory-saving mode.')
# Parse arguments

args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Construct result directory name
if 'spmotif' in args.dataset.lower():
    result_fname = f"{args.dataset}_randomFeats_{args.addRandomFeature}/"
else:
    result_fname = f"{args.dataset}/"

print ('result name:',result_fname)


if args.adapt_option != "":
    result_fname += f"TTT_{args.adapt_option}_{args.adapt_params}/"
if args.fname_str != "":
    result_fname += f"{args.fname_str}/"
if not args.useAutoAug:
    result_fname += f"ERM/{args.base_gnn}_nhid_{args.nhid}_nlayers_{args.nlayers}_dropout_{args.dropout}"
else:
    result_fname += f"autoAug_{args.useAutoAug}/{args.base_gnn}_nhid_{args.nhid}_nlayers_{args.nlayers}_dropout_{args.dropout}"
result_fname += f"_saveMem_{args.save_mem}_jk_{args.jk}_withBN_{args.with_bn}"
result_fname += f"_penalty_{args.penalty}_gradPenalty_{args.gradMatching_penalty}"
result_fname += f"_egLayers_{args.edge_gnn_layers}_edgeBudget_{args.edge_budget}_edgeUniformPenalty_{args.edge_uniform_penalty}_edgePenalty_{args.edge_penalty}_edgeProbThres_{args.edge_prob_thres}_numSamples_{args.num_samples}_seed_{args.seed}/"


# Example usage
result_dir = os.path.join("exp_results/", result_fname)
result_dir_top3 = os.path.join("exp_results/top3/", result_fname)

# if result_dir exists, sys.exit()
if os.path.exists(result_dir) and not args.debug:
    print (result_dir)
    sys.exit("Result directory already exists. Exiting...")

args = vars(args)

dataset_name = args['dataset']
if 'drugood' in dataset_name.lower():
    metric_name = 'auc'
if 'spmotif' in dataset_name.lower():
    metric_name = 'acc'
if 'ogbg' in dataset_name.lower():
    metric_name = 'auc'
s = time.time()

args['device'] = 'cpu' if not torch.cuda.is_available() else args['device']
print ('using device:',args['device'])


dataset_name = args["dataset"]
args['dataset_name'] = dataset_name

if 'drugood' in dataset_name.lower():
    args['nclass'] = 2
    args["nfeat"] = 39


if 'spmotif' in dataset_name.lower():
    args['nclass'] = 3
    args["nfeat"] = 10  if args["addRandomFeature"] else 4

#! Init dataset

workers = 2 if torch.cuda.is_available() else 0
if args['dataset'].lower().startswith('spmotif') or args['dataset'].lower().startswith('tspmotif'):
    train_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='train',transform=FeatureSelect(args["nfeat"]))
    val_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='val',transform=FeatureSelect(args["nfeat"]))
    test_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='test',transform=FeatureSelect(args["nfeat"]))
    train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    valid_loader = DataLoader(val_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)

    args['nclass'] = 3
    metric_name='acc'
    args['valid_metric'] = metric_name


elif args['dataset'].lower().startswith('drugood'):
    #drugood_lbap_core_ic50_assay.json
    metric_name='auc'
    args['valid_metric'] = metric_name
    config_path = os.path.join("configs", args["dataset"] + ".py")
    cfg = Config.fromfile(config_path)
    root = os.path.join(args["root"],"DrugOOD")
    train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args["dataset"], mode="train")
    val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args["dataset"], mode="ood_val")
    test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args["dataset"], mode="ood_test")
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=workers)
    valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader_single_data = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=workers)
    print ('len of test dataset of:',args['dataset'])
    print (len(test_dataset))



elif args["dataset"].lower().startswith('ogbg'):
    def add_zeros(data):
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
        return data

    if 'ppa' in args["dataset"].lower():
        dataset = PygGraphPropPredDataset(root=args["root"], name=args["dataset"], transform=add_zeros)
        input_dim = -1
        num_classes = dataset.num_classes
    else:
        dataset = PygGraphPropPredDataset(root=args["root"], name=args["dataset"])
        input_dim = 1
        num_classes = dataset["num_tasks"]
        args['nclass'] = dataset["num_tasks"]
        args["nfeat"] = 1
    split_idx = dataset.get_idx_split()
    metric_name = 'auc' #! watch out this!
    args['valid_metric'] = metric_name
    ### automatic evaluator. takes dataset name as input

    train_loader = DataLoader(dataset[split_idx["train"]],
                                batch_size=args["batch_size"],
                                shuffle=True,
                                num_workers=workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]],
                                batch_size=args["batch_size"],
                                shuffle=False,
                                num_workers=workers)
    test_loader = DataLoader(dataset[split_idx["test"]],
                                batch_size=args["batch_size"],
                                shuffle=False,
                                num_workers=workers)

elif args["dataset"].lower() in ['graph-sst5','graph-sst2']:
    dataset = get_dataset(dataset_dir=args["root"], dataset_name=args["dataset"], task=None)
    dataloader,train_dataset,val_dataset,test_dataset\
            = get_dataloader_per(dataset, batch_size=args["batch_size"], small_to_large=True, seed=args["seed"],return_set=True)
    train_loader = dataloader['train']
    valid_loader = dataloader['eval']
    test_loader = dataloader['test']
    test_loader_single_data = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=workers)
    
    input_dim = 768
    num_classes = int(args["dataset"][-1].lower()) if args["dataset"][-1].lower() in ['2', '5'] else 3
    args['nclass'] = num_classes
    args["nfeat"] = input_dim
    metric_name='acc'
    args['valid_metric'] = metric_name


elif args["dataset"].lower() in ['graph-twitter']:
    dataset = get_dataset(dataset_dir=args["root"], dataset_name=args["dataset"], task=None)
    dataloader,train_dataset,val_dataset,test_dataset \
    = get_dataloader_per(dataset, batch_size=args["batch_size"], small_to_large=False, seed=args["seed"],return_set=True)
    train_loader = dataloader['train']
    valid_loader = dataloader['eval']
    test_loader = dataloader['test']
    test_loader_single_data = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=workers)
    input_dim = 768
    num_classes = int(args["dataset"][-1].lower()) if args["dataset"][-1].lower() in ['2', '5'] else 3
    args['nclass'] = num_classes
    args["nfeat"] = input_dim
    metric_name='acc'
    args['valid_metric'] = metric_name
    
elif args["dataset"].lower() in ['cmnist']:
    n_val_data = 5000
    train_dataset = CMNIST75sp(os.path.join(args["root"], 'CMNISTSP/'), mode='train')
    test_dataset = CMNIST75sp(os.path.join(args["root"], 'CMNISTSP/'), mode='test')
    perm_idx = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(0))
    test_val = test_dataset[perm_idx]
    val_dataset, test_dataset = test_val[:n_val_data], test_val[n_val_data:]
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True,num_workers=workers)
    valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    # print (test_dataset)
    # sys.exit(0)
    # test_loader_single_data = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=workers)
    input_dim = 7
    num_classes = 2
    args['nclass'] = num_classes
    args["nfeat"] = input_dim
    metric_name='acc'
    args['valid_metric'] = metric_name

elif args["dataset"].lower() in ['mutag', 'proteins', 'dd', 'nci1', 'nci109']:
    args.root = "data"
    if args.dataset.lower() == 'mutag':
        dataset = Mutag(osp.join(args["root"], "TU","Mutagenicity"))
        perm_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
        train_idx = perm_idx[:len(dataset) * 8 // 10].numpy()
        val_idx = perm_idx[len(dataset) * 8 // 10:len(dataset) * 9 // 10].numpy()
        test_idx = perm_idx[len(dataset) * 8 // 10:].numpy()
        # train_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'train_idx.txt'), dtype=np.int64)
        # val_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'val_idx.txt'), dtype=np.int64)
        # test_idx = np.loadtxt(osp.join(args.root, "TU", args.dataset.upper(), 'test_idx.txt'), dtype=np.int64)
    else:
        dataset = TUDataset(osp.join(args["root"], "TU"), name=args["dataset"].upper())
        train_idx = np.loadtxt(osp.join(args["root"], "TU", args["dataset"].upper(), 'train_idx.txt'), dtype=np.int64)
        val_idx = np.loadtxt(osp.join(args["root"], "TU", args["dataset"].upper(), 'val_idx.txt'), dtype=np.int64)
        test_idx = np.loadtxt(osp.join(args["root"], "TU", args["dataset"].upper(), 'test_idx.txt'), dtype=np.int64)

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True,num_workers=workers)
    valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    input_dim = dataset[0].x.size(1)
    num_classes = dataset.num_classes
    args['nclass'] = num_classes
    args["nfeat"] = input_dim
    metric_name='acc'
    args['valid_metric'] = metric_name
else:
    raise Exception("Invalid dataset name")

# log
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

#! init model

model = Model(**args)
model.to(args['device'])


# print ('model summary')
# print (model)
# q = model.state_dict()
# q = list(q.keys())
# fil = [i for i in q if 'edge_linear' in i or 'dataAug_gnn' in i or 'meta_loss' in i]
# print (fil)
# sys.exit(0)


#! here main.

if args['erm']:
    print ('running ERM')
    model.fit_erm(train_loader,valid_loader,test_loader,epochs=args["epochs"])
    res = model.valid_metric_list
    res = sorted(res,key = lambda x:x[0],reverse=True)
    debug_out = res[:3]
    val_score,test_score = res[0]
    test_score_top3 = np.max([x[1] for x in res[:3]])
    res = np.array([val_score,test_score])
    res_top3 = np.array([val_score,test_score_top3])
    if not args['debug']:
        save_numpy_array_to_file(res,result_dir,"val_test_metric")
        save_numpy_array_to_file(res_top3,result_dir_top3,"val_test_metric")
    else:
        print ('final output for cls, top3_cls:')
        print (debug_out)

else:
    if args['fname_str'] == '' or 'meta_loss' not in args['fname_str']:
        model.fit(train_loader,valid_loader,test_loader,epochs=args["epochs"])
        model.load_state_dict(model.best_states)
        res = model.valid_metric_list
        res = sorted(res,key = lambda x:x[0],reverse=True)
        debug_out = res[:3]
        # print ('top 5:',res[:5])
        val_score,test_score = res[0]
        test_score_top3 = np.max([x[1] for x in res[:3]])
        # test_auc = model.evaluate_model(test_loader,'test',True)
        # val_score = model.best_valid_metric

        res = np.array([val_score,test_score])
        res_top3 = np.array([val_score,test_score_top3])
        if not args['debug']:
            save_numpy_array_to_file(res,result_dir,"val_test_metric")
            save_numpy_array_to_file(res_top3,result_dir_top3,"val_test_metric")
        else:
            print ('final output for cls, top3_cls:')
            print (debug_out)
            

    if 'meta_loss' in args['fname_str']:
        model.fit_meta_loss(train_loader,valid_loader,test_loader,epochs=args["epochs"],meta_opt_use_all_params=True)
        res = model.valid_metric_list
        res = sorted(res,key = lambda x:x[0],reverse=True)
        debug_out = res[:3]
        val_score,test_score = res[0]
        test_score_top3 = np.max([x[1] for x in res[:3]])
        # model.load_state_dict(model.best_states)
        # model.eval()
        # test_auc = model.evaluate_model(test_loader,'test',True)
        # val_score = model.best_valid_metric
        res = np.array([val_score,test_score])
        res_top3 = np.array([val_score,test_score_top3])
        
        if not args['debug']:
            save_numpy_array_to_file(res,result_dir,"val_test_metric")
            save_numpy_array_to_file(res_top3,result_dir_top3,"val_test_metric")
        else:
            print ('final output for cls, top3_cls:')
            print (debug_out)



    # for b in train_loader:
    #     model.fit_meta_loss_step(b)
    #     break
    # model.fit_mixed(train_loader,valid_loader,test_loader,epochs=args["epochs"])

    #! save results for every initialization of labelled pool
    # print (f'save results for iter:{trialId}')


