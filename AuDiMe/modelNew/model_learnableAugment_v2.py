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
from torch.autograd import grad
from itertools import chain
from torchviz import make_dot
import sys
from tqdm import tqdm

import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader,Data

from torchmetrics import AUROC,Accuracy
import numpy as np
import pandas as pd
import random
import string
from termcolor import colored
from copy import deepcopy

from modelNew.base_model import BaseModel
from modelNew.ssl_module import Encoder
from modelNew.gcn import GCN
from modelNew.gin import GIN
from modelNew.utils import *

# try:
#     from modelNew.base_model import BaseModel
#     from modelNew.ssl_module import Encoder
#     from modelNew.gcn import GCN
#     from modelNew.gin import GIN
#     from modelNew.utils import *
# except:
#     print ('import from local')
#     from base_model import BaseModel
#     from gcn import GCN
#     from gin import GIN
#     from ssl_module import Encoder
#     from utils import *


class Model(BaseModel):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5,edge_dim = -1,save_mem=True,jk='last',node_cls=False,pooling='sum',
                with_bn=False, weight_decay=5e-6,lr=1e-3,adapt_lr=1e-4,lr_scheduler=True,patience=50,early_stop_epochs=5,lr_decay=0.75,penalty=0.1,gradMatching_penalty=1.0,project_layer_num=2,edge_gnn='none',edge_gnn_layers=2,edge_budget=0.75,num_samples=1,edge_penalty=1.0,edge_uniform_penalty=1e-2,edge_prob_thres=50,featureMasking=True,temp=0.2,adapt_params='edge_gnn', with_bias=True,base_gnn='gin',valid_metric='acc', device='cpu', **args):

        super(Model, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.rnd_id = ''.join(random.choices(string.digits, k=16))  # for caching stuffs
        self.debug = args['debug']
        self.useAutoAug = args['useAutoAug']
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.best_states = None
        self.best_meta_model = None
        self.device = device
        self.jk = jk
        self.node_cls = node_cls
        self.edge_dim = edge_dim
        self.cls_header = self.create_mlp(nhid,nhid,nclass,project_layer_num)
        self.ssl_header = self.create_mlp(nhid,nhid,nclass,project_layer_num,cls=False)
        self.linear_refit_layer = self.create_mlp(nhid,nhid,nclass,project_layer_num)
        self.meta_linear_cls = nn.Linear(nhid, nclass)

        self.penalty = penalty
        self.gradMatching_penalty = gradMatching_penalty
        self.edge_budget = edge_budget
        self.edge_penalty = edge_penalty
        self.edge_uniform_penalty = edge_uniform_penalty
        self.featureMasking = featureMasking
        self.edge_prob_thres = edge_prob_thres
        
        self.featsMask = nn.Parameter(torch.zeros(nfeat)+5.0).view(1,-1).to(device)
        self.label_emb = nn.Embedding(nclass, nhid)
        
        #  pretraining epochs
        self.pe = args['pretraining_epochs']
        
        
        if base_gnn=='gin':
            self.gnn = GIN(nfeat, nhid, nclass, nlayers, dropout=dropout,edge_dim=edge_dim,jk=jk,node_cls=node_cls,pooling=pooling,
                with_bn=with_bn, weight_decay=weight_decay)
        if base_gnn=='gcn':
            self.gnn = GCN(nfeat, nhid,nclass, nlayers=nlayers, dropout=dropout,save_mem=save_mem,jk=jk,node_cls=node_cls,pooling=pooling,
                with_bn=with_bn, weight_decay=weight_decay, with_bias=with_bias)
        
        self.edge_gnn = edge_gnn
        if edge_gnn=='gin':
            self.dataAug_gnn = GIN(nfeat, nhid, nclass, nlayers=edge_gnn_layers, edge_dim=edge_dim,dropout=dropout,jk='last',node_cls=True,
                with_bn=with_bn, weight_decay=weight_decay)
        if edge_gnn=='gcn':
            self.dataAug_gnn = GCN(nfeat, nhid,nclass, nlayers=edge_gnn_layers, dropout=dropout,save_mem=save_mem,jk='last',node_cls=True,
                with_bn=with_bn, weight_decay=weight_decay, with_bias=with_bias)
        if edge_gnn != 'none':
            self.edge_linear = nn.Sequential(nn.Linear(3*nhid, 2*nhid),nn.ReLU(),nn.Linear(2*nhid, 2)) 
        
        self.edge_loss = DataAugLoss(threshold=edge_budget)

        self.gnn.to(device)
        aug1 = A.EdgeRemoving(pe=0.1)
        aug2 = A.EdgeRemoving(pe=0.1)
        
        self.encoder_model = Encoder(self.gnn, [aug1,aug2],learnable_aug=self.useAutoAug).to(device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=temp), mode='G2G').to(device)
        self.contrast_model_non_agg = DualBranchContrast(loss=modInfoNCE(tau=temp), mode='G2G').to(device)
        
        self.lr_scheduler = lr_scheduler
        if adapt_params=='edge_gnn':
            self.edge_gnn_optimizer = Adam(list(self.dataAug_gnn.parameters())+list(self.edge_linear.parameters()), lr=adapt_lr,weight_decay=5e-4)
        elif adapt_params=='edge_linear':
            self.edge_gnn_optimizer = Adam(self.edge_linear.parameters(), lr=adapt_lr,weight_decay=5e-4)
        else:
            assert "Please specify  correct 'adapt_params'!"
        
        # meta loss learner
        # self.meta_loss_head = nn.Sequential(nn.Linear(nhid+1, nhid),nn.ReLU(),nn.Linear(nhid, nhid),nn.ReLU()).to(device)
        # self.meta_loss1 = nn.Linear(nhid, 1).to(device)
        # self.meta_loss2 = nn.Linear(nhid, 2).to(device)
        
        self.meta_loss_head = nn.Sequential(nn.Linear(1, 32),nn.ReLU(),nn.Linear(32, 32),nn.ReLU()).to(device)
        self.meta_loss1 = nn.Linear(32, 1).to(device)
        self.meta_loss2 = nn.Linear(32, nclass).to(device)
        
        # self.exclude_meta_mlp_optimizer = Adam([params for name, params in self.named_parameters() if "meta" not in name], lr=lr)
        
        self.optimizer = Adam(self.parameters(), lr=lr,weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_decay, patience=patience, min_lr=1e-3)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.metric_func = Accuracy(task='multiclass',num_classes=nclass,top_k=1).to(device) if valid_metric=='acc' else AUROC(task='binary').to(device)
        self.metric_name = valid_metric
        self.train_grad_sim = []
        self.val_grad_sim = []
        
        self.valid_metric_list = []
        self.meta_valid_metric_list = []
        self.best_valid_metric = -1.
        self.test_metric=-1.
        self.early_stop_epochs = early_stop_epochs
        self.epochs_since_improvement=0
        self.stop_training=False
        
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.train_grad_sim = []
        self.val_grad_sim = []
        self.test_grad_sim = []
        self.gradsim_avg_ssl = []
        self.gradsim_avg_meta = []


    def create_mlp(self,input_dim, hidden_dim, output_dim, num_layers,cls=True):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        if cls:
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    

    def fit_meta_loss(self,dataloader,valid_dloader,test_dloader=None,epochs=10,meta_opt_use_all_params=True):
        self.epochs_since_improvement=0
        self.stop_training=False
        if not meta_opt_use_all_params:
            self.meta_loss_optimizer = Adam(list(self.dataAug_gnn.parameters())+list(self.edge_linear.parameters())+list(self.meta_loss_head.parameters())+list(self.meta_loss1.parameters())+list(self.meta_loss2.parameters()),lr=self.lr,weight_decay=self.weight_decay)
        else:
            self.meta_loss_optimizer = Adam(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)

        state = deepcopy([self.useAutoAug,self.encoder_model.learnable_aug,0,self.edge_uniform_penalty,self.penalty,self.edge_penalty,self.gradMatching_penalty])
        if self.pe>0:
            self.fit_erm(dataloader,valid_dloader,test_dloader,epochs=self.pe)
        self.best_valid_metric = 0.
        
        self.useAutoAug,self.encoder_model.learnable_aug,self.epochs_since_improvement,self.edge_uniform_penalty,self.penalty,self.edge_penalty,self.gradMatching_penalty = state
        for e in range(epochs):
            if self.stop_training: break
            print (colored(f'Current Epoch {e} for meta loss learning','red','on_yellow'))
            total_losses = 0.
            steps = 0.
            self.meta_loss_optimizer.zero_grad()
            for data in dataloader:
                t_ = self.fit_meta_loss_step(data)
                if t_==-1: continue
                step_loss,meta_ce_loss,ssl_loss = t_
                # tot_loss = meta_ce_loss + self.penalty*ssl_loss.mean()  # no cls
                tot_loss = step_loss + self.gradMatching_penalty*meta_ce_loss + self.penalty*ssl_loss.mean()
                tot_loss.backward()
                self.meta_loss_optimizer.step()
                self.optimizer.zero_grad()
                
                total_losses += tot_loss
                steps +=1
                # use colored to print three losses

            train_metric = self.evaluate_model(dataloader,'train',is_dataloader=True)
            valid_metric = self.evaluate_model(valid_dloader,'valid',is_dataloader=True)
            test_metric = self.evaluate_model(test_dloader,'test',is_dataloader=True)
            self.valid_metric_list.append((valid_metric,test_metric))
            self.train_metrics.append(train_metric)
            self.val_metrics.append(valid_metric)
            self.test_metrics.append(test_metric)
            
            # self.evaluate_gradient(valid_dloader,'valid',is_dataloader=True)
            print (colored(f'Epoch {e} Total Loss: {total_losses/steps}, \n Train metric score:{train_metric}; Valid metric score:{valid_metric}; Test metric: {test_metric}','blue','on_yellow'))
            
            # train_metric_score = self.evaluate_model(dataloader,'train',is_dataloader=True)
            # val_metric_score = self.evaluate_model(valid_dloader,'valid',is_dataloader=True)
            # self.train_metrics.append(train_metric_score)
            # self.val_metrics.append(val_metric_score)
    
    
    def fit_meta_loss_step(self,data,phase='train'):
        # update meta loss mlp \phi.
        ## get ssl loss
        data = data.to(self.device)
        y = data.y
        if phase=='train':
            self.train()
        else:
            self.eval()
        
        self.meta_loss_optimizer.zero_grad()
        assert self.edge_gnn!='none', "edge_gnn shouldn't be none!"

        edge_weights1,_ = self.learn_edge_weight(data)
        edge_weights2,_ = self.learn_edge_weight(data)

        self.encoder_model.learnable_edge_weight1 = edge_weights1
        self.encoder_model.learnable_edge_weight2 = edge_weights2

        _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch,edge_attr=data.edge_attr if self.useAutoAug else None)
        g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
        ssl_loss = self.contrast_model_non_agg(g1=g1, g2=g2, batch=data.batch).view(-1,1)
        # update meta loss mlp
        # meta_loss = self.penalty*torch.mean(self.meta_loss_mlp(g1+g2))
        # meta_loss = self.meta_loss1(self.meta_loss_head(torch.cat([ssl_loss,g1*0.],dim=1)))
        meta_loss = self.meta_loss1(self.meta_loss_head(ssl_loss))
        mean_meta_loss = self.penalty*meta_loss.mean()
        if phase=='valid':
            return meta_loss.mean()
        if  torch.isnan(mean_meta_loss):
            return -1.
        # perform gd update on meta loss
        mean_meta_loss.backward()
        self.meta_loss_optimizer.step()
        self.meta_loss_optimizer.zero_grad()
        
        # forward meta loss again
        edge_weights1,edge_tv_distance1 = self.learn_edge_weight(data)
        edge_weights2,edge_tv_distance2 = self.learn_edge_weight(data)
        
        self.encoder_model.learnable_edge_weight1 = edge_weights1
        self.encoder_model.learnable_edge_weight2 = edge_weights2

        _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch,edge_attr=data.edge_attr if self.useAutoAug else None)
        g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
        ssl_loss = self.contrast_model_non_agg(g1=g1, g2=g2, batch=data.batch).view(-1,1)
        
        # meta_logits = self.meta_loss2(self.meta_loss_head(torch.cat([ssl_loss,g1*0.],dim=1)))
        meta_logits = self.meta_loss2(self.meta_loss_head(ssl_loss))
        meta_loss_ce = F.cross_entropy(meta_logits, y)
        if self.debug:
            print (f'meta train phase: ssl_loss {ssl_loss.mean().item()}, meta learned loss: {mean_meta_loss.item()/self.penalty}, meta ce loss: {meta_loss_ce.item()}')

        # standard cross-entropy ERM
        edge_weight,edge_tv_distance3 = self.learn_edge_weight(data)
        if self.featureMasking:
            g = self.gnn(data.x*torch.sigmoid(self.featsMask), data.edge_index,edge_attr = data.edge_attr, batch=data.batch,edge_weight=edge_weight, return_both_rep=False)
        else:
            g = self.gnn(data.x, data.edge_index,edge_attr = data.edge_attr, edge_weight=edge_weight, batch=data.batch, return_both_rep=False)
        logits = self.cls_header(g)
        loss = F.cross_entropy(logits, y)
        tot_edges = data.edge_index.shape[1]
        edge_reg = self.edge_loss(edge_weight.sum()/tot_edges)
        step_loss = loss + self.edge_penalty*edge_reg + self.edge_uniform_penalty*(edge_tv_distance1+edge_tv_distance2+edge_tv_distance3)/3.0
        if step_loss>1e4 or torch.isnan(step_loss):
            return -1.
        # print (f'cls loss:{loss.item()}, ssl_loss:{ssl_loss.mean().item()}, meta_ce_loss:{meta_loss_ce.item()}, mean_meta_loss:{mean_meta_loss.item()}')
        return step_loss,meta_loss_ce,ssl_loss


    def valid_meta_loss(self,dataloader):
        # calc cos similarity of meta loss and ce loss
        self.eval()
        loss_cls = []
        meta_loss_cls = []
        # opt = torch.optim.Adam(self.parameters(), lr=1e-2,weight_decay=5e-6)
        assert self.edge_gnn!='none', "edge_gnn shouldn't be none!"
        for data in dataloader:
            edge_weights1 = self.learn_edge_weight(data)
            edge_weights2 = self.learn_edge_weight(data)
            self.encoder_model.learnable_edge_weight1 = edge_weights1
            self.encoder_model.learnable_edge_weight2 = edge_weights2
            
            _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
            g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
            ssl_loss = self.contrast_model_non_agg(g1=g1, g2=g2, batch=data.batch).view(-1,1)
            
            # update meta loss mlp
            meta_loss = self.meta_loss_mlp(torch.cat([ssl_loss,g1+g2],dim=1))
            
            edge_weight = self.learn_edge_weight(data)
            if self.featureMasking:
                g = self.gnn(data.x*torch.sigmoid(self.featsMask), data.edge_index, batch=data.batch,edge_weight=edge_weight, return_both_rep=False)
            else:
                g = self.gnn(data.x, data.edge_index,edge_weight=edge_weight,  batch=data.batch, return_both_rep=False)
            logits = self.cls_header(g)
            loss = F.cross_entropy(logits, data.y,reduction='none')
            loss = loss.view(-1)
            meta_loss = meta_loss.view(-1)
            loss_cls.append(loss)
            meta_loss_cls.append(meta_loss)
        loss = torch.cat(loss_cls).view(1,-1)
        meta_loss = torch.cat(meta_loss_cls).view(1,-1)
        # calc normalized correlation between meta loss and ce loss
        score = F.cosine_similarity(meta_loss,loss).item()
        return score
    
    
    def learn_edge_weight(self, data,tau=0.2):
        edge_index = data.edge_index
        X = self.dataAug_gnn(data.x,data.edge_index,edge_attr=data.edge_attr) # (N,F)
        s = X[edge_index[0]]
        t = X[edge_index[1]]
        edge_embeddings1 = torch.cat([s, t, s + t], dim=1)
        edge_embeddings2 = torch.cat([t, s, s + t], dim=1)
        edge_logits = self.edge_linear(edge_embeddings1+edge_embeddings2)
        # calc egge_prob and TV distance
        edge_probs = F.softmax(edge_logits, dim=1)[:,1]
        sorted_probs, _ = torch.sort(edge_probs)
        k = int(len(sorted_probs) * self.edge_prob_thres / 100.0)  # Calculate the number of lowest values to select
        sorted_probs = sorted_probs[:k]
        edge_tv_distance = total_variation_distance(sorted_probs)
        edge_weights = F.gumbel_softmax(edge_logits, tau=tau, hard=True, dim=1)[:,1]
        return edge_weights,edge_tv_distance


    def train_ssl_one_step(self,data):
        # self.encoder_model.train()
        data = data.to(self.device)
        y = data.y.long().view(-1,)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        #! using learned weights for edge removal
        tot_edges = data.edge_index.shape[1]
        assert self.edge_gnn!='none', "edge_gnn shouldn't be none!"
        if self.useAutoAug:
            edge_weights1,edge_tv_distance1 = self.learn_edge_weight(data)
            edge_weights2,edge_tv_distance2 = self.learn_edge_weight(data)
            self.encoder_model.learnable_edge_weight1 = edge_weights1
            self.encoder_model.learnable_edge_weight2 = edge_weights2
        else:
            edge_weights1,edge_tv_distance1 = None,None
            edge_weights2,edge_tv_distance2 = None,None
        _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch,edge_attr=data.edge_attr if self.useAutoAug else None,label_emb = None)
        g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
        loss = self.contrast_model(g1=g1, g2=g2, batch=data.batch)
        # edge_reg = (torch.sum(edge_weights1)/tot_edges-self.edge_budget)**2 + (torch.sum(edge_weights2)/tot_edges-self.edge_budget)**2
        if self.useAutoAug:
            edge_reg = self.edge_loss(edge_weights1.sum()/tot_edges) + self.edge_loss(edge_weights2.sum()/tot_edges)
            return loss + self.edge_penalty*edge_reg + self.edge_uniform_penalty*(edge_tv_distance1+edge_tv_distance2)/2.0
        else:
            return loss


    def train_labelled_one_step(self,data):
        # self.encoder_model.train()
        data = data.to(self.device)
        y = data.y
        tot_edges = data.edge_index.shape[1]
        edge_weight,edge_tv_distance = self.learn_edge_weight(data) if self.useAutoAug else (None,None)
        g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=edge_weight, return_both_rep=False)
        logits = self.cls_header(g)
        loss = self.ce_loss(logits, y)
        # edge_reg = (torch.sum(edge_weight)/tot_edges-self.edge_budget)**2
        if self.useAutoAug:
            edge_reg = self.edge_loss(edge_weight.sum()/tot_edges)
            return loss + self.edge_penalty*edge_reg + self.edge_uniform_penalty*edge_tv_distance
        else:
            return loss


    def fit_mixed_joint(self,dataloader,valid_dloader,test_dloader,epochs=50):
        # train erm first, then load best model, train meta loss learner
        # self.fit(dataloader,valid_dloader,test_dloader,epochs=epochs)
        # self.load_state_dict(self.best_model)
        for batch in dataloader:
            meta_loss = self.fit_meta_loss_step(batch)
            erm_loss = self.train_labelled_one_step(batch)
            tot_loss = self.penalty*meta_loss + erm_loss
            tot_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_metric_score = self.evaluate_model(dataloader,'train',is_dataloader=True)
        val_metric_score = self.evaluate_model(valid_dloader,'valid',is_dataloader=True)
        if test_dloader is not None:
            test_metric_score= self.evaluate_model(test_dloader,'test',is_dataloader=True)
        #? put aside
        
        



    # def test_time_adapt(self,dataloader,valid_dloader,test_dloader,test_dloader_single_data,epochs=50,test_time_steps = 10,option='ssl'): # can be ssl or meta_loss
    #     self.to(self.device)
    #     test_perf_tracker = []
    #     test_perf_score = []
    #     labels = []
    #     batch_size = test_dloader_single_data.batch_size
    #     if option=='ssl':
    #         self.fit(dataloader,valid_dloader,test_dloader,epochs=epochs)
    #     elif option=='meta_loss':
    #         self.fit_meta_loss(dataloader,valid_dloader,test_dloader,epochs=epochs)
    #     else:
    #         assert "wrong option for test-time adaptation!"
        
    #     self.load_state_dict(self.best_states)
    #     metric_score = self.evaluate_model(test_dloader,'test',is_dataloader=True)   # for step 0 
    #     test_perf_score.append(metric_score)
    #     self.optimizer.zero_grad()
    #     for batch in tqdm(test_dloader_single_data):
    #         batch = batch.to(self.device)
    #         labels.append(batch.y.view(-1,))
    #         test_perf_tracker_step = []
    #         for i in range(test_time_steps):
    #             loss_val = self.train_ssl_one_step(batch)
    #             loss_val.backward()
    #             self.edge_gnn_optimizer.step()
    #             self.optimizer.zero_grad()
    #             # get the edge weight
    #             min_loss = 1e5
    #             best_edge_weight = None
    #             for _ in range(10):
    #                 loss_,w_ = self.calc_best_loss_(batch,option=option)
    #                 if loss_<min_loss:
    #                     min_loss = loss_
    #                     best_edge_weight = w_    
                
    #             logits = self.evaluate_model(batch,'test',is_dataloader=False,best_edge_weight=best_edge_weight)
    #             test_perf_tracker_step.append(logits.view(batch_size,1,-1))
    #             self.load_state_dict(self.best_states)

    #         test_perf_tracker_step = torch.cat(test_perf_tracker_step,dim=1) # shape:(B,20,C)
    #         # test_perf_tracker_step = test_perf_tracker_step.unsqueeze(0) # shape:(B,20,C)
    #         test_perf_tracker.append(test_perf_tracker_step)
    #     test_perf_tracker = torch.cat(test_perf_tracker,dim=0) # shape:(N,20,C)
    #     labels = torch.cat(labels,dim=0)
    #     for j in range(test_time_steps):
    #         logits = test_perf_tracker[:,j,:]
    #         if self.metric_name=='acc':
    #             metric_score = self.metric_func(logits, labels).item()
    #         if self.metric_name=='auc':
    #             metric_score = self.metric_func(logits[:,1], labels).item()
    #         test_perf_score.append(metric_score)
    #     if self.debug:
    #         print ("test perf tracker:")
    #         print (test_perf_score)
    #     return test_perf_score # (test_time_steps,)
    


    def fit(self,dataloader,valid_dloader,test_dloader=None,epochs=50):
        if self.pe>0:
            state = deepcopy([self.useAutoAug,self.encoder_model.learnable_aug,0,self.edge_uniform_penalty,self.penalty,self.edge_penalty])
            self.useAutoAug = False
            self.encoder_model.learnable_aug=False
            self.epochs_since_improvement=0
            self.edge_uniform_penalty= 0.
            self.penalty = 0.
            self.edge_penalty = 0.
        for e in range(epochs):
            print (colored(f'Current Epoch {e}','red','on_yellow'))
            if e==10 and self.pe>0:
                self.useAutoAug,self.encoder_model.learnable_aug,self.epochs_since_improvement,self.edge_uniform_penalty,self.penalty,self.edge_penalty = state
                self.best_valid_metric = 0.
            if self.stop_training:
                break
            erm_losses = 0.
            ssl_losses = 0.
            total_losses = 0.
            steps = 0
            for data in dataloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                ssl_loss = self.train_ssl_one_step(data)
                labelled_loss = self.train_labelled_one_step(data)
                # do sth to get the gradients
                loss = self.penalty*ssl_loss + labelled_loss
                # print ('label loss:',labelled_loss.item(),'ssl loss:',ssl_loss.item()*self.penalty)
                
                loss.backward()
                self.optimizer.step()
                erm_losses += labelled_loss.item()
                ssl_losses += ssl_loss.item()
                total_losses += loss.item()
                steps +=1
                # use colored to print three losses
            
            print (colored(f'Epoch {e} SSL Loss: {ssl_losses/steps} Labelled Loss (with autoaug loss): {erm_losses/steps} Total Loss: {total_losses/steps}','red','on_white'))
            train_metric_score = self.evaluate_model(dataloader,'train',is_dataloader=True)
            val_metric_score = self.evaluate_model(valid_dloader,'valid',is_dataloader=True)
            self.train_metrics.append(train_metric_score)
            self.val_metrics.append(val_metric_score)
            # self.train_grad_sim.append(train_avg_grad_sim)
            # self.val_grad_sim.append(val_avg_grad_sim)     
            
            if test_dloader is not None:
                test_metric_score= self.evaluate_model(test_dloader,'test',is_dataloader=True)
                self.test_metrics.append(test_metric_score)
                # self.test_grad_sim.append(test_avg_grad_sim)
            self.valid_metric_list.append((val_metric_score,test_metric_score))
            self.train()

    def fit_erm(self,dataloader,valid_dloader,test_dloader=None,epochs=50):
        self.useAutoAug = False
        self.encoder_model.learnable_aug=False
        self.epochs_since_improvement=0
        self.edge_uniform_penalty= 0.
        self.penalty = 0.
        self.edge_penalty = 0.
        
        for e in range(epochs):
            print (colored(f'Current Epoch {e}','red','on_yellow'))
            if self.stop_training:
                break
            erm_losses = 0.
            ssl_losses = 0.
            total_losses = 0.
            steps = 0
            for data in dataloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                labelled_loss = self.train_labelled_one_step(data)
                # do sth to get the gradients
                loss = labelled_loss
                # print ('label loss:',labelled_loss.item(),'ssl loss:',ssl_loss.item()*self.penalty)
                
                loss.backward()
                self.optimizer.step()
                erm_losses += labelled_loss.item()
                steps +=1
                # use colored to print three losses
            
            print (colored(f'Epoch {e}: Labelled Loss: {erm_losses/steps}','red','on_white'))
            train_metric_score = self.evaluate_model(dataloader,'train',is_dataloader=True)
            val_metric_score = self.evaluate_model(valid_dloader,'valid',is_dataloader=True)
            self.train_metrics.append(train_metric_score)
            self.val_metrics.append(val_metric_score)
            # self.train_grad_sim.append(train_avg_grad_sim)
            # self.val_grad_sim.append(val_avg_grad_sim)     
            
            if test_dloader is not None:
                test_metric_score= self.evaluate_model(test_dloader,'test',is_dataloader=True)
                self.test_metrics.append(test_metric_score)
                # self.test_grad_sim.append(test_avg_grad_sim)
            self.valid_metric_list.append((val_metric_score,test_metric_score))
            self.train()

    # def evaluate_model(self, data_input,phase,is_dataloader):
    #     """
    #     Evaluate the model on a given dataset.

    #     Args:
    #     - dataloader: DataLoader for the dataset to evaluate.
    #     - phase: Phase of evaluation ('valid' or 'test') to control output messaging.

    #     Returns:
    #     - The average metric value across the dataset.
    #     - The average cosine similarity between SSL and ERM gradients.
    #     """

    #     self.eval()  # Set model to evaluation mode
    #     grads_sim = []
    #     logits_list = []
    #     labels_list = []
    #     steps = 0
    #     with torch.no_grad():
    #         if is_dataloader:
    #             for data in data_input:
    #                 data = data.to(self.device)
    #                 # Metric computation
    #                 edge_weights1 = self.learn_edge_weight(data)
    #                 edge_weights2 = self.learn_edge_weight(data)
                    
    #                 self.encoder_model.learnable_edge_weight1 = edge_weights1
    #                 self.encoder_model.learnable_edge_weight2 = edge_weights2

    #                 _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
    #                 g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
    #                 ssl_loss = self.contrast_model_non_agg(g1=g1, g2=g2, batch=data.batch).view(-1,1)
    #                 logits = self.meta_loss2(self.meta_loss_head(ssl_loss))
    #                 logits_list.append(logits)
    #                 labels_list.append(data.y.view(-1,))
    #                 steps += 1
                
    #             all_logits = torch.cat(logits_list, dim=0)
    #             all_labels = torch.cat(labels_list, dim=0)
        
    #             # Compute metric with all logits and labels
    #             if self.metric_name=='acc':
    #                 metric_score = self.metric_func(all_logits, all_labels).item()
    #             if self.metric_name=='auc':
    #                 metric_score = self.metric_func(all_logits[:,1], all_labels).item() # use pos logits

    #             if phase.lower()=='valid' and self.lr_scheduler:
    #                 self.scheduler.step(metric_score)
                
    #             if phase.lower()=='valid':
    #                 if metric_score>self.best_valid_metric:
    #                     self.best_valid_metric = metric_score
    #                     self.epochs_since_improvement=0
    #                     # save model
    #                     self.best_states = deepcopy(self.state_dict())
    #                 else:
    #                     self.epochs_since_improvement+=1
    #                     if self.epochs_since_improvement>=self.early_stop_epochs:
    #                         self.stop_training=True
    #                         print (colored(f'Early Stopping: No improvement for {self.early_stop_epochs} epochs','red','on_white'))
                
    #             print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_white'))
    #             return metric_score
    #         else:
    #             data = data_input
                
    #             # Metric computation
    #             logits = self.cls_header(self.gnn(data.x, data.edge_index, batch=data.batch, return_both_rep=False))
    #             if self.metric_name=='acc':
    #                 metric_score = self.metric_func(logits, data.y.view(-1,)).item()
    #             if self.metric_name=='auc':
    #                 metric_score = self.metric_func(logits, data.y.view(-1,)).item()
    #             return metric_score



    def evaluate_model_meta_loss(self, data_input,phase,is_dataloader):
        """
        For cls header evaluation
        Evaluate the model on a given dataset.

        Args:
        - dataloader: DataLoader for the dataset to evaluate.
        - phase: Phase of evaluation ('valid' or 'test') to control output messaging.

        Returns:
        - The average metric value across the dataset.
        - The average cosine similarity between SSL and ERM gradients.
        """

        self.eval()  # Set model to evaluation mode
        grads_sim = []
        logits_list = []
        labels_list = []
        steps = 0
        with torch.no_grad():
            if not is_dataloader:
                data_input = [data_input]
            for data in data_input:
                data = data.to(self.device)
                # Metric computation
                if self.useAutoAug:
                    if self.num_samples==1:
                        edge_weight1,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                        edge_weight2,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                    else:
                        edge_weight1_avg = []
                        edge_weight2_avg = []
                        for _ in range(self.num_samples):
                            edge_weight1,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                            edge_weight2,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                            edge_weight1_avg.append(edge_weight1.view(1,-1))
                            edge_weight2_avg.append(edge_weight2.view(1,-1))
                        edge_weight1 = torch.cat(edge_weight1_avg,dim=0).mean(dim=0)
                        edge_weight2 = torch.cat(edge_weight2_avg,dim=0).mean(dim=0)
                        self.encoder_model.learnable_edge_weight1 = edge_weight1
                        self.encoder_model.learnable_edge_weight2 = edge_weight2
                else:
                    edge_weight = None

                _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch,data.edge_attr)
                g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
                ssl_loss = self.contrast_model_non_agg(g1=g1, g2=g2, batch=data.batch).view(-1,1)
                logits = self.meta_loss2(self.meta_loss_head(ssl_loss))
                logits_list.append(logits)
                labels_list.append(data.y.view(-1,))
                steps += 1

            all_logits = torch.cat(logits_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)

            # Compute metric with all logits and labels
            if self.metric_name=='acc':
                metric_score = self.metric_func(all_logits, all_labels).item()
            if self.metric_name=='auc':
                metric_score = self.metric_func(all_logits[:,1], all_labels).item() # use pos logits

            if phase.lower()=='valid' and 0:
                self.scheduler.step(metric_score)
                
            if phase.lower()=='valid':
                if metric_score>self.best_valid_metric:
                    self.best_valid_metric = metric_score
                    self.epochs_since_improvement=0
                    # save model
                    self.best_states = deepcopy(self.state_dict())
                else:
                    self.epochs_since_improvement+=1
                    if self.epochs_since_improvement>=self.early_stop_epochs:
                        self.stop_training=True
                        print (colored(f'Early Stopping: No improvement for {self.early_stop_epochs} epochs','red','on_white'))
            if phase=='test':
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_yellow'))
            else:
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_white'))
            self.train()
            return metric_score
    
    def evaluate_model(self, data_input,phase,is_dataloader=True,meta_loss=False,best_edge_weight=None):
        """
        For cls header evaluation
        Evaluate the model on a given dataset.

        Args:
        - dataloader: DataLoader for the dataset to evaluate.
        - phase: Phase of evaluation ('valid' or 'test') to control output messaging.

        Returns:
        - The average metric value across the dataset.
        - The average cosine similarity between SSL and ERM gradients.
        """

        self.eval()  # Set model to evaluation mode
        grads_sim = []
        logits_list = []
        labels_list = []
        steps = 0
        with torch.no_grad():
            if not is_dataloader:
                data_input = [data_input]
            for data in data_input:
                data = data.to(self.device)
                # Metric computation
                if best_edge_weight is not None:
                    edge_weight = best_edge_weight
                elif self.useAutoAug:
                    if self.num_samples==1:
                        edge_weight,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                    else:
                        edge_weight_avg = []
                        for _ in range(self.num_samples):
                            edge_weight,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                            edge_weight_avg.append(edge_weight.view(1,-1))
                        edge_weight = torch.cat(edge_weight_avg,dim=0).mean(dim=0)
                else:
                    edge_weight = None
                    
                if self.featureMasking:
                    g = self.gnn(data.x*torch.sigmoid(self.featsMask), data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=edge_weight, return_both_rep=False)
                else:
                    g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=edge_weight, return_both_rep=False)
                logits = self.cls_header(g)
                if not is_dataloader:
                    self.train()
                    return logits
                
                logits_list.append(logits)
                labels_list.append(data.y.view(-1,))
                steps += 1

            all_logits = torch.cat(logits_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            # Compute metric with all logits and labels
            if self.metric_name=='acc':
                metric_score = self.metric_func(all_logits, all_labels).item()
            
            if self.metric_name=='auc':
                metric_score = self.metric_func(all_logits[:,1], all_labels).item() # use pos logits
            
            if phase.lower()=='valid' and self.lr_scheduler:
                self.scheduler.step(metric_score)
            
            if phase.lower()=='valid':
                if metric_score>self.best_valid_metric:
                    self.best_valid_metric = metric_score
                    self.epochs_since_improvement=0
                    # save model
                    self.best_states = deepcopy(self.state_dict())
                else:
                    self.epochs_since_improvement+=1
                    if self.epochs_since_improvement>=self.early_stop_epochs:
                        self.stop_training=True
                        print (colored(f'Early Stopping: No improvement for {self.early_stop_epochs} epochs','red','on_white'))
            if phase=='test':
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_yellow'))
            else:
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_white'))
            self.train()
            return metric_score
    
    def evaluate_gradient(self, data_input,phase,is_dataloader):
        """
        Evaluate the model on a given dataset.
        Args:
        - dataloader: DataLoader for the dataset to evaluate.
        - phase: Phase of evaluation ('valid' or 'test') to control output messaging.

        Returns:
        - The average metric value across the dataset.
        - The average cosine similarity between SSL and ERM gradients.
        """

        self.eval()  # Set model to evaluation mode
        grads_sim_ssl = []
        grads_sim_meta = []
        steps = 0
        if is_dataloader:
            for data in data_input:
                # Forward passes to compute losses
                data=data.to(self.device)
                # ssl_loss = self.train_ssl_one_step(data)
                labelled_loss = self.train_labelled_one_step(data)
                _,meta_loss_ce,ssl_loss = self.fit_meta_loss_step(data,'valid')
                
                # Enable gradients temporarily for gradient similarity calculation
                ssl_loss.backward(retain_graph=True)
                ssl_gradients = get_model_gradients_vector(self.dataAug_gnn).view(1, -1)
                self.optimizer.zero_grad()  # Reset gradients
                
                labelled_loss.backward(retain_graph=True)
                labelled_gradients = get_model_gradients_vector(self.dataAug_gnn).view(1, -1)
                self.optimizer.zero_grad()  # Reset gradients again
                
                meta_loss_ce.backward(retain_graph=True)
                meta_cls_gradients = get_model_gradients_vector(self.dataAug_gnn).view(1, -1)
                self.optimizer.zero_grad()  # Reset gradients again
                v1 = F.cosine_similarity(ssl_gradients, labelled_gradients).item()
                v2 = F.cosine_similarity(meta_cls_gradients,labelled_gradients).item()
                grads_sim_ssl.append(v1)
                grads_sim_meta.append(v2)
                steps += 1
            
            average_grad_sim_ssl = np.mean(np.asarray(grads_sim_ssl))
            average_grad_sim_meta = np.mean(np.asarray(grads_sim_meta))
            self.gradsim_avg_ssl.append(average_grad_sim_ssl)
            self.gradsim_avg_meta.append(average_grad_sim_meta)
            if self.debug:
                print(colored(f'{phase} Phase, Average Average Grad Sim: {average_grad_sim_ssl},{average_grad_sim_meta}', 'blue','on_white'))
            
            self.train()
            return -1.
        else:
            data = data_input
            ssl_loss = self.train_ssl_one_step(data)
            labelled_loss = self.train_labelled_one_step(data)
            
            # Enable gradients temporarily for gradient similarity calculation
            ssl_loss.backward(retain_graph=True)
            ssl_gradients = get_model_gradients_vector(self.dataAug_gnn).view(1, -1)
            self.optimizer.zero_grad()  # Reset gradients
            
            labelled_loss.backward(retain_graph=True)
            labelled_gradients = get_model_gradients_vector(self.dataAug_gnn).view(1, -1)
            self.optimizer.zero_grad()  # Reset gradients again
            grads_sim_score = F.cosine_similarity(ssl_gradients, labelled_gradients).item() 
            print(colored(f'{phase} Phase, Average Average Grad Sim: {average_grad_sim}', 'blue','on_white')) 
            self.train()
            return grads_sim_score  
    
    def calc_best_loss_(self,data,option='ssl'):
        data = data.to(self.device)
        self.eval()
        with torch.no_grad():
            if self.useAutoAug:
                if self.num_samples==1:
                    edge_weight1,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                    edge_weight2,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                else:
                    edge_weight_avg1 = []
                    edge_weight_avg2 = []
                    for _ in range(self.num_samples):
                        edge_weight1,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                        edge_weight_avg1.append(edge_weight1.view(1,-1))
                        edge_weight2,_ = self.learn_edge_weight(data) if self.useAutoAug else None
                        edge_weight_avg2.append(edge_weight2.view(1,-1))
                        
                    edge_weight1 = torch.cat(edge_weight_avg1,dim=0).mean(dim=0)
                    edge_weight2 = torch.cat(edge_weight_avg2,dim=0).mean(dim=0)
                    self.encoder_model.learnable_edge_weight1 = edge_weight1
                    self.encoder_model.learnable_edge_weight2 = edge_weight2
            else:
                edge_weight1 = None
                edge_weight2 = None

            _, _, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
            g1, g2 = [self.ssl_header(g) for g in [g1, g2]]
            ssl_loss = self.contrast_model_non_agg(g1=g1, g2=g2, batch=data.batch).view(-1,1)
            self.train()
            if option=='ssl':
                return ssl_loss.mean().item(),(edge_weight1+edge_weight2)/2.
            else:
                meta_loss = self.meta_loss1(self.meta_loss_head(ssl_loss))
                return meta_loss.mean().item(),(edge_weight1+edge_weight2)/2.
    
    
    
    