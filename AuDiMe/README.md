# How to run the codes

In this dir, we have provided code implementing our framework `AuDi`, the code for `AuDiMe` (with meta-learning components) is masked, and will be released if our paper get accepted. Here is a example on how to run our code:

```bash
python run_dataAug.py --base_gnn gin --early_stop_epochs 10  --nhid 32 --epochs 50 --pretraining_epochs 10 --dataset drugood_lbap_core_ec50_scaffold --edge_dim 10 --device 0 --nlayers 4 --edge_gnn_layers 2 --edge_gnn gin --edge_uniform_penalty 0.01 --edge_prob_thres 50 --edge_budget 0.75 --edge_penalty 10.0 --penalty 0.001  --gradMatching_penalty 0.0  --seed 1 --fname_str erm_ssl_autoaug_pe_10_es_10 --useAutoAug
```


`base_gnn` is the GNN encoder used for `h()`. For `t()`, the GNN encoder `edge_gnn` is default to be 2-layer GIN. `nhid` is the hidden dimensions. `edge_dim` is the edge feature dimension, which can be -1 if not using edge features. `edge_uniform_penalty` is regularization coefficient for $\mathcal{L}_{div}$, `edge_prob_thres` is the lowest K\% edges. `edge_budget` is the $\eta$ for the edge count budgets. `edge_penalty` is the regularization coefficient for $\mathcal{L}_e$. `penalty` is the regularization coefficient for SSL loss. `useAutoAug` is to enable the learnable data transformation. Make sure in `fname_str`, it contains `erm_ssl` key words, and the results will be saved to `exp_results/` dir in numpy format.
