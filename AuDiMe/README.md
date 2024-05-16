In this dir, we have provided code implementing our framework `AuDi`, the code for `AuDiMe` (with meta-learning components) is masked, and will be released if our paper get accepted. Here is a example on how to run our code:

```bash
python run_dataAug.py --base_gnn gin --early_stop_epochs 10  --nhid 32 --epochs 50 --pretraining_epochs 10 --dataset drugood_lbap_core_ec50_scaffold --edge_dim 10 --device 0 --nlayers 4 --edge_gnn_layers 2 --edge_gnn gin --edge_uniform_penalty 0.01 --edge_prob_thres 50 --edge_budget 0.75 --edge_penalty 10.0 --penalty 0.001  --gradMatching_penalty 0.0  --seed 1 --fname_str erm_ssl_autoaug_pe_10_es_10 --useAutoAug
```
