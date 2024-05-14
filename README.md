# AuDiMe_graphOOD
This repository contains codes for data preparation, baselines and our proposed method. `ood-baselines/` contains codes for all the baselines excluding iMoLD. `iMoLD/` contains the codes for the baseline `iMoLD`. `AuDiMe/` contains the implementation of the proposed framework. For all three folders, the data preparation steps are the same.

## Data Preparation

### SPMotif

Go to one of the subdir, e.g., `ood-baselines/`, then `cd dataset_gen/`, change the `global_b` parameter in `gen_struc.py` to get various `bias` for SPMotif datasets. Then run:

```bash
python gen_struc.py
```

The data will be saved to `XXX/data/`, e.g., `ood-baselines/data/`. 


### DrugOOD

To use DrugOOD datasets, curate the datasets according to [DrugOOD Benchmark repo](https://github.com/tencent-ailab/DrugOOD) based on commit `eeb00b8da7646e1947ca7aec93041052a48bd45e`, After curating the datasets, put the corresponding json files under `XXX/data/DrugOOD/`. 


### Graph-SST

Get the datasets from [this url](https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z), then unzip each dataset to `XXX/data/`.


