# Multi-label-GCond

[Graph Condensation Survey](https://arxiv.org/pdf/2402.02000)


Abstract
----
Given the widespread use of multi-label graphs in real-world scenarios, adapting existing state-of-the-art graph condensation methods for efficient training of neural models on these datasets remains an open challenge. In this work, we extend available graph condensation techniques to multi-label graph node classification by introducing probabilistic label and subgraph initialization methods, along with a generalized multi-label loss function for adaptation. We aim to condense large multi-label graphs into small, synthetic graphs that retain essential information, allowing graph neural networks (GNNs) trained on the condensed graphs to achieve comparable performance to those trained on the original datasets. We evaluate our approach across eight diverse multi-label graph datasets, comparing the statistical properties of the synthetic graphs to their original counterparts. Additionally, we benchmark the performance of these GC methods against coreset selection techniques. Our extensive experiments demonstrate that the proposed adaptations achieve significant reductions in graph size while maintaining competitive classification accuracy, thus paving the way for more efficient learning from large-scale multi-label graph data.

<div align=center><img src="https://github.com/liangliang6v6/Multi-GCond/blob/main/structure.png" width="800"/></div>

## Requirements
```
torch==1.7.0
torch_geometric==1.6.3
scipy==1.6.2
numpy==1.19.2
ogb==1.3.0
tqdm==4.59.0
torch_sparse==0.6.9
deeprobust==0.2.4
scikit_learn==1.0.2
```

## Run the Code
For example, run GCond method with ppi dataset with condensation ratio=0.01:
```
python train_gcond_transduct.py --dataset ppi --nlayers=2 --lr_feat=1e-2 --gpu_id=0  --lr_adj=1e-2 --r=0.01 
```
Similarly, run SGDD, GCDM methods:
```
python train_sgdd.py --dataset ppi --nlayers=2 --lr_feat=1e-2 --gpu_id=0  --lr_adj=1e-2 --r=0.01
python train_gcdm.py --dataset ppi --nlayers=2 --lr_feat=1e-2 --gpu_id=0  --lr_adj=1e-2 --r=0.01 
```

## Coreset Performance
Run the following code to get the coreset performance.
```
python train_coreset.py --dataset ppi --r=0.01  --method=random
python train_coreset.py --dataset ppi --r=0.01  --method=herding
python train_coreset.py --dataset ppi --r=0.01  --method=kcenter
```

## Original Methods
For the complete GC methods, please check the [Awesome-Graph-Condensation](https://github.com/Frostland12138/Awesome-Graph-Condensation).

## Dataset
In our code, datasets name include: `ppi`, `ppi-large`, `yelp`, `dblp`, `pcg`, `hg`, `eg`, `ogbn-pro`.

| Datasets     | #Nodes  | #Edges     | #Features | #Labels | Train/Val/Test     |
| ------------ | ------- | ---------- | --------- | ------- | ------------------ |
| PPI          | 14,755  | 225,270    | 50        | 121     | 0.66 / 0.12 / 0.22 |
| PPI-large    | 56,944  | 818,716    | 50        | 121     | 0.79 / 0.11 / 0.10 |
| Yelp         | 716,847 | 6,977,410  | 300       | 100     | 0.75 / 0.10 / 0.15 |
| DBLP         | 28,702  | 68,335     | 300       | 4       | 0.60 / 0.20 / 0.20 |
| PCG          | 3,000   | 37,000     | 32        | 15      | 0.60 / 0.20 / 0.20 |
| HumanGo      | 3,106   | 18,496     | 32        | 14      | 0.60 / 0.20 / 0.20 |
| EukaryoteGo  | 7,766   | 13,818     | 32        | 22      | 0.60 / 0.20 / 0.20 |
| OGBN-Proteins | 132,000 | 39,000,000 | 8         | 112     | 0.60 / 0.20 / 0.20 |
