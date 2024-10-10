from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
import sys
from deeprobust.graph.utils import *
import torch.nn.functional as F
from configs import load_config
from utils import *
from utils_graphsaint import DataGraphSAINT
from models.multi_gcn import GCN
from coreset import KCenter, Herding, Random
from tqdm import tqdm
from utils import match_loss, regularization, row_normalize_tensor,ml_acc
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=7, help='gpu id')
parser.add_argument('--dataset', type=str, default='pcg')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=22, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--inductive', type=int, default=1)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--method', type=str, choices=['kcenter', 'herding', 'random'])
parser.add_argument('--reduction_rate', type=float, default=0.04)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
args = load_config(args)
print(args)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


data_full = get_dataset(args.dataset, args.normalize_features)
data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

features = data_full.features
adj = data_full.adj
labels = data_full.labels
idx_train = data_full.idx_train
idx_val = data_full.idx_val
idx_test = data_full.idx_test

# Setup GCN Model
device = 'cuda'
model = GCN(nfeat=features.shape[1], nhid=256, nclass=data.nclass, device=device, weight_decay=args.weight_decay)

model = model.to(device)
model.fit(features, adj, labels, idx_train, idx_val, train_iters=600, verbose=False)

model.eval()
model.test(idx_test)

embeds = model.predict().detach()

if args.method == 'kcenter':
    agent = KCenter(data, args, device='cuda')
if args.method == 'herding':
    agent = Herding(data, args, device='cuda')
if args.method == 'random':
    agent = Random(data, args, device='cuda')

idx_selected = agent.select(embeds)


feat_train = features[idx_selected]
adj_train = adj[np.ix_(idx_selected, idx_selected)]
labels_train = labels[idx_selected]

if args.save:
    np.save(f'saved/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy', idx_selected)


res = []
runs = 10
for _ in tqdm(range(runs)):
    model.initialize()
    model.fit_with_val(feat_train, adj_train, labels_train, data,
                 train_iters=600, normalize=True, verbose=False)

    model.eval()
    labels_test = torch.LongTensor(data.labels_test).cuda()

    # Full graph
    output = model.predict(data.feat_full, data.adj_full)
    loss_test = nn.BCEWithLogitsLoss()(output[data.idx_test], labels_test.float())
    f1_micro_test,f1_macro,f1_weight = ml_acc(output[data.idx_test], labels_test)
    
    print("Test results:",
            "loss= {:.4f}".format(loss_test),
            "F1-micro= {:.4f}".format(f1_micro_test),
            "F1-macro= {:.4f}".format(f1_macro),
            "F1-weighted= {:.4f}".format(f1_weight)
            )
    res.append(f1_micro_test.item())

res = np.array(res)
print('Mean F1 score:', repr([res.mean(), res.std()]))
