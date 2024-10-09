from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from gcdm_agent import GCDM
from utils_graphsaint import DataGraphSAINT
from torch_geometric.datasets import Yelp
import os

import torch, gc

torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='ppi')
# condensation
parser.add_argument('--reduction_rate', type=float, default=0.01)

parser.add_argument('--loss', type=str, default='BCE') # BCE OR OTHERS
parser.add_argument('--lab_prob', type=int, default=0)
parser.add_argument('--subgraph', type=int, default=1)
parser.add_argument('--method', type=str, default="random") #choices=['kcenter', 'herding', 'random']

parser.add_argument('--lr_adj', type=float, default=1e-2)
parser.add_argument('--lr_feat', type=float, default=1e-2)
#encoder
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--hop', type=int, default=1)
parser.add_argument('--activation', type=bool, default=True)
# eval model
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
# dataset
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)

parser.add_argument('--save', type=int, default=0)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

data_full = get_dataset(args.dataset, args.normalize_features)
data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
agent = GCDM(data, args, device='cuda')
agent.train()
