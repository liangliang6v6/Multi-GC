from deeprobust.graph.data import Dataset
from sntk_agent import SNTK
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from utils_graphsaint import DataGraphSAINT
import os
from GCSNTK.krr import KernelRidgeRegression
from GCSNTK.sntk import StructureBasedNeuralTangentKernel
from GCSNTK.utils import update_E, sub_E


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="ppi", help='name of dataset [Cora, Citeseer, Pubmed, Photo, Computers] (default: Cora)')
parser.add_argument('--loss', type=str, default='BCE') # SML
parser.add_argument('--subgraph', type=int, default=0)
parser.add_argument('--cond_ratio', type=float, default=0.001, help='condensed ratio of the training set (default: 0.5, the condened set is 0.5*training set)')
parser.add_argument('--ridge', type=float, default=1e-3, help='ridge parameter of KRR (default: 1e-4)')
parser.add_argument('--k', type=int, default=1, help='the iteration times of the Graph Convolution when loading data (default: 3)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.005)')
parser.add_argument('--K', type=int, default=1, help='number of aggr in SNTK (default: 2)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 2)')
parser.add_argument('--scale', type=str, default='average', help='scale of SNTK [average,add] (default: average)')
parser.add_argument('--set_seed', type=bool, default=True, help='setup the random seed (default: True)')
parser.add_argument('--save', type=bool, default=False, help='save the results (default: False)')
parser.add_argument('--adj', type=bool, default=False, help='condese adj or not (default: False)')
parser.add_argument('--seed', type=int, default=5, help='setup the random seed (default: 5)')
parser.add_argument('--iter', type=int, default=3, help='iteration times of the experiments (default: 5)')
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)

args = parser.parse_args()


# random seed setting
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    setup_seed(args.seed)

print(args)

data_full = get_dataset(args.dataset, args.normalize_features)
data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
agent = SNTK(data, args, device='cuda')
agent.main()
