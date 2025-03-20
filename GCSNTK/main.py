# 1. Imports and device setup (unchanged)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, roc_auc_score
from krr import KernelRidgeRegression
from sntk import StructureBasedNeuralTangentKernel
from LoadData import load_data
from utils import update_E, sub_E
import argparse
import numpy as np
import random
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 2. Argument parser (mostly unchanged, set defaults appropriately)
parser = argparse.ArgumentParser(description='SNTK computation')
parser.add_argument('--dataset', type=str, default="Citeseer", help='dataset name')
parser.add_argument('--cond_ratio', type=float, default=0.5, help='ratio of condensed training set')
parser.add_argument('--ridge', type=float, default=1e-3, help='ridge parameter for KRR')
parser.add_argument('--k', type=int, default=3, help='iteration times for graph convolution')
parser.add_argument('--epochs', type=int, default=120, help='training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--K', type=int, default=2, help='number of aggregations in SNTK')
parser.add_argument('--L', type=int, default=2, help='layers after each aggregation')
parser.add_argument('--scale', type=str, default='average', help='scale type [average, add]')
parser.add_argument('--set_seed', type=bool, default=True, help='whether to set seed')
parser.add_argument('--save', type=bool, default=False, help='whether to save results')
parser.add_argument('--adj', type=bool, default=False, help='condense adjacency matrix or not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--iter', type=int, default=3, help='iteration count for experiments')
args = parser.parse_args()

# 3. Load Multi-label Dataset (customize load_data function accordingly)
root = './datasets/'
adj, x, labels, idx_train, _, idx_test,\
                        x_train, _, x_test,\
                        y_train, _, y_test,\
                        y_train_multi_hot, _, y_test_multi_hot, _ = load_data(root=root, name=args.dataset, k=args.k)

# Labels should be multi-hot encoded [num_samples x num_classes]

n_class    = labels.shape[1] # labels in multi-label: [N x C]
n, dim     = x.shape
n_train    = len(y_train)
Cond_size  = round(n_train * args.cond_ratio)

print(f"Dataset       :{args.dataset}")
print(f"Training Set  :{len(y_train)}")
print(f"Testing Set   :{len(y_test)}")
print(f"Classes       :{n_class}")
print(f"Feature Dim   :{dim}")
print(f"Total Nodes   :{n}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.set_seed:
    setup_seed(args.seed)

E_train = sub_E(idx_train, adj)
E_test  = sub_E(idx_test, adj)

SNTK = StructureBasedNeuralTangentKernel(K=args.K, L=args.L, scale=args.scale).to(device)
ridge = torch.tensor(args.ridge).to(device)
KRR  = KernelRidgeRegression(SNTK.nodes_gram, ridge).to(device)

# 4. Change loss function suitable for multi-label classification
criterion = nn.BCEWithLogitsLoss().to(device)

adj = adj.to(device)
x = x.to(device)
x_train = x_train.to(device)
x_test = x_test.to(device)
E_test = E_test.to(device)
E_train = E_train.to(device)
y_train_multi_hot = y_train_multi_hot.to(device).float()
y_test_multi_hot = y_test_multi_hot.to(device).float()

optimizer = torch.optim.Adam(SNTK.parameters(), lr=args.lr)

# 5. Training loop for multi-label classification
for epoch in range(args.epochs):
    SNTK.train()
    optimizer.zero_grad()

    # compute kernel
    K_train = SNTK(x_train, E_train, adj)

    # Fit the regression model
    y_pred = KRR(K_train, y_train_multi_hot)

    loss = criterion(y_pred, y_train_multi_hot)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{args.epochs} Loss: {loss.item():.4f}")

# 6. Evaluate using multi-label metrics
SNTK.eval()
with torch.no_grad():
    K_test = SNTK(x_test, E_test, adj)
    y_test_pred_logits = KRR(K_test, y_train_multi_hot)
    y_test_pred = torch.sigmoid(y_test_pred_logits)

# Threshold predictions at 0.5
y_test_pred_binary = (y_test_pred.cpu().numpy() > 0.5).astype(int)
y_test_true = y_test_multi_hot.cpu().numpy()

f1_micro = f1_score(y_test_true, y_test_pred_binary, average='micro')
f1_macro = f1_score(y_test_true, y_test_pred_binary, average='macro')
roc_auc_micro = roc_auc_score(y_test_true, y_test_pred.cpu().numpy(), average='micro')

print("Evaluation Metrics (Multi-label):")
print(f"Micro F1 Score : {f1_micro:.4f}")
print(f"Macro F1 Score : {f1_macro:.4f}")
print(f"Micro ROC AUC  : {roc_auc_micro:.4f}")