from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from utils_graphsaint import DataGraphSAINT
from torch_geometric.datasets import Yelp
import os
from GCSNTK.krr import KernelRidgeRegression
from GCSNTK.sntk import StructureBasedNeuralTangentKernel
from GCSNTK.utils import update_E, sub_E
import torch, gc
from utils import ml_acc
import deeprobust.graph.utils as utils
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
class SNTK:
    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        labels = torch.tensor(self.data.labels_train, dtype=torch.float32, device=device)
        n = int(data.feat_train.shape[0] * args.cond_ratio)
        d = data.feat_train.shape[1]
        
        self.nnodes_syn = n

        self.labels_syn = self.generate_labels_syn(n).to(device)
        self.feat_syn = nn.Parameter(self.generate_feat_syn(self.labels_syn, n).to(device))

        SNTK = StructureBasedNeuralTangentKernel(K=args.K, L=args.L, scale=args.scale).to(device)
        ridge = torch.tensor(args.ridge, device=device)
        self.KRR = KernelRidgeRegression(SNTK.nodes_gram, ridge).to(device)

        if args.loss == "BCE":
            print("@BCE")
            self.criterion = nn.BCEWithLogitsLoss().to(device)
        elif args.loss == "SML":
            print("@SML")
            self.criterion = nn.MultiLabelSoftMarginLoss().to(device)
        else:
            print("@not implement")

        # self.feat_syn.requires_grad = True
        self.optimizer = torch.optim.Adam([self.feat_syn], lr=args.lr)

    def generate_labels_syn(self, n):
        labels_train = self.data.labels_train
        print('@Probability synthetic labels for init')
        num, class_num = labels_train.shape
        p_labels = np.sum(labels_train, axis=0) / num
        syn_labels = np.zeros((n, class_num), dtype=float)

        for cls in range(class_num):
            set_one = int(n * p_labels[cls])
            indices = np.random.choice(n, set_one, replace=False)
            syn_labels[indices, cls] = 1

        return torch.tensor(syn_labels, device=self.device)
    
    def label_sim(self, syn_label):
        labels = self.data.labels_train
        sim = np.dot(labels, syn_label)
        sim_idx = np.argmax(sim)
        # print('sim:',syn_label, labels[sim_idx], sim_idx)
        return sim_idx
    
    def generate_feat_syn(self, labels_syn, n):
        labels_syn = labels_syn.cpu().numpy()  # Move to CPU for numpy operations
        feat_train = self.data.feat_train
        select_idx = []

        for i in range(n):
            label = labels_syn[i]
            idx = self.label_sim(label)
            select_idx.append(idx)

        syn_feat = feat_train[select_idx]
        print('@Similar feature init based on syn_labels')
        return torch.tensor(syn_feat, dtype=torch.float32, device=self.device)

    def train(self, E_s):
        device = self.device
        data = self.data

        KRR = self.KRR
        optimizer = self.optimizer

        G_t, E_t, y_t = data.feat_train, data.adj_train, data.labels_train
        G_t, E_t, y_t = utils.to_tensor(G_t, E_t, y_t, device=device)
        G_s, y_s = self.feat_syn.to(device), self.labels_syn.to(device)

        criterion = self.criterion
        pred, f1_micro, f1_macro, f1_weight = KRR.forward(G_t, G_s, y_t, y_s, E_t, E_s.to(device))

        loss = nn.BCEWithLogitsLoss()(pred, y_t.float())
        loss = loss.to(torch.float32)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()

        print(f"Training loss: {loss:>7f} Training F1-micro: {f1_micro:>7f}", end=' ')
        return G_s, y_s, loss, f1_micro * 100

    def test(self, G_s, y_s, E_s):
        device = self.device
        data = self.data
        G_t, E_t, y_t = data.feat_test, data.adj_test, data.labels_test
        G_t, E_t, y_t = utils.to_tensor(G_t, E_t, y_t, device=device)

        KRR = self.KRR
        criterion = self.criterion

        with torch.no_grad():
            pred, _ = KRR.forward(G_t, G_s.to(device), y_t, y_s.to(device), E_t, E_s.to(device))
            test_loss = nn.BCEWithLogitsLoss()(pred, y_t.float()).item()
            
            f1_micro, f1_macro, f1_weight = ml_acc(pred, y_t)

        print(f"Avg loss: {test_loss:.6f}, F1 Micro: {f1_micro:.2f}, F1 Macro: {f1_macro:.2f}, F1 Weighted: {f1_weight:.2f}")
        return test_loss, f1_micro

    def main(self):
        args = self.args
        device = self.device
        G_s, y_s = self.feat_syn, self.labels_syn

        n = len(y_s)
        E_s = torch.zeros((n, n), device=device)
    
        Time = torch.zeros(args.epochs, args.iter, device=device)
        Acc = torch.zeros(args.epochs, args.iter, device=device)

        for iter in range(args.iter):
            print('--------------------------------------------------')
            print(f'The {iter+1}th Iteration:')
            print('--------------------------------------------------')

            T = 0
            for epoch in range(args.epochs):
                a = time.time()
                print(f"Epoch {epoch+1}", end=" ")
                x_s, y_s, training_loss, training_acc = self.train(E_s)
                T += time.time() - a
                Time[epoch, iter] = T

                test_loss, test_acc = self.test(x_s, y_s, E_s)
                Acc[epoch, iter] = test_acc

        Acc_mean, Acc_std = torch.mean(Acc, dim=1), torch.std(Acc, dim=1)

        Time_mean = torch.mean(Time, dim=1)
        Time_Acc = torch.cat((Time_mean.reshape(-1, 1), Acc_mean.reshape(-1, 1)), dim=1)
        print(np.array(Time_Acc.cpu()))

        print(f'Mean and std of test data: {Acc_mean[-1]:.4f}, {Acc_std[-1]:.4f}')
        print("--------------- Train Done! ----------------")