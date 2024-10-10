import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor,ml_acc,syn_label_distribution,syn_label_corr
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.multi_gcn import GCN
from models.multi_sgc import SGC
from models.mlp import MLP

from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
import random
from sklearn.metrics import accuracy_score
from coreset import KCenter, Herding, Random

class MGCond:
    def __init__(self, data, args, device='cuda', **kwargs):
        # torch.autograd.set_detect_anomaly(True)
        self.data = data
        self.args = args
        self.device = device

        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        # self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        
        self.pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)

        if self.args.subgraph:
            feat_syn, labels_syn = self.coreset_init()
            self.labels_syn = labels_syn.to(device)
            self.feat_syn = nn.Parameter(feat_syn.to(device))
        else:
            self.labels_syn = self.generate_labels_syn(n).to(device)
            self.feat_syn = nn.Parameter(self.generate_feat_syn(self.labels_syn, n).to(device))

        if args.loss=="BCE+":
            print("@BCELOSS+Coefficience")
        elif args.loss=="BCE":
            print("@BCE")
        else:
            print("@SoftMarginLoss")
        
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)
        # print('@original syn feature', self.feat_syn[0])
    
    def coreset_init(self):
        data = self.data
        args = self.args
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        idx_train, idx_val, idx_test  = data.idx_train, data.idx_val, data.idx_test

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
        print('@'+args.method+'init')
        idx_selected = agent.select(embeds)
        feat_train = features[idx_selected]
        feat_syn = torch.FloatTensor(feat_train)

        # adj_syn = [np.ix_(idx_selected, idx_selected)]

        labels_syn = torch.tensor(labels[idx_selected])
        return feat_syn, labels_syn

    def generate_labels_syn(self, n):
        labels_train = self.data.labels_train
        print('@prabability synthetic labels for init')
        num, class_num = labels_train.shape
        p_labels = np.sum(labels_train, axis=0) / num
        syn_labels = np.zeros((n, class_num), dtype=float)
        for cls in range(class_num):
            set_one = int(n * p_labels[cls])
            indices = np.random.choice(n, set_one ,replace=False)
            syn_labels[indices, cls] = 1

        return torch.tensor(syn_labels)
    
    def label_sim(self, syn_label):
        labels = self.data.labels_train
        sim = np.dot(labels, syn_label)
        sim_idx = np.argmax(sim)
        # print('sim:',syn_label, labels[sim_idx], sim_idx)
        return sim_idx
    def generate_feat_syn(self, labels_syn, n):
        labels_syn = labels_syn.detach().cpu().numpy()
        feat_train = self.data.feat_train
        # labels_train = self.data.labels_train
        select_idx = []
        syn_feat = []
        for i in range(n):
            label = labels_syn[i]
            idx = self.label_sim(label)
            select_idx.append(idx)
        # print('select the index:',select_idx)
        syn_feat = feat_train[select_idx]
        print('@Similar feature init based on syn_labels')
        return torch.tensor(syn_feat, dtype=torch.float32)
        
    def label_correlations_matrix(self):
        labels = torch.tensor(self.data.labels_train, dtype=torch.float32)
        M = torch.matmul(labels.t(), labels)
        N = torch.sum(labels, dim=0)
        N = len(labels)
        P = M / N
        # print('P=M/n',P)
        # P = M / N.view(-1, 1)
        # symmetric P
        # P = (P + P.t())/2
        # print('P=(P+P.t())/2',P)
        D = torch.diag(torch.sum(P, dim = 0))
        L_o = D - P
        self.L_o = L_o

    def corr_loss(self):
        device = self.device
        Y = self.labels_syn.to(device)
        L_o = self.L_o.to(device)
        loss_matrix = torch.matmul(torch.matmul(Y.float(), L_o.float()), Y.t().float())
        label_loss = torch.trace(loss_matrix)
        return label_loss

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn.detach()
        
        '''fine tune the learning model'''
        # if self.args.dataset in ['yelp']:
        #     # print('GCN for yelp')
        #     model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0,
        #                 weight_decay=0, nlayers=2,
        #                 nclass=data.nclass, device=device).to(device)
        # else:
        if 'mlp' == self.args.test_model:
            model = MLP(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0,
                    weight_decay=1e-5, nlayers=2,
                    nclass=data.nclass, device=device).to(device)
        else:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0,
                        weight_decay=0, nlayers=2,
                        nclass=data.nclass, device=device).to(device)
        # print('@test model', model)

        adj_syn = pge.inference(feat_syn)

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=200, normalize=True, verbose=False)
        model.eval()
        labels_train = torch.LongTensor(data.labels_train).to(device)
        labels_test = torch.LongTensor(data.labels_test).to(device)
                        # Calculate the coefficience
        class_counts = labels_train.sum(dim=0)
        epsilon = 1e-6
        class_weights = 1.0 / (class_counts + epsilon)
        class_weights = class_weights / class_weights.sum() * len(class_weights)

        # loss = nn.MultiLabelSoftMarginLoss() nn.BCEWithLogitsLoss()
        if self.args.loss == "BCE":
            criterion = nn.BCEWithLogitsLoss()
        elif self.args.loss == "BCE+":
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = nn.MultiLabelSoftMarginLoss()

        output = model.predict(data.feat_train, data.adj_train)
        loss_train = criterion(output, labels_train.float())
        f1_micro,f1_macro,f1_weight = ml_acc(output, labels_train)
        
        if verbose:
            print("Train results:",
                  "loss= {:.4f}".format(loss_train),
                  "F1-micro= {:.4f}".format(f1_micro),
                  "F1-macro= {:.4f}".format(f1_macro),
                  "F1-weighted= {:.4f}".format(f1_weight)
                  )
        res.append(f1_micro.item())

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        loss_test = criterion(output[data.idx_test], labels_test.float())
        f1_micro_test,f1_macro,f1_weight = ml_acc(output[data.idx_test], labels_test)
        
        res.append(f1_micro_test.item())
        if verbose:
            print("Test results:",
                  "loss= {:.4f}".format(loss_test),
                  "F1-micro= {:.4f}".format(f1_micro_test),
                  "F1-macro= {:.4f}".format(f1_macro),
                  "F1-weighted= {:.4f}".format(f1_weight)
                  )
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        # Calculate the coefficience
        class_counts = labels.sum(dim=0)
        epsilon = 1e-6
        class_weights = 1.0 / (class_counts + epsilon)
        class_weights = class_weights / class_weights.sum() * len(class_weights)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()
        # loss = nn.MultiLabelSoftMarginLoss() nn.BCEWithLogitsLoss()
        if self.args.loss == "BCE":
            criterion = nn.BCEWithLogitsLoss()
        elif self.args.loss == "BCE+":
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = nn.MultiLabelSoftMarginLoss()
        outer_loop, inner_loop = get_loops(args)
        # loss_avg = 0
        for it in range(args.epochs+1):
            if 0==it:
                res = []
                runs = 3
                for i in range(runs):
                    res.append(self.test_with_val())
                res = np.array(res)
                print('Train/Test Mean Accuracy:',
                        repr([res.mean(0), res.std(0)]))
                
            if 'sgc' == args.c_model:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif 'mlp' == args.c_model:
                model = MLP(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)
            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)                
            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()


            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)

                loss = torch.tensor(0.0).to(self.device)
                # train model with original data
                output = model.forward(features, adj_norm)
                loss_real = criterion(output, labels.float())
                gw_real = torch.autograd.grad(loss_real, model_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                # train model with syntehtic data
                output_syn = model.forward(feat_syn, adj_syn_norm)
                loss_syn = criterion(output_syn, labels_syn.float())
                gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                # matching the gradient
                loss = match_loss(gw_syn, gw_real, args, device=self.device)
   
                '''update features and adj'''
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                # if args.lab_up:
                #     self.optimizer_label.zero_grad()
                #     '''update labels with label correlation loss'''
                #     if args.lcorr:
                #         label_loss = self.corr_loss()
                #         # print('gradient loss', loss,' label_corr loss',label_loss)
                #         loss += args.loss_lab * label_loss
                # loss.backward(retain_graph=True)
                loss.backward()
                # if it % 2 == 0:
                #     self.optimizer_label.step()
                #     self.labels_syn.data.clamp_(min=0, max=1)
                #     print('updated labels:\n', torch.sum(labels_syn, dim=0))
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()
                # '''update labels'''
                # if args.lab_up and it % args.lab_step == 0:
                #     self.optimizer_label.step()
                #     self.labels_syn.data.clamp_(min=0, max=1)

                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = feat_syn.detach()
                labels_syn_inner = labels_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                
                # divide zero warning
                # adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)

                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner, adj_syn_inner)
                    loss_syn_inner = criterion(output_syn_inner, labels_syn_inner.float())
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step() # update gnn param
            if it % 50 == 0:
                print('Epoch {}, loss: {}'.format(it, loss))
            if self.args.dataset in ['yelp'] and it % 10 == 0:
                print('Epoch {}, loss: {}'.format(it, loss))
            # if args.lcorr:
            #     print('Lcorr:',label_loss)

            eval_epochs = list(range(0, 5000, 50))
            if it==args.epochs:
                print('@final syn_labels:\n', torch.sum(labels_syn, dim=0))
                # print('@final syn feature', self.feat_syn[0])
                # get the label correlations
                # syn_label_corr(labels_syn, args)
            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 50 == 0:
                res = []
                runs = 3
                for i in range(runs):
                    res.append(self.test_with_val())
                res = np.array(res)
                print('Train/Test Mean Accuracy:',
                        repr([res.mean(0), res.std(0)]))

def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        print('@one-step loop')
        return 1, 0
    else:
        return 10, 5

