import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor, ml_acc
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.multi_gcn import GCN
from models.multi_sgc import SGC

from models.igrn import GraphonLearner as IGNR
import scipy.sparse as sp
from torch_sparse import SparseTensor
from tqdm import trange
from ray import train
from coreset import KCenter, Herding, Random

class SGDD:

    def __init__(self, data, args, device='cuda', **kwargs):
        torch.autograd.set_detect_anomaly(True)
        self.data = data
        self.args = args
        self.device = device

        n = int(data.feat_train.shape[0] * args.reduction_rate) 
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
    
        self.IGNR = IGNR(node_feature=d, nfeat=128, nnodes=n, device=device, args=args).to(device)
        self.graphon = 1

        if self.args.lab_prob:
            self.labels_syn = self.generate_labels_syn(n).to(device)
            self.feat_syn = nn.Parameter(self.generate_feat_syn(self.labels_syn, n).to(device))

        elif self.args.subgraph:
            feat_syn, labels_syn = self.coreset_init()
            self.labels_syn = labels_syn.to(device)
            self.feat_syn = nn.Parameter(feat_syn.to(device))
        else:
            print("@random subgraph")
            sub_nodes = np.random.choice(data.feat_train.shape[0], n, replace=False)
            # print('subgraph index',sub_nodes)
            self.labels_syn = torch.FloatTensor(data.labels_train[sub_nodes]).to(device)
            self.feat_syn = nn.Parameter(torch.FloatTensor(data.feat_train[sub_nodes]).to(device))


        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_IGNR = torch.optim.Adam(self.IGNR.parameters(), lr=args.lr_adj)
        
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

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
        # print('true_label:\n',np.sum(labels_train, axis=0))
        # print('syn_label:\n',np.sum(syn_labels, axis=0))
        
        syn_labels = torch.tensor(syn_labels)
        return syn_labels
    
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

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, IGNR, labels_syn = self.feat_syn.detach(), \
                                self.IGNR, self.labels_syn

        
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=data.nclass, device=device).to(device)

        adj_syn = IGNR.inference(feat_syn)
        args = self.args
        
        import os
        if not os.path.exists('saved_ours'):
            os.makedirs('saved_ours')
        # if self.args.save:
        #     torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
        #     torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
        #     torch.save(labels_syn, f'saved_ours/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))
    
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=2000, normalize=False, verbose=False)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        
        output = model.predict(data.feat_train, data.adj_train)
        # loss = nn.MultiLabelSoftMarginLoss() nn.BCEWithLogitsLoss()
        if self.args.loss == "BCE":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MultiLabelSoftMarginLoss()
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
        feat_syn, IGNR, labels_syn = self.feat_syn, self.IGNR, self.labels_syn
        
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()
        print('adj:',adj,'size:',adj.size(0))
        outer_loop, inner_loop = get_loops(args)
        # loss_avg = 0
        for it in trange(args.epochs+1):
            if args.sgc == 1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)


            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()
            # loss = nn.MultiLabelSoftMarginLoss() nn.BCEWithLogitsLoss()
            if self.args.loss == "BCE":
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.MultiLabelSoftMarginLoss()

            for ol in range(outer_loop):
                if adj.size(0) > 5000:
                    random_nodes = np.random.choice(list(range(adj.size(0))), 5000, replace=False)
                    # sub_adj = adj[random_nodes].to_dense()[:, random_nodes]
                    random_nodes_tensor = torch.tensor(random_nodes)
                    dense_adj = adj.to_dense()
                    sub_adj = dense_adj[random_nodes_tensor][:, random_nodes_tensor]

                    adj_syn, opt_loss = IGNR(self.feat_syn, Lx=sub_adj)
                else:
                    adj_syn, opt_loss = IGNR(self.feat_syn, Lx=adj)
                
                
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
                print('epoch:',it,' loss:',loss)
                
                # if args.beta > 0:
                #     loss_reg = args.beta * regularization(adj_syn, utils.tensor2onehot(labels_syn).to(adj_syn.device))
                # else:
                #     loss_reg = torch.tensor(0)

                # loss = loss + loss_reg
         
                self.optimizer_feat.zero_grad()
                self.optimizer_IGNR.zero_grad()
                
                if it % 50 < 10:
                    loss = loss + args.opt_scale* opt_loss
                    loss.backward()
                    self.optimizer_IGNR.step()
                else:
                    loss.backward()
                    self.optimizer_feat.step()

                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = IGNR.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = criterion(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    
                    optimizer_model.step() 

            if it % 50 == 0:
                print('Epoch {}, loss: {}'.format(it, loss))

            eval_epochs = list(range(0, 5000, 50))
            if verbose and it in eval_epochs:
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv'] else 3
                
                for i in range(runs):
                    res.append(self.test_with_val())

                res = np.array(res)
                print('Train/Test Mean Accuracy:',
                        repr([res.mean(0), res.std(0)]))
                
                # Update the cross validation result in ray train report
                train.report({"f1_micro": res.mean(0)[1]})

def get_loops(args):
    if args.one_step:
        if args.dataset =='ogbn-arxiv':
            return 5, 0
        return 1, 0
    if args.dataset in ['ogbn-arxiv']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 15 
    else:
        return 20, 10

