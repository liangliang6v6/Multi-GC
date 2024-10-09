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
from models.encoder import Encoder
from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
from tqdm import trange
from ray import tune
from ray import train
from coreset import KCenter, Herding, Random

class GCDM:

    def __init__(self, data, args, device='cuda', **kwargs):
        torch.autograd.set_detect_anomaly(True)
        self.data = data
        self.args = args
        self.device = device

        n = int(data.feat_train.shape[0] * args.reduction_rate) 
        d = data.feat_train.shape[1]
        self.nnodes_syn = n

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
        if args.loss=="BCE":
            print("@BCELOSS")
        else:
            print("@SoftMarginLoss")
        
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
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
        data, args = self.data, self.args
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn.detach()

        adj_syn = pge.inference(feat_syn)
        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        model = GCN(nfeat=feat_syn.shape[1], nhid=args.hidden, dropout=args.dropout,
            weight_decay=args.weight_decay, nlayers=args.nlayers,
            nclass=data.nclass, device=self.device).to(self.device)
        
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
                     train_iters=600, normalize=False, verbose=False)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        
        output = model.predict(data.feat_train, data.adj_train)
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
        loss_test = nn.BCEWithLogitsLoss()(output[data.idx_test], labels_test.float())
        f1_micro_test,f1_macro,f1_weight = ml_acc(output[data.idx_test], labels_test)
        
        res.append(f1_micro_test.item())
        if verbose:
            print("Test results:",
                  "loss= {:.4f}".format(loss_test),
                  "F1-micro= {:.4f}".format(f1_micro_test),
                  "F1-macro= {:.4f}".format(f1_macro),
                  "F1-weighted= {:.4f}".format(f1_weight)
                  )
        res.append(f1_macro.item())
        res.append(f1_weight.item())
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn

        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()
        
        # loss_avg = 0
        encoder = Encoder(data.feat_train.shape[1], args.hid_dim, args.emb_dim, args.n_layers, args.hop, args.activation).to(self.device)

        for it in trange(args.epochs+1):
            encoder.initialize()
            with torch.no_grad():
                emb_real = encoder.encode(features.to(self.device), adj.to(self.device))
                emb_real = F.normalize(emb_real)
            emb_cond = encoder.encode_without_e(feat_syn.to(self.device))
            emb_cond = F.normalize(emb_cond)
            
            loss = torch.tensor(0.).to(self.device)
            
            # for multi-label nodes
            dist = torch.mean(emb_real, 0) - torch.mean(emb_cond, 0)
            loss += torch.sum(dist ** 2)

            self.optimizer_feat.zero_grad()
            loss.backward()
            self.optimizer_feat.step()
            
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
                
                # Update the cross validation result
                train.report({"f1_micro": res.mean(0)[1],
                              "f1_macro": res.mean(0)[2],
                              "f1_weight": res.mean(0)[3]})
                
        return res.mean(0)[1]
    
