import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.datasets import Planetoid
import random

def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
        dataset.multi = False
    elif name in ['ppi','ppi-large','yelp']:
        dataset = torch.load(path+'/data.pt')[0]
        dataset.name = name
        dataset.num_classes = dataset.y.shape[1]
        dataset.multi = True
    elif name in ['dblp','dblp-single']:
        dataset = torch.load('/data/liang/workspace/code/GCond'+'/data/dblp/data.pt')
        dataset.name = name
        dataset.num_classes = dataset.y.shape[1]
        dataset.multi = True
    elif name in ['hg','pcg','eg']:
        #dataset = torch.load('data/'+name+'/data.pt')
        dataset = torch.load("/data/liang/workspace/code/GCond" + '/data/' + name + '/data.pt')
        dataset.name = name
        dataset.num_classes = dataset.y.shape[1]
        dataset.multi = True
    elif name in ['ogbn-pro']:
        data = PygNodePropPredDataset(root='data',name = 'ogbn-proteins',transform=T.ToSparseTensor(attr='edge_attr')) 
        graph = data[0] # pyg graph object
        # Move edge features to node features
        graph.x = graph.adj_t.mean(dim=1)
        graph.adj_t.set_value_(None)
        # split dataset
        dataset =  T.RandomNodeSplit(num_val = 0.2, num_test= 0.2)(graph)
        # Pre-compute GCN normalization.
        adj_t = dataset.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

        dataset.name = name
        dataset.num_classes = dataset.y.shape[1]
        dataset.multi = True

    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dpr_data = Pyg2Dpr(dataset)

    return dpr_data


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        self.num_classes = pyg_data.num_classes
        if not pyg_data.multi:
            pyg_data = pyg_data[0]
        n = pyg_data.num_nodes
        if dataset_name in ['ogbn-pro']:
            row, col, value = pyg_data.adj_t.coo()
            if value is None:
                value = np.ones(row.shape[0])
            else:
                value = value.cpu().numpy()
            row = row.cpu().numpy()
            col = col.cpu().numpy()
            # Get the shape of the adjacency matrix
            n = pyg_data.adj_t.size(0)
            # Create a SciPy sparse CSR matrix
            self.adj = sp.csr_matrix((value, (row, col)), shape=(n, n))
        else:
            self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
                (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        #transfer multi-label into multi-calss format
        if dataset_name in ['dblp-single']:
            multi_class_labels = [random.choice(np.where(label == 1)[0]) for label in self.labels]
            self.labels = np.array(multi_class_labels)
            print('DBLP multi-class format',self.labels)

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape

        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask



class Transd2Ind:
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio):
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        # self.nclass = labels.max()+1
        self.nclass = dpr_data.num_classes
        self.adj_full, self.feat_full, self.labels_full = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1-keep_ratio,
                                            stratify=labels[idx_train])

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        print('size of adj_train:', self.adj_train.shape)
        print('#edges in adj_train:', self.adj_train.sum())

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]


        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=12, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx


        if self.samplers is None:
            self.samplers = []
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
                for i in range(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[i])
                    layer_samplers.append(NeighborSampler(adj,
                                        node_idx=node_idx,
                                        sizes=sizes, batch_size=num,
                                        num_workers=12, return_e_id=False,
                                        num_nodes=adj.size(0),
                                        shuffle=True))
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[args.nlayers-1][c].sample(batch)
        return out



def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def calc_f1(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def evaluate(output, labels, args):
    data_graphsaint = ['yelp', 'ppi', 'ppi-large', 'flickr', 'reddit', 'amazon']
    if args.dataset in data_graphsaint:
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        if len(labels.shape) > 1:
            micro, macro = calc_f1(labels, output, is_sigmoid=True)
        else:
            micro, macro = calc_f1(labels, output, is_sigmoid=False)
        print("Test set results:", "F1-micro= {:.4f}".format(micro),
                "F1-macro= {:.4f}".format(macro))
    else:
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return


from torchvision import datasets, transforms
def get_mnist(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no        augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    labels = []
    feat = []
    for x, y in dst_train:
        feat.append(x.view(1, -1))
        labels.append(y)
    feat = torch.cat(feat, axis=0).numpy()
    from utils_graphsaint import GraphData
    adj = sp.eye(len(feat))
    idx = np.arange(len(feat))
    dpr_data = GraphData(adj-adj, feat, labels, idx, idx, idx)
    from deeprobust.graph.data import Dpr2Pyg
    return Dpr2Pyg(dpr_data)

def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss

def maxdegree(adj):
    n = adj.shape[0]
    return F.relu(max(adj.sum(1))/n - 0.5)

def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro

def sparsity(adj):
    n = adj.shape[0]
    thresh = n * n * 0.01
    return F.relu(adj.sum()-thresh)
    # return F.relu(adj.sum()-thresh) / n**2

def feature_smoothing(adj, X):
    adj = (adj.t() + adj)/2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-8
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat

def row_normalize_tensor(mx):
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx

# multi-label acc
def ml_acc(output, labels, threshold=0.5):
    binary_pred = (torch.sigmoid(output) >= threshold).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # print('true sum:\n',np.sum(labels,axis=0))
    # print('pred sum:\n',np.sum(binary_pred,axis=0))
    f1_micro = metrics.f1_score(y_pred=binary_pred,y_true = labels,average='micro', zero_division=0)
    f1_macro = metrics.f1_score(y_pred=binary_pred,y_true = labels,average='macro', zero_division=0)
    f1_weight = metrics.f1_score(y_pred=binary_pred,y_true = labels,average='weighted', zero_division=0)
    # print("!!!Test set results:", "F1-micro= {:.4f}".format(f1_micro))
    # if 0 == f1_micro:
    #     print('true sum:\n',np.sum(labels,axis=0))
    #     print('pred sum:\n',np.sum(binary_pred,axis=0))
    return f1_micro,f1_macro,f1_weight

import matplotlib.pyplot as plt
import seaborn as sns
# plt.rc('font',family='Times New Roman')
def syn_label_corr(labels, args, dataset):
    # co-ocurrence count matrix M
    M = torch.matmul(labels.t(), labels)
    # total occurrenvr of each class
    N = torch.sum(labels, dim=0)
    # conditional probability matrix P
    P = M.float() / N.float().view(-1, 1)
    # Create diagonal matrix D
    # D = torch.diag(torch.diag(P))
    # the laplacian matrix L_o
    # L_o = D - P
    # sns.set()
    # plt.figure()
    # sns.heatmap(P.cpu().numpy(), annot=True, cmap="coolwarm", fmt=".2f", xticklabels=labels.shape[1], yticklabels=labels.shape[1])
    # plt.title(dataset_str+" Labels Correlation")
    plt.imshow(P.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
    # values
    if args.dataset in ['dblp','DBLP']:
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                plt.text(j, i, f'{P[i, j]:.3f}', ha='center', va='center', color='white')

    plt.colorbar() 
    plt.title(f'{dataset} Multi-Label Correlations')
    plt.xlabel('Labels')
    plt.ylabel('Labels')
    plt.savefig('./images/' + dataset +'labels'+ '.svg', format='svg', dpi = 600)
    plt.close()
    print(dataset+'Image saved')

def syn_label_distribution(labels, dataset_str):
    class_distribution = {}
    for node_idx, node_labels in enumerate(labels):
        active_classes = torch.nonzero(node_labels, as_tuple=True)[0].tolist()

        for class_idx in active_classes:
            if class_idx not in class_distribution:
                class_distribution[class_idx] = 1
            else:
                class_distribution[class_idx] += 1

    class_indices, counts = zip(*class_distribution.items())

    plt.bar(class_indices, counts)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Nodes')
    plt.title( dataset_str + 'Multi-Label Class Distribution')
 
    plt.savefig('./images/' + dataset_str + '.svg', format='svg', dpi = 600)
    plt.close()
    print(dataset_str+'Image saved')
