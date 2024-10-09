import torch
from utils import *
from sgdd_agent import SGDD
import random
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import argparse
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import RunConfig
from tqdm import tqdm
from models.multi_gcn import GCN
from coreset import KCenter, Herding, Random
from utils import match_loss, regularization, row_normalize_tensor,ml_acc
import torch.nn as nn
from ray import train


class Args:
    def __init__(self,
                 reduction_rate,
                 hidden,
                 lr,
                 weight_decay,
                 nlayers,
                 #epochs,
                 inductive,
                 method):
        self.dataset = None # prompt from command line
        self.reduction_rate = reduction_rate
        self.hidden = hidden
        self.keep_ratio = 1.0
        self.lr = lr
        self.weight_decay = weight_decay
        self.nlayers = nlayers
        self.epochs = 400 # prompt from command line
        self.inductive = inductive
        self.save = 0
        self.method = method # prompt from command line
        self.reduction_rate = reduction_rate # prompt from command line
        

def trainable(config):
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Setup Arguments
    args = Args(
        reduction_rate=config['reduction_rate'],
        hidden=config['hidden'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        nlayers=config['nlayers'],
        epochs=config['epochs'],
        inductive=config['inductive'],
        method=config['method']
    )
    
    # Prepare dataset
    data_full = get_dataset(args.dataset, True)
    data = Transd2Ind(data_full, 1.0)
    
    features = data_full.features
    adj = data_full.adj
    labels = data_full.labels
    idx_train = data_full.idx_train
    idx_val = data_full.idx_val
    idx_test = data_full.idx_test
    
    # Setup GCN Model
    model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=data.nclass)
    model = model.to('cuda')
    
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
    #train.report({"f1_micro": res.mean(0)[1]})
    
    return res.mean()


# Define the search space
config = {
    "hidden": tune.qrandint(128, 512, 128),
    "lr": tune.loguniform(1e-3, 1e-1),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "nlayers": tune.randint(2, 5),
    "inductive": tune.choice([0, 1]),
}


if __name__ == "__main__":
    # Query dataset, reduction_rate and method
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--reduction_rate', type=float)
    parser.add_argument('--method', type=str)
    
    # Update config
    config.update(vars(parser.parse_args()))
    
    # Start hyperparameter search
    trainable_with_resources = tune.with_resources(trainable, {"gpu": 1, "cpu": 2})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=HyperOptSearch(),
            metric='f1_micro',
            mode='max',
            num_samples=40,
        ),
        param_space=config,
        run_config=RunConfig(storage_path=f"/data/liang/code/GCond/results/Coreset_{config['method']}_hyperTune/", name=f"{config['dataset']}_{config['reduction_rate']}")
    )
    results = tuner.fit()