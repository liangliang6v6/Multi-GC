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


class Args:
    def __init__(self,
                 reduction_rate,
                 epochs,
                 nlayers,
                 hidden,
                 lr_adj,
                 lr_feat,
                 lr_model,
                 weight_decay,
                 dropout,
                 #beta,
                 ep_ratio,
                 sinkhorn_iter,
                 opt_scale,
                 #ignr_epochs
                 ):
        self.dataset = None
        self.reduction_rate = reduction_rate
        self.lab_rand = 1
        self.dis_metric = 'ours'
        self.epochs = epochs
        self.nlayers = nlayers
        self.hidden = hidden
        self.lr_adj = lr_adj
        self.lr_feat = lr_feat
        self.lr_model = lr_model
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.keep_ratio = 1.0
        #self.beta = beta
        self.ep_ratio = ep_ratio
        self.sinkhorn_iter = sinkhorn_iter
        self.opt_scale = opt_scale
        #self.ignr_epochs = ignr_epochs
        self.debug = 0
        self.option = 0
        self.sgc = 1
        self.inner = 0
        self.outer = 20
        self.save = 1
        self.one_step = 1
        self.mode = 'disabled'
        self.mz_size = None


# Define the trainable function
def trainable(config):
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Setup arguments
    args = Args(
        reduction_rate=config['reduction_rate'],
        epochs=config['epochs'],
        nlayers=config['nlayers'],
        hidden=config['hidden'],
        lr_adj=config['lr_adj'],
        lr_feat=config['lr_feat'],
        lr_model=config['lr_model'],
        weight_decay=config['weight_decay'],
        dropout=config['dropout'],
        #beta=config['beta'],
        ep_ratio=config['ep_ratio'],
        sinkhorn_iter=config['sinkhorn_iter'],
        opt_scale=config['opt_scale'],
        #ignr_epochs=config['ignr_epochs']
    )

    # Load Dataset
    data_full = get_dataset(config['dataset'], True)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    
    if data_full.adj.shape[0] < 5000:
        args.mx_size = data_full.adj.shape[0]
    else:
        args.mx_size = 5000
        data_full.adj_mx = data_full.adj[:args.mx_size, :args.mx_size]
    
    agent = SGDD(data, args, device='cuda')
    agent.train()

# Define the search space
config = {
    'nlayers': tune.randint(2,5),
    'hidden': tune.qrandint(32,128,32),
    'lr_adj': tune.loguniform(1e-4, 1e-2),
    'lr_feat': tune.loguniform(1e-3, 1e-1),
    'lr_model': tune.loguniform(1e-3, 1e-1),
    'weight_decay': tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
    'dropout': tune.quniform(0, 0.5, 0.1),
    'ep_ratio': tune.quniform(0.3, 0.7, 0.1),
    'sinkhorn_iter': tune.randint(3, 7),
    'opt_scale': tune.loguniform(1e-11, 1e-9),
}

# Apply ASHA scheduler
scheduler = ASHAScheduler(
    time_attr='training_iteration',
    max_t=60,
    grace_period=10,
    reduction_factor=2,
    brackets=1,
)

if __name__ == '__main__':
    # Query dataset, reduction_rate, and epochs
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--reduction_rate', type=float)
    parser.add_argument('--epochs', type=int)
    
    # Update into config
    config.update(vars(parser.parse_args()))
    
    # Start the hyperparameter search, using HyperOpt Algorithm
    trainable_with_resources = tune.with_resources(trainable, {"gpu": 2, "cpu": 4})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config = tune.TuneConfig(
            search_alg=HyperOptSearch(),
            metric='f1_micro',
            mode='max',
            num_samples=40,
            scheduler=scheduler,
        ),
        param_space=config,
        #run_config=RunConfig(storage_path="./results", name="GCDM_hyperTune")
        run_config=RunConfig(storage_path="/data/liang/code/GCond/results/SGDD_hyperTune/", name=f"{config['dataset']}_{config['reduction_rate']}")
    )
    results = tuner.fit()
    
    