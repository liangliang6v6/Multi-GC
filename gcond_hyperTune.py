import torch
from utils import *
from gcond_agent_transduct import MGCond
import random
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import argparse
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import RunConfig
import os


class Args:
    def __init__(self,  
                 reduction_rate, 
                 lr_adj, 
                 lr_feat,
                 epochs,
                 n_layers,
                 hid_dim,
                 emb_dim,
                 hop,
                 activation,
                 weight_decay,
                 dropout,
                 nlayers,
                 hidden,
                 lr_model,
                 alpha
                 ) -> None:
        self.dataset = None,
        self.reduction_rate = reduction_rate
        self.lab_rand = 1
        self.c_model = 'sgc'
        self.test_model = 'gcn'
        self.dis_metric = 'ours'
        self.nlayers = nlayers
        self.hidden = hidden
        self.lr_adj = lr_adj
        self.lr_feat = lr_feat
        self.lr_model = lr_model
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.alpha = alpha
        
        self.epochs = epochs
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.hop = hop
        self.activation = activation
        
        
        
        

def trainable(config):
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Setup dataset
    data_full = get_dataset(config['dataset'], True)
    data = Transd2Ind(data_full, keep_ratio=1.0)
    
    # Setup training args
    args = Args(
        reduction_rate=config['reduction_rate'],
        lr_adj=config['lr_adj'],
        lr_feat=config['lr_feat'],
        epochs=config['epochs'],
        n_layers=config['n_layers'],
        hid_dim=config['hid_dim'],
        emb_dim=config['emb_dim'],
        hop=config['hop'],
        activation=True,
        weight_decay=config['weight_decay'],
        dropout=config['dropout'],
        nlayers=config['nlayers'],
        hidden=config['hidden']
    )
    args.dataset = config['dataset']
    
    agent = GCDM(data, args, device='cuda')
    agent.train()
    

# Define the search space
config = {
    "lr_adj": tune.loguniform(1e-3, 1e-1),
    "lr_feat": tune.loguniform(1e-3, 1e-1),
    "n_layers": tune.randint(2, 5),
    "hid_dim": tune.qrandint(256, 1024, 256),
    "emb_dim": tune.qrandint(128, 512, 128),
    "hop": tune.randint(1, 3),
    "weight_decay": tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
    "dropout": tune.quniform(0, 0.3, 0.1),
    "nlayers": tune.randint(2, 4),
    "hidden": tune.qrandint(128, 512, 128)
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
    trainable_with_resources = tune.with_resources(trainable, {"gpu": 1, "cpu": 2})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config = tune.TuneConfig(
            search_alg=HyperOptSearch(),
            metric='f1_micro',
            mode='max',
            num_samples=40,
            scheduler=scheduler,
            max_concurrent_trials=8,
        ),
        param_space=config,
        #run_config=RunConfig(storage_path="./results", name="GCDM_hyperTune")
        run_config=RunConfig(storage_path="/data/liang/code/GCond/results/GCDM_hyperTune/", name=f"{config['dataset']}_{config['reduction_rate']}")
    )
    results = tuner.fit()
    
    