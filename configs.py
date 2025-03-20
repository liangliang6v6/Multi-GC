'''Configuration'''

def load_config(args):
    dataset = args.dataset
    if dataset in ['ppi', 'ppi-large']:
        args.nlayers = 2
        args.hidden = 256
        args.weight_decay = 5e-3
        args.dropout = 0.0

    if dataset in ['dblp', 'pcg', 'hg', 'eg']:
        args.nlayers = 2
        args.hidden = 256
        args.weight_decay = 0e-4
        args.dropout = 0

    if dataset in ['ogbn-pro', 'yelp']:
        args.hidden = 256
        args.weight_decay = 0
        args.dropout = 0

    return args

