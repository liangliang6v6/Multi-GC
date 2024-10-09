from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(root='data',name = 'ogbn-proteins') 

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0]
print(graph.y[0])