import torch
import numpy as np
"""
multi-label task scenario
"""
class Base:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        # self.labels_syn = self.generate_labels_syn(n).to(device)

    def generate_labels_syn(self, n):
        labels_train = self.data.labels_train
        print('@prabability sampled multi-label')
        num, class_num = labels_train.shape
        p_labels = np.sum(labels_train, axis=0) / num
        syn_labels = np.zeros((n, class_num), dtype=float)
        for cls in range(class_num):
            set_one = int(n * p_labels[cls])
            indices = np.random.choice(n, set_one ,replace=False)
            syn_labels[indices, cls] = 1
        
        syn_labels = torch.tensor(syn_labels)
        return syn_labels

    def select(self):
        return
   
class Random(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(Random, self).__init__(data, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train

        idx_selected = []
        n = self.nnodes_syn
        selected = np.random.permutation(idx_train)
        idx_selected.append(selected[:n])
        return np.hstack(idx_selected)

class KCenter(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(KCenter, self).__init__(data, args, device, **kwargs)

    def select(self, embeds, inductive=False):
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []

        n = self.nnodes_syn

        embeds = embeds.cpu().numpy()
        
        # Initialize distances
        min_distances = np.full(len(embeds), np.inf)
        
        # Randomly select the first point
        first_selected = np.random.choice(idx_train)
        selected = [first_selected]
        
        # Create a mask for remaining indices
        remaining_mask = np.ones(len(embeds), dtype=bool)
        remaining_mask[first_selected] = False
        
        for _ in range(n - 1):
            # Update minimum distances for remaining indices
            remaining_idx = np.where(remaining_mask)[0]
            distances = np.linalg.norm(embeds[remaining_idx] - embeds[selected[-1]], axis=1)
            min_distances[remaining_idx] = np.minimum(min_distances[remaining_idx], distances)
            # Select the point with the maximum minimum distance
            max_index = remaining_idx[np.argmax(min_distances[remaining_idx])]
            selected.append(max_index)
            # Update remaining_mask
            remaining_mask[max_index] = False
        
        idx_selected.append(selected)
        return np.hstack(idx_selected)

class Herding(Base):

    def __init__(self, data, args, device='cuda', **kwargs):
        super(Herding, self).__init__(data, args, device, **kwargs)

    def select(self, embeds, inductive=False):
        if inductive:
            idx_train = np.arange(len(self.data.idx_train))
        else:
            idx_train = self.data.idx_train
        labels_train = self.data.labels_train
        idx_selected = []
        n = self.nnodes_syn
        embeds = embeds.cpu().numpy()
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
        
        # Compute centroid
        centroid = np.mean(embeds[idx_train], axis=0)
        
        # Herding selection
        selected = []
        remaining_idx = np.array(idx_train)
        for _ in range(n):
            # Compute cosine similarity with the centroid
            similarity = embeds[remaining_idx].dot(centroid)
            # Select the index with the maximum similarity
            max_index = remaining_idx[np.argmax(similarity)]
            selected.append(max_index)
            # Remove the selected index from remaining_idx
            remaining_idx = np.setdiff1d(remaining_idx, max_index)

        idx_selected.append(selected)
        return np.hstack(idx_selected)
