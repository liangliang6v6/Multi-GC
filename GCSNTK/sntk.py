import torch
import math
import torch.nn as nn

class StructureBasedNeuralTangentKernel(nn.Module):
    def __init__(self, K=2, L=2, scale='add'):
        super(StructureBasedNeuralTangentKernel, self).__init__()
        self.K = K
        self.L = L
        self.scale = scale

    def sparse_kron(self, A, B):
        # Ensure matrices are sparse
        if not A.is_sparse:
            A = A.to_sparse()
        if not B.is_sparse:
            B = B.to_sparse()

        m, n = A.shape
        p, q = B.shape
        n_A  = A._values().numel()
        n_B  = B._values().numel()

        indices_A = A.coalesce().indices()
        indices_B = B.coalesce().indices()
        indices_A[0,:] *= p
        indices_A[1,:] *= q

        indices = (indices_A.repeat(n_B, 1) + indices_B.t().reshape(2*n_B, 1))
        ind_row = indices[0::2].reshape(-1)
        ind_col = indices[1::2].reshape(-1)

        new_ind = torch.cat((ind_row, ind_col)).reshape(2, n_A*n_B)
        values = torch.ones(n_A*n_B, device=A.device)
        new_shape = (m * p, n * q)
        
        return torch.sparse_coo_tensor(new_ind, values, new_shape)

    def aggr(self, S, aggr_optor, n1, n2, scale_mat):
        S = torch.sparse.mm(aggr_optor, S.reshape(-1)[:, None]).reshape(n1, n2) * scale_mat
        return S

    def update_sigma(self, S, diag1, diag2):
        S /= diag1[:, None] * diag2[None, :]
        S = torch.clip(S, -0.9999, 0.9999)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        degree_sigma = (math.pi - torch.arccos(S)) / math.pi
        S *= diag1[:, None] * diag2[None, :]
        return S, degree_sigma
    
    def update_diag(self, S):
        diag = torch.sqrt(torch.diag(S))
        S /= diag[:, None] * diag[None, :]
        S = torch.clip(S, -0.9999, 0.9999)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        S *= diag[:, None] * diag[None, :]
        return S, diag
    
    def diag(self, g, A):
        n = A.shape[0]
        aggr_optor = self.sparse_kron(A, A)
        
        # Adjust this operation to avoid .to_dense() where possible
        # Ideally, compute scale_mat in a sparse-friendly way
        if self.scale == 'add':
            scale_mat = 1.
        else:
            if aggr_optor._nnz() == 0:
                scale_mat = torch.zeros(n, n, device=g.device)
            else:
                summed = torch.sparse.sum(aggr_optor, dim=1).to_dense()
                scale_mat = (1. / summed).reshape(n, n)
        
        diag_list = []
        sigma = torch.matmul(g, g.t())
        
        for k in range(self.K):
            sigma = self.aggr(sigma, aggr_optor, n, n, scale_mat)
            for l in range(self.L):
                sigma, diag = self.update_diag(sigma)
                diag_list.append(diag)
        
        return diag_list

    def nodes_gram(self, g1, g2, A1, A2):
        n1, n2 = len(g1), len(g2)
        aggr_optor = self.sparse_kron(A1, A2)

        # Check for empty sparse tensor and handle
        if self.scale != 'add':
            if aggr_optor._nnz() == 0:
                scale_mat = torch.zeros(n1, n2, device=g1.device)
            else:
                scale_mat = (1./torch.sparse.sum(aggr_optor.to_dense(), 1)).reshape(n1, n2)
        else:
            scale_mat = 1.

        sigma = torch.matmul(g1, g2.t())
        theta = sigma
        diag_list1, diag_list2 = self.diag(g1, A1), self.diag(g2, A2)

        for k in range(self.K):
            sigma = self.aggr(sigma, aggr_optor, n1, n2, scale_mat)
            theta = self.aggr(theta, aggr_optor, n1, n2, scale_mat)
            for l in range(self.L):
                sigma, degree_sigma = self.update_sigma(sigma, diag_list1[k*self.L+l], diag_list2[k*self.L+l])
                theta = theta * degree_sigma + sigma

        return theta