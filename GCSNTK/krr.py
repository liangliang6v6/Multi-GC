import torch
import torch.nn as nn
from sklearn import metrics
def ml_acc(output, labels, threshold=0.5):
    binary_pred = (torch.sigmoid(output) >= threshold).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    f1_micro = metrics.f1_score(y_pred=binary_pred,y_true = labels,average='micro', zero_division=0)
    f1_macro = metrics.f1_score(y_pred=binary_pred,y_true = labels,average='macro', zero_division=0)
    f1_weight = metrics.f1_score(y_pred=binary_pred,y_true = labels,average='weighted', zero_division=0)

    return f1_micro,f1_macro,f1_weight

class KernelRidgeRegression(nn.Module):
    def __init__(self, kernel, ridge):
        super(KernelRidgeRegression, self).__init__()
        self.kernel   = kernel
        self.ridge    = ridge
    
    # Original graph node features: G_t; label: y_t; edge: E_t
    # synthetic graph node features: G_s; label: y_s; edge: E_s
    def forward(self, G_t, G_s, y_t, y_s, E_t, E_s):
        K_ss      = self.kernel(G_s, G_s, E_s, E_s)
        K_ts      = self.kernel(G_t, G_s, E_t, E_s)
        n = len(G_s)
        K_ss = K_ss.to(dtype=torch.float32)
        y_s = y_s.to(dtype=torch.float32)
        regulizer = self.ridge * torch.trace(K_ss) * torch.eye(n, device=G_s.device) / n
        b = torch.linalg.solve(K_ss + regulizer, y_s)
        preds = torch.matmul(K_ts, b)
        f1_micro,f1_macro,f1_weight = ml_acc(preds, y_t)

        return preds, f1_micro,f1_macro,f1_weight