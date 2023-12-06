import torch
import gc
gc.enable()

def calculate_loss_tp(X1, X2, alpha, W, delta, device):
    loss = 0
    W_sparse = W[0].to_sparse()
    W_nonzero_indices = W_sparse.indices()
    W_nonzero_values = W_sparse.values()
    selected = torch.randint(0,len(W_nonzero_values), (1,delta))
    W_nonzero_indices = W_nonzero_indices.T[selected[0]].T
    W_nonzero_values = W_nonzero_values[selected[0]]
    dist_bet = torch.cdist(X1.T, X2.T, p=2).to(device)
    dist_in1 = torch.cdist(X1.T, X1.T, p=2).to(device)
    dist_in2 = torch.cdist(X2.T, X2.T, p=2).to(device)
    L = torch.zeros(W[0].shape).to(device)
    torch.cuda.empty_cache()
    gc.collect()
    for j in W_nonzero_indices[0]:
        for j_prime in W_nonzero_indices[1]:
            L[j,j_prime] = ((dist_in1[j][W_nonzero_indices[0]] - dist_in2[j_prime][W_nonzero_indices[1]]).square() * W_nonzero_values).sum()
    L_sub = torch.trace(dist_bet.T.to(device)@W[0])
    L_rel = torch.sum(L * W[0])
    loss += alpha*L_sub + (1-alpha)*L_rel
    del(dist_bet, dist_in1, dist_in2, W_sparse, W_nonzero_indices)
    torch.cuda.empty_cache()
    gc.collect()
    return loss

def l_sub(X1, X2, W, closure=False):
    loss = 0
    for i in range(len(X1)):
        dist_bet = torch.cdist(X1[i], X2[i], p=2)
        #if(i==0 and not closure):
        #    loss += torch.trace(dist_bet[1:,1:].T@W[i])
        #else:
        loss += torch.trace(dist_bet.T@W[i])
    return loss

def l_rel(X1, X2, delta, W, device, closure = False):
    loss = 0
    for i in range(len(X1)):
        W_sparse = W[i].to_sparse()
        W_nonzero_indices = W_sparse.indices()
        W_nonzero_values = W_sparse.values()
        selected = torch.randint(0,len(W_nonzero_values), (1,delta))
        W_nonzero_indices = W_nonzero_indices.T[selected[0]].T
        W_nonzero_values = W_nonzero_values[selected[0]]
        dist_in1 = torch.cdist(X1[i], X1[i], p=2).to(device)
        dist_in2 = torch.cdist(X2[i], X2[i], p=2).to(device)
        if(i==0):
            L = torch.zeros((W[i].shape[0]+1,W[i].shape[0]+1)).to(device)
        else:
            L = torch.zeros(W[i].shape).to(device)
        torch.cuda.empty_cache()
        gc.collect()
        for j in W_nonzero_indices[0]:
            for j_prime in W_nonzero_indices[1]:
                L[j,j_prime] = ((dist_in1[j][W_nonzero_indices[0]] - dist_in2[j_prime][W_nonzero_indices[1]]).square() * W_nonzero_values).sum()
        if(i==0):
            dgw = torch.sum(L[1:, 1:] * W[i])
        else:
            dgw = torch.sum(L * W[i])
        loss += dgw
        del(dist_in1, dist_in2, W_sparse, W_nonzero_indices)
        torch.cuda.empty_cache()
        gc.collect()
    return loss

