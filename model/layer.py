import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from random import shuffle

import warnings
warnings.filterwarnings("ignore")

import gc
gc.enable()

device = torch.device("cuda:1")

class GATLayer(nn.Module):
    def __init__(self, input_size, output_size, bias = True):
        super().__init__()
        self.feature_size = input_size
        self.a_1 = nn.Linear(output_size, 1, bias = bias)
        self.a_2 = nn.Linear(output_size, 1, bias = bias)
        self.layer = nn.Linear(input_size, output_size, bias = bias)

    def forward(self, features, adj, boundary=False, lower_boundary = False):
        original_size = adj.shape
        original_features = features.shape[0]
        if(boundary):
            if(adj.shape[1]>adj.shape[0]):
                original = features.shape[0]
                features = torch.cat((features, torch.zeros(max(adj.shape)-features.shape[0], self.feature_size).to(device)))
                adj = torch.cat((adj, torch.zeros(max(adj.shape)-min(adj.shape), max(adj.shape)).to(device)))
            else:
                original = features.shape[0]
                features = torch.cat((features, torch.zeros(max(adj.shape)-features.shape[0], self.feature_size).to(device)))
                adj = torch.cat((adj, torch.zeros(max(adj.shape), max(adj.shape)-min(adj.shape)).to(device)), dim=1)
        row,col = np.where(adj.cpu().detach().numpy()==1)
        indices = np.vstack([row,col])
        features = self.layer(features)
        a_1 = self.a_1(features)
        a_2 = self.a_2(features)
        v =  (a_1+a_2.T).clone()[indices]
        v = nn.LeakyReLU(inplace=False)(v)
        e = torch.sparse_coo_tensor(indices, v, size = (adj.shape[0],adj.shape[1]))
        attention = torch.sparse.softmax(e, dim=1)
        features = features[:original_features]
        
        if(boundary):
            attention = attention.to_dense()[:original_size[0], :original_size[1]]
            if(lower_boundary):
                output = attention.T@features
            else:
                output = attention@features
        else:
            output = torch.sparse.mm(attention, features)
        return output

class SANLayer(nn.Module):

    def __init__(self, input_size, output_size, dimensions, agg = 'sum', bias=False, orientated=False):

        super().__init__()
        
        layer = GATLayer
        self.activation = F.tanh
        self.dimensions = dimensions
        self.l_d_layers = nn.ModuleList([layer(input_size, output_size, bias) for i in range(dimensions)])
        self.l_u_layers = nn.ModuleList([layer(input_size, output_size, bias) for i in range(dimensions)])
        self.p_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.s_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        

    def forward(self, X, L, L_u, L_d, dimensions):
        out = []
        for i in range(dimensions):
            if(X[i] is not None):
                h_p = self.p_layers[i](X[i])
                h_p = self.activation(torch.sparse.mm(L[i], h_p))
                h_s = self.activation(self.s_layers[i](X[i]))
                h_u, h_d = torch.zeros(h_p.shape, device = device), torch.zeros(h_p.shape, device = device)
                if L_u[i] is not None:
                    h_u = self.activation(self.l_u_layers[i](X[i], L_u[i]))
                if L_d[i] is not None:
                    h_d = self.activation(self.l_d_layers[i](X[i], L_d[i]))

                out.append(h_s + h_u + h_d + h_p)
        return out

class SCNNLayer(nn.Module):

    def __init__(self, input_size, output_size, dimensions, agg = 'sum', bias=False, orientated=False):

        super().__init__()
        
        self.activation = F.tanh
        self.dimensions = dimensions
        self.b_d_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.b_u_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.s_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        
    def forward(self, X, B, dimensions):
        out = []
        for i in range(dimensions):
            h_s = self.activation(self.s_layers[i](X[i]))

            h_bu, h_bd = torch.zeros(h_s.shape, device = device), torch.zeros(h_s.shape, device = device)
            if B[i] is not None:
                if(i<dimensions-1):
                    h_bd = self.activation(torch.sparse.mm(B[i], self.b_d_layers[i](X[i+1])))
            if(i<dimensions-1):
                if B[i+1] is not None:
                    if(i>0):
                        h_bu = self.activation(torch.sparse.mm(B[i-1].T, self.b_u_layers[i](X[i-1])))
            out.append(h_s + h_bd + h_bu)
        return out

class SCNLayer(nn.Module):

    def __init__(self, input_size, output_size, dimensions, agg = 'sum', bias=False, orientated=False):

        super().__init__()
        
        self.activation = F.tanh
        self.dimensions = dimensions
        self.l_d_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.l_u_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.b_u_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.b_d_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.p_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        self.s_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias) for i in range(dimensions)])
        

    def forward(self, X, B, L, L_u, L_d, dimensions):
        out = []
        for i in range(dimensions):
            if(X[i] is not None):
                h_p = self.activation(torch.sparse.mm(L[i], self.p_layers[i](X[i])))
                h_s = self.activation(self.s_layers[i](X[i]))

                h_u, h_d = torch.zeros(h_p.shape, device = device), torch.zeros(h_p.shape, device = device)
                if L_u[i] is not None:
                    h_u = self.activation(torch.sparse.mm(L_u[i], self.l_u_layers[i](X[i])))
                if L_d[i] is not None:
                    h_u = self.activation(torch.sparse.mm(L_d[i], self.l_d_layers[i](X[i])))

                h_bu, h_bd = torch.zeros(h_p.shape, device = device), torch.zeros(h_p.shape, device = device)
                if B[i] is not None:
                    if(i<dimensions-1):
                        h_bd = self.activation(torch.sparse.mm(B[i], self.b_d_layers[i](X[i+1])))
                if(i<dimensions-1):
                    if B[i+1] is not None:
                        if(i>0):
                            h_bu = self.activation(torch.sparse.mm(B[i-1].T, self.b_u_layers[i](X[i-1])))
                out.append(h_s + h_u + h_d + h_p + h_bd + h_bu)
        return out

class SCNLLayer(nn.Module):

    def __init__(self, input_size, output_size, dimensions, agg = 'sum', bias=False, orientated=False):

        super().__init__()
        
        self.activation = F.tanh
        self.dimensions = dimensions
        self.l_d_layers = nn.Linear(input_size, output_size, bias)
        self.l_u_layers = nn.Linear(input_size, output_size, bias) 
        self.s_layers = nn.Linear(input_size, output_size, bias) 

    def forward(self, X, L_u, L_d):
        h_s = self.activation(self.s_layers(X))
        h_u, h_d = torch.zeros(h_s.shape, device = device), torch.zeros(h_s.shape, device = device)
        if L_u is not None:
            h_u = self.activation(torch.sparse.mm(self.l_u_layers(X),L_u))
        if L_d is not None:
            h_d = self.activation(torch.sparse.mm(self.l_d_layers(X),L_d))
        return h_s + h_u + h_d