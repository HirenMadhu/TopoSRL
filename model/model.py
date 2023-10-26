import torch.nn.functional as F
import torch.nn as nn
from random import shuffle
from model.layer import SANLayer, SCNNLayer, SCNLayer, SCNLLayer
import torch.nn as nn
from dgl.nn import GraphConv, GATConv, SAGEConv, GINConv
import torch
import warnings
warnings.filterwarnings("ignore")

import gc
gc.enable()

device = torch.device("cuda:1")
class SAN(nn.Module):
    def __init__(self, num_feats, hidden_size, output_size, dimensions, agg='sum', f=F.tanh, bias=False, classification=False):
        super().__init__()
        f_size = 32
        self.f = f
        self.classification = classification
        self.dimensions = dimensions
        self.layer1 = SANLayer(num_feats, hidden_size, dimensions)
        self.layer2 = SANLayer(hidden_size, hidden_size, dimensions)
        if(self.classification):
            self.fc = nn.Linear(hidden_size, output_size) 
        else:
            self.layer3 = SANLayer(hidden_size, output_size, dimensions)

    def forward(self, X, B, L, L_u, L_d, hyperedges_classification=False, hyperedges= None):
        out = self.layer1(X, L, L_u, L_d,self.dimensions)
        out = self.layer2(out, L, L_u, L_d,self.dimensions)
        if(self.classification):
            if(hyperedges_classification):
                out = torch.stack([out[0][indices].mean(dim=0) for indices in hyperedges])
                out = self.fc(out)    
            else:
                out = self.fc(out[0][1:])
        else:
            out = self.layer3(out, L, L_u, L_d,self.dimensions)
        return out

class SCNN(nn.Module):
    def __init__(self, num_feats, hidden_size, output_size, dimensions, agg='sum', f=F.tanh, bias=False):
        super().__init__()
        f_size = 32
        self.f = f
        self.dimensions = dimensions
        self.layer1 = SCNNLayer(num_feats, hidden_size, dimensions)
        self.layer2 = SCNNLayer(hidden_size, hidden_size, dimensions)
        self.layer3 = SCNNLayer(hidden_size, output_size, dimensions)
        self.fc = nn.Linear(output_size,output_size, bias=False)

    def forward(self, X, B, L, L_u, L_d, hyperedges_classification=False, hyperedges= None):
        out = self.layer1(X, B, self.dimensions)
        out = self.layer2(out, B, self.dimensions)
        out = self.layer3(out, B, self.dimensions)
        if(hyperedges_classification):
            out[0] = torch.stack([out[0][indices].mean(dim=0) for indices in hyperedges])
        for i in range(len(out)):
            out[i] = self.fc(out[i])
        return out

class MPSN(nn.Module):
    def __init__(self, num_feats, hidden_size, output_size, dimensions, agg='sum', f=F.tanh, bias=False):
        super().__init__()
        f_size = 32
        self.f = f
        self.dimensions = dimensions
        self.output_size = output_size
        self.layer1 = SCNLayer(num_feats, hidden_size, dimensions)
        self.layer2 = SCNLayer(hidden_size, hidden_size, dimensions)
        self.layer3 = SCNLayer(hidden_size, output_size, dimensions)
        self.fc = nn.Linear(output_size,output_size, bias=False)

    def forward(self, X, B, L, L_u, L_d, hyperedges_classification=False, hyperedges= None):
        out = self.layer1(X, B, L, L_u, L_d, self.dimensions)
        out = self.layer2(out, B, L, L_u, L_d, self.dimensions)
        out = self.layer3(out, B, L, L_u, L_d, self.dimensions)
        if(hyperedges_classification):
            out[0] = torch.stack([out[0][indices].mean(dim=0) for indices in hyperedges])
        for i in range(len(out)):
            out[i] = self.fc(out[i])
        return out
    def encode(self, X, B, L, L_u, L_d):
        out = self.layer1(X, B, L, L_u, L_d, self.dimensions)
        out = self.layer2(out, B, L, L_u, L_d, self.dimensions)
        out = self.layer3(out, B, L, L_u, L_d, self.dimensions)
        return out

class MPSN_L(nn.Module):
    def __init__(self, num_feats, hidden_size, output_size, dimensions, agg='sum', f=F.tanh, bias=False):
        super().__init__()
        f_size = 32
        self.f = f
        self.dimensions = dimensions
        self.output_size = output_size
        self.layer1 = SCNLLayer(num_feats, hidden_size, dimensions)
        self.layer2 = SCNLLayer(hidden_size, hidden_size, dimensions)
        self.layer3 = SCNLLayer(hidden_size, output_size, dimensions)
        self.fc = nn.Linear(output_size,output_size, bias=False)

    def forward(self, X, L_u, L_d):
        out = self.layer1(X, L_u, L_d)
        out = self.layer2(out, L_u, L_d)
        out = self.layer3(out, L_u, L_d)
        out = F.tanh(F.normalize(self.fc(out)))
        return out

class semiMPSN(nn.Module):
    def __init__(self, num_feats, hidden_size, output_size, num_classes, dimensions, agg='sum', f=F.tanh, bias=False):
        super().__init__()
        f_size = 32
        self.f = f
        self.dimensions = dimensions
        self.output_size = output_size
        self.layer1 = SCNLayer(num_feats, hidden_size, dimensions)
        self.layer2 = SCNLayer(hidden_size, hidden_size, dimensions)
        self.layer3 = SCNLayer(hidden_size, output_size, dimensions)
        self.fc = nn.Linear(output_size,output_size, bias=False)
        self.classifier = nn.Linear(output_size,num_classes)

    def forward(self, X, B, L, L_u, L_d, node_class = False, hyperedges_classification=False, hyperedges= None):
        out = self.layer1(X, B, L, L_u, L_d, self.dimensions)
        out = self.layer2(out, B, L, L_u, L_d, self.dimensions)
        out = self.layer3(out, B, L, L_u, L_d, self.dimensions)
        logits = None
        if(hyperedges_classification):
            out[0] = torch.stack([out[0][indices].mean(dim=0) for indices in hyperedges])
        for i in range(len(out)):
            out[i] = self.fc(out[i])
        if(node_class):
            logits = self.classifier(out[0][1:])
        return out, logits   

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat, hyperedges_classification=False, hyperedges= None):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        if(hyperedges_classification):
            h = torch.stack([h[0][indices].mean(dim=0) for indices in hyperedges])
        return h
    
    def encode(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        return h
    
class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, 1)
        self.conv2 = GATConv(h_feats, h_feats, 1)
        self.conv3 = GATConv(h_feats, num_classes, 1)

    def forward(self, g, in_feat, hyperedges_classification=False, hyperedges= None):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = h.view(len(h), h.shape[-1])
        if(hyperedges_classification):
            h = torch.stack([h[0][indices].mean(dim=0) for indices in hyperedges])
        return h
    

class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, in_feat, hyperedges_classification=False, hyperedges= None):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        if(hyperedges_classification):
            h = torch.stack([h[0][indices].mean(dim=0) for indices in hyperedges])
        return h
    
class GIN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GIN, self).__init__()
        W1 = nn.Linear(in_feats, h_feats)
        W2 = nn.Linear(h_feats, h_feats)
        W3 = nn.Linear(h_feats, num_classes)
        self.conv1 = GINConv(W1)
        self.conv2 = GINConv(W2)
        self.conv3 = GINConv(W3)

    def forward(self, g, in_feat, hyperedges_classification=False, hyperedges= None):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        if(hyperedges_classification):
            h = torch.stack([h[0][indices].mean(dim=0) for indices in hyperedges])
        return h

    def encode(self, g, in_feat, hyperedges_classification=False, hyperedges= None):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        return h
    
class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return x