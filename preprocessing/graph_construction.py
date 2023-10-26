import numpy as np
import dgl
from dgl import function as fn
import torch
import networkx as nx
import random
device = torch.device("cpu")
device = torch.device("cuda:1")

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.Tensor(np.eye(num_classes, dtype='uint8')[y])

def _get_graph(edges, num_nodes = None, features=None, constraint = True):
    if(num_nodes is not None):
        g = dgl.graph((edges[:,0], edges[:,1]), num_nodes=num_nodes).to(device)
    else:
        g = dgl.graph((edges[:,0], edges[:,1])).to(device)
    g = dgl.add_reverse_edges(g)
    #g = dgl.add_self_loop(g)
    netxG = nx.from_edgelist(np.vstack([g.edges()[0].cpu().detach().numpy(), g.edges()[1].cpu().detach().numpy()]).T)
    if(features is not None):
        g.ndata['features'] = features
        return g, netxG
    while(True):
        g.edata['distance'] = torch.ones(g.num_edges()).to(device)
        if(nx.is_connected(netxG)):
            anchor_nodes = torch.randint(high = g.num_nodes(), size = (nx.diameter(netxG),)).to(device)
        else:
            anchor_nodes = []
            for i in nx.connected_components(netxG):
                anchor_nodes += random.choices(list(i), k=nx.diameter(netxG.subgraph(i)))
            x = []
            for i in anchor_nodes:
                if(type(i)==list):
                    for j in i:
                        x.append(j)
                else:
                    x.append(i)
            anchor_nodes = torch.Tensor(x).to(device).long()
        print("Anchor nodes initialized")
        distances = torch.ones((len(anchor_nodes), g.num_nodes())).to(device)
        for k,i in enumerate(anchor_nodes):
            g.ndata['x'] = torch.zeros(g.num_nodes()).to(device)
            g.ndata['x'][:] = 1000000
            g.ndata['x'][i] = 0
            for _ in range(len(anchor_nodes)):
                g.update_all(fn.u_add_e('x', 'distance', 'm'), fn.min('m', 'm'))
                g.ndata['x'] = torch.min(g.ndata['x'], g.ndata['m'])
            distances[k] = g.ndata['x']
        if(nx.is_connected(netxG) and constraint):
            if((distances.T[anchor_nodes].sum(axis=1)/(nx.diameter(netxG)-1)).mean()>=len(anchor_nodes)*0.33):
                break
        else:
                break
    g.ndata['features'] = to_categorical(torch.argmin(distances, dim=0).to(torch.device("cpu")), len(anchor_nodes)).to(device)
    return g, netxG