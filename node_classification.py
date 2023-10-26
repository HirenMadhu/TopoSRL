import argparse

from preprocessing.simplicial_construction import get_boundary_matrices, get_neighbors, get_weight_matrix_graph, get_weight_matrix_simplex,generate_triangles,_get_laplacians,_get_simplex_features,augment_simplex,augment_simplex_open
import numpy as np
from tqdm import tqdm
import torch
from preprocessing.graph_construction import _get_graph
from model.model import MPSN,SCNN,SAN, LogReg
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from model.loss import l_rel, l_sub
import time
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()

parser = argparse.ArgumentParser(description='TopoSRL')

parser.add_argument('--dataname', type=str, default='contact-high-school', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of TopoSRL encoder.')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay of MLP.')

parser.add_argument('--dim', type=int, default=4, help='Order of the simplicial complex.')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha.')
parser.add_argument('--snn', type=str, default='MPSN', help='Type of SNN')
parser.add_argument('--delta', type=int, default=300, help='Number of samples to calculate L_rel')
parser.add_argument('--augmentation', type=str,  default='open', help='Type of agumentation')
parser.add_argument('--rho', type=float, default=0.1, help='Simplex removing and adding ratio.')

args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)
    with open('only open.txt', 'a') as f:
        print(args, file=f)
    simplex_tree, sc, boundry_matrices, labels =  get_boundary_matrices(args.dataname, args.dim)
    print("Got boundaries")
    g, netxG = _get_graph(sc[1])
    g = g.to(args.device)
    A = nx.adjacency_matrix(netxG).todense()
    sm = torch.nn.Softmax(dim=1)
    W0 = get_weight_matrix_graph(A)
    W0 = sm(torch.FloatTensor(W0).to(args.device))
    W0 = W0 * (W0!=W0.min(axis=1).values.unsqueeze(-1))
    laplacians, lower_laplacians, upper_laplacians = _get_laplacians(boundry_matrices)
    _X = _get_simplex_features(sc[1:4], g.ndata['features'])
    
    triangles = generate_triangles(get_neighbors(A, True))
    open_triangles = []
    print("Getting open triangles:")
    for i in tqdm(triangles):
        if(i not in sc[2]):
            open_triangles.append(i)
    open_triangles = np.array(open_triangles)
    feature_size = g.ndata['features'].shape[1]
    if(args.snn.lower() == 'mpsn'):
        model = MPSN(feature_size, 10*feature_size, 20*feature_size, args.dim-1, agg='sum').to(args.device)
    elif(args.snn.lower() == 'scnn'):
        model = SCNN(feature_size, 10*feature_size, 20*feature_size, args.dim-1, agg='sum').to(args.device)
    elif(args.snn.lower() == 'san'):
        model = SAN(feature_size, 10*feature_size, 20*feature_size, args.dim-1, agg='sum').to(args.device)
    else:
        raise TypeError("Enter correct option for SNN encder. The three available options are:\n(1) MPSN\n(2)SCNN\n(3)SAN")
    opt = torch.optim.Adam(model.parameters(), lr = args.lr1, weight_decay=args.wd2)

    for epoch in range(args.epochs):
        print('Training epoch: ', epoch)
        start_epoch = time.time()
        model.train()
        opt.zero_grad()
        if(args.augmentation.lower() == 'open'):
            st1, sc1, bm1,ind1 = augment_simplex_open(simplex_tree, sc, args.rho, open_triangles)
            st2, sc2, bm2,ind2 = augment_simplex_open(simplex_tree, sc, args.rho, open_triangles)
        else:
            st1, sc1, bm1,ind1 = augment_simplex(simplex_tree, sc, args.rho)
            st2, sc2, bm2,ind2 = augment_simplex(simplex_tree, sc, args.rho)

        l1, l1_d, l1_u = _get_laplacians(bm1)
        l2, l2_d, l2_u = _get_laplacians(bm2)
        
        W1 = get_weight_matrix_simplex(1, sc1, sc2, ind1, ind2, l1)
        W2 = get_weight_matrix_simplex(2, sc1, sc2, ind1, ind2, l1)
        W1 = sm(torch.FloatTensor(W1)).to(args.device)
        W2 = sm(torch.FloatTensor(W2)).to(args.device)
        W1 = W1 * (W1!=W1.min(axis=1).values.unsqueeze(-1))
        W2 = W2 * (W2!=W2.min(axis=1).values.unsqueeze(-1))

        X1 = _get_simplex_features(sc1[1:], g.ndata['features'])
        X2 = _get_simplex_features(sc2[1:], g.ndata['features'])

        outputs1 = model(X1,bm1,l1,l1_u,l1_d)
        outputs2 = model(X2,bm2,l2,l2_u,l2_d)
        del(st1, sc1, bm1,ind1, l1, l1_d, l1_u, st2, sc2, bm2,ind2, l2, l2_d, l2_u, X1, X2)
        torch.cuda.empty_cache()
        gc.collect()
        if(args.alpha>0):
            L_sub = args.alpha * l_sub(outputs1, outputs2, [W0, W1, W2])
        if(args.alpha<1):
            L_rel = (1-args.alpha) * l_rel(outputs1, outputs2, args.delta, [W0, W1, W2], args.device)
        loss = L_sub + L_rel
        loss.backward()
        opt.step()
        print(f"At epoch:{epoch+1},\tTime: {time.time() - start_epoch}\t Loss:{loss.item()}")
        del(outputs1, outputs2, W1, W2, loss)
        torch.cuda.empty_cache()
        gc.collect()
    
    num_class = len(np.unique(labels))

    X = model(_X,boundry_matrices,laplacians,upper_laplacians,lower_laplacians)[0][1:].cpu().detach().numpy()
    
    accuracies = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X,labels,test_size=0.2)
        classifier = RidgeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test,y_pred))
    accuracies = np.array(accuracies)
    print(f"Average accuracy: {accuracies.mean()}, Standard Deviation: {accuracies.std()}")
    