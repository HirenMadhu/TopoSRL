import argparse

from preprocessing.simplicial_construction import get_simplicies_closure, get_boundary_matrices, get_neighbors, get_weight_matrix_graph,get_weight_matrix_simplex,get_boundary_matrices_from_processed_tree,generate_triangles,_get_laplacians,_get_simplex_features,augment_simplex,augment_simplex_open
import numpy as np
import torch
from preprocessing.graph_construction import _get_graph
from model.model import MPSN,SCNN,SAN, LogReg
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
import torch.nn as nn
from model.loss import l_rel, l_sub
import time
import numpy as np
import networkx as nx

import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()

parser = argparse.ArgumentParser(description='TopoSRL')

parser.add_argument('--dataname', type=str, default='contact-high-school', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of TopoSRL encoder.')
parser.add_argument('--wd', type=float, default=0.1, help='Weight decay of MLP.')

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
    st_train, sc_train, indices_train, st_test, sc_test, indices_test, train, test = get_simplicies_closure('email-Enron', 10)
    g, netxG = _get_graph(sc_train[1])
    netxG.add_nodes_from(np.setdiff1d(np.arange(1,max(list(netxG.nodes()))), np.array(list(netxG.nodes()))))
    g = g.to(args.device)
    A = g.adj().to_dense().cpu().detach().numpy()
    sm = torch.nn.Softmax(dim=1)
    W0 = get_weight_matrix_graph(A)
    W0 = sm(torch.FloatTensor(W0).to(args.device))
    W0 = W0 * (W0!=W0.min(axis=1).values.unsqueeze(-1))
    open_triangles = train[np.where(train[:,-1]==0), :-1][0]

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
        start_epoch = time.time()
        model.train()
        opt.zero_grad()
        if(args.augmentation.lower() == 'open'):
            st1, sc1, bm1,ind1 = augment_simplex_open(st_train, sc_train, args.rho, open_triangles)
            st2, sc2, bm2,ind2 = augment_simplex_open(st_train, sc_train, args.rho, open_triangles)
        else:
            st1, sc1, bm1,ind1 = augment_simplex(st_train, sc_train, args.rho)
            st2, sc2, bm2,ind2 = augment_simplex(st_train, sc_train, args.rho)

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
            L_sub = args.alpha * l_sub(outputs1, outputs2, [W0, W1, W2], True)
        if(args.alpha<1):
            L_rel = (1-args.alpha) * l_rel(outputs1, outputs2, args.delta, [W0, W1, W2], args.device)
        loss = L_sub + L_rel
        loss.backward()
        opt.step()
        print(f"At epoch:{epoch+1},\tTime: {time.time() - start_epoch}\t Loss:{loss.item()}")
        del(outputs1, outputs2, W1, W2, loss)
        torch.cuda.empty_cache()
        gc.collect()

    _, _, bm = get_boundary_matrices_from_processed_tree(st_train, sc_train, indices_train, 3)
    l, l_d, l_u = _get_laplacians(bm)
    X = _get_simplex_features(sc_train[1:4], g.ndata['features'])
    X = model(X,bm,l,l_u,l_d)[0].cpu().detach().numpy()
    X = X[train[:,:-1]].mean(axis=1)
    y = train[:,-1]

    _, _, bm = get_boundary_matrices_from_processed_tree(st_test, sc_test, indices_test, 3)
    l, l_d, l_u = _get_laplacians(bm)
    X_test = _get_simplex_features(sc_test[1:4], g.ndata['features'])
    X_test = model(X_test,bm,l,l_u,l_d)[0].cpu().detach().numpy()
    X_test = X_test[test[:,:-1]].mean(axis=1)
    y_test = test[:,-1]

    f1 = []
    for i in range(10):
        classifier = RidgeClassifier()
        classifier.fit(X, y)
        y_pred = classifier.predict(X_test)
        f1.append(f1_score(y_test,y_pred, average='macro'))
    f1 = np.array(f1)
    print(f"Average accuracy: {f1.mean()}, Standard Deviation: {f1.std()}")