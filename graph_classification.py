import argparse

import numpy as np
from tqdm import tqdm
import gudhi
import torch
from preprocessing.simplicial_construction import get_boundary_matrices, get_boundary_matrices_from_processed_tree, process_simplex_tree, get_neighbors, get_weight_matrix_graph, get_weight_matrix_simplex,generate_triangles, augment_simplex_open_gc, _get_laplacians,_get_simplex_features_gc
from preprocessing.graph_construction import _get_graph
from model.model import MPSN,SCNN,SAN
import torch.nn as nn
from model.loss import l_rel, l_sub
import copy
import time
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import torch
import networkx as nx
from sklearn import metrics
from sklearn.metrics import classification_report,f1_score, accuracy_score
import sys

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import gc
gc.enable()

parser = argparse.ArgumentParser(description='TopoSRL')

parser.add_argument('--dataname', type=str, default='proteins', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of TopoSRL encoder.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay of TopoSRL encoder.')

parser.add_argument('--dim', type=int, default=4, help='Order of the simplicial complex.')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha.')
parser.add_argument('--snn', type=str, default='MPSN', help='Type of SNN')
parser.add_argument('--delta', type=int, default=20, help='Number of samples to calculate L_rel')
parser.add_argument('--augmentation', type=str,  default='open', help='Type of agumentation')
parser.add_argument('--rho', type=float, default=0.1, help='Simplex removing and adding ratio.')

args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    data = args.dataname
    alpha = args.alpha
    delta = args.delta
    epochs = args.epochs
    labels = np.load('data/graph classification/'+data+'/label_sets_'+data+'.npy', allow_pickle=True)
    simplicial = np.load('data/graph classification/'+data+'/simplicial_sets_'+data+'.npy', allow_pickle=True)

    SCs = []
    INDs = []
    _labels = []
    simplex_trees = []
    node_attributes = []
    netxG = []
    for p in range(len(simplicial)):
        for q in range(len(simplicial[p])):
            simplex_tree = gudhi.SimplexTree()
            sc = [[] for i in range(4)]
            for i in range(4):
                for j in simplicial[p][q][i]:
                    sc[i].append(list(j))
                    simplex_tree.insert(list(j))
            for i in range(len(sc)):
                sc[i] = np.array(sc[i])
            if(len(sc[3])):
                INDs.append(simplicial[p][q])
                g = nx.from_edgelist(sc[1])
                _labels.append(labels[p][q])
                node_attributes.append(nx.adjacency_matrix(g).todense().sum(1))
                netxG.append(g)
                SCs.append(sc)    
                simplex_trees.append(simplex_tree)
    labels = np.array(_labels)
    print("Length of dataset:", len(SCs))
    feature_size = 1
    model = MPSN(feature_size, 10*feature_size, 20*feature_size, 3, agg='sum').to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    sm = torch.nn.Softmax(dim=1)
    accuracies = []
    for epoch in range(epochs):
            start_epoch = time.time()
            model.train()
            opt.zero_grad()
            loss = 0
            skipped = []
            print(f"Epoch {epoch+1} training:")
            for i in tqdm(range(len(labels))):
                try:
                    simplex_tree = simplex_trees[i]
                    sc = SCs[i]
                    st1, sc1, bm1,ind1 = augment_simplex_open_gc(simplex_tree, sc, [])
                    l1, l1_d, l1_u = _get_laplacians(bm1)
                    st2, sc2, bm2,ind2 = augment_simplex_open_gc(simplex_tree, sc, [])
                    l2, l2_d, l2_u = _get_laplacians(bm2)
                    W0 = get_weight_matrix_graph(nx.adjacency_matrix(netxG[i]).todense())
                    W0 = sm(torch.FloatTensor(W0).to(args.device))
                    W0 = W0 * (W0!=W0.min(axis=1).values.unsqueeze(-1))
                    W1 = get_weight_matrix_simplex(1, sc1, sc2, ind1, ind2, l1)
                    W2 = get_weight_matrix_simplex(2, sc1, sc2, ind1, ind2, l1)
                    W1 = sm(torch.FloatTensor(W1)).to(args.device)
                    W2 = sm(torch.FloatTensor(W2)).to(args.device)
                    W1 = W1 * (W1!=W1.min(axis=1).values.unsqueeze(-1))
                    W2 = W2 * (W2!=W2.min(axis=1).values.unsqueeze(-1))

                    X1 = _get_simplex_features_gc(sc1[1:], torch.from_numpy(node_attributes[i]).to(args.device).float())
                    X2 = _get_simplex_features_gc(sc2[1:], torch.from_numpy(node_attributes[i]).to(args.device).float())

                    outputs1 = model(X1,bm1,l1,l1_u,l1_d)
                    outputs2 = model(X2,bm2,l2,l2_u,l2_d)
                    if(args.alpha>0):
                        L_sub = args.alpha * l_sub(outputs1, outputs2, [W0, W1, W2], True)
                    if(args.alpha<1):
                        L_rel = (1-args.alpha) * l_rel(outputs1, outputs2, args.delta, [W0, W1, W2], args.device)
                    loss = L_sub + L_rel
                    del(st1, sc1, bm1,ind1, l1, l1_d, l1_u, st2, sc2, bm2,ind2, l2, l2_d, l2_u, X1, X2)
                    torch.cuda.empty_cache()
                    gc.collect()
                except RuntimeError:
                    skipped.append(i)
            loss.backward()
            opt.step()
            print(f"At epoch:{epoch+1},\tTime: {time.time() - start_epoch}\t Loss:{loss.item()}")
            del(loss)
            embeddings = []
            model.eval()
            _labels = []
            print(f"Epoch {epoch+1} evaluating:")
            for i in tqdm(range(len(labels))):
                try:
                    _,_,bm, = get_boundary_matrices_from_processed_tree(simplex_trees[i], SCs[i], INDs[i], 3)
                    l1, l1_d, l1_u = _get_laplacians(bm)
                    _X = torch.FloatTensor(node_attributes[i]).to(args.device).view(len(node_attributes[i]),1)
                    X1 = _get_simplex_features_gc(SCs[i][1:3],_X)
                    outputs = model(X1,bm,l1,l1_u,l1_d)
                    embeddings.append(torch.cat((outputs[0].mean(0), outputs[1].mean(0), outputs[2].mean(0))).cpu().detach().numpy())
                    _labels.append(labels[i])
                except RuntimeError:
                    skipped.append(i)
            _labels = np.array(_labels)         
            embeddings=np.array(embeddings)
            X_train, X_test, y_train, y_test = train_test_split(embeddings,_labels,test_size=0.2, random_state = 60, stratify=_labels)
            lr = RidgeClassifier()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            print(f'Test accuracy: {accuracy_score(y_test, y_pred)}')
            accuracies.append(accuracy_score(y_test, y_pred))
            torch.cuda.empty_cache()
            gc.collect()
    print(f'Best Accuracy: {max(accuracies)}')