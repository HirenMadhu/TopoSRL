import argparse

import numpy as np
import torch
import torch.nn.functional
import torch.utils.data as data
import numpy as np
import time
from preprocessing.simplicial_construction import get_boundary_matrices, get_weight_matrix_simplex, process_simplex_tree, get_neighbors, get_weight_matrix_graph, get_weight_matrix_simplex,generate_triangles,_get_laplacians,_get_simplex_features,augment_simplex_tp
import pickle
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from model.loss import calculate_loss_tp
from tqdm import tqdm
import gudhi
import torch
from preprocessing.graph_construction import _get_graph
from model.model import MPSN,SCNN,SAN, MPSN_L
import torch.nn as nn
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
import gc
gc.enable()

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def save_variable(variable,filename):
  pickle.dump(variable,open(filename, "wb"))
def load_variable(filename):
  return pickle.load(open(filename,'rb')) 

def _get_triangles(edges,triangles):
    triangles_2 = []
    for i in triangles:
        triangles_2.append(np.unique(np.append(edges[i[0]], edges[i[1]])))
    return np.array(triangles_2)
parser = argparse.ArgumentParser(description='TopoSRL')

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
  dataset = 'ocean'
  datapath = 'data/tp/ocean/'

  X = np.load(datapath+'flows_in.npy')

  train_mask = np.load(datapath+'train_mask.npy')
  test_mask = np.load(datapath+'test_mask.npy')

  last_nodes = np.load(datapath+'last_nodes.npy')
  target_nodes = np.load(datapath+'target_nodes.npy')
  Y = np.load(datapath+'targets.npy')
  y_tr = [list(np.squeeze(Y[np.where(train_mask!=0)])[i]) for i in range(len(Y[np.where(train_mask!=0)]))]
  y_tr_ = len(y_tr)*[1] # 160 for buoy

  for i in range(len(y_tr)): #160
    if y_tr[i] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]: y_tr_[i] = [0]
    elif y_tr[i] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]: y_tr_[i] = [1]
    elif y_tr[i] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]: y_tr_[i] = [2]
    elif y_tr[i] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]: y_tr_[i] = [3]
    elif y_tr[i] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]: y_tr_[i] = [4]
    elif y_tr[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]: y_tr_[i] = [5]
    
  y_tr = torch.squeeze(torch.Tensor(np.array([[int(y_tr_[i][0])] for i in range(len(y_tr_))])))
  y_test = [list(np.squeeze(Y[test_mask!=0])[i]) for i in range(len(Y[test_mask!=0]))]
  y_test_ = len(y_test)*[1] # 40 for buoy
  for i in range(len(y_test)): # 40
    if y_test[i] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]: y_test_[i] = [0] #[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for buoy
    elif y_test[i] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]: y_test_[i] = [1]
    elif y_test[i] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]: y_test_[i] = [2]
    elif y_test[i] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]: y_test_[i] = [3]
    elif y_test[i] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]: y_test_[i] = [4]
    elif y_test[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]: y_test_[i] = [5]
    
  y_test = torch.squeeze(torch.Tensor(np.array([[y_test_[i][0]] for i in range(len(y_test_))])))

  B1 = np.load(datapath+'B1.npy')
  B2 = np.load(datapath+'B2.npy')
  G = load_variable(datapath+'G_undir.pkl')

  N0 = (abs(B1@B1.T).shape)[0]
  N1 = (abs(B2@B2.T).shape)[0]
  N2 = (abs(B2.T@B2).shape)[0]

  sm = torch.nn.Softmax(dim=1)

  edges = np.array([np.where(x)[0] for x in B1.T if len(np.where(x!=0)[0])==2])
  triangles = np.array([np.where(x)[0] for x in B2.T if len(np.where(x!=0)[0])==3])
  triangles = _get_triangles(edges,triangles)
  L_d = B1.T@B1
  L_u = B2@B2.T
  st = gudhi.SimplexTree()
  for i in triangles:
      st.insert(i)
  for i in edges:
      st.insert(i)
  for i in range(B1.shape[0]):
      st.insert([i])
  simplex_tree, sc, indices = process_simplex_tree(st, B1.shape[0])
  feature_size = X.shape[1]
  X = np.squeeze(X)
  model = MPSN_L(feature_size, feature_size, feature_size, 3, agg='sum').to(args.device).float()
  opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
  X = torch.from_numpy(X).to(args.device).float()
  for epoch in range(args.epochs):
          start_epoch = time.time()
          model.train()
          opt.zero_grad()
          st1, sc1, bm1,ind1 = augment_simplex_tp(simplex_tree, sc, [])
          st2, sc2, bm2,ind2 = augment_simplex_tp(simplex_tree, sc, [])
          
          l1, l1_d, l1_u = _get_laplacians(bm1)
          l2, l2_d, l2_u = _get_laplacians(bm2)
          
          W = get_weight_matrix_simplex(1, sc1, sc2, ind1, ind2, l1)
          W = sm(torch.FloatTensor(W)).to(args.device)
          W = W * (W!=W.min(axis=1).values.unsqueeze(-1))
          outputs1 = model(X,l1_u[1],l1_d[1])
          outputs2 = model(X,l2_u[1],l2_d[1])
          del(st1, sc1, bm1,ind1, l1, l1_d, l1_u, st2, sc2, bm2,ind2, l2, l2_d, l2_u)
          torch.cuda.empty_cache()
          gc.collect()
          loss = calculate_loss_tp(outputs1, outputs2, args.alpha, [W], args.delta, args.device)
          loss.backward(retain_graph=True)
          opt.step()
          print(f"At epoch:{epoch+1},\tTime: {time.time() - start_epoch}\t Loss:{loss.item()}")
          del(outputs1, outputs2, W, loss)
          torch.cuda.empty_cache()
          gc.collect()
  X = model(X, torch.from_numpy(L_u).float().to(args.device), torch.from_numpy(L_d).float().to(args.device)).cpu().detach().numpy()
  X_tr = X[train_mask!=0]
  X_test = X[test_mask!=0]
  x1_0 = np.squeeze(X_tr)
  x1_0_test = np.squeeze(X_test)

  Z_ = []     
  for l in range(len(last_nodes)): #200 for buoy 
      i = last_nodes[l]
      Z__ = np.zeros((B1.shape[0]))
      Z__[[int(j) for j in G.neighbors(i)]]=1
      Z_.append(list(Z__))
  Z_ = np.array(Z_)
  Z_tr_ = Z_[train_mask!=0]
  Z_test = Z_[test_mask!=0]

  X_train = x1_0@B1.T*Z_tr_
  X_test = x1_0_test@B1.T*Z_test
  classifier = RidgeClassifier()
  classifier.fit(X_train, y_tr.cpu().detach().numpy())
  print(classification_report(y_test.cpu().detach().numpy(), classifier.predict(X_test)))