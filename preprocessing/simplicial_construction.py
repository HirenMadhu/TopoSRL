import numpy as np
from tqdm import tqdm
import copy
import gudhi
import torch
device = torch.device("cuda:1")

def process_simplex_tree(simplex_tree, num_nodes):
    sc = [list() for _ in range(simplex_tree.dimension()+1)]
    sc[0] = [[i] for i in range(num_nodes)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        sc[len(simplex)-1].append(np.array(simplex))
    indices = []
    for i in range(len(sc)):
        sc[i] = np.array(sc[i])
        sc[i] = np.unique(sc[i],axis=0)
        index = {}
        for k,j in enumerate(sc[i]):
            index[frozenset(j)] = k
        indices.append(index)
    sc[0] = np.unique(np.array(sc[0]),axis=0)
    return simplex_tree, sc, indices

def get_simplicies_closure(data,dim):
    with open('data/closure/'+ data + '/' + data +'-nverts.txt', 'r') as f:
        nverts = np.array(list(map(lambda x:int(x), f.readlines())))
    with open('data/closure/'+ data + '/' + data +'-simplices.txt', 'r') as f:
        simplices = np.array(list(map(lambda x:int(x), f.readlines())))
    print(simplices.min())
    train = np.load('data/closure/'+ data + '/' + data +'-train.npy', allow_pickle=False)
    test = np.load('data/closure/'+ data + '/' + data +'-test.npy', allow_pickle=False)
    bipartite_graph = np.zeros((len(nverts), np.max(simplices))).astype('bool')
    seen = 0
    for i in range(len(nverts)):
        nodes = []
        for j in range(seen, seen+nverts[i]):
            nodes.append(simplices[j]-1)
        bipartite_graph[i,nodes]=1
        seen+=nverts[i]

    simplex_tree_train = gudhi.SimplexTree()
    for i in bipartite_graph[:int(len(bipartite_graph)*0.8)]:
        simplex = np.where(i)[0]
        if(len(simplex)<=dim+1):
            simplex_tree_train.insert(simplex)
    simplex_tree_train.prune_above_dimension(dim)
    st_train, sc_train, indices_train = process_simplex_tree(simplex_tree_train, simplices.max())

    simplex_tree_test = gudhi.SimplexTree()
    for i in bipartite_graph:
        simplex = np.where(i)[0]
        if(len(simplex)<=dim+1):
            simplex_tree_test.insert(simplex)
    simplex_tree_test.prune_above_dimension(dim)
    st_test, sc_test, indices_test = process_simplex_tree(simplex_tree_test, simplices.max())


    return st_train, sc_train, indices_train, st_test, sc_test, indices_test, train, test

def get_simplicies(data,dim):
    with open('data/node classification/node-labels-'+ data +'.txt', 'r') as f:
        labels = np.array(list(map(lambda x:int(x), f.readlines())))
    print("Read the labels")
    simplicies = []
    num_nodes = 0
    print("Reading the simplicies")
    with open('data/node classification/hyperedges-'+ data +'.txt', 'r') as f:
        for i in f.readlines():
            simplicies.append(np.array([int(y) for y in i[:-1].split(',')]))
            if(simplicies[-1].max()>num_nodes):
                num_nodes = simplicies[-1].max()
    simplex_tree = gudhi.SimplexTree()
    print("Creating tree")
    for i in simplicies:
        if len(i)<=dim+1:
            simplex_tree.insert(np.array(i)-1)
    simplex_tree.prune_above_dimension(dim)
    simplex_tree, sc, indices = process_simplex_tree(simplex_tree, num_nodes)
    return simplex_tree, sc, indices, labels

def get_boundary_matrices(data, dim):
  simplex_tree, sc, indices, labels = get_simplicies(data, dim)
  boundry_matrices = []
  for i in range(1,dim+1):
      print(f"Computing boundary matrice for dimension {i}")
      boundry_matrix = np.zeros((len(sc[i-1]),len(sc[i])), dtype=bool)
      for m,j in enumerate(sc[i]):
        idx = np.arange(1, i+1) - np.tri(i+1, i, k=-1, dtype=bool)
        for k in idx:
            boundry_matrix[indices[i-1][frozenset(j[k])], indices[i][frozenset(j)]] = 1
      boundry_matrices.append(boundry_matrix)
  return simplex_tree, sc[:dim], boundry_matrices, labels

def get_boundary_matrices_from_processed_tree(simplex_tree, sc, indices, dim):
  boundry_matrices = []
  for i in range(1,dim+1):
      boundry_matrix = np.zeros((len(sc[i-1]),len(sc[i])), dtype=bool)
      for m,j in enumerate(sc[i]):
        idx = np.arange(1, i+1) - np.tri(i+1, i, k=-1, dtype=bool)
        for k in idx:
            boundry_matrix[indices[i-1][frozenset(j[k])], indices[i][frozenset(j)]] = 1
      boundry_matrices.append(boundry_matrix)
  return simplex_tree, sc[:dim], boundry_matrices

def _get_simplex_features(simplicies,features):
    X = []
    X.append(features)
    for i in simplicies:
        X.append(features[i].sum(axis=1).clip(0,1))
    return X

def _get_laplacians(boundary_matrices):
  for i,k in enumerate(boundary_matrices):
    boundary_matrices[i] = torch.FloatTensor(k).to(device)
  laplacians = [None]*len(boundary_matrices)
  laplacians[0] = boundary_matrices[0]@boundary_matrices[0].T
  laplacians[-1] = boundary_matrices[-2].T@boundary_matrices[-2]
  for i in range(1,len(boundary_matrices)-1):
    laplacians[i] = boundary_matrices[i-1].T@boundary_matrices[i-1] + boundary_matrices[i]@boundary_matrices[i].T
  lower_laplacians = [None]*len(boundary_matrices)
  upper_laplacians = [None]*len(boundary_matrices)
  for i in range(1,len(boundary_matrices)):
    lower_laplacians[i] = boundary_matrices[i-1].T@boundary_matrices[i-1]
  for i in range(0,len(boundary_matrices)-1):
    upper_laplacians[i] = boundary_matrices[i]@boundary_matrices[i].T
  return laplacians, lower_laplacians, upper_laplacians

def get_neighbors(A, return_dict = False):
    one_hop = []
    for i in A:
        try:
            one_hop.append(np.where(i)[1])
        except IndexError:
            one_hop.append(np.where(i)[0])
    if return_dict:
        d = {}
        for i in range(len(A)):
            d[i] = tuple(one_hop[i])
        return d
    return one_hop

def get_neighbors_laplacians(L):
    one_hop = []
    for i in L:
        one_hop.append(np.where(i)[0])
    return one_hop

def augment_simplex(simplex_tree, sc, rho):
    st1 = simplex_tree.__deepcopy__()
    sc1 = copy.deepcopy(sc)
    st1.prune_above_dimension(4)
    delete_mask = np.random.binomial(1, rho, len(sc1[2]))
    delete_idx = np.where(delete_mask)[0]
    sc1[2] = np.delete(sc1[2] ,delete_idx, axis=0)
    sc1[2]= np.append(sc1[2] , np.random.randint(1,len(sc[0]), (len(sc[2])//10,3)),axis=0)
    for i in sc1[2]:
        st1.insert(i)
    st1, sc1, ind1 = process_simplex_tree(st1, len(sc[0])-1)
    st1, sc1, bm1 = get_boundary_matrices_from_processed_tree(st1,sc1,ind1,3)
    return st1,sc1,bm1,ind1

def augment_simplex_open(simplex_tree, sc, rho, open_triangles):
    st1 = simplex_tree.__deepcopy__()
    sc1 = copy.deepcopy(sc)
    st1.prune_above_dimension(3)
    add_mask = np.random.binomial(1, rho, len(open_triangles))
    add_idx = np.where(add_mask)[0]
    sc1[2]= np.append(sc1[2] , open_triangles[add_idx],axis=0)
    for i in sc1[2]:
        st1.insert(i)
    for i in sc[3]:
        st1.insert(i)
    st1, sc1, ind1 = process_simplex_tree(st1, len(sc[0])-1)
    st1, sc1, bm1 = get_boundary_matrices_from_processed_tree(st1,sc1,ind1,3)
    return st1,sc1,bm1,ind1

def get_weight_matrix_graph(A):
    W0 = np.zeros(A.shape)
    np.fill_diagonal(W0,3)
    one_hop_neighbors = get_neighbors(A)
    for k,row in enumerate(one_hop_neighbors):
        W0[k][row] = 2
        neighborhood = np.array([])
        two_hops = []
        for i in row:
            two_hops.append(one_hop_neighbors[i])
        for i in two_hops:
            neighborhood = np.append(neighborhood,i)
        neighbors = np.unique(neighborhood)
        neighbors = np.delete(neighbors, np.where(neighbors==k))
        neighbors = np.setdiff1d(neighbors, row).astype(int)
        W0[k][neighbors] = 1
    return W0

def get_weight_matrix_simplex(dimension, sc1, sc2, ind1, ind2, l1):
    W = np.zeros((len(sc1[dimension]), len(sc2[dimension])))
    sc1_set, sc2_set = set(map(frozenset, sc1[dimension])), set(map(frozenset, sc2[dimension]))
    intersection = sc1_set & sc2_set
    
    for i in intersection:
        W[ind1[dimension][i], ind2[dimension][i]] = 3

    one_hop_neighbors = get_neighbors_laplacians(l1[dimension].cpu().detach().numpy())
    
    for k, row in enumerate(one_hop_neighbors):
        row = np.delete(row, row==k)
        row_set = set(map(frozenset, sc1[dimension][row]))
        neighbors = row_set & sc2_set
        for i in neighbors:
            W[k, ind2[dimension][i]] = 2
    return W

def generate_triangles(nodes):
    """Generate triangles. Weed out duplicates."""
    visited_ids = set() 
    triangles = []
    for node_a_id in nodes:
        for node_b_id in nodes[node_a_id]:
            if node_b_id == node_a_id:
                raise ValueError # nodes shouldn't point to themselves
            if node_b_id in visited_ids:
                continue # we should have already found b->a->??->b
            for node_c_id in nodes[node_b_id]:
                if node_c_id in visited_ids:
                    continue # we should have already found c->a->b->c
                if node_a_id in nodes[node_c_id]:
                    triangles.append(sorted([node_a_id, node_b_id, node_c_id]))
        visited_ids.add(node_a_id) # don't search a - we already have all those cycles
    return np.unique(np.array(triangles), axis=1)

def augment_simplex_open_gc(simplex_tree, sc, open_triangles):
    st1 = simplex_tree.__deepcopy__()
    sc1 = copy.deepcopy(sc)
    if(len(sc[1])>=10):
        st1.prune_above_dimension(3)
        sc1[2] = np.delete(sc1[2] ,np.random.randint(0,len(sc[2]), int(len(sc[2])//10)), axis=0)
        for i in sc1[2]:
            st1.insert(i)
    st1, sc1, ind1 = process_simplex_tree(st1, len(sc[0]))
    st1, _sc1, bm1 = get_boundary_matrices_from_processed_tree(st1,sc1,ind1,3)
    return st1,sc1,bm1,ind1
def _get_simplex_features_gc(simplicies,features):
    X = []
    X.append(features)
    for i in simplicies:
        if(i.max()==len(features)):
          X.append(features[torch.tensor(i)-1].sum(axis=1))
        else:
          X.append(features[torch.tensor(i)].sum(axis=1))
    return X
def augment_simplex_tp(simplex_tree, sc, open_triangles):
    st1 = simplex_tree.__deepcopy__()
    sc1 = copy.deepcopy(sc)
    st1.prune_above_dimension(3)
    sc1[2] = np.delete(sc1[2] ,np.random.randint(0,len(sc[2]), int(len(sc[2])//10)), axis=0)
    for i in sc1[2]:
        st1.insert(i)
    st1, sc1, ind1 = process_simplex_tree(st1, len(sc[0]))
    st1, sc1, bm1 = get_boundary_matrices_from_processed_tree(st1,sc1,ind1,2)
    return st1,sc1,bm1,ind1