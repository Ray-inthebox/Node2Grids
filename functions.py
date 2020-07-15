# coding:utf-8

import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import itertools
import numba as nb
import numpy as np


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask"""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_small_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features = sp.lil_matrix(features)
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask

def findTopK(adj:sp.lil_matrix, idx:list, k:int, nodesdegree:np.ndarray,nums_worker = 1):
    jobs = []
    res = [0]*nums_worker
    queue = mp.Queue()
    idxs = split_list_n_list(idx,nums_worker)
    i=0
    for idx in idxs:
        p = Process(target = _findTopK,args = (adj,idx,k,nodesdegree,queue,i))
        i+=1
        jobs.append(p)
    for p in jobs:
        p.start()
    for p in jobs:
        num,val = queue.get()
        res[num]=val
    return np.concatenate(np.array(res),axis=0)

# find topk (first-order or second-order) neighbor
def _findTopK(adj:sp.lil_matrix, idx:list, k:int, nodesdegree:np.ndarray,queue:mp.Queue(),series:int):
    allNeigbor = []
    for node in idx:
        sortedNeighbor = []
        r1_node = sp.find(adj[node])[1]
        r1_degrees = nodesdegree[r1_node]
        sortidx = r1_degrees.argsort()[::-1]
        r1_node = r1_node[sortidx]
        if len(r1_node)<k:
            # find r2 node
            r2_node = set()
            for candidate in r1_node:
                r2_node.update(sp.find(adj[candidate])[1])
            # difference between r2 and r1
            if r2_node:
                r2_node = r2_node-set(list(r1_node))
                if node in r2_node:
                    r2_node.remove(node)
            if r2_node:
                r2_node = np.array(list(r2_node))
                r2_degrees = nodesdegree[r2_node]
                sortidx = r2_degrees.argsort()[::-1]
                r2_node = r2_node[sortidx]
                if len(r2_node)<k-len(r1_node):
                    temp = np.full((1,k),node)[0]
                    r2_node = np.hstack((r2_node,temp)) 
            else:
                r2_node = np.full((1,k),node)[0]
            sortedNeighbor = np.hstack((r1_node,r2_node))[:k]
        else:
            sortedNeighbor = r1_node[:k]
        allNeigbor.append(sortedNeighbor)
    
    queue.put((series,allNeigbor))

# create grid-like map
@nb.njit(cache=True, nogil=True, fastmath=True)
def _createMap(Topk, feas,  idx, biasfactor, mapsize_a, mapsize_b):
    Map = np.zeros(shape=(len(Topk), mapsize_a, mapsize_b, feas.shape[1]),dtype=np.float32)
    for i,centralNode in enumerate(idx):
        temp_map = np.zeros(shape=(1, mapsize_a * mapsize_b, feas.shape[1]))
        temp_map[0] = (1 - biasfactor) * feas[Topk[i],:]+biasfactor * feas[centralNode]
        Map[i] = np.reshape(temp_map, (mapsize_a, mapsize_b, feas.shape[1]))
    return Map

