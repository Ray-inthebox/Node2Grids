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

#choose top first-order neighbor
def topNeighbor1(keys, values, k, rowRank):
    nodetodegree = dict(map(lambda x, y: [x, y], keys, values))
    for i in range(len(nodetodegree)):
        max_degree_key = max(nodetodegree, key=lambda j: nodetodegree[j])
        nodetodegree.pop(max_degree_key)
        if max_degree_key not in rowRank:
            rowRank.append(max_degree_key)
            if len(rowRank) == k:
                return rowRank
    return rowRank

#top second-order neighbor
def topNeighbor2(keys, values, k, rowRank, centralnode):
    nodetodegree = dict(map(lambda x, y: [x, y], keys, values))
    for i in range(len(nodetodegree)):
        max_degree_key = max(nodetodegree, key=lambda j: nodetodegree[j])
        nodetodegree.pop(max_degree_key)
        if max_degree_key not in rowRank:
            if max_degree_key != centralnode:
                rowRank.append(max_degree_key)
                if len(rowRank) == k:
                    return rowRank
    return rowRank


# find topk (first-order or second-order) neighbor
def findTopK(G, start, end, k, nodesdegree):
    centralnode = start
    allNeighbor = []
    while (centralnode < end):
        neighbor1 = list(G.neighbors(centralnode))
        chooseneighbor1 = []
        for node in neighbor1:
            degree = nodesdegree[node]
            chooseval = degree
            chooseneighbor1.append(chooseval)
        if (len(neighbor1) >= k):
            NeighborSorted = topNeighbor1(neighbor1, chooseneighbor1, k, [])
            allNeighbor.append(NeighborSorted)
            centralnode = centralnode + 1
            continue
        else:
            NeighborSorted = topNeighbor2(neighbor1, chooseneighbor1, k, [], centralnode)
            neighbor2 = []
            rowsumneighbor2 = []
            for node in neighbor1:
                neighbor = list(G.neighbors(node))
                neighbor2.append(neighbor)
            neighbor2 = itertools.chain.from_iterable(neighbor2)
            neighbor2 = list(neighbor2)
            neighbor2 = list(set(neighbor2))
            for item in neighbor2:
                degree = nodesdegree[item]
                chooseval = degree
                rowsumneighbor2.append(chooseval)
            NeighborSorted = topNeighbor2(neighbor2, rowsumneighbor2, k, NeighborSorted, centralnode)
            l = list(set(NeighborSorted))
            l.sort(key=NeighborSorted.index)
            if centralnode in l:
                l.remove(centralnode)
            while len(l) < k:
                l.append(-1)
            l = l[:k]
            allNeighbor.append(l)
            centralnode = centralnode + 1
    return allNeighbor

# create grid-like map
@nb.njit(nopython=True, cache=True, nogil=True, fastmath=True)
def createMap(Topk, feas, biasfactor, startnode, mapsize_a, mapsize_b):
    Map = np.zeros(shape=(len(Topk), mapsize_a, mapsize_b, feas.shape[1]))
    j = 0
    for i in range(len(Topk)):
        pos = 0
        centralnode = j + startnode
        temp_map = np.zeros(shape=(1, mapsize_a * mapsize_b, feas.shape[1]))
        for item in Topk[i]:
            if (item != -1 and pos < mapsize_a * mapsize_b):
                for h in range(feas.shape[1]):
                    temp_map[0][pos][h] = (1 - biasfactor) * feas[item, h] + biasfactor * feas[centralnode, h]
                pos += 1
        map = np.reshape(temp_map, (mapsize_a, mapsize_b, feas.shape[1]))
        Map[j] = map
        j += 1
    return Map

