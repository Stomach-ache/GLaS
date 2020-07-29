# -*- coding: utf-8 -*-

import hnswlib
import numpy as np

def buildIndex(X):
    dim = X.shape[1]
    num_elements = X.shape[0]
    data_labels = np.arange(num_elements)
    p = hnswlib.Index(space = 'cosine', dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
    p.add_items(X, data_labels)
    p.set_ef(5)
    return p

def searchIndex(p, X, k=5):
    labels, distances = p.knn_query(X, k = k)
    return labels

