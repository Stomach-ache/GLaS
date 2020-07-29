#cython: boundscheck=False, wraparound=False
cimport cython
import numpy as np
cimport numpy as np
import random
from scipy.sparse import csr_matrix
from cython.parallel import prange
from sklearn.svm import LinearSVC

from libc.math cimport log, abs, exp, pow, sqrt
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdlib cimport malloc, free
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool
from libcpp.algorithm cimport sort as stdsort
from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef pair[vector[int],vector[int]] LR_SET
ctypedef pair[int,int] I_PAIR
ctypedef pair[int,float] DP
ctypedef vector[I_PAIR] COUNTER
ctypedef vector[vector[int]] YSET
ctypedef vector[pair[int,float]] SR
ctypedef vector[SR] CSR
ctypedef vector[vector[float]] Mat

cdef precision_at_k(ground_truth, predictions, k=5, pos_label=1):
    #assert len(ground_truth) == len(predictions), "P@k: Length mismatch"

    #desc_order = np.argsort(predictions)[::-1]
    #ground_truth = np.take(ground_truth, desc_order[:k])
    ground_truth = np.take(ground_truth, predictions[:k])
    relevant_preds = (ground_truth == pos_label).sum()

    return relevant_preds / k

cdef dcg_score_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
        ground_truth : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        DCG @ k for a given ground truth - prediction pair.
    """
    #assert len(ground_truth) == len(predictions), "DCG@k: Length mismatch"

    #desc_order = np.argsort(predictions)[::-1]  # ::-1 reverses array
    #ground_truth = np.take(ground_truth, desc_order[:k])  # the top indices
    ground_truth = np.take(ground_truth, predictions[:k])
    gains = 2 ** ground_truth - 1

    discounts = np.log2(np.arange(1, len(ground_truth) + 1) + 1)
    return np.sum(gains / discounts)

cdef ndcg_score_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
    Function to evaluate the Discounted Cumulative Gain @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        NDCG @ k for a given ground truth - prediction pair.
    """
    dcg_at_k = dcg_score_at_k(ground_truth, predictions, k, pos_label)
    #best_dcg_at_k = dcg_score_at_k(ground_truth, ground_truth, k, pos_label)
    if sum(ground_truth) == 0:
        return 0
    else:
        logs = 1.0 / np.log2(np.arange(min(k, sum(ground_truth))) + 2)
        return dcg_at_k / sum(logs)


cpdef evaluations(ground_truth, predictions, k=5, pos_label=1):
    cdef int i = 0, n = len(predictions)

    #assert n == ground_truth.shape[0], 'shape mismatch'

    cdef float prec1 = 0, prec3 = 0, prec5 = 0
    cdef float ndcg1 = 0, ndcg3 = 0, ndcg5 = 0
    for i in range(n):
        prec1 += precision_at_k(ground_truth[i].toarray()[0], predictions[i], 1)
        #prec3 += precision_at_k(ground_truth[i].toarray()[0], predictions[i], 3)
        #prec5 += precision_at_k(ground_truth[i].toarray()[0], predictions[i], 5)

        ndcg1 += ndcg_score_at_k(ground_truth[i].toarray()[0], predictions[i], 1)
        #ndcg3 += ndcg_score_at_k(ground_truth[i].toarray()[0], predictions[i], 3)
        #ndcg5 += ndcg_score_at_k(ground_truth[i].toarray()[0], predictions[i], 5)

    prec1 /= n
    #prec3 /= n
    #prec5 /= n

    ndcg1 /= n
    #ndcg3 /= n
    #ndcg5 /= n

    #return (prec1, prec3, prec5), (ndcg1, ndcg3, ndcg5)
    return prec1, ndcg1

cpdef input_dropout(object A, rho=0.2):
    cdef int i, j = 0
    cdef vector[int] ind = A.indices, indptr = A.indptr
    cdef int n = indptr.size() - 1

    for i in range(n):
        while j < indptr[i + 1]:
            if random.random() < rho:
                A.data[j] = 0
            j += 1
    A.eliminate_zeros()
    return A

cpdef (vector[int], vector[int]) stocastic_negative_sampling(object Y, int max_num_pairs=10):
    cdef int i, j, k, p
    cdef int n = Y.shape[0], m = Y.shape[1]
    cdef vector[int] pos_indices, neg_indices

    for i in range(n):
        pos = Y[i].indices
        neg = list(set(range(m)) - set(pos))
        #pos = np.random.permutation(pos)
        while max_num_pairs > 0:
            max_num_pairs -= 1
            neg = np.random.permutation(neg)
            for p, k in zip(pos, neg):
                pos_indices.push_back(p + i * m)
                neg_indices.push_back(k + i * m)
    return pos_indices, neg_indices
