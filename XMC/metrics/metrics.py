import numpy as np
import torch

def ps_precision_at_k(ground_truth, predictions, k=5, pos_label=1, device='cpu', wts=None):
    assert wts is not None
    #assert ground_truth.size() == predictions.size(), "PSP@k: Length mismatch"
    assert ground_truth.size(1) == len(wts)

    if ground_truth.size() == predictions.size():
        desc_order = torch.argsort(predictions, dim=1, descending=True)[:, :k]
    else:
        desc_order = predictions[:, :k]

    den = torch.argsort(torch.mul(ground_truth, wts), dim=1, descending=True)[:, :k].sum()

    ground_truth = torch.gather(ground_truth, 1, desc_order)  # the top indices
    discounts = wts[desc_order]
    #print (ground_truth)
    #print (discounts)
    relevant_preds = torch.mul(ground_truth, discounts).sum() / den
    print (relevant_preds / ground_truth.size(0))

    return relevant_preds / k

def precision_at_k(ground_truth, predictions, k=5, pos_label=1, device='cpu'):
    """
    Function to evaluate the precision @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of
                       label vector
        predictions : np.array consisting of predictive probabilities
                      for every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        precision @ k for a given ground truth - prediction pair.
    """
    #assert ground_truth.size() == predictions.size(), "P@k: Length mismatch"

    if ground_truth.size() == predictions.size():
        desc_order = torch.argsort(predictions, dim=1, descending=True) # ::-1 reverses array
    else:
        desc_order = predictions
    #print("The top scores for predictions", predictions[desc_order[0]])
    ground_truth = torch.gather(ground_truth, 1, desc_order[:, :k])  # the top indices
    relevant_preds = (ground_truth == pos_label).sum()

    return 1.0 * relevant_preds / k


def dcg_score_at_k(ground_truth, predictions, k=5, pos_label=1, device='cpu'):
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
        DCG @ k for a given ground truth - prediction pair.
    """
    #assert ground_truth.size() == predictions.size(), "P@k: Length mismatch"

    if ground_truth.size() == predictions.size():
        desc_order = torch.argsort(predictions, dim=1, descending=True) # ::-1 reverses array
    else:
        desc_order = predictions
    ground_truth = torch.gather(ground_truth, 1, desc_order[:, :k])  # the top indices
    gains = torch.pow(2.0, ground_truth) - 1

    discounts = torch.log2(torch.arange(1, ground_truth.size(1) + 1) + 1.0)
    return torch.sum(torch.div(gains, discounts.to(device=device)), dim=1)


def ndcg_score_at_k(ground_truth, predictions, k=5, pos_label=1, device='cpu'):
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
    dcg_at_k = dcg_score_at_k(ground_truth, predictions, k, pos_label, device)
    best_dcg_at_k = dcg_score_at_k(ground_truth, ground_truth, k, pos_label, device)
    dcg_at_k[(best_dcg_at_k == 0).nonzero()] = 0
    best_dcg_at_k[(best_dcg_at_k == 0).nonzero()] = 1
    #if best_dcg_at_k == 0:
    #    return 0
    #else:
    return torch.div(dcg_at_k, best_dcg_at_k).sum()
