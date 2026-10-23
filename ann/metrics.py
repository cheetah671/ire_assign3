"""Recall@k, written by hand (no library recall functions)."""
import numpy as np


def exact_distances(xq, xb, ids):
    """Exact squared L2 distance between each query and each of its candidate ids.

    ids is (nq, m); entries of -1 (FAISS padding) get +inf.
    """
    ids = np.asarray(ids)
    safe = np.where(ids < 0, 0, ids)
    diff = xb[safe] - xq[:, None, :]
    d = np.einsum("qmd,qmd->qm", diff, diff)
    d[ids < 0] = np.inf
    return d


def recall_at_k(pred_ids, gt_ids, k=10, pred_dists=None, gt_dists=None,
                rtol=1e-5, atol=1e-6, per_query=False):
    """Fraction of the true k nearest neighbours found in the first k predictions.

    pred_ids : (nq, >=k) ids returned by the index, best first. -1 = no result.
    gt_ids   : (nq, >=k) exact neighbour ids, best first.
    pred_dists, gt_dists : optional EXACT distances (same metric) for the
        predicted ids and the true neighbours. When given, ties at the k-th
        neighbour are handled: a returned id whose exact distance is no larger
        than the true k-th distance counts as correct even if the exact search
        happened to list a different id with the same distance.

    Invalid (-1) and duplicate ids never count, and only the first k
    predictions are looked at, so recall is always in [0, 1].
    """
    pred = np.asarray(pred_ids)[:, :k]
    gt = np.asarray(gt_ids)[:, :k]
    if pred.shape[1] < k:
        raise ValueError("need at least k predictions per query")

    hit = (pred[:, :, None] == gt[:, None, :]).any(axis=2)
    if pred_dists is not None and gt_dists is not None:
        kth = np.asarray(gt_dists)[:, k - 1:k]
        pd = np.asarray(pred_dists)[:, :k]
        hit |= pd <= kth * (1 + rtol) + atol
    hit &= pred >= 0

    # drop duplicates: keep only the first occurrence of each id in a row
    order = np.argsort(pred, axis=1, kind="stable")
    sorted_ids = np.take_along_axis(pred, order, axis=1)
    dup_sorted = np.zeros_like(sorted_ids, dtype=bool)
    dup_sorted[:, 1:] = sorted_ids[:, 1:] == sorted_ids[:, :-1]
    dup = np.zeros_like(dup_sorted)
    np.put_along_axis(dup, order, dup_sorted, axis=1)
    hit &= ~dup

    r = hit.sum(axis=1) / k
    return r if per_query else float(r.mean())


def recall_vs_truth(I, xq, xb, gt_D, gt_I, k=10):
    """Tie-aware recall@k of a search result I against exact ground truth."""
    pd = exact_distances(xq, xb, I[:, :k])
    return recall_at_k(I, gt_I, k=k, pred_dists=pd, gt_dists=gt_D)
