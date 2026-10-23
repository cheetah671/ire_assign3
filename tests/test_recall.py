import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann.metrics import exact_distances, recall_at_k  # noqa: E402


def gt_row(n=10):
    return np.arange(n)[None, :]


def test_perfect_match():
    assert recall_at_k(gt_row(), gt_row()) == 1.0


def test_order_inside_top_k_does_not_matter():
    assert recall_at_k(gt_row()[:, ::-1], gt_row()) == 1.0


def test_disjoint():
    assert recall_at_k(gt_row() + 100, gt_row()) == 0.0


def test_partial():
    pred = np.array([[0, 1, 2, 3, 4, 50, 51, 52, 53, 54]])
    assert recall_at_k(pred, gt_row()) == 0.5


def test_only_first_k_predictions_count():
    # the true neighbours 5..9 are returned, but only after position 10
    pred = np.array([[0, 1, 2, 3, 4, 50, 51, 52, 53, 54, 5, 6, 7, 8, 9]])
    assert recall_at_k(pred, gt_row(), k=10) == 0.5


def test_padding_minus_one_not_counted():
    pred = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1]])
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, -1]])  # even if gt had -1
    assert recall_at_k(pred, gt) == 0.5


def test_duplicates_not_double_counted():
    pred = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert recall_at_k(pred, gt_row()) == 0.1


def test_tie_at_kth_neighbour_counts_as_correct():
    # ids 9 and 99 are both at distance 9.0 from the query; exact search
    # listed 9, the index returned 99. That is not a mistake.
    gt_d = np.arange(10, dtype=np.float32)[None, :]
    pred = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 99]])
    pred_d = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9.0]], dtype=np.float32)
    assert recall_at_k(pred, gt_row(), pred_dists=pred_d, gt_dists=gt_d) == 1.0
    # without distances the tie cannot be seen
    assert recall_at_k(pred, gt_row()) == 0.9


def test_farther_than_kth_is_a_miss():
    gt_d = np.arange(10, dtype=np.float32)[None, :]
    pred = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 99]])
    pred_d = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9.5]], dtype=np.float32)
    assert recall_at_k(pred, gt_row(), pred_dists=pred_d, gt_dists=gt_d) == 0.9


def test_per_query_and_mean():
    pred = np.vstack([gt_row()[0], gt_row()[0] + 100])
    gt = np.vstack([gt_row()[0], gt_row()[0]])
    np.testing.assert_allclose(recall_at_k(pred, gt, per_query=True), [1.0, 0.0])
    assert recall_at_k(pred, gt) == 0.5


def test_too_few_predictions_raises():
    with pytest.raises(ValueError):
        recall_at_k(gt_row(5), gt_row(), k=10)


def test_exact_distances_and_ties_end_to_end():
    # 1-D points: query at 0, base points at 1, -1, 2, -2, 3. Points 0 and 1
    # are tied as nearest neighbours.
    xb = np.array([[1.0], [-1.0], [2.0], [-2.0], [3.0]], dtype=np.float32)
    xq = np.zeros((1, 1), dtype=np.float32)
    gt_i = np.array([[0, 1]])
    gt_d = exact_distances(xq, xb, gt_i)
    np.testing.assert_allclose(gt_d, [[1.0, 1.0]])
    pred = np.array([[1, 0]])
    pd = exact_distances(xq, xb, pred)
    assert recall_at_k(pred, gt_i, k=2, pred_dists=pd, gt_dists=gt_d) == 1.0
    pred = np.array([[1, 2]])  # 2 is at distance 4 > 1: a miss
    pd = exact_distances(xq, xb, pred)
    assert recall_at_k(pred, gt_i, k=2, pred_dists=pd, gt_dists=gt_d) == 0.5
    # padding gets infinite distance
    assert np.isinf(exact_distances(xq, xb, np.array([[-1]])))[0, 0]


def test_candidate_recall():
    from ann.metrics import candidate_recall
    # 1-D: query at 0, base points at distance 1..6
    xb = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float32)
    xq = np.zeros((1, 1), dtype=np.float32)
    gt_d = np.array([[1.0, 4.0]])  # squared distances of the true top 2
    # top-2 found among 4 candidates in the wrong order -> 1.0
    assert candidate_recall(np.array([[3, 1, 0, 5]]), xq, xb, gt_d, k=2) == 1.0
    # only one of them present, plus duplicates and padding -> 0.5
    assert candidate_recall(np.array([[0, 0, -1, 4]]), xq, xb, gt_d, k=2) == 0.5
