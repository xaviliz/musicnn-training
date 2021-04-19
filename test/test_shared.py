import numpy as np

import shared


def test_auc():
    """
    Test combined calculation of PR-AUC and ROC-AUC
    """
    predicted = [
        [0.2, 0.3, 0.5],
        [0.2, 0.41, 0.39],
        [0.2, 0.1, 0.7],
        [0.8, 0.1, 0.1],
        [0.4, 0.3, 0.3],
    ]
    groundtruth = [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
    ]
    roc_auc, pr_auc = shared.compute_auc(groundtruth, predicted)
    # These values were computed using default scikit-learn parameters
    np.testing.assert_allclose(roc_auc, 0.8611111)
    np.testing.assert_allclose(pr_auc, 0.8444444)


def test_type_of_groundtruth():
    assert shared.type_of_groundtruth([1, -1, -1, 1]) == "binary"
    assert (
        shared.type_of_groundtruth(np.array([[0, 1], [1, 1]])) == "multilabel-indicator"
    )
    assert (
        shared.type_of_groundtruth(np.array([[0, 1], [1, 0]])) == "multiclass-indicator"
    )


def test_compute_accuracy():
    y_pred = [
        [0.2, 0.3, 0.5],  # fp 0, 0, 1
        [0.8, 0.1, 0.1],  # tp 1, 0, 0
        [0.4, 0.3, 0.3],  # fp 1, 0, 0
        [0.2, 0.41, 0.39],  # tp 0, 1, 0
        [0.25, 0.4, 0.35],  # fp 1, 0, 0
        [0.2, 0.1, 0.7],  # tp 0, 0, 1
    ]
    y_true = [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]

    acc = shared.compute_accuracy(y_true, y_pred)
    np.testing.assert_allclose(acc, 0.5)
