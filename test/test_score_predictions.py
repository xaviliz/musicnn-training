import numpy as np

from score_predictions import get_metrics


def test_get_metrics():
    predicted_0 = [
        [0.2, 0.3, 0.5],
        [0.2, 0.41, 0.39],
        [0.2, 0.1, 0.7],
        [0.8, 0.1, 0.1],
        [0.7, 0.6, 0.3],
    ]
    groundtruth_0 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
    ]
    predicted_1 = [
        [0.4, 0.1, 0.5],
        [0.2, 0.21, 0.59],
        [0.1, 0.1, 0.8],
        [0.1, 0.1, 0.8],
        [0.6, 0.7, 0.3],
    ]
    groundtruth_1 = [
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]

    y_pred = predicted_0.copy()
    y_pred.extend(predicted_1)

    y_true = groundtruth_0.copy()
    y_true.extend(groundtruth_1)

    fold_pred = [predicted_0, predicted_1]
    fold_gt = [groundtruth_0, groundtruth_1]

    roc_auc_score, pr_auc_score, macro_acc_score, micro_acc, report = get_metrics(
        y_true, y_pred, fold_gt, fold_pred, n_folds=2
    )
    np.testing.assert_allclose(micro_acc, 0.4)
    np.testing.assert_allclose(macro_acc_score.std, 0.2)
    assert not np.isnan(roc_auc_score.mean)
    assert not np.isnan(pr_auc_score.mean)
