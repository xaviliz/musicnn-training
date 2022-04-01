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


def test_get_metrics_for_regression():

    predicted_0 = [
        [2.2, 4.3],
        [1.0, 8.41],
        [3.2, 9.4],
        [5.8, 7.1],
        [3.7, 5.6],
    ]
    groundtruth_0 = [
        [3.0, 4.1],
        [0.5, 10.9],
        [3.7, 9.9],
        [6.5, 6.0],
        [4.5, 4.9],
    ]
    predicted_1 = [
        [4.5, 9.8],
        [5.2, 7.21],
        [1.1, 5.1],
        [7.8, 0.5],
        [8.6, 3.7],
    ]
    groundtruth_1 = [
        [3.0, 10.0],
        [5.0, 8.0],
        [2.5, 6.3],
        [8.5, 0.5],
        [8.3, 2.5],
    ]

    y_pred = predicted_0.copy()
    y_pred.extend(predicted_1)

    y_true = groundtruth_0.copy()
    y_true.extend(groundtruth_1)

    fold_pred = [predicted_0, predicted_1]
    fold_gt = [groundtruth_0, groundtruth_1]

    task_type = "regression"
    metrics, report = get_metrics(
        y_true, y_pred, fold_gt, fold_pred, n_folds=2, task_type=task_type
    )

    (p_corr_score, ccc_score, r2_score, adjusted_r2_score, rmse_score, micro_metrics) = metrics

    np.testing.assert_allclose(micro_metrics["p_corr"], [0.94601526, 0.95196423])
    np.testing.assert_allclose(p_corr_score.std, [0.02602254, 0.02779262], rtol=1e-06)

    np.testing.assert_allclose(ccc_score.mean, [1.21135712, 0.73753691])
    np.testing.assert_allclose(r2_score.mean, [0.8925784, 0.89345496])
    np.testing.assert_allclose(adjusted_r2_score.mean, [0.7851568, 0.78690992])
    np.testing.assert_allclose(rmse_score.mean, [0.71, 1.17342])

