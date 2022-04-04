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


def test_compute_accuracy_multiclass():
    y_pred = [
        [0.2, 0.3, 0.5],  # fp 0, 0, 1
        [0.8, 0.1, 0.1],  # tp 1, 0, 0
        [0.4, 0.3, 0.3],  # fp 1, 0, 0
        [0.2, 0.41, 0.39],  # tp 0, 1, 0
        [0.25, 0.4, 0.35],  # fp 0, 1, 0
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


def test_compute_accuracy_multilabel():
    y_pred_ml = [
        [0.21, 0.31, 0.5],  # fp 0, 0, 1
        [0.81, 0.71, 0.1],  # tp 1, 1, 0
        [0.41, 0.31, 0.3],  # fp 1, 0, 0
        [0.2, 0.51, 0.69],  # tp 0, 1, 1
        [0.25, 0.4, 0.35],  # fp 0, 0, 0
        [0.51, 0.15, 0.7],  # tp 0, 0, 1
    ]
    y_true_ml = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]

    acc_multilabel = shared.compute_accuracy(y_true_ml, y_pred_ml)
    np.testing.assert_allclose(acc_multilabel, 0.5)


def test_compute_pearson_correlation():

    # defines predictions
    y_pred_pc = [
        [0.89, 0.11],
        [0.32, 0.59],
        [0.78, 0.39],
        [0.12, 0.85],
        [0.59, 0.73],
    ]

    # define groundtruth
    y_true_ml = [
        [0.92, 0.11],
        [0.39, 0.65],
        [0.78, 0.48],
        [0.08, 0.86],
        [0.63, 0.74],
    ]

    # compute p corr
    p_corr = shared.compute_pearson_correlation(y_pred_pc, y_true_ml)
    np.testing.assert_allclose(p_corr, [0.99259384, 0.99102183])


def test_compute_ccc():

    # defines predictions
    y_pred_ccc = [
        [0.81, 0.17],
        [0.32, 0.53],
        [0.79, 0.35],
        [0.18, 0.81],
        [0.52, 0.76],
    ]

    # define groundtruth
    y_true_ccc = [
        [0.96, 0.13],
        [0.39, 0.65],
        [0.73, 0.42],
        [0.09, 0.84],
        [0.61, 0.71],
    ]

    # compute ccc
    ccc = shared.compute_ccc(y_pred_ccc, y_true_ccc)
    np.testing.assert_allclose(ccc, [0.85452581, 1.06378999])


def test_compute_r2_score():
    # defines predictions
    y_pred_r2 = [
        [0.81, 0.17],
        [0.32, 0.53],
        [0.79, 0.35],
        [0.18, 0.81],
        [0.52, 0.76],
    ]

    # define groundtruth
    y_true_r2 = [
        [0.96, 0.13],
        [0.39, 0.65],
        [0.73, 0.42],
        [0.09, 0.84],
        [0.61, 0.71],
    ]

    # compute r2 score
    r2_score = shared.compute_r2_score(y_pred_r2, y_true_r2)
    np.testing.assert_allclose(r2_score, [0.84896967, 0.9170988])


def test_compute_adjusted_r2_score():
    # defines predictions
    y_pred_r2 = [
        [0.81, 0.17],
        [0.32, 0.53],
        [0.79, 0.35],
        [0.18, 0.81],
        [0.52, 0.76],
    ]

    # define groundtruth
    y_true_r2 = [
        [0.96, 0.13],
        [0.39, 0.65],
        [0.73, 0.42],
        [0.09, 0.84],
        [0.61, 0.71],
    ]

    # compute r2 score
    adjusted_r2_score = shared.compute_adjusted_r2_score(y_pred_r2, y_true_r2, p=np.shape(y_true_r2)[1])
    np.testing.assert_allclose(adjusted_r2_score, [0.69793933, 0.8341976])


def test_compute_root_mean_squared_error():
    # defines predictions
    y_pred_rmse = [
        [0.81, 0.17],
        [0.32, 0.53],
        [0.79, 0.35],
        [0.18, 0.81],
        [0.52, 0.76],
    ]

    # define groundtruth
    y_true_rmse = [
        [0.96, 0.13],
        [0.39, 0.65],
        [0.73, 0.42],
        [0.09, 0.84],
        [0.61, 0.71],
    ]

    # compute rmse
    rmse = shared.compute_root_mean_squared_error(y_pred_rmse, y_true_rmse)
    np.testing.assert_allclose(rmse, [0.00944, 0.00486])


def test_compute_mean_squared_error():
    # define predictions
    y_pred_rmse = [
        [0.81, 0.17],
        [0.32, 0.53],
        [0.79, 0.35],
        [0.18, 0.81],
        [0.52, 0.76],
    ]

    # define groundtruth
    y_true_rmse = [
        [0.96, 0.13],
        [0.39, 0.65],
        [0.73, 0.42],
        [0.09, 0.84],
        [0.61, 0.71],
    ]

    # compute mean squared error
    mse = shared.compute_mean_squared_error(y_pred_rmse, y_true_rmse)
    np.testing.assert_allclose(mse, [0.09715966, 0.0697137])
