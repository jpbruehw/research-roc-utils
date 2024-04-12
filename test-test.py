import numpy as np
from scipy.stats import percentileofscore
from sklearn.metrics import roc_auc_score

def pvalue(
    y_true,
    y_pred1,
    y_pred2,
    score_fun,
    sample_weight=None,
    n_bootstraps=2000,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True,
):
    """
    Compute p-value for hypothesis that score function for model I predictions is higher than for model II predictions
    using bootstrapping.
    :param y_true: 1D list or array of labels.
    :param y_pred1: 1D list or array of predictions for model I corresponding to elements in y_true.
    :param y_pred2: 1D list or array of predictions for model II corresponding to elements in y_true.
    :param score_fun: Score function for which confidence interval is computed. (e.g. sklearn.metrics.accuracy_score)
    :param sample_weight: 1D list or array of sample weights to pass to score_fun, see e.g. sklearn.metrics.roc_auc_score.
    :param n_bootstraps: The number of bootstraps. (default: 2000)
    :param two_tailed: Whether to use two-tailed test. (default: True)
    :param seed: Random seed for reproducibility. (default: None)
    :param reject_one_class_samples: Whether to reject bootstrapped samples with only one label. For scores like AUC we
    need at least one positive and one negative sample. (default: True)
    :return: Computed p-value, array of bootstrapped differences of scores.
    """

    assert len(y_true) == len(y_pred1)
    assert len(y_true) == len(y_pred2)

    return pvalue_stat(
        y_true=y_true,
        y_preds1=y_pred1,
        y_preds2=y_pred2,
        score_fun=score_fun,
        sample_weight=sample_weight,
        n_bootstraps=n_bootstraps,
        two_tailed=two_tailed,
        seed=seed,
        reject_one_class_samples=reject_one_class_samples,
    )


def pvalue_stat(
    y_true,
    y_preds1,
    y_preds2,
    score_fun,
    stat_fun=np.mean,
    compare_fun=np.subtract,
    sample_weight=None,
    n_bootstraps=50,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True,
):
    """
    Compute p-value for hypothesis that given statistic of score function for model I predictions is higher than for
    model II predictions using bootstrapping.
    :param y_true: 1D list or array of labels.
    :param y_preds1: A list of lists or 2D array of predictions for model I corresponding to elements in y_true.
    :param y_preds2: A list of lists or 2D array of predictions for model II corresponding to elements in y_true.
    :param score_fun: Score function for which confidence interval is computed. (e.g. sklearn.metrics.accuracy_score)
    :param stat_fun: Statistic for which p-value is computed. (e.g. np.mean)
    :param compare_fun: Function to determine relative performance. (default: score1 - score2)
    :param sample_weight: 1D list or array of sample weights to pass to score_fun, see e.g. sklearn.metrics.roc_auc_score.
    :param n_bootstraps: The number of bootstraps. (default: 2000)
    :param two_tailed: Whether to use two-tailed test. (default: True)
    :param seed: Random seed for reproducibility. (default: None)
    :param reject_one_class_samples: Whether to reject bootstrapped samples with only one label. For scores like AUC we
    need at least one positive and one negative sample. (default: True)
    :return: Computed p-value, array of bootstrapped differences of scores.
    """

    y_true = np.array(y_true)
    y_preds1 = np.atleast_2d(y_preds1)
    y_preds2 = np.atleast_2d(y_preds2)
    assert all(len(y_true) == len(y) for y in y_preds1)
    assert all(len(y_true) == len(y) for y in y_preds2)

    np.random.seed(seed)
    z = []
    for i in range(n_bootstraps):
        readers1 = np.random.randint(0, len(y_preds1), len(y_preds1))
        readers2 = np.random.randint(0, len(y_preds2), len(y_preds2))
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            continue
        reader1_scores = []
        for r in readers1:
            if sample_weight is not None:
                reader1_scores.append(score_fun(y_true[indices], y_preds1[r][indices], sample_weight=sample_weight[indices]))
            else:
                reader1_scores.append(score_fun(y_true[indices], y_preds1[r][indices]))
        score1 = stat_fun(reader1_scores)
        reader2_scores = []
        for r in readers2:
            if sample_weight is not None:
                reader2_scores.append(score_fun(y_true[indices], y_preds2[r][indices], sample_weight=sample_weight[indices]))
            else:
                reader2_scores.append(score_fun(y_true[indices], y_preds2[r][indices]))
        score2 = stat_fun(reader2_scores)
        z.append(compare_fun(score1, score2))

    p = percentileofscore(z, 0.0, kind="weak") / 100.0
    if two_tailed:
        p *= 2.0
    return p, z

y_true = np.random.randint(0, 2, size=100)  # binary true labels
y_pred_1 = np.random.rand(100)  # random predictions for model 1
y_pred_2 = np.random.rand(100)  # random predictions for model 2

p, z= pvalue(y_true, y_pred_1, y_pred_2,n_bootstraps=10000, score_fun=roc_auc_score)