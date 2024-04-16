# import packages
import numpy as np
import scikits.bootstrap as boot
from scipy.stats import percentileofscore
from sklearn.metrics import roc_auc_score

def calculate_roc_score(y_true, y_pred, indices, score_fun, sample_weight):
    if sample_weight is not None:
        return score_fun(y_true[indices], y_pred[indices], sample_weight=sample_weight[indices])
    else:
      return score_fun(y_true[indices], y_pred[indices])
  
# H0: Model 1 is significantly better than Model 2
# H1: performance is not significantly different from zero
def p_val(
    y_true,
    y_pred_1,
    y_pred_2,
    compare_fun=np.subtract,
    score_fun=roc_auc_score,
    sample_weight=None,
    n_resamples=5000,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True,
):
    z = []
    
    indices = list(boot.bootstrap_indices(y_true, n_samples=n_resamples, seed=seed))
    
    for idx_vals in indices:
        if reject_one_class_samples and len(np.unique(y_true[idx_vals])) < 2:
            continue
        score_1 = calculate_roc_score(y_true, y_pred_1, idx_vals, score_fun, sample_weight)
        score_2 = calculate_roc_score(y_true, y_pred_2, idx_vals, score_fun, sample_weight)
        z.append(compare_fun(score_1, score_2))

    p = percentileofscore(z, 0.0, kind="mean") / 100.0

    if two_tailed:
        p = 2 * min(p, 1-p)
        
    return p, z