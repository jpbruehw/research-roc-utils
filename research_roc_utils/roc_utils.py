# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikits.bootstrap as boot
from scipy.stats import percentileofscore
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

"""
This is the production version of the research_roc_utils package
for local development, use the local-dev versions
this is a consolidated version optimized for build which is
less modular and includes all packages

-----
The goal of this package is to optimize the comparison of ROC curves
based on the groundbreaking work of Hanley and McNeil who originally
applied these concepts in the field of medicine.

These concepts have since been expanded on and optimized for econometric
research by Weiling Liu and Emanuel Moench as well as Jorda and Taylor

----
Papers

Jorda and Taylor (2011): https://shorturl.at/ftwB5
Liu and Moench (2014): https://shorturl.at/clvZ9
Hanley and McNeil (1982): https://shorturl.at/foSU7
Hanley and Mcneil (1983): https://shorturl.at/joqv9

"""

# START CODE
#----------#
# p1: Hanley & McNeil Corr. Coeff. Table
# p2: helper functions to assist with calculations in other sections
# p3: function to find p-val for hypothesis testing using bootstrap resampling
##############################################################################
# p1: Corr. Coeff. table to find r value based on avg auroc diff and avg. corr

# create table
index = [0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,
         0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,
         0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,
         0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9]

columns = [0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975]

table_rows = [
              [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01],
              [0.04,0.04,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.02,0.02,0.02],
              [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.03,0.02],
              [0.07,0.07,0.07,0.07,0.07,0.06,0.06,0.06,0.06,0.05,0.04,0.03],
              [0.09,0.09,0.09,0.09,0.08,0.08,0.08,0.07,0.07,0.06,0.06,0.04],
              [0.11,0.11,0.11,0.1,0.1,0.1,0.09,0.09,0.08,0.08,0.07,0.05],
              [0.13,0.12,0.12,0.12,0.12,0.11,0.11,0.11,0.1,0.09,0.08,0.06],
              [0.14,0.14,0.14,0.14,0.13,0.13,0.13,0.12,0.11,0.11,0.09,0.07],
              [0.16,0.16,0.16,0.16,0.15,0.15,0.14,0.14,0.13,0.12,0.11,0.09],
              [0.18,0.18,0.18,0.17,0.17,0.17,0.16,0.15,0.15,0.14,0.12,0.1],
              [0.2,0.2,0.19,0.19,0.19,0.18,0.18,0.17,0.16,0.15,0.14,0.11],
              [0.22,0.22,0.21,0.21,0.21,0.2,0.19,0.19,0.18,0.17,0.15,0.12],
              [0.24,0.23,0.23,0.23,0.22,0.22,0.21,0.2,0.19,0.18,0.16,0.13],
              [0.26,0.25,0.25,0.25,0.24,0.24,0.23,0.22,0.21,0.2,0.18,0.15],
              [0.27,0.27,0.27,0.26,0.26,0.25,0.25,0.24,0.23,0.21,0.19,0.16],
              [0.29,0.29,0.29,0.28,0.28,0.27,0.26,0.26,0.24,0.23,0.21,0.18],
              [0.31,0.31,0.31,0.3,0.3,0.29,0.28,0.27,0.26,0.25,0.23,0.19],
              [0.33,0.33,0.32,0.32,0.31,0.31,0.3,0.29,0.28,0.26,0.24,0.21],
              [0.35,0.35,0.34,0.34,0.33,0.33,0.32,0.31,0.3,0.28,0.26,0.22],
              [0.37,0.37,0.36,0.36,0.35,0.35,0.34,0.33,0.32,0.3,0.28,0.24],
              [0.39,0.39,0.38,0.38,0.37,0.36,0.36,0.35,0.33,0.32,0.29,0.25],
              [0.41,0.4,0.4,0.4,0.39,0.38,0.38,0.37,0.35,0.34,0.31,0.27],
              [0.43,0.42,0.42,0.42,0.41,0.4,0.39,0.38,0.37,0.35,0.33,0.29],
              [0.45,0.44,0.44,0.43,0.43,0.42,0.41,0.4,0.39,0.37,0.35,0.3],
              [0.47,0.46,0.46,0.45,0.45,0.44,0.43,0.42,0.41,0.39,0.37,0.32],
              [0.49,0.48,0.48,0.47,0.47,0.46,0.45,0.44,0.43,0.41,0.39,0.34],
              [0.51,0.5,0.5,0.49,0.49,0.48,0.47,0.46,0.45,0.43,0.41,0.36],
              [0.53,0.52,0.52,0.51,0.51,0.5,0.49,0.48,0.47,0.45,0.43,0.38],
              [0.55,0.54,0.54,0.53,0.53,0.52,0.51,0.5,0.49,0.47,0.45,0.4],
              [0.57,0.56,0.56,0.55,0.55,0.54,0.53,0.52,0.51,0.49,0.47,0.42],
              [0.59,0.58,0.58,0.57,0.57,0.56,0.55,0.54,0.53,0.51,0.49,0.45],
              [0.61,0.6,0.6,0.59,0.59,0.58,0.58,0.57,0.55,0.54,0.51,0.47],
              [0.63,0.62,0.62,0.62,0.61,0.6,0.6,0.59,0.57,0.56,0.53,0.49],
              [0.65,0.64,0.64,0.64,0.63,0.62,0.62,0.61,0.6,0.58,0.56,0.51],
              [0.67,0.66,0.66,0.66,0.65,0.65,0.64,0.63,0.62,0.6,0.58,0.54],
              [0.69,0.69,0.68,0.68,0.67,0.67,0.66,0.65,0.64,0.63,0.6,0.56],
              [0.71,0.71,0.7,0.7,0.69,0.69,0.68,0.67,0.66,0.65,0.63,0.59],
              [0.73,0.73,0.72,0.72,0.72,0.71,0.71,0.7,0.69,0.67,0.65,0.61],
              [0.75,0.75,0.75,0.74,0.74,0.73,0.73,0.72,0.71,0.7,0.68,0.64],
              [0.77,0.77,0.77,0.76,0.76,0.76,0.75,0.74,0.73,0.72,0.7,0.67],
              [0.79,0.79,0.79,0.79,0.78,0.78,0.77,0.77,0.76,0.75,0.73,0.7],
              [0.82,0.81,0.81,0.81,0.81,0.8,0.8,0.79,0.78,0.77,0.76,0.73],
              [0.84,0.84,0.83,0.83,0.83,0.82,0.82,0.81,0.81,0.8,0.78,0.75],
              [0.86,0.86,0.86,0.85,0.85,0.85,0.84,0.84,0.83,0.82,0.81,0.79],
              [0.88,0.88,0.88,0.88,0.87,0.87,0.87,0.86,0.86,0.85,0.84,0.82]
              ] 

table_raw = pd.DataFrame(table_rows, index=index, columns=columns)

# function to find the r value
# we pass in the average area of the two model curves
# being compares as well as the average correlation
# coefficient between the two models
def find_r_val(avg_corr, avg_area):
    # make sure values are positive
    abs_corr = np.absolute(avg_corr)
    abs_area = np.absolute(avg_area)
    # calculate the diff between
    # the correlation and index coll
    corr_diff = np.absolute(table_raw.index - abs_corr)
    # get list of columns
    area_cols = table_raw.columns
    # convert to floats
    area_cols = [float(col) for col in area_cols]
    area_diff = np.absolute(area_cols - abs_area)
    # get the index vals
    corr_idx = corr_diff.argmin()
    area_idx = area_diff.argmin()
    # get value based on index and column
    r = table_raw.iloc[corr_idx, area_idx]
    
    return r

##############################################################################
# p2: helper functions to split up calculations

# HELPER FUNCTIONS
#----------------#

# function to help make initial checks
def check_passed_data(y_true, y_pred_1, y_pred_2):
    # make sure everything is 1D array
    check_list = [y_true, y_pred_1, y_pred_2]
    # loop and make assertions
    for data in check_list:
        assert hasattr(data, '__len__'), "Input data does not have a length attribute"
        assert len(data) > 0, "Input data is empty"
        # if numpy array check it is one dimensional
        if isinstance(data, np.ndarray):
            assert data.ndim == 1, "Input data is not a 1D NumPy array"
        assert isinstance(data, (list, np.ndarray, pd.Series)), "Input data is not one-dimensional"
    # check the data is correctly formatted
    assert len(y_true) == len(y_pred_1) == len(y_pred_2), 'Length mismatch'
    
# Q1 and Q2 calculations
# see Hanley and McNeil (1982)
# link: https://shorturl.at/foSU7
def q_calculations(roc_auc):
    q_1 = roc_auc / (2 - roc_auc)
    q_2 = 2 * roc_auc**2 / (1 + roc_auc)
    return q_1, q_2

# find the average correlation
# to find the correlation coefficient
# we use the method outlined in
# Jorda and Taylor (2011) which is
# the average of the 0 case correlation
# and the 1 case correlation
# Liu and Emanuel use the Kendall's Tau
# correlation coefficient but can also
# use the Pearson Correlation coefficient
# other corr method can also be passed in as long
# as it returns a coefficient and a p-val
# p-val is not needed in this case
# link: https://shorturl.at/cwBUZ
def avg_corr_fun(y_true, y_pred_1, y_pred_2, corr_method):
    # concat the data to a dataframe
    # to line up all the values
    df = pd.DataFrame({'true': y_true, 'model_1': y_pred_1, 'model_2': y_pred_2})
    # split the df into false and true
    df_case_true = df[df['true'] == 1]
    df_case_false = df[df['true'] == 0]
    # find coefficient of true/false case
    coeff_true, _ = corr_method(df_case_true['model_1'], df_case_true['model_2'])
    coeff_false, _ = corr_method(df_case_false['model_1'], df_case_false['model_2'])
    # find the average of the two coefficients
    avg_coeff = (coeff_true + coeff_false) / 2
    # return the resulting r value
    return avg_coeff

# find the variance also
# based on Hanley and McNeil (1983)
# takes the auc score and the
# true positive and true negative classes
def get_variance_t_stat(roc_auc, q_1, q_2, tp, tn):
    return (1 / (tp * tn)) * np.sqrt(roc_auc * (1 - roc_auc) + (tp - 1) * (q_1 - roc_auc**2) + (tn - 1) * (q_2 - roc_auc**2))

# variance method used by
# Jorda and Taylor is slightly
# different than Liu and Emanuel
# refer to page 14 of linked paper
# by Jorda and Taylor for details
def get_variance_z_score(roc_auc, q_1, q_2, tp, tn):
    return roc_auc * (1 - roc_auc) + (tp - 1) * (q_1 - roc_auc**2) + (tn - 1) * (q_2 - roc_auc**2)

# put the pieces together and get the stat
def get_test_stat(roc_auc_1, roc_auc_2, model_1_var, model_2_var, r):
    numerator = roc_auc_1 - roc_auc_2
    denominator = np.sqrt(model_1_var + model_2_var - 2 * r * model_1_var * model_2_var)
    stat = numerator / denominator
    return stat

##############################################################################
# p3: bootstrap p-val calulator, suitable for one and two sided tests

# BOOTSTRAP RESAMPLING P-VALUE
#----------------------------#

def calculate_roc_score(y_true, y_pred, indices, score_fun, sample_weight):
    if sample_weight is not None:
        return score_fun(y_true[indices], y_pred[indices], sample_weight=sample_weight[indices])
    else:
        return score_fun(y_true[indices], y_pred[indices])
  
def convert_np_array(list_like):
    if isinstance(list_like, np.ndarray):
        return list_like
    else:
        return np.array(list_like)
  
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
    
    # ensure that lists are array like object that have shape
    y_true = convert_np_array(y_true)
    y_pred_1 = convert_np_array(y_pred_1)
    y_pred_2 = convert_np_array(y_pred_2)
    
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

##############################################################################
# p4: Binary Model comparison functions that provide functionality
# provides methods for t-stat, z-score, threshold calculations, as well as
# creating stacked ROC curve plot to visualize multiple models
# based on the academic research cited above 

# BINARY MODEL COMPARISON FUNCTIONS
#---------------------------------#

# this function performs a t-test for classification
# model assessment as outlined in the paper
# What Predicts U.S. Recessions? by Weiling Liu and Emanuel Moench
# see pages 8-13 to get an overview of the model itself
# paper link: https://shorturl.at/clvZ9
def roc_t_stat(y_true, y_pred_1, y_pred_2, corr_method, roc_auc_fun=roc_auc_score):
    # call helper function to make initial checks
    check_passed_data(y_true, y_pred_1, y_pred_2)

    # INITIAL CHECKS COMPLETE
    #########################
    # dependency link: https://shorturl.at/hwxy6
    
    # get roc scores
    roc_auc_1 = roc_auc_fun(y_true, y_pred_1)
    roc_auc_2 = roc_auc_fun(y_true, y_pred_2)
    
    # get the q stats for each model
    model_1_q_1, model_1_q_2 = q_calculations(roc_auc_1)
    model_2_q_1, model_2_q_2 = q_calculations(roc_auc_2)
    
    # get the r value
    # see corr_coeff_table.py for implementation
    avg_corr = avg_corr_fun(y_true, y_pred_1, y_pred_2, corr_method)
    avg_area = (roc_auc_1 + roc_auc_2) / 2
    r = find_r_val(avg_corr, avg_area)
    
    # get the true and false counts
    tp = np.sum(y_true == 1)
    tn = np.sum(y_true == 0)
    
    # get the variance for each model
    model_1_var = get_variance_t_stat(roc_auc_1, model_1_q_1, model_1_q_2, tp, tn)
    model_2_var = get_variance_t_stat(roc_auc_2, model_2_q_1, model_2_q_2, tp, tn)
    
    # calculate the t val
    t_stat = get_test_stat(roc_auc_1, roc_auc_2, model_1_var, model_2_var, r)
    
    return t_stat

# z-score function based on the
# groundbreaking work from Hanley and McNeil (1983)
def roc_z_score(y_true, y_pred_1, y_pred_2, corr_method, roc_auc_fun=roc_auc_score):
    # call helper function to make initial checks
    check_passed_data(y_true, y_pred_1, y_pred_2)

    # INITIAL CHECKS COMPLETE
    #########################
    # dependency link: https://shorturl.at/hwxy6
    
    # get roc scores
    roc_auc_1 = roc_auc_fun(y_true, y_pred_1)
    roc_auc_2 = roc_auc_fun(y_true, y_pred_2)
    
    # get the q stats for each model
    model_1_q_1, model_1_q_2 = q_calculations(roc_auc_1)
    model_2_q_1, model_2_q_2 = q_calculations(roc_auc_2)
    
    # get the r value
    # see corr_coeff_table.py for implementation
    avg_corr = avg_corr_fun(y_true, y_pred_1, y_pred_2, corr_method)
    avg_area = (roc_auc_1 + roc_auc_2) / 2
    r = find_r_val(avg_corr, avg_area)
    
    # get the true and false counts
    tp = np.sum(y_true == 1)
    tn = np.sum(y_true == 0)
    
    # get the variance for each model
    model_1_var = get_variance_z_score(roc_auc_1, model_1_q_1, model_1_q_2, tp, tn)
    model_2_var = get_variance_z_score(roc_auc_2, model_2_q_1, model_2_q_2, tp, tn)
    
    # calculate the z-score
    z_score = get_test_stat(roc_auc_1, roc_auc_2, model_1_var, model_2_var, r)
    
    return z_score

# calculate p-value with bootstrapping
# method for comparing two models using 
# at a given significance level with bootstrapping
# 1 - p hypothesis test:
# H0: Model 1 and Model 2 have no difference in performance
# H1: Model 2's performance is better than Model 1
# returns p-val and list of the differences between models
def boot_p_val(
    y_true,
    y_pred_1,
    y_pred_2,
    compare_fun=np.subtract,
    score_fun=roc_auc_score,
    sample_weight=None,
    n_resamples=5000,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True
    ):
    
    # call helper function to make initial checks
    check_passed_data(y_true, y_pred_1, y_pred_2)
    
    # INITIAL CHECKS COMPLETE
    #########################
    # dependency link: https://shorturl.at/hwxy6
    
    # call function
    return p_val(
           y_true=y_true,
           y_pred_1=y_pred_1,
           y_pred_2=y_pred_2,
           compare_fun=compare_fun,
           score_fun=score_fun,
           sample_weight=sample_weight,
           n_resamples=n_resamples,
           two_tailed=two_tailed,
           seed=seed,
           reject_one_class_samples=reject_one_class_samples
           )

# PLOT MULTIPLE AUROC GRAPHS
#--------------------------#

# returns plt object which can be further
# edited as the user wishes using the 
# matplotlib library
# docs: https://matplotlib.org/stable/index.html
def stacked_roc_plt(
        y_true,
        model_preds,
        model_names,
        roc_fun=roc_curve,
        auc_fun=auc,
        fig_size=(8,6),
        linewidth=2,
        linestyle='-',
        rand_guess_color='black'
        ):
    """
    Params
    ------
    y_true: arrary like list of binary values
    model_preds: array of list like objects holding model predictions
    model_names: array of strings containing names for the model predictions
                must be in the same order as the model_pred data
    fig_size: optional param for fig size
    roc_fun: func to calc roc_curve, must return fpr, tpr, and thresholds
    auc_fun: takes fpr and tpr as args to return area under curve
    linewidth: set line width for plots --> see Matplotlib docs for reference
    linestyle: set line style for plots --> see Matplotlib docs for reference
    rand_guess_color: color to set for random guess line --> see Matplotlib docs for reference
    
    Return val
    ---------
    func returns matplotlib plot object, does not add any styles other than
    fig size basic line elements
    """
    # make sure lengths match
    assert len(model_preds) == len(model_names), 'Length mismatch between models and model names'
    # initialize fig and loop over results
    # create plot
    plt.figure(figsize=fig_size)
    for model_y_pred, model_name in zip(model_preds, model_names):
        # get the fpr, trp, and thresholds
        fpr, tpr, _ = roc_fun(y_true, model_y_pred)
        # call method on fpr and tpr
        roc_auc = auc_fun(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=linewidth, linestyle=linestyle, label=f'{model_name} ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color=rand_guess_color, linewidth=linewidth, linestyle=':')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        # add a bit of margin to y-axis
        plt.ylim([0.0, 1.05])
        
    return plt
 
# THRESHOLD FUNCTIONS
#-------------------#

# get the optimal threshold for classification
# based on the roc curve
# works best for data that is NOT highly imbalanced
def optimal_threshold(y_true, y_pred):
    # calc the curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # find the optimal threshold
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    
    return best_threshold

# finding the optimal cutoff for highly imbalanced data
# can be more challenging, therefore the prodecure
# needs to be slightly adjusted
# uses geometric mean method
def optimal_threshold_imbalanced(y_true, y_pred):
    # calc the curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # calculate means and get idx
    gmeans = np.sqrt(tpr * (1-fpr))
    gmeans = np.array(gmeans)
    best_idx = np.argmax(gmeans)
    best_threshold = thresholds[best_idx]
    
    return best_threshold

# NON-PARAMETRIC AUROC FUNCTION
#-----------------------------#

# non-parametric approach for
# estimating the AUCROC
# paper: Performance Evaluation of Zero Net-Investment Strategies (2010)
# by Oscar Jorda and Alan M. Taylor
# the original implementation of this 
# methodology was to compare two investment strategies
# paper link: https://shorturl.at/oEORV
def auroc_non_parametric(y_true, y_pred):
    # get true neg and pos counts
    n_1 = sum(1 for y in y_true if y == 1)
    n_0 = sum(1 for y in y_true if y == 0)
    # get reciprocal product
    rec_prod = 1 / (n_1 * n_0)
    # create df and seperate
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # get predictions at 0 and 1
    y_pred_true = df[df['y_true'] == 1]['y_pred']
    y_pred_false = df[df['y_true'] == 0]['y_pred']
    # find sum
    total_sum = 0
    for zi in y_pred_true:
        for xj in y_pred_false:
            if zi > xj:
                total_sum += 1
            elif zi == xj:
                total_sum += 0.5
    auroc = rec_prod * total_sum
    
    return auroc