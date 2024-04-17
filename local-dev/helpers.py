import numpy as np
import pandas as pd

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

# find the correlation coefficient
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
def correlation_coef(y_true, y_pred_1, y_pred_2, corr_method):
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