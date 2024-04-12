# table for finding the correlation coefficients
# between two variances as outlined in the
# landmark paper:
# A Method of Comparing the Areas under Receiver Operating Characteristic Curves Derived fromthe Same Cases
# by James A. Hanley, Ph.D. and Barbara J. McNeil, M.D., Ph.D.

# import packages
import pandas as pd
import numpy as np

# read excel file
table_raw = pd.read_csv('hanley-mcneil-table.csv', index_col=0)

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
