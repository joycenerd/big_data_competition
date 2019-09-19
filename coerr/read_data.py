import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# read in data
train_file_path = "../data/train.csv"
train_data = pd.read_csv(train_file_path, encoding="Big5", low_memory=False)


# transform true and false to 1 and 0


def bool_to_num(arr):
    num_arr = []
    for val in arr:
        if val == "N":
            num_arr.append(0)
        else:
            num_arr.append(1)
    return num_arr


y = train_data['Y1']
y_num = bool_to_num(y)
# print(train_data.head())


# calculate correlation coefficient between two columns


def calculate_corr(x, y):
    x = np.float64(x)
    y = np.float64(y)
    corr = np.corrcoef(x, y)[0, 1]
    return corr

# calculate IF_...series coefficient coefficient


def if_issue_corr(data):
    target_col = train_data[data]
    num_arr = bool_to_num(target_col)
    corr = calculate_corr(num_arr, y_num)
    return corr


# calculate "IF_ISSUE_K_IND" correlation
cols = ['IF_ISSUE_K_IND', 'IF_ISSUE_L_IND', 'IF_ISSUE_M_IND', 'IF_ISSUE_N_IND', 'IF_ISSUE_O_IND', 'IF_ISSUE_P_IND',
        'IF_ISSUE_Q_IND', 'IF_ADD_F_IND', 'IF_ADD_L_IND', 'IF_ADD_Q_IND', 'IF_ADD_G_IND', 'IF_ADD_R_IND', 'IF_ADD_IND']
for col in cols:
    corr = if_issue_corr(col)
    print(col, ":", corr)
