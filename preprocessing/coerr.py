import pandas as pd
import numpy as np
import math
import csv
from scipy.stats import pearsonr

# read in data
train_file_path = "../data/train.csv"
train_data = pd.read_csv(train_file_path, encoding="Big5", low_memory=False)


# transform true and false to 1 and 0


def bool_to_num(arr):
    num_arr = []
    for val in arr:
        if val == "N":
            num_arr.append(-1)
        else:
            num_arr.append(1)
    return num_arr


y = train_data['Y1']
y_num = bool_to_num(y)
# print(train_data.head())

select_cols = []


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


# calculate "IF_ISSUE_..." correlation
cols = ['IF_ISSUE_K_IND', 'IF_ISSUE_L_IND', 'IF_ISSUE_M_IND', 'IF_ISSUE_N_IND', 'IF_ISSUE_O_IND',
        'IF_ISSUE_P_IND', 'IF_ISSUE_Q_IND', 'IF_ADD_F_IND', 'IF_ADD_L_IND', 'IF_ADD_Q_IND', 'IF_ADD_G_IND',
        'IF_ADD_R_IND', 'IF_ADD_IND', 'L1YR_PAYMENT_REMINDER_IND', 'L1YR_LAPSE_IND', 'LAST_B_CONTACT_DT',
        'LAST_C_DT', 'IF_S_REAL_IND', 'IF_Y_REAL_IND', 'IM_IS_A_IND', 'IM_IS_B_IND']
for col in cols:
    corr = if_issue_corr(col)
    print(col, ":", corr)
    if(corr > 0.05):
        select_cols.append(col)


def count_NA(col):
    data = train_data[col]
    # print(data)
    nan_total = data.isnull().sum()
    percentage = float(nan_total)/len(data)
    if(percentage > 0.1):
        return True
    return False

# calculate correlation coefficient of T/N columns with NaN values


tn_na_cols = ['A_IND', 'B_IND', 'C_IND']

for col in tn_na_cols:
    na_over = count_NA(col)
    if(na_over == True):
        print(col, ": Too many NA values")
    else:
        corr = if_issue_corr(col)
        print(col, ":", corr)
        if(corr > 0.05):
            select_cols.append(col)

numerical_cols = ['ANNUAL_PREMIUM_AMT', 'AG_CNT', 'AG_NOW_CNT', 'CLC_CUR_NUM', 'ANNUAL_INCOME_AMT',
                  'L1YR_C_CNT', 'BANK_NUMBER_CNT', 'INSD_LAST_YEARDIF_CNT', 'BMI', 'IM_CNT']

for col in numerical_cols:
    na_over = count_NA(col)
    if(na_over == True):
        print(col, ": Too many NA values")
    else:
        corr = calculate_corr(train_data[col], y_num)
        print(col, ":", corr)
        if(corr > 0.05):
            select_cols.append(col)

print(select_cols)

select_cols = np.array(select_cols)
select_cols_2d = np.reshape(select_cols, (-1, 1))

df = pd.DataFrame(select_cols_2d, columns=["SELECT"])
df.to_csv("../data/select.csv")
