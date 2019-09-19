import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn 


data_path = "D:/VScode workshop/big_data_competition/data/"
train_df = pd.read_csv(data_path + "train.csv" , encoding = "big5" , low_memory = False)

bin_col = []
for i in train_df.columns.to_list() : 
	if train_df[i].nunique() == 2 : 
		bin_col.append(i)
for i in bin_col : 
	train_df[i].replace({'Y' : 1 , 'N' : 0} , inplace = True)

sub_df = train_df.iloc[ : , 72 : 109]
sub_df = pd.concat([sub_df , train_df[['Y1']]] , axis = 1)
print(sub_df.corr()['Y1'])

#|r| >= 0.05 (相關係數前十大)
#X_B_IND、X_C_IND、X_E_IND、X_H_IND、TOOL_VISIT_1YEAR_CNT、DIEBENEFIT_AMT、
#DIEACCIDENT_AMT、MONTHLY_CARE_AMT、LIFE_INSD_CNT、IF_ISSUE_INSD_I_IND

r_col = ["X_B_IND" , "X_C_IND" , "X_E_IND" , "X_H_IND" , "TOOL_VISIT_1YEAR_CNT" , "DIEBENEFIT_AMT" , 
"DIEACCIDENT_AMT" , "MONTHLY_CARE_AMT" , "LIFE_INSD_CNT" , "IF_ISSUE_INSD_I_IND"]

'''
for i in r_col : 
	plt.scatter(range(100000) , train_df[i])
	plt.xlabel("count")
	plt.ylabel(i)
	plt.title("EDA scatter")
	plt.show()

for i in r_col : 
	if train_df[i].nunique() > 2 : 
		sns.boxplot(x = 'Y1' , y = i , data = train_df , palette = "hls")
		plt.show()
'''
