import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


data_path = "D:/VScode workshop/big_data_competition/data/"
train_df = pd.read_csv(data_path + "train.csv" , encoding = "big5")


bins = [(72 , 84) , (84 , 91) , (91 , 97) , (97 , 103) , (103 , 109)]

for i , j in bins : 
	sub_df = train_df.iloc[ : , i : j]
	sub_df = pd.concat([sub_df , train_df[['Y1']]] , axis = 1)
	bin_col = []
	for k in sub_df.columns.to_list() : 
		if sub_df[k].nunique() == 2 : 
			bin_col.append(k)
	for k in bin_col : 
		sub_df[k].replace({'Y' : 1 , 'N' : 0} , inplace = True)
	print(sub_df.corr())

#|r| >= 0.05 (相關係數前十大)
#X_B_IND、X_C_IND、X_E_IND、X_H_IND、TOOL_VISIT_1YEAR_CNT、DIEBENEFIT_AMT、
#DIEACCIDENT_AMT、MONTHLY_CARE_AMT、LIFE_INSD_CNT、IF_ISSUE_INSD_I_IND


bin_col = []
for i in train_df.columns.to_list() : 
	if train_df[i].nunique() == 2 : 
		bin_col.append(i)
for i in bin_col : 
	train_df[i].replace({'Y' : 1 , 'N' : 0} , inplace = True)

r_col = ["X_B_IND" , "X_C_IND" , "X_E_IND" , "X_H_IND" , "TOOL_VISIT_1YEAR_CNT" , "DIEBENEFIT_AMT" , 
"DIEACCIDENT_AMT" , "MONTHLY_CARE_AMT" , "LIFE_INSD_CNT" , "IF_ISSUE_INSD_I_IND"]
#r_df = train_df.loc[: , r_col]
#print(r_df)

print(train_df['X_B_IND'].head())

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

