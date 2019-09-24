import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn 

data_con="D:/TBARDATA/train/"
train_dataframe=pd.read_csv(data_con+"train.csv",encoding="big5",low_memory=False)
for i in train_dataframe.columns.to_list():
     if train_dataframe[i].nunique() == 2:
         train_dataframe[i].replace({'Y' : 1 , 'N' : 0} , inplace = True)

sub_data_df=train_dataframe.iloc[:,110:131]
sub_data_df=pd.concat([sub_data_df , train_dataframe[['Y1']]] , axis = 1)
print(sub_data_df.corr()['Y1'])
print(sub_data_df.corr()['Y1'].sort_values(ascending=False))
result_column = sub_data_df.corr()['Y1'].sort_values(ascending=False)[1:11].index.to_list()
print(result_column)


for i in result_column : 
	plt.scatter(range(100000) , train_dataframe[i])
	plt.xlabel("count")
	plt.ylabel(i)
	plt.title("EDA scatter")
	plt.show()

for i in result_column : 
	if train_dataframe[i].nunique() > 2 : 
		sns.boxplot(x = 'Y1' , y = i , data = train_dataframe , palette = "hls")
		plt.show()