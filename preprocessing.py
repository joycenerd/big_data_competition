import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os


def read_data() : 
	data_path = os.getcwd() +  "/data/"
	try : 
		train_df = pd.read_csv(data_path + "train.csv" , encoding = "big5" , low_memory = False)
		return train_df
	except : 
		print("readfile error")
	
def transform_label_data(train_df) : 
	for i in train_df.columns.to_list() : 
		if train_df[i].nunique() == 2 : 
			train_df[i].replace({'Y' : 1 , 'N' : 0} , inplace = True)
	return train_df

def get_corrcoef(train_df , start , stop) : 
	sub_df = train_df.iloc[ : , start : stop]
	sub_df = pd.concat([sub_df , train_df[['Y1']]] , axis = 1)
	cor = sub_df.corr()['Y1']
	print(cor)
	return cor

def EDA_scatter(train_df , col) : 
	for i in col : 
		plt.scatter(range(100000) , train_df[i])
		plt.xlabel("count")
		plt.ylabel(i)
		plt.title("EDA scatter")
		plt.show()	

def box_plot(train_df , col) : 
	for i in col : 
		if train_df[i].nunique() > 2 : 
			sns.boxplot(x = 'Y1' , y = i , data = train_df , palette = "hls")
			plt.show()



train_df = read_data()
train_df = transform_label_data(train_df)
cor_Y1 = get_corrcoef(train_df , 73 , 109)
r_col = cor_Y1.sort_values(ascending = False)[:10].index.to_list()
EDA_scatter(train_df , r_col)
box_plot(train_df , r_col)




