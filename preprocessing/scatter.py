import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os


train_file_path = "../data/train.csv"
train_data = pd.read_csv(train_file_path, encoding="Big5", low_memory=False)
select_file_path = "../data/select.csv"
select_data = pd.read_csv(select_file_path)

# print(select_data)

directory = os.path.abspath("../graph")
if not os.path.exists(directory):
    os.makedirs(directory)

train_data["Y1"].replace({'Y': 1, 'N': 0}, inplace=True)

for col in select_data["SELECT"]:
    train_data[col].replace('Y', 1, inplace=True)
    train_data[col].replace('N', 0, inplace=True)
    # print(train_data[col].head())

for col in select_data["SELECT"]:
    plt.figure()
    my_title = str(col)+" and Y1"
    plt.title(my_title)
    sns_plot = sns.regplot(x=col, y="Y1", data=train_data)
    img_name = str(col)+".png"
    sns_plot.figure.savefig(os.path.join(directory, img_name))
