import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trainPath = "../data/train.csv"

train_df = pd.read_csv(trainPath, encoding = "big5")

print(train_df)


# df1 for 1 ~ 36
df1 = train_df.iloc[:, 0:36]

# drop columns
drop_col = [3, 4]
df1.drop(df1.columns[drop_col], axis=1, inplace=True)

# print(df1.info())

df1 = pd.concat([df1, train_df[['Y1']]], axis=1)

for k in df1.columns.to_list():
    if df1[k].nunique() == 2:
        df1[k].replace({'M' : 1, 'F' : 0, 'Y' : 1 ,'N' : 0}, inplace=True)
    if (df1[k].nunique() == 4 )& (df1[k].dtype == object):
        df1[k].replace({'低':0, '中':1, '中高':2, '高': 3}, inplace=True)

# n = 0
# while n < 36:
#     print(df1.iloc[:,n : n + 4])
#     n += 4
corr = df1.corr()['Y1']

plt.scatter(corr, range(corr.shape[0]))
plt.show()


