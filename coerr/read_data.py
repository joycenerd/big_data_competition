import pandas as pd

train_file_path = "../data/train.csv"
train_data = pd.read_csv(train_file_path, encoding="Big5", low_memory=False)
print(train_data.head())
