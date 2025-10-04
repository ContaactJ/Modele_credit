import pandas as pd, numpy as np
df = pd.read_csv("data_cox.csv")
rng = np.random.default_rng(42)
idx = np.arange(len(df))
rng.shuffle(idx)
n = len(df)
i1 = int(0.7*n); i2 = int(0.85*n)
train = df.iloc[idx[:i1]]; valid = df.iloc[idx[i1:i2]]; test = df.iloc[idx[i2:]]
train.to_csv("data_cox_train.csv", index=False)
valid.to_csv("data_cox_valid.csv", index=False)
test.to_csv("data_cox_test.csv", index=False)
print(len(train), len(valid), len(test))
