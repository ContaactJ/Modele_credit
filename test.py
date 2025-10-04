import pandas as pd
df = pd.read_csv("data_cox_train.csv")  # ou le CSV exact que tu as utilisé
print("Nb d'observations :", len(df))
print("Nb d'événements   :", df['event'].sum())
print("Nb de temps événements distincts :", df.loc[df['event']==1, 'time'].nunique())
