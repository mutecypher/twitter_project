import pandas as pd
import numpy as np
import datetime as dt

file1 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_2_nn_scored.csv'

df = pd.read_csv(file1, header='infer', index_col=0)

df.created_at = pd.to_datetime(df.created_at)
df['Date'] = df.created_at.dt.date
print("columms are ", df.columns)


def weights(group):
    return (group['pos'] * group['retweets'])/group['retweets'].sum()


df1 = df.groupby('Date')

print(df1.head())
