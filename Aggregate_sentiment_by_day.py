# import os
# from bs4 import BeautifulSoup
# import requests
import pandas as pd
import numpy as np
import datetime as dt

file1 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_2_nn_scored.csv'

df = pd.read_csv(file1, header='infer', index_col=0)

# df.created_at = pd.to_datetime(df.created_at)

df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%m/%d/%Y')


# This code assumes that your data file is in
# CSV format and is named data.csv. It reads
# the data file into a pandas dataframe,
# converts the created_at column to a
# datetime object, groups the data by date,
# and then calculates the weighted average
# for each sentiment (neg, neu, and pos)
# using the formula you provided.
# Finally, it prints the result.

# Note that the code assumes that your data file has columns named tweets, likes, neg, neu, and pos containing the corresponding values for each tweet. If your data file has different column names, you will need to modify the code accordingly.
df = df[['created_at', 'text', 'likes', 'retweets', 'neg', 'neu', 'pos']]

# group the data by date and calculate the weighted average for each sentiment

grouped = df.groupby('created_at')

# calculate weighted average for neg, neu, and pos, add one to each value to include the tweet itself in the calculation
weighted_avg = grouped.apply(lambda x: (
    ((x['retweets']+1)*x['neg']) + ((x['likes']+1)*x['neg'])) / (x['retweets'] + x['likes'] + 2))

weighted_avg.columns = ['created_at', 'neg']
print(weighted_avg.head())
