import os
from bs4 import BeautifulSoup
import requests
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


# Define a function to calculate weighted average


def weighted_avg(group):
    xy_sum = group['retweets'] + group['likes']
    poz = group['pos'] * xy_sum
    neut = group['neu'] * xy_sum
    nega = group['neg'] * xy_sum

    return poz.sum()/xy_sum.sum(), neut.sum()/xy_sum.sum(), nega.sum()/xy_sum.sum()


wgt_r = 1.0
wgt_l = 0.5
df['pos_wgt'] = df['pos'] * ((df['retweets'] * wgt_r) + (df['likes'] * wgt_l))
df['neu_wgt'] = df['neu'] * ((df['retweets'] * wgt_r) + (df['likes'] * wgt_l))
df['neg_wgt'] = df['neg'] * ((df['retweets'] * wgt_r) + (df['likes'] * wgt_l))


# df1 = pd.DataFrame(df1)
df1 = df.drop(['created_at', 'text', 'user_id', 'lang', 'tweet_id'], axis=1)
##df1 = df1.reset_index()
print("the shape of df1 is ", df1.shape)
print("the head of df1 is ", df1.head())

df2 = df1.sort_values(by=['Date', 'pos_wgt'], ascending=[True, False])

print("the head of df2 is \n", df2.head())

df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m-%d')
df3 = df2.loc[(df2['Date'] == '2022-06-27')]

print("the head of df3 is \n",
      df3[['pos', 'neg', 'neu', 'likes', 'retweets', 'pos_wgt', 'neg_wgt', 'neu_wgt']].head())


url = 'https://example.com'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
img_tags = soup.find_all('img')

for img in img_tags:
    img_url = img['src']
    filename = os.path.join('images', img_url.split('/')[-1])
    with open(filename, 'wb') as f:
        f.write(requests.get(img_url).content)
