# import os
# from bs4 import BeautifulSoup
# import requests
import pandas as pd
import numpy as np
import datetime

file1 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_2_nn_scored.csv'


def comb_and_agg(df):

    # df.created_at = pd.to_datetime(df.created_at)

    df.created_at = pd.to_datetime(df['created_at']).dt.strftime('%m/%d/%Y')

    # Note that the code assumes that your data file has columns named tweets, likes, neg, neu, and pos containing the corresponding values for each tweet. If your data file has different column names, you will need to modify the code accordingly.
    df = df[['created_at', 'text', 'likes', 'retweets', 'neg', 'neu', 'pos']]
    df = df.sort_values(by='created_at')
    df = df.reset_index(drop=True)
    df = df.dropna()
    print()
    print("the dataframe head is ", df.head())
    print()
    print("the dataframe shape is ", df.shape)
    print()
    print("the dataframe columns are ", df.columns)
    print()
    # group the data by date and calculate the weighted average for each sentiment

    df.to_csv(
        '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_dummy.csv', index=False)
    a = 0
    created_dates = []
    agged_neg = []
    agged_neu = []
    agged_pos = []
    # print("the contents of row 56600 and neg is ", df.loc[df.shape[0]-1, 'neg'])
    # print("the contents of row 56600 and created_at is ",df.loc[df.shape[0]-1, 'created_at'])

    while a < (df.shape[0] - 2):
        temp_neg = 0
        temp_neu = 0
        temp_pos = 0
        temp_likes = 0
        temp_retweets = 0
        # print("the value of a is ", a)
        while df.loc[a, 'created_at'] == df.loc[a+1, 'created_at'] and (a < (df.shape[0] - 2)):
            # print("the value of a is ", a)
            temp_neg += (df.loc[a, 'neg'] *
                         (df.loc[a, 'likes'] + 2 + df.loc[a, 'retweets']))
            temp_neu += (df.loc[a, 'neu'] *
                         (df.loc[a, 'likes'] + 2 + df.loc[a, 'retweets']))
            temp_pos += (df.loc[a, 'pos'] *
                         (df.loc[a, 'likes'] + 2 + df.loc[a, 'retweets']))
            temp_likes += df.loc[a, 'likes'] + 1
            temp_retweets += df.loc[a, 'retweets'] + 1
            # print("the value of a is ", a)
            a += 1

        created_dates.append(df.loc[a - 1, 'created_at'])
        if (temp_likes + temp_retweets) > 2:
            agged_neg.append(temp_neg / (temp_likes + temp_retweets))
            agged_neu.append(temp_neu / (temp_likes + temp_retweets))
            agged_pos.append(temp_pos / (temp_likes + temp_retweets))
        else:
            agged_neg.append(temp_neg / 2)
            agged_neu.append(temp_neu / 2)
            agged_pos.append(temp_pos / 2)
        # print("the value of a is ", a)
        # print("the sum of the values is ", (temp_neg +temp_neu + temp_pos)/(temp_likes + temp_retweets))
        a += 1

    dict = {
        'created_at': created_dates,
        'neg': agged_neg,
        'neu': agged_neu,
        'pos': agged_pos
    }
    agged_df = pd.DataFrame(dict)

    # print the result
    print()
    print("The shape of the aggregated dataframe is \n", agged_df.shape)
    print()
    print("The aggregated head is \n", agged_df.head())
    return agged_df


df = pd.read_csv(file1, header='infer', index_col=0)
# save the result to a CSV file
try1 = comb_and_agg(df)
try1.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_agged_df.csv')

print("the shape of the dataframe is ", try1.shape)
print("the tail of the dataframe is ", try1.tail())
