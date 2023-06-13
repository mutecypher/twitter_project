# import os
# from bs4 import BeautifulSoup
# import requests
import pandas as pd
import numpy as np
import datetime


def comb_and_agg(df):

    # Line below is "new"
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce').dt.tz_convert(
        'US/Eastern').dt.tz_localize(None)

# Remove rows with null values
    df = df[pd.notnull(df['created_at'])]
    if 'likes' not in df.columns:
        if 'user.favourites_count' in df.columns:
            df = df.rename(columns={'user.favourites_count': 'likes'})
        else:
            raise ValueError(
                'Could not user.favourites_count column in dataframe')
    if 'retweets' not in df.columns:
        if 'retweet_count' in df.columns:
            df = df.rename(columns={'retweet_count': 'retweets'})
        else:
            raise ValueError('Could not retweet_count column in dataframe')

    # Note that the code assumes that your data file has columns named tweets, likes, neg, neu, and pos containing the corresponding values for each tweet. If your data file has different column names, you will need to modify the code accordingly.
    df = df[['created_at', 'text', 'likes', 'retweets', 'neg', 'neu', 'pos']]
    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%m/%d/%Y')
    df = df.sort_values(by='created_at', ascending=False)

    # df['likes'] = df['likes'].apply(lambda x: int(
    # x) if str(x).isdigit() else None).dropna()
    # df['retweets'] = df['retweets'].apply(
    # lambda x: int(x) if str(x).isdigit() else None).dropna()
    df['likes'] = df['likes'].apply(pd.to_numeric, errors='coerce')
    df['retweets'] = df['retweets'].apply(pd.to_numeric, errors='coerce')
    # df['neu'] = df['neu'].apply(lambda x: float(x) if str(
    # x).replace('.', '').isdigit() else None).dropna()
    # df['neg'] = df['neg'].apply(lambda x: float(x) if str(
    # x).replace('.', '').isdigit() else None).dropna()
    # df['pos'] = df['pos'].apply(lambda x: float(x) if str(
    # x).replace('.', '').isdigit() else None).dropna()
    df = df.dropna()
    df = df.reset_index(drop=True)

    print()
    # group the data by date and calculate the weighted average for each sentiment

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

        created_dates.append(df.loc[a, 'created_at'])
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

    return agged_df


file1 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_2_nn_scored.csv'

df = pd.read_csv(file1, header='infer')
# save the result to a CSV file
try1 = comb_and_agg(df)
try1.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_agged_df.csv')

print("the shape of the Amazon_2_nn_scored redone is ", try1.shape)
print("the tail of the Amazon_2_nn_scored redone is ", try1.tail())

file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_nn_scored.csv'

df = pd.read_csv(file2, header='infer')
print()
print("the shape of the Amazon_nn_scored is ", df.shape)
print()
df['retweets'] = df['retweet_count']
df['likes'] = df['user.favourites_count']
# save the result to a CSV file
try2 = comb_and_agg(df)
print()
print("the shape of the Amazon_2_agged_df is ", try2.shape)
print()
try2.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_2_agged_df.csv')

print("the shape of the Amazon_2_agged_df redone is ", try2.shape)
print("the tail of the Amazon_2_agged_df redone is ", try2.tail())


file3 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_tn_scored.csv'

df = pd.read_csv(file3, header='infer')
print()
print("the shape of the Amazon_tn_norm is ", df.shape)
print()
# df['retweets'] = df['retweet_count']
# df['likes'] = df['user.favourites_count']
# save the result to a CSV file
try3 = comb_and_agg(df)
try3.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_3_agged_df.csv')

print("the shape of the Amazon_tn_norm redone is ", try3.shape)
print("the tail of the Amazon_tn_norm redone is ", try3.tail())


file4 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_tn_norm.csv'
file5 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_nn_scored.csv'
file6 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_tn_scored.csv'

df4 = pd.read_csv(file4, header='infer')
df5 = pd.read_csv(file5, header='infer')
df6 = pd.read_csv(file6, header='infer')

print()
try4 = comb_and_agg(df4)
try4.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_tn_norm_agged.csv')
print("the shape of the ATEN_tn_norm redone is ", try4.shape)

print()
try5 = comb_and_agg(df5)
try5.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_nn_scored_agged.csv')
print("the shape of the ATEN_nn_scored redone is ", try5.shape)

print()
try6 = comb_and_agg(df6)
try6.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_tn_scored_agged.csv')
print("the shape of the ATEN_tn_scored redone is ", try6.shape)


file7 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_tn_norm.csv'
file8 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_nn_scored.csv'
file9 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_tn_scored.csv'

df7 = pd.read_csv(file7, header='infer')
df8 = pd.read_csv(file8, header='infer')
df9 = pd.read_csv(file9, header='infer')

print()
try7 = comb_and_agg(df7)
try7.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files//Exxon_tn_norm_agged.csv')
print("the shape of the Exxon_tn_norm redone is ", try7.shape)

print()
try8 = comb_and_agg(df8)
try8.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files//Exxon_nn_scored_agged.csv')
print("the shape of the Exxon_nn_scored redone is ", try8.shape)

print()
try9 = comb_and_agg(df9)
try9.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files//Exxon_tn_scored_agged.csv')
print("the shape of the Exxon_tn_scored redone is ", try9.shape)


file10 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_tn_norm.csv'
file11 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_nn_scored.csv'
file12 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_tn_scored.csv'

df10 = pd.read_csv(file10, header='infer')
df11 = pd.read_csv(file11, header='infer')
df12 = pd.read_csv(file12, header='infer')

print()
try10 = comb_and_agg(df10)
try10.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files//INSP_tn_norm_agged.csv')
print("the shape of the INSP_tn_norm redone is ", try10.shape)

print()
try11 = comb_and_agg(df11)
try11.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files//INSP_nn_scored_agged.csv')
print("the shape of the INSP_nn_scored redone is ", try11.shape)

print()
try12 = comb_and_agg(df12)
try12.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files//INSP_tn_scored_agged.csv')
print("the shape of the INSP_tn_scored redone is ", try12.shape)


file13 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_tn_norm.csv'
file14 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_nn_scored.csv'
file15 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_tn_scored.csv'

df13 = pd.read_csv(file13, header='infer')
df14 = pd.read_csv(file14, header='infer')
df15 = pd.read_csv(file15, header='infer')

print()
try13 = comb_and_agg(df13)
try13.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_tn_norm_agged.csv')
print("the shape of the KMI_tn_norm redone is ", try13.shape)

print()
try14 = comb_and_agg(df14)
try14.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_nn_scored_agged.csv')
print("the shape of the KMI_nn_scored redone is ", try14.shape)

print()
try15 = comb_and_agg(df15)
try15.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_tn_scored_agged.csv')
print("the shape of the KMI_tn_scored redone is ", try15.shape)
