import time
import tweepy
import matplotlib.pyplot as plt
import sys
import datetime
import json
import numpy as np
import pandas as pd
print("pandas version is ", pd.__version__)


my_bearer_token = 'AAAAAAAAAAAAAAAAAAAAADigdwEAAAAAyGEJIbrn1BPYIMXFEiICWFaB1%2F0%3DW2q0OE9Bf0w1qZGvRwwiwwE6rreDjJP3MaIn1abqaMTuxFFOSE'

API_key = 'OEnnzk5ro76PdhQHTJVxa3qTv'
# API_key = 'qchdahMH5QJp0xd1dPQPTdxAJ'
API_SECRET_KEY = 'sj4lfiejDGosykNA62YaHqBP1J9RyESbMqKboxQp9sW51gJylC'

how_many_tries = 32
items_to_get = 100
whats_today = datetime.datetime.now()
# whats_today = whats_today.strftime("%Y%m%d%H%M")


minus_one = datetime.datetime.now() - datetime.timedelta(days=1)

minus_two = datetime.datetime.now() - datetime.timedelta(days=2)

minus_thirty = datetime.datetime.now() - datetime.timedelta(days=30)


# minus_thirty = datetime.datetime.strptime(minus_thirty, "%Y%M%d%H%m")


print(sys.version)
print("uncorrected today's date is ", whats_today)


def get_them_tweets(src_wd_1, src_wd_2, num_hrs, num_tweets):

    tweetz = []
    for i in range(num_hrs):
        time.sleep(0.1)
        strt_date = datetime.datetime.now() - datetime.timedelta(hours=i)
        stp_date = datetime.datetime.now() - datetime.timedelta(hours=i+1)

        client = tweepy.Client(bearer_token=my_bearer_token)
        search_words = src_wd_1


# tweets = api.search_tweets(q = search_words, lang = "en", count = items_to_get )
        tweets = client.search_recent_tweets(query=search_words, start_time=stp_date, end_time=strt_date,
                                             tweet_fields=[
                                                 'id', 'text', 'created_at', 'lang', 'public_metrics', 'author_id'],

                                             max_results=num_tweets)

        for tweet in tweets.data:
            tweet_info = {
                'created_at': tweet.created_at,
                'user_id': tweet.author_id,
                'text': tweet.text,
                'tweet_id': tweet.id,
                'lang': tweet.lang,
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count']
            }
            tweetz.append(tweet_info)

        search_words = src_wd_2

        tweets = client.search_recent_tweets(query=search_words, start_time=stp_date, end_time=strt_date,
                                             tweet_fields=[
                                                 'id', 'text', 'created_at', 'lang', 'public_metrics', 'author_id'],
                                             max_results=num_tweets)
        for tweet in tweets.data:
            tweet_info = {
                'created_at': tweet.created_at,
                'user_id': tweet.author_id,
                'text': tweet.text,
                'tweet_id': tweet.id,
                'lang': tweet.lang,
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count']
            }
            tweetz.append(tweet_info)
    return tweetz


headley = ['index', 'created_at', 'user_id', 'text', 'source',
           'lang', 'likes', 'retweets']
tweets_AMZN = get_them_tweets('AMZN', 'Amazon', how_many_tries, items_to_get)

Amazon_df = pd.DataFrame(tweets_AMZN)


print(Amazon_df.head())


Amazon_df.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_api2.csv', mode='a', index=True, header=False)

print("the AMZN_rob shape is", Amazon_df.shape)

# ricky_bobby = pd.read_csv(
# '/Volumes/Elements/GitHub/twitter-project/Data Files/Amazon_df_api2.csv', header='infer')

ricky_bobby = pd.read_table(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_api2.csv', sep=',', header='infer')
print(ricky_bobby.head())
print("the initial shape is ", ricky_bobby.shape)
##
ricky_bobby = ricky_bobby[ricky_bobby['lang'] == 'en']
ricky_bobby = ricky_bobby.drop_duplicates()
ricky_bobby.reset_index(inplace=True, drop=True)


print("the final shape is ", ricky_bobby.shape)

ricky_bobby.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_api2.csv', index=False)
print("saulgoodman Amazon")
#################

tweets_ATEN = get_them_tweets('ATEN', 'A10', how_many_tries, items_to_get)

ATEN_df = pd.DataFrame(tweets_ATEN)

headley2 = ['created_at', 'user_id', 'text', 'source',
            'lang', 'likes', 'retweets']
ATEN_df.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_df_api2.csv', mode='a', index=True, header=False)

ricky_bobby = pd.read_table(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_df_api2.csv', sep=',', header='infer')
print(ricky_bobby.head())
print("the initial shape is ", ricky_bobby.shape)
##
ricky_bobby = ricky_bobby[ricky_bobby['lang'] == 'en']
ricky_bobby = ricky_bobby.drop_duplicates()
ricky_bobby.reset_index(inplace=True, drop=True)


print("the final shape is ", ricky_bobby.shape)

ricky_bobby.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_df_api2.csv', index=False)
print("saulgoodman ATEN")

#################
tweets_ATEN = get_them_tweets('ATEN', 'A10', how_many_tries, items_to_get)
print("OK jot")
tweets_KMI = get_them_tweets(
    'KMI', 'Kinder', how_many_tries, items_to_get)

KMI_df = pd.DataFrame(tweets_KMI)

headley2 = ['created_at', 'user_id', 'text', 'source',
            'lang', 'likes', 'retweets']
KMI_df.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_df_api2.csv', mode='a', index=True, header=False)

ricky_bobby = pd.read_table(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_df_api2.csv', sep=',', header='infer')
print(ricky_bobby.head())
print("the initial shape is ", ricky_bobby.shape)
##
ricky_bobby = ricky_bobby[ricky_bobby['lang'] == 'en']
ricky_bobby = ricky_bobby.drop_duplicates()
ricky_bobby.reset_index(inplace=True, drop=True)


print("the final shape is ", ricky_bobby.shape)

ricky_bobby.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_df_api2.csv', index=False)
print("saulgoodman KMI")
