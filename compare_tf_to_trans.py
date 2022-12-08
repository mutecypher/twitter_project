import datetime
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras
import math
import numpy as np
import tensorflow as tf
import re
import random
from transformers import pipeline


def clean_it_up(text):
    texty_yo = re.sub(r'https?:\/\/.\S+', "", text)
    texty_yo = re.sub(r'#', '', texty_yo)
    texty_yo = re.sub(r'^RT[\s]+', '', texty_yo)
    texty_yo = re.sub(r'[^a-zA-Z ]+', '', texty_yo)
    texty_yo = re.sub(r' +', ' ', texty_yo)
    texty_yo = texty_yo.lower()
    return texty_yo


def clean_output(text):
    cln = str(text)
    cln = cln.strip('[')
    cln = cln.strip(']')
    cln = cln.strip("'")
    return cln


def clean_the_text_col(df):
    df['text'] = df['text'].apply(clean_output)
    df['text'] = df['text'].apply(clean_it_up)

    return df


def trans_scores(filington, colington):
    sentiment_pipeline = pipeline(
        'sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")
    filington.columns = colington
    filington = clean_the_text_col(filington)
    text_list = filington['text'].astype(str).tolist()
    print("got to here")
    y_trans_pred = sentiment_pipeline(text_list)
    t_file = pd.DataFrame(y_trans_pred, columns=['label', 'score'])
    return t_file


def trans_scorez(filington, colington):
    sentiment_pipeline = pipeline(
        'sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")
    filington.columns = colington
    filington = clean_the_text_col(filington)

    # Define a function that applies the sentiment analysis pipeline to a single row
    def apply_pipeline(row):
        text = row['text']
        y_trans_pred = sentiment_pipeline(text)
        return y_trans_pred

    # Use the apply method to apply the sentiment analysis pipeline to each row
    y_trans_pred_2 = filington.apply(apply_pipeline, axis=1)
    t_file1 = pd.DataFrame(y_trans_pred_2, columns=['label', 'score'])
    return t_file1


file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/twitter_sentiment_learn.csv'

learning_df = pd.read_csv(file)
print(learning_df.shape)
print(learning_df.head())


def find_the_most(df):
    if (df.loc["NEG"] >= df.loc["NEU"]) and (df.loc["NEG"] >= df.loc["POS"]):
        df.loc["bad"] = 1
    elif (df.loc["NEU"] >= df.loc["NEG"]) and (df.loc["NEU"] >= df.loc["POS"]):
        df.loc["meh"] = 1
    else:
        df.loc["good"] = 1

    return df


learning_df[["neg", "neu", "pos"]] = 0

# strt = datetime.datetime.now()
# for i in range(learning_df.shape[0]):
# if (learning_df.loc[i, "NEG"] >= learning_df.loc[i, "NEU"]) and (learning_df.loc[i, "NEG"] >= learning_df.loc[i, "POS"]):
# learning_df.loc[i, "bad"] = 1
# elif (learning_df.loc[i, "NEU"] >= learning_df.loc[i, "NEG"]) and (learning_df.loc[i, "NEU"] >= learning_df.loc[i, "POS"]):
# learning_df.loc[i, "meh"] = 1
# else:
# learning_df.loc[i, "good"] = 1
# end = datetime.datetime.now()
# print("finished assigning scores with for loop, and it took ", end - strt)

strt = datetime.datetime.now()

max_value = learning_df[['NEG', 'NEU', 'POS']].max(axis=1)
learning_df['neg'] = (learning_df['NEG'] == max_value)
learning_df['neu'] = (learning_df['NEU'] == max_value)
learning_df['pos'] = (learning_df['POS'] == max_value)
end = datetime.datetime.now()
print("finished assigning scores via map, and it took ", end - strt)

# learning_df["verdict"] = 'i'

# strt = datetime.datetime.now()
# for i in range(learning_df.shape[0]):
# if learning_df.loc[i, "bad"] == 1:
# learning_df.loc[i, "verdict"] = 'NEG'
# elif learning_df.loc[i, "meh"] == 1:
# learning_df.loc[i, "verdict"] = 'NEU'
# else:
# learning_df.loc[i, "verdict"] = 'POS'
# end = datetime.datetime.now()
# print("finished assigning verdicts, and it took ", end - strt)
print()


print()
# print("finished the verdicts")

strt = datetime.datetime.now()
# Create a dictionary mapping each column name to a corresponding verdict
column_mapping = {'neg': 'NEG', 'neu': 'NEU', 'pos': 'POS'}

# Create a new column called 'verdict' and initialize it to an empty string
learning_df['verdict_dict'] = ''

# Loop through the columns in the dataframe
for column in learning_df.columns:
    # If the column is one of the three columns we are interested in
    if column in column_mapping:
        # Set the corresponding value in the 'verdict' column to be the corresponding verdict
        # if the value in the original dataframe is 1, and do nothing otherwise
        learning_df.loc[learning_df[column] == 1,
                        'verdict_dict'] = column_mapping[column]
end = datetime.datetime.now()

print("finished assigning verdicts_dict, and it took ", end - strt)
learning_df = learning_df.drop(
    ['NEG', 'NEU', 'POS', 'neg', 'neu', 'pos'], axis=1)

learning_df.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/bare_learning_df.csv', index=False)
print("saved the new file")
common_cols = ['index', 'TWID',
               'text',  'verdict_dict']


print("starting the trans scores \n", learning_df.head())
strt = datetime.datetime.now()
learn_df_tn = trans_scores(learning_df, common_cols)
end = datetime.datetime.now()

learn_df_tn.columns = ['label', 'score']
learning_df['Tforms'] = learn_df_tn['label']
learning_df.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/short_bare_tf.csv', index=False)

print()
print("Time to run transformers in my form: ", end - strt)
print()

common_colz = ['index', 'TWID',
               'text',  'verdict_dict', 'Tforms']
print("starting the chatGPT optimized scores \n", learning_df.head())
strt = datetime.datetime.now()
##learn_df_tn = trans_scorez(learning_df, common_colz)
end = datetime.datetime.now()

print()
print("Time to run transformers in chatGPT form: ", end - strt)
print()

# learn_df_tn.to_csv(
# '/Volumes/Elements/GitHub/twitter-project/Data_Files/learn_df_tf.csv', index=False)
##learn_df_tn.columns = ['labelz', 'scorez']
##learning_df['Tformz'] = learn_df_tn['labelz']
# learning_df['Tscore'] = learn_df_tn['score']
# learning_df.to_csv(
# '/Volumes/Elements/GitHub/twitter-project/Data_Files/learning_tf.csv', index=False)
# print(learning_df.head())


conditions = [[learning_df['verdict_dict'] == learning_df['Tforms']],
              learning_df['verdict_dict'] != learning_df['Tforms']]

choices = [1, 0]

print("now running the choices")
##learning_df['match'] = np.select(conditions, choices, default=1)
learning_df['match'] = np.where(
    learning_df['verdict_dict'] == learning_df['Tforms'], 1, 0)
print("the shape is ", learning_df.shape)
print("the number of matches is ", learning_df['match'].sum())

learning_df.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/trans_compare.csv', header=True, index=False)


print()
print("head of the file \n", learning_df.head())
