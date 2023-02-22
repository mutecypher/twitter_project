from nltk.corpus import stopwords
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
import emoji

max_tokens = 30

pad = 'pre'

model_file = '/Volumes/Elements/GitHub/twitter-project/twitter_project/19_best_model.h5'

loaded_model = load_model(
    model_file,
    custom_objects=None,
    compile=True
)


def clean_it_up(text):
    texty_yo = re.sub(r'http?:\/\/.\S+', "", text)
    texty_yo = re.sub(r'#', '', texty_yo)
    texty_yo = re.sub(r'^RT[\s]+', '', texty_yo)
    texty_yo = re.sub(r'[^a-zA-Z ]+', '', texty_yo)
    texty_yo = re.sub(r' +', ' ', texty_yo)
    texty_yo = texty_yo.lower()
    texty_yo = texty_yo[0: 300]
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
    # stop_words = stopwords.words('english')
    # df['text'] = df['text'].apply(
    # lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    return df


def assign_scores(filington, colington):
    filington.columns = colington
    filington = clean_the_text_col(filington)
    text_list = filington['text'].astype(str).tolist()
    num_words = 30000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)
    x_test = pad_sequences(sequences, maxlen=max_tokens,
                           padding=pad, truncating=pad)
    y_pred = loaded_model.predict(x_test)
    predictions = pd.DataFrame(y_pred, columns=['neg', 'neu', 'pos'])
    new_file = filington.join(predictions)
    return new_file


def trans_scores(filington, colington):
    sentiment_pipeline = pipeline(
        'sentiment-analysis',
        model="finiteautomata/bertweet-base-sentiment-analysis")
    # 'distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')
    filington.columns = colington
    filington = clean_the_text_col(filington)
    text_list = filington['text'].astype(str).tolist()
    # text_list = text_list[0:19]
    print("got to here in trans_scores")
    # specific_model = sentiment_pipeline(
    # model="finiteautomata/bertweet-base-sentiment-analysis")
    y_trans_pred = sentiment_pipeline(text_list)
    # print(y_trans_pred)
    t_file = pd.DataFrame(y_trans_pred, columns=['label', 'score'])
    return t_file


def normy(file):
    file[['summy', 'neg1', 'neu1', 'pos1']] = 0
    file['neg1'] = file['neg']/(file['neg'] + file['neu'] + file['pos'])
    file['neu1'] = file['neu']/(file['neg'] + file['neu'] + file['pos'])
    file['pos1'] = file['pos']/(file['neg'] + file['neu'] + file['pos'])
    file['summy'] = file['neg1'] + file['neu1'] + file['pos1']
    file['neg'] = file['neg1']
    file['neu'] = file['neu1']
    file['pos'] = file['pos1']
    file = file.drop(columns=['neg1', 'neu1', 'pos1'])
    return file


def relabel(file):
    neg_mask = file['label'] == 'NEG'
    neu_mask = file['label'] == 'NEU'
    pos_mask = file['label'] == 'POS'
    score = file['score']

    file['neg'] = np.where(neg_mask, score, (1 - score)/2)
    file['neu'] = np.where(neu_mask, score, (1 - score)/2)
    file['pos'] = np.where(pos_mask, score, (1 - score)/2)

    file['neg'] = np.where(neu_mask, (1 - score)/2, file['neg'])
    file['neu'] = np.where(neg_mask, (1 - score)/2, file['neu'])
    file['pos'] = np.where(neg_mask, (1 - score)/2, file['pos'])
    file['neg'] = np.where(pos_mask, (1 - score)/2, file['neg'])
    file['neu'] = np.where(pos_mask, (1 - score)/2, file['neu'])

    return file


amzn_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_json.csv'

amzn_file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_api2.csv'

amzn_df = pd.read_csv(amzn_file, header=0, index_col=0, parse_dates=True)
amzn_df2 = pd.read_csv(amzn_file2, header=0,
                       index_col=0, parse_dates=True)

common_cols = ['created_at', 'user_id', 'text',
               'lang', 'likes', 'retweets', 'tweet_id']
am_columns = ['numbers', 'created_at', 'text', 'retweet_count', 'user_id',
              'user.favourites_count', 'user.followers_count', 'AMZN', 'Amazon', 'dupe']

amzn_df.columns = am_columns
print()
amzn_df = amzn_df.drop_duplicates()
print("got the first amzn_df done")
print("the shape of the first amzn_df2 is: ", amzn_df2.shape)
print("the shape of the common_cols is: ", len(common_cols))
print()
amzn_df2 = amzn_df2[amzn_df2.columns.intersection(common_cols)]
amzn_df2.columns = common_cols
amzn_df2 = amzn_df2.drop_duplicates()

amzn_df_nn = assign_scores(amzn_df, am_columns)
print("Assigning scores to amzn_df_nn")
amzn_df_nn = normy(amzn_df_nn)
amzn_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_nn_scored.csv', header=True)

print()
print(amzn_df_nn.head())
print()
print("Assigning scores to amzn_df2_nn")
amzn_df2_nn = assign_scores(amzn_df2, common_cols)
print("the head of amzn_df2_nn is \n", amzn_df2_nn.head())
amzn_df2_nn = normy(amzn_df2_nn)


amzn_df2_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_2_nn_scored.csv', header=True)


print()
print("amzn_df2_nn, the head is \n",
      amzn_df2_nn[['pos', 'neu', 'neg', 'summy']].head())
print()

strt = datetime.datetime.now()
amzn_df_tn = trans_scores(amzn_df2, common_cols)
end = datetime.datetime.now()
print("the  time for the first amzn_df_tn is: ", end - strt)
print("About to relabel amzn_df_tn \n", amzn_df_tn.head())
amzn_df_tn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_tn_scored.csv', header=True)

amzn_df_tn = relabel(amzn_df_tn)

print("Have normalize amzn_df_tn \n", amzn_df_tn.head())
amzn_df_tn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_tn_scored.csv', header=True)
print()
print("now the transformers version")
print("the Amazon transformers head is ", amzn_df_tn.head())
print()
#######
# And now the original file
# learn_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/twitter_sentiment_learn.csv'
# learn_columns = ['TWID', 'NEG', 'NEU', 'POS', 'text']

# learn_df = pd.read_csv(learn_file, header=0, index_col=0, parse_dates=True)
# learn_df.columns = learn_columns
# print("the learn df head is: \n", learn_df.head())

# learn_nn = assign_scores(learn_df, learn_columns)

print("completed learn with nn")
print()
# learn_nn.to_csv(
# '/Volumes/Elements/GitHub/twitter-project/Data_Files/learn_nn_scored.csv', header=True)

# learn_tf = trans_scores(learn_df, learn_columns)
print("completed learn with tf")
print()
# learn_tf.to_csv(
# '/Volumes/Elements/GitHub/twitter-project/Data_Files/learn_tf_scored.csv', header = True)


########
# Inspire
insp_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_df_json.csv'

insp_file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_df_api2.csv'

insp_columns = ['created_at', 'text', 'retweet_count', 'user_id',
                'user.favourites_count', 'user.followers_count' 'INSP', 'Inspire', 'dupe', 'dupe']


insp_df = pd.read_csv(insp_file, header=0, index_col=0, parse_dates=True)


# insp_df.columns = insp_columns
print()
print("The shape of the inspire df is: ", insp_df.shape)
insp_df2 = pd.read_csv(insp_file2, header=0,    index_col=0, parse_dates=True)

insp_df2 = insp_df2[insp_df2.columns.intersection(common_cols)]

insp_df2 = insp_df2.drop_duplicates()
inspf_df = insp_df.drop_duplicates()
insp_df2.columns = common_cols
print("The shape of the inspire df2 is: ", insp_df2.shape)
print()

print("Doing the inspire df now, with nn.")
insp_df_nn = assign_scores(insp_df, insp_columns)

insp_df_nn = normy(insp_df_nn)
insp_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_nn_scored.csv', header=True)

insp_df2_nn = assign_scores(insp_df2, common_cols)

insp_df2_nn = normy(insp_df2_nn)
insp_df2_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_nn2_scored.csv', header=True)

print()
print(insp_df_nn.head())
print()

########


print()
print(insp_df2_nn.head())
print()

####
print("doing the inspire df now, with tn.")
strt = datetime.datetime.now()
insp_df_tn = trans_scores(insp_df, insp_columns)
end = datetime.datetime.now()
print("the  time for the first insp_df_tn is: ", end - strt)

insp_df_tn = relabel(insp_df_tn)
insp_df_tn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_tn_scored.csv', header=True)
print()
print("now the transformers version of INSP and it took ", end - strt, "to run")
print(insp_df_tn.head())
print()


###

aten_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_df_api2.csv'

aten_columns = ['created_at', 'text', 'retweet_count', 'user_id',
                'user.favourites_count', 'user.followers_count', 'ATEN', 'Aten', 'dupe']

aten_df = pd.read_csv(aten_file, header=0, index_col=0, parse_dates=True)

aten_df = aten_df[aten_df.columns.intersection(common_cols)]
aten_df.columns = common_cols

aten_df_nn = assign_scores(aten_df, common_cols)

aten_df_nn = normy(aten_df_nn)
aten_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_nn_scored.csv', header=True)

aten_df = aten_df.drop_duplicates()
print()
print("aten shape is ", aten_df.shape)
print()

####

strt = datetime.datetime.now()
aten_df_tn = trans_scores(aten_df, common_cols)
end = datetime.datetime.now()

aten_df_tn = relabel(aten_df_tn)
print("the  transformers version of aten is ", aten_df_tn.head())
aten_df_tn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_tn_scored.csv', header=True)
print()
print("now the transformers version and it took ", end - strt, "to run")


###

KMI_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_df_api2.csv'

KMI_columns = ['created_at', 'user_id', 'text', 'tweet_id',
               'lang', 'likes', 'retweet_count']

KMI_df = pd.read_csv(KMI_file, header=0, index_col=0, parse_dates=True)

KMI_df = KMI_df[KMI_df.columns.intersection(common_cols)]
KMI_df.columns = KMI_columns

KMI_df = KMI_df.drop_duplicates()
strt = datetime.datetime.now()
KMI_df_nn = assign_scores(KMI_df, common_cols)
end = datetime.datetime.now()

KMI_df_nn = normy(KMI_df_nn)
print("the time for the first KMI_df_nn is: ", end - strt)
KMI_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_nn_scored.csv', header=True)

print()

print(KMI_df_nn.head())
print()

####

strt = datetime.datetime.now()
KMI_df_tn = trans_scores(KMI_df, common_cols)
end = datetime.datetime.now()

KMI_df_tn = relabel(KMI_df_tn)
print("the  transformers version for KMI is done\n", KMI_df_tn.head())


print("now the transformers version of KMI and it took ", end - strt, "to run")


###

Exxon_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_df_api2.csv'

Exxon_columns = ['created_at', 'text', 'retweet_count', 'user_id',
                 'user.favourites_count', 'user.followers_count', 'XOM', 'Exxon', 'dupe']

Exxon_df = pd.read_csv(Exxon_file, header=0,
                       index_col=0, parse_dates=True)

Exxon_df = Exxon_df[Exxon_df.columns.intersection(common_cols)]

Exxon_df.columns = common_cols

Exxon_df = Exxon_df.drop_duplicates()

strt = datetime.datetime.now()
Exxon_df_nn = assign_scores(Exxon_df, common_cols)
end = datetime.datetime.now()

Exxon_df_nn = normy(Exxon_df_nn)
print("the time for the first Exxon_df_nn is: ", end - strt)
Exxon_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_nn_scored.csv', header=True)

print()

print(Exxon_df_nn.head())
print()

####
strt = datetime.datetime.now()
Exxon_df_tn = trans_scores(Exxon_df, common_cols)
end = datetime.datetime.now()

Exxon_df_tn = relabel(Exxon_df_tn)
print("the  transformers version", Exxon_df_tn.head())
Exxon_df_tn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_tn_scored.csv', header=True)
print()
print("now the transformers version and it took ", end - strt, "to run")


KMI_df_tn = relabel(KMI_df_tn)
KMI_df_tn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_tn_scored.csv', header=True)

print()
print("Finally got the KMI.")
