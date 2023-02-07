
from tensorflow.keras.activations import relu, sigmoid, softmax, tanh, hard_sigmoid, softsign, softplus, linear
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard, LambdaCallback
import datetime
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import sklearn.model_selection as sk
from skopt.utils import use_named_args
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_convergence
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize, forest_minimize
import skopt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam, Ftrl, Adamax, SGD, Adadelta, Nadam, Optimizer, RMSprop, Adagrad
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

max_tokens = 30

pad = 'pre'

model_file = '/Volumes/Elements/GitHub/twitter-project/twitter_project/19_best_model.h5'

loaded_model = load_model(
    model_file,
    custom_objects=None,
    compile=True
)


amzn_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_json.csv'

amzn_file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_api2.csv'

amzn_df = pd.read_csv(amzn_file, header=0, index_col=0, parse_dates=True)
print()
print("The column headers are ", list(amzn_df.columns.values))
print()


amzn_df2 = pd.read_csv(amzn_file2, header=0, index_col=0, parse_dates=True)

print()
print("The column headers are ", list(amzn_df2.columns.values))
print()

am2_cols = ['created_at', 'user_id', 'text',
            'tweet_id', 'lang', 'likes', 'retweets']
am_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
              'user.favourites_count', 'user.followers_count', 'AMZN', 'Amazon', 'dupe']
amzn_df.columns = am_columns

print()
print("the shape of amzn_df2 is ", amzn_df2.shape)
amzn_df2.columns = am2_cols


amzn_text = amzn_df['full_text'].astype(str)

amzn_text2 = amzn_df2['text'].astype(str)

amzn_x = amzn_text.to_list()

amzn_x2 = amzn_text2.to_list()

bob = amzn_x[0:7]

print()
print("the type for bob is ", type(bob))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(amzn_x)

amzn_train_tokens = tokenizer.texts_to_sequences(amzn_x)
amzn2_train_tokens = tokenizer.texts_to_sequences(amzn_x2)


amzn_pad = pad_sequences(amzn_train_tokens, maxlen=max_tokens,
                         padding=pad, truncating=pad)

amzn_sent = loaded_model.predict(x=amzn_pad)

amzn_pad2 = pad_sequences(amzn2_train_tokens, maxlen=max_tokens,
                          padding=pad, truncating=pad)

amzn_sent2 = loaded_model.predict(x=amzn_pad2)


print("The old Amazon data is \n", amzn_sent[1:3])
print()
print("The new Amazon data is \n", amzn_sent2[1:3])

predictions = pd.DataFrame(amzn_sent, columns=['neg', 'neu', 'pos'])

amzn_df_nn = amzn_df.join(predictions)
amzn_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                      'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

amzn_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_nn_scored.csv', header=True)

print('the shape of Amazon_df_nn is ', amzn_df_nn.shape)

# ## KMI

KMI_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_df_json.csv'

KMI_df = pd.read_csv(KMI_file, header=0, index_col=0, parse_dates=True)

KMI_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
               'user.favourites_count', 'user.followers_count', 'KMI', 'Kinder_Morgan', 'dupe']
KMI_df.columns = KMI_columns

KMI_text = KMI_df['full_text'].astype(str)


KMI_x = KMI_text.to_list()

print(type(KMI_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(KMI_x)

KMI_train_tokens = tokenizer.texts_to_sequences(KMI_x)

KMI_pad = pad_sequences(KMI_train_tokens, maxlen=max_tokens,
                        padding=pad, truncating=pad)

KMI_sent = loaded_model.predict(x=KMI_pad)

print(KMI_sent[1:3])

predictions = pd.DataFrame(KMI_sent, columns=['neg', 'neu', 'pos'])

KMI_df_nn = KMI_df.join(predictions)
KMI_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                     'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

KMI_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_nn_scored.csv', header=True)

print('the shape of KMI _df_nn is ', KMI_df_nn.shape)

# ## Now CrowdSource

CRWD_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/CRWD_df_json.csv'

CRWD_df = pd.read_csv(CRWD_file, header=0, index_col=0, parse_dates=True)

CRWD_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
                'user.favourites_count', 'user.followers_count', 'CRWD', 'CrowdSource', 'dupe']
CRWD_df.columns = CRWD_columns

CRWD_text = CRWD_df['full_text'].astype(str)


CRWD_x = CRWD_text.to_list()

print(type(CRWD_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(CRWD_x)

CRWD_train_tokens = tokenizer.texts_to_sequences(CRWD_x)

CRWD_pad = pad_sequences(CRWD_train_tokens, maxlen=max_tokens,
                         padding=pad, truncating=pad)

CRWD_sent = loaded_model.predict(x=CRWD_pad)

print(CRWD_sent[1:3])

predictions = pd.DataFrame(CRWD_sent, columns=['neg', 'neu', 'pos'])

CRWD_df_nn = CRWD_df.join(predictions)
CRWD_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                      'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

CRWD_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/CRWD_nn_scored.csv', header=True)

print('the shape of CRWD_df_nn is ', CRWD_df_nn.shape)


# ## Now Appian

APPN_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/APPN_df_json.csv'

APPN_df = pd.read_csv(APPN_file, header=0, index_col=0, parse_dates=True)

APPN_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
                'user.favourites_count', 'user.followers_count', 'APPN', 'Appian', 'dupe']
APPN_df.columns = APPN_columns

APPN_text = APPN_df['full_text'].astype(str)


APPN_x = APPN_text.to_list()

print(type(APPN_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(APPN_x)

APPN_train_tokens = tokenizer.texts_to_sequences(APPN_x)

APPN_pad = pad_sequences(APPN_train_tokens, maxlen=max_tokens,
                         padding=pad, truncating=pad)

APPN_sent = loaded_model.predict(x=APPN_pad)

print(APPN_sent[1:3])

predictions = pd.DataFrame(APPN_sent, columns=['neg', 'neu', 'pos'])

APPN_df_nn = APPN_df.join(predictions)
APPN_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                      'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

APPN_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/APPN_nn_scored.csv', header=True)

print('the shape of APPN_df_nn is ', APPN_df_nn.shape)


# ## Now Inspire


INSP_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_df_json.csv'

INSP_df = pd.read_csv(INSP_file, header=0, index_col=0, parse_dates=True)

INSP_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
                'user.favourites_count', 'user.followers_count', 'INSP', 'Inspire_Systems', 'dupe']
INSP_df.columns = INSP_columns

INSP_text = INSP_df['full_text'].astype(str)


INSP_x = INSP_text.to_list()

print(type(INSP_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(INSP_x)

INSP_train_tokens = tokenizer.texts_to_sequences(INSP_x)

INSP_pad = pad_sequences(INSP_train_tokens, maxlen=max_tokens,
                         padding=pad, truncating=pad)

INSP_sent = loaded_model.predict(x=INSP_pad)

print(INSP_sent[1:3])

predictions = pd.DataFrame(INSP_sent, columns=['neg', 'neu', 'pos'])

INSP_df_nn = INSP_df.join(predictions)
INSP_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                      'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

INSP_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_nn_scored.csv', header=True)

print('the shape of INSP_df_nn is ', INSP_df_nn.shape)


# ## Sanofy


SNY_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/SNY_df_json.csv'

SNY_df = pd.read_csv(SNY_file, header=0, index_col=0, parse_dates=True)

SNY_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
               'user.favourites_count', 'user.followers_count', 'SNY', 'Sanofy', 'dupe']
SNY_df.columns = SNY_columns

SNY_text = SNY_df['full_text'].astype(str)


SNY_x = SNY_text.to_list()

print(type(SNY_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(SNY_x)

SNY_train_tokens = tokenizer.texts_to_sequences(SNY_x)

SNY_pad = pad_sequences(SNY_train_tokens, maxlen=max_tokens,
                        padding=pad, truncating=pad)

SNY_sent = loaded_model.predict(x=SNY_pad)

print(SNY_sent[1:3])

predictions = pd.DataFrame(SNY_sent, columns=['neg', 'neu', 'pos'])

SNY_df_nn = SNY_df.join(predictions)
SNY_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                     'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

SNY_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/SNY_nn_scored.csv', header=True)

print('the shape of SNY_df_nn is ', SNY_df_nn.shape)


# ## Everbridge


EVBG_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/EVBG_df_json.csv'

EVBG_df = pd.read_csv(EVBG_file, header=0, index_col=0, parse_dates=True)

EVBG_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
                'user.favourites_count', 'user.followers_count', 'EVBG', 'Everbridge', 'dupe']
EVBG_df.columns = EVBG_columns

EVBG_text = EVBG_df['full_text'].astype(str)


EVBG_x = EVBG_text.to_list()

print(type(EVBG_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(EVBG_x)

EVBG_train_tokens = tokenizer.texts_to_sequences(EVBG_x)

EVBG_pad = pad_sequences(EVBG_train_tokens, maxlen=max_tokens,
                         padding=pad, truncating=pad)

EVBG_sent = loaded_model.predict(x=EVBG_pad)

print(EVBG_sent[1:3])

predictions = pd.DataFrame(EVBG_sent, columns=['neg', 'neu', 'pos'])

EVBG_df_nn = EVBG_df.join(predictions)
EVBG_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                      'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

EVBG_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_FilesEVBG_nn_scored.csv', header=True)

print('the shape of EVBG_df_nn is ', EVBG_df_nn.shape)


# ## And now Exxon


XOM_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/XOM_df_json.csv'

XOM_df = pd.read_csv(XOM_file, header=0, index_col=0, parse_dates=True)

XOM_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
               'user.favourites_count', 'user.followers_count', 'XOM', 'Exxon', 'dupe']
XOM_df.columns = XOM_columns

XOM_text = XOM_df['full_text'].astype(str)


XOM_x = XOM_text.to_list()

print(type(XOM_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(XOM_x)

XOM_train_tokens = tokenizer.texts_to_sequences(XOM_x)

XOM_pad = pad_sequences(XOM_train_tokens, maxlen=max_tokens,
                        padding=pad, truncating=pad)

XOM_sent = loaded_model.predict(x=XOM_pad)

print(XOM_sent[1:3])

predictions = pd.DataFrame(XOM_sent, columns=['neg', 'neu', 'pos'])

XOM_df_nn = XOM_df.join(predictions)
XOM_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                     'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

XOM_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/XOM_nn_scored.csv', header=True)

print('the shape of XOM_df_nn is ', XOM_df_nn.shape)


# The true "class" for the first 1000 texts in the test-set are needed for comparison.

# ## And GMED

GMED_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/GMED_df_json.csv'

GMED_df = pd.read_csv(GMED_file, header=0, index_col=0, parse_dates=True)

GMED_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
                'user.favourites_count', 'user.followers_count', 'GMED', 'Global', 'dupe']
GMED_df.columns = GMED_columns

GMED_text = GMED_df['full_text'].astype(str)


GMED_x = GMED_text.to_list()

print(type(GMED_x))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(GMED_x)

GMED_train_tokens = tokenizer.texts_to_sequences(GMED_x)

GMED_pad = pad_sequences(GMED_train_tokens, maxlen=max_tokens,
                         padding=pad, truncating=pad)

GMED_sent = loaded_model.predict(x=GMED_pad)

print(GMED_sent[1:3])

predictions = pd.DataFrame(GMED_sent, columns=['neg', 'neu', 'pos'])

GMED_df_nn = GMED_df.join(predictions)
GMED_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                      'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

GMED_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/GMED_nn_scored.csv', header=True)

print('the shape of GMED_df_nn is ', GMED_df_nn.shape)
