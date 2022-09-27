
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


file1 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/twitter_sentiment_learn.csv'
file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/cleaned_and_augmented.csv'

learning_df = pd.read_csv(file1)
learning_df = learning_df.drop_duplicates()
learning_df = learning_df.drop_duplicates(subset=['text'])

print(learning_df.shape)
print(learning_df.head())

aug_df = pd.read_csv(file2)

x = learning_df['text'].to_list()

print("The initial text length is ", len(x))
print("and triple is ", 3*len(x))
aug_aug20 = aug_df['twenty_per_syn'].to_list()
print("aug 20 length is ", len(aug_aug20))
aug_aug30 = aug_df['thirty_per_syn'].to_list()
print("aug 30 length is ", len(aug_aug30))

for i in range(len(aug_aug20)):
    x.append(str(aug_aug20[i]))
for j in range(len(aug_aug30)):
    x.append(str(aug_aug30[j]))

print("The length of x is ", len(x))
y_neg = learning_df['NEG'].to_list()

b = len(y_neg)

for i in range(b):
    y_neg.append(y_neg[i])

for i in range(b):
    y_neg.append(y_neg[i])
print("The length of y_neg is ", len(y_neg))

y_neu = learning_df['NEU'].to_list()

c = len(y_neu)

for i in range(c):
    y_neu.append(y_neu[i])

for i in range(c):
    y_neu.append(y_neu[i])

print("The length of neutral is ", len(y_neu))

y_pos = learning_df['POS'].to_list()

d = len(y_pos)

for i in range(d):
    y_pos.append(y_pos[i])

for i in range(d):
    y_pos.append(y_pos[i])
print("The length of y_pos is ", len(y_pos))

y = pd.DataFrame({'NEG': y_neg, 'NEU': y_neu, 'POS': y_pos})
x_train, x_test, y_train, y_test = sk.train_test_split(
    x, y, test_size=0.25
    # ,random_state=42
)

y_train = np.array(y_train)
y_test = np.array(y_test)

print("Train-set size: ", len(x_train))
print("Test-set size:  ", len(x_test))

data_text = x_train + x_test

print("\n", x_train[1])
print("\n", learning_df.head())
print("\n", y_train[1:5])


num_words = 30000

tokenizer = Tokenizer(num_words=num_words)


tokenizer.fit_on_texts(data_text)

x_train_tokens = tokenizer.texts_to_sequences(x_train)

x_test_tokens = tokenizer.texts_to_sequences(x_test)


num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
max_tokens = np.mean(num_tokens) + 2.5 * np.std(num_tokens)
max_tokens = math.floor(max_tokens)

np.sum(num_tokens < max_tokens) / len(num_tokens)
pad = 'pre'

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]

    # Concatenate all words.
    text = " ".join(words)

    return text


model = Sequential()
figure_of_merit = 2 * max_tokens  # was 100
first_layer = math.floor(max_tokens/2) + 2
second_layer = math.floor(max_tokens/2) + 1
third_layer = math.floor(max_tokens/3) + 3
fourth_layer = math.floor(max_tokens/2)
embedding_size = figure_of_merit
model.add(Embedding(input_dim=num_words,  # was num_words
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
model.add(GRU(units=first_layer,
              activation='relu',  # was tanh
              recurrent_activation='sigmoid',
              return_sequences=True))
model.add(GRU(units=second_layer, activation='tanh',  # was tanh
              recurrent_activation='softmax',
              return_sequences=True))
model.add(GRU(units=third_layer, activation='tanh',  # was tanh
              recurrent_activation='softmax',
              return_sequences=True))
model.add(GRU(units=fourth_layer, activation='tanh',  # was tanh
              recurrent_activation='sigmoid', return_sequences=False))
model.add(Dense(3, activation='softmax'))  # was 3
learning_rat = 3e-3
optimizer = Nadam(learning_rate=learning_rat)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# set up the hyperparameters
dim_learning_rate = Categorical(categories=[1e-2, 5e-3, 2e-3, 1e-3, 4e-4, 1e-4, 4e-5],
                                name='learning_rate')

dim_num_dense_layers = Categorical(
    categories=[1, 2, 3, 4, 5, 6], name='num_dense_layers')

dim_num_dense_nodes = Categorical(
    categories=[10, 12, 15, 19, 24, 30], name='num_dense_nodes')


dim_activation_1 = Categorical(categories=['relu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                           'softsign', 'softplus', 'linear'],
                               name='activation_1')
dim_activation_2 = Categorical(categories=['relu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                           'softsign', 'softplus', 'linear'],
                               name='activation_2')

dim_recurrent_1 = Categorical(categories=['relu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                          'softsign', 'softplus', 'linear'],
                              name='recurrent_activation_1')

dim_recurrent_2 = Categorical(categories=['relu', 'sigmoid', 'softmax', 'tanh', 'hard_sigmoid',
                                          'softsign', 'softplus', 'linear'],
                              name='recurrent_activation_2')

dim_batches_yo = size = Categorical(
    categories=[128, 160, 192, 256, 384, 512],  name='batchez_yo')


dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation_1,
              dim_activation_2,
              dim_recurrent_1,
              dim_recurrent_2,
              dim_batches_yo]


default_parameters = [1e-3, 3, 10, 'softplus',
                      'relu', 'sigmoid', 'softsign', 128]

# more setup


def log_dir_name(learning_rate,
                 num_dense_layers,
                 num_dense_nodes,
                 activation_1,
                 activation_2,
                 recurrent_activation_1,
                 recurrent_activation_2,
                 batchez_yo):
    # ,
 # optimizations_yo):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation_1,
                       activation_2,
                       recurrent_activation_1,
                       recurrent_activation_2,
                       batchez_yo)
    # ,
    # optimizations_yo)
    tf.autograph.experimental.do_not_convert(
        func=None)

    return log_dir

# Make the model


def create_model(learning_rate,
                 num_dense_layers,
                 num_dense_nodes,
                 activation_1,
                 activation_2,
                 recurrent_activation_1,
                 recurrent_activation_2,
                 batchez_yo):
    # ,
    # optimizations_yo):

    # Start construction of a Keras Sequential model.
    model = Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(Embedding(input_dim=num_words,
                        output_dim=embedding_size,
                        input_length=max_tokens,
                        name='layer_embedding'))

    # Put in the GRU flavor
    model.add(GRU(units=first_layer, activation=activation_1,
                  recurrent_activation=recurrent_activation_1, return_sequences=True))

    i = 0
    j = num_dense_layers
    while i < j:
        model.add(GRU(units=math.floor(max_tokens/(i+2)) + 2, activation=activation_2,
                      recurrent_activation=recurrent_activation_2, return_sequences=True))
        i = i + 1

    model.add(GRU(units=fourth_layer, activation=activation_1,
                  recurrent_activation=recurrent_activation_1, return_sequences=False))
    # , return_sequences = False))

    model.add(Dense(3, activation=activation_2))  # was 3

    optimizering = Adam(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizering,
                  metrics=['accuracy'])

    tf.autograph.experimental.do_not_convert(
        func=None)

    return model


path_best_model = '19_best_model.h5'


best_accuracy = 0.0

##validation_data = (data.x_val, data.y_val)


@use_named_args(dimensions=dimensions)
def fitness(learning_rate,
            num_dense_layers,
            num_dense_nodes,
            activation_1,
            activation_2,
            recurrent_activation_1,
            recurrent_activation_2,
            batchez_yo):
    # ,
    # optimizations_yo):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    tf.autograph.experimental.do_not_convert(
        func=None)

    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation_1=activation_1,
                         activation_2=activation_2,
                         recurrent_activation_1=recurrent_activation_1,
                         recurrent_activation_2=recurrent_activation_2,
                         batchez_yo=batchez_yo)
    # ,
    # optimizations_yo = optimizations_yo)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate,
                           num_dense_layers,
                           num_dense_nodes,
                           activation_1,
                           activation_2,
                           recurrent_activation_1,
                           recurrent_activation_2,
                           batchez_yo)
    # ,
    # optimizations_yo)
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)

    callbackx = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 patience=0,
                                                 restore_best_weights=True)

    history = model.fit(x=x_train_pad,
                        y=y_train,
                        epochs=8,
                        batch_size=batchez_yo,
                        validation_split=0.3,
                        callbacks=[callback_log, callbackx])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_accuracy'][-2]

    # Print the classification accuracy.
    print('+++++++++++++++++++++++++++')
    print('Accuracy: {0:.2%}'.format(accuracy))
    print('@@@@@@@@@@@@@@@@@@@@@')
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation_1:', activation_1)
    print('activation_2:', activation_2)
    print('recurrent_activation_1:', recurrent_activation_1)
    print('recurrent_activation_2:', recurrent_activation_2)
    print('batches are ', batchez_yo)
    print('********************')
    print('The model summary is ', model.summary)

    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)

        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()

    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.

    return -accuracy


search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected of improvement.
                            n_calls=11,
                            x0=default_parameters)

plot_convergence(search_result)

search_result.x

model = load_model(path_best_model)

result = model.evaluate(x=x_test_pad,
                        y=y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))


###
result = model.evaluate(x_test_pad, y_test)
print("what?")

# %%
print("Accuracy: {0:.2%}".format(result[1]))


# %% [markdown]
# ## save model

# %%
##from keras.models import models_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

model.save("model.h5")
# print(model_json)
print("saved model")

# %% [markdown]
# ## Example of Mis-Classified Text
#
# In order to show an example of mis-classified text, we first calculate the predicted sentiment for the first 1000 texts in the test-set.

# %%
# Load the saved model and see if the results differ

model_file = '/Volumes/Elements/GitHub/twitter-project/twitter_project/model.h5'

loaded_model = load_model(
    model_file,
    custom_objects=None,
    compile=True
)


# %%

##y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = loaded_model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]


# %% [markdown]
# These predicted numbers fall between 0.0 and 1.0. We use a cutoff / threshold and say that all values above 0.5 are taken to be 1.0 and all values below 0.5 are taken to be 0.0. This gives us a predicted "class" of either 0.0 or 1.0.

# %% [markdown]
# ## Test on Amazon data

# %%
amzn_file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_df_json.csv'

amzn_df = pd.read_csv(amzn_file, header=0, index_col=0, parse_dates=True)

am_columns = ['numbers', 'created_at', 'full_text', 'retweet_count', 'user_id',
              'user.favourites_count', 'user.followers_count', 'AMZN', 'Amazon', 'dupe']
amzn_df.columns = am_columns

amzn_text = amzn_df['full_text'].astype(str)


amzn_x = amzn_text.to_list()

bob = amzn_x[0:7]

print(type(bob))

num_words = 30000

tokenizer = Tokenizer(num_words=num_words)

# %%time
tokenizer.fit_on_texts(amzn_x)

amzn_train_tokens = tokenizer.texts_to_sequences(amzn_x)

amzn_pad = pad_sequences(amzn_train_tokens, maxlen=max_tokens,
                         padding=pad, truncating=pad)

amzn_sent = loaded_model.predict(x=amzn_pad)

print(amzn_sent[1:3])

predictions = pd.DataFrame(amzn_sent, columns=['neg', 'neu', 'pos'])

amzn_df_nn = amzn_df.join(predictions)
amzn_df_nn.columns = ['number', 'created_at', 'full_text', 'retweet_count', 'user_id',
                      'user.favourites_count', 'user.followers_count', 'Symbol', 'Name', 'dupe', 'neg', 'neu', 'pos']

##    print("\nThe head of the ", i, " dataframe is \n", Krusty[i].head())

amzn_df_nn.to_csv(
    '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_nn_scored.csv', header=True)

print('the shape of Amazon_df_nn is ', amzn_df_nn.shape)

# %%
cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])

# %%


# %% [markdown]
# ## KMI

# %%
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

# %%time
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

# %% [markdown]
# ## Now CrowdSource

# %%
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

# %%time
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


# %% [markdown]
# ## Now Appian

# %%
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

# %%time
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


# %% [markdown]
# ## Now Inspire

# %%
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

# %%time
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


# %% [markdown]
# ## Sanofy

# %%
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

# %%time
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


# %% [markdown]
# ## Everbridge

# %%
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

# %%time
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


# %% [markdown]
# ## And now Exxon

# %%
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

# %%time
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


# %% [markdown]
# The true "class" for the first 1000 texts in the test-set are needed for comparison.

# %% [markdown]
# ## And GMED

# %%
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

# %%time
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


# %%
cls_true = np.array(y_test[0:1000])

# %%


# %% [markdown]
# We can then get indices for all the texts that were incorrectly classified by comparing all the "classes" of these two arrays.

# %%
incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

# %% [markdown]
# Of the 1000 texts used, how many were mis-classified?

# %%
len(incorrect)

# %% [markdown]
# Let us look at the first mis-classified text. We will use its index several times.

# %%
idx = incorrect[0]
idx

# %% [markdown]
# The mis-classified text is:

# %%
text = x_test[idx]
text

# %% [markdown]
# These are the predicted and true classes for the text:

# %%
y_pred[idx]

# %%
cls_true[idx]

# %% [markdown]
# ## New Data
#
# Let us try and classify new texts that we make up. Some of these are obvious, while others use negation and sarcasm to try and confuse the model into mis-classifying the text.

# %%
text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
text9 = 'This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good'
text10 = 'One of the worst movies of all time. I cannot believe I wasted two hours of my life for this movie'
text11 = 'At no point can I say I loved this movie.'
text12 = 'No complaint about this movie could be justified, just pure perfection.'

texts = [text1, text2, text3, text4, text5, text6,
         text7, text8, text9, text10, text11, text12]

# %% [markdown]
# We first convert these texts to arrays of integer-tokens because that is needed by the model.

# %%
tokens = tokenizer.texts_to_sequences(texts)

# %% [markdown]
# To input texts with different lengths into the model, we also need to pad and truncate them.

# %%
tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
tokens_pad.shape

# %% [markdown]
# We can now use the trained model to predict the sentiment for these texts.

# %%
# Note, using the loaded prediciton model.

loaded_model.predict(tokens_pad)

# %% [markdown]
# A value close to 0.0 means a negative sentiment and a value close to 1.0 means a positive sentiment. These numbers will vary every time you train the model.

# %% [markdown]
# ## Embeddings
#
# The model cannot work on integer-tokens directly, because they are integer values that may range between 0 and the number of words in our vocabulary, e.g. 10000. So we need to convert the integer-tokens into vectors of values that are roughly between -1.0 and 1.0 which can be used as input to a neural network.
#
# This mapping from integer-tokens to real-valued vectors is also called an "embedding". It is essentially just a matrix where each row contains the vector-mapping of a single token. This means we can quickly lookup the mapping of each integer-token by simply using the token as an index into the matrix. The embeddings are learned along with the rest of the model during training.
#
# Ideally the embedding would learn a mapping where words that are similar in meaning also have similar embedding-values. Let us investigate if that has happened here.
#
# First we need to get the embedding-layer from the model:

# %%
layer_embedding = model.get_layer('layer_embedding')

# %% [markdown]
# We can then get the weights used for the mapping done by the embedding-layer.

# %%
weights_embedding = layer_embedding.get_weights()[0]

# %% [markdown]
# Note that the weights are actually just a matrix with the number of words in the vocabulary times the vector length for each embedding. That's because it is basically just a lookup-matrix.

# %%
weights_embedding.shape

# %% [markdown]
# Let us get the integer-token for the word 'good', which is just an index into the vocabulary.

# %%
token_good = tokenizer.word_index['good']
token_good

# %% [markdown]
# Let us also get the integer-token for the word 'great'.

# %%
token_great = tokenizer.word_index['great']
token_great

# %% [markdown]
# These integertokens may be far apart and will depend on the frequency of those words in the data-set.
#
# Now let us compare the vector-embeddings for the words 'good' and 'great'. Several of these values are similar, although some values are quite different. Note that these values will change every time you train the model.

# %%
weights_embedding[token_good]

# %%
weights_embedding[token_great]

# %% [markdown]
# Similarly, we can compare the embeddings for the words 'bad' and 'horrible'.

# %%


# %%


# %%


# %% [markdown]
# ### Sorted Words
#
# We can also sort all the words in the vocabulary according to their "similarity" in the embedding-space. We want to see if words that have similar embedding-vectors also have similar meanings.
#
# Similarity of embedding-vectors can be measured by different metrics, e.g. Euclidean distance or cosine distance.
#
# We have a helper-function for calculating these distances and printing the words in sorted order.

# %%
def print_sorted_words(word, metric='cosine'):
    """
    Print the words in the vocabulary sorted according to their
    embedding-distance to the given word.
    Different metrics can be used, e.g. 'cosine' or 'euclidean'.
    """

    # Get the token (i.e. integer ID) for the given word.
    token = tokenizer.word_index[word]

    # Get the embedding for the given word. Note that the
    # embedding-weight-matrix is indexed by the word-tokens
    # which are integer IDs.
    embedding = weights_embedding[token]

    # Calculate the distance between the embeddings for
    # this word and all other words in the vocabulary.
    distances = cdist(weights_embedding, [embedding],
                      metric=metric).T[0]

    # Get an index sorted according to the embedding-distances.
    # These are the tokens (integer IDs) for words in the vocabulary.
    sorted_index = np.argsort(distances)

    # Sort the embedding-distances.
    sorted_distances = distances[sorted_index]

    # Sort all the words in the vocabulary according to their
    # embedding-distance. This is a bit excessive because we
    # will only print the top and bottom words.
    sorted_words = [inverse_map[token] for token in sorted_index
                    if token != 0]

    # Helper-function for printing words and embedding-distances.
    def _print_words(words, distances):
        for word, distance in zip(words, distances):
            print("{0:.3f} - {1}".format(distance, word))

    # Number of words to print from the top and bottom of the list.
    k = 10

    print("Distance from '{0}':".format(word))

    # Print the words with smallest embedding-distance.
    _print_words(sorted_words[0:k], sorted_distances[0:k])

    print("...")

    # Print the words with highest embedding-distance.
    _print_words(sorted_words[-k:], sorted_distances[-k:])

# %% [markdown]
# We can then print the words that are near and far from the word 'great' in terms of their vector-embeddings. Note that these may change each time you train the model.

# %%
##print_sorted_words('great', metric='cosine')

# %% [markdown]
# Similarly, we can print the words that are near and far from the word 'worst' in terms of their vector-embeddings.

# %%
##print_sorted_words('worst', metric='cosine')


# %%
now = datetime.datetime.now()
print("\n This finished at ", now)


# %% [markdown]
# ## Conclusion
#
# This tutorial showed the basic methods for doing Natural Language Processing (NLP) using a Recurrent Neural Network with integer-tokens and an embedding layer. This was used to do sentiment analysis of movie reviews from IMDB. It works reasonably well if the hyper-parameters are chosen properly. But it is important to understand that this is not human-like comprehension of text. The system does not have any real understanding of the text. It is just a clever way of doing pattern-recognition.

# %% [markdown]
# ## Exercises
#
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
#
# You may want to backup this Notebook before making any changes.
#
# * Run more training-epochs. Does it improve performance?
# * If your model overfits the training-data, try using dropout-layers and dropout inside the GRU.
# * Increase or decrease the number of words in the vocabulary. This is done when the `Tokenizer` is initialized. Does it affect performance?
# * Increase the size of the embedding-vectors to e.g. 200. Does it affect performance?
# * Try varying all the different hyper-parameters for the Recurrent Neural Network.
# * Use Bayesian Optimization from Tutorial #19 to find the best choice of hyper-parameters.
# * Use 'post' for padding and truncating in `pad_sequences()`. Does it affect the performance?
# * Use individual characters instead of tokenized words as the vocabulary. You can then use one-hot encoded vectors for each character instead of using the embedding-layer.
# * Use `model.fit_generator()` instead of `model.fit()` and make your own data-generator, which creates a batch of data using a random subset of `x_train_tokens`. The sequences must be padded so they all match the length of the longest sequence.
# * Explain to a friend how the program works.

# %% [markdown]
# ## License (MIT)
#
# Copyright (c) 2018 by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# %%


# %%
