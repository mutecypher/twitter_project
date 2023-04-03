
from tensorflow.keras.activations import relu, sigmoid, softmax, tanh, hard_sigmoid, softsign, softplus, linear
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard, LambdaCallback
import datetime
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import sklearn.model_selection as sk
from skopt.utils import use_named_args
##from skopt.plots import plot_histogram, plot_objective_2D
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_convergence
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize, forest_minimize, BayesSearchCV
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
##from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import InputLayer, Input
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


learning_df[["bad", "meh", "good"]] = 0
strt = datetime.datetime.now()
learning_df["bad"] = np.where((learning_df["NEG"] >= learning_df["NEU"]) & (
    learning_df["NEG"] >= learning_df["POS"]), 1, 0)
learning_df["meh"] = np.where((learning_df["NEU"] >= learning_df["NEG"]) & (
    learning_df["NEU"] >= learning_df["POS"]), 1, 0)
learning_df["good"] = np.where((learning_df["POS"] >= learning_df["NEG"]) & (
    learning_df["POS"] >= learning_df["NEU"]), 1, 0)
endy = datetime.datetime.now()

print("time to convert to categorical via np.where is ", endy - strt)


print("The initial text length is ", len(x))
print("and triple is ", 3*len(x))
aug_aug20 = aug_df['twenty_per_syn'].to_list()
print("aug 20 length is ", len(aug_aug20))
aug_aug30 = aug_df['thirty_per_syn'].to_list()
print("aug 30 length is ", len(aug_aug30))

##
# for i in range(len(aug_aug20)):
# x.append(str(aug_aug20[i]))
# for j in range(len(aug_aug30)):
# x.append(str(aug_aug30[j]))

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

# Used this without the categorical conversion

## y = pd.DataFrame({'NEG': y_neg, 'NEU': y_neu, 'POS': y_pos})

# Using the categorical conversion

y = pd.DataFrame(
    {'NEG': learning_df['bad'], 'NEU': learning_df['meh'], 'POS': learning_df['good']})

print("The length of x is ", len(x))
print("The length of y is ", y.shape)
x_train, x_test, y_train, y_test = sk.train_test_split(x, y, test_size=0.25
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

max_tokens = 50

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


##model = Sequential()
figure_of_merit = 2 * max_tokens  # was 100
first_layer = math.floor(max_tokens/2) + 2
second_layer = math.floor(max_tokens/2) + 1
third_layer = math.floor(max_tokens/3) + 3
fourth_layer = math.floor(max_tokens/2)
embedding_size = figure_of_merit


# set up the hyperparameters
dim_learning_rate = Categorical(categories=[1e-2, 5e-3, 1e-3, 3e-4, 8e-5, 1e-5, 4e-6],
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
    categories=[64, 128, 160, 192, 256, 384, 512],  name='batchez_yo')


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
                                                 patience=1,
                                                 restore_best_weights=True)

    history = model.fit(x=x_train_pad,
                        y=y_train,
                        epochs=10,
                        batch_size=batchez_yo,
                        validation_split=0.25,
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
    global n
    if n == 0:
        print("This is the first iteration.")
    elif n == 1:
        print("This is the second iteration.")
    else:
        print("The function has iterated ", n+1, " times.")
    n += 1
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)

        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    K.clear_session()


# accuracy, we need to negate this number so it can be minimized.
    return -accuracy


n_cally = 150
n = 0

begy = datetime.datetime.now()
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func="gp_hedge",
                            n_calls=n_cally,
                            n_random_starts=8,
                            noise="gaussian",
                            x0=default_parameters)

endy = datetime.datetime.now()
print()
print("It took ", endy-begy, " to complete ", n_cally, " iterations")
print()
print("And now the final results with the best model")
print()


plot_convergence(search_result)
plt.show()


# search_result.x

model = load_model(path_best_model)

result = model.evaluate(x=x_test_pad,
                        y=y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))


###
result = model.evaluate(x_test_pad, y_test)
print()
print("what?")
print()

# %%
print("Accuracy: {0:.2%}".format(result[1]))


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

model.save("model.h5")
# print(model_json)
print("saved model")


model_file = '/Volumes/Elements/GitHub/twitter-project/twitter_project/model.h5'

loaded_model = load_model(
    model_file,
    custom_objects=None,
    compile=True
)


now = datetime.datetime.now()
print("\n This finished at ", now)
