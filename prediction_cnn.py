from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# load stock prices and twitter data into pandas dataframes
stock_prices = pd.read_csv("stock_prices.csv")
twitter_data = pd.read_csv("twitter_data.csv")

# convert date strings to datetime objects in both dataframes
stock_prices['date'] = pd.to_datetime(stock_prices['date'])
twitter_data['date'] = pd.to_datetime(twitter_data['date'])

# merge the two dataframes on date
merged_data = pd.merge(stock_prices, twitter_data, on='date')

# use the nltk library to calculate sentiment scores for each tweet
sia = SentimentIntensityAnalyzer()
merged_data['sentiment_score'] = merged_data['text'].apply(
    lambda x: sia.polarity_scores(x)['compound'])

# calculate the average sentiment score for each day
merged_data['avg_sentiment_score'] = merged_data.groupby(by='date')[
                                                         'sentiment_score'].mean()

# plot the average sentiment score and stock price over time
plt.plot(merged_data['date'],
         merged_data['avg_sentiment_score'], label='sentiment score')
plt.plot(merged_data['date'], merged_data['close'], label='stock price')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sentiment Score / Stock Price')
plt.title('Sentiment Analysis of Twitter Data and Stock Prices')
plt.show()


# load stock prices and twitter data into pandas dataframes
stock_prices = pd.read_csv("stock_prices.csv")
twitter_data = pd.read_csv("twitter_data.csv")

# convert date strings to datetime objects in both dataframes
stock_prices['date'] = pd.to_datetime(stock_prices['date'])
twitter_data['date'] = pd.to_datetime(twitter_data['date'])

# merge the two dataframes on date
merged_data = pd.merge(stock_prices, twitter_data, on='date')

# use the nltk library to calculate sentiment scores for each tweet
sia = SentimentIntensityAnalyzer()
merged_data['sentiment_score'] = merged_data['text'].apply(
    lambda x: sia.polarity_scores(x)['compound'])

# calculate the average sentiment score for each day
merged_data['avg_sentiment_score'] = merged_data.groupby(by='date')[
                                                         'sentiment_score'].mean()

# drop any rows with missing values
merged_data = merged_data.dropna()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    merged_data[['avg_sentiment_score']], merged_data[['close']], test_size=0.2)

# fit a linear regression model on the training data
reg = LinearRegression().fit(X_train, y_train)

# make predictions on the testing data
y_pred = reg.predict(X_test)

# evaluate the model's performance using mean squared error and R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2:", r2)

# plot the predicted stock price and actual stock price over time
plt.plot(X_test.index, y_pred, label='predicted stock price')
plt.plot(X_test.index, y_test, label='actual stock price')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Predicted Stock Price Based on Twitter Sentiments')
plt.show()


# load stock prices and twitter data into pandas dataframes
stock_prices = pd.read_csv("stock_prices.csv")
twitter_data = pd.read_csv("twitter_data.csv")

# convert date strings to datetime objects in both dataframes
stock_prices['date'] = pd.to_datetime(stock_prices['date'])
twitter_data['date'] = pd.to_datetime(twitter_data['date'])

# merge the two dataframes on date
merged_data = pd.merge(stock_prices, twitter_data, on='date')

# use the nltk library to calculate sentiment scores for each tweet
sia = SentimentIntensityAnalyzer()
merged_data['sentiment_score'] = merged_data['text'].apply(
    lambda x: sia.polarity_scores(x)['compound'])

# calculate the average sentiment score for each day
merged_data['avg_sentiment_score'] = merged_data.groupby(by='date')[
                                                         'sentiment_score'].mean()

# drop any rows with missing values
merged_data = merged_data.dropna()

# split the data into training and testing sets
X = merged_data[['avg_sentiment_score']]
y = merged_data[['close']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reshape the data for use in a CNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# define the CNN model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2,
          activation='relu', input_shape=(1, 1)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=32


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from nltk.sentiment import SentimentIntensityAnalyzer

# load stock prices and twitter data into pandas dataframes
stock_prices=pd.read_csv("stock_prices.csv")
twitter_data=pd.read_csv("twitter_data.csv")

# convert date strings to datetime objects in both dataframes
stock_prices['date']=pd.to_datetime(stock_prices['date'])
twitter_data['date']=pd.to_datetime(twitter_data['date'])

# merge the two dataframes on date
merged_data=pd.merge(stock_prices, twitter_data, on='date')

# use the nltk library to calculate sentiment scores for each tweet
sia=SentimentIntensityAnalyzer()
merged_data['sentiment_score']=merged_data['text'].apply(
    lambda x: sia.polarity_scores(x)['compound'])

# calculate the average sentiment score for each day
merged_data['avg_sentiment_score']=merged_data.groupby(by='date')[
                                                       'sentiment_score'].mean()

# drop any rows with missing values
merged_data=merged_data.dropna()

# plot the time series data to get an idea of its trend and seasonality
plt.plot(merged_data['date'], merged_data['avg_sentiment_score'])
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.show()

# check if the sentiment scores are stationary
result=adfuller(merged_data['avg_sentiment_score'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# if the sentiment scores are not stationary, difference them until they are
differenced=merged_data['avg_sentiment_score']
while adfuller(differenced)[1] >= 0.05:
    differenced=differenced.diff().dropna()

# fit the ARIMA model
model=ARIMA(differenced, order=(1, 1, 1))
model_fit=model.fit()

# make predictions
predictions=model_fit.predict(start=len(differenced), end=len(differenced)+7)

# reverse the differencing to get the predicted sentiment scores
predictions=np.cumsum(predictions)
predictions=pd.Series(predictions, name='Predicted Sentiment Scores')

# plot the predicted sentiment scores
plt.plot(predictions.index, predictions.values
