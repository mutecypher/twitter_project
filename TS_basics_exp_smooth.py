import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming TS3 is a pandas DataFrame object with 'timestamp' and 'measurement' columns

# Step 1: Check for seasonality
res = sm.tsa.seasonal_decompose(TS3['measurement'], period=24)  # Assuming seasonality is daily with a period of 24 hours
seasonality = res.seasonal
is_seasonal = np.any(seasonality > 0)  # Check if there is any seasonality

if is_seasonal:
    print("TS3 exhibits seasonality.")
else:
    print("TS3 does not exhibit seasonality.")

# Step 2: Check for stationarity
adf_result = sm.tsa.stattools.adfuller(TS3['measurement'])
is_stationary = adf_result[1] < 0.05  # Check if the p-value is less than 0.05

if is_stationary:
    print("TS3 is stationary.")
else:
    print("TS3 is not stationary.")

# Step 3: Check for autocorrelation
acf_result = sm.tsa.stattools.acf(TS3['measurement'], nlags=24)  # Assuming a lag of 24 hours
is_autocorrelated = np.any(np.abs(acf_result[1:]) > 0.2)  # Check if any autocorrelation exceeds the threshold of 0.2

if is_autocorrelated:
    print("TS3 exhibits autocorrelation.")
else:
    print("TS3 does not exhibit autocorrelation.")

# Step 4: Split TS3 into training, validation, and test sets
train, test = train_test_split(TS3, test_size=0.2, shuffle=False)
train, val = train_test_split(train, test_size=0.2, shuffle=False)

# Step 5: Build and evaluate the exponential smoothing model
model = ExponentialSmoothing(train['measurement'], trend='add', seasonal='add', seasonal_periods=24).fit()

# Predict on the validation set
val_predictions = model.predict(start=val.index[0], end=val.index[-1])

# Calculate root mean squared error (RMSE) on the validation set
val_rmse = np.sqrt(mean_squared_error(val['measurement'], val_predictions))
print("Validation RMSE:", val_rmse)

# Predict on the test set
test_predictions = model.predict(start=test.index[0], end=test.index[-1])

# Calculate root mean squared error (RMSE) on the test set
test_rmse = np.sqrt(mean_squared_error(test['measurement'], test_predictions))
print("Test RMSE:", test_rmse)
