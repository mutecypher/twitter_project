import pandas as pd
import statsmodels.api as sm

# Create a DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, None, None, 5], 'B': [3, None, None, 6, 7]},
                  index=pd.date_range(start='2022-01-01', periods=5))

# Decompose the time series into seasonal, trend, and residual components
decomp = sm.tsa.seasonal_decompose(df.ffill(), model='additive', freq=1)

# Use the trend component to impute missing values
imputed_values = decomp.trend[df.isna()].values

# Update the DataFrame with imputed values
df.loc[df.isna()] = imputed_values

print(df)

# for ARIMA models


# Create a DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, None, None, 5], 'B': [3, None, None, 6, 7]},
                  index=pd.date_range(start='2022-01-01', periods=5))

# Fit an ARIMA model to the time series
model = sm.tsa.arima.ARIMA(df.ffill(), order=(1, 1, 1))
result = model.fit()

# Use the ARIMA model to predict missing values
predicted_values = result.predict(
    start=df.index[0], end=df.index[-1], typ='levels')[df.isna()].values

# Update the DataFrame with imputed values
df.loc[df.isna()] = predicted_values

print(df)
