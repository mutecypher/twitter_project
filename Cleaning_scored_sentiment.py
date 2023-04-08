import pandas as pd
import datetime as dt
import pyzt

file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_nn_scored.csv'

df = pd.read_csv(file2, header='infer', index_col=0)

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Remove rows with null values


# Strip the datetime column to just month/day/year


print("the shape of the original dataframe is ", df.shape)
print()
print()
df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce').dt.tz_convert(
    'US/Eastern').dt.tz_localize(None)
print("Got the time zones right for original dataframe")
print()
print()
df_cleaned = df[pd.notnull(df['created_at'])]

print("Got the time zones right for cleaned dataframe")
print()
print()
df_cleaned['created_at'] = df_cleaned['created_at'].dt.strftime('%m/%d/%Y')


print("the shape of the original dataframe is ", df.shape)
print("the shape of the cleaned dataframe is ", df_cleaned.shape)
print("the data types are ", df_cleaned.dtypes)

# Example dataframe
