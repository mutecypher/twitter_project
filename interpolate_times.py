import pandas as pd
import numpy as np
import datetime

# Assuming TS1 and TS2 are pandas DataFrame objects with 'timestamp' and 'measurement' columns

# Step 1: Find the day with the median number of measurements in TS1
file_1 = "/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_3_agged_df.csv"
file_2 = "/Volumes/Elements/GitHub/twitter-project/Data_Files/amzn_stock_df.csv"
TS1 = pd.read_csv(file_1, parse_dates=['created_at'])
TS1 = TS1.rename(columns = {'created_at': 'timestamp'})
TS2 = pd.read_csv(file_2, parse_dates=['date'])
TS2['date'] = pd.to_datetime(TS2['date'], utc=True, errors='coerce').dt.tz_convert(
        'US/Eastern').dt.tz_localize(None)
TS2 = TS2.rename(columns = {'date': 'timestamp'})

print("The first one is the transformer sentiment data")
print("the second data is the stock data")


ts1_grouped = TS1.groupby(TS1['timestamp'].dt.date)
median_measurements = ts1_grouped.size().median()

# Step 2: Interpolate the same number of measurements per day in TS1
ts1_interpolated = pd.DataFrame()
ts1_cols = ['neg', 'neu','pos']
print()
print("TS1 columns are /n", ts1_grouped)
print()
for day, group in ts1_grouped:
    if len(group) < median_measurements:
        interpolated_group = group.reindex(pd.date_range(start=group['timestamp'].min(), end=group['timestamp'].max(), freq='H'))
        for coz in ts1_cols:
            interpolated_group[coz] = interpolated_group[coz].interpolate()
            interpolated_group = interpolated_group.resample('D').first().ffill()
        ts1_interpolated = pd.concat([ts1_interpolated, interpolated_group])
    else:
        ts1_interpolated = pd.concat([ts1_interpolated, group])

# Step 3: Create identical timestamps and measurements per day in TS2
TS2 = TS2[['timestamp', 'close']]
ts2_grouped = TS2.groupby(TS2['timestamp'].dt.date)
##col_names = TS2.columns.tolist()
##col_names = col_names[1:]
col_names = ['close']
print()
print("column names are ", col_names)
print()
ts2_aligned = pd.DataFrame()
for day, group in ts2_grouped:
    group[col_names] = np.mean(group[col_names])
    aligned_group = pd.DataFrame({'timestamp': pd.date_range(start=group['timestamp'].min(), end=group['timestamp'].max(), freq='H'),
                                  col_names: group[col_names].values[0]})
    ts2_aligned = pd.concat([ts2_aligned, aligned_group])

# Output the interpolated TS1 and aligned TS2
print("Interpolated TS1 shape:")
print(ts1_interpolated.shape)
print()

print("Aligned TS2:")
print(ts2_aligned.shape)
