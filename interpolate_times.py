import pandas as pd
import numpy as np

# Assuming TS1 and TS2 are pandas DataFrame objects with 'timestamp' and 'measurement' columns

# Step 1: Find the day with the median number of measurements in TS1
ts1_grouped = TS1.groupby(TS1['timestamp'].dt.date)
median_measurements = ts1_grouped.size().median()

# Step 2: Interpolate the same number of measurements per day in TS1
ts1_interpolated = pd.DataFrame()
for day, group in ts1_grouped:
    if len(group) < median_measurements:
        interpolated_group = group.reindex(pd.date_range(start=group['timestamp'].min(), end=group['timestamp'].max(), freq='H'))
        interpolated_group['measurement'] = interpolated_group['measurement'].interpolate()
        interpolated_group = interpolated_group.resample('D').first().ffill()
        ts1_interpolated = pd.concat([ts1_interpolated, interpolated_group])
    else:
        ts1_interpolated = pd.concat([ts1_interpolated, group])

# Step 3: Create identical timestamps and measurements per day in TS2
ts2_grouped = TS2.groupby(TS2['timestamp'].dt.date)
ts2_aligned = pd.DataFrame()
for day, group in ts2_grouped:
    group['measurement'] = np.mean(group['measurement'])
    aligned_group = pd.DataFrame({'timestamp': pd.date_range(start=group['timestamp'].min(), end=group['timestamp'].max(), freq='H'),
                                  'measurement': group['measurement'].values[0]})
    ts2_aligned = pd.concat([ts2_aligned, aligned_group])

# Output the interpolated TS1 and aligned TS2
print("Interpolated TS1:")
print(ts1_interpolated)

print("Aligned TS2:")
print(ts2_aligned)
