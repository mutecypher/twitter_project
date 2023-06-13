import pandas as pd

file1 ="/Volumes/Elements/GitHub/twitter-project/Data_Files/amzn_stock_df.csv"
file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_agged_df.csv'

TS1 = pd.read_csv(file1, header='infer')
TS2 = pd.read_csv(file2, header='infer')

colz = ['','meaningless', 'high', 'low', 'open','close','volume',
        'adjclose','date','pct','rolling3',	'var3', 'covar3', 'beta3',
        'var10', 'covar10',	'beta10',	
        'beta3_clf', 'beta3_ndaq','beta3_djia','rolling1',
        'beta10_clf', 'beta10_djia', 'beta10_ndaq',	
        'rolling20', 'var20','covar20',	'beta20_clf','beta20_djia',	'beta20_ndaq',
        'rolling30', 'var30','covar30',	'beta30_clf', 'beta30_djia', 'beta30_ndaq',
        'rolling50', 'varfifty', 'covarfifty', 'betafifty',
        'varqrtr',	'covarqrtr', 'betaqtr_clf',	'betaqtr_ndaq',	'betaqtr_djia',
        'rolling250', 'varyear', 'covaryear',	
        'betayr_clf', 'betayr_ndaq', 'betayr_djia']

##colz = colz.tolist()

not_dt_colz = [	'high',	'low',	'open',	'close', 'volume',
        'adjclose',	'pct',	'rolling3',	'var3', 'covar3', 'beta3',
        'beta3_clf', 'beta3_ndaq',	'beta3_djia',	'rolling1',
        'beta10_clf', 'beta10_djia', 'beta10_ndaq',	
        'rolling20', 'var20','covar20',	'beta20_clf','beta20_djia',	'beta20_ndaq',
        'rolling30', 'var30','covar30',	'beta30_clf', 'beta30_djia', 'beta30_ndaq',
        'rolling50', 'varfifty', 'covarfifty', 'betafifty',
        'varqrtr',	'covarqrtr', 'betaqtr_clf',	'betaqtr_ndaq',	'betaqtr_djia',
        'rolling250', 'varyear', 'covaryear',	
        'betayr_clf', 'betayr_ndaq', 'betayr_djia']

##not_dt_colz = not_dt_colz.tolist()

TS1.columns = colz

TS1['date'] = pd.to_datetime(TS1['date'], utc=True, errors='coerce').dt.tz_convert(
        'US/Eastern').dt.tz_localize(None)

TS1.set_index('date', inplace=True)

print("the non-weekend shape is ", TS1.shape)
print()
print("the tail of TS1 is ", TS1.tail())
# Assuming TS1 and TS2 are pandas Series or DataFrames with a datetime index

# Step 1: Identify missing weekend timestamps in TS1
weekdays_TS1 = TS1.index.weekday
print("the weekdays_TS1 shape is ", weekdays_TS1.shape)
print()
print("the weekdays_TS1 is ", weekdays_TS1)
print()
print("the shape of TS2 is ", TS2.shape)
weekend_timestamps_TS2 = TS2[~weekdays_TS1.isin([5, 6])].index

# Step 2: Create a new DataFrame to store interpolated data
interpolated_TS1 = pd.DataFrame(index=TS2.index, columns=[not_dt_colz])

# Step 3: Interpolate missing data points in TS1
interpolated_TS1.loc[TS1.index] = TS1
interpolated_TS1[not_dt_colz] = interpolated_TS1[not_dt_colz].interpolate(method='linear')

print("the interpolated_TS1 shape is ", interpolated_TS1.shape)

TS1.to_csv("/Volumes/Elements/GitHub/twitter-project/Data_Files/interpolated_amzn_stock.csv")


# Step 4: Merge interpolated TS1 with TS2
##merged_TS = pd.concat([interpolated_TS1, TS2], axis=1)

# The merged_TS DataFrame will contain the interpolated TS1 data for weekends and the original TS1 and TS2 data for weekdays and non-intersecting timestamps.

# Optionally, you can sort the merged_TS DataFrame by timestamp
##merged_TS = merged_TS.sort_index()
