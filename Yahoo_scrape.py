
import pandas as pd
from yahoofinancials import YahooFinancials
import datetime
import statistics as st
import numpy as np

whats_today = datetime.datetime.now().date()


# June 2023 1132
minus_fifteen_years = whats_today - datetime.timedelta(days=1163 + 11)
minus_fifteen_years = whats_today - datetime.timedelta(days=500)

whats_today = whats_today.strftime('%Y-%m-%d')
minus_fifteen_years = minus_fifteen_years.strftime('%Y-%m-%d')
print(whats_today)
print(minus_fifteen_years)

print("earliest date should be 3-24-2020")

marketz = ['DJIA', 'NDAQ', '^GSPC']
namez = ['djia_df', 'nasdaq_df', 'spy_df']


i = 0
for market in marketz:
    yahoo_financials = YahooFinancials(market)

    data = yahoo_financials.get_historical_price_data(start_date="2021-11-01",
                                                      end_date="2023-06-11", time_interval='daily')
    print("got to here")
    namez[i] = pd.DataFrame(data[market]['prices'])
    namez[i] = namez[i].drop('date', axis=1)
    print(namez[i].shape)

    barky = namez[i]['adjclose']
    namez[i]['pct'] = barky.pct_change(1)
    roll_3_am = namez[i]['pct'].rolling(3, min_periods=1).mean()
    namez[i]['rolling3'] = roll_3_am

    # three day rolling averages
    namez[i]['var3'] = 0
    namez[i]['covar3'] = 0
    namez[i]['beta3'] = 0
    namez[i]['var10'] = 0
    namez[i]['covar10'] = 0
    namez[i]['beta10'] = 0
    for j in range(2, namez[i].shape[0]):
        namez[i].loc[j, 'var3'] = namez[i].loc[j-2:j, 'pct'].var()
        namez[i].loc[j, 'covar3'] = np.cov(
            namez[i].loc[j-2:j, 'pct'], namez[i].loc[j-2:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar3'] != 0:
            namez[i].loc[j, 'beta3'] = namez[i].loc[j,
                                                    'covar3']/namez[i].loc[j, 'var3']
        else:
            namez[i].loc[j, 'beta3'] = 0
        j += 1.  # it was added here
    # Now for the 10 day rolling averages
    roll_ten_am = namez[i]['pct'].rolling(10, min_periods=1).mean()
    namez[i]['rolling1'] = roll_ten_am

    for j in range(9, namez[i].shape[0]):
        namez[i].loc[j, 'var10'] = namez[i].loc[j-9:j, 'pct'].var()
        namez[i].loc[j, 'covar10'] = np.cov(
            namez[i].loc[j-9:j, 'pct'], namez[i].loc[j-9:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar10'] != 0:
            namez[i].loc[j, 'beta10'] = namez[i].loc[j,
                                                     'covar10']/namez[i].loc[j, 'var10']
        else:
            namez[i].loc[j, 'beta10'] = 0

        j += 1

    for j in range(9, namez[i].shape[0]):
        namez[i].loc[j, 'var10'] = namez[i].loc[j-9:j, 'pct'].var()
        namez[i].loc[j, 'covar10'] = np.cov(
            namez[i].loc[j-9:j, 'pct'], namez[i].loc[j-9:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar10'] != 0:
            namez[i].loc[j, 'beta10'] = namez[i].loc[j,
                                                     'covar10']/namez[i].loc[j, 'var10']
        else:
            namez[i].loc[j, 'beta10'] = 0

        j += 1
  # 20 day averages and things

    roll_twty_am = namez[i]['pct'].rolling(20, min_periods=1).mean()
    namez[i]['rolling20'] = roll_twty_am

    namez[i]['var20'] = 0
    namez[i]['covar20'] = 0
    namez[i]['beta20'] = 0
    for j in range(19, namez[i].shape[0]):
        namez[i].loc[j, 'var20'] = namez[i].loc[j-19:j, 'pct'].var()
        namez[i].loc[j, 'covar20'] = np.cov(
            namez[i].loc[j-19:j, 'pct'], namez[i].loc[j-19:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar20'] != 0:
            namez[i].loc[j, 'beta20'] = namez[i].loc[j,
                                                     'covar20']/namez[i].loc[j, 'var20']
        else:
            namez[i].loc[j, 'beta20'] = 0
        j += 1

# Thirty Days

    roll_thrt_am = namez[i]['pct'].rolling(30, min_periods=1).mean()
    namez[i]['rolling30'] = roll_thrt_am

    namez[i]['var30'] = 0
    namez[i]['covar30'] = 0
    namez[i]['beta30'] = 0
    for j in range(29, namez[i].shape[0]):
        namez[i].loc[j, 'var30'] = namez[i].loc[j-29:j, 'pct'].var()
        namez[i].loc[j, 'covar30'] = np.cov(
            namez[i].loc[j-29:j, 'pct'], namez[i].loc[j-29:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar30'] != 0:
            namez[i].loc[j, 'beta30'] = namez[i].loc[j,
                                                     'covar30']/namez[i].loc[j, 'var30']
        else:
            namez[i].loc[j, 'beta50'] = 0
        j += 1

# Rolling 50 day
    roll_ffty_am = namez[i]['pct'].rolling(50, min_periods=1).mean()
    namez[i]['rolling50'] = roll_ffty_am

    namez[i]['varfifty'] = 0
    namez[i]['covarfifty'] = 0
    namez[i]['betafifty'] = 0
    for j in range(64, namez[i].shape[0]):
        namez[i].loc[j, 'varfifty'] = namez[i].loc[j-64:j, 'pct'].var()
        namez[i].loc[j, 'covarfifty'] = np.cov(
            namez[i].loc[j-64:j, 'pct'], namez[i].loc[j-64:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covarfifty'] != 0:
            namez[i].loc[j, 'betafifty'] = namez[i].loc[j,
                                                        'covarfifty']/namez[i].loc[j, 'varfifty']
        else:
            namez[i].loc[j, 'betafifty'] = 0
        j += 1


# 250 calendar days - 178 market days

    roll_250_am = namez[i]['pct'].rolling(250, min_periods=1).mean()
    namez[i]['rolling250'] = roll_250_am

    namez[i]['var250'] = 0
    namez[i]['covar250'] = 0
    namez[i]['beta250'] = 0
    for j in range(178, namez[i].shape[0]):
        namez[i].loc[j, 'var250'] = namez[i].loc[j-178:j, 'pct'].var()
        namez[i].loc[j, 'covar250'] = np.cov(
            namez[i].loc[j-178:j, 'pct'], namez[i].loc[j-178:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar250'] != 0:
            namez[i].loc[j, 'beta250'] = namez[i].loc[j,
                                                      'covar250']/namez[i].loc[j, 'var250']
        else:
            namez[i].loc[j, 'beta250'] = 0

        j += 1

    namez[i] = namez[i].dropna(axis=0)
    namez[i].reset_index(drop=True, inplace=True)
# print(namez[i].head())
# print(namez[i].shape)
    i = i+1

 # Rolling 250 day


now = datetime.datetime.now()
print("\nthis finished at ", now)


sp500_df = namez[2]
ndaq_df = namez[1]
djia_df = namez[0]

sp500_df.to_csv("spy500_df.csv", header=True)
print("the head of sp500_df is \n", sp500_df.head())
ndaq_df.to_csv("ndaq_df.csv", header=True)
djia_df.to_csv("djia__df.csv", header=True)


marketz = ['DJIA', 'NDAQ', '^GSPC']

stawkz = ['AMZN',
          'TSLA', 'AAPL', 'EVBG', 'KMI', 'SHOP', 'GMED', 'SNY', 'MCK',
          'ET', 'INSP', 'CRWD', 'CL=F', 'UVXY', 'GLD', 'GBTC']
namez = ['AMZN_df',
         'TSLA_df', 'AAPL_df', 'EVBG_df', 'KMI_df', 'SHOP_df', 'GMED_df', 'SNY_df',
         # namez = ['AMZN_df','TSLA_df','AAPL_df','EVBG_df','KMI_df','SHOP_df','GMED_df','SNY_df',
         # 'MCK_df','ET_df','INSP_df','CL=F_df']
         'MCK_df', 'ET_df', 'INSP_df', 'CRWD_df', 'CL=F_df', 'UVXY_df', 'GLD_df', 'GBTC_df']

i = 0
for stawk in stawkz:
    yahoo_financials = YahooFinancials(stawk)

    print("/n The stawk I'm doing is ", stawk)
    print()
    print("the start date is ", minus_fifteen_years)
    print()
    print("the end date is ", whats_today)
    print()
    data = yahoo_financials.get_historical_price_data(start_date="2021-11-01",
                                                      end_date="2023-06-10",
                                                      time_interval="daily")

    namez[i] = pd.DataFrame(data[stawk]['prices'])
    ##namez[i] = namez[i].drop('date', axis=1)

# print("the head of sp500 is \n", sp500_df.head())

    barky = namez[i]['adjclose']
    namez[i]['pct'] = barky.pct_change(1)
    roll_3_am = namez[i]['pct'].rolling(3, min_periods=1).mean()
    namez[i]['rolling3'] = roll_3_am
    namez[i] = namez[i].dropna(axis=0)
    namez[i].reset_index(drop=True, inplace=True)

    # three day rolling averages

    namez[i]['var3'] = 0
    namez[i]['covar3'] = 0
    namez[i]['beta3'] = 0
    namez[i]['var10'] = 0
    namez[i]['covar10'] = 0
    namez[i]['beta10'] = 0

    k = min(namez[i].shape[0], ndaq_df.shape[0],
            sp500_df.shape[0], djia_df.shape[0])

    for j in range(2, k):
        namez[i].loc[j, 'var3'] = namez[i].loc[j-2:j, 'pct'].var()
        namez[i].loc[j, 'covar3'] = np.cov(
            sp500_df.loc[j-2:j, 'pct'], namez[i].loc[j-2:j, 'pct'])[0][1]

        if namez[i].loc[j, 'covar3'] != 0:
            namez[i].loc[j, 'beta3_clf'] = namez[i].loc[j,
                                                        'covar3']/namez[i].loc[j, 'var3']
        else:
            namez[i].loc[j, 'beta3_clf'] = 0
            namez[i].loc[j, 'covar3'] = np.cov(
                ndaq_df.loc[j-2:j, 'pct'], namez[i].loc[j-2:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar3'] != 0:
            namez[i].loc[j, 'beta3_ndaq'] = namez[i].loc[j,
                                                         'covar3']/namez[i].loc[j, 'var3']
        else:
            namez[i].loc[j, 'beta3_ndaq'] = 0

        namez[i].loc[j, 'covar3'] = np.cov(
            djia_df.loc[j-2:j, 'pct'], namez[i].loc[j-2:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar3'] != 0:
            namez[i].loc[j, 'beta3_djia'] = namez[i].loc[j,
                                                         'covar3']/namez[i].loc[j, 'var3']
        else:
            namez[i].loc[j, 'beta3_djia'] = 0

        j += 1

    # Now for the 10 day rolling averages
    namez[i]['var10'] = 0
    namez[i]['covar10'] = 0

    roll_ten_am = namez[i]['pct'].rolling(10, min_periods=1).mean()
    namez[i]['rolling1'] = roll_ten_am

    namez[i]['beta10_clf'] = 0
    namez[i]['beta10_djia'] = 0
    namez[i]['beta10_ndaq'] = 0
    for j in range(9, k):
        namez[i].loc[j, 'var10'] = namez[i].loc[j-9:j, 'pct'].var()
        namez[i].loc[j, 'covar10'] = np.cov(
            sp500_df.loc[j-9:j, 'pct'], namez[i].loc[j-9:j, 'pct'])[0][1]

        if namez[i].loc[j, 'covar10'] != 0:
            namez[i].loc[j, 'beta10_clf'] = namez[i].loc[j,
                                                         'covar10']/namez[i].loc[j, 'var10']
        else:
            namez[i].loc[j, 'beta10_clf'] = 0
        namez[i].loc[j, 'covar10'] = np.cov(
            ndaq_df.loc[j-9:j, 'pct'], namez[i].loc[j-9:j, 'pct'])[0][1]
        if namez[i].loc[i, 'covar10'] != 0:
            namez[i].loc[j, 'beta10_ndaq'] = namez[i].loc[j,
                                                          'covar10']/namez[i].loc[j, 'var10']
        else:
            namez[i].loc[j, 'beta3_ndaq'] = 0

        namez[i].loc[j, 'covar10'] = np.cov(
            djia_df.loc[j-9:j, 'pct'], namez[i].loc[j-9:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar10'] != 0:
            namez[i].loc[j, 'beta10_djia'] = namez[i].loc[j,
                                                          'covar10']/namez[i].loc[j, 'var10']
        else:
            namez[i].loc[j, 'beta10_djia'] = 0

        j += 1
  # 20 day averages and things

    roll_twty_am = namez[i]['pct'].rolling(20, min_periods=1).mean()
    namez[i]['rolling20'] = roll_twty_am

    namez[i]['var20'] = 0
    namez[i]['covar20'] = 0
    namez[i]['beta20_clf'] = 0
    namez[i]['beta20_djia'] = 0
    namez[i]['beta20_ndaq'] = 0

    for j in range(19, k):
        namez[i].loc[j, 'var20'] = namez[i].loc[j-19:j, 'pct'].var()
        namez[i].loc[j, 'covar20'] = np.cov(
            sp500_df.loc[j-19:j, 'pct'], namez[i].loc[j-19:j, 'pct'])[0][1]

        if namez[i].loc[j, 'covar20'] != 0:
            namez[i].loc[j, 'beta20_clf'] = namez[i].loc[j,
                                                         'covar20']/namez[i].loc[j, 'var20']
        else:
            namez[i].loc[j, 'beta20_clf'] = 0
        namez[i].loc[j, 'covar20'] = np.cov(
            ndaq_df.loc[j-19:j, 'pct'], namez[i].loc[j-19:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar20'] != 0:
            namez[i].loc[j, 'beta20_ndaq'] = namez[i].loc[j,
                                                          'covar20']/namez[i].loc[j, 'var20']
        else:
            namez[i].loc[j, 'beta20_ndaq'] = 0

        namez[i].loc[j, 'covar20'] = np.cov(
            djia_df.loc[j-19:j, 'pct'], namez[i].loc[j-19:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar20'] != 0:
            namez[i].loc[j, 'beta20_djia'] = namez[i].loc[i,
                                                          'covar20']/namez[i].loc[j, 'var20']
        else:
            namez[i].loc[j, 'beta20_djia'] = 0

        j += 1


# Thirty Days

    roll_thrt_am = namez[i]['pct'].rolling(30, min_periods=1).mean()
    namez[i]['rolling30'] = roll_thrt_am

    namez[i]['var30'] = 0
    namez[i]['covar30'] = 0
    namez[i]['beta30_clf'] = 0
    namez[i]['beta30_djia'] = 0
    namez[i]['beta30_ndaq'] = 0

    for j in range(29, k):
        namez[i].loc[j, 'var30'] = namez[i].loc[j-29:j, 'pct'].var()
        namez[i].loc[j, 'covar30'] = np.cov(
            sp500_df.loc[j-29:j, 'pct'], namez[i].loc[j-29:j, 'pct'])[0][1]

        if namez[i].loc[j, 'covar30'] != 0:
            namez[i].loc[j, 'beta30_clf'] = namez[i].loc[j,
                                                         'covar30']/namez[i].loc[j, 'var30']
        else:
            namez[i].loc[j, 'beta30_clf'] = 0
        namez[i].loc[j, 'covar30'] = np.cov(
            ndaq_df.loc[j-29:j, 'pct'], namez[i].loc[j-29:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar30'] != 0:
            namez[i].loc[j, 'beta30_ndaq'] = namez[i].loc[j,
                                                          'covar30']/namez[i].loc[j, 'var30']
        else:
            namez[i].loc[j, 'beta30_ndaq'] = 0

        namez[i].loc[j, 'covar30'] = np.cov(
            djia_df.loc[j-29:j, 'pct'], namez[i].loc[j-29:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covar30'] != 0:
            namez[i].loc[j, 'beta30_djia'] = namez[i].loc[j,
                                                          'covar30']/namez[i].loc[j, 'var30']
        else:
            namez[i].loc[j, 'beta30_djia'] = 0
        j += 1


# Rolling 50 day
    roll_ffty_am = namez[i]['pct'].rolling(50, min_periods=1).mean()
    namez[i]['rolling50'] = roll_ffty_am

    namez[i]['varfifty'] = 0
    namez[i]['covarfifty'] = 0
    namez[i]['betafifty'] = 0
    for j in range(64, k):
        namez[i].loc[j, 'varqrtr'] = namez[i].loc[j-64:j, 'pct'].var()
        namez[i].loc[j, 'covarqrtr'] = np.cov(
            sp500_df.loc[j-64:j, 'pct'], namez[i].loc[j-64:j, 'pct'])[0][1]

        if namez[i].loc[j, 'covarqrtr'] != 0:
            namez[i].loc[j, 'betaqtr_clf'] = namez[i].loc[j,
                                                          'covarqrtr']/namez[i].loc[j, 'varqrtr']
        else:
            namez[i].loc[j, 'beta30_clf'] = 0
        namez[i].loc[j, 'covarqrtr'] = np.cov(
            ndaq_df.loc[j-64:j, 'pct'], namez[i].loc[j-64:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covarqrtr'] != 0:
            namez[i].loc[j, 'betaqtr_ndaq'] = namez[i].loc[j,
                                                           'covarqrtr']/namez[i].loc[j, 'varqrtr']
        else:
            namez[i].loc[j, 'betaqtr_ndaq'] = 0

        namez[i].loc[j, 'covarqrtr'] = np.cov(
            djia_df.loc[j-64:j, 'pct'], namez[i].loc[j-64:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covarqrtr'] != 0:
            namez[i].loc[j, 'betaqtr_djia'] = namez[i].loc[j,
                                                           'covarqrtr']/namez[i].loc[j, 'varqrtr']
        else:
            namez[i].loc[j, 'betaqtr_djia'] = 0
        j += 1


# 250 calendar days - 178 stawk days

    roll_250_am = namez[i]['pct'].rolling(250, min_periods=1).mean()
    namez[i]['rolling250'] = roll_250_am

    namez[i]['varyear'] = 0
    namez[i]['covaryear'] = 0

    namez[i]['betayr_clf'] = 0
    namez[i]['betayr_ndaq'] = 0
    namez[i]['betayr_djia'] = 0

    for j in range(259, k):
        namez[i].loc[j, 'varyear'] = namez[i].loc[j-259:j, 'pct'].var()
        namez[i].loc[j, 'covaryear'] = np.cov(
            sp500_df.loc[j-259:j, 'pct'], namez[i].loc[j-259:j, 'pct'])[0][1]

        if namez[i].loc[j, 'covaryear'] != 0:
            namez[i].loc[j, 'betayr_clf'] = namez[i].loc[j,
                                                         'covaryear']/namez[i].loc[j, 'varyear']
        else:
            namez[i].loc[j, 'betayr_clf'] = 0
        namez[i].loc[j, 'covaryear'] = np.cov(
            ndaq_df.loc[j-259:j, 'pct'], namez[i].loc[j-259:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covaryear'] != 0:
            namez[i].loc[j, 'betayr_ndaq'] = namez[i].loc[j,
                                                          'covaryear']/namez[i].loc[j, 'varyear']
        else:
            namez[i].loc[j, 'betayr_ndaq'] = 0

        namez[i].loc[j, 'covaryear'] = np.cov(
            djia_df.loc[j-259:j, 'pct'], namez[i].loc[j-259:j, 'pct'])[0][1]
        if namez[i].loc[j, 'covaryear'] != 0:
            namez[i].loc[j, 'betayr_djia'] = namez[i].loc[j,
                                                          'covaryear']/namez[i].loc[j, 'varyear']
        else:
            namez[i].loc[j, 'betayr_djia'] = 0
        j += 1

    namez[i] = namez[i].dropna(axis=0)
    namez[i].reset_index(drop=True, inplace=True)
    i = i+1

 # Rolling 250 day


now = datetime.datetime.now()
print("\n This finished at ", now)
colz = ['meaningless', 'high', 'low', 'open','close','volume',
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


amzn_df = namez[0]
amzn_df.columns = colz

amzn_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/amzn_stock_df.csv", header=True)
print()
print("the head of amazon is: \n", amzn_df.loc[:,'date'].head())
print()
print("the tail of amazon is: \n", amzn_df.loc[:,'date'].tail())

tsla_df = namez[1]
tsla_df.columns = colz
tsla_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/tsla_stock_df.csv", header=True)
aapl_df = namez[2]
aapl_df.columns = colz
aapl_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/appl_stock_df.csv", header=True)
evbg_df = namez[3]
evbg_df.columns = colz

evbg_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/evbg_stock_df.csv", header=True)
kmi_df = namez[4]
kmi_df.columns = colz

kmi_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/kmi_stock_df.csv", header=True)
shop_df = namez[5]

shop_df.columns = colz
shop_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/shop_stock_df.csv", header=True)
gmed_df = namez[6]
gmed_df.columns = colz

gmed_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/gmed_stock_df.csv", header=True)
sny_df = namez[7]
sny_df.columns = colz

sny_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/sny_stock_df.csv", header=True)
mck_df = namez[8]
mck_df.columns = colz

mck_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/mck_stock_df.csv", header=True)
et_df = namez[9]
et_df.columns = colz
et_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/et_stock_df.csv", header=True)
insp_df = namez[10]
insp_df.columns = colz
insp_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/insp_stock_df.csv", header=True)
crwd_df = namez[11]
crwd_df.columns = colz
crwd_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Filescrwd_stock_df.csv", header=True)
clf_df = namez[12]
clf_df.columns = colz
clf_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/clf_stock_df.csv", header=True)
UVXY_df = namez[12]
UVXY_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/UVXY_etf_df.csv", header=True)
GLD_df = namez[13]
GLD_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/GLD_etf_df.csv", header=True)
GBTC_df = namez[14]
GBTC_df.to_csv(
    "/Volumes/Elements/GitHub/twitter-project/Data_Files/GBTC_etf_df.csv", header=True)





