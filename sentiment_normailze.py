import pandas as pd
import numpy as np
import datetime as dt

file = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_tn_scored.csv'

df = pd.read_csv(file, index_col=0)


def relabel2(file):
    neg_mask = file['label'] == 'NEG'
    neu_mask = file['label'] == 'NEU'
    pos_mask = file['label'] == 'POS'
    score = file['score']

    file['neg'] = np.where(neg_mask, score, (1 - score)/2)
    file['neu'] = np.where(neu_mask, score, (1 - score)/2)
    file['pos'] = np.where(pos_mask, score, (1 - score)/2)

    file['neg'] = np.where(neu_mask, (1 - score)/2, file['neg'])
    file['neu'] = np.where(neg_mask, (1 - score)/2, file['neu'])
    file['pos'] = np.where(neg_mask, (1 - score)/2, file['pos'])
    file['neg'] = np.where(pos_mask, (1 - score)/2, file['neg'])
    file['neu'] = np.where(pos_mask, (1 - score)/2, file['neu'])

    return file


strt = dt.datetime.now()
df1 = relabel2(df)
endy = dt.datetime.now()

print("The time taken for the first function is: ", endy - strt)
print("the head of df1 is \n", df1.head())

df1.to_csv('/Volumes/Elements/GitHub/twitter-project/Data_Files/Amazon_tn_norm.csv')
file2 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_tn_scored.csv'

df2 = pd.read_csv(file2, index_col=0)
strt = dt.datetime.now()
df2 = relabel2(df2)
endy = dt.datetime.now()

print("The time taken for the second function is: ", endy - strt)
print("the head of df2 is \n", df2.head())

df2.to_csv('/Volumes/Elements/GitHub/twitter-project/Data_Files/ATEN_tn_norm.csv')

file3 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_tn_scored.csv'

df3 = pd.read_csv(file3, index_col=0)

strt = dt.datetime.now()
df3 = relabel2(df3)
endy = dt.datetime.now()

print("The time taken for the third function is: ", endy - strt)
print("the head of df3 is \n", df3.head())

df3.to_csv('/Volumes/Elements/GitHub/twitter-project/Data_Files/Exxon_tn_norm.csv')

file4 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_tn_scored.csv'

df4 = pd.read_csv(file4, index_col=0)

strt = dt.datetime.now()
df4 = relabel2(df4)
endy = dt.datetime.now()

print("The time taken for the fourth function is: ", endy - strt)
print("the head of df4 is \n", df4.head())

df4.to_csv('/Volumes/Elements/GitHub/twitter-project/Data_Files/INSP_tn_norm.csv')

file5 = '/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_tn_scored.csv'

df5 = pd.read_csv(file5, index_col=0)

strt = dt.datetime.now()
df5 = relabel2(df5)
endy = dt.datetime.now()

print("The time taken for the fifth function is: ", endy - strt)
print("the head of df5 is \n", df5.head())

df5.to_csv('/Volumes/Elements/GitHub/twitter-project/Data_Files/KMI_tn_norm.csv')
