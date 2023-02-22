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

strt = dt.datetime.now()
df2 = relabel2(df)
endy = dt.datetime.now()

print("The time taken for the second function is: ", endy - strt)
print("the head of df2 is \n", df2.head())
