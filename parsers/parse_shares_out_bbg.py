import pandas as pd, numpy as np
import os
import sqlite3

db = 'C:\\Datasets\\thesis.db'
conn = sqlite3.connect(db)
c = conn.cursor()

shares_out = pd.read_excel('bbg//SharesOut.xlsx').iloc[2:,:4]
shares_out = shares_out.reindex()

print(shares_out.head())

shares_out['Ticker'] = shares_out['Ticker'].apply(lambda tkr: tkr[:tkr.find(' ')])
shares_out['Ticker'] = shares_out['Ticker'].apply(lambda tkr: tkr.replace('/','.').replace('-','.'))

shares_out.columns = ['ticker','name','shares_out', 'float']

for value in ['shares_out','float']:
	shares_out[value] = shares_out[value].replace(' ',0).fillna(0)
	shares_out[value] = shares_out[value].apply(lambda so: float(so))

print(shares_out)

shares_out.to_sql('shares_out', conn, if_exists='replace', index=False)