# System libraries
import sys, os, gc
import datetime

# Math libraries
import math, random
import pandas as pd, numpy as np
import scipy
from scipy import stats
from datetime import timedelta
from datetime import datetime
import itertools

# Data storage libraries
import pickle, sqlite3, simpledbf, boto3

DEBUG = '--debug' in [arg.lower() for arg in sys.argv]

db = ['C:','Datasets','thesis.db']
overleaf = ['C:','Users','bryce','OneDrive','Documents','Overleaf','Thesis']

conn = sqlite3.connect('\\'.join(db))
c = conn.cursor()



px_query = '''
    SELECT date, ticker,
    	o, h, l, c, vol
    FROM cfmrc
    --DEBUGLIMIT 10000
    '''

if DEBUG:
	px_query = px_query.replace('--DEBUG','')

px = pd.read_sql(px_query, conn)

print(' > %d rows of market data returned for %d tickers' % (len(px.index), len(px['ticker'].unique())))

print(px.head())

if len(px.index)<2:
	raise Exception('Too little market data.')

# Calculate intraday returns
px['r_intraday']=(px['c']-px['o'])/px['o']

returns = px.pivot_table(index=['date'],columns=['ticker'],values=['o','c','r_intraday','vol'])
print(returns.head())

bad_tickers = []
for ticker in px['ticker'].unique():
	try:
		returns['r_overnight',ticker] = (returns['o',ticker].shift(1)-returns['c',ticker])/returns['o',ticker].shift(1)
		returns['r_daily', ticker] = returns['r_overnight',ticker].fillna(0) + returns['r_intraday', ticker]
	except Exception as e:
		print('Error:')
		print(str(e))
		print(ticker)
		print(px.loc[px['ticker']==ticker].head())
		returns['r_overnight',ticker] = np.nan
		returns['r_intraday', ticker] = np.nan
		returns['r_daily', ticker] = np.nan
		print()
		bad_tickers.append(ticker)
print('Done computing daily returns')
print(returns.head())
print('Unstacking')
returns = returns.unstack().reset_index()
returns.columns = ['metric','ticker','date','value']
print(returns.head())
print('Dropping %d bad tickers' % len(bad_tickers))
returns = returns.loc[~returns['ticker'].isin(bad_tickers)]
print(returns.head())

print('Restacking')
returns = returns.pivot_table(index=['date','ticker'],columns=['metric'],values=['value']).reset_index()
returns.columns = returns.columns.get_level_values(1)
returns.columns = ['date','ticker','c','o','r_daily','r_intraday','r_overnight','vol']

print(returns.head())
returns.to_sql('daily_returns', conn, index_label=['date','ticker'], if_exists='replace', index=False)