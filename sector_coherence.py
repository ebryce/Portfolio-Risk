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

REFRESH = '--refresh' in [arg.lower() for arg in sys.argv]

db = ['C:','Datasets','thesis.db']
overleaf = ['C:','Users','bryce','OneDrive','Documents','Overleaf','Thesis']
conn = sqlite3.connect('\\'.join(db))
c = conn.cursor()

hdf_path = ['C:','Datasets','ferstenberg.h5']
hdf = pd.HDFStore('\\'.join(hdf_path))

print('Loaded hdf at %s with keys: %s' % ('\\'.join(hdf_path), ', '.join(hdf.keys())))
if REFRESH:
	print(' > Will force full dataset refresh')

sector_query = '''
    SELECT ticker, exchange, share_type,
    	sector, industry
    FROM sectors
    WHERE exchange='TSX'
    '''

px_query = '''
    SELECT date, ticker,
    	o, h, l, c, vol
    FROM cfmrc
    WHERE ticker IN ('{Tickers}')
    '''

df_sector = pd.read_sql(sector_query, conn)

fields = ['sector','industry']

for field in fields:
	key = '%s_returns' % field
	if (('/%s' % key) not in hdf.keys()) or REFRESH:
		print('Computing %s returns' % field)
		# SECTOR RETURNS
		sector_returns = []
		i = 1
		for sector in df_sector[field].unique():

			if sector=='-':
				print(' Ignoring useless %s: %s' % (field, sector))
				continue

			print('Sector: %s (%d of %d)' %(sector, i, len(df_sector[field].unique())))
			tickers = df_sector.loc[df_sector[field]==sector]['ticker'].unique()
			print(' > %d tickers' % (len(tickers)))

			# Get px data from our SQLite instance
			px = pd.read_sql(px_query.format(Tickers="','".join(tickers)), conn)
			print(' > %d rows of market data returned for %d tickers' % (len(px.index), len(px['ticker'].unique())))

			# Calculate intraday returns
			px['r_intraday']=(px['c']-px['o'])/px['o']

			# Calculate overnight returns
			returns = px.pivot_table(index=['date'],columns=['ticker'],values=['o','c','r_intraday','vol'])
			for ticker in px['ticker'].unique():
				returns['r_overnight',ticker] = (returns['o',ticker].shift(1)-returns['c',ticker])/returns['o',ticker].shift(1)

			# No longer need open and close
			returns.drop(['c','o'], inplace=True, axis=1)
			returns = returns.replace([np.inf, -np.inf], np.nan)

			# Compute averages
			returns['r_overnight','mean'] = returns.iloc[:, (returns.columns.get_level_values(0)=='r_overnight') & (returns.columns.get_level_values(1).isin(tickers))].mean(axis=1)
			returns['r_overnight','std'] = returns.iloc[:, (returns.columns.get_level_values(0)=='r_overnight') & (returns.columns.get_level_values(1).isin(tickers))].std(axis=1)
			returns['r_intraday','mean'] = returns.iloc[:, (returns.columns.get_level_values(0)=='r_intraday') & (returns.columns.get_level_values(1).isin(tickers))].mean(axis=1)
			returns['r_intraday','std'] = returns.iloc[:, (returns.columns.get_level_values(0)=='r_intraday') & (returns.columns.get_level_values(1).isin(tickers))].std(axis=1)
			returns['vol','mean'] = returns.iloc[:, (returns.columns.get_level_values(0)=='vol') & (returns.columns.get_level_values(1).isin(tickers))].mean(axis=1)
			returns['vol','std'] = returns.iloc[:, (returns.columns.get_level_values(0)=='vol') & (returns.columns.get_level_values(1).isin(tickers))].std(axis=1)
			returns['vol','sum'] = returns.iloc[:, (returns.columns.get_level_values(0)=='vol') & (returns.columns.get_level_values(1).isin(tickers))].sum(axis=1)
			returns['vol','count'] = returns.iloc[:, (returns.columns.get_level_values(0)=='vol') & (returns.columns.get_level_values(1).isin(tickers))].count(axis=1)

			# Drop all stock specific data
			returns = returns.iloc[:, ~(returns.columns.get_level_values(1).isin(tickers))]
			returns[field] = sector

			print(returns.tail())

			sector_returns.append(returns)
			print()
			i+=1
		sector_returns = pd.concat(sector_returns)

		hdf.put(key=key, value=sector_returns, format='t', append=False)
	else:
		print('Skipping %s return calculations' % field)

hdf.close()