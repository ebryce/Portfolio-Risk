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
BUILD_CHILDREN =  '--refactor-msi' in [arg.lower() for arg in sys.argv]

db = ['C:','Datasets','thesis.db']
overleaf = ['C:','Users','bryce','OneDrive','Documents','Overleaf','Thesis']

conn = sqlite3.connect('\\'.join(db))
c = conn.cursor()

hdf_path = ['C:','Datasets','ferstenberg.h5']
hdf = pd.HDFStore('\\'.join(hdf_path))
print('Using SQLite database at %s' % '\\'.join(db))
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

mapping_query = '''SELECT DISTINCT sector, industry FROM sectors'''

df_sector = pd.read_sql(sector_query, conn)

fields = ['industry','sector']

if REFRESH:
	print('Dropping existing tables')
	for field in fields:
		key = '%s_returns' % field
		try:
			print(' > Dropping {Table}'.format(Table=key))
			c.execute('DROP TABLE {Table};'.format(Table=key))
			conn.commit()
		except:
			print(' > Could not drop {Table}'.format(Table=key))

for field in fields:
	key = '%s_returns' % field
	if (('/%s' % key) not in hdf.keys()) or REFRESH:
		print('Computing %s returns' % field)
		# SECTOR RETURNS
		all_returns = []
		i = 1
		for sector in df_sector[field].unique():

			try:

				if sector=='-':
					raise Exception('Ignoring useless %s: %s' % (field, sector))

				print('%s: %s (%d of %d)' %(field, sector, i, len(df_sector[field].unique())))
				tickers = df_sector.loc[df_sector[field]==sector]['ticker'].unique()
				print(' > %d tickers' % (len(tickers)))

				if len(tickers)==0:
					raise Exception('Too few tickers; skipping')

				# Get px data from our SQLite instance
				px = pd.read_sql(px_query.format(Tickers="','".join(tickers)), conn)
				print(' > %d rows of market data returned for %d tickers' % (len(px.index), len(px['ticker'].unique())))

				if len(px.index)<2:
					raise Exception('Too little market data; skipping')

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

				all_returns.append(returns)
				print()
			except Exception as e:
				
				error_msg = ['-----',
					datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
					'  ERROR: %s' % e,
					'  %s'%', '.join(sys.argv),
					'  %s/%s'%(field, sector),'','']

				error_msg = '\n'.join(error_msg)
				print(error_msg)
				with open('errlog.txt', 'a') as f:
					f.write(error_msg)
			i+=1
		print('Done compiling tables; concatenating')
		all_returns = pd.concat(all_returns)
		print(all_returns.head())
		print(all_returns.tail())

		hdf.put(key=key, value=all_returns, format='t', append=False)

		all_returns = all_returns.reset_index()
		
		all_returns.columns = ['date','r_overnight_mean','r_overnight_std','r_intraday_mean','r_intraday_s',
			'vol_mean','vol_std','vol_sum','count',field]

		all_returns.to_sql(key, conn, index_label=['date',field], if_exists='replace', index=False)
	else:
		print('Skipping %s return calculations' % field)

hdf.close()

if BUILD_CHILDREN and REFRESH:
	print('Fama/French Data')
	import parsers.parse_findata

	stock_specific=True

	print('Computing market-sector-industry returns')

	msi_query = '''
	SELECT date, d.sector, d.industry,--STOCK_SPECIFICm.ticker,
	    fr AS riskfree_return,
	    mr AS market_return,
	    mr-fr AS market_excess_return,
	    sr AS sector_return,
	    sr-mr AS sector_excess_return,
	    ir AS industry_return,
	    ir-sr AS industry_excess_return
	FROM(
	    SELECT
	        i.date, i.industry, s.sector,
	        COALESCE(i.r_overnight_mean,0)+COALESCE(i.r_intraday_mean,0) AS ir,
	        COALESCE(s.r_overnight_mean,0)+COALESCE(s.r_intraday_mean,0) AS sr,
	        COALESCE(f.RF,0) as fr, COALESCE(f.Rm,0) AS mr
	    FROM industry_returns i
	    INNER JOIN (SELECT DISTINCT industry, sector FROM sectors) m
	        ON m.industry=i.industry
	    LEFT JOIN sector_returns s
	        ON s.sector=m.sector AND s.date=i.date
	    LEFT JOIN french f
	        ON f.date=i.date
	    ORDER BY i.date
	    ) d
	--STOCK_SPECIFICINNER JOIN (SELECT industry, sector, ticker FROM sectors) m
	--STOCK_SPECIFIC    ON m.industry=d.industry AND m.sector=d.sector
	'''

	#index_label = ['date','ticker']
	if stock_specific:
		msi_query = msi_query.replace('--STOCK_SPECIFIC','')
	#else:
	#	index_label = ['date','sector','industry']

	df = pd.read_sql(msi_query, conn)
	print(df.head())
	df.to_sql('msi', conn, if_exists='replace', index=False)

if REFRESH:
	print('Refreshing daily returns')
	import parsers.parse_daily_returns
else:
	print('Not reloading daily returns')

study_indices = ['S&P/TSX Composite Index']
print('Preparing dataset for regression')
print('Studying only %s' % ', '.join(study_indices))
regression_query = '''
	SELECT
		s.ticker, s.[date],
		fs.[date] AS rebal_date,
		fs.Action AS action,
		fs.[index] AS [index],
		s.r_daily,
		msi.riskfree_return,
		msi.market_excess_return,
		msi.sector_excess_return,
		msi.industry_excess_return
	FROM daily_returns s
	INNER JOIN msi 
		ON s.[date]=msi.[date]
		AND s.ticker=msi.ticker
	INNER JOIN factset_index_changes fs
		ON fs.ticker=s.ticker
		AND fs.[index] IN ('{StudyIndex}')
	--WHERE msi.industry='Asset Management and Custody Banks'
	'''.format(StudyIndex="','".join(study_indices))

print('Querying for price data')
df = pd.read_sql(regression_query, conn)
print('Queried price data')

print(df.head())

if REFRESH:
	print('Calculating correlation coefficients for factors')
	X_cols = df.columns[-4:]
	y_col = 'r_daily'
	label = 'ticker'
	print('Per %s: regresing %s against %s' % (label, y_col, ', '.join(X_cols)))
	from sklearn import linear_model
	import warnings
	warnings.simplefilter('ignore')


	regressions = []
	i = 1
	n = len(df[label].unique())
	for ticker in df[label].unique():

		subset = df.loc[df[label]==ticker]

		actions = subset['action'].unique()
		rebal_dates = subset['rebal_date'].unique()
		cutoff_date = rebal_dates.max()
		print('%s: %s on %s (%d of %d), only evaluating pre-%s'% (ticker, '/'.join(actions), '/'.join(rebal_dates), i, n, cutoff_date))
		subset = subset.loc[subset['rebal_date']==cutoff_date]
		action = subset['action'].unique()[0]

		# Data cleaning, remove all strings
		subset[y_col] = subset[y_col].apply(lambda value: np.nan if type(value)!=float else value)
		for col in X_cols:
			subset[col] = subset[col].apply(lambda value: np.nan if type(value)!=float else value)

		# Remove all extreme values
		subset = subset.replace(np.inf, 1)
		subset = subset.replace(-np.inf, -1)
		subset = subset.fillna(0)
		subset[y_col] = subset[y_col].apply(lambda value: max(value,-1))
		for col in X_cols:
			subset[col] = subset[col].apply(lambda value: max(value,-1))

		reg = linear_model.LinearRegression()

		for pre_event in [False, True]:
			regression = pd.Series(name=ticker)
			if pre_event:
				subset = subset.loc[subset['date']<cutoff_date]

			if len(subset.index)==0:
				print('  Too little market data for the pre-event %s; skipping' % pre_event)
				continue

			reg.fit(subset[X_cols].values, subset[y_col].values)

			for i in range(len(X_cols)):
				regression[X_cols[i]] = reg.coef_[i]
			regression['residual'] = reg.intercept_

			regression = pd.DataFrame(regression).T
			regression['pre_event'] = pre_event
			regression['action'] = action
			regressions.append(regression)
			print(regression)
		#break
		i+=1
	regressions = pd.concat(regressions).reset_index()
	regressions.columns = ['ticker','riskfree_return','market_excess_return','sector_excess_return','industry_excess_return','residual','pre_event','action']

	bounds = [-5,5]
	bounds_rf = [-1,1]
	for column in ['market_excess_return','sector_excess_return','industry_excess_return']:
		regressions[column].clip(upper=max(bounds), lower=min(bounds))
	regressions['riskfree_return'] = regressions['riskfree_return'].clip(upper=max(bounds_rf), lower=min(bounds_rf))

	regressions.to_sql('factor_coefficients', conn, if_exists='replace', index=False)
else:
	print('Skipping calculation of correlation coefficients')
	regressions = pd.read_sql('''SELECT * FROM factor_coefficients''', conn)
print(regressions.head())


if REFRESH:
	print('Calculating abnormal returns')
	df['pre_event'] = df['date']<df['rebal_date']

	df = df.merge(regressions, on=['ticker','pre_event','action'], suffixes=('','_corr'), how='inner')

	bounds = [-1,10]
	
	df['er_daily'] = df['riskfree_return']*df['riskfree_return_corr'] + df['market_excess_return']*df['market_excess_return_corr'] + df['sector_excess_return']*df['sector_excess_return_corr'] + df['industry_excess_return']*df['industry_excess_return_corr']
	
	df['er_daily'] = df['er_daily'].clip(upper=max(bounds), lower=min(bounds))
	df['r_daily'] = df['r_daily'].clip(upper=max(bounds), lower=min(bounds))
	
	df['ar_daily'] = df['r_daily'] - df['er_daily']

	df.to_sql('daily_abnormal_returns', conn, if_exists='replace', index=False)

	print(df.head())
else:
	print('Skipping abnormal return calculation')