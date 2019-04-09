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

# Custom financial data libraries
import utils.findata_utils as fd
import utils.ml_utils as ml_utils

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams

import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore")

from importlib import reload
fd = reload(fd)

import sklearn as sk
#import tensorflow as tf
import xgboost as xgb
#import keras

from imblearn.over_sampling import RandomOverSampler

from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import explained_variance_score, mean_squared_error, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib 

#from keras.models import Sequential
#from keras.optimizers import SGD
#from keras.layers import Dense, Dropout
#from keras.wrappers.scikit_learn import KerasRegressor

#from yellowbrick.regressor import ResidualsPlot, PredictionError

# Connect to databases
db = 'C:\\Datasets\\thesis.db'
overleaf = ['C:','Users','bryce','OneDrive','Documents','Overleaf','Thesis','assets','exports']
conn = sqlite3.connect(db)
c = conn.cursor()

hdf_path = 'C:\\Datasets\\thesis.h5'
hdf = pd.HDFStore(hdf_path)

import warnings
if not sys.warnoptions:
	warnings.simplefilter("ignore")

signals = pd.read_sql('''
	SELECT * FROM signals
	''',
	conn)

signals['trade_date'] = pd.to_datetime(signals['trade_date'])

returns = pd.read_sql('''
	SELECT * FROM daily_abnormal_returns
	WHERE ticker in ('{Tickers}')
	'''.format(Tickers="','".join(signals['ticker'].unique())),
	conn)

returns['datetime'] = returns['date'].apply(lambda date: datetime.strptime(date, '%Y-%m-%d'))

holding_period = 30
look_back = 30

port_return = []
skipped_signals = []

try:
	for i, signal in signals.iterrows():
		
		print('Buy %s (p=%f)' % (signal['ticker'], signal['probability']))
		
		look_back_to = signal['trade_date']-timedelta(days=look_back)
		
		# Choose the day to open the position on
		#  if the first day of the month is a weekend, find the next monday
		open_position_on = signal['trade_date']
		while open_position_on.strftime('%Y-%m-%d') not in returns['date'].values:
			open_position_on = open_position_on+timedelta(days=1)
		print(' > Open position on %s'%open_position_on.strftime('%Y-%m-%d'))
		
		close_position_on = signal['trade_date']+timedelta(days=holding_period)
		ss_return = returns.loc[(returns['ticker']==signal['ticker']) & (returns['datetime'].between(look_back_to, close_position_on))]
		
		# Choose the most appropriate abnormal return calculation
		rebal_dates = returns.loc[returns['ticker']==signal['ticker']]['rebal_date'].unique()
		try:
			rebal_date_to_use = datetime.strptime(max(rebal_dates)[:10],'%Y-%m-%d')
		except:
			print(' > No abnormal return data for %s; skipping' % signal['ticker'])
			signal['reason'] = 'No factor model data for this security'
			skipped_signals.append(pd.DataFrame(signal).T)
			continue
		for possible_date in rebal_dates:
			#print(possible_date)
			if signal['trade_date'] < datetime.strptime(possible_date[:10],'%Y-%m-%d'):
				rebal_date_to_use = possible_date
		ss_return = ss_return.loc[ss_return['rebal_date']==rebal_date_to_use]
		
		print(' > Using CAR calculations for %s' % str(rebal_date_to_use)[:10])
		
		ss_return['d'] = pd.to_datetime(ss_return['date']) - signal['trade_date']
		
		ss_return['culm_return'] = (ss_return['r_daily']+1).cumprod()
		ss_return['car'] = (ss_return['ar_daily']+1).cumprod()
		
		try:
			return_index = ss_return.loc[ss_return['datetime']==open_position_on].iloc[0]
		except:
			if str(open_position_on)[:10] < returns.loc[(returns['ticker']=='AEM')]['date'].min():
				signal['reason'] = 'No market history to this date'
				print(' > Do not have market history to this date; skipping')
				skipped_signals.append(pd.DataFrame(signal).T)
				continue
			else:
				raise(' > Something went wrong')
				signal['reason'] = 'Unknown error'
				skipped_signals.append(pd.DataFrame(signal).T)
				#break
				continue
		
		for field in ['culm_return','car']:
			ss_return[field] = ss_return[field] - return_index[field]
			
		ss_return = ss_return[['ticker','date','d','culm_return', 'car']]

		ss_return['d'] = ss_return['d'].apply(lambda d: d.days)

		ss_return['trade'] = i
		ss_return['probability'] = signal['probability']
		port_return.append(ss_return)

except Exception as e:
	print(ss_return)
	print(signal)
	print()

port_return = pd.concat(port_return)
skipped_signals = pd.concat(skipped_signals)

print('Skipped signals:')
print(skipped_signals)

#probabilities = port_return['probability']
pivot = port_return.pivot_table(index=['d'], columns=['trade'], values=['culm_return','car','probability'], aggfunc=np.mean)
pivot.fillna(method='ffill', inplace=True)
pivot.fillna(method='bfill', inplace=True)

for trade in pivot.columns.get_level_values(1):
	pivot['weight',trade] = pivot['probability',trade]/pivot['probability'].sum(axis=1)


for metric in ['culm_return','car']:
	pivot[metric,'portfolio'] = (pivot[metric]*pivot['weight']).sum(axis=1)
print(pivot)


# Plot the results
fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('white')

ax = fig.add_subplot(1, 1, 1)

ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.grid(True,axis='both',linestyle=':')

colors = {'car':'xkcd:forest green',
	'culm:return':'xkcd_pale orange'}

for metric in ['culm_return','car']:
	ax.plot(pivot.index, pivot[metric,'portfolio'], label=metric, color=colors[metric])
plt.legend(frameon=False, loc='best')

plt.title('Portfolio Returns')
plt.ylabel('Culmulative Returns\\Indexed to Day-0')
plt.xlabel('Days Since (To) End of the Ranking Month')
plt.show()
fig.savefig('\\'.join(overleaf+['portfolio_backtest.png']))
plt.close()