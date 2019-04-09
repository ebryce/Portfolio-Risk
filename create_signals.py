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
import tensorflow as tf
import xgboost as xgb
import keras

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

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

from yellowbrick.regressor import ResidualsPlot, PredictionError

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

index_mkt_cap = pd.read_sql('''SELECT * FROM index_mkt_cap_bbgmethod''', conn)
index_mkt_cap.columns = ['date','price','pct_day0','SPTSXComp']
#index_mkt_cap['threshold'] = index_mkt_cap['SPTSXComp']*0.0004
#index_mkt_cap['lower'] = index_mkt_cap['threshold']*(1-limit)
#index_mkt_cap['upper'] = index_mkt_cap['threshold']*(1+limit)

index_mkt_cap['datetime'] = index_mkt_cap['date'].apply(lambda date: datetime.strptime(str(date)[:10], '%Y-%m-%d'))
index_mkt_cap['month'] = index_mkt_cap['datetime'].apply(lambda date: date.strftime('%Y-%m'))
index_mkt_cap['date'] = index_mkt_cap['datetime'].apply(lambda date: date.strftime('%Y-%m-%d'))
index_mkt_cap['is_ranking_week'] = index_mkt_cap['datetime'].apply(lambda date: (date+timedelta(days=14)).month!=date.month)

index_mkt_cap.drop(['datetime'],axis=1, inplace=True)

print(index_mkt_cap.head())

avg_mkt_caps = pd.DataFrame(index_mkt_cap.loc[index_mkt_cap['is_ranking_week']].groupby(by=['month'])['SPTSXComp'].mean()).reset_index()
print(avg_mkt_caps.head())

# MARGINAL SECURITIES ONLY

index_changes = pd.read_sql('''
	SELECT date, action, ticker,
		SUBSTR(date,0,8) AS month
	FROM factset_index_changes
	WHERE [index] IN ('{StudyIndex}')
	'''.format(StudyIndex='S&P/TSX Composite Index'), conn)

index_changes['ranking_month'] = index_changes['month'].apply(lambda month: (datetime.strptime(month, '%Y-%m')-timedelta(days=28)).strftime('%Y-%m'))
#tickers = index_changes['ticker'].unique()

px_cfmrc = pd.read_sql('''
	SELECT c.date, c.ticker, c.c, c.vol--, s.shares_out, c.c*s.shares_out AS mkt_cap
	FROM cfmrc c
	--WHERE mkt_cap>{MinMktCap}*0.75
		--AND c.ticker IN ('{Tickers}')
	'''.format(Tickers='', MinMktCap='')
	, conn)

so = pd.read_sql('''
	SELECT so.ticker, so.shares_out, so.float
	FROM shares_out so
	--WHERE mkt_cap>{MinMktCap}*0.75
	--WHERE c.ticker IN ('{Tickers}')
	'''.format(Tickers='', MinMktCap=''),
				 conn)

print(index_changes.head())
print(px_cfmrc.head())
print(so.head())

# TICKER CHANGES

ticker_changes = pd.read_sql('''
	SELECT * FROM ticker_changes
	WHERE date>'{cfmrc_date}'
	'''.format(cfmrc_date=px_cfmrc.date.max())
	, conn)[['from','to']]

#print(ticker_changes.head())

for i, change in ticker_changes.iterrows():
	px_cfmrc['ticker'].replace(change['from'],change['to'], inplace=True)
	
#print(px_cfmrc.head())

px = px_cfmrc.merge(so, on=['ticker'], how='inner')

# Deal with nonsense
px['c'] = px['c'].apply(lambda c: float(c))
px['shares_out'] = px['shares_out'].replace(' ',0)
px['shares_out'] = px['shares_out'].apply(lambda so: float(so))

#print(px.head())
px['mkt_cap'] = px['shares_out']*px['c']
print(px.head())

#SECURITY LIST
# Do not look at securities already in the index
indexed = pd.read_sql('SELECT * FROM in_the_comp', conn).set_index(['date']).unstack().reset_index()
indexed.columns = ['ticker','date','indexed']
indexed['indexed'] = indexed['indexed'].replace(1,True).replace(0,False)
indexed['month'] = indexed['date'].apply(lambda date: date[:7])

# Aggregated indexing per month
#indexed = indexed.merge(index_changes, on=['ticker','month'])#
#indexed = indexed.drop(['date_x','date_y'],axis=1)
indexed = indexed.drop_duplicates().groupby(by=['ticker','month'])
indexed = indexed['indexed'].max().reset_index()#.drop(['action'],axis=1)

print(indexed.head())

sample = px.copy(deep=True)
sample['datetime'] = sample['date'].apply(lambda date: datetime.strptime(str(date)[:10], '%Y-%m-%d'))
sample['is_ranking_week'] = sample['datetime'].apply(lambda date: (date+timedelta(days=14)).month!=date.month)
sample['month'] = sample['date'].apply(lambda date: str(date)[:7])

# Only look at the ranking week
sample = sample.loc[sample['is_ranking_week']]

sample = sample.merge(index_mkt_cap, on=['date'], how='inner', suffixes=('','_idc'))
sample.drop(['datetime','price','pct_day0','month_idc','is_ranking_week_idc'], axis=1, inplace=True)

print(sample.head())

sample['mkt_cap_X_vol'] = sample['mkt_cap']*sample['vol']
sample['c_X_vol'] = sample['c']*sample['vol']

grouped = sample.groupby(by=['ticker','month'])[['vol','mkt_cap_X_vol', 'c_X_vol']].sum().reset_index()
grouped['VWMC'] = grouped['mkt_cap_X_vol']/grouped['vol']
grouped['VWAP'] = grouped['c_X_vol']/grouped['vol']
grouped.drop(['mkt_cap_X_vol','c_X_vol'], axis=1, inplace=True)
print(grouped.head())

# Merge to list of changes
grouped = grouped.merge(index_changes, left_on=['month','ticker'],
					  right_on=['ranking_month','ticker'],
					  how='outer', suffixes=('','_chg'))

# Merge to list of indexed securities
grouped = grouped.merge(indexed, left_on=['ticker','month'],
					 right_on=['ticker','month'],
					 how='outer', suffixes=('','_idx'))

grouped = grouped.merge(avg_mkt_caps, on=['month'], how='left')

grouped = grouped.merge(so[['ticker','float']], on=['ticker'], how='inner')
grouped['turnover'] = grouped['vol']/grouped['float']
grouped['turnover'] = grouped['turnover'].replace([np.inf, -np.inf], np.nan)
grouped['turnover'].fillna(grouped['turnover'].mean(), inplace=True)

#sample['indexed']=sample['indexed'].fillna(sample['action']=='Delete')
grouped['indexed'].fillna(False, inplace=True)
grouped['pct_contrib'] = grouped['VWMC']/(grouped['VWMC']+grouped['SPTSXComp'])

grouped.drop(['date','month_chg','ranking_month'],axis=1,inplace=True)

print(grouped.head())

# AGG PER RANKING PERIOD

# Apply restrictions
filtered = grouped.loc[grouped['month']>'2010-01']
filtered = filtered.loc[(~filtered['indexed']) | ~(filtered['action'].isna())]
filtered = filtered.loc[filtered['VWAP']>0.5]
print(filtered.head())

# TRAIN THE MODEL

filtered['add']=filtered['action']=='Add'
filtered['del']=filtered['action']=='Delete'

from imblearn.over_sampling import SMOTE

SEED=0xDEADBEEF

y_col = 'add'
X_cols = ['pct_contrib','turnover','VWAP','vol','VWMC','SPTSXComp']
X = filtered[X_cols]
y = filtered[y_col]

X_test, X_train, y_test, y_train  = sk.model_selection.train_test_split(X.values, y.values, test_size=0.2, random_state=SEED)

sm = SMOTE(random_state=SEED)
X_train_resample, y_train_resamle = sm.fit_resample(X_train, y_train)

print(len(X_train), len(X_test))

log_clf = LogisticRegression()
log_clf.fit(X_train_resample, y_train_resamle)

print(log_clf.score(X_train, y_train))

y_pred = log_clf.predict(X_test)
y_pred_prob = log_clf.predict_proba(X_test)
ml_utils.clf_model_eval(y_test, y_pred, classes=['n/c','Add'])
plt.show()
plt.close()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(5,5))

if len(X_cols)>1:
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X.values[:,0],X.values[:,1],y.values)
	ax.set_zlabel(y_col)
else:
	ax = fig.add_subplot(111)

	ax.spines['left'].set_visible(True)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(True)
	ax.grid(True,axis='both',linestyle=':')

	ax.scatter(X.values, y.values)
	ax.set_xlabel(X_cols[0])
	ax.set_ylabel(y_col)
plt.show()
fig.savefig('\\'.join(overleaf+['confusion_matrix.png']))
plt.close()

#from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob[:,1])
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve
fig = plt.figure(figsize=(5,5))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.grid(True,axis='both',linestyle=':')

ax.plot(fpr, tpr, label='ROC (A=%0.2f)' % roc_auc)
plt.legend(frameon=False, loc='best')

plt.title('ROC Curve for the Signal Generator')
plt.ylabel('False Positive Rate')
plt.xlabel('False Negative Rate')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
fig.savefig('\\'.join(overleaf+['roc_curve.png']))
plt.close()

# CONSTRUCT A PORTFOLIO

signals = filtered.copy(deep=True)
signals['prediction'] = log_clf.predict(X)
signals['probability'] = log_clf.predict_proba(X)[:,1]

signals = signals.loc[signals['prediction']]

signals['trade_date'] = signals['month'].apply(lambda month: (datetime.strptime(month, '%Y-%m')+timedelta(days=31)).replace(day=1))
signals = signals[['ticker','trade_date','probability']]

signals.to_sql('signals', conn, if_exists='replace', index=False)

import backtest