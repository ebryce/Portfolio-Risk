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

db = ['C:','Datasets','thesis.db']
conn = sqlite3.connect('\\'.join(db))
c = conn.cursor()

rebal_query = '''
	SELECT f.date,f.action,f.[index],f.exchange,f.ticker,f.name,
		CASE WHEN f.action='Add' THEN 1 ELSE 0 END AS [add],
		CASE WHEN f.action='Delete' THEN 1 ELSE 0 END AS [delete],
		CASE WHEN f.action='Add' THEN 1 ELSE -1 END AS flag,
		c.c AS rebalpx
	FROM factset_index_changes f
	INNER JOIN cfmrc c
		ON c.ticker=f.ticker
		AND c.date=SUBSTR(f.date,0,11)
	'''

rebal = pd.read_sql(rebal_query, conn)
print(rebal.head())
rebal.to_sql('priced_index_changes', conn, index_label=['date','ticker'], if_exists='replace', index=False)