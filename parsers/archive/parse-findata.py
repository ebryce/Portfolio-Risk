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
overleaf = ['C:','Users','bryce','OneDrive','Documents','Overleaf','Thesis']

conn = sqlite3.connect('\\'.join(db))
c = conn.cursor()

import findata_utils as fd

# Get Fama-French data

REFRESH = '--refresh' in [arg.lower() for arg in sys.argv]

if REFRESH:
	for table in ['french']:
		c.execute('DROP TABLE {Table}'.format(Table=table))
	conn.commit()

for dataset in ['North_America_3_Factors_Daily']:
	print('French dataset: %s' % dataset)
	threefactor = fd.get_french(dataset)
	for column in threefactor.columns[1:]:
		threefactor[column] = threefactor[column]*0.01
	threefactor['Rm'] = threefactor['Mkt-RF']+threefactor['RF']
	# Reformat dates
	threefactor['date'] = threefactor['Date'].apply(lambda Date: datetime.strptime(str(Date),'%Y%m%d'))
	threefactor['date'] = threefactor['date'].apply(lambda Date: Date.strftime('%Y-%m-%d'))
	threefactor.drop(['Date'],inplace=True, axis=1)

	threefactor['culm_Rm'] = (threefactor['Rm']+1).cumprod()
	threefactor['dataset'] = dataset
	print(' > %d rows, %d columns downloaded' % (len(threefactor.index), len(threefactor.columns)))

	try:
		existing_rows_query = '''
			SELECT *
			FROM french
			WHERE dataset!='{Dataset}'
			'''.format(Dataset=dataset)

		existing_rows = pd.read_sql(existing_rows_query, conn)
		print(' > %d rows, %d columns from the existing rows query' % (len(existing_rows.index), len(existing_rows.columns)))
		if len(existing_rows.index)>0:
			threefactor = pd.concat([existing_rows,threefactor])
		else:
			print(' > no existing rows; will not merge')
	except Exception as e:
		if 'no such table: french' in str(e):
			print(' > No french table in db')
		else:
			print(str(e))

	print(' > %d rows, %d columns together' % (len(threefactor.index), len(threefactor.columns)))

	#threefactor['date'] = pd.to_datetime(threefactor['date'],format=datetime,errors='coerce')

	print(threefactor.head())

	threefactor.to_sql('french', conn, index_label=['date'], if_exists='replace', index=False)