# System libraries
import sys, os
import pandas as pd, numpy as np
import sqlite3

db = 'C:\\Datasets\\thesis.db'
conn = sqlite3.connect(db)
c = conn.cursor()

DEBUG = False
study_index = 'S&P/TSX Composite Index'
exchange = 'TSX'

changes = pd.read_sql('''
    SELECT
        ticker, SUBSTR(date,0,11) AS date, action
    FROM factset_index_changes
    WHERE [index] IN ('{Index}')
        AND exchange=('{Exchange}')
    --LIMIT 100
    '''.format(Index=study_index, Exchange=exchange), conn)
print(changes.head())
print()

holdings = pd.read_sql('''
    SELECT
        ticker
    FROM index_holdings_2019
    WHERE [index] IN ('{Index}')
        AND exchange=('{Exchange}')
    --LIMIT 100
    '''.format(Index=study_index, Exchange=exchange), conn)
print(holdings.head())
print()

tickers = np.concatenate([holdings['ticker'].unique(), changes['ticker'].unique()])
#print(tickers)
#print()

all_mkt_cap = pd.read_sql('''
    SELECT
        c.ticker, c.date, 
        COALESCE(s.shares_out,0)*COALESCE(c.c,0) AS mkt_cap
    FROM cfmrc c
    LEFT JOIN shares_out s
        ON c.ticker=s.ticker
    WHERE mkt_cap>0 AND c.ticker IN ('{Tickers}')'''.format(Tickers="','".join(tickers)), conn)

#all_mkt_cap = all_mkt_cap.loc[all_mkt_cap['ticker'].isin(tickers)]
print(all_mkt_cap.head())

# Index composition
index = changes.drop_duplicates().set_index(['date','ticker']).unstack()
index = index.fillna(method='ffill').fillna('Delete').unstack().reset_index()
index.drop(['level_0'], axis=1, inplace=True)
index.columns = ['ticker','date','state']
index = index.replace('Delete', 0).replace('Add',1)
index = index.set_index(['date','ticker']).unstack()

# Market capitalizations
mkt_cap = all_mkt_cap.drop_duplicates().pivot_table(index=['date'],columns=['ticker'],values=['mkt_cap'])
contribs = mkt_cap.merge(index, left_on=mkt_cap.index, right_on=index.index, how='left')
contribs['state'] = contribs['state'].fillna(method='ffill')

# Fill the blanks
index=contribs['state']
mkt_cap=contribs['mkt_cap']
index.set_index(contribs['key_0'], inplace=True)
mkt_cap.set_index(contribs['key_0'], inplace=True)

# Find overall market capitalization of the index
contribs = index.multiply(mkt_cap)
contribs = contribs.loc[contribs.sum(axis=1)>0]

mkt_cap = contribs.sum(axis=1)
mkt_cap.name=study_index
mkt_cap.index.name='date'
mkt_cap = pd.DataFrame(mkt_cap).reset_index()
mkt_cap.columns = ['date','SPTSXComp']

mkt_cap.to_sql('index_mkt_cap', conn, if_exists='replace', index=False)
index = index.dropna().reset_index()
cols = ['date']
cols.extend(list(index.columns[1:]))
index.columns = cols
index.to_sql('in_the_comp', conn, if_exists='replace', index=False)

print(mkt_cap)