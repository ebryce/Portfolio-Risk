# System libraries
import sys, os
import pandas as pd, numpy as np
import sqlite3

db = 'C:\\Datasets\\thesis.db'
conn = sqlite3.connect(db)
c = conn.cursor()

date='2018-06-29'

tickers = pd.read_sql('''
	SELECT * FROM in_the_comp WHERE date='{Date}'
	'''.format(Date=date),
	conn).unstack().reset_index()
tickers.columns = ['ticker','null','held']
tickers.drop(['null'],axis=1,inplace=True)
tickers = tickers.iloc[1:,]
tickers = list(tickers.loc[tickers['held']==1]['ticker'].unique())

print(tickers)

mkt_caps = pd.read_sql('''
	SELECT c.date, c.ticker,
		COALESCE(c.c*so.shares_out, 0) AS mkt_cap
	FROM cfmrc c

	-- Merge to shares outstanding to calculate the market cap
	INNER JOIN shares_out so
		ON so.ticker=c.ticker

	-- Only show TSX Composite securities
	--INNER JOIN index_holdings_2019 i
	--	ON i.[index]='S&P/TSX Composite Index'
	--	AND i.exchange='TSX'
	--	AND i.ticker=c.ticker

	WHERE
		c.date='{Date}'					-- Only show latest data
		AND c.ticker IN ('{Tickers}') 	-- Only show names in the comp
	'''.format(Tickers="','".join(tickers), Date=date),
		conn).drop_duplicates()

mkt_cap_day0 = mkt_caps['mkt_cap'].sum()

print(mkt_caps.head())
print(mkt_cap_day0)

index_values = pd.read_excel('.//bbg//SPTSXIndex_Value.xlsx')[['Date','Last Price']]

index_value_day0 = index_values.loc[index_values['Date']==date]['Last Price'].values[0]
print(index_value_day0)
index_values['index'] = index_values['Last Price']/index_value_day0
index_values['mkt_cap'] = index_values['index'] * mkt_cap_day0

index_values.columns = ['date','price','indexed','mkt_cap']

print(index_values)

index_values.to_sql('index_mkt_cap_bbgmethod', conn, if_exists='replace', index=False)