import pandas as pd, numpy as np, sqlite3

t1 = pd.read_excel(".//ciq//Ticker-Changes.xls", skiprows=7)

#t1['Ticker_Pair'] = t1['Ticker']
#t1['Exchange'] = t1['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[:ticker_pair.find(':')])
#t1['Ticker'] = t1['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[ticker_pair.find(':')+1:])
#t1.drop(['Ticker_Pair'], axis=1, inplace=True)

start_flag = ['will Change its Ticker to ', 'has Changed its Ticker to ', ' will change itsTicker to ']
mid_flag = ' from '

# Only show ticker changes; not promotions
print(t1.columns)
t1 = t1.loc[t1['Key Developments by Type'].isin(['Ticker Changes'])]

# Parse the table
t1['date'] = t1['Key Developments By Date']
t1['headline'] = t1['Key Development Headline'].apply(lambda headline:
	headline[headline.find(start_flag[0])+len(start_flag[0]):] if start_flag[0] in headline
	else headline[headline.find(start_flag[1])+len(start_flag[1]):] if start_flag[1] in headline
	else headline[headline.find(start_flag[2])+len(start_flag[2]):])

t1['from'] = t1['headline'].apply(lambda headline:
	headline.split(mid_flag)[1])

t1['to'] = t1['headline'].apply(lambda headline:
	headline.split(mid_flag)[0])

t1['exchange'] = t1['Company Name(s)'].apply(lambda name:
	name[name.find('(')+1:name.find(':')])

# Drop useless columns
t1 = t1[['date','exchange','from','to']]

t1 = t1.loc[t1['exchange'].isin(['TSX'])]

print(t1)

print(list(t1.exchange.unique()))

db = 'C:\\Datasets\\thesis.db'
conn = sqlite3.connect(db)
c = conn.cursor()

t1.to_sql('ticker_changes', conn, if_exists='replace', index=False)