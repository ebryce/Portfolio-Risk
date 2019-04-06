import pandas as pd, numpy as np, sqlite3

t1 = pd.read_excel("..//factset//All-Canadian-Index_Ratios_1.xls", skiprows=7)
t2 = pd.read_excel("..//factset//All-Canadian-Index_Ratios_2.xls", skiprows=7)
t3 = t1.merge(t2, on=['Ticker','Security Name'])

t3 = t3.replace('-',np.nan)
t3 = t3.replace('NM',np.nan)

columns = []
for column in t3.columns:
    newcol = column
    if '%' in column:
        t3[column]=t3[column]/100
        column = column.replace('%','')
    if '[CQ' in column:
        date = column[column.find('[')+1:column.find(']')].split(' ')
        year=int(date[1])
        quarter=int(date[0][2:])
        
        factor = column[:column.find('[')]
        factors = factor.split(' ')
        factors = [word[0].upper()+word[1:] if len(word)>0 else word for word in factors]
        factor = ''.join(factors)
        
        newcol = '_'.join([factor, str(year), 'Q%s'%str(quarter)])
    newcol = newcol.replace(',', '_')
    newcol = newcol.replace('&', '')
    newcol = newcol.replace('YearGrowth', 'ag')
    newcol = newcol.replace('YrGrowth', 'ag')
    newcol = newcol.replace(' ','')
    columns.append(newcol)
t3.columns = columns

t3['Ticker_Pair'] = t3['Ticker']
t3['Exchange'] = t3['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[:ticker_pair.find(':')])
t3['Ticker'] = t3['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[ticker_pair.find(':')+1:])

db = 'C:\\Datasets\\thesis.db'
conn = sqlite3.connect(db)
c = conn.cursor()

t3.to_sql('ratios', conn, if_exists='replace', index=False)