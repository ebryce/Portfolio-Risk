import pandas as pd, numpy as np, sqlite3, datetime

db = 'C:\\Datasets\\thesis.db'
conn = sqlite3.connect(db)

histvol = pd.read_excel('../bbg/SPTSX-Comp_Hist-Vol.xlsx')
histvol['CR'] = histvol['CR'].apply(lambda cr: datetime.datetime.strptime(cr, '%m/%d/%y'))

newcols = ['date']
for column in histvol.columns[1:]:
    days = column[column.rfind('(')+1:column.rfind(')')]
    product = ''.join(column.split(' ')[:2])
    newcols.append('_'.join([product,days]))
histvol.columns = newcols

histvol.to_sql('vola', conn, if_exists='replace', index=False, index_label=['CR'])

conn.close()