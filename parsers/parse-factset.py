import pandas as pd, numpy as np
import os
import sqlite3

db = 'C:\\Datasets\\thesis.db'
conn = sqlite3.connect(db)
c = conn.cursor()

directory = ['..','factset']
tables = {}

for file in os.listdir('//'.join(directory)):
    if '.xls' in file:
        file_path = '\\'.join(directory+[file])
        tables[file[:file.find('.')]] = pd.read_excel(file_path, skiprows=7)
print(list(tables.keys()))

for table_name in reversed(list(tables.keys())):
    print(table_name)
    if 'Key-Developments' in table_name:
        table = tables[table_name]
        drop_phrase = ' dropped from '
        add_phrase = ' added to '
        table['Index'] = table['Key Development Headline'].apply(lambda headline: 
                                                                 headline[headline.find(drop_phrase)-len(headline)+len(drop_phrase):] 
                                                                 if drop_phrase in headline 
                                                                 else headline[headline.find(add_phrase)-len(headline)+len(add_phrase):])
        table['Action'] = table['Action'].apply(lambda action: 'Delete' if 'Drop' in action else 'Add')
        table['Ticker_Pair'] = table['Company Name'].apply(lambda name: name[name.rfind('('):][1:-1])
        table['Ticker'] = table['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[:ticker_pair.find(':')])
        table['Exchange'] = table['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[ticker_pair.find(':')+1:])
        table['Name'] = table['Company Name'].apply(lambda name: name[:name.rfind('(')])
        table = table[['Key Developments By Date','Action','Index','Ticker','Exchange','Name']]
        table.columns = ['date','action','index','exchange','ticker','name']
        #tables[table_name]=table
        print(table.columns)
        print(table.head())
        table.to_sql('factset_index_changes', conn, if_exists='replace', index=False)
    elif 'Holdings' in table_name:
        table = tables[table_name]
        
        table['Ticker_Pair'] = table['Ticker']
        table['Exchange'] = table['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[:ticker_pair.find(':')])
        table['Ticker'] = table['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[ticker_pair.find(':')+1:])
        table = table[['Ticker','Exchange','Index Constituents']]
        
        split_indices = []
        for i, row in table.iterrows():
            indices = row['Index Constituents'].split('; ')
            if len(indices)>1:
                for index in indices:
                    working_row = row.copy(deep=True)
                    working_row['Index Constituents'] = index
                    split_indices.append(pd.DataFrame(working_row).T)
            else:
                split_indices.append(pd.DataFrame(row).T)
        table = pd.concat(split_indices)
        table = table.reset_index().drop(['index'],axis=1)
        table.columns = ['ticker','exchange','index']
        print(table.columns)
        print(table.head())
        #tables[table_name]=table
        table.to_sql('index_holdings_2019', conn, if_exists='replace', index=False)
    elif 'Sector' in table_name:
        table = tables[table_name]
        table['Ticker_Pair'] = table['Ticker']
        table['Exchange'] = table['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[:ticker_pair.find(':')]
                                                      if type(ticker_pair)==type('  ') else ticker_pair)
        table['Ticker'] = table['Ticker_Pair'].apply(lambda ticker_pair: ticker_pair[ticker_pair.find(':')+1:]
                                                      if type(ticker_pair)==type('  ') else ticker_pair)
        table.columns = ['ticker','share_type','name','industries','sector','comps','supls','industry','corp_type','exchange_name','ticker_pair','exchange']
        #tables[table_name]=table
        print(table.columns)
        print(table.head())
        table.to_sql('sectors', conn, if_exists='replace', index=False)
    else:
        print(' > Unhandled table',table_name)