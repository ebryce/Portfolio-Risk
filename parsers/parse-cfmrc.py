import pandas as pd
import sqlite3
import sys

# Import data from the CFMRC CHASS datasets into my local Sqlite database

print('Setting up variables')
db = 'C:\\Datasets\\thesis.db'
file = 'C:\\Datasets\\cfmrc_annual.txt'

n_before_query = 7500
n_before_commit = 25000
n_before_read = 34

query_base = '''INSERT INTO cfmrc VALUES'''
query_teminator = ''' ;'''
query_part_template = '''({VALUES})'''

# Init the query for the first go
query_parts = []
query = query_base

print('Connecting to %s' % db)
conn = sqlite3.connect(db)
c = conn.cursor()

inserted_tickers = set(pd.read_sql('''SELECT DISTINCT ticker FROM cfmrc''',conn)['ticker'].values)
print('%d tickers already in table' % len(inserted_tickers))

with open(file, "r") as f:
    print('Open file %s' % file)
    row = 0
    for l in f:
        if row > n_before_read:
            # Because the first n rows are junk

            # Deal with text formatting
            l = l.split('"')[1:44:2]

            if l[1] not in inserted_tickers:

                # The cells we actually want to push over
                cells_to_insert = [0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                
                # The cells we want to convert to strings
                cells_to_stringify = [0,1]
                
                # Pulls old cells into the new query
                insertions = []
                for i in cells_to_insert:
                    insertions.append(l[i])
                
                # Captures strings in quotes
                for i in cells_to_stringify:
                    insertions[i] = '"{0}"'.format(insertions[i])
                    
                # If there are empty cells
                for i in range(len(cells_to_insert)):
                    if len(insertions[i])==0:
                        insertions[i]='0'
                
                # Builds this row of the query
                query_part = query_part_template.format(VALUES=','.join(insertions))
                query_parts.append(query_part)
                
            if len(query_parts)>0:
                # Every so often, push the changes to the db
                if row % n_before_query==0:
                    try:
                        # Combine the query parts
                        query += ','.join(query_parts)

                        # Terminate the query
                        query += query_teminator

                        # Runt the query
                        c.execute(query)

                        # Reset the query for next time
                        query_parts = []
                        query = query_base

                        print(' > Queried on row %d' % row)
                    except:
                        print(query)

                # Commit every n rows
                if row % n_before_commit == 0:
                    print('Committed on row %d' % row)
                    conn.commit()
        else:
            if row % n_before_commit == 0:
                print('Nothing to do on row %d' % row)

        row+=1

conn.close()