import pandas as pd
import requests

# To get data from Economagic
from bs4 import BeautifulSoup

# To get zip files from French
import zipfile
from zipfile import ZipFile
import io
    
def get_damodaram(key):
    # Programatically download Damodaran's data
    url = 'http://www.stern.nyu.edu/~adamodar/pc/datasets/{KEY}.xls'.format(KEY=key)
    df = pd.read_excel(url, skiprows=7, header=0)#, index_col=0)
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    return df

def get_economagic(src, key, date=None):
    # Programatically download economagic data
    url = 'http://www.economagic.com/em-cgi/data.exe/{SRC}/{SET}'.format(SRC=src,SET=key)
    
    if date!=None: url+='@@VPD{DATE}'.format(DATE=date)
        
    soup = BeautifulSoup(requests.get(url).content, 'lxml')    
    data_soup = soup.pre.text
    data_rows = data_soup[data_soup.find('Go to end of data set')+len('Go to end of data set'):].split('\r\n')[1:-1]

    header = soup.findAll('table')[1].font.b.text.split('\n')[1]

    table = []
    for data_row in data_rows:
        try:
            # replace double spaces
            while '  ' in data_row:
                data_row = data_row.replace('  ',' ')
            data_row = data_row.replace('·','.')

            cells = data_row.split(' ')[1:]
            try:
                index = int(cells[0]+cells[1])
            except:
                print('%s-%s is not a date'% (cells[0], cells[1]))
                continue
            series = pd.Series(name=index)
            try:
                series[header] = float(cells[2])
            except:
                print('%s is not a float' % cells[2])
                continue
            table.append(pd.DataFrame(series).T.reset_index())
        except:
            print('Had to skip a row: %s' % data_row)

    df = pd.concat(table)
    return df

def get_french(key):
    # Programmatically download Ken French's data
    url = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{KEY}_CSV.zip'.format(KEY=key)
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_file = z.open('{KEY}.csv'.format(KEY=key))
    df = pd.read_csv(csv_file, skiprows=6, header=0)
    df.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
    df = df.sort_values(by='Date').reset_index().drop(0)
    df.drop('index', axis=1, inplace=True)

    return df