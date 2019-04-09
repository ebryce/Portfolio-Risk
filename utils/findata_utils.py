import pandas as pd,  numpy as np
import requests
import boto3

# To get data from Economagic
from bs4 import BeautifulSoup

# To get zip files from French
import zipfile
from zipfile import ZipFile
import io

def sync_to_db(df, table, db='C:\\Datasets\\thesis.db'):
    return None
    
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

def list_files(bucket='workspace-grapevine-andreas', subdir=None):
    s3 = boto3.client('s3')
    objects = s3.list_objects(Bucket=bucket)
    files = [file['Key'] for file in objects['Contents']]
    
    if type(subdir)==type(None):
        return files
    else:
        files_filtered = []
        for file in files:
            if file[:len(subdir)]==subdir:
                files_filtered.append(file) 
        return files_filtered

def from_grapevine(filename, bucket='workspace-grapevine-andreas', hdf_key=None):
    # Reads a dataframe from S3
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=filename)
    body = io.BytesIO(obj['Body'].read())
    
    ext = filename[filename.rfind('.')+1:]
    
    if ext =='csv':
        df = pd.read_csv(body)
    elif ext=='h5':
        df = pd.read_hdf(body, key=hdf_key)
    elif ext=='pkl':
        df = pd.read_pickle(body) 
    else:
        raise ValueError('Invalid extension {ext}'.format(ext=ext))
        
    return df

def to_grapevine(df, filename, bucket='workspace-grapevine-andreas', hdf_key='Default',print_index=False):
    # Writes a dataframe to Grapevine
    ext = filename[filename.rfind('.')+1:]
    if ext=='pkl':
        buf = io.BytesIO()
    else:
        buf = io.StringIO()
    
    if ext =='csv':
        df = df.to_csv(buf,index=print_index)
    elif ext=='h5':
        df = df.to_hdf(buf, key=hdf_key)
    elif ext=='pkl':
        df = df.to_pickle(buf)
    else:
        raise ValueError('Invalid extension {ext}'.format(ext=ext))
        
    s3_resource = boto3.resource('s3')
    resp = s3_resource.Object(bucket, filename).put(Body=buf.getvalue())

    if resp['ResponseMetadata']['HTTPStatusCode']!=200:
        raise ValueError(resp)

#df = from_grapevine(filename='user/files/test2.csv')
def get_tgam(ticker, quarterly=False):
    ticker = ticker.replace('.','-')+'-T'
    
    table = []
    for statement in ['incomeStatement','cashFlow','balanceSheet']:
        url = 'https://globeandmail.pl.barchart.com/module/financials/{Statement}.html'.format(Statement=statement)
        
        period = '3m' if quarterly else '12m'
        
        r = requests.post(url, data={"period":period,"symbol":ticker,"showPeriodTabs":0})
        table.append(tgam_html(r))
        
    table = pd.concat(table).T.reset_index()
    table['period'] = 'Q' if quarterly else 'A'
    table['ticker'] = ticker[:ticker.find('-')]
    return table

def tgam_html(response):
    html = str(response.content)[2:]
    table = []

    dates = html.split('<tr>')[1].split('<th')[1:]
    dates = [date[date.find('>')+1:] for date in dates]
    dates = [date[:date.find('<')] for date in dates]
    dates = dates[1:]
    dates = [datetime.strptime(date,'%m-%Y') for date in dates]

    rows = html.split('<tr')[2:]
    rows_temp = []
    n = None

    statement = []
    header = None
    
    for row in rows:

        cells = row.split('<td')

        cells = [cell[cell.find('>')+1:] for cell in cells]
        cells = [cell[:cell.find('<')] for cell in cells]

        if cells[0]=='':
            cells = cells[1:]

        statement_i = pd.Series(name=cells[0].replace('&#039;','').replace('&amp;','and'))
        cells = cells[1:]

        # Fix numbers
        cells = [cell.replace('M','000') for cell in cells]
        cells = [cell.replace(' ','') for cell in cells]
        cells = [cell.replace(',','') for cell in cells]
        cells = [cell.replace('&nbsp;','N/A') for cell in cells]
        cells = [np.nan if cell=='N/A' else float(cell) for cell in cells]

        for i in range(len(cells)):
            statement_i[dates[i]]=cells[i]
            
        if (statement_i.name=='&nbsp;'):
            pass
        elif ((pd.DataFrame(statement_i).isna().sum()/pd.DataFrame(statement_i).isna().count()<1).iloc[0]):
            if statement_i.name=='Total':
                statement_i.name = ' '.join([statement_i.name, header])
            statement.append(pd.DataFrame(statement_i).T)
        else:
            header = statement_i.name
            pass
        
    statement = pd.concat(statement)
    return statement