import requests, pandas as pd, numpy as np

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

def tgam_financials(ticker, quarterly=False):
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