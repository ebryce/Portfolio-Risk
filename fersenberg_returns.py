
print('Loading dataframes')
sector = hdf.get(key='sector_returns')[[('sector',''),('r_overnight','mean'),('r_intraday','mean'),('vol','count')]]
#sector.columns = ['_'.join(column) for column in sector.columns[:-1]].extend(''.join([sector.columns[-1]]))
sector_returns.to_sql('sector_returns', conn, if_exists='replace', index=False)
print(sector.head())
industry = hdf.get(key='industry_returns')[[('industry',''),('r_overnight','mean'),('r_intraday','mean'),('vol','count')]]
#industry.columns = ['_'.join(column) for column in industry.columns[:-1]].extend(''.join([industry.columns[-1]]))
industry.to_sql('industry_returns', conn, if_exists='replace', index=False)
print(industry.head())
print('Obtaining sector mappings')
mapped = pd.read_sql(mapping_query, conn)
mapped.columns = [('sector',),('industry',)]
print(mapped.head())

print('Merging files')
mapped = industry.merge(mapped, on=[('industry',)], how='left')
mapped.index = industry.index
#print(mapped.head())
mapped = sector.merge(mapped, on=[('sector',),('date')], how='left', suffixes=('_sector','_industry'))
mapped = mapped.reset_index()
mapped.drop([('sector',''),('industry','')], axis=1, inplace=True)

#print(mapped.head())
mapped.columns = ['date','sector',
	'r_overnight_sector','r_intraday_sector','count_sector',
	'industry','r_overnight_industry','r_intraday_industry','count_industry']
mapped['r_intraday_industry'] = mapped['r_intraday_industry']-mapped['r_intraday_sector']
mapped['r_overnight_industry'] = mapped['r_overnight_industry']-mapped['r_overnight_sector']
mapped = mapped[['date','sector','industry',
	'r_intraday_industry','r_intraday_sector','r_overnight_industry','r_overnight_sector',
	'count_industry','count_sector']]

print(mapped.head())