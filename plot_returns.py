aggregated = df[df['d'].between(-365, 365, inclusive=True) & df['flag'].isin([-1,1]) & (~df['alpha_to_rebal'].isna())].groupby(by='d').agg({'alpha_to_rebal_side-adj':['mean','std'],
                                                                                                                                           'return_to_rebal_side-adj':['mean','std']}).reset_index()
print(aggregated.head())

# Plot it

plt.style.use('classic')

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('white')

ax = fig.add_subplot(1, 1, 1)

ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.grid(True,axis='both',linestyle=':')

lines = {'alpha_to_rebal_side-adj':'xkcd:forest green',
        'return_to_rebal_side-adj':'xkcd:pale orange'}

for line in lines:
    ax.plot(aggregated['d'],aggregated[line]['mean'], color=lines[line])
    ax.fill_between(aggregated['d'], aggregated[line]['mean']-aggregated[line]['std'], aggregated[line]['mean']+aggregated[line]['std'], color=lines[line], alpha=0.05)

# Plot the ranking period
ax.fill_between([-(length_of_ranking_period+avg_days),-length_of_ranking_period], [1,1], color='xkcd:grey', alpha=0.2)
ax.fill_between([-(length_of_ranking_period+avg_days),-length_of_ranking_period], [-1,-1], color='xkcd:grey', alpha=0.2)

ax.legend([line.replace('_',' ') for line in lines.keys()], frameon=False, loc='best')

plt.title('Portfolio returns')
plt.ylabel('Culmulative returns')
plt.xlabel('Days Since (To) Rebalance')

xlim = [-365,365]

plt.xlim(time_range)
plt.ylim(-0.5,0.5)
plt.show()