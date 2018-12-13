import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load League of Legends dataset
df = pd.read_csv('../dataset/games.csv')

# look up first 5 lines
print(df.head(5))

# check the dataset overview
#print(df.info())

# print the characteristics of each informations
print(df.describe())

# print columns of first Blood and class (in this case 'winner') only
#print(df[['firstBlood', 'winner']])

# make a new group that 'firstInhibitor' information-based, calculate the averarge and then sorting by average in ascending order
print(df[['firstInhibitor', 'winner']].groupby(['firstInhibitor'], as_index=False).mean().sort_values(by='firstInhibitor', ascending=True))

# draw a graph which shows the relation of parameters
# decide the attribute of the graph. Set the vmax to 0.5 which enables to display brighter if the value is closer to 0.5
sns.heatmap(df.corr(), vmin=-1, vmax=1, linecolor='white', linewidths=0.005)
plt.show()

# draw a graph that shows relation between class and some parameter
grid = sns.FacetGrid(df, col='winner')
grid.map(plt.hist, 't2_ban1',  bins=100)
plt.show()

'''
print(df.head())

sns.pairplot(df, hue='winner')
plt.show()

pearson correlation coefficient
'''