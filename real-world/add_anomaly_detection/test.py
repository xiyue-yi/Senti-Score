from anom_detect import anom_detect
import pandas as pd

df = pd.read_csv('score.txt',sep='\t',header=None)
df.head()

df.index = df[0].tolist()
df.drop(df.columns[0], axis=1, inplace=True)
df.index.name = 'time'
df.columns = ['sunspots']
df.head()
df.dtypes

outliers_index = ['05-13', '05-20', '05-06', '05-23', '05-11', '05-18', '05-21', '04-01', '04-21', '04-02', '04-29', '05-12', '05-01', '05-14', '04-24', '05-19']
anoma_points = pd.DataFrame(df[['sunspots']].loc['05-13'])
print(anoma_points)
#an = anom_detect()
#an.evaluate(df)
