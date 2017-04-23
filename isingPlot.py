import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
df = pd.DataFrame(pd.read_csv('isingMiddleData20170422045105.csv'))

df['x'],df['y'] = np.log(df['x']), np.log(df['y'])
plt.scatter(df['x'],df['y'],s=5,c=df['T'])
plt.savefig('scatter{}.png'.format(now))
print df['T'].value_counts()
plt.show()

xSplit = -15.33
rightSide = df[df['x'] > xSplit]
leftSide = df[df['x'] < xSplit]

rightCount = rightSide['T'].value_counts().values
rightT = rightSide['T'].value_counts().index.values

leftCount = leftSide['T'].value_counts().values
leftT = leftSide['T'].value_counts().index.values

leftCount = leftCount/np.linalg.norm(leftCount)
rightCount = rightCount/np.linalg.norm(rightCount)

plt.title('divided clusters by right or left of X = 0.006')
plt.ylabel('normalized num of points in cluster')
plt.xlabel('Temp')
plt.scatter(rightT,rightCount,color='r',label='right side cluster')
plt.scatter(leftT,leftCount,color='b', label='left side cluster')
plt.legend()
# plt.savefig('clusterNormalization{}.png'.format(now))
plt.show()
