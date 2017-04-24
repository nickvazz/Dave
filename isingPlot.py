import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
df = pd.DataFrame(pd.read_csv('isingMiddleData2D20170423202814.csv'))

# df['x'],df['y'] = np.log(df['x']), np.log(df['y'])
plt.scatter(df['x'],df['y'],s=5,c=df['T'])
# plt.savefig('scatter{}.png'.format(now))
print df['T'].value_counts()
plt.show()

xSplit = 1.6

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

topSide = df[df['y'] > 1.6]
middle = df[df['y'] < 1.6]
middle = middle[middle['y'] > 1.2]
botSide = df[df['y'] < 1.2]

topCount = topSide['T'].value_counts().values
topT = topSide['T'].value_counts().index.values

middleCount = middle['T'].value_counts().values
middleT = middle['T'].value_counts().index.values

botCount = botSide['T'].value_counts().values
botT = botSide['T'].value_counts().index.values

topCount = topCount/np.linalg.norm(topCount)
botCount = botCount/np.linalg.norm(botCount)
middleCount = middleCount/np.linalg.norm(middleCount)


top = np.asarray(sorted(zip(topT, topCount)))
middle = sorted(zip(middleT, middleCount))
bot = np.asarray(sorted(zip(botT, botCount)))

topT = top[:,0]
topCount = top[:,1]

botT = bot[:,0]
botCount = bot[:,1]

topBotCount = topCount+botCount
print topBotCount
plt.plot(botT, topBotCount, color='purple',label='top+bot')

plt.scatter(topT, topCount, color='r',label='top')
plt.scatter(middleT, middleCount, color='b',label='middle')
plt.scatter(botT, botCount, color='g',label='bot')
# plt.scatter(np.append(botT,topT), np.append(botCount,topCount),color='y',label='top&bot',s=10)
plt.legend()
plt.show()
