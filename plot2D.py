import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from collections import Counter

plt.plot([4,5,6,8,9,10,12,14,16,20],[.27,.26,.25,.23,.25,.29,.27,.2,.2,.2], label='new')

U = [4, 5, 6, 8, 9, 10, 12, 14, 16, 20]

khatami = [.16,.22,.30,.34,.35,.33,.30,.26,.23,.19]

plt.plot(U,khatami,label='khatami paper')
plt.xlim(0,22)
plt.ylim(.05,.4)
plt.xlabel('U')
plt.ylabel('T')
plt.legend()
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U20/embedded_data/30_25_10_2_10_0.01_100/AfterL1/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U20')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U16/embedded_data/30_25_10_2_10_0.01_100/AfterL1/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U16')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()


df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U14/embedded_data/30_25_10_2_10_0.01_100/AfterL1/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
plt.subplot(211)
plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
plt.subplot(212)
plt.scatter(df.x,df.y,c=y_pred)
plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U14')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U12/embedded_data/30_25_10_2_10_0.01_100/AfterL3/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U12')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U10/embedded_data/30_25_10_2_10_0.01_100/AfterL2/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U10')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()


df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U9/embedded_data/30_25_10_2_10_0.01_100/AfterL1/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)

two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U9')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U8/embedded_data/30_25_10_2_10_0.01_100/AfterL1/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U8')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U6/embedded_data/30_25_10_2_10_0.01_100/AfterL1/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U6')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U5/embedded_data/30_25_10_2_10_0.01_100/AfterL3/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U5')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()

df = pd.DataFrame(pd.read_csv('/home/nick/Desktop/Research/Dave/run1_U4/embedded_data/30_25_10_2_10_0.01_100/AfterL1/randTrees.txt', sep='\s+', header=None).T)
df.columns = ['x','y','T']
# print df.head()
df['T'] = [round(t,3) for t in df['T']]
# plt.subplot(211)
# plt.scatter(df.x, df.y, c=df['T'],s=10)



two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42).fit(zip(df.x,df.y))
y_pred = two_means.predict(zip(df.x,df.y))
# print y_pred
# plt.subplot(212)
# plt.scatter(df.x,df.y,c=y_pred)
# plt.show()


df['pred'] = y_pred
# print df.head()
dfPred0 = df[df['pred']==0].sort(columns=['T']).reset_index()
dfPred1 = df[df['pred']==1]
# print dfPred0.head()
a = dfPred0['T'].value_counts()
b = dfPred1['T'].value_counts()
# print a.index.values
# print a.values
plt.title('U4')
plt.scatter(a.index.values, a.values, c='r')
plt.scatter(b.index.values, b.values, c='b')
plt.show()
