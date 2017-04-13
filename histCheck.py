import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

files = glob.glob('run1_U8/emb*/200_100_10*/*/*pca*')
print files
print files[-1]

df = pd.DataFrame(pd.read_csv(files[-1], delimiter='\s+').T).reset_index()
df.columns = ['x','y','T']
# print df.head()
counter = 1
df = df[df['T'] >= 0.1]
# print len(df['T'].unique())
# print sorted(df['T'].unique())
df['x'] = [float(d[:6]) for d in df['x']]

plt.figure(figsize=(10,10))
for T in sorted(df['T'].unique()):
    tempDF = df[df['T'] == T]
    # print tempDF.head()
    plt.subplot(6,6,counter)
    plt.title(T)
    plt.hist2d(tempDF['x'],tempDF['y'])
    plt.xlim(0.25,.75)
    plt.ylim(0.25,.75)
    counter += 1
plt.show()
