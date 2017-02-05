import glob, time, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import StandardScaler


U = 6
L = 3
which_data = 1 #they are in alphabetical order PCA/randTrees/tSNE

# files = glob.glob('U' + str(U) + '/embedded_data/*/AfterL' + str(L) + '/*')
files = glob.glob('embedded_data/*/AfterL' + str(L) + '/*')
data = map(np.loadtxt,files)
datasets = []
for sets in data:
    datasets.append([np.asarray(zip(sets[0],sets[1])), np.asarray(sets[2])])

print(files)
X,y = datasets[which_data] #they are in alphabetical order PCA/randTrees/tSNE
# X = StandardScaler().fit_transform(X)
Len = len(X)
# Len = 10000
X = StandardScaler().fit_transform(X[:Len])
y = y[:Len]

spectral = cluster.SpectralClustering(n_clusters=2,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")



t0 = time.time()
spectral.fit(X)
t1 = time.time()
print(t1-t0)
y_pred = spectral.labels_.astype(np.int)
f, axarr = plt.subplots(3)

axarr[0].scatter(X[:,0],X[:,1], c=y)
axarr[0].set_title('color by temp')

axarr[1].scatter(X[:,0],X[:,1], c=y_pred)
axarr[1].set_title('color by clustering alg')

clust0 = []; clust1 = []
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        clust0.append(y[i])
    else:
        clust1.append(y[i])
# plt.figure()
# plt.title('cluster classification by temp')
n0, bins0, patches0 = plt.hist(clust0, 30, alpha=0.50, range=[.1,.4])
n1, bins1, patches1 = plt.hist(clust1, 30, alpha=0.50, range=[.1,.4])
# plt.xlabel('temp')

# print n0, '\n', bins0, '\n'
# print n1, '\n', bins1, '\n'
# print '\n'

# if bins1[0] > bins0[0]:
#     for item in zip(n0, bins0): print item
#     for item in zip(n1, bins1): print item
# else:
#     for item in zip(n1, bins1): print item
#     for item in zip(n0, bins0): print item

# print sorted(list(set(y)))

# axarr[2].hist(clust0, 10, alpha=0.50, normed=1)
# axarr[2].hist(clust1, 10, alpha=0.50, normed=1)
axarr[2].set_title('cluster classification by temp')

if not os.path.exists('histCluster_figures/'):
    os.makedirs('histCluster_figures')

plt.savefig('histCluster_figures/histCluster_U' + str(U) + 'L' + str(L) + '.png')
plt.show()
