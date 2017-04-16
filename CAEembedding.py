import pandas as pd
from sklearn import manifold, ensemble

df = pd.DataFrame(pd.read_csv('dataCAE.csv'))[['x','y','T']]

print df.head()

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets, decomposition

# n_points = 1000
# X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
X, color = zip(df.x,df.y), df['T']
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(251)
ax.scatter(df.x, df.y, c=color, cmap=plt.cm.get_cmap("jet",50), s=10)
    # ax.view_init(4, -72)

methods = ['standard', 'ltsa', 'hessian', 'modified']
methods = ['standard', 'ltsa', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                        eigen_solver='auto',
                                        method=method).fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(252 + i)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.get_cmap("jet",50),s=10)
    plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

ax = fig.add_subplot(255)
hasher = ensemble.RandomTreesEmbedding(n_estimators=1000, random_state=0, max_depth=10, n_jobs=-1) # regular
# hasher = ensemble.RandomTreesEmbedding(n_estimators=1000, random_state=0, max_depth=2, n_jobs=-1, min_impurity_split=1e2)
t0 = time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=2)
Y = pca.fit_transform(X_transformed)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.get_cmap("jet",50), s=10)
plt.title("Random Trees (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
Y = decomposition.PCA(n_components=2, random_state=42).fit_transform(X)
t1 = time()
ax = fig.add_subplot(256)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.get_cmap("jet",50), s=10)
plt.title("PCA (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.get_cmap("jet",50), s=10)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.get_cmap("jet",50), s=10)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.get_cmap("jet",50), s=10)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(2, 5, 10)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.get_cmap("jet",50),s=10)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()
