import numpy as np
import glob, os
import matplotlib.pyplot as plt
from collections import Counter
import time
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

U = 12
L = 4

length = 1000

# tsne_files = glob.glob('U' + str(U) + '/embedded_data/*/AfterL' + str(L) + '/tsne.txt')
tsne_files = glob.glob('embedded_data/*/AfterL' + str(L) + '/tsne.txt')
data1 = np.loadtxt(tsne_files[0][:length])
X1 = np.asarray(zip(data1[0],data1[1]))
Y1 = np.asarray(data1[2])

# pca_files = glob.glob('U' + str(U) + '/embedded_data/*/AfterL'+ str(L) +'/pca.txt')
pca_files = glob.glob('embedded_data/*/AfterL' + str(L) + '/pca.txt')
data2 = np.loadtxt(pca_files[0][:length])
X2 = np.asarray(zip(data2[0],data2[1]))
Y2 = np.asarray(data2[2])

# randTrees_files = glob.glob('U' + str(U) + '/embedded_data/*/AfterL' + str(L) + '/randTrees.txt')
randTrees_files = glob.glob('embedded_data/*/AfterL' + str(L) + '/randTrees.txt')
data3 = np.loadtxt(randTrees_files[0][:length])
X3 = np.asarray(zip(data3[0],data3[1]))
Y3 = np.asarray(data3[2])

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

clustering_names = [
    # 'MiniBatchKMeans',
    # 'AffinityPropagation',
    # 'MeanShift',
    # 'SpectralClustering',
    # 'Ward',
    # 'AgglomerativeClustering',
    # 'DBSCAN',
    'Birch'
    ]

plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

datasets = [[X1,Y1],[X2,Y2], [X3,Y3]]
datasets = [[X3,Y3]]

# datasets = [noisy_circles, noisy_moons, blobs, my_data]# no_structure]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    print '1'
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    print '2'
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    print '3'
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    print '4'
    # create clustering estimators
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    print '5'
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    print '6'
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           connectivity=connectivity)
    print '7'
    spectral = cluster.SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    print '8'
    dbscan = cluster.DBSCAN(eps=.2)
    print '9'
    affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)
    print '10'
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=2,
        connectivity=connectivity)
    print '11'
    birch = cluster.Birch(n_clusters=2)
    print '12'
    clustering_algorithms = [
        two_means, affinity_propagation, ms, spectral, ward, average_linkage,
        dbscan, birch]

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        print t0
        algorithm.fit(X)
        t1 = time.time()
        print t1
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # plot
        plt.subplot(4, len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
        print plot_num

if not os.path.exists('classifications_figures/'):
    os.makedirs('classifications_figures/')
plt.savefig('classifications_figures/classification_U' + str(U) + 'L' + str(L) + '.png')
plt.show()
