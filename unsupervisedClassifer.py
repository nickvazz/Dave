import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import glob
import os

if not os.path.exists("figure_classification/"):
    os.makedirs("figure_classification/")

tsne_files = glob.glob('/home/nick/Desktop/Ising Autoencoders Dec 19/embedded_data/*/*/*tsne.txt')
tsne_data = []
tsne_filenames = []
L = 1000
for file_ in sorted(tsne_files):
    tsne_new_file = np.loadtxt(file_, delimiter=' ')
    tsne_data_ = (np.asarray(zip(tsne_new_file[0][:L], tsne_new_file[1][:L])), np.asarray(tsne_new_file[2][:L]))
    tsne_data.append(tsne_data_)
    # print file_.split('/')
    tsne_filenames.append(file_.split('/')[-3] + '_' + file_.split('/')[-2])

pca_files = glob.glob('/home/nick/Desktop/Ising Autoencoders Dec 19/embedded_data/*/*/*pca.txt')
pca_data = []
pca_filenames = []
for file_ in sorted(pca_files):
    pca_new_file = np.loadtxt(file_, delimiter=' ')
    pca_data_ = (np.asarray(zip(pca_new_file[0][:L], pca_new_file[1][:L])), np.asarray(pca_new_file[2][:L]))
    pca_data.append(pca_data_)
    # print file_.split('/')
    pca_filenames.append(file_.split('/')[-3] + '_' + file_.split('/')[-2])

randTrees_files = glob.glob('/home/nick/Desktop/Ising Autoencoders Dec 19/embedded_data/*/*/*randTrees.txt')
randTrees_data = []
randTrees_filenames = []
for file_ in sorted(randTrees_files):
    randTrees_new_file = np.loadtxt(file_, delimiter=' ')
    randTrees_data_ = (np.asarray(zip(randTrees_new_file[0][:L], randTrees_new_file[1][:L])), np.asarray(randTrees_new_file[2][:L]))
    randTrees_data.append(randTrees_data_)
    # print file_.split('/')
    randTrees_filenames.append(file_.split('/')[-3] + '_' + file_.split('/')[-2])

datasets_total = zip(pca_data, randTrees_data, tsne_data)


np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
#
# n_samples = 1500
# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
#                                       noise=.05)
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

clustering_names = [
    'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'DBSCAN', 'Birch']

plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

# datasets = [noisy_circles, noisy_moons, blobs, no_structure]
datasets_total = datasets_total[1]
for j in range(len(datasets_total)):
    try:
        datasets = datasets_total[j]
        for i_dataset, dataset in enumerate(datasets):
            print i_dataset
            X, y = dataset
            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # create clustering estimators
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=2)
            ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                                   connectivity=connectivity)
            spectral = cluster.SpectralClustering(n_clusters=2,
                                                  eigen_solver='arpack',
                                                  affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=.2)
            affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                               preference=-200)

            average_linkage = cluster.AgglomerativeClustering(
                linkage="average", affinity="cityblock", n_clusters=2,
                connectivity=connectivity)

            birch = cluster.Birch(n_clusters=2)
            clustering_algorithms = [
                two_means, affinity_propagation, ms, spectral, ward, average_linkage,
                dbscan, birch]

            for name, algorithm in zip(clustering_names, clustering_algorithms):
                # predict cluster memberships
                t0 = time.time()
                algorithm.fit(X)
                t1 = time.time()
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
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                plot_num += 1

        plt.show()
    except:
        pass
