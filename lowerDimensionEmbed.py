from time import time
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from StringIO import StringIO
import random

batch_size = 10
learning_rate = 0.1
training_epochs = 1*1E2

random.seed(10)
r = random.random()

layer_trials = [[200,100,10,5]]

for L1, L2, L3, L4 in layer_trials:
    try:
        for layer in [1,2,3,4]:
            trial = str(L1) + '_' + str(L2) + '_' + str(L3) + '_' + str(L4) + '_' +str(batch_size)+'_'+str(learning_rate)+'_'+str(int(training_epochs))
            embed_folder_path = 'embedded_data/' + trial + '/AfterL' + str(layer)

            if not os.path.exists(embed_folder_path):
                os.makedirs(embed_folder_path)

            newX = []
            newDataLabels = []
            text2 = np.loadtxt('reduced_data/' + trial + '/AfterL' + str(layer) + '_' + trial + '.txt')
            for i in range(len(text2)):
                newX.append(text2[i][0:100])
                newDataLabels.append(text2[i][-1])

            random.shuffle(newX, lambda:r)
            random.shuffle(newDataLabels, lambda:r)

            X = newX
            y = newDataLabels
            print "\nLayer = %s" % layer
            #----------------------------------------------------------------------
            # Scale and visualize the embedding vectors
            def embedding(X):
                x_min, x_max = np.min(X, 0), np.max(X, 0)
                X = (X - x_min) / (x_max - x_min)
                xs = []; ys = []
                for i in range(len(X)):
                    xs.append(X[i][0])
                    ys.append(X[i][1])
                return (xs, ys)

            def plotting(xs, ys, title, a, b):
                # plt.subplot(1,3,a)
                plt.subplot(1,2,a) # no t-SNE
                plt.scatter(xs,ys, c=y, cmap=plt.cm.get_cmap("plasma",50))
                plt.colorbar(ticks=range(0,2,1))
                plt.xticks([]), plt.yticks([])
                if title is not None:
                    plt.title(title)

            # plt.style.use('bmh')
            # plt.style.use('ggplot')
            # plt.style.use('dark_background')
            plt.figure(figsize=(30,10))

            # Projection on to the first 2 principal components
            print("Computing PCA projection")
            t0 = time()
            X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
            xs, ys = embedding(X_pca)
            np.savetxt(embed_folder_path + '/pca.txt', (xs,ys,y), fmt='%10.5f')
            # plotting(xs,ys,"PCA",1,4) # with hist
            plotting(xs,ys,"PCA",1,50) # no hist
            print ("(time %.2fs)" % (time() - t0))

            #
            # Random Trees embedding of the digits dataset
            print("Computing Totally Random Trees embedding")
            # hasher = ensemble.RandomTreesEmbedding(n_estimators=1000, random_state=0, max_depth=10, n_jobs=-1, verbose=3)
            hasher = ensemble.RandomTreesEmbedding(n_estimators=1000, random_state=0, max_depth=10, n_jobs=-1)
            t0 = time()
            X_transformed = hasher.fit_transform(X)
            pca = decomposition.TruncatedSVD(n_components=2)
            X_reduced = pca.fit_transform(X_transformed)
            xs, ys = embedding(X_reduced)
            np.savetxt(embed_folder_path + '/randTrees.txt', (xs,ys,y), fmt='%10.5f')
            # plotting(xs,ys,"Random Forest",7,10) # with hist
            plotting(xs,ys,"Random Forest",2,50) # no hist
            print("(time %.2fs)" %(time() - t0))

            # # Spectral embedding of the digits dataset
            # print("Computing Spectral embedding")
            # embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
            # t0 = time()
            # X_se = embedder.fit_transform(X)
            # xs, ys = embedding(X_se)
            # np.savetxt(embed_folder_path + 'spectral.txt', (xs,ys,y), fmt='%10.5f')
            # # plotting(xs,ys,"Spectral",8,11) # with hist
            # plotting(xs,ys,"Spectral",5,50) # no hist
            # print("(time %.2fs)" % (time() - t0))

            # # t-SNE embedding of the digits dataset
            # print("Computing t-SNE embedding")
            # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, n_iter=1000, verbose=4)
            # t0 = time()
            # X_tsne = tsne.fit_transform(X)
            # xs, ys = embedding(X_tsne)
            # np.savetxt(embed_folder_path + '/tsne.txt', (xs,ys,y), fmt='%10.5f')
            # # plotting(xs,ys,"t-SNE",9,12) # with hist
            # plotting(xs,ys,"t-SNE",3,50) # no hist
            # print("(time %.2fs)" % (time() - t0))

            # # # Isomap projection of the digits dataset
            # print("Computing Isomap embedding")
            # t0 = time()
            # n_neighbors = 30
            # X_iso = manifold.Isomap(n_neighbors,n_components=2, n_jobs=-1).fit_transform(X)
            # xs, ys = embedding(X_iso)
            # np.savetxt(embed_folder_path + '_isomap.txt', (xs,ys,y), fmt='%10.5f')
            # # plotting(xs,ys,"Isomap",2,5) # with hist
            # plotting(xs,ys,"Isomap",4,50) # no hist
            # print("(time %.2fs)" % (time() - t0))

            # # MDS  embedding of the digits dataset
            # print("Computing MDS embedding")
            # clf = manifold.MDS(n_components=2, n_init=1, max_iter=1000, n_jobs=-1, verbose=4)
            # t0 = time()
            # X_mds = clf.fit_transform(X)
            # print("Stress: %f" % clf.stress_)
            # xs, ys = embedding(X_mds)
            # np.savetxt(embed_folder_path + '_mds.txt', (xs,ys,y), fmt='%10.5f')
            # plotting(xs,ys,"MDS",6,50) # no hist
            # # plotting(xs,ys,"MDS",3,6) # with hist
            # print("(time %.2fs)" % (time() - t0))

            if not os.path.exists('layer_figures/' + trial):
                os.makedirs('layer_figures/' + trial)
            plt.savefig('layer_figures/' + trial + '/layer' + str(layer) + '_' + str(len(X)) +'.png')
            plt.clf()

    except:
        print 'failed to find layer', L1,L2,L3,L4
# os.system('python classificationEmbed.py')
os.system("python fanFitting.py")
