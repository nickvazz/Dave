from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import random, argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-U','--U', help='Us to train (separated by commas)', required=True)
parser.add_argument('-R', '--Runs', help='# of runs', required=True)
parser.add_argument('-LR', '--LearningRate', help='', required=True)
parser.add_argument('-TE', '--TrainingEpochs', help='Number of training steps', required=True)
parser.add_argument('-BS', '--BatchSize', help='Size of batch to lessen memory strain', required=True)
parser.add_argument('-LT', '--LayerTrials', help='L1,L2,L3,L4', required=True)
parser.add_argument('-P', '--Plotting', help='Turning plotting on/off', required=True)
parser.add_argument('-WP','--WhichPlots', help='Which specific plots', required=True)
parser.add_argument('-ZT', '--ZoomTemps', help='Zoom in on temps = True/False', required=True)
parser.add_argument('-Tmin', '--TempMin', help='Min temp to load from data', required=True)
parser.add_argument('-Tmax', '--TempMax', help='Max temp to load from data', required=True)

args = vars(parser.parse_args())

U = args['U']
run_num = int(args['Runs'])
learning_rate = float(args['LearningRate'])
training_epochs = int(args['TrainingEpochs'])
batch_size = int(args['BatchSize'])
layer_trials = [map(int, args['LayerTrials'].split(','))]
tempRange = eval(args['ZoomTemps'])
tempMin = float(args['TempMin'])
tempMax = float(args['TempMax'])
plotOn = eval(args['Plotting'])
whichPlots = list(map(int, args['WhichPlots'].split(',')))
# print(whichPlots)

run_str = 'run' + str(run_num) + '_U' + str(U) + '/'
random.seed(10)
r = random.random()

for L1, L2, L3, L4 in layer_trials:
    try:
        for layer in [1,2,3,4]:
            trial = str(L1) + '_' + str(L2) + '_' + str(L3) + '_' + str(L4) + '_' +str(batch_size)+'_'+str(learning_rate)+'_'+str(int(training_epochs))
            embed_folder_path = run_str + 'embedded_data/' + trial + '/AfterL' + str(layer)

            if not os.path.exists(embed_folder_path):
                os.makedirs(embed_folder_path)

            newX = []
            newDataLabels = []
            text2 = np.loadtxt(run_str + 'reduced_data/' + trial + '/AfterL' + str(layer) + '_' + trial + '.txt')
            for i in range(len(text2)):
                newX.append(text2[i][0:100])
                newDataLabels.append(text2[i][-1])

            random.shuffle(newX, lambda:r)
            random.shuffle(newDataLabels, lambda:r)

            X = newX
            y = newDataLabels
            print("\nLayer = %s" % layer)
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
                plt.subplot(1,len(whichPlots),a)
                # plt.subplot(1,2,a) # no t-SNE
                plt.scatter(xs,ys, c=y, s=10, cmap=plt.cm.get_cmap("jet",50))
                plt.colorbar(ticks=range(0,2,1))
                plt.xticks([]), plt.yticks([])
                if title is not None:
                    plt.title(title)

            # plt.style.use('bmh')
            # plt.style.use('ggplot')
            # plt.style.use('dark_background')
            if plotOn == True:
                plt.figure(figsize=(30,10))
            # Projection on to the first 2 principal components
            counter = 1
            if 1 in whichPlots:
                print("Computing PCA projection")
                t0 = time()
                X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
                xs, ys = embedding(X_pca)
                np.savetxt(embed_folder_path + '/pca.txt', (xs,ys,y), fmt='%10.5f')
                if plotOn == True:
                    plotting(xs,ys,"PCA",counter,50)
                    counter += 1
                print ("(time %.2fs)" % (time() - t0))

            if 2 in whichPlots:
                # Random Trees embedding of the digits dataset
                print("Computing Totally Random Trees embedding")
                # hasher = ensemble.RandomTreesEmbedding(n_estimators=1000, random_state=0, max_depth=10, n_jobs=-1, verbose=3)
                hasher = ensemble.RandomTreesEmbedding(n_estimators=100, random_state=0, max_depth=10, n_jobs=-1) # regular
                # hasher = ensemble.RandomTreesEmbedding(n_estimators=1000, random_state=0, max_depth=2, n_jobs=-1, min_impurity_split=1e2)
                t0 = time()
                X_transformed = hasher.fit_transform(X)
                pca = decomposition.TruncatedSVD(n_components=2)
                X_reduced = pca.fit_transform(X_transformed)
                xs, ys = embedding(X_reduced)
                np.savetxt(embed_folder_path + '/randTrees.txt', (xs,ys,y), fmt='%10.5f')
                if plotOn == True:
                    plotting(xs,ys,"Random Forest",counter,50)
                    counter += 1
                print("(time %.2fs)" %(time() - t0))

            if 3 in whichPlots:
                # # t-SNE embedding of the digits dataset
                print("Computing t-SNE embedding")
                tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, n_iter=1000, verbose=4)
                t0 = time()
                X_tsne = tsne.fit_transform(X)
                xs, ys = embedding(X_tsne)
                np.savetxt(embed_folder_path + '/tsne.txt', (xs,ys,y), fmt='%10.5f')
                if plotOn == True:
                    plotting(xs,ys,"t-SNE",counter,50)
                    counter += 1
                print("(time %.2fs)" % (time() - t0))

            if 4 in whichPlots:
                # Spectral embedding of the digits dataset
                print("Computing Spectral embedding")
                embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
                t0 = time()
                X_se = embedder.fit_transform(X)
                xs, ys = embedding(X_se)
                np.savetxt(embed_folder_path + 'spectral.txt', (xs,ys,y), fmt='%10.5f')
                if plotOn == True:
                    plotting(xs,ys,"Spectral",counter,50)
                    counter += 1
                print("(time %.2fs)" % (time() - t0))


            if 5 in whichPlots:
                # # Isomap projection of the digits dataset
                print("Computing Isomap embedding")
                t0 = time()
                n_neighbors = 30
                X_iso = manifold.Isomap(n_neighbors,n_components=2, n_jobs=-1).fit_transform(X)
                xs, ys = embedding(X_iso)
                np.savetxt(embed_folder_path + '_isomap.txt', (xs,ys,y), fmt='%10.5f')
                if plotOn == True:
                    plotting(xs,ys,"Isomap",counter,50)
                    counter += 1
                print("(time %.2fs)" % (time() - t0))

            if 6 in whichPlots:
                # MDS  embedding of the digits dataset
                print("Computing MDS embedding")
                clf = manifold.MDS(n_components=2, n_init=1, max_iter=1000, n_jobs=-1, verbose=4)
                t0 = time()
                X_mds = clf.fit_transform(X)
                print("Stress: %f" % clf.stress_)
                xs, ys = embedding(X_mds)
                np.savetxt(embed_folder_path + '_mds.txt', (xs,ys,y), fmt='%10.5f')
                if plotOn == True:
                    plotting(xs,ys,"MDS",counter,50)
                    counter += 1
                print("(time %.2fs)" % (time() - t0))

            if plotOn == True:
                if not os.path.exists(run_str + 'layer_figures/' + trial):
                    os.makedirs(run_str + 'layer_figures/' + trial)
                plt.savefig(run_str + 'layer_figures/' + trial + '/layer' + str(layer) + '_' + str(len(X)) +'.png')
                plt.clf()
        # plt.clf()
    except:
        print('failed to find layer', L1,L2,L3,L4)
