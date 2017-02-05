import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import glob

n_neighbors = 15

# import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features. We could
#                       # avoid this ugly slicing by using a two-dim dataset
# y = iris.target

L = 4
# data = glob.glob('U12/embedded_data/200_100_10_5_10_0.1_100/AfterL4/*')
data = glob.glob('embedded_data/*/AfterL' + str(L) + '/*')

data1 = np.loadtxt(data[0]); data1[2] = map(lambda x: x*1000, data1[2])
data2 = np.loadtxt(data[1]); data2[2] = map(lambda x: x*1000, data2[2])
data3 = np.loadtxt(data[2]); data3[2] = map(lambda x: x*1000, data3[2])
# datasets = [(zip(data1[0],data1[1]), data1[2]),(zip(data2[0],data2[1]),data2[2]),(zip(data3[0],data3[1]),data3[2])]
L = 100
datasets = [(zip(data1[0][:L],data1[1][:L]), data1[2][:L]),(zip(data2[0][:L],data2[1][:L]), data2[2][:L]),(zip(data3[0][:L],data3[1][:L]), data3[2][:L])]
X = datasets[0][0]
y = datasets[0][1]

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
