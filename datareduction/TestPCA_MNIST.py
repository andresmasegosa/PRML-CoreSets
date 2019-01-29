import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

from datareduction.bayesian_pca_DR import BayesianPCA_DR
from prml.feature_extractions import BayesianPCA, PCA
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

data = mnist.train.images
target = mnist.train.labels
x_train = data[np.random.choice(np.where(target == 3)[0], 1000)]
D=data.shape[1]

data = mnist.test.images
target = mnist.test.labels
x_test = data[np.random.choice(np.where(target == 3)[0], 1000)]

mnist3 = x_train

pca = BayesianPCA(n_components=4)
pca.fit(mnist3, initial="eigen")
plt.figure(0)
plt.subplot(1, 5, 1)
plt.imshow(pca.mean.reshape(28, 28))
plt.axis('off')
for i, w in enumerate(pca.W.T[::-1]):
    plt.subplot(1, 5, i + 2)
    plt.imshow(w.reshape(28, 28))
    plt.axis('off')
#plt.show()

pca = BayesianPCA_DR(n_components=4)
pca.fit(mnist3, initial="eigen", n_clusters=100, cluster_method="SS")
plt.figure(1)
plt.subplot(1, 5, 1)
plt.imshow(pca.mean.reshape(28, 28))
plt.axis('off')
for i, w in enumerate(pca.W.T[::-1]):
    plt.subplot(1, 5, i + 2)
    plt.imshow(w.reshape(28, 28))
    plt.axis('off')

pca = BayesianPCA_DR(n_components=4)
pca.fit(mnist3, initial="eigen", n_clusters=10, cluster_method="NoSS")
plt.figure(2)
plt.subplot(1, 5, 1)
plt.imshow(pca.mean.reshape(28, 28))
plt.axis('off')
for i, w in enumerate(pca.W.T[::-1]):
    plt.subplot(1, 5, i + 2)
    plt.imshow(w.reshape(28, 28))
    plt.axis('off')
