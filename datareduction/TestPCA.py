import numpy as np
import inferpy as inf

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

from datareduction.bayesian_pca_DR import BayesianPCA_DR
from prml.feature_extractions import BayesianPCA, PCA

seed = 12341
np.random.seed(seed)


############################################

N=1000
K=1
M=3
D=2


def create_toy_data(sample_size=100, ndim_hidden=1, ndim_observe=2, std=1.):
    Z = np.random.normal(size=(sample_size, ndim_hidden))
    mu = np.random.uniform(-5, 5, size=(ndim_observe))
    W = np.random.uniform(-5, 5, (ndim_hidden, ndim_observe))
    #print(W.T)
    X = Z.dot(W) + mu + np.random.normal(scale=std, size=(sample_size, ndim_observe))
    return X

data = create_toy_data(sample_size=N, ndim_hidden=K, ndim_observe=D, std=1.)



#data = datasets.load_iris().data
#data = datasets.fetch_california_housing().data
#data = datasets.load_digits().data


np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
N=data.shape[0]
D=data.shape[1]

x_train=data[0:int(2.0*N/3),:]
x_test=data[int(N/3.0):N,:]

############################################
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("MNIST_data/")
#
# #data = data[np.random.choice(np.where(target == 3)[0], 10000)]
# np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
# np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)
#
# D=data.shape[1]
#
# x_train = mnist.train.images[0:1000,:]
# x_test = mnist.test.images[1000:2000,:]
############################################

#M=x_train.shape[0]

np.random.seed(seed)
bpca = BayesianPCA(n_components=K)
bpca.fit(x_train, initial="eigen")
print(sum(bpca.log_proba(x_test)))
#print(bpca.W)

np.random.seed(seed)
bpca_dr = BayesianPCA_DR(n_components=K)
bpca_dr.fit(x_train, initial="eigen", n_clusters = M)
print(sum(bpca_dr.log_proba(x_test)))
#print(bpca_dr.W)

np.random.seed(seed)
bpca_dr2 = BayesianPCA_DR(n_components=K)
bpca_dr2.fit(x_train, initial="eigen", n_clusters = M, cluster_method="NoSS")
print(sum(bpca_dr2.log_proba(x_test)))
#print(bpca_dr2.W)


#plt.scatter(x_train[:,0],x_train[:,1], c='b')
#plt.scatter(bpca_dr.X_dr['X'][:,0],bpca_dr.X_dr['X'][:,1], c='r')
#plt.scatter(bpca_dr2.X_dr['X'][:,0],bpca_dr2.X_dr['X'][:,1], c='y')

plt.figure(0)
plt.scatter(x_train[:,0],x_train[:,1], c=bpca_dr.kmeans.labels_)
plt.figure(1)
plt.scatter(x_train[:,0],x_train[:,1], c=bpca_dr2.kmeans.labels_)

XX = x_train ** 2
XJoin = np.concatenate((x_train, XX), axis=1)
pca = PCA(n_components=2)
Z = pca.fit_transform(XJoin)

plt.figure(2)
plt.scatter(Z[:,0],Z[:,1], c=bpca_dr.kmeans.labels_)
plt.figure(3)
plt.scatter(Z[:,0],Z[:,1], c=bpca_dr2.kmeans.labels_)
