import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import inferpy as inf

from datareduction.bayesian_pca_DR import BayesianPCA_DR
from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR
from prml.feature_extractions import BayesianPCA
from prml.rv import VariationalGaussianMixture
from prml.features import PolynomialFeatures
from prml.linear import (
    VariationalLinearRegressor,
    VariationalLogisticRegressor
)

np.random.seed(0)

############## GENERATE DATA ########################
N=200
K=10
M=10
D=10

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

######################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=data.shape[1]

x_train = mnist.train.images[0:2000,:]
x_test = mnist.test.images[0:2000,:]



#####################################################

#bpca = BayesianPCA(n_components=K)
#bpca.fit(x_train, initial="eigen")
#print(np.sum(bpca.log_proba(x_test)))
#test_ll[0,:] = np.repeat(np.sum(bpca.log_proba(x_test)),10)
######################################################

samples = np.zeros(10)

samples = np.array([int(x_train.shape[0]*(m+1)/100) for m in range(0,10) ])
samples = np.array([25, 50, 100, 250, 500, 750, 1000])
#samples = np.array([25, 50, 100])
#samples = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#samples = np.array([20, 50, 100, 250, 500, 1000])

R = 30

clusterError = np.zeros((samples.shape[0],R))
test_ll = np.zeros((4,samples.shape[0],R))

for m in range(0,samples.shape[0]):
    print(samples[m])
    M=samples[m]
    np.random.seed(1234)
    for r in range(0,R):
        print(r)
        bpca_dr = BayesianPCA_DR(n_components=K)
        bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="SS")
        test_ll[1,m,r]=np.sum(bpca_dr.log_proba(x_test))
        clusterError[m,r]=bpca_dr.clusterError
        bpca_dr = BayesianPCA_DR(n_components=K)
        bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="NoSS")
        test_ll[2,m,r]= np.sum(bpca_dr.log_proba(x_test))
        bpca_dr = BayesianPCA_DR(n_components=K)
        bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="random")
        test_ll[3,m,r]= np.sum(bpca_dr.log_proba(x_test))

np.savetxt('./figs/PCA_MINST_Multi_clustererror.txt', clusterError)
np.savetxt('./figs/PCA_MINST_Multi_data_SS.txt',test_ll[1])
np.savetxt('./figs/PCA_MINST_Multi_data_NoSS.txt',test_ll[2])
np.savetxt('./figs/PCA_MINST_Multi_data_Random.txt',test_ll[3])

#
# test_ll = np.zeros((4,samples.shape[0],R))
# test_ll[1] = np.loadtxt('./datareduction/figs/PCA_MINST_Multi_data_SS.txt')
# test_ll[2] = np.loadtxt('./datareduction/figs/PCA_MINST_Multi_data_NoSS.txt')
# test_ll[3] = np.loadtxt('./datareduction/figs/PCA_MINST_Multi_data_Random.txt')
# clusterError = np.loadtxt('./datareduction/figs/PCA_MINST_Multi_clustererror.txt')
#
#
# import matplotlib.pyplot as pyplt
# # Create a figure instance
# fig = pyplt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax = fig.add_subplot(111)
#
# bp1 = ax.boxplot([test_ll[1,i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp1['boxes']:
#     # change outline color
#     box.set( color='b', linewidth=2)
#     box.set( facecolor = 'b' )
# ## change color and linewidth of the whiskers
# for whisker in bp1['whiskers']:
#     whisker.set(color='b', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp1['caps']:
#     cap.set(color='b', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp1['medians']:
#     median.set(color='b', linewidth=2)
#
#
# bp2 = ax.boxplot([test_ll[2,i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp2['boxes']:
#     # change outline color
#     box.set( color='g', linewidth=2)
#     box.set( facecolor = 'g' )
# ## change color and linewidth of the whiskers
# for whisker in bp2['whiskers']:
#     whisker.set(color='g', linewidth=2)
# ## change color and linewidth of the caps
# for cap in bp2['caps']:
#     cap.set(color='g', linewidth=2)
# ## change color and linewidth of the medians
# for median in bp2['medians']:
#     median.set(color='g', linewidth=2)
#
#
# bp3 = ax.boxplot([test_ll[3,i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp3['boxes']:
#     # change outline color
#     box.set( color='y', linewidth=2)
#     box.set( facecolor = 'y' )
# ## change color and linewidth of the whiskers
# for whisker in bp3['whiskers']:
#     whisker.set(color='y', linewidth=2)
# ## change color and linewidth of the caps
# for cap in bp3['caps']:
#     cap.set(color='y', linewidth=2)
# ## change color and linewidth of the medians
# for median in bp3['medians']:
#     median.set(color='y', linewidth=2)
#
#
# ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['DR-SS', 'DR-NoSS', 'DR-Random'], loc='lower right')
#
# ax.set_xticklabels(samples)

