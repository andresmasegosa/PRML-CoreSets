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
K=5
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

# ######################################################
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
# x_train = mnist.train.images#[0:2000,:]
# x_test = mnist.test.images#[0:2000,:]
#

########################GPS DATA ##############################

x_train = np.loadtxt('./gpsdata/FA_25_10_train.csv', delimiter=',', skiprows=1)
x_test = np.loadtxt('./gpsdata/FA_25_10_test.csv', delimiter=',', skiprows=1)

np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)
np.take(x_test,np.random.permutation(x_test.shape[0]),axis=0,out=x_test)

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

R = 1

K = 10

clusterError = np.zeros((samples.shape[0],R))
test_ll = np.zeros((4,samples.shape[0],R))
from slackclient import SlackClient
sc = SlackClient('xoxp-157419655798-161969456967-412895555920-73008d912fdc1999899080b1d1bc44eb')

import time

for m in range(0,samples.shape[0]):
    print(samples[m])
    M=samples[m]

    text = "PCA-time" + str(M)
    sc.api_call(
        "chat.postMessage",
        channel="@andresmasegosa",
        text=text
    )

    np.random.seed(1234)
    for r in range(0,R):
        print(r)
        start = time.time()
        bpca_dr = BayesianPCA_DR(n_components=K)
        bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="SS")
        end = time.time()
        test_ll[1,m,r]=end - start

np.savetxt('./figs/PCA_GPS_Time_Multi_SS.txt',test_ll[1])

