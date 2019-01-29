import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import inferpy as inf
from sklearn import metrics

from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR
from prml.rv import VariationalGaussianMixture
from prml.features import PolynomialFeatures
from prml.linear import (
    VariationalLinearRegressor,
    VariationalLogisticRegressor
)

np.random.seed(1234)

N=10000
K=5
M=10
D=20


x_train = inf.models.Normal(0,1, dim = D).sample(int(N/K))
x_test = inf.models.Normal(0,1, dim = D).sample(int(N/K))
y_test = np.repeat(0,int(N/K))
for i in range(1,K):
    x_train=np.append(x_train, inf.models.Normal(i,1, dim = D).sample(int(N/K)),axis=0)
    x_test=np.append(x_test, inf.models.Normal(i,1, dim = D).sample(int(N/K)),axis=0)
    y_test = np.append(y_test, np.repeat(i, int(N / K)))

np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)

######################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=mnist.train.images.shape[1]

x_train = mnist.train.images[0:100,:]
x_test = mnist.test.images[100:200,:]
######################################################

np.random.seed(1234)
vgmm = VariationalGaussianMixture(n_components=K)
vgmm.fit(x_train)
print(np.sum(vgmm.logpdf(x_test)))
print(metrics.adjusted_rand_score(y_test,vgmm.classify(x_test)))

np.random.seed(1234)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters = M, cluster_method="SS")
print(np.sum(vgmm_dr.logpdf(x_test)))
print(metrics.adjusted_rand_score(y_test,vgmm_dr.classify(x_test)))

np.random.seed(1234)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters = M, cluster_method="NoSS")
print(np.sum(vgmm_dr.logpdf(x_test)))
print(metrics.adjusted_rand_score(y_test,vgmm_dr.classify(x_test)))


np.random.seed(1234)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters = M, cluster_method="random")
print(np.sum(vgmm_dr.logpdf(x_test)))
print(metrics.adjusted_rand_score(y_test,vgmm_dr.classify(x_test)))
