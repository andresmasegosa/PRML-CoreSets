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

x_train = mnist.train.images#[0:2000,:]
x_test = mnist.test.images#[0:2000,:]



#####################################################

#bpca = BayesianPCA(n_components=K)
#bpca.fit(x_train, initial="eigen")
#print(np.sum(bpca.log_proba(x_test)))
#test_ll[0,:] = np.repeat(np.sum(bpca.log_proba(x_test)),10)
######################################################

samples = np.zeros(10)

samples = np.array([int(x_train.shape[0]*(m+1)/100) for m in range(0,10) ])
samples = np.array([25, 50, 100, 250, 500, 750, 1000])
#samples = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#samples = np.array([20, 50, 100, 250, 500, 1000])

clusterError = np.zeros(samples.shape[0])

test_ll = np.zeros((4,samples.shape[0]))
test_ll[0,:]=samples

for m in range(0,samples.shape[0]):
    print(samples[m])
    M=samples[m]
    np.random.seed(1234)
    bpca_dr = BayesianPCA_DR(n_components=K)
    bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="SS")
    test_ll[1,m]=np.sum(bpca_dr.log_proba(x_test))
    clusterError[m]=bpca_dr.clusterError
    print(test_ll[1,m])
    print(clusterError[m])
    print(np.sum(bpca_dr.log_proba(x_test)))
    #distance_ss[m]=np.linalg.norm(bpca.W - bpca_dr.W)
    np.random.seed(1234)
    bpca_dr = BayesianPCA_DR(n_components=K)
    bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="NoSS")
    test_ll[2,m]= np.sum(bpca_dr.log_proba(x_test))
    print(np.sum(bpca_dr.log_proba(x_test)))
    #distance_noss[m]=np.linalg.norm(bpca.W - bpca_dr.W)
    np.random.seed(1234)
    bpca_dr = BayesianPCA_DR(n_components=K)
    bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="random")
    test_ll[3,m]= np.sum(bpca_dr.log_proba(x_test))
    print(np.sum(bpca_dr.log_proba(x_test)))
    #distance_noss[m]=np.linalg.norm(bpca.W - bpca_dr.W)


np.savetxt('./figs/PCA_MINST_clustererror.txt', clusterError)
np.savetxt('./figs/PCA_MINST_data.txt',test_ll)

test_ll = np.loadtxt('./datareduction/figs/PCA_MINST_data.txt')
clusterError = np.loadtxt('./datareduction/figs/PCA_MINST_clustererror.txt')

x = [m for m in range(0,test_ll.shape[1])]

plt.figure(0)
plt.plot(x,test_ll[1,:], c='b', label='DR-SS')
plt.plot(x,test_ll[2,:], c='g', label='DR-NoSS')
plt.plot(x,test_ll[3,:], c='y', label='DR-Random')
plt.legend(loc='lower right', shadow=True)
plt.xticks(x, test_ll[0,:])
plt.ylim(-0.5e07, 0.2e07, 100)
plt.savefig("./datareduction/figs/PCA_MINST_LL.pdf",bbox_inches='tight')

plt.figure(1)
plt.plot(x,test_ll[1,:], c='b', label='Log-Likelihood')
plt.plot(x,clusterError, c='k', label='ClusterError')
plt.legend(loc='center right', shadow=True)
plt.xticks(x, test_ll[0,:])
plt.ylim(2e05, 2e06, 100)
plt.savefig("./datareduction/figs/PCA_MINST_ClusterError.pdf",bbox_inches='tight')
plt.show()


from tabulate import tabulate
print(tabulate(test_ll, tablefmt="latex", floatfmt=".2f"))
print(tabulate(clusterError[None,:], tablefmt="latex", floatfmt=".2f"))
