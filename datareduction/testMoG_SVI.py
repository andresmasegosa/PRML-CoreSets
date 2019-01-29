import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import inferpy as inf

from datareduction.bayesian_pca_DR import BayesianPCA_DR
from datareduction.bayesian_pca_SVI import BayesianPCA_SVI
from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR
from datareduction.variational_gaussian_mixture_SVI import VariationalGaussianMixture_SVI
from prml.feature_extractions import BayesianPCA
from prml.rv import VariationalGaussianMixture
from prml.features import PolynomialFeatures
from prml.linear import (
    VariationalLinearRegressor,
    VariationalLogisticRegressor
)

np.random.seed(1234)

N=10000
K=10
M=10
D=10


x_train = inf.models.Normal(0,1, dim = D).sample(int(N/K))
x_test = inf.models.Normal(0,1, dim = D).sample(int(N/K))

for i in range(1,K):
    x_train=np.append(x_train, inf.models.Normal(i,1, dim = D).sample(int(N/K)),axis=0)
    x_test=np.append(x_test, inf.models.Normal(i,1, dim = D).sample(int(N/K)),axis=0)


np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)
np.take(x_test,np.random.permutation(x_test.shape[0]),axis=0,out=x_test)

######################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=mnist.train.images.shape[1]

x_train = mnist.train.images[0:1000,:]
x_test = mnist.test.images[1000:2000,:]
######################################################
from skimage.transform import resize
x_train2 = np.zeros((x_train.shape[0],100))
x_test2 = np.zeros((x_test.shape[0],100))

for i in range(0, x_train.shape[0]):
    x_train2[i,:]=np.resize(resize(np.resize(x_train[i],(28,28)), (10, 10)),(1,100))

for i in range(0, x_test.shape[0]):
    x_test2[i,:]=np.resize(resize(np.resize(x_test[i],(28,28)), (10, 10)),(1,100))

x_train = x_train2
x_test = x_test2


np.random.seed(1234)
vgmm = VariationalGaussianMixture(n_components=K)
vgmm.fit(x_train)
print(np.sum(vgmm.logpdf(x_test)))

np.random.seed(1234)
vgmm_svi = VariationalGaussianMixture_SVI(n_components=K,learning_decay=0.6)
ll = vgmm_svi.fit(x_train, iter_max=25, batch_size=300, testset = x_test)
print(np.sum(vgmm_svi.logpdf(x_test)))
print(ll)

np.random.seed(1234)
bpca = BayesianPCA(n_components=K)
bpca.fit(x_train, initial="eigen")
print(sum(bpca.log_proba(x_test)))

np.random.seed(1234)
bpca_svi = BayesianPCA_SVI(n_components=K,learning_decay=0.6)
ll = bpca_svi.fit(x_train, initial="eigen", iter_max=25, batch_size=250, testset = x_test)
print(sum(bpca_svi.log_proba(x_test)))
print(ll)