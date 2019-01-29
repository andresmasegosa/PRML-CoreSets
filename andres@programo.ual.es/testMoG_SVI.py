import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import inferpy as inf

from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR
from datareduction.variational_gaussian_mixture_SVI import VariationalGaussianMixture_SVI
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

x_train = mnist.train.images[0:100,:]
x_test = mnist.test.images[100:200,:]
######################################################

# np.random.seed(1234)
# vgmm = VariationalGaussianMixture(n_components=K)
# vgmm.fit(x_train)
# print(np.sum(vgmm.logpdf(x_test)))

np.random.seed(1234)
vgmm_svi = VariationalGaussianMixture_SVI(n_components=K)
vgmm_svi.fit(x_train, iter_max=1, batch_size=10)
print(np.sum(vgmm_svi.logpdf(x_test)))
