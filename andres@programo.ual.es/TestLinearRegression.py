import inferpy as inf

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from datareduction.variational_linear_regressor_DR import VariationalLinearRegressor_DR

from prml.rv import VariationalGaussianMixture
from prml.features import PolynomialFeatures
from prml.linear import (
    VariationalLinearRegressor,
    VariationalLogisticRegressor
)

np.random.seed(1234)

N=10000
K=50
D=10

# def create_toy_data(func, sample_size, std, domain=[0, 1]):
#     x = np.linspace(domain[0], domain[1], sample_size)
#     np.random.shuffle(x)
#     t = func(x) + np.random.normal(scale=std, size=x.shape)
#     return x, t
#
# def cubic(x):
#     return x * (x - 5) * (x + 5)
#
# x_train, y_train = create_toy_data(cubic, N, 10., [-5, 5])
# x = np.linspace(-5, 5, 100)
# y = cubic(x)

X_train=np.ones((N,D+1))
X_train[0:int(N/2),:] = inf.models.Normal(0,1,dim = D+1).sample(int(N/2))
X_train[int(N/2):N,:] = inf.models.Normal(10,1,dim = D+1).sample(int(N/2))
w = np.random.rand(D+1)
y_train = X_train@w.T

X=np.ones((N,D+1))
X[0:int(N/2),:] = inf.models.Normal(0,1,dim = D+1).sample(int(N/2))
X[int(N/2):N,:] = inf.models.Normal(10,1,dim = D+1).sample(int(N/2))
y = X@w.T


#feature = PolynomialFeatures(degree=D)
#X_train = feature.transform(x_train)
#X = feature.transform(x)

vlr = VariationalLinearRegressor(beta=0.01)
vlr.fit(X_train, y_train)
y_mean, y_std = vlr.predict(X, return_std=True)
# plt.scatter(x_train, y_train, s=100, facecolor="none", edgecolor="b")
# plt.plot(x, y, c="g", label="$\sin(2\pi x)$")
# plt.plot(x, y_mean, c="r", label="prediction")
# plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color="pink")
# plt.legend()
# plt.show()

normal = inf.models.Normal(y_mean,y_std)

l = normal.log_prob(y)
print(np.sum(l))


y_repeated = np.repeat(np.expand_dims(y_train,axis=1),X_train.shape[1],axis=1)
XY_train = np.multiply(X_train,y_repeated)

# np.multiply(np.expand_dims(X_train,axis=2),np.expand_dims(X_train,axis=1))[1] == np.matmul(np.expand_dims(X_train[1],axis=1), np.expand_dims(X_train[1],axis=1).T)
XX_train = np.multiply(np.expand_dims(X_train,axis=2),np.expand_dims(X_train,axis=1))

XX_train = XX_train.reshape((XX_train.shape[0],-1))

XJoin_train = np.concatenate((XY_train,XX_train),axis=1)


kmeans = KMeans(n_clusters=K, random_state=0).fit(XJoin_train)
weights = np.asarray([sum(kmeans.labels_==x) for x in range(0, K)])

clusters_centers = np.multiply(kmeans.cluster_centers_,np.repeat(weights.reshape(K,1),kmeans.cluster_centers_.shape[1],axis=1))

clusters_sum = np.sum(clusters_centers,axis=0)

X_dr = {'XY': clusters_sum[0:(D+1)],'XX': clusters_sum[(D+1):(D+1)+(D+1)*(D+1)].reshape((D+1,D+1))}


vlr_dr = VariationalLinearRegressor_DR(beta=0.01)
vlr_dr.fit(X_dr)
y_mean_dr, y_std_dr = vlr_dr.predict(X, return_std=True)
# plt.scatter(x_train, y_train, s=100, facecolor="none", edgecolor="b")
# plt.plot(x, y, c="g", label="$\sin(2\pi x)$")
# plt.plot(x, y_mean, c="r", label="prediction")
# plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color="pink")
# plt.legend()
# plt.show()

normal_dr = inf.models.Normal(y_mean_dr,y_std_dr)

l_dr = normal_dr.log_prob(y)
print(np.sum(l_dr))
