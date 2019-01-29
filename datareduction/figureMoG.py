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

from scipy import random, linalg

############## GENERATE DATA ########################

N=1000
K=2
M=10
D=2



np.random.seed(10)

cov = np.random.rand(D,D)
cov = np.dot(cov,cov.transpose())

x_train = np.random.multivariate_normal(np.repeat(5,D),cov,int(N/K))
x_test = np.random.multivariate_normal(np.repeat(5,D),cov,int(N/K))
y_test = np.repeat(0,int(N/K))

for i in range(1,K):
    x_train=np.append(x_train, np.random.multivariate_normal(np.repeat(10*i,D),cov,int(N/K)),axis=0)
    x_test=np.append(x_test, np.random.multivariate_normal(np.repeat(10*i,D),cov,int(N/K)),axis=0)
    y_test = np.append(y_test, np.repeat(i, int(N / K)))


np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)


a=0
b=15
c=0
d=15


x_train = x_train - np.mean(x_train,axis=0)

a=-5
b=5
c=-5
d=5

# X=x_train
# np.mean(X,axis=0)
# XX = np.multiply(X[:, :, None], X[:, None, :])
# XX = XX.reshape((XX.shape[0], -1))
# XJoin = np.concatenate((X, XX), axis=1)
#
# # from sklearn import preprocessing
# # XJoin = preprocessing.scale(XJoin)
# # np.mean(XJoin,axis=0)
# # np.var(XJoin,axis=0)
#
# n_clusters=2
# from sklearn.cluster import MiniBatchKMeans, KMeans
# kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(x_train)
# center = kmeans.cluster_centers_
# centerXX = np.multiply(center[:, :, None], center[:, None, :])
# centerXX = centerXX.reshape((centerXX.shape[0], -1))
# centerJoin = np.concatenate((center, centerXX), axis=1)
#
# kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=centerJoin).fit(XJoin)
# weights = np.asarray([sum(kmeans.labels_ == x) for x in range(0, n_clusters)])
# plt.scatter(XJoin[:, 0], XJoin[:, 3], c=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,3], c='k', s=50.0, marker='+')
#
# plt.scatter(x_train[:, 0], x_train[:, 1])



plt.figure(0)
np.random.seed(0)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters=2, cluster_method="SS")
vgmm_dr.mu

plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm_dr.classify(x_train))
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(vgmm_dr.logpdf(x)).reshape(1000, 1000))
plt.scatter(vgmm_dr.X_dr['X'][:,0],vgmm_dr.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/MoG_Artificial_SS_M_2.pdf",bbox_inches='tight')

plt.figure(1)
np.random.seed(1)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters=10, cluster_method="SS")
vgmm_dr.mu

plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm_dr.classify(x_train))
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(vgmm_dr.logpdf(x)).reshape(1000, 1000))
plt.scatter(vgmm_dr.X_dr['X'][:,0],vgmm_dr.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/MoG_Artificial_SS_M_10.pdf",bbox_inches='tight')

plt.figure(2)
np.random.seed(1)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters=2, cluster_method="NoSS")
vgmm_dr.mu

plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm_dr.classify(x_train))
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(vgmm_dr.logpdf(x)).reshape(1000, 1000))
plt.scatter(vgmm_dr.X_dr['X'][:,0],vgmm_dr.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/MoG_Artificial_NoSS_M_2.pdf",bbox_inches='tight')

plt.figure(3)
np.random.seed(10)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters=10, cluster_method="NoSS")
vgmm_dr.mu

plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm_dr.classify(x_train))
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(vgmm_dr.logpdf(x)).reshape(1000, 1000))
plt.scatter(vgmm_dr.X_dr['X'][:,0],vgmm_dr.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/MoG_Artificial_NoSS_M_10.pdf",bbox_inches='tight')


plt.figure(4)
np.random.seed(2)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters=10, cluster_method="random")
vgmm_dr.mu

plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm_dr.classify(x_train))
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(vgmm_dr.logpdf(x)).reshape(1000, 1000))
plt.scatter(vgmm_dr.X_dr['X'][:,0],vgmm_dr.X_dr['X'][:,1], c='k', s=100.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/MoG_Artificial_Random_M_10_0.pdf",bbox_inches='tight')

plt.figure(5)
np.random.seed(123456)
vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
vgmm_dr.fit(x_train, n_clusters=10, cluster_method="random")
vgmm_dr.mu

plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm_dr.classify(x_train))
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(vgmm_dr.logpdf(x)).reshape(1000, 1000))
plt.scatter(vgmm_dr.X_dr['X'][:,0],vgmm_dr.X_dr['X'][:,1], c='k', s=100.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/MoG_Artificial_Random_M_10_1.pdf",bbox_inches='tight')
plt.show()