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
#plt.scatter(x_train[:,0],x_train[:,1])
# plt.figure(0)
# np.random.seed(1234)
# vgmm = VariationalGaussianMixture(n_components=2)
# vgmm.fit(x_train)
# vgmm.mu
#
# plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm.classify(x_train))
# x0, x1 = np.meshgrid(np.linspace(a, b, 100), np.linspace(c, d, 100))
# x = np.array([x0, x1]).reshape(2, -1).T
# plt.contour(x0, x1, np.exp(vgmm.logpdf(x)).reshape(100, 100))
# plt.xlim(-5, 10, 100)
# plt.ylim(-5, 10, 100)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig("./figs/MoG_Artificial_TrueVI.pdf",bbox_inches='tight')

plt.figure(0)
np.random.seed(1234)
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
np.random.seed(12)
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
plt.savefig("./figs/MoG_Artificial_SS_M_    10.pdf",bbox_inches='tight')

plt.figure(2)
np.random.seed(10)
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
np.random.seed(0)
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