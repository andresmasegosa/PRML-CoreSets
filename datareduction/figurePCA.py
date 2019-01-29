import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import inferpy as inf
from sklearn import metrics

from datareduction.bayesian_pca_DR import BayesianPCA_DR
from prml.feature_extractions import BayesianPCA, PCA

############## GENERATE DATA ########################
############################################

np.random.seed(1234)

N=2000
K=1
M=3
D=2


def create_toy_data(sample_size=100, ndim_hidden=1, ndim_observe=2, std=1.):
    Z = np.random.normal(size=(sample_size, ndim_hidden))
    mu = 5#np.random.uniform(-5, 5, size=(ndim_observe))
    W = np.random.uniform(-5, 5, (ndim_hidden, ndim_observe))
    print(W.T)
    print(mu)
    X = Z.dot(W) + mu + np.random.normal(scale=std, size=(sample_size, ndim_observe))
    return X

data = create_toy_data(sample_size=N, ndim_hidden=K, ndim_observe=D, std=0.5)


np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
N=data.shape[0]
D=data.shape[1]

x_train=data[0:int(N/2.0),:]
x_test=data[int(N/2.0):N,:]

M=1

a=-2.5
b=12.5
c=0
d=10


x_train = x_train - np.mean(x_train,axis=0)

a=-5
b=5
c=-5
d=5



#plt.scatter(x_train[:,0],x_train[:,1])
plt.figure(0)
np.random.seed(1234)
bpca = BayesianPCA(n_components=1)
bpca.fit(x_train, initial="random")
print(sum(bpca.log_proba(x_test)))
print("Figure 0")
print(bpca.W)
print(bpca.var)
print(bpca.C)

plt.scatter(x_train[:, 0], x_train[:, 1])
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(bpca.log_proba(x)).reshape(1000, 1000))
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/PCA_Artificial_TrueVI.pdf",bbox_inches='tight')

plt.figure(1)
np.random.seed(1234)
bpca_dr1 = BayesianPCA_DR(n_components=1)
bpca_dr1.fit(x_train, initial="random", n_clusters = 1, cluster_method="SS")
print("Figure 1")
print(sum(bpca_dr1.log_proba(x_test)))
print(bpca_dr1.W)
print(bpca_dr1.var)
print(bpca_dr1.C)


#plt.scatter(x_train[:, 0], x_train[:, 1], c = bpca_dr1.kmeans.labels_)
plt.scatter(x_train[:, 0], x_train[:, 1])
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(bpca_dr1.log_proba(x)).reshape(1000, 1000))
plt.scatter(bpca_dr1.X_dr['X'][:,0],bpca_dr1.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/PCA_Artificial_SS_M_1.pdf",bbox_inches='tight')

plt.figure(2)
np.random.seed(1234)
bpca_dr1 = BayesianPCA_DR(n_components=1)
bpca_dr1.fit(x_train, initial="random", n_clusters = 5, cluster_method="SS")
print("Figure 2")
print(sum(bpca_dr1.log_proba(x_test)))
print(bpca_dr1.W)
print(bpca_dr1.var)
print(bpca_dr1.C)


plt.scatter(x_train[:, 0], x_train[:, 1], c = bpca_dr1.kmeans.labels_)
plt.scatter(x_train[:, 0], x_train[:, 1])
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(bpca_dr1.log_proba(x)).reshape(1000, 1000))
plt.scatter(bpca_dr1.X_dr['X'][:,0],bpca_dr1.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/PCA_Artificial_SS_M_5.pdf",bbox_inches='tight')


plt.figure(3)
np.random.seed(1234)
bpca_dr2 = BayesianPCA_DR(n_components=1)
bpca_dr2.fit(x_train, initial="random", n_clusters = 1, cluster_method="NoSS")
print("Figure 3")
print(sum(bpca_dr2.log_proba(x_test)))
print(bpca_dr2.W)
print(bpca_dr2.var)
print(bpca_dr2.C)

plt.scatter(x_train[:, 0], x_train[:, 1])
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(bpca_dr2.log_proba(x)).reshape(1000, 1000))
plt.scatter(bpca_dr2.X_dr['X'][:,0],bpca_dr2.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/PCA_Artificial_NoSS_M_1.pdf",bbox_inches='tight')

plt.figure(4)
np.random.seed(1234)
bpca_dr2 = BayesianPCA_DR(n_components=1)
bpca_dr2.fit(x_train, initial="random", n_clusters = 5, cluster_method="NoSS")
print("Figure 4")
print(sum(bpca_dr2.log_proba(x_test)))
print(bpca_dr2.W)
print(bpca_dr2.var)
print(bpca_dr2.C)

plt.scatter(x_train[:, 0], x_train[:, 1], c = bpca_dr2.kmeans.labels_)
plt.scatter(x_train[:, 0], x_train[:, 1])
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(bpca_dr2.log_proba(x)).reshape(1000, 1000))
plt.scatter(bpca_dr2.X_dr['X'][:,0],bpca_dr2.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/PCA_Artificial_NoSS_M_5.pdf",bbox_inches='tight')

plt.figure(5)
np.random.seed(1234)
bpca_dr2 = BayesianPCA_DR(n_components=1)
bpca_dr2.fit(x_train, initial="random", n_clusters = 5, cluster_method="random")
print("Figure 5")
print(sum(bpca_dr2.log_proba(x_test)))
print(bpca_dr2.W)
print(bpca_dr2.var)
print(bpca_dr2.C)

plt.scatter(x_train[:, 0], x_train[:, 1])
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(bpca_dr2.log_proba(x)).reshape(1000, 1000))
plt.scatter(bpca_dr2.X_dr['X'][:,0],bpca_dr2.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/PCA_Artificial_Random_M_5_1.pdf",bbox_inches='tight')

plt.figure(6)
np.random.seed(123)
bpca_dr2 = BayesianPCA_DR(n_components=1)
bpca_dr2.fit(x_train, initial="random", n_clusters = 5, cluster_method="random")
print("Figure 6")
print(sum(bpca_dr2.log_proba(x_test)))
print(bpca_dr2.W)
print(bpca_dr2.var)
print(bpca_dr2.C)

plt.scatter(x_train[:, 0], x_train[:, 1])
x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
x = np.array([x0, x1]).reshape(2, -1).T
plt.contour(x0, x1, np.exp(bpca_dr2.log_proba(x)).reshape(1000, 1000))
plt.scatter(bpca_dr2.X_dr['X'][:,0],bpca_dr2.X_dr['X'][:,1], c='k', s=50.0, marker='+')
plt.xlim(a, b, 100)
plt.ylim(c, d, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("./figs/PCA_Artificial_Random_M_5_2.pdf",bbox_inches='tight')
plt.show()
