import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
import inferpy as inf
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR

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
    mu = np.random.uniform(-5, 5, size=(ndim_observe))
    W = np.random.uniform(-5, 5, (ndim_hidden, ndim_observe))
    print(W.T)
    X = Z.dot(W) + mu + np.random.normal(scale=std, size=(sample_size, ndim_observe))
    return X

data = create_toy_data(sample_size=N, ndim_hidden=K, ndim_observe=D, std=1)


np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
N=data.shape[0]
D=data.shape[1]

X=data[0:int(2.0*N/3),:]
x_test=data[int(N/3.0):N,:]
n_clusters = M

np.random.seed(14)

XX = X ** 2
XJoin = np.concatenate((X, XX), axis=1)
scaler = StandardScaler()
XJoin_scaled=scaler.fit_transform(XJoin)
kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(XJoin_scaled)
weights = np.asarray([sum(kmeans.labels_ == x) for x in range(0, n_clusters)])
D = X.shape[1]
X_dr = {'X': kmeans.cluster_centers_[:, 0:D], 'XX': kmeans.cluster_centers_[:, D:2 * D], 'W': weights}

plt.scatter(X[:, 0], X[:, 1], c = kmeans.labels_)

np.random.seed(1)
X= np.linspace(-10,10,100)[:,None]
XX = X ** 2
XJoin = np.concatenate((X, XX), axis=1)
kmeans = MiniBatchKMeans(n_clusters=5).fit(XJoin)
plt.scatter(XJoin[:, 0], XJoin[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='+', s=100, c = np.arange(0,5))

sum = 0
for i in range(0,5):
    a = XJoin[kmeans.labels_ == i, :] - kmeans.cluster_centers_[i, :]
    sum += np.sqrt((a*a).sum(axis=1)).sum(axis=0)

print(sum)


kmeans = MiniBatchKMeans(n_clusters=20).fit(XJoin)
plt.scatter(XJoin[:, 0], XJoin[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='+', s=100)

sum = 0
for i in range(0,20):
    a = XJoin[kmeans.labels_ == i, :] - kmeans.cluster_centers_[i, :]
    sum += np.sqrt((a*a).sum(axis=1)).sum(axis=0)

print(sum)





######################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=mnist.train.images.shape[1]

x_train = mnist.train.images[0:1000,:]
x_test = mnist.test.images[0:1000,:]
y_test =mnist.test.labels#[0:1000]

x_train2 = np.zeros((1000,100))

from skimage.transform import resize
for i in range(0,x_train.shape[0]):
    x_train2[i,:]=np.resize(resize(np.resize(x_train[i],(28,28)), (10, 10)),(1,100))



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






import matplotlib.pyplot as plt
import numpy as np

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)

# basic plot
plt.boxplot(data)

# notched plot
plt.figure()
plt.boxplot(data, 1)

# change outlier point symbols
plt.figure()
plt.boxplot(data, 0, 'gD')

# don't show outlier points
plt.figure()
plt.boxplot(data, 0, '')

# horizontal boxes
plt.figure()
plt.boxplot(data, 0, 'rs', 0)

# change whisker length
plt.figure()
plt.boxplot(data, 0, 'rs', 0, 0.75)

# fake up some more data
spread = np.random.rand(50) * 100
center = np.ones(25) * 40
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
data.shape = (-1, 1)
d2.shape = (-1, 1)
# data = concatenate( (data, d2), 1 )
# Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# This is actually more efficient because boxplot converts
# a 2-D array into a list of vectors internally anyway.
data = [data, d2, d2[::2, 0]]
# multiple box plots on one figure
plt.figure()
plt.boxplot(data)

plt.show()


np.random.seed(10)
collectn_1 = np.random.normal(100, 10, 200)
collectn_2 = np.random.normal(80, 30, 200)
collectn_3 = np.random.normal(90, 20, 200)
collectn_4 = np.random.normal(70, 25, 200)

data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]

a = np.stack(data_to_plot, axis=0)

plt.boxplot(data_to_plot)

np.random.seed(1220)
collectn_1 = np.random.normal(100, 10, 200)
collectn_2 = np.random.normal(80, 30, 200)
collectn_3 = np.random.normal(90, 20, 200)
collectn_4 = np.random.normal(70, 25, 200)

data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]

a = np.stack(data_to_plot, axis=0)

plt.boxplot(data_to_plot)
