import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
import inferpy as inf
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR

from datareduction.bayesian_pca_DR import BayesianPCA_DR
from prml.feature_extractions import BayesianPCA, PCA
#
# ############## GENERATE DATA ########################
# ############################################

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
from sklearn import preprocessing
XJoin = preprocessing.scale(XJoin)
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

#
#
#
#
# ######################################################
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("MNIST_data/")
#
# #data = data[np.random.choice(np.where(target == 3)[0], 10000)]
# np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
# np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)
#
# D=mnist.train.images.shape[1]
#
# x_train = mnist.train.images[0:1000,:]
# x_test = mnist.test.images[0:1000,:]
# y_test =mnist.test.labels#[0:1000]
#
# x_train2 = np.zeros((1000,100))
#
# from skimage.transform import resize
# for i in range(0,x_train.shape[0]):
#     x_train2[i,:]=np.resize(resize(np.resize(x_train[i],(28,28)), (10, 10)),(1,100))
#
#
#
# ############## GENERATE DATA ########################
#
# N=1000
# K=2
# M=10
# D=2
#
#
#
# np.random.seed(10)
#
# cov = np.random.rand(D,D)
# cov = np.dot(cov,cov.transpose())
#
# x_train = np.random.multivariate_normal(np.repeat(5,D),cov,int(N/K))
# x_test = np.random.multivariate_normal(np.repeat(5,D),cov,int(N/K))
# y_test = np.repeat(0,int(N/K))
#
# for i in range(1,K):
#     x_train=np.append(x_train, np.random.multivariate_normal(np.repeat(10*i,D),cov,int(N/K)),axis=0)
#     x_test=np.append(x_test, np.random.multivariate_normal(np.repeat(10*i,D),cov,int(N/K)),axis=0)
#     y_test = np.append(y_test, np.repeat(i, int(N / K)))
#
#
# np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)
# a=0
# b=15
# c=0
# d=15
# np.random.seed(123456)
# vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
# vgmm_dr.fit(x_train, n_clusters=10, cluster_method="random")
# vgmm_dr.mu
#
# plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm_dr.classify(x_train))
# x0, x1 = np.meshgrid(np.linspace(a, b, 1000), np.linspace(c, d, 1000))
# x = np.array([x0, x1]).reshape(2, -1).T
# plt.contour(x0, x1, np.exp(vgmm_dr.logpdf(x)).reshape(1000, 1000))
# plt.scatter(vgmm_dr.X_dr['X'][:,0],vgmm_dr.X_dr['X'][:,1], c='k', s=100.0, marker='+')
# plt.xlim(a, b, 100)
# plt.ylim(c, d, 100)
# plt.gca().set_aspect('equal', adjustable='box')
#
#
#
#
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # fake up some data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 50
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# data = np.concatenate((spread, center, flier_high, flier_low), 0)
#
# # basic plot
# plt.boxplot(data)
#
# # notched plot
# plt.figure()
# plt.boxplot(data, 1)
#
# # change outlier point symbols
# plt.figure()
# plt.boxplot(data, 0, 'gD')
#
# # don't show outlier points
# plt.figure()
# plt.boxplot(data, 0, '')
#
# # horizontal boxes
# plt.figure()
# plt.boxplot(data, 0, 'rs', 0)
#
# # change whisker length
# plt.figure()
# plt.boxplot(data, 0, 'rs', 0, 0.75)
#
# # fake up some more data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 40
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
# data.shape = (-1, 1)
# d2.shape = (-1, 1)
# # data = concatenate( (data, d2), 1 )
# # Making a 2-D array only works if all the columns are the
# # same length.  If they are not, then use a list instead.
# # This is actually more efficient because boxplot converts
# # a 2-D array into a list of vectors internally anyway.
# data = [data, d2, d2[::2, 0]]
# # multiple box plots on one figure
# plt.figure()
# plt.boxplot(data)
#
# plt.show()
#
#
import numpy as np
import matplotlib.pyplot as plt
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
#
#
#
#

#############

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

from datareduction.online_lda_full import LatentDirichletAllocationFULL

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

# # Fit the NMF model
# print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
#       "n_samples=%d and n_features=%d..."
#       % (n_samples, n_features))
# t0 = time()
# nmf = NMF(n_components=n_components, random_state=1,
#           alpha=.1, l1_ratio=.5).fit(tfidf)
# print("done in %0.3fs." % (time() - t0))
#
# print("\nTopics in NMF model (Frobenius norm):")
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# print_top_words(nmf, tfidf_feature_names, n_top_words)
#
# # Fit the NMF model
# print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
#       "tf-idf features, n_samples=%d and n_features=%d..."
#       % (n_samples, n_features))
# t0 = time()
# nmf = NMF(n_components=n_components, random_state=1,
#           beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
#           l1_ratio=.5).fit(tfidf)
# print("done in %0.3fs." % (time() - t0))
#
# print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# print_top_words(nmf, tfidf_feature_names, n_top_words)
#
# print("Fitting LDA models with tf features, "
#       "n_samples=%d and n_features=%d..."
#       % (n_samples, n_features))
# lda = LatentDirichletAllocation(n_components=n_components,
#                                 learning_method='batch',
#                                 random_state=0)
# t0 = time()
# lda.fit(tf)
# print("done in %0.3fs." % (time() - t0))
#
# print("\nTopics in LDA model:")
# tf_feature_names = tf_vectorizer.get_feature_names()
# print_top_words(lda, tf_feature_names, n_top_words)
#

lda = LatentDirichletAllocationFULL(n_components=n_components,
                                learning_method='batch',
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# from kmodes.kmodes import KModes
#
# km = KModes(n_clusters=4, init='random', n_init=1)
#
# clusters = km.fit_predict(tf.toarray())
#
import numpy as np
id = np.array([1, 3])
w = np.array([3, 1])

m = np.array([[0, 1, 2, 3, 4, 5],[6, 7, 8, 9, 10,11]])

m[:,id]

np.repeat(m[:,id],w, axis = 1).shape




######################################################
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=mnist.train.images.shape[1]

X = mnist.train.images[0:1000,:]
x_test = mnist.test.images[0:1000,:]
y_test =mnist.test.labels#[0:1000]



a = np.array_split(X,10, axis=0)

X_slide = a[0]
XX_slide = np.multiply(X_slide[:, :, None], X_slide[:, None, :])
XX_slide = XX_slide.reshape((XX_slide.shape[0], -1))
XJoin_slide = np.concatenate((X_slide, XX_slide), axis=1)
XJoin_slide.shape

from sklearn import random_projection
transformer = random_projection.GaussianRandomProjection(n_components=28*28)
#transformer = random_projection.SparseRandomProjection(n_components=28*28)
X_proyected = transformer.fit_transform(XJoin_slide)
X_proyected.shape

for i in range(1,len(a)):
    X_slide = a[i]
    XX_slide = np.multiply(X_slide[:, :, None], X_slide[:, None, :])
    XX_slide = XX_slide.reshape((XX_slide.shape[0], -1))
    XJoin_slide = np.concatenate((X_slide, XX_slide), axis=1)
    X_proyected=np.append(X_proyected, transformer.transform(XJoin_slide),axis=0)


kmeans = MiniBatchKMeans(n_clusters=5).fit(X_proyected)

clusters = np.zeros((5,28**2+28**4))
for i in range(0,5):
    XCluster = X[kmeans.labels_ == i]
    XXCluster = np.multiply(XCluster[:, :, None], XCluster[:, None, :])
    XXCluster = XXCluster.reshape((XXCluster.shape[0], -1))
    XCluster_Join = np.concatenate((XCluster, XXCluster), axis=1)
    clusters[i,:]=np.mean(XCluster_Join,axis=0)



from slackclient import SlackClient

sc = SlackClient('xoxp-157419655798-161969456967-412895555920-73008d912fdc1999899080b1d1bc44eb')

text = "Hola"+str(4)
sc.api_call(
  "chat.postMessage",
  channel="@andresmasegosa",
  text=text
)

weights = np.array([3,4,5,0,6])
for i in range(0, weights.shape[0]):
    if (weights[i] == 0):
        continue
    print(weights[i])
