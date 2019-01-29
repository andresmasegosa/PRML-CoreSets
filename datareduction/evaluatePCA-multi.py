import numpy as np
import scipy
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
K=5
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

# ######################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=data.shape[1]

x_train = mnist.train.images#[0:2000,:]
x_test = mnist.test.images#[0:2000,:]


########################GPS DATA ##############################
#
# x_train = np.loadtxt('./gpsdata/FA_25_10_train.csv', delimiter=',', skiprows=1)
# x_test = np.loadtxt('./gpsdata/FA_25_10_test.csv', delimiter=',', skiprows=1)
#
# np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)
# np.take(x_test,np.random.permutation(x_test.shape[0]),axis=0,out=x_test)

#####################################################

#bpca = BayesianPCA(n_components=K)
#bpca.fit(x_train, initial="eigen")
#print(np.sum(bpca.log_proba(x_test)))
#test_ll[0,:] = np.repeat(np.sum(bpca.log_proba(x_test)),10)
######################################################

samples = np.zeros(10)

#samples = np.array([int(x_train.shape[0]*(m+1)/100) for m in range(0,10) ])
samples = np.array([25, 50, 100, 250, 500, 750, 1000])
#samples = np.array([2, 3, 4, 5])
#samples = np.array([0.1, 0.25, 0.5, 0.75, 1.0])


R = 10

K = 10

clusterError = np.zeros((samples.shape[0],R))
test_ll = np.zeros((4,samples.shape[0],R))
from slackclient import SlackClient
sc = SlackClient('xoxp-157419655798-161969456967-412895555920-73008d912fdc1999899080b1d1bc44eb')

for m in range(0,samples.shape[0]):
    print(samples[m])
    M=samples[m]

    text = "PCA-Final-MINST" + str(M)
    sc.api_call(
        "chat.postMessage",
        channel="@andresmasegosa",
        text=text
    )

    np.random.seed(1234)
    for r in range(0,R):
        print(r)

        #M = int(samples[m]/100.0*x_train.shape[0]/(2.0))
        bpca_dr = BayesianPCA_DR(n_components=K)
        bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="SS")
        test_ll[1,m,r]=np.sum(bpca_dr.log_proba(x_test))
        clusterError[m,r]=bpca_dr.clusterError

        #M = int(samples[m]/100.0*x_train.shape[0])
        bpca_dr = BayesianPCA_DR(n_components=K)
        bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="NoSS")
        test_ll[2,m,r]= np.sum(bpca_dr.log_proba(x_test))

        #M = int(samples[m]/100.0*x_train.shape[0])
        bpca_dr = BayesianPCA_DR(n_components=K)
        bpca_dr.fit(x_train, initial="eigen", n_clusters=M, cluster_method="random")
        test_ll[3,m,r]= np.sum(bpca_dr.log_proba(x_test))

np.savetxt('./figs/PCA_MINST_Multi_clustererror.txt', clusterError)
np.savetxt('./figs/PCA_MINST_Multi_data_SS.txt',test_ll[1])
np.savetxt('./figs/PCA_MINST_Multi_data_NoSS.txt',test_ll[2])
np.savetxt('./figs/PCA_MINST_Multi_data_Random.txt',test_ll[3])


import matplotlib.pyplot as plt
import numpy as np
samples = np.array([25, 50, 100, 250, 500, 750, 1000])
R = 10

test_ll = np.zeros((4,samples.shape[0],R))
test_ll[1] = np.loadtxt('./datareduction/figs2/PCA_GPS_Multi_data_SS.txt')
test_ll[2] = np.loadtxt('./datareduction/figs2/PCA_GPS_Multi_data_NoSS.txt')
test_ll[3] = np.loadtxt('./datareduction/figs2/PCA_GPS_Multi_data_Random.txt')
clusterError = np.loadtxt('./datareduction/figs2/PCA_GPS_Multi_clustererror.txt')
SVI = np.loadtxt('./datareduction/figs2/PCA_GPS_SVI.txt')
#SVI = np.repeat(np.max(SVI),6)


from tabulate import tabulate
print(tabulate(np.repeat(np.mean(SVI),7)[None,:], tablefmt="latex", floatfmt=".2e"))
print(tabulate(np.mean(test_ll,axis = 2), tablefmt="latex", floatfmt=".2e"))
print(tabulate(np.repeat(np.std(SVI),7)[None,:], tablefmt="latex", floatfmt=".2e"))
print(tabulate(np.std(test_ll,axis = 2), tablefmt="latex", floatfmt=".2e"))



import matplotlib.pyplot as pyplt

# Create a figure instance
fig = pyplt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

bp4 = ax.boxplot([SVI for i in range(0,samples.shape[0])], patch_artist=True)
for box in bp4['boxes']:
    # change outline color
    box.set( color='r', linewidth=2)
    box.set( facecolor = 'w')
## change color and linewidth of the whiskers
for whisker in bp4['whiskers']:
    whisker.set(color='r', linewidth=2)
## change color and linewidth of the caps
for cap in bp4['caps']:
    cap.set(color='r', linewidth=2)
## change color and linewidth of the medians
for median in bp4['medians']:
    median.set(color='r', linewidth=2)
## change the style of fliers and their fill
for flier in bp4['fliers']:
    flier.set(markerfacecolor='r', markeredgecolor='y')

bp1 = ax.boxplot([test_ll[1,i] for i in range(0,samples.shape[0])], patch_artist=True)
for box in bp1['boxes']:
    # change outline color
    box.set( color='b', linewidth=2)
    box.set( facecolor = 'w' )
## change color and linewidth of the whiskers
for whisker in bp1['whiskers']:
    whisker.set(color='b', linewidth=2)
## change color and linewidth of the caps
for cap in bp1['caps']:
    cap.set(color='b', linewidth=2)
## change color and linewidth of the medians
for median in bp1['medians']:
    median.set(color='b', linewidth=2)
## change the style of fliers and their fill
for flier in bp1['fliers']:
    flier.set(markerfacecolor='b', markeredgecolor='b')

bp2 = ax.boxplot([test_ll[2,i] for i in range(0,samples.shape[0])], patch_artist=True)
for box in bp2['boxes']:
    # change outline color
    box.set( color='g', linewidth=2)
    box.set( facecolor = 'w' )
## change color and linewidth of the whiskers
for whisker in bp2['whiskers']:
    whisker.set(color='g', linewidth=2)
## change color and linewidth of the caps
for cap in bp2['caps']:
    cap.set(color='g', linewidth=2)
## change color and linewidth of the medians
for median in bp2['medians']:
    median.set(color='g', linewidth=2)
## change the style of fliers and their fill
for flier in bp2['fliers']:
    flier.set(markerfacecolor='g', markeredgecolor='g')


bp3 = ax.boxplot([test_ll[3,i] for i in range(0,samples.shape[0])], patch_artist=True)
for box in bp3['boxes']:
    # change outline color
    box.set( color='y', linewidth=2)
    box.set( facecolor = 'w' )
## change color and linewidth of the whiskers
for whisker in bp3['whiskers']:
    whisker.set(color='y', linewidth=2)
## change color and linewidth of the caps
for cap in bp3['caps']:
    cap.set(color='y', linewidth=2)
## change color and linewidth of the medians
for median in bp3['medians']:
    median.set(color='y', linewidth=2)
## change the style of fliers and their fill
for flier in bp3['fliers']:
    flier.set(markerfacecolor='y', markeredgecolor='y')


ax.legend([bp4["boxes"][0], bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['SVI', 'DR-SSS', 'DR-PSSS', 'DR-Random'], loc='lower right')

ax.set_xticklabels(samples)
ax.set_ylim([-0.5e07, 0.2e07])
#ax.set_ylim([-0.3e11, 0.5e10])
ax.set_xlabel('Samples')
ax.set_ylabel('Test Log-Likelihood')
plt.savefig("./datareduction/figs2/PCA_MINST_Multi_LL.pdf",bbox_inches='tight')

# #########
#
# fig = pyplt.figure(2, figsize=(9, 6))
#
# # Create an axes instance
# ax = fig.add_subplot(111)
#
# bp1 = ax.boxplot([test_ll[1,i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp1['boxes']:
#     # change outline color
#     box.set( color='b', linewidth=2)
#     box.set( facecolor = 'w' )
# ## change color and linewidth of the whiskers
# for whisker in bp1['whiskers']:
#     whisker.set(color='b', linewidth=2)
# ## change color and linewidth of the caps
# for cap in bp1['caps']:
#     cap.set(color='b', linewidth=2)
# ## change color and linewidth of the medians
# for median in bp1['medians']:
#     median.set(color='b', linewidth=2)
# ## change the style of fliers and their fill
# for flier in bp1['fliers']:
#     flier.set(markerfacecolor='b', markeredgecolor='b')
#
# bp2 = ax.boxplot([clusterError[i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp2['boxes']:
#     # change outline color
#     box.set( color='g', linewidth=2)
#     box.set( facecolor = 'w' )
# ## change color and linewidth of the whiskers
# for whisker in bp2['whiskers']:
#     whisker.set(color='g', linewidth=2)
# ## change color and linewidth of the caps
# for cap in bp2['caps']:
#     cap.set(color='g', linewidth=2)
# ## change color and linewidth of the medians
# for median in bp2['medians']:
#     median.set(color='g', linewidth=2)
# ## change the style of fliers and their fill
# for flier in bp2['fliers']:
#     flier.set(markerfacecolor='g', markeredgecolor='g')
#
# ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Log-Likelihood', 'Clustering-Error'], loc='center right')
#
# ax.set_xticklabels(samples)
# ax.set_xlabel('Samples')
# ax.set_ylabel('Clustering-Error/Log-Likelihood')
#
# plt.savefig("./datareduction/figs/PCA_MINST_Multi_ClusterError.pdf",bbox_inches='tight')
