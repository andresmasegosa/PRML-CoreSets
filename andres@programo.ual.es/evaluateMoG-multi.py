import numpy as np
import inferpy as inf
from skimage.transform import resize
import matplotlib.pyplot as plt

from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR
from prml.rv import VariationalGaussianMixture

############## GENERATE DATA ########################

N=10000
K=10
M=10
D=10

x_train = inf.models.Normal(0,0.1, dim = D).sample(int(N/K))
x_test = inf.models.Normal(0,0.1, dim = D).sample(1000)
y_test = np.repeat(0,int(N/K))

for i in range(1,K):
    x_train=np.append(x_train, inf.models.Normal(i,0.1, dim = D).sample(int(N/K)),axis=0)
    x_test=np.append(x_test, inf.models.Normal(i,0.1, dim = D).sample(1000),axis=0)
    y_test = np.append(y_test, np.repeat(i, int(N / K)))


np.random.seed(10)

cov = np.random.rand(D,D)
cov = np.dot(cov,cov.transpose())

x_train = np.random.multivariate_normal(np.repeat(0,D),cov,int(N/K))
x_test = np.random.multivariate_normal(np.repeat(0,D),cov,int(N/K))
y_test = np.repeat(0,int(N/K))

for i in range(1,K):
    x_train=np.append(x_train, np.random.multivariate_normal(np.repeat(10*i,D),cov,int(N/K)),axis=0)
    x_test=np.append(x_test, np.random.multivariate_normal(np.repeat(10*i,D),cov,int(N/K)),axis=0)
    y_test = np.append(y_test, np.repeat(i, int(N / K)))



np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)


######################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=mnist.train.images.shape[1]

x_train = mnist.train.images#[0:1000,:]
x_test = mnist.test.images#[0:1000,:]
y_test =mnist.test.labels#[0:1000]

x_train2 = np.zeros((x_train.shape[0],100))
x_test2 = np.zeros((x_test.shape[0],100))

for i in range(0, x_train.shape[0]):
    x_train2[i,:]=np.resize(resize(np.resize(x_train[i],(28,28)), (10, 10)),(1,100))

for i in range(0, x_test.shape[0]):
    x_test2[i,:]=np.resize(resize(np.resize(x_test[i],(28,28)), (10, 10)),(1,100))

x_train = x_train2
x_test = x_test2

######################################################
np.random.seed(1234)

#
# vgmm = VariationalGaussianMixture(n_components=K)
# vgmm.fit(x_train)
#
# test_ll[0,:] = np.repeat(np.sum(vgmm.logpdf(x_test)),10)
# similarty[0,:] = np.repeat(metrics.adjusted_mutual_info_score(y_test,vgmm.classify(x_test)),10)
# #print(test_ll[0, 0])
# #print(similarty[0, 0])
# print(np.sum([np.linalg.det(vgmm.W[k]) for k in range(i, K)]))
# params = np.hstack([p.flatten() for p in vgmm.get_params()])
######################################################

samples = np.zeros(10)

samples = [int(x_train.shape[0]*(m+1)/1000) for m in range(0,10) ]
samples = np.array([25, 50, 100, 250, 500, 750, 1000])
#samples = np.array([25, 50, 100])

R = 30

clusterError = np.zeros((samples.shape[0],R))
test_ll = np.zeros((4,samples.shape[0],R))

for m in range(0,samples.shape[0]):
    print(samples[m])
    M=samples[m]
    np.random.seed(1234)
    for r in range(0,R):
        print(r)
        vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
        vgmm_dr.fit(x_train, n_clusters=M, cluster_method="SS")
        test_ll[1,m,r]=np.sum(vgmm_dr.logpdf(x_test))
        clusterError[m,r]=vgmm_dr.clusterError
        vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
        vgmm_dr.fit(x_train, n_clusters=M, cluster_method="NoSS")
        test_ll[2,m,r]= np.sum(vgmm_dr.logpdf(x_test))
        vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
        vgmm_dr.fit(x_train, n_clusters=M, cluster_method="random")
        test_ll[3,m,r]= np.sum(vgmm_dr.logpdf(x_test))

np.savetxt('./figs/MoG_MINST_Multi_clustererror.txt', clusterError)
np.savetxt('./figs/MoG_MINST_Multi_data_SS.txt',test_ll[1])
np.savetxt('./figs/MoG_MINST_Multi_data_NoSS.txt',test_ll[2])
np.savetxt('./figs/MoG_MINST_Multi_data_Random.txt',test_ll[3])


# test_ll = np.zeros((4,samples.shape[0],R))
# test_ll[1] = np.loadtxt('./datareduction/figs/MoG_MINST_Multi_data_SS.txt')
# test_ll[2] = np.loadtxt('./datareduction/figs/MoG_MINST_Multi_data_NoSS.txt')
# test_ll[3] = np.loadtxt('./datareduction/figs/MoG_MINST_Multi_data_Random.txt')
# clusterError = np.loadtxt('./datareduction/figs/MoG_MINST_Multi_clustererror.txt')
#
#
# import matplotlib.pyplot as pyplt
# # Create a figure instance
# fig = pyplt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax = fig.add_subplot(111)
#
# bp1 = ax.boxplot([test_ll[1,i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp1['boxes']:
#     # change outline color
#     box.set( color='b', linewidth=2)
#     box.set( facecolor = 'b' )
# ## change color and linewidth of the whiskers
# for whisker in bp1['whiskers']:
#     whisker.set(color='b', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp1['caps']:
#     cap.set(color='b', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp1['medians']:
#     median.set(color='b', linewidth=2)
#
#
# bp2 = ax.boxplot([test_ll[2,i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp2['boxes']:
#     # change outline color
#     box.set( color='g', linewidth=2)
#     box.set( facecolor = 'g' )
# ## change color and linewidth of the whiskers
# for whisker in bp2['whiskers']:
#     whisker.set(color='g', linewidth=2)
# ## change color and linewidth of the caps
# for cap in bp2['caps']:
#     cap.set(color='g', linewidth=2)
# ## change color and linewidth of the medians
# for median in bp2['medians']:
#     median.set(color='g', linewidth=2)
#
#
# bp3 = ax.boxplot([test_ll[3,i] for i in range(0,samples.shape[0])], patch_artist=True)
# for box in bp3['boxes']:
#     # change outline color
#     box.set( color='y', linewidth=2)
#     box.set( facecolor = 'y' )
# ## change color and linewidth of the whiskers
# for whisker in bp3['whiskers']:
#     whisker.set(color='y', linewidth=2)
# ## change color and linewidth of the caps
# for cap in bp3['caps']:
#     cap.set(color='y', linewidth=2)
# ## change color and linewidth of the medians
# for median in bp3['medians']:
#     median.set(color='y', linewidth=2)
#
#
# ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['DR-SS', 'DR-NoSS', 'DR-Random'], loc='lower right')
#
# ax.set_xticklabels(samples)
