import numpy as np
import inferpy as inf
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

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


# ######################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

#data = data[np.random.choice(np.where(target == 3)[0], 10000)]
np.take(mnist.train.images,np.random.permutation(mnist.train.images.shape[0]),axis=0,out=mnist.train.images)
np.take(mnist.test.images,np.random.permutation(mnist.test.images.shape[0]),axis=0,out=mnist.test.images)

D=mnist.train.images.shape[1]

x_train = mnist.train.images#[0:1000,:]
x_test = mnist.test.images#[0:1000,:]
y_test =mnist.test.labels#[0:1000]

######################################################
x_train2 = np.zeros((x_train.shape[0],100))
x_test2 = np.zeros((x_test.shape[0],100))

for i in range(0, x_train.shape[0]):
    x_train2[i,:]=np.resize(resize(np.resize(x_train[i],(28,28)), (10, 10)),(1,100))

for i in range(0, x_test.shape[0]):
    x_test2[i,:]=np.resize(resize(np.resize(x_test[i],(28,28)), (10, 10)),(1,100))

x_train = x_train2
x_test = x_test2

########################GPS DATA ##############################

# x_train = np.loadtxt('./gpsdata/FA_25_10_train.csv', delimiter=',', skiprows=1)
# x_test = np.loadtxt('./gpsdata/FA_25_10_test.csv', delimiter=',', skiprows=1)
#
# np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)
# np.take(x_test,np.random.permutation(x_test.shape[0]),axis=0,out=x_test)
#
# x_test = x_test*1000
# x_train = x_train*1000

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
######################################################
X=x_train
#a = np.array_split(X, int(X.shape[0] / 1000), axis=0)
X_slide = X #a[0]
XX_slide = np.multiply(X_slide[:, :, None], X_slide[:, None, :])
XX_slide = XX_slide.reshape((XX_slide.shape[0], -1))
XJoin_slide = np.concatenate((X_slide, XX_slide), axis=1)
#
from sklearn import random_projection
#
transformer = random_projection.GaussianRandomProjection(n_components=X.shape[1])
# # transformer = random_projection.SparseRandomProjection(n_components=100)
X_proyected = transformer.fit_transform(XJoin_slide)
# for i in range(1, len(a)):
#     X_slide = a[i]
#     XX_slide = np.multiply(X_slide[:, :, None], X_slide[:, None, :])
#     XX_slide = XX_slide.reshape((XX_slide.shape[0], -1))
#     XJoin_slide = np.concatenate((X_slide, XX_slide), axis=1)
#     X_proyected = np.append(X_proyected, transformer.transform(XJoin_slide), axis=0)
#     print(i)

# np.savetxt("./dataProyected.txt",X_proyected)
# X_proyected = np.loadtxt("./datareduction/dataProyected.txt")
print("Proyection")
######################################################

samples = np.zeros(10)

#samples = [int(x_train.shape[0]*(m+1)/1000) for m in range(0,10) ]
samples = np.array([25, 50, 100, 250, 500, 750, 1000])
#samples = np.array([2, 3, 4, 5, 6])

R = 10

clusterError = np.zeros((samples.shape[0],R))
test_ll = np.zeros((4,samples.shape[0],R))

from slackclient import SlackClient
sc = SlackClient('xoxp-157419655798-161969456967-412895555920-73008d912fdc1999899080b1d1bc44eb')

text = "INIT MoG-Multi "
sc.api_call(
    "chat.postMessage",
    channel="@andresmasegosa",
    text=text
)

for m in range(0,samples.shape[0]):
    print(samples[m])
    M=samples[m]
    np.random.seed(1234)
    for r in range(0,R):
        print(r)

        #M = int(samples[m]/100.0*x_train.shape[0]/(x_train.shape[1]+1))
        vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
        vgmm_dr.fit(x_train, n_clusters=M, cluster_method="SS")
        test_ll[0,m,r]= np.sum(vgmm_dr.logpdf(x_test))

        #M = int(samples[m]/100.0*x_train.shape[0]/(x_train.shape[1]+1))
        vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
        vgmm_dr.fit(x_train, n_clusters=M, cluster_method="SSProyection", transformer=transformer,X_proyected=X_proyected)
        ll = vgmm_dr.logpdf(x_test)
        test_ll[1,m,r]= np.sum(ll)

        clusterError[m,r]=vgmm_dr.clusterError

        #M = int(samples[m]/100.0*x_train.shape[0])
        vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
        vgmm_dr.fit(x_train, n_clusters=M, cluster_method="NoSS")
        test_ll[2,m,r]= np.sum(vgmm_dr.logpdf(x_test))

        #M = int(samples[m]/100.0*x_train.shape[0])
        vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
        vgmm_dr.fit(x_train, n_clusters=M, cluster_method="random")
        test_ll[3,m,r]= np.sum(vgmm_dr.logpdf(x_test))

        text = "MoG-Final-MINST-Multi " + str(samples[m]) +"\n" + str(r) + " + "+ str(test_ll[0,m,r]) +" + " + str(test_ll[1,m,r])+ " + "+ str(test_ll[2,m,r]) +" + " + str(test_ll[3,m,r]) + "\n"
        #text = text +" + " + str(vgmm_dr.W)
        sc.api_call(
            "chat.postMessage",
            channel="@andresmasegosa",
            text=text
        )


np.savetxt('./figs/MoG10_GPS_Multi_clustererror.txt', clusterError)
np.savetxt('./figs/MoG10_GPS_Multi_data_SS.txt',test_ll[0])
np.savetxt('./figs/MoG10_GPS_Multi_data_SSProyected.txt',test_ll[1])
np.savetxt('./figs/MoG10_GPS_Multi_data_NoSS.txt',test_ll[2])
np.savetxt('./figs/MoG10_GPS_Multi_data_Random.txt',test_ll[3])
#

import matplotlib.pyplot as plt
samples = np.array([25, 50, 100, 250, 500, 750, 1000])
R=10
test_ll = np.zeros((4,samples.shape[0],R))
test_ll[0] = np.loadtxt('./datareduction/figs2/MoG_MINST_Multi_data_SS.txt')
test_ll[1] = np.loadtxt('./datareduction/figs2/MoG_MINST_Multi_data_SSProyected.txt')
test_ll[2] = np.loadtxt('./datareduction/figs2/MoG_MINST_Multi_data_NoSS.txt')
test_ll[3] = np.loadtxt('./datareduction/figs2/MoG_MINST_Multi_data_Random.txt')
clusterError = np.loadtxt('./datareduction/figs2/MoG_MINST_Multi_clustererror.txt')
SVI = np.loadtxt('./datareduction/figs2/MoG_MINST_SVI.txt')
#SVI = np.repeat(np.max(SVI),6)
#

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


bp0 = ax.boxplot([test_ll[0,i] for i in range(0,samples.shape[0])], patch_artist=True)
for box in bp0['boxes']:
    # change outline color
    box.set( color='b', linewidth=2)
    box.set( facecolor = 'w' )
## change color and linewidth of the whiskers
for whisker in bp0['whiskers']:
    whisker.set(color='b', linewidth=2)
## change color and linewidth of the caps
for cap in bp0['caps']:
    cap.set(color='b', linewidth=2)
## change color and linewidth of the medians
for median in bp0['medians']:
    median.set(color='b', linewidth=2)
## change the style of fliers and their fill
for flier in bp0['fliers']:
    flier.set(markerfacecolor='b', markeredgecolor='b')


bp1 = ax.boxplot([test_ll[1,i] for i in range(0,samples.shape[0])], patch_artist=True)
for box in bp1['boxes']:
    # change outline color
    box.set( color='black', linewidth=2)
    box.set( facecolor = 'w' )
## change color and linewidth of the whiskers
for whisker in bp1['whiskers']:
    whisker.set(color='black', linewidth=2)
## change color and linewidth of the caps
for cap in bp1['caps']:
    cap.set(color='black', linewidth=2)
## change color and linewidth of the medians
for median in bp1['medians']:
    median.set(color='black', linewidth=2)
## change the style of fliers and their fill
for flier in bp1['fliers']:
    flier.set(markerfacecolor='black', markeredgecolor='b')

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
    box.set( facecolor = 'w')
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



ax.legend([bp4["boxes"][0], bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['SVI', 'DR-SSS', 'DR-PSSS', 'DR-ODS', 'DR-Random'], loc='lower right')

ax.set_xticklabels(samples)
ax.set_ylim([-0.5e07, 0.2e07])
#ax.set_ylim([-4e07, 0])
ax.set_xlabel('Samples')
ax.set_ylabel('Test Log-Likelihood')
plt.savefig("./datareduction/figs2/MoG_MINST_Multi_LL.pdf",bbox_inches='tight')
# #

# #########
#
# fig = pyplt.figure(1, figsize=(9, 6))
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
# ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Log-Likelihood', 'ClusterError'], loc='center right')
# ax.set_xticklabels(samples)
# ax.set_xlabel('Samples')
# ax.set_ylabel('Clustering-Error/Log-Likelihood')
#
# plt.savefig("./datareduction/figs/MoG_MINST_Multi_ClusterError.pdf",bbox_inches='tight')
