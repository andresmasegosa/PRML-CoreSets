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
#samples = np.array([25, 50])

clusterError = np.zeros(samples.shape[0])

test_ll = np.zeros((4,samples.shape[0]))
test_ll[0,:]=samples

for m in range(0,samples.shape[0]):
    print(samples[m])
    M=samples[m]
    np.random.seed(1234)
    vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
    vgmm_dr.fit(x_train, n_clusters=M, cluster_method="SS")
    #print(np.sum([np.linalg.det(vgmm_dr.W[k]) for k in range(i,K)]))
    test_ll[1,m]=np.sum(vgmm_dr.logpdf(x_test))
    clusterError[m]=vgmm_dr.clusterError
    #similarty[1,m] = metrics.adjusted_rand_score(y_test, vgmm_dr.classify(x_test))
    print(test_ll[1,m])
    #print(similarty[1,m])
    #distance_ss[m]=np.linalg.norm(params-np.hstack([p.flatten() for p in vgmm_dr.get_params()]))
    np.random.seed(1234)
    vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
    vgmm_dr.fit(x_train, n_clusters=M, cluster_method="NoSS")
    #print(np.sum([np.linalg.det(vgmm_dr.W[k]) for k in range(i,K)]))
    test_ll[2,m]= np.sum(vgmm_dr.logpdf(x_test))
    #similarty[2,m] = metrics.adjusted_rand_score(y_test, vgmm_dr.classify(x_test))
    print(test_ll[2,m])
    #print(similarty[2,m])
    #distance_noss[m]=np.linalg.norm(params-np.hstack([p.flatten() for p in vgmm_dr.get_params()]))
    np.random.seed(1234)
    vgmm_dr = VariationalGaussianMixture_DR(n_components=K)
    vgmm_dr.fit(x_train, n_clusters=M, cluster_method="random")
    #print(np.sum([np.linalg.det(vgmm_dr.W[k]) for k in range(i,K)]))
    test_ll[3,m]= np.sum(vgmm_dr.logpdf(x_test))
    #similarty[3,m] = metrics.adjusted_rand_score(y_test, vgmm_dr.classify(x_test))
    print(test_ll[3,m])
    #print(similarty[3,m])
    #distance_noss[m]=np.linalg.norm(params-np.hstack([p.flatten() for p in vgmm_dr.get_params()]))


np.savetxt('./figs/MoG_MINST_clustererror.txt', clusterError)
np.savetxt('./figs/MoG_MINST_data.txt',test_ll)

clusterError = np.loadtxt('./datareduction/figs/MoG_MINST_clustererror.txt')
test_ll = np.loadtxt('./datareduction/figs/MoG_MINST_data.txt')


x = [m for m in range(0,test_ll.shape[1])]

plt.figure(0)
plt.plot(x,test_ll[1,:], c='b', label='DR-SS')
plt.plot(x,test_ll[2,:], c='g', label='DR-NoSS')
plt.plot(x,test_ll[3,:], c='y', label='DR-Random')
plt.legend(loc='lower right', shadow=True)
plt.xticks(x, test_ll[0,:])
plt.ylim(-0.5e07, 0.2e07, 100)
plt.savefig("./datareduction/figs/MoG_MINST_LL.pdf",bbox_inches='tight')

plt.figure(1)
plt.plot(x,test_ll[1,:], c='b', label='Log-Likelihood')
plt.plot(x,clusterError, c='k', label='ClusterError')
plt.legend(loc='center right', shadow=True)
plt.xticks(x, test_ll[0,:])
plt.ylim(2e05, 2e06, 100)
plt.savefig("./datareduction/figs/MoG_MINST_ClusterError.pdf",bbox_inches='tight')
plt.show()


from tabulate import tabulate
print(tabulate(test_ll, tablefmt="latex", floatfmt=".2f"))
print(tabulate(clusterError[None,:], tablefmt="latex", floatfmt=".2f"))
