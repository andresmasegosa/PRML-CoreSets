import numpy as np
import inferpy as inf
from skimage.transform import resize

from datareduction.bayesian_pca_SVI import BayesianPCA_SVI
from datareduction.variational_gaussian_mixture_DR import VariationalGaussianMixture_DR
from datareduction.variational_gaussian_mixture_SVI import VariationalGaussianMixture_SVI
from prml.rv import VariationalGaussianMixture

############## GENERATE DATA ########################

N=10000
K=10
M=10
D=10


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
# x_train = mnist.train.images#[0:1000,:]
# x_test = mnist.test.images#[0:1000,:]
# y_test =mnist.test.labels#[0:1000]
#
#
# ######################################################

########################GPS DATA ##############################

x_train = np.loadtxt('./gpsdata/FA_25_10_train.csv', delimiter=',', skiprows=1)
x_test = np.loadtxt('./gpsdata/FA_25_10_test.csv', delimiter=',', skiprows=1)

np.take(x_train,np.random.permutation(x_train.shape[0]),axis=0,out=x_train)
np.take(x_test,np.random.permutation(x_test.shape[0]),axis=0,out=x_test)

######################################################
np.random.seed(1234)

samples = np.zeros(10)

samples = np.array([250, 250, 500, 500, 1000, 1000])
lr = np.array([0.65, 0.85, 0.65, 0.85, 0.65, 0.85])

R = 10

clusterError = np.zeros((samples.shape[0],R))
test_ll = np.zeros(samples.shape[0])

from slackclient import SlackClient
sc = SlackClient('xoxp-157419655798-161969456967-412895555920-73008d912fdc1999899080b1d1bc44eb')

for m in range(0,samples.shape[0]):
    np.random.seed(1234)
    bpca_svi = BayesianPCA_SVI(n_components=K, learning_decay=lr[m])
    ll, _ = bpca_svi.fit(x_train, initial="eigen", iter_max=25, batch_size=samples[m], testset=x_test)
    test_ll[m] = np.sum(bpca_svi.log_proba(x_test))
    print(test_ll[m])

    text = "PCA-SVI " + str(m)
    sc.api_call(
        "chat.postMessage",
        channel="@andresmasegosa",
        text=text
    )


np.savetxt('./figs/PCA_GPS_SVI.txt',test_ll)
