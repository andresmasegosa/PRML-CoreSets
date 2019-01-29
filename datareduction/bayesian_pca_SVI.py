import numpy as np
from prml.feature_extractions.pca import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

class BayesianPCA_SVI(PCA):
    def __init__(self, n_components, learning_decay=0.7):
        """
        construct principal component analysis

        Parameters
        ----------
        n_components : int
            number of components
        """
        assert isinstance(n_components, int)
        self.n_components = n_components
        self.learning_decay=learning_decay

    def _batch(self, X, total_size):
        weights = np.repeat(total_size/X.shape[0],X.shape[0])
        D=X.shape[1]
        self.X_dr = {'X': X, 'XX': X**2, 'W': weights}

    def eigen(self, X_dr, *arg):
        sample_size = int(np.sum(X_dr['W']))
        X = self.X_dr['W'][:,None]*self.X_dr['X']
        n_features = X.shape[1]
        if sample_size >= n_features:
            cov = np.cov(X, rowvar=False)
            values, vectors = np.linalg.eigh(cov)
            index = n_features - self.n_components
        else:
            cov = np.cov(X)
            values, vectors = np.linalg.eigh(cov)
            vectors = (X.T @ vectors) / np.sqrt(sample_size * values)
            index = sample_size - self.n_components
        self.I = np.eye(self.n_components)
        if index == 0:
            self.var = 0
        else:
            self.var = np.mean(values[:index])

        self.W = vectors[:, index:].dot(np.sqrt(np.diag(values[index:]) - self.var * self.I))
        self.__M = self.W.T @ self.W + self.var * self.I
        self.C = self.W @ self.W.T + self.var * np.eye(n_features)
        if index == 0:
            self.Cinv = np.linalg.inv(self.C)
        else:
            self.Cinv = np.eye(n_features) / np.sqrt(self.var) - self.W @ np.linalg.inv(self.__M) @ self.W.T / self.var

    def fit(self, X, iter_max=100, initial="random", batch_size=10, testset = None):
        """
        empirical bayes estimation of pca parameters

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        iter_max : int
            maximum number of em steps

        Returns
        -------
        mean : (n_features,) ndarray
            sample mean fo the input data
        W : (n_features, n_components) ndarray
            projection matrix
        var : float
            variance of observation noise
        """
        import time
        start = time.time()

        self._batch(np.array_split(X, int(X.shape[0]/batch_size))[0], X.shape[0])

        initial_list = ["random", "eigen"]
        self.mean = np.sum(self.X_dr['W'][:,None]*self.X_dr['X'], axis=0)/sum(self.X_dr['W'])
        self.I = np.eye(self.n_components)
        if initial not in initial_list:
            print("availabel initializations are {}".format(initial_list))
        if initial == "random":
            self.W = np.eye(np.size(self.X_dr['X'], 1), self.n_components)
            self.var = 1.
        elif initial == "eigen":
            self.eigen(self.X_dr)
        self.alpha = len(self.mean) / np.sum(self.W ** 2, axis=0).clip(min=1e-10)

        self.n_batch_iter_=1

        ll = np.zeros(iter_max)
        time_s = np.zeros(iter_max)
        for i in range(iter_max):
            W = np.copy(self.W)
            np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)
            for batch in np.array_split(X, int(X.shape[0]/batch_size), axis=0):
                self.n_batch_iter_ += 1
                self._batch(batch, X.shape[0])
                Ez, Ezz = self._expectation(self.X_dr['X']-self.mean)
                self._maximization(self.X_dr, Ez, Ezz)
                #self.alpha = len(self.mean) / np.sum(self.W ** 2, axis=0).clip(min=1e-10)
            if np.allclose(W, self.W):
                break
            print(i)
            self.n_iter = i + 1
            self.C = self.W @ self.W.T + self.var * np.eye(np.size(self.X_dr['X'], 1))
            self.Cinv = np.linalg.inv(self.C)
            ll[i]=sum(self.log_proba(testset))
            time_s[i]= time.time() - start
            print(ll[i])
            print(time_s[i])

        return ll, time_s

    def _maximization(self, X_dr, Ez, Ezz):
        X_mean = (X_dr['X']-self.mean)
        self.new_W = (X_mean*X_dr['W'][:,None]).T @ Ez @ np.linalg.inv(np.sum(Ezz*X_dr['W'][:,None,None], axis=0) + self.var * np.diag(self.alpha))
        self.new_var = np.sum(
            (np.mean((X_dr['XX'] - 2*X_dr['X']*self.mean + self.mean ** 2), axis=-1)
            - 2 * np.mean(Ez @ self.W.T * X_mean, axis=-1)
            + np.trace((Ezz @ self.W.T @ self.W).T)/ len(self.mean))*X_dr['W'])/sum(X_dr['W'])

        self.learning_offset = 10
        learning_rate = np.power(self.learning_offset + self.n_batch_iter_,
                          -self.learning_decay)

        self.W = (1-learning_rate)*self.W + learning_rate*self.new_W
        self.var = (1-learning_rate)*self.var + learning_rate*self.new_var
        self.var=max(self.var,0.000001)

