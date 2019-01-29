import numpy as np
from prml.feature_extractions.pca import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class BayesianPCA_DR(PCA):

    def _clusteringError(self, X, kmeans):
        sum = 0
        for i in range(0, kmeans.cluster_centers_.shape[0]):
            a = X[kmeans.labels_ == i, :] - kmeans.cluster_centers_[i, :]
            sum += np.sqrt((a * a).sum(axis=1)).sum(axis=0)
        return sum

    def _random(self, X, n_clusters):

        centers_X = X[np.random.choice(X.shape[0], n_clusters, replace=False),:]
        centers_XX = centers_X**2
        weights = np.repeat(X.shape[0]/n_clusters,n_clusters)

        self.X_dr = {'X': centers_X, 'XX': centers_XX,
                'W': weights}

    def _clusterSS(self, X, n_clusters):
        center = MiniBatchKMeans(n_clusters=n_clusters).fit(X).cluster_centers_
        centerXX = center**2
        centerJoin = np.concatenate((center, centerXX), axis=1)

        XX = X ** 2
        XJoin = np.concatenate((X, XX), axis=1)
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=centerJoin).fit(XJoin)
        weights = np.asarray([sum(self.kmeans.labels_ == x) for x in range(0, n_clusters)])
        D=X.shape[1]
        self.X_dr = {'X': self.kmeans.cluster_centers_[:, 0:D], 'XX': self.kmeans.cluster_centers_[:, D:2 * D], 'W': weights}
        self.clusterError = self._clusteringError(XJoin,self.kmeans)

    def _cluster(self, X, n_clusters):
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(X)
        weights = np.asarray([sum(self.kmeans.labels_ == x) for x in range(0, n_clusters)])
        self.X_dr = {'X': self.kmeans.cluster_centers_, 'XX': self.kmeans.cluster_centers_ ** 2, 'W': weights}

    # def _clusterSS(self, X, n_clusters):
    #     scaler = StandardScaler()
    #     XX = X ** 2
    #     XJoin = np.concatenate((X, XX), axis=1)
    #     self.kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(scaler.fit_transform(XJoin))
    #     weights = np.asarray([sum(self.kmeans.labels_ == x) for x in range(0, n_clusters)])
    #     D=X.shape[1]
    #     self.kmeans.cluster_centers_=scaler.inverse_transform(self.kmeans.cluster_centers_)
    #     self.X_dr = {'X': self.kmeans.cluster_centers_[:, 0:D], 'XX': self.kmeans.cluster_centers_[:, D:2 * D], 'W': weights}
    #
    # def _cluster(self, X, n_clusters):
    #     scaler = StandardScaler()
    #     self.kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(scaler.fit_transform(X))
    #     weights = np.asarray([sum(self.kmeans.labels_ == x) for x in range(0, n_clusters)])
    #     self.kmeans.cluster_centers_=scaler.inverse_transform(self.kmeans.cluster_centers_)
    #     self.X_dr = {'X': self.kmeans.cluster_centers_, 'XX': self.kmeans.cluster_centers_ ** 2, 'W': weights}

    def eigen(self, X_dr, *arg):
        sample_size = np.sum(X_dr['W'])
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

    def fit(self, X, iter_max=100, initial="random", n_clusters=10, cluster_method="SS"):
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
        if cluster_method== "SS":
            self._clusterSS(X,n_clusters)
        elif cluster_method== "NoSS":
            self._cluster(X,n_clusters)
        elif cluster_method == "random":
            self._random(X,n_clusters)

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


        for i in range(iter_max):
            W = np.copy(self.W)
            Ez, Ezz = self._expectation(self.X_dr['X']-self.mean)
            self._maximization(self.X_dr, Ez, Ezz)
            #self.alpha = len(self.mean) / np.sum(self.W ** 2, axis=0).clip(min=1e-10)
            if np.allclose(W, self.W):
                break
        self.n_iter = i + 1
        self.C = self.W @ self.W.T + self.var * np.eye(np.size(self.X_dr['X'], 1))
        self.Cinv = np.linalg.inv(self.C)

    def _maximization(self, X_dr, Ez, Ezz):
        X_mean = (X_dr['X']-self.mean)
        self.W = (X_mean*X_dr['W'][:,None]).T @ Ez @ np.linalg.inv(np.sum(Ezz*X_dr['W'][:,None,None], axis=0) + self.var * np.diag(self.alpha))
        self.var = np.sum(
            (np.mean((X_dr['XX'] - 2*X_dr['X']*self.mean + self.mean ** 2), axis=-1)
            #(np.mean((X_mean** 2), axis=-1)
            - 2 * np.mean(Ez @ self.W.T * X_mean, axis=-1)
            + np.trace((Ezz @ self.W.T @ self.W).T)/ len(self.mean))*X_dr['W'])/sum(X_dr['W'])
        self.var=max(self.var,0.000001)

    def maximize(self, D, Ez, Ezz):
        self.W = D.T.dot(Ez).dot(np.linalg.inv(np.sum(Ezz, axis=0) + self.var * np.diag(self.alpha)))
        self.var = np.mean(
            np.mean(D ** 2, axis=-1)
            - 2 * np.mean(Ez.dot(self.W.T) * D, axis=-1)
            + np.trace(Ezz.dot(self.W.T).dot(self.W).T) / self.ndim)
