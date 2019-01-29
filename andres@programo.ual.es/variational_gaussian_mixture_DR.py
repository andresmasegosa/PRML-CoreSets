import numpy as np
from scipy.misc import logsumexp
from scipy.special import digamma, gamma, gammaln
from prml.rv.rv import RandomVariable
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans, KMeans


class VariationalGaussianMixture_DR(RandomVariable):

    def __init__(self, n_components=1, alpha0=None, m0=None, W0=1., dof0=None, beta0=1.):
        """
        construct variational gaussian mixture model
        Parameters
        ----------
        n_components : int
            maximum numnber of gaussian components
        alpha0 : float
            parameter of prior dirichlet distribution
        m0 : float
            mean parameter of prior gaussian distribution
        W0 : float
            mean of the prior Wishart distribution
        dof0 : float
            number of degrees of freedom of the prior Wishart distribution
        beta0 : float
            prior on the precision distribution
        """
        super().__init__()
        self.n_components = n_components
        if alpha0 is None:
            self.alpha0 = 1 / n_components
        else:
            self.alpha0 = alpha0
        self.m0 = m0
        self.W0 = W0
        self.dof0 = dof0
        self.beta0 = beta0

    def _init_params(self, X):
        sample_size, self.ndim = X.shape
        self.alpha0 = np.ones(self.n_components) * self.alpha0
        if self.m0 is None:
            self.m0 = np.mean(X, axis=0)
        else:
            self.m0 = np.zeros(self.ndim) + self.m0
        self.W0 = np.eye(self.ndim) * self.W0
        if self.dof0 is None:
            self.dof0 = self.ndim

        self.component_size = sample_size / self.n_components + np.zeros(self.n_components)
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        indices = np.random.choice(sample_size, self.n_components, replace=False)
        self.mu = X[indices]
        self.W = np.tile(self.W0, (self.n_components, 1, 1))
        self.dof = self.dof0 + self.component_size

    @property
    def alpha(self):
        return self.parameter["alpha"]

    @alpha.setter
    def alpha(self, alpha):
        self.parameter["alpha"] = alpha

    @property
    def beta(self):
        return self.parameter["beta"]

    @beta.setter
    def beta(self, beta):
        self.parameter["beta"] = beta

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        self.parameter["mu"] = mu

    @property
    def W(self):
        return self.parameter["W"]

    @W.setter
    def W(self, W):
        self.parameter["W"] = W

    @property
    def dof(self):
        return self.parameter["dof"]

    @dof.setter
    def dof(self, dof):
        self.parameter["dof"] = dof

    def get_params(self):
        return self.alpha, self.beta, self.mu, self.W, self.dof

    def _random(self, X, n_clusters):

        centers_X = X[np.random.choice(X.shape[0], n_clusters, replace=False),:]
        centers_XX = np.multiply(centers_X[:, :, None], centers_X[:, None, :])
        weights = np.repeat(X.shape[0]/n_clusters,n_clusters)

        self.X_dr = {'X': centers_X, 'XX': centers_XX,
                'W': weights}

    def _clusteringError(self, X, kmeans):
        sum = 0
        for i in range(0, kmeans.cluster_centers_.shape[0]):
            a = X[kmeans.labels_ == i, :] - kmeans.cluster_centers_[i, :]
            sum += np.sqrt((a * a).sum(axis=1)).sum(axis=0)
        return sum

    def _clusterSS(self, X, n_clusters):
        XX = np.multiply(X[:, :, None], X[:, None, :])
        XX = XX.reshape((XX.shape[0], -1))
        XJoin = np.concatenate((X, XX), axis=1)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(XJoin)
        weights = np.asarray([sum(kmeans.labels_ == x) for x in range(0, n_clusters)])
        D=X.shape[1]
        self.X_dr = {'X': kmeans.cluster_centers_[:, 0:D], 'XX': kmeans.cluster_centers_[:, D:D + D * D].reshape((n_clusters, D, D)),
                'W': weights}
        self.clusterError = self._clusteringError(XJoin,kmeans)


    def _cluster(self, X, n_clusters):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(X)
        weights = np.asarray([sum(kmeans.labels_ == x) for x in range(0, n_clusters)])
        self.X_dr = {'X': kmeans.cluster_centers_,
                'XX': np.matmul(kmeans.cluster_centers_[:, :, None], kmeans.cluster_centers_[:, None, :]), 'W': weights}

    def fit(self, X, iter_max=100, n_clusters=10, cluster_method="SS"):
        if cluster_method== "SS":
            self._clusterSS(X,n_clusters)
        elif cluster_method== "NoSS":
            self._cluster(X,n_clusters)
        elif cluster_method == "random":
            self._random(X,n_clusters)

        self._init_params(self.X_dr['X'])
        for _ in range(iter_max):
            params = np.hstack([p.flatten() for p in self.get_params()])
            r = self._variational_expectation_DR(self.X_dr)
            self._variational_maximization_DR(self.X_dr, r)
            if np.allclose(params, np.hstack([p.flatten() for p in self.get_params()])):
                break

    def _variational_expectation_DR(self, X_dr):

        a = X_dr['XX'] #np.matmul(X_dr['X'][:, :, None], X_dr['X'][:, None, :])
        b = np.multiply(a[:,None,:,:],self.W)
        c = np.sum(np.sum(b,axis=-1),axis=-1)

        # maha_sq2 = -0.5 * (
        #     self.ndim / self.beta
        #     + self.dof * np.sum(
        #         np.einsum("kij,nkj->nki", self.W, X_dr['X'][:, None, :]) * X_dr['X'][:, None, :], axis=-1))


        maha_sq = -0.5 * (
            self.ndim / self.beta
            + self.dof * c)

        maha_sq = maha_sq + 2*0.5 * (
            self.ndim / self.beta
            + self.dof * np.sum(
                np.einsum("kij,nkj->nki", self.W, X_dr['X'][:, None, :]) * self.mu, axis=-1))

        maha_sq = maha_sq - 0.5 * (
            self.ndim / self.beta
            + self.dof * np.sum(
                np.einsum("kij,kj->ki", self.W, self.mu) * self.mu, axis=-1))


        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]
        ln_r = ln_pi + 0.5 * ln_Lambda + maha_sq
        ln_r -= logsumexp(ln_r, axis=-1)[:, None]
        r = np.exp(ln_r)
        return np.multiply(r,X_dr['W'][:,None])

    def _variational_expectation(self, X):
        d = X[:, None, :] - self.mu
        maha_sq = -0.5 * (
            self.ndim / self.beta
            + self.dof * np.sum(
                np.einsum("kij,nkj->nki", self.W, d) * d, axis=-1))
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]
        ln_r = ln_pi + 0.5 * ln_Lambda + maha_sq
        ln_r -= logsumexp(ln_r, axis=-1)[:, None]
        r = np.exp(ln_r)
        return r

    def _variational_maximization_DR(self, X_dr, r):
        self.component_size = r.sum(axis=0)
        Xm = (X_dr['X'].T.dot(r) / self.component_size).T
        # d = X_dr['X'][:, None, :] - Xm
        # S1 = np.einsum('nki,nkj->kij', d, r[:, :, None] * d) / self.component_size[:, None, None]

        S_partial = np.einsum('nij,nk->kij', X_dr['XX'], r)

        a = np.einsum('nk,ni->ki', r, X_dr['X'])
        S_partial =  S_partial - 2*np.matmul(a[:,:,None],Xm[:,None,:])

        b = np.matmul(Xm[:,:,None],Xm[:,None,:])
        S_partial =  S_partial + np.multiply(np.sum(r,axis=0)[:,None,None],b)

        S =  S_partial / self.component_size[:, None, None]

        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        self.mu = (self.beta0 * self.m0 + self.component_size[:, None] * Xm) / self.beta[:, None]
        d = Xm - self.m0
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.component_size * S.T).T
            + (
            self.beta0 * self.component_size * np.einsum('ki,kj->kij', d, d).T / (self.beta0 + self.component_size)).T)
        self.dof = self.dof0 + self.component_size

    def _variational_maximization(self, X, r):
        self.component_size = r.sum(axis=0)
        Xm = (X.T.dot(r) / self.component_size).T
        d = X[:, None, :] - Xm
        S = np.einsum('nki,nkj->kij', d, r[:, :, None] * d) / self.component_size[:, None, None]
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        self.mu = (self.beta0 * self.m0 + self.component_size[:, None] * Xm) / self.beta[:, None]
        d = Xm - self.m0
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.component_size * S.T).T
            + (self.beta0 * self.component_size * np.einsum('ki,kj->kij', d, d).T / (self.beta0 + self.component_size)).T)
        self.dof = self.dof0 + self.component_size

    def classify(self, X):
        """
        index of highest posterior of the latent variable
        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input
        Returns
        -------
        output : (sample_size, n_components) ndarray
            index of maximum posterior of the latent variable
        """
        return np.argmax(self._variational_expectation(X), 1)

    def classify_proba(self, X):
        """
        compute posterior of the latent variable
        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input
        Returns
        -------
        output : (sample_size, n_components) ndarray
            posterior of the latent variable
        """
        return self._variational_expectation(X)

    def student_t(self, X):
        nu = self.dof + 1 - self.ndim
        L = (nu * self.beta * self.W.T / (1 + self.beta)).T
        d = X[:, None, :] - self.mu
        maha_sq = np.sum(np.einsum('nki,kij->nkj', d, L) * d, axis=-1)
        return (
            gamma(0.5 * (nu + self.ndim))
            * np.sqrt(np.linalg.det(L))
            * (1 + maha_sq / nu) ** (-0.5 * (nu + self.ndim))
            / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * self.ndim)))

    def _pdf(self, X):
        return (self.alpha * self.student_t(X)).sum(axis=-1) / self.alpha.sum()

    def     logpdf(self, X):
        log_probs = self.logstudent_t(X)
        max_logs = np.max(log_probs, axis=1)
        log_probs = log_probs - np.repeat(np.expand_dims(max_logs, axis=1), log_probs.shape[1], axis=1)
        a = self.alpha * np.exp(log_probs) / self.alpha.sum()
        return np.log(np.sum(a, axis=1)) + max_logs

    def logstudent_t(self, X):
        nu = self.dof + 1 - self.ndim
        L = (nu * self.beta * self.W.T / (1 + self.beta)).T
        d = X[:, None, :] - self.mu
        maha_sq = np.sum(np.einsum('nki,kij->nkj', d, L) * d, axis=-1)
        (sign, logdet) = np.linalg.slogdet(L)
        return (
            gammaln(0.5 * (nu + self.ndim))
            +0.5*logdet
            +(-0.5 * (nu + self.ndim))*np.log(1 + maha_sq / nu)
            - (gammaln(0.5 * nu) + (0.5 * self.ndim)*np.log(nu * np.pi)))
