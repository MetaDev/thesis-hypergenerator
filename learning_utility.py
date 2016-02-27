# -*- coding: utf-8 -*-
import numpy.linalg as linalg
import numpy as np
from copy import deepcopy
from sklearn import mixture

def cond_dist(Y, centroids, ccov, mc):
    """Finds the conditional distribution p(X|Y) for a GMM.

    Parameters
    ----------
    Y : D array
        An array of inputs. Inputs set to NaN are not set, and become inputs to
        the resulting distribution. Order is preserved.
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one) by default equal
        for each cluster.

    Returns
    -------
    res : tuple
        A tuple containing a new set of (centroids, ccov, mc) for the
        conditional distribution.
    """
    not_set_idx = np.nonzero(np.isnan(Y))[0]
    set_idx = np.nonzero(True - np.isnan(Y))[0]
    new_idx = np.concatenate((not_set_idx, set_idx))
    y = Y[set_idx]
    # New centroids and covar matrices
    new_cen = []
    new_ccovs = []
    # Appendix A in C. E. Rasmussen & C. K. I. Williams, Gaussian Processes
    # for Machine Learning, the MIT Press, 2006
    fk = []
    for i in range(len(centroids)):
        # Make a new co-variance matrix with correct ordering
        new_ccov = deepcopy(ccov[i])
        new_ccov = new_ccov[:,new_idx]
        new_ccov = new_ccov[new_idx,:]
        #ux = centroids[i][not_set_idx]
        #uy = centroids[i][set_idx]
        #A = new_ccov[0:len(not_set_idx), 0:len(not_set_idx)]
        #B = new_ccov[len(not_set_idx):, len(not_set_idx):]
        #C = new_ccov[0:len(not_set_idx), len(not_set_idx):]
        #cen = ux + np.dot(np.dot(C, np.linalg.inv(B)), (y - uy))
        #cov = A - np.dot(np.dot(C, np.linalg.inv(B)), C.transpose())
        ua = centroids[i][not_set_idx]
        ub = centroids[i][set_idx]
        Saa = new_ccov[0:len(not_set_idx), 0:len(not_set_idx)]
        Sbb = new_ccov[len(not_set_idx):, len(not_set_idx):]
        Sab = new_ccov[0:len(not_set_idx), len(not_set_idx):]
        L = np.linalg.inv(new_ccov)
        Laa = L[0:len(not_set_idx), 0:len(not_set_idx)]
        Lbb = L[len(not_set_idx):, len(not_set_idx):]
        Lab = L[0:len(not_set_idx), len(not_set_idx):]
        cen = ua - np.dot(np.dot(np.linalg.inv(Laa), Lab), (y-ub))
        cov = np.linalg.inv(Laa)
        new_cen.append(cen)
        new_ccovs.append(cov)
        #fk.append(mulnormpdf(Y[set_idx], uy, B)) # Used for normalizing the mc
        fk.append(mulnormpdf(Y[set_idx], ub, Sbb)) # Used for normalizing the mc
    # Normalize the mixing coef: p(X|Y) = p(Y,X) / p(Y) using the marginal dist.
    fk = np.array(fk).flatten()
    new_mc = (mc*fk)
    new_mc = new_mc / np.sum(new_mc)
    return (new_cen, new_ccovs, new_mc)

def mulnormpdf(X, MU, SIGMA):
    """Evaluates the PDF for the multivariate Guassian distribution.

    Parameters
    ----------
    X : np array
        Inputs/entries row-wise. Can also be a 1-d array if only a 
        single point is evaluated.
    MU : nparray
        Center/mean, 1d array. 
    SIGMA : 2d np array
        Covariance matrix.

    Returns
    -------
    prob : 1d np array
        Probabilities for entries in `X`.
    
    Examples
    --------
    ::

        from pypr.clustering import *
        from numpy import *
        X = array([[0,0],[1,1]])
        MU = array([0,0])
        SIGMA = diag((1,1))
        gmm.mulnormpdf(X, MU, SIGMA)

    """
    # Check if inputs are ok:
    if MU.ndim != 1:
        raise ValueError( "MU must be a 1 dimensional array")
    
    # Evaluate pdf at points or point:
    mu = MU
    x = X.T
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    sigma = np.atleast_2d(SIGMA) # So we also can use it for 1-d distributions

    N = len(MU)
    ex1 = np.dot(linalg.inv(sigma), (x.T-mu).T)
    ex = -0.5 * (x.T-mu).T * ex1
    if ex.ndim == 2: ex = np.sum(ex, axis = 0)
    K = 1 / np.sqrt ( np.power(2*np.pi, N) * linalg.det(sigma) )
    return K*np.exp(ex)
def sample_gaussian_mixture(centroids, ccov, mc = None, samples = 1,loose_norm_weights=False):
    """
    Draw samples from a Mixture of Gaussians (MoG)

    Parameters
    ----------
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one)
                  by default equal for each cluster.

    Returns
    -------
    X : 2d np array
         A matrix with samples rows, and input dimension columns.

    Examples
    --------
    ::

        from pypr.clustering import *
        from numpy import *
        centroids=[array([10,10])]
        ccov=[array([[1,0],[0,1]])]
        samples = 10
        gmm.sample_gaussian_mixture(centroids, ccov, samples=samples)

    """
    cc = centroids
    D = len(cc[0]) # Determin dimensionality
    
    # Check if inputs are ok:
    K = len(cc)
    if mc is None: # Default equally likely clusters
        mc = np.ones(K) / K
    if len(ccov) != K:
        raise ValueError("centroids and ccov must contain the same number" +\
            "of elements.")
    if len(mc) != K:
        raise ValueError ("centroids and mc must contain the same number" +\
            "of elements.")

    # Check if the mixing coefficients sum to one:
    if loose_norm_weights:
        EPS=1e-12
    else:
        EPS = mixture.gmm.EPS
    n=np.abs(1-np.sum(mc)) 
    if np.abs(1-np.sum(mc)) > EPS:
        raise ValueError ("The sum of mc must be 1.0")

    # Cluster selection
    cs_mc = np.cumsum(mc)
    cs_mc = np.concatenate(([0], cs_mc))
    sel_idx = np.random.rand(samples)

    # Draw samples
    res = np.zeros((samples, D))
    for k in range(K):
        idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k+1])
        ksamples = np.sum(idx)
        drawn_samples = np.random.multivariate_normal(\
            cc[k], ccov[k], ksamples)
        res[idx,:] = drawn_samples
    return res

from sklearn import cluster
from time import time
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import  check_array
from sklearn.utils.extmath import logsumexp
class GMM_weighted(mixture.gmm.GMM):

    

    def weighted_fit(self, X, Xweights, y=None, do_prediction=False):
            #sample weight need to be normalised
            """Estimate model parameters with the EM algorithm.
            A initialization step is performed before entering the
            expectation-maximization (EM) algorithm. If you want to avoid
            this step, set the keyword argument init_params to the empty
            string '' when creating the GMM object. Likewise, if you would
            like just to do an initialization, set n_iter=0.
            Parameters
            ----------
            X : array_like, shape (n, n_features)
                List of n_features-dimensional data points. Each row
                corresponds to a single data point.
            Returns
            -------
            responsibilities : array, shape (n_samples, n_components)
                Posterior probabilities of each mixture component for each
                observation.
            """
    
            # initialization step
            X = check_array(X, dtype=np.float64, ensure_min_samples=2,
                            estimator=self)
            if X.shape[0] < self.n_components:
                raise ValueError(
                    'GMM estimation with %s components, but got only %s samples' %
                    (self.n_components, X.shape[0]))
    
            max_log_prob = -np.infty
    
            if self.verbose > 0:
                print('Expectation-maximization algorithm started.')
    
            for init in range(self.n_init):
                if self.verbose > 0:
                    print('Initialization ' + str(init + 1))
                    start_init_time = time()
    
                if 'm' in self.init_params or not hasattr(self, 'means_'):
                    self.means_ = cluster.KMeans(
                        n_clusters=self.n_components,
                        random_state=self.random_state).fit(X).cluster_centers_
                    if self.verbose > 1:
                        print('\tMeans have been initialized.')
    
                if 'w' in self.init_params or not hasattr(self, 'weights_'):
                    self.weights_ = np.tile(1.0 / self.n_components,
                                            self.n_components)
                    if self.verbose > 1:
                        print('\tWeights have been initialized.')
    
                if 'c' in self.init_params or not hasattr(self, 'covars_'):
                    cv = np.cov(X.T,aweights=Xweights) + self.min_covar * np.eye(X.shape[1])
                    if not cv.shape:
                        cv.shape = (1, 1)
                    self.covars_ = \
                        mixture.gmm.distribute_covar_matrix_to_match_covariance_type(
                            cv, self.covariance_type, self.n_components)
                    if self.verbose > 1:
                        print('\tCovariance matrices have been initialized.')
    
                # EM algorithms
                current_log_likelihood = None
                # reset self.converged_ to False
                self.converged_ = False
    
                for i in range(self.n_iter):
                    if self.verbose > 0:
                        print('\tEM iteration ' + str(i + 1))
                        start_iter_time = time()
                    prev_log_likelihood = current_log_likelihood
                    # Expectation step
                    log_likelihoods, responsibilities = self.score_weighted_samples(X,Xweights)
                    
                    
                    
                    current_log_likelihood = log_likelihoods.mean()
    
                    # Check for convergence.
                    if prev_log_likelihood is not None:
                        change = abs(current_log_likelihood - prev_log_likelihood)
                        if self.verbose > 1:
                            print('\t\tChange: ' + str(change))
                        if change < self.tol:
                            self.converged_ = True
                            if self.verbose > 0:
                                print('\t\tEM algorithm converged.')
                            break
    
                    # Maximization step
                    self._do_weighted_mstep(X,Xweights,responsibilities, self.params,
                                   self.min_covar)
                    if self.verbose > 1:
                        print('\t\tEM iteration ' + str(i + 1) + ' took {0:.5f}s'.format(
                            time() - start_iter_time))
    
                # if the results are better, keep it
                if self.n_iter:
                    if current_log_likelihood > max_log_prob:
                        max_log_prob = current_log_likelihood
                        best_params = {'weights': self.weights_,
                                       'means': self.means_,
                                       'covars': self.covars_}
                        if self.verbose > 1:
                            print('\tBetter parameters were found.')
    
                if self.verbose > 1:
                    print('\tInitialization ' + str(init + 1) + ' took {0:.5f}s'.format(
                        time() - start_init_time))
    
            # check the existence of an init param that was not subject to
            # likelihood computation issue.
            if np.isneginf(max_log_prob) and self.n_iter:
                raise RuntimeError(
                    "EM algorithm was never able to compute a valid likelihood " +
                    "given initial parameters. Try different init parameters " +
                    "(or increasing n_init) or check for degenerate data.")
    
            if self.n_iter:
                self.covars_ = best_params['covars']
                self.means_ = best_params['means']
                self.weights_ = best_params['weights']
            else:  # self.n_iter == 0 occurs when using GMM within HMM
                # Need to make sure that there are responsibilities to output
                # Output zeros because it was just a quick initialization
                responsibilities = np.zeros((X.shape[0], self.n_components))
    
            return responsibilities
    def score_weighted_samples(self, X,Xweights):
        """Return the per-sample likelihood of the data under the model.
        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.
        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X.
        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        check_is_fitted(self, 'means_')

        X = check_array(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        lpr = (mixture.gmm.log_multivariate_normal_density(X, self.means_, self.covars_,
                                               self.covariance_type) +
               np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        logprob=np.array([l*w for l,w in zip(logprob,Xweights)])
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities
    def _do_weighted_mstep(self, X,Xweights, responsibilities, params, min_covar=0):
        """Perform the Mstep of the EM algorithm and return the cluster weights.
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * mixture.gmm.EPS)

        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * mixture.gmm.EPS) + mixture.gmm.EPS)
        if 'm' in params:
            self.means_ = weighted_X_sum * inverse_weights
        if 'c' in params:
            covar_mstep_func = mixture.gmm._covar_mstep_funcs[self.covariance_type]
            self.covars_ = covar_mstep_func(
                self, X, responsibilities, weighted_X_sum, inverse_weights,
                min_covar)
        return weights