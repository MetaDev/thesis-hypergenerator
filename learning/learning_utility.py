# -*- coding: utf-8 -*-
import numpy.linalg as linalg
import numpy as np
from copy import deepcopy
from sklearn import mixture



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
            #normalise the weights between [0,1]
            Xweights=(np.array(Xweights)-np.min(Xweights))/(np.max(Xweights)-np.min(Xweights))
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
                    self._do_mstep(X,responsibilities, self.params,
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
