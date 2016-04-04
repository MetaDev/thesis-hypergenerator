from pomegranate import GeneralMixtureModel,MultivariateGaussianDistribution
import gmr
from enum import Enum
import learning.gmm_sk_ext as sk
import numpy as np

#use pomegranate/sklearn for training
#use sklearn for initilisation
#use gmr for marginal and conditioning


class GMM:


    def __init__(self, n_components=None, random_state=None):

        self.n_components = n_components

        self.random_state = random_state





    def set_params_for_sampling_cond_marg(self,weights,means,covars):
        self.means_=means
        self.weights_=weights
        self.n_components=len(weights)
        self.covars_=covars
        self._set_params_gmr()
        self.fitted=False



    def _init_sk(self,X,weights,tol,min_covar,n_init,max_iter):
        self.gmm_sk= sk.GMM(n_components=self.n_components,covariance_type='full',tol=tol, min_covar=min_covar,
                 n_iter=max_iter)
    def _init_sk_bayes(self,X,weights,tol,prune_thresh,n_init,max_iter,n_mfa_iter):
        self.gmm_sk= sk.VBGMMARD(n_components=self.n_components,tol=tol,n_mfa_iter = n_mfa_iter,
                                  n_iter=max_iter)



    def get_params(self):
        return self.weights_,self.means_,self.covars_

    def fit(self, X, weights=None,infinite=False, tol=1e-3, min_covar=1e-3,prune_thresh=1e-2,n_init=5,
                 max_iter=100,n_mfa_iter=5):
        self.fitted=True
        if infinite:
            self._init_sk_bayes(X,weights,tol,prune_thresh,n_init,max_iter,n_mfa_iter)
        else:
            self._init_sk(X,weights,tol,prune_thresh,n_init,max_iter)
        self.gmm_sk.weighted_fit(X,weights)
        self.set_params_for_sampling_cond_marg(*GMM._get_params_sk(self.gmm_sk))


    @staticmethod
    def _get_params_sk(gmm_sk):
        return gmm_sk.weights_, gmm_sk.means_,gmm_sk.covars_
    @staticmethod
    def _get_params_gmr(gmm_gmr):
        return gmm_gmr.priors,gmm_gmr.means, gmm_gmr.covariances


    def _set_params_gmr(self):
        self.gmm_gmr=gmr.GMM(n_components=self.n_components, priors=self.weights_, means=self.means_,
                                 covariances=self.covars_, random_state=self.random_state)


    def _is_fitted(self):
        return self.covars_ is not None
    #TODO
    def variance(self):
        pass
    def condition(self, indices, x):
        if self._is_fitted():
            gmm = self.gmm_gmr.condition(indices,x)
            weights,means,covars=GMM._get_params_gmr(gmm)
            gmm=GMM(random_state=self.random_state)
            gmm.set_params_for_sampling_cond_marg(weights=weights,means=means,covars=covars)
            return gmm

    def sample(self,n):
        if self._is_fitted():
            return self.gmm_gmr.sample(n)
    #TODO check if this is mathematically correct
    def marginalise(self,indices):
        weights,means,covars = self.get_params()
        if self._is_fitted():
            mvns=[]
            for k in range(self.n_components):
                mvn = gmr.MVN(mean=means[k],covariance=covars[k],
                              random_state=self.random_state)
                mvn.
                mvn= mvn.marginalize(indices)
                mvns.append(mvn)
            means_marg=np.array([mvn.mean for mvn in mvns])
            covars_marg=np.array([mvn.covariance for mvn in mvns])
            #return a GMM of this class
            gmm=GMM(random_state=self.random_state)
            gmm.set_params_for_sampling_cond_marg(weights=weights,means=means_marg,covars=covars_marg)
            return gmm

#TODO add visualisation
