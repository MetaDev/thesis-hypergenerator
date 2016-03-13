from pomegranate import GeneralMixtureModel,MultivariateGaussianDistribution
import gmr
from enum import Enum
import learning.gmm_sk_ext as sk
import numpy as np

#use pomegranate/sklearn for training
#use sklearn for initilisation
#use gmr for marginal and conditioning


class GMM:
    class Train_Lib(Enum):
        sklearn=1
        #pomegranate not working for the moment
        #pomegranate=2

    def __init__(self, n_components=None,weights=None,means=None,covariances=None, random_state=None):

        self.n_components = n_components

        self.random_state = random_state

        if not n_components:
            if weights is not None:
                self.n_components=len(weights)
            else:
                raise ValueError("The GMM should be initialised with either n_components or it's parameters.")
        self._set_params(weights,means,covariances)
        self._set_gmms()



    def _set_params(self,weights,means,covariances):
        self._means=means
        self._weights=weights
        self._covariances=covariances


    def _init_sk(self,X,weights,tol,min_covar,max_iter):
        self.gmm_sk= sk.GMM(n_components=self.n_components,covariance_type='full',tol=tol, min_covar=min_covar,
                 n_iter=max_iter)
        self.gmm_sk.n_iter=0
        #initialise means,covars and weights
        self.gmm_sk.weighted_fit(X,weights)
        self.gmm_sk.n_iter=max_iter



    def _set_gmms(self):
        if self._is_fitted():
#            self._set_params_pmgr()
            self._set_params_sk()
            self._set_params_gmr()
    def get_params(self):
        return self._weights,self._means,self._covariances

    def fit( self, X, weights=None, reset=True, tol=1e-3, min_covar=1e-3,
                 max_iter=100,train_lib=Train_Lib.sklearn):

        if reset or not self._check_fitted():
            self._init_sk(X,weights,tol,min_covar,max_iter)
            self._set_params(*GMM._get_params_sk(self.gmm_sk))

        if train_lib==GMM.Train_Lib.sklearn:
            self.gmm_sk.weighted_fit(X,weights)
            self._set_params(*GMM._get_params_sk(self.gmm_sk))
#            self._set_params_pmgr()
#        else:
#            if reset or not self._fit:
#                self._set_params_pmgr()
#            weigths=np.array(weights)/np.max(weights)
#            self.gmm_pmgr.train( np.array(X), weigths, stop_threshold=tol, max_iterations=max_iter)
#            self._set_params(*GMM._get_params_pmgr(self.gmm_pmgr))
#            self._set_params_sk()

        self._set_params_gmr()
    @staticmethod
    def _get_params_sk(gmm_sk):
        return gmm_sk.weights_, gmm_sk.means_,gmm_sk._get_covars()
    @staticmethod
    def _get_params_gmr(gmm_gmr):
        return gmm_gmr.priors,gmm_gmr.means, gmm_gmr.covariances
    @staticmethod
    def _get_params_pmgr(gmm_pmgr):
        means=[distr.mu for distr in gmm_pmgr.distributions]
        covars=[distr.cov for distr in gmm_pmgr.distributions]
        return np.exp(gmm_pmgr.weights), np.array(means), np.array(covars)

    def _set_params_sk(self):
        self.gmm_sk= sk.GMM(n_components=len(self._weights),covariance_type='full')
        self.gmm_sk._set_covars( self._covariances)
        self.gmm_sk.weights_=self._weights
        self.gmm_sk.means_=self._means

    def _set_params_pmgr(self):
        mgs = [ MultivariateGaussianDistribution(mean,covariance ) for mean,covariance in zip(self._means,self._covariances) ]
        self.gmm_pmgr=GeneralMixtureModel(mgs,weights=self._weights)

    def _set_params_gmr(self):
        self.gmm_gmr=gmr.GMM(n_components=self.n_components, priors=self._weights, means=self._means,
                                 covariances=self._covariances, random_state=self.random_state)


    def _is_fitted(self):
        return self._means is not None and self._covariances is not None and self._weights is not None

    def condition(self, indices, x):
        if self._is_fitted():
            gmm = self.gmm_gmr.condition(indices,x)
            weights,means,covars=GMM._get_params_gmr(gmm)
            return GMM(weights=weights,means=means,covariances=covars,random_state= self.random_state)

    def sample(self,n):
        if self._is_fitted():
            return self.gmm_gmr.sample(n)

    def marginalise(self,indices):
        if self._is_fitted():
            mvns=[]
            for k in range(self.n_components):
                mvn = gmr.MVN(mean=self.gmm_gmr.means[k], covariance=self.gmm_gmr.covariances[k],random_state=self.gmm_gmr.random_state)
                mvn= mvn.marginalize(indices)
                mvns.append(mvn)
            means=np.array([mvn.mean for mvn in mvns])
            covariances=np.array([mvn.covariance for mvn in mvns])
            #return a GMM of this class
            return GMM(weights=self._weights,means=means,covariances=covariances,
                       random_state= self.random_state)

#TODO add visualisation
