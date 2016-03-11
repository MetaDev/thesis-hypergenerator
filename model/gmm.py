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
        pomegranate=2

    def __init__(self, n_components,weights,means,covariances, random_state=None):
        self.n_components = n_components

        #training params
        self.tol = tol
        self.min_covar = min_covar
        self.max_iter = max_iter
        self.train_lib = train_lib

        self.random_state = random_state

        self._set_params(weights,means,covariances)
        self._set_gmms()

    def _set_params(self,weights,means,covariances):
        self._means=means
        self._weights=weights
        self._covariances=covariances

    def _init_sk(self,X,weights,tol,min_covar,max_iter):
        self.gmm_sk= sk.GMM(covariance_type='full',tol=tol, min_covar=min_covar,
                 n_iter=max_iter)
        self.gmm_sk.n_iter=0
        self.gmm_sk.weighted_fit(X,weights)
        self.gmm_sk.n_iter=self.max_iter

    def _sk_train_params(self):
        self.gmm_sk.tol = tol
        self.gmm_sk.min_covar = min_covar
        self.gmm_sk.n_iter=max_iter

    def _set_gmms(self):
        self._set_params_pmgr()
        self._set_params_sk()
        self._set_params_gmr()


    def fit( self, X, weights=None, reset=True, tol=1e-3, min_covar=1e-3,
                 max_iter=100,train_lib=Train_Lib.sklearn):

        if reset or not self._check_fitted():
            self._init_sk(X,weights,tol,min_covar,max_iter)
            self._set_params_pmgr()

        if self.train_lib==self.Train_Lib.sklearn:
            self.gmm_sk.weighted_fit(X,weights)
        else:
            if reset or not self._fit:
                self._init_params_pmgr()
            self.pmgr_gmm.train( X, weights, self.tol, self.max_iter)
        self._update_gmr()

    def _get_params_sk(self):

    def _set_params_sk(self):
        self.gmm_sk._set_covars( self._covariances)
        self.gmm_sk.weights_=self._weights
        self.gmm_sk.means_=self._means

    def _set_params_pmgr(self):
        mgs = [ MultivariateGaussianDistribution(mean,covariance ) for mean,covariance in zip(self._means,self._covariances) ]
        self.gmm_pmgr=GeneralMixtureModel(mgs,weights=self._weights)

    def _set_params_gmr(self):
        if self._check_fitted():

            if self.train_lib==self.Train_Lib.sklearn:
                means=self.gmm_sk.means_
                weights=self.gmm_sk.weights_
                covariances=self.gmm_sk._get_covars()

            else:
                means=[distr.mu for distr in self.gmm_pmgr.distributions ]
                covariances=[distr.cov for distr in self.gmm_pmgr.distributions ]
                weights=self.gmm_pmgr.weights

            self.gmm_gmr=gmr.GMM(n_components=self.n_components, priors=weights, means=means,
                                 covariances=covariances, random_state=self.random_state)



    def _is_fitted(self):
        return self._means !=None and self._covariances !=None and self._weights !=None
    def condition(self, indices, x):
        if self._fit:
            return self.gmm_gmr.condition(indices,x)

    def sample(self,n):
        if self._fit:
            return self.gmm_gmr.sample(n)

    def marginalise(self,indices):
        if self._fit:
            mvns=[]
            for k in range(self.n_components):
                mvn = gmr.MVN(mean=self.gmm_gmr.means[k], covariance=self.gmm_gmr.covariances[k],random_state=self.gmm_gmr.random_state)
                mvn= mvn.marginalize(indices)
                mvns.append(mvn)
            means=np.array([self.gmm_gmr.mean for mvn in mvns])
            covariances=np.array([self.gmm_gmr.covariance for mvn in mvns])
            #return a GMM of this class
            return GMM(gmm.n_components,self.gmm_gmr.priors,means,covariances,gmm.random_state)

