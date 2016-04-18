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
        else:
            raise ValueError("GMM can not be sampled before fitting.")
    #TODO check if this is mathematically correct
    def marginalise(self,indices):
        weights,means,covars = self.get_params()
        if self._is_fitted():
            mvns=[]
            for k in range(self.n_components):
                mvn = gmr.MVN(mean=means[k],covariance=covars[k],
                              random_state=self.random_state)
                mvn= mvn.marginalize(indices)
                mvns.append(mvn)
            means_marg=np.array([mvn.mean for mvn in mvns])
            covars_marg=np.array([mvn.covariance for mvn in mvns])
            #return a GMM of this class
            gmm=GMM(random_state=self.random_state)
            gmm.set_params_for_sampling_cond_marg(weights=weights,means=means_marg,covars=covars_marg)
            return gmm
#visualisation

import scipy.stats as ss
import util.utility as ut

def visualise_gmm_marg_density_1D_GMM(ax,marg_index,gmm,factor=3,verbose=False):
    visualise_gmm_marg_1D_density(ax,marg_index,*gmm.get_params(),
                                 factor,verbose)
#TODO check dimensionality
def visualise_gmm_marg_1D_density(ax,marg_index,gmm_weights,gmm_means,gmm_cov,
                                 factor=2,xrange=None,verbose=False):
    means=[]
    weights=gmm_weights
    stdevs=[]
    for k in range(len(gmm_means)):
        means.append(gmm_means[k][marg_index])
        #calc std
        cov = np.asanyarray(gmm_cov[k])
        stdevs.append(np.sqrt(np.diag(cov))[marg_index])

    min_x=np.min([mean-factor*stdev for mean,stdev in zip(means,stdevs)])
    max_x=np.max([mean+factor*stdev for mean,stdev in zip(means,stdevs)])
    x = np.arange(min_x, max_x, 0.01)
    pdfs = [w * ss.norm.pdf(x, mu, sd) for mu, sd, w in zip(means, stdevs, weights)]
    density = np.sum(np.array(pdfs), axis=0)
    if verbose:
        print("Expected mean value: "+ ut.format_float(np.mean([ w*m for m,w in zip(means, weights)])))
        print("Expected mean standard dev: " + ut.format_float(np.mean([ w*sd for sd,w in zip(stdevs, weights)])))
    #unfirom distribution line
    y=[1/(max_x-min_x)]*len(x)
    ax.plot(x, density)
    ax.plot(x,y,color="r")

#TODO check dimensionality
def visualise_gmm_marg_2D_density(ax,gmm,
                                  min_factor=.5,max_factor=3,steps=5, colors=["r","g","b"]):
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    min_alpha=0.03
    max_alpha=0.4

    for factor in np.linspace(min_factor, max_factor, steps):
        for (mean, (angle, width, height)),weight in zip(gmm.gmm_gmr.to_ellipses(factor),gmm.weights_):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(min_alpha+(max_alpha-min_alpha)*weight)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)