# -*- coding: utf-8 -*-
#find out how to sample from P(Y>t)
#if P(Y) and t are known

import numpy as np
import scipy.interpolate.rbf as rbf

#use N dimensional interpolation for the inv_cdf
#SciPy has an Rbf interpolation method (radial basis function) which allows better than linear interpolation at arbitrary dimensions.
#
#Taking a variable data with rows of (x1,x2,x3...,xn,v) values, the follow code modification to the original post allows for interpolation: compare rbf and http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.interpolate.griddata.html
#
#
#mesh = np.meshgrid(*[i['grid2'] for i in self.cambParams], indexing='ij')
#chi2 = rbfi(*mesh)


def inverse_transform_sampling(data, n_bins, n_samples):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    #use N multidimensional interpolation
    inv_cdf = rbf(*data.T,d=bin_edges)
    #generate N random values in N dim [0,1]
    r = np.random.rand(n_samples,len(data[0]))
    return inv_cdf(r)


#The other function allows for non-zero means and covariance (as opposed to correlation) matrices, but it doesn’t, technically speaking, allow for integration to or from +/-Infinity:
#
#value,inform = mvnun(lower,upper,means,covar,...])
from scipy.stats import mvn
low = np.array([-10, -10])
upp = np.array([.1, -.2])
mu = np.array([-.3, .17])
#covar
S = np.array([[1.2,.35],[.35,2.1]])
p,i = mvn.mvnun(low,upp,mu,S)
print (p,i)

