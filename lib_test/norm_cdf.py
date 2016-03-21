

import numpy as np

#The other function allows for non-zero means and covariance (as opposed to correlation) matrices, but it doesn’t, technically speaking, allow for integration to or from +/-Infinity:


# apply this function for N points between the desired fitness interval [a,b]-> [a,infty] , [a+(b-a)/N,infty],...
#value,inform = mvnun(lower,upper,means,covar,...])
from scipy.stats import mvn
low = np.array([-10, -10])
upp = np.array([.1, -.2])
mu = np.array([-.3, .17])
#covar
S = np.array([[1.2,.35],[.35,2.1]])
p,i = mvn.mvnun(low,upp,mu,S)
print (p,i)

