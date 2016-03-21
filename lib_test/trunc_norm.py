from numpy.linalg import inv
import numpy as np
import emcee

ndim, nwalkers,nstep = 10, 100,1000

def lnprob_trunc_norm(x, mean, bounds, C):
    if np.any(x < bounds[:,0]) or np.any(x > bounds[:,1]):
        return -np.inf
    else:
        return -0.5*(x-mean).dot(inv(C)).dot(x-mean)
low = np.array([-10, -10])
upp = np.array([.1, -.2])
bounds=np.hstack((low,upp))
mean = np.array([-.3, .17])
#covar
C = np.array([[1.2,.35],[.35,2.1]])
#for given mean, bounds and C. You need an initial guess for the walkers' positions pos, which could be a ball around the mean
pos = emcee.utils.sample_ball(mean, np.sqrt(np.diag(C)), size=nwalkers)
#sampled from an untruncated multivariate normal,

pos = np.random.multivariate_normal(mean, C, size=nwalkers)

S = emcee.EnsembleSampler(nwalkers, ndim, lnprob_trunc_norm, args = (mean, bounds, C))

pos, prob, state = S.run_mcmc(pos, nstep)

