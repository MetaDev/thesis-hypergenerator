# -*- coding: utf-8 -*-
import numpy.linalg as linalg
import numpy as np
from copy import deepcopy
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
    print(new_idx)
    print(ccov[0])
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
        print(new_ccov)
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
def sample_gaussian_mixture(centroids, ccov, mc = None, samples = 1):
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
    EPS = 1E-15
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