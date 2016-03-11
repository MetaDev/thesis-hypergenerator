from pomegranate import *
from pomegranate import MultivariateGaussianDistribution



mu = np.arange(5)
cov = np.eye(5)

mgs = [ MultivariateGaussianDistribution( mu*i, cov ) for i in range(5) ]
gmm = GeneralMixtureModel( mgs )
print(gmm.sample())