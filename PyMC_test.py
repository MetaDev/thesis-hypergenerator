import pymc
import pymc3
import numpy as np
def make_model(x):
    a = pymc.Exponential('a', beta=x, value=0.5)

    @pymc.deterministic
    def b(a=a):
        return 100-a
        
    @pymc.stochastic
    def c(value=0.5, a=a, b=b):
        def logp(value, a, b):
            if value > b or value < a:
                return -np.inf
            else:
                return -np.log(b - a + 1)
        
        def random(a, b):
            from numpy.random import random
            return (a - b) * random() + a

    return locals()

M = pymc.MCMC(make_model(3))
for i in range(100):
    M.draw_from_prior()
    print(M.a.value)
basic_model = pymc3.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pymc3.Uniform('alpha', lower=0, upper=10)
    beta = pymc3.Uniform('alpha',lower=alpha, upper=alpha+5)
    test = pymc3.DiscreteUniform('alpha', lower=1, upper=10)
for i in range(100):
    print(alpha.random())
    print(beta.random())
    print(test.random())
    
