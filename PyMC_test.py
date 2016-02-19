import pymc
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
    
