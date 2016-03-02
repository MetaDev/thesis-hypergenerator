# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:53:27 2015

@author: Harald
"""
#polygon imports

import search_space
import statsmodels.api as sm
import visualisation
import utility
import fitness
from matplotlib import pyplot as plt

#learn imports
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import itertools


ndata=100
data=[]
fitness_values=[]
fitness_test = lambda fitness , value : fitness > value
for i in range(ndata):
    result=[]
    #train per child
    result,_=search_space.test()
    sample_data = utility.extract_samples_attributes(result,attr_name="independent_attr",sample_name="chair")
    [data.append(ind_attr["position"]) for ind_attr in sample_data]
    
    fitness_v=fitness.dist_between_parent_child(utility.extract_samples_attributes(result,sample_name="table")[0],utility.extract_samples_attributes(result,sample_name="chair"))
    fitness_values.extend(([fitness_v]*len(sample_data)))

accept = lambda f : f > 10
subset =np.asarray( [d for d,f in zip(data,fitness_values) if accept(f)])
dens_u = sm.nonparametric.KDEMultivariate(data=subset, var_type='cc', bw='normal_reference')
#sample from data
rng = np.random.RandomState(42)
n_samples=100
i = rng.randint(subset.shape[0], size=n_samples)
samples= np.atleast_2d(rng.normal(subset[i], dens_u.bw))
print(samples)
#learn stuff
